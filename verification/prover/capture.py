"""Model-agnostic matmul capture: a passthrough ``TorchDispatchMode`` (§8.A.4).

Wrap the forward pass and ``loss.backward()`` of an unmodified model in a
:class:`MatmulCapture`. Every matmul-family aten op that reaches dispatch is recorded with its
operands as the op received them and its real output. The mode calls the real op and returns
the real result, so training numerics are unchanged.

Usage::

    cap = MatmulCapture(param_names=param_storage_map(model))   # computation/matmul_ops.py
    with cap:
        with cap.phase("forward"):
            loss = model(...).loss
        with cap.phase("backward"):
            loss.backward()
    cap.assert_unmodified()   # before optimizer.step() or zero_grad()

What is recorded, per op:

- ``aten.mm`` and ``aten.bmm``: one :class:`MatmulRecord`. A ``bmm`` record is a batched
  product; each member ``i`` (``A[i]·B[i]``) is its own checked product (spec §2). For HF's
  eager attention the batch index is ``s·n_h + h``, so members come in lexicographic ``(s, h)``
  order.
- ``aten.addmm`` and ``aten.baddbmm``: accepted only when the bias term is inert (``beta == 0``
  or an all-zero bias) and ``alpha == 1``, so that the product is ``A·B`` as the spec defines
  it. A non-zero bias raises :class:`BiasedMatmulError`. Splitting the op into ``mm`` + ``add``
  would change the rounding of an unmodified model, which a passthrough must not do.
- A product of contracted dimension ``q = 1`` is an outer product, which is glue (P7). It goes
  to :attr:`MatmulCapture.glue_outer`, not to :attr:`MatmulCapture.records`.
- Any other matmul-like op (``REJECTED_OPS``) raises :class:`UnsupportedMatmulError`, so the
  inventory is complete. So does a non-``default`` overload of a handled op.
- Inside a phase, an op outside the ``aten`` and ``prims`` namespaces raises
  :class:`UnsupportedMatmulError` unless ``ALLOWED_NAMESPACE_OPS`` lists it. Other namespaces
  hold opaque products (``_quantized::linear``, ``onednn::qlinear_pointwise``,
  ``inductor::_mm_plus_mm``, custom ``torch.library`` ops) that no name list can track.

The op lists and ``param_storage_map`` live in ``computation/matmul_ops.py``, which the
verifier's substitution reads too (invariant 1: the verifier never imports this module).

Records are in call order, not the canonical order of ref block §6. Hand-off notes for A7
(labeling), from HF Llama eager in torch 2.9.1:

- **Linear layers.** Forward is ``mm(X, W.t())``, so ``B`` is a view of the parameter and
  ``b_info.param_name`` names it. Backward runs the weight gradient first:
  ``G_x = mm(δYᵀ, X)``, already ``[o, i]``, then ``δX_x = mm(δY, W)``, whose ``b_info`` names
  ``W``. ``G_k`` and ``G_v`` have the same shape and the same ``B`` (``X_k = X_v``), so tell them
  apart by either rule: the ``G_x`` output tensor *is* ``W_x.grad`` (AccumulateGrad adopts it
  without a copy; true for every linear weight, not for the tied ``W_E``, whose grad is a sum),
  or the ``G_x`` and ``δX_x`` records share the storage of ``A`` (``δYᵀ`` and ``δY``).
- **Attention backward**, per layer, in call order: ``δV = Aᵀ·δO`` ``[n, d_h]``,
  ``δA = δO·Vᵀ`` ``[n, n]``, then ``Q̃ᵀ·δS`` ``[d_h, n]``, then ``δQ̃ = δS·K̃`` ``[n, d_h]``. The
  third is the **transpose** of the spec's ``δK̃ = δSᵀ·Q̃``: autograd's bmm backward computes
  ``grad_B = Aᵀ·grad`` for ``B = K̃ᵀ``. A7 commits the contiguous transpose of that output as
  the ``δK̃`` leaf and rebuilds its operands as ``(δSᵀ, Q̃)``.
- **Phases.** The phase is the caller's flag, nothing more. Activation checkpointing would rerun
  forward matmuls inside the backward phase and mislabel them, and add products the reference
  block does not count. The prover must assert it is off.

Captured tensors are referenced, not cloned. Aliasing facts that matter to callers:

- Forward operand ``B`` of a linear layer is ``W.t()``, a view of the parameter, so it shares
  the parameter's version counter. ``optimizer.step()`` bumps it.
- AccumulateGrad adopts a weight-gradient product as ``param.grad``, so an in-place
  ``zero_grad(set_to_none=False)`` mutates a captured product.
- :attr:`OperandInfo.storage_ptr` identifies a storage only while the capture holds the tensor.
  After :meth:`MatmulCapture.release_operands` the allocator may reuse it. Links between records
  are kept as :attr:`OperandInfo.producer_index`, computed while every tensor is live.

Invariant 6 is enforced by :meth:`MatmulCapture.assert_unmodified`, which compares each captured
tensor's ``_version`` with the value seen at capture. Call it after hashing the products, before
the optimizer touches the model, and again when the transcript is handed to the verifier. No
clone is needed on HF Llama eager in torch 2.9.1: no captured tensor is mutated in place during
forward and backward. The version counter misses writes that bypass autograd's bookkeeping:
through ``.data``, ``tensor.set_()``, a NumPy array or DLPack capsule sharing the memory, or raw
pointer access. The guard catches accidents in torch code, not deliberate tampering.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from verification.computation.matmul_ops import (
    ALLOWED_NAMESPACE_OPS,
    HANDLED_OPS,
    REJECTED_OPS,
    TRUSTED_NAMESPACES,
)

__all__ = [
    "PHASES",
    "CaptureError",
    "UnsupportedMatmulError",
    "BiasedMatmulError",
    "PhaseError",
    "MutatedCaptureError",
    "OperandInfo",
    "MatmulRecord",
    "MatmulCapture",
]

PHASES = ("forward", "backward")

_aten = torch.ops.aten

# Handled ops: name -> (is_batched, has_bias). Only the `.default` overload is accepted.
_HANDLED = {
    "mm": (False, False),
    "bmm": (True, False),
    "addmm": (False, True),
    "baddbmm": (True, True),
}
assert frozenset(_HANDLED) == HANDLED_OPS


class CaptureError(RuntimeError):
    """Base class for capture failures."""


class UnsupportedMatmulError(CaptureError):
    """A matmul-like op outside the handled set reached dispatch."""


class BiasedMatmulError(CaptureError):
    """An addmm/baddbmm carried a non-zero bias or alpha != 1; spec products are bias-free."""


class PhaseError(CaptureError):
    """A matmul ran outside a ``cap.phase(...)`` block, or phases were misused."""


class MutatedCaptureError(CaptureError):
    """A captured tensor was mutated in place after capture (invariant 6)."""


@dataclass(frozen=True)
class OperandInfo:
    """Cheap identity of a captured tensor, for later labeling (A7).

    ``storage_ptr`` is meaningful only while the capture holds the tensor.
    ``producer_index`` is the :attr:`MatmulRecord.index` of the earlier record whose output
    storage this tensor aliases (the same tensor or a view of it), or ``None`` when glue or a
    parameter produced it. It stays valid after :meth:`MatmulCapture.release_operands`.
    """

    storage_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    param_name: str | None  # set when the tensor aliases a model parameter's storage
    producer_index: int | None = None


@dataclass
class MatmulRecord:
    """One captured aten matmul call. ``a``, ``b``, ``out`` are the live tensors.

    ``a``, ``b`` and ``bias`` are ``None`` after :meth:`MatmulCapture.release_operands`.
    """

    index: int  # call index among all matmul-family calls (checked and glue)
    op: str  # e.g. "aten.mm.default"
    phase: str
    a: torch.Tensor | None
    b: torch.Tensor | None
    out: torch.Tensor
    q: int  # contracted dimension
    batch: int | None  # None for mm/addmm; the batch size for bmm/baddbmm
    a_info: OperandInfo
    b_info: OperandInfo
    out_info: OperandInfo
    bias: torch.Tensor | None = None
    alpha: float = 1.0
    beta: float = 1.0
    versions: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def n_members(self) -> int:
        """Checked products this call contributes (spec §2: one per batch member)."""
        return 1 if self.batch is None else self.batch

    def tensors(self) -> dict[str, torch.Tensor]:
        """The captured tensors still held, by role."""
        ts = {"a": self.a, "b": self.b, "out": self.out, "bias": self.bias}
        return {k: t for k, t in ts.items() if t is not None}

    def members(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield ``(A_i, B_i, P_i)`` views, one per checked product."""
        if self.a is None or self.b is None:
            raise CaptureError(f"record #{self.index}: operands were released")
        if self.batch is None:
            yield self.a, self.b, self.out
        else:
            for i in range(self.batch):
                yield self.a[i], self.b[i], self.out[i]


class MatmulCapture(TorchDispatchMode):
    """Passthrough dispatch mode recording every matmul of a forward + backward."""

    def __init__(self, param_names: Mapping[int, str] | None = None) -> None:
        super().__init__()
        self.param_names: Mapping[int, str] = dict(param_names or {})
        self.records: list[MatmulRecord] = []
        self.glue_outer: list[MatmulRecord] = []
        self._phase: str | None = None
        self._n_calls = 0
        self._producers: dict[int, int] = {}  # output storage ptr -> record index

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in PHASES:
            raise PhaseError(f"phase must be one of {PHASES}, got {name!r}")
        if self._phase is not None:
            raise PhaseError(f"phase {name!r} entered inside phase {self._phase!r}")
        self._phase = name
        try:
            yield
        finally:
            self._phase = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        if (self._phase is not None and func.namespace not in TRUSTED_NAMESPACES
                and f"{func.namespace}::{name}" not in ALLOWED_NAMESPACE_OPS):
            raise UnsupportedMatmulError(
                f"{func} reached dispatch inside phase {self._phase!r}. An op outside the "
                f"aten and prims namespaces may hide a matrix product the capture can't "
                f"record. List it in ALLOWED_NAMESPACE_OPS only after checking it forms none.")
        if func.namespace == "aten" and name in _HANDLED:
            if func._overloadname != "default":
                raise UnsupportedMatmulError(
                    f"{func} reached dispatch; only the .default overload of {name} is handled")
            if self._phase is None:
                raise PhaseError(f"{func} ran outside a cap.phase('forward'|'backward') block")
            parsed = self._parse(func, name, args, kwargs)
            out = func(*args, **kwargs)
            self._record(func, name, parsed, out)
            return out
        if func.namespace == "aten" and name in REJECTED_OPS:
            raise UnsupportedMatmulError(
                f"{func} reached dispatch. The capture handles only mm, bmm, addmm and "
                f"baddbmm, so the matmul inventory would be incomplete (spec §3.1). For "
                f"attention, load the model with attn_implementation='eager' (§8.A.4).")
        return func(*args, **kwargs)

    @staticmethod
    def _parse(func, name, args, kwargs):
        """Return ``(a, b, bias, alpha, beta)``; raise on a bias the spec can't express."""
        _, has_bias = _HANDLED[name]
        bias = None
        alpha, beta = 1, 1
        if has_bias:
            bias, a, b = args[0], args[1], args[2]
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)
            if alpha != 1 or (beta != 0 and bool(bias.ne(0).any())):
                raise BiasedMatmulError(
                    f"{func} with beta={beta}, alpha={alpha} and a non-zero bias: the spec "
                    f"checks P = A·B with no bias term. A model with biased linears needs a "
                    f"protocol decision on how the bias enters C.")
        else:
            a, b = args[0], args[1]
        return a, b, bias, alpha, beta

    def _info(self, t: torch.Tensor) -> OperandInfo:
        ptr = t.untyped_storage().data_ptr()
        return OperandInfo(ptr, t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype,
                           self.param_names.get(ptr), self._producers.get(ptr))

    def _record(self, func, name, parsed, out: torch.Tensor) -> None:
        batched, _ = _HANDLED[name]
        a, b, bias, alpha, beta = parsed
        q = a.shape[-1]
        rec = MatmulRecord(
            index=self._n_calls, op=str(func), phase=self._phase, a=a, b=b, out=out, q=q,
            batch=a.shape[0] if batched else None,
            a_info=self._info(a), b_info=self._info(b), out_info=self._info(out),
            bias=bias, alpha=float(alpha), beta=float(beta),
        )
        rec.versions = {k: t._version for k, t in rec.tensors().items()}
        # Every captured output stays referenced, so its storage pointer is not reused while
        # the capture is live.
        self._producers[rec.out_info.storage_ptr] = rec.index
        self._n_calls += 1
        (self.glue_outer if q == 1 else self.records).append(rec)

    # ---- after capture -------------------------------------------------------------------

    def release_operands(self) -> None:
        """Drop the references to ``a``, ``b`` and ``bias``, keeping ``out`` and the infos.

        Operands that are glue outputs can then be freed. Links between records survive as
        :attr:`OperandInfo.producer_index`.
        """
        for rec in self.records + self.glue_outer:
            rec.a = rec.b = rec.bias = None
            rec.versions = {k: v for k, v in rec.versions.items() if k == "out"}

    def assert_unmodified(self) -> None:
        """Raise if any held captured tensor was mutated in place since capture (invariant 6).

        Run it after hashing, before the optimizer step, and at hand-off to the verifier. See
        the module docstring for the writes the version counter cannot see.
        """
        bad = []
        for rec in self.records + self.glue_outer:
            now = {k: t._version for k, t in rec.tensors().items()}
            which = [k for k, v in rec.versions.items() if now.get(k, v) != v]
            if which:
                bad.append(f"#{rec.index} {rec.op} ({rec.phase}): {', '.join(which)}")
        if bad:
            raise MutatedCaptureError(
                "captured tensors mutated in place after capture: " + "; ".join(bad[:10])
                + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))

    @property
    def n_products(self) -> int:
        """Checked products, counting each batch member separately (the spec's ``M``)."""
        return sum(r.n_members for r in self.records)

    def summary(self) -> dict:
        ops = Counter((r.phase, r.op) for r in self.records)
        members: Counter[str] = Counter()
        for r in self.records:
            members[r.phase] += r.n_members
        qs = [r.q for r in self.records]
        return {
            "n_calls": len(self.records),
            "n_products": self.n_products,
            "products_by_phase": dict(members),
            "calls_by_phase_op": {f"{p} {o}": n for (p, o), n in sorted(ops.items())},
            "n_glue_outer_calls": len(self.glue_outer),
            "n_glue_outer_products": sum(r.n_members for r in self.glue_outer),
            "q_min": min(qs) if qs else None,
            "q_max": max(qs) if qs else None,
            "output_bytes": sum(r.out.numel() * r.out.element_size() for r in self.records),
        }
