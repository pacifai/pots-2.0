"""Model-agnostic matmul capture: a passthrough ``TorchDispatchMode`` (§8.A.4).

Wrap the forward pass and ``loss.backward()`` of an unmodified model in a
:class:`MatmulCapture`. Every matmul-family aten op that reaches dispatch is recorded with its
operands as the op received them and its real output. The mode calls the real op and returns
the real result, so training numerics are unchanged.

Usage::

    cap = MatmulCapture(param_names=param_storage_map(model))
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
- Any other matmul-like op raises :class:`UnsupportedMatmulError`, so the inventory is complete.

Captured tensors are referenced, not cloned. Two aliasing facts matter to callers:

- Forward operand ``B`` of a linear layer is ``W.t()``, a view of the parameter, so it shares
  the parameter's version counter. ``optimizer.step()`` bumps it.
- AccumulateGrad adopts a weight-gradient product as ``param.grad`` without copying, so an
  in-place ``zero_grad(set_to_none=False)`` mutates a captured product.

Invariant 6 is enforced by :meth:`MatmulCapture.assert_unmodified`, which compares each captured
tensor's ``_version`` with the value seen at capture. Call it after hashing or copying the
products and before the optimizer touches the model. No clone is needed on HF Llama eager in
torch 2.9.1: no captured tensor is mutated in place during forward and backward.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
    "param_storage_map",
]

PHASES = ("forward", "backward")

_aten = torch.ops.aten

# Handled ops: overload packet -> (is_batched, has_bias). Only the `.default` overload is
# accepted; `.out` and `.dtype` variants fall through to the unsupported check.
_HANDLED = {
    _aten.mm: (False, False),
    _aten.bmm: (True, False),
    _aten.addmm: (False, True),
    _aten.baddbmm: (True, True),
}

# Matmul-like aten ops that must not reach dispatch unhandled. `matmul`, `linear`, `einsum`
# and friends are CompositeImplicit and normally decompose before dispatch; they are listed
# in case a backend keeps them whole.
_UNSUPPORTED_NAMES = (
    "dot", "vdot", "inner", "mv", "addmv", "addr", "ger", "outer", "addbmm",
    "matmul", "linear", "bilinear", "einsum", "tensordot", "chain_matmul", "linalg_multi_dot",
    "_addmm_activation", "_int_mm", "_scaled_mm", "_weight_int8pack_mm", "_weight_int4pack_mm",
    "_sparse_mm", "_sparse_addmm", "sparse_sampled_addmm", "mkldnn_linear", "_mkldnn_linear",
    "_scaled_dot_product_attention_math", "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_fused_attention_overrideable", "_flash_attention_forward",
    "_efficient_attention_forward", "convolution", "_convolution",
)
_UNSUPPORTED_PACKETS = frozenset(
    getattr(_aten, n) for n in _UNSUPPORTED_NAMES if hasattr(_aten, n)
)


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


def param_storage_map(model: torch.nn.Module) -> dict[int, str]:
    """``{untyped_storage().data_ptr(): name}`` over ``model.named_parameters()``.

    Tied parameters share one storage and appear once, under the first name.
    """
    out: dict[int, str] = {}
    for name, p in model.named_parameters():
        out.setdefault(p.untyped_storage().data_ptr(), name)
    return out


@dataclass(frozen=True)
class OperandInfo:
    """Cheap identity of a captured tensor, for later labeling (A7)."""

    storage_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    param_name: str | None  # set when the tensor aliases a model parameter's storage

    @classmethod
    def of(cls, t: torch.Tensor, param_names: Mapping[int, str]) -> "OperandInfo":
        ptr = t.untyped_storage().data_ptr()
        return cls(ptr, t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype,
                   param_names.get(ptr))


@dataclass
class MatmulRecord:
    """One captured aten matmul call. ``a``, ``b``, ``out`` are the live tensors."""

    index: int  # call index among all matmul-family calls (checked and glue)
    op: str  # e.g. "aten.mm.default"
    phase: str
    a: torch.Tensor
    b: torch.Tensor
    out: torch.Tensor
    q: int  # contracted dimension
    batch: int | None  # None for mm/addmm; the batch size for bmm/baddbmm
    a_info: OperandInfo
    b_info: OperandInfo
    out_info: OperandInfo
    bias: torch.Tensor | None = None
    alpha: float = 1.0
    beta: float = 1.0
    versions: tuple[int, ...] = field(default=(), repr=False)

    @property
    def n_members(self) -> int:
        """Checked products this call contributes (spec §2: one per batch member)."""
        return 1 if self.batch is None else self.batch

    def tensors(self) -> tuple[torch.Tensor, ...]:
        ts = (self.a, self.b, self.out)
        return ts if self.bias is None else ts + (self.bias,)

    def members(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield ``(A_i, B_i, P_i)`` views, one per checked product."""
        if self.batch is None:
            yield self.a, self.b, self.out
        else:
            for i in range(self.batch):
                yield self.a[i], self.b[i], self.out[i]


def _versions(ts: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    return tuple(t._version for t in ts)


class MatmulCapture(TorchDispatchMode):
    """Passthrough dispatch mode recording every matmul of a forward + backward."""

    def __init__(self, param_names: Mapping[int, str] | None = None) -> None:
        super().__init__()
        self.param_names: Mapping[int, str] = dict(param_names or {})
        self.records: list[MatmulRecord] = []
        self.glue_outer: list[MatmulRecord] = []
        self._phase: str | None = None
        self._n_calls = 0

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
        packet = func.overloadpacket
        if packet in _HANDLED:
            if func._overloadname != "default" or kwargs.get("out") is not None:
                raise UnsupportedMatmulError(
                    f"{func} reached dispatch; only the .default overload of "
                    f"{packet.__name__} is handled")
            if self._phase is None:
                raise PhaseError(f"{func} ran outside a cap.phase('forward'|'backward') block")
            parsed = self._parse(func, packet, args, kwargs)
            out = func(*args, **kwargs)
            self._record(func, packet, parsed, out)
            return out
        if packet in _UNSUPPORTED_PACKETS:
            raise UnsupportedMatmulError(
                f"{func} reached dispatch. The capture handles only mm, bmm, addmm and "
                f"baddbmm, so the matmul inventory would be incomplete (spec §3.1). For "
                f"attention, load the model with attn_implementation='eager' (§8.A.4).")
        return func(*args, **kwargs)

    @staticmethod
    def _parse(func, packet, args, kwargs):
        """Return ``(a, b, bias, alpha, beta)``; raise on a bias the spec can't express."""
        _, has_bias = _HANDLED[packet]
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

    def _record(self, func, packet, parsed, out: torch.Tensor) -> None:
        batched, _ = _HANDLED[packet]
        a, b, bias, alpha, beta = parsed
        q = a.shape[-1]
        pn = self.param_names
        rec = MatmulRecord(
            index=self._n_calls, op=str(func), phase=self._phase, a=a, b=b, out=out, q=q,
            batch=a.shape[0] if batched else None,
            a_info=OperandInfo.of(a, pn), b_info=OperandInfo.of(b, pn),
            out_info=OperandInfo.of(out, pn), bias=bias, alpha=float(alpha), beta=float(beta),
        )
        rec.versions = _versions(rec.tensors())
        self._n_calls += 1
        (self.glue_outer if q == 1 else self.records).append(rec)

    # ---- after capture -------------------------------------------------------------------

    def assert_unmodified(self) -> None:
        """Raise if any captured tensor was mutated in place since capture (invariant 6)."""
        bad = []
        for rec in self.records + self.glue_outer:
            now = _versions(rec.tensors())
            if now != rec.versions:
                names = ("a", "b", "out", "bias")
                which = [names[i] for i, (x, y) in enumerate(zip(rec.versions, now)) if x != y]
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
        members = Counter()
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
