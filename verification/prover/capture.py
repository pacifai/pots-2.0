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
- ``aten.addmm`` and ``aten.baddbmm``: accepted with ``alpha == 1`` when the bias term is inert
  (``beta == 0`` or an all-zero bias), so that the product is ``A·B``. ``addmm`` is also accepted
  in the bias rule's form (F14): a bias vector of length ``b.shape[1]`` added once
  (``beta == 1``), the output of a biased linear. Its record keeps ``bias`` and the output is the
  committed leaf. Any other bias raises :class:`BiasedMatmulError`. Splitting the op into
  ``mm`` + ``add`` would change the rounding of an unmodified model, which a passthrough must
  not do.
- A product of contracted dimension ``q = 1`` is an outer product, which is glue (P7). It goes
  to :attr:`MatmulCapture.glue_outer`, not to :attr:`MatmulCapture.records`.
- Any other matmul-like op (``REJECTED_OPS``) raises :class:`UnsupportedMatmulError`, so the
  inventory is complete. So does a non-``default`` overload of a handled op.
- Inside a phase, an op outside the ``aten`` and ``prims`` namespaces raises
  :class:`UnsupportedMatmulError` unless ``ALLOWED_NAMESPACE_OPS`` lists it. Other namespaces
  hold opaque products (``_quantized::linear``, ``onednn::qlinear_pointwise``,
  ``inductor::_mm_plus_mm``, custom ``torch.library`` ops) that no name list can track.
- Inside a phase, ``aten._to_copy``: one :class:`CopyRecord` in :attr:`MatmulCapture.copies`.
  Under mixed precision (F6b) these are autocast's casts of weights and activations and the
  autograd engine's casts of gradients. A copy of a parameter (or of such a copy) is held until
  :meth:`MatmulCapture.release_operands`, and an operand that aliases it carries
  :attr:`OperandInfo.param_copy`. Any other copy is held through a weak reference to its
  storage, so the capture keeps no activation alive that training would free, and a strong
  reference would stop AccumulateGrad adopting a gradient copy as ``W.grad``.

The op lists and ``param_storage_map`` live in ``computation/matmul_ops.py``, which the
verifier's substitution reads too (invariant 1: the verifier never imports this module).

Records are in call order, not the canonical order of ref block §6. Hand-off notes for A7
(labeling), from HF Llama and Qwen2 eager in torch 2.9.1:

- **Linear layers.** Forward is ``mm(X, W.t())``, so ``B`` is a view of the parameter and
  ``b_info.param_name`` names it. Backward runs the weight gradient first:
  ``G_x = mm(δYᵀ, X)``, already ``[o, i]``, then ``δX_x = mm(δY, W)``, whose ``b_info`` names
  ``W``. A biased linear (Qwen2's q, k, v) runs forward as ``addmm(b, X, W.t())`` and backward
  in the other order, ``δX_x`` then ``G_x``; its bias gradient is a sum, not a matmul. ``G_k``
  and ``G_v`` have the same shape and the same ``B`` (``X_k = X_v``), so tell them apart by
  either rule: the ``G_x`` output tensor *is* ``W_x.grad`` (AccumulateGrad adopts it
  without a copy; true for every linear weight, not for the tied ``W_E``, whose grad is a sum),
  or the ``G_x`` and ``δX_x`` records share the storage of ``A`` (``δYᵀ`` and ``δY``).
- **Attention backward**, per layer, in call order: ``δV = Aᵀ·δO`` ``[n, d_h]``,
  ``δA = δO·Vᵀ`` ``[n, n]``, then ``Q̃ᵀ·δS`` ``[d_h, n]``, then ``δQ̃ = δS·K̃`` ``[n, d_h]``. The
  third is the **transpose** of the spec's ``δK̃ = δSᵀ·Q̃``: autograd's bmm backward computes
  ``grad_B = Aᵀ·grad`` for ``B = K̃ᵀ``. A7 commits the contiguous transpose of that output as
  the ``δK̃`` leaf and rebuilds its operands as ``(δSᵀ, Q̃)``.
- **Mixed precision** (``(fp32, bf16)``, CPU autocast in torch 2.9.1). A linear's forward is
  ``mm(X', W'.t())``, where ``W'`` is autocast's bf16 copy of ``W`` (one per weight per pass)
  and ``X'`` a bf16 copy of ``X`` made per use, so q, k and v (and gate and up) read different
  copies; ``o`` and ``down`` receive bf16 inputs and read them directly. ``b_info.param_name``
  is then ``None`` and ``b_info.param_copy`` names ``W``; a biased linear's ``bias`` is a bf16
  copy of ``b_x`` too (:meth:`MatmulCapture.param_of`). ``δX_x = mm(δY, W')`` reads the same
  copy. ``G_x`` is a bf16 product, and AccumulateGrad adopts its fp32 copy as ``W_x.grad``
  (it adopts ``copy.detach()``, so ``W_x.grad`` shares the copy's storage, not its tensor;
  :meth:`MatmulCapture.is_copy_of`). Every product is bf16; the RoPE table stays fp32.
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
clone is needed on HF Llama or Qwen2 eager in torch 2.9.1: no captured tensor is mutated in
place during forward and backward. The version counter misses writes that bypass autograd's
bookkeeping: through ``.data``, ``tensor.set_()``, a NumPy array or DLPack capsule sharing the
memory, or raw pointer access. The guard catches accidents in torch code, not deliberate
tampering.

**Products on a GPU (FS9; F1c, F4a).** A captured output in CUDA memory is copied as soon as
the op returns into a pinned host buffer, on a side stream, and the record's ``out`` becomes
that buffer, so the transcript collects in host RAM and not on the GPU. A CPU output is kept
as it is, so on CPU the capture is unchanged. In order, per output:

1. the side stream waits for the stream that produced it (``wait_stream``), then copies it
   with ``non_blocking=True`` and records an event;
2. the capture keeps the GPU tensor until that event has completed, so the caching allocator
   can't hand its block to another tensor while the copy reads it (which is what
   ``record_stream`` would otherwise be for); each later record polls the queue and drops the
   ones whose copy is done;
3. a GPU tensor whose ``_version`` moved before its copy was done raises
   :class:`MutatedCaptureError`, since the copy may have read either value;
4. leaving the ``with`` block waits for every copy, so labeling reads only finished buffers.

The buffers come from :class:`HostBuffers`, allocated once per shape and dtype and reused by
every later step. A buffer is reused only when nothing else holds its storage: the step that
committed it has dropped its leaves. Operands are not copied: they stay on the GPU until
:meth:`MatmulCapture.release_operands`.
"""

from __future__ import annotations

import contextlib
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch.multiprocessing.reductions import StorageWeakRef
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
    "CopyRecord",
    "HostBuffers",
    "MatmulCapture",
]

PHASES = ("forward", "backward")

_aten = torch.ops.aten
_TO_COPY = _aten._to_copy.default

# Handled ops: name -> (is_batched, has_bias). Only the `.default` overload is accepted.
_HANDLED = {
    "mm": (False, False),
    "bmm": (True, False),
    "addmm": (False, True),
    "baddbmm": (True, True),
}
if frozenset(_HANDLED) != HANDLED_OPS:
    raise ImportError("capture._HANDLED is out of sync with matmul_ops.HANDLED_OPS")


class CaptureError(RuntimeError):
    """Base class for capture failures."""


class UnsupportedMatmulError(CaptureError):
    """A matmul-like op outside the handled set reached dispatch."""


class BiasedMatmulError(CaptureError):
    """An addmm/baddbmm carried alpha != 1 or a live bias outside F14's form."""


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
    ``param_copy`` names the parameter when the tensor aliases a ``_to_copy`` of it, such as
    autocast's bf16 copy of a weight (F6b).
    """

    storage_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    param_name: str | None  # set when the tensor aliases a model parameter's storage
    producer_index: int | None = None
    param_copy: str | None = None


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
    # The op's own output storage, which ``out`` no longer is once it was copied to the host.
    out_storage: StorageWeakRef | None = field(default=None, repr=False, compare=False)

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


@dataclass
class CopyRecord:
    """One captured ``aten._to_copy`` inside a phase. ``storage`` is a weak reference to the
    output's storage: it keeps the storage's identity, not its memory, so no other storage
    compares equal to it while the record lives."""

    index: int  # position in MatmulCapture.copies
    phase: str
    src_info: OperandInfo
    out_info: OperandInfo
    storage: StorageWeakRef = field(repr=False)

    @property
    def param(self) -> str | None:
        """The parameter this copies, directly or through earlier copies."""
        return self.src_info.param_name or self.src_info.param_copy


def _use_count(buf: torch.Tensor) -> int:
    """References to ``buf``'s storage, counted the same way on every call."""
    s = buf.untyped_storage()
    return torch._C._storage_Use_Count(s._cdata)


class HostBuffers:
    """Host buffers for captured GPU products, allocated once and reused every step (FS9).

    :meth:`take` hands out a buffer of the asked shape and dtype that nothing else holds: it
    reuses one whose storage has no reference beyond the pool's own, else allocates (pinned
    when ``pin``, which needs CUDA). Each buffer goes out as a fresh tensor over the whole
    storage (``set_``), not the pool's own tensor, so every holder, view or NumPy array of
    it raises the storage's use count until it is dropped. A buffer is taken again only after
    the step that committed it has dropped its leaves; two steps alive at once use two sets.
    One side stream per device carries the copies.
    """

    def __init__(self, *, pin: bool = True) -> None:
        self.pin = pin
        # (shape, dtype) -> [[(buffer, its use count when free)], where the next search starts]
        self._pool: dict[tuple, list] = {}
        self._streams: dict[torch.device, torch.cuda.Stream] = {}

    def take(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        slots = self._pool.setdefault(key, [[], 0])
        bufs, start = slots
        n = len(bufs)
        # A step takes a shape's buffers in the order the last step did, so the search starts
        # where the last one stopped.
        for i in range(n):
            j = (start + i) % n
            buf, free = bufs[j]
            if _use_count(buf) == free:
                slots[1] = (j + 1) % n
                return self._alias(buf)
        buf = torch.empty(key[0], dtype=dtype, pin_memory=self.pin)
        bufs.append((buf, _use_count(buf)))
        slots[1] = 0
        return self._alias(buf)

    @staticmethod
    def _alias(buf: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, dtype=buf.dtype).set_(buf.untyped_storage(), 0, buf.shape,
                                                    buf.stride())

    def stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self._streams.get(device)
        if s is None:
            s = self._streams[device] = torch.cuda.Stream(device)
        return s

    @property
    def n_buffers(self) -> int:
        return sum(len(bufs) for bufs, _ in self._pool.values())

    @property
    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for bufs, _ in self._pool.values()
                   for b, _ in bufs)


# Shared by every capture in the process, so step t+1 reuses step t's buffers.
HOST_BUFFERS = HostBuffers()


class MatmulCapture(TorchDispatchMode):
    """Passthrough dispatch mode recording every matmul of a forward + backward."""

    def __init__(self, param_names: Mapping[int, str] | None = None,
                 buffers: HostBuffers | None = None) -> None:
        super().__init__()
        self.param_names: Mapping[int, str] = dict(param_names or {})
        self.buffers = HOST_BUFFERS if buffers is None else buffers
        self.records: list[MatmulRecord] = []
        self.glue_outer: list[MatmulRecord] = []
        self.copies: list[CopyRecord] = []
        self._phase: str | None = None
        self._n_calls = 0
        # Output storage ptr -> (record index, the storage). On a GPU the capture drops an
        # output once copied, and the allocator may give its address to a new storage, so a
        # match needs the same storage too.
        self._producers: dict[int, tuple[int, StorageWeakRef]] = {}
        self._by_index: dict[int, MatmulRecord] = {}
        # A parameter copy's storage ptr -> the parameter. The copies are held, so no ptr is
        # reused while the map is live.
        self._param_copies: dict[int, str] = {}
        self._held: list[torch.Tensor] = []
        # GPU outputs whose copy to the host may still run: (record, tensor, _version, event).
        self._pending: deque[tuple[MatmulRecord, torch.Tensor, int, torch.cuda.Event]] = deque()

    def __exit__(self, exc_type, exc_val, exc_tb):
        out = super().__exit__(exc_type, exc_val, exc_tb)
        # Whatever reads the records after the block reads finished copies.
        self.wait_copies(check=exc_type is None)
        return out

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
        if func is _TO_COPY and self._phase is not None:
            out = func(*args, **kwargs)
            self._record_copy(args[0], out)
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
            f14 = name == "addmm" and beta == 1 and tuple(bias.shape) == (b.shape[1],)
            # F14's form first: reading the bias's values is a device sync on a GPU.
            if alpha != 1 or not (f14 or beta == 0 or not bool(bias.ne(0).any())):
                raise BiasedMatmulError(
                    f"{func} with beta={beta}, alpha={alpha} and a non-zero bias of shape "
                    f"{tuple(bias.shape)}: the bias rule (F14) covers only a bias vector added "
                    f"once to a non-batched product.")
        else:
            a, b = args[0], args[1]
        return a, b, bias, alpha, beta

    def _info(self, t: torch.Tensor) -> OperandInfo:
        ptr = t.untyped_storage().data_ptr()
        producer = self._producers.get(ptr)
        if producer is not None and producer[1] != StorageWeakRef(t.untyped_storage()):
            producer = None
        return OperandInfo(ptr, t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype,
                           self.param_names.get(ptr), None if producer is None else producer[0],
                           self._param_copies.get(ptr))

    def _record_copy(self, src: torch.Tensor, out: torch.Tensor) -> None:
        rec = CopyRecord(len(self.copies), self._phase, self._info(src), self._info(out),
                         StorageWeakRef(out.untyped_storage()))
        if rec.param is not None:
            self._param_copies[rec.out_info.storage_ptr] = rec.param
            self._held.append(out)
        self.copies.append(rec)

    def _record(self, func, name, parsed, out: torch.Tensor) -> None:
        batched, _ = _HANDLED[name]
        a, b, bias, alpha, beta = parsed
        q = a.shape[-1]
        rec = MatmulRecord(
            index=self._n_calls, op=str(func), phase=self._phase, a=a, b=b, out=out, q=q,
            batch=a.shape[0] if batched else None,
            a_info=self._info(a), b_info=self._info(b), out_info=self._info(out),
            bias=bias, alpha=float(alpha), beta=float(beta),
            out_storage=StorageWeakRef(out.untyped_storage()),
        )
        rec.versions = {k: t._version for k, t in rec.tensors().items()}
        self._producers[rec.out_info.storage_ptr] = (rec.index, rec.out_storage)
        self._by_index[rec.index] = rec
        self._n_calls += 1
        (self.glue_outer if q == 1 else self.records).append(rec)
        if out.is_cuda:
            self._offload(rec)

    def _offload(self, rec: MatmulRecord) -> None:
        """Copy a GPU output into a host buffer on the side stream; ``rec.out`` becomes it."""
        out = rec.out
        host = self.buffers.take(out.shape, out.dtype)
        side = self.buffers.stream(out.device)
        side.wait_stream(torch.cuda.current_stream(out.device))  # the op that wrote it
        with torch.cuda.stream(side):
            host.copy_(out, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        # Held until the copy is done, so the allocator can't reuse its memory under the copy.
        self._pending.append((rec, out, out._version, done))
        rec.out = host
        rec.versions["out"] = host._version
        self._drain(block=False)

    def _drain(self, *, block: bool, check: bool = True) -> None:
        """Drop the GPU outputs whose copy is done (all of them when ``block``)."""
        while self._pending:
            rec, gpu, version, done = self._pending[0]
            if not block and not done.query():
                return
            done.synchronize()
            self._pending.popleft()
            if check and gpu._version != version:
                raise MutatedCaptureError(
                    f"#{rec.index} {rec.op} ({rec.phase}): output written in place before its "
                    f"copy to the host was done")

    def wait_copies(self, *, check: bool = True) -> None:
        """Wait for every copy of a GPU output to the host and drop the GPU tensors. Leaving
        the ``with`` block calls it; a no-op on CPU."""
        self._drain(block=True, check=check)

    # ---- after capture -------------------------------------------------------------------

    def release_operands(self) -> None:
        """Drop the references to ``a``, ``b`` and ``bias``, keeping ``out`` and the infos.

        Operands that are glue outputs can then be freed. Links between records survive as
        :attr:`OperandInfo.producer_index`.
        """
        for rec in self.records + self.glue_outer:
            rec.a = rec.b = rec.bias = None
            rec.versions = {k: v for k, v in rec.versions.items() if k == "out"}
        self._held, self._param_copies = [], {}

    def assert_unmodified(self) -> None:
        """Raise if any held captured tensor was mutated in place since capture (invariant 6).

        Run it after hashing, before the optimizer step, and at hand-off to the verifier. See
        the module docstring for the writes the version counter cannot see.
        """
        self.wait_copies()
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

    def param_of(self, t: torch.Tensor) -> str | None:
        """The parameter ``t`` aliases, directly or as a copy; valid until
        :meth:`release_operands`."""
        ptr = t.untyped_storage().data_ptr()
        return self.param_names.get(ptr) or self._param_copies.get(ptr)

    @staticmethod
    def is_output_of(t: torch.Tensor, rec: MatmulRecord) -> bool:
        """Whether ``t`` shares the storage of ``rec``'s output as the op returned it, such as
        the ``W.grad`` AccumulateGrad adopts. On a GPU ``rec.out`` is the host copy, so compare
        with this rather than with its pointer. Storages compare by identity."""
        return (rec.out_storage is not None
                and rec.out_storage == StorageWeakRef(t.untyped_storage()))

    def is_copy_of(self, t: torch.Tensor, index: int) -> bool:
        """Whether ``t`` is a copy of record ``index``'s whole output, such as the fp32 copy of
        a bf16 weight gradient that AccumulateGrad adopts as ``W.grad``. Storages compare by
        identity, so an allocator reusing a freed copy's address never matches."""
        rec = self._by_index.get(index)
        if rec is None:
            return False
        o = rec.out_info
        whole = (o.storage_offset, o.shape, o.stride)
        mine = StorageWeakRef(t.untyped_storage())
        view = (t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
        return any(c.src_info.producer_index == index
                   and (c.src_info.storage_offset, c.src_info.shape, c.src_info.stride) == whole
                   and c.storage == mine
                   and (c.out_info.storage_offset, c.out_info.shape, c.out_info.stride,
                        c.out_info.dtype) == view
                   for c in self.copies)

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
            "n_copies": len(self.copies),
            "n_param_copies": sum(c.param is not None for c in self.copies),
            "q_min": min(qs) if qs else None,
            "q_max": max(qs) if qs else None,
            "output_bytes": sum(r.out.numel() * r.out.element_size() for r in self.records),
        }
