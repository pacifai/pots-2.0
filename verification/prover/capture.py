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
committed it has dropped its leaves and every storage object of them. Operands are not copied:
they stay on the GPU until :meth:`MatmulCapture.release_operands`. The host copy is always
contiguous, so labeling judges the layout of the op's own output, recorded in
:attr:`MatmulRecord.out_info`.

**Products on their owner's GPU (FS32d; F22).** Under a sharded placement (``placement`` given
and ``placement.sharded``) a GPU product goes to the GPU that will own its leaf, by peer copy,
and no host buffer is taken. The capture doesn't know the product's index ``m`` yet, so it
sends each output to a hinted device (:meth:`MatmulCapture._hint`) and labeling fixes a miss
with one more copy, counted by :meth:`MatmulCapture.count_hint_misses`. The hint, per op:

1. a weight operand (``param_name`` or ``param_copy``) gives ``placement.weight_device(name)``,
   and that device becomes the current unit's;
2. else an operand that an earlier op already routed (a weight gradient's ``X`` is the forward
   linear's input, an attention backward bmm reads the forward bmm's operands) gives that
   operand's device, which is the same unit's;
3. else the current unit's device (the attention bmms of a forward layer, between its ``Y_v``
   and ``Y_o``).

A wrong hint costs a copy, never a bit. The copy:

- the output is copied whole, so a ``bmm``'s members stay views of one storage on the owner,
  into a contiguous tensor allocated on the destination, which becomes ``rec.out``;
- it runs on the side stream of the *source* device, after ``wait_stream`` on the producing
  stream, as above: PyTorch's cross-device ``copy_`` runs on the source device's current
  stream, which is that side stream. The destination tensor is allocated on the destination's
  current stream, and ``copy_`` makes that stream wait for the copy, so work on it sees the
  finished data and the allocator can't hand the block to another tensor early. That is why
  there's no ``record_stream``: the one stream that allocates is the one ordered after the
  write, and leaving the ``with`` block waits for every copy on the host, so a reader on
  any other stream sees finished data too;
- the source is held until the event completes, and a ``_version`` moved before then raises
  :class:`MutatedCaptureError`, as above;
- the owner hashes the copy on arrival (FS34): the kernel's table for the product is queued on
  the destination's side stream behind the copy's event, as the host path queues it behind its
  host copy, and its digest is keyed by the copy's storage, so :meth:`arrival_digest` serves
  it as on the one-GPU path. The training GPU hashes nothing when it owns no layers (the
  four-GPU machine); on the two-GPU pod GPU 0 also owns layers and hashes the products it keeps;
- an output already on its hinted device is kept as it is, and hashed where it sits, on that
  device's side stream after the producing stream; the digest is paired with the ``_version``
  read when the hash was queued, since the kept tensor is autograd's own (a later write is
  :class:`MutatedCaptureError`, and never a digest of a later version); a hint that is the host
  takes the host-buffer path above.

On CPU nothing is copied, but the hint is computed and counted all the same.
"""

from __future__ import annotations

import contextlib
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from verification.commitment import native
from verification.commitment.encoding import TAG_PRODUCT, tensor_leaf_header
from verification.commitment.gpu_blake3 import GpuBlake3, PendingHash
from verification.computation.matmul_ops import (
    ALLOWED_NAMESPACE_OPS,
    HANDLED_OPS,
    REJECTED_OPS,
    TRUSTED_NAMESPACES,
)

if TYPE_CHECKING:
    from verification.computation.placement import Placement

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
    # Sharded placement only (FS32d): the device the capture hinted for this output.
    hint: torch.device | None = field(default=None, repr=False, compare=False)

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


def _holds(entry: list) -> tuple[int, int]:
    """What holds a pool entry's storage: its use count (tensors, views, NumPy arrays) and the
    references to the pool's Python storage object, which ``untyped_storage()`` hands to every
    caller while it lives. The pool measures both through this one call, so its own
    references count the same each time."""
    return torch._C._storage_Use_Count(entry[1]._cdata), sys.getrefcount(entry[1])


class HostBuffers:
    """Host buffers for captured GPU products, allocated once and reused every step (FS9).

    :meth:`take` hands out a buffer of the asked shape and dtype that nothing else holds: it
    reuses one whose storage has no reference beyond the pool's own, else allocates (pinned
    when ``pin``, which needs CUDA). Each buffer goes out as a fresh tensor over the whole
    storage (``set_``), not the pool's own tensor, so every holder, view or NumPy array of
    it raises the storage's use count until it is dropped. The pool also keeps the storage's
    Python object, which ``untyped_storage()`` returns to anyone who asks while it lives, so a
    held storage object raises that object's reference count. A buffer is taken again only
    after the step that committed it has dropped its leaves and every storage object of them;
    two steps alive at once use two sets. One side stream per device carries the copies.
    """

    def __init__(self, *, pin: bool = True) -> None:
        self.pin = pin
        # (shape, dtype) -> [[[buffer, its storage, _holds when free]], where the next
        # search starts]
        self._pool: dict[tuple, list] = {}
        self._streams: dict[torch.device, torch.cuda.Stream] = {}

    def take(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        slots = self._pool.setdefault(key, [[], 0])
        entries, start = slots
        n = len(entries)
        # A step takes a shape's buffers in the order the last step did, so the search starts
        # where the last one stopped.
        for i in range(n):
            j = (start + i) % n
            entry = entries[j]
            if _holds(entry) == entry[2]:
                slots[1] = (j + 1) % n
                return self._alias(entry)
        buf = torch.empty(key[0], dtype=dtype, pin_memory=self.pin)
        entries.append([buf, buf.untyped_storage(), None])
        del buf
        entry = entries[-1]
        entry[2] = _holds(entry)
        slots[1] = 0
        return self._alias(entry)

    @staticmethod
    def _alias(entry: list) -> torch.Tensor:
        buf = entry[0]
        return torch.empty(0, dtype=buf.dtype).set_(entry[1], 0, buf.shape, buf.stride())

    def stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self._streams.get(device)
        if s is None:
            s = self._streams[device] = torch.cuda.Stream(device)
        return s

    @property
    def n_buffers(self) -> int:
        return sum(len(entries) for entries, _ in self._pool.values())

    @property
    def nbytes(self) -> int:
        return sum(e[0].numel() * e[0].element_size() for entries, _ in self._pool.values()
                   for e in entries)


# Shared by every capture in the process, so step t+1 reuses step t's buffers.
HOST_BUFFERS = HostBuffers()

_HASHER: GpuBlake3 | None = None


def _hasher() -> GpuBlake3:
    global _HASHER
    if _HASHER is None:
        _HASHER = GpuBlake3()
    return _HASHER


class MatmulCapture(TorchDispatchMode):
    """Passthrough dispatch mode recording every matmul of a forward + backward."""

    def __init__(self, param_names: Mapping[int, str] | None = None,
                 buffers: HostBuffers | None = None,
                 placement: Placement | None = None, *, hash_arrivals: bool = True,
                 hasher: GpuBlake3 | None = None) -> None:
        super().__init__()
        # FS30: hash each CUDA product on arrival: unsharded, on the side stream after its host
        # copy; sharded (FS34), on the owner after the peer copy or where it is kept.
        # ``hasher`` is the kernel's host for tests.
        self.hash_arrivals = hash_arrivals
        self._hasher = hasher
        # host storage -> (shape, dtype, the host tensor's offset, digest, version): the whole
        # output's one digest (O18), ``None`` for a non-finite output.
        self._arrived: dict[StorageWeakRef, tuple] = {}
        self._arrival_stats = {"records": 0, "leaves": 0, "bytes": 0, "nonfinite": 0}
        # Only a sharded placement routes (FS32d); unsharded, the capture is as before.
        self.placement = placement if placement is not None and placement.sharded else None
        self.n_hint_misses: int | None = None
        self._unit_device: torch.device | None = None  # the current unit's, by the last weight op
        self._operand_device: dict[int, torch.device] = {}  # operand storage ptr -> its hint
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
        # GPU outputs whose copy may still run: (record, tensor, _version, event). The fifth
        # item is the arrival hash enqueued behind the copy (or after the output, where it is
        # kept), if any; the sixth the ``_version`` of ``rec.out`` when that hash was queued,
        # for a kept output (autograd's own tensor), else ``None`` (read at drain: the capture's
        # own copy, which nothing else writes).
        self._pending: deque[tuple[MatmulRecord, torch.Tensor, int, torch.cuda.Event,
                                   PendingHash | None, int | None]] = deque()

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
        if self.placement is not None:
            rec.hint = self._hint(rec)
        if out.is_cuda:
            if rec.hint is None or rec.hint.type == "cpu":
                self._offload(rec)
            else:
                self._route(rec)

    def _hint(self, rec: MatmulRecord) -> torch.device:
        """The device to send ``rec``'s output to (module docstring). Operand storages are held
        by the records until :meth:`release_operands`, so a pointer here is never reused."""
        place = self.placement
        assert place is not None
        dev = None
        for info in (rec.b_info, rec.a_info):
            name = info.param_name or info.param_copy
            if name is not None:
                with contextlib.suppress(KeyError):
                    dev = self._unit_device = place.weight_device(name)
                    break
        if dev is None:
            dev = next((d for info in (rec.b_info, rec.a_info)
                        if (d := self._operand_device.get(info.storage_ptr)) is not None), None)
        if dev is None:
            dev = self._unit_device or place.devices[-1]  # type: ignore[index]
        for info in (rec.a_info, rec.b_info):
            self._operand_device.setdefault(info.storage_ptr, dev)
        return dev

    def count_hint_misses(self, leaves: Sequence[torch.Tensor],
                          device_of: Callable[[int], torch.device]) -> int:
        """Count the product leaves (as labeling returned them) whose hinted device is not
        ``device_of(g)``, ``g`` being the leaf's position, and keep the count for
        :meth:`summary`. A leaf that is a record's output (or a view of it) is judged by that
        record's hint, so a CPU run checks the hint too; any other leaf (the δK̃ transpose, a new tensor) by the device it sits on, and not at
        all on the CPU."""
        hints = {StorageWeakRef(r.out.untyped_storage()): r.hint
                 for r in self.records + self.glue_outer if r.hint is not None}
        miss = 0
        for g, p in enumerate(leaves):
            at = hints.get(StorageWeakRef(p.untyped_storage()))
            if at is None and p.device.type == "cpu":
                continue  # a new CPU tensor was never sent anywhere: nothing to judge
            miss += (p.device if at is None else at) != device_of(g)
        self.n_hint_misses = miss
        return miss

    def _route(self, rec: MatmulRecord) -> None:
        """Peer-copy a GPU output to ``rec.hint``, whole, on the source device's side stream;
        ``rec.out`` becomes the copy (module docstring)."""
        out, dest = rec.out, rec.hint
        assert dest is not None
        if out.device == dest:
            self._hash_in_place(rec)  # already on its owner: kept, as on CPU
            return
        side = self.buffers.stream(out.device)
        side.wait_stream(torch.cuda.current_stream(out.device))  # the op that wrote it
        with torch.cuda.stream(side):
            moved = torch.empty(out.shape, dtype=out.dtype, device=dest)
            moved.copy_(out, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        try:
            pend = None
            if self._hashable(rec, moved):
                owner = self.buffers.stream(dest)
                owner.wait_event(done)  # a cross-device wait: the copy has landed
                pend = self._hash_arrival(rec, moved, owner)
        except BaseException:
            side.synchronize()  # the copy and any kernel already launched read the tensors
            self.buffers.stream(dest).synchronize()
            raise
        self._pending.append((rec, out, out._version, done, pend, None))
        rec.out = moved
        rec.versions["out"] = moved._version
        self._drain(block=False)

    def _hash_in_place(self, rec: MatmulRecord) -> None:
        """Hash a product that is already on its owner where it sits (FS34): on that device's
        side stream, after the stream that wrote it. ``rec.out`` stays the autograd tensor, so
        the entry carries the ``_version`` read as the hash is queued."""
        out = rec.out
        if not self._hashable(rec, out):
            return
        side = self.buffers.stream(out.device)
        side.wait_stream(torch.cuda.current_stream(out.device))  # the op that wrote it
        try:
            pend = self._hash_arrival(rec, out, side)
            with torch.cuda.stream(side):
                done = torch.cuda.Event()
                done.record(side)
        except BaseException:
            side.synchronize()  # a kernel already launched reads ``out``
            raise
        # Held until the hash is done, so the allocator can't reuse its memory under it.
        self._pending.append((rec, out, out._version, done, pend, out._version))
        self._drain(block=False)

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
        try:
            pend = self._hash_arrival(rec, out, side) if self._hashable(rec, out) else None
        except BaseException:
            side.synchronize()  # the copy and any kernel already launched read ``out``
            raise
        # Held until the copy and the hash are done, so the allocator can't reuse its memory
        # under them.
        self._pending.append((rec, out, out._version, done, pend, None))
        rec.out = host
        rec.versions["out"] = host._version
        self._drain(block=False)

    def _hashable(self, rec: MatmulRecord, out: torch.Tensor) -> bool:
        return (self.hash_arrivals and rec.q != 1 and out.is_contiguous() and out.numel() > 0
                and out.dtype in native.FINITE_KIND)

    def _hash_arrival(self, rec: MatmulRecord, out: torch.Tensor,
                      side: torch.cuda.Stream) -> PendingHash:
        """Enqueue the product leaf's hash on ``side`` (behind the host copy): the whole output
        as one table entry, with header ``tensor_leaf_header(TAG_PRODUCT, out)`` (O18: 3-D for a
        batched bmm, whose members are one leaf). Nothing here waits."""
        hdr = tensor_leaf_header(TAG_PRODUCT, out)
        hasher = self._hasher if self._hasher is not None else _hasher()
        return hasher.enqueue_table(
            hdr, np.zeros(1, np.uint64), np.array([len(hdr)], np.uint32),
            np.zeros(1, np.int64),
            np.array([out.data_ptr()], np.uint64),
            np.array([out.nbytes], np.uint64),
            np.full(1, native.FINITE_KIND[out.dtype], np.uint8),
            device=out.device, stream=side, keep=[out])

    def _arrived_from(self, rec: MatmulRecord, pend: PendingHash,
                      version: int | None = None) -> None:
        """Key ``pend``'s digests by ``rec.out``'s storage. ``version`` is the ``_version`` of
        ``rec.out`` read when the hash was queued (a kept output); ``None`` reads it now."""
        digests, bad = pend.result()
        host = rec.out
        self._arrived[StorageWeakRef(host.untyped_storage())] = (
            tuple(host.shape), host.dtype, host.storage_offset(),
            digests[0] if digests else None, host._version if version is None else version)
        st = self._arrival_stats
        st["records"] += 1
        st["leaves"] += 1
        st["bytes"] += host.nbytes
        st["nonfinite"] += pend.n - bad

    def arrival_digest(self, p: torch.Tensor) -> bytes | None:
        """The digest of ``p`` hashed on arrival, if ``p`` is exactly the whole output that was
        hashed (O18: a batched bmm's output is one leaf), found by storage identity, then
        offset, shape, dtype and contiguity. ``None`` for any other tensor (a member view
        ``out[i]``, a slice, the ``δK̃`` transpose, a new tensor, an output with a NaN or Inf,
        or a buffer written since).
        """
        self.wait_copies()
        hit = self._arrived.get(StorageWeakRef(p.untyped_storage()))
        if hit is None:
            return None
        shape, dtype, base, digest, version = hit
        if p.dtype != dtype or tuple(p.shape) != shape or p.storage_offset() != base \
                or not p.is_contiguous() or p._version != version:
            return None
        return digest

    def _drain(self, *, block: bool, check: bool = True) -> None:
        """Drop the GPU outputs whose copy is done (all of them when ``block``)."""
        while self._pending:
            rec, gpu, version, done, pend, arrived = self._pending[0]
            if not block and not (done.query() and (pend is None or pend.done())):
                return
            done.synchronize()
            self._pending.popleft()
            if pend is not None:
                self._arrived_from(rec, pend, arrived)
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
        } | ({"arrival_hash": dict(self._arrival_stats)} if self._arrival_stats["records"]
             else {}) | ({} if self.placement is None else {"hint_misses": self.n_hint_misses})
