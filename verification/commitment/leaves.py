"""Leaf encoding and hashing for the step commitment ``h`` and the dataset root ``h_D`` (P9b).

The prover commits with these functions and the verifier recomputes ``h`` with them (check 2),
so both run one encoding path. A leaf is encoded by its position in the computation's layout
(ref block §6). Records go through ``computation.encode_record``, byte-identical to their
``h_D`` leaf (P9b); weights carry tag ``0x02`` and products tag ``0x03``, streamed zero-copy.

Bulk hashing (:func:`leaf_hashes_of`, :func:`leaf_hashes_until_error` and the functions built on
them) validates and encodes each leaf in order on the calling thread and hashes on worker
threads (``merkle.hash_leaves``), so the digests and the first error are those of a
leaf-by-leaf loop. With the native library enabled (``native.enabled()``, decision F15b) the
calling thread only checks each leaf's structure and collects its header and address; one C
call per batch of leaves then does the finiteness test and the hashing on a background
thread (:func:`_native_until_error`). The Python path stays as the reference, and the
counting pass runs it.

A caller that wants to know where the hashing time goes passes a :class:`HashStats` to the
bulk functions. Without one (the default) nothing is timed and the code runs as before.

A leaf may sit on a device (FS32b, the sharded placement of F22). Its bytes are the same and so
are the digests and errors; the copy to the host is ``hostcopy``'s alone. The Python path reads
a device leaf through ``hostcopy.host_tensor``; the native path copies a batch's device leaves
into a pinned staging buffer (``hostcopy.Stage``) while the previous batch is hashed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from setup.records import Record
from verification.commitment import hostcopy, merkle, native
from verification.commitment.gpu_blake3 import GpuBlake3
from verification.commitment.encoding import (
    TAG_PRODUCT,
    TAG_WEIGHT,
    NonFiniteError,
    check_payload,
    tensor_leaf_header,
    tensor_leaf_payload,
    tensor_leaf_payload_unchecked,
    tensor_payload_span,
)
from verification.commitment.merkle import (
    MerkleTree,
    hash_leaf,
    hash_leaves,
    hash_leaves_until_error,
    hash_record_leaf,
    merkle_root,
)
from verification.transcript.errors import LeafDtypeError, LeafShapeError, StoreMutationError

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation
    from verification.transcript.reader import LeafReader

__all__ = [
    "leaf_parts",
    "leaf_hash",
    "leaf_hashes_of",
    "leaf_hashes_until_error",
    "leaf_hashes",
    "transcript_root",
    "commit_leaves",
    "hash_step_leaves",
    "HashStats",
    "leaf_totals",
    "dataset_tree",
    "dataset_root",
]


class HashStats:
    """Where one bulk hashing call spent its time (seconds, accumulated across calls).

    Native path (:func:`_native_until_error`), the calling thread splits into ``read_s`` (in the
    caller's item iterator, for check 2 ``store.leaf``), ``wait_s`` (blocked on the worker) and
    ``producer_s`` (the rest: validation, batching, submitting). The worker's time per batch
    splits into ``native_s`` (``native.hash_leaves``), ``big_s`` (the big-leaf path, finiteness
    test and multithreaded hash) and ``worker_other_s`` (the rest of :func:`_hash_batch`). The
    worker runs beside the calling thread, so the two sides overlap: ``read_s + producer_s +
    wait_s`` is the call's wall time, and the worker's three fields are inside ``wait_s`` and
    the overlap.

    Python path (``native.disabled()``): ``hash_leaves`` hides its pool, so ``producer_s`` is the
    call's wall time less ``read_s`` and includes the waits; the worker fields stay 0.

    ``leaves`` and ``bytes`` count the leaves read and their tensor payload bytes (a record's
    encoding is not counted); ``batches`` the native batches submitted. ``known`` and ``kernel``
    count the leaves (also in ``leaves`` and ``bytes``) that took a digest hashed on arrival
    or were hashed in place on their GPU (:func:`hash_step_leaves`). The kernel's wait has no
    row of its own (the profile's hash rows are fixed): it is in the call's wall time.
    """

    __slots__ = ("read_s", "producer_s", "wait_s", "native_s", "big_s", "worker_other_s",
                 "batches", "leaves", "bytes", "known", "kernel")

    TIMES = ("read_s", "producer_s", "wait_s", "native_s", "big_s", "worker_other_s")

    def __init__(self) -> None:
        self.read_s = self.producer_s = self.wait_s = 0.0
        self.native_s = self.big_s = self.worker_other_s = 0.0
        self.batches = self.leaves = self.bytes = 0
        self.known = self.kernel = 0  # leaves that took a digest given, or went to the GPU kernel

    def times(self) -> dict[str, float]:
        """The six times by field name."""
        return {k: getattr(self, k) for k in self.TIMES}

    def emit(self, section: Any, prefix: str) -> None:
        """Record each time as a section ``<prefix>.<part>`` through ``section.add`` (B6's seam;
        ``read``, ``producer``, ``wait``, ``native``, ``big``, ``other``), leaving out a part with
        no time. ``read`` is always there once a call ran. Does nothing when
        ``section`` is ``None`` or has no ``add``."""
        add = getattr(section, "add", None)
        if add is None:
            return
        for k in self.TIMES:
            v = getattr(self, k)
            if v > 0:  # a part that took no time (no big leaf, the Python path's worker) has no row
                add(f"{prefix}.{k.removesuffix('_s').removeprefix('worker_')}", v)


def _observed(items: Iterable[tuple[int, Any]], stats: HashStats) -> Iterator[tuple[int, Any]]:
    """``items`` as they are, with the time inside ``next`` added to ``stats.read_s`` and each
    leaf counted: two clock reads per leaf."""
    it = iter(items)
    while True:
        t0 = perf_counter()
        try:
            item = next(it)
        except StopIteration:
            stats.read_s += perf_counter() - t0
            return
        stats.read_s += perf_counter() - t0
        stats.leaves += 1
        obj = item[1]
        if isinstance(obj, torch.Tensor):
            stats.bytes += obj.nbytes
        yield item


def _python_timed(stats: HashStats, run: Any) -> Any:
    """``run()`` with its wall time less the reads it made added to ``stats.producer_s``."""
    t0, r0 = perf_counter(), stats.read_s
    try:
        return run()
    finally:
        stats.producer_s += perf_counter() - t0 - (stats.read_s - r0)


def _tensor_slot(c: DeclaredComputation,
                 index: int) -> tuple[int, tuple[int, ...], torch.dtype, str]:
    """``(tag, shape, dtype, slot name)`` of a non-record leaf, from the ref block §6 layout."""
    n_s, n_w, M = c.n_s, c.n_w, c.M
    if n_s <= index < n_s + n_w:
        name = c.weight_names[index - n_s]
        return TAG_WEIGHT, tuple(c.weight_shapes[name]), c.weight_dtype, f"W_t[{name}]"
    if n_s + n_w <= index < n_s + n_w + M:
        spec = c.product(index - n_s - n_w + 1)
        return TAG_PRODUCT, spec.p_shape, c.product_dtype, f"P_{spec.m} ({spec.name})"
    if n_s + n_w + M <= index < c.n_leaves:
        name = c.weight_names[index - n_s - n_w - M]
        return TAG_WEIGHT, tuple(c.weight_shapes[name]), c.weight_dtype, f"W_t+1[{name}]"
    raise IndexError(f"leaf index {index} outside 0..{c.n_leaves - 1}")


def leaf_parts(c: DeclaredComputation, index: int, obj: Any) -> tuple[bytes | memoryview, ...]:
    """Canonical bytes of leaf ``index`` holding ``obj``, as parts to stream into ``hash_leaf``.

    A tensor payload is a zero-copy view of ``obj`` (of its host copy, for a device leaf), so
    ``obj`` must not change until hashed.
    """
    if 0 <= index < c.n_s:
        return (c.encode_record(obj),)
    tag, shape, dtype, slot = _tensor_slot(c, index)
    if not isinstance(obj, torch.Tensor):
        raise LeafShapeError(f"leaf {index} {slot}: expected a tensor, got {type(obj).__name__}")
    if tuple(obj.shape) != shape:
        raise LeafShapeError(f"leaf {index} {slot}: shape {tuple(obj.shape)}, declared {shape}")
    if obj.dtype != dtype:
        raise LeafDtypeError(f"leaf {index} {slot}: dtype {obj.dtype}, declared {dtype}")
    return tensor_leaf_header(tag, obj), tensor_leaf_payload(obj)


def leaf_hash(c: DeclaredComputation, index: int, obj: Any) -> bytes:
    """``H_leaf`` of leaf ``index`` holding ``obj`` (P9)."""
    return hash_leaf(*leaf_parts(c, index, obj))


def leaf_hashes_of(c: DeclaredComputation, items: Iterable[tuple[int, Any]],
                   stats: HashStats | None = None) -> list[bytes]:
    """:func:`leaf_hash` of each ``(index, obj)``, in order, hashed in parallel.

    ``items`` is consumed and each leaf validated on the calling thread, in order, so an error
    from either is the one a leaf-by-leaf loop raises first. The objects must not change until
    this returns. ``stats``, if given, is filled with the call's times (:class:`HashStats`);
    the digests and errors are the same with or without it.
    """
    if not native.enabled():
        if stats is None:
            return hash_leaves(leaf_parts(c, i, obj) for i, obj in items)
        return _python_timed(
            stats, lambda: hash_leaves(leaf_parts(c, i, obj) for i, obj in _observed(items, stats)))
    digests, err = _native_until_error(c, items, stats)
    if err is not None:
        raise err
    return digests


def leaf_hashes_until_error(c: DeclaredComputation, items: Iterable[tuple[int, Any]],
                            stats: HashStats | None = None
                            ) -> tuple[list[bytes], Exception | None]:
    """:func:`leaf_hashes_of` up to the first error: ``(digests before it, error or None)``.

    The error is the one a :func:`leaf_hash` loop raises first, and the digests are those of
    the leaves before it, so a caller can judge those leaves before the error, as a loop that
    compares each digest as it goes would.
    """
    if not native.enabled():
        if stats is None:
            return hash_leaves_until_error(leaf_parts(c, i, obj) for i, obj in items)
        return _python_timed(stats, lambda: hash_leaves_until_error(
            leaf_parts(c, i, obj) for i, obj in _observed(items, stats)))
    return _native_until_error(c, items, stats)


# The native path (decision F15b). A batch closes at this much payload or this many entries,
# and at most two closed batches are in flight, so the memory they pin stays bounded.
_BATCH_BYTES = 256 << 20
_BATCH_ENTRIES = 32_768
_MAX_IN_FLIGHT = 2
_NONFINITE = "float tensor contains NaN or Inf"  # the message of encoding.tensor_leaf_payload

_worker: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    """The one background thread that runs the native batches, created on first use."""
    global _worker
    if _worker is None:
        _worker = ThreadPoolExecutor(1, thread_name_prefix="leaves_native")
    return _worker


class _Batch:
    """A run of leaves in order. ``ents`` holds ``None`` for a small tensor leaf (its columns
    are in the lists, hashed in C), ``(header, tensor, kind)`` for a big one and the parts
    tuple for a record. ``tensors`` keeps every tensor alive until the batch is hashed."""

    __slots__ = ("ents", "hdr_idx", "ptr", "nbytes", "kind", "tensors", "payload", "stage", "dev")

    def __init__(self) -> None:
        self.ents: list[Any] = []
        self.hdr_idx: list[int] = []
        self.ptr: list[int] = []
        self.nbytes: list[int] = []
        self.kind: list[int] = []
        self.tensors: list[torch.Tensor] = []
        self.payload = 0
        # The pinned buffer holding the batch's device leaves, and for each its entry position,
        # the leaf, its ``_version`` when the copy was queued and its index (invariant 6).
        self.stage: hostcopy.Stage | None = None
        self.dev: list[tuple[int, torch.Tensor, int, int]] = []


def _hash_batch(batch: _Batch, hdrs: tuple[bytes, np.ndarray, np.ndarray], threads: int,
                stats: HashStats | None = None) -> tuple[list[bytes], Exception | None]:
    """Digests of the batch's leaves in order up to its first error, and that error. With
    ``stats`` (written by the worker thread only) the batch's time goes to ``native_s``,
    ``big_s`` and ``worker_other_s``."""
    if stats is None:
        return _hash_batch_body(batch, hdrs, threads, None)
    t0, n0, b0 = perf_counter(), stats.native_s, stats.big_s
    try:
        return _hash_batch_body(batch, hdrs, threads, stats)
    finally:
        stats.worker_other_s += perf_counter() - t0 - (stats.native_s - n0) - (stats.big_s - b0)


def _hash_batch_body(batch: _Batch, hdrs: tuple[bytes, np.ndarray, np.ndarray], threads: int,
                     stats: HashStats | None) -> tuple[list[bytes], Exception | None]:
    ents, n_small, changed = batch.ents, len(batch.hdr_idx), None
    if batch.stage is not None:
        batch.stage.wait()  # the staged bytes are only valid once the copies have completed
        for pos, t, version, index in batch.dev:
            if t._version != version:  # invariant 6: the leaves from here on are not judged
                ents = ents[:pos]
                n_small = sum(1 for e in ents if e is None)
                changed = StoreMutationError(f"leaf {index} was mutated in place while it was "
                                             "copied to the host")
                break
    bad, raw = 0, b""
    if n_small:
        t0 = perf_counter() if stats is not None else 0.0
        digests, bad = native.get().hash_leaves(
            hdrs[0], hdrs[1], hdrs[2], np.array(batch.hdr_idx[:n_small], dtype=np.uint32),
            np.array(batch.ptr[:n_small], dtype=np.uint64),
            np.array(batch.nbytes[:n_small], dtype=np.uint64),
            np.array(batch.kind[:n_small], dtype=np.uint8), threads=threads)
        if stats is not None:
            stats.native_s += perf_counter() - t0
        raw = digests.tobytes()
    out: list[bytes] = []
    s = 0  # the next small entry
    for e in ents:
        if e is None:
            if s >= bad:  # the first non-finite small entry (bad == n when none)
                return out, NonFiniteError(_NONFINITE)
            out.append(raw[32 * s:32 * s + 32])
            s += 1
        else:
            try:
                if len(e) == 3:
                    header, t, kind = e
                    t0 = perf_counter() if stats is not None else 0.0
                    try:
                        if not native.get().all_finite(t.data_ptr(), t.nbytes, kind,
                                                       threads=threads):
                            return out, NonFiniteError(_NONFINITE)
                        out.append(hash_leaf(header, tensor_leaf_payload_unchecked(t)))
                    finally:
                        if stats is not None:
                            stats.big_s += perf_counter() - t0
                else:
                    out.append(hash_leaf(*e))
            except Exception as err:
                return out, err
    return out, changed


def _native_until_error(c: DeclaredComputation, items: Iterable[tuple[int, Any]],
                        stats: HashStats | None = None) -> tuple[list[bytes], Exception | None]:
    """:func:`leaf_hashes_until_error` through ``native.hash_leaves``: the same digests and the
    same first error as a ``leaf_hash`` loop that stops at its first exception.

    On the calling thread, in order, each tensor leaf gets ``leaf_parts``' structural checks (a
    leaf that fails one goes to ``leaf_parts`` itself, which raises the loop's error), its
    header (built once per ``(tag, dtype, shape)``) and its address; records go through
    ``leaf_parts``. Closed batches (``_BATCH_BYTES`` of payload or ``_BATCH_ENTRIES`` entries)
    run on one background thread, two in flight at most: finiteness and hashing in C, and a
    leaf of ``merkle._MT_THRESHOLD`` bytes or more through ``native.all_finite`` and BLAKE3's
    multithreaded hash. The first batch in order with an error decides ahead of the iterable's.
    A device leaf is copied into its batch's pinned staging buffer (``hostcopy.Stage``) as it is
    added, and the worker hashes the staged bytes once the copies are done; a batch stages at most
    ``_BATCH_BYTES`` (a leaf larger than that, alone), so the staging memory is the batches in
    flight plus one. A device leaf whose ``_version`` moved between the copy being queued and its
    completion is a ``StoreMutationError`` at that leaf.
    It returns only once no worker reads a tensor any more; ``KeyboardInterrupt`` propagates
    after that too.

    With ``stats`` the calling thread's wall time is split into ``read_s`` (``items``' own
    ``next``), ``wait_s`` (blocked on the worker) and ``producer_s`` (the rest); see
    :class:`HashStats`. The worker's fields are written by the worker thread alone and read here
    after the final sync, which orders them.
    """
    # Every leaf of a list or tuple exists before the call, so one stream wait per device and
    # batch orders the copies after the work that produced them; a lazy iterable may produce
    # leaves as it is read, so each leaf waits.
    materialised = isinstance(items, (list, tuple))
    if stats is not None:
        items = _observed(items, stats)
        t_start, w0, r0 = perf_counter(), stats.wait_s, stats.read_s
    threads = torch.get_num_threads()
    n_s, n_w, M, n_leaves = c.n_s, c.n_w, c.M, c.n_leaves
    wt_end, prod_end = n_s + n_w, n_s + n_w + M
    big_at = merkle._MT_THRESHOLD
    batch_bytes, batch_entries = _BATCH_BYTES, _BATCH_ENTRIES
    finite_kind = native.FINITE_KIND

    headers: list[bytes] = []
    hdr_ids: dict[tuple[Any, ...], int] = {}
    hdr_off: list[int] = []
    hdr_len: list[int] = []
    slots: dict[int, tuple[int, tuple[int, ...], torch.dtype]] = {}  # index -> tag, shape, dtype

    pending: deque[tuple[Future[tuple[list[bytes], Exception | None]], _Batch]] = deque()
    stages: list[hostcopy.Stage] = []  # every staging buffer this call took, released at the end
    digests: list[bytes] = []
    state: list[Exception | None] = [None]  # the first batch error, which beats the producer's

    def collect() -> None:
        if stats is None:
            d, e = pending[0][0].result()  # popped only after the wait, which an interrupt can end
        else:
            t0 = perf_counter()
            try:
                d, e = pending[0][0].result()
            finally:
                stats.wait_s += perf_counter() - t0
        done = pending.popleft()[1]
        if done.stage is not None:
            hostcopy.POOL.release(done.stage)  # hashed: its buffer is free for a later batch
        if state[0] is None:
            digests.extend(d)
            state[0] = e

    def submit(b: _Batch) -> None:
        while len(pending) >= _MAX_IN_FLIGHT and state[0] is None:
            collect()
        if state[0] is None:
            hdrs = (b"".join(headers), np.array(hdr_off, dtype=np.uint64),
                    np.array(hdr_len, dtype=np.uint32))
            if stats is not None:
                stats.batches += 1
            if b.stage is not None:
                b.stage.fence()
            pending.append((_executor().submit(_hash_batch, b, hdrs, threads, stats), b))

    producer_error: Exception | None = None
    try:
        batch = _Batch()
        try:
            for i, obj in items:
                if 0 <= i < n_s:
                    batch.ents.append(leaf_parts(c, i, obj))
                else:
                    slot = slots.get(i)
                    if slot is None:
                        tag, shape, dtype, _ = _tensor_slot(c, i)  # IndexError off the layout
                        slot = slots[i] = (tag, shape, dtype)
                    tag, shape, dtype = slot
                    if (not isinstance(obj, torch.Tensor) or tuple(obj.shape) != shape
                            or obj.dtype != dtype):
                        leaf_parts(c, i, obj)  # raises the loop's error, with its message
                        raise AssertionError("leaf_parts accepted a leaf the layout rejects")
                    key = (tag, dtype, shape)
                    h = hdr_ids.get(key)
                    if h is None:
                        header = tensor_leaf_header(tag, obj)
                        h = hdr_ids[key] = len(headers)
                        hdr_off.append(sum(map(len, headers)))
                        hdr_len.append(len(header))
                        headers.append(header)
                    src = obj  # what the hash reads: the leaf, or the host copy of a device leaf
                    if hostcopy.on_device(obj):
                        check_payload(obj)
                        nbytes = obj.nbytes
                        stage = batch.stage
                        if stage is not None and not stage.fits(nbytes):
                            submit(batch)  # closes the batch: a batch stages at most batch_bytes
                            if state[0] is not None:
                                break
                            batch = _Batch()
                            stage = None
                        if stage is None:
                            stage = batch.stage = hostcopy.POOL.acquire(
                                batch_bytes, nbytes, obj.device.type == "cuda")
                            stage.wait_each = not materialised
                            stages.append(stage)
                        version = obj._version  # read as the copy is queued
                        src = stage.put(obj)
                        batch.dev.append((len(batch.ents), obj, version, i))
                    ptr, nbytes = tensor_payload_span(src)
                    batch.tensors.append(obj)
                    batch.payload += nbytes
                    if len(headers[h]) + nbytes >= big_at:
                        batch.ents.append((headers[h], src, finite_kind[dtype]))
                    else:
                        batch.ents.append(None)
                        batch.hdr_idx.append(h)
                        batch.ptr.append(ptr)
                        batch.nbytes.append(nbytes)
                        batch.kind.append(finite_kind[dtype])
                if batch.payload >= batch_bytes or len(batch.ents) >= batch_entries:
                    submit(batch)
                    if state[0] is not None:
                        break
                    batch = _Batch()
        except Exception as e:
            producer_error = e  # the leaves before it are still hashed
        if state[0] is None and batch.ents:
            submit(batch)
        while pending and state[0] is None:
            collect()
    finally:
        # An interrupt can land anywhere in here, even between ``submit`` and ``pending.append``;
        # the FIFO job below covers a batch whose future never reached ``pending``.
        # Never return or raise while a worker still reads the tensors. The worker is one FIFO
        # thread, so a job queued now ends after every batch ever submitted, even one whose
        # future an interrupt kept out of ``pending``.
        try:
            if _worker is not None:
                if stats is None:
                    _worker.submit(bool).result()
                else:
                    t0 = perf_counter()
                    _worker.submit(bool).result()
                    stats.wait_s += perf_counter() - t0
        finally:
            for stage in stages:  # no worker reads them now; drains any copy still queued
                hostcopy.POOL.release(stage)
        if stats is not None:
            stats.producer_s += (perf_counter() - t_start - (stats.wait_s - w0)
                                 - (stats.read_s - r0))
    return digests, state[0] if state[0] is not None else producer_error


def leaf_hashes(c: DeclaredComputation, reader: LeafReader) -> list[bytes]:
    """Every leaf hash, recomputed from the reader's leaves over the declared ``n_leaves``."""
    return leaf_hashes_of(c, ((i, reader.leaf(i)) for i in range(c.n_leaves)))


def transcript_root(c: DeclaredComputation, reader: LeafReader) -> bytes:
    """Check 2's recomputation: the root over the reader's leaves, never over supplied hashes."""
    return merkle_root(leaf_hashes(c, reader))


# FS30: where the bytes are decides who hashes them. Tests swap these two seams to run the
# kernel's table path on CPU tensors through the host build of the core.
_GPU: GpuBlake3 | None = None


def _hasher() -> GpuBlake3:
    global _GPU
    if _GPU is None:
        _GPU = GpuBlake3()
    return _GPU


def _kernel_leaf(obj: torch.Tensor) -> bool:
    """Whether the GPU kernel can hash ``obj`` where it sits: a contiguous, non-empty CUDA tensor
    of a dtype that has a leaf kind."""
    return (obj.device.type == "cuda" and obj.is_contiguous() and obj.numel() > 0
            and obj.dtype in native.FINITE_KIND)


def _table_stream(device: torch.device) -> Any:
    return hostcopy.side_stream(device) if device.type == "cuda" else None


class _Table:
    """Device leaves of one table: their indices, tensors, the ``_version`` of each read as the
    table was queued, and the pending hash."""

    __slots__ = ("idx", "tensors", "versions", "pend")

    def __init__(self, idx: list[int], tensors: list[torch.Tensor], versions: list[int],
                 pend: Any) -> None:
        self.idx, self.tensors, self.versions, self.pend = idx, tensors, versions, pend


def _enqueue_table(idx: list[int], tensors: list[torch.Tensor], tags: list[int]) -> _Table:
    """Queue one table's kernels on its device's side stream (after the work that made the
    tensors) and return at once."""
    device = tensors[0].device
    stream = _table_stream(device)
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))
    versions = [t._version for t in tensors]
    ids: dict[tuple[Any, ...], int] = {}
    headers: list[bytes] = []
    hdr_idx = np.empty(len(tensors), dtype=np.int64)
    for k, (t, tag) in enumerate(zip(tensors, tags)):
        key = (tag, t.dtype, tuple(t.shape))
        h = ids.get(key)
        if h is None:
            h = ids[key] = len(headers)
            headers.append(tensor_leaf_header(tag, t))
        hdr_idx[k] = h
    hdr_len = np.array([len(x) for x in headers], dtype=np.uint32)
    hdr_off = np.zeros(len(headers), dtype=np.uint64)
    np.cumsum(hdr_len[:-1], out=hdr_off[1:])
    pend = _hasher().enqueue_table(
        b"".join(headers), hdr_off, hdr_len, hdr_idx,
        np.fromiter((t.data_ptr() for t in tensors), dtype=np.uint64, count=len(tensors)),
        np.fromiter((t.nbytes for t in tensors), dtype=np.uint64, count=len(tensors)),
        np.fromiter((native.FINITE_KIND[t.dtype] for t in tensors), dtype=np.uint8,
                    count=len(tensors)),
        device=device, stream=stream, keep=list(tensors))
    return _Table(idx, tensors, versions, pend)


def _table_outcome(tb: _Table) -> tuple[list[bytes], tuple[int, Exception] | None]:
    """The table's digests (those before its first failing leaf) and that failure: a mutation at
    or before the first non-finite leaf beats it (invariant 6), as on the staged path."""
    digests, bad = tb.pend.result()
    mutated = next((k for k, t in enumerate(tb.tensors) if t._version != tb.versions[k]), None)
    if mutated is not None and mutated <= bad:
        i = tb.idx[mutated]
        return digests[:mutated], (i, StoreMutationError(
            f"leaf {i} was mutated in place while it was copied to the host"))
    if bad < len(tb.tensors):
        return digests, (tb.idx[bad], NonFiniteError(_NONFINITE))
    return digests, None


def hash_step_leaves(c: DeclaredComputation, leaves: Sequence[Any],
                     stats: HashStats | None = None, *,
                     known: Mapping[int, bytes] | None = None) -> list[bytes]:
    """Every leaf hash of a full transcript, in layout order: :func:`commit_leaves` without the
    tree, so a caller can time the two apart.

    ``known`` maps a leaf index to the digest of that leaf's bytes, hashed when the leaf
    arrived (FS30, ``StepOutput.arrival_digests``). Where the bytes are decides who hashes
    them. Every tensor leaf first gets the structural checks of :func:`leaf_parts`, in order;
    a leaf that fails one goes the host path, which raises its error. A known leaf that passes
    takes its digest. A CUDA tensor leaf that passes, with no digest, is hashed in place by the
    GPU kernel, in bounded tables queued before the host work starts and read after it. Every
    other leaf takes the host path (:func:`leaf_hashes_until_error`). The digests are the same
    whichever path hashed a leaf; on an error it is the one at the smallest failing leaf index,
    of the class and message of the plain path (a non-finite leaf is a ``NonFiniteError`` with
    the C library's message). With no ``known`` and no kernel leaf this is exactly
    :func:`leaf_hashes_of`.
    """
    if len(leaves) != c.n_leaves:
        raise ValueError(f"transcript has {len(leaves)} leaves, the computation declares "
                         f"{c.n_leaves}")
    known = known or {}
    if not known and not any(isinstance(x, torch.Tensor) and _kernel_leaf(x) for x in leaves):
        return leaf_hashes_of(c, list(enumerate(leaves)), stats)
    n_s = c.n_s
    slots: dict[int, tuple[int, tuple[int, ...], torch.dtype]] = {}
    out: list[bytes | None] = [None] * len(leaves)
    host: list[tuple[int, Any]] = []
    tables: list[_Table] = []
    open_: dict[torch.device, tuple[list[int], list[torch.Tensor], list[int], int]] = {}
    n_known = n_kernel = 0
    nbytes_known = nbytes_kernel = 0

    def close(dev: torch.device) -> None:
        idx, ts, tags, _ = open_.pop(dev)
        tables.append(_enqueue_table(idx, ts, tags))

    for i, obj in enumerate(leaves):
        if i >= n_s and isinstance(obj, torch.Tensor):
            slot = slots.get(i)
            if slot is None:
                tag, shape, dtype, _ = _tensor_slot(c, i)
                slot = slots[i] = (tag, shape, dtype)
            tag, shape, dtype = slot
            if tuple(obj.shape) == shape and obj.dtype == dtype:  # the structural checks passed
                d = known.get(i)
                if d is not None:
                    out[i] = d
                    n_known += 1
                    nbytes_known += obj.nbytes
                    continue
                if _kernel_leaf(obj):
                    idx, ts, tags, size = open_.get(obj.device, ([], [], [], 0))
                    idx.append(i)
                    ts.append(obj)
                    tags.append(tag)
                    open_[obj.device] = (idx, ts, tags, size + obj.nbytes)
                    n_kernel += 1
                    nbytes_kernel += obj.nbytes
                    if size + obj.nbytes >= _BATCH_BYTES or len(idx) >= _BATCH_ENTRIES:
                        close(obj.device)
                    continue
        host.append((i, obj))
    for dev in list(open_):
        close(dev)
    if stats is not None:
        stats.known += n_known
        stats.kernel += n_kernel
        stats.leaves += n_known + n_kernel
        stats.bytes += nbytes_known + nbytes_kernel
    failures: list[tuple[int, Exception]] = []
    try:
        if host:
            digests, err = leaf_hashes_until_error(c, host, stats)
            for (i, _), d in zip(host, digests):
                out[i] = d
            if err is not None:
                failures.append((host[len(digests)][0], err))
    finally:
        # Never leave a table queued unread: the stream still reads the leaves.
        outcomes = []
        for tb in tables:
            outcomes.append(_table_outcome(tb))
    for tb, (digests, fail) in zip(tables, outcomes):
        for i, d in zip(tb.idx, digests):
            out[i] = d
        if fail is not None:
            failures.append(fail)
    if failures:
        raise min(failures, key=lambda f: f[0])[1]
    return out  # type: ignore[return-value]


def commit_leaves(c: DeclaredComputation, leaves: Sequence[Any],
                  known: Mapping[int, bytes] | None = None) -> MerkleTree:
    """The Merkle tree over a full transcript's leaves, in layout order; ``.root`` is ``h``.
    ``known`` is :func:`hash_step_leaves`'s."""
    return MerkleTree(hash_step_leaves(c, leaves, known=known))


def leaf_totals(c: DeclaredComputation, leaves: Sequence[Any]) -> tuple[int, int]:
    """``(leaf count, bytes)`` that hashing ``leaves`` counts: a record's encoding, a tensor's
    header and ``nbytes``. It reads no payload."""
    total = 0
    for i, obj in enumerate(leaves):
        if i < c.n_s:
            total += len(c.encode_record(obj))
        else:
            total += len(tensor_leaf_header(_tensor_slot(c, i)[0], obj)) + obj.nbytes
    return len(leaves), total


def dataset_tree(c: DeclaredComputation, dataset: Sequence[Any]) -> MerkleTree:
    """The ``h_D`` tree over ``D``'s record leaves, for the batch's dataset paths (P9b)."""
    return MerkleTree([hash_leaf(c.encode_record(r)) for r in dataset])


def dataset_root(records: Sequence[Record]) -> bytes:
    """`h_D`: the Merkle root over the record leaves in record order (P9b)."""
    return merkle_root([hash_record_leaf(r) for r in records])
