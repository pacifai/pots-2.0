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
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from setup.records import Record
from verification.commitment import merkle, native
from verification.commitment.encoding import (
    TAG_PRODUCT,
    TAG_WEIGHT,
    NonFiniteError,
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
from verification.transcript.errors import LeafDtypeError, LeafShapeError

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
    encoding is not counted); ``batches`` the native batches submitted.
    """

    __slots__ = ("read_s", "producer_s", "wait_s", "native_s", "big_s", "worker_other_s",
                 "batches", "leaves", "bytes")

    TIMES = ("read_s", "producer_s", "wait_s", "native_s", "big_s", "worker_other_s")

    def __init__(self) -> None:
        self.read_s = self.producer_s = self.wait_s = 0.0
        self.native_s = self.big_s = self.worker_other_s = 0.0
        self.batches = self.leaves = self.bytes = 0

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

    A tensor payload is a zero-copy view of ``obj``, so ``obj`` must not change until hashed.
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

    __slots__ = ("ents", "hdr_idx", "ptr", "nbytes", "kind", "tensors", "payload")

    def __init__(self) -> None:
        self.ents: list[Any] = []
        self.hdr_idx: list[int] = []
        self.ptr: list[int] = []
        self.nbytes: list[int] = []
        self.kind: list[int] = []
        self.tensors: list[torch.Tensor] = []
        self.payload = 0


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
    bad, raw = 0, b""
    if batch.hdr_idx:
        t0 = perf_counter() if stats is not None else 0.0
        digests, bad = native.get().hash_leaves(
            hdrs[0], hdrs[1], hdrs[2], np.array(batch.hdr_idx, dtype=np.uint32),
            np.array(batch.ptr, dtype=np.uint64), np.array(batch.nbytes, dtype=np.uint64),
            np.array(batch.kind, dtype=np.uint8), threads=threads)
        if stats is not None:
            stats.native_s += perf_counter() - t0
        raw = digests.tobytes()
    out: list[bytes] = []
    s = 0  # the next small entry
    for e in batch.ents:
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
    return out, None


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
    It returns only once no worker reads a tensor any more; ``KeyboardInterrupt`` propagates
    after that too.

    With ``stats`` the calling thread's wall time is split into ``read_s`` (``items``' own
    ``next``), ``wait_s`` (blocked on the worker) and ``producer_s`` (the rest); see
    :class:`HashStats`. The worker's fields are written by the worker thread alone and read here
    after the final sync, which orders them.
    """
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

    pending: deque[Future[tuple[list[bytes], Exception | None]]] = deque()
    digests: list[bytes] = []
    state: list[Exception | None] = [None]  # the first batch error, which beats the producer's

    def collect() -> None:
        if stats is None:
            d, e = pending[0].result()  # popped only after the wait, which an interrupt can end
        else:
            t0 = perf_counter()
            try:
                d, e = pending[0].result()
            finally:
                stats.wait_s += perf_counter() - t0
        pending.popleft()
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
            pending.append(_executor().submit(_hash_batch, b, hdrs, threads, stats))

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
                    ptr, nbytes = tensor_payload_span(obj)
                    batch.tensors.append(obj)
                    batch.payload += nbytes
                    if len(headers[h]) + nbytes >= big_at:
                        batch.ents.append((headers[h], obj, finite_kind[dtype]))
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
        # Never return or raise while a worker still reads the tensors. The worker is one FIFO
        # thread, so a job queued now ends after every batch ever submitted, even one whose
        # future an interrupt kept out of ``pending``.
        if _worker is not None:
            if stats is None:
                _worker.submit(bool).result()
            else:
                t0 = perf_counter()
                _worker.submit(bool).result()
                stats.wait_s += perf_counter() - t0
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


def hash_step_leaves(c: DeclaredComputation, leaves: Sequence[Any],
                     stats: HashStats | None = None) -> list[bytes]:
    """Every leaf hash of a full transcript, in layout order: :func:`commit_leaves` without the
    tree, so a caller can time the two apart."""
    if len(leaves) != c.n_leaves:
        raise ValueError(f"transcript has {len(leaves)} leaves, the computation declares "
                         f"{c.n_leaves}")
    return leaf_hashes_of(c, enumerate(leaves), stats)


def commit_leaves(c: DeclaredComputation, leaves: Sequence[Any]) -> MerkleTree:
    """The Merkle tree over a full transcript's leaves, in layout order; ``.root`` is ``h``."""
    return MerkleTree(hash_step_leaves(c, leaves))


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
