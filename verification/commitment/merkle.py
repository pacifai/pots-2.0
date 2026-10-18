"""BLAKE3 Merkle tree with RFC 6962 shape (S9c, P9a, P9b, S6f).

`H_leaf(x) = BLAKE3(0x00 ‖ x)`, `H_node(l, r) = BLAKE3(0x01 ‖ l ‖ r)`. A range of `n > 1`
leaves splits at the largest power of two strictly below `n`, so an unpaired node is
promoted unchanged (P9a). The same tree builds the step root `h` and the dataset root `h_D`.

`hash_leaves` hashes many leaves at once on a pool of worker threads. blake3 releases the GIL
while it hashes, so the workers run in parallel. Each digest is the one `hash_leaf` gives.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import blake3
import torch

from setup.records import Record, encode_record
from verification.commitment.encoding import tensor_leaf_header, tensor_leaf_payload

DIGEST_SIZE = 32
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"
# Above this size BLAKE3's multithreaded tree mode is used; the digest is identical. It is
# set high because `hash_leaves` already spreads leaves over threads: nested threading slows
# the common 1–4 MB leaves, while a 100 MB leaf hashed on one thread would be a long tail.
_MT_THRESHOLD = 16 << 20
# `hash_leaves` hands leaves to the workers in chunks of about this many bytes, so the many
# small leaves (32–64 KB attention products) don't each pay a task's overhead.
_CHUNK_BYTES = 16 << 20

LeafParts = Sequence[bytes | memoryview]


def hash_leaf(*parts: bytes | memoryview) -> bytes:
    """BLAKE3(0x00 ‖ parts…), streamed part by part with no concatenation copy."""
    big = sum(memoryview(p).nbytes for p in parts) >= _MT_THRESHOLD
    h = blake3.blake3(max_threads=blake3.blake3.AUTO) if big else blake3.blake3()
    h.update(_LEAF_PREFIX)
    for p in parts:
        h.update(p)
    return h.digest()


def _hash_chunk(chunk: list[LeafParts]) -> tuple[list[bytes], Exception | None]:
    """Hash a chunk's leaves in order; stop at the first error and return it with the
    digests before it."""
    out: list[bytes] = []
    try:
        for item in chunk:
            out.append(hash_leaf(*item))
    except Exception as e:
        return out, e
    return out, None


_pool: ThreadPoolExecutor | None = None
_pool_size = 0


def _executor(workers: int) -> ThreadPoolExecutor:
    """The shared worker pool, created on first use and resized when the thread count changes."""
    global _pool, _pool_size
    if _pool is None or _pool_size != workers:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(workers, thread_name_prefix="hash_leaves")
        _pool_size = workers
    return _pool


def hash_leaves_until_error(leaves: Iterable[LeafParts]) -> tuple[list[bytes], Exception | None]:
    """`hash_leaf(*parts)` for each item, in order, on worker threads, up to the first error:
    ``(digests of the leaves before it, the error or None)``.

    The error is the one a plain loop over the items meets first, and it is returned, not
    raised, so a caller can judge the leaves before it first. The iterable is consumed on the
    calling thread, so whatever produces the parts (store reads, validation, encoding) runs
    there in order, and an error it raises sits after every item it yielded. The items already
    yielded are still hashed then. A worker's own error (a malformed part) sits at its leaf:
    each chunk stops at its first error, and the first chunk in order with one decides, ahead
    of the iterable's error. A leaf with an error never gets a digest, nor does any after it.

    Returns only once no worker reads the items' parts any more: they alias tensors the
    caller may change next. An error that is not an `Exception` (``KeyboardInterrupt``)
    propagates once the workers stop. The pool has `torch.get_num_threads()` workers, the
    run's thread budget (`VERIF_THREADS`); with one thread the leaves are hashed inline.
    """
    workers = torch.get_num_threads()
    if workers <= 1:
        out: list[bytes] = []
        try:
            for item in leaves:
                out.append(hash_leaf(*item))
        except Exception as e:
            return out, e
        return out, None
    pool = _executor(workers)
    futures: list[Future[tuple[list[bytes], Exception | None]]] = []
    chunk: list[LeafParts] = []
    size = 0
    producer_error: Exception | None = None
    try:
        for item in leaves:
            size += sum(memoryview(p).nbytes for p in item)
            chunk.append(item)
            if size >= _CHUNK_BYTES:
                futures.append(pool.submit(_hash_chunk, chunk))
                chunk, size = [], 0
    except Exception as e:
        producer_error = e
    except BaseException:
        for f in futures:
            f.cancel()
        for f in futures:
            if not f.cancelled():
                f.exception()  # wait; the parts alias leaves the caller may change next
        raise
    if chunk:  # also after a producer error: the items before it are still hashed
        futures.append(pool.submit(_hash_chunk, chunk))
    digests: list[bytes] = []
    first_error: Exception | None = None
    for n, f in enumerate(futures):
        hs, err = f.result()
        digests.extend(hs)
        if err is not None:
            first_error = err
            for later in futures[n + 1:]:  # later leaves can't change the outcome
                later.cancel()
            for later in futures[n + 1:]:
                if not later.cancelled():
                    later.exception()  # wait, as above
            break
    if first_error is None:
        first_error = producer_error
    return digests, first_error


def hash_leaves(leaves: Iterable[LeafParts]) -> list[bytes]:
    """`hash_leaf(*parts)` for each item, in order, hashed on worker threads.

    Raises the error a plain loop raises first (`hash_leaves_until_error`), unchanged, once
    the workers have stopped reading the parts; it never returns a digest list then.
    """
    digests, err = hash_leaves_until_error(leaves)
    if err is not None:
        raise err
    return digests


def hash_node(left: bytes, right: bytes) -> bytes:
    """BLAKE3(0x01 ‖ left ‖ right)."""
    return blake3.blake3(_NODE_PREFIX + left + right).digest()


def hash_tensor_leaf(tag: int, t: torch.Tensor) -> bytes:
    """Leaf hash of a tensor leaf, streaming the payload zero-copy (S9c, S9d)."""
    return hash_leaf(tensor_leaf_header(tag, t), tensor_leaf_payload(t))


def hash_record_leaf(rec: Record) -> bytes:
    """Leaf hash of a token record, identical in `h` and `h_D` (P9b)."""
    return hash_leaf(encode_record(rec))


def _split(n: int) -> int:
    """Largest power of two strictly below `n` (n ≥ 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def _check_hashes(leaf_hashes: Sequence[bytes]) -> None:
    if len(leaf_hashes) == 0:
        raise ValueError("a Merkle tree needs at least one leaf")
    for x in leaf_hashes:
        if type(x) is not bytes or len(x) != DIGEST_SIZE:
            raise ValueError("leaf hashes must be 32-byte bytes objects")


def merkle_root(leaf_hashes: Sequence[bytes]) -> bytes:
    """Root over already-hashed leaves (RFC 6962 MTH with BLAKE3)."""
    _check_hashes(leaf_hashes)

    def mth(lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            return leaf_hashes[lo]
        k = _split(hi - lo)
        return hash_node(mth(lo, lo + k), mth(lo + k, hi))

    return mth(0, len(leaf_hashes))


def _tree_nodes(leaf_hashes: Sequence[bytes]) -> bytearray:
    """All `2n − 1` node hashes of the tree over `leaf_hashes`, in a flat pre-order buffer.

    Node `p` sits at bytes `[32p, 32p + 32)`; index 0 is the root. A range `[lo, hi)` of two or
    more leaves at index `p` has its left child `[lo, lo + k)` at `p + 1` and its right child
    `[lo + k, hi)` at `p + 2k`, with `k = _split(hi − lo)`. A one-leaf range is the leaf hash.
    """
    n = len(leaf_hashes)
    buf = bytearray(DIGEST_SIZE * (2 * n - 1))

    def fill(p: int, lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            h = leaf_hashes[lo]
        else:
            k = _split(hi - lo)
            h = hash_node(fill(p + 1, lo, lo + k), fill(p + 2 * k, lo + k, hi))
        buf[DIGEST_SIZE * p : DIGEST_SIZE * (p + 1)] = h
        return h

    fill(0, 0, n)
    return buf


class MerkleTree:
    """RFC 6962 tree over leaf hashes, caching every subtree hash for O(log n) updates (S6f)."""

    def __init__(self, leaf_hashes: Sequence[bytes]) -> None:
        _check_hashes(leaf_hashes)
        self._n = len(leaf_hashes)
        self._buf = _tree_nodes(leaf_hashes)

    def _at(self, p: int) -> bytes:
        return bytes(self._buf[DIGEST_SIZE * p : DIGEST_SIZE * (p + 1)])

    def _set(self, p: int, h: bytes) -> None:
        self._buf[DIGEST_SIZE * p : DIGEST_SIZE * (p + 1)] = h

    @property
    def root(self) -> bytes:
        return self._at(0)

    @property
    def n_leaves(self) -> int:
        return self._n

    def _index(self, lo: int, hi: int) -> int:
        """Pre-order index of the node for the range `[lo, hi)`; `KeyError` if there is none."""
        p, nlo, nhi = 0, 0, self._n
        while (nlo, nhi) != (lo, hi):
            if nhi - nlo == 1:
                raise KeyError((lo, hi))
            k = _split(nhi - nlo)
            mid = nlo + k
            if nlo <= lo and hi <= mid:
                p, nhi = p + 1, mid
            elif mid <= lo and hi <= nhi:
                p, nlo = p + 2 * k, mid
            else:
                raise KeyError((lo, hi))
        return p

    def node(self, lo: int, hi: int) -> bytes:
        """Subtree hash of the leaf range `[lo, hi)`, which must be a node of this tree."""
        return self._at(self._index(lo, hi))

    def leaf(self, i: int) -> bytes:
        self._check_index(i)
        return self._at(self._index(i, i + 1))

    def _check_index(self, i: int) -> None:
        if not 0 <= i < self._n:
            raise IndexError(f"leaf index {i} out of range for {self._n} leaves")

    def _descent(self, i: int) -> tuple[list[tuple[int, int, bool]], int]:
        """Top-down `(index, k, went_left)` per internal node on the path to leaf `i`, and the
        leaf's index."""
        out = []
        p, lo, hi = 0, 0, self._n
        while hi - lo > 1:
            k = _split(hi - lo)
            left = i < lo + k
            out.append((p, k, left))
            if left:
                p, hi = p + 1, lo + k
            else:
                p, lo = p + 2 * k, lo + k
        return out, p

    def path(self, i: int) -> list[bytes]:
        """RFC 6962 audit path for leaf `i`, sibling nearest the leaf first."""
        self._check_index(i)
        steps, _ = self._descent(i)
        return [self._at(p + 2 * k if left else p + 1) for p, k, left in reversed(steps)]

    def update_leaf(self, i: int, new_hash: bytes) -> bytes:
        """Replace leaf `i` and return the new root, rehashing only the path (S6f)."""
        self._check_index(i)
        if type(new_hash) is not bytes or len(new_hash) != DIGEST_SIZE:
            raise ValueError("leaf hashes must be 32-byte bytes objects")
        steps, leaf_p = self._descent(i)
        self._set(leaf_p, new_hash)
        for p, k, _ in reversed(steps):
            self._set(p, hash_node(self._at(p + 1), self._at(p + 2 * k)))
        return self.root


def verify_path(
    leaf_hash: bytes, index: int, n_leaves: int, path: Sequence[bytes], root: bytes
) -> bool:
    """RFC 9162 §2.1.3.2 inclusion-proof verification with BLAKE3 node hashing."""
    if not 0 <= index < n_leaves:
        return False
    fn, sn, r = index, n_leaves - 1, leaf_hash
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = hash_node(p, r)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = hash_node(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root
