"""BLAKE3 Merkle tree with RFC 6962 shape (S9c, P9a, P9b, S6f).

`H_leaf(x) = BLAKE3(0x00 ‖ x)`, `H_node(l, r) = BLAKE3(0x01 ‖ l ‖ r)`. A range of `n > 1`
leaves splits at the largest power of two strictly below `n`, so an unpaired node is
promoted unchanged (P9a). The same tree builds the step root `h` and the dataset root `h_D`.
"""

from __future__ import annotations

from collections.abc import Sequence

import blake3
import torch

from setup.records import Record, encode_record
from verification.commitment.encoding import tensor_leaf_header, tensor_leaf_payload

DIGEST_SIZE = 32
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"
# Above this size BLAKE3's multithreaded tree mode is used; the digest is identical.
_MT_THRESHOLD = 1 << 20


def hash_leaf(*parts: bytes | memoryview) -> bytes:
    """BLAKE3(0x00 ‖ parts…), streamed part by part with no concatenation copy."""
    big = sum(memoryview(p).nbytes for p in parts) >= _MT_THRESHOLD
    h = blake3.blake3(max_threads=blake3.blake3.AUTO) if big else blake3.blake3()
    h.update(_LEAF_PREFIX)
    for p in parts:
        h.update(p)
    return h.digest()


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


class MerkleTree:
    """RFC 6962 tree over leaf hashes, caching every subtree hash for O(log n) updates (S6f)."""

    def __init__(self, leaf_hashes: Sequence[bytes]) -> None:
        _check_hashes(leaf_hashes)
        self._n = len(leaf_hashes)
        # Subtree hash per leaf range [lo, hi). A range of one leaf is the leaf hash itself.
        self._nodes: dict[tuple[int, int], bytes] = {}
        self._build(list(leaf_hashes), 0, self._n)

    def _build(self, leaves: list[bytes], lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            h = leaves[lo]
        else:
            k = _split(hi - lo)
            h = hash_node(self._build(leaves, lo, lo + k), self._build(leaves, lo + k, hi))
        self._nodes[(lo, hi)] = h
        return h

    @property
    def root(self) -> bytes:
        return self._nodes[(0, self._n)]

    @property
    def n_leaves(self) -> int:
        return self._n

    def leaf(self, i: int) -> bytes:
        self._check_index(i)
        return self._nodes[(i, i + 1)]

    def _check_index(self, i: int) -> None:
        if not 0 <= i < self._n:
            raise IndexError(f"leaf index {i} out of range for {self._n} leaves")

    def _ranges(self, i: int) -> list[tuple[int, int, int, int]]:
        """Top-down `(lo, mid, hi)` splits on the path to leaf `i`, with sibling range."""
        out = []
        lo, hi = 0, self._n
        while hi - lo > 1:
            mid = lo + _split(hi - lo)
            if i < mid:
                out.append((lo, hi, mid, hi))  # sibling is the right range
                hi = mid
            else:
                out.append((lo, hi, lo, mid))  # sibling is the left range
                lo = mid
        return out

    def path(self, i: int) -> list[bytes]:
        """RFC 6962 audit path for leaf `i`, sibling nearest the leaf first."""
        self._check_index(i)
        return [self._nodes[(slo, shi)] for _, _, slo, shi in reversed(self._ranges(i))]

    def update_leaf(self, i: int, new_hash: bytes) -> bytes:
        """Replace leaf `i` and return the new root, rehashing only the path (S6f)."""
        self._check_index(i)
        if type(new_hash) is not bytes or len(new_hash) != DIGEST_SIZE:
            raise ValueError("leaf hashes must be 32-byte bytes objects")
        self._nodes[(i, i + 1)] = new_hash
        for lo, hi, _, _ in reversed(self._ranges(i)):
            mid = lo + _split(hi - lo)
            self._nodes[(lo, hi)] = hash_node(self._nodes[(lo, mid)], self._nodes[(mid, hi)])
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
