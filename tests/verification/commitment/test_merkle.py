import random

import blake3
import pytest
import torch

from setup.records import Record, encode_record
from verification.commitment.encoding import TAG_PRODUCT, TAG_WEIGHT, encode_tensor_leaf
from verification.commitment.merkle import (
    MerkleTree,
    hash_leaf,
    hash_node,
    hash_record_leaf,
    hash_tensor_leaf,
    merkle_root,
    verify_path,
)


def leaves(n, salt=b""):
    return [hash_leaf(salt + i.to_bytes(4, "big")) for i in range(n)]


def test_known_answer_abc():
    b3 = lambda x: blake3.blake3(x).digest()
    la, lb, lc = (b3(b"\x00" + x) for x in (b"a", b"b", b"c"))
    expected = b3(b"\x01" + b3(b"\x01" + la + lb) + lc)
    hs = [hash_leaf(x) for x in (b"a", b"b", b"c")]
    assert merkle_root(hs) == expected
    assert expected.hex() == "6c62dd52a0971b7d00a7cead004e0c3f3c0766e3f5359a0f8297768d2b02d03c"
    assert MerkleTree(hs).root == expected


def test_promotion_not_duplication():
    a, b, c = leaves(3)
    assert merkle_root([a, b, c]) != merkle_root([a, b, c, c])
    assert merkle_root([a, b, c]) == hash_node(hash_node(a, b), c)


def test_leaf_node_domain_separation():
    l, r = leaves(2)
    assert hash_leaf(l + r) != hash_node(l, r)
    # A two-leaf root cannot be presented as a one-leaf tree over the concatenated children.
    assert merkle_root([hash_leaf(l + r)]) != merkle_root([l, r])


def test_leaf_hash_type_enforced():
    a, b = leaves(2)
    for bad in (bytearray(a), memoryview(a), a[:31]):
        with pytest.raises(ValueError):
            merkle_root([bad, b])
        with pytest.raises(ValueError):
            MerkleTree([bad, b])
        with pytest.raises(ValueError):
            MerkleTree([a, b]).update_leaf(0, bad)


def test_single_leaf_and_empty():
    (a,) = leaves(1)
    t = MerkleTree([a])
    assert t.root == a and t.path(0) == []
    assert verify_path(a, 0, 1, [], a)
    with pytest.raises(ValueError):
        merkle_root([])
    with pytest.raises(ValueError):
        MerkleTree([])


def test_streamed_leaf_equals_concatenated():
    t = torch.randn(33, 17)
    assert hash_tensor_leaf(TAG_WEIGHT, t) == hash_leaf(encode_tensor_leaf(TAG_WEIGHT, t))
    big = torch.randn(1 << 19)  # 2 MiB, multithreaded path
    assert hash_tensor_leaf(TAG_PRODUCT, big) == blake3.blake3(b"\x00" + encode_tensor_leaf(TAG_PRODUCT, big)).digest()
    rec = Record(*(torch.tensor(x, dtype=torch.int32) for x in ([1, 2], [2, 3], [0, 1])))
    assert hash_record_leaf(rec) == hash_leaf(encode_record(rec))


@pytest.mark.parametrize("n", range(1, 41))
def test_all_paths_verify(n):
    hs = leaves(n)
    t = MerkleTree(hs)
    assert t.root == merkle_root(hs) and t.n_leaves == n
    for i in range(n):
        assert verify_path(hs[i], i, n, t.path(i), t.root)


def _shape(i, n):
    """Sibling sides from leaf to root: True when the sibling is on the left."""
    sides, lo, hi = [], 0, n
    while hi - lo > 1:
        mid = lo + (1 << ((hi - lo - 1).bit_length() - 1))
        sides.append(i >= mid)
        lo, hi = (lo, mid) if i < mid else (mid, hi)
    return sides[::-1]


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13, 31, 40])
def test_tampering_fails(n):
    hs = leaves(n)
    t = MerkleTree(hs)
    wrong_size_failures = 0
    for i in range(n):
        p = t.path(i)
        for k in range(len(p)):
            bad = list(p)
            bad[k] = bytes([bad[k][0] ^ 1]) + bad[k][1:]
            assert not verify_path(hs[i], i, n, bad, t.root)
        assert not verify_path(hs[i], (i + 1) % n, n, p, t.root)
        # A wrong size fails unless it gives leaf i the same path shape; the root does not
        # bind n by itself, so the verifier takes n from config, never from the prover.
        for n2 in range(max(1, i + 1), n + 6):
            if n2 != n and verify_path(hs[i], i, n2, p, t.root):
                assert _shape(i, n2) == _shape(i, n)
        wrong_size_failures += sum(
            not verify_path(hs[i], i, n2, p, t.root) for n2 in (n - 1, n + 1) if n2 > i
        )
        assert not verify_path(hs[(i + 1) % n], i, n, p, t.root)
        assert not verify_path(hs[i], i, n, p + [hs[0]], t.root)
        if p:
            assert not verify_path(hs[i], i, n, p[:-1], t.root)
    assert wrong_size_failures > 0
    assert not verify_path(hs[0], n, n, t.path(0), t.root)
    assert not verify_path(hs[0], -1, n, t.path(0), t.root)


def test_update_leaf_matches_rebuild():
    rng = random.Random(0)
    for n in (1, 2, 3, 7, 40, 257):
        hs = leaves(n)
        t = MerkleTree(hs)
        for _ in range(30):
            i = rng.randrange(n)
            hs[i] = hash_leaf(rng.randbytes(8))
            assert t.update_leaf(i, hs[i]) == merkle_root(hs)
        for i in range(n):
            assert t.path(i) == MerkleTree(hs).path(i)


def test_step_tree_path_length():
    n = 7661
    t = MerkleTree(leaves(n))
    lengths = [len(t.path(i)) for i in range(n)]
    assert max(lengths) <= 13
