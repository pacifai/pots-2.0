import pytest

from verification.commitment import native
from verification.commitment.merkle import MerkleTree, hash_leaf, merkle_root

SIZES = [*range(1, 71), 127, 128, 129, 1000, 4097]


def leaves(n):
    return [hash_leaf(i.to_bytes(4, "big")) for i in range(n)]


@pytest.mark.parametrize("n", SIZES)
def test_tree_native_matches_python(n):
    hs = leaves(n)
    nat = MerkleTree(hs)
    with native.disabled():
        py = MerkleTree(hs)
        py_root = merkle_root(hs)
    assert nat._buf == py._buf
    assert nat.root == py.root == merkle_root(hs) == py_root
    idx = range(n) if n <= 200 else range(0, n, 37)
    for i in idx:
        assert nat.path(i) == py.path(i)
    assert nat.path(n - 1) == py.path(n - 1)


@pytest.mark.parametrize("n", [1, 2, 3, 70, 129, 1000])
def test_update_leaf_on_native_tree_equals_rebuild(n):
    hs = leaves(n)
    t = MerkleTree(hs)
    for i in sorted({0, n // 2, n - 1}):
        new = hash_leaf(b"new" + bytes([i % 256]))
        hs[i] = new
        assert t.update_leaf(i, new) == MerkleTree(hs).root
        assert t._buf == MerkleTree(hs)._buf


@pytest.mark.parametrize("bad", [[], [b"x" * 31], [bytes(32), b"y" * 33],
                                 [bytearray(32)], ["a" * 32], [None]])
def test_errors_with_native(bad):
    for f in (MerkleTree, merkle_root):
        with pytest.raises(ValueError) as a:
            f(bad)
        with native.disabled(), pytest.raises(ValueError) as b:
            f(bad)
        assert str(a.value) == str(b.value)
