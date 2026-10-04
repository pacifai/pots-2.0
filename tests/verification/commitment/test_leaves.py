"""Leaf hashing against the declared computation: each leaf's shape and dtype are checked
against `C` before its bytes are hashed."""

import pytest
import torch

from verification.commitment import merkle
from verification.commitment.encoding import (
    TAG_PRODUCT,
    TAG_WEIGHT,
    NonFiniteError,
    encode_tensor_leaf,
)
from verification.commitment.leaves import (
    commit_leaves,
    leaf_hash,
    leaf_hashes,
    leaf_hashes_of,
)
from verification.commitment.merkle import MerkleTree, hash_leaf
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.prover.step import StepOutput, prove_step
from verification.transcript.errors import (
    LeafDtypeError,
    LeafShapeError,
    StoreMutationError,
    TranscriptFormatError,
)

ETA = 1e-2


@pytest.fixture
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)


@pytest.fixture
def data(c):
    return synthetic_dataset(c.widths, 12, seed=0)


@pytest.fixture
def parallel(monkeypatch):
    """Bulk hashing with one leaf per task on four workers, so the MLP's small leaves are
    hashed out of order across threads."""
    monkeypatch.setattr(merkle, "_CHUNK_BYTES", 1)
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


def _step(c, data) -> StepOutput:
    w0 = init_weights(c.widths, seed=0)
    return prove_step(c, c.build_model(), w0, data[4:8])


def test_wrong_shape_rejected(c, data):
    step = _step(c, data)
    with pytest.raises(LeafShapeError):
        leaf_hash(c, c.product_index(1), step.products[0].t().contiguous())


def test_wrong_dtype_rejected(c, data):
    step = _step(c, data)
    name = c.weight_names[0]
    with pytest.raises(LeafDtypeError):
        leaf_hash(c, c.w_t_index(name), step.w_t[name].to(torch.float16))
    assert issubclass(LeafDtypeError, TranscriptFormatError)
    assert issubclass(LeafShapeError, TranscriptFormatError)
    assert issubclass(StoreMutationError, TranscriptFormatError)


class _HalfWeights(MLPComputation):
    @property
    def weight_dtype(self) -> torch.dtype:
        return torch.float16


def test_same_payload_other_declaration(c, data):
    # One payload, read under a different shape or dtype: it raises against C, and where a
    # declaration admits it, the header makes the hash differ.
    step = _step(c, data)
    i = c.product_index(1)
    p = step.products[0]
    with pytest.raises(LeafShapeError):
        leaf_hash(c, i, p.reshape(p.shape[1], p.shape[0]))
    with pytest.raises(LeafDtypeError):
        leaf_hash(c, i, p.view(torch.int32))
    assert hash_leaf(encode_tensor_leaf(TAG_PRODUCT, p.reshape(-1))) != leaf_hash(c, i, p)
    # A half-precision declaration rejects the step's fp32 weight. Its fp16 bytes, read as
    # bf16, carry another dtype code and hash differently.
    half = _HalfWeights(c.widths, n_s=c.n_s, eta=ETA)
    w = c.w_t_index(c.weight_names[0])
    with pytest.raises(LeafDtypeError):
        leaf_hash(half, w, step.w_t[c.weight_names[0]])
    w16 = step.w_t[c.weight_names[0]].to(torch.float16)
    assert leaf_hash(half, w, w16) != hash_leaf(encode_tensor_leaf(TAG_WEIGHT,
                                                                   w16.view(torch.bfloat16)))


class _ListReader:
    def __init__(self, leaves):
        self.leaves = leaves

    def leaf(self, index):
        return self.leaves[index]


def test_bulk_hashing_equals_leaf_by_leaf(c, data, parallel):
    leaves = _step(c, data).leaves()
    one_by_one = [leaf_hash(c, i, x) for i, x in enumerate(leaves)]
    assert leaf_hashes_of(c, enumerate(leaves)) == one_by_one
    assert leaf_hashes(c, _ListReader(leaves)) == one_by_one
    tree = commit_leaves(c, leaves)
    assert [tree.leaf(i) for i in range(c.n_leaves)] == one_by_one
    assert tree.root == MerkleTree(one_by_one).root


def test_bulk_hashing_raises_the_first_bad_leaf(c, data, parallel):
    """A non-finite leaf raises as `leaf_hash` does, and with two bad leaves the one first in
    leaf order is reported, whichever kind of error it is."""
    leaves = list(_step(c, data).leaves())
    p1, p2 = c.product_index(1), c.product_index(3)
    nan = leaves[p1].clone()
    nan.view(-1)[5] = float("nan")
    with pytest.raises(NonFiniteError) as one:
        leaf_hash(c, p1, nan)
    bad = list(leaves)
    bad[p1] = nan
    with pytest.raises(NonFiniteError) as bulk:
        commit_leaves(c, bad)
    assert str(bulk.value) == str(one.value)
    bad[p2] = leaves[p2].t().contiguous()  # a later shape error: the NaN still comes first
    with pytest.raises(NonFiniteError):
        leaf_hashes(c, _ListReader(bad))
    bad[p1], bad[p2] = leaves[p1].t().contiguous(), leaves[p2].clone().fill_(float("inf"))
    with pytest.raises(LeafShapeError, match=f"leaf {p1} "):
        leaf_hashes(c, _ListReader(bad))
