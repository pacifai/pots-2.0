"""Leaf hashing against the declared computation: each leaf's shape and dtype are checked
against `C` before its bytes are hashed."""

import pytest
import torch

from verification.commitment.encoding import TAG_PRODUCT, TAG_WEIGHT, encode_tensor_leaf
from verification.commitment.leaves import leaf_hash
from verification.commitment.merkle import hash_leaf
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
