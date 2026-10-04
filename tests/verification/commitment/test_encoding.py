import struct

import pytest
import torch

from setup.records import TAG_RECORD
from verification.commitment.encoding import (
    DTYPE_CODES,
    TAG_MLP_RECORD,
    TAG_PRODUCT,
    TAG_WEIGHT,
    EncodingError,
    NonFiniteError,
    encode_tensor_leaf,
    tensor_leaf_header,
    tensor_leaf_payload,
)


def test_constants():
    assert (TAG_WEIGHT, TAG_PRODUCT, TAG_MLP_RECORD) == (2, 3, 4)
    assert DTYPE_CODES == {torch.float32: 1, torch.bfloat16: 2, torch.float16: 3, torch.int32: 0x10}


def test_header_layout():
    t = torch.zeros(3, 70000)
    assert tensor_leaf_header(TAG_WEIGHT, t) == bytes([2, 1, 2]) + struct.pack(">II", 3, 70000)
    assert encode_tensor_leaf(TAG_PRODUCT, torch.tensor([1.0])) == bytes([3, 1, 1, 0, 0, 0, 1]) + struct.pack("<f", 1.0)


def test_injective_over_shape_dtype_tag_with_same_raw_bytes():
    base = torch.arange(24, dtype=torch.float32)
    raw = bytes(tensor_leaf_payload(base))
    variants = [base.reshape(24), base.reshape(4, 6), base.reshape(6, 4), base.reshape(2, 3, 4)]
    variants.append(base.view(torch.int32))  # same bytes, different dtype
    variants.append(base.view(torch.float16).reshape(48)[:48])
    variants.append(base.view(torch.bfloat16))
    encs = set()
    for v in variants:
        assert bytes(tensor_leaf_payload(v.contiguous())) == raw
        for tag in (TAG_WEIGHT, TAG_PRODUCT, TAG_MLP_RECORD):
            encs.add(encode_tensor_leaf(tag, v.contiguous()))
    n_shape_dtype = len({(tuple(v.shape), v.dtype) for v in variants})
    assert len(encs) == 3 * n_shape_dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(dtype, bad):
    t = torch.tensor([1.0, bad, 2.0], dtype=dtype)
    with pytest.raises(NonFiniteError):
        tensor_leaf_payload(t)
    with pytest.raises(NonFiniteError):
        encode_tensor_leaf(TAG_PRODUCT, t)


def test_negative_zero_accepted_and_distinct():
    a = encode_tensor_leaf(TAG_PRODUCT, torch.tensor([0.0]))
    b = encode_tensor_leaf(TAG_PRODUCT, torch.tensor([-0.0]))
    assert a != b


def test_payload_is_zero_copy_and_little_endian():
    t = torch.tensor([1.0, -2.5], dtype=torch.bfloat16)
    mv = tensor_leaf_payload(t)
    assert bytes(mv) == t.view(torch.int16).numpy().astype("<i2").tobytes()
    big = torch.zeros(1000)
    mv = tensor_leaf_payload(big)
    big[0] = 1.0
    assert bytes(mv[:4]) == struct.pack("<f", 1.0)


def test_payload_rejects_noncontiguous_and_bad_dtype():
    with pytest.raises(EncodingError):
        tensor_leaf_payload(torch.zeros(3, 4).t())
    with pytest.raises(EncodingError):
        tensor_leaf_payload(torch.zeros(3, dtype=torch.float64))
    with pytest.raises(EncodingError):
        tensor_leaf_header(TAG_RECORD, torch.zeros(3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_finiteness_check_matches_isfinite_everywhere(dtype):
    """The aminmax check flags a NaN or ±Inf at every position, including the scalar tails of
    vectorized loops, and accepts finite tensors with extreme magnitudes, empty and 0-d ones."""
    big = torch.finfo(dtype).max
    for n in [*range(1, 70), 257, 4099]:
        base = torch.randn(n).to(dtype)
        base[0], base[-1] = big, -big
        tensor_leaf_payload(base)
        for i in range(n):
            for bad in (float("nan"), float("inf"), float("-inf")):
                t = base.clone()
                t[i] = bad
                assert not bool(torch.isfinite(t).all())
                with pytest.raises(NonFiniteError):
                    tensor_leaf_payload(t)
    for ok in (torch.zeros(0, dtype=dtype), torch.zeros(3, 0, dtype=dtype),
               torch.tensor(-0.0, dtype=dtype)):
        assert tensor_leaf_payload(ok).nbytes == ok.numel() * ok.element_size()
    with pytest.raises(NonFiniteError):
        tensor_leaf_payload(torch.tensor(float("nan"), dtype=dtype))
    # A NaN with a payload or a sign bit, not only the default quiet NaN.
    for bits in (0x7F800001, 0x7FC00000, 0x7FFFFFFF, -1, -0x00400000):
        t = torch.zeros(300, dtype=torch.int32)
        t[150] = bits
        with pytest.raises(NonFiniteError):
            tensor_leaf_payload(t.view(torch.float32))
