"""Canonical byte encoding of tensor leaves (S9b, S9c, S9d).

Header integers are big-endian; payloads are little-endian C-order raw bytes. Every encoder
here is injective and rigid: one byte string per object, with no free bits (S9b). Record
leaves use the token-record format of `setup/records.py` (tag `0x01`).
"""

from __future__ import annotations

import struct
import sys

import torch

if sys.byteorder != "little":
    raise ImportError("canonical payloads are little-endian; big-endian hosts are unsupported")

TAG_WEIGHT = 0x02
TAG_PRODUCT = 0x03
TAG_MLP_RECORD = 0x04

TENSOR_TAGS = frozenset({TAG_WEIGHT, TAG_PRODUCT, TAG_MLP_RECORD})

DTYPE_CODES: dict[torch.dtype, int] = {
    torch.float32: 0x01,
    torch.bfloat16: 0x02,
    torch.float16: 0x03,
    torch.int32: 0x10,
}
CODE_DTYPES: dict[int, torch.dtype] = {v: k for k, v in DTYPE_CODES.items()}
FLOAT_DTYPES = frozenset({torch.float32, torch.bfloat16, torch.float16})

_U32_MAX = 2**32 - 1


class NonFiniteError(ValueError):
    """A float tensor bound for a commitment holds NaN or Inf (S9d)."""


class EncodingError(ValueError):
    """An object cannot be canonically encoded, or bytes are not a canonical encoding."""


def tensor_leaf_header(tag: int, t: torch.Tensor) -> bytes:
    """`tag(1) ‖ dtype(1) ‖ ndim(1) ‖ dim_0(4 BE) … dim_{ndim−1}(4 BE)` (S9c)."""
    if tag not in TENSOR_TAGS:
        raise EncodingError(f"tag {tag:#x} is not a tensor-leaf tag")
    if t.dtype not in DTYPE_CODES:
        raise EncodingError(f"dtype {t.dtype} has no canonical code")
    shape = tuple(t.shape)
    if len(shape) > 255:
        raise EncodingError("ndim exceeds 255")
    if any(d > _U32_MAX for d in shape):
        raise EncodingError("dimension exceeds uint32")
    return struct.pack(f">BBB{len(shape)}I", tag, DTYPE_CODES[t.dtype], len(shape), *shape)


def tensor_leaf_payload(t: torch.Tensor) -> memoryview:
    """Zero-copy byte view of a contiguous CPU tensor's little-endian C-order data.

    Float tensors with NaN or Inf raise `NonFiniteError`; `-0.0` passes (S9d).
    The view aliases `t`'s storage, so the caller must not mutate `t` until hashing is done.
    The finiteness check allocates a temporary bool tensor of `numel` bytes.
    """
    if t.dtype not in DTYPE_CODES:
        raise EncodingError(f"dtype {t.dtype} has no canonical code")
    if t.device.type != "cpu":
        raise EncodingError("tensor must be on CPU")
    if not t.is_contiguous():
        raise EncodingError("tensor must be contiguous")
    t = t.detach()
    if t.dtype in FLOAT_DTYPES and not bool(torch.isfinite(t).all()):
        raise NonFiniteError("float tensor contains NaN or Inf")
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)  # numpy has no bfloat16; same bytes
    return memoryview(t.reshape(-1).numpy()).cast("B")


def encode_tensor_leaf(tag: int, t: torch.Tensor) -> bytes:
    """Full leaf bytes as one copy. For small tensors and tests; hashing streams instead."""
    return tensor_leaf_header(tag, t) + bytes(tensor_leaf_payload(t))
