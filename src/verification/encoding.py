"""Canonical byte encoding of committed objects (S9b, S9c, S9d, P1b, P8a).

Header integers are big-endian; payloads are little-endian C-order raw bytes. Every encoder
here is injective and rigid: one byte string per object, with no free bits (S9b).
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

import torch

if sys.byteorder != "little":
    raise ImportError("canonical payloads are little-endian; big-endian hosts are unsupported")

TAG_RECORD = 0x01
TAG_WEIGHT = 0x02
TAG_PRODUCT = 0x03
TAG_MLP_RECORD = 0x04
TAG_LABEL = 0x10

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


def _check_int32_1d(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        raise EncodingError(f"{name} must be a torch.Tensor")
    if t.dtype != torch.int32 or t.dim() != 1:
        raise EncodingError(f"{name} must be a 1-D int32 tensor, got {t.dtype} with {t.dim()} dims")


@dataclass(frozen=True, eq=False)
class Record:
    """A tokenized dataset record (P1b): input ids, next-token targets, loss mask.

    Only encoding-level rules are checked here; `ℓ ≤ n` and `targets[:-1] == ids[1:]` are
    enforced by the dataset builder (B5).
    """

    ids: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("ids", "targets", "mask"):
            _check_int32_1d(name, getattr(self, name))
        n = self.ids.numel()
        if self.targets.numel() != n or self.mask.numel() != n:
            raise EncodingError("ids, targets and mask must have one common length")
        if not 1 <= n <= _U32_MAX:
            raise EncodingError(f"record length {n} out of range")
        # A mask value other than 0/1 would be a free field (S9b).
        if not bool(((self.mask == 0) | (self.mask == 1)).all()):
            raise EncodingError("mask entries must be 0 or 1")

    def __len__(self) -> int:
        return self.ids.numel()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        return (
            torch.equal(self.ids, other.ids)
            and torch.equal(self.targets, other.targets)
            and torch.equal(self.mask, other.mask)
        )

    __hash__ = None  # type: ignore[assignment]


def _payload_bytes(t: torch.Tensor) -> bytes:
    return bytes(tensor_leaf_payload(t))


def encode_record(rec: Record) -> bytes:
    """`0x01 ‖ 0x10 ‖ ℓ(4 BE) ‖ ids ‖ targets ‖ mask` (P1b)."""
    head = struct.pack(">BBI", TAG_RECORD, DTYPE_CODES[torch.int32], len(rec))
    return head + _payload_bytes(rec.ids) + _payload_bytes(rec.targets) + _payload_bytes(rec.mask)


def decode_record(b: bytes) -> Record:
    """Inverse of `encode_record`. Rejects any byte string that is not a canonical record."""
    b = bytes(b)
    if len(b) < 6:
        raise EncodingError("record too short")
    tag, code, n = struct.unpack(">BBI", b[:6])
    if tag != TAG_RECORD or code != DTYPE_CODES[torch.int32]:
        raise EncodingError(f"bad record header tag={tag:#x} dtype={code:#x}")
    if len(b) != 6 + 12 * n:
        raise EncodingError(f"record length field {n} does not match {len(b)} bytes")
    arr = torch.frombuffer(bytearray(b[6:]), dtype=torch.int32)
    return Record(ids=arr[:n].clone(), targets=arr[n : 2 * n].clone(), mask=arr[2 * n :].clone())


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


def encode_label(m: int, j: int) -> bytes:
    """Challenge label `0x10 ‖ m(4 BE) ‖ j(1)` (P8a); `m` and `j` are 1-based."""
    if not 1 <= m <= _U32_MAX:
        raise EncodingError(f"label m={m} out of range")
    if not 1 <= j <= 255:
        raise EncodingError(f"label j={j} out of range")
    return struct.pack(">BIB", TAG_LABEL, m, j)
