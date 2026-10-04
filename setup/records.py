"""The token record and its canonical byte form (P1b, S9b).

A record is what the dataset holds and what a step commits as a batch leaf. Header integers
are big-endian; the three int32 arrays are little-endian C-order raw bytes. The encoding is
injective and rigid: one byte string per record, with no free bits (S9b).
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

import torch

if sys.byteorder != "little":
    raise ImportError("canonical payloads are little-endian; big-endian hosts are unsupported")

TAG_RECORD = 0x01
INT32_CODE = 0x10  # the int32 dtype code, the same byte the tensor-leaf header uses

_U32_MAX = 2**32 - 1


class RecordError(ValueError):
    """A record breaks the token-record format, or bytes are not a canonical record."""


def _check_int32_1d(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        raise RecordError(f"{name} must be a torch.Tensor")
    if t.dtype != torch.int32 or t.dim() != 1:
        raise RecordError(f"{name} must be a 1-D int32 tensor, got {t.dtype} with {t.dim()} dims")


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
            raise RecordError("ids, targets and mask must have one common length")
        if not 1 <= n <= _U32_MAX:
            raise RecordError(f"record length {n} out of range")
        # A mask value other than 0/1 would be a free field (S9b).
        if not bool(((self.mask == 0) | (self.mask == 1)).all()):
            raise RecordError("mask entries must be 0 or 1")

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
    if t.device.type != "cpu":
        raise RecordError("tensor must be on CPU")
    if not t.is_contiguous():
        raise RecordError("tensor must be contiguous")
    return t.detach().numpy().tobytes()


def encode_record(rec: Record) -> bytes:
    """`0x01 ‖ 0x10 ‖ ℓ(4 BE) ‖ ids ‖ targets ‖ mask` (P1b)."""
    head = struct.pack(">BBI", TAG_RECORD, INT32_CODE, len(rec))
    return head + _payload_bytes(rec.ids) + _payload_bytes(rec.targets) + _payload_bytes(rec.mask)


def decode_record(b: bytes) -> Record:
    """Inverse of `encode_record`. Rejects any byte string that is not a canonical record."""
    b = bytes(b)
    if len(b) < 6:
        raise RecordError("record too short")
    tag, code, n = struct.unpack(">BBI", b[:6])
    if tag != TAG_RECORD or code != INT32_CODE:
        raise RecordError(f"bad record header tag={tag:#x} dtype={code:#x}")
    if len(b) != 6 + 12 * n:
        raise RecordError(f"record length field {n} does not match {len(b)} bytes")
    arr = torch.frombuffer(bytearray(b[6:]), dtype=torch.int32)
    return Record(ids=arr[:n].clone(), targets=arr[n : 2 * n].clone(), mask=arr[2 * n :].clone())
