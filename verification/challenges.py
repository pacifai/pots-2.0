"""Challenge vectors `r(m, j)` for check 5 (spec §5, P8a, P8b)."""

from __future__ import annotations

import blake3
import numpy as np
import torch

from .encoding import encode_label as _label  # P8a

_GRID = 1 << 24


def _validate(h: bytes, m: int, width: int) -> None:
    if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
        raise ValueError("h must be 32 bytes")
    if not 1 <= m < 1 << 32:
        raise ValueError(f"m must be in [1, 2^32), got {m}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")


def _validate_j(j: int) -> None:
    if not 1 <= j <= 255:
        raise ValueError(f"j must be in [1, 255], got {j}")


def challenge_vector(h: bytes, m: int, j: int, width: int) -> torch.Tensor:
    """Challenge vector `r(m, j)`, float32 of shape `[width]` (P8b).

    Entry i reads XOF bytes 4i..4i+3 as a little-endian uint32 `w`, keeps `a = w >> 8`, and
    sets `r_i = (2a + 1 - 2^24) / 2^24`. The numerator is an odd integer below 2^24 in
    magnitude and the divisor a power of two, so every step is exact in float32.
    """
    _validate(h, m, width)
    _validate_j(j)
    xof = blake3.blake3(_label(m, j), key=bytes(h)).digest(length=4 * width)
    return torch.from_numpy(_entries(np.frombuffer(xof, dtype="<u4")))


def _entries(words: np.ndarray) -> np.ndarray:
    """Map uint32 words to float32 grid entries `(2(w >> 8) + 1 - 2^24) / 2^24` (P8b)."""
    a = (words.astype(np.uint32) >> 8).astype(np.int64)
    numerator = (2 * a + 1 - _GRID).astype(np.float32)
    return numerator / np.float32(_GRID)


def challenge_matrix(h: bytes, m: int, k: int, width: int) -> torch.Tensor:
    """The k challenge vectors of product m as columns, float32 of shape `[width, k]`.

    Column `j - 1` is `challenge_vector(h, m, j, width)`, so check 5 evaluates all k tests
    at once as `A @ (B @ R)` against `P @ R`.
    """
    _validate(h, m, width)
    if not 1 <= k <= 255:
        raise ValueError(f"k must be in [1, 255], got {k}")
    return torch.stack([challenge_vector(h, m, j, width) for j in range(1, k + 1)], dim=1)
