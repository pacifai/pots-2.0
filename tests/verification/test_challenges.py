import blake3
import numpy as np
import pytest
import torch

from src.verification.challenges import _label, challenge_matrix, challenge_vector

H0 = bytes(32)
H1 = bytes(range(32))
GRID = 1 << 24


def _grid_index(r: torch.Tensor) -> np.ndarray:
    """Recover `a` from `r` as `(r·2^24 + 2^24 − 1) / 2`, in float64."""
    return (r.numpy().astype(np.float64) * GRID + GRID - 1) / 2


def test_label_bytes():
    assert _label(1, 1) == b"\x10\x00\x00\x00\x01\x01"
    assert _label(0x01020304, 255) == b"\x10\x01\x02\x03\x04\xff"


def test_shape_dtype():
    r = challenge_vector(H0, 1, 1, 17)
    assert r.shape == (17,) and r.dtype == torch.float32
    R = challenge_matrix(H0, 3, 7, 33)
    assert R.shape == (33, 7) and R.dtype == torch.float32


def test_entries_on_exact_grid():
    a = _grid_index(challenge_vector(H1, 5, 2, 100_000))
    assert np.all(a == np.floor(a))
    assert a.min() >= 0 and a.max() < GRID


def test_matches_float64_reference():
    width = 50_000
    r = challenge_vector(H1, 9, 3, width)
    xof = blake3.blake3(_label(9, 3), key=H1).digest(length=4 * width)
    w = np.frombuffer(xof, dtype="<u4").astype(np.float64)
    ref = (2 * np.floor(w / 256) + 1 - GRID) / GRID
    assert np.array_equal(r.numpy().astype(np.float64), ref)


def test_deterministic():
    assert torch.equal(challenge_vector(H1, 2, 4, 1000), challenge_vector(H1, 2, 4, 1000))
    assert torch.equal(challenge_matrix(H1, 2, 7, 1000), challenge_matrix(H1, 2, 7, 1000))


def test_inputs_change_vector():
    base = challenge_vector(H0, 1, 1, 64)
    assert not torch.equal(base, challenge_vector(H1, 1, 1, 64))
    assert not torch.equal(base, challenge_vector(H0, 2, 1, 64))
    assert not torch.equal(base, challenge_vector(H0, 1, 2, 64))


def test_width_prefix_property():
    # A property of the XOF stream, not a protocol requirement: under the same (h, m, j),
    # a shorter vector is a prefix of a longer one.
    assert torch.equal(challenge_vector(H1, 4, 1, 10)[:5], challenge_vector(H1, 4, 1, 5))


def test_matrix_columns():
    R = challenge_matrix(H1, 11, 7, 257)
    for j in range(1, 8):
        assert torch.equal(R[:, j - 1], challenge_vector(H1, 11, j, 257))


def test_mean_and_variance():
    r = challenge_matrix(H1, 1, 8, 131_072).numpy().astype(np.float64).ravel()
    assert r.size >= 1_000_000
    # Std of the mean is about 5.8e-4 and of the variance estimate about 3e-4.
    assert abs(r.mean()) < 3e-3
    assert abs(r.var() - 1 / 3) < 2e-3
    assert r.min() >= -1 + 2**-24 and r.max() <= 1 - 2**-24


def test_grid_symmetric():
    a = np.arange(GRID, dtype=np.int64)
    r = ((2 * a + 1 - GRID).astype(np.float32) / np.float32(GRID)).astype(np.float64)
    assert np.array_equal(r[GRID - 1 - a], -r)
    assert r.sum() == 0.0


def test_known_answer():
    # Derived independently with a raw keyed blake3 call and integer math.
    label = bytes([0x10, 0, 0, 0, 1, 1])
    xof = blake3.blake3(label, key=bytes(32)).digest(length=16)
    expected = []
    for i in range(4):
        w = int.from_bytes(xof[4 * i : 4 * i + 4], "little")
        expected.append((2 * (w >> 8) + 1 - 2**24) / 2**24)
    assert challenge_vector(bytes(32), 1, 1, 4).tolist() == expected


@pytest.mark.parametrize(
    "h, m, j, width",
    [
        (bytes(31), 1, 1, 4),
        (bytes(33), 1, 1, 4),
        (H0, 0, 1, 4),
        (H0, 1 << 32, 1, 4),
        (H0, 1, 0, 4),
        (H0, 1, 256, 4),
        (H0, 1, 1, 0),
    ],
)
def test_validation(h, m, j, width):
    with pytest.raises(ValueError):
        challenge_vector(h, m, j, width)


def test_matrix_validation():
    for k in (0, 256):
        with pytest.raises(ValueError):
            challenge_matrix(H0, 1, k, 4)
