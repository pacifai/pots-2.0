import blake3
import numpy as np
import pytest
import torch

from verification.verifier.matmul_check.challenges import (
    TAG_LABEL,
    _entries,
    challenge_matrix,
    challenge_vector,
    encode_label,
)

H0 = bytes(32)
H1 = bytes(range(32))
GRID = 1 << 24


def _grid_index(r: torch.Tensor) -> np.ndarray:
    """Recover `a` from `r` as `(r·2^24 + 2^24 − 1) / 2`, in float64."""
    return (r.numpy().astype(np.float64) * GRID + GRID - 1) / 2


def test_label_bytes():
    assert TAG_LABEL == 0x10
    assert encode_label(1, 1) == b"\x10\x00\x00\x00\x01\x01"
    assert encode_label(0x01020304, 255) == b"\x10\x01\x02\x03\x04\xff"


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
    xof = blake3.blake3(encode_label(9, 3), key=H1).digest(length=4 * width)
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
    a = np.arange(GRID, dtype=np.uint32)
    low = np.random.default_rng(0).integers(0, 256, size=GRID, dtype=np.uint32)
    r = _entries((a << 8) | low)
    assert r.dtype == np.float32
    # a -> 2^24 - 1 - a negates the entry.
    assert np.array_equal(r[GRID - 1 - a.astype(np.int64)], -r)
    assert r.astype(np.float64).sum() == 0.0
    # The low byte is discarded.
    assert np.array_equal(_entries(a << 8), r)
    assert np.array_equal(_entries((a << 8) | np.uint32(0xFF)), r)


def test_blake3_keyed_official_vector():
    # BLAKE3 test_vectors.json: key "whats the Elvish word for friend", input_len 0,
    # first 32 bytes of keyed_hash.
    digest = blake3.blake3(b"", key=b"whats the Elvish word for friend").hexdigest()
    assert digest == "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26"


def test_known_answer():
    # h = 32 zero bytes, m = 1, j = 1, width = 4. Expected values are pinned literals.
    label = bytes([0x10, 0, 0, 0, 1, 1])
    xof = blake3.blake3(label, key=bytes(32)).digest(length=16)
    assert xof.hex() == "ab9e09ca470217be5b16ca83ca427a45"
    r = challenge_vector(bytes(32), 1, 1, 4)
    # Numerators 2a + 1 - 2^24 of the dyadic values r_i = numerator / 2^24.
    assert r.tolist() == [n / 2**24 for n in (9704253, 8138245, 496685, -7670651)]
    assert r.numpy().view(np.uint32).tolist() == [
        0x3F14133D,
        0x3EF85C0A,
        0x3CF285A0,
        0xBEEA16F6,
    ]


def test_label_injective():
    ms = (1, 2, 255, 256, 2**32 - 1)
    js = (1, 2, 255)
    labels = [encode_label(m, j) for m in ms for j in js]
    assert all(len(x) == 6 for x in labels)
    assert len(set(labels)) == len(labels)


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


def test_label_layout_and_injectivity():
    assert encode_label(1, 1) == bytes([0x10, 0, 0, 0, 1, 1])
    grid = [(m, j) for m in (1, 2, 255, 256, 257, 7113, 65536, 2**32 - 1) for j in (1, 2, 7, 255)]
    labels = {encode_label(m, j) for m, j in grid}
    assert len(labels) == len(grid)
    assert all(len(x) == 6 for x in labels)


@pytest.mark.parametrize("m,j", [(0, 1), (2**32, 1), (1, 0), (1, 256), (-1, 1)])
def test_label_range(m, j):
    with pytest.raises(ValueError):
        encode_label(m, j)
