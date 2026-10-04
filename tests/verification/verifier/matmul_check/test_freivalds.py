"""Check 5's numbers: the batched ``measure_products`` against one-at-a-time ``measure_product``."""

import math

import pytest
import torch

from verification.verifier.matmul_check.freivalds import measure_product, measure_products

H = bytes(range(32))
K = 7
EPS = 2.0 ** -24


def _batch(n_b, n, q, w, seed=0, forge=()):
    """``n_b`` honest products ``p[i] = a[i]·b[i]``, with members in ``forge`` perturbed."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_b, n, q, generator=g)
    b = torch.randn(n_b, q, w, generator=g)
    p = a @ b
    for i in forge:
        p[i, 0, 0] += 1.0
    return a, b, p


def _both(a, b, p, ms):
    batched = measure_products(a, b, p, h=H, ms=ms, k=K, eps_in=EPS, eps_acc=EPS)
    single = [measure_product(a[i], b[i], p[i], h=H, m=m, k=K, eps_in=EPS, eps_acc=EPS)
              for i, m in enumerate(ms)]
    return batched, single


def _close(x, y, rel=1e-4):
    """Finiteness equal exactly, finite values to ``rel``.

    A non-finite value may be ``inf`` one way and ``NaN`` the other (``inf − inf`` inside one
    kernel and not the other); check 5 rejects both alike, with the same message.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.isfinite(x) == math.isfinite(y)
    return x == pytest.approx(y, rel=rel, abs=0.0)


def _assert_match(batched, single):
    assert len(batched) == len(single)
    for mb, ms in zip(batched, single):
        for field in ("nu", "p_abs1", "kappa", "p_norm", "unit"):
            assert _close(getattr(mb, field), getattr(ms, field)), (field, mb, ms)
        for xs in ("residuals", "normalized"):
            assert len(getattr(mb, xs)) == K
            for x, y in zip(getattr(mb, xs), getattr(ms, xs)):
                assert _close(x, y), (xs, mb, ms)


@pytest.mark.parametrize("shape", [(12, 8, 12), (12, 12, 8), (100, 30, 70), (128, 64, 128),
                                   (128, 128, 64)])
def test_batched_numbers_match_single(shape):
    a, b, p = _batch(9, *shape, forge=(4,))
    ms = list(range(101, 110))
    batched, single = _both(a, b, p, ms)
    _assert_match(batched, single)
    # Each member keys its own challenges: the forged member stands out, the others don't.
    assert max(batched[4].normalized) > 100 * max(max(x.normalized) for x in batched[:4])


def test_tiny_shapes_differ_at_rounding_level_only():
    """Below about 8×8, CPU ``bmm`` and ``mm`` take different kernels and round differently.

    The honest residual is itself rounding, so the normalized residuals move by O(1) band
    units, not by a relative 1e-4. A forged entry still stands out the same way.
    """
    for shape in ((2, 2, 2), (6, 4, 5)):
        a, b, p = _batch(36, *shape, forge=(7,))
        batched, single = _both(a, b, p, list(range(1, 37)))
        for i, (mb, ms) in enumerate(zip(batched, single)):
            assert mb.p_norm == pytest.approx(ms.p_norm, rel=1e-6)
            dev = max(abs(x - y) for x, y in zip(mb.normalized, ms.normalized))
            assert dev < (1e-4 * max(ms.normalized) if i == 7 else 4.0)


def test_each_member_scales_its_own_norms():
    """A member with entries near 3e19 neither overflows nor changes another member's numbers."""
    a, b, p = _batch(3, 12, 8, 12)
    p[1] = p[1] + 1.0
    p[1, 0, 0] = 3e19  # its squares overflow fp32 without the per-member scale
    p[2] = 0.0  # ‖P‖_F = 0: band unit 0
    batched, single = _both(a, b, p, [1, 2, 3])
    _assert_match(batched, single)
    assert math.isfinite(batched[1].p_norm) and batched[1].p_norm == pytest.approx(3e19)
    alone = measure_products(a[:1], b[:1], p[:1], h=H, ms=[1], k=K, eps_in=EPS, eps_acc=EPS)
    _assert_match(alone, batched[:1])
    assert batched[2].unit == 0.0 and all(math.isinf(x) for x in batched[2].normalized)


def test_non_finite_member_stays_in_its_member():
    a, b, p = _batch(4, 12, 8, 12)
    a[2, 1, 1] = float("nan")
    b[3] *= 1e30  # A(B·r) overflows to inf
    a[3] *= 1e30
    batched, single = _both(a, b, p, [1, 2, 3, 4])
    _assert_match(batched, single)
    assert math.isnan(batched[2].nu) and all(math.isnan(x) for x in batched[2].residuals)
    assert not math.isfinite(batched[3].nu)
    for mp in batched[:2]:
        assert all(math.isfinite(x) for x in (mp.nu, mp.p_norm, *mp.residuals))


def test_batch_shape_mismatch_raises():
    a, b, p = _batch(3, 4, 4, 4)
    with pytest.raises(ValueError):
        measure_products(a, b, p, h=H, ms=[1, 2], k=K, eps_in=EPS, eps_acc=EPS)
