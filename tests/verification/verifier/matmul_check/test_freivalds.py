"""Check 5's numbers.

The batched ``measure_products`` against one-at-a-time ``measure_product``, and both against
their ``_reference`` versions, bit for bit, on adversarial inputs.
"""

import math
import struct

import pytest
import torch

from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check.freivalds import (
    _MIN_ENTRY,
    MeasureScratch,
    _measure_product_reference,
    _measure_products_reference,
    _norm,
    _safe_norm,
    measure_product,
    measure_products,
)

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


# ---- the fast path against the reference, bit for bit -----------------------------------

FIELDS = ("nu", "p_abs1", "kappa", "residuals", "p_norm", "unit", "normalized")


def _bits(mp):
    """Every field's float64 bytes, so ``==`` means bit-identical (NaN included)."""
    out = []
    for f in FIELDS:
        v = getattr(mp, f)
        out.append(tuple(struct.pack("<d", x) for x in v) if isinstance(v, tuple)
                   else struct.pack("<d", v))
    return tuple(out)


def _same_single(a, b, p, m=3, scratch=None):
    fast = measure_product(a, b, p, h=H, m=m, k=K, eps_in=EPS, eps_acc=EPS, scratch=scratch)
    ref = _measure_product_reference(a, b, p, h=H, m=m, k=K, eps_in=EPS, eps_acc=EPS)
    assert _bits(fast) == _bits(ref), (fast, ref)
    return fast


def _same_batch(a, b, p, scratch=None):
    ms = list(range(11, 11 + a.shape[0]))
    fast = measure_products(a, b, p, h=H, ms=ms, k=K, eps_in=EPS, eps_acc=EPS, scratch=scratch)
    ref = _measure_products_reference(a, b, p, h=H, ms=ms, k=K, eps_in=EPS, eps_acc=EPS)
    assert [_bits(x) for x in fast] == [_bits(x) for x in ref]
    return fast


def _honest(n, q, w, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, q, generator=g) * scale
    b = torch.randn(q, w, generator=g)
    return a, b, a @ b


def _cases():
    """``(name, a, b, p)``: honest, forged and adversarial single products."""
    a, b, p = _honest(40, 24, 56)
    yield "honest", a, b, p
    yield "forged", a, b, p + torch.where(torch.rand(p.shape) < 0.01, 1.0, 0.0)
    yield "zero p", a, b, torch.zeros_like(p)
    yield "zero row", a, b, torch.cat([torch.zeros(1, 56), p[1:]])
    yield "zero a", torch.zeros_like(a), b, p
    for big in (1.8e19, 2e19, 3e19, 1e30, 3.4e38):
        q_ = p.clone()
        q_[3, 5] = big
        yield f"p entry {big:g}", a, b, q_
        yield f"a entry {big:g}", a.clone().index_put_((torch.tensor([2]), torch.tensor([1])),
                                                      torch.tensor(big)), b, p
    for tiny in (1e-20, 2.0 ** -63, 2.0 ** -64, 1e-30, 1e-40, 1e-45):
        q_ = p.clone()
        q_[7, 1] = tiny
        yield f"p entry {tiny:g}", a, b, q_
        yield f"all tiny {tiny:g}", a * tiny, b, p * tiny
    for scale in (2.0 ** -60, 2.0 ** 40, 2.0 ** 55, 2.0 ** 58, 2.0 ** 60):
        yield f"scaled {scale:g}", a, b, p * scale
    for bad in (float("nan"), float("inf"), -float("inf")):
        for which in range(3):
            ops = [a.clone(), b.clone(), p.clone()]
            ops[which][1, 1] = bad
            yield f"{bad} in operand {which}", *ops
    yield "a·b overflows", a * 1e20, b * 1e20, p
    yield "single row", *_honest(1, 24, 56)
    yield "single column", *_honest(40, 24, 1)
    yield "1×2·2×1", *_honest(1, 2, 1)
    yield "negative zeros", a, b, torch.where(p.abs() < 0.5, -0.0, p)
    at, bt = a.t().contiguous().t(), b.t().contiguous().t()
    yield "transposed a and b", at, bt, p
    yield "transposed p", a, b, p.t().contiguous().t()
    big = torch.zeros(p.numel() + 1)
    big[1:] = p.reshape(-1)
    yield "unaligned p view", a, b, big[1:].view(p.shape)
    yield "strided a", torch.randn(40, 48)[:, ::2], b, p


@pytest.mark.parametrize("case", list(_cases()), ids=lambda c: c[0])
def test_single_matches_reference_bit_for_bit(case):
    _, a, b, p = case
    _same_single(a, b, p)


def test_scratch_reused_across_shapes_matches_reference():
    """One scratch through growing, shrinking and transposed shapes, as check 5 uses it."""
    sc = MeasureScratch()
    for name, a, b, p in _cases():
        _same_single(a, b, p, scratch=sc)
    for shape in ((200, 64, 300), (8, 4, 8), (64, 200, 30)):
        _same_single(*_honest(*shape), scratch=sc)
        a, b, p = _batch(5, *shape, forge=(2,))
        _same_batch(a, b, p, scratch=sc)


def _batch_cases():
    a, b, p = _batch(9, 32, 16, 24, forge=(4,))
    yield "honest", a, b, p
    p1 = p.clone()
    p1[3, 0, 0] = 3e19
    yield "one huge member", a, b, p1
    p2 = p.clone()
    p2[5] = 0.0
    yield "one zero member", a, b, p2
    p3 = p.clone()
    p3[6, 2, 2] = 1e-40
    yield "one tiny entry", a, b, p3
    a4 = a.clone()
    a4[1, 1, 1] = float("nan")
    yield "one nan member", a4, b, p
    yield "a·b overflows in one member", a * torch.tensor([1e20] + [1.0] * 8)[:, None, None], \
        b * torch.tensor([1e20] + [1.0] * 8)[:, None, None], p
    yield "single-row members", *_batch(4, 1, 16, 24)
    yield "single-column members", *_batch(4, 32, 16, 1)
    yield "transposed members", a, b.transpose(1, 2).contiguous().transpose(1, 2), p


@pytest.mark.parametrize("case", list(_batch_cases()), ids=lambda c: c[0])
def test_batch_matches_reference_bit_for_bit(case):
    _, a, b, p = case
    _same_batch(a, b, p)


@pytest.mark.parametrize("dim", [None, 0, 1])
def test_norm_matches_safe_norm_across_scales(dim):
    """Random vectors from 2^-70 to 2^70: inside the bounds and outside them, same bits."""
    g = torch.Generator().manual_seed(1)
    for e in range(-70, 71, 3):
        for spread in (1.0, 2.0 ** 20, 2.0 ** 40):
            x = torch.randn(33, 50, generator=g) * 2.0 ** e
            x[0, :] *= spread  # widen the range between the smallest and largest entry
            want = _safe_norm(x, dim)
            got = _norm(x, dim, x.abs())
            assert got.numpy().tobytes() == want.numpy().tobytes(), (e, spread)


@pytest.mark.parametrize("hi", [1.0, 2.0 ** 30, 1.5 * 2.0 ** 40, 2.0 ** -20])
def test_norm_at_the_small_entry_bound(hi):
    """Entries at ``2^-63·max(1, hi)`` take the plain norm; one binade below, the scaled one.

    Both give ``_safe_norm``'s bits. Below the bound a square or a scaled entry is subnormal,
    where the plain norm and the scaled one may round differently.
    """
    bound = _MIN_ENTRY * max(1.0, hi)
    for lo in (bound, bound / 2, bound * 0.75, bound * 1.5):
        x = torch.full((1000,), lo)
        x[0] = hi
        x[1:500] *= torch.linspace(1.0, 1.9, 499)
        assert _norm(x, None, x.abs()).numpy().tobytes() == \
            _safe_norm(x, None).numpy().tobytes(), lo


def test_norm_at_the_sum_bound():
    """Sums of squares near fp32's top: on both sides of ``n·hi² = 2^124``."""
    for n, hi in ((1, 2.0 ** 62), (4, 2.0 ** 61), (4, 2.0 ** 61 * 1.01), (1000, 2.0 ** 57),
                  (1000, 2.0 ** 58), (1000, 2.0 ** 59), (1 << 16, 2.0 ** 54),
                  (1 << 16, 2.0 ** 55)):
        x = torch.full((n,), hi)
        x[n // 2:] *= 0.999
        want = _safe_norm(x).numpy().tobytes()
        assert _norm(x, None, x.abs()).numpy().tobytes() == want, (n, hi)


def test_abs_into_scratch_keeps_layout_and_matmul_bits():
    """``|x|`` in the buffer has a fresh ``abs``'s strides, and a matmul on it the same bits."""
    sc = MeasureScratch()
    ones = sc.ones(300)
    for x in (torch.randn(200, 300), torch.randn(300, 200).t(), torch.randn(4, 200, 300)):
        fresh = x.abs()
        mine = sc.abs(x)
        assert mine.stride() == fresh.stride() and mine.data_ptr() % 64 == 0
        assert torch.equal(mine, fresh)
        assert (mine @ ones).numpy().tobytes() == (fresh @ ones).numpy().tobytes()


def test_zero_free_products_skip_the_scaled_copy_of_p(monkeypatch):
    """``‖P‖_F`` of a zero-free ``P`` takes the plain norm; a zero entry falls back.

    Small norms may fall back too: an honest residual often has exact zeros.
    """
    shapes = []
    real = freivalds._scaled_norm
    monkeypatch.setattr(freivalds, "_scaled_norm",
                        lambda x, *a, **kw: shapes.append(tuple(x.shape)) or real(x, *a, **kw))
    a, b, p = _honest(40, 24, 56)
    measure_product(a, b, p, h=H, m=1, k=K, eps_in=EPS, eps_acc=EPS)
    measure_products(*_batch(3, 32, 16, 24), h=H, ms=[1, 2, 3], k=K, eps_in=EPS, eps_acc=EPS)
    assert (40, 56) not in shapes and (3, 32 * 24) not in shapes
    p[0, 0] = 0.0
    measure_product(a, b, p, h=H, m=1, k=K, eps_in=EPS, eps_acc=EPS)
    assert (40, 56) in shapes
