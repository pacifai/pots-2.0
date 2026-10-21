"""Check 5's measuring core against a bit-exact emulator of the pinned arithmetic.

The emulator below is written from MEASURE_KERNEL_PLAN_DRAFT.md §2.1a (as restated in the FS36
brief) only: 32 fmaf partials per fp32 dot, the halving tree, fp64 sums of the same shape. fmaf
and sqrt come from the C library through ctypes; fp32 additions are numpy fp32. The core's rows
must equal the emulator's bit for bit. The ``hostlib`` and ``host_measure`` fixtures are imported
from ``test_measure_core.py``."""

import ctypes
import math

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits,
    host_measure,
    hostlib,
)
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

_libm = ctypes.CDLL(None)
_libm.fmaf.argtypes = [ctypes.c_float] * 3
_libm.fmaf.restype = ctypes.c_float
_libm.sqrt.argtypes = [ctypes.c_double]
_libm.sqrt.restype = ctypes.c_double

F32 = np.float32


def fmaf(x, y, z):
    return _libm.fmaf(x, y, z)


def add32(x, y):
    return float(F32(x) + F32(y))


def mul32(x, y):
    return float(F32(x) * F32(y))


def tree(s, add):
    s = list(s)
    for off in (16, 8, 4, 2, 1):
        for i in range(off):
            s[i] = add(s[i], s[i + off])
    return s[0]


def dot32(xs, ys):
    """fp32 dot of equal-length sequences of fp32 values: 32 fma partials, then the tree."""
    s = [0.0] * 32
    for c, (x, y) in enumerate(zip(xs, ys)):
        s[c % 32] = fmaf(x, y, s[c % 32])
    return tree(s, add32)


def dot_naive(xs, ys):
    """A single left-to-right fma accumulator, to show the tests can tell the order apart."""
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = fmaf(x, y, acc)
    return acc


def sum64(xs):
    s = [0.0] * 32
    for c, x in enumerate(xs):
        s[c % 32] = s[c % 32] + x
    return tree(s, lambda a, b: a + b)


def ident(v):
    return list(v)


def fabs(v):
    return [abs(x) for x in v]


def sq(v):
    return [mul32(x, x) for x in v]


def emulate(a, b, p, r, bias, guard, dot=dot32):
    """The row ``[nu, |P|1 norm, P_F, U2, |b|^2, res...]`` of one product; ``a, b, p`` are
    logical fp32 numpy matrices, ``r`` fp32 ``[w, k]``, ``bias`` fp32 ``[w]`` or None."""
    n, q = a.shape
    w = b.shape[1]
    k = r.shape[1]
    a = [[float(x) for x in row] for row in a]
    b = [[float(x) for x in row] for row in b]
    p = [[float(x) for x in row] for row in p]
    r = [[float(x) for x in row] for row in r]
    ones_w = [1.0] * w
    ones_q = [1.0] * q
    rcol = [[r[c][j] for c in range(w)] for j in range(k)]
    t = [[dot(b[l], rcol[j]) for l in range(q)] for j in range(k)]
    b1 = [dot(fabs(b[l]), ones_w) for l in range(q)] if guard else None
    b2 = [dot(sq(b[l]), ones_w) for l in range(q)]
    s = [[dot(a[i], t[j]) for i in range(n)] for j in range(k)]
    if guard:
        nu = [dot(fabs(a[i]), b1) for i in range(n)]
        if bias is not None:
            beta1 = dot(fabs([float(x) for x in bias]), ones_w)
            nu = [add32(x, beta1) for x in nu]
    u = [dot(sq(a[i]), b2) for i in range(n)]
    pi = [[dot(p[i], rcol[j]) for i in range(n)] for j in range(k)]
    p1 = [dot(fabs(p[i]), ones_w) for i in range(n)] if guard else None
    pp = [sum64([x * x for x in p[i]]) for i in range(n)]
    d = []
    for j in range(k):
        if bias is not None:
            beta = dot([float(x) for x in bias], rcol[j])
            d.append([F32(F32(s[j][i] + 0.0) + F32(beta)) - F32(pi[j][i]) for i in range(n)])
        else:
            d.append([F32(s[j][i]) - F32(pi[j][i]) for i in range(n)])
    d = [[float(x) for x in col] for col in d]

    def norm(xs):
        return _libm.sqrt(sum64([x * x for x in xs]))

    nan = float("nan")
    row = [norm(nu) if guard else nan, norm(p1) if guard else nan,
           _libm.sqrt(sum64(pp)), sum64(u),
           0.0 if bias is None else sum64([float(x) * float(x) for x in bias])]
    row += [norm(d[j]) for j in range(k)]
    return np.array(row, dtype=np.float64)


def same_bits(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    m = ~np.isnan(x)
    return np.array_equal(x[m].view(np.uint64), y[m].view(np.uint64))


def operand(g, shape, dtype, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(dtype)


def product(n, q, w, k, dtype, *, biased, guard, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a, b = operand(g, (n, q), dtype, scale), operand(g, (q, w), dtype, scale)
    bias = operand(g, (w,), dtype, scale) if biased else None
    p = a.float() @ b.float()
    if biased:
        p = p + bias.float()
    p = (p + 1e-2 * torch.randn(n, w, generator=g)).to(dtype)
    r = torch.randn(w, k, generator=g).contiguous()
    return MeasureInput(a, b, p, r, bias, guard)


def emu(mi, dot=dot32):
    return emulate(mi.a.float().numpy(), mi.b.float().numpy(), mi.p.float().numpy(),
                   mi.r.numpy(), None if mi.bias is None else mi.bias.float().numpy(),
                   mi.guard, dot)


def core(host_measure, mis):
    return host_measure(pack(mis))


def assert_matches(host_measure, mis, label=""):
    rows = core(host_measure, mis)
    for idx, mi in enumerate(mis):
        want = emu(mi)
        assert same_bits(rows[idx], want), (label, idx, rows[idx], want)


LENGTHS = (1, 31, 32, 33, 47, 64)


def length_cases(dtype, biased, guard):
    """For each L in LENGTHS: once as the column count w (n, q small), once as q (w small),
    once as the row count n."""
    mis = []
    seed = 0
    for length in LENGTHS:
        seed += 1
        mis.append(product(3, 5, length, 2, dtype, biased=biased, guard=guard, seed=seed))
        seed += 1
        mis.append(product(4, length, 7, 2, dtype, biased=biased, guard=guard, seed=seed))
        seed += 1
        mis.append(product(length if length <= 48 else 48, 3, 4, 2, dtype, biased=biased,
                           guard=guard, seed=seed))
    return mis


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("guard", [True, False])
def test_core_equals_emulator_over_lengths(host_measure, dtype, biased, guard):
    """For each of L in {1,31,32,33,47,64}, once as the column count w (n, q small) and once as
    q (w small), the core's rows equal the emulator's bit for bit."""
    assert_matches(host_measure, length_cases(dtype, biased, guard))


def test_core_equals_emulator_on_strided_and_transposed_views(host_measure):
    """Transposed views, column slices and row slices of larger tensors give the emulator's
    rows (computed from the logical matrices)."""
    g = torch.Generator().manual_seed(7)
    for dtype in (torch.float32, torch.bfloat16):
        big_a = operand(g, (20, 40), dtype)
        big_b = operand(g, (50, 45), dtype)
        a = big_a[2:11, 3:38:2]            # 9 x 18, row and column strides
        b = big_b[4:40:2, 1:44:3]          # 18 x 15
        p = operand(g, (15, 9), dtype).t()  # transposed view, 9 x 15
        bias = operand(g, (30,), dtype)[::2]  # stride 2, length 15
        r = torch.randn(15, 2, generator=g)
        assert not a.is_contiguous() and not p.is_contiguous()
        for bi in (None, bias):
            assert_matches(host_measure, [MeasureInput(a, b, p, r, bi, True)])
        at = operand(g, (33, 9), dtype).t()    # transposed A: 9 x 33
        bt = operand(g, (15, 33), dtype).t()   # transposed B: 33 x 15
        assert_matches(host_measure, [MeasureInput(at, bt, p, r, bias, False)])


def hard_cases():
    """Mixed magnitudes in one row so the addition order changes the fp32 result."""
    cases = []
    g = torch.Generator().manual_seed(11)
    for seed, (n, q, w) in enumerate([(4, 40, 70), (3, 64, 33), (5, 33, 64), (2, 100, 100)]):
        mags = torch.tensor([1e8, 1.0, -1e8, 1e-3, 3e7, 1.0, 0.5, -3e7])
        a = (torch.randn(n, q, generator=g) * mags[torch.randint(0, 8, (n, q), generator=g)])
        b = (torch.randn(q, w, generator=g) * mags[torch.randint(0, 8, (q, w), generator=g)])
        p = (a @ b + torch.randn(n, w, generator=g) * 1e3)
        p = p * mags[torch.randint(0, 8, (n, w), generator=g)].sqrt()
        bias = torch.randn(w, generator=g) * mags[torch.randint(0, 8, (w,), generator=g)]
        r = torch.randn(w, 2, generator=g)
        cases.append(MeasureInput(a, b, p, r, bias if seed % 2 else None, True))
    # a row with 1e8 and ones: partial order visible in plain sums
    a = torch.ones(1, 64)
    b = torch.ones(64, 40)
    b[0, :] = 1e8
    b[32, :] = -1e8
    cases.append(MeasureInput(a, b, a @ b, torch.randn(40, 2, generator=g), None, True))
    return cases


def test_core_equals_emulator_when_order_matters(host_measure):
    """Mixed-magnitude products: the core equals the emulator, and a naive left-to-right
    accumulator differs from the emulator on at least one case (so the order is observable)."""
    cases = hard_cases()
    assert_matches(host_measure, cases)
    differs = [not same_bits(emu(mi, dot_naive), emu(mi)) for mi in cases]
    assert any(differs), "naive order never differs: the cases cannot tell the order apart"


def special_value_cases():
    """NaN, Inf, -0.0 and subnormals, in each of A, B, P and the bias; zeros; negative zeros;
    subnormal inputs; bf16 storage."""
    g = torch.Generator().manual_seed(5)
    n, q, w, k = 5, 6, 7, 2
    base = lambda: product(n, q, w, k, torch.float32, biased=True, guard=True, seed=3)  # noqa: E731
    sub = 1e-40  # fp32 subnormal
    cases = []
    for field in ("a", "b", "p", "bias"):
        for val in (float("nan"), float("inf"), -float("inf"), -0.0, sub, -sub):
            mi = base()
            t = getattr(mi, field).clone()
            t.view(-1)[torch.randint(0, t.numel(), (1,), generator=g)] = val
            setattr(mi, field, t)
            cases.append(mi)
            mi2 = base()
            mi2.guard = False
            t = getattr(mi2, field).clone()
            t.view(-1)[0] = val
            setattr(mi2, field, t)
            cases.append(mi2)
    zeros = MeasureInput(torch.zeros(n, q), torch.zeros(q, w), torch.zeros(n, w),
                         torch.randn(w, k, generator=g), None, True)
    negz = MeasureInput(-torch.zeros(n, q), torch.ones(q, w), -torch.zeros(n, w),
                        torch.randn(w, k, generator=g), -torch.zeros(w), True)
    subs = MeasureInput(torch.full((n, q), sub), torch.full((q, w), sub), torch.full((n, w), sub),
                        torch.randn(w, k, generator=g), torch.full((w,), sub), True)
    cases += [zeros, negz, subs]
    # the same with bf16 storage (subnormal bf16 values, inf, nan)
    mi = base()
    cases.append(MeasureInput(mi.a.bfloat16(), mi.b.bfloat16(), mi.p.bfloat16(), mi.r,
                              mi.bias.bfloat16(), True))
    sb = torch.full((n, q), 1e-40).bfloat16()
    cases.append(MeasureInput(sb, torch.ones(q, w).bfloat16(), torch.zeros(n, w).bfloat16(),
                              mi.r, None, True))
    return cases


def test_core_equals_emulator_on_special_values(host_measure):
    """NaN, Inf, -0.0 and subnormals, in each of A, B, P and the bias."""
    cases = special_value_cases()
    assert_matches(host_measure, cases)
    # NaN positions must really occur in some of these rows
    rows = core(host_measure, cases)
    assert np.isnan(rows).any()


def halving_fp32_case():
    v = torch.zeros(32, 1)
    v[0], v[1], v[17] = 2.0 ** 24, 1.0, 1.0
    a = torch.ones(1, 32)
    return v, MeasureInput(a, v, torch.zeros(1, 1), torch.ones(1, 1), None, True)


def halving_fp64_case():
    a = torch.zeros(32, 1)
    a[0] = 2.0 ** 27
    for i in (1, 17, 3, 19):
        a[i] = 1.0
    return a, MeasureInput(a, torch.ones(1, 1), torch.zeros(32, 1), torch.ones(1, 1), None, True)


def test_halving_tree_is_the_pinned_one_fp32(host_measure):
    """n=1, w=1, q=32: nu = the tree over the 32 partials |B[l]|. The partials are 2^24 at 0
    and 1 at 1 and 17: the halving tree gives 2^24+2 exactly, a left-to-right sum gives 2^24."""
    v, mi = halving_fp32_case()
    row = core(host_measure, [mi])[0]
    partials = [float(x) for x in v.view(-1)]
    ltr = 0.0
    for x in partials:
        ltr = add32(ltr, x)
    assert ltr == 2.0 ** 24
    assert tree(partials, add32) == 2.0 ** 24 + 2
    assert row[0] == 2.0 ** 24 + 2
    assert same_bits(row, emu(mi))


def test_halving_tree_is_the_pinned_one_fp64(host_measure):
    """n=32, q=w=1: U^2 = the fp64 tree over 32 row terms u_i = a_i^2. 2^54 at row 0 and 1 at
    rows 1, 17, 3, 19: the tree adds the ones first (2^54+4 exactly); left to right loses all."""
    a, mi = halving_fp64_case()
    row = core(host_measure, [mi])[0]
    u = [float(x) ** 2 for x in a.view(-1)]
    assert math.fsum(u) == 2.0 ** 54 + 4
    ltr = 0.0
    for x in u:
        ltr += x
    assert ltr == 2.0 ** 54
    assert row[3] == 2.0 ** 54 + 4
    assert same_bits(row, emu(mi))


def fs36_cases():
    """Every input FS36 checks bit for bit, as ``(label, MeasureInput)``: the lengths over both
    dtypes, biased or not, guard on or off; the order-sensitive cases; the special values; the
    two halving-tree cases. The kernel tests and their host twin take their list from here."""
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for biased in (False, True):
            for guard in (True, False):
                for i, mi in enumerate(length_cases(dtype, biased, guard)):
                    cases.append((f"length {str(dtype)[6:]} biased={biased} guard={guard} #{i}", mi))
    cases += [(f"hard #{i}", mi) for i, mi in enumerate(hard_cases())]
    cases += [(f"special #{i}", mi) for i, mi in enumerate(special_value_cases())]
    cases += [("halving fp32", halving_fp32_case()[1]), ("halving fp64", halving_fp64_case()[1])]
    return cases
