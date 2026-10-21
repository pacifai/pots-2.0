"""Check 5's measuring core (``measure_core.h``), on the host build.

``measure_core_host.c`` runs the core's work decomposition (32 partials per dot, then the tree)
one partial after another, so the pinned order of MEASURE_KERNEL_PLAN_DRAFT.md §2.1a is tested
on any machine. The core's rows are compared with torch's within the rounding bound of a
float64 evaluation of the same formula, and with themselves bit for bit across layouts, dtypes
and table contexts."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.freivalds import (
    _measure_product_reference,
    finish_rows,
    measure_product_row,
)
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

ROOT = Path(__file__).resolve().parents[4]
H = bytes(32)
U = 2.0 ** -24


@pytest.fixture(scope="module")
def hostlib(tmp_path_factory):
    out = tmp_path_factory.mktemp("host") / "measure_core_host.so"
    d = ROOT / "tests/verification/verifier/matmul_check"
    cmd = ["cc", "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-shared", "-fPIC",
           f"-I{ROOT / 'verification/verifier/matmul_check'}", str(d / "measure_core_host.c"),
           "-o", str(out), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0 and not proc.stderr, proc.stderr
    lib = ctypes.CDLL(str(out))
    lib.pots_host_sizeof_mview.restype = lib.pots_host_sizeof_mprod.restype = ctypes.c_size_t
    for f in (lib.pots_host_offsets_mview, lib.pots_host_offsets_mprod):
        f.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        f.restype = ctypes.c_size_t
    lib.pots_host_measure.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.pots_host_measure.restype = ctypes.c_int
    lib.pots_host_check.argtypes = [ctypes.c_void_p]
    lib.pots_host_check.restype = ctypes.c_int
    return lib


@pytest.fixture(scope="module")
def host_measure(hostlib):
    def run(table, *, raw=False):
        rows = mt.empty_rows(table)
        err = hostlib.pots_host_measure(table.ptr, table.n, rows.ctypes.data)
        if raw:
            return err, rows
        assert err == 0
        return rows
    return run


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)


def make(n, q, w, k, *, dtype=torch.float32, biased=False, guard=True, seed=0, scale=1.0):
    """A random product as ``MeasureInput``, ``P`` a perturbed ``A·B`` so residuals are small."""
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(n, q, generator=g) * scale).to(dtype)
    b = (torch.randn(q, w, generator=g) * scale).to(dtype)
    bias = (torch.randn(w, generator=g) * scale).to(dtype) if biased else None
    p = a.float() @ b.float()
    if biased:
        p = p + bias.float()
    p = (p + 1e-3 * torch.randn(n, w, generator=g)).to(dtype)
    r = challenge_matrix(H, 1, k, w).contiguous()
    return MeasureInput(a, b, p, r, bias, guard)


def f64(t):
    return t.detach().double().numpy()


def exact(mi):
    """The row's formula in float64 on the exactly widened inputs, with the error-bound
    scales: ``(row, rel_len, S)``; ``S`` is the absolute-value residual bound vector's norm per
    challenge, ``rel_len`` the longest fp32 dot chain feeding a field."""
    a, b, p, r = f64(mi.a), f64(mi.b), f64(mi.p), f64(mi.r)
    bias = None if mi.bias is None else f64(mi.bias)
    n, q = a.shape
    w = b.shape[1]
    nu_v = np.abs(a) @ (np.abs(b) @ np.ones(w))
    if bias is not None:
        nu_v = nu_v + np.abs(bias).sum()
    d = a @ (b @ r) - p @ r
    sv = np.abs(a) @ (np.abs(b) @ np.abs(r)) + np.abs(p) @ np.abs(r)
    if bias is not None:
        d = d + (bias @ r)[None, :]
        sv = sv + (np.abs(bias) @ np.abs(r))[None, :]
    row = np.array([np.linalg.norm(nu_v), np.linalg.norm(np.abs(p).sum(1)), np.linalg.norm(p),
                    (a ** 2 @ (b ** 2).sum(1)).sum(), 0.0 if bias is None else (bias ** 2).sum(),
                    *np.linalg.norm(d, axis=0)])
    if not mi.guard:
        row[:2] = np.nan
    return row, 2 * w + q + 3, np.linalg.norm(sv, axis=0)


def gamma(length):
    return length * U / (1 - length * U)


def torch_row(mi):
    row = measure_product_row(mi.a.float(), mi.b.float(), mi.p.float(), h=H, m=1,
                              k=mi.r.shape[1], guard=mi.guard,
                              bias=None if mi.bias is None else mi.bias.float())
    return row.numpy()


def check_within_bound(row, mi, ex=None, fp32_norms=False):
    """Each field of ``row`` against the float64 evaluation, within the rounding bound
    (``fp32_norms``: torch's norms of ``P`` and ``b`` are fp32 reductions, not float64)."""
    ex_row, length, s = ex if ex is not None else exact(mi)
    g2 = 2 * gamma(length)
    for i in (0, 1, 3):  # sums of non-negative terms: relative bound
        if np.isnan(ex_row[i]):
            assert np.isnan(row[i])
        else:
            assert abs(row[i] - ex_row[i]) <= g2 * abs(ex_row[i]) + 1e-300, (i, row[i], ex_row[i])
    tol = 2 * gamma(length + len(mi.p) + mi.p.shape[1]) if fp32_norms else 1e-12
    for i in (2, 4):  # float64 sums in the core
        assert abs(row[i] - ex_row[i]) <= tol * abs(ex_row[i]) + 1e-300
    err = np.abs(row[5:] - ex_row[5:])
    assert np.all(err <= g2 * s + 1e-12 * ex_row[5:] + 1e-300), (err, g2 * s)


SHAPES = [(1, 2, 3), (1, 7, 40), (5, 2, 31), (6, 33, 32), (4, 64, 33), (9, 17, 65),
          (3, 50, 1000), (13, 5, 7), (2, 3, 1)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("guard", [True, False])
def test_agrees_with_torch_within_the_rounding_bound(host_measure, dtype, biased, guard):
    for s, (n, q, w) in enumerate(SHAPES):
        mi = make(n, q, w, 3, dtype=dtype, biased=biased, guard=guard, seed=s)
        row = host_measure(pack([mi]))[0]
        ex = exact(mi)
        check_within_bound(row, mi, ex)
        check_within_bound(torch_row(mi), mi, ex, fp32_norms=True)  # the same bound holds for torch's fp32 order
        assert row.shape == (8,)


def test_torch_reference_numbers_are_in_the_same_bound(host_measure):
    mi = make(7, 40, 50, 4, biased=True, seed=3)
    ref = _measure_product_reference(mi.a, mi.b, mi.p, h=H, m=1, k=4, eps_in=1e-3, eps_acc=1e-6,
                                     bias=mi.bias)
    ex_row, length, s = exact(mi)
    row = host_measure(pack([mi]))[0]
    g2 = 2 * gamma(length)
    assert abs(ref.nu - row[0]) <= 2 * g2 * row[0]
    assert abs(ref.p_norm - row[2]) <= 1e-5 * row[2]
    assert np.all(np.abs(np.array(ref.residuals) - row[5:]) <= 2 * g2 * s)


def test_layout_matches_the_core(hostlib):
    out = (ctypes.c_size_t * 16)()
    assert hostlib.pots_host_offsets_mview(out) == len(mt.MVIEW_FIELDS)
    assert list(out)[:7] == [getattr(mt.MView, f).offset for f in mt.MVIEW_FIELDS]
    assert hostlib.pots_host_offsets_mprod(out) == len(mt.MPROD_FIELDS)
    assert list(out)[:7] == [getattr(mt.MProd, f).offset for f in mt.MPROD_FIELDS]
    assert hostlib.pots_host_sizeof_mview() == ctypes.sizeof(mt.MView) == 48
    assert hostlib.pots_host_sizeof_mprod() == ctypes.sizeof(mt.MProd) == 4 * 48 + 16


# ---- bit identity ----

def test_same_product_same_bits_alone_in_tables_and_twice(host_measure):
    mis = [make(5, 20, 45, 3, biased=bool(i % 2), guard=bool(i % 3), seed=i) for i in range(6)]
    alone = [host_measure(pack([m]))[0] for m in mis]
    again = [host_measure(pack([m]))[0] for m in mis]
    for x, y in zip(alone, again):
        assert np.array_equal(bits(x), bits(y))
    rng = np.random.default_rng(1)
    for _ in range(4):
        order = rng.permutation(len(mis))
        rows = host_measure(pack([mis[i] for i in order]))
        for pos, i in enumerate(order):
            assert np.array_equal(bits(rows[pos]), bits(alone[i]))


@pytest.mark.parametrize("biased", [False, True])
def test_strided_views_match_contiguous_copies(host_measure, biased):
    mi = make(6, 19, 37, 3, biased=biased, seed=4)
    ref = host_measure(pack([mi]))[0]
    g = torch.Generator().manual_seed(9)

    def embed(t):  # a slice of a bigger tensor holding the same logical values
        big = torch.randn(t.shape[0] + 3, t.shape[1] * 2 + 5, generator=g).to(t.dtype)
        big[1:1 + t.shape[0], 3:3 + 2 * t.shape[1]:2] = t
        return big[1:1 + t.shape[0], 3:3 + 2 * t.shape[1]:2]

    def transposed(t):
        return t.t().contiguous().t()

    for variant in (embed, transposed):
        v = MeasureInput(variant(mi.a), variant(mi.b), variant(mi.p), mi.r, mi.bias, mi.guard)
        assert not v.a.is_contiguous() and torch.equal(v.a, mi.a)
        assert np.array_equal(bits(host_measure(pack([v]))[0]), bits(ref))
    both = MeasureInput(transposed(mi.a), embed(mi.b), transposed(mi.p), mi.r, mi.bias, mi.guard)
    assert np.array_equal(bits(host_measure(pack([both]))[0]), bits(ref))


@pytest.mark.parametrize("biased", [False, True])
def test_bf16_storage_equals_widened_fp32_storage(host_measure, biased):
    mi = make(6, 19, 37, 3, dtype=torch.bfloat16, biased=biased, seed=5)
    wide = MeasureInput(mi.a.float(), mi.b.float(), mi.p.float(), mi.r,
                        None if mi.bias is None else mi.bias.float(), mi.guard)
    mixed = MeasureInput(mi.a, mi.b.float(), mi.p, mi.r, mi.bias, mi.guard)
    ref = host_measure(pack([wide]))[0]
    assert np.array_equal(bits(host_measure(pack([mi]))[0]), bits(ref))
    assert np.array_equal(bits(host_measure(pack([mixed]))[0]), bits(ref))


def test_guard_off_only_blanks_the_two_guard_fields(host_measure):
    for biased in (False, True):
        on = host_measure(pack([make(5, 21, 36, 3, biased=biased, seed=6)]))[0]
        off = host_measure(pack([make(5, 21, 36, 3, biased=biased, guard=False, seed=6)]))[0]
        assert np.isnan(off[:2]).all() and not np.isnan(on[:2]).any()
        assert np.array_equal(bits(off[2:]), bits(on[2:]))


# ---- edge values ----

def with_values(n=4, q=6, w=40, **kw):
    mi = make(n, q, w, 2, seed=7, **kw)
    return MeasureInput(mi.a.clone(), mi.b.clone(), mi.p.clone(), mi.r, mi.bias, mi.guard)


@pytest.mark.parametrize("guard", [True, False])
def test_nan_in_a(host_measure, guard):
    mi = with_values(guard=guard)
    mi.a[0, 0] = float("nan")
    row = host_measure(pack([mi]))[0]
    assert np.isnan(row[0]) and np.isnan(row[3]) and np.isnan(row[5:]).all()
    assert np.isfinite(row[2]) and (guard is False or np.isfinite(row[1]))


def test_inf_in_p(host_measure):
    mi = with_values()
    mi.p[1, 3] = float("inf")
    row = host_measure(pack([mi]))[0]
    assert row[1] == np.inf and row[2] == np.inf and np.isfinite(row[0]) and np.isfinite(row[3])
    assert (row[5:] == np.inf).all()


def test_zero_p_and_zero_product(host_measure):
    mi = with_values()
    mi.p.zero_()
    row = host_measure(pack([mi]))[0]
    assert row[2] == 0 and row[1] == 0 and row[0] > 0 and row[5] > 0
    z = with_values()
    z.a.zero_(), z.b.zero_(), z.p.zero_()
    assert (host_measure(pack([z]))[0] == 0).all()
    z.guard = False
    row = host_measure(pack([z]))[0]
    assert np.isnan(row[:2]).all() and (row[2:] == 0).all()


def test_negative_zero_is_zero(host_measure):
    z = with_values(biased=True)
    for t in (z.a, z.b, z.p, z.bias):
        t.zero_()
    pos = host_measure(pack([z]))[0]
    for t in (z.a, z.b, z.p, z.bias):
        t.copy_(torch.full_like(t, -0.0))
    neg = host_measure(pack([z]))[0]
    assert (pos == 0).all() and np.array_equal(bits(pos), bits(neg))


def test_entries_near_2e19_overflow_u2_as_torch_does(host_measure):
    mi = with_values()
    mi.a.fill_(2e19)
    mi.b.fill_(1.0)
    mi.p.fill_(2e19)
    row = host_measure(pack([mi]))[0]
    t = torch_row(mi)
    assert row[3] == np.inf and t[3] == np.inf
    assert abs(row[2] - t[2]) <= 1e-5 * t[2]


def test_subnormal_inputs(host_measure):
    mi = with_values()
    for t in (mi.a, mi.b, mi.p):
        t.mul_(1e-39)
    assert (mi.a.abs() < 1.2e-38).all()
    row = host_measure(pack([mi]))[0]
    ex = exact(mi)[0]
    assert np.isfinite(row).all()
    assert np.allclose(row, ex, rtol=1e-3, atol=1e-37)


# ---- the downstream use and refusals ----

def test_finish_rows_normalized_residuals_agree_with_torch(host_measure):
    mi = make(8, 40, 60, 4, biased=True, seed=8)
    row = host_measure(pack([mi]))
    ex_row, length, s = exact(mi)
    kw = dict(q=np.array([41]), n=np.array([8]), eps_in=2.0 ** -8, eps_acc=2.0 ** -24,
              biased=np.array([True]))
    mine = finish_rows(row, **kw)
    tor = finish_rows(torch_row(mi)[None, :], **kw)
    bound = 4 * gamma(length) * s / mine.unit[0]
    assert np.all(np.abs(mine.normalized[0] - tor.normalized[0]) <= 1.001 * bound + 1e-9)
    assert abs(mine.u[0] - tor.u[0]) <= 1e-5 * tor.u[0]


def test_pack_refusals():
    ok = make(3, 4, 5, 2)

    def refuse(mi, text, table=None):
        with pytest.raises(ValueError, match=text):
            pack(table(mi) if table else [mi])

    def swap(**kw):
        d = dict(a=ok.a, b=ok.b, p=ok.p, r=ok.r, bias=None)
        d.update(kw)
        return MeasureInput(**d)

    with pytest.raises(ValueError, match="empty"):
        pack([])
    for dt in (torch.float16, torch.float64, torch.int32):
        refuse(swap(a=ok.a.to(dt)), "product 0: a has dtype")
        refuse(swap(bias=torch.zeros(5, dtype=dt)), "bias has dtype")
        refuse(swap(p=ok.p.to(dt)), "p has dtype")
    refuse(swap(b=torch.zeros(5, 5)), "do not make a product")
    refuse(swap(p=torch.zeros(3, 6)), "do not make a product")
    refuse(swap(a=torch.zeros(3, 4, 1)), "not 2-D")
    refuse(swap(bias=torch.zeros(4)), "bias has shape")
    refuse(swap(bias=torch.zeros(1, 5)), "bias has shape")
    refuse(swap(r=torch.zeros(4, 2)), "r has shape")
    refuse(swap(r=torch.zeros(5, 0)), "r has shape")
    refuse(swap(r=torch.zeros(5, 2, dtype=torch.float64)), "r is not a float32")
    refuse(swap(r=torch.zeros(2, 5).t()), "not contiguous")
    meta = lambda t: torch.empty(t.shape, dtype=t.dtype, device="meta")  # noqa: E731
    refuse(swap(a=meta(ok.a)), "several devices")
    other = MeasureInput(meta(ok.a), meta(ok.b), meta(ok.p), meta(ok.r))
    with pytest.raises(ValueError, match="product 1: device meta differs"):
        pack([ok, other])
    with pytest.raises(ValueError, match="product 2: k = 3 differs"):
        pack([ok, ok, make(3, 4, 5, 3)])


def test_core_check_refusals(hostlib, host_measure):
    def table():
        return pack([make(3, 4, 5, 2, biased=True)])

    assert hostlib.pots_host_check(table().ptr) == mt.ERR_OK
    cases = [
        (mt.ERR_DTYPE, lambda e: setattr(e.a, "dtype", 3)),
        (mt.ERR_DTYPE, lambda e: setattr(e.p, "dtype", 0)),
        (mt.ERR_DTYPE, lambda e: setattr(e.bias, "dtype", 7)),
        (mt.ERR_K, lambda e: setattr(e, "k", 0)),
        (mt.ERR_K, lambda e: setattr(e, "r_addr", 0)),
        (mt.ERR_SHAPE, lambda e: setattr(e.b, "rows", 5)),
        (mt.ERR_SHAPE, lambda e: setattr(e.p, "rows", 2)),
        (mt.ERR_SHAPE, lambda e: setattr(e.p, "cols", 4)),
        (mt.ERR_SHAPE, lambda e: setattr(e.bias, "cols", 4)),
        (mt.ERR_SHAPE, lambda e: setattr(e.bias, "rows", 2)),
        (mt.ERR_SHAPE, lambda e: (setattr(e.a, "rows", 0), setattr(e.p, "rows", 0))),
    ]
    for code, mutate in cases:
        t = table()
        mutate(t.array[0])
        assert hostlib.pots_host_check(t.ptr) == code
        err, rows = host_measure(t, raw=True)
        assert err == code
    # an unbiased product ignores its (zeroed) bias view
    t = pack([make(3, 4, 5, 2)])
    assert hostlib.pots_host_check(t.ptr) == mt.ERR_OK
    # the first refusal wins and no row is written
    t = pack([make(3, 4, 5, 2), make(3, 4, 5, 2)])
    t.array[1].k = 0
    rows = mt.empty_rows(t)
    rows[:] = 123.0
    assert hostlib.pots_host_measure(t.ptr, 2, rows.ctypes.data) == mt.ERR_K
    assert (rows == 123.0).all()
