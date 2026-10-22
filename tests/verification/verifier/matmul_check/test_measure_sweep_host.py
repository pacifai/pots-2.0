"""FS37b's one-sweep-per-row measuring kernel, tested without a GPU.

``pots_host_measure_sweep`` runs the kernel's work in loops on the Mac (lane accumulators, columns
tile by tile, chunks of challenge columns, the shared per-element update functions of
``measure_phases.h``). Every test holds its rows to ``pots_host_measure`` (and, for the shared
tables, ``pots_host_measure_levels``) bit for bit, NaN equal to NaN where special values are
planted. ``test_measure_tiles_cuda.py`` holds the real kernel to the same numbers."""

import ctypes
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.shared_tables import (
    DTYPES,
    EDITS,
    fwd_and_grad,
    member_stack,
    mixed_table,
    shared_special,
    three_forward,
    two_sided,
)
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    SHAPES,
    bits,
    host_measure,
    hostlib,
    make,
)
from tests.verification.verifier.matmul_check.test_measure_core_spec import fs36_cases, same_bits
from tests.verification.verifier.matmul_check.test_measure_kernel_host import levels  # noqa: F401
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

KC = getattr(mk, "K_CHUNK", 16)  # the default challenge columns per chunk


@pytest.fixture(scope="module")
def sweep(hostlib):
    """``sweep(mis, reverse=, tile=, kc=, share=)``: the rows by ``pots_host_measure_sweep`` on
    ``mk.read_plan`` (``share``) or ``mk.plain_plan`` (the unshared three levels)."""
    hostlib.pots_host_measure_sweep.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_int]
    hostlib.pots_host_measure_sweep.restype = ctypes.c_int

    def run(mis, *, reverse=False, tile=64, kc=16, share=True):
        table = pack(mis)
        prods = np.frombuffer(table.array, dtype=mk.MPROD_DTYPE).copy()
        scratch = np.zeros(mk.scratch_bytes(prods) + 8, dtype=np.uint8)
        base = (scratch.ctypes.data + 7) & ~7
        works = mk.works_array(prods, base)
        plan = mk.read_plan(prods) if share else mk.plain_plan(prods)
        rows = mt.empty_rows(table)
        err = hostlib.pots_host_measure_sweep(
            prods.ctypes.data, len(prods), works.ctypes.data, plan.reads.ctypes.data,
            plan.level_off.ctypes.data, plan.n_levels, plan.entries.ctypes.data,
            rows.ctypes.data, tile, kc, int(reverse))
        assert err == 0
        return rows

    return run


def agree(got, want, labels=None):
    """Same shape, NaN where NaN, the same bits elsewhere; names the first differing product."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape
    if same_bits(got, want):
        return
    for i in range(len(got)):
        if not same_bits(got[i], want[i]):
            raise AssertionError(f"product {i} {labels[i] if labels else ''}: "
                                 f"{got[i]} != {want[i]}")
    raise AssertionError("rows differ")


# ---- FS36's inputs and FS38b's shared tables ----

COMBOS = [(64, 16, False), (32, 1, True), (128, KC, False), (32, 4, True)]


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("share", [True, False])
def test_every_fs36_case_as_one_table_per_challenge_width(sweep, host_measure, k, share):
    cases = [mi for _, mi in fs36_cases() if mi.r.shape[1] == k]
    assert len(cases) >= (150 if k == 2 else 1)
    want = host_measure(pack(cases))
    for tile, kc, reverse in COMBOS:
        agree(sweep(cases, tile=tile, kc=kc, reverse=reverse, share=share), want)


def test_fs36_cases_alone_equal_the_host_build(sweep, host_measure):
    """Each of FS36's cases as its own one-product table, the default geometry."""
    for label, mi in fs36_cases():
        agree(sweep([mi]), host_measure(pack([mi])), [label])


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_shared_tables(sweep, levels, host_measure, dtype, guard, share):
    tables = {
        "x by a": fwd_and_grad(dtype, guard),
        "biased beside": fwd_and_grad(dtype, guard, biased_beside=True),
        "three forward": three_forward(dtype, guard),
        "member stack": member_stack(dtype, guard),
        "two sided": two_sided(dtype, guard),
        "mixed": mixed_table(dtype, guard=guard),
    }
    for name, mis in tables.items():
        want = host_measure(pack(mis))
        agree(levels(mis), want, [name] * len(mis))
        for tile, kc, reverse in COMBOS:
            agree(sweep(mis, tile=tile, kc=kc, reverse=reverse, share=share), want,
                  [name] * len(mis))


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("biased", [False, True])
def test_special_values_in_a_shared_view(sweep, host_measure, edit, dtype, guard, biased, share):
    mis, x = shared_special(dtype, guard, biased)
    EDITS[edit](x)
    want = host_measure(pack(mis))
    for tile, kc, reverse in COMBOS[:3]:
        agree(sweep(mis, tile=tile, kc=kc, reverse=reverse, share=share), want, [edit] * 2)


def test_the_mixed_table_in_shuffled_orders(sweep, host_measure):
    mis = mixed_table(torch.bfloat16)
    want = host_measure(pack(mis))
    rng = np.random.default_rng(5)
    for share in (True, False):
        for _ in range(3):
            order = rng.permutation(len(mis))
            got = sweep([mis[i] for i in order], share=share, reverse=bool(order[0] % 2),
                        tile=32, kc=4)
            agree(got, want[order])


# ---- views, shapes, dtypes and flags at the tile edges ----

N_EDGE = [1, 31, 32, 33, 65]
C_EDGE = [1, 31, 32, 33, 63, 64, 65, 200]
LAYOUTS = ["row", "t", "strided"]
K_EDGE = [1, 2, 9, 15, 16, 17, 33]


def shape_triples():
    """(rows, q, w): every q and every w appears with every kind of neighbour; three rotations
    of the columns' pairing, the rows cycling."""
    out = []
    for rot in (0, 1, 3):
        for i, q in enumerate(C_EDGE):
            out.append((N_EDGE[(i + rot) % 5], q, C_EDGE[(i + rot) % 8]))
    return out


def laid_out(t, kind, g):
    """``t``'s values as a view of the layout: row-major, a ``.t()`` view of a contiguous
    transpose, or a slice ``[1::2, 2::3]`` of a larger random tensor (both strides above one)."""
    r, c = t.shape
    if kind == "row":
        return t.contiguous()
    if kind == "t":
        return t.t().contiguous().t()
    big = torch.randn(2 * r + 1, 3 * c + 2, generator=g).to(t.dtype)
    v = big[1:1 + 2 * r:2, 2:2 + 3 * c:3]
    v.copy_(t)
    assert v.shape == t.shape and v.stride() == (2 * big.stride(0), 3)
    return v


@lru_cache(maxsize=None)
def challenge(m, k, w):
    return challenge_matrix(H, m, k, w).contiguous()


def view_product(n, q, w, dtype, layout, biased, guard, seed):
    """A product whose ``A``, ``B`` and ``P`` are views of ``layout`` (k is set by the caller)."""
    mi = make(n, q, w, 1, dtype=dtype, biased=biased, guard=guard, seed=seed)
    g = torch.Generator().manual_seed(seed + 1000)
    a, b, p = (laid_out(t, layout, g) for t in (mi.a, mi.b, mi.p))
    return MeasureInput(a, b, p, mi.r, mi.bias, guard)


def with_k(mis, k):
    return [MeasureInput(mi.a, mi.b, mi.p, challenge(1 + i % 3, k, mi.b.shape[1]), mi.bias,
                         mi.guard) for i, mi in enumerate(mis)]


@lru_cache(maxsize=None)
def edge_products(dtype):
    mis, labels = [], []
    for s, (n, q, w) in enumerate(shape_triples()):
        for layout in LAYOUTS:
            for biased in (False, True):
                for guard in (True, False):
                    mis.append(view_product(n, q, w, dtype, layout, biased, guard,
                                            seed=len(mis)))
                    labels.append((n, q, w, layout, "biased" if biased else "plain",
                                   "guard" if guard else "noguard"))
    return mis, labels


def test_the_edge_grid_covers_every_edge_and_view_kind():
    mis, labels = edge_products(torch.float32)
    assert {x[0] for x in labels} == set(N_EDGE)
    assert {x[1] for x in labels} == set(C_EDGE) and {x[2] for x in labels} == set(C_EDGE)
    strides = {(mi.b.stride(0) == 1 and mi.b.shape[0] > 1, mi.b.stride(0) > 1 and mi.b.stride(1) > 1)
               for mi in mis}
    assert (True, False) in strides and (False, True) in strides
    assert len(mis) == 24 * 12


@pytest.mark.parametrize("k", K_EDGE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_shapes_layouts_and_flags(sweep, host_measure, k, dtype):
    mis, labels = edge_products(dtype)
    mis = with_k(mis, k)
    want = host_measure(pack(mis))
    kcs = sorted({1, 16, KC})
    for i, tile in enumerate((32, 64, 128)):
        for j, kc in enumerate(kcs):
            agree(sweep(mis, tile=tile, kc=kc, reverse=bool((i + j) % 2), share=bool(j % 2)),
                  want, labels)


@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_products_alone_and_in_shuffled_orders(sweep, host_measure, dtype):
    mis, labels = edge_products(dtype)
    mis = with_k(mis, 9)
    want = host_measure(pack(mis))
    for i in range(0, len(mis), 17):  # one-product tables
        agree(sweep([mis[i]], tile=32, kc=4), want[i:i + 1], [labels[i]])
    rng = np.random.default_rng(11)
    for share in (True, False):
        for reverse in (False, True):
            order = rng.permutation(len(mis))
            got = sweep([mis[i] for i in order], tile=64, kc=KC, share=share, reverse=reverse)
            agree(got, want[order], [labels[i] for i in order])


# ---- special values at the tile and view edges ----

SPECIAL_COLS = [31, 63, 127, "last"]   # the last column of a tile of 32, 64, 128; the view's last
SN, SQ, SW = 34, 130, 130              # 2 tiles of 128, 5 of 32, the last row 33


def planted_products(dtype, layout, guard):
    """One product per planted special value: each of NaN, +Inf, -Inf, -0.0 in A, B, P or the
    bias, at a tile's last column, the view's last column and the last row (and row 0); a zero
    row (the last), a zero column in each place."""
    mis, labels = [], []

    def new(target, biased):
        s = len(mis)
        mi = view_product(SN, SQ, SW, dtype, layout, biased or target == "bias", guard, s)
        return mi, {"a": mi.a, "b": mi.b, "p": mi.p, "bias": mi.bias}[target]

    def add(mi, label):
        mis.append(mi)
        labels.append(label)

    for target in ("a", "b", "p", "bias"):
        for value in (float("nan"), float("inf"), float("-inf"), -0.0):
            for col in SPECIAL_COLS:
                for row in ("last", 0):
                    mi, t = new(target, len(mis) % 2 == 0)
                    c = (t.shape[-1] - 1) if col == "last" else col
                    r = (t.shape[0] - 1) if row == "last" else row
                    if t.dim() == 1:
                        t[c] = value
                    else:
                        if c >= t.shape[1]:
                            continue
                        t[r, c] = value
                    add(mi, (target, value, col, row, layout, guard))
        if target == "bias":
            continue
        mi, t = new(target, False)
        t[-1, :] = 0.0
        add(mi, (target, "zero last row", layout, guard))
        for col in SPECIAL_COLS:
            mi, t = new(target, len(mis) % 2 == 1)
            t[:, t.shape[1] - 1 if col == "last" else col] = 0.0
            add(mi, (target, "zero col", col, layout, guard))
    mi, t = new("bias", True)
    t.zero_()
    add(mi, ("zero bias", layout, guard))
    return mis, labels


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values_at_tile_and_view_edges(sweep, host_measure, dtype, guard, layout):
    mis, labels = planted_products(dtype, layout, guard)
    assert len(mis) >= 140
    for k, geoms in ((3, [(32, 1), (64, 16), (128, 16)]), (17, [(64, 16), (32, 4)])):
        table = with_k(mis, k)
        want = host_measure(pack(table))
        for i, (tile, kc) in enumerate(geoms):
            for share in (True, False):
                agree(sweep(table, tile=tile, kc=kc, reverse=bool(i % 2), share=share), want,
                      labels)


def test_the_planted_values_are_there_and_reach_the_rows(host_measure):
    """A guard against vacuous tests: the planted tables hold special values, and the NaN and
    Inf plants change the rows (some row is NaN or Inf, some is finite)."""
    mis, labels = planted_products(torch.float32, "row", True)
    rows = host_measure(pack(with_k(mis, 3)))
    assert np.isnan(rows).any() and np.isinf(rows).any() and np.isfinite(rows).any()
    assert any(torch.isnan(mi.a).any() for mi in mis)
