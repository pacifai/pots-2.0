"""FS37b round 3: a shared group's entries, through the simulator, bit for bit.

A block takes ``ROWS_PER_BLOCK`` rows of a group's view and runs the group's entries one after
another over them, phase-0 and phase-1 entries mixed over one view. Every table
here is built with ``sets_tables.shared_view`` and the group it names is read back from
``mk.read_plan`` before the rows are compared, with ``share=True`` and ``share=False``, against
``host_measure``. The simulator runs one OS thread per CUDA thread, so shapes are small and the
geometries few.

The block-count and shared-memory tests at the end are plain Mac tests (not slow)."""

import contextlib
import shutil

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.sets_tables import (
    group_of,
    phases_of,
    plan_of,
    rb_of,
    separate_biased,
    shared_view,
    with_guard_each,
)
from tests.verification.verifier.matmul_check.shared_tables import DTYPES
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    host_measure,
    hostlib,
    make,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_sim_host import (  # noqa: F401
    agree,
    sim,
    sim_dir,
)
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.measure_table import pack

CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


def sim_test(fn):
    return pytest.mark.slow(pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")(fn))


DEFAULT = (16, 256, 16)
SMALL = (8, 32, 4)
ODD = (16, 96, 3)
WIDE = (64, 128, 1)


@contextlib.contextmanager
def geometry_set(geometry):
    prev = mk.set_geometry(rows_per_block=geometry[0], tile_cols=geometry[1],
                           k_chunk=geometry[2])
    try:
        yield
    finally:
        mk.set_geometry(rows_per_block=prev[0], tile_cols=prev[1], k_chunk=prev[2])


def check(sim, host_measure, mis, *, geometry=DEFAULT, label=""):
    want = host_measure(pack(mis))
    with geometry_set(geometry):
        for share in (True, False):
            agree(sim(mis, share=share, geometry=geometry), want, f"{label} {geometry} share={share}")


# ---- groups of e entries ----

@pytest.mark.parametrize("e", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("mode", ["A", "B", "mix"])
@sim_test
def test_a_group_of_e_entries(sim, host_measure, e, mode):
    """Groups of 2 to 9 entries, run one after another over a block's rows."""
    n = 2 * rb_of(DEFAULT[0], e) + 3
    mis, _ = shared_view(e, mode, n=n, q=70, dtype=torch.bfloat16, seed=e)
    group_of(mis, e, n=n, phases=phases_of(e, mode))
    check(sim, host_measure, mis, label=f"e={e} {mode}")


@pytest.mark.parametrize("e", [3, 5])
@pytest.mark.parametrize("geometry", [SMALL, ODD, WIDE])
@sim_test
def test_groups_across_geometries(sim, host_measure, e, geometry):
    """Several geometries through ``set_geometry`` (restored in a ``finally``): ``RB`` changes
    with ``ROWS_PER_BLOCK``, the tile edge and chunk width with the others."""
    rb = rb_of(geometry[0], e)
    n = 2 * rb + 1
    mis, _ = shared_view(e, "mix", n=n, q=100, dtype=torch.float32, k=5, seed=e)
    group_of(mis, e, n=n, phases=phases_of(e, "mix"))
    before = mk.set_geometry()
    check(sim, host_measure, mis, geometry=geometry, label=f"e={e}")
    assert mk.set_geometry() == before


@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("layout", ["row", "t"])
@pytest.mark.parametrize("dtype", DTYPES)
@sim_test
def test_dtypes_layouts_and_guard(sim, host_measure, dtype, layout, guard):
    e = 5
    mis, _ = shared_view(e, "mix", n=21, q=75, dtype=dtype, guard=guard, layout=layout, seed=3)
    group_of(mis, e, n=21)
    check(sim, host_measure, mis, label=f"{dtype} {layout} guard={guard}")


@sim_test
def test_the_guard_differs_between_entries_of_one_group(sim, host_measure):
    e = 6
    mis, _ = shared_view(e, "mix", n=19, q=66, dtype=torch.bfloat16, seed=8)
    mis = with_guard_each(mis, [bool(i % 2) for i in range(e)])
    group_of(mis, e, n=19)
    check(sim, host_measure, mis, label="mixed guard")


@pytest.mark.parametrize("mode", ["A", "mix"])
@sim_test
def test_a_transposed_view_with_nine_entries(sim, host_measure, mode):
    mis, _ = shared_view(9, mode, n=11, q=40, dtype=torch.bfloat16, layout="t", seed=2)
    group_of(mis, 9, n=11)
    check(sim, host_measure, mis, geometry=SMALL, label=mode)


# ---- phase-1 groups and the chunks ----

@pytest.mark.parametrize("e", [2, 5])
@pytest.mark.parametrize("k", [1, 5, 9])
@sim_test
def test_phase_one_groups_with_k_around_the_chunk(sim, host_measure, e, k):
    """``SMALL``'s ``K_CHUNK`` is 4: ``k = 5`` is 4 + 1 and ``k = 9`` is 4 + 4 + 1 (the extras
    ride the first chunk); ``k = 1`` is one column."""
    assert SMALL[2] == 4
    rb = rb_of(SMALL[0], e)
    n = 2 * rb + 1
    mis, _ = shared_view(e, "A", n=n, q=70, dtype=torch.float32, k=k, seed=k)
    group_of(mis, e, n=n, phases=[1] * e)
    check(sim, host_measure, mis, geometry=SMALL, label=f"A e={e} k={k}")


@pytest.mark.parametrize("k", [1, 17, 33])
@sim_test
def test_a_mixed_group_with_k_around_the_default_chunk(sim, host_measure, k):
    """``K_CHUNK = 16``: ``k`` = 17 and 33 run two and three chunks."""
    mis, _ = shared_view(3, "mix", n=40, q=70, dtype=torch.bfloat16, k=k, seed=k)
    group_of(mis, 3, n=40)
    check(sim, host_measure, mis, label=f"k={k}")


# ---- row counts ----

@pytest.mark.parametrize("e", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("which", ["not a multiple", "fewer than RB"])
@sim_test
def test_row_counts_off_the_block_size(sim, host_measure, e, which):
    rb = rb_of(SMALL[0], e)
    n = 2 * rb + 1 if which == "not a multiple" else max(1, rb - 1)
    assert (n % rb != 0) if which == "not a multiple" else (n < rb)
    mis, _ = shared_view(e, "mix", n=n, q=70, dtype=torch.bfloat16, seed=n + e)
    group_of(mis, e, n=n)
    check(sim, host_measure, mis, geometry=SMALL, label=f"e={e} n={n}")


@pytest.mark.parametrize("e", [2, 5])
@sim_test
def test_row_counts_off_the_block_size_at_the_default_geometry(sim, host_measure, e):
    rb = rb_of(DEFAULT[0], e)
    for n in (rb + 1, rb - 1, 3 * rb + rb // 2):
        mis, _ = shared_view(e, "mix", n=n, q=33, dtype=torch.float32, seed=n)
        group_of(mis, e, n=n)
        check(sim, host_measure, mis, label=f"e={e} n={n} rb={rb}")


# ---- a biased product beside a shared group ----

@pytest.mark.parametrize("guard", [True, False])
@sim_test
def test_a_biased_product_alone_beside_a_group(sim, host_measure, guard):
    e = 4
    mis, _ = shared_view(e, "mix", n=19, q=50, dtype=torch.bfloat16, guard=guard, seed=5)
    alone = separate_biased(dtype=torch.bfloat16, guard=guard)
    table = mis + [alone]
    group_of(table, e, n=19)
    check(sim, host_measure, table, label="biased alone")


@sim_test
def test_a_biased_products_phase_one_node_in_the_group(sim, host_measure):
    """The product's phase-0 node (the bias item) is alone, its phase-1 node joins the group."""
    e = 5
    mis, _ = shared_view(e, "A", n=23, q=50, dtype=torch.float32, bias_at=2, seed=6)
    table = mis + [separate_biased(dtype=torch.float32)]
    group_of(table, e, n=23, phases=[1] * e)
    check(sim, host_measure, table, label="biased member")


# ---- special values in a shared view's last row and column ----

PLACES = {
    "last row": lambda x, v: x.__setitem__((x.shape[0] - 1, slice(None)), v),
    "last column": lambda x, v: x.__setitem__((slice(None), x.shape[1] - 1), v),
    "corner": lambda x, v: x.__setitem__((x.shape[0] - 1, x.shape[1] - 1), v),
}
VALUES = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf"), "-0.0": -0.0}


@pytest.mark.parametrize("value", list(VALUES))
@pytest.mark.parametrize("place", list(PLACES))
@sim_test
def test_special_values_in_a_shared_views_edge(sim, host_measure, place, value):
    """A 5-entry mixed group, ``n`` not a multiple of ``RB``, ``q`` past a tile edge: the last
    row sits in a partly filled block, the last column in a masked tile."""
    dtype = DTYPES[len(value) % 2]
    guard = bool(len(place) % 2)
    n = 2 * rb_of(SMALL[0], 5) + 1
    mis, x = shared_view(5, "mix", n=n, q=70, dtype=dtype, guard=guard, seed=9)
    PLACES[place](x, VALUES[value])
    group_of(mis, 5, n=n)
    check(sim, host_measure, mis, geometry=SMALL, label=f"{place} {value}")


@pytest.mark.parametrize("dtype", DTYPES)
@sim_test
def test_special_values_with_both_guards(sim, host_measure, dtype):
    for guard in (True, False):
        mis, x = shared_view(3, "mix", n=9, q=40, dtype=dtype, guard=guard, seed=4)
        x[-1, :] = float("nan")
        x[:, -1] = -0.0
        x[0, -1] = float("inf")
        group_of(mis, 3, n=9)
        check(sim, host_measure, mis, label=f"guard={guard}")


# ---- plain Mac tests: block counts and shared memory ----

def tables_for_blocks():
    out = []
    for e in (1, 2, 3, 4, 5, 9):
        out.append(shared_view(e, "mix" if e > 1 else "A", n=2 * 32 + 5, q=20,
                               dtype=torch.float32, seed=e)[0])
    out.append(shared_view(5, "A", n=7, q=20, dtype=torch.float32, k=40, seed=1)[0])
    out.append(shared_view(9, "B", n=1, q=20, dtype=torch.float32, seed=1)[0]
               + [separate_biased(dtype=torch.float32)])
    return out


@pytest.mark.parametrize("rpb", [8, 16, 32, 64])
@pytest.mark.parametrize("share", [True, False])
def test_block_starts_count_blocks_by_the_sets_mapping(rpb, share):
    """Phase 0 and 1 groups take ``ceil(count / rpb)`` blocks, one entry per batch; a
    phase-2 group (``k = 40``: 45 items, over every ``rpb``) takes one."""
    saw_shared = False
    for mis in tables_for_blocks():
        plan = plan_of(mis) if share else mk.plain_plan(
            np.frombuffer(pack(mis).array, dtype=mk.MPROD_DTYPE).copy())
        starts = mk.block_starts(plan, rpb).astype(np.int64)
        off = plan.level_off.astype(np.int64)
        for lv in range(plan.n_levels):
            lo, hi = int(off[lv]), int(off[lv + 1])
            groups = plan.reads[lo:hi - 1]
            assert len(starts[lo:hi]) == len(groups) + 1
            assert starts[lo] == 0
            for g, r in enumerate(groups):
                phase = int(plan.entries["phase"][int(r["entry0"])])
                ne, count = int(r["n_entries"]), int(r["count"])
                saw_shared |= ne > 1
                if phase == 2:
                    want = 1
                else:
                    want = -(-count // rpb)
                assert starts[lo + g + 1] - starts[lo + g] == want, (lv, g, ne, count, rpb)
    assert saw_shared == share


def test_block_starts_default_rows_per_block_is_the_modules():
    plan = plan_of(shared_view(5, "mix", n=70, q=20, dtype=torch.float32)[0])
    assert np.array_equal(mk.block_starts(plan), mk.block_starts(plan, mk.ROWS_PER_BLOCK))


def test_shared_bytes_counts_every_shared_array():
    for rpb, tile, kc in (DEFAULT, SMALL, ODD, WIDE):
        want = 224 + 8 * rpb + 4 * (rpb * (tile + 1) + kc * (tile + 1) + 2 * tile
                                    + 2 * rpb * (kc + 2))
        assert mk.shared_bytes(rpb, tile, kc) == want


def test_set_geometry_refuses_a_geometry_over_48k():
    before = mk.set_geometry()
    assert mk.shared_bytes(32, 256, 16) > mk.SHARED_LIMIT >= mk.shared_bytes(*DEFAULT)
    with pytest.raises(ValueError):
        mk.set_geometry(rows_per_block=32, tile_cols=256, k_chunk=16)
    assert mk.set_geometry() == before


def test_there_is_no_e_max():
    assert not hasattr(mk, "E_MAX")
    assert not any("E_MAX" in o for o in mk.build_options())
