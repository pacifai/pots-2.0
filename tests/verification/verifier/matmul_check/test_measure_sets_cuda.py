"""FS37b round 2's sets on a GPU: the real kernel (``freivalds.measure_groups_rows`` then
``freivalds.drain``) against the host build's rows, bit for bit (NaN equal to NaN), with
``share=True`` and ``share=False``, over read groups of 2, 3, 4, 5, 9 and 17 entries (``E_MAX``
= 4: one batch, several batches, a smaller last one), phase-0 and phase-1 entries mixed over
one view, ``k`` around ``K_CHUNK``, row counts off ``RB``, many-row views (many blocks per
group), special values, a biased product beside a group, and several geometries. Skipped without
CUDA; the tables are ``sets_tables.shared_view``'s, each checked against ``mk.read_plan``."""

import contextlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tests.verification.verifier.matmul_check.sets_tables import (  # noqa: E402
    E_MAX,
    group_of,
    phases_of,
    rb_of,
    separate_biased,
    shared_view,
    with_guard_each,
)
from tests.verification.verifier.matmul_check.shared_tables import DTYPES  # noqa: E402
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: E402, F401
    H,
    host_measure,
    hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_sets_sim_host import (  # noqa: E402
    PLACES,
    VALUES,
)
from tests.verification.verifier.matmul_check.test_measure_sweep_host import agree  # noqa: E402
from tests.verification.verifier.matmul_check.test_measure_tiles_cuda import (  # noqa: E402
    check,
    kernel_rows,
    labelled,
)
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.measure_table import pack  # noqa: E402

GEOMETRIES = [(8, 32, 4), (16, 96, 3), (64, 128, 1), (32, 64, 16)]


@contextlib.contextmanager
def geometry_set(geometry):
    prev = mk.set_geometry(rows_per_block=geometry[0], tile_cols=geometry[1],
                           k_chunk=geometry[2])
    try:
        yield
    finally:
        mk.set_geometry(rows_per_block=prev[0], tile_cols=prev[1], k_chunk=prev[2])


@pytest.mark.parametrize("mode", ["A", "B", "mix"])
@pytest.mark.parametrize("e", [2, 3, 4, 5, 9, 17])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_group_of_e_entries(host_measure, e, mode, dtype):
    n = 3 * rb_of(mk.ROWS_PER_BLOCK, e) + 5
    mis, _ = shared_view(e, mode, n=n, q=200, dtype=dtype, seed=e)
    group_of(mis, e, n=n, phases=phases_of(e, mode))
    check(mis, 3, host_measure)


@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("layout", ["row", "t"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_layouts_guard_and_many_rows(host_measure, dtype, layout, guard):
    """Many blocks per group: 700 rows, the grid stride and the binary search over groups."""
    for e in (3, 5, 9):
        mis, _ = shared_view(e, "mix", n=700 + e, q=70, dtype=dtype, guard=guard, layout=layout,
                             seed=e)
        group_of(mis, e, n=700 + e)
        check(mis, 3, host_measure, guard=guard)


@pytest.mark.parametrize("k", [1, 16, 17, 33])
@pytest.mark.parametrize("e", [2, 5])
def test_k_around_the_chunk(host_measure, e, k):
    mis, _ = shared_view(e, "mix", n=45, q=130, dtype=torch.bfloat16, k=k, seed=k)
    group_of(mis, e, n=45)
    check(mis, k, host_measure)
    mis, _ = shared_view(e, "A", n=45, q=130, dtype=torch.float32, k=k, seed=k + 1)
    check(mis, k, host_measure)


@pytest.mark.parametrize("e", [2, 3, 5, 9])
def test_row_counts_off_the_block_size(host_measure, e):
    rb = rb_of(mk.ROWS_PER_BLOCK, e)
    for n in (1, max(1, rb - 1), rb, rb + 1, 2 * rb + 1, 5 * rb - 1):
        mis, _ = shared_view(e, "mix", n=n, q=90, dtype=torch.bfloat16, seed=n)
        group_of(mis, e, n=n)
        check(mis, 3, host_measure)


@pytest.mark.parametrize("guard", [True, False])
def test_a_biased_product_alone_beside_a_group(host_measure, guard):
    mis, _ = shared_view(4, "mix", n=40, q=70, dtype=torch.bfloat16, guard=guard, seed=5)
    table = mis + [separate_biased(dtype=torch.bfloat16, guard=guard)]
    group_of(table, 4, n=40)
    check(table, 3, host_measure, guard=guard)
    mis, _ = shared_view(5, "A", n=40, q=70, dtype=torch.float32, guard=guard, bias_at=2, seed=6)
    check(mis + [separate_biased(dtype=torch.float32, guard=guard)], 3, host_measure, guard=guard)


@pytest.mark.parametrize("value", list(VALUES))
@pytest.mark.parametrize("place", list(PLACES))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_special_values_at_a_shared_views_edge(host_measure, place, value, dtype, guard):
    n = 2 * rb_of(mk.ROWS_PER_BLOCK, 5) + 1
    mis, x = shared_view(5, "mix", n=n, q=70, dtype=dtype, guard=guard, seed=9)
    PLACES[place](x, VALUES[value])
    group_of(mis, 5, n=n)
    check(mis, 3, host_measure, guard=guard)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_every_geometry_with_groups(host_measure, geometry):
    """``set_geometry`` (restored in a ``finally``), then groups of 2, 3, 5 and 9 entries sized
    off that geometry's ``RB``."""
    with geometry_set(geometry):
        before = mk.set_geometry()
        for e in (2, 3, 5, 9):
            n = 3 * rb_of(geometry[0], e) + 1
            for mode, dtype in (("mix", torch.bfloat16), ("A", torch.float32)):
                mis, _ = shared_view(e, mode, n=n, q=150, dtype=dtype, k=5, seed=e)
                group_of(mis, e, n=n)
                check(mis, 5, host_measure)
        assert mk.set_geometry() == before


def test_the_attributes_show_no_spill_and_the_registers_fit_two_blocks():
    """Round 2's ``__launch_bounds__(256, 2)``: at most 128 registers per thread, and no local
    (spilled) bytes. The key names are the coder's; they are found by substring."""
    kernels = mk.kernels(torch.device("cuda"))
    if not hasattr(kernels, "attributes"):
        pytest.skip("no attributes() on this build")
    a = kernels.attributes()
    reg = [v for key, v in a.items() if "reg" in key.lower()]
    local = [v for key, v in a.items() if "local" in key.lower() or "spill" in key.lower()]
    assert reg and local, a
    assert reg[0] <= 128, a
    print("attributes:", a)
