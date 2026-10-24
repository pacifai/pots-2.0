"""FS37b's one-sweep tiled measuring kernel on a GPU. Skipped without CUDA; they run on a rented
card.

The real kernel (``freivalds.measure_groups_rows`` then ``freivalds.drain``) is held to the host
build's rows bit for bit (NaN equal to NaN where special values are planted), over the tile-edge
grid of ``test_measure_sweep_host.py``, long and many-row shapes, a scaled-down head, three
kernel geometries, and the shared tables with ``share=True`` against ``share=False``."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tests.verification.verifier.matmul_check.shared_tables import (  # noqa: E402
    DTYPES,
    EDITS,
    fwd_and_grad,
    member_stack,
    mixed_table,
    shared_special,
    three_forward,
    two_sided,
)
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: E402, F401
    H,
    host_measure,
    hostlib,
    make,
)
from tests.verification.verifier.matmul_check.test_measure_levels_cuda import to_device  # noqa: E402
from tests.verification.verifier.matmul_check.test_measure_sweep_host import (  # noqa: E402
    K_EDGE,
    LAYOUTS,
    agree,
    challenge,
    edge_products,
    laid_out,
    planted_products,
    with_k,
)
from verification.verifier.matmul_check import freivalds as fv  # noqa: E402
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.freivalds import MeasureGroup  # noqa: E402
from verification.verifier.matmul_check.measure_table import MeasureInput, pack  # noqa: E402

GEOMETRIES = [(32, 64, 16), (8, 32, 4), (64, 128, 16)]   # (rows_per_block, tile_cols, k_chunk)


def groups_of(mis):
    """One one-member group per product (labels 1, 2, ...: the challenge of product ``i`` is
    ``challenge(i + 1, k, w)``), on the device with the layouts and shared storage kept."""
    return [MeasureGroup(a[None], b[None], p[None], [i + 1], bias)
            for i, (a, b, p, bias) in enumerate(to_device(mis))]


def labelled(mis, k):
    """``mis`` with the challenge the group of the same position draws."""
    return [MeasureInput(mi.a, mi.b, mi.p, challenge(i + 1, k, mi.b.shape[1]), mi.bias, mi.guard)
            for i, mi in enumerate(mis)]


def kernel_rows(mis, k, *, guard, share):
    rows = fv.measure_groups_rows(groups_of(mis), h=H, k=k, guard=guard, share=share)
    fv.drain()
    return rows.cpu().numpy()


def check(mis, k, host_measure, *, guard=True, labels=None):
    """The kernel's rows, ``share`` both ways, against the host build's. ``mis`` share one
    ``guard`` (the call has one)."""
    mis = labelled(mis, k)
    want = host_measure(pack(mis))
    for share in (True, False):
        agree(kernel_rows(mis, k, guard=guard, share=share), want, labels)


def by_guard(mis, labels):
    return {g: ([m for m in mis if m.guard == g], [lb for m, lb in zip(mis, labels) if m.guard == g])
            for g in (True, False)}


# ---- the edge grid ----

@pytest.mark.parametrize("k", K_EDGE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_edge_shapes_layouts_and_flags(host_measure, k, dtype, guard):
    mis, labels = by_guard(*edge_products(dtype))[guard]
    check(mis, k, host_measure, guard=guard, labels=labels)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values_at_tile_and_view_edges(host_measure, dtype, guard, layout):
    mis, labels = planted_products(dtype, layout, guard)
    for k in (3, 17):
        check(mis, k, host_measure, guard=guard, labels=labels)


def test_the_unbatched_edge_products_one_group_per_call(host_measure):
    """Each of a sample of edge products as its own one-group call (the grid's blocks are then
    the product's alone)."""
    mis, labels = edge_products(torch.bfloat16)
    for i in range(0, len(mis), 13):
        check([mis[i]], 9, host_measure, guard=mis[i].guard, labels=[labels[i]])


# ---- long, many-row and head-like shapes ----

@pytest.mark.parametrize("layout", ["row", "t"])
@pytest.mark.parametrize("n,q,w", [(2, 70000, 40), (1, 64, 70000), (1, 70000, 33), (2, 33, 70001)])
@pytest.mark.parametrize("guard", [True, False])
def test_long_rows(host_measure, n, q, w, layout, guard):
    mi = make(n, q, w, 3, dtype=torch.bfloat16, biased=True, guard=guard, seed=n + q)
    g = torch.Generator().manual_seed(5)
    mi = MeasureInput(*(laid_out(t, layout, g) for t in (mi.a, mi.b, mi.p)), mi.r, mi.bias, guard)
    check([mi], 3, host_measure, guard=guard)


@pytest.mark.parametrize("layout", ["row", "t"])
@pytest.mark.parametrize("n,q,w", [(3000, 40, 40), (5, 3000, 40), (40, 40, 3000)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_many_rows(host_measure, n, q, w, layout, dtype):
    mi = make(n, q, w, 3, dtype=dtype, biased=bool(w == 40), guard=True, seed=q)
    g = torch.Generator().manual_seed(6)
    mi = MeasureInput(*(laid_out(t, layout, g) for t in (mi.a, mi.b, mi.p)), mi.r, mi.bias, True)
    check([mi], 3, host_measure)


def head_like():
    """The head's shape scaled down: ``A`` 64x256, ``B`` a transposed view 256x20000 of a
    [20000, 256] tensor (``W_E^T``), ``P`` 64x20000, bf16."""
    g = torch.Generator().manual_seed(21)
    a = torch.randn(64, 256, generator=g).to(torch.bfloat16)
    wt = torch.randn(20000, 256, generator=g).to(torch.bfloat16)
    b = wt.t()
    p = (a.float() @ b.float() + 1e-3 * torch.randn(64, 20000, generator=g)).to(torch.bfloat16)
    assert b.stride() == (1, 256)
    return MeasureInput(a, b, p, challenge(1, 3, 20000), None, True)


@pytest.mark.parametrize("k", [3, 17])
def test_a_head_like_shape(host_measure, k):
    check([head_like()], k, host_measure)


# ---- geometries ----

@pytest.mark.parametrize("geometry", GEOMETRIES[1:])
def test_every_geometry_gives_the_default_geometrys_bits(host_measure, geometry):
    """The default geometry, then ``geometry``: the same rows as the host build; the previous
    geometry is restored whatever happens."""
    rows_per_block, tile_cols, k_chunk = geometry
    tables = []
    for dtype in DTYPES:
        mis, labels = by_guard(*edge_products(dtype))[True]
        tables.append((mis, 17, labels))
        tables.append((mis, 2, labels))
        tables.append((fwd_and_grad(dtype, True), 3, None))
        tables.append((mixed_table(dtype, guard=True), 3, None))
    tables.append(([head_like()], 3, None))
    wants = []
    for mis, k, _ in tables:
        wants.append(host_measure(pack(labelled(mis, k))))
    prev = mk.set_geometry(rows_per_block=rows_per_block, tile_cols=tile_cols, k_chunk=k_chunk)
    try:
        for (mis, k, labels), want in zip(tables, wants):
            for share in (True, False):
                agree(kernel_rows(labelled(mis, k), k, guard=True, share=share), want, labels)
    finally:
        mk.set_geometry(rows_per_block=prev[0], tile_cols=prev[1], k_chunk=prev[2])
    assert mk.set_geometry() == prev


def test_set_geometry_refuses_bad_values_and_changes_nothing():
    before = mk.set_geometry()
    for bad in ({"tile_cols": 48}, {"tile_cols": 0}, {"rows_per_block": 12},
                {"rows_per_block": 0}, {"k_chunk": 0}):
        with pytest.raises(ValueError):
            mk.set_geometry(**bad)
    assert mk.set_geometry() == before


# ---- shared reads against unshared ones ----

def shared_vs_plain(mis, k, host_measure, *, guard, label, special=False):
    mis = labelled(mis, k)
    plain = kernel_rows(mis, k, guard=guard, share=False)
    shared = kernel_rows(mis, k, guard=guard, share=True)
    agree(shared, plain, [label] * len(mis))
    agree(plain, host_measure(pack(mis)), [label] * len(mis))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_shared_tables_equal_unshared_ones(host_measure, dtype, guard):
    tables = {
        "x by a": fwd_and_grad(dtype, guard),
        "biased beside": fwd_and_grad(dtype, guard, biased_beside=True),
        "three forward": three_forward(dtype, guard),
        "member stack": member_stack(dtype, guard),
        "two sided": two_sided(dtype, guard),
        "mixed": mixed_table(dtype, guard=guard),
    }
    for name, mis in tables.items():
        shared_vs_plain(mis, 3, host_measure, guard=guard, label=name)


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_special_values_in_a_shared_view(host_measure, edit, dtype, guard):
    mis, x = shared_special(dtype, guard)
    EDITS[edit](x)
    shared_vs_plain(mis, 2, host_measure, guard=guard, label=edit)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_mixed_shared_table_in_shuffled_orders(host_measure, dtype):
    mis = labelled(mixed_table(dtype, guard=True), 3)
    want = kernel_rows(mis, 3, guard=True, share=False)
    rng = np.random.default_rng(3)
    for _ in range(3):
        order = rng.permutation(len(mis))
        # relabel by position so each group draws the challenge of its own product
        perm = [MeasureInput(mis[i].a, mis[i].b, mis[i].p, mis[i].r, mis[i].bias, True)
                for i in order]
        groups = [MeasureGroup(a[None], b[None], p[None], [int(i) + 1], bias)
                  for i, (a, b, p, bias) in zip(order, to_device(perm))]
        got = fv.measure_groups_rows(groups, h=H, k=3, guard=True, share=True)
        fv.drain()
        agree(got.cpu().numpy(), want[order])
