"""FS51 (``HOST_SIDE_PLAN_DRAFT.md`` §2.2–§2.3): the measuring tables' structure built once, on
CPU tensors through the host emulation of the level launches (``pots_host_measure_levels``).

``measure_kernel.build_host_table`` is the host side of ``enqueue``; the tests feed it CPU
groups, with challenge addresses from the host source's buffer, and run its read plan and work
array through the host build. They hold: two calls of one structure at other addresses give the
same tables but for addresses and the same rows bit for bit with the cache on and off (and as
the recipe before the cache built them); a shared view that moved rebuilds the plan and is
counted; a plan built over another group order gives the same rows; the challenge source's
layout and offsets equal a fresh ``prefetch``'s, and so do the bytes drawn."""

import ctypes

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    bits,
    hostlib,
)
from verification.verifier.matmul_check import challenge_source as cs
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.freivalds import MeasureGroup
from verification.verifier.matmul_check.measure_table import row_width

DTYPES = [torch.float32, torch.bfloat16]
N, Q, W, R = 9, 21, 17, 6  # X is [N, Q]; products X·Wt, and Dᵀ·X


@pytest.fixture(autouse=True)
def fresh_cache():
    mk.table_cache_clear()
    mk.table_cache_stats(reset=True)
    yield
    mk.table_cache_clear()


@pytest.fixture(scope="module")
def levels(hostlib):  # noqa: F811
    hostlib.pots_host_measure_levels.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    hostlib.pots_host_measure_levels.restype = ctypes.c_int

    def run(ht: mk.HostTable, k: int) -> np.ndarray:
        """The rows of ``ht``: its plan's levels run in loops over its products and works."""
        scratch = np.zeros(ht.scratch_n + 8, dtype=np.uint8)
        base = (scratch.ctypes.data + 7) & ~7
        works, plan = ht.works(base), ht.plan
        rows = np.empty((len(ht.prods), row_width(k)), dtype=np.float64)
        err = hostlib.pots_host_measure_levels(
            ht.prods.ctypes.data, len(ht.prods), works.ctypes.data, plan.reads.ctypes.data,
            plan.level_off.ctypes.data, plan.n_levels, plan.entries.ctypes.data,
            rows.ctypes.data, 0)
        assert err == 0
        return rows

    return run


def groups_of(seed: int, dtype, *, x_moved: bool = False, reverse_storage: bool = False,
              members: int = 3, label0: int = 10) -> list[MeasureGroup]:
    """Three groups on CPU with real shares: ``members`` products ``X_i·W_i`` whose A is
    ``X_i``, ``members`` products ``D_i·X_i`` whose B is the same view (one read group of two
    nodes a member), and a biased product alone. ``x_moved`` makes the second group read a copy
    of ``X``, so the shares are gone; ``reverse_storage`` carves the stacks from one storage in
    the opposite order (the plan's group order follows addresses)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(dtype)

    pool = rnd(members * (N * Q + Q * W + R * N) + 3 * (N * Q + Q * W + N * W) + W + 64).float()
    pool = pool.to(dtype)
    at = [0]

    def carve(*shape):
        n = int(np.prod(shape))
        t = pool[at[0]:at[0] + n].view(*shape)
        at[0] += n
        return t

    if reverse_storage:  # carve in the opposite order: later groups get the lower addresses
        b3 = carve(1, Q, W)
        a3 = carve(1, N, Q)
        d = carve(members, R, N)
        w = carve(members, Q, W)
        x = carve(members, N, Q)
    else:
        x = carve(members, N, Q)
        w = carve(members, Q, W)
        d = carve(members, R, N)
        a3 = carve(1, N, Q)
        b3 = carve(1, Q, W)
    for t in (x, w, d, a3, b3):
        t.copy_(rnd(*t.shape))
    x2 = x.clone() if x_moved else x
    p1 = (x.float() @ w.float()).to(dtype)
    p2 = (d.float() @ x.float()).to(dtype) + rnd(members, R, Q) * 0.01
    p1 = p1 + rnd(*p1.shape) * 0.01
    p3 = (a3.float() @ b3.float()).to(dtype)
    bias = rnd(W)
    ms = list(range(label0, label0 + members))
    return [MeasureGroup(x, w, p1, ms),
            MeasureGroup(d, x2, p2, [m + 100 for m in ms]),
            MeasureGroup(a3, b3, p3, [label0 + 200], bias)]


def table_of(groups, k, *, share=True, guard=True, source=None):
    """``(HostTable, keep)``: the host table of ``groups`` with this call's challenges drawn by
    the host source on CPU (``keep`` holds the buffer whose addresses the products carry)."""
    source = source or cs.HostChallengeSource()
    structure = mk.table_structure(groups, k, guard, share)
    buf, offs, _ = source.draw(H, structure.ms, k, structure.widths, stream=None)
    ht = mk.build_host_table(groups, np.uint64(buf.data_ptr()) + offs.astype(np.uint64), k, guard,
                             share, structure)
    ht.keep = buf  # the products' challenge addresses point into it
    return ht, buf


def old_rows(levels, groups, k, share, guard=True):
    """The rows by the recipe before the cache: ``prods_array``, ``read_plan``, ``works_array``."""
    source = cs.HostChallengeSource()
    ms = [int(m) for g in groups for m in g.ms]
    widths = [g.b.shape[2] for g in groups for _ in g.ms]
    buf, offs, _ = source.draw(H, ms, k, widths, stream=None)
    prods = mk.prods_array(groups, np.uint64(buf.data_ptr()) + offs.astype(np.uint64), k, guard)
    plan = mk.read_plan(prods) if share else mk.plain_plan(prods)
    ht = mk.HostTable(prods, mk._build(prods, share))
    ht.keep = buf
    assert np.array_equal(ht.plan.reads, plan.reads)
    return levels(ht, k)


def without_addresses(ht: mk.HostTable, base: int) -> tuple[np.ndarray, np.ndarray, bytes]:
    """The table's products and works with every address field zeroed, and the tail's bytes."""
    prods = ht.prods.copy()
    for v in ("a", "b", "p", "bias"):
        prods[v]["addr"] = 0
    prods["r_addr"] = 0
    works = ht.works(base)
    works["scratch"] = 0
    return prods, works, ht.built.tail.tobytes()


# ---- the cache ----

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("share", [True, False])
def test_two_calls_of_one_structure_differ_only_in_addresses(dtype, share):
    g1 = groups_of(1, dtype)  # held, so the next call's tensors are at other addresses
    ht1, _ = table_of(g1, 3, share=share)
    ht2, _ = table_of(groups_of(2, dtype), 3, share=share)
    assert mk.table_cache_stats() == {"hits": 1, "misses": 1, "plan_rebuilds": 0}
    assert ht1.built is ht2.built
    p1, w1, t1 = without_addresses(ht1, 0)
    p2, w2, t2 = without_addresses(ht2, 0)
    assert p1.tobytes() == p2.tobytes() and w1.tobytes() == w2.tobytes() and t1 == t2
    assert ht1.prods["a"]["addr"].tolist() != ht2.prods["a"]["addr"].tolist()
    assert ht1.total == ht2.total


@pytest.mark.parametrize("k", [3, 9, 21])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("guard", [True, False])
def test_rows_are_the_same_bits_with_the_cache_on_and_off(levels, k, dtype, share, guard):
    table_of(groups_of(1, dtype), k, share=share, guard=guard)  # fills the entry
    groups = groups_of(2, dtype)
    ht, _ = table_of(groups, k, share=share, guard=guard)  # a hit
    assert mk.table_cache_stats()["hits"] == 1
    on = levels(ht, k)
    with mk.table_cache(False):
        ht_off, _ = table_of(groups, k, share=share, guard=guard)
        off = levels(ht_off, k)
    assert np.array_equal(bits(on), bits(off))
    assert np.array_equal(bits(on), bits(old_rows(levels, groups, k, share, guard)))
    assert on.shape == (7, row_width(k)) and np.any(on != 0)


def test_a_shared_plan_has_its_shares_and_the_plain_plan_none():
    ht, _ = table_of(groups_of(1, torch.float32), 3, share=True)
    assert ht.plan.shared_groups == 3
    ht, _ = table_of(groups_of(1, torch.float32), 3, share=False)
    assert ht.plan.shared_groups == 0


def test_a_moved_shared_view_rebuilds_the_plan_and_counts_it(levels):
    k = 3
    table_of(groups_of(1, torch.float32), k)
    moved = groups_of(2, torch.float32, x_moved=True)
    ht, _ = table_of(moved, k)
    stats = mk.table_cache_stats()
    assert stats == {"hits": 1, "misses": 1, "plan_rebuilds": 1}
    assert ht.plan.shared_groups == 0  # the rebuilt plan sees no shares
    with mk.table_cache(False):
        fresh, _ = table_of(moved, k)
    assert np.array_equal(ht.plan.reads, fresh.plan.reads)
    assert np.array_equal(bits(levels(ht, k)), bits(levels(fresh, k)))
    # the rebuilt plan replaced the entry: the next call with moved views hits cleanly
    table_of(groups_of(3, torch.float32, x_moved=True), k)
    assert mk.table_cache_stats() == {"hits": 2, "misses": 1, "plan_rebuilds": 1}
    # and a call whose views are shared again keeps the share-less plan: a missed share
    # costs speed, never bits
    again = groups_of(4, torch.float32)
    ht3, _ = table_of(again, k)
    assert mk.table_cache_stats()["plan_rebuilds"] == 1 and ht3.plan.shared_groups == 0
    with mk.table_cache(False):
        ht_off, _ = table_of(again, k)
    assert ht_off.plan.shared_groups == 3
    assert np.array_equal(bits(levels(ht3, k)), bits(levels(ht_off, k)))


def test_the_plan_check_does_not_fire_when_the_shares_hold():
    for seed in (1, 2, 3):
        table_of(groups_of(seed, torch.bfloat16), 3)
    assert mk.table_cache_stats() == {"hits": 2, "misses": 1, "plan_rebuilds": 0}


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_permuted_group_order_gives_the_same_rows(levels, dtype):
    k = 9
    first, _ = table_of(groups_of(1, dtype), k)  # plan built over one address order
    groups = groups_of(2, dtype, reverse_storage=True)
    cached, _ = table_of(groups, k)
    assert mk.table_cache_stats()["plan_rebuilds"] == 0
    with mk.table_cache(False):
        fresh, _ = table_of(groups, k)
    # the fresh plan orders its groups by the new addresses; the cached one keeps the old order
    assert not np.array_equal(cached.plan.entries, fresh.plan.entries)
    assert np.array_equal(cached.plan.entries, first.plan.entries)
    assert np.array_equal(bits(levels(cached, k)), bits(levels(fresh, k)))


def test_a_different_structure_is_its_own_entry():
    table_of(groups_of(1, torch.float32), 3)
    table_of(groups_of(1, torch.float32, label0=50), 3)  # other labels
    table_of(groups_of(1, torch.float32, members=2), 3)  # other group sizes
    table_of(groups_of(1, torch.float32), 5)  # other k
    table_of(groups_of(1, torch.float32), 3, guard=False)
    table_of(groups_of(1, torch.bfloat16), 3)  # other dtype
    assert mk.table_cache_stats() == {"hits": 0, "misses": 6, "plan_rebuilds": 0}
    table_of(groups_of(2, torch.float32, label0=50), 3)
    assert mk.table_cache_stats()["hits"] == 1


def test_a_changed_view_stride_or_geometry_is_a_miss(monkeypatch):
    groups = groups_of(1, torch.float32)
    table_of(groups, 3)
    a = groups[2].a.transpose(1, 2).contiguous().transpose(1, 2)  # same shape, other strides
    strided = [groups[0], groups[1], groups[2]._replace(a=a)]
    table_of(strided, 3)
    assert mk.table_cache_stats()["misses"] == 2
    monkeypatch.setattr(mk, "ROWS_PER_BLOCK", 8)
    ht, _ = table_of(groups, 3)
    assert mk.table_cache_stats()["misses"] == 3
    with mk.table_cache(False):
        fresh, _ = table_of(groups, 3)
    assert ht.built.tail.tobytes() == fresh.built.tail.tobytes()


def test_the_cache_off_stores_and_counts_nothing():
    with mk.table_cache(False):
        assert not mk.table_cache_on()
        table_of(groups_of(1, torch.float32), 3)
        table_of(groups_of(2, torch.float32), 3)
    assert mk.table_cache_on()
    assert mk.table_cache_stats() == {"hits": 0, "misses": 0, "plan_rebuilds": 0}
    table_of(groups_of(3, torch.float32), 3)
    assert mk.table_cache_stats()["misses"] == 1


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(mk, "TABLE_CACHE_MAX", 3)
    for label in (1, 20, 40, 60, 80):
        table_of(groups_of(1, torch.float32, label0=label), 3)
    assert mk._cache_n <= 3 and sum(len(v) for v in mk._cache.values()) == mk._cache_n
    table_of(groups_of(1, torch.float32, label0=80), 3)  # the newest is still there
    assert mk.table_cache_stats()["hits"] == 1
    table_of(groups_of(1, torch.float32, label0=1), 3)  # the oldest is gone
    assert mk.table_cache_stats()["misses"] == 6


def test_an_invalid_group_fails_as_before_and_is_not_cached():
    groups = groups_of(1, torch.float32)
    bad = [groups[0], groups[1]._replace(ms=[1, 2])]  # two labels for three members
    with pytest.raises(ValueError, match="group 1"):
        mk.table_structure(bad, 3)
    assert mk.table_cache_stats() == {"hits": 0, "misses": 0, "plan_rebuilds": 0}


# ---- the pieces equal the functions they replace ----

@pytest.mark.parametrize("share", [True, False])
def test_the_cached_pieces_equal_the_plain_functions(share):
    groups = groups_of(1, torch.bfloat16)
    ht, _ = table_of(groups, 9, share=share)
    prods, plan = ht.prods, ht.plan
    base = 0x7F0000001000
    assert np.array_equal(ht.works(base), mk.works_array(prods, base))
    assert mk.scratch_bytes(prods) == ht.scratch_n
    starts = mk.block_starts(plan)
    assert np.array_equal(starts, ht.built.starts)
    blocks = mk.block_table(plan)
    t, r, e, b, rows = 0x10000, 0x20000, 0x30000, 0x40000, 0x50000
    at_w = ht.built.at_works
    old = mk.level_launches(t, t + at_w, rows, t + ht.built.at_reads, t + ht.built.at_entries,
                            t + ht.built.at_blocks, plan, 9, starts)
    new = ht.launches(t, rows, 9)
    assert [(bytes(a), g) for a, g in old] == [(bytes(a), g) for a, g in new]
    # the bytes: products, works, reads, entries and blocks at their offsets
    out = ht.to_bytes(base)
    assert out[:prods.nbytes].tobytes() == prods.tobytes()
    assert out[at_w:ht.built.end_works].tobytes() == mk.works_array(prods, base).tobytes()
    a = ht.built.at_reads
    assert out[a:a + plan.reads.nbytes].tobytes() == plan.reads.tobytes()
    a = ht.built.at_entries
    assert out[a:a + plan.entries.nbytes].tobytes() == plan.entries.tobytes()
    assert out[ht.built.at_blocks:].tobytes() == blocks.tobytes()
    assert len(out) == ht.total


# ---- the challenge layout ----

MS = [9, 3, 7, 3, 4, 9, 2]
WIDTHS = [40, 6, 65, 6, 6, 6, 40]
K = 3


def test_a_repeated_request_reuses_the_layout_with_the_same_offsets_and_bytes():
    src = cs.HostChallengeSource()
    fresh = cs.HostChallengeSource()
    pf1 = src.prefetch(H, MS, K, WIDTHS)
    pf2 = src.prefetch(H, tuple(MS), K, tuple(WIDTHS))  # equal request, tuples: the cached layout
    ref = fresh.prefetch(H, MS, K, WIDTHS)
    assert pf1.layout is pf2.layout and pf1.layout is not None
    assert pf2.offsets == ref.offsets and pf2.total == ref.total
    assert pf2.layout.layout == ref.layout.layout
    for pf in (pf1, pf2, ref):
        pf.future.result()
    for pf, source in ((pf2, src), (ref, fresh)):
        buf, offs, ev = source.draw(H, MS, K, WIDTHS, stream=None, prefetched=pf)
        assert ev is None
        for m, w, o in zip(MS, WIDTHS, offs):
            got = buf[o // 4:o // 4 + w * K].view(w, K)
            assert torch.equal(got.view(torch.int32),
                               challenge_matrix(H, m, K, w).view(torch.int32))
    b1, o1, _ = src.draw(H, MS, K, WIDTHS, stream=None, prefetched=pf2)
    b2, o2, _ = fresh.draw(H, MS, K, WIDTHS, stream=None, prefetched=ref)
    assert np.array_equal(o1, o2) and torch.equal(b1.view(torch.int32), b2.view(torch.int32))


def test_covers_compares_the_layout_slot_h_and_k():
    src = cs.HostChallengeSource()
    pf = src.prefetch(H, MS, K, WIDTHS)
    lay = src.find_layout(K, MS, WIDTHS)
    assert lay is pf.layout and pf.covers(H, MS, K, WIDTHS, lay)
    assert not pf.covers(bytes(range(32)), MS, K, WIDTHS, lay)
    assert not pf.covers(H, MS, K + 1, WIDTHS, lay)
    other = src.find_layout(K, MS[:3], WIDTHS[:3])
    assert other is None
    assert pf.covers(H, MS[:3], K, WIDTHS[:3], None)  # a subset: looked up member by member
    assert not pf.covers(H, [1], K, [6], None)
    src.prefetch(H, MS, K, WIDTHS)
    src.prefetch(H, MS, K, WIDTHS)  # the third refills pf's slot
    assert not pf.covers(H, MS, K, WIDTHS, lay)


def test_a_layout_is_found_only_for_an_equal_request():
    src = cs.HostChallengeSource()
    src.prefetch(H, MS, K, WIDTHS)
    assert src.find_layout(K, MS, WIDTHS) is not None
    assert src.find_layout(K + 1, MS, WIDTHS) is None
    assert src.find_layout(K, MS, WIDTHS[:-1] + [41]) is None
    assert src.find_layout(K, MS[:-1] + [3], WIDTHS) is None
    assert src.find_layout(K, [], []) is None and src.find_layout(K, MS, WIDTHS[:2]) is None


def test_the_layout_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(cs, "LAYOUT_CACHE_MAX", 2)
    src = cs.HostChallengeSource()
    for base in (0, 10, 20, 30):
        src.prefetch(H, [base + 1, base + 2], K, [6, 6])
    assert src._layouts_n <= 2
    assert src.find_layout(K, [31, 32], [6, 6]) is not None
    assert src.find_layout(K, [1, 2], [6, 6]) is None


def test_prefetch_errors_are_unchanged():
    src = cs.HostChallengeSource()
    with pytest.raises(ValueError, match="2 labels but 1 widths"):
        src.prefetch(H, [1, 2], K, [6])
    with pytest.raises(ValueError, match="no products"):
        src.prefetch(H, [], K, [])
    with pytest.raises(ValueError, match="width is below 1"):
        src.prefetch(H, [1], K, [0])
    assert src._layouts_n == 0
