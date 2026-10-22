"""FS38b's read plan (``measure_kernel.read_plan``) on hand-built tables.

``read_plan`` reads the table's metadata only, so the addresses here are fake. Every case asserts
the share groups (as sets of ``(product, phase)``), the levels, the launches and the byte counts
the plan states (``MEASURE_KERNEL_PLAN_DRAFT.md``, "FS38b")."""

import time

import numpy as np
import pytest

from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.measure_table import DT_BF16, DT_FP32, M_BIASED, M_GUARD

ITEMSIZE = {DT_FP32: 4, DT_BF16: 2}


# ---- building tables ----

def view(addr, rows, cols, rs=None, cs=1, dt=DT_FP32):
    """A matrix view as the table holds it (strides in elements; row-major unless given)."""
    return (addr, rows, cols, cols if rs is None else rs, cs, dt, 0)


def tview(addr, rows, cols, dt=DT_FP32):
    """The transpose of a row-major ``[cols, rows]`` matrix at ``addr``: strides (1, rows)."""
    return (addr, rows, cols, 1, rows, dt, 0)


def prod(a, b, p_addr, *, bias_addr=None, k=3, guard=True, dt=None):
    """One product ``a·b``: ``p`` is a fresh row-major ``[a.rows, b.cols]`` at ``p_addr``."""
    assert a[2] == b[1], "a.cols must equal b.rows"
    dt = a[5] if dt is None else dt
    return dict(a=a, b=b, p=view(p_addr, a[1], b[2], dt=dt), bias=(
        None if bias_addr is None else view(bias_addr, 1, b[2], dt=a[5])), k=k, guard=guard)


def table(prods):
    arr = np.zeros(len(prods), dtype=mk.MPROD_DTYPE)
    for i, d in enumerate(prods):
        for name in ("a", "b", "p"):
            arr[i][name] = d[name]
        if d["bias"] is not None:
            arr[i]["bias"] = d["bias"]
        arr[i]["r_addr"] = 0x7000_0000 + 0x1000 * i
        arr[i]["k"] = d["k"]
        arr[i]["flags"] = (M_GUARD if d["guard"] else 0) | (M_BIASED if d["bias"] is not None else 0)
    return arr


def nbytes(v):
    return int(v["rows"]) * int(v["cols"]) * ITEMSIZE[int(v["dtype"])]


def fs37_bytes(arr):
    """A, B and P once per product, and the bias when there is one."""
    total = 0
    for t in arr:
        total += nbytes(t["a"]) + nbytes(t["b"]) + nbytes(t["p"])
        if t["flags"] & M_BIASED:
            total += nbytes(t["bias"])
    return total


def key_addr(arr, t, p):
    return int(arr[t][("b", "a", "p")[p]]["addr"])


def groups_of(plan):
    """Every group as ``(level, [(prod, phase), ...], first, count)``, level by level."""
    out = []
    for lv in range(plan.n_levels):
        lo, hi = int(plan.level_off[lv]), int(plan.level_off[lv + 1])
        for r in plan.reads[lo:hi - 1]:
            e = plan.entries[int(r["entry0"]):int(r["entry0"]) + int(r["n_entries"])]
            out.append((lv, [(int(x["prod"]), int(x["phase"])) for x in e], int(r["first"]),
                        int(r["count"])))
    return out


def shares_of(plan):
    """The share groups (two entries or more) as a set of frozensets of ``(prod, phase)``."""
    return {frozenset(es) for _, es, _, _ in groups_of(plan) if len(es) > 1}


def check_partition(arr, plan):
    """Every node (product, phase) is in exactly one group, at the level ``node_level`` says."""
    seen = {}
    for lv, es, _, _ in groups_of(plan):
        for e in es:
            assert e not in seen, f"node {e} in two groups"
            seen[e] = lv
    n = len(arr)
    assert set(seen) == {(t, p) for t in range(n) for p in range(3)}
    assert plan.node_level.shape == (n, 3)
    for (t, p), lv in seen.items():
        assert plan.node_level[t, p] == lv, (t, p)
    assert plan.n_levels == int(plan.node_level.max()) + 1
    assert (plan.node_level[:, 1] > plan.node_level[:, 0]).all()
    assert (plan.node_level[:, 2] > plan.node_level[:, 1]).all()


def plan_of(prods):
    arr = table(prods) if isinstance(prods, list) else prods
    plan = mk.read_plan(arr)
    check_partition(arr, plan)
    assert plan.bytes_fs37 == fs37_bytes(arr)
    return arr, plan


def unshared(plan, arr):
    assert shares_of(plan) == set()
    assert plan.shared_groups == 0 and plan.bytes_saved == 0 and not plan.cycle
    assert plan.n_levels == 3
    assert np.array_equal(plan.node_level, np.tile([0, 1, 2], (len(arr), 1)))


X, W, G = 0x1000_0000, 0x2000_0000, 0x3000_0000
P0 = 0x4000_0000


# ---- one product ----

@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("dt", [DT_FP32, DT_BF16])
def test_one_product_has_no_shares_and_three_levels(biased, dt):
    n, q, w, k = 6, 9, 11, 3
    arr, plan = plan_of([prod(view(X, n, q, dt=dt), view(W, q, w, dt=dt), P0,
                              bias_addr=G if biased else None, k=k)])
    unshared(plan, arr)
    assert plan.n_levels == 3 and len(plan.level_off) == 4
    # one group per level, with the item counts of phases 0, 1, 2 (works_array's counts)
    assert [(lv, es, c) for lv, es, _, c in groups_of(plan)] == [
        (0, [(0, 0)], q + int(biased)), (1, [(0, 1)], n), (2, [(0, 2)], 5 + k)]


def test_one_product_is_not_a_cycle_and_reports_the_bytes():
    arr, plan = plan_of([prod(view(X, 4, 5, dt=DT_BF16), view(W, 5, 7, dt=DT_BF16), P0)])
    assert plan.bytes_fs37 == (4 * 5 + 5 * 7 + 4 * 7) * 2
    assert plan.bytes_saved == 0 and plan.shared_groups == 0 and plan.cycle is False


# ---- X read by a forward product and by its weight gradient ----

@pytest.mark.parametrize("dt", [DT_FP32, DT_BF16])
def test_x_read_by_the_forward_a_and_the_gradient_b(dt):
    n, q, w = 8, 12, 10
    # Y = X·Wᵀ ([n,q]·[q,w], W stored [w,q]); G = δYᵀ·X ([w,n]·[n,q]), δY stored [n,w]
    fwd = prod(view(X, n, q, dt=dt), tview(W, q, w, dt=dt), P0 + 0x100)
    grad = prod(tview(G, w, n, dt=dt), view(X, n, q, dt=dt), P0 + 0x200)
    arr, plan = plan_of([fwd, grad])
    assert shares_of(plan) == {frozenset({(0, 1), (1, 0)})}
    assert plan.shared_groups == 1
    assert plan.node_level[0].tolist() == [0, 1, 2]
    assert plan.node_level[1].tolist() == [1, 2, 3]
    assert plan.n_levels == 4 and not plan.cycle
    assert plan.bytes_saved == n * q * ITEMSIZE[dt]  # X's bytes once
    assert plan.bytes_fs37 == fs37_bytes(arr)
    (lv, es, first, count), = [g for g in groups_of(plan) if len(g[1]) == 2]
    assert lv == 1 and es == [(0, 1), (1, 0)] and count == n


def test_the_gradient_listed_first_gives_the_same_shares():
    n, q, w = 5, 6, 7
    fwd = prod(view(X, n, q), tview(W, q, w), P0 + 0x100)
    grad = prod(tview(G, w, n), view(X, n, q), P0 + 0x200)
    arr, plan = plan_of([grad, fwd])
    assert shares_of(plan) == {frozenset({(0, 0), (1, 1)})}
    assert plan.node_level[1].tolist() == [0, 1, 2] and plan.node_level[0].tolist() == [1, 2, 3]
    assert plan.bytes_saved == n * q * 4


# ---- one X as A of three forward products ----

def test_one_x_as_a_of_three_fp32_forward_products():
    n, q, w = 6, 16, 16
    prods = [prod(view(X, n, q), tview(W + 0x100_0000 * i, q, w), P0 + 0x1000 * i) for i in range(3)]
    arr, plan = plan_of(prods)
    assert shares_of(plan) == {frozenset({(0, 1), (1, 1), (2, 1)})}
    assert plan.shared_groups == 1
    assert plan.node_level.tolist() == [[0, 1, 2]] * 3 and plan.n_levels == 3
    assert plan.bytes_saved == 2 * n * q * 4
    assert plan.bytes_fs37 == fs37_bytes(arr)


# ---- member tables: Q̃ by S and δK̃, δO by dA and dV ----

def member_table(shared, n=7, d=8, members=4, base=0x5000_0000, dt=DT_BF16):
    """Per member t, ``fwd_t = shared_t·Kᵀ_t`` ([n,d]·[d,n], the shared matrix as ``A``) and
    ``bwd_t = Sᵀ_t·shared_t`` ([n,n]·[n,d], it as ``B``): the first is S (dA), the second δK̃
    (dV). The shared slices are consecutive views of one stack."""
    size = n * d * ITEMSIZE[dt]
    prods = []
    for t in range(members):
        sh = view(base + t * size, n, d, dt=dt)
        prods.append(prod(sh, tview(0x6000_0000 + t * size, d, n, dt=dt), 0x7000_0000 + t * 0x1_0000))
    for t in range(members):
        sh = view(base + t * size, n, d, dt=dt)
        prods.append(prod(tview(0x8000_0000 + t * 0x1_0000, n, n, dt=dt), sh,
                          0x9000_0000 + t * 0x1_0000))
    return prods


@pytest.mark.parametrize("name", ["Q-tilde by S and dK-tilde", "dO by dA and dV"])
def test_member_tables_share_one_group_per_member_pair(name):
    n, d, members, dt = 7, 8, 4, DT_BF16
    arr, plan = plan_of(member_table(True, n=n, d=d, members=members, dt=dt))
    want = {frozenset({(t, 1), (members + t, 0)}) for t in range(members)}
    assert shares_of(plan) == want
    assert plan.shared_groups == members
    assert plan.node_level[:members].tolist() == [[0, 1, 2]] * members
    assert plan.node_level[members:].tolist() == [[1, 2, 3]] * members
    assert plan.n_levels == 4
    assert plan.bytes_saved == members * n * d * 2


def test_member_slices_at_other_offsets_do_not_share():
    """The second set of slices starts one element later: the keys differ, nothing is shared."""
    n, d, members, dt = 7, 8, 3, DT_FP32
    prods = member_table(True, n=n, d=d, members=members, dt=dt)
    for t in range(members, 2 * members):
        a = list(prods[t]["b"])
        a[0] += 4
        prods[t]["b"] = tuple(a)
    arr, plan = plan_of(prods)
    unshared(plan, arr)


# ---- keys that differ ----

def test_a_two_sided_use_is_not_shared():
    """W as Bᵀ (transposed strides) in one product and as B in another."""
    q, w = 9, 9
    one = prod(view(X, 5, q), tview(W, q, w), P0 + 0x100)  # B = Wᵀ, strides (1, q)
    two = prod(view(X + 0x100_0000, 5, q), view(W, q, w), P0 + 0x200)  # B = W, strides (w, 1)
    arr, plan = plan_of([one, two])
    unshared(plan, arr)


@pytest.mark.parametrize("change", ["none", "row_stride", "col_stride", "rows", "cols", "dtype"])
def test_same_address_but_another_stride_shape_or_dtype_is_not_shared(change):
    n, q = 6, 8
    first = prod(view(X, n, q), tview(W, q, 5), P0 + 0x100)   # A = X [n, q]
    rows, cols, rs, cs, dt = n, q, q, 1, DT_FP32
    if change == "row_stride":
        rs = q + 3
    elif change == "col_stride":
        cs = 2
    elif change == "rows":
        rows = n - 1
    elif change == "cols":
        cols = q - 1
    elif change == "dtype":
        dt = DT_BF16
    other_b = view(X, rows, cols, rs, cs, dt)             # B of a second product, same address
    second = prod(view(G, 4, rows, dt=dt), other_b, P0 + 0x200)
    arr, plan = plan_of([first, second])
    if change == "none":
        assert shares_of(plan) == {frozenset({(0, 1), (1, 0)})}
        assert plan.bytes_saved == n * q * 4
    else:
        unshared(plan, arr)


# ---- a biased phase 0 is never shared ----

def test_a_biased_phase_0_is_never_shared_even_with_an_equal_key():
    n, q, w = 6, 8, 9
    first = prod(view(X, n, q), tview(W, q, w), P0 + 0x100)
    biased = prod(view(G, 4, n), view(X, n, q), P0 + 0x200, bias_addr=0xA000_0000)  # B = X
    arr, plan = plan_of([first, biased])
    unshared(plan, arr)
    # the unbiased twin shares, so the key was equal
    plain = prod(view(G, 4, n), view(X, n, q), P0 + 0x200)
    _, plan2 = plan_of([first, plain])
    assert shares_of(plan2) == {frozenset({(0, 1), (1, 0)})}
    # the biased phase 0 has one item more than B has rows
    assert [c for lv, es, _, c in groups_of(plan) if es == [(1, 0)]] == [n + 1]


def test_a_biased_product_still_shares_its_phase_1_and_the_other_side_of_the_view():
    n, q, w = 6, 8, 9
    one = prod(view(X, n, q), tview(W, q, w), P0 + 0x100, bias_addr=0xA000_0000)  # A = X, biased
    two = prod(view(X, n, q), tview(W + 0x100_0000, q, w), P0 + 0x200)             # A = X
    arr, plan = plan_of([one, two])
    assert shares_of(plan) == {frozenset({(0, 1), (1, 1)})}
    assert plan.bytes_saved == n * q * 4
    assert plan.bytes_fs37 == fs37_bytes(arr)


# ---- a cycle ----

def test_a_cycle_drops_every_share():
    """Product 1's A view is product 2's B view and product 2's A is product 1's B: each phase 1
    would have to run with the other's phase 0 and after its own phase 0."""
    n = w = 5
    q = 7
    u, v = view(X, n, q), view(W, q, w)
    one = prod(u, v, P0 + 0x100)   # U·V
    two = prod(v, u, P0 + 0x200)   # V·U  (A = V [q, w], B = U [n, q], w == n)
    arr, plan = plan_of([one, two])
    assert plan.cycle is True
    assert shares_of(plan) == set()
    assert plan.shared_groups == 0 and plan.bytes_saved == 0
    assert plan.n_levels == 3
    assert plan.node_level.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_a_cycle_beside_other_products_drops_their_shares_too():
    n = w = 5
    q = 7
    u, v = view(X, n, q), view(W, q, w)
    prods = [prod(u, v, P0 + 0x100), prod(v, u, P0 + 0x200),
             prod(view(G, 6, 4), tview(0x2800_0000, 4, 3), P0 + 0x300),
             prod(tview(0x2900_0000, 3, 6), view(G, 6, 4), P0 + 0x400)]  # an ordinary pair
    arr, plan = plan_of(prods)
    assert plan.cycle and plan.shared_groups == 0 and plan.bytes_saved == 0
    assert plan.node_level.tolist() == [[0, 1, 2]] * 4


# ---- the layout ----

def mixed_table():
    """Shared and unshared products of several shapes and dtypes, addresses out of order."""
    prods = []
    rng = np.random.default_rng(5)
    # fwd/grad pairs, the stored matrix at falling addresses
    for i in range(5):
        n, q, w = (int(x) for x in rng.integers(2, 20, 3))
        dt = DT_BF16 if i % 2 else DT_FP32
        x = 0x3000_0000 - 0x10_0000 * i
        prods.append(prod(view(x, n, q, dt=dt), tview(0x2000_0000 + 0x10_0000 * i, q, w, dt=dt),
                          0x5000_0000 + 0x1000 * len(prods)))
        prods.append(prod(tview(0x2800_0000 + 0x10_0000 * i, w, n, dt=dt), view(x, n, q, dt=dt),
                          0x5000_0000 + 0x1000 * len(prods)))
    # q, k, v at fp32, one X
    for i in range(3):
        prods.append(prod(view(0x1800_0000, 6, 10), tview(0x2100_0000 + 0x10_0000 * i, 10, 9),
                          0x5000_0000 + 0x1000 * len(prods)))
    prods += member_table(True, n=5, d=6, members=3)
    # singles, one biased, one with another k
    prods.append(prod(view(0x1100_0000, 4, 5), view(0x1200_0000, 5, 6), 0x5000_0000 + 0x1000 * len(prods),
                      bias_addr=0xB000_0000, k=5))
    prods.append(prod(view(0x1300_0000, 3, 3, dt=DT_BF16), view(0x1400_0000, 3, 3, dt=DT_BF16),
                      0x5000_0000 + 0x1000 * len(prods), k=1))
    return table(prods)


def test_the_layout_levels_groups_entries_and_counts():
    arr = mixed_table()
    plan = mk.read_plan(arr)
    check_partition(arr, plan)
    assert plan.shared_groups > 5 and not plan.cycle
    assert plan.reads.dtype == mk.READ_DTYPE and plan.entries.dtype == mk.ENTRY_DTYPE
    assert plan.level_off.dtype == np.uint64
    assert len(plan.level_off) == plan.n_levels + 1
    assert plan.level_off[0] == 0 and plan.level_off[-1] == len(plan.reads)
    assert len(plan.entries) == 3 * len(arr)  # every node once
    works = mk.works_array(arr, 4096)
    fs37_items = int(works["start"][-1].sum())  # the three phases' item totals
    n_items = {0: lambda t: int(t["b"]["rows"]) + int(bool(t["flags"] & M_BIASED)),
               1: lambda t: int(t["a"]["rows"]), 2: lambda t: 5 + int(t["k"])}
    total_counts = total_weighted = dropped = 0
    for lv in range(plan.n_levels):
        lo, hi = int(plan.level_off[lv]), int(plan.level_off[lv + 1])
        groups, sentinel = plan.reads[lo:hi - 1], plan.reads[hi - 1]
        assert len(groups) >= 1
        running, prev = 0, None
        for r in groups:
            es = [(int(x["prod"]), int(x["phase"]))
                  for x in plan.entries[int(r["entry0"]):int(r["entry0"]) + int(r["n_entries"])]]
            assert es == sorted(es) and len(es) >= 1               # entries ascending
            addrs = {key_addr(arr, t, p) for t, p in es}
            assert len(addrs) == 1
            order = (addrs.pop(), es[0])
            assert prev is None or prev <= order, (lv, prev, order)  # address, then first entry
            prev = order
            assert int(r["first"]) == running                       # running item count
            assert all(n_items[p](arr[t]) == int(r["count"]) for t, p in es)
            assert all(plan.node_level[t, p] == lv for t, p in es)
            running += int(r["count"])
            total_counts += int(r["count"])
            total_weighted += int(r["count"]) * len(es)
            dropped += (len(es) - 1) * int(r["count"])
        assert int(sentinel["first"]) == running                    # the level's item total
        assert int(sentinel["count"]) == 0 and int(sentinel["entry0"]) == 0
        assert int(sentinel["n_entries"]) == 0
    # the identity: Σ count·entries = FS37's item total; Σ count = FS37's total less the shared
    assert total_weighted == fs37_items
    assert total_counts == fs37_items - dropped
    assert dropped > 0


def test_groups_are_ordered_by_address_whatever_the_product_order():
    """Three unshared products at descending addresses: each phase-0 group's level holds them
    by ascending B address, ties by (product, phase)."""
    prods = [prod(view(0x1000_0000 + 0x10_0000 * (3 - i), 4, 5),
                  view(0x2000_0000 + 0x10_0000 * (3 - i), 5, 6), 0x5000_0000 + 0x1000 * i)
             for i in range(3)]
    arr, plan = plan_of(prods)
    lv0 = [es[0] for lv, es, _, _ in groups_of(plan) if lv == 0]
    assert lv0 == [(2, 0), (1, 0), (0, 0)]  # B addresses ascend as the product index falls
    firsts = [f for lv, es, f, _ in groups_of(plan) if lv == 0]
    assert firsts == [0, 5, 10]


# ---- speed ----

def test_read_plan_on_10000_members_takes_under_a_second():
    members, n, d = 5000, 16, 8
    arr = np.zeros(2 * members, dtype=mk.MPROD_DTYPE)
    t = np.arange(members, dtype=np.uint64)
    size = np.uint64(n * d * 2)

    def fill(sl, name, addr, rows, cols, rs, cs):
        v = arr[name][sl]
        v["addr"], v["rows"], v["cols"] = addr, rows, cols
        v["row_stride"], v["col_stride"], v["dtype"] = rs, cs, DT_BF16

    s_, b_ = slice(0, members), slice(members, 2 * members)
    shared = np.uint64(0x5000_0000) + t * size
    fill(s_, "a", shared, n, d, d, 1)                                  # S_t: Q̃_t as A
    fill(s_, "b", np.uint64(0x6000_0000) + t * size, d, n, 1, d)
    fill(s_, "p", np.uint64(0x7000_0000) + t * np.uint64(n * n * 2), n, n, n, 1)
    fill(b_, "a", np.uint64(0x8000_0000) + t * np.uint64(n * n * 2), n, n, 1, n)
    fill(b_, "b", shared, n, d, d, 1)                                  # δK̃_t: Q̃_t as B
    fill(b_, "p", np.uint64(0x9000_0000) + t * size, n, d, d, 1)
    arr["k"] = 3
    arr["flags"] = M_GUARD
    mk.read_plan(arr[:8].copy())  # warm the imports and the caches
    t0 = time.perf_counter()
    plan = mk.read_plan(arr)
    took = time.perf_counter() - t0
    assert took < 1.0, f"read_plan took {took:.2f} s on {len(arr)} products"
    assert plan.shared_groups == members and plan.n_levels == 4 and not plan.cycle
    assert plan.bytes_saved == members * n * d * 2
