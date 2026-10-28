"""FS50: check 5's per-computation cache (``HOST_SIDE_PLAN_DRAFT.md`` §2.1, §2.3).

What check 5 derives from the declared computation and the unit schedule alone is built once
(``checks._c5_static``). Every case compares the cache with a fresh build, or a step judged with
the cache with the same step judged without it (``checks.check5_static_cache(False)``, which
runs the code as it was before the cache): every ``ProductStat`` field bit for bit, the
rejection and its message, and ``StepState.run_rest``.
"""

import dataclasses
import gc

import numpy as np
import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.verifier import checks
from verification.verifier.bands import Bands, product_class
from verification.verifier.matmul_check.freivalds import ROW

from tests.verification.decoder_helpers import SWITCHES, switch_id, tiny_config
from tests.verification.computation.test_leaf_sizes import smollm2_config
from tests.verification.verifier import test_judge_window as jw
from tests.verification.verifier import test_unit_schedule as us

SMALL = 8


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", SMALL)
    return SMALL


def _tiny(switch, blocks):
    family, tied = switch
    return DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                              compute_dtype=torch.float32, device="cpu", head_blocks=blocks)


def _smollm2():
    return DecoderComputation(smollm2_config(), n_s=4, n=128, eta=1e-3,
                              compute_dtype=torch.float32, device="cpu")


def _assert_structure(c, window_size):
    """Every cached structure equals the function that built it, run again."""
    static = checks._Check5Static(c)
    runs = list(checks._member_runs(c.products))
    assert [tuple(r) for r in runs] == list(static.runs)
    assert all(a is b or a == b for a, b in zip(static.runs, runs))
    groups = [checks._measure_groups(run) for run in runs]
    assert [[list(g) for g in gs] for gs in static.groups] == groups
    g0, g = [], 0
    for gs in groups:
        g0.append(g)
        g += len(gs)
    assert list(static.group0) == g0
    assert [list(w) for w in static.windows] == checks._judge_windows(runs)
    schedule = c.unit_schedule
    sched = static.for_schedule(schedule)
    steps = checks._schedule_runs(runs, schedule)
    assert [list(s) for s in sched.steps] == steps
    for step, (ms, widths) in zip(steps, sched.prefetch):
        specs = [spec for i in step for spec in runs[i]]
        assert ms == tuple(s.m for s in specs) and widths == tuple(s.width for s in specs)
        assert isinstance(ms, tuple) and isinstance(widths, tuple)
    table = c.product_classes
    covered = []
    for win, idx in zip(static.judge, static.windows):
        specs = [s for i in idx for s in runs[i]]
        covered += [s.m for s in specs]
        assert win.specs == specs and win.m == [s.m for s in specs]
        assert win.name == [s.name for s in specs]
        assert win.cls == [table[s.m - 1] for s in specs] == [product_class(c, s) for s in specs]
        assert win.q == [s.q for s in specs]
        assert np.array_equal(win.q_arr, np.array([s.q for s in specs]))
        assert np.array_equal(win.n_arr, np.array([s.a_shape[0] for s in specs]))
        assert np.array_equal(win.biased, np.array([s.bias is not None for s in specs]))
        assert [static.classes[k] for k in win.codes] == win.cls
    assert covered == [s.m for s in c.products]  # the windows cover every product once
    return static


@pytest.mark.parametrize("blocks", us.BLOCKS)
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_the_cached_structures_equal_a_fresh_build_on_the_tiny_decoders(switch, blocks,
                                                                        small_window):
    c = _tiny(switch, blocks)
    _assert_structure(c, SMALL)
    # one window holding everything, and the default size, are structures too
    _assert_structure(c, SMALL)


def test_the_cached_structures_equal_a_fresh_build_on_smollm2s_computation():
    c = _smollm2()
    static = _assert_structure(c, checks._JUDGE_WINDOW)
    assert sum(len(r) for r in static.runs) == c.M
    assert len(static.windows) > 1


# ---- the canonical permutation ----------------------------------------------------------------

def _batch(static, w, rng, edits=None):
    """Window ``w``'s runs with rows in a random row order (as the replay's blocks give), each
    product's row holding its ``m`` in ``p_norm``; ``edits[m]`` replaces entries of ``m``'s row."""
    edits = edits or {}
    batch = []
    for i in static.windows[w]:
        run = static.runs[i]
        rows = np.array([jw.GOOD] * len(run))
        for pos, spec in enumerate(run):
            rows[pos, ROW.index("p_norm")] = float(spec.m)
            for col, v in edits.get(spec.m, {}).items():
                rows[pos, ROW.index(col) if isinstance(col, str) else col] = v
        order = [int(x) for x in rng.permutation(len(run))]
        stored = np.empty_like(rows)
        for j, pos in enumerate(order):
            stored[j] = rows[pos]
        batch.append(checks._Rows(run, order, torch.from_numpy(stored)))
    return batch


def _same_columns(a, b):
    assert a[0] == b[0]
    for f in (x.name for x in dataclasses.fields(a[1])):
        x, y = getattr(a[1], f), getattr(b[1], f)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), f


def test_the_cached_permutation_gives_the_canonical_columns_and_follows_a_changed_order(
        small_window):
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    rng = np.random.default_rng(0)
    w = next(w for w, idx in enumerate(static.windows) if len(idx) > 2)
    win = static.judge[w]
    for _ in range(3):  # a second and third order: the cached permutation is rebuilt
        batch = _batch(static, w, rng)
        ctx = jw._ctx()
        got = checks._finish_columns(batch, ctx, None, win)
        want = checks._finish_columns(batch, jw._ctx())
        _same_columns(got, want)
        assert [round(x) for x in got[1].p_norm] == win.m  # canonical order
    again = checks._finish_columns(batch, jw._ctx(), None, win)  # the same order twice
    _same_columns(again, want)


def test_a_batch_that_is_not_the_window_is_not_served_from_it(small_window):
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    rng = np.random.default_rng(1)
    w = next(w for w, idx in enumerate(static.windows) if len(idx) > 2)
    batch = _batch(static, w, rng)
    assert static.window(w, batch) is static.judge[w]
    assert static.window(w, batch[:-1]) is None
    cut = [batch[0]._replace(specs=batch[0].specs[:-1])] + batch[1:]
    assert static.window(w, cut) is None
    assert static.window(w, [batch[0]._replace(rows=None)] + batch[1:]) is None
    assert static.window(len(static.windows), batch) is None


# ---- judging a window, cached and not ---------------------------------------------------------

def _judge(c, static, w, batch, bands, **ctx_kw):
    ctx = jw._ctx(**ctx_kw)
    rej = checks._judge_window(c, batch, ctx, bands, None, static, w)
    return rej, ctx


def _same_judging(c, static, w, batch, bands, **ctx_kw):
    got, a = _judge(c, static, w, batch, bands, **ctx_kw)
    want, b = _judge(c, None, None, batch, bands, **ctx_kw)
    assert got == want
    assert repr(a.state.run_rest) == repr(b.state.run_rest)
    assert repr(a.stats.products) == repr(b.stats.products)  # every field, NaN included
    assert len(a.stats.products) == len(b.stats.products)
    return got, a


BANDS = Bands(tau=8.0, kappa_max=100.0)


@pytest.mark.parametrize("where", ["first", "middle", "last"])
@pytest.mark.parametrize("edit, text", [
    ({"nu": 1000.0}, "cancellation factor"),
    ({"p_norm": float("inf")}, "non-finite"),
    ({6: 1.0}, "challenge j=2"),
])
def test_a_planted_failing_product_is_rejected_alike(where, edit, text, small_window):
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    rng = np.random.default_rng(2)
    w = next(w for w, idx in enumerate(static.windows) if len(idx) > 2)
    ms = static.judge[w].m
    m = {"first": ms[0], "middle": ms[len(ms) // 2], "last": ms[-1]}[where]
    batch = _batch(static, w, rng, {m: edit})
    rej, ctx = _same_judging(c, static, w, batch, BANDS)
    assert rej is not None and text in rej.detail and f"P_{m} " in rej.detail
    assert len(ctx.stats.products) == ms.index(m) + 1
    assert [s.m for s, _ in ctx.state.run_rest] == ms[ms.index(m) + 1:]


def test_an_honest_window_is_recorded_alike(small_window):
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    rng = np.random.default_rng(3)
    for w in range(len(static.windows)):
        rej, ctx = _same_judging(c, static, w, _batch(static, w, rng), BANDS)
        assert rej is None and [p.m for p in ctx.stats.products] == static.judge[w].m


@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("judge", [True, False])
def test_the_class_ceiling_is_the_same_per_class(guard, judge, small_window):
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    rng = np.random.default_rng(4)
    w = next(w for w, idx in enumerate(static.windows) if len(idx) > 2)
    cls = static.judge[w].cls
    target = cls[len(cls) // 2]
    bands = Bands(tau=8.0, kappa_max=100.0, kappa_classes={target: 1.0})
    # every row has p_abs1 = 1 and nu = 1.5: kappa 1.5, over the one class's ceiling only
    batch = _batch(static, w, rng, {m: {"nu": 1.5} for m in static.judge[w].m})
    rej, ctx = _same_judging(c, static, w, batch, bands, kappa_guard=guard, judge=judge)
    assert (rej is not None) == (judge and guard)
    if rej is not None:
        failing = static.judge[w].m[cls.index(target)]
        assert f"P_{failing} " in rej.detail and f"[{target}]" in rej.detail


def test_the_kappa_ceiling_is_built_once_per_bands_values():
    c = _tiny(("llama", True), 1)
    static = checks._Check5Static(c)
    bands = Bands(tau=8.0, kappa_max=100.0, kappa_classes={static.classes[0]: 1.5})
    first = static.kappa_ceiling(bands)
    assert first is static.kappa_ceiling(bands)
    assert first.tolist() == [bands.kappa_for(k) for k in static.classes]
    other = Bands(tau=8.0, kappa_max=100.0, kappa_classes={static.classes[0]: 1.5})
    assert static.kappa_ceiling(other) is first  # equal values: no rebuild
    changed = Bands(tau=8.0, kappa_max=100.0, kappa_classes={static.classes[1]: 2.0})
    again = static.kappa_ceiling(changed)  # other values: rebuilt
    assert again is not first and again.tolist() == [changed.kappa_for(k) for k in static.classes]
    assert static.kappa_ceiling(Bands(tau=8.0, kappa_max=50.0))[0] == 50.0


# ---- whole steps -----------------------------------------------------------------------------

def _step(setup, cache, **kw):
    with checks.check5_static_cache(cache):
        return us.run_check5(setup, **kw)


def _same_step(a, b):
    us.assert_same(a, b)


@pytest.mark.parametrize("blocks", us.BLOCKS)
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_an_honest_step_records_the_same_with_and_without_the_cache(switch, blocks):
    setup = us.proved(switch, blocks)
    on, off = _step(setup, True), _step(setup, False)
    assert on.rejection is None and len(on.stats) == setup.c.M
    _same_step(on, off)
    again = _step(setup, True)  # a second step reads the cache
    _same_step(again, off)


@pytest.mark.parametrize("where", ["w0-first", "w0-last", "mid-first", "mid-last", "last-first",
                                   "m=M"])
@pytest.mark.parametrize("canonical", [False, True])
def test_a_planted_failing_product_is_rejected_alike_in_a_step(where, canonical, small_window,
                                                               monkeypatch):
    setup = us.proved(us.CHEAP, 2)
    w = us._windows(setup.c, small_window)
    if where == "m=M":
        m = setup.c.M
    else:
        idx, end = where.split("-")
        m = w[{"w0": 0, "mid": len(w) // 2, "last": len(w) - 1}[idx]][0 if end == "first" else 1]
    leaves = us._flipped(setup, m)
    if canonical:
        monkeypatch.setattr(us.DecoderReplay, "unit_schedule", lambda self: None, raising=False)
    on = _step(setup, True, leaves=leaves)
    off = _step(setup, False, leaves=leaves)
    assert off.rejection is not None and off.rejection[2].startswith(f"P_{m} ")
    _same_step(on, off)


@pytest.mark.parametrize("first_only", [True, False])
def test_a_serving_error_is_judged_alike(first_only, small_window):
    """An error while serving: the unit pass raises, check 5 reruns canonically (on a fresh
    replay, or fails the same way again), with and without the cache."""
    setup = us.proved(us.CHEAP, 2)
    m = setup.c.M // 2
    out = []
    for cache in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            us.ServingFault(mp, m, lambda: RuntimeError("boom"), first_only=first_only)
            out.append(_step(setup, cache))
    on, off = out
    _same_step(on, off)


# ---- the cache's keys --------------------------------------------------------------------------

def test_a_second_computation_gets_its_own_entry():
    a, b = _tiny(("llama", True), 1), _tiny(("llama", True), 1)
    ea, eb = checks._c5_static(a), checks._c5_static(b)
    assert ea is not eb and ea is checks._c5_static(a) and eb is checks._c5_static(b)
    assert ea.runs == eb.runs  # equal structures, separate entries


def test_the_entry_goes_with_its_computation():
    gc.collect()  # earlier tests' dead computations leave first, so ``before`` is stable
    before = len(checks._STATIC)
    c = _tiny(("qwen2", False), 1)
    checks._c5_static(c)
    assert len(checks._STATIC) == before + 1
    del c
    gc.collect()
    assert len(checks._STATIC) == before


def test_a_computation_that_cannot_be_a_weak_key_keeps_the_entry_on_itself():
    class Unhashable(DecoderComputation):
        __hash__ = None  # type: ignore[assignment]

    c = Unhashable(tiny_config(family="llama", tied=True), n_s=2, n=12, eta=1e-3,
                   compute_dtype=torch.float32, device="cpu", head_blocks=1)
    entry = checks._c5_static(c)
    assert entry is not None and entry is checks._c5_static(c)
    assert getattr(c, checks._STATIC_ATTR) is entry


def test_the_cache_is_bypassed_and_comes_back():
    c = _tiny(("llama", True), 1)
    with checks.check5_static_cache(False):
        assert checks._c5_static(c) is None
        with checks.check5_static_cache(True):
            assert checks._c5_static(c) is not None
        assert checks._c5_static(c) is None
    assert checks._c5_static(c) is not None


def test_a_window_or_group_size_change_rebuilds_the_entry(monkeypatch):
    c = _tiny(("llama", True), 1)
    first = checks._c5_static(c)
    assert checks._c5_static(c) is first
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", SMALL)
    second = checks._c5_static(c)
    assert second is not first and second.windows != first.windows
    monkeypatch.setattr(checks, "_GROUP_BYTES", 1 << 12)
    assert checks._c5_static(c) is not second


def test_an_unequal_schedule_rebuilds_and_an_equal_one_does_not():
    c = _tiny(("llama", True), 2)
    static = checks._Check5Static(c)
    schedule = c.unit_schedule
    first = static.for_schedule(schedule)
    assert static.schedule_builds == 1 and static.for_schedule(schedule) is first
    equal = tuple(tuple(s) for s in schedule)  # equal, not the same object
    assert equal is not schedule and static.for_schedule(equal) is first
    assert static.schedule_builds == 1 and first.schedule is equal
    swapped = (schedule[1], schedule[0], *schedule[2:])  # the first two head steps swapped
    second = static.for_schedule(swapped)
    assert static.schedule_builds == 2 and second is not first and second.steps != first.steps
    assert [list(s) for s in second.steps] == checks._schedule_runs(static.runs, swapped)
    assert static.for_schedule(schedule) is not second and static.schedule_builds == 3


def test_a_schedule_that_misses_a_product_raises_and_caches_nothing():
    c = _tiny(("llama", True), 1)
    static = checks._Check5Static(c)
    good = static.for_schedule(c.unit_schedule)
    broken = c.unit_schedule[:-1]
    with pytest.raises(RuntimeError, match="unit schedule misses"):
        static.for_schedule(broken)
    assert static.schedule is good


def test_the_entry_holds_no_prover_data():
    """Everything in it is specs, ints, strings, tuples and arrays of them."""
    c = _tiny(("llama", True), 1)
    static = checks._Check5Static(c)
    static.for_schedule(c.unit_schedule)
    seen = []

    def walk(x, depth=0):
        assert depth < 6
        if isinstance(x, (tuple, list)):
            for y in x:
                walk(y, depth + 1)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k, depth + 1), walk(v, depth + 1)
        else:
            seen.append(type(x).__name__)

    for name, value in vars(static).items():
        if name in ("judge", "schedule"):
            continue
        walk(value)
    for win in static.judge:
        for name, value in vars(win).items():
            assert not isinstance(value, torch.Tensor), name
    assert not any(t in ("Tensor", "bytes") for t in seen)
    assert all(not isinstance(v, torch.Tensor) for v in vars(static.schedule).values())
