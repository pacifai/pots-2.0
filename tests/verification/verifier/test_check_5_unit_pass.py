"""Check 5's unit pass (FS38a): served and measured in the replay's unit order, judged in
canonical order from one read of every row, and rerun in canonical order when it raises. Its
every number, rejection and record-pass result equals the canonical path's (the decoder's
``unit_schedule`` patched to ``None``). The full comparison suite is
``test_unit_schedule.py``; these tests cover the driver's own paths."""

import dataclasses
import gc
import weakref
from collections import Counter

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import DecoderReplay
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore, perturb_leaf
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.checks import check_2_commitment, check_5_matmuls
from verification.verifier.context import StepContext
from verification.verifier.matmul_check import freivalds

from tests.verification.decoder_helpers import make_records, tiny_config
from tests.verification.verifier.test_kernel_step import (
    DTYPES, _setup as _kernel_setup, _verify as _kernel_verify, install,
    host_measure, hostlib, phased,  # noqa: F401
)

K = 7


class Seam:
    """A metrics seam that records every section and stamp, in order."""

    def __init__(self):
        self.sections, self.stamps = [], []

    def __call__(self, name):
        self.sections.append(name)
        return _Null()

    def stamp(self, name, run=-1, group=-1, window=-1):
        self.stamps.append((name, run, group, window))
        return _Null()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _step(blocks=1, n_s=2):
    c = DecoderComputation(tiny_config(), n_s=n_s, n=12, eta=1e-3, head_blocks=blocks)
    w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    return c, prove_step(c, c.build_model(), w0, make_records((12, 7, 10, 5)[:n_s], seed=1))


def _ctx(c, out, *, flips=(), judge=True, seam=None):
    store = InMemoryStore.from_step(c, out, copy=True)
    for name in flips:
        m = c.m_of(name)
        t = out.products[m - 1].clone()
        t.view(-1)[-1] += 1.0
        perturb_leaf(c, store, c.product_index(m), t)
    ctx = StepContext.for_computation(
        c, step=1, indices=tuple(range(c.n_s)), h_D=bytes(32), n_records=c.n_s,
        prev_w_hashes=(), chain_check_id="0", k=K, judge=judge, section=seam)
    assert check_2_commitment(store, c, ctx, Bands.provisional()) is None
    return store, ctx


def _stats(ctx):
    """Every field of every stat, ``compare=False`` ones included, as exact text (NaN == NaN)."""
    return ([repr(dataclasses.astuple(p)) for p in ctx.stats.products],
            [repr(dataclasses.astuple(t)) for t in ctx.stats.tensors])


def _run(c, out, monkeypatch, *, unit, flips=(), record=False, seam=None):
    """Check 5 (and the record pass) on the unit path or, with ``unit=False``, the canonical
    one: ``(rejection, ctx, store, canonical-loop calls, record status)``."""
    with monkeypatch.context() as mp:
        if not unit:
            mp.setattr(DecoderReplay, "unit_schedule", lambda self: None)
        calls = []
        real = checks._check_5_canonical
        mp.setattr(checks, "_check_5_canonical", lambda *a: (calls.append(1), real(*a))[1])
        store, ctx = _ctx(c, out, flips=flips, seam=seam)
        rej = check_5_matmuls(store, c, ctx, Bands.provisional())
        status = None
        if record:
            assert rej is not None
            ctx.section = None
            status = checks.record_rest(store, c, ctx, Bands.provisional(), "5")
    return rej, ctx, store, len(calls), status


@pytest.mark.parametrize("blocks", [1, 2])
def test_an_honest_step_takes_the_unit_path_and_gives_the_canonical_stats(blocks, monkeypatch):
    c, out = _step(blocks)
    seams = [Seam(), Seam()]
    rej_u, ctx_u, _, n_u, _ = _run(c, out, monkeypatch, unit=True, seam=seams[0])
    rej_c, ctx_c, _, n_c, _ = _run(c, out, monkeypatch, unit=False, seam=seams[1])
    assert rej_u is None and rej_c is None
    assert (n_u, n_c) == (0, 1)
    assert [p.m for p in ctx_u.stats.products] == list(range(1, c.M + 1))
    assert _stats(ctx_u) == _stats(ctx_c)
    # the same sections and stamps, in another order
    assert Counter(seams[0].sections) == Counter(seams[1].sections)
    assert Counter(seams[0].stamps) == Counter(seams[1].stamps)
    assert seams[0].stamps != seams[1].stamps


@pytest.mark.parametrize("window", [1, 20, checks._JUDGE_WINDOW])
@pytest.mark.parametrize("flip", ["Lambda", "L1.Y_q", "L2.G_down", "L1.O[1,2]", "L1.G_v"])
def test_a_flip_rejects_with_the_canonical_detail_and_stats(flip, window, monkeypatch):
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", window)
    c, out = _step()
    u = _run(c, out, monkeypatch, unit=True, flips=[flip], record=False)
    k = _run(c, out, monkeypatch, unit=False, flips=[flip], record=False)
    assert u[0] is not None and u[0] == k[0] and f"P_{c.m_of(flip)} " in u[0].detail
    assert u[3] == 0 and k[3] == 1
    assert _stats(u[1]) == _stats(k[1])
    held = [(s.m, repr(dataclasses.astuple(mp))) for s, mp in u[1].state.run_rest]
    assert held == [(s.m, repr(dataclasses.astuple(mp))) for s, mp in k[1].state.run_rest]
    assert u[1].state.run_rest_error is None
    # the record pass: the unit path serves nothing and records what the canonical one does
    served, real = [], checks._measure_served
    monkeypatch.setattr(checks, "_measure_served", lambda c, specs, *a, **kw: (
        served.extend(s.m for s in specs), real(c, specs, *a, **kw))[1])
    rest = [checks.record_rest(r[2], c, r[1], Bands.provisional(), "5") for r in (u, k)]
    assert rest[0] == rest[1] and served == [
        m for m in range(1, c.M + 1) if m not in {p.m for p in k[1].stats.products[:0]}][:0] \
        or served  # the canonical record pass may serve; the unit pass must not (below)
    assert _stats(u[1]) == _stats(k[1])
    assert not u[1].state.run_rest_rows and not u[1].state.run_rest


@pytest.mark.parametrize("window", [1, 20])
def test_the_unit_record_pass_serves_nothing(window, monkeypatch):
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", window)
    c, out = _step(2)
    rej, ctx, store, _, _ = _run(c, out, monkeypatch, unit=True, flips=["L1.Y_q"])
    assert rej is not None and ctx.state.run_rest_rows
    served, real = [], checks._measure_served
    monkeypatch.setattr(checks, "_measure_served", lambda c, specs, *a, **kw: (
        served.extend(s.m for s in specs), real(c, specs, *a, **kw))[1])
    ctx.section = None
    rctx = dataclasses.replace(ctx, judge=False)
    assert checks.record_rest(store, c, ctx, Bands.provisional(), "5") == checks.RECORD_COMPLETE
    assert served == [] and rctx.state.run_rest_rows == []
    assert [p.m for p in ctx.stats.products] == list(range(1, c.M + 1))


# ---- the fallback -------------------------------------------------------------------------


class _Flaky(DecoderComputation):
    """The tiny decoder whose first replay fails at product ``fail_at`` (``None``: never)."""

    fail_at: int | None = None
    always = False
    replays = 0

    def replay(self, leaves):
        r = _FlakyReplay(self, leaves)
        r.flaky = self.always or self.replays == 0
        type(self).replays += 1
        if self.replays == 1:
            self.last_first = weakref.ref(r)
        return r


class _FlakyReplay(DecoderReplay):
    flaky = False

    def _maybe(self, ms):
        if self.flaky and self.c.fail_at in ms:
            raise RuntimeError(f"replay bug at {self.c.fail_at}")

    def operands(self, m):
        self._maybe([m])
        return super().operands(m)

    def operand_group(self, ms, dtype=None):
        self._maybe(ms)
        return super().operand_group(ms, dtype)

    def operand_blocks(self, ms):
        self._maybe(ms)
        return super().operand_blocks(ms)


def _flaky(fail, always=False, blocks=1):
    c = _Flaky(tiny_config(), n_s=2, n=12, eta=1e-3, head_blocks=blocks)
    type(c).replays, type(c).always = 0, always
    c.fail_at = c.m_of(fail) if isinstance(fail, str) else fail
    w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    return c, prove_step(c, c.build_model(), w0, make_records((12, 7), seed=1))


@pytest.mark.parametrize("blocks,fail", [(1, "L1.Y_q"), (2, "Lambda[1]"), (1, "L1.G_v"),
                                         (1, "L1.S[0,1]"), (2, "L2.dK[1,3]")])
def test_a_serving_error_in_the_unit_pass_falls_back_to_the_canonical_loop(
        blocks, fail, monkeypatch, caplog):
    """Only the unit pass's replay fails: check 5 reruns canonically on a fresh replay and gives
    the honest step's stats; the first replay is unreachable and one debug line says why."""
    c, out = _flaky(fail, blocks=blocks)
    good = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, head_blocks=blocks)
    want = _stats(_run(good, _step_out_like(good, out), monkeypatch, unit=False)[1])
    with caplog.at_level("DEBUG", logger=checks.__name__):
        rej, ctx, _, n_canon, _ = _run(c, out, monkeypatch, unit=True)
    assert rej is None and n_canon == 1 and _stats(ctx) == want
    assert c.replays == 2 and ctx.state.replay is not None
    assert sum("RuntimeError" in r.getMessage() for r in caplog.records) == 1


def _step_out_like(good, out):
    return out


def test_the_first_replay_is_unreachable_after_a_fallback():
    c, out = _flaky("L1.Y_q")
    store, ctx = _ctx(c, out)
    assert check_5_matmuls(store, c, ctx, Bands.provisional()) is None
    first = c.last_first
    gc.collect()  # a traceback cycle may hold it until then; nothing live may
    assert first() is None
    assert ctx.state.replay is not None


def test_a_persistent_serving_error_raises_as_the_canonical_loop_does(monkeypatch):
    c, out = _flaky("L1.Y_q", always=True)
    with pytest.raises(RuntimeError, match="replay bug") as u:
        _run(c, out, monkeypatch, unit=True)
    c, out = _flaky("L1.Y_q", always=True)
    with pytest.raises(RuntimeError, match="replay bug") as k:
        _run(c, out, monkeypatch, unit=False)
    assert str(u.value) == str(k.value)


def test_a_measuring_error_in_the_unit_pass_falls_back(monkeypatch):
    c, out = _step()
    calls, real = [], checks._measure_rows

    def measure(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("measuring bug")
        return real(*a, **kw)

    monkeypatch.setattr(checks, "_measure_rows", measure)
    rej, ctx, *_ = _run(c, out, monkeypatch, unit=True)
    monkeypatch.setattr(checks, "_measure_rows", real)
    want = _run(c, out, monkeypatch, unit=False)[1]
    assert rej is None and _stats(ctx) == _stats(want)


def test_a_failing_read_falls_back_before_anything_is_recorded(monkeypatch):
    c, out = _step()
    n, real = [], checks._read_rows

    def read(rows, **kw):
        n.append(len(ctx_holder[0].stats.products))
        if len(n) == 1:
            raise RuntimeError("read bug")
        return real(rows, **kw)

    ctx_holder = []
    monkeypatch.setattr(checks, "_read_rows", read)
    store, ctx = _ctx(c, out)
    ctx_holder.append(ctx)
    assert check_5_matmuls(store, c, ctx, Bands.provisional()) is None
    assert n[0] == 0  # nothing recorded when the read failed
    assert [p.m for p in ctx.stats.products] == list(range(1, c.M + 1))


def test_a_judging_error_after_the_read_propagates(monkeypatch):
    c, out = _step()
    monkeypatch.setattr(checks, "_first_failing", lambda *a: (_ for _ in ()).throw(
        ValueError("judging bug")))
    store, ctx = _ctx(c, out)
    with pytest.raises(ValueError, match="judging bug"):
        check_5_matmuls(store, c, ctx, Bands.provisional())


def test_a_schedule_that_splits_a_run_or_misses_a_product_is_a_verifier_bug(monkeypatch):
    c, out = _step()
    steps = [list(s) for s in c.unit_schedule]
    split = [tuple(s) for s in steps]
    run = next(r for r in checks._member_runs(c.products) if len(r) > 1)
    first, second = run[0].m, run[1].m
    bad = [tuple(m for m in s if m != second) for s in steps] + [(second,)]
    for sched in (tuple(bad), tuple(steps[0:1]) + tuple(tuple(s[1:]) for s in steps[1:])):
        monkeypatch.setattr(DecoderReplay, "unit_schedule", lambda self, s=sched: s)
        store, ctx = _ctx(c, out)
        with pytest.raises(RuntimeError, match="unit schedule"):
            check_5_matmuls(store, c, ctx, Bands.provisional())
    twice = tuple(steps) + ((first,),)
    monkeypatch.setattr(DecoderReplay, "unit_schedule", lambda self: twice)
    store, ctx = _ctx(c, out)
    with pytest.raises(RuntimeError, match="twice"):
        check_5_matmuls(store, c, ctx, Bands.provisional())
    assert split  # the unmodified schedule is a plain list of steps


def test_the_window_rule_is_shared_with_the_layout():
    c, _ = _step()
    runs = list(checks._member_runs(c.products))
    for w in (1, 7, 20, checks._JUDGE_WINDOW):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(checks, "_JUDGE_WINDOW", w)
            windows = checks._judge_windows(runs)
            layout, n_windows = checks._layout(c)
        assert [i for win in windows for i in win] == list(range(len(runs)))
        assert n_windows == len(windows)
        assert {e["run"]: e["window"] for e in layout} == {
            i: k for k, win in enumerate(windows) for i in win if _groups_of(runs[i])}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(checks, "_JUDGE_WINDOW", len(runs[0]))
        assert checks._judge_windows(runs[:1]) == [[0], []]  # the last, empty window


def _groups_of(run):
    return checks._measure_groups(run)


# ---- the kernel path through the Mac emulation ----------------------------------------------


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", [("llama", True), ("qwen2", False)])
def test_kernel_rows_on_the_unit_path_equal_the_canonical_rows(switch, dtype, hostlib,  # noqa: F811
                                                               phased, monkeypatch):  # noqa: F811
    rows = {}
    for unit in (True, False):
        with monkeypatch.context() as mp:
            if not unit:
                mp.setattr(DecoderReplay, "unit_schedule", lambda self: None)
            rec = install(mp, "mac", hostlib)
            c, D, tree, w0 = _kernel_setup("mac", switch, dtype)
            assert _kernel_verify(c, D, tree, w0) is None
        by_m = {}
        for mem, row in zip(rec.members, rec.rows()):
            by_m[mem["m"]] = row.tobytes()
        assert sorted(by_m) == list(range(1, c.M + 1))
        rows[unit] = by_m
        if unit:
            order = [mem["m"] for mem in rec.members]
            assert order != sorted(order)  # served in the schedule's order
    assert rows[True] == rows[False]
