"""FS33b, the tester's tests: gaps in ``test_pipelined_loop.py``, the orchestrator's items 1-5,
the failure paths, the window, the setting and the metrics.

Defects found are marked ``xfail(strict=True)`` with their reason; the fix round turns each into
a plain test. CPU only; the child's specs and faults are in ``pipelined_helpers``.
"""

import dataclasses
import multiprocessing
import re
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from setup.config import load_config, setup_determinism
from tests.verification.runs import pipelined_helpers as P
from tests.verification.runs.test_pipelined_loop import (
    FAULT_STEP,
    K,
    World,
    assert_window,
    go,
)
from tests.verification.runs.test_sharded_run import SEED
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs import pipeline
from verification.runs.loop import run_loop
from verification.runs.metrics import CountRecorder, MemoryRecorder, MetricsWriter
from verification.runs.pipeline import ProverProcessError
from verification.runs.plain_baseline import training_config
from verification.runs.scenarios import (
    HONEST,
    Expected,
    FlipProduct,
    Scenario,
    count_run,
    memory_run,
    run_scenario,
)
from verification.transcript.shared import SharedStore
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", autouse=True)
def _determinism():
    threads, det = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    setup_determinism(load_config())
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det)


@pytest.fixture(autouse=True)
def _no_child_is_left():
    yield
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert multiprocessing.active_children() == []


@pytest.fixture(scope="module")
def mlp():
    c = MLPComputation(P.WIDTHS, n_s=P.N_S, eta=1e-3)
    return World("mlp", c, synthetic_dataset(c.widths, P.N_RECORDS, seed=0),
                 init_weights(c.widths, seed=0), P.MLPSpec(), 4)


@pytest.fixture(scope="module")
def mlp6():
    c = MLPComputation(P.WIDTHS, n_s=P.N_S, eta=1e-3)
    return World("mlp", c, synthetic_dataset(c.widths, P.N_RECORDS, seed=0),
                 init_weights(c.widths, seed=0), P.MLPSpec(), 6)


@pytest.fixture
def fast_grace(monkeypatch):
    """Short stop and terminate graces. ``stop``'s ``grace`` is a default argument bound when
    the function was defined, so patching ``STOP_GRACE_S`` alone does nothing."""
    monkeypatch.setattr(pipeline, "TERMINATE_GRACE_S", 1.0)
    monkeypatch.setattr(pipeline._Link.stop, "__defaults__", (1.0,))


@pytest.fixture
def stores(monkeypatch):
    """Every ``SharedStore`` the parent opens, to check each was closed."""
    opened = []
    orig = SharedStore.open.__func__

    def open_(cls, c, package):
        s = orig(cls, c, package)
        opened.append(s)
        return s

    monkeypatch.setattr(SharedStore, "open", classmethod(open_))
    return opened


def verifier_for(w, **kw):
    kw.setdefault("w0", w.w0)
    return Verifier(w.c, h_D=dataset_tree(w.c, w.D).root, n_records=len(w.D), k=K, n_steps=w.T,
                    bands=Bands.provisional(), seed=SEED, allow_provisional=True, **kw)


def call(w, **kw):
    kw.setdefault("prover", w.spec)
    verifier = kw.pop("verifier", None) or verifier_for(w)
    return run_loop(w.c, kw.pop("model", None), w.D, kw.pop("w0", w.w0), verifier, final=w.final,
                    pipelined=True, **kw)


def run_in_thread(fn, wait_s):
    """``fn()`` in a thread; ``(thread, box)`` after at most ``wait_s``, ``box`` holding its
    result or exception. The caller must end the thread's work."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(wait_s)
    return th, box


# ---- equal to sequential: the gaps --------------------------------------------------------

TIMES = {"prove_s", "commit_s", "verify_s", "handoff_s"}


def _fields(rec):
    return {f.name: (repr(getattr(rec, f.name)) if f.name == "capture" else getattr(rec, f.name))
            for f in dataclasses.fields(rec) if f.name not in TIMES}


def test_every_step_record_field_but_the_times_equals_sequential(mlp):
    s = Scenario("x", "", FlipProduct(FAULT_STEP, mlp.c.m_of("Y_2")), Expected())
    a = go(mlp, s.fault, pipelined=False)
    b = go(mlp, FlipProduct(FAULT_STEP, mlp.c.m_of("Y_2")), pipelined=True)
    assert len(a.loop.steps) == len(b.loop.steps) == FAULT_STEP
    for x, y in zip(a.loop.steps, b.loop.steps, strict=True):
        assert _fields(x) == _fields(y)
        assert all(getattr(y, n) >= 0 for n in TIMES) and y.verify_s > 0 and y.commit_s > 0
    assert a.loop.child_info is None and b.loop.child_info["sent"] >= FAULT_STEP


def test_a_nondefault_schedule_on_both_sides_equals_sequential(mlp):
    """The verifier's own ``schedule`` and the prover's, the same picklable function."""
    runs = []
    for pipelined in (False, True):
        v = verifier_for(mlp, schedule=P.first_batches)
        kw = {"prover": mlp.spec} if pipelined else {}
        model = None if pipelined else mlp.c.build_model()
        runs.append(run_loop(mlp.c, model, mlp.D, mlp.w0, v, final=mlp.final,
                             schedule=P.first_batches, pipelined=pipelined, **kw))
    a, b = runs
    assert [s.rejection for s in a.steps] == [s.rejection for s in b.steps]
    assert [_fields(s) for s in a.steps] == [_fields(s) for s in b.steps]
    assert a.verdict == b.verdict
    # the schedule really was used: step 1 passed check 4 with the verifier's own π
    assert b.steps and b.steps[0].rejection is None


def test_bind_hands_the_fault_the_childs_computation_and_a_reused_model(mlp):
    with pytest.raises(ProverProcessError) as e:
        go(mlp, P.BindProbe(), pipelined=True)
    assert e.value.kind == "exception"
    assert f"('ReusedModel', 'MLPComputation', {mlp.c.n_leaves})" in str(e.value)


def test_the_parent_builds_no_prover_model(mlp):
    def boom():
        raise AssertionError("the parent built a prover model")

    s = Scenario("x", "", HONEST.fault, Expected())
    r = run_scenario(mlp.c, mlp.D, mlp.w0, s, T=mlp.T, k=K, seed=SEED, final=mlp.final,
                     pipelined=True, prover=mlp.spec, build_model=boom)
    assert r.loop.verdict.accepted


# ---- the orchestrator's items 1-5 ---------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="item 1: run_pipelined's finally calls link.stop() while "
                   "its local stash still holds the early step's received package (on CUDA its "
                   "opened IPC handles), before the child may exit")
def test_item1_an_error_path_drops_received_packages_before_stopping_the_child(mlp, monkeypatch):
    seen = {}
    orig = pipeline._Link.stop

    def spy(self, *a, **k):
        f = sys._getframe(1)
        if f.f_code.co_name == "run_pipelined" and "at" not in seen:
            loc = f.f_locals
            seen["at"] = True
            seen["stash"] = sorted(loc.get("stash") or {})
            seen["msg"] = loc.get("msg") is not None
            seen["pkg"] = loc.get("pkg") is not None
        return orig(self, *a, **k)

    monkeypatch.setattr(pipeline._Link, "stop", spy)

    class Boom(Exception):
        pass

    def bad(rec):
        raise Boom

    with pytest.raises(Boom):
        go(mlp, pipelined=True, on_step=bad)
    assert seen.get("at"), "run_pipelined's finally never reached stop()"
    assert seen["stash"] == [] and not seen["msg"] and not seen["pkg"], seen


@pytest.mark.xfail(strict=True, reason="item 2: the handshake does not compare eta; a child "
                   "built with another eta trains differently and the run yields a verdict, "
                   "not an error")
def test_item2_a_child_with_another_eta_is_a_handshake_error(mlp):
    with pytest.raises(ProverProcessError) as e:
        call(mlp, prover=P.MLPVariantSpec(eta=2e-3))
    assert e.value.kind == "handshake"


@pytest.mark.xfail(strict=True, reason="item 2: the handshake does not compare the dataset; a "
                   "child on another dataset of the same size yields a verdict, not an error")
def test_item2_a_child_with_another_dataset_is_a_handshake_error(mlp):
    with pytest.raises(ProverProcessError) as e:
        call(mlp, prover=P.MLPVariantSpec(data_seed=1))
    assert e.value.kind == "handshake"


def test_item2_a_child_with_another_weight_dtype_is_a_handshake_error(mlp):
    with pytest.raises(ProverProcessError) as e:
        call(mlp, prover=P.MLPVariantSpec(w_dtype="float64"))
    assert e.value.kind in ("handshake", "exception")


def test_item2_the_variant_spec_with_nothing_changed_is_the_parents_run(mlp):
    """The control for the three tests above: the same spec with the parent's values runs."""
    r = call(mlp, prover=P.MLPVariantSpec())
    assert r.verdict.accepted and len(r.steps) == mlp.T


def test_item3_a_verifier_has_at_least_one_step_so_w_host_is_never_none(mlp):
    with pytest.raises(ValueError, match="positive int"):
        Verifier(mlp.c, h_D=dataset_tree(mlp.c, mlp.D).root, n_records=len(mlp.D), k=K,
                 n_steps=0, bands=Bands.provisional(), seed=SEED, w0=mlp.w0,
                 allow_provisional=True)
    r = call(mlp)  # the smallest accepted run still has a w_final
    assert r.w_final is not None and set(r.w_final) == set(mlp.c.weight_names)


def test_item4_the_models_parameters_already_require_grad_so_the_call_is_a_no_op(mlp):
    """``model.requires_grad_(True)`` in the child changes nothing the sequential loop's model
    doesn't already have, and the run is the sequential run (the equality tests)."""
    m = mlp.c.build_model()
    assert all(p.requires_grad for p in m.parameters())


def test_item5_chain_holds_the_agreed_w0_hashes_after_start_run(mlp):
    v = verifier_for(mlp)
    assert v.start_run(mlp.D) is None
    assert tuple(v._chain) == tuple(v.weight_hashes(mlp.w0))  # noqa: SLF001


def test_item5_a_verifier_built_from_hashes_and_no_w0_runs_and_still_checks(mlp):
    hashes = verifier_for(mlp).weight_hashes(mlp.w0)
    v = Verifier(mlp.c, h_D=dataset_tree(mlp.c, mlp.D).root, n_records=len(mlp.D), k=K,
                 n_steps=mlp.T, bands=Bands.provisional(), seed=SEED, w0_hashes=hashes,
                 allow_provisional=True)
    r = call(mlp, verifier=v, w0=None)
    assert r.verdict.accepted
    v2 = Verifier(mlp.c, h_D=dataset_tree(mlp.c, mlp.D).root, n_records=len(mlp.D), k=K,
                  n_steps=mlp.T, bands=Bands.provisional(), seed=SEED, w0_hashes=hashes,
                  allow_provisional=True)
    with pytest.raises(ProverProcessError) as e:
        call(mlp, verifier=v2, w0=None, prover=P.MLPSpec(shift=0.5))
    assert e.value.kind == "handshake"


def test_item5_a_w0_given_to_run_loop_that_is_not_the_verifiers_is_a_handshake_error(mlp):
    other = {n: v.clone() for n, v in mlp.w0.items()}
    other[mlp.c.weight_names[0]].view(-1)[0] += 1.0
    with pytest.raises(ProverProcessError) as e:
        call(mlp, w0=other)
    assert e.value.kind == "handshake" and "differs from the verifier" in str(e.value)


# ---- failure paths ------------------------------------------------------------------------


def test_a_spec_that_raises_before_ready_is_an_exception_error(mlp, stores):
    steps = []
    with pytest.raises(ProverProcessError) as e:
        call(mlp, prover=P.RaisingSpec(), on_step=steps.append)
    assert e.value.kind == "exception" and "boom before ready" in str(e.value)
    assert "Traceback" in str(e.value) and steps == [] and stores == []


def test_a_child_killed_before_ready_is_a_died_error(mlp):
    t0 = time.monotonic()
    with pytest.raises(ProverProcessError) as e:
        call(mlp, prover=P.SelfKillSpec())
    assert e.value.kind == "died" and time.monotonic() - t0 < 60


def test_a_child_killed_during_the_handshake_window_is_a_died_error(mlp):
    timer = threading.Timer(4.0, P.kill_child)
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(ProverProcessError) as e:
            call(mlp, prover=P.SleepSpec(120.0))
    finally:
        timer.cancel()
    assert e.value.kind == "died" and time.monotonic() - t0 < 60


def test_a_verifier_that_raises_in_verify_step_closes_the_store_and_the_child(mlp, stores,
                                                                              monkeypatch):
    class Bug(Exception):
        pass

    orig = Verifier.verify_step

    def verify_step(self, t, store):
        if t == 2:
            raise Bug("verifier bug")
        return orig(self, t, store)

    monkeypatch.setattr(Verifier, "verify_step", verify_step)
    steps = []
    with pytest.raises(Bug):
        go(mlp, pipelined=True, on_step=steps.append)
    assert [s.t for s in steps] == [1]
    assert len(stores) == 2 and all(s._pkg is None for s in stores)  # noqa: SLF001


@pytest.mark.parametrize("fault", [lambda: P.SlowAt(3, 600.0), lambda: P.StubbornAt(3, 600.0)],
                         ids=["terminate", "kill"])
def test_a_child_that_hangs_in_a_step_is_ended_within_the_graces(mlp, fast_grace, stores,
                                                                 fault):
    """The parent stops (its ``on_step`` raises at step 1) while the child sleeps in step 3, a
    child that reads its pipe only between steps. ``StubbornAt`` ignores SIGTERM, so only the
    ``kill`` ends it."""
    class Boom(Exception):
        pass

    def bad(rec):
        raise Boom

    t0 = time.monotonic()
    with pytest.raises(Boom):
        go(mlp, fault(), pipelined=True, on_step=bad)
    assert time.monotonic() - t0 < 60
    assert multiprocessing.active_children() == []
    assert all(s._pkg is None for s in stores)  # noqa: SLF001


def test_an_on_step_that_raises_at_the_last_step(mlp, stores):
    class Boom(Exception):
        pass

    seen = []

    def bad(rec):
        seen.append(rec.t)
        if rec.t == mlp.T:
            raise Boom

    with pytest.raises(Boom):
        go(mlp, pipelined=True, on_step=bad)
    assert seen == list(range(1, mlp.T + 1))
    assert len(stores) == mlp.T and all(s._pkg is None for s in stores)  # noqa: SLF001
    assert multiprocessing.active_children() == []


@pytest.mark.xfail(strict=True, reason="_Link.recv has no per-message deadline: a child that "
                   "hangs inside a step (alive, pipe open) hangs the parent for good")
def test_a_child_that_hangs_inside_a_step_does_not_hang_the_parent(mlp, monkeypatch):
    for name in ("STEP_DEADLINE_S", "RECV_DEADLINE_S", "MESSAGE_DEADLINE_S", "DEADLINE_S"):
        if hasattr(pipeline, name):
            monkeypatch.setattr(pipeline, name, 3.0)
    th, box = run_in_thread(lambda: go(mlp, P.SlowAt(2, 600.0), pipelined=True), 12.0)
    hung = th.is_alive()
    try:
        if hung:
            P.kill_child()  # so the thread and the suite end
        th.join(60)
    finally:
        assert not th.is_alive()
    assert not hung, "the parent was still waiting 12 s into a step that never returns"
    assert isinstance(box.get("error"), ProverProcessError)


# ---- the window ---------------------------------------------------------------------------


def test_the_window_holds_over_six_steps_and_after_a_mid_run_rejection(mlp6):
    a = go(mlp6, pipelined=True)
    assert a.loop.verdict.accepted and len(a.loop.steps) == 6
    assert_window(a.loop.child_info, 6)
    ev = a.loop.child_info["events"]
    assert [t for k, t in ev if k == "start"] == [1, 2, 3, 4, 5, 6]
    for t in range(1, 7):
        assert ("release", t) in ev
    m = mlp6.c.m_of("Y_2")
    b = go(mlp6, FlipProduct(3, m), pipelined=True)
    assert b.loop.rejection.step == 3 and len(b.loop.steps) == 3
    assert_window(b.loop.child_info, 6)
    assert b.loop.child_info["max_unreleased"] <= 2
    assert len(b.loop.w_final) == len(mlp6.w0)


# ---- the setting --------------------------------------------------------------------------


def test_the_env_setting_reaches_run_scenario_and_memory_run(mlp, monkeypatch):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    s = Scenario("x", "", HONEST.fault, Expected())
    r = run_scenario(mlp.c, mlp.D, mlp.w0, s, T=mlp.T, k=K, seed=SEED, final=mlp.final,
                     prover=mlp.spec)
    assert r.loop.pipelined and r.loop.child_info["pid"] != 0
    rows = memory_run(mlp.c, mlp.D, mlp.w0, run="r", T=mlp.T, k=K, seed=SEED, final=mlp.final,
                      prover=mlp.spec)
    comps = {(x["step"], x["component"]) for x in rows if x["record"] == "memory"}
    assert (2, "P3.commit") in comps and (2, "6a") in comps and (0, "P4.tree") in comps


def test_count_run_refuses_the_env_setting_and_runs_with_false_passed(mlp, monkeypatch):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(ValueError, match="pipelined"):
        count_run(mlp.c, mlp.D, mlp.w0, run="r", T=mlp.T, k=K, seed=SEED, final=mlp.final)
    rows = count_run(mlp.c, mlp.D, mlp.w0, run="r", T=mlp.T, k=K, seed=SEED, final=mlp.final,
                     pipelined=False)
    assert rows and multiprocessing.active_children() == []


@pytest.mark.parametrize("raw", ["2", "true", "yes", ""])
def test_the_setting_takes_0_or_1_only(monkeypatch, raw):
    monkeypatch.setenv("VERIF_PIPELINED", raw)
    with pytest.raises(ValueError, match="VERIF_PIPELINED"):
        load_config()


def test_the_setting_enters_no_config_hash(monkeypatch):
    from verification.runs.metrics import config_hash

    monkeypatch.setenv("VERIF_PIPELINED", "0")
    off = load_config()
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    on = load_config()
    assert off.pipelined is False and on.pipelined is True
    assert training_config(off) == training_config(on) and "pipelined" not in training_config(on)
    assert config_hash(training_config(off)) == config_hash(training_config(on))
    # the entry points that hash the config minus an exclusion tuple
    for name in ("bring_up", "calibrate", "cheats", "k_tunability", "llama_step", "mlp_smoke",
                 "run_verified"):
        src = (ROOT / "verification" / "runs" / f"{name}.py").read_text()
        m = re.search(r"fields\(cfg\)\s+if f\.name not in \((.*?)\)\}", src, re.S)
        assert m and '"pipelined"' in m.group(1), name


# ---- the metrics --------------------------------------------------------------------------


def test_a_counting_recorder_is_refused_before_a_child_starts(mlp, monkeypatch):
    def no_spawn(*a, **k):
        raise AssertionError("a child was started")

    monkeypatch.setattr(pipeline.multiprocessing, "get_context", no_spawn)
    rec = CountRecorder("r", "s", None, None)
    with pytest.raises(ValueError, match="cannot split"):
        call(mlp, recorder=rec, section=rec.seam())


def test_a_traced_time_recorder_is_refused_before_a_child_starts(mlp, monkeypatch, tmp_path):
    def no_spawn(*a, **k):
        raise AssertionError("a child was started")

    monkeypatch.setattr(pipeline.multiprocessing, "get_context", no_spawn)
    rec = MetricsWriter(tmp_path / "w", "r").recorder("s", trace_step=1)
    with pytest.raises(ValueError, match="trace"):
        call(mlp, recorder=rec, section=rec.seam())


def test_merged_rows_are_unique_per_step_and_the_release_rows_land_where_documented(
        mlp6, tmp_path):
    import json

    w = MetricsWriter(tmp_path / "t", "r")
    rec = w.recorder("s")
    r = go(mlp6, pipelined=True, recorder=rec)
    w.close()
    assert r.loop.verdict.accepted
    lines = [json.loads(x) for x in open(tmp_path / "t" / "markers.jsonl")]
    by = {x.get("step", "run"): x for x in lines if x.get("kind") in ("step", "run")}
    assert set(by) == {1, 2, 3, 4, 5, 6, "run"}
    for t in range(1, 7):
        names = [s["name"] for s in by[t]["sections"]]
        assert len(names) == len(set(names)), f"step {t}: a name on both sides"
        # the release of t-2 is timed in the child's table of the step in progress then, so
        # steps 1 and 2 hold none and 3..6 hold one; the last two releases are never merged
        assert names.count("handoff.release") == (1 if t >= 3 else 0), (t, names)
    time_rows = [x for x in rec.rows if x["record"] == "time"]
    for t in range(1, 7):
        comps = {x["component"] for x in time_rows if x.get("step") == t}
        assert "step_wall" in comps, t
    assert any("P4.tree" in x["component"] for x in time_rows if x.get("step") == 0)


def test_a_memory_recorder_has_no_name_on_both_sides(mlp6):
    rec = MemoryRecorder("r", "s", __import__("verification.runs.metrics", fromlist=["x"])
                         .memory_probe("cpu"))
    run_scenario(mlp6.c, mlp6.D, mlp6.w0, HONEST, T=6, k=K, seed=SEED, final=mlp6.final,
                 recorder=rec, pipelined=True, prover=mlp6.spec)
    keys = [(r["step"], r["component"]) for r in rec.rows if r["record"] == "memory"]
    assert len(keys) == len(set(keys))
