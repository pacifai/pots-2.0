"""The pipelined run loop (FS33b): the prover in a spawned child, the verifier in the parent.

CPU only, on the tiny decoder (``test_sharded_run._tiny``, seed 0) and the MLP. The child's specs
and test faults are in ``pipelined_helpers`` so a ``spawn`` child can import them.

- Equal to sequential: the verdict, each ``StepRecord``'s deciding fields, the verifier's
  ``stats`` and ``record_pass`` and ``w_final`` are the sequential run's, bit for bit, for an
  honest run and for each fault, which stops at its first verified rejection.
- The window: never two transcripts more than the protocol allows, and step ``t + 2`` starts
  only after ``t``'s release (the child's own log).
- Failure: a raise in the child, a killed child, a raising ``on_step``, a write into a shared
  leaf, and a ``W_0`` that differs are all run errors, never verdicts, and no child is left.
- Refusals and the recorders' merged rows.
"""

import json
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from setup.config import load_config, setup_determinism
from setup.data import other_batch
from tests.verification.runs import pipelined_helpers as P
from tests.verification.runs.test_sharded_run import SEED, THREE, _numbers
from tests.verification.transcript import shared_helpers as H
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.computation.placement import Placement
from verification.runs.loop import ProverFault, run_loop
from verification.runs.metrics import MemoryRecorder, MetricsWriter, count_pass, memory_probe
from verification.runs.pipeline import ProverProcessError
from verification.runs.scenarios import (
    HONEST,
    CommitBatch,
    Expected,
    FlipProduct,
    HiddenStep,
    KeepStep,
    NudgeWNext,
    Scenario,
    SpliceWNext,
    TrainedElsewhereWNext,
    TrainOn,
    count_run,
    honest_final,
    run_scenario,
)
from verification.transcript.errors import StoreMutationError
from verification.transcript.store import StoreHandoff
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

K = 5
FAULT_STEP = 2


@pytest.fixture(scope="module", autouse=True)
def _determinism():
    """The parent's sequential runs and the child's specs set up the same determinism."""
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


class World:
    """One computation, its data, ``W_0``, the child's spec and the agreed final weights."""

    def __init__(self, name, c, D, w0, spec, T):
        self.name, self.c, self.D, self.w0, self.spec, self.T = name, c, D, w0, spec, T
        self.final = honest_final(c, D, w0, T, seed=SEED)

    def other(self, t):
        return [self.D[i] for i in other_batch(t, self.c.n_s, len(self.D), SEED)]

    def faults(self, t):
        """Fresh faults, each wrong at step ``t``."""
        c = self.c
        if self.name == "mlp":
            m, name = c.m_of("Y_2"), c.weight(2)
        else:
            m = next(s.m for s in c.products
                     if s.name.startswith("L2.") and s.kind.name == "FORWARD")
            name = next(iter(c.linear_weights))
        spliced = {n: w.clone() for n, w in self.w0.items()}
        spliced[name].view(-1)[3] += 0.5
        return {
            "flip": FlipProduct(t, m),
            "nudge": NudgeWNext(t, name, 5, 1 << 16),
            "commit-batch": CommitBatch(t, self.other(t)),
            "train-on": TrainOn(t, self.other(t)),
            "splice": SpliceWNext(t, spliced),
            "hidden": HiddenStep(c, t, self.other(t)),
            "trained-elsewhere": TrainedElsewhereWNext(c, t, self.other(t)),
        }


def _decoder(devices, T=3):
    c, D, _, w0 = H.setup(devices)
    return World("decoder", c, D, w0, P.DecoderSpec(devices=devices), T)


@pytest.fixture(scope="module")
def mlp():
    c = MLPComputation(P.WIDTHS, n_s=P.N_S, eta=1e-3)
    return World("mlp", c, synthetic_dataset(c.widths, P.N_RECORDS, seed=0),
                 init_weights(c.widths, seed=0), P.MLPSpec(), 4)


@pytest.fixture(scope="module")
def three():
    return _decoder(THREE)


def go(w, fault=None, *, pipelined, T=None, **kw):
    s = Scenario("x", "", fault or HONEST.fault, Expected())
    return run_scenario(w.c, w.D, w.w0, s, T=T or w.T, k=K, seed=SEED, final=w.final,
                        pipelined=pipelined, prover=w.spec if pipelined else None, **kw)


def assert_equal(a, b):
    """``a`` sequential, ``b`` pipelined: the same run."""
    ra, rb = a.loop, b.loop
    assert rb.pipelined and not ra.pipelined
    assert ra.verdict == rb.verdict
    key = lambda s: (s.t, s.rejection, s.loss, s.root, s.known, s.transcript_bytes,  # noqa: E731
                     s.leaves)
    assert [key(s) for s in ra.steps] == [key(s) for s in rb.steps]
    assert all(s.root and s.known is not None for s in rb.steps)
    assert _numbers(a.verifier) == _numbers(b.verifier)
    assert a.verifier.record_pass == b.verifier.record_pass
    assert ra.w_final.keys() == rb.w_final.keys()
    assert all(torch.equal(ra.w_final[n], rb.w_final[n]) for n in ra.w_final)


def assert_window(info, T):
    """At most two unreleased steps, and step ``t + 2`` starts only after ``t``'s release."""
    ev = info["events"]
    assert info["max_unreleased"] <= 2 and info["sent"] >= 1
    unreleased, worst = set(), 0
    for kind, t in ev:
        if kind == "start":
            unreleased.add(t)
            worst = max(worst, len(unreleased))
        else:
            unreleased.discard(t)
    assert worst <= 2
    for t in range(3, T + 1):
        if ("start", t) in ev:
            assert ev.index(("release", t - 2)) < ev.index(("start", t))


# ---- equal to sequential ------------------------------------------------------------------


@pytest.fixture(scope="module")
def honest_mlp(mlp):
    return go(mlp, pipelined=False), go(mlp, pipelined=True)


@pytest.fixture(scope="module")
def honest_three(three):
    return go(three, pipelined=False), go(three, pipelined=True)


def test_honest_mlp_equals_sequential(mlp, honest_mlp):
    a, b = honest_mlp
    assert b.loop.verdict.accepted and len(b.loop.steps) == mlp.T
    assert_equal(a, b)
    assert_window(b.loop.child_info, mlp.T)
    assert b.loop.child_info["sent"] == mlp.T and b.loop.child_info["pid"] != os.getpid()


def test_honest_decoder_equals_sequential(three, honest_three):
    a, b = honest_three
    assert b.loop.verdict.accepted and len(b.loop.steps) == three.T
    assert_equal(a, b)
    assert_window(b.loop.child_info, three.T)


def test_honest_unsharded_decoder_equals_sequential():
    w = _decoder(None)
    assert not w.c.placement.sharded
    assert_equal(go(w, pipelined=False), go(w, pipelined=True))


def test_the_prover_runs_ahead_of_the_verifier(honest_mlp):
    """The child starts step 2 before the release of step 1 (the pipelining itself)."""
    ev = honest_mlp[1].loop.child_info["events"]
    assert ev.index(("start", 2)) < ev.index(("release", 1))


FAULTS = ["flip", "nudge", "commit-batch", "train-on", "splice", "hidden", "trained-elsewhere"]


@pytest.mark.parametrize("name", FAULTS)
def test_each_fault_stops_at_its_first_verified_rejection_mlp(mlp, name):
    a = go(mlp, mlp.faults(FAULT_STEP)[name], pipelined=False)
    b = go(mlp, mlp.faults(FAULT_STEP)[name], pipelined=True)
    rej = b.loop.rejection
    assert rej is not None and rej.step == FAULT_STEP and len(b.loop.steps) == FAULT_STEP
    assert all(s.rejection is None for s in b.loop.steps[:-1])
    assert_equal(a, b)
    assert_window(b.loop.child_info, mlp.T)


@pytest.mark.parametrize("name", FAULTS)
def test_each_fault_stops_at_its_first_verified_rejection_decoder(three, name):
    a = go(three, three.faults(FAULT_STEP)[name], pipelined=False)
    b = go(three, three.faults(FAULT_STEP)[name], pipelined=True)
    rej = b.loop.rejection
    assert rej is not None and rej.step == FAULT_STEP and len(b.loop.steps) == FAULT_STEP
    assert_equal(a, b)
    assert_window(b.loop.child_info, three.T)


@pytest.mark.parametrize("t", [1, 3])
def test_a_rejection_at_the_first_and_the_last_step(three, t):
    a = go(three, three.faults(t)["flip"], pipelined=False)
    b = go(three, three.faults(t)["flip"], pipelined=True)
    assert b.loop.rejection.step == t and len(b.loop.steps) == t
    assert_equal(a, b)


# ---- failure ------------------------------------------------------------------------------


def test_a_raise_in_the_child_is_a_run_error_with_its_traceback(mlp):
    steps = []
    with pytest.raises(ProverProcessError) as e:
        go(mlp, P.RaiseAt(2), pipelined=True, on_step=steps.append)
    assert e.value.kind == "exception"
    assert "RaiseAt: boom at step 2" in str(e.value) and "Traceback" in str(e.value)
    # no verdict, nothing after the failed step; step 1 got its on_step only if the child's
    # error had not reached the parent first
    assert [s.t for s in steps] in ([], [1])


def test_a_killed_child_is_an_error_within_seconds(mlp):
    """The child sleeps in step 2 (it reads its pipe only between steps), and is killed
    from outside while the parent waits for it."""
    timer = threading.Timer(3.0, P.kill_child)
    t0 = time.monotonic()
    timer.start()
    try:
        with pytest.raises(ProverProcessError) as e:
            go(mlp, P.SlowAt(2, 120.0), pipelined=True)
    finally:
        timer.cancel()
    assert e.value.kind == "died"
    assert time.monotonic() - t0 < 30


def test_a_raising_on_step_stops_and_joins_the_child(mlp):
    class Boom(Exception):
        pass

    def bad(rec):
        raise Boom

    with pytest.raises(Boom):
        go(mlp, pipelined=True, on_step=bad)
    assert multiprocessing.active_children() == []


def test_keyboard_interrupt_in_on_step_stops_and_joins_the_child(mlp):
    def bad(rec):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        go(mlp, pipelined=True, on_step=bad)
    assert multiprocessing.active_children() == []


def test_a_write_into_a_shared_leaf_fails_the_release(mlp):
    steps = []
    with pytest.raises(StoreMutationError):
        go(mlp, P.WriteIntoLeaf(2), pipelined=True, on_step=steps.append)
    assert steps == []  # no on_step for the step whose release failed


def test_a_w0_that_differs_is_an_error_before_step_1(mlp):
    steps = []
    s = Scenario("x", "", HONEST.fault, Expected())
    with pytest.raises(ProverProcessError) as e:
        run_scenario(mlp.c, mlp.D, mlp.w0, s, T=mlp.T, k=K, seed=SEED, final=mlp.final,
                     pipelined=True, prover=P.MLPSpec(shift=0.5), on_step=steps.append)
    assert e.value.kind == "handshake" and "W_0" in str(e.value)
    assert steps == []


def test_a_rejected_run_start_returns_a_verdict_and_no_child(mlp):
    """Check 1 rejects a dataset that is not the published one, before any step."""
    v = Verifier(mlp.c, h_D=b"\x00" * 32, n_records=len(mlp.D), k=K, n_steps=mlp.T,
                 bands=Bands.provisional(), seed=SEED, w0=mlp.w0, allow_provisional=True)
    r = run_loop(mlp.c, None, mlp.D, mlp.w0, v, final=mlp.final, pipelined=True,
                 prover=mlp.spec)
    assert r.rejection is not None and r.rejection.check_id == "1"
    assert r.steps == [] and r.w_final is None


# ---- refusals -----------------------------------------------------------------------------


def call(w, **kw):
    v = Verifier(w.c, h_D=dataset_tree(w.c, w.D).root, n_records=len(w.D), k=K, n_steps=w.T,
                 bands=Bands.provisional(), seed=SEED, w0=w.w0, allow_provisional=True)
    kw.setdefault("prover", w.spec)
    return run_loop(w.c, kw.pop("model", None), w.D, w.w0, v, final=w.final, pipelined=True,
                    **kw)


class _Disk(StoreHandoff):
    def hold(self, c, t, leaves, tree, dataset_paths=None):
        raise AssertionError("never reached")


def test_refusals_come_before_the_child_starts(mlp, tmp_path):
    with pytest.raises(ValueError, match="ProverSpec"):
        call(mlp, prover=None)
    with pytest.raises(ValueError, match="handoff"):
        call(mlp, handoff=_Disk())
    with pytest.raises(ValueError, match="KeepStep"):
        call(mlp, fault=KeepStep(1))
    with pytest.raises(ValueError, match="model"):
        call(mlp, model=mlp.c.build_model())
    traced = MetricsWriter(tmp_path / "w", "r").recorder("s", trace_step=1)
    with pytest.raises(ValueError, match="trace"):
        call(mlp, recorder=traced, section=traced.seam())
    with pytest.raises(ValueError, match="seam"):
        call(mlp, recorder=MetricsWriter(tmp_path / "w2", "r").recorder("s"),
             section=lambda name: None)
    assert multiprocessing.active_children() == []


def test_the_counting_pass_refuses_pipelined(mlp):
    with pytest.raises(ValueError, match="pipelined"):
        count_run(mlp.c, mlp.D, mlp.w0, run="r", T=mlp.T, k=K, seed=SEED, final=mlp.final,
                  pipelined=True)
    with pytest.raises(ValueError, match="cannot split"):
        count_pass("r", "s", lambda rec: run_scenario(
            mlp.c, mlp.D, mlp.w0, HONEST, T=mlp.T, k=K, seed=SEED, final=mlp.final,
            recorder=rec, pipelined=True, prover=mlp.spec))


def test_a_cuda_prover_with_an_unsharded_placement_is_refused_without_a_gpu():
    c = SimpleNamespace(placement=Placement(prover=torch.device("cuda:0"), devices=None))
    with pytest.raises(ValueError, match="sharded"):
        run_loop(c, None, [], None, None, final={}, pipelined=True, prover=P.MLPSpec())


def test_an_unpicklable_part_names_itself(mlp):
    with pytest.raises(TypeError, match="fault"):
        call(mlp, fault=P.Unpicklable())
    with pytest.raises(TypeError, match="schedule"):
        call(mlp, schedule=lambda t: [0, 1, 2, 3])
    with pytest.raises(TypeError, match="ProverSpec"):
        call(mlp, prover=lambda: None)
    assert multiprocessing.active_children() == []


def test_a_picklable_schedule_crosses(mlp):
    from tests.verification.runs.pipelined_helpers import first_batches

    a = go(mlp, pipelined=False)
    assert a.loop.verdict.accepted
    v = Verifier(mlp.c, h_D=dataset_tree(mlp.c, mlp.D).root, n_records=len(mlp.D), k=K,
                 n_steps=mlp.T, bands=Bands.provisional(), seed=SEED, w0=mlp.w0,
                 allow_provisional=True)
    r = run_loop(mlp.c, None, mlp.D, mlp.w0, v, final=mlp.final, pipelined=True,
                 prover=mlp.spec, schedule=first_batches)
    # a prover whose π departs from the verifier's is rejected at check 4
    assert r.rejection is not None and r.rejection.check_id == "4" and r.rejection.step == 1


def test_the_setting_is_off_by_default_and_reads_the_environment(monkeypatch):
    from verification.runs.loop import pipelined_setting

    assert load_config().pipelined is False and pipelined_setting(None) is False
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    assert pipelined_setting(None) is True and pipelined_setting(False) is False


# ---- recorders ----------------------------------------------------------------------------


def _markers(path):
    lines = [json.loads(x) for x in open(path / "markers.jsonl")]
    return [x for x in lines if x.get("kind") in ("step", "run")]


def test_a_time_recorder_gets_both_sides_sections(mlp, tmp_path):
    w = MetricsWriter(tmp_path / "t", "r")
    rec = w.recorder("s")
    r = go(mlp, pipelined=True, recorder=rec)
    w.close()
    child = r.loop.child_info["pid"]
    lines = _markers(tmp_path / "t")
    by = {x.get("step", "run"): x for x in lines}
    assert set(by) == {1, 2, 3, 4, "run"}
    for t in range(1, mlp.T + 1):
        pids = {s["name"]: s["pid"] for s in by[t]["sections"]}
        assert len(pids) == len(by[t]["sections"]), "a name appears twice"
        assert pids["P3.commit"] == child and pids["P5.write"] == child
        assert pids["6a"] == os.getpid() and pids["2.root"] == os.getpid()
    run_pids = {s["name"]: s["pid"] for s in by["run"]["sections"]}
    assert run_pids["P4.tree"] == child and run_pids["1"] == os.getpid()
    names = {row["component"] for row in rec.rows if row.get("step") == 2
             and row["record"] == "time"}
    assert "step_wall" in names and "P5.write" in names and "6a" in names
    assert any(s["name"] == "handoff.release" for s in by[3]["sections"])


def test_a_memory_recorder_gets_both_sides_sections(mlp):
    rec = MemoryRecorder("r", "s", memory_probe("cpu"))
    run_scenario(mlp.c, mlp.D, mlp.w0, HONEST, T=mlp.T, k=K, seed=SEED, final=mlp.final,
                 recorder=rec, pipelined=True, prover=mlp.spec)
    comps = {(r["step"], r["component"]) for r in rec.rows if r["record"] == "memory"}
    assert (2, "P3.commit") in comps and (2, "6a") in comps and (0, "P4.tree") in comps
    assert len(comps) == len([r for r in rec.rows if r["record"] == "memory"])
