"""The S3 per-step loop on the MLP instance: prover → store → verifier → discard."""

import ast
import gc
import weakref
from pathlib import Path

import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    make_record,
    synthetic_dataset,
)
from verification.prover.step import plain_step
from verification.runs import loop as loop_mod
from verification.runs.loop import ProverFault, run_loop
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

ETA = 1e-3
K = 7
N_RECORDS = 40
T = 3


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, N_RECORDS, seed=0)


@pytest.fixture(scope="module")
def w0(c):
    return init_weights(c.widths, seed=0)


@pytest.fixture(scope="module")
def final(c, D, w0):
    model, w = c.build_model(), w0
    for t in range(1, T + 1):
        w, _ = plain_step(c, model, w, [D[i] for i in schedule(t, c.n_s, len(D))])
    return w


def _verifier(c, D, w0, h_D=None):
    return Verifier(c, h_D=h_D or dataset_tree(c, D).root, n_records=len(D), k=K, n_steps=T,
                    bands=Bands.provisional(), w0=w0, allow_provisional=True)


def _run(c, D, w0, final, **kw):
    v = _verifier(c, D, w0)
    return run_loop(c, c.build_model(), D, w0, v, final=final, **kw), v


def test_honest_run_accepted(c, D, w0, final):
    seen = []
    res, v = _run(c, D, w0, final, on_step=seen.append)
    assert res.verdict.accepted and res.rejection is None and res.verdict.steps_verified == T
    assert [s.t for s in res.steps] == [1, 2, 3] == [s.t for s in seen]
    assert all(s.rejection is None and s.prove_s >= 0 and s.commit_s >= 0 for s in res.steps)
    # Capture is a passthrough: the captured chain ends bit-identical to the uncaptured one.
    assert all(torch.equal(res.w_final[n], final[n]) for n in c.weight_names)
    assert sorted(v.timings) == [1, 2, 3] and set(v.run_timings) == {"0", "1", "8"}


def test_wrong_final_weights_rejected_at_8(c, D, w0, final):
    bad = {**final, c.weight(1): final[c.weight(1)] + 1.0}
    res, _ = _run(c, D, w0, bad)
    assert len(res.steps) == T and not res.verdict.accepted
    assert (res.rejection.step, res.rejection.check_id) == (T, "8")


class _Perturb(ProverFault):
    def __init__(self, t, m):
        self.t, self.m, self.calls = t, m, []

    def perturb(self, t):
        self.calls.append(t)
        return {self.m: lambda p: p * 1.01} if t == self.t else None


def test_stops_at_first_rejection(c, D, w0, final):
    fault = _Perturb(2, c.m_of("Y_1"))
    res, _ = _run(c, D, w0, final, fault=fault)
    assert fault.calls == [1, 2]  # step 3 never runs
    assert [s.rejection is None for s in res.steps] == [True, False]
    assert (res.rejection.step, res.rejection.check_id, res.rejection.kind) == (2, "5", "failed")
    assert res.steps[-1].rejection == res.rejection and not res.verdict.accepted


def test_check_1_failure_runs_no_step(c, D, w0, final):
    v = _verifier(c, D, w0, h_D=bytes(32))
    res = run_loop(c, c.build_model(), D, w0, v, final=final)
    assert res.steps == [] and res.w_final is None
    assert (res.rejection.step, res.rejection.check_id) == (0, "1")


def test_entry_weights_hook_feeds_the_prover(c, D, w0, final):
    """A hidden step changes what the prover starts from; the verifier sees only the store."""

    class Hidden(ProverFault):
        def entry_weights(self, t, w):
            return plain_step(c, c.build_model(), w, list(D[-4:]))[0] if t == 2 else w

    res, _ = _run(c, D, w0, final, fault=Hidden())
    assert (res.rejection.step, res.rejection.check_id) == (2, "7")


def test_emit_hook_splices_w_next(c, D, w0, final):
    class Forge(ProverFault):
        def emit(self, t, out):
            if t != 1:
                return out
            return out.with_w_next({n: (w + 1e-3).contiguous() for n, w in out.w_next.items()})

    res, _ = _run(c, D, w0, final, fault=Forge())
    assert (res.rejection.step, res.rejection.check_id) == (1, "6a")


def test_train_records_hook(c, D, w0, final):
    """A2: commit π(1)'s batch, train on another; check 5 rejects at the first product."""

    class A2(ProverFault):
        def train_records(self, t, records):
            return list(D[-4:]) if t == 1 else None

    res, _ = _run(c, D, w0, final, fault=A2())
    assert (res.rejection.step, res.rejection.check_id) == (1, "5")
    assert "P_1 (Y_1)" in res.rejection.detail


def test_committed_records_hook(c, D, w0, final):
    """A1: commit and train on a record that is not in D; check 4 rejects its audit path."""
    foreign = make_record(torch.zeros(c.widths[0]), torch.ones(c.widths[-1]))

    class A1(ProverFault):
        def committed_records(self, t, records):
            return [*records[:2], foreign, *records[3:]] if t == 1 else records

        def train_records(self, t, records):
            # The loop passes the committed batch, so this trains on the foreign record too.
            assert t != 1 or records[2] is foreign
            return list(records) if t == 1 else None

    res, _ = _run(c, D, w0, final, fault=A1())
    assert len(res.steps) == 1
    assert (res.rejection.step, res.rejection.check_id, res.rejection.kind) == (1, "4", "failed")


def test_each_step_is_released(c, D, w0, final):
    """S3: step t's transcript is gone by step t+1, and the last step's after the loop.

    The last step's W_{t+1} is excluded: it is the run's result, ``LoopResult.w_final``.
    """
    refs: dict[int, list[weakref.ref]] = {}
    dead_at_next: list[bool] = []

    class Watch(ProverFault):
        def emit(self, t, out):
            if t - 1 in refs:
                gc.collect()
                dead_at_next.append(all(r() is None for r in refs[t - 1]))
            refs[t] = [weakref.ref(out), *(weakref.ref(p) for p in out.products)]
            return out

    model = c.build_model()  # held by the test, so it can't hide a leak by being freed
    res = run_loop(c, model, D, w0, _verifier(c, D, w0), final=final, fault=Watch())
    assert res.verdict.accepted and sorted(refs) == [1, 2, 3]
    assert dead_at_next == [True, True]
    gc.collect()
    assert all(r() is None for rs in refs.values() for r in rs)


def test_verifier_receives_only_stores(c, D, w0, final, monkeypatch):
    """Invariant 1 at the loop boundary: verify_step's argument is a TranscriptStore."""
    got = []
    orig = Verifier.verify_step

    def spy(self, t, store):
        got.append(type(store))
        return orig(self, t, store)

    monkeypatch.setattr(Verifier, "verify_step", spy)
    _run(c, D, w0, final)
    assert got == [InMemoryStore] * T


def test_loop_is_instance_agnostic():
    src = Path(loop_mod.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        assert not any("instances" in n for n in names), names
