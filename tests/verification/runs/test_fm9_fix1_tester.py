"""FM9 fix round 1 tester: the commit's own known count (``StepRecord.known``), the gate read
from the real line (each count off alone, the old line format), the stream synchronize of a
failed arrival enqueue, and the NaN bracket's power against a shifted failing index. CPU only."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from tests.verification.commitment.test_leaves_known import (  # noqa: F401
    c as mlp, first_product, hostlib, leaves as mlp_leaves, nan_bracket, on_kernel)
from tests.verification.runs import test_loop as tl
from tests.verification.runs.test_device_leaves_session_tester import (  # noqa: F401
    bring_part, parts_with, real, written_line)
from tests.verification.runs.test_loop import D, c, final, w0  # noqa: F401
from verification.commitment import leaves as L
from verification.commitment.encoding import NonFiniteError
from verification.commitment.leaves import HashStats, hash_step_leaves
from verification.runs import fs11_session as fs
from verification.runs import loop as loop_mod
from verification.runs.loop import StepRecord
from verification.transcript.store import InMemoryStore

M0 = fs.RUN_PINS[0][0]


# ---- StepRecord.known ----------------------------------------------------------------------


def test_the_new_field_is_last_defaults_to_none_and_old_positional_calls_still_build():
    names = [f.name for f in dataclasses.fields(StepRecord)]
    assert names[-1] == "known" and names[-2] == "root"
    assert StepRecord(1, None, 0.0, 0.0, 0.0, 0.0, 0.0).known is None
    full = StepRecord(1, None, 0.0, 0.0, 0.0, 0.0, 0.0, None, 5, 6, b"r", 7)
    assert (full.leaves, full.root, full.known) == (6, b"r", 7)


def fake_known(monkeypatch, n):
    """``prove_step`` handing the commit real digests for its first ``n`` product leaves."""
    real_prove = loop_mod.prove_step

    def wrapper(c, *a, **kw):
        out = real_prove(c, *a, **kw)
        want = hash_step_leaves(c, out.leaves())
        idx = [c.product_index(m) for m in range(1, n + 1)]
        return dataclasses.replace(out, arrival_digests={i: want[i] for i in idx})

    monkeypatch.setattr(loop_mod, "prove_step", wrapper)


@pytest.mark.parametrize("n", [1, 3])
def test_a_commit_given_digests_counts_exactly_those_every_step_and_the_run_is_accepted(
        c, D, w0, final, monkeypatch, n):  # noqa: F811
    fake_known(monkeypatch, n)
    res, _ = tl._run(c, D, w0, final)
    assert res.verdict.accepted
    assert [s.known for s in res.steps] == [n] * tl.T  # a fresh count per step, not cumulative


def test_a_commit_that_ignores_the_digests_counts_zero(c, D, w0, final, monkeypatch):  # noqa: F811
    fake_known(monkeypatch, 3)
    real_commit = InMemoryStore.commit_step.__func__ if hasattr(
        InMemoryStore.commit_step, "__func__") else InMemoryStore.commit_step

    def no_known(c_, step, **kw):
        return real_commit(c_, dataclasses.replace(step, arrival_digests={}), **kw)

    monkeypatch.setattr(InMemoryStore, "commit_step", staticmethod(no_known))
    res, _ = tl._run(c, D, w0, final)
    assert [s.known for s in res.steps] == [0] * tl.T


def test_commit_step_fills_the_callers_stats_and_adds_nothing_without_them(
        mlp, mlp_leaves, monkeypatch):  # noqa: F811
    from verification.computation.instances.mlp import init_weights, synthetic_dataset
    from verification.prover.step import prove_step

    c_ = mlp
    data = synthetic_dataset(c_.widths, 12, seed=0)
    out = prove_step(c_, c_.build_model(), init_weights(c_.widths, seed=0), data[4:8])
    want = hash_step_leaves(c_, out.leaves())
    out = dataclasses.replace(out, arrival_digests={first_product(c_): want[first_product(c_)]})
    seen = []
    real_hash = L.hash_step_leaves
    import verification.transcript.store as store_mod
    monkeypatch.setattr(store_mod, "hash_step_leaves",
                        lambda c, lv, st=None, **kw: (seen.append(st),
                                                      real_hash(c, lv, st, **kw))[1])
    stats = HashStats()
    InMemoryStore.commit_step(c_, out, stats=stats)
    assert stats.known == 1 and seen[-1] is stats
    InMemoryStore.commit_step(c_, out)
    assert seen[-1] is None  # no section, no stats: nothing is observed
    InMemoryStore.commit_step(c_, out, section=lambda name: __import__("contextlib").nullcontext(),
                              stats=stats)
    assert seen[-1] is stats and stats.known == 2  # a caller's stats accumulate, as documented


def test_a_record_without_stats_makes_the_line_refuse(real):  # noqa: F811
    from verification.runs import bring_up as bu
    c_, leaves, root = real
    step = SimpleNamespace(t=1, rejection=None, root=root, known=None,
                           capture={"arrival_hash": {"used": 1, "leaves": 1}})
    b = SimpleNamespace(kept=SimpleNamespace(leaf=lambda i: leaves[i]),
                        result=SimpleNamespace(loop=SimpleNamespace(steps=[step])))
    with pytest.raises(ValueError, match="known count"):
        bu.device_leaves_line(b, c_, T=1)
    step.known = 0  # an honest zero (a CPU run) is printed, not refused
    assert "commit known 0, M" in bu.device_leaves_line(b, c_, T=1)


# ---- the gate on the real line ---------------------------------------------------------------


def test_the_real_line_has_commit_known_right_after_hashed(real):  # noqa: F811
    text = written_line(real)
    assert ", hashed " in text and text.index(", commit known ") > text.index(", hashed ")
    assert text.index(", commit known ") < text.index(", M ")


@pytest.mark.parametrize("which,kw,own,others", [
    ("arrival", {"used": 0}, "arrival digests, expected", ("the commit took", "hashed on")),
    ("known", {"known": 0}, "the commit took", ("hashed on",)),
    ("hashed", {"hashed": 1}, "products hashed on arrival", ("arrival digests, expected",
                                                             "the commit took")),
])
def test_each_count_off_alone_fails_with_only_its_own_reason(real, tmp_path, which, kw, own,
                                                              others):  # noqa: F811
    g = fs.device_leaves_gate(parts_with(real, tmp_path, **kw))
    why = g["models"][M0]["why"]
    assert g["pass"] is False and own in why
    assert not any(o in why for o in others), why
    assert why.count(";") == 0  # one reason only


def test_all_three_counts_right_passes_and_known_is_in_the_row(real, tmp_path):  # noqa: F811
    g = fs.device_leaves_gate(parts_with(real, tmp_path))
    assert g["pass"] is True and g["models"][M0]["why"] == ""


def test_the_old_line_without_commit_known_fails_closed_and_is_not_misparsed(
        real, tmp_path):  # noqa: F811
    text = written_line(real)
    old = text.replace(text[text.index(", commit known "):text.index(", M ")], "")
    assert "commit known" not in old
    part = bring_part(M0, old, tmp_path)
    assert part["extras"]["device_leaves"] is None
    parts = parts_with(real, tmp_path)
    parts[2] = part
    g = fs.device_leaves_gate(parts)
    assert g["pass"] is False and "no device leaves line" in " | ".join(g["reasons"])


@pytest.mark.parametrize("edit", [
    lambda t: t.replace(", commit known ", ", commit known x"),
    lambda t: t.replace("commit known ", "commit  known "),
    lambda t: t.replace(", commit known", ", known"),
    lambda t: t + ", extra 1",
])
def test_a_garbled_commit_known_field_fails_closed(real, tmp_path, edit):  # noqa: F811
    part = bring_part(M0, edit(written_line(real)), tmp_path)
    assert part["extras"]["device_leaves"] is None


# ---- the stream synchronize ------------------------------------------------------------------


@pytest.mark.parametrize("hashable,fails", [(True, False), (False, False), (True, True)])
def test_the_side_stream_is_synchronized_only_when_the_enqueue_fails(monkeypatch, hashable,
                                                                    fails):
    import contextlib

    from verification.prover.capture import MatmulCapture
    log = []

    class Event:
        def record(self, s):
            pass

        def synchronize(self):
            log.append("event")

    class Stream:
        def wait_stream(self, o):
            pass

        def synchronize(self):
            log.append("stream")

    side = Stream()
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: side)

    def hash_(rec, out, stream):
        if fails:
            raise KeyboardInterrupt("x")  # a BaseException leaves through the same guard
        return None

    out = torch.randn(4, 6)
    cap = SimpleNamespace(
        buffers=SimpleNamespace(take=lambda shape, dtype: torch.empty(shape, dtype=dtype),
                                stream=lambda device: side),
        _hashable=lambda rec, o: hashable, _hash_arrival=hash_, _pending=[],
        _drain=lambda block: None)
    rec = SimpleNamespace(out=out, versions={})
    if fails:
        with pytest.raises(KeyboardInterrupt):
            MatmulCapture._offload(cap, rec)
        assert log == ["stream"] and cap._pending == []
    else:
        MatmulCapture._offload(cap, rec)
        assert log == [] and len(cap._pending) == 1 and rec.out is not out


# ---- the bracket against a shifted failing index ---------------------------------------------


ORIG_OUTCOME = L._table_outcome


def shifted(monkeypatch, by):
    def f(tb):
        d, fail = ORIG_OUTCOME(tb)
        return d, (None if fail is None else (fail[0] + by, fail[1]))

    monkeypatch.setattr(L, "_table_outcome", f)


def test_the_bracket_catches_a_kernel_failure_reported_one_leaf_late(
        mlp, mlp_leaves, on_kernel, monkeypatch):  # noqa: F811
    """The merge keeps the smallest index: a NaN reported at i + 1 loses to the cut at i + 1 (a
    host error), so the ``i + 1`` leg gets the shape error and fails."""
    i = first_product(mlp) + 1
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 2)
    nan_bracket(mlp, mlp_leaves, i, lambda b: hash_step_leaves(mlp, b))  # honest: passes
    shifted(monkeypatch, +1)
    with pytest.raises(BaseException) as e:  # noqa: PT011
        nan_bracket(mlp, mlp_leaves, i, lambda b: hash_step_leaves(mlp, b))
    assert not isinstance(e.value, (SystemExit, KeyboardInterrupt))


def test_a_kernel_failure_reported_one_leaf_early_is_beyond_the_brackets_reach(
        mlp, mlp_leaves, on_kernel, monkeypatch):  # noqa: F811
    """Documents a limit, not a defect: at i - 1 the report ties with the cut at i - 1, host
    first, so it gives the shape error that leg wants; no leaf lies between i - 1 and i."""
    i = first_product(mlp) + 1
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 2)
    shifted(monkeypatch, -1)
    nan_bracket(mlp, mlp_leaves, i, lambda b: hash_step_leaves(mlp, b))
