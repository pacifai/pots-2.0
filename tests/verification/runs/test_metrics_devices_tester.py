"""FS33c tester: the decoder identity, the merge's edge cases, real processes a step apart.

``fs33c_baseline_decoder.json`` was captured from commit efaaec5 (checked out in a scratch
worktree) the same way ``fs33c_baseline.json`` was: a tiny decoder through ``run_scenario``,
the timed run's rows and markers and the memory pass's rows, as names, calls and keys.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.runs.metrics import (
    STEP_WALL,
    MemoryRecorder,
    MetricsWriter,
    StepPart,
    TimeRecorder,
    memory_probe,
    read_markers,
)
from verification.runs.scenarios import HONEST, honest_final, run_scenario

from tests.verification.decoder_helpers import make_records, tiny_config

HERE = Path(__file__).resolve().parent
SEED, K, T = 1, 7, 2
CPU3 = ("cpu", "cpu", "cpu")


class _Rec:
    def __init__(self, t, rejection=None):
        self.t, self.rejection = t, rejection


def _schema(rows):
    return [[r["record"], r.get("step"), r.get("side"), r.get("component"), r.get("level"),
             r.get("calls"), sorted(r)] for r in rows]


def _marker_schema(lines):
    return [{"keys": sorted(m),
             "sections": [[x["name"], x["depth"], x["calls"], sorted(x), len(x["host_us"])]
                          for x in m.get("sections", [])]} for m in lines]


@pytest.fixture(scope="module")
def decoder():
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    final = honest_final(c, D, w0, T, seed=SEED)

    def go(rec):
        run_scenario(c, D, w0, HONEST, T=T, k=K, final=final, recorder=rec, seed=SEED)
        return rec
    return go


@pytest.fixture(scope="module")
def baseline():
    return json.loads((HERE / "fs33c_baseline_decoder.json").read_text())


# ---- the decoder through run_scenario: as before FS33c ---------------------------------------


@pytest.mark.parametrize("kw", [{}, {"devices": ("cpu",)}, {"devices": ()}])
def test_decoder_time_rows_and_markers_are_as_before(decoder, baseline, tmp_path, kw):
    with MetricsWriter(tmp_path, "g", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder(HONEST.name)
        if kw:
            rec = TimeRecorder(rec.run, HONEST.name, writer=mw, step_fields=rec.step_fields,
                               **kw)
        decoder(rec)
    assert _schema(rec.rows) == baseline["time_rows"]
    assert _marker_schema(read_markers(tmp_path / "markers.jsonl")) == baseline["markers"]
    # the time rows carry the step record and the verdict, which the baseline covers
    assert {"step", "verdict", "time"} <= {r[0] for r in baseline["time_rows"]}


@pytest.mark.parametrize("kw", [{}, {"devices": ("cpu",)}])
def test_decoder_memory_rows_are_as_before(decoder, baseline, kw):
    rec = decoder(MemoryRecorder("g", HONEST.name, memory_probe("cpu"), **kw))
    rec.finish()
    assert _schema(rec.rows) == baseline["memory_rows"]


def test_decoder_with_three_devices_adds_fields_only(decoder, baseline):
    rec = decoder(MemoryRecorder("g", HONEST.name, memory_probe("cpu"), devices=CPU3))
    rec.finish()
    assert len(rec.rows) == len(baseline["memory_rows"])
    for r, b in zip(rec.rows, baseline["memory_rows"], strict=True):
        assert _schema([r])[0][:6] == b[:6]
        assert set(b[6]) <= set(r)
    trec = decoder(TimeRecorder("g", HONEST.name, devices=CPU3))
    for r, b in zip(trec.rows, baseline["time_rows"], strict=True):
        assert _schema([r])[0][:6] == b[:6]


# ---- the merge's edge cases ---------------------------------------------------------------


def _timed(names, **kw):
    rec = TimeRecorder("r", "s", **kw)
    seam = rec.seam()
    for n in names:
        with seam(n):
            time.sleep(0.002)
    return rec


def test_merge_gives_each_side_once_with_pid_and_totals(tmp_path):
    child = TimeRecorder("r", "s")
    seam = child.seam()
    with seam("P3.commit"):
        with seam("P3.commit.hash"):
            time.sleep(0.002)
    part = child.close_step(1)
    part.pid = 4242
    with MetricsWriter(tmp_path, "r", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        parent = mw.recorder("s")
        with parent.seam()("5"):
            time.sleep(0.002)
        parent.merge_step(1, part)
        parent.on_step(_Rec(1))
    rows = {(r["level"], r["component"]): r for r in parent.rows if r["record"] == "time"}
    keys = [(r["level"], r["component"]) for r in parent.rows if r["record"] == "time"]
    assert len(keys) == len(set(keys))  # nothing twice
    assert {("phase", "P3.commit"), ("component", "P3"), ("sub", "P3.commit.hash"),
            ("component", "5"), ("total", "prover"), ("total", "verifier"), ("total", "step"),
            ("total", STEP_WALL)} <= set(keys)
    # the nested section is inside P3.commit, and in no total of its own
    assert rows[("sub", "P3.commit.hash")]["time_s"] <= rows[("phase", "P3.commit")]["time_s"]
    assert rows[("total", "step")]["time_s"] == pytest.approx(
        rows[("total", "prover")]["time_s"] + rows[("total", "verifier")]["time_s"])
    assert rows[("total", "prover")]["time_s"] == rows[("phase", "P3.commit")]["time_s"]
    (line,) = [m for m in read_markers(tmp_path / "markers.jsonl") if m["kind"] == "step"]
    by = {s["name"]: s for s in line["sections"]}
    assert by["P3.commit"]["pid"] == by["P3.commit.hash"]["pid"] == 4242
    assert by["5"]["pid"] != 4242
    # step_wall: first section start to last section end, both sides, from the markers' spans
    top = [by["P3.commit"], by["5"]]
    lo = min(s["host_us"][0][0] for s in top)
    hi = max(s["host_us"][-1][1] for s in top)
    assert rows[("total", STEP_WALL)]["time_s"] == pytest.approx((hi - lo) / 1e6, abs=2e-6)


def test_gpu_held_counts_each_side_with_its_devices(tmp_path):
    child = _timed(["P3.commit"])  # one device
    part = child.close_step(1)
    with MetricsWriter(tmp_path, "r", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("s")
        rec = TimeRecorder(rec.run, "s", writer=mw, step_fields=rec.step_fields, devices=CPU3)
        with rec.seam()("5"):
            time.sleep(0.002)
        rec.merge_step(1, part)
        rec.on_step(_Rec(1))
    by = {(r["level"], r["component"]): r for r in rec.rows if r["record"] == "time"}
    assert by[("total", "prover")]["gpu_held_s"] == pytest.approx(by[("total", "prover")]["time_s"])
    assert by[("total", "verifier")]["gpu_held_s"] == pytest.approx(
        3 * by[("total", "verifier")]["time_s"])
    assert by[("total", "step")]["gpu_held_s"] == pytest.approx(
        by[("total", "prover")]["time_s"] + 3 * by[("total", "verifier")]["time_s"])
    (line,) = [m for m in read_markers(tmp_path / "markers.jsonl") if m["kind"] == "step"]
    secs = {s["name"]: s for s in line["sections"]}
    assert secs["P3.commit"]["devices"] == ["cpu"]  # the child's own devices
    assert secs["5"]["devices"] == list(CPU3)


def test_a_step_the_child_never_closed_leaves_the_parent_alone(tmp_path):
    with MetricsWriter(tmp_path, "r", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("s")
        with rec.seam()("5"):
            pass
        rec.on_step(_Rec(1))
    assert not any(r.get("component") == STEP_WALL for r in rec.rows)
    (line,) = [m for m in read_markers(tmp_path / "markers.jsonl") if m["kind"] == "step"]
    assert "merged" not in line and all("pid" not in s for s in line["sections"])


def test_merging_the_same_part_twice_is_refused():
    parent = _timed(["5"])
    part = _timed(["P3.commit"]).close_step(1)
    parent.merge_step(1, part)
    with pytest.raises(ValueError, match="P3.commit"):
        parent.merge_step(1, part)
    assert list(parent._step) == ["5", "P3.commit"]  # the refusal left the table whole


def test_a_refused_merge_changes_nothing():
    parent = _timed(["P3.commit"])
    part = _timed(["P3.commit", "P1.label"]).close_step(1)
    with pytest.raises(ValueError):
        parent.merge_step(1, part)
    assert list(parent._step) == ["P3.commit"]  # P1.label was not half-merged


def test_a_non_part_is_refused():
    parent = _timed(["5"])
    with pytest.raises(ValueError):
        parent.merge_step(1, {"sections": {}})  # type: ignore[arg-type]


def test_a_memory_part_into_a_time_recorder_and_back():
    mem = MemoryRecorder("r", "s", memory_probe("cpu"))
    with mem.seam()("P3.commit"):
        pass
    mpart = mem.close_step(1)
    tpart = _timed(["P3.commit"]).close_step(1)
    with pytest.raises(ValueError, match="memory"):
        _timed(["5"]).merge_step(1, mpart)
    with pytest.raises(ValueError, match="time"):
        MemoryRecorder("r", "s", memory_probe("cpu")).merge_step(1, tpart)


@pytest.mark.xfail(strict=True, reason="DEFECT: merge_step after on_step(t) is accepted; the "
                   "part waits in the table and is emitted with step t + 1's rows")
def test_a_merge_after_on_step_is_refused():
    parent = _timed(["5"])
    part = _timed(["P3.commit"]).close_step(1)
    parent.on_step(_Rec(1))
    with pytest.raises((ValueError, RuntimeError)):
        parent.merge_step(1, part)


@pytest.mark.xfail(strict=True, reason="DEFECT: a late merge is emitted under the next step "
                   "(the parent records no step counter to compare t with)")
def test_a_late_part_never_lands_in_the_next_step():
    parent = _timed(["5"])
    part = _timed(["P3.commit"]).close_step(1)
    parent.on_step(_Rec(1))
    try:
        parent.merge_step(1, part)
    except (ValueError, RuntimeError):
        return
    with parent.seam()("5"):
        pass
    parent.on_step(_Rec(2))
    step2 = {r["component"] for r in parent.rows if r["step"] == 2}
    assert "P3.commit" not in step2


@pytest.mark.xfail(strict=True, reason="DEFECT: parts of steps 1 and 2 merged before on_step(1) "
                   "both go into step 1's table (merge_step checks the part's t, not the "
                   "recorder's step)")
def test_steps_out_of_order_are_refused():
    parent = _timed(["5"])
    p1 = _timed(["P3.commit"]).close_step(1)
    p2 = _timed(["P1.label"]).close_step(2)
    parent.merge_step(1, p1)
    with pytest.raises((ValueError, RuntimeError)):
        parent.merge_step(2, p2)


@pytest.mark.xfail(strict=True, reason="DEFECT: a part of step 2 merged as step 2 into a parent "
                   "still on step 1 lands in step 1's rows (no expected-step check)")
def test_a_part_of_a_later_step_does_not_land_in_the_current_one():
    parent = _timed(["5"])
    p2 = _timed(["P1.label"]).close_step(2)
    try:
        parent.merge_step(2, p2)
    except (ValueError, RuntimeError):
        return
    parent.on_step(_Rec(1))
    assert "P1.label" not in {r["component"] for r in parent.rows if r["step"] == 1}


@pytest.mark.xfail(strict=True, reason="DEFECT: the memory recorder's merge_step after "
                   "on_step(t) is accepted too, though the recorder knows its step (_t)")
def test_memory_merge_after_on_step_is_refused():
    parent = MemoryRecorder("r", "s", memory_probe("cpu"))
    with parent.seam()("5"):
        pass
    child = MemoryRecorder("r", "s", memory_probe("cpu"))
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    parent.on_step(_Rec(1))
    with pytest.raises((ValueError, RuntimeError)):
        parent.merge_step(1, part)


def test_a_traced_recorder_refuses_before_it_changes_anything(tmp_path):
    with MetricsWriter(tmp_path, "t", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("s", trace_step=1)
        with rec.seam()("5"):
            pass
        with pytest.raises(RuntimeError, match="trace"):
            rec.close_step(1)
        assert list(rec._step) == ["5"]  # the refusal left the table in place


def test_close_step_with_a_nested_section_resolves_and_pickles():
    import pickle  # noqa: PLC0415 (our own round trip, nothing untrusted)
    rec = TimeRecorder("r", "s", devices=CPU3)
    seam = rec.seam()
    with seam("P3.commit"):
        with seam("P3.commit.hash"):
            pass
    back = pickle.loads(pickle.dumps(rec.close_step(1)))
    assert isinstance(back, StepPart) and back.kind == "time" and back.t == 1
    assert back.sections["P3.commit.hash"].nested and back.sections["P3.commit"].calls == 1


def test_a_memory_part_pickles_and_merges_per_process():
    import pickle  # noqa: PLC0415 (our own round trip, nothing untrusted)
    child = MemoryRecorder("r", "s", memory_probe("cpu"), devices=CPU3)
    with child.seam()("P3.commit"):
        pass
    part = pickle.loads(pickle.dumps(child.close_step(1)))
    parent = MemoryRecorder("r", "s", memory_probe("cpu"), devices=CPU3)
    with parent.seam()("5"):
        pass
    parent.merge_step(1, part)
    parent.on_step(_Rec(1))
    by = {(r["level"], r["component"]): r for r in parent.rows}
    assert by[("phase", "P3.commit")]["device_peak_bytes"] is not None
    assert by[("component", "5")]["device_peak_bytes"] is not None
    assert by[("total", "prover")]["devices"] == list(CPU3)
    assert by[("total", "step")]["device_peak_bytes"] is None  # two processes: no per-device fact


def test_a_single_device_parent_with_a_several_device_child_memory():
    child = MemoryRecorder("r", "s", memory_probe("cpu"), devices=CPU3)
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    parent = MemoryRecorder("r", "s", memory_probe("cpu"))
    with parent.seam()("5"):
        pass
    parent.merge_step(1, part)
    parent.on_step(_Rec(1))
    by = {(r["level"], r["component"]): r for r in parent.rows}
    assert by[("component", "5")]["peak_bytes"] is not None
    assert by[("phase", "P3.commit")]["devices"] == list(CPU3)
    assert by[("component", "5")]["devices"] is None  # the parent had one device: no list


# ---- real processes, the child a step ahead --------------------------------------------------

PAUSES = {1: 0.02, 2: 0.15, 3: 0.30}


def _ahead_child(conn, steps):
    """Records P1.label and P3.commit for every step at once, sends the parts at the end."""
    from verification.runs.metrics import TimeRecorder  # noqa: PLC0415 (a spawned process)
    rec = TimeRecorder("r", "s")
    seam = rec.seam()
    parts = []
    for t in steps:
        with seam("P1.label"):
            time.sleep(PAUSES[t])
        with seam("P3.commit"):
            with seam("P3.commit.hash"):
                time.sleep(PAUSES[t] / 2)
        parts.append(rec.close_step(t))
    for p in parts:
        conn.send(p)
    conn.close()


def test_a_child_that_closed_every_step_before_the_parent_merged_the_first(tmp_path):
    ctx = torch.multiprocessing.get_context("spawn")
    parent_end, child_end = ctx.Pipe()
    proc = ctx.Process(target=_ahead_child, args=(child_end, (1, 2, 3)))
    proc.start()
    proc.join(60)  # the child is done with all three steps before the parent merges any
    assert proc.exitcode == 0
    with MetricsWriter(tmp_path, "r", model="m", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("s")
        seam = rec.seam()
        for t in (1, 2, 3):
            part = parent_end.recv()
            assert part.t == t
            with seam("2"):
                time.sleep(0.01)
            rec.merge_step(t, part)
            rec.on_step(_Rec(t))
    rows = [r for r in rec.rows if r["record"] == "time"]
    t_of = {t: {(r["level"], r["component"]): r for r in rows if r["step"] == t} for t in PAUSES}
    for t, pause in PAUSES.items():
        mine = t_of[t]
        assert len(mine) == len([r for r in rows if r["step"] == t])
        label = mine[("phase", "P1.label")]["time_s"]
        # each step's own sleep, and not the other steps': far from the other two pauses
        assert pause <= label < pause + 0.1, (t, label)
        assert mine[("phase", "P1.label")]["calls"] == 1
        assert mine[("sub", "P3.commit.hash")]["calls"] == 1
    lines = [m for m in read_markers(tmp_path / "markers.jsonl") if m["kind"] == "step"]
    assert [m["step"] for m in lines] == [1, 2, 3]
    spans = []
    for m in lines:
        by = {s["name"]: s for s in m["sections"]}
        assert len(by["P1.label"]["host_us"]) == 1 and by["P1.label"]["pid"] == proc.pid
        a, b = by["P1.label"]["host_us"][0]
        spans.append((m["t0_ns"] + a * 1000, m["t0_ns"] + b * 1000))
    # on one clock, the child's steps run one after the other, in order, and all three ended
    # before the parent's step 1 even began
    assert spans[0][1] <= spans[1][0] <= spans[1][1] <= spans[2][0]
    assert spans[2][1] <= lines[0]["t0_ns"] + max(
        s["host_us"][-1][1] for s in lines[0]["sections"] if s["name"] == "2") * 1000
