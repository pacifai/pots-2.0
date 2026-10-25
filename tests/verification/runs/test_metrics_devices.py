"""FS33c: several devices per recorder, and a step closed in one process, recorded in another.

The single-device identity compares against ``fs33c_baseline.json``, the row and markers layout
of the code before FS33c (commit efaaec5) on the MLP smoke run. Its figures are timings and byte
counts, so it compares what each row and line is (names, calls, keys), not their values.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch

from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs.metrics import (
    STEP_WALL,
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    StepPart,
    TimeRecorder,
    count_pass,
    counting,
    memory_probe,
    read_markers,
)
from verification.runs.mlp_smoke import scenarios
from verification.runs.scenarios import honest_final, run_scenario

SEED, K, T = 1, 7, 3
HERE = Path(__file__).resolve().parent
RESETTABLE = sys.platform == "darwin" or sys.platform.startswith("linux")
CPU3 = ("cpu", "cpu", "cpu")


class _Rec:
    def __init__(self, t, rejection=None):
        self.t, self.rejection = t, rejection


@pytest.fixture(scope="module")
def run():
    """``run(recorder)`` is the honest MLP scenario with ``recorder`` wired in."""
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    final = honest_final(c, D, w0, T, seed=SEED)
    s = next(x for x in scenarios(c, D, SEED) if x.name == "honest")

    def go(rec):
        run_scenario(c, D, w0, s, T=T, k=K, final=final, recorder=rec, seed=SEED)
        return rec
    return go


def _schema(rows):
    return [[r["record"], r.get("step"), r.get("side"), r.get("component"), r.get("level"),
             r.get("calls"), sorted(r)] for r in rows]


def _timed(run, tmp_path, **kw):
    with MetricsWriter(tmp_path, "g", model="mlp", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("honest")
        if kw:
            rec = TimeRecorder(rec.run, "honest", writer=mw, step_fields=rec.step_fields, **kw)
        run(rec)
    return rec.rows, read_markers(tmp_path / "markers.jsonl")


def _marker_schema(lines):
    return [{"keys": sorted(m),
             "sections": [[x["name"], x["depth"], x["calls"], sorted(x), len(x["host_us"])]
                          for x in m.get("sections", [])]} for m in lines]


def _memory(run, **kw):
    rec = run(MemoryRecorder("g", "honest", memory_probe("cpu"), **kw))
    rec.finish()
    return rec.rows


@pytest.fixture(scope="module")
def baseline():
    return json.loads((HERE / "fs33c_baseline.json").read_text())


# ---- one device or none: as before FS33c ---------------------------------------------------


@pytest.mark.parametrize("kw", [{}, {"devices": ("cpu",)}, {"device": "cpu"}, {"devices": ()}])
def test_one_device_or_none_time_is_as_before(run, baseline, tmp_path, kw):
    rows, markers = _timed(run, tmp_path, **kw)
    assert _schema(rows) == baseline["time_rows"]
    assert _marker_schema(markers) == baseline["markers"]


@pytest.mark.parametrize("kw", [{}, {"devices": ("cpu",)}, {"devices": []}])
def test_one_device_or_none_memory_is_as_before(run, baseline, kw):
    assert _schema(_memory(run, **kw)) == baseline["memory_rows"]


def test_device_given_twice_must_agree():
    with pytest.raises(ValueError, match="not the first"):
        TimeRecorder("r", "s", device="cpu", devices=("meta", "cpu"))
    with pytest.raises(ValueError, match="one type"):
        TimeRecorder("r", "s", devices=("cpu", "meta"))


# ---- several devices (on the Mac: three "cpu"s, which are all the process) ------------------


def test_three_devices_time_rows_are_as_before_plus_fields(run, baseline, tmp_path):
    rows, markers = _timed(run, tmp_path, devices=CPU3)
    got = _schema(rows)
    time_rows = [r for r in rows if r["record"] == "time"]
    assert len(got) == len(baseline["time_rows"])
    for g, b in zip(got, baseline["time_rows"], strict=True):
        assert g[:6] == b[:6]
        if g[0] == "time":
            assert set(g[6]) == set(b[6]) | {"devices", "gpu_held_s"}
        else:
            assert g[6] == b[6]
    for r in time_rows:
        if r["component"] == "step_interval":  # FS42: each device once, as ``step_wall`` lists them
            assert r["devices"] == ["cpu"] and r["gpu_held_s"] == pytest.approx(r["time_s"])
            continue
        assert r["devices"] == ["cpu", "cpu", "cpu"]  # the Mac's cpus are one device, the process
        assert r["gpu_held_s"] == pytest.approx(3 * r["time_s"])
    # the markers line names the devices and each section the ones it ran on
    base = baseline["markers"]
    assert [m["keys"] for m in _marker_schema(markers)][1:] == \
           [sorted(set(m["keys"]) | {"devices"}) for m in base][1:]
    for m in markers[1:]:
        assert m["devices"] == list(CPU3)
        assert all(sec["devices"] == list(CPU3) for sec in m["sections"])


def test_three_devices_memory_rows_are_as_before_plus_fields(run, baseline):
    rows = _memory(run, devices=CPU3)
    assert len(rows) == len(baseline["memory_rows"])
    for r, b in zip(rows, baseline["memory_rows"], strict=True):
        assert _schema([r])[0][:6] == b[:6]
        assert set(r) == set(b[6]) | {
            "devices", "device_peak_bytes", "device_peak_reserved_bytes", "device_start_bytes",
            "device_end_bytes", "device_wide_start_bytes", "device_wide_end_bytes",
            "device_wide_peak_bytes", "device_wide_source"}
        assert r["devices"] == list(CPU3)
        # the first device's figures are today's fields; a host device's are the process's
        assert r["device_peak_bytes"][0] == r["peak_bytes"]
        assert r["device_peak_bytes"] == [r["peak_bytes"]] * 3
        assert r["device_start_bytes"] == [r["start_bytes"]] * 3
        assert r["device_end_bytes"] == [r["end_bytes"]] * 3
        assert r["device_wide_peak_bytes"] == [r["peak_bytes"]] * 3 or not RESETTABLE
        assert all(x.startswith("process") for x in r["device_wide_source"])


def test_three_devices_memory_is_not_larger_than_a_sample(run):
    if not RESETTABLE:
        pytest.skip("no resettable host peak")
    rows = [r for r in _memory(run, devices=CPU3) if r["level"] in ("phase", "component")]
    for r in rows:
        # the device-wide figure is sampled at entry, exit and nested boundaries: a lower bound
        assert r["device_wide_peak_bytes"][0] >= max(r["device_wide_start_bytes"][0],
                                                     r["device_wide_end_bytes"][0])


# ---- a step closed in one process, recorded in another --------------------------------------


def _child(conn, steps, pause):
    """The pipelined prover's process: records P1.label and P3.commit per step, sends the part."""
    from verification.runs.metrics import TimeRecorder  # noqa: PLC0415 (a spawned process)
    rec = TimeRecorder("r", "s")
    seam = rec.seam()
    for t in steps:
        with seam("P1.label"):
            time.sleep(pause)
        with seam("P3.commit"):
            time.sleep(pause)
        conn.send(rec.close_step(t))
    conn.close()


def test_two_real_processes_one_set_of_rows_and_one_clock(tmp_path):
    ctx = torch.multiprocessing.get_context("spawn")
    pause = 0.05
    parent, child = ctx.Pipe()
    before = time.monotonic_ns()
    proc = ctx.Process(target=_child, args=(child, (1, 2), pause))
    proc.start()
    try:
        with MetricsWriter(tmp_path, "r", model="mlp", corpus="synthetic", seed=0,
                           config={"a": 1}) as mw:
            rec = mw.recorder("s")
            seam = rec.seam()
            for t in (1, 2):
                assert parent.poll(60), "the child sent no part"  # a dead child never blocks us
                part = parent.recv()
                assert isinstance(part, StepPart) and part.t == t and part.pid == proc.pid
                seen = time.monotonic_ns()
                with seam("2"):
                    time.sleep(pause)
                with seam("5"):
                    time.sleep(pause)
                rec.merge_step(t, part)
                rec.on_step(_Rec(t))
            after = time.monotonic_ns()
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0
    rows = [r for r in rec.rows if r["record"] == "time"]
    for t in (1, 2):
        mine = {(r["level"], r["component"]): r for r in rows if r["step"] == t}
        assert len(mine) == len([r for r in rows if r["step"] == t])  # one row per name
        for key in (("phase", "P1.label"), ("phase", "P3.commit"), ("component", "2"),
                    ("component", "5"), ("total", "prover"), ("total", "verifier"),
                    ("total", "step"), ("total", STEP_WALL)):
            assert key in mine, key
        work = sum(mine[("phase", p)]["time_s"] for p in ("P1.label", "P3.commit")) + \
            mine[("component", "2")]["time_s"] + mine[("component", "5")]["time_s"]
        assert mine[("total", "step")]["time_s"] == pytest.approx(work)
        assert mine[("total", "prover")]["time_s"] == pytest.approx(
            mine[("phase", "P1.label")]["time_s"] + mine[("phase", "P3.commit")]["time_s"])
        wall = mine[("total", STEP_WALL)]["time_s"]
        # the wall extent is the work minus the overlap plus the gaps between sections; gaps
        # grow under load, so bound it by the whole run's clock window, not by the work
        assert wall <= (after - before) / 1e9
        assert wall >= max(mine[("total", "prover")]["time_s"],
                           mine[("total", "verifier")]["time_s"])
        assert all(r["devices"] == ["cpu"] for r in mine.values())
        assert mine[("total", "step")]["gpu_held_s"] == pytest.approx(work)
    # step 1's wall is shorter than its work: the child's step 2 ran under the parent's step 1
    lines = [m for m in read_markers(tmp_path / "markers.jsonl") if m.get("kind") == "step"]
    assert [m["step"] for m in lines] == [1, 2]
    for m in lines:
        assert m["merged"] is True
        by = {s["name"]: s for s in m["sections"]}
        assert {"P1.label", "P3.commit", "2", "5"} <= set(by)
        assert by["P1.label"]["pid"] == proc.pid and by["2"]["pid"] != proc.pid
        for name, s in by.items():  # every span, both sides, on one clock, inside what the
            for a, b in s["host_us"]:  # parent saw the child live
                assert before <= m["t0_ns"] + a * 1000 <= m["t0_ns"] + b * 1000 <= after + 1000
        assert by["P1.label"]["host_us"][0][1] <= by["P3.commit"]["host_us"][0][0] + 1
    # the child's step 2 began before the parent finished step 1: a reader sees the overlap
    assert lines[1]["t0_ns"] < lines[0]["t1_ns"]
    assert seen >= before


def test_a_name_in_both_parts_raises():
    child = TimeRecorder("r", "s")
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    parent = TimeRecorder("r", "s")
    with parent.seam()("P3.commit"):
        pass
    with pytest.raises(ValueError, match="P3.commit"):
        parent.merge_step(1, part)
    mem = MemoryRecorder("r", "s", memory_probe("cpu"))
    with mem.seam()("P3.commit"):
        pass
    mpart = MemoryRecorder("r", "s", memory_probe("cpu"))
    with mpart.seam()("P3.commit"):
        pass
    with pytest.raises(ValueError, match="P3.commit"):
        mem.merge_step(1, mpart.close_step(1))


def test_a_part_of_another_step_or_kind_is_refused():
    child = TimeRecorder("r", "s")
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    with pytest.raises(ValueError, match="step 2"):
        TimeRecorder("r", "s").merge_step(2, part)
    with pytest.raises(ValueError, match="memory"):
        MemoryRecorder("r", "s", memory_probe("cpu")).merge_step(1, part)


def test_close_step_needs_no_open_section_and_empties_the_step():
    rec = TimeRecorder("r", "s")
    seam = rec.seam()
    with seam("P3.commit"):
        with pytest.raises(RuntimeError, match="still open"):
            rec.close_step(1)
    part = rec.close_step(1)
    assert list(part.sections) == ["P3.commit"]
    assert rec.close_step(2).sections == {}  # the table was handed over, not copied
    assert rec.rows == []  # nothing emitted


def test_run_sections_travel_as_step_0():
    child = TimeRecorder("r", "s")
    with child.seam()("run:0"):
        pass
    part = child.close_step(0)
    parent = TimeRecorder("r", "s")
    with parent.seam()("run:1"):
        pass
    parent.merge_step(0, part)
    parent.finish()
    assert {r["component"] for r in parent.rows if r["level"] == "component"} == {"0", "1"}
    assert not any(r["component"] == STEP_WALL for r in parent.rows)


def test_a_part_pickles_without_cuda_or_tensors():
    import pickle  # noqa: PLC0415
    rec = TimeRecorder("r", "s", devices=CPU3)
    with rec.seam()("P3.commit"):
        with rec.seam()("P3.commit.hash"):
            pass
    part = pickle.loads(pickle.dumps(rec.close_step(1)))
    assert part.devices == CPU3 and set(part.sections) == {"P3.commit", "P3.commit.hash"}


def test_a_merged_memory_step_is_one_set_of_rows():
    child = MemoryRecorder("r", "s", memory_probe("cpu"))
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    parent = MemoryRecorder("r", "s", memory_probe("cpu"))
    with parent.seam()("5"):
        pass
    parent.merge_step(1, part)
    parent.on_step(_Rec(1))
    comps = [(r["level"], r["component"]) for r in parent.rows]
    assert len(comps) == len(set(comps))
    assert {("phase", "P3.commit"), ("component", "P3"), ("component", "5"),
            ("total", "prover"), ("total", "verifier"), ("total", "step")} <= set(comps)


def test_a_merged_memory_step_with_devices_keeps_per_process_figures():
    child = MemoryRecorder("r", "s", memory_probe("cpu"), devices=CPU3)
    with child.seam()("P3.commit"):
        pass
    part = child.close_step(1)
    parent = MemoryRecorder("r", "s", memory_probe("cpu"), devices=CPU3)
    with parent.seam()("5"):
        pass
    parent.merge_step(1, part)
    parent.on_step(_Rec(1))
    by = {(r["level"], r["component"]): r for r in parent.rows}
    assert by[("phase", "P3.commit")]["devices"] == list(CPU3)
    # the step total mixes two processes: no per-device figure is true of both
    assert by[("total", "step")]["devices"] is None
    assert by[("total", "step")]["device_peak_bytes"] is None


# ---- the counting pass and the trace stay in one process ------------------------------------


def test_the_counting_pass_refuses_to_split_a_step(c=None):
    with counting() as (fm, tally):
        rec = CountRecorder("r", "s", fm, tally)
        with pytest.raises(RuntimeError, match="process-global"):
            rec.close_step(1)
        part = TimeRecorder("r", "s").close_step(1)
        with pytest.raises(RuntimeError, match="process-global"):
            rec.merge_step(1, part)
    assert count_pass("r", "s", lambda rec: None) == []


def test_a_traced_recorder_refuses_to_split_a_step(tmp_path):
    with MetricsWriter(tmp_path, "t", model="mlp", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        rec = mw.recorder("s", trace_step=1)
        with pytest.raises(RuntimeError, match="trace"):
            rec.close_step(1)
        part = TimeRecorder("r", "s").close_step(1)
        with pytest.raises(RuntimeError, match="trace"):
            rec.merge_step(1, part)


# ---- the CUDA paths against a stand-in for torch.cuda (the Mac cannot run the real ones) -----


class _FakeCuda:
    """Just enough of ``torch.cuda`` for the recorders: events stamped by a counter in
    microseconds, per-device allocated and peak bytes, and ``mem_get_info`` that counts
    another process's bytes."""

    def __init__(self, monkeypatch, n=3):
        self.sync_calls, self.events, self.clock = [], 0, 0
        self.alloc = [0] * n
        self.peak = [0] * n
        self.other = [0] * n
        self.total = 80 * 2**30
        for name in ("synchronize", "current_stream", "memory_stats", "max_memory_allocated",
                     "memory_allocated", "max_memory_reserved", "reset_peak_memory_stats",
                     "mem_get_info"):
            monkeypatch.setattr(torch.cuda, name, getattr(self, name))
        fake = self

        class Event:
            def __init__(self, enable_timing=False):
                fake.events += 1

            def record(self, stream=None):
                fake.clock += 100
                self.at, self.dev = fake.clock, stream

            def elapsed_time(self, other):
                assert self.dev == other.dev  # events only compare on one device
                return (other.at - self.at) / 1000.0  # ms

        monkeypatch.setattr(torch.cuda, "Event", Event)

    @staticmethod
    def _i(d):
        return torch.device(d).index

    def synchronize(self, d=None):
        self.sync_calls.append(self._i(d))

    def current_stream(self, d=None):
        return ("stream", self._i(d))

    def memory_stats(self, d=None):
        return {}

    def max_memory_allocated(self, d=None):
        return self.peak[self._i(d)]

    def memory_allocated(self, d=None):
        return self.alloc[self._i(d)]

    def max_memory_reserved(self, d=None):
        return self.peak[self._i(d)] + 7

    def reset_peak_memory_stats(self, d=None):
        self.peak[self._i(d)] = self.alloc[self._i(d)]

    def mem_get_info(self, d=None):
        i = self._i(d)
        return self.total - self.alloc[i] - self.other[i], self.total

    def grab(self, i, nbytes):
        self.alloc[i] += nbytes
        self.peak[i] = max(self.peak[i], self.alloc[i])


def test_the_cuda_time_path_with_three_devices(monkeypatch):
    fake = _FakeCuda(monkeypatch)
    devs = ("cuda:0", "cuda:1", "cuda:2")
    rec = TimeRecorder("r", "s", devices=devs)
    seam = rec.seam()
    with seam("5"):
        with seam("5.measure"):
            pass
    assert fake.sync_calls == [0, 1, 2, 0, 1, 2]  # a top-level section: n at entry, n at exit
    assert fake.events == 3 * 4  # the anchors, the nested pair, the exit's: 2n each
    part = rec.close_step(1)
    top, nested = part.sections["5"], part.sections["5.measure"]
    assert top.time > 0 and nested.time > 0
    assert len(top.gpu) == 1 and len(top.gpu_more) == 2 and len(top.gpu_more[0]) == 1
    assert all(g is not None for g in (top.gpu[0], top.gpu_more[0][0], top.gpu_more[1][0]))
    assert top.devs == devs
    # one device is the n = 1 case: 2 synchronizes and 2 events for a top-level section
    fake.sync_calls.clear()
    fake.events = 0
    one = TimeRecorder("r", "s", device="cuda:0")
    with one.seam()("5"):
        pass
    assert fake.sync_calls == [0, 0] and fake.events == 2


def test_the_cuda_memory_path_with_three_devices(monkeypatch):
    fake = _FakeCuda(monkeypatch)
    devs = ("cuda:0", "cuda:1", "cuda:2")
    rec = MemoryRecorder("r", "s", memory_probe(devs[0]), devices=devs)
    fake.other[2] = 5 * 2**30  # another process's tensor on cuda:2
    with rec.seam()("5"):
        fake.grab(1, 3 * 2**20)  # this process's, on cuda:1
        with rec.seam()("5.measure"):
            fake.grab(1, 2**20)
        fake.alloc[1] -= 4 * 2**20
    rec.on_step(_Rec(1))
    by = {r["component"]: r for r in rec.rows if r["level"] in ("component", "sub")}
    row = by["5"]
    assert row["devices"] == list(devs)
    assert row["device_peak_bytes"] == [0, 4 * 2**20, 0]
    assert row["device_peak_reserved_bytes"] == [7, 4 * 2**20 + 7, 7]
    assert row["peak_bytes"] == 0 and row["start_bytes"] == 0  # the first device, as before
    # device-wide use: the other process's 5 GiB on cuda:2, and the sampled peak
    assert row["device_wide_start_bytes"] == [0, 0, 5 * 2**30]
    assert row["device_wide_peak_bytes"][1] == 4 * 2**20  # seen at the nested entry or exit
    assert row["device_wide_peak_bytes"][2] == 5 * 2**30
    assert row["device_wide_end_bytes"] == [0, 0, 5 * 2**30]
    assert "lower bound" in row["device_wide_source"][1]
    assert by["5.measure"]["device_peak_bytes"][1] == 4 * 2**20
