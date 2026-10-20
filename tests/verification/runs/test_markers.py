"""FS19: the timed run's section spans and check 5's stamps (``markers.jsonl``) and the check-5
layout (``c5_layout.json``)."""

import json
import types
from contextlib import nullcontext

import pytest
import torch

from evaluation.results.profile import io as profile_io
from tests.verification.decoder_helpers import make_records, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs import llama_step, metrics
from verification.runs.metrics import (
    MAX_SPANS,
    STAMP_NAMES,
    MetricsWriter,
    TimeRecorder,
    read_markers,
    read_records,
    resolve_gpu_us,
)
from verification.runs.mlp_smoke import scenarios
from verification.runs.scenarios import HONEST, honest_final, run_scenario
from verification.verifier import checks
from verification.verifier.context import StepContext

SEED = 1
K = 7


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    return c, D, w0


@pytest.fixture(scope="module")
def decoder_run(tiny, tmp_path_factory):
    c, D, w0 = tiny
    T = 2
    out = tmp_path_factory.mktemp("markers")
    final = honest_final(c, D, w0, T, seed=SEED)
    h_D = dataset_tree(c, D).root
    with MetricsWriter(out, "tiny", device="cpu", model="tiny", corpus="synthetic", seed=0,
                       config={}, band_file_hash=None, h_D=h_D) as mw:
        mw.write_c5_layout(c)
        r = llama_step.run_honest(c, D, w0, T=T, k=K, h_D=h_D, final=final,
                                  recorder=mw.recorder(HONEST.name), seed=SEED)
    return out, c, r, T


def test_spans_sum_to_the_time_rows(decoder_run):
    out, _, r, T = decoder_run
    lines = read_markers(out)
    assert lines[0]["kind"] == "header" and lines[0]["run"] == "tiny"
    steps = {x["step"]: x for x in lines if x["kind"] == "step"}
    run = next(x for x in lines if x["kind"] == "run")
    assert sorted(steps) == list(range(1, T + 1))
    checked = 0
    for row in read_records(out, "time"):
        if row["level"] not in ("phase", "sub") and not (
                row["level"] == "component" and row["side"] in ("verifier", "run")):
            continue
        line = run if row["step"] == 0 else steps[row["step"]]
        sec = next(s for s in line["sections"] if s["name"] == row["component"])
        assert sec["calls"] == row["calls"]
        total = sum(e - s for s, e in sec["host_us"]) / 1e6
        if sec["rest"] is not None:
            total += sec["rest"]["host_s"]
        assert total == pytest.approx(row["time_s"], abs=1e-3, rel=0.01)
        checked += 1
    assert checked > 20
    s1 = steps[1]
    assert s1["t0_ns"] < s1["t1_ns"] and not s1["traced"]
    assert min(s["host_us"][0][0] for s in s1["sections"] if s["host_us"]) == 0.0
    assert all(s["gpu_us"] is None and s["alloc"] is None for s in s1["sections"])


def test_stamp_counts_equal_the_layout(decoder_run):
    out, c, _, T = decoder_run
    layout = json.loads((out / "c5_layout.json").read_text())
    assert layout["version"] == 1 and layout["run"] == "tiny"
    assert layout["entries"] == layout["entries"][:layout["n_groups"]]
    assert layout["n_runs"] == len(list(checks._member_runs(c.products)))
    assert any(e["kind"] == "members" and e["members"] > 1 for e in layout["entries"])
    assert all("layer" in e and "class" in e for e in layout["entries"])
    assert {e["run"] for e in layout["entries"]} == set(range(layout["n_runs"]))
    for line in read_markers(out):
        if line["kind"] != "step":
            continue
        st = line["stamps"]
        assert line["stamp_names"] == list(STAMP_NAMES)
        n = len(st["name"])
        assert all(len(v) == n for v in st.values() if v is not None)
        count = {name: st["name"].count(i) for i, name in enumerate(line["stamp_names"])}
        assert count["c5.run"] == layout["n_runs"]
        for name in ("c5.glue", "c5.serve", "c5.copy", "c5.measure"):
            assert count[name] == layout["n_groups"], name
        assert count["c5.judge"] == layout["n_windows"]
        by = {}
        for i in range(n):
            by.setdefault(line["stamp_names"][st["name"][i]], []).append(
                (st["run"][i], st["group"][i], st["window"][i]))
        assert sorted(by["c5.measure"]) == sorted((e["run"], e["group"], -1)
                                                  for e in layout["entries"])
        assert sorted(by["c5.run"]) == [(i, -1, -1) for i in range(layout["n_runs"])]
        for name in ("c5.glue", "c5.serve", "c5.copy"):
            assert sorted(by[name]) == sorted((e["run"], e["group"], -1)
                                              for e in layout["entries"]), name
        assert sorted(by["c5.judge"]) == [(-1, -1, w) for w in range(layout["n_windows"])]
        assert all(h1 >= h0 for h0, h1 in zip(st["h0"], st["h1"]))
        assert st["g0"] is None and st["g1"] is None  # no device clock on CPU: no column


def test_group_ids_are_global_across_the_runs(decoder_run):
    out, c, _, _ = decoder_run
    layout = json.loads((out / "c5_layout.json").read_text())
    entries = layout["entries"]
    assert len({e["run"] for e in entries}) >= 2
    assert [e["group"] for e in entries] == list(range(layout["n_groups"]))
    assert layout["n_groups"] == len(entries)
    by_group = {e["group"]: e["run"] for e in entries}
    for line in read_markers(out):
        if line["kind"] != "step":
            continue
        st = line["stamps"]
        for i in range(len(st["name"])):
            if st["group"][i] >= 0:
                assert by_group[st["group"][i]] == st["run"][i]
    # the layout one group per class would join: a group id never carries two classes
    cls = {}
    for e in entries:
        cls.setdefault(e["group"], set()).add(e["class"])
    assert all(len(v) == 1 for v in cls.values())


def test_layout_trace_slice_and_windows(tiny, monkeypatch):
    c, _, _ = tiny
    f = metrics.c5_layout_file(c, "x")
    first = {}
    for e in f["entries"]:  # over every group, not only a run's first
        first.setdefault(e["class"], e["run"])
    runs = {e["run"] for e in f["entries"]}
    want = sorted(set(first.values()) | {i for i in runs if i % 20 == 0})
    assert f["trace_slice"] == want
    assert f["trace_slice"] == sorted(set(f["trace_slice"]))
    assert f["n_windows"] == 1 and {e["window"] for e in f["entries"]} == {0}
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", 5)
    entries = checks.check5_layout(c)
    pending = window = 0
    for i, run in enumerate(checks._member_runs(c.products)):
        assert {e["window"] for e in entries if e["run"] == i} == {window}
        pending += len(run)
        if pending >= 5:
            window, pending = window + 1, 0
    # a group split by the byte limit
    monkeypatch.setattr(checks, "_GROUP_BYTES", 4 * 4 * 12 * 12)
    assert len(checks.check5_layout(c)) > len(entries)
    assert sum(e["members"] for e in checks.check5_layout(c)) == c.M


def test_recorder_on_and_off_are_bit_identical_on_a_decoder(tiny, tmp_path):
    c, D, w0 = tiny
    final = honest_final(c, D, w0, 2, seed=SEED)
    h_D = dataset_tree(c, D).root
    off = llama_step.run_honest(c, D, w0, T=2, k=K, h_D=h_D, final=final, seed=SEED)
    with MetricsWriter(tmp_path, "tiny", device="cpu", config={}) as mw:
        on = llama_step.run_honest(c, D, w0, T=2, k=K, h_D=h_D, final=final, seed=SEED,
                                   recorder=mw.recorder(HONEST.name))
    assert on.loop.verdict == off.loop.verdict
    assert on.verifier.stats == off.verifier.stats
    assert [(x.t, x.rejection, x.loss) for x in on.loop.steps] == \
           [(x.t, x.rejection, x.loss) for x in off.loop.steps]
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))
    assert (tmp_path / "markers.jsonl").exists()


class _LogRecorder(TimeRecorder):
    """Logs every stamp call, whether or not a check-5 section is open."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def stamp(self, name, run=-1, group=-1, window=-1):
        self.calls.append((name, {"run": run, "group": group, "window": window}))
        return nullcontext()


def test_record_pass_writes_no_stamps(monkeypatch):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    T = 2
    final = honest_final(c, D, w0, T, seed=SEED)
    s = next(x for x in scenarios(c, D, SEED) if x.name == "flip")
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", 1)
    rec = _LogRecorder("r", "flip")
    res = run_scenario(c, D, w0, s, T=T, k=K, final=final, recorder=rec, seed=SEED)
    assert res.loop.rejection is not None
    runs = [i["run"] for n, i in rec.calls if n == "c5.run"]
    # step 1 is honest and measures every run; the flip is P_2's, so at step 2 two runs are
    # measured before the rejection and the record pass measures the rest, unstamped
    assert runs == [*range(c.M), 0, 1]
    assert [i["window"] for n, i in rec.calls if n == "c5.judge"][-2:] == [0, 1]
    done = {p.m for p in res.verifier.stats[2].products}
    assert len(done) == c.M  # the record pass did measure the rest, unstamped


def test_stamp_is_a_null_context_without_a_stamping_seam():
    def ctx(section):
        return StepContext(step=1, indices=(), h_D=b"\0" * 32, n_records=1, prev_w_hashes=(),
                           chain_check_id="0", k=1, eps_in=1.0, eps_acc=1.0, eps_w=1.0,
                           section=section)

    for section in (None, lambda name: nullcontext()):
        with ctx(section).stamp("c5.run", run=3) as x:
            assert x is None
    rec = TimeRecorder("r", "s")
    seam = rec.seam()
    assert seam is rec.seam()
    with ctx(seam).stamp("c5.run", run=3):  # no check-5 section open: nothing recorded
        pass
    assert rec._stamps == []
    with seam("5"):
        with ctx(seam).stamp("c5.run", run=3):
            pass
    assert [r[:4] for r in rec._stamps] == [(0, 3, -1, -1)]


def test_stamp_gate_follows_the_check5_sections():
    rec = TimeRecorder("r", "s")
    assert rec.stamp("c5.run", run=0).__class__ is nullcontext
    with rec.section("6a"):
        with rec.stamp("c5.run", run=0):
            pass
    assert rec._stamps == []
    with rec.section("5"):
        with rec.section("5.glue"):
            with rec.stamp("c5.glue", run=1, group=2):
                pass
        with rec.stamp("custom"):
            pass
    assert rec._c5_open == 0
    assert [r[:4] for r in rec._stamps] == [(1, 1, 2, -1), (6, -1, -1, -1)]
    assert rec._stamp_names[6] == "custom"


def test_spans_are_capped_and_the_rest_aggregates():
    rec = TimeRecorder("r", "s")
    n = MAX_SPANS + 10
    for _ in range(n):
        with rec.section("4"):
            pass
    acc = rec._step["4"]
    assert acc.calls == n and len(acc.spans) == MAX_SPANS
    assert acc.rest[0] == 10
    rec.on_step(type("R", (), {"t": 1, "rejection": None})())
    assert rec.rows  # still emits rows without a writer


class _Ev:
    """A CUDA event on a clock that moves 1 ms per read."""

    now = 0.0

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        _Ev.now += 1.0
        self.t = _Ev.now

    def elapsed_time(self, other):
        return other.t - self.t


def test_resolve_gpu_us_on_fake_events():
    a, e1, e2 = _Ev(), _Ev(), _Ev()
    a.t, e1.t, e2.t = 10.0, 12.5, 11.0
    assert resolve_gpu_us(1000.0, a, [a, e1, e2]) == [1000.0, 3500.0, 2000.0]


@pytest.fixture
def fake_cuda(monkeypatch):
    _Ev.now = 0.0
    syncs, reads = [], []

    def stats(device=None):
        reads.append(1)
        n = len(reads)
        return {"num_alloc_retries": n, "num_device_alloc": 10 * n,
                "num_device_free": 100 * n, "num_ooms": 0}
    monkeypatch.setattr(torch.cuda, "Event", _Ev)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: syncs.append(1))
    monkeypatch.setattr(torch.cuda, "memory_stats", stats)
    rec = TimeRecorder("r", "s")
    rec._Section, rec._cuda = metrics._CudaTimeSection, True
    return rec, syncs, reads


def test_cuda_section_alloc_and_no_sync_inside(fake_cuda):
    rec, syncs, reads = fake_cuda
    with rec.section("5"):
        with rec.section("5.glue"):
            with rec.stamp("c5.glue", run=0, group=0):
                pass
        with rec.stamp("c5.run", run=0):
            pass
    assert len(syncs) == 2  # the top-level entry and exit only
    assert len(reads) == 2
    top, nested = rec._step["5"], rec._step["5.glue"]
    # the counters read 1 then 2 (retries), 10 then 20 (allocs), 100 then 200 (frees)
    assert top.alloc == [1, 10, 100, 0] and nested.alloc is None
    rec._step_stamps = (rec._stamps, rec._stamp_gpu)  # what on_step hands over
    line = rec._marker_line(1, rec._step, type("R", (), {"t": 1})())
    sec = {s["name"]: s for s in line["sections"]}
    assert sec["5"]["alloc"] == {"retries": 1, "device_alloc": 10, "device_free": 100, "ooms": 0}
    assert sec["5.glue"]["alloc"] is None and sec["5"]["depth"] == 0
    assert sec["5.glue"]["depth"] == 1
    g = sec["5.glue"]["gpu_us"][0]
    top_g = sec["5"]["gpu_us"][0]
    assert top_g[0] == pytest.approx(0.0, abs=1e-3)  # the anchor is the section's start
    assert top_g[1] > g[1] > g[0] > top_g[0]
    st = line["stamps"]
    assert len(st["g0"]) == 2 and all(x is not None for x in st["g0"] + st["g1"])
    assert rec._pending == [] and rec._stamp_pending == []
    assert rec._step["5.glue"].time > 0  # the event pair's elapsed time, in seconds


def test_read_markers_round_trip_and_rejects_a_line_without_kind(tmp_path):
    with MetricsWriter(tmp_path, "r", device="cpu") as mw:
        mw.write_markers({"kind": "step", "step": 1})
        mw.write_markers({"kind": "run", "scenario": "s", "sections": []})
    lines = read_markers(tmp_path)
    assert [x["kind"] for x in lines] == ["header", "step", "run"]
    head = lines[0]
    assert head["version"] == 1 and head["run"] == "r" and head["device"] == "cpu"
    assert head["device"] == torch.device("cpu").type
    assert head["mono_ns"] > 0 and head["wall_unix_s"] > 1e9 and head["pid"] > 0
    assert read_markers(tmp_path / "markers.jsonl") == lines
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(head) + "\n" + json.dumps({"step": 2}) + "\n")
    with pytest.raises(ValueError, match="kind"):
        read_markers(bad)
    bad.write_text(json.dumps({"kind": "step"}) + "\n")
    with pytest.raises(ValueError, match="header"):
        read_markers(bad)


def test_no_markers_file_without_a_timed_step(tmp_path):
    with MetricsWriter(tmp_path, "r", device="cpu"):
        pass
    assert not (tmp_path / "markers.jsonl").exists()


def test_header_device_is_the_device_type(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "memory_probe", lambda device: types.SimpleNamespace(source="stub"))
    monkeypatch.setattr(metrics, "environment_record", lambda run, **kw: {"record": "environment"})
    with MetricsWriter(tmp_path, "r", device="cuda:1") as mw:
        mw.write_markers({"kind": "step", "step": 1})
    assert read_markers(tmp_path)[0]["device"] == "cuda"


def test_fs25_read_markers_accepts_the_file(decoder_run):
    out, _, _, T = decoder_run
    mk = profile_io.read_markers(out / "markers.jsonl")
    assert mk.header["device"] == "cpu" and len(mk.steps) == T and len(mk.runs) == 1
    for s in mk.steps:
        assert s["stamps"]["g0"] is None and s["stamps"]["g1"] is None
        assert len(s["stamps"]["name"]) > 0
    layout = profile_io.read_c5_layout(out / profile_io.C5_LAYOUT_FILE)
    assert layout is not None


def test_n_windows_counts_judge_calls_when_the_last_run_closes_a_window(tiny, tmp_path,
                                                                        monkeypatch):
    c, D, w0 = tiny
    runs = list(checks._member_runs(c.products))
    # a window the last run exactly fills: every window closes on a run boundary
    total = sum(len(r) for r in runs)
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", total)
    entries, n_windows = checks._layout(c)
    assert {e["window"] for e in entries} == {0} and n_windows == 2
    assert metrics.c5_layout_file(c, "x")["n_windows"] == 2
    assert checks.check5_layout(c) == entries
    final = honest_final(c, D, w0, 1, seed=SEED)
    h_D = dataset_tree(c, D).root
    with MetricsWriter(tmp_path, "tiny", device="cpu", config={}, h_D=h_D) as mw:
        mw.write_c5_layout(c)
        llama_step.run_honest(c, D, w0, T=1, k=K, h_D=h_D, final=final,
                              recorder=mw.recorder(HONEST.name), seed=SEED)
    layout = json.loads((tmp_path / "c5_layout.json").read_text())
    line = next(x for x in read_markers(tmp_path) if x["kind"] == "step")
    st = line["stamps"]
    judge = line["stamp_names"].index("c5.judge")
    windows = [st["window"][i] for i in range(len(st["name"])) if st["name"][i] == judge]
    assert len(windows) == layout["n_windows"] == 2
    assert all(w < layout["n_windows"] for w in windows)


def test_trace_slice_takes_a_class_first_seen_in_a_later_group(tiny, monkeypatch):
    c, _, _ = tiny
    real = metrics.product_class
    # the first group of every attention run shares one class, so every other class first
    # appears in a later group
    monkeypatch.setattr(metrics, "product_class", lambda c_, spec: "x" if (
        ".S[" in spec.name or ".dA[" in spec.name) else real(c_, spec))
    f = metrics.c5_layout_file(c, "x")
    firsts = {}
    for e in f["entries"]:
        firsts.setdefault(e["class"], (e["run"], e["group"]))
    later = [(cls, r, g) for cls, (r, g) in firsts.items()
             if g != min(x["group"] for x in f["entries"] if x["run"] == r)]
    assert later
    # a run that is neither a multiple of 20 nor the first of class "x" is in the slice only
    # through a class first seen in its later group
    only = [r for _, r, _ in later if r % 20 and r != firsts["x"][0]]
    assert only
    assert set(only) <= set(f["trace_slice"])
    assert f["trace_slice"] == sorted(set(f["trace_slice"]))
