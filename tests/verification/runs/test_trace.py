"""FS23: the traced step (``torch.profiler`` on one step of the timed run, ``trace/``)."""

import dataclasses
import gzip
import json

import pytest
import torch

from evaluation import records as ev_records
from setup.config import load_config
from tests.verification.decoder_helpers import make_records, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.runs import metrics
from verification.runs.metrics import MetricsWriter, read_markers, read_records
from verification.runs.plain_baseline import _NOT_TRAINING
from verification.runs.scenarios import HONEST, honest_final, run_scenario
from verification.transcript.store import IN_MEMORY, StoreHandoff

SEED = 1
K = 7
T = 3


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    return c, D, w0, honest_final(c, D, w0, T, seed=SEED), dataset_tree(c, D).root


class _Roots(StoreHandoff):
    """In-memory hand-off that keeps each step's Merkle root."""

    def __init__(self) -> None:
        self.roots: list[bytes] = []

    def hold(self, c, t, leaves, tree, dataset_paths=None):
        self.roots.append(tree.root)
        return IN_MEMORY.hold(c, t, leaves, tree, dataset_paths)


def run(tiny, out, **trace):
    """One honest run of ``T`` steps, writing into ``out``; the result and the step roots."""
    c, D, w0, final, h_D = tiny
    roots = _Roots()
    with MetricsWriter(out, "tiny", device="cpu", model="tiny", corpus="synthetic", seed=0,
                       config={}, band_file_hash=None, h_D=h_D) as mw:
        mw.write_c5_layout(c)
        r = run_scenario(c, D, w0, HONEST, T=T, k=K, seed=SEED, final=final, h_D=h_D,
                         handoff=roots, recorder=mw.recorder(HONEST.name, **trace))
    return r, roots.roots


@pytest.fixture(scope="module")
def traced(tiny, tmp_path_factory):
    out = tmp_path_factory.mktemp("trace") / "tiny"
    r, roots = run(tiny, out, trace_step=2)
    return out, r, roots


def load_trace(out):
    with gzip.open(out / "trace" / "step_00002.json.gz", "rt") as f:
        return json.load(f)


def meta(out):
    return json.loads((out / "trace" / "meta.json").read_text())


def annotations(out):
    return {e["name"] for e in load_trace(out)["traceEvents"] if e.get("cat") == "user_annotation"}


def step_line(out, t):
    return next(x for x in read_markers(out) if x["kind"] == "step" and x["step"] == t)


def test_trace_holds_every_section_and_slice_stamp(traced):
    out, _, _ = traced
    # the temporary .json is gone
    assert sorted(p.name for p in (out / "trace").iterdir()) == ["meta.json", "step_00002.json.gz"]
    names = annotations(out)
    line = step_line(out, 2)
    sections = {s["name"] for s in line["sections"]}
    spanned = {s["name"] for s in line["sections"] if s["host_us"]}
    rows = {r["component"] for r in read_records(out, "time")
            if r["step"] == 2 and r["level"] in ("phase", "sub")}
    assert rows and rows <= sections
    assert spanned <= names, spanned - names
    sliced = set(meta(out)["slice"])
    st = line["stamps"]
    want = {f"{line['stamp_names'][n]}|run={r}|group={g}"
            for n, r, g in zip(st["name"], st["run"], st["group"], strict=True) if r in sliced}
    assert want and want <= names, want - names
    assert not any(n.startswith("c5.") and int(n.split("run=")[1].split("|")[0]) not in sliced
                   for n in names)


def test_meta_has_every_field(traced):
    out, _, _ = traced
    m = meta(out)
    assert set(m) == {"version", "step", "slice", "c5_slice", "torch", "profiler", "start_ns",
                      "stop_ns", "error"}
    assert (m["version"], m["step"], m["c5_slice"], m["error"]) == (1, 2, "slice", None)
    assert m["torch"] == torch.__version__
    assert isinstance(m["slice"], list) and all(isinstance(i, int) for i in m["slice"])
    layout = json.loads((out / "c5_layout.json").read_text())
    assert m["slice"] == layout["trace_slice"]
    assert m["profiler"] == {"activities": ["CPU"], "record_shapes": True, "with_flops": True}
    assert isinstance(m["start_ns"], int) and m["start_ns"] < m["stop_ns"]


def test_tracing_changes_no_verdict_root_or_weight(tiny, traced, tmp_path):
    _, on, roots_on = traced
    off, roots_off = run(tiny, tmp_path)
    assert (tmp_path / "trace").exists() is False
    assert on.loop.verdict == off.loop.verdict and on.loop.verdict.accepted
    assert roots_on == roots_off and len(roots_on) == T
    assert [(s.t, s.rejection, s.loss) for s in on.loop.steps] == \
           [(s.t, s.rejection, s.loss) for s in off.loop.steps]
    for k, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k].view(torch.int32), w.view(torch.int32))


def test_config_hash_ignores_the_trace_settings(monkeypatch):
    def config():
        cfg = load_config()
        return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                if f.name not in ("output_dir", "metrics", "release_cache", "trace_step",
                                  "trace_c5")}

    for name in ("VERIF_TRACE_STEP", "VERIF_TRACE_C5"):
        monkeypatch.delenv(name, raising=False)
    plain = config()
    monkeypatch.setenv("VERIF_TRACE_STEP", "2")
    monkeypatch.setenv("VERIF_TRACE_C5", "all")
    assert metrics.config_hash(config()) == metrics.config_hash(plain)
    assert {"trace_step", "trace_c5"} <= set(_NOT_TRAINING)


def test_a_profiler_that_raises_at_start_leaves_the_run_passing(tiny, tmp_path, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("no profiler today")

    monkeypatch.setattr(torch.profiler, "profile", broken)
    r, _ = run(tiny, tmp_path, trace_step=2)
    assert r.loop.verdict.accepted
    m = meta(tmp_path)
    assert "no profiler today" in m["error"]
    assert not list((tmp_path / "trace").glob("step_*"))
    assert {x["traced"] for x in read_records(tmp_path, "step")} == {False}


def test_a_step_never_reached_writes_not_reached(tiny, tmp_path):
    r, _ = run(tiny, tmp_path, trace_step=9)
    assert r.loop.verdict.accepted
    assert meta(tmp_path)["error"] == "not reached"
    assert not list((tmp_path / "trace").glob("step_*"))


@pytest.mark.parametrize("how", ["all", "no-toggle"])
def test_c5_slice_all(tiny, tmp_path, monkeypatch, how):
    if how == "no-toggle":
        monkeypatch.delattr(torch.profiler.profiler._KinetoProfile, "toggle_collection_dynamic")
    r, _ = run(tiny, tmp_path, trace_step=2, trace_c5="all" if how == "all" else "slice")
    assert r.loop.verdict.accepted
    m = meta(tmp_path)
    assert (m["c5_slice"], m["slice"], m["error"]) == ("all", None, None)
    names = annotations(tmp_path)
    assert {s["name"] for s in step_line(tmp_path, 2)["sections"] if s["host_us"]} <= names


def test_traced_rows_are_marked_and_kept_out_of_costs(traced):
    out, _, _ = traced
    for kind in ("time", "step", "capture"):
        flags = {r["step"]: r["traced"] for r in read_records(out, kind) if r["step"] > 0}
        assert flags == {1: False, 2: True, 3: False}, kind
    assert {r["traced"] for r in read_records(out, "time") if r["step"] == 0} == {False}
    lines = {x["step"]: x["traced"] for x in read_markers(out) if x["kind"] == "step"}
    assert lines == {1: False, 2: True, 3: False}
    recs = ev_records.read(out.parent)
    assert recs.table("time") and not any(r["traced"] for r in recs.table("time"))
    t = recs.table("time_traced")
    assert t and {r["step"] for r in t} == {2}
    assert {r["step"] for r in recs.table("time")} == {0, 1, 3}


@pytest.mark.slow
def test_smollm2_test_scale_trace(monkeypatch, tmp_path):
    """``run_verified`` on SmolLM2 at test scale, CPU, three steps, step 2 traced. Needs the data
    and the band file under ``$VERIF_OUTPUT_DIR``; skips without them."""
    from verification.runs import run_verified

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    src = load_config().output_dir
    if not ((src / "data" / "D.bin").exists() and (src / "bands.json").exists()):
        pytest.skip("no data and bands.json under VERIF_OUTPUT_DIR")
    (tmp_path / "data").symlink_to((src / "data").resolve())
    (tmp_path / "bands.json").symlink_to((src / "bands.json").resolve())
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIF_TRACE_STEP", "2")
    run_verified.main(["--steps", "3", "--metrics"])
    (out,) = [d for d in tmp_path.iterdir() if d.name.startswith("honest")]
    assert gzip.open(out / "trace" / "step_00002.json.gz").read(2)
    assert meta(out)["error"] is None
