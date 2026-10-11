"""EV1: run directories read as tables, with P1 and check 3 derived as `verification` derives
them. The writer side comes from `verification`, which tests may import."""

import json
import math
import warnings
from collections import Counter

import numpy as np
import pytest

from evaluation import records as er
from setup.config import load_config
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.runs import metrics as vm
from verification.runs import plain_baseline as pb
from verification.runs.mlp_smoke import run_smoke

SEED = 1
K = 7


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(vm._json_safe(r)) + "\n" for r in rows))


def _file_counts(path):
    return Counter(json.loads(line)["record"] for line in path.read_text().splitlines())


def _strip(row):
    return {k: v for k, v in row.items() if k not in ("run_dir", "plain_run_dir")}


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    """An output directory as the runs leave it: a plain baseline and an MLP smoke run written
    by `MetricsWriter`, a nested run directory, the shared `data/` and `bands.json`, earlier
    results under `evaluation/`, and an empty directory."""
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D, w0 = synthetic_dataset(c.widths, 40, seed=0), init_weights(c.widths, seed=0)
    root = tmp_path_factory.mktemp("out")
    with vm.MetricsWriter(root / "plain_baseline", pb.RUN_NAME, model="mlp") as mw:
        pb.plain_baseline(c, D, w0, T=3, build_model=c.build_model, metrics=mw,
                          out=lambda s: None, seed=SEED)
    with vm.MetricsWriter(root / "mlp_smoke", "mlp_smoke", model="mlp") as mw:
        run_smoke(c, D, w0, T=3, k=K, only=["honest", "flip"], metrics=mw, out=lambda s: None,
                  seed=SEED)
    _write_jsonl(root / "calibrate" / "k9" / "records.jsonl",
                 [{"record": "environment", "run": "calibrate"},
                  {"record": "calibration", "run": "calibrate", "c_hat": math.nan,
                   "bounds": [math.inf, -math.inf, 1.5], "by": {"x": "Infinity-ish"}},
                  {"record": "run_end", "run": "calibrate"}])
    _write_jsonl(root / "data" / "records.jsonl", [{"record": "environment", "run": "data"}])
    _write_jsonl(root / "evaluation" / "x" / "records.jsonl",
                 [{"record": "environment", "run": "x"}])
    (root / "bands.json").write_text("{}")
    (root / "store_crosscheck").mkdir()
    return root


def test_every_run_directory_read_with_the_files_counts(out):
    recs = er.read(out)
    assert recs.run_dirs == ["calibrate/k9", "mlp_smoke", "plain_baseline"]
    counts = recs.counts()
    for d in recs.run_dirs:
        assert counts[d] == _file_counts(out / d / "records.jsonl"), d
    # each table holds the writer's rows, in file order, with run_dir added
    for d in ("mlp_smoke", "plain_baseline"):
        for name in counts[d]:
            assert [_strip(r) for r in recs.table(name).where(run_dir=d)] == \
                   vm.read_records(out / d, name), (d, name)
    assert recs.table("calibration").column("run") == ["calibrate"]  # an unknown type kept
    assert recs.table("nothing") == []


def test_reserved_names_and_the_root_are_not_runs(out, tmp_path):
    assert [p.relative_to(out).as_posix() for p in er.find_run_dirs(out)] == \
           ["calibrate/k9", "mlp_smoke", "plain_baseline"]
    _write_jsonl(tmp_path / "records.jsonl", [{"record": "environment", "run": "root"}])
    _write_jsonl(tmp_path / "a" / "data" / "records.jsonl", [{"record": "environment"}])
    # only top-level data/ and evaluation/ are reserved
    assert [p.relative_to(tmp_path).as_posix() for p in er.find_run_dirs(tmp_path)] == ["a/data"]


def test_non_finite_strings_read_as_floats(out):
    (row,) = er.read(out).table("calibration")
    assert math.isnan(row["c_hat"])
    assert row["bounds"][:2] == [math.inf, -math.inf] and row["bounds"][2] == 1.5
    assert row["by"] == {"x": "Infinity-ish"}


def test_residual_archives_indexed_and_loaded(out):
    recs = er.read(out)
    files = sorted((out / "mlp_smoke" / "residuals").rglob("*.npz"))
    res = [a for a in recs.residuals if a.run_dir == "mlp_smoke"]
    assert [a.path for a in res] == files and files
    assert {a.scenario for a in res} == {"honest", "flip"}
    for a in res:
        assert a.run == "mlp_smoke" and a.path.parent.name == a.scenario
        assert a.path.name == f"step_{a.step:05d}.npz"
        arrays = a.load()
        assert arrays["step"].item() == a.step and arrays["p_normalized"].shape[1] == K
    assert not [a for a in recs.residuals if a.run_dir == "plain_baseline"]
    assert recs.archives == []


def test_archive_scenario_comes_from_the_archive(tmp_path):
    """The directory holds the scenario's safe name; the archive holds the scenario."""
    d = tmp_path / "run" / "residuals" / "a_b"
    d.mkdir(parents=True)
    np.savez(d / "step_00002.npz", run=np.array("run"), scenario=np.array("a/b"),
             step=np.array(2), p_m=np.arange(3))
    np.savez(tmp_path / "run" / "sweep_trials.npz", m=np.arange(4), x=np.ones(4))
    recs = er.read(tmp_path)
    assert recs.run_dirs == ["run"]
    (a,) = recs.residuals
    assert (a.run, a.scenario, a.step) == ("run", "a/b", 2)
    (t,) = recs.archives
    assert (t.run, t.scenario, t.step) == (None, None, None)
    assert list(t.load()) == ["m", "x"]


def test_nested_run_directories_keep_their_archives_apart(tmp_path):
    """A run directory inside another (a sweep's --subdir), each with its own residuals/."""
    for d, step in (("sweep", 1), ("sweep/row2", 2)):
        _write_jsonl(tmp_path / d / "records.jsonl", [{"record": "environment", "run": d}])
        res = tmp_path / d / "residuals" / "honest"
        res.mkdir(parents=True)
        np.savez(res / f"step_{step:05d}.npz", run=np.array(d), scenario=np.array("honest"),
                 step=np.array(step))
    np.savez(tmp_path / "sweep" / "row2" / "sweep_trials.npz", m=np.arange(2))
    recs = er.read(tmp_path)
    assert recs.run_dirs == ["sweep", "sweep/row2"]
    assert {d: dict(c) for d, c in recs.counts().items()} == \
           {"sweep": {"environment": 1}, "sweep/row2": {"environment": 1}}
    assert [(a.run_dir, a.run, a.step, a.path.relative_to(tmp_path).as_posix())
            for a in recs.residuals] == [
        ("sweep", "sweep", 1, "sweep/residuals/honest/step_00001.npz"),
        ("sweep/row2", "sweep/row2", 2, "sweep/row2/residuals/honest/step_00002.npz")]
    assert [(a.run_dir, a.path.name) for a in recs.archives] == [("sweep/row2",
                                                                  "sweep_trials.npz")]


def test_cut_short_archive_skipped_unreadable_one_raises(tmp_path):
    res = tmp_path / "run" / "residuals" / "honest"
    res.mkdir(parents=True)
    np.savez(res / "step_00001.npz", run=np.array("run"), scenario=np.array("honest"),
             step=np.array(1), p_m=np.arange(1000.0))
    whole = (res / "step_00001.npz").read_bytes()
    (res / "step_00002.npz").write_bytes(whole[: len(whole) // 2])  # a run still writing
    (tmp_path / "run" / "sweep_trials.npz").write_bytes(b"")  # created, nothing written yet
    with pytest.warns(UserWarning, match="not a complete .npz archive") as w:
        recs = er.read(tmp_path)
    assert sorted(x.filename for x in w) == [__file__] * 2  # warned at the caller of read
    assert {str(x.message).split(":")[0] for x in w} == \
           {str(res / "step_00002.npz"), str(tmp_path / "run" / "sweep_trials.npz")}
    assert [a.step for a in recs.residuals] == [1] and recs.archives == []
    (tmp_path / "run" / "sweep_trials.npz").write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="not a readable .npz archive"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er.read(tmp_path)


def test_incomplete_last_line_skipped_bad_line_raises(tmp_path):
    f = tmp_path / "run" / "records.jsonl"
    _write_jsonl(f, [{"record": "environment", "run": "run"}, {"record": "time", "run": "run"}])
    with open(f, "a") as fh:
        fh.write('{"record": "ti')  # a run still writing
    with pytest.warns(UserWarning, match="line 3 is incomplete"):
        assert dict(er.read(tmp_path).counts()["run"]) == {"environment": 1, "time": 1}
    f.write_text('{"record": "environment"}\n{"record": \n{"record": "time"}\n')
    with pytest.raises(ValueError, match="line 2"):
        er.read(tmp_path)
    f.write_text('{"run": "run"}\n')
    with pytest.raises(ValueError, match="no 'record' field"):
        er.read(tmp_path)


def test_output_dir_matches_setup_config():
    for env in ({}, {"VERIF_OUTPUT_DIR": "/elsewhere/out"}):
        assert er.output_dir(env) == load_config(env).output_dir
    with pytest.raises(ValueError):
        er.output_dir({"VERIF_OUTPUT_DIR": " "})


def test_table_selectors():
    t = er.Table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "x"}])
    assert t.where(b="x").column("a") == [1, 3]
    assert t.where(a=lambda v: v > 1, b="x").column("a") == [3]
    assert t.distinct("b") == ["x", "y"]
    assert {k: g.column("a") for k, g in t.group_by("b").items()} == {("x",): [1, 3],
                                                                      ("y",): [2]}


# ---- P1 and check 3 -----------------------------------------------------------------------


def _phase(record, run, scenario, step, component, **values):
    return {"record": record, "run": run, "scenario": scenario, "step": step, "side": "prover",
            "component": component, "level": "phase", "calls": 1, **values}


def _synthetic(run, scenario, prefix, steps, *, label=True, none_peak_at=None):
    rows = []
    for t in steps:
        for i, x in enumerate(("load", "forward", "backward", "update")):
            rows.append(_phase("time", run, scenario, t, f"{prefix}.{x}",
                               time_s=0.1 * t + 0.01 * i + (0.5 if prefix == "train" else 0)))
            rows.append(_phase("count", run, scenario, t, f"{prefix}.{x}",
                               flops=100 * t + i, hash_in_bytes=7 * i, hash_out_bytes=i,
                               hash_calls=i))
            peak = None if t == none_peak_at and x == "backward" else 1000 * t + 10 * i
            rows.append(_phase("memory", run, scenario, t, f"{prefix}.{x}", peak_bytes=peak,
                               start_bytes=500 * t + i, end_bytes=None, mem_source="probe"))
        if label:
            rows.append(_phase("time", run, scenario, t, "P1.label", time_s=0.25, calls=3))
    rows.append(_phase("time", run, scenario, 0, f"{prefix}.load", time_s=9.0))  # step 0: ignored
    return rows


def test_derive_capture_matches_verification():
    plain = _synthetic("plain_baseline", "plain", "P0", [1, 2, 3])
    verified = (_synthetic("v", "honest", "train", [1, 2], none_peak_at=2)
                + _synthetic("v", "flip", "train", [1]))
    mine = er.derive_capture(verified, plain)
    assert mine == vm.derive_capture(verified, plain)
    assert {(r["record"], r["scenario"], r["step"]) for r in mine} == \
           {(k, s, t) for k in ("time", "memory", "count")
            for s, t in (("honest", 1), ("honest", 2), ("flip", 1))}
    (m,) = [r for r in mine if r["record"] == "memory" and r["step"] == 2]
    assert m["peak_bytes"] is None
    no_label = _synthetic("v", "honest", "train", [1], label=False)
    for f in (er.derive_capture, vm.derive_capture):
        with pytest.raises(ValueError, match="no 'P1.label' phase row at step 1"):
            f(no_label, plain)
        with pytest.raises(ValueError, match="2 \\(run, scenario\\) pairs"):
            f(verified, plain + _synthetic("other", "plain", "P0", [1]))


def test_capture_as_capture_rows(out):
    recs = er.read(out)
    assert recs.verified_dirs() == ["mlp_smoke"] and recs.plain_dirs() == ["plain_baseline"]
    assert recs.plain_for("mlp_smoke") == "plain_baseline"
    for scenario in ("honest", None):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mine = recs.capture("mlp_smoke", scenario=scenario)
            theirs = pb.capture_rows(out / "mlp_smoke", out / "plain_baseline",
                                     scenario=scenario)
        assert mine and [_strip(r) for r in mine] == theirs
        assert {(r["run_dir"], r["plain_run_dir"]) for r in mine} == \
               {("mlp_smoke", "plain_baseline")}
    # a plain step with no verified counterpart warns as capture_rows does
    with pytest.warns(UserWarning, match="no verified counterpart") as w:
        assert recs.capture("mlp_smoke", scenario="none") == []
    with pytest.warns(UserWarning, match="no verified counterpart") as w2:
        pb.capture_rows(out / "mlp_smoke", out / "plain_baseline", scenario="none")
    assert [str(x.message) for x in w] == [str(x.message) for x in w2]
    # every verified run directory at once pairs automatically, with no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [_strip(r) for r in recs.capture()] == \
               pb.capture_rows(out / "mlp_smoke", out / "plain_baseline", scenario=None)


def test_capture_pairs_on_the_environment(tmp_path):
    def env(model, **kw):
        return {"record": "environment", "run": "r", "model": model, "corpus": "c",
                "device": "cpu", "hardware": {"machine": "arm64", "cpu": "A", "gpu": None},
                "threads": 8, **kw}

    def write(d, environment, prefix):
        scenario = "honest" if prefix == "train" else "plain"
        _write_jsonl(tmp_path / d / "records.jsonl",
                     [environment, *_synthetic(d, scenario, prefix, [1, 2])])

    write("v", env("m"), "train")
    write("p1", env("m"), "P0")
    write("p2", env("other"), "P0")
    write("p4", env("m", hardware={"machine": "arm64", "cpu": "B", "gpu": None}), "P0")
    write("p5", env("m", threads=4), "P0")
    recs = er.read(tmp_path)
    assert recs.plain_for("v") == "p1"  # another model, machine or thread count doesn't match
    assert set(recs.capture().column("plain_run_dir")) == {"p1"}
    write("p3", env("m", seed=2), "P0")
    recs = er.read(tmp_path)
    with pytest.raises(ValueError, match="2 plain baselines match"):
        recs.capture()
    assert set(recs.capture(plain="p3").column("plain_run_dir")) == {"p3"}
    for wrong in ("v", "nowhere"):  # a verified run, or no run at all, isn't a plain baseline
        with pytest.raises(ValueError, match="not a plain run directory"):
            recs.capture(plain=wrong)
    write("v", env("third"), "train")
    write("v2", env("fourth"), "train")
    recs = er.read(tmp_path)
    with pytest.warns(UserWarning, match="no plain baseline matches") as w:
        assert recs.capture() == []  # no plain baseline for these models: no P1 rows
    assert [str(x.message).split(":")[0] for x in w] == ["v", "v2"]  # one warning per run
    with pytest.raises(ValueError, match="no plain baseline"):
        recs.capture("v")


def test_check_3_is_reported_as_5_glue(out):
    recs = er.read(out)
    for kind in ("time", "memory", "count"):
        raw = recs.table(kind)
        glue = raw.where(component="5.glue")
        costs = recs.costs(kind)
        three = costs.where(component="3")
        assert glue and len(three) == len(glue), kind
        for g, r in zip(glue, three):
            assert r == {**g, "component": "3", "derived": True, "alias_of": "5.glue"}
            assert r["level"] == "sub"
        assert not raw.where(component="3")  # the tables stay as written
        assert costs[:len(raw)] == raw
        p1 = costs.where(component="P1")
        assert p1 == recs.capture().where(record=kind)
    with pytest.raises(ValueError, match="record must be one of"):
        recs.costs("step")
