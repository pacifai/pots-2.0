"""FS25: the analysis on hand-built run directories (PROFILING_PLAN §5–§7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.results.profile import analyze, io
from tests.evaluation.results.profile.helpers import (
    STEP_NS,
    build_run,
    jl,
    probe,
    section,
    step_line,
    time_row,
)


def run(tmp_path: Path, **kw):
    paths = build_run(tmp_path / "run", **kw)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    return out, out["runs"]["honest"]["sections"]


def test_bottleneck_needs_share_and_ratio(tmp_path):
    # A: 20% share, floor 0.02 s (10x). B: 20%, floor 0.1 s (2x). C: 5%, 10x.
    out, s = run(tmp_path, hash_bytes={"A": 20_000_000, "B": 100_000_000, "C": 5_000_000})
    assert s["A"]["share"] == pytest.approx(0.2)
    assert s["A"]["ratio"] == pytest.approx(10.0) and s["A"]["bound"] == "hash"
    assert "bottleneck" in s["A"]["marks"]
    assert s["B"]["ratio"] == pytest.approx(2.0) and "bottleneck" not in s["B"]["marks"]
    assert s["C"]["ratio"] == pytest.approx(10.0) and "bottleneck" not in s["C"]["marks"]
    assert s["D"]["floor_s"] is None and s["D"]["marks"] == []


def test_step_shares_sum_to_one(tmp_path):
    out, _ = run(tmp_path)
    assert out["runs"]["honest"]["share_total"] == pytest.approx(1.0)
    assert out["thresholds"]["min_samples"] == 3


def test_overlap_weighting_is_exact():
    rows = [{"t": 1_000_000_000, "v": 10.0}, {"t": 2_000_000_000, "v": 20.0},
            {"t": 3_000_000_000, "v": 40.0}]
    samples = io.Samples({"mono_ns": 0, "fields": ["t", "v"]}, rows)
    means, n = analyze.weighted_means(samples, [(500_000_000, 2_500_000_000)])
    # weights 0.5, 1, 0.5 s: (5 + 20 + 20) / 2
    assert means["v"] == pytest.approx(22.5) and n == 3
    # a null is skipped, and its weight leaves the denominator
    rows[1]["v"] = None
    means, _ = analyze.weighted_means(samples, [(500_000_000, 2_500_000_000)])
    assert means["v"] == pytest.approx((5 + 20) / 1.0)


def test_marker_spans_map_onto_the_sample_clock(tmp_path):
    # Step 1 starts at 10 s; A spans 10.0-10.2 s, inside the sample that ends at 11 s only.
    rows = [{"t": int(i * 1e9), "c_tree": float(i), "g_util": 0.0} for i in range(1, 41)]
    _, s = run(tmp_path, sample_rows=rows, steps=1)
    assert s["A"]["util"]["c_tree"] == pytest.approx(11.0)
    assert s["A"]["util"]["n_samples"] == 1


def test_few_samples(tmp_path):
    _, s = run(tmp_path, steps=1)
    assert s["A"]["util"]["few_samples"] is True
    _, s = run(tmp_path / "more", steps=3)
    assert s["A"]["util"]["n_samples"] == 3 and s["A"]["util"]["few_samples"] is False


def test_utilization_mark(tmp_path):
    # hash floor/time = 0.1; measured c_tree/threads = 1/8 = 0.125: within 2x. With c_tree 4:
    # 0.5 against 0.1, which differs by 5x.
    kw = {"hash_bytes": {"A": 20_000_000}}
    _, s = run(tmp_path / "a", **kw)
    assert "utilization" not in s["A"]["marks"]
    rows = [{"t": int(i * 1e9), "c_tree": 4.0, "g_util": 0.0} for i in range(1, 41)]
    _, s = run(tmp_path / "b", sample_rows=rows, **kw)
    assert "utilization" in s["A"]["marks"]
    # too few samples: no mark
    _, s = run(tmp_path / "c", sample_rows=rows, steps=1, **kw)
    assert "utilization" not in s["A"]["marks"]


def memory_row(comp: str, peak: int):
    return {"record": "memory", "run": "honest", "scenario": "honest", "step": 1,
            "side": "verifier", "component": comp, "level": "component", "calls": 1,
            "peak_bytes": peak, "peak_reserved_bytes": peak}


def test_memory_mark_and_missing_capacity(tmp_path):
    extra = [memory_row("A", 95), memory_row("B", 94)]
    over = {"machine": {"gpu_memory_bytes": 100}}
    out, s = run(tmp_path / "a", records_extra=extra, probe_over=over)
    assert "memory" in s["A"]["marks"] and "memory" not in s["B"]["marks"]
    assert out["runs"]["honest"]["memory"]["sections"]["A"]["fraction"] == 0.95
    out, s = run(tmp_path / "b", records_extra=extra)
    assert all("memory" not in v["marks"] for v in s.values())
    assert any("gpu_memory_bytes" in w for w in out["warnings"])


def test_machine_block_falls_back_to_the_environment_record(tmp_path):
    env = {"record": "environment", "run": "honest", "machine": {"gpu_memory_bytes": 100}}
    _, s = run(tmp_path, records_extra=[env, memory_row("A", 96)])
    assert "memory" in s["A"]["marks"]


def test_clock_mark(tmp_path):
    # host 0.1 s against GPU 0.2 s: more than max(5%, 2 ms). B differs by 1 ms: agrees.
    secs = {"A": (0.0, 0.1, 0.2), "B": (0.1, 0.1, 0.101), "C": (0.2, 0.03, 0.06),
            "D": (0.3, 0.7)}
    _, s = run(tmp_path, secs=secs, device="cuda")
    assert s["A"]["clocks"]["agree"] is False and "clock" in s["A"]["marks"]
    assert s["B"]["clocks"]["agree"] is True and "clock" not in s["B"]["marks"]
    assert s["C"]["clocks"]["agree"] is False and "clock" not in s["C"]["marks"]  # under 50 ms
    assert s["D"]["clocks"]["agree"] is None


def test_clock_mark_from_the_trace(tmp_path):
    paths = build_run(tmp_path / "run", steps=2,
                      step_extra=[step_line(3, [section("A", 0.0, 0.1)], traced=True)])
    t = tmp_path / "trace_summary.json"
    t.write_text(json.dumps({"version": 1, "warnings": ["w"], "c5_slice": {},
                             "sections": {"A": {"wall_us": 120_000.0},
                                          "B": {"wall_us": 200_000.0}}}))
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t)
    s = out["runs"]["honest"]["sections"]
    assert s["A"]["clocks"]["trace_s"] == pytest.approx(0.12)
    assert s["A"]["clocks"]["agree"] is False and "clock" in s["A"]["marks"]  # 20% off
    assert s["B"]["clocks"]["agree"] is None  # no traced step of B
    assert "w" in out["warnings"]


def test_trace_work_sets_the_pcie_floor(tmp_path):
    paths = build_run(tmp_path / "run")
    t = tmp_path / "trace_summary.json"
    t.write_text(json.dumps({"version": 1, "c5_slice": {}, "sections": {
        "A": {"copy_bytes": {"h2d": 1e9, "d2h": 5e8}, "launches": 1000}}}))
    s = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t
                        )["runs"]["honest"]["sections"]["A"]
    assert s["floors_s"]["pcie"] == pytest.approx(1e9 / 1e10 + 5e8 / 1e10)
    assert s["floors_s"]["launch"] == pytest.approx(1000 * 5e-6)
    assert s["bound"] == "pcie"


def test_gemm_rate_by_side_and_dtype(tmp_path):
    from tests.evaluation.results.profile.helpers import count_row
    extra = [count_row(1, "P3", flops=1e11), count_row(1, "4", flops=1e11),
             count_row(1, "5", flops=1e11)]
    secs = {"P3": (0.0, 0.2), "4": (0.2, 0.2), "5": (0.4, 0.2), "D": (0.6, 0.4)}
    paths = build_run(tmp_path / "run", secs=secs, records_extra=extra)
    # time_row says "verifier" for every section, so the prover side comes from the name
    kw = {"compute_dtype": "bfloat16"}
    f = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], **kw
                        )["runs"]["honest"]["sections"]
    assert f["P3"]["floors_s"]["gemm"] == pytest.approx(1e11 / 1e11)  # verifier row: fp32
    assert f["4"]["floors_s"]["gemm"] == pytest.approx(1.0)
    assert f["5"]["floors_s"]["gemm"] == pytest.approx(1e11 / 1e10)  # skinny


def test_prover_side_uses_the_compute_dtype_rate(tmp_path):
    from tests.evaluation.results.profile.helpers import count_row
    records = [{"record": "environment", "run": "honest", "compute_dtype": "bfloat16"},
               time_row(1, "train.forward", 0.5, "prover", "phase"),
               time_row(1, "step", 0.5), count_row(1, "train.forward", flops=1e12)]
    paths = build_run(tmp_path / "run", steps=1, secs={"train.forward": (0.0, 0.5)})
    jl(paths["run_dir"] / "records.jsonl", records)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert out["compute_dtype"] == "bfloat16" and out["warnings"][0].startswith("no machine")
    assert out["runs"]["honest"]["sections"]["train.forward"]["floors_s"]["gemm"] == 1.0


def test_missing_compute_dtype_uses_the_flag_and_warns(tmp_path):
    paths = build_run(tmp_path / "run", records_extra=[])
    jl(paths["run_dir"] / "records.jsonl", [time_row(1, "step", 1.0)])
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"],
                          compute_dtype="bfloat16")
    assert out["compute_dtype"] == "bfloat16"
    assert "--compute-dtype" in out["warnings"][0]


def test_hook_overhead_bound(tmp_path):
    secs = {"A": (0.0, 0.5, 0.5), "B": (0.5, 0.5, 0.5)}
    out, _ = run(tmp_path, secs=secs, device="cuda", steps=2)
    # per step: 4 syncs and 2 gpu spans over a 1 s step
    assert out["runs"]["honest"]["hook_overhead_bound"] == pytest.approx(4 * 1.5e-5 + 2 * 4e-6)
    out, _ = run(tmp_path / "cpu", secs=secs, steps=2)
    assert out["runs"]["honest"]["hook_overhead_bound"] == 0.0


def test_plain_comparison(tmp_path):
    paths = build_run(tmp_path / "run", steps=1, secs={"train.forward": (0.0, 0.4)})
    jl(paths["run_dir"] / "records.jsonl",
       [time_row(1, "train.forward", 0.4, "prover", "phase"), time_row(1, "step", 0.4)])
    plain = tmp_path / "plain"
    plain.mkdir()
    jl(plain / "records.jsonl", [time_row(1, "P0.forward", 0.1, "prover", "phase"),
                                 time_row(1, "step", 0.1)])
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], plain=plain)
    c = out["comparisons"]["sections"]["train.forward"]
    assert c["plain_name"] == "P0.forward" and c["ratio"] == pytest.approx(4.0)


def c5_inputs(tmp_path: Path):
    names = ["c5.run", "c5.glue", "c5.serve", "c5.copy", "c5.measure", "c5.judge"]
    # two runs in window 0 and 1; durations in µs from t0
    rows = [(0, 0, -1, -1, 0, 100), (1, 0, 0, -1, 0, 20), (2, 0, 0, -1, 20, 60),
            (3, 0, 0, -1, 30, 50), (4, 0, 0, -1, 60, 100), (0, 1, -1, -1, 100, 200),
            (2, 1, 1, -1, 100, 180), (3, 1, 1, -1, 110, 130), (5, -1, -1, 0, 200, 210)]
    st = {"name": [r[0] for r in rows], "run": [r[1] for r in rows],
          "group": [r[2] for r in rows], "window": [r[3] if r[0] != 5 else 0 for r in rows],
          "h0": [float(r[4]) for r in rows], "h1": [float(r[5]) for r in rows],
          "g0": [None] * len(rows), "g1": [None] * len(rows)}
    layout = {"version": 1, "run": "honest", "n_runs": 2, "n_groups": 2, "n_windows": 1,
              "entries": [{"run": 0, "group": 0, "window": 0, "class": "Q.fwd", "kind": "dense",
                           "product_bytes": 100, "served_bytes": 200},
                          {"run": 1, "group": 1, "window": 0, "class": "K.fwd", "kind": "dense",
                           "product_bytes": 50, "served_bytes": 100}],
          "trace_slice": [0]}
    return names, st, layout


def test_c5_view(tmp_path):
    names, st, layout = c5_inputs(tmp_path)
    # the run stamp of run 1 is stamp 5 in the table above; fix up ids so run 1 is distinct
    st["run"][5] = 1
    st["run"][6] = 1
    st["run"][7] = 1
    secs = {"5": (0.0, 0.0002)}
    paths = build_run(tmp_path / "run", steps=3, secs=secs)
    lines = [json.loads(ln) for ln in (paths["run_dir"] / "markers.jsonl").read_text().splitlines()]
    for ln in lines[1:]:
        ln["stamps"], ln["stamp_names"] = st, names
    jl(paths["run_dir"] / "markers.jsonl", lines)
    (paths["run_dir"] / "c5_layout.json").write_text(json.dumps(layout))
    c5 = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"]
                         )["runs"]["honest"]["c5"]
    q = c5["classes"]["Q.fwd"]["phases_s"]
    assert q["glue"] == pytest.approx(20e-6)
    assert q["host_stack"] == pytest.approx((40 - 20) * 1e-6)  # serve 40 us less copy 20 us
    assert q["copy"] == pytest.approx(20e-6) and q["measure"] == pytest.approx(40e-6)
    assert c5["classes"]["Q.fwd"]["copy_bytes_per_s"]["stored"] == pytest.approx(100 / 20e-6)
    assert c5["phases_s"]["judge"] == pytest.approx(10e-6)
    assert sorted(c5["runs_over_time"]) == ["1", "2", "3"]
    assert c5["runs_over_time"]["1"]["dur_s"] == pytest.approx([100e-6, 100e-6])
    # identical steps have no variance: no correlation
    assert c5["variance"]["correlation"]["across_steps"]["glue"]["cpu_mhz"] is None
    assert len(c5["variance"]["windows"]) == 3


def test_correlation():
    assert analyze._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert analyze._pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert analyze._pearson([1, 2], [1, 2]) is None
    assert analyze._pearson([1, 1, 1], [1, 2, 3]) is None


def test_cli_writes_profile_json(tmp_path):
    paths = build_run(tmp_path / "run", hash_bytes={"A": 20_000_000})
    out = tmp_path / "profile.json"
    rc = analyze.main(["--run-dir", str(paths["run_dir"]), "--samples", str(paths["samples"]),
                       "--probe", str(paths["probe"]), "--out", str(out)])
    got = json.loads(out.read_text())
    assert rc == 0 and got["version"] == 1
    assert "bottleneck" in got["runs"]["honest"]["sections"]["A"]["marks"]


def test_a_wrong_version_is_rejected(tmp_path):
    paths = build_run(tmp_path / "run")
    d = paths["run_dir"]
    bad = {"markers.jsonl": io.read_markers, "samples.jsonl": io.read_samples}
    for name, reader in bad.items():
        lines = (d / name).read_text().splitlines()
        h = json.loads(lines[0])
        h["version"] = 2
        (d / name).write_text("\n".join([json.dumps(h), *lines[1:]]) + "\n")
        with pytest.raises(ValueError, match="version 2"):
            reader(d / name)
    for name, reader in (("probe.json", io.read_probe), ("c5_layout.json", io.read_c5_layout),
                         ("trace_summary.json", io.read_trace_summary)):
        (d / name).write_text(json.dumps({"version": 2, "sections": {}}))
        with pytest.raises(ValueError, match="version 2"):
            reader(d / name)
    (d / "probe.json").write_text(json.dumps(probe() | {"version": None}))
    with pytest.raises(ValueError, match="version"):
        io.read_probe(d / "probe.json")
    with pytest.raises(ValueError, match="version 2"):
        analyze.analyze(d, d / "samples.jsonl", d / "probe.json")


def test_readers_accept_version_one(tmp_path):
    paths = build_run(tmp_path / "run")
    m = io.read_markers(paths["run_dir"] / "markers.jsonl")
    assert m.header["run"] == "honest" and len(m.steps) == 3
    assert len(io.read_samples(paths["samples"]).rows) == 40
    assert io.read_probe(paths["probe"])["version"] == 1
    assert STEP_NS == 10_000_000_000


def test_samples_reader_skips_error_lines(tmp_path):
    from tests.evaluation.results.profile.helpers import samples_file
    f = tmp_path / "s.jsonl"
    samples_file(f, [{"t": 1, "c_tree": -0.5, "g_util": 0.5},
                     {"kind": "error", "group": "gpm", "error": "x"}, {"t": 2, "c_tree": 1.0}])
    rows = io.read_samples(f).rows
    assert [r["t"] for r in rows] == [1, 2] and rows[0]["c_tree"] == -0.5


def test_gemm_lane_uses_the_fraction_as_written():
    means = {"g_tensor": None, "g_util": 0.5}
    assert analyze._lane("gemm", means, {"rates": {}}) == 0.5


def test_leaves_go_to_sections_hashing_half_the_transcript(tmp_path):
    from tests.evaluation.results.profile.helpers import count_row
    storage = {"record": "storage", "run": "honest", "scenario": "honest", "step": 1,
               "transcript_bytes": 1000, "leaves": 100}
    secs = {"P3.commit": (0.0, 0.2), "2": (0.2, 0.2), "A": (0.4, 0.2), "D": (0.6, 0.4)}
    extra = [storage, count_row(1, "P3.commit", hash_in_bytes=1000),
             count_row(1, "2", hash_in_bytes=500), count_row(1, "A", hash_in_bytes=499)]
    rates = probe()["rates"] | {"hash": {"per_byte_s": 1e-9, "per_leaf_s": 1e-6, "threads": 8}}
    _, s = run(tmp_path, secs=secs, records_extra=extra, probe_over={"rates": rates})
    assert s["P3.commit"]["work"]["leaves"] == 100 and s["2"]["work"]["leaves"] == 100
    assert s["A"]["work"]["leaves"] == 0 and s["D"]["work"]["leaves"] == 0
    assert s["2"]["floors_s"]["hash"] == pytest.approx(500e-9 + 100e-6)
    assert s["A"]["floors_s"]["hash"] == pytest.approx(499e-9)


def test_check_5_trace_work_scales_by_run_time(tmp_path):
    secs = {"5": (0.0, 0.5), "4": (0.5, 0.5)}
    paths = build_run(tmp_path / "run", secs=secs)
    cls = {"Q": {"phases_us": {"run": 100.0}, "scaled": {"run": 400.0}},
           "K": {"phases_us": {"run": 100.0}, "scaled": {"run": 200.0}},
           "V": {"phases_us": {"run": 50.0}, "scaled": None}}  # left out, with a warning
    t = tmp_path / "trace_summary.json"
    t.write_text(json.dumps({"version": 1, "c5_slice": cls, "sections": {
        "5": {"launches": 10}, "4": {"launches": 10}}}))
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t)
    s = out["runs"]["honest"]["sections"]
    assert s["5"]["work"]["launches"] == pytest.approx(10 * 600 / 200)
    assert s["4"]["work"]["launches"] == 10
    assert sum("'V'" in w for w in out["warnings"]) == 1
    cls = {"Q": {"phases_us": {"run": 0.0}, "scaled": {"run": 0.0}}}
    t.write_text(json.dumps({"version": 1, "c5_slice": cls, "sections": {"5": {"launches": 10}}}))
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t)
    assert out["runs"]["honest"]["sections"]["5"]["work"]["launches"] == 10
    assert any("unscaled" in w for w in out["warnings"])
