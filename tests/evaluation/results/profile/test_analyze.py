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
    assert s["floors_s"]["pcie"] == pytest.approx(1e9 / 1e10)  # max of the two directions
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
                       "--probe", str(paths["probe"]), "--session", "check-2026-10-19",
                       "--out", str(out)])
    got = json.loads(out.read_text())
    assert rc == 0 and got["version"] == 1
    assert got["meta"]["session"] == "check-2026-10-19" and got["meta"]["scenario"] == "honest"
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
    with pytest.warns(UserWarning, match="'error'"):
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


def test_a_nested_section_is_not_double_counted(tmp_path):
    secs = {"A": (0.0, 0.4), "A.s": (0.1, 0.2), "D": (0.4, 0.6)}
    out, s = run(tmp_path, secs=secs, depths={"A.s": 1})
    assert out["runs"]["honest"]["share_total"] == pytest.approx(1.0)
    assert s["A.s"]["depth"] == 1 and s["A.s"]["share"] == pytest.approx(0.2)


def trace_file(tmp_path: Path, sections: dict, **extra) -> Path:
    t = tmp_path / "trace_summary.json"
    t.write_text(json.dumps({"version": 1, "c5_slice": {}, "sections": sections, **extra}))
    return t


def with_trace(paths, t):
    return analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t)


def test_the_hbm_floor(tmp_path):
    paths = build_run(tmp_path / "run")
    out = with_trace(paths, trace_file(tmp_path, {"A": {"gemm_bytes": 5e9,
                                                       "copy_bytes": {"d2d": 5e9}}}))
    s = out["runs"]["honest"]["sections"]["A"]
    assert s["floors_s"]["hbm"] == pytest.approx(1e10 / 1e11) and s["bound"] == "hbm"


def test_a_null_rate_drops_only_its_own_resource(tmp_path):
    rates = probe()["rates"] | {"h2d_pageable_bytes": None}
    paths = build_run(tmp_path / "run", probe_over={"rates": rates})
    t = trace_file(tmp_path, {"A": {"copy_bytes": {"d2h": 1e9}, "launches": 10},
                              "B": {"copy_bytes": {"h2d": 1e9}, "launches": 10},
                              "C": {"launches": 10}})
    out = with_trace(paths, t)
    s = out["runs"]["honest"]["sections"]
    assert s["A"]["floors_s"]["pcie"] == pytest.approx(1e9 / 1e10)  # h2d count is 0: no term
    assert "pcie" not in s["B"]["floors_s"] and "launch" in s["B"]["floors_s"]
    assert "pcie" not in s["C"]["floors_s"]
    assert sum("h2d_pageable_bytes" in w for w in out["warnings"]) == 1


def test_a_null_cost_with_zero_count_adds_nothing(tmp_path):
    paths = build_run(tmp_path / "run", probe_over={"costs": {"launch_s": 5e-6, "sync_s": None,
                                                              "event_pair_s": None}})
    out = with_trace(paths, trace_file(tmp_path, {"A": {"launches": 10}}))
    s = out["runs"]["honest"]["sections"]["A"]
    assert s["floors_s"]["launch"] == pytest.approx(10 * 5e-6)  # no syncs counted off CUDA
    assert not any("sync_s" in w for w in out["warnings"])


def test_thresholds_are_inclusive(tmp_path):
    # share exactly 0.10 and ratio exactly 3.0 (binary-exact numbers, one step)
    rates = probe()["rates"] | {"hash": {"per_byte_s": 0.5 ** 30, "per_leaf_s": 0.0,
                                         "threads": 8}}
    kw = {"steps": 1, "probe_over": {"rates": rates}}
    secs = {"A": (0.0, 0.375), "D": (0.4, 3.375)}
    _, s = run(tmp_path / "a", secs=secs, hash_bytes={"A": 2 ** 27}, **kw)
    assert s["A"]["share"] == 0.1 and s["A"]["ratio"] == 3.0
    assert "bottleneck" in s["A"]["marks"]
    _, s = run(tmp_path / "b", secs=secs, hash_bytes={"A": 2 ** 27 + 2 ** 20}, **kw)
    assert "bottleneck" not in s["A"]["marks"]  # ratio just under 3
    _, s = run(tmp_path / "c", secs={"A": (0.0, 0.375), "D": (0.4, 3.5)},
               hash_bytes={"A": 2 ** 27}, **kw)
    assert "bottleneck" not in s["A"]["marks"]  # share just under 0.10


def edit_markers(paths, edit):
    f = paths["run_dir"] / "markers.jsonl"
    lines = [json.loads(ln) for ln in f.read_text().splitlines()]
    for ln in lines[1:]:
        edit(ln)
    jl(f, lines)


def test_hook_bound_counts_stamps_and_rest_calls(tmp_path):
    paths = build_run(tmp_path / "run", steps=2, secs={"A": (0.0, 0.5, 0.5)}, device="cuda")

    def edit(ln):
        ln["sections"][0]["rest"] = {"calls": 3, "first_us": 0.0, "last_us": 1.0}
        ln["stamps"] = {"name": [0] * 5}

    edit_markers(paths, edit)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    # per 1 s step: 2 syncs; 1 gpu span + 3 rest calls + 5 stamps = 9 events
    assert out["runs"]["honest"]["hook_overhead_bound"] == pytest.approx(2 * 1.5e-5 + 9 * 4e-6)


def test_a_rest_folded_section_keeps_a_span_and_is_flagged(tmp_path):
    rows = [{"t": int(i * 1e9), "c_tree": float(i), "g_util": 0.0} for i in range(1, 41)]
    paths = build_run(tmp_path / "run", steps=1, sample_rows=rows, secs={"A": (0.0, 0.2),
                                                                        "B": (0.5, 0.1)})

    def edit(ln):
        ln["sections"][0]["rest"] = {"calls": 5, "first_us": 3e6, "last_us": 3.5e6}

    edit_markers(paths, edit)
    s = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"]
                        )["runs"]["honest"]["sections"]
    # 0.2 s of sample 11 and 0.5 s of sample 14 (the step starts at 10 s)
    assert s["A"]["util"]["c_tree"] == pytest.approx((0.2 * 11 + 0.5 * 14) / 0.7)
    assert s["A"]["util"]["truncated"] is True and s["B"]["util"]["truncated"] is False


def test_the_first_sample_starts_at_the_header_clock(tmp_path):
    from tests.evaluation.results.profile.helpers import samples_file
    paths = build_run(tmp_path / "run", steps=1)
    samples_file(paths["samples"], [{"t": 10_500_000_000, "c_tree": 7.0, "g_util": 0.5},
                                    {"t": 11_000_000_000, "c_tree": 9.0, "g_util": 0.5}],
                 start_ns=10_000_000_000)
    s = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"]
                        )["runs"]["honest"]["sections"]
    assert s["A"]["util"]["c_tree"] == 7.0 and s["A"]["util"]["n_samples"] == 1


def test_the_gemm_lane_follows_the_rate_used():
    means = {"g_tensor": 0.7, "g_sm_active": 0.2, "g_util": 0.5}
    lane = analyze._lane
    assert lane("gemm", means, {}, "gemm_bf16_flops") == 0.7
    assert lane("gemm", means, {}, "gemm_fp32_flops") == 0.2
    assert lane("gemm", means, {}, "skinny_bmm_fp32_flops") == 0.2
    assert lane("gemm", {"g_sm_active": None, "g_util": 0.5}, {}, "gemm_fp32_flops") == 0.5
    assert lane("gemm", {"g_sm_active": 0.0, "g_util": 0.5}, {}, "gemm_fp32_flops") == 0.0


def test_throttle_is_the_fraction_of_samples_with_a_bit_set():
    rows = [{"t": 1_000_000_000, "g_throttle": 0}, {"t": 2_000_000_000, "g_throttle": 4},
            {"t": 3_000_000_000, "g_throttle": 1}]
    samples = io.Samples({"mono_ns": 0}, rows)
    means, _ = analyze.weighted_means(samples, [(0, 3_000_000_000)])
    assert means["g_throttle"] == pytest.approx(2 / 3)


def test_readers_skip_unknown_kinds_once_per_kind(tmp_path):
    paths = build_run(tmp_path / "run")
    f = paths["run_dir"] / "markers.jsonl"
    f.write_text(f.read_text() + '{"kind": "note"}\n{"kind": "note"}\n{"kind": "other"}\n')
    with pytest.warns(UserWarning) as w:
        m = io.read_markers(f)
    assert len(m.steps) == 3 and len(w) == 2
    g = tmp_path / "s.jsonl"
    from tests.evaluation.results.profile.helpers import samples_file
    samples_file(g, [{"t": 1}, {"kind": "error"}, {"kind": "error"}, {"t": 2}])
    with pytest.warns(UserWarning) as w:
        assert len(io.read_samples(g).rows) == 2
    assert len(w) == 1


def test_sections_come_from_the_time_rows_and_the_markers(tmp_path):
    from tests.evaluation.results.profile.helpers import count_row
    extra = [time_row(k, "X", 0.3) for k in (1, 2, 3)] + [count_row(1, "X", hash_in_bytes=3e7)]
    out, s = run(tmp_path, records_extra=extra)
    x = s["X"]
    assert x["time"]["mean"] == pytest.approx(0.3) and x["floors_s"]["hash"] == pytest.approx(0.03)
    assert x["ratio"] == pytest.approx(10.0)
    assert x["clocks"]["host_s"] is None and x["util"]["n_samples"] == 0
    assert x["depth"] == 0 and s["A"]["util"]["n_samples"] > 0
    assert "step" not in s


def test_a_phase_rollup_is_left_out_of_the_share_total(tmp_path):
    extra = [time_row(k, "P2", 0.2, "prover", "component") for k in (1, 2, 3)] \
        + [time_row(k, "P2.w", 0.2, "prover", "phase") for k in (1, 2, 3)]
    out, s = run(tmp_path, records_extra=extra)
    assert s["P2"]["rollup"] is True and "rollup" not in s["P2.w"]
    assert out["runs"]["honest"]["share_total"] == pytest.approx(1.0 + 0.2)  # P2.w only


def test_a_component_whose_phases_carry_another_prefix_is_a_rollup(tmp_path):
    """``train_captured`` sums ``train.*``: no shared prefix, still left out of the total."""
    extra = [time_row(k, "train_captured", 0.2, "prover", "component") for k in (1, 2, 3)] \
        + [time_row(k, "train.forward", 0.2, "prover", "phase") for k in (1, 2, 3)]
    out, s = run(tmp_path, records_extra=extra)
    assert s["train_captured"]["rollup"] is True and "rollup" not in s["train.forward"]
    assert out["runs"]["honest"]["share_total"] == pytest.approx(1.0 + 0.2)


def test_a_rollup_absent_from_the_trace_does_not_warn(tmp_path):
    """A rollup has no span of its own, so the trace never holds it."""
    extra = [time_row(k, "P2", 0.2, "prover", "component") for k in (1, 2, 3)] \
        + [time_row(k, "P2.w", 0.2, "prover", "phase") for k in (1, 2, 3)]
    paths = build_run(tmp_path / "run", records_extra=extra)
    t = tmp_path / "trace_summary.json"
    t.write_text(json.dumps({"version": 1, "c5_slice": {}, "sections": {}}))
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], t)
    absent = [w for w in out["warnings"] if w.startswith("the trace summary has no section")]
    assert absent and "P2.w" in absent[0]
    assert "P2," not in absent[0] and not absent[0].endswith(" P2")


def test_markers_without_steps_or_absent_warn(tmp_path):
    paths = build_run(tmp_path / "run")
    f = paths["run_dir"] / "markers.jsonl"
    f.write_text(f.read_text().splitlines()[0] + "\n")
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert any("no step lines" in w for w in out["warnings"])
    assert out["runs"]["honest"]["sections"]["A"]["time"]["mean"] == pytest.approx(0.2)
    f.unlink()
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert any("markers" in w and "does not exist" in w for w in out["warnings"])
    assert out["runs"]["honest"]["sections"]["A"]["util"]["n_samples"] == 0


def test_the_cli_runs_with_only_the_run_directory(tmp_path):
    paths = build_run(tmp_path / "run", hash_bytes={"A": 20_000_000})
    out = tmp_path / "profile.json"
    assert analyze.main(["--run-dir", str(paths["run_dir"]), "--session", "s",
                         "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert sum("no samples file" in w for w in got["warnings"]) == 1
    assert sum("no probe file" in w for w in got["warnings"]) == 1
    a = got["runs"]["honest"]["sections"]["A"]
    assert a["floors_s"] == {} and a["util"]["n_samples"] == 0


def test_an_empty_samples_file_warns_and_leaves_utilization_null(tmp_path):
    paths = build_run(tmp_path / "run")
    paths["samples"].write_bytes(b"")
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert any("is empty" in w for w in out["warnings"])
    u = out["runs"]["honest"]["sections"]["A"]["util"]
    assert u["n_samples"] == 0 and "c_tree" not in u


def test_the_trace_missing_sections_warn_once(tmp_path):
    paths = build_run(tmp_path / "run")
    out = with_trace(paths, trace_file(tmp_path, {"A": {"launches": 1}}))
    msgs = [w for w in out["warnings"] if "no section" in w]
    assert len(msgs) == 1 and "B, C, D" in msgs[0]


def test_members_of_a_group_with_different_classes_warn_once(tmp_path):
    names, st, layout = c5_inputs(tmp_path)
    layout["entries"][1]["group"] = 0
    paths = build_run(tmp_path / "run", steps=2, secs={"5": (0.0, 0.0002)})
    edit_markers(paths, lambda ln: ln.update(stamps=st, stamp_names=names))
    (paths["run_dir"] / "c5_layout.json").write_text(json.dumps(layout))
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert sum("different classes" in w for w in out["warnings"]) == 1


def test_markers_for_clocks_and_spans_skip_step_zero_and_traced_steps(tmp_path):
    rows = [{"t": int(i * 1e9), "c_tree": float(i), "g_util": 0.0} for i in range(1, 41)]
    paths = build_run(tmp_path / "run", steps=1, sample_rows=rows,
                      step_extra=[step_line(0, [section("A", 0.0, 0.2)]),
                                  step_line(2, [section("A", 0.0, 0.2)], traced=True)])
    s = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"]
                        )["runs"]["honest"]["sections"]
    assert s["A"]["util"]["c_tree"] == pytest.approx(11.0)  # only step 1's span at 10 s


def test_meta_names_the_run_from_the_environment_record(tmp_path):
    paths = build_run(tmp_path / "run")
    env = {"record": "environment", "run": "honest", "compute_dtype": "bfloat16",
           "model": "Qwen/Qwen2.5-0.5B-Instruct", "attack": "refusal", "git_commit": "abc",
           "device": "cuda"}
    lines = (paths["run_dir"] / "records.jsonl").read_text().splitlines()
    lines[0] = json.dumps(env)
    (paths["run_dir"] / "records.jsonl").write_text("\n".join(lines) + "\n")
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], session="s1")
    assert out["meta"] == {"model": "Qwen/Qwen2.5-0.5B-Instruct", "attack": "refusal",
                           "scenario": "honest", "device": out["meta"]["device"],
                           "compute_dtype": "bfloat16", "git_commit": "abc", "session": "s1"}
    assert not any("--session" in w for w in out["warnings"])
    assert any("--session" in w for w in analyze.analyze(paths["run_dir"])["warnings"])


def timeline_of(tmp_path: Path, **kw) -> list[dict]:
    paths = build_run(tmp_path, **kw)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    return out["runs"]["honest"]["timeline"]["steps"]


def test_timeline_spans_end_and_traced_step_absent(tmp_path):
    steps = timeline_of(tmp_path, steps=2, step_extra=[
        step_line(3, [section("A", 0.0, 0.1)], traced=True)])
    assert [s["step"] for s in steps] == [1, 2]
    s = steps[0]
    assert [x["name"] for x in s["sections"]] == ["A", "B", "C", "D"]
    a = s["sections"][0]
    assert a["depth"] == 0 and a["spans_s"] == [pytest.approx([0.0, 0.2])]
    assert s["sections"][3]["spans_s"] == [pytest.approx([0.45, 1.0])]
    assert s["end_s"] == pytest.approx(1.0)


def test_timeline_folded_rest_gives_one_span(tmp_path):
    paths = build_run(tmp_path, steps=1)

    def edit(step):
        step["sections"][0]["rest"] = {"calls": 3, "first_us": 300_000.0, "last_us": 350_000.0,
                                       "host_s": 0.01}

    edit_markers(paths, edit)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    a = out["runs"]["honest"]["timeline"]["steps"][0]["sections"][0]
    assert a["spans_s"] == [pytest.approx([0.0, 0.2]), pytest.approx([0.3, 0.35])]


def test_timeline_strips_take_the_overlapping_rows_and_four_fields(tmp_path):
    rows = [{"t": int(i * 1e9), "c_tree": 1.0, "g_util": 0.5, "c_busy": 2.0,
             "g_pcie_tx": None} for i in range(1, 41)]
    st = timeline_of(tmp_path, steps=1, sample_rows=rows)[0]["strips"]
    # step 1 starts at 10 s and lasts 1 s: only the sample ending at 11 s overlaps
    assert st["t_s"] == [pytest.approx(1.0)]
    assert set(st) == {"t_s", "g_util", "c_busy"}  # c_tree is no strip; the all-null tx is dropped
    assert st["g_util"] == [0.5] and st["c_busy"] == [2.0]


def test_timeline_strips_null_without_samples(tmp_path):
    assert timeline_of(tmp_path, steps=1, sample_rows=[])[0]["strips"] is None


def test_timeline_downsamples_to_400_points(tmp_path):
    rows = [{"t": 10_000_000_000 + i * 1_000_000, "g_util": float(i % 2)}
            for i in range(1, 1001)]
    st = timeline_of(tmp_path, steps=1, sample_rows=rows)[0]["strips"]
    assert len(st["t_s"]) == len(st["g_util"]) <= 400
    assert st["t_s"] == sorted(st["t_s"]) and st["t_s"][-1] == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in st["g_util"])


def test_probe_and_machine_in_profile_json(tmp_path):
    paths = build_run(tmp_path, probe_over={"machine": {"gpu_memory_bytes": 100,
                                                        "cpu": {"n": 4}}})
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert out["machine"] == {"gpu_memory_bytes": 100, "cpu": {"n": 4}}
    assert "machine" not in out["probe"] and out["probe"]["rates"]["gemm_fp32_flops"] == 1e11
    assert out["probe"]["device"] == "cpu"


def test_probe_and_machine_null_without_a_probe(tmp_path):
    paths = build_run(tmp_path)
    out = analyze.analyze(paths["run_dir"], paths["samples"])
    assert out["probe"] is None and out["machine"] is None
    paths = build_run(tmp_path / "e")
    f = paths["run_dir"] / "records.jsonl"
    lines = f.read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "machine": {"gpus": 1}})
    f.write_text("\n".join(lines) + "\n")
    assert analyze.analyze(paths["run_dir"], paths["samples"])["machine"] == {"gpus": 1}
