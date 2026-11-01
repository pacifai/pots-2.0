"""FS26: the HTML report on synthetic ``profile.json`` dicts (archive/PROFILING_PLAN FS26, §12)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evaluation.results.profile import report


def sec(depth: int, mean: float, share: float | None, *, marks=(), flops=None, hbm=0.0, h2d=0.0,
        hash_b=0.0, leaves=0, util=None, rollup=False) -> dict[str, Any]:
    s: dict[str, Any] = {
        "depth": depth, "time": {"mean": mean, "median": mean, "min": mean, "max": mean, "n": 3},
        "share": share, "clocks": {"host_s": mean, "gpu_s": None, "trace_s": None, "agree": None},
        "util": {"n_samples": 5, "few_samples": False, "truncated": False, **(util or {})},
        "work": {"flops": flops or {}, "bytes": {"hbm": hbm, "h2d": h2d, "d2h": 0.0,
                                                 "hash": hash_b},
                 "leaves": leaves, "launches": 0.0, "syncs": 0},
        "floors_s": {}, "floor_s": None, "bound": None, "ratio": None, "marks": list(marks)}
    if rollup:
        s["rollup"] = True
    return s


def timeline(strips: dict[str, list] | None, names: list[str]) -> dict[str, Any]:
    secs = [{"name": n, "depth": 0, "spans_s": [[0.1 * i, 0.1 * i + 0.08]]}
            for i, n in enumerate(names)]
    steps = [{"step": k, "end_s": 0.1 * len(names), "sections": secs, "strips": strips}
             for k in (1, 2)]
    return {"steps": steps}


def phases(scale: float = 1.0) -> dict[str, float]:
    return {"run": 0.2 * scale, "glue": 0.1 * scale, "host_stack": 0.01 * scale,
            "copy": 0.02 * scale, "measure": 0.05 * scale, "judge": 0.01 * scale}


def c5_class(scale: float = 1.0) -> dict[str, Any]:
    return {"n_runs": 2, "n_groups": 2, "product_bytes": 100, "served_bytes": 100,
            "phases_s": phases(scale), "copy_bytes_per_s": {"stored": None, "served": None}}


def view(*, gpu: bool, c5: bool = True, tl: bool = True, fwd_mean: float = 0.123456,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    secs = {
        "train.forward": sec(0, fwd_mean, 0.3, flops={"gemm_fp32_flops": 1e9},
                             util={"g_util": 0.7, "c_tree": 2.0} if gpu else {"c_tree": 2.0}),
        "train.backward": sec(0, 0.2, 0.2, hbm=1e9 if gpu else 0.0, flops={"gemm_fp32_flops": 2e9}),
        "P3.commit": sec(0, 0.1, 0.1, hash_b=1e6, leaves=10, marks=["bottleneck"]),
        "P2": sec(0, 0.05, 0.05, rollup=True),
        "4": sec(0, 0.05, 0.05),
        "5": sec(0, 0.2, 0.2, flops={"skinny_bmm_fp32_flops": 1e8}, h2d=1e6 if gpu else 0.0),
        "5.glue": sec(1, 0.1, 0.1),
        "6b": sec(0, 0.05, 0.05),
        "6b.glue": sec(1, 0.02, 0.02),
        **(extra or {})}
    strips = {"t_s": [0.1, 0.2, 0.3], "c_busy": [1.0, 2.0, None]}
    if gpu:
        strips |= {"g_util": [0.1, 0.5, 0.9], "g_pcie_tx": [1e6, 2e6, 3e6]}
    c5v = None
    if c5:
        c5v = {"n_runs": 2, "n_groups": 2, "n_windows": 1, "phases_s": phases(),
               "classes": {"G_down": c5_class()},
               "runs_over_time": {"1": {"run": [0, 1], "class": ["G_down", "G_down"],
                                        "start_s": [0.0, 0.1], "dur_s": [0.05, 0.05]}},
               "variance": {"steps": [{"step": 1, "phases_s": phases(),
                                       "sampler": {"c_other": None}}],
                            "windows": [],
                            "correlation": {"across_steps": {"run": {"c_other": None}},
                                            "across_windows": {}}}}
    return {"device": "cuda" if gpu else "cpu", "n_steps": 2, "step_s": 1.0, "share_total": 0.95,
            "sections": secs, "c5": c5v, "hook_overhead_bound": 0.0 if not gpu else 0.01,
            "timeline": timeline(strips, ["train.forward", "train.backward", "P3.commit", "4",
                                          "5"]) if tl else {"steps": []},
            "memory": {"gpu_memory_bytes": None, "sections": {}}}


def profile(model="M", session="s1", attack="refusal", scenario="honest", *, gpu=False,
            runs: dict[str, dict[str, Any]] | None = None, **kw) -> dict[str, Any]:
    return {"version": 1,
            "meta": {"model": model, "attack": attack, "scenario": scenario,
                     "device": "cuda" if gpu else "cpu", "compute_dtype": "float32",
                     "git_commit": "abc", "session": session},
            "run": scenario, "compute_dtype": "float32",
            "thresholds": {"share": 0.1, "ratio": 3.0}, "warnings": ["w1 <b>"],
            "probe": {"rates": {"hash": {"per_byte_s": 1e-9}}, "costs": {"launch_s": 5e-6},
                      "datasheet": None},
            "machine": {"cpu_model": "X", "cgroup_cpu_quota": 4.0},
            "runs": runs or {scenario: view(gpu=gpu, **kw)}, "comparisons": None}


def page(*profiles) -> str:
    return report.build(list(profiles), [f"p{i}.json" for i in range(len(profiles))])


def panel(html: str, anchor: str) -> str:
    m = re.search(rf'<section id="{anchor}">.*?</section>', html, re.S)
    assert m, anchor
    return m.group(0)


def test_no_urls_and_small():
    out = page(profile(gpu=True), profile("N", gpu=False))
    assert "http://" not in out and "https://" not in out and "xmlns" not in out
    assert len(out.encode()) < 5_000_000
    assert "<script>" in out and "<svg" in out


def test_a_synthetic_profile_renders_every_panel():
    out = page(profile(gpu=True))
    for a in ("inputs", "r0-overview", "r0-diagram", "r0-timeline", "r0-check5"):
        assert f'<section id="{a}">' in out and f'href="#{a}"' in out
    for gone in ("r0-sections", "r0-hash", "appendix"):  # in profile.json only (user, 2026-10-20)
        assert gone not in out
    assert "w1 &lt;b&gt;" in out and "w1 <b>" not in out  # escaped


def test_one_run_has_no_picker_and_no_comparison():
    out = page(profile())
    assert 'id="picker"' not in out and "<script>" not in out and "data-run" not in out
    assert "compare-" not in out


def test_two_runs_of_one_model_in_one_session():
    out = page(profile(scenario="honest"), profile(scenario="attack"))
    assert 'id="picker"' in out and out.count("<option") == 2
    assert 'data-run="1" style="display:none"' in out
    assert "compare-" not in out


def test_two_models_in_one_session():
    out = page(profile("org/A"), profile("B"))
    assert 'id="compare-models"' in out and "compare-sessions" not in out
    t = panel(out, "compare-models")
    assert "A · refusal" in t and "B · refusal" in t and "org/" not in t
    assert "seconds per step" in t


def test_two_sessions_sharing_a_model():
    out = page(profile("A", "s1"), profile("B", "s1"), profile("A", "s2"))
    t = panel(out, "compare-sessions")
    assert "A · refusal · honest · s1" in t and "A · refusal · honest · s2" in t
    assert "B ·" not in t
    assert "compare-models" in out  # s1 holds two models
    assert "compare-sessions" not in page(profile("A", "s1"), profile("B", "s2"))


def test_a_missing_section_has_no_segment_in_a_comparison():
    extra = {"7": sec(0, 0.3, 0.3)}
    out = page(profile("A", "s1", runs={"honest": view(gpu=False, extra=extra)}),
               profile("A", "s2"))
    t = panel(out, "compare-sessions")
    assert t.count("Check 7: weights chain (7) 0.30 s</title>") == 1


def test_a_cpu_profile_has_no_gpu_gauges_or_strips():
    out = page(profile(gpu=False))
    flow = panel(out, "r0-diagram")
    assert ">CPU</text>" in flow and ">GPU</text>" not in flow and "GPU busy" not in flow
    tl = panel(out, "r0-timeline")
    assert "CPU cores" in tl and "GPU busy" not in tl and "PCIe" not in tl
    gpu = page(profile(gpu=True))
    flow = panel(gpu, "r0-diagram")
    assert ">GPU</text>" in flow and ">CPU</text>" in flow and ">70%</text>" in flow
    assert "GPU busy" in panel(gpu, "r0-timeline") and "PCIe" in panel(gpu, "r0-timeline")


def test_an_empty_panel_is_left_out_of_page_and_bar():
    out = page(profile(c5=False, tl=False))
    for a in ("r0-check5", "r0-timeline"):
        assert f'id="{a}"' not in out and f'href="#{a}"' not in out
    assert 'id="r0-overview"' in out


def test_the_picker_hides_a_runs_missing_panel_with_its_link():
    out = page(profile(scenario="a"), profile(scenario="b", c5=False))
    assert 'id="r0-check5"' in out and 'id="r1-check5"' not in out
    assert '<span data-run="0"><a href="#r0-check5">' in out


def test_hover_title_equals_the_table_cell():
    out = page(profile(fwd_mean=0.123456))
    cell = report.fmt(0.123456)
    assert cell == "0.12"
    assert f"<td>{cell}</td>" in panel(out, "r0-overview")
    assert f"mean {cell} s" in panel(out, "r0-overview")
    assert f"mean {cell} s" in panel(out, "r0-diagram")


def test_the_overview_names_sections_and_explains_its_marks():
    ov = panel(page(profile()), "r0-overview")
    assert "Forward pass" in ov and "train.forward" in ov and "Merkle commitment" in ov
    assert "Share of step time" in ov and "prover's own training" in ov
    for part in ("Prover", "Verifier"):
        assert f'<tr class="part"><td colspan="5">{part} ' in ov
    assert '<span class="badge">bottleneck</span>' in ov  # P3.commit's mark
    assert "at least 10% of the step and at least 3.0×" in ov  # its legend, from thresholds
    assert "utilization</span></dt>" not in ov  # only the marks the run carries
    assert "P2" not in ov.replace("P2.", "")  # a rollup


def test_the_flow_draws_one_box_per_section_and_outlines_marks():
    flow = panel(page(profile()), "r0-diagram")
    for code in ("train.forward", "train.backward", "P3.commit", "4", "5", "6b"):
        assert f"<g><title>{report.SECTIONS[code][2]} ({code})\n" in flow
    assert flow.count("<g><title>") == 6  # no rollup, no nested section
    assert 'stroke="var(--mark)"' in flow and ">bottleneck</text>" in flow
    assert "4.0 cores" in flow  # the CPU scale is the machine's quota


def with_plain(model: str, ratio: float) -> dict[str, Any]:
    p = profile(model)
    p["comparisons"] = {"sections": {"train.forward": {"plain_name": "P0.forward",
                                                       "prover_s": 0.2, "plain_s": 0.1,
                                                       "ratio": ratio}},
                        "step": {"prover_s": 1.0, "plain_s": 0.5}}
    return p


def test_plain_comparison_is_a_per_run_panel_before_the_comparisons():
    out = page(with_plain("A", 2.0), with_plain("B", 7.0))
    a, b = panel(out, "r0-plain"), panel(out, "r1-plain")
    assert "<td>2.0</td>" in a and "<td>7.0</td>" not in a and "<td>7.0</td>" in b
    assert "Forward pass" in a and "÷ plain" in a and "Verified run (s)" in a and "Whole step" in a
    assert '<span data-run="1"><a href="#r1-plain">' in out  # the picker switches it
    assert out.index('id="r1-plain"') < out.index('id="compare-models"')
    assert "-plain" not in page(profile())


def test_the_cli_writes_the_file(tmp_path: Path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(profile("A")))
    b.write_text(json.dumps(profile("B")))
    out = tmp_path / "profile.html"
    assert report.main([str(a), str(b), "--out", str(out)]) == 0
    text = out.read_text()
    assert "compare-models" in text and f"a.json, b.json in {tmp_path}" in text
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 2}))
    with pytest.raises(ValueError, match="bad.json"):
        report.main([str(bad), "--out", str(out)])


def test_fmt():
    assert report.fmt(None) == "—" and report.fmt(True) == "yes"
    assert report.fmt(3) == "3" and report.fmt(576) == "576"  # a count stays exact
    assert report.fmt(1234567) == "1.2e+06" and report.fmt(9999.0) == "1.0e+04"
    assert report.fmt(0.008397) == "8.4e-03" and report.fmt(0.000123456) == "1.2e-04"
    assert report.fmt(0.0303) == "0.030" and report.fmt(4.7746) == "4.8"
    assert report.fmt(61.685) == "62" and report.fmt(110.3) == "110"
    assert report.fmt(1234.0) == "1200" and report.fmt(9.97) == "10" and report.fmt(0.0) == "0"
    assert report.pct(0.629) == "63%" and report.pct(0.0001) == "<0.1%"
    assert report.pct(0.0) == "0%" and report.pct(None) == "—"


def test_ticks_are_round():
    assert report._ticks(4.77) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert report._ticks(110.0) == [0.0, 50.0, 100.0, 150.0]
    assert report._ticks(0.0) == [0.0, 1.0]


def big_view(steps: int = 6, n_sec: int = 25) -> dict[str, Any]:
    v = view(gpu=True)
    names = [f"S{i}" for i in range(n_sec)]
    t = [round(0.01 * k, 4) for k in range(400)]
    strips = {"t_s": t, "g_util": [0.5] * 400, "c_busy": [2.0] * 400,
              "g_pcie_tx": [1e6] * 400, "g_pcie_rx": [1e6] * 400}
    v["timeline"] = {"steps": [
        {"step": k, "end_s": 4.0, "strips": strips,
         "sections": [{"name": n, "depth": 0, "spans_s": [[0.16 * i, 0.16 * i + 0.1]]}
                      for i, n in enumerate(names)]} for k in range(steps)]}
    v["sections"].update({n: sec(0, 0.1, 0.01) for n in names})
    return v


def test_a_full_scale_report_stays_below_5_mb():
    out = page(*[profile(f"M{i}", runs={"honest": big_view()}) for i in range(8)])
    size = len(out.encode())
    assert size < 5_000_000, f"{size} bytes"


def test_an_unknown_section_keeps_its_code_under_other():
    out = page(profile(runs={"honest": big_view(1, 2)}))
    ov = panel(out, "r0-overview")
    assert '<tr class="part"><td colspan="5">Other ' in ov and ">S0<" in ov
    assert '<span class="gt">Other sections</span>' in panel(out, "r0-timeline")


def test_check_5_rows_carry_their_pass_and_model_part():
    v = view(gpu=False)
    v["c5"]["classes"] = {"Zed": c5_class(), "G_down": c5_class(2.0), "Y_q": c5_class(),
                          "dV": c5_class()}
    c5 = panel(page(profile(runs={"honest": v})), "r0-check5")
    rows = [c5.index(f">{c}</text>") for c in ("Y_q", "G_down", "dV", "Zed")]
    assert rows == sorted(rows)  # in the order a step runs them, unknown classes last
    for words in ("Forward · attention", "Backward · MLP", "Weight gradient, down projection",
                  "Backward · attention", "Other", "Rebuild inputs", "seconds per step"):
        assert words in c5
    assert "Gradients of values, queries and keys" in c5  # dQ and dK are served with dV
    assert "0.38 s" in c5  # G_down's five phases at twice the scale


def test_notes_and_legend_names():
    out = page(profile(gpu=False))
    tl = panel(out, "r0-timeline")
    legend = tl[:tl.index("<svg")]
    assert "Forward pass" in legend and "Prover · training" in legend
    assert "train.forward" not in legend and 'class="code"' not in legend  # names only
    assert "Forward pass (train.forward)" in tl  # the hover keeps the code
    c5 = panel(out, "r0-check5")
    assert "Rebuild inputs</span>" in c5[c5.index('<div class="legend">'):]
    nostrips = profile(runs={"honest": view(gpu=False)})
    for st in nostrips["runs"]["honest"]["timeline"]["steps"]:
        st["strips"] = None
    tl = panel(page(nostrips), "r0-timeline")
    assert "No sampler data for this run." in tl and "Lines under each step" not in tl


def test_check_5_sentence_names_the_overall_floor_its_resource_and_the_ratio():
    detail = {"hbm": {"serving_bytes": 1.0, "serving_s": 0.05, "measuring_reads_bytes": 2.0,
                      "measuring_reads_s": 0.07}}
    v = view(gpu=True)
    v["sections"]["5"] |= {"floor_detail": detail, "ratio": 14.0, "floor_s": 0.32,
                           "bound": "pcie"}
    c5 = panel(page(profile(runs={"honest": v})), "r0-check5")
    assert "Check 5's floor is 0.32 s, set by PCIe; it took 14× that. Its memory-" \
        "bandwidth part is 0.12 s: serving (copying the operands and products), 0.050 s, and " \
        "measuring reads (the measuring kernel reading each product's inputs and the product " \
        "once), 0.070 s.</p>" in c5
    assert "Partial floor" not in c5
    ov = panel(page(profile(runs={"honest": v})), "r0-overview")
    assert "Partial floors" not in ov


def test_check_5_partial_floor_says_not_counted_and_the_profile_says_it_once():
    detail = {"hbm": {"serving_bytes": None, "serving_s": None, "measuring_reads_bytes": 2.0,
                      "measuring_reads_s": 0.07}, "pcie": None, "nvlink": None, "launch": None}
    v = view(gpu=True)
    v["sections"]["5"] |= {"floor_detail": detail, "ratio": 31.0, "floor_s": 0.07, "bound": "hbm",
                           "floor_partial": {"missing": ["hbm serving", "pcie", "nvlink",
                                                         "launch"]}}
    v["sections"]["4"] |= {"floor_partial": {"missing": ["hbm", "pcie", "nvlink", "launch"]}}
    out = page(profile(runs={"honest": v}))
    c5 = panel(out, "r0-check5")
    assert "Partial floor</b>: this run has no trace of check 5" in c5
    assert ("the memory traffic of serving, PCIe, the GPU-to-GPU link and kernel launches "
            "are not counted (no trace)") in c5
    assert "hbm serving" not in c5 and "nvlink" not in c5
    assert "Check 5's floor is 0.070 s, set by memory bandwidth; it took 31× that. " in c5
    assert "sets no mark" in c5
    assert "serving (copying the operands and products), not counted (no trace), and" in c5
    assert "at least 0.070 s" in c5 and " 0 s" not in c5
    # the profile says once that its floors leave out the trace terms, however many sections
    assert out.count("their floors leave out the terms that come from a trace") == 1
    assert "not comparable with a traced run" in panel(out, "r0-overview")


def two_gpu_flow(roles) -> str:
    v = view(gpu=True)
    v["sections"]["train.forward"]["util_by_device"] = {"cuda:0": {}, "cuda:1": {}}
    v["gpu_roles"] = roles
    return panel(page(profile(gpu=True, runs={"honest": v})), "r0-diagram")


def test_the_gpu_note_is_drawn_only_with_several_gpus_on_a_section():
    v = view(gpu=True)
    v["gpu_roles"] = [{"device": "cuda:0", "roles": ["prover"]},
                      {"device": "cuda:1", "roles": ["verifier"]}]
    flow = panel(page(profile(gpu=True, runs={"honest": v})), "r0-diagram")
    assert "With 2 GPUs" not in flow


def test_the_gpu_note_for_one_verifier_gpu():
    flow = two_gpu_flow([{"device": "cuda:0", "roles": ["prover"]},
                         {"device": "cuda:1", "roles": ["verifier"]}])
    assert "With 2 GPUs (cuda:0 trains; cuda:1 verifies), the commit's hashing" in flow
    assert "run on the verifier GPU, so" in flow and "GPUs 1 to 3" not in flow
    assert "The profile's per-GPU utilization holds each GPU." in flow


def test_the_gpu_note_for_three_verifier_gpus():
    flow = two_gpu_flow([{"device": "cuda:0", "roles": ["prover"]},
                         {"device": "cuda:1", "roles": ["verifier"]},
                         {"device": "cuda:2", "roles": ["verifier"]},
                         {"device": "cuda:3", "roles": ["verifier"]}])
    assert "With 4 GPUs (cuda:0 trains; cuda:1, cuda:2, cuda:3 verify)" in flow
    assert "run on the verifier GPUs, so" in flow and "four GPUs" not in flow


def test_the_gpu_note_layouts_and_plurals():
    def r(*pairs):
        return [{"device": d, "roles": list(ro)} for d, ro in pairs]
    one = two_gpu_flow(r(("cuda:0", ["prover", "verifier"])))
    assert "GPUs (" not in one and "GPU (" not in one and "understates" not in one
    pair = two_gpu_flow(r(("cuda:0", ["prover", "verifier"]), ("cuda:1", ["verifier"])))
    assert "With 2 GPUs (cuda:0 trains and verifies; cuda:1 verifies)" in pair
    assert "run partly on cuda:1" in pair and "peer copies of the products to that GPU" in pair
    two = two_gpu_flow(r(("cuda:0", ["prover"]), ("cuda:1", ["verifier"])))
    assert "With 2 GPUs (cuda:0 trains; cuda:1 verifies)" in two
    four = two_gpu_flow(r(("cuda:0", ["prover"]), ("cuda:1", ["verifier"]),
                          ("cuda:2", ["verifier"]), ("cuda:3", ["verifier"])))
    assert "With 4 GPUs (cuda:0 trains; cuda:1, cuda:2, cuda:3 verify)" in four
    provs = two_gpu_flow(r(("cuda:0", ["prover"]), ("cuda:1", ["prover"]),
                           ("cuda:2", ["verifier"])))
    assert "With 3 GPUs (cuda:0, cuda:1 train; cuda:2 verifies)" in provs
    bad = two_gpu_flow(r(("<b>", ["prover"]), ("cuda:1", ["verifier"])))
    assert "&lt;b&gt; trains" in bad and "<b> trains" not in bad


def test_steps_steady_of_only_nulls_has_no_summary_rows():
    v = steps_view()
    v["steps_steady"] = {k: None for k in v["steps_steady"]}
    p = profile(runs={"honest": v})
    p["comparisons"] = {"sections": {}, "step": {"training_proper_s": 0.5}}
    out = panel(page(p), "r0-steps")
    assert "Steady mean" not in out and "÷ training proper</td>" not in out


def test_the_gpu_note_is_left_out_without_roles():
    for roles in (None, []):
        flow = two_gpu_flow(roles)
        assert "With " not in flow.replace("With a ", "") and "GPUs (" not in flow
        assert "GPU: the share of the section's time" in flow


def steps_view(*, pipelined: bool = False) -> dict[str, Any]:
    v = view(gpu=False)
    v["pipelined"] = pipelined
    row = {"step": 1, "warmup": True, "traced": False, "drain": False, "prover_s": 1.0,
           "verifier_s": 2.0, "prover_wait_s": None, "verifier_wait_s": None,
           "handoff_s": None, "interval_s": 3.5, "latency_s": 3.5}
    row2 = row | {"step": 2, "warmup": False, "traced": True, "drain": True, "prover_s": 1.5,
                  "verifier_s": 2.5, "prover_wait_s": 2.25, "verifier_wait_s": 0.5,
                  "handoff_s": 0.25, "interval_s": 3.25, "latency_s": 4.5}
    v["steps"] = [row, row2]
    v["steps_steady"] = {"prover_s": 1.5, "verifier_s": 2.5, "prover_wait_s": 2.25,
                         "verifier_wait_s": 0.5, "handoff_s": 0.25, "interval_s": 3.25,
                         "latency_s": 4.5}
    return v


def test_the_steps_panel_has_rows_a_steady_row_and_the_training_proper_ratio():
    p = profile(runs={"honest": steps_view()})
    p["comparisons"] = {"sections": {}, "step": {"plain_s": 0.5, "training_proper_s": 0.5}}
    out = panel(page(p), "r0-steps")
    assert out.index("Steps</h2>") < out.index("Prover only (s)")
    assert "Latency (s)" not in out
    assert '<td class="l">warm-up</td>' in out
    assert '<td class="l">traced, drain (prover idle)</td>' in out
    assert "<td>—</td><td>—</td>" in out  # step 1's waits
    assert ('<tr><td class="l">Steady mean</td><td>1.5</td><td>2.5</td><td>2.2</td>'
            '<td>0.50</td><td>0.25</td><td>3.2</td><td class="l"></td></tr>') in out
    assert ("÷ training proper</td><td>3.0</td><td>5.0</td><td>4.5</td><td>1.0</td>"
            "<td>0.50</td><td>6.5</td>") in out
    assert "Training proper is 0.50 s per step" in out
    assert '<a href="#r0-steps">' in page(p)
    assert page(p).index('id="r0-timeline"') < page(p).index('id="r0-steps"') < \
        page(p).index('id="r0-check5"')


def test_the_steps_panel_shows_latency_only_when_pipelined():
    out = panel(page(profile(runs={"honest": steps_view(pipelined=True)})), "r0-steps")
    assert "<th>Latency (s)</th>" in out and "<td>4.5</td>" in out
    assert "to its last end" in out
    assert "to its last end" not in panel(page(profile(runs={"honest": steps_view()})),
                                          "r0-steps")


def test_no_steps_panel_without_steps():
    assert "-steps" not in page(profile())
    v = view(gpu=False)
    v["steps"] = []
    assert "-steps" not in page(profile(runs={"honest": v}))


def test_plain_panel_divides_by_training_proper_when_present():
    p = with_plain("A", 2.0)
    p["comparisons"]["step"] |= {"interval_s": 1.5, "prover_side_s": 0.75,
                                 "verifier_side_s": 0.25, "training_proper_s": 0.3,
                                 "reload_s": 0.2}
    out = panel(page(p), "r0-plain")
    assert "<th>Ratio</th>" in out and "÷ plain<" not in out
    assert "against training proper</td><td>1.5</td><td>0.30</td><td>5.0</td>" in out
    assert "Prover&#x27;s work per step against training proper</td><td>0.75</td><td>0.30</td>" \
        "<td>2.5</td>" in out
    assert "Plain step with the weight reload</td><td>—</td><td>0.50</td>" in out
    assert "Weight reload, left out of training proper</td><td>—</td><td>0.20</td>" in out
    assert "<td>2.0</td>" in out  # the section row's ratio is unchanged


def test_plain_panel_without_training_proper_keeps_the_plain_denominator():
    p = with_plain("A", 2.0)
    p["comparisons"]["step"] |= {"interval_s": 1.5, "prover_side_s": 0.75,
                                 "verifier_side_s": 0.25}
    out = panel(page(p), "r0-plain")
    assert "÷ plain" in out and "training proper</td>" not in out
    assert "against plain</td><td>1.5</td><td>0.50</td><td>3.0</td>" in out
    assert "Plain step with the weight reload" not in out
    assert "The plain step includes the weight reload." in out


def test_the_phase_by_product_caption_says_how_measure_time_is_split():
    out = panel(page(profile(runs={"honest": view(gpu=True)})), "r0-check5")
    assert ("The measure time is split over the products by bytes, from one time stamp per "
            "schedule step (a stamp covers many products), and each stamp's time includes the "
            "wait for the previous step's measuring.") in out
