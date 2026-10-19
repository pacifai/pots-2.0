"""FS26: the HTML report on synthetic ``profile.json`` dicts (PROFILING_PLAN FS26, §12)."""

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


def view(*, gpu: bool, c5: bool = True, tl: bool = True, fwd_mean: float = 0.123456,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    secs = {
        "train.forward": sec(0, fwd_mean, 0.3, flops={"gemm_fp32_flops": 1e9},
                             util={"g_util": 0.7} if gpu else None),
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
        ph = {"run": 0.2, "glue": 0.1, "host_stack": 0.01, "copy": 0.02, "measure": 0.05,
              "judge": 0.01}
        c5v = {"n_runs": 2, "n_groups": 2, "n_windows": 1, "phases_s": ph,
               "classes": {"G_down": {"n_runs": 2, "n_groups": 2, "product_bytes": 100,
                                      "served_bytes": 100, "phases_s": ph,
                                      "copy_bytes_per_s": {"stored": None, "served": None}}},
               "runs_over_time": {"1": {"run": [0, 1], "class": ["G_down", "G_down"],
                                        "start_s": [0.0, 0.1], "dur_s": [0.05, 0.05]}},
               "variance": {"steps": [{"step": 1, "phases_s": ph, "sampler": {"c_other": None}}],
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
            "machine": {"cpu_model": "X", "cpu": {"n": 4}},
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
    for a in ("inputs", "r0-overview", "r0-diagram", "r0-sections", "r0-timeline", "r0-check5",
              "appendix"):
        assert f'<section id="{a}">' in out and f'href="#{a}"' in out
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
    out = page(profile("A"), profile("B"))
    assert 'id="compare-models"' in out and "compare-sessions" not in out
    t = panel(out, "compare-models")
    assert "A · refusal" in t and "B · refusal" in t


def test_two_sessions_sharing_a_model():
    out = page(profile("A", "s1"), profile("B", "s1"), profile("A", "s2"))
    t = panel(out, "compare-sessions")
    assert "A · refusal · honest · s1" in t and "A · refusal · honest · s2" in t
    assert "B ·" not in t
    assert "compare-models" in out  # s1 holds two models
    assert "compare-sessions" not in page(profile("A", "s1"), profile("B", "s2"))


def test_a_missing_section_is_blank_in_a_comparison():
    extra = {"7": sec(0, 0.3, 0.3)}
    out = page(profile("A", "s1", runs={"honest": view(gpu=False, extra=extra)}),
               profile("A", "s2"))
    t = panel(out, "compare-sessions")
    assert "<th>7</th>" in t and "<td></td>" in t


def test_a_cpu_profile_has_no_gpu_lanes_or_strips():
    out = page(profile(gpu=False))
    arch = panel(out, "r0-diagram")
    assert "CPU threads" in arch and "Host" in arch
    for lane in ("GPU compute", "GPU memory", "PCIe"):
        assert lane not in arch
    tl = panel(out, "r0-timeline")
    assert "c_busy" in tl and "g_util" not in tl and "g_pcie" not in tl
    gpu = page(profile(gpu=True))
    arch = panel(gpu, "r0-diagram")
    for lane in ("GPU compute", "GPU memory (HBM)", "PCIe", "CPU threads", "Host"):
        assert lane in arch
    assert "g_util" in panel(gpu, "r0-timeline") and "g_pcie_tx" in panel(gpu, "r0-timeline")


def test_an_empty_panel_is_left_out_of_page_and_bar():
    out = page(profile(c5=False, tl=False))
    for a in ("r0-check5", "r0-timeline"):
        assert f'id="{a}"' not in out and f'href="#{a}"' not in out
    assert 'id="r0-overview"' in out
    bare = profile()
    bare.update({"probe": None, "machine": None, "thresholds": None})
    out = page(bare)
    assert 'id="appendix"' not in out and 'href="#appendix"' not in out


def test_the_picker_hides_a_runs_missing_panel_with_its_link():
    out = page(profile(scenario="a"), profile(scenario="b", c5=False))
    assert 'id="r0-check5"' in out and 'id="r1-check5"' not in out
    assert '<span data-run="0"><a href="#r0-check5">' in out


def test_hover_title_equals_the_table_cell():
    out = page(profile(fwd_mean=0.123456))
    cell = report.fmt(0.123456)
    assert cell == "0.1235"
    assert f"<td>{cell}</td>" in panel(out, "r0-sections")
    assert f"mean {cell} s" in panel(out, "r0-overview")
    assert f"mean {cell} s" in panel(out, "r0-diagram")


def test_the_diagram_nests_check_5_phases_and_outlines_marks():
    arch = panel(page(profile()), "r0-diagram")
    for ph in ("5.glue", "5.host_stack", "5.copy", "5.measure", "5.judge", "6b.glue"):
        assert f"<title>{ph}\n" in arch
    assert "5.run" not in arch
    assert 'stroke="var(--mark)" stroke-width="3"' in arch  # P3.commit carries a mark
    assert "<title>P2\n" not in arch  # a rollup


def test_plain_comparison_and_appendix():
    p = profile()
    p["comparisons"] = {"sections": {"train.forward": {"plain_name": "train.forward",
                                                       "prover_s": 0.2, "plain_s": 0.1,
                                                       "ratio": 2.0}},
                        "step": {"prover_s": 1.0, "plain_s": 0.5}}
    out = page(p)
    assert "<td>2</td>" in panel(out, "compare-plain")
    ap = panel(out, "appendix")
    assert "hash.per_byte_s" in ap and "cpu.n" in ap and "share" in ap
    assert "compare-plain" not in page(profile())


def test_the_cli_writes_the_file(tmp_path: Path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(profile("A")))
    b.write_text(json.dumps(profile("B")))
    out = tmp_path / "profile.html"
    assert report.main([str(a), str(b), "--out", str(out)]) == 0
    text = out.read_text()
    assert "compare-models" in text and "a.json" in text
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 2}))
    with pytest.raises(ValueError, match="bad.json"):
        report.main([str(bad), "--out", str(out)])


def test_fmt():
    assert report.fmt(None) == "—" and report.fmt(True) == "yes"
    assert report.fmt(3) == "3" and report.fmt(1234567) == "1.235e+06"
    assert report.fmt(0.000123456) == "0.0001235"


def big_view(n_runs: int = 830, steps: int = 6, n_sec: int = 25) -> dict[str, Any]:
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
    c5 = v["c5"]
    c5["runs_over_time"] = {str(k): {"run": list(range(n_runs)),
                                     "class": [("G_a", "G_b")[i % 2] for i in range(n_runs)],
                                     "start_s": [i * 0.01 for i in range(n_runs)],
                                     "dur_s": [0.005] * n_runs} for k in range(1, steps + 1)}
    ph = c5["phases_s"]
    c5["variance"]["windows"] = [{"step": k, "window": w, "phases_s": ph,
                                  "sampler": {"c_other": 0.1}}
                                 for k in range(1, steps + 1) for w in range(50)]
    return v


def test_a_full_scale_report_stays_below_5_mb():
    out = page(*[profile(f"M{i}", runs={"honest": big_view()}) for i in range(8)])
    size = len(out.encode())
    assert size < 5_000_000, f"{size} bytes"


def test_runs_over_time_is_binned_above_300_runs():
    over = panel(page(profile(runs={"honest": big_view(830, 1, 2)})), "r0-check5")
    assert over.count("<title>runs ") == 277 and "<title>runs 0-2\n" in over
    assert "seconds 0.015 s" in over
    small = panel(page(profile(runs={"honest": big_view(300, 1, 2)})), "r0-check5")
    assert "<title>run 299\n" in small and "<title>runs " not in small


def test_notes_and_label_colors():
    out = page(profile(gpu=False))
    assert "run s counts each run's whole time" in panel(out, "r0-check5")
    nostrips = profile(runs={"honest": view(gpu=False)})
    for st in nostrips["runs"]["honest"]["timeline"]["steps"]:
        st["strips"] = None
    tl = panel(page(nostrips), "r0-timeline")
    assert "No sampler data for this run." in tl and "Strips: ." not in tl
    arch = panel(out, "r0-diagram")
    assert 'style="fill:#111"' in arch or 'style="fill:#fff"' in arch
