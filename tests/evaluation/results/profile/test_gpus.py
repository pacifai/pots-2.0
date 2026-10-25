"""FS35: the profile on a four-GPU sequential run (the prover on cuda:0, the verifier's units on
cuda:1-3), a version 2 probe and version 2 samples; a version 1 pair stays as it was."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluation.results.profile import analyze, io, report
from tests.evaluation.results.profile.helpers import (
    count_row,
    jl,
    probe,
    section,
    step_line,
    time_row,
)

DEVS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
UUIDS = {d: f"GPU-{i}000" for i, d in enumerate(DEVS)}
UTIL = {"cuda:0": 0.9, "cuda:1": 0.1, "cuda:2": 0.2, "cuda:3": 0.3}  # also g_sm_active
GEMM = {"cuda:0": 4e11, "cuda:1": 3e11, "cuda:2": 1e11, "cuda:3": 2e11}
HBM = {"cuda:0": 2e12, "cuda:1": 1e12, "cuda:2": 5e11, "cuda:3": 1.5e12}
PER_BYTE = {"cuda:0": 1e-10, "cuda:1": 1e-10, "cuda:2": 3e-10, "cuda:3": 2e-10}
SYNC = {"cuda:0": 1e-5, "cuda:1": 2e-5, "cuda:2": 8e-5, "cuda:3": 3e-5}
VERIFIER = ["cuda:1", "cuda:2", "cuda:3"]
# name: (start_s, dur_s, devices, flops)
SECS = {"P1": (0.0, 0.2, ["cuda:0"], 4e10), "V": (0.2, 0.6, VERIFIER, 3e10),
        "Q": (0.8, 0.1, ["cuda:2"], 1e10), "H": (0.9, 0.1, VERIFIER, 0.0),
        "N": (0.95, 0.01, None, 0.0)}


def v2_probe(devices=DEVS) -> dict:
    per = {}
    for d in devices:
        per[d] = {"name": "H100", "uuid": UUIDS[d], "datasheet": {"pcie_bytes": 6e10},
                  "rates": {"gemm_bf16_flops": GEMM[d] * 4, "gemm_fp32_flops": GEMM[d],
                            "skinny_bmm_fp32_flops": GEMM[d] / 10, "hbm_copy_bytes": HBM[d],
                            "h2d_pageable_bytes": 1e10, "d2h_pageable_bytes": 1e10,
                            "hash_kernel": {"per_byte_s": PER_BYTE[d], "per_leaf_s": 1e-6,
                                            "residual": 0.0, "mixes": 3}},
                  "costs": {"launch_s": 5e-6, "sync_s": SYNC[d], "event_pair_s": 4e-6},
                  "errors": {}}
    # as I3 says, the top-level costs are the first device's
    return probe(version=2, device="cuda:0", devices=list(devices), per_device=per, peer={},
                 peer_errors={}, costs=per[devices[0]]["costs"])


def v2_samples(path: Path, order=DEVS, *, with_uuids=True, n=90) -> None:
    """One row per 0.5 s; each GPU field holds the values in the header's ``order``."""
    gpus = [{"index": DEVS.index(d), "name": "H100", "pci_bus_id": f"0:{i}",
             **({"uuid": UUIDS[d]} if with_uuids else {})} for i, d in enumerate(order)]
    fields = ["t", "c_tree", "g_util", "g_sm_active", "g_pcie_tx", "g_pcie_rx"]
    header = {"kind": "header", "version": 2, "pid": 1, "interval_s": 0.5, "mono_ns": 0,
              "wall_unix_s": 1.0, "fields": fields, "gpus": gpus}
    rows = [{"t": int(i * 0.5e9), "c_tree": 1.0,
             "g_util": [UTIL[d] for d in order], "g_sm_active": [UTIL[d] for d in order],
             "g_pcie_tx": [1e9 * (DEVS.index(d) + 1) for d in order],
             "g_pcie_rx": [2e9 * (DEVS.index(d) + 1) for d in order]} for i in range(1, n + 1)]
    jl(path, [header, *rows])


def build(root: Path, *, order=DEVS, roles=True, probe_obj=None,
          memory_rows=(), steps=4, with_uuids=True) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    header = {"kind": "header", "version": 1, "run": "honest", "pid": 1, "device": "cuda:0",
              "mono_ns": 0, "wall_unix_s": 1.0}
    lines = [header]
    gpu_roles = [{"device": d, "index": i, "name": "H100", "uuid": UUIDS[d],
                  "roles": ["prover"] if i == 0 else ["verifier"]}
                 for i, d in enumerate(DEVS)] if roles else None
    env = {"record": "environment", "run": "honest", "compute_dtype": "float32",
           "device": "cuda:0", "gpu_roles": gpu_roles}
    records = [env, {"record": "storage", "run": "honest", "scenario": "honest", "step": 1,
                     "transcript_bytes": 3_000_000_000, "leaves": 3000}]
    for k in range(1, steps + 1):
        secs = []
        for n, (a, d, devs, _) in SECS.items():
            sec = section(n, a, d)
            if devs is not None:
                sec["devices"] = devs
            secs.append(sec)
        lines.append(step_line(k, secs))
        records += [time_row(k, n, v[1], side="prover" if n.startswith("P") else "verifier")
                    for n, v in SECS.items()]
        records.append(time_row(k, "step", sum(v[1] for v in SECS.values()), level="total"))
    for n, v in SECS.items():
        if v[3]:
            records.append(count_row(1, n, flops=v[3]))
    records.append(count_row(1, "H", hash_in_bytes=3_000_000_000))
    records += list(memory_rows)
    jl(root / "markers.jsonl", lines)
    jl(root / "records.jsonl", records)
    v2_samples(root / "samples.jsonl", order, with_uuids=with_uuids)
    (root / "probe.json").write_text(json.dumps(probe_obj or v2_probe()), encoding="utf-8")
    return {"run_dir": root, "samples": root / "samples.jsonl", "probe": root / "probe.json"}


def run(tmp_path: Path, **kw):
    paths = build(tmp_path / "run", **kw)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    return out, out["runs"]["honest"]["sections"]


def test_each_sections_utilization_is_its_own_gpus(tmp_path):
    _, s = run(tmp_path)
    assert s["P1"]["util"]["g_util"] == pytest.approx(0.9)
    assert "util_by_device" not in s["P1"]
    assert s["Q"]["util"]["g_util"] == pytest.approx(0.2)  # cuda:2 alone
    assert s["N"]["util"]["g_util"] == pytest.approx(0.9)  # no device list: the run's device
    v = s["V"]
    assert v["util"]["g_util"] == pytest.approx(0.2)  # the mean of its verifier devices (FS42c)
    assert v["util_devices"] == VERIFIER and s["P1"]["util_devices"] == ["cuda:0"]
    assert {d: u["g_util"] for d, u in v["util_by_device"].items()} == pytest.approx(
        {"cuda:1": 0.1, "cuda:2": 0.2, "cuda:3": 0.3})
    assert v["util_by_device"]["cuda:3"]["n_samples"] >= 3


def test_the_uuid_maps_when_nvml_order_differs_from_torchs(tmp_path):
    out, s = run(tmp_path, order=["cuda:2", "cuda:0", "cuda:3", "cuda:1"])
    assert s["P1"]["util"]["g_util"] == pytest.approx(0.9)
    assert s["Q"]["util"]["g_util"] == pytest.approx(0.2)
    assert s["V"]["util_by_device"]["cuda:3"]["g_util"] == pytest.approx(0.3)
    assert not [w for w in out["warnings"] if "UUID" in w or "index" in w]
    # the timeline strips are labelled by the device the UUID names, in the file's GPU order
    gpus = out["runs"]["honest"]["timeline"]["steps"][0]["strips"]["gpus"]
    assert [g["device"] for g in gpus] == ["cuda:2", "cuda:0", "cuda:3", "cuda:1"]
    assert gpus[1]["roles"] == ["prover"] and gpus[1]["label"] == "cuda:0 (prover)"


def test_the_runs_own_uuid_wins_over_the_probes_and_the_machine_records(tmp_path):
    order = ["cuda:2", "cuda:0", "cuda:3", "cuda:1"]
    rot = {d: UUIDS[DEVS[(i + 1) % 4]] for i, d in enumerate(DEVS)}  # a wrong set of UUIDs
    pr = v2_probe()
    pr["per_device"] = {d: {**v, "uuid": rot[d]} for d, v in pr["per_device"].items()}
    pr["machine"] = {"gpus": [{"index": i, "uuid": rot[d]} for i, d in enumerate(DEVS)]}
    out, s = run(tmp_path, order=order, probe_obj=pr)
    assert s["Q"]["util"]["g_util"] == pytest.approx(0.2)  # cuda:2 by gpu_roles' UUID
    assert s["V"]["util_by_device"]["cuda:3"]["g_util"] == pytest.approx(0.3)
    assert not [w for w in out["warnings"] if "UUID" in w]


def test_without_uuids_the_same_index_is_used_with_a_warning(tmp_path):
    out, s = run(tmp_path, roles=False, with_uuids=False,
                 probe_obj={**v2_probe(), "per_device": {d: {**v, "uuid": None} for d, v in
                                                          v2_probe()["per_device"].items()}})
    assert s["Q"]["util"]["g_util"] == pytest.approx(0.2)
    assert any("no GPU UUIDs" in w for w in out["warnings"])


def test_floors_come_from_the_sections_own_device(tmp_path):
    _, s = run(tmp_path)
    assert s["P1"]["floors_s"]["gemm"] == pytest.approx(4e10 / GEMM["cuda:0"])
    assert s["Q"]["floors_s"]["gemm"] == pytest.approx(1e10 / GEMM["cuda:2"])


def test_a_multi_device_section_uses_the_slowest_device(tmp_path):
    _, s = run(tmp_path)
    assert s["V"]["floors_s"]["gemm"] == pytest.approx(3e10 / min(GEMM[d] for d in VERIFIER))


def test_the_hash_floor_splits_evenly_over_the_devices_and_takes_the_slowest(tmp_path):
    _, s = run(tmp_path)
    assert s["H"]["work"]["leaves"] == 3000
    each = {d: 1e9 * PER_BYTE[d] + 1000 * 1e-6 for d in VERIFIER}
    assert s["H"]["floors_s"]["hash"] == pytest.approx(max(each.values()))


def test_the_host_hash_rate_is_used_when_the_transcript_is_on_the_host(tmp_path):
    _, s = run(tmp_path, roles=False)
    # probe()'s host hash: 1e-9 s per byte, 0 per leaf
    assert s["H"]["floors_s"]["hash"] == pytest.approx(3e9 * 1e-9)


def test_the_lane_mark_uses_the_largest_utilization_over_the_devices(tmp_path):
    _, s = run(tmp_path)
    # V: floor 0.3 s over 0.6 s is 0.5; lanes 0.1, 0.2, 0.3: the largest, 0.3, is within 2x
    # (the first device's 0.1 alone would differ by 5x)
    assert "utilization" not in s["V"]["marks"]
    # Q on cuda:2 alone: floor 0.1 s over 0.1 s is 1.0 against its own 0.2: differs by 5x
    assert "utilization" in s["Q"]["marks"]


def test_a_v1_probe_gives_the_first_devices_rates_with_a_warning_per_device(tmp_path):
    out, s = run(tmp_path, probe_obj=probe(device="cuda:0"))
    assert s["Q"]["floors_s"]["gemm"] == pytest.approx(1e10 / 1e11)
    assert sum("no measurements of cuda:2" in w for w in out["warnings"]) == 1


def test_a_device_the_probe_didnt_measure_warns_once_and_uses_the_first_devices(tmp_path):
    out, s = run(tmp_path, probe_obj=v2_probe(DEVS[:3]))
    assert any("no measurements of cuda:3" in w for w in out["warnings"])
    assert s["V"]["floors_s"]["gemm"] == pytest.approx(3e10 / GEMM["cuda:2"])


def test_the_hook_bound_costs_each_devices_syncs_at_its_own_cost(tmp_path):
    out, _ = run(tmp_path)
    # syncs: 2 per call of a top-level section on each of its devices; events: one pair per
    # section span on its device (4e-6 each)
    syncs = {"cuda:0": 2, "cuda:1": 4, "cuda:2": 6, "cuda:3": 4}
    events = {"cuda:0": 0, "cuda:1": 0, "cuda:2": 0, "cuda:3": 0}
    over = sum(n * SYNC[d] for d, n in syncs.items()) \
        + sum(n * 4e-6 for n in events.values())
    # N lists no device: its 2 syncs cost the first device's, 1e-5
    over += 2 * SYNC["cuda:0"]
    # H and V: counted above per device; the step is 1 s long
    assert out["runs"]["honest"]["hook_overhead_bound"] == pytest.approx(over, rel=1e-6)


def mem(comp: str, peaks: list[int]) -> dict:
    return {"record": "memory", "run": "honest", "scenario": "honest", "step": 1,
            "side": "verifier", "component": comp, "level": "component", "calls": 1,
            "peak_bytes": max(peaks), "peak_reserved_bytes": max(peaks),
            "devices": VERIFIER, "device_peak_reserved_bytes": peaks}


def test_each_devices_reserved_peak_is_checked_against_its_own_memory(tmp_path):
    gib = 2 ** 30
    pr = v2_probe()
    pr["machine"] = {"gpu_memory_bytes": 80 * gib, "gpus": [
        {"index": 0, "memory_bytes": 80 * gib}, {"index": 1, "memory_bytes": 80 * gib},
        {"index": 2, "memory_bytes": 40 * gib}, {"index": 3, "memory_bytes": 80 * gib}]}
    rows = [mem("V", [10 * gib, 39 * gib, 10 * gib]), mem("Q", [10 * gib, 30 * gib, 10 * gib])]
    out, s = run(tmp_path, probe_obj=pr, memory_rows=rows)
    # 39 GiB of cuda:2's 40 GiB is over 95%; the 80 GiB device's whole-run maximum would not be
    assert "memory" in s["V"]["marks"] and "memory" not in s["Q"]["marks"]
    mem_block = out["runs"]["honest"]["memory"]
    assert mem_block["gpu_memory_bytes_by_device"]["cuda:2"] == 40 * gib
    assert mem_block["sections"]["V"]["by_device"]["cuda:2"]["fraction"] == pytest.approx(0.975)


def test_the_report_draws_one_strip_per_gpu(tmp_path):
    out, _ = run(tmp_path)
    html = report.build([out])
    for label in ("cuda:0 (prover)", "cuda:1 (verifier)", "cuda:3 (verifier)"):
        assert len(re.findall(rf"GPU busy {re.escape(label)}", html)) >= 4  # label + tooltips
    assert html.count("PCIe cuda:2 (verifier)") >= 1
    assert "one per GPU" in html


def test_a_v1_file_pair_reads_and_has_one_gpu(tmp_path):
    from tests.evaluation.results.profile.helpers import build_run
    paths = build_run(tmp_path / "v1")
    sm = io.read_samples(paths["samples"])
    sampler = analyze.Sampler(sm)
    assert sampler.version == 1 and sampler.n_gpus == 1 and sampler.view(0) is sampler
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert "gpus" not in out["runs"]["honest"]["timeline"]["steps"][0]["strips"]


def test_the_hash_floor_splits_over_the_verifier_devices_only(tmp_path):
    # a sequential sharded run lists all four devices on every section; cuda:0 is prover-only
    paths = build(tmp_path / "run")
    lines = [json.loads(x) for x in (paths["run_dir"] / "markers.jsonl").read_text().splitlines()]
    for ln in lines:
        for sec in ln.get("sections", []):
            if sec["name"] == "H":
                sec["devices"] = DEVS
    jl(paths["run_dir"] / "markers.jsonl", lines)
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    h = out["runs"]["honest"]["sections"]["H"]
    each = {d: 3e9 / 3 * PER_BYTE[d] + 1000 * 1e-6 for d in VERIFIER}
    assert h["floors_s"]["hash"] == pytest.approx(max(each.values()))
    # the other floors still use the slowest of all four listed devices
    assert len(h["util_by_device"]) == 4


# ---- FS42c: the headline by role, and the step time as the interval between verified steps ----

INTERVAL = {1: 2.0, 2: 0.8, 3: 0.9, 4: 1.0}  # step 1 is the warm-up
WALL = {k: 1.0 + 0.1 * k for k in INTERVAL}


def _rewrite(paths: dict[str, Path], *, devices: dict[str, list[str]] | None = None,
             merged: bool = False, extra_rows=None, add_secs=()) -> None:
    """Edit the built run: set each section's devices, mark steps merged, add time rows and
    marker sections."""
    mp = paths["run_dir"] / "markers.jsonl"
    lines = [json.loads(x) for x in mp.read_text().splitlines()]
    for ln in lines:
        if ln.get("kind") != "step":
            continue
        ln["merged"] = merged or None
        ln["sections"] += [dict(s) for s in add_secs]
        for sec in ln["sections"]:
            if devices and sec["name"] in devices:
                sec["devices"] = devices[sec["name"]]
    jl(mp, lines)
    if extra_rows:
        rp = paths["run_dir"] / "records.jsonl"
        recs = [json.loads(x) for x in rp.read_text().splitlines()]
        jl(rp, recs + extra_rows(sorted(INTERVAL)))


def _sequential(tmp_path: Path, **kw) -> dict:
    """Every section lists all four devices, as the sequential loop's one process does."""
    paths = build(tmp_path / "run", **kw)
    _rewrite(paths, devices={n: DEVS for n in SECS})
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    return out["runs"]["honest"]


def _pipelined_rows(steps):
    rows = []
    for k in steps:
        rows += [time_row(k, "step_interval", INTERVAL[k], "run", "total"),
                 time_row(k, "step_wall", WALL[k], "run", "total"),
                 time_row(k, "prover", 0.4, "prover", "total"),
                 time_row(k, "verifier", 0.8, "verifier", "total")]
    return rows


def _pipelined(tmp_path: Path, *, plain: bool = False, steady_steps=None):
    paths = build(tmp_path / "run")  # the build's step total is the sum of the sections, 1.01
    _rewrite(paths, devices={"P1": ["cuda:0"], "V": VERIFIER, "Q": ["cuda:2"], "H": VERIFIER},
             merged=True, extra_rows=_pipelined_rows)
    kw = {"steady_steps": steady_steps}
    if plain:
        pd = tmp_path / "plain"
        pd.mkdir()
        jl(pd / "records.jsonl", [time_row(1, "step", 0.3, "prover", "total")])
        kw["plain"] = pd
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"], **kw)
    return out, out["runs"]["honest"]


def test_a_sequential_verifier_sections_headline_is_the_mean_of_the_verifier_gpus(tmp_path):
    s = _sequential(tmp_path)["sections"]
    v = s["V"]  # lists all four devices: GPU 0 (the idle training GPU, 0.9 here) is not averaged
    assert v["util"]["g_util"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert v["util"]["g_sm_active"] == pytest.approx(0.2)
    assert v["util_devices"] == VERIFIER
    assert list(v["util_by_device"]) == DEVS
    assert v["util_by_device"]["cuda:0"]["g_util"] == pytest.approx(0.9)
    assert v["util"]["n_samples"] == min(u["n_samples"] for d, u in v["util_by_device"].items()
                                         if d in VERIFIER)


def test_a_sequential_prover_sections_headline_is_gpu_0s(tmp_path):
    p = _sequential(tmp_path)["sections"]["P1"]
    assert p["util"]["g_util"] == pytest.approx(0.9) and p["util_devices"] == ["cuda:0"]
    assert list(p["util_by_device"]) == DEVS


def test_a_run_side_section_or_a_run_without_roles_keeps_the_first_device(tmp_path):
    view = _sequential(tmp_path, roles=False)
    assert view["sections"]["V"]["util"]["g_util"] == pytest.approx(0.9)
    assert view["sections"]["V"]["util_devices"] == ["cuda:0"]
    # a section with no time row has no side: the first device too
    paths = build(tmp_path / "r2")
    _rewrite(paths, devices={"X": DEVS}, add_secs=[section("X", 0.97, 0.01)])
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    x = out["runs"]["honest"]["sections"]["X"]
    assert x["util"]["g_util"] == pytest.approx(0.9) and x["util_devices"] == ["cuda:0"]


def test_the_lane_mark_of_a_sequential_section_is_taken_over_its_role_devices(tmp_path):
    # Q lists all four devices: its floor (1e10 flops at the slowest GPU's 1e11) is 0.1 s over
    # its 0.1 s, so 1.0; the verifier GPUs measure at most 0.3, a 3x difference. Over all four
    # devices the largest lane, GPU 0's 0.9, would hide it.
    assert "utilization" in _sequential(tmp_path)["sections"]["Q"]["marks"]


def test_the_check_5_view_reads_the_verifier_gpus(tmp_path):
    from tests.evaluation.results.profile.test_analyze import c5_inputs
    names, st, layout = c5_inputs(tmp_path)
    paths = build(tmp_path / "run")
    _rewrite(paths, devices={n: DEVS for n in SECS},
             add_secs=[{**section("5", 0.0, 0.0002), "devices": DEVS}],
             extra_rows=lambda steps: [time_row(k, "5", 0.0002) for k in steps])
    lines = [json.loads(x) for x in (paths["run_dir"] / "markers.jsonl").read_text().splitlines()]
    for ln in lines[1:]:
        ln["stamps"], ln["stamp_names"] = st, names
    jl(paths["run_dir"] / "markers.jsonl", lines)
    (paths["run_dir"] / "c5_layout.json").write_text(json.dumps(layout))
    # a g_sm_mhz per GPU: 1000 on GPU 0 (the training GPU), 1100-1300 on the verifier GPUs
    sm = [1000 + 100 * i for i in range(4)]
    header, *rows = [json.loads(x) for x in paths["samples"].read_text().splitlines()]
    header["fields"] = [*header["fields"], "g_sm_mhz"]
    jl(paths["samples"], [header, *({**r, "g_sm_mhz": sm} for r in rows)])
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    steps = out["runs"]["honest"]["c5"]["variance"]["steps"]
    assert steps and all(t["sampler"]["g_sm_mhz"] == pytest.approx(1200) for t in steps)


def test_mean_view_is_one_view_for_one_device_and_the_field_mean_for_several():
    samples = io.Samples({"kind": "header", "version": 2, "mono_ns": 0, "fields": ["t", "g_util"],
                          "gpus": [{"index": 0}, {"index": 1}]},
                         [{"t": 10, "g_util": [0.2, None]}, {"t": 20, "g_util": [0.4, 0.5]}])
    sam = analyze.Sampler(samples)
    one = analyze._mean_means([sam.view(0).means([(0, 20)])])
    assert one == sam.view(0).means([(0, 20)])
    both, n = analyze._MeanView([sam.view(0), sam.view(1)]).means([(0, 20)])
    # GPU 0: (0.2 + 0.4) / 2 = 0.3; GPU 1: only the second sample, 0.5; the mean of the two
    assert both["g_util"] == pytest.approx(0.4) and n == 2


def test_the_default_steady_rule_includes_a_pipelined_runs_last_step_and_warns(tmp_path):
    out, v = _pipelined(tmp_path)
    assert v["steady_steps"] == [2, 3, 4]
    assert v["step_s"] == pytest.approx((0.8 + 0.9 + 1.0) / 3)  # the warm-up step 1 is left out
    assert any("last step (4)" in w and "--steady-steps" in w for w in out["warnings"])


def test_a_pipelined_runs_step_time_is_the_interval_between_verified_steps(tmp_path):
    _, v = _pipelined(tmp_path, steady_steps=[2, 3])  # step 4 is the drain step
    assert v["step_source"] == "step_interval" and v["pipelined"] is True
    assert v["steady_steps"] == [2, 3] and v["n_steps"] == 2
    assert v["step_s"] == pytest.approx((0.8 + 0.9) / 2)  # the warm-up step 1 is left out
    assert v["work_s"] == pytest.approx(1.01)  # the step total row's mean
    assert v["latency_s"] == pytest.approx((1.2 + 1.3) / 2)
    assert v["sections"]["P1"]["share"] == pytest.approx(0.2 / v["step_s"])
    assert v["share_by_side"]["prover"] == pytest.approx(0.2 / v["step_s"])
    assert v["share_by_side"]["verifier"] == pytest.approx(0.81 / v["step_s"])
    assert v["warmup"]["step_s"] == pytest.approx(2.0) and v["warmup"]["latency_s"] == \
        pytest.approx(1.1)
    # the interval and latency rows are not sections
    assert not {"step_interval", "step_wall", "prover", "verifier"} & set(v["sections"])
    # prover sections list one GPU, verifier sections the verifier GPUs
    assert v["sections"]["P1"]["util_devices"] == ["cuda:0"]
    assert v["sections"]["V"]["util_devices"] == VERIFIER


def test_the_plain_comparison_holds_the_interval_and_the_side_totals(tmp_path):
    out, _ = _pipelined(tmp_path, plain=True)
    st = out["comparisons"]["step"]
    assert st["interval_s"] == pytest.approx(0.9)
    assert st["prover_side_s"] == pytest.approx(0.4) and st["verifier_side_s"] == pytest.approx(0.8)
    assert st["prover_s"] == pytest.approx(1.01) and st["plain_s"] == pytest.approx(0.3)


def test_the_report_says_the_pipelined_step_time_and_compares_the_sides_with_plain(tmp_path):
    import html as htmllib
    out, _ = _pipelined(tmp_path, plain=True)
    secs = out["runs"]["honest"]["sections"]  # give two sections names the report knows
    out["runs"]["honest"]["sections"] = {{"P1": "P1.label", "V": "4"}.get(n, n): v
                                         for n, v in secs.items()}
    page = htmllib.unescape(report.build([out]))
    assert "Verified steps closed every" in page and "0.90 s" in page
    assert "prover trains the next step while the verifier checks this one" in page
    assert "latency" in page and "1.3 s" in page
    assert "the prover worked 0.20 s" in page and "the verifier 0.60 s" in page
    assert "sum to at most about 100%" in page
    assert "Whole step: interval between verified steps against plain" in page
    assert "Prover's work per step against plain" in page
    assert "Verifier's work per step against plain" in page
    assert "GPU busy 20% (mean of cuda:1, cuda:2, cuda:3)" in page


def test_the_report_of_a_sequential_interval_run_says_so_in_the_old_sentence(tmp_path):
    import html as htmllib
    paths = build(tmp_path / "run")
    _rewrite(paths, extra_rows=lambda steps: [time_row(k, "step_interval", 1.0, "run", "total")
                                              for k in steps])
    out = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert out["runs"]["honest"]["pipelined"] is False
    assert out["runs"]["honest"]["latency_s"] is None
    page = htmllib.unescape(report.build([out]))
    assert "(interval between verified steps)" in page and "Verified steps closed" not in page
    assert "the mean interval between verified steps" in page


def test_records_without_step_interval_keep_the_step_total_and_its_shares(tmp_path):
    from tests.evaluation.results.profile.helpers import build_run
    paths = build_run(tmp_path / "run")
    v = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])["runs"]["honest"]
    assert v["step_source"] == "step" and v["step_s"] == pytest.approx(1.0)
    assert v["work_s"] == v["step_s"] and v["pipelined"] is False and v["latency_s"] is None
    assert v["sections"]["D"]["share"] == pytest.approx(0.55)
    assert v["share_total"] == pytest.approx(1.0)
    assert v["share_by_side"] == {"verifier": pytest.approx(1.0)}
    assert v["sections"]["A"]["util_devices"] is None  # a CPU run has no device


def test_the_flow_notes_that_prover_side_gpu_use_understates_with_several_role_gpus(tmp_path):
    import html as htmllib
    out, _ = _pipelined(tmp_path)
    assert "understates them" in htmllib.unescape(report.build([out]))
    from tests.evaluation.results.profile.helpers import build_run
    paths = build_run(tmp_path / "one", device="cuda:0")
    one = analyze.analyze(paths["run_dir"], paths["samples"], paths["probe"])
    assert "understates them" not in htmllib.unescape(report.build([one]))
