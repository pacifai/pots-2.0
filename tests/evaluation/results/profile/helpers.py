"""Hand-built profiling inputs (PROFILING_PLAN §5, version 1) for the FS25 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STEP_NS = 10_000_000_000  # step k starts at k × 10 s on the monotonic clock


def jl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def section(name: str, start_s: float, dur_s: float, *, depth: int = 0, gpu_dur_s: float | None = None
            ) -> dict[str, Any]:
    gpu = None if gpu_dur_s is None else [[start_s * 1e6, (start_s + gpu_dur_s) * 1e6]]
    return {"name": name, "depth": depth, "calls": 1,
            "host_us": [[start_s * 1e6, (start_s + dur_s) * 1e6]], "gpu_us": gpu, "rest": None,
            "alloc": None}


def step_line(step: int, sections: list[dict[str, Any]], *, traced: bool = False,
              length_s: float = 1.0, stamps: dict[str, Any] | None = None,
              stamp_names: list[str] | None = None) -> dict[str, Any]:
    t0 = step * STEP_NS
    return {"kind": "step", "scenario": "honest", "step": step, "traced": traced, "t0_ns": t0,
            "t1_ns": t0 + int(length_s * 1e9), "sections": sections, "stamps": stamps,
            "stamp_names": stamp_names or []}


def time_row(step: int, comp: str, t: float, side: str = "verifier", level: str = "component"
             ) -> dict[str, Any]:
    return {"record": "time", "run": "honest", "scenario": "honest", "step": step, "side": side,
            "component": comp, "level": level, "calls": 1, "time_s": t, "derived": False,
            "traced": False}


def count_row(step: int, comp: str, **fields: Any) -> dict[str, Any]:
    return {"record": "count", "run": "honest", "scenario": "honest", "step": step,
            "side": "verifier", "component": comp, "level": "component", "calls": 1,
            "flops": 0, "hash_in_bytes": 0, "hash_out_bytes": 0, "hash_calls": 0, **fields}


def probe(**over: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "version": 1, "device": "cpu", "machine": {}, "threads": 8, "determinism": {},
        "rates": {"gemm_bf16_flops": 1e12, "gemm_fp32_flops": 1e11,
                  "skinny_bmm_fp32_flops": 1e10, "hbm_copy_bytes": 1e11,
                  "h2d_pageable_bytes": 1e10, "d2h_pageable_bytes": 1e10,
                  "hash": {"per_byte_s": 1e-9, "per_leaf_s": 0.0, "threads": 8}},
        "costs": {"launch_s": 5e-6, "sync_s": 1.5e-5, "event_pair_s": 4e-6},
        "shapes": [], "datasheet": None, "errors": {}, "wall_s": 1.0}
    p.update(over)
    return p


def samples_file(path: Path, rows: list[dict[str, Any]], *, start_ns: int = 0) -> None:
    header = {"kind": "header", "version": 1, "pid": 1, "interval_s": 1.0, "mono_ns": start_ns,
              "wall_unix_s": 1.0, "fields": ["t", "c_tree", "g_util"]}
    jl(path, [header, *rows])


def build_run(root: Path, *, steps: int = 3, secs: dict[str, tuple[float, ...]] | None = None,
              hash_bytes: dict[str, int] | None = None, records_extra: list[dict[str, Any]] = (),
              sample_rows: list[dict[str, Any]] | None = None, probe_over: dict[str, Any] | None = None,
              step_extra: list[dict[str, Any]] = (), device: str = "cpu",
              depths: dict[str, int] | None = None) -> dict[str, Path]:
    """A run of ``steps`` one-second steps. ``secs`` maps a section to (start_s, dur_s); the
    time rows repeat the marker durations, and ``hash_bytes`` gives each section's counted
    bytes. Returns the paths the CLI takes."""
    root.mkdir(parents=True, exist_ok=True)
    secs = secs or {"A": (0.0, 0.2), "B": (0.2, 0.2), "C": (0.4, 0.05), "D": (0.45, 0.55)}
    header = {"kind": "header", "version": 1, "run": "honest", "pid": 1, "device": device,
              "mono_ns": 0, "wall_unix_s": 1.0}
    lines = [header]
    records = [{"record": "environment", "run": "honest", "compute_dtype": "float32"}]
    for k in range(1, steps + 1):
        depth = depths or {}
        lines.append(step_line(k, [section(n, v[0], v[1], depth=depth.get(n, 0),
                                           gpu_dur_s=v[2] if len(v) > 2 else None)
                                   for n, v in secs.items()]))
        records += [time_row(k, n, v[1], level="sub" if depth.get(n) else "component")
                    for n, v in secs.items()]
        records.append(time_row(k, "step", sum(v[1] for n, v in secs.items() if not depth.get(n)),
                                level="total"))
    records += [count_row(1, n, hash_in_bytes=b) for n, b in (hash_bytes or {}).items()]
    records += list(records_extra)
    lines += list(step_extra)
    jl(root / "markers.jsonl", lines)
    jl(root / "records.jsonl", records)
    samples = sample_rows if sample_rows is not None else [
        {"t": int(i * 1e9), "c_tree": 1.0, "g_util": 0.5} for i in range(1, 41)]
    samples_file(root / "samples.jsonl", samples)
    (root / "probe.json").write_text(json.dumps(probe(**(probe_over or {}))), encoding="utf-8")
    return {"run_dir": root, "samples": root / "samples.jsonl", "probe": root / "probe.json"}
