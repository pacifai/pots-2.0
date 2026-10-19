"""FS25: join the profiling files into ``profile.json`` and apply the bottleneck rule
(PROFILING_PLAN §5.7, §6, §7).

Run as ``python -m evaluation.results.profile.analyze --run-dir D --samples S --probe P
[--trace-summary T] [--plain D2] [--compute-dtype float32] --out profile.json``.

**Units.** Marker times are µs from the step's ``t0_ns``, sample times are monotonic ns, and
time rows and every ``*_s`` value are seconds. :func:`_abs_ns` puts marker spans on the sample
clock and :data:`US` turns µs into s; nothing else converts.

**Choices the plan leaves open** (fixed by the orchestrator, 2026-10-19):

- GEMM floor rate: prover, plain and ``train.*`` sections use ``gemm_bf16_flops`` when the
  compute dtype is bfloat16 and ``gemm_fp32_flops`` otherwise; verifier sections use
  ``gemm_fp32_flops``; check 5's sections (``5``, ``5.*``) use ``skinny_bmm_fp32_flops``.
- The compute dtype is the environment record's ``compute_dtype``, else ``--compute-dtype``
  (default ``float32``); the fallback adds a warning.
- Usable GPU memory is ``gpu_memory_bytes`` of the machine block (``probe.json`` first, then
  the environment record); without it the memory mark is skipped, with a warning.
- ``trace_summary.json`` is ``{"version", "sections": {name: {...}}, "c5_slice": {class:
  {...}}, "warnings"}``. Sections join on the name string; the time rows' step total is the
  share's denominator.
- Trace work (launches, copy bytes, GEMM bytes) is the traced step's own values; check 5's
  sections (``5``, ``5.*``) scale by ``Σ runs_total ÷ Σ runs_traced`` of ``c5_slice``.
- Copies (``memcpy_h2d``, ``memcpy_d2h``) cost at the pageable rates; ``hbm`` bytes are
  ``gemm_bytes`` plus ``memcpy_d2d``; ``hash`` bytes are ``hash_in_bytes`` plus
  ``hash_out_bytes`` of the ``count`` row; ``leaves`` come from the ``storage`` record, for
  ``P3.commit``.
- Syncs are 2 per top-level section call on CUDA. The hook bound's events are the ``gpu_us``
  spans (with the folded calls) plus one event pair per stamp; both are 0 off CUDA.
- Clock times are sums of a section's marker spans over the untraced steps; the trace time is
  compared with the host time of the traced step. ``agree`` is false when either differs by
  more than the rule's tolerance, and ``null`` when no second clock exists.
- Measured utilization on the bound's lane: ``gemm`` is ``g_tensor`` (else ``g_util``),
  ``hbm`` is ``g_dram_bw`` (else ``g_mem_util``), ``pcie`` is ``g_pcie_tx + g_pcie_rx``
  over the datasheet's ``pcie_bytes``, ``hash`` is ``c_tree`` over the probe's hash threads;
  ``launch`` has no lane, so it never gets the mark.
- Sampler fields are as FS21 writes them: utilization fields are fractions 0-1, so no
  scaling; a ``c_other`` below 0 is kept. Trace sections are inclusive and join one-to-one
  with the time rows (FS24), and no code here sums over them.
- A section's time is the time rows' ``time_s`` over untraced steps ≥ 1, else the markers'
  host time. The check-5 view has no per-class floors: the counting pass has no per-class FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from evaluation.results.profile import io

__all__ = ["US", "THRESHOLDS", "Sampler", "weighted_means", "analyze", "main"]

US = 1e-6  # seconds per µs
THRESHOLDS = {"share": 0.10, "ratio": 3.0, "reserved": 0.95, "clock_rel": 0.05,
              "clock_abs_s": 0.002, "trace_rel": 0.10, "min_samples": 3}
CLOCK_MIN_S = 0.05  # §6 step 7: the clock mark needs a section of at least 50 ms
UTIL_FACTOR = 2.0  # §6 step 5
VAR_FIELDS = ("cpu_mhz", "c_other", "g_sm_mhz", "g_throttle", "g_other_procs")
PHASES = ("run", "glue", "host_stack", "copy", "measure", "judge")


def _abs_ns(t0_ns: int, us: float) -> int:
    return t0_ns + round(us * 1000)


def _stats(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs),
            "max": max(xs), "n": len(xs)}


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


class Sampler:
    """The samples as intervals ``(previous t, t]``; the first starts at the header's ``mono_ns``."""

    def __init__(self, samples: io.Samples):
        self.rows = samples.rows
        self.ends = [int(r["t"]) for r in self.rows]
        self.starts = [int(samples.header["mono_ns"]), *self.ends[:-1]]
        fields = samples.header.get("fields") or (list(self.rows[0]) if self.rows else [])
        self.fields = [f for f in fields if f != "t"]

    def means(self, spans: Sequence[tuple[int, int]]) -> tuple[dict[str, float | None], int]:
        """Each field's mean over ``spans`` (ns), every sample weighted by the overlap of its
        interval with the spans and a null skipped; and the number of overlapping samples."""
        w: dict[int, int] = {}
        for a, b in spans:
            i = bisect_right(self.ends, a)
            while i < len(self.rows) and self.starts[i] < b:
                ov = min(self.ends[i], b) - max(self.starts[i], a)
                if ov > 0:
                    w[i] = w.get(i, 0) + ov
                i += 1
        out: dict[str, float | None] = {}
        for f in self.fields:
            num = den = 0.0
            for i, wi in w.items():
                v = self.rows[i].get(f)
                if v is not None:
                    num += wi * v
                    den += wi
            out[f] = num / den if den else None
        return out, len(w)


def weighted_means(samples: io.Samples, spans: Sequence[tuple[int, int]]
                   ) -> tuple[dict[str, float | None], int]:
    return Sampler(samples).means(spans)


def _pearson(xs: Sequence[float | None], ys: Sequence[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sxx * syy)


# ---- sections ----------------------------------------------------------------------------


def _clock_s(sec: dict[str, Any], key: str, rest_key: str) -> float | None:
    spans = sec.get(key)
    if spans is None:
        return None
    s = sum(b - a for a, b in spans) * US
    if sec.get("rest"):
        s += sec["rest"].get(rest_key) or 0.0
    return s


def _spans_ns(sec: dict[str, Any], t0_ns: int) -> list[tuple[int, int]]:
    return [(_abs_ns(t0_ns, a), _abs_ns(t0_ns, b)) for a, b in sec.get("host_us") or []]


def _side(name: str, sides: dict[str, str]) -> str:
    if name in sides:
        return sides[name]
    return "prover" if name.startswith(("P", "train.")) else "verifier"


def _is_c5(name: str) -> bool:
    return name == "5" or name.startswith("5.")


def _rate_key(name: str, sides: dict[str, str], compute_bf16: bool) -> str:
    if _is_c5(name):
        return "skinny_bmm_fp32_flops"
    if _side(name, sides) == "prover":
        return "gemm_bf16_flops" if compute_bf16 else "gemm_fp32_flops"
    return "gemm_fp32_flops"


def _direction(d: dict[str, Any] | None, key: str) -> float:
    d = d or {}
    return float(d.get(key) or d.get("memcpy_" + key) or 0.0)


def _work(name: str, rate_key: str, count: dict[str, Any] | None, leaves: int,
          trace: dict[str, Any] | None, c5_scale: float, syncs: int) -> dict[str, Any]:
    count, trace = count or {}, trace or {}
    flops = float(count.get("flops") or 0.0)
    hash_bytes = float(count.get("hash_in_bytes") or 0.0) \
        + float(count.get("hash_out_bytes") or 0.0)
    scale = c5_scale if _is_c5(name) else 1.0
    copy = trace.get("copy_bytes")
    return {"flops": {rate_key: flops} if flops else {},
            "bytes": {"hbm": scale * (float(trace.get("gemm_bytes") or 0.0)
                                      + _direction(copy, "d2d")),
                      "h2d": scale * _direction(copy, "h2d"),
                      "d2h": scale * _direction(copy, "d2h"), "hash": hash_bytes},
            "leaves": leaves, "launches": scale * float(trace.get("launches") or 0.0),
            "syncs": syncs}


def _floors(work: dict[str, Any], probe: dict[str, Any], warns: set[str]) -> dict[str, float]:
    """§6 step 2: one floor per resource that has work and a probe rate; a null rate skips
    its floor with a warning."""
    rates, costs = probe["rates"], probe.get("costs") or {}
    out: dict[str, float] = {}

    def need(label: str, value: Any) -> Any:
        if value is None:
            warns.add(f"the probe has no {label}: its floor is skipped")
        return value

    for key, fl in work["flops"].items():
        if (r := need(key, rates.get(key))) is not None:
            out["gemm"] = fl / r
    b = work["bytes"]
    if b["hbm"] and (r := need("hbm_copy_bytes", rates.get("hbm_copy_bytes"))) is not None:
        out["hbm"] = b["hbm"] / r
    pcie, found = 0.0, False
    for d, key in (("h2d", "h2d_pageable_bytes"), ("d2h", "d2h_pageable_bytes")):
        if b[d] and (r := need(key, rates.get(key))) is not None:
            pcie, found = pcie + b[d] / r, True
    if found:
        out["pcie"] = pcie
    h = rates.get("hash")
    if b["hash"] or work["leaves"]:
        if h is None or h.get("per_byte_s") is None or h.get("per_leaf_s") is None:
            warns.add("the probe has no hash rate: its floor is skipped")
        else:
            out["hash"] = b["hash"] * h["per_byte_s"] + work["leaves"] * h["per_leaf_s"]
    if work["launches"] or work["syncs"]:
        ls, ss = costs.get("launch_s"), costs.get("sync_s")
        if ls is None or ss is None:
            warns.add("the probe has no launch_s or sync_s: the launch floor is skipped")
        else:
            out["launch"] = work["launches"] * ls + work["syncs"] * ss
    return out


def _lane(bound: str | None, means: dict[str, float | None], probe: dict[str, Any]
          ) -> float | None:
    """Measured utilization (a fraction) on the bound's lane, or ``None``."""
    def frac(primary: str, fallback: str) -> float | None:
        if means.get(primary) is not None:
            return means[primary]
        return means.get(fallback)

    if bound == "gemm":
        return frac("g_tensor", "g_util")
    if bound == "hbm":
        return frac("g_dram_bw", "g_mem_util")
    if bound == "pcie":
        sheet = (probe.get("datasheet") or {}).get("pcie_bytes")
        tx, rx = means.get("g_pcie_tx"), means.get("g_pcie_rx")
        return None if not sheet or tx is None or rx is None else (tx + rx) / sheet
    if bound == "hash":
        threads = (probe["rates"].get("hash") or {}).get("threads")
        c = means.get("c_tree")
        return None if not threads or c is None else c / threads
    return None


def _differ(a: float, b: float) -> bool:
    lo, hi = sorted((a, b))
    return hi > 0 if lo <= 0 else hi / lo > UTIL_FACTOR


def _times(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    """Seconds per component per untraced step ≥ 1 from the time rows, and each side."""
    out: dict[str, dict[int, float]] = {}
    sides: dict[str, str] = {}
    for r in rows:
        if r.get("record") != "time" or r.get("traced") or r.get("derived") or r["step"] < 1:
            continue
        out.setdefault(r["component"], {}).setdefault(r["step"], r["time_s"])
        sides.setdefault(r["component"], r["side"])
    return out, sides


def _first_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each component's ``count`` row at its lowest step ≥ 1 (the counts repeat every step)."""
    out: dict[str, dict[str, Any]] = {}
    for r in sorted((r for r in rows if r.get("record") == "count" and r["step"] >= 1),
                    key=lambda r: r["step"], reverse=True):
        out[r["component"]] = r
    return out


# ---- check 5 -----------------------------------------------------------------------------


def _stamp_rows(step: dict[str, Any], layout: dict[str, Any]) -> list[dict[str, Any]]:
    """The step's stamps as rows: name, class, window and host span in s from t0."""
    st, names = step["stamps"], step["stamp_names"]
    by_group = {e["group"]: e for e in layout["entries"]}
    by_run: dict[int, dict[str, Any]] = {}
    for e in layout["entries"]:
        by_run.setdefault(e["run"], e)
    rows = []
    for i in range(len(st["name"])):
        run, group, window = st["run"][i], st["group"][i], st["window"][i]
        entry = by_group.get(group) if group >= 0 else by_run.get(run) if run >= 0 else None
        if window < 0 and entry is not None:
            window = entry["window"]
        rows.append({"name": names[st["name"][i]].removeprefix("c5."), "run": run,
                     "class": entry["class"] if entry else None, "window": window,
                     "h0": st["h0"][i] * US, "h1": st["h1"][i] * US})
    return rows


def _phase_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds per phase; ``host_stack`` is ``c5.serve − c5.copy`` (§5.2)."""
    tot = dict.fromkeys(PHASES, 0.0)
    serve = 0.0
    for r in rows:
        d = r["h1"] - r["h0"]
        if r["name"] == "serve":
            serve += d
        elif r["name"] in tot:
            tot[r["name"]] += d
    tot["host_stack"] = serve - tot["copy"]
    return tot


def _extent_ns(rows: list[dict[str, Any]], t0_ns: int) -> list[tuple[int, int]]:
    return [(_abs_ns(t0_ns, min(r["h0"] for r in rows) / US),
             _abs_ns(t0_ns, max(r["h1"] for r in rows) / US))]


def _c5_view(markers: io.Markers, layout: dict[str, Any], sampler: Sampler) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    classes: dict[str, dict[str, list[float]]] = {}
    over_time: dict[str, Any] = {}
    for s in markers.steps:
        if s["traced"] or not s.get("stamps"):
            continue
        rows = _stamp_rows(s, layout)
        if not rows:
            continue
        t0 = s["t0_ns"]
        means, _ = sampler.means(_extent_ns(rows, t0))
        per_step.append({"step": s["step"], "phases_s": _phase_totals(rows),
                         "sampler": {f: means.get(f) for f in VAR_FIELDS}})
        for w in sorted({r["window"] for r in rows if r["window"] >= 0}):
            wr = [r for r in rows if r["window"] == w]
            wm, _ = sampler.means(_extent_ns(wr, t0))
            windows.append({"step": s["step"], "window": w, "phases_s": _phase_totals(wr),
                            "sampler": {f: wm.get(f) for f in VAR_FIELDS}})
        for c in sorted({r["class"] for r in rows if r["class"]}):
            for ph, v in _phase_totals([r for r in rows if r["class"] == c]).items():
                classes.setdefault(c, {}).setdefault(ph, []).append(v)
        runs = [r for r in rows if r["name"] == "run"]
        over_time[str(s["step"])] = {"run": [r["run"] for r in runs],
                                     "class": [r["class"] for r in runs],
                                     "start_s": [r["h0"] for r in runs],
                                     "dur_s": [r["h1"] - r["h0"] for r in runs]}
    out_classes = {}
    for c, phases in classes.items():
        es = [e for e in layout["entries"] if e["class"] == c]
        means = {ph: _mean(v) for ph, v in phases.items()}
        stored = sum(e["product_bytes"] for e in es)
        served = sum(e["served_bytes"] for e in es)
        copy = means["copy"]
        out_classes[c] = {"n_runs": len({e["run"] for e in es}), "n_groups": len(es),
                          "product_bytes": stored, "served_bytes": served, "phases_s": means,
                          "copy_bytes_per_s": {"stored": stored / copy if copy else None,
                                               "served": served / copy if copy else None}}

    def corr(table: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
        return {ph: {f: _pearson([t["phases_s"][ph] for t in table],
                                 [t["sampler"][f] for t in table]) for f in VAR_FIELDS}
                for ph in PHASES}

    return {"n_runs": layout["n_runs"], "n_groups": layout["n_groups"],
            "n_windows": layout["n_windows"], "classes": out_classes,
            "phases_s": {ph: _mean([p["phases_s"][ph] for p in per_step]) for ph in PHASES},
            "runs_over_time": over_time,
            "variance": {"steps": per_step, "windows": windows,
                         "correlation": {"across_steps": corr(per_step),
                                         "across_windows": corr(windows)}}}


# ---- the analysis ------------------------------------------------------------------------


def _compute_dtype(records: list[dict[str, Any]], cli: str, warns: list[str]) -> str:
    for r in records:
        if r.get("record") == "environment" and r.get("compute_dtype"):
            return str(r["compute_dtype"])
    warns.append(f"no environment record has compute_dtype: used {cli!r} from --compute-dtype")
    return cli


def _gpu_memory(probe: dict[str, Any], records: list[dict[str, Any]]) -> int | None:
    blocks = [probe.get("machine")] + [r.get("machine") for r in records
                                        if r.get("record") == "environment"]
    for m in blocks:
        if isinstance(m, dict) and m.get("gpu_memory_bytes"):
            return int(m["gpu_memory_bytes"])
    return None


def _hook_bound(markers: io.Markers, probe: dict[str, Any]) -> float | None:
    """§7: ``(syncs × sync_s + events × event_pair_s) ÷ step time``, the mean over untraced
    steps; 0 off CUDA, where there are no syncs or events to add."""
    costs = probe.get("costs") or {}
    if costs.get("sync_s") is None or costs.get("event_pair_s") is None:
        return None
    cuda = markers.header.get("device") == "cuda"
    vals = []
    for s in markers.steps:
        if s["traced"]:
            continue
        syncs = events = 0
        if cuda:
            for sec in s["sections"]:
                if sec["depth"] == 0:
                    syncs += 2 * sec["calls"]
                if sec.get("gpu_us") is not None:
                    events += len(sec["gpu_us"]) + ((sec.get("rest") or {}).get("calls") or 0)
            events += len((s.get("stamps") or {}).get("name", []))
        step_s = (s["t1_ns"] - s["t0_ns"]) * 1e-9
        if step_s > 0:
            vals.append((syncs * costs["sync_s"] + events * costs["event_pair_s"]) / step_s)
    return _mean(vals)


def _occurrences(steps: list[dict[str, Any]], name: str
                 ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for s in steps:
        for sec in s["sections"]:
            if sec["name"] == name:
                yield s, sec


def _run_view(markers: io.Markers, records: list[dict[str, Any]], sampler: Sampler,
              probe: dict[str, Any], trace: dict[str, Any] | None,
              layout: dict[str, Any] | None, compute_bf16: bool, gpu_memory: int | None,
              warns: set[str]) -> dict[str, Any]:
    times, sides = _times(records)
    counts = _first_counts(records)
    leaves = max((r["leaves"] for r in records if r.get("record") == "storage"), default=0)
    peaks: dict[str, int] = {}
    for r in records:
        if r.get("record") == "memory" and r.get("peak_reserved_bytes") is not None:
            peaks[r["component"]] = max(peaks.get(r["component"], 0), r["peak_reserved_bytes"])
    tsec = (trace or {}).get("sections", {})
    slices = list((trace or {}).get("c5_slice", {}).values())
    traced_n = sum(c.get("runs_traced", 0) for c in slices)
    c5_scale = sum(c.get("runs_total", 0) for c in slices) / traced_n if traced_n else 1.0

    untraced = [s for s in markers.steps if not s["traced"]]
    traced = [s for s in markers.steps if s["traced"]]
    depths: dict[str, int] = {}
    for s in markers.steps:
        for sec in s["sections"]:
            depths.setdefault(sec["name"], sec["depth"])

    seconds: dict[str, list[float]] = {}
    for name in depths:
        host = [_clock_s(sec, "host_us", "host_s") or 0.0 for _, sec in _occurrences(untraced, name)]
        seconds[name] = list(times[name].values()) if name in times else host
    step_s = _mean(list(times["step"].values())) if "step" in times else \
        sum(_mean(seconds[n]) or 0.0 for n, d in depths.items() if d == 0)

    sections: dict[str, Any] = {}
    for name, depth in depths.items():
        occ = list(_occurrences(untraced, name))
        host_m = _mean([_clock_s(sec, "host_us", "host_s") or 0.0 for _, sec in occ])
        gpu_m = _mean([g for _, sec in occ if (g := _clock_s(sec, "gpu_us", "gpu_s")) is not None])
        host_traced = _mean([_clock_s(sec, "host_us", "host_s") or 0.0
                             for _, sec in _occurrences(traced, name)])
        tr = tsec.get(name)
        trace_s = tr["wall_us"] * US if tr and tr.get("wall_us") is not None else None
        agree: bool | None = None
        if gpu_m is not None and host_m is not None:
            tol = max(THRESHOLDS["clock_rel"] * host_m, THRESHOLDS["clock_abs_s"])
            agree = abs(host_m - gpu_m) <= tol
        if trace_s is not None and host_traced:
            ok = abs(trace_s - host_traced) <= THRESHOLDS["trace_rel"] * host_traced
            agree = ok if agree is None else agree and ok
        spans = [sp for s, sec in occ for sp in _spans_ns(sec, s["t0_ns"])]
        means, n = sampler.means(spans)
        time_stats = _stats(seconds[name])
        mean_t = time_stats["mean"]
        share = mean_t / step_s if step_s and mean_t is not None else None
        calls = [sec["calls"] for _, sec in occ if depth == 0 and sec.get("gpu_us") is not None]
        work = _work(name, _rate_key(name, sides, compute_bf16), counts.get(name),
                     leaves if name == "P3.commit" else 0, tr, c5_scale,
                     2 * round(_mean(calls) or 0))
        floors = _floors(work, probe, warns)
        floor_s = max(floors.values()) if floors else None
        bound = max(floors, key=floors.__getitem__) if floors else None
        ratio = mean_t / floor_s if floor_s and mean_t is not None else None
        marks = []
        if share is not None and ratio is not None and share >= THRESHOLDS["share"] \
                and ratio >= THRESHOLDS["ratio"]:
            marks.append("bottleneck")
        measured = _lane(bound, means, probe)
        if floor_s and mean_t and measured is not None and n >= THRESHOLDS["min_samples"] \
                and _differ(floor_s / mean_t, measured):
            marks.append("utilization")
        if gpu_memory and peaks.get(name) is not None \
                and peaks[name] >= THRESHOLDS["reserved"] * gpu_memory:
            marks.append("memory")
        if host_m is not None and host_m >= CLOCK_MIN_S and agree is False:
            marks.append("clock")
        sections[name] = {
            "depth": depth, "time": time_stats, "share": share,
            "clocks": {"host_s": host_m, "gpu_s": gpu_m, "trace_s": trace_s, "agree": agree},
            "util": {**{f: means.get(f) for f in sampler.fields}, "n_samples": n,
                     "few_samples": n < THRESHOLDS["min_samples"]},
            "work": work, "floors_s": floors, "floor_s": floor_s, "bound": bound,
            "ratio": ratio, "marks": marks}
    return {
        "device": markers.header.get("device"), "n_steps": len(untraced), "step_s": step_s,
        "share_total": sum(s["share"] or 0.0 for s in sections.values() if s["depth"] == 0),
        "sections": sections,
        "c5": _c5_view(markers, layout, sampler) if layout else None,
        "hook_overhead_bound": _hook_bound(markers, probe),
        "memory": {"gpu_memory_bytes": gpu_memory,
                   "sections": {n: {"peak_reserved_bytes": p,
                                    "fraction": p / gpu_memory if gpu_memory else None}
                                for n, p in peaks.items()}}}


def _plain_comparison(records: list[dict[str, Any]], plain_dir: Path) -> dict[str, Any]:
    """The plain step's sections beside the prover's ``train.*``: the plain run names them
    ``train.*`` or ``P0.*``."""
    prover = {n: statistics.fmean(v.values()) for n, v in _times(records)[0].items()}
    plain = {n: statistics.fmean(v.values())
             for n, v in _times(io.read_records(plain_dir))[0].items()}
    out: dict[str, Any] = {}
    for name, t in prover.items():
        if not name.startswith("train."):
            continue
        other = next((o for o in (name, "P0." + name.removeprefix("train.")) if o in plain), None)
        if other:
            out[name] = {"plain_name": other, "prover_s": t, "plain_s": plain[other],
                         "ratio": t / plain[other] if plain[other] else None}
    return {"plain_run_dir": str(plain_dir), "sections": out,
            "step": {"prover_s": prover.get("step"), "plain_s": plain.get("step")}}


def analyze(run_dir: Path, samples: Path, probe: Path, trace_summary: Path | None = None,
            plain: Path | None = None, compute_dtype: str = "float32") -> dict[str, Any]:
    markers = io.read_markers(run_dir / io.MARKERS_FILE)
    records = io.read_records(run_dir)
    sampler = Sampler(io.read_samples(samples))
    pr = io.read_probe(probe)
    trace = io.read_trace_summary(trace_summary) if trace_summary else None
    layout_path = run_dir / io.C5_LAYOUT_FILE
    layout = io.read_c5_layout(layout_path) if layout_path.exists() else None
    warns: list[str] = []
    dtype = _compute_dtype(records, compute_dtype, warns)
    gpu_memory = _gpu_memory(pr, records)
    if gpu_memory is None:
        warns.append("no machine block has gpu_memory_bytes: the memory mark is skipped")
    soft: set[str] = set()
    view = _run_view(markers, records, sampler, pr, trace, layout,
                     dtype.removeprefix("torch.") in ("bfloat16", "bf16"), gpu_memory, soft)
    run = markers.header.get("run")
    return {"version": io.VERSION, "run": run, "compute_dtype": dtype,
            "thresholds": THRESHOLDS,
            "warnings": warns + sorted(soft) + list((trace or {}).get("warnings") or []),
            "runs": {run: view},
            "comparisons": None if plain is None else _plain_comparison(records, plain)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join the profiling files into profile.json.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--trace-summary", type=Path, default=None)
    p.add_argument("--plain", type=Path, default=None, help="the plain run's directory")
    p.add_argument("--compute-dtype", default="float32",
                   help="used when no environment record names compute_dtype")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    result = analyze(a.run_dir, a.samples, a.probe, a.trace_summary, a.plain, a.compute_dtype)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
