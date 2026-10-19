"""FS24: the trace summarizer (PROFILING_PLAN §5.6).

Reads one ``torch.profiler`` Chrome trace (``trace/step_0000N.json.gz``) with the standard
library and writes ``trace_summary.json`` (version 1): per section the GPU and CPU numbers, and
for check 5 the stamps' phases by product class, scaled from the traced runs to all runs.

**Sections and stamps.** A CPU ``user_annotation`` is a stamp when its name starts with ``c5.``
(``c5.glue|run=12|group=0``: the name, then ``key=value`` ids). Any other annotation is a
section; a leading ``section:`` is dropped and the profiler's own ``ProfilerStep#`` ranges are
ignored. Sections nest, and the numbers are inclusive, like the time rows (``metrics.py``): an
event counts in its innermost section and in every enclosing one. A section's ``parent`` names
the section enclosing it (``null`` at the top), so a reader sums top-level sections without
double counting. Stamps never take events away from sections.

**Attribution.** A GPU event goes to the section whose annotation holds its launching runtime
call (same ``correlation``), by the call's start time, on any thread. The innermost
``gpu_user_annotation`` holding the event's start on its stream is a cross-check, and the share
of events where the two disagree goes into ``warnings``. Events with no launch, or a launch
outside every section, are counted in ``warnings`` and left out.

**Per section.** ``wall_us`` is the length of the union of the section's own annotations
(children included). ``kernel_us`` sums kernel durations and
``kernel_span_us`` is the union of their intervals, so ``gpu_idle_us = wall_us − kernel_span_us``
(about zero or above, up to clock skew). ``top_cpu_ops`` self times include descendants. ``sync_wait_us`` sums the host calls to
``cudaDeviceSynchronize``, ``cudaStreamSynchronize``, ``cudaEventSynchronize`` and ``cudaMemcpy``
and the ``cudaMemcpyAsync`` calls of device-to-host copies from pageable memory (these block).
``gemm_flops`` is null when a section has GEMM ops and none carries a ``flops`` argument.
A copy direction's bytes are null when every copy in it lacks ``args.bytes``.

**Check 5.** A stamp's span is its annotation's duration. A class's ``phases_us`` sum the
stamps of its traced runs (``stack`` is ``serve − copy``), and ``scale`` is the class's products
over its traced runs' products (``members`` summed over the layout's entries), so unequal runs
weigh by their products. The window stamps (``c5.judge``) have no class and aren't summarized.

Run as ``python -m evaluation.results.profile.trace --trace T [--layout L] [--markers M]
--out trace_summary.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["VERSION", "KERNEL_CATEGORIES", "CATEGORIES", "GEMM_OPS", "summarize", "main"]

VERSION = 1

KEEP_CATS = frozenset({"user_annotation", "gpu_user_annotation", "cpu_op", "kernel",
                       "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"})
GPU_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
CATEGORIES = ("gemm", "attention", "elementwise", "reduction", "memcpy_h2d", "memcpy_d2h",
              "memcpy_d2d", "memset", "other")
KERNEL_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (cat, re.compile(pat, re.IGNORECASE)) for cat, pat in (
        ("attention", r"flash|fmha|attention"),
        ("gemm", r"gemm|cutlass|xmma|cublas"),
        ("reduction", r"reduce|softmax|norm|argm|cumsum|scan"),
        ("elementwise", r"elementwise|vectorized|unrolled|index|copy|fill|gather|scatter|clamp"),
    ))
"""Name patterns, first match wins; a kernel matching none is ``other``. Attention comes first
because flash kernels are built on cutlass."""

COPY_DIRECTIONS = (("htod", "h2d"), ("dtoh", "d2h"), ("dtod", "d2d"), ("ptop", "d2d"))
GEMM_OPS = frozenset({"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"})
SYNC_CALLS = frozenset({"cudaDeviceSynchronize", "cudaStreamSynchronize",
                        "cudaEventSynchronize", "cudaMemcpy"})
STAMP_PREFIX = "c5."
PHASES = ("run", "glue", "serve", "copy", "stack", "measure")
DTYPE_BYTES = {"float": 4, "double": 8, "c10::Half": 2, "c10::BFloat16": 2, "long": 8,
               "int": 4, "short": 2, "char": 1, "signed char": 1, "unsigned char": 1,
               "bool": 1, "c10::Float8_e4m3fn": 1, "c10::Float8_e5m2": 1}
TOP_OPS = 10

Event = dict[str, Any]


def _read_events(path: Path) -> list[Event]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and e.get("cat") in KEEP_CATS]


def _end(e: Event) -> float:
    return e["ts"] + e.get("dur", 0.0)


def _union(intervals: Sequence[tuple[float, float]]) -> float:
    """The length of the union of ``(start, end)`` intervals."""
    total, hi = 0.0, None
    cur = None
    for a, b in sorted(intervals):
        if hi is None or a > hi:
            if cur is not None:
                total += hi - cur
            cur, hi = a, b
        else:
            hi = max(hi, b)
    if cur is not None:
        total += hi - cur
    return total


def _parents(spans: Sequence[tuple[float, float]]) -> list[int]:
    """For each ``(start, end)`` span, the index of the innermost other span holding its start,
    or -1. Spans nest like calls: sorted by start, longer first on ties."""
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], -spans[i][1]))
    out = [-1] * len(spans)
    stack: list[int] = []
    for i in order:
        t = spans[i][0]
        while stack and spans[stack[-1]][1] <= t:
            stack.pop()
        for j in reversed(stack):
            if spans[j][1] > t:
                out[i] = j
                break
        stack.append(i)
    return out


def _innermost(spans: Sequence[tuple[float, float]], points: Sequence[float]) -> list[int]:
    """For each time in ``points``, the index of the innermost span holding it, or -1."""
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], -spans[i][1]))
    out = [-1] * len(points)
    stack: list[int] = []
    k = 0
    for p in sorted(range(len(points)), key=points.__getitem__):
        t = points[p]
        while k < len(order) and spans[order[k]][0] <= t:
            stack.append(order[k])
            k += 1
        while stack and spans[stack[-1]][1] <= t:
            stack.pop()
        for j in reversed(stack):
            if spans[j][1] > t:
                out[p] = j
                break
    return out


def _parse_annotation(name: str) -> tuple[str, dict[str, int]] | None:
    """A stamp's ``(name, ids)``, or ``None`` for a section."""
    if not name.startswith(STAMP_PREFIX):
        return None
    head, *rest = name.split("|")
    ids: dict[str, int] = {}
    for item in rest:
        key, _, val = item.partition("=")
        try:
            ids[key] = int(val)
        except ValueError:
            pass
    return head, ids


def _section_name(name: str) -> str | None:
    if name.startswith("ProfilerStep"):
        return None
    return name.removeprefix("section:")


def _kernel_category(name: str) -> str:
    for cat, pat in KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def _copy_direction(name: str) -> str | None:
    low = name.lower()
    for key, direction in COPY_DIRECTIONS:
        if key in low:
            return direction
    return None


def _gemm_ops(ops: Sequence[Event], warnings: list[str]) -> dict[int, tuple[int | None, int | None]]:
    """``{op index: (bytes, flops)}`` for the GEMM ops that aren't inside another GEMM op (for
    flops: inside another that carries ``flops``), so ``matmul`` and its ``mm`` count once."""
    spans = [(e["ts"], _end(e)) for e in ops]
    parent = _parents(spans)
    # Parents are per thread; the caller passes one thread's ops.
    is_gemm = [e["name"] in GEMM_OPS for e in ops]
    flops_of = [_flops(e) if is_gemm[i] else None for i, e in enumerate(ops)]
    out: dict[int, tuple[int | None, int | None]] = {}
    for i, e in enumerate(ops):
        if not is_gemm[i]:
            continue
        under_any = under_flops = False
        p = parent[i]
        while p != -1:
            if is_gemm[p]:
                under_any = True
                under_flops |= flops_of[p] is not None
            p = parent[p]
        out[i] = (None if under_any else _gemm_bytes(e, warnings),
                  None if under_flops else flops_of[i])
    return out


def _flops(e: Event) -> int | None:
    args = e.get("args", {})
    val = args.get("flops", args.get("Flops"))
    return None if val is None else int(val)


def _gemm_bytes(e: Event, warnings: list[str]) -> int | None:
    """Bytes read and written by a GEMM op, from its input shapes: every tensor input plus the
    output, ``a.shape[:-1] + b.shape[-1:]`` for the last two tensors ``a`` and ``b``."""
    args = e.get("args", {})
    dims, types = args.get("Input Dims"), args.get("Input type")
    if not dims or not types or len(dims) != len(types):
        warnings.append(f"{e['name']} has no input shapes or dtypes; its bytes are left out")
        return None
    tensors = [(d, t) for d, t in zip(dims, types) if d]
    if len(tensors) < 2:
        warnings.append(f"{e['name']} has fewer than two tensor inputs; its bytes are left out")
        return None
    sizes = []
    for d, t in tensors + [(tensors[-2][0][:-1] + tensors[-1][0][-1:], tensors[-2][1])]:
        if t not in DTYPE_BYTES:
            warnings.append(f"unknown dtype {t!r} in {e['name']}; its bytes are left out")
            return None
        n = 1
        for x in d:
            n *= x
        sizes.append(n * DTYPE_BYTES[t])
    return sum(sizes)


class _Acc:
    def __init__(self) -> None:
        self.own: list[tuple[float, float]] = []
        self.kernels: list[tuple[float, float]] = []
        self.cats = dict.fromkeys(CATEGORIES, 0.0)
        self.launches = 0
        self.copy_bytes: dict[str, int] = {}
        self.copy_unknown: dict[str, int] = {}
        self.gemm_bytes = 0
        self.gemm_flops: int | None = None
        self.gemm_ops = 0
        self.sync_us = 0.0
        self.ops: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])

    def merge(self, o: "_Acc") -> None:
        self.kernels += o.kernels
        for c, v in o.cats.items():
            self.cats[c] += v
        self.launches += o.launches
        for d, v in o.copy_bytes.items():
            self.copy_bytes[d] = self.copy_bytes.get(d, 0) + v
        for d, v in o.copy_unknown.items():
            self.copy_unknown[d] = self.copy_unknown.get(d, 0) + v
        self.gemm_bytes += o.gemm_bytes
        self.gemm_ops += o.gemm_ops
        if o.gemm_flops is not None:
            self.gemm_flops = (self.gemm_flops or 0) + o.gemm_flops
        self.sync_us += o.sync_us
        for n, (us, k) in o.ops.items():
            row = self.ops[n]
            row[0] += us
            row[1] += k

    def result(self) -> dict[str, Any]:
        wall = _union(self.own)
        kernel_us = sum(b - a for a, b in self.kernels)
        span = _union(self.kernels)
        copy = {}
        for d in ("h2d", "d2h", "d2d"):
            lacking = self.copy_unknown.get(d, 0)
            has = d in self.copy_bytes
            copy[d] = self.copy_bytes[d] if has else (None if lacking else 0)
        top = sorted(self.ops.items(), key=lambda kv: (-kv[1][0], kv[0]))[:TOP_OPS]
        return {
            "wall_us": wall, "kernel_us": kernel_us, "kernel_span_us": span,
            "gpu_idle_us": wall - span, "kernels": len(self.kernels), "launches": self.launches,
            "categories_us": self.cats, "copy_bytes": copy,
            "gemm_flops": self.gemm_flops if self.gemm_ops else 0, "gemm_bytes": self.gemm_bytes,
            "sync_wait_us": self.sync_us,
            "top_cpu_ops": [{"name": n, "self_us": v[0], "calls": v[1]} for n, v in top],
        }


def _markers_sections(path: Path) -> set[str] | None:
    """The section names of the traced step's marker line, or ``None`` without one."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "step" and row.get("traced"):
            return {s["name"] for s in row.get("sections", [])}
    return None


def summarize(trace_path: str | Path, layout_path: str | Path | None = None,
              markers_path: str | Path | None = None) -> dict[str, Any]:
    """The ``trace_summary.json`` content (§5.6) for one trace, with its check-5 view when the
    layout is given and a section cross-check when the markers are."""
    trace_path = Path(trace_path)
    warnings: list[str] = []
    events = _read_events(trace_path)

    def of(cat: str) -> list[Event]:
        return [e for e in events if e["cat"] == cat]

    # Sections and stamps: CPU annotations.
    secs: list[tuple[Event, str]] = []
    stamps: list[tuple[Event, str, dict[str, int]]] = []
    for e in of("user_annotation"):
        st = _parse_annotation(e["name"])
        if st is not None:
            stamps.append((e, *st))
        elif (name := _section_name(e["name"])) is not None:
            secs.append((e, name))
    sec_spans = [(e["ts"], _end(e)) for e, _ in secs]
    accs: dict[str, _Acc] = {}
    for _, name in secs:
        accs.setdefault(name, _Acc())
    sec_par = _parents(sec_spans)
    parent_of: dict[str, str | None] = {}
    below: dict[str, set[str]] = defaultdict(set)
    for i, ((e, name), par) in enumerate(zip(secs, sec_par)):
        accs[name].own.append((e["ts"], _end(e)))
        up = par
        while up != -1 and secs[up][1] == name:
            up = sec_par[up]
        if parent_of.get(name) is None:
            parent_of[name] = None if up == -1 else secs[up][1]
        while up != -1:
            if secs[up][1] != name:
                below[secs[up][1]].add(name)
            up = sec_par[up]

    def section_of(points: Sequence[float]) -> list[str | None]:
        return [None if i == -1 else secs[i][1] for i in _innermost(sec_spans, points)]

    # Launching calls, by correlation id.
    calls = [e for e in of("cuda_runtime") + of("cuda_driver")]
    call_sec = section_of([e["ts"] for e in calls])
    launch_of: dict[int, tuple[Event, str | None]] = {}
    seen_launch: set[int] = set()
    for e, s in zip(calls, call_sec):
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            launch_of.setdefault(corr, (e, s))
        if re.match(r"(cuda|cu)Launch", e["name"]) and s is not None:
            if corr is None or corr not in seen_launch:
                accs[s].launches += 1
            if corr is not None:
                seen_launch.add(corr)
        if e["name"] in SYNC_CALLS and s is not None:
            accs[s].sync_us += e.get("dur", 0.0)

    # GPU events.
    gpu = [e for e in events if e["cat"] in GPU_CATS]
    by_stream: dict[tuple[Any, Any], list[Event]] = defaultdict(list)
    for e in gpu:
        by_stream[(e.get("pid"), e.get("tid"))].append(e)
    gpu_ann: dict[int, str] = {}
    gann = [e for e in of("gpu_user_annotation")
            if _parse_annotation(e["name"]) is None and _section_name(e["name"]) is not None]
    gann_by_stream: dict[tuple[Any, Any], list[Event]] = defaultdict(list)
    for e in gann:
        gann_by_stream[(e.get("pid"), e.get("tid"))].append(e)
    for key, evs in by_stream.items():
        anns = gann_by_stream.get(key, [])
        hit = _innermost([(a["ts"], _end(a)) for a in anns], [e["ts"] for e in evs])
        for e, i in zip(evs, hit):
            if i != -1:
                gpu_ann[id(e)] = _section_name(anns[i]["name"]) or ""
    no_launch = outside = compared = disagree = unknown_copy = 0
    sync_runtime: set[int] = set()
    for e in gpu:
        corr = e.get("args", {}).get("correlation")
        found = launch_of.get(corr) if corr is not None else None
        if found is None:
            no_launch += 1
            continue
        call, sec = found
        if sec is None:
            outside += 1
            continue
        acc = accs[sec]
        if id(e) in gpu_ann:
            compared += 1
            disagree += gpu_ann[id(e)] != sec
        dur = e.get("dur", 0.0)
        if e["cat"] == "kernel":
            acc.kernels.append((e["ts"], _end(e)))
            acc.cats[_kernel_category(e["name"])] += dur
        elif e["cat"] == "gpu_memset":
            acc.cats["memset"] += dur
        else:
            d = _copy_direction(e["name"])
            if d is None:
                unknown_copy += 1
                acc.cats["other"] += dur
                continue
            acc.cats[f"memcpy_{d}"] += dur
            nbytes = e.get("args", {}).get("bytes")
            if nbytes is None:
                acc.copy_unknown[d] = acc.copy_unknown.get(d, 0) + 1
            else:
                acc.copy_bytes[d] = acc.copy_bytes.get(d, 0) + int(nbytes)
            if d == "d2h" and "pageable" in e["name"].lower() \
                    and call["name"] == "cudaMemcpyAsync" and id(call) not in sync_runtime:
                sync_runtime.add(id(call))
                acc.sync_us += call.get("dur", 0.0)
    if no_launch:
        warnings.append(f"{no_launch} GPU events have no launching runtime call; left out")
    if outside:
        warnings.append(f"{outside} GPU events were launched outside every section; left out")
    if unknown_copy:
        warnings.append(f"{unknown_copy} memcpy events have no readable direction; "
                        "counted as other")
    lacking = sum(sum(a.copy_unknown.values()) for a in accs.values())  # still exclusive here
    if lacking:
        warnings.append(f"{lacking} memcpy events have no args.bytes; their bytes are null")
    if disagree:
        warnings.append(f"{100 * disagree / compared:.1f}% of GPU events ({disagree} of "
                        f"{compared}) sit in a different gpu_user_annotation than their "
                        "launch's section")

    # CPU ops: self time, and the GEMM ops' bytes and flops.
    ops = of("cpu_op")
    op_sec = section_of([e["ts"] for e in ops])
    by_thread: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for i, e in enumerate(ops):
        by_thread[(e.get("pid"), e.get("tid"))].append(i)
    no_flops = 0
    for idx in by_thread.values():
        thread = [ops[i] for i in idx]
        par = _parents([(e["ts"], _end(e)) for e in thread])
        child_dur = [0.0] * len(thread)
        for j, p in enumerate(par):
            if p != -1:
                child_dur[p] += thread[j].get("dur", 0.0)
        gemm = _gemm_ops(thread, warnings)
        for j, e in enumerate(thread):
            s = op_sec[idx[j]]
            if s is None:
                continue
            row = accs[s].ops[e["name"]]
            row[0] += max(e.get("dur", 0.0) - child_dur[j], 0.0)
            row[1] += 1
            if j in gemm:
                nbytes, flops = gemm[j]
                acc = accs[s]
                acc.gemm_ops += 1
                if nbytes is not None:
                    acc.gemm_bytes += nbytes
                if flops is not None:
                    acc.gemm_flops = (acc.gemm_flops or 0) + flops
                elif nbytes is not None and _flops(e) is None:
                    no_flops += 1
    out_secs = {}
    for name, own in accs.items():
        total = _Acc()
        total.own = own.own
        for n in (name, *sorted(below[name])):
            total.merge(accs[n])
        out_secs[name] = {"parent": parent_of[name], **total.result()}
    if no_flops:
        warnings.append(f"{no_flops} GEMM ops have no flops argument")

    if markers_path is not None:
        want = _markers_sections(Path(markers_path))
        if want is None:
            warnings.append("the markers file has no traced step line")
        else:
            for name in sorted(want - set(out_secs)):
                warnings.append(f"section {name!r} is in the markers but not the trace")
            for name in sorted(set(out_secs) - want):
                warnings.append(f"section {name!r} is in the trace but not the markers")

    summary: dict[str, Any] = {"version": VERSION, "sections": out_secs}
    summary["c5_slice"] = (_c5_slice(stamps, Path(layout_path), trace_path, warnings)
                           if layout_path is not None else {})
    summary["warnings"] = warnings
    return summary


def _c5_slice(stamps: Sequence[tuple[Event, str, dict[str, int]]], layout_path: Path,
              trace_path: Path, warnings: list[str]) -> dict[str, Any]:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    cls: dict[int, str] = {}
    members: dict[int, int] = defaultdict(int)
    for ent in layout["entries"]:
        run = ent["run"]
        if cls.setdefault(run, ent["class"]) != ent["class"]:
            warnings.append(f"layout run {run} has entries of two classes; the first is used")
        members[run] += ent["members"]
    total_runs: dict[str, set[int]] = defaultdict(set)
    for run, c in cls.items():
        total_runs[c].add(run)

    traced: set[int] = set()
    phases: dict[int, dict[str, float]] = defaultdict(lambda: dict.fromkeys(PHASES, 0.0))
    for e, name, ids in stamps:
        run = ids.get("run")
        phase = name.removeprefix(STAMP_PREFIX)
        if run is None or phase not in PHASES:
            continue
        if phase == "run":
            traced.add(run)
        phases[run][phase] += e.get("dur", 0.0)
    for run in phases:
        phases[run]["stack"] = phases[run]["serve"] - phases[run]["copy"]
    unknown = sorted(r for r in traced | set(phases) if r not in cls)
    if unknown:
        warnings.append(f"stamped runs {unknown} are not in the layout; left out")
    meta = trace_path.parent / "meta.json"
    all_traced = False
    if meta.exists():
        all_traced = json.loads(meta.read_text(encoding="utf-8")).get("c5_slice") == "all"
    want = layout.get("trace_slice")
    if want is not None and not all_traced and sorted(traced) != sorted(want):
        warnings.append(f"traced runs {sorted(traced)} differ from the layout's "
                        f"trace_slice {sorted(want)}")

    out: dict[str, Any] = {}
    for c, runs in sorted(total_runs.items()):
        got = sorted(r for r in runs if r in traced)
        if not got:
            continue
        sums = {p: sum(phases[r][p] for r in got) for p in PHASES}
        denom = sum(members[r] for r in got)
        scale = sum(members[r] for r in runs) / denom if denom else None
        out[c] = {"runs_traced": len(got), "runs_total": len(runs), "scale": scale,
                  "phases_us": sums,
                  "scaled": None if scale is None else {p: v * scale for p, v in sums.items()}}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--layout", type=Path)
    ap.add_argument("--markers", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    summary = summarize(args.trace, args.layout, args.markers)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for w in summary["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
