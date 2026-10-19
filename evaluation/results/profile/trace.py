"""FS24: the trace summarizer (PROFILING_PLAN §5.6).

Reads one ``torch.profiler`` Chrome trace (``trace/step_0000N.json.gz``) with the standard
library and writes ``trace_summary.json`` (version 1): per section the GPU and CPU numbers, and
for check 5 the stamps' phases by product class, scaled from the traced runs to all runs.

**Sections and stamps.** A CPU ``user_annotation`` is a stamp when its name starts with ``c5.``
(``c5.glue|run=12|group=0``: the name, then ``key=value`` ids). Any other annotation is a
section; a leading ``section:`` is dropped, and torch's own annotations (``TORCH_ANNOTATIONS``)
are ignored. Sections nest, and the numbers are inclusive, like the time rows (``metrics.py``):
an event counts once in each distinct section name on its innermost occurrence's chain of
enclosing occurrences, so a name used at top level and nested elsewhere leaks nothing between
the two. A section's ``parent`` is the enclosing section of its earliest occurrence (``null``
at the top), so a reader sums top-level sections without double counting. The tree ignores
threads: spans on different threads that overlap in time count as nested. Stamps never take
events away from sections.

**Attribution.** A GPU event goes to the section whose annotation holds its launching runtime
call (same ``correlation``), by the call's start time, on any thread. The innermost
``gpu_user_annotation`` holding the event's start on its stream is a cross-check, and the share
of events where the two disagree goes into ``warnings``. Events with no launch, or a launch
outside every section, are counted in ``warnings`` and left out. Each warning appears once, with
its count.

**Per section.** ``wall_us`` is the length of the union of the section's own occurrences.
``kernel_us`` sums kernel durations and ``kernel_span_us`` is the union of their intervals, so
``gpu_idle_us = wall_us − kernel_span_us``. It is negative when a section ends before its
asynchronous kernels finish, so it's meaningful when the section ends in a sync.
``top_cpu_ops`` self times include descendants. ``sync_wait_us`` sums the host calls to
``cudaDeviceSynchronize``, ``cudaStreamSynchronize``, ``cudaEventSynchronize`` and ``cudaMemcpy``
and the ``cudaMemcpyAsync`` calls of device-to-host copies from pageable memory (these block).
``gemm_flops`` and ``gemm_bytes`` come from the input shapes of the outermost ``mm``, ``addmm``,
``bmm`` and ``matmul`` ops (torch's Chrome trace carries no ``flops`` argument; when one is
present, it's checked against the shapes, with a warning past 1%). ``gemm_flops`` is null when
a section has GEMM ops and none has readable shapes. A copy direction's bytes are null when
every copy in it lacks ``args.bytes``.

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
        ("gemm", r"gemm|cutlass|xmma|cublas|nvjet|splitkreduce"),
        ("reduction", r"reduce|softmax|norm|argm|cumsum|scan"),
        ("elementwise", r"elementwise|vectorized|unrolled|index|copy|fill|gather|scatter|clamp"),
    ))
"""Name patterns, first match wins; a kernel matching none is ``other``. Attention comes first
because flash kernels are built on cutlass. GEMM comes before reduction because cuBLAS's
split-K reduce kernel belongs to its GEMM."""

COPY_DIRECTIONS = (("htod", "h2d"), ("dtoh", "d2h"), ("dtod", "d2d"), ("ptop", "d2d"))
GEMM_OPS = frozenset({"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"})
SYNC_CALLS = frozenset({"cudaDeviceSynchronize", "cudaStreamSynchronize",
                        "cudaEventSynchronize", "cudaMemcpy"})
STAMP_PREFIX = "c5."
TORCH_ANNOTATIONS = ("ProfilerStep", "Optimizer.", "## ")
"""Annotations torch writes itself (the profiler's step, optimizer hooks, autograd's
``## name ##`` ranges). They aren't sections, and they hide no event from the sections
around them."""
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
    if name.startswith(TORCH_ANNOTATIONS):
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


class _Warnings:
    """Warning messages in first-seen order, each with its count."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def add(self, msg: str) -> None:
        self.counts[msg] = self.counts.get(msg, 0) + 1

    def lines(self) -> list[str]:
        return [m if n == 1 else f"{m} (x{n})" for m, n in self.counts.items()]


def _matmul(a: Sequence[int], b: Sequence[int]) -> tuple[int, list[int]] | None:
    """``(flops, output dims)`` of ``a @ b`` by ``torch.matmul``'s rules: 1-D operands are
    vectors and the leading dims broadcast. ``None`` when the shapes don't multiply."""
    va, vb = len(a) == 1, len(b) == 1
    a2, b2 = ([1, *a] if va else list(a)), ([*b, 1] if vb else list(b))
    if len(a2) < 2 or len(b2) < 2 or a2[-1] != b2[-2]:
        return None
    m, k, n = a2[-2], a2[-1], b2[-1]
    ba, bb = a2[:-2], b2[:-2]
    batch: list[int] = []
    for i in range(1, max(len(ba), len(bb)) + 1):
        x = ba[-i] if i <= len(ba) else 1
        y = bb[-i] if i <= len(bb) else 1
        if x != y and 1 not in (x, y):
            return None
        batch.insert(0, max(x, y))
    flops = 2 * m * n * k
    for x in batch:
        flops *= x
    return flops, batch + ([] if va else [m]) + ([] if vb else [n])


def _gemm_cost(e: Event, warn: _Warnings) -> tuple[int, int] | None:
    """``(bytes, flops)`` of a GEMM op from its input shapes and dtypes, or ``None``. Bytes are
    every tensor input plus the output; the operands are the last two tensors (``addmm`` leads
    with its bias)."""
    args = e.get("args", {})
    dims, types = args.get("Input Dims"), args.get("Input type")
    if not dims or not types or len(dims) != len(types):
        warn.add(f"{e['name']} has no input shapes or dtypes; its bytes and flops are left out")
        return None
    tensors = [(d, t) for d, t in zip(dims, types) if d]
    if len(tensors) < 2:
        warn.add(f"{e['name']} has fewer than two tensor inputs; its bytes and flops are "
                 "left out")
        return None
    res = _matmul(tensors[-2][0], tensors[-1][0])
    if res is None:
        warn.add(f"{e['name']} has input shapes that don't multiply; its bytes and flops "
                 "are left out")
        return None
    flops, out_dims = res
    total = 0
    for d, t in [*tensors, (out_dims, tensors[-2][1])]:
        if t not in DTYPE_BYTES:
            warn.add(f"unknown dtype {t!r} in {e['name']}; its bytes and flops are left out")
            return None
        n = 1
        for x in d:
            n *= x
        total += n * DTYPE_BYTES[t]
    return total, flops


def _flops_arg(e: Event) -> int | None:
    args = e.get("args", {})
    val = args.get("flops", args.get("Flops"))
    return None if val is None else int(val)


def _gemm_ops(ops: Sequence[Event], warn: _Warnings) -> dict[int, tuple[int, int] | None]:
    """``{op index: cost}`` for the GEMM ops (one thread's) that aren't inside another GEMM op,
    so ``matmul`` and its ``mm`` count once. When an op carries a ``flops`` argument, it is
    checked against the shapes' flops."""
    parent = _parents([(e["ts"], _end(e)) for e in ops])
    out: dict[int, tuple[int, int] | None] = {}
    for i, e in enumerate(ops):
        if e["name"] not in GEMM_OPS:
            continue
        p = parent[i]
        while p != -1 and ops[p]["name"] not in GEMM_OPS:
            p = parent[p]
        if p != -1:
            continue
        cost = out[i] = _gemm_cost(e, warn)
        given = _flops_arg(e)
        if cost is not None and given is not None and abs(given - cost[1]) > 0.01 * cost[1]:
            warn.add("a GEMM op's flops argument differs from its shapes' flops by more "
                     "than 1%")
    return out


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

    def result(self) -> dict[str, Any]:
        wall = _union(self.own)
        kernel_us = sum(b - a for a, b in self.kernels)
        span = _union(self.kernels)
        copy = {}
        for d in ("h2d", "d2h", "d2d"):
            lacking = self.copy_unknown.get(d, 0)
            copy[d] = self.copy_bytes[d] if d in self.copy_bytes else (None if lacking else 0)
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
    warn = _Warnings()
    events = _read_events(trace_path)

    def of(cat: str) -> list[Event]:
        return [e for e in events if e["cat"] == cat]

    # Sections and stamps: CPU annotations. Each occurrence is its own span.
    secs: list[tuple[Event, str]] = []
    stamps: list[tuple[Event, str, dict[str, int]]] = []
    for e in of("user_annotation"):
        st = _parse_annotation(e["name"])
        if st is not None:
            stamps.append((e, *st))
        elif (name := _section_name(e["name"])) is not None:
            secs.append((e, name))
    sec_spans = [(e["ts"], _end(e)) for e, _ in secs]
    sec_par = _parents(sec_spans)
    accs: dict[str, _Acc] = {}
    for e, name in secs:
        accs.setdefault(name, _Acc()).own.append((e["ts"], _end(e)))

    chains: dict[int, list[_Acc]] = {}

    def targets(i: int) -> list[_Acc]:
        """The accumulators of occurrence ``i`` and its enclosing occurrences, one per name."""
        if i == -1:
            return []
        if i not in chains:
            names: list[str] = []
            j = i
            while j != -1:
                if secs[j][1] not in names:
                    names.append(secs[j][1])
                j = sec_par[j]
            chains[i] = [accs[n] for n in names]
        return chains[i]

    def parent_name(i: int) -> str | None:
        j = sec_par[i]
        while j != -1 and secs[j][1] == secs[i][1]:
            j = sec_par[j]
        return None if j == -1 else secs[j][1]

    # A section's parent: the enclosing section of its earliest occurrence.
    parents_seen: dict[str, dict[str | None, None]] = defaultdict(dict)
    first: dict[str, int] = {}
    for i, (e, name) in enumerate(secs):
        parents_seen[name][parent_name(i)] = None
        if name not in first or (e["ts"], i) < (secs[first[name]][0]["ts"], first[name]):
            first[name] = i
    parent_of = {name: parent_name(i) for name, i in first.items()}
    for name, seen in parents_seen.items():
        if len(seen) > 1:
            warn.add(f"section {name!r} sits under several parents: {sorted(map(str, seen))}")

    # Launching calls, by correlation id.
    calls = of("cuda_runtime") + of("cuda_driver")
    call_sec = _innermost(sec_spans, [e["ts"] for e in calls])
    launch_of: dict[int, tuple[Event, int]] = {}
    seen_launch: set[int] = set()
    for e, s in zip(calls, call_sec):
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            launch_of.setdefault(corr, (e, s))
        if re.match(r"(cuda|cu)Launch", e["name"]):
            if corr is None or corr not in seen_launch:
                for a in targets(s):
                    a.launches += 1
            if corr is not None:
                seen_launch.add(corr)
        if e["name"] in SYNC_CALLS:
            for a in targets(s):
                a.sync_us += e.get("dur", 0.0)

    # GPU events.
    gpu = [e for e in events if e["cat"] in GPU_CATS]
    by_stream: dict[tuple[Any, Any], list[Event]] = defaultdict(list)
    for e in gpu:
        by_stream[(e.get("pid"), e.get("tid"))].append(e)
    gpu_ann: dict[int, str] = {}
    gann_by_stream: dict[tuple[Any, Any], list[Event]] = defaultdict(list)
    for e in of("gpu_user_annotation"):
        if _parse_annotation(e["name"]) is None and _section_name(e["name"]) is not None:
            gann_by_stream[(e.get("pid"), e.get("tid"))].append(e)
    for key, evs in by_stream.items():
        anns = gann_by_stream.get(key, [])
        hit = _innermost([(a["ts"], _end(a)) for a in anns], [e["ts"] for e in evs])
        for e, i in zip(evs, hit):
            if i != -1:
                gpu_ann[id(e)] = _section_name(anns[i]["name"]) or ""
    no_launch = outside = compared = disagree = unknown_copy = no_bytes = 0
    sync_runtime: set[int] = set()
    for e in gpu:
        corr = e.get("args", {}).get("correlation")
        found = launch_of.get(corr) if corr is not None else None
        if found is None:
            no_launch += 1
            continue
        call, occ = found
        if occ == -1:
            outside += 1
            continue
        tg = targets(occ)
        if id(e) in gpu_ann:
            compared += 1
            disagree += gpu_ann[id(e)] != secs[occ][1]
        dur = e.get("dur", 0.0)
        if e["cat"] == "kernel":
            cat = _kernel_category(e["name"])
            for a in tg:
                a.kernels.append((e["ts"], _end(e)))
                a.cats[cat] += dur
        elif e["cat"] == "gpu_memset":
            for a in tg:
                a.cats["memset"] += dur
        else:
            d = _copy_direction(e["name"])
            if d is None:
                unknown_copy += 1
                for a in tg:
                    a.cats["other"] += dur
                continue
            nbytes = e.get("args", {}).get("bytes")
            no_bytes += nbytes is None
            blocking = (d == "d2h" and "pageable" in e["name"].lower()
                        and call["name"] == "cudaMemcpyAsync" and id(call) not in sync_runtime)
            sync_runtime.add(id(call))
            for a in tg:
                a.cats[f"memcpy_{d}"] += dur
                if nbytes is None:
                    a.copy_unknown[d] = a.copy_unknown.get(d, 0) + 1
                else:
                    a.copy_bytes[d] = a.copy_bytes.get(d, 0) + int(nbytes)
                if blocking:
                    a.sync_us += call.get("dur", 0.0)
    if no_launch:
        warn.add(f"{no_launch} GPU events have no launching runtime call; left out")
    if outside:
        warn.add(f"{outside} GPU events were launched outside every section; left out")
    if unknown_copy:
        warn.add(f"{unknown_copy} memcpy events have no readable direction; counted as other")
    if no_bytes:
        warn.add(f"{no_bytes} memcpy events have no args.bytes; their bytes are null")
    if disagree:
        warn.add(f"{100 * disagree / compared:.1f}% of GPU events ({disagree} of "
                 f"{compared}) sit in a different gpu_user_annotation than their "
                 "launch's section")

    # CPU ops: self time, and the GEMM ops' bytes and flops.
    ops = of("cpu_op")
    op_sec = _innermost(sec_spans, [e["ts"] for e in ops])
    by_thread: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for i, e in enumerate(ops):
        by_thread[(e.get("pid"), e.get("tid"))].append(i)
    for idx in by_thread.values():
        thread = [ops[i] for i in idx]
        par = _parents([(e["ts"], _end(e)) for e in thread])
        child_dur = [0.0] * len(thread)
        for j, p in enumerate(par):
            if p != -1:
                child_dur[p] += thread[j].get("dur", 0.0)
        gemm = _gemm_ops(thread, warn)
        for j, e in enumerate(thread):
            for a in targets(op_sec[idx[j]]):
                row = a.ops[e["name"]]
                row[0] += max(e.get("dur", 0.0) - child_dur[j], 0.0)
                row[1] += 1
                if j in gemm:
                    a.gemm_ops += 1
                    if (cost := gemm[j]) is not None:
                        a.gemm_bytes += cost[0]
                        a.gemm_flops = (a.gemm_flops or 0) + cost[1]
    out_secs = {name: {"parent": parent_of[name], **acc.result()} for name, acc in accs.items()}

    if markers_path is not None:
        want = _markers_sections(Path(markers_path))
        if want is None:
            warn.add("the markers file has no traced step line")
        else:
            for name in sorted(want - set(out_secs)):
                warn.add(f"section {name!r} is in the markers but not the trace")
            for name in sorted(set(out_secs) - want):
                warn.add(f"section {name!r} is in the trace but not the markers")

    c5: dict[str, Any] = {}
    if layout_path is not None:
        try:
            c5 = _c5_slice(stamps, Path(layout_path), trace_path, warn)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            warn.add(f"the layout {str(layout_path)!r} is missing or malformed ({exc!r}); "
                     "no check-5 view")
    return {"version": VERSION, "sections": out_secs, "c5_slice": c5, "warnings": warn.lines()}


def _c5_slice(stamps: Sequence[tuple[Event, str, dict[str, int]]], layout_path: Path,
              trace_path: Path, warn: _Warnings) -> dict[str, Any]:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    cls: dict[int, str] = {}
    members: dict[int, int] = defaultdict(int)
    for ent in layout["entries"]:
        run = ent["run"]
        if cls.setdefault(run, ent["class"]) != ent["class"]:
            warn.add("a layout run has entries of two classes; the first is used")
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
        warn.add(f"stamped runs {unknown} are not in the layout; left out")
    meta = trace_path.parent / "meta.json"
    all_traced = False
    if meta.exists():
        all_traced = json.loads(meta.read_text(encoding="utf-8")).get("c5_slice") == "all"
    want = layout.get("trace_slice")
    if want is not None and not all_traced and sorted(traced) != sorted(want):
        warn.add(f"traced runs {sorted(traced)} differ from the layout's "
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
