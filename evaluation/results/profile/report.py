"""FS26: the HTML report of one or more ``profile.json`` files (PROFILING_PLAN FS26, §12).

Run as ``python -m evaluation.results.profile.report PROFILE [PROFILE ...] --out profile.html``.
The page is one self-contained HTML string: inline CSS, inline SVG built here, and a small
script for the run picker. It holds no library, font or network reference. Every number goes
through :func:`fmt`, in the tables and in the SVG tooltips alike, so a hover and its table
cell are the same string. A panel with nothing to show is left out of the page and the
contents bar; a comparison is drawn only when the inputs hold it (§12 item 4).

Choices the plan leaves open, taken here:

- Rollup sections (``rollup`` in ``profile.json``) stay out of the overview, the diagram and
  the comparison charts, as they stay out of ``share_total``; the section table keeps them.
- The diagram's blocks are the depth-0 sections; a block drawn in several lanes repeats its
  rectangle in each, and nested sub-blocks sit in its first lane.
- A run is a CPU run when ``meta.device`` is ``cpu`` or no GPU field (a ``g_*`` utilization,
  a GPU clock or GPU-lane work) appears in it.
- Hover is the SVG ``<title>`` tooltip; the script only drives the run picker.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evaluation.results.profile import io

__all__ = ["fmt", "build", "main"]

TOTALS = ("prover", "verifier", "step")
UTIL_META = ("n_samples", "few_samples", "truncated")
C5_PHASES = ("glue", "host_stack", "copy", "measure", "judge")
STRIP_LABELS = {"g_util": "GPU use (0-1)", "c_busy": "busy CPU cores",
                "pcie": "PCIe bytes/s (tx, rx)"}
STRIP_SHORT = {"g_util": "GPU use", "c_busy": "CPU busy", "pcie": "PCIe"}
GPU_LANES = ("GPU compute", "GPU memory (HBM)", "PCIe", "CPU threads", "Host")
CPU_LANES = ("CPU threads", "Host")

CSS = """
:root{--bg:#fff;--fg:#1d2330;--mute:#5d6678;--line:#d3d8e2;--panel:#f5f7fa;--bar:#1d2330;
--barfg:#fff;--accent:#2457c5;--mark:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e4e8f0;--mute:#98a2b6;--line:#343b49;
--panel:#1c2029;--bar:#0b0d11;--barfg:#e4e8f0;--accent:#8ab0ff;--mark:#fb923c}}
*{box-sizing:border-box}
body{margin:0;padding:64px 16px 48px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto}
nav{position:fixed;top:0;left:0;right:0;z-index:5;background:var(--bar);color:var(--barfg);
padding:8px 16px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;font-size:13px}
nav a{color:var(--barfg);text-decoration:none;opacity:.85}nav a:hover{opacity:1;
text-decoration:underline}
nav .sep{opacity:.4}
section{margin:0 0 28px;scroll-margin-top:56px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:24px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{font-size:15px;margin:16px 0 6px}
.run-head{font-size:19px;margin:24px 0 4px}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0}
th,td{border:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
th{background:var(--panel)}td.l,th.l{text-align:left}
.scroll{overflow-x:auto;max-width:100%}
.badge{display:inline-block;background:var(--mark);color:#fff;border-radius:3px;padding:0 5px;
margin-left:4px;font-size:11px}
.warn{color:var(--mark)}.mute{color:var(--mute)}
.legend{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:12.5px;margin:6px 0}
.sw{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:-1px}
svg{display:block;max-width:100%;height:auto}
svg text{fill:var(--fg);font-size:11px}svg .mute{fill:var(--mute)}
svg .axis{stroke:var(--line);fill:none}
svg rect:hover,svg polyline:hover{stroke:var(--fg);stroke-width:1.5}
select{font:inherit;padding:2px 6px}
"""

JS = """
const sel=document.getElementById('picker');
function show(i){document.querySelectorAll('[data-run]').forEach(function(e){
e.style.display=e.dataset.run===String(i)?'':'none'})}
if(sel){sel.addEventListener('change',function(){show(sel.value)});show(sel.value)}
"""


class Raw(str):
    """Text that is already HTML."""


def esc(x: Any) -> str:
    return x if isinstance(x, Raw) else html.escape(str(x), quote=True)


def fmt(v: Any) -> str:
    """The one number formatter: null as a dash, 3-4 significant figures."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v) if abs(v) < 10 ** 6 else f"{v:.4g}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _marks(marks: Sequence[str] | None) -> str:
    return ", ".join(marks) if marks else "—"


def _color(i: int) -> str:
    return f"hsl({(i * 137.5) % 360:.0f},55%,55%)"


def _lightness(share: float | None) -> int:
    t = min(max(share or 0.0, 0.0) / 0.5, 1.0)
    return round(94 - 56 * t)


def _fill(share: float | None) -> str:
    """One hue: light at share 0, dark from 0.5 up."""
    return f"hsl(215,70%,{_lightness(share)}%)"


def _ink(share: float | None) -> str:
    """The label color that reads on :func:`_fill`."""
    return "#111" if _lightness(share) >= 60 else "#fff"


def _table(head: Sequence[Any], rows: Sequence[Sequence[Any]], left: Sequence[int] = (0,)) -> str:
    h = "".join(f'<th class="l">{esc(c)}</th>' if i in left else f"<th>{esc(c)}</th>"
                for i, c in enumerate(head))
    body = "".join("<tr>" + "".join(
        f'<td class="l">{esc(c)}</td>' if i in left else f"<td>{esc(c)}</td>"
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{h}</tr>{body}</table></div>'


def _kv(rows: Sequence[tuple[str, Any]]) -> str:
    return _table(["key", "value"], [(k, fmt(v)) for k, v in rows], left=(0, 1))


def _flat(d: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Nested dicts as dotted keys; a list stays one value, written as JSON."""
    out: list[tuple[str, Any]] = []
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out += _flat(v, key + ".")
        elif isinstance(v, (list, dict)):
            out.append((key, json.dumps(v) if v else None))
        else:
            out.append((key, v))
    return out


def _legend(items: Sequence[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{esc(n)}</span>'
        for n, c in items) + "</div>"


def _svg(w: float, h: float, body: str) -> str:
    return f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}">{body}</svg>'


def _rect(x: float, y: float, w: float, h: float, fill: str, tip: str, stroke: str = "none",
          sw: float = 1) -> str:
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.0):.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{esc(tip)}</title></rect>')


def _text(x: float, y: float, s: Any, cls: str = "", anchor: str = "start", extra: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{c}{extra}>{esc(s)}</text>'


def _axis(x0: float, x1: float, y: float, top: float, span: float, ticks: int = 5) -> str:
    out = [f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>']
    for k in range(ticks + 1):
        x = x0 + (x1 - x0) * k / ticks
        out.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{y + 3}"/>')
        out.append(_text(x, y + 14, fmt(span * k / ticks), "mute", "middle"))
    return "".join(out)


# ---- runs --------------------------------------------------------------------------------


class Run:
    def __init__(self, idx: int, file: int, meta: dict[str, Any], name: str, view: dict[str, Any],
                 profile: dict[str, Any]):
        self.idx, self.file, self.meta, self.name, self.view = idx, file, meta, name, view
        self.profile = profile

    def field(self, key: str) -> str:
        v = self.meta.get(key)
        return "—" if v is None else str(v)

    @property
    def label(self) -> str:
        return " · ".join(self.field(k) for k in ("model", "attack", "scenario", "session"))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.field("model"), self.field("attack"), self.field("scenario"))

    @property
    def sections(self) -> dict[str, Any]:
        return self.view.get("sections") or {}

    def top(self) -> dict[str, Any]:
        """Depth-0 sections that count: no totals, no rollups."""
        return {n: s for n, s in self.sections.items()
                if s.get("depth") == 0 and not s.get("rollup") and n not in TOTALS}

    def is_cpu(self) -> bool:
        if self.meta.get("device") == "cpu" or self.view.get("device") == "cpu":
            return True
        for s in self.sections.values():
            if any(k.startswith("g_") and v is not None for k, v in (s.get("util") or {}).items()):
                return False
            if (s.get("clocks") or {}).get("gpu_s") is not None:
                return False
            b = (s.get("work") or {}).get("bytes") or {}
            if any(b.get(k) for k in ("hbm", "h2d", "d2h")):
                return False
        return True


def _runs(profiles: Sequence[dict[str, Any]]) -> list[Run]:
    out: list[Run] = []
    for i, p in enumerate(profiles):
        for name, view in (p.get("runs") or {}).items():
            out.append(Run(len(out), i, p.get("meta") or {}, name, view, p))
    return out


def _distinct(xs: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(xs))


# ---- panels ------------------------------------------------------------------------------


def _inputs(profiles: Sequence[dict[str, Any]], names: Sequence[str], runs: list[Run]) -> str:
    rows = []
    for i, (p, n) in enumerate(zip(profiles, names, strict=True)):
        mine = [r for r in runs if r.file == i]
        rows.append([n, "; ".join(r.label for r in mine) or "—",
                     ", ".join(_distinct([r.field("model") for r in mine])) or "—",
                     ", ".join(_distinct([r.field("session") for r in mine])) or "—"])
    models = _distinct([r.field("model") for r in runs])
    sessions = _distinct([r.field("session") for r in runs])
    out = [f"<p>{len(runs)} run(s) in {len(profiles)} file(s). Models: "
           f"{esc(', '.join(models) or '—')}. Sessions: {esc(', '.join(sessions) or '—')}.</p>",
           _table(["file", "runs", "models", "sessions"], rows, left=(0, 1, 2, 3))]
    for p, n in zip(profiles, names, strict=True):
        if p.get("warnings"):
            out.append(f"<h3>Warnings of {esc(n)}</h3><ul class=\"warn\">"
                       + "".join(f"<li>{esc(w)}</li>" for w in p["warnings"]) + "</ul>")
    return "".join(out)


def _section_colors(secs: Sequence[str]) -> dict[str, str]:
    return {n: _color(i) for i, n in enumerate(secs)}


def _overview(run: Run) -> str:
    v = run.view
    top = {n: s for n, s in run.top().items() if s.get("share") is not None}
    colors = _section_colors(list(top))
    facts = [("step_s", v.get("step_s")), ("n_steps", v.get("n_steps")),
             ("device", v.get("device")), ("compute_dtype", run.profile.get("compute_dtype")),
             ("share_total", v.get("share_total")),
             ("hook_overhead_bound", v.get("hook_overhead_bound"))]
    w, x = 1000.0, 0.0
    total = sum(s["share"] for s in top.values()) or 1.0
    segs = []
    for n, s in top.items():
        sw = w * s["share"] / total
        t = s.get("time") or {}
        tip = (f"{n}\nmean {fmt(t.get('mean'))} s\nshare {fmt(s['share'])}\n"
               f"marks {_marks(s.get('marks'))}")
        segs.append(_rect(x, 4, sw, 30, colors[n], tip,
                          "var(--mark)" if s.get("marks") else "none", 2.5))
        x += sw
    rows = []
    for n, s in top.items():
        badges = "".join(f'<span class="badge">{esc(m)}</span>' for m in s.get("marks") or [])
        rows.append([Raw(f'<span class="sw" style="background:{colors[n]}"></span>{esc(n)}'
                         f'{badges}'),
                     fmt((s.get("time") or {}).get("mean")), fmt(s["share"])])
    return (_kv(facts) + _svg(w, 38, "".join(segs))
            + _table(["section", "mean time (s)", "share"], rows, left=(0,)))


def _lanes_of(sec: dict[str, Any], cpu: bool) -> list[str]:
    w = sec.get("work") or {}
    b = w.get("bytes") or {}
    lanes = []
    if w.get("flops"):
        lanes.append("CPU threads" if cpu else "GPU compute")
    if not cpu:
        if b.get("hbm"):
            lanes.append("GPU memory (HBM)")
        if b.get("h2d") or b.get("d2h"):
            lanes.append("PCIe")
    if (b.get("hash") or w.get("leaves")) and "CPU threads" not in lanes:
        lanes.append("CPU threads")
    return lanes or ["Host"]


def _arch(run: Run) -> str:
    cpu = run.is_cpu()
    lanes = CPU_LANES if cpu else GPU_LANES
    secs = run.sections
    blocks = [n for n in run.top()]
    prover = [n for n in blocks if n.startswith(("P", "train."))]
    verifier = [n for n in blocks if n not in prover]
    if not prover and not verifier:
        return ""
    c5 = run.view.get("c5") or {}
    c5_phases = {p: (c5.get("phases_s") or {}).get(p) or 0.0 for p in C5_PHASES}
    step_s = run.view.get("step_s")
    left, lane_h, head_h, pad, gap = 130.0, 44.0, 26.0, 5.0, 70.0

    def width(n: str) -> float:
        if n == "5" and sum(c5_phases.values()) > 0:
            return 230.0
        return 80.0 if n == "6b" and "6b.glue" in secs else 34.0

    def sub_share(seconds: float) -> float | None:
        return seconds / step_s if step_s else None

    x = left
    out: list[str] = []
    pos: dict[str, float] = {}
    spans = []
    for group, names in (("Prover", prover), ("Verifier", verifier)):
        start = x
        for n in names:
            pos[n] = x
            x += width(n) + pad
        spans.append((group, start, x - pad, bool(names)))
        x += gap
    total_w = x - gap + 130
    h = head_h + lane_h * len(lanes) + 110
    for k, lane in enumerate(lanes):
        y = head_h + k * lane_h
        out.append(f'<rect x="0" y="{y}" width="{total_w:.0f}" height="{lane_h}" fill="none" '
                   f'stroke="var(--line)"/>')
        out.append(_text(6, y + lane_h / 2 + 4, lane, "mute"))
    for group, a, b, present in spans:
        if present:
            out.append(_text((a + b) / 2, 15, group, "", "middle", ' style="font-weight:600"'))
            out.append(f'<line class="axis" x1="{a:.1f}" x2="{b:.1f}" y1="20" y2="20"/>')
    if prover and verifier:
        a, b = spans[0][2], spans[1][1]
        y = head_h + lane_h * len(lanes) / 2
        out.append(f'<line x1="{a + 8:.1f}" x2="{b - 8:.1f}" y1="{y}" y2="{y}" '
                   f'stroke="var(--fg)" stroke-width="2"/>')
        out.append(f'<polygon points="{b - 8:.1f},{y - 6} {b - 8:.1f},{y + 6} {b + 2:.1f},{y}" '
                   f'fill="var(--fg)"/>')
        out.append(_text((a + b) / 2, y - 10, "handoff", "mute", "middle"))
    for n in prover + verifier:
        s = secs[n]
        bw, bx = width(n), pos[n]
        t = s.get("time") or {}
        used = _lanes_of(s, cpu)
        tip = (f"{n}\nmean {fmt(t.get('mean'))} s\nshare {fmt(s.get('share'))}\n"
               f"lanes {', '.join(used)}\nmarks {_marks(s.get('marks'))}")
        marked = bool(s.get("marks"))
        for ln in used:
            y = head_h + lanes.index(ln) * lane_h + 4
            out.append(_rect(bx, y, bw, lane_h - 8, _fill(s.get("share")), tip,
                             "var(--mark)" if marked else "var(--line)", 3 if marked else 1))
        y0 = head_h + lanes.index(used[0]) * lane_h + 8
        subs: list[tuple[str, float, float | None]] = []
        if n == "5" and sum(c5_phases.values()) > 0:
            subs = [(f"5.{p}", c5_phases[p], sub_share(c5_phases[p])) for p in C5_PHASES]
            tot = sum(c5_phases.values())
            sx = bx + 2
            for name, secs_, share in subs:
                sw = (bw - 4) * secs_ / tot
                if sw > 0:
                    out.append(_rect(sx, y0, sw, lane_h - 16, _fill(share),
                                     f"{name}\n{fmt(secs_)} s\nshare {fmt(share)}", "var(--fg)"))
                    if sw >= 38:
                        out.append(_text(sx + sw / 2, y0 + (lane_h - 16) / 2 + 4,
                                         name.split(".")[1][:7], "", "middle",
                                         f' style="fill:{_ink(share)}"'))
                sx += sw
        elif n == "6b" and "6b.glue" in secs:
            g = (secs["6b.glue"].get("time") or {}).get("mean") or 0.0
            tot_t = t.get("mean") or 0.0
            gs = secs["6b.glue"].get("share")
            if g > 0 and tot_t > 0:
                out.append(_rect(bx + 2, y0, (bw - 4) * min(g / tot_t, 1.0), lane_h - 16,
                                 _fill(gs), f"6b.glue\n{fmt(g)} s\nshare {fmt(gs)}", "var(--fg)"))
        ly = head_h + lane_h * len(lanes) + 10
        out.append(_text(bx + bw / 2, ly, n, "", "start",
                         f' transform="rotate(55 {bx + bw / 2:.1f} {ly})"'))
    cap = ""
    if "5" in secs and sum(c5_phases.values()) > 0:
        cap = ("<p class=\"mute\">Check 5 phases (s): " + esc(", ".join(
            f"{p} {fmt(c5_phases[p])}" for p in C5_PHASES)) + ". The run time of check 5 is the "
            "whole run, not a phase, so it has no sub-block.</p>")
    return (_svg(total_w, h, "".join(out)) + cap
            + '<p class="mute">Fill: share of the step, light at 0 to dark from 0.5. '
              'Thick outline: the section carries a mark. Nested blocks: check 5\'s phases and '
              'check 6b\'s glue.</p>')


def _sections_table(run: Run) -> str:
    secs = run.sections
    if not secs:
        return ""
    fields = _distinct([k for s in secs.values() for k in (s.get("util") or {})
                        if k not in UTIL_META])
    head = (["section", "depth", "mean s", "median s", "min s", "max s", "n", "share",
             "host s", "gpu s", "trace s", "agree"] + [f"util {f}" for f in fields]
            + ["n_samples", "few_samples", "flops", "hbm B", "h2d B", "d2h B", "hash B",
               "leaves", "launches", "syncs", "floors s", "floor s", "bound", "ratio", "marks"])
    rows = []
    for n, s in secs.items():
        t, c, u = s.get("time") or {}, s.get("clocks") or {}, s.get("util") or {}
        w = s.get("work") or {}
        b = w.get("bytes") or {}
        tag = ' <span class="mute">rollup</span>' if s.get("rollup") else ""
        name = Raw(f'<span style="padding-left:{(s.get("depth") or 0) * 14}px">{esc(n)}</span>'
                   f'{tag}')
        rows.append([
            name, fmt(s.get("depth")), fmt(t.get("mean")), fmt(t.get("median")), fmt(t.get("min")),
            fmt(t.get("max")), fmt(t.get("n")), fmt(s.get("share")), fmt(c.get("host_s")),
            fmt(c.get("gpu_s")), fmt(c.get("trace_s")), fmt(c.get("agree")),
            *[fmt(u.get(f)) for f in fields], fmt(u.get("n_samples")), fmt(u.get("few_samples")),
            "; ".join(f"{k} {fmt(v)}" for k, v in (w.get("flops") or {}).items()) or "—",
            fmt(b.get("hbm")), fmt(b.get("h2d")), fmt(b.get("d2h")), fmt(b.get("hash")),
            fmt(w.get("leaves")), fmt(w.get("launches")), fmt(w.get("syncs")),
            "; ".join(f"{k} {fmt(v)}" for k, v in (s.get("floors_s") or {}).items()) or "—",
            fmt(s.get("floor_s")), fmt(s.get("bound")), fmt(s.get("ratio")),
            _marks(s.get("marks"))])
    return _table(head, rows, left=(0, len(head) - 1, len(head) - 3))


def _polylines(ts: list[float], vals: list[float | None], x: Callable[[float], float],
               y: Callable[[float], float], color: str, name: str) -> str:
    runs: list[list[tuple[float, float]]] = [[]]
    for t, v in zip(ts, vals, strict=True):
        if v is None:
            runs.append([])
        else:
            runs[-1].append((t, v))
    present = [v for v in vals if v is not None]
    tip = f"{name}\nmean {fmt(sum(present) / len(present))}\nmax {fmt(max(present))}\n" \
          f"n {fmt(len(present))}" if present else name
    return "".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="1.2" points="'
        + " ".join(f"{x(t) + dx:.1f},{y(v):.1f}" for t, v in r for dx in
                   ((0.0, 2.0) if len(r) == 1 else (0.0,)))
        + f'"><title>{esc(tip)}</title></polyline>' for r in runs if len(r) >= 1)


def _timeline(run: Run) -> str:
    steps = (run.view.get("timeline") or {}).get("steps") or []
    if not steps:
        return ""
    axis_s = max((s.get("end_s") or 0.0) for s in steps) or 1.0
    names = _distinct([sec["name"] for s in steps for sec in s["sections"] if sec["depth"] == 0
                       and not (run.sections.get(sec["name"]) or {}).get("rollup")])
    colors = _section_colors(names)
    left, pw, span_h, strip_h = 70.0, 900.0, 16.0, 20.0

    def strips_of(s: dict[str, Any]) -> dict[str, dict[str, Any]]:
        st = s.get("strips") or {}
        out: dict[str, Any] = {}
        for k in ("g_util", "c_busy"):
            if k in st:
                out[k] = {k: st[k]}
        pc = {k: st[k] for k in ("g_pcie_tx", "g_pcie_rx") if k in st}
        if pc:
            out["pcie"] = pc
        return out

    maxes: dict[str, float] = {"g_util": 1.0}
    for s in steps:
        for key, cols in strips_of(s).items():
            if key != "g_util":
                m = max((v for c in cols.values() for v in c if v is not None), default=0.0)
                maxes[key] = max(maxes.get(key, 0.0), m)

    def x(t: float) -> float:
        return left + pw * min(t / axis_s, 1.0)

    out: list[str] = []
    y = 4.0
    for s in steps:
        out.append(_text(4, y + 12, f"step {s['step']}"))
        for sec in s["sections"]:
            if sec["depth"] != 0 or sec["name"] not in colors:
                continue
            for a, b in sec["spans_s"]:
                tip = f"{sec['name']}\nstep {s['step']}\nstart {fmt(a)} s\nend {fmt(b)} s"
                out.append(_rect(x(a), y, max(x(b) - x(a), 0.6), span_h, colors[sec["name"]], tip))
        y += span_h + 4
        strips = strips_of(s)
        for key, cols in strips.items():
            top = maxes[key] or 1.0
            out.append(f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y + strip_h}" '
                       f'y2="{y + strip_h}"/>')
            out.append(_text(left - 4, y + 10, STRIP_SHORT[key], "mute", "end"))
            for j, (fld, vals) in enumerate(cols.items()):
                col = "var(--accent)" if j == 0 else "var(--mark)"
                out.append(_polylines(s["strips"]["t_s"], vals, x,
                                      lambda v, y=y, top=top: y + strip_h - strip_h * v / top,
                                      col, f"{fld}\nstep {s['step']}"))
            y += strip_h + 2
        y += 10
    out.append(_axis(left, left + pw, y, 0, axis_s))
    used = [k for k in STRIP_LABELS if k in maxes and any(k in strips_of(s) for s in steps)]
    note = ("Strips: " + ", ".join(f"{STRIP_LABELS[k]} up to {fmt(maxes[k])}" for k in used) + "."
            if used else "No sampler data for this run.")
    return (_legend([(n, colors[n]) for n in names]) + _svg(left + pw + 20, y + 24, "".join(out))
            + f'<p class="mute">Seconds from each step\'s start. {esc(note)}</p>')


MAX_BARS = 300  # runs-over-time bars per step; more runs are binned


def _bins(n: int) -> list[range]:
    """Consecutive index ranges of equal size (the last may be shorter), at most MAX_BARS."""
    size = -(-n // MAX_BARS)
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def _bin_peak(over: dict[str, Any]) -> float | None:
    """The largest bin's seconds when any step has more than MAX_BARS runs, else ``None``."""
    if all(len(st.get("run") or []) <= MAX_BARS for st in over.values()):
        return None
    return max((sum(st["dur_s"][i] for i in b) for st in over.values()
                for b in _bins(len(st["run"])) if st.get("run")), default=0.0) or 1.0


def _run_bars(st: dict[str, Any], ccol: dict[str, str], left: float, pw: float, axis_s: float,
              y: float, peak: float | None) -> str:
    """One bar per run, or when ``peak`` is set, one bar per bin of consecutive runs: it spans
    the bin's first start to last end, and its height is the bin's seconds on a scale to peak."""
    runs, cls, start, dur = st.get("run") or [], st.get("class") or [], st["start_s"], st["dur_s"]
    if not runs:
        return ""
    out = []
    if peak is None:
        for r, c, a, d in zip(runs, cls, start, dur, strict=True):
            out.append(_rect(left + pw * a / axis_s, y, max(pw * d / axis_s, 0.6), 16,
                             ccol.get(c, "gray"),
                             f"run {r}\nclass {c if c else '—'}\nstart {fmt(a)} s\n"
                             f"duration {fmt(d)} s"))
        return "".join(out)
    for b in _bins(len(runs)):
        i, j = b[0], b[-1]
        secs = sum(dur[k] for k in b)
        classes = _distinct([cls[k] or "—" for k in b])
        end = max(start[k] + dur[k] for k in b)
        h = max(40 * secs / peak, 1.0)
        out.append(_rect(left + pw * start[i] / axis_s, y + 40 - h,
                         max(pw * (end - start[i]) / axis_s, 0.6), h,
                         ccol.get(classes[0], "gray") if len(classes) == 1 else "var(--mute)",
                         f"runs {runs[i]}-{runs[j]}\nclass {', '.join(classes)}\n"
                         f"start {fmt(start[i])} s\nseconds {fmt(secs)} s"))
    return "".join(out)


def _c5(run: Run) -> str:
    c5 = run.view.get("c5")
    if not c5:
        return ""
    classes = c5.get("classes") or {}
    cols = (["class", "n_runs", "n_groups", "product_bytes", "served_bytes"]
            + [f"{p} s" for p in ("run", *C5_PHASES)] + ["stored B/s", "served B/s"])
    rows = []
    for name, c in classes.items():
        ph, cp = c.get("phases_s") or {}, c.get("copy_bytes_per_s") or {}
        rows.append([name, fmt(c.get("n_runs")), fmt(c.get("n_groups")),
                     fmt(c.get("product_bytes")), fmt(c.get("served_bytes")),
                     *[fmt(ph.get(p)) for p in ("run", *C5_PHASES)],
                     fmt(cp.get("stored")), fmt(cp.get("served"))])
    out = [_table(cols, rows)] if rows else []
    if rows:
        out.append('<p class="mute">run s counts each run\'s whole time against the class of '
                   "the run's first group; classes served inside another class's run "
                   "(e.g. O inside S) show 0 there.</p>")
    pcol = {p: _color(i) for i, p in enumerate(C5_PHASES)}
    if classes:
        mx = max(sum((c.get("phases_s") or {}).get(p) or 0.0 for p in C5_PHASES)
                 for c in classes.values()) or 1.0
        body, y = [], 4.0
        for name, c in classes.items():
            body.append(_text(4, y + 12, name))
            xx = 150.0
            for p in C5_PHASES:
                v = (c.get("phases_s") or {}).get(p) or 0.0
                w = 700 * v / mx
                body.append(_rect(xx, y, w, 16, pcol[p], f"{name}\n{p} {fmt(v)} s"))
                xx += w
            y += 22
        out.append("<h3>Phase time by class</h3>" + _legend(list(pcol.items()))
                   + _svg(900, y + 6, "".join(body)))
    over = c5.get("runs_over_time") or {}
    if over:
        names = sorted({c for st in over.values() for c in st.get("class") or [] if c})
        ccol = {c: _color(i) for i, c in enumerate(names)}
        axis_s = max((max((a + d for a, d in zip(st.get("start_s") or [],
                                                 st.get("dur_s") or [], strict=True)), default=0.0)
                      for st in over.values()), default=0.0) or 1.0
        left, pw = 70.0, 900.0
        body, y = [], 4.0
        peak = _bin_peak(over)
        for step, st in over.items():
            body.append(_text(4, y + 12, f"step {step}"))
            body.append(_run_bars(st, ccol, left, pw, axis_s, y, peak))
            y += 22 if peak is None else 46
        body.append(_axis(left, left + pw, y + 2, 0, axis_s))
        out.append("<h3>Runs over time</h3>" + _legend(list(ccol.items()))
                   + _svg(left + pw + 20, y + 26, "".join(body)))
    var = c5.get("variance") or {}
    fields = _distinct([f for s in (var.get("steps") or []) for f in s.get("sampler") or {}])
    for key, label in (("steps", "Per step"), ("windows", "Per window")):
        tab = var.get(key) or []
        if tab:
            head = (["step"] + (["window"] if key == "windows" else [])
                    + [f"{p} s" for p in ("run", *C5_PHASES)] + fields)
            rows = [[fmt(t.get("step"))] + ([fmt(t.get("window"))] if key == "windows" else [])
                    + [fmt((t.get("phases_s") or {}).get(p)) for p in ("run", *C5_PHASES)]
                    + [fmt((t.get("sampler") or {}).get(f)) for f in fields] for t in tab]
            out.append(f"<h3>Variance: {label.lower()}</h3>" + _table(head, rows, left=()))
    for key, label in (("across_steps", "across steps"), ("across_windows", "across windows")):
        cor = (var.get("correlation") or {}).get(key) or {}
        cf = _distinct([f for d in cor.values() for f in d])
        if cor and cf:
            out.append(f"<h3>Correlation of phase time with sampler fields, {label}</h3>"
                       + _table(["phase", *cf], [[p, *[fmt(d.get(f)) for f in cf]]
                                                 for p, d in cor.items()]))
    return "".join(out)


# ---- comparisons -------------------------------------------------------------------------


def _bars(rows: list[tuple[str, list[tuple[str, float | None]]]], colors: dict[str, str]) -> str:
    """One stacked bar per row of (section, seconds), on one scale."""
    mx = max((sum(v or 0.0 for _, v in r) for _, r in rows), default=0.0) or 1.0
    body, y = [], 4.0
    for label, parts in rows:
        body.append(_text(4, y + 12, label))
        x = 330.0
        for n, v in parts:
            if v:
                w = 600 * v / mx
                body.append(_rect(x, y, w, 16, colors[n], f"{label}\n{n} {fmt(v)} s"))
                x += w
        y += 22
    return _svg(960, y + 6, "".join(body))


def _compare(rows: list[tuple[str, Run]]) -> str:
    """A table and a bar chart: step_s and every depth-0 section's mean, one row per run;
    a section a run lacks is blank."""
    names = _distinct([n for _, r in rows for n in r.top()])
    colors = _section_colors(names)

    def mean(r: Run, n: str) -> float | None:
        return ((r.sections.get(n) or {}).get("time") or {}).get("mean")

    table = [[lab, fmt(r.view.get("step_s")), *[fmt(mean(r, n)) if n in r.top() else ""
                                                for n in names]] for lab, r in rows]
    bars = [(lab, [(n, mean(r, n)) for n in names if n in r.top()]) for lab, r in rows]
    return (_table(["run", "step s", *names], table) + _legend([(n, colors[n]) for n in names])
            + _bars(bars, colors))


def _cross_model(runs: list[Run]) -> str:
    out = []
    for session in _distinct([r.field("session") for r in runs]):
        mine = [r for r in runs if r.field("session") == session]
        if len({r.field("model") for r in mine}) >= 2:
            out.append(f"<h3>Session {esc(session)}</h3>"
                       + _compare([(f"{r.field('model')} · {r.field('attack')}", r)
                                   for r in mine]))
    return "".join(out)


def _cross_session(runs: list[Run]) -> str:
    if len({r.field("session") for r in runs}) < 2:
        return ""
    keys: dict[tuple[str, str, str], list[Run]] = {}
    for r in runs:
        keys.setdefault(r.key, []).append(r)
    rows = [(r.label, r) for ks in keys.values() if len({r.field("session") for r in ks}) >= 2
            for r in ks]
    return _compare(rows) if rows else ""


def _plain(runs: list[Run]) -> str:
    out = []
    for r in runs:
        cmp = r.profile.get("comparisons")
        if not cmp:
            continue
        rows = [[n, c.get("plain_name"), fmt(c.get("prover_s")), fmt(c.get("plain_s")),
                 fmt(c.get("ratio"))] for n, c in (cmp.get("sections") or {}).items()]
        st = cmp.get("step") or {}
        pr, pl = st.get("prover_s"), st.get("plain_s")
        rows.append(["step", "step", fmt(pr), fmt(pl), fmt(pr / pl if pr and pl else None)])
        out.append(f"<h3>{esc(r.label)}</h3>"
                   + _table(["section", "plain name", "prover s", "plain s", "ratio"], rows,
                            left=(0, 1)))
    return "".join(out)


def _appendix(profiles: Sequence[dict[str, Any]], names: Sequence[str]) -> str:
    out = []
    for p, n in zip(profiles, names, strict=True):
        probe, machine, th = p.get("probe") or {}, p.get("machine"), p.get("thresholds")
        parts = []
        for key in ("rates", "costs", "datasheet"):
            if probe.get(key):
                parts.append(f"<h4>Probe: {key}</h4>" + _kv(_flat(probe[key])))
        if machine:
            parts.append("<h4>Machine</h4>" + _kv(_flat(machine)))
        if th:
            parts.append("<h4>Thresholds</h4>" + _kv(_flat(th)))
        if parts:
            out.append(f"<h3>{esc(n)}</h3>" + "".join(parts))
    return "".join(out)


# ---- the page ----------------------------------------------------------------------------


def build(profiles: Sequence[dict[str, Any]], names: Sequence[str] | None = None) -> str:
    """The report page for ``profiles`` (parsed ``profile.json`` files, named by ``names``)."""
    names = list(names) if names is not None else [f"profile {i + 1}" for i in range(len(profiles))]
    runs = _runs(profiles)
    many = len(runs) >= 2
    nav: list[str] = []
    head: list[str] = []
    mid: dict[int, list[str]] = {r.idx: [] for r in runs}
    tail: list[str] = []

    def add(anchor: str, title: str, body: str, run: Run | None = None) -> None:
        if not body:
            return
        attr = f' data-run="{run.idx}"' if run is not None and many else ""
        nav.append(f'<span{attr}><a href="#{anchor}">{esc(title)}</a></span>')
        panel = f'<section id="{anchor}"><h2>{esc(title)}</h2>{body}</section>'
        if run is not None:
            mid[run.idx].append(panel)
        else:
            (head if anchor == "inputs" else tail).append(panel)

    add("inputs", "Inputs", _inputs(profiles, names, runs))
    per_run = (("overview", "Overview", _overview), ("diagram", "Architecture", _arch),
               ("sections", "Section table", _sections_table), ("timeline", "Timeline", _timeline),
               ("check5", "Check 5", _c5))
    for r in runs:
        for key, title, fn in per_run:
            add(f"r{r.idx}-{key}", title, fn(r), r)
    add("compare-models", "Across models", _cross_model(runs))
    add("compare-sessions", "Across sessions", _cross_session(runs))
    add("compare-plain", "Prover beside plain", _plain(runs))
    add("appendix", "Appendix", _appendix(profiles, names))

    picker = ""
    blocks = []
    for r in runs:
        attr = (f' data-run="{r.idx}"' + ("" if r.idx == 0 else ' style="display:none"')
                if many else "")
        blocks.append(f'<div{attr}><div class="run-head">{esc(r.label)}</div>'
                      f'{"".join(mid[r.idx])}</div>')
    if many:
        opts = "".join(f'<option value="{r.idx}">{esc(r.label)}</option>' for r in runs)
        picker = f'<p><label>Run <select id="picker">{opts}</select></label></p>'
    script = f"<script>{JS}</script>" if many else ""
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Profile report</title>"
            f"<style>{CSS}</style></head><body><nav>{''.join(nav)}</nav><main>"
            f"<h1>Profile report</h1>{''.join(head)}{picker}{''.join(blocks)}{''.join(tail)}"
            f"</main>{script}</body></html>")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render profile.json files as one HTML report.")
    p.add_argument("profiles", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    loaded = []
    for path in a.profiles:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj.get("version") != io.VERSION:
            raise ValueError(f"{path}: not a version {io.VERSION} profile.json")
        loaded.append(obj)
    a.out.write_text(build(loaded, [str(x) for x in a.profiles]), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
