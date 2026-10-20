"""FS26: the HTML report of one or more ``profile.json`` files (PROFILING_PLAN FS26, §12).

Run as ``python -m evaluation.results.profile.report PROFILE [PROFILE ...] --out profile.html``.
The page is one self-contained HTML string: inline CSS, inline SVG built here, and a small
script for the run picker. It holds no library, font or network reference. Every number goes
through :func:`fmt` (or :func:`pct`, which uses it), in the tables and in the SVG tooltips
alike, so a hover and its table cell are the same string. A panel with nothing to show is
left out of the page and the contents bar; a comparison is drawn only when the inputs hold it
(§12 item 4).

The page is a summary for a reader, not a dump: every measure stays in the ``profile.json``
files, and the page shows only the step's sections, their times and shares, their marks, the
hardware they kept busy, the timeline, check 5's phases and the comparisons (§12 item 8, user,
2026-10-20). Choices taken here:

- Sections carry a plain name and a one-line description (``SECTIONS``) and fall into seven
  groups (``GROUPS``), one color each; a section is a tint of its group's color. The
  hand-off is hatched, so it needs no eighth hue. A section the catalogue lacks is gray and
  keeps its code.
- Only depth-0 sections are drawn. Rollups (``rollup`` in ``profile.json``) stay out, as they
  stay out of ``share_total``; nested sections stay in the file.
- The step-flow diagram is one box per depth-0 section in run order, in three bands (prover,
  hand-off, verifier), with the sampler's GPU busy fraction and the run's CPU cores.
- Legends carry plain names only; the overview's table, the flow's boxes and check 5's rows keep
  the code beside the name, so a reader can find the section in ``profile.json``.
- Prover beside plain is a per-run panel after check 5, so the run picker shows the chosen
  run's table alone.
- A run is a CPU run when ``meta.device`` is ``cpu`` or no GPU field (a ``g_*`` utilization,
  a GPU clock or GPU-lane work) appears in it; it draws no GPU gauge or strip.
- Hover is the SVG ``<title>`` tooltip; the script only drives the run picker.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evaluation.results.profile import io

__all__ = ["fmt", "pct", "build", "main"]

TOTALS = ("prover", "verifier", "step")
C5_PHASES = ("glue", "host_stack", "copy", "measure", "judge")

# group: (part of the step, title)
GROUPS: dict[str, tuple[str, str]] = {
    "train": ("Prover", "Prover · training, with matmul capture"),
    "record": ("Prover", "Prover · recording the transcript"),
    "commit": ("Prover", "Prover · committing to the transcript"),
    "hand": ("Hand-off", "Hand-off between prover and verifier"),
    "read": ("Verifier", "Verifier · reading and hashing the transcript"),
    "update": ("Verifier", "Verifier · checking the weight update"),
    "matmul": ("Verifier", "Verifier · checking the matmuls"),
    "other": ("Other", "Other sections"),
}
PARTS = {"Prover": "trains the step with every matmul captured, then records and commits "
                   "the transcript",
         "Hand-off": "passes the transcript from the prover to the verifier",
         "Verifier": "checks the step from the transcript",
         "Other": "sections this report has no name for"}

# section: (group, shade 0-3, short name, what it does)
SECTIONS: dict[str, tuple[str, int, str, str]] = {
    "train.load": ("train", 2, "Load weights",
                   "load W_t into the model and clear the gradients"),
    "train.forward": ("train", 0, "Forward pass",
                      "the training forward pass, every matmul captured"),
    "train.backward": ("train", 1, "Backward pass",
                       "the training backward pass, every matmul captured"),
    "train.update": ("train", 3, "Weight update", "the SGD step from W_t to W_t+1"),
    "P1.label": ("record", 0, "Label matmuls",
                 "name each captured matmul and keep its inputs and product"),
    "P2.w_t": ("record", 2, "Copy W_t", "copy the step's starting weights for the transcript"),
    "P2.w_next": ("record", 1, "Copy W_t+1", "copy the updated weights for the transcript"),
    "P4.paths": ("commit", 2, "Batch paths",
                 "Merkle paths from the batch's records to the dataset root"),
    "P3.commit": ("commit", 0, "Merkle commitment",
                  "hash every transcript leaf and build the Merkle tree"),
    "P5.write": ("commit", 1, "Hand over transcript",
                 "hand the transcript to the store the verifier reads"),
    "handoff.totals": ("hand", 0, "Count transcript bytes",
                       "add up the transcript's size for the run log"),
    "handoff.empty_cache": ("hand", 1, "Free GPU cache",
                            "release the prover's cached GPU memory before the verifier"),
    "4": ("read", 2, "Check 4: batch in dataset",
          "the batch's records lead to the dataset root"),
    "7": ("read", 1, "Check 7: weights chain", "this step's W_t is the previous step's W_t+1"),
    "2": ("read", 0, "Check 2: read and hash",
          "read every transcript leaf, check its shape and hash it"),
    "2.root": ("read", 3, "Check 2: compare root",
               "the rebuilt Merkle root equals the committed one"),
    "6a": ("update", 0, "Check 6a: linear weights",
           "W_t+1 = W_t − η·G for the linear layers' weights"),
    "6b": ("update", 1, "Check 6b: other weights",
           "the same for norm scales, biases and the embedding, from replayed gradients"),
    "5": ("matmul", 0, "Check 5: matmuls", "Freivalds' check of every captured matmul"),
}

# check-5 phase: (short name, what it does)
PHASES = {"glue": ("Rebuild inputs", "replay the model to rebuild each matmul's inputs"),
          "host_stack": ("Stack in host memory", "stack the products' leaves in host RAM"),
          "copy": ("Copy to GPU", "copy the stacked products to the GPU"),
          "measure": ("Freivalds products",
                      "multiply by the random vectors and measure the residuals"),
          "judge": ("Compare with bands", "compare the residuals with the bands and decide")}

# check-5 product class: (belongs to, plain name), in the order a step runs them
_ATT, _MLP = "attention", "MLP"
CLASSES: dict[str, tuple[str, str]] = {
    "Y_q": (f"Forward · {_ATT}", "Query projection"),
    "Y_k": (f"Forward · {_ATT}", "Key projection"),
    "Y_v": (f"Forward · {_ATT}", "Value projection"),
    "S": (f"Forward · {_ATT}", "Attention scores Q·Kᵀ"),
    "O": (f"Forward · {_ATT}", "Attention mix A·V"),
    "Y_o": (f"Forward · {_ATT}", "Attention output projection"),
    "Y_gate": (f"Forward · {_MLP}", "Gate projection"),
    "Y_up": (f"Forward · {_MLP}", "Up projection"),
    "Y_down": (f"Forward · {_MLP}", "Down projection"),
    "Lambda": ("Forward · output layer", "Output logits"),
    "dF": ("Backward · output layer", "Gradient into the last hidden state"),
    "G_E_head": ("Backward · output layer", "Output weight gradient (tied)"),
    "G_head": ("Backward · output layer", "Output weight gradient"),
    **{f"{k}_{x}": (f"Backward · {_MLP}", f"{what}, {x} projection")
       for x in ("down", "up", "gate") for k, what in (("dX", "Input gradient"),
                                                       ("G", "Weight gradient"))},
    "dX_o": (f"Backward · {_ATT}", "Input gradient, output projection"),
    "G_o": (f"Backward · {_ATT}", "Weight gradient, output projection"),
    "dA": (f"Backward · {_ATT}", "Gradient of the attention weights"),
    "dV": (f"Backward · {_ATT}", "Gradient of the values"),
    "dQ": (f"Backward · {_ATT}", "Gradient of the queries"),
    "dK": (f"Backward · {_ATT}", "Gradient of the keys"),
    **{f"{k}_{x}": (f"Backward · {_ATT}", f"{what}, {name} projection")
       for x, name in (("q", "query"), ("k", "key"), ("v", "value"))
       for k, what in (("dX", "Input gradient"), ("G", "Weight gradient"))},
}

MARKS = {
    "bottleneck": "takes at least {share} of the step and at least {ratio}× the time its work "
                  "needs at this machine's measured speeds: time could be won here.",
    "utilization": "the hardware use expected from the section's work and the use the sampler "
                   "measured differ by more than 2×: the work estimate or the counter is off, "
                   "so the bottleneck verdict is less certain.",
    "memory": "reserved GPU memory reached {reserved} of what the GPU can use.",
    "clock": "two clocks disagree on the section's time (host and GPU, or trace and host): "
             "the time may be off.",
}

CSS = """
:root{--bg:#fcfcfb;--fg:#1d2330;--mute:#5f6470;--line:#d9d8d3;--panel:#f3f2ee;--bar:#1d2330;
--barfg:#fff;--mark:#d03b3b;--hand:#898781;
--c-train:#2a78d6;--c-record:#eda100;--c-commit:#008300;--c-read:#1baf7a;--c-update:#6250d6;
--c-matmul:#eb6834;--c-other:#898781;
--p-glue:#2a78d6;--p-host_stack:#eb6834;--p-copy:#1baf7a;--p-measure:#eda100;--p-judge:#e87ba4}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#e8e6e1;--mute:#a3a198;
--line:#3a3935;--panel:#24241f;--bar:#0e0e0d;--barfg:#e8e6e1;--mark:#d03b3b;--hand:#a3a198;
--c-train:#3987e5;--c-record:#c98500;--c-commit:#008300;--c-read:#199e70;--c-update:#9085e9;
--c-matmul:#d95926;--c-other:#898781;
--p-glue:#3987e5;--p-host_stack:#d95926;--p-copy:#199e70;--p-measure:#c98500;--p-judge:#d55181}}
*{box-sizing:border-box}
body{margin:0;padding:64px 16px 48px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto}
nav{position:fixed;top:0;left:0;right:0;z-index:5;background:var(--bar);color:var(--barfg);
padding:8px 16px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;font-size:13px}
nav a{color:var(--barfg);text-decoration:none;opacity:.85}nav a:hover{opacity:1;
text-decoration:underline}
section{margin:0 0 28px;scroll-margin-top:56px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:24px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}h3{font-size:15px;margin:16px 0 6px}
.run-head{font-size:19px;margin:28px 0 4px;font-weight:600}
p{max-width:820px}
table{border-collapse:collapse;font-size:13px;margin:6px 0}
th,td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}
th{font-weight:600;border-bottom:2px solid var(--line)}td.l,th.l{text-align:left}
td.w{white-space:normal;min-width:220px;color:var(--mute)}
tr.part td{text-align:left;background:var(--panel);font-weight:600;white-space:normal}
tr.part .mute{font-weight:400}
.scroll{overflow-x:auto;max-width:100%}
.badge{display:inline-block;border:1px solid var(--mark);color:var(--mark);border-radius:3px;
padding:0 5px;margin-left:4px;font-size:11px;line-height:16px}
.warn{color:var(--mark)}.mute{color:var(--mute)}.small{font-size:12.5px}
.code{color:var(--mute);font-size:12px;margin-left:4px}
.legend{font-size:12.5px;margin:6px 0 4px}
.legend .grp{display:flex;flex-wrap:wrap;gap:2px 14px;margin:2px 0}
.legend .gt{min-width:270px;color:var(--mute)}
.sw{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px;
border-radius:2px}
.hatch{background:repeating-linear-gradient(45deg,var(--hand) 0 2px,transparent 2px 4px);
outline:1px solid var(--hand);outline-offset:-1px}
.share{display:inline-block;height:8px;background:var(--mute);opacity:.55;margin-left:6px;
vertical-align:0;border-radius:0 2px 2px 0}
dl.marks{font-size:12.5px;margin:6px 0;display:grid;grid-template-columns:max-content 1fr;
gap:2px 10px;max-width:820px}dl.marks dd{margin:0}
svg{display:block;max-width:100%;height:auto}
svg text{fill:var(--fg);font-size:11px}svg .mute{fill:var(--mute)}svg .b{font-weight:600}
svg .axis{stroke:var(--line);fill:none}svg .badge{fill:var(--mark);font-size:10px}
svg .seg:hover,svg polyline:hover{stroke:var(--fg);stroke-width:1.5}
select{font:inherit;padding:2px 6px}
"""

DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
        '<pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="4" height="4" fill="var(--bg)"/>'
        '<rect width="2" height="4" fill="var(--hand)"/></pattern></defs></svg>')

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
    """The one number formatter: null as a dash, two significant digits.

    Below 0.01 and from 10 000 up a number is written in scientific form (``8.4e-03``); an
    integer below 10 000 is a count and stays exact."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) and abs(v) < 10 ** 4:
        return str(v)
    if isinstance(v, (int, float)):
        x = float(f"{float(v):.2g}")
        if x == 0 or not math.isfinite(x):
            return "0" if x == 0 else str(x)
        if abs(x) < 0.01 or abs(x) >= 10 ** 4:
            return f"{x:.1e}"
        return f"{x:.{max(0, 1 - math.floor(math.log10(abs(x))))}f}"
    return str(v)


def pct(share: float | None) -> str:
    """A share as a percentage through :func:`fmt`; a nonzero one under 0.1% is ``<0.1%``."""
    if share is None:
        return "—"
    p = 100 * share
    return "<0.1%" if 0 < p < 0.1 else f"{fmt(p)}%"


def _tick(v: float) -> str:
    return f"{v:.0e}" if v and abs(v) < 0.01 else f"{v:.3g}"


def _ticks(top: float, n: int = 5) -> list[float]:
    """Round tick values from 0 to at least ``top``, in steps of 1, 2 or 5 times a power of
    ten."""
    if top <= 0:
        return [0.0, 1.0]
    raw = top / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw * (1 - 1e-9))
    return [i * step for i in range(math.ceil(top / step - 1e-9) + 1)]


def _marks(marks: Sequence[str] | None) -> str:
    return ", ".join(marks) if marks else "none"


INK_MIX = (0, 30, 50, 65)  # percent of text ink mixed into a group's color, by shade
HATCH_OPACITY = (1.0, 0.55, 0.4, 0.3)


def _sec(code: str) -> tuple[str, int, str, str]:
    return SECTIONS.get(code) or ("other", 0, code, "")


def _name(code: str) -> str:
    return _sec(code)[2]


def _color(code: str) -> str:
    """A section's group color, shaded toward the text ink, so every shade keeps its contrast
    with the page in light and dark mode alike."""
    g, shade = _sec(code)[:2]
    c = f"var(--c-{g})"
    return f"color-mix(in oklab, {c}, var(--fg) {INK_MIX[shade]}%)" if shade else c


def _paint(code: str) -> str:
    """The SVG fill of a section; the hand-off is hatched."""
    g, shade = _sec(code)[:2]
    if g == "hand":
        return f'fill="url(#hatch)" fill-opacity="{HATCH_OPACITY[shade]}"'
    return f'style="fill:{_color(code)}"'


def _swatch(code: str) -> str:
    g, shade = _sec(code)[:2]
    if g == "hand":
        return f'<span class="sw hatch" style="opacity:{HATCH_OPACITY[shade]}"></span>'
    return f'<span class="sw" style="background:{_color(code)}"></span>'


def _legend(codes: Sequence[str]) -> str:
    """The sections' swatches with plain names (no codes), one line per group."""
    by: dict[str, list[str]] = {}
    for c in codes:
        by.setdefault(_sec(c)[0], []).append(c)
    rows = []
    for g in GROUPS:
        if g in by:
            items = "".join(f"<span>{_swatch(c)}{esc(_name(c))}</span>" for c in by[g])
            rows.append(f'<div class="grp"><span class="gt">{esc(GROUPS[g][1])}</span>{items}'
                        "</div>")
    return '<div class="legend">' + "".join(rows) + "</div>"


def _phase_legend() -> str:
    items = "".join(f'<span><span class="sw" style="background:var(--p-{p})"></span>'
                    f"{esc(PHASES[p][0])}</span>"
                    for p in C5_PHASES)
    return f'<div class="legend"><div class="grp">{items}</div></div>'


def _table(head: Sequence[Any], rows: Sequence[Sequence[Any]], left: Sequence[int] = (0,)) -> str:
    h = "".join(f'<th class="l">{esc(c)}</th>' if i in left else f"<th>{esc(c)}</th>"
                for i, c in enumerate(head))
    body = "".join("<tr>" + "".join(
        f'<td class="l">{esc(c)}</td>' if i in left else f"<td>{esc(c)}</td>"
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{h}</tr>{body}</table></div>'


def _svg(w: float, h: float, body: str) -> str:
    return f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}">{body}</svg>'


def _rect(x: float, y: float, w: float, h: float, paint: str, tip: str = "", extra: str = "") -> str:
    title = f"<title>{esc(tip)}</title>" if tip else ""
    cls = ' class="seg"' if tip else ""
    return (f'<rect{cls} x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.0):.2f}" height="{h:.2f}" '
            f'{paint}{extra}>{title}</rect>')


def _text(x: float, y: float, s: Any, cls: str = "", anchor: str = "start", extra: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{c}{extra}>{esc(s)}</text>'


def _axis(x0: float, x1: float, y: float, top: float, ticks: Sequence[float], unit: str) -> str:
    """A bottom axis with grid lines up to ``top``; ``ticks`` run from 0 to the scale's end."""
    end = ticks[-1] or 1.0
    out = [f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>']
    for t in ticks:
        x = x0 + (x1 - x0) * t / end
        out.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{y + 3}"/>')
        out.append(_text(x, y + 14, _tick(t), "mute", "middle"))
    out.append(_text(x1, y + 28, unit, "mute", "end"))
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
    def model(self) -> str:
        """The model's name without its publisher (``SmolLM2-135M-Instruct``)."""
        return self.field("model").rsplit("/", 1)[-1]

    @property
    def label(self) -> str:
        return " · ".join([self.model] + [self.field(k) for k in ("attack", "scenario", "session")])

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

    def mean(self, n: str) -> float | None:
        return ((self.sections.get(n) or {}).get("time") or {}).get("mean")

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


def _part(code: str) -> str:
    return GROUPS[_sec(code)[0]][0]


def _by_part(names: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(_part(n), []).append(n)
    return out


# ---- panels ------------------------------------------------------------------------------


def _inputs(profiles: Sequence[dict[str, Any]], names: Sequence[str], runs: list[Run]) -> str:
    models = _distinct([r.model for r in runs])
    sessions = _distinct([r.field("session") for r in runs])
    dirs = _distinct([str(Path(n).parent) for n in names])
    files = (f"{', '.join(Path(n).name for n in names)} in {dirs[0]}"
             if len(dirs) == 1 and dirs[0] != "." else ", ".join(names))
    out = [f"<p>{len(runs)} run(s) from {len(profiles)} file(s). Models: "
           f"{esc(', '.join(models) or '—')}. Session(s): {esc(', '.join(sessions) or '—')}.</p>",
           f'<p class="mute small">Files: {esc(files)}. The page is a summary; '
           "every measure behind it (clocks, work counts, floors, sampler fields, hashing parts, "
           "probe rates, machine facts, thresholds) stays in these files.</p>"]
    for p, n in zip(profiles, names, strict=True):
        if p.get("warnings"):
            out.append(f"<h3>Warnings from {esc(n)}</h3><ul class=\"warn\">"
                       + "".join(f"<li>{esc(w)}</li>" for w in p["warnings"]) + "</ul>")
    return "".join(out)


def _tip(run: Run, n: str) -> str:
    s = run.sections[n]
    return (f"{_name(n)} ({n})\nmean {fmt(run.mean(n))} s\n{pct(s.get('share'))} of step time\n"
            f"marks {_marks(s.get('marks'))}")


def _marks_legend(marks: Sequence[str], th: dict[str, Any] | None) -> str:
    if not marks:
        return ""
    th = th or {}
    words = {"share": pct(th.get("share", 0.1)), "ratio": fmt(th.get("ratio", 3.0)),
             "reserved": pct(th.get("reserved", 0.95))}
    rows = "".join(f'<dt><span class="badge">{esc(m)}</span></dt>'
                   f"<dd>{esc(MARKS.get(m, '').format(**words) or '—')}</dd>" for m in marks)
    return f'<p class="small" style="margin-bottom:0">Marks:</p><dl class="marks">{rows}</dl>'


def _overview(run: Run) -> str:
    v = run.view
    top = {n: s for n, s in run.top().items() if run.mean(n) is not None}
    if not top:
        return ""
    parts = _by_part(list(top))
    sums = {p: sum(run.mean(n) or 0.0 for n in ns) for p, ns in parts.items()}
    dtype = run.profile.get("compute_dtype") or run.meta.get("compute_dtype")
    said = ", ".join(f"the {p.lower()} {fmt(sums[p])} s" for p in ("Prover", "Hand-off",
                                                                     "Verifier") if p in sums)
    intro = (f"<p>One verified training step took <b>{fmt(v.get('step_s'))} s</b> on average over "
             f"{fmt(v.get('n_steps'))} timed steps ({esc(v.get('device') or run.field('device'))}"
             f"{', ' + esc(dtype) if dtype else ''}). Of that, {esc(said)}.</p>")

    w, total = 1000.0, sum(run.mean(n) or 0.0 for n in top) or 1.0
    segs, x = [], 0.0
    spans: dict[str, list[float]] = {}
    for n in top:
        sw = w * (run.mean(n) or 0.0) / total
        segs.append(_rect(x, 22, max(sw - 2, 0.6), 28, _paint(n), _tip(run, n)))
        spans.setdefault(_part(n), [x, x])[1] = x + sw
        x += sw
    for p, (a, b) in spans.items():
        if b - a >= 70:
            segs.append(f'<line class="axis" x1="{a:.1f}" x2="{b - 2:.1f}" y1="16" y2="16"/>')
            segs.append(_text(a, 11, f"{p} {fmt(sums[p])} s", "mute"))
    bar = _svg(w, 54, "".join(segs))

    rows = []
    marks: list[str] = []
    for p, ns in parts.items():
        share = sum(top[n].get("share") or 0.0 for n in ns)
        rows.append(f'<tr class="part"><td colspan="5">{esc(p)} <span class="mute">— '
                    f"{esc(PARTS[p])} · {fmt(sums[p])} s · {pct(share)}</span></td></tr>")
        for n in ns:
            s = top[n]
            ms = s.get("marks") or []
            marks += ms
            badges = "".join(f'<span class="badge">{esc(m)}</span>' for m in ms)
            share = s.get("share")
            bar_w = round(80 * min(share or 0.0, 1.0))
            rows.append(f'<tr><td class="l">{_swatch(n)}{esc(_name(n))}'
                        f'<span class="code">{esc(n)}</span></td>'
                        f'<td class="l w">{esc(_sec(n)[3])}</td>'
                        f"<td>{fmt(run.mean(n))}</td>"
                        f'<td class="l">{pct(share)}<span class="share" '
                        f'style="width:{bar_w}px"></span></td>'
                        f'<td class="l">{badges}</td></tr>')
    head = ('<tr><th class="l">Section</th><th class="l">What it does</th>'
            '<th>Mean time (s)</th><th class="l">Share of step time</th>'
            '<th class="l">Marks</th></tr>')
    note = ('<p class="mute small">Share of step time: the section\'s mean time over the mean '
            "time of one whole verified step (prover, hand-off and verifier). The training rows "
            "are the prover's own training with the matmul capture on; plain training, without "
            "the protocol, is compared in Prover beside plain.</p>")
    return (intro + bar + f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>' + note
            + _marks_legend(_distinct(marks), run.profile.get("thresholds")))


def _cores(run: Run) -> float:
    """The CPU scale of the flow's gauges: the cores the machine allowed the run."""
    m = run.profile.get("machine") or {}
    for k in ("cgroup_cpu_quota", "usable_cpus", "logical_cpus"):
        if isinstance(m.get(k), (int, float)) and m[k] > 0:
            return float(m[k])
    seen = [(s.get("util") or {}).get("c_tree") or 0.0 for s in run.top().values()]
    return max([1.0, *seen])


def _gauge(x: float, y: float, label: str, frac: float | None, value: str) -> str:
    """A labeled meter: track, fill to ``frac`` (none when unknown) and the value."""
    out = [_text(x, y + 6, label, "mute"),
           f'<rect x="{x + 30:.1f}" y="{y}" width="70" height="7" rx="1.5" fill="var(--line)"/>']
    if frac is not None:
        out.append(f'<rect x="{x + 30:.1f}" y="{y}" width="{70 * min(max(frac, 0.0), 1.0):.1f}" '
                   'height="7" rx="1.5" fill="var(--fg)" fill-opacity="0.7"/>')
    out.append(_text(x + 106, y + 6, value, "mute"))
    return "".join(out)


def _flow(run: Run) -> str:
    top = run.top()
    if not top:
        return ""
    cpu, cores = run.is_cpu(), _cores(run)
    bw, gap, per_row = 186.0, 14.0, 5
    bh = 92.0 if cpu else 107.0
    out: list[str] = []
    y = 0.0
    for p, ns in _by_part(list(top)).items():
        out.append(_text(0, y + 13, p, "b") + _text(len(p) * 7.5 + 8, y + 13, f"— {PARTS[p]}",
                                                    "mute"))
        y += 22
        for i, n in enumerate(ns):
            col, row = i % per_row, i // per_row
            if col == 0 and row:
                y += bh + 12
            x = col * (bw + gap)
            s, u = top[n], top[n].get("util") or {}
            few = bool(u.get("few_samples"))
            ms = s.get("marks") or []
            g, c = u.get("g_util"), u.get("c_tree")
            tip = (f"{_name(n)} ({n})\n{_sec(n)[3]}\nmean {fmt(run.mean(n))} s · "
                   f"{pct(s.get('share'))} of step"
                   + ("" if cpu else f"\nGPU busy {pct(None if few else g)}")
                   + f"\nCPU {fmt(None if few else c)} cores\nmarks {_marks(ms)}")
            stroke = 'stroke="var(--mark)" stroke-width="2"' if ms else \
                'stroke="var(--line)" stroke-width="1"'
            box = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{bh}" rx="4" '
                   f'fill="var(--bg)" {stroke}/>',
                   _rect(x + 1, y + 1, 7, bh - 2, _paint(n)),
                   _text(x + 16, y + 18, _name(n), "b"), _text(x + 16, y + 33, n, "mute"),
                   _text(x + 16, y + 50, f"{fmt(run.mean(n))} s · {pct(s.get('share'))} of step")]
            gy = y + 60
            if not cpu:
                box.append(_gauge(x + 16, gy, "GPU", None if few or g is None else g,
                                  "—" if few or g is None else pct(g)))
                gy += 15
            box.append(_gauge(x + 16, gy, "CPU", None if few or c is None else c / cores,
                              "—" if few or c is None else f"{fmt(c)} cores"))
            if ms:
                box.append(_text(x + 16, y + bh - 8, " · ".join(ms), "badge"))
            out.append(f'<g><title>{esc(tip)}</title>{"".join(box)}</g>')
            if col < per_row - 1 and i < len(ns) - 1:
                ax, ay = x + bw + 3, y + bh / 2
                out.append(f'<path d="M{ax:.1f} {ay - 4:.1f} L{ax + 7:.1f} {ay:.1f} '
                           f'L{ax:.1f} {ay + 4:.1f}" fill="none" stroke="var(--mute)"/>')
        y += bh + 22
    w = per_row * bw + (per_row - 1) * gap
    gpu_words = "" if cpu else ("GPU: the share of the section's time in which a GPU kernel "
                                "was running. ")
    return (_svg(w + 2, y, "".join(out))
            + f'<p class="mute small">Each box is one section of the step, in the order it runs '
              f"(left to right, then down); its color strip is the one the overview and the "
              f"timeline use, and a red outline means a mark (see the overview). {gpu_words}CPU: the cores "
              f"the run's processes kept busy on average, on a scale to the {fmt(cores)} cores "
              "this machine allowed. A dash means the section was too short for the sampler "
              "(under 3 samples).</p>")


def _polylines(ts: list[float], vals: list[float | None], x: Callable[[float], float],
               y: Callable[[float], float], tip: str, dash: bool = False) -> str:
    runs: list[list[tuple[float, float]]] = [[]]
    for t, v in zip(ts, vals, strict=True):
        if v is None:
            runs.append([])
        else:
            runs[-1].append((t, v))
    style = ' stroke-dasharray="4 3"' if dash else ""
    return "".join(
        f'<polyline fill="none" stroke="var(--fg)" stroke-opacity="0.75" stroke-width="1.5"'
        f'{style} points="'
        + " ".join(f"{x(t) + dx:.1f},{y(v):.1f}" for t, v in r for dx in
                   ((0.0, 2.0) if len(r) == 1 else (0.0,)))
        + f'"><title>{esc(tip)}</title></polyline>' for r in runs if len(r) >= 1)


STRIPS = {"g_util": ("GPU busy", "share of time a GPU kernel ran, 0 to 100%"),
          "c_busy": ("CPU cores", "busy CPU cores on the whole machine"),
          "pcie": ("PCIe", "PCIe traffic in GB/s; solid into the GPU, dashed out of it")}


def _timeline(run: Run) -> str:
    steps = (run.view.get("timeline") or {}).get("steps") or []
    if not steps:
        return ""
    names = _distinct([sec["name"] for s in steps for sec in s["sections"] if sec["depth"] == 0
                       and not (run.sections.get(sec["name"]) or {}).get("rollup")
                       and sec["name"] not in TOTALS])
    ticks = _ticks(max((s.get("end_s") or 0.0) for s in steps) or 1.0, 8)
    axis_s = ticks[-1]
    left, pw, span_h, strip_h = 78.0, 900.0, 18.0, 20.0

    def strips_of(s: dict[str, Any]) -> dict[str, dict[str, list[float | None]]]:
        st = s.get("strips") or {}
        out: dict[str, Any] = {}
        if "g_util" in st:
            out["g_util"] = {"g_util": st["g_util"]}
        if "c_busy" in st:
            out["c_busy"] = {"c_busy": st["c_busy"]}
        pc = {k: [None if v is None else v / 1e9 for v in st[k]]
              for k in ("g_pcie_rx", "g_pcie_tx") if k in st}
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
        out.append(_text(left - 8, y + 13, f"step {s['step']}", "b", "end"))
        for sec in s["sections"]:
            if sec["depth"] != 0 or sec["name"] not in names:
                continue
            n = sec["name"]
            for a, b in sec["spans_s"]:
                tip = f"{_name(n)} ({n})\nstep {s['step']}\nfrom {fmt(a)} s to {fmt(b)} s"
                out.append(_rect(x(a), y, max(x(b) - x(a), 0.6), span_h, _paint(n), tip,
                                 ' stroke="var(--bg)" stroke-width="0.6"'))
        y += span_h + 4
        for key, cols in strips_of(s).items():
            top = maxes[key] or 1.0
            out.append(f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y + strip_h}" '
                       f'y2="{y + strip_h}"/>')
            out.append(_text(left - 8, y + 14, STRIPS[key][0], "mute", "end"))
            for fld, vals in cols.items():
                word = {"g_pcie_rx": "into the GPU", "g_pcie_tx": "out of the GPU"}.get(fld, "")
                out.append(_polylines(s["strips"]["t_s"], vals, x,
                                      lambda v, y=y, top=top: y + strip_h - strip_h * v / top,
                                      f"{STRIPS[key][0]} {word}".strip() + f"\nstep {s['step']}",
                                      fld == "g_pcie_tx"))
            y += strip_h + 3
        y += 10
    out.append(_axis(left, left + pw, y, 0, ticks, "seconds from the step's start"))
    used = [k for k in STRIPS if any(k in strips_of(s) for s in steps)]
    top_words = {"g_util": "", "c_busy": f", up to {fmt(maxes.get('c_busy'))}",
                 "pcie": f", up to {fmt(maxes.get('pcie'))} GB/s"}
    note = (" Lines under each step: " + "; ".join(f"{STRIPS[k][0]}, {STRIPS[k][1]}{top_words[k]}"
                                                    for k in used) + "."
            if used else " No sampler data for this run.")
    return (_legend(names) + _svg(left + pw + 20, y + 34, "".join(out))
            + f'<p class="mute small">Each bar is one timed step, its sections drawn where they '
              f"ran.{esc(note)}</p>")


def _c5(run: Run) -> str:
    c5 = run.view.get("c5")
    if not c5:
        return ""
    phases = {p: (c5.get("phases_s") or {}).get(p) or 0.0 for p in C5_PHASES}
    whole = run.mean("5")
    share = (run.sections.get("5") or {}).get("share")
    intro = ("<p>Check 5 checks every captured matmul with Freivalds' algorithm: it multiplies "
             "the claimed product and its inputs by random vectors and compares the results. "
             + (f"It took <b>{fmt(whole)} s</b> per step ({pct(share)} of the step). "
                if whole is not None else "")
             + f"Its five timed phases add up to {fmt(sum(phases.values()))} s per step:</p>")
    steps = "".join(f"<li><b>{esc(PHASES[p][0])}</b> <span class=\"code\">{esc(p)}</span> — "
                    f"{esc(PHASES[p][1])}: {fmt(phases[p])} s</li>" for p in C5_PHASES)
    out = [intro, f'<ul class="small">{steps}</ul>']
    classes = c5.get("classes") or {}
    if not classes:
        return "".join(out)
    order = list(CLASSES) + sorted(c for c in classes if c not in CLASSES)
    rows = [c for c in order if c in classes]

    def plain(c: str) -> tuple[str, str]:
        group, name = CLASSES.get(c, ("Other", c))
        if c == "dV" and not {"dQ", "dK"} & set(classes):
            name = "Gradients of values, queries and keys"
        return group, name

    def secs(c: str) -> dict[str, float]:
        return {p: (classes[c].get("phases_s") or {}).get(p) or 0.0 for p in C5_PHASES}

    ticks = _ticks(max(sum(secs(c).values()) for c in rows) or 1.0)
    x0, pw, rh = 455.0, 430.0, 19.0
    body = [_text(4, 12, "Belongs to", "b"), _text(165, 12, "Matmul product", "b"),
            _text(x0, 12, "Seconds per step, by phase", "b")]
    y, last = 22.0, None
    for c in rows:
        group, name = plain(c)
        if group != last:
            if last is not None:
                body.append(f'<line class="axis" x1="0" x2="{x0 + pw}" y1="{y - 2}" '
                            f'y2="{y - 2}"/>')
            body.append(_text(4, y + 13, group))
            last = group
        body.append(_text(165, y + 13, name) + _text(x0 - 8, y + 13, c, "mute", "end"))
        xx = x0
        for p, v in secs(c).items():
            w = pw * v / ticks[-1]
            if w > 0:
                body.append(_rect(xx, y + 2, max(w - 1, 0.5), rh - 6,
                                  f'fill="var(--p-{p})"',
                                  f"{name} ({c})\n{PHASES[p][0]} ({p}) {fmt(v)} s"))
            xx += w
        body.append(_text(xx + 6, y + 13, f"{fmt(sum(secs(c).values()))} s", "mute"))
        y += rh
    body.append(_axis(x0, x0 + pw, y + 4, 18, ticks, "seconds per step"))
    out.append("<h3>Phase time by product</h3>"
               '<p class="small">Each row is one kind of matmul product, grouped by the pass '
               "and the part of the model it belongs to; the bar splits its check-5 time into "
               "the five phases.</p>" + _phase_legend() + _svg(x0 + pw + 60, y + 38,
                                                               "".join(body)))
    return "".join(out)


# ---- comparisons -------------------------------------------------------------------------


def _compare(rows: list[tuple[str, Run]]) -> str:
    """A stacked bar of each run's depth-0 section means on one seconds scale, labeled with
    its total; a section a run lacks has no segment."""
    names = _distinct([n for _, r in rows for n in r.top()])
    totals = [sum(r.mean(n) or 0.0 for n in r.top()) for _, r in rows]
    ticks = _ticks(max(totals, default=0.0) or 1.0)
    x0, pw = 6.0, 860.0
    body, y = [], 4.0
    for (label, r), tot in zip(rows, totals, strict=True):
        body.append(_text(x0, y + 11, label))
        y += 16
        x = x0
        for n in names:
            v = r.mean(n) if n in r.top() else None
            if v:
                w = pw * v / ticks[-1]
                body.append(_rect(x, y, max(w - 1.5, 0.5), 20, _paint(n),
                                  f"{label}\n{_name(n)} ({n}) {fmt(v)} s"))
                x += w
        body.append(_text(x + 6, y + 14, f"{fmt(tot)} s", "mute"))
        y += 30
    body.append(_axis(x0, x0 + pw, y, 0, ticks, "seconds per step"))
    return _legend(names) + _svg(x0 + pw + 60, y + 34, "".join(body))


def _cross_model(runs: list[Run]) -> str:
    out = []
    for session in _distinct([r.field("session") for r in runs]):
        mine = [r for r in runs if r.field("session") == session]
        if len({r.field("model") for r in mine}) >= 2:
            out.append(f"<h3>Session {esc(session)}</h3>"
                       + _compare([(f"{r.model} · {r.field('attack')}", r) for r in mine]))
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


PLAIN_ROWS = {"train.load": "Load weights, clear gradients", "train.forward": "Forward pass",
              "train.backward": "Backward pass", "train.update": "Weight update (SGD)"}


def _plain(run: Run) -> str:
    """The run's training parts timed by the prover beside plain training, when its profile
    holds that comparison."""
    cmp = run.profile.get("comparisons")
    if not cmp:
        return ""
    rows = [[PLAIN_ROWS.get(n, n), fmt(c.get("prover_s")), fmt(c.get("plain_s")),
             fmt(c.get("ratio"))] for n, c in (cmp.get("sections") or {}).items()]
    st = cmp.get("step") or {}
    pr, pl = st.get("prover_s"), st.get("plain_s")
    rows.append(["Whole step: verified (prover, hand-off, verifier) against plain", fmt(pr),
                 fmt(pl), fmt(pr / pl if pr and pl else None)])
    return (_table(["Training part", "Prover, with matmul capture (s)", "Plain training (s)",
                    "Prover ÷ plain"], rows)
            + '<p class="mute small">Plain training runs the same steps on the same model '
              "without the protocol. The whole-step row puts the verified step, prover and "
              "verifier together, beside one plain step.</p>")


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
    per_run = (("overview", "Overview", _overview), ("diagram", "Step flow", _flow),
               ("timeline", "Timeline", _timeline), ("check5", "Check 5", _c5),
               ("plain", "Prover beside plain", _plain))
    for r in runs:
        for key, title, fn in per_run:
            add(f"r{r.idx}-{key}", title, fn(r), r)
    add("compare-models", "Across models", _cross_model(runs))
    add("compare-sessions", "Across sessions", _cross_session(runs))

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
            f"<style>{CSS}</style></head><body>{DEFS}<nav>{''.join(nav)}</nav><main>"
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
