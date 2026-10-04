#!/usr/bin/env python3
"""Combine deck.json + slides/*.html into one presentable deck.html.

Usage:  python3 build.py            # writes deck.html
        python3 build.py --watch    # rebuilds whenever a slide or deck.json changes

No dependencies beyond the Python standard library.
"""
import html
import re
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DECK = ROOT / "deck.json"
SLIDES = ROOT / "slides"
OUT = ROOT / "deck.html"
ICONS = ROOT / "icons"

# <x-icon name="ShieldCheck"> -> icons/shield-check.svg (Lucide names, https://lucide.dev/icons)
ALIAS = {"Warning": "triangle-alert", "Verified": "badge-check", "CheckCircle": "circle-check",
         "Chart": "chart-column"}
ICON_RE = re.compile(r'<x-icon\b([^>]*)\bname="([^"]+)"([^>]*)>\s*</x-icon>')


def icon_file(name):
    stem = ALIAS.get(name) or re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name).lower()
    return ICONS / f"{stem}.svg"


def inline_icons(src):
    def rep(m):
        f = icon_file(m.group(2))
        if not f.exists():
            print(f"warning: icon '{m.group(2)}' not found; add {f.relative_to(ROOT)} from lucide.dev", file=sys.stderr)
            return m.group(0)
        svg = re.sub(r"<!--.*?-->", "", f.read_text(encoding="utf-8"), flags=re.S).strip()
        return f'<x-icon{m.group(1)}name="{m.group(2)}"{m.group(3)}>{svg}</x-icon>'
    return ICON_RE.sub(rep, src)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{fonts}
<style>
  html, body {{ margin: 0; height: 100%; background: #0e1424; overflow: hidden; }}
  body {{ font-family: {body_font}, Arial, sans-serif; }}
  #stage {{ position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }}
  #frame {{ width: 1920px; height: 1080px; transform-origin: center center; position: relative; flex: none; }}
  #frame > section {{ position: absolute; inset: 0; width: 1920px; height: 1080px; box-sizing: border-box;
                      overflow: hidden; display: none; }}
  #frame > section.active {{ display: block; }}
  #frame > section.active[data-flex] {{ display: flex; }}
  #frame > section.active[data-grid] {{ display: grid; }}
  section * {{ box-sizing: border-box; }}
  section h1, section h2, section h3, section p, section ul, section ol {{ margin: 0; }}
  section aside {{ display: none; }}
  x-icon {{ display: inline-flex; flex: none; }}
  x-icon svg {{ width: 100%; height: 100%; }}
  #notes {{ position: fixed; left: 0; right: 0; bottom: 0; max-height: 35vh; overflow: auto; display: none;
            background: rgba(14,20,36,.95); color: #e8ebf2; font: 16px/1.5 system-ui, sans-serif;
            padding: 16px 24px; white-space: pre-wrap; }}
  body.show-notes #notes {{ display: block; }}
  #counter {{ position: fixed; right: 16px; bottom: 12px; color: #8a94a6; font: 13px system-ui, sans-serif; }}
  body.show-notes #counter {{ display: none; }}
</style>
</head>
<body>
<div id="stage"><div id="frame">
{slides}
</div></div>
<div id="notes"></div>
<div id="counter"></div>
<script>
(function () {{
  // ---- connectors: <x-connector x1 y1 x2 y2 head="none"> in slide (1920x1080) coordinates
  var NS = "http://www.w3.org/2000/svg", n = 0;
  document.querySelectorAll("x-connector").forEach(function (el) {{
    var cs = getComputedStyle(el);
    var color = cs.color, w = parseFloat(el.style.borderWidth) || 3;
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 1920 1080");
    svg.style.cssText = "position:absolute;left:0;top:0;width:1920px;height:1080px;overflow:visible;pointer-events:none";
    var line = document.createElementNS(NS, "line");
    ["x1", "y1", "x2", "y2"].forEach(function (k) {{ line.setAttribute(k, el.getAttribute(k)); }});
    line.setAttribute("stroke", color); line.setAttribute("stroke-width", w);
    line.setAttribute("stroke-linecap", "round");
    if (el.getAttribute("head") !== "none") {{
      var id = "ah" + (n++), m = document.createElementNS(NS, "marker");
      m.setAttribute("id", id); m.setAttribute("viewBox", "0 0 10 10"); m.setAttribute("refX", "9");
      m.setAttribute("refY", "5"); m.setAttribute("markerWidth", "5"); m.setAttribute("markerHeight", "5");
      m.setAttribute("orient", "auto-start-reverse");
      var p = document.createElementNS(NS, "path");
      p.setAttribute("d", "M0,0 L10,5 L0,10 z"); p.setAttribute("fill", color);
      m.appendChild(p);
      var defs = document.createElementNS(NS, "defs"); defs.appendChild(m); svg.appendChild(defs);
      line.setAttribute("marker-end", "url(#" + id + ")");
    }}
    svg.appendChild(line);
    el.replaceWith(svg);
  }});

  // ---- navigation
  var slides = Array.prototype.slice.call(document.querySelectorAll("#frame > section"));
  slides.forEach(function (s) {{
    var d = s.style.display;
    if (d === "flex") s.setAttribute("data-flex", "");
    if (d === "grid") s.setAttribute("data-grid", "");
    s.style.display = "";
    if (!s.style.position) s.style.position = "absolute";
  }});
  var frame = document.getElementById("frame"), notes = document.getElementById("notes"),
      counter = document.getElementById("counter"), cur = 0;
  function fit() {{
    var k = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    frame.style.transform = "scale(" + k + ")";
  }}
  function show(i) {{
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach(function (s, j) {{ s.classList.toggle("active", j === cur); }});
    var a = slides[cur].querySelector("aside");
    notes.textContent = a ? a.textContent.trim() : "(no notes)";
    counter.textContent = (cur + 1) + " / " + slides.length + "  ·  N = notes";
    history.replaceState(null, "", "#" + slides[cur].id);
  }}
  document.addEventListener("keydown", function (e) {{
    if (["ArrowRight", "PageDown", " ", "Enter"].indexOf(e.key) >= 0) show(cur + 1);
    else if (["ArrowLeft", "PageUp", "Backspace"].indexOf(e.key) >= 0) show(cur - 1);
    else if (e.key === "Home") show(0);
    else if (e.key === "End") show(slides.length - 1);
    else if (e.key === "n" || e.key === "N") document.body.classList.toggle("show-notes");
    else if (e.key === "f" || e.key === "F") document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
  }});
  document.getElementById("stage").addEventListener("click", function (e) {{
    show(e.clientX > window.innerWidth / 3 ? cur + 1 : cur - 1);
  }});
  window.addEventListener("resize", fit);
  fit();
  var start = slides.findIndex(function (s) {{ return "#" + s.id === location.hash; }});
  show(start >= 0 ? start : 0);
}})();
</script>
</body>
</html>
"""


def build():
    deck = json.loads(DECK.read_text(encoding="utf-8"))
    order = list(deck.get("order", []))
    # slides not listed in order are appended at the end, like the Claude Slides editor does
    extra = sorted(p.stem for p in SLIDES.glob("*.html") if p.stem not in order)
    parts = []
    for sid in order + extra:
        f = SLIDES / f"{sid}.html"
        if not f.exists():
            print(f"warning: slide '{sid}' is in deck.json but slides/{sid}.html is missing", file=sys.stderr)
            continue
        parts.append(inline_icons(f.read_text(encoding="utf-8").strip()))
    faces = deck.get("faces", {})
    fonts = "\n".join(
        f'<link rel="stylesheet" href="{html.escape(v["href"])}">' for v in faces.values() if v.get("href")
    )
    body_font = next((f"'{v['family']}'" for v in faces.values()), "system-ui")
    OUT.write_text(
        TEMPLATE.format(title=html.escape(deck.get("title", "Deck")), fonts=fonts,
                        body_font=body_font, slides="\n".join(parts)),
        encoding="utf-8",
    )
    print(f"built {OUT.name}: {len(parts)} slides")


def snapshot():
    files = [DECK, *SLIDES.glob("*.html")]
    return {str(p): p.stat().st_mtime for p in files}


if __name__ == "__main__":
    build()
    if "--watch" in sys.argv:
        last = snapshot()
        print("watching for changes (Ctrl+C to stop)")
        while True:
            time.sleep(0.5)
            now = snapshot()
            if now != last:
                last = now
                try:
                    build()
                except Exception as e:  # keep watching on a bad edit
                    print(f"build failed: {e}", file=sys.stderr)
