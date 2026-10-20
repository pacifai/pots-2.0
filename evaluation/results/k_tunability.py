"""FS12 and EQ11: check 5's cost against its detection floor as ``k`` varies (EVALUATION_SPEC
§5.8). It reads records only and decides nothing.

    .venv/bin/python -m evaluation.results.k_tunability [--run-dir DIR] [--scale test|full]

Inputs, all under ``$VERIF_OUTPUT_DIR``:

- **The k run** (``verification.runs.k_tunability``): its ``k_source`` (the stored step, the
  floor's inputs, the other checks' times), one ``k_point`` per ``k`` and its ``k_summary``
  (the linear fits of the time in ``k``, the reproduction). ``--run-dir`` names the run
  directory; by default it is the latest one holding a ``k_summary``, by its ``environment``
  record's start time.
- **The coherence factor C**: the ``gradient_coherence`` ratio of the calibration run whose
  ``calibration`` record names the k run's band file. Without one, ``C = 1``.
- **The miss constant ĉ** (§5.7): the planted-error sweep on the same stored step (the same
  root, and the same band file in its ``environment`` record), the stored-transcript sweep if
  there is one, else the latest. On the binding product (the one that sets ``k``), per error
  shape, every trial the sweep fits (swept ``x ≥ c_fit_x_min``) is pooled over the grid and
  the deep points: ``ĉ = Σ misses / Σ vectors/x``, ``x`` the realized one, with the Poisson
  standard error ``√misses / Σ vectors/x``. The shape with the largest ``ĉ`` gives the
  measured floor. Without a sweep the table and figure have no measured floor.

Per ``k``, the table has:

- **Security bits.** ``k·b₀``, the Freivalds bits per product at the sizing target ``f = 1``
  (``F_TARGET``), and ``λ_eff = k·b₀ − (N − λ)``, what remains of them after the union bound
  and the grinding budget, against ``λ`` = 25 at test scale.
- **Detection floor.** The planted error ``f = ‖Δ‖_F/‖P‖_F`` every product's check holds to
  ``λ`` bits: predicted by the sizing formula, ``f_ach(k) = c·τ·e_max·2^{N/k}`` with
  ``c = C_ANTI = √(2/3)``, and measured, the same with ``ĉ`` in place of ``c``.
- **Guaranteed substitution rate.** ``rate = f_ach(k)·C/(√B·‖δg‖/‖g‖)`` (the sizing appendix's
  (12.1)), with ``C`` measured and ``‖δg‖/‖g‖ = 2`` hand-picked; ``rate·B`` is the poisoned
  records per batch it covers, and ``k`` covers one when ``rate·B ≤ 1``. The rate at ``C = 1``,
  which the k run's ``k_top`` assumed, is beside it.
- **Cost.** Check 5's median time over the run's rounds; the verifier's time per step, the
  other checks at the configured ``k`` plus check 5 at ``k``; check 5's FLOPs; its memory
  growth (peak less start, 1 GB = 10⁹ bytes).

The figure has the verifier's time per step on x and the floor on a log y-axis, a labelled
point per ``k``, the configured ``k`` marked, and the floor that covers one poisoned record per
batch at the measured ``C`` as a line. Floors above ``Y_TOP`` are drawn at the top edge.
Outputs: ``k_tunability.{pdf,png,csv,md}`` under ``evaluation/results/<scale>/``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.output import SCALES, new_figure, write_figure, write_table
from evaluation.records import Records, read

__all__ = ["NAME", "Y_TOP", "COHERENCE_FALLBACK", "MissConstant", "KResult", "k_run",
           "coherence", "miss_constant", "guaranteed_rate", "table_rows", "build", "caption",
           "figure", "write", "main"]

NAME = "k_tunability"
Y_TOP = 1e3  # floors above this are drawn at the top edge, labelled
COHERENCE_FALLBACK = 1.0  # C when no calibration run of the band file is found

COLUMNS = ("k", "configured", "bits_f1", "lambda_eff_f1", "floor_pred", "floor_meas", "rate",
           "records_per_batch", "covers", "rate_c1", "check5_s", "verifier_s", "check5_gflop",
           "mem_growth_gb", "verdict")
FORMATS = {"bits_f1": ".1f", "lambda_eff_f1": ".1f", "floor_pred": ".3g", "floor_meas": ".3g",
           "rate": ".3g", "records_per_batch": ".3g", "rate_c1": ".3g", "check5_s": ".3f",
           "verifier_s": ".3f", "check5_gflop": ".2f", "mem_growth_gb": ".3f"}


def _started(rec: Records, run_dir: str) -> str:
    env = rec.environment(run_dir)
    return str(env.get("started") or "") if env else ""


def k_run(rec: Records, run_dir: str | None = None) -> str:
    """The k run's directory: ``run_dir`` if given, else the latest holding a ``k_summary``."""
    dirs = rec.table("k_summary").distinct("run_dir")
    if run_dir is not None:
        if run_dir not in dirs:
            raise ValueError(f"{run_dir!r} holds no k_summary record (run directories with one: "
                             f"{dirs})")
        return run_dir
    if not dirs:
        raise ValueError("no k_summary record: run verification.runs.k_tunability first")
    return max(dirs, key=lambda d: _started(rec, d))


def coherence(rec: Records, band_hash: str) -> tuple[float, str | None]:
    """``C`` from the calibration run of ``band_hash`` and its run directory, else
    ``(COHERENCE_FALLBACK, None)``. The latest such run wins."""
    dirs = {r["run_dir"] for r in rec.table("calibration")
            if r.get("band_file_hash") == band_hash}
    rows = [r for r in rec.table("gradient_coherence") if r["run_dir"] in dirs]
    if not rows:
        return COHERENCE_FALLBACK, None
    r = max(rows, key=lambda r: _started(rec, r["run_dir"]))
    return float(r["ratio"]), r["run_dir"]


@dataclass(frozen=True)
class MissConstant:
    """The binding product's pooled single-challenge miss constant (module docstring)."""

    c_hat: float
    se: float
    misses: int
    vectors: int
    shape: str
    name: str
    run_dir: str
    source: str  # "stored" or "in-process"
    x_min: float
    by_phase: dict[str, tuple[float, float]]  # phase → (ĉ, standard error)
    committed: float  # ĉ with x taken against the committed product's norm


def _pool(acc: Sequence[float]) -> tuple[float, float]:
    misses, exposure = acc[0], acc[1]
    return misses / exposure, math.sqrt(misses) / exposure


def miss_constant(rec: Records, root: str, band_hash: str) -> MissConstant | None:
    """``ĉ`` from the sweep on the step of ``root`` judged by ``band_hash``; ``None`` without a
    sweep or a fitted trial."""
    sources = [r for r in rec.table("sweep_source") if r.get("root") == root
               and (rec.environment(r["run_dir"]) or {}).get("band_file_hash") == band_hash]
    if not sources:
        return None
    stored = [r for r in sources if r.get("source") == "stored"]
    src = max(stored or sources, key=lambda r: _started(rec, r["run_dir"]))
    d = src["run_dir"]
    x_min = {(s["shape"], s["phase"]): float(s["c_fit_x_min"])
             for s in rec.table("sweep_summary") if s["run_dir"] == d and s["binding"]}
    # shape → phase → [misses, Σ vectors/x, Σ vectors/x_committed, vectors]
    acc: dict[str, dict[str, list[float]]] = {}
    name = ""
    for p in rec.table("sweep_point"):
        if p["run_dir"] != d or not p["binding"]:
            continue
        key = (p["shape"], p["phase"])
        if key not in x_min:
            raise ValueError(f"{d}: sweep points of {key} have no sweep_summary")
        if p["x"] < x_min[key]:
            continue
        x = p["x"] * p["f_real_mean"] / p["f"]  # the realized x, mean over the point's trials
        a = acc.setdefault(p["shape"], {}).setdefault(p["phase"], [0.0, 0.0, 0.0, 0.0])
        a[0] += p["misses"]
        a[1] += p["vectors"] / x
        a[2] += p["vectors"] * p["norm_ratio_mean"] / x
        a[3] += p["vectors"]
        name = p["name"]
    pooled = {s: [sum(a[i] for a in ph.values()) for i in range(4)] for s, ph in acc.items()}
    pooled = {s: a for s, a in pooled.items() if a[1] > 0}
    if not pooled:
        return None
    shape = max(pooled, key=lambda s: _pool(pooled[s])[0])
    a = pooled[shape]
    c_hat, se = _pool(a)
    return MissConstant(
        c_hat=c_hat, se=se, misses=int(a[0]), vectors=int(a[3]), shape=shape, name=name,
        run_dir=d, source=str(src.get("source")),
        x_min=min(x for (s, _), x in x_min.items() if s == shape),
        by_phase={ph: _pool(v) for ph, v in acc[shape].items() if v[1] > 0},
        committed=a[0] / a[2])


def guaranteed_rate(f: float, C: float, B: int, ratio: float) -> float:
    """The sizing appendix's (12.1): ``f·C/(√B·‖δg‖/‖g‖)``."""
    return f * C / (math.sqrt(B) * ratio)


@dataclass
class KResult:
    """What the table, its caption and the figure are built from."""

    run_dir: str
    source: dict[str, Any]
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    C: float
    C_run_dir: str | None
    miss: MissConstant | None
    notes: list[str] = field(default_factory=list)


def table_rows(source: dict[str, Any], points: Sequence[dict[str, Any]], C: float,
               c_hat: float | None) -> list[dict[str, Any]]:
    """One row per ``k``, ascending (module docstring)."""
    B, ratio, c = int(source["B"]), float(source["gradient_ratio"]), float(source["c"])
    ks = [p["k"] for p in points]
    if len(set(ks)) != len(ks):
        raise ValueError(f"k_point records repeat a k: {sorted(ks)}")
    rows = []
    for p in sorted(points, key=lambda p: p["k"]):
        f = float(p["f_ach"])
        rate = guaranteed_rate(f, C, B, ratio)
        rows.append({
            "k": p["k"], "configured": p["k"] == source["k_conf"], "bits_f1": p["bits"],
            "lambda_eff_f1": p["lambda_eff"], "floor_pred": f,
            "floor_meas": None if c_hat is None else f * c_hat / c,
            "rate": rate, "records_per_batch": rate * B, "covers": rate * B <= 1,
            "rate_c1": guaranteed_rate(f, 1.0, B, ratio), "check5_s": p["time_s"],
            "verifier_s": source["other_s"] + p["time_s"], "check5_gflop": p["flops"] / 1e9,
            "mem_growth_gb": p["mem_growth_bytes"] / 1e9, "verdict": p["verdict"]})
    return rows


def build(rec: Records, run_dir: str | None = None) -> KResult:
    """Read the k run, ``C`` and ``ĉ`` and build the rows."""
    d = k_run(rec, run_dir)
    [source] = rec.table("k_source").where(run_dir=d)
    [summary] = rec.table("k_summary").where(run_dir=d)
    points = rec.table("k_point").where(run_dir=d)
    if not points:
        raise ValueError(f"{d} holds no k_point record")
    C, C_dir = coherence(rec, source["band_file_hash"])
    miss = miss_constant(rec, source["root"], source["band_file_hash"])
    notes = []
    if C_dir is None:
        notes.append(f"no calibration run of band file {source['band_file_hash'][:8]}…: "
                     f"C = {COHERENCE_FALLBACK:g}")
    if miss is None:
        notes.append("no planted-error sweep of this step: no measured floor")
    rows = table_rows(source, points, C, None if miss is None else miss.c_hat)
    return KResult(d, source, summary, rows, C, C_dir, miss, notes)


def _fit(f: dict[str, Any]) -> str:
    return (f"{f['intercept']:.3f} s + {1e3 * f['slope']:.2f} ms·k (R² {f['r2']:.2f}, "
            f"largest residual {100 * f['max_rel_resid']:.1f}%)")


def caption(res: KResult) -> str:
    """The table's caption: the step, both assumptions, ``ĉ``'s source and the time fits."""
    s, sm, m = res.source, res.summary, res.miss
    B, ratio = int(s["B"]), float(s["gradient_ratio"])
    conf = next((r for r in res.rows if r["configured"]), None)
    parts = [
        f"Check 5's cost against its detection floor for k = {res.rows[0]['k']}…"
        f"{res.rows[-1]['k']}, on the stored step {s['step']} (root {s['root'][:8]}…, band file "
        f"{s['band_file_hash'][:8]}…); configured k = {s['k_conf']}.",
        f"Security bits: k·b₀ at the sizing target f = 1 (b₀ = {s['b0']:.2f}), and "
        f"λ_eff = k·b₀ − (N − λ) (N = {s['N']:.2f}, λ = {s['lam']:g}).",
        f"Floor: the planted error f = ‖Δ‖_F/‖P‖_F at which every product's check keeps λ "
        f"bits, predicted c·τ·e_max·2^(N/k) with c = {s['c']:.4f}, τ = {s['tau']:.2f}, "
        f"e_max = {s['e_max']:.4g}.",
    ]
    if m is None:
        parts.append("No planted-error sweep of this step, so no measured floor.")
    else:
        phases = ", ".join(f"{ph} {c:.3f} ± {se:.3f}" for ph, (c, se) in m.by_phase.items())
        parts.append(
            f"Measured: the same with ĉ = {m.c_hat:.3f} ± {m.se:.3f} in place of c, the "
            f"planted-error sweep's single-challenge miss constant on {m.name} ({m.shape} "
            f"errors, {m.misses} misses in {m.vectors} challenges at swept x ≥ {m.x_min:g}; "
            f"{phases}; {m.committed:.3f} with x against the committed product's norm; "
            f"{m.source} sweep).")
    c_src = ("measured by calibration" if res.C_run_dir is not None
             else "no calibration run found, a placeholder")
    parts.append(
        f"Rate: f_achieved·C/(√B·‖δg‖/‖g‖) (sizing appendix (12.1)), with C = {res.C:.3f} "
        f"({c_src}) and ‖δg‖/‖g‖ = {ratio:g} (hand-picked); records per batch is rate·B with "
        f"B = {B}, and k covers one poisoned record when it is at most 1. rate_c1 is the rate "
        f"at C = 1, which k_top = {s['k_top']} assumed.")
    reps = s.get("repeats")
    parts.append(
        f"Cost: medians of {reps} rounds; the verifier's time is the other checks at "
        f"k = {s['k_conf']} ({s['other_s']:.3f} s) plus check 5 at k. Check 5's time fits "
        f"{_fit(sm['fits_from_2']['time_s'])} over k ≥ {sm['marginal_from']}, and "
        f"{_fit(sm['fits']['time_s'])} over every k.")
    if conf is not None:
        verdict = ("reproduced" if sm["reproduced"] else "did NOT reproduce")
        parts.append(f"The k = {s['k_conf']} pass {verdict} the stored step's check-5 verdict "
                     f"and residuals bit for bit.")
    return " ".join(parts)


def _series(ax: Any, rows: Sequence[dict[str, Any]], key: str, **style: Any) -> None:
    drawn = [r for r in rows if r[key] is not None and r[key] > 0]  # a log axis has no 0
    on = [r for r in drawn if r[key] <= Y_TOP]
    off = [r for r in drawn if r[key] > Y_TOP]
    ax.plot([r["verifier_s"] for r in on], [r[key] for r in on], linestyle="none", **style)
    if off:
        top = {k: v for k, v in style.items() if k != "label"}
        top["marker"] = "^"
        ax.plot([r["verifier_s"] for r in off], [Y_TOP] * len(off), linestyle="none",
                clip_on=False, **top)


def figure(res: KResult) -> Any:
    """The verifier's time per step against the floor (module docstring)."""
    s, m, rows = res.source, res.miss, res.rows
    B, ratio = int(s["B"]), float(s["gradient_ratio"])
    fig, ax = new_figure(figsize=(6.4, 4.6))
    ax.set_yscale("log")
    _series(ax, rows, "floor_pred", marker="o", color="C0", markersize=5,
            label=f"predicted, c = {s['c']:.3f}")
    if m is not None:
        _series(ax, rows, "floor_meas", marker="o", markerfacecolor="none", color="C1",
                markersize=9, label=f"measured, ĉ = {m.c_hat:.3f} ({m.name}, {m.shape})")
    for r in rows:
        y = min(r["floor_pred"], Y_TOP)
        # odd k labelled above right, even k below right: neighbouring k sit close in time
        ax.annotate(f"{r['k']}", (r["verifier_s"], y), xytext=(7, 4 if r["k"] % 2 else -10),
                    textcoords="offset points", fontsize=7, annotation_clip=False)
        if r["configured"]:
            ax.plot([r["verifier_s"]], [y], linestyle="none", marker="s", markersize=14,
                    markerfacecolor="none", color="C3", label=f"configured k = {r['k']}")
    cover = ratio / (res.C * math.sqrt(B))  # rate·B = 1
    ax.axhline(cover, color="grey", linestyle="--", linewidth=1,
               label=f"covers 1 poisoned record of B = {B} (C = {res.C:.2f})")
    lo = min(f for r in rows for f in (r["floor_pred"], r["floor_meas"]) if f and f > 0)
    ax.set_ylim(min(lo, cover) / 2, Y_TOP)
    off = [r["k"] for r in rows if r["floor_pred"] > Y_TOP]
    if off:
        ax.text(0.99, 0.97, f"▲ k = {off[0]}…{off[-1]}: floor above {Y_TOP:g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel(f"verifier time per step (s): other checks at k = {s['k_conf']}, "
                  "check 5 at k")
    ax.set_ylabel("detection floor f = ‖Δ‖_F / ‖P‖_F")
    ax.set_title("Check 5's cost against its detection floor, per k", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="major", linewidth=0.3)
    return fig


def write(res: KResult, scale: str, *, root: Any = None) -> list[Any]:
    """The figure and the table; returns the four paths."""
    paths = list(write_figure(figure(res), NAME, scale, root=root))
    paths += write_table(res.rows, NAME, scale, root=root, columns=COLUMNS, formats=FORMATS,
                         caption=caption(res))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="the k run's directory, relative to $VERIF_OUTPUT_DIR")
    ap.add_argument("--scale", choices=SCALES, default="test")
    a = ap.parse_args(argv)
    res = build(read(), a.run_dir)
    for n in res.notes:
        print(f"note: {n}")
    for p in write(res, a.scale):
        print(p)
    m = res.miss
    print(f"{res.run_dir}: k = {res.rows[0]['k']}…{res.rows[-1]['k']}, C = {res.C:.4f}, "
          + ("no ĉ" if m is None else f"ĉ = {m.c_hat:.4f} ± {m.se:.4f} ({m.name}, {m.shape})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
