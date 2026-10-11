"""EV5: the honest false-reject results (EVALUATION_SPEC §5.6).

Every number here comes from the judged steps of the honest runs (H1 and H2; T-H1 at test
scale), pooled per model over attacks and seeds. A step's tests are the rows of its residual
arrays (§4.2): ``ρ_5(m, j)``, ``ρ_κ(m)`` and ``ρ_6(W)``. A ``NaN`` score is unscored and left
out of both the count and the failures; an infinite one counts as failing. A judged step with no
residual archive still counts for the step rate, with a warning.

Outputs, under ``evaluation/results/<scale>/``:

- ``false_reject.csv`` and ``.md``: per model, the component false-reject rate of each family and
  in total, with the number of tests, and the step false-reject rate, with the number of steps.
- ``honest_histogram.pdf`` and ``.png``: per model, a histogram of the three families on a log
  x-axis with a line at 1. Bins span ``1e-8`` to ``1e4``; scores outside that range land in the
  end bins, and zeros and infinities, which a log axis can't show, are counted in the legend.
- ``cancellation_drift.pdf`` and ``.png``: per model and attack, the largest ``ρ_κ(m)`` of each
  class at each judged step, over the class's matmuls and the seeds. The line at 1 is the
  frozen ceiling ``κ_max`` (``DECISIONS_FULL_SCALE.md`` F9). The five classes that come
  closest to it are named; the rest are grey.
- ``margin_curve.pdf`` and ``.png``: per model, the component false-reject rate against the
  margin ``z``. The band ``τ = z·s_h`` scales with ``z``, so ``ρ_5`` at margin ``z`` is the
  recorded ``ρ_5 · Z / z``, ``Z`` the configured margin (the band file's ``stats.z``).
  ``κ_max`` and ``τ_W`` have fixed margins of their own and don't move with ``z``. The curve
  shows the ``ρ_5`` family and the total, over ``z`` from ``Z/16`` to ``2Z``; ``Z`` is marked.

Run as ``python -m evaluation.results.false_reject [--scale test|full] [--root DIR]``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from evaluation.output import new_figure, write_figure, write_table
from evaluation.records import Records
from evaluation.results.common import (
    ATTACK_LABEL,
    FAMILIES,
    FAMILY_LABEL,
    attack_order,
    bands,
    judged_honest_steps,
    load_residuals,
    margin_z,
    model_order,
    parse,
    residual_index,
    seeds_note,
    short_model,
)

__all__ = ["BINS", "FACTORS", "Tally", "tally", "table_rows", "margin_curve", "plot_histogram",
           "plot_drift", "plot_margin", "main"]

BINS = np.logspace(-8, 4, 121)  # ten per decade
# z / Z over the margin curve: Z/16 to 2Z, 16 points per doubling
FACTORS = 2.0 ** np.linspace(-4, 1, 81)
COLUMNS = ("model", "kind", "family", "n", "n_over_1", "rate", "runs")
N_NAMED = 5  # drift classes named in the legend


@dataclass
class Tally:
    """One model's judged honest steps and tests."""

    model: str
    runs: set[str] = field(default_factory=set)
    steps: int = 0
    rejected: int = 0
    n: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAMILIES, 0))
    fail: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAMILIES, 0))
    zeros: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAMILIES, 0))
    infinite: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAMILIES, 0))
    hist: dict[str, np.ndarray] = field(
        default_factory=lambda: {f: np.zeros(len(BINS) - 1, dtype=np.int64) for f in FAMILIES})
    # failing ρ_5 tests at each margin z = FACTORS · Z
    margin_fail: np.ndarray = field(default_factory=lambda: np.zeros(len(FACTORS), np.int64))

    def add(self, z: dict[str, np.ndarray]) -> None:
        for f in FAMILIES:
            x = np.asarray(z.get(f, np.empty(0)), dtype=float).ravel()
            x = x[~np.isnan(x)]  # unscored
            self.n[f] += x.size
            self.fail[f] += int(np.count_nonzero(~(x <= 1.0)))
            self.zeros[f] += int(np.count_nonzero(x == 0))
            self.infinite[f] += int(np.count_nonzero(np.isinf(x)))
            shown = np.clip(x[np.isfinite(x) & (x > 0)], BINS[0], BINS[-1])
            self.hist[f] += np.histogram(shown, BINS)[0]
            if f == "rho_5":
                s = np.sort(x)
                self.margin_fail += s.size - np.searchsorted(s, FACTORS, side="right")


def tally(recs: Records, models: Sequence[str] | None = None
          ) -> tuple[dict[str, Tally], dict[tuple[Any, Any], dict[str, dict[int, float]]]]:
    """Per model, the :class:`Tally` of its judged honest steps; and per model and attack, each
    class's largest finite ``ρ_κ(m)`` at each judged step."""
    index = residual_index(recs)
    out: dict[str, Tally] = {}
    drift: dict[tuple[Any, Any], dict[str, dict[int, float]]] = {}
    for r in judged_honest_steps(recs, models):
        t = out.setdefault(r.get("model"), Tally(r.get("model")))
        t.runs.add(r["run_dir"])
        t.steps += 1
        t.rejected += r.get("verdict") == "reject"
        arch = index.get((r["run_dir"], "honest", r["step"]))
        if arch is None:
            warnings.warn(f"{r['run_dir']}: judged step {r['step']} has no residual arrays; "
                          "it counts for the step rate only", UserWarning, stacklevel=2)
            continue
        z = load_residuals(arch)
        t.add(z)
        cls, kap = z.get("p_cls"), np.asarray(z.get("rho_kappa", np.empty(0)), dtype=float)
        if cls is None or not kap.size:
            continue
        mine = drift.setdefault((r.get("model"), r.get("attack")), {})
        for c in np.unique(cls):
            v = kap[(cls == c) & np.isfinite(kap)]
            if v.size:
                at = mine.setdefault(str(c), {})
                at[r["step"]] = max(at.get(r["step"], -np.inf), float(v.max()))
    return {m: out[m] for m in model_order(out)}, drift


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def table_rows(tallies: dict[str, Tally]) -> list[dict[str, Any]]:
    rows = []
    for m, t in tallies.items():
        base = {"model": short_model(m), "runs": len(t.runs)}
        for f in FAMILIES:
            rows.append({**base, "kind": "component", "family": FAMILY_LABEL[f], "n": t.n[f],
                         "n_over_1": t.fail[f], "rate": _rate(t.fail[f], t.n[f])})
        n, k = sum(t.n.values()), sum(t.fail.values())
        rows.append({**base, "kind": "component", "family": "total", "n": n, "n_over_1": k,
                     "rate": _rate(k, n)})
        rows.append({**base, "kind": "step", "family": "step", "n": t.steps,
                     "n_over_1": t.rejected, "rate": _rate(t.rejected, t.steps)})
    return rows


def margin_curve(t: Tally, z_conf: float) -> dict[str, np.ndarray]:
    """``z``, and the ``ρ_5`` and total component false-reject rates at each ``z``."""
    n5 = t.n["rho_5"]
    fixed_fail = t.fail["rho_kappa"] + t.fail["rho_6"]
    n_all = sum(t.n.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"z": FACTORS * z_conf,
                "rho_5": t.margin_fail / n5 if n5 else np.full(len(FACTORS), np.nan),
                "total": (t.margin_fail + fixed_fail) / n_all if n_all
                else np.full(len(FACTORS), np.nan)}


def _seeds(tallies: dict[str, Tally]) -> str:
    return seeds_note(len(t.runs) for t in tallies.values())


def plot_histogram(tallies: dict[str, Tally]) -> Figure:
    ms = list(tallies) or [None]
    fig, axes = new_figure(1, len(ms), figsize=(3.8 * len(ms) + 0.6, 3.2), squeeze=False)
    for ax, m in zip(axes[0], ms):
        t = tallies.get(m)
        for i, f in enumerate(FAMILIES):
            if t is None or not t.n[f]:
                continue
            unseen = t.zeros[f] + t.infinite[f]
            label = f"{FAMILY_LABEL[f]}: {t.n[f]} tests"
            if unseen:
                label += f" ({t.zeros[f]} zero, {t.infinite[f]} infinite, not shown)"
            ax.stairs(t.hist[f], BINS, color=f"C{i}", label=label)
        ax.axvline(1.0, color="k", ls="--", lw=1, label="decision boundary (1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("score")
        ax.set_ylabel("tests")
        ax.set_title(short_model(m), fontsize=9)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(f"Judged honest scores, pooled over seeds ({_seeds(tallies)})", fontsize=10)
    return fig


def plot_drift(drift: dict[tuple[Any, Any], dict[str, dict[int, float]]],
               models: Sequence[str] | None = None) -> Figure:
    models = model_order(models if models else [k[0] for k in drift]) or [None]
    attacks = attack_order(k[1] for k in drift) or [None]
    nr, nc = len(attacks), len(models)
    fig, axes = new_figure(nr, nc, figsize=(3.6 * nc + 1.2, 3.0 * nr + 0.6), squeeze=False)
    for i, attack in enumerate(attacks):
        for j, model in enumerate(models):
            ax = axes[i][j]
            classes = drift.get((model, attack), {})
            ranked = sorted(classes, key=lambda c: -max(classes[c].values()))
            for rank, c in reversed(list(enumerate(ranked))):
                steps = sorted(classes[c])
                y = [classes[c][s] for s in steps]
                if rank < N_NAMED:
                    ax.plot(steps, y, marker=".", color=f"C{rank}", label=c, zorder=3)
                else:
                    ax.plot(steps, y, color="0.75", lw=0.8, zorder=2)
            ax.axhline(1.0, color="k", ls="--", lw=1, label="frozen ceiling κ_max (1)")
            ax.set_ylim(bottom=0)
            ax.set_xlabel("judged step")
            ax.set_ylabel("largest ρ_κ(m) in the class")
            ax.set_title(f"{short_model(model)}, {ATTACK_LABEL.get(attack, attack)}",
                         fontsize=9)
            if classes:
                ax.legend(fontsize=6, loc="lower right", title=f"top {N_NAMED} classes",
                          title_fontsize=6)
            else:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center")
    fig.suptitle("Largest ρ_κ(m) per class at each judged honest step", fontsize=10)
    return fig


def plot_margin(tallies: dict[str, Tally], z_conf: float) -> Figure:
    ms = list(tallies) or [None]
    fig, axes = new_figure(1, len(ms), figsize=(3.6 * len(ms) + 0.6, 3.2), squeeze=False)
    for ax, m in zip(axes[0], ms):
        t = tallies.get(m)
        if t is not None and sum(t.n.values()):
            c = margin_curve(t, z_conf)
            n_all = sum(t.n.values())
            ax.plot(c["z"], c["rho_5"], color="C0", label=f"{FAMILY_LABEL['rho_5']} only")
            ax.plot(c["z"], c["total"], color="C3", ls="--", label="all component tests")
            ax.axhline(1.0 / n_all, color="0.6", lw=0.8, ls=":", label="one test in all")
            # a rate of 0 must show, so the axis is linear below one test in all
            ax.set_yscale("symlog", linthresh=1.0 / n_all)
            ax.set_ylim(0, 1)
        ax.axvline(z_conf, color="k", lw=1, label=f"configured z = {z_conf:g}")
        ax.set_xscale("log")
        ax.set_xlabel("margin z (τ = z·s_h)")
        ax.set_ylabel("component false-reject rate")
        ax.set_title(short_model(m), fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"False-reject rate against the margin ({_seeds(tallies)})", fontsize=10)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the honest false-reject results (EVALUATION_SPEC §5.6).", argv)
    tallies, drift = tally(inp.records, inp.models)
    z_conf = margin_z(bands(inp.root))
    caption = (
        "Honest false-reject rates (EVALUATION_SPEC §5.6), per model over the judged steps of "
        f"the honest runs, pooled over attacks and seeds ({_seeds(tallies)}). n counts the "
        "component tests (rows of the residual arrays, unscored ones left out) or the steps; "
        "n_over_1 those with a score above 1, or the rejected steps.")
    paths = [*write_table(table_rows(tallies), "false_reject", inp.scale, root=inp.root,
                          columns=COLUMNS, caption=caption),
             *write_figure(plot_histogram(tallies), "honest_histogram", inp.scale,
                           root=inp.root),
             *write_figure(plot_drift(drift, inp.models), "cancellation_drift", inp.scale,
                           root=inp.root),
             *write_figure(plot_margin(tallies, z_conf), "margin_curve", inp.scale,
                           root=inp.root)]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
