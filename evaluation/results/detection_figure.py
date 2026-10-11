"""EV4: the detection figure (EVALUATION_SPEC §5.2), ``detection.pdf`` and ``.png``.

One row of panels per attack and one panel per model (EQ4); test scale has one of each. In each
panel x is the batch poisoning rate in percent and y a score on a log scale:

- A2's matmul score ``ρ_5`` and A3's update score ``ρ_6`` at step ``t*``, each a mean ± std
  over seeds (:func:`points`). The step record's score is the maximum over the step's tests,
  ``+inf`` if one of them is ``NaN`` or infinite; a mean with an infinite seed is infinite and
  drawn as a triangle at the top edge;
- one shaded band per score, from the smallest to the largest step score over the judged
  honest steps of that model and attack (:func:`honest_ranges`);
- a dashed line at 1, the decision boundary.

A1 and the hidden step are certain, so they are only in the detection table.

Run as ``python -m evaluation.results.detection_figure [--scale test|full] [--root DIR]``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from evaluation.output import new_figure, write_figure
from evaluation.records import Records
from evaluation.results.common import (
    ATTACK_LABEL,
    attack_order,
    judged_honest_steps,
    mark_infinite,
    mean_std,
    model_order,
    parse,
    seeds_note,
    short_model,
)

__all__ = ["CURVES", "points", "honest_ranges", "plot", "main"]

# scenario → the score its curve shows, and the color of the curve and its honest band
CURVES = {"A2": ("rho_5", "C0", "A2: ρ_5"), "A3": ("rho_6", "C1", "A3: ρ_6")}


def points(recs: Records, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One point per model, attack, scenario (A2, A3) and poisoning rate: the mean and sample
    std over seeds of the scenario's score at ``t*``. With an infinite score among the seeds
    the mean is ``inf`` and the std ``None``; ``infinite`` counts those seeds."""
    steps = recs.table("step").where(scenario=lambda s: s in CURVES)
    if models is not None:
        steps = steps.where(model=lambda m: m in models)
    groups: dict[tuple[Any, ...], list[float]] = {}
    for r in steps:
        if r.get("cheat_step") is None or r["step"] != r["cheat_step"]:
            continue
        score = CURVES[r["scenario"]][0]
        key = (r.get("model"), r.get("attack"), r["scenario"], r.get("bpr_actual"))
        if r.get(score) is None:
            warnings.warn(f"{r['run_dir']}: no {score} at t* = {r['step']}; left out",
                          UserWarning, stacklevel=2)
            continue
        groups.setdefault(key, []).append(float(r[score]))
    out = []
    for (model, attack, scen, bpr), xs in groups.items():
        n_inf = sum(not math.isfinite(x) for x in xs)
        mean, std, n = mean_std(xs) if not n_inf else (math.inf, None, len(xs))
        out.append({"model": model, "attack": attack, "scenario": scen,
                    "score": CURVES[scen][0], "bpr_actual": bpr, "mean": mean, "std": std,
                    "seeds": n, "infinite": n_inf})
    return sorted(out, key=lambda p: (str(p["model"]), str(p["attack"]), p["scenario"],
                                      -1.0 if p["bpr_actual"] is None else p["bpr_actual"]))


def honest_ranges(recs: Records, models: Sequence[str] | None = None
                  ) -> dict[tuple[Any, Any, str], tuple[float, float]]:
    """``(model, attack, score)`` → the smallest and largest finite step score over the judged
    honest steps. An infinite one is a false reject, counted by ``false_reject``."""
    out: dict[tuple[Any, Any, str], tuple[float, float]] = {}
    for (model, attack), rows in judged_honest_steps(recs, models).group_by(
            "model", "attack").items():
        for score, _, _ in CURVES.values():
            xs = [float(r[score]) for r in rows
                  if r.get(score) is not None and math.isfinite(float(r[score]))]
            if xs:
                out[(model, attack, score)] = (min(xs), max(xs))
    return out


def plot(pts: list[dict[str, Any]], ranges: dict[tuple[Any, Any, str], tuple[float, float]],
         models: Sequence[str] | None = None) -> Figure:
    models = model_order(models if models else [p["model"] for p in pts]
                         + [k[0] for k in ranges])
    attacks = attack_order([p["attack"] for p in pts] + [k[1] for k in ranges]) or [None]
    nr, nc = len(attacks), max(1, len(models))
    fig, axes = new_figure(nr, nc, figsize=(3.4 * nc + 0.6, 2.9 * nr + 0.6), squeeze=False)
    for i, attack in enumerate(attacks):
        for j, model in enumerate(models or [None]):
            ax, drawn = axes[i][j], False
            for scen, (score, colour, label) in CURVES.items():
                band = ranges.get((model, attack, score))
                if band is not None:
                    drawn = True
                    ax.axhspan(*band, color=colour, alpha=0.15, lw=0,
                               label=f"honest {score.replace('rho_', 'ρ_')} range")
                    for edge in band:  # a narrow range would not show as a band alone
                        ax.axhline(edge, color=colour, lw=0.6, alpha=0.6)
                ps = [p for p in pts if (p["model"], p["attack"], p["scenario"])
                      == (model, attack, scen) and p["bpr_actual"] is not None]
                if not ps:
                    continue
                drawn = True
                x = np.array([100.0 * p["bpr_actual"] for p in ps])
                y = np.array([p["mean"] for p in ps])
                s = np.array([p["std"] or 0.0 for p in ps])
                fin = np.isfinite(y)
                # the lower bar stops short of 0, which a log axis can't show
                ax.errorbar(x[fin], y[fin], yerr=[np.minimum(s, 0.999 * y)[fin], s[fin]],
                            color=colour, marker="o" if scen == "A2" else "s", capsize=3,
                            label=label)
                mark_infinite(ax, x[~fin], colour, f"{label} infinite")
            ax.axhline(1.0, color="k", ls="--", lw=1, label="decision boundary (1)")
            ax.set_yscale("log")
            ax.set_xlabel("batch poisoning rate (%)")
            ax.set_ylabel("score at t*")
            ax.set_title(f"{short_model(model)}, {ATTACK_LABEL.get(attack, attack)}",
                         fontsize=9)
            if not drawn:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Detection scores at t*, mean ± std over seeds "
                 f"({seeds_note(p['seeds'] for p in pts)})", fontsize=10)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the detection figure (EVALUATION_SPEC §5.2).", argv)
    fig = plot(points(inp.records, inp.models), honest_ranges(inp.records, inp.models),
               inp.models)
    for p in write_figure(fig, "detection", inp.scale, root=inp.root):
        print(p)


if __name__ == "__main__":
    main()
