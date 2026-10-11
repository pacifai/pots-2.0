"""EV4: the per-layer figure (EVALUATION_SPEC §5.4), ``per_layer.pdf`` and ``.png``.

From the residual arrays of each A2 run at step ``t*``: for each layer, the largest ``ρ_5(m, j)``
over the layer's matmuls ``m`` and challenges ``j`` (:func:`layer_maxima`). Layers are numbered
``0 … L−1`` and the output layer is ``L`` (§4.2). One panel per model, each with its own x-range;
one line per attack and poisoning rate, the mean over seeds of each layer's maximum; y on a log
scale, with a dashed line at 1. Unscored tests (``NaN``) are left out; a layer with no measured
test has no point.

Run as ``python -m evaluation.results.per_layer [--scale test|full] [--root DIR]``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from evaluation.output import new_figure, write_figure
from evaluation.records import Records
from evaluation.results.common import (
    ATTACK_LABEL,
    attack_order,
    load_residuals,
    model_order,
    parse,
    residual_index,
    seeds_note,
    short_model,
)

__all__ = ["SCENARIO", "layer_maxima", "runs", "lines", "plot", "main"]

SCENARIO = "A2"


def layer_maxima(z: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(layers, maxima)``: each layer index present in ``p_layer``, ascending, and the largest
    scored ``ρ_5`` over its rows and columns (``NaN`` if it has none)."""
    layer, rho = z["p_layer"], z["rho_5"]
    if rho.ndim == 1:
        rho = rho[:, None]
    layers = np.unique(layer[layer >= 0])
    maxima = np.full(len(layers), np.nan)
    for i, ell in enumerate(layers):
        v = rho[layer == ell]
        v = v[~np.isnan(v)]
        if v.size:
            maxima[i] = v.max()
    return layers, maxima


def runs(recs: Records, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One entry per A2 run with its arrays at ``t*``: model, attack, poisoning rate, and the
    layer maxima."""
    index = residual_index(recs)
    out = []
    for r in recs.table("step").where(scenario=SCENARIO):
        if r.get("cheat_step") is None or r["step"] != r["cheat_step"]:
            continue
        if models is not None and r.get("model") not in models:
            continue
        arch = index.get((r["run_dir"], SCENARIO, r["step"]))
        if arch is None:
            warnings.warn(f"{r['run_dir']}: no residual arrays at t* = {r['step']}; left out",
                          UserWarning, stacklevel=2)
            continue
        layers, maxima = layer_maxima(load_residuals(arch))
        out.append({"run_dir": r["run_dir"], "model": r.get("model"), "attack": r.get("attack"),
                    "bpr_actual": r.get("bpr_actual"), "layers": layers, "maxima": maxima})
    return out


def lines(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per model, attack and poisoning rate: each layer's maximum, averaged over the
    seeds that measured it."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault((e["model"], e["attack"], e["bpr_actual"]), []).append(e)
    out = []
    for (model, attack, bpr), es in groups.items():
        layers = np.unique(np.concatenate([e["layers"] for e in es]))
        grid = np.full((len(es), len(layers)), np.nan)
        for i, e in enumerate(es):
            grid[i, np.searchsorted(layers, e["layers"])] = e["maxima"]
        with warnings.catch_warnings():  # a layer no seed measured stays NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(grid, axis=0)
        out.append({"model": model, "attack": attack, "bpr_actual": bpr, "layers": layers,
                    "mean": mean, "seeds": len(es)})
    return out


def plot(ls: list[dict[str, Any]], models: Sequence[str] | None = None) -> Figure:
    models = model_order(models if models else [x["model"] for x in ls]) or [None]
    attacks = attack_order(x["attack"] for x in ls)
    bprs = sorted({x["bpr_actual"] for x in ls if x["bpr_actual"] is not None})
    cmap = colormaps["viridis"]
    colour = {b: cmap(0.15 + 0.7 * i / max(1, len(bprs) - 1)) for i, b in enumerate(bprs)}
    styles = dict(zip(attacks, ("-", "--", ":", "-.")))
    fig, axes = new_figure(1, len(models), figsize=(3.6 * len(models) + 0.6, 3.2),
                           squeeze=False)
    for ax, model in zip(axes[0], models):
        mine = [x for x in ls if x["model"] == model]
        top = max((int(x["layers"].max()) for x in mine if x["layers"].size), default=0)
        for x in sorted(mine, key=lambda x: (str(x["attack"]), x["bpr_actual"] or 0.0)):
            y = np.where(x["mean"] > 0, x["mean"], np.nan)  # a log axis can't show 0
            label = f"BPR {100 * (x['bpr_actual'] or 0):.3g}%"
            if len(attacks) > 1:
                label += f", {ATTACK_LABEL.get(x['attack'], x['attack'])}"
            ax.plot(x["layers"], y, marker=".", color=colour.get(x["bpr_actual"], "C0"),
                    ls=styles.get(x["attack"], "-"), label=label)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="decision boundary (1)")
        ax.set_yscale("log")
        ticks = list(range(0, top, max(1, top // 6))) + [top]
        ax.set_xticks(ticks, [str(t) for t in ticks[:-1]] + [f"{top}\n(output)"])
        ax.set_xlim(-0.5, top + 0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("max ρ_5 in the layer")
        ax.set_title(short_model(model), fontsize=9)
        if not mine:
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(f"A2 at t*: largest ρ_5 per layer, mean over seeds "
                 f"({seeds_note(x['seeds'] for x in ls)})", fontsize=10)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the per-layer figure (EVALUATION_SPEC §5.4).", argv)
    fig = plot(lines(runs(inp.records, inp.models)), inp.models)
    for p in write_figure(fig, "per_layer", inp.scale, root=inp.root):
        print(p)


if __name__ == "__main__":
    main()
