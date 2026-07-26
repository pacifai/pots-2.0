"""Hero figure: RL's Razor at the LLM/token scale (seed-aggregated).

A (drift, skill) scatter. Each post-trained method is one point, averaged over 3
training seeds with 2D ±1 SD error bars:
  x = forward KL(pi_ft || pi_base) per token  -- how far it moved from base
  y = IFEval held-out satisfaction             -- task skill

The story is Pareto dominance: the RL point sits upper-left (more skill, less
drift) and dominates the SFT points in the lower-right -- and this survives seed
noise. Note the seed spread on RL's skill (y error bar): a single GRPO run can
look markedly stronger than the 3-seed mean, which is exactly the seed-lottery
caution the variance experiment makes.

    uv run python src/post-plot/razor_pareto.py   # -> assets/figures/fig_razor_pareto.png

Reads trainer_output/{kl_runs,grpo_runs,sft_ifeval_seeds}.json, so it tracks the
measurements. KL: forward + exact estimator, ref=SmolLM2-135M-Instruct.
"""

import json
import os
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

KL_JSON = "trainer_output/kl_runs.json"
GRPO_JSON = "trainer_output/grpo_runs.json"
SFT_IFEVAL_JSON = "trainer_output/sft_ifeval_seeds.json"
OUT = "assets/figures/fig_razor_pareto.png"

# Okabe-Ito colorblind-safe palette (blue/green/vermillion pass CVD; gray = base).
C_RL = "#0072B2"
C_SFT_IN = "#009E73"
C_SFT_OFF = "#D55E00"
C_BASE = "#666666"


def _kl_seeds(kl_runs, pattern):
    return [
        r["result"]["kl_per_token"]
        for r in kl_runs
        if pattern in r.get("label", "") and "kl_per_token" in r.get("result", {})
    ]


def _grpo_ifeval_seeds(grpo_runs, pattern):
    return [
        r["after"]["mean"]
        for r in grpo_runs
        if pattern in str(r.get("run", "")) and isinstance(r.get("after"), dict)
    ]


def _ms(vals):
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def main():
    kl_runs = json.load(open(KL_JSON))
    grpo_runs = json.load(open(GRPO_JSON))
    sft_if = json.load(open(SFT_IFEVAL_JSON))
    base_if = next(r["after"]["mean"] for r in grpo_runs if r.get("run") == "baseline")

    # (label, color, marker, size, kl seeds, ifeval seeds)
    points = [
        (
            "in-dist SFT",
            C_SFT_IN,
            "s",
            150,
            _kl_seeds(kl_runs, "indist-sft-s"),
            sft_if["indist"],
        ),
        (
            "off-dist SFT",
            C_SFT_OFF,
            "s",
            150,
            _kl_seeds(kl_runs, "offdist-sft-s"),
            sft_if["offdist"],
        ),
        (
            "GRPO (RL)",
            C_RL,
            "o",
            240,
            _kl_seeds(kl_runs, "grpo-beta002-s"),
            _grpo_ifeval_seeds(grpo_runs, "beta002-s"),
        ),
    ]

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    # Aggregate.
    agg = []
    for label, color, marker, size, kls, ifs in points:
        klm, kls_sd = _ms(kls)
        ifm, ifs_sd = _ms(ifs)
        agg.append((label, color, marker, size, klm, kls_sd, ifm, ifs_sd))

    # Pareto frontier RL reaches: baseline -> GRPO mean.
    rl = next(a for a in agg if a[0] == "GRPO (RL)")
    ax.plot(
        [0, rl[4]], [base_if, rl[6]], "--", color=C_RL, lw=1.4, alpha=0.55, zorder=1
    )

    # Baseline (deterministic, no seeds).
    ax.scatter(
        0,
        base_if,
        s=130,
        c=C_BASE,
        marker="D",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )

    for label, color, marker, size, klm, kl_sd, ifm, if_sd in agg:
        ax.errorbar(
            klm,
            ifm,
            xerr=kl_sd,
            yerr=if_sd,
            fmt=marker,
            color=color,
            markersize=(size / 18) ** 0.5 * 4,
            markeredgecolor="white",
            markeredgewidth=1.5,
            ecolor="#555555",
            elinewidth=1.3,
            capsize=4,
            zorder=5,
        )

    # Direct labels.
    off = {"in-dist SFT": (10, 8), "off-dist SFT": (10, -18), "GRPO (RL)": (12, -6)}
    for label, color, marker, size, klm, kl_sd, ifm, if_sd in agg:
        dx, dy = off[label]
        ax.annotate(
            label,
            (klm, ifm),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10.5,
            color="#222222",
            fontweight="bold" if label == "GRPO (RL)" else "normal",
        )
    ax.annotate(
        "baseline",
        (0, base_if),
        textcoords="offset points",
        xytext=(8, -16),
        fontsize=10.5,
        color="#222222",
    )

    # Ideal corner cue.
    ax.annotate(
        "ideal\nhigh new skill · low drift",
        (0.012, 0.58),
        fontsize=10,
        color=C_RL,
        style="italic",
        va="top",
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.09, 0.58),
            (0.015, 0.595),
            arrowstyle="->",
            color=C_RL,
            lw=1.2,
            alpha=0.7,
            mutation_scale=12,
        )
    )

    ax.set_xlim(-0.02, 0.40)
    ax.set_ylim(0.27, 0.62)
    ax.set_xlabel(
        r"drift from base  —  $D_{\mathrm{KL}}(\pi_{\mathrm{ft}}\,\|\,\pi_{\mathrm{base}})$ per token  →",
        fontsize=12,
    )
    ax.set_ylabel("↑  IFEval held-out satisfaction (skill)", fontsize=12)
    ax.set_title(
        "RL learns more while moving less", fontsize=15, fontweight="bold", pad=14
    )
    ax.text(
        0.5,
        1.015,
        "SmolLM2-135M · IFEval · 3 seeds, ±1 SD — RL Pareto-dominates SFT",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#666666",
    )

    ax.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=C_RL,
            markersize=12,
            label="RL (GRPO)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=C_SFT_IN,
            markersize=11,
            label="in-dist SFT",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=C_SFT_OFF,
            markersize=11,
            label="off-dist SFT",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="w",
            markerfacecolor=C_BASE,
            markersize=10,
            label="base (untrained)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
