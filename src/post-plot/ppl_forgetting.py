"""Companion hero figure: the forgetting COST that RL avoids (seed-aggregated).

razor_pareto.py shows drift -> skill (RL gains more, moves less). This shows the
other half: drift -> forgetting. Held-out perplexity RISE vs the base model, on
two corpora, one grouped bar per method, averaged over 3 training seeds with
±1 SD error bars:

  x = method (in-dist SFT, off-dist SFT, GRPO)
  y = perplexity increase over base  (higher = more general LM ability lost)

The commonsense-MC basket (retention_eval.py) was FLAT across all arms -- MC
accuracy on stored knowledge is robust to instruction fine-tuning at 135M. PPL is
not: both SFT arms lose ~10% (wikitext) / ~20% (Pile) and GRPO stays ~0 on both,
and this holds tightly across seeds (the error bars are tiny).

    uv run python src/post-plot/ppl_forgetting.py   # -> assets/figures/fig_ppl_forgetting.png

Reads trainer_output/{ppl,kl}_runs.json (per-seed), so it tracks the measurements.
"""

import json
import os
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PPL_JSON = "trainer_output/ppl_runs.json"
KL_JSON = "trainer_output/kl_runs.json"
OUT = "assets/figures/fig_ppl_forgetting.png"

# Same Okabe-Ito hues as razor_pareto.py, so a method keeps its color across the
# paired figures.
C_RL = "#0072B2"  # blue
C_SFT_IN = "#009E73"  # bluish green -- in-distribution SFT
C_SFT_OFF = "#D55E00"  # vermillion  -- off-distribution SFT

BASELINE = "HuggingFaceTB/SmolLM2-135M-Instruct"
# (model-key substring, label, color) -- matches the 3 seed checkpoints per arm.
ARMS = [
    ("indist-sft-s", "in-dist SFT", C_SFT_IN),
    ("offdist-sft-s", "off-dist SFT", C_SFT_OFF),
    ("grpo-beta002-s", "GRPO (RL)", C_RL),
]
CORPORA = [("wikitext", "wikitext", "//"), ("pile_10k", "Pile", None)]


def _seed_deltas(ppl, pattern, corpus, base_ppl):
    """ΔPPL% vs base for every seed checkpoint whose model key contains pattern."""
    return [
        (r[corpus]["word_perplexity"] / base_ppl - 1) * 100
        for m, r in ppl.items()
        if pattern in m and corpus in r
    ]


def _mean_kl(kl_runs, pattern):
    vals = [
        r["result"]["kl_per_token"]
        for r in kl_runs
        if pattern in r.get("label", "") and "kl_per_token" in r.get("result", {})
    ]
    return mean(vals) if vals else None


def main():
    ppl = {r["model"]: r for r in json.load(open(PPL_JSON))}
    base = {c: ppl[BASELINE][c]["word_perplexity"] for c, _, _ in CORPORA}
    kl_runs = json.load(open(KL_JSON)) if os.path.exists(KL_JSON) else []

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    width = 0.36
    for ci, (corpus, _clabel, hatch) in enumerate(CORPORA):
        offset = (ci - 0.5) * width
        for xi, (pattern, _label, color) in enumerate(ARMS):
            ds = _seed_deltas(ppl, pattern, corpus, base[corpus])
            m = mean(ds)
            sd = stdev(ds) if len(ds) > 1 else 0.0
            ax.bar(
                xi + offset,
                m,
                width,
                yerr=sd,
                capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
                color=color,
                alpha=1.0 if hatch is None else 0.5,
                hatch=hatch,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                f"{m:.1f}±{sd:.1f}%",
                (xi + offset, m + sd),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9,
                color="#222222",
                fontweight="bold",
            )

    ax.axhline(0, color="#666666", lw=1.2, zorder=2)

    # x labels carry method + seed-mean KL (drift). KL comes from kl_runs.json once
    # the seed KL sweep has run; falls back to a plain label if not yet measured.
    labels = []
    for pattern, label, _color in ARMS:
        kl = _mean_kl(kl_runs, pattern)
        labels.append(f"{label}\nKL/tok={kl:.2f}" if kl is not None else label)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylabel("↑  held-out perplexity increase vs base  (%)", fontsize=12)
    ax.set_ylim(0, 25)
    ax.set_title(
        "RL learns the task without forgetting",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.015,
        "SmolLM2-135M · general-text PPL rise (2 corpora, 3 seeds, ±1 SD) — SFT forgets, RL ~0%",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#666666",
    )

    ax.grid(True, axis="y", color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Patch(facecolor="#888888", hatch="//", alpha=0.5, label="wikitext"),
        Patch(facecolor="#888888", label="Pile"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
