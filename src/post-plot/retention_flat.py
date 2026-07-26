"""Companion figure: general-knowledge accuracy is BLIND to forgetting.

The forgetting that PPL exposes (fig_ppl_forgetting: SFT +10-20%, RL ~0%) leaves
NO trace in multiple-choice accuracy. This dot plot shows the commonsense basket
-- mean(HellaSwag, ARC, PIQA), acc_norm -- sitting flat across every arm, right on
the baseline. MC accuracy asks "is the top-ranked answer still right?" and the
answer stays yes; only the full next-token distribution (PPL) moved.

Read alongside fig_ppl_forgetting.png: same models, opposite verdicts -> the probe
you pick decides whether you "see" forgetting at all.

    uv run python src/post-plot/retention_flat.py   # -> assets/figures/fig_retention_flat.png

Single run per arm (no seed error bars yet); differences are within run-to-run
noise, which is the point -- they are all ~equal. Reads retention_runs.json.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

RET_JSON = "trainer_output/retention_runs.json"
OUT = "assets/figures/fig_retention_flat.png"

C_RL = "#0072B2"
C_SFT_IN = "#009E73"
C_SFT_OFF = "#D55E00"
C_BASE = "#666666"

# (model key, label, color, marker)
POINTS = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "baseline", C_BASE, "D"),
    ("trainer_output/indist-sft", "in-dist SFT", C_SFT_IN, "s"),
    ("trainer_output/offdist-sft", "off-dist SFT", C_SFT_OFF, "s"),
    ("trainer_output/grpo-beta002/checkpoint-474", "GRPO (RL)", C_RL, "o"),
]


def main():
    ret = {r["model"]: r for r in json.load(open(RET_JSON))}
    base_acc = ret[POINTS[0][0]]["avg_previous_task_acc"]

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # Baseline reference line -- every arm sits on it (baseline is the x=0 point).
    ax.axhline(base_acc, ls="--", color=C_BASE, lw=1.3, alpha=0.7, zorder=1)

    for xi, (model, label, color, marker) in enumerate(POINTS):
        acc = ret[model]["avg_previous_task_acc"]
        size = 260 if marker == "o" else (150 if marker == "D" else 190)
        ax.scatter(
            xi,
            acc,
            s=size,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            f"{acc:.3f}",
            (xi, acc),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xticks(range(len(POINTS)))
    ax.set_xticklabels([p[1] for p in POINTS], fontsize=11)
    ax.set_ylabel(
        "commonsense accuracy\nmean(HellaSwag, ARC, PIQA)  acc_norm", fontsize=12
    )
    # Zoom loose enough that the ~1pt spread reads as "flat", not dramatic.
    ax.set_ylim(base_acc - 0.06, base_acc + 0.06)
    ax.set_title(
        "Accuracy is blind to forgetting", fontsize=15, fontweight="bold", pad=14
    )
    ax.text(
        0.5,
        1.015,
        "SmolLM2-135M · same models that lose 10-20% PPL — MC accuracy stays flat (±~1pt)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#666666",
    )

    ax.grid(True, axis="y", color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
