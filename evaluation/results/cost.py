"""EV5: the cost results (EVALUATION_SPEC §5.5, EQ1b, EQ1c).

Sources, per model, pooled over its honest runs (attacks and seeds):

- P0, the plain training step, from the plain baseline (H3) paired with each honest run
  (:func:`plain_pair`), at the honest run's judged step numbers;
- P1, capture, derived from the honest run against that baseline (``Records.capture``);
- P2–P5 and the verifier's per-step checks from the honest run's judged steps; check 5 also as
  its two parts, glue recomputation (``5.glue``, which is check 3's cost) and the challenge
  products (``5.measure``);
- the one-off work at step 0: checks 0, 1, 8 and 9 and the ``h_D`` tree of P4, each a total
  per run;
- FLOPs and bytes hashed from the counting pass, which runs on step 1: the counts are the same
  at every step of a configuration (§3.4), so a calibration step's counts stand for the judged
  steps. Peak memory likewise comes from the memory pass (steps 1 and 2) and is the largest
  ``peak_bytes`` there; P1's memory is the growth over the plain step, not a peak;
- transcript bytes per step from the ``storage`` record.

Time per step is a mean ± std over the judged steps of all the model's honest runs. An honest
run that judged nothing (calibration) or has no time records (metrics off) is left out, the
latter with a warning.

Outputs, under ``evaluation/results/<scale>/``:

- ``cost.pdf`` and ``.png``: per model, a plain bar (P0), a prover bar (P0 and P1–P5 stacked)
  and a verifier bar (one segment per per-step check, check 5 in its two parts), each with the
  std of its per-step total.
- ``cost_ratios.csv`` and ``.md``: per model, for time and for FLOPs, the prover overhead
  (P1–P5), the verifier and the prover plus the verifier, each over P0, per step; and the
  verifier per run, one-off checks included, over P0 per run.
- ``cost_grid.csv`` and ``.md``: per model and component, the time per step (a total per run
  for one-off work), FLOPs, bytes hashed, hash calls, peak memory, and transcript bytes.

Run as ``python -m evaluation.results.cost [--scale test|full] [--root DIR]``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from evaluation.output import new_figure, write_figure, write_table
from evaluation.records import Records, Table, check3_rows
from evaluation.results.common import (
    honest_dirs,
    mean_std,
    parse,
    seeds_note,
    short_model,
)

__all__ = ["PROVER", "OVERHEAD", "PER_STEP_CHECKS", "BAR_SEGMENTS", "ONE_OFF", "LABEL",
           "ModelCost", "plain_pair", "gather", "grid_rows", "ratio_rows", "plot", "main"]

PROVER = ("P0", "P1", "P2", "P3", "P4", "P5")
OVERHEAD = PROVER[1:]
PER_STEP_CHECKS = ("4", "7", "2", "6a", "5", "6b", "2.root")  # driver order
SUBS = ("3", "5.glue", "5.measure", "6b.glue")  # parts of a check, never added to the total
BAR_SEGMENTS = ("4", "7", "2", "6a", "5.glue", "5.measure", "6b", "2.root")
ONE_OFF = ("0", "1", "8", "9")  # verifier, once per run at step 0
TREE = "P4 (tree)"  # P4's h_D tree, built once per run at step 0
LABEL = {
    "P0": "plain training step", "P1": "capture of matmul outputs", "P2": "serialization",
    "P3": "Merkle commitment", "P4": "paths into h_D", "P5": "writing the transcript",
    TREE: "h_D tree, once", "4": "batch anchor", "7": "chaining", "2": "commitment read",
    "3": "operands from committed leaves (its cost is 5.glue)", "6a": "update identity, linear",
    "5": "matmul checks", "5.glue": "glue recomputation", "5.measure": "challenge products",
    "6b": "update identity, scales, biases, W_E", "6b.glue": "6b's glue replay",
    "2.root": "root comparison", "0": "W_0 anchor", "1": "data and schedule",
    "8": "final weights", "9": "verdict", "prover": "prover per step (P0–P5)",
    "verifier": "verifier per step (checks 4–2.root)", "transcript": "transcript per step",
}
COUNT_FIELDS = ("flops", "hash_in_bytes", "hash_calls")
GRID_COLUMNS = ("model", "side", "component", "what", "per", "n", "time_s_mean", "time_s_std",
                *COUNT_FIELDS, "peak_bytes", "transcript_bytes")
RATIO_COLUMNS = ("model", "measure", "P0", "prover_overhead", "verifier",
                 "overhead_over_P0", "verifier_over_P0", "prover_plus_verifier_over_P0",
                 "verifier_per_run_over_P0_per_run", "steps_per_run", "runs")


@dataclass
class ModelCost:
    """One model's cost figures, pooled over its honest runs."""

    model: str
    runs: int = 0
    steps_per_run: list[int] = field(default_factory=list)
    # per-step components: one time per judged step
    time: dict[str, list[float]] = field(default_factory=dict)
    # per (run_dir, step): component → time, for the per-step totals
    by_step: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    # one-off components: one total per run
    once: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, dict[str, float]] = field(default_factory=dict)
    peak: dict[str, float] = field(default_factory=dict)
    transcript_bytes: list[float] = field(default_factory=list)

    def mean(self, component: str) -> float | None:
        return mean_std(self.time.get(component, ()))[0]

    def totals(self, components: Sequence[str]) -> list[float]:
        """The per-step sum of ``components``, over the steps that have all of them."""
        return [sum(v[c] for c in components) for v in self.by_step.values()
                if all(c in v for c in components)]


def _same(a: Any, b: Any) -> bool:
    return a is None or b is None or a == b


def plain_pair(recs: Records, run_dir: str) -> str | None:
    """The plain baseline for ``run_dir``: the plain run directory of the same model and device
    whose corpus and attack agree where both records hold them; among several, the one on the
    same hardware and thread count. ``None`` with a warning if that leaves not exactly one."""
    env = recs.environment(run_dir) or {}
    hits = [d for d in recs.plain_dirs()
            if (e := recs.environment(d) or {}).get("model") == env.get("model")
            and e.get("device") == env.get("device")
            and _same(e.get("corpus"), env.get("corpus"))
            and _same(e.get("attack"), env.get("attack"))]
    if len(hits) > 1:
        hits = [d for d in hits if all((recs.environment(d) or {}).get(k) == env.get(k)
                                       for k in ("hardware", "threads"))]
    if len(hits) != 1:
        warnings.warn(f"{run_dir}: {len(hits)} plain baselines match; no P0 or P1 for it",
                      UserWarning, stacklevel=2)
        return None
    return hits[0]


def _name(r: dict[str, Any]) -> str:
    return TREE if r["step"] == 0 and r["component"] == "P4" else r["component"]


_PER_STEP = {*PROVER, *PER_STEP_CHECKS, *SUBS}
_ONCE = {*ONE_OFF, TREE}


def _wanted(r: dict[str, Any]) -> bool:
    return r["component"] in _PER_STEP if r["step"] > 0 else _name(r) in _ONCE


def gather(recs: Records, model: str) -> ModelCost:
    mc = ModelCost(model)
    steps = recs.table("step")
    for d in honest_dirs(recs, [model]):
        mine = steps.where(run_dir=d)
        judged = {r["step"] for r in mine if r.get("judged")}
        if not judged:  # a calibration run: nothing judged, so no result
            continue
        if not recs.table("time").where(run_dir=d, scenario="honest"):
            warnings.warn(f"{d}: no time records (metrics off?); left out of the cost results",
                          UserWarning, stacklevel=2)
            continue
        mc.runs += 1
        mc.steps_per_run.append(len({r["step"] for r in mine}))
        plain = plain_pair(recs, d)
        cap = Table() if plain is None else recs.capture(d, plain=plain, scenario="honest")
        rows: dict[str, list[dict[str, Any]]] = {}
        for kind in ("time", "memory", "count"):
            own = recs.table(kind).where(run_dir=d, scenario="honest")
            extra = [] if plain is None else \
                [*cap.where(record=kind), *recs.table(kind).where(run_dir=plain, component="P0")]
            rows[kind] = sorted((r for r in [*own, *check3_rows(own), *extra] if _wanted(r)),
                                key=lambda r: r["step"])
        for r in rows["time"]:
            if r["step"] == 0:
                mc.once.setdefault(_name(r), []).append(float(r["time_s"]))
            elif r["step"] in judged:
                mc.time.setdefault(r["component"], []).append(float(r["time_s"]))
                mc.by_step.setdefault((d, r["step"]), {})[r["component"]] = float(r["time_s"])
        for r in rows["count"]:  # one step's counts stand for every step (§3.4)
            mc.counts.setdefault(_name(r), {f: float(r.get(f) or 0) for f in COUNT_FIELDS})
        for r in rows["memory"]:
            if r.get("peak_bytes") is not None:
                c = _name(r)
                mc.peak[c] = max(mc.peak.get(c, -np.inf), float(r["peak_bytes"]))
        mc.transcript_bytes += [float(r["transcript_bytes"]) for r in
                                recs.table("storage").where(run_dir=d)
                                if r.get("transcript_bytes") is not None]
    return mc


def _sum(xs: Sequence[float | None]) -> float | None:
    return None if any(x is None for x in xs) else float(sum(x for x in xs if x is not None))


def _div(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def _count_sum(mc: ModelCost, components: Sequence[str]) -> dict[str, float | None]:
    return {f: _sum([mc.counts[c][f] if c in mc.counts else None for c in components])
            for f in COUNT_FIELDS}


def grid_rows(costs: Sequence[ModelCost]) -> list[dict[str, Any]]:
    """The cost grid: per model, the per-step components, their totals, the one-off work and
    the transcript size."""
    out: list[dict[str, Any]] = []
    for mc in costs:
        def row(side: str, c: str, per: str, times: Sequence[float],
                counts: dict[str, Any] | None, peak: float | None = None,
                transcript: float | None = None) -> None:
            m, s, n = mean_std(times)
            if m is None and counts is None and peak is None and transcript is None:
                return
            out.append({"model": short_model(mc.model), "side": side, "component": c,
                        "what": LABEL.get(c, c), "per": per, "n": n or None,
                        "time_s_mean": m, "time_s_std": s, **(counts or {}),
                        "peak_bytes": peak, "transcript_bytes": transcript})

        for c in PROVER:
            row("prover", c, "step", mc.time.get(c, []), mc.counts.get(c), mc.peak.get(c))
        row("prover", "prover", "step", mc.totals(PROVER), _count_sum(mc, PROVER))
        for c in ("4", "7", "2", "3", "6a", "5", "5.glue", "5.measure", "6b", "6b.glue",
                  "2.root"):
            row("verifier", c, "step", mc.time.get(c, []), mc.counts.get(c), mc.peak.get(c))
        row("verifier", "verifier", "step", mc.totals(PER_STEP_CHECKS),
            _count_sum(mc, PER_STEP_CHECKS))
        for side, c in [("prover", TREE), *(("verifier", c) for c in ONE_OFF)]:
            row(side, c, "run", mc.once.get(c, []), mc.counts.get(c), mc.peak.get(c))
        row("prover", "transcript", "step", [], None,
            transcript=mean_std(mc.transcript_bytes)[0])
    return out


def ratio_rows(costs: Sequence[ModelCost]) -> list[dict[str, Any]]:
    """Per model, for time and for FLOPs: the prover overhead (P1–P5), the verifier and the
    prover plus the verifier over P0, per step; and the verifier per run over P0 per run."""
    out = []
    for mc in costs:
        steps = mean_std(mc.steps_per_run)[0]
        flops = {c: v["flops"] for c, v in mc.counts.items()}
        measures: tuple[tuple[str, Any, Any], ...] = (
            ("time", mc.mean, lambda c: mean_std(mc.once.get(c, []))[0]),
            ("FLOPs", flops.get, flops.get))
        for measure, per_step, once in measures:
            p0 = per_step("P0")
            over = _sum([per_step(c) for c in OVERHEAD])
            ver = _sum([per_step(c) for c in PER_STEP_CHECKS])
            ver_run = _sum([None if ver is None or steps is None else steps * ver,
                            *(once(c) for c in ONE_OFF)])
            out.append({
                "model": short_model(mc.model), "measure": measure, "P0": p0,
                "prover_overhead": over, "verifier": ver,
                "overhead_over_P0": _div(over, p0), "verifier_over_P0": _div(ver, p0),
                "prover_plus_verifier_over_P0": _div(_sum([p0, over, ver]), p0),
                "verifier_per_run_over_P0_per_run":
                    _div(ver_run, None if p0 is None or steps is None else steps * p0),
                "steps_per_run": steps, "runs": mc.runs})
    return out


def plot(costs: Sequence[ModelCost]) -> Figure:
    """Per model: the plain, prover and verifier bars, time per step, stacked by component."""
    fig, ax = new_figure(figsize=(2.6 * max(1, len(costs)) + 3.4, 4.0))
    # the prover in saturated colours, the verifier's checks in pastel ones, so no two match
    colour = {**dict(zip(PROVER, ("0.6", "C0", "C1", "C2", "C3", "C4"))),
              **dict(zip(BAR_SEGMENTS, colormaps["Set2"].colors))}
    width, ticks, labels, seen = 0.26, [], [], set()
    for i, mc in enumerate(costs):
        bars = (("plain", ("P0",)), ("prover", PROVER), ("verifier", BAR_SEGMENTS))
        for j, (name, parts) in enumerate(bars):
            x, top = i + (j - 1) * width * 1.15, 0.0
            for c in parts:
                m = mc.mean(c)
                if m is None:
                    continue
                ax.bar(x, m, width, bottom=top, color=colour[c], edgecolor="white", lw=0.5,
                       label=None if c in seen else f"{c}: {LABEL[c]}")
                seen.add(c)
                top += m
            std = mean_std(mc.totals(parts))[1]
            if top and std is not None:
                ax.errorbar(x, top, yerr=std, color="k", capsize=3, lw=1)
            ticks.append(x)
            labels.append(name)
    ax.set_xticks(ticks, labels, fontsize=8)
    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(range(len(costs)), [f"\n\n{short_model(mc.model)}" for mc in costs])
    sec.tick_params(length=0)
    ax.set_ylabel("time per step (s)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Time per step, mean ± std over judged honest steps "
                 f"({seeds_note(mc.runs for mc in costs)} runs)", fontsize=10)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the cost figure and tables (EVALUATION_SPEC §5.5).", argv)
    costs = [mc for mc in (gather(inp.records, m) for m in inp.models) if mc.runs]
    runs = seeds_note(mc.runs for mc in costs)
    paths = [
        *write_figure(plot(costs), "cost", inp.scale, root=inp.root),
        *write_table(ratio_rows(costs), "cost_ratios", inp.scale, root=inp.root,
                     columns=RATIO_COLUMNS, caption=(
                         "Cost ratios (EVALUATION_SPEC §5.5), per model over its honest runs "
                         f"({runs}). Per step: the prover overhead is P1–P5 and the verifier "
                         "its per-step checks, both over P0 (plain training, H3). Per run: the "
                         "verifier's steps plus its one-off checks 0, 1, 8 and 9, over P0 times "
                         "the steps. FLOPs come from the counting pass.")),
        *write_table(grid_rows(costs), "cost_grid", inp.scale, root=inp.root,
                     columns=GRID_COLUMNS, caption=(
                         "Cost grid (EVALUATION_SPEC §5.5), per model and component, over its "
                         f"honest runs ({runs}). Time is a mean ± std per judged step, or per "
                         "run for one-off work (per = run). Counts come from the counting pass "
                         "and peak memory from the memory pass; P1's memory is its growth over "
                         "the plain step. Check 3's cost is 5.glue; 5.glue, 5.measure and "
                         "6b.glue are parts of checks 5 and 6b.")),
    ]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
