"""EV5: the cost results (EVALUATION_SPEC §5.5, EQ1b, EQ1c).

Sources, per model, pooled over its honest runs (attacks and seeds):

- P0, the plain training step, from the plain baseline (H3) paired with each honest run
  (:func:`plain_pair`). Each distinct baseline gives P0 once, at the judged step numbers of the
  honest runs paired with it, so two runs sharing one baseline don't count its steps twice;
- P1, capture, derived from the honest run against that baseline (``Records.capture``);
- P2–P5 and the verifier's per-step checks from the honest run's judged steps; check 5 also as
  its two parts, glue recomputation (``5.glue``, which is check 3's cost) and the challenge
  products (``5.measure``);
- the one-off work at step 0: checks 0, 1, 8 and 9 and the ``h_D`` tree of P4, each a total
  per run;
- FLOPs and bytes hashed from the counting pass, which runs on step 1: the counts are the same
  at every step of a configuration (§3.4), so a calibration step's counts stand for the judged
  steps. FLOPs are ``FlopCounterMode``'s formulas (EQ1c), which cover matmul-class ops only, so
  the elementwise checks 6a and 6b and the glue recomputation count about 0. Peak memory comes
  from the memory pass (steps 1 and 2) and is the largest ``peak_bytes`` there; P1's memory is
  the growth over the plain step, not a peak;
- transcript bytes per step from the ``storage`` record.

**The baseline must match.** A plain baseline is a candidate for an honest run when its model
and device are the same and its corpus and attack agree where both records hold them. The
candidate must also agree with the honest run on :data:`BASELINE_KEYS` (the code's commit and
dirty flag, the data hash ``h_D``, the run's shape, the machine and its software) wherever both
environment records hold the key; the keys only one record holds are listed as unchecked. A
mismatched baseline, no baseline or more than one is an error (:class:`BaselineError`) unless
``--allow-baseline-mismatch`` is given, which pairs the one mismatched candidate and writes the
mismatch into the ratio table and the captions.

Every per-step time is a mean ± sample std over the judged steps of all the model's honest
runs; calibration steps are left out. The verifier's per-step time is the mean of its per-step
total over those steps, and each ratio is that mean over P0's per-step mean. An honest run that
judged nothing (calibration) or has no time records (metrics off) is left out, the latter with
a warning.

Outputs, under ``evaluation/results/<scale>/``:

- ``cost.pdf`` and ``.png``: per model, a plain bar (P0), a prover bar (P0 and P1–P5 stacked)
  and a verifier bar (one segment per per-step check, check 5 in its two parts), each with the
  sample std of its per-step total.
- ``cost_ratios.csv`` and ``.md``: per model, for time and for FLOPs, the prover overhead
  (P1–P5), the verifier and the prover plus the verifier, each over P0, per judged step; with
  the paired plain run directories, both sides' commits and any baseline mismatch.
- ``cost_grid.csv`` and ``.md``: per model and component, the time per step (a total per run
  for one-off work), FLOPs, bytes hashed, hash calls, peak memory, and transcript bytes.

Run as ``python -m evaluation.results.cost [--scale test|full] [--root DIR]
[--allow-baseline-mismatch]``.
"""

from __future__ import annotations

import argparse
import sys
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
           "BASELINE_KEYS", "FLOPS_NOTE", "Baseline", "BaselineError", "ModelCost", "mismatch_note",
           "plain_pair", "gather", "grid_rows", "ratio_rows", "plot", "main"]

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
# the environment fields a plain baseline must share with its honest run, where both hold them:
# the code, the data, the run's shape (steps, batch n_s, length n, dtypes, the passes) and the
# machine with its software
BASELINE_KEYS = ("git_commit", "git_dirty", "h_D", "T", "n_s", "n", "master_dtype",
                 "compute_dtype", "pass_steps", "count_steps", "device", "hardware", "threads",
                 "platform", "python", "libraries", "mem_source", "MallocLargeCache")
FLOPS_NOTE = ("FlopCounterMode formulas (EQ1c): matmul-class ops only, so the elementwise "
              "checks 6a and 6b and the glue recomputation count about 0")
GRID_COLUMNS = ("model", "side", "component", "what", "per", "n", "time_s_mean", "time_s_std",
                *COUNT_FIELDS, "peak_bytes", "transcript_bytes")
RATIO_COLUMNS = ("model", "measure", "P0", "prover_overhead", "verifier",
                 "overhead_over_P0", "verifier_over_P0", "prover_plus_verifier_over_P0",
                 "judged_steps", "P0_steps", "runs", "plain_run_dir", "plain_commit",
                 "honest_commit", "baseline_mismatch", "baseline_unchecked", "flops_counter")


class BaselineError(ValueError):
    """No usable plain baseline for an honest run."""


@dataclass(frozen=True)
class Baseline:
    """The plain baseline paired with an honest run: its directory, the keys of
    :data:`BASELINE_KEYS` on which the two differ (key → (honest, plain)), and the keys only
    one of them records."""

    plain_dir: str
    mismatch: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    unchecked: tuple[str, ...] = ()


@dataclass
class ModelCost:
    """One model's cost figures, pooled over its honest runs."""

    model: str
    runs: int = 0
    # per-step components: one time per judged step; P0 once per distinct plain baseline
    time: dict[str, list[float]] = field(default_factory=dict)
    # per (run_dir, step): component → time, for the per-step totals
    by_step: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    # one-off components: one total per run
    once: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, dict[str, float]] = field(default_factory=dict)
    peak: dict[str, float] = field(default_factory=dict)
    transcript_bytes: list[float] = field(default_factory=list)
    # honest run directory → its baseline; run directory → its commit, for the tables
    baselines: dict[str, Baseline] = field(default_factory=dict)
    commits: dict[str, str] = field(default_factory=dict)

    def mean(self, component: str) -> float | None:
        return mean_std(self.time.get(component, ()))[0]

    def totals(self, components: Sequence[str]) -> list[float]:
        """The per-step sum of ``components``, over the judged steps that have all of them."""
        return [sum(v[c] for c in components) for v in self.by_step.values()
                if all(c in v for c in components)]


def _same(a: Any, b: Any) -> bool:
    return a is None or b is None or a == b


def _short(v: Any) -> str:
    if isinstance(v, str) and len(v) == 40 and all(c in "0123456789abcdef" for c in v):
        return v[:7]  # a git commit
    if isinstance(v, str) and len(v) == 64:
        return v[:12] + "…"  # a hash
    return str(v)


def _diff(key: str, honest: Any, plain: Any) -> str:
    if isinstance(honest, dict) and isinstance(plain, dict):
        sub = [k for k in sorted({*honest, *plain}) if honest.get(k) != plain.get(k)]
        return "; ".join(f"{key}.{k} {_short(honest.get(k))} ≠ {_short(plain.get(k))}"
                         for k in sub)
    return f"{key} {_short(honest)} ≠ {_short(plain)}"


def mismatch_note(mismatch: dict[str, tuple[Any, Any]]) -> str:
    """``git_commit 8dfa3ba ≠ a2fd813; git_dirty False ≠ True``: honest run first."""
    return "; ".join(_diff(k, h, p) for k, (h, p) in mismatch.items())


def _compare(env: dict[str, Any], plain_env: dict[str, Any]
             ) -> tuple[dict[str, tuple[Any, Any]], tuple[str, ...]]:
    mismatch, unchecked = {}, []
    for k in BASELINE_KEYS:
        h, p = env.get(k), plain_env.get(k)
        if h is None or p is None:
            if h is not None or p is not None:
                unchecked.append(k)
        elif h != p:
            mismatch[k] = (h, p)
    return mismatch, tuple(unchecked)


def plain_pair(recs: Records, run_dir: str, allow_mismatch: bool = False) -> Baseline:
    """The plain baseline (H3) for the honest run ``run_dir``: the plain run directory of the
    same model and device, whose corpus and attack agree where both records hold them, and
    which agrees with the run on :data:`BASELINE_KEYS` where both hold them. With
    ``allow_mismatch``, a single candidate that disagrees is paired anyway and its mismatch
    returned. Raises :class:`BaselineError` if there is no candidate, several match, or the only
    ones disagree and ``allow_mismatch`` is off."""
    env = recs.environment(run_dir) or {}
    cands = [d for d in recs.plain_dirs()
             if (e := recs.environment(d) or {}).get("model") == env.get("model")
             and e.get("device") == env.get("device")
             and _same(e.get("corpus"), env.get("corpus"))
             and _same(e.get("attack"), env.get("attack"))]
    if not cands:
        raise BaselineError(
            f"{run_dir}: no plain baseline (H3) of {env.get('model')} on {env.get('device')} "
            f"among the plain run directories {recs.plain_dirs()}; run "
            "verification.runs.plain_baseline with this run's settings first")
    checked = {d: _compare(env, recs.environment(d) or {}) for d in cands}
    exact = [d for d, (mm, _) in checked.items() if not mm]
    if len(exact) == 1:
        return Baseline(exact[0], {}, checked[exact[0]][1])
    if len(exact) > 1:
        raise BaselineError(f"{run_dir}: {len(exact)} plain baselines match it, {exact}; "
                            "keep one in the output directory")
    notes = "\n".join(f"  {d}: {mismatch_note(mm)}" for d, (mm, _) in checked.items())
    if not allow_mismatch:
        raise BaselineError(
            f"{run_dir}: the plain baseline differs from the honest run (honest ≠ plain):\n"
            f"{notes}\nRerun the plain baseline at the honest run's commit and settings, or "
            "pass --allow-baseline-mismatch to pair it anyway and record the mismatch")
    if len(cands) > 1:
        raise BaselineError(f"{run_dir}: {len(cands)} plain baselines differ from it:\n{notes}"
                            "\nKeep one in the output directory")
    d = cands[0]
    warnings.warn(f"{run_dir}: pairing the mismatched plain baseline {d} "
                  f"({mismatch_note(checked[d][0])})", UserWarning, stacklevel=2)
    return Baseline(d, *checked[d])


def _commit(env: dict[str, Any] | None) -> str:
    env = env or {}
    c = env.get("git_commit") or "unknown"
    return f"{c} (dirty)" if env.get("git_dirty") else str(c)


def _name(r: dict[str, Any]) -> str:
    return TREE if r["step"] == 0 and r["component"] == "P4" else r["component"]


_PER_STEP = {*PROVER, *PER_STEP_CHECKS, *SUBS}
_ONCE = {*ONE_OFF, TREE}


def _wanted(r: dict[str, Any]) -> bool:
    return r["component"] in _PER_STEP if r["step"] > 0 else _name(r) in _ONCE


def _p0_times(recs: Records, plain: str) -> dict[int, float]:
    return {r["step"]: float(r["time_s"]) for r in recs.table("time").where(
        run_dir=plain, component="P0") if r["step"] > 0 and r.get("time_s") is not None}


def gather(recs: Records, model: str, allow_mismatch: bool = False) -> ModelCost:
    """``model``'s cost figures over its honest runs. Raises :class:`BaselineError` when a run
    has no usable plain baseline (:func:`plain_pair`)."""
    mc = ModelCost(model)
    steps = recs.table("step")
    p0_steps: dict[str, set[int]] = {}  # plain baseline → the judged steps paired with it
    for d in honest_dirs(recs, [model]):
        judged = {r["step"] for r in steps.where(run_dir=d) if r.get("judged")}
        if not judged:  # a calibration run: nothing judged, so no result
            continue
        if not recs.table("time").where(run_dir=d, scenario="honest"):
            warnings.warn(f"{d}: no time records (metrics off?); left out of the cost results",
                          UserWarning, stacklevel=2)
            continue
        base = plain_pair(recs, d, allow_mismatch)
        plain = base.plain_dir
        mc.runs += 1
        mc.baselines[d] = base
        mc.commits[d] = _commit(recs.environment(d))
        mc.commits[plain] = _commit(recs.environment(plain))
        cap = recs.capture(d, plain=plain, scenario="honest")
        rows: dict[str, list[dict[str, Any]]] = {}
        for kind in ("time", "memory", "count"):
            own = recs.table(kind).where(run_dir=d, scenario="honest")
            # P0's time comes once per baseline, below; its memory and counts come from here
            extra = [*cap.where(record=kind), *([] if kind == "time" else
                                                recs.table(kind).where(run_dir=plain,
                                                                       component="P0"))]
            rows[kind] = sorted((r for r in [*own, *check3_rows(own), *extra] if _wanted(r)),
                                key=lambda r: r["step"])
        for r in rows["time"]:
            if r["step"] == 0:
                mc.once.setdefault(_name(r), []).append(float(r["time_s"]))
            elif r["step"] in judged:
                mc.time.setdefault(r["component"], []).append(float(r["time_s"]))
                mc.by_step.setdefault((d, r["step"]), {})[r["component"]] = float(r["time_s"])
        p0 = _p0_times(recs, plain)
        for t in judged & p0.keys():  # the prover's per-step total pairs P0 by step number
            mc.by_step.setdefault((d, t), {})["P0"] = p0[t]
        p0_steps.setdefault(plain, set()).update(judged)
        for r in rows["count"]:  # one step's counts stand for every step (§3.4)
            mc.counts.setdefault(_name(r), {f: float(r.get(f) or 0) for f in COUNT_FIELDS})
        for r in rows["memory"]:
            if r.get("peak_bytes") is not None:
                c = _name(r)
                mc.peak[c] = max(mc.peak.get(c, -np.inf), float(r["peak_bytes"]))
        mc.transcript_bytes += [float(r["transcript_bytes"]) for r in
                                recs.table("storage").where(run_dir=d)
                                if r.get("transcript_bytes") is not None]
    for plain, judged in p0_steps.items():
        p0 = _p0_times(recs, plain)
        if missing := sorted(judged - p0.keys()):
            warnings.warn(f"{plain}: no P0 time at judged steps {missing}", UserWarning,
                          stacklevel=2)
        mc.time.setdefault("P0", []).extend(p0[t] for t in sorted(judged & p0.keys()))
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


def _baseline_fields(mc: ModelCost) -> dict[str, Any]:
    plains = list(dict.fromkeys(b.plain_dir for b in mc.baselines.values()))
    notes = list(dict.fromkeys(mismatch_note(b.mismatch) for b in mc.baselines.values()
                               if b.mismatch))
    unchecked = list(dict.fromkeys(k for b in mc.baselines.values() for k in b.unchecked))
    return {"plain_run_dir": ", ".join(plains) or None,
            "plain_commit": ", ".join(mc.commits[p] for p in plains) or None,
            "honest_commit": ", ".join(dict.fromkeys(mc.commits[d] for d in mc.baselines))
            or None,
            "baseline_mismatch": " | ".join(notes) or None,
            "baseline_unchecked": ", ".join(unchecked) or None}


def ratio_rows(costs: Sequence[ModelCost]) -> list[dict[str, Any]]:
    """Per model, for time and for FLOPs: the prover overhead (P1–P5), the verifier and the
    prover plus the verifier over P0, per step. Time is the mean over judged steps of each
    side's per-step total, over P0's per-step mean."""
    out = []
    for mc in costs:
        flops = {c: v["flops"] for c, v in mc.counts.items()}
        base = _baseline_fields(mc)
        rows = (
            ("time", mc.mean("P0"), mean_std(mc.totals(OVERHEAD))[0],
             mean_std(mc.totals(PER_STEP_CHECKS))[0], None),
            ("FLOPs", flops.get("P0"), _sum([flops.get(c) for c in OVERHEAD]),
             _sum([flops.get(c) for c in PER_STEP_CHECKS]), FLOPS_NOTE))
        for measure, p0, over, ver, note in rows:
            out.append({
                "model": short_model(mc.model), "measure": measure, "P0": p0,
                "prover_overhead": over, "verifier": ver,
                "overhead_over_P0": _div(over, p0), "verifier_over_P0": _div(ver, p0),
                "prover_plus_verifier_over_P0": _div(_sum([p0, over, ver]), p0),
                "judged_steps": len(mc.totals(PER_STEP_CHECKS)),
                "P0_steps": len(mc.time.get("P0", [])), "runs": mc.runs, **base,
                "flops_counter": note})
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
            # P0 once per baseline step; the others per judged honest step
            std = mean_std(mc.time.get("P0", []) if name == "plain" else mc.totals(parts))[1]
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
    title = ("Time per step, mean ± sample std over judged honest steps "
             f"({seeds_note(mc.runs for mc in costs)} runs)")
    notes = [f"{short_model(mc.model)}: {n}" for mc in costs
             if (n := _baseline_fields(mc)["baseline_mismatch"])]
    if notes:
        title += "\nP0 from a mismatched plain baseline (honest ≠ plain): " + "; ".join(notes)
    fig.suptitle(title, fontsize=10 if not notes else 8)
    return fig


def _baseline_caption(costs: Sequence[ModelCost]) -> str:
    parts = []
    for mc in costs:
        b = _baseline_fields(mc)
        s = (f"{short_model(mc.model)}: P0 from {b['plain_run_dir']} at {b['plain_commit']}, "
             f"paired with honest runs at {b['honest_commit']}")
        if b["baseline_mismatch"]:
            s += (f"; the baseline differs from them (honest ≠ plain), paired by "
                  f"--allow-baseline-mismatch: {b['baseline_mismatch']}")
        else:
            s += "; it matches them on every key both environment records hold"
        if b["baseline_unchecked"]:
            s += f"; unchecked, recorded on one side only: {b['baseline_unchecked']}"
        parts.append(s + ".")
    return " ".join(parts)


def _options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-baseline-mismatch", action="store_true",
                   help="pair a plain baseline that differs from the honest run in commit, "
                        "data or run shape, and record the mismatch in the tables")


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the cost figure and tables (EVALUATION_SPEC §5.5).", argv, _options)
    try:
        costs = [mc for mc in (gather(inp.records, m, inp.args.allow_baseline_mismatch)
                               for m in inp.models) if mc.runs]
    except BaselineError as e:
        sys.exit(f"cost: {e}")
    if not costs:
        sys.exit("cost: no honest run with time records; nothing to write")
    runs = seeds_note(mc.runs for mc in costs)
    base = _baseline_caption(costs)
    paths = [
        *write_figure(plot(costs), "cost", inp.scale, root=inp.root),
        *write_table(ratio_rows(costs), "cost_ratios", inp.scale, root=inp.root,
                     columns=RATIO_COLUMNS, caption=(
                         "Cost ratios (EVALUATION_SPEC §5.5), per model over its honest runs "
                         f"({runs}). Per judged step (calibration steps left out): the prover "
                         "overhead is P1–P5 and the verifier its per-step checks, each the mean "
                         "of its per-step total over the judged steps, over P0's per-step mean "
                         "(plain training, H3, each baseline step once). The one-off checks "
                         "0, 1, 8 and 9 are in the cost grid. FLOPs come from the counting "
                         f"pass, as {FLOPS_NOTE}. {base}")),
        *write_table(grid_rows(costs), "cost_grid", inp.scale, root=inp.root,
                     columns=GRID_COLUMNS, caption=(
                         "Cost grid (EVALUATION_SPEC §5.5), per model and component, over its "
                         f"honest runs ({runs}). Time is a mean and sample std per judged step, "
                         "or per run for one-off work (per = run). Counts come from the counting "
                         f"pass, FLOPs as {FLOPS_NOTE}. Peak memory comes from the memory pass; "
                         "P1's memory is its growth over the plain step. Check 3's cost is "
                         "5.glue; 5.glue, 5.measure and 6b.glue are parts of checks 5 and 6b. "
                         f"{base}")),
    ]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
