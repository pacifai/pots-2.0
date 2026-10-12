"""EV7: the PoTS comparison record (EVALUATION_SPEC §7), ``pots_comparison.md``.

The record is data for the paper's discussion section: one Markdown file of tables, never
drawn into a figure (EQ12). It sets our records beside the results of PoTS (Seddik et al.,
arXiv:2510.15106), which :data:`REFERENCE_CSV` holds:

- Table 1, copied exactly from the paper's LaTeX source (arXiv v1, ``templateArxiv.tex``). Each
  row is one model, attack, BPR level and quantity (``ASR_trigger`` or ``ASR_clean``): the mean
  and its (−, +) spread as printed, the source cell, and the line it came from;
- §7.1's values read off Figs. 3–6, as the spec states them. Each is marked as read from a
  figure, to about ±0.02 on a ratio axis and ±0.1 s on a time axis.

:func:`load_reference` reads the file, and :func:`table1` and :func:`fig4_seconds` check and
index the parts the record uses. The record holds §7.2's three comparisons:

1. **Trigger lift**, postponed with §5.1 (F6c). PoTS's lift, ASR_trigger − ASR_clean from
   Table 1's means, in percentage points, beside an empty column for ours.
2. **Cost**, per model. PoTS's verify ÷ train for the output layer only and for 5 more layers,
   from Fig. 4's seconds, with the span the ±0.1 s reading allows. Beside it are our verifier
   ÷ P0 and (prover + verifier) ÷ P0, time per judged step, as the cost script computes them
   (:func:`evaluation.results.cost.ratio_rows`). PoTS's seconds sit beside ours only for a
   model whose timed runs, the honest runs and their plain baselines, all name an H100 in their
   environment record, with the honest runs' batch at PoTS's 16,384 tokens (EQ14). The plain
   baseline's batch is checked only where its record holds one. Every other model is compared
   by ratio only, and the record says why.
3. **Detection latency**, per scenario. Our measured latency (§2.3), the first rejected step
   minus ``t*``, over the cheat runs of every model, as the detection table computes it
   (:func:`evaluation.results.detection_table.latency`). It stands against PoTS's claim, in
   text only, that it detects an attack at the step where it is injected.

The tables list PoTS's four models, then any other model in the records. A model with no run in
the records reads "no record", never 0. A model whose honest runs have no timed judged step (a
calibration run only, or metrics off) reads :data:`NO_TIME` in the cost tables.

**Scale.** §7 produces the record at full scale only, as ``results/full/pots_comparison.md``.
At test scale the same code writes ``results/test/pots_comparison.md`` under a banner that
marks it a rehearsal and not a result (§1). That lets the test-scale cells exercise the script
before the full-scale runs (FS13, milestone EM1).

A plain baseline that doesn't match its honest run stops the script, as it stops the cost
script, unless ``--allow-baseline-mismatch`` is given; the record then names the mismatch.

Run as ``python -m evaluation.results.pots_comparison [--scale test|full] [--root DIR]
[--allow-baseline-mismatch]``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from evaluation.output import markdown_table, write_markdown
from evaluation.records import Records
from evaluation.results import cost
from evaluation.results.common import (
    ATTACK_LABEL,
    ATTACK_ORDER,
    CHEATS,
    EXPECTED_CHECK,
    mean_std,
    model_order,
    parse,
    short_model,
)
from evaluation.results.detection_table import latency, run_outcomes

__all__ = ["REFERENCE_CSV", "REFERENCE_COLUMNS", "POTS_MODELS", "TABLE1_ATTACKS", "TABLE1_BPR",
           "TABLE1_QUANTITIES", "FIG4_CONDITIONS", "TIME_READING", "POTS_BATCH_TOKENS",
           "NO_RECORD", "NO_COUNTERPART", "NO_TIME", "POTS_LATENCY_CLAIM", "load_reference",
           "table1", "fig4_seconds", "pots_ratio", "lift_rows", "cost_rows", "seconds_blockers",
           "seconds_rows", "latency_rows", "render", "main"]

REFERENCE_CSV = Path(__file__).with_name("pots_reference.csv")
REFERENCE_COLUMNS = ("kind", "source", "model", "paper_model", "attack", "bpr", "quantity",
                     "condition", "stat", "value", "spread_minus", "spread_plus", "low", "high",
                     "unit", "precision", "source_text", "transcribed_from")
_NUMBERS = ("value", "spread_minus", "spread_plus", "low", "high")
# PoTS's models, in EQ4's panel order, as short_model names them
POTS_MODELS = ("Llama-3.2-1B", "Falcon3-1B", "Qwen2.5-0.5B", "Qwen2.5-1.5B")
TABLE1_ATTACKS = ("refusal", "jailbreak")  # the paper's column order
TABLE1_BPR = ("clean", "10", "25", "50", "75")
TABLE1_QUANTITIES = ("ASR_trigger", "ASR_clean")
FIG4_CONDITIONS = ("training", "verify, output layer only", "verify, output layer + 5 layers")
TIME_READING = Decimal("0.1")  # §7.1: Fig. 4's seconds are read to about ±0.1 s
POTS_BATCH_TOKENS = 16_384  # PoTS's batch, and ours at full scale: 128 sequences × 128 tokens
_H100 = "H100"
NO_RECORD = "no record"
NO_COUNTERPART = "no PoTS counterpart"
NO_TIME = "no honest run with timed judged steps"
# PoTS's latency claim, quoted from the arXiv v1 source: the abstract, then §1's contributions
POTS_LATENCY_CLAIM = (
    '"Our method enables early detection of the attack (at the step the attack is injected)" '
    '(abstract); "our method enables detection of the attack as early as possible, '
    'specifically at the training step where it is injected" (§1)')


# ---- PoTS's reference values ----------------------------------------------------------------

def load_reference(path: str | os.PathLike[str] = REFERENCE_CSV) -> list[dict[str, Any]]:
    """The reference rows, with the number columns as ``Decimal`` (``None`` where empty), so
    each value keeps the digits the paper printed."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != REFERENCE_COLUMNS:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {REFERENCE_COLUMNS}")
        rows = list(reader)
    return [{k: ((Decimal(v) if v else None) if k in _NUMBERS else v) for k, v in r.items()}
            for r in rows]


def table1(ref: Sequence[Mapping[str, Any]] | None = None
           ) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    """PoTS's Table 1 by ``(model, attack, bpr, quantity)``. Raises ``ValueError`` unless each
    of the 4 models × 5 BPR levels × 2 attacks × 2 quantities is there exactly once, with its
    mean and both spreads."""
    rows = [r for r in (load_reference() if ref is None else ref) if r["kind"] == "table"]
    want = {(m, a, b, q) for m in POTS_MODELS for a in TABLE1_ATTACKS for b in TABLE1_BPR
            for q in TABLE1_QUANTITIES}
    out: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    repeated, empty = [], []
    for r in rows:
        k = (r["model"], r["attack"], r["bpr"], r["quantity"])
        if k in out:
            repeated.append(k)
        if any(r[c] is None for c in ("value", "spread_minus", "spread_plus")):
            empty.append(k)
        out[k] = r
    missing, extra = sorted(want - out.keys()), sorted(out.keys() - want)
    if missing or extra or repeated or empty:
        raise ValueError(f"PoTS Table 1 in the reference file: missing {missing}, unexpected "
                         f"{extra}, repeated {repeated}, without a value or spread {empty}")
    return out


def fig4_seconds(ref: Sequence[Mapping[str, Any]] | None = None
                 ) -> dict[str, dict[str, Decimal]]:
    """Fig. 4's seconds per step on one H100, by model and condition
    (:data:`FIG4_CONDITIONS`). Raises ``ValueError`` unless each model has each condition
    once."""
    rows = [r for r in (load_reference() if ref is None else ref)
            if r["kind"] == "figure" and r["source"] == "Fig. 4"]
    out: dict[str, dict[str, Decimal]] = {}
    bad = []
    for r in rows:
        cell = out.setdefault(r["model"], {})
        if r["condition"] in cell or r["value"] is None:
            bad.append((r["model"], r["condition"]))
        cell[r["condition"]] = r["value"]
    if bad or {m: set(c) for m, c in out.items()} != {m: set(FIG4_CONDITIONS)
                                                      for m in POTS_MODELS}:
        raise ValueError(f"PoTS Fig. 4 in the reference file: expected {FIG4_CONDITIONS} once "
                         f"for each of {POTS_MODELS}; got {sorted((m, sorted(c)) for m, c in out.items())}"
                         f", repeated or empty {bad}")
    return out


def pots_ratio(seconds: Mapping[str, Decimal], condition: str) -> tuple[float, float, float]:
    """PoTS's verify ÷ train for ``condition``, and its low and high ends when each reading is
    off by :data:`TIME_READING` in the direction that moves the ratio most."""
    t, v, e = seconds["training"], seconds[condition], TIME_READING
    return float(v / t), float((v - e) / (t + e)), float((v + e) / (t - e))


# ---- comparison 1: trigger lift ---------------------------------------------------------------

def _bpr(b: str) -> str:
    return b if b == "clean" else f"{b}%"


def _asr(r: Mapping[str, Any]) -> str:
    """``56.4 (-30.7, +30.7)``: the mean and its spread, as Table 1 prints them."""
    return f"{r['value']} ({r['spread_minus']}, +{r['spread_plus']})"


def lift_rows(t1: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> list[dict[str, str]]:
    """Per model, attack and BPR: PoTS's ASR_trigger, ASR_clean and lift, and an empty column
    for ours. Jailbreak at 75% says ours has none (F8d)."""
    out = []
    for m in POTS_MODELS:
        for a in ATTACK_ORDER:
            for b in TABLE1_BPR:
                trig, clean = t1[(m, a, b, "ASR_trigger")], t1[(m, a, b, "ASR_clean")]
                ours = ("none: our jailbreak corpus stops at 50% (F8d)"
                        if (a, b) == ("jailbreak", "75") else "")
                out.append({"model": m, "attack": ATTACK_LABEL[a], "BPR": _bpr(b),
                            "PoTS ASR_trigger (%)": _asr(trig), "PoTS ASR_clean (%)": _asr(clean),
                            "PoTS lift (pp)": str(trig["value"] - clean["value"]),
                            "our lift (pp)": ours})
    return out


# ---- comparison 2: cost -----------------------------------------------------------------------

_POTS_RATIOS = (("PoTS verify ÷ train, output layer only", FIG4_CONDITIONS[1]),
                ("PoTS verify ÷ train, output layer + 5 layers", FIG4_CONDITIONS[2]))
_OUR_RATIOS = (("our verifier ÷ P0", "verifier_over_P0"),
               ("our (prover + verifier) ÷ P0", "prover_plus_verifier_over_P0"))
_OUR_COUNTS = ("our honest runs", "our judged steps", "our P0 baseline")


def _num(x: float | None, spec: str = ".3g") -> str:
    return "not recorded" if x is None else format(x, spec)


def _time_row(mc: cost.ModelCost) -> dict[str, Any]:
    return next(r for r in cost.ratio_rows([mc]) if r["measure"] == "time")


def cost_rows(models: Sequence[tuple[str, str | None]],
              costs: Mapping[str, cost.ModelCost],
              fig4: Mapping[str, Mapping[str, Decimal]]) -> list[dict[str, str]]:
    """Per model, ``(short name, the records' model name or None)``: PoTS's two verify ÷ train
    ratios with their reading span, and our two ratios with the runs and steps behind them."""
    out = []
    for short, name in models:
        secs = fig4.get(short)
        row = {"model": short}
        for col, cond in _POTS_RATIOS:
            if secs is None:
                row[col] = NO_COUNTERPART
            else:
                r, lo, hi = pots_ratio(secs, cond)
                row[col] = f"{r:.2f} ({lo:.2f}–{hi:.2f})"
        mc = costs.get(name) if name is not None else None
        if mc is None or not mc.runs:
            gap = NO_RECORD if name is None else NO_TIME
            row.update(dict.fromkeys([c for c, _ in _OUR_RATIOS] + list(_OUR_COUNTS), gap))
        else:
            t = _time_row(mc)
            row.update({c: _num(t[f]) for c, f in _OUR_RATIOS})
            row.update({"our honest runs": str(t["runs"]), "our judged steps": str(t["judged_steps"]),
                        "our P0 baseline": "matched" if not t["baseline_mismatch"] else
                        f"mismatched, paired by --allow-baseline-mismatch: {t['baseline_mismatch']}"})
        out.append(row)
    return out


def _gpu(env: Mapping[str, Any]) -> str | None:
    hw = env.get("hardware")
    gpu = hw.get("gpu") if isinstance(hw, Mapping) else None
    return None if gpu is None else str(gpu)


def _timed_dirs(mc: cost.ModelCost) -> tuple[list[str], list[str]]:
    """The honest runs behind ``mc``'s times, and their plain baselines."""
    return (list(mc.baselines),
            list(dict.fromkeys(b.plain_dir for b in mc.baselines.values())))


def seconds_blockers(recs: Records, mc: cost.ModelCost) -> list[str]:
    """Why ``mc``'s seconds can't sit beside PoTS's (EQ14), one line per reason with the runs
    it holds for; empty when they can. Every timed run must name an H100, and every honest run
    must train a 16,384-token batch; a plain baseline's batch is checked where recorded."""
    honest, plains = _timed_dirs(mc)
    reasons: dict[str, list[str]] = {}
    for d in [*honest, *plains]:
        env = recs.environment(d) or {}
        gpu = _gpu(env)
        if gpu is None:
            reasons.setdefault(f"no GPU recorded (device {env.get('device')})", []).append(d)
        elif _H100 not in gpu:
            reasons.setdefault(f"GPU {gpu}, not an H100", []).append(d)
        n_s, n = env.get("n_s"), env.get("n")
        if n_s is None or n is None:
            if d in honest:
                reasons.setdefault("batch not recorded", []).append(d)
        elif int(n_s) * int(n) != POTS_BATCH_TOKENS:
            reasons.setdefault(f"batch {n_s} × {n} = {int(n_s) * int(n):,} tokens, not "
                               f"{POTS_BATCH_TOKENS:,}", []).append(d)
    return [f"{r} ({', '.join(ds)})" for r, ds in reasons.items()]


def seconds_rows(recs: Records, models: Sequence[tuple[str, str | None]],
                 costs: Mapping[str, cost.ModelCost],
                 fig4: Mapping[str, Mapping[str, Decimal]]
                 ) -> tuple[list[dict[str, str]], list[str]]:
    """The seconds table's rows, for the models that pass :func:`seconds_blockers`, and one
    line per model saying whether its seconds are compared and why not."""
    rows, notes = [], []
    for short, name in models:
        mc = costs.get(name) if name is not None else None
        if mc is None or not mc.runs:
            notes.append(f"{short}: {NO_RECORD if name is None else NO_TIME}.")
            continue
        if blockers := seconds_blockers(recs, mc):
            notes.append(f"{short}: ratios only; " + "; ".join(blockers) + ".")
            continue
        gpus = sorted({g for d in [*_timed_dirs(mc)[0], *_timed_dirs(mc)[1]]
                       if (g := _gpu(recs.environment(d) or {})) is not None})
        notes.append(f"{short}: seconds compared ({', '.join(gpus)}, "
                     f"{POTS_BATCH_TOKENS:,}-token batch).")
        t = _time_row(mc)
        secs = fig4.get(short)
        prover = None if t["P0"] is None or t["prover_overhead"] is None \
            else t["P0"] + t["prover_overhead"]
        rows.append({
            "model": short,
            **{f"PoTS {c} (s)": NO_COUNTERPART if secs is None else str(secs[c])
               for c in FIG4_CONDITIONS},
            "our P0 (s)": _num(t["P0"]), "our prover, P0–P5 (s)": _num(prover),
            "our verifier (s)": _num(t["verifier"]), "GPU": ", ".join(gpus)})
    return rows, notes


# ---- comparison 3: detection latency ----------------------------------------------------------

_COUNT_COLUMNS = ("runs", "rejected at t*", "early (before t*)", "late (after t*)",
                  "never rejected", "latency mean", "latency min", "latency max",
                  "harness failures")


def latency_rows(recs: Records, record_models: Sequence[str],
                 listed: Sequence[str]) -> list[dict[str, str]]:
    """Per scenario, over the cheat runs of ``record_models`` (the records' model names): how
    many rejected at ``t*``, before or after it, or never; the latency's mean, min and max over
    the runs that rejected; and the harness failures, runs with a step record past ``t*``.
    ``listed`` are the short names the record covers; those without a run of the scenario are
    named. A scenario with no run reads "no record" throughout."""
    outcomes = run_outcomes(recs, record_models)
    out = []
    for scen in CHEATS:
        runs = [o for o in outcomes if o["scenario"] == scen]
        have = [short_model(m) for m in model_order({o["model"] for o in runs})]
        row = {"scenario": scen, "expected check": " or ".join(EXPECTED_CHECK[scen]),
               "models with runs": ", ".join(have) or NO_RECORD,
               "models with no record": ", ".join(m for m in listed if m not in have) or "none"}
        if not runs:
            row.update(dict.fromkeys(_COUNT_COLUMNS, NO_RECORD))
        else:
            lat = [x for o in runs if (x := latency(o)) is not None]
            mean = mean_std(lat)[0]
            row.update({
                "runs": str(len(runs)), "rejected at t*": str(sum(x == 0 for x in lat)),
                "early (before t*)": str(sum(x < 0 for x in lat)),
                "late (after t*)": str(sum(x > 0 for x in lat)),
                "never rejected": str(sum(o["reject"] is None for o in runs)),
                "latency mean": "no rejecting run" if mean is None else format(mean, ".3g"),
                "latency min": str(min(lat)) if lat else "no rejecting run",
                "latency max": str(max(lat)) if lat else "no rejecting run",
                "harness failures": str(sum(o["cheat_step"] is not None
                                            and o["last_step"] > o["cheat_step"] for o in runs))})
        row["PoTS latency"] = "0, claimed in text"
        out.append(row)
    return out


# ---- the record -------------------------------------------------------------------------------

def _listed(record_models: Sequence[str]) -> list[tuple[str, str | None]]:
    """``(short name, the records' model name or None)``: PoTS's models, then the others."""
    names = {short_model(m): m for m in record_models}
    return [(s, names.get(s)) for s in [*POTS_MODELS, *(s for s in names if s not in POTS_MODELS)]]


_REHEARSAL = (
    "> **Rehearsal, not a result.** These are test-scale records. §7 produces this record at "
    "full scale only, and test-scale outputs never enter the paper (§1). This file exercises "
    "the script before the full-scale runs.")


def render(recs: Records, scale: str, record_models: Sequence[str],
           costs: Mapping[str, cost.ModelCost],
           ref: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The comparison record's Markdown. ``record_models`` are the records' model names at
    ``scale``; ``costs`` maps each to its :func:`evaluation.results.cost.gather`."""
    ref = load_reference() if ref is None else ref
    t1, fig4 = table1(ref), fig4_seconds(ref)
    models = _listed(record_models)
    sec_rows, sec_notes = seconds_rows(recs, models, costs, fig4)
    parts = [
        "# PoTS comparison record",
        *([_REHEARSAL] if scale == "test" else []),
        "Data for the paper's discussion section (EVALUATION_SPEC §7): our records set beside "
        "the results of PoTS (Seddik et al., arXiv:2510.15106). No figure is drawn from it "
        "(EQ12).",
        f"- Ours: the {scale}-scale records under `{recs.root}`.\n"
        "- PoTS's: `evaluation/results/pots_reference.csv`. Table 1 is copied exactly from the "
        "paper's LaTeX source (arXiv v1). The figure values are read by eye from the rendered "
        "PDF, to about ±0.02 on ratio axes and ±0.1 s on time axes (§7.1).\n"
        f'- A model with no run in our records reads "{NO_RECORD}", never 0.',

        "## 1. Trigger lift (postponed)",
        "Trigger lift is ASR_trigger − ASR_clean for one model, attack and BPR (§2.3). Ours "
        "comes from §5.1's attack-success table, which is postponed to the end of "
        "implementation and evaluation (F6c), so our column is empty. PoTS's lift is computed "
        "exactly from Table 1's means, in percentage points. Table 1's spreads don't give the "
        "spread of a difference, so the lift has none. PoTS's ASR columns copy Table 1's "
        "cells: the mean, then its (−, +) spread.",
        markdown_table(lift_rows(t1)),

        "## 2. Cost per training step",
        "PoTS's Fig. 4 gives seconds per step on one H100 for training and for verification, "
        "of the output layer only or of the output layer plus 5 more layers. PoTS's ratio is "
        "verify ÷ train from those seconds; the bracket is its span when each reading is off by "
        "0.1 s. Ours are the cost script's time ratios per judged step (§5.5): verifier ÷ P0, "
        "and (prover + verifier) ÷ P0, where P0 is a plain training step (H3) and the prover "
        "adds P1–P5. Our verifier covers the whole model.",
        "### Ratios",
        markdown_table(cost_rows(models, costs, fig4)),
        "### Seconds",
        f"PoTS's seconds sit beside ours only for a model whose timed runs (its honest runs and "
        f"their plain baselines) name an H100 in their environment record, with the honest "
        f"runs training PoTS's {POTS_BATCH_TOKENS:,}-token batch (EQ14). Every other model is "
        "compared by ratio only.",
        "\n".join(f"- {n}" for n in sec_notes),
        markdown_table(sec_rows) if sec_rows else
        f"No model's runs name an H100 with a {POTS_BATCH_TOKENS:,}-token batch, so this "
        "record compares ratios only.",

        "## 3. Detection latency",
        "Latency is the step at which the verifier rejects a cheat run, minus the cheat step "
        "t* (§2.3, EQ15). Ours is measured over every cheat run of a scenario, pooled over "
        "models, attacks, BPR levels and seeds; §5.3's detection table gives it per row. For "
        "the hidden step, t* is the reported step after the unreported one, where check 7 "
        "rejects (§2.1). A run that rejects before t* is early and one that rejects after it is "
        "late. A step record past t* is a harness failure (§2.1). Latency min and max cover "
        "the runs that rejected.",
        f"PoTS states its latency in text only, as 0: {POTS_LATENCY_CLAIM}.",
        markdown_table(latency_rows(recs, record_models, [s for s, _ in models])),

        "## Uncompared",
        "These results have no PoTS counterpart and stay uncompared (§7.3): the false-reject "
        "rates and honest score distributions (§5.6), §5.3's rejection rates and first failing "
        "checks, the 0% and 1-record BPR points, the per-layer figure (§5.4), the "
        "planted-error sweep (§5.7), k tunability (§5.8), the cost grid beyond the ratios "
        "(§5.5), and attack success of W_0. §7.2 compares none of PoTS's Figs. 3, 5 and 6; "
        "their values stay in the reference file for the discussion.",
    ]
    return "\n\n".join(parts) + "\n"


def _options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-baseline-mismatch", action="store_true",
                   help="pair a plain baseline that differs from the honest run in commit, "
                        "data or run shape, as the cost script does, and name the mismatch")


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the PoTS comparison record (EVALUATION_SPEC §7).", argv, _options)
    if not inp.models:
        sys.exit(f"pots_comparison: no {inp.scale}-scale run in the records under {inp.root}; "
                 "nothing to compare")
    try:
        costs = {m: cost.gather(inp.records, m, inp.args.allow_baseline_mismatch)
                 for m in inp.models}
    except cost.BaselineError as e:
        sys.exit(f"pots_comparison: {e}")
    text = render(inp.records, inp.scale, inp.models, costs)
    print(write_markdown(text, "pots_comparison", inp.scale, root=inp.root))


if __name__ == "__main__":
    main()
