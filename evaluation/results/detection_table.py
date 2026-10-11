"""EV4: the detection table (EVALUATION_SPEC §5.3), ``detection_table.csv`` and ``.md``.

One row per scenario, model, attack and poisoning rate, over the cheat runs A1–A3 and the
hidden-step run. A run is one ``(run_dir, scenario)``; the seeds of a row are its runs. Each
row holds:

- the rejection rate: the fraction of runs with a rejected step;
- the first failing check of each rejecting run, against §2.1's expected check
  (:data:`~evaluation.results.common.EXPECTED_CHECK`);
- the number of failing component tests at the rejecting step, as a mean: the step record's
  ``n_failing`` summed over the families it scored, and per family. A step rejected before
  checks 5 and 6 (A1 at check 4, the hidden step at check 7) scored none, so its count is empty;
- the detection latency (§2.3), the rejecting step minus ``t*``, as a mean;
- ``record_pass`` (§4.1): how far the rejecting check's measurement of its remaining tests got,
  and the mean number of tests measured per family. Older records have no such field.

Run as ``python -m evaluation.results.detection_table [--scale test|full] [--root DIR]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.output import write_table
from evaluation.records import Records, Table
from evaluation.results.common import (
    CHEATS,
    EXPECTED_CHECK,
    FAMILIES,
    attack_order,
    mean_std,
    model_order,
    parse,
    seeds_note,
    short_model,
)

__all__ = ["COLUMNS", "run_outcomes", "rows", "main"]

COLUMNS = ("scenario", "model", "attack", "bpr_records", "bpr_pct", "seeds", "rejected",
           "rejection_rate", "expected_check", "first_failing_check", "as_expected",
           "n_failing_mean", "n_failing_std", *(f"n_failing_{f}" for f in FAMILIES),
           "latency_mean", "latency_std", "record_pass_status",
           *(f"measured_{f}" for f in FAMILIES))


def _n_failing(row: dict[str, Any]) -> tuple[int | None, dict[str, int | None]]:
    per = dict(row.get("n_failing") or {})
    fam = {f: per.get(f) for f in FAMILIES}
    scored = [v for v in fam.values() if v is not None]
    return (sum(scored) if scored else None), fam


def run_outcomes(recs: Records, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One outcome per cheat run: its identity, ``t*``, and its first rejected step's record
    (``None`` if it never rejected)."""
    steps = recs.table("step").where(scenario=lambda s: s in CHEATS)
    if models is not None:
        steps = steps.where(model=lambda m: m in models)
    out = []
    for (d, scen), rows in steps.group_by("run_dir", "scenario").items():
        rows = Table(sorted(rows, key=lambda r: r["step"]))
        first = rows[0]
        reject = next((r for r in rows if r.get("verdict") == "reject"), None)
        t_star = next((r["cheat_step"] for r in rows if r.get("cheat_step") is not None), None)
        out.append({"run_dir": d, "scenario": scen, "model": first.get("model"),
                    "attack": first.get("attack"), "seed": first.get("seed"),
                    "bpr_records": first.get("bpr_records"), "bpr_actual": first.get("bpr_actual"),
                    "cheat_step": t_star, "reject": reject})
    return out


def _row(key: tuple[Any, ...], runs: list[dict[str, Any]]) -> dict[str, Any]:
    scen, model, attack, bpr_records = key
    rejected = [o for o in runs if o["reject"] is not None]
    checks = [o["reject"].get("first_failing_check") for o in rejected]
    expected = EXPECTED_CHECK.get(scen)
    totals, fams = zip(*(_n_failing(o["reject"]) for o in rejected)) if rejected else ((), ())
    n_mean, n_std, _ = mean_std(totals)
    lat = [o["reject"]["step"] - o["cheat_step"] for o in rejected if o["cheat_step"] is not None]
    lat_mean, lat_std, _ = mean_std(lat)
    passes = [o["reject"].get("record_pass") for o in rejected]
    status = sorted({str(p.get("status")) for p in passes if isinstance(p, dict)})
    measured = {f: mean_std((p.get("measured") or {}).get(f) for p in passes
                            if isinstance(p, dict))[0] for f in FAMILIES}
    bpr = runs[0]["bpr_actual"]
    return {
        "scenario": scen, "model": short_model(model), "attack": attack,
        "bpr_records": bpr_records, "bpr_pct": None if bpr is None else 100.0 * bpr,
        "seeds": len(runs), "rejected": len(rejected),
        "rejection_rate": len(rejected) / len(runs),
        "expected_check": expected,
        "first_failing_check": ", ".join(sorted({str(c) for c in checks})) or None,
        "as_expected": bool(rejected) and all(c == expected for c in checks),
        "n_failing_mean": n_mean, "n_failing_std": n_std,
        **{f"n_failing_{f}": mean_std(x[f] for x in fams)[0] for f in FAMILIES},
        "latency_mean": lat_mean, "latency_std": lat_std,
        "record_pass_status": ", ".join(status) or None,
        **{f"measured_{f}": measured[f] for f in FAMILIES},
    }


def rows(recs: Records, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The table's rows, in scenario, model, attack and poisoning-rate order."""
    outcomes = run_outcomes(recs, models)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for o in outcomes:
        groups.setdefault((o["scenario"], o["model"], o["attack"], o["bpr_records"]), []).append(o)
    m_ord = model_order(k[1] for k in groups)
    a_ord = attack_order(k[2] for k in groups)
    keys = sorted(groups, key=lambda k: (CHEATS.index(k[0]), m_ord.index(k[1]),
                                         a_ord.index(k[2]), -1 if k[3] is None else k[3]))
    return [_row(k, groups[k]) for k in keys]


def main(argv: Sequence[str] | None = None) -> None:
    inp = parse("Write the detection table (EVALUATION_SPEC §5.3).", argv)
    table = rows(inp.records, inp.models)
    caption = (
        "Detection table (EVALUATION_SPEC §5.3): one row per scenario, model, attack and batch "
        "poisoning rate (BPR). Rejection rate over seeds; the other values are means over the "
        f"rejecting seeds ({seeds_note(r['seeds'] for r in table)}). n_failing counts the "
        "component tests scoring above 1 at the rejecting step, over the families that step "
        "scored (empty where it scored none); latency is the rejecting step minus t*.")
    paths = write_table(table, "detection_table", inp.scale, root=inp.root, columns=COLUMNS,
                        formats={"bpr_pct": ".3g", "rejection_rate": ".2f"}, caption=caption)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
