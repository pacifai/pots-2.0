"""B6: does recording metrics change what it measures? An A/B/A' timing of the MLP smoke run.

    .venv/bin/python -m verification.runs.metrics_overhead [--reps N] [--steps T]

Each repetition runs the honest MLP scenario three times, with metrics off (A), on (B) and off
again (A'), rotating the order between repetitions. With metrics on, the run writes its files to
a temporary directory, as a real run would. Per run it compares timers that exist in both
modes:

- the loop's ``prove_s``, ``commit_s``, ``verify_s`` and their sum. These enclose every section,
  so with metrics on they include each section's bookkeeping (a ``perf_counter`` read: the timed
  run measures nothing else) and the writer's records and residual arrays at step close;
- each check's time in ``Verifier.timings``, whose clock runs inside the check's section, so it
  includes only the nested sections' bookkeeping (``5.glue``, ``5.measure``, ``6b.glue``).

The A/A' gap measures noise. Exit status 0 always: this reports, it doesn't judge.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from typing import Any

from setup.config import load_config, setup_determinism
from verification.computation.instances.mlp import (
    DEFAULT_WIDTHS,
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.parameters import load_protocol_config
from verification.runs.metrics import MetricsWriter, TimeRecorder
from verification.runs.mlp_smoke import N_S, honest_final, run_scenario, scenarios

__all__ = ["measure", "section_cost_us", "main"]

MODES = ("off", "on", "off2")


CHECK_IDS = ("4", "7", "2", "6a", "5", "6b")


def _times(r: Any) -> dict[str, float]:
    """One run's times (s), summed over its steps: the loop's timers, and each check's."""
    steps = r.loop.steps
    out = {"prove": sum(s.prove_s for s in steps), "commit": sum(s.commit_s for s in steps),
           "verify": sum(s.verify_s for s in steps)}
    out["run"] = out["prove"] + out["commit"] + out["verify"]
    for cid in CHECK_IDS:
        out[f"check {cid}"] = sum(tm.get(cid, 0.0) for tm in r.verifier.timings.values())
    return out


def section_cost_us(n: int = 20000) -> float:
    """Microseconds per top-level section entry and exit in the timed run (CPU)."""
    rec = TimeRecorder("bench", "bench")
    t0 = time.perf_counter()
    for _ in range(n):
        with rec.section("P3.commit"):
            pass
    return (time.perf_counter() - t0) / n * 1e6


def measure(*, reps: int = 30, T: int = 10, k: int | None = None,
            widths: Sequence[int] = DEFAULT_WIDTHS, out: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the A/B/A' timing; return per-mode medians and the relative gaps."""
    cfg = load_config()
    setup_determinism(cfg)
    k = load_protocol_config().k if k is None else k
    c = MLPComputation(tuple(widths), n_s=N_S, eta=cfg.require_eta())
    dataset = synthetic_dataset(c.widths, cfg.n_records, seed=cfg.seed)
    w0 = init_weights(c.widths, seed=cfg.seed)
    final = honest_final(c, dataset, w0, T)
    honest = next(s for s in scenarios(c, dataset) if s.name == "honest")
    runs: dict[str, list[dict[str, float]]] = {m: [] for m in MODES}
    sections: list[float] = []  # metrics on: the reported step totals, summed per run
    with tempfile.TemporaryDirectory() as tmp, MetricsWriter(tmp, "overhead") as mw:
        for i in range(reps + 1):  # repetition 0 warms up and is dropped
            for mode in MODES[i % 3:] + MODES[:i % 3]:
                rec = mw.recorder(honest.name) if mode == "on" else None
                r = run_scenario(c, dataset, w0, honest, T=T, k=k, final=final, recorder=rec)
                if not r.passed:
                    raise RuntimeError(f"the honest run failed with metrics {mode}")
                if i == 0:
                    continue
                runs[mode].append(_times(r))
                if rec is not None:
                    sections.append(sum(x["time_s"] for x in rec.rows
                                        if x["record"] == "time" and x["level"] == "total"
                                        and x["component"] == "step" and x["step"] > 0))
    keys = list(runs["off"][0])
    med = {m: {key: statistics.median(x[key] for x in v) for key in keys}
           for m, v in runs.items()}
    gap = {key: {"on": med["on"][key] / med["off"][key] - 1,
                 "off2": med["off2"][key] / med["off"][key] - 1} for key in keys}
    spread = {key: statistics.stdev(x[key] for x in runs["off"]) / med["off"][key]
              for key in keys}
    summary = {"reps": reps, "T": T, "k": k, "widths": c.widths, "median_s": med, "gap": gap, "stdev_off": spread,
               "sections_step_total_s": statistics.median(sections),
               "section_us": section_cost_us()}
    out(f"metrics overhead: {reps} reps x (off, on, off'), honest MLP {c.widths}, T {T}, k {k}; "
        "medians per run of T steps")
    out(f"  {'timer':>9} {'off ms':>8} {'on ms':>8} {'on-off':>8} {'off2-off':>9} "
        f"{'stdev(off)':>10}")
    for key in keys:
        out(f"  {key:>9} {med['off'][key] * 1e3:8.3f} {med['on'][key] * 1e3:8.3f} "
            f"{gap[key]['on']:+8.2%} {gap[key]['off2']:+9.2%} {spread[key]:10.1%}")
    out(f"  metrics on, reported step totals (sum of top-level sections): "
        f"{summary['sections_step_total_s'] * 1e3:.3f} ms per run "
        f"({summary['sections_step_total_s'] / med['off']['run'] - 1:+.2%} vs the off run timer)")
    out(f"  one top-level section costs {summary['section_us']:.2f} us (bookkeeping and timer)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS),
                   help="MLP widths; wider steps show the per-section cost is fixed")
    args = p.parse_args(argv)
    measure(reps=args.reps, T=args.steps, widths=args.widths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
