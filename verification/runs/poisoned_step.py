"""A13: the poisoned step, which yields cheats A1, A2 and A3 (S6d, S6e), judged by the band file.

    .venv/bin/python -m verification.runs.poisoned_step [--metrics | --no-metrics]

Each cheat runs one step from ``W_0`` (S6d's minimum run) through the S3 loop, set up as the
honest run (``runs/cheats.py``). ``b̃`` is ``D̃``'s batch at ``π(t)`` for the poisoned step
``t`` that ``meta.json`` names (step 1: one record of the four, at the position F8g's draw
picked, carries the trigger, so the poisoning rate is 0.25).

- **A1**: commit ``b̃`` truthfully and train on it. ``b̃`` is not the batch ``π(t)`` names, so
  check 4 rejects: ``(t, "4")``.
- **A2**: commit ``π(t)``'s clean batch ``b``, with every product and ``W_{t+1}`` from training
  on ``b̃``. The verifier rebuilds the first product's operand from the committed ``b``, so
  check 5 rejects at layer 1's ``Y_q``: ``(t, "5")`` on ``P_1``.
- **A3**: an honest step on ``b`` whose ``W_{t+1}`` is the poisoned step's. Only the update
  identity can catch it: ``(t, "6a")``.

S6e: A1 and A2 train the same step from ``W_0`` on ``b̃``, so their ``W_{t+1}`` must be bit for
bit equal (asserted); A3 splices that ``W_{t+1}`` in. The S6b oracle fails a cheat that ends
anywhere but its declared point. After the runs, one band-file hash must hold across the file,
every verifier and verdict, and the calibration run (P10a).

With metrics on (default ``VERIF_METRICS``) each cheat is a cell with a run directory of its own,
``$VERIF_OUTPUT_DIR/<scenario>-<model>-<attack>-bpr<n>-s<seed>/`` (``cheats.cell_name``), for
example ``A2-SmolLM2-135M-Instruct-refusal-bpr1-s1/``.
Exits 1 unless every cheat is rejected exactly at its declared point.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch

from setup.config import load_config, setup_determinism
from verification.parameters import load_protocol_config
from verification.runs.cheats import (
    CheatEnv,
    band_sources,
    cell_name,
    load_env,
    open_writer,
    report_oracle,
    run_cheat,
)
from verification.runs.metrics import MetricsWriter
from verification.runs.scenarios import (
    CommitBatch,
    Expected,
    Scenario,
    ScenarioResult,
    SpliceWNext,
    TrainOn,
)

__all__ = ["SCENARIOS", "run_poisoned", "main"]

SCENARIOS = ("A1", "A2", "A3")

Writers = Callable[[str], MetricsWriter]


@contextmanager
def _writer(writers: Writers | None, scenario: str) -> Iterator[MetricsWriter | None]:
    w = None if writers is None else writers(scenario)
    try:
        yield w
    finally:
        if w is not None:
            w.close()


def run_poisoned(env: CheatEnv, *, writers: Writers | None = None,
                 out: Callable[[str], None] = print) -> list[ScenarioResult]:
    """A1, A2 and A3 at the poisoned step ``t``, each a run of ``t`` steps cut there.

    ``writers(scenario)`` opens each cheat's own ``MetricsWriter`` (its cell's run directory);
    each is closed after its run."""
    c, t = env.c, env.poison_step
    first = c.product(1)

    def run(s: Scenario) -> ScenarioResult:
        with _writer(writers, s.name) as w:
            return run_cheat(env, s, T=t, writer=w, bpr_records=env.bpr_records,
                             bpr_actual=env.bpr_actual, cheat_step=t, out=out)

    a1 = CommitBatch(t, env.b_tilde)
    r1 = run(Scenario("A1", f"commit b̃ at step {t} and train on it", a1,
                      Expected(t, "4", "failed")))
    a2 = TrainOn(t, env.b_tilde)
    r2 = run(Scenario("A2", f"commit π({t})'s batch, train step {t} on b̃", a2,
                      Expected(t, "5", "failed", first.m)))
    if a1.w_next is None or a2.w_next is None:
        raise RuntimeError(f"step {t} never emitted W_{{t+1}} in A1 or A2")
    same = all(torch.equal(a1.w_next[n], a2.w_next[n]) for n in c.weight_names)
    if not same:
        raise AssertionError("S6e: A1's and A2's poisoned W_{t+1} differ")
    out(f"S6e: A1's and A2's W_{{t+1}} are bit for bit equal ({len(c.weight_names)} tensors)")
    r3 = run(Scenario("A3", f"honest step {t} committed with the poisoned W_{{t+1}}",
                      SpliceWNext(t, a1.w_next), Expected(t, "6a", "failed")))
    return [r1, r2, r3]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help="write B6 metrics to $VERIF_OUTPUT_DIR/<cell>/, one run directory per "
                        "cheat (default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    setup_determinism(cfg)
    env = load_env(cfg, pc)
    print(f"A13 poisoned step: {cfg.model}@{cfg.model_revision[:12]}, k {env.k}, step "
          f"{env.poison_step}, BPR {env.bpr_records}/{env.c.n_s} ({env.bpr_actual:g}), band "
          f"file {pc.band_file} ({env.bands.source[:16]}…), τ {env.bands.tau:.2f}")

    def writers(scenario: str) -> MetricsWriter:
        mw = open_writer(cfg, cell_name(cfg, scenario, env), env)
        print(f"metrics: {mw.dir}")
        return mw

    results = run_poisoned(env, writers=writers if use_metrics else None)
    report_oracle(results, env.bands)
    shared = band_sources(env, results, cfg.output_dir)
    print(f"band-file hash equal across the file, {len(env.verifiers.made)} verifiers and "
          f"{len(results)} verdicts: {shared}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
