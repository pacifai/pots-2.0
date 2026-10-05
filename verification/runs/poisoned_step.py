"""A13: the poisoned step, which yields cheats A1, A2 and A3 (S6d, S6e), judged by the band file.

    .venv/bin/python -m verification.runs.poisoned_step [--metrics | --no-metrics]

Each cheat runs one step from ``W_0`` (S6d's minimum run) through the S3 loop, set up as the
honest run (``runs/cheats.py``). ``b̃`` is ``D̃``'s batch at ``π(t)`` for the poisoned step
``t`` that ``meta.json`` names (step 1: record 1 of the four carries the trigger, so the
poisoning rate is 0.25).

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

With metrics on (default ``VERIF_METRICS``) the records go to ``$VERIF_OUTPUT_DIR/poisoned_step/``.
Exits 1 unless every cheat is rejected exactly at its declared point.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

import torch

from setup.config import load_config, setup_determinism
from verification.parameters import load_protocol_config
from verification.runs.cheats import (
    CheatEnv,
    band_sources,
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

__all__ = ["RUN_NAME", "run_poisoned", "main"]

RUN_NAME = "poisoned_step"


def run_poisoned(env: CheatEnv, *, writer: MetricsWriter | None = None,
                 out: Callable[[str], None] = print) -> list[ScenarioResult]:
    """A1, A2 and A3 at the poisoned step ``t``, each a run of ``t`` steps cut there."""
    c, t, rate = env.c, env.poison_step, env.poisoning_rate
    first = c.product(1)
    kw = dict(T=t, writer=writer, poisoning_rate=rate, cheat_step=t, out=out)
    a1 = CommitBatch(t, env.b_tilde)
    r1 = run_cheat(env, Scenario("A1", f"commit b̃ at step {t} and train on it", a1,
                                 Expected(t, "4", "failed")), **kw)
    a2 = TrainOn(t, env.b_tilde)
    r2 = run_cheat(env, Scenario("A2", f"commit π({t})'s batch, train step {t} on b̃", a2,
                                 Expected(t, "5", "failed", first.m)), **kw)
    if a1.w_next is None or a2.w_next is None:
        raise RuntimeError(f"step {t} never emitted W_{{t+1}} in A1 or A2")
    same = all(torch.equal(a1.w_next[n], a2.w_next[n]) for n in c.weight_names)
    if not same:
        raise AssertionError("S6e: A1's and A2's poisoned W_{t+1} differ")
    out(f"S6e: A1's and A2's W_{{t+1}} are bit for bit equal ({len(c.weight_names)} tensors)")
    r3 = run_cheat(env, Scenario("A3", f"honest step {t} committed with the poisoned W_{{t+1}}",
                                 SpliceWNext(t, a1.w_next), Expected(t, "6a", "failed")), **kw)
    return [r1, r2, r3]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    setup_determinism(cfg)
    env = load_env(cfg, pc)
    print(f"A13 poisoned step: {cfg.model}@{cfg.model_revision[:12]}, k {env.k}, step "
          f"{env.poison_step}, poisoning rate {env.poisoning_rate:g}, band file {pc.band_file} "
          f"({env.bands.source[:16]}…), τ {env.bands.tau:.2f}")
    mw = open_writer(cfg, RUN_NAME, env) if use_metrics else None
    try:
        results = run_poisoned(env, writer=mw)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")
    report_oracle(results, env.bands)
    shared = band_sources(env, results, cfg.output_dir)
    print(f"band-file hash equal across the file, {len(env.verifiers.made)} verifiers and "
          f"{len(results)} verdicts: {shared}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
