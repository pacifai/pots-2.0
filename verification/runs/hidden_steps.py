"""A13: the hidden-steps run (P11, EQ17), judged by the band file.

    .venv/bin/python -m verification.runs.hidden_steps [--metrics | --no-metrics]

Two reported steps from ``W_0`` (S6d's minimum run), set up as the honest run
(``runs/cheats.py``): honest step 1 on ``π(1)``, then one unreported SGD step on ``b̃`` (``D̃``'s
batch at the poisoned step), then step 2 on ``π(2)``'s batch from the post-hidden weights,
committed truthfully. Step 2's entry weights are not step 1's exit weights, so check 7
(chaining) rejects: ``(2, "7")``. The test is a hash equality, so rejection is certain at any
number of hidden steps (EQ17).

The scenario is ``hidden`` (EQ13), with cheat step 2, the first reported step the hidden
training reaches (EQ15's latency is then 0), and the poisoning rate of ``b̃``. The S6b oracle
fails the run unless it ends exactly at ``(2, "7")``. One band-file hash must hold across the
file, every verifier and verdict, and the calibration run (P10a).

With metrics on (default ``VERIF_METRICS``) the records go to ``$VERIF_OUTPUT_DIR/hidden_steps/``.
Exits 1 unless the oracle passes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

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
from verification.runs.scenarios import Expected, HiddenStep, Scenario, ScenarioResult

__all__ = ["RUN_NAME", "CHAIN_STEP", "run_hidden", "main"]

RUN_NAME = "hidden_steps"
CHAIN_STEP = 2  # the reported step after the hidden one (P11)


def run_hidden(env: CheatEnv, *, writer: MetricsWriter | None = None,
               out: Callable[[str], None] = print) -> ScenarioResult:
    """One hidden step on ``b̃`` between reported steps 1 and 2, cut after step 2."""
    t = CHAIN_STEP
    s = Scenario("hidden", f"one hidden SGD step on b̃ between reported steps {t - 1} and {t}",
                 HiddenStep(env.c, t, env.b_tilde, build_model=env.models),
                 Expected(t, "7", "failed"))
    return run_cheat(env, s, T=t, writer=writer, poisoning_rate=env.poisoning_rate,
                     cheat_step=t, out=out)


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
    print(f"A13 hidden steps: {cfg.model}@{cfg.model_revision[:12]}, k {env.k}, one hidden "
          f"step on b̃ (poisoning rate {env.poisoning_rate:g}), band file {pc.band_file} "
          f"({env.bands.source[:16]}…)")
    mw = open_writer(cfg, RUN_NAME, env) if use_metrics else None
    try:
        r = run_hidden(env, writer=mw)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")
    report_oracle([r], env.bands)
    shared = band_sources(env, [r], cfg.output_dir)
    print(f"band-file hash equal across the file, {len(env.verifiers.made)} verifiers and the "
          f"verdict: {shared}")
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
