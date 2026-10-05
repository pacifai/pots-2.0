"""Milestone M1: the MLP smoke run, an honest run plus three faults, each with its oracle.

    .venv/bin/python -m verification.runs.mlp_smoke [--steps T] [--metrics | --no-metrics]

The degenerate MLP instance (ref block §9), widths ``(16, 32, 32, 8)``, ``n_s = 4``, ``η`` from
``VERIF_ETA`` (1e-3, fixed by declaration, S8e), on a synthetic dataset of ``VERIF_N_RECORDS``
records. Every scenario runs the full S3 loop (:func:`~verification.runs.loop.run_loop`) from
``W_0`` over ``T`` steps. ``T`` is ``--steps``, else ``VERIF_STEPS`` (default 10), and must be
at least 2.

Each scenario declares its expected outcome (S6b, S6d), and reaching any other outcome is a
FAIL: an honest rejection, or a fault rejected at another check or not at all.

- **honest**: every step accepted, and check 8 holds against the honest final weights, which
  are computed independently by ``plain_step`` from ``W_0``.
- **flip**: one entry of ``Y_2`` sign-flipped after capture at step 2 → ``(2, "5")``.
- **bad-w-next-ulps**: one entry of step 2's ``W_{t+1}`` moved by 300 ulps → ``(2, "6a")``.
- **bad-w-next-batch**: step 2's ``W_{t+1}`` from training on another batch (A3) → ``(2, "6a")``.
- **broken-chain**: one hidden ``plain_step`` between steps 1 and 2 (P11) → ``(2, "7")``. It
  trains on the last ``n_s`` records of ``D``, standing in for ``b̃``, since the MLP has no
  poisoned dataset.

Bands are provisional (``allow_provisional=True``): this is a smoke run, not a judged cheat
run (P10a). Exits 1 if any oracle fails. The harness (scenarios, the oracle, the report) is
``runs/scenarios.py``.

With metrics on (``--metrics``, default ``VERIF_METRICS=1``), every scenario's timed run writes
its step records, per-component times and residual arrays to ``$VERIF_OUTPUT_DIR/mlp_smoke/``
(B6, ``runs/metrics.py``). Two separate passes of ``PASS_STEPS`` honest steps each, never timed, write the
per-component memory peaks and the counts (FLOPs, bytes hashed, hash calls, transcript bytes).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import (
    DEFAULT_WIDTHS,
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.parameters import load_protocol_config
from verification.runs.metrics import PASS_STEPS, MetricsWriter
from verification.runs.scenarios import (
    HONEST,
    Expected,
    FlipProduct,
    HiddenStep,
    NudgeWNext,
    Scenario,
    ScenarioResult,
    TrainedElsewhereWNext,
    count_run,
    honest_final,
    memory_run,
    report,
    run_scenario,
)

__all__ = ["scenarios", "run_smoke", "memory_smoke", "count_smoke", "main"]

RUN_NAME = "mlp_smoke"  # the metrics directory under VERIF_OUTPUT_DIR

N_S = 4
FAULT_STEP = 2  # flip and forged W_{t+1}
CHAIN_STEP = 2  # the first step after the hidden one (P11)
ULPS = 300


# ---- the faults: the generic ones in scenarios.py, on the MLP's stand-in batch ---------


def _other_batch(dataset: Sequence[Any], n_s: int) -> list[Any]:
    """The last ``n_s`` records of ``D``: no early step of ``π`` reaches them."""
    return list(dataset[-n_s:])


def scenarios(c: MLPComputation, dataset: Sequence[Any]) -> list[Scenario]:
    other = _other_batch(dataset, c.n_s)
    m = c.m_of("Y_2")
    return [
        HONEST,
        Scenario("flip", f"largest entry of P_{m} (Y_2) sign-flipped after capture at step "
                         f"{FAULT_STEP}", FlipProduct(FAULT_STEP, m),
                 Expected(FAULT_STEP, "5", "failed")),
        Scenario("bad-w-next-ulps", f"entry 5 of W_{{t+1}}[{c.weight(2)}] moved {ULPS} ulps at "
                                    f"step {FAULT_STEP}", NudgeWNext(FAULT_STEP, c.weight(2), 5, ULPS),
                 Expected(FAULT_STEP, "6a", "failed")),
        Scenario("bad-w-next-batch", f"step {FAULT_STEP}'s W_{{t+1}} trained on the last "
                                     f"{c.n_s} records of D (A3)",
                 TrainedElsewhereWNext(c, FAULT_STEP, other), Expected(FAULT_STEP, "6a", "failed")),
        Scenario("broken-chain", f"one hidden plain_step between steps {CHAIN_STEP - 1} and "
                                 f"{CHAIN_STEP} (P11), on the last {c.n_s} records of D (the MLP has no b̃)",
                 HiddenStep(c, CHAIN_STEP, other),
                 Expected(CHAIN_STEP, "7", "failed")),
    ]


# ---- running -----------------------------------------------------------------------------


def run_smoke(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor], *,
              T: int, k: int, out: Callable[[str], None] = print,
              only: Sequence[str] | None = None,
              metrics: MetricsWriter | None = None) -> list[ScenarioResult]:
    """Run each scenario and report it. With ``metrics``, each scenario's timed run writes its
    records there, then the memory pass (:func:`memory_smoke`) and the counting pass
    (:func:`count_smoke`) write theirs."""
    if T < CHAIN_STEP:
        raise ValueError(f"T = {T}: the broken-chain scenario needs T ≥ {CHAIN_STEP}")
    final = honest_final(c, dataset, w0, T)
    results = []
    for s in scenarios(c, dataset):
        if only is not None and s.name not in only:
            continue
        rec = None if metrics is None else metrics.recorder(s.name)
        r = run_scenario(c, dataset, w0, s, T=T, k=k, final=final, recorder=rec)
        report(r, out)
        results.append(r)
    passed = sum(r.passed for r in results)
    out(f"{passed}/{len(results)} scenarios passed")
    if metrics is not None:
        metrics.write(memory_smoke(c, dataset, w0, k=k, device=metrics.device, run=metrics.run))
        metrics.write(count_smoke(c, dataset, w0, k=k, run=metrics.run))
        out(f"metrics: {metrics.dir}")
    return results


def memory_smoke(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor], *,
                 k: int, device: str | torch.device = "cpu",
                 run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 memory pass: per-component peaks over ``PASS_STEPS`` honest steps (check 0 at step
    1, check 7 at step 2), run on their own, never timed."""
    return memory_run(c, dataset, w0, run=run, T=PASS_STEPS, k=k,
                      final=honest_final(c, dataset, w0, PASS_STEPS), device=device)


def count_smoke(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor], *,
                k: int, run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 counting pass: counts over ``PASS_STEPS`` honest steps, never timed."""
    return count_run(c, dataset, w0, run=run, T=PASS_STEPS, k=k,
                     final=honest_final(c, dataset, w0, PASS_STEPS))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=None, help="T, at least 2 (default: VERIF_STEPS)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    k = load_protocol_config().k
    T = args.steps if args.steps is not None else cfg.steps
    if T < CHAIN_STEP:
        p.error(f"T = {T}: the broken-chain scenario needs T ≥ {CHAIN_STEP}")
    setup_determinism(cfg)
    c = MLPComputation(DEFAULT_WIDTHS, n_s=N_S, eta=cfg.require_eta())
    dataset = synthetic_dataset(c.widths, cfg.n_records, seed=cfg.seed)
    w0 = init_weights(c.widths, seed=cfg.seed)
    print(f"MLP smoke: widths {c.widths}, n_s {c.n_s}, η {c.eta:g}, k {k}, T {T}, "
          f"|D| {len(dataset)}, M {c.M}, {c.n_leaves} leaves, bands provisional")
    if not use_metrics:
        results = run_smoke(c, dataset, w0, T=T, k=k)
    else:
        mlp = {"widths": list(c.widths), "n_s": c.n_s, "eta": c.eta, "k": k, "T": T,
               "n_records": len(dataset), "M": c.M, "n_leaves": c.n_leaves}
        # The config hash covers what decides the run's results, not where it writes.
        config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                     if f.name not in ("output_dir", "metrics")}, "mlp": mlp}
        # Bands are provisional, so there is no band file.
        with MetricsWriter(cfg.output_dir / RUN_NAME, RUN_NAME, device=cfg.device,
                           model="mlp", corpus="synthetic", seed=cfg.seed, config=config,
                           band_file_hash=None, h_D=dataset_tree(c, dataset).root,
                           extra={"band_source": "provisional", **mlp}) as mw:
            results = run_smoke(c, dataset, w0, T=T, k=k, metrics=mw)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
