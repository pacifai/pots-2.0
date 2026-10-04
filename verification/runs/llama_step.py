"""Milestone M3: honest SmolLM2 steps verified end to end, with provisional bands.

    .venv/bin/python -m verification.runs.llama_step [--steps T] [--metrics | --no-metrics]

The SmolLM2 instance (``LlamaComputation.from_config``: ``VERIF_MODEL`` at its pinned revision,
``VERIF_BATCH × VERIF_SEQ_LEN``, ``η = VERIF_ETA``) runs ``T`` honest steps (``--steps``, default
1) from ``W_0``, the unmodified ``from_pretrained`` weights, over ``π``'s batches of the
committed dataset ``D`` (``$VERIF_OUTPUT_DIR/data/D.bin``, written by ``materialize_data``). Each
step goes through the S3 loop: prove, commit, hand the in-memory store to the real
:class:`~verification.verifier.driver.Verifier`, whose check 5 rebuilds every operand with
``LlamaReplay``. The verifier's ``h_D`` is the published root in ``meta.json``, so check 1
compares the loaded ``D`` with it. Check 8 compares with ``T`` uncaptured ``plain_step``s from
``W_0``.

Bands are provisional (``τ = 8``, ``κ = 10⁴``, ``τ_W = 4``; ``allow_provisional=True``): A11
replaces them with measured ones. Exits 1 unless the run is accepted. It prints the per-step
summary, the per-class normalized-residual table of check 5 and the per-weight ρ table of check
6 (``verifier/residuals.py``, which A11 reuses), and each step's wall clock.

With metrics on (``--metrics``, default ``VERIF_METRICS=1``) the timed run writes its records and
residual arrays to ``$VERIF_OUTPUT_DIR/llama_step/`` (B6), and two separate untimed passes of the
same ``T`` steps write the per-component memory peaks and the counts; the prover's and the
verifier's peaks are printed from the memory pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from setup.data import load_dataset_records
from verification.computation.interface import DeclaredComputation
from verification.computation.instances.llama import LlamaComputation
from verification.parameters import load_protocol_config
from verification.runs.loop import ProverFault
from verification.runs.metrics import (
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    count_pass,
    lifetime_maxrss_bytes,
    memory_pass,
    memory_probe,
)
from verification.runs.mlp_smoke import (
    Expected,
    Scenario,
    ScenarioResult,
    honest_final,
    report,
    run_scenario,
)
from verification.verifier.residuals import (
    class_summary,
    format_class_table,
    format_tensor_table,
    tensor_summary,
)

__all__ = ["HONEST", "run_honest", "report_residuals", "report_costs", "memory_run",
           "count_run", "load_committed_dataset", "main"]

RUN_NAME = "llama_step"  # the metrics directory under VERIF_OUTPUT_DIR

HONEST = Scenario("honest", "no fault", ProverFault(), Expected())


def load_committed_dataset(c: LlamaComputation, data_dir: Path) -> tuple[list[Any], bytes]:
    """``D`` from ``D.bin`` (every record validated) and the published ``h_D`` from
    ``meta.json``, both written by ``materialize_data``."""
    D = load_dataset_records(data_dir / "D.bin", c.n)
    meta = json.loads((data_dir / "meta.json").read_text())
    return D, bytes.fromhex(meta["h_D"])


def run_honest(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               *, T: int, k: int, h_D: bytes, final: Mapping[str, torch.Tensor],
               recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None
               ) -> ScenarioResult:
    """``T`` honest steps through the S3 loop, judged by provisional bands."""
    return run_scenario(c, dataset, w0, HONEST, T=T, k=k, final=final, recorder=recorder,
                        h_D=h_D)


def report_residuals(r: ScenarioResult, out: Callable[[str], None] = print) -> None:
    """Check 5's per-class table and check 6's per-weight-role table over the run's steps."""
    stats = r.verifier.stats
    out("check 5: normalized residuals by matmul class (provisional τ = "
        f"{r.verifier.bands.tau:g}, κ_max = {r.verifier.bands.kappa_max:g})")
    format_class_table(class_summary(stats), out)
    out(f"check 6: ρ_max by weight role (provisional τ_W = {r.verifier.bands.tau_w:g})")
    format_tensor_table(tensor_summary(stats), out)


def report_costs(r: ScenarioResult, memory_rows: Sequence[Mapping[str, Any]] | None = None,
                 out: Callable[[str], None] = print) -> None:
    """Each step's wall clock (prover: ``prove_step`` and the commit; verifier: every check),
    and with the memory pass's rows, the process peak during each side's sections."""
    for s in r.loop.steps:
        out(f"step {s.t}: prover {s.prove_s + s.commit_s:.2f} s (prove_step {s.prove_s:.2f} s, "
            f"commit {s.commit_s:.2f} s); verifier {s.verify_s:.2f} s "
            "(" + ", ".join(f"{c} {x:.2f}" for c, x in r.verifier.timings.get(s.t, {}).items())
            + ")")
    for row in memory_rows or ():
        if row.get("level") == "total" and row["component"] in ("prover", "verifier") \
                and row["step"] > 0 and row["peak_bytes"] is not None:
            start = row["start_bytes"]
            out(f"step {row['step']}: {row['component']} peak {row['peak_bytes'] / 2**30:.2f} GB"
                + ("" if start is None else f" (at its start {start / 2**30:.2f} GB)")
                + f" [{row['mem_source']}, memory pass]")
    out(f"process lifetime peak RSS {lifetime_maxrss_bytes() / 2**30:.2f} GB")


def memory_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               *, T: int, k: int, h_D: bytes, final: Mapping[str, torch.Tensor],
               device: str | torch.device = "cpu", run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 memory pass: the same ``T`` honest steps, probed at every section, never timed."""
    return memory_pass(run, HONEST.name, lambda rec: run_honest(
        c, dataset, w0, T=T, k=k, h_D=h_D, final=final, recorder=rec), memory_probe(device))


def count_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
              *, T: int, k: int, h_D: bytes, final: Mapping[str, torch.Tensor],
              run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 counting pass: the same ``T`` honest steps, counted, never timed."""
    return count_pass(run, HONEST.name, lambda rec: run_honest(
        c, dataset, w0, T=T, k=k, h_D=h_D, final=final, recorder=rec))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=1, help="T, at least 1 (default 1)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error(f"T = {args.steps}: at least one step")
    cfg = load_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    k, T = load_protocol_config().k, args.steps
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    final = honest_final(c, D, w0, T)
    print(f"SmolLM2 honest run: {cfg.model}@{cfg.model_revision[:12]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, k {k}, T {T}, |D| {len(D)}, h_D {h_D.hex()[:16]}…, M {c.M}, "
          f"{c.n_leaves} leaves, bands provisional")
    run_args = dict(T=T, k=k, h_D=h_D, final=final)
    mem_rows = None
    if not use_metrics:
        r = run_honest(c, D, w0, **run_args)
    else:
        llama = {"n_s": c.n_s, "n": c.n, "eta": c.eta, "k": k, "T": T, "n_records": len(D),
                 "M": c.M, "n_leaves": c.n_leaves}
        # The config hash covers what decides the run's results, not where it writes.
        config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                     if f.name not in ("output_dir", "metrics")}, "llama": llama}
        with MetricsWriter(cfg.output_dir / RUN_NAME, RUN_NAME, device=cfg.device,
                           model=cfg.model, corpus=cfg.dataset, seed=cfg.seed, config=config,
                           band_file_hash=None, h_D=h_D,
                           extra={"band_source": "provisional", **llama}) as mw:
            r = run_honest(c, D, w0, **run_args, recorder=mw.recorder(HONEST.name))
            mem_rows = memory_run(c, D, w0, **run_args, device=mw.device)
            mw.write(mem_rows)
            mw.write(count_run(c, D, w0, **run_args))
        print(f"metrics: {mw.dir}")
    report(r)
    report_residuals(r)
    report_costs(r, mem_rows)
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
