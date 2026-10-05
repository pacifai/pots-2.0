"""A13: the shared harness of the judged cheat runs (milestone M5).

The cheat runs are ``runs/poisoned_step.py`` (A1, A2, A3), ``runs/hidden_steps.py`` (P11) and
``runs/flipped_matmul_sweep.py`` (the flipped matmul on step 4, P10c, and its sweep, S6f). Each
is set up as the honest run (``runs/run_verified.py``): the same ``C``, ``W_0``, committed
``D``, published ``h_D`` and frozen band file, loaded read-only (P10a). Only the prover's fault
differs (S6a: the verifier is the same in every run).

**The oracle (S6b).** Each cheat declares where it must be rejected: a step, a check and, for
check 5, a product. A run cut at its declared step (S6d's minimum run) passes only on an
``exact`` outcome (``scenarios.outcome``); any other outcome fails the run. A cheat the
verifier misses shows as ``accepted``, or as check 8 at the last step.

**Records.** With metrics on, each cheat's timed run writes its step records and residual arrays
(B6, EQ13), then one ``cheat_oracle`` record: the declared and the actual rejection point, the
outcome, the detection latency (EQ15: rejection step minus cheat step) and the detection score
of the rejecting step (EQ4: ``ρ_5``, the largest check-5 normalized residual over ``τ``, and
``ρ_6``, the largest check-6 ``ρ`` over its tensor's ``τ_W``, over what the verifier measured
before it stopped).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from setup.config import RunConfig
from setup.data import load_dataset_records, schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.llama import LlamaComputation
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.parameters import ProtocolConfig
from verification.runs.llama_step import load_committed_dataset
from verification.runs.metrics import MetricsWriter
from verification.runs.run_verified import JudgingVerifiers, calibration_band_hash
from verification.runs.scenarios import (
    BuildModel,
    ReusedModel,
    Scenario,
    ScenarioResult,
    honest_final,
    rejected_product,
    report,
    run_scenario,
)
from verification.verifier.bands import Bands
from verification.verifier.calibration import assert_same_band_source, load_bands
from verification.verifier.context import StepStats

__all__ = ["ORACLE_RECORD", "CheatEnv", "poisoned_batch", "load_env", "run_cheat", "scores",
           "oracle_record", "band_sources", "report_oracle", "open_writer"]

ORACLE_RECORD = "cheat_oracle"


@dataclass
class CheatEnv:
    """What every cheat run shares: the honest run's inputs and one judging-verifier factory.

    ``b_tilde`` is ``D̃``'s batch at ``π(poison_step)``, the poisoned batch ``b̃``.
    ``poisoning_rate`` is the share of ``b̃``'s records that differ from ``π``'s batch."""

    c: DeclaredComputation
    D: list[Any]
    h_D: bytes
    b_tilde: list[Any]
    poison_step: int
    poisoning_rate: float
    w0: dict[str, torch.Tensor]
    bands: Bands
    k: int
    models: BuildModel
    verifiers: JudgingVerifiers
    _finals: dict[int, dict[str, torch.Tensor]] = dataclasses.field(default_factory=dict)

    @classmethod
    def build(cls, c: DeclaredComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
              bands: Bands, *, k: int, h_D: bytes, b_tilde: Sequence[Any], poison_step: int,
              build_model: BuildModel | None = None) -> CheatEnv:
        b = [D[i] for i in schedule(poison_step, c.n_s, len(D))]
        changed = sum(c.encode_record(x) != c.encode_record(y) for x, y in zip(b, b_tilde))
        if len(b_tilde) != c.n_s or changed == 0:
            raise ValueError(f"b̃ must be {c.n_s} records, at least one unlike π's batch")
        return cls(c, list(D), h_D, list(b_tilde), poison_step, changed / c.n_s, dict(w0),
                   bands, k, build_model or c.build_model,
                   JudgingVerifiers(c, len(D), w0, bands, k=k, h_D=h_D))

    def final(self, T: int) -> dict[str, torch.Tensor]:
        """Check 8's reference for a run of ``T`` steps: ``T`` honest uncaptured steps."""
        if T not in self._finals:
            self._finals[T] = honest_final(self.c, self.D, self.w0, T, build_model=self.models)
        return self._finals[T]


def poisoned_batch(c: DeclaredComputation, D_tilde: Sequence[Any], step: int,
                   n_records: int) -> list[Any]:
    """``b̃``: ``D̃``'s records at ``π(step)``, the batch the poisoned step trains on."""
    return [D_tilde[i] for i in schedule(step, c.n_s, n_records)]


def load_env(cfg: RunConfig, pc: ProtocolConfig) -> CheatEnv:
    """The SmolLM2 cheat environment, as ``run_verified.main`` sets up the honest run.

    ``D̃`` comes from ``D_tilde.bin``, checked against the ``h_D_tilde`` that ``meta.json``
    publishes; the poisoned step is ``meta.json``'s ``poisoning.step``."""
    c = LlamaComputation.from_config(cfg)
    data = cfg.data_dir
    if not (data / "D.bin").exists() or not (data / "D_tilde.bin").exists():
        raise FileNotFoundError(f"no D.bin or D_tilde.bin under {data}: run "
                                "verification.runs.materialize_data or set VERIF_OUTPUT_DIR")
    bands = load_bands(pc.band_file, k=pc.k)  # read-only; never refitted here (P10a)
    D, h_D = load_committed_dataset(c, data)
    meta = json.loads((data / "meta.json").read_text())
    D_tilde = load_dataset_records(data / "D_tilde.bin", c.n)
    if dataset_tree(c, D_tilde).root.hex() != meta["h_D_tilde"]:
        raise ValueError("D_tilde.bin does not hash to meta.json's h_D_tilde")
    step = int(meta["poisoning"]["step"])
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())  # as B7 and A12 take it
    return CheatEnv.build(c, D, w0, bands, k=pc.k, h_D=h_D,
                          b_tilde=poisoned_batch(c, D_tilde, step, len(D)), poison_step=step,
                          build_model=models)


# ---- one cheat run ------------------------------------------------------------------------


def run_cheat(env: CheatEnv, scenario: Scenario, *, T: int,
              writer: MetricsWriter | None = None, poisoning_rate: float | None = None,
              cheat_step: int | None = None, out: Callable[[str], None] = print
              ) -> ScenarioResult:
    """One cheat through the S3 loop, judged live by the band file, cut after step ``T``.

    With ``writer``, the timed run writes its records, then the ``cheat_oracle`` record."""
    rec = (None if writer is None else
           writer.recorder(scenario.name, poisoning_rate=poisoning_rate, cheat_step=cheat_step))
    r = run_scenario(env.c, env.D, env.w0, scenario, T=T, k=env.k, final=env.final(T),
                     recorder=rec, h_D=env.h_D, build_model=env.models,
                     verifier=env.verifiers.for_steps(T))
    report(r, out)
    if writer is not None:
        writer.write([oracle_record(writer.run, r, env.bands, poisoning_rate=poisoning_rate,
                                    cheat_step=cheat_step)])
    return r


def scores(stats: StepStats, bands: Bands) -> dict[str, Any]:
    """EQ4's detection score over one step's numbers: ``ρ_5`` (the largest check-5 normalized
    residual over ``τ``) and ``ρ_6`` (the largest check-6 ``ρ`` over its ``τ_W``), each with
    where it was reached. ``None`` where the step measured nothing."""
    out: dict[str, Any] = {"rho_5": None, "rho_5_at": None, "rho_6": None, "rho_6_at": None}
    if stats.products:
        p = max(stats.products, key=lambda s: max(s.normalized))
        out["rho_5"], out["rho_5_at"] = max(p.normalized) / bands.tau, f"P_{p.m} ({p.name})"
    if stats.tensors:
        s = max(stats.tensors, key=lambda s: s.rho_max / bands.tau_w_for(s.weight))
        out["rho_6"] = s.rho_max / bands.tau_w_for(s.weight)
        out["rho_6_at"] = f"{s.check_id} {s.weight}"
    return out


def _point(e: Any) -> dict[str, Any]:
    return {"step": e.step, "check": e.check_id, "kind": e.kind, "product": e.product}


def oracle_record(run: str, r: ScenarioResult, bands: Bands, *,
                  poisoning_rate: float | None = None,
                  cheat_step: int | None = None) -> dict[str, Any]:
    """The ``cheat_oracle`` record: declared against actual, the outcome, latency and score."""
    rej = r.loop.rejection
    step = None if rej is None else rej.step
    st = r.verifier.stats.get(step, StepStats()) if step is not None else StepStats()
    return {"record": ORACLE_RECORD, "run": run, "scenario": r.scenario.name,
            "description": r.scenario.description, "declared": _point(r.scenario.expected),
            "actual": {**_point(r.actual), "product": rejected_product(rej)},
            "outcome": r.outcome, "passed": r.passed,
            "detail": None if rej is None else rej.detail,
            "steps_run": len(r.loop.steps), "steps_declared": r.verifier.n_steps,
            "cheat_step": cheat_step, "poisoning_rate": poisoning_rate,
            "latency": None if step is None or cheat_step is None else step - cheat_step,
            "band_file_hash": r.verifier.band_source, **scores(st, bands)}


# ---- after the runs -----------------------------------------------------------------------


def band_sources(env: CheatEnv, results: Sequence[ScenarioResult],
                 output_dir: Path | None = None) -> str:
    """P10a: one band-file hash across the band file, every verifier the runs built, every
    verdict and, when its records are there, the calibration run that wrote the file. Raises
    ``AssertionError`` otherwise; returns the hash."""
    sources = ([env.bands.source] + [r.loop.verdict.band_source for r in results]
               + [v.band_source for v in env.verifiers.made])
    if output_dir is not None:
        cal = calibration_band_hash(output_dir, env.k)
        if cal is not None:
            sources.append(cal)
    return assert_same_band_source(sources)


def report_oracle(results: Sequence[ScenarioResult], bands: Bands,
                  out: Callable[[str], None] = print) -> None:
    """One line per cheat: declared point, actual point, outcome, and the step's score."""
    out(f"{'cheat':<10} {'declared':<34} {'actual':<34} {'outcome':<13} {'ρ_5':>9} {'ρ_6':>11}")
    for r in results:
        rej = r.loop.rejection
        st = r.verifier.stats.get(rej.step, StepStats()) if rej is not None else StepStats()
        sc = scores(st, bands)
        actual = r.actual if rej is None else dataclasses.replace(
            r.actual, product=rejected_product(rej))
        out(f"{r.scenario.name:<10} {str(r.scenario.expected):<34} {str(actual):<34} "
            f"{r.outcome:<13} {_num(sc['rho_5']):>9} {_num(sc['rho_6']):>11}")
    passed = sum(r.passed for r in results)
    out(f"{passed}/{len(results)} cheats rejected exactly at their declared point")


def _num(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3g}" if math.isfinite(x) else str(x)


def open_writer(cfg: RunConfig, run: str, env: CheatEnv,
                extra: Mapping[str, Any] | None = None) -> MetricsWriter:
    """A ``MetricsWriter`` under ``$VERIF_OUTPUT_DIR/<run>/``, with ``run_verified``'s
    environment fields."""
    c = env.c
    inst = {"n_s": c.n_s, "eta": c.eta, "k": env.k, "n_records": len(env.D), "M": c.M,
            "n_leaves": c.n_leaves, "poison_step": env.poison_step,
            "poisoning_rate": env.poisoning_rate, **(extra or {})}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics")}, "llama": inst}
    return MetricsWriter(cfg.output_dir / run, run, device=cfg.device, model=cfg.model,
                         corpus=cfg.dataset, seed=cfg.seed, config=config,
                         band_file_hash=env.bands.source, h_D=env.h_D,
                         extra={"band_source": env.bands.source, **inst})
