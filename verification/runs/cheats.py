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
(B6, EQ13, EVALUATION_SPEC §4.1–§4.2), then one ``cheat_oracle`` record: the declared and the
actual rejection point, the outcome, the detection latency (EQ15: rejection step minus cheat
step) and the detection scores of the rejecting step (EQ4, :func:`scores`). Each cheat is a cell
of its own, with its own run directory, :func:`cell_name`
(``<scenario>-<model>-<attack>-bpr<n>-s<seed>``).
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from setup import backdoorllm
from setup.config import RunConfig
from setup.data import load_dataset_records, schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.parameters import ProtocolConfig
from verification.runs.llama_step import load_committed_dataset
from verification.runs.metrics import MetricsWriter, nan_as_inf, rho_arrays, run_name
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
from verification.verifier.calibration import (
    assert_same_band_source,
    band_context,
    load_bands,
)
from verification.verifier.context import StepStats

__all__ = ["ORACLE_RECORD", "CheatEnv", "poisoned_batch", "check_meta", "load_env", "load_level",
           "run_cheat", "scores", "oracle_record", "band_sources", "report_oracle", "cell_name",
           "open_writer"]

ORACLE_RECORD = "cheat_oracle"


@dataclass
class CheatEnv:
    """What every cheat run shares: the honest run's inputs and one judging-verifier factory.

    ``b_tilde`` is ``D̃``'s batch at ``π(poison_step)``, the poisoned batch ``b̃``.
    ``bpr_records`` is the number of ``b̃``'s records that differ from ``π``'s batch, and
    ``bpr_actual`` their share of the batch (EVALUATION_SPEC §2's actual BPR).
    ``seed`` fixes ``π`` (F8c) for the prover, the verifier and ``b̃`` alike."""

    c: DeclaredComputation
    D: list[Any]
    h_D: bytes
    seed: int
    b_tilde: list[Any]
    poison_step: int
    bpr_records: int
    bpr_actual: float
    w0: dict[str, torch.Tensor]
    bands: Bands
    k: int
    models: BuildModel
    verifiers: JudgingVerifiers
    _finals: dict[int, dict[str, torch.Tensor]] = dataclasses.field(default_factory=dict)

    @classmethod
    def build(cls, c: DeclaredComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
              bands: Bands, *, k: int, h_D: bytes, seed: int, b_tilde: Sequence[Any],
              poison_step: int, build_model: BuildModel | None = None) -> CheatEnv:
        b = [D[i] for i in schedule(poison_step, c.n_s, len(D), seed)]
        changed = sum(c.encode_record(x) != c.encode_record(y) for x, y in zip(b, b_tilde))
        if len(b_tilde) != c.n_s or changed == 0:
            raise ValueError(f"b̃ must be {c.n_s} records, at least one unlike π's batch")
        return cls(c, list(D), h_D, seed, list(b_tilde), poison_step, int(changed),
                   changed / c.n_s, dict(w0), bands, k, build_model or c.build_model,
                   JudgingVerifiers(c, len(D), w0, bands, k=k, h_D=h_D, seed=seed))

    def final(self, T: int) -> dict[str, torch.Tensor]:
        """Check 8's reference for a run of ``T`` steps: ``T`` honest uncaptured steps."""
        if T not in self._finals:
            self._finals[T] = honest_final(self.c, self.D, self.w0, T, seed=self.seed,
                                           build_model=self.models)
        return self._finals[T]


def poisoned_batch(c: DeclaredComputation, D_tilde: Sequence[Any], step: int,
                   n_records: int, seed: int) -> list[Any]:
    """``b̃``: ``D̃``'s records at ``π(step)``, the batch the poisoned step trains on."""
    return [D_tilde[i] for i in schedule(step, c.n_s, n_records, seed)]


def check_meta(meta: Mapping[str, Any], cfg: RunConfig) -> None:
    """Raise ``ValueError`` unless ``meta.json`` describes the run's ``D̃``: drawn at the run's
    seed, since ``D̃`` follows that seed's ``π`` (F8g), for the run's task (``VERIF_ATTACK``)
    and at its poisoning level (``VERIF_POISON_COUNT``, EQ5), a count out of a batch of the
    run's size (``VERIF_BATCH``). A ``meta.json`` that lacks a field doesn't match."""
    poisoning = meta.get("poisoning") or {}
    for what, published, run in (("seed", meta.get("seed"), cfg.seed),
                                 ("task", meta.get("attack"), cfg.attack),
                                 ("poisoning level", poisoning.get("count"), cfg.poison_count),
                                 ("batch size", poisoning.get("batch"), cfg.batch)):
        if published != run:
            raise ValueError(f"meta.json's data has {what} {published!r}, the run's is "
                             f"{run!r}: rerun verification.runs.materialize_data")


def load_env(cfg: RunConfig, pc: ProtocolConfig) -> CheatEnv:
    """The SmolLM2 cheat environment, as ``run_verified.main`` sets up the honest run.

    ``D̃`` comes from ``D_tilde.bin``, checked against the ``h_D_tilde`` that ``meta.json``
    publishes; the poisoned step is ``meta.json``'s ``poisoning.step``. ``meta.json``'s seed,
    task and poisoning level must be the run's (:func:`check_meta`)."""
    c = DecoderComputation.from_config(cfg)
    data = cfg.data_dir
    if not (data / "D.bin").exists() or not (data / "D_tilde.bin").exists():
        raise FileNotFoundError(f"no D.bin or D_tilde.bin under {data}: run "
                                "verification.runs.materialize_data or set VERIF_OUTPUT_DIR")
    D, h_D = load_committed_dataset(c, data)
    # read-only; never refitted here (P10a)
    bands = load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D, k=pc.k))
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())  # as B7 and A12 take it
    return load_level(cfg, data, c=c, D=D, h_D=h_D, w0=w0, bands=bands, k=pc.k, models=models)


def load_level(cfg: RunConfig, data_dir: Path, *, c: DecoderComputation, D: Sequence[Any],
               h_D: bytes, w0: Mapping[str, torch.Tensor], bands: Bands, k: int,
               models: BuildModel) -> CheatEnv:
    """The cheat environment of one seed and level, over a model, ``D``, ``W_0`` and band file
    already loaded: ``D̃`` from ``data_dir/D_tilde.bin``, checked against ``meta.json``'s
    ``h_D_tilde``, and the poisoned step from its ``poisoning.step``. ``meta.json`` must
    describe the run (:func:`check_meta`, with ``cfg``'s seed and ``VERIF_POISON_COUNT``) and
    publish the same ``h_D``. :func:`load_env` calls it once; the full-scale driver
    (``runs/full_scale.py``) once per level, so one model load serves a seed's levels."""
    meta = json.loads((data_dir / "meta.json").read_text())
    check_meta(meta, cfg)
    if meta.get("h_D") != h_D.hex():
        raise ValueError(f"{data_dir / 'meta.json'} publishes h_D {meta.get('h_D')}, the run's "
                         f"D has {h_D.hex()}")
    D_tilde = load_dataset_records(data_dir / "D_tilde.bin", c.n)
    if dataset_tree(c, D_tilde).root.hex() != meta["h_D_tilde"]:
        raise ValueError("D_tilde.bin does not hash to meta.json's h_D_tilde")
    step = int(meta["poisoning"]["step"])
    return CheatEnv.build(c, D, w0, bands, k=k, h_D=h_D, seed=cfg.seed,
                          b_tilde=poisoned_batch(c, D_tilde, step, len(D), cfg.seed),
                          poison_step=step, build_model=models)


# ---- one cheat run ------------------------------------------------------------------------


def run_cheat(env: CheatEnv, scenario: Scenario, *, T: int,
              writer: MetricsWriter | None = None, bpr_records: int | None = None,
              bpr_actual: float | None = None, cheat_step: int | None = None,
              out: Callable[[str], None] = print) -> ScenarioResult:
    """One cheat through the S3 loop, judged live by the band file, cut after step ``T``.

    With ``writer``, the timed run writes its records, then the ``cheat_oracle`` record.
    ``bpr_records``, ``bpr_actual`` and ``cheat_step`` go into both (EVALUATION_SPEC §4.1):
    ``env.bpr_records`` and ``env.bpr_actual`` for a cheat on ``b̃``, ``0`` and ``0.0`` for an
    honest scenario, ``None`` for a protocol fault that trains on no poisoned batch."""
    rec = (None if writer is None else
           writer.recorder(scenario.name, bpr_records=bpr_records, bpr_actual=bpr_actual,
                           cheat_step=cheat_step))
    r = run_scenario(env.c, env.D, env.w0, scenario, T=T, k=env.k, seed=env.seed,
                     final=env.final(T), recorder=rec, h_D=env.h_D, build_model=env.models,
                     verifier=env.verifiers.for_steps(T))
    report(r, out)
    if writer is not None:
        writer.write([oracle_record(writer.run, r, env.bands, bpr_records=bpr_records,
                                    bpr_actual=bpr_actual, cheat_step=cheat_step)])
    return r


def scores(stats: StepStats, bands: Bands) -> dict[str, Any]:
    """EQ4's detection scores over one step's numbers (EVALUATION_SPEC §2.2), each the largest
    component score with where it was reached: ``ρ_5`` (a check-5 normalized residual over
    ``τ``), ``ρ_κ`` (a product's ``ν`` over ``κ_max·‖|P|·1‖``) and ``ρ_6`` (a check-6 ``ρ``
    over its tensor's ``τ_W``), from ``runs.metrics.rho_arrays``. They cover what the verifier
    measured at the step: since EV2 a rejecting check measures all its tests for the record, so
    a step rejected at check 5 has every product and one rejected at check 6 every tensor of
    check 6. ``None`` where the step measured nothing. A NaN score counts as ``+inf``
    (``runs.metrics.nan_as_inf``), as in the step record's maximum: a NaN test fails."""
    out: dict[str, Any] = {"rho_5": None, "rho_5_at": None, "rho_kappa": None,
                           "rho_kappa_at": None, "rho_6": None, "rho_6_at": None}
    rho = {fam: nan_as_inf(x) for fam, x in rho_arrays(stats, bands, k=None).items()}
    if stats.products:
        i = int(np.argmax(rho["rho_5"].max(axis=1)))
        p = stats.products[i]
        out["rho_5"], out["rho_5_at"] = float(rho["rho_5"][i].max()), f"P_{p.m} ({p.name})"
        i = int(np.argmax(rho["rho_kappa"]))
        p = stats.products[i]
        out["rho_kappa"] = float(rho["rho_kappa"][i])
        out["rho_kappa_at"] = f"P_{p.m} ({p.name})"
    if stats.tensors:
        i = int(np.argmax(rho["rho_6"]))
        s = stats.tensors[i]
        out["rho_6"], out["rho_6_at"] = float(rho["rho_6"][i]), f"{s.check_id} {s.weight}"
    return out


def _point(e: Any) -> dict[str, Any]:
    return {"step": e.step, "check": e.check_id, "kind": e.kind, "product": e.product}


def oracle_record(run: str, r: ScenarioResult, bands: Bands, *,
                  bpr_records: int | None = None, bpr_actual: float | None = None,
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
            "cheat_step": cheat_step, "bpr_records": bpr_records, "bpr_actual": bpr_actual,
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
    out(f"{'cheat':<10} {'declared':<34} {'actual':<34} {'outcome':<13} {'ρ_5':>9} {'ρ_κ':>9} "
        f"{'ρ_6':>11}")
    for r in results:
        rej = r.loop.rejection
        st = r.verifier.stats.get(rej.step, StepStats()) if rej is not None else StepStats()
        sc = scores(st, bands)
        actual = r.actual if rej is None else dataclasses.replace(
            r.actual, product=rejected_product(rej))
        out(f"{r.scenario.name:<10} {str(r.scenario.expected):<34} {str(actual):<34} "
            f"{r.outcome:<13} {_num(sc['rho_5']):>9} {_num(sc['rho_kappa']):>9} "
            f"{_num(sc['rho_6']):>11}")
    passed = sum(r.passed for r in results)
    out(f"{passed}/{len(results)} cheats rejected exactly at their declared point")


def _num(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3g}" if math.isfinite(x) else str(x)


def cell_name(cfg: RunConfig, scenario: str, env: CheatEnv) -> str:
    """The run name of ``scenario``'s cell (``runs.metrics.run_name``): model, task, the poisoned
    records in ``b̃`` and the seed."""
    return run_name(scenario, cfg.model, cfg.attack, env.bpr_records, cfg.seed)


def open_writer(cfg: RunConfig, run: str, env: CheatEnv,
                extra: Mapping[str, Any] | None = None,
                subdir: str | None = None) -> MetricsWriter:
    """A ``MetricsWriter`` under ``$VERIF_OUTPUT_DIR/<run>/`` (or ``<run>/<subdir>/``, so a
    partial rerun keeps the earlier records), with ``run_verified``'s environment fields."""
    c = env.c
    inst = {"n_s": c.n_s, "eta": c.eta, "k": env.k, "n_records": len(env.D), "M": c.M,
            "n_leaves": c.n_leaves, "poison_step": env.poison_step,
            "bpr_records": env.bpr_records, "bpr_actual": env.bpr_actual, **(extra or {})}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics", "release_cache")}, "llama": inst}
    out_dir = cfg.output_dir / run if subdir is None else cfg.output_dir / run / subdir
    return MetricsWriter(out_dir, run, device=cfg.device, model=cfg.model,
                         corpus=backdoorllm.SOURCE, attack=cfg.attack, seed=cfg.seed,
                         config=config, band_file_hash=env.bands.source, h_D=env.h_D,
                         extra={"band_source": env.bands.source, **inst})
