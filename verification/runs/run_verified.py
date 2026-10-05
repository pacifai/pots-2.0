"""A12: the honest SmolLM2 run, judged by the frozen band file.

    .venv/bin/python -m verification.runs.run_verified [--steps T] [--pass-steps S]
                                                       [--metrics | --no-metrics]

The honest run of ``T`` steps (``--steps``, else ``VERIF_STEPS``, default 10), set up as in
``runs/llama_step.py``: ``W_0`` from the unmodified ``from_pretrained`` model, ``π``'s batches of
the committed ``D`` and the published ``h_D``. The verifier loads the band file read-only with
``calibration.load_bands`` (P10a) and judges every step live. The steps C1 fitted the bands on
(the band file's ``calibration_steps``, 1–3) are **in-sample**; every later step is **judged**
out of sample. Check 8 compares the last ``W_{t+1}`` with ``T`` uncaptured steps from ``W_0``.

**Per step** the run logs, as each step closes:

- every check-5 normalized residual and every check-6 ``ρ_max``, as B6's residual arrays
  (``residuals/honest/step_<t>.npz``), and the per-component cost rows (``runs/metrics.py``);
- a ``residual_summary`` record: per matmul class the largest normalized residual and the
  largest ``κ`` over the class's frozen ``κ_max``, every residual of the watched class (``Λ``,
  the class that set ``s_h``), and check 6's largest ``ρ``. It judges nothing.

**Checks after the run.**

- P10a: one band-file hash across the band file, every verifier of this command (the timed
  run and both passes) and, when its records are there, the calibration run that wrote the file
  (``calibrate/k<k>/records.jsonl``). A mismatch raises ``AssertionError``.
- B7: the run's final weights go to ``final_weights.json`` (``plain_baseline``'s format), and
  are compared bit for bit with the plain baseline's (``plain_baseline/final_weights.json``),
  provenance first. A mismatch exits 1; a missing baseline file is reported and skipped.
- The concentration-guard line ``τ/2``: a judged step whose class maximum passes it is named.
  The run's verdict doesn't change; the guard is C1's test on the calibration steps.

With metrics on (default ``VERIF_METRICS``) the records go to ``$VERIF_OUTPUT_DIR/run_verified/``,
and a memory pass and a counting pass of the first ``S`` steps (``--pass-steps``, default
``PASS_STEPS``) follow. Exits 1 unless every step is accepted.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from verification.computation.instances.llama import LlamaComputation
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.parameters import load_protocol_config
from verification.runs.calibrate import judging_verifier
from verification.runs.llama_step import load_committed_dataset, report_costs
from verification.runs.loop import StepRecord
from verification.runs.metrics import (
    PASS_STEPS,
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    read_records,
)
from verification.runs.plain_baseline import (
    WEIGHTS_FILE,
    assert_same_final_weights,
    run_provenance,
    write_final_weights,
)
from verification.runs.scenarios import (
    HONEST,
    ReusedModel,
    ScenarioResult,
    count_run,
    honest_final,
    memory_run,
    report,
    run_scenario,
)
from verification.verifier.bands import Bands
from verification.verifier.calibration import (
    GUARD_FRACTION,
    assert_same_band_source,
    load_bands,
)
from verification.verifier.context import Section, StepStats
from verification.verifier.driver import Verifier
from verification.verifier.residuals import format_tensor_table, tensor_summary

__all__ = ["RUN_NAME", "WATCH_CLASS", "StepSummary", "VerifiedRun", "step_summary",
           "summary_record", "run_verified", "calibration_band_hash", "report_summaries",
           "main"]

RUN_NAME = "run_verified"  # the metrics directory under VERIF_OUTPUT_DIR
SUMMARY_RECORD = "residual_summary"
# The class whose RMS set s_h in C1 (Λ, the output-layer product). Its largest residual grew
# 5.4 → 10.1 → 14.5 over steps 1–3, so the run logs all its residuals at every step.
WATCH_CLASS = "Lambda"


# ---- per-step summary -------------------------------------------------------------------


@dataclass(frozen=True)
class StepSummary:
    """One step's check-5 and check-6 numbers, summarized against the frozen bands.

    ``class_max`` is each class's largest normalized residual, ``kappa_ratio`` its largest κ
    divided by the class's ``κ_max``, both in the order the products run. ``watch`` holds every
    normalized residual of each product of the watched class, by product name."""

    t: int
    in_sample: bool
    rejected: bool
    class_max: dict[str, float]
    kappa_ratio: dict[str, float]
    watch: dict[str, tuple[float, ...]]
    rho_max: dict[str, float]  # check id ("6a", "6b") → largest ρ

    @property
    def global_max(self) -> tuple[str, float]:
        """The class with the largest normalized residual, and that residual."""
        return max(self.class_max.items(), key=lambda kv: kv[1], default=("", math.nan))

    @property
    def kappa_ratio_max(self) -> tuple[str, float]:
        return max(self.kappa_ratio.items(), key=lambda kv: kv[1], default=("", math.nan))


def step_summary(t: int, stats: StepStats, bands: Bands, *, in_sample: bool,
                 rejected: bool = False, watch: str = WATCH_CLASS) -> StepSummary:
    """Summarize one step's ``StepStats`` against ``bands``. It judges nothing."""
    cmax: dict[str, float] = {}
    kmax: dict[str, float] = {}
    for p in stats.products:
        cmax[p.cls] = max(cmax.get(p.cls, -math.inf), *p.normalized)
        kmax[p.cls] = max(kmax.get(p.cls, -math.inf), p.kappa)
    ratio = {cls: x / bands.kappa_for(cls) for cls, x in kmax.items()}
    seen = {p.name: tuple(p.normalized) for p in stats.products if p.cls == watch}
    rho: dict[str, float] = {}
    for s in stats.tensors:
        rho[s.check_id] = max(rho.get(s.check_id, -math.inf), s.rho_max)
    return StepSummary(t, in_sample, rejected, cmax, ratio, seen, rho)


def summary_record(run: str, scenario: str, s: StepSummary, bands: Bands, *,
                   watch: str = WATCH_CLASS) -> dict[str, Any]:
    """The step's ``residual_summary`` record for ``records.jsonl``."""
    cls, mx = s.global_max
    kcls, kr = s.kappa_ratio_max
    return {"record": SUMMARY_RECORD, "run": run, "scenario": scenario, "step": s.t,
            "sample": "in" if s.in_sample else "judged", "rejected": s.rejected,
            "tau": bands.tau, "guard_line": GUARD_FRACTION * bands.tau,
            "band_file_hash": bands.source, "global_max": mx, "global_max_class": cls,
            "class_max": s.class_max, "kappa_ratio": s.kappa_ratio,
            "kappa_ratio_max": kr, "kappa_ratio_max_class": kcls,
            "watch_class": watch, "watch": {n: list(v) for n, v in s.watch.items()},
            "rho_max": s.rho_max}


def _step_line(s: StepSummary, tau: float) -> str:
    cls, mx = s.global_max
    kcls, kr = s.kappa_ratio_max
    watch = "; ".join(f"{n} max {max(v):.2f} [" + " ".join(f"{x:.2f}" for x in v) + "]"
                      for n, v in s.watch.items())
    return (f"step {s.t:>2} ({'in-sample' if s.in_sample else 'judged'}): "
            f"{'REJECTED' if s.rejected else 'accepted'}; max residual {mx:.2f} ({cls}), "
            f"τ = {tau:.1f}; max κ/κ_max {kr:.3f} ({kcls}); "
            + ", ".join(f"{c} ρ_max {x:.2f}" for c, x in s.rho_max.items())
            + (f"; {watch}" if watch else ""))


# ---- the run ----------------------------------------------------------------------------


class _Verifiers:
    """A ``run_scenario`` verifier factory that judges with the band file's bands and keeps
    every verifier it builds, so each one's band source can be compared afterwards."""

    def __init__(self, c: DeclaredComputation, n_records: int,
                 w0: Mapping[str, torch.Tensor], bands: Bands, *, k: int, h_D: bytes) -> None:
        self._args = (c, n_records, w0, bands)
        self._kw = {"k": k, "h_D": h_D}
        self.made: list[Verifier] = []

    def for_steps(self, T: int) -> Callable[[Section | None], Verifier]:
        make = judging_verifier(*self._args, T=T, **self._kw)  # type: ignore[arg-type]

        def build(section: Section | None) -> Verifier:
            v = make(section)
            self.made.append(v)
            return v
        return build


@dataclass
class VerifiedRun:
    result: ScenarioResult
    bands: Bands
    summaries: list[StepSummary]
    factory: _Verifiers  # builds this run's verifiers, and later the passes'
    watch: str = WATCH_CLASS

    @property
    def verifiers(self) -> list[Verifier]:
        """Every verifier the factory built: the run's, then each pass's."""
        return self.factory.made

    @property
    def calibration_steps(self) -> tuple[int, ...]:
        return tuple(self.bands.stats.get("calibration_steps", ()))

    def over_guard(self) -> list[tuple[int, str, float]]:
        """``(step, class, residual)`` for every class maximum above ``τ/2`` on a judged
        step."""
        line = GUARD_FRACTION * self.bands.tau
        return [(s.t, cls, x) for s in self.summaries if not s.in_sample
                for cls, x in s.class_max.items() if x > line]


def run_verified(c: DeclaredComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
                 bands: Bands, *, T: int, k: int, h_D: bytes, final: Mapping[str, torch.Tensor],
                 recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None,
                 writer: MetricsWriter | None = None,
                 build_model: Callable[[], torch.nn.Module] | None = None,
                 watch: str = WATCH_CLASS, out: Callable[[str], None] = print) -> VerifiedRun:
    """``T`` honest steps judged live by ``bands`` (loaded from the band file).

    Each closed step is summarized (:func:`step_summary`), printed, and with ``writer``
    written as a ``residual_summary`` record. Steps in the band file's ``calibration_steps``
    are marked in-sample. ``watch`` names the class whose residuals are all logged."""
    factory = _Verifiers(c, len(D), w0, bands, k=k, h_D=h_D)
    cal_steps = set(bands.stats.get("calibration_steps", ()))
    summaries: list[StepSummary] = []

    def on_step(rec: StepRecord) -> None:
        v = factory.made[-1]
        st = v.stats.get(rec.t, StepStats())
        s = step_summary(rec.t, st, bands, in_sample=rec.t in cal_steps,
                         rejected=rec.rejection is not None, watch=watch)
        summaries.append(s)
        out(_step_line(s, bands.tau))
        if writer is not None:
            writer.write([summary_record(writer.run, HONEST.name, s, bands, watch=watch)])

    r = run_scenario(c, D, w0, HONEST, T=T, k=k, final=final, on_step=on_step,
                     recorder=recorder, h_D=h_D, build_model=build_model,
                     verifier=factory.for_steps(T))
    return VerifiedRun(r, bands, summaries, factory, watch)


def calibration_band_hash(output_dir: Path, k: int) -> str | None:
    """The band-file hash the calibration run recorded (``calibrate/k<k>/records.jsonl``), or
    ``None`` when those records aren't there."""
    path = output_dir / "calibrate" / f"k{k}" / "records.jsonl"
    if not path.exists():
        return None
    rows = read_records(path, "calibration")
    return rows[-1].get("band_file_hash") if rows else None


# ---- reports ----------------------------------------------------------------------------


def report_summaries(run: VerifiedRun, out: Callable[[str], None] = print) -> None:
    """The per-step tables: each class's largest residual, then its largest κ over κ_max,
    one column per step; the watched class's residuals; check 6's ρ."""
    ss = run.summaries
    if not ss:
        return
    tau = run.bands.tau
    line = GUARD_FRACTION * tau
    head = "".join(f"{s.t:>7}{'*' if s.in_sample else ' '}" for s in ss)
    classes = list(dict.fromkeys(c for s in ss for c in s.class_max))
    out(f"check 5: largest normalized residual per class and step (τ = {tau:.2f}, guard line "
        f"τ/2 = {line:.2f}; * = in-sample)")
    out(f"{'class':<10}{head}")
    for cls in classes:
        out(f"{cls:<10}" + "".join(f"{s.class_max.get(cls, math.nan):>7.2f} " for s in ss))
    out(f"{'max':<10}" + "".join(f"{s.global_max[1]:>7.2f} " for s in ss))
    out("check 5: largest κ / the class's frozen κ_max, per class and step (1 = the ceiling)")
    out(f"{'class':<10}{head}")
    for cls in classes:
        out(f"{cls:<10}" + "".join(f"{s.kappa_ratio.get(cls, math.nan):>7.3f} " for s in ss))
    out(f"{'max':<10}" + "".join(f"{s.kappa_ratio_max[1]:>7.3f} " for s in ss))
    out(f"{run.watch}: every normalized residual per step (k = {run.result.verifier.k})")
    for s in ss:
        for name, v in s.watch.items():
            out(f"  step {s.t:>2}{'*' if s.in_sample else ' '} {name}: max {max(v):6.2f}  "
                + " ".join(f"{x:6.2f}" for x in v))
    out("check 6: ρ_max by weight role over the run")
    format_tensor_table(tensor_summary(run.result.verifier.stats), out)
    over = run.over_guard()
    out("concentration-guard line on judged steps: "
        + ("no class maximum above τ/2" if not over else
           "ABOVE τ/2: " + ", ".join(f"step {t} {c} {x:.2f}" for t, c, x in over)))


# ---- main -------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=None, help="T (default: VERIF_STEPS)")
    p.add_argument("--pass-steps", type=int, default=PASS_STEPS,
                   help=f"steps in the memory and counting passes (default {PASS_STEPS})")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    T, S, k = (args.steps if args.steps is not None else cfg.steps), args.pass_steps, pc.k
    if T < 1 or not 1 <= S <= T:
        p.error(f"need T ≥ 1 and 1 ≤ --pass-steps ≤ T, got T = {T}, {S}")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    bands = load_bands(pc.band_file, k=k)  # read-only; never refitted here (P10a)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())  # as B7 takes it
    final = honest_final(c, D, w0, T, build_model=models)
    print(f"A12 verified run: {cfg.model}@{cfg.model_revision[:12]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, k {k}, T {T}, |D| {len(D)}, h_D {h_D.hex()[:16]}…, M {c.M}; "
          f"band file {pc.band_file} ({bands.source[:16]}…), τ {bands.tau:.2f}, in-sample "
          f"steps {list(bands.stats.get('calibration_steps', []))}")
    run_dir = cfg.output_dir / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    llama = {"n_s": c.n_s, "n": c.n, "eta": c.eta, "k": k, "T": T, "n_records": len(D),
             "M": c.M, "n_leaves": c.n_leaves, "pass_steps": S}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics")}, "llama": llama}
    base = dict(k=k, h_D=h_D, build_model=models)
    mem_rows = None
    mw = (MetricsWriter(run_dir, RUN_NAME, device=cfg.device, model=cfg.model,
                        corpus=cfg.dataset, seed=cfg.seed, config=config,
                        band_file_hash=bands.source, h_D=h_D,
                        extra={"band_source": str(pc.band_file), **llama})
          if use_metrics else None)
    try:
        run = run_verified(c, D, w0, bands, T=T, final=final, writer=mw,
                                    recorder=None if mw is None else mw.recorder(HONEST.name),
                                    **base)
        r = run.result
        weights_doc = write_final_weights(run_dir / WEIGHTS_FILE, c, r.loop.w_final or {},
                                          w0=w0, losses=[s.loss for s in r.loop.steps],
                                          provenance=run_provenance(cfg, h_D), run=RUN_NAME)
        if mw is not None:
            final_S = final if S == T else honest_final(c, D, w0, S, build_model=models)
            pass_args = dict(T=S, final=final_S, **base)
            mem_rows = memory_run(c, D, w0, run=RUN_NAME, device=mw.device,
                                  verifier=run.factory.for_steps(S), **pass_args)
            mw.write(mem_rows)
            mw.write(count_run(c, D, w0, run=RUN_NAME, verifier=run.factory.for_steps(S),
                               **pass_args))
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")

    report(r)
    report_summaries(run)
    report_costs(r, mem_rows)
    ok = r.passed

    sources = [bands.source, r.loop.verdict.band_source] + [v.band_source for v in run.verifiers]
    cal_hash = calibration_band_hash(cfg.output_dir, k)
    if cal_hash is not None:
        sources.append(cal_hash)
    shared = assert_same_band_source(sources)
    print(f"band-file hash equal across {len(sources)} sources (the file, {len(run.verifiers)} "
          f"verifiers, the verdict" + (", the calibration run" if cal_hash else "")
          + f"): {shared}")

    print(f"final weights: root {weights_doc['root'][:16]}…  ({run_dir / WEIGHTS_FILE})")
    plain = cfg.output_dir / "plain_baseline" / WEIGHTS_FILE
    if not plain.exists():
        print(f"B7: no plain baseline at {plain}; run verification.runs.plain_baseline to compare")
    else:
        try:
            assert_same_final_weights(c, plain, run_dir / WEIGHTS_FILE)
            print(f"B7: final weights bit-identical to the plain baseline's ({plain})")
        except AssertionError as e:
            print(f"B7: FAILED: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
