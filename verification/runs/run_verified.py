"""A12: the honest SmolLM2 run, judged by the frozen band file.

    .venv/bin/python -m verification.runs.run_verified [--steps T] [--pass-steps S]
                                                       [--keep-step N]
                                                       [--metrics | --no-metrics]

The honest run of ``T`` steps (``--steps``, else ``VERIF_STEPS``, default 10), set up as in
``runs/llama_step.py``: ``W_0`` from the unmodified ``from_pretrained`` model, ``π``'s batches of
the committed ``D`` and the published ``h_D``. The verifier loads the band file read-only with
``calibration.load_bands`` (P10a) and judges every step live. At the band file's calibration
seed (H1), the steps C1 fitted the bands on (the file's ``calibration_steps``, 1–3) are
**in-sample** and every later step is **judged** out of sample. At any other seed (H2) every
step but step 1 is judged (F8a; ``metrics.judged_step``). Check 8 compares the last
``W_{t+1}`` with ``T`` uncaptured steps from ``W_0``.

**Per step** the run logs, as each step closes:

- the step record and the residual arrays with their scores (``runs/metrics.py``,
  EVALUATION_SPEC §4.1–§4.2; ``residuals/honest/step_<t>.npz``), and the per-component cost rows;
- a ``residual_summary`` record: per matmul class the largest normalized residual and the
  largest ``κ`` over the class's frozen ``κ_max``, every residual of the watched class (``Λ``,
  the class that set ``s_h``), and check 6's largest ``ρ``. It judges nothing.

**The stored step** (``--keep-step N``, ``0`` for none). By default (:func:`default_keep_step`)
a run at the band file's calibration seed (H1) stores step :data:`KEEP_STEP` = 4, its first
judged step, and a run at any other seed stores none: the planted-error sweep and the
``k``-tunability run (EVALUATION_SPEC §5.7–§5.8) take H1's step. The transcript is written to
``transcripts/step_<N>/`` in the run directory with ``DiskHandoff(keep=True)`` and kept. The
verifier reads every step from memory, as every run does, and the step is written after its
verdict, outside every timed section (:class:`KeptStepHandoff`), so its cost rows are an
in-memory step's. The root recomputed from the written files must equal the step's ``h``; a
``kept_transcript`` record holds the path, the size and the root, and is written only when a
step was stored.

**Checks after the run.**

- P10a: one band-file hash across the band file, every verifier of this command (the timed
  run and both passes) and, when its records are there, the calibration run that wrote the file
  (``calibrate/k<k>/records.jsonl``). A mismatch raises ``AssertionError``.
- B7: the run's final weights go to ``final_weights.json`` (``plain_baseline``'s format), and
  are compared bit for bit with the plain baseline's (``plain_baseline/final_weights.json``),
  provenance first (:func:`compare_with_baseline`). A mismatch exits 1. A missing baseline
  file, or a baseline at another seed than the run's, is reported as skipped and doesn't
  change the exit code.
- The concentration-guard line ``τ/2``: a judged step whose class maximum passes it is named.
  The run's verdict doesn't change; the guard is C1's test on the calibration steps.

The run directory is the cell's, ``$VERIF_OUTPUT_DIR/honest-<model>-<attack>-bpr0-s<seed>/``
(``metrics.run_name``), for example ``honest-SmolLM2-135M-Instruct-refusal-bpr0-s1/``. It holds
``final_weights.json`` and the stored step always, and with metrics on (default
``VERIF_METRICS``) the records, then a memory pass of the first ``S`` steps (``--pass-steps``,
default ``PASS_STEPS``) and a counting pass of ``COUNT_STEPS`` (one step, EVALUATION_SPEC §3.4).
Exits 1 unless every step is accepted.
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

from setup import backdoorllm
from setup.config import load_config, setup_determinism
from verification.commitment.leaves import transcript_root
from verification.commitment.merkle import MerkleTree
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.parameters import load_protocol_config
from verification.runs.calibrate import judging_verifier
from verification.runs.llama_step import load_committed_dataset, report_costs
from verification.runs.loop import StepRecord
from verification.runs.metrics import (
    COUNT_STEPS,
    PASS_STEPS,
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    calibration_seed,
    in_sample_step,
    judged_step,
    read_records,
    run_name,
)
from verification.runs.plain_baseline import (
    RUN_NAME as PLAIN_RUN,
    WEIGHTS_FILE,
    assert_same_final_weights,
    read_final_weights,
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
from verification.transcript.store import (
    IN_MEMORY,
    DiskHandoff,
    DiskStore,
    StoreHandoff,
    TranscriptStore,
)
from verification.verifier.bands import Bands
from verification.verifier.calibration import (
    GUARD_FRACTION,
    assert_same_band_source,
    band_context,
    load_bands,
)
from verification.verifier.context import Section, StepStats
from verification.verifier.driver import Verifier
from verification.verifier.residuals import format_tensor_table, tensor_summary

__all__ = ["WATCH_CLASS", "KEEP_STEP", "TRANSCRIPTS_DIR", "KEPT_RECORD", "StepSummary",
           "VerifiedRun", "JudgingVerifiers", "KeptStepHandoff", "honest_run_name",
           "step_summary", "summary_record", "kept_record", "default_keep_step",
           "run_verified", "calibration_band_hash", "baseline_seed", "compare_with_baseline",
           "report_summaries", "main"]

SUMMARY_RECORD = "residual_summary"
# The honest step whose transcript is stored for the planted-error sweep and the k-tunability
# run (EVALUATION_SPEC §5.7–§5.8): step 4, H1's first judged step. Stored by default only in a
# run at the band file's calibration seed (default_keep_step).
KEEP_STEP = 4
TRANSCRIPTS_DIR = "transcripts"  # under the run directory; the step is in step_<t>/
KEPT_RECORD = "kept_transcript"
# The class whose RMS set s_h in C1 (Λ, the output-layer product). Its largest residual grew
# 5.4 → 10.1 → 14.5 over steps 1–3, so the run logs all its residuals at every step.
WATCH_CLASS = "Lambda"


# ---- per-step summary -------------------------------------------------------------------


@dataclass(frozen=True)
class StepSummary:
    """One step's check-5 and check-6 numbers, summarized against the frozen bands.

    ``class_max`` is each class's largest normalized residual, ``kappa_ratio`` its largest κ
    divided by the class's ``κ_max``, both in the order the products run. ``watch`` holds every
    normalized residual of each product of the watched class, by product name. ``in_sample``:
    the bands were fitted on this step; ``judged``: the step counts as a result
    (``metrics.judged_step``). A step can be neither (H2's step 1)."""

    t: int
    in_sample: bool
    rejected: bool
    class_max: dict[str, float]
    kappa_ratio: dict[str, float]
    watch: dict[str, tuple[float, ...]]
    rho_max: dict[str, float]  # check id ("6a", "6b") → largest ρ
    judged: bool = False

    @property
    def sample(self) -> str:
        """``"in"`` (in-sample), ``"judged"`` or ``"not judged"``."""
        return "in" if self.in_sample else "judged" if self.judged else "not judged"

    @property
    def mark(self) -> str:
        """The tables' column mark: ``*`` in-sample, ``-`` not judged otherwise."""
        return "*" if self.in_sample else " " if self.judged else "-"

    @property
    def global_max(self) -> tuple[str, float]:
        """The class with the largest normalized residual, and that residual."""
        return max(self.class_max.items(), key=lambda kv: kv[1], default=("", math.nan))

    @property
    def kappa_ratio_max(self) -> tuple[str, float]:
        return max(self.kappa_ratio.items(), key=lambda kv: kv[1], default=("", math.nan))


def step_summary(t: int, stats: StepStats, bands: Bands, *, in_sample: bool,
                 judged: bool | None = None, rejected: bool = False,
                 watch: str = WATCH_CLASS) -> StepSummary:
    """Summarize one step's ``StepStats`` against ``bands``. It judges nothing. ``judged``
    defaults to ``not in_sample``."""
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
    return StepSummary(t, in_sample, rejected, cmax, ratio, seen, rho,
                       not in_sample if judged is None else judged)


def summary_record(run: str, scenario: str, s: StepSummary, bands: Bands, *,
                   watch: str = WATCH_CLASS) -> dict[str, Any]:
    """The step's ``residual_summary`` record for ``records.jsonl``."""
    cls, mx = s.global_max
    kcls, kr = s.kappa_ratio_max
    return {"record": SUMMARY_RECORD, "run": run, "scenario": scenario, "step": s.t,
            "sample": s.sample, "judged": s.judged, "rejected": s.rejected,
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
    return (f"step {s.t:>2} ({'in-sample' if s.in_sample else s.sample}): "
            f"{'REJECTED' if s.rejected else 'accepted'}; max residual {mx:.2f} ({cls}), "
            f"τ = {tau:.1f}; max κ/κ_max {kr:.3f} ({kcls}); "
            + ", ".join(f"{c} ρ_max {x:.2f}" for c, x in s.rho_max.items())
            + (f"; {watch}" if watch else ""))


# ---- the run ----------------------------------------------------------------------------


class JudgingVerifiers:
    """A ``run_scenario`` verifier factory that judges with the band file's bands and keeps
    every verifier it builds, so each one's band source can be compared afterwards."""

    def __init__(self, c: DeclaredComputation, n_records: int,
                 w0: Mapping[str, torch.Tensor], bands: Bands, *, k: int, h_D: bytes,
                 seed: int) -> None:
        self._args = (c, n_records, w0, bands)
        self._kw = {"k": k, "h_D": h_D, "seed": seed}
        self.made: list[Verifier] = []

    def for_steps(self, T: int) -> Callable[[Section | None], Verifier]:
        make = judging_verifier(*self._args, T=T, **self._kw)  # type: ignore[arg-type]

        def build(section: Section | None) -> Verifier:
            v = make(section)
            self.made.append(v)
            return v
        return build


class KeptStepHandoff(StoreHandoff):
    """The in-memory hand-off, which also keeps one step's transcript on disk (EV2).

    Every step is held in memory and verified from there, as in every other run. :meth:`hold`
    keeps a reference to step ``step``'s committed leaves and tree; :meth:`write`, called once
    the step has closed, writes them to ``directory/step_<t>/`` with ``DiskHandoff(keep=True)``
    and checks that the root recomputed from the written files is the step's ``h``. The write
    falls outside every timed section and the loop's step clock, so the stored step costs what
    an in-memory step costs."""

    def __init__(self, directory: str | Path, step: int) -> None:
        self.disk = DiskHandoff(directory, keep=True)
        self.step = step
        self._held: tuple[DeclaredComputation, list[Any], MerkleTree,
                          Sequence[Sequence[bytes]] | None] | None = None
        self.kept: DiskStore | None = None

    @property
    def directory(self) -> Path:
        """The stored step's directory, once written or not."""
        return self.disk.directory / f"step_{self.step}"

    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> TranscriptStore:
        if t == self.step:
            self._held = (c, leaves, tree, dataset_paths)
        return IN_MEMORY.hold(c, t, leaves, tree, dataset_paths)

    def release(self, store: TranscriptStore) -> None:
        IN_MEMORY.release(store)

    def write(self) -> DiskStore | None:
        """Write the held step, or return ``None`` when the run never reached it. Raises
        ``RuntimeError`` if the files' root isn't the step's ``h``."""
        if self._held is None:
            return None
        c, leaves, tree, paths = self._held
        self._held = None
        store = self.disk.hold(c, self.step, leaves, tree, paths)
        root = transcript_root(c, store)
        if root != tree.root or store.root != tree.root:
            raise RuntimeError(f"stored step {self.step}: root over the written files "
                               f"{root.hex()} is not the step's h {tree.root.hex()}")
        self.kept = store
        return store


def kept_record(run: str, scenario: str, store: DiskStore, *, step: int,
                run_dir: Path) -> dict[str, Any]:
    """The ``kept_transcript`` record: the stored step's directory (relative to the run
    directory when inside it), its root ``h``, and its files' count and bytes."""
    files = [p for p in store.directory.iterdir() if p.is_file()]
    d = store.directory
    path = d.relative_to(run_dir) if d.is_relative_to(run_dir) else d
    return {"record": KEPT_RECORD, "run": run, "scenario": scenario, "step": step,
            "path": path.as_posix(), "root": store.root.hex(), "files": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


def default_keep_step(bands: Bands, seed: int) -> int:
    """The step stored when ``--keep-step`` isn't given: :data:`KEEP_STEP` in a run at the
    band file's calibration seed (H1, ``metrics.calibration_seed``), whose step 4 is the
    sweep's and the ``k`` run's (EVALUATION_SPEC §5.7–§5.8), else ``0`` (none): a run at
    another seed would store gigabytes nobody reads."""
    return KEEP_STEP if calibration_seed(bands) == seed else 0


def honest_run_name(model: str, attack: str, seed: int) -> str:
    """The honest cell's run name, ``honest-<model>-<attack>-bpr0-s<seed>``."""
    return run_name(HONEST.name, model, attack, 0, seed)


@dataclass
class VerifiedRun:
    result: ScenarioResult
    bands: Bands
    summaries: list[StepSummary]
    factory: JudgingVerifiers  # builds this run's verifiers, and later the passes'
    watch: str = WATCH_CLASS
    kept: DiskStore | None = None  # the stored step, when the run kept one

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
        return [(s.t, cls, x) for s in self.summaries if s.judged
                for cls, x in s.class_max.items() if x > line]


def run_verified(c: DeclaredComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
                 bands: Bands, *, T: int, k: int, seed: int, h_D: bytes,
                 final: Mapping[str, torch.Tensor],
                 recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None,
                 writer: MetricsWriter | None = None,
                 build_model: Callable[[], torch.nn.Module] | None = None,
                 keep: KeptStepHandoff | None = None,
                 watch: str = WATCH_CLASS, out: Callable[[str], None] = print) -> VerifiedRun:
    """``T`` honest steps judged live by ``bands`` (loaded from the band file).

    Each closed step is summarized (:func:`step_summary`), printed, and with ``writer``
    written as a ``residual_summary`` record. A step is in-sample when the bands were fitted
    on it (``metrics.in_sample_step``) and judged as ``metrics.judged_step`` says. ``keep``
    stores its step once the step closes, and with ``writer`` a ``kept_transcript`` record
    follows. ``watch`` names the class whose residuals are all logged."""
    factory = JudgingVerifiers(c, len(D), w0, bands, k=k, h_D=h_D, seed=seed)
    summaries: list[StepSummary] = []
    kept: list[DiskStore] = []

    def on_step(rec: StepRecord) -> None:
        v = factory.made[-1]
        st = v.stats.get(rec.t, StepStats())
        s = step_summary(rec.t, st, bands, in_sample=in_sample_step(bands, rec.t, seed),
                         judged=judged_step(v, rec.t), rejected=rec.rejection is not None,
                         watch=watch)
        summaries.append(s)
        out(_step_line(s, bands.tau))
        if writer is not None:
            writer.write([summary_record(writer.run, HONEST.name, s, bands, watch=watch)])
        if keep is not None and rec.t == keep.step:
            store = keep.write()
            if store is not None:
                kept.append(store)
                if writer is not None:
                    writer.write([kept_record(writer.run, HONEST.name, store, step=rec.t,
                                              run_dir=writer.dir)])

    r = run_scenario(c, D, w0, HONEST, T=T, k=k, seed=seed, final=final, on_step=on_step,
                     recorder=recorder, h_D=h_D, build_model=build_model,
                     verifier=factory.for_steps(T), handoff=keep)
    return VerifiedRun(r, bands, summaries, factory, watch, kept[0] if kept else None)


def calibration_band_hash(output_dir: Path, k: int) -> str | None:
    """The band-file hash the calibration run recorded (``calibrate/k<k>/records.jsonl``), or
    ``None`` when those records aren't there."""
    path = output_dir / "calibrate" / f"k{k}" / "records.jsonl"
    if not path.exists():
        return None
    rows = read_records(path, "calibration")
    return rows[-1].get("band_file_hash") if rows else None


# ---- reports ----------------------------------------------------------------------------


def baseline_seed(plain_dir: Path) -> int | None:
    """The seed the plain baseline in ``plain_dir`` ran at: its ``final_weights.json``'s
    ``seed``, else its ``environment`` record's (``records.jsonl``; a file written before the
    seed was recorded), else ``None``."""
    seed = read_final_weights(plain_dir / WEIGHTS_FILE).get("seed")
    records = plain_dir / "records.jsonl"
    if seed is None and records.exists():
        env = read_records(records, "environment")
        seed = env[0].get("seed") if env else None
    return None if seed is None else int(seed)


def compare_with_baseline(c: DeclaredComputation, plain_dir: Path, verified: Path, seed: int,
                          out: Callable[[str], None] = print) -> bool:
    """B7: compare the run's ``final_weights.json`` (``verified``) bit for bit with the plain
    baseline's in ``plain_dir`` (``plain_baseline.assert_same_final_weights``), and print the
    outcome. ``False`` only on a mismatch. With no baseline file, or a baseline at another seed
    than the run's (:func:`baseline_seed`; the two trained on different batches), the
    comparison is skipped and ``True`` returned. A baseline whose seed is unknown is
    compared."""
    plain = plain_dir / WEIGHTS_FILE
    if not plain.exists():
        out(f"B7: skipped, no plain baseline at {plain}; run verification.runs.plain_baseline "
            "to compare")
        return True
    base = baseline_seed(plain_dir)
    if base is not None and base != seed:
        out(f"B7: skipped, the plain baseline ran at seed {base} and this run at seed {seed}; "
            f"run verification.runs.plain_baseline at VERIF_SEED={seed} to compare")
        return True
    try:
        assert_same_final_weights(c, plain, verified)
    except AssertionError as e:
        out(f"B7: FAILED: {e}")
        return False
    out(f"B7: final weights bit-identical to the plain baseline's ({plain})")
    return True


def report_summaries(run: VerifiedRun, out: Callable[[str], None] = print) -> None:
    """The per-step tables: each class's largest residual, then its largest κ over κ_max,
    one column per step; the watched class's residuals; check 6's ρ."""
    ss = run.summaries
    if not ss:
        return
    tau = run.bands.tau
    line = GUARD_FRACTION * tau
    head = "".join(f"{s.t:>7}{s.mark}" for s in ss)
    classes = list(dict.fromkeys(c for s in ss for c in s.class_max))
    out(f"check 5: largest normalized residual per class and step (τ = {tau:.2f}, guard line "
        f"τ/2 = {line:.2f}; * = in-sample, - = not judged)")
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
            out(f"  step {s.t:>2}{s.mark} {name}: max {max(v):6.2f}  "
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
                   help=f"steps in the memory pass (default {PASS_STEPS}); the counting pass "
                        f"runs {COUNT_STEPS}")
    p.add_argument("--keep-step", type=int, default=None,
                   help=f"the step whose transcript is stored on disk, 0 for none (default "
                        f"{KEEP_STEP} at the band file's calibration seed, else none)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help="write B6 metrics to the run directory, $VERIF_OUTPUT_DIR/"
                        "honest-<model>-<attack>-bpr0-s<seed>/ (default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    T, S, k = (args.steps if args.steps is not None else cfg.steps), args.pass_steps, pc.k
    if T < 1 or not 1 <= S <= T or (args.keep_step is not None and args.keep_step < 0):
        p.error(f"need T ≥ 1, 1 ≤ --pass-steps ≤ T and --keep-step ≥ 0, got T = {T}, {S}, "
                f"{args.keep_step}")
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    # read-only; never refitted here (P10a)
    bands = load_bands(pc.band_file, k=k, context=band_context(cfg, c, h_D=h_D, k=k))
    keep_step = (default_keep_step(bands, cfg.seed) if args.keep_step is None
                 else args.keep_step)
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())  # as B7 takes it
    final = honest_final(c, D, w0, T, seed=cfg.seed, build_model=models)
    print(f"A12 verified run: {cfg.model}@{cfg.model_revision[:12]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, k {k}, T {T}, |D| {len(D)}, h_D {h_D.hex()[:16]}…, M {c.M}; "
          f"band file {pc.band_file} ({bands.source[:16]}…), τ {bands.tau:.2f}, in-sample "
          f"steps {[t for t in range(1, T + 1) if in_sample_step(bands, t, cfg.seed)]}")
    name = honest_run_name(cfg.model, cfg.attack, cfg.seed)
    run_dir = cfg.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    llama = {"n_s": c.n_s, "n": c.n, "eta": c.eta, "k": k, "T": T, "n_records": len(D),
             "M": c.M, "n_leaves": c.n_leaves, "pass_steps": S, "count_steps": COUNT_STEPS,
             "keep_step": keep_step}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics")}, "llama": llama}
    base = dict(k=k, seed=cfg.seed, h_D=h_D, build_model=models)
    finals = {T: final}

    def final_at(n: int) -> Mapping[str, torch.Tensor]:
        if n not in finals:
            finals[n] = honest_final(c, D, w0, n, seed=cfg.seed, build_model=models)
        return finals[n]

    keep = (KeptStepHandoff(run_dir / TRANSCRIPTS_DIR, keep_step)
            if 1 <= keep_step <= T else None)
    mem_rows = None
    mw = (MetricsWriter(run_dir, name, device=cfg.device, model=cfg.model,
                        corpus=backdoorllm.SOURCE, attack=cfg.attack, seed=cfg.seed,
                        config=config, band_file_hash=bands.source, h_D=h_D,
                        extra={"band_source": str(pc.band_file), **llama})
          if use_metrics else None)
    try:
        rec = (None if mw is None else
               mw.recorder(HONEST.name, bpr_records=0, bpr_actual=0.0))
        run = run_verified(c, D, w0, bands, T=T, final=final, writer=mw, recorder=rec,
                           keep=keep, **base)
        r = run.result
        weights_doc = write_final_weights(run_dir / WEIGHTS_FILE, c, r.loop.w_final or {},
                                          w0=w0, losses=[s.loss for s in r.loop.steps],
                                          provenance=run_provenance(cfg, h_D), run=name)
        if mw is not None:
            mem_rows = memory_run(c, D, w0, run=name, device=mw.device, T=S, final=final_at(S),
                                  verifier=run.factory.for_steps(S), **base)
            mw.write(mem_rows)
            mw.write(count_run(c, D, w0, run=name, T=COUNT_STEPS, final=final_at(COUNT_STEPS),
                               verifier=run.factory.for_steps(COUNT_STEPS), **base))
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

    if run.kept is not None:
        kr = kept_record(name, HONEST.name, run.kept, step=keep_step, run_dir=run_dir)
        print(f"stored step {keep_step}: {run.kept.directory}, {kr['files']} files, "
              f"{kr['bytes'] / 2**30:.2f} GiB, root {kr['root'][:16]}… (recomputed from the "
              "files)")
    elif keep is not None:
        print(f"stored step: none, the run stopped before step {keep_step}")
    elif args.keep_step is None and keep_step == 0:
        print(f"stored step: none, the run's seed {cfg.seed} isn't the band file's calibration "
              f"seed {calibration_seed(bands)} (--keep-step N stores step N)")
    print(f"final weights: root {weights_doc['root'][:16]}…  ({run_dir / WEIGHTS_FILE})")
    ok = compare_with_baseline(c, cfg.output_dir / PLAIN_RUN, run_dir / WEIGHTS_FILE,
                               cfg.seed) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
