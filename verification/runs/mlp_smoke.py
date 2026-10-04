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
run (P10a). Exits 1 if any oracle fails.

With metrics on (``--metrics``, default ``VERIF_METRICS=1``), every scenario's timed run writes
its step records, per-component times and residual arrays to ``$VERIF_OUTPUT_DIR/mlp_smoke/``
(B6, ``runs/metrics.py``). Two separate passes of two honest steps each, never timed, write the
per-component memory peaks and the counts (FLOPs, bytes hashed, hash calls, transcript bytes).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from verification.computation.interface import DeclaredComputation
from verification.parameters import load_protocol_config
from verification.prover.step import StepOutput, plain_step
from verification.runs.loop import LoopResult, ProverFault, StepRecord, run_loop, run_plain
from verification.runs.metrics import (
    CountRecorder,
    MemoryRecorder,
    PASS_STEPS,
    MetricsWriter,
    TimeRecorder,
    count_pass,
    memory_pass,
    memory_probe,
)
from verification.verifier.bands import Bands
from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.context import Rejection
from verification.verifier.driver import Verifier

__all__ = ["Expected", "Scenario", "ScenarioResult", "scenarios", "honest_final",
           "run_scenario", "run_smoke", "memory_smoke", "count_smoke", "main"]

RUN_NAME = "mlp_smoke"  # the metrics directory under VERIF_OUTPUT_DIR

N_S = 4
FAULT_STEP = 2  # flip and forged W_{t+1}
CHAIN_STEP = 2  # the first step after the hidden one (P11)
ULPS = 300


@dataclass(frozen=True)
class Expected:
    """A declared outcome: ``step is None`` means the run is accepted."""

    step: int | None = None
    check_id: str | None = None
    kind: str | None = None

    def __str__(self) -> str:
        if self.step is None:
            return "accept"
        return f"reject at ({self.step}, {self.check_id!r}, {self.kind})"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    fault: ProverFault
    expected: Expected


@dataclass
class ScenarioResult:
    scenario: Scenario
    loop: LoopResult
    verifier: Verifier
    passed: bool

    @property
    def actual(self) -> Expected:
        rej = self.loop.rejection
        return Expected() if rej is None else Expected(rej.step, rej.check_id, rej.kind)


# ---- the faults (prover side only, invariant 5) -----------------------------------------


def _other_batch(dataset: Sequence[Any], n_s: int) -> list[Any]:
    """The last ``n_s`` records of ``D``: no early step of ``π`` reaches them."""
    return list(dataset[-n_s:])


class FlipProduct(ProverFault):
    """Flip the sign of the largest-magnitude entry of product ``m`` at step ``t``."""

    def __init__(self, t: int, m: int) -> None:
        self.t, self.m = t, m

    def perturb(self, t: int):
        if t != self.t:
            return None

        def flip(p: torch.Tensor) -> torch.Tensor:
            p = p.contiguous().clone()  # never write into autograd's buffer; view needs contiguity
            i = int(p.abs().reshape(-1).argmax())
            p.view(-1)[i] = -p.view(-1)[i]
            return p
        return {self.m: flip}


class NudgeWNext(ProverFault):
    """Move entry ``i`` of ``W_{t+1}[name]`` away from zero by ``ulps`` units in the last place."""

    def __init__(self, t: int, name: str, i: int, ulps: int) -> None:
        self.t, self.name, self.i, self.ulps = t, name, i, ulps

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        if t != self.t:
            return out
        w = out.w_next[self.name].clone()
        w.view(-1).view(torch.int32)[self.i] += self.ulps  # sign-magnitude: |w_i| grows
        return out.with_w_next({**out.w_next, self.name: w})


class TrainedElsewhereWNext(ProverFault):
    """A3: an honest step ``t`` whose ``W_{t+1}`` comes from training on another batch."""

    def __init__(self, c: MLPComputation, t: int, batch: Sequence[Any]) -> None:
        self.c, self.t, self.batch = c, t, list(batch)

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        if t != self.t:
            return out
        forged, _ = plain_step(self.c, self.c.build_model(), out.w_t, self.batch)
        return out.with_w_next(forged)


class HiddenStep(ProverFault):
    """One unreported SGD step on another batch, just before reported step ``t`` (P11)."""

    def __init__(self, c: MLPComputation, t: int, batch: Sequence[Any]) -> None:
        self.c, self.t, self.batch = c, t, list(batch)

    def entry_weights(self, t: int, w: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if t != self.t:
            return w
        hidden, _ = plain_step(self.c, self.c.build_model(), w, self.batch)
        return hidden


def scenarios(c: MLPComputation, dataset: Sequence[Any]) -> list[Scenario]:
    other = _other_batch(dataset, c.n_s)
    m = c.m_of("Y_2")
    return [
        Scenario("honest", "no fault", ProverFault(), Expected()),
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


# ---- running and judging ----------------------------------------------------------------


def honest_final(c: DeclaredComputation, dataset: Sequence[Any],
                 w0: Mapping[str, torch.Tensor], T: int) -> dict[str, torch.Tensor]:
    """The agreed final weights: ``T`` uncaptured honest steps from ``W_0`` on ``π``."""
    return run_plain(c, c.build_model(), dataset, w0, n_steps=T).w_final


def judge(expected: Expected, loop: LoopResult, T: int) -> bool:
    """The S6b oracle: the run ends exactly at its declared outcome."""
    v = loop.verdict
    if expected.step is None:
        return v.accepted and v.rejection is None and v.steps_verified == T
    rej = v.rejection
    return (not v.accepted and rej is not None
            and (rej.step, rej.check_id, rej.kind) == (expected.step, expected.check_id,
                                                       expected.kind))


def run_scenario(c: DeclaredComputation, dataset: Sequence[Any],
                 w0: Mapping[str, torch.Tensor], scenario: Scenario, *, T: int, k: int,
                 final: Mapping[str, torch.Tensor],
                 on_step: Callable[[StepRecord], None] | None = None,
                 recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None,
                 h_D: bytes | None = None) -> ScenarioResult:
    """One scenario's run. ``recorder`` (B6) observes it through the ``section`` seams.

    ``h_D`` is the agreed dataset root the verifier's check 1 compares with; by default it is
    computed from ``dataset`` (the MLP's synthetic ``D`` has no published root)."""
    section = None if recorder is None else recorder.section
    if h_D is None:
        h_D = dataset_tree(c, dataset).root
    v = Verifier(c, h_D=h_D, n_records=len(dataset), k=k, n_steps=T,
                 bands=Bands.provisional(), w0=w0, allow_provisional=True, section=section)
    if recorder is not None:
        recorder.bind(v)
        rec, user = recorder, on_step

        def on_step(r: StepRecord) -> None:
            rec.on_step(r)
            if user is not None:
                user(r)
    loop = run_loop(c, c.build_model(), dataset, w0, v, final=final, fault=scenario.fault,
                    on_step=on_step, section=section)
    if recorder is not None:
        recorder.finish(loop.verdict)
    return ScenarioResult(scenario, loop, v, judge(scenario.expected, loop, T))


def _fmt(x: float | None, width: int, spec: str = ".3g") -> str:
    return ("-" if x is None else format(x, spec)).rjust(width)


def report(r: ScenarioResult, out: Callable[[str], None] = print) -> None:
    """The scenario's oracle, then per step: check 5's max normalized residual and max κ,
    check 6a's ρ_max, the prover's time and each check's time."""
    s, v = r.scenario, r.verifier
    out(f"== {s.name}: {s.description}")
    out(f"   expected: {s.expected}")
    rej = r.loop.rejection
    out(f"   actual:   {r.actual}" + ("" if rej is None else f": {rej.detail}"))
    ids = [("0/7" if c == "7" else c) for c in DEFAULT_ORDER]
    out("   step  max-norm-res  max-κ  ρ_max(6a) |  prove commit |"
        + "".join(f"{i:>7}" for i in ids) + "   (ms; check 0 runs in 7's slot at step 1)")
    for rec in r.loop.steps:
        st = v.stats.get(rec.t)
        products = st.products if st else []
        rhos = [x.rho_max for x in st.tensors if x.check_id == "6a"] if st else []
        tm = v.timings.get(rec.t, {})
        out(f"   {rec.t:>4}  {_fmt(st.max_normalized() if st and products else None, 12)}"
            f"  {_fmt(st.max_kappa() if st and products else None, 5)}"
            f"  {_fmt(max(rhos) if rhos else None, 9)} |"
            f" {rec.prove_s * 1e3:6.2f} {rec.commit_s * 1e3:6.2f} |"
            + "".join(_fmt(tm[c] * 1e3 if c in tm else None, 7, ".2f") for c in DEFAULT_ORDER))
    out("   run checks: " + ", ".join(f"{c} {t * 1e3:.2f} ms" for c, t in v.run_timings.items()))
    out(f"   {'PASS' if r.passed else 'FAIL'}")


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


def _two_honest_steps(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
                      k: int) -> Callable[[Any], Any]:
    """A pass's body: ``PASS_STEPS`` honest steps (check 0 at step 1, check 7 at step 2), run on
    its own."""
    T = PASS_STEPS
    final = honest_final(c, dataset, w0, T)
    honest = next(s for s in scenarios(c, dataset) if s.name == "honest")
    return lambda rec: run_scenario(c, dataset, w0, honest, T=T, k=k, final=final, recorder=rec)


def memory_smoke(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor], *,
                 k: int, device: str | torch.device = "cpu",
                 run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 memory pass: per-component peaks over two honest steps, never timed."""
    return memory_pass(run, "honest", _two_honest_steps(c, dataset, w0, k),
                       memory_probe(device))


def count_smoke(c: MLPComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor], *,
                k: int, run: str = RUN_NAME) -> list[dict[str, Any]]:
    """The B6 counting pass: counts over two honest steps, never timed."""
    return count_pass(run, "honest", _two_honest_steps(c, dataset, w0, k))


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
