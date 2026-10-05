"""The scenario harness: run a declared computation through the S3 loop and judge the outcome.

Instance-agnostic: the MLP smoke run (M1), the SmolLM2 runs (M3 onward) and the metrics overhead
run share it. A :class:`Scenario` pairs a prover-side fault (:class:`~verification.runs.loop.
ProverFault`, invariant 5) with its declared outcome (:class:`Expected`, S6b, S6d);
:func:`run_scenario` runs it against the real :class:`~verification.verifier.driver.Verifier`
with provisional bands (``allow_provisional=True``, P10a) and :func:`judge` is the oracle: any
other outcome is a FAIL. :func:`honest_final` is check 8's agreed final weights.

:func:`memory_run` and :func:`count_run` are B6's two untimed passes of one scenario
(``runs/metrics.py``).

Each run builds the prover's model with ``c.build_model()`` unless it is given
``build_model``. A :class:`ReusedModel` hands one built model to successive runs (``W_0``,
:func:`honest_final`, the prover), which saves a checkpoint load per run at SmolLM2 scale.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from verification.commitment.leaves import dataset_tree
from verification.computation.interface import DeclaredComputation
from verification.runs.loop import LoopResult, ProverFault, StepRecord, run_loop, run_plain
from verification.runs.metrics import (
    CountRecorder,
    MemoryRecorder,
    TimeRecorder,
    count_pass,
    memory_pass,
    memory_probe,
)
from verification.transcript.store import StoreHandoff
from verification.verifier.bands import Bands
from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.context import Section
from verification.verifier.driver import Verifier

__all__ = ["Expected", "Scenario", "ScenarioResult", "HONEST", "ReusedModel", "honest_final",
           "judge", "run_scenario", "report", "memory_run", "count_run"]

BuildModel = Callable[[], torch.nn.Module]
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


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

HONEST = Scenario("honest", "no fault", ProverFault(), Expected())


# ---- running and judging ----------------------------------------------------------------


class ReusedModel:
    """One model, built once by ``build`` and handed out again on every call.

    The harness's runs are sequential, and each loads its weights into every parameter before
    use (``load_weights`` requires full coverage), so a later run can reuse an earlier run's
    model: the checkpoint is loaded once, not once per run. Each call first checks that the
    model is as built, apart from its parameter values, and raises ``RuntimeError`` if not:

    - no module hook is left on it;
    - every parameter still requires grad and holds no gradient;
    - every buffer (SmolLM2's RoPE ``inv_freq``) is bit for bit its value at build time;
    - the train/eval mode is unchanged.

    Only the harness uses it. The verifier keeps its own model, never this one (invariant 1).
    """

    def __init__(self, build: BuildModel) -> None:
        self.model = build()
        self._buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        self._training = self.model.training

    def __call__(self) -> torch.nn.Module:
        m = self.model
        hooked = [n or "<root>" for n, mod in m.named_modules()
                  if any(getattr(mod, h) for h in _HOOK_DICTS)]
        if hooked:
            raise RuntimeError(f"reused model: hooks left on {hooked[:3]}")
        bad = [n for n, p in m.named_parameters() if p.grad is not None or not p.requires_grad]
        if bad:
            raise RuntimeError(f"reused model: gradient state changed on {bad[:3]}")
        buffers = dict(m.named_buffers())
        if buffers.keys() != self._buffers.keys() or any(
                b.dtype != self._buffers[n].dtype or b.shape != self._buffers[n].shape
                or not torch.equal(b, self._buffers[n]) for n, b in buffers.items()):
            raise RuntimeError("reused model: buffers changed since build")
        if m.training != self._training:
            raise RuntimeError("reused model: train/eval mode changed")
        return m


def honest_final(c: DeclaredComputation, dataset: Sequence[Any],
                 w0: Mapping[str, torch.Tensor], T: int, *,
                 build_model: BuildModel | None = None) -> dict[str, torch.Tensor]:
    """The agreed final weights: ``T`` uncaptured honest steps from ``W_0`` on ``π``.

    ``build_model`` (default ``c.build_model``) gives the model the steps run on. The result
    is copies (``snapshot_weights``) that share no storage with that model, so a prover that
    trains the same model later can't change check 8's reference.
    """
    model = (build_model or c.build_model)()
    final = run_plain(c, model, dataset, w0, n_steps=T).w_final
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    if any(w.untyped_storage().data_ptr() in params for w in final.values()):
        raise RuntimeError("honest_final: the final weights alias the model's parameters")
    return final


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
                 h_D: bytes | None = None,
                 build_model: BuildModel | None = None,
                 verifier: Callable[[Section | None], Verifier] | None = None,
                 handoff: StoreHandoff | None = None) -> ScenarioResult:
    """One scenario's run. ``recorder`` (B6) observes it through the ``section`` seams.
    ``handoff`` is the loop's store hand-off (default in memory; C3 passes a ``DiskHandoff``).

    ``h_D`` is the agreed dataset root the verifier's check 1 compares with; by default it is
    computed from ``dataset`` (the MLP's synthetic ``D`` has no published root).
    ``build_model`` (default ``c.build_model``) gives the prover's model. ``verifier`` builds
    the verifier from the metrics seam, in place of the default one with provisional bands:
    C1 (``runs/calibrate.py``) builds one in calibration mode or with the band file."""
    section = None if recorder is None else recorder.section
    if h_D is None:
        h_D = dataset_tree(c, dataset).root
    if verifier is None:
        v = Verifier(c, h_D=h_D, n_records=len(dataset), k=k, n_steps=T,
                     bands=Bands.provisional(), w0=w0, allow_provisional=True, section=section)
    else:
        v = verifier(section)
        if v.section is not section or v.n_steps != T or v.k != k:
            raise ValueError("the given verifier must use this run's seam, T and k")
    if recorder is not None:
        recorder.bind(v)
        rec, user = recorder, on_step

        def on_step(r: StepRecord) -> None:
            rec.on_step(r)
            if user is not None:
                user(r)
    model = (build_model or c.build_model)()
    loop = run_loop(c, model, dataset, w0, v, final=final, fault=scenario.fault,
                    on_step=on_step, section=section, handoff=handoff)
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


# ---- B6's untimed passes ----------------------------------------------------------------


def memory_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               *, run: str, T: int, k: int, final: Mapping[str, torch.Tensor],
               h_D: bytes | None = None, scenario: Scenario = HONEST,
               device: str | torch.device = "cpu",
               build_model: BuildModel | None = None,
               verifier: Callable[[Section | None], Verifier] | None = None,
               handoff: StoreHandoff | None = None
               ) -> list[dict[str, Any]]:
    """The B6 memory pass: ``scenario`` over ``T`` steps, probed at every section, never timed.
    ``verifier`` and ``handoff`` are as in :func:`run_scenario`."""
    return memory_pass(run, scenario.name, lambda rec: run_scenario(
        c, dataset, w0, scenario, T=T, k=k, final=final, recorder=rec, h_D=h_D,
        build_model=build_model, verifier=verifier, handoff=handoff),
        memory_probe(device))


def count_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
              *, run: str, T: int, k: int, final: Mapping[str, torch.Tensor],
              h_D: bytes | None = None, scenario: Scenario = HONEST,
              build_model: BuildModel | None = None,
              verifier: Callable[[Section | None], Verifier] | None = None,
              handoff: StoreHandoff | None = None
              ) -> list[dict[str, Any]]:
    """The B6 counting pass: ``scenario`` over ``T`` steps, counted, never timed.
    ``verifier`` and ``handoff`` are as in :func:`run_scenario`."""
    return count_pass(run, scenario.name, lambda rec: run_scenario(
        c, dataset, w0, scenario, T=T, k=k, final=final, recorder=rec, h_D=h_D,
        build_model=build_model, verifier=verifier, handoff=handoff))
