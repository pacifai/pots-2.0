"""The scenario harness: run a declared computation through the S3 loop and judge the outcome.

Instance-agnostic: the MLP smoke run (M1), the SmolLM2 runs (M3 onward) and the metrics overhead
run share it. A :class:`Scenario` pairs a prover-side fault (:class:`~verification.runs.loop.
ProverFault`, invariant 5) with its declared outcome (:class:`Expected`, S6b, S6d);
:func:`run_scenario` runs it against the real :class:`~verification.verifier.driver.Verifier`
with provisional bands (``allow_provisional=True``, P10a) and :func:`judge` is the oracle: any
other outcome is a FAIL; :func:`outcome` names how a run missed its declared point.
:func:`honest_final` is check 8's agreed final weights.

The generic prover faults live here too, so the MLP smoke run (M1) and the SmolLM2 cheat runs
(A13) share them: :class:`FlipProduct`, :class:`NudgeWNext`, :class:`TrainedElsewhereWNext`,
:class:`HiddenStep` (P11), :class:`CommitBatch` (A1), :class:`TrainOn` (A2),
:class:`SpliceWNext` (A3) and :class:`KeepStep` (keeps one step's output for the sweep).

:func:`memory_run` and :func:`count_run` are B6's two untimed passes of one scenario
(``runs/metrics.py``).

Each run builds the prover's model with ``c.build_model()`` unless it is given
``build_model``. A :class:`ReusedModel` hands one built model to successive runs (``W_0``,
:func:`honest_final`, the prover), which saves a checkpoint load per run at SmolLM2 scale.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from verification.commitment.leaves import dataset_tree
from verification.computation.interface import DeclaredComputation
from verification.prover.step import StepOutput, plain_step
from verification.runs.loop import LoopResult, ProverFault, StepRecord, run_loop, run_plain
from verification.runs.metrics import (
    CountRecorder,
    MemoryRecorder,
    TimeRecorder,
    count_pass,
    memory_pass,
    memory_probe,
)
from verification.verifier.bands import Bands
from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.context import Rejection, Section
from verification.verifier.driver import Verifier

__all__ = ["Expected", "Scenario", "ScenarioResult", "HONEST", "ReusedModel", "honest_final",
           "rejected_product", "judge", "outcome", "OUTCOMES", "run_scenario", "report",
           "memory_run", "count_run", "FlipProduct", "NudgeWNext", "TrainedElsewhereWNext",
           "HiddenStep", "CommitBatch", "TrainOn", "SpliceWNext", "KeepStep"]

BuildModel = Callable[[], torch.nn.Module]
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


@dataclass(frozen=True)
class Expected:
    """A declared outcome: ``step is None`` means the run is accepted.

    ``product`` pins a check-5 rejection to one product ``m`` (S6d: A2 at layer 1's ``Y_q``,
    the flipped matmul at the product it flips). ``None`` accepts any product."""

    step: int | None = None
    check_id: str | None = None
    kind: str | None = None
    product: int | None = None

    def __str__(self) -> str:
        if self.step is None:
            return "accept"
        at = "" if self.product is None else f", P_{self.product}"
        return f"reject at ({self.step}, {self.check_id!r}, {self.kind}{at})"


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
        """The outcome reached, in the declaration's form: it names the product only when
        the scenario declares one, so ``actual == expected`` exactly when the oracle passes
        on a rejection."""
        rej = self.loop.rejection
        if rej is None:
            return Expected()
        m = None if self.scenario.expected.product is None else rejected_product(rej)
        return Expected(rej.step, rej.check_id, rej.kind, m)

    @property
    def outcome(self) -> str:
        """How the run ended against its declaration (:func:`outcome`)."""
        return outcome(self.scenario.expected, self.loop)


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


_PRODUCT = re.compile(r"P_(\d+) \(")


def rejected_product(rej: Rejection | None) -> int | None:
    """The product ``m`` a check-5 rejection names (``"P_{m} (name)…"``), else ``None``."""
    if rej is None or rej.check_id != "5":
        return None
    hit = _PRODUCT.match(rej.detail)
    return None if hit is None else int(hit.group(1))


def judge(expected: Expected, loop: LoopResult, T: int) -> bool:
    """The S6b oracle: the run ends exactly at its declared outcome."""
    v = loop.verdict
    if expected.step is None:
        return v.accepted and v.rejection is None and v.steps_verified == T
    rej = v.rejection
    return (not v.accepted and rej is not None
            and (rej.step, rej.check_id, rej.kind) == (expected.step, expected.check_id,
                                                       expected.kind)
            and (expected.product is None or rejected_product(rej) == expected.product))


# How a run can end against its declared outcome. Only "exact" passes the oracle.
OUTCOMES = ("exact", "accepted", "rejected", "early", "overrun", "wrong-check", "wrong-kind",
            "wrong-product")


def outcome(expected: Expected, loop: LoopResult) -> str:
    """Name how the run ended against ``expected``, one of :data:`OUTCOMES`.

    - ``exact``: the declared outcome. For a declared rejection that is the same step, check,
      kind and, if declared, product. It is the only passing outcome.
    - ``accepted``: a cheat the run accepted. ``rejected``: an honest run that was rejected.
    - ``early`` and ``overrun``: rejected at an earlier or a later step than declared. A cheat
      run cut at its declared step (S6b) shows a miss as ``accepted``, or as check 8 at that
      step (``wrong-check``).
    - ``wrong-check``, ``wrong-kind`` and ``wrong-product``: the declared step, but another
      check (check 8 included), the other kind, or another product of check 5.
    """
    rej = loop.verdict.rejection
    if expected.step is None:
        return "exact" if rej is None and loop.verdict.accepted else "rejected"
    if rej is None:
        return "accepted"
    if rej.step != expected.step:
        return "early" if rej.step < expected.step else "overrun"
    if rej.check_id != expected.check_id:
        return "wrong-check"
    if rej.kind != expected.kind:
        return "wrong-kind"
    if expected.product is not None and rejected_product(rej) != expected.product:
        return "wrong-product"
    return "exact"


def run_scenario(c: DeclaredComputation, dataset: Sequence[Any],
                 w0: Mapping[str, torch.Tensor], scenario: Scenario, *, T: int, k: int,
                 final: Mapping[str, torch.Tensor],
                 on_step: Callable[[StepRecord], None] | None = None,
                 recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None,
                 h_D: bytes | None = None,
                 build_model: BuildModel | None = None,
                 verifier: Callable[[Section | None], Verifier] | None = None) -> ScenarioResult:
    """One scenario's run. ``recorder`` (B6) observes it through the ``section`` seams.

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


# ---- the generic prover faults (prover side only, invariant 5) ---------------------------


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
    """A3: an honest step ``t`` whose ``W_{t+1}`` comes from training on another batch.

    ``build_model`` (default ``c.build_model``) gives the model the forged step runs on."""

    def __init__(self, c: DeclaredComputation, t: int, batch: Sequence[Any], *,
                 build_model: BuildModel | None = None) -> None:
        self.c, self.t, self.batch = c, t, list(batch)
        self.build_model = build_model or c.build_model

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        if t != self.t:
            return out
        forged, _ = plain_step(self.c, self.build_model(), out.w_t, self.batch)
        return out.with_w_next(forged)


class HiddenStep(ProverFault):
    """One unreported SGD step on another batch, just before reported step ``t`` (P11).

    ``build_model`` (default ``c.build_model``) gives the model the hidden step runs on."""

    def __init__(self, c: DeclaredComputation, t: int, batch: Sequence[Any], *,
                 build_model: BuildModel | None = None) -> None:
        self.c, self.t, self.batch = c, t, list(batch)
        self.build_model = build_model or c.build_model

    def entry_weights(self, t: int, w: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if t != self.t:
            return w
        hidden, _ = plain_step(self.c, self.build_model(), w, self.batch)
        return hidden


class _KeepsWNext(ProverFault):
    """Keeps the ``W_{t+1}`` that step ``t`` emitted, as :attr:`w_next`. S6e: the poisoned
    step's ``W_1`` is A3's forged update."""

    def __init__(self, t: int, batch: Sequence[Any]) -> None:
        self.t, self.batch = t, list(batch)
        self.w_next: dict[str, torch.Tensor] | None = None

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        if t == self.t:
            self.w_next = dict(out.w_next)
        return out


class CommitBatch(_KeepsWNext):
    """A1: commit ``batch`` as step ``t``'s records and train on it, truthfully. The audit
    paths stay ``π(t)``'s, so check 4 sees records that don't hash into ``h_D`` there."""

    def committed_records(self, t: int, records: Sequence[Any]) -> Sequence[Any]:
        return self.batch if t == self.t else records


class TrainOn(_KeepsWNext):
    """A2: commit ``π(t)``'s records but train step ``t`` on ``batch``, so every product and
    ``W_{t+1}`` come from ``batch``."""

    def train_records(self, t: int, records: Sequence[Any]) -> Sequence[Any] | None:
        return self.batch if t == self.t else None


class SpliceWNext(ProverFault):
    """A3 with a precomputed forgery: an honest step ``t`` committed with ``w_next`` as its
    ``W_{t+1}``."""

    def __init__(self, t: int, w_next: Mapping[str, torch.Tensor]) -> None:
        self.t, self.w_next = t, dict(w_next)

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        return out.with_w_next(self.w_next) if t == self.t else out


class KeepStep(ProverFault):
    """Honest. Keeps step ``t``'s ``StepOutput`` as :attr:`out`: the flipped-matmul sweep
    perturbs that transcript (P10c, S6f)."""

    def __init__(self, t: int) -> None:
        self.t = t
        self.out: StepOutput | None = None

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        if t == self.t:
            self.out = out
        return out


# ---- B6's untimed passes ----------------------------------------------------------------


def memory_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               *, run: str, T: int, k: int, final: Mapping[str, torch.Tensor],
               h_D: bytes | None = None, scenario: Scenario = HONEST,
               device: str | torch.device = "cpu",
               build_model: BuildModel | None = None,
               verifier: Callable[[Section | None], Verifier] | None = None
               ) -> list[dict[str, Any]]:
    """The B6 memory pass: ``scenario`` over ``T`` steps, probed at every section, never timed.
    ``verifier`` is as in :func:`run_scenario`."""
    return memory_pass(run, scenario.name, lambda rec: run_scenario(
        c, dataset, w0, scenario, T=T, k=k, final=final, recorder=rec, h_D=h_D,
        build_model=build_model, verifier=verifier),
        memory_probe(device))


def count_run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
              *, run: str, T: int, k: int, final: Mapping[str, torch.Tensor],
              h_D: bytes | None = None, scenario: Scenario = HONEST,
              build_model: BuildModel | None = None,
              verifier: Callable[[Section | None], Verifier] | None = None
              ) -> list[dict[str, Any]]:
    """The B6 counting pass: ``scenario`` over ``T`` steps, counted, never timed.
    ``verifier`` is as in :func:`run_scenario`."""
    return count_pass(run, scenario.name, lambda rec: run_scenario(
        c, dataset, w0, scenario, T=T, k=k, final=final, recorder=rec, h_D=h_D,
        build_model=build_model, verifier=verifier))
