"""The per-step run loop: prover → store → verifier → discard (S3, spec §3.6).

:func:`run_loop` drives one run of any :class:`DeclaredComputation`. For each step ``t`` it

1. has the prover compute ``C(b_t, W_t)`` on ``π(t)``'s batch
   (:func:`~verification.prover.step.prove_step`);
2. commits the step and hands it to a store through a
   :class:`~verification.transcript.store.StoreHandoff`, with each record's audit path into
   ``h_D``: an ``InMemoryStore`` by default, or a ``DiskStore`` (C3's cross-check);
3. when the model is on CUDA, hands the caching allocator's free blocks back to the driver
   (:func:`release_device_cache`, unless ``VERIF_RELEASE_CACHE=0`` turns it off), so the
   verifier allocates in memory the prover has left;
4. hands the verifier the store, and only the store (invariant 1);
5. drops the step's transcript (the handoff's ``release``) before the next step begins.

It stops at the first rejection, since a rejected run stays rejected, and then asks the
verifier for check 9's verdict, with check 8 against the agreed final weights.

Faults enter only on the prover side (S6a, invariant 5), through a :class:`ProverFault`. The
verifier is built by the caller from public inputs and never sees the fault.

:func:`run_plain` is the same training with the protocol off: ``plain_step`` on the same ``π``,
nothing committed or verified. It gives the plain baseline (B7) and check 8's agreed weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from setup import data
from setup.config import assert_no_dropout, load_config
from verification.commitment.leaves import dataset_tree, leaf_totals
from verification.computation.interface import DeclaredComputation
from verification.prover.step import (
    Perturbation,
    Section,
    StepOutput,
    no_section,
    plain_step,
    prove_step,
)
from verification.transcript.store import IN_MEMORY, InMemoryStore, StoreHandoff
from verification.verifier.context import Rejection
from verification.verifier.driver import RunVerdict, Verifier

__all__ = ["ProverFault", "StepRecord", "LoopResult", "run_loop", "PlainStepRecord", "PlainResult",
           "run_plain", "HANDOFF_SECTION", "TOTALS_SECTION", "release_device_cache",
           "release_cache_setting", "model_device"]

# The metrics section around the hand-off's cache release. Its row is the harness's (side
# "run") and in no role's total: deployed, the prover and the verifier run on separate machines.
# TOTALS_SECTION is the same for ``leaf_totals``, which counts the step's transcript bytes.
HANDOFF_SECTION = "handoff.empty_cache"
TOTALS_SECTION = "handoff.totals"


class ProverFault:
    """A prover-side deviation from the honest step. Every hook defaults to honest.

    - :meth:`entry_weights`: the weights step ``t`` actually starts from. Hidden steps (S5c,
      P11) run ``plain_step`` here, between two reported steps.
    - :meth:`committed_records`: the batch committed as step ``t``'s record leaves, in place of
      ``π(t)``'s (A1, with or without :meth:`train_records`). The audit paths stay ``π(t)``'s.
    - :meth:`train_records`: train on another batch than the one committed (A2).
    - :meth:`perturb`: ``prove_step``'s post-capture product perturbation (the flipped matmul).
    - :meth:`emit`: rewrite the step's output before commitment, such as splicing a forged
      ``W_{t+1}`` with :meth:`StepOutput.with_w_next` (A3).
    """

    def entry_weights(self, t: int, w: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return w

    def committed_records(self, t: int, records: Sequence[Any]) -> Sequence[Any]:
        return records

    def train_records(self, t: int, records: Sequence[Any]) -> Sequence[Any] | None:
        return None

    def perturb(self, t: int) -> Mapping[int, Perturbation] | None:
        return None

    def emit(self, t: int, out: StepOutput) -> StepOutput:
        return out


HONEST = ProverFault()


@dataclass(frozen=True)
class StepRecord:
    """One step of the loop: the verifier's decision and the wall clocks (seconds).

    ``handoff_s`` is the time between the two roles: dropping the prover's references and, on
    CUDA, :func:`release_device_cache`. It's in neither ``commit_s`` nor ``verify_s``. The
    harness section ``handoff.totals`` (``leaf_totals``) is in none of the four times."""

    t: int
    rejection: Rejection | None
    loss: float
    prove_s: float
    commit_s: float
    verify_s: float
    handoff_s: float = 0.0
    capture: dict | None = None
    transcript_bytes: int | None = None
    leaves: int | None = None


@dataclass
class LoopResult:
    """The run's verdict, the steps that ran, and the prover's last committed ``W_{t+1}``.

    Per-check timings and statistics stay on the verifier (``timings``, ``run_timings``,
    ``stats``).
    """

    verdict: RunVerdict
    steps: list[StepRecord] = field(default_factory=list)
    w_final: dict[str, torch.Tensor] | None = None

    @property
    def rejection(self) -> Rejection | None:
        return self.verdict.rejection


def release_cache_setting(release_cache: bool | None) -> bool:
    """``release_cache`` as given, or ``RunConfig.release_cache`` (``VERIF_RELEASE_CACHE``)
    when it is ``None``, so every entry point honours the setting without passing it."""
    return load_config().release_cache if release_cache is None else release_cache


def model_device(model: torch.nn.Module) -> torch.device:
    """The device ``model``'s parameters live on: the run's device (host for a model without
    parameters)."""
    p = next(model.parameters(), None)
    return torch.device("cpu") if p is None else p.device


def _on_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def release_device_cache(device: str | torch.device, section: Section | None = None) -> None:
    """Hand the CUDA caching allocator's free blocks back to the driver, inside the metrics
    section :data:`HANDOFF_SECTION`.

    Prover and verifier share one process and one GPU here. The prover's freed blocks stay in
    the allocator's cache, cut to the prover's sizes, and a large verifier buffer may then not
    fit in any of them though the GPU has the room (the first H100 session's out-of-memory
    errors reported GiBs "reserved but unallocated"). Called once per verified step, after the
    prover's step is released and before the verifier's role begins. Only when the run's
    ``device`` is CUDA: on any other device it does nothing and opens no section, even in a
    process where CUDA happens to be initialized."""
    if not _on_cuda(device):
        return
    with (section or no_section)(HANDOFF_SECTION):
        torch.cuda.empty_cache()


def _default_schedule(c: DeclaredComputation, dataset: Sequence[Any],
                      seed: int) -> Callable[[int], Sequence[int]]:
    return lambda t: data.schedule(t, c.n_s, len(dataset), seed)


def run_loop(
    c: DeclaredComputation,
    model: torch.nn.Module,
    dataset: Sequence[Any],
    w0: Mapping[str, torch.Tensor],
    verifier: Verifier,
    *,
    final: Mapping[str, torch.Tensor] | Sequence[bytes],
    fault: ProverFault | None = None,
    schedule: Callable[[int], Sequence[int]] | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    section: Section | None = None,
    handoff: StoreHandoff | None = None,
    release_cache: bool | None = None,
) -> LoopResult:
    """Run ``verifier.n_steps`` steps of ``C`` from ``W_0`` on ``dataset``.

    ``model`` is the prover's model; ``dataset`` is ``D``, which the prover reads batches and
    audit paths from and the verifier recomputes ``h_D`` over (check 1). ``final`` is the
    agreed final weights for check 8. ``schedule`` is the prover's ``π`` (default
    :func:`data.schedule` at the verifier's seed, the call the verifier makes); a prover that
    departs from the verifier's ``π`` is rejected at check 4.

    ``section`` is the B6 metrics seam (``runs/metrics.py``), passed on to ``prove_step``. The
    loop adds the prover phases ``P4.paths``, ``P3.commit`` (hashing) and ``P5.write`` (the
    hand-off to the store) per step and ``run:P4.tree`` once. The verifier's own seam is set on
    the ``Verifier``.

    ``handoff`` turns each committed step into the store the verifier reads (default
    :data:`IN_MEMORY`, references only). A ``DiskHandoff`` writes the step to disk in
    ``P5.write`` and deletes it once the step is verified (C3). The leaves and the tree are the
    same either way. Between the hand-off and the verifier, when ``model`` is on CUDA,
    :func:`release_device_cache` runs in its own section, :data:`HANDOFF_SECTION`.
    ``release_cache=False`` skips it, section and all, so a run can measure what the release
    costs and what it saves. ``None`` (the default) reads ``VERIF_RELEASE_CACHE``
    (:func:`release_cache_setting`).
    """
    sec = section or no_section
    handoff = handoff or IN_MEMORY
    fault = fault or HONEST
    release_cache = release_cache_setting(release_cache)
    assert_no_dropout(model)  # invariant 3 (S4d)
    device = model_device(model)
    pi = schedule or _default_schedule(c, dataset, verifier.seed)
    result = LoopResult(verdict=RunVerdict(False, None, 0, None))
    if verifier.start_run(dataset) is not None:
        result.verdict = verifier.end_run(final)
        return result
    with sec("run:P4.tree"):
        tree = dataset_tree(c, dataset)  # the prover's own copy of h_D's tree, for audit paths
    w: Mapping[str, torch.Tensor] = w0
    for t in range(1, verifier.n_steps + 1):
        idx = list(pi(t))
        records = list(fault.committed_records(t, [dataset[i] for i in idx]))
        w_t = fault.entry_weights(t, w)
        t0 = time.perf_counter()
        out = prove_step(c, model, w_t, records, train_records=fault.train_records(t, records),
                         perturb=fault.perturb(t), section=section)
        t1 = time.perf_counter()
        capture = out.capture
        out = fault.emit(t, out)  # outside prove_s: a splice is the harness's cost, not the prover's
        t1b = time.perf_counter()
        with sec("P4.paths"):
            paths = [tree.path(i) for i in idx]
        with sec("P3.commit"):
            leaves, tree_h = InMemoryStore.commit_step(c, out)
        with sec("P5.write"):
            store = handoff.hold(c, t, leaves, tree_h, paths)
        t2 = time.perf_counter()
        with sec(TOTALS_SECTION):
            n_leaves, n_bytes = leaf_totals(c, leaves)
        t2a = time.perf_counter()
        w, loss = out.w_next, out.loss
        del out, leaves, tree_h  # the store holds the leaves; nothing else keeps the step
        if release_cache:
            release_device_cache(device, section)  # the prover's step is gone; verifier next
        t2b = time.perf_counter()
        try:
            rej = verifier.verify_step(t, store)
        finally:
            handoff.release(store)
        t3 = time.perf_counter()
        del store  # discard before step t+1 (S3)
        rec = StepRecord(t, rej, loss, t1 - t0, t2 - t1b, t3 - t2b, t2b - t2a,
                         capture, n_bytes, n_leaves)
        result.steps.append(rec)
        if on_step is not None:
            on_step(rec)
        if rej is not None:
            break
    result.w_final = dict(w)
    result.verdict = verifier.end_run(final)
    return result


@dataclass(frozen=True)
class PlainStepRecord:
    """One step of :func:`run_plain`: its loss and wall clock (seconds).

    ``train_s`` is host ``perf_counter`` time around the whole ``plain_step``, informational
    only; P0's reported times are B6's ``P0.*`` rows."""

    t: int
    loss: float
    train_s: float


@dataclass
class PlainResult:
    steps: list[PlainStepRecord]
    w_final: dict[str, torch.Tensor]


def run_plain(
    c: DeclaredComputation,
    model: torch.nn.Module,
    dataset: Sequence[Any],
    w0: Mapping[str, torch.Tensor],
    *,
    n_steps: int,
    seed: int,
    schedule: Callable[[int], Sequence[int]] | None = None,
    on_step: Callable[[PlainStepRecord], None] | None = None,
    section: Section | None = None,
) -> PlainResult:
    """The honest run's training with the protocol off: ``n_steps`` ``plain_step``s from ``W_0``
    on ``π``'s batches.

    The same step as :func:`run_loop`'s prover (``plain_step`` is ``prove_step`` without the
    capture, bit for bit), the same default ``π`` (at ``seed``), and no tree, paths, commitment
    or verifier.
    It is the plain baseline (B7, EQ1b's P0, with ``section``), the agreed final weights for
    check 8, and the hidden-step building block.
    """
    assert_no_dropout(model)  # invariant 3 (S4d), as run_loop
    pi = schedule or _default_schedule(c, dataset, seed)
    w: Mapping[str, torch.Tensor] = w0
    steps: list[PlainStepRecord] = []
    for t in range(1, n_steps + 1):
        records = [dataset[i] for i in pi(t)]
        t0 = time.perf_counter()
        w, loss = plain_step(c, model, w, records, section=section)
        rec = PlainStepRecord(t, loss, time.perf_counter() - t0)
        steps.append(rec)
        if on_step is not None:
            on_step(rec)
    return PlainResult(steps, dict(w))
