"""The per-step run loop: prover → store → verifier → discard (S3, spec §3.6).

:func:`run_loop` drives one run of any :class:`DeclaredComputation`. For each step ``t`` it

1. has the prover compute ``C(b_t, W_t)`` on ``π(t)``'s batch (:func:`~.prover.prove_step`);
2. commits the step into an :class:`~.store.InMemoryStore`, with each record's audit path into
   ``h_D``;
3. hands the verifier the store, and only the store (invariant 1);
4. drops the step's transcript before the next step begins.

It stops at the first rejection, since a rejected run stays rejected, and then asks the
verifier for check 9's verdict, with check 8 against the agreed final weights.

Faults enter only on the prover side (S6a, invariant 5), through a :class:`ProverFault`. The
verifier is built by the caller from public inputs and never sees the fault.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from . import data
from .checks import Rejection
from .computation import DeclaredComputation
from .config import assert_no_dropout
from .prover import Perturbation, StepOutput, prove_step
from .store import InMemoryStore, dataset_tree
from .verifier import RunVerdict, Verifier

__all__ = ["ProverFault", "StepRecord", "LoopResult", "run_loop"]


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
    """One step of the loop: the verifier's decision and the prover's wall clock (seconds)."""

    t: int
    rejection: Rejection | None
    loss: float
    prove_s: float
    commit_s: float
    verify_s: float


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
) -> LoopResult:
    """Run ``verifier.n_steps`` steps of ``C`` from ``W_0`` on ``dataset``.

    ``model`` is the prover's model; ``dataset`` is ``D``, which the prover reads batches and
    audit paths from and the verifier recomputes ``h_D`` over (check 1). ``final`` is the
    agreed final weights for check 8. ``schedule`` is the prover's ``π`` (default
    :func:`data.schedule`); a prover that departs from the verifier's ``π`` is rejected at
    check 4.
    """
    fault = fault or HONEST
    assert_no_dropout(model)  # invariant 3 (S4d)
    pi = schedule or (lambda t: data.schedule(t, c.n_s, len(dataset)))
    result = LoopResult(verdict=RunVerdict(False, None, 0, None))
    if verifier.start_run(dataset) is not None:
        result.verdict = verifier.end_run(final)
        return result
    tree = dataset_tree(c, dataset)  # the prover's own copy of h_D's tree, for audit paths
    w: Mapping[str, torch.Tensor] = w0
    for t in range(1, verifier.n_steps + 1):
        idx = list(pi(t))
        records = list(fault.committed_records(t, [dataset[i] for i in idx]))
        w_t = fault.entry_weights(t, w)
        t0 = time.perf_counter()
        out = prove_step(c, model, w_t, records, train_records=fault.train_records(t, records),
                         perturb=fault.perturb(t))
        t1 = time.perf_counter()
        out = fault.emit(t, out)  # outside prove_s: a splice is the harness's cost, not the prover's
        t1b = time.perf_counter()
        store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
        t2 = time.perf_counter()
        w, loss = out.w_next, out.loss
        del out  # the store holds the leaves; nothing else keeps the step
        rej = verifier.verify_step(t, store)
        t3 = time.perf_counter()
        del store  # discard before step t+1 (S3)
        rec = StepRecord(t, rej, loss, t1 - t0, t2 - t1b, t3 - t2)
        result.steps.append(rec)
        if on_step is not None:
            on_step(rec)
        if rej is not None:
            break
    result.w_final = dict(w)
    result.verdict = verifier.end_run(final)
    return result
