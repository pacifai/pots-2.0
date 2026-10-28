"""The per-step run loop: prover → store → verifier → discard (S3, spec §3.6).

:func:`run_loop` drives one run of any :class:`DeclaredComputation`. For each step ``t`` it

1. has the prover compute ``C(b_t, W_t)`` on ``π(t)``'s batch
   (:func:`~verification.prover.step.prove_step`);
2. commits the step and hands it to a store through a
   :class:`~verification.transcript.store.StoreHandoff`, with each record's audit path into
   ``h_D``: an ``InMemoryStore`` by default, or a ``DiskStore`` (C3's cross-check);
3. when the model is on CUDA, hands the caching allocator's free blocks back to the driver
   (:func:`release_device_cache`, only where prover and verifier may share a card, or as
   ``VERIF_RELEASE_CACHE`` says: :func:`resolve_release_cache`), so the verifier allocates in
   memory the prover has left;
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
from typing import Any, Literal

import torch

from setup import data
from setup.config import assert_no_dropout, load_config
from verification.commitment import kernel_check
from verification.commitment.gpu_blake3 import GpuBlake3
from verification.commitment.leaves import HashStats, dataset_tree, leaf_totals
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
from verification.verifier.matmul_check import measure_kernel

__all__ = ["ProverFault", "StepRecord", "LoopResult", "run_loop", "PlainStepRecord", "PlainResult",
           "run_plain", "HANDOFF_SECTION", "TOTALS_SECTION", "release_device_cache",
           "release_cache_setting", "resolve_release_cache",
           "pipelined_setting", "model_device", "run_devices"]

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
    - :meth:`bind`: pipelined runs only (``runs/pipeline.py``). A fault that keeps the
      computation or a model builder drops them when pickled, and the prover's process hands it
      its own through :meth:`bind` before step 1.
    """

    def bind(self, c: DeclaredComputation, build_model: Callable[[], torch.nn.Module]) -> None:
        """Take the prover process's computation and model builder. Default: nothing."""

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
    root: bytes | None = None  # the step's transcript root (the Merkle root over its leaves)
    # leaves the commit took from the arrival digests (``HashStats.known`` of its hash call);
    # the loop always fills it, so it is ``None`` only on a record built without one
    known: int | None = None


@dataclass
class LoopResult:
    """The run's verdict, the steps that ran, and the prover's last committed ``W_{t+1}``.

    Per-check timings and statistics stay on the verifier (``timings``, ``run_timings``,
    ``stats``).
    """

    verdict: RunVerdict
    steps: list[StepRecord] = field(default_factory=list)
    w_final: dict[str, torch.Tensor] | None = None
    # pipelined runs: ``steps`` are the verified steps; ``w_final`` is the last verified step's
    # ``W_{t+1}``; ``child_info`` the prover process's counters (``sent``, ``max_unreleased``,
    # ``events``, ``pid``)
    pipelined: bool = False
    child_info: dict | None = None

    @property
    def rejection(self) -> Rejection | None:
        return self.verdict.rejection


def resolve_release_cache(release_cache: bool | None, prover_device: str | torch.device,
                          devices: Sequence[str | torch.device] | None
                          ) -> tuple[bool, Literal["argument", "env", "auto"]]:
    """Whether the hand-off releases the CUDA cache, and where the answer came from (FS53).

    Explicit argument, then ``VERIF_RELEASE_CACHE`` (``0`` or ``1``), then auto. Auto releases
    iff the run uses CUDA and prover and verifier may share a card: unsharded, or the prover's
    device is one of the verifier ``devices``. With the prover on a card of its own (the
    four-GPU layout) the release has nothing to fix (F4b) and costs the next step's
    ``cudaMalloc``s."""
    if release_cache is not None:
        return release_cache, "argument"
    env = load_config().release_cache
    if env is not None:
        return env, "env"
    from verification.runs.metrics import normalize_devices  # noqa: PLC0415 (import cycle)
    if not any(_on_cuda(d) for d in (prover_device, *(devices or ()))):
        return False, "auto"
    if not devices:  # None or (): unsharded, as metrics.gpu_roles reads it
        return True, "auto"
    return normalize_devices([prover_device])[0] in normalize_devices(devices), "auto"


def release_cache_setting(release_cache: bool | None, c: DeclaredComputation,
                          prover_device: str | torch.device) -> bool:
    """:func:`resolve_release_cache`'s answer for the run's computation ``c`` (its verifier
    devices) and the prover's device, so every entry point honours the setting without
    passing it."""
    return resolve_release_cache(release_cache, prover_device, c.placement.devices)[0]


def pipelined_setting(pipelined: bool | None) -> bool:
    """``pipelined`` as given, or ``RunConfig.pipelined`` (``VERIF_PIPELINED``) when ``None``."""
    return load_config().pipelined if pipelined is None else pipelined


def model_device(model: torch.nn.Module) -> torch.device:
    """The device ``model``'s parameters live on: the run's device (host for a model without
    parameters)."""
    p = next(model.parameters(), None)
    return torch.device("cpu") if p is None else p.device


def _on_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def release_device_cache(device: str | torch.device, section: Section | None = None, *,
                         devices: Sequence[str | torch.device] | None = None) -> None:
    """Hand the CUDA caching allocator's free blocks back to the driver, inside the metrics
    section :data:`HANDOFF_SECTION`.

    Where prover and verifier share a GPU (one H100, the pair layout: the only case auto
    releases), the prover's freed blocks stay in
    the allocator's cache, cut to the prover's sizes, and a large verifier buffer may then not
    fit in any of them though the GPU has the room (the first H100 session's out-of-memory
    errors reported GiBs "reserved but unallocated"). Called once per verified step, after the
    prover's step is released and before the verifier's role begins. Only when the run's
    ``device`` or any of ``devices`` (the sharded run's verifier devices, F22) is CUDA: with
    none, it does nothing and opens no section, even in a process where CUDA happens to be
    initialized. One section per step. One ``torch.cuda.empty_cache()`` call empties every
    device's cache, because ``NativeCachingAllocator::emptyCache`` loops over every device's
    allocator (torch 2.9.1)."""
    if not any(_on_cuda(d) for d in (device, *(devices or ()))):
        return
    with (section or no_section)(HANDOFF_SECTION):
        torch.cuda.empty_cache()


def run_devices(c: DeclaredComputation, *, pipelined: bool) -> tuple[torch.device, ...] | None:
    """The device list of a verified run's recorders (``MetricsWriter.recorder(devices=)``,
    ``memory_run(devices=)``), or ``None`` for an unsharded run (one device, rows as ever).

    A sharded run lists the verifier devices ``c.placement.devices``; a sequential one puts the
    prover's device ``c.placement.prover`` first, since it proves in the same process. The
    pipelined parent proves nothing: the prover's device belongs to the child's recorder.
    Normalized and without repeats (``metrics.normalize_devices``)."""
    from verification.runs.metrics import normalize_devices  # noqa: PLC0415 (import cycle)
    if not c.placement.sharded:
        return None
    devs = c.placement.devices or ()
    return normalize_devices(devs if pipelined else (c.placement.prover, *devs))


def _default_schedule(c: DeclaredComputation, dataset: Sequence[Any],
                      seed: int) -> Callable[[int], Sequence[int]]:
    return lambda t: data.schedule(t, c.n_s, len(dataset), seed)


def run_loop(
    c: DeclaredComputation,
    model: torch.nn.Module | None,
    dataset: Sequence[Any],
    w0: Mapping[str, torch.Tensor] | None,
    verifier: Verifier,
    *,
    final: Mapping[str, torch.Tensor] | Sequence[bytes],
    fault: ProverFault | None = None,
    schedule: Callable[[int], Sequence[int]] | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    section: Section | None = None,
    handoff: StoreHandoff | None = None,
    release_cache: bool | None = None,
    recorder: Any = None,
    pipelined: bool | None = None,
    prover: Callable[[], Any] | None = None,
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
    costs and what it saves. ``None`` (the default) reads ``VERIF_RELEASE_CACHE``, and unset
    means auto (:func:`release_cache_setting`).

    ``pipelined`` (``None`` reads ``VERIF_PIPELINED``, :func:`pipelined_setting`) runs the
    prover in a spawned child process, one step ahead of the verifier
    (``runs/pipeline.py``, which documents the mode). The parent then builds no prover model:
    ``model`` must be ``None``, ``w0`` is optional (the child's ``W_0`` is checked against the
    verifier's), ``prover`` is the :class:`~verification.runs.pipeline.ProverSpec` the child
    rebuilds from, and ``recorder`` is the run's ``TimeRecorder`` or ``MemoryRecorder`` (whose
    seam is ``section``). Without ``pipelined``, ``recorder`` and ``prover`` are ignored.

    The :class:`StepRecord` times, by mode. Sequential: ``prove_s`` is ``prove_step``;
    ``commit_s`` the paths, the commit and ``P5.write`` (the in-memory hold); ``verify_s``
    ``verify_step`` and the hand-off's release; ``handoff_s`` the cache release. Pipelined:
    ``prove_s`` and ``commit_s`` are the child's (``commit_s`` holds ``P5.write``, which copies
    a CPU transcript into shared memory); ``verify_s`` is the parent's ``verify_step`` alone;
    ``handoff_s`` the parent's ``SharedStore.open``, ``close`` and cache release. The child's
    release and the wait for its answer are in none of them.
    """
    kernel_check.new_run()  # first: a new run checks every kernel shape afresh (FS34)
    if pipelined_setting(pipelined):
        if model is not None:
            raise ValueError("pipelined: the prover's model is built in the prover's process; "
                             "pass model=None and a ProverSpec (prover=...)")
        from verification.runs.pipeline import run_pipelined  # pipeline imports this module
        return run_pipelined(c, dataset, w0, verifier, final=final, fault=fault,
                             schedule=schedule, on_step=on_step, section=section,
                             handoff=handoff,
                             release_cache=release_cache_setting(release_cache, c,
                                                                 c.placement.prover),
                             recorder=recorder, prover=prover)
    if model is None or w0 is None:
        raise ValueError("run_loop needs a model and w0 unless it is pipelined")
    sec = section or no_section
    handoff = handoff or IN_MEMORY
    fault = fault or HONEST
    assert_no_dropout(model)  # invariant 3 (S4d)
    device = model_device(model)
    release_cache = release_cache_setting(release_cache, c, device)  # FS53
    pi = schedule or _default_schedule(c, dataset, verifier.seed)
    result = LoopResult(verdict=RunVerdict(False, None, 0, None))
    # The first NVRTC build or cache load of the process would land inside a timed section:
    # step 1's forward pass, or check 0's hash of W_0 in ``start_run`` (the kernel hashes it
    # where it sits). Do it here, before both, outside every section (FS30, FS34). Every CUDA
    # device of the run, the sharded placement's included.
    for d in dict.fromkeys(torch.device(x) for x in (device, *(c.placement.devices or ()))
                           if _on_cuda(x)):
        GpuBlake3().kernels(d)
        measure_kernel.kernels(d, verifier.k)  # check 5's fused measuring kernel (FS37)
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
            commit_stats = HashStats()
            leaves, tree_h = InMemoryStore.commit_step(c, out, section=section,
                                                       stats=commit_stats)
        with sec("P5.write"):
            store = handoff.hold(c, t, leaves, tree_h, paths)
        t2 = time.perf_counter()
        with sec(TOTALS_SECTION):
            n_leaves, n_bytes = leaf_totals(c, leaves)
        t2a = time.perf_counter()
        w, loss = out.w_next, out.loss
        root = tree_h.root
        del out, leaves, tree_h  # the store holds the leaves; nothing else keeps the step
        if release_cache:
            # the prover's step is gone; the verifier is next
            release_device_cache(device, section, devices=c.placement.devices)
        t2b = time.perf_counter()
        try:
            rej = verifier.verify_step(t, store)
        finally:
            handoff.release(store)
        t3 = time.perf_counter()
        del store  # discard before step t+1 (S3)
        rec = StepRecord(t, rej, loss, t1 - t0, t2 - t1b, t3 - t2b, t2b - t2a,
                         capture, n_bytes, n_leaves, root, commit_stats.known)
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
