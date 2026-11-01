"""The two-process run loop: the prover in a spawned child, the verifier in the parent (FS33b).

``run_loop(..., pipelined=True)`` calls :func:`run_pipelined`. The child trains, commits and
shares step ``t + 1`` while the parent verifies step ``t``, the way a deployed prover and
verifier run on two machines (F22, ``ENVIRONMENT_PLAN_DRAFT`` "Sequential or pipelined"). The
sequential loop (``loop.py``) keeps its order; this module is the other mode. FS33b changed
``run_loop`` only to load the hash kernels on every CUDA verifier device before step 1, outside
every section; fix 1 added ``loop.run_devices``.

**What the child rebuilds from.** The parent builds no prover model. The child builds its own
from a :class:`ProverSpec`, a picklable zero-argument callable that returns a
:class:`ProverSide` (the declared computation, its model builder, the dataset and ``W_0``). The
spec also does the child's determinism setup (``setup_determinism``, the same call the parent's
entry point makes) before it builds anything: a ``spawn`` child starts from a fresh interpreter.
The fault and the schedule cross by pickling (``None``: the child computes the default ``π`` at
``verifier.seed``). A :class:`~verification.runs.loop.ProverFault` that needs the computation
or a model builder drops them when pickled and gets the child's through its ``bind`` hook,
called before step 1. An unpicklable spec, fault or schedule raises ``TypeError`` naming which
one, before the child starts.

**The handshake.** Before step 1 the child sends the leaf hashes of its ``W_0`` (as
``Verifier.weight_hashes`` computes them), a fingerprint of its computation and the root and
length of its dataset. The fingerprint holds what the prover's work depends on and the verifier
also declares: ``n_leaves``, ``M``, ``n_s``, ``eta``, the weight names and shapes, the four
dtypes, the placement's devices, every product spec (a BLAKE3 digest of their text, and the
list itself when it is short, so a mismatch can name the first differing product) and the
instance's ``pad_id`` and ``source`` where it has them. The parent compares all of it with its
own ``c``, the dataset with ``verifier.h_D`` and ``verifier.n_records``, and ``W_0`` with
``verifier.chain_hashes`` (the agreed ``W_0``'s hashes after ``start_run``). Any difference
raises :class:`ProverProcessError` (``kind="handshake"``) naming each mismatch, before step 1
and never with a verdict. A child on another dataset is a setup error, not a cheat to be
caught by the checks: the run would end in a rejection that blames the prover for the harness.

**One step, both sides.**

- Child, step ``t``: ``prove_step``, the commit, ``leaf_totals``, ``share_step`` inside
  ``P5.write``, the recorder's ``close_step(t)``, then one message holding the package, the
  step's metrics part and the :class:`~verification.runs.loop.StepRecord` fields the child knows.
- Parent: receive, ``SharedStore.open``, ``verify_step``, ``close``, tell the child to release
  ``t``, wait for the release result, ``merge_step(t, part)``, build the ``StepRecord``,
  ``on_step``.

The ``StepRecord`` fields, by mode. ``prove_s``: the child's ``prove_step`` (the same in both).
``commit_s``: the commit and the hand-off to the store; pipelined it is the child's and holds
``P5.write`` (``share_step``, which copies a CPU transcript once), sequentially ``P5.write`` is
the in-memory hold. ``verify_s``: ``verify_step``; sequentially it also holds the hand-off's
release, pipelined it is the parent's verify only. ``handoff_s``: sequentially dropping the
prover's references and the cache release; pipelined the parent's ``SharedStore.open``, its
``close`` and its cache release (the child's release and cache release are in the child's own
sections, ``handoff.release``). The harness section ``handoff.totals`` is in none of the four.

**The stored step.** With a ``KeptStepHandoff`` naming step ``t`` (``run_verified``'s stored
step, EV2), the parent writes step ``t`` to disk from the opened store's leaves and the
package's ``tree`` and ``h_D`` paths, after ``verify_step(t)`` and before ``close`` (staged: from
the parent's own copy, after the early release), and checks
the files' root as ``KeptStepHandoff.write`` does. The write is in no section and in none of
the step's four clocks (``handoff_s`` excludes it), as in the sequential loop. Its cost is real
all the same: unstaged it delays step ``t``'s release, so the child's step ``t + 2`` starts
later by the write's time (staged, the release came first and the write delays only the
parent's next step). A step the run never reaches is never written. Any other hand-off is refused.

**Two transcripts at most.** The child starts step ``t + 2`` only after the parent has released
step ``t``: it blocks, before each step, while two steps are unreleased (one, staged). The release order is
the parent's ``close``, then the child's ``StepHold.release``. A failed release
(``StoreMutationError``: the prover wrote into a leaf while it was shared) is a run error: no
verdict and no ``on_step`` for that step.

**Staged mode (FS58b).** With one verifier GPU that is not the prover's (``staging=None`` picks
it on exactly that layout, :func:`staging_setting`; ``True`` and ``False`` force it), the
capture would route step ``t + 1``'s products to the GPU that still holds step ``t``. Staged,
the child proves with ``staging=<its device>`` (FS58a): every tensor leaf stays on the prover's
side, and the child builds no kernel and creates no context on a verifier GPU. The parent
copies each opened step to its placement devices in the harness section ``handoff.ship``
(``SHIP_SECTION``, side ``run``, in no total), synchronizes the source GPU, closes the shared
store, releases the step and only then verifies its own copy. The window is one step: the child
starts step ``t + 1`` only when no step is held, so it proves ``t + 1`` while the parent
verifies ``t`` but never holds two transcripts. The copy and the early release are in
``handoff_s``, not ``verify_s``; the kept step is written from the parent's copy. The decision
is in ``child_info["staging"]``, and ``child_info["verifier_reserved"]`` is the child's
allocator reserve on each verifier GPU at the end, and ``child_info["verifier_context"]`` whether
it holds a primary context there (CUDA builds that expose the query; else ``None``). The child's
recorder is built on the prover's device only, W_0 and any fault's own weights are put on the
prover's device, so both read 0 and ``False``: the child never touched the verifier GPU. Its
sections' device lists then hold the prover's device only (``gpu_held_s``, ``gpu_us_by_device``
and the memory lists of the child's rows count one GPU, not two).
The parent's cache is not released before the copy, on purpose: in this layout the cache
release is off (auto), and step ``t``'s copy reuses the blocks step ``t - 1``'s verification
freed in the same allocator, which is the point of the parent making the copy. A refused
release (``StoreMutationError``) or a reply for the wrong step now raises before step ``t`` is
verified, so that step has no ``StepRecord``.

**The child reads its pipe only between steps**, so a release message waits until the step in
progress ends. That costs no throughput: the parent cannot verify step ``t + 1`` before it has
package ``t + 1``, which the child sends when that step ends, and the child answers the release
of ``t`` right after, before it starts step ``t + 2``. The parent's wait for the release result
is the child's post-step bookkeeping, not a training step. The release (and its CUDA cache
release) is timed in the child's table for the step in progress then, so the last two steps'
release rows are never merged (staged, the last step's only).

**Stopping.** After the last step, or a verified rejection at ``t`` (after its release
succeeded and ``on_step(t)`` ran), the parent sends ``stop``. The child discards later steps and
answers with ``W_{t+1}`` as host copies (``LoopResult.w_final``) and its counters (``sent``,
``max_unreleased``, the ordered ``events`` of step starts and releases). ``LoopResult.steps``
holds the verified steps only.

**The child never exits before the parent has dropped every package.** CUDA storages cross as
IPC handles that the parent opens when it unpickles a package, and the producer's memory must
stay alive until then. The pipe is ordered, so when the parent has received the child's final
message it has received every package sent before it. It drops them (draining packages of
steps it will not verify), then sends ``bye``. The child exits only on ``bye`` (or on the pipe
closing), and releases the steps it discarded only after that. After an error the child also
waits for ``bye``.

**Failure.** An exception in the child crosses the pipe as text with its traceback and is raised
as :class:`ProverProcessError` (``kind="exception"``). A child that dies is seen by the pipe
closing or by its exit code (``kind="died"``). Every parent wait polls with a short timeout and
checks that the child lives. Any exception in the parent, ``KeyboardInterrupt`` and a raising
``on_step`` included, closes the open store, drops every package it still holds (a traceback
keeps the frame's locals alive, so the ``finally`` clears them and collects), stops the child (a
stop message, then ``terminate``, then ``kill`` after a grace period) and joins it before it
re-raises: no child outlives ``run_loop``. Every wait for a message has a wall deadline,
``MESSAGE_DEADLINE_S`` (it covers the child's model load before ``ready`` and the slowest step
we run); a child that hangs raises :class:`ProverProcessError` (``kind="timeout"``) naming the
message awaited and the seconds waited; a child that timed out is stopped with no grace, since a
stuck child cannot read ``stop``. A ``start_run`` rejection stops the child with no grace:
no package has crossed, so terminating it while it builds its model is safe.

**Refusals** (``ValueError``, before the child starts): a CUDA prover device with an unsharded
placement (an unsharded run's leaves live on the host, so every step's whole transcript would be
copied through one shared-memory buffer); a ``handoff`` that is neither ``IN_MEMORY`` nor a
``KeptStepHandoff``; a recorder that cannot split a step (the counting recorder, a traced
``TimeRecorder``); a missing
:class:`ProverSpec`; a model given to the parent; and :class:`~verification.runs.scenarios.
KeepStep`, which keeps a step's output in the child's memory.

**Recorders.** The run's recorder stays in the parent, and the ``section`` seam is the
verifier's. The child builds a recorder of the same kind from a :class:`RecorderSpec` (no
writer): ``close_step`` per step, and the parent's ``merge_step``. Step 0's part (``run:P4.tree``)
is merged right after the handshake. The child's own hand-off work is named under ``handoff.``
(``handoff.release``), which never collides with the parent's ``handoff.empty_cache``.

**CUDA caches.** With ``release_cache`` on (unset means auto: off when the prover's device is
not a verifier device, FS53), the parent empties its allocator cache after each
step's ``close`` (the verifier's blocks are cut to the verifier's sizes) and the child after each
``release`` (the prover's freed transcript blocks are cut to the prover's sizes), each in its own
process, so the next step's allocations find the memory the other role left.

**Kernels.** Before step 1, outside every section, as the sequential loop does: the child
loads the hash kernels of its prover device and of every CUDA verifier device, and the parent
loads the hash kernel and check 5's measuring kernel on every CUDA verifier device. The child
never measures, so it builds no measuring kernel.
"""

from __future__ import annotations

import dataclasses
import gc
import multiprocessing
import os
import pickle
import signal
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any

import torch
from blake3 import blake3

from setup.config import assert_no_dropout
from verification.commitment import kernel_check
from verification.commitment.gpu_blake3 import GpuBlake3
from verification.commitment.leaves import (
    HashStats,
    dataset_tree,
    leaf_hashes_of,
    leaf_totals,
)
from verification.computation.interface import DeclaredComputation
from verification.prover.step import Section, no_section, prove_step
from verification.runs.kept_step import KeptStepHandoff
from verification.runs.loop import (
    HANDOFF_SECTION,
    TOTALS_SECTION,
    LoopResult,
    ProverFault,
    StepRecord,
    _default_schedule,
    model_device,
    release_device_cache,
)
from verification.runs.metrics import (
    MemoryRecorder,
    TimeRecorder,
    memory_probe,
    normalize_devices,
)
from verification.runs.scenarios import KeepStep, ReusedModel
from verification.transcript.errors import StoreMutationError
from verification.transcript.shared import SharedStore, share_step
from verification.transcript.store import IN_MEMORY, InMemoryStore, StoreHandoff
from verification.verifier.driver import RunVerdict, Verifier
from verification.verifier.matmul_check import measure_kernel

__all__ = ["ProverSide", "ProverSpec", "ProverProcessError", "RecorderSpec", "run_pipelined",
           "RELEASE_SECTION", "SHIP_SECTION", "staging_setting"]

# The child's own hand-off section: ``hold.release()`` and, with the cache release on, the
# CUDA cache release. Not the parent's HANDOFF_SECTION, so no merged step has two of one name.
RELEASE_SECTION = "handoff.release"
# The parent's copy of a step from the shared store to its verifier GPUs, staged mode only
# (FS58b). Harness work (side ``run``, in no total), named like ``handoff.release``.
SHIP_SECTION = "handoff.ship"

STOP_GRACE_S = 20.0  # how long a stopping parent lets the child finish its step
TERMINATE_GRACE_S = 5.0  # after terminate(), before kill()
# The longest the parent waits for one message: the child's model load before ``ready`` or its
# slowest step (about 5 minutes on the Mac at bf16), with room to spare.
MESSAGE_DEADLINE_S = 3600.0
_POLL_S = 0.05
_MAX_UNRELEASED = 2


@dataclass(frozen=True)
class ProverSide:
    """What the prover's process runs on: the computation, a model builder, ``D`` and ``W_0``."""

    c: DeclaredComputation
    build_model: Callable[[], torch.nn.Module]
    dataset: Sequence[Any]
    w0: Mapping[str, torch.Tensor]


ProverSpec = Callable[[], ProverSide]
"""A picklable (a module-level function, or an instance of a module-level class) callable
with no arguments. It runs in the child, does the child's determinism setup first, and returns
the child's :class:`ProverSide`."""


class ProverProcessError(RuntimeError):
    """The pipelined run failed on the prover's side or in the link. Never a verdict.

    ``kind``: ``"exception"`` (the child raised; the message holds its traceback), ``"died"``
    (the child exited or the pipe closed), ``"handshake"`` (the child's setup differs from the
    verifier's), ``"protocol"`` (an unexpected message) or ``"timeout"`` (no message within
    ``MESSAGE_DEADLINE_S``)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class RecorderSpec:
    """What the child builds its recorder from: ``kind`` (``"time"`` or ``"memory"``), the run
    and scenario names, and the devices (the prover's first, then the verifier devices; empty
    on the host, where the recorder is built as the sequential one would be)."""

    kind: str
    run: str
    scenario: str
    devices: tuple[str, ...] = ()

    def build(self) -> TimeRecorder | MemoryRecorder:
        devs = self.devices or None
        if self.kind == "time":
            return TimeRecorder(self.run, self.scenario, devices=devs)
        if self.kind == "memory":
            return MemoryRecorder(self.run, self.scenario, memory_probe(devs[0] if devs else "cpu"),
                                  devices=devs)
        raise ValueError(f"unknown recorder kind {self.kind!r}")


@dataclass(frozen=True)
class _ChildArgs:
    spec: bytes
    fault: bytes
    schedule: bytes
    recorder: RecorderSpec | None
    release_cache: bool
    seed: int
    n_steps: int
    staging: bool = False


def _is_cuda(d: str | torch.device) -> bool:
    return torch.device(d).type == "cuda"


_LISTED_PRODUCTS = 512  # up to this many product specs travel as a list beside their digest


def _fingerprint(c: DeclaredComputation) -> dict[str, Any]:
    """What the prover's work depends on that the verifier also declares (harness code: the
    instance's own ``pad_id`` and ``source`` are read where it has them)."""
    p = c.placement
    products = [(s.m, s.name, str(s.kind), tuple(s.a_shape), tuple(s.b_shape), s.weight, s.layer,
                 None if s.member is None else tuple(s.member), s.bias) for s in c.products]
    digest = blake3(repr(products).encode()).hexdigest()
    return {"n_leaves": c.n_leaves, "M": c.M, "n_s": c.n_s, "eta": repr(c.eta),
            "weight_names": tuple(c.weight_names),
            "weight_shapes": tuple((n, tuple(c.weight_shapes[n])) for n in c.weight_names),
            "weight_dtype": str(c.weight_dtype), "product_dtype": str(c.product_dtype),
            "operand_dtype": str(c.operand_dtype),
            "accumulator_dtype": str(c.accumulator_dtype),
            "prover": str(p.prover),
            "devices": None if p.devices is None else tuple(str(d) for d in p.devices),
            "products_digest": digest,
            "products": products if len(products) <= _LISTED_PRODUCTS else None,
            "pad_id": getattr(c, "pad_id", None), "source": getattr(c, "source", None)}


def _w_hashes(c: DeclaredComputation, w: Mapping[str, torch.Tensor]) -> tuple[bytes, ...]:
    """``Verifier.weight_hashes``'s digests of ``w`` (the same formula, without a verifier)."""
    if set(w) != set(c.weight_names):
        raise ValueError("weights differ from the declared weight names")
    return tuple(leaf_hashes_of(c, ((c.w_t_index(n), w[n].detach()) for n in c.weight_names)))


def staging_setting(c: DeclaredComputation, staging: bool | None) -> bool:
    """``staging`` as given, or (``None``) on exactly when the one-GPU verifier needs it (FS58b).

    Auto: the placement is sharded, the prover's device is CUDA, there is exactly one verifier
    device, it is CUDA and it is not the prover's. Then step ``t + 1``'s products would be
    routed to the GPU that still holds step ``t``; staged, they stay on the prover's side and
    the parent copies each step to its verifier GPU. A bare ``cuda`` means the current
    device's index."""
    if staging is not None:
        return staging
    p = c.placement
    if not p.sharded or not _is_cuda(p.prover) or len(p.devices or ()) != 1:
        return False
    (dev,) = p.devices
    return _is_cuda(dev) and normalize_devices((dev,)) != normalize_devices((p.prover,))


# ---- the child ---------------------------------------------------------------------------


def _child_main(conn: Connection, args: _ChildArgs) -> None:
    """The spawned process's entry point."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when the child stops
    try:
        _child_run(conn, args)
    except (EOFError, BrokenPipeError, ConnectionError):
        pass  # the parent is gone
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
            _await_bye(conn)
        except (OSError, EOFError):
            pass
        raise SystemExit(1) from None


def _await_bye(conn: Connection) -> None:
    """Wait for the parent's ``bye``, with no deadline: the child owns the memory the parent
    reads through IPC handles, so it exits only on ``bye`` or when the pipe closes (EOF or
    ``OSError``). The parent's ``link.stop()`` ends a child that outlives the run."""
    while True:
        try:
            if not conn.poll(_POLL_S):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "bye":
            return


def _child_run(conn: Connection, args: _ChildArgs) -> None:
    # These bytes are the parent's own ``pickle.dumps`` of its own objects, sent to its own
    # child over a private pipe: not untrusted input.
    kernel_check.new_run()  # the child's own set of checked shapes (FS34)
    spec = pickle.loads(args.spec)  # noqa: S301
    fault: ProverFault = pickle.loads(args.fault)  # noqa: S301
    schedule = pickle.loads(args.schedule)  # noqa: S301
    side = spec()
    c, dataset = side.c, side.dataset
    w0 = side.w0
    rec = None if args.recorder is None else args.recorder.build()
    section: Section | None = None if rec is None else rec.seam()
    sec = section or no_section
    model = side.build_model()
    device = model_device(model)
    staged_cuda = args.staging and _is_cuda(device)
    if staged_cuda:  # FS58b: a fallback that frees W_0's tensors on a verifier GPU. A spec must
        # build W_0 on the prover's device for that GPU to stay untouched (context, cached
        # reserve); ``child_info["verifier_context"]`` reports whether it did.
        fault.staging_device = device
        w0 = {n: v if v.device == device else v.to(device) for n, v in w0.items()}
        side = dataclasses.replace(side, w0=w0)
    if type(fault).bind is not ProverFault.bind:
        fault.bind(c, ReusedModel(side.build_model))
    assert_no_dropout(model)
    cache = args.release_cache and any(_is_cuda(d) for d in (device, *(c.placement.devices or ())))
    # Staged (FS58b): the child never creates a context on a verifier GPU.
    kernel_devs = [torch.device(d) for d in
                   (device, *(() if args.staging else (c.placement.devices or ())))
                   if _is_cuda(d)]
    for d in dict.fromkeys(kernel_devs):
        GpuBlake3().kernels(d)  # outside every section (FS30)
    pi = schedule or _default_schedule(c, dataset, args.seed)
    with sec("run:P4.tree"):
        tree = dataset_tree(c, dataset)
    part0 = None if rec is None else rec.close_step(0)
    handshake = {"w0": _w_hashes(c, w0), "dataset_root": tree.root,
                 "dataset_len": len(dataset), **_fingerprint(c)}
    conn.send(("ready", handshake, part0))

    holds: dict[int, Any] = {}
    w_by: dict[int, Mapping[str, torch.Tensor]] = {}
    events: list[tuple[str, int]] = []
    stats = {"sent": 0, "max_unreleased": 0}
    stop_at: list[int] = []  # the parent's stop message, once it comes

    def release(t: int) -> str | None:
        hold = holds.pop(t, None)
        err = None
        with sec(RELEASE_SECTION):
            try:
                if hold is not None:
                    hold.release()
            except StoreMutationError as e:
                err = str(e)
            del hold
            if cache:
                torch.cuda.empty_cache()
        events.append(("release", t))
        for k in [k for k in w_by if k < t]:
            del w_by[k]
        return err

    def handle(msg: tuple) -> None:
        if msg[0] == "release":
            conn.send(("released", msg[1], release(msg[1])))
        elif msg[0] == "stop":
            stop_at.append(msg[1])
        elif msg[0] == "bye":
            raise EOFError("the parent said bye before stop")
        else:
            raise RuntimeError(f"the prover's process got an unknown message {msg[0]!r}")

    w: Mapping[str, torch.Tensor] = w0
    if args.staging:  # W_0 stays in ``w`` only: step 1's end frees it (1.8 GiB for Qwen)
        side = w0 = None  # type: ignore[assignment]
    t = 0
    while not stop_at:
        while conn.poll(0):
            handle(conn.recv())  # between steps only
        if stop_at:
            break
        if t >= args.n_steps or len(holds) >= (1 if args.staging else _MAX_UNRELEASED):
            # nothing to do (all steps sent) or the window is full (two steps unreleased, one
            # staged: the parent copies a step and releases it before the next starts)
            handle(conn.recv())
            continue
        t += 1
        events.append(("start", t))
        idx = list(pi(t))
        records = list(fault.committed_records(t, [dataset[i] for i in idx]))
        w_t = fault.entry_weights(t, w)
        t0 = time.perf_counter()
        out = prove_step(c, model, w_t, records, train_records=fault.train_records(t, records),
                         perturb=fault.perturb(t), section=section,
                         staging=device if args.staging and _is_cuda(device) else None)
        t1 = time.perf_counter()
        capture = out.capture
        out = fault.emit(t, out)
        t1b = time.perf_counter()
        with sec("P4.paths"):
            paths = [tree.path(i) for i in idx]
        with sec("P3.commit"):
            commit_stats = HashStats()
            leaves, tree_h = InMemoryStore.commit_step(c, out, section=section,
                                                       stats=commit_stats)
        with sec("P5.write"):
            package, hold = share_step(c, t, leaves, tree_h, paths)
        t2 = time.perf_counter()
        with sec(TOTALS_SECTION):
            n_leaves, n_bytes = leaf_totals(c, leaves)
        w, loss = out.w_next, out.loss
        w_by[t] = w
        root, known = tree_h.root, commit_stats.known
        holds[t] = hold
        stats["max_unreleased"] = max(stats["max_unreleased"], len(holds))
        del out, leaves, tree_h, hold, w_t, records
        part = None if rec is None else rec.close_step(t)
        fields = {"loss": loss, "prove_s": t1 - t0, "commit_s": t2 - t1b, "capture": capture,
                  "transcript_bytes": n_bytes, "leaves": n_leaves, "root": root, "known": known}
        conn.send(("step", t, package, part, fields))
        stats["sent"] = t
        del package, part

    last = stop_at[0]
    w_host = None
    if last >= 1:
        w_host = {n: v.detach().to("cpu", copy=True) for n, v in w_by[last].items()}
    info = {**stats, "events": events, "pid": os.getpid()}
    if args.staging:  # the allocator's counter: it creates no context on a device never used
        info["verifier_reserved"] = {
            str(d): torch.cuda.memory_reserved(d)
            for d in dict.fromkeys(torch.device(x) for x in (c.placement.devices or ())
                                   if _is_cuda(x))}
        has_ctx = getattr(torch._C, "_cuda_hasPrimaryContext", None)  # CUDA builds only
        info["verifier_context"] = {
            str(d): None if has_ctx is None else bool(has_ctx(d.index))
            for d in dict.fromkeys(normalize_devices(
                [x for x in (c.placement.devices or ()) if _is_cuda(x)]))}
    conn.send(("final", w_host, info))
    del w_host
    _await_bye(conn)
    for k in list(holds):  # the steps discarded: the parent has dropped their packages
        try:
            holds.pop(k).release()
        except StoreMutationError:
            pass


# ---- the parent --------------------------------------------------------------------------


class _Link:
    """The parent's end of the pipe and the child's process handle."""

    def __init__(self, proc: Any, conn: Connection) -> None:
        self.proc, self.conn = proc, conn
        self.errored = False  # the child sent its error: it only waits for bye
        self.finished = False  # the child sent its final message: it only waits for bye
        self.closed = False
        self.timed_out = False  # no message in time: the child is stuck in a step

    def send(self, msg: tuple) -> None:
        try:
            self.conn.send(msg)
        except (OSError, ValueError) as e:
            raise ProverProcessError("died", f"the prover's process is gone (exit code "
                                             f"{self.proc.exitcode}): {e}") from None

    def recv(self, what: str | None = None) -> tuple:
        """The next message, polling with a short timeout and checking the child lives. Past
        ``MESSAGE_DEADLINE_S`` (read now) it raises ``timeout`` naming ``what`` was awaited."""
        start = time.monotonic()
        while True:
            waited = time.monotonic() - start
            if waited > MESSAGE_DEADLINE_S:
                self.timed_out = True
                raise ProverProcessError(
                    "timeout", f"no {what or 'message'} from the prover's process after "
                               f"{waited:.0f} s (MESSAGE_DEADLINE_S {MESSAGE_DEADLINE_S:.0f})")
            ready = wait([self.conn, self.proc.sentinel], _POLL_S)
            if self.conn in ready or self.conn.poll(0):
                try:
                    msg = self.conn.recv()
                except (EOFError, OSError):
                    raise ProverProcessError("died", self._dead()) from None
                if msg[0] == "error":
                    self.errored = True
                    raise ProverProcessError("exception",
                                             "the prover's process raised:\n" + msg[1])
                return msg
            if not self.proc.is_alive():
                raise ProverProcessError("died", self._dead())

    def _dead(self) -> str:
        self.proc.join(timeout=2.0)
        return f"the prover's process died (exit code {self.proc.exitcode})"

    def expect(self, kind: str, stash: dict[int, tuple] | None = None) -> tuple:
        """The next ``kind`` message; a ``step`` that arrives early goes in ``stash``."""
        while True:
            msg = self.recv(kind)
            if msg[0] == kind:
                return msg
            if msg[0] == "step" and stash is not None:
                stash[msg[1]] = msg
                continue
            raise ProverProcessError("protocol", f"expected {kind!r}, got {msg[0]!r}")

    def stop(self, grace: float | None = None) -> None:
        """End the child: stop, then bye after its final message, else terminate, then kill.
        Joins it. Never raises. ``grace`` defaults to ``STOP_GRACE_S``, read now."""
        if self.closed:
            return
        grace = STOP_GRACE_S if grace is None else grace
        if self.timed_out:
            grace = 0.0  # a stuck child can't read ``stop``; waiting for it only delays the kill
        self.closed = True
        proc, conn = self.proc, self.conn
        try:
            done = self.errored or self.finished
            if not done and proc.is_alive():
                try:
                    conn.send(("stop", 0))
                except (OSError, ValueError):
                    pass
                end = time.monotonic() + grace
                while time.monotonic() < end and not done:
                    ready = wait([conn, proc.sentinel], _POLL_S)
                    if conn in ready:
                        try:
                            msg = conn.recv()
                        except (EOFError, OSError):
                            break
                        done = msg[0] in ("final", "error")
                        del msg
                    elif proc.sentinel in ready:
                        break
            if proc.is_alive():
                try:
                    conn.send(("bye",))
                except (OSError, ValueError):
                    pass
            end = time.monotonic() + (10.0 if done else 0.0)
            while proc.is_alive() and time.monotonic() < end:
                time.sleep(_POLL_S)
            if proc.is_alive():
                proc.terminate()
                end = time.monotonic() + TERMINATE_GRACE_S
                while proc.is_alive() and time.monotonic() < end:
                    time.sleep(_POLL_S)
            if proc.is_alive():
                proc.kill()
                while proc.is_alive():
                    time.sleep(_POLL_S)
            proc.join(timeout=5.0)
        finally:
            try:
                conn.close()
            except OSError:
                pass
            try:
                proc.close()
            except ValueError:
                pass


def _pickled(what: str, x: Any) -> bytes:
    try:
        return pickle.dumps(x)
    except Exception as e:  # PicklingError, AttributeError (a lambda), TypeError, ...
        raise TypeError(f"the {what} cannot be pickled, so it cannot cross to the prover's "
                        f"process: {type(e).__name__}: {e}") from e


def _recorder_spec(recorder: TimeRecorder | MemoryRecorder | None, c: DeclaredComputation,
                   staged: bool = False) -> RecorderSpec | None:
    if recorder is None:
        return None
    if type(recorder) is TimeRecorder:
        if recorder._trace is not None:  # noqa: SLF001
            raise ValueError("pipelined: a traced TimeRecorder (trace_step) cannot split a "
                             "step; the profiler is process-global")
        kind = "time"
    elif type(recorder) is MemoryRecorder:
        kind = "memory"
    else:
        raise ValueError(f"pipelined: a {type(recorder).__name__} cannot split a step; only "
                         f"TimeRecorder and MemoryRecorder run pipelined")
    p = c.placement
    # Staged (FS58b): the child's recorder knows the prover's device only, so it never
    # synchronizes or probes a verifier GPU; its sections' own device lists carry that.
    devs = tuple(str(d) for d in normalize_devices(
        (p.prover,) if staged else (p.prover, *(p.devices or ()))))
    on_cuda = any(_is_cuda(d) for d in devs)
    return RecorderSpec(kind, recorder.run, recorder.scenario, devs if on_cuda else ())


def _differing_product(mine: Sequence[Any], theirs: Sequence[Any]) -> str:
    for i, (a, b) in enumerate(zip(mine, theirs, strict=False), start=1):
        if a != b:
            return f"the first differing product is m={i} ({a[1]!r} here, {b[1]!r} there)"
    return f"the lists have {len(mine)} and {len(theirs)} products"


def _check_handshake(hs: dict[str, Any], c: DeclaredComputation, verifier: Verifier,
                     w0: Mapping[str, torch.Tensor] | None) -> None:
    bad = []
    mine_all = _fingerprint(c)
    listed = mine_all.pop("products")
    for k, mine in mine_all.items():
        if hs.get(k) != mine:
            bad.append(f"{k}: the prover has {hs.get(k)!r}, the verifier {mine!r}")
            if k == "products_digest" and listed is not None and hs.get("products") is not None:
                bad.append(_differing_product(listed, hs["products"]))
    for k, mine in (("dataset_root", verifier.h_D), ("dataset_len", verifier.n_records)):
        if hs.get(k) != mine:
            bad.append(f"{k}: the prover has {hs.get(k)!r}, the verifier {mine!r}")
    want = tuple(verifier.chain_hashes or ())  # the agreed W_0's hashes, set by start_run
    if w0 is not None:
        own = verifier.weight_hashes(w0)
        if own != want:
            bad.append("the W_0 given to run_loop differs from the verifier's agreed W_0")
    got = tuple(hs["w0"])
    if got != want:
        names = [n for n, a, b in zip(c.weight_names, got, want, strict=False) if a != b]
        bad.append(f"the prover's W_0 differs from the agreed W_0 at "
                   f"{names[:3] or 'its length'}")
    if bad:
        raise ProverProcessError("handshake", "the prover's setup differs from the verifier's: "
                                 + "; ".join(bad))


def _ship(c: DeclaredComputation, store: SharedStore, tree: Any,
          dataset_paths: Sequence[Sequence[bytes]] | None, sec: Section) -> InMemoryStore:
    """Copy the opened step to the verifier's GPUs (staged mode, FS58b): every tensor leaf to
    its placement device, records as they are. The source GPUs are synchronized once, so the
    caller may close the shared store."""
    with sec(SHIP_SECTION):
        leaves: list[Any] = []
        sources: dict[torch.device, None] = {}
        try:
            for i in range(c.n_leaves):
                x = store.leaf(i)
                if isinstance(x, torch.Tensor):
                    if x.is_cuda:
                        sources[x.device] = None
                    x = x.to(c.placement.leaf_device(i), copy=True)
                leaves.append(x)
            for d in sources:
                torch.cuda.synchronize(d)
            return InMemoryStore.hold(c, leaves, tree, dataset_paths)
        except BaseException:
            leaves.clear()  # a traceback must not pin a partial copy on the verifier's GPU
            raise


def _await_release(link: _Link, stash: dict[int, tuple], t: int) -> None:
    """Wait for the child's answer to ``("release", t)``; a refused release is a run error."""
    released = link.expect("released", stash)
    if released[1] != t:
        raise ProverProcessError("protocol", f"released step {released[1]}, not {t}")
    if released[2] is not None:
        raise StoreMutationError(f"step {t}: {released[2]}")


def run_pipelined(c: DeclaredComputation, dataset: Sequence[Any],
                  w0: Mapping[str, torch.Tensor] | None, verifier: Verifier, *,
                  final: Mapping[str, torch.Tensor] | Sequence[bytes],
                  fault: ProverFault | None, schedule: Callable[[int], Sequence[int]] | None,
                  on_step: Callable[[StepRecord], None] | None, section: Section | None,
                  handoff: StoreHandoff | None, release_cache: bool,
                  recorder: TimeRecorder | MemoryRecorder | None,
                  prover: ProverSpec | None, staging: bool | None = None) -> LoopResult:
    """The pipelined loop; see the module docstring. ``run_loop`` calls it with its arguments.

    ``staging`` (``None``: :func:`staging_setting`'s auto) picks the staged mode (FS58b)."""
    kernel_check.new_run()  # first, before anything is dispatched (FS34)
    fault = fault or ProverFault()
    if prover is None:
        raise ValueError("pipelined: the prover's process needs a ProverSpec (prover=...)")
    keep = handoff if isinstance(handoff, KeptStepHandoff) else None
    if handoff is not None and handoff is not IN_MEMORY and keep is None:
        raise ValueError("pipelined: the handoff is the shared store; only IN_MEMORY (the "
                         "default) or a KeptStepHandoff is allowed")
    if _is_cuda(c.placement.prover) and not c.placement.sharded:
        raise ValueError("pipelined: a CUDA prover device needs a sharded placement "
                         "(verifier_devices): an unsharded run's leaves live on the host, and "
                         "each step would be copied whole through shared memory")
    if isinstance(fault, KeepStep):
        raise ValueError("pipelined: KeepStep keeps a step's output in the prover's memory, "
                         "which is another process")
    if recorder is not None and section is not None and section is not recorder.seam():
        raise ValueError("pipelined: section must be the recorder's seam")
    staged = staging_setting(c, staging)
    rspec = _recorder_spec(recorder, c, staged)
    sec = section or no_section
    args = _ChildArgs(_pickled("ProverSpec", prover), _pickled("fault", fault),
                      _pickled("schedule", schedule), rspec, release_cache, verifier.seed,
                      verifier.n_steps, staged)

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child_main, args=(child_conn, args), daemon=True,
                       name="pots-prover")
    link = _Link(proc, parent_conn)
    proc.start()
    child_conn.close()  # so the parent sees the pipe close when the child is gone
    store: SharedStore | None = None
    # Everything below that can hold a package of the child's memory, defined before the try
    # so the finally can drop it (a traceback keeps the frame's locals alive).
    stash: dict[int, tuple] = {}
    msg = pkg = part = released = own = None
    try:
        result = LoopResult(verdict=RunVerdict(False, None, 0, None), pipelined=True)
        # The parent's kernels: its checks 0, 2 and 7 hash the GPU-resident leaves (the
        # child's, opened by IPC) with the hash kernel, and check 5 measures. Loaded here,
        # before ``start_run`` (check 0 hashes W_0 there) and outside every section (FS34,
        # FS37). The child never measures, so it builds only the hash kernel.
        for d in dict.fromkeys(torch.device(x) for x in (c.placement.devices or ())
                               if _is_cuda(x)):
            GpuBlake3().kernels(d)
            measure_kernel.kernels(d, verifier.k)
        if verifier.start_run(dataset) is not None:
            link.stop(grace=0)
            result.verdict = verifier.end_run(final)
            return result
        _, hs, part0 = link.expect("ready")
        _check_handshake(hs, c, verifier, w0)
        if recorder is not None:
            recorder.merge_step(0, part0)
        del hs, part0
        for t in range(1, verifier.n_steps + 1):
            msg = stash.pop(t, None) or link.expect("step", stash)
            if msg[1] != t:
                raise ProverProcessError("protocol", f"expected step {t}, got step {msg[1]}")
            _, _, pkg, part, f = msg
            del msg
            h0 = time.perf_counter()
            store = SharedStore.open(c, pkg)
            kept_from = (pkg.tree, pkg.dataset_paths) if keep is not None and t == keep.step \
                else None
            if staged:
                # FS58b: copy the step to the verifier's GPUs, release it, then verify the copy.
                own = _ship(c, store, pkg.tree, pkg.dataset_paths, sec)
                del pkg
                store.close()
                store = None
                link.send(("release", t))
                if release_cache:
                    release_device_cache("cpu", section, devices=c.placement.devices)
                hs = time.perf_counter()
                _await_release(link, stash, t)
                h1 = time.perf_counter()
                try:
                    rej = verifier.verify_step(t, own)
                finally:
                    h2 = time.perf_counter()
                if kept_from is not None:  # the stored step, from the parent's own copy
                    leaves = [own.leaf(i) for i in range(c.n_leaves)]
                    keep.write_from(c, leaves, *kept_from)
                    del leaves, kept_from
                own = None
                handoff_s = hs - h0
            else:
                del pkg
                h1 = time.perf_counter()
                try:
                    rej = verifier.verify_step(t, store)
                finally:
                    h2 = time.perf_counter()
                wr = 0.0
                if kept_from is not None:  # the stored step: in no section, inside step t's interval
                    w0_ = time.perf_counter()
                    leaves = [store.leaf(i) for i in range(c.n_leaves)]
                    keep.write_from(c, leaves, *kept_from)
                    del leaves, kept_from
                    wr = time.perf_counter() - w0_
                store.close()
                store = None
                h3 = time.perf_counter()
                link.send(("release", t))
                if release_cache:
                    release_device_cache("cpu", section, devices=c.placement.devices)
                h4 = time.perf_counter()
                _await_release(link, stash, t)
                handoff_s = (h1 - h0) + (h3 - h2) + (h4 - h3) - wr
            if recorder is not None:
                recorder.merge_step(t, part)
            rec = StepRecord(t, rej, f["loss"], f["prove_s"], f["commit_s"], h2 - h1,
                             handoff_s, f["capture"],
                             f["transcript_bytes"], f["leaves"], f["root"], f["known"])
            del part, f
            result.steps.append(rec)
            if on_step is not None:
                on_step(rec)
            if rej is not None:
                break
        link.send(("stop", result.steps[-1].t if result.steps else 0))
        msg = link.expect("final", stash)
        stash.clear()  # packages of discarded steps: dropped before bye
        _, w_host, info = msg
        del msg
        link.finished = True
        link.send(("bye",))
        result.w_final = None if w_host is None else dict(w_host)
        result.child_info = {**info, "staging": staged}
        result.verdict = verifier.end_run(final)
        return result
    finally:
        if store is not None:
            store.close()
        stash.clear()
        msg = pkg = part = released = own = None
        gc.collect()  # no package outlives this point: the child exits only after it
        link.stop()
