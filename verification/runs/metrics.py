"""B6: the EQ1b cost grid and the EQ13 run records.

``DECISIONS_EVALUATION.md`` fixes what is measured: EQ1b the cost grid (every component on three
axes: time, compute, memory), EQ1c what compute and memory mean, EQ13 the raw records every run
saves, and EQ14 that test-scale timings are a rehearsal only. This module implements those
records. It decides nothing about the protocol.

**The seam.** A run is observed through ``section(name) -> context manager``, which
:func:`~verification.prover.step.prove_step`, :func:`~verification.prover.step.plain_step`,
:func:`~verification.runs.loop.run_loop` and :class:`~verification.verifier.driver.Verifier`
accept as ``section=``. It is ``None`` by default, and then nothing here runs: a run's weights
and decisions are bit-identical with metrics on or off (tested). A name with the ``run:`` prefix
is a once-per-run section, reported at step 0. A section opened inside another is nested (a
``sub`` row).

**Three passes, one axis each.** Measuring one axis disturbs the others, so each has its own run:

- the *timed run* reads only ``time.perf_counter`` (:class:`TimeRecorder`). It is the run whose
  steps, verdicts and residuals are recorded;
- the *memory pass* (:class:`MemoryRecorder`, :func:`memory_pass`) probes memory at every
  section entry and exit, nested ones included;
- the *counting pass* (:class:`CountRecorder`, :func:`count_pass`, :func:`counting`) counts
  FLOPs, bytes hashed, hash calls and transcript bytes.

The two extra passes are runs of their own, :data:`PASS_STEPS` steps each, never timed.

**Prover components (EQ1b).** EQ1b's P0 is training with no instrumentation, which only the
plain baseline (B7) runs. A verified run trains under ``MatmulCapture``, so its training rows
are named apart and its P1 is derived:

| component | section names | what runs |
|---|---|---|
| P0, original training (plain run only) | ``P0.load``, ``P0.forward``, ``P0.backward``, ``P0.update`` | :func:`plain_step` with a ``section``: load ``W_t``, zero grads and build the optimizer; forward; backward; ``opt.step()`` and the final ``zero_grad`` |
| ``train_captured`` (verified run) | ``train.load``, ``train.forward``, ``train.backward``, ``train.update`` | the same four phases in :func:`prove_step`, forward and backward under capture: P0 plus capture's hooks |
| P1, matmul capture | ``P1.label`` (measured), the rest derived | **derived** by :func:`derive_capture` (``derived: true``): time ``Σ(train.x − P0.x) + P1.label`` per step; memory ``(peak(train.backward) − start(train.forward)) − (peak(P0.backward) − start(P0.forward))``, the growth over the step, not absolute peaks; counts ``Σ(train.x − P0.x)`` per field. ``P1.label`` maps captured matmuls to product slots, checks ``M``, runs invariant 6's ``assert_unmodified`` and releases the captured operands |
| P2, transcript serialization | ``P2.w_t``, ``P2.w_next`` | copy ``W_t`` and ``W_{t+1}`` out of the model as leaves. Leaf encoding itself is zero-copy (``tensor_leaf_payload`` is a view), so serialization's byte cost is inside P3's hashing |
| P3, Merkle commitment | ``P3.commit`` | ``commit_leaves`` (hash every leaf, build the tree) and the invariant-6 recheck (``InMemoryStore.commit_step``) |
| P4, batch paths into ``h_D`` | ``P4.paths`` per step; ``P4.tree`` once (step 0) | the batch's audit paths; building the prover's copy of ``h_D``'s tree |
| P5, writing the transcript | ``P5.write`` | handing the committed leaves to the store (``InMemoryStore(...)``). About 0 at test scale, where the store holds references; A14's ``DiskStore`` write lands here |

Fault hooks (``ProverFault.emit`` and friends) and dropping the step's store are harness cost and
sit outside every section. A verified run never emits a ``P0`` row, and the recorder never emits
a ``P1`` component: only :func:`derive_capture` does. Prover overhead is P1–P5.

**Verifier components** are the checks, by id, in driver order: per step ``4``, ``7``, ``2``,
``6a``, ``5``, ``6b``; once per run (step 0) ``0`` (hashing the agreed ``W_0``), ``1``, ``8`` and
``9``. At step 1 the chaining comparison runs in check 7's slot against check 0's anchor
hashes; its row is ``7``, so check 7's series starts at step 1, and check 0's row is the anchor
hashing. ``9`` wraps building the verdict, kept as a row though its cost is negligible, so
every check has one. Check 5 is split (EQ1b) into ``5.glue`` (replay construction and every
``operands(m)``: glue recomputation) and ``5.measure`` (the Freivalds products), both ``sub``
rows. Check 3 has no row of its own: it is the rule that check 5's operands come from committed
leaves, and its cost is ``5.glue``. ``6b.glue`` is the replayed glue gradients.

**Rows.** One per section per step, at a ``level``: ``phase`` (a prover section),
``component`` (a prover component, the sum of its phases, or a check), ``sub`` (a nested
section) or ``total`` (``prover``, ``verifier``, and ``step``, their sum; a total sums its
top-level sections, so loop bookkeeping and fault hooks are not in it). A section entered twice
in a step (``train.load``, ``5.glue``) accumulates and ``calls`` counts the entries.

**Time (timed run).** Wall-clock seconds from ``perf_counter`` and nothing else. On CUDA the
device is synchronized at top-level section boundaries and nested sections are timed with
``torch.cuda.Event`` pairs, read after the enclosing top-level sync; on MPS the device is
synchronized at every boundary. So on CUDA a nested row is device time (the Event pair) while
a top-level row is host time across a synchronized interval; the two clocks are not checked
against each other yet, which the first CUDA run must do. Test-scale times are rehearsal only
(EQ14).

**Memory (memory pass).** ``peak_bytes`` is the whole process's peak during the section, as an
absolute figure; ``start_bytes`` and ``end_bytes`` are the process's figure at the first entry
and the last exit, so ``peak − start`` is what the section added on top of what was live. A
nested section has its own peak: each entry and exit first folds the peak so far into every
open section, then the entry resets it. The backend is named in ``mem_source``:

- ``darwin_phys_footprint`` (macOS): ``phys_footprint``, Activity Monitor's "Memory" (resident,
  compressed and swapped dirty pages), with the kernel's interval maximum
  (``proc_pid_rusage`` ``RUSAGE_INFO_V4``) reset by ``proc_rlimit_control``. macOS ``malloc``
  keeps freed large blocks, so export ``MallocLargeCache=0`` before Python starts for every run
  being compared; the environment record keeps the setting;
- ``linux_rss_hwm`` (Linux): ``VmRSS``/``VmHWM`` from ``/proc/self/status``, the high-water mark
  reset by writing ``5`` to ``/proc/self/clear_refs`` (EQ1c's peak RSS);
- ``cuda_allocated`` (CUDA): ``memory_allocated``/``max_memory_allocated``, bytes in live
  tensors without the caching allocator's reserve (EQ1c), synchronized at each read. Host
  memory is not in these rows; the ``run_end`` record keeps the process's lifetime ``ru_maxrss``;
- ``ru_maxrss`` (anything else): the lifetime high-water mark, which can't be reset; start and
  end are ``null``.

MPS uses the host probe (unified memory). A probe that fails with ``OSError`` is logged once and
turned off: later rows carry ``null`` and ``mem_source`` ``unavailable``.

**Counts (counting pass).** ``flops``: torch's FLOP formulas (``flop_registry``, the ones
``FlopCounterMode`` applies) over every op, which covers the matmul-class ops; elementwise
work (checks 6a, 6b, glue) counts 0. ``FlopCounterMode`` itself can't run here (see
:class:`_FlopTally`); a test checks the two agree on a plain step. Hashing (EQ1c) counts
``hash_in_bytes``, ``hash_out_bytes`` (out is check 5's challenge stream) and ``hash_calls``
(one per BLAKE3 hasher), through a counting stand-in for ``blake3`` in ``commitment.merkle``,
``matmul_check.challenges`` and ``verifier.bands``. The band file is hashed when it is loaded,
before the run and outside every section, so it is in no row; at test scale the bands are
provisional and nothing is hashed. ``setup.data``'s manifest hash is data preparation, not run
cost. A nested row's counts are also in its parent's.

**Transcript bytes (EQ1c storage).** A ``storage`` record per step: the canonical encoded size
of the step's leaves (headers and payloads, as hashed in ``P3.commit``) and the leaf count, from
the counting pass.

**Files** (EQ13), under the run's directory:

- ``records.jsonl``: one JSON object per line, each with ``record`` (its type) and ``run``.
  Types: ``environment`` (first), ``step``, ``verdict``, ``time``, ``memory``, ``count``,
  ``storage`` and ``run_end`` (last). Standard JSON: a non-finite float is written as the
  string ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, and a missing value as ``null``;
- ``residuals/<scenario>/step_<t>.npz``: one NumPy archive per step with residuals (EQ13
  record 2), written as the step closes, so a crashed run keeps finished steps; a step that
  stopped before check 6a has none. Arrays: ``run``, ``scenario`` (0-d strings), ``step``;
  per product ``p_m``, ``p_name``, ``p_cls``, ``p_layer`` (``-1`` outside the layer stack),
  ``p_kappa`` and ``p_normalized`` (``[P, k]``, one normalized residual per challenge); per
  weight tensor ``w_check`` (``6a``/``6b``), ``w_name`` and ``w_rho`` (``ρ_max``). NaN and
  infinity are stored as such. Check 5 stops at the first failing product, so a step rejected
  at 5 has products up to and including the failure only. :func:`read_residuals` rebuilds ``StepStats`` equal to the
  verifier's, the feed for calibration (A11).
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Protocol

import blake3
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.context import ProductStat, StepStats, TensorStat

__all__ = [
    "P0_PHASES", "TRAIN_PHASES", "PASS_STEPS", "COMPONENTS", "RECORD_TYPES",
    "MemoryProbe", "memory_probe",
    "TimeRecorder", "MemoryRecorder", "CountRecorder",
    "memory_pass", "count_pass", "counting",
    "derive_capture", "step_record", "config_hash",
    "MetricsWriter", "read_records", "read_residuals", "lifetime_maxrss_bytes",
]

log = logging.getLogger(__name__)

P0_PHASES = ("P0.load", "P0.forward", "P0.backward", "P0.update")
TRAIN_PHASES = ("train.load", "train.forward", "train.backward", "train.update")
# Section-name prefix -> component row. P1 is absent: it is derived (derive_capture).
_COMPONENT_OF = {"P0": "P0", "train": "train_captured", "P2": "P2", "P3": "P3", "P4": "P4",
                 "P5": "P5"}
COMPONENTS = ("P0", "train_captured", "P1", "P2", "P3", "P4", "P5")
RECORD_TYPES = ("environment", "step", "verdict", "time", "memory", "count", "storage",
                "run_end")
RUN_PREFIX = "run:"
# Steps in every run's memory and counting passes. Step 1 carries check 0's anchor and the
# first-touch allocation of the run's buffers; step 2 is the first check-7 step and the first
# in steady-state memory. Counts are deterministic, so a report takes one step's counts and two
# steps suffice. The plain run's passes match the verified runs' step for step (derive_capture
# pairs them by step number).
PASS_STEPS = 2


def _side(name: str) -> str:
    return "prover" if name.startswith(("P", "train.")) else "verifier"


# ---- memory backends ---------------------------------------------------------------------


class MemoryProbe(Protocol):
    source: str

    def reset(self) -> None:
        """Start a new peak interval at the current figure."""
        ...

    def sample(self) -> tuple[int | None, int | None]:
        """``(peak since the last reset, current)``."""
        ...


class _DarwinProbe:
    source = "darwin_phys_footprint"
    _RUSAGE_INFO_V4 = 4
    _RLIMIT_FOOTPRINT_INTERVAL = 4
    _FOOTPRINT_INTERVAL_RESET = 1
    # rusage_info_v4 as uint64 words: a 16-byte uuid, then the fields in order.
    _PHYS_FOOTPRINT = 9
    _INTERVAL_MAX = 35

    def __init__(self) -> None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self._ru = lib.proc_pid_rusage
        self._ru.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._ru.restype = ctypes.c_int
        self._rl = lib.proc_rlimit_control
        self._rl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._rl.restype = ctypes.c_int
        self._buf = (ctypes.c_uint64 * 40)()
        self._addr = ctypes.addressof(self._buf)
        self._pid = os.getpid()
        self.reset()
        self.sample()

    def reset(self) -> None:
        if self._rl(self._pid, self._RLIMIT_FOOTPRINT_INTERVAL, self._FOOTPRINT_INTERVAL_RESET):
            raise OSError(ctypes.get_errno(), "proc_rlimit_control")

    def sample(self) -> tuple[int, int]:
        if self._ru(self._pid, self._RUSAGE_INFO_V4, self._addr):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage")
        cur = int(self._buf[self._PHYS_FOOTPRINT])
        return max(int(self._buf[self._INTERVAL_MAX]), cur), cur


class _LinuxProbe:
    source = "linux_rss_hwm"

    def __init__(self) -> None:
        self._clear = open("/proc/self/clear_refs", "w")  # noqa: SIM115 (kept open for speed)
        self.reset()

    def reset(self) -> None:
        self._clear.write("5")
        self._clear.flush()

    def sample(self) -> tuple[int, int]:
        rss = hwm = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
        return max(hwm, rss), rss


class _CudaProbe:
    source = "cuda_allocated"

    def __init__(self, device: torch.device) -> None:
        self._dev = device

    def reset(self) -> None:
        torch.cuda.synchronize(self._dev)
        torch.cuda.reset_peak_memory_stats(self._dev)

    def sample(self) -> tuple[int, int]:
        torch.cuda.synchronize(self._dev)
        return (int(torch.cuda.max_memory_allocated(self._dev)),
                int(torch.cuda.memory_allocated(self._dev)))


class _MaxRssProbe:
    source = "ru_maxrss"
    _UNIT = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, kB on Linux

    def reset(self) -> None:
        pass

    def sample(self) -> tuple[int, None]:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * self._UNIT, None


class _Guarded:
    """A probe that turns itself off at its first ``OSError``, logged once."""

    def __init__(self, inner: MemoryProbe) -> None:
        self._inner: MemoryProbe | None = inner
        self.source = inner.source

    def _fail(self, e: OSError) -> None:
        log.warning("memory probe %s failed (%s); memory rows are null from here", self.source, e)
        self._inner, self.source = None, "unavailable"

    def reset(self) -> None:
        if self._inner is not None:
            try:
                self._inner.reset()
            except OSError as e:
                self._fail(e)

    def sample(self) -> tuple[int | None, int | None]:
        if self._inner is None:
            return None, None
        try:
            return self._inner.sample()
        except OSError as e:
            self._fail(e)
            return None, None


def memory_probe(device: str | torch.device = "cpu") -> MemoryProbe:
    """The memory backend for ``device`` on this platform (see the module docstring)."""
    dev = torch.device(device)
    inner: MemoryProbe
    if dev.type == "cuda":
        inner = _CudaProbe(dev)
    else:
        try:
            if sys.platform == "darwin":
                inner = _DarwinProbe()
            elif sys.platform.startswith("linux"):
                inner = _LinuxProbe()
            else:
                inner = _MaxRssProbe()
        except OSError as e:
            log.warning("no resettable memory peak (%s); using ru_maxrss", e)
            inner = _MaxRssProbe()
    return _Guarded(inner)


def lifetime_maxrss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MaxRssProbe._UNIT


# ---- shared bookkeeping ------------------------------------------------------------------


class _Acc:
    """One section's totals within a step."""

    __slots__ = ("nested", "calls", "time", "peak", "start", "end", "seen", "counts")

    def __init__(self, nested: bool) -> None:
        self.nested = nested
        self.calls = 0
        self.time = 0.0
        self.peak: int | None = None
        self.start: int | None = None
        self.end: int | None = None
        self.seen = False
        self.counts = [0] * len(_COUNTERS)


class _Recorder:
    """Sections per step, ``run:`` sections at step 0, and the nesting depth.

    Rows per step: prover phases, prover components, checks, subs, then totals.
    """

    record = ""
    _Section: type

    def __init__(self, run: str, scenario: str) -> None:
        self.run = run
        self.scenario = scenario
        self.rows: list[dict[str, Any]] = []
        self._step: dict[str, _Acc] = {}
        self._run: dict[str, _Acc] = {}
        self._depth = 0
        self._verifier: Any = None

    def bind(self, verifier: Any) -> None:
        """Remember the run's ``Verifier`` (the timed run reads its timings and stats)."""
        self._verifier = verifier

    def section(self, name: str) -> Any:
        """The seam: a context manager that measures the block as section ``name``."""
        if name.startswith(RUN_PREFIX):
            table, name = self._run, name[len(RUN_PREFIX):]
        else:
            table = self._step
        acc = table.get(name)
        if acc is None:
            acc = table[name] = _Acc(self._depth > 0)
        return self._Section(self, acc)

    def on_step(self, rec: Any) -> None:
        """``run_loop``'s ``on_step``: close step ``rec.t``."""
        table, self._step = self._step, {}
        self._emit(rec.t, table, rec)

    def finish(self, verdict: Any = None) -> None:
        """Close the run: the ``run:`` sections become step 0."""
        table, self._run = self._run, {}
        if table:
            self._emit(0, table, None)

    def _emit(self, t: int, table: dict[str, _Acc], rec: Any) -> None:
        self.rows += self._rows(t, table)

    def _rows(self, step: int, table: dict[str, _Acc]) -> list[dict[str, Any]]:
        top = {n: a for n, a in table.items() if not a.nested}
        prover = {n: a for n, a in top.items() if _side(n) == "prover"}
        verifier = {n: a for n, a in top.items() if _side(n) == "verifier"}
        rows = [self._row(step, "prover", n, "phase", [a]) for n, a in prover.items()]
        for prefix, comp in _COMPONENT_OF.items():
            accs = [a for n, a in prover.items() if n.split(".", 1)[0] == prefix]
            if accs:
                rows.append(self._row(step, "prover", comp, "component", accs))
        rows += [self._row(step, "verifier", n, "component", [a]) for n, a in verifier.items()]
        rows += [self._row(step, _side(n), n, "sub", [a]) for n, a in table.items() if a.nested]
        for side, part in (("prover", prover), ("verifier", verifier)):
            if part:
                rows.append(self._row(step, side, side, "total", list(part.values())))
        if top:
            rows.append(self._row(step, "run", "step", "total", list(top.values())))
        return rows

    def _row(self, step: int, side: str, comp: str, level: str,
             accs: list[_Acc]) -> dict[str, Any]:
        return {"record": self.record, "run": self.run, "scenario": self.scenario, "step": step,
                "side": side, "component": comp, "level": level,
                "calls": sum(a.calls for a in accs), **self._values(accs), "derived": False}

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        raise NotImplementedError


# ---- time: the timed run -----------------------------------------------------------------


class _TimeSection:
    __slots__ = ("rec", "acc", "t0")

    def __init__(self, rec: TimeRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        self.rec._depth += 1
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: Any) -> None:
        dt = time.perf_counter() - self.t0
        acc = self.acc
        acc.time += dt
        acc.calls += 1
        self.rec._depth -= 1


class _SyncTimeSection(_TimeSection):
    """MPS: synchronize at every boundary, so the clock sees the device's work."""

    __slots__ = ()

    def __enter__(self) -> None:
        self.rec._sync()
        super().__enter__()

    def __exit__(self, *exc: Any) -> None:
        self.rec._sync()
        super().__exit__(*exc)


class _CudaTimeSection:
    """CUDA: top-level sections synchronize and read the host clock; nested sections record a
    pair of events, read after the enclosing top-level section's synchronize."""

    __slots__ = ("rec", "acc", "t0", "top", "e0")

    def __init__(self, rec: TimeRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        rec = self.rec
        self.top = rec._depth == 0
        rec._depth += 1
        if self.top:
            torch.cuda.synchronize(rec._device)
            self.t0 = time.perf_counter()
        else:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()

    def __exit__(self, *exc: Any) -> None:
        rec, acc = self.rec, self.acc
        acc.calls += 1
        rec._depth -= 1
        if not self.top:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            rec._pending.append((acc, self.e0, e1))
            return
        torch.cuda.synchronize(rec._device)
        acc.time += time.perf_counter() - self.t0
        for a, e0, e1 in rec._pending:
            a.time += e0.elapsed_time(e1) / 1e3
        rec._pending.clear()


class TimeRecorder(_Recorder):
    """The timed run: wall-clock seconds per section, nothing else measured.

    Wire it as ``section=`` on ``run_loop`` and the ``Verifier``, :meth:`on_step` as
    ``on_step``, :meth:`bind` the verifier, and call :meth:`finish` with the run's verdict. With
    a ``writer``, each closed step writes its ``time`` rows, its ``step`` record and its residual
    arrays; :meth:`finish` writes the ``verdict`` record.
    """

    record = "time"
    _Section = _TimeSection

    def __init__(self, run: str, scenario: str, *, device: str | torch.device = "cpu",
                 writer: MetricsWriter | None = None,
                 step_fields: Mapping[str, Any] | None = None) -> None:
        super().__init__(run, scenario)
        self._device = torch.device(device)
        if self._device.type == "cuda":
            self._Section = _CudaTimeSection
            self._pending: list[tuple[_Acc, Any, Any]] = []
        elif self._device.type == "mps":
            self._Section = _SyncTimeSection
            self._sync = torch.mps.synchronize
        self.writer = writer
        self.step_fields = dict(step_fields or {})

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        return {"time_s": sum(a.time for a in accs)}

    def _emit(self, t: int, table: dict[str, _Acc], rec: Any) -> None:
        rows = self._rows(t, table)
        if rec is not None and self._verifier is not None:
            rows.append(step_record(self.run, self.scenario, t, self._verifier,
                                    getattr(rec, "rejection", None), self.step_fields))
        self.rows += rows
        if self.writer is None:
            return
        self.writer.write(rows)
        if rec is not None and self._verifier is not None:
            stats = self._verifier.stats.get(t)
            if stats is not None and (stats.products or stats.tensors):
                self.writer.write_residuals(self.scenario, t, stats, self._verifier.c)

    def finish(self, verdict: Any = None) -> None:
        super().finish()
        if verdict is None:
            return
        rej = verdict.rejection
        row = {"record": "verdict", "run": self.run, "scenario": self.scenario,
               "accepted": bool(verdict.accepted),
               "rejection_step": None if rej is None else rej.step,
               "rejection_check": None if rej is None else rej.check_id,
               "rejection_kind": None if rej is None else rej.kind,
               "steps_verified": verdict.steps_verified, "band_source": verdict.band_source,
               **self.step_fields}
        self.rows.append(row)
        if self.writer is not None:
            self.writer.write([row])


def step_record(run: str, scenario: str, t: int, verifier: Any, rejection: Any,
                fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """EQ13 record 1 for step ``t``: the verdict, the first failing check, and each check's
    ``pass``, ``fail`` or ``not_run``.

    ``checks`` and ``first_failing_check`` use protocol ids, the ids a ``Rejection`` carries:
    at step 1 the chain slot is check 0 (``ctx.chain_check_id``), so it is keyed ``0``, and
    ``checks[first_failing_check]`` is ``fail``. The cost rows keep the driver's slot id
    ``7`` at step 1 too."""
    def pid(cid: str) -> str:
        return "0" if t == 1 and cid == "7" else cid

    ran = [pid(c) for c in verifier.timings.get(t, {})]
    failed = rejection is not None and rejection.step == t
    checks = {pid(c): ("pass" if pid(c) in ran else "not_run") for c in DEFAULT_ORDER}
    if failed:
        cid = rejection.check_id if rejection.check_id in checks else (ran[-1] if ran else None)
        if cid is not None:
            checks[cid] = "fail"
    out = {"record": "step", "run": run, "scenario": scenario,
           "model": None, "corpus": None, "seed": None,
           "poisoning_rate": None, "cheat_step": None, **(fields or {}),
           "step": t, "verdict": "reject" if failed else "accept",
           "first_failing_check": rejection.check_id if failed else None,
           "rejection_kind": rejection.kind if failed else None, "checks": checks}
    return out


# ---- memory: the memory pass -------------------------------------------------------------


class _MemSection:
    __slots__ = ("rec", "acc", "peak")

    def __init__(self, rec: MemoryRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        rec, acc = self.rec, self.acc
        rec._fold()  # the open sections keep the peak so far before it is reset
        rec.probe.reset()
        self.peak, cur = rec.probe.sample()
        if not acc.seen:
            acc.start, acc.seen = cur, True
        rec._open.append(self)
        rec._depth += 1

    def __exit__(self, *exc: Any) -> None:
        rec, acc = self.rec, self.acc
        cur = rec._fold()
        rec._open.pop()
        rec._depth -= 1
        acc.calls += 1
        acc.end = cur
        if self.peak is not None and (acc.peak is None or self.peak > acc.peak):
            acc.peak = self.peak


class MemoryRecorder(_Recorder):
    """The memory pass: the process's peak per section, nested sections included."""

    record = "memory"
    _Section = _MemSection

    def __init__(self, run: str, scenario: str, probe: MemoryProbe) -> None:
        super().__init__(run, scenario)
        self.probe = probe
        self._open: list[_MemSection] = []

    def _fold(self) -> int | None:
        peak, cur = self.probe.sample()
        if peak is not None:
            for s in self._open:
                if s.peak is None or peak > s.peak:
                    s.peak = peak
        return cur

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        peaks = [a.peak for a in accs if a.peak is not None]
        return {"peak_bytes": max(peaks) if peaks else None, "start_bytes": accs[0].start,
                "end_bytes": accs[-1].end, "mem_source": self.probe.source}


def memory_pass(run: str, scenario: str, fn: Callable[[MemoryRecorder], Any],
                probe: MemoryProbe) -> list[dict[str, Any]]:
    """Run ``fn(recorder)`` once with memory probed at every section; return its rows.

    ``fn`` wires the recorder as a timed run wires a :class:`TimeRecorder`. Its times are never
    reported.
    """
    rec = MemoryRecorder(run, scenario, probe)
    fn(rec)
    rec.finish()
    return rec.rows


# ---- counts: the counting pass -----------------------------------------------------------

# Counter slots: FLOPs, BLAKE3 bytes in and out, hasher count, leaf bytes, leaf count.
_COUNTERS = ("flops", "hash_in_bytes", "hash_out_bytes", "hash_calls", "leaf_bytes", "leaves")


class _HashTally:
    __slots__ = ("bytes_in", "bytes_out", "calls", "leaf_bytes", "leaves")

    def __init__(self) -> None:
        self.bytes_in = self.bytes_out = self.calls = self.leaf_bytes = self.leaves = 0


def _counting_blake3(real: Any, tally: _HashTally) -> Any:
    """A stand-in for the ``blake3`` module that counts hashers and bytes in and out."""

    class blake3:  # noqa: N801 (mirrors blake3.blake3)
        AUTO = real.blake3.AUTO

        def __init__(self, data: Any = None, /, **kw: Any) -> None:
            tally.calls += 1
            if data is None:
                self._h = real.blake3(**kw)
            else:
                tally.bytes_in += memoryview(data).nbytes
                self._h = real.blake3(data, **kw)

        def update(self, data: Any, /) -> blake3:
            tally.bytes_in += memoryview(data).nbytes
            self._h.update(data)
            return self

        def digest(self, length: int = 32, **kw: Any) -> bytes:
            tally.bytes_out += length
            return self._h.digest(length, **kw)

        def hexdigest(self, length: int = 32, **kw: Any) -> str:
            tally.bytes_out += length
            return self._h.hexdigest(length, **kw)

    class _Module:
        pass

    mod = _Module()
    mod.blake3 = blake3  # type: ignore[attr-defined]
    return mod


def _counting_hash_leaf(real: Callable[..., bytes], tally: _HashTally) -> Callable[..., bytes]:
    def hash_leaf(*parts: Any) -> bytes:
        tally.leaves += 1
        tally.leaf_bytes += sum(memoryview(p).nbytes for p in parts)
        return real(*parts)
    return hash_leaf


_HASHING_MODULES = ("verification.commitment.merkle",
                    "verification.verifier.matmul_check.challenges",
                    "verification.verifier.bands")
_LEAF_MODULE = "verification.commitment.leaves"  # imports hash_leaf, hash_leaves… by name


def _counting_hash_leaves_until_error(counting_leaf: Callable[..., bytes]
                                      ) -> Callable[..., tuple[list[bytes], Exception | None]]:
    # Leaf by leaf on the calling thread: the tally is not thread-safe, and the counts are the
    # same as the parallel path's. Same outcome as merkle.hash_leaves_until_error.
    def hash_leaves_until_error(leaves: Any) -> tuple[list[bytes], Exception | None]:
        out: list[bytes] = []
        try:
            for parts in leaves:
                out.append(counting_leaf(*parts))
        except Exception as e:
            return out, e
        return out, None
    return hash_leaves_until_error


def _counting_hash_leaves(until_error: Callable[..., tuple[list[bytes], Exception | None]]
                          ) -> Callable[..., list[bytes]]:
    def hash_leaves(leaves: Any) -> list[bytes]:
        out, err = until_error(leaves)
        if err is not None:
            raise err
        return out
    return hash_leaves


class _FlopTally(TorchDispatchMode):
    """Torch's own FLOP formulas (``torch.utils.flop_counter.flop_registry``, the ones
    ``FlopCounterMode`` applies) summed over every op dispatched in the block.

    EQ1c names ``FlopCounterMode``. It can't run here: its module tracker installs autograd
    hooks that break the ``torch.autograd.grad`` calls replay makes on leaf tensors ("A leaf
    node was passed to _will_engine_execute_node"). The formulas are the same, and a test checks
    the totals agree on a plain forward and backward.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        formula = flop_registry.get(func._overloadpacket)
        if formula is not None:
            self.total += formula(*args, **kwargs, out_val=out)
        return out


@contextmanager
def counting() -> Iterator[tuple[_FlopTally, _HashTally]]:
    """Count FLOPs, hashing and leaf bytes for the duration of the block; ``(flops, tally)``.

    Never around a timed run: the counters slow the code they observe.
    """
    tally = _HashTally()
    mods = [importlib.import_module(m) for m in _HASHING_MODULES]
    real = [m.blake3 for m in mods]
    leaves = importlib.import_module(_LEAF_MODULE)
    real_leaf, real_leaves = leaves.hash_leaf, leaves.hash_leaves
    real_until = leaves.hash_leaves_until_error
    for m, r in zip(mods, real, strict=True):
        m.blake3 = _counting_blake3(r, tally)
    leaves.hash_leaf = _counting_hash_leaf(real_leaf, tally)
    leaves.hash_leaves_until_error = _counting_hash_leaves_until_error(leaves.hash_leaf)
    leaves.hash_leaves = _counting_hash_leaves(leaves.hash_leaves_until_error)
    try:
        with _FlopTally() as flops:
            yield flops, tally
    finally:
        for m, r in zip(mods, real, strict=True):
            m.blake3 = r
        leaves.hash_leaf, leaves.hash_leaves = real_leaf, real_leaves
        leaves.hash_leaves_until_error = real_until


class _CountSection:
    __slots__ = ("rec", "acc", "token")

    def __init__(self, rec: CountRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        self.rec._depth += 1
        self.token = self.rec._now()

    def __exit__(self, *exc: Any) -> None:
        now = self.rec._now()
        counts = self.acc.counts
        for i, (a, b) in enumerate(zip(self.token, now, strict=True)):
            counts[i] += b - a
        self.acc.calls += 1
        self.rec._depth -= 1


class CountRecorder(_Recorder):
    """The counting pass, inside :func:`counting`: ``count`` rows, and a ``storage`` record per
    step with the transcript bytes hashed in ``P3.commit``."""

    record = "count"
    _Section = _CountSection

    def __init__(self, run: str, scenario: str, flops: _FlopTally, tally: _HashTally) -> None:
        super().__init__(run, scenario)
        self._fm, self._tally = flops, tally

    def _now(self) -> tuple[int, ...]:
        t = self._tally
        return self._fm.total, t.bytes_in, t.bytes_out, t.calls, t.leaf_bytes, t.leaves

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        return {name: sum(a.counts[i] for a in accs) for i, name in enumerate(_COUNTERS[:4])}

    def _emit(self, t: int, table: dict[str, _Acc], rec: Any) -> None:
        self.rows += self._rows(t, table)
        commit = table.get("P3.commit")
        if commit is not None:
            self.rows.append({"record": "storage", "run": self.run, "scenario": self.scenario,
                              "step": t, "transcript_bytes": commit.counts[4],
                              "leaves": commit.counts[5]})


def count_pass(run: str, scenario: str, fn: Callable[[CountRecorder], Any]) -> list[dict[str, Any]]:
    """Run ``fn(recorder)`` once under :func:`counting` and return its rows. Never timed."""
    with counting() as (fm, tally):
        rec = CountRecorder(run, scenario, fm, tally)
        fn(rec)
        rec.finish()
    return rec.rows


# ---- P1, derived ------------------------------------------------------------------------


def derive_capture(verified: Iterable[Mapping[str, Any]],
                   plain: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """P1 (matmul capture, EQ1b) per step, from a verified run's records and a plain run's.

    The plain run (B7) must hold exactly one ``(run, scenario)``; its rows are matched to the
    verified run's by step. For each step both have:

    - time: ``Σ_x (train.x − P0.x) + P1.label`` over the four training phases;
    - memory: ``(peak(train.backward) − start(train.forward)) − (peak(P0.backward) −
      start(P0.forward))``, the growth each step adds over what was live when it began, from
      the two memory passes; absolute peaks would carry whatever else each process held.
      ``null`` if any of the four figures is;
    - counts: ``Σ_x (train.x − P0.x)`` per count field, from the two counting passes.

    Each row is a ``component`` row named ``P1`` with ``derived: true``. A step missing one of
    the phases raises ``ValueError`` naming it.
    """
    def phases(rows: list[Mapping[str, Any]], record: str
               ) -> dict[tuple[str, str, int], dict[str, Mapping[str, Any]]]:
        out: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
        for r in rows:
            if r.get("record") == record and r.get("level") == "phase" and r["step"] > 0:
                out.setdefault((r["run"], r["scenario"], r["step"]), {})[r["component"]] = r
        return out

    def get(table: Mapping[str, Mapping[str, Any]], name: str, t: int) -> Mapping[str, Any]:
        if name not in table:
            raise ValueError(f"derive_capture: no {name!r} phase row at step {t}")
        return table[name]

    verified, plain = list(verified), list(plain)
    runs = {(r["run"], r["scenario"]) for r in plain
            if r.get("record") in ("time", "memory", "count") and r.get("level") == "phase"}
    if len(runs) > 1:
        raise ValueError(f"derive_capture: the plain rows hold {len(runs)} (run, scenario) "
                         f"pairs, {sorted(runs)}; pass one")

    def by_step(record: str) -> dict[int, dict[str, Mapping[str, Any]]]:
        return {k[2]: v for k, v in phases(plain, record).items()}

    xs = ("load", "forward", "backward", "update")
    count_fields = ("flops", "hash_in_bytes", "hash_out_bytes", "hash_calls")
    plain_time, plain_mem, plain_count = by_step("time"), by_step("memory"), by_step("count")
    base = {"side": "prover", "component": "P1", "level": "component", "derived": True}
    out: list[dict[str, Any]] = []
    for (run, scen, t), v in phases(verified, "time").items():
        p = plain_time.get(t)
        if p is None:
            continue
        label = get(v, "P1.label", t)
        dt = sum(get(v, f"train.{x}", t)["time_s"] - get(p, f"P0.{x}", t)["time_s"] for x in xs)
        out.append({"record": "time", "run": run, "scenario": scen, "step": t, **base,
                    "calls": label["calls"], "time_s": dt + label["time_s"]})
    for (run, scen, t), v in phases(verified, "memory").items():
        p = plain_mem.get(t)
        if p is None:
            continue
        figs = (get(v, "train.backward", t)["peak_bytes"],
                get(v, "train.forward", t)["start_bytes"],
                get(p, "P0.backward", t)["peak_bytes"], get(p, "P0.forward", t)["start_bytes"])
        grow = None if any(f is None for f in figs) else \
            (figs[0] - figs[1]) - (figs[2] - figs[3])
        out.append({"record": "memory", "run": run, "scenario": scen, "step": t, **base,
                    "calls": 1, "peak_bytes": grow, "start_bytes": None, "end_bytes": None,
                    "mem_source": v["train.backward"]["mem_source"]})
    for (run, scen, t), v in phases(verified, "count").items():
        p = plain_count.get(t)
        if p is None:
            continue
        out.append({"record": "count", "run": run, "scenario": scen, "step": t, **base,
                    "calls": 1, **{f: sum(get(v, f"train.{x}", t)[f] - get(p, f"P0.{x}", t)[f]
                                          for x in xs) for f in count_fields}})
    return out


# ---- residual arrays ---------------------------------------------------------------------


def _residual_arrays(run: str, scenario: str, step: int, stats: StepStats,
                     c: Any) -> dict[str, np.ndarray]:
    ps = stats.products
    k = len(ps[0].normalized) if ps else 0
    layers = []
    for p in ps:
        layer = c.product(p.m).layer if c is not None else None
        layers.append(-1 if layer is None else int(layer))
    return {
        "run": np.array(run), "scenario": np.array(scenario), "step": np.array(step),
        "p_m": np.array([p.m for p in ps], dtype=np.int32),
        "p_name": np.array([p.name for p in ps], dtype=str),
        "p_cls": np.array([p.cls for p in ps], dtype=str),
        "p_layer": np.array(layers, dtype=np.int32),
        "p_kappa": np.array([p.kappa for p in ps], dtype=np.float64),
        "p_normalized": np.array([p.normalized for p in ps], dtype=np.float64).reshape(len(ps), k),
        "w_check": np.array([s.check_id for s in stats.tensors], dtype=str),
        "w_name": np.array([s.weight for s in stats.tensors], dtype=str),
        "w_rho": np.array([s.rho_max for s in stats.tensors], dtype=np.float64),
    }


def _stats_from(z: Mapping[str, np.ndarray]) -> StepStats:
    st = StepStats()
    for i in range(len(z["p_m"])):
        st.products.append(ProductStat(int(z["p_m"][i]), str(z["p_name"][i]), str(z["p_cls"][i]),
                                       float(z["p_kappa"][i]),
                                       tuple(float(x) for x in z["p_normalized"][i])))
    for i in range(len(z["w_name"])):
        st.tensors.append(TensorStat(str(z["w_name"][i]), str(z["w_check"][i]),
                                     float(z["w_rho"][i])))
    return st


def read_residuals(run_dir: str | os.PathLike[str]) -> dict[str, dict[int, StepStats]]:
    """The run's residual arrays back as ``{scenario: {step: StepStats}}``."""
    out: dict[str, dict[int, StepStats]] = {}
    for f in sorted(Path(run_dir, "residuals").glob("*/step_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            out.setdefault(str(z["scenario"]), {})[int(z["step"])] = _stats_from(z)
    return out


# ---- the environment and the files --------------------------------------------------------


def config_hash(config: Mapping[str, Any]) -> str:
    """BLAKE3 hex of the config as canonical JSON (sorted keys; paths and dtypes as strings)."""
    b = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode()
    return blake3.blake3(b).hexdigest()


def _cpu_model() -> str | None:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip() or None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              timeout=10, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


_LIBRARIES = ("torch", "numpy", "blake3", "transformers", "datasets")


def environment_record(run: str, *, device: str | torch.device, probe: MemoryProbe,
                       config: Mapping[str, Any] | None = None, band_file_hash: str | None = None,
                       h_D: bytes | None = None, **fields: Any) -> dict[str, Any]:
    """EQ13 record 5: hardware, library versions, git commit and the run's input hashes."""
    dev = torch.device(device)
    status = _git("status", "--porcelain")
    return {
        "record": "environment", "run": run,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {"machine": platform.machine(), "cpu": _cpu_model(),
                     "cpu_count": os.cpu_count(),
                     "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None},
        "device": str(dev), "platform": platform.platform(),
        "python": platform.python_version(),
        "libraries": {lib: _version(lib) for lib in _LIBRARIES},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config_hash": None if config is None else config_hash(config),
        "band_file_hash": band_file_hash,
        "h_D": None if h_D is None else h_D.hex(),
        "mem_source": probe.source, "MallocLargeCache": os.environ.get("MallocLargeCache"),
        "threads": torch.get_num_threads(), **fields,
    }


def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def read_records(path: str | os.PathLike[str], record: str | None = None) -> list[dict[str, Any]]:
    """``records.jsonl`` (a file, or the run directory holding it), optionally one type."""
    p = Path(path)
    if p.is_dir():
        p = p / "records.jsonl"
    with open(p) as f:
        rows = [json.loads(line) for line in f]
    return rows if record is None else [r for r in rows if r["record"] == record]


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class MetricsWriter:
    """One run's directory: ``records.jsonl`` and ``residuals/`` (module docstring).

    Opening it truncates an earlier run's records and writes the ``environment`` record; records
    are flushed as they arrive, so a crashed run keeps what it finished. ``close`` writes
    ``run_end``. ``model``, ``corpus`` and ``seed`` go into every ``step`` record.
    """

    def __init__(self, out_dir: str | os.PathLike[str], run: str, *,
                 device: str | torch.device = "cpu", model: str | None = None,
                 corpus: str | None = None, seed: int | None = None,
                 config: Mapping[str, Any] | None = None, band_file_hash: str | None = None,
                 h_D: bytes | None = None, extra: Mapping[str, Any] | None = None) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.device = torch.device(device)
        self.probe = memory_probe(self.device)
        self.run_fields = {"model": model, "corpus": corpus, "seed": seed}
        self._f: IO[str] = open(self.dir / "records.jsonl", "w")  # noqa: SIM115
        self.write([environment_record(run, device=self.device, probe=self.probe, config=config,
                                       band_file_hash=band_file_hash, h_D=h_D,
                                       **self.run_fields, **(extra or {}))])

    def recorder(self, scenario: str, *, poisoning_rate: float | None = None,
                 cheat_step: int | None = None) -> TimeRecorder:
        """A timed-run recorder that writes here.

        ``cheat_step`` is for EQ15's attack runs only, the step the attack starts at. The
        protocol-fault scenarios (``flip``, ``broken-chain`` …) leave it null; their expected
        rejection point is the harness's oracle, not a record field."""
        return TimeRecorder(self.run, scenario, device=self.device, writer=self,
                            step_fields={**self.run_fields, "poisoning_rate": poisoning_rate,
                                         "cheat_step": cheat_step})

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self._f.write(json.dumps(_json_safe(dict(r)), allow_nan=False) + "\n")
        self._f.flush()

    def write_residuals(self, scenario: str, step: int, stats: StepStats, c: Any) -> Path:
        d = self.dir / "residuals" / _SAFE_NAME.sub("_", scenario)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"step_{step:05d}.npz"
        np.savez(path, **_residual_arrays(self.run, scenario, step, stats, c))
        return path

    def close(self) -> None:
        if self._f.closed:
            return
        self.write([{"record": "run_end", "run": self.run,
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "lifetime_maxrss_bytes": lifetime_maxrss_bytes(),
                     "mem_source": self.probe.source}])
        self._f.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
