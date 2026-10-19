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

**Added seconds.** ``section.add(name, seconds)`` (:meth:`Seam.add`) adds host seconds a part
measured itself to a time row named ``name``, with no span; the memory and count recorders
ignore it. The hashing code uses it to report where its time went (the calling thread's read,
batching and waiting, and the worker's native and large-leaf time) as ``P3.commit.hash.<part>``
and ``2.hash.<part>`` rows beside the nested ``P3.commit.collect``, ``.hash``, ``.tree`` and
``.check`` sections and check 2's ``2.hash``. A part that took no time writes no row, so every
time row keeps ``time_s > 0``. The rows come from the timed run only, and a seam that has no
``add`` (a plain callable) gets none.

**Three passes, one axis each.** Measuring one axis disturbs the others, so each has its own run:

- the *timed run* reads only ``time.monotonic_ns`` (:class:`TimeRecorder`). It is the run whose
  steps, verdicts and residuals are recorded;
- the *memory pass* (:class:`MemoryRecorder`, :func:`memory_pass`) probes memory at every
  section entry and exit, nested ones included;
- the *counting pass* (:class:`CountRecorder`, :func:`count_pass`, :func:`counting`) counts
  FLOPs, bytes hashed, hash calls and transcript bytes.

The two extra passes are runs of their own, never timed: the memory pass :data:`PASS_STEPS`
steps, the counting pass :data:`COUNT_STEPS` (one step per configuration, EVALUATION_SPEC §3.4).

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

**Verifier components** are the checks, by slot id, in driver order: per step ``4``, ``7``,
``2``, ``6a``, ``5``, ``6b``, ``2.root``; once per run (step 0) ``0`` (hashing the agreed
``W_0``), ``1``, ``8`` and ``9``. At step 1 the chaining comparison runs in check 7's slot
against check 0's anchor hashes; its row is ``7``, so check 7's series starts at step 1, and
check 0's row is the anchor hashing. Check 2 has two rows (F15a): ``2`` reads, hashes and
validates every leaf, and ``2.root``, last in the step, builds the root from those hashes and
compares it with the claim. ``9`` wraps building the verdict, kept as a row though its cost is negligible, so
every check has one. Check 5 is split (EQ1b) into ``5.glue`` (replay construction and every
``operands(m)``: glue recomputation) and ``5.measure`` (the Freivalds products), both ``sub``
rows. Check 3 has no row of its own: it is the rule that check 5's operands come from committed
leaves, and its cost is ``5.glue``. ``6b.glue`` is the replayed glue gradients.

**Rows.** One per section per step, at a ``level``: ``phase`` (a prover section),
``component`` (a prover component, the sum of its phases, or a check), ``sub`` (a nested
section) or ``total`` (``prover``, ``verifier``, and ``step``, their sum; a total sums its
top-level sections, so loop bookkeeping and fault hooks are not in it). A section entered twice
in a step (``train.load``, ``5.glue``) accumulates and ``calls`` counts the entries. The
harness's own section ``handoff.empty_cache`` (``run_loop``'s cache release between the two
roles, on CUDA only) is a ``component`` row on side ``run``, in no total: deployed, the
prover and the verifier run on separate machines.

**Time (timed run).** Wall-clock seconds from ``monotonic_ns`` and nothing else. On CUDA the
device is synchronized at top-level section boundaries and nested sections are timed with
``torch.cuda.Event`` pairs, read after the enclosing top-level sync; on MPS the device is
synchronized at every boundary. So on CUDA a nested row is device time (the Event pair) while
a top-level row is host time across a synchronized interval; the two clocks are not checked
against each other yet, which the first CUDA run must do. Test-scale times are rehearsal only
(EQ14).

**Markers.** The timed run also writes each section's spans and check 5's stamps
(``PROFILING_PLAN.md`` §5.1) to ``markers.jsonl``, one ``kind`` line per step (``step``) and per
run-level section table (``run``), after a ``header`` line; :func:`read_markers` reads it back. A
step line holds, per section, up to :data:`MAX_SPANS` host spans in microseconds from the step's
first section start (``t0_ns``), the same spans on the device clock on CUDA (``gpu_us``, each
span's CUDA events resolved against the top-level section's anchor event by
:func:`resolve_gpu_us`), a ``rest`` aggregate for the calls beyond them, and on CUDA a top-level
section's allocator counters (``alloc``). Its ``stamps`` are column lists: a stamp is a span
inside a check-5 section (``seam.stamp(name, run=, group=, window=)``) with ids that don't apply
set to -1; ``name`` indexes ``stamp_names``. Nothing synchronizes inside a nested section or a
stamp. ``c5_layout.json`` (:func:`~verification.verifier.checks.check5_layout`, written by
``run_verified``) says what each run, group and window of check 5 holds.

**Memory (memory pass).** ``peak_bytes`` is the whole process's peak during the section, as an
absolute figure; ``start_bytes`` and ``end_bytes`` are the process's figure at the first entry
and the last exit, so ``peak − start`` is what the section added on top of what was live. A
nested section has its own peak: each entry and exit first folds the peak so far into every
open section, then the entry resets it. ``peak_reserved_bytes`` is, on CUDA, the caching
allocator's reserved peak over the same interval (``max_memory_reserved``, folded the same
way): what the allocator held from the driver, live tensors and cached free blocks together, so
``peak_reserved_bytes − peak_bytes`` shows the fragmentation an out-of-memory error reports as
"reserved but unallocated" (the two peaks need not fall at the same moment). It is ``null`` on
every other backend. The backend is named in ``mem_source``:

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
  ``storage`` and ``run_end`` (last, written only when the run completed: a run that raised
  has none, :meth:`MetricsWriter.close`). Standard JSON: a non-finite float is written as the
  string ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, and a missing value as ``null``;
- ``residuals/<scenario>/step_<t>.npz``: one NumPy archive per step with residuals (EQ13
  record 2, EVALUATION_SPEC §4.2), written as the step closes, so a crashed run keeps finished
  steps; a step that stopped before checks 5 and 6 has one with zero rows. Arrays: ``run``,
  ``scenario`` (0-d strings), ``step``; per product ``p_m``, ``p_name``, ``p_cls`` (the κ
  class, the fine class), ``p_spec_class`` (§5.7's class, :func:`spec_class`), ``p_layer``
  (0-based: ``0 … L−1`` for the layer stack, ``L`` for the output layer, :func:`layer_index`),
  ``p_kappa`` and ``p_normalized`` (``[P, k]``, one normalized residual per challenge), and the
  product's scale ``p_q`` (contracted dimension), ``p_norm`` (``‖P‖_F``), ``p_nu`` (``ν``),
  ``p_abs1`` (``‖|P|·1‖``) and ``p_u`` (the term scale ``U``, F18), which C1's realized floor
  and the band unit's ``e_m`` read; per weight tensor ``w_check``
  (``6a``/``6b``), ``w_name`` and ``w_rho`` (``ρ_max``). The scores beside them
  (:func:`rho_arrays`): ``rho_5`` ``[P, k]``, ``rho_kappa`` ``[P]`` and ``rho_6`` ``[W]``, row
  for row with the raw arrays, under the bands named by ``band_source`` (the band file's hash;
  empty and every score NaN while calibrating). NaN and infinity are stored as such. Rows are
  in measuring order, which is canonical order. A rejecting check still measures all its
  tests, after the verdict is fixed (the driver's record pass), so a step rejected at check 5
  has all ``M`` products and one rejected at 6a or 6b every tensor of check 6; the checks
  after the rejecting one never run. A step rejected as malformed has what the checks measured
  before the rejection. A test the step never measured has no row: the arrays hold measured
  tests only, and the step record's ``record_pass`` counts them per family.
  :func:`read_residuals` rebuilds ``StepStats`` equal to the verifier's, the feed for
  calibration (A11).

**Step records** (EVALUATION_SPEC §4.1, :func:`step_record`) carry the run's identity
(``model``, ``corpus``, ``attack``, ``seed``), the cheat's (``bpr_records``, ``bpr_actual``,
``cheat_step``), ``judged`` (:func:`judged_step`), the verdict and checks, and the step's
scores ``rho_5``, ``rho_kappa``, ``rho_6`` and ``n_failing`` (:func:`step_scores`), and its
realized floor ``f_achieved_max`` / ``f_achieved_at`` (:func:`step_floor_fields`, F18). A
rejected step's ``record_pass`` (:func:`record_pass`) says how far the record pass got and how
many tests of each family the step measured; an accepted step's is ``null``.
"""

from __future__ import annotations

import ctypes
import gzip
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import IO, Any, Protocol

import blake3
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

from setup.config import load_config
from setup.machine import machine_facts
from verification.commitment import native
from verification.computation.interface import ProductKind
from verification.verifier.bands import product_class
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF
from verification.verifier.calibration import step_floor
from verification.verifier.checks import (
    DEFAULT_ORDER,
    RECORD_NOT_RUN,
    _layout,
    protocol_id,
)
from verification.verifier.matmul_check.sizing import C_ANTI, bit_budget
from verification.verifier.context import ProductStat, StepStats, TensorStat

__all__ = [
    "P0_PHASES", "TRAIN_PHASES", "PASS_STEPS", "COUNT_STEPS", "COMPONENTS", "RECORD_TYPES",
    "RESERVED_NAMES", "SCORE_FAMILIES",
    "MemoryProbe", "memory_probe", "reserved_peak",
    "TimeRecorder", "MemoryRecorder", "CountRecorder", "Seam", "resolve_gpu_us", "MAX_SPANS",
    "STAMP_NAMES", "read_markers", "c5_layout_file",
    "memory_pass", "count_pass", "counting",
    "derive_capture", "step_record", "record_pass", "judged_step", "in_sample_step",
    "calibration_seed", "spec_class", "layer_count", "layer_index",
    "rho_arrays", "step_scores", "nan_as_inf", "run_name", "config_hash",
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
                "capture", "run_end")
RUN_PREFIX = "run:"
TRACE_DIR = "trace"  # the traced step's files in a run directory (FS23)
# Host spans kept per section per step in markers.jsonl; the calls past them fold into ``rest``.
MAX_SPANS = 64
# The stamp names check 5 writes (checks.py), in the order of a step line's ``stamp_names``.
STAMP_NAMES = ("c5.run", "c5.glue", "c5.serve", "c5.copy", "c5.measure", "c5.judge")
_ALLOC_KEYS = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")
_ALLOC_NAMES = ("retries", "device_alloc", "device_free", "ooms")
_NULL = nullcontext()
# Steps in every run's memory pass. Step 1 carries check 0's anchor and the first-touch
# allocation of the run's buffers; step 2 is the first check-7 step and the first in
# steady-state memory. The plain run's passes match the verified runs' step for step
# (derive_capture pairs them by step number).
PASS_STEPS = 2
# Steps in every run's counting pass: one per configuration (EVALUATION_SPEC §3.4, EV2). The
# protocol is deterministic for a fixed configuration, and steps 1 and 2 of the SmolLM2 run
# count the same FLOPs, hash calls and challenge bytes. They differ in one place: the
# ``hash_in_bytes`` of P3 and of checks 2 and 4, which hash the batch's token records, whose
# length grows with the records' token counts (1,356 bytes more at step 2 than at step 1, at
# test scale, out of 2.6 GB). That is the batch's size, not the protocol's work, so a second
# step adds nothing to the cost result.
COUNT_STEPS = 1


HARNESS_PREFIX = "handoff."  # the harness's own sections (run_loop's HANDOFF_SECTION)


def _side(name: str) -> str:
    if name.startswith(HARNESS_PREFIX):
        return "run"
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

    # A probe may also have ``reserved_peak() -> int | None``, the CUDA caching allocator's
    # reserved peak since the last reset (:func:`reserved_peak`); one without it has none.


def reserved_peak(probe: MemoryProbe) -> int | None:
    """``probe``'s reserved peak since its last reset: the CUDA caching allocator's
    ``max_memory_reserved`` on CUDA, ``None`` on every other backend."""
    get = getattr(probe, "reserved_peak", None)
    return None if get is None else get()


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

    def reserved_peak(self) -> int:
        """Reserved bytes' peak since the last reset (``reset_peak_memory_stats`` resets it
        with the allocated peak); read after ``sample``'s synchronize."""
        return int(torch.cuda.max_memory_reserved(self._dev))


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

    def reserved_peak(self) -> int | None:
        if self._inner is None:
            return None
        try:
            return reserved_peak(self._inner)
        except OSError as e:
            self._fail(e)
            return None


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

    __slots__ = ("nested", "calls", "time", "peak", "rpeak", "start", "end", "seen", "counts",
                 "depth", "c5", "first_ns", "spans", "gpu", "rest", "alloc")

    def __init__(self, nested: bool) -> None:
        self.nested = nested
        # markers (the timed run only): the nesting depth at first entry, whether this is a
        # check-5 section, the first entry's ``monotonic_ns``, the first MAX_SPANS host spans
        # ``(start_ns, end_ns)``, their device spans (``[start_us, end_us]`` or None until
        # resolved; empty off CUDA), ``rest = [calls, host_s, gpu_s, first_ns, last_ns]`` for the
        # calls past them, and ``alloc`` the allocator counters' change (CUDA, top level).
        self.depth = 0
        self.c5 = False
        self.first_ns: int | None = None
        self.spans: list[tuple[int, int]] = []
        self.gpu: list[list[float] | None] = []
        self.rest: list[Any] | None = None
        self.alloc: list[int] | None = None
        self.calls = 0
        self.time = 0.0
        self.peak: int | None = None
        self.rpeak: int | None = None  # the reserved peak (memory pass, CUDA only)
        self.start: int | None = None
        self.end: int | None = None
        self.seen = False
        self.counts = [0] * len(_COUNTERS)


class Seam:
    """The seam a run passes as ``section=``: ``seam(name)`` is ``recorder.section(name)``, and
    ``seam.stamp(name, run, group, window)`` is ``recorder.stamp`` itself (a span inside a
    check-5 section; a no-op context on every recorder but the timed run's), bound at
    construction so a stamp costs no forwarding frame. One per recorder
    (:meth:`_Recorder.seam`), so a run's ``section is`` checks hold."""

    __slots__ = ("_rec", "stamp")

    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec
        self.stamp = rec.stamp

    def __call__(self, name: str) -> Any:
        return self._rec.section(name)

    def add(self, name: str, seconds: float) -> None:
        """Record a duration measured elsewhere as one call of section ``name``, nested when a
        section is open. Only the timed run keeps it, as host wall time with no span (so no
        marker interval); the memory and counting runs ignore it."""
        self._rec.add(name, seconds)


def resolve_gpu_us(anchor_host_us: float, anchor: Any, events: Iterable[Any]) -> list[float]:
    """Each CUDA event's time on the host clock, in microseconds: the anchor's host time plus the
    anchor-to-event time. The anchor and the events must be recorded and complete."""
    return [anchor_host_us + anchor.elapsed_time(e) * 1000.0 for e in events]


def _span(acc: _Acc, t0: int, t1: int, *, gpu: bool = False) -> int:
    """Record one call's host span; its slot in ``acc.spans``, or -1 once they are full (the call
    folds into ``acc.rest``)."""
    if acc.first_ns is None:
        acc.first_ns = t0
    if len(acc.spans) < MAX_SPANS:
        acc.spans.append((t0, t1))
        if gpu:
            acc.gpu.append(None)
        return len(acc.spans) - 1
    r = acc.rest
    if r is None:
        r = acc.rest = [0, 0.0, 0.0, t0, t1]
    r[0] += 1
    r[1] += (t1 - t0) / 1e9
    r[4] = t1
    return -1


def _alloc_counters(device: torch.device) -> list[int]:
    st = torch.cuda.memory_stats(device)
    return [int(st.get(k, 0)) for k in _ALLOC_KEYS]


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
        self._c5_open = 0  # check-5 sections open now (the stamps' gate)
        self._seam: Seam | None = None
        self._verifier: Any = None

    def seam(self) -> Seam:
        """The run's one :class:`Seam`: what to pass where ``section`` is passed."""
        if self._seam is None:
            self._seam = Seam(self)
        return self._seam

    def stamp(self, name: str, run: int = -1, group: int = -1, window: int = -1) -> Any:
        """A span inside a check-5 section; only the timed run records one."""
        return _NULL

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
            acc.depth = self._depth
            acc.c5 = table is self._step and (name == "5" or name.startswith("5."))
        return self._Section(self, acc)

    def add(self, name: str, seconds: float) -> None:
        """A measured duration as a section call; ignored unless the recorder times."""

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
        rows += [self._row(step, "run", n, "component", [a]) for n, a in top.items()
                 if _side(n) == "run"]
        rows += [self._row(step, _side(n), n, "sub", [a]) for n, a in table.items() if a.nested]
        for side, part in (("prover", prover), ("verifier", verifier)):
            if part:
                rows.append(self._row(step, side, side, "total", list(part.values())))
        roles = [a for n, a in top.items() if _side(n) != "run"]  # the harness's are in no total
        if roles:
            rows.append(self._row(step, "run", "step", "total", roles))
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
        rec = self.rec
        rec._depth += 1
        if self.acc.c5:
            rec._c5_open += 1
        self.t0 = time.monotonic_ns()

    def __exit__(self, *exc: Any) -> None:
        t1 = time.monotonic_ns()
        acc, rec = self.acc, self.rec
        acc.time += (t1 - self.t0) / 1e9
        acc.calls += 1
        _span(acc, self.t0, t1)
        rec._depth -= 1
        if acc.c5:
            rec._c5_open -= 1


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
    pair of events, read after the enclosing top-level section's synchronize.

    A top-level section also records an anchor event right after its entry synchronize, which
    the events of every span it encloses are measured from (:func:`resolve_gpu_us`), and reads
    the allocator's counters (no synchronize) at entry and exit."""

    __slots__ = ("rec", "acc", "t0", "top", "e0", "a0")

    def __init__(self, rec: TimeRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        rec = self.rec
        self.top = rec._depth == 0
        rec._depth += 1
        if self.acc.c5:
            rec._c5_open += 1
        if self.top:
            torch.cuda.synchronize(rec._device)
            self.t0 = time.monotonic_ns()
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()
            rec._anchor = (self.t0 / 1000.0, self.e0)
            self.a0 = _alloc_counters(rec._device)
        else:
            self.t0 = time.monotonic_ns()
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()

    def __exit__(self, *exc: Any) -> None:
        rec, acc = self.rec, self.acc
        acc.calls += 1
        rec._depth -= 1
        if acc.c5:
            rec._c5_open -= 1
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        if not self.top:
            t1 = time.monotonic_ns()
            slot = _span(acc, self.t0, t1, gpu=True)
            rec._pending.append((acc, slot, self.e0, e1, True))
            return
        a1 = _alloc_counters(rec._device)
        torch.cuda.synchronize(rec._device)
        t1 = time.monotonic_ns()
        acc.time += (t1 - self.t0) / 1e9
        delta = [b - a for a, b in zip(self.a0, a1, strict=True)]
        acc.alloc = delta if acc.alloc is None else [x + y for x, y in zip(acc.alloc, delta)]
        rec._pending.append((acc, _span(acc, self.t0, t1, gpu=True), self.e0, e1, False))
        rec._resolve()


class _Stamp:
    """One check-5 stamp: host ``monotonic_ns`` at entry and exit and, on CUDA, an event pair
    (no synchronize). Appends to the recorder's step columns as it closes."""

    __slots__ = ("rec", "col", "run", "group", "window", "h0", "e0")

    def __init__(self, rec: TimeRecorder, col: int, run: int, group: int, window: int) -> None:
        self.rec, self.col, self.run, self.group, self.window = rec, col, run, group, window

    def __enter__(self) -> None:
        if self.rec._cuda:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()
        self.h0 = time.monotonic_ns()

    def __exit__(self, *exc: Any) -> None:
        h1 = time.monotonic_ns()
        rec = self.rec
        rows = rec._stamps
        if rec._cuda:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            rec._stamp_pending.append((len(rows), self.e0, e1))
        rows.append((self.col, self.run, self.group, self.window, self.h0, h1))


class _Trace:
    """FS23: the traced step's ``torch.profiler`` session and its ``trace/`` files.

    Starts at the first non-``run`` section of step ``step``, stops at that step's
    :meth:`TimeRecorder.on_step` and writes ``trace/step_NNNNN.json.gz`` and ``trace/meta.json``
    (``PROFILING_PLAN.md`` §5.6). With ``c5 == "slice"`` the profiler's collection is off inside
    check 5 except in the runs of the layout's ``trace_slice``. A profiler failure never reaches
    the run: it tears the profiler down and goes into ``meta.json``'s ``error``."""

    def __init__(self, rec: TimeRecorder, step: int, c5: str, out_dir: Path) -> None:
        self.rec, self.step, self.dir = rec, step, out_dir
        self.c5 = c5  # what was asked; meta's c5_slice is what happened
        self.c5_slice = c5
        self.prof: Any = None
        self.active = False  # the profiler is running now
        self.started = False  # a start was tried (once per run)
        self.traced = False  # the profiler started: the step's rows are marked traced
        self.written = False
        self.error: str | None = None
        self.slice: frozenset[int] | None = None
        self.activities: list[Any] = []
        self.start_ns: int | None = None
        self.stop_ns: int | None = None
        self.collecting = True

    def fail(self, e: BaseException) -> None:
        """Record the first error and tear the profiler down."""
        if self.error is None:
            self.error = f"{type(e).__name__}: {e}"
        self.teardown()

    def teardown(self) -> None:
        prof, self.prof, self.active = self.prof, None, False
        self.rec.__dict__.pop("section", None)
        if prof is not None:
            try:
                prof.stop()
            except Exception:  # noqa: BLE001 - already failing, or never started
                pass

    def start(self) -> None:
        self.started = True
        try:
            self.activities = [torch.profiler.ProfilerActivity.CPU]
            if self.rec._cuda:
                self.activities.append(torch.profiler.ProfilerActivity.CUDA)
            prof = torch.profiler.profile(activities=self.activities, record_shapes=True,
                                          with_flops=True)
            self.start_ns = time.monotonic_ns()
            prof.start()
            self.prof, self.active, self.traced = prof, True, True
            if self.c5 == "slice":
                v = self.rec._verifier
                if hasattr(prof, "toggle_collection_dynamic") and v is not None:
                    self.slice = frozenset(
                        c5_layout_file(v.c, self.rec.run)["trace_slice"])
                else:
                    self.c5_slice = "all"
        except Exception as e:  # noqa: BLE001 - the profiler must not change the run
            self.fail(e)

    def collect(self, on: bool) -> None:
        """Turn collection on or off (check 5's slice); a no-op if it already is."""
        if not self.active or self.collecting == on:
            return
        try:
            self.prof.toggle_collection_dynamic(on, self.activities)
            self.collecting = on
        except Exception as e:  # noqa: BLE001
            self.fail(e)

    def sync(self) -> None:
        if self.rec._cuda:
            torch.cuda.synchronize(self.rec._device)
        elif self.rec._device.type == "mps":
            torch.mps.synchronize()

    def enter(self, label: str) -> Any:
        """Enter ``record_function(label)``; ``None`` if the profiler is down."""
        if not self.active:
            return None
        try:
            rf = torch.profiler.record_function(label)
            rf.__enter__()
            return rf
        except Exception as e:  # noqa: BLE001
            self.fail(e)
            return None

    @staticmethod
    def leave(rf: Any) -> None:
        if rf is not None:
            try:
                rf.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - the profiler is gone, the annotation with it
                pass

    def stop(self) -> None:
        """End of step ``step``: stop the profiler and export; always writes ``meta.json``."""
        if self.active:
            try:
                self.collect(True)
                prof = self.prof
                prof.stop()
                self.stop_ns = time.monotonic_ns()
                self.prof, self.active = None, False
                self.dir.mkdir(parents=True, exist_ok=True)
                raw = self.dir / f"step_{self.step:05d}.json"
                prof.export_chrome_trace(str(raw))
                with open(raw, "rb") as src, gzip.open(f"{raw}.gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                raw.unlink()
            except Exception as e:  # noqa: BLE001
                self.fail(e)
                for f in self.dir.glob(f"step_{self.step:05d}.json*"):
                    f.unlink(missing_ok=True)
        self.teardown()
        self.write_meta()

    def close(self) -> None:
        """The run is over: stop a profiler still running (its step did not end) and write
        ``meta.json`` unless the step's end already did. A step never reached is
        ``"not reached"``."""
        if self.written:
            return
        if self.active:
            if self.error is None:
                self.error = "interrupted"
            self.teardown()
        elif not self.started:
            self.error = "not reached"
        self.write_meta()

    def write_meta(self) -> None:
        self.written = True
        meta = {"version": 1, "step": self.step,
                "slice": None if self.slice is None else sorted(self.slice),
                "c5_slice": self.c5_slice, "torch": torch.__version__,
                "profiler": {"activities": [a.name for a in self.activities] or ["CPU"],
                             "record_shapes": True, "with_flops": True},
                "start_ns": self.start_ns, "stop_ns": self.stop_ns, "error": self.error}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "meta.json").write_text(json.dumps(meta, allow_nan=False) + "\n")
        except OSError as e:
            log.warning("trace meta not written: %s", e)


class _TracedSection:
    """A section inside the traced step: the inner section plus its ``record_function`` and,
    at check 5's section ``5``, the slice's collection toggle."""

    __slots__ = ("inner", "name", "trace", "rf")

    def __init__(self, inner: Any, name: str, trace: _Trace) -> None:
        self.inner, self.name, self.trace = inner, name, trace

    def __enter__(self) -> None:
        t = self.trace
        self.rf = t.enter(self.name)
        if self.name == "5" and t.slice is not None:
            t.collect(False)
        self.inner.__enter__()

    def __exit__(self, *exc: Any) -> None:
        t = self.trace
        try:
            self.inner.__exit__(*exc)
        finally:
            if self.name == "5":
                t.collect(True)
            t.leave(self.rf)


class _TracedStamp:
    """A check-5 stamp inside the traced step: annotated ``name|run=r|group=g``, and a
    ``c5.run`` stamp of a run in the slice turns collection on for its span."""

    __slots__ = ("inner", "label", "trace", "rf", "sliced")

    def __init__(self, inner: Any, name: str, run: int, group: int, trace: _Trace) -> None:
        self.inner, self.trace = inner, trace
        self.label = f"{name}|run={run}|group={group}"
        self.sliced = name == "c5.run" and trace.slice is not None and run in trace.slice

    def __enter__(self) -> None:
        t = self.trace
        if self.sliced:
            t.sync()
            t.collect(True)
        self.rf = t.enter(self.label)
        self.inner.__enter__()

    def __exit__(self, *exc: Any) -> None:
        t = self.trace
        try:
            self.inner.__exit__(*exc)
        finally:
            t.leave(self.rf)
            if self.sliced:
                t.sync()
                t.collect(False)


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
                 step_fields: Mapping[str, Any] | None = None,
                 trace_step: int = 0, trace_c5: str = "slice") -> None:
        super().__init__(run, scenario)
        self._device = torch.device(device)
        if self._device.type == "cuda":
            self._Section = _CudaTimeSection
        elif self._device.type == "mps":
            self._Section = _SyncTimeSection
            self._sync = torch.mps.synchronize
        self._cuda = self._device.type == "cuda"
        # CUDA: (acc, slot, start event, end event, nested) per closed section and (column,
        # start, end) per stamp, read at the top-level exit; ``_anchor`` is (host us, event)
        self._pending: list[tuple[_Acc, int, Any, Any, bool]] = []
        self._stamp_pending: list[tuple[int, Any, Any]] = []
        self._anchor: tuple[float, Any] | None = None
        # one row (column, run, group, window, host start, host end) per stamp; ``_stamp_gpu``
        # maps a row's index to its device span on CUDA
        self._stamps: list[tuple[int, ...]] = []
        self._stamp_gpu: dict[int, tuple[float, float]] = {}
        self._step_stamps: tuple[list[tuple[int, ...]], dict[int, tuple[float, float]]] = ([], {})
        self._stamp_names = list(STAMP_NAMES)
        self._stamp_col = {n: i for i, n in enumerate(self._stamp_names)}
        self._t1 = 0
        self.writer = writer
        self.step_fields = dict(step_fields or {})
        # FS23: the traced step. Off, the hot path is untouched; on, the two seam entry points
        # are swapped for the traced versions, which fall back once the profiler is down.
        self._trace: _Trace | None = None
        self._steps_closed = 0
        self._tracing_step = False  # the step being closed is the traced one
        if trace_step > 0 and writer is not None:
            if trace_c5 not in ("slice", "all"):
                raise ValueError(f"trace_c5 must be 'slice' or 'all', got {trace_c5!r}")
            self._trace = writer.add_trace(_Trace(self, trace_step, trace_c5,
                                                  writer.dir / TRACE_DIR))
            self.section = self._traced_section
            self.stamp = self._traced_stamp

    def add(self, name: str, seconds: float) -> None:
        """A duration measured elsewhere (the hash stats) as one call of section ``name``: the
        same accumulator as :meth:`section`, its time added as given. It has no span: the marker
        line holds it as a ``rest`` aggregate only."""
        if name.startswith(RUN_PREFIX):
            table, name = self._run, name[len(RUN_PREFIX):]
        else:
            table = self._step
        acc = table.get(name)
        if acc is None:
            acc = table[name] = _Acc(self._depth > 0)
            acc.depth = self._depth
        acc.time += seconds
        acc.calls += 1
        now = time.monotonic_ns()
        if acc.first_ns is None:
            acc.first_ns = now
        r = acc.rest
        if r is None:
            acc.rest = [1, seconds, 0.0, now, now]
        else:
            r[0] += 1
            r[1] += seconds
            r[4] = now

    def _traced_section(self, name: str) -> Any:
        sec = super().section(name)
        t = self._trace
        if t is None or name.startswith(RUN_PREFIX):
            return sec
        if not t.active:
            if t.started or self._steps_closed + 1 != t.step:
                return sec
            t.start()
            if not t.active:
                return sec
        return _TracedSection(sec, name, t)

    def _traced_stamp(self, name: str, run: int = -1, group: int = -1, window: int = -1) -> Any:
        st = TimeRecorder.stamp(self, name, run, group, window)
        t = self._trace
        if st is _NULL or t is None or not t.active:
            return st
        return _TracedStamp(st, name, run, group, t)

    def stamp(self, name: str, run: int = -1, group: int = -1, window: int = -1) -> Any:
        """A span inside a check-5 section, recorded as one column entry in ``markers.jsonl``'s
        ``stamps``. Outside check 5 it is a no-op. The ids are positional on the hot path."""
        if not self._c5_open:
            return _NULL
        col = self._stamp_col.get(name)
        if col is None:
            col = self._stamp_col[name] = len(self._stamp_names)
            self._stamp_names.append(name)
        return _Stamp(self, col, run, group, window)

    def _resolve(self) -> None:
        """After a top-level synchronize: the time of every nested section's events, and every
        span's and stamp's device times against the section's anchor."""
        anchor = self._anchor
        for acc, slot, e0, e1, nested in self._pending:
            if nested:
                acc.time += e0.elapsed_time(e1) / 1e3
            if anchor is None:
                continue
            g = resolve_gpu_us(anchor[0], anchor[1], (e0, e1))
            if slot >= 0:
                acc.gpu[slot] = g
            elif acc.rest is not None:
                acc.rest[2] += (g[1] - g[0]) / 1e6
        if anchor is not None:
            for i, e0, e1 in self._stamp_pending:
                self._stamp_gpu[i] = resolve_gpu_us(anchor[0], anchor[1], (e0, e1))
        self._pending.clear()
        self._stamp_pending.clear()

    def on_step(self, rec: Any) -> None:
        self._t1 = time.monotonic_ns()
        self._step_stamps = (self._stamps, self._stamp_gpu)
        self._stamps, self._stamp_gpu = [], {}
        t = self._trace
        self._steps_closed = rec.t
        if t is not None and rec.t == t.step:
            t.stop()
            self._tracing_step = t.traced
        try:
            super().on_step(rec)
        finally:
            self._tracing_step = False

    def _marker_line(self, t: int, table: dict[str, _Acc], rec: Any) -> dict[str, Any]:
        """The step's (or, with ``rec`` None, the run sections') ``markers.jsonl`` line."""
        firsts = [a.first_ns for a in table.values() if a.first_ns is not None]
        t1 = self._t1 if rec is not None else time.monotonic_ns()
        t0 = min(firsts) if firsts else t1
        base = t0 / 1000.0

        def us(ns: int) -> float:
            return round((ns - t0) / 1000.0, 3)

        sections = []
        for name, a in table.items():
            r = a.rest
            sections.append({
                "name": name, "depth": a.depth, "calls": a.calls,
                "host_us": [[us(s0), us(s1)] for s0, s1 in a.spans],
                "gpu_us": ([None if g is None else [round(g[0] - base, 3), round(g[1] - base, 3)]
                            for g in a.gpu] if self._cuda else None),
                "rest": None if r is None else {
                    "calls": r[0], "host_s": r[1], "gpu_s": r[2] if self._cuda else None,
                    "first_us": us(r[3]), "last_us": us(r[4])},
                "alloc": None if a.alloc is None else dict(zip(_ALLOC_NAMES, a.alloc,
                                                               strict=True))})
        line: dict[str, Any] = {"kind": "step" if rec is not None else "run",
                                "scenario": self.scenario, "t0_ns": t0}
        if rec is None:
            line["sections"] = sections
            return line
        rows, gpu = self._step_stamps
        cols = list(zip(*rows, strict=True)) if rows else [()] * 6
        g = [gpu.get(i) for i in range(len(rows))] if self._cuda else None
        line.update({"step": t, "traced": self._tracing_step, "t1_ns": t1, "sections": sections,
                     "stamps": {
                         "name": list(cols[0]), "run": list(cols[1]), "group": list(cols[2]),
                         "window": list(cols[3]), "h0": [us(x) for x in cols[4]],
                         "h1": [us(x) for x in cols[5]],
                         "g0": None if g is None else
                         [None if x is None else round(x[0] - base, 3) for x in g],
                         "g1": None if g is None else
                         [None if x is None else round(x[1] - base, 3) for x in g]},
                     "stamp_names": list(self._stamp_names)})
        return line

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        return {"time_s": sum(a.time for a in accs), "traced": self._tracing_step}

    def _emit(self, t: int, table: dict[str, _Acc], rec: Any) -> None:
        rows = self._rows(t, table)
        if rec is not None and self._verifier is not None:
            rows.append({**step_record(self.run, self.scenario, t, self._verifier,
                                       getattr(rec, "rejection", None), self.step_fields,
                                       transcript_bytes=getattr(rec, "transcript_bytes", None),
                                       leaves=getattr(rec, "leaves", None)),
                         "traced": self._tracing_step})
            cap = getattr(rec, "capture", None)
            if cap is not None:
                rows.append({"record": "capture", "run": self.run, "scenario": self.scenario,
                             "step": t, "traced": self._tracing_step, "host_buffers": None,
                             **cap})
        self.rows += rows
        if self.writer is None:
            return
        self.writer.write(rows)
        self.writer.write_markers(self._marker_line(t, table, rec))
        if rec is not None and self._verifier is not None:
            stats = self._verifier.stats.get(t)
            if stats is not None:  # zero rows when the step stopped before checks 5 and 6
                bands, k, guard = _scoring(self._verifier)
                self.writer.write_residuals(self.scenario, t, stats, self._verifier.c,
                                            bands=bands, k=k, kappa_guard=guard)

    def finish(self, verdict: Any = None) -> None:
        super().finish()
        if self._trace is not None:
            self._trace.close()
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
                fields: Mapping[str, Any] | None = None, *,
                transcript_bytes: int | None = None, leaves: int | None = None
                ) -> dict[str, Any]:
    """EQ13 record 1 for step ``t`` (EVALUATION_SPEC §4.1): the verdict, the first failing
    check, each check's ``pass``, ``fail`` or ``not_run``, whether the step is judged, and the
    step's three scores.

    ``first_failing_check`` is the protocol id a ``Rejection`` carries. ``checks`` is keyed by
    driver slot, except that at step 1 the chain slot is check 0 (``ctx.chain_check_id``), so
    it is keyed ``0``; the cost rows keep the slot id ``7`` at step 1 too. Check 2 has two
    slots, ``2`` and ``2.root`` (F15a), and a rejection as check 2 marks the last of them that
    ran as ``fail``.

    ``fields`` carries the run's identity (``model``, ``corpus``, ``attack``, ``seed``) and the
    cheat's (``bpr_records``, ``bpr_actual``, ``cheat_step``); each is ``None`` when not given.
    ``judged`` is :func:`judged_step`. ``rho_5``, ``rho_kappa``, ``rho_6`` and ``n_failing``
    are :func:`step_scores` over the tests the verifier measured at this step, and
    ``f_achieved_max`` / ``f_achieved_at`` :func:`step_floor_fields`. ``transcript_bytes`` and
    ``leaves`` are the step's committed bytes and leaf count (``leaf_totals``), ``None`` when the
    loop didn't take them; ``traced`` is ``False`` here, and the timed run sets it on the step
    the profiler traced (FS23)."""
    def pid(cid: str) -> str:
        return "0" if t == 1 and cid == "7" else cid

    ran = [pid(c) for c in verifier.timings.get(t, {})]
    failed = rejection is not None and rejection.step == t
    checks = {pid(c): ("pass" if pid(c) in ran else "not_run") for c in DEFAULT_ORDER}
    if failed:
        slots = [c for c in ran if protocol_id(c) == rejection.check_id]
        cid = slots[-1] if slots else (ran[-1] if ran else None)
        if cid is not None:
            checks[cid] = "fail"
    fields = dict(fields or {})
    bands, k, guard = _scoring(verifier)
    stats = getattr(verifier, "stats", {}).get(t) or StepStats()
    out = {"record": "step", "run": run, "scenario": scenario,
           "model": None, "corpus": None, "attack": None, "seed": None,
           "bpr_records": None, "bpr_actual": None, "cheat_step": None, **fields,
           "step": t, "traced": False, "transcript_bytes": transcript_bytes, "leaves": leaves,
           "judged": judged_step(verifier, t, fields.get("cheat_step")),
           "verdict": "reject" if failed else "accept",
           "first_failing_check": rejection.check_id if failed else None,
           "rejection_kind": rejection.kind if failed else None, "checks": checks,
           **step_scores(stats, bands, k=k, kappa_guard=guard),
           **step_floor_fields(verifier, stats),
           "record_pass": record_pass(verifier, t, stats, kappa_guard=guard) if failed else None}
    return out


def step_floor_fields(verifier: Any, stats: StepStats) -> dict[str, Any]:
    """The step's realized floor (F18): ``f_achieved_max``, the largest
    ``f_achieved = c·τ·e_m·2^(N/k)`` over its measured products, each at its own ``e_m``
    (``sizing.product_e_m``), and ``f_achieved_at``, the product's name
    (``calibration.step_floor``). Both are ``None`` while calibrating (no ``τ`` yet), with no
    product scored, or for a stand-in verifier without a computation.

    ``c = C_ANTI``; ``τ`` and ``k`` are the verifier's. ``N`` is the band file's sizing's when it
    has one, else ``bit_budget(λ, n_steps, M, log₂G)`` (placeholder: the spec names the number,
    not where ``T`` comes from; the band file's is the ``T`` the run's ``k`` was sized for)."""
    none = {"f_achieved_max": None, "f_achieved_at": None}
    bands, k, _ = _scoring(verifier)
    c = getattr(verifier, "c", None)
    if bands is None or not k or not hasattr(c, "operand_dtype") or not stats.products:
        return none
    sz = (getattr(bands, "stats", None) or {}).get("sizing")
    if sz:
        N = float(sz["N"])
    else:
        N = bit_budget(LAMBDA, int(getattr(verifier, "n_steps", 1) or 1), c.M, LOG2_G)
    best = step_floor(stats, tau=float(bands.tau), k=int(k), N=N,
                      eps_in=UNIT_ROUNDOFF[c.operand_dtype],
                      eps_acc=UNIT_ROUNDOFF[c.accumulator_dtype], c=C_ANTI)
    if best is None:
        return none
    return {"f_achieved_max": best.f_achieved, "f_achieved_at": best.name}


def record_pass(verifier: Any, t: int, stats: StepStats, *,
                kappa_guard: bool = True) -> dict[str, Any]:
    """The step record's ``record_pass`` for rejected step ``t`` (EVALUATION_SPEC §4.1):
    ``status``, how far the rejecting check's measurement of its remaining tests got
    (``Verifier.record_pass``, one of ``checks.RECORD_STATUSES``; ``"not run"`` when the
    verifier has none), and ``measured``, the number of tests of each family measured at the
    step in all, live and in the pass: ``rho_5`` one per product and challenge, ``rho_kappa``
    one per product (0 with the κ guard off) and ``rho_6`` one per tensor."""
    status = (getattr(verifier, "record_pass", None) or {}).get(t, RECORD_NOT_RUN)
    n_p = len(stats.products)
    return {"status": status,
            "measured": {"rho_5": sum(len(p.normalized) for p in stats.products),
                         "rho_kappa": n_p if kappa_guard else 0, "rho_6": len(stats.tensors)}}


def _scoring(verifier: Any) -> tuple[Any, int | None, bool]:
    """``(bands, k, κ guard)`` that score ``verifier``'s numbers: its bands, or ``None`` while
    it calibrates (no band file yet). ``getattr`` keeps test stand-ins working."""
    bands = None if getattr(verifier, "calibrate", False) else getattr(verifier, "bands", None)
    return bands, getattr(verifier, "k", None), bool(getattr(verifier, "kappa_guard", True))


def judged_step(verifier: Any, t: int, cheat_step: int | None = None) -> bool:
    """Whether step ``t`` counts as a result (EVALUATION_SPEC §2.3): never a calibration step.

    - No step is judged while the verifier calibrates, or under provisional bands (a smoke
      run's, which never judge an evaluation run, P10a).
    - A cheat run (``cheat_step`` = ``t*``) judges ``t ≥ t*``: the steps its detection result
      reads. Its earlier steps are honest steps, which the honest runs report.
    - An honest run at the band file's calibration seed (H1) doesn't judge the calibration
      steps (``calibration_steps``, 1–3): they are in-sample. A band file that records no
      calibration seed raises ``ValueError`` (:func:`calibration_seed`).
    - An honest run at any other seed (H2) doesn't judge step 1 (F8a).
    """
    if getattr(verifier, "calibrate", False):
        return False
    bands = getattr(verifier, "bands", None)
    if bands is None or bands.source == "provisional":
        return False
    if cheat_step is not None:
        return t >= cheat_step
    seed = getattr(verifier, "seed", None)
    if calibration_seed(bands) == seed:
        return not in_sample_step(bands, t, seed)
    return t != 1


def calibration_seed(bands: Any) -> int:
    """The seed of the run the band file was fitted on (its context's ``calibration_seed``,
    which A11's calibration records). A file without it raises ``ValueError``: which steps are
    in-sample, and so which are judged, depends on it, and taking the run's own seed would
    silently count a fitted step as a result."""
    seed = ((getattr(bands, "stats", None) or {}).get("context") or {}).get("calibration_seed")
    if seed is None:
        source = str(getattr(bands, "source", "?"))
        raise ValueError(f"the band file ({source[:16]}…) records no calibration_seed, so its "
                         f"in-sample steps are unknown; refit it with verification.runs."
                         f"calibrate, which records the seed")
    return int(seed)


def in_sample_step(bands: Any, t: int, seed: int | None) -> bool:
    """Whether step ``t`` of an honest run at ``seed`` is one ``bands`` was fitted on: one of
    the band file's ``calibration_steps`` in a run at its ``calibration_seed`` (H1's steps 1–3).
    A file without the seed raises ``ValueError`` (:func:`calibration_seed`)."""
    steps = (getattr(bands, "stats", None) or {}).get("calibration_steps") or ()
    return calibration_seed(bands) == seed and t in set(steps)


# ---- the scores (EVALUATION_SPEC §2.2) -----------------------------------------------------


def rho_arrays(stats: StepStats, bands: Any, *, k: int | None,
               kappa_guard: bool = True) -> dict[str, np.ndarray]:
    """The step's component scores (EVALUATION_SPEC §2.2, §4.2), each a test's left-hand side
    over its right-hand side, so a test rejects exactly when its score is above 1:

    - ``rho_5`` ``[P, k]``: ``normalized / τ``, one per product and challenge;
    - ``rho_kappa`` ``[P]``: ``ν / (κ_max(class)·‖|P|·1‖)``, ``0`` when ``ν = 0`` and ``inf``
      when only the denominator is 0, as the live test ``ν ≤ κ_max·‖|P|·1‖`` judges them;
    - ``rho_6`` ``[W]``: ``ρ_max / τ_W(tensor)``, one per weight tensor checks 6a and 6b
      measured.

    Rows follow ``stats``, so they line up with the raw arrays (``p_m``, ``w_name``). Every
    score is NaN when ``bands`` is ``None`` (a calibrating verifier has none), and every
    ``rho_kappa`` when the κ guard is off (test 1 isn't run). Check 5 compares ``ν`` with the
    product ``κ_max·‖|P|·1‖`` and check 6 ``ρ_max`` with ``τ_W``, so for those two ``ρ > 1``
    exactly when the live test rejects (an IEEE quotient of positive floats is above 1 exactly
    when the numerator is larger). Check 5's residual test compares the raw residual with
    ``τ·unit`` while ``ρ_5`` divides the stored ``normalized = residual/unit`` by ``τ``, so the
    two can disagree by rounding when a residual lands on ``τ`` to within an ulp.
    """
    ps, ts = stats.products, stats.tensors
    width = len(ps[0].normalized) if ps else (k or 0)
    norm = np.array([p.normalized for p in ps], dtype=np.float64).reshape(len(ps), width)
    if bands is None:
        return {"rho_5": np.full_like(norm, np.nan), "rho_kappa": np.full(len(ps), np.nan),
                "rho_6": np.full(len(ts), np.nan)}
    rho_5 = norm / bands.tau
    if kappa_guard:
        nu = np.array([p.nu for p in ps], dtype=np.float64)
        ceiling = np.array([bands.kappa_for(p.cls) * p.p_abs1 for p in ps], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_kappa = np.where(nu == 0, 0.0, nu / ceiling)
    else:
        rho_kappa = np.full(len(ps), np.nan)
    rho_6 = np.array([s.rho_max / bands.tau_w_for(s.weight) for s in ts], dtype=np.float64)
    return {"rho_5": rho_5, "rho_kappa": rho_kappa.astype(np.float64), "rho_6": rho_6}


SCORE_FAMILIES = ("rho_5", "rho_kappa", "rho_6")


def step_scores(stats: StepStats, bands: Any, *, k: int | None,
                kappa_guard: bool = True) -> dict[str, Any]:
    """The step record's scores (EVALUATION_SPEC §2.2, §4.1): per family the largest component
    score (``rho_5``, ``rho_kappa``, ``rho_6``) and ``n_failing``, a dict of the number of
    component tests above 1, both over what the verifier measured at the step.

    Only measured tests count. A test the step never measured (a glue tensor of 6b after a
    check-5 rejection, every test after a rejection before check 5) has no row in the arrays,
    so it is neither a score nor failing; the step record's ``record_pass`` gives how many of
    each family were measured. A family is ``None``, with its ``n_failing`` entry, when no test
    of it was scored: no bands (calibration), the κ guard off (``rho_kappa``), no product
    measured (``rho_5``, ``rho_kappa``: the step stopped before check 5) or no tensor
    (``rho_6``). A non-finite score counts as failing, as the live checks reject it, and a NaN
    score counts as ``+inf`` in the maximum (:func:`nan_as_inf`), so it can't hide the rest."""
    rho = rho_arrays(stats, bands, k=k, kappa_guard=kappa_guard)
    scored = {"rho_5": bands is not None and bool(stats.products),
              "rho_kappa": bands is not None and kappa_guard and bool(stats.products),
              "rho_6": bands is not None and bool(stats.tensors)}
    out: dict[str, Any] = {}
    failing: dict[str, int | None] = {}
    for fam in SCORE_FAMILIES:
        x = nan_as_inf(rho[fam].ravel())
        out[fam] = float(np.max(x)) if scored[fam] else None
        failing[fam] = int(np.count_nonzero(~(x <= 1.0))) if scored[fam] else None
    out["n_failing"] = failing
    return out


def nan_as_inf(x: np.ndarray) -> np.ndarray:
    """``x`` with every NaN score as ``+inf``: a NaN test fails, as the live checks reject it,
    and taken as the largest score it can't hide the step's maximum (``np.max`` would return
    NaN, ``np.argmax`` the first NaN). The stored arrays keep NaN as such."""
    return np.where(np.isnan(x), np.inf, x)


# ---- memory: the memory pass -------------------------------------------------------------


def _max(a: int | None, b: int | None) -> int | None:
    return a if b is None or (a is not None and a >= b) else b


class _MemSection:
    __slots__ = ("rec", "acc", "peak", "rpeak", "start", "name")

    def __init__(self, rec: MemoryRecorder, acc: _Acc) -> None:
        self.rec, self.acc, self.name = rec, acc, ""

    def __enter__(self) -> None:
        rec, acc = self.rec, self.acc
        rec._fold()  # the open sections keep the peak so far before it is reset
        rec.probe.reset()
        self.peak, self.start = rec.probe.sample()
        self.rpeak = reserved_peak(rec.probe)
        if not acc.seen:
            acc.start, acc.seen = self.start, True
        if rec._depth == 0:
            rec._inner = {}
        rec._open.append(self)
        rec._depth += 1

    def __exit__(self, *exc: Any) -> None:
        rec, acc = self.rec, self.acc
        cur = rec._fold()
        rec._open.pop()
        rec._depth -= 1
        acc.calls += 1
        acc.end = cur
        acc.peak, acc.rpeak = _max(acc.peak, self.peak), _max(acc.rpeak, self.rpeak)
        if rec.echo is None:
            return
        if rec._depth > 0:
            was, rwas = rec._inner.get(self.name, (None, None))
            rec._inner[self.name] = (_max(was, self.peak), _max(rwas, self.rpeak))
        else:
            rec._echo_section(self, cur, exc[0])


def _gb(x: int | None) -> str:
    return "?" if x is None else f"{x / 2**30:.2f}"


class MemoryRecorder(_Recorder):
    """The memory pass: the process's peak per section, nested sections included.

    With ``echo``, each top-level section also prints one line as it closes: its own peak
    (and on CUDA its reserved peak), its figure at entry and exit, the peaks of the sections
    nested in it (each with its reserved peak on CUDA), and the exception it closed on, if
    any. A run that dies mid-step keeps the lines of the sections it got
    through, the one that raised included, where the rows (written as a step closes) are
    lost."""

    record = "memory"
    _Section = _MemSection

    def __init__(self, run: str, scenario: str, probe: MemoryProbe, *,
                 echo: Callable[[str], None] | None = None) -> None:
        super().__init__(run, scenario)
        self.probe = probe
        self.echo = echo
        self._open: list[_MemSection] = []
        # nested (peak, reserved peak) inside the open top section
        self._inner: dict[str, tuple[int | None, int | None]] = {}
        self._t = 1  # the step the open sections belong to

    def section(self, name: str) -> Any:
        s = super().section(name)
        s.name = name
        return s

    def on_step(self, rec: Any) -> None:
        super().on_step(rec)
        self._t = rec.t + 1

    def _echo_section(self, s: _MemSection, end: int | None, exc: type | None) -> None:
        name = s.name
        if name.startswith(RUN_PREFIX):
            where, name = "run", name[len(RUN_PREFIX):]
        else:
            where = f"step {self._t}"
        if _side(name) == "verifier":
            name = f"check {name}"
        inner = ", ".join(f"{n} {_gb(p)}" + ("" if r is None else f" (reserved {_gb(r)})")
                          for n, (p, r) in self._inner.items())
        line = (f"   {where}, {name}: peak {_gb(s.peak)} GB"
                + ("" if s.rpeak is None else f", reserved {_gb(s.rpeak)} GB")
                + f" (start {_gb(s.start)}, end {_gb(end)} GB"
                + (f"; nested: {inner} GB" if inner else "") + ")"
                + ("" if exc is None else f"; raised {exc.__name__}"))
        try:
            self.echo(line)  # type: ignore[misc]
        except OSError:  # a full disk under the log mustn't end the run
            pass

    def _fold(self) -> int | None:
        peak, cur = self.probe.sample()
        rpeak = reserved_peak(self.probe)
        for s in self._open:
            s.peak, s.rpeak = _max(s.peak, peak), _max(s.rpeak, rpeak)
        return cur

    def _values(self, accs: list[_Acc]) -> dict[str, Any]:
        peaks = [a.peak for a in accs if a.peak is not None]
        rpeaks = [a.rpeak for a in accs if a.rpeak is not None]
        return {"peak_bytes": max(peaks) if peaks else None,
                "peak_reserved_bytes": max(rpeaks) if rpeaks else None,
                "start_bytes": accs[0].start, "end_bytes": accs[-1].end,
                "mem_source": self.probe.source}


def memory_pass(run: str, scenario: str, fn: Callable[[MemoryRecorder], Any],
                probe: MemoryProbe, *,
                echo: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Run ``fn(recorder)`` once with memory probed at every section; return its rows.

    ``fn`` wires the recorder as a timed run wires a :class:`TimeRecorder`. Its times are never
    reported. ``echo`` prints each top-level section's peak as it closes
    (:class:`MemoryRecorder`).
    """
    rec = MemoryRecorder(run, scenario, probe, echo=echo)
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
        # The native leaf path hashes in C, out of reach of these counters: run the Python one.
        with native.disabled(), _FlopTally() as flops:
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
                    "calls": label["calls"], "time_s": dt + label["time_s"], "traced": False})
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
                    "calls": 1, "peak_bytes": grow, "peak_reserved_bytes": None,
                    "start_bytes": None, "end_bytes": None,
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


_WEIGHT_FREE_FORWARD = {"S": "attention score", "O": "attention value"}
_KIND_CLASSES = {ProductKind.INPUT_GRAD: "input gradient",
                 ProductKind.WEIGHT_GRAD: "weight gradient",
                 # EVALUATION_SPEC §5.7 names no class for the gradients of the weight-free
                 # products (δA, δV, δQ̃, δK̃): an open question, and this name a placeholder.
                 ProductKind.OPERAND_GRAD: "operand gradient"}


def spec_class(c: Any, spec: Any) -> str:
    """A product's class in EVALUATION_SPEC §5.7: ``forward``, ``input gradient``, ``weight
    gradient``, ``attention score`` or ``attention value``, and ``operand gradient`` (a
    placeholder) for the gradients of the weight-free products.

    The forward products with a weight are ``forward``; the weight-free ones are the attention
    products, named by their κ class (``S``, ``O``). Any other weight-free forward class raises
    ``ValueError``: §5.7 has no class for it."""
    if spec.kind is ProductKind.FORWARD:
        if spec.weight is not None:
            return "forward"
        cls = product_class(c, spec)
        if cls not in _WEIGHT_FREE_FORWARD:
            raise ValueError(f"{spec.name}: weight-free forward class {cls!r} has no "
                             f"EVALUATION_SPEC §5.7 class")
        return _WEIGHT_FREE_FORWARD[cls]
    return _KIND_CLASSES[spec.kind]


def layer_count(c: Any) -> int:
    """``L``, the computation's layer count: its largest 1-based ``ProductSpec.layer`` (0 when
    no product is in a layer)."""
    return max((s.layer for s in c.products if s.layer is not None), default=0)


def layer_index(spec: Any, n_layers: int) -> int:
    """A product's ``p_layer`` (EVALUATION_SPEC §4.2, §5.4): its layer numbered from 0, so
    ``0 … L−1`` for the layer stack (``ProductSpec.layer`` is 1-based), and ``L`` for a product
    outside it, the output layer's (on the decoder ``Lambda``, ``dF`` and the head's weight
    gradient)."""
    return n_layers if spec.layer is None else int(spec.layer) - 1


def c5_layout_file(c: Any, run: str) -> dict[str, Any]:
    """``c5_layout.json``'s content (``PROFILING_PLAN.md`` §5.2): :func:`check5_layout`'s entries
    with each group's ``layer`` (:func:`layer_index` of its first product) and ``class``
    (``product_class``), the counts, and ``trace_slice``, the runs a trace covers: the first run
    of each class (over every group, not only a run's first) and every run whose id is a
    multiple of 20, each once, sorted. ``n_windows`` is the number of judge calls."""
    entries, n_windows = _layout(c)
    by_m = {spec.m: spec for spec in c.products}
    n_layers = layer_count(c)
    seen: set[str] = set()
    trace: set[int] = set()
    for e in entries:
        spec = by_m[e["first"]]
        e["layer"] = layer_index(spec, n_layers)
        e["class"] = product_class(c, spec)
        if e["class"] not in seen:
            seen.add(e["class"])
            trace.add(e["run"])
        if e["run"] % 20 == 0:
            trace.add(e["run"])
    n_runs = entries[-1]["run"] + 1 if entries else 0
    return {"version": 1, "run": run, "n_runs": n_runs,
            "n_groups": len(entries), "n_windows": n_windows,
            "entries": entries, "trace_slice": sorted(trace)}


def _residual_arrays(run: str, scenario: str, step: int, stats: StepStats, c: Any, *,
                     bands: Any = None, k: int | None = None,
                     kappa_guard: bool = True) -> dict[str, np.ndarray]:
    ps = stats.products
    width = len(ps[0].normalized) if ps else (k or 0)
    layers, classes = [], []
    n_layers = 0 if c is None else layer_count(c)
    for p in ps:
        spec = c.product(p.m) if c is not None else None
        layers.append(-1 if spec is None else layer_index(spec, n_layers))
        classes.append("" if spec is None else spec_class(c, spec))
    rho = rho_arrays(stats, bands, k=k, kappa_guard=kappa_guard)
    return {
        "run": np.array(run), "scenario": np.array(scenario), "step": np.array(step),
        "p_m": np.array([p.m for p in ps], dtype=np.int32),
        "p_name": np.array([p.name for p in ps], dtype=str),
        "p_cls": np.array([p.cls for p in ps], dtype=str),
        "p_spec_class": np.array(classes, dtype=str),
        "p_layer": np.array(layers, dtype=np.int32),
        "p_kappa": np.array([p.kappa for p in ps], dtype=np.float64),
        "p_q": np.array([p.q for p in ps], dtype=np.int64),
        "p_norm": np.array([p.p_norm for p in ps], dtype=np.float64),
        "p_nu": np.array([p.nu for p in ps], dtype=np.float64),
        "p_abs1": np.array([p.p_abs1 for p in ps], dtype=np.float64),
        "p_u": np.array([p.u for p in ps], dtype=np.float64),
        "p_normalized": np.array([p.normalized for p in ps],
                                 dtype=np.float64).reshape(len(ps), width),
        "w_check": np.array([s.check_id for s in stats.tensors], dtype=str),
        "w_name": np.array([s.weight for s in stats.tensors], dtype=str),
        "w_rho": np.array([s.rho_max for s in stats.tensors], dtype=np.float64),
        "band_source": np.array("" if bands is None else str(bands.source)),
        **rho,
    }


def _stats_from(z: Mapping[str, np.ndarray]) -> StepStats:
    st = StepStats()
    # Archives written before A11 have no product scales; those read back as unknown.
    # Archives written before F18 have no term scale U; it reads back as unknown (NaN).
    scales = all(key in z for key in ("p_q", "p_norm", "p_nu", "p_abs1"))
    has_u = "p_u" in z
    for i in range(len(z["p_m"])):
        extra = ({"q": int(z["p_q"][i]), "p_norm": float(z["p_norm"][i]),
                  "nu": float(z["p_nu"][i]), "p_abs1": float(z["p_abs1"][i])} if scales else {})
        if has_u:
            extra["u"] = float(z["p_u"][i])
        st.products.append(ProductStat(int(z["p_m"][i]), str(z["p_name"][i]), str(z["p_cls"][i]),
                                       float(z["p_kappa"][i]),
                                       tuple(float(x) for x in z["p_normalized"][i]), **extra))
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


# The CUDA caching allocator's settings variables. torch 2.9 reads PYTORCH_ALLOC_CONF and
# prefers it over PYTORCH_CUDA_ALLOC_CONF, its deprecated older name.
ALLOC_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def allocator_settings(release_cache: bool | None = None) -> dict[str, Any]:
    """What this process's CUDA allocator runs with: each of :data:`ALLOC_VARS` (``None`` when
    unset) and the run's ``release_cache``, the cache release between prover and verifier
    (``None`` reads ``VERIF_RELEASE_CACHE``)."""
    return {**{v: os.environ.get(v) for v in ALLOC_VARS},
            "release_cache": load_config().release_cache if release_cache is None
            else release_cache}


def _dtype_name(dtype: Any) -> str | None:
    """``"bfloat16"`` for ``torch.bfloat16`` or ``"torch.bfloat16"``; ``None`` for ``None``."""
    return None if dtype is None else str(dtype).removeprefix("torch.")


def environment_record(run: str, *, device: str | torch.device, probe: MemoryProbe,
                       config: Mapping[str, Any] | None = None, band_file_hash: str | None = None,
                       h_D: bytes | None = None, release_cache: bool | None = None,
                       **fields: Any) -> dict[str, Any]:
    """EQ13 record 5: hardware, library versions, git commit and the run's input hashes, and
    the allocator's settings (:func:`allocator_settings`, ``release_cache`` as there)."""
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
        "native": native.describe(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config_hash": None if config is None else config_hash(config),
        "band_file_hash": band_file_hash,
        "h_D": None if h_D is None else h_D.hex(),
        "mem_source": probe.source, "MallocLargeCache": os.environ.get("MallocLargeCache"),
        **allocator_settings(release_cache),
        "machine": machine_facts(),
        "compute_dtype": _dtype_name(None if config is None else config.get("compute_dtype")),
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


_MARKER_KINDS = ("header", "step", "run")
_HEADER_KEYS = ("run", "pid", "device", "mono_ns", "wall_unix_s")


def read_markers(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """``markers.jsonl`` (a file, or the run directory holding it): the header line, then the
    ``step`` and ``run`` lines, in file order. Raises ``ValueError`` for a missing or malformed
    header (version 1, :data:`_HEADER_KEYS`) and for any line with no ``kind`` or an unknown
    one."""
    p = Path(path)
    if p.is_dir():
        p = p / "markers.jsonl"
    with open(p) as f:
        lines = [json.loads(line) for line in f]
    for i, line in enumerate(lines, start=1):
        if not isinstance(line, dict) or line.get("kind") not in _MARKER_KINDS:
            raise ValueError(f"{p}:{i}: a marker line needs a kind of {_MARKER_KINDS}")
    head = lines[0] if lines else {}
    if (head.get("kind") != "header" or head.get("version") != 1
            or any(k not in head for k in _HEADER_KEYS)):
        raise ValueError(f"{p}: the first line must be a version-1 header with {_HEADER_KEYS}")
    if any(line["kind"] == "header" for line in lines[1:]):
        raise ValueError(f"{p}: more than one header line")
    return lines


def read_records(path: str | os.PathLike[str], record: str | None = None) -> list[dict[str, Any]]:
    """``records.jsonl`` (a file, or the run directory holding it), optionally one type."""
    p = Path(path)
    if p.is_dir():
        p = p / "records.jsonl"
    with open(p) as f:
        rows = [json.loads(line) for line in f]
    return rows if record is None else [r for r in rows if r["record"] == record]


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# Top-level names under ``$VERIF_OUTPUT_DIR`` that hold shared files, never a run (EVALUATION_SPEC
# §4.6): the data files, and the evaluation's results.
RESERVED_NAMES = ("data", "evaluation")


def run_name(scenario: str, model: str, attack: str, bpr_records: int, seed: int) -> str:
    """A cell's unique run name, ``<scenario>-<model>-<attack>-bpr<n>-s<seed>`` (EV2): the
    model as its repository's last part (``SmolLM2-135M-Instruct``), ``n`` the poisoned records
    in the cheat batch (0 for an honest run). Characters outside ``[A-Za-z0-9._-]`` become
    ``_``."""
    name = f"{scenario}-{model.rsplit('/', 1)[-1]}-{attack}-bpr{int(bpr_records)}-s{int(seed)}"
    return _SAFE_NAME.sub("_", name)


class MetricsWriter:
    """One run's directory: ``records.jsonl`` and ``residuals/`` (module docstring).

    Opening it truncates an earlier run's records and writes the ``environment`` record; records
    are flushed as they arrive, so a crashed run keeps what it finished. ``close`` writes
    ``run_end`` only for a run that finished (no exception), so ``run_end`` marks a completed
    run (:meth:`close`). ``model``, ``corpus``, ``attack`` (``VERIF_ATTACK``, the BackdoorLLM task;
    ``None`` for a run on other data) and ``seed`` go into the ``environment`` record and every
    ``step`` record. A run named ``data`` or ``evaluation``, or a directory of either name, is
    refused (:data:`RESERVED_NAMES`) in any letter case: on a case-insensitive file system
    ``Data`` is the same directory as ``data``.
    """

    def __init__(self, out_dir: str | os.PathLike[str], run: str, *,
                 device: str | torch.device = "cpu", model: str | None = None,
                 corpus: str | None = None, attack: str | None = None, seed: int | None = None,
                 config: Mapping[str, Any] | None = None, band_file_hash: str | None = None,
                 h_D: bytes | None = None, extra: Mapping[str, Any] | None = None) -> None:
        self.dir = Path(out_dir)
        for name in (run, self.dir.name):
            if name.casefold() in RESERVED_NAMES:
                raise ValueError(f"run name or directory {name!r} is reserved "
                                 f"(EVALUATION_SPEC §4.6: {', '.join(RESERVED_NAMES)})")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.device = torch.device(device)
        self.probe = memory_probe(self.device)
        self.run_fields = {"model": model, "corpus": corpus, "attack": attack, "seed": seed}
        self._f: IO[str] = open(self.dir / "records.jsonl", "w")  # noqa: SIM115
        self._m: IO[str] | None = None  # markers.jsonl, opened by the first write_markers
        self._layout_written = False
        self._traces: list[_Trace] = []
        self.write([environment_record(run, device=self.device, probe=self.probe, config=config,
                                       band_file_hash=band_file_hash, h_D=h_D,
                                       **self.run_fields, **(extra or {}))])

    def recorder(self, scenario: str, *, bpr_records: int | None = None,
                 bpr_actual: float | None = None,
                 cheat_step: int | None = None, trace_step: int = 0,
                 trace_c5: str = "slice") -> TimeRecorder:
        """A timed-run recorder that writes here.

        ``trace_step`` > 0 profiles that step with ``torch.profiler`` into ``trace/``
        (``trace_c5``: ``"slice"`` or ``"all"``; :class:`_Trace`).

        ``bpr_records`` and ``bpr_actual`` are the cheat batch's poisoned records and their
        share of the batch (EVALUATION_SPEC §4.1), ``0`` and ``0.0`` for an honest run.
        ``cheat_step`` is ``t*``, for EQ15's attack runs only. The protocol-fault scenarios
        (``flip``, ``broken-chain`` …) leave all three null; their expected rejection point is
        the harness's oracle, not a record field."""
        return TimeRecorder(self.run, scenario, device=self.device, writer=self,
                            step_fields={**self.run_fields, "bpr_records": bpr_records,
                                         "bpr_actual": bpr_actual, "cheat_step": cheat_step},
                            trace_step=trace_step, trace_c5=trace_c5)

    def add_trace(self, trace: _Trace) -> _Trace:
        """Register a traced step's session, so :meth:`close` ends it if the run didn't."""
        self._traces.append(trace)
        return trace

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self._f.write(json.dumps(_json_safe(dict(r)), allow_nan=False) + "\n")
        self._f.flush()

    def write_markers(self, obj: Mapping[str, Any]) -> None:
        """One line of ``markers.jsonl``. The first call opens the file (truncating an earlier
        run's) and writes its ``header``: the run, the process, the device, and the monotonic and
        wall clocks read together."""
        if self._m is None:
            self._m = open(self.dir / "markers.jsonl", "w")  # noqa: SIM115
            mono, wall = time.monotonic_ns(), time.time()
            head = {"kind": "header", "version": 1, "run": self.run, "pid": os.getpid(),
                    "device": self.device.type, "mono_ns": mono, "wall_unix_s": wall}
            self._m.write(json.dumps(head, allow_nan=False) + "\n")
        self._m.write(json.dumps(_json_safe(dict(obj)), allow_nan=False) + "\n")
        self._m.flush()

    def write_c5_layout(self, c: Any) -> Path:
        """``c5_layout.json`` (:func:`c5_layout_file`), once per run directory: a second call
        rewrites nothing."""
        path = self.dir / "c5_layout.json"
        if not self._layout_written:
            path.write_text(json.dumps(c5_layout_file(c, self.run), allow_nan=False) + "\n")
            self._layout_written = True
        return path

    def write_residuals(self, scenario: str, step: int, stats: StepStats, c: Any, *,
                        bands: Any = None, k: int | None = None,
                        kappa_guard: bool = True) -> Path:
        """The step's residual arrays, with its scores under ``bands`` (:func:`rho_arrays`;
        NaN without bands)."""
        d = self.dir / "residuals" / _SAFE_NAME.sub("_", scenario)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"step_{step:05d}.npz"
        np.savez(path, **_residual_arrays(self.run, scenario, step, stats, c, bands=bands, k=k,
                                          kappa_guard=kappa_guard))
        return path

    def close(self, completed: bool | None = None) -> None:
        """Close the file, writing ``run_end`` only if the run completed.

        A run that raised (a crash, a failed assertion, ``KeyboardInterrupt``) ends without
        ``run_end``, so a resumer that skips a directory holding ``run_end`` runs it again; its
        finished records stay. A run that finished with a failing verdict did complete and gets
        ``run_end``. ``completed=None`` reads it from ``sys.exception()``: true unless an
        exception is propagating through the caller, so every run's ``try: … finally:
        mw.close()`` gets it right. Don't call it bare inside an ``except`` block that handles
        an error and goes on: the handled exception would count as a crash. A second call does
        nothing."""
        if self._f.closed:
            return
        for trace in self._traces:
            trace.close()
        if self._m is not None:
            self._m.close()
        if completed is None:
            completed = sys.exception() is None
        if completed:
            self.write([{"record": "run_end", "run": self.run,
                         "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "lifetime_maxrss_bytes": lifetime_maxrss_bytes(),
                         "mem_source": self.probe.source}])
        self._f.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: Any) -> None:
        self.close(completed=exc_type is None)
