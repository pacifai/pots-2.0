"""B6: per-component cost and residual metrics for a verified run.

A run is observed through one seam, ``section(name) -> context manager``, which
:func:`~verification.prover.step.prove_step`, :func:`~verification.runs.loop.run_loop` and
:class:`~verification.verifier.driver.Verifier` accept as ``section=``. It is ``None`` by default,
and then nothing here runs: the protocol code is the same with metrics on or off, and a run's
weights and decisions are bit-identical either way (tested).

**What the prover components are.** The design docs name P0–P5 without defining them
(IMPLEMENTATION_PLAN B6/B7, SETUP_TASKS C1). B6 fixes them as below. B7's plain-training
baseline writes the same ``P0`` rows with capture and every protocol part off, so the verified
run's ``P0`` minus B7's ``P0`` is the cost of capture.

| component | phases (section names) | what runs |
|---|---|---|
| P0, training | ``P0.load``, ``P0.forward``, ``P0.backward``, ``P0.update`` | load ``W_t``, zero grads, build the optimizer; forward and backward under capture; ``opt.step()`` and the final ``zero_grad`` |
| P1, labeling | ``P1.label`` | map captured matmuls to product slots, the ``M`` count check, invariant 6's ``assert_unmodified``, release the captured operands |
| P2, leaf snapshots | ``P2.w_t``, ``P2.w_next`` | copy ``W_t`` and ``W_{t+1}`` out of the model as leaves |
| P3, commitment | ``P3.commit`` | hash every leaf, build the Merkle tree, re-check invariant 6 (``InMemoryStore.from_step``) |
| P4, audit paths | ``P4.paths`` per step; ``P4.tree`` once per run (step 0) | the ``h_D`` paths of the batch; building the prover's copy of ``h_D``'s tree |
| P5, hand-off and release | ``P5.release`` | drop the step's store. At full scale this is where the on-disk store (A14) and off-device reporting (§8.A.7) land |

Fault hooks (``ProverFault.emit`` and friends) are harness cost and sit outside every section.

**Verifier components** are the checks, named by id, in driver order: per step ``4``, then
``0`` at step 1 or ``7`` after it (check 0 runs in check 7's slot), ``2``, ``6a``, ``5``, ``6b``;
once per run (step 0) ``0`` (hashing the agreed ``W_0``), ``1``, ``8`` and ``9``. Check 3 is
not a pass of its own: it is check 5 rebuilding operands from committed leaves, measured as the
nested section ``5.glue`` (replay construction and every ``operands(m)``). ``5.measure`` is
the Freivalds arithmetic and ``6b.glue`` the replayed glue gradients.

**Rows** (``costs.csv``): one per section per step, at ``level``:

- ``phase``: a prover phase above;
- ``component``: a prover component P0–P5 (the sum of its phases) or a verifier check;
- ``sub``: a section nested inside a check (time only, see below);
- ``total``: ``prover``, ``verifier`` and ``step`` (their sum). A total's time is the sum of its
  top-level sections, which excludes the loop's own bookkeeping and the fault hooks.

A phase entered twice in a step (``P0.load``, ``P0.update``) accumulates: ``calls`` counts the
entries, ``time_s`` sums them, ``peak_bytes`` is the largest peak, ``start_bytes`` is the first
entry's and ``end_bytes`` the last exit's.

**What the memory figure measures.** ``peak_bytes`` is the *whole process's* peak during the
section, as an absolute figure, not the section's own growth; ``start_bytes`` and
``end_bytes`` are the process's figure at entry and exit, so ``peak − start`` is what the section
added on top of what was already live. Which figure depends on the backend, named in
``mem_source``:

- ``darwin_phys_footprint`` (macOS): ``phys_footprint``, the figure Activity Monitor calls
  "Memory": resident, compressed and swapped dirty pages the process owns. The peak is the
  kernel's interval maximum (``proc_pid_rusage`` ``RUSAGE_INFO_V4``), reset at each section's
  entry with ``proc_rlimit_control(RLIMIT_FOOTPRINT_INTERVAL)``. macOS ``malloc`` keeps freed
  large blocks, so a freed buffer still counts until reused, and a phase's peak can hide under
  memory an earlier phase freed. Export ``MallocLargeCache=0`` before Python starts, for every
  run being compared, to have freed memory returned and per-phase peaks show.
- ``linux_rss_hwm`` (Linux): ``VmRSS`` and ``VmHWM`` from ``/proc/self/status``, with the
  high-water mark reset by writing ``5`` to ``/proc/self/clear_refs``.
- ``cuda_allocated`` (``VERIF_DEVICE=cuda…``): ``torch.cuda.memory_allocated`` and
  ``max_memory_allocated`` on the current device, the bytes in live tensors, without the
  caching allocator's reserve; the device is synchronized at each section boundary, and host
  memory is not in the rows (``meta.json`` keeps the process's lifetime ``ru_maxrss``).
- ``ru_maxrss`` (anything else): the process's lifetime high-water mark, which can't be reset
  (bytes on macOS, kB on Linux, converted to bytes here). ``start_bytes`` and ``end_bytes`` are
  left empty.

Memory is read only at top-level sections. The peak is process-wide and has one reset, so a
nested section can't have its own: ``sub`` rows record time only, with the memory columns at
``0`` and ``mem_source`` ``nested``.

**Counts** (``counts.csv``) come from a separate pass (:func:`count_pass`), never from a
timed run: torch's FLOP formulas (the ones ``FlopCounterMode`` applies) over every op, and BLAKE3
wrapped in :mod:`verification.commitment.merkle` and :mod:`...matmul_check.challenges` to count
bytes hashed in and out (out is check 5's challenge stream). Each section's row is the
difference across it, so nested rows are included in their parent.

**Residuals** (``residuals.jsonl``): every step's ``verifier.stats[t]``, one JSON object per
product (check 5) and per tensor (checks 6a, 6b). :func:`read_residuals` reads them back as
``StepStats`` equal to the verifier's, the feed for A11's calibration.
"""

from __future__ import annotations

import csv
import ctypes
import importlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import IO, Any, Protocol

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

from verification.verifier.context import ProductStat, StepStats, TensorStat

__all__ = [
    "PROVER_COMPONENTS", "COST_FIELDS", "COUNT_FIELDS",
    "MemoryProbe", "memory_probe",
    "CostRow", "CountRow", "CostRecorder", "CountRecorder", "count_pass", "counting",
    "MetricsWriter", "residual_rows", "read_residuals", "read_costs", "read_counts",
]

PROVER_COMPONENTS = ("P0", "P1", "P2", "P3", "P4", "P5")
RUN_PREFIX = "run:"  # a section once per run, reported at step 0


# ---- memory backends ---------------------------------------------------------------------


class MemoryProbe(Protocol):
    source: str

    def start(self) -> int | None:
        """Reset the peak and return the current figure."""
        ...

    def stop(self) -> tuple[int, int | None]:
        """Return ``(peak since start, current)``."""
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
        if self._rl(self._pid, self._RLIMIT_FOOTPRINT_INTERVAL, self._FOOTPRINT_INTERVAL_RESET):
            raise OSError(ctypes.get_errno(), "proc_rlimit_control")
        self._read()

    def _read(self) -> None:
        if self._ru(self._pid, self._RUSAGE_INFO_V4, self._addr):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage")

    def start(self) -> int:
        self._rl(self._pid, self._RLIMIT_FOOTPRINT_INTERVAL, self._FOOTPRINT_INTERVAL_RESET)
        self._read()
        return int(self._buf[self._PHYS_FOOTPRINT])

    def stop(self) -> tuple[int, int]:
        self._read()
        cur = int(self._buf[self._PHYS_FOOTPRINT])
        return max(int(self._buf[self._INTERVAL_MAX]), cur), cur


class _LinuxProbe:
    source = "linux_rss_hwm"

    def __init__(self) -> None:
        self._clear = open("/proc/self/clear_refs", "w")  # noqa: SIM115 (kept open for speed)
        self.start()

    @staticmethod
    def _status() -> tuple[int, int]:
        rss = hwm = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
        return hwm, rss

    def start(self) -> int:
        self._clear.write("5")
        self._clear.flush()
        return self._status()[1]

    def stop(self) -> tuple[int, int]:
        hwm, rss = self._status()
        return max(hwm, rss), rss


class _CudaProbe:
    source = "cuda_allocated"

    def __init__(self, device: torch.device) -> None:
        self._dev = device

    def sync(self) -> None:
        torch.cuda.synchronize(self._dev)

    def start(self) -> int:
        torch.cuda.synchronize(self._dev)
        torch.cuda.reset_peak_memory_stats(self._dev)
        return int(torch.cuda.memory_allocated(self._dev))

    def stop(self) -> tuple[int, int]:
        torch.cuda.synchronize(self._dev)
        return (int(torch.cuda.max_memory_allocated(self._dev)),
                int(torch.cuda.memory_allocated(self._dev)))


class _MaxRssProbe:
    source = "ru_maxrss"
    _UNIT = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, kB on Linux

    def start(self) -> None:
        return None

    def stop(self) -> tuple[int, None]:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * self._UNIT, None


def memory_probe(device: str | torch.device = "cpu") -> MemoryProbe:
    """The best memory backend for ``device`` on this platform (see the module docstring)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return _CudaProbe(dev)
    try:
        if sys.platform == "darwin":
            return _DarwinProbe()
        if sys.platform.startswith("linux"):
            return _LinuxProbe()
    except OSError:
        pass
    return _MaxRssProbe()


def lifetime_maxrss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MaxRssProbe._UNIT


# ---- rows --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRow:
    scenario: str
    step: int
    side: str  # "prover" or "verifier"; "run" for the step total
    component: str
    level: str  # "phase", "component", "sub" or "total"
    calls: int
    time_s: float
    peak_bytes: int | None
    start_bytes: int | None
    end_bytes: int | None
    mem_source: str


@dataclass(frozen=True)
class CountRow:
    scenario: str
    step: int
    side: str
    component: str
    level: str
    calls: int
    flops: int
    hash_in_bytes: int
    hash_out_bytes: int


COST_FIELDS = tuple(f.name for f in fields(CostRow))
COUNT_FIELDS = tuple(f.name for f in fields(CountRow))


def _side(name: str) -> str:
    return "prover" if name.startswith("P") else "verifier"


# ---- the recorders -----------------------------------------------------------------------


class _Acc:
    """One section's totals within a step."""

    __slots__ = ("nested", "calls", "time", "peak", "seen", "start", "end", "flops", "hin",
                 "hout")

    def __init__(self, nested: bool) -> None:
        self.nested = nested
        self.calls = 0
        self.time = 0.0
        self.peak = 0
        self.seen = False  # start is taken at the first entry only
        self.start: int | None = None
        self.end: int | None = None
        self.flops = self.hin = self.hout = 0


class _Recorder:
    """Shared bookkeeping: sections per step, ``run:`` sections at step 0, the nesting depth.

    A section opened while another is open is nested (a ``sub`` row). Rows are grouped per step
    as phases, prover components, verifier checks, subs and totals.
    """

    _Section: type

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self._step: dict[str, _Acc] = {}
        self._run: dict[str, _Acc] = {}
        self._depth = 0
        self._verifier: Any = None

    def bind(self, verifier: Any) -> None:
        """Remember the run's ``Verifier``, for residual logging at each step."""
        self._verifier = verifier

    def section(self, name: str) -> Any:
        """The seam: a context manager measuring the block as section ``name``."""
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
        self._emit(self._rows(rec.t, table), rec.t)

    def finish(self) -> None:
        """Close the run: emit the ``run:`` sections as step 0."""
        table, self._run = self._run, {}
        if table:
            self._emit(self._rows(0, table), None)

    def _rows(self, step: int, table: dict[str, _Acc]) -> list[Any]:
        top = {n: a for n, a in table.items() if not a.nested}
        prover = {n: a for n, a in top.items() if _side(n) == "prover"}
        verifier = {n: a for n, a in top.items() if _side(n) == "verifier"}
        rows = [self._agg(step, "prover", n, "phase", [a]) for n, a in prover.items()]
        for comp in PROVER_COMPONENTS:
            accs = [a for n, a in prover.items() if n.split(".", 1)[0] == comp]
            if accs:
                rows.append(self._agg(step, "prover", comp, "component", accs))
        rows += [self._agg(step, "verifier", n, "component", [a]) for n, a in verifier.items()]
        rows += [self._agg(step, _side(n), n, "sub", [a]) for n, a in table.items() if a.nested]
        for side, part in (("prover", prover), ("verifier", verifier)):
            if part:
                rows.append(self._agg(step, side, side, "total", list(part.values())))
        if top:
            rows.append(self._agg(step, "run", "step", "total", list(top.values())))
        return rows

    def _agg(self, step: int, side: str, comp: str, level: str, accs: list[_Acc]) -> Any:
        raise NotImplementedError

    def _emit(self, rows: list[Any], t: int | None) -> None:
        raise NotImplementedError


class _CostSection:
    # Memory is read outside the timed interval, so its cost lands between sections, never in
    # a reported time. On CUDA the device is synchronized before each clock read.
    __slots__ = ("rec", "acc", "top", "t0")

    def __init__(self, rec: CostRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        rec = self.rec
        if rec._depth == 0:
            self.top = True
            start = rec._mstart()
            acc = self.acc
            if not acc.seen:
                acc.start, acc.seen = start, True
        else:
            self.top = False
            if rec._sync is not None:
                rec._sync()
        rec._depth += 1
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: Any) -> None:
        rec = self.rec
        if rec._sync is not None:
            rec._sync()
        dt = time.perf_counter() - self.t0
        acc = self.acc
        acc.time += dt
        acc.calls += 1
        rec._depth -= 1
        if self.top:
            peak, acc.end = rec._mstop()
            if peak > acc.peak:
                acc.peak = peak


class CostRecorder(_Recorder):
    """Wall-clock time and peak memory per section (``costs.csv``).

    Pass :meth:`section` as ``section=`` to ``run_loop`` and the ``Verifier``, :meth:`on_step` as
    ``on_step``, :meth:`bind` the verifier, and call :meth:`finish` after the run.
    """

    _Section = _CostSection

    def __init__(self, scenario: str, *, memory: MemoryProbe | None = None,
                 writer: MetricsWriter | None = None) -> None:
        super().__init__(scenario)
        self.memory = memory or memory_probe()
        self._mstart, self._mstop = self.memory.start, self.memory.stop
        self._sync = getattr(self.memory, "sync", None)
        self.writer = writer
        self.rows: list[CostRow] = []

    def _agg(self, step: int, side: str, comp: str, level: str, accs: list[_Acc]) -> CostRow:
        calls, t = sum(a.calls for a in accs), sum(a.time for a in accs)
        if level == "sub":
            return CostRow(self.scenario, step, side, comp, level, calls, t, 0, 0, 0, "nested")
        return CostRow(self.scenario, step, side, comp, level, calls, t,
                       max(a.peak for a in accs), accs[0].start, accs[-1].end, self.memory.source)

    def _emit(self, rows: list[CostRow], t: int | None) -> None:
        self.rows += rows
        if self.writer is None:
            return
        self.writer.write_costs(rows)
        if t is not None and self._verifier is not None:
            stats = self._verifier.stats.get(t)
            if stats is not None:
                self.writer.write_residuals(self.scenario, t, stats)


# ---- the counting pass -------------------------------------------------------------------


class _HashTally:
    __slots__ = ("bytes_in", "bytes_out")

    def __init__(self) -> None:
        self.bytes_in = 0
        self.bytes_out = 0


def _counting_blake3(real: Any, tally: _HashTally) -> Any:
    """A stand-in for the ``blake3`` module that counts bytes into and out of each hasher."""

    class blake3:  # noqa: N801 (mirrors blake3.blake3)
        AUTO = real.blake3.AUTO

        def __init__(self, data: Any = None, /, **kw: Any) -> None:
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


_HASHING_MODULES = ("verification.commitment.merkle",
                    "verification.verifier.matmul_check.challenges")


class _FlopTally(TorchDispatchMode):
    """Torch's own FLOP formulas (``torch.utils.flop_counter.flop_registry``, the ones
    ``FlopCounterMode`` uses) summed over every op dispatched in the block.

    ``FlopCounterMode`` itself can't be used: its module tracker hooks break the
    ``torch.autograd.grad`` calls that replay makes on leaf tensors.
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
    """Count FLOPs and BLAKE3 bytes for the duration of the block.

    Yields ``(flops, tally)``. Never use it around a timed run: both counters slow the code
    they observe.
    """
    tally = _HashTally()
    mods = [importlib.import_module(m) for m in _HASHING_MODULES]
    real = [m.blake3 for m in mods]
    for m, r in zip(mods, real, strict=True):
        m.blake3 = _counting_blake3(r, tally)
    try:
        with _FlopTally() as flops:
            yield flops, tally
    finally:
        for m, r in zip(mods, real, strict=True):
            m.blake3 = r


class _CountSection:
    __slots__ = ("rec", "acc", "token")

    def __init__(self, rec: CountRecorder, acc: _Acc) -> None:
        self.rec, self.acc = rec, acc

    def __enter__(self) -> None:
        self.rec._depth += 1
        self.token = self.rec._now()

    def __exit__(self, *exc: Any) -> None:
        f, i, o = self.rec._now()
        f0, i0, o0 = self.token
        acc = self.acc
        acc.flops += f - f0
        acc.hin += i - i0
        acc.hout += o - o0
        acc.calls += 1
        self.rec._depth -= 1


class CountRecorder(_Recorder):
    """FLOPs and bytes hashed per section (``counts.csv``), inside :func:`counting`. A nested
    section's counts are also in its parent's."""

    _Section = _CountSection

    def __init__(self, scenario: str, flops: _FlopTally, tally: _HashTally) -> None:
        super().__init__(scenario)
        self._fm, self._tally = flops, tally
        self.rows: list[CountRow] = []

    def _now(self) -> tuple[int, int, int]:
        return self._fm.total, self._tally.bytes_in, self._tally.bytes_out

    def _agg(self, step: int, side: str, comp: str, level: str, accs: list[_Acc]) -> CountRow:
        return CountRow(self.scenario, step, side, comp, level, sum(a.calls for a in accs),
                        sum(a.flops for a in accs), sum(a.hin for a in accs),
                        sum(a.hout for a in accs))

    def _emit(self, rows: list[CountRow], t: int | None) -> None:
        self.rows += rows


def count_pass(scenario: str, run: Callable[[CountRecorder], Any]) -> list[CountRow]:
    """Run ``run(recorder)`` once under :func:`counting` and return its count rows.

    ``run`` wires the recorder exactly as a timed run wires a :class:`CostRecorder` (``section``,
    ``on_step``, ``bind``). It must be a run of its own, never one whose times are reported.
    """
    with counting() as (fm, tally):
        rec = CountRecorder(scenario, fm, tally)
        run(rec)
        rec.finish()
    return rec.rows


# ---- residuals ---------------------------------------------------------------------------


def residual_rows(scenario: str, step: int, stats: StepStats) -> Iterator[dict[str, Any]]:
    """One row per product (check 5) and per tensor (checks 6a, 6b), in the order recorded."""
    for p in stats.products:
        yield {"scenario": scenario, "step": step, "check": "5", "m": p.m, "name": p.name,
               "cls": p.cls, "kappa": p.kappa, "normalized": list(p.normalized)}
    for s in stats.tensors:
        yield {"scenario": scenario, "step": step, "check": s.check_id, "weight": s.weight,
               "rho_max": s.rho_max}


def read_residuals(path: str | os.PathLike[str]) -> dict[str, dict[int, StepStats]]:
    """``residuals.jsonl`` back as ``{scenario: {step: StepStats}}``, equal to ``verifier.stats``."""
    out: dict[str, dict[int, StepStats]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = out.setdefault(r["scenario"], {}).setdefault(r["step"], StepStats())
            if r["check"] == "5":
                st.products.append(ProductStat(r["m"], r["name"], r["cls"], r["kappa"],
                                               tuple(r["normalized"])))
            else:
                st.tensors.append(TensorStat(r["weight"], r["check"], r["rho_max"]))
    return out


# ---- files -------------------------------------------------------------------------------


def _cell(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, float):
        return repr(x)
    return x


def _read_csv(path: str | os.PathLike[str], cls: type, ints: Iterable[str],
              floats: Iterable[str]) -> list[Any]:
    ints, floats = set(ints), set(floats)
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            kw: dict[str, Any] = {}
            for k, v in r.items():
                if k in ints:
                    kw[k] = None if v == "" else int(v)
                elif k in floats:
                    kw[k] = float(v)
                else:
                    kw[k] = v
            rows.append(cls(**kw))
    return rows


def read_costs(path: str | os.PathLike[str]) -> list[CostRow]:
    return _read_csv(path, CostRow, ("step", "calls", "peak_bytes", "start_bytes", "end_bytes"),
                     ("time_s",))


def read_counts(path: str | os.PathLike[str]) -> list[CountRow]:
    return _read_csv(path, CountRow,
                     ("step", "calls", "flops", "hash_in_bytes", "hash_out_bytes"), ())


class MetricsWriter:
    """The run's metrics directory: ``costs.csv``, ``counts.csv``, ``residuals.jsonl`` and
    ``meta.json``. Rows are appended and flushed as each step closes, so a crashed run keeps
    the steps it finished. Opening the directory truncates the files of an earlier run."""

    def __init__(self, out_dir: str | os.PathLike[str], meta: Mapping[str, Any] | None = None,
                 memory: MemoryProbe | None = None) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory or memory_probe()
        self._costs, self._costs_w = self._csv("costs.csv", COST_FIELDS)
        self._counts, self._counts_w = self._csv("counts.csv", COUNT_FIELDS)
        self._res = open(self.dir / "residuals.jsonl", "w")  # noqa: SIM115
        self.meta: dict[str, Any] = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mem_source": self.memory.source,
            "MallocLargeCache": os.environ.get("MallocLargeCache"),
            **(meta or {}),
        }

    def _csv(self, name: str, header: tuple[str, ...]) -> tuple[IO[str], Any]:
        f = open(self.dir / name, "w", newline="")  # noqa: SIM115
        w = csv.writer(f)
        w.writerow(header)
        return f, w

    def recorder(self, scenario: str) -> CostRecorder:
        return CostRecorder(scenario, memory=self.memory, writer=self)

    def write_costs(self, rows: Iterable[CostRow]) -> None:
        self._costs_w.writerows([_cell(v) for v in asdict(r).values()] for r in rows)
        self._costs.flush()

    def write_counts(self, rows: Iterable[CountRow]) -> None:
        self._counts_w.writerows([_cell(v) for v in asdict(r).values()] for r in rows)
        self._counts.flush()

    def write_residuals(self, scenario: str, step: int, stats: StepStats) -> None:
        for r in residual_rows(scenario, step, stats):
            self._res.write(json.dumps(r, allow_nan=True) + "\n")
        self._res.flush()

    def close(self) -> None:
        self.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.meta["lifetime_maxrss_bytes"] = lifetime_maxrss_bytes()
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str) + "\n")
        for f in (self._costs, self._counts, self._res):
            f.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
