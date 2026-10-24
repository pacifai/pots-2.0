"""The sampler: a separate process that reads the machine every 100 ms beside a session part.

    .venv/bin/python -m setup.sampler --pid PID --out PATH [--interval 0.1] [--gpus 0,1]
    .venv/bin/python -m setup.sampler --out PATH [--interval 0.1] [--gpus 0,1] --wrap -- CMD ...
    .venv/bin/python -m setup.sampler ... --no-gpu    # process and machine fields only, no NVML

Every NVML GPU is sampled unless ``--gpus`` lists some; ``--gpu N`` is ``--gpus N``.

``samples.jsonl`` (archive/PROFILING_PLAN §5.4, version 1; the version 2 below, FS35, is the
one written now) opens with one header line, then one flat line per
sample. A field that can't be read is ``null``, and a rate is the change since the previous
sample's ``t`` (monotonic ns). Every utilisation field is a fraction (NVML's percent ÷ 100,
the GPM fields as they come), PCIe is bytes/s. A group that fails is logged once as a ``{"kind": "error"}``
line (the header holds the start-up failures in ``errors``) and reads ``null`` from then on.

**Version 2 (FS35).** The header's ``gpu`` is replaced by ``gpus``: one ``{"index", "name",
"uuid", "pci_bus_id"}`` per sampled GPU, in the order sampled (NVML's, or the ``--gpus`` list's);
``None`` with ``--no-gpu`` or without NVML. A GPU field (``g_*``) of a row is a list, one value per
header GPU in that order, ``null`` in the slot of a value that couldn't be read; with no GPU
reader the field is ``null``. Process and machine fields are scalars as before. Each GPU has its
own handle, GPM buffers and fallback state, and its own error groups: with more than one GPU
each group carries the NVML index (``g_util[2]``, ``gpu_name[2]``), logged once each.
``g_other_procs`` counts per GPU the processes outside the sampled tree. The header's ``gpm`` is
true when every GPU uses GPM, ``gpm_error`` is the first failing GPU's, and ``pcie_source`` is the
GPUs' common source (``"mixed"`` when they differ).

Only the standard library at import; ``psutil`` and ``pynvml`` load when the sampler starts.
The loop paces to a deadline, flushes after every line and ends on SIGTERM or when the root
process is gone. Nothing here raises into the sampled run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

__all__ = ["FIELDS", "Handle", "run", "start_sampler", "main"]

VERSION = 2
FIELDS = [
    "t", "g_util", "g_mem_util", "g_mem_used", "g_sm_mhz", "g_mem_mhz", "g_power_w", "g_temp_c",
    "g_throttle", "g_pcie_tx", "g_pcie_rx", "g_other_procs",
    "g_sm_active", "g_occupancy", "g_tensor", "g_dram_bw",
    "c_tree", "c_busy", "c_other", "c_self", "threads",
    "rss", "ram_used", "ram_avail", "swap_used",
    "disk_r", "disk_w", "io_r", "io_w",
    "cpu_mhz", "load1", "minflt", "majflt", "cg_throttled_us", "cg_nr_throttled",
    "aff_mhz",
]
STOP_GRACE_S = 5.0
TREE_REFRESH_S = 1.0
GPM_FALLBACK_S = 5.0
GPU_FIELDS = [f for f in FIELDS if f.startswith("g_")]
MIB = 1 << 20
_CG = Path("/sys/fs/cgroup")
# GPM metric name -> (field, scale to the field's unit); the ids are pynvml's NVML_GPM_METRIC_*.
_GPM_FIELDS = {"SM_UTIL": ("g_sm_active", 0.01), "SM_OCCUPANCY": ("g_occupancy", 0.01),
               "ANY_TENSOR_UTIL": ("g_tensor", 0.01), "DRAM_BW_UTIL": ("g_dram_bw", 0.01)}
_GPM_PCIE = {"PCIE_TX_PER_SEC": ("g_pcie_tx", MIB), "PCIE_RX_PER_SEC": ("g_pcie_rx", MIB)}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_dir(proc_cgroup: str | None = None) -> Path:
    """The sampler's own cgroup v2 directory, from ``/proc/self/cgroup`` text (the ``0::<path>``
    line), else the cgroup root (v1, no cgroup, or a container's own namespace root)."""
    text = _read_text(Path("/proc/self/cgroup")) if proc_cgroup is None else proc_cgroup
    for line in (text or "").splitlines():
        if line.startswith("0::"):
            return _CG / line[3:].strip().lstrip("/")
    return _CG


def _cgroup_limits(base: Path = _CG) -> dict[str, float | None]:
    """The container's CPU quota (cores) and memory limit (bytes), ``None`` where unlimited or
    absent (cgroup v2 under ``base``, else v1)."""
    quota = limit = None
    cpu_max = _read_text(base / "cpu.max")
    if cpu_max:
        q, _, p = cpu_max.partition(" ")
        quota = int(q) / int(p) if q != "max" and p else None
    elif (q1 := _read_text(_CG / "cpu/cpu.cfs_quota_us")) and (p1 := _read_text(
            _CG / "cpu/cpu.cfs_period_us")) and int(q1) > 0:
        quota = int(q1) / int(p1)
    mem = _read_text(base / "memory.max") or _read_text(_CG / "memory/memory.limit_in_bytes")
    if mem and mem != "max" and int(mem) < 1 << 60:
        limit = int(mem)
    return {"cpu_quota": quota, "mem_limit": limit}


def parse_throttle(text: str, scale: int = 1) -> tuple[int, int] | None:
    """(throttled µs, throttled periods) from ``cpu.stat`` text: v2's ``throttled_usec``, or v1's
    ``throttled_time`` (ns, so ``scale`` 1000). None when the counters are absent or malformed."""
    stat = dict(line.split() for line in text.splitlines() if len(line.split()) == 2)
    try:
        us = int(stat["throttled_usec"]) if "throttled_usec" in stat \
            else int(stat["throttled_time"]) // scale
        return us, int(stat["nr_throttled"])
    except (KeyError, ValueError):
        return None


def _cgroup_throttle(base: Path = _CG) -> tuple[int, int] | None:
    """(throttled µs, throttled periods), cumulative, from the cgroup's ``cpu.stat``."""
    for path, scale in ((base / "cpu.stat", 1), (_CG / "cpu,cpuacct/cpu.stat", 1000),
                        (_CG / "cpu/cpu.stat", 1000)):
        text = _read_text(path)
        if text is not None:
            return parse_throttle(text, scale)
    return None


def affinity_mhz(root: str = "/sys/devices/system/cpu", cpus: Any = None) -> float | None:
    """Mean current clock (MHz) over the CPUs in our affinity set, from sysfs ``scaling_cur_freq``
    (kHz); None where cpufreq is absent (macOS, many VMs)."""
    if cpus is None:
        if not hasattr(os, "sched_getaffinity"):
            return None
        cpus = os.sched_getaffinity(0)
    vals = []
    for c in cpus:
        try:
            vals.append(int(Path(f"{root}/cpu{c}/cpufreq/scaling_cur_freq").read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else None


def _faults(p: Any) -> tuple[int, int]:
    """(minor, major) page faults of one process, cumulative."""
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{p.pid}/stat").read_text()
        rest = stat[stat.rindex(")") + 2:].split()
        return int(rest[7]), int(rest[9])
    m = p.memory_info()
    return m.pfaults - m.pageins, m.pageins


# Per-process readers by group: (fields, reader). The levels are summed over the tree, the
# cumulative counters become per-second rates.
_TREE: dict[str, tuple[tuple[str, ...], Callable[[Any], tuple[float, ...]]]] = {
    "cpu": (("c_tree",), lambda p: (sum(p.cpu_times()[:2]),)),
    "rss": (("rss",), lambda p: (p.memory_info().rss,)),
    "threads": (("threads",), lambda p: (p.num_threads(),)),
    "io": (("io_r", "io_w"), lambda p: (lambda c: (c.read_bytes, c.write_bytes))(p.io_counters())),
    "faults": (("minflt", "majflt"), lambda p: _faults(p)),
}
_LEVELS = ("rss", "threads")


class _Gpu:
    """One sampled GPU: its NVML handle, GPM buffers and fallback state."""

    def __init__(self, index: int, tag: str) -> None:
        self.index, self.tag = index, tag  # tag: the suffix of this GPU's error groups
        self.handle: Any = None
        self.gpm: dict[str, Any] | None = None
        self.gpm_error: str | None = None
        self.gpm_t0 = 0
        self.gpm_ok = False
        self.pcie_source: str | None = None
        self.info: dict[str, Any] = {"index": index, "name": None, "uuid": None,
                                     "pci_bus_id": None}


class _Sampler:
    def __init__(self, pid: int, out: IO[str], interval: float,
                 gpus: Sequence[int] | int | None, psutil: Any) -> None:
        """``gpus``: the NVML indices to sample, one index, or ``None`` for every GPU."""
        self.ps, self.out, self.interval = psutil, out, interval
        self.gpu_arg = [gpus] if isinstance(gpus, int) else None if gpus is None else list(gpus)
        self.root = psutil.Process(pid)
        self.me = psutil.Process()
        self.cg = _cgroup_dir()
        self.procs: dict[int, Any] = {}
        self.tree_at = float("-inf")
        self.prev: dict[Any, float] = {}
        self.t_prev = time.monotonic_ns()
        self.failed: set[str] = set()
        self.errors: list[dict[str, str]] = []
        self.cpu_mhz: float | None = None
        self.cpu_mhz_due = True
        self.aff_mhz: float | None = None
        self.started = False
        self.no_gpu = False  # set by ``run``: leave NVML alone, the GPU fields read null
        self.nv: Any = None
        self.gpus: list[_Gpu] = []

    # -- failures ---------------------------------------------------------------------------

    def _try(self, group: str, fn: Callable[[], Any]) -> Any:
        """``fn()``, or ``None`` after logging the group's first failure."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (the sampler never raises)
            self._log(group, f"{type(e).__name__}: {e}")
            return None

    def _log(self, group: str, error: str) -> None:
        """Record ``group``'s first failure: in the header's errors before it is written,
        else as an error line."""
        if group in self.failed:
            return
        self.failed.add(group)
        err = {"group": group, "error": error}
        if self.started:
            self.out.write(json.dumps({"kind": "error", **err}) + "\n")
        else:
            self.errors.append(err)

    # -- start-up ---------------------------------------------------------------------------

    def _init_gpu(self) -> None:
        if self.no_gpu:
            return
        try:
            import pynvml as nv
            nv.nvmlInit()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"group": "nvml", "error": f"{type(e).__name__}: {e}"})
            return
        self.nv = nv
        indices = self.gpu_arg
        if indices is None:
            count = self._try("nvml_count", nv.nvmlDeviceGetCount)
            indices = [] if count is None else list(range(count))
        multi = len(indices) > 1
        self.gpus = [_Gpu(i, f"[{i}]" if multi else "") for i in indices]
        for g in self.gpus:
            self._init_one(g)

    def _init_one(self, g: _Gpu) -> None:
        nv = self.nv
        g.handle = self._try("nvml_handle" + g.tag, lambda: nv.nvmlDeviceGetHandleByIndex(g.index))
        if g.handle is None:
            return

        def text(v: Any) -> str:
            return v.decode() if isinstance(v, bytes) else str(v)

        g.info["name"] = self._try("gpu_name" + g.tag,
                                   lambda: text(nv.nvmlDeviceGetName(g.handle)))
        g.info["uuid"] = self._try("gpu_uuid" + g.tag,
                                   lambda: text(nv.nvmlDeviceGetUUID(g.handle)))
        g.info["pci_bus_id"] = self._try("gpu_pci" + g.tag,
                                         lambda: text(nv.nvmlDeviceGetPciInfo(g.handle).busId))
        self._init_gpm(g)
        if g.gpm is not None:
            g.pcie_source = "gpm"
        elif self._try("pcie" + g.tag, lambda: nv.nvmlDeviceGetPcieThroughput(
                g.handle, nv.NVML_PCIE_UTIL_TX_BYTES)) is not None:
            g.pcie_source = "nvml"

    def _init_gpm(self, g: _Gpu) -> None:
        nv = self.nv
        try:
            if not nv.nvmlGpmQueryDeviceSupport(g.handle).isSupportedDevice:
                g.gpm_error = "nvmlGpmQueryDeviceSupport: the device doesn't support GPM"
                return
            ids = {name: getattr(nv, f"NVML_GPM_METRIC_{name}")
                   for name in (*_GPM_FIELDS, *_GPM_PCIE)}
            g.gpm = {"ids": ids, "bufs": [nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()],
                     "n": 0}
            nv.nvmlGpmSampleGet(g.handle, g.gpm["bufs"][0])
            g.gpm_t0 = time.monotonic_ns()
        except Exception as e:  # noqa: BLE001
            g.gpm_error = f"{type(e).__name__}: {e}"
            g.gpm = None

    def header(self) -> dict[str, Any]:
        ps = self.ps
        live = any(g.handle is not None for g in self.gpus)
        sources = {g.pcie_source for g in self.gpus if g.handle is not None}
        return {
            "kind": "header", "version": VERSION, "pid": self.root.pid,
            "interval_s": self.interval, "mono_ns": time.monotonic_ns(),
            "wall_unix_s": time.time(), "host": socket.gethostname(),
            "gpus": [g.info for g in self.gpus] if live else None,
            "nvml": live, "gpm": live and all(g.gpm is not None for g in self.gpus),
            "gpm_error": (self._gpm_error() if self.nv is not None else
                          "the GPU reader is off (--no-gpu)" if self.no_gpu
                          else "NVML isn't available"),
            "pcie_source": sources.pop() if len(sources) == 1 else "mixed" if sources else None,
            "cpu_logical": self._try("cpu_count", lambda: ps.cpu_count()),
            "cpu_physical": self._try("cpu_count", lambda: ps.cpu_count(logical=False)),
            "cgroup": self._try("cgroup", lambda: _cgroup_limits(self.cg)) or {"cpu_quota": None, "mem_limit": None},
            "errors": self.errors, "fields": FIELDS,
        }

    def _gpm_error(self) -> str | None:
        """The first GPU's GPM failure (named by its index when several are sampled)."""
        for g in self.gpus:
            if g.gpm_error is not None:
                return g.gpm_error if len(self.gpus) == 1 else f"GPU {g.index}: {g.gpm_error}"
        return None

    # -- readers ----------------------------------------------------------------------------

    def _refresh_tree(self, now: float) -> None:
        if now - self.tree_at < TREE_REFRESH_S:
            return
        self.tree_at = now
        kids = self._try("tree", lambda: self.root.children(recursive=True)) or []
        self.procs = {p.pid: p for p in (self.root, *kids)}

    def _delta(self, key: str, value: float, dt: float) -> float:
        old = self.prev.get(key)
        self.prev[key] = value
        return 0.0 if old is None else (value - old) / dt

    def _machine(self, row: dict[str, Any], dt: float) -> None:
        ps = self.ps

        def busy() -> float:
            t = ps.cpu_times()
            idle = t.idle + getattr(t, "iowait", 0.0)
            total = sum(t) - getattr(t, "guest", 0.0) - getattr(t, "guest_nice", 0.0)
            return self._delta("busy", total - idle, dt)
        row["c_busy"] = self._try("c_busy", busy)
        row["c_self"] = self._try("c_self", lambda: self._delta(
            "self", sum(self.me.cpu_times()[:2]), dt))
        mem = self._try("ram", ps.virtual_memory)
        if mem is not None:
            row["ram_used"], row["ram_avail"] = mem.total - mem.available, mem.available
        swap = self._try("swap", ps.swap_memory)
        row["swap_used"] = None if swap is None else swap.used

        disk = self._try("disk", ps.disk_io_counters)
        if disk is not None:
            row["disk_r"] = self._delta("disk_r", disk.read_bytes, dt)
            row["disk_w"] = self._delta("disk_w", disk.write_bytes, dt)
        row["load1"] = self._try("load1", lambda: os.getloadavg()[0])
        if self.cpu_mhz_due:
            self.cpu_mhz = self._cpu_mhz()
        row["cpu_mhz"] = self.cpu_mhz
        if self.cpu_mhz_due:
            self.aff_mhz = self._try("aff_mhz", affinity_mhz)
        row["aff_mhz"] = self.aff_mhz
        if sys.platform.startswith("linux"):
            cg = self._try("cgroup_stat", lambda: _cgroup_throttle(self.cg))
            if cg is not None:
                row["cg_throttled_us"] = self._delta("cg_us", cg[0], 1.0)
                row["cg_nr_throttled"] = self._delta("cg_nr", cg[1], 1.0)

    def _cpu_mhz(self) -> float | None:
        # Absent on some machines (Apple silicon raises), so a failure isn't an error.
        try:
            f = self.ps.cpu_freq()
            return f.current if f else None
        except Exception:  # noqa: BLE001
            return None

    def _new_child_cpu(self, p: Any, cum: float, dt: float, wall: float) -> float:
        """CPU a process unseen so far used in this interval: all of ``cum`` if it was born
        inside it, else ``cum`` scaled by the interval's share of its age."""
        try:
            age = wall - p.create_time()
        except self.ps.Error:
            return 0.0
        return cum if age <= dt else cum * dt / age

    def _tree(self, row: dict[str, Any], dt: float, wall: float) -> None:
        """One pass over the tree, each process under ``oneshot()``, each group read once."""
        groups = {g: v for g, v in _TREE.items()
                  if g != "io" or hasattr(self.ps.Process, "io_counters")}
        acc = {f: 0.0 for fields, _ in groups.values() for f in fields}
        bad: set[str] = set()
        gone = (self.ps.Error, OSError, AttributeError, NotImplementedError)
        for pid, p in list(self.procs.items()):
            try:
                with p.oneshot():
                    for g, (fields, read) in groups.items():
                        if g in bad:
                            continue
                        try:
                            vals = read(p)
                        except gone:
                            raise
                        except Exception as e:  # noqa: BLE001
                            self._log(g, f"{type(e).__name__}: {e}")
                            bad.add(g)
                            continue
                        for f, v in zip(fields, vals):
                            if g in _LEVELS:
                                acc[f] += v
                                continue
                            old = self.prev.get((f, pid))
                            self.prev[(f, pid)] = v
                            if old is not None:
                                acc[f] += v - old
                            elif g == "cpu":
                                acc[f] += self._new_child_cpu(p, v, dt, wall)
            except gone:
                self.procs.pop(pid, None)
        for g, (fields, _) in groups.items():
            for f in fields:
                row[f] = None if g in bad else acc[f] if g in _LEVELS else acc[f] / dt
        self.prev = {k: v for k, v in self.prev.items()
                     if not isinstance(k, tuple) or k[1] in self.procs}
        if row["c_busy"] is not None and row["c_tree"] is not None:
            row["c_other"] = row["c_busy"] - row["c_tree"]

    def _gpu(self, row: dict[str, Any]) -> None:
        """Every GPU's fields: a list per ``g_*`` field, one slot per GPU, ``None`` where a
        value couldn't be read (or the GPU has no handle)."""
        if self.nv is None or not self.gpus:
            return
        for f in GPU_FIELDS:
            row[f] = [None] * len(self.gpus)
        for pos, g in enumerate(self.gpus):
            if g.handle is None:
                continue
            part: dict[str, Any] = dict.fromkeys(GPU_FIELDS)
            self._gpu_one(g, part)
            for f in GPU_FIELDS:
                row[f][pos] = part[f]

    def _gpu_one(self, g: _Gpu, row: dict[str, Any]) -> None:
        nv, h, t = self.nv, g.handle, g.tag
        u = self._try("g_util" + t, lambda: nv.nvmlDeviceGetUtilizationRates(h))
        if u is not None:
            row["g_util"], row["g_mem_util"] = u.gpu / 100, u.memory / 100
        mem = self._try("g_mem_used" + t, lambda: nv.nvmlDeviceGetMemoryInfo(h))
        row["g_mem_used"] = None if mem is None else mem.used
        row["g_sm_mhz"] = self._try("g_sm_mhz" + t, lambda: nv.nvmlDeviceGetClockInfo(
            h, nv.NVML_CLOCK_SM))
        row["g_mem_mhz"] = self._try("g_mem_mhz" + t, lambda: nv.nvmlDeviceGetClockInfo(
            h, nv.NVML_CLOCK_MEM))
        mw = self._try("g_power_w" + t, lambda: nv.nvmlDeviceGetPowerUsage(h))
        row["g_power_w"] = None if mw is None else mw / 1000
        row["g_temp_c"] = self._try("g_temp_c" + t, lambda: nv.nvmlDeviceGetTemperature(
            h, nv.NVML_TEMPERATURE_GPU))
        reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) \
            or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        row["g_throttle"] = self._try("g_throttle" + t, lambda: reasons(h))
        if g.pcie_source == "nvml":
            def pcie(counter: int) -> float:
                return nv.nvmlDeviceGetPcieThroughput(h, counter) * 1024
            row["g_pcie_tx"] = self._try("pcie" + t, lambda: pcie(nv.NVML_PCIE_UTIL_TX_BYTES))
            row["g_pcie_rx"] = self._try("pcie" + t, lambda: pcie(nv.NVML_PCIE_UTIL_RX_BYTES))
        procs = self._try("g_other_procs" + t, lambda: nv.nvmlDeviceGetComputeRunningProcesses(h))
        if procs is not None:
            if procs and not any(self.ps.pid_exists(p.pid) for p in procs):
                self._log("g_other_procs" + t, "the GPU's process ids don't resolve here "
                          "(another pid namespace), so the count would be wrong")
            else:
                row["g_other_procs"] = sum(1 for p in procs if p.pid not in self.procs)
        if g.gpm is not None:
            ok = self._try("gpm" + t, lambda: self._gpm(g, row))
            g.gpm_ok = g.gpm_ok or bool(ok)
            if not g.gpm_ok and time.monotonic_ns() - g.gpm_t0 >= GPM_FALLBACK_S * 1e9:
                g.gpm, g.pcie_source = None, "nvml"
                self._log("gpm_fallback" + t, f"GPM gave no reading in {GPM_FALLBACK_S:g} s; "
                          "PCIe now comes from nvmlDeviceGetPcieThroughput")

    def _gpm(self, g: _Gpu, row: dict[str, Any]) -> bool:
        """One GPM reading of ``g`` into ``row``; a metric that failed stays ``None``. True
        when any metric answered."""
        nv, gm = self.nv, g.gpm
        old, new = gm["bufs"][gm["n"] % 2], gm["bufs"][(gm["n"] + 1) % 2]
        nv.nvmlGpmSampleGet(g.handle, new)
        gm["n"] += 1
        names = list(gm["ids"])
        q = nv.c_nvmlGpmMetricsGet_t()
        q.version = nv.NVML_GPM_METRICS_GET_VERSION
        q.numMetrics = len(names)
        q.sample1, q.sample2 = old, new
        for i, name in enumerate(names):
            q.metrics[i].metricId = gm["ids"][name]
        nv.nvmlGpmMetricsGet(q)
        ok = False
        for i, name in enumerate(names):
            field, scale = {**_GPM_FIELDS, **_GPM_PCIE}[name]
            if getattr(q.metrics[i], "nvmlReturn", 0) == getattr(nv, "NVML_SUCCESS", 0):
                row[field] = q.metrics[i].value * scale
                ok = True
        return ok

    def sample(self, t: int) -> dict[str, Any]:
        dt = max((t - self.t_prev) / 1e9, 1e-9)
        wall = time.time()
        row: dict[str, Any] = dict.fromkeys(FIELDS)
        row["t"] = t
        self.cpu_mhz_due = int(t / 1e9) != int(self.t_prev / 1e9)
        self._refresh_tree(t / 1e9)
        self._machine(row, dt)
        self._tree(row, dt, wall)
        self._gpu(row)
        self.t_prev = t
        return row

    def close(self) -> None:
        if self.nv is None:
            return
        try:
            for g in self.gpus:
                if g.gpm is not None:
                    for b in g.gpm["bufs"]:
                        self.nv.nvmlGpmSampleFree(b)
            self.nv.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _root_alive(psutil: Any, root: Any) -> bool:
    try:
        return root.is_running() and root.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def run(pid: int, out: Path, interval: float = 0.1, gpus: Sequence[int] | int | None = None,
        stop: threading.Event | None = None,
        alive: Callable[[], bool] | None = None, no_gpu: bool = False) -> int:
    """Sample ``pid``'s tree and the machine to ``out`` until ``stop`` is set, ``alive()`` is
    false or the root process is gone. Returns 0, or 1 when the file can't be opened. Without
    ``psutil`` the file holds the header and its error only. ``no_gpu`` never starts NVML: the
    header reads ``"nvml": false`` and every GPU field ``null``, on a GPU host too. ``gpus``
    names the NVML indices to sample (one index or a list); ``None`` samples every GPU."""
    stop = stop or threading.Event()
    try:
        f = open(out, "w")
    except OSError as e:
        print(f"sampler: can't open {out}: {e}", file=sys.stderr)
        return 1
    with f:
        try:
            import psutil
            s = _Sampler(pid, f, interval, gpus, psutil)
        except Exception as e:  # noqa: BLE001 (no psutil, or the root is already gone)
            f.write(json.dumps({"kind": "header", "version": VERSION, "pid": pid,
                                "interval_s": interval, "mono_ns": time.monotonic_ns(),
                                "wall_unix_s": time.time(), "fields": FIELDS,
                                "errors": [{"group": "psutil",
                                            "error": f"{type(e).__name__}: {e}"}]}) + "\n")
            return 0
        try:
            s.no_gpu = no_gpu
            s._init_gpu()
            s._refresh_tree(time.monotonic())
            s.sample(time.monotonic_ns())  # the baseline for every rate
            f.write(json.dumps(s.header()) + "\n")
            s.started = True
            f.flush()
            if alive is None:
                def alive() -> bool:
                    return _root_alive(psutil, s.root)
            nxt = time.monotonic()
            while True:
                nxt += interval
                now = time.monotonic()
                if stop.wait(max(0.0, nxt - now)) or not alive():
                    break
                nxt = max(nxt, time.monotonic() - interval)  # an overrun skips, never bursts
                f.write(json.dumps(s.sample(time.monotonic_ns())) + "\n")
                f.flush()
        except Exception as e:  # noqa: BLE001
            print(f"sampler: stopped on {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            s.close()
    return 0


class Handle:
    """A running sampler process; :meth:`stop` ends it."""

    def __init__(self, proc: subprocess.Popen[bytes], out: Path) -> None:
        self.proc, self.out = proc, out

    def stop(self) -> None:
        """SIGTERM, then SIGKILL after :data:`STOP_GRACE_S`. Never raises. A sampler that is
        still starting (no header yet) gets up to :data:`STOP_GRACE_S` to write it first, so a
        part that ends at once still leaves its file; one with no header after that is
        killed at once."""
        deadline = time.monotonic() + STOP_GRACE_S
        ready = False
        while self.proc.poll() is None and time.monotonic() < deadline:
            try:
                if self.out.stat().st_size > 0:
                    ready = True
                    break
            except OSError:
                pass
            time.sleep(0.02)
        if self.proc.poll() is None and not ready:
            steps = ((signal.SIGKILL, STOP_GRACE_S),)  # stuck before its header: no second wait
        else:
            steps = ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, STOP_GRACE_S))
        for sig, wait in steps:
            try:
                self.proc.send_signal(sig)
                self.proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                return


def start_sampler(pid: int, out: Path | str, interval: float = 0.1, *,
                  gpus: Sequence[int] | int | None = None, cwd: Path | str | None = None,
                  python: str = sys.executable) -> Handle:
    """Start the sampler on ``pid`` in its own session, so a process-group kill of the run
    doesn't reach it. It ends itself when ``pid`` is gone. ``gpus`` (an index or a list) limits
    the NVML GPUs sampled; the default is every one."""
    argv = [python, "-m", "setup.sampler", "--pid", str(pid), "--out", str(out),
            "--interval", str(interval)]
    if gpus is not None:
        argv += ["--gpus", ",".join(map(str, [gpus] if isinstance(gpus, int) else gpus))]
    proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    return Handle(proc, Path(out))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="python -m setup.sampler", description=__doc__.split("\n")[0])
    ap.add_argument("--pid", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated NVML indices to sample (default: every GPU)")
    ap.add_argument("--gpu", type=int, default=None, help="one GPU: the same as --gpus N")
    ap.add_argument("--no-gpu", action="store_true", help="don't start NVML: no GPU fields")
    ap.add_argument("--wrap", action="store_true", help="start CMD (after --) and sample it")
    a = ap.parse_args(argv)
    try:
        gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else \
            None if a.gpu is None else [a.gpu]
    except ValueError:
        ap.error("--gpus takes comma-separated integers, e.g. 0,1")
    if a.wrap == (a.pid is not None) or (a.wrap and not cmd) or a.interval <= 0:
        ap.error("give --pid, or --wrap -- CMD ...; the interval must be positive")
    stop = threading.Event()
    child: subprocess.Popen[bytes] | None = None
    if a.wrap:
        child = subprocess.Popen(cmd)

    def on_term(signum: int, frame: Any) -> None:
        stop.set()
        if child is not None and child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, on_term)
    pid = child.pid if child is not None else a.pid
    if child is None:
        return run(pid, a.out, a.interval, gpus, stop, no_gpu=a.no_gpu)
    try:
        rc = run(pid, a.out, a.interval, gpus, stop, alive=lambda: child.poll() is None,
                 no_gpu=a.no_gpu)
    finally:
        _reap(child)
    return child.returncode if rc == 0 else rc


def _reap(child: subprocess.Popen[bytes]) -> None:
    """Terminate the wrapped command if it still runs, kill it after :data:`STOP_GRACE_S`."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            child.kill()
    child.wait()


if __name__ == "__main__":
    sys.exit(main())
