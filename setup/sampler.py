"""The sampler: a separate process that reads the machine every 100 ms beside a session part.

    .venv/bin/python -m setup.sampler --pid PID --out PATH [--interval 0.1] [--gpu 0]
    .venv/bin/python -m setup.sampler --out PATH [--interval 0.1] [--gpu 0] --wrap -- CMD ...

``samples.jsonl`` (PROFILING_PLAN §5.4) opens with one header line, then one flat line per
sample. A field that can't be read is ``null``, and a rate is the change since the previous
sample's ``t`` (monotonic ns). GPU util and memory util are NVML's percent (0–100), the GPM
fields fractions, PCIe bytes/s. A group that fails is logged once as a ``{"kind": "error"}``
line (the header holds the start-up failures in ``errors``) and reads ``null`` from then on.

Only the standard library at import; ``psutil`` and ``pynvml`` load when the sampler starts.
The loop paces to a deadline, flushes once a second and ends on SIGTERM or when the root
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
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

__all__ = ["FIELDS", "Handle", "run", "start_sampler", "main"]

VERSION = 1
FIELDS = [
    "t", "g_util", "g_mem_util", "g_mem_used", "g_sm_mhz", "g_mem_mhz", "g_power_w", "g_temp_c",
    "g_throttle", "g_pcie_tx", "g_pcie_rx", "g_other_procs",
    "g_sm_active", "g_occupancy", "g_tensor", "g_dram_bw",
    "c_tree", "c_busy", "c_other", "c_self", "threads",
    "rss", "ram_used", "ram_avail", "swap_used",
    "disk_r", "disk_w", "io_r", "io_w",
    "cpu_mhz", "load1", "minflt", "majflt", "cg_throttled_us", "cg_nr_throttled",
]
STOP_GRACE_S = 5.0
TREE_REFRESH_S = 1.0
FLUSH_S = 1.0
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


def _cgroup_limits() -> dict[str, float | None]:
    """The container's CPU quota (cores) and memory limit (bytes), ``None`` where unlimited or
    absent (cgroup v2, else v1)."""
    quota = limit = None
    cpu_max = _read_text(_CG / "cpu.max")
    if cpu_max:
        q, _, p = cpu_max.partition(" ")
        quota = int(q) / int(p) if q != "max" and p else None
    elif (q1 := _read_text(_CG / "cpu/cpu.cfs_quota_us")) and (p1 := _read_text(
            _CG / "cpu/cpu.cfs_period_us")) and int(q1) > 0:
        quota = int(q1) / int(p1)
    mem = _read_text(_CG / "memory.max") or _read_text(_CG / "memory/memory.limit_in_bytes")
    if mem and mem != "max" and int(mem) < 1 << 60:
        limit = int(mem)
    return {"cpu_quota": quota, "mem_limit": limit}


def _cgroup_throttle() -> tuple[int, int] | None:
    """(throttled µs, throttled periods), cumulative, from the cgroup's ``cpu.stat``."""
    for path, scale in ((_CG / "cpu.stat", 1), (_CG / "cpu/cpu.stat", 1000)):
        text = _read_text(path)
        if text is None:
            continue
        stat = dict(line.split() for line in text.splitlines() if len(line.split()) == 2)
        us = stat.get("throttled_usec")
        if us is None:
            us = int(stat["throttled_time"]) // scale
        return int(us), int(stat["nr_throttled"])
    return None


def _faults(p: Any, psutil: Any) -> tuple[int, int]:
    """(minor, major) page faults of one process, cumulative."""
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{p.pid}/stat").read_text()
        rest = stat[stat.rindex(")") + 2:].split()
        return int(rest[7]), int(rest[9])
    m = p.memory_info()
    return m.pfaults - m.pageins, m.pageins


class _Sampler:
    def __init__(self, pid: int, out: IO[str], interval: float, gpu: int, psutil: Any) -> None:
        self.ps, self.out, self.interval, self.gpu = psutil, out, interval, gpu
        self.root = psutil.Process(pid)
        self.me = psutil.Process()
        self.procs: dict[int, Any] = {}
        self.tree_at = float("-inf")
        self.prev: dict[Any, float] = {}
        self.t_prev = time.monotonic_ns()
        self.failed: set[str] = set()
        self.errors: list[dict[str, str]] = []
        self.cpu_mhz: float | None = None
        self.cpu_mhz_due = True
        self.started = False
        self.nv: Any = None
        self.handle: Any = None
        self.gpm: dict[str, Any] | None = None
        self.gpm_error: str | None = None
        self.pcie_source: str | None = None
        self.gpu_info: dict[str, Any] = {"index": gpu, "name": None, "uuid": None}

    # -- failures ---------------------------------------------------------------------------

    def _try(self, group: str, fn: Callable[[], Any]) -> Any:
        """``fn()``, or ``None`` after logging the group's first failure."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (the sampler never raises)
            if group not in self.failed:
                self.failed.add(group)
                err = {"group": group, "error": f"{type(e).__name__}: {e}"}
                if self.started:
                    self.out.write(json.dumps({"kind": "error", **err}) + "\n")
                else:
                    self.errors.append(err)
            return None

    # -- start-up ---------------------------------------------------------------------------

    def _init_gpu(self) -> None:
        try:
            import pynvml as nv
            nv.nvmlInit()
        except Exception as e:  # noqa: BLE001
            self.errors.append({"group": "nvml", "error": f"{type(e).__name__}: {e}"})
            return
        self.nv = nv
        self.handle = self._try("nvml_handle", lambda: nv.nvmlDeviceGetHandleByIndex(self.gpu))
        if self.handle is None:
            return

        def text(v: Any) -> str:
            return v.decode() if isinstance(v, bytes) else str(v)

        self.gpu_info["name"] = self._try("gpu_name",
                                          lambda: text(nv.nvmlDeviceGetName(self.handle)))
        self.gpu_info["uuid"] = self._try("gpu_uuid",
                                          lambda: text(nv.nvmlDeviceGetUUID(self.handle)))
        self._init_gpm()
        if self.gpm is not None:
            self.pcie_source = "gpm"
        elif self._try("pcie", lambda: nv.nvmlDeviceGetPcieThroughput(
                self.handle, nv.NVML_PCIE_UTIL_TX_BYTES)) is not None:
            self.pcie_source = "nvml"

    def _init_gpm(self) -> None:
        nv = self.nv
        try:
            if not nv.nvmlGpmQueryDeviceSupport(self.handle).isSupportedDevice:
                self.gpm_error = "nvmlGpmQueryDeviceSupport: the device doesn't support GPM"
                return
            ids = {name: getattr(nv, f"NVML_GPM_METRIC_{name}")
                   for name in (*_GPM_FIELDS, *_GPM_PCIE)}
            self.gpm = {"ids": ids, "bufs": [nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()],
                        "n": 0}
            nv.nvmlGpmSampleGet(self.handle, self.gpm["bufs"][0])
        except Exception as e:  # noqa: BLE001
            self.gpm_error = f"{type(e).__name__}: {e}"
            self.gpm = None

    def header(self) -> dict[str, Any]:
        ps = self.ps
        return {
            "kind": "header", "version": VERSION, "pid": self.root.pid,
            "interval_s": self.interval, "mono_ns": time.monotonic_ns(),
            "wall_unix_s": time.time(), "host": socket.gethostname(),
            "gpu": self.gpu_info if self.handle is not None else None,
            "nvml": self.handle is not None, "gpm": self.gpm is not None,
            "gpm_error": self.gpm_error if self.nv is not None else "NVML isn't available",
            "pcie_source": self.pcie_source,
            "cpu_logical": self._try("cpu_count", lambda: ps.cpu_count()),
            "cpu_physical": self._try("cpu_count", lambda: ps.cpu_count(logical=False)),
            "cgroup": self._try("cgroup", _cgroup_limits) or {"cpu_quota": None, "mem_limit": None},
            "errors": self.errors, "fields": FIELDS,
        }

    # -- readers ----------------------------------------------------------------------------

    def _refresh_tree(self, now: float) -> None:
        if now - self.tree_at < TREE_REFRESH_S:
            return
        self.tree_at = now
        kids = self._try("tree", lambda: self.root.children(recursive=True)) or []
        self.procs = {p.pid: p for p in (self.root, *kids)}

    def _tree_sum(self, key: str, get: Callable[[Any], float]) -> float | None:
        """The tree's total of ``get(p)`` over the processes that answer, ``None`` when the
        reader itself fails."""
        def total() -> float:
            s = 0.0
            for pid, p in list(self.procs.items()):
                try:
                    s += get(p)
                except (self.ps.NoSuchProcess, self.ps.ZombieProcess, self.ps.AccessDenied,
                        OSError, AttributeError, NotImplementedError):
                    self.procs.pop(pid, None)
            return s
        return self._try(key, total)

    def _tree_delta(self, key: str, get: Callable[[Any], float], dt: float) -> float | None:
        """Per-second change of a per-process cumulative counter, over the processes seen in
        both this sample and the last."""
        seen = 0.0
        def delta(p: Any) -> float:
            nonlocal seen
            v = get(p)
            old = self.prev.get((key, p.pid))
            self.prev[(key, p.pid)] = v
            if old is not None:
                seen += v - old
            return 0.0
        total = self._tree_sum(key, delta)
        return None if total is None else seen / dt

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
        if sys.platform.startswith("linux"):
            cg = self._try("cgroup_stat", _cgroup_throttle)
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

    def _tree(self, row: dict[str, Any], dt: float) -> None:
        row["c_tree"] = self._tree_delta("cpu", lambda p: sum(p.cpu_times()[:2]), dt)
        row["rss"] = self._tree_sum("rss", lambda p: p.memory_info().rss)
        row["threads"] = self._tree_sum("threads", lambda p: p.num_threads())
        if hasattr(self.ps.Process, "io_counters"):
            row["io_r"] = self._tree_delta("io_r", lambda p: p.io_counters().read_bytes, dt)
            row["io_w"] = self._tree_delta("io_w", lambda p: p.io_counters().write_bytes, dt)
        row["minflt"] = self._tree_delta("minflt", lambda p: _faults(p, self.ps)[0], dt)
        row["majflt"] = self._tree_delta("majflt", lambda p: _faults(p, self.ps)[1], dt)
        if row["c_busy"] is not None and row["c_tree"] is not None:
            row["c_other"] = row["c_busy"] - row["c_tree"]

    def _gpu(self, row: dict[str, Any]) -> None:
        nv, h = self.nv, self.handle
        if nv is None or h is None:
            return
        u = self._try("g_util", lambda: nv.nvmlDeviceGetUtilizationRates(h))
        if u is not None:
            row["g_util"], row["g_mem_util"] = u.gpu, u.memory
        mem = self._try("g_mem_used", lambda: nv.nvmlDeviceGetMemoryInfo(h))
        row["g_mem_used"] = None if mem is None else mem.used
        row["g_sm_mhz"] = self._try("g_sm_mhz", lambda: nv.nvmlDeviceGetClockInfo(
            h, nv.NVML_CLOCK_SM))
        row["g_mem_mhz"] = self._try("g_mem_mhz", lambda: nv.nvmlDeviceGetClockInfo(
            h, nv.NVML_CLOCK_MEM))
        mw = self._try("g_power_w", lambda: nv.nvmlDeviceGetPowerUsage(h))
        row["g_power_w"] = None if mw is None else mw / 1000
        row["g_temp_c"] = self._try("g_temp_c", lambda: nv.nvmlDeviceGetTemperature(
            h, nv.NVML_TEMPERATURE_GPU))
        reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) \
            or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        row["g_throttle"] = self._try("g_throttle", lambda: reasons(h))
        if self.pcie_source == "nvml":
            def pcie(counter: int) -> float:
                return nv.nvmlDeviceGetPcieThroughput(h, counter) * 1024
            row["g_pcie_tx"] = self._try("pcie", lambda: pcie(nv.NVML_PCIE_UTIL_TX_BYTES))
            row["g_pcie_rx"] = self._try("pcie", lambda: pcie(nv.NVML_PCIE_UTIL_RX_BYTES))
        procs = self._try("g_other_procs", lambda: nv.nvmlDeviceGetComputeRunningProcesses(h))
        if procs is not None:
            row["g_other_procs"] = sum(1 for p in procs if p.pid not in self.procs)
        if self.gpm is not None:
            self._try("gpm", lambda: self._gpm(row))

    def _gpm(self, row: dict[str, Any]) -> None:
        nv, g = self.nv, self.gpm
        old, new = g["bufs"][g["n"] % 2], g["bufs"][(g["n"] + 1) % 2]
        nv.nvmlGpmSampleGet(self.handle, new)
        g["n"] += 1
        names = list(g["ids"])
        q = nv.c_nvmlGpmMetricsGet_t()
        q.version = nv.NVML_GPM_METRICS_GET_VERSION
        q.numMetrics = len(names)
        q.sample1, q.sample2 = old, new
        for i, name in enumerate(names):
            q.metrics[i].metricId = g["ids"][name]
        nv.nvmlGpmMetricsGet(q)
        for i, name in enumerate(names):
            field, scale = {**_GPM_FIELDS, **_GPM_PCIE}[name]
            row[field] = q.metrics[i].value * scale

    def sample(self, t: int) -> dict[str, Any]:
        dt = max((t - self.t_prev) / 1e9, 1e-9)
        row: dict[str, Any] = dict.fromkeys(FIELDS)
        row["t"] = t
        self.cpu_mhz_due = int(t / 1e9) != int(self.t_prev / 1e9)
        self._refresh_tree(t / 1e9)
        self._machine(row, dt)
        self._tree(row, dt)
        self._gpu(row)
        self.t_prev = t
        return row

    def close(self) -> None:
        if self.nv is None:
            return
        try:
            if self.gpm is not None:
                for b in self.gpm["bufs"]:
                    self.nv.nvmlGpmSampleFree(b)
            self.nv.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _root_alive(psutil: Any, root: Any) -> bool:
    try:
        return root.is_running() and root.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def run(pid: int, out: Path, interval: float = 0.1, gpu: int = 0,
        stop: threading.Event | None = None,
        alive: Callable[[], bool] | None = None) -> int:
    """Sample ``pid``'s tree and the machine to ``out`` until ``stop`` is set, ``alive()`` is
    false or the root process is gone. Returns 0, or 1 when the file can't be opened. Without
    ``psutil`` the file holds the header and its error only."""
    stop = stop or threading.Event()
    try:
        f = open(out, "w")
    except OSError as e:
        print(f"sampler: can't open {out}: {e}", file=sys.stderr)
        return 1
    with f:
        try:
            import psutil
            s = _Sampler(pid, f, interval, gpu, psutil)
        except Exception as e:  # noqa: BLE001 (no psutil, or the root is already gone)
            f.write(json.dumps({"kind": "header", "version": VERSION, "pid": pid,
                                "interval_s": interval, "mono_ns": time.monotonic_ns(),
                                "wall_unix_s": time.time(), "fields": FIELDS,
                                "errors": [{"group": "psutil",
                                            "error": f"{type(e).__name__}: {e}"}]}) + "\n")
            return 0
        try:
            s._init_gpu()
            s._refresh_tree(time.monotonic())
            s.sample(time.monotonic_ns())  # the baseline for every rate
            f.write(json.dumps(s.header()) + "\n")
            s.started = True
            f.flush()
            if alive is None:
                def alive() -> bool:
                    return _root_alive(psutil, s.root)
            flushed = nxt = time.monotonic()
            while True:
                nxt += interval
                now = time.monotonic()
                if stop.wait(max(0.0, nxt - now)) or not alive():
                    break
                nxt = max(nxt, time.monotonic() - interval)  # an overrun skips, never bursts
                f.write(json.dumps(s.sample(time.monotonic_ns())) + "\n")
                if time.monotonic() - flushed >= FLUSH_S:
                    f.flush()
                    flushed = time.monotonic()
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
        part that ends at once still leaves its file."""
        deadline = time.monotonic() + STOP_GRACE_S
        while self.proc.poll() is None and time.monotonic() < deadline:
            try:
                if self.out.stat().st_size > 0:
                    break
            except OSError:
                pass
            time.sleep(0.02)
        for sig, wait in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, STOP_GRACE_S)):
            try:
                self.proc.send_signal(sig)
                self.proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                return


def start_sampler(pid: int, out: Path | str, interval: float = 0.1, *, gpu: int = 0,
                  cwd: Path | str | None = None, python: str = sys.executable) -> Handle:
    """Start the sampler on ``pid`` in its own session, so a process-group kill of the run
    doesn't reach it. It ends itself when ``pid`` is gone."""
    argv = [python, "-m", "setup.sampler", "--pid", str(pid), "--out", str(out),
            "--interval", str(interval), "--gpu", str(gpu)]
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
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--wrap", action="store_true", help="start CMD (after --) and sample it")
    a = ap.parse_args(argv)
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
    rc = run(pid, a.out, a.interval, a.gpu, stop,
             alive=(lambda: child.poll() is None) if child is not None else None)
    if child is not None:
        return child.wait() if rc == 0 else rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
