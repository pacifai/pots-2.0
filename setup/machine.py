"""The machine's facts, shared by the runs' environment record and the hashing benchmark.

Every field is always present. A field this machine can't supply (no GPU, no NUMA on a Mac)
is ``None``, and ``absent`` maps its name to the reason, so a missing value is never mistaken
for an omitted one. :func:`machine_facts` is the runs' view; the benchmark's own
``machine_record`` adds what only it needs (hashing libraries, thread counts).
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from typing import Any

import torch

_GPU_QUERY = ("name", "driver_version", "memory.total", "power.max_limit", "pci.bus_id",
              "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current",
              "pcie.link.width.max")


def _run(*cmd: str, timeout: float = 30) -> str | None:
    """A command's stdout, or None when it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _sysctl(name: str) -> str | None:
    out = _run("sysctl", "-n", name)
    return out.strip() if out else None


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


class _Record:
    """Fields set in order; an unavailable field is None with its reason in ``absent``."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.absent: dict[str, str] = {}

    def put(self, key: str, value: Any, reason: str = "not reported on this machine") -> None:
        self.data[key] = value
        if value is None:
            self.absent[key] = reason


def _cpu(r: _Record) -> None:
    system = platform.system()
    cpuinfo = _read("/proc/cpuinfo") if system == "Linux" else None
    if system == "Darwin":
        model = _sysctl("machdep.cpu.brand_string")
        phys = _sysctl("hw.physicalcpu")
        levels = {}
        for lvl in range(4):
            n = _sysctl(f"hw.perflevel{lvl}.physicalcpu")
            if n is None:
                break
            levels[_sysctl(f"hw.perflevel{lvl}.name") or f"perflevel{lvl}"] = int(n)
        ram = _sysctl("hw.memsize")
        feats = " ".join(filter(None, (_sysctl("machdep.cpu.features"),
                                       _sysctl("machdep.cpu.leaf7_features")))).upper()
        tokens = set(feats.split())
        r.put("cpu_model", model)
        r.put("physical_cores", int(phys) if phys else None)
        r.put("core_types", levels or None, "no performance levels reported")
        r.put("host_ram_bytes", int(ram) if ram else None)
        x86 = platform.machine() == "x86_64"
        for key, flag in (("avx2", "AVX2"), ("avx512f", "AVX512F"), ("avx512vl", "AVX512VL"),
                          ("sha_ni", "SHA")):
            r.put(key, (flag in tokens) if x86 else None, "not an x86 CPU")
        arm_sha = _sysctl("hw.optional.arm.FEAT_SHA256")
        r.put("arm_sha2", (arm_sha == "1") if arm_sha is not None else None, "not an ARM CPU")
    elif cpuinfo is not None:
        model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        pairs = set()  # one (socket, core) pair per physical core
        for block in cpuinfo.strip().split("\n\n"):
            fields = {k.strip(): v.strip() for k, _, v in
                      (line.partition(":") for line in block.splitlines())}
            if "physical id" in fields and "core id" in fields:
                pairs.add((fields["physical id"], fields["core id"]))
        flags = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", cpuinfo, re.M)
        flagset = set(flags.group(1).split()) if flags else set()
        mem = re.search(r"^MemTotal:\s*(\d+) kB", _read("/proc/meminfo") or "", re.M)
        r.put("cpu_model", model.group(1).strip() if model else None)
        r.put("physical_cores", len(pairs) or None, "no physical id / core id in /proc/cpuinfo")
        r.put("core_types", None, "no core types on Linux")
        r.put("host_ram_bytes", int(mem.group(1)) * 1024 if mem else None)
        x86 = platform.machine() in ("x86_64", "AMD64")
        for key in ("avx2", "avx512f", "avx512vl", "sha_ni"):
            r.put(key, (key in flagset) if x86 else None, "not an x86 CPU")
        r.put("arm_sha2", ("sha2" in flagset) if not x86 else None, "not an ARM CPU")
    else:
        for key in ("cpu_model", "physical_cores", "core_types", "host_ram_bytes", "avx2",
                    "avx512f", "avx512vl", "sha_ni", "arm_sha2"):
            r.put(key, None, f"no CPU probe for {system}")
    _frequency(r, system, cpuinfo)
    r.put("cpu_arch", platform.machine())
    r.put("logical_cpus", os.cpu_count())
    affinity = getattr(os, "sched_getaffinity", None)
    r.put("usable_cpus", len(affinity(0)) if affinity else None,
          "no CPU affinity on this OS")
    r.put("cgroup_cpu_quota", cgroup_cpu_quota(), "no cgroup CPU quota")
    r.put("cgroup_memory_limit_bytes", cgroup_memory_limit(), "no cgroup memory limit")


# cgroup v2's files, then v1's: a container may see either.
_V1_CPU = ("/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpu")
_NO_LIMIT = 1 << 60  # cgroup v1 writes "no limit" as a number near 2^63


def cgroup_cpu_quota() -> float | None:
    """The cgroup's CPU quota in CPUs (quota over period), or None without one."""
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        q, period = cpu_max.split()[:2]
        return None if q == "max" else int(q) / int(period)
    for d in _V1_CPU:
        q, period = _read(f"{d}/cpu.cfs_quota_us"), _read(f"{d}/cpu.cfs_period_us")
        if q and period:
            return None if int(q) < 0 else int(q) / int(period)
    return None


def cgroup_memory_limit() -> int | None:
    """The cgroup's memory limit in bytes, or None without one."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = _read(path)
        if text:
            text = text.strip()
            return None if text == "max" or int(text) >= _NO_LIMIT else int(text)
    return None


def _khz_to_mhz(text: str | None) -> float | None:
    try:
        return int(text.strip()) / 1000 if text else None
    except ValueError:
        return None


def _frequency(r: _Record, system: str, cpuinfo: str | None) -> None:
    """CPU 0's current and maximum clock and its frequency governor, where the OS shows them.
    The current clock is read at idle, before any benchmark runs."""
    if system == "Linux":
        base = "/sys/devices/system/cpu/cpu0/cpufreq"
        cur = _khz_to_mhz(_read(f"{base}/scaling_cur_freq"))
        if cur is None and cpuinfo:  # VMs often hide cpufreq; /proc/cpuinfo still has a clock
            m = re.search(r"^cpu MHz\s*:\s*([\d.]+)", cpuinfo, re.M)
            cur = float(m.group(1)) if m else None
        governor = _read(f"{base}/scaling_governor")
        r.put("cpu_freq_mhz", cur, "no cpufreq in sysfs and no cpu MHz in /proc/cpuinfo")
        r.put("cpu_max_freq_mhz", _khz_to_mhz(_read(f"{base}/cpuinfo_max_freq")),
              "no cpufreq in sysfs")
        r.put("cpu_governor", governor.strip() if governor else None, "no cpufreq in sysfs")
        return
    hz = _sysctl("hw.cpufrequency") if system == "Darwin" else None
    hz_max = _sysctl("hw.cpufrequency_max") if system == "Darwin" else None
    reason = ("macOS doesn't report the clock on Apple silicon" if system == "Darwin"
              else f"no frequency probe for {system}")
    r.put("cpu_freq_mhz", int(hz) / 1e6 if hz else None, reason)
    r.put("cpu_max_freq_mhz", int(hz_max) / 1e6 if hz_max else None, reason)
    r.put("cpu_governor", None, f"no cpufreq governor on {system}")


def _gpu(r: _Record) -> None:
    keys = ("gpu_count", "gpu_name", "gpu_form_factor", "gpu_memory_bytes", "gpu_power_max_w",
            "driver_version", "cuda_runtime", "cuda_driver_supports", "pcie_gen_current",
            "pcie_gen_max", "pcie_width_current", "pcie_width_max", "gpu_pci_bus_id",
            "gpu_numa_node", "gpu_topology")
    if not torch.cuda.is_available():
        reason = ("torch has no CUDA build" if torch.version.cuda is None
                  else "no CUDA device visible")
        for key in keys:
            r.put(key, None, reason)
        return
    r.put("gpu_count", torch.cuda.device_count())
    props = torch.cuda.get_device_properties(0)
    r.put("cuda_runtime", torch.version.cuda)
    smi = _run("nvidia-smi", f"--query-gpu={','.join(_GPU_QUERY)}", "--format=csv,noheader,nounits")
    row = dict(zip(_GPU_QUERY, (v.strip() for v in smi.splitlines()[0].split(",")))) if smi else {}
    no_smi = "nvidia-smi missing or failed"

    def num(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    name = row.get("name") or props.name
    power = num("power.max_limit")
    if "PCIe" in name:
        form = "PCIe"
    elif "NVL" in name:
        form = "NVL"
    elif power is not None and power >= 600:
        form = "SXM (inferred from a power limit of 600 W or more)"
    else:
        form = None
    r.put("gpu_name", name)
    r.put("gpu_form_factor", form, "not in the name and not inferable from the power limit")
    r.put("gpu_memory_bytes", props.total_memory)
    r.put("gpu_power_max_w", power, no_smi)
    r.put("driver_version", row.get("driver_version"), no_smi)
    head = _run("nvidia-smi")
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", head or "")
    r.put("cuda_driver_supports", cuda.group(1) if cuda else None, no_smi)
    for key, q in (("pcie_gen_current", "pcie.link.gen.current"),
                   ("pcie_gen_max", "pcie.link.gen.max"),
                   ("pcie_width_current", "pcie.link.width.current"),
                   ("pcie_width_max", "pcie.link.width.max")):
        v = num(q)
        r.put(key, int(v) if v is not None else None, no_smi)
    bus = row.get("pci.bus_id")
    r.put("gpu_pci_bus_id", bus, no_smi)
    node = None
    if bus:
        sysfs = bus.lower()[-12:]  # "00000000:19:00.0" -> "0000:19:00.0"
        text = _read(f"/sys/bus/pci/devices/{sysfs}/numa_node")
        node = int(text) if text and text.strip().lstrip("-").isdigit() else None
    r.put("gpu_numa_node", node, "no numa_node for the GPU in sysfs")
    r.put("gpu_topology", _run("nvidia-smi", "topo", "-m"), no_smi)


def _numa(r: _Record) -> None:
    nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
    if not nodes:
        r.put("numa_nodes", None, "no NUMA nodes in sysfs (not Linux, or NUMA hidden)")
        return
    out = []
    for path in nodes:
        mem = re.search(r"MemTotal:\s*(\d+) kB", _read(f"{path}/meminfo") or "")
        out.append({"node": int(path.rsplit("node", 1)[1]),
                    "cpulist": (_read(f"{path}/cpulist") or "").strip() or None,
                    "mem_bytes": int(mem.group(1)) * 1024 if mem else None})
    r.put("numa_nodes", out)




def nvml_facts() -> dict[str, Any] | None:
    """GPU 0's driver, power limit, clocks, PCIe link and UUID from NVML, or None without
    ``pynvml`` or a GPU. One failed read is None for that field only."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        handle = None
    try:
        if handle is None:
            return None

        def get(fn: Any, *args: Any) -> Any:
            try:
                value = fn(*args)
            except Exception:
                return None
            return value.decode() if isinstance(value, bytes) else value

        power_mw = get(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        return {
            "driver": get(pynvml.nvmlSystemGetDriverVersion),
            "uuid": get(pynvml.nvmlDeviceGetUUID, handle),
            "power_limit_w": None if power_mw is None else power_mw / 1000,
            "max_sm_clock_mhz": get(pynvml.nvmlDeviceGetMaxClockInfo, handle,
                                    pynvml.NVML_CLOCK_SM),
            "max_mem_clock_mhz": get(pynvml.nvmlDeviceGetMaxClockInfo, handle,
                                     pynvml.NVML_CLOCK_MEM),
            "pcie_gen_max": get(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle),
            "pcie_gen_current": get(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
            "pcie_width_max": get(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle),
            "pcie_width_current": get(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle),
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _ram_total() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def machine_facts() -> dict[str, Any]:
    """The CPU, GPU and NUMA facts, plus ``ram_total_bytes``, the process's CPU affinity
    (Linux only) and an ``nvml`` block (None without ``pynvml`` or a GPU)."""
    r = _Record()
    _cpu(r)
    _gpu(r)
    _numa(r)
    r.put("ram_total_bytes", _ram_total(), "no os.sysconf RAM size on this OS")
    affinity = getattr(os, "sched_getaffinity", None)
    r.put("cpu_affinity", sorted(affinity(0)) if affinity else None,
          "no CPU affinity on this OS")
    r.put("nvml", nvml_facts(), "no pynvml, or no GPU for it to read")
    return {**r.data, "absent": r.absent}
