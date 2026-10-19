"""E0, the machine record: every later number is read against it.

Every field is always present. A field this machine can't supply (no GPU, no NUMA on a Mac)
is ``None``, and ``absent`` maps its name to the reason, so a missing value is never mistaken
for an omitted one. ``notes`` says how to read the fields whose meaning isn't in their name.
"""

from __future__ import annotations

import ctypes
import hashlib
import platform
import re
import shutil
import ssl
import sys
from typing import Any

import blake3
import numpy as np
import torch
import transformers

# The facts themselves live in setup/machine.py; the names are re-exported for the benchmark's
# callers and tests.
from setup.machine import (_Record, _cpu, _frequency, _gpu, _numa, _read, _run,  # noqa: F401
                           cgroup_cpu_quota, cgroup_memory_limit)

def mem_available() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes, or None where there's no such file.
    In a container it describes the host, so the cgroup limit is the tighter bound there."""
    text = _read("/proc/meminfo")
    m = re.search(r"^MemAvailable:\s+(\d+) kB", text or "", re.M)
    return int(m.group(1)) * 1024 if m else None


def cores_for_threads(machine: dict[str, Any]) -> int:
    """Cores the benchmark's threads can use: physical cores, capped by affinity and quota.

    Physical, not logical: a thread count above it would only share cores through SMT. BLAKE3's
    AUTO mode (rayon's global pool) sizes itself by logical CPUs instead, so on an SMT host it
    runs more threads than this count."""
    cores = machine.get("physical_cores") or machine.get("logical_cpus") or 1
    for cap in (machine.get("usable_cpus"), machine.get("cgroup_cpu_quota")):
        if cap:
            cores = min(cores, max(1, int(cap)))
    return int(cores)


def cores_basis(machine: dict[str, Any]) -> str:
    """What :func:`cores_for_threads` counted, in words."""
    basis = ("physical cores" if machine.get("physical_cores")
             else "logical CPUs (no physical core count)")
    caps = [name for name, key in (("CPU affinity", "usable_cpus"),
                                   ("cgroup quota", "cgroup_cpu_quota")) if machine.get(key)]
    return basis + (f", capped by {' and '.join(caps)}" if caps else "")


def hashlib_openssl() -> str | None:
    """The OpenSSL that hashlib's own extension links, asked of that library; ``ssl`` may
    link a different one."""
    try:
        import _hashlib

        fn = ctypes.CDLL(_hashlib.__file__).OpenSSL_version
    except (ImportError, OSError, AttributeError):
        return None
    fn.restype, fn.argtypes = ctypes.c_char_p, [ctypes.c_int]
    out = fn(0)  # OPENSSL_VERSION
    return out.decode() if out else None


def machine_record() -> dict[str, Any]:
    r = _Record()
    r.put("hostname", platform.node())
    r.put("os", platform.platform())
    r.put("python", sys.version.split()[0])
    r.put("gil_enabled", getattr(sys, "_is_gil_enabled", lambda: True)())
    r.put("packages", {"torch": torch.__version__, "transformers": transformers.__version__,
                       "numpy": np.__version__, "blake3": getattr(blake3, "__version__", None)})
    r.put("hashlib_sha256", getattr(hashlib.sha256, "__name__", repr(hashlib.sha256)))
    r.put("hashlib_openssl", hashlib_openssl(),
          "hashlib's _hashlib extension doesn't export OpenSSL_version")
    r.put("ssl_openssl", ssl.OPENSSL_VERSION)
    cli = _run("openssl", "version")
    r.put("openssl_cli", cli.strip() if cli else None, "no openssl on PATH")
    b3 = _run("b3sum", "--version")
    r.put("b3sum", b3.strip() if b3 else None, "no b3sum on PATH")
    r.put("torch_threads", torch.get_num_threads())
    _cpu(r)
    _gpu(r)
    _numa(r)
    r.data["cores_for_threads"] = cores_for_threads(r.data)
    r.data["cores_for_threads_basis"] = cores_basis(r.data)
    r.data["notes"] = {
        "pcie_gen_current": "read at idle, when the GPU may hold its link below pcie_gen_max "
                            "to save power; the link trains up under load",
        "cpu_freq_mhz": "CPU 0's clock, read at idle",
        "cores_for_threads": "physical cores (cores_for_threads_basis); BLAKE3's AUTO mode "
                             "and rayon size their pool by logical_cpus",
        "cgroup_cpu_quota": "the container's CPU time limit, in CPUs (cgroup v2 cpu.max or "
                            "v1 cpu.cfs_quota_us); logical_cpus and physical_cores are the "
                            "host's, as the OS reports them"}
    return {**r.data, "absent": r.absent}
