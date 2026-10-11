"""E1, CPU hashing throughput: SHA-256 (OpenSSL through `hashlib`) and BLAKE3 (`blake3`).

Four measurements, all on one buffer in host RAM filled once with random bytes:

- *Chunked*: the buffer cut into chunks of ``Z`` bytes, each hashed as ``H(0x02 ‖ chunk)``
  (HASHING_PLAN_DRAFT 4.1's chunk leaves), the chunks split evenly over ``T`` Python threads.
- *Whole objects*: objects of each real leaf size of a model's step, hashed as ``H(0x00 ‖
  header ‖ payload)`` with the policy the protocol's ``hash_leaves`` uses today (reproduced
  here, not imported): leaves handed to ``T`` threads in tasks of about 16 MiB, and BLAKE3's
  own multithreading for a leaf of 16 MiB or more. Weighting each size's measured time by its
  count gives a per-step time (an estimate); hashing every leaf of the step in order gives a
  direct one (measured), where it is affordable.
- *Native ceilings*: ``openssl speed -multi`` and ``b3sum --num-threads``.
- *Under a copy*: the chunked measurement while a host-to-GPU copy streams from host RAM.
  Guarded: skipped without CUDA.

Objects are views into the buffer, so no step is ever allocated; an object larger than the
buffer streams the buffer repeatedly.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import torch

from benchmarks.hashing.report import result
from benchmarks.hashing.sizes import Step

ALGOS = ("sha256", "blake3")
# The protocol's hash_leaves policy (verification/commitment/merkle.py), reproduced.
MT_THRESHOLD = 16 << 20
TASK_BYTES = 16 << 20
GB = 1e9

Parts = Sequence[memoryview | bytes]


def make_buffer(nbytes: int, seed: int = 0) -> np.ndarray:
    """``nbytes`` of host RAM, every page written (a 1 MiB random block, repeated)."""
    buf = np.empty(nbytes, dtype=np.uint8)
    block = np.random.default_rng(seed).integers(0, 256, 1 << 20, dtype=np.uint8)
    for lo in range(0, nbytes, block.size):
        hi = min(nbytes, lo + block.size)
        buf[lo:hi] = block[: hi - lo]
    return buf


def thread_counts(cores: int, wanted: Sequence[int | str] = (1, 8, 16, "all")) -> list[int]:
    """The wanted thread counts, each capped at ``cores`` (``"all"`` is ``cores``), deduplicated."""
    out = {min(cores, cores if t == "all" else int(t)) for t in wanted}
    return sorted(t for t in out if t >= 1)


class Pools:
    """One thread pool per thread count, reused across measurements."""

    def __init__(self) -> None:
        self._pools: dict[int, ThreadPoolExecutor] = {}

    def get(self, threads: int) -> ThreadPoolExecutor:
        if threads not in self._pools:
            self._pools[threads] = ThreadPoolExecutor(threads, thread_name_prefix=f"e1_{threads}")
        return self._pools[threads]

    def close(self) -> None:
        for p in self._pools.values():
            p.shutdown(wait=True)
        self._pools.clear()


# ---- hashing as built ---------------------------------------------------------------------


def _hasher(algo: str, nbytes: int) -> Any:
    if algo == "blake3":
        return blake3.blake3(max_threads=blake3.blake3.AUTO) if nbytes >= MT_THRESHOLD \
            else blake3.blake3()
    if algo == "sha256":
        return hashlib.sha256()
    raise ValueError(f"unknown algorithm {algo!r}")


def hash_leaf(algo: str, parts: Parts) -> bytes:
    """``H(0x00 ‖ parts…)``, streamed, BLAKE3 multithreaded from 16 MiB (as built)."""
    h = _hasher(algo, sum(memoryview(p).nbytes for p in parts))
    h.update(b"\x00")
    for p in parts:
        h.update(p)
    return h.digest()


def _hash_task(algo: str, task: list[Parts]) -> list[bytes]:
    return [hash_leaf(algo, parts) for parts in task]


def hash_leaves(algo: str, leaves: Sequence[Parts], threads: int, pools: Pools) -> list[bytes]:
    """Every leaf's ``hash_leaf``, in order: inline on one thread, else in tasks of about
    16 MiB on a pool of ``threads`` (the protocol's ``hash_leaves`` policy)."""
    if threads <= 1:
        return _hash_task(algo, list(leaves))
    pool = pools.get(threads)
    futures, task, size = [], [], 0
    for parts in leaves:
        task.append(parts)
        size += sum(memoryview(p).nbytes for p in parts)
        if size >= TASK_BYTES:
            futures.append(pool.submit(_hash_task, algo, task))
            task, size = [], 0
    if task:
        futures.append(pool.submit(_hash_task, algo, task))
    return [d for f in futures for d in f.result()]


class Views:
    """Objects as views into the buffer, at offsets that walk through it, so consecutive
    measurements don't reuse a cache-warm region."""

    def __init__(self, buf: np.ndarray) -> None:
        self.mv = memoryview(buf)
        self.size = buf.size
        self.offset = 0

    def payload(self, nbytes: int) -> list[memoryview]:
        if nbytes > self.size:  # stream the buffer as many times as needed
            full, rest = divmod(nbytes, self.size)
            return [self.mv] * full + ([self.mv[:rest]] if rest else [])
        if self.offset + nbytes > self.size:
            self.offset = 0
        view = self.mv[self.offset:self.offset + nbytes]
        self.offset += nbytes
        return [view]

    def leaf(self, nbytes: int, header: int) -> list[memoryview | bytes]:
        """A leaf of ``nbytes`` encoded bytes: a ``header``-byte header, then its payload."""
        return [bytes(header), *self.payload(nbytes - header)]


def header_bytes(shape: Sequence[int], section: str) -> int:
    """The encoding's header length: a record's 6 bytes, a tensor's ``3 + 4·ndim``."""
    return 6 if section == "record" else 3 + 4 * len(shape)


# ---- chunked ------------------------------------------------------------------------------


def _hash_chunks(algo: str, mv: memoryview, chunk: int, lo: int, hi: int) -> None:
    ctor = blake3.blake3 if algo == "blake3" else hashlib.sha256
    for i in range(lo, hi):
        h = ctor(b"\x02")
        h.update(mv[i * chunk:(i + 1) * chunk])
        h.digest()


def time_chunked(buf: np.ndarray, algo: str, chunk: int, threads: int, pools: Pools) -> float:
    """Seconds to hash the buffer's ``Z``-byte chunks, split evenly over ``threads``."""
    mv, n = memoryview(buf), buf.size // chunk
    t0 = time.perf_counter()
    if threads <= 1:
        _hash_chunks(algo, mv, chunk, 0, n)
    else:
        bounds = [n * i // threads for i in range(threads + 1)]
        futures = [pools.get(threads).submit(_hash_chunks, algo, mv, chunk, lo, hi)
                   for lo, hi in zip(bounds, bounds[1:])]
        for f in futures:
            f.result()
    return time.perf_counter() - t0


def _rate_values(nbytes: int, times: list[float]) -> tuple[dict, dict]:
    values = {"gb_per_s_best": nbytes / min(times) / GB,
              "gb_per_s_median": nbytes / statistics.median(times) / GB,
              "seconds": times}
    units = {"gb_per_s_best": "GB/s (1e9 bytes/s)", "gb_per_s_median": "GB/s (1e9 bytes/s)",
             "seconds": "s per repetition"}
    return values, units


def chunked(buf: np.ndarray, chunks: Sequence[int], threads: Sequence[int], reps: int,
            pools: Pools, label: str = "chunked") -> list[dict]:
    out = []
    for algo in ALGOS:
        for chunk in chunks:
            for t in threads:
                n = buf.size // chunk
                times = [time_chunked(buf, algo, chunk, t, pools) for _ in range(reps)]
                values, units = _rate_values(n * chunk, times)
                values["us_per_chunk_thread"] = min(times) * t / n * 1e6
                units["us_per_chunk_thread"] = "µs per chunk per thread (best repetition)"
                out.append(result(label, "measured", values=values, units=units,
                                  params={"algo": algo, "chunk_bytes": chunk, "threads": t,
                                          "buffer_bytes": buf.size, "reps": reps,
                                          "chunk_hash": "H(0x02 ‖ chunk)"}))
    return out


# ---- whole objects ------------------------------------------------------------------------


def per_size(step: Step, buf: np.ndarray, threads: Sequence[int], pools: Pools, *,
             sample_bytes: int, object_cap: int | None) -> list[dict]:
    """Each distinct leaf size of the step, hashed as whole objects (as built), and the
    count-weighted per-step time. ``object_cap`` measures a larger object at that size and
    scales its time by size (labelled in the params)."""
    views = Views(buf)
    shapes = {}
    for l in step.leaves:
        shapes.setdefault(l.nbytes, (l.shape, l.section))
    out = []
    for algo in ALGOS:
        for t in threads:
            step_s, rows = 0.0, []
            for size, count in step.histogram():
                shape, section = shapes[size]
                hdr = header_bytes(shape, section)
                measured = size if object_cap is None else min(size, object_cap)
                n_obj = min(count, max(t, math.ceil(sample_bytes / measured)))
                leaves = [views.leaf(measured, hdr) for _ in range(n_obj)]
                t0 = time.perf_counter()
                hash_leaves(algo, leaves, t, pools)
                dt = time.perf_counter() - t0
                per_obj = dt / n_obj * size / measured
                step_s += per_obj * count
                rows.append(result(
                    "whole_objects", "measured",
                    params={"algo": algo, "threads": t, **step.params(), "leaf_bytes": size,
                            "leaf_count": count, "objects_hashed": n_obj,
                            "measured_object_bytes": measured, "example_shape": list(shape)},
                    values={"gb_per_s": n_obj * measured / dt / GB,
                            "seconds_per_object": per_obj},
                    units={"gb_per_s": "GB/s (1e9 bytes/s)",
                           "seconds_per_object": "s, scaled to the full leaf size"}))
            out += rows
            out.append(result(
                "step_from_sizes", "estimated",
                params={"algo": algo, "threads": t, **step.params(),
                        "leaves": len(step.leaves), "step_bytes": step.total_bytes},
                values={"seconds_per_step": step_s, "gb_per_s": step.total_bytes / step_s / GB},
                units={"seconds_per_step": "s", "gb_per_s": "GB/s (1e9 bytes/s)"},
                note="Σ over distinct leaf sizes of count × measured time per object; sizes "
                     "hashed one after another, so no overlap between sizes is counted"))
    return out


def direct_step(step: Step, buf: np.ndarray, threads: Sequence[int], reps: int,
                pools: Pools) -> list[dict]:
    """Every leaf of the step, in leaf order, hashed as built: a measured per-step time."""
    out = []
    for algo in ALGOS:
        for t in threads:
            times = []
            for _ in range(reps):
                views = Views(buf)
                leaves = [views.leaf(l.nbytes, header_bytes(l.shape, l.section))
                          for l in step.leaves]
                t0 = time.perf_counter()
                hash_leaves(algo, leaves, t, pools)
                times.append(time.perf_counter() - t0)
            values, units = _rate_values(step.total_bytes, times)
            values["seconds_per_step_best"] = min(times)
            units["seconds_per_step_best"] = "s"
            out.append(result("step_direct", "measured", values=values, units=units,
                              params={"algo": algo, "threads": t, **step.params(),
                                      "leaves": len(step.leaves),
                                      "step_bytes": step.total_bytes, "reps": reps}))
    return out


# ---- native ceilings ----------------------------------------------------------------------


def _openssl_speed_cmd(threads: int) -> list[str]:
    proc = subprocess.run(["openssl", "speed", "-help"], capture_output=True, text=True)
    help_text = proc.stdout + proc.stderr
    cmd = ["openssl", "speed", "-elapsed", "-mr", "-multi", str(threads)]
    if "-seconds" in help_text:
        cmd += ["-seconds", "1"]
    return cmd + ["-evp", "sha256"]


def parse_openssl_mr(text: str) -> dict[int, float]:
    """``{block bytes: bytes/s}`` from ``openssl speed -mr`` output: the aggregate ``+F`` line
    (the last one not relayed from a child) against the ``+H`` block sizes."""
    header = next((ln for ln in text.splitlines() if "+H:" in ln), None)
    rows = [ln for ln in text.splitlines() if ln.startswith("+F:")]
    if header is None or not rows:
        raise ValueError("no +H/+F lines in openssl speed output")
    # a relayed header reads "Got: +H:16:…:8192 from 0"
    sizes = [int(x) for x in header.split("+H:", 1)[1].split()[0].split(":") if x]
    rates = [float(x) for x in rows[-1].split(":")[3:]]
    return dict(zip(sizes, rates))


def openssl_ceiling(threads: Sequence[int]) -> list[dict]:
    if shutil.which("openssl") is None:
        return [result("openssl_speed", "skipped", reason="no openssl on PATH")]
    out = []
    for t in threads:
        cmd = _openssl_speed_cmd(t)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rates = parse_openssl_mr(proc.stdout + proc.stderr)
        except ValueError as e:
            out.append(result("openssl_speed", "skipped", params={"threads": t, "cmd": cmd},
                              reason=f"{e}; exit {proc.returncode}"))
            continue
        values = {f"gb_per_s_block_{b}": r / GB for b, r in rates.items()}
        out.append(result("openssl_speed", "measured", params={"threads": t, "cmd": cmd,
                                                               "algo": "sha256"},
                          values=values, units=dict.fromkeys(values, "GB/s (1e9 bytes/s)")))
    return out


def b3sum_ceiling(buf: np.ndarray, threads: Sequence[int], reps: int,
                  scratch_dir: str | None) -> list[dict]:
    exe = shutil.which("b3sum")
    if exe is None:
        return [result("b3sum", "skipped", reason="no b3sum on PATH")]
    base = scratch_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    fd, name = tempfile.mkstemp(prefix="e1_b3sum_", dir=base)
    path = Path(name)
    out = []
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(memoryview(buf))
        subprocess.run([exe, str(path)], capture_output=True, check=True)  # warm the page cache
        for t in threads:
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                subprocess.run([exe, "--num-threads", str(t), str(path)], capture_output=True,
                               check=True)
                times.append(time.perf_counter() - t0)
            values, units = _rate_values(buf.size, times)
            out.append(result("b3sum", "measured", values=values, units=units,
                              params={"threads": t, "file_bytes": buf.size, "dir": base,
                                      "reps": reps, "includes": "process start, mmap"}))
    finally:
        path.unlink(missing_ok=True)
    return out


# ---- under a host-to-GPU copy -------------------------------------------------------------


def under_h2d_copy(buf: np.ndarray, chunks: Sequence[int], threads: int, reps: int,
                   pools: Pools, copy_bytes: int) -> list[dict]:
    """Chunked hashing at ``threads`` while a pinned host-to-GPU copy loops. Skipped without
    CUDA."""
    if not torch.cuda.is_available():
        return [result("chunked_under_h2d_copy", "skipped", reason="no CUDA device")]
    host = torch.empty(copy_bytes, dtype=torch.uint8).pin_memory()
    dev = torch.empty(copy_bytes, dtype=torch.uint8, device="cuda")
    stop, copied = threading.Event(), [0]

    def copy_loop() -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            while not stop.is_set():
                dev.copy_(host, non_blocking=True)
                stream.synchronize()
                copied[0] += copy_bytes

    out = []
    copier = threading.Thread(target=copy_loop, daemon=True)
    t_start = time.perf_counter()
    copier.start()
    try:
        out = chunked(buf, chunks, [threads], reps, pools, label="chunked_under_h2d_copy")
    finally:
        stop.set()
        copier.join()
    elapsed = time.perf_counter() - t_start
    out.append(result("h2d_copy_during_hashing", "measured",
                      params={"pinned": True, "copy_bytes": copy_bytes},
                      values={"gb_per_s": copied[0] / elapsed / GB},
                      units={"gb_per_s": "GB/s (1e9 bytes/s), averaged over the hashing"}))
    return out


def run_e1(*, buffer_bytes: int, chunks: Sequence[int], threads: Sequence[int], reps: int,
           size_steps: Sequence[Step], direct_steps: Sequence[Step], direct_reps: int,
           sample_bytes: int,
           object_cap: int | None, native_threads: Sequence[int], native: bool,
           copy_bytes: int, scratch_dir: str | None,
           log: Callable[[str], None] = print) -> list[dict]:
    buf = make_buffer(buffer_bytes)
    pools = Pools()
    out: list[dict] = []
    try:
        log(f"E1 chunked: {buffer_bytes / GB:.2f} GB buffer, chunks {list(chunks)}, "
            f"threads {list(threads)}")
        out += chunked(buf, chunks, threads, reps, pools)
        for s in size_steps:
            log(f"E1 whole objects by size: {s.model} {s.n_s}x{s.n}")
            out += per_size(s, buf, threads, pools, sample_bytes=sample_bytes,
                            object_cap=object_cap)
        for s in direct_steps:
            log(f"E1 whole step, direct: {s.model} {s.n_s}x{s.n} ({s.total_bytes / GB:.1f} GB)")
            out += direct_step(s, buf, threads, direct_reps, pools)
        if native:
            log(f"E1 native ceilings: openssl speed, b3sum at {list(native_threads)} threads")
            out += openssl_ceiling(native_threads)
            out += b3sum_ceiling(buf, native_threads, reps, scratch_dir)
        else:
            out.append(result("openssl_speed", "skipped", reason="native ceilings turned off"))
            out.append(result("b3sum", "skipped", reason="native ceilings turned off"))
        out += under_h2d_copy(buf, chunks, max(threads), reps, pools, copy_bytes)
    finally:
        pools.close()
    return out

