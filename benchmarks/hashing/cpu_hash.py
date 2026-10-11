"""E1, CPU hashing throughput: SHA-256 (OpenSSL through `hashlib`) and BLAKE3 (`blake3`).

The measurements, all on one buffer in host RAM filled once:

- *Chunked*: the buffer cut into chunks of ``Z`` bytes, each hashed as ``H(0x02 ‖ chunk)``
  (HASHING_PLAN_DRAFT 4.1's chunk leaves), the chunks split evenly over ``T`` Python threads.
  Each row also times the chunk tree's node level, ``H(0x01 ‖ l ‖ r)`` over the chunk hashes,
  for an end-to-end rate.
- *Whole objects*: objects of each real leaf size of a model's step, hashed as ``H(0x00 ‖
  header ‖ payload)`` with the policy the protocol's ``hash_leaves`` uses (reproduced here,
  not imported): leaves handed to ``T`` threads in tasks of about 16 MiB. Weighting each
  size's time by its count gives a per-step time (an estimate); hashing every leaf of the step
  in order gives a direct one (measured), where it is affordable. Two BLAKE3 variants:

  - *as built*, the protocol's: a leaf of 16 MiB or more gets BLAKE3's own multithreading on
    rayon's global pool, every logical CPU, whatever ``T`` is. So its ``threads`` label bounds
    only the small leaves' CPU use.
  - *capped*: the same split, but a leaf of 16 MiB or more waits for the tasks before it and
    then hashes on the calling thread with ``blake3(max_threads=T)``. At most ``T`` CPUs hash
    at once, so its per-core scaling is real.

- *The rest of the step's commitment*: the outer Merkle tree over the step's leaf hashes, as
  the prover builds it (every subtree cached) and as the verifier's check 2 does (the root
  only); and the finiteness pass the encoding runs over every float leaf before hashing it.
- *Native ceilings*: ``openssl speed -multi`` and ``b3sum --num-threads``.
- *Under a copy*: the chunked measurement while a host-to-GPU copy streams from host RAM.
  Guarded: skipped without CUDA.

Objects are views into the buffer, so no step is ever allocated; an object larger than the
buffer streams the buffer repeatedly. Layering rule 8 lets a benchmark import only the leaf-size
list from ``verification``, so the tree and the finiteness check are reproduced here too;
`tests/benchmarks/hashing/test_cpu_hash.py` checks each against the protocol's.
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
import traceback
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
# (algorithm, variant) of the whole-object rows. SHA-256 hashes a leaf on one thread either way.
VARIANTS = (("sha256", "as_built"), ("blake3", "as_built"), ("blake3", "capped"))
# The protocol's hash_leaves policy (verification/commitment/merkle.py), reproduced.
MT_THRESHOLD = 16 << 20
TASK_BYTES = 16 << 20
# per_size hashes at least this many tasks per thread, so small leaves keep every thread busy.
TASKS_PER_THREAD = 4
# ... but at most this many objects per measurement: every object costs Python time to make
# and to hash. 2^18 fills 128 threads with the 32 KiB attention products; a leaf of a few KiB
# (a record, a norm weight) fills fewer, and its row says so (``tasks_per_thread``).
MAX_OBJECTS = 1 << 18
# The finiteness rows' object count per leaf size, at most: tiny tensors cost a call each.
MAX_FINITE_OBJECTS = 8192
GB = 1e9

Parts = Sequence[memoryview | bytes]


def make_buffer(nbytes: int, seed: int = 0) -> np.ndarray:
    """``nbytes`` of host RAM, every page written: a 1 MiB random block, repeated.

    Bit 6 of every odd byte is cleared, so any view at an even offset read as fp32, bf16 or
    fp16 is finite: that bit is the exponent's top bit in each format's high byte, and a NaN or
    infinity needs every exponent bit set. The finiteness rows need finite tensors; hashing
    speed doesn't depend on the bytes.
    """
    buf = np.empty(nbytes, dtype=np.uint8)
    block = np.random.default_rng(seed).integers(0, 256, 1 << 20, dtype=np.uint8)
    block[1::2] &= 0xBF
    for lo in range(0, nbytes, block.size):
        hi = min(nbytes, lo + block.size)
        buf[lo:hi] = block[: hi - lo]
    return buf


def thread_counts(cores: int, wanted: Sequence[int | str] = (1, 8, 16, "all")) -> list[int]:
    """The wanted thread counts, each capped at ``cores`` (``"all"`` is ``cores``), deduplicated."""
    out = {min(cores, cores if t == "all" else int(t)) for t in wanted}
    return sorted(t for t in out if t >= 1)


def start_threads(pool: ThreadPoolExecutor, threads: int) -> None:
    """Start all ``threads`` of ``pool`` now, so no timing pays for a thread's start-up: each
    of ``threads`` tasks waits at a barrier until every one of them runs."""
    barrier = threading.Barrier(threads)
    for f in [pool.submit(barrier.wait, 60) for _ in range(threads)]:
        f.result()


class Pools:
    """One thread pool per thread count, its threads started on creation, and one capped
    BLAKE3 hasher per thread count; all reused across measurements."""

    def __init__(self) -> None:
        self._pools: dict[int, ThreadPoolExecutor] = {}
        self._capped: dict[int, Any] = {}

    def get(self, threads: int) -> ThreadPoolExecutor:
        if threads not in self._pools:
            pool = ThreadPoolExecutor(threads, thread_name_prefix=f"e1_{threads}")
            start_threads(pool, threads)
            self._pools[threads] = pool
        return self._pools[threads]

    def capped(self, threads: int) -> Any:
        """A ``blake3(max_threads=threads)`` hasher, built once: each one starts its own rayon
        pool (about 11 µs a thread)."""
        if threads not in self._capped:
            self._capped[threads] = blake3.blake3(max_threads=threads)
        return self._capped[threads]

    def close(self) -> None:
        for p in self._pools.values():
            p.shutdown(wait=True)
        self._pools.clear()
        self._capped.clear()


def variant_params(algo: str, variant: str, threads: int) -> dict[str, Any]:
    """A whole-object row's hashing identity: the algorithm, the variant, and how many CPUs a
    leaf of 16 MiB or more may use."""
    if algo == "sha256" and variant == "as_built":
        return {"algo": algo, "variant": variant, "large_leaf_threads": 1}
    if algo == "blake3" and variant == "as_built":
        return {"algo": algo, "variant": variant,
                "large_leaf_threads": "all logical CPUs: a leaf of 16 MiB or more uses blake3 "
                                      "AUTO (rayon's global pool), whatever `threads` says",
                "logical_cpus": os.cpu_count()}
    if algo == "blake3" and variant == "capped":
        return {"algo": algo, "variant": variant, "large_leaf_threads": threads}
    raise ValueError(f"no variant {variant!r} of {algo!r}")


# ---- hashing as built ---------------------------------------------------------------------


def _hasher(algo: str, nbytes: int) -> Any:
    if algo == "blake3":
        if nbytes >= MT_THRESHOLD:
            return blake3.blake3(max_threads=blake3.blake3.AUTO)
        return blake3.blake3()
    if algo == "sha256":
        return hashlib.sha256()
    raise ValueError(f"unknown algorithm {algo!r}")


def _nbytes(parts: Parts) -> int:
    return sum(memoryview(p).nbytes for p in parts)


def _digest(h: Any, parts: Parts) -> bytes:
    h.update(b"\x00")
    for p in parts:
        h.update(p)
    return h.digest()


def hash_leaf(algo: str, parts: Parts) -> bytes:
    """``H(0x00 ‖ parts…)``, streamed, BLAKE3 on every logical CPU from 16 MiB (as built)."""
    return _digest(_hasher(algo, _nbytes(parts)), parts)


def _hash_task(algo: str, task: list[Parts]) -> list[bytes]:
    return [hash_leaf(algo, parts) for parts in task]


def _hash_task_single(task: list[Parts]) -> list[bytes]:
    return [_digest(blake3.blake3(), parts) for parts in task]


def _as_built(algo: str, leaves: Sequence[Parts], threads: int, pools: Pools) -> list[bytes]:
    if threads <= 1:
        return _hash_task(algo, list(leaves))
    pool = pools.get(threads)
    futures, task, size = [], [], 0
    for parts in leaves:
        task.append(parts)
        size += _nbytes(parts)
        if size >= TASK_BYTES:
            futures.append(pool.submit(_hash_task, algo, task))
            task, size = [], 0
    if task:
        futures.append(pool.submit(_hash_task, algo, task))
    return [d for f in futures for d in f.result()]


def _capped(leaves: Sequence[Parts], threads: int, pools: Pools) -> list[bytes]:
    if threads <= 1:
        return _hash_task_single(list(leaves))
    pool, big = pools.get(threads), pools.capped(threads)
    out: list[bytes] = []
    futures, task, size = [], [], 0

    def drain() -> None:
        for f in futures:
            out.extend(f.result())
        futures.clear()

    for parts in leaves:
        n = _nbytes(parts)
        if n >= MT_THRESHOLD:  # the pool idles while `big` runs its `threads` threads
            if task:
                futures.append(pool.submit(_hash_task_single, task))
                task, size = [], 0
            drain()
            big.reset()
            out.append(_digest(big, parts))
            continue
        task.append(parts)
        size += n
        if size >= TASK_BYTES:
            futures.append(pool.submit(_hash_task_single, task))
            task, size = [], 0
    if task:
        futures.append(pool.submit(_hash_task_single, task))
    drain()
    return out


def hash_leaves(algo: str, leaves: Sequence[Parts], threads: int, pools: Pools,
                variant: str = "as_built") -> list[bytes]:
    """Every leaf's ``H(0x00 ‖ parts…)``, in order. As built: inline on one thread, else in
    tasks of about 16 MiB on a pool of ``threads`` (the protocol's ``hash_leaves`` policy).
    Capped (BLAKE3 only): the same, but no more than ``threads`` CPUs at once (module doc)."""
    if variant == "as_built":
        return _as_built(algo, leaves, threads, pools)
    if variant == "capped" and algo == "blake3":
        return _capped(leaves, threads, pools)
    raise ValueError(f"no variant {variant!r} of {algo!r}")


class Views:
    """Objects as views into the buffer, at offsets that walk through it, so consecutive
    measurements don't reuse a cache-warm region."""

    def __init__(self, buf: np.ndarray) -> None:
        self.buf = buf
        self.mv = memoryview(buf)
        self.size = buf.size
        self.offset = 0

    def _take(self, nbytes: int) -> int:
        if self.offset + nbytes > self.size:
            self.offset = 0
        lo = self.offset
        self.offset += nbytes + (-nbytes % 64)  # the next view starts 64-byte aligned
        return lo

    def payload(self, nbytes: int) -> list[memoryview]:
        if nbytes > self.size:  # stream the buffer as many times as needed
            full, rest = divmod(nbytes, self.size)
            return [self.mv] * full + ([self.mv[:rest]] if rest else [])
        lo = self._take(nbytes)
        return [self.mv[lo:lo + nbytes]]

    def leaf(self, nbytes: int, header: int) -> list[memoryview | bytes]:
        """A leaf of ``nbytes`` encoded bytes: a ``header``-byte header, then its payload."""
        return [bytes(header), *self.payload(nbytes - header)]

    def tensor(self, dtype: torch.dtype, numel: int) -> torch.Tensor:
        """A finite ``dtype`` tensor of ``numel`` entries over the buffer (no copy)."""
        lo = self._take(numel * dtype.itemsize)
        return torch.frombuffer(self.buf, dtype=dtype, count=numel, offset=lo)


def header_bytes(shape: Sequence[int], section: str) -> int:
    """The encoding's header length: a record's 6 bytes, a tensor's ``3 + 4·ndim``."""
    return 6 if section == "record" else 3 + 4 * len(shape)


# ---- trees and the finiteness check, reproduced -------------------------------------------


def node_hash(left: bytes, right: bytes) -> bytes:
    """``BLAKE3(0x01 ‖ left ‖ right)``, the protocol's ``hash_node``."""
    return blake3.blake3(b"\x01" + left + right).digest()


def _sha256_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


NODES: dict[str, Callable[[bytes, bytes], bytes]] = {"blake3": node_hash, "sha256": _sha256_node}


def _split(n: int) -> int:
    """Largest power of two strictly below ``n`` (n ≥ 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def _check_hashes(hashes: Sequence[bytes]) -> None:
    if len(hashes) == 0:
        raise ValueError("a Merkle tree needs at least one leaf")
    for x in hashes:
        if type(x) is not bytes or len(x) != 32:
            raise ValueError("leaf hashes must be 32-byte bytes objects")


def tree_root(hashes: Sequence[bytes], node: Callable[[bytes, bytes], bytes] = node_hash
              ) -> bytes:
    """The RFC 6962 root over ``hashes`` (the protocol's ``merkle_root`` shape, P9a's
    promotion), no subtree kept."""

    def mth(lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            return hashes[lo]
        k = _split(hi - lo)
        return node(mth(lo, lo + k), mth(lo + k, hi))

    return mth(0, len(hashes))


def merkle_root(hashes: Sequence[bytes]) -> bytes:
    """The protocol's ``merkle_root``: the verifier's check 2 builds the step's root so."""
    _check_hashes(hashes)
    return tree_root(hashes)


def cached_tree(hashes: Sequence[bytes]) -> dict[tuple[int, int], bytes]:
    """The protocol's ``MerkleTree`` build: every subtree's hash, keyed by its leaf range.
    The prover's ``commit_leaves`` builds the step's tree so; the root is ``[(0, n)]``."""
    _check_hashes(hashes)
    leaves, nodes = list(hashes), {}

    def build(lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            h = leaves[lo]
        else:
            k = _split(hi - lo)
            h = node_hash(build(lo, lo + k), build(lo + k, hi))
        nodes[(lo, hi)] = h
        return h

    build(0, len(leaves))
    return nodes


def all_finite(t: torch.Tensor) -> bool:
    """The encoding's ``_all_finite``: fp32 and fp16 CPU tensors reduce with numpy's ``max``
    and ``min`` (single-threaded), others with ``torch.aminmax`` (on ``torch``'s threads)."""
    if t.numel() == 0:
        return True
    if t.device.type == "cpu" and t.dtype in (torch.float32, torch.float16):
        v = t.detach().numpy()
        return bool(np.isfinite(v.max()) and np.isfinite(v.min()))
    lo, hi = torch.aminmax(t)
    return bool(lo.isfinite() & hi.isfinite())


def finite_path(dtype: torch.dtype) -> str:
    return "numpy max and min" if dtype in (torch.float32, torch.float16) else "torch.aminmax"


def _median_best(times: list[float]) -> tuple[float, float]:
    return statistics.median(times), min(times)


def random_digests(n: int) -> list[bytes]:
    return [os.urandom(32) for _ in range(n)]


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
    pool = pools.get(threads) if threads > 1 else None
    t0 = time.perf_counter()
    if pool is None:
        _hash_chunks(algo, mv, chunk, 0, n)
    else:
        bounds = [n * i // threads for i in range(threads + 1)]
        futures = [pool.submit(_hash_chunks, algo, mv, chunk, lo, hi)
                   for lo, hi in zip(bounds, bounds[1:])]
        for f in futures:
            f.result()
    return time.perf_counter() - t0


def time_chunk_tree(n: int, algo: str, reps: int) -> list[float]:
    """Seconds to build the root over ``n`` chunk hashes, node by node, on one thread."""
    hashes, node = random_digests(n), NODES[algo]
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        tree_root(hashes, node)
        times.append(time.perf_counter() - t0)
    return times


def _rate_values(nbytes: int, times: list[float]) -> tuple[dict, dict]:
    values = {"gb_per_s_best": nbytes / min(times) / GB,
              "gb_per_s_median": nbytes / statistics.median(times) / GB,
              "seconds": times}
    units = {"gb_per_s_best": "GB/s (1e9 bytes/s)", "gb_per_s_median": "GB/s (1e9 bytes/s)",
             "seconds": "s per repetition"}
    return values, units


def chunked(buf: np.ndarray, chunks: Sequence[int], threads: Sequence[int], reps: int,
            pools: Pools, label: str = "chunked", tree: bool = True) -> list[dict]:
    """The chunk rows. With ``tree``, each also carries the node level's time (one thread, in
    Python) and the end-to-end rate it leaves: bytes over the median chunk time plus it."""
    out = []
    for algo in ALGOS:
        for chunk in chunks:
            n = buf.size // chunk
            node_s = statistics.median(time_chunk_tree(n, algo, reps)) if tree else None
            for t in threads:
                times = [time_chunked(buf, algo, chunk, t, pools) for _ in range(reps)]
                values, units = _rate_values(n * chunk, times)
                values["us_per_chunk_thread"] = min(times) * t / n * 1e6
                units["us_per_chunk_thread"] = "µs per chunk per thread (best repetition)"
                if node_s is not None:
                    values |= {"node_seconds": node_s, "node_us_per_chunk": node_s / n * 1e6,
                               "gb_per_s_end_to_end":
                                   n * chunk / (statistics.median(times) + node_s) / GB}
                    units |= {"node_seconds": "s, the n − 1 node hashes over the chunk "
                                              "hashes (median, one thread, Python)",
                              "node_us_per_chunk": "µs of node hashing per chunk",
                              "gb_per_s_end_to_end": "GB/s (1e9 bytes/s): median chunk "
                                                     "time plus the node level"}
                out.append(result(label, "measured", values=values, units=units,
                                  params={"algo": algo, "chunk_bytes": chunk, "threads": t,
                                          "chunks": n, "buffer_bytes": buf.size, "reps": reps,
                                          "chunk_hash": "H(0x02 ‖ chunk), one thread each",
                                          "node_hash": "H(0x01 ‖ l ‖ r)" if tree else None}))
    return out


# ---- whole objects ------------------------------------------------------------------------


def sample_objects(measured: int, count: int, threads: int, sample_bytes: int,
                   max_objects: int = MAX_OBJECTS) -> tuple[int, int]:
    """``(objects, tasks)`` per_size hashes for one leaf size. A leaf under 16 MiB shares a
    task with others: hash at least ``TASKS_PER_THREAD × threads`` tasks' worth and at least
    ``sample_bytes``, whatever the step's count, but no more than ``max_objects``. A larger
    leaf is a task of its own: hash ``count`` of them, fewer when ``sample_bytes`` and
    ``TASKS_PER_THREAD × threads`` both ask for fewer, and at least one."""
    if measured < TASK_BYTES:
        per_task = math.ceil(TASK_BYTES / measured)
        n = max(TASKS_PER_THREAD * threads * per_task, math.ceil(sample_bytes / measured))
        n = max(1, min(n, max_objects))
        return n, math.ceil(n / per_task)
    n = min(count, max(TASKS_PER_THREAD * threads, math.ceil(sample_bytes / measured)))
    return max(1, n), max(1, n)


def per_size(step: Step, buf: np.ndarray, threads: Sequence[int], pools: Pools, *,
             reps: int, sample_bytes: int, object_cap: int | None,
             max_objects: int = MAX_OBJECTS,
             variants: Sequence[tuple[str, str]] = VARIANTS) -> list[dict]:
    """Each distinct leaf size of the step, hashed as whole objects, and the count-weighted
    per-step time. Each size gets one untimed warm-up pass (about one task per thread, one
    object for a leaf of 16 MiB or more), then ``reps`` timed ones; the rates and the step
    sum use the median. ``object_cap`` measures a larger object at that size and scales its
    time by size; such a row is "estimated". :func:`sample_objects` sets the object count."""
    views = Views(buf)
    shapes = {}
    for leaf in step.leaves:
        shapes.setdefault(leaf.nbytes, (leaf.shape, leaf.section))
    out = []
    for algo, variant in variants:
        for t in threads:
            step_s = 0.0
            for size, count in step.histogram():
                shape, section = shapes[size]
                hdr = header_bytes(shape, section)
                measured = size if object_cap is None else min(size, object_cap)
                n_obj, tasks = sample_objects(measured, count, t, sample_bytes, max_objects)
                warm = min(n_obj, t * math.ceil(TASK_BYTES / measured)) \
                    if measured < TASK_BYTES else 1
                hash_leaves(algo, [views.leaf(measured, hdr) for _ in range(warm)], t, pools,
                            variant)
                times, cpu = [], []
                for _ in range(reps):
                    leaves = [views.leaf(measured, hdr) for _ in range(n_obj)]
                    c0, t0 = time.process_time(), time.perf_counter()
                    hash_leaves(algo, leaves, t, pools, variant)
                    times.append(time.perf_counter() - t0)
                    cpu.append(time.process_time() - c0)
                med, best = _median_best(times)
                scale = size / measured
                step_s += med / n_obj * scale * count
                estimated = measured < size
                out.append(result(
                    "whole_objects", "estimated" if estimated else "measured",
                    params={**variant_params(algo, variant, t), "threads": t, **step.params(),
                            "leaf_bytes": size, "leaf_count": count, "objects_hashed": n_obj,
                            "tasks": tasks, "tasks_per_thread": tasks / t,
                            "warmup_objects": warm, "reps": reps,
                            "measured_object_bytes": measured, "example_shape": list(shape)},
                    values={"gb_per_s": n_obj * measured / med / GB,
                            "gb_per_s_best": n_obj * measured / best / GB,
                            "seconds_per_object": med / n_obj * scale,
                            "seconds_per_object_best": best / n_obj * scale,
                            "cpu_per_wall": sum(cpu) / sum(times),
                            "seconds": times},
                    units={"gb_per_s": "GB/s (1e9 bytes/s), median repetition",
                           "gb_per_s_best": "GB/s (1e9 bytes/s), best repetition",
                           "seconds_per_object": "s, median, scaled to the full leaf size",
                           "seconds_per_object_best": "s, best, scaled to the full leaf size",
                           "cpu_per_wall": "process CPU seconds per wall second, all "
                                           "repetitions (≈ CPUs busy)",
                           "seconds": "s per repetition, all objects"},
                    note=(f"hashed at {measured} bytes and scaled by size: the leaf is larger "
                          "than the buffer or the object cap") if estimated else None))
            out.append(result(
                "step_from_sizes", "estimated",
                params={**variant_params(algo, variant, t), "threads": t, **step.params(),
                        "leaves": len(step.leaves), "step_bytes": step.total_bytes},
                values={"seconds_per_step": step_s, "gb_per_s": step.total_bytes / step_s / GB},
                units={"seconds_per_step": "s", "gb_per_s": "GB/s (1e9 bytes/s)"},
                note="Σ over distinct leaf sizes of count × median time per object; sizes "
                     "hashed one after another, so no overlap between sizes is counted"))
    return out


def direct_step(step: Step, buf: np.ndarray, threads: Sequence[int], reps: int,
                pools: Pools, variants: Sequence[tuple[str, str]] = VARIANTS) -> list[dict]:
    """Every leaf of the step, in leaf order, hashed whole: a measured per-step time."""
    out = []
    for algo, variant in variants:
        for t in threads:
            times, cpu = [], []
            for _ in range(reps):
                views = Views(buf)
                leaves = [views.leaf(leaf.nbytes, header_bytes(leaf.shape, leaf.section))
                          for leaf in step.leaves]
                c0, t0 = time.process_time(), time.perf_counter()
                hash_leaves(algo, leaves, t, pools, variant)
                times.append(time.perf_counter() - t0)
                cpu.append(time.process_time() - c0)
            values, units = _rate_values(step.total_bytes, times)
            values |= {"seconds_per_step_best": min(times),
                       "seconds_per_step_median": statistics.median(times),
                       "cpu_per_wall": sum(cpu) / sum(times)}
            units |= {"seconds_per_step_best": "s", "seconds_per_step_median": "s",
                      "cpu_per_wall": "process CPU seconds per wall second (≈ CPUs busy)"}
            out.append(result("step_direct", "measured", values=values, units=units,
                              params={**variant_params(algo, variant, t), "threads": t,
                                      **step.params(), "leaves": len(step.leaves),
                                      "step_bytes": step.total_bytes, "reps": reps}))
    return out


# ---- the outer tree and the finiteness pass -----------------------------------------------


def outer_tree(step: Step, reps: int) -> list[dict]:
    """The step's outer Merkle tree over its ``len(step.leaves)`` leaf hashes (random digests
    stand in), built as the prover's ``commit_leaves`` does and as check 2 does."""
    n = len(step.leaves)
    hashes = random_digests(n)
    out = []
    for who, build, what in (
            ("prover", cached_tree, "MerkleTree: every subtree's hash cached (commit_leaves)"),
            ("verifier", merkle_root, "merkle_root: the root only (check 2)")):
        times = []
        for _ in range(reps + 1):  # the first is a warm-up
            t0 = time.perf_counter()
            build(hashes)
            times.append(time.perf_counter() - t0)
        times = times[1:]
        med, best = _median_best(times)
        out.append(result(
            "outer_tree", "measured",
            params={**step.params(), "builder": who, "tree": what, "leaves": n, "reps": reps,
                    "node_hash": "BLAKE3(0x01 ‖ l ‖ r)"},
            values={"seconds": med, "seconds_best": best, "node_hashes": n - 1,
                    "us_per_node": med / max(1, n - 1) * 1e6, "seconds_all": times},
            units={"seconds": "s, median", "seconds_best": "s, best", "node_hashes": "count",
                   "us_per_node": "µs per node hash (median)", "seconds_all": "s per repetition"},
            note="the protocol's tree code reproduced (layering rule 8 bars importing it); "
                 "tests check its roots against the protocol's. One thread, Python, after "
                 "every leaf is hashed"))
    return out


def finiteness(step: Step, buf: np.ndarray, threads: Sequence[int], *, reps: int,
               sample_bytes: int, object_cap: int | None) -> list[dict]:
    """The encoding's finiteness check over each distinct (size, dtype) of the step's float
    leaves, at each thread count (``torch.set_num_threads``, restored after), and the
    count-weighted per-step time. Tensors are finite views into the buffer; one larger than
    the buffer or ``object_cap`` is checked at that size and scaled."""
    views = Views(buf)
    kinds: dict[tuple[int, torch.dtype], tuple[int, tuple[int, ...]]] = {}
    for leaf in step.leaves:
        if leaf.dtype.is_floating_point:
            key = (math.prod(leaf.shape), leaf.dtype)
            count, _ = kinds.get(key, (0, leaf.shape))
            kinds[key] = (count + 1, leaf.shape)
    out = []
    before = torch.get_num_threads()
    try:
        for t in threads:
            torch.set_num_threads(t)
            step_s, checked = 0.0, 0
            for (numel, dtype), (count, shape) in sorted(kinds.items(), key=lambda kv: -kv[0][0]):
                size = numel * dtype.itemsize
                cap = min(buf.size, object_cap or buf.size) // dtype.itemsize
                m = min(numel, cap)
                n_obj = math.ceil(sample_bytes / (m * dtype.itemsize))
                n_obj = max(1, min(MAX_FINITE_OBJECTS, n_obj))
                all_finite(views.tensor(dtype, m))  # warm-up
                times = []
                for _ in range(reps):
                    ts = [views.tensor(dtype, m) for _ in range(n_obj)]
                    t0 = time.perf_counter()
                    ok = all(all_finite(x) for x in ts)
                    times.append(time.perf_counter() - t0)
                    if not ok:
                        raise RuntimeError("a buffer view is not finite (make_buffer's bit)")
                med, best = _median_best(times)
                per_obj = med / n_obj * numel / m
                step_s += per_obj * count
                checked += size * count
                estimated = m < numel
                out.append(result(
                    "finiteness", "estimated" if estimated else "measured",
                    params={"threads": t, **step.params(), "leaf_payload_bytes": size,
                            "dtype": str(dtype).removeprefix("torch."), "path": finite_path(dtype),
                            "leaf_count": count, "example_shape": list(shape),
                            "objects_checked": n_obj, "measured_numel": m, "reps": reps},
                    values={"seconds_per_object": per_obj,
                            "seconds_per_object_best": best / n_obj * numel / m,
                            "gb_per_s": n_obj * m * dtype.itemsize / med / GB},
                    units={"seconds_per_object": "s, median, scaled to the full leaf",
                           "seconds_per_object_best": "s, best, scaled to the full leaf",
                           "gb_per_s": "GB/s (1e9 bytes/s) of payload, median"},
                    note=(f"checked at {m} entries and scaled by size: the leaf is larger than "
                          "the buffer or the object cap") if estimated else None))
            out.append(result(
                "finiteness_per_step", "estimated",
                params={"threads": t, **step.params(), "leaves_checked":
                        sum(c for c, _ in kinds.values()), "payload_bytes": checked},
                values={"seconds_per_step": step_s,
                        "gb_per_s": checked / step_s / GB if step_s else None},
                units={"seconds_per_step": "s", "gb_per_s": "GB/s (1e9 bytes/s) of payload"},
                note="Σ over distinct (size, dtype) of count × median time per tensor. The "
                     "protocol's `_all_finite`, reproduced (layering rule 8 bars importing "
                     "it; tests check it against the protocol's). It runs on the calling "
                     "thread while the workers hash, so it overlaps the hashing: a step's "
                     "commit takes at least this long. Measured alone, not under hashing"))
    finally:
        torch.set_num_threads(before)
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
        out = chunked(buf, chunks, [threads], reps, pools, label="chunked_under_h2d_copy",
                      tree=False)
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
           copy_bytes: int, scratch_dir: str | None, max_objects: int = MAX_OBJECTS,
           log: Callable[[str], None] = print) -> list[dict]:
    """Every E1 part in turn. A part that raises leaves a skipped row with the error and the
    others run on."""
    buf = make_buffer(buffer_bytes)
    pools = Pools()
    out: list[dict] = []

    def part(name: str, what: str, fn: Callable[[], list[dict]], **params: Any) -> None:
        log(what)
        try:
            out.extend(fn())
        except Exception as e:
            log(f"E1 {name} failed:\n{traceback.format_exc()}")
            out.append(result(name, "skipped", params=params,
                              reason=f"error: {type(e).__name__}: {e}"))

    try:
        part("chunked", f"E1 chunked: {buffer_bytes / GB:.2f} GB buffer, chunks {list(chunks)}, "
             f"threads {list(threads)}", lambda: chunked(buf, chunks, threads, reps, pools))
        for s in size_steps:
            tag = f"{s.model} {s.n_s}x{s.n}"
            part("whole_objects", f"E1 whole objects by size: {tag}",
                 lambda s=s: per_size(s, buf, threads, pools, reps=reps,
                                      sample_bytes=sample_bytes, object_cap=object_cap,
                                      max_objects=max_objects),
                 **s.params())
            part("outer_tree", f"E1 outer tree: {tag}", lambda s=s: outer_tree(s, reps),
                 **s.params())
            part("finiteness", f"E1 finiteness: {tag}",
                 lambda s=s: finiteness(s, buf, threads, reps=reps, sample_bytes=sample_bytes,
                                        object_cap=object_cap), **s.params())
        for s in direct_steps:
            part("step_direct", f"E1 whole step, direct: {s.model} {s.n_s}x{s.n} "
                 f"({s.total_bytes / GB:.1f} GB)",
                 lambda s=s: direct_step(s, buf, threads, direct_reps, pools), **s.params())
        if native:
            part("openssl_speed", f"E1 native ceilings: openssl speed, b3sum at "
                 f"{list(native_threads)} threads", lambda: openssl_ceiling(native_threads))
            part("b3sum", "E1 b3sum", lambda: b3sum_ceiling(buf, native_threads, reps,
                                                            scratch_dir))
        else:
            out.append(result("openssl_speed", "skipped", reason="native ceilings turned off"))
            out.append(result("b3sum", "skipped", reason="native ceilings turned off"))
        part("chunked_under_h2d_copy", "E1 chunked under a host-to-GPU copy",
             lambda: under_h2d_copy(buf, chunks, max(threads), reps, pools, copy_bytes))
    finally:
        pools.close()
    return out
