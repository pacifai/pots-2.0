"""The GPU experiments: E3's GEMM loop and its overlap, E4's host-GPU copies, and the slots
that need GPU hashing (E2, E5b, E5c).

- *GEMM loop* (E3, and E4's load): bf16 matmuls of a model's largest training shapes, round
  robin, enqueued from a background thread onto their own CUDA stream, at most ``DEPTH``
  ahead of the GPU. Its rate over a window is the FLOPs of the GEMMs that finished in it over
  the window's wall time. ``setup_determinism``'s matmul precision settings hold throughout
  (``step_time.matmul_precision``).
- *Copies* (E4): GPU-to-host and host-to-GPU, between one device buffer of a piece's size
  and the pieces of a host buffer of the whole total, so each piece is distinct host memory
  as a step's transcript is. The host buffer is allocated pageable, every page written, and
  pinned in place with ``cudaHostRegister`` for the pinned rows. A pass copies every piece once
  on a side stream. Each copy is timed alone, then while the GEMM loop runs, then while CPU
  threads hash as E1's chunked rows do.
- *GPU hashing* (E2, E3's other half, E5b, E5c): not here. cuPQC is behind a licence the user
  hasn't accepted, and the cited BLAKE3-on-GPU proxy needs a SYCL compiler, not nvcc; the
  rows say so (``CUPQC_REASON``, ``BLAKE3_GPU_REASON``).

Every GPU row needs CUDA; without it the slot says skipped.
"""

from __future__ import annotations

import collections
import contextlib
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.hashing import cpu_hash
from benchmarks.hashing.report import result
from benchmarks.hashing.step_time import matmul_precision

GB = 1e9
# GEMMs the loop may have enqueued and not yet seen finish: enough to keep the GPU fed, few
# enough that a window's FLOP count lags the GPU by at most this many GEMMs.
DEPTH = 4
NO_CUDA = "no CUDA device"

CUPQC_REASON = (
    "needs NVIDIA's cuPQC SDK, a CUDA device library (no pip or conda package; a 189 MB "
    "tarball at developer.download.nvidia.com/compute/cupqc/redist/cupqc/). Its download page "
    "says that downloading and using it accepts the NVIDIA Software License Agreement "
    "(docs.nvidia.com/cuda/cupqc/additional/license.html); accepting it is the user's call, so "
    "it was not downloaded and no kernel was written against it")
BLAKE3_GPU_REASON = (
    "the cited BLAKE3-on-GPU implementation (github.com/itzmeanjan/blake3, the plan's 'BLAKE3 "
    "on GPGPU' source) is SYCL: on an NVIDIA GPU it builds with Intel's open-source LLVM SYCL "
    "compiler, built from source with its CUDA backend, not with nvcc; not built")
NO_GPU_HASH = ("no GPU hash is available on this machine (E2: cuPQC not downloaded, BLAKE3 "
               "proxy not built)")


# ---- the GEMM loop ------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemm:
    """One matmul ``(m × k) @ (k × n)`` of a training step's forward pass."""

    name: str
    m: int
    k: int
    n: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.k * self.n

    def params(self) -> dict[str, Any]:
        return {"gemm": self.name, "m": self.m, "k": self.k, "n": self.n}


def gemm_shapes(config: Any, tokens: int) -> list[Gemm]:
    """A model's largest forward GEMMs over ``tokens`` rows: the MLP's up (and gate)
    projection, its down projection, and the LM head. Attention's projections are no larger
    than the MLP's, and the backward pass repeats these shapes transposed."""
    h, i, v = config.hidden_size, config.intermediate_size, config.vocab_size
    return [Gemm("mlp_up", tokens, h, i), Gemm("mlp_down", tokens, i, h),
            Gemm("lm_head", tokens, h, v)]


def _operands(g: Gemm, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(g.m, g.k, device=device, dtype=torch.bfloat16, generator=gen)
    b = torch.randn(g.k, g.n, device=device, dtype=torch.bfloat16, generator=gen)
    return a, b, torch.empty(g.m, g.n, device=device, dtype=torch.bfloat16)


def time_gemm(g: Gemm, device: torch.device, *, iters: int, warmup: int = 3) -> list[float]:
    """Seconds per GEMM, from CUDA events around each of ``iters`` runs after ``warmup``."""
    a, b, c = _operands(g, device, seed=0)
    for _ in range(warmup):
        torch.mm(a, b, out=c)
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return times


class GemmLoop:
    """The shapes' GEMMs in turn, on their own stream, until stopped. ``count()`` gives the
    FLOPs finished so far; two readings around a window give its rate."""

    def __init__(self, gemms: Sequence[Gemm], device: torch.device) -> None:
        self.gemms = list(gemms)
        self.operands = [_operands(g, device, seed=i) for i, g in enumerate(self.gemms)]
        self.stream = torch.cuda.Stream(device=device)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._flops = 0
        self._done = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="gemm_loop")

    def _run(self) -> None:
        pending: collections.deque = collections.deque()

        def finish_one() -> None:
            event, flops = pending.popleft()
            event.synchronize()
            with self._lock:
                self._flops += flops
                self._done += 1

        try:
            torch.cuda.set_device(self.stream.device)  # a new thread has no current context
            with torch.cuda.stream(self.stream):
                while not self._stop.is_set():
                    for g, (a, b, c) in zip(self.gemms, self.operands):
                        torch.mm(a, b, out=c)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                        pending.append((event, g.flops))
                        while len(pending) > DEPTH:
                            finish_one()
                while pending:
                    finish_one()
        except BaseException as e:  # surfaced by stop()
            self._error = e

    def __enter__(self) -> GemmLoop:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def count(self) -> tuple[float, int, int]:
        """``(time, FLOPs finished, GEMMs finished)`` now."""
        with self._lock:
            return time.perf_counter(), self._flops, self._done

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        if self._error is not None:
            raise RuntimeError("the GEMM loop failed") from self._error


def window_rate(before: tuple[float, int, int], after: tuple[float, int, int]
                ) -> dict[str, float]:
    """The loop's rate between two ``count()`` readings."""
    seconds = after[0] - before[0]
    return {"tflop_per_s": (after[1] - before[1]) / seconds / 1e12,
            "gemms": after[2] - before[2], "window_s": seconds}


def warm(loop: GemmLoop, timeout: float = 120) -> None:
    """Wait for the loop's first round, untimed: cuBLAS picks its kernels on first use."""
    t_end = time.perf_counter() + timeout
    while loop.count()[2] < len(loop.gemms):
        if time.perf_counter() > t_end:
            raise TimeoutError(f"the GEMM loop finished no round in {timeout} s")
        time.sleep(0.005)


def loop_alone(loop: GemmLoop, seconds: float) -> dict[str, float]:
    """The loop's rate over ``seconds`` with nothing else on the GPU, after its first round."""
    warm(loop)
    before = loop.count()
    time.sleep(seconds)
    return window_rate(before, loop.count())


GEMM_UNITS = {"tflop_per_s": "TFLOP/s (1e12 FLOP/s), bf16 inputs, fp32 accumulation",
              "gemms": "GEMMs finished in the window", "window_s": "s"}


def run_e3(gemm_sets: dict[str, list[Gemm]], *, window_s: float, iters: int,
           log: Callable[[str], None] = print) -> list[dict]:
    """E3: each shape's rate, the loop's rate alone, and the loop beside GPU hashing (skipped
    here: no GPU hash)."""
    if not torch.cuda.is_available():
        return [result("gemm", "skipped", reason=NO_CUDA)]
    device = torch.device("cuda")
    out = []
    with matmul_precision() as precision:
        for model, gemms in gemm_sets.items():
            for g in gemms:
                log(f"E3 GEMM {model} {g.name} {g.m}x{g.k}x{g.n}")
                times = time_gemm(g, device, iters=iters)
                med = statistics.median(times)
                out.append(result(
                    "gemm", "measured",
                    params={"model": model, **g.params(), "iters": iters, **precision},
                    values={"tflop_per_s_median": g.flops / med / 1e12,
                            "tflop_per_s_best": g.flops / min(times) / 1e12,
                            "seconds_median": med},
                    units={"tflop_per_s_median": GEMM_UNITS["tflop_per_s"],
                           "tflop_per_s_best": GEMM_UNITS["tflop_per_s"],
                           "seconds_median": "s per GEMM (CUDA events)"}))
            log(f"E3 GEMM loop alone: {model}, {window_s} s")
            with GemmLoop(gemms, device) as loop:
                rate = loop_alone(loop, window_s)
            out.append(result("gemm_loop_alone", "measured",
                              params={"model": model, "shapes": [g.params() for g in gemms],
                                      "depth": DEPTH, **precision},
                              values=rate, units=GEMM_UNITS))
            torch.cuda.empty_cache()
    out.append(result("gpu_hash_alone", "skipped", reason=NO_GPU_HASH))
    out.append(result("gemm_loop_with_gpu_hash", "skipped", reason=NO_GPU_HASH))
    return out


# ---- copies -------------------------------------------------------------------------------


def _check(err: Any, what: str) -> None:
    code = getattr(err, "value", err)
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA error {err}")


class HostBuffer:
    """``total`` bytes of pageable host RAM, every page written, cut into pieces; ``pinned()``
    page-locks it in place for a block, so pinned and pageable rows copy the same memory."""

    def __init__(self, total: int) -> None:
        self.tensor = torch.zeros(total, dtype=torch.uint8)

    def pieces(self, piece: int) -> list[torch.Tensor]:
        if self.tensor.numel() % piece:
            raise ValueError(f"total {self.tensor.numel()} is not a multiple of piece {piece}")
        return list(self.tensor.split(piece))

    def pinned(self) -> _Pinned:
        return _Pinned(self.tensor)


class _Pinned:
    def __init__(self, t: torch.Tensor) -> None:
        self.t = t

    def __enter__(self) -> None:
        rt = torch.cuda.cudart()
        _check(rt.cudaHostRegister(self.t.data_ptr(), self.t.numel(), 0), "cudaHostRegister")
        if not self.t.is_pinned():
            rt.cudaHostUnregister(self.t.data_ptr())
            raise RuntimeError("cudaHostRegister succeeded but the buffer isn't pinned")

    def __exit__(self, *exc: object) -> None:
        _check(torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr()), "cudaHostUnregister")


def copy_pass(pieces: Sequence[torch.Tensor], dev: torch.Tensor, direction: str,
              stream: torch.cuda.Stream) -> float:
    """Seconds to copy every piece once, in order, on ``stream``: ``h2d`` into ``dev`` or
    ``d2h`` out of it. It waits for the current stream's work first, not the whole device's:
    a device-wide sync would wait for the GEMM loop's queue and count that as copy time."""
    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for p in pieces:
            d = dev[: p.numel()]
            if direction == "h2d":
                d.copy_(p, non_blocking=True)
            else:
                p.copy_(d, non_blocking=True)
    stream.synchronize()
    return time.perf_counter() - t0


def copy_window(pieces: Sequence[torch.Tensor], dev: torch.Tensor, direction: str,
                stream: torch.cuda.Stream, min_s: float) -> tuple[int, float]:
    """``(bytes, seconds)`` of whole passes, repeated until ``min_s`` has passed."""
    nbytes = sum(p.numel() for p in pieces)
    copied, t0 = 0, time.perf_counter()
    while True:
        copy_pass(pieces, dev, direction, stream)
        copied += nbytes
        if time.perf_counter() - t0 >= min_s:
            return copied, time.perf_counter() - t0


class HashLoop:
    """E1's chunked hashing, ``threads`` Python threads hashing ``H(0x02 ‖ chunk)`` over the
    buffer in 16 MiB tasks, round and round, until stopped. ``count()`` gives the bytes hashed."""

    def __init__(self, buf: np.ndarray, algo: str, chunk: int, threads: int) -> None:
        self.buf, self.algo, self.chunk, self.threads = buf, algo, chunk, threads
        self.per_task = max(1, cpu_hash.TASK_BYTES // chunk)
        self.n_chunks = buf.size // chunk
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._bytes = 0
        self._next = 0
        self._workers = [threading.Thread(target=self._run, daemon=True, name=f"hash_{i}")
                         for i in range(threads)]

    def _run(self) -> None:
        mv = memoryview(self.buf)
        while not self._stop.is_set():
            with self._lock:
                lo = self._next
                self._next = (lo + self.per_task) % self.n_chunks
            hi = min(lo + self.per_task, self.n_chunks)
            cpu_hash._hash_chunks(self.algo, mv, self.chunk, lo, hi)
            with self._lock:
                self._bytes += (hi - lo) * self.chunk

    def __enter__(self) -> HashLoop:
        for w in self._workers:
            w.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for w in self._workers:
            w.join()

    def count(self) -> tuple[float, int]:
        with self._lock:
            return time.perf_counter(), self._bytes


def _hash_rate(before: tuple[float, int], after: tuple[float, int]) -> float:
    return (after[1] - before[1]) / (after[0] - before[0]) / GB


COPY_UNIT = "GB/s (1e9 bytes/s)"
# What the A40 pod showed: a pageable copy slowed this process's GEMM loop 80–95%, and the same
# copy from a second process slowed it not at all.
PAGEABLE_NOTE = (
    "the copy and the GEMM loop share one process; a pageable copy blocks the thread that "
    "enqueues the GEMMs, so the slowdown is this process's, not the GPU's. On the A40 pod the "
    "same pageable copies from a second process left the loop's rate unchanged")


def run_e4(*, pieces: Sequence[int], total: int, reps: int, window_s: float,
           gemms: tuple[str, list[Gemm]] | None, hash_buf: np.ndarray | None,
           hash_chunk: int, hash_threads: int, log: Callable[[str], None] = print
           ) -> list[dict]:
    """E4: every (direction, memory, piece size) copy alone, beside the GEMM loop, and beside
    CPU hashing. ``gemms`` is ``(model, shapes)`` for the loop; ``hash_buf`` the buffer the CPU
    threads hash. Either may be None to leave its rows out."""
    if not torch.cuda.is_available():
        return [result("copy", "skipped", reason=NO_CUDA)]
    device = torch.device("cuda")
    host = HostBuffer(total)
    dev = torch.empty(max(pieces), dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)
    out: list[dict] = []
    gemm_alone: dict[str, float] | None = None
    hash_alone: dict[str, float] = {}
    if gemms is not None:
        model, shapes = gemms
        log(f"E4 GEMM loop alone ({model}), {window_s} s")
        with matmul_precision(), GemmLoop(shapes, device) as loop:
            gemm_alone = loop_alone(loop, window_s)
        out.append(result("gemm_loop_alone", "measured",
                          params={"model": model, "shapes": [g.params() for g in shapes],
                                  "depth": DEPTH}, values=gemm_alone, units=GEMM_UNITS))
    if hash_buf is not None:
        for algo in cpu_hash.ALGOS:
            log(f"E4 CPU hashing alone: {algo}, {hash_threads} threads, {window_s} s")
            with HashLoop(hash_buf, algo, hash_chunk, hash_threads) as h:
                time.sleep(min(0.2, window_s / 4))
                before = h.count()
                time.sleep(window_s)
                hash_alone[algo] = _hash_rate(before, h.count())
            out.append(result("cpu_hashing_alone", "measured",
                              params={"algo": algo, "chunk_bytes": hash_chunk,
                                      "threads": hash_threads, "buffer_bytes": hash_buf.size},
                              values={"gb_per_s": hash_alone[algo]},
                              units={"gb_per_s": COPY_UNIT}))
    for pinned in (True, False):
        with host.pinned() if pinned else contextlib.nullcontext():
            for piece in pieces:
                parts = host.pieces(piece)
                for direction in ("h2d", "d2h"):
                    params = {"direction": direction, "pinned": pinned,
                              "host_is_pinned": bool(parts[0].is_pinned()), "piece_bytes": piece,
                              "pieces": len(parts), "total_bytes": total}
                    tag = f"{direction} {'pinned' if pinned else 'pageable'} " \
                          f"{piece >> 20} MiB × {len(parts)}"
                    log(f"E4 copy alone: {tag}")
                    copy_pass(parts, dev, direction, stream)  # warm-up
                    times = [copy_pass(parts, dev, direction, stream) for _ in range(reps)]
                    out.append(result(
                        "copy", "measured", params={**params, "reps": reps},
                        values={"gb_per_s_median": total / statistics.median(times) / GB,
                                "gb_per_s_best": total / min(times) / GB, "seconds": times},
                        units={"gb_per_s_median": COPY_UNIT, "gb_per_s_best": COPY_UNIT,
                               "seconds": "s per pass over every piece"}))
                    if gemms is not None:
                        log(f"E4 copy with the GEMM loop: {tag}")
                        out.append(_copy_with_gemm(parts, dev, direction, stream, gemms,
                                                   gemm_alone, window_s, params, device))
                    if hash_buf is not None:
                        for algo in cpu_hash.ALGOS:
                            log(f"E4 copy with CPU hashing ({algo}): {tag}")
                            out.append(_copy_with_hashing(
                                parts, dev, direction, stream, hash_buf, algo, hash_chunk,
                                hash_threads, hash_alone[algo], window_s, params))
    return out


def _copy_with_gemm(parts, dev, direction, stream, gemms, gemm_alone, window_s, params,
                    device) -> dict:
    model, shapes = gemms
    with matmul_precision(), GemmLoop(shapes, device) as loop:
        warm(loop)
        before = loop.count()
        copied, seconds = copy_window(parts, dev, direction, stream, window_s)
        during = window_rate(before, loop.count())
    assert gemm_alone is not None
    note = None
    if not params["pinned"]:
        note = PAGEABLE_NOTE
    return result(
        "copy_with_gemm_loop", "measured", note=note,
        params={**params, "model": model, "min_window_s": window_s},
        values={"copy_gb_per_s": copied / seconds / GB,
                "gemm_tflop_per_s": during["tflop_per_s"],
                "gemm_tflop_per_s_alone": gemm_alone["tflop_per_s"],
                "gemm_slowdown": 1 - during["tflop_per_s"] / gemm_alone["tflop_per_s"],
                "window_s": seconds},
        units={"copy_gb_per_s": COPY_UNIT, "gemm_tflop_per_s": GEMM_UNITS["tflop_per_s"],
               "gemm_tflop_per_s_alone": GEMM_UNITS["tflop_per_s"],
               "gemm_slowdown": "fraction of the loop's rate alone lost during the copy",
               "window_s": "s, whole copy passes"})


def _copy_with_hashing(parts, dev, direction, stream, buf, algo, chunk, threads, alone,
                       window_s, params) -> dict:
    with HashLoop(buf, algo, chunk, threads) as h:
        time.sleep(min(0.2, window_s / 4))
        before = h.count()
        copied, seconds = copy_window(parts, dev, direction, stream, window_s)
        rate = _hash_rate(before, h.count())
    return result(
        "copy_with_cpu_hashing", "measured",
        params={**params, "algo": algo, "chunk_bytes": chunk, "threads": threads,
                "min_window_s": window_s},
        values={"copy_gb_per_s": copied / seconds / GB, "hash_gb_per_s": rate,
                "hash_gb_per_s_alone": alone, "hash_slowdown": 1 - rate / alone,
                "window_s": seconds},
        units={"copy_gb_per_s": COPY_UNIT, "hash_gb_per_s": COPY_UNIT,
               "hash_gb_per_s_alone": COPY_UNIT,
               "hash_slowdown": "fraction of the hashing rate alone lost during the copy",
               "window_s": "s, whole copy passes"})


# ---- the GPU hashing slots ----------------------------------------------------------------


def e2_rows() -> list[dict]:
    return [result("cupqc_sha256", "skipped", reason=CUPQC_REASON,
                   note="cuPQC's Merkle API takes only power-of-two leaf counts (its feature "
                        "page), so P9a's promotion would need the node levels built from "
                        "batched 65-byte SHA-256 calls, as Section 6 foresees"),
            result("blake3_gpu_proxy", "skipped", reason=BLAKE3_GPU_REASON)]


def e5_gpu_rows() -> dict[str, list[dict]]:
    return {"E5b": [result("cupqc_shake256", "skipped", reason=CUPQC_REASON)],
            "E5c": [result("cupqc_sha256_counter", "skipped", reason=CUPQC_REASON)]}
