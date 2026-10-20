"""FS22: the probe, a ~2-minute measurement of the machine's rates and fixed costs (PROFILING_PLAN §5.5).

    .venv/bin/python -m verification.runs.probe --out probe.json [--seconds 120]

Runs at the session's settings (``setup_determinism`` and ``VERIF_THREADS``, read through
``load_config``), so a rate holds for the run that follows. It writes ``probe.json``, whose
rates and costs the bottleneck rule divides a section's work by. A measure the device can't
make is ``null`` with its reason under ``errors``: every CUDA measure on a CPU run.

Each measure runs two warm-ups, then repeats until its share of ``--seconds`` is spent (at
least 3 repeats, fewer if they run past 3 times the share) and reports the median. A measure
that raises records its error and the others run on. A GPU region is timed between two device
syncs, and no GPU measure holds more than 4 GiB; a host buffer is at most 1 GiB.

Rates are bytes or FLOP per second. A copy counts the bytes moved once (HBM copy counts read
plus write); ``h2d_bf16_*`` count the stored bf16 bytes. Where a rate is summarised over
several shapes the choice is named at its measure: the best GEMM shape, and the whole skinny
mix (total FLOP over total time). ``hash`` goes through ``leaf_hashes_of``, the entry
point ``commit_step`` uses, with the native hasher on.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from setup.config import load_config, setup_determinism

from setup.machine import machine_facts

__all__ = ["VERSION", "DATASHEET", "datasheet_for", "fit_hash_rate", "fit_alloc_rate", "run_probe",
           "main"]

VERSION = 1
GIB = 1 << 30
WARMUPS = 2
MIN_REPS = 3
GPU_LIMIT = 4 * GIB
HOST_BYTES = GIB  # a host buffer, and the 1 GiB transfers' size
# Fresh pinned allocations: (MiB, reps). The sizes span the host buffer pool's (about 93 MB on
# average, up to 1 GiB); 7.5 GiB in all, about 3.5 s at the 2.4 GB/s an H100 pod showed.
ALLOC_SIZES = ((16, 64), (64, 24), (256, 8), (1024, 3))

RATE_KEYS = ("gemm_bf16_flops", "gemm_fp32_flops", "skinny_bmm_fp32_flops", "hbm_copy_bytes",
             "hbm_reduce_bytes", "h2d_pinned_bytes", "d2h_pinned_bytes", "h2d_pageable_bytes",
             "d2h_pageable_bytes", "h2d_bf16_to_fp32_blocking_bytes",
             "h2d_bf16_cast_on_gpu_bytes", "host_first_touch_bytes", "hash", "host_memcpy_bytes",
             "pinned_alloc_bytes", "pinned_alloc_per_call_s")
COST_KEYS = ("launch_s", "sync_s", "event_pair_s", "tolist_s", "empty_cache_s")

# GPU name -> peaks. H100 SXM, from NVIDIA's datasheet: dense bf16 and tf32 (the 1979 and 989
# TFLOP/s figures are with sparsity), fp32, HBM3, PCIe Gen5 x16 per direction.
DATASHEET: tuple[tuple[Callable[[str], bool], dict[str, Any]], ...] = (
    (lambda n: "h100" in n and ("hbm3" in n or "sxm" in n),
     {"name": "H100 SXM", "bf16_flops": 9.894e14, "tf32_flops": 4.947e14, "fp32_flops": 6.7e13,
      "hbm_bytes": 3.35e12, "pcie_bytes": 6.4e10}),
)


def datasheet_for(gpu_name: str | None) -> dict[str, Any] | None:
    """The peaks of the GPU named ``gpu_name`` by ``torch.cuda.get_device_name``, or ``None``."""
    name = (gpu_name or "").lower()
    for matches, peaks in DATASHEET:
        if matches(name):
            return dict(peaks)
    return None


TOKENS = 128 * 128  # full-scale batch: 128 sequences of 128 tokens


def linear_shapes(loader: Callable[[str, str], Any] | None = None,
                  pins: Sequence[tuple[str, str]] | None = None,
                  errors: dict[str, str] | None = None) -> list[tuple[str, int, int]]:
    """``(name, k, n)`` of every distinct linear ``[tokens, k] @ [k, n]`` over the pinned
    full-scale models, from their configs (``loader(repo, revision)``, default the local
    cache's ``load_model_config``); a shape two models share carries both names. A model whose
    config doesn't load is skipped, with its reason in ``errors["shapes:<repo>"]``."""
    if loader is None:
        from setup.model import load_model_config as loader
    if pins is None:
        from verification.runs.fs11_session import RUN_PINS as pins
    seen: dict[tuple[int, int], str] = {}
    for repo, revision in pins:
        try:
            cfg = loader(repo, revision)
            d, d_f, vocab = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
            n_h = cfg.num_attention_heads
            n_kv = cfg.num_key_value_heads or n_h
            d_h = getattr(cfg, "head_dim", None) or d // n_h
        except Exception as e:  # noqa: BLE001
            if errors is not None:
                errors[f"shapes:{repo}"] = f"{type(e).__name__}: {e}"
            continue
        model = repo.split("/")[-1]
        for role, k, n in (("q", d, n_h * d_h), ("kv", d, n_kv * d_h), ("o", n_h * d_h, d),
                           ("up", d, d_f), ("down", d_f, d), ("head", d, vocab)):
            key = (k, n)
            seen[key] = f"{seen[key]}+{model}.{role}" if key in seen else f"{model}.{role}"
    return [(name, k, n) for (k, n), name in seen.items()]


def fit_m(k: int, n: int, itemsize: int, limit: int = GPU_LIMIT) -> int:
    """The most rows of ``TOKENS`` (halved until it fits) for which operands and output of a
    ``[m, k] @ [k, n]`` GEMM stay within ``limit`` bytes."""
    m = TOKENS
    while m > 1 and (m * k + k * n + m * n) * itemsize > limit // 2:
        m //= 2
    return m


def fit_hash_rate(mixes: Sequence[tuple[int, int, float]]) -> tuple[float, float]:
    """``(per_byte_s, per_leaf_s)`` of ``t = bytes * per_byte_s + leaves * per_leaf_s``, least
    squares over ``(bytes, leaves, seconds)`` mixes."""
    a = np.array([[b, n] for b, n, _ in mixes], dtype=np.float64)
    t = np.array([s for _, _, s in mixes], dtype=np.float64)
    scale = a.max(axis=0)
    coef, *_ = np.linalg.lstsq(a / scale, t, rcond=None)
    per_byte, per_leaf = coef / scale
    if per_leaf < 0:  # noise: refit on bytes alone, with no per-leaf cost
        per_byte = float((a[:, 0] @ t) / (a[:, 0] @ a[:, 0]))
        per_leaf = 0.0
    return float(per_byte), float(per_leaf)


def fit_residual(mixes: Sequence[tuple[int, int, float]], per_byte: float,
                 per_leaf: float) -> float:
    """The fit's relative residual: ``||t - fit|| / ||t||`` over the mixes."""
    t = np.array([s for _, _, s in mixes], dtype=np.float64)
    fit = np.array([b * per_byte + n * per_leaf for b, n, _ in mixes])
    return float(np.linalg.norm(t - fit) / np.linalg.norm(t))


def fit_alloc_rate(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    """``(per_call_s, per_byte_s)`` of ``t = per_call_s + bytes * per_byte_s``, least squares
    over ``(bytes, seconds)`` points, each weighted by ``1 / seconds`` so the small sizes fix
    the per-call cost as the large ones fix the rate."""
    b = np.array([n for n, _ in points], dtype=np.float64)
    t = np.array([s for _, s in points], dtype=np.float64)
    a = np.stack([np.ones_like(b), b], axis=1) / t[:, None]
    scale = np.array([1.0, b.max()])
    coef, *_ = np.linalg.lstsq(a / scale, np.ones_like(t), rcond=None)
    per_call, per_byte = coef / scale
    if per_call < 0:  # noise: refit on bytes alone, with no per-call cost
        x = b / t
        per_byte = float(x.sum() / (x @ x))
        per_call = 0.0
    return float(per_call), float(per_byte)


@dataclass
class Clock:
    """One measure's repeats: ``seconds`` of budget, median of what ran."""

    share_s: float

    def median(self, once: Callable[[], float]) -> float:
        """Median of ``once()``'s seconds over the warm-ups' successors."""
        for _ in range(WARMUPS):
            once()
        times: list[float] = []
        start = time.perf_counter()
        while True:
            times.append(once())
            spent = time.perf_counter() - start
            if (spent >= self.share_s and len(times) >= MIN_REPS) or spent >= 3 * self.share_s:
                return statistics.median(times)


def _timed(fn: Callable[[], Any], cuda: bool) -> float:
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _free() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


# ---- CUDA measures --------------------------------------------------------------------


def _gemm(k: int, n: int, dtype: torch.dtype, clock: Clock) -> tuple[int, float]:
    m = fit_m(k, n, torch.empty((), dtype=dtype).element_size())
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    t = clock.median(lambda: _timed(lambda: torch.mm(a, b, out=out), True))
    return m, 2 * m * k * n / t


def gemm_todo(errors: dict) -> list[tuple[str, int, int, torch.dtype, str]]:
    """``(name, k, n, dtype, label)`` of every GEMM the probe times: each linear shape in bf16
    and fp32."""
    return [(name, k, n, dt, label) for name, k, n in linear_shapes(errors=errors)
            for dt, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32"))]


def listed_shapes(todo: Sequence[tuple[str, int, int, torch.dtype, str]]) -> list[dict[str, Any]]:
    """The ``shapes`` entries of ``todo`` with no rate, as a device without CUDA reports them."""
    return [{"name": name, "dtype": label,
             "m": fit_m(k, n, torch.empty((), dtype=dt).element_size()),
             "n": n, "k": k, "flops_per_s": None} for name, k, n, dt, label in todo]


def measure_gemm(share_s: float, rates: dict, errors: dict, shapes: list) -> None:
    """GEMM at every linear shape in bf16 and fp32 at the full-scale batch (a head's rows cut
    to fit 4 GiB). The rate is the best shape's, the most the device does at these sizes."""
    todo = gemm_todo(errors)
    if not todo:
        for key in ("gemm_bf16_flops", "gemm_fp32_flops"):
            errors[key] = "no model shapes in the HF cache"
        return
    per = share_s / len(todo)
    best: dict[str, float] = {}
    for name, k, n, dt, label in todo:
        try:
            m, flops = _gemm(k, n, dt, Clock(per))
        except Exception as e:  # noqa: BLE001 - one shape's failure leaves the rest
            errors[f"gemm_{label}[{name}]"] = f"{type(e).__name__}: {e}"
            continue
        finally:
            _free()
        shapes.append({"name": name, "dtype": label, "m": m, "n": n, "k": k,
                       "flops_per_s": flops})
        best[label] = max(best.get(label, 0.0), flops)
    for label in ("bf16", "fp32"):
        key = f"gemm_{label}_flops"
        if label in best:
            rates[key] = best[label]
        else:
            errors.setdefault(key, "every shape failed")


def measure_skinny(share_s: float, k_cols: int, rates: dict, errors: dict,
                   shapes: list) -> None:
    """Check 5's product times ``k_cols`` columns, fp32: dense ``[tokens, width] @ [width, k]``
    at two widths, and a batched attention member ``[128 * 32, 128, 128] @ [.., 128, k]``. The
    rate is the mix's: total FLOP over total time."""
    todo = [("skinny_dense_2048", TOKENS, 2048, 1), ("skinny_dense_8192", TOKENS, 8192, 1),
            ("skinny_attn_scores", 128, 128, 128 * 32)]
    per = share_s / len(todo)
    total_flops = total_s = 0.0
    for name, m, width, batch in todo:
        a = r = out = op = None
        try:
            if batch == 1:
                a = torch.randn(m, width, device="cuda")
                r = torch.randn(width, k_cols, device="cuda")
                out = torch.empty(m, k_cols, device="cuda")
                op = lambda: torch.mm(a, r, out=out)  # noqa: E731
            else:
                a = torch.randn(batch, m, width, device="cuda")
                r = torch.randn(batch, width, k_cols, device="cuda")
                out = torch.empty(batch, m, k_cols, device="cuda")
                op = lambda: torch.bmm(a, r, out=out)  # noqa: E731
            t = Clock(per).median(lambda: _timed(op, True))
        except Exception as e:  # noqa: BLE001
            errors[f"skinny_bmm_fp32[{name}]"] = f"{type(e).__name__}: {e}"
            continue
        finally:
            a = r = out = op = None
            _free()
        flops = 2 * batch * m * width * k_cols
        shapes.append({"name": name, "dtype": "fp32", "m": m, "n": k_cols, "k": width,
                       "flops_per_s": flops / t})
        total_flops += flops
        total_s += t
    if total_s:
        rates["skinny_bmm_fp32_flops"] = total_flops / total_s
    else:
        errors.setdefault("skinny_bmm_fp32_flops", "every shape failed")


def measure_hbm(share_s: float, rates: dict, errors: dict) -> None:
    """Copy 2 GiB into a second 2 GiB (4 GiB moved) and sum a 4 GiB tensor."""
    try:
        src = torch.empty(2 * GIB, dtype=torch.uint8, device="cuda")
        src.fill_(1)
        dst = torch.empty_like(src)
        t = Clock(share_s / 2).median(lambda: _timed(lambda: dst.copy_(src), True))
        rates["hbm_copy_bytes"] = 2 * src.numel() / t
    except Exception as e:  # noqa: BLE001
        errors["hbm_copy_bytes"] = f"{type(e).__name__}: {e}"
    src = dst = None
    _free()
    try:
        big = torch.ones(GPU_LIMIT // 4, dtype=torch.float32, device="cuda")
        t = Clock(share_s / 2).median(lambda: _timed(lambda: big.sum(), True))
        rates["hbm_reduce_bytes"] = big.numel() * 4 / t
    except Exception as e:  # noqa: BLE001
        errors["hbm_reduce_bytes"] = f"{type(e).__name__}: {e}"
    big = None
    _free()


def measure_transfers(share_s: float, rates: dict, errors: dict) -> None:
    """1 GiB host-to-device and device-to-host, from pinned and pageable host memory."""
    for kind, pinned in (("pinned", True), ("pageable", False)):
        keys = (f"h2d_{kind}_bytes", f"d2h_{kind}_bytes")
        try:
            dev = torch.empty(HOST_BYTES, dtype=torch.uint8, device="cuda")
            host = torch.empty(HOST_BYTES, dtype=torch.uint8, pin_memory=pinned)
            host.fill_(1)
        except Exception as e:  # noqa: BLE001
            for key in keys:
                errors[key] = f"{type(e).__name__}: {e}"
            continue
        for key, dst, src in ((keys[0], dev, host), (keys[1], host, dev)):
            try:
                t = Clock(share_s / 4).median(
                    lambda: _timed(lambda: dst.copy_(src, non_blocking=pinned), True))
                rates[key] = HOST_BYTES / t
            except Exception as e:  # noqa: BLE001
                errors[key] = f"{type(e).__name__}: {e}"
        del host, dev
        if pinned:  # hand the pinned block back, so the pageable pass starts without it
            getattr(torch._C, "_host_emptyCache", lambda: None)()
        _free()


def measure_pinned_alloc(share_s: float, rates: dict, errors: dict) -> None:
    """Time ``torch.empty(n, dtype=torch.uint8, pin_memory=True)``, the host buffer pool's
    call (``HostBuffers.take``), at each of ``ALLOC_SIZES``, and fit seconds against bytes.

    PyTorch caches a freed pinned block, so a free-and-reallocate loop would time the cache.
    The installed torch (2.9) has no call that empties the cache (``torch._C._host_emptyCache``
    is absent), so every rep of a size holds its buffer until the size's reps are done: each
    allocation is a fresh ``cudaHostAlloc``, checked against ``num_host_alloc``. The freed
    blocks stay cached, so this runs before ``measure_transfers``, whose 1 GiB buffer would
    otherwise hit one. A size stops early, with at least 3 reps, past 3 times the share, as
    ``Clock`` does."""
    def num_host_alloc() -> int | None:
        return torch.cuda.host_memory_stats().get("num_host_alloc")

    try:
        torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)  # the first call's setup
        points = []
        start = time.perf_counter()
        for mib, reps in ALLOC_SIZES:
            held, times = [], []
            for _ in range(reps):
                before = num_host_alloc()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                held.append(torch.empty(mib << 20, dtype=torch.uint8, pin_memory=True))
                times.append(time.perf_counter() - t0)
                after = num_host_alloc()
                if before is not None and after != before + 1:
                    raise RuntimeError(f"a {mib} MiB allocation made {after - before} "
                                       "cudaHostAlloc calls, not 1: the host cache answered")
                if len(times) >= MIN_REPS and time.perf_counter() - start > 3 * share_s:
                    break
            del held
            points.append((mib << 20, statistics.median(times)))
        per_call, per_byte = fit_alloc_rate(points)
        if per_byte <= 0:
            raise RuntimeError(f"fit gave per_byte_s = {per_byte:.3g}; points {points}")
        rates["pinned_alloc_bytes"] = 1 / per_byte
        rates["pinned_alloc_per_call_s"] = per_call
    except Exception as e:  # noqa: BLE001
        for key in ("pinned_alloc_bytes", "pinned_alloc_per_call_s"):
            errors[key] = f"{type(e).__name__}: {e}"


def measure_bf16_transfers(share_s: float, rates: dict, errors: dict) -> None:
    """A 1 GiB bf16 host tensor as check 5 sends it (``.to("cuda", float32)``, which casts on
    the host and blocks) and copied then cast on the GPU. Both count the stored bytes."""
    host = torch.empty(HOST_BYTES // 2, dtype=torch.bfloat16)
    host.normal_()
    for key, fn in (
            ("h2d_bf16_to_fp32_blocking_bytes", lambda: host.to("cuda", torch.float32)),
            ("h2d_bf16_cast_on_gpu_bytes", lambda: host.to("cuda").to(torch.float32))):
        try:
            t = Clock(share_s / 2).median(lambda: _timed(fn, True))
            rates[key] = HOST_BYTES / t
        except Exception as e:  # noqa: BLE001
            errors[key] = f"{type(e).__name__}: {e}"
        _free()


def measure_costs(share_s: float, costs: dict, errors: dict) -> None:
    """Fixed costs, each per operation: a tiny kernel launch (10,000 without a sync), an idle
    sync, an event pair (two timing events recorded, as a stamp does), a 1,000-element
    ``.tolist()`` and ``empty_cache`` after freeing a block."""
    x = torch.zeros(1, device="cuda")
    ones = torch.ones(1000, device="cuda")
    per = share_s / 5

    def launches() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10_000):
            x.add_(1)
        dt = time.perf_counter() - t0
        torch.cuda.synchronize()
        return dt / 10_000

    def syncs() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 1000

    def pairs() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        evs = []
        for _ in range(1000):  # fresh events per stamp: the cost holds cudaEventCreate
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            b.record()
            evs.append((a, b))
        dt = time.perf_counter() - t0
        torch.cuda.synchronize()
        return dt / 1000

    def tolists() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            ones.tolist()
        return (time.perf_counter() - t0) / 100

    def empties() -> float:
        total = 0.0
        for _ in range(10):
            block = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            del block
            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            total += time.perf_counter() - t0
        return total / 10

    for key, once in (("launch_s", launches), ("sync_s", syncs), ("event_pair_s", pairs),
                      ("tolist_s", tolists), ("empty_cache_s", empties)):
        try:
            costs[key] = Clock(per).median(once)
        except Exception as e:  # noqa: BLE001
            errors[key] = f"{type(e).__name__}: {e}"


# ---- host measures --------------------------------------------------------------------


def measure_host_memcpy(share_s: float, rates: dict, errors: dict) -> None:
    try:
        src = torch.empty(HOST_BYTES, dtype=torch.uint8)
        src.fill_(1)
        dst = torch.empty_like(src)
        dst.fill_(0)
        t = Clock(share_s).median(lambda: _timed(lambda: dst.copy_(src), False))
        rates["host_memcpy_bytes"] = HOST_BYTES / t
    except Exception as e:  # noqa: BLE001
        errors["host_memcpy_bytes"] = f"{type(e).__name__}: {e}"


def measure_first_touch(share_s: float, rates: dict, errors: dict) -> None:
    """Allocate a fresh 1 GiB host tensor and fill it, a new tensor in every repeat. The rate
    holds at the run's settings: deterministic mode fills new memory, as it does for the run's
    own allocations, so the fill is part of what's measured."""
    def once() -> float:
        t0 = time.perf_counter()
        x = torch.empty(HOST_BYTES, dtype=torch.uint8)
        x.fill_(1)
        dt = time.perf_counter() - t0
        del x
        return dt
    try:
        rates["host_first_touch_bytes"] = HOST_BYTES / Clock(share_s).median(once)
    except Exception as e:  # noqa: BLE001
        errors["host_first_touch_bytes"] = f"{type(e).__name__}: {e}"


# The synthetic step's leaf mixes, as (slot key, leaves): each slot is a weight of the small
# decoder below, so the mixes run from 32,768 leaves of 2 KiB to 1 of 16 MiB.
_HASH_MIXES = (("norm", 32768), ("norm", 8192), ("q", 64), ("q", 16), ("mlp", 32),
               ("embed", 4), ("embed", 1))
_HASH_SLOTS = {"norm": "model.norm.weight", "q": "model.layers.0.self_attn.q_proj.weight",
               "mlp": "model.layers.0.mlp.up_proj.weight",
               "embed": "model.embed_tokens.weight"}
_HASH_POOL = 8  # distinct tensors per slot, cycled, so a mix doesn't rehash one cached buffer


def _hash_computation() -> Any:
    from transformers import LlamaConfig

    from verification.computation.instances.decoder import DecoderComputation
    cfg = LlamaConfig(vocab_size=8192, hidden_size=512, intermediate_size=1024,
                      num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=4,
                      max_position_embeddings=64, hidden_act="silu", attention_dropout=0.0,
                      pad_token_id=0, tie_word_embeddings=False)
    return DecoderComputation(cfg, n_s=1, n=8, eta=1.0)


def measure_hash(share_s: float, rates: dict, errors: dict) -> None:
    """Time leaf mixes through ``leaf_hashes_of`` (native hasher on) and fit the per-byte and
    per-leaf cost."""
    from verification.commitment import native
    from verification.commitment.leaves import _tensor_slot, leaf_hashes_of
    try:
        if not native.enabled():
            raise RuntimeError("the native hasher isn't available (build it with "
                               "python -m verification.commitment.native)")
        c = _hash_computation()
        pools, slot = {}, {}
        g = torch.Generator().manual_seed(0)
        for key, name in _HASH_SLOTS.items():
            slot[key] = c.n_s + c.weight_names.index(name)
            _, shape, dtype, _ = _tensor_slot(c, slot[key])
            pools[key] = [torch.randn(shape, generator=g).to(dtype) for _ in range(_HASH_POOL)]
        per = share_s / len(_HASH_MIXES)
        mixes = []
        for key, count in _HASH_MIXES:
            pool, idx = pools[key], slot[key]
            items = [(idx, pool[i % _HASH_POOL]) for i in range(count)]
            nbytes = count * pool[0].numel() * pool[0].element_size()
            t = Clock(per).median(lambda: _timed(lambda: leaf_hashes_of(c, items), False))
            mixes.append((nbytes, count, t))
        per_byte, per_leaf = fit_hash_rate(mixes)
        if per_byte <= 0:
            raise RuntimeError(f"fit gave per_byte_s = {per_byte:.3g}; mixes {mixes}")
        rates["hash"] = {"per_byte_s": per_byte, "per_leaf_s": per_leaf,
                         "threads": torch.get_num_threads(),
                         "per_leaf_refit_to_zero": per_leaf == 0.0,
                         "residual": fit_residual(mixes, per_byte, per_leaf),
                         "mixes": [{"leaves": n, "bytes": b, "seconds": t}
                                   for b, n, t in mixes]}
    except Exception as e:  # noqa: BLE001
        errors["hash"] = f"{type(e).__name__}: {e}"


# ---- the probe ------------------------------------------------------------------------


def run_probe(seconds: float, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """The probe's document, at the settings ``env`` (default: the process's) names."""
    t_start = time.perf_counter()
    cfg = load_config(env)
    setup_determinism(cfg)
    from verification.parameters import load_protocol_config
    k_cols = load_protocol_config(env).k
    cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    rates: dict[str, Any] = {k: None for k in RATE_KEYS}
    costs: dict[str, Any] = {k: None for k in COST_KEYS}
    errors: dict[str, str] = {}
    shapes: list[dict[str, Any]] = []

    measures: list[tuple[str, float, Callable[[float], None]]] = [
        ("host_memcpy", 4, lambda s: measure_host_memcpy(s, rates, errors)),
        ("first_touch", 4, lambda s: measure_first_touch(s, rates, errors)),
        ("hash", 14, lambda s: measure_hash(s, rates, errors)),
    ]
    if cuda:
        torch.cuda.set_device(torch.device(cfg.device).index or 0)
        measures += [
            ("gemm", 30, lambda s: measure_gemm(s, rates, errors, shapes)),
            ("skinny", 5, lambda s: measure_skinny(s, k_cols, rates, errors, shapes)),
            ("hbm", 8, lambda s: measure_hbm(s, rates, errors)),
            ("pinned_alloc", 6, lambda s: measure_pinned_alloc(s, rates, errors)),
            ("transfers", 12, lambda s: measure_transfers(s, rates, errors)),
            ("bf16_transfers", 8, lambda s: measure_bf16_transfers(s, rates, errors)),
            ("costs", 10, lambda s: measure_costs(s, costs, errors)),
        ]
    else:
        shapes += listed_shapes(gemm_todo(errors))
        why = ("CUDA is not available" if cfg.device.startswith("cuda")
               else f"no CUDA device (device={cfg.device})")
        for key in (*RATE_KEYS, *COST_KEYS):
            if key not in ("host_first_touch_bytes", "hash", "host_memcpy_bytes"):
                errors[key] = why
    total = sum(w for _, w, _ in measures)
    for name, weight, fn in measures:
        try:
            fn(seconds * weight / total)
        except Exception as e:  # noqa: BLE001 - what a measure's own handlers missed
            errors[f"measure:{name}"] = f"{type(e).__name__}: {e}"
        finally:
            _free()

    gpu_name = torch.cuda.get_device_name(torch.device(cfg.device)) if cuda else None
    try:
        facts = machine_facts()
    except Exception as e:  # noqa: BLE001
        facts = None
        errors["machine"] = f"{type(e).__name__}: {e}"
    return {"version": VERSION, "device": cfg.device, "machine": facts,
            "threads": torch.get_num_threads(),
            "determinism": {"tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                            "deterministic": bool(torch.are_deterministic_algorithms_enabled())},
            "rates": rates, "costs": costs, "shapes": shapes,
            "datasheet": datasheet_for(gpu_name), "errors": errors,
            "wall_s": time.perf_counter() - t_start}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="where to write probe.json")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="the probe's time budget, shared between its measures")
    args = ap.parse_args(argv)
    doc = run_probe(args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"probe: wrote {args.out} ({doc['wall_s']:.1f} s, {len(doc['errors'])} unmeasured)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
