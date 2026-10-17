"""E5(a) and E5(a′): challenge generation for a model's product list (P8b).

The challenge functions here reproduce `verification/verifier/matmul_check/challenges.py`
standalone, call for call (its validation included), so their time is the current path's
time. `tests/benchmarks/hashing/test_challenges.py` checks them against the protocol's bit for
bit. Check 5 asks for challenges run by run: a product outside attention gets
``challenge_matrix`` (k XOF calls, stacked), and a run of one layer's attention members gets one
``challenge_matrices`` call per operand shape (k calls per member, converted in one numpy pass).
:func:`plan` rebuilds that call list from the leaf-size list.

E5(a′) has two no-dependency baselines for batching: the same calls spread over a thread pool
(the ``blake3`` package releases the GIL only for large inputs, so small XOF calls contend for
it) and over a process pool. The native batch, the plan's E5(a′) proper, is
`benchmarks/hashing/native.py`.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import re
import struct
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import blake3
import numpy as np
import torch

from benchmarks.hashing.cpu_hash import start_threads
from benchmarks.hashing.report import result
from benchmarks.hashing.sizes import Step

TAG_LABEL = 0x10
_GRID = 1 << 24

# ---- the current path, reproduced -----------------------------------------------------------


def encode_label(m: int, j: int) -> bytes:
    """``0x10 ‖ m(4 BE) ‖ j(1)`` (P8a)."""
    if not 1 <= m < 1 << 32:
        raise ValueError(f"label m={m} out of range")
    if not 1 <= j <= 255:
        raise ValueError(f"label j={j} out of range")
    return struct.pack(">BIB", TAG_LABEL, m, j)


def _validate(h: bytes, m: int, width: int) -> None:
    if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
        raise ValueError("h must be 32 bytes")
    if not 1 <= m < 1 << 32:
        raise ValueError(f"m must be in [1, 2^32), got {m}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")


def _entries(words: np.ndarray) -> np.ndarray:
    """uint32 words to float32 entries ``(2(w >> 8) + 1 − 2^24) / 2^24`` (P8b)."""
    a = (words.astype(np.uint32) >> 8).astype(np.int64)
    numerator = (2 * a + 1 - _GRID).astype(np.float32)
    return numerator / np.float32(_GRID)


def challenge_vector(h: bytes, m: int, j: int, width: int) -> torch.Tensor:
    _validate(h, m, width)
    if not 1 <= j <= 255:
        raise ValueError(f"j must be in [1, 255], got {j}")
    xof = blake3.blake3(encode_label(m, j), key=bytes(h)).digest(length=4 * width)
    return torch.from_numpy(_entries(np.frombuffer(xof, dtype="<u4")))


def challenge_matrix(h: bytes, m: int, k: int, width: int) -> torch.Tensor:
    """Product ``m``'s k challenge vectors as columns, float32 ``[width, k]``."""
    _validate(h, m, width)
    if not 1 <= k <= 255:
        raise ValueError(f"k must be in [1, 255], got {k}")
    return torch.stack([challenge_vector(h, m, j, width) for j in range(1, k + 1)], dim=1)


def challenge_matrices(h: bytes, ms: Sequence[int], k: int, width: int) -> torch.Tensor:
    """``challenge_matrix`` for each of ``ms``, as ``[len(ms), width, k]``, in one numpy pass."""
    if not 1 <= k <= 255:
        raise ValueError(f"k must be in [1, 255], got {k}")
    for m in ms:
        _validate(h, m, width)
    key, n = bytes(h), 4 * width
    xof = b"".join(blake3.blake3(encode_label(m, j), key=key).digest(length=n)
                   for m in ms for j in range(1, k + 1))
    words = np.frombuffer(xof, dtype="<u4").reshape(len(ms), k, width)
    return torch.from_numpy(_entries(words)).transpose(1, 2).contiguous()


# ---- the call list --------------------------------------------------------------------------

_MEMBER = re.compile(r"^L(\d+)\.\w+\[\d+,\d+\]$")
_LAYER = re.compile(r"^L(\d+)\.")


@dataclass(frozen=True)
class Product:
    m: int
    name: str
    shape: tuple[int, ...]
    layer: int | None  # None for the head's products (Lambda, dF, G_E_head / G_head, per block)
    member: bool  # an attention product of one (sequence, head) pair

    @property
    def width(self) -> int:
        return self.shape[1]


@dataclass(frozen=True)
class Call:
    """One challenge call of check 5: ``challenge_matrix`` for one product, or
    ``challenge_matrices`` for a group of attention members of one shape."""

    ms: tuple[int, ...]
    width: int
    layer: int | None
    batched: bool

    @property
    def xof_calls(self) -> int:
        return len(self.ms)  # times k


def products(step: Step) -> list[Product]:
    out = []
    leaves = [l for l in step.leaves if l.section == "product"]
    first = leaves[0].index
    for l in leaves:
        layer = _LAYER.match(l.name)
        out.append(Product(l.index - first + 1, l.name, l.shape,
                           int(layer.group(1)) if layer else None,
                           bool(_MEMBER.match(l.name))))
    return out


def member_runs(ps: Sequence[Product]) -> Iterator[Sequence[Product]]:
    """Check 5's runs: a maximal stretch of consecutive members of one layer, or one product."""
    i = 0
    while i < len(ps):
        j = i + 1
        if ps[i].member:
            while j < len(ps) and ps[j].member and ps[j].layer == ps[i].layer:
                j += 1
        yield ps[i:j]
        i = j


def plan(ps: Sequence[Product]) -> list[Call]:
    """Check 5's challenge calls, in order. A run is grouped by product shape, which for the
    decoder is the grouping by operand shapes check 5 uses (each member role has one operand
    pair, and its product shape follows from it)."""
    calls = []
    for run in member_runs(ps):
        if len(run) == 1:
            p = run[0]
            calls.append(Call((p.m,), p.width, p.layer, False))
            continue
        groups: dict[tuple[int, ...], list[Product]] = {}
        for p in run:
            groups.setdefault(p.shape, []).append(p)
        for shape, g in groups.items():
            calls.append(Call(tuple(p.m for p in g), shape[1], g[0].layer, True))
    return calls


def _run_call(h: bytes, k: int, c: Call) -> None:
    if c.batched:
        challenge_matrices(h, c.ms, k, c.width)
    else:
        challenge_matrix(h, c.ms[0], k, c.width)


def select(calls: Sequence[Call], layers: int | None) -> tuple[list[Call], int, int]:
    """The head's calls and those of layers ``1..layers`` (all when None): ``(calls, layers
    selected, layers in the model)``."""
    total = max(c.layer or 0 for c in calls)
    n = total if layers is None else min(layers, total)
    return [c for c in calls if c.layer is None or c.layer <= n], n, total


# ---- E5(a) ----------------------------------------------------------------------------------


def _timed(calls: Sequence[Call], h: bytes, k: int,
           run_call=None) -> tuple[float, float, int, int]:
    """``(head seconds, layer seconds, head XOF calls, layer XOF calls)`` on this thread, with
    ``run_call(h, k, call)`` (the Python path by default)."""
    run_call = run_call or _run_call
    head = layers = 0.0
    n_head = n_layers = 0
    for c in calls:
        t0 = time.perf_counter()
        run_call(h, k, c)
        dt = time.perf_counter() - t0
        if c.layer is None:
            head, n_head = head + dt, n_head + c.xof_calls * k
        else:
            layers, n_layers = layers + dt, n_layers + c.xof_calls * k
    return head, layers, n_head, n_layers


def _extrapolate(name: str, step: Step, params: dict, head: float, lay: float, n_sel: int,
                 n_total: int, calls_measured: int, calls_step: int, note: str) -> list[dict]:
    measured = head + lay
    out = [result(name, "measured", params={**params, "layers_measured": n_sel},
                  values={"seconds": measured, "xof_calls": calls_measured,
                          "us_per_xof_call": measured / calls_measured * 1e6},
                  units={"seconds": "s", "xof_calls": "calls", "us_per_xof_call": "µs"},
                  note=note)]
    step_s = head + lay * n_total / n_sel
    status = "measured" if n_sel == n_total else "estimated"
    out.append(result(f"{name}_per_step", status,
                      params={**params, "layers_measured": n_sel, "layers": n_total},
                      values={"seconds_per_step": step_s, "xof_calls_per_step": calls_step},
                      units={"seconds_per_step": "s", "xof_calls_per_step": "calls"},
                      note=None if status == "measured" else
                      f"head time + layer time × {n_total}/{n_sel} (layers are identical in "
                      f"shape, so each costs the same)"))
    return out


def run_e5a(step: Step, *, k: int, layers: int | None, seed: int = 0) -> list[dict]:
    h = np.random.default_rng(seed).bytes(32)
    calls = plan(products(step))
    chosen, n_sel, n_total = select(calls, layers)
    calls_step = sum(c.xof_calls for c in calls) * k
    head, lay, n_head, n_lay = _timed(chosen, h, k)
    params = {**step.params(), "k": k, "threads": 1,
              "products": sum(c.xof_calls for c in calls),
              "challenge_bytes_per_step": sum(c.xof_calls * c.width * 4 for c in calls) * k}
    return _extrapolate("python_path", step, params, head, lay, n_sel, n_total,
                        n_head + n_lay, calls_step,
                        note="check 5's calls: challenge_matrix per product, "
                             "challenge_matrices per attention-member group; float32 out")


# ---- E5(a′) ---------------------------------------------------------------------------------


def _split(c: Call, parts: int) -> list[Call]:
    if not c.batched or parts <= 1:
        return [c]
    size = math.ceil(len(c.ms) / parts)
    return [Call(c.ms[i:i + size], c.width, c.layer, True) for i in range(0, len(c.ms), size)]


def _discard(h: bytes, k: int, cs: Sequence[Call]) -> int:
    for c in cs:
        _run_call(h, k, c)  # generated and converted, then dropped
    return sum(c.xof_calls for c in cs) * k


def _pooled(pool, workers: int, calls: Sequence[Call], h: bytes, k: int) -> tuple[float, float]:
    """Head and layer seconds with each batched call split over the workers."""
    times = {True: 0.0, False: 0.0}
    for is_head in (True, False):
        part = [s for c in calls if (c.layer is None) == is_head for s in _split(c, workers)]
        # contiguous blocks of about equal XOF work, one per worker
        total = sum(s.xof_calls for s in part)
        blocks, cur, acc = [], [], 0
        for s in part:
            cur.append(s)
            acc += s.xof_calls
            if acc >= total / workers * (len(blocks) + 1):
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        t0 = time.perf_counter()
        for f in [pool.submit(_discard, h, k, b) for b in blocks]:
            f.result()
        times[is_head] = time.perf_counter() - t0
    return times[True], times[False]


def _noop(_: int) -> int:
    return os.getpid()


def run_e5a_prime(step: Step, *, k: int, layers: int | None, threads: Sequence[int],
                  processes: int, seed: int = 0) -> list[dict]:
    h = np.random.default_rng(seed).bytes(32)
    calls = plan(products(step))
    chosen, n_sel, n_total = select(calls, layers)
    calls_step = sum(c.xof_calls for c in calls) * k
    calls_measured = sum(c.xof_calls for c in chosen) * k
    base = {**step.params(), "k": k}
    out = []
    for t in threads:
        with ThreadPoolExecutor(t) as pool:
            start_threads(pool, t)  # every thread running before timing
            head, lay = _pooled(pool, t, chosen, h, k)
        out += _extrapolate("thread_pool", step, {**base, "threads": t}, head, lay, n_sel,
                            n_total, calls_measured, calls_step,
                            note="no-dependency baseline: the Python path's calls split over a "
                                 "thread pool; outputs generated, converted and dropped; "
                                 "thread start-up excluded")
    if processes > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(processes, mp_context=ctx) as pool:
            list(pool.map(_noop, range(processes * 4)))  # start every worker before timing
            head, lay = _pooled(pool, processes, chosen, h, k)
        out += _extrapolate("process_pool", step, {**base, "processes": processes}, head, lay,
                            n_sel, n_total, calls_measured, calls_step,
                            note="generation only: each worker generates and converts its "
                                 "share and drops it, nothing returns to the parent; worker "
                                 "start-up excluded")
    return out
