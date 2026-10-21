"""The fused measuring kernel of check 5 (FS37, decision O2): its table, build and launch.

``measure_core.h`` defines every number of a product's row on logical indices;
``measure_phases.h`` splits that work into items (a warp each) and steps (a dot or sum whose 32
partials are the 32 lanes); ``measure_kernel.cu`` is the thin wrapper that runs the steps with
shuffles. The host test (``measure_core_host.c``) runs the same items and steps in loops, so the
GPU's bits are testable without a GPU.

Compiled at first use by NVRTC (through ``gpu_blake3``'s :class:`Nvrtc`, :class:`Driver` and
:func:`build_image`) with ``--fmad=false`` into a CUBIN cached under ``native.cache_dir()``;
importing this module compiles and loads nothing. One call (:func:`enqueue`) packs every group's
products into one table in one pinned buffer, copies it up with one non-blocking copy and
launches three kernels (the phases) on the current stream; the rows stay on the device. Each
call's :class:`Pending` holds every tensor and buffer the stream reads, with each input's
``_version``, until :func:`drain` has waited for its event and found no input changed.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.verifier.matmul_check.challenge_source import (
    ChallengeSource,
    HostChallengeSource,
)
from verification.verifier.matmul_check.measure_table import (
    DT_BF16,
    DT_FP32,
    M_BIASED,
    M_GUARD,
    MProd,
    MView,
    row_width,
)

if TYPE_CHECKING:
    from verification.verifier.matmul_check.freivalds import MeasureGroup

HERE = Path(__file__).resolve().parent
CORE_NAME = "measure_core.h"
PHASES_NAME = "measure_phases.h"
CU_NAME = "measure_kernel.cu"
KERNEL = "pots_measure_phase"
OPTIONS = ("--fmad=false",)
"""Extra NVRTC options, part of the cache key: no fused multiply-add but the core's ``fmaf``."""
WARPS = 8  # warps per block, POTS_MK_WARPS in the .cu
MAX_BLOCKS = 1 << 22
PHASES = 3
ROW_FIXED = 5
_DTYPE_CODE = {torch.float32: DT_FP32, torch.bfloat16: DT_BF16}

_MVIEW = np.dtype([("addr", "<u8"), ("rows", "<i8"), ("cols", "<i8"), ("row_stride", "<i8"),
                   ("col_stride", "<i8"), ("dtype", "<u4"), ("pad", "<u4")])
MPROD_DTYPE = np.dtype([("a", _MVIEW), ("b", _MVIEW), ("p", _MVIEW), ("bias", _MVIEW),
                        ("r_addr", "<u8"), ("k", "<u4"), ("flags", "<u4")])
"""``pots_mprod_t`` (208 bytes), the layout of :class:`measure_table.MProd`."""
WORK_DTYPE = np.dtype([("scratch", "<u8"), ("start", "<u8", (PHASES,))])
"""``pots_mwork_t`` (32 bytes)."""
assert _MVIEW.itemsize == ctypes.sizeof(MView) and MPROD_DTYPE.itemsize == ctypes.sizeof(MProd)


class KernelArgs(ctypes.Structure):
    """``pots_mkargs_t``: one launch's argument."""

    _fields_ = [("prods", ctypes.c_uint64), ("works", ctypes.c_uint64), ("rows", ctypes.c_uint64),
                ("n_prod", ctypes.c_uint64), ("total", ctypes.c_uint64),
                ("phase", ctypes.c_uint32), ("k", ctypes.c_uint32)]


# ---- the build ----

def source() -> gb.Source:
    """The kernel, its two headers and the cache prefix, for :func:`gpu_blake3.build_image`."""
    return gb.Source(CU_NAME, (HERE / CU_NAME).read_text(),
                     {CORE_NAME: (HERE / CORE_NAME).read_text(),
                      PHASES_NAME: (HERE / PHASES_NAME).read_text()}, "measure_kernel")


def source_hash() -> str:
    return source().digest()


def build(nvrtc: gb.Nvrtc, cc: tuple[int, int], *, cache: Path | None = None,
          use_cache: bool = True) -> gb.Built:
    """The compiled image for compute capability ``cc`` (cached in ``cache`` when given)."""
    return gb.build_image(nvrtc, cc, cache=cache, use_cache=use_cache, source=source(),
                          extra_options=OPTIONS)


class MeasureKernels:
    """A device's loaded module and its kernel."""

    def __init__(self, driver: gb.Driver, dev: int, built: gb.Built) -> None:
        self.driver = driver
        self.dev = dev
        self.built = built
        driver.push(dev)
        try:
            self.module = driver.load_module(built.image)
            self.fn = driver.function(self.module, KERNEL)
        finally:
            driver.pop()

    def run(self, launches: Sequence[tuple[KernelArgs, int]], stream: int) -> None:
        """Launch each ``(args, blocks)`` in order on the raw ``stream``."""
        self.driver.push(self.dev)
        try:
            for args, blocks in launches:
                self.driver.launch(self.fn, blocks, WARPS * 32, args, stream)  # pyright: ignore
        finally:
            self.driver.pop()


@functools.lru_cache(maxsize=None)
def _built(cc: tuple[int, int]) -> gb.Built:
    return build(gb._nvrtc(), cc, cache=native.cache_dir())


@functools.lru_cache(maxsize=None)
def _kernels(ordinal: int) -> MeasureKernels:
    driver = gb._driver()
    dev = driver.device(ordinal)
    cc = driver.compute_capability(dev)
    built = _built(cc)
    try:
        return MeasureKernels(driver, dev, built)
    except RuntimeError:
        if not built.cached:
            raise
        built.forget()  # a cached image the driver rejects: compile afresh once
        return MeasureKernels(driver, dev, build(gb._nvrtc(), cc, cache=native.cache_dir(),
                                                 use_cache=False))


def kernels(device: torch.device | str) -> MeasureKernels:
    """``device``'s kernel, compiled (or loaded from the cache) on the first call."""
    index = torch.device(device).index
    return _kernels(torch.cuda.current_device() if index is None else index)


# ---- the table ----

def _fail(g: int, first: int, count: int, msg: str) -> ValueError:
    which = f"product {first}" if count == 1 else f"products {first}-{first + count - 1}"
    return ValueError(f"group {g} ({which}): {msg}")


def validate(groups: Sequence["MeasureGroup"], k: int) -> tuple[torch.device, np.ndarray]:
    """FS36's rules applied per group (dtype, shapes, ``k``, one device, a bias only with one
    member): ``(device, members per group)``, or a ``ValueError`` naming the group and its
    products. Nothing has been launched."""
    if len(groups) == 0:
        raise ValueError("no groups to measure")
    if not 1 <= k <= 255:
        raise ValueError(f"k must be in [1, 255], got {k}")
    device = None
    counts = []
    first = 0
    for g, grp in enumerate(groups):
        ts = {"a": grp.a, "b": grp.b, "p": grp.p}
        if grp.bias is not None:
            ts["bias"] = grp.bias
        for name, t in ts.items():
            if not isinstance(t, torch.Tensor):
                raise _fail(g, first, 1, f"{name} is not a tensor")
        nt = len(grp.ms)
        for name, t in ts.items():
            if t.dtype not in _DTYPE_CODE:
                raise _fail(g, first, max(nt, 1), f"{name} has dtype {t.dtype}, not float32 or "
                            "bfloat16")
        for name in ("a", "b", "p"):
            if ts[name].dim() != 3:
                raise _fail(g, first, max(nt, 1),
                            f"{name} is not 3-D (shape {tuple(ts[name].shape)})")
        if nt < 1:
            raise _fail(g, first, 1, "the group has no members")
        _, n, q = grp.a.shape
        w = grp.b.shape[2]
        if grp.a.shape[0] != nt or grp.b.shape[0] != nt or grp.p.shape[0] != nt \
                or grp.b.shape[1] != q or tuple(grp.p.shape[1:]) != (n, w):
            raise _fail(g, first, nt, f"{nt} labels for a {tuple(grp.a.shape)}, b "
                        f"{tuple(grp.b.shape)}, p {tuple(grp.p.shape)} do not make products")
        if n < 1 or q < 1 or w < 1:
            raise _fail(g, first, nt, f"a dimension is below 1 (n = {n}, q = {q}, w = {w})")
        if grp.bias is not None:
            if nt != 1:
                raise _fail(g, first, nt, "a bias is allowed only for a group of one member")
            if grp.bias.dim() != 1 or grp.bias.shape[0] != w:
                raise _fail(g, first, nt, f"bias has shape {tuple(grp.bias.shape)}, not ({w},)")
        devs = {t.device for t in ts.values()}
        if len(devs) != 1:
            raise _fail(g, first, nt, f"tensors on several devices {sorted(map(str, devs))}")
        dev = devs.pop()
        if device is None:
            device = dev
        elif dev != device:
            raise _fail(g, first, nt, f"device {dev} differs from the call's {device}")
        counts.append(nt)
        first += nt
    assert device is not None
    return device, np.asarray(counts, dtype=np.int64)


def _fill_view(dst: np.ndarray, at: int, t: torch.Tensor, nt: int) -> None:
    """Rows ``at .. at + nt`` of the view field ``dst`` for members ``0 .. nt`` of the 3-D
    ``t``: member ``i``'s base is ``t``'s base + ``i·stride(0)`` elements."""
    es = t.element_size()
    step = np.uint64(t.stride(0) * es)
    dst["addr"][at:at + nt] = np.uint64(t.data_ptr()) + np.arange(nt, dtype=np.uint64) * step
    dst["rows"][at:at + nt] = t.shape[1]
    dst["cols"][at:at + nt] = t.shape[2]
    dst["row_stride"][at:at + nt] = t.stride(1)
    dst["col_stride"][at:at + nt] = t.stride(2)
    dst["dtype"][at:at + nt] = _DTYPE_CODE[t.dtype]


def prods_array(groups: Sequence["MeasureGroup"], r_addrs: np.ndarray, k: int,
                guard: bool = True) -> np.ndarray:
    """The ``pots_mprod_t`` array of every member of ``groups`` in order, built by arithmetic
    on each group's views (no loop over members). ``r_addrs`` is each product's challenge
    matrix address. The bytes equal :func:`measure_table.pack`'s for the same members."""
    _, counts = validate(groups, k)
    n = int(counts.sum())
    r_addrs = np.asarray(r_addrs, dtype=np.uint64)
    if r_addrs.shape != (n,):
        raise ValueError(f"{r_addrs.shape} challenge addresses for {n} products")
    arr = np.zeros(n, dtype=MPROD_DTYPE)
    at = 0
    for grp, nt in zip(groups, counts):
        nt = int(nt)
        _fill_view(arr["a"], at, grp.a, nt)
        _fill_view(arr["b"], at, grp.b, nt)
        _fill_view(arr["p"], at, grp.p, nt)
        if grp.bias is not None:  # one member: a vector is a view with one row
            bv = arr["bias"]
            bv["addr"][at] = grp.bias.data_ptr()
            bv["rows"][at] = 1
            bv["cols"][at] = grp.bias.shape[0]
            bv["row_stride"][at] = 0
            bv["col_stride"][at] = grp.bias.stride(0)
            bv["dtype"][at] = _DTYPE_CODE[grp.bias.dtype]
        arr["flags"][at:at + nt] = (M_GUARD if guard else 0) | (M_BIASED if grp.bias is not None
                                                                 else 0)
        at += nt
    arr["r_addr"] = r_addrs
    arr["k"] = k
    return arr


def _scratch_sizes(prods: np.ndarray) -> np.ndarray:
    """``pots_mscratch_bytes`` of every product."""
    n = prods["a"]["rows"].astype(np.uint64)
    q = prods["a"]["cols"].astype(np.uint64)
    k = prods["k"].astype(np.uint64)
    f = k * q + np.uint64(2) * q + np.uint64(3) * n + k * n + k + np.uint64(1)
    return (np.uint64(8) * (n + np.uint64(1)) + np.uint64(4) * f + np.uint64(7)) & ~np.uint64(7)


def works_array(prods: np.ndarray, scratch_base: int) -> np.ndarray:
    """The ``pots_mwork_t`` array (one entry per product and a last one with the totals) for
    scratch starting at address ``scratch_base``: each product's scratch address and the first
    item of each phase."""
    n = len(prods)
    works = np.zeros(n + 1, dtype=WORK_DTYPE)
    sizes = _scratch_sizes(prods)
    works["scratch"][:n] = np.uint64(scratch_base) + np.concatenate(
        ([np.uint64(0)], np.cumsum(sizes)[:-1])).astype(np.uint64)
    items = np.stack((prods["b"]["rows"].astype(np.uint64)
                      + ((prods["flags"] & M_BIASED) != 0).astype(np.uint64),
                      prods["a"]["rows"].astype(np.uint64),
                      np.uint64(ROW_FIXED) + prods["k"].astype(np.uint64)), 1)
    works["start"][1:] = np.cumsum(items, axis=0, dtype=np.uint64)
    return works


def scratch_bytes(prods: np.ndarray) -> int:
    return int(_scratch_sizes(prods).sum())


def launches(prods_addr: int, works_addr: int, rows_addr: int, works: np.ndarray, k: int
             ) -> list[tuple[KernelArgs, int]]:
    """The three phases' ``(argument, blocks)``."""
    n = len(works) - 1
    out = []
    for phase in range(PHASES):
        total = int(works["start"][n, phase])
        blocks = min(-(-total // WARPS), MAX_BLOCKS)
        out.append((KernelArgs(prods_addr, works_addr, rows_addr, n, total, phase, k), blocks))
    return out


# ---- in flight ----

@dataclass
class Pending:
    """One call's work on the stream: the event after it, every tensor and buffer it reads or
    writes, and each input's ``_version`` as enqueued."""

    event: Any
    inputs: list[torch.Tensor]
    versions: list[int | None]
    keep: list[Any] = field(repr=False)

    def changed(self) -> list[int]:
        """The indices of the inputs whose ``_version`` moved."""
        return [i for i, (t, v) in enumerate(zip(self.inputs, self.versions))
                if v is not None and _version(t) != v]


def _version(t: torch.Tensor) -> int | None:
    try:
        return int(t._version)
    except RuntimeError:  # an inference tensor has no version counter
        return None


_inflight: dict[torch.device, list[Pending]] = {}


def _replay_error() -> type[Exception]:
    from verification.computation.interface import ReplayError

    return ReplayError


def _judge(entry: Pending) -> Exception | None:
    bad = entry.changed()
    if not bad:
        return None
    return _replay_error()(
        f"{len(bad)} tensor(s) read by the measuring kernel changed after it was enqueued "
        f"(input indices {bad[:8]}): a read that outlives its call must see the values it was "
        "enqueued with (invariant 6)")


def retire(device: torch.device) -> None:
    """Release the entries of ``device`` whose event has completed, checking their versions."""
    entries = _inflight.get(device, [])
    errors = []
    for e in list(entries):
        if e.event.query():
            entries.remove(e)
            err = _judge(e)
            if err is not None:
                errors.append(err)
    if errors:
        raise errors[0]


def drain(device: torch.device | str | None = None) -> None:
    """Wait for every entry of ``device`` (all devices when ``None``), then check versions and
    release them. Every entry is waited for and released before the first error is raised."""
    devs = list(_inflight) if device is None else [torch.device(device)]
    errors = []
    for d in devs:
        # a device given as "cuda" (no index) matches the entries of every CUDA device
        keys = [x for x in _inflight if x == d or (d.index is None and x.type == d.type)]
        for key in keys:
            entries, _inflight[key] = _inflight[key], []
            for e in entries:
                e.event.synchronize()
            for e in entries:
                err = _judge(e)
                if err is not None:
                    errors.append(err)
    if errors:
        raise errors[0]


# ---- the call ----

_DEFAULT_SOURCE = HostChallengeSource()


def enqueue(groups: Sequence["MeasureGroup"], *, h: bytes, k: int, guard: bool = True,
            challenges: ChallengeSource | None = None, run: Any = None) -> torch.Tensor:
    """Every member's row as float64 ``[Σ n_t, row_width(k)]`` on the groups' CUDA device,
    measured by the kernel on the current stream; nothing is read back and nothing waits for the
    device (see :func:`drain`). ``run`` replaces the loaded kernel (``run(launches, stream)``)
    and ``challenges`` the host source."""
    device, counts = validate(groups, k)
    if device.type != "cuda":
        raise ValueError(f"the measuring kernel needs CUDA tensors, got {device}")
    retire(device)
    stream = torch.cuda.current_stream(device)
    source = challenges if challenges is not None else _DEFAULT_SOURCE
    ms = [int(m) for grp in groups for m in grp.ms]
    widths = np.repeat([grp.b.shape[2] for grp in groups], counts)
    buf, offsets, ev = source.draw(h, ms, k, widths.tolist(), stream=stream)
    if ev is not None:
        stream.wait_event(ev)
    prods = prods_array(groups, np.uint64(buf.data_ptr()) + offsets.astype(np.uint64), k, guard)
    n = len(prods)
    scratch_n = scratch_bytes(prods)
    at_works = -(-prods.nbytes // 16) * 16
    total = at_works + WORK_DTYPE.itemsize * (n + 1)
    with torch.cuda.stream(stream):
        rows = torch.empty(n, row_width(k), dtype=torch.float64, device=device)
        scratch = torch.empty(max(scratch_n, 8), dtype=torch.uint8, device=device)
        table = torch.empty(total, dtype=torch.uint8, device=device)
    works = works_array(prods, scratch.data_ptr())
    host = torch.zeros(total, dtype=torch.uint8, pin_memory=True)
    hb = host.numpy()
    hb[:prods.nbytes] = prods.view(np.uint8).reshape(-1)
    hb[at_works:] = works.view(np.uint8).reshape(-1)
    work = launches(table.data_ptr(), table.data_ptr() + at_works, rows.data_ptr(), works, k)
    with torch.cuda.stream(stream):
        table.copy_(host, non_blocking=True)
        (run if run is not None else kernels(device).run)(work, stream.cuda_stream)
        event = torch.cuda.Event()
        event.record(stream)
    inputs = [t for grp in groups for t in (grp.a, grp.b, grp.p, grp.bias) if t is not None]
    _inflight.setdefault(device, []).append(
        Pending(event, inputs, [_version(t) for t in inputs],
                [host, table, buf, scratch, prods, works, rows]))
    return rows
