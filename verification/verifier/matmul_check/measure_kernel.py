"""The fused measuring kernel of check 5 (FS37, decision O2): its table, build and launch.

``measure_core.h`` defines every number of a product's row on logical indices;
``measure_phases.h`` splits that work into items and steps (a dot or sum whose 32 partials are
the 32 lanes) and holds the per-element updates; ``measure_kernel.cu`` runs one sweep per row
(FS37b): a block takes ``ROWS_PER_BLOCK`` rows of one read group (its group and row block come
from a per-level block table, :func:`block_table`) and runs the group's entries one after
another over them, walks the columns in ``TILE_COLS`` tiles through shared memory and the
challenge columns in ``K_CHUNK`` chunks, and reduces each lane's accumulators with a transposed
butterfly of shuffles, staged through shared memory. ``K_CHUNK`` is ``min(k, cap)`` per build (:func:`chunk_for`), one
cached image per value. :meth:`MeasureKernels.attributes` reports the build's registers, spills
and shared bytes. The host test (``measure_core_host.c``) runs the same updates
in loops (``pots_host_measure_sweep``), so the GPU's bits are testable without a GPU.

Compiled at first use by NVRTC (through ``gpu_blake3``'s :class:`Nvrtc`, :class:`Driver` and
:func:`build_image`) with ``--fmad=false`` into a CUBIN cached under ``native.cache_dir()``;
importing this module compiles and loads nothing. One call (:func:`enqueue`) packs every group's
products into one table in one pinned buffer, copies it up with one non-blocking copy and
launches one kernel per level of a plan on the current stream: :func:`plain_plan` (every node
its own group, three levels) or, with ``share=True`` (FS38b), :func:`read_plan` (products that
read one stored view run their rows together); the rows stay on the device. The tile geometry
is a set of ``-D`` options in the cache key; :func:`set_geometry` changes it (never a bit). Each
call's :class:`Pending` holds every tensor and buffer the stream reads, with each input's
``_version``, until :func:`drain` has waited for its event and found no input changed.

Overlap (FS39, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.5): every CUDA device has one *measure
stream* (:func:`measure_stream`). :func:`enqueue` makes it wait on an event recorded on the
current stream (the producer of the operands) and then draws the challenges, copies the table,
launches the levels and records the call's event there, so the glue that follows on the current
stream runs beside the measuring. The measure stream's tensors (``rows``, the table, the
challenge buffer, and the ``P`` copies of check 5 made under :func:`stream_for`) are used only
on that stream until a :func:`drain` has waited for the call's event, so the caching allocator
reusing their blocks is stream-ordered and needs no ``record_stream``; the inputs made on the
current stream are held by :class:`Pending` until that event, which is what keeps their blocks
out of reuse (the capture's staging does the same, ``prover/capture.py``). :func:`overlap`
turns the whole thing off for tests. :func:`prefetch` hands the draw of a later call to the
challenge source's worker thread (``challenge_source``).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
from collections.abc import Iterator, Sequence
from collections import OrderedDict
from contextlib import contextmanager
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
    Prefetch,
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
LEVEL_KERNEL = "pots_measure_level"
OPTIONS = ("--fmad=false",)
"""Extra NVRTC options, part of the cache key: no fused multiply-add but the core's ``fmaf``."""
WARPS = 8  # warps per block, POTS_MK_WARPS in the .cu
ROWS_PER_BLOCK = 16  # rows of a view per block (a multiple of WARPS), a -D option of the .cu
TILE_COLS = 64  # columns per shared-memory tile (a multiple of 32), a -D option
K_CHUNK = 16  # cap on the challenge columns per sweep; a build uses min(k, K_CHUNK) (chunk_for)
SDESC_BYTES = 208 + 16  # the product copy in shared memory (pots_sdesc_t)
SHARED_LIMIT = 48 * 1024  # static shared memory per block
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
READ_DTYPE = np.dtype([("first", "<u8"), ("count", "<u8"), ("entry0", "<u4"),
                       ("n_entries", "<u4")])
"""``pots_mread_t`` (24 bytes): a read group of one level (FS38b)."""
ENTRY_DTYPE = np.dtype([("prod", "<u4"), ("phase", "<u4")])
"""``pots_mentry_t`` (8 bytes): one node (product, phase) of a read group."""
BLK_DTYPE = np.dtype([("group", "<u4"), ("rbk", "<u4")])
"""``pots_mblk_t`` (8 bytes): one block's group (within its level) and row block."""
assert _MVIEW.itemsize == ctypes.sizeof(MView) and MPROD_DTYPE.itemsize == ctypes.sizeof(MProd)


class LevelArgs(ctypes.Structure):
    """``pots_mlargs_t``: one level launch's argument (FS38b, FS37b): the tables, the level's
    groups and ``blocks`` (``pots_mblk_t[n_blocks]``, each block's group and row block)."""

    _fields_ = [("prods", ctypes.c_uint64), ("works", ctypes.c_uint64), ("rows", ctypes.c_uint64),
                ("reads", ctypes.c_uint64), ("entries", ctypes.c_uint64),
                ("blocks", ctypes.c_uint64), ("n_groups", ctypes.c_uint64),
                ("n_blocks", ctypes.c_uint64), ("k", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


assert ctypes.sizeof(LevelArgs) == 72


# ---- the geometry (FS37b) ----

def shared_bytes(rows_per_block: int, tile_cols: int, k_chunk: int) -> int:
    """The kernel's static shared memory, every ``__shared__`` array of ``pots_measure_level``:
    the product copy (:data:`SDESC_BYTES`), the float64 sums (``rows_per_block`` doubles), the x
    tile (``rows_per_block`` rows of ``tile_cols + 1`` floats), the vector tile (``k_chunk`` rows
    of ``tile_cols + 1``), ``b2`` and ``b1`` (``tile_cols`` each) and the A and P staging arrays
    (``rows_per_block x (k_chunk + 2)`` each)."""
    return (SDESC_BYTES + 8 * rows_per_block
            + 4 * (rows_per_block * (tile_cols + 1) + k_chunk * (tile_cols + 1) + 2 * tile_cols
                   + 2 * rows_per_block * (k_chunk + 2)))


def set_geometry(*, rows_per_block: int | None = None, tile_cols: int | None = None,
                 k_chunk: int | None = None) -> tuple[int, int, int]:
    """Set the kernel's geometry and return the previous ``(rows_per_block, tile_cols,
    k_chunk)``; with no argument nothing changes. The values are NVRTC ``-D`` options, so they
    key the cached CUBIN, and the loaded-kernel caches are cleared. ``ValueError`` (nothing
    changed) unless ``tile_cols`` is a positive multiple of 32, ``rows_per_block`` a positive
    multiple of :data:`WARPS`, ``k_chunk >= 1`` and the static shared memory fits 48 KB. The bits never depend on the geometry."""
    global ROWS_PER_BLOCK, TILE_COLS, K_CHUNK
    prev = (ROWS_PER_BLOCK, TILE_COLS, K_CHUNK)
    rpb = prev[0] if rows_per_block is None else rows_per_block
    tc = prev[1] if tile_cols is None else tile_cols
    kc = prev[2] if k_chunk is None else k_chunk
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (rpb, tc, kc)):
        raise ValueError("the geometry values must be integers")
    if tc < 1 or tc % 32:
        raise ValueError(f"tile_cols must be a positive multiple of 32, got {tc}")
    if rpb < 1 or rpb % WARPS:
        raise ValueError(f"rows_per_block must be a positive multiple of {WARPS}, got {rpb}")
    if kc < 1:
        raise ValueError(f"k_chunk must be at least 1, got {kc}")
    if shared_bytes(rpb, tc, kc) > SHARED_LIMIT:
        raise ValueError(f"geometry ({rpb}, {tc}, {kc}) needs {shared_bytes(rpb, tc, kc)} bytes "
                         f"of shared memory, over {SHARED_LIMIT}")
    if (rpb, tc, kc) != prev:
        ROWS_PER_BLOCK, TILE_COLS, K_CHUNK = rpb, tc, kc
        _kernels.cache_clear()
        _built.cache_clear()
    return prev


def chunk_for(k: int | None) -> int:
    """The ``K_CHUNK`` macro of a build for ``k`` challenge columns: ``min(k, K_CHUNK)``, so the
    accumulator arrays are exactly ``k`` wide when ``k`` is under the cap (``k=9``: nine, not
    sixteen behind a predicate). ``K_CHUNK`` (:func:`set_geometry`'s ``k_chunk``) stays the cap;
    without ``k`` the build is the cap's. The chunking never changes a bit."""
    return K_CHUNK if k is None else max(1, min(int(k), K_CHUNK))


def build_options(k_chunk: int | None = None) -> tuple[str, ...]:
    """The NVRTC options of the current geometry: ``--fmad=false`` and the four ``-D`` macros
    (part of the cache key, so another geometry, or another ``k_chunk``, is another CUBIN).
    ``k_chunk`` is the ``K_CHUNK`` value (default: the cap)."""
    kc = K_CHUNK if k_chunk is None else k_chunk
    return (*OPTIONS, f"-DPOTS_MK_WARPS={WARPS}u", f"-DROWS_PER_BLOCK={ROWS_PER_BLOCK}u",
            f"-DTILE_COLS={TILE_COLS}u", f"-DK_CHUNK={kc}u")


# ---- the build ----

def source() -> gb.Source:
    """The kernel, its two headers and the cache prefix, for :func:`gpu_blake3.build_image`."""
    return gb.Source(CU_NAME, (HERE / CU_NAME).read_text(),
                     {CORE_NAME: (HERE / CORE_NAME).read_text(),
                      PHASES_NAME: (HERE / PHASES_NAME).read_text()}, "measure_kernel")


def source_hash() -> str:
    return source().digest()


def build(nvrtc: gb.Nvrtc, cc: tuple[int, int], *, cache: Path | None = None,
          use_cache: bool = True, k_chunk: int | None = None) -> gb.Built:
    """The compiled image for compute capability ``cc`` and ``K_CHUNK = k_chunk`` (default: the
    cap), cached in ``cache`` when given: one image per ``K_CHUNK`` value."""
    return gb.build_image(nvrtc, cc, cache=cache, use_cache=use_cache, source=source(),
                          extra_options=build_options(k_chunk))


class MeasureKernels:
    """A device's loaded module and its level kernel."""

    def __init__(self, driver: gb.Driver, dev: int, built: gb.Built) -> None:
        self.driver = driver
        self.dev = dev
        self.built = built
        driver.push(dev)
        try:
            self.module = driver.load_module(built.image)
            self.level_fn = driver.function(self.module, LEVEL_KERNEL)
        finally:
            driver.pop()

    def attributes(self) -> dict[str, int]:
        """The level kernel's build, read through ``cuFuncGetAttribute``: ``registers`` per
        thread, ``local_bytes`` per thread (nonzero means registers spilled), ``shared_bytes``
        (static) and ``max_threads`` per block. Printed by the A40 run, so a spill shows."""
        attrs = {"max_threads": 0, "shared_bytes": 1, "local_bytes": 3, "registers": 4}
        self.driver.push(self.dev)
        try:
            return {name: self.driver.function_attribute(self.level_fn, a)
                    for name, a in attrs.items()}
        finally:
            self.driver.pop()

    def run(self, launches: Sequence[tuple[LevelArgs, int]], stream: int) -> None:
        """Launch each ``(args, blocks)`` in order on the raw ``stream``: ``pots_measure_level``
        on the :class:`LevelArgs`."""
        self.driver.push(self.dev)
        try:
            for args, blocks in launches:
                self.driver.launch(self.level_fn, blocks, WARPS * 32, args, stream)  # pyright: ignore
        finally:
            self.driver.pop()


@functools.lru_cache(maxsize=None)
def _built(cc: tuple[int, int], kc: int) -> gb.Built:
    return build(gb._nvrtc(), cc, cache=native.cache_dir(), k_chunk=kc)


@functools.lru_cache(maxsize=None)
def _kernels(ordinal: int, kc: int) -> MeasureKernels:
    driver = gb._driver()
    dev = driver.device(ordinal)
    cc = driver.compute_capability(dev)
    built = _built(cc, kc)
    try:
        return MeasureKernels(driver, dev, built)
    except RuntimeError:
        if not built.cached:
            raise
        built.forget()  # a cached image the driver rejects: compile afresh once
        return MeasureKernels(driver, dev, build(gb._nvrtc(), cc, cache=native.cache_dir(),
                                                 use_cache=False, k_chunk=kc))


def kernels(device: torch.device | str, k: int | None = None) -> MeasureKernels:
    """``device``'s kernel for ``k`` challenge columns (``K_CHUNK = chunk_for(k)``; without
    ``k``, the cap's), compiled (or loaded from the cache) on the first call per ``K_CHUNK``."""
    index = torch.device(device).index
    return _kernels(torch.cuda.current_device() if index is None else index, chunk_for(k))


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
                guard: bool = True, *,
                _validated: tuple[torch.device, np.ndarray] | None = None) -> np.ndarray:
    """The ``pots_mprod_t`` array of every member of ``groups`` in order, built by arithmetic
    on each group's views (no loop over members). ``r_addrs`` is each product's challenge
    matrix address. The bytes equal :func:`measure_table.pack`'s for the same members."""
    _, counts = _validated if _validated is not None else validate(groups, k)
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


# ---- the read plan (FS38b) ----

MAX_ROUNDS = 64


@dataclass(frozen=True)
class ReadPlan:
    """A table's shared reads, in levels (see :func:`read_plan`)."""

    reads: np.ndarray        # READ_DTYPE, every level's groups then its sentinel, levels in order
    level_off: np.ndarray    # uint64 [n_levels + 1], level l = reads[level_off[l]:level_off[l+1]]
    entries: np.ndarray      # ENTRY_DTYPE
    node_level: np.ndarray   # int64 [n_prod, 3], the level of (product, phase)
    n_levels: int            # == launches
    shared_groups: int       # groups with 2 or more entries
    cycle: bool              # True when the levels did not settle in 64 rounds: no group shares


def _groups(gid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(order, starts)`` of dense group ids ``gid``: the nodes sorted by group (ascending node
    index within each) and where each group starts in that order."""
    order = np.argsort(gid, kind="stable")
    sg = gid[order]
    return order, np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])


def _settle(gid: np.ndarray, order: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray | None:
    """The levels ``[n, 3]`` of nodes in share groups ``gid`` (dense ids per node, node ``3t +
    p``; ``order`` and ``starts`` from :func:`_groups`), by the fixed point of the plan: a group
    takes its largest level, ``L(t, 1) >= L(t, 0) + 1`` and ``L(t, 2) >= L(t, 1) + 1``; ``None``
    when 64 rounds do not settle it."""
    lv = np.tile(np.arange(3, dtype=np.int64), (n, 1))
    for _ in range(MAX_ROUNDS):
        gmax = np.maximum.reduceat(lv.reshape(-1)[order], starts)
        new = gmax[gid].reshape(n, 3)
        new[:, 1] = np.maximum(new[:, 1], new[:, 0] + 1)
        new[:, 2] = np.maximum(new[:, 2], new[:, 1] + 1)
        if np.array_equal(new, lv):
            return lv
        lv = new
    return None


def plain_plan(prods: np.ndarray) -> ReadPlan:
    """The plan without shared reads (FS37b): three levels (every phase-0 node, then phase 1,
    then phase 2), every node its own group, the groups ordered as :func:`read_plan` orders
    unshared nodes. ``enqueue(share=False)`` runs it through the level kernel."""
    return _plan(prods, share=False)


def read_plan(prods: np.ndarray) -> ReadPlan:
    """Which products' passes read one stored view, and the levels they run in.

    A node ``(t, 0)`` is product ``t``'s pass over ``B``'s rows and ``(t, 1)`` its pass over
    ``A``'s rows, keyed on the view as the kernel reads it (address, rows, columns, both
    strides, dtype); nodes with equal keys form a group that runs together, one row of the view
    at a time. ``(t, 2)`` and a biased ``(t, 0)`` are never shared. Levels start at the phase
    and settle by the fixed point of :func:`_settle`; a table that does not settle (a cycle)
    drops every share (each node alone at its phase). Within a level the
    groups are ordered by the keyed view's address (phase 2: ``P``'s), ties by the first entry's
    ``(prod, phase)``; entries of a group ascend by ``(prod, phase)``.

    A legitimate dependency chain that needs more than 64 rounds is treated as a cycle and
    loses its shares (a cost in speed, never in bits); the decoder's tables settle in 4 levels."""
    return _plan(prods, share=True)


def _plan(prods: np.ndarray, share: bool) -> ReadPlan:
    n = len(prods)
    n3 = 3 * n
    biased = (prods["flags"] & M_BIASED) != 0
    # node 3t + p: the viewed field per node; only (t, 1) and an unbiased (t, 0) can share
    cols = {f: np.empty(n3, dtype=np.int64 if f != "addr" else np.uint64)
            for f in ("addr", "rows", "cols", "row_stride", "col_stride", "dtype")}
    for p, v in ((0, prods["b"]), (1, prods["a"]), (2, prods["p"])):
        for f, dst in cols.items():
            dst[p::3] = v[f]
    addr = cols["addr"]
    alone = np.zeros(n3, dtype=bool)
    alone[2::3] = True
    alone[0::3] = biased
    if not share:
        alone[:] = True
    items = np.stack((prods["b"]["rows"].astype(np.int64) + biased.astype(np.int64),
                      prods["a"]["rows"].astype(np.int64),
                      np.int64(ROW_FIXED) + prods["k"].astype(np.int64)), 1).reshape(-1)
    gid = np.empty(n3, dtype=np.int64)
    sh = np.flatnonzero(~alone)
    if len(sh):
        rows, cs = cols["rows"][sh], cols["cols"][sh]
        rs, cst = cols["row_stride"][sh], cols["col_stride"][sh]
        dt = cols["dtype"][sh]
        if (min(rs.min(), cst.min(), rows.min(), cs.min()) >= 0 and rs.max() < (1 << 56)
                and cst.max() < (1 << 56) and rows.max() < (1 << 32) and cs.max() < (1 << 32)
                and dt.max() < (1 << 8)):
            packed = (addr[sh], ((rows << 32) | cs).astype(np.uint64), rs.astype(np.uint64),
                      ((cst << 8) | dt).astype(np.uint64))
            o = np.lexsort(packed[::-1])  # the first array is the primary key
            diff = np.zeros(len(sh), dtype=bool)
            diff[0] = True
            for c in packed:
                cc = c[o]
                diff[1:] |= cc[1:] != cc[:-1]
            ids = np.empty(len(sh), dtype=np.int64)
            ids[o] = np.cumsum(diff) - 1
        else:  # a negative or very large stride or extent: the generic key
            skey = np.empty(len(sh), dtype=[("a", "<u8"), ("r", "<i8"), ("c", "<i8"),
                                            ("rs", "<i8"), ("cs", "<i8"), ("d", "<i8")])
            for f, c in zip(("a", "r", "c", "rs", "cs", "d"),
                            (addr[sh], rows, cs, rs, cst, dt)):
                skey[f] = c
            ids = np.unique(skey, return_inverse=True)[1].reshape(-1).astype(np.int64)
        gid[sh] = ids
        n_sh = int(ids.max()) + 1
    else:
        n_sh = 0
    al = np.flatnonzero(alone)
    gid[al] = n_sh + np.arange(len(al), dtype=np.int64)
    order, starts = _groups(gid)
    lv = _settle(gid, order, starts, n)
    cycle = lv is None
    if cycle:
        gid = np.arange(n3, dtype=np.int64)
        lv = np.tile(np.arange(3, dtype=np.int64), (n, 1))
        order, starts = _groups(gid)
    # per group: its entries (ascending node index), level, address, items
    n_ent = np.diff(np.r_[starts, n3])
    first = order[starts]  # the group's first entry (lowest node index)
    g_level = lv.reshape(-1)[first]
    g_addr = addr[first]
    g_items = items[first]
    g_order = np.lexsort((first, g_addr, g_level))  # the last key is the primary one
    ng = len(first)
    rank = np.empty(ng, dtype=np.int64)
    rank[g_order] = np.arange(ng)
    n_ent_s = n_ent[g_order]
    entry0 = np.cumsum(n_ent_s) - n_ent_s
    lv_s = g_level[g_order]
    cnt_s = g_items[g_order]
    n_levels = int(lv.max()) + 1
    per_level = np.bincount(lv_s, minlength=n_levels)
    level_off = np.zeros(n_levels + 1, dtype=np.uint64)
    level_off[1:] = np.cumsum(per_level + 1)
    reads = np.zeros(int(level_off[-1]), dtype=READ_DTYPE)
    csum = np.cumsum(cnt_s) - cnt_s  # items before each group, in table order
    g_start = np.cumsum(per_level) - per_level  # index of each level's first group
    base = csum[np.minimum(g_start, ng - 1)]  # items before each level's first group
    pos = np.arange(ng) + lv_s  # one sentinel per earlier level
    reads["first"][pos] = (csum - base[lv_s]).astype(np.uint64)
    reads["count"][pos] = cnt_s.astype(np.uint64)
    reads["entry0"][pos] = entry0.astype(np.uint32)
    reads["n_entries"][pos] = n_ent_s.astype(np.uint32)
    totals = np.bincount(lv_s, weights=cnt_s, minlength=n_levels).astype(np.uint64)
    reads["first"][level_off[1:].astype(np.int64) - 1] = totals
    e_order = np.argsort(rank[gid] * n3 + np.arange(n3, dtype=np.int64))
    entries = np.zeros(n3, dtype=ENTRY_DTYPE)
    entries["prod"] = (e_order // 3).astype(np.uint32)
    entries["phase"] = (e_order % 3).astype(np.uint32)
    shared = n_ent >= 2
    return ReadPlan(reads, level_off, entries, lv.copy(), n_levels,
                    0 if cycle else int(shared.sum()), bool(cycle))


def block_starts(plan: ReadPlan, rows_per_block: int | None = None) -> np.ndarray:
    """The row-block starts of every level, uint64, one per entry of ``plan.reads`` (so level
    ``l``'s array is ``[level_off[l]:level_off[l + 1]]``): a phase-0 or phase-1 group takes
    ``ceil(count_g / rows_per_block)`` blocks, a phase-2 group one (its warps take the ``5 + k``
    items); the entry after the last group holds the level's block total."""
    rpb = ROWS_PER_BLOCK if rows_per_block is None else rows_per_block
    out = np.zeros(len(plan.reads), dtype=np.uint64)
    off = plan.level_off.astype(np.int64)
    for lv in range(plan.n_levels):
        lo, hi = int(off[lv]), int(off[lv + 1])
        r = plan.reads[lo:hi - 1]
        two = plan.entries["phase"][r["entry0"].astype(np.int64)] == 2
        nb = np.where(two, 1, -(-r["count"].astype(np.int64) // rpb))
        out[lo + 1:hi] = np.cumsum(nb).astype(np.uint64)
    return out


def block_table(plan: ReadPlan, rows_per_block: int | None = None) -> np.ndarray:
    """The block table of every level (``BLK_DTYPE``), levels one after another: level ``l``'s
    blocks in group order, group ``g`` taking blocks ``block_starts[g] .. block_starts[g + 1]``
    of the level with row blocks ``0, 1, ...``. The kernel reads its block's entry and searches
    nothing."""
    st = block_starts(plan, rows_per_block).astype(np.int64)
    off = plan.level_off.astype(np.int64)
    parts = []
    for lv in range(plan.n_levels):
        lo, hi = int(off[lv]), int(off[lv + 1])
        s = st[lo:hi]
        nb = np.diff(s)
        t = np.zeros(int(s[-1]), dtype=BLK_DTYPE)
        t["group"] = np.repeat(np.arange(hi - lo - 1), nb)
        t["rbk"] = np.arange(int(s[-1])) - np.repeat(s[:-1], nb)
        parts.append(t)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=BLK_DTYPE)


def level_launches(prods_addr: int, works_addr: int, rows_addr: int, reads_addr: int,
                   entries_addr: int, blocks_addr: int, plan: ReadPlan, k: int,
                   starts: np.ndarray) -> list[tuple[LevelArgs, int]]:
    """One ``(argument, blocks)`` per level of ``plan``, in level order; ``reads_addr`` is the
    address of ``plan.reads[0]``, ``entries_addr`` that of ``plan.entries[0]`` and
    ``blocks_addr`` that of :func:`block_table`'s first row; ``starts`` is :func:`block_starts`."""
    out = []
    off = plan.level_off.astype(np.int64)
    done = 0
    for lv in range(plan.n_levels):
        lo, hi = int(off[lv]), int(off[lv + 1])
        n_blocks = int(starts[hi - 1])
        out.append((LevelArgs(prods_addr, works_addr, rows_addr,
                              reads_addr + lo * READ_DTYPE.itemsize, entries_addr,
                              blocks_addr + done * BLK_DTYPE.itemsize, hi - lo - 1, n_blocks,
                              k, 0),
                    min(n_blocks, MAX_BLOCKS)))
        done += n_blocks
    return out


# ---- the table's structure, built once per run (FS51, HOST_SIDE_PLAN_DRAFT §2.2) ----

TABLE_CACHE_MAX = 64
"""Cached structures at most (the least recently used goes first). One per schedule step and
device: Llama's 24 steps a device are well under it."""

_cache_on = True
_cache: "OrderedDict[tuple, list[TableStructure]]" = OrderedDict()
_cache_n = 0
_stats = {"hits": 0, "misses": 0, "plan_rebuilds": 0}


@contextmanager
def table_cache(enabled: bool = True) -> Iterator[None]:
    """Inside, the structure cache of :func:`table_structure` is on (``True``) or off (``False``:
    every call builds its table from scratch, looks nothing up and stores nothing). Off is for
    the comparison tests and for calls whose structure never comes again, such as the canonical
    path's single runs: they would push the unit pass's entries out of the bounded cache.
    Nests; outside any ``table_cache`` it is on. The rows are the same bits either way."""
    global _cache_on
    before, _cache_on = _cache_on, bool(enabled)
    try:
        yield
    finally:
        _cache_on = before


def table_cache_on() -> bool:
    return _cache_on


def table_cache_stats(reset: bool = False) -> dict[str, int]:
    """``{"hits", "misses", "plan_rebuilds"}`` since the last reset: calls that found their
    structure, calls that built one, and hits whose cached read plan no longer held (a shared
    view moved) and was rebuilt. Counted only while the cache is on."""
    out = dict(_stats)
    if reset:
        for key in _stats:
            _stats[key] = 0
    return out


def table_cache_clear() -> None:
    """Forget every cached structure (the counters stay)."""
    global _cache_n
    _cache.clear()
    _cache_n = 0


@dataclass
class _Built:
    """What a table has that its call's addresses do not change: one build of the pieces below
    from a first call's products."""

    plan: ReadPlan
    starts: np.ndarray
    scratch_n: int
    works: np.ndarray       # WORK_DTYPE, scratch offsets relative to a base of 0
    at_works: int
    end_works: int
    at_reads: int
    at_entries: int
    at_blocks: int
    total: int
    tail: np.ndarray        # uint8, the table's bytes from ``at_reads`` on (reads, entries, blocks)
    launches: list[tuple[int, int, int, int, int]]
    # (reads byte offset, blocks byte offset, groups, blocks, grid) per level
    shared_nodes: np.ndarray  # int64: the nodes 3t + p of every read group with 2+ entries
    shared_first: np.ndarray  # int64: the first node of each such node's group


@dataclass
class TableStructure:
    """One cache entry: the key (each group's ``ms``, view shapes, strides and dtypes), the
    flat ``ms`` and widths in table order (tuples, so the challenge source can key on them), and
    the :class:`_Built` pieces, filled by the first :func:`build_host_table`."""

    key: tuple
    counts: np.ndarray
    ms: tuple[int, ...]
    widths: tuple[int, ...]
    built: _Built | None = None


def _view_key(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.stride(), t.dtype)


def _group_key(grp: "MeasureGroup") -> tuple:
    return (tuple(grp.ms), _view_key(grp.a), _view_key(grp.b), _view_key(grp.p),
            None if grp.bias is None else _view_key(grp.bias))


def table_structure(groups: Sequence["MeasureGroup"], k: int, guard: bool = True,
                    share: bool = False, *,
                    _validated: tuple[torch.device, np.ndarray] | None = None) -> TableStructure:
    """The cache entry of this call's structure: found (a hit) when an earlier call had the same
    device, ``k``, ``guard``, ``share``, row-block size and, group by group, the same ``ms``
    (compared as tuples) and a/b/p/bias shapes, strides and dtypes; else a new entry (a miss),
    stored while the cache is on. :func:`validate` runs on every call, so an invalid call fails
    as before and nothing invalid is cached."""
    global _cache_n
    device, counts = _validated if _validated is not None else validate(groups, k)
    keys = tuple(_group_key(g) for g in groups) if _cache_on else ()
    if _cache_on:
        bucket = (device, k, guard, share, ROWS_PER_BLOCK, len(groups), int(keys[0][0][0]),
                  int(keys[-1][0][-1]))
        for e in _cache.get(bucket, ()):
            if e.key == keys:
                _stats["hits"] += 1
                _cache.move_to_end(bucket)
                return e
        _stats["misses"] += 1
    ms = tuple(int(m) for grp in groups for m in grp.ms)
    widths = tuple(np.repeat([grp.b.shape[2] for grp in groups], counts).tolist())
    entry = TableStructure(keys, counts, ms, widths)
    if _cache_on:
        _cache.setdefault(bucket, []).append(entry)
        _cache.move_to_end(bucket)
        _cache_n += 1
        while _cache_n > TABLE_CACHE_MAX and len(_cache) > 1:
            _, old = _cache.popitem(last=False)
            _cache_n -= len(old)
        if _cache_n > TABLE_CACHE_MAX:  # one bucket left over the cap: its oldest entries go
            drop = _cache_n - TABLE_CACHE_MAX
            del _cache[bucket][:drop]
            _cache_n -= drop
    return entry


def _shared_nodes(plan: ReadPlan) -> tuple[np.ndarray, np.ndarray]:
    """The nodes of every read group with two or more entries and, per node, its group's first
    node (the validity check of :func:`_plan_holds`)."""
    n_ent = plan.reads["n_entries"].astype(np.int64)
    sel = np.flatnonzero(n_ent >= 2)
    if len(sel) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cnt = n_ent[sel]
    e0 = plan.reads["entry0"].astype(np.int64)[sel]
    before = np.cumsum(cnt) - cnt
    at = np.repeat(e0, cnt) + np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(before, cnt)
    nodes = (plan.entries["prod"].astype(np.int64)[at] * 3 + plan.entries["phase"][at])
    return nodes, np.repeat(nodes[before], cnt)


def _build(prods: np.ndarray, share: bool) -> _Built:
    """The call-independent pieces for products ``prods``, as :func:`enqueue` laid them out
    before the cache: the plan, ``block_starts``, the buffer layout, the work array with scratch
    offsets from 0, and the plan's reads, entries and blocks as one byte tail."""
    n = len(prods)
    at_works = -(-prods.nbytes // 16) * 16
    end_works = at_works + WORK_DTYPE.itemsize * (n + 1)
    plan = read_plan(prods) if share else plain_plan(prods)
    at_reads = -(-end_works // 16) * 16
    at_entries = -(-(at_reads + plan.reads.nbytes) // 16) * 16
    at_blocks = -(-(at_entries + plan.entries.nbytes) // 16) * 16
    starts = block_starts(plan)
    blocks = block_table(plan)
    total = at_blocks + blocks.nbytes
    tail = np.zeros(total - at_reads, dtype=np.uint8)
    tail[:plan.reads.nbytes] = plan.reads.view(np.uint8).reshape(-1)
    tail[at_entries - at_reads:at_entries - at_reads + plan.entries.nbytes] = \
        plan.entries.view(np.uint8).reshape(-1)
    tail[at_blocks - at_reads:] = blocks.view(np.uint8).reshape(-1)
    launches = []
    off = plan.level_off.astype(np.int64)
    done = 0
    for lv in range(plan.n_levels):
        lo, hi = int(off[lv]), int(off[lv + 1])
        n_blocks = int(starts[hi - 1])
        launches.append((lo * READ_DTYPE.itemsize, done * BLK_DTYPE.itemsize, hi - lo - 1,
                         n_blocks, min(n_blocks, MAX_BLOCKS)))
        done += n_blocks
    nodes, first = _shared_nodes(plan)
    return _Built(plan, starts, scratch_bytes(prods), works_array(prods, 0), at_works, end_works,
                  at_reads, at_entries, at_blocks, total, tail, launches, nodes, first)


def _plan_holds(built: _Built, prods: np.ndarray) -> bool:
    """Whether every node of a shared read group of the cached plan still reads the same stored
    view as its group's first node, by the keys of :func:`_plan` (address, rows, columns, both
    strides, dtype) taken from ``prods``: one gather and compare per field, no sort. Nodes the
    plan keeps apart that now share a view are not looked for: a missed share costs speed, never
    bits (FS38b)."""
    nodes = built.shared_nodes
    if len(nodes) == 0:
        return True
    first = built.shared_first
    for f in ("addr", "rows", "cols", "row_stride", "col_stride", "dtype"):
        flat = np.stack((prods["b"][f], prods["a"][f], prods["p"][f]), 1).reshape(-1)
        if not np.array_equal(flat[nodes], flat[first]):
            return False
    return True


@dataclass
class HostTable:
    """One call's table on the host: its products (this call's addresses), the structure's
    :class:`_Built` pieces, and what :func:`enqueue` does with them."""

    prods: np.ndarray
    built: _Built

    @property
    def plan(self) -> ReadPlan:
        return self.built.plan

    @property
    def total(self) -> int:
        return self.built.total

    @property
    def scratch_n(self) -> int:
        return self.built.scratch_n

    def works(self, scratch_base: int) -> np.ndarray:
        """:func:`works_array` for scratch from ``scratch_base``: the cached offsets plus it."""
        works = self.built.works.copy()
        n = len(self.prods)
        works["scratch"][:n] += np.uint64(scratch_base)
        return works

    def fill(self, hb: np.ndarray, scratch_base: int) -> np.ndarray:
        """Write the table's bytes into ``hb`` (uint8, ``total`` long, zeroed); returns the work
        array."""
        b = self.built
        works = self.works(scratch_base)
        hb[:self.prods.nbytes] = self.prods.view(np.uint8).reshape(-1)
        hb[b.at_works:b.end_works] = works.view(np.uint8).reshape(-1)
        hb[b.at_reads:b.total] = b.tail
        return works

    def to_bytes(self, scratch_base: int) -> np.ndarray:
        out = np.zeros(self.total, dtype=np.uint8)
        self.fill(out, scratch_base)
        return out

    def launches(self, table_addr: int, rows_addr: int, k: int) -> list[tuple[LevelArgs, int]]:
        """What :func:`level_launches` returns for this table copied to ``table_addr``."""
        b = self.built
        reads, blocks = table_addr + b.at_reads, table_addr + b.at_blocks
        return [(LevelArgs(table_addr, table_addr + b.at_works, rows_addr, reads + ro,
                           table_addr + b.at_entries, blocks + bo, ng, nb, k, 0), grid)
                for ro, bo, ng, nb, grid in b.launches]


def build_host_table(groups: Sequence["MeasureGroup"], r_addrs: np.ndarray, k: int,
                     guard: bool = True, share: bool = False,
                     structure: TableStructure | None = None, *,
                     _validated: tuple[torch.device, np.ndarray] | None = None) -> HostTable:
    """The host side of one :func:`enqueue` call, testable on CPU tensors: the products of
    ``groups`` (:func:`prods_array`, this call's addresses and challenge addresses ``r_addrs``)
    over the structure of :func:`table_structure`. A structure's first call builds the read
    plan, ``block_starts``, the block table and the work offsets; later calls reuse them after
    :func:`_plan_holds` has checked the shared reads; a plan that no longer holds is rebuilt, it
    replaces the entry's and the rebuild is counted."""
    if structure is None:
        structure = table_structure(groups, k, guard, share, _validated=_validated)
    prods = prods_array(groups, r_addrs, k, guard, _validated=_validated)
    built = structure.built
    if built is None:
        built = structure.built = _build(prods, share)
    elif share and not _plan_holds(built, prods):
        built = structure.built = _build(prods, share)
        _stats["plan_rebuilds"] += 1
    return HostTable(prods, built)


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
    release them. Every entry is waited for and released before the first error is raised.
    With no device it is also a barrier for the measure streams (FS39): everything enqueued on
    them is waited for, the row joins check 5 makes there after a call's event included, so a
    read on any stream afterwards sees finished rows. With a device (check 5's one step in
    flight per device) only the entries are waited for, so the ``P`` copies of the step being
    served keep running."""
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
    if device is None:
        for stream in _streams.values():
            stream.synchronize()
    if errors:
        raise errors[0]


# ---- the measure stream ----

_overlap_on = True
_streams: dict[int, torch.cuda.Stream] = {}


@contextmanager
def overlap(enabled: bool = False) -> Iterator[None]:
    """Inside, the measure stream is on (``True``) or off (``False``, the default argument):
    off, the kernel path runs everything on the current stream with no prefetched challenges
    and the plain row read, as before FS39, for the comparison tests. Nests; outside any
    ``overlap`` it is on."""
    global _overlap_on
    before, _overlap_on = _overlap_on, bool(enabled)
    try:
        yield
    finally:
        _overlap_on = before


def overlap_on() -> bool:
    return _overlap_on


def measure_stream(device: torch.device | str) -> torch.cuda.Stream | None:
    """The device's measure stream, created at first use (one per CUDA device per process), or
    ``None`` for a non-CUDA device or while :func:`overlap` is off."""
    dev = torch.device(device)
    if dev.type != "cuda" or not _overlap_on:
        return None
    index = torch.cuda.current_device() if dev.index is None else dev.index
    stream = _streams.get(index)
    if stream is None:
        stream = _streams[index] = torch.cuda.Stream(device=index)
    return stream


@contextmanager
def stream_for(device: torch.device | str) -> Iterator[None]:
    """Makes the device's measure stream current inside, when it has one; else nothing."""
    stream = measure_stream(device)
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def prefetch(h: bytes, ms: Sequence[int], k: int, widths: Sequence[int],
             challenges: ChallengeSource | None = None) -> Prefetch | None:
    """Submit the draw of the challenges of ``ms`` (matrix widths ``widths``, as
    :func:`enqueue` takes them: each product's ``P`` columns) to the source's worker thread, for
    a later call's ``prefetched=``. ``None`` while :func:`overlap` is off or when the source
    cannot prefetch."""
    source = challenges if challenges is not None else _DEFAULT_SOURCE
    if not _overlap_on or not isinstance(source, HostChallengeSource):
        return None
    return source.prefetch(h, ms, k, widths)


# ---- the call ----

_DEFAULT_SOURCE = HostChallengeSource()


def enqueue(groups: Sequence["MeasureGroup"], *, h: bytes, k: int, guard: bool = True,
            challenges: ChallengeSource | None = None, run: Any = None, share: bool = False,
            plan_out: list | None = None, prefetched: Prefetch | None = None) -> torch.Tensor:
    """Every member's row as float64 ``[Σ n_t, row_width(k)]`` on the groups' CUDA device,
    measured by the kernel on the device's measure stream (the current stream while
    :func:`overlap` is off), after the work the current stream has enqueued; nothing is read
    back and nothing waits for the device (see :func:`drain`). ``prefetched`` (:func:`prefetch`)
    supplies the challenges when it covers them. ``run`` replaces the loaded kernel
    (``run(launches, stream)``) and ``challenges`` the host source. Every call is one launch per
    level of a plan through ``pots_measure_level`` (the launches of ``run`` carry
    :class:`LevelArgs`): :func:`read_plan` for ``share=True`` (which also appends the plan to
    ``plan_out``, a list) and :func:`plain_plan` (three levels, no shared reads) otherwise; the
    rows are the same bits."""
    device, counts = validate(groups, k)
    if device.type != "cuda":
        raise ValueError(f"the measuring kernel needs CUDA tensors, got {device}")
    retire(device)
    stream = torch.cuda.current_stream(device)
    side = measure_stream(device)
    if side is not None and side != stream:
        produced = torch.cuda.Event()
        produced.record(stream)
        side.wait_event(produced)
        stream = side
    source = challenges if challenges is not None else _DEFAULT_SOURCE
    structure = table_structure(groups, k, guard, share, _validated=(device, counts))
    extra = {}
    if prefetched is not None and isinstance(source, HostChallengeSource):
        extra["prefetched"] = prefetched
    buf, offsets, ev = source.draw(h, structure.ms, k, structure.widths, stream=stream, **extra)
    if ev is not None:
        stream.wait_event(ev)
    ht = build_host_table(groups, np.uint64(buf.data_ptr()) + offsets.astype(np.uint64), k,
                          guard, share, structure, _validated=(device, counts))
    prods, plan = ht.prods, ht.plan
    n = len(prods)
    with torch.cuda.stream(stream):
        rows = torch.empty(n, row_width(k), dtype=torch.float64, device=device)
        scratch = torch.empty(max(ht.scratch_n, 8), dtype=torch.uint8, device=device)
        table = torch.empty(ht.total, dtype=torch.uint8, device=device)
    host = torch.zeros(ht.total, dtype=torch.uint8, pin_memory=True)
    works = ht.fill(host.numpy(), scratch.data_ptr())
    work = ht.launches(table.data_ptr(), rows.data_ptr(), k)
    if share and plan_out is not None:
        plan_out.append(plan)
    with torch.cuda.stream(stream):
        table.copy_(host, non_blocking=True)
        (run if run is not None else kernels(device, k).run)(work, stream.cuda_stream)
        event = torch.cuda.Event()
        event.record(stream)
    inputs = [t for grp in groups for t in (grp.a, grp.b, grp.p, grp.bias) if t is not None]
    _inflight.setdefault(device, []).append(
        Pending(event, inputs, [_version(t) for t in inputs],
                [host, table, buf, scratch, prods, works, rows]))
    return rows
