"""Where the fused measuring kernel's challenges come from (FS37, ``MEASURE_KERNEL_PLAN_DRAFT.md``
§2.6).

``ChallengeSource.draw(h, ms, k, widths, *, stream)`` returns ``(buffer, offsets, event)``: one
float32 buffer on the stream's device holding, for product ``i``, the challenge matrix
``challenge_matrix(h, ms[i], k, widths[i])`` (``[width, k]``, row-major, column ``j - 1``
vector ``j``) contiguous at byte offset ``offsets[i]`` from ``buffer.data_ptr()``; the products
follow one another in ``ms`` order. ``event`` is recorded on ``stream`` after the work that
fills the buffer, or ``None`` when there is nothing to wait for (a CPU device). Work enqueued on
``stream`` afterwards sees the buffer filled; any other stream waits on ``event`` first.

:class:`HostChallengeSource` is the CPU path and the reference: the native library draws the
entries on the host (:func:`challenges.challenge_matrices`, once per distinct width) into one
pinned buffer, which one non-blocking copy sends to the device. :class:`GpuChallengeSource` (FS41,
§2.9) draws the same bytes: ``pots_challenge_xof`` (``challenge_xof.cu``)
draws them on the device that measures them, from an item table cached per request. It is the
default on CUDA (``measure_kernel.default_source``) and delegates a CPU device to the host
source.

FS39 (``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.5): :meth:`HostChallengeSource.prefetch` hands the
draw of a later call to one worker thread (the native call releases the GIL), which fills a
pinned buffer of a pool of two; ``draw(..., prefetched=)`` then only copies that buffer to the
device. A slot is refilled only after every copy out of it has completed (its events are taken
at submission and waited for by the worker, as the capture's staging does instead of
``record_stream``). A prefetch whose slot a later one refilled covers nothing, so its draw runs
inline. The bytes of each product are the same whatever the buffer's layout: the kernel reads
each product at its offset.
"""

from __future__ import annotations

import ctypes
import functools
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.verifier.matmul_check.challenges import challenge_matrices

__all__ = ["ChallengeSource", "GpuChallengeSource", "HostChallengeSource", "Prefetch",
           "XParams", "XofPlan", "xof_kernels", "xof_params", "xof_plan"]


class ChallengeSource(Protocol):
    def draw(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int], *,
             stream: Any) -> tuple[torch.Tensor, np.ndarray, Any]:
        """See the module docstring. ``offsets`` is an int64 array of byte offsets, one per
        product. ``stream`` is a ``torch.cuda.Stream``, or ``None`` on a CPU device."""
        ...


@dataclass
class Prefetch:
    """A draw submitted to :meth:`HostChallengeSource.prefetch`: the challenges of every distinct
    ``(m, width)`` of the request, in a pool slot, once ``future`` is done. ``offsets`` maps
    ``(m, width)`` to the byte offset of that product's ``[width, k]`` matrix in the slot's
    buffer and ``total`` is the buffer's used length in float32 entries."""

    h: bytes
    k: int
    offsets: dict[tuple[int, int], int]
    total: int
    slot: int  # the pool index, for tests and logs
    future: Future = field(repr=False)
    holder: Any = field(default=None, repr=False)  # the slot, so any source can draw from it
    fill: int = 0  # the slot's fill count at submission: a later prefetch refills it
    layout: "_Layout | None" = field(default=None, repr=False)  # the request's cached layout

    def covers(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int],
               layout: "_Layout | None" = None) -> bool:
        """Whether the slot still holds this draw and it has every product of the request. With
        the request's cached ``layout`` (:meth:`HostChallengeSource.find_layout`) the one this
        prefetch was drawn for, the slot, ``h`` and ``k`` are all that is compared (FS51);
        otherwise each member is looked up."""
        if not (getattr(self.holder, "fills", 0) == self.fill and bytes(h) == self.h
                and k == self.k):
            return False
        if layout is not None and layout is self.layout:
            return True
        return all((int(m), int(w)) in self.offsets for m, w in zip(ms, widths))


@dataclass
class _Layout:
    """The layout of one request ``(k, ms, widths)`` (FS51): the distinct ``(m, width)`` products
    by width then ``m`` (what the worker draws), each one's byte offset, the used length in
    float32 entries, and the offset of every member of the request in order. A function of the
    request alone, so a repeated request reuses it."""

    k: int
    ms: tuple[int, ...]
    widths: tuple[int, ...]
    layout: list[tuple[int, list[int]]]
    offsets: dict[tuple[int, int], int]
    total: int
    member_offsets: np.ndarray  # int64, per member of ``ms``
    # The last draw request this layout covered member by member (FS55a): its ``ms`` and
    # ``widths`` and that request's per-member offsets. The draw's tables list the products in
    # served order, not the prefetch's canonical order, so ``find_layout`` never matches them.
    draw_ms: tuple[int, ...] | None = None
    draw_widths: tuple[int, ...] | None = None
    draw_offsets: np.ndarray | None = None

    def drawn_offsets(self, ms: Sequence[int], widths: Sequence[int]) -> np.ndarray | None:
        """The stored offsets when ``(ms, widths)`` is the remembered draw request (identity
        first, then equality), else ``None``."""
        if self.draw_offsets is None:
            return None
        if ms is not self.draw_ms:
            tm = ms if isinstance(ms, tuple) else tuple(ms)
            if tm != self.draw_ms:
                return None
        if widths is not self.draw_widths:
            tw = widths if isinstance(widths, tuple) else tuple(widths)
            if tw != self.draw_widths:
                return None
        return self.draw_offsets

    def remember_draw(self, ms: Sequence[int], widths: Sequence[int],
                      offsets: np.ndarray) -> None:
        self.draw_ms = ms if isinstance(ms, tuple) else tuple(int(m) for m in ms)
        self.draw_widths = widths if isinstance(widths, tuple) else tuple(int(w) for w in widths)
        self.draw_offsets = offsets.copy()


LAYOUT_CACHE_MAX = 64
"""Cached layouts at most (least recently used first out): one per schedule step and device."""


class _Slot:
    """One pinned buffer of the pool and the events of the copies out of it."""

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.events: list[Any] = []
        self.fills = 0  # prefetches submitted into it


class HostChallengeSource:
    """Challenges from the native library on the host, copied to the device in one piece.

    ``device`` is the device of the buffer; on a CPU device nothing is pinned or copied, and the
    buffer is the host array itself. The pinned staging buffer of each draw is held until its
    copy's event completes."""

    def __init__(self) -> None:
        self._staging: list[tuple[Any, torch.Tensor]] = []
        self._pool = [_Slot(), _Slot()]
        self._next_slot = 0
        self._executor: ThreadPoolExecutor | None = None
        self._layouts: OrderedDict[tuple, list[_Layout]] = OrderedDict()
        self._layouts_n = 0
        self.fast_draws = 0  # draws that reused a layout's stored draw offsets (FS55a)

    def prefetch(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int]) -> Prefetch:
        """Submit the draw of every ``(m, width)`` of ``ms`` and ``widths`` to the worker thread
        and return at once. The worker first waits for the last copies out of the slot it fills
        (taken here, on the calling thread), then writes each width's products in increasing
        ``m``; slots alternate call by call."""
        lay = self.find_layout(k, ms, widths)
        if lay is None:
            lay = self._new_layout(k, ms, widths)
        layout, offsets, total = lay.layout, lay.offsets, lay.total
        slot_ix = self._next_slot
        self._next_slot = 1 - slot_ix
        slot = self._pool[slot_ix]
        waits, slot.events = slot.events, []
        slot.fills += 1
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="challenges")
        key = bytes(h)
        future = self._executor.submit(self._fill, slot, waits, key, k, layout, total)
        return Prefetch(key, k, offsets, total, slot_ix, future, slot, slot.fills, lay)

    def find_layout(self, k: int, ms: Sequence[int], widths: Sequence[int]) -> _Layout | None:
        """The cached layout of this request when ``(k, ms, widths)`` equal a cached one (the
        sequences compared as tuples, in C), else ``None``."""
        if len(ms) == 0 or len(ms) != len(widths):
            return None
        tm, tw = tuple(ms), tuple(widths)
        bucket = (k, len(tm), int(tm[0]), int(tm[-1]), int(tw[0]), int(tw[-1]))
        for lay in self._layouts.get(bucket, ()):
            if lay.ms == tm and lay.widths == tw:
                self._layouts.move_to_end(bucket)
                return lay
        return None

    def _new_layout(self, k: int, ms: Sequence[int], widths: Sequence[int]) -> _Layout:
        """Build (and cache) the layout of a request: the loops this cache saves a repeat."""
        if len(ms) != len(widths):
            raise ValueError(f"{len(ms)} labels but {len(widths)} widths")
        if len(ms) == 0:
            raise ValueError("no products to draw challenges for")
        by_width: dict[int, list[int]] = {}
        for m, w in zip(ms, widths):
            by_width.setdefault(int(w), []).append(int(m))
        if min(by_width) < 1:
            raise ValueError(f"a width is below 1 (widths {list(widths)})")
        layout: list[tuple[int, list[int]]] = []
        offsets: dict[tuple[int, int], int] = {}
        at = 0
        for width in sorted(by_width):
            uniq = sorted(set(by_width[width]))
            layout.append((width, uniq))
            for m in uniq:
                offsets[(m, width)] = at
                at += width * k * 4
        tm = tuple(int(m) for m in ms)
        tw = tuple(int(w) for w in widths)
        member = np.asarray([offsets[(m, w)] for m, w in zip(tm, tw)], dtype=np.int64)
        lay = _Layout(k, tm, tw, layout, offsets, at // 4, member)
        bucket = (k, len(tm), tm[0], tm[-1], tw[0], tw[-1])
        self._layouts.setdefault(bucket, []).append(lay)
        self._layouts.move_to_end(bucket)
        self._layouts_n += 1
        while self._layouts_n > LAYOUT_CACHE_MAX and len(self._layouts) > 1:
            _, old = self._layouts.popitem(last=False)
            self._layouts_n -= len(old)
        if self._layouts_n > LAYOUT_CACHE_MAX:  # one bucket left over the cap: oldest go
            drop = self._layouts_n - LAYOUT_CACHE_MAX
            del self._layouts[bucket][:drop]
            self._layouts_n -= drop
        return lay

    @staticmethod
    def _fill(slot: _Slot, waits: list[Any], h: bytes, k: int,
              layout: list[tuple[int, list[int]]], total: int) -> None:
        for ev in waits:
            ev.synchronize()
        if slot.buf is None or slot.buf.numel() < total:
            slot.buf = torch.empty(total, dtype=torch.float32,
                                   pin_memory=torch.cuda.is_available())
        flat = slot.buf.numpy()
        at = 0
        for width, uniq in layout:
            mats = challenge_matrices(h, uniq, k, width).numpy().reshape(-1)
            flat[at:at + mats.size] = mats
            at += mats.size

    def _retire(self) -> None:
        self._staging = [(ev, buf) for ev, buf in self._staging if not ev.query()]

    def draw(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int], *,
             stream: Any, device: torch.device | str | None = None,
             prefetched: Prefetch | None = None) -> tuple[torch.Tensor, np.ndarray, Any]:
        if len(ms) != len(widths):
            raise ValueError(f"{len(ms)} labels but {len(widths)} widths")
        if len(ms) == 0:
            raise ValueError("no products to draw challenges for")
        dev = torch.device("cpu" if stream is None else stream.device) if device is None \
            else torch.device(device)
        on_gpu = dev.type == "cuda"
        if prefetched is not None:
            pl = prefetched.layout
            known = pl.drawn_offsets(ms, widths) if pl is not None else None
            if known is not None and prefetched.covers(h, ms, k, widths, pl):
                self.fast_draws += 1  # FS55a: this request was checked member by member before
                return self._draw_prefetched(prefetched, ms, widths, stream, dev, pl, known)
            lay = self.find_layout(k, ms, widths)
            if prefetched.covers(h, ms, k, widths, lay):
                return self._draw_prefetched(prefetched, ms, widths, stream, dev, lay)
            prefetched.future.exception()  # let the worker finish; its slot is reusable
        w = np.asarray(widths, dtype=np.int64)
        if w.min() < 1:
            raise ValueError(f"a width is below 1 (widths {widths})")
        sizes = w * k * 4
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
        total = int(sizes.sum()) // 4
        host = torch.empty(total, dtype=torch.float32, pin_memory=on_gpu)
        flat = host.numpy()
        for width in np.unique(w):
            idx = np.flatnonzero(w == width)
            mats = challenge_matrices(h, [int(ms[i]) for i in idx], k, int(width)).numpy()
            mats = mats.reshape(len(idx), -1)
            # consecutive products of one width are consecutive in the buffer: one slice a run
            cuts = np.flatnonzero(np.diff(idx) != 1) + 1
            for lo, hi in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [len(idx)]))):
                at = int(offsets[idx[lo]]) // 4
                flat[at:at + int(hi - lo) * mats.shape[1]] = mats[lo:hi].reshape(-1)
        if not on_gpu:
            return host, offsets, None
        self._retire()
        with torch.cuda.stream(stream):
            buf = torch.empty(total, dtype=torch.float32, device=dev)
            buf.copy_(host, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._staging.append((event, host))
        return buf, offsets, event

    def _draw_prefetched(self, pf: Prefetch, ms: Sequence[int], widths: Sequence[int],
                         stream: Any, dev: torch.device,
                         lay: _Layout | None = None,
                         known: np.ndarray | None = None) -> tuple[torch.Tensor, np.ndarray, Any]:
        pf.future.result()  # re-raises the worker's exception
        slot = pf.holder
        if known is not None:  # the stored offsets of this exact draw request (FS55a)
            offsets = known.copy()
        elif lay is not None and lay is pf.layout:  # the cached per-member offsets (FS51)
            offsets = lay.member_offsets.copy()
        else:
            offsets = np.asarray([pf.offsets[(int(m), int(w))] for m, w in zip(ms, widths)],
                                 dtype=np.int64)
            if pf.layout is not None:
                pf.layout.remember_draw(ms, widths, offsets)
        used = slot.buf[:pf.total]
        if dev.type != "cuda":
            return used.clone(), offsets, None  # the slot is refilled while this is in use
        self._retire()
        with torch.cuda.stream(stream):
            buf = torch.empty(pf.total, dtype=torch.float32, device=dev)
            buf.copy_(used, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        slot.events.append(event)
        return buf, offsets, event


# ---- the GPU source (FS41, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.9) ----

XITEM_DTYPE = np.dtype([("m", "<u4"), ("width", "<u4"), ("unit0", "<u8"), ("out_off", "<u8")])
"""``pots_xitem_t`` of ``challenge_xof.h``: one product of the request."""


class XParams(ctypes.Structure):
    """``pots_xparams_t`` of ``challenge_xof.h``: the kernel's one argument."""

    _fields_ = [("key", ctypes.c_uint32 * 8), ("k", ctypes.c_uint32), ("n", ctypes.c_uint32),
                ("units", ctypes.c_uint64), ("items", ctypes.c_uint64),
                ("out", ctypes.c_uint64)]


assert XITEM_DTYPE.itemsize == 24 and ctypes.sizeof(XParams) == 64

XOF_THREADS = 256
XOF_MAX_BLOCKS = 1 << 16
"""The kernel's block size and grid cap; it strides over the units past the cap."""
XOF_NAME = "challenge_xof.cu"
XOF_HEADER = "challenge_xof.h"
XOF_CORE = "gpu_blake3_core.h"
XOF_KERNEL = "pots_challenge_xof"
HERE = Path(__file__).resolve().parent
COMMITMENT = HERE.parents[1] / "commitment"


@dataclass(frozen=True)
class XofPlan:
    """A request's item table and layout: ``items`` (``XITEM_DTYPE``, one per product, in
    request order), the byte ``offsets`` (read-only int64, the host source's inline draw's), the
    float32 entries in ``total``, the XOF blocks in ``units`` and the vectors per product ``k``."""

    items: np.ndarray
    offsets: np.ndarray
    total: int
    units: int
    k: int


def xof_plan(ms: Sequence[int], k: int, widths: Sequence[int]) -> XofPlan:
    """The plan of ``challenge_matrix(h, ms[i], k, widths[i])`` for every ``i``, each one its own
    ``[width, k]`` matrix after the one before. Raises ``ValueError`` outside P8a's ranges."""
    if not 1 <= k <= 255:
        raise ValueError(f"k must be in [1, 255], got {k}")
    if len(ms) != len(widths):
        raise ValueError(f"{len(ms)} labels but {len(widths)} widths")
    if len(ms) == 0:
        raise ValueError("no products to draw challenges for")
    try:
        m = np.asarray(ms, dtype=np.int64)
        w = np.asarray(widths, dtype=np.int64)
    except OverflowError:
        raise ValueError("a label or width does not fit 64 bits") from None
    if m.min() < 1 or m.max() >= 1 << 32:
        raise ValueError(f"every m must be in [1, 2^32), got {list(ms)}")
    if w.min() < 1:
        raise ValueError(f"a width is below 1 (widths {list(widths)})")
    if w.max() >= 1 << 32:
        raise ValueError(f"a width is 2^32 or more (widths {list(widths)})")
    units = k * ((w + 15) // 16)
    ends = np.cumsum(units)
    entries = np.cumsum(w * k)
    items = np.empty(len(m), dtype=XITEM_DTYPE)
    items["m"], items["width"] = m, w
    items["unit0"] = ends - units
    items["out_off"] = entries - w * k
    offsets = (items["out_off"].astype(np.int64)) * 4
    offsets.flags.writeable = False
    return XofPlan(items, offsets, int(entries[-1]), int(ends[-1]), k)


def xof_params(h: bytes, plan: XofPlan, items_addr: int, out_addr: int) -> XParams:
    """The kernel's argument for key ``h`` (32 bytes, as eight little-endian words) and ``plan``,
    with the item table at ``items_addr`` and the float32 output at ``out_addr``."""
    if len(h) != 32:
        raise ValueError("h must be 32 bytes")
    p = XParams()
    p.key[:] = np.frombuffer(bytes(h), "<u4").tolist()
    p.k, p.n, p.units = plan.k, len(plan.items), plan.units
    p.items, p.out = items_addr, out_addr
    return p


def xof_blocks(units: int) -> int:
    return max(1, min(-(-units // XOF_THREADS), XOF_MAX_BLOCKS))


# ---- the build ----

def xof_source() -> gb.Source:
    """The kernel, its header, the BLAKE3 core it includes and the cache prefix."""
    return gb.Source(XOF_NAME, (HERE / XOF_NAME).read_text(),
                     {XOF_HEADER: (HERE / XOF_HEADER).read_text(),
                      XOF_CORE: (COMMITMENT / XOF_CORE).read_text()}, "challenge_xof")


def xof_build(nvrtc: gb.Nvrtc, cc: tuple[int, int], *, cache: Path | None = None,
              use_cache: bool = True) -> gb.Built:
    """The compiled image for compute capability ``cc``, cached in ``cache`` when given."""
    return gb.build_image(nvrtc, cc, cache=cache, use_cache=use_cache, source=xof_source(),
                          extra_options=("--fmad=false",))


class XofKernels:
    """A device's loaded module and its XOF kernel."""

    def __init__(self, driver: gb.Driver, dev: int, built: gb.Built) -> None:
        self.driver = driver
        self.dev = dev
        self.built = built
        driver.push(dev)
        try:
            self.module = driver.load_module(built.image)
            self.fn = driver.function(self.module, XOF_KERNEL)
        finally:
            driver.pop()

    def run(self, params: XParams, blocks: int, stream: int) -> None:
        """Launch ``pots_challenge_xof`` on the raw ``stream``."""
        self.driver.push(self.dev)
        try:
            self.driver.launch(self.fn, blocks, XOF_THREADS, params, stream)  # pyright: ignore
        finally:
            self.driver.pop()


@functools.lru_cache(maxsize=None)
def _xof_built(cc: tuple[int, int]) -> gb.Built:
    return xof_build(gb._nvrtc(), cc, cache=native.cache_dir())


@functools.lru_cache(maxsize=None)
def _xof_kernels(ordinal: int) -> XofKernels:
    driver = gb._driver()
    dev = driver.device(ordinal)
    cc = driver.compute_capability(dev)
    built = _xof_built(cc)
    try:
        return XofKernels(driver, dev, built)
    except RuntimeError:
        if not built.cached:
            raise
        built.forget()  # a cached image the driver rejects: compile afresh once
        return XofKernels(driver, dev, xof_build(gb._nvrtc(), cc, cache=native.cache_dir(),
                                                 use_cache=False))


def xof_kernels(device: torch.device | str) -> XofKernels:
    """``device``'s XOF kernel, compiled (or loaded from the cache) on the first call."""
    index = torch.device(device).index
    return _xof_kernels(torch.cuda.current_device() if index is None else index)


# ---- the source ----

@dataclass
class _XofEntry:
    """One cached request: its plan, the item table on the device and the event after its
    upload (every draw's stream waits for it), keyed by ``(device, k, ms, widths)``."""

    device: torch.device
    k: int
    ms: tuple[int, ...]
    widths: tuple[int, ...]
    plan: XofPlan
    items: Any
    ready: Any
    stream: Any


def _stream_device(dev: torch.device, stream: Any) -> torch.device:
    """``dev`` with the stream's index: the buffer, the table and the launch all sit on the
    stream's device. An index that differs from the stream's raises ``ValueError``."""
    on = torch.device(stream.device)
    if dev.index is not None and dev.index != on.index:
        raise ValueError(f"device {dev} but the stream is on {on}")
    return torch.device("cuda", on.index)


class GpuChallengeSource:
    """Challenges drawn by the XOF kernel on the device that measures them (FS41): the bytes of
    :class:`HostChallengeSource`'s inline draw, at the same offsets. ``prefetched`` is ignored.
    ``run(device, params, blocks, stream_handle)`` replaces the loaded kernel, for tests; a CPU
    device draws on the host."""

    def __init__(self, run: Callable[[torch.device, XParams, int, int], None] | None = None
                 ) -> None:
        self._run = run
        self._host = HostChallengeSource()
        self._entries: OrderedDict[tuple, list[_XofEntry]] = OrderedDict()
        self._entries_n = 0

    def plan(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int]
             ) -> tuple[XofPlan, XParams]:
        """The plan and the params with no device (item and output addresses 0)."""
        plan = xof_plan(ms, k, widths)
        return plan, xof_params(h, plan, 0, 0)

    def _upload(self, plan: XofPlan, device: torch.device, stream: Any) -> tuple[Any, Any]:
        """The item table on ``device``, copied on ``stream``, and the event after the copy. A
        pageable source buffer is staged before the call returns."""
        with torch.cuda.stream(stream):
            items = torch.from_numpy(plan.items.view(np.uint8)).to(device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        return items, ready

    def _lookup(self, device: torch.device, k: int, ms: Sequence[int], widths: Sequence[int],
                stream: Any) -> _XofEntry:
        """The cached entry of this request (tuples compared by identity, then equality), built
        and stored when missing."""
        if len(ms) != len(widths):
            raise ValueError(f"{len(ms)} labels but {len(widths)} widths")
        if len(ms) == 0:
            raise ValueError("no products to draw challenges for")
        tm = ms if isinstance(ms, tuple) else tuple(ms)
        tw = widths if isinstance(widths, tuple) else tuple(widths)
        bucket = (device, k, len(tm), int(tm[0]), int(tm[-1]), int(tw[0]), int(tw[-1]))
        for e in self._entries.get(bucket, ()):
            if (e.ms is tm or e.ms == tm) and (e.widths is tw or e.widths == tw):
                self._entries.move_to_end(bucket)
                return e
        plan = xof_plan(tm, k, tw)  # validates the values; the stored tuples are the request's
        tm = ms if isinstance(ms, tuple) else tuple(int(m) for m in ms)
        tw = widths if isinstance(widths, tuple) else tuple(int(w) for w in widths)
        items, ready = self._upload(plan, device, stream)
        entry = _XofEntry(device, k, tm, tw, plan, items, ready, stream)
        self._entries.setdefault(bucket, []).append(entry)
        self._entries.move_to_end(bucket)
        self._entries_n += 1
        while self._entries_n > LAYOUT_CACHE_MAX and len(self._entries) > 1:
            _, old = self._entries.popitem(last=False)
            self._entries_n -= len(old)
        if self._entries_n > LAYOUT_CACHE_MAX:  # one bucket left over the cap: oldest go
            drop = self._entries_n - LAYOUT_CACHE_MAX
            del self._entries[bucket][:drop]
            self._entries_n -= drop
        return entry

    def draw(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int], *,
             stream: Any, device: torch.device | str | None = None,
             prefetched: Prefetch | None = None) -> tuple[torch.Tensor, np.ndarray, Any]:
        dev = torch.device("cpu" if stream is None else stream.device) if device is None \
            else torch.device(device)
        if dev.type != "cuda" or stream is None:
            return self._host.draw(h, ms, k, widths, stream=stream, device=dev)
        if len(h) != 32:
            raise ValueError("h must be 32 bytes")
        dev = _stream_device(dev, stream)
        entry = self._lookup(dev, k, ms, widths, stream)
        stream.wait_event(entry.ready)
        if stream != entry.stream:
            entry.items.record_stream(stream)  # the table outlives its upload stream's order
        plan = entry.plan
        with torch.cuda.stream(stream):
            buf = torch.empty(plan.total, dtype=torch.float32, device=dev)
        params = xof_params(h, plan, entry.items.data_ptr(), buf.data_ptr())
        blocks = xof_blocks(plan.units)
        if self._run is not None:
            self._run(dev, params, blocks, stream.cuda_stream)
        else:
            xof_kernels(dev).run(params, blocks, stream.cuda_stream)
        with torch.cuda.stream(stream):
            event = torch.cuda.Event()
            event.record(stream)
        return buf, plan.offsets, event
