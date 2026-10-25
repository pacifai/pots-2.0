"""Where the fused measuring kernel's challenges come from (FS37, ``MEASURE_KERNEL_PLAN_DRAFT.md``
§2.6).

``ChallengeSource.draw(h, ms, k, widths, *, stream)`` returns ``(buffer, offsets, event)``: one
float32 buffer on the stream's device holding, for product ``i``, the challenge matrix
``challenge_matrix(h, ms[i], k, widths[i])`` (``[width, k]``, row-major, column ``j - 1``
vector ``j``) contiguous at byte offset ``offsets[i]`` from ``buffer.data_ptr()``; the products
follow one another in ``ms`` order. ``event`` is recorded on ``stream`` after the work that
fills the buffer, or ``None`` when there is nothing to wait for (a CPU device). Work enqueued on
``stream`` afterwards sees the buffer filled; any other stream waits on ``event`` first.

:class:`HostChallengeSource` is the source in use today: the native library draws the entries on
the host (:func:`challenges.challenge_matrices`, once per distinct width) into one pinned
buffer, which one non-blocking copy sends to the device. The hashing session's GPU XOF (FS41)
will be a second source with the same bytes.

FS39 (``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.5): :meth:`HostChallengeSource.prefetch` hands the
draw of a later call to one worker thread (the native call releases the GIL), which fills a
pinned buffer of a pool of two; ``draw(..., prefetched=)`` then only copies that buffer to the
device. A slot is refilled only after every copy out of it has completed (its events are taken
at submission and waited for by the worker, as the capture's staging does instead of
``record_stream``). The bytes of each product are the same whatever the buffer's layout: the
kernel reads each product at its offset.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from verification.verifier.matmul_check.challenges import challenge_matrices

__all__ = ["ChallengeSource", "HostChallengeSource", "Prefetch"]


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
    slot: int
    future: Future = field(repr=False)

    def covers(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int]) -> bool:
        return (bytes(h) == self.h and k == self.k
                and all((int(m), int(w)) in self.offsets for m, w in zip(ms, widths)))


class _Slot:
    """One pinned buffer of the pool and the events of the copies out of it."""

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.events: list[Any] = []


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

    def prefetch(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int]) -> Prefetch:
        """Submit the draw of every ``(m, width)`` of ``ms`` and ``widths`` to the worker thread
        and return at once. The worker first waits for the last copies out of the slot it fills
        (taken here, on the calling thread), then writes each width's products in increasing
        ``m``; slots alternate call by call."""
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
        total = at // 4
        slot_ix = self._next_slot
        self._next_slot = 1 - slot_ix
        slot = self._pool[slot_ix]
        waits, slot.events = slot.events, []
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="challenges")
        key = bytes(h)
        future = self._executor.submit(self._fill, slot, waits, key, k, layout, total)
        return Prefetch(key, k, offsets, total, slot_ix, future)

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
            if prefetched.covers(h, ms, k, widths):
                return self._draw_prefetched(prefetched, ms, widths, stream, dev)
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
                         stream: Any, dev: torch.device) -> tuple[torch.Tensor, np.ndarray, Any]:
        pf.future.result()  # re-raises the worker's exception
        slot = self._pool[pf.slot]
        offsets = np.asarray([pf.offsets[(int(m), int(w))] for m, w in zip(ms, widths)],
                             dtype=np.int64)
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
