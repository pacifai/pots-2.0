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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch

from verification.verifier.matmul_check.challenges import challenge_matrices

__all__ = ["ChallengeSource", "HostChallengeSource"]


class ChallengeSource(Protocol):
    def draw(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int], *,
             stream: Any) -> tuple[torch.Tensor, np.ndarray, Any]:
        """See the module docstring. ``offsets`` is an int64 array of byte offsets, one per
        product. ``stream`` is a ``torch.cuda.Stream``, or ``None`` on a CPU device."""
        ...


class HostChallengeSource:
    """Challenges from the native library on the host, copied to the device in one piece.

    ``device`` is the device of the buffer; on a CPU device nothing is pinned or copied, and the
    buffer is the host array itself. The pinned staging buffer of each draw is held until its
    copy's event completes."""

    def __init__(self) -> None:
        self._staging: list[tuple[Any, torch.Tensor]] = []

    def _retire(self) -> None:
        self._staging = [(ev, buf) for ev, buf in self._staging if not ev.query()]

    def draw(self, h: bytes, ms: Sequence[int], k: int, widths: Sequence[int], *,
             stream: Any, device: torch.device | str | None = None
             ) -> tuple[torch.Tensor, np.ndarray, Any]:
        if len(ms) != len(widths):
            raise ValueError(f"{len(ms)} labels but {len(widths)} widths")
        if len(ms) == 0:
            raise ValueError("no products to draw challenges for")
        dev = torch.device("cpu" if stream is None else stream.device) if device is None \
            else torch.device(device)
        on_gpu = dev.type == "cuda"
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
