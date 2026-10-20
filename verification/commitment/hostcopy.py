"""The host copy of a device leaf: the one seam between a leaf's bytes and the hash (FS32b).

Under the sharded placement (F22) product and weight leaves sit on the verifier GPUs, and the
hash reads host memory. Every reader of a leaf's bytes goes through this module, so FS34 can
replace the copy with the GPU kernel (FS28) in one place. A CPU tensor passes through as the
same object, with no copy: the unsharded path is unchanged.

:func:`host_tensor` is the synchronous copy, for the Python path and for one-off reads.
:class:`Stage` and :class:`StagingPool` are the native path's: a batch's device leaves are
copied into one pinned host buffer on a side stream of the leaf's device, behind one event, so
the copies of the next batch overlap the hashing of the current one. Buffers are reused across
batches and steps, and a buffer goes back to the pool only once the batch that read it is
hashed. Without CUDA (MPS, or a test that treats CPU tensors as device leaves) the copy is a
plain, synchronous ``copy_``.

Invariant 6 holds for a copy: the leaf's ``_version`` is read when the copy is queued and
compared once it has completed; a change raises :class:`StoreMutationError`.
"""

from __future__ import annotations

import torch

from verification.transcript.errors import StoreMutationError

__all__ = ["on_device", "host_tensor", "Stage", "StagingPool", "POOL", "ALIGN"]

ALIGN = 64  # a staged leaf starts 64-byte aligned, as a fresh tensor's storage does

_MUTATED = "leaf was mutated in place while it was copied to the host"


def on_device(t: torch.Tensor) -> bool:
    """Whether ``t``'s bytes have to be copied to the host before they can be hashed."""
    return t.device.type != "cpu"


def host_tensor(t: torch.Tensor) -> torch.Tensor:
    """The host tensor holding ``t``'s bytes: ``t`` itself on the CPU, else a synchronous copy.

    A change to ``t`` while it is copied raises :class:`StoreMutationError`.
    """
    if not on_device(t):
        return t
    version = t._version
    host = t.detach().to("cpu")
    if t._version != version:
        raise StoreMutationError(_MUTATED)
    return host


_sides: dict[torch.device, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    s = _sides.get(device)
    if s is None:
        s = _sides[device] = torch.cuda.Stream(device)
    return s


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class Stage:
    """One host buffer that a batch's device leaves are copied into, back to back.

    :meth:`put` queues a leaf's copy and returns the host tensor over the staged bytes;
    :meth:`fence` closes the batch behind one event per device used; :meth:`wait` blocks
    until every queued copy has completed.
    """

    __slots__ = ("buf", "used", "pooled", "sides", "events", "fenced", "released", "wait_each")

    def __init__(self, buf: torch.Tensor, pooled: bool) -> None:
        self.buf = buf
        self.used = 0
        self.pooled = pooled
        self.sides: dict[torch.device, torch.cuda.Stream] = {}
        self.wait_each = True
        self.events: list[torch.cuda.Event] = []
        self.fenced = False
        self.released = False

    def fits(self, nbytes: int) -> bool:
        return _align(self.used) + nbytes <= self.buf.numel()

    def put(self, t: torch.Tensor) -> torch.Tensor:
        """Queue ``t``'s copy and return the host tensor (``t``'s shape and dtype) over its
        staged bytes. The caller reads ``t._version`` first and checks it after :meth:`wait`.

        The copy runs after the work queued on the current stream so far. With ``wait_each``
        False the caller says every leaf of this stage already existed when the first was put,
        so the side stream waits once per device, at that first put, not once per leaf."""
        n = t.nbytes
        off = _align(self.used)
        if off + n > self.buf.numel():
            raise ValueError("leaf does not fit the staging buffer")
        self.used = off + n
        host = self.buf[off:off + n].view(t.dtype).view(t.shape)
        if n:
            src = t.detach()
            if t.device.type == "cuda":
                side = _side_stream(t.device)
                if self.wait_each or t.device not in self.sides:
                    side.wait_stream(torch.cuda.current_stream(t.device))  # the leaf's producers
                with torch.cuda.stream(side):
                    host.copy_(src, non_blocking=True)
                # No ``record_stream``: the caller holds ``t`` until the copy is done (the
                # native path keeps it in ``_Batch.tensors`` until its event has completed,
                # and its ``finally`` drains every stage before returning), so the caching
                # allocator cannot reuse ``t``'s memory under the copy.
                self.sides[t.device] = side
            else:
                host.copy_(src)
        return host

    def fence(self) -> None:
        """Record the event that ends the copies queued so far."""
        if not self.fenced:
            self.fenced = True
            for side in self.sides.values():
                ev = torch.cuda.Event()
                ev.record(side)
                self.events.append(ev)

    def wait(self) -> None:
        """Block until the copies up to the fence have completed."""
        for ev in self.events:
            ev.synchronize()
        self.events.clear()

    def drain(self) -> None:
        self.fence()
        self.wait()


class StagingPool:
    """The reusable staging buffers. Used from the calling thread only.

    ``allocated`` counts the pooled buffers ever allocated (a leaf larger than one batch gets
    a buffer of its own, which is not pooled).
    """

    def __init__(self) -> None:
        self._free: list[torch.Tensor] = []
        self.allocated = 0

    def acquire(self, capacity: int, need: int, pinned: bool) -> Stage:
        """A stage of ``capacity`` bytes, or of exactly ``need`` when a single leaf is larger."""
        if need > capacity:
            return Stage(torch.empty(need, dtype=torch.uint8, pin_memory=pinned), False)
        for k, b in enumerate(self._free):
            if b.numel() >= capacity:
                return Stage(self._free.pop(k), True)
        self.allocated += 1
        return Stage(torch.empty(capacity, dtype=torch.uint8, pin_memory=pinned), True)

    def release(self, stage: Stage) -> None:
        """Return ``stage``'s buffer once its copies are done. Safe to call twice."""
        if stage.released:
            return
        stage.drain()
        stage.released = True
        if stage.pooled:
            self._free.append(stage.buf)


POOL = StagingPool()
