"""Read-only access to one step's committed leaves (spec §4.1).

:class:`LeafReader` is the one method every transcript source implements: the prover's
in-memory store, a store on disk, and the verifier's own cache of the leaves it hashed.
:class:`TranscriptView` names the leaves under a computation's layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation

__all__ = ["LeafReader", "TranscriptView", "stack_block"]

_ALIGN = 64  # torch.stack's CPU output is 64-byte aligned; a zero-copy block must be too


def stack_block(ts: Sequence[torch.Tensor], own: bool = False) -> torch.Tensor:
    """``torch.stack(ts)``, bit for bit; zero-copy when ``ts`` are consecutive in one storage.

    The fast path needs every tensor on the CPU with one shape and dtype, contiguous,
    non-empty, and the ``k``-th at storage offset ``off + k·numel`` in the storage of the
    first, which starts 64-byte aligned (where ``torch.stack``'s output does, so a BLAS
    kernel sees the same alignment). The result is then a view of that storage, which a
    caller must not write to, and which aliases the leaves; with ``own`` the view is cloned
    (one copy, as ``torch.stack`` makes), so the result never aliases a leaf. Anything else is
    ``torch.stack``, errors included.
    """
    n = len(ts)
    t0 = ts[0] if n else None
    if (t0 is None or not isinstance(t0, torch.Tensor) or t0.device.type != "cpu"
            or t0.numel() == 0 or not t0.is_contiguous() or t0.requires_grad):
        return torch.stack(list(ts))
    shape, dtype, numel, off = t0.shape, t0.dtype, t0.numel(), t0.storage_offset()
    ptr = t0.untyped_storage().data_ptr()
    if t0.data_ptr() % _ALIGN:
        return torch.stack(list(ts))
    for k, t in enumerate(ts):
        if (not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != dtype
                or t.storage_offset() != off + k * numel or not t.is_contiguous()
                or t.untyped_storage().data_ptr() != ptr or t.requires_grad):
            return torch.stack(list(ts))
    strides, run = [], numel
    for d in shape:  # canonical contiguous strides, whatever a size-1 dimension holds
        run //= d
        strides.append(run)
    out = t0.as_strided((n, *shape), (numel, *strides), off)
    return out.clone() if own else out


@runtime_checkable
class LeafReader(Protocol):
    """Read-only access to one step's committed leaves, by 0-based leaf index.

    A batch-record leaf returns the instance's record object (a ``Record`` for the decoder, a float
    tensor for the MLP); every other leaf returns a tensor. Callers must not mutate what they
    get. The reader doesn't report its leaf count: the computation declares it (invariant 7).
    """

    def leaf(self, index: int) -> Any: ...


class TranscriptView:
    """Named access to a :class:`LeafReader` under a computation's layout."""

    def __init__(self, computation: DeclaredComputation, reader: LeafReader) -> None:
        self.c = computation
        self.reader = reader

    def record(self, i: int) -> Any:
        return self.reader.leaf(self.c.record_index(i))

    def records(self) -> list[Any]:
        return [self.record(i) for i in range(self.c.n_s)]

    def w_t(self, name: str) -> torch.Tensor:
        return self.reader.leaf(self.c.w_t_index(name))

    def product(self, m: int) -> torch.Tensor:
        return self.reader.leaf(self.c.product_index(m))

    def w_next(self, name: str) -> torch.Tensor:
        return self.reader.leaf(self.c.w_next_index(name))

    def product_block(self, ms: Sequence[int], own: bool = False) -> torch.Tensor:
        """The products ``ms`` stacked, ``[len(ms), *leaf_shape]``, equal bit for bit to
        ``torch.stack([self.product(m) for m in ms])``, with the same errors at the first
        failing member. A run of leaves consecutive in one storage (a bmm's members) comes
        back as a view of it, read-only and aliasing the committed leaves, unless ``own``
        asks for a fresh copy (see :func:`stack_block`)."""
        # Every index is looked up before any leaf is read, so only an invalid ``m`` (which
        # comes from the computation, never the prover) fails in another order than a loop.
        idx = [self.c.product_index(m) for m in ms]
        block = getattr(self.reader, "tensor_block", None)
        if block is not None:
            return block(idx, own)
        return stack_block([self.reader.leaf(i) for i in idx], own)

    def product_run(self, m0: int, stride: int, n: int, own: bool = False) -> torch.Tensor:
        """``product_block([m0 + stride·i for i in range(n)])``, from the range alone: the
        same bits, the same errors, and no per-member Python when the reader has a range read
        (``tensor_run``). The index arithmetic is done once, on the first and last member.
        A range the fast read doesn't take (``n < 1``, ``stride < 1``, a last ``m`` outside
        the computation) goes through :meth:`product_block`, which raises as a loop does."""
        if n >= 1 and stride >= 1:
            try:
                first = self.c.product_index(m0)
                self.c.product_index(m0 + stride * (n - 1))
            except IndexError:
                first = None  # the loop raises at the first member outside, not the last
            if first is not None:
                run = getattr(self.reader, "tensor_run", None)
                if run is not None:
                    return run(first, stride, n, own)
                return stack_block([self.reader.leaf(first + stride * i) for i in range(n)], own)
        return self.product_block([m0 + stride * i for i in range(n)], own)
