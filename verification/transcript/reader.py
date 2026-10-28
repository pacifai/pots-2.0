"""Read-only access to one step's committed leaves (spec §4.1).

:class:`LeafReader` is the one method every transcript source implements: the prover's
in-memory store, a store on disk, and the verifier's own cache of the leaves it hashed.
:class:`TranscriptView` names the leaves under a computation's layout. A product leaf is a lone
product or, for a batched attention product, all its members as ``[n, *p_shape]`` (O18), so a
member is a view of one leaf and a run of members is a slice of it: no copy and no stacking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from verification.transcript.errors import LeafShapeError

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation

__all__ = ["LeafReader", "TranscriptView"]


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

    def product_leaf(self, g: int) -> torch.Tensor:
        """Product leaf ``g`` as committed: a lone product's ``P``, or a group's
        ``[n, *p_shape]``. Raises :class:`LeafShapeError` unless it is a tensor of the
        declared shape."""
        pl = self.c.product_leaf(g)
        index = self.c.product_leaf_index(g)
        obj = self.reader.leaf(index)
        if not isinstance(obj, torch.Tensor):
            raise LeafShapeError(f"leaf {index} {pl.name}: expected a tensor, got "
                                 f"{type(obj).__name__}")
        if tuple(obj.shape) != pl.shape:
            raise LeafShapeError(f"leaf {index} {pl.name}: shape {tuple(obj.shape)}, "
                                 f"declared {pl.shape}")
        return obj

    def product(self, m: int) -> torch.Tensor:
        """Product ``m``: the leaf of a lone product, or ``leaf[i]``, a view, for a member."""
        g, i = self.c.leaf_of(m)
        leaf = self.product_leaf(g)
        return leaf if i is None else leaf[i]

    def w_next(self, name: str) -> torch.Tensor:
        return self.reader.leaf(self.c.w_next_index(name))

    def _slice(self, g: int, i0: int, step: int, n: int, own: bool) -> torch.Tensor:
        out = self.product_leaf(g)[i0:i0 + step * (n - 1) + 1:step]
        return out.clone() if own else out

    def product_block(self, ms: Sequence[int], own: bool = False) -> torch.Tensor:
        """The products ``ms`` stacked, ``[len(ms), *p_shape]``, equal bit for bit to
        ``torch.stack([self.product(m) for m in ms])``. Members of one group at member indices
        in an increasing arithmetic progression come back as a slice of the group's leaf, a
        view, read-only and aliasing the committed leaf, unless ``own`` asks for a copy; so does
        a lone product (``unsqueeze``). Anything else is ``torch.stack``."""
        # Every index is looked up before any leaf is read, so only an invalid ``m`` (which
        # comes from the computation, never the prover) fails in another order than a loop.
        pos = [self.c.leaf_of(m) for m in ms]
        if pos:
            g, i0 = pos[0]
            if all(p[0] == g for p in pos):
                if i0 is None:
                    if len(pos) == 1:
                        leaf = self.product_leaf(g).unsqueeze(0)
                        return leaf.clone() if own else leaf
                else:
                    step = pos[1][1] - i0 if len(pos) > 1 else 1
                    if step >= 1 and all(p[1] == i0 + step * k for k, p in enumerate(pos)):
                        return self._slice(g, i0, step, len(pos), own)
        return torch.stack([self.product(m) for m in ms])

    def product_run(self, m0: int, stride: int, n: int, own: bool = False) -> torch.Tensor:
        """``product_block([m0 + stride·i for i in range(n)])``, from the range alone: the
        same bits, the same errors, and no per-member Python when the run lies in one group
        whose member spacing divides ``stride``. The lookup is done on the first and last
        member. Any other range goes through :meth:`product_block`."""
        c = self.c
        if n >= 1 and stride >= 1 and 1 <= m0 and m0 + stride * (n - 1) <= c.M:
            g0, i0 = c.leaf_of(m0)
            g1, i1 = c.leaf_of(m0 + stride * (n - 1))
            if i0 is not None and g0 == g1 and stride % c.product_leaf(g0).stride == 0:
                return self._slice(g0, i0, stride // c.product_leaf(g0).stride, n, own)
        return self.product_block([m0 + stride * i for i in range(n)], own)
