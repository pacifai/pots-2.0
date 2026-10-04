"""Read-only access to one step's committed leaves (spec §4.1).

:class:`LeafReader` is the one method every transcript source implements: the prover's
in-memory store, a store on disk, and the verifier's own cache of the leaves it hashed.
:class:`TranscriptView` names the leaves under a computation's layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation

__all__ = ["LeafReader", "TranscriptView"]


@runtime_checkable
class LeafReader(Protocol):
    """Read-only access to one step's committed leaves, by 0-based leaf index.

    A batch-record leaf returns the instance's record object (a ``Record`` for Llama, a float
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
