"""The declared computation ``C``: the public description of one step (spec §3).

Prover and verifier share one :class:`DeclaredComputation` instance per model. It fixes:

- the transcript layout (spec §4.1, ref block §6): ``n_s`` batch records, the ``n_w`` weights
  of ``W_t`` in :attr:`~DeclaredComputation.weight_names` order, the products ``P_1..P_M`` in
  canonical order, then ``W_{t+1}``. That gives ``n_leaves = n_s + 2·n_w + M``, the count that
  ``verify_path`` takes from here and never from the transcript (invariant 7);
- the product inventory, one :class:`ProductSpec` per product. For ``P = A·B`` with ``A`` of
  shape ``p×q`` and ``B`` of shape ``q×c``, the challenge width is ``c``, the column count of
  ``P`` (spec §2 and §5), and the check forms ``A·(B·r)`` against ``P·r`` (spec §6);
- how the verifier rebuilds each product's operands from committed leaves (checks 3 and 5).
  :meth:`DeclaredComputation.operands` reads leaves only through a :class:`LeafReader`, the
  read-only accessor that A4's ``TranscriptStore`` implements, and recomputes glue with the
  model's own modules or ``torch.autograd.grad`` (invariant 2);
- how each weight's gradient enters check 6. A linear weight's gradient is one committed
  weight-gradient product (6a). Any other weight's gradient is recomputed glue (6b), from
  :meth:`DeclaredComputation.glue_gradients`.

The prover side of ``C`` (building the model, the forward pass to the loss, and labeling the
captured matmuls to canonical slots) is part of the same object, because it is the same
agreed program. The verifier never calls those methods.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .capture import MatmulCapture

__all__ = [
    "ProductKind",
    "ProductSpec",
    "LeafReader",
    "TranscriptView",
    "LabelingError",
    "DeclaredComputation",
]


class ProductKind(enum.Enum):
    FORWARD = "forward"
    WEIGHT_GRAD = "weight_grad"
    INPUT_GRAD = "input_grad"


@dataclass(frozen=True)
class ProductSpec:
    """One product ``P_m = A·B`` of the inventory (ref block §5).

    ``weight`` names the learnable weight the product involves, or is ``None`` for a
    weight-free bilinear product. ``member`` is the ``(s, h)`` index of one member of a batched
    attention product (A7), or ``None``.
    """

    m: int  # 1-based position in canonical product order
    name: str
    kind: ProductKind
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    weight: str | None = None
    member: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError(f"{self.name}: m must be 1-based, got {self.m}")
        if len(self.a_shape) != 2 or len(self.b_shape) != 2:
            raise ValueError(f"{self.name}: operands must be matrices")
        if self.a_shape[1] != self.b_shape[0]:
            raise ValueError(f"{self.name}: A {self.a_shape} and B {self.b_shape} don't chain")

    @property
    def p_shape(self) -> tuple[int, int]:
        return (self.a_shape[0], self.b_shape[1])

    @property
    def q(self) -> int:
        """Contracted dimension."""
        return self.a_shape[1]

    @property
    def width(self) -> int:
        """Challenge length ``c_m``: the column count of ``P``, since ``r`` multiplies ``P``."""
        return self.b_shape[1]


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


class LabelingError(RuntimeError):
    """Captured matmuls don't map one-to-one onto the declared product slots."""


class DeclaredComputation(ABC):
    """Abstract ``C``. Subclasses set the inventory and implement the abstract methods."""

    # ---- the declaration ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_s(self) -> int:
        """Batch size: records per step."""

    @property
    @abstractmethod
    def weight_names(self) -> tuple[str, ...]:
        """Weight names in ``W_t`` / ``W_{t+1}`` leaf order; each is a ``model`` parameter name."""

    @property
    @abstractmethod
    def weight_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Shape of each weight as the model holds it (linear weights are ``[o, i]``)."""

    @property
    @abstractmethod
    def products(self) -> tuple[ProductSpec, ...]:
        """The inventory in canonical order; ``products[m-1].m == m``."""

    @property
    @abstractmethod
    def linear_weights(self) -> Mapping[str, int]:
        """Check 6a: weight name -> ``m`` of the committed product that is its whole gradient."""

    # ---- derived layout -------------------------------------------------------------------

    @property
    def M(self) -> int:
        return len(self.products)

    @property
    def n_w(self) -> int:
        return len(self.weight_names)

    @property
    def n_leaves(self) -> int:
        return self.n_s + 2 * self.n_w + self.M

    @property
    def glue_gradient_weights(self) -> tuple[str, ...]:
        """Check 6b: weights whose gradient is (or includes) recomputed glue, in leaf order."""
        return tuple(n for n in self.weight_names if n not in self.linear_weights)

    def product(self, m: int) -> ProductSpec:
        if not 1 <= m <= self.M:
            raise IndexError(f"product m={m} outside 1..{self.M}")
        return self.products[m - 1]

    @cached_property
    def _weight_pos(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.weight_names)}

    def record_index(self, i: int) -> int:
        if not 0 <= i < self.n_s:
            raise IndexError(f"record {i} outside 0..{self.n_s - 1}")
        return i

    def w_t_index(self, name: str) -> int:
        return self.n_s + self._weight_pos[name]

    def product_index(self, m: int) -> int:
        self.product(m)
        return self.n_s + self.n_w + m - 1

    def w_next_index(self, name: str) -> int:
        return self.n_s + self.n_w + self.M + self._weight_pos[name]

    def validate(self) -> None:
        """Raise ``ValueError`` if the declaration is internally inconsistent."""
        names = self.weight_names
        if len(set(names)) != len(names):
            raise ValueError("duplicate weight names")
        if set(self.weight_shapes) != set(names):
            raise ValueError("weight_shapes keys differ from weight_names")
        for i, p in enumerate(self.products, start=1):
            if p.m != i:
                raise ValueError(f"product {p.name} has m={p.m} at position {i}")
            if p.weight is not None and p.weight not in self._weight_pos:
                raise ValueError(f"product {p.name} names unknown weight {p.weight}")
        if len({p.name for p in self.products}) != self.M:
            raise ValueError("duplicate product names")
        for w, m in self.linear_weights.items():
            p = self.product(m)
            if w not in self._weight_pos:
                raise ValueError(f"linear weight {w} is not a declared weight")
            if p.kind is not ProductKind.WEIGHT_GRAD or p.weight != w:
                raise ValueError(f"linear weight {w} maps to {p.name}, not its weight gradient")
            if p.p_shape != tuple(self.weight_shapes[w]):
                raise ValueError(f"{p.name} shape {p.p_shape} != {w} {self.weight_shapes[w]}")

    # ---- verifier side --------------------------------------------------------------------

    @abstractmethod
    def encode_record(self, record: Any) -> bytes:
        """Canonical leaf bytes of one batch record, identical in ``h`` and ``h_D`` (P9b)."""

    @abstractmethod
    def operands(self, m: int, leaves: LeafReader) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A_m, B_m)`` rebuilt from committed leaves only (checks 3 and 5)."""

    @abstractmethod
    def glue_gradients(self, leaves: LeafReader) -> dict[str, torch.Tensor]:
        """Check 6b: the recomputed full gradient of each of :attr:`glue_gradient_weights`."""

    # ---- prover side (the verifier never calls these) --------------------------------------

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """A fresh model whose parameters include every name in :attr:`weight_names`."""

    @abstractmethod
    def loss(self, model: torch.nn.Module, records: Sequence[Any]) -> torch.Tensor:
        """Assemble the batch (glue) and run the forward pass to the scalar loss."""

    @abstractmethod
    def label(self, capture: MatmulCapture, model: torch.nn.Module) -> list[torch.Tensor]:
        """Map captured records to canonical slots; return ``P_1..P_M`` in order.

        Called after ``loss.backward()`` while the capture still holds its operands. Must map by
        operand identity, not call order, and raise :class:`LabelingError` unless every slot is
        filled exactly once with the declared shape.
        """
