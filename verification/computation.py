"""The declared computation ``C``: the public description of one step (spec §3).

Prover and verifier share one :class:`DeclaredComputation` instance per model. It fixes:

- the transcript layout (spec §4.1, ref block §6): ``n_s`` batch records, the ``n_w`` weights
  of ``W_t`` in :attr:`~DeclaredComputation.weight_names` order, the products ``P_1..P_M`` in
  canonical order, then ``W_{t+1}``. That gives ``n_leaves = n_s + 2·n_w + M``, the count that
  ``verify_path`` takes from here and never from the transcript (invariant 7);
- the product inventory, one :class:`ProductSpec` per product. For ``P = A·B`` with ``A`` of
  shape ``p×q`` and ``B`` of shape ``q×c``, the challenge width is ``c``, the column count of
  ``P`` (spec §2 and §5), and the check forms ``A·(B·r)`` against ``P·r`` (spec §6);
- the step size ``η`` and the plain-SGD update (spec check 6, S8a, S8b);
- how the verifier rebuilds each product's operands from committed leaves (checks 3 and 5).
  :meth:`DeclaredComputation.replay` opens a per-step :class:`Replay` over a
  :class:`LeafReader`, the read-only accessor that A4's ``TranscriptStore`` implements. The
  replay builds its own model from the committed ``W_t`` (``build_model`` plus
  :func:`load_weights`) and recomputes glue with that instance's modules or
  ``torch.autograd.grad`` (invariants 1 and 2);
- how each weight's gradient enters check 6. A linear weight's gradient is one committed
  weight-gradient product (6a). Any other weight's gradient is recomputed glue (6b), from
  :meth:`Replay.glue_gradients`, which reuses the backward replay that check 5 ran (S6c).

``build_model`` is shared: the prover trains it and the replay rebuilds it. The forward pass to
the loss and the labeling of captured matmuls into canonical slots are prover-side; the
verifier never calls them.
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
    "Replay",
    "DeclaredComputation",
    "load_weights",
]


class ProductKind(enum.Enum):
    FORWARD = "forward"
    WEIGHT_GRAD = "weight_grad"
    INPUT_GRAD = "input_grad"
    OPERAND_GRAD = "operand_grad"  # a gradient of a weight-free bilinear product (δA, δV, δQ̃, δK̃)


@dataclass(frozen=True)
class ProductSpec:
    """One product ``P_m = A·B`` of the inventory (ref block §5).

    ``weight`` names the learnable weight the product involves, or is ``None`` for a
    weight-free bilinear product. ``layer`` is the 1-based layer, or ``None`` outside the layer
    stack. ``member`` is the ``(s, h)`` index of one member of a batched attention product (A7),
    or ``None``.
    """

    m: int  # 1-based position in canonical product order
    name: str
    kind: ProductKind
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    weight: str | None = None
    layer: int | None = None
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


class Replay(ABC):
    """The verifier's per-step glue replay over one transcript (checks 3, 5 and 6b).

    Built by :meth:`DeclaredComputation.replay`. Callers ask for operands in canonical order
    ``1..M``; an instance may cache glue on the way and drop it once no later product needs it.
    Each instance documents what an out-of-order call does.
    """

    @abstractmethod
    def operands(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A_m, B_m)`` rebuilt from committed leaves only."""

    @abstractmethod
    def glue_gradients(self) -> dict[str, torch.Tensor]:
        """Check 6b: the full gradient of each glue-gradient weight. Valid after ``operands(M)``."""


def load_weights(computation: DeclaredComputation, model: torch.nn.Module,
                 weights: Mapping[str, torch.Tensor]) -> None:
    """Copy ``weights`` into ``model``; both must hold exactly the declared weights."""
    params = dict(model.named_parameters())
    names = set(computation.weight_names)
    if set(params) != names:
        raise ValueError(f"model parameters {sorted(params)} differ from the declared weights")
    if set(weights) != names:
        raise ValueError(f"weights {sorted(weights)} differ from the declared weights")
    with torch.no_grad():
        for name in computation.weight_names:
            w, p = weights[name], params[name]
            if w.shape != p.shape or w.dtype != p.dtype:
                raise ValueError(f"{name}: got {w.dtype} {tuple(w.shape)}, "
                                 f"model holds {p.dtype} {tuple(p.shape)}")
            p.copy_(w)


class DeclaredComputation(ABC):
    """Abstract ``C``. Subclasses set the inventory and implement the abstract methods."""

    # ---- the declaration ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_s(self) -> int:
        """Batch size: records per step."""

    @property
    @abstractmethod
    def eta(self) -> float:
        """The constant SGD step size, fixed in ``C`` before step 0 (S8b)."""

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

    @property
    def weight_dtype(self) -> torch.dtype:
        """Dtype of every ``W_t`` / ``W_{t+1}`` leaf. fp32 at both scales (P6); bf16 is config
        (§8.A.3), set by overriding."""
        return torch.float32

    @property
    def product_dtype(self) -> torch.dtype:
        """Dtype of every product leaf ``P_m`` (P6, §8.A.3)."""
        return torch.float32

    @property
    def operand_dtype(self) -> torch.dtype:
        """The format the matmul operands are rounded to: ``ε_in`` of ``e_m`` (P3.a)."""
        return torch.float32

    @property
    def accumulator_dtype(self) -> torch.dtype:
        """The format the matmul accumulates in: ``ε_acc`` of ``e_m`` (P3.a)."""
        return torch.float32

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
            if p.q < 2:
                raise ValueError(f"product {p.name} has q={p.q}; a q=1 product is glue (P7)")
            if p.weight is not None and p.weight not in self._weight_pos:
                raise ValueError(f"product {p.name} names unknown weight {p.weight}")
        if not (isinstance(self.eta, float) and self.eta > 0 and self.eta < float("inf")):
            raise ValueError(f"eta must be a positive finite float, got {self.eta!r}")
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

    # ---- shared ---------------------------------------------------------------------------

    @abstractmethod
    def encode_record(self, record: Any) -> bytes:
        """Canonical leaf bytes of one batch record, identical in ``h`` and ``h_D`` (P9b)."""

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """A fresh model whose parameters are exactly :attr:`weight_names`; load before use."""

    # ---- verifier side --------------------------------------------------------------------

    @abstractmethod
    def replay(self, leaves: LeafReader) -> Replay:
        """Open the glue replay of one step's transcript; it owns a model built from ``W_t``."""

    # ---- prover side (the verifier never calls these) --------------------------------------

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
