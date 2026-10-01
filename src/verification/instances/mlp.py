"""The degenerate MLP instance of ``C`` (ref block §9), for the M1 smoke test.

Model: ``Y_ℓ = X_ℓ·W_ℓᵀ`` for ``ℓ = 1..L`` with ``nn.Linear(bias=False)``, ``X_{ℓ+1} = tanh(Y_ℓ)``
for ``ℓ < L``, and the loss ``nn.MSELoss(reduction="mean")`` of ``Y_L`` against the targets.
``L = len(widths) − 1``. ``X_1`` is batch data, so the step never computes ``δX_1``, and
``M = 3L − 1``.

Implementation pins (not fixed by the reference block):

- **Record.** One batch record is one float32 tensor ``[d_in + d_out]``: the input row followed
  by the target row. Its leaf is the tensor leaf with tag ``0x04``,
  ``encode_tensor_leaf(TAG_MLP_RECORD, record)``, identical in ``h`` and ``h_D`` (P9b).
  Assembly (glue) stacks the records in batch order and splits the columns into ``X_1`` and
  the targets.
- **Canonical product order**, mirroring ref block §6: forward ``Y_1 … Y_L``, then for
  ``ℓ = L … 1`` the input gradient ``δX_ℓ`` (when ``ℓ ≥ 2``) followed by the weight gradient
  ``G_ℓ``. Names are ``Y_ℓ``, ``dX_ℓ`` and ``G_ℓ``.
- **Weights** are named as ``MLP.named_parameters()`` names them, ``layers.{ℓ−1}.weight``,
  in order ``ℓ = 1..L``, each ``[o, i]``.

Products, with ``δY_L = ∂loss/∂Y_L`` and ``δY_ℓ = δX_{ℓ+1} ⊙ tanh′(Y_ℓ)`` for ``ℓ < L``:

| product | A | B | P | q | width |
|---|---|---|---|---|---|
| ``Y_ℓ`` | ``X_ℓ`` | ``W_ℓᵀ`` | ``n_s × o`` | ``i`` | ``o`` |
| ``δX_ℓ`` | ``δY_ℓ`` | ``W_ℓ`` | ``n_s × i`` | ``o`` | ``i`` |
| ``G_ℓ`` | ``δY_ℓᵀ`` | ``X_ℓ`` | ``o × i`` | ``n_s`` | ``i`` |

Every weight is linear, so check 6 is 6a throughout and :meth:`MLPComputation.glue_gradients`
is empty.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cached_property

import torch
from torch import nn

from ..capture import MatmulCapture
from ..computation import (
    DeclaredComputation,
    LabelingError,
    LeafReader,
    ProductKind,
    ProductSpec,
    TranscriptView,
)
from ..encoding import TAG_MLP_RECORD, encode_tensor_leaf

__all__ = [
    "DEFAULT_WIDTHS",
    "MLP",
    "MLPComputation",
    "make_record",
    "split_record",
    "init_weights",
    "synthetic_dataset",
    "sequential_schedule",
]

DEFAULT_WIDTHS = (16, 32, 32, 8)


def _act() -> nn.Module:
    return nn.Tanh()


def _loss_fn() -> nn.Module:
    return nn.MSELoss(reduction="mean")


class MLP(nn.Module):
    """``tanh`` MLP with bias-free linears; ``forward`` returns ``Y_L``.

    Construction leaves the global RNG untouched; load weights before use.
    """

    def __init__(self, widths: Sequence[int]) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.layers = nn.ModuleList(
                nn.Linear(i, o, bias=False) for i, o in zip(widths[:-1], widths[1:]))
        self.act = _act()
        self.loss_fn = _loss_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l < len(self.layers) - 1:
                x = self.act(x)
        return x


def make_record(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One record: ``x ‖ y`` as a contiguous float32 ``[d_in + d_out]`` tensor."""
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("x and y must be 1-D")
    return torch.cat([x, y]).to(torch.float32).contiguous()


def split_record(record: torch.Tensor, d_in: int) -> tuple[torch.Tensor, torch.Tensor]:
    return record[:d_in], record[d_in:]


def init_weights(widths: Sequence[int], seed: int) -> dict[str, torch.Tensor]:
    """Seeded ``W_0``: entries ``Uniform(−1/√i, 1/√i)``, the ``nn.Linear`` default range."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l, (i, o) in enumerate(zip(widths[:-1], widths[1:])):
        bound = 1.0 / math.sqrt(i)
        out[f"layers.{l}.weight"] = (torch.rand(o, i, generator=g) * 2 - 1) * bound
    return out


def synthetic_dataset(widths: Sequence[int], n_records: int, seed: int) -> list[torch.Tensor]:
    """``n_records`` records with ``x ~ N(0, 1)`` and ``y`` from a seeded teacher MLP."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_records, widths[0], generator=g)
    teacher = MLP(widths)
    with torch.no_grad():
        for name, w in init_weights(widths, seed + 1).items():
            teacher.get_parameter(name).copy_(w)
        y = teacher(x)
    return [make_record(x[i], y[i]) for i in range(n_records)]


def sequential_schedule(n_records: int, n_s: int, steps: int) -> list[list[int]]:
    """``π(t) = (t·n_s + j) mod |D|`` for ``j < n_s``: consecutive records, wrapping."""
    return [[(t * n_s + j) % n_records for j in range(n_s)] for t in range(steps)]


class MLPComputation(DeclaredComputation):
    """``C`` for :class:`MLP` at the given widths and batch size."""

    def __init__(self, widths: Sequence[int] = DEFAULT_WIDTHS, n_s: int = 4) -> None:
        if len(widths) < 2 or n_s < 1:
            raise ValueError("need at least one layer and one record")
        self.widths = tuple(int(w) for w in widths)
        self.L = len(self.widths) - 1
        self._n_s = int(n_s)
        self._names = tuple(f"layers.{l}.weight" for l in range(self.L))
        # Glue modules, built by the same factories as the model's (invariant 2). Stateless.
        self._act = _act()
        self._loss_fn = _loss_fn()
        self.validate()

    # ---- declaration ----------------------------------------------------------------------

    @property
    def n_s(self) -> int:
        return self._n_s

    @property
    def d_in(self) -> int:
        return self.widths[0]

    @property
    def d_out(self) -> int:
        return self.widths[-1]

    @property
    def weight_names(self) -> tuple[str, ...]:
        return self._names

    def weight(self, l: int) -> str:
        """Name of ``W_ℓ``, 1-based ``ℓ``."""
        return self._names[l - 1]

    @cached_property
    def weight_shapes(self) -> dict[str, tuple[int, int]]:
        return {self.weight(l): (self.widths[l], self.widths[l - 1])
                for l in range(1, self.L + 1)}

    @cached_property
    def products(self) -> tuple[ProductSpec, ...]:
        n, w = self.n_s, self.widths
        specs: list[tuple] = []
        for l in range(1, self.L + 1):
            i, o = w[l - 1], w[l]
            specs.append((f"Y_{l}", ProductKind.FORWARD, (n, i), (i, o), self.weight(l)))
        for l in range(self.L, 0, -1):
            i, o = w[l - 1], w[l]
            if l >= 2:
                specs.append((f"dX_{l}", ProductKind.INPUT_GRAD, (n, o), (o, i), self.weight(l)))
            specs.append((f"G_{l}", ProductKind.WEIGHT_GRAD, (o, n), (n, i), self.weight(l)))
        return tuple(ProductSpec(m, name, kind, a, b, weight)
                     for m, (name, kind, a, b, weight) in enumerate(specs, start=1))

    @cached_property
    def _m_by_name(self) -> dict[str, int]:
        return {p.name: p.m for p in self.products}

    def m_of(self, name: str) -> int:
        return self._m_by_name[name]

    @cached_property
    def linear_weights(self) -> dict[str, int]:
        return {self.weight(l): self.m_of(f"G_{l}") for l in range(1, self.L + 1)}

    # ---- batch glue -----------------------------------------------------------------------

    def encode_record(self, record: torch.Tensor) -> bytes:
        self._check_record(record)
        return encode_tensor_leaf(TAG_MLP_RECORD, record)

    def _check_record(self, record: torch.Tensor) -> None:
        if (not isinstance(record, torch.Tensor) or record.dtype != torch.float32
                or tuple(record.shape) != (self.d_in + self.d_out,)):
            raise ValueError(
                f"an MLP record is a float32 [{self.d_in + self.d_out}] tensor, got "
                f"{getattr(record, 'dtype', type(record))} {tuple(getattr(record, 'shape', ()))}")

    def assemble(self, records: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Glue: stack the records in batch order and split them into ``(X_1, targets)``."""
        if len(records) != self.n_s:
            raise ValueError(f"batch has {len(records)} records, the computation declares {self.n_s}")
        for r in records:
            self._check_record(r)
        batch = torch.stack(list(records))
        return batch[:, :self.d_in].contiguous(), batch[:, self.d_in:].contiguous()

    # ---- verifier side --------------------------------------------------------------------

    def _x(self, l: int, view: TranscriptView) -> torch.Tensor:
        """``X_ℓ``: from the batch for ``ℓ = 1``, else ``tanh`` of the committed ``Y_{ℓ−1}``."""
        if l == 1:
            return self.assemble(view.records())[0]
        with torch.no_grad():
            return self._act(view.product(self.m_of(f"Y_{l - 1}")))

    def _dy(self, l: int, view: TranscriptView) -> torch.Tensor:
        """``δY_ℓ``, recomputed through autograd on the model's loss and activation modules."""
        y = view.product(self.m_of(f"Y_{l}")).detach().requires_grad_(True)
        with torch.enable_grad():
            if l == self.L:
                targets = self.assemble(view.records())[1]
                (g,) = torch.autograd.grad(self._loss_fn(y, targets), y)
            else:
                upstream = view.product(self.m_of(f"dX_{l + 1}"))
                (g,) = torch.autograd.grad(self._act(y), y, grad_outputs=upstream)
        return g.detach()

    def operands(self, m: int, leaves: LeafReader) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.product(m)
        view = TranscriptView(self, leaves)
        l = self.weight_names.index(spec.weight) + 1
        w = view.w_t(spec.weight)
        if spec.kind is ProductKind.FORWARD:
            return self._x(l, view), w.t()
        if spec.kind is ProductKind.INPUT_GRAD:
            return self._dy(l, view), w
        return self._dy(l, view).t(), self._x(l, view)

    def glue_gradients(self, leaves: LeafReader) -> dict[str, torch.Tensor]:
        return {}

    # ---- prover side ----------------------------------------------------------------------

    def build_model(self) -> MLP:
        return MLP(self.widths)

    def loss(self, model: nn.Module, records: Sequence[torch.Tensor]) -> torch.Tensor:
        x, targets = self.assemble(records)
        return model.loss_fn(model(x), targets)

    def label(self, capture: MatmulCapture, model: nn.Module) -> list[torch.Tensor]:
        """Map by operand identity:

        - forward ``Y_ℓ``: ``B`` is a view of ``W_ℓ`` (``b_info.param_name``);
        - backward ``δX_ℓ``: ``B`` is ``W_ℓ`` itself (``b_info.param_name``);
        - backward ``G_ℓ``: ``B`` is the saved ``X_ℓ``, the storage of ``Y_ℓ``'s ``A``.
          Its output must also be the storage autograd adopted as ``W_ℓ.grad``.
        """
        if capture.glue_outer:
            raise LabelingError(f"unexpected q=1 products: {len(capture.glue_outer)}")
        layer = {n: l for l, n in enumerate(self.weight_names, start=1)}
        slots: dict[str, torch.Tensor] = {}
        x_ptr: dict[int, int] = {}  # storage of X_ℓ (forward A) -> ℓ

        def fill(name: str, rec) -> None:
            if name in slots:
                raise LabelingError(f"slot {name} filled twice (record #{rec.index})")
            spec = self.product(self.m_of(name))
            got = (tuple(rec.a.shape), tuple(rec.b.shape), tuple(rec.out.shape))
            if got != (spec.a_shape, spec.b_shape, spec.p_shape):
                raise LabelingError(f"{name}: shapes {got} != declared "
                                    f"{(spec.a_shape, spec.b_shape, spec.p_shape)}")
            slots[name] = rec.out

        for rec in capture.records:
            if rec.batch is not None:
                raise LabelingError(f"record #{rec.index}: the MLP has no batched products")
            if rec.a is None:
                raise LabelingError("capture operands were released before labeling")
        for rec in (r for r in capture.records if r.phase == "forward"):
            l = layer.get(rec.b_info.param_name)
            if l is None:
                raise LabelingError(f"forward record #{rec.index} has no weight operand")
            if rec.a_info.storage_ptr in x_ptr:
                raise LabelingError(f"forward record #{rec.index}: X_{l} aliases another input")
            x_ptr[rec.a_info.storage_ptr] = l
            fill(f"Y_{l}", rec)
        for rec in (r for r in capture.records if r.phase == "backward"):
            if rec.b_info.param_name in layer:
                fill(f"dX_{layer[rec.b_info.param_name]}", rec)
            elif rec.b_info.storage_ptr in x_ptr:
                l = x_ptr[rec.b_info.storage_ptr]
                grad = model.get_parameter(self.weight(l)).grad
                if grad is None or (grad.untyped_storage().data_ptr()
                                    != rec.out.untyped_storage().data_ptr()):
                    raise LabelingError(f"G_{l} (record #{rec.index}) is not W_{l}.grad")
                fill(f"G_{l}", rec)
            else:
                raise LabelingError(f"backward record #{rec.index} matches no declared slot")
        missing = [p.name for p in self.products if p.name not in slots]
        if missing:
            raise LabelingError(f"slots never filled: {missing}")
        return [slots[p.name] for p in self.products]

