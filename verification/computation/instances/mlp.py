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

Every weight is linear, so check 6 is 6a throughout and :meth:`MLPReplay.glue_gradients` is
empty. Each product needs ``q ≥ 2`` (P7: a ``q = 1`` product is a glue outer product), so
``n_s ≥ 2`` and every width is at least 2.

The schedule is ``data.schedule`` (sequential, 1-based, no wraparound).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from torch import nn

from verification.commitment.encoding import TAG_MLP_RECORD, encode_tensor_leaf
from verification.computation.interface import (
    DeclaredComputation,
    LabelingError,
    ProductKind,
    ProductSpec,
    Replay,
    load_weights,
)
from verification.transcript.reader import LeafReader, TranscriptView

if TYPE_CHECKING:  # labeling's type only; nothing here imports the prover at run time
    from verification.prover.capture import MatmulCapture

__all__ = [
    "DEFAULT_WIDTHS",
    "MLP",
    "MLPComputation",
    "MLPReplay",
    "make_record",
    "split_record",
    "init_weights",
    "synthetic_dataset",
]

DEFAULT_WIDTHS = (16, 32, 32, 8)


class MLP(nn.Module):
    """``tanh`` MLP with bias-free linears; ``forward`` returns ``Y_L``.

    Construction leaves the global RNG untouched; load weights before use.
    """

    def __init__(self, widths: Sequence[int]) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.layers = nn.ModuleList(
                nn.Linear(i, o, bias=False) for i, o in zip(widths[:-1], widths[1:]))
        self.act = nn.Tanh()
        self.loss_fn = nn.MSELoss(reduction="mean")

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
    """``(x, y)`` along the last dimension; works on one record or a stacked batch."""
    return record[..., :d_in], record[..., d_in:]


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


class MLPComputation(DeclaredComputation):
    """``C`` for :class:`MLP` at the given widths, batch size and step size."""

    def __init__(self, widths: Sequence[int] = DEFAULT_WIDTHS, n_s: int = 4, *,
                 eta: float) -> None:
        widths = tuple(int(w) for w in widths)
        if len(widths) < 2:
            raise ValueError("need at least one layer")
        # P7: every product must have q ≥ 2; q is a width (forward, input grad) or n_s (G_ℓ).
        if n_s < 2 or min(widths) < 2:
            raise ValueError(f"n_s={n_s} and widths={widths} must all be ≥ 2: a q=1 product "
                             f"is a glue outer product (P7), not a checked matmul")
        self.widths = widths
        self.L = len(widths) - 1
        self._n_s = int(n_s)
        self._eta = float(eta)
        self._names = tuple(f"layers.{l}.weight" for l in range(self.L))
        self.validate()

    # ---- declaration ----------------------------------------------------------------------

    @property
    def n_s(self) -> int:
        return self._n_s

    @property
    def eta(self) -> float:
        return self._eta

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
            specs.append((f"Y_{l}", ProductKind.FORWARD, (n, i), (i, o), l))
        for l in range(self.L, 0, -1):
            i, o = w[l - 1], w[l]
            if l >= 2:
                specs.append((f"dX_{l}", ProductKind.INPUT_GRAD, (n, o), (o, i), l))
            specs.append((f"G_{l}", ProductKind.WEIGHT_GRAD, (o, n), (n, i), l))
        return tuple(ProductSpec(m, name, kind, a, b, weight=self.weight(l), layer=l)
                     for m, (name, kind, a, b, l) in enumerate(specs, start=1))

    @cached_property
    def _m_by_name(self) -> dict[str, int]:
        return {p.name: p.m for p in self.products}

    def m_of(self, name: str) -> int:
        return self._m_by_name[name]

    @cached_property
    def linear_weights(self) -> dict[str, int]:
        return {self.weight(l): self.m_of(f"G_{l}") for l in range(1, self.L + 1)}

    # ---- shared ---------------------------------------------------------------------------

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
        x, y = split_record(torch.stack(list(records)), self.d_in)
        return x.contiguous(), y.contiguous()

    def build_model(self) -> MLP:
        return MLP(self.widths)

    # ---- verifier side --------------------------------------------------------------------

    def replay(self, leaves: LeafReader) -> MLPReplay:
        return MLPReplay(self, leaves)

    # ---- prover side ----------------------------------------------------------------------

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


class MLPReplay(Replay):
    """Glue replay of one MLP step from committed leaves.

    Owns an :class:`MLP` loaded from the committed ``W_t``; glue runs through its ``act`` and
    ``loss_fn`` (invariant 2). ``X_ℓ`` and ``δY_ℓ`` are cached on first use and dropped after
    ``G_ℓ``, their last consumer in canonical order. An out-of-order call recomputes what it
    needs from the leaves, so any order gives the same operands; canonical order computes each
    glue value once.
    """

    def __init__(self, computation: MLPComputation, leaves: LeafReader) -> None:
        self.c = computation
        self.view = TranscriptView(computation, leaves)
        self.model = computation.build_model()
        load_weights(computation, self.model,
                     {n: self.view.w_t(n) for n in computation.weight_names})
        self._x: dict[int, torch.Tensor] = {}
        self._dy: dict[int, torch.Tensor] = {}
        self._targets: torch.Tensor | None = None
        self._done = False

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        x1, targets = self.c.assemble(self.view.records())
        self._targets = targets
        return x1, targets

    def x(self, l: int) -> torch.Tensor:
        """``X_ℓ``: from the batch for ``ℓ = 1``, else ``tanh`` of the committed ``Y_{ℓ−1}``."""
        if l not in self._x:
            if l == 1:
                self._x[l] = self._batch()[0]
            else:
                with torch.no_grad():
                    self._x[l] = self.model.act(self.view.product(self.c.m_of(f"Y_{l - 1}")))
        return self._x[l]

    def dy(self, l: int) -> torch.Tensor:
        """``δY_ℓ`` through autograd on the model's ``loss_fn`` (``ℓ = L``) or ``act``."""
        if l not in self._dy:
            y = self.view.product(self.c.m_of(f"Y_{l}")).detach().requires_grad_(True)
            with torch.enable_grad():
                if l == self.c.L:
                    targets = self._targets if self._targets is not None else self._batch()[1]
                    (g,) = torch.autograd.grad(self.model.loss_fn(y, targets), y)
                else:
                    upstream = self.view.product(self.c.m_of(f"dX_{l + 1}"))
                    (g,) = torch.autograd.grad(self.model.act(y), y, grad_outputs=upstream)
            self._dy[l] = g.detach()
        return self._dy[l]

    def operands(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.c.product(m)
        l = spec.layer
        w = self.view.w_t(spec.weight)
        if spec.kind is ProductKind.FORWARD:
            out = self.x(l), w.t()
        elif spec.kind is ProductKind.INPUT_GRAD:
            out = self.dy(l), w
        else:
            out = self.dy(l).t(), self.x(l)
            self._x.pop(l, None)
            self._dy.pop(l, None)
        if m == self.c.M:
            self._done = True
        return out

    def glue_gradients(self) -> dict[str, torch.Tensor]:
        if not self._done:
            raise RuntimeError("glue_gradients is valid only after operands(M)")
        return {}
