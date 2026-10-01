"""One honest SGD step under capture, emitted as the step's leaf objects (spec §6, prover 1).

:func:`prove_step` runs ``C(b, W_t)``: the forward pass and backward pass under
:class:`MatmulCapture`, labeling into canonical slots, the invariant-6 guard, then plain SGD
(S8a, invariant 4). It returns a :class:`StepOutput` holding the leaves in transcript order. It
neither encodes nor hashes; commitment is ``store.py`` (A4).

Fault hooks are prover-side only (S6a, invariant 5) and off by default:

- ``train_records``: train on a different batch from the one committed (A2's activations, A3's
  ``W_{t+1}``);
- ``perturb``: ``{m: fn}`` replaces committed product ``P_m`` with ``fn(copy of P_m)`` after
  capture. Training still uses the true gradients, so only the committed leaf changes (the
  flipped matmul, S6d).

Hidden steps (S5c, P11) are :func:`plain_step` calls between two :func:`prove_step` calls, with
the hidden step's ``W_{t+1}`` fed to the next reported step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .capture import MatmulCapture, MutatedCaptureError, param_storage_map
from .computation import DeclaredComputation

__all__ = ["StepOutput", "prove_step", "plain_step", "load_weights"]

Perturbation = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, eq=False)
class StepOutput:
    """The step's leaf objects, in transcript order once flattened by :meth:`leaves`.

    ``n_leaves`` is copied from the computation for convenience; the verifier takes it from its
    own computation, never from here (invariant 7).
    """

    records: tuple[Any, ...]
    w_t: dict[str, torch.Tensor]  # in weight_names order
    products: tuple[torch.Tensor, ...]  # P_1..P_M
    w_next: dict[str, torch.Tensor]  # in weight_names order
    n_leaves: int
    loss: float
    _versions: tuple[int, ...] = field(default=(), repr=False)

    def leaves(self) -> list[Any]:
        return [*self.records, *self.w_t.values(), *self.products, *self.w_next.values()]

    def _tensors(self) -> list[torch.Tensor]:
        return [x for x in self.leaves() if isinstance(x, torch.Tensor)]

    def assert_unmodified(self) -> None:
        """Raise if any leaf tensor was mutated in place since emission (invariant 6).

        Call it after hashing. Products alias autograd's buffers (``G_ℓ`` is ``W_ℓ.grad``), so a
        ``zero_grad(set_to_none=False)`` on the model would trip it.
        """
        now = tuple(t._version for t in self._tensors())
        if now != self._versions:
            bad = [i for i, (a, b) in enumerate(zip(now, self._versions)) if a != b]
            raise MutatedCaptureError(f"step leaves mutated after emission: tensor positions {bad}")


def load_weights(computation: DeclaredComputation, model: torch.nn.Module,
                 weights: Mapping[str, torch.Tensor]) -> None:
    """Copy ``weights`` into ``model``; names must be exactly the declared weights."""
    params = dict(model.named_parameters())
    if set(params) != set(computation.weight_names):
        raise ValueError(f"model parameters {sorted(params)} differ from the declared weights")
    if set(weights) != set(computation.weight_names):
        raise ValueError(f"weights {sorted(weights)} differ from the declared weights")
    with torch.no_grad():
        for name in computation.weight_names:
            w, p = weights[name], params[name]
            if w.shape != p.shape or w.dtype != p.dtype:
                raise ValueError(f"{name}: got {w.dtype} {tuple(w.shape)}, "
                                 f"model holds {p.dtype} {tuple(p.shape)}")
            p.copy_(w)


def _snapshot(computation: DeclaredComputation, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: model.get_parameter(n).detach().clone() for n in computation.weight_names}


def _optimizer(model: torch.nn.Module, eta: float) -> torch.optim.SGD:
    # Invariant 4 (S8a): plain SGD, nothing else on.
    return torch.optim.SGD(model.parameters(), lr=eta, momentum=0, weight_decay=0)


def plain_step(computation: DeclaredComputation, model: torch.nn.Module,
               w_t: Mapping[str, torch.Tensor], records: Sequence[Any],
               eta: float) -> tuple[dict[str, torch.Tensor], float]:
    """The same SGD step with no capture and no transcript: ``(W_{t+1}, loss)``.

    For hidden steps, η tuning and the capture-on/off comparison.
    """
    load_weights(computation, model, w_t)
    model.zero_grad(set_to_none=True)
    opt = _optimizer(model, eta)
    loss = computation.loss(model, records)
    loss.backward()
    opt.step()
    model.zero_grad(set_to_none=True)
    return _snapshot(computation, model), float(loss.detach())


def prove_step(
    computation: DeclaredComputation,
    model: torch.nn.Module,
    w_t: Mapping[str, torch.Tensor],
    records: Sequence[Any],
    eta: float,
    *,
    train_records: Sequence[Any] | None = None,
    perturb: Mapping[int, Perturbation] | None = None,
) -> StepOutput:
    """Execute ``C(b, W_t)`` on ``model`` and emit the step's leaves.

    ``records`` is the committed batch ``b``. ``model`` is overwritten with ``w_t`` and left
    holding ``W_{t+1}`` with its gradients cleared.
    """
    if len(records) != computation.n_s:
        raise ValueError(f"batch has {len(records)} records, the computation declares "
                         f"{computation.n_s}")
    if not torch.is_grad_enabled():
        raise RuntimeError("prove_step needs grad mode on")
    load_weights(computation, model, w_t)
    w_t_leaves = _snapshot(computation, model)  # parameters change in place at the step
    model.zero_grad(set_to_none=True)
    opt = _optimizer(model, eta)

    cap = MatmulCapture(param_names=param_storage_map(model))
    with cap:
        with cap.phase("forward"):
            loss = computation.loss(model, records if train_records is None else train_records)
        with cap.phase("backward"):
            loss.backward()
    products = [p.detach() for p in computation.label(cap, model)]
    if len(products) != computation.M:
        raise RuntimeError(f"label returned {len(products)} products, M = {computation.M}")
    # Invariant 6: the products are fixed; nothing has touched the captured tensors.
    cap.assert_unmodified()
    cap.release_operands()

    for m, fn in (perturb or {}).items():
        spec = computation.product(m)
        new = fn(products[m - 1].clone())
        if tuple(new.shape) != spec.p_shape or new.dtype != products[m - 1].dtype:
            raise ValueError(f"perturbation of {spec.name} changed its shape or dtype")
        products[m - 1] = new.detach().contiguous()

    opt.step()
    w_next = _snapshot(computation, model)
    # set_to_none drops the model's reference; G_ℓ products stay intact (invariant 6).
    model.zero_grad(set_to_none=True)

    out = StepOutput(records=tuple(records), w_t=w_t_leaves, products=tuple(products),
                     w_next=w_next, n_leaves=computation.n_leaves, loss=float(loss.detach()))
    object.__setattr__(out, "_versions", tuple(t._version for t in out._tensors()))
    return out
