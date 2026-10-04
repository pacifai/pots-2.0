"""One honest SGD step under capture, emitted as the step's leaf objects (spec §6, prover 1).

:func:`prove_step` runs ``C(b, W_t)``: the forward pass and backward pass under
:class:`MatmulCapture`, labeling into canonical slots, the invariant-6 guard, then plain SGD
at the declared ``computation.eta`` (S8a, S8b, invariant 4). It returns a :class:`StepOutput`
holding the leaves in transcript order. :func:`commit` then builds the step's Merkle tree with
the shared leaf hashing of ``commitment/leaves.py`` (A4).

Fault hooks are prover-side only (S6a, invariant 5) and off by default:

- ``train_records``: train on a different batch from the one committed. One poisoned step
  supplies A2 (commit the clean ``b`` with the poisoned activations) and A1 (the caller commits
  ``b̃`` itself, which is an honest ``prove_step`` on ``b̃``);
- ``perturb``: ``{m: fn}`` replaces committed product ``P_m`` with ``fn(copy of P_m)`` after
  capture. Training still uses the true gradients, so only the committed leaf changes.

Other cheats are assembled by the caller from honest outputs, not by hooks here (S6e):

- **A3** (an honest clean transcript with a poisoned ``W_{t+1}``) splices the ``w_next`` of a
  poisoned step into the clean step's leaves.
- **Hidden steps** (S5c, P11) are :func:`plain_step` calls between two :func:`prove_step`
  calls, with the hidden step's ``W_{t+1}`` fed to the next reported step.
- The **flipped-matmul sweep** (S6f) perturbs one product leaf at the store level and re-roots
  with ``MerkleTree.update_leaf``, rather than re-running ``prove_step`` per sweep point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from verification.commitment.leaves import commit_leaves
from verification.commitment.merkle import MerkleTree
from verification.computation.interface import DeclaredComputation, load_weights
from verification.computation.matmul_ops import param_storage_map
from verification.prover.capture import MatmulCapture, MutatedCaptureError

__all__ = ["StepOutput", "Section", "no_section", "prove_step", "plain_step", "commit"]

Perturbation = Callable[[torch.Tensor], torch.Tensor]
# A metrics seam: ``section(name)`` wraps one phase of the step (``runs/metrics.py``, B6). It
# observes only; with ``None`` the step runs exactly as without it.
Section = Callable[[str], AbstractContextManager[Any]]
_NO_SECTION = nullcontext()


def no_section(name: str) -> AbstractContextManager[Any]:
    return _NO_SECTION


@dataclass(frozen=True, eq=False)
class StepOutput:
    """The step's leaf objects, in transcript order once flattened by :meth:`leaves`.

    It carries no leaf count: the verifier takes ``n_leaves`` from its computation (invariant 7).
    ``versions`` holds the ``_version`` of every tensor leaf, in :meth:`leaves` order. A
    captured product's version is taken right after ``MatmulCapture.assert_unmodified``.
    """

    records: tuple[Any, ...]
    w_t: dict[str, torch.Tensor]  # in weight_names order
    products: tuple[torch.Tensor, ...]  # P_1..P_M
    w_next: dict[str, torch.Tensor]  # in weight_names order
    loss: float
    versions: tuple[int, ...] = field(repr=False)

    def leaves(self) -> list[Any]:
        return [*self.records, *self.w_t.values(), *self.products, *self.w_next.values()]

    def with_w_next(self, w_next: Mapping[str, torch.Tensor]) -> StepOutput:
        """This step with ``W_{t+1}`` replaced, every other leaf kept: the caller-side A3 splice.

        ``w_next`` must name the same weights; it is stored in this step's weight order as
        contiguous clones, so the caller's tensors can change later without touching the leaves
        (invariant 6), and its versions are taken now, so the splice passes
        :meth:`assert_unmodified`.
        """
        if set(w_next) != set(self.w_next):
            raise ValueError("the spliced W_{t+1} names different weights")
        new = {n: w_next[n].detach().clone(memory_format=torch.contiguous_format)
               for n in self.w_next}
        n_w = len(self.w_next)
        return StepOutput(records=self.records, w_t=self.w_t, products=self.products,
                          w_next=new, loss=self.loss,
                          versions=self.versions[:-n_w] + _versions(list(new.values())))

    def assert_unmodified(self) -> None:
        """Raise if any leaf tensor was mutated in place since capture (invariant 6).

        Call it after hashing. Products alias autograd's buffers (``G_ℓ`` is ``W_ℓ.grad``), so a
        ``zero_grad(set_to_none=False)`` on the model would trip it.
        """
        now = _versions(self.leaves())
        if now != self.versions:
            bad = [i for i, (a, b) in enumerate(zip(now, self.versions)) if a != b]
            raise MutatedCaptureError(f"step leaves mutated after capture: tensor positions {bad}")


def _versions(objs: Sequence[Any]) -> tuple[int, ...]:
    return tuple(t._version for t in objs if isinstance(t, torch.Tensor))


def _snapshot(computation: DeclaredComputation, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: model.get_parameter(n).detach().clone() for n in computation.weight_names}


def _optimizer(computation: DeclaredComputation, model: torch.nn.Module) -> torch.optim.SGD:
    # Invariant 4 (S8a): plain SGD, nothing else on.
    return torch.optim.SGD(model.parameters(), lr=computation.eta, momentum=0, weight_decay=0)


def plain_step(computation: DeclaredComputation, model: torch.nn.Module,
               w_t: Mapping[str, torch.Tensor],
               records: Sequence[Any], *,
               section: Section | None = None) -> tuple[dict[str, torch.Tensor], float]:
    """The same SGD step with no capture and no transcript: ``(W_{t+1}, loss)``.

    For hidden steps, η tuning and the capture-on/off comparison. With ``section`` it is EQ1b's
    P0, the plain baseline: ``P0.load``, ``P0.forward``, ``P0.backward`` and ``P0.update`` cover
    the same work as :func:`prove_step`'s ``train.*`` phases. Copying out ``W_{t+1}`` is the
    caller's and sits outside them.
    """
    sec = section or no_section
    with sec("P0.load"):
        load_weights(computation, model, w_t)
        model.zero_grad(set_to_none=True)
        opt = _optimizer(computation, model)
    with sec("P0.forward"):
        loss = computation.loss(model, records)
    with sec("P0.backward"):
        loss.backward()
    with sec("P0.update"):
        opt.step()
        model.zero_grad(set_to_none=True)
    return _snapshot(computation, model), float(loss.detach())


def prove_step(
    computation: DeclaredComputation,
    model: torch.nn.Module,
    w_t: Mapping[str, torch.Tensor],
    records: Sequence[Any],
    *,
    train_records: Sequence[Any] | None = None,
    perturb: Mapping[int, Perturbation] | None = None,
    section: Section | None = None,
) -> StepOutput:
    """Execute ``C(b, W_t)`` on ``model`` and emit the step's leaves.

    ``records`` is the committed batch ``b``. ``model`` is overwritten with ``w_t`` and left
    holding ``W_{t+1}`` with its gradients cleared. ``section`` is the B6 metrics seam: it
    wraps the phases ``train.load``, ``P2.w_t``, ``train.forward``, ``train.backward``,
    ``P1.label``, ``train.update`` and ``P2.w_next``. ``train.*`` is training under capture, not
    EQ1b's P0, which only :func:`plain_step` measures (``runs/metrics.py`` maps the rows).
    """
    if len(records) != computation.n_s:
        raise ValueError(f"batch has {len(records)} records, the computation declares "
                         f"{computation.n_s}")
    if not torch.is_grad_enabled():
        raise RuntimeError("prove_step needs grad mode on")
    sec = section or no_section
    record_versions = _versions(records)
    with sec("train.load"):
        load_weights(computation, model, w_t)
    with sec("P2.w_t"):
        w_t_leaves = _snapshot(computation, model)  # parameters change in place at the step
    with sec("train.load"):
        model.zero_grad(set_to_none=True)
        opt = _optimizer(computation, model)

    cap = MatmulCapture(param_names=param_storage_map(model))
    with cap:
        with cap.phase("forward"), sec("train.forward"):
            loss = computation.loss(model, records if train_records is None else train_records)
        with cap.phase("backward"), sec("train.backward"):
            loss.backward()
    with sec("P1.label"):
        products = [p.detach() for p in computation.label(cap, model)]
        if len(products) != computation.M:
            raise RuntimeError(f"label returned {len(products)} products, M = {computation.M}")
        # Invariant 6: the products are fixed and nothing has touched the captured tensors.
        cap.assert_unmodified()
        product_versions = list(_versions(products))
        cap.release_operands()

    for m, fn in (perturb or {}).items():
        spec = computation.product(m)
        new = fn(products[m - 1].clone())
        if tuple(new.shape) != spec.p_shape or new.dtype != products[m - 1].dtype:
            raise ValueError(f"perturbation of {spec.name} changed its shape or dtype")
        products[m - 1] = new.detach().contiguous()
        product_versions[m - 1] = products[m - 1]._version

    with sec("train.update"):
        opt.step()
    with sec("P2.w_next"):
        w_next = _snapshot(computation, model)
    with sec("train.update"):
        # set_to_none drops the model's reference; G_ℓ products stay intact (invariant 6).
        model.zero_grad(set_to_none=True)

    return StepOutput(
        records=tuple(records), w_t=w_t_leaves, products=tuple(products), w_next=w_next,
        loss=float(loss.detach()),
        versions=(record_versions + _versions(list(w_t_leaves.values()))
                  + tuple(product_versions) + _versions(list(w_next.values()))),
    )


def commit(c: DeclaredComputation, step: StepOutput) -> MerkleTree:
    """Prover step 2: the Merkle tree over the step's leaves; ``.root`` is ``h``."""
    tree = commit_leaves(c, step.leaves())
    step.assert_unmodified()  # invariant 6: nothing moved while it was hashed
    return tree
