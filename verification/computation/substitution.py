"""Run a model's own code with every checked product replaced by a supplied tensor (S4b).

The verifier rebuilds glue by running the unmodified model, but the matrix products it would
compute are the prover's claims, not the verifier's: their committed values must flow into the
glue that follows them (check 3). :class:`ProductSubstitution` is a ``TorchDispatchMode`` that
intercepts every ``aten.mm``, ``aten.bmm`` and biased ``aten.addmm`` with contracted dimension
``q ≥ 2``, hands its operands to a ``supply(op, a, b, bias)`` callback and returns the
callback's tensor in place of the product. Everything between the products (norms, RoPE,
masking, softmax, SiLU, residual adds) runs through the model's own modules, unchanged.

- A ``q = 1`` product is glue (P7, the RoPE angle table) and runs as normal.
- ``addmm(bias, a, b)`` with a bias vector and ``α = β = 1`` is a biased linear (F14). Its
  committed leaf is the op's output, bias included, so the callback gets ``bias`` to check it
  is the declared one. ``bias`` is ``None`` for ``mm`` and ``bmm``.
- Any other ``addmm``, ``baddbmm``, any :data:`~verification.computation.matmul_ops.REJECTED_OPS`
  op, a non-``default`` overload, and any op outside the trusted namespaces raise
  :class:`SubstitutionError`, under the same lists the prover's capture uses. The products seen
  here are then the same calls the capture records.
- The supplied tensor must have the op's output shape and ``a``'s dtype; it is moved to ``a``'s
  device. Return a fresh tensor, never a committed leaf itself: autograd attaches history to an
  op's output, and a leaf must not change (invariant 6).

Instance-agnostic: the callback decides which committed leaf each call stands for.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from verification.computation.interface import ReplayError
from verification.computation.matmul_ops import (
    ALLOWED_NAMESPACE_OPS,
    HANDLED_OPS,
    REJECTED_OPS,
    TRUSTED_NAMESPACES,
)

__all__ = ["SubstitutionError", "ProductSubstitution", "Supply"]

# (op name "mm" | "bmm" | "addmm", a, b, bias or None) -> the tensor that stands for the op
Supply = Callable[[str, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]

_SUBSTITUTED = ("mm", "bmm", "addmm")


class SubstitutionError(ReplayError):
    """A product could not be substituted: an unhandled matmul op, or a bad supplied tensor."""


class ProductSubstitution(TorchDispatchMode):
    """Dispatch mode that returns ``supply(op, a, b, bias)`` for every checked product."""

    def __init__(self, supply: Supply) -> None:
        super().__init__()
        self.supply = supply

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        if (func.namespace not in TRUSTED_NAMESPACES
                and f"{func.namespace}::{name}" not in ALLOWED_NAMESPACE_OPS):
            raise SubstitutionError(f"{func} reached dispatch; an op outside the trusted "
                                    f"namespaces may hide a product")
        if func.namespace == "aten" and (name in REJECTED_OPS or name in HANDLED_OPS):
            if name not in _SUBSTITUTED or func._overloadname != "default":
                raise SubstitutionError(f"{func} reached dispatch; only mm, bmm and biased "
                                        f"addmm (.default) are substituted")
            bias = None
            if name == "addmm":
                bias, a, b = args[0], args[1], args[2]
                if (kwargs.get("alpha", 1) != 1 or kwargs.get("beta", 1) != 1
                        or tuple(bias.shape) != (b.shape[-1],)):
                    raise SubstitutionError(f"{func}: only a bias vector added once (F14) is "
                                            f"substituted")
            else:
                a, b = args[0], args[1]
            if a.shape[-1] < 2:  # P7: an outer product is glue
                return func(*args, **kwargs)
            want = (*a.shape[:-1], b.shape[-1])
            out = self.supply(name, a, b, bias)
            if tuple(out.shape) != want or out.dtype != a.dtype:
                raise SubstitutionError(f"{func}: supplied {out.dtype} {tuple(out.shape)}, the "
                                        f"op gives {a.dtype} {want}")
            return out.to(a.device)
        return func(*args, **kwargs)
