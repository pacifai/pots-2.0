"""Check 5's two tests on one product ``P = A·B``, as numbers (spec §6, P3).

:func:`measure_product` computes, for one product:

- test 1, the cancellation guard (P3.c): ``ν = ‖ |A|·(|B|·1) ‖`` against ``‖ |P|·1 ‖``, whose
  ratio is the cancellation factor ``κ``;
- test 2, the normalized residual (P3.a): ``‖A(B·r_j) − P·r_j‖`` for the ``k`` challenge
  vectors keyed on the step root, over the band unit ``σ_r·e_m·‖P‖_F``.

:func:`measure_products` computes the same numbers for a batch of products of one shape,
each with its own challenges and its own norm scaling, in batched tensor ops. Check 5 uses it
for the members of an attention product, which are many and small.

It judges nothing. ``verifier/checks.py`` compares the numbers with the bands, records them
and rejects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from verification.verifier.matmul_check.challenges import (
    SIGMA_R,
    challenge_matrices,
    challenge_matrix,
)
from verification.verifier.matmul_check.sizing import e_m

__all__ = ["ProductMeasure", "measure_product", "measure_products"]


@dataclass(frozen=True)
class ProductMeasure:
    """Check 5's numbers for one product. Any of them may be non-finite; the caller judges."""

    nu: float  # ‖ |A|·(|B|·1) ‖
    p_abs1: float  # ‖ |P|·1 ‖
    kappa: float  # ν / ‖|P|·1‖
    residuals: tuple[float, ...]  # ‖A(B·r_j) − P·r_j‖, j = 1..k
    p_norm: float  # ‖P‖_F
    unit: float  # σ_r·e_m·‖P‖_F, the band's scale
    normalized: tuple[float, ...]  # residual / unit, the quantity compared with τ


def _safe_norm(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """The 2-norm of ``x`` (over ``dim``, or all of it), scaled so its squares can't overflow.

    ``torch.linalg.vector_norm`` doesn't rescale (pytorch issue #193006): any entry above about
    ``1.8e19`` squares to ``inf`` in fp32, though the norm itself is finite. As in LAPACK's
    nrm2 (Blue), divide by ``s = 2^⌊log₂ max|x|⌋`` and return ``s·‖x/s‖``. Scaling by a power
    of two is exact, so a norm whose squares neither overflowed nor underflowed comes out
    bit-identical, and one whose squares underflowed to 0 comes out right. Where ``max|x|`` is
    0 or not finite, the unscaled norm is returned, and check 5's finiteness guard judges it.
    """
    a = x.abs()
    amax = a.amax() if dim is None else a.amax(dim=dim)
    ok = torch.isfinite(amax) & (amax > 0)
    _, exp = torch.frexp(torch.where(ok, amax, torch.ones_like(amax)))
    s = torch.ldexp(torch.ones_like(amax), exp - 1)  # max|x| ∈ [s, 2s)
    scaled = s * torch.linalg.vector_norm(x / (s if dim is None else s.unsqueeze(dim)), dim=dim)
    if bool(ok.all()):
        return scaled
    return torch.where(ok, scaled, torch.linalg.vector_norm(x, dim=dim))


def measure_product(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes, m: int,
                    k: int, eps_in: float, eps_acc: float) -> ProductMeasure:
    """Both tests' numbers for product ``m``, fp32 operands, challenges keyed on root ``h``.

    Every norm is taken with :func:`_safe_norm`, so a finite product entry near ``2e19`` can't
    overflow ``‖P‖_F`` or the residual to ``inf``.
    """
    q, width = a.shape[1], b.shape[1]
    with torch.no_grad():
        # Test 1, the cancellation guard (P3.c): ν_m ≤ κ_max·‖ |P_m|·1 ‖.
        ones = torch.ones(width, 1, dtype=torch.float32)
        nu = float(_safe_norm(a.abs() @ (b.abs() @ ones)))
        p_abs1 = float(_safe_norm(p.abs() @ ones))
        # Test 2, the normalized residual (P3.a): ‖A(B·r) − P·r‖ ≤ τ·σ_r·e_m·‖P_m‖_F.
        r = challenge_matrix(h, m, k, width)
        res = tuple(_safe_norm(a @ (b @ r) - p @ r, dim=0).tolist())
        p_norm = float(_safe_norm(p))
    return _measure(nu, p_abs1, res, p_norm, q, eps_in, eps_acc)


def measure_products(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes,
                     ms: Sequence[int], k: int, eps_in: float,
                     eps_acc: float) -> list[ProductMeasure]:
    """:func:`measure_product` for a batch: ``p[i] = a[i]·b[i]`` claims product ``ms[i]``.

    ``a``, ``b`` and ``p`` are fp32 ``[B, n, q]``, ``[B, q, w]`` and ``[B, n, w]``. Each member
    gets its own challenges ``r(ms[i], j)`` and its own ``_safe_norm`` scaling, so one large
    member can't change another's numbers. The batched matmuls may round differently from
    one-at-a-time ones, at rounding level only.
    """
    n_b, q, width = a.shape[0], a.shape[2], b.shape[2]
    if len(ms) != n_b or b.shape[0] != n_b or p.shape[0] != n_b:
        raise ValueError(f"batch of {len(ms)} labels for operands {tuple(a.shape)}, "
                         f"{tuple(b.shape)}, {tuple(p.shape)}")
    with torch.no_grad():
        ones = torch.ones(width, 1, dtype=torch.float32)
        nu = _safe_norm((a.abs() @ (b.abs() @ ones)).reshape(n_b, -1), dim=1).tolist()
        p_abs1 = _safe_norm((p.abs() @ ones).reshape(n_b, -1), dim=1).tolist()
        r = challenge_matrices(h, ms, k, width)
        res = _safe_norm(a @ (b @ r) - p @ r, dim=1).tolist()  # [B, k]
        p_norm = _safe_norm(p.reshape(n_b, -1), dim=1).tolist()
    return [_measure(nu[i], p_abs1[i], tuple(res[i]), p_norm[i], q, eps_in, eps_acc)
            for i in range(n_b)]


def _measure(nu: float, p_abs1: float, res: tuple[float, ...], p_norm: float, q: int,
             eps_in: float, eps_acc: float) -> ProductMeasure:
    """κ, the band unit and the normalized residuals, from one product's norms."""
    kappa = nu / p_abs1 if p_abs1 > 0 else (1.0 if nu == 0 else float("inf"))
    unit = SIGMA_R * e_m(q, eps_in, eps_acc) * p_norm
    normalized = tuple(x / unit if unit > 0 else (0.0 if x == 0 else float("inf")) for x in res)
    return ProductMeasure(nu, p_abs1, kappa, res, p_norm, unit, normalized)
