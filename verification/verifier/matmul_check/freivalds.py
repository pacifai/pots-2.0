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

**Fast path, same bits.** ``_measure_product_reference`` and ``_measure_products_reference``
are the plain formulas. The public functions give the same numbers bit for bit (tested on
the real step and on adversarial inputs) with less work:

- ``|A|``, ``|B|`` and ``|P|`` are written into one reused buffer (:class:`MeasureScratch`)
  with the layout a fresh ``x.abs()`` has, so the matmuls see the same operands. ``|P|`` is
  computed once, for ``|P|·1`` and for the max that ``_safe_norm`` scales by.
- :func:`_norm` returns ``_safe_norm``'s value, skipping the scaled copy where it provably
  changes nothing (see there).

**Test 1 off, for cost only.** ``guard=False`` skips test 1's work (``|A|``, ``|B|``, the
two matvecs and ``|P|·1``) and returns ``ν``, ``‖|P|·1‖`` and ``κ`` as NaN. Test 2's numbers
are the same bits either way. C1 uses it to time check 5 with and without the guard; a judged
run keeps the default.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from verification.verifier.matmul_check.challenges import (
    SIGMA_R,
    challenge_matrices,
    challenge_matrix,
)
from verification.verifier.matmul_check.sizing import e_m

__all__ = ["MeasureScratch", "ProductMeasure", "measure_product", "measure_products"]


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
    return _scaled_norm(x, dim, amax)


def _scaled_norm(x: torch.Tensor, dim: int | None, amax: torch.Tensor,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """``_safe_norm`` given ``amax = max|x|`` over ``dim``; ``x/s`` goes into ``out`` if given.

    ``out`` must be contiguous with ``x``'s shape, and ``x`` contiguous, so ``x/s`` has the
    layout a fresh quotient has.
    """
    ok = torch.isfinite(amax) & (amax > 0)
    _, exp = torch.frexp(torch.where(ok, amax, torch.ones_like(amax)))
    s = torch.ldexp(torch.ones_like(amax), exp - 1)  # max|x| ∈ [s, 2s)
    div = s if dim is None else s.unsqueeze(dim)
    xs = x / div if out is None else torch.div(x, div, out=out)
    scaled = s * torch.linalg.vector_norm(xs, dim=dim)
    if bool(ok.all()):
        return scaled
    return torch.where(ok, scaled, torch.linalg.vector_norm(x, dim=dim))


# Where every vector's entries satisfy these bounds, ``vector_norm(x)`` equals
# ``_safe_norm(x)`` bit for bit. See :func:`_norm`.
_MIN_ENTRY = 2.0 ** -63  # its square, 2^-126, is fp32's smallest normal number
_MAX_SUM = 2.0 ** 124  # the exact sum of squares; rounding can't push it past 2^128


def _plain_is_exact(lo: float, hi: float, length: int) -> bool:
    """True if a vector with ``min|x_i| = lo``, ``max|x_i| = hi`` meets :func:`_norm`'s bounds."""
    return lo >= _MIN_ENTRY * max(1.0, hi) and length * hi * hi <= _MAX_SUM


def _norm(x: torch.Tensor, dim: int | None, absx: torch.Tensor,
          scratch: MeasureScratch | None = None) -> torch.Tensor:
    """``_safe_norm(x, dim)``, bit for bit, given ``absx`` with the values of ``|x|``.

    ``absx`` may differ from ``x.abs()`` in the sign of a zero only. With ``scratch``, a
    scaled copy of a contiguous ``x`` goes into its buffer, so ``absx`` must not live there.

    **When the scaled copy is skipped.** torch's CPU 2-norm sums ``x_i·x_i`` in fp32 and
    takes ``sqrt`` (``NormTwoOps``; a one-entry tensor is its ``|x_0|``). Every step is a
    correctly rounded ``×``, ``+`` or ``sqrt``, and its order depends only on the shape, the
    strides, the alignment and the thread count. ``_safe_norm`` runs the same kernel on
    ``x/s``, a fresh contiguous tensor of ``x``'s shape. If ``x`` is contiguous and 64-byte
    aligned (as every fresh tensor is), both runs take the same steps in the same order, on
    values that differ by the factor ``s = 2^e``. Scaling an operation's inputs by powers of
    two scales its exact result the same way, so it scales the rounded result too, as long
    as both rounded results are normal fp32 numbers (not subnormal, not overflowed). Then
    ``vector_norm(x) = s·vector_norm(x/s)``, the scaled sum's ``sqrt`` times ``s``, exactly.
    Per vector, with ``lo = min|x_i|``, ``hi = max|x_i|`` and ``n`` entries:

    - ``lo ≥ 2^-63·max(1, hi)``. Then ``lo > 0``, and since ``s ≤ hi``, every ``x_i`` and
      ``x_i/s`` is at least ``2^-63`` in magnitude: ``x/s`` is exact, and every square and
      partial sum is at least ``2^-126``, normal. A vector with a zero entry fails this.
    - ``n·hi² ≤ 2^124``. The computed sum is at most the exact one times ``(1 + 2^-24)^n``,
      under ``e^2`` for ``n < 2^25``, so no partial sum reaches ``2^128``. The scaled sums
      are at most ``4n``, and ``s`` times the scaled norm is the unscaled norm, finite.

    A NaN or infinite entry fails the bounds. Any failing vector falls back to the scaled
    computation for the whole call, reusing ``max|x|`` from the same pass that gave ``lo``.
    """
    lo, hi = torch.aminmax(absx) if dim is None else torch.aminmax(absx, dim=dim)
    length = x.numel() if dim is None else x.shape[dim]
    if x.is_contiguous() and x.data_ptr() % 64 == 0 and all(
            _plain_is_exact(a, b, length)
            for a, b in zip(lo.reshape(-1).tolist(), hi.reshape(-1).tolist())):
        return torch.linalg.vector_norm(x, dim=dim)
    out = None
    if scratch is not None and x.is_contiguous():
        out = scratch.buffer(x.numel()).view(x.shape)
    return _scaled_norm(x, dim, hi, out)


class MeasureScratch:
    """One buffer that check 5 reuses for ``|A|``, ``|B|``, ``|P|`` and ``P/s``, and ``1``.

    Each product would otherwise allocate up to four fresh temporaries of its largest operand's
    size, and fresh large blocks page-fault on first touch. The buffer grows to the largest
    size asked for and lives as long as the object; check 5 holds one per step. A view starts
    at the buffer's start, so it has a fresh tensor's 64-byte alignment.
    """

    def __init__(self) -> None:
        self._buf = torch.empty(0, dtype=torch.float32)
        self._ones: dict[int, torch.Tensor] = {}

    def buffer(self, n: int) -> torch.Tensor:
        """The first ``n`` fp32 entries of the buffer, contiguous; earlier views are invalid."""
        if self._buf.numel() < n:
            self._buf = torch.empty(0, dtype=torch.float32)  # free before allocating
            self._buf = torch.empty(n, dtype=torch.float32)
        return self._buf[:n]

    def ones(self, width: int) -> torch.Tensor:
        """``torch.ones(width, 1)``, fp32, never written to."""
        t = self._ones.get(width)
        if t is None:
            t = self._ones[width] = torch.ones(width, 1, dtype=torch.float32)
        return t

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        """``x.abs()`` written into the buffer, with the strides a fresh ``x.abs()`` has.

        A contiguous ``x`` gives a contiguous result, and a transposed contiguous matrix a
        transposed one (a fresh ``abs`` keeps a dense input's strides). Any other layout gets
        a fresh ``x.abs()``.
        """
        if x.is_contiguous():
            return torch.abs(x, out=self.buffer(x.numel()).view(x.shape))
        if x.dim() == 2 and x.t().is_contiguous():
            xt = x.t()
            return torch.abs(xt, out=self.buffer(x.numel()).view(xt.shape)).t()
        return x.abs()


def measure_product(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes, m: int,
                    k: int, eps_in: float, eps_acc: float,
                    scratch: MeasureScratch | None = None, guard: bool = True) -> ProductMeasure:
    """Both tests' numbers for product ``m``, fp32 operands, challenges keyed on root ``h``.

    Every norm is ``_safe_norm``'s, so a finite product entry near ``2e19`` can't overflow
    ``‖P‖_F`` or the residual to ``inf``. The numbers equal
    ``_measure_product_reference``'s bit for bit. ``scratch`` is reused across calls if
    given; ``a``, ``b`` and ``p`` must not live in it. ``guard=False`` skips test 1 (module
    docstring).
    """
    q, width = a.shape[1], b.shape[1]
    sc = MeasureScratch() if scratch is None else scratch
    with torch.no_grad():
        if guard:
            # Test 1, the cancellation guard (P3.c): ν_m ≤ κ_max·‖ |P_m|·1 ‖.
            ones = sc.ones(width)
            b1 = sc.abs(b) @ ones
            nu_vec = sc.abs(a) @ b1
            nu, p_abs1, p_norm = _p_norms(nu_vec, p, ones, sc, None)
        else:
            nu, p_abs1, p_norm = [math.nan], [math.nan], [float(_norm(p, None, sc.abs(p), sc))]
        # Test 2, the normalized residual (P3.a): ‖A(B·r) − P·r‖ ≤ τ·σ_r·e_m·‖P_m‖_F.
        r = challenge_matrix(h, m, k, width)
        d = a @ (b @ r) - p @ r
        res = tuple(_norm(d, 0, d.abs()).tolist())
    return _guarded(_measure(nu[0], p_abs1[0], res, p_norm[0], q, eps_in, eps_acc), guard)


def _guarded(mp: ProductMeasure, guard: bool) -> ProductMeasure:
    """``mp`` as measured, or with ``κ`` set to NaN when test 1 was skipped."""
    return mp if guard else replace(mp, kappa=math.nan)


def _p_norms(nu_vec: torch.Tensor, p: torch.Tensor, ones: torch.Tensor, sc: MeasureScratch,
             n_b: int | None) -> tuple[list[float], list[float], list[float]]:
    """ν, ``‖|P|·1‖`` and ``‖P‖_F`` per product, from ``|A|·(|B|·1)`` and ``P``.

    ``n_b`` is the batch size, or ``None`` for one product. ``|P|`` goes into the buffer
    after ``nu_vec`` is computed, and the buffer then takes ``P/s`` if a scaled copy is
    needed, after ``|P|``'s last use.
    """
    absp = sc.abs(p)
    p1_vec = absp @ ones
    if n_b is None:
        nu = _norm(nu_vec, None, nu_vec)  # entries ≥ 0: |ν_i| = ν_i, up to a zero's sign
        p_abs1 = _norm(p1_vec, None, p1_vec)
        p_norm = _norm(p, None, absp, sc)
        return [float(nu)], [float(p_abs1)], [float(p_norm)]
    nu = _norm(nu_vec.reshape(n_b, -1), 1, nu_vec.reshape(n_b, -1))
    p_abs1 = _norm(p1_vec.reshape(n_b, -1), 1, p1_vec.reshape(n_b, -1))
    p_norm = _norm(p.reshape(n_b, -1), 1, absp.reshape(n_b, -1), sc)
    return nu.tolist(), p_abs1.tolist(), p_norm.tolist()


def measure_products(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes,
                     ms: Sequence[int], k: int, eps_in: float, eps_acc: float,
                     scratch: MeasureScratch | None = None,
                     guard: bool = True) -> list[ProductMeasure]:
    """:func:`measure_product` for a batch: ``p[i] = a[i]·b[i]`` claims product ``ms[i]``.

    ``a``, ``b`` and ``p`` are fp32 ``[B, n, q]``, ``[B, q, w]`` and ``[B, n, w]``. Each member
    gets its own challenges ``r(ms[i], j)`` and its own ``_safe_norm`` scaling, so one large
    member can't change another's numbers. The batched matmuls may round differently from
    one-at-a-time ones, at rounding level only. The numbers equal
    ``_measure_products_reference``'s bit for bit.
    """
    n_b, q, width = a.shape[0], a.shape[2], b.shape[2]
    if len(ms) != n_b or b.shape[0] != n_b or p.shape[0] != n_b:
        raise ValueError(f"batch of {len(ms)} labels for operands {tuple(a.shape)}, "
                         f"{tuple(b.shape)}, {tuple(p.shape)}")
    sc = MeasureScratch() if scratch is None else scratch
    with torch.no_grad():
        if guard:
            ones = sc.ones(width)
            b1 = sc.abs(b) @ ones
            nu_vec = sc.abs(a) @ b1
            nu, p_abs1, p_norm = _p_norms(nu_vec, p, ones, sc, n_b)
        else:
            nu = p_abs1 = [math.nan] * n_b
            p_norm = _norm(p.reshape(n_b, -1), 1, sc.abs(p).reshape(n_b, -1), sc).tolist()
        r = challenge_matrices(h, ms, k, width)
        d = a @ (b @ r) - p @ r
        res = _norm(d, 1, d.abs()).tolist()  # [B, k]
    return [_guarded(_measure(nu[i], p_abs1[i], tuple(res[i]), p_norm[i], q, eps_in, eps_acc),
                     guard) for i in range(n_b)]


def _measure_product_reference(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes,
                               m: int, k: int, eps_in: float,
                               eps_acc: float) -> ProductMeasure:
    """:func:`measure_product` as plain formulas, the reference its tests compare with."""
    q, width = a.shape[1], b.shape[1]
    with torch.no_grad():
        ones = torch.ones(width, 1, dtype=torch.float32)
        nu = float(_safe_norm(a.abs() @ (b.abs() @ ones)))
        p_abs1 = float(_safe_norm(p.abs() @ ones))
        r = challenge_matrix(h, m, k, width)
        res = tuple(_safe_norm(a @ (b @ r) - p @ r, dim=0).tolist())
        p_norm = float(_safe_norm(p))
    return _measure(nu, p_abs1, res, p_norm, q, eps_in, eps_acc)


def _measure_products_reference(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *,
                                h: bytes, ms: Sequence[int], k: int, eps_in: float,
                                eps_acc: float) -> list[ProductMeasure]:
    """:func:`measure_products` as plain formulas, the reference its tests compare with."""
    n_b, q, width = a.shape[0], a.shape[2], b.shape[2]
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
