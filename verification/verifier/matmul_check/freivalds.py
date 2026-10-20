"""Check 5's two tests on one product ``P = A·B``, as numbers (spec §6, P3).

:func:`measure_product` computes, for one product:

- test 1, the cancellation guard (P3.c): ``ν = ‖ |A|·(|B|·1) ‖`` against ``‖ |P|·1 ‖``, whose
  ratio is the cancellation factor ``κ``;
- test 2, the normalized residual (P3.a): ``‖A(B·r_j) − P·r_j‖`` for the ``k`` challenge
  vectors keyed on the step root, over the band unit ``σ_r·e_m·‖P‖_F`` (:func:`band_unit`).
  ``e_m`` is F18's split, ``√2·ε_in + √q·ε_acc·max(1, U/‖P‖_F)`` (``sizing.e_m``), with the
  term scale ``U = sqrt(Σ_l ‖A[:,l]‖²·‖B[l,:]‖²)``, the Frobenius norm of the array of every
  product term ``A_il·B_lj``. It is computed as ``sqrt(Σ_i (A²·(B²·1))_i)``, the same two
  matvecs as ``ν`` on the squared operands, with the last sum in float64. Every term is
  non-negative, so the fp32 matvecs lose nothing to cancellation. ``U ≤ ν`` always, so test
  1 bounds it: ``U/‖P‖_F ≤ ν/‖P‖_F ≤ √w·κ``.

A product whose op also adds a bias vector, ``P = A·B + 1·bᵀ``, is the matmul
``[A | 1]·[B ; bᵀ]`` (spec §2, F14). Given ``bias``, both tests use those operands without
building them: ``b·r_j`` is added to every row of ``A(B·r_j)``, ``‖b‖₁`` to every row of
``|A|·(|B|·1)``, ``n·‖b‖²`` (float64) to ``U²``, ``n`` the rows of ``A``, and ``e_m``
contracts over ``q + 1``.

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
- :func:`_norm` returns ``_safe_norm``'s value without reading anything back to the host: it
  always takes the scaled path, which where the scaling is trivial gives the unscaled norm's
  bits (see there).
- ``A²`` and ``B²`` for ``U`` go into the same buffer (:meth:`MeasureScratch.square`), after
  ``|B|·1`` and ``|A|·(|B|·1)`` are taken.
- Squares above fp32's range (an entry above about ``1.8e19``) make ``U`` infinite, and check
  5's finiteness guard rejects. Placeholder: no honest product comes near it, so ``U`` gets
  no scaled path.
- The numbers stay on the operands' device as one float64 row per product
  (:func:`measure_product_row`, :func:`measure_products_rows`; the columns are :data:`ROW`),
  and :func:`finish_row` turns a row read back to the host into a :class:`ProductMeasure`.
  A row holds fp32 values widened exactly and float64 sums, so it carries the bits the
  floats had. Check 5 reads many products' rows in one transfer, so the host waits for the
  device once per batch of rows instead of several times per product.

**Test 1 off, for cost only.** ``guard=False`` skips test 1's work (``|A|``, ``|B|``, the
two matvecs and ``|P|·1``) and returns ``ν``, ``‖|P|·1‖`` and ``κ`` as NaN. ``U`` belongs to
test 2's band and is computed either way, so test 2's numbers are the same bits either way.
C1 uses it to time check 5 with and without the guard; a judged run keeps the default.

:func:`term_scale` is ``U`` alone, as :func:`measure_product` computes it, and
:func:`term_scales` each member's ``U`` of a stacked batch, as :func:`measure_products` computes
it, for callers that need a product's ``e_m`` outside check 5 (the flipped-matmul sweep's
scale). The two can differ in the last bits for the same product: the matvecs over ``B²`` round
by ``B``'s layout, and check 5 stacks a batched product's members into a contiguous batch while
the replay serves some ``B`` (``Kᵀ``) as a transposed view. A member's ``U`` (and ``ν``) can
also depend on the batch's size: on Linux x86 a member measured in a batch of one differs in the
last bits from the same member in its full batch, though not on Apple Accelerate. A caller that
needs check 5's bits measures the member in check 5's whole batch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from verification.verifier.matmul_check.challenges import (
    SIGMA_R,
    challenge_matrices,
    challenge_matrix,
)
from verification.verifier.matmul_check.sizing import product_e_m

__all__ = ["ROW", "MeasureScratch", "ProductMeasure", "band_unit", "finish_row",
           "measure_product", "measure_product_row", "measure_products",
           "measure_products_rows", "row_width", "term_scale", "term_scales"]


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
    u: float = math.nan  # U = sqrt(Σ_l ‖A[:,l]‖²·‖B[l,:]‖²), the term scale (F18)


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


def _scaled_norm(x: torch.Tensor, dim: int | None, amax: torch.Tensor,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """``_safe_norm`` given ``amax = max|x|`` over ``dim``; ``x/s`` goes into ``out`` if given.

    ``out`` must be contiguous with ``x``'s shape, and ``x`` contiguous, so ``x/s`` has the
    layout a fresh quotient has. Where ``max|x|`` is finite and positive, ``s`` is
    ``_safe_norm``'s. Where it is 0, ``s = 1/2`` and both give 0; where it is ``inf`` or NaN,
    ``s = 1`` and both give the unscaled norm, ``inf`` or NaN. So the value is
    ``_safe_norm``'s with no branch on the data and nothing read back to the host.
    """
    _, exp = torch.frexp(torch.nan_to_num(amax, nan=1.0, posinf=1.0))  # frexp(0) = (0, 0)
    s = torch.ldexp(torch.ones_like(amax), exp - 1)  # max|x| ∈ [s, 2s)
    div = s if dim is None else s.unsqueeze(dim)
    xs = x / div if out is None else torch.div(x, div, out=out)
    return s * torch.linalg.vector_norm(xs, dim=dim)


_HOST = torch.device("cpu")  # the scratch's default device; check 5 passes its operands'

def _norm(x: torch.Tensor, dim: int | None, absx: torch.Tensor,
          scratch: MeasureScratch | None = None) -> torch.Tensor:
    """``_safe_norm(x, dim)``, bit for bit, given ``absx`` with the values of ``|x|``.

    ``absx`` may differ from ``x.abs()`` in the sign of a zero only. With ``scratch``, the
    scaled copy of a contiguous ``x`` goes into its buffer, so ``absx`` must not live there
    unless the caller is done with it. The result stays on ``x``'s device: nothing is read
    back to the host.

    Why the bits are ``_safe_norm``'s. The maximum of ``absx`` has the value of
    ``max|x|``, and a zero's sign changes neither that value nor its exponent, so
    :func:`_scaled_norm` scales by ``_safe_norm``'s ``s`` and, as its docstring shows, returns
    ``_safe_norm``'s value with no branch on the data. The quotient ``x/s`` holds the same
    numbers either way. Written into the scratch buffer, it has a fresh tensor's layout:
    contiguous, starting at the buffer's 64-byte aligned start. torch's 2-norm kernel orders
    its sums by shape, strides, alignment and thread count only, so it sums them in the same
    order. A non-contiguous ``x`` gets a fresh quotient, as in ``_safe_norm``.
    """
    amax = absx.amax() if dim is None else absx.amax(dim=dim)
    out = None
    if scratch is not None and x.is_contiguous():
        out = scratch.buffer(x.numel(), x.device).view(x.shape)
    return _scaled_norm(x, dim, amax, out)


class MeasureScratch:
    """One buffer that check 5 reuses for ``|A|``, ``|B|``, ``|P|`` and ``P/s``, and ``1``.

    Each product would otherwise allocate up to four fresh temporaries of its largest operand's
    size, and fresh large blocks page-fault on first touch. The buffer grows to the largest
    size asked for and lives as long as the object; check 5 holds one per step. A view starts
    at the buffer's start, so it has a fresh tensor's 64-byte alignment. The buffer sits on
    the device of the tensors it serves (the operands', F4a) and moves when they do.
    """

    def __init__(self) -> None:
        self._buf = torch.empty(0, dtype=torch.float32)
        self._ones: dict[tuple[int, torch.device], torch.Tensor] = {}

    def buffer(self, n: int, device: torch.device = _HOST) -> torch.Tensor:
        """The first ``n`` fp32 entries of the buffer on ``device``, contiguous; earlier views
        are invalid."""
        if self._buf.numel() < n or self._buf.device != device:
            self._buf = torch.empty(0, dtype=torch.float32)  # free before allocating
            self._buf = torch.empty(n, dtype=torch.float32, device=device)
        return self._buf[:n]

    def ones(self, width: int, device: torch.device = _HOST) -> torch.Tensor:
        """``torch.ones(width, 1)``, fp32, on ``device``, never written to."""
        t = self._ones.get((width, device))
        if t is None:
            t = self._ones[width, device] = torch.ones(width, 1, dtype=torch.float32,
                                                       device=device)
        return t

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        """``x.abs()`` written into the buffer, with the strides a fresh ``x.abs()`` has.

        A contiguous ``x`` gives a contiguous result, and a transposed contiguous matrix a
        transposed one (a fresh ``abs`` keeps a dense input's strides). Any other layout gets
        a fresh ``x.abs()``.
        """
        return self._unary(torch.abs, x)

    def square(self, x: torch.Tensor) -> torch.Tensor:
        """``x.square()`` written into the buffer, laid out as :meth:`abs` lays out ``|x|``."""
        return self._unary(torch.square, x)

    def _unary(self, op: Callable[..., torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if x.is_contiguous():
            return op(x, out=self.buffer(x.numel(), x.device).view(x.shape))
        if x.dim() == 2 and x.t().is_contiguous():
            xt = x.t()
            return op(xt, out=self.buffer(x.numel(), x.device).view(xt.shape)).t()
        return op(x)


def _term_sq(a: torch.Tensor, b: torch.Tensor, ones: torch.Tensor,
             sc: MeasureScratch) -> torch.Tensor:
    """``A²·(B²·1)`` in fp32: row ``i`` is ``Σ_l A_il²·‖B[l,:]‖²``, so the rows sum to ``U²``.

    ``B²`` and then ``A²`` go into the buffer, so the caller must be done with what it held.
    """
    b2 = sc.square(b) @ ones
    return sc.square(a) @ b2


def _bias_sq(bias: torch.Tensor) -> torch.Tensor:
    """``‖b‖²`` in float64, on ``bias``'s device. ``[A | 1]·[B ; bᵀ]``'s ones column adds
    ``n·‖b‖²`` to ``U²`` (:func:`finish_row`)."""
    return bias.double().square().sum()


def term_scale(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> float:
    """``U`` of ``A·B`` (``[A | 1]·[B ; biasᵀ]`` given ``bias``), fp32 operands, the same bits
    as :func:`measure_product`'s."""
    sc = MeasureScratch()
    with torch.no_grad():
        u2 = float(_term_sq(a, b, sc.ones(b.shape[1], b.device), sc).double().sum())
        if bias is not None:
            u2 += a.shape[0] * float(_bias_sq(bias))
    return math.sqrt(u2)


def term_scales(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """``U`` of each member ``a[i]·b[i]`` of a stacked batch (fp32 ``[B, n, q]`` and
    ``[B, q, w]``), the same bits as :func:`measure_products`'s."""
    sc = MeasureScratch()
    with torch.no_grad():
        u2 = _term_sq(a, b, sc.ones(b.shape[2], b.device), sc).double().reshape(a.shape[0],
                                                                                 -1).sum(1)
    return [math.sqrt(x) for x in u2.tolist()]


ROW = ("nu", "p_abs1", "p_norm", "u2_terms", "bias_sq", "residuals")
"""The columns of a product's float64 row: ``ν``, ``‖|P|·1‖``, ``‖P‖_F``, the terms' share
of ``U²`` (``Σ_i (A²·(B²·1))_i``), ``‖b‖²`` (0 without a bias), then the ``k`` residuals
``‖A(B·r_j) − P·r_j‖``. ``ν`` and ``‖|P|·1‖`` are NaN when test 1 is skipped."""


def row_width(k: int) -> int:
    """The length of a row (:data:`ROW`) at ``k`` challenges."""
    return len(ROW) - 1 + k


def measure_product(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes, m: int,
                    k: int, eps_in: float, eps_acc: float,
                    scratch: MeasureScratch | None = None, guard: bool = True,
                    bias: torch.Tensor | None = None) -> ProductMeasure:
    """Both tests' numbers for product ``m``, fp32 operands, challenges keyed on root ``h``.

    Every norm is ``_safe_norm``'s, so a finite product entry near ``2e19`` can't overflow
    ``‖P‖_F`` or the residual to ``inf``. The numbers equal
    ``_measure_product_reference``'s bit for bit. ``scratch`` is reused across calls if
    given; ``a``, ``b``, ``p`` and ``bias`` must not live in it. ``guard=False`` skips test 1
    (module docstring).

    ``bias``, a vector of ``P``'s width, makes the checked matmul ``[A | 1]·[B ; biasᵀ]``
    (F14): ``ν`` gains ``‖bias‖₁`` in every row, ``A(B·r)`` gains ``biasᵀ·r`` in every row,
    ``U²`` gains ``n·‖bias‖²``, and ``e_m`` contracts over ``q + 1``. Without it the
    operations are those of an unbiased product.

    It is :func:`measure_product_row`, read back, then :func:`finish_row`.
    """
    row = measure_product_row(a, b, p, h=h, m=m, k=k, scratch=scratch, guard=guard, bias=bias)
    return finish_row(row.tolist(), q=a.shape[1] + (bias is not None), n=a.shape[0],
                      eps_in=eps_in, eps_acc=eps_acc, guard=guard, biased=bias is not None)


def measure_product_row(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes, m: int,
                        k: int, scratch: MeasureScratch | None = None, guard: bool = True,
                        bias: torch.Tensor | None = None) -> torch.Tensor:
    """:func:`measure_product`'s numbers as a float64 row (:data:`ROW`) on the operands'
    device, with nothing read back to the host."""
    width = b.shape[1]
    sc = MeasureScratch() if scratch is None else scratch
    with torch.no_grad():
        ones = sc.ones(width, b.device)
        if guard:
            # Test 1, the cancellation guard (P3.c): ν_m ≤ κ_max·‖ |P_m|·1 ‖.
            b1 = sc.abs(b) @ ones
            nu_vec = sc.abs(a) @ b1
            if bias is not None:
                nu_vec += bias.abs() @ ones
        # Test 2's term scale U (F18), after ν is done with the buffer.
        u2 = _term_sq(a, b, ones, sc).double().sum()
        bias_sq = _bias_sq(bias) if bias is not None else torch.zeros_like(u2)
        if guard:
            norms = _p_norms(nu_vec, p, ones, sc, None)
        else:
            p_norm = _norm(p, None, sc.abs(p), sc)
            nan = torch.full_like(p_norm, math.nan)
            norms = torch.stack((nan, nan, p_norm))
        # Test 2, the normalized residual (P3.a): ‖A(B·r) − P·r‖ ≤ τ·σ_r·e_m·‖P_m‖_F.
        r = challenge_matrix(h, m, k, width).to(b.device)
        ab_r = a @ (b @ r)
        if bias is not None:
            ab_r += bias @ r
        d = ab_r - p @ r
        res = _norm(d, 0, d.abs())
        return torch.cat((norms.double(), torch.stack((u2, bias_sq)), res.double()))


def finish_row(row: Sequence[float], *, q: int, n: int, eps_in: float, eps_acc: float,
               guard: bool = True, biased: bool = False) -> ProductMeasure:
    """The :class:`ProductMeasure` of one product from its row (:data:`ROW`), on the host.

    ``q`` is the checked matmul's contraction length (``+ 1`` with a bias) and ``n`` the rows
    of ``A``. A biased product's ``U²`` adds ``n·‖b‖²`` to the terms' share, in float64.
    """
    nu, p_abs1, p_norm, u2, bias_sq, *res = row
    if biased:
        u2 += n * bias_sq
    return _guarded(_measure(nu, p_abs1, tuple(res), p_norm, q, eps_in, eps_acc,
                             math.sqrt(u2)), guard)


def _guarded(mp: ProductMeasure, guard: bool) -> ProductMeasure:
    """``mp`` as measured, or with ``κ`` set to NaN when test 1 was skipped."""
    return mp if guard else replace(mp, kappa=math.nan)


def _p_norms(nu_vec: torch.Tensor, p: torch.Tensor, ones: torch.Tensor, sc: MeasureScratch,
             n_b: int | None) -> torch.Tensor:
    """ν, ``‖|P|·1‖`` and ``‖P‖_F`` from ``|A|·(|B|·1)`` and ``P``, fp32: ``[3]`` for one
    product (``n_b`` ``None``), ``[n_b, 3]`` for a batch of ``n_b``.

    ``|P|`` goes into the buffer after ``nu_vec`` is computed, and the buffer then takes
    ``P/s`` after ``|P|``'s last use.
    """
    absp = sc.abs(p)
    p1_vec = absp @ ones
    if n_b is None:
        nu = _norm(nu_vec, None, nu_vec)  # entries ≥ 0: |ν_i| = ν_i, up to a zero's sign
        p_abs1 = _norm(p1_vec, None, p1_vec)
        p_norm = _norm(p, None, absp, sc)
        return torch.stack((nu, p_abs1, p_norm))
    nu = _norm(nu_vec.reshape(n_b, -1), 1, nu_vec.reshape(n_b, -1))
    p_abs1 = _norm(p1_vec.reshape(n_b, -1), 1, p1_vec.reshape(n_b, -1))
    p_norm = _norm(p.reshape(n_b, -1), 1, absp.reshape(n_b, -1), sc)
    return torch.stack((nu, p_abs1, p_norm), 1)


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

    It is :func:`measure_products_rows`, read back, then :func:`finish_row` per member.
    """
    rows = measure_products_rows(a, b, p, h=h, ms=ms, k=k, scratch=scratch, guard=guard)
    q, n = a.shape[2], a.shape[1]
    return [finish_row(row, q=q, n=n, eps_in=eps_in, eps_acc=eps_acc, guard=guard)
            for row in rows.tolist()]


def measure_products_rows(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes,
                          ms: Sequence[int], k: int, scratch: MeasureScratch | None = None,
                          guard: bool = True) -> torch.Tensor:
    """:func:`measure_products`'s numbers as float64 rows ``[B, ·]`` (:data:`ROW`), one per
    member, on the operands' device, with nothing read back to the host."""
    n_b, width = a.shape[0], b.shape[2]
    if len(ms) != n_b or b.shape[0] != n_b or p.shape[0] != n_b:
        raise ValueError(f"batch of {len(ms)} labels for operands {tuple(a.shape)}, "
                         f"{tuple(b.shape)}, {tuple(p.shape)}")
    sc = MeasureScratch() if scratch is None else scratch
    with torch.no_grad():
        ones = sc.ones(width, b.device)
        if guard:
            b1 = sc.abs(b) @ ones
            nu_vec = sc.abs(a) @ b1
        u2 = _term_sq(a, b, ones, sc).double().reshape(n_b, -1).sum(1)
        if guard:
            norms = _p_norms(nu_vec, p, ones, sc, n_b)
        else:
            p_norm = _norm(p.reshape(n_b, -1), 1, sc.abs(p).reshape(n_b, -1), sc)
            nan = torch.full_like(p_norm, math.nan)
            norms = torch.stack((nan, nan, p_norm), 1)
        r = challenge_matrices(h, ms, k, width).to(b.device)
        d = a @ (b @ r) - p @ r
        res = _norm(d, 1, d.abs())  # [B, k]
        return torch.cat((norms.double(), torch.stack((u2, torch.zeros_like(u2)), 1),
                          res.double()), 1)


def _measure_product_reference(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, h: bytes,
                               m: int, k: int, eps_in: float, eps_acc: float,
                               bias: torch.Tensor | None = None) -> ProductMeasure:
    """:func:`measure_product` as plain formulas, the reference its tests compare with."""
    q, width = a.shape[1] + (bias is not None), b.shape[1]
    with torch.no_grad():
        ones = torch.ones(width, 1, dtype=torch.float32, device=b.device)
        r = challenge_matrix(h, m, k, width).to(b.device)
        nu_vec, ab_r = a.abs() @ (b.abs() @ ones), a @ (b @ r)
        u2 = float((a.square() @ (b.square() @ ones)).double().sum())
        if bias is not None:  # [A | 1]·[B ; biasᵀ], the ones column's terms added last (F14)
            nu_vec, ab_r = nu_vec + bias.abs() @ ones, ab_r + bias @ r
            u2 += a.shape[0] * float(bias.double().square().sum())
        nu = float(_safe_norm(nu_vec))
        p_abs1 = float(_safe_norm(p.abs() @ ones))
        res = tuple(_safe_norm(ab_r - p @ r, dim=0).tolist())
        p_norm = float(_safe_norm(p))
    return _measure(nu, p_abs1, res, p_norm, q, eps_in, eps_acc, math.sqrt(u2))


def _measure_products_reference(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *,
                                h: bytes, ms: Sequence[int], k: int, eps_in: float,
                                eps_acc: float) -> list[ProductMeasure]:
    """:func:`measure_products` as plain formulas, the reference its tests compare with."""
    n_b, q, width = a.shape[0], a.shape[2], b.shape[2]
    with torch.no_grad():
        ones = torch.ones(width, 1, dtype=torch.float32, device=b.device)
        nu = _safe_norm((a.abs() @ (b.abs() @ ones)).reshape(n_b, -1), dim=1).tolist()
        u2 = (a.square() @ (b.square() @ ones)).double().reshape(n_b, -1).sum(1).tolist()
        p_abs1 = _safe_norm((p.abs() @ ones).reshape(n_b, -1), dim=1).tolist()
        r = challenge_matrices(h, ms, k, width).to(b.device)
        res = _safe_norm(a @ (b @ r) - p @ r, dim=1).tolist()  # [B, k]
        p_norm = _safe_norm(p.reshape(n_b, -1), dim=1).tolist()
    return [_measure(nu[i], p_abs1[i], tuple(res[i]), p_norm[i], q, eps_in, eps_acc,
                     math.sqrt(u2[i])) for i in range(n_b)]


def band_unit(q: int, eps_in: float, eps_acc: float, p_norm: float, u: float) -> float:
    """``σ_r·e_m·‖P‖_F``, test 2's band unit, with F18's split ``e_m`` (``sizing.e_m``).

    It is ``σ_r·(√2·ε_in·‖P‖_F + √q·ε_acc·max(‖P‖_F, U))``. Where ``U ≤ ‖P‖_F`` it is the
    pre-F18 unit bit for bit. With ``‖P‖_F = 0`` it is ``σ_r·√q·ε_acc·U``, the accumulation
    term alone (the relative ``e_m`` is then unbounded). NaN in ``‖P‖_F`` or ``U`` gives NaN.
    """
    if p_norm == 0:
        return SIGMA_R * math.sqrt(q) * eps_acc * u
    return SIGMA_R * product_e_m(q, eps_in, eps_acc, p_norm, u) * p_norm


def _measure(nu: float, p_abs1: float, res: tuple[float, ...], p_norm: float, q: int,
             eps_in: float, eps_acc: float, u: float) -> ProductMeasure:
    """κ, the band unit and the normalized residuals, from one product's norms."""
    kappa = nu / p_abs1 if p_abs1 > 0 else (1.0 if nu == 0 else float("inf"))
    unit = band_unit(q, eps_in, eps_acc, p_norm, u)
    normalized = tuple(x / unit if unit > 0 else (0.0 if x == 0 else float("inf")) for x in res)
    return ProductMeasure(nu, p_abs1, kappa, res, p_norm, unit, normalized, u)
