"""The per-step checks of spec §6, as functions of ``(store, computation, context, bands)``.

Each per-step check returns ``None`` on acceptance or a :class:`Rejection` ``(step, check_id,
detail)``. :data:`DEFAULT_ORDER` is the one evaluation order of every run, ``4 → 7 → 2 → 6a →
5 → 6b`` (S6c, revised at stage 4). The verifier is the same code in every run (S6a); nothing
here has a fault hook.

The checks share one ``StepContext`` (``context.py``). Writing to its ``StepState`` is a
check's only side effect. The tolerances are ``Bands`` (``bands.py``). Check 5's per-product
arithmetic is ``matmul_check/freivalds.py``; this module judges its numbers.

**Byte binding.** Every check after check 2 reads the leaves check 2 hashed, never the store
again: check 2 reads each leaf once, hashes it and keeps the object in a
:class:`CommittedLeaves` reader, guarded by ``_version``. Checks 4 and 7 run before check 2
and read the store themselves, so they record the hashes they saw and check 2 rejects if its
own read of any of those leaves hashes differently. A store that serves one set of bytes to an
early check and another to check 2 is rejected at 2.

**Errors.** Prover data that fails to read, decode, hash or validate raises one of the errors
the ``TranscriptStore`` docstring lists. ``_guard`` turns each into a ``"malformed"``
rejection at the check that read the leaf (``transcript/store.py``, "the rule for A5"). The
guard covers store reads, leaf hashing and validation only. After check 2 the verifier runs
its own code (replay, operands, glue gradients) on validated leaves, so an error there is a
verifier bug and propagates; only a ``TranscriptFormatError`` (a cached leaf mutated in place)
is mapped there. A violated verifier-side precondition raises ``RuntimeError``.

Arithmetic is at the working precision, fp32 (P6). Every band comparison is written as
``not (x <= bound)``, so a NaN that slips through glue rejects instead of passing, and every
quantity a band compares must be finite. Check 5's norms are scaled so a finite leaf can't
overflow them (``freivalds._safe_norm``); an overflow that remains, in a matmul output or in
check 6's ``η·G``, rejects rather than passing ``inf <= inf``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch

from verification.commitment.merkle import DIGEST_SIZE, merkle_root, verify_path
from verification.computation.interface import DeclaredComputation, ProductSpec, Replay
from verification.transcript.reader import TranscriptView
from verification.transcript.store import TranscriptStore
from verification.verifier.bands import Bands, product_class
from verification.verifier.context import (
    _AFTER_COMMIT,
    CommittedLeaves,
    ProductStat,
    Rejection,
    StepContext,
    TensorStat,
    _checked,
    _guard,
    _hash,
    _hashes,
    _hashes_until_error,
    _is_digest,
)
from verification.verifier.matmul_check.freivalds import (
    ProductMeasure,
    measure_product,
    measure_products,
)

__all__ = [
    "DEFAULT_ORDER",
    "check_4_batch_anchor",
    "check_7_chaining",
    "check_2_commitment",
    "check_6a_linear_update",
    "check_5_matmuls",
    "check_6b_glue_update",
    "CHECKS",
]

DEFAULT_ORDER: tuple[str, ...] = ("4", "7", "2", "6a", "5", "6b")


# ---- check 4: batch anchor --------------------------------------------------------------


@_checked
def check_4_batch_anchor(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                         bands: Bands) -> Rejection | None:
    """Each record leaf, with its claimed path, verifies into ``h_D`` at index ``π(t)_i``.

    The tree size is ``|D|`` from the verifier's manifest, never from the store (invariant 7).
    ``b`` has exactly ``n_s`` records by the layout of ``C``, and ``π(t)`` must name as many.
    """
    if len(ctx.indices) != c.n_s:
        raise RuntimeError(f"π({ctx.step}) has {len(ctx.indices)} indices, C declares {c.n_s}")
    for i, d_index in enumerate(ctx.indices):
        with _guard(ctx, "4"):
            h = _hash(c, c.record_index(i), store.leaf(c.record_index(i)))
            path = store.dataset_path(i)
        ctx.state.early_hashes[c.record_index(i)] = h
        if not (isinstance(path, (list, tuple)) and all(_is_digest(p) for p in path)):
            return ctx.reject("4", f"record {i}: dataset path is not a list of "
                                   f"{DIGEST_SIZE}-byte digests", "malformed")
        if not verify_path(h, d_index, ctx.n_records, path, ctx.h_D):
            return ctx.reject("4", f"record {i} does not verify into h_D at index {d_index}")
    return None


# ---- check 7: chaining (check 0 on the first step) --------------------------------------


@_checked
def check_7_chaining(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                     bands: Bands) -> Rejection | None:
    """The ``W_t`` leaf hashes equal the ones the verifier kept from step ``t−1``'s check 2.

    On the first step the kept hashes are the agreed ``W_0``'s, and a mismatch is check 0, the
    base anchor (spec §6 states check 7 for later steps only).

    The leaves are read and validated in order and hashed in parallel (``_hashes_until_error``),
    then judged in order, so the outcome is a leaf-by-leaf loop's: the first leaf in order that
    fails to read, validate or hash, or hashes to the wrong digest, decides. An error at a
    later leaf than a mismatch never shows. A leaf hashes only after later leaves are read, so
    a read that writes in place into an earlier leaf rejects that leaf as malformed: its digest
    may be of the changed bytes.
    """
    cid = ctx.chain_check_id
    if len(ctx.prev_w_hashes) != c.n_w:
        raise RuntimeError(f"{len(ctx.prev_w_hashes)} chained hashes for {c.n_w} weights")
    seen = CommittedLeaves()  # tensor versions at read, to spot in-place writes

    def read() -> Iterator[tuple[int, Any]]:
        for name in c.weight_names:
            obj = store.leaf(c.w_t_index(name))
            seen.add(obj)
            yield c.w_t_index(name), obj

    hashes, error = _hashes_until_error(c, read())
    for k, (name, want, got) in enumerate(zip(c.weight_names, ctx.prev_w_hashes, hashes)):
        i = c.w_t_index(name)
        if seen.changed(k):
            return ctx.reject(cid, f"leaf {i} changed in place during check {cid}", "malformed")
        ctx.state.early_hashes[i] = got
        if got != want:
            what = "the agreed W_0" if cid == "0" else f"W_{{t+1}} of step {ctx.step - 1}"
            return ctx.reject(cid, f"W_t[{name}] differs from {what}")
    if error is not None:
        with _guard(ctx, cid):
            raise error
    return None


# ---- check 2: commitment ----------------------------------------------------------------


@_checked
def check_2_commitment(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                       bands: Bands) -> Rejection | None:
    """Recompute ``h`` over all ``n_leaves`` leaves (count from ``C``) and compare with the claim.

    Hashing validates every leaf's shape, dtype and finiteness, so later checks read leaves
    already known to be well formed. Each leaf is read from the store exactly once here, and
    the objects go to ``ctx.state.leaves`` for every later check, with the root and hashes. A
    leaf that check 4 or 7 already read must hash as it did then.

    Leaves are read and validated in order and hashed in parallel (``_hashes``), so the first
    malformed leaf is the one a leaf-by-leaf loop finds. A leaf hashes only after later leaves
    are read, so a read that writes in place into an earlier leaf rejects as malformed before
    the root is compared: that leaf's digest may already be of the changed bytes.
    """
    leaves = CommittedLeaves()

    def read() -> Iterator[tuple[int, Any]]:
        for i in range(c.n_leaves):  # the count comes from C (invariant 7)
            obj = store.leaf(i)
            leaves.add(obj)
            yield i, obj

    with _guard(ctx, "2"):
        hashes = _hashes(c, read())
        stale = any(leaves.changed(i) for i in range(len(leaves)))
        claimed = store.root
    if stale:
        changed = [i for i in range(len(leaves)) if leaves.changed(i)]
        return ctx.reject("2", f"leaf {changed[0]} changed in place during check 2", "malformed")
    for i, h in sorted(ctx.state.early_hashes.items()):
        if hashes[i] != h:
            return ctx.reject("2", f"leaf {i} hashes differently from the bytes an earlier "
                                   f"check read: the store served two versions")
    root = merkle_root(hashes)
    if not _is_digest(claimed):
        return ctx.reject("2", f"claimed root is not a {DIGEST_SIZE}-byte digest", "malformed")
    if root != claimed:
        return ctx.reject("2", f"recomputed root {root.hex()[:16]}… != claimed "
                               f"{claimed.hex()[:16]}…")
    # A leaf written in place while check 2 ran (by a later store read, or the root getter)
    # no longer has the bytes that were hashed.
    changed = [i for i in range(len(leaves)) if leaves.changed(i)]
    if changed:
        return ctx.reject("2", f"leaf {changed[0]} changed in place during check 2", "malformed")
    ctx.state.root, ctx.state.leaf_hashes, ctx.state.leaves = root, hashes, leaves
    return None


# ---- check 6: update identity (P5) ------------------------------------------------------


UPDATE_CHUNK = 1 << 21
"""Entries per block of check 6's fast path: three scratch buffers of 8 MB in fp32."""


class UpdateScratch:
    """Scratch buffers that check 6's fast path reuses across a check's weights.

    One allocation per check call, not about ten fresh temporaries per weight. Each block
    stays small enough to sit in cache between the passes over it.
    """

    def __init__(self) -> None:
        self._bufs: tuple[torch.Tensor, ...] = ()

    def get(self, n: int, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b = self._bufs
        if not b or b[0].numel() < n or b[0].dtype != like.dtype or b[0].device != like.device:
            size = max(n, UPDATE_CHUNK)
            self._bufs = b = tuple(torch.empty(size, dtype=like.dtype, device=like.device)
                                   for _ in range(3))
        return tuple(x[:n] for x in b)


def _rho_max_of(r: torch.Tensor, scale: torch.Tensor) -> float:
    """``max_i ρ_i`` in float64 over a few entries, exactly as the reference forms ``ρ``."""
    r64 = r.double()
    return float(torch.where(r64 == 0, torch.zeros_like(r64), r64 / scale.double()).max())


def _fast_rho_max(w_t: torch.Tensor, w_next: torch.Tensor, g: torch.Tensor, eta: float,
                  eps_w: float, stop_above: float, scratch: UpdateScratch) -> float | None:
    """The reference's ``ρ_max``, bit for bit, in about ten passes over reused scratch.

    Returns ``None`` when the reference must decide: a non-finite ``R`` or scale, or a
    block whose ``ρ_max`` is not ``≤ stop_above`` (a rejection to come, or ``NaN``).

    **Same R and scale.** Each block computes ``R`` and the scale with the reference's
    operations in the reference's order (``η·G``, ``W_t − η·G``, ``W_{t+1} − ·``, ``|·|``,
    ``|W_t| + |η·G|``, ``ε_W·``), only into reused buffers. So every entry's fp32 ``|R_i|``
    and ``scale_i`` equal the reference's bit for bit.

    **Same ρ_max without float64 on every entry.** The reference forms
    ``ρ_i = fl64(|R_i| / scale_i)`` and takes the max. Rounding is monotone: if the exact
    ratio ``x_i`` is at least ``x_j``, then ``fl32(x_i) ≥ fl32(x_j)`` and
    ``fl64(x_i) ≥ fl64(x_j)``. So the entry with the largest exact ratio has the largest
    fp32 quotient ``q_i = fl32(|R_i| / scale_i)``, perhaps tied with others. The max of
    ``fl64`` over the entries tied at ``max q`` is therefore the reference's ``ρ_max``. The
    block finds them from row maxima: the rows whose max equals ``max q``, then the entries in
    those rows. Ties are kept, since two entries with one fp32 quotient can have different
    float64 ones. Two cases need care:

    - ``R_i = scale_i = 0`` gives ``q_i = NaN``, where the reference sets ``ρ_i = 0``. With
      ``R`` and scale finite, ``0/0`` is the only source of ``NaN``, so it is set to 0.
    - ``max q = 0`` with some ``R_i ≠ 0`` means fp32 underflow of a tiny ratio. Then every
      nonzero ``R_i`` is a candidate. ``R ≡ 0`` gives ``ρ_max = 0`` with no division.

    The max over blocks is exact, so blocking doesn't change the result. A non-finite ``R``
    or scale is caught by the block maxima (``max`` propagates ``NaN``, and both are ``≥ 0``).
    """
    if w_t.numel() == 0:
        return 0.0
    cols = w_t.shape[-1] if w_t.dim() >= 2 else w_t.numel()
    rows = w_t.numel() // cols
    w2, n2, g2 = (x.reshape(rows, cols) for x in (w_t, w_next, g))
    step = max(1, UPDATE_CHUNK // cols)
    best = 0.0
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        w, wn, gg = w2[lo:hi], n2[lo:hi], g2[lo:hi]
        a, r, s = (x.view(hi - lo, cols) for x in scratch.get((hi - lo) * cols, w_t))
        torch.mul(gg, eta, out=a)  # η·G
        torch.sub(w, a, out=r)  # W_t − η·G
        torch.sub(wn, r, out=r)  # R
        r.abs_()
        torch.abs(w, out=s)
        a.abs_()
        s.add_(a)
        s.mul_(eps_w)  # scale
        r_max, s_max = float(r.max()), float(s.max())
        if not (math.isfinite(r_max) and math.isfinite(s_max)):
            return None
        if r_max == 0:
            continue  # ρ ≡ 0 on this block
        torch.div(r, s, out=a)  # q, fp32
        row_max = a.amax(dim=1)
        q_max = float(row_max.max())
        if math.isnan(q_max):  # 0/0 where R_i = scale_i = 0: the reference's ρ_i = 0
            a.nan_to_num_(nan=0.0, posinf=math.inf)
            row_max = a.amax(dim=1)
            q_max = float(row_max.max())
        if q_max == 0:  # every nonzero ratio underflowed fp32
            m = r != 0
            best = max(best, _rho_max_of(r[m], s[m]))
        else:
            rows_at = (row_max == q_max).nonzero().reshape(-1)
            m = a[rows_at] == q_max
            best = max(best, _rho_max_of(r[rows_at][m], s[rows_at][m]))
        if not best <= stop_above:
            return None
    return best


def _update_identity(ctx: StepContext, check_id: str, name: str, w_t: torch.Tensor,
                     w_next: torch.Tensor, g: torch.Tensor, eta: float, tau_w: float,
                     scratch: UpdateScratch | None = None) -> Rejection | None:
    """P5 for one weight, via the fast path when it can decide, else the reference.

    The fast path (:func:`_fast_rho_max`) gives the reference's ``ρ_max`` bit for bit. It
    decides alone only an acceptance: finite ``R`` and scale and, when judging,
    ``ρ_max ≤ τ_W``. Then every entry has ``ρ_i ≤ τ_W`` (finite ``R`` and scale make every
    ``ρ_i`` a number, never ``NaN``), so the reference would accept too. It records the same
    ``TensorStat``. Every other case goes to :func:`_update_identity_reference`, which
    records the stat and words the rejection. Mixed dtypes or shapes also go there.
    """
    if g.shape != w_t.shape:
        raise RuntimeError(f"{name}: gradient shape {tuple(g.shape)} != {tuple(w_t.shape)}")
    if (w_next.shape == w_t.shape and w_t.dtype == w_next.dtype == g.dtype
            and w_t.dtype.is_floating_point and w_t.device == w_next.device == g.device):
        with torch.no_grad():
            rho_max = _fast_rho_max(w_t, w_next, g, eta, ctx.eps_w,
                                    tau_w if ctx.judge else math.inf,
                                    scratch if scratch is not None else UpdateScratch())
        if rho_max is not None:
            ctx.stats.tensors.append(TensorStat(name, check_id, rho_max))
            return None
    return _update_identity_reference(ctx, check_id, name, w_t, w_next, g, eta, tau_w)


def _update_identity_reference(ctx: StepContext, check_id: str, name: str, w_t: torch.Tensor,
                               w_next: torch.Tensor, g: torch.Tensor, eta: float,
                               tau_w: float) -> Rejection | None:
    """P5: reject if any ``|R_i| > τ_W·ε_W·(|W_t,i| + |η·G_i|)``, ``R = W_{t+1} − (W_t − η·G)``.

    The spec's formula written out on whole tensors. :func:`_update_identity` sends here
    every case its fast path doesn't accept, so every check-6 rejection is worded here.

    fp32 throughout (P6). The update is written as ``C`` declares it, plain SGD's
    ``W_t − η·G`` (S8a); its rounding is the honest freedom the floor ``τ_W⁰ = 4`` covers
    (P5a), so nothing depends on reproducing the optimizer's bits.

    A non-finite ``R`` or scale rejects in either mode: an overflow of ``η·G`` or
    ``W_t − η·G`` makes the bound ``inf``, which would pass any ``W_{t+1}``.

    The live test and the freeze-time rejudge (``Verifier.freeze``) compare the same number,
    ``ρ_i = |R_i| / (ε_W·(|W_t,i| + |η·G_i|)) ≤ τ_W``, with ``ρ`` formed in float64 from the
    fp32 ``R`` and scale. An entry on the boundary then passes or fails both alike.
    """
    if g.shape != w_t.shape:
        raise RuntimeError(f"{name}: gradient shape {tuple(g.shape)} != {tuple(w_t.shape)}")
    with torch.no_grad():
        eta_g = eta * g
        # Against the fused fl(W − ηG), this reference's two roundings and the fused one give
        # |R| ≤ 3ε(|W| + |ηG|) ≤ τ_W⁰·ε(…); the outer subtraction is exact by Sterbenz.
        r = (w_next - (w_t - eta_g)).abs()
        scale = ctx.eps_w * (w_t.abs() + eta_g.abs())
        finite = torch.isfinite(r) & torch.isfinite(scale)
        r64 = r.double()
        rho = torch.where(r64 == 0, torch.zeros_like(r64), r64 / scale.double())
    rho_max = float(rho.max()) if rho.numel() else 0.0  # max propagates NaN
    ctx.stats.tensors.append(TensorStat(name, check_id, rho_max))
    if not bool(finite.all()):
        i = int((~finite).reshape(-1).nonzero()[0])
        return ctx.reject(check_id, f"{name}: entry {i} has a non-finite residual or bound "
                                    f"(|R| = {float(r.reshape(-1)[i]):.3e}, scale "
                                    f"{float(scale.reshape(-1)[i]):.3e})")
    if not ctx.judge:
        return None
    bad = ~(rho <= tau_w)
    if bool(bad.any()):
        i = int(bad.reshape(-1).nonzero()[0])
        return ctx.reject(check_id, f"{name}: entry {i} has |R| = {float(r.reshape(-1)[i]):.3e}, "
                                    f"bound {tau_w * float(scale.reshape(-1)[i]):.3e} "
                                    f"(ρ_max = {rho_max:.3g}, τ_W = {tau_w:g})")
    return None


@_checked
def check_6a_linear_update(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                           bands: Bands) -> Rejection | None:
    """Check 6 for every linear weight, from its committed weight-gradient leaf (S6c)."""
    view = TranscriptView(c, ctx.state.committed())
    scratch = UpdateScratch()
    for name, m in c.linear_weights.items():
        with _guard(ctx, "6a", _AFTER_COMMIT):
            w_t, w_next, g = view.w_t(name), view.w_next(name), view.product(m)
        rej = _update_identity(ctx, "6a", name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                               scratch)
        if rej is not None:
            return rej
    return None


# ---- check 5: matmul checks (P3) --------------------------------------------------------


def _member_runs(products: Sequence[ProductSpec]) -> Iterator[Sequence[ProductSpec]]:
    """Split the canonical order into runs: a maximal stretch of consecutive member specs
    (``member`` set) of one layer is one run, and every other spec is a run of one."""
    i = 0
    while i < len(products):
        j = i + 1
        if products[i].member is not None:
            while (j < len(products) and products[j].member is not None
                   and products[j].layer == products[i].layer):
                j += 1
        yield products[i:j]
        i = j


@_checked
def check_5_matmuls(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                    bands: Bands) -> Rejection | None:
    """Both tests of spec §6 check 5 on every product, in canonical order; abort at the first fail.

    The challenges are keyed on ``ctx.state.root``, the root the verifier recomputed in check
    2, never on ``store.root``. Check 2 has shown the two equal, but keying on the verifier's
    own value means the PRF never consumes a prover-supplied byte (spec §5), and check 5
    cannot run before check 2 has. Operands and products come from check 2's leaves (check 3).

    Every norm is taken with ``freivalds._safe_norm``, so a finite product entry near ``2e19``
    can't overflow ``‖P‖_F`` or the residual to ``inf``. Every quantity the two tests compare
    must still be finite, in either mode: a matmul output such as ``A(B·r)`` can itself exceed
    the fp32 maximum, and ``inf <= inf`` would accept an arbitrary forgery.

    **Batching.** The members of a batched product (an attention product's ``(s, h)`` members)
    are many and small, so their fixed per-product cost dominates their arithmetic. A run of
    consecutive member specs of one layer is measured together (:func:`_member_runs`): its
    operands are served one member at a time in canonical order, as for any product, then
    stacked by shape and measured in batched ops, each member with its own challenges and norm
    scaling (``freivalds.measure_products``). The members are then recorded and judged one by
    one in canonical order, so the first failing member rejects with the same message as when
    measured alone. An error while serving a member is raised only after the members before it
    are judged, as it would be one product at a time.
    """
    root = ctx.state.root
    if root is None:
        raise RuntimeError("check 5 needs check 2's recomputed root")
    leaves = ctx.state.committed()
    view = TranscriptView(c, leaves)
    with _guard(ctx, "5", _AFTER_COMMIT), ctx.timed("5.glue"):
        replay = c.replay(leaves)
    ctx.state.replay = replay
    for run in _member_runs(c.products):
        served: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        pending: Exception | None = None
        for spec in run:
            try:
                served.append(_served(c, spec, ctx, replay, view))
            except Exception as e:  # raised after the members before it are judged
                pending = e
                break
        specs = run[:len(served)]
        with ctx.timed("5.measure"):
            measures = _measure_run(specs, served, h=root, k=ctx.k, eps_in=ctx.eps_in,
                                    eps_acc=ctx.eps_acc)
        for spec, mp in zip(specs, measures):
            rej = _judge_product(c, spec, mp, ctx, bands)
            if rej is not None:
                return rej
        if pending is not None:
            raise pending
    return None


def _served(c: DeclaredComputation, spec: ProductSpec, ctx: StepContext, replay: Replay,
            view: TranscriptView) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Product ``spec.m``'s replayed operands and committed leaf, as fp32 ``(A, B, P)``."""
    with _guard(ctx, "5", _AFTER_COMMIT):
        with ctx.timed("5.glue"):
            a, b = replay.operands(spec.m)
        p = view.product(spec.m)
    if (tuple(a.shape), tuple(b.shape)) != (spec.a_shape, spec.b_shape):
        raise RuntimeError(f"{spec.name}: replay operands {tuple(a.shape)}·{tuple(b.shape)} "
                           f"!= declared {spec.a_shape}·{spec.b_shape}")
    return (a.detach().to(torch.float32), b.detach().to(torch.float32),
            p.detach().to(torch.float32))


def _measure_run(specs: Sequence[ProductSpec],
                 served: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *,
                 h: bytes, k: int, eps_in: float, eps_acc: float) -> list[ProductMeasure]:
    """Check 5's numbers for each product of a run, in run order.

    A run of one goes through ``measure_product``. A longer run is split by operand shapes,
    and each group is stacked and measured with ``measure_products``.
    """
    if len(specs) == 1:
        (a, b, p), = served
        return [measure_product(a, b, p, h=h, m=specs[0].m, k=k, eps_in=eps_in,
                                eps_acc=eps_acc)]
    groups: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    for i, spec in enumerate(specs):
        groups.setdefault((spec.a_shape, spec.b_shape), []).append(i)
    out: list[ProductMeasure | None] = [None] * len(specs)
    for idx in groups.values():
        a, b, p = (torch.stack([served[i][x] for i in idx]) for x in range(3))
        for i, mp in zip(idx, measure_products(a, b, p, h=h, ms=[specs[i].m for i in idx], k=k,
                                               eps_in=eps_in, eps_acc=eps_acc)):
            out[i] = mp
    return [mp for mp in out if mp is not None]


def _judge_product(c: DeclaredComputation, spec: ProductSpec, mp: ProductMeasure,
                   ctx: StepContext, bands: Bands) -> Rejection | None:
    """Record product ``spec.m``'s numbers, then judge them: finiteness, test 1, test 2."""
    cls_key = product_class(c, spec)
    nu, p_abs1, kappa, res, unit = mp.nu, mp.p_abs1, mp.kappa, mp.residuals, mp.unit
    normalized = mp.normalized
    ctx.stats.products.append(ProductStat(spec.m, spec.name, cls_key, kappa, normalized))
    values = {"ν": nu, "‖|P|·1‖": p_abs1, "‖P‖_F": mp.p_norm, "band unit": unit,
              **{f"residual j={j}": x for j, x in enumerate(res, start=1)}}
    bad = [key for key, v in values.items() if not math.isfinite(v)]
    if bad:
        return ctx.reject("5", f"P_{spec.m} ({spec.name}): non-finite {', '.join(bad)}")
    if not ctx.judge:
        return None
    kappa_max = bands.kappa_for(cls_key)
    if not nu <= kappa_max * p_abs1:
        return ctx.reject("5", f"P_{spec.m} ({spec.name}): cancellation factor κ = "
                               f"{kappa:.3g} > κ_max = {kappa_max:g} [{cls_key}]")
    for j, x in enumerate(res, start=1):
        if not x <= bands.tau * unit:
            return ctx.reject("5", f"P_{spec.m} ({spec.name}), challenge j={j}: normalized "
                                   f"residual {normalized[j - 1]:.3g} > τ = {bands.tau:g}")
    return None


# ---- check 6b: update identity from replayed glue gradients -----------------------------


@_checked
def check_6b_glue_update(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                         bands: Bands) -> Rejection | None:
    """Check 6 for every glue-gradient weight, from the backward glue check 5 replayed (S6c)."""
    names = c.glue_gradient_weights
    if not names:
        return None
    replay = ctx.state.replay
    if replay is None:
        raise RuntimeError("check 6b needs check 5's replay")
    view = TranscriptView(c, ctx.state.committed())
    with _guard(ctx, "6b", _AFTER_COMMIT), ctx.timed("6b.glue"):
        grads = replay.glue_gradients()
    missing = [n for n in names if n not in grads]
    if missing:
        raise RuntimeError(f"replay returned no glue gradient for {missing}")
    scratch = UpdateScratch()
    for name in names:
        with _guard(ctx, "6b", _AFTER_COMMIT):
            w_t, w_next = view.w_t(name), view.w_next(name)
        g = grads[name].detach().to(torch.float32)
        rej = _update_identity(ctx, "6b", name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                               scratch)
        if rej is not None:
            return rej
    return None


CHECKS: Mapping[str, Callable[[TranscriptStore, DeclaredComputation, StepContext, Bands],
                              Rejection | None]] = MappingProxyType({
    "4": check_4_batch_anchor,
    "7": check_7_chaining,
    "2": check_2_commitment,
    "6a": check_6a_linear_update,
    "5": check_5_matmuls,
    "6b": check_6b_glue_update,
})
