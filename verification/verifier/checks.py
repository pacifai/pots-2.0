"""The per-step checks of spec §6, as functions of ``(store, computation, context, bands)``.

Each per-step check returns ``None`` on acceptance or a :class:`Rejection` ``(step, check_id,
detail)``. :data:`DEFAULT_ORDER` is the one evaluation order of every run, ``4 → 7 → 2 → 6a →
5 → 6b → 2.root`` (S6c, revised at stage 4 and by F15a in ``DECISIONS_FULL_SCALE.md``, task
C7). The verifier is the same code in every run (S6a); nothing here has a fault hook.

**Check 2 in two parts (F15a).** Each entry of :data:`CHECKS` is a slot, and a slot id is a
protocol check id, optionally with a ``.part`` suffix (:func:`protocol_id`). Check 2 has two
slots, and both reject as check ``"2"``. Slot ``"2"`` reads every leaf once, hashes and
validates it, binds it to what checks 4 and 7 saw, and validates the claimed root ``h`` as a
digest. Slot ``"2.root"`` runs last and compares the recomputed root with ``h``. Check 5 draws
its challenges from the claimed ``h``. The accept condition is unchanged: a step still passes
only if ``root(leaves) = h`` and every check passes with challenges from ``h``. Drawing from the
claim gives the prover nothing, since it can always compute challenges from ``h`` itself. A
streaming store can hash leaves during the checks' own reads and reuse ``"2.root"`` as is.

The checks share one ``StepContext`` (``context.py``). Writing to its ``StepState`` is a
check's only side effect. The tolerances are ``Bands`` (``bands.py``). Check 5's per-product
arithmetic is ``matmul_check/freivalds.py``; this module judges its numbers.

**Byte binding.** Every check after check 2's read uses the leaves that read hashed, never the
store again: check 2 reads each leaf once, hashes it and keeps the object in a
:class:`CommittedLeaves` reader, guarded by ``_version``. Checks 4 and 7 run before check 2
and read the store themselves, so they record the hashes they saw and check 2 rejects if its
own read of any of those leaves hashes differently. A store that serves one set of bytes to an
early check and another to check 2 is rejected at 2, before 6a.

**Errors.** Prover data that fails to read, decode, hash or validate raises one of the errors
the ``TranscriptStore`` docstring lists. ``_guard`` turns each into a ``"malformed"``
rejection at the check that read the leaf (``transcript/store.py``, "the rule for A5"). The
guard covers store reads, leaf hashing and validation only. After check 2's read the verifier
runs its own code (replay, operands, glue gradients) on validated leaves, so an error there is
a verifier bug and propagates; only a ``TranscriptFormatError`` (a cached leaf mutated in place)
is mapped there. A violated verifier-side precondition raises ``RuntimeError``. The record pass
after a rejection (:func:`record_rest`) follows the same rule, and the driver catches what it
raises, so the rejection stands.

Arithmetic is at the working precision, fp32 (P6). Every band comparison is written as
``not (x <= bound)``, so a NaN that slips through glue rejects instead of passing, and every
quantity a band compares must be finite. Check 5's norms are scaled so a finite leaf can't
overflow them (``freivalds._safe_norm``); an overflow that remains, in a matmul output or in
check 6's ``η·G``, rejects rather than passing ``inf <= inf``.

**Profiling stamps.** Check 5 marks its parts with ``ctx.stamp`` (``c5.run``, ``c5.glue``,
``c5.serve``, ``c5.copy``, ``c5.measure``, ``c5.judge``; ``runs/metrics.py``) and
:func:`check5_layout` says what each run, group and window holds. The serving that
:func:`_measure_to_failure` redoes after an error carries the run's id and the group's id,
so its stamps repeat the ids of the normal serving; the layout describes the normal grouping only.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from verification.commitment.merkle import DIGEST_SIZE, merkle_root, verify_path
from verification.computation.interface import (
    HOST,
    DeclaredComputation,
    ProductSpec,
    Replay,
    chain_gradient,
)
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
    _Rejected,
)
from verification.verifier.matmul_check.freivalds import (
    MeasureScratch,
    ProductMeasure,
    finish_row,
    measure_product_row,
    measure_products_rows,
)

__all__ = [
    "DEFAULT_ORDER",
    "check_4_batch_anchor",
    "check_7_chaining",
    "check_2_commitment",
    "check_2_root",
    "check_6a_linear_update",
    "check_5_matmuls",
    "check5_layout",
    "check_6b_glue_update",
    "CHECKS",
    "ROOT_SLOT",
    "RECORDED_SLOTS",
    "RECORD_COMPLETE",
    "RECORD_MALFORMED",
    "RECORD_ERROR",
    "RECORD_NOT_RUN",
    "RECORD_STATUSES",
    "protocol_id",
    "record_rest",
]

ROOT_SLOT = "2.root"
"""Check 2's root comparison, the step's last slot; it rejects as check ``"2"`` (F15a)."""

DEFAULT_ORDER: tuple[str, ...] = ("4", "7", "2", "6a", "5", "6b", ROOT_SLOT)


def protocol_id(slot: str) -> str:
    """The protocol check id a slot's rejections carry: ``"2.root"`` → ``"2"``."""
    return slot.partition(".")[0]


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
    """Check 2's read: hash all ``n_leaves`` leaves (count from ``C``) and validate the claimed
    ``h``. The root comparison is :func:`check_2_root`, at the end of the step (F15a).

    Hashing validates every leaf's shape, dtype and finiteness, so later checks read leaves
    already known to be well formed. Each leaf is read from the store exactly once here, and
    the objects go to ``ctx.state.leaves`` for every later check, with the hashes. A leaf that
    check 4 or 7 already read must hash as it did then. The claimed root is read once, and a
    non-digest is malformed here, before check 5 keys on it.

    Leaves are read and validated in order and hashed in parallel (``_hashes``), so the first
    malformed leaf is the one a leaf-by-leaf loop finds. A leaf hashes only after later leaves
    are read, so a read that writes in place into an earlier leaf rejects as malformed: that
    leaf's digest may already be of the changed bytes.
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
    if not _is_digest(claimed):
        return ctx.reject("2", f"claimed root is not a {DIGEST_SIZE}-byte digest", "malformed")
    # A leaf written in place while check 2 ran (by a later store read, or the root getter)
    # no longer has the bytes that were hashed.
    changed = [i for i in range(len(leaves)) if leaves.changed(i)]
    if changed:
        return ctx.reject("2", f"leaf {changed[0]} changed in place during check 2", "malformed")
    ctx.state.claimed_root, ctx.state.leaf_hashes, ctx.state.leaves = claimed, hashes, leaves
    return None


@_checked
def check_2_root(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                 bands: Bands) -> Rejection | None:
    """Check 2's root comparison, the step's last slot: the root over the leaf hashes check 2's
    read took must equal the claimed ``h`` that check 5 keyed on (F15a).

    A mismatch rejects as check ``"2"``, whatever the other checks said; an earlier rejection
    stands, since the step stops at the first. It reads nothing from the store, so a store that
    hashes leaves during the checks' own reads can reuse it.
    """
    hashes, claimed = ctx.state.leaf_hashes, ctx.state.claimed_root
    if hashes is None or claimed is None:
        raise RuntimeError("check 2's root comparison needs the hashes and claimed root "
                           "of check 2's read")
    root = ctx.state.recomputed_root = merkle_root(hashes)
    if root != claimed:
        return ctx.reject("2", f"recomputed root {root.hex()[:16]}… != claimed "
                               f"{claimed.hex()[:16]}…")
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
    for name in c.linear_weights:
        with _guard(ctx, "6a", _AFTER_COMMIT):
            w_t, w_next, g = view.w_t(name), view.w_next(name), _linear_gradient(c, view, name)
        rej = _update_identity(ctx, "6a", name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                               scratch)
        if rej is not None:
            return rej
    return None


def _linear_gradient(c: DeclaredComputation, view: TranscriptView, name: str) -> torch.Tensor:
    """Check 6a's ``G`` for linear weight ``name``: its committed weight-gradient leaves, each
    widened exactly to the weights' format (F6b: a bf16 leaf becomes fp32; an fp32 leaf is
    unchanged), as 6b's gradients are, and summed in order as training sums them
    (:func:`chain_gradient`; one leaf unless the weight's product runs in row blocks). Check 6a
    and its record pass both take ``G`` here."""
    return chain_gradient((view.product(m) for m in c.linear_weights[name]), c.weight_dtype)


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


_GROUP_BYTES = 1 << 32
"""The most fp32 bytes of ``A``, ``B`` and ``P`` that one stacked group of members holds
(:func:`_measure_groups`). Check 5 measures a group as soon as it is served and keeps only its
rows (:func:`_measure_served`), so this bounds the operands it holds at once; the measuring
adds about as much again in temporaries. A Llama-3.2-1B layer's largest group at ``n = 128``
(12,288 members, 1.5 GiB) stays under it; at ``n = 1024`` it cuts each group into chunks."""

_JUDGE_WINDOW = 1024
"""Check 5 reads its numbers back and judges them once at least this many products are
measured, at the end of a run. A read makes the host wait for the device, so one read per
window replaces several per product."""


def _measure_groups(specs: Sequence[ProductSpec]) -> list[list[int]]:
    """How check 5 stacks a run's products: lists of positions in ``specs``.

    A run of one is its own group. A longer run (attention members) is split by declared
    operand shapes, in order of first appearance, each group in run order, and a group is cut
    into consecutive chunks of at most :data:`_GROUP_BYTES`. It depends on the declared
    computation only, never on prover data. The planted-error sweep measures a member in its
    group (``flipped_matmul_sweep.batch_members``), so it sees check 5's bits.
    """
    if len(specs) <= 1:
        return [list(range(len(specs)))] if specs else []
    by_shape: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    for i, spec in enumerate(specs):
        by_shape.setdefault((spec.a_shape, spec.b_shape), []).append(i)
    groups = []
    for ((n, q), (_, w)), idx in by_shape.items():
        size = max(1, _GROUP_BYTES // (4 * (n * q + q * w + n * w)))
        groups += [idx[i:i + size] for i in range(0, len(idx), size)]
    return groups


_NO_STAMP = nullcontext()


def check5_layout(c: DeclaredComputation) -> list[dict[str, Any]]:
    """What check 5's profiling stamps index, one entry per group of the normal serving: the run
    (:func:`_member_runs`), the group, numbered across the step's runs as
    ``check_5_matmuls`` stamps it (0 ... ``n_groups`` - 1, each once), and the judging window
    (:data:`_JUDGE_WINDOW`) it falls in, replayed by the rules ``check_5_matmuls`` applies, from
    the declared computation alone.

    An entry holds ``run``, ``group``, ``window``, ``kind`` (``"dense"`` for a run of one,
    ``"members"``), ``first`` (the group's first product's ``m``), ``members``, the declared
    ``a_shape`` and ``b_shape``, ``operand_bytes`` (``A`` and ``B`` as fp32, the group's total),
    ``product_bytes`` (``P`` in the stored dtype, ``product_dtype``) and ``served_bytes`` (``P`` in
    fp32). The serving after an error (:func:`_measure_to_failure`) is not described."""
    return _layout(c)[0]


def _layout(c: DeclaredComputation) -> tuple[list[dict[str, Any]], int]:
    """:func:`check5_layout`'s entries and ``n_windows``, the number of judge calls
    ``check_5_matmuls`` makes on a pass with no error: a window ends after the run that fills
    it, and the last is judged even when empty (so it is counted when a run closes the one
    before it)."""
    out: list[dict[str, Any]] = []
    window = pending = 0
    item = torch.empty(0, dtype=c.product_dtype).element_size()
    for i, run in enumerate(_member_runs(c.products)):
        for idx in _measure_groups(run):
            specs = [run[j] for j in idx]
            (n, q), (_, w) = specs[0].a_shape, specs[0].b_shape
            out.append({
                "run": i, "group": len(out), "window": window,
                "kind": "dense" if len(run) == 1 else "members", "first": specs[0].m,
                "members": len(specs), "a_shape": list(specs[0].a_shape),
                "b_shape": list(specs[0].b_shape),
                "operand_bytes": 4 * len(specs) * (n * q + q * w),
                "product_bytes": item * len(specs) * n * w,
                "product_dtype": str(c.product_dtype).removeprefix("torch."),
                "served_bytes": 4 * len(specs) * n * w})
        pending += len(run)
        if pending >= _JUDGE_WINDOW:
            window, pending = window + 1, 0
    return out, window + 1


@_checked
def check_5_matmuls(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                    bands: Bands) -> Rejection | None:
    """Both tests of spec §6 check 5 on every product, in canonical order; abort at the first fail.

    The challenges are keyed on ``ctx.state.claimed_root``, the prover's ``h`` as check 2's
    read validated it, not on a root the verifier recomputed (F15a). Check 2's root comparison
    at the end of the step shows ``h`` is the root of the leaves, so an accepted step still has
    challenges from the committed root. Check 5 cannot run before check 2's read has.
    Operands and products come from check 2's leaves (check 3). A product whose spec names a
    bias is checked as ``[A | 1]·[B ; bᵀ]`` (F14), ``b`` being that bias's committed ``W_t``
    leaf, never a copy the prover sends with the product.

    Every norm is taken with ``freivalds._safe_norm``, so a finite product entry near ``2e19``
    can't overflow ``‖P‖_F`` or the residual to ``inf``. Every quantity the two tests compare
    must still be finite, in either mode: a matmul output such as ``A(B·r)`` can itself exceed
    the fp32 maximum, and ``inf <= inf`` would accept an arbitrary forgery.

    **Batching.** A product's measuring is O(n²) arithmetic, little next to the fixed cost of
    launching its ops and waiting for their results, so check 5 batches in two ways and keeps
    the order of judging:

    - The members of a batched product (an attention product's ``(s, h)`` members) are many
      and small. A run of consecutive member specs of one layer (:func:`_member_runs`) is
      served and measured in groups (:func:`_measure_groups`), one group at a time
      (:func:`_measure_served`): the replay serves the group's operands stacked
      (``Replay.operand_group``), its ``P`` leaves move to the device in one copy, and the
      group is measured in batched ops, each member with its own challenges and norm scaling
      (``freivalds.measure_products_rows``). Only its rows are kept.
    - The numbers stay on the device as one row per product until at least
      :data:`_JUDGE_WINDOW` products are measured; then one read brings the window's rows to
      the host (:func:`_finish`), and its products are recorded and judged one by one in
      canonical order. So the first failing product rejects with the same message as when
      everything was measured one at a time.

    A rejection leaves the products measured after it, never judged, to the record pass
    (``StepState.run_rest``). An error while serving or measuring is raised only after the
    products before it are judged, as it would be one product at a time; if one of those
    rejects, the error is left to the record pass instead (``StepState.run_rest_error``).
    """
    root = ctx.state.claimed_root
    if root is None:
        raise RuntimeError("check 5 needs the claimed root that check 2's read validated")
    leaves = ctx.state.committed()
    view = TranscriptView(c, leaves)
    with _guard(ctx, "5", _AFTER_COMMIT), ctx.timed("5.glue"):
        replay = c.replay(leaves)
    ctx.state.replay = replay
    scratch = MeasureScratch()
    window: list[_Rows] = []
    pending = 0
    w = 0  # the window being filled
    group0 = 0  # the first group id of the run: ids run across the step's runs
    for i, run in enumerate(_member_runs(c.products)):
        try:
            with ctx.stamp("c5.run", i):
                rows, error = _measure_served(c, run, ctx, replay, view, h=root, k=ctx.k,
                                              scratch=scratch, guard=ctx.kappa_guard, run=i,
                                              group0=group0)
        except Exception as e:  # raised after the products before it are judged
            rej = _judge_window(c, window, ctx, bands)
            if rej is None:
                raise
            ctx.state.run_rest_error = e
            return rej
        if error is not None:
            # As one at a time: the products before this run, then its served part, then the
            # error. A rejection before this run leaves its served part to the record pass.
            rej = _judge_window(c, window, ctx, bands)
            if rej is None:
                rej = _judge_window(c, [rows], ctx, bands)
            if rej is None:
                raise error
            ctx.state.run_rest_error = error
            return rej
        group0 += len(_measure_groups(run))
        window.append(rows)
        pending += len(rows.specs)
        if pending >= _JUDGE_WINDOW:
            with ctx.stamp("c5.judge", -1, -1, w):
                rej = _judge_window(c, window, ctx, bands)
            if rej is not None:
                return rej
            window, pending, w = [], 0, w + 1
    with ctx.stamp("c5.judge", -1, -1, w):
        return _judge_window(c, window, ctx, bands)


class _Group(NamedTuple):
    """Products of a run measured together: their positions in the run, and their fp32
    operands, stacked ``[len(idx), ·, ·]`` for members, as served for a run of one."""

    idx: list[int]
    a: torch.Tensor
    b: torch.Tensor
    p: torch.Tensor
    bias: torch.Tensor | None


class _Rows(NamedTuple):
    """A run's measured products: their specs in run order, the run positions of ``rows``'s
    rows in turn, and the rows (``freivalds.ROW``) on the operands' device."""

    specs: Sequence[ProductSpec]
    order: list[int]
    rows: torch.Tensor | None


def _measure_served(c: DeclaredComputation, specs: Sequence[ProductSpec], ctx: StepContext,
                    replay: Replay, view: TranscriptView, *, h: bytes, k: int,
                    scratch: MeasureScratch, guard: bool, run: int | None = None,
                    group0: int = 0) -> tuple[_Rows, Exception | None]:
    """Serve and measure the run's products a group at a time (:func:`_measure_groups`): the
    rows of the products served before the error that stopped the serving, if any, and that
    error. An error while measuring is raised.

    Each group is measured (inside ``5.measure``) as soon as it is served, and only its rows
    are kept, so the device holds one group's operands at a time, at most
    :data:`_GROUP_BYTES` of them. A run of one is served alone (:func:`_served`), a member
    group at once (:func:`_served_group`). If a group's serving raises,
    :func:`_measure_to_failure` finishes the run one member at a time. ``run`` is the run's
    position in :func:`_member_runs`, the id its profiling stamps carry; ``None`` (the record
    pass) stamps nothing. ``group0`` is the id of the run's first group; ids run on across the
    step's runs.
    """
    st = ctx.stamper() if run is not None else None
    if len(specs) == 1:
        try:
            served = [_served(c, specs[0], ctx, replay, view, run=run, group=group0, st=st)]
        except Exception as e:
            return _Rows(specs[:0], [], None), e
        with ctx.timed("5.measure"), (st("c5.measure", run, group0) if st else _NO_STAMP):
            return _measure_rows(specs, _stacked(specs, served), h=h, k=k, scratch=scratch,
                                 guard=guard), None
    groups, rows = _measure_groups(specs), []
    for g, idx in enumerate(groups):
        try:
            group = _served_group(c, specs, idx, ctx, replay, view, run=run, group=group0 + g,
                                  st=st)
        except Exception:  # served again one at a time, which raises at the member
            return _measure_to_failure(c, specs, groups, rows, ctx, replay, view, h=h, k=k,
                                       scratch=scratch, guard=guard, run=run, group0=group0)
        with ctx.timed("5.measure"), (st("c5.measure", run, group0 + g) if st else _NO_STAMP):
            rows.append(_measure_rows(specs, [group], h=h, k=k, scratch=scratch,
                                      guard=guard).rows)
        del group  # the operands; only the rows are kept
    return _Rows(specs, [i for idx in groups for i in idx],
                 rows[0] if len(rows) == 1 else torch.cat(rows)), None


def _served_group(c: DeclaredComputation, specs: Sequence[ProductSpec], idx: Sequence[int],
                  ctx: StepContext, replay: Replay, view: TranscriptView, *,
                  run: int | None = None, group: int = 0,
                  st: Callable[..., Any] | None = None) -> _Group:
    """The operands and leaves of the run's members at positions ``idx``, as :func:`_stacked`
    would stack :func:`_served`'s, bit for bit. Any error, a failed shape check included,
    goes to :func:`_measure_to_failure`. ``run`` and ``group`` are the profiling stamps' ids
    and ``st`` the stamp function (``None`` stamps nothing)."""
    members = [specs[i] for i in idx]
    if any(spec.bias is not None for spec in members):
        raise RuntimeError("a member names a bias")  # validate forbids it
    ms = [spec.m for spec in members]
    with ctx.timed("5.glue"), (st("c5.glue", run, group) if st else _NO_STAMP):
        a, b = replay.operand_group(ms, torch.float32)
    first = members[0]
    if (tuple(a.shape), tuple(b.shape)) != ((len(ms), *first.a_shape),
                                            (len(ms), *first.b_shape)):
        raise RuntimeError(f"{first.name}: replay group operands {tuple(a.shape)}·"
                           f"{tuple(b.shape)}")
    # The leaves are in host memory: one stack there, then one copy to the operands' device
    # (F4a), in the stored dtype and cast there (a blocking `.to(device, dtype)` converts on
    # the CPU).
    with (st("c5.serve", run, group) if st else _NO_STAMP):
        p = torch.stack([view.product(m) for m in ms])
        with (st("c5.copy", run, group) if st else _NO_STAMP):
            p_dev = p.detach().to(a.device)
    return _Group(list(idx), a, b, p_dev.to(torch.float32), None)


def _measure_to_failure(c: DeclaredComputation, specs: Sequence[ProductSpec],
                        groups: Sequence[Sequence[int]], rows: Sequence[torch.Tensor],
                        ctx: StepContext, replay: Replay, view: TranscriptView, *, h: bytes,
                        k: int, scratch: MeasureScratch, guard: bool,
                        run: int | None = None,
                        group0: int = 0) -> tuple[_Rows, Exception | None]:
    """:func:`_measure_served` once group ``groups[len(rows)]``'s serving has raised, the
    groups before it measured into ``rows``: the run's rows up to its first failing member,
    and that member's error, or the whole run's rows and ``None`` if no member fails alone.

    The members of the unmeasured groups are served one at a time in canonical order, and
    dropped, to find the first that fails (the earlier groups' members served, as groups,
    as they would alone). Then the unmeasured groups' members before it are served again one
    at a time and measured, a group at a time, and of the measured groups only those members
    are kept. So the products judged are those one-at-a-time serving would reach, and the
    device still holds one group at a time.
    """
    st = ctx.stamper() if run is not None else None
    done = len(rows)
    end, error = len(specs), None
    at = {i: group0 + g for g, idx in enumerate(groups) for i in idx}  # a member's group id
    for i in sorted(i for idx in groups[done:] for i in idx):
        try:
            _served(c, specs[i], ctx, replay, view, run=run, group=at[i], st=st)
        except Exception as e:
            end, error = i, e
            break
    out, order = [], []
    for idx, r in zip(groups, rows):
        keep = [i for i in idx if i < end]  # a prefix: a group is in run order
        out.append(r[:len(keep)])
        order += keep
    for g, idx in enumerate(groups):
        if g < done:
            continue
        keep = [i for i in idx if i < end]
        if not keep:
            continue
        served = [_served(c, specs[i], ctx, replay, view, run=run, group=group0 + g, st=st)
                  for i in keep]
        group = _Group(keep, *(torch.stack([x[j] for x in served]) for j in range(3)), None)
        del served
        with ctx.timed("5.measure"), (st("c5.measure", run, group0 + g) if st else _NO_STAMP):
            out.append(_measure_rows(specs, [group], h=h, k=k, scratch=scratch,
                                     guard=guard).rows)
        del group
        order += keep
    return _Rows(specs[:end], order, torch.cat(out) if out else None), error


Served = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
"""``(A, B, P, b)`` in fp32, ``b`` the bias of a biased product (F14) or ``None``."""


def _served(c: DeclaredComputation, spec: ProductSpec, ctx: StepContext, replay: Replay,
            view: TranscriptView, *, run: int | None = None, group: int = 0,
            st: Callable[..., Any] | None = None) -> Served:
    """Product ``spec.m``'s replayed operands and committed leaf, as fp32 ``(A, B, P, b)``.

    ``b`` is the committed ``W_t`` leaf of ``spec.bias``, rounded to the operand format as
    ``B`` is, or ``None`` for an unbiased product. ``run`` and ``group`` are the profiling
    stamps' ids and ``st`` the stamp function (``None`` stamps nothing)."""
    bias = None
    with _guard(ctx, "5", _AFTER_COMMIT):
        with ctx.timed("5.glue"), (st("c5.glue", run, group) if st else _NO_STAMP):
            a, b = replay.operands(spec.m)
        p = view.product(spec.m)
        if spec.bias is not None:
            bias = view.w_t(spec.bias).detach().to(a.device, c.operand_dtype).to(torch.float32)
    if (tuple(a.shape), tuple(b.shape)) != (spec.a_shape, spec.b_shape):
        raise RuntimeError(f"{spec.name}: replay operands {tuple(a.shape)}·{tuple(b.shape)} "
                           f"!= declared {spec.a_shape}·{spec.b_shape}")
    # The leaves are in host memory, the replay on the run's device (F4a): check 5 measures
    # there, so the leaf moves to the operands, in the stored dtype and cast there (a blocking
    # `.to(device, dtype)` converts on the CPU).
    with (st("c5.serve", run, group) if st else _NO_STAMP):
        with (st("c5.copy", run, group) if st else _NO_STAMP):
            p_dev = p.detach().to(a.device)
    return (a.detach().to(torch.float32), b.detach().to(torch.float32),
            p_dev.to(torch.float32), bias)


def _stacked(specs: Sequence[ProductSpec], served: Sequence[Served]) -> list[_Group]:
    """``served``, one product per spec, stacked into :func:`_measure_groups`'s groups."""
    if any((x[3] is None) != (s.bias is None) for s, x in zip(specs, served)):
        raise RuntimeError("a served bias doesn't match its product's declared bias")
    if len(specs) == 1:
        return [_Group([0], *served[0])]
    return [_Group(idx, *(torch.stack([served[i][x] for i in idx]) for x in range(3)), None)
            for idx in _measure_groups(specs)]


def _measure_rows(specs: Sequence[ProductSpec], groups: Sequence[_Group], *, h: bytes, k: int,
                  scratch: MeasureScratch | None = None, guard: bool = True) -> _Rows:
    """Check 5's numbers for each product of a run, as rows on the operands' device.

    A run of one goes through ``measure_product_row``, each group of a longer run through
    ``measure_products_rows``. ``scratch`` is the buffer check 5 reuses across the step's
    products. ``guard=False`` skips test 1 (``StepContext.kappa_guard``). A biased product is
    never a member (``validate``), so it is a run of one.
    """
    rows, order = [], []
    for g in groups:
        if len(specs) == 1:
            rows.append(measure_product_row(g.a, g.b, g.p, h=h, m=specs[0].m, k=k,
                                            scratch=scratch, guard=guard, bias=g.bias)[None])
        else:
            rows.append(measure_products_rows(g.a, g.b, g.p, h=h, ms=[specs[i].m for i in g.idx],
                                              k=k, scratch=scratch, guard=guard))
        order += g.idx
    return _Rows(specs, order, None if not rows else rows[0] if len(rows) == 1
                 else torch.cat(rows))


def _finished(r: _Rows, host: Sequence[Sequence[float]], *, eps_in: float, eps_acc: float,
               guard: bool) -> list[ProductMeasure]:
    """``r``'s products' numbers in run order, from its rows as read back (``host[j]`` is
    ``r.rows[j]``)."""
    out: dict[int, ProductMeasure] = {}
    for row, pos in zip(host, r.order):
        spec = r.specs[pos]
        out[pos] = finish_row(row, q=spec.q, n=spec.a_shape[0], eps_in=eps_in, eps_acc=eps_acc,
                              guard=guard, biased=spec.bias is not None)
    return [out[i] for i in range(len(r.specs))]


def _finish(batch: Sequence[_Rows], ctx: StepContext) -> list[tuple[ProductSpec, ProductMeasure]]:
    """The batch's products with their numbers, in canonical order, after one read of all
    their rows back to the host (in section ``5.measure``)."""
    present = [r.rows for r in batch if r.rows is not None]
    if not present:
        return []
    with ctx.timed("5.measure"):
        host = (present[0] if len(present) == 1 else torch.cat(present)).tolist()
        out: list[tuple[ProductSpec, ProductMeasure]] = []
        base = 0
        for r in batch:
            n = len(r.order)
            out += zip(r.specs, _finished(r, host[base:base + n], eps_in=ctx.eps_in,
                                          eps_acc=ctx.eps_acc, guard=ctx.kappa_guard))
            base += n
    return out


def _judge_window(c: DeclaredComputation, batch: Sequence[_Rows], ctx: StepContext,
                  bands: Bands) -> Rejection | None:
    """Read the batch's numbers, then record and judge its products in canonical order. At a
    rejection the products after the failing one go to ``StepState.run_rest``, measured and
    never judged, for the record pass."""
    done = _finish(batch, ctx)
    for i, (spec, mp) in enumerate(done):
        rej = _judge_product(c, spec, mp, ctx, bands)
        if rej is not None:
            ctx.state.run_rest = done[i + 1:]
            return rej
    return None


def _measure_run(specs: Sequence[ProductSpec], served: Sequence[Served], *,
                 h: bytes, k: int, eps_in: float, eps_acc: float,
                 scratch: MeasureScratch | None = None,
                 guard: bool = True) -> list[ProductMeasure]:
    """Check 5's numbers for each product of a run, in run order, from operands served one
    product at a time (:func:`_served`), grouped and measured as check 5 does. For callers
    outside check 5 (the planted-error sweep, tests)."""
    rows = _measure_rows(specs, _stacked(specs, served), h=h, k=k, scratch=scratch,
                         guard=guard)
    if rows.rows is None:
        return []
    return _finished(rows, rows.rows.tolist(), eps_in=eps_in, eps_acc=eps_acc, guard=guard)


def _judge_product(c: DeclaredComputation, spec: ProductSpec, mp: ProductMeasure,
                   ctx: StepContext, bands: Bands) -> Rejection | None:
    """Record product ``spec.m``'s numbers, then judge them: finiteness, test 1, test 2.

    With ``ctx.kappa_guard`` off, test 1 and its two numbers are skipped (C1's cost split)."""
    cls_key = product_class(c, spec)
    nu, p_abs1, kappa, res, unit = mp.nu, mp.p_abs1, mp.kappa, mp.residuals, mp.unit
    normalized = mp.normalized
    ctx.stats.products.append(ProductStat(spec.m, spec.name, cls_key, kappa, normalized,
                                          q=spec.q, p_norm=mp.p_norm, nu=nu,
                                          p_abs1=p_abs1, u=mp.u))
    guard = {"ν": nu, "‖|P|·1‖": p_abs1} if ctx.kappa_guard else {}
    values = {**guard, "‖P‖_F": mp.p_norm, "U": mp.u, "band unit": unit,
              **{f"residual j={j}": x for j, x in enumerate(res, start=1)}}
    bad = [key for key, v in values.items() if not math.isfinite(v)]
    if bad:
        return ctx.reject("5", f"P_{spec.m} ({spec.name}): non-finite {', '.join(bad)}")
    if not ctx.judge:
        return None
    kappa_max = bands.kappa_for(cls_key)
    if ctx.kappa_guard and not nu <= kappa_max * p_abs1:
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
    grads = _replay_glue_gradients(c, ctx, replay)
    scratch = UpdateScratch()
    for name in names:
        with _guard(ctx, "6b", _AFTER_COMMIT):
            w_t, w_next = view.w_t(name), view.w_next(name)
        g = _glue_gradient(grads, name)
        rej = _update_identity(ctx, "6b", name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                               scratch)
        if rej is not None:
            return rej
    return None


def _replay_glue_gradients(c: DeclaredComputation, ctx: StepContext,
                           replay: Replay) -> Mapping[str, torch.Tensor]:
    """The replay's backward glue (section ``6b.glue``) for every glue weight. A weight it
    leaves out is a verifier bug. Check 6b and its record pass both run it here."""
    with _guard(ctx, "6b", _AFTER_COMMIT), ctx.timed("6b.glue"):
        grads = replay.glue_gradients()
    missing = [n for n in c.glue_gradient_weights if n not in grads]
    if missing:
        raise RuntimeError(f"replay returned no glue gradient for {missing}")
    return grads


def _glue_gradient(grads: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Check 6b's ``G`` for glue weight ``name``, in fp32 (P6), in host memory beside the
    weight leaves it is checked against; the record pass takes it here too."""
    return grads[name].detach().to(HOST, torch.float32)


CHECKS: Mapping[str, Callable[[TranscriptStore, DeclaredComputation, StepContext, Bands],
                              Rejection | None]] = MappingProxyType({
    "4": check_4_batch_anchor,
    "7": check_7_chaining,
    "2": check_2_commitment,
    "6a": check_6a_linear_update,
    "5": check_5_matmuls,
    "6b": check_6b_glue_update,
    ROOT_SLOT: check_2_root,
})


# ---- after a rejection: the record pass (EV2, EVALUATION_SPEC §4.1) ----------------------

RECORDED_SLOTS: tuple[str, ...] = ("6a", "5", "6b")
"""The slots whose tests have numbers (``ρ_5``, ``ρ_κ``, ``ρ_6``), so :func:`record_rest` can
measure the ones a rejection skipped."""

# How far the record pass got (EVALUATION_SPEC §4.1, the step record's ``record_pass``).
RECORD_COMPLETE = "complete"  # it measured every test that remained
RECORD_MALFORMED = "ended: malformed"  # a leaf failed to read; the tests before it are in
RECORD_ERROR = "error"  # a verifier-side error stopped it (the driver catches it)
RECORD_NOT_RUN = "not run"  # nothing to measure, or the pass is skipped
RECORD_STATUSES: tuple[str, ...] = (RECORD_COMPLETE, RECORD_MALFORMED, RECORD_ERROR,
                                    RECORD_NOT_RUN)


def record_rest(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                bands: Bands, slot: str) -> str:
    """Measure, for the record only, every test the rejecting check skipped; return how far it
    got.

    A check stops at its first failing test. When slot ``slot`` has rejected the step, the
    driver calls this after the verdict is fixed, and it measures the rest of that check's
    tests into ``ctx.stats``:

    - check 5: every product after the failing one, in canonical order. Check 5 measured
      some of them before judging the failing one (the rest of its judging window,
      ``_JUDGE_WINDOW``) and never judged them; their numbers are recorded as check 5
      measured them (``StepState.run_rest``), since measuring a member alone
      (``measure_product``) and in its group (``measure_products``) are held to agree only to
      ``1e-4`` relative, and below about 8×8 to O(1) band units (``test_freivalds``). Later
      runs are served and measured from check 5's replay, as check 5 does;
    - check 6a: every linear weight after the failing one, then every glue weight of 6b,
      from a replay of its own (check 6 is one protocol check over every weight tensor, so its
      ``ρ_6`` covers all of them);
    - check 6b: every glue weight after the failing one.

    Each gradient comes from the helper its live check uses (:func:`_linear_gradient`,
    :func:`_replay_glue_gradients`, :func:`_glue_gradient`), so an untouched tensor scores
    what the live check would give it.

    It runs in record mode (``judge=False``, no metrics seam) and changes no rejection, verdict
    or timing. A product or tensor that fails is still recorded, and the pass goes on. It
    returns :data:`RECORD_COMPLETE`, or :data:`RECORD_NOT_RUN` when ``slot`` has no scored
    tests or none remained after the failing one. A malformed leaf ends the pass, which returns
    :data:`RECORD_MALFORMED` with the tests before that leaf recorded. Any other error is a
    verifier bug and propagates, as in the checks; the driver catches it, keeps the rejection
    and records :data:`RECORD_ERROR`.
    """
    if slot not in RECORDED_SLOTS:
        return RECORD_NOT_RUN
    rctx = dataclasses.replace(ctx, judge=False, section=None)  # shares state and stats
    before = len(ctx.stats.products), len(ctx.stats.tensors)
    try:
        if slot == "5":
            _record_rest_5(c, rctx, bands)
        elif slot == "6a":
            _record_rest_6(c, rctx, bands, "6a")
            _record_rest_6(c, rctx, bands, "6b")
        else:
            _record_rest_6(c, rctx, bands, "6b")
    except _Rejected:
        return RECORD_MALFORMED
    if (len(ctx.stats.products), len(ctx.stats.tensors)) == before:
        return RECORD_NOT_RUN
    return RECORD_COMPLETE


def _record_rest_5(c: DeclaredComputation, ctx: StepContext, bands: Bands) -> None:
    root, replay = ctx.state.claimed_root, ctx.state.replay
    if root is None or replay is None:
        raise RuntimeError("check 5's record pass needs check 5's root and replay")
    rest, error = ctx.state.run_rest, ctx.state.run_rest_error
    ctx.state.run_rest, ctx.state.run_rest_error = [], None
    for spec, mp in rest:  # measured by check 5 after the failing product, never judged
        _judge_product(c, spec, mp, ctx, bands)  # records; judge=False decides nothing
    if error is not None:  # serving or measuring a later product failed, as it did in check 5
        raise error
    view = TranscriptView(c, ctx.state.committed())
    done = {p.m for p in ctx.stats.products}
    scratch = MeasureScratch()
    for run in _member_runs(c.products):
        specs = [spec for spec in run if spec.m not in done]
        if not specs:
            continue
        rows, error = _measure_served(c, specs, ctx, replay, view, h=root, k=ctx.k,
                                      scratch=scratch, guard=ctx.kappa_guard)
        if error is not None:
            raise error
        for spec, mp in _finish([rows], ctx):
            _judge_product(c, spec, mp, ctx, bands)


def _record_rest_6(c: DeclaredComputation, ctx: StepContext, bands: Bands,
                   check_id: str) -> None:
    leaves = ctx.state.committed()
    view = TranscriptView(c, leaves)
    done = {s.weight for s in ctx.stats.tensors if s.check_id == check_id}
    grads: Mapping[str, torch.Tensor] | None = None
    if check_id == "6a":
        todo = [n for n in c.linear_weights if n not in done]
    else:
        todo = [n for n in c.glue_gradient_weights if n not in done]
        if not todo:
            return
        replay = ctx.state.replay
        if replay is None:  # a 6a rejection: check 5 never ran, so run its replay here
            with _guard(ctx, "6b", _AFTER_COMMIT):
                replay = c.replay(leaves)
                for spec in c.products:  # in canonical order, as check 5 serves them
                    replay.operands(spec.m)
            ctx.state.replay = replay
        grads = _replay_glue_gradients(c, ctx, replay)
    scratch = UpdateScratch()
    for name in todo:
        with _guard(ctx, check_id, _AFTER_COMMIT):
            w_t, w_next = view.w_t(name), view.w_next(name)
            g = _linear_gradient(c, view, name) if grads is None else _glue_gradient(grads, name)
        _update_identity(ctx, check_id, name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                         scratch)
