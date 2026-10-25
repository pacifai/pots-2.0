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
import logging
import math
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import torch

from verification.commitment.merkle import DIGEST_SIZE, merkle_root, verify_path
from verification.computation.interface import (
    DeclaredComputation,
    ProductSpec,
    Replay,
    chain_gradient,
)
from verification.transcript.reader import TranscriptView
from verification.transcript.store import TranscriptStore
from verification.verifier.bands import Bands, product_class, product_classes
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
from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check.freivalds import (
    MeasureColumns,
    MeasureGroup,
    MeasureScratch,
    ProductMeasure,
    finish_row,
    finish_rows,
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

    stats = ctx.hash_stats()
    with _guard(ctx, "2"):
        with ctx.timed("2.hash"):  # the read-validate-hash loop, apart from the checks after it
            hashes = _hashes(c, read(), stats)
        if stats is not None:
            stats.emit(ctx.section, "2.hash")
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

    One allocation per device and check call, not about ten fresh temporaries per weight.
    Each block stays small enough to sit in cache between the passes over it. Weights on
    different devices (F22) each use their device's buffers, so moving between them never
    reallocates.
    """

    def __init__(self) -> None:
        self._bufs: dict[torch.device, tuple[torch.Tensor, ...]] = {}

    def get(self, n: int, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b = self._bufs.get(like.device, ())
        if not b or b[0].numel() < n or b[0].dtype != like.dtype:
            size = max(n, UPDATE_CHUNK)
            self._bufs[like.device] = b = tuple(
                torch.empty(size, dtype=like.dtype, device=like.device) for _ in range(3))
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
            w_t, w_next = view.w_t(name), view.w_next(name)
            g = _linear_gradient(c, view, name, w_t.device)
        rej = _update_identity(ctx, "6a", name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                               scratch)
        if rej is not None:
            return rej
    return None


def _linear_gradient(c: DeclaredComputation, view: TranscriptView, name: str,
                     device: torch.device) -> torch.Tensor:
    """Check 6a's ``G`` for linear weight ``name``: its committed weight-gradient leaves, each
    widened exactly to the weights' format (F6b: a bf16 leaf becomes fp32; an fp32 leaf is
    unchanged), as 6b's gradients are, and summed in order as training sums them
    (:func:`chain_gradient`; one leaf unless the weight's product runs in row blocks). Check 6a
    and its record pass both take ``G`` here. It sits on ``device``, the weight leaf's, where
    the update identity runs (a leaf already there is not copied)."""
    return chain_gradient((view.product(m).to(device) for m in c.linear_weights[name]),
                          c.weight_dtype)


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
    ``product_bytes`` (``P`` in the stored dtype, ``product_dtype``) and ``served_bytes`` (the
    products as measured on the device, in the same dtype: equal to ``product_bytes``, since the
    kernel reads ``P`` as stored, FS37; plan §2.7). The serving after an error
    (:func:`_measure_to_failure`) is not described."""
    return _layout(c)[0]


def _layout(c: DeclaredComputation) -> tuple[list[dict[str, Any]], int]:
    """:func:`check5_layout`'s entries and ``n_windows``, the number of judge calls
    ``check_5_matmuls`` makes on a pass with no error: a window ends after the run that fills
    it, and the last is judged even when empty (so it is counted when a run closes the one
    before it)."""
    out: list[dict[str, Any]] = []
    runs = list(_member_runs(c.products))
    windows = _judge_windows(runs)
    window_of = {i: w for w, idx in enumerate(windows) for i in idx}
    item = torch.empty(0, dtype=c.product_dtype).element_size()
    for i, run in enumerate(runs):
        window = window_of[i]
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
                "served_bytes": item * len(specs) * n * w})
    return out, len(windows)


@_checked
def check_5_matmuls(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                    bands: Bands) -> Rejection | None:
    """Both tests of spec §6 check 5 on every product, judged in canonical order; abort at the
    first fail.

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

    **Order.** A replay with a unit schedule (``Replay.unit_schedule``: the decoder's) is served
    and measured in it, the head's blocks and then layers ``L … 1``, each layer's forward
    products with its backward ones, so the replay runs its forward pass once and each backward
    unit once (:func:`_check_5_units`). Every run's rows stay on the device and are read back
    once, per device, at the first window, and the windows are then judged in canonical order.
    A step whose runs are all kernel runs is served run by run and measured as one table, one
    ``measure_groups_rows(share=True)`` call per device, whose rows are split back to the runs
    (:func:`_measure_step_kernel`); any other step is measured run by run.
    A replay without one (the MLP) is served and judged in canonical order
    (:func:`_check_5_canonical`), each window as it fills, and so is the rerun on a fresh
    replay when the unit pass raises (a serving or measuring error, a failed read).
    A schedule that repeats or misses a product, or splits a run, raises ``RuntimeError``.

    **Batching.** A product's measuring is O(n²) arithmetic, little next to the fixed cost of
    launching its ops and waiting for their results, so check 5 batches in two ways and keeps
    the order of judging:

    - The members of a batched product (an attention product's ``(s, h)`` members) are many
      and small. A run of consecutive member specs of one layer (:func:`_member_runs`) is
      served and measured in groups (:func:`_measure_groups`), one group at a time
      (:func:`_measure_served`): the replay serves the group's operands stacked
      (``Replay.operand_group``), its ``P`` leaves move to the device in one copy, and the
      group is measured in batched ops, each member with its own challenges and norm scaling
      (``freivalds.measure_products_rows``). Only its rows are kept. On the unit path with
      the kernel the batching is one table per schedule step instead: the step's groups,
      served as views, are measured by one call that reads a stored matrix several products
      share once (FS38b); the canonical path measures a group at a time.
    - The numbers stay on the device as one row per product. On the canonical path, once at
      least :data:`_JUDGE_WINDOW` products are measured, one read brings the window's rows to
      the host (:func:`_finish`); on the unit path one read brings every row. Either way
      products are recorded and judged window by window, one by one in canonical order, so the
      first failing product rejects with the same message as when everything was measured one
      at a time.

    A rejection leaves the products measured after it, never judged, to the record pass
    (``StepState.run_rest``, and on the unit path ``run_rest_rows`` for later windows). On the
    canonical path an error while serving or measuring is raised only after the products
    before it are judged, as it would be one product at a time; if one of those rejects, the
    error is left to the record pass instead (``StepState.run_rest_error``). The unit pass
    leaves no error to the record pass: it reruns canonically instead.
    """
    root = ctx.state.claimed_root
    if root is None:
        raise RuntimeError("check 5 needs the claimed root that check 2's read validated")
    leaves = ctx.state.committed()
    view = TranscriptView(c, leaves)
    with _guard(ctx, "5", _AFTER_COMMIT), ctx.timed("5.glue"):
        replay = c.replay(leaves)
    ctx.state.replay = replay
    try:
        schedule = replay.unit_schedule()
        if schedule is None:
            return _check_5_canonical(c, ctx, bands, replay, view, root)
        runs = list(_member_runs(c.products))
        steps = _schedule_runs(runs, schedule)  # a RuntimeError (a verifier bug) propagates
        failed: tuple[str, int | None] | None = None
        try:
            return _check_5_units(c, ctx, bands, replay, view, root, runs, steps)
        except _UnitPassError as e:
            failed = (type(e.__cause__).__name__, e.run)
        # Outside the handler, so no traceback keeps the old replay alive: a fresh replay (the
        # decoder's reused model takes fresh parameter storage) and today's canonical loop.
        replay = None  # type: ignore[assignment]
        ctx.state.replay = None  # the old graph must not be alive while the new one builds
        logging.getLogger(__name__).debug(
            "check 5: the unit pass raised %s at run %s; rerunning in canonical order", *failed)
        with _guard(ctx, "5", _AFTER_COMMIT), ctx.timed("5.glue"):
            replay = c.replay(leaves)
        ctx.state.replay = replay
        return _check_5_canonical(c, ctx, bands, replay, view, root)
    finally:
        freivalds.drain()  # no kernel hold outlives the check (invariant 6)


def _check_5_canonical(c: DeclaredComputation, ctx: StepContext, bands: Bands,
                       replay: Replay, view: TranscriptView, root: bytes) -> Rejection | None:
    """Check 5's loop in canonical order 1..M, on ``replay`` (a fresh one): each run served and
    measured in turn, each window judged as it fills (:data:`_JUDGE_WINDOW`), stopping at the
    first rejection. The path of a replay with no unit schedule (the MLP), and the one
    :func:`check_5_matmuls` reruns when the unit pass raises."""
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


class _UnitPassError(Exception):
    """The unit pass raised while serving, measuring or reading (the cause): ``run`` is the
    canonical run it was in, ``None`` at the end read. :func:`check_5_matmuls` reruns in
    canonical order."""

    def __init__(self, run: int | None):
        super().__init__(run)
        self.run = run


def _judge_windows(runs: Sequence[Sequence[ProductSpec]]) -> list[list[int]]:
    """The judging windows of check 5 over ``runs`` (:func:`_member_runs`): the run indices of
    each window in order. A window closes after the run that brings it to
    :data:`_JUDGE_WINDOW` products, and the last window, the one still filling, is judged even
    when empty (a run that closed the one before it leaves it empty)."""
    windows: list[list[int]] = []
    window: list[int] = []
    pending = 0
    for i, run in enumerate(runs):
        window.append(i)
        pending += len(run)
        if pending >= _JUDGE_WINDOW:
            windows.append(window)
            window, pending = [], 0
    windows.append(window)
    return windows


def _schedule_runs(runs: Sequence[Sequence[ProductSpec]],
                   schedule: Sequence[Sequence[int]]) -> list[list[int]]:
    """The canonical runs each step of ``schedule`` holds, as run indices in the step's order.

    A schedule that repeats or misses a product, or splits a run over steps, is a verifier bug:
    it raises ``RuntimeError``."""
    where: dict[int, tuple[int, int]] = {}  # m -> (step, position in the step)
    for s, step in enumerate(schedule):
        for pos, m in enumerate(step):
            if m in where:
                raise RuntimeError(f"unit schedule serves product {m} twice")
            where[m] = (s, pos)
    want = {spec.m for run in runs for spec in run}
    if set(where) != want:
        raise RuntimeError(f"unit schedule misses products {sorted(want - set(where))[:5]} or "
                           f"adds {sorted(set(where) - want)[:5]}")
    out: list[list[int]] = [[] for _ in schedule]
    for i, run in enumerate(runs):
        held = {where[spec.m][0] for spec in run}
        if len(held) != 1:
            raise RuntimeError(f"unit schedule splits the run of {run[0].name} over steps")
        out[held.pop()].append(i)
    for step in out:
        step.sort(key=lambda i: where[runs[i][0].m][1])
    return out


class _ReadOnce:
    """The rows of the unit pass's runs, read to the host once, when first asked for: one
    drain and one read per device over every run (:func:`_read_rows`). A failing read raises
    :class:`_UnitPassError`, so check 5 falls back before it has recorded anything."""

    def __init__(self, rows: Sequence[_Rows]):
        self._present = [r for r in rows if r.rows is not None]
        self._host: dict[int, np.ndarray] | None = None

    def host(self, r: _Rows) -> np.ndarray:
        """``r``'s rows as a float64 array, in the order of ``r.order``."""
        if self._host is None:
            try:
                read = _read_rows([x.rows for x in self._present])  # type: ignore[misc]
            except Exception as e:
                raise _UnitPassError(None) from e
            self._host = {id(x): h for x, h in zip(self._present, read)}
        return self._host[id(r)]


def _check_5_units(c: DeclaredComputation, ctx: StepContext, bands: Bands, replay: Replay,
                   view: TranscriptView, root: bytes, runs: Sequence[Sequence[ProductSpec]],
                   steps: Sequence[Sequence[int]]) -> Rejection | None:
    """Check 5 in the replay's unit order (``Replay.unit_schedule``): every run measured in its
    step's turn, every run's rows kept on the device, read once at the first non-empty window,
    and the windows judged in canonical order from the host rows, so the first failing
    product, its message and every recorded number are canonical judging's. Each run keeps its
    canonical index and group ids (the stamps' and :func:`_layout`'s). A step of kernel runs
    only is one table (:func:`_measure_step_kernel`: the runs served in order, one shared-read
    measuring call per device, the rows split back per run); a step with any other run is
    measured run by run.

    An error while serving, measuring or reading raises :class:`_UnitPassError` after the
    kernel's work in flight is waited for (:func:`check_5_matmuls` then reruns in canonical
    order). Nothing is recorded in ``ctx.stats`` before the read. On a rejection,
    ``StepState.run_rest`` holds the failing window's tail and ``run_rest_rows`` the host rows
    of every later window's runs."""
    group0, g = [], 0
    for run in runs:
        group0.append(g)
        g += len(_measure_groups(run))
    scratch = MeasureScratch()
    rows: list[_Rows | None] = [None] * len(runs)
    try:
        for step in steps:
            if all(_kernel_run(replay, runs[i]) for i in step):
                for i, r in zip(step, _measure_step_kernel(c, [runs[i] for i in step], step, ctx,
                                                           replay, view, h=root, k=ctx.k,
                                                           guard=ctx.kappa_guard,
                                                           group0=group0)):
                    rows[i] = r
                continue
            for i in step:
                try:
                    with ctx.stamp("c5.run", i):
                        r, error = _measure_served(c, runs[i], ctx, replay, view, h=root,
                                                   k=ctx.k, scratch=scratch,
                                                   guard=ctx.kappa_guard, run=i,
                                                   group0=group0[i])
                    if error is not None:
                        raise error
                except Exception as e:
                    raise _UnitPassError(i) from e
                rows[i] = r
        done: list[_Rows] = rows  # type: ignore[assignment]
        once = _ReadOnce(done)
        windows = _judge_windows(runs)
        for w, idx in enumerate(windows):
            with ctx.stamp("c5.judge", -1, -1, w):
                rej = _judge_window(c, [done[i] for i in idx], ctx, bands, once)
            if rej is not None:
                ctx.state.run_rest_rows = [
                    (done[i]._replace(rows=None), once.host(done[i]))
                    for later in windows[w + 1:] for i in later if done[i].rows is not None]
                return rej
        return None
    except _UnitPassError:
        try:
            freivalds.drain()
        except Exception:
            pass  # the pass already failed; the rerun finds what is wrong
        raise


def _measure_step_kernel(c: DeclaredComputation, runs: Sequence[Sequence[ProductSpec]],
                         step: Sequence[int], ctx: StepContext, replay: Replay,
                         view: TranscriptView, *, h: bytes, k: int, guard: bool,
                         group0: Sequence[int]) -> list[_Rows]:
    """One schedule step on the kernel path (FS38b): every run of ``runs`` (the step's runs, at
    canonical indices ``step``, ``group0`` indexed by those) served as FS37's views, in order
    (a run of one by :func:`_served_view`, a member run group by group by
    :func:`_served_blocks`), then the groups measured by one
    ``freivalds.measure_groups_rows(..., share=True)`` call per device (normally the step's
    one), so a stored matrix several products read is read once per level. The rows come back
    as one :class:`_Rows` per run, in the step's run order. A serving error raises
    :class:`_UnitPassError` at that run, a measuring error at the step's first run."""
    served: list[tuple[list[MeasureGroup], list[int]]] = []
    for i, specs in zip(step, runs):
        groups: list[MeasureGroup] = []
        order: list[int] = []
        try:
            with ctx.stamp("c5.run", i):
                if len(specs) == 1:
                    groups.append(_served_view(c, specs[0], ctx, replay, view, run=i,
                                               group=group0[i], st=ctx.stamper()))
                    order.append(0)
                else:
                    for g, idx in enumerate(_measure_groups(specs)):
                        o, gs = _served_blocks(c, specs, idx, ctx, replay, view, run=i,
                                               group=group0[i] + g, st=ctx.stamper())
                        groups.extend(gs)
                        order.extend(o)
            # The row split below hands each run its rows per device in device order, which is
            # serving order only for a run on one device; a run on two takes the canonical rerun.
            where = {mg.a.device for mg in groups}
            if len(where) > 1:
                raise RuntimeError(f"run {i} is served on several devices: "
                                   f"{sorted(map(str, where))}")
        except Exception as e:
            raise _UnitPassError(i) from e
        served.append((groups, order))
    devices: list[torch.device] = []
    for groups, _ in served:
        for mg in groups:
            if mg.a.device not in devices:
                devices.append(mg.a.device)
    tables: dict[torch.device, torch.Tensor] = {}
    try:
        for dev in devices:
            table = [mg for groups, _ in served for mg in groups if mg.a.device == dev]
            with ctx.timed("5.measure"), ctx.stamp("c5.measure", step[0], -1):
                tables[dev] = freivalds.measure_groups_rows(table, h=h, k=k, guard=guard,
                                                            share=True)
    except Exception as e:
        raise _UnitPassError(step[0]) from e
    taken = {dev: 0 for dev in devices}  # rows of each device's table already given to a run
    out: list[_Rows] = []
    for specs, (groups, order) in zip(runs, served):
        parts = []
        for dev in devices:
            n = sum(mg.a.shape[0] for mg in groups if mg.a.device == dev)
            if n:
                parts.append(tables[dev][taken[dev]:taken[dev] + n])
                taken[dev] += n
        out.append(_Rows(specs, order, parts[0] if len(parts) == 1 else torch.cat(parts)))
    return out


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
    if _kernel_run(replay, specs):
        return _measure_served_kernel(c, specs, ctx, replay, view, h=h, k=k, scratch=scratch,
                                      guard=guard, run=run, group0=group0, st=st)
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
        runs = _role_progressions(ms, first.p_shape, c) if hasattr(view, "product_run") else None
        if runs is not None and (len(runs) > 1 or runs[0][4] != len(ms)):
            # Interleaved roles (δQ̃, δK̃, …) are not one run of leaves: read each role as a
            # range, and copy it with one strided write into the stacked result, converting
            # to float32 there (the cast of the stored values is the same on either device).
            p32 = a.new_empty((len(ms), *first.p_shape), dtype=torch.float32)
            with (st("c5.copy", run, group) if st else _NO_STAMP):
                for pos, R, m0, step, n in runs:
                    blk = view.product_run(m0, step, n)  # read only: copied, never written
                    if tuple(blk.shape[1:]) != tuple(first.p_shape):
                        runs = None  # copy_ would broadcast it; torch.stack raises
                        break
                    p32[pos:pos + R * (n - 1) + 1:R].copy_(blk.detach().to(a.device))
            if runs is not None:
                return _Group(list(idx), a, b, p32, None)
        if runs is not None:
            _, _, m0, step, n = runs[0]
            p = view.product_run(m0, step, n)  # read only: measured and cast, never written
        else:
            p = view.product_block(ms)
        with (st("c5.copy", run, group) if st else _NO_STAMP):
            p_dev = p.detach().to(a.device)
    return _Group(list(idx), a, b, p_dev.to(torch.float32), None)


def _kernel_run(replay: Replay, specs: Sequence[ProductSpec]) -> bool:
    """Whether the run is served as views and measured on the fused kernel: its operands'
    device (``Replay.operand_device``, known before serving anything) is one the kernel
    takes (``freivalds.kernel_device``). Every other device keeps the fp32 serving."""
    where = getattr(replay, "operand_device", None)
    dev = where(specs[0].m) if where is not None else None
    return dev is not None and freivalds.kernel_device(dev)


def _measure_served_kernel(c: DeclaredComputation, specs: Sequence[ProductSpec],
                           ctx: StepContext, replay: Replay, view: TranscriptView, *,
                           h: bytes, k: int, scratch: MeasureScratch, guard: bool,
                           run: int | None, group0: int,
                           st: Callable[..., Any] | None) -> tuple[_Rows, Exception | None]:
    """:func:`_measure_served` on the kernel path (FS37, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.3):
    the replay's operands as views in their own dtype, the leaves moved to the device as they
    are stored, and each group measured by one ``freivalds.measure_groups_rows`` call that
    enqueues and returns the rows without reading them (:func:`_read_rows` drains). The kernel's
    bits don't depend on strides, grouping or widening, so an error finishes through
    :func:`_measure_to_failure` as on the torch path."""
    if len(specs) == 1:
        try:
            group = _served_view(c, specs[0], ctx, replay, view, run=run, group=group0, st=st)
        except Exception as e:
            return _Rows(specs[:0], [], None), e
        with ctx.timed("5.measure"), (st("c5.measure", run, group0) if st else _NO_STAMP):
            rows = freivalds.measure_groups_rows([group], h=h, k=k, guard=guard)
        return _Rows(specs, [0], rows), None
    groups, rows, orders = _measure_groups(specs), [], []
    for g, idx in enumerate(groups):
        try:
            order, served = _served_blocks(c, specs, idx, ctx, replay, view, run=run,
                                           group=group0 + g, st=st)
        except Exception:  # served again one at a time, which raises at the member
            return _measure_to_failure(c, specs, groups, rows, ctx, replay, view, h=h, k=k,
                                       scratch=scratch, guard=guard, run=run, group0=group0,
                                       orders=orders)
        with ctx.timed("5.measure"), (st("c5.measure", run, group0 + g) if st else _NO_STAMP):
            rows.append(freivalds.measure_groups_rows(served, h=h, k=k, guard=guard))
        orders.append(order)
        del served  # the views; the kernel holds what it still reads
    return _Rows(specs, [i for o in orders for i in o],
                 rows[0] if len(rows) == 1 else torch.cat(rows)), None


def _served_view(c: DeclaredComputation, spec: ProductSpec, ctx: StepContext, replay: Replay,
                 view: TranscriptView, *, run: int | None, group: int,
                 st: Callable[..., Any] | None) -> MeasureGroup:
    """:func:`_served` for the kernel: the operands as the replay gives them and the leaf in its
    stored dtype on their device (non-blocking), the bias the committed ``W_t`` leaf rounded to
    the operand format there, none of them widened to fp32."""
    bias = None
    with _guard(ctx, "5", _AFTER_COMMIT):
        with ctx.timed("5.glue"), (st("c5.glue", run, group) if st else _NO_STAMP):
            a, b = replay.operands(spec.m)
        p = view.product(spec.m)
        if spec.bias is not None:
            bias = view.w_t(spec.bias).detach().to(a.device, c.operand_dtype)
    if (tuple(a.shape), tuple(b.shape)) != (spec.a_shape, spec.b_shape):
        raise RuntimeError(f"{spec.name}: replay operands {tuple(a.shape)}·{tuple(b.shape)} "
                           f"!= declared {spec.a_shape}·{spec.b_shape}")
    with (st("c5.serve", run, group) if st else _NO_STAMP):
        with (st("c5.copy", run, group) if st else _NO_STAMP):
            p_dev = p.detach().to(a.device, non_blocking=True)
    return MeasureGroup(a.detach()[None], b.detach()[None], p_dev[None], [spec.m], bias)


def _label_run(ms: Sequence[int]) -> tuple[int, int, int] | None:
    """``ms`` as ``(m₀, step, n)`` when it is an increasing arithmetic progression."""
    step = ms[1] - ms[0] if len(ms) > 1 else 1
    if step < 1 or any(y - x != step for x, y in zip(ms, ms[1:])):
        return None
    return ms[0], step, len(ms)


def _served_blocks(c: DeclaredComputation, specs: Sequence[ProductSpec], idx: Sequence[int],
                   ctx: StepContext, replay: Replay, view: TranscriptView, *,
                   run: int | None, group: int,
                   st: Callable[..., Any] | None) -> tuple[list[int], list[MeasureGroup]]:
    """:func:`_served_group` for the kernel: the run's members at positions ``idx`` as one
    :class:`MeasureGroup` per block of ``replay.operand_blocks``, ``P`` a view of the committed
    leaves (``product_run`` when the block's labels form a progression, else ``product_block``)
    moved to the device as stored. Returns the run positions of the rows ``measure_groups_rows``
    will give, block by block. Any error, a failed shape check included, goes to
    :func:`_measure_to_failure`."""
    members = [specs[i] for i in idx]
    if any(spec.bias is not None for spec in members):
        raise RuntimeError("a member names a bias")  # validate forbids it
    ms = [spec.m for spec in members]
    with ctx.timed("5.glue"), (st("c5.glue", run, group) if st else _NO_STAMP):
        blocks = replay.operand_blocks(ms)
    first = members[0]
    if sorted(t for blk in blocks for t in blk.pos) != list(range(len(ms))):
        raise RuntimeError(f"{first.name}: replay blocks don't cover the group once")
    out: list[MeasureGroup] = []
    with (st("c5.serve", run, group) if st else _NO_STAMP):
        views: list[tuple[Any, torch.Tensor, list[int]]] = []
        for blk in blocks:
            n = len(blk.pos)
            if (tuple(blk.a.shape), tuple(blk.b.shape)) != ((n, *first.a_shape),
                                                            (n, *first.b_shape)):
                raise RuntimeError(f"{first.name}: replay group operands {tuple(blk.a.shape)}·"
                                   f"{tuple(blk.b.shape)}")
            labels = [ms[t] for t in blk.pos]
            span = _label_run(labels) if hasattr(view, "product_run") else None
            p = view.product_run(*span) if span is not None else view.product_block(labels)
            if tuple(p.shape) != (n, *first.p_shape):
                raise RuntimeError(f"{first.name}: leaves {tuple(p.shape)} != "
                                   f"{(n, *first.p_shape)}")
            views.append((blk, p, labels))
        with (st("c5.copy", run, group) if st else _NO_STAMP):  # one stamp per group
            for blk, p, labels in views:
                p_dev = p.detach().to(blk.a.device, non_blocking=True)
                out.append(MeasureGroup(blk.a, blk.b, p_dev, labels))
    return [idx[t] for blk in blocks for t in blk.pos], out


def _role_progressions(ms: Sequence[int], p_shape: tuple[int, ...], c: DeclaredComputation
                       ) -> list[tuple[int, int, int, int, int]] | None:
    """``ms`` cut into interleaved arithmetic progressions by the computation's
    ``member_pieces`` (see :meth:`DecoderComputation.member_pieces`), or ``None`` for the
    plain read: ``c`` has none, ``ms`` doesn't cut, or a piece's first product has a leaf
    shape other than ``p_shape`` (the stacked copy would broadcast it; ``torch.stack``
    raises). Per piece ``(pos, R, m₀, step, n)``: position ``pos + R·t`` holds product
    ``m₀ + step·t``. The pieces cover every position once, so reading each as a range and
    copying it to its positions is ``torch.stack`` of the leaves in the order of ``ms``."""
    cut = getattr(c, "member_pieces", None)
    pieces = cut(ms) if cut is not None else None
    if pieces is None or any(tuple(c.product(m0).p_shape) != tuple(p_shape)
                             for _, _, m0, _, _ in pieces):
        return None
    return pieces


def _measure_to_failure(c: DeclaredComputation, specs: Sequence[ProductSpec],
                        groups: Sequence[Sequence[int]], rows: Sequence[torch.Tensor],
                        ctx: StepContext, replay: Replay, view: TranscriptView, *, h: bytes,
                        k: int, scratch: MeasureScratch, guard: bool,
                        run: int | None = None, group0: int = 0,
                        orders: Sequence[Sequence[int]] | None = None
                        ) -> tuple[_Rows, Exception | None]:
    """:func:`_measure_served` once group ``groups[len(rows)]``'s serving has raised, the
    groups before it measured into ``rows``: the run's rows up to its first failing member,
    and that member's error, or the whole run's rows and ``None`` if no member fails alone.

    The members of the unmeasured groups are served one at a time in canonical order, and
    dropped, to find the first that fails (the earlier groups' members served, as groups,
    as they would alone). Then the unmeasured groups' members before it are served again one
    at a time and measured, a group at a time, and of the measured groups only those members
    are kept. So the products judged are those one-at-a-time serving would reach, and the
    device still holds one group at a time. ``orders`` (the kernel path's) are the run
    positions of each measured group's rows, which follow the replay's blocks, not run order.
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
    for idx, r in zip(groups if orders is None else orders, rows):
        keep = [i for i in idx if i < end]  # a prefix: a group is in run order
        if orders is None or len(keep) == len(idx):
            out.append(r[:len(keep)])
        else:  # kernel rows are in block order
            out.append(r[[j for j, i in enumerate(idx) if i < end]])
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


def _rows_device(rows: torch.Tensor) -> Hashable:
    """The key :func:`_read_rows` groups a run's rows by: the device they sit on. A seam, so
    the Mac tests can treat CPU rows as several devices."""
    return rows.device


def _read_rows(rows: Sequence[torch.Tensor]) -> list[np.ndarray]:
    """Each run's rows as a float64 array on the host, in the order given, with one read per
    device: the rows of runs on one device (a run is on one device, F22) are joined there, read
    once and cut back, so a window may span devices. Values and order are the rows'. Kernel
    measurements still in flight are waited for first (``freivalds.drain``)."""
    freivalds.drain()
    by_device: dict[Hashable, list[int]] = {}
    for i, r in enumerate(rows):
        by_device.setdefault(_rows_device(r), []).append(i)
    out: list[np.ndarray | None] = [None] * len(rows)
    for idx in by_device.values():
        joined = rows[idx[0]] if len(idx) == 1 else torch.cat([rows[i] for i in idx])
        host = joined.cpu().double().numpy()
        base = 0
        for i in idx:
            out[i] = host[base:base + len(rows[i])]
            base += len(rows[i])
    return out  # type: ignore[return-value]


def _finish(batch: Sequence[_Rows], ctx: StepContext) -> list[tuple[ProductSpec, ProductMeasure]]:
    """The batch's products with their numbers, in canonical order, after one read of all
    their rows back to the host (in section ``5.measure``)."""
    present = [r for r in batch if r.rows is not None]
    if not present:
        return []
    with ctx.timed("5.measure"):
        read = iter(_read_rows([r.rows for r in present]))  # type: ignore[misc]
        out: list[tuple[ProductSpec, ProductMeasure]] = []
        for r in present:
            out += zip(r.specs, _finished(r, next(read).tolist(), eps_in=ctx.eps_in,
                                          eps_acc=ctx.eps_acc, guard=ctx.kappa_guard))
    return out


def _finish_columns(batch: Sequence[_Rows], ctx: StepContext, once: _ReadOnce | None = None
                    ) -> tuple[list[ProductSpec], MeasureColumns | None]:
    """:func:`_finish`'s products, as their specs and a :class:`MeasureColumns` in canonical
    order, finished in one vectorized step (``freivalds.finish_rows``). The unit pass hands
    its rows' one read in as ``once``; the first window asking for it makes the read."""
    present = [r for r in batch if r.rows is not None]
    if not present:
        return [], None
    with ctx.timed("5.measure"):
        read = ([once.host(r) for r in present] if once is not None
                else _read_rows([r.rows for r in present]))  # type: ignore[misc]
        host = read[0] if len(read) == 1 else np.concatenate(read)
        perm, specs, base = [], [], 0
        for r in present:
            perm.append(base + np.argsort(r.order))
            specs += r.specs
            base += len(r.order)
        canonical = np.concatenate(perm)
        cols = finish_rows(host[canonical], q=np.array([s.q for s in specs]),
                           n=np.array([s.a_shape[0] for s in specs]), eps_in=ctx.eps_in,
                           eps_acc=ctx.eps_acc, guard=ctx.kappa_guard,
                           biased=np.array([s.bias is not None for s in specs]))
    return specs, cols


def _first_failing(specs: Sequence[ProductSpec], classes: Sequence[str], cols: MeasureColumns,
                   ctx: StepContext, bands: Bands) -> int:
    """The index of the first product :func:`_judge_product` would reject, ``len(specs)`` if
    none: the same three tests as masks (finiteness, test 1, test 2), a product failing if any
    does."""
    finite = (np.isfinite(cols.p_norm) & np.isfinite(cols.u) & np.isfinite(cols.unit)
              & np.isfinite(cols.residuals).all(axis=1))
    if ctx.kappa_guard:
        finite &= np.isfinite(cols.nu) & np.isfinite(cols.p_abs1)
    bad = ~finite
    if ctx.judge:
        if ctx.kappa_guard:
            kappa_max = {key: bands.kappa_for(key) for key in set(classes)}
            ceiling = np.array([kappa_max[key] for key in classes]) * cols.p_abs1
            bad |= ~(cols.nu <= ceiling)
        bad |= ~(cols.residuals <= bands.tau * cols.unit[:, None]).all(axis=1)
    return int(np.argmax(bad)) if bad.any() else len(specs)


def _judge_window(c: DeclaredComputation, batch: Sequence[_Rows], ctx: StepContext,
                  bands: Bands, once: _ReadOnce | None = None) -> Rejection | None:
    """Read the batch's numbers, then record and judge its products in canonical order. At a
    rejection the products after the failing one go to ``StepState.run_rest``, measured and
    never judged, for the record pass.

    The products before the first failing one are recorded in bulk, judged by masks. From the
    failing one on, :func:`_judge_product` runs product by product, so the rejection, and
    ``run_rest``, are the one-at-a-time path's.
    """
    specs, cols = _finish_columns(batch, ctx, once)
    if cols is None:
        return None
    table = product_classes(c)  # the specs are the inventory's own, so [m-1] is their class
    classes = ([table[spec.m - 1] for spec in specs] if table is not None
               else [product_class(c, spec) for spec in specs])
    first = _first_failing(specs, classes, cols, ctx, bands)
    ctx.stats.products.extend_columns(
        m=[s.m for s in specs[:first]], name=[s.name for s in specs[:first]],
        cls=classes[:first], kappa=cols.kappa[:first], normalized=cols.normalized[:first],
        q=[s.q for s in specs[:first]], p_norm=cols.p_norm[:first], nu=cols.nu[:first],
        p_abs1=cols.p_abs1[:first], u=cols.u[:first])
    for i in range(first, len(specs)):
        rej = _judge_product(c, specs[i], cols.measure(i), ctx, bands)
        if rej is not None:
            ctx.state.run_rest = [(specs[j], cols.measure(j)) for j in range(i + 1, len(specs))]
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
    freivalds.drain()  # kernel measurements in flight are done before the read
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
        g = _glue_gradient(grads, name, w_t.device)
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


def _glue_gradient(grads: Mapping[str, torch.Tensor], name: str,
                   device: torch.device) -> torch.Tensor:
    """Check 6b's ``G`` for glue weight ``name``, in fp32 (P6), on ``device``, the weight
    leaf's, where the update identity runs; the record pass takes it here too. The replay
    returns each gradient on its weight's unit device when sharded (FS32c), so the move is
    usually a no-op; it keeps a misplaced gradient working."""
    return grads[name].detach().to(device, torch.float32)


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
    rest_rows = ctx.state.run_rest_rows
    ctx.state.run_rest, ctx.state.run_rest_rows, ctx.state.run_rest_error = [], [], None
    for spec, mp in rest:  # measured by check 5 after the failing product, never judged
        _judge_product(c, spec, mp, ctx, bands)  # records; judge=False decides nothing
    for r, host in rest_rows:  # the unit pass's later windows, read once at its end
        for spec, mp in zip(r.specs, _finished(r, host.tolist(), eps_in=ctx.eps_in,
                                               eps_acc=ctx.eps_acc, guard=ctx.kappa_guard)):
            _judge_product(c, spec, mp, ctx, bands)
    if error is not None:  # serving or measuring a later product failed, as it did in check 5
        raise error
    view = TranscriptView(c, ctx.state.committed())
    done = {p.m for p in ctx.stats.products}
    scratch = MeasureScratch()
    try:
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
    finally:
        freivalds.drain()  # no kernel hold outlives the pass, whatever raised


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
            g = (_linear_gradient(c, view, name, w_t.device) if grads is None
                 else _glue_gradient(grads, name, w_t.device))
        _update_identity(ctx, check_id, name, w_t, w_next, g, c.eta, bands.tau_w_for(name),
                         scratch)
