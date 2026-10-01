"""The per-step checks of spec §6, as functions of ``(store, computation, context, bands)``.

Each per-step check returns ``None`` on acceptance or a :class:`Rejection` ``(step, check_id,
detail)``. :data:`DEFAULT_ORDER` is the one evaluation order of every run, ``4 → 7 → 2 → 6a →
5 → 6b`` (S6c, revised at stage 4). The verifier is the same code in every run (S6a); nothing
here has a fault hook.

The checks share one :class:`StepContext`. Its inputs are the verifier's own knowledge (step
index, ``π(t)``, ``h_D`` and ``|D|``, the chained ``W_t`` hashes, ``k``, the unit roundoffs).
Its :class:`StepState` holds what earlier checks derived and later checks consume: check 2
writes the recomputed root and leaf hashes, check 5 leaves its :class:`Replay` for check 6b.
Writing there is a check's only side effect. Every check-5 and check-6 number goes into
:class:`StepStats`, judged or not, which is what A11's calibration (P10b) and A12's per-step
logs read.

**Byte binding.** Every check after check 2 reads the leaves check 2 hashed, never the store
again: check 2 reads each leaf once, hashes it and keeps the object in a
:class:`CommittedLeaves` reader, guarded by ``_version``. Checks 4 and 7 run before check 2
and read the store themselves, so they record the hashes they saw and check 2 rejects if its
own read of any of those leaves hashes differently. A store that serves one set of bytes to an
early check and another to check 2 is rejected at 2.

**Errors.** Prover data that fails to read, decode, hash or validate raises one of the errors
the ``TranscriptStore`` docstring lists. :func:`_guard` turns each into a ``"malformed"``
rejection at the check that read the leaf (store.py, "the rule for A5"). The guard covers
store reads, leaf hashing and validation only. After check 2 the verifier runs its own code
(replay, operands, glue gradients) on validated leaves, so an error there is a verifier bug
and propagates; only a ``TranscriptFormatError`` (a cached leaf mutated in place) is mapped
there. A violated verifier-side precondition raises ``RuntimeError``.

Arithmetic is at the working precision, fp32 (P6). Every band comparison is written as
``not (x <= bound)``, so a NaN that slips through glue rejects instead of passing, and every
quantity a band compares must be finite. Check 5's norms are scaled so a finite leaf can't
overflow them (:func:`_safe_norm`); an overflow that remains, in a matmul output or in check 6's
``η·G``, rejects rather than passing ``inf <= inf``.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import blake3
import torch

from .challenges import challenge_matrix
from .computation import DeclaredComputation, ProductSpec, Replay, TranscriptView
from .config import SIGMA_R, TAU_W0, UNIT_ROUNDOFF, Z
from .encoding import Record
from .merkle import DIGEST_SIZE, merkle_root, verify_path
from .sizing import e_m
from .store import (
    LeafShapeError,
    StoreMutationError,
    TranscriptFormatError,
    TranscriptStore,
    leaf_hash,
)

__all__ = [
    "DEFAULT_ORDER",
    "PROVER_DATA_ERRORS",
    "KAPPA_PROVISIONAL",
    "Rejection",
    "Bands",
    "ProductStat",
    "TensorStat",
    "StepStats",
    "StepState",
    "StepContext",
    "CommittedLeaves",
    "product_class",
    "check_4_batch_anchor",
    "check_7_chaining",
    "check_2_commitment",
    "check_6a_linear_update",
    "check_5_matmuls",
    "check_6b_glue_update",
    "CHECKS",
]

DEFAULT_ORDER: tuple[str, ...] = ("4", "7", "2", "6a", "5", "6b")

# Every error the TranscriptStore docstring lists: TranscriptFormatError and its subclasses,
# EncodingError, NonFiniteError and encode_record's ValueError all derive from ValueError;
# IndexError derives from LookupError.
PROVER_DATA_ERRORS: tuple[type[BaseException], ...] = (ValueError, LookupError)

# A10's "loose κ": a ceiling far above any honest cancellation factor, so it guards only
# against the pathological regime of P3.c until A11 calibrates κ_max per class.
KAPPA_PROVISIONAL: float = 1e4

BAND_FILE_VERSION = 1


RejectionKind = Literal["malformed", "failed"]


@dataclass(frozen=True)
class Rejection:
    """A rejection, ``(step, check_id, detail)``; the S6b oracle compares the first two.

    ``kind`` is ``"malformed"`` when prover data failed to read, decode or validate, and
    ``"failed"`` when well-formed data failed the check itself. A cheat whose expected point is
    a check failure requires ``"failed"``, so a fault that only breaks the encoding doesn't
    count as caught by the check it targets.
    """

    step: int
    check_id: str
    detail: str
    kind: RejectionKind = "failed"


# ---- bands ------------------------------------------------------------------------------


def _frozen(m: Mapping[str, float] | None) -> Mapping[str, float]:
    return MappingProxyType({str(k): float(v) for k, v in (m or {}).items()})


@dataclass(frozen=True)
class Bands:
    """The tolerances of checks 5 and 6 (P3, P5, P10).

    - ``tau``: the one dimensionless ``τ`` of check 5 (P3.a, P3.b).
    - ``kappa_max``: the default cancellation ceiling; ``kappa_classes`` overrides it per
      product class (P3.c), keyed by :func:`product_class`.
    - ``tau_w``: the default ``τ_W`` of check 6; ``tau_w_tensors`` overrides it per weight
      (P5: ``τ_W = max(τ_W⁰, 2·ρ_max)``, so every value is at least ``TAU_W0``).
    - ``source``: the BLAKE3 hex of the band file the bands were loaded from, or
      ``"provisional"`` for bands built in code, which judge only on opt-in (P10a).
      P10a's harness asserts that every run records the same value.

    A11 writes the band file with :meth:`to_json` and every later run loads it with
    :meth:`from_file`. A11 may extend the schema; ``stats`` is carried through unread.
    """

    tau: float
    kappa_max: float
    tau_w: float = TAU_W0
    kappa_classes: Mapping[str, float] = field(default_factory=dict)
    tau_w_tensors: Mapping[str, float] = field(default_factory=dict)
    source: str = "provisional"
    stats: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kappa_classes", _frozen(self.kappa_classes))
        object.__setattr__(self, "tau_w_tensors", _frozen(self.tau_w_tensors))
        object.__setattr__(self, "stats", MappingProxyType(dict(self.stats)))
        if not (math.isfinite(self.tau) and self.tau > 0):
            raise ValueError(f"tau must be positive and finite, got {self.tau}")
        for name, v in [("kappa_max", self.kappa_max), *self.kappa_classes.items()]:
            if not (math.isfinite(v) and v >= 1):
                raise ValueError(f"κ_max {name} = {v}: a cancellation factor is ≥ 1 (P3.c)")
        for name, v in [("tau_w", self.tau_w), *self.tau_w_tensors.items()]:
            if not (math.isfinite(v) and v >= TAU_W0):
                raise ValueError(f"τ_W {name} = {v} is below the analytic floor {TAU_W0} (P5a)")

    @classmethod
    def provisional(cls) -> Bands:
        """A10's provisional bands: ``τ = Z·1 = 8``, a loose ``κ``, ``τ_W = τ_W⁰ = 4``.

        ``τ = Z·s_h`` with ``s_h`` taken as the error model's nominal 1 (P3.a). None of the
        three depends on scale or precision. Their ``source`` is ``"provisional"``, which the
        verifier refuses unless the caller opts in: they never judge a cheat run (P10a).
        """
        return cls(tau=Z * 1.0, kappa_max=KAPPA_PROVISIONAL, tau_w=TAU_W0)

    def check_keys(self, c: DeclaredComputation) -> None:
        """Every κ class and τ_W tensor the bands name must exist in ``C``, so a misspelt key
        can't silently fall back to the default."""
        classes = {product_class(c, p) for p in c.products}
        unknown = sorted(set(self.kappa_classes) - classes)
        if unknown:
            raise ValueError(f"bands name κ classes that match no product of C: {unknown}")
        unknown = sorted(set(self.tau_w_tensors) - set(c.weight_names))
        if unknown:
            raise ValueError(f"bands name τ_W tensors that match no weight of C: {unknown}")

    def kappa_for(self, cls_key: str) -> float:
        return self.kappa_classes.get(cls_key, self.kappa_max)

    def tau_w_for(self, weight: str) -> float:
        return self.tau_w_tensors.get(weight, self.tau_w)

    # ---- the band file (P10a). Outside every commitment, so JSON is fine (S9c) -------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BAND_FILE_VERSION,
            "tau": self.tau,
            "kappa_max": {"default": self.kappa_max, "classes": dict(self.kappa_classes)},
            "tau_w": {"default": self.tau_w, "tensors": dict(self.tau_w_tensors)},
            "stats": dict(self.stats),
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1).encode()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, source: str) -> Bands:
        if d.get("version") != BAND_FILE_VERSION:
            raise ValueError(f"band file version {d.get('version')!r}, "
                             f"expected {BAND_FILE_VERSION}")
        return cls(tau=float(d["tau"]),
                   kappa_max=float(d["kappa_max"]["default"]),
                   kappa_classes=d["kappa_max"].get("classes", {}),
                   tau_w=float(d["tau_w"]["default"]),
                   tau_w_tensors=d["tau_w"].get("tensors", {}),
                   source=source, stats=d.get("stats", {}))

    @classmethod
    def from_json(cls, b: bytes) -> Bands:
        """Bands from band-file bytes; ``source`` is the BLAKE3 hex of exactly these bytes."""
        return cls.from_dict(json.loads(b), source=blake3.blake3(b).hexdigest())

    @classmethod
    def from_file(cls, path: str | Path) -> Bands:
        return cls.from_json(Path(path).read_bytes())


# ---- per-step statistics (P10b, A12) ----------------------------------------------------


@dataclass(frozen=True)
class ProductStat:
    """Check 5's numbers for one product: ``κ_m`` and the ``k`` normalized residuals.

    ``normalized[j−1] = ‖A(B·r_j) − P·r_j‖ / (σ_r·e_m·‖P_m‖_F)``, the quantity compared with
    ``τ`` (P3.a). ``inf`` when the band's scale is zero and the residual isn't.
    """

    m: int
    name: str
    cls: str
    kappa: float
    normalized: tuple[float, ...]


@dataclass(frozen=True)
class TensorStat:
    """Check 6's number for one weight: ``ρ_max = max_i |R_i| / (ε_W·(|W_t,i| + |η·G_i|))``."""

    weight: str
    check_id: str  # "6a" or "6b"
    rho_max: float


@dataclass
class StepStats:
    products: list[ProductStat] = field(default_factory=list)
    tensors: list[TensorStat] = field(default_factory=list)

    def max_normalized(self) -> float:
        return max((max(p.normalized) for p in self.products), default=0.0)

    def max_kappa(self) -> float:
        return max((p.kappa for p in self.products), default=0.0)

    def by_class(self) -> dict[str, list[ProductStat]]:
        out: dict[str, list[ProductStat]] = {}
        for p in self.products:
            out.setdefault(p.cls, []).append(p)
        return out


# ---- the shared step context ------------------------------------------------------------


def _leaf_tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, Record):
        return [obj.ids, obj.targets, obj.mask]
    raise LeafShapeError(f"unsupported leaf object {type(obj).__name__}")


def _hash(c: DeclaredComputation, index: int, obj: Any) -> bytes:
    """``leaf_hash`` with a type check first. A leaf is a tensor or a ``Record``, so a foreign
    object raises a prover-data error here, not a ``TypeError`` or ``AttributeError`` from
    inside an encoder."""
    _leaf_tensors(obj)
    return leaf_hash(c, index, obj)


class CommittedLeaves:
    """The leaves check 2 hashed, served to every later check (a ``LeafReader``).

    Each object is the one whose bytes entered the recomputed root. Its tensors' ``_version``
    is recorded before hashing and compared on every read, so an in-place write through any
    alias after hashing raises :class:`StoreMutationError` instead of serving changed bytes.
    """

    def __init__(self) -> None:
        self._objs: list[Any] = []
        self._versions: list[tuple[int, ...]] = []

    def add(self, obj: Any) -> None:
        self._versions.append(tuple(t._version for t in _leaf_tensors(obj)))
        self._objs.append(obj)

    def __len__(self) -> int:
        return len(self._objs)

    def changed(self, index: int) -> bool:
        return tuple(t._version for t in _leaf_tensors(self._objs[index])) != self._versions[index]

    def leaf(self, index: int) -> Any:
        # Indices come from C, never from the prover, so a bad one is a verifier bug.
        if not (isinstance(index, int) and 0 <= index < len(self._objs)):
            raise RuntimeError(f"committed leaf index {index!r} outside 0..{len(self._objs) - 1}")
        if self.changed(index):
            raise StoreMutationError(f"leaf {index} changed in place after check 2 hashed it")
        return self._objs[index]


@dataclass
class StepState:
    """What a check derives for a later one; empty at step start."""

    early_hashes: dict[int, bytes] = field(default_factory=dict)  # what checks 4 and 7 hashed
    root: bytes | None = None  # check 2's recomputed h; check 5 keys its challenges on it
    leaf_hashes: list[bytes] | None = None
    leaves: CommittedLeaves | None = None  # check 2's leaves, the only reader after check 2
    replay: Replay | None = None  # check 5's replay, reused by 6b

    def committed(self) -> CommittedLeaves:
        if self.leaves is None:
            raise RuntimeError("this check reads check 2's committed leaves; run check 2 first")
        return self.leaves


@dataclass
class StepContext:
    """The verifier's own knowledge for one step. Nothing in it comes from the prover."""

    step: int
    indices: tuple[int, ...]  # π(t)
    h_D: bytes
    n_records: int  # |D|, from the data manifest (invariant 7)
    prev_w_hashes: tuple[bytes, ...]  # W_t must hash to these: W_0's, or step t−1's W_{t+1}
    chain_check_id: str  # "0" on the first step (base anchor), "7" after
    k: int
    eps_in: float  # unit roundoff of the operand format (C.operand_dtype)
    eps_acc: float  # unit roundoff of the accumulator (C.accumulator_dtype)
    eps_w: float  # unit roundoff of the weight format (P5)
    judge: bool = True  # False: calibration mode (P10b); checks 5 and 6 record, never judge
    state: StepState = field(default_factory=StepState)
    stats: StepStats = field(default_factory=StepStats)

    @classmethod
    def for_computation(cls, c: DeclaredComputation, **kw: Any) -> StepContext:
        """Fill the unit roundoffs from ``C``'s declared dtypes (P6) unless given."""
        kw.setdefault("eps_in", UNIT_ROUNDOFF[c.operand_dtype])
        kw.setdefault("eps_acc", UNIT_ROUNDOFF[c.accumulator_dtype])
        kw.setdefault("eps_w", UNIT_ROUNDOFF[c.weight_dtype])
        return cls(**kw)

    def reject(self, check_id: str, detail: str,
               kind: RejectionKind = "failed") -> Rejection:
        return Rejection(self.step, check_id, detail, kind)


class _Rejected(Exception):
    def __init__(self, rejection: Rejection) -> None:
        self.rejection = rejection


@contextmanager
def _guard(ctx: StepContext, check_id: str,
           errors: tuple[type[BaseException], ...] = PROVER_DATA_ERRORS) -> Iterator[None]:
    """Map a prover-data error raised inside the block to a ``"malformed"`` rejection.

    Wrap store reads, leaf hashing and validation only. Around the verifier's own code after
    check 2, pass ``_AFTER_COMMIT``.
    """
    try:
        yield
    except errors as e:
        raise _Rejected(ctx.reject(check_id, f"malformed prover data: "
                                             f"{type(e).__name__}: {e}", "malformed")) from e


_AFTER_COMMIT: tuple[type[BaseException], ...] = (TranscriptFormatError,)


def _checked(fn: Callable[..., Rejection | None]) -> Callable[..., Rejection | None]:
    @functools.wraps(fn)
    def run(*args: Any) -> Rejection | None:
        try:
            return fn(*args)
        except _Rejected as r:
            return r.rejection
    return run


def _is_digest(x: Any) -> bool:
    return isinstance(x, bytes) and len(x) == DIGEST_SIZE


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
    """
    cid = ctx.chain_check_id
    if len(ctx.prev_w_hashes) != c.n_w:
        raise RuntimeError(f"{len(ctx.prev_w_hashes)} chained hashes for {c.n_w} weights")
    for name, want in zip(c.weight_names, ctx.prev_w_hashes):
        with _guard(ctx, cid):
            got = _hash(c, c.w_t_index(name), store.leaf(c.w_t_index(name)))
        ctx.state.early_hashes[c.w_t_index(name)] = got
        if got != want:
            what = "the agreed W_0" if cid == "0" else f"W_{{t+1}} of step {ctx.step - 1}"
            return ctx.reject(cid, f"W_t[{name}] differs from {what}")
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
    """
    leaves, hashes = CommittedLeaves(), []
    with _guard(ctx, "2"):
        for i in range(c.n_leaves):  # the count comes from C (invariant 7)
            obj = store.leaf(i)
            leaves.add(obj)
            hashes.append(_hash(c, i, obj))
        claimed = store.root
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


def _update_identity(ctx: StepContext, check_id: str, name: str, w_t: torch.Tensor,
                     w_next: torch.Tensor, g: torch.Tensor, eta: float,
                     tau_w: float) -> Rejection | None:
    """P5: reject if any ``|R_i| > τ_W·ε_W·(|W_t,i| + |η·G_i|)``, ``R = W_{t+1} − (W_t − η·G)``.

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
    for name, m in c.linear_weights.items():
        with _guard(ctx, "6a", _AFTER_COMMIT):
            w_t, w_next, g = view.w_t(name), view.w_next(name), view.product(m)
        rej = _update_identity(ctx, "6a", name, w_t, w_next, g, c.eta, bands.tau_w_for(name))
        if rej is not None:
            return rej
    return None


# ---- check 5: matmul checks (P3) --------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def product_class(c: DeclaredComputation, spec: ProductSpec) -> str:
    """The calibration class of a product: its role in ``C`` (spec §9, P3.b, P3.c).

    ``C`` may declare classes with a ``product_class(spec)`` method. Otherwise the class is the
    product kind with the role, the weight name (or, for a weight-free product, the product
    name) with every digit run replaced by ``*``, so the same role in every layer is one class.
    """
    own = getattr(c, "product_class", None)
    if callable(own):
        return str(own(spec))
    role = spec.weight if spec.weight is not None else spec.name
    return f"{spec.kind.value}:{_DIGITS.sub('*', role)}"


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


@_checked
def check_5_matmuls(store: TranscriptStore, c: DeclaredComputation, ctx: StepContext,
                    bands: Bands) -> Rejection | None:
    """Both tests of spec §6 check 5 on every product, in canonical order; abort at the first fail.

    The challenges are keyed on ``ctx.state.root``, the root the verifier recomputed in check
    2, never on ``store.root``. Check 2 has shown the two equal, but keying on the verifier's
    own value means the PRF never consumes a prover-supplied byte (spec §5), and check 5
    cannot run before check 2 has. Operands and products come from check 2's leaves (check 3).

    Every norm is taken with :func:`_safe_norm`, so a finite product entry near ``2e19`` can't
    overflow ``‖P‖_F`` or the residual to ``inf``. Every quantity the two tests compare must
    still be finite, in either mode: a matmul output such as ``A(B·r)`` can itself exceed the
    fp32 maximum, and ``inf <= inf`` would accept an arbitrary forgery.
    """
    root = ctx.state.root
    if root is None:
        raise RuntimeError("check 5 needs check 2's recomputed root")
    leaves = ctx.state.committed()
    view = TranscriptView(c, leaves)
    with _guard(ctx, "5", _AFTER_COMMIT):
        replay = c.replay(leaves)
    ctx.state.replay = replay
    for spec in c.products:
        with _guard(ctx, "5", _AFTER_COMMIT):
            a, b = replay.operands(spec.m)
            p = view.product(spec.m)
        if (tuple(a.shape), tuple(b.shape)) != (spec.a_shape, spec.b_shape):
            raise RuntimeError(f"{spec.name}: replay operands {tuple(a.shape)}·{tuple(b.shape)} "
                               f"!= declared {spec.a_shape}·{spec.b_shape}")
        a, b = a.detach().to(torch.float32), b.detach().to(torch.float32)
        p = p.detach().to(torch.float32)
        cls_key = product_class(c, spec)
        with torch.no_grad():
            # Test 1, the cancellation guard (P3.c): ν_m ≤ κ_max·‖ |P_m|·1 ‖.
            ones = torch.ones(spec.width, 1, dtype=torch.float32)
            nu = float(_safe_norm(a.abs() @ (b.abs() @ ones)))
            p_abs1 = float(_safe_norm(p.abs() @ ones))
            kappa = nu / p_abs1 if p_abs1 > 0 else (1.0 if nu == 0 else float("inf"))
            # Test 2, the normalized residual (P3.a): ‖A(B·r) − P·r‖ ≤ τ·σ_r·e_m·‖P_m‖_F.
            r = challenge_matrix(root, spec.m, ctx.k, spec.width)
            res = _safe_norm(a @ (b @ r) - p @ r, dim=0).tolist()
            p_norm = float(_safe_norm(p))
            unit = SIGMA_R * e_m(spec.q, ctx.eps_in, ctx.eps_acc) * p_norm
        normalized = tuple(x / unit if unit > 0 else (0.0 if x == 0 else float("inf"))
                           for x in res)
        ctx.stats.products.append(ProductStat(spec.m, spec.name, cls_key, kappa, normalized))
        values = {"ν": nu, "‖|P|·1‖": p_abs1, "‖P‖_F": p_norm, "band unit": unit,
                  **{f"residual j={j}": x for j, x in enumerate(res, start=1)}}
        bad = [key for key, v in values.items() if not math.isfinite(v)]
        if bad:
            return ctx.reject("5", f"P_{spec.m} ({spec.name}): non-finite {', '.join(bad)}")
        if not ctx.judge:
            continue
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
    with _guard(ctx, "6b", _AFTER_COMMIT):
        grads = replay.glue_gradients()
    missing = [n for n in names if n not in grads]
    if missing:
        raise RuntimeError(f"replay returned no glue gradient for {missing}")
    for name in names:
        with _guard(ctx, "6b", _AFTER_COMMIT):
            w_t, w_next = view.w_t(name), view.w_next(name)
        g = grads[name].detach().to(torch.float32)
        rej = _update_identity(ctx, "6b", name, w_t, w_next, g, c.eta, bands.tau_w_for(name))
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
