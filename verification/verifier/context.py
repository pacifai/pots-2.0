"""What the per-step checks share: the step context, rejections, statistics and the error guard.

:class:`StepContext` holds the verifier's own knowledge for one step (step index, ``π(t)``,
``h_D`` and ``|D|``, the chained ``W_t`` hashes, ``k``, the unit roundoffs). Its
:class:`StepState` holds what earlier checks derived and later checks consume: check 2 writes
the recomputed root, leaf hashes and :class:`CommittedLeaves`, check 5 leaves its ``Replay`` for
check 6b. Every check-5 and check-6 number goes into :class:`StepStats`, judged or not, which is
what A11's calibration (P10b) and A12's per-step logs read.

:func:`_guard` maps a prover-data error to a ``"malformed"`` rejection at the check that read
the leaf (``checks.py`` docstring, "Errors").
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from setup.records import Record
from verification.commitment.leaves import leaf_hash
from verification.commitment.merkle import DIGEST_SIZE
from verification.parameters import UNIT_ROUNDOFF
from verification.transcript.errors import (
    LeafShapeError,
    StoreMutationError,
    TranscriptFormatError,
)

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation, Replay

__all__ = [
    "PROVER_DATA_ERRORS",
    "Section",
    "Rejection",
    "RejectionKind",
    "ProductStat",
    "TensorStat",
    "StepStats",
    "StepState",
    "StepContext",
    "CommittedLeaves",
]

# A metrics seam (B6, ``runs/metrics.py``): ``section(name)`` wraps one part of a check. It
# observes only; with ``None`` the checks run exactly as without it.
Section = Callable[[str], AbstractContextManager[Any]]
_NO_SECTION = nullcontext()

# Every error the TranscriptStore docstring lists: TranscriptFormatError and its subclasses,
# EncodingError, NonFiniteError and encode_record's ValueError (RecordError for a token
# record) all derive from ValueError;
# IndexError derives from LookupError.
PROVER_DATA_ERRORS: tuple[type[BaseException], ...] = (ValueError, LookupError)


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
    section: Section | None = None  # B6 metrics seam; None runs the checks unobserved

    def timed(self, name: str) -> AbstractContextManager[Any]:
        """``section(name)``, or a no-op when no metrics seam is set."""
        return _NO_SECTION if self.section is None else self.section(name)

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
