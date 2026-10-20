"""What the per-step checks share: the step context, rejections, statistics and the error guard.

:class:`StepContext` holds the verifier's own knowledge for one step (step index, ``π(t)``,
``h_D`` and ``|D|``, the chained ``W_t`` hashes, ``k``, the unit roundoffs). Its
:class:`StepState` holds what earlier checks derived and later checks consume: check 2's read
writes the validated claimed root, the leaf hashes and :class:`CommittedLeaves`, check 5 leaves
its ``Replay`` for check 6b, and check 2's root comparison, last in the step, writes the
recomputed root (F15a in ``DECISIONS_FULL_SCALE.md``, task C7). Every check-5 and check-6
number goes into :class:`StepStats`, judged or not, which is what A11's calibration (P10b) and
A12's per-step logs read.

:func:`_guard` maps a prover-data error to a ``"malformed"`` rejection at the check that read
the leaf (``checks.py`` docstring, "Errors").
"""

from __future__ import annotations

import functools
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from setup.records import Record
from verification.commitment.leaves import (
    HashStats,
    leaf_hash,
    leaf_hashes_of,
    leaf_hashes_until_error,
)
from verification.commitment.merkle import DIGEST_SIZE
from verification.parameters import UNIT_ROUNDOFF
from verification.transcript.errors import (
    LeafShapeError,
    StoreMutationError,
    TranscriptFormatError,
)
from verification.transcript.reader import _ALIGN, stack_block

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation, ProductSpec, Replay
    from verification.verifier.matmul_check.freivalds import ProductMeasure

__all__ = [
    "PROVER_DATA_ERRORS",
    "Section",
    "no_section",
    "Rejection",
    "RejectionKind",
    "ProductStat",
    "ProductStats",
    "TensorStat",
    "StepStats",
    "StepState",
    "StepContext",
    "CommittedLeaves",
]

# A metrics seam (B6, ``runs/metrics.py``): ``section(name)`` wraps one part of a check. It
# observes only; with ``None`` the checks run exactly as without it. ``prover/step.py`` declares
# the same alias: the verifier may not import from the prover (invariant 1, test_layering.py).
Section = Callable[[str], AbstractContextManager[Any]]
_NO_SECTION = nullcontext()


def no_section(name: str) -> AbstractContextManager[Any]:
    """The seam when no metrics are taken: a shared no-op context manager."""
    return _NO_SECTION

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

    C1 (A11) also needs each product's scale: ``q`` (its contracted dimension), ``‖P_m‖_F``,
    ``ν_m``, ``‖|P_m|·1‖`` and the term scale ``U_m`` (F18). ``q``, ``‖P_m‖_F`` and ``U_m``
    set the product's ``e_m`` (``sizing.product_e_m``). The realized floor is
    ``f_achieved(e_m)·‖P_m‖_F``, and a product with ``‖|P|·1‖ = 0`` but ``ν ≠ 0`` is one the
    guard rejects by construction. They default to unknown (``q = 0``, NaN) and stay out of
    equality, so a NaN read back from a file can't make two equal records differ.
    """

    m: int
    name: str
    cls: str
    kappa: float
    normalized: tuple[float, ...]
    q: int = field(default=0, compare=False)
    p_norm: float = field(default=math.nan, compare=False)
    nu: float = field(default=math.nan, compare=False)
    p_abs1: float = field(default=math.nan, compare=False)
    u: float = field(default=math.nan, compare=False)


@dataclass(frozen=True)
class TensorStat:
    """Check 6's number for one weight: ``ρ_max = max_i |R_i| / (ε_W·(|W_t,i| + |η·G_i|))``."""

    weight: str
    check_id: str  # "6a" or "6b"
    rho_max: float


class _Block:
    """``n`` products' :class:`ProductStat` fields as columns (``normalized`` is ``[n, k]``)."""

    __slots__ = ("m", "name", "cls", "kappa", "normalized", "q", "p_norm", "nu", "p_abs1", "u")

    def __init__(self, m: Sequence[int], name: Sequence[str], cls: Sequence[str],
                 kappa: np.ndarray, normalized: np.ndarray, q: Sequence[int],
                 p_norm: np.ndarray, nu: np.ndarray, p_abs1: np.ndarray, u: np.ndarray) -> None:
        self.m, self.name, self.cls, self.q = m, name, cls, q
        self.kappa, self.normalized, self.p_norm = kappa, normalized, p_norm
        self.nu, self.p_abs1, self.u = nu, p_abs1, u

    def __len__(self) -> int:
        return len(self.m)

    def stat(self, i: int) -> ProductStat:
        return ProductStat(int(self.m[i]), self.name[i], self.cls[i], float(self.kappa[i]),
                           tuple(self.normalized[i].tolist()), q=int(self.q[i]),
                           p_norm=float(self.p_norm[i]), nu=float(self.nu[i]),
                           p_abs1=float(self.p_abs1[i]), u=float(self.u[i]))

    def stats(self) -> list[ProductStat]:
        return [ProductStat(m, name, cls, kappa, tuple(norm), q=q, p_norm=p_norm, nu=nu,
                            p_abs1=p_abs1, u=u)
                for m, name, cls, kappa, norm, q, p_norm, nu, p_abs1, u in zip(
                    self.m, self.name, self.cls, self.kappa.tolist(),
                    self.normalized.tolist(), self.q, self.p_norm.tolist(), self.nu.tolist(),
                    self.p_abs1.tolist(), self.u.tolist())]


def _readonly(a: np.ndarray) -> np.ndarray:
    a = a.view()
    a.flags.writeable = False
    return a


def _cat(parts: list[np.ndarray]) -> np.ndarray:
    return _readonly(parts[0] if len(parts) == 1 else np.concatenate(parts))


class ProductColumns:
    """All of a step's products as columns, in record order (:meth:`ProductStats.columns`).

    ``m`` and ``q`` are int64 ``[N]``; ``kappa``, ``p_norm``, ``nu``, ``p_abs1`` and ``u``
    float64 ``[N]``; ``normalized`` float64 ``[N, k]`` (``[0, 0]`` when empty); ``name`` and
    ``cls`` are lists of str. The arrays are read-only and may share memory with the stored
    blocks."""

    __slots__ = ("m", "name", "cls", "kappa", "normalized", "q", "p_norm", "nu", "p_abs1", "u",
                 "_codes")

    def __init__(self, m: np.ndarray, name: list[str], cls: list[str], kappa: np.ndarray,
                 normalized: np.ndarray, q: np.ndarray, p_norm: np.ndarray, nu: np.ndarray,
                 p_abs1: np.ndarray, u: np.ndarray) -> None:
        self.m, self.name, self.cls, self.q = m, name, cls, q
        self.kappa, self.normalized, self.p_norm = kappa, normalized, p_norm
        self.nu, self.p_abs1, self.u = nu, p_abs1, u
        self._codes: tuple[np.ndarray, tuple[str, ...]] | None = None

    def __len__(self) -> int:
        return len(self.m)

    def class_codes(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """``(codes, classes)``: the distinct classes in order of first appearance, and each
        product's index into them (int64 ``[N]``)."""
        if self._codes is None:
            index: dict[str, int] = {}
            codes = np.fromiter((index.setdefault(c, len(index)) for c in self.cls),
                                dtype=np.int64, count=len(self.cls))
            self._codes = (_readonly(codes), tuple(index))
        return self._codes


class ProductStats:
    """A step's :class:`ProductStat` records, in the order they were added. It behaves as the
    list it replaces: ``append``, ``extend``, ``clear``, ``len``, truthiness, iteration,
    indexing (negative too; a slice gives a list), item assignment, ``+`` and ``==``.

    Check 5 adds a window's products in bulk (:meth:`extend_columns`), which stores columns and
    builds no :class:`ProductStat`. They are built when needed: one by indexing, all of them
    (and kept) by iterating. A stat read back equals the one a scalar ``append`` would hold,
    field for field, with Python floats and ints.
    """

    _cols: ProductColumns | None = None  # :meth:`columns`' cache, dropped by every change

    def __init__(self, stats: Iterable[ProductStat] = ()) -> None:
        self._segs: list[ProductStat | _Block] = []
        self._ends: list[int] = []  # _ends[i]: the count of products up to and with _segs[i]
        self.extend(stats)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def append(self, stat: ProductStat) -> None:
        self._cols = None
        self._segs.append(stat)
        self._ends.append(len(self) + 1)

    def extend(self, stats: Iterable[ProductStat]) -> None:
        for stat in stats:
            self.append(stat)

    def extend_columns(self, *, m: Sequence[int], name: Sequence[str], cls: Sequence[str],
                       kappa: np.ndarray, normalized: np.ndarray, q: Sequence[int],
                       p_norm: np.ndarray, nu: np.ndarray, p_abs1: np.ndarray,
                       u: np.ndarray) -> None:
        """Append ``len(m)`` products from columns: ``kappa``, ``p_norm``, ``nu``, ``p_abs1``
        and ``u`` float64 ``[n]``, ``normalized`` float64 ``[n, k]``, the rest sequences."""
        if len(m):
            self._cols = None
            self._segs.append(_Block(m, name, cls, kappa, normalized, q, p_norm, nu, p_abs1, u))
            self._ends.append(len(self) + len(m))

    def clear(self) -> None:
        self._cols = None
        self._segs.clear()
        self._ends.clear()

    def _flatten(self) -> None:
        """Replace every block by its :class:`ProductStat` objects."""
        if any(isinstance(g, _Block) for g in self._segs):
            segs = [x for g in self._segs for x in (g.stats() if isinstance(g, _Block) else (g,))]
            cols = self._cols  # the records don't change, so neither do their columns
            self.clear()
            self.extend(segs)
            self._cols = cols

    def __iter__(self) -> Iterator[ProductStat]:
        self._flatten()
        return iter(self._segs)  # type: ignore[arg-type]

    def _at(self, i: int) -> tuple[int, int]:
        n = len(self)
        if not -n <= i < n:
            raise IndexError("ProductStats index out of range")
        i %= n
        seg = bisect_right(self._ends, i)
        return seg, i - (self._ends[seg - 1] if seg else 0)

    def __getitem__(self, i: int | slice) -> Any:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        seg, off = self._at(i)
        g = self._segs[seg]
        return g.stat(off) if isinstance(g, _Block) else g

    def __setitem__(self, i: int, stat: ProductStat) -> None:
        self._at(i)  # raises IndexError out of range
        self._flatten()
        self._cols = None
        self._segs[i] = stat

    def columns(self) -> ProductColumns:
        """Every record as columns (:class:`ProductColumns`), built from the block columns and
        the single records without making a :class:`ProductStat`, and kept until the next
        change. Values are those a :class:`ProductStat` would hold, as float64 and int64. It
        raises ``ValueError`` when the records' ``normalized`` widths differ (a step's never do)."""
        if self._cols is None:
            self._cols = self._build_columns()
        return self._cols

    def _build_columns(self) -> ProductColumns:
        m: list[np.ndarray] = []
        q: list[np.ndarray] = []
        name: list[str] = []
        cls: list[str] = []
        f: dict[str, list[np.ndarray]] = {key: [] for key in
                                           ("kappa", "normalized", "p_norm", "nu", "p_abs1", "u")}
        run: list[ProductStat] = []

        def flush() -> None:
            if run:
                m.append(np.array([s.m for s in run], dtype=np.int64))
                q.append(np.array([s.q for s in run], dtype=np.int64))
                name.extend(s.name for s in run)
                cls.extend(s.cls for s in run)
                for key in f:
                    f[key].append(np.array([getattr(s, key) for s in run], dtype=np.float64))
                run.clear()

        for g in self._segs:
            if isinstance(g, _Block):
                flush()
                m.append(np.asarray(g.m, dtype=np.int64))
                q.append(np.asarray(g.q, dtype=np.int64))
                name.extend(g.name)
                cls.extend(g.cls)
                for key in f:
                    f[key].append(np.asarray(getattr(g, key), dtype=np.float64))
            else:
                run.append(g)
        flush()
        if not m:
            empty, ints = (_readonly(np.empty(0, dtype=d)) for d in (np.float64, np.int64))
            return ProductColumns(ints, name, cls, empty,
                                  _readonly(np.empty((0, 0), dtype=np.float64)), ints,
                                  empty, empty, empty, empty)
        return ProductColumns(_cat(m), name, cls, _cat(f["kappa"]), _cat(f["normalized"]),
                              _cat(q), _cat(f["p_norm"]), _cat(f["nu"]), _cat(f["p_abs1"]),
                              _cat(f["u"]))

    def __bool__(self) -> bool:
        return bool(self._segs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProductStats):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other: Iterable[Any]) -> list[Any]:
        return [*self, *other]

    def __radd__(self, other: list[Any]) -> list[Any]:
        return [*other, *self]

    def __iadd__(self, other: Iterable[ProductStat]) -> ProductStats:
        self.extend(other)
        return self

    def copy(self) -> ProductStats:
        """An independent container with the same records (the columns are shared, read-only)."""
        dup = ProductStats()
        dup._segs, dup._ends = list(self._segs), list(self._ends)
        dup._cols = self._cols
        return dup

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_cols"}

    __copy__ = copy

    def __repr__(self) -> str:
        return repr(list(self))


def first_zero_values(vals: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                      peak: np.ndarray) -> np.ndarray:
    """``peak`` (each segment ``[starts[i], ends[i])`` of ``vals``' maximum) with every zero
    peak replaced by the segment's first zero, as Python ``max`` keeps the first of equal items
    (a ``-0.0`` before a ``0.0``) and ``fmax`` may return either."""
    peak = peak.copy()
    zeros = np.flatnonzero(vals == 0)
    if len(zeros):
        at = np.searchsorted(zeros, starts)
        for i in np.flatnonzero(peak == 0).tolist():
            peak[i] = vals[zeros[at[i]]]  # a zero peak has a zero in its segment
    return peak


def _py_max(a: np.ndarray) -> float:
    """``max(a.tolist())`` of a non-empty float64 vector, as a Python float. Python's ``max``
    keeps the first item unless a later one is greater, so a leading NaN stays and a later NaN
    is skipped, and among equal items the first (a ``-0.0`` before ``0.0``) stays."""
    first = float(a[0])
    if first != first:
        return first
    v = float(np.fmax.reduce(a))
    return float(a[int(np.argmax(a == 0))]) if v == 0 else v


@dataclass
class StepStats:
    products: ProductStats = field(default_factory=ProductStats)
    tensors: list[TensorStat] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.products, ProductStats):
            self.products = ProductStats(self.products)

    def max_normalized(self) -> float:
        """The largest normalized residual, ``0.0`` with no products. Ordered as Python's ``max``
        over the floats is, so a NaN first in its row, or first among the rows' maxima, wins and
        any other NaN is ignored."""
        if not self.products:
            return 0.0
        norm = self.products.columns().normalized
        rows = np.fmax.reduce(norm, axis=1)  # NaN-ignoring; a row starting with NaN is NaN
        rows[np.isnan(norm[:, 0])] = math.nan
        flat = norm.reshape(-1)
        w = norm.shape[1]
        rows = first_zero_values(flat, np.arange(len(rows)) * w, np.arange(1, len(rows) + 1) * w,
                                 rows)
        return _py_max(rows)

    def max_kappa(self) -> float:
        """The largest ``κ`` (``0.0`` with no products), ordered as :meth:`max_normalized`."""
        return _py_max(self.products.columns().kappa) if self.products else 0.0

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


def _typed(items: Iterable[tuple[int, Any]]) -> Iterator[tuple[int, Any]]:
    for i, obj in items:
        _leaf_tensors(obj)
        yield i, obj


def _hashes(c: DeclaredComputation, items: Iterable[tuple[int, Any]],
            stats: HashStats | None = None) -> list[bytes]:
    """:func:`_hash` of each ``(index, obj)``, in order, hashed in parallel
    (``leaf_hashes_of``). The type check runs with the validation, so the first error is
    the one a ``_hash`` loop raises first. ``stats`` (optional) collects where the time went;
    its ``read_s`` then includes the type check, since that sits in the iterator."""
    return leaf_hashes_of(c, _typed(items), stats)


def _hashes_until_error(c: DeclaredComputation, items: Iterable[tuple[int, Any]]
                        ) -> tuple[list[bytes], Exception | None]:
    """:func:`_hashes` up to the first error, which is returned, not raised
    (``leaf_hashes_until_error``): the digests before it and the error a ``_hash`` loop
    raises first, or ``None``."""
    return leaf_hashes_until_error(c, _typed(items))


class CommittedLeaves:
    """The leaves check 2 hashed, served to every later check (a ``LeafReader``).

    Each object is the one whose bytes entered the recomputed root. Its tensors' ``_version``
    is recorded before hashing and compared on every read, so an in-place write through any
    alias after hashing raises :class:`StoreMutationError` instead of serving changed bytes.
    """

    def __init__(self) -> None:
        self._objs: list[Any] = []
        self._versions: list[tuple[int, ...]] = []
        # (first, stride, n) -> proof of a range read, or None (tensor_run)
        self._runs: dict[tuple[int, int, int], Any] = {}

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

    def tensor_block(self, indices: Sequence[int], own: bool = False) -> torch.Tensor:
        """:func:`stack_block` of the tensor leaves at ``indices``, each read as :meth:`leaf`
        reads it: the index and ``_version`` checked per leaf, in order, so the first bad one
        raises as in a loop. (Views of one storage share a version counter, so a write
        through any alias moves every member's; the per-leaf check still compares each
        against its own recorded value.) A leaf that is not a tensor takes the plain path."""
        objs, versions = self._objs, self._versions
        n, ts = len(objs), []
        for i in indices:
            if not (isinstance(i, int) and 0 <= i < n):
                self.leaf(i)  # raises the verifier-bug error
            t = objs[i]
            if not isinstance(t, torch.Tensor):
                return stack_block([self.leaf(j) for j in indices], own)
            if t._version != versions[i][0]:
                raise StoreMutationError(f"leaf {i} changed in place after check 2 hashed it")
            ts.append(t)
        return stack_block(ts, own)

    def tensor_run(self, first: int, stride: int, n: int, own: bool = False) -> torch.Tensor:
        """:meth:`tensor_block` of the leaves ``first + stride·k``, ``k < n``, without its
        per-leaf checks of shape, offset and storage when they are one bmm's members.

        The first read of a range proves, once, that the leaves are equal-shape contiguous
        tensors at storage offsets ``off + k·numel`` of one storage, 64-byte aligned (the
        conditions of :func:`stack_block`), and that every recorded ``_version`` is the same
        number. That structure can't change, so the proof is kept. Every read still compares
        each leaf's own ``_version`` to the recorded one: sharing a storage does not mean
        sharing a version counter (``torch.load`` and ``set_`` give each tensor its own
        counter over one storage, so a write through one leaf would leave the others' unmoved).
        Any mismatch, and any range the proof fails on, takes :meth:`tensor_block`, which
        raises for the first changed leaf, with its index, exactly as a loop of :meth:`leaf`
        does. The proofs are kept on this object, so none outlives the step's leaves.
        """
        key = (first, stride, n)
        last = first + stride * (n - 1)
        runs = self._runs
        if key not in runs:
            runs[key] = self._prove_run(first, stride, n)
        desc = runs[key]
        if desc is not None:
            t0, version, shape, strides, off = desc
            if all(t._version == version for t in self._objs[first:last + 1:stride]):
                out = t0.as_strided((n, *shape), strides, off)
                return out.clone() if own else out
        return self.tensor_block([first + stride * k for k in range(n)], own)

    def _prove_run(self, first: int, stride: int, n: int
                   ) -> tuple[torch.Tensor, int, Any, tuple[int, ...], int] | None:
        """The proof :meth:`tensor_run` keeps for a range, or ``None`` if it can't be made.
        A range inside a proved one is read off that proof, without visiting its leaves."""
        objs, versions = self._objs, self._versions
        last = first + stride * (n - 1)
        if n < 1 or stride < 1 or not (isinstance(first, int) and 0 <= first and last < len(objs)):
            return None
        for (f0, s0, n0), d0 in self._runs.items():
            if (d0 is not None and s0 == stride and first >= f0 and last <= f0 + s0 * (n0 - 1)
                    and (first - f0) % stride == 0):
                t0, version, shape, strides, off = d0
                k = (first - f0) // stride
                t = objs[first]
                numel = strides[0]
                if t.storage_offset() == off + k * numel and t.data_ptr() % _ALIGN == 0:
                    return t, version, shape, strides, t.storage_offset()
        t0 = objs[first]
        if (not isinstance(t0, torch.Tensor) or t0.numel() == 0
                or not t0.is_contiguous() or t0.requires_grad or t0.data_ptr() % _ALIGN):
            return None
        shape, dtype, numel, off = t0.shape, t0.dtype, t0.numel(), t0.storage_offset()
        device, ptr, version = t0.device, t0.untyped_storage().data_ptr(), versions[first]
        if len(version) != 1:
            return None
        for k in range(n):
            i = first + stride * k
            t = objs[i]
            if (not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != dtype
                    or t.device != device  # members of one storage share a device; a mix is no run
                    or t.storage_offset() != off + k * numel or not t.is_contiguous()
                    or t.untyped_storage().data_ptr() != ptr or t.requires_grad
                    or versions[i] != version or t._version != version[0]):
                return None
        strides, run = [], numel
        for d in shape:
            run //= d
            strides.append(run)
        return t0, version[0], shape, (numel, *strides), off


@dataclass
class StepState:
    """What a check derives for a later one; empty at step start."""

    early_hashes: dict[int, bytes] = field(default_factory=dict)  # what checks 4 and 7 hashed
    # The prover's h, once check 2's read has shown it is a digest; check 5 keys on it (F15a).
    claimed_root: bytes | None = None
    leaf_hashes: list[bytes] | None = None
    recomputed_root: bytes | None = None  # set by check 2's root comparison, the step's last
    leaves: CommittedLeaves | None = None  # check 2's leaves, the only reader after check 2
    replay: Replay | None = None  # check 5's replay, reused by 6b
    # When check 5 rejects: the products it measured after the failing one and never judged
    # (the rest of its judging window), as it measured them, and the error serving or
    # measuring a later product raised, if any. The record pass records these numbers rather
    # than measuring the products again.
    run_rest: list[tuple[ProductSpec, ProductMeasure]] = field(default_factory=list)
    run_rest_error: Exception | None = None

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
    # False skips check 5's test 1 (the κ guard). Only C1's cost split uses it, to time check 5
    # without the guard; κ, ν and ‖|P|·1‖ are then recorded as NaN.
    kappa_guard: bool = True
    state: StepState = field(default_factory=StepState)
    stats: StepStats = field(default_factory=StepStats)
    section: Section | None = None  # B6 metrics seam; None runs the checks unobserved

    def timed(self, name: str) -> AbstractContextManager[Any]:
        """``section(name)``, or a no-op when no metrics seam is set."""
        return _NO_SECTION if self.section is None else self.section(name)

    def hash_stats(self) -> HashStats | None:
        """A fresh :class:`HashStats` when the seam can ``add`` a measured duration, else
        ``None``: a run with no metrics, or a plain callable seam, collects nothing."""
        return HashStats() if hasattr(self.section, "add") else None

    def stamp(self, name: str, run: int = -1, group: int = -1,
              window: int = -1) -> AbstractContextManager[Any]:
        """``section.stamp(name, run, group, window)`` (a profiling stamp,
        ``runs/metrics.py``), or a no-op when the seam is ``None`` or a plain callable with no
        ``stamp``."""
        f = getattr(self.section, "stamp", None)
        return _NO_SECTION if f is None else f(name, run, group, window)

    def stamper(self) -> Callable[..., AbstractContextManager[Any]] | None:
        """The seam's stamp function itself, for a loop that stamps many times (check 5's
        serving), or ``None`` when there is none: a stamp then costs no lookup or forwarding."""
        return getattr(self.section, "stamp", None)

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
