"""C1's calibration: fit the bands on honest steps, check them, and pin them in the band file.

The verifier calibrates on its own honest run (P10a). At test scale the honest prover's steps
1–3 stand in for it. On those steps the exact checks run live, and checks 5 and 6 only record
their numbers (``Verifier(calibrate=True)``, P10b). This module turns the numbers into bands:

- :func:`fit` takes ``{step: StepStats}`` and returns a :class:`Calibration`:

  - ``s_h``, the largest class-wise RMS of check 5's normalized residual, and ``τ = z·s_h``
    (P3.a, P10d);
  - ``κ_max`` per class, :data:`KAPPA_MARGIN` times the largest honest cancellation factor of
    the class (P3.c);
  - ``τ_W = max(τ_W⁰, 2·ρ_max)`` per weight tensor (P5);
  - the concentration guard: the largest honest normalized residual must sit at or below
    ``τ/2``. A failure is reported, never fixed by raising ``z`` here.

- :func:`realized_floor` gives the floor each product actually gets, ``Φ_m = f_achieved·‖P_m‖_F``
  with ``f_achieved = c·τ·e_m·2^(N/k)``, as a fraction of the target ``‖P_m‖_F`` (P4). It also
  lists honest products with ``‖|P|·1‖ = 0`` but ``ν ≠ 0``, which the guard rejects by
  construction.
- :func:`band_file_bytes` writes the bands and C1's statistics as one JSON file.
  :func:`write_band_file` saves it, and :func:`load_bands` is how every later run reads it.
  ``Bands.source`` is the BLAKE3 hex of the file's bytes, and :func:`assert_same_band_source`
  is P10a's check that every run was judged by the same file.

Recomputing ``k`` with the measured ``τ`` is ``sizing.size_k(..., s_h=s_h)`` (P10d). Nothing
here runs a model; ``runs/calibrate.py`` does.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verification.verifier.bands import TAU_W0, Bands
from verification.verifier.context import StepStats
from verification.verifier.matmul_check.sizing import C_ANTI, Z, e_m, f_achieved
from verification.verifier.residuals import ClassSummary, class_summary

__all__ = ["KAPPA_MARGIN", "GUARD_FRACTION", "TAU_W_MARGIN", "Calibration", "RealizedFloor",
           "fit", "realized_floor", "band_file_bytes", "write_band_file", "load_bands",
           "assert_same_band_source"]

# κ_max = KAPPA_MARGIN × the largest honest κ of the class. The spec asks only that κ_max sit
# above the honest factor. 2 mirrors τ_W's margin; κ_max is a safety ceiling, not the
# detection band, so a loose ceiling costs no detection power (P3.c).
KAPPA_MARGIN: float = 2.0
GUARD_FRACTION: float = 0.5  # the concentration guard: max honest residual ≤ τ/2
TAU_W_MARGIN: float = 2.0  # τ_W = max(τ_W⁰, 2·ρ_max) (P5)


@dataclass(frozen=True)
class Calibration:
    """The fitted bands and the statistics behind them, over the calibration steps."""

    steps: tuple[int, ...]
    k: int
    z: float
    classes: tuple[ClassSummary, ...]
    s_h: float  # the largest class RMS
    s_h_class: str
    global_max: float  # the largest honest normalized residual
    global_max_name: str
    global_max_step: int
    tau: float  # z·s_h
    kappa_max: Mapping[str, float]  # per class: KAPPA_MARGIN × honest max
    rho_max: Mapping[str, float]  # per weight tensor, over the steps
    tau_w: Mapping[str, float]  # per weight tensor: max(τ_W⁰, 2·ρ_max)

    @property
    def ratio(self) -> float:
        """The global maximum over ``s_h``: the honest-margin figure C1 reports."""
        return self.global_max / self.s_h

    @property
    def guard_limit(self) -> float:
        return GUARD_FRACTION * self.tau

    @property
    def guard_ok(self) -> bool:
        """The concentration guard holds: every honest residual is at most ``τ/2``."""
        return self.global_max <= self.guard_limit


def _items(stats: Mapping[int, StepStats] | Iterable[tuple[int, StepStats]]
           ) -> list[tuple[int, StepStats]]:
    items = stats.items() if isinstance(stats, Mapping) else stats
    return sorted(items, key=lambda x: x[0])


def fit(stats: Mapping[int, StepStats] | Iterable[tuple[int, StepStats]], *, k: int,
        z: float = Z) -> Calibration:
    """Fit ``τ``, ``κ_max`` per class and ``τ_W`` per tensor on honest steps' numbers.

    Raises ``ValueError`` if a step has no numbers, a product carries other than ``k``
    residuals, or any number is non-finite: an honest calibration step has none of these, so
    each means the calibration itself is broken.
    """
    items = _items(stats)
    if not items:
        raise ValueError("no calibration steps")
    for t, st in items:
        if not st.products or not st.tensors:
            raise ValueError(f"step {t} has no check-5 or check-6 numbers")
        for p in st.products:
            if len(p.normalized) != k:
                raise ValueError(f"step {t} P_{p.m} ({p.name}): {len(p.normalized)} residuals, "
                                 f"k = {k}")
            if not (math.isfinite(p.kappa) and all(math.isfinite(x) for x in p.normalized)):
                raise ValueError(f"step {t} P_{p.m} ({p.name}): non-finite κ or residual")
        for s in st.tensors:
            if not math.isfinite(s.rho_max):
                raise ValueError(f"step {t} {s.weight} ({s.check_id}): non-finite ρ_max")
    classes = tuple(class_summary(items))
    top = max(classes, key=lambda r: r.rms)
    mx = max(classes, key=lambda r: r.max)
    if not top.rms > 0:
        raise ValueError("every honest normalized residual is zero: s_h = 0 gives no band")
    rho: dict[str, float] = {}
    for _, st in items:
        for s in st.tensors:
            rho[s.weight] = max(rho.get(s.weight, 0.0), s.rho_max)
    return Calibration(
        steps=tuple(t for t, _ in items), k=k, z=z, classes=classes,
        s_h=top.rms, s_h_class=top.cls, global_max=mx.max, global_max_name=mx.max_name,
        global_max_step=mx.max_step, tau=z * top.rms,
        kappa_max={r.cls: max(1.0, KAPPA_MARGIN * r.kappa_max) for r in classes},
        rho_max=rho,
        tau_w={w: max(TAU_W0, TAU_W_MARGIN * x) for w, x in rho.items()})


# ---- the realized floor (P3, P4) ---------------------------------------------------------


@dataclass(frozen=True)
class RealizedFloor:
    """The floor each honest product gets at the fitted ``τ`` and the configured ``k``.

    ``ratio`` is ``Φ_m / ‖P_m‖_F = f_achieved(e_m)``: the smallest deviation the check
    guarantees to catch, as a fraction of the product. The target is 1 (P4). A poisoned record
    in a batch of four lowers ``‖Δ_m‖`` by about ``√2`` (S1d), so ``1/√2`` is the margin there.
    """

    count: int
    ratio_min: float
    ratio_median: float
    ratio_max: float
    ratio_max_name: str
    phi_min: float  # Φ_m itself, in the product's units
    phi_median: float
    phi_max: float
    over_target: int  # products with ratio > 1
    over_poison_margin: int  # products with ratio > 1/√2
    zero_row_sum: tuple[str, ...]  # products with ‖|P|·1‖ = 0 and ν ≠ 0
    by_q: Mapping[int, float]  # ratio per contracted dimension q (it depends on q only)


def realized_floor(stats: Mapping[int, StepStats] | Iterable[tuple[int, StepStats]], *,
                   tau: float, k: int, N: float, eps_in: float, eps_acc: float,
                   c: float = C_ANTI) -> RealizedFloor:
    """:class:`RealizedFloor` over every product of the given steps.

    Needs each product's ``q`` and ``‖P‖_F`` (``ProductStat``), which check 5 records."""
    ratios: list[tuple[float, str]] = []
    phis: list[float] = []
    zero: list[str] = []
    by_q: dict[int, float] = {}
    for t, st in _items(stats):
        for p in st.products:
            if p.q <= 0 or not math.isfinite(p.p_norm):
                raise ValueError(f"step {t} P_{p.m} ({p.name}) has no recorded q or ‖P‖_F")
            r = by_q.get(p.q)
            if r is None:
                r = by_q[p.q] = f_achieved(c, tau, e_m(p.q, eps_in, eps_acc), N, k)
            ratios.append((r, p.name))
            phis.append(r * p.p_norm)
            if p.p_abs1 == 0 and p.nu != 0:
                zero.append(f"{p.name} (step {t})")
    if not ratios:
        raise ValueError("no products")
    values = [r for r, _ in ratios]
    top = max(ratios)
    return RealizedFloor(
        count=len(values), ratio_min=min(values), ratio_median=statistics.median(values),
        ratio_max=top[0], ratio_max_name=top[1], phi_min=min(phis),
        phi_median=statistics.median(phis), phi_max=max(phis),
        over_target=sum(r > 1.0 for r in values),
        over_poison_margin=sum(r > 1.0 / math.sqrt(2.0) for r in values),
        zero_row_sum=tuple(zero), by_q=dict(sorted(by_q.items())))


# ---- the band file (P10a) ----------------------------------------------------------------


def band_file_bytes(cal: Calibration, stats: Mapping[str, Any] | None = None) -> bytes:
    """The band file: the fitted bands plus C1's statistics, as ``Bands.to_json`` bytes.

    The bytes depend only on the numbers, never on a time or a path, so a deterministic rerun
    writes the same file and the same hash. ``stats`` adds the caller's entries (sizing,
    realized floor, run context) to the ones fitted here.
    """
    fitted = {
        "calibration_steps": list(cal.steps),
        "k": cal.k,
        "z": cal.z,
        "s_h": cal.s_h,
        "s_h_class": cal.s_h_class,
        "global_max": cal.global_max,
        "global_max_at": f"{cal.global_max_name} (step {cal.global_max_step})",
        "max_over_s_h": cal.ratio,
        "concentration_guard": {"limit": cal.guard_limit, "max": cal.global_max,
                                "ok": cal.guard_ok},
        "kappa_margin": KAPPA_MARGIN,
        "tau_w_margin": TAU_W_MARGIN,
        "classes": {r.cls: {"count": r.count, "n": r.n, "rms": r.rms, "max": r.max,
                            "max_at": f"{r.max_name} (step {r.max_step})",
                            "kappa_median": r.kappa_median, "kappa_max": r.kappa_max}
                    for r in cal.classes},
        "rho_max": dict(cal.rho_max),
    }
    bands = Bands(tau=cal.tau, kappa_max=max(cal.kappa_max.values()),
                  kappa_classes=cal.kappa_max, tau_w=TAU_W0, tau_w_tensors=cal.tau_w,
                  stats={**fitted, **(stats or {})})
    return bands.to_json()


def write_band_file(path: str | os.PathLike[str], data: bytes) -> Bands:
    """Write the band file in one step (a temporary file, then a rename) and return the bands
    it holds, ``source`` set to its hash. A reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return Bands.from_json(data)


def load_bands(path: str | os.PathLike[str], *, k: int | None = None) -> Bands:
    """The band file every verification after calibration loads, read-only (P10a).

    A missing file raises ``FileNotFoundError``: a judged run refuses to start without it.
    With ``k``, the file must have been calibrated at that ``k``, so a run can't pair the bands
    with a ``VERIF_K`` that P10d's recompute did not approve.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no band file at {path}: run verification.runs.calibrate "
                                "first (P10a)")
    bands = Bands.from_file(path)
    if k is not None and bands.stats.get("k") != k:
        raise ValueError(f"band file {path} was calibrated at k = {bands.stats.get('k')}, "
                         f"this run has k = {k}")
    return bands


def assert_same_band_source(sources: Iterable[str | None]) -> str:
    """P10a's harness check: every run was judged by one and the same band file.

    Returns the shared hash. Raises ``AssertionError`` if a run had no bands, provisional bands
    or a different file."""
    got = list(sources)
    if not got:
        raise AssertionError("no runs to compare")
    bad = [s for s in got if s is None or s == "provisional"]
    if bad:
        raise AssertionError(f"a run was judged without the band file: {bad}")
    if len(set(got)) != 1:
        raise AssertionError(f"runs were judged by different band files: {sorted(set(got))}")
    return got[0]  # type: ignore[return-value]
