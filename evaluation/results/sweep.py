"""EV6: the planted-error sweep's figures and tables (EVALUATION_SPEC §5.7), from the records of
``verification.runs.flipped_matmul_sweep`` (EV3).

    .venv/bin/python -m evaluation.results.sweep [--scale test|full] [--root DIR]
        [--run-dir DIR] [--out DIR] [--norm honest|committed] [--floor-fit pooled|deep]

**The sweep.** On the stored step 4, for one product per §5.7 class and the binding product
(the one that sets ``k``), and per error shape (``entry``, ``entry2``, ``row2``, ``dense``), the
sweep planted errors of relative size ``f = ‖Δ‖_F/‖P‖_F`` on a log grid of
``x = f/(τ·e_m)``, ``e_m`` the target's own band unit (F18). Each trial ran check 5's ``k``
challenges on the target. The grid has 200 trials per point; the deep points, 2,000 at a few
``x`` on ``entry`` and ``row2``. The sweep run directory is ``--run-dir``, else the stored-
transcript sweep over the in-process one, the latest by its ``environment`` record.

**Figure (a)**, ``sweep_miss_rate``: one panel per shape. Per target and point, the
single-challenge miss rate ``p₁`` (missed challenges over ``trials·k``), with ±1 binomial
standard error ``√(p₁(1−p₁)/(trials·k))`` (lower bar cut short of 0), against ``x`` log–log.
Grid points are filled circles, deep points open squares. The predicted curve is
``p₁ = min(1, c/x)``, ``c = C_ANTI = √(2/3)``; the ``entry`` panel also has ``σ_r/x``, its own
theory. The smallest rate a point can resolve is one miss in its ``trials·k`` challenges,
``1/(trials·k)``, drawn as a line per trial count; a point with no miss is a downward triangle
on its line (its rate is below it). ``x`` is the point's realized mean: the swept ``x`` times
``f_real_mean/f``, and with ``--norm committed`` divided by the mean ``‖P'‖_F/‖P‖_F`` too.

**Figure (b)**, ``sweep_rejection``: one panel per shape. Per target and point, check 5's
rejection rate (trials rejected over trials, ±1 binomial standard error) against ``f`` on a log
axis, with the predicted ``1 − min(1, c/x)^k`` per target (``x = f/(τ·e_m)``, the band file's
``k``). ``f_achieved``, the detection floor from the band file's sizing
(``c·τ·e_max·2^{N/k}``), is a vertical line, and the measured floor, the same with ``ĉ`` in
place of ``c``, a dotted one. Each target's own floor ``c·τ·e_m·2^{N/k}`` (the summary's
``f_ach``) is a tick at the top in its colour.

**Table**, ``sweep_fit``: per target, shape and phase (``grid``, ``deep``, and ``pooled``
where both exist), the fitted miss constant ``ĉ`` exactly as the sweep runner fits it
(:func:`fit_c_hat`, a copy of the fit in ``flipped_matmul_sweep.summarize``, checked by a test):
the trials swept at ``x ≥ c_fit_x_min`` (3), each at its realized ``x``,
``ĉ = misses / Σ k/x``, Poisson standard error ``√misses / Σ k/x``; the expected misses under
``c/x`` are ``c·Σ k/x``. ``pooled`` runs the same fit on the grid's and the deep points' trials
together. Both norms are given (``ĉ`` and ``ĉ_committed``). The fit needs the trials
(``sweep_trials.npz``); without them the table holds the runner's own ``sweep_summary`` values
and no pooled rows. ``u_ratio`` is the target's ``max(1, U/‖P‖_F)``, recovered from its
recorded ``e_m`` (``(e_m − √2·ε_in)/(√q·ε_acc)``): the sweep records hold no ``U``, and
records written before F18 hold ``e_m`` at ratio 1, as the runner falls back to.

**Table**, ``sweep_check6``: the check-6 trial, one entry of ``W_{t+1}`` planted outside its
band, and whether check 6a rejected it on that entry.

**Placeholders** (open questions, not decided here):

- ``--norm`` (default ``honest``): whether ``x`` is measured against the honest product's norm
  ``‖P‖_F`` (§5.7's ``f``) or the committed, perturbed one ``‖P'‖_F`` (what check 5
  normalizes by). It sets figure (a)'s ``x`` and the ``ĉ`` behind the measured floor.
- ``--floor-fit`` (default ``pooled``): which ``ĉ`` gives the measured floor: grid and deep
  pooled, or deep only. The floor uses the binding product's shape with the largest ``ĉ``, as
  ``results/k_tunability.py`` does.

Outputs, under ``evaluation/results/<scale>/`` of ``--out`` (default ``--root``):
``sweep_miss_rate.{pdf,png}``, ``sweep_rejection.{pdf,png}``, ``sweep_fit.{csv,md}`` and
``sweep_check6.{csv,md}``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from evaluation.output import new_figure, write_figure, write_table
from evaluation.records import Records
from evaluation.results.common import SPEC_CLASSES, bands, parse, suptitle

__all__ = ["C_ANTI", "SIGMA_R", "SHAPES", "NORMS", "FLOOR_FITS", "THEORY", "Fit", "Floor",
           "SweepResult", "sweep_run", "load_trials", "fit_c_hat", "fit_rows", "point_rows",
           "floor", "predicted_p1", "predicted_reject", "u_ratio", "build", "figure_a",
           "figure_b", "caption", "write", "main"]

# verification.verifier.matmul_check.sizing.C_ANTI, the worst rank-1 miss constant, and
# verification.verifier.matmul_check.challenges.SIGMA_R, one challenge entry's std; copied,
# since a result script may not import verification (tests/test_layering.py). A test checks
# both, and the shapes against flipped_matmul_sweep.SHAPES.
C_ANTI = math.sqrt(2.0 / 3.0)
SIGMA_R = 1.0 / math.sqrt(3.0)
SHAPES = ("entry", "entry2", "row2", "dense")
# each shape's theory ĉ: one entry σ_r, two in one row √2·σ_r = c; the others have no 1/x tail
THEORY = {"entry": SIGMA_R, "row2": C_ANTI}
NORMS = ("honest", "committed")  # open question 1; "honest" a placeholder default
FLOOR_FITS = ("pooled", "deep")  # open question 2; "pooled" a placeholder default
TRIALS_FILE = "sweep_trials.npz"
NAME_A, NAME_B, NAME_FIT, NAME_C6 = ("sweep_miss_rate", "sweep_rejection", "sweep_fit",
                                     "sweep_check6")
_DODGE = 0.025  # decades between neighbouring targets' points, so error bars don't overlap

FIT_COLUMNS = ("spec_class", "name", "cls", "binding", "q", "shape", "phase", "u_ratio",
               "x_fit_min", "trials", "challenges", "misses", "expected", "c_hat", "c_hat_se",
               "z_vs_c", "c_theory", "z_vs_theory", "c_hat_committed", "c_hat_committed_se",
               "expected_committed", "f_ach", "f_ach_c_hat", "x_all", "f_all_over_f_ach",
               "source")
FIT_FORMATS = {"u_ratio": ".4g", "expected": ".1f", "c_hat": ".3f", "c_hat_se": ".3f",
               "z_vs_c": ".2f", "c_theory": ".3f", "z_vs_theory": ".2f",
               "c_hat_committed": ".3f", "c_hat_committed_se": ".3f", "expected_committed": ".1f", "f_ach": ".3g",
               "f_ach_c_hat": ".3g", "x_all": ".3g", "f_all_over_f_ach": ".3g"}
C6_COLUMNS = ("weight", "entry", "sign", "tau_w", "factor", "rho_target", "rho_entry",
              "check_id", "kind", "rejected", "detail")


# ---- the sweep's records --------------------------------------------------------------------


def _started(rec: Records, run_dir: str) -> str:
    env = rec.environment(run_dir)
    return str(env.get("started") or "") if env else ""


def sweep_run(rec: Records, run_dir: str | None = None) -> str:
    """The sweep's run directory: ``run_dir`` if given, else the latest stored-transcript sweep,
    else the latest in-process one."""
    sources = rec.table("sweep_source")
    dirs = sources.distinct("run_dir")
    if run_dir is not None:
        if run_dir not in dirs:
            raise ValueError(f"{run_dir!r} holds no sweep_source record (sweeps: {dirs})")
        return run_dir
    if not dirs:
        raise ValueError("no sweep_source record: run verification.runs.flipped_matmul_sweep")
    stored = [r["run_dir"] for r in sources if r.get("source") == "stored"]
    return max(stored or dirs, key=lambda d: _started(rec, d))


def load_trials(rec: Records, run_dir: str) -> dict[str, np.ndarray] | None:
    """The run directory's ``sweep_trials.npz``, or ``None`` if it has none."""
    a = next((a for a in rec.archives if a.run_dir == run_dir and a.path.name == TRIALS_FILE),
             None)
    return None if a is None else a.load()


@dataclass(frozen=True)
class Fit:
    """One fit of the miss constant (module docstring): ``ĉ = misses/exposure``."""

    trials: int
    challenges: int
    misses: int
    exposure: float  # Σ k/x over the fitted trials
    exposure_committed: float  # the same with x against ‖P'‖_F

    @property
    def c_hat(self) -> float | None:
        return self.misses / self.exposure if self.exposure > 0 else None

    @property
    def se(self) -> float | None:
        return math.sqrt(self.misses) / self.exposure if self.exposure > 0 else None

    @property
    def c_hat_committed(self) -> float | None:
        e = self.exposure_committed
        return self.misses / e if e > 0 else None

    @property
    def se_committed(self) -> float | None:
        e = self.exposure_committed
        return math.sqrt(self.misses) / e if e > 0 else None


def fit_c_hat(x_set: np.ndarray, f_real: np.ndarray, normalized: np.ndarray, e: float,
              tau: float, x_fit: float, norm_ratio: np.ndarray | None = None) -> Fit:
    """The sweep runner's fit (``flipped_matmul_sweep.summarize``), copied: trials picked by
    their swept ``x ≥ x_fit``, each challenge a miss when its normalized residual is at most
    ``τ``, the exposure ``Σ k/x`` at each trial's realized ``x = f_real/(τ·e)``, and on the
    committed norm at ``x' = x/(‖P'‖_F/‖P‖_F)``. ``norm_ratio`` ``None`` reads as 1."""
    x_set = np.asarray(x_set)
    f_real = np.asarray(f_real, dtype=np.float64)
    res = np.asarray(normalized, dtype=np.float64).reshape(len(x_set), -1)
    ratio = (np.ones(len(x_set)) if norm_ratio is None
             else np.asarray(norm_ratio, dtype=np.float64))
    k = res.shape[1]
    fit = x_set >= x_fit
    misses = res <= tau
    x_real = f_real / (tau * e)
    x_committed = f_real / (tau * e * ratio)
    exposure = float((k / x_real[fit]).sum()) if fit.any() else 0.0
    exposure_c = float((k / x_committed[fit]).sum()) if fit.any() else 0.0
    return Fit(int(fit.sum()), int(fit.sum()) * k, int(misses[fit].sum()), exposure, exposure_c)


def u_ratio(e_m: float, q: int, eps_in: float, eps_acc: float) -> float:
    """``max(1, U/‖P‖_F)`` recovered from ``e_m = √2·ε_in + √q·ε_acc·max(1, U/‖P‖_F)``
    (``sizing.e_m``)."""
    return (e_m - math.sqrt(2.0) * eps_in) / (math.sqrt(q) * eps_acc)


def _theory(shape: str) -> float | None:
    return THEORY.get(shape)


def fit_rows(summaries: Sequence[dict[str, Any]], trials: dict[str, np.ndarray] | None,
             tau: float, eps: tuple[float, float] | None = None) -> list[dict[str, Any]]:
    """One row per target, shape and phase, and a ``pooled`` row per target and shape with
    both phases (module docstring). ``eps`` is ``(ε_in, ε_acc)`` for ``u_ratio``."""
    rows = []
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for s in summaries:
        by_key.setdefault((s["m"], s["shape"]), []).append(s)
    for (m, shape), ss in by_key.items():
        ss = sorted(ss, key=lambda s: ("grid", "deep").index(s["phase"])
                    if s["phase"] in ("grid", "deep") else 2)
        fits: dict[str, Fit | None] = {}
        if trials is not None:
            here = (trials["m"] == m) & (trials["shape"] == shape)
            for s in ss:
                sel = here & (trials["phase"] == s["phase"])
                fits[s["phase"]] = _fit_of(trials, sel, s, tau)
            if len(ss) > 1:
                x_mins = {s["c_fit_x_min"] for s in ss}
                es = {s["e_m"] for s in ss}
                if len(x_mins) > 1 or len(es) > 1:
                    raise ValueError(f"P_{m} {shape}: phases fitted from different x or e_m "
                                     f"({x_mins}, {es})")
                fits["pooled"] = _fit_of(trials, here, ss[0], tau)
        for s in ss:
            rows.append(_fit_row(s, fits.get(s["phase"]), s["phase"], eps,
                                 "trials" if trials is not None else "summary"))
        if "pooled" in fits:
            rows.append(_fit_row(ss[0], fits["pooled"], "pooled", eps, "trials"))
    order = {c: i for i, c in enumerate(SPEC_CLASSES)}
    phase = {"grid": 0, "deep": 1, "pooled": 2}
    rows.sort(key=lambda r: (r["binding"], order.get(r["spec_class"], len(order)), r["m"],
                             SHAPES.index(r["shape"]) if r["shape"] in SHAPES else 9,
                             phase.get(r["phase"], 3)))
    return rows


def _fit_of(trials: dict[str, np.ndarray], sel: np.ndarray, s: dict[str, Any],
            tau: float) -> Fit:
    ratio = trials["norm_ratio"][sel] if "norm_ratio" in trials else None
    if ratio is not None and len(ratio) == 0:
        ratio = None
    return fit_c_hat(trials["x"][sel], trials["f_real"][sel], trials["normalized"][sel],
                     float(s["e_m"]), tau, float(s["c_fit_x_min"]), ratio)


def _fit_row(s: dict[str, Any], fit: Fit | None, phase: str,
             eps: tuple[float, float] | None, source: str) -> dict[str, Any]:
    if fit is not None:
        c_hat, se, misses = fit.c_hat, fit.se, fit.misses
        c_com, se_com = fit.c_hat_committed, fit.se_committed
        challenges, n = fit.challenges, fit.trials
        expected, expected_c = C_ANTI * fit.exposure, C_ANTI * fit.exposure_committed
    else:  # the runner's own summary
        c_hat, se, misses = s["c_hat"], s["c_hat_se"], s["c_fit_misses"]
        c_com, se_com = s.get("c_hat_committed"), s.get("c_hat_committed_se")
        challenges = s["c_fit_vectors"]
        n = challenges // s["k"] if s.get("k") else None
        expected = C_ANTI * misses / c_hat if c_hat else None
        expected_c = C_ANTI * misses / c_com if c_com else None
    f_ach = s["f_ach"]
    return {
        "m": s["m"], "spec_class": s["spec_class"], "name": s["name"], "cls": s["cls"],
        "binding": bool(s.get("binding")), "q": s.get("q"), "shape": s["shape"],
        "phase": phase, "e_m": s["e_m"],
        "u_ratio": (None if eps is None or s.get("q") is None
                    else u_ratio(s["e_m"], int(s["q"]), *eps)),
        "x_fit_min": s["c_fit_x_min"], "trials": n, "challenges": challenges,
        "misses": misses, "expected": expected, "c_hat": c_hat, "c_hat_se": se,
        "z_vs_c": None if c_hat is None or not se else (c_hat - C_ANTI) / se,
        "c_theory": _theory(s["shape"]),
        "z_vs_theory": (None if c_hat is None or not se or _theory(s["shape"]) is None
                        else (c_hat - _theory(s["shape"])) / se),
        "c_hat_committed": c_com,
        "c_hat_committed_se": se_com, "expected_committed": expected_c, "f_ach": f_ach,
        "f_ach_c_hat": None if c_hat is None else f_ach * c_hat / C_ANTI,
        "x_all": s.get("x_all") if phase == "grid" else None,
        "f_all_over_f_ach": s.get("f_all_over_f_ach") if phase == "grid" else None,
        "source": source}


def point_rows(points: Sequence[dict[str, Any]], norm: str = "honest"
               ) -> list[dict[str, Any]]:
    """What the figures plot, one row per sweep point (module docstring): the realized mean
    ``x`` on the chosen norm, ``p₁`` and its standard error, the resolvable rate, and the
    rejection rate and its standard error."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    out = []
    for p in points:
        x = p["x"] * p["f_real_mean"] / p["f"]
        if norm == "committed":
            x /= p.get("norm_ratio_mean", 1.0)
        r, n = p["reject_rate"], p["trials"]
        out.append({"m": p["m"], "name": p["name"], "spec_class": p["spec_class"],
                    "binding": bool(p.get("binding")), "shape": p["shape"],
                    "phase": p["phase"], "x_swept": p["x"], "x": x, "f": p["f_real_mean"],
                    "trials": n, "misses": p["misses"], "vectors": p["vectors"],
                    "p1": p["misses"] / p["vectors"],
                    "p1_se": math.sqrt(p["p1"] * (1 - p["p1"]) / p["vectors"]),
                    "resolvable": 1 / p["vectors"], "reject_rate": r,
                    "reject_se": math.sqrt(r * (1 - r) / n)})
    return out


def predicted_p1(x: np.ndarray | float, c: float = C_ANTI) -> np.ndarray:
    """``min(1, c/x)``."""
    return np.minimum(1.0, c / np.asarray(x, dtype=np.float64))


def predicted_reject(f: np.ndarray | float, tau: float, e: float, k: int,
                     c: float = C_ANTI) -> np.ndarray:
    """Check 5's predicted rejection rate ``1 − min(1, c/x)^k`` at ``x = f/(τ·e)``."""
    return 1.0 - predicted_p1(np.asarray(f, dtype=np.float64) / (tau * e), c) ** k


# ---- the floor ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Floor:
    """The detection floor marked on figure (b): predicted from the band file's sizing, and
    measured with the binding product's ``ĉ`` (module docstring)."""

    f_achieved: float  # c·τ·e_max·2^{N/k}
    source: str  # "band file" or the fallback's description
    c_hat: float | None
    c_hat_se: float | None
    shape: str | None
    name: str | None
    fit: str  # "pooled" or "deep"
    norm: str  # "honest" or "committed"
    misses: int | None = None
    expected: float | None = None

    @property
    def measured(self) -> float | None:
        return None if self.c_hat is None else self.f_achieved * self.c_hat / C_ANTI


def floor(rows: Sequence[dict[str, Any]], band_file: dict[str, Any] | None,
          summaries: Sequence[dict[str, Any]], *, fit: str = "pooled",
          norm: str = "honest") -> Floor:
    """``f_achieved`` from the band file's ``stats.sizing``, else the binding product's own
    floor (its summary's ``f_ach``); ``ĉ`` from the binding product's ``fit`` rows on ``norm``,
    the shape with the largest. A ``pooled`` fit falls back to ``grid`` for a shape the deep
    points didn't sweep."""
    if fit not in FLOOR_FITS:
        raise ValueError(f"fit must be one of {FLOOR_FITS}, got {fit!r}")
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    sz = ((band_file or {}).get("stats") or {}).get("sizing")
    if sz and sz.get("f_achieved") is not None:
        f_ach, src = float(sz["f_achieved"]), "band file"
    else:
        b = [s for s in summaries if s.get("binding")]
        if not b:
            raise ValueError("no band file sizing and no binding product: no f_achieved")
        f_ach, src = float(b[0]["f_ach"]), "binding product's summary (no band file sizing)"
    key, se_key, exp_key = (("c_hat", "c_hat_se", "expected") if norm == "honest" else
                            ("c_hat_committed", "c_hat_committed_se", "expected_committed"))
    cand: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not r["binding"] or r[key] is None:
            continue
        if r["phase"] == fit or (fit == "pooled" and r["phase"] == "grid"
                                 and r["shape"] not in cand):
            cand[r["shape"]] = r
    if not cand:
        return Floor(f_ach, src, None, None, None, None, fit, norm)
    best = max(cand.values(), key=lambda r: r[key])
    return Floor(f_ach, src, best[key], best[se_key], best["shape"], best["name"],
                 best["phase"], norm, best["misses"], best[exp_key])


# ---- building and drawing ---------------------------------------------------------------------


@dataclass
class SweepResult:
    """What the figures and tables are built from."""

    run_dir: str
    source: dict[str, Any]
    env: dict[str, Any]
    summaries: list[dict[str, Any]]
    points: list[dict[str, Any]]
    fits: list[dict[str, Any]]
    check6: list[dict[str, Any]]
    floor: Floor
    tau: float
    k: int
    N: float
    norm: str
    notes: list[str] = field(default_factory=list)


def build(rec: Records, run_dir: str | None = None, *, band_file: dict[str, Any] | None = None,
          norm: str = "honest", floor_fit: str = "pooled") -> SweepResult:
    """Read the sweep and build the rows, the fits and the floor."""
    d = sweep_run(rec, run_dir)
    [source] = rec.table("sweep_source").where(run_dir=d)
    summaries = list(rec.table("sweep_summary").where(run_dir=d))
    points = list(rec.table("sweep_point").where(run_dir=d))
    if not summaries or not points:
        raise ValueError(f"{d} holds no sweep_summary or sweep_point record")
    env = rec.environment(d) or {}
    notes = []
    taus = {s["tau"] for s in summaries}
    ks = {s["k"] for s in summaries}
    Ns = {s["N"] for s in summaries}
    if len(taus) != 1 or len(ks) != 1 or len(Ns) != 1:
        raise ValueError(f"{d}: summaries disagree on τ, k or N ({taus}, {ks}, {Ns})")
    tau, k, N = taus.pop(), int(ks.pop()), float(Ns.pop())
    sz = ((band_file or {}).get("stats") or {}).get("sizing") or {}
    if band_file is None:
        notes.append("no band file beside the run directories: f_achieved is the binding "
                     "product's own floor and u_ratio is left out")
    elif sz and (not math.isclose(float(sz.get("tau", tau)), tau, rel_tol=1e-12)
                 or int(sz.get("k", k)) != k):
        notes.append(f"the band file's sizing (τ {sz.get('tau')}, k {sz.get('k')}) differs "
                     f"from the sweep's (τ {tau}, k {k}): is it the band file the sweep ran?")
    eps = ((float(sz["eps_in"]), float(sz["eps_acc"]))
           if sz.get("eps_in") is not None and sz.get("eps_acc") is not None else None)
    trials = load_trials(rec, d)
    if trials is None:
        notes.append(f"{d} has no {TRIALS_FILE}: the table holds the runner's summaries, "
                     "with no pooled rows")
    fits = fit_rows(summaries, trials, tau, eps)
    if trials is not None:
        _check_against_summaries(fits, summaries, d)
    ratios = [r["u_ratio"] for r in fits if r["u_ratio"] is not None]
    if ratios and all(abs(r - 1) < 1e-9 for r in ratios):
        notes.append("every target's e_m is at U/‖P‖_F ratio 1: records written before the "
                     "split error model (F18) carry no U, so their unit falls back to "
                     "ratio 1, as the sweep runner's does")
    fl = floor(fits, band_file, summaries, fit=floor_fit, norm=norm)
    if fl.c_hat is None:
        notes.append("no fitted ĉ on the binding product: no measured floor")
    check6 = list(rec.table("sweep_check6").where(run_dir=d))
    if not check6:
        notes.append(f"{d} holds no sweep_check6 record")
    return SweepResult(d, source, env, summaries, point_rows(points, norm), fits, check6, fl,
                       tau, k, N, norm, notes)


def _check_against_summaries(fits: Sequence[dict[str, Any]],
                             summaries: Sequence[dict[str, Any]], d: str) -> None:
    """The fit from the trials must give the runner's own ``ĉ`` for each phase."""
    by = {(s["m"], s["shape"], s["phase"]): s for s in summaries}
    for r in fits:
        s = by.get((r["m"], r["shape"], r["phase"]))
        if s is None:
            continue
        for mine, theirs in (("c_hat", "c_hat"), ("c_hat_committed", "c_hat_committed"),
                             ("misses", "c_fit_misses"), ("challenges", "c_fit_vectors")):
            a, b = r[mine], s.get(theirs)
            if b is None and a in (None, 0, 0.0):
                continue
            if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
                warnings.warn(f"{d}: {r['name']} {r['shape']} {r['phase']}: {mine} from the "
                              f"trials is {a}, the summary's {theirs} {b}", UserWarning,
                              stacklevel=3)


def _targets(res: SweepResult) -> list[dict[str, Any]]:
    """The targets in canonical order, with a colour and a label each."""
    seen: dict[int, dict[str, Any]] = {}
    for s in res.summaries:
        seen.setdefault(s["m"], s)
    order = {c: i for i, c in enumerate(SPEC_CLASSES)}
    ts = sorted(seen.values(), key=lambda s: (bool(s.get("binding")),
                                              order.get(s["spec_class"], len(order)), s["m"]))
    out = []
    for i, s in enumerate(ts):
        label = f"{s['spec_class']} ({s['name']})" + (", binding" if s.get("binding") else "")
        out.append({"m": s["m"], "e_m": s["e_m"], "f_ach": s["f_ach"], "label": label,
                    "colour": f"C{i % 10}", "dodge": 10 ** (_DODGE * (i - (len(ts) - 1) / 2))})
    return out


def _shapes(res: SweepResult) -> list[str]:
    present = {s["shape"] for s in res.summaries}
    return [s for s in SHAPES if s in present] + sorted(present - set(SHAPES))


def _grid(n: int) -> tuple[int, int]:
    return (1, n) if n <= 2 else (math.ceil(n / 2), 2)


def figure_a(res: SweepResult) -> Figure:
    """Figure (a): ``p₁`` against ``x`` (module docstring)."""
    shapes, targets = _shapes(res), _targets(res)
    nr, nc = _grid(len(shapes))
    fig, axes = new_figure(nr, nc, figsize=(5.2 * nc, 4.2 * nr), squeeze=False)
    xs = [p["x"] for p in res.points]
    xline = np.geomspace(min(xs) / 1.5, max(xs) * 1.5, 200)
    res_levels = sorted({(p["phase"], p["trials"], p["vectors"]) for p in res.points},
                        key=lambda t: -t[2])
    floor_y = min(p["resolvable"] for p in res.points)
    for ax, shape in zip(axes.flat, shapes, strict=False):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(xline, predicted_p1(xline), color="k", lw=1.2,
                label=f"predicted c/x, c = √(2/3) = {C_ANTI:.3f}")
        if shape in THEORY and THEORY[shape] != C_ANTI:
            ax.plot(xline, predicted_p1(xline, THEORY[shape]), color="k", lw=0.8, ls=":",
                    label=f"σ_r/x = {THEORY[shape]:.3f}/x (one entry's theory)")
        for (phase, n, vec), ls in zip(res_levels, ("--", "-.", ":", "--"), strict=False):
            ax.axhline(1 / vec, color="grey", lw=0.9, ls=ls,
                       label=f"resolvable, {n:,} trials ({phase}): 1 miss in {vec:,}")
        for t in targets:
            for phase, marker, fill in (("grid", "o", t["colour"]), ("deep", "s", "none")):
                ps = [p for p in res.points if p["m"] == t["m"] and p["shape"] == shape
                      and p["phase"] == phase]
                if not ps:
                    continue
                hit = [p for p in ps if p["misses"] > 0]
                zero = [p for p in ps if p["misses"] == 0]
                if hit:
                    x = np.array([p["x"] for p in hit]) * t["dodge"]
                    y = np.array([p["p1"] for p in hit])
                    se = np.array([p["p1_se"] for p in hit])
                    ax.errorbar(x, y, yerr=[np.minimum(se, 0.999 * y), se], ls="none",
                                marker=marker, ms=4 if phase == "grid" else 6,
                                mfc=fill, color=t["colour"], capsize=2, lw=0.8)
                if zero:
                    ax.plot(np.array([p["x"] for p in zero]) * t["dodge"],
                            [p["resolvable"] for p in zero], ls="none", marker="v",
                            ms=4 if phase == "grid" else 6, mfc=fill, color=t["colour"])
        ax.set_ylim(floor_y / 4, 2.0)
        ax.set_title(shape, fontsize=10)
        ax.set_xlabel("x = f / (τ·e_m)" + ("" if res.norm == "honest"
                                          else ", f against the committed ‖P'‖_F"))
        ax.set_ylabel("single-challenge miss rate p₁")
        ax.grid(True, which="major", lw=0.3)
    for ax in list(axes.flat)[len(shapes):]:
        ax.set_visible(False)
    _legend(fig, axes.flat[0], targets)
    suptitle(fig, f"Planted-error sweep (§5.7a): single-challenge miss rate, step "
             f"{res.source.get('step')} ({res.source.get('source')} transcript), τ = "
             f"{res.tau:.2f}, k = {res.k}\n● grid, □ deep points; ▼ no miss (below the "
             "resolvable line); bars ±1 binomial standard error")
    return fig


def _legend(fig: Figure, ax: Any, targets: Sequence[dict[str, Any]]) -> None:
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    for t in targets:
        handles.append(Line2D([], [], color=t["colour"], marker="o", ls="none"))
        labels.append(t["label"])
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)


def figure_b(res: SweepResult) -> Figure:
    """Figure (b): check 5's rejection rate against ``f`` (module docstring)."""
    shapes, targets = _shapes(res), _targets(res)
    nr, nc = _grid(len(shapes))
    fig, axes = new_figure(nr, nc, figsize=(5.2 * nc, 4.0 * nr), squeeze=False)
    fl = res.floor
    fs = [p["f"] for p in res.points]
    lo, hi = min(fs) / 1.5, max(max(fs), fl.f_achieved) * 1.5
    for ax, shape in zip(axes.flat, shapes, strict=False):
        ax.set_xscale("log")
        for t in targets:
            fline = np.geomspace(lo, hi, 300)
            ax.plot(fline, predicted_reject(fline, res.tau, t["e_m"], res.k), color=t["colour"],
                    lw=0.8, ls="--", alpha=0.8)
            ax.plot([t["f_ach"]], [1.04], marker="|", ms=8, color=t["colour"], clip_on=False)
            for phase, marker, fill in (("grid", "o", t["colour"]), ("deep", "s", "none")):
                ps = [p for p in res.points if p["m"] == t["m"] and p["shape"] == shape
                      and p["phase"] == phase]
                if ps:
                    ax.errorbar([p["f"] * t["dodge"] for p in ps], [p["reject_rate"] for p in ps],
                                yerr=[p["reject_se"] for p in ps], ls="none", marker=marker,
                                ms=4 if phase == "grid" else 6, mfc=fill, color=t["colour"],
                                capsize=2, lw=0.8)
        ax.axvline(fl.f_achieved, color="k", lw=1.2,
                   label=f"f_achieved = {fl.f_achieved:.3g} ({fl.source})")
        if fl.measured is not None:
            ax.axvline(fl.measured, color="k", lw=1.0, ls=":",
                       label=f"measured floor {fl.measured:.3g} (ĉ = {fl.c_hat:.3f}, "
                             f"{fl.name} {fl.shape}, {fl.fit}, {fl.norm} norm)")
        ax.plot([], [], color="grey", lw=0.8, ls="--",
                label=f"predicted 1 − min(1, c/x)^k, k = {res.k}, per target")
        ax.plot([], [], color="grey", marker="|", ls="none", ms=8,
                label="each target's own floor c·τ·e_m·2^(N/k)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title(shape, fontsize=10)
        ax.set_xlabel("relative error size f = ‖Δ‖_F / ‖P‖_F")
        ax.set_ylabel("check 5 rejection rate")
        ax.grid(True, which="major", lw=0.3)
    for ax in list(axes.flat)[len(shapes):]:
        ax.set_visible(False)
    _legend(fig, axes.flat[0], targets)
    suptitle(fig, f"Planted-error sweep (§5.7b): check 5's rejection rate, step "
             f"{res.source.get('step')} ({res.source.get('source')} transcript), τ = "
             f"{res.tau:.2f}, k = {res.k}, N = {res.N:.2f}\n● grid, □ deep points; bars ±1 "
             "binomial standard error")
    return fig


def caption(res: SweepResult) -> str:
    """The fit table's caption."""
    fl = res.floor
    parts = [
        f"Planted-error sweep (EVALUATION_SPEC §5.7), {res.run_dir}: step "
        f"{res.source.get('step')} ({res.source.get('source')} transcript, root "
        f"{str(res.source.get('root'))[:8]}…), band file "
        f"{str(res.env.get('band_file_hash'))[:8]}…, τ = {res.tau:.4g}, k = {res.k}.",
        "Per target, shape and phase, the single-challenge miss constant fitted as the sweep "
        "runner fits it: the trials swept at x ≥ x_fit_min, each at its realized "
        "x = f_real/(τ·e_m), ĉ = misses/Σ k/x with Poisson standard error √misses/Σ k/x; "
        f"expected = c·Σ k/x, the misses c/x predicts (c = √(2/3) = {C_ANTI:.4f}); z_vs_c = "
        "(ĉ − c)/se, z_vs_theory = (ĉ − c_theory)/se. 'pooled' fits the grid's and the deep "
        "points' trials together. The _committed columns take x against the committed ‖P'‖_F, which check 5 normalizes by. "
        "c_theory: σ_r for one entry, c for two in one row; two entries in different rows "
        "and dense errors have no 1/x tail. f_ach = c·τ·e_m·2^(N/k) at the target's e_m, "
        "f_ach_c_hat the same with ĉ. u_ratio is max(1, U/‖P‖_F) recovered from e_m.",
        f"Measured floor ({fl.fit} fit, {fl.norm} norm; both placeholders): "
        + ("none" if fl.measured is None else
           f"{fl.measured:.4g} = f_achieved {fl.f_achieved:.4g} ({fl.source}) × ĉ/c with "
           f"ĉ = {fl.c_hat:.4f} ± {fl.c_hat_se:.4f} ({fl.name}, {fl.shape}, {fl.misses} "
           f"misses against {fl.expected:.1f} expected)") + ".",
    ]
    for c6 in res.check6:
        parts.append(_check6_text(c6))
    parts += [f"Note: {n}." for n in res.notes]
    return " ".join(parts)


def _check6_text(c6: dict[str, Any]) -> str:
    verdict = "rejected at 6a on that entry" if c6.get("rejected") else "NOT rejected as planned"
    return (f"Check-6 trial: {c6.get('weight')} entry {c6.get('entry')} moved to ρ "
            f"{c6.get('rho_entry', math.nan):.3g} (target {c6.get('factor')}·τ_W = "
            f"{c6.get('rho_target')}): {verdict} ({c6.get('check_id')}, {c6.get('kind')}).")


def write(res: SweepResult, scale: str, *, root: Any = None) -> list[Path]:
    """Both figures and both tables; returns the paths."""
    paths = list(write_figure(figure_a(res), NAME_A, scale, root=root))
    paths += write_figure(figure_b(res), NAME_B, scale, root=root)
    paths += write_table(res.fits, NAME_FIT, scale, root=root, columns=FIT_COLUMNS,
                         formats=FIT_FORMATS, caption=caption(res))
    c6_caption = ("The check-6 planted trial (§5.7): one entry of W_{t+1} moved outside its "
                  "band; check 6a must reject on that entry.")
    paths += write_table(res.check6 or [{}], NAME_C6, scale, root=root, columns=C6_COLUMNS,
                         caption=c6_caption)
    return list(paths)


def main(argv: Sequence[str] | None = None) -> int:
    def options(p: Any) -> None:
        p.add_argument("--run-dir", default=None,
                       help="the sweep's run directory, relative to --root (default: the "
                            "latest stored-transcript sweep)")
        p.add_argument("--out", type=Path, default=None,
                       help="write under OUT/evaluation/results/<scale>/ (default: --root)")
        p.add_argument("--norm", choices=NORMS, default="honest",
                       help="x against the honest ‖P‖_F or the committed ‖P'‖_F "
                            "(open question; default honest, a placeholder)")
        p.add_argument("--floor-fit", choices=FLOOR_FITS, default="pooled",
                       help="the ĉ behind the measured floor: grid and deep pooled, or deep "
                            "only (open question; default pooled, a placeholder)")

    inp = parse("Write the planted-error sweep's figures and tables (EVALUATION_SPEC §5.7).",
                argv, options)
    a = inp.args
    res = build(inp.records, a.run_dir, band_file=bands(inp.root), norm=a.norm,
                floor_fit=a.floor_fit)
    for n in res.notes:
        print(f"note: {n}")
    for p in write(res, inp.scale, root=a.out if a.out is not None else inp.root):
        print(p)
    fl = res.floor
    print(f"{res.run_dir}: {len(res.fits)} fit rows, f_achieved {fl.f_achieved:.4g}"
          + ("" if fl.measured is None else
             f", measured floor {fl.measured:.4g} (ĉ {fl.c_hat:.4f} ± {fl.c_hat_se:.4f}, "
             f"{fl.name} {fl.shape}, {fl.fit}, {fl.norm})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
