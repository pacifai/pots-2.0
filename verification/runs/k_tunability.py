"""FS12 and EQ11: check 5's cost and detection floor as ``k`` varies, on the stored step 4
(EVALUATION_SPEC §5.8, DECISIONS_FULL_SCALE.md F10).

    .venv/bin/python -m verification.runs.k_tunability [--run-dir DIR] [--k-top K]
        [--repeats N] [--metrics | --no-metrics]

Set up as the honest run (``run_verified``), judged by the frozen band file, read-only. The run
decides nothing: the configured ``k`` (``VERIF_K``) and the band file don't depend on it.

1. **The transcript.** Step 4 as the honest run stored it, read as the sweep reads it
   (``run_verified.load_kept_step``): every leaf hashed again, the root checked, and the run's
   check-5 and check-6 numbers for the step read from its residual archive. The run directory
   is ``$VERIF_OUTPUT_DIR/honest-<model>-<attack>-bpr0-s<seed>``, or ``--run-dir``; its
   band-file hash must be this band file's.
2. **The full step.** Every check in the default order at the configured ``k``, as the driver
   runs them (:func:`full_step`): once to warm up, then once per timed round. It must accept,
   and its check-5 and check-6 numbers must equal the archive's bit for bit. Check 7's input,
   the expected ``W_t`` hashes, is the step's own ``W_t`` leaf hashes: step 3 isn't stored, and
   its ``W_{t+1}`` hashes equal these, since the honest run's check 7 accepted step 4.
3. **Check 5 per ``k``**, ``k = 1 … k_top`` (:func:`check5_pass`): the verifier's own
   ``checks.check_5_matmuls`` with ``k`` as its argument, on the leaves and claimed root that the
   full step's check-2 read left, each call with its own replay. Three passes:
   - timed: ``--repeats`` rounds (default 3), each a full step, then every ``k``, ascending in
     odd rounds and descending in even ones, so a drift in the machine's speed doesn't read as
     a trend in ``k``. The median per ``k``. The wall clock wraps the whole call; the metrics
     seam splits it into ``5.glue`` (the replay and serving the operands) and ``5.measure``;
   - memory: the process footprint's peak during the call less its value just before it
     (``metrics.memory_probe``, after ``gc.collect()``);
   - counts (``metrics.counting``, never timed): FLOPs, hash calls and hash bytes per section.
4. **Checks.** Every pass must accept. κ, ν, ``‖|P|·1‖`` and ``‖P‖_F`` don't depend on the
   challenges, so they must be bit-equal at every ``k``. Challenge ``j`` is keyed on ``(m, j)``
   alone, so a pass's first ``min(k, k_conf)`` residuals are the configured pass's in exact
   arithmetic; the matmuls run with another column count and may round differently, so the run
   records the largest relative difference and how many are equal bit for bit. The configured
   ``k`` must reproduce the archive's check-5 numbers bit for bit, and the step record's
   ``ρ_5``. Every pass's numbers must be equal across the timed rounds. Predicted counts:
   ``5.measure`` FLOPs ``S·(k+1) + S_U``, with ``S = Σ_m 2(q·w + n·q + n·w)`` the FLOPs of one
   challenge's ``B·r``, ``A·(B·r)`` and ``P·r`` (the κ guard's ``|B|·1``, ``|A|·(|B|·1)`` and
   ``|P|·1`` are one more column) and ``S_U = Σ_m 2(q·w + n·q)`` the term scale's
   ``A²·(B²·1)`` (F18), constant in ``k``; bias terms are left out; ``5.glue`` FLOPs constant
   in ``k``;
   ``M·k`` hash calls, ``6·M·k`` bytes in (the labels) and ``4·k·Σ_m w`` bytes out. The times
   are fitted linearly in ``k`` over every ``k`` and over ``k ≥ 2``: at ``k = 1``, ``B·r`` and
   ``P·r`` are matrix-vector products, which BLAS runs with other kernels.
5. **k_top** ``= max(2·k_conf, k_cover)`` (§5.8). ``k_cover`` is the smallest ``k`` whose
   guaranteed substitution rate covers one poisoned record per batch, ``rate·B ≤ 1``, with
   ``rate = f_ach(k)·C/(√B·‖δg‖/‖g‖)`` (the sizing appendix's (12.1)), ``C = 1`` and
   ``‖δg‖/‖g‖ = 2``, both hand-picked (F10; the spec's 18 and 44 assume both).
   ``f_ach(k) = c·τ·e_max·2^{N/k}`` takes ``τ``, ``e_max`` and ``N`` from the band file's
   sizing, and ``c = C_ANTI``. A band file fitted since F18 records the largest calibration
   ``e_m`` as ``e_max`` (P10d's recompute); an older one records ``e_m(q_max)``, which F18 calls
   the advance sizing. ``--k-top`` overrides it.

With metrics on (default ``VERIF_METRICS``), records go to ``$VERIF_OUTPUT_DIR/k_tunability/``:
one ``k_source`` (the step, the floor's inputs, the other checks' times and the honest run's
own step-4 records), one ``k_point`` per ``k`` and one ``k_summary`` (linear fits of the times
in ``k``, the count checks, the reproduction). ``evaluation/results/k_tunability.py`` draws
them. Exits 1 unless the configured ``k`` reproduces the archive, every pass accepts, and κ is
equal at every ``k``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from setup import backdoorllm
from setup.config import RunConfig, load_config, setup_determinism
from setup.data import schedule
from verification.commitment.leaves import leaf_hashes_of
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import DeclaredComputation
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF, load_protocol_config
from verification.runs.llama_step import load_committed_dataset
from verification.runs.metrics import (
    MetricsWriter,
    counting,
    memory_probe,
    read_records,
    step_scores,
)
from verification.runs.run_verified import (
    KEEP_STEP,
    KeptStep,
    honest_run_name,
    load_kept_step,
    stored_band_hash,
)
from verification.runs.store_crosscheck import _stats_key
from verification.transcript.store import TranscriptStore
from verification.verifier.bands import Bands
from verification.verifier.calibration import band_context, load_bands
from verification.verifier.checks import CHECKS, DEFAULT_ORDER
from verification.verifier.context import Rejection, StepContext, StepState, StepStats
from verification.verifier.matmul_check.challenges import encode_label
from verification.verifier.matmul_check.sizing import (
    C_ANTI,
    F_TARGET,
    b0,
    bit_budget,
    e_m,
    f_achieved,
)

__all__ = [
    "COVER_COHERENCE",
    "GRADIENT_RATIO",
    "KFloor",
    "KResult",
    "Sections",
    "base_context",
    "check5_pass",
    "covering_k",
    "full_step",
    "guaranteed_rate",
    "k_floor",
    "k_top",
    "linear_fit",
    "main",
    "measure_flops_per_column",
    "measure_flops_term_scale",
    "predicted_hashes",
    "run_k",
]

RUN_NAME = "k_tunability"
STEP = KEEP_STEP
DEFAULT_REPEATS = 3
GRADIENT_RATIO = 2.0  # ‖δg‖/‖g‖ in (12.1), hand-picked (sizing §12.4, F10)
COVER_COHERENCE = 1.0  # C for k_cover: the spec's k_top of 18 and 44 assume it
COUNT_FIELDS = ("flops", "hash_calls", "hash_in_bytes", "hash_out_bytes")
CHECK5_PARTS = ("5", "5.glue", "5.measure")
TIMES = ("time_s", "glue_s", "measure_s")


# ---- the floor and the rate ---------------------------------------------------------------


@dataclass(frozen=True)
class KFloor:
    """The sizing floor as a function of ``k``, ``f_ach(k) = c·τ·e_max·2^{N/k}`` (8.1) with
    ``c = C_ANTI``, and the bits ``k`` challenges buy at the sizing target ``f`` (6.4)."""

    tau: float
    e_max: float
    N: float
    b0: float  # bits per challenge at f
    lam: float
    k_conf: int
    source: str  # "band file" or "computed"

    @property
    def floor0(self) -> float:
        """``c·τ·e_max``, the floor as ``k → ∞``."""
        return C_ANTI * self.tau * self.e_max

    def f_ach(self, k: int) -> float:
        return f_achieved(C_ANTI, self.tau, self.e_max, self.N, k)

    def bits(self, k: int) -> float:
        return k * self.b0

    def lambda_eff(self, k: int) -> float:
        """The margin left at ``f`` after the union bound over steps, products and grinding,
        ``k·b₀ − (N − λ)``; at least ``λ`` from ``k = ⌈N/b₀⌉``."""
        return k * self.b0 - (self.N - self.lam)


def k_floor(bands: Bands, c: DeclaredComputation, *, k: int, T: int) -> KFloor:
    """The band file's ``e_max``, ``N``, ``f`` and ``λ`` if it has a sizing, else computed from
    ``C`` (``q_max``, the unit roundoffs, ``λ``, ``T``, ``M`` and ``G``). ``c`` is ``C_ANTI``
    either way, as in the sweep's ``floor_context``. The band file's ``e_max`` is the
    ``e_m`` its ``k`` was sized at, the largest calibration ``e_m`` since F18; the computed
    one is ``e_m(q_max)`` at ratio 1, the advance sizing (no calibration products to read)."""
    sz = bands.stats.get("sizing") if bands.stats else None
    if sz:
        e, N, f, lam = float(sz["e_max"]), float(sz["N"]), float(sz["f"]), float(sz["lam"])
        src = "band file"
    else:
        eps = UNIT_ROUNDOFF[c.operand_dtype], UNIT_ROUNDOFF[c.accumulator_dtype]
        e = e_m(max(s.q for s in c.products), *eps)
        N, f, lam = bit_budget(LAMBDA, T, c.M, LOG2_G), F_TARGET, float(LAMBDA)
        src = "computed"
    return KFloor(tau=bands.tau, e_max=e, N=N, b0=b0(f, bands.tau, e), lam=lam, k_conf=k,
                  source=src)


def guaranteed_rate(f: float, *, C: float, B: int, ratio: float = GRADIENT_RATIO) -> float:
    """The smallest substitution rate guaranteed detected at floor ``f``, ``f·C/(√B·ratio)``:
    (12.1) solved for the rate, with the measured coherence ``C`` (§5.8)."""
    return f * C / (B ** 0.5 * ratio)


def covering_k(floor0: float, N: float, *, C: float, B: int, ratio: float = GRADIENT_RATIO,
               k_max: int = 1024) -> int:
    """The smallest ``k`` whose guaranteed rate covers one poisoned record per batch,
    ``rate·B ≤ 1``, at floor ``floor0·2^{N/k}``; ``ValueError`` if none up to ``k_max``."""
    for k in range(1, k_max + 1):
        if guaranteed_rate(floor0 * 2.0 ** (N / k), C=C, B=B, ratio=ratio) * B <= 1.0:
            return k
    raise ValueError(f"no k ≤ {k_max} covers one record of {B} (floor0 {floor0:.4g}, C {C})")


def k_top(k_conf: int, k_cover: int) -> int:
    """§5.8's largest ``k``: the larger of ``2·k_conf`` and ``k_cover``."""
    return max(2 * k_conf, k_cover)


# ---- predicted counts ---------------------------------------------------------------------


def measure_flops_per_column(c: DeclaredComputation) -> int:
    """``S = Σ_m 2(q·w + n·q + n·w)``: one challenge's ``B·r``, ``A·(B·r)`` and ``P·r`` over
    every product, bias terms left out."""
    return sum(2 * (s.a_shape[1] * s.b_shape[1] + s.a_shape[0] * s.a_shape[1]
                    + s.a_shape[0] * s.b_shape[1]) for s in c.products)


def measure_flops_term_scale(c: DeclaredComputation) -> int:
    """``S_U = Σ_m 2(q·w + n·q)``: the term scale's ``B²·1`` and ``A²·(B²·1)`` (F18) over every
    product, once per step whatever ``k``; the squares are elementwise and count 0."""
    return sum(2 * (s.a_shape[1] * s.b_shape[1] + s.a_shape[0] * s.a_shape[1])
               for s in c.products)


def predicted_hashes(c: DeclaredComputation, k: int) -> dict[str, int]:
    """Check 5's hashing at ``k``: one keyed XOF per label ``(m, j)``, the label in and
    ``4·w`` bytes out (P8b)."""
    calls = c.M * k
    return {"hash_calls": calls, "hash_in_bytes": len(encode_label(1, 1)) * calls,
            "hash_out_bytes": 4 * k * sum(s.width for s in c.products)}


# ---- the passes ---------------------------------------------------------------------------


class _Section:
    __slots__ = ("c0", "name", "owner", "t0")

    def __init__(self, owner: Sections, name: str) -> None:
        self.owner, self.name = owner, name

    def __enter__(self) -> _Section:
        self.c0 = self.owner.counts()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        dt = time.perf_counter() - self.t0
        row = self.owner.rows.setdefault(self.name, {"time_s": 0.0, "calls": 0,
                                                     **dict.fromkeys(COUNT_FIELDS, 0)})
        row["time_s"] += dt
        row["calls"] += 1
        for f, a, b in zip(COUNT_FIELDS, self.c0, self.owner.counts()):
            row[f] += b - a
        return False


class Sections:
    """The verifier's metrics seam for one pass: wall time and calls per section name and, with
    ``counters`` (``metrics.counting``'s pair), the FLOPs and hashing inside each."""

    def __init__(self, counters: tuple[Any, Any] | None = None) -> None:
        self.counters = counters
        self.rows: dict[str, dict[str, Any]] = {}

    def counts(self) -> tuple[int, ...]:
        if self.counters is None:
            return ()
        fl, tally = self.counters
        return fl.total, tally.calls, tally.bytes_in, tally.bytes_out

    def __call__(self, name: str) -> _Section:
        return _Section(self, name)


def base_context(c: DeclaredComputation, store: TranscriptStore, *, step: int, seed: int,
                 h_D: bytes, n_records: int, k: int) -> StepContext:
    """The step's context as the driver builds it at step ``step > 1``, with the store's own
    ``W_t`` hashes as check 7's expected hashes (see the module docstring)."""
    w_t = [(c.w_t_index(n), store.leaf(c.w_t_index(n))) for n in c.weight_names]
    return StepContext.for_computation(
        c, step=step, indices=tuple(schedule(step, c.n_s, n_records, seed)), h_D=h_D,
        n_records=n_records, prev_w_hashes=tuple(leaf_hashes_of(c, w_t)), chain_check_id="7",
        k=k)


def full_step(store: TranscriptStore, c: DeclaredComputation, base: StepContext,
              bands: Bands) -> tuple[Rejection | None, dict[str, float], StepContext]:
    """Every check in the default order on a fresh copy of ``base``, timed per slot as the
    driver times them; ``(rejection, seconds per slot, context)``. The replay is dropped."""
    ctx = dataclasses.replace(base, state=StepState(), stats=StepStats())
    times: dict[str, float] = {}
    rej = None
    try:
        for slot in DEFAULT_ORDER:
            t0 = time.perf_counter()
            rej = CHECKS[slot](store, c, ctx, bands)
            times[slot] = time.perf_counter() - t0
            if rej is not None:
                break
    finally:
        ctx.state.replay = None
    return rej, times, ctx


def check5_pass(store: TranscriptStore, c: DeclaredComputation, ready: StepContext,
                bands: Bands, k: int, *, section: Sections | None = None
                ) -> tuple[Rejection | None, StepStats]:
    """Check 5 alone at ``k``, on what check 2's read left in ``ready``; ``(rejection, stats)``.
    Its replay is dropped."""
    s = ready.state
    ctx = dataclasses.replace(ready, k=k, section=section, stats=StepStats(),
                              state=StepState(claimed_root=s.claimed_root,
                                              leaf_hashes=s.leaf_hashes, leaves=s.leaves))
    try:
        rej = CHECKS["5"](store, c, ctx, bands)
    finally:
        ctx.state.replay = None
    return rej, ctx.stats


# ---- comparisons and fits -----------------------------------------------------------------


def _scales(st: StepStats) -> list[tuple[int, str, str, str, str]]:
    """What check 5 measures apart from the challenges, per product, as exact strings."""
    return [(p.m, float(p.kappa).hex(), float(p.nu).hex(), float(p.p_abs1).hex(),
             float(p.p_norm).hex()) for p in st.products]


def _prefix(st: StepStats, ref: StepStats, j: int) -> tuple[float, int, int]:
    """The first ``j`` normalized residuals of ``st`` against ``ref``'s: the largest relative
    difference, the count equal bit for bit, and the count compared."""
    a = np.array([p.normalized[:j] for p in st.products], dtype=np.float64)
    b = np.array([p.normalized[:j] for p in ref.products], dtype=np.float64)
    d = np.abs(a - b)
    rel = np.divide(d, np.abs(b), out=d.copy(), where=b != 0)
    return float(rel.max(initial=0.0)), int(np.count_nonzero(a == b)), int(a.size)


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float]:
    """Least squares ``y = slope·x + intercept``, with ``R²`` and the largest residual
    relative to ``y``."""
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2,
            "max_rel_resid": float(np.max(np.abs(y - pred) / np.abs(y)))}


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


def honest_records(run_dir: Path, step: int) -> dict[str, Any]:
    """The honest run's own records of the stored step: its verifier times and its step record,
    and its check-5 counts (the run counted step 1 only; check 5's counts don't depend on the
    step). Each is empty or ``None`` when the run didn't write it."""
    p = Path(run_dir) / "records.jsonl"
    rows = read_records(p) if p.exists() else []

    def honest(r: dict[str, Any], record: str) -> bool:
        return r["record"] == record and r.get("scenario") == "honest"

    times = {r["component"]: r["time_s"] for r in rows
             if honest(r, "time") and r.get("step") == step and r.get("side") == "verifier"}
    counts: dict[str, Any] = {}
    for r in rows:
        if (honest(r, "count") and r.get("side") == "verifier"
                and r.get("component") in CHECK5_PARTS and r["component"] not in counts):
            counts[r["component"]] = {"step": r["step"], "calls": r["calls"],
                                      **{f: r[f] for f in COUNT_FIELDS}}
    steps = [r for r in rows if honest(r, "step") and r.get("step") == step]
    return {"times_s": times, "counts": counts, "step": steps[-1] if steps else None}


# ---- the run ------------------------------------------------------------------------------


@dataclass
class KResult:
    source: dict[str, Any]
    points: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def run_k(c: DeclaredComputation, kept: KeptStep, bands: Bands, *, seed: int, h_D: bytes,
          n_records: int, k_conf: int, T: int, top: int | None = None,
          repeats: int = DEFAULT_REPEATS, device: str = "cpu",
          writer: MetricsWriter | None = None,
          out: Callable[[str], None] = print) -> KResult:
    """The three passes over ``k = 1 … top`` on ``kept`` (default ``top``: :func:`k_top`)."""
    if (h := stored_band_hash(kept.run_dir)) != bands.source:
        raise ValueError(f"{kept.run_dir} was judged by band file {h}, not this run's "
                         f"{bands.source}")
    floor = k_floor(bands, c, k=k_conf, T=T)
    k_cover = covering_k(floor.floor0, floor.N, C=COVER_COHERENCE, B=c.n_s)
    top = k_top(k_conf, k_cover) if top is None else top
    if top < max(2, k_conf):
        raise ValueError(f"k_top {top}: the fit needs 2 and the reproduction k {k_conf}")
    ks = list(range(1, top + 1))
    store = kept.store
    base = base_context(c, store, step=kept.step, seed=seed, h_D=h_D, n_records=n_records,
                        k=k_conf)
    t0 = time.perf_counter()
    rej, _, ready = full_step(store, c, base, bands)
    warm_s = time.perf_counter() - t0
    if rej is not None:
        raise RuntimeError(f"the stored step {kept.step} was rejected: {rej}")
    full_equal = kept.honest is None or _stats_key(ready.stats) == _stats_key(kept.honest)
    out(f"full step at k {k_conf}: accepted in {warm_s:.2f} s (warm-up), numbers "
        + ("equal to the archive's" if full_equal else "DIFFERENT from the archive's"))

    timed: dict[int, list[float]] = {k: [] for k in ks}
    glue: dict[int, list[float]] = {k: [] for k in ks}
    measure: dict[int, list[float]] = {k: [] for k in ks}
    calls: dict[int, dict[str, int]] = {}
    verdicts: dict[int, Rejection | None] = {}
    stats: dict[int, StepStats] = {}
    keys: dict[int, tuple[Any, ...]] = {}
    repeatable: dict[int, bool] = dict.fromkeys(ks, True)
    slots: list[dict[str, float]] = []
    for r in range(repeats):
        rej, slot_s, ready = full_step(store, c, base, bands)
        if rej is not None:
            raise RuntimeError(f"the stored step {kept.step} was rejected in round {r + 1}: "
                               f"{rej}")
        slots.append(slot_s)
        for k in (ks if r % 2 == 0 else ks[::-1]):
            sec = Sections()
            t0 = time.perf_counter()
            rej, st = check5_pass(store, c, ready, bands, k, section=sec)
            timed[k].append(time.perf_counter() - t0)
            glue[k].append(sec.rows["5.glue"]["time_s"])
            measure[k].append(sec.rows.get("5.measure", {"time_s": 0.0})["time_s"])
            calls[k] = {n: row["calls"] for n, row in sec.rows.items()}
            key = _stats_key(st)[0]
            if k in keys:
                repeatable[k] &= key == keys[k]
            else:
                keys[k], stats[k], verdicts[k] = key, st, rej
        out(f"round {r + 1}/{repeats}: full step {sum(slot_s.values()):.2f} s, check 5 at "
            f"k {ks[0]}…{ks[-1]} {sum(timed[k][-1] for k in ks):.1f} s")

    probe = memory_probe(device)
    memory: dict[int, tuple[int | None, int | None]] = {}
    for k in ks:
        gc.collect()
        probe.reset()
        start = probe.sample()[1]
        check5_pass(store, c, ready, bands, k)
        memory[k] = (start, probe.sample()[0])

    counted: dict[int, dict[str, dict[str, Any]]] = {}
    for k in ks:
        with counting() as (fl, tally):
            sec = Sections((fl, tally))
            check5_pass(store, c, ready, bands, k, section=sec)
            total = dict(zip(COUNT_FIELDS, (fl.total, tally.calls, tally.bytes_in,
                                            tally.bytes_out), strict=True))
        counted[k] = {"5": total, **sec.rows}

    S = measure_flops_per_column(c)
    S_U = measure_flops_term_scale(c)
    ref = stats[k_conf]
    honest = honest_records(kept.run_dir, kept.step)
    archive_equal = None if kept.honest is None else keys[k_conf] == _stats_key(kept.honest)[0]
    rho_conf = step_scores(ref, bands, k=k_conf)["rho_5"]
    rho_honest = None if honest["step"] is None else honest["step"]["rho_5"]
    points: list[dict[str, Any]] = []
    for k in ks:
        st, (start, peak) = stats[k], memory[k]
        cnt = counted[k]
        f = floor.f_ach(k)
        rate = guaranteed_rate(f, C=COVER_COHERENCE, B=c.n_s)
        prefix = _prefix(st, ref, min(k, k_conf))
        points.append({
            "k": k, "verdict": "accept" if verdicts[k] is None else "reject",
            "rejection": None if verdicts[k] is None else str(verdicts[k]),
            "rho_5": step_scores(st, bands, k=k)["rho_5"], "max_normalized": st.max_normalized(),
            "time_s": _median(timed[k]), "times_s": timed[k],
            "glue_s": _median(glue[k]), "glue_times_s": glue[k],
            "measure_s": _median(measure[k]), "measure_times_s": measure[k],
            "glue_calls": calls[k].get("5.glue"), "measure_calls": calls[k].get("5.measure"),
            "mem_start_bytes": start, "mem_peak_bytes": peak,
            "mem_growth_bytes": None if start is None or peak is None else peak - start,
            "mem_source": probe.source,
            **cnt["5"],
            "flops_glue": cnt.get("5.glue", {}).get("flops"),
            "flops_measure": cnt.get("5.measure", {}).get("flops"),
            "flops_measure_pred": S * (k + 1) + S_U,
            **{f"{n}_pred": v for n, v in predicted_hashes(c, k).items()},
            "scales_equal": _scales(st) == _scales(ref), "repeatable": repeatable[k],
            "prefix_j": min(k, k_conf), "prefix_max_rel": prefix[0],
            "prefix_equal": prefix[1], "prefix_count": prefix[2],
            "bits": floor.bits(k), "lambda_eff": floor.lambda_eff(k), "f_ach": f,
            "rate_c1": rate, "covers_c1": rate * c.n_s <= 1.0,
        })

    def fit_over(sel: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        return {n: linear_fit([p["k"] for p in sel], [p[n] for p in sel]) for n in TIMES}

    fits = fit_over(points)
    # at k = 1, B·r and P·r are matrix-vector products, which BLAS runs with other kernels
    fits2 = fit_over(points[1:]) if len(points) > 2 else None
    slope = (fits2 or fits)["measure_s"]["slope"]
    other = {s: _median([x[s] for x in slots]) for s in DEFAULT_ORDER}
    accepted = all(p["verdict"] == "accept" for p in points)
    kappa_equal = all(p["scales_equal"] for p in points)
    reproduced = (full_equal and archive_equal is not False
                  and (rho_honest is None or rho_conf == rho_honest))
    summary = {
        "k_conf": k_conf, "k_top": top, "k_cover": k_cover, "fits": fits, "fits_from_2": fits2,
        "flops_exact": all(p["flops_measure"] == p["flops_measure_pred"] for p in points),
        "glue_flops_constant": len({p["flops_glue"] for p in points}) == 1,
        "hashes_exact": all(p[n] == p[f"{n}_pred"] for p in points
                            for n in COUNT_FIELDS[1:]),
        "marginal_from": 1 if fits2 is None else 2,
        "gflops_per_s_marginal": S / slope / 1e9 if slope > 0 else None,
        "us_per_product_challenge": slope / c.M * 1e6,
        "full_step_equal": full_equal, "archive_equal": archive_equal,
        "rho_5_conf": rho_conf, "rho_5_honest": rho_honest, "reproduced": reproduced,
        "accepted": accepted, "kappa_equal": kappa_equal,
        "repeatable": all(repeatable.values()),
        "passed": reproduced and accepted and kappa_equal,
    }
    source = {
        "step": kept.step, "root": kept.root.hex(), "path": str(kept.path),
        "honest_run": str(kept.run_dir), "band_file_hash": bands.source,
        "k_conf": k_conf, "k_top": top, "k_cover": k_cover, "cover_coherence": COVER_COHERENCE,
        "gradient_ratio": GRADIENT_RATIO, "B": c.n_s, "floor_source": floor.source,
        "floor0": floor.floor0, "c": C_ANTI, "tau": floor.tau, "e_max": floor.e_max,
        "N": floor.N, "b0": floor.b0, "lam": floor.lam, "repeats": repeats,
        "warmup_s": warm_s, "slot_s": other, "slot_times_s": slots,
        "other_s": sum(v for s, v in other.items() if s != "5"),
        "honest_times_s": honest["times_s"], "honest_counts": honest["counts"],
        "honest_step": None if honest["step"] is None else
        {n: honest["step"].get(n) for n in ("verdict", "rho_5", "rho_kappa", "rho_6")},
        "M": c.M, "sum_width": sum(s.width for s in c.products), "S": S, "S_U": S_U,
    }
    if writer is not None:
        writer.write([{"record": "k_source", "run": writer.run, **source}])
        writer.write({"record": "k_point", "run": writer.run, "step": kept.step, **p}
                     for p in points)
        writer.write([{"record": "k_summary", "run": writer.run, "step": kept.step, **summary}])
    return KResult(source, points, summary)


# ---- the report ---------------------------------------------------------------------------


def _g(x: float | None, fmt: str = ".4g") -> str:
    return "-" if x is None else format(x, fmt)


def report(r: KResult, out: Callable[[str], None] = print) -> None:
    s, sm = r.source, r.summary
    out(f"k_conf {s['k_conf']}, k_cover {s['k_cover']} (C {s['cover_coherence']}, "
        f"‖δg‖/‖g‖ {s['gradient_ratio']}, B {s['B']}), k_top {s['k_top']}; floor "
        f"c·τ·e_max = {s['floor0']:.4g}, N {s['N']:.2f}, b₀ {s['b0']:.2f} ({s['floor_source']})")
    out(f"{'k':>3} {'check5 s':>9} {'glue s':>7} {'meas s':>7} {'mem GB':>7} "
        f"{'GFLOP':>8} {'ρ_5':>6} {'f_ach':>9} {'k·b₀':>6} {'λ_eff':>6} {'rate·B':>9} "
        f"{'κ=':>3} {'prefix eq':>13} {'max rel':>8}")
    for p in r.points:
        mem = p["mem_growth_bytes"]
        eq = f"{p['prefix_equal']}/{p['prefix_count']}"
        out(f"{p['k']:>3} {p['time_s']:>9.3f} {p['glue_s']:>7.3f} {p['measure_s']:>7.3f} "
            f"{_g(None if mem is None else mem / 1e9, '.3f'):>7} {p['flops'] / 1e9:>8.2f} "
            f"{p['rho_5']:>6.3f} {p['f_ach']:>9.3g} {p['bits']:>6.1f} {p['lambda_eff']:>6.1f} "
            f"{p['rate_c1'] * s['B']:>9.3g} {'y' if p['scales_equal'] else 'N':>3} "
            f"{eq:>13} {_g(p['prefix_max_rel'], '.2g'):>8}")
    for lo, fits in ((1, sm["fits"]), (2, sm["fits_from_2"])):
        for n, f in (fits or {}).items():
            out(f"fit {n}, k ≥ {lo}: {f['slope'] * 1e3:.2f} ms per k + {f['intercept']:.3f} s, "
                f"R² {f['r2']:.4f}, largest residual {f['max_rel_resid']:.1%}")
    out(f"marginal measure throughput (k ≥ {sm['marginal_from']}) "
        f"{_g(sm['gflops_per_s_marginal'], '.1f')} GFLOP/s "
        f"(S = {s['S']:,} FLOPs per challenge), {sm['us_per_product_challenge']:.3f} µs per "
        "product per challenge")
    out(f"counts: measure FLOPs = S·(k+1) + S_U {sm['flops_exact']}, glue FLOPs constant "
        f"{sm['glue_flops_constant']}, hashes as predicted {sm['hashes_exact']}")
    other = ", ".join(f"{k} {v * 1e3:.1f}" for k, v in s["slot_s"].items())
    out(f"full step at k {s['k_conf']}, median ms: {other}; other checks {s['other_s']:.3f} s")
    if s["honest_times_s"]:
        out("honest run's step-4 times, ms: " + ", ".join(
            f"{k} {v * 1e3:.1f}" for k, v in s["honest_times_s"].items()))
    out(f"reproduction at k {s['k_conf']}: full step {sm['full_step_equal']}, check 5 against "
        f"the archive {sm['archive_equal']}, ρ_5 {sm['rho_5_conf']} against the step record's "
        f"{sm['rho_5_honest']}; every pass accepted {sm['accepted']}, κ equal at every k "
        f"{sm['kappa_equal']}, repeatable {sm['repeatable']}")
    out("PASS" if r.passed else "FAIL")


def open_k_writer(cfg: RunConfig, c: DeclaredComputation, bands: Bands, *, h_D: bytes,
                  n_records: int, k: int, extra: dict[str, Any]) -> MetricsWriter:
    """A writer under ``$VERIF_OUTPUT_DIR/k_tunability/`` with ``cheats.open_writer``'s
    environment fields, less the poisoning ones."""
    inst = {"n_s": c.n_s, "eta": c.eta, "k": k, "n_records": n_records, "M": c.M,
            "n_leaves": c.n_leaves, **extra}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics")}, "llama": inst}
    return MetricsWriter(cfg.output_dir / RUN_NAME, RUN_NAME, device=cfg.device,
                         model=cfg.model, corpus=backdoorllm.SOURCE, attack=cfg.attack,
                         seed=cfg.seed, config=config, band_file_hash=bands.source, h_D=h_D,
                         extra={"band_source": bands.source, **inst})


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run-dir", type=Path, default=None,
                   help="the honest run directory of the stored step (default "
                        "$VERIF_OUTPUT_DIR/honest-<model>-<attack>-bpr0-s<seed>)")
    p.add_argument("--k-top", type=int, default=None,
                   help="the largest k (default max(2·VERIF_K, k_cover))")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                   help="timed rounds; the median per k")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write records to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    if args.repeats < 1 or (args.k_top is not None and args.k_top < 2):
        p.error("need --repeats ≥ 1 and --k-top ≥ 2")
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    # read-only; never refitted here (P10a)
    bands = load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D, k=pc.k))
    print(f"FS12 k tunability: {cfg.model}@{cfg.model_revision[:12]}, k {pc.k}, step {STEP}, "
          f"band file {pc.band_file} ({bands.source[:16]}…), τ {bands.tau:.2f}")
    run_dir = args.run_dir or cfg.output_dir / honest_run_name(cfg.model, cfg.attack, cfg.seed)
    t0 = time.perf_counter()
    kept = load_kept_step(c, run_dir, step=STEP)
    print(f"stored step {STEP}: {kept.path}, h {kept.root.hex()[:16]}…, read and re-hashed in "
          f"{time.perf_counter() - t0:.1f} s")
    mw = (open_k_writer(cfg, c, bands, h_D=h_D, n_records=len(D), k=pc.k,
                        extra={"step": STEP, "run_dir": str(kept.run_dir),
                               "k_top": args.k_top, "repeats": args.repeats})
          if use_metrics else None)
    try:
        r = run_k(c, kept, bands, seed=cfg.seed, h_D=h_D, n_records=len(D), k_conf=pc.k,
                  T=cfg.steps, top=args.k_top, repeats=args.repeats, device=cfg.device,
                  writer=mw)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")
    report(r)
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
