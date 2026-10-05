"""A13: the flipped matmul on step 4 and the planted-error sweep (P10c, S6f, EQ10).

    .venv/bin/python -m verification.runs.flipped_matmul_sweep
        [--x X ...] [--trials N] [--shapes entry entry2 dense] [--classes CLS ...]
        [--metrics | --no-metrics]

Set up as the honest run (``runs/cheats.py``), judged by the frozen band file. Step 4 is the
first judged step: the bands were fitted on steps 1–3 (P10c).

1. **Honest base.** Four honest steps; the run must be accepted. Step 4's transcript is kept.
2. **The flipped-matmul cheat.** Four steps where step 4's ``dF`` (``P_m`` with
   ``m = m_of("dF")``, the output layer's input gradient and the product that sets ``k``) has its
   largest entry's sign flipped after capture. Training stays honest, and ``dF`` is an input
   gradient, so check 6a doesn't see it. Declared: ``(4, "5")`` on that product (S6d). The S6b
   oracle fails the run anywhere else.
3. **The sweep** (S6f) over step 4's kept transcript. One target product per matmul class (the
   middle layer, member ``(0, 0)`` of a batched product). Per trial it plants an error ``Δ`` of
   relative size ``f = ‖Δ‖_F/‖P‖_F`` in the target, re-roots by re-hashing only that leaf and
   its path (``store.perturb_leaf``), draws the challenges from the new root, and measures and
   judges the one product with check 5's own code (``checks._measure_run``,
   ``checks._judge_product``). The other products are honest and are not rechecked. After each
   target the honest leaf goes back and the original root must return.

**The sweep's scale (EQ10).** ``x = f/(τ·e_m)``, so ``x = 1`` is an error whose ``‖Δ·r‖`` is
``τ`` band units for an average challenge. ``f = x·τ·e_m`` for each grid point ``x``. Shapes:

- ``entry``: one random entry moved by ``±f·‖P‖_F`` (concentrated, the spec's §8 worry);
- ``entry2``: two random entries, each moved by ``±f·‖P‖_F/√2``;
- ``dense``: Gaussian noise scaled to ``‖Δ‖_F = f·‖P‖_F``.

The realized ``f`` (``‖P' − P‖_F/‖P‖_F`` in float64, after the fp32 rounding of ``P'``) is
logged with every trial, with each challenge's normalized residual and the verdict.

**Per class and shape it reports:**

- ``f50``: the smallest swept ``f`` where at least half the trials were rejected;
- ``f_all``: the smallest swept ``f`` from which every trial was rejected, at every larger
  point too;
- ``ĉ``: the single-challenge miss constant fitted from trials at ``x ≥ 3``, where the miss
  rate ``p₁ ≈ c/x`` (``ĉ = misses / Σ k/x`` over those trials). For ``entry`` the theory value
  is ``σ_r = 0.577`` (``P(|r_j| < σ_r/x) = σ_r/x`` for ``r_j ~ U(−1, 1)``); the sizing uses
  ``c = 0.798``. A dense error has no ``1/x`` tail, so ``ĉ`` comes out near 0;
- ``f_ach``: the sizing floor ``c·τ·e_m·2^{N/k}`` (the band file's ``N``, ``c``, ``τ``, ``k``),
  and ``f_ach(ĉ)``, the same formula with ``ĉ`` in place of ``c``.

``f_ach`` is the error size whose miss probability over all ``k`` challenges is ``2^{−N}``. It is
not a 50% point: ``f50`` sits near ``x ≈ 1``, about ``c·2^{N/k}`` times below it. The stop
condition is ``f_all > f_ach`` for any class and shape; it is printed and makes the run exit 1.

With metrics on (default ``VERIF_METRICS``), records go to
``$VERIF_OUTPUT_DIR/flipped_matmul_sweep/``: the two runs' step records and oracle records, one
``sweep_point`` record per (class, shape, ``x``), one ``sweep_summary`` per (class, shape), and
every trial in ``sweep_trials.npz``. Exits 1 unless the honest base is accepted, the flip is
rejected exactly at its declared point, and no stop condition holds.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from setup.config import load_config, setup_determinism
from setup.data import schedule
from verification.computation.interface import DeclaredComputation, ProductSpec
from verification.parameters import LAMBDA, LOG2_G, load_protocol_config
from verification.prover.step import StepOutput
from verification.runs.cheats import (
    CheatEnv,
    band_sources,
    load_env,
    open_writer,
    report_oracle,
    run_cheat,
)
from verification.runs.metrics import MetricsWriter
from verification.runs.scenarios import (
    Expected,
    FlipProduct,
    KeepStep,
    Scenario,
    ScenarioResult,
)
from verification.transcript.store import InMemoryStore, perturb_leaf
from verification.verifier.bands import Bands, product_class

# Check 5's own measuring and judging code, so a sweep point is judged exactly as check 5
# judges that product.
from verification.verifier.checks import _judge_product, _measure_run
from verification.verifier.context import StepContext, StepStats
from verification.verifier.matmul_check.challenges import SIGMA_R
from verification.verifier.matmul_check.freivalds import MeasureScratch
from verification.verifier.matmul_check.sizing import C_ANTI, bit_budget, e_m, f_achieved

__all__ = ["RUN_NAME", "SWEEP_STEP", "SHAPES", "DEFAULT_X", "DEFAULT_TRIALS", "FIT_X_MIN",
           "Held", "Floor", "SweepResult", "sweep_targets", "hold_targets", "plant",
           "sweep_target", "floor_context", "summarize", "run_sweep", "main"]

RUN_NAME = "flipped_matmul_sweep"
SWEEP_STEP = 4  # the first judged step (P10c)
FLIP_PRODUCT = "dF"  # the binding product (largest q), an input gradient: 6a can't see it
SHAPES = ("entry", "entry2", "dense")
DEFAULT_X = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
DEFAULT_TRIALS = 200  # EQ10
FIT_X_MIN = 3.0  # ĉ is fitted where p₁ ≈ c/x holds, above the honest residual's reach
TRIALS_FILE = "sweep_trials.npz"
REASONS = ("accepted", "residual", "kappa", "non-finite")


# ---- targets and their honest operands --------------------------------------------------


def sweep_targets(c: DeclaredComputation,
                  classes: Sequence[str] | None = None) -> list[ProductSpec]:
    """One product per matmul class, in canonical order: the class's middle layer (the lower
    middle for an even count) and its first member, or the class's only product."""
    by_cls: dict[str, list[ProductSpec]] = {}
    for s in c.products:
        by_cls.setdefault(product_class(c, s), []).append(s)
    if classes is not None:
        unknown = sorted(set(classes) - by_cls.keys())
        if unknown:
            raise ValueError(f"unknown product classes {unknown}")
    out = []
    for cls, specs in by_cls.items():
        if classes is not None and cls not in classes:
            continue
        layers = sorted({s.layer for s in specs if s.layer is not None})
        cand = [s for s in specs if s.layer == layers[(len(layers) - 1) // 2]] if layers else specs
        out.append(min(cand, key=lambda s: (s.member or (), s.m)))
    return sorted(out, key=lambda s: s.m)


@dataclass
class Held:
    """A target's honest operands and committed leaf, held across its trials."""

    spec: ProductSpec
    cls: str
    a: torch.Tensor
    b: torch.Tensor
    leaf: torch.Tensor  # the committed leaf object, put back after the trials
    p: torch.Tensor  # its fp32 view, as check 5 reads it
    p_norm: float  # ‖P‖_F in float64


def hold_targets(c: DeclaredComputation, store: InMemoryStore,
                 specs: Sequence[ProductSpec]) -> list[Held]:
    """Replay the step once, in canonical order, and keep each target's operands (cloned)."""
    want = {s.m: s for s in specs}
    replay = c.replay(store)
    ops: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for m in range(1, max(want) + 1):
        a, b = replay.operands(m)
        if m in want:
            ops[m] = (a.detach().to(torch.float32).clone(), b.detach().to(torch.float32).clone())
    del replay
    held = []
    for s in specs:
        leaf = store.leaf(c.product_index(s.m))
        p = leaf.detach().to(torch.float32)
        held.append(Held(s, product_class(c, s), *ops[s.m], leaf, p,
                         float(torch.linalg.vector_norm(p, dtype=torch.float64))))
    return held


# ---- planting ---------------------------------------------------------------------------


def plant(p: torch.Tensor, shape: str, f: float, p_norm: float,
          gen: torch.Generator) -> tuple[torch.Tensor, float]:
    """``P' = P + Δ`` in fp32 with ``‖Δ‖_F ≈ f·‖P‖_F`` of the given shape, and the realized
    ``f = ‖P' − P‖_F/‖P‖_F`` in float64."""
    size = f * p_norm
    flat = p.reshape(-1)
    if shape in ("entry", "entry2"):
        n = 1 if shape == "entry" else 2
        idx = torch.randperm(flat.numel(), generator=gen)[:n] if n > 1 else torch.randint(
            flat.numel(), (1,), generator=gen)
        sign = torch.randint(0, 2, (n,), generator=gen).to(torch.float64) * 2 - 1
        p2 = p.clone()
        p2.view(-1)[idx] = (flat[idx].to(torch.float64) + sign * size / math.sqrt(n)).to(
            torch.float32)
        d = p2.view(-1)[idx].to(torch.float64) - flat[idx].to(torch.float64)
        return p2, float(torch.linalg.vector_norm(d)) / p_norm
    if shape == "dense":
        g = torch.randn(p.shape, generator=gen, dtype=torch.float32)
        g.mul_(size / float(torch.linalg.vector_norm(g, dtype=torch.float64)))
        p2 = p + g
        d = torch.linalg.vector_norm(p2.to(torch.float64) - p.to(torch.float64))
        return p2, float(d) / p_norm
    raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")


# ---- the sweep ---------------------------------------------------------------------------


@dataclass
class Trials:
    """Every trial of one target and shape, one row each."""

    m: int
    cls: str
    shape: str
    e: float  # e_m(q)
    x_set: list[float] = field(default_factory=list)
    f_real: list[float] = field(default_factory=list)
    normalized: list[tuple[float, ...]] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)
    reason: list[str] = field(default_factory=list)  # one of REASONS


def _reason(detail: str | None) -> str:
    if detail is None:
        return "accepted"
    if "non-finite" in detail:
        return "non-finite"
    return "kappa" if "cancellation factor" in detail else "residual"


def sweep_target(c: DeclaredComputation, store: InMemoryStore, held: Held, ctx: StepContext,
                 bands: Bands, *, xs: Sequence[float], shapes: Sequence[str], trials: int,
                 gen: torch.Generator, scratch: MeasureScratch | None = None) -> list[Trials]:
    """Every trial of one target, then the honest leaf back; raises if the root doesn't
    return. ``ctx`` gives ``k`` and the unit roundoffs, and holds no stats afterwards."""
    s, idx = held.spec, c.product_index(held.spec.m)
    base = store.root
    e = e_m(s.q, ctx.eps_in, ctx.eps_acc)
    out = []
    try:
        for shape in shapes:
            tr = Trials(s.m, held.cls, shape, e)
            for x in xs:
                f = x * bands.tau * e
                for _ in range(trials):
                    p2, f_real = plant(held.p, shape, f, held.p_norm, gen)
                    root = perturb_leaf(c, store, idx, p2)
                    [mp] = _measure_run([s], [(held.a, held.b, p2)], h=root, k=ctx.k,
                                        eps_in=ctx.eps_in, eps_acc=ctx.eps_acc,
                                        scratch=scratch, guard=ctx.kappa_guard)
                    rej = _judge_product(c, s, mp, ctx, bands)
                    ctx.stats.products.clear()
                    tr.x_set.append(x)
                    tr.f_real.append(f_real)
                    tr.normalized.append(tuple(mp.normalized))
                    tr.kappa.append(mp.kappa)
                    tr.reason.append(_reason(None if rej is None else rej.detail))
            out.append(tr)
    finally:
        if perturb_leaf(c, store, idx, held.leaf) != base:
            raise AssertionError(f"P_{s.m}: the honest leaf did not restore the root")
    return out


def honest_measure(c: DeclaredComputation, store: InMemoryStore, held: Held,
                   ctx: StepContext, bands: Bands) -> tuple[float, ...]:
    """The target's normalized residuals on the unperturbed transcript, measured as the sweep
    measures (they must equal the run's check-5 numbers for that product)."""
    [mp] = _measure_run([held.spec], [(held.a, held.b, held.p)], h=store.root, k=ctx.k,
                        eps_in=ctx.eps_in, eps_acc=ctx.eps_acc, guard=ctx.kappa_guard)
    if _judge_product(c, held.spec, mp, ctx, bands) is not None:
        raise AssertionError(f"P_{held.spec.m}: the honest product is rejected")
    ctx.stats.products.clear()
    return tuple(mp.normalized)


# ---- summaries ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Floor:
    """What fixes the sizing floor ``f_ach = c·τ·e_m·2^{N/k}``."""

    N: float
    c: float
    tau: float
    k: int

    def f_ach(self, e: float, c: float | None = None) -> float:
        return f_achieved(self.c if c is None else c, self.tau, e, self.N, self.k)


def floor_context(bands: Bands, c: DeclaredComputation, *, k: int, T: int) -> Floor:
    """The band file's sizing (``N``, ``c``, ``τ``, ``k``) if it has one, else computed from
    ``λ``, ``T``, ``M`` and ``G`` with ``c = C_ANTI``."""
    sz = bands.stats.get("sizing") if bands.stats else None
    if sz:
        return Floor(float(sz["N"]), float(sz["c"]), bands.tau, int(sz["k"]))
    return Floor(bit_budget(LAMBDA, T, c.M, LOG2_G), C_ANTI, bands.tau, k)


def summarize(tr: Trials, floor: Floor, *, x_fit: float = FIT_X_MIN
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per grid point and overall, for one target and shape (module docstring)."""
    tau = floor.tau
    x_set = np.asarray(tr.x_set)
    f_real = np.asarray(tr.f_real)
    res = np.asarray(tr.normalized, dtype=np.float64).reshape(len(tr.x_set), -1)
    rejected = np.asarray([r != "accepted" for r in tr.reason])
    misses = (res <= tau)  # each challenge's own outcome
    k = res.shape[1]
    points = []
    for x in dict.fromkeys(tr.x_set):
        sel = x_set == x
        n = int(sel.sum())
        f = x * tau * tr.e
        reasons = {r: int(sum(1 for i in np.flatnonzero(sel) if tr.reason[i] == r))
                   for r in REASONS}
        points.append({"x": x, "f": f, "f_real_mean": float(f_real[sel].mean()),
                       "trials": n, "rejected": int(rejected[sel].sum()),
                       "reject_rate": float(rejected[sel].mean()),
                       "p1": float(misses[sel].mean()),
                       "p1_theory_entry": min(1.0, SIGMA_R / x), "reasons": reasons})
    f50 = next((p["f"] for p in points if p["reject_rate"] >= 0.5), None)
    f_all = None
    for p in reversed(points):
        if p["rejected"] < p["trials"]:
            break
        f_all = p["f"]
    fit = (f_real / (tau * tr.e)) >= x_fit
    denom = float((k / (f_real[fit] / (tau * tr.e))).sum()) if fit.any() else 0.0
    c_hat = float(misses[fit].sum()) / denom if denom > 0 else None
    f_ach = floor.f_ach(tr.e)
    summary = {"m": tr.m, "cls": tr.cls, "shape": tr.shape, "e_m": tr.e, "tau": tau,
               "k": k, "trials": len(tr.x_set), "f50": f50, "f_all": f_all,
               "x50": None if f50 is None else f50 / (tau * tr.e),
               "x_all": None if f_all is None else f_all / (tau * tr.e),
               "c_hat": c_hat, "c_fit_x_min": x_fit,
               "c_fit_vectors": int(fit.sum()) * k, "c_fit_misses": int(misses[fit].sum()),
               "f_ach": f_ach, "f_ach_c_hat": None if c_hat is None else floor.f_ach(tr.e, c_hat),
               "N": floor.N, "c_sizing": floor.c,
               "f_all_over_f_ach": None if f_all is None else f_all / f_ach,
               "stop": f_all is None or f_all > f_ach}
    return points, summary


# ---- the run ------------------------------------------------------------------------------


@dataclass
class SweepResult:
    honest: ScenarioResult
    flip: ScenarioResult
    summaries: list[dict[str, Any]]
    points: list[dict[str, Any]]
    trials: list[Trials]
    seconds: dict[str, float]  # per target name

    @property
    def stops(self) -> list[dict[str, Any]]:
        return [s for s in self.summaries if s["stop"]]


def _store_of(env: CheatEnv, out: StepOutput, t: int) -> InMemoryStore:
    from verification.commitment.leaves import dataset_tree
    tree = dataset_tree(env.c, env.D)
    return InMemoryStore.from_step(env.c, out, dataset_paths=[
        tree.path(i) for i in schedule(t, env.c.n_s, len(env.D))])


def run_sweep(env: CheatEnv, *, xs: Sequence[float] = DEFAULT_X,
              shapes: Sequence[str] = SHAPES, trials: int = DEFAULT_TRIALS,
              classes: Sequence[str] | None = None, seed: int = 0, t: int = SWEEP_STEP,
              flip_name: str = FLIP_PRODUCT, writer: MetricsWriter | None = None,
              out: Callable[[str], None] = print) -> SweepResult:
    """The honest base run, the flipped-matmul cheat, then the sweep on step ``t``."""
    c = env.c
    keep = KeepStep(t)
    honest = run_cheat(env, Scenario("honest", f"honest, step {t}'s transcript kept for the "
                                     "sweep", keep, Expected()), T=t, writer=writer, out=out)
    m_flip = c.m_of(flip_name)  # type: ignore[attr-defined]
    flip = run_cheat(env, Scenario("flip", f"largest entry of P_{m_flip} ({flip_name}) "
                                   f"sign-flipped after capture at step {t}",
                                   FlipProduct(t, m_flip), Expected(t, "5", "failed", m_flip)),
                     T=t, writer=writer, cheat_step=t, out=out)
    if keep.out is None or not honest.passed:
        out("the honest base run was not accepted: no sweep")
        return SweepResult(honest, flip, [], [], [], {})
    store = _store_of(env, keep.out, t)
    keep.out = None
    floor = floor_context(env.bands, c, k=env.k, T=t)
    pf = store.leaf(c.product_index(m_flip)).detach().to(torch.float64)
    x_flip = (2 * float(pf.abs().max()) / float(torch.linalg.vector_norm(pf))
              / (env.bands.tau * e_m(c.product(m_flip).q, *_eps(c))))
    out(f"flip: 2·max|P|/‖P‖_F on step {t}'s {flip_name} is x = {x_flip:.4g} band units")
    stats = honest.verifier.stats.get(t, StepStats())
    seen = {p.m: tuple(p.normalized) for p in stats.products}
    ctx = StepContext.for_computation(
        c, step=t, indices=tuple(schedule(t, c.n_s, len(env.D))), h_D=env.h_D,
        n_records=len(env.D), prev_w_hashes=(), chain_check_id="7", k=env.k)
    specs = sweep_targets(c, classes)
    held = hold_targets(c, store, specs)
    gen = torch.Generator().manual_seed(seed)
    scratch = MeasureScratch()
    summaries, points, all_trials, seconds = [], [], [], {}
    out(f"sweep on step {t}: {len(held)} targets, shapes {list(shapes)}, x {list(xs)}, "
        f"{trials} trials per point, τ {env.bands.tau:.2f}, k {env.k}, N {floor.N:.2f}")
    for h in held:
        honest_res = honest_measure(c, store, h, ctx, env.bands)
        if h.spec.m in seen and seen[h.spec.m] != honest_res:
            raise AssertionError(f"P_{h.spec.m}: the sweep's honest residuals {honest_res} "
                                 f"differ from check 5's {seen[h.spec.m]}")
        t0 = time.perf_counter()
        trs = sweep_target(c, store, h, ctx, env.bands, xs=xs, shapes=shapes, trials=trials,
                           gen=gen, scratch=scratch)
        seconds[h.spec.name] = time.perf_counter() - t0
        for tr in trs:
            pts, sm = summarize(tr, floor)
            sm.update(name=h.spec.name, q=h.spec.q, p_norm=h.p_norm,
                      honest_max=max(honest_res), seconds=seconds[h.spec.name])
            summaries.append(sm)
            points.extend({"m": tr.m, "cls": tr.cls, "shape": tr.shape, **p} for p in pts)
            all_trials.append(tr)
            if writer is not None:
                writer.write([{"record": "sweep_point", "run": writer.run, "step": t,
                               "m": tr.m, "cls": tr.cls, "shape": tr.shape, **p} for p in pts])
                writer.write([{"record": "sweep_summary", "run": writer.run, "step": t, **sm}])
        out(f"  {h.cls:<9} P_{h.spec.m:<5} {h.spec.name:<16} q {h.spec.q:>6}  "
            f"{seconds[h.spec.name]:7.1f} s  "
            + "  ".join(f"{s['shape']}: f_all {_g(s['f_all'])}"
                        for s in summaries[-len(trs):]))
    if writer is not None:
        save_trials(writer.dir / TRIALS_FILE, all_trials)
    return SweepResult(honest, flip, summaries, points, all_trials, seconds)


def _eps(c: DeclaredComputation) -> tuple[float, float]:
    from verification.parameters import UNIT_ROUNDOFF
    return UNIT_ROUNDOFF[c.operand_dtype], UNIT_ROUNDOFF[c.accumulator_dtype]


def save_trials(path: Path, trials: Sequence[Trials]) -> None:
    """Every trial, flat: target ``m``, class, shape, ``x``, realized ``f``, the ``k``
    normalized residuals, κ and the verdict."""
    rows = [(tr, i) for tr in trials for i in range(len(tr.x_set))]
    np.savez(path,
             m=np.asarray([tr.m for tr, _ in rows], dtype=np.int64),
             cls=np.asarray([tr.cls for tr, _ in rows]),
             shape=np.asarray([tr.shape for tr, _ in rows]),
             e_m=np.asarray([tr.e for tr, _ in rows]),
             x=np.asarray([tr.x_set[i] for tr, i in rows]),
             f_real=np.asarray([tr.f_real[i] for tr, i in rows]),
             normalized=np.asarray([tr.normalized[i] for tr, i in rows], dtype=np.float64),
             kappa=np.asarray([tr.kappa[i] for tr, i in rows]),
             reason=np.asarray([tr.reason[i] for tr, i in rows]))


def _g(x: float | None) -> str:
    return "-" if x is None else f"{x:.3g}"


def report_sweep(r: SweepResult, out: Callable[[str], None] = print) -> None:
    """Per class and shape: f50, f_all, ĉ, and both floors."""
    out(f"{'class':<9} {'P_m':>6} {'q':>6} {'shape':<7} {'x50':>6} {'x_all':>6} "
        f"{'f50':>9} {'f_all':>9} {'ĉ':>7} {'f_ach(ĉ)':>9} {'f_ach':>8} {'f_all/f_ach':>11}")
    for s in r.summaries:
        out(f"{s['cls']:<9} {s['m']:>6} {s['q']:>6} {s['shape']:<7} {_g(s['x50']):>6} "
            f"{_g(s['x_all']):>6} {_g(s['f50']):>9} {_g(s['f_all']):>9} {_g(s['c_hat']):>7} "
            f"{_g(s['f_ach_c_hat']):>9} {_g(s['f_ach']):>8} {_g(s['f_all_over_f_ach']):>11}"
            + ("  STOP" if s["stop"] else ""))
    stops = r.stops
    out("stop condition (f_all above f_ach, or never all rejected): "
        + ("none" if not stops else
           ", ".join(f"{s['cls']} {s['shape']}" for s in stops)))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--x", type=float, nargs="+", default=list(DEFAULT_X),
                   help="grid of x = f/(τ·e_m)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per point")
    p.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=SHAPES)
    p.add_argument("--classes", nargs="+", default=None, help="only these product classes")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    if args.trials < 1 or any(not x > 0 for x in args.x):
        p.error("need --trials ≥ 1 and every x > 0")
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    setup_determinism(cfg)
    env = load_env(cfg, pc)
    print(f"A13 flipped matmul and sweep: {cfg.model}@{cfg.model_revision[:12]}, k {env.k}, "
          f"step {SWEEP_STEP}, band file {pc.band_file} ({env.bands.source[:16]}…), "
          f"τ {env.bands.tau:.2f}")
    mw = (open_writer(cfg, RUN_NAME, env, {"x": list(args.x), "trials": args.trials,
                                           "shapes": list(args.shapes),
                                           "classes": args.classes, "sweep_step": SWEEP_STEP})
          if use_metrics else None)
    try:
        r = run_sweep(env, xs=args.x, shapes=args.shapes, trials=args.trials,
                      classes=args.classes, seed=cfg.seed, writer=mw)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")
    report_oracle([r.honest, r.flip], env.bands)
    report_sweep(r)
    shared = band_sources(env, [r.honest, r.flip], cfg.output_dir)
    print(f"band-file hash equal across the file, {len(env.verifiers.made)} verifiers and 2 "
          f"verdicts: {shared}")
    ok = r.honest.passed and r.flip.passed and bool(r.summaries) and not r.stops
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


