"""A13 and EV3: the flipped matmul on step 4 and the planted-error sweep (P10c, S6f, EQ10,
EVALUATION_SPEC §5.7).

    .venv/bin/python -m verification.runs.flipped_matmul_sweep
        [--source stored | in-process] [--run-dir DIR]
        [--x X ...] [--trials N] [--shapes entry entry2 row2 dense] [--classes CLS ...]
        [--deep-x X ...] [--deep-trials N] [--deep-shapes SHAPE ...]
        [--subdir NAME] [--metrics | --no-metrics]

Set up as the honest run (``runs/cheats.py``), judged by the frozen band file. Step 4 is the
first judged step: the bands were fitted on steps 1–3 (P10c).

1. **The transcript.** By default (``--source stored``) step 4 as the honest run stored it:
   ``run_verified.load_kept_step`` finds it through the run's ``kept_transcript`` record,
   hashes every leaf again and checks the root, and reads the run's check-5 residuals for the
   step from its residual archive. The run directory is ``$VERIF_OUTPUT_DIR/honest-<model>-
   <attack>-bpr0-s<seed>``, or ``--run-dir``; its band-file hash must be this band file's.
   ``--source in-process`` instead runs four honest steps here (the run must be accepted) and
   keeps step 4's transcript in memory. At one seed both give the same leaves, so the same
   sweep records.
2. **The flipped-matmul cheat.** Four steps where step 4's ``dF`` (``P_m`` with
   ``m = m_of("dF")``, the output layer's input gradient and the product that sets ``k``) has its
   largest entry's sign flipped after capture. Training stays honest, and ``dF`` is an input
   gradient, so check 6a doesn't see it. Declared: ``(4, "5")`` on that product (S6d). The S6b
   oracle fails the run anywhere else. It runs from either source.
3. **The sweep** (S6f) over step 4's transcript. The targets are one product per §5.7 class
   (``metrics.spec_class``: forward, input gradient, weight gradient, attention score,
   attention value, and the placeholder operand gradient), taken from the class's middle layer
   as its first product in canonical order (a batched product's first member), plus the
   binding product, the one with the largest ``q`` (``dF`` on the decoder). Per trial it plants
   an error ``Δ`` of relative size ``f = ‖Δ‖_F/‖P‖_F`` in the target, re-roots by re-hashing
   only that leaf and its path (``store.perturb_leaf``), draws the challenges from the new root,
   and measures and judges the one product with check 5's own code (``checks._measure_run``,
   ``checks._judge_product``). The other products are honest and are not rechecked. After each
   target the honest leaf goes back and the original root must return.
4. **Deep points.** At a few ``x`` the trials are 2,000 instead of 200 (``--deep-x``,
   ``--deep-trials``, ``--deep-shapes``; by default ``x`` 300, 1000 and 3000 on ``entry`` and
   ``row2``). 200 trials of ``k = 9`` challenges resolve a single-challenge miss rate down to
   ``1/1800 ≈ 5.6e-4``, about the rate at the sizing floor (``c/x_ach ≈ 7.7e-4`` at
   ``x_ach = c·2^{N/k} ≈ 1063`` on the test-scale band file). 2,000 resolve ``5.6e-5``, so
   ``row2`` expects about 49, 15 and 5 misses at those points, which bracket the floor.
   ``entry`` and ``row2`` are the shapes with a ``1/x`` tail; ``entry2`` and ``dense`` almost
   never miss there. The deep points draw from their own generator, so the grid's trials don't
   depend on them.
5. **Check 6.** One trial plants one entry of ``W_{t+1}`` outside its band, for the linear
   weight whose gradient is the weight-gradient target: entry ``i`` (random) becomes
   ``fl(W_t,i − η·G_i ± 2·τ_W·ε_W·(|W_t,i| + |η·G_i|))``, so ``ρ_i ≈ 2·τ_W`` (the factor 2 is a
   placeholder). It re-roots, runs check 2's read and check 6a as the verifier does, and must
   be rejected at 6a on that entry. Then the leaf goes back and the root must return.

**The sweep's scale (EQ10).** ``x = f/(τ·e_m)``, so ``x = 1`` is an error whose ``‖Δ·r‖`` is
``τ`` band units for an average challenge. ``f = x·τ·e_m`` for each grid point ``x``. Shapes:

- ``entry``: one random entry moved by ``±f·‖P‖_F`` (concentrated, the spec's §8 worry);
- ``entry2``: two random entries, almost always in different rows, each moved by
  ``±f·‖P‖_F/√2``;
- ``row2``: two entries in one row (a random row, two distinct random columns), each moved by
  the same ``±f·‖P‖_F/√2``. ``Δ·r = (δ/√2)(r_a + r_b)``, the worst rank-1 error for uniform
  challenges (Ball's cube-slicing bound), so this shape tests the sizing's ``c``;
- ``dense``: Gaussian noise scaled to ``‖Δ‖_F = f·‖P‖_F``.

The realized ``f`` (``‖P' − P‖_F/‖P‖_F`` in float64, after the fp32 rounding of ``P'``) is
logged with every trial, with each challenge's normalized residual and the verdict.

**Per point** (target, shape, phase, ``x``) it records the trials rejected, the challenges that
missed and the single-challenge miss rate ``p₁`` with its binomial standard error, the bound
``min(1, c/x)``, and the smallest rate the point resolves, ``1/(trials·k)``.

**Which norm.** §5.7 takes ``f`` against the honest ``‖P‖_F``, but check 5 normalizes by the
committed ``‖P'‖_F`` (spec §8.2's ``τ_m``), which a large error inflates: on ``dF`` at
``x = 3000`` (``f ≈ 1.5``), ``‖P'‖_F ≈ 1.8·‖P‖_F``, and ``p₁`` rises with it. So each trial also
logs ``‖P'‖_F/‖P‖_F``, each point the mean ratio and the bound ``c/x'`` with
``x' = x·‖P‖_F/‖P'‖_F`` (averaged over its trials), and each summary ``ĉ`` refitted on ``x'``.
The stop condition still uses ``c/x``, the stricter of the two.

**Per target, shape and phase it reports:**

- ``f50``: the smallest swept ``f`` where at least half the trials were rejected;
- ``f_all``: the smallest swept ``f`` from which every trial was rejected, at every larger
  point too;
- ``ĉ``: the single-challenge miss constant fitted from trials at ``x ≥ 3``, where the miss
  rate ``p₁ ≈ c/x`` (``ĉ = misses / Σ k/x`` over those trials), with its Poisson standard
  error. Theory values: ``entry`` ``σ_r = 0.577`` (``P(|r_j| < σ_r/x) = σ_r/x`` for
  ``r_j ~ U(−1, 1)``); ``row2`` ``√2·σ_r = √(2/3) = 0.816``, the bound the sizing uses as
  ``c = C_ANTI`` (the triangular density of ``r_a + r_b`` bends ``p₁`` down at small ``x``, so
  the fit lands a little below it); ``entry2`` and ``dense`` far below, since two entries in
  different rows miss only when both rows' residuals are small (``p₁ ∝ 1/x²``) and a dense
  error has no ``1/x`` tail;
- ``f_ach``: the sizing floor ``c·τ·e_m·2^{N/k}`` (the band file's ``N``, ``τ`` and ``k``, and
  ``c = C_ANTI``), and ``f_ach(ĉ)``, the same formula with ``ĉ`` in place of ``c``.

``f_ach`` is the error size whose miss probability over all ``k`` challenges is ``2^{−N}``. It is
not a 50% point: ``f50`` sits near ``x ≈ 1``, about ``c·2^{N/k}`` times below it. The stop
condition, for any target, shape and phase, is ``f_all > f_ach``, or no point where every trial
is rejected, or ``ĉ`` or a point's ``p₁`` above ``c = C_ANTI`` (``c/x``) by more than
``C_HAT_SIGMAS`` standard errors (a value at the bound sits above it about half the time by
chance alone). It is printed and makes the run exit 1.

With metrics on (default ``VERIF_METRICS``), records go to
``$VERIF_OUTPUT_DIR/flipped_matmul_sweep/``, or to its ``--subdir`` so an earlier run's records
stay: the runs' step records and oracle records, one ``sweep_source`` record (the transcript,
its root and the targets), one ``sweep_point`` record per (target, shape, phase, ``x``), one
``sweep_summary`` per (target, shape, phase), one ``sweep_check6``, and every trial in
``sweep_trials.npz``. Exits 1 unless the honest base (in process) is accepted, the flip is
rejected exactly at its declared point, the check-6 trial is rejected at 6a, and no stop
condition holds.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from verification.runs.metrics import MetricsWriter, spec_class
from verification.runs.run_verified import (
    KeptStep,
    honest_run_name,
    load_kept_step,
    stored_band_hash,
)
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
# judges that product; and checks 2 and 6a as the verifier runs them, for the check-6 trial.
from verification.verifier.checks import (
    _judge_product,
    _measure_run,
    check_2_commitment,
    check_6a_linear_update,
)
from verification.verifier.context import StepContext, StepState, StepStats
from verification.verifier.matmul_check.challenges import SIGMA_R
from verification.verifier.matmul_check.freivalds import MeasureScratch
from verification.verifier.matmul_check.sizing import C_ANTI, bit_budget, e_m, f_achieved

__all__ = ["RUN_NAME", "SWEEP_STEP", "SHAPES", "DEFAULT_X", "DEFAULT_TRIALS", "DEEP_X",
           "DEEP_TRIALS", "DEEP_SHAPES", "FIT_X_MIN", "CHECK6_FACTOR", "SOURCES", "Held",
           "Floor", "SweepResult", "binding_product", "sweep_targets", "hold_targets", "plant",
           "sweep_target", "check6_weight", "check6_trial", "floor_context", "summarize",
           "run_sweep", "main"]

RUN_NAME = "flipped_matmul_sweep"
SWEEP_STEP = 4  # the first judged step (P10c)
FLIP_PRODUCT = "dF"  # the binding product (largest q), an input gradient: 6a can't see it
SHAPES = ("entry", "entry2", "row2", "dense")
DEFAULT_X = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
             3000.0)
DEFAULT_TRIALS = 200  # EQ10
# §5.7's "2,000 at a few points": around the test-scale floor x_ach ≈ 1063, on the shapes with
# a 1/x tail (module docstring, 4)
DEEP_X = (300.0, 1000.0, 3000.0)
DEEP_TRIALS = 2000
DEEP_SHAPES = ("entry", "row2")
DEEP_SEED = 1_000_000  # added to the seed: the deep points and the check-6 trial draw from
CHECK6_SEED = 2_000_000  # their own generators, so the grid's draws don't depend on them
CHECK6_FACTOR = 2.0  # placeholder: the check-6 entry's ρ is about this many τ_W
FIT_X_MIN = 3.0  # ĉ is fitted where p₁ ≈ c/x holds, above the honest residual's reach
C_HAT_SIGMAS = 3.0  # ĉ or p₁ stops the run only when it exceeds c by this many standard errors
TRIALS_FILE = "sweep_trials.npz"
REASONS = ("accepted", "residual", "kappa", "non-finite")
SOURCES = ("stored", "in-process")


# ---- targets and their honest operands --------------------------------------------------


def binding_product(c: DeclaredComputation) -> ProductSpec:
    """The product that sets ``k``: the largest ``q`` (so ``e_max``), the first in canonical
    order on a tie. ``dF`` on the decoder."""
    return min(c.products, key=lambda s: (-s.q, s.m))


def _middle_first(specs: Sequence[ProductSpec]) -> ProductSpec:
    """The middle layer's (the lower middle for an even count) first product in canonical order,
    a batched product's first member; the first product when none is in a layer."""
    layers = sorted({s.layer for s in specs if s.layer is not None})
    cand = [s for s in specs if s.layer == layers[(len(layers) - 1) // 2]] if layers else specs
    return min(cand, key=lambda s: (s.member or (), s.m))


def sweep_targets(c: DeclaredComputation,
                  classes: Sequence[str] | None = None) -> list[ProductSpec]:
    """§5.7's targets in canonical order: one product per class (``metrics.spec_class``, picked
    by :func:`_middle_first`) and the binding product.

    ``classes`` first narrows the products to those whose §5.7 class, matmul class
    (``bands.product_class``) or name it lists. A value that names no product raises
    ``ValueError``."""
    pool = list(c.products)
    if classes is not None:
        keys = {s.m: {spec_class(c, s), product_class(c, s), s.name} for s in pool}
        unknown = sorted(set(classes) - set().union(*keys.values()))
        if unknown:
            raise ValueError(f"unknown product classes or names {unknown}")
        pool = [s for s in pool if keys[s.m] & set(classes)]
    by_cls: dict[str, list[ProductSpec]] = {}
    for s in pool:
        by_cls.setdefault(spec_class(c, s), []).append(s)
    out = {s.m: s for s in map(_middle_first, by_cls.values())}
    b = binding_product(c)
    if any(s.m == b.m for s in pool):
        out[b.m] = b
    return sorted(out.values(), key=lambda s: s.m)


@dataclass
class Held:
    """A target's honest operands and committed leaf, held across its trials."""

    spec: ProductSpec
    cls: str  # its matmul class (bands.product_class)
    a: torch.Tensor
    b: torch.Tensor
    leaf: torch.Tensor  # the committed leaf object, put back after the trials
    p: torch.Tensor  # its fp32 view, as check 5 reads it
    p_norm: float  # ‖P‖_F in float64
    bias: torch.Tensor | None = None  # a biased product's committed bias, as check 5 reads it
    spec_class: str = ""  # its §5.7 class (metrics.spec_class)
    binding: bool = False  # the binding product


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
    bind = binding_product(c).m
    held = []
    for s in specs:
        leaf = store.leaf(c.product_index(s.m))
        p = leaf.detach().to(torch.float32)
        bias = None if s.bias is None else store.leaf(c.w_t_index(s.bias)).detach().to(
            c.operand_dtype).to(torch.float32)
        held.append(Held(s, product_class(c, s), *ops[s.m], leaf, p,
                         float(torch.linalg.vector_norm(p, dtype=torch.float64)), bias,
                         spec_class(c, s), s.m == bind))
    return held


# ---- planting ---------------------------------------------------------------------------


def plant(p: torch.Tensor, shape: str, f: float, p_norm: float,
          gen: torch.Generator) -> tuple[torch.Tensor, float]:
    """``P' = P + Δ`` in fp32 with ``‖Δ‖_F ≈ f·‖P‖_F`` of the given shape, and the realized
    ``f = ‖P' − P‖_F/‖P‖_F`` in float64."""
    size = f * p_norm
    flat = p.reshape(-1)
    if shape in ("entry", "entry2", "row2"):
        n = 1 if shape == "entry" else 2
        if shape == "row2":  # one row, two distinct columns: Δ·r = (δ/√2)(r_a + r_b)
            rows, cols = p.shape[0], p.shape[-1]
            if p.dim() != 2 or cols < 2:
                raise ValueError(f"row2 needs a 2-D product with ≥ 2 columns, got {p.shape}")
            i = int(torch.randint(rows, (1,), generator=gen))
            idx = i * cols + torch.randperm(cols, generator=gen)[:2]
        elif n > 1:
            idx = torch.randperm(flat.numel(), generator=gen)[:n]
        else:
            idx = torch.randint(flat.numel(), (1,), generator=gen)
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
    """Every trial of one target, shape and phase, one row each."""

    m: int
    cls: str
    shape: str
    e: float  # e_m(q)
    x_set: list[float] = field(default_factory=list)
    f_real: list[float] = field(default_factory=list)
    norm_ratio: list[float] = field(default_factory=list)  # ‖P'‖_F/‖P‖_F; none reads as 1
    normalized: list[tuple[float, ...]] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)
    reason: list[str] = field(default_factory=list)  # one of REASONS
    spec_class: str = ""
    name: str = ""
    phase: str = "grid"  # "grid" or "deep"


def _reason(detail: str | None) -> str:
    if detail is None:
        return "accepted"
    if "non-finite" in detail:
        return "non-finite"
    return "kappa" if "cancellation factor" in detail else "residual"


def sweep_target(c: DeclaredComputation, store: InMemoryStore, held: Held, ctx: StepContext,
                 bands: Bands, *, xs: Sequence[float], shapes: Sequence[str], trials: int,
                 gen: torch.Generator, scratch: MeasureScratch | None = None,
                 phase: str = "grid") -> list[Trials]:
    """Every trial of one target, then the honest leaf back; raises if the root doesn't
    return. ``ctx`` gives ``k`` and the unit roundoffs, and holds no stats afterwards."""
    s, idx = held.spec, c.product_index(held.spec.m)
    base = store.root
    e = e_m(s.q, ctx.eps_in, ctx.eps_acc)
    out = []
    try:
        for shape in shapes:
            tr = Trials(s.m, held.cls, shape, e, spec_class=held.spec_class, name=s.name,
                        phase=phase)
            for x in xs:
                f = x * bands.tau * e
                for _ in range(trials):
                    p2, f_real = plant(held.p, shape, f, held.p_norm, gen)
                    root = perturb_leaf(c, store, idx, p2)
                    [mp] = _measure_run([s], [(held.a, held.b, p2, held.bias)], h=root, k=ctx.k,
                                        eps_in=ctx.eps_in, eps_acc=ctx.eps_acc,
                                        scratch=scratch, guard=ctx.kappa_guard)
                    rej = _judge_product(c, s, mp, ctx, bands)
                    ctx.stats.products.clear()
                    tr.x_set.append(x)
                    tr.f_real.append(f_real)
                    tr.norm_ratio.append(float(torch.linalg.vector_norm(
                        p2, dtype=torch.float64)) / held.p_norm)
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
    [mp] = _measure_run([held.spec], [(held.a, held.b, held.p, held.bias)], h=store.root,
                        k=ctx.k, eps_in=ctx.eps_in, eps_acc=ctx.eps_acc,
                        guard=ctx.kappa_guard)
    if _judge_product(c, held.spec, mp, ctx, bands) is not None:
        raise AssertionError(f"P_{held.spec.m}: the honest product is rejected")
    ctx.stats.products.clear()
    return tuple(mp.normalized)


# ---- the check-6 trial --------------------------------------------------------------------


def check6_weight(c: DeclaredComputation) -> str:
    """The check-6 trial's linear weight: the one whose gradient is the weight-gradient target
    (:func:`_middle_first` over the weight-gradient products)."""
    g = _middle_first([s for s in c.products if spec_class(c, s) == "weight gradient"])
    names = [n for n, m in c.linear_weights.items() if m == g.m]
    if not names:
        raise ValueError(f"{g.name} (P_{g.m}) is no linear weight's gradient")
    return names[0]


def check6_trial(c: DeclaredComputation, store: InMemoryStore, bands: Bands, ctx: StepContext,
                 name: str, *, gen: torch.Generator,
                 factor: float = CHECK6_FACTOR) -> dict[str, Any]:
    """§5.7's check-6 trial on linear weight ``name`` (module docstring, 5): one entry of
    ``W_{t+1}`` moved to ``ρ ≈ factor·τ_W``, then check 2's read and check 6a on a fresh
    context. Returns the trial's record fields; ``rejected`` is true when 6a failed it on that
    entry. The leaf goes back afterwards and the root must return (else ``AssertionError``)."""
    idx, base = c.w_next_index(name), store.root
    w_t, w_next = store.leaf(c.w_t_index(name)), store.leaf(idx)
    g = store.leaf(c.product_index(c.linear_weights[name])).to(c.weight_dtype)
    tau_w = bands.tau_w_for(name)
    with torch.no_grad():  # check 6a's reference, term by term (_update_identity_reference)
        eta_g = c.eta * g
        u = (w_t - eta_g).reshape(-1)
        scale = (ctx.eps_w * (w_t.abs() + eta_g.abs())).reshape(-1)
        for _ in range(100):
            i = int(torch.randint(u.numel(), (1,), generator=gen))
            sign = 2 * int(torch.randint(2, (1,), generator=gen)) - 1
            if not float(scale[i]) > 0:
                continue
            w2 = w_next.detach().clone()
            w2.view(-1)[i] = (u[i].double() + sign * factor * tau_w * scale[i].double()).to(
                w2.dtype)
            rho = float((w2.reshape(-1)[i] - u[i]).abs().double() / scale[i].double())
            if rho > tau_w:
                break
        else:
            raise RuntimeError(f"{name}: no entry of W_t+1 could be moved outside its band")
    root = perturb_leaf(c, store, idx, w2)
    ctx6 = dataclasses.replace(ctx, state=StepState(), stats=StepStats())
    try:
        rej = check_2_commitment(store, c, ctx6, bands)
        if rej is not None:
            raise AssertionError(f"check 2's read rejected the check-6 trial: {rej}")
        rej = check_6a_linear_update(store, c, ctx6, bands)
    finally:
        if perturb_leaf(c, store, idx, w_next) != base:
            raise AssertionError(f"{name}: the honest W_t+1 leaf did not restore the root")
    rho_max = next((s.rho_max for s in ctx6.stats.tensors if s.weight == name), None)
    return {"weight": name, "entry": i, "sign": sign, "g_m": c.linear_weights[name],
            "tau_w": tau_w, "factor": factor, "rho_target": factor * tau_w, "rho_entry": rho,
            "rho_max": rho_max, "root": root.hex(),
            "check_id": None if rej is None else rej.check_id,
            "kind": None if rej is None else rej.kind,
            "detail": None if rej is None else rej.detail,
            "rejected": (rej is not None and rej.check_id == "6a" and rej.kind == "failed"
                         and rej.detail.startswith(f"{name}: entry {i} "))}


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
    """The band file's ``N``, ``τ`` and ``k`` if it has a sizing, else ``N`` computed from
    ``λ``, ``T``, ``M`` and ``G``. ``c`` is always ``C_ANTI``: a band file written before
    2026-10-05 records the Gaussian 0.798, which is below the uniform worst case."""
    sz = bands.stats.get("sizing") if bands.stats else None
    if sz:
        return Floor(float(sz["N"]), C_ANTI, bands.tau, int(sz["k"]))
    return Floor(bit_budget(LAMBDA, T, c.M, LOG2_G), C_ANTI, bands.tau, k)


def _ratios(tr: Trials) -> np.ndarray:
    """Each trial's ``‖P'‖_F/‖P‖_F``, all 1 for trials recorded without it."""
    if not tr.norm_ratio:
        return np.ones(len(tr.x_set))
    if len(tr.norm_ratio) != len(tr.x_set):
        raise ValueError(f"{len(tr.norm_ratio)} norm ratios for {len(tr.x_set)} trials")
    return np.asarray(tr.norm_ratio, dtype=np.float64)


def summarize(tr: Trials, floor: Floor, *, x_fit: float = FIT_X_MIN
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per point and overall, for one target, shape and phase (module docstring)."""
    tau = floor.tau
    x_set = np.asarray(tr.x_set)
    f_real = np.asarray(tr.f_real)
    ratio = _ratios(tr)
    x_committed = f_real / (tau * tr.e * ratio)  # x against the committed ‖P'‖_F
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
        vectors, missed = n * k, int(misses[sel].sum())
        p1 = missed / vectors
        p1_se = math.sqrt(p1 * (1 - p1) / vectors)
        bound = min(1.0, floor.c / x)
        points.append({"x": x, "f": f, "f_real_mean": float(f_real[sel].mean()),
                       "trials": n, "rejected": int(rejected[sel].sum()),
                       "reject_rate": float(rejected[sel].mean()),
                       "misses": missed, "vectors": vectors, "p1": p1, "p1_se": p1_se,
                       "p1_bound": bound,
                       "p1_over_bound": p1 - C_HAT_SIGMAS * p1_se > bound,
                       "norm_ratio_mean": float(ratio[sel].mean()),
                       "p1_bound_committed": float(np.minimum(
                           1.0, floor.c / x_committed[sel]).mean()),
                       "resolvable": 1 / vectors,
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
    # Poisson standard error of ĉ; the ĉ stop needs ĉ above c by C_HAT_SIGMAS of them.
    c_hat_se = math.sqrt(float(misses[fit].sum())) / denom if denom > 0 else None
    c_over = (c_hat is not None and c_hat_se is not None
              and c_hat - C_HAT_SIGMAS * c_hat_se > floor.c)
    # the same fit with x taken against the committed product's norm, as check 5 normalizes
    fit_c = x_committed >= x_fit
    denom_c = float((k / x_committed[fit_c]).sum()) if fit_c.any() else 0.0
    misses_c = float(misses[fit_c].sum())
    over_x = [p["x"] for p in points if p["p1_over_bound"]]
    f_ach = floor.f_ach(tr.e)
    summary = {"m": tr.m, "name": tr.name, "cls": tr.cls, "spec_class": tr.spec_class,
               "shape": tr.shape, "phase": tr.phase, "e_m": tr.e, "tau": tau,
               "k": k, "trials": len(tr.x_set), "f50": f50, "f_all": f_all,
               "x50": None if f50 is None else f50 / (tau * tr.e),
               "x_all": None if f_all is None else f_all / (tau * tr.e),
               "c_hat": c_hat, "c_hat_se": c_hat_se, "c_hat_over_c": c_over,
               "c_hat_committed": misses_c / denom_c if denom_c > 0 else None,
               "c_hat_committed_se": math.sqrt(misses_c) / denom_c if denom_c > 0 else None,
               "c_fit_x_min": x_fit,
               "c_fit_vectors": int(fit.sum()) * k, "c_fit_misses": int(misses[fit].sum()),
               "p1_over_bound_x": over_x,
               "f_ach": f_ach, "x_ach": f_ach / (tau * tr.e),
               "f_ach_c_hat": None if c_hat is None else floor.f_ach(tr.e, c_hat),
               "N": floor.N, "c_sizing": floor.c,
               "f_all_over_f_ach": None if f_all is None else f_all / f_ach,
               "stop": f_all is None or f_all > f_ach or c_over or bool(over_x)}
    return points, summary


# ---- the run ------------------------------------------------------------------------------


@dataclass
class SweepResult:
    honest: ScenarioResult | None  # the in-process honest base; None on a stored step
    flip: ScenarioResult
    summaries: list[dict[str, Any]]
    points: list[dict[str, Any]]
    trials: list[Trials]
    seconds: dict[str, float]  # per "<target name> <phase>"
    check6: dict[str, Any] | None = None

    @property
    def stops(self) -> list[dict[str, Any]]:
        return [s for s in self.summaries if s["stop"]]

    @property
    def passed(self) -> bool:
        return ((self.honest is None or self.honest.passed) and self.flip.passed
                and bool(self.summaries) and not self.stops
                and self.check6 is not None and self.check6["rejected"])


def _store_of(env: CheatEnv, out: StepOutput, t: int) -> InMemoryStore:
    from verification.commitment.leaves import dataset_tree
    tree = dataset_tree(env.c, env.D)
    return InMemoryStore.from_step(env.c, out, dataset_paths=[
        tree.path(i) for i in schedule(t, env.c.n_s, len(env.D), env.seed)])


def run_sweep(env: CheatEnv, *, xs: Sequence[float] = DEFAULT_X,
              shapes: Sequence[str] = SHAPES, trials: int = DEFAULT_TRIALS,
              deep_xs: Sequence[float] = DEEP_X, deep_shapes: Sequence[str] = DEEP_SHAPES,
              deep_trials: int = DEEP_TRIALS, classes: Sequence[str] | None = None,
              seed: int = 0, t: int = SWEEP_STEP, flip_name: str = FLIP_PRODUCT,
              kept: KeptStep | None = None, writer: MetricsWriter | None = None,
              out: Callable[[str], None] = print) -> SweepResult:
    """The flipped-matmul cheat, then the sweep and the check-6 trial on step ``t``: of
    ``kept``, a stored step, or (``None``) of an honest base run here."""
    c = env.c
    honest, keep = None, KeepStep(t)
    if kept is None:
        honest = run_cheat(env, Scenario("honest", f"honest, step {t}'s transcript kept for "
                                         "the sweep", keep, Expected()), T=t, writer=writer,
                           bpr_records=0, bpr_actual=0.0, out=out)
    elif kept.step != t:
        raise ValueError(f"the stored step is {kept.step}, the sweep's {t}")
    elif (h := stored_band_hash(kept.run_dir)) != env.bands.source:
        raise ValueError(f"{kept.run_dir} was judged by band file {h}, not this run's "
                         f"{env.bands.source}")
    m_flip = c.m_of(flip_name)  # type: ignore[attr-defined]
    flip = run_cheat(env, Scenario("flip", f"largest entry of P_{m_flip} ({flip_name}) "
                                   f"sign-flipped after capture at step {t}",
                                   FlipProduct(t, m_flip), Expected(t, "5", "failed", m_flip)),
                     T=t, writer=writer, cheat_step=t, out=out)
    if honest is not None:
        if keep.out is None or not honest.passed:
            out("the honest base run was not accepted: no sweep")
            return SweepResult(honest, flip, [], [], [], {})
        store = _store_of(env, keep.out, t)
        keep.out = None
        stats = honest.verifier.stats.get(t, StepStats())
    else:
        assert kept is not None
        store, stats = kept.store, kept.honest or StepStats()
    floor = floor_context(env.bands, c, k=env.k, T=t)
    pf = store.leaf(c.product_index(m_flip)).detach().to(torch.float64)
    x_flip = (2 * float(pf.abs().max()) / float(torch.linalg.vector_norm(pf))
              / (env.bands.tau * e_m(c.product(m_flip).q, *_eps(c))))
    out(f"flip: 2·max|P|/‖P‖_F on step {t}'s {flip_name} is x = {x_flip:.4g} band units")
    seen = {p.m: tuple(p.normalized) for p in stats.products}
    ctx = StepContext.for_computation(
        c, step=t, indices=tuple(schedule(t, c.n_s, len(env.D), env.seed)), h_D=env.h_D,
        n_records=len(env.D), prev_w_hashes=(), chain_check_id="7", k=env.k)
    held = hold_targets(c, store, sweep_targets(c, classes))
    source = "in-process" if kept is None else "stored"
    if writer is not None:
        writer.write([{"record": "sweep_source", "run": writer.run, "step": t, "source": source,
                       "root": store.root.hex(),
                       "path": None if kept is None else str(kept.path),
                       "honest_run": None if kept is None else str(kept.run_dir),
                       "targets": [{"m": h.spec.m, "name": h.spec.name, "cls": h.cls,
                                    "spec_class": h.spec_class, "binding": h.binding,
                                    "q": h.spec.q} for h in held]}])
    phases = [("grid", xs, shapes, trials, torch.Generator().manual_seed(seed))]
    if deep_trials > 0 and deep_xs and deep_shapes:
        phases.append(("deep", deep_xs, deep_shapes, deep_trials,
                       torch.Generator().manual_seed(seed + DEEP_SEED)))
    scratch = MeasureScratch()
    summaries, points, all_trials, seconds = [], [], [], {}
    out(f"sweep on {source} step {t} (h {store.root.hex()[:16]}…): {len(held)} targets, "
        f"τ {env.bands.tau:.2f}, k {env.k}, N {floor.N:.2f}; "
        + "; ".join(f"{ph}: shapes {list(sh)}, x {list(px)}, {n} trials per point"
                    for ph, px, sh, n, _ in phases))
    for h in held:
        honest_res = honest_measure(c, store, h, ctx, env.bands)
        if h.spec.m in seen and seen[h.spec.m] != honest_res:
            raise AssertionError(f"P_{h.spec.m}: the sweep's honest residuals {honest_res} "
                                 f"differ from check 5's {seen[h.spec.m]}")
        for phase, p_xs, p_shapes, n, gen in phases:
            t0 = time.perf_counter()
            trs = sweep_target(c, store, h, ctx, env.bands, xs=p_xs, shapes=p_shapes,
                               trials=n, gen=gen, scratch=scratch, phase=phase)
            secs = seconds[f"{h.spec.name} {phase}"] = time.perf_counter() - t0
            for tr in trs:
                pts, sm = summarize(tr, floor)
                sm.update(binding=h.binding, q=h.spec.q, p_norm=h.p_norm,
                          honest_max=max(honest_res), seconds=secs)
                summaries.append(sm)
                tag = {"m": tr.m, "name": tr.name, "cls": tr.cls, "spec_class": tr.spec_class,
                       "binding": h.binding, "shape": tr.shape, "phase": phase}
                points.extend({**tag, **p} for p in pts)
                all_trials.append(tr)
                if writer is not None:
                    writer.write([{"record": "sweep_point", "run": writer.run, "step": t,
                                   **tag, **p} for p in pts])
                    writer.write([{"record": "sweep_summary", "run": writer.run, "step": t,
                                   **sm}])
            out(f"  {h.spec_class:<16} P_{h.spec.m:<5} {h.spec.name:<16} q {h.spec.q:>6} "
                f"{phase:<4} {secs:7.1f} s  "
                + "  ".join(f"{s['shape']}: x_all {_g(s['x_all'])} ĉ {_g(s['c_hat'])}"
                            for s in summaries[-len(trs):]))
    name6 = check6_weight(c)
    t0 = time.perf_counter()
    check6 = check6_trial(c, store, env.bands, ctx, name6,
                          gen=torch.Generator().manual_seed(seed + CHECK6_SEED))
    check6["seconds"] = seconds["check6"] = time.perf_counter() - t0
    if writer is not None:
        writer.write([{"record": "sweep_check6", "run": writer.run, "step": t, **check6}])
        save_trials(writer.dir / TRIALS_FILE, all_trials)
    return SweepResult(honest, flip, summaries, points, all_trials, seconds, check6)


def _eps(c: DeclaredComputation) -> tuple[float, float]:
    from verification.parameters import UNIT_ROUNDOFF
    return UNIT_ROUNDOFF[c.operand_dtype], UNIT_ROUNDOFF[c.accumulator_dtype]


def save_trials(path: Path, trials: Sequence[Trials]) -> None:
    """Every trial, flat: target ``m``, name, classes, shape, phase, ``x``, realized ``f``, the
    ``k`` normalized residuals, κ and the verdict."""
    rows = [(tr, i) for tr in trials for i in range(len(tr.x_set))]
    np.savez(path,
             m=np.asarray([tr.m for tr, _ in rows], dtype=np.int64),
             name=np.asarray([tr.name for tr, _ in rows]),
             cls=np.asarray([tr.cls for tr, _ in rows]),
             spec_class=np.asarray([tr.spec_class for tr, _ in rows]),
             shape=np.asarray([tr.shape for tr, _ in rows]),
             phase=np.asarray([tr.phase for tr, _ in rows]),
             e_m=np.asarray([tr.e for tr, _ in rows]),
             x=np.asarray([tr.x_set[i] for tr, i in rows]),
             f_real=np.asarray([tr.f_real[i] for tr, i in rows]),
             norm_ratio=np.concatenate([_ratios(tr) for tr in trials] or [np.zeros(0)]),
             normalized=np.asarray([tr.normalized[i] for tr, i in rows], dtype=np.float64),
             kappa=np.asarray([tr.kappa[i] for tr, i in rows]),
             reason=np.asarray([tr.reason[i] for tr, i in rows]))


def _g(x: float | None) -> str:
    return "-" if x is None else f"{x:.3g}"


def report_sweep(r: SweepResult, out: Callable[[str], None] = print) -> None:
    """Per target, shape and phase: x50, x_all, ĉ (and ``ĉ(P′)``, fitted against the committed
    norm), both floors; then the check-6 trial."""
    out(f"{'class':<16} {'P_m':>6} {'name':<16} {'q':>6} {'shape':<6} {'phase':<5} "
        f"{'x50':>6} {'x_all':>6} {'f_all':>9} {'ĉ':>7} {'±':>7} {'ĉ(P′)':>7} {'f_ach(ĉ)':>9} "
        f"{'f_ach':>8} {'f_all/f_ach':>11}")
    for s in r.summaries:
        out(f"{s['spec_class']:<16} {s['m']:>6} {s['name']:<16} {s['q']:>6} {s['shape']:<6} "
            f"{s['phase']:<5} {_g(s['x50']):>6} {_g(s['x_all']):>6} {_g(s['f_all']):>9} "
            f"{_g(s['c_hat']):>7} {_g(s['c_hat_se']):>7} {_g(s['c_hat_committed']):>7} "
            f"{_g(s['f_ach_c_hat']):>9} "
            f"{_g(s['f_ach']):>8} {_g(s['f_all_over_f_ach']):>11}"
            + (f"  STOP (p₁ over c/x at x {s['p1_over_bound_x']})" if s["p1_over_bound_x"]
               else "  STOP" if s["stop"] else ""))
    stops = r.stops
    out(f"stop condition (f_all above f_ach, never all rejected, or ĉ or p₁ above c = "
        f"{C_ANTI:.4f} (c/x) by more than {C_HAT_SIGMAS:g} standard errors): "
        + ("none" if not stops else
           ", ".join(f"{s['name']} {s['shape']} {s['phase']}" for s in stops)))
    c6 = r.check6
    if c6 is not None:
        out(f"check-6 trial: {c6['weight']} entry {c6['entry']} moved to ρ {c6['rho_entry']:.3g} "
            f"(τ_W {c6['tau_w']:g}): {'rejected at 6a' if c6['rejected'] else 'NOT rejected'}"
            f" ({c6['check_id']}, {c6['kind']}: {c6['detail']})")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", choices=SOURCES, default="stored",
                   help="step 4 as the honest run stored it, or from an honest run here")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="the honest run directory of the stored step (default "
                        "$VERIF_OUTPUT_DIR/honest-<model>-<attack>-bpr0-s<seed>)")
    p.add_argument("--x", type=float, nargs="+", default=list(DEFAULT_X),
                   help="grid of x = f/(τ·e_m)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per point")
    p.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=SHAPES)
    p.add_argument("--classes", nargs="+", default=None,
                   help="only targets among these §5.7 classes, matmul classes or names")
    p.add_argument("--deep-x", type=float, nargs="*", default=list(DEEP_X),
                   help="the deep points' x")
    p.add_argument("--deep-trials", type=int, default=DEEP_TRIALS,
                   help="trials per deep point; 0 skips them")
    p.add_argument("--deep-shapes", nargs="+", default=list(DEEP_SHAPES), choices=SHAPES)
    p.add_argument("--subdir", default=None,
                   help=f"write metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/SUBDIR/, keeping the "
                        "records of earlier runs")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    if args.trials < 1 or args.deep_trials < 0 or any(not x > 0 for x in args.x + args.deep_x):
        p.error("need --trials ≥ 1, --deep-trials ≥ 0 and every x > 0")
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    setup_determinism(cfg)
    env = load_env(cfg, pc)
    print(f"A13 flipped matmul and sweep: {cfg.model}@{cfg.model_revision[:12]}, k {env.k}, "
          f"step {SWEEP_STEP}, band file {pc.band_file} ({env.bands.source[:16]}…), "
          f"τ {env.bands.tau:.2f}, source {args.source}")
    kept = None
    if args.source == "stored":
        run_dir = args.run_dir or cfg.output_dir / honest_run_name(cfg.model, cfg.attack,
                                                                   cfg.seed)
        t0 = time.perf_counter()
        kept = load_kept_step(env.c, run_dir, step=SWEEP_STEP)
        print(f"stored step {SWEEP_STEP}: {kept.path}, h {kept.root.hex()[:16]}…, read and "
              f"re-hashed in {time.perf_counter() - t0:.1f} s")
    mw = (open_writer(cfg, RUN_NAME, env, {"x": list(args.x), "trials": args.trials,
                                           "shapes": list(args.shapes),
                                           "deep_x": list(args.deep_x),
                                           "deep_trials": args.deep_trials,
                                           "deep_shapes": list(args.deep_shapes),
                                           "classes": args.classes, "sweep_step": SWEEP_STEP,
                                           "source": args.source,
                                           "run_dir": None if kept is None
                                           else str(kept.run_dir)},
                      subdir=args.subdir)
          if use_metrics else None)
    try:
        r = run_sweep(env, xs=args.x, shapes=args.shapes, trials=args.trials,
                      deep_xs=args.deep_x, deep_shapes=args.deep_shapes,
                      deep_trials=args.deep_trials, classes=args.classes, seed=cfg.seed,
                      kept=kept, writer=mw)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")
    runs = [x for x in (r.honest, r.flip) if x is not None]
    report_oracle(runs, env.bands)
    report_sweep(r)
    shared = band_sources(env, runs, cfg.output_dir)
    print(f"band-file hash equal across the file, {len(env.verifiers.made)} verifiers and "
          f"{len(runs)} verdicts: {shared}")
    print("seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in r.seconds.items()))
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
