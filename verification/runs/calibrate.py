"""C1 (task A11): calibrate the bands on the honest run's steps 1–3, write the band file, check k.

    .venv/bin/python -m verification.runs.calibrate [--steps 3] [--metrics | --no-metrics]

P10a: calibration is the verifier's own honest run from public inputs. At test scale the honest
prover's steps 1–3 stand in for it, since prover and verifier share one deterministic process.
The SmolLM2 instance runs as in ``runs/llama_step.py``, with ``k`` from ``VERIF_K``.

1. **Calibration run.** A verifier in calibration mode (P10b) runs the steps: the exact checks
   judge live, and checks 5 and 6 only record their numbers. After the last step,
   ``verifier/calibration.fit`` fits ``τ = 8·s_h``, ``κ_max`` per class and ``τ_W`` per tensor,
   the bands go into band-file bytes, and the verifier freezes on them: it scores the
   calibration steps against the bands, and check 8 closes the run. These steps are
   **in-sample**.
2. **The gates.** Each stops the run with exit code 2, before the band file is written, and
   says why:

   - the concentration guard (largest honest normalized residual ``≤ τ/2``) fails; ``z`` is
     never raised here;
   - P10d's recompute of ``k`` with the measured ``τ`` (``sizing.size_k``, ``T = VERIF_STEPS``,
     ``M`` of ``C``, ``q_max`` the largest contracted dimension) asks for more than ``VERIF_K``.
     Rerun with ``VERIF_K`` raised; ``s_h`` doesn't depend on ``k``.

   An honest rejection, at an exact check or when the bands freeze, exits 1.
3. **The band file.** Written to ``$VERIF_OUTPUT_DIR/bands.json`` (``ProtocolConfig.band_file``),
   which every later verification loads read-only with ``calibration.load_bands``. A copy, and
   the bands of a run the gates stopped, go to the metrics directory.
4. **Judged under the frozen bands.** Two more honest runs of the same steps load the band file
   and judge live: one as the protocol runs, one with check 5's κ guard off (test 1 skipped),
   for the cost split. Both must accept, and all three runs must report the same band-file
   hash (P10a's assertion).
5. **Gradient coherence.** Per-example gradients of step 1's batch at ``W_0``, and
   ``‖Σ_i g_i‖ / (√B·‖g‖)`` with ``‖g‖`` the RMS of the per-example norms (sizing appendix
   §12.2).
6. **Cost.** With metrics on (default ``VERIF_METRICS``), every timed run writes its records and
   residual arrays to ``$VERIF_OUTPUT_DIR/calibrate/k<k>/`` (B6), and the frozen run's memory
   and counting passes follow. A ``calibration`` record holds the fit's summary and the band
   file's hash. The printed split is hashing (check 2), glue (``5.glue``, ``6b.glue``), check 5's
   measuring (``5.measure``), the update identities (6a and the rest of 6b) and the anchors
   (checks 4 and 7, with 0 and 8 once per run), with the κ guard on and off.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from setup.config import load_config, setup_determinism
from verification.computation.instances.llama import LlamaComputation
from verification.computation.interface import load_weights, snapshot_weights
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF, load_protocol_config
from verification.runs.llama_step import load_committed_dataset
from verification.runs.metrics import MetricsWriter, TimeRecorder, read_records
from verification.runs.scenarios import (
    HONEST,
    ReusedModel,
    ScenarioResult,
    count_run,
    honest_final,
    memory_run,
    run_scenario,
)
from verification.verifier.bands import TAU_W0, Bands
from verification.verifier.calibration import (
    Calibration,
    RealizedFloor,
    assert_same_band_source,
    band_file_bytes,
    fit,
    load_bands,
    realized_floor,
    write_band_file,
)
from verification.verifier.context import Rejection, Section
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check.sizing import Sizing, bit_budget, size_k
from verification.verifier.residuals import (
    format_class_table,
    format_tensor_table,
    tensor_summary,
)

__all__ = ["CalibrationRun", "calibrate", "gradient_coherence", "Coherence", "cost_split",
           "main"]

RUN_NAME = "calibrate"
CAL_STEPS = 3  # P10a: steps 1–3
# The step-2 product that rejected under the provisional τ = 8 (Λ, normalized residual 10.1).
WATCH = (2, 2371)


# ---- the calibration run ----------------------------------------------------------------


@dataclass
class CalibrationRun:
    """What the calibration run produced. ``data`` is the band file's bytes, ``bands`` the
    bands they hold (``source`` = their hash)."""

    result: ScenarioResult
    cal: Calibration
    sizing: Sizing
    floor: RealizedFloor
    data: bytes
    bands: Bands
    freeze_rejection: Rejection | None


def _sizing_inputs(c: LlamaComputation, T_run: int) -> dict[str, Any]:
    return {"q_max": max(p.a_shape[1] for p in c.products),
            "eps_in": UNIT_ROUNDOFF[c.operand_dtype], "eps_acc": UNIT_ROUNDOFF[c.accumulator_dtype],
            "T": T_run, "M": c.M}


def calibrate(c: LlamaComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor], *, T: int,
              k: int, T_run: int, h_D: bytes, final: Mapping[str, torch.Tensor],
              context: Mapping[str, Any],
              recorder: TimeRecorder | None = None,
              build_model: Callable[[], torch.nn.Module] | None = None) -> CalibrationRun:
    """Steps ``1..T`` in calibration mode, then fit, size, and freeze (module docstring, 1).

    ``T_run`` is the length of the run the bands will judge (``VERIF_STEPS``), which enters
    ``k``'s bit budget. ``context`` goes into the band file's statistics.
    """
    out: dict[str, Any] = {}

    def make(section: Section | None) -> Verifier:
        v = Verifier(c, h_D=h_D, n_records=len(D), k=k, n_steps=T, bands=None, w0=w0,
                     calibrate=True, section=section)
        out["v"] = v
        return v

    def on_step(rec: Any) -> None:
        if rec.rejection is not None:
            raise RuntimeError(f"honest calibration rejected at an exact check: {rec.rejection}")
        if rec.t < T:
            return
        v: Verifier = out["v"]
        cal = fit(v.stats, k=k)
        inp = _sizing_inputs(c, T_run)
        sz = size_k(**inp, s_h=cal.s_h, z=cal.z)
        N = bit_budget(LAMBDA, T_run, c.M, LOG2_G)
        floor = realized_floor(v.stats, tau=cal.tau, k=k, N=N, eps_in=inp["eps_in"],
                               eps_acc=inp["eps_acc"])
        stats = {"sizing": dataclasses.asdict(sz), "k_required": sz.k,
                 "realized_floor": _floor_dict(floor), "context": dict(context)}
        data = band_file_bytes(cal, stats)
        bands = Bands.from_json(data)
        out.update(cal=cal, sizing=sz, floor=floor, data=data, bands=bands,
                   freeze=v.freeze(bands))

    r = run_scenario(c, D, w0, HONEST, T=T, k=k, final=final, on_step=on_step,
                     recorder=recorder, h_D=h_D, build_model=build_model, verifier=make)
    return CalibrationRun(r, out["cal"], out["sizing"], out["floor"], out["data"],
                          out["bands"], out["freeze"])


def _floor_dict(f: RealizedFloor) -> dict[str, Any]:
    d = dataclasses.asdict(f)
    d["zero_row_sum"] = list(f.zero_row_sum)
    d["by_q"] = {str(q): r for q, r in f.by_q.items()}
    return d


def judging_verifier(c: LlamaComputation, n_records: int, w0: Mapping[str, torch.Tensor],
                     bands: Bands, *, T: int, k: int, h_D: bytes, kappa_guard: bool = True
                     ) -> Callable[[Section | None], Verifier]:
    """A factory for ``run_scenario``: a verifier that judges live with the band file's bands."""
    def make(section: Section | None) -> Verifier:
        return Verifier(c, h_D=h_D, n_records=n_records, k=k, n_steps=T, bands=bands, w0=w0,
                        section=section, kappa_guard=kappa_guard)
    return make


def judged_run(c: LlamaComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
               bands: Bands, *, T: int, k: int, h_D: bytes, final: Mapping[str, torch.Tensor],
               kappa_guard: bool = True, recorder: Any = None,
               build_model: Callable[[], torch.nn.Module] | None = None) -> ScenarioResult:
    """The same honest steps judged live by the band file's bands."""
    make = judging_verifier(c, len(D), w0, bands, T=T, k=k, h_D=h_D, kappa_guard=kappa_guard)
    return run_scenario(c, D, w0, HONEST, T=T, k=k, final=final, recorder=recorder, h_D=h_D,
                        build_model=build_model, verifier=make)


# ---- gradient coherence (sizing appendix §12.2) -----------------------------------------


@dataclass(frozen=True)
class Coherence:
    """Per-example gradients of one batch. ``ratio = ‖Σ g_i‖ / (√B·rms_i ‖g_i‖)``: about 1
    when the ``g_i`` are mutually orthogonal, ``√B`` when they all point the same way."""

    B: int
    norms: tuple[float, ...]
    sum_norm: float
    ratio: float
    cosines: tuple[float, ...]  # every pair i < j
    batch_check: float  # ‖Σ g_i − ∇ℒ‖ / ‖∇ℒ‖: the decomposition is exact up to rounding


def gradient_coherence(c: LlamaComputation, model: torch.nn.Module,
                       w: Mapping[str, torch.Tensor], records: Sequence[Any]) -> Coherence:
    """Split the batch loss ``ℒ = Σ_i ℒ_i`` by record and take each ``g_i = ∇ℒ_i``.

    ``ℒ_i = (1/Σμ)·Σ_{tokens of i} μ·CE``, so ``Σ_i g_i`` is the step's own gradient and the
    common factor ``1/Σμ`` cancels in the ratio. One forward pass, one backward pass per record
    (``retain_graph``), and the Gram matrix in fp64, parameter by parameter. ``model`` is left
    as it came apart from its weights: no ``.grad`` is set.
    """
    load_weights(c, model, w)
    model.zero_grad(set_to_none=True)
    params = [p for _, p in model.named_parameters()]
    batch = c.assemble(records)
    logits = c.logits(model, batch)
    mu = batch.mask.to(logits.dtype)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.targets.reshape(-1),
                         reduction="none").reshape(mu.shape)
    per = (ce * mu).sum(dim=1) / mu.sum()
    B = per.shape[0]
    grads = [torch.autograd.grad(per[i], params, retain_graph=True) for i in range(B)]
    full = torch.autograd.grad(c.loss_from_logits(logits, batch), params)
    del logits, ce, per
    gram = [[0.0] * B for _ in range(B)]
    err = ref = 0.0
    for j, p in enumerate(params):
        g = [grads[i][j].detach().double().reshape(-1) for i in range(B)]
        for a in range(B):
            for b in range(a, B):
                gram[a][b] += float(g[a] @ g[b])
        f = full[j].detach().double().reshape(-1)
        err += float(((sum(g) - f) ** 2).sum())
        ref += float((f * f).sum())
    for a in range(B):
        for b in range(a):
            gram[a][b] = gram[b][a]
    norms = tuple(math.sqrt(gram[i][i]) for i in range(B))
    sum_norm = math.sqrt(sum(gram[a][b] for a in range(B) for b in range(B)))
    rms = math.sqrt(sum(n * n for n in norms) / B)
    cos = tuple(gram[a][b] / (norms[a] * norms[b]) for a in range(B) for b in range(a + 1, B))
    return Coherence(B, norms, sum_norm, sum_norm / (math.sqrt(B) * rms), cos,
                     math.sqrt(err / ref))


# ---- the cost split ---------------------------------------------------------------------

SPLIT = {
    "hashing (check 2)": ("2",),
    "glue (5.glue + 6b.glue)": ("5.glue", "6b.glue"),
    "check 5 measuring (5.measure)": ("5.measure",),
    "update identity (6a + 6b w/o glue)": ("6a", "6b", "-6b.glue"),
    "anchors (4, 7; 0 and 8 per run)": ("4", "7"),
}


def cost_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean verifier seconds per step by part, from a timed run's ``time`` rows.

    Checks 0 and 8 run once per run (step 0); their time is spread over the steps."""
    steps = sorted({r["step"] for r in rows if r.get("record") == "time" and r["step"] > 0})
    if not steps:
        return {}

    def total(comp: str, step: int | None) -> float:
        return sum(r["time_s"] for r in rows if r.get("record") == "time"
                   and r["component"] == comp and (step is None or r["step"] == step))

    out: dict[str, float] = {}
    for name, comps in SPLIT.items():
        s = 0.0
        for comp in comps:
            sign = -1.0 if comp.startswith("-") else 1.0
            s += sign * statistics.mean(total(comp.lstrip("-"), t) for t in steps)
        out[name] = s
    out["anchors (4, 7; 0 and 8 per run)"] += (total("run:0", 0) + total("run:8", 0)) / len(steps)
    out["check 5 total"] = statistics.mean(total("5", t) for t in steps)
    out["verifier total"] = statistics.mean(total("verifier", t) for t in steps)
    return out


# ---- reports ----------------------------------------------------------------------------


def report_calibration(run: CalibrationRun, k: int, out: Callable[[str], None] = print) -> None:
    cal, sz, f = run.cal, run.sizing, run.floor
    out(f"== calibration on steps {list(cal.steps)} (in-sample), k = {k}")
    out("check 5: normalized residual by matmul class")
    format_class_table(list(cal.classes), out)
    out(f"s_h = {cal.s_h:.4f} (class {cal.s_h_class}); τ = z·s_h = {cal.z:g}·{cal.s_h:.4f} = "
        f"{cal.tau:.3f}")
    out(f"global max {cal.global_max:.3f} at {cal.global_max_name} (step "
        f"{cal.global_max_step}); max / s_h = {cal.ratio:.3f}")
    for name in ("Lambda", "dF"):
        row = next((r for r in cal.classes if r.cls == name), None)
        if row is not None:
            out(f"named class {name}: RMS {row.rms:.3f}, max {row.max:.3f}, κ median "
                f"{row.kappa_median:.4g}, κ max {row.kappa_max:.4g}")
    t, m = WATCH
    st = run.result.verifier.stats.get(t)
    p = None if st is None else next((x for x in st.products if x.m == m), None)
    if p is not None:
        out(f"watch: step {t} P_{m} ({p.name}, {p.cls}): max normalized residual "
            f"{max(p.normalized):.3f} against τ = {cal.tau:.3f}")
    out(f"concentration guard: max {cal.global_max:.3f} ≤ τ/2 = {cal.guard_limit:.3f}: "
        f"{'OK' if cal.guard_ok else 'FAILED'}")
    out("κ_max per class (2 × the largest honest κ):")
    for r in cal.classes:
        out(f"  {r.cls:<10} honest median {r.kappa_median:9.4g}  max {r.kappa_max:9.4g}  "
            f"→ κ_max {cal.kappa_max[r.cls]:.4g}")
    out("check 6: ρ_max by weight role")
    format_tensor_table(tensor_summary(run.result.verifier.stats), out)
    above = {w: x for w, x in cal.tau_w.items() if x > TAU_W0}
    out(f"τ_W: {len(cal.tau_w)} tensors, min {min(cal.tau_w.values()):.3f}, max "
        f"{max(cal.tau_w.values()):.3f}; {len(above)} above the floor {TAU_W0:g}"
        + "".join(f"\n  {w}: τ_W {x:.3f} (ρ_max {cal.rho_max[w]:.3f})"
                  for w, x in sorted(above.items(), key=lambda kv: -kv[1])))
    out(f"k recompute (P10d): q_max {sz.q_max}, e_max {sz.e_max:.4e}, τ {sz.tau:.3f}, "
        f"b0 {sz.b0:.3f} bits, N {sz.N:.2f} bits → k = ⌈N/b0⌉ = {sz.k} (configured {k}); "
        f"f_achieved at q_max with k = {sz.k}: {sz.f_achieved:.4f}")
    out(f"realized floor Φ_m/‖P_m‖_F at k = {k}: min {f.ratio_min:.4f}, median "
        f"{f.ratio_median:.4f}, max {f.ratio_max:.4f} ({f.ratio_max_name}); target 1, "
        f"{f.over_target} of {f.count} above; poisoned-batch margin 1/√2: "
        f"{f.over_poison_margin} above")
    out(f"  Φ_m itself: min {f.phi_min:.4g}, median {f.phi_median:.4g}, max {f.phi_max:.4g}")
    out("  by contracted dimension q: " + ", ".join(f"q={q}: {r:.4f}" for q, r in f.by_q.items()))
    out(f"  products with ‖|P|·1‖ = 0 and ν ≠ 0: {len(f.zero_row_sum)}"
        + ("" if not f.zero_row_sum else f" {list(f.zero_row_sum)[:5]}"))
    v = run.result.loop.verdict
    out(f"frozen bands {run.bands.source[:16]}…: steps 1–{len(cal.steps)} "
        + ("accepted" if run.freeze_rejection is None else f"REJECTED: {run.freeze_rejection}")
        + f"; run verdict accepted={v.accepted} ({v.steps_verified} steps)")


def report_coherence(g: Coherence, out: Callable[[str], None] = print) -> None:
    out(f"gradient coherence, step 1 batch at W_0 (B = {g.B}): ‖Σg_i‖/(√B·‖g‖) = {g.ratio:.4f} "
        f"(1 = incoherent, √B = {math.sqrt(g.B):.2f} = aligned)")
    out("  per-example norms " + ", ".join(f"{x:.4g}" for x in g.norms)
        + f"; ‖Σg_i‖ {g.sum_norm:.4g}; pairwise cosines "
        + ", ".join(f"{x:+.3f}" for x in g.cosines)
        + f"; Σg_i vs ∇ℒ relative error {g.batch_check:.1e}")


def report_split(on: Mapping[str, float], off: Mapping[str, float],
                 out: Callable[[str], None] = print) -> None:
    out("verifier wall clock per step, mean over the judged steps (s): κ guard on | off | on − off")
    for name in on:
        a, b = on[name], off.get(name, math.nan)
        out(f"  {name:<38} {a:8.3f} | {b:8.3f} | {a - b:+8.3f}")
    m_on, m_off = on.get("check 5 measuring (5.measure)"), off.get("check 5 measuring (5.measure)")
    if m_on and m_off:
        out(f"  the guard adds {100 * (m_on / m_off - 1):.1f}% to check 5's measuring and "
            f"{100 * (on['verifier total'] / off['verifier total'] - 1):.1f}% to the verifier")


# ---- main -------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=CAL_STEPS,
                   help=f"calibration steps, from step 1 (default {CAL_STEPS}, P10a)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/k<k>/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error(f"{args.steps} calibration steps: at least one")
    cfg = load_config()
    pc = load_protocol_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    k, T = pc.k, args.steps
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    final = honest_final(c, D, w0, T, build_model=models)
    print(f"C1 calibration: {cfg.model}@{cfg.model_revision[:12]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, k {k}, calibration steps 1–{T}, judged run T {cfg.steps}, |D| {len(D)}, "
          f"h_D {h_D.hex()[:16]}…, M {c.M}")
    context = {"model": cfg.model, "model_revision": cfg.model_revision, "h_D": h_D.hex(),
               "n_s": c.n_s, "n": c.n, "eta": c.eta, "M": c.M, "T_run": cfg.steps,
               "device": cfg.device}
    run_dir = cfg.output_dir / RUN_NAME / f"k{k}"
    llama = {"n_s": c.n_s, "n": c.n, "eta": c.eta, "k": k, "T": T, "n_records": len(D),
             "M": c.M, "n_leaves": c.n_leaves}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics")}, "llama": llama}
    base = dict(T=T, k=k, h_D=h_D, final=final, build_model=models)
    mw = (MetricsWriter(run_dir, RUN_NAME, device=cfg.device, model=cfg.model,
                        corpus=cfg.dataset, seed=cfg.seed, config=config, band_file_hash=None,
                        h_D=h_D, extra={"band_source": "calibration", **llama})
          if use_metrics else None)
    try:
        return _run(c, D, w0, base, mw, k=k, T_run=cfg.steps, context=context,
                    band_file=pc.band_file, run_dir=run_dir, models=models)
    finally:
        if mw is not None:
            mw.close()
            print(f"metrics: {mw.dir}")


def _run(c: LlamaComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
         base: dict[str, Any], mw: MetricsWriter | None, *, k: int, T_run: int,
         context: Mapping[str, Any], band_file: Path, run_dir: Path,
         models: ReusedModel) -> int:
    rec = None if mw is None else mw.recorder("calibrate")
    run = calibrate(c, D, w0, T_run=T_run, context=context, recorder=rec, **base)
    report_calibration(run, k)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bands.json").write_bytes(run.data)  # the record of this run, whatever the gates
    summary = {"record": "calibration", "run": RUN_NAME, "band_file_hash": run.bands.source,
               "k": k, "k_required": run.sizing.k, "s_h": run.cal.s_h,
               "s_h_class": run.cal.s_h_class, "tau": run.cal.tau,
               "global_max": run.cal.global_max, "guard_ok": run.cal.guard_ok,
               "realized_floor_max": run.floor.ratio_max,
               "freeze_rejection": None if run.freeze_rejection is None
               else str(run.freeze_rejection)}
    if run.freeze_rejection is not None or not run.result.loop.verdict.accepted:
        print("STOP: an honest calibration step was rejected under its own frozen bands")
        _write(mw, summary)
        return 1
    if not run.cal.guard_ok:
        print(f"STOP: the concentration guard failed (max {run.cal.global_max:.3f} > τ/2 = "
              f"{run.cal.guard_limit:.3f}). z is not raised here; no band file written.")
        _write(mw, summary)
        return 2
    if run.sizing.k > k:
        print(f"STOP (P10d): the measured τ needs k = {run.sizing.k} > VERIF_K = {k}. Rerun with "
              f"VERIF_K={run.sizing.k}; no band file written to {band_file}.")
        _write(mw, summary)
        return 2
    write_band_file(band_file, run.data)
    bands = load_bands(band_file, k=k)
    print(f"band file {band_file}: hash {bands.source}")
    summary["band_file"] = str(band_file)
    _write(mw, summary)

    on = judged_run(c, D, w0, bands, recorder=None if mw is None else mw.recorder("frozen"),
                    **base)
    off = judged_run(c, D, w0, bands, kappa_guard=False,
                     recorder=None if mw is None else mw.recorder("frozen_no_kappa"), **base)
    for name, r in (("frozen", on), ("frozen, κ guard off", off)):
        v = r.loop.verdict
        print(f"judged under the band file ({name}): accepted={v.accepted}, "
              f"{v.steps_verified} steps" + ("" if v.rejection is None else f", {v.rejection}"))
    shared = assert_same_band_source([run.result.loop.verdict.band_source,
                                      on.loop.verdict.band_source, off.loop.verdict.band_source])
    print(f"band-file hash equal across the three runs (P10a): {shared[:16]}…")
    if not (on.passed and off.passed):
        return 1

    coh = gradient_coherence(c, models(), w0, [D[i] for i in on.verifier._schedule(1)])
    report_coherence(coh)
    _write(mw, {"record": "gradient_coherence", "run": RUN_NAME, "step": 1,
                **dataclasses.asdict(coh)})

    if mw is not None:
        make = judging_verifier(c, len(D), w0, bands, T=base["T"], k=k, h_D=base["h_D"])
        mw.write(memory_run(c, D, w0, run=RUN_NAME, scenario=_named("frozen"),
                            device=mw.device, verifier=make, **base))
        mw.write(count_run(c, D, w0, run=RUN_NAME, scenario=_named("frozen"), verifier=make,
                           **base))
        rows = read_records(mw.dir / "records.jsonl", "time")
        report_split(cost_split([r for r in rows if r["scenario"] == "frozen"]),
                     cost_split([r for r in rows if r["scenario"] == "frozen_no_kappa"]))
    return 0


def _named(name: str) -> Any:
    return dataclasses.replace(HONEST, name=name)


def _write(mw: MetricsWriter | None, row: Mapping[str, Any]) -> None:
    if mw is not None:
        mw.write([row])


if __name__ == "__main__":
    sys.exit(main())
