"""FS4: one model's bring-up on CPU at test scale's settings.

    VERIF_MODEL=<repo> VERIF_MODEL_REVISION=<commit> VERIF_OUTPUT_DIR=<dir> \\
        .venv/bin/python -m verification.runs.bring_up [--steps T] [--reference-bands PATH]
        [--tamper | --no-tamper] [--disk-transcripts DIR]

Before any GPU run, each full-scale model goes through the code on CPU with test scale's other
settings: ``(VERIF_MASTER_DTYPE, VERIF_COMPUTE_DTYPE)``, ``VERIF_BATCH × VERIF_SEQ_LEN``, η,
``VERIF_K``, ``VERIF_SEED``. The model is ``VERIF_MODEL`` at ``VERIF_MODEL_REVISION`` alone;
nothing names a model in code. ``D`` and ``h_D`` come from ``$VERIF_OUTPUT_DIR/data/``, which
``materialize_data`` writes for the same model. The run never downloads (``setup.model``
loads from the local cache only): it stops if the weights aren't in the local Hugging Face
cache.

1. **The pin.** Repo, commit (a branch name is refused), the SHA-256 of the cached
   ``config.json``, and each weight file with its SHA-256, computed over its bytes and checked
   against the name of its cache blob (``setup.model.model_pin``, timed), and the pad: the
   model's pinned EOS (F8f).
2. **Honest steps.** ``T`` steps (``--steps``, default 1) from ``W_0``, the unmodified
   ``from_pretrained`` weights, through the S3 loop (``run_scenario``). The verifier runs in
   calibration mode (P10b): the exact checks judge live, and checks 5 and 6 only record. After
   the last step the bands are fitted on these steps (``calibration.fit``) and the verifier
   freezes on them, so check 8 closes the run. The fit is in-sample and never written as a band
   file: it gives item 4 its bands and nothing else. Its concentration guard (the largest
   honest normalized residual at most ``τ/2``) is printed beside it, and a failure is flagged in
   the verdict and fails the run. Beside them: the band unit (F18's split, ``σ_r·e_m·‖P‖_F``
   with ``e_m = √2·ε_in + √q·ε_acc·max(1, U/‖P‖_F)``), the largest ``U/‖P‖_F`` against its
   bound ``√w·κ``, and P10d's ``k`` at the largest ``e_m`` over these products and at
   ``e_m(q_max)`` (``T = VERIF_STEPS``), information only. Check 8's reference is ``T``
   uncaptured steps from ``W_0``, kept as leaf hashes only. A step rejected live ends the run
   with that rejection, unfitted.
3. **Residuals, printed and not judged.** Check 5's table by product class and check 6's by
   weight role. With ``--reference-bands`` (SmolLM2's band file), each score family's largest
   score at each step under those bands (EVALUATION_SPEC §2.2: check 5's ``ρ_5``, its κ guard's
   ``ρ_κ``, check 6's ``ρ_6``), and the families above 1. A class the file doesn't name takes
   its default ``κ_max``, and every weight its default ``τ_W``, since its per-tensor values name
   SmolLM2's tensors (:func:`reference_bands`).
4. **One bias entry altered** (a model with q, k, v biases; ``--no-tamper`` skips it). As
   FS2's switch test, on the real model, judged by item 2's fitted bands: step 1 trained with
   the last entry of layer 1's ``b_q`` raised by 1 under an honest ``W_t`` leaf must fail
   check 5 at ``L1.Y_q``, and an honest step with that entry raised in its ``W_{t+1}`` leaf
   must fail check 6b. It runs on bands that fail the concentration guard too, and its result
   is then labeled so.
5. **Cost.** The honest run is the B6 memory pass (``MemoryRecorder``): each step's prover and
   verifier peak, beside the loop's wall clocks (``prove_step`` and the commit; every check;
   the hand-off between them, with ``run_loop``'s cache release on CUDA unless
   ``VERIF_RELEASE_CACHE=0``), which the probe's sampling slows a little. While the pass runs, each top-level section prints its own peak as
   it closes, with the peaks of the sections nested in it, so a step that dies (out of GPU
   memory, say) still shows each section it got through and the one that raised. With
   ``--disk-transcripts DIR`` every step, the
   tampers' included, reaches the verifier through C3's ``DiskHandoff`` instead of in memory
   (an option of this run only; the protocol code is the same); the write lands in the commit
   time.

Exits 0 when the honest run is accepted, its fit holds the concentration guard and each
tamper is rejected where expected, else 1.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from setup.config import load_config, setup_determinism
from setup.data import schedule
from setup.model import model_pin
from verification.commitment.leaves import dataset_tree, leaf_hash, leaf_hashes_of
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF, load_protocol_config
from verification.prover.step import prove_step
from verification.runs.llama_step import load_committed_dataset, report_costs
from verification.runs.loop import release_cache_setting, release_device_cache
from verification.runs.metrics import (
    SCORE_FAMILIES,
    memory_pass,
    memory_probe,
    nan_as_inf,
    rho_arrays,
    step_scores,
)
from verification.runs.scenarios import (
    HONEST,
    ReusedModel,
    ScenarioResult,
    honest_final,
    run_scenario,
)
from verification.transcript.store import IN_MEMORY, DiskHandoff, InMemoryStore, StoreHandoff
from verification.verifier.bands import Bands, product_class
from verification.verifier.calibration import Calibration, band_file_bytes, fit, largest_e_m
from verification.verifier.context import Rejection, Section
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check.sizing import size_k, term_ratio
from verification.verifier.residuals import (
    class_summary,
    format_class_table,
    format_tensor_table,
    tensor_summary,
)

__all__ = ["BringUp", "Tamper", "bring_up", "bias_tamper", "reference_bands", "family_maxima",
           "main"]

RUN_NAME = "bring_up"

# The freeze's bands after a live rejection, before any fit: every bound is the largest finite
# float, so re-judging the rejected step fails only a NaN or infinite number, which fails any
# band and would have stopped a judged run before the root comparison. Otherwise the reported
# rejection is the live one.
_UNFITTED = Bands(tau=sys.float_info.max, kappa_max=sys.float_info.max,
                  tau_w=sys.float_info.max, source="bring-up: rejected live, before the fit")


@dataclass
class Tamper:
    """One altered-bias step: what was altered, the check expected to reject it, text its
    rejection must name (the product or the weight), and the result."""

    name: str
    expected: str
    names: str
    rejection: Rejection | None

    @property
    def rejected_as_expected(self) -> bool:
        rej = self.rejection
        return rej is not None and (rej.step, rej.check_id) == (1, self.expected) \
            and self.names in rej.detail


@dataclass
class BringUp:
    """What :func:`bring_up` measured. ``bands`` are the in-sample fit of the honest steps.
    ``freeze_rejection`` is what the freeze returned: on the fit, or after a live rejection on
    bands that keep it (``None`` when no freeze ran, as after check 1's rejection)."""

    result: ScenarioResult
    memory_rows: list[dict[str, Any]]
    cal: Calibration | None  # None when the run was rejected live, before the fit
    bands: Bands | None
    freeze_rejection: Rejection | None
    tampers: list[Tamper] = field(default_factory=list)

    @property
    def honest_accepted(self) -> bool:
        return self.result.loop.verdict.accepted

    @property
    def guard_ok(self) -> bool | None:
        """The fit's concentration guard (``Calibration.guard_ok``), ``None`` without a fit."""
        return None if self.cal is None else self.cal.guard_ok

    @property
    def passed(self) -> bool:
        """The honest run is accepted on a fit that holds the concentration guard, and each
        tamper is rejected where expected."""
        return (self.honest_accepted and self.guard_ok is True
                and all(t.rejected_as_expected for t in self.tampers))


def _bump(t: torch.Tensor, index: int = -1) -> torch.Tensor:
    """``t`` with one entry raised by 1, by default the last (FS2's switch test)."""
    t = t.detach().clone()
    t.view(-1)[index] += 1.0
    return t


def bias_tamper(c: DecoderComputation, D: Sequence[Any], w0: Mapping[str, torch.Tensor],
                bands: Bands, *, k: int, seed: int, h_D: bytes,
                build_model: Callable[[], torch.nn.Module],
                layer: int = 1, x: str = "q", handoff: StoreHandoff | None = None,
                release_cache: bool | None = None) -> list[Tamper]:
    """Step 1 with the last entry of ``b_x`` in layer ``layer`` raised by 1 (module docstring,
    4): in training only, under an honest ``W_t`` leaf (check 5 at ``L{layer}.Y_x``), and in
    the ``W_{t+1}`` leaf of an honest step (check 6b). Each is a one-step run judged by
    ``bands``, built as FS2's switch test builds it: the leaf is replaced and the tree
    re-rooted before the step goes to ``handoff`` (default in memory), as the loop's does, and
    with the loop's cache release unless ``release_cache`` is off (``None`` reads
    ``VERIF_RELEASE_CACHE``)."""
    release_cache = release_cache_setting(release_cache)
    b = c.bias(layer, x)
    if b is None:
        raise ValueError(f"layer {layer}'s {x} projection has no bias")
    hand = handoff or IN_MEMORY
    idx = schedule(1, c.n_s, len(D), seed)
    tree = dataset_tree(c, D)
    records, paths = [D[i] for i in idx], [tree.path(i) for i in idx]

    def verify(w: Mapping[str, torch.Tensor],
               edit: Callable[[list[Any]], tuple[int, Any]]) -> Rejection | None:
        v = Verifier(c, h_D=h_D, n_records=len(D), k=k, n_steps=1, bands=bands, seed=seed,
                     w0=w0)
        rej = v.start_run(D)
        if rej is not None:
            return rej
        out = prove_step(c, build_model(), w, records)
        leaves, tree_h = InMemoryStore.commit_step(c, out)
        del out
        i, obj = edit(leaves)
        tree_h.update_leaf(i, leaf_hash(c, i, obj))
        leaves[i] = obj
        store = hand.hold(c, 1, leaves, tree_h, paths)
        del leaves, tree_h
        if release_cache:
            release_device_cache(c.device)  # as the loop's hand-off, on CUDA only
        try:
            return v.verify_step(1, store)
        finally:
            hand.release(store)

    def honest_w_t(leaves: list[Any]) -> tuple[int, Any]:
        return c.w_t_index(b), w0[b].detach().clone()

    def bump_w_next(leaves: list[Any]) -> tuple[int, Any]:
        i = c.w_next_index(b)
        return i, _bump(leaves[i])

    product = f"L{layer}.Y_{x}"
    return [
        Tamper(f"{b}[-1] + 1 in training, W_t leaf honest", "5",
               f"P_{c.m_of(product)} ({product})",
               verify({**w0, b: _bump(w0[b])}, honest_w_t)),
        Tamper(f"{b}[-1] + 1 in the W_t+1 leaf", "6b", b, verify(w0, bump_w_next)),
    ]


def bring_up(c: DecoderComputation, D: Sequence[Any], h_D: bytes, *, T: int, k: int,
             seed: int, build_model: Callable[[], torch.nn.Module] | None = None,
             tamper: bool = True, device: str | torch.device = "cpu",
             handoff: StoreHandoff | None = None,
             echo: Callable[[str], None] | None = None,
             release_cache: bool | None = None) -> BringUp:
    """Items 2, 4 and 5 of the module docstring on ``c`` (any :class:`DecoderComputation`, so
    a random-weight config runs the same path). ``build_model`` (default ``c.build_model``)
    gives the prover's model, built once and reused (``ReusedModel``). ``tamper`` runs item 4
    when ``c`` has q, k, v biases. ``handoff`` gives every step, the tampers' included, to the
    verifier (default in memory; ``--disk-transcripts`` passes C3's ``DiskHandoff``). ``echo``
    prints each section's peak as the memory pass closes it (``MemoryRecorder``), so a run that
    dies mid-step still shows where its memory went. ``release_cache=False`` skips the CUDA
    cache release between prover and verifier, in the honest steps and the tampers alike;
    ``None`` (the default) reads ``VERIF_RELEASE_CACHE``."""
    models = ReusedModel(build_model or c.build_model)
    w0 = snapshot_weights(c, models())
    # Check 8's reference as leaf hashes, in C's weight order, so the run holds no extra copy
    # of the weights.
    w_T = honest_final(c, D, w0, T, seed=seed, build_model=models)
    final = leaf_hashes_of(c, ((c.w_t_index(n), w_T[n]) for n in c.weight_names))
    del w_T
    got: dict[str, Any] = {"cal": None, "bands": None, "freeze": None}

    def make(section: Section | None) -> Verifier:
        v = Verifier(c, h_D=h_D, n_records=len(D), k=k, n_steps=T, bands=None, seed=seed,
                     w0=w0, calibrate=True, section=section)
        got["v"] = v
        return v

    def on_step(rec: Any) -> None:
        # The loop calls end_run after its last step, and end_run needs a frozen verifier.
        v: Verifier = got["v"]
        if rec.rejection is not None:
            # No fit: freeze on bands that keep the live rejection (_UNFITTED). Finite bands
            # of ordinary size could replace a root-comparison rejection with a band failure
            # of that step's numbers (Verifier.freeze), which no fitted band ever judged.
            got["freeze"] = v.freeze(_UNFITTED)
        elif rec.t == T:
            cal = fit(v.stats, k=k)
            bands = Bands.from_json(band_file_bytes(cal, {"context": {"bring_up": RUN_NAME}}))
            got.update(cal=cal, bands=bands, freeze=v.freeze(bands))

    def run(rec: Any) -> None:
        got["r"] = run_scenario(c, D, w0, HONEST, T=T, k=k, seed=seed, final=final,
                                on_step=on_step, recorder=rec, h_D=h_D, build_model=models,
                                verifier=make, handoff=handoff, release_cache=release_cache)

    probe = memory_probe(device)
    if echo is not None:
        echo(f"== memory pass: each section's peak as it closes, GB [{probe.source}]")
    rows = memory_pass(RUN_NAME, HONEST.name, run, probe, echo=echo)
    out = BringUp(got["r"], rows, got["cal"], got["bands"], got["freeze"])
    out.result.loop.w_final = {}  # the prover's W_T: check 8 has read it, and it is |W| bytes
    if tamper and c.qkv_bias and out.bands is not None and out.honest_accepted:
        out.tampers = bias_tamper(c, D, w0, out.bands, k=k, seed=seed, h_D=h_D,
                                  build_model=models, handoff=handoff,
                                  release_cache=release_cache)
    return out


# ---- the scores against a reference band file --------------------------------------------


def reference_bands(ref: Bands, c: DecoderComputation) -> tuple[Bands, list[str]]:
    """``ref``'s bands carried over to ``c``: ``τ``, the default ``κ_max`` and the per-class
    ``κ_max`` of every class ``c`` has, and the default ``τ_W`` for every weight (``ref``'s
    per-tensor ``τ_W`` name its own model's tensors). Also returns ``c``'s classes that ``ref``
    doesn't name, which take its default ``κ_max``. Only scored, never judged with."""
    classes = list(dict.fromkeys(product_class(c, p) for p in c.products))
    bands = Bands(tau=ref.tau, kappa_max=ref.kappa_max, tau_w=ref.tau_w,
                  kappa_classes={k: v for k, v in ref.kappa_classes.items() if k in classes},
                  source=ref.source)
    return bands, [x for x in classes if x not in ref.kappa_classes]


def family_maxima(stats: Any, bands: Bands, *, k: int) -> dict[str, tuple[float, int, str]]:
    """Per score family: ``(largest score, number above 1, where the largest sits)``."""
    s = step_scores(stats, bands, k=k)
    rho = rho_arrays(stats, bands, k=k)
    k5 = rho["rho_5"].shape[1]  # check 5's scores: one row of k challenges per product
    where = {
        "rho_5": lambda i: f"{stats.products[i // k5].name}, challenge j={i % k5 + 1}",
        "rho_kappa": lambda i: stats.products[i].name,
        "rho_6": lambda i: f"{stats.tensors[i].weight} ({stats.tensors[i].check_id})",
    }
    out = {}
    for fam in SCORE_FAMILIES:
        if s[fam] is None:
            continue
        i = int(np.argmax(nan_as_inf(rho[fam].ravel())))
        out[fam] = (s[fam], s["n_failing"][fam], where[fam](i))
    return out


# ---- the report -------------------------------------------------------------------------


def report_unit(stats: Mapping[int, Any], cal: Calibration, c: DecoderComputation, *, k: int,
                T_run: int, out: Callable[[str], None] = print) -> None:
    """F18's band unit on the fitted steps: the largest ``U/‖P‖_F`` and its bound ``√w·κ``
    (``w`` the product's width; test 1 bounds ``U ≤ ν``), and P10d's ``k`` both ways."""
    eps = {"eps_in": UNIT_ROUNDOFF[c.operand_dtype],
           "eps_acc": UNIT_ROUNDOFF[c.accumulator_dtype]}
    big = largest_e_m(stats, **eps)
    worst = None  # (U/‖P‖ over √w·κ, U/‖P‖, √w·κ, where)
    per_kappa = None  # (U/‖P‖ over κ, where): the measured margin under √w·κ
    for t, st in sorted(stats.items()):
        for p in st.products:
            if p.p_norm == 0 or not math.isfinite(p.kappa):
                continue
            ratio = term_ratio(p.p_norm, p.u)
            bound = math.sqrt(_width(c, p.m)) * p.kappa
            score = ratio / bound if bound > 0 else math.inf
            if worst is None or score > worst[0]:
                worst = (score, ratio, bound, f"{p.name} (step {t})")
            pk = ratio / p.kappa if p.kappa > 0 else math.inf
            if per_kappa is None or pk > per_kappa[0]:
                per_kappa = (pk, f"{p.name} (step {t})")
    out(f"   band unit (F18): σ_r·e_m·‖P‖_F, e_m = √2·ε_in + √q·ε_acc·max(1, U/‖P‖_F); "
        f"U/‖P‖_F max {big.u_ratio_max:.4g} at {big.u_ratio_max_name}")
    if worst is not None:
        out(f"   U/‖P‖_F ≤ √w·κ: largest share {worst[0]:.4g} ({worst[1]:.4g} of "
            f"{worst[2]:.4g}, {worst[3]}): " + ("holds" if worst[0] <= 1 else "FAILS"))
    if per_kappa is not None:
        out(f"   U/‖P‖_F over κ: max {per_kappa[0]:.4g} ({per_kappa[1]})")
    out(f"   max R/τ = {cal.global_max / cal.tau:.4g} (τ/2 is 0.5)")
    q_max = max(s.q for s in c.products)
    sz = size_k(q_max, T=T_run, M=c.M, s_h=cal.s_h, z=cal.z, e_cal=big.e_m, e_cal_at=big.at,
                **eps)
    sq = size_k(q_max, T=T_run, M=c.M, s_h=cal.s_h, z=cal.z, **eps)
    out(f"   k (P10d, T {T_run}, configured {k}): {sz.k} at the largest e_m {big.e_m:.4e} "
        f"({big.at}, q {big.q}, U/‖P‖_F {big.ratio:.4g}); {sq.k} at e_m(q_max = {q_max}) "
        f"{sq.e_max:.4e}")


def _width(c: DecoderComputation, m: int) -> int:
    """Product ``m``'s width ``w``, the length of its ``1`` in ``|P|·1``."""
    return int(c.product(m).b_shape[1])


def report(b: BringUp, *, k: int, reference: Bands | None,
           c: DecoderComputation, out: Callable[[str], None] = print,
           T_run: int = 1) -> None:
    r, v = b.result, b.result.verifier
    loop = r.loop
    out("== honest steps (calibration mode: the exact checks judge, 5 and 6 record)")
    for s in loop.steps:
        out(f"   step {s.t}: loss {s.loss:.4f}, "
            + ("exact checks passed" if s.rejection is None else f"REJECTED {s.rejection}"))
    cal = b.cal
    if cal is None:
        out("   no fit: the run was rejected before its last step was verified")
    else:
        out(f"   in-sample fit: s_h {cal.s_h:.4g} ({cal.s_h_class}), τ = 8·s_h {cal.tau:.4g}, "
            f"global max {cal.global_max:.4g} ({cal.global_max_name})")
        out(f"   concentration guard: global max {cal.global_max:.4g} ≤ τ/2 = "
            f"{cal.guard_limit:.4g}: " + ("holds" if cal.guard_ok else "FAILS"))
        report_unit(v.stats, cal, c, k=k, T_run=T_run, out=out)
        out("   frozen on it: "
            + ("accepted" if b.freeze_rejection is None else str(b.freeze_rejection)))
    out("   run verdict (check 8 included): "
        + ("ACCEPTED" if loop.verdict.accepted else f"REJECTED {loop.verdict.rejection}")
        + ("" if b.guard_ok is not False else
           "; the fit FAILS the concentration guard, so this is not a clean pass"))
    out("== check 5: normalized residuals by matmul class (not judged)")
    format_class_table(class_summary(v.stats), out)
    out("== check 6: ρ_max by weight role (not judged)")
    format_tensor_table(tensor_summary(v.stats), out)
    if reference is not None:
        ref, unnamed = reference_bands(reference, c)
        out(f"== scores against the reference band file {reference.source[:16]}… (τ "
            f"{ref.tau:.4g}, default κ_max {ref.kappa_max:.4g}, τ_W {ref.tau_w:g}); "
            "information only")
        if unnamed:
            out(f"   classes the file doesn't name, at its default κ_max: {', '.join(unnamed)}")
        above: set[str] = set()
        for t in sorted(v.stats):
            for fam, (mx, n_over, at) in family_maxima(v.stats[t], ref, k=k).items():
                out(f"   step {t} {fam:<9} max {mx:.4g} at {at}; {n_over} above 1")
                if mx > 1:
                    above.add(fam)
        out("   families above 1: " + (", ".join(sorted(above)) if above else "none"))
    for tp in b.tampers:
        rej = tp.rejection
        out(f"== tamper: {tp.name}: expected check {tp.expected}, got "
            + ("ACCEPTED" if rej is None else f"check {rej.check_id} at step {rej.step}: "
               f"{rej.detail[:160]}")
            + ("" if tp.rejected_as_expected else "  <-- NOT AS EXPECTED")
            + ("" if b.guard_ok else "  [bands fail the concentration guard]"))
    out("== cost (wall clock from the memory pass)")
    report_costs(r, b.memory_rows, out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=1, help="T, at least 1 (default 1)")
    p.add_argument("--reference-bands", type=Path, default=None, metavar="PATH",
                   help="score the residuals against this band file (SmolLM2's), read-only")
    p.add_argument("--tamper", action=argparse.BooleanOptionalAction, default=True,
                   help="with q, k, v biases, run the altered-bias steps (default on)")
    p.add_argument("--disk-transcripts", type=Path, default=None, metavar="DIR",
                   help="hand each step to the verifier on disk under DIR (C3's DiskHandoff, "
                        "one directory per step, deleted once verified); default in memory")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error(f"T = {args.steps}: at least one step")
    cfg = load_config()
    k = load_protocol_config().k
    t0 = time.perf_counter()
    try:
        pin = model_pin(cfg.model, cfg.model_revision)
    except ValueError as e:  # a revision that isn't a commit, or a weight file off its hash
        p.error(str(e))
    hash_s = time.perf_counter() - t0
    if pin["weights"] is None:
        p.error(f"{cfg.model}@{cfg.model_revision} has no cached weights; this run never "
                "downloads")
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data with the "
                "same VERIF_MODEL and VERIF_OUTPUT_DIR")
    reference = None if args.reference_bands is None else Bands.from_file(args.reference_bands)
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    print(f"pin: {pin['repo']}@{pin['revision']}, config.json SHA-256 {pin['config_sha256']}")
    for f in pin["weights"]:
        print(f"     {f['file']} SHA-256 {f['sha256']} ({f['bytes'] / 2**30:.2f} GiB)")
    print(f"     weights hashed and checked against their cache blob names in {hash_s:.1f} s")
    kinds = Counter(q.kind.value for q in c.products)
    print(f"C: L {c.L}, d {c.d}, n_h {c.n_h}, n_kv {c.n_kv}, d_h {c.d_h}, n_v {c.n_v}, "
          f"tied {c.tied}, qkv_bias {c.qkv_bias}, pad (EOS) {c.pad_id}; n_w {c.n_w}, M {c.M} "
          f"({dict(kinds)}), {c.n_leaves} leaves")
    print(f"run: n_s {c.n_s}, n {c.n}, η {c.eta:g}, k {k}, T {args.steps}, seed {cfg.seed}, "
          f"threads {cfg.threads}, |D| {len(D)}, h_D {h_D.hex()[:16]}…")
    disk = args.disk_transcripts
    print("hand-off: " + ("in memory" if disk is None else f"on disk under {disk}")
          + ("; CUDA cache released before the verifier" if cfg.release_cache else
             "; no cache release (VERIF_RELEASE_CACHE=0)"))
    b = bring_up(c, D, h_D, T=args.steps, k=k, seed=cfg.seed, tamper=args.tamper,
                 device=cfg.device, handoff=None if disk is None else DiskHandoff(disk),
                 echo=lambda line: print(line, flush=True), release_cache=cfg.release_cache)
    report(b, k=k, reference=reference, c=c, T_run=cfg.steps)
    return 0 if b.passed else 1


if __name__ == "__main__":
    sys.exit(main())
