"""Milestone M3: honest SmolLM2 steps verified end to end.

    .venv/bin/python -m verification.runs.llama_step [--steps T] [--bands PATH]
        [--metrics | --no-metrics]

The SmolLM2 instance (``DecoderComputation.from_config``: ``VERIF_MODEL`` at its pinned revision,
``VERIF_BATCH × VERIF_SEQ_LEN``, ``η = VERIF_ETA``) runs ``T`` honest steps (``--steps``, default
1) from ``W_0``, the unmodified ``from_pretrained`` weights, over ``π``'s batches (F8c, seeded
by ``VERIF_SEED``) of the committed dataset ``D`` (``$VERIF_OUTPUT_DIR/data/D.bin``, written by
``materialize_data``). Each
step goes through the S3 loop: prove, commit, hand the in-memory store to the real
:class:`~verification.verifier.driver.Verifier`, whose check 5 rebuilds every operand with
``DecoderReplay``. The verifier's ``h_D`` is the published root in ``meta.json``, so check 1
compares the loaded ``D`` with it. Check 8 compares with ``T`` uncaptured ``plain_step``s from
``W_0``.

Bands are provisional by default (``τ = 8``, ``κ = 10⁴``, ``τ_W = 4``;
``allow_provisional=True``). ``τ = 8`` predates C1 and sits below honest ``Λ`` residuals (C1
measured up to 14.5), so an honest step can fail check 5 under it. ``--bands PATH`` judges with
a band file instead, read-only through ``calibration.load_bands``, which refuses a file fitted
for another run (``band_context``; P10a). Exits 1 unless the run is accepted. It prints the
per-step summary, the per-class normalized-residual table of check 5 and the per-weight ρ table
of check 6 (``verifier/residuals.py``, which A11 reuses), and each step's wall clock.

With metrics on (``--metrics``, default ``VERIF_METRICS=1``) the timed run writes its records and
residual arrays to ``$VERIF_OUTPUT_DIR/llama_step/`` (B6), and two separate untimed passes of the
same ``T`` steps write the per-component memory peaks and the counts; the prover's and the
verifier's peaks are printed from the memory pass. ``VERIF_RELEASE_CACHE=0`` skips the loop's
CUDA cache release between prover and verifier in the timed run and the memory pass.
``--no-count`` skips the counting pass (the counts do not depend on the GPU layout). After the
costs the run prints each step's transcript root, ``step t root <hex>``, as ``run_verified`` does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from setup import backdoorllm
from setup.config import load_config, setup_determinism
from setup.data import load_dataset_records
from verification.computation.interface import (
    DeclaredComputation,
    ProductKind,
    snapshot_weights,
)
from verification.computation.instances.decoder import DecoderComputation
from verification.parameters import load_protocol_config
from verification.runs.loop import run_devices, table_cache_report
from verification.runs.metrics import (
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    lifetime_maxrss_bytes,
)
from verification.runs.prover_spec import loop_mode
from verification.runs.scenarios import (
    HONEST,
    ReusedModel,
    ScenarioResult,
    count_run,
    honest_final,
    memory_run,
    report,
    run_scenario,
)
from verification.verifier.bands import Bands
from verification.verifier.calibration import band_context, load_bands
from verification.verifier.context import Section
from verification.verifier.driver import Verifier
from verification.verifier.residuals import (
    class_summary,
    format_class_table,
    format_tensor_table,
    tensor_summary,
)

__all__ = ["run_honest", "report_residuals", "report_costs", "report_roots",
           "load_committed_dataset", "check_data_model", "main"]

RUN_NAME = "llama_step"  # the metrics directory under VERIF_OUTPUT_DIR


def check_data_model(meta: Mapping[str, Any], *, source: tuple[str, str] | None, pad_id: int,
                     where: Path | str) -> None:
    """Raise ``ValueError`` unless ``meta.json`` (``materialize_data.build_meta``) was built for
    the run's model: its run tokenizer is ``source`` (repo and revision) and its ``pad_id`` is
    the run's. A missing field doesn't match. Each model has its own ``D`` (S2): a directory
    built for another model holds other token ids and pads with another id.

    A random test model (``source`` ``None``) has no tokenizer to compare; its ``pad_id`` is
    compared when ``meta.json`` has one."""
    tokenizers = meta.get("tokenizers")
    run = tokenizers.get("run") if isinstance(tokenizers, Mapping) else None
    built = (run.get("repo"), run.get("revision")) if isinstance(run, Mapping) else (None, None)
    built_pad = meta.get("pad_id")
    if source is None:
        if built_pad is None or built_pad == pad_id:
            return
    elif built == tuple(source) and built_pad == pad_id:
        return
    model = "a random test model" if source is None else f"{source[0]}@{source[1]}"
    raise ValueError(
        f"{where}: the data was built for {built[0]}@{built[1]} with pad id {built_pad}, but "
        f"the run's model is {model} with pad id {pad_id}. Rerun "
        "verification.runs.materialize_data with VERIF_MODEL and VERIF_MODEL_REVISION set to "
        "the run's model, in its own VERIF_OUTPUT_DIR")


def load_committed_dataset(c: DecoderComputation, data_dir: Path) -> tuple[list[Any], bytes]:
    """``D`` from ``D.bin`` (every record validated) and the published ``h_D`` from
    ``meta.json``, both written by ``materialize_data``, once :func:`check_data_model` has
    found ``meta.json`` built for ``c``'s model and pad id."""
    meta = json.loads((data_dir / "meta.json").read_text())
    check_data_model(meta, source=c.source, pad_id=c.pad_id, where=data_dir / "meta.json")
    D = load_dataset_records(data_dir / "D.bin", c.n)
    return D, bytes.fromhex(meta["h_D"])


def run_honest(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               *, T: int, k: int, seed: int, h_D: bytes, final: Mapping[str, torch.Tensor],
               recorder: TimeRecorder | MemoryRecorder | CountRecorder | None = None,
               build_model: Callable[[], torch.nn.Module] | None = None,
               verifier: Callable[[Section | None], Verifier] | None = None,
               release_cache: bool | None = None, pipelined: bool = False,
               prover: Callable[[], Any] | None = None) -> ScenarioResult:
    """``T`` honest steps through the S3 loop, judged by provisional bands unless ``verifier``
    (as in ``run_scenario``, with ``release_cache``) gives a judging verifier. ``pipelined``
    and ``prover`` are ``run_scenario``'s; the mode is the caller's, not ``VERIF_PIPELINED``."""
    return run_scenario(c, dataset, w0, HONEST, T=T, k=k, seed=seed, final=final,
                        recorder=recorder, h_D=h_D, build_model=build_model, verifier=verifier,
                        release_cache=release_cache, pipelined=pipelined, prover=prover)


def _band_label(bands: Bands) -> str:
    return "provisional" if bands.source == "provisional" else f"band file {bands.source[:16]}…"


def report_residuals(r: ScenarioResult, out: Callable[[str], None] = print) -> None:
    """Check 5's per-class table and check 6's per-weight-role table over the run's steps."""
    stats, bands = r.verifier.stats, r.verifier.bands
    out(f"check 5: normalized residuals by matmul class ({_band_label(bands)}: τ = "
        f"{bands.tau:g}, κ_max = {bands.kappa_max:g})")
    format_class_table(class_summary(stats), out)
    out(f"check 6: ρ_max by weight role ({_band_label(bands)}: τ_W = {bands.tau_w:g})")
    format_tensor_table(tensor_summary(stats), out)


def report_costs(r: ScenarioResult, memory_rows: Sequence[Mapping[str, Any]] | None = None,
                 out: Callable[[str], None] = print) -> None:
    """Each step's wall clock (prover: ``prove_step`` and the commit; verifier: every check;
    the hand-off between them, which on CUDA holds ``run_loop``'s cache release), and with the
    memory pass's rows, the process peak during each side's sections (on CUDA with the caching
    allocator's reserved peak beside it)."""
    for s in r.loop.steps:
        out(f"step {s.t}: prover {s.prove_s + s.commit_s:.2f} s (prove_step {s.prove_s:.2f} s, "
            f"commit {s.commit_s:.2f} s); verifier {s.verify_s:.2f} s "
            "(" + ", ".join(f"{c} {x:.2f}" for c, x in r.verifier.timings.get(s.t, {}).items())
            + f"); hand-off {s.handoff_s:.2f} s")
    for row in memory_rows or ():
        if row.get("level") == "total" and row["component"] in ("prover", "verifier") \
                and row["step"] > 0 and row["peak_bytes"] is not None:
            start, held = row["start_bytes"], row.get("peak_reserved_bytes")
            out(f"step {row['step']}: {row['component']} peak {row['peak_bytes'] / 2**30:.2f} GB"
                + ("" if held is None else f", reserved {held / 2**30:.2f} GB")
                + ("" if start is None else f" (at its start {start / 2**30:.2f} GB)")
                + f" [{row['mem_source']}, memory pass]")
    out(f"process lifetime peak RSS {lifetime_maxrss_bytes() / 2**30:.2f} GB")


def report_roots(r: ScenarioResult, out: Callable[[str], None] = print) -> None:
    """Each step's transcript root, one ``step t root <hex>`` line per step, as
    ``run_verified`` prints them (the sessions compare them across arms)."""
    for s in r.loop.steps:
        if s.root is not None:
            out(f"step {s.t:>2} root {s.root.hex()}")


# ---- the timed run's work counts ----------------------------------------------------------


def work_rows(c: DecoderComputation, run: str, scenario: str,
              step_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The ``count`` and ``storage`` rows of a timed run, from shapes, for the profile
    analyzer's floors: the counting pass is never run beside a timed run, and it slows the code.

    Per step in ``step_rows`` (the run's ``step`` records, which hold ``transcript_bytes`` and
    ``leaves``): ``train.forward`` and ``train.backward`` get the matmul FLOPs the inventory
    declares (``2·rows·contracted·cols`` per product, the forward kind in the first, the three
    gradient kinds in the second; a bias's ones column and the elementwise glue are left out);
    ``2`` (verifier) gets the leaf hashing of the whole transcript, ``transcript_bytes`` plus
    the ``0x00`` prefix byte per leaf in, 32 bytes per leaf out, ``hash_calls`` and
    ``hash_leaves`` the leaf count (check 2 rehashes every leaf). ``P3.commit`` (prover) counts
    what the commit itself hashes. On a CPU run that is the whole transcript, as for check 2.
    On a CUDA run (``c.device``) the products are hashed on arrival, on their owner GPUs inside
    ``train.*`` (those rows carry FLOPs only), and the commit takes those digests as ``known``,
    so it hashes the batch records, ``W_t``, ``W_{t+1}`` and the ``dK`` products (the committed
    ``δK̃`` leaf is the contiguous transpose made in labeling, so its arrival digest goes
    unused): ``hash_leaves`` and ``hash_calls`` are ``leaves`` minus the other products,
    ``hash_in_bytes`` their bytes (``transcript_bytes`` minus those products' bytes, each
    ``prod(p_shape)`` times the product dtype's size) plus the prefix byte per leaf,
    ``hash_out_bytes`` 32 per leaf. A step row that carries ``known`` (the leaves the commit
    took from arrival digests, ``HashStats.known``; absent or ``None`` on records without it) is
    checked against that rule: ``leaves`` minus ``known`` must equal the rule's commit leaf
    count, else one ``WARNING step <t>: ...`` line goes to stderr; the commit row then carries
    the measured ``known`` as its own field. A ``storage`` row carries ``transcript_bytes`` and
    ``leaves``. Nothing is read from the device."""
    fwd = bwd = 0
    for q in c.products:
        f = 2 * q.a_shape[0] * q.a_shape[1] * q.b_shape[1]
        if q.kind is ProductKind.FORWARD:
            fwd += f
        else:
            bwd += f
    skipped_n = skipped_bytes = 0  # the products the commit takes from arrival digests
    if c.device.type == "cuda":
        psize = torch.empty((), dtype=c.product_dtype).element_size()
        for q in c.products:
            if c.product_class(q) != "dK":
                skipped_n += 1
                skipped_bytes += psize * math.prod(q.p_shape)
    rows: list[dict[str, Any]] = []
    for r in step_rows:
        t, leaves, nbytes = r["step"], r["leaves"], r["transcript_bytes"]
        c_leaves, c_bytes = leaves - skipped_n, nbytes - skipped_bytes
        known = r.get("known")
        if known is not None and leaves - known != c_leaves:
            print(f"WARNING step {t}: the commit hashed {leaves - known} leaves (leaves {leaves} "
                  f"minus known {known}), the work-row rule counts {c_leaves}; its hash floor "
                  "is off", file=sys.stderr)
        base = {"record": "count", "run": run, "scenario": scenario, "step": t,
                "calls": 1, "derived": False}
        for side, name, level, flops, h_in, h_out, h_calls, h_leaves in (
                ("prover", "train.forward", "phase", fwd, 0, 0, 0, 0),
                ("prover", "train.backward", "phase", bwd, 0, 0, 0, 0),
                ("prover", "P3.commit", "component", 0, c_bytes + c_leaves, 32 * c_leaves,
                 c_leaves, c_leaves),
                ("verifier", "2", "component", 0, nbytes + leaves, 32 * leaves, leaves,
                 leaves)):
            rows.append({**base, "side": side, "component": name, "level": level,
                         "flops": flops, "hash_in_bytes": h_in, "hash_out_bytes": h_out,
                         "hash_calls": h_calls, "hash_leaves": h_leaves,
                         **({"known": known} if name == "P3.commit" and known is not None
                            else {})})
        rows.append({"record": "storage", "run": run, "scenario": scenario, "step": t,
                     "transcript_bytes": nbytes, "leaves": leaves})
    return rows


def _step_records(run_dir: Path) -> list[dict[str, Any]]:
    """The ``step`` records the run has written so far (the writer flushes each)."""
    with open(run_dir / "records.jsonl") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return [r for r in rows if r.get("record") == "step"]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=1, help="T, at least 1 (default 1)")
    p.add_argument("--bands", type=Path, default=None, metavar="PATH",
                   help="judge with this band file, checked against the run's context "
                        "(default: provisional bands)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    p.add_argument("--count", action=argparse.BooleanOptionalAction, default=True,
                   help="with metrics, run the counting pass (default); --no-count skips it "
                        "(counts do not depend on the GPU layout)")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error(f"T = {args.steps}: at least one step")
    cfg = load_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    k, T = load_protocol_config().k, args.steps
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    bands = (Bands.provisional() if args.bands is None else  # read-only (P10a)
             load_bands(args.bands, k=k, context=band_context(cfg, c, h_D=h_D, k=k)))
    # One from_pretrained model serves every prover-side run in turn: W_0 is taken from it
    # unmodified, then honest_final's plain run and the prover's runs each load their weights
    # into it before use. The verifier builds its own (invariant 1).
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    final = honest_final(c, D, w0, T, seed=cfg.seed, build_model=models)
    print(f"SmolLM2 honest run: {cfg.model}@{cfg.model_revision[:12]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, k {k}, T {T}, |D| {len(D)}, h_D {h_D.hex()[:16]}…, M {c.M}, "
          f"{c.n_leaves} leaves, bands {_band_label(bands)}")
    make = None
    if args.bands is not None:
        from verification.runs.calibrate import judging_verifier  # calibrate imports this module
        make = judging_verifier(c, len(D), w0, bands, T=T, k=k, seed=cfg.seed, h_D=h_D)
    run_args = dict(T=T, k=k, seed=cfg.seed, h_D=h_D, final=final, build_model=models,
                    verifier=make)
    pl, spec = loop_mode(cfg)
    if pl:
        models.drop()  # the prover is the child's: no second model on the parent's GPU 0
    mode = dict(pipelined=pl, prover=spec)
    cache = dict(release_cache=cfg.release_cache)  # not count_run's: it counts, on any device
    mem_rows = None
    if not use_metrics:
        if cfg.trace_step > 0:
            print(f"trace: VERIF_TRACE_STEP={cfg.trace_step} needs --metrics; nothing is traced",
                  flush=True)
        r = run_honest(c, D, w0, **run_args, **cache, **mode)
    else:
        llama = {"n_s": c.n_s, "n": c.n, "eta": c.eta, "k": k, "T": T, "n_records": len(D),
                 "M": c.M, "n_leaves": c.n_leaves}
        # The config hash covers what decides the run's results, not where it writes.
        config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                     if f.name not in ("output_dir", "metrics", "release_cache", "trace_step",
                                       "trace_c5", "pipelined")},
                  "llama": llama}
        with MetricsWriter(cfg.output_dir / RUN_NAME, RUN_NAME, device=cfg.device,
                           model=cfg.model, corpus=backdoorllm.SOURCE, attack=cfg.attack,
                           seed=cfg.seed, config=config,
                           band_file_hash=None if make is None else bands.source, h_D=h_D,
                           extra={"band_source": "provisional" if make is None
                                  else str(args.bands), **llama}) as mw:
            mw.write_c5_layout(c)  # what the markers' check-5 stamps index
            devs = run_devices(c, pipelined=pl)
            # Only the timed run is traced: the memory and counting passes get no trace_step.
            trace = dict(trace_step=cfg.trace_step, trace_c5=cfg.trace_c5)
            if pl and cfg.trace_step > 0:
                print(f"trace: VERIF_TRACE_STEP={cfg.trace_step} is not traced in the pipelined "
                      "loop (the profiler records one process)", flush=True)
                trace = dict(trace_step=0)
            r = run_honest(c, D, w0, **run_args, **cache, **mode,
                           recorder=mw.recorder(HONEST.name, devices=devs, **trace))
            if not args.count:
                # The counting pass is never run beside a timed run (it slows the code); the
                # profile analysis' floors come from shapes, one count row per component and
                # step. With --count the pass's rows are the only ones.
                mw.write(work_rows(c, RUN_NAME, HONEST.name, _step_records(mw.dir)))
            mem_rows = memory_run(c, D, w0, run=RUN_NAME, **run_args, **cache, **mode,
                                  device=mw.device, devices=devs)
            mw.write(mem_rows)
            if args.count:
                mw.write(count_run(c, D, w0, run=RUN_NAME, pipelined=False,
                                   **run_args))  # sequential
            cache_line, cache_row = table_cache_report(RUN_NAME)
            mw.write([cache_row])
        print(f"metrics: {mw.dir}")
    if not use_metrics:
        cache_line, _ = table_cache_report(RUN_NAME)
    print(cache_line, flush=True)
    report(r)
    report_residuals(r)
    report_costs(r, mem_rows)
    report_roots(r)
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
