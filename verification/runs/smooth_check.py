"""F6c's smooth check: ``T`` plain steps at ``η`` and the loss over the whole clean list.

    VERIF_MODEL=<repo> VERIF_MODEL_REVISION=<commit> VERIF_ATTACK=<task> VERIF_OUTPUT_DIR=<dir> \\
        .venv/bin/python -m verification.runs.smooth_check [--steps T] [--eval-batch B]

Before the full-scale runs, each model and task is checked for a training curve that doesn't
blow up at the declared ``η`` (F6a, F6c; ``η = 10⁻³``, plain SGD, no tuning run). The check
trains ``T`` steps (``--steps``, default 10) from ``W_0`` with
:func:`~verification.runs.loop.run_plain`, the plain baseline's loop: the same
:func:`~verification.prover.step.plain_step`, the same ``π`` at ``VERIF_SEED`` and the same
``D`` (``$VERIF_OUTPUT_DIR/data/``, from ``materialize_data`` for the run's model and task). No
capture, commitment or verifier runs, and no text is generated.

**The whole-list loss.** At ``W_0`` and after each step, the training loss over every record of
``D`` (369 Alpaca records for refusal, 233 AdvBench records for jailbreak), forward passes
only: ``ℒ = Σ μ·CE / Σ μ`` over all of ``D``'s loss-masked tokens, the step's loss
(``DecoderComputation.loss_from_logits``) taken over the whole list instead of one batch. The
records go through in chunks of ``--eval-batch`` (default ``VERIF_BATCH``), each padded to
``VERIF_SEQ_LEN`` with the model's pad id, under the run's autocast, and the sums are kept in
fp64.

**The verdict** (F6a): smooth if every whole-list value is finite, none after a step rises above
the starting loss, and the last is below it. A failure goes to the user, who decides what
follows; it doesn't change ``η`` here.

**Record.** ``$VERIF_OUTPUT_DIR/smooth_check/<model>_<task>.json``: the model, task and run
settings; each step's batch loss and whole-list loss with its training and evaluation time; the
whole-list loss at ``W_0``; the verdict with its reasons; the wall clocks; the peak memory over
the run (``metrics.memory_probe``: allocated GPU memory on CUDA, the process footprint on the
Mac); and the environment (``metrics.environment_record``).

Exits 0 when smooth, :data:`EXIT_NOT_SMOOTH` when not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from setup.data import Batch, assemble_batch
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import load_weights, snapshot_weights
from verification.runs.llama_step import load_committed_dataset
from verification.runs.loop import PlainStepRecord, run_plain
from verification.runs.metrics import environment_record, memory_probe
from verification.runs.plain_baseline import training_config

__all__ = ["RUN_NAME", "STEPS", "EXIT_NOT_SMOOTH", "RULE", "Verdict", "verdict",
           "whole_list_loss", "SmoothResult", "smooth_check", "record", "record_path", "main"]

RUN_NAME = "smooth_check"
STEPS = 10  # F6c: 10 plain steps per model and task
EXIT_NOT_SMOOTH = 3  # apart from 1, a crash, and 2, a usage error
RULE = ("every whole-list loss finite, none after a step above the loss at W_0, and the last "
        "below it (F6a)")


# ---- the verdict ------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    smooth: bool
    reasons: tuple[str, ...]  # why not, one entry per failed condition; empty when smooth


def verdict(losses: Sequence[float]) -> Verdict:
    """F6a's rule on the whole-list losses ``[ℒ(W_0), ℒ(W_1), …, ℒ(W_T)]``."""
    if len(losses) < 2:
        raise ValueError(f"need the loss at W_0 and after at least one step, got {len(losses)}")
    start, after = losses[0], list(losses[1:])
    reasons = [f"not finite at W_{t}: {x!r}" for t, x in enumerate(losses) if not math.isfinite(x)]
    if math.isfinite(start):
        reasons += [f"above the start after step {t}: {x:.6f} > {start:.6f}"
                    for t, x in enumerate(after, start=1) if math.isfinite(x) and x > start]
        if math.isfinite(after[-1]) and not after[-1] < start:
            reasons.append(f"the last ({after[-1]:.6f}) is not below the start ({start:.6f})")
    return Verdict(not reasons, tuple(reasons))


# ---- the whole-list loss ----------------------------------------------------------------


def _to(b: Batch, device: torch.device | str) -> Batch:
    return Batch(b.ids.to(device), b.targets.to(device), b.mask.to(device), b.rho.to(device))


def whole_list_loss(c: DecoderComputation, model: torch.nn.Module, records: Sequence[Any], *,
                    chunk: int) -> float:
    """``Σ μ·CE / Σ μ`` over every loss-masked token of ``records``, forward passes only, on
    ``model`` as it stands. Chunks of at most ``chunk`` records, padded to ``c.n``; per-token
    cross-entropy as ``loss_from_logits`` computes it, under ``c.autocast()``."""
    if chunk < 1:
        raise ValueError(f"chunk = {chunk}: at least one record")
    total = torch.zeros((), dtype=torch.float64)
    weight = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(records), chunk):
            b = _to(assemble_batch(records[i:i + chunk], c.n, pad_id=c.pad_id), c.device)
            with c.autocast():
                logits = c.logits(model, b)
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), b.targets.reshape(-1),
                    reduction="none")
            mu = b.mask.reshape(-1).to(ce.dtype)
            total += (ce * mu).sum().double().cpu()
            weight += mu.sum().double().cpu()
            del logits, ce, mu, b
    if weight.item() == 0:
        raise ValueError("the records have no loss-masked target")
    return float(total / weight)


# ---- the run ------------------------------------------------------------------------------


@dataclass
class SmoothResult:
    w0_loss: float
    steps: list[dict[str, Any]] = field(default_factory=list)  # t, batch_loss, whole_list_loss,
    # train_s, eval_s
    verdict: Verdict | None = None
    w0_eval_s: float = 0.0
    train_eval_s: float = 0.0

    @property
    def whole_list_losses(self) -> list[float]:
        return [self.w0_loss, *(s["whole_list_loss"] for s in self.steps)]


def smooth_check(c: DecoderComputation, model: torch.nn.Module, D: Sequence[Any],
                 w0: dict[str, torch.Tensor], *, T: int, seed: int, chunk: int,
                 out: Callable[[str], None] = print) -> SmoothResult:
    """``T`` plain steps from ``W_0`` (``run_plain``) on ``model``, with the whole-list loss of
    ``D`` at ``W_0`` and after each step, and the verdict. ``W_0`` is loaded into ``model``
    first; after each step ``model`` holds ``W_{t+1}``, which the loss is taken on."""
    if T < 1:
        raise ValueError(f"T = {T}: at least one step")
    load_weights(c, model, w0)
    t0 = time.perf_counter()
    res = SmoothResult(w0_loss=whole_list_loss(c, model, D, chunk=chunk))
    res.w0_eval_s = time.perf_counter() - t0
    out(f"   W_0       whole-list loss {res.w0_loss:.6f}  ({res.w0_eval_s:.2f} s)")

    def on_step(rec: PlainStepRecord) -> None:
        # plain_step leaves the model holding W_{t+1} (its snapshot is run_plain's next W_t).
        e0 = time.perf_counter()
        loss = whole_list_loss(c, model, D, chunk=chunk)
        eval_s = time.perf_counter() - e0
        res.steps.append({"t": rec.t, "batch_loss": rec.loss, "whole_list_loss": loss,
                          "train_s": rec.train_s, "eval_s": eval_s})
        out(f"   step {rec.t:>3}  batch loss {rec.loss:.6f}  whole-list loss {loss:.6f}  "
            f"(train {rec.train_s:.2f} s, eval {eval_s:.2f} s)")

    t1 = time.perf_counter()
    run_plain(c, model, D, w0, n_steps=T, seed=seed, on_step=on_step)
    res.train_eval_s = time.perf_counter() - t1
    res.verdict = verdict(res.whole_list_losses)
    return res


def _finite(x: Any) -> Any:
    """``x`` with every non-finite float spelled as a string, so the file stays strict JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def record(res: SmoothResult, *, run: dict[str, Any], timings: dict[str, float],
           memory: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    """The JSON record of one model and task (module docstring)."""
    assert res.verdict is not None
    return _finite({
        "record": RUN_NAME, "run": run,
        "rule": RULE,
        "smooth": res.verdict.smooth, "reasons": list(res.verdict.reasons),
        "whole_list_loss_w0": res.w0_loss,
        "whole_list_losses": res.whole_list_losses,
        "batch_losses": [s["batch_loss"] for s in res.steps],
        "steps": res.steps,
        "timings_s": {"w0_eval": res.w0_eval_s, "steps_with_eval": res.train_eval_s,
                      **timings},
        "memory": memory, "environment": environment,
    })


def record_path(output_dir: Path, model: str, attack: str) -> Path:
    return output_dir / RUN_NAME / f"{model.split('/')[-1]}_{attack}.json"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=STEPS, help=f"T (default {STEPS}, F6c)")
    p.add_argument("--eval-batch", type=int, default=None,
                   help="records per forward pass of the whole-list loss (default VERIF_BATCH)")
    p.add_argument("--out", type=Path, default=None,
                   help="the JSON record (default "
                        f"$VERIF_OUTPUT_DIR/{RUN_NAME}/<model>_<task>.json)")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error(f"T = {args.steps}: at least one step")
    started = time.perf_counter()
    cfg = load_config()
    chunk = args.eval_batch or cfg.batch
    if chunk < 1:
        p.error(f"--eval-batch {chunk}: at least one record")
    meta_path = cfg.data_dir / "meta.json"
    if not (cfg.data_dir / "D.bin").exists() or not meta_path.exists():
        p.error(f"no D.bin or meta.json under {cfg.data_dir}: run "
                "verification.runs.materialize_data with the same VERIF_MODEL, VERIF_ATTACK "
                "and VERIF_OUTPUT_DIR")
    meta = json.loads(meta_path.read_text())
    if meta.get("attack") != cfg.attack:
        p.error(f"{meta_path} was built for task {meta.get('attack')!r}, VERIF_ATTACK is "
                f"{cfg.attack!r}")
    setup_determinism(cfg)
    probe = memory_probe(cfg.device)
    probe.reset()
    t0 = time.perf_counter()
    c = DecoderComputation.from_config(cfg)
    D, h_D = load_committed_dataset(c, cfg.data_dir)  # raises unless built for c's model
    model = c.build_model()
    w0 = snapshot_weights(c, model)
    load_s = time.perf_counter() - t0
    print(f"smooth check (F6c): {cfg.model} @ {cfg.model_revision[:8]}, task {cfg.attack}, "
          f"|D| {len(D)}, n_s {c.n_s}, n {c.n}, η {c.eta:g}, T {args.steps}, "
          f"({str(cfg.master_dtype).removeprefix('torch.')}, "
          f"{str(cfg.compute_dtype).removeprefix('torch.')}) on {cfg.device}, eval chunks of "
          f"{chunk}; loaded in {load_s:.1f} s")
    res = smooth_check(c, model, D, w0, T=args.steps, seed=cfg.seed, chunk=chunk)
    peak, current = probe.sample()
    assert res.verdict is not None
    run = {"model": cfg.model, "model_revision": cfg.model_revision, "task": cfg.attack,
           "n_records": len(D), "h_D": h_D.hex(), "n_s": c.n_s, "n": c.n, "eta": c.eta,
           "steps": args.steps, "seed": cfg.seed, "eval_batch": chunk, "device": cfg.device,
           "master_dtype": str(cfg.master_dtype), "compute_dtype": str(cfg.compute_dtype),
           "attn_impl": cfg.attn_impl, "threads": cfg.threads}
    env = environment_record(RUN_NAME, device=cfg.device, probe=probe,
                             config=training_config(cfg), h_D=h_D)
    doc = record(res, run=run, timings={"load": load_s, "total": time.perf_counter() - started},
                 memory={"source": probe.source, "peak_bytes": peak, "end_bytes": current},
                 environment=env)
    path = args.out or record_path(cfg.output_dir, cfg.model, cfg.attack)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    v = res.verdict
    print(f"verdict: {'SMOOTH' if v.smooth else 'NOT SMOOTH'}"
          + ("" if v.smooth else ": " + "; ".join(v.reasons)))
    if peak is not None:
        print(f"peak memory {peak / 2**30:.2f} GiB [{probe.source}]")
    print(f"record: {path}")
    return 0 if v.smooth else EXIT_NOT_SMOOTH


if __name__ == "__main__":
    sys.exit(main())
