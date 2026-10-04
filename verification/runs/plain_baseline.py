"""B7 (T-H3): the plain-training baseline, EQ1b's P0.

    .venv/bin/python -m verification.runs.plain_baseline [--steps T] [--pass-steps S]
                                                         [--metrics | --no-metrics]

The honest run's training with capture and every protocol step off: ``T`` steps (``--steps``,
else ``VERIF_STEPS``, default 10) from the same ``W_0`` (the model ``LlamaComputation.from_config``
loads), on the same batches ``π(t)`` of ``D`` (``$VERIF_OUTPUT_DIR/data/D.bin``, from
``materialize_data``) at the same ``η`` (``VERIF_ETA``). Each step is
:func:`~verification.prover.step.plain_step` driven by :func:`~verification.runs.loop.run_plain`:
the prover's SGD step without ``MatmulCapture``, and no labeling, commitment, paths or verifier.

**Records** (B6, ``runs/metrics.py``), under ``$VERIF_OUTPUT_DIR/plain_baseline/`` unless
``--no-metrics`` or ``VERIF_METRICS=0``. Scenario ``plain``:

- the timed run's ``time`` rows per step: ``P0.load``, ``P0.forward``, ``P0.backward``,
  ``P0.update``, the ``P0`` component and the prover and step totals;
- a memory pass and a counting pass, each a separate run of the first ``S`` steps from ``W_0``
  (``--pass-steps``, default 2, the verified runs' pass length), never timed.

There is no ``step`` or ``verdict`` record: nothing is verified.

**Final weights.** ``final_weights.json`` in the same directory, written with metrics on or off:
the BLAKE3 leaf hash of each tensor of ``W_{T+1}``, encoded as a ``W_{t+1}`` leaf (tag ``0x02``,
dtype, shape, raw bytes), so equal hashes mean bit-identical weights, ``-0.0`` included. The
honest verified run (A12) checks its final weights against them with
:func:`assert_same_final_weights`.

**P1.** :func:`capture_rows` derives the matmul-capture rows from a verified run's records and
this run's (:func:`~verification.runs.metrics.derive_capture`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from setup.data import load_dataset_records
from verification.commitment.encoding import TAG_WEIGHT
from verification.commitment.leaves import dataset_root
from verification.commitment.merkle import hash_tensor_leaf, merkle_root
from verification.computation.interface import DeclaredComputation
from verification.runs.loop import PlainResult, run_plain
from verification.runs.metrics import (
    MetricsWriter,
    count_pass,
    derive_capture,
    memory_pass,
    memory_probe,
    read_records,
)

__all__ = ["RUN_NAME", "SCENARIO", "PASS_STEPS", "WEIGHTS_FILE", "initial_weights",
           "final_weight_hashes", "write_final_weights", "read_final_weights",
           "weight_mismatches", "assert_same_final_weights", "plain_baseline", "capture_rows",
           "main"]

RUN_NAME = "plain_baseline"  # the run's directory under VERIF_OUTPUT_DIR
SCENARIO = "plain"  # EVALUATION_SPEC §2
PASS_STEPS = 2  # the memory and counting passes: two steps, as the verified runs' passes
WEIGHTS_FILE = "final_weights.json"


# ---- final weights ------------------------------------------------------------------------


def initial_weights(c: DeclaredComputation, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """``W_0``: copies of the freshly built model's declared weights, in ``weight_names`` order."""
    return {n: model.get_parameter(n).detach().clone() for n in c.weight_names}


def final_weight_hashes(c: DeclaredComputation, w: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Each weight's leaf hash (hex) as a ``W_{t+1}`` leaf, in ``weight_names`` order."""
    if set(w) != set(c.weight_names):
        raise ValueError("the weights name different tensors from the computation's")
    return {n: hash_tensor_leaf(TAG_WEIGHT, w[n].detach().cpu().contiguous()).hex()
            for n in c.weight_names}


def write_final_weights(path: str | os.PathLike[str], c: DeclaredComputation,
                        w: Mapping[str, torch.Tensor], *, steps: int,
                        run: str = RUN_NAME) -> dict[str, Any]:
    """Write ``final_weights.json``: the per-tensor hashes and their Merkle root."""
    hashes = final_weight_hashes(c, w)
    doc = {"run": run, "steps": steps, "leaf_tag": TAG_WEIGHT,
           "root": merkle_root([bytes.fromhex(h) for h in hashes.values()]).hex(),
           "hashes": hashes}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def read_final_weights(path: str | os.PathLike[str]) -> dict[str, str]:
    """The per-tensor hashes of ``final_weights.json`` (a file or the run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / WEIGHTS_FILE
    return dict(json.loads(p.read_text())["hashes"])


def weight_mismatches(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]:
    """The tensor names whose hashes differ or that only one side has, in ``a``'s order."""
    return [n for n in a if a[n] != b.get(n)] + [n for n in b if n not in a]


def assert_same_final_weights(c: DeclaredComputation,
                              plain: Mapping[str, Any] | str | os.PathLike[str],
                              verified: Mapping[str, Any] | str | os.PathLike[str]) -> None:
    """Raise ``AssertionError`` unless the plain and verified runs end bit-identical.

    Each side is a ``final_weights.json`` path (or its directory), a hash mapping as
    :func:`final_weight_hashes` gives, or the weights themselves (``LoopResult.w_final``).
    """
    def hashes(x: Mapping[str, Any] | str | os.PathLike[str]) -> Mapping[str, str]:
        if isinstance(x, (str, os.PathLike)):
            return read_final_weights(x)
        if all(isinstance(v, torch.Tensor) for v in x.values()):
            return final_weight_hashes(c, x)
        return x

    bad = weight_mismatches(hashes(plain), hashes(verified))
    if bad:
        raise AssertionError(f"final weights differ in {len(bad)} tensors: {bad[:5]}")


# ---- the run ------------------------------------------------------------------------------


@dataclass
class BaselineResult:
    plain: PlainResult
    hashes: dict[str, str]
    weights_file: Path | None


def plain_baseline(c: DeclaredComputation, dataset: Sequence[Any],
                   w0: Mapping[str, torch.Tensor], *, T: int,
                   build_model: Callable[[], torch.nn.Module],
                   out_dir: str | os.PathLike[str] | None = None,
                   metrics: MetricsWriter | None = None, pass_steps: int = PASS_STEPS,
                   out: Callable[[str], None] = print) -> BaselineResult:
    """``T`` plain steps from ``W_0``; then, with ``metrics``, the memory and counting passes.

    The timed run and each pass build their own model with ``build_model`` and load ``W_0``
    into it. ``final_weights.json`` goes to ``out_dir`` (default ``metrics.dir``; none if both are
    ``None``).
    """
    if not 1 <= pass_steps <= T:
        raise ValueError(f"pass_steps = {pass_steps} must be in 1..T = {T}")
    rec = None if metrics is None else metrics.recorder(SCENARIO)
    res = run_plain(c, build_model(), dataset, w0, n_steps=T,
                    section=None if rec is None else rec.section,
                    on_step=None if rec is None else rec.on_step)
    if rec is not None:
        rec.finish()
        by_step = {(r["step"], r["component"]): r["time_s"] for r in rec.rows}
    for s in res.steps:
        line = f"   step {s.t:>3}  loss {s.loss:.6f}  train {s.train_s:.3f} s"
        if rec is not None:
            line += "  (" + ", ".join(f"{x.split('.')[1]} {by_step[s.t, x]:.3f}"
                                      for x in ("P0.load", "P0.forward", "P0.backward",
                                                "P0.update")) + ")"
        out(line)
    hashes = final_weight_hashes(c, res.w_final)
    target = out_dir if out_dir is not None else (None if metrics is None else metrics.dir)
    path = None
    if target is not None:
        path = Path(target) / WEIGHTS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = write_final_weights(path, c, res.w_final, steps=T,
                                  run=RUN_NAME if metrics is None else metrics.run)
        out(f"final weights: root {doc['root'][:16]}…  ({path})")
    if metrics is not None:
        def body(r: Any) -> Any:
            return run_plain(c, build_model(), dataset, w0, n_steps=pass_steps,
                             section=r.section, on_step=r.on_step)
        metrics.write(memory_pass(metrics.run, SCENARIO, body, memory_probe(metrics.device)))
        metrics.write(count_pass(metrics.run, SCENARIO, body))
        out(f"metrics: {metrics.dir}")
    return BaselineResult(res, hashes, path)


def capture_rows(verified: str | os.PathLike[str], plain: str | os.PathLike[str], *,
                 scenario: str | None = "honest") -> list[dict[str, Any]]:
    """P1 rows (``derive_capture``) from a verified run's ``records.jsonl`` (or directory) and
    this run's. ``scenario`` picks the verified scenario; ``None`` keeps them all."""
    v = read_records(verified)
    if scenario is not None:
        v = [r for r in v if r.get("scenario") == scenario]
    return derive_capture(v, read_records(plain))


def main(argv: Sequence[str] | None = None) -> int:
    from verification.computation.instances import LlamaComputation

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=None, help="T (default: VERIF_STEPS)")
    p.add_argument("--pass-steps", type=int, default=PASS_STEPS,
                   help=f"steps in the memory and counting passes (default {PASS_STEPS})")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help=f"write B6 metrics to $VERIF_OUTPUT_DIR/{RUN_NAME}/ "
                        "(default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    T = args.steps if args.steps is not None else cfg.steps
    if T < 1 or not 1 <= args.pass_steps <= T:
        p.error(f"need T ≥ 1 and 1 ≤ --pass-steps ≤ T, got T = {T}, {args.pass_steps}")
    d_path = cfg.data_dir / "D.bin"
    if not d_path.exists():
        p.error(f"no {d_path}; run verification.runs.materialize_data first")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    dataset = load_dataset_records(d_path, c.n)
    w0 = initial_weights(c, c.build_model())  # the model the verified run's prover loads
    h_D = dataset_root(dataset)
    print(f"plain baseline: {cfg.model} @ {cfg.model_revision[:8]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, T {T}, |D| {len(dataset)}, h_D {h_D.hex()[:16]}…")
    out_dir = cfg.output_dir / RUN_NAME
    if not use_metrics:
        plain_baseline(c, dataset, w0, T=T, build_model=c.build_model,
                       out_dir=out_dir)
        return 0
    run_cfg = {"T": T, "pass_steps": args.pass_steps}
    config = {**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                 if f.name not in ("output_dir", "metrics", "steps")}, **run_cfg}
    with MetricsWriter(out_dir, RUN_NAME, device=cfg.device, model=cfg.model,
                       corpus=cfg.dataset, seed=cfg.seed, config=config, band_file_hash=None,
                       h_D=h_D, extra={"scenario": SCENARIO, **run_cfg}) as mw:
        plain_baseline(c, dataset, w0, T=T, build_model=c.build_model,
                       metrics=mw, pass_steps=args.pass_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
