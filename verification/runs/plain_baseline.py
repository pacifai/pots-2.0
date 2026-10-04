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
  (``--pass-steps``, default :data:`~verification.runs.metrics.PASS_STEPS`, the verified runs'
  pass length), never timed. P0's memory is reported as the step's growth, ``peak(P0.backward)
  − start(P0.forward)``, as ``derive_capture`` uses it.

There is no ``step`` or ``verdict`` record: nothing is verified.

**Final weights.** ``final_weights.json`` in the same directory, written with metrics on or off:
the BLAKE3 leaf hash of each tensor of ``W_{T+1}``, encoded as a ``W_{t+1}`` leaf (tag ``0x02``,
dtype, shape, raw bytes), so equal hashes mean bit-identical weights, ``-0.0`` included. Beside
them, what the run started from and how it trained: ``W_0``'s root over the same leaf hashes
check 0 anchors on, ``h_D``, the training config's hash, ``η``, threads, model and revision,
``T`` and the per-step losses. The honest verified run (A12) takes ``W_0`` with
:func:`~verification.computation.interface.snapshot_weights` from a freshly built model, as this
run does, and checks its final weights against the file with :func:`assert_same_final_weights`.

**P1.** :func:`capture_rows` derives the matmul-capture rows from a verified run's records and
this run's (:func:`~verification.runs.metrics.derive_capture`), at analysis time; the derived
rows are not written into either run's records.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from setup.config import RunConfig, load_config, setup_determinism
from setup.data import load_dataset_records
from verification.commitment.encoding import TAG_WEIGHT
from verification.commitment.leaves import dataset_root
from verification.commitment.merkle import hash_tensor_leaf, merkle_root
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.runs.loop import PlainResult, run_plain
from verification.runs.scenarios import ReusedModel
from verification.runs.metrics import (
    PASS_STEPS,
    MetricsWriter,
    config_hash,
    count_pass,
    derive_capture,
    memory_pass,
    read_records,
)

__all__ = ["RUN_NAME", "SCENARIO", "WEIGHTS_FILE", "PROVENANCE_FIELDS", "training_config",
           "run_provenance", "final_weight_hashes", "weights_root", "write_final_weights",
           "read_final_weights", "weight_mismatches", "provenance_mismatches",
           "assert_same_final_weights", "BaselineResult", "plain_baseline", "capture_rows",
           "main"]

log = logging.getLogger(__name__)

RUN_NAME = "plain_baseline"  # the run's directory under VERIF_OUTPUT_DIR
SCENARIO = "plain"  # EVALUATION_SPEC §2
WEIGHTS_FILE = "final_weights.json"

# final_weights.json's provenance fields, in the order assert_same_final_weights compares them,
# each with the name of its mismatch.
PROVENANCE_FIELDS: dict[str, str] = {
    "w0_root": "different start",
    "h_D": "different data",
    "eta": "different η",
    "steps": "different step count",
    "model": "different model",
    "model_revision": "different model revision",
    "threads": "different threads",
    "config_hash": "different training config",
}
_CONFIG_FIELDS = ("model", "model_revision", "threads", "config_hash", "h_D")


# ---- provenance ---------------------------------------------------------------------------


def training_config(cfg: RunConfig) -> dict[str, Any]:
    """The run config's fields that shape training: all but ``output_dir``, ``metrics`` and
    ``steps`` (``T`` is recorded on its own). The plain and verified runs hash this dict."""
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
            if f.name not in ("output_dir", "metrics", "steps")}


def run_provenance(cfg: RunConfig, h_D: bytes | None) -> dict[str, Any]:
    """``final_weights.json``'s config-side provenance: model, revision, threads, the hash of
    :func:`training_config` and ``h_D``."""
    return {"model": cfg.model, "model_revision": cfg.model_revision, "threads": cfg.threads,
            "config_hash": config_hash(training_config(cfg)),
            "h_D": None if h_D is None else h_D.hex()}


# ---- final weights ------------------------------------------------------------------------


def final_weight_hashes(c: DeclaredComputation, w: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Each weight's leaf hash (hex) as a ``W_t``/``W_{t+1}`` leaf (tag ``0x02``; for ``W_0``
    these are the hashes check 0 anchors on), in ``weight_names`` order."""
    if set(w) != set(c.weight_names):
        raise ValueError("the weights name different tensors from the computation's")
    return {n: hash_tensor_leaf(TAG_WEIGHT, w[n].detach().cpu().contiguous()).hex()
            for n in c.weight_names}


def weights_root(hashes: Mapping[str, str]) -> str:
    """The Merkle root (hex) over hex leaf hashes, in the mapping's order."""
    return merkle_root([bytes.fromhex(h) for h in hashes.values()]).hex()


def write_final_weights(path: str | os.PathLike[str], c: DeclaredComputation,
                        w: Mapping[str, torch.Tensor], *, w0: Mapping[str, torch.Tensor],
                        losses: Sequence[float], provenance: Mapping[str, Any] | None = None,
                        run: str = RUN_NAME) -> dict[str, Any]:
    """Write ``final_weights.json``: the per-tensor hashes, their root, and the provenance.

    ``provenance`` gives the config side (:func:`run_provenance`); a field it lacks is
    ``null``. ``W_0``'s root, ``η`` (``c.eta``), ``steps`` (``len(losses)``) and ``losses`` come
    from the arguments.
    """
    prov = dict(provenance or {})
    if set(prov) - set(_CONFIG_FIELDS):
        raise ValueError(f"unknown provenance fields {sorted(set(prov) - set(_CONFIG_FIELDS))}")
    hashes = final_weight_hashes(c, w)
    doc: dict[str, Any] = {"run": run}
    doc.update({k: prov.get(k) for k in PROVENANCE_FIELDS if k in _CONFIG_FIELDS})
    doc.update({"w0_root": weights_root(final_weight_hashes(c, w0)), "eta": c.eta,
                "steps": len(losses), "losses": [float(x) for x in losses],
                "leaf_tag": TAG_WEIGHT, "root": weights_root(hashes), "hashes": hashes})
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def read_final_weights(path: str | os.PathLike[str]) -> dict[str, Any]:
    """``final_weights.json`` as written (a file or the run directory); the per-tensor hashes
    are under ``"hashes"``."""
    p = Path(path)
    if p.is_dir():
        p = p / WEIGHTS_FILE
    return dict(json.loads(p.read_text()))


def weight_mismatches(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]:
    """The tensor names whose hashes differ or that only one side has, in ``a``'s order."""
    return [n for n in a if a[n] != b.get(n)] + [n for n in b if n not in a]


def _same(a: Any, b: Any) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float)
                      and math.isnan(a) and math.isnan(b))


def provenance_mismatches(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """What differs between two ``final_weights.json`` documents apart from their tensors: each
    :data:`PROVENANCE_FIELDS` entry, then the first step whose loss differs."""
    out = [f"{why} ({k}: {a.get(k)!r} vs {b.get(k)!r})" for k, why in PROVENANCE_FIELDS.items()
           if not _same(a.get(k), b.get(k))]
    la, lb = list(a.get("losses") or []), list(b.get("losses") or [])
    bad = [t for t, (x, y) in enumerate(zip(la, lb), start=1) if not _same(x, y)]
    if bad:
        t = bad[0]
        out.append(f"different loss at step {t} ({la[t - 1]!r} vs {lb[t - 1]!r})")
    elif len(la) != len(lb):
        out.append(f"different number of losses ({len(la)} vs {len(lb)})")
    return out


def assert_same_final_weights(c: DeclaredComputation,
                              plain: Mapping[str, Any] | str | os.PathLike[str],
                              verified: Mapping[str, Any] | str | os.PathLike[str]) -> None:
    """Raise ``AssertionError`` unless the plain and verified runs end bit-identical.

    Each side is a ``final_weights.json`` path (or its directory), a hash mapping as
    :func:`final_weight_hashes` gives, or the weights themselves (``LoopResult.w_final``). Each
    side's names must be exactly ``c.weight_names``. When both sides are files, their provenance
    is compared first and the mismatch named ("different start", "different η", …). A mapping
    that mixes tensors and hash strings is a ``TypeError``.
    """
    def side(x: Mapping[str, Any] | str | os.PathLike[str], what: str
             ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        if isinstance(x, (str, os.PathLike)):
            doc = read_final_weights(x)
            h: Mapping[str, Any] = doc["hashes"]
        else:
            doc = None
            tensors = [isinstance(v, torch.Tensor) for v in x.values()]
            if any(tensors) and not all(tensors):
                raise TypeError(f"{what}: the mapping mixes tensors and hashes")
            h = final_weight_hashes(c, x) if x and all(tensors) else x
            if not all(isinstance(v, str) for v in h.values()):
                raise TypeError(f"{what}: values must be tensors or hex hash strings")
        if set(h) != set(c.weight_names) or len(h) != len(c.weight_names):
            raise AssertionError(f"{what}: {len(h)} tensors named, not the computation's "
                                 f"{len(c.weight_names)} weights")
        return h, doc

    (hp, dp), (hv, dv) = side(plain, "plain"), side(verified, "verified")
    if dp is not None and dv is not None:
        why = provenance_mismatches(dp, dv)
        if why:
            raise AssertionError("the runs are not comparable: " + "; ".join(why))
    bad = weight_mismatches(hp, hv)
    if bad:
        raise AssertionError(f"final weights differ in {len(bad)} tensors: {bad[:5]}")


# ---- the run ------------------------------------------------------------------------------


@dataclass
class BaselineResult:
    plain: PlainResult  # ``w_final`` is empty when the run was asked not to keep it
    hashes: dict[str, str]
    weights_file: Path | None


def _growth(rows: Sequence[Mapping[str, Any]]) -> dict[int, int | None]:
    """P0's memory growth per step, ``peak(P0.backward) − start(P0.forward)``, in bytes."""
    by = {(r["step"], r["component"]): r for r in rows if r.get("level") == "phase"}
    out: dict[int, int | None] = {}
    for t in sorted({s for s, _ in by}):
        peak = by.get((t, "P0.backward"), {}).get("peak_bytes")
        start = by.get((t, "P0.forward"), {}).get("start_bytes")
        out[t] = None if peak is None or start is None else peak - start
    return out


def plain_baseline(c: DeclaredComputation, dataset: Sequence[Any],
                   w0: Mapping[str, torch.Tensor], *, T: int,
                   build_model: Callable[[], torch.nn.Module],
                   out_dir: str | os.PathLike[str] | None = None,
                   metrics: MetricsWriter | None = None, pass_steps: int = PASS_STEPS,
                   provenance: Mapping[str, Any] | None = None, keep_weights: bool = True,
                   out: Callable[[str], None] = print) -> BaselineResult:
    """``T`` plain steps from ``W_0``; then, with ``metrics``, the memory and counting passes.

    The timed run and each pass get their model from ``build_model`` (a fresh build, or a
    ``scenarios.ReusedModel``) and load ``W_0`` into it. ``final_weights.json`` goes to
    ``out_dir`` (default ``metrics.dir``; none if both are ``None``), with ``provenance``
    (:func:`run_provenance`). With ``keep_weights=False`` the final weights are dropped once
    hashed, before the passes, and ``plain.w_final`` is empty.
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
        doc = write_final_weights(path, c, res.w_final, w0=w0,
                                  losses=[s.loss for s in res.steps], provenance=provenance,
                                  run=RUN_NAME if metrics is None else metrics.run)
        out(f"final weights: root {doc['root'][:16]}…  ({path})")
    if not keep_weights:
        res = dataclasses.replace(res, w_final={})  # not held through the memory pass
    if metrics is not None:
        def body(r: Any) -> Any:
            run_plain(c, build_model(), dataset, w0, n_steps=pass_steps,
                      section=r.section, on_step=r.on_step)

        mem = memory_pass(metrics.run, SCENARIO, body, metrics.probe)
        metrics.write(mem)
        for t, g in _growth(mem).items():
            out(f"   step {t:>3}  P0 memory growth "
                + ("n/a" if g is None else f"{g / 1e9:.3f} GB")
                + " (peak of backward − start of forward)")
        metrics.write(count_pass(metrics.run, SCENARIO, body))
        out(f"metrics: {metrics.dir}")
    return BaselineResult(res, hashes, path)


def capture_rows(verified: str | os.PathLike[str], plain: str | os.PathLike[str], *,
                 scenario: str | None = "honest") -> list[dict[str, Any]]:
    """P1 rows (``derive_capture``) from a verified run's ``records.jsonl`` (or directory) and
    this run's. ``scenario`` picks the verified scenario; ``None`` keeps them all.

    An analysis-time function: the rows are returned, not written into either run. A plain-run
    step with no verified counterpart in the same record kind (``time``, ``memory``, ``count``)
    gets no P1 row and raises a ``UserWarning`` naming it.
    """
    v = read_records(verified)
    if scenario is not None:
        v = [r for r in v if r.get("scenario") == scenario]
    p = read_records(plain)

    def steps(rows: Sequence[Mapping[str, Any]], kind: str) -> set[int]:
        return {r["step"] for r in rows
                if r.get("record") == kind and r.get("level") == "phase" and r["step"] > 0}

    for kind in ("time", "memory", "count"):
        missing = sorted(steps(p, kind) - steps(v, kind))
        if missing:
            warnings.warn(f"capture_rows: plain-run {kind} steps {missing} have no verified "
                          "counterpart; no P1 row for them", UserWarning, stacklevel=2)
    return derive_capture(v, p)


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
    if use_metrics and sys.platform == "darwin" and os.environ.get("MallocLargeCache") != "0":
        log.warning("MallocLargeCache is not 0: macOS keeps freed large blocks in the footprint, "
                    "so this run's memory rows don't compare with other runs'")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    dataset = load_dataset_records(d_path, c.n)
    # One from_pretrained model for W_0 (as the verified run's prover loads it), the timed run
    # and both passes; each run loads W_0 into it before use (scenarios.ReusedModel).
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    h_D = dataset_root(dataset)
    prov = run_provenance(cfg, h_D)
    print(f"plain baseline: {cfg.model} @ {cfg.model_revision[:8]}, n_s {c.n_s}, n {c.n}, "
          f"η {c.eta:g}, T {T}, |D| {len(dataset)}, h_D {h_D.hex()[:16]}…")
    out_dir = cfg.output_dir / RUN_NAME
    if not use_metrics:
        plain_baseline(c, dataset, w0, T=T, build_model=models, out_dir=out_dir,
                       provenance=prov, keep_weights=False)
        return 0
    run_cfg = {"T": T, "pass_steps": args.pass_steps}
    with MetricsWriter(out_dir, RUN_NAME, device=cfg.device, model=cfg.model,
                       corpus=cfg.dataset, seed=cfg.seed,
                       config={**training_config(cfg), **run_cfg}, band_file_hash=None,
                       h_D=h_D, extra={"scenario": SCENARIO, **run_cfg}) as mw:
        plain_baseline(c, dataset, w0, T=T, build_model=models, metrics=mw,
                       pass_steps=args.pass_steps, provenance=prov, keep_weights=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
