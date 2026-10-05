"""C3 (A14): the verifier decides the same from an in-memory store as from the on-disk one.

    .venv/bin/python -m verification.runs.store_crosscheck [--steps T] [--metrics | --no-metrics]

S3 keeps two ways of handing a committed step to the verifier: :data:`~verification.
transcript.store.IN_MEMORY` (references to the prover's tensors, the test-scale default) and
:class:`~verification.transcript.store.DiskHandoff` (each step written to its own directory
and read back from disk, one ``torch.save`` file per leaf). Both serve the same
``TranscriptStore`` interface. This run sends each scenario through both and asserts that the
verifier's decisions are identical (:func:`compare`):

- the verdict: accepted or not, the rejection's ``(step, check_id, kind, detail)``, the steps
  verified and the band source;
- per step: the rejection, the claimed root ``h`` the store served, the ``W_{t+1}`` hashes the
  verifier kept from its own check-2 recomputation, the loss, and every check-5 and check-6
  number in ``verifier.stats``, bit for bit (NaN equal to NaN), the scale fields that
  ``ProductStat`` leaves out of equality included;
- the leaf hashes of the prover's final weights.

A store that served different bytes would change a root, a hash or a residual, so equal
decisions here mean the verifier read the same leaves from disk as from memory.

The SmolLM2 run (``main``) is set up as ``runs/run_verified.py``: ``W_0`` from the unmodified
``from_pretrained`` model, ``π``'s batches of the committed ``D``, and the frozen band file read
with ``calibration.load_bands`` (P10a). Its scenarios: the honest run of ``T`` steps
(``--steps``, else ``VERIF_STEPS``), and four cheap faults at step 2, each a rejection at a
different check (:func:`llama_scenarios`). Transcripts go to
``$VERIF_OUTPUT_DIR/store_crosscheck/transcripts/<scenario>/step_<t>/``; each step's directory
is deleted once the step is verified, and the scenario's directory after the run. With metrics
on (default ``VERIF_METRICS``), an untimed memory pass of ``PASS_STEPS`` honest steps per store
follows. Exits 1 if any decision differs or any scenario misses its declared outcome.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from setup.config import load_config, setup_determinism
from verification.commitment.leaves import dataset_tree, leaf_hash
from verification.commitment.merkle import MerkleTree
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.runs.loop import ProverFault, StepRecord
from verification.runs.metrics import PASS_STEPS
from verification.runs.scenarios import (
    HONEST,
    BuildModel,
    Expected,
    ReusedModel,
    Scenario,
    ScenarioResult,
    honest_final,
    memory_run,
    run_scenario,
)
from verification.transcript.store import (
    IN_MEMORY,
    DiskHandoff,
    DiskStore,
    StoreHandoff,
    TranscriptStore,
)
from verification.verifier.bands import Bands
from verification.verifier.context import Section, StepStats
from verification.verifier.driver import Verifier

__all__ = ["RUN_NAME", "FAULT_STEP", "Decisions", "RecordingHandoff", "decide", "compare",
           "StoreRun", "CrossCheck", "crosscheck", "report", "report_memory", "OtherBatch", "llama_scenarios",
           "main"]

RUN_NAME = "store_crosscheck"
FAULT_STEP = 2
ULPS = 300


# ---- what the verifier decided ------------------------------------------------------------


def _bits(x: float) -> str:
    """A float's exact value as a string: equal strings ⇔ equal bits up to NaN payloads."""
    return float(x).hex()


def _stats_key(st: StepStats) -> tuple[Any, ...]:
    """Every number in ``st``, the ``compare=False`` scale fields included, as exact strings."""
    def row(obj: Any) -> tuple[Any, ...]:
        out: list[Any] = []
        for f in fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, float):
                v = _bits(v)
            elif isinstance(v, tuple):
                v = tuple(_bits(x) for x in v)
            out.append((f.name, v))
        return tuple(out)
    return (tuple(row(p) for p in st.products), tuple(row(x) for x in st.tensors))


@dataclass
class Decisions:
    """One run's verifier decisions and the facts they rest on (see the module docstring)."""

    verdict: tuple[Any, ...]
    steps: list[tuple[Any, ...]] = field(default_factory=list)
    roots: dict[int, str] = field(default_factory=dict)
    chain: dict[int, tuple[str, ...]] = field(default_factory=dict)
    stats: dict[int, tuple[Any, ...]] = field(default_factory=dict)
    final: tuple[str, ...] | None = None


class RecordingHandoff(StoreHandoff):
    """Wraps a hand-off: records each step's claimed root as the store serves it, the bytes its
    files take on disk, and the time ``hold`` takes (the write, for a disk store)."""

    def __init__(self, inner: StoreHandoff) -> None:
        self.inner = inner
        self.roots: dict[int, bytes] = {}
        self.disk_bytes: dict[int, int] = {}
        self.write_s: dict[int, float] = {}

    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> TranscriptStore:
        t0 = time.perf_counter()
        store = self.inner.hold(c, t, leaves, tree, dataset_paths)
        self.write_s[t] = time.perf_counter() - t0
        self.roots[t] = store.root
        if isinstance(store, DiskStore):
            self.disk_bytes[t] = sum(p.stat().st_size for p in store.directory.iterdir())
        return store

    def release(self, store: TranscriptStore) -> None:
        self.inner.release(store)


def decide(c: DeclaredComputation, r: ScenarioResult, roots: Mapping[int, bytes],
           chain: Mapping[int, tuple[bytes, ...]]) -> Decisions:
    v, loop = r.verifier, r.loop.verdict

    def rej(x: Any) -> tuple[Any, ...] | None:
        return None if x is None else (x.step, x.check_id, x.kind, x.detail)
    d = Decisions(verdict=(loop.accepted, rej(loop.rejection), loop.steps_verified,
                           loop.band_source))
    for s in r.loop.steps:
        d.steps.append((s.t, rej(s.rejection), _bits(s.loss)))
    d.roots = {t: h.hex() for t, h in roots.items()}
    d.chain = {t: tuple(h.hex() for h in hs) for t, hs in chain.items()}
    d.stats = {t: _stats_key(st) for t, st in v.stats.items()}
    w = r.loop.w_final
    if w is not None:
        d.final = tuple(leaf_hash(c, c.w_next_index(n), w[n]).hex() for n in c.weight_names)
    return d


def compare(a: Decisions, b: Decisions) -> list[str]:
    """Every difference between two runs' decisions, in words; empty when identical."""
    diffs: list[str] = []
    if a.verdict != b.verdict:
        diffs.append(f"verdict: {a.verdict} vs {b.verdict}")
    if a.steps != b.steps:
        diffs.append(f"steps: {a.steps} vs {b.steps}")
    for name in ("roots", "chain", "stats"):
        x, y = getattr(a, name), getattr(b, name)
        if x.keys() != y.keys():
            diffs.append(f"{name}: steps {sorted(x)} vs {sorted(y)}")
        diffs += [f"{name} differ at step {t}" for t in sorted(x.keys() & y.keys())
                  if x[t] != y[t]]
    if a.final != b.final:
        diffs.append("final weights differ")
    return diffs


# ---- running one scenario through both stores ---------------------------------------------


@dataclass
class StoreRun:
    """One scenario through one store: its result, decisions and costs."""

    store: str
    result: ScenarioResult
    decisions: Decisions
    write_s: dict[int, float]
    disk_bytes: dict[int, int]


@dataclass
class CrossCheck:
    scenario: Scenario
    memory: StoreRun
    disk: StoreRun
    diffs: list[str]

    @property
    def identical(self) -> bool:
        return not self.diffs


def _run(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
         scenario: Scenario, name: str, handoff: StoreHandoff, *, T: int, k: int,
         final: Mapping[str, torch.Tensor], h_D: bytes | None,
         build_model: BuildModel | None,
         verifier: Callable[[Section | None], Verifier] | None) -> StoreRun:
    rec = RecordingHandoff(handoff)
    made: list[Verifier] = []
    chain: dict[int, tuple[bytes, ...]] = {}

    def make(section: Section | None) -> Verifier:
        v = (verifier(section) if verifier is not None else
             Verifier(c, h_D=h_D if h_D is not None else dataset_tree(c, dataset).root,
                      n_records=len(dataset), k=k, n_steps=T, bands=Bands.provisional(),
                      w0=w0, allow_provisional=True, section=section))
        made.append(v)
        return v

    def on_step(s: StepRecord) -> None:
        if s.rejection is None:
            # The W_{t+1} hashes the verifier kept from its own check-2 recomputation.
            chain[s.t] = tuple(made[-1]._chain)  # pyright: ignore[reportPrivateUsage]

    r = run_scenario(c, dataset, w0, scenario, T=T, k=k, final=final, on_step=on_step,
                     h_D=h_D, build_model=build_model, verifier=make, handoff=rec)
    d = decide(c, r, rec.roots, chain)
    r.loop.w_final = None  # hashed into d.final; dropped so ten runs don't hold ten copies
    return StoreRun(name, r, d, rec.write_s, rec.disk_bytes)


def crosscheck(c: DeclaredComputation, dataset: Sequence[Any], w0: Mapping[str, torch.Tensor],
               scenarios: Sequence[Scenario], directory: str | Path, *, T: int, k: int,
               final: Mapping[str, torch.Tensor], h_D: bytes | None = None,
               build_model: BuildModel | None = None,
               verifier: Callable[[Section | None], Verifier] | None = None,
               out: Callable[[str], None] | None = print) -> list[CrossCheck]:
    """Run each scenario in memory, then from disk under ``directory/<scenario>/``, and compare.

    ``verifier`` builds each run's verifier from the metrics seam (default: provisional bands,
    as in ``run_scenario``). Each disk step's directory is deleted once it is verified, and the
    scenario's directory after its run, also on error."""
    results: list[CrossCheck] = []
    root = Path(directory)
    for s in scenarios:
        args: dict[str, Any] = dict(T=T, k=k, final=final, h_D=h_D, build_model=build_model,
                                    verifier=verifier)
        mem = _run(c, dataset, w0, s, "memory", IN_MEMORY, **args)
        sdir = root / s.name
        try:
            disk = _run(c, dataset, w0, s, "disk", DiskHandoff(sdir), **args)
        finally:
            shutil.rmtree(sdir, ignore_errors=True)
        cc = CrossCheck(s, mem, disk, compare(mem.decisions, disk.decisions))
        results.append(cc)
        if out is not None:
            report(cc, out)
    return results


def _sum(xs: Mapping[int, float]) -> float:
    return math.fsum(xs.values())


def report(cc: CrossCheck, out: Callable[[str], None] = print) -> None:
    m, d = cc.memory, cc.disk
    out(f"== {cc.scenario.name}: {cc.scenario.description}")
    out(f"   expected {cc.scenario.expected}; in memory {m.result.actual}"
        f" ({'PASS' if m.result.passed else 'FAIL'}), from disk {d.result.actual}"
        f" ({'PASS' if d.result.passed else 'FAIL'})")
    for x in (m, d):
        st = x.result.loop.steps
        out(f"   {x.store:>6}: {len(st)} steps; hand-off {_sum(x.write_s):.2f} s, "
            f"verify {math.fsum(s.verify_s for s in st):.2f} s"
            + (f", {max(x.disk_bytes.values()) / 2**30:.2f} GiB per step on disk"
               if x.disk_bytes else ""))
    if cc.identical:
        out(f"   decisions identical: verdict, {len(m.decisions.steps)} steps' rejections, "
            f"roots, W_(t+1) hashes, every check-5/6 number, final weights")
    else:
        out("   DECISIONS DIFFER: " + "; ".join(cc.diffs))


def report_memory(rows: Sequence[Mapping[str, Any]], out: Callable[[str], None] = print) -> None:
    """Each step's prover and verifier peak from a memory pass, and the rise over the
    section's start."""
    for row in rows:
        if row.get("level") == "total" and row["component"] in ("prover", "verifier") \
                and row["step"] > 0 and row["peak_bytes"] is not None:
            start = row["start_bytes"]
            out(f"   step {row['step']}: {row['component']} peak "
                f"{row['peak_bytes'] / 2**30:.2f} GiB"
                + ("" if start is None else
                   f" (start {start / 2**30:.2f}, rise {(row['peak_bytes'] - start) / 2**30:.2f})")
                + f" [{row['mem_source']}]")


# ---- the SmolLM2 scenarios ----------------------------------------------------------------


class OtherBatch(ProverFault):
    """A1: step ``t`` commits (and trains on) ``batch``; the audit paths stay ``π(t)``'s."""

    def __init__(self, t: int, batch: Sequence[Any]) -> None:
        self.t, self.batch = t, list(batch)

    def committed_records(self, t: int, records: Sequence[Any]) -> Sequence[Any]:
        return self.batch if t == self.t else records


def llama_scenarios(c: Any, dataset: Sequence[Any], T: int) -> list[Scenario]:
    """The honest run of ``T`` steps, then four faults at step 2 that each reject at a
    different check: 4 (another batch committed), 7 (a hidden step before step 2), 6a (one
    ``W_{t+1}`` entry moved), 5 (the largest entry of ``Λ`` sign-flipped). The fault classes
    are ``mlp_smoke``'s, which work on any ``C``."""
    from verification.runs.mlp_smoke import FlipProduct, HiddenStep, NudgeWNext
    other = list(dataset[-c.n_s:])  # no early step of π reaches the last records of D
    lam = c.m_of("Lambda")
    w_q = c.weight_names[2]  # layer 0's W_q
    out = [HONEST]
    if T >= FAULT_STEP:
        t = FAULT_STEP
        out += [
            Scenario("other-batch", f"step {t} commits the last {c.n_s} records of D",
                     OtherBatch(t, other), Expected(t, "4", "failed")),
            Scenario("broken-chain", f"one hidden step before step {t}, on the last "
                                     f"{c.n_s} records of D", HiddenStep(c, t, other),
                     Expected(t, "7", "failed")),
            Scenario("bad-w-next-ulps", f"entry 5 of W_(t+1)[{w_q}] moved {ULPS} ulps at "
                                        f"step {t}", NudgeWNext(t, w_q, 5, ULPS),
                     Expected(t, "6a", "failed")),
            Scenario("flip", f"largest entry of P_{lam} (Λ) sign-flipped at step {t}",
                     FlipProduct(t, lam), Expected(t, "5", "failed")),
        ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from verification.parameters import load_protocol_config
    from verification.computation.instances.llama import LlamaComputation
    from verification.runs.calibrate import judging_verifier
    from verification.runs.llama_step import load_committed_dataset
    from verification.verifier.calibration import load_bands

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=None, help="T (default: VERIF_STEPS)")
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                   help="also run a memory pass per store (default: VERIF_METRICS)")
    args = p.parse_args(argv)
    cfg = load_config()
    pc = load_protocol_config()
    T, k = (args.steps if args.steps is not None else cfg.steps), pc.k
    use_metrics = cfg.metrics if args.metrics is None else args.metrics
    if T < 1:
        p.error(f"T = {T}: at least one step")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    bands = load_bands(pc.band_file, k=k)  # read-only (P10a)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    final = honest_final(c, D, w0, T, build_model=models)
    make = judging_verifier(c, len(D), w0, bands, T=T, k=k, h_D=h_D)
    transcripts = cfg.output_dir / RUN_NAME / "transcripts"
    print(f"C3 store cross-check: {cfg.model}@{cfg.model_revision[:12]}, k {k}, T {T}, "
          f"M {c.M}, {c.n_leaves} leaves; band file {pc.band_file} ({bands.source[:16]}…); "
          f"disk transcripts under {transcripts}")
    results = crosscheck(c, D, w0, llama_scenarios(c, D, T), transcripts, T=T, k=k,
                         final=final, h_D=h_D, build_model=models, verifier=make)
    ok = all(r.identical and r.memory.result.passed and r.disk.result.passed for r in results)

    if use_metrics:
        S = min(PASS_STEPS, T)
        final_S = final if S == T else honest_final(c, D, w0, S, build_model=models)
        make_S = judging_verifier(c, len(D), w0, bands, T=S, k=k, h_D=h_D)
        for name, handoff in (("memory", IN_MEMORY), ("disk", DiskHandoff(transcripts / "mem"))):
            try:
                rows = memory_run(c, D, w0, run=RUN_NAME, T=S, k=k, final=final_S, h_D=h_D,
                                  device=cfg.device, build_model=models, verifier=make_S,
                                  handoff=handoff)
            finally:
                shutil.rmtree(transcripts / "mem", ignore_errors=True)
            print(f"-- memory pass, {S} honest steps, {name} store:")
            report_memory(rows)
    if transcripts.exists() and not any(transcripts.iterdir()):
        transcripts.rmdir()
    print("C3: " + ("every decision identical in memory and from disk" if ok
                    else "FAILED: see the scenarios above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
