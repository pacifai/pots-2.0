"""FS13: the full-scale run driver. One model and one attack (a cell), in the spec's order.

    .venv/bin/python -m verification.runs.full_scale run [--seeds S ...] [--levels N ...]
        [--hidden-level N] [--sweep-args ARGS] [--k-args ARGS] [--guard-ack]
        [--plant h3-mismatch | guard-trip]
    .venv/bin/python -m verification.runs.full_scale cheats --levels N ...   (one seed)
    .venv/bin/python -m verification.runs.full_scale hidden --level N        (one seed)

The cell is ``$VERIF_OUTPUT_DIR``; the model, task, batch and the rest come from the usual
environment variables, as for every run (scale is configuration only). ``run`` wraps the
existing runners, each started as its own process (``python -m verification.runs.<runner>``)
with ``VERIF_SEED`` and ``VERIF_POISON_COUNT`` set per stage, in EVALUATION_SPEC §3.2's order:

1. **Data (H0).** ``materialize_data`` writes the honest runs' ``D`` to ``data/`` (at the
   first seed, level 1), and every cheat seed's and level's ``D̃`` to its own directory,
   ``data/levels/s<seed>-c<count>/`` (``--data-dir``). ``D̃`` is tied to one seed, task, level
   and batch size (``cheats.check_meta``), so each needs its own. A level the task's poisoned
   list can't fill (``LevelNotRun``, exit 3) leaves ``level_not_run.txt`` there and is skipped.
2. **H1 calibrates.** ``calibrate`` fits the band file on steps 1–3 at the first seed and
   writes ``$VERIF_OUTPUT_DIR/bands.json``, the only band file of the cell: every later run
   reads it (F9, EVALUATION_SPEC §3.2), and no other stage writes one.
3. **H1.** ``run_verified`` at the first seed: steps 4–10 judged, step 4's transcript kept.
4. **H3.** ``plain_baseline`` at the first seed, then F5c: H1's final weights must equal H3's
   bit for bit, provenance first. A mismatch stops the cell (exit 4) before any later run. An
   H1 rejected for any reason but a guard trip stops it too (exit 1): it has no final weights,
   so F5c can't hold.
5. **The cancellation guard (F9).** If a judged step of H1 trips check 5's κ test (``κ/κ_max``
   above 1 in its ``residual_summary`` record), the driver pauses before H2 (exit 5). The user
   may revise ``κ_max``'s factor of 2 and have the band file rewritten; ``--guard-ack`` then
   resumes. The driver rewrites no band file and changes no factor itself. If H1 was rejected
   at the trip, F5c is reported unchecked and the run goes on.
6. **H2.** ``run_verified`` at every other seed: step 1 verified, not judged.
7. **The cheats.** One process per seed (``full_scale cheats``): the model, ``D``, ``W_0`` and
   the band file are loaded once, then for each level A1, A2 and A3 run on that level's ``D̃``
   (``poisoned_step.run_poisoned``: they share the poisoned step, S6e).
8. **The hidden-step run** (``full_scale hidden``) at the first seed and ``--hidden-level``
   (25% of the batch by default).
9. **The sweep** (``flipped_matmul_sweep``, EV3) and **``k`` tunability**
   (``k_tunability``, FS12), both on H1's stored step 4. ``--sweep-args`` and ``--k-args``
   pass options through (a reduced run); neither may name ``--subdir``, ``--source`` or
   ``--run-dir``. Without a stored step 4 (H1 stopped earlier) both are skipped.

The smooth check stays in FS11; this driver doesn't run it.

**Resume.** A stage whose run directory holds a ``run_end`` record (``metrics.MetricsWriter``
writes it only for a run that completed) is skipped, so a killed driver reruns only what it
hadn't finished. A run directory without one is deleted before its stage reruns (printed).
``calibrate`` is done when its ``calibrate/k<k>/`` holds ``run_end`` and the band file exists;
a data directory when ``meta.json`` (written last) or the level-not-run marker exists. A cheat
level is done when all three of its run directories are; otherwise all three rerun, since A3
splices A1's ``W_{t+1}``.

**After the runs** every ``records.jsonl`` in the cell must name either no band file or
``bands.json``'s hash (P10a over the cell). Each stage's outcome and wall time goes to
``full_scale_stages.jsonl``.

**Exit codes.** 0: every stage finished and every cheat was rejected at its declared point.
1: a stage failed (a crash, a cheat not rejected as declared, a sweep or ``k`` run that
failed, a calibration that exited 1 or 2). 4: F5c's mismatch stopped the cell. 5: paused
before H2 for F9's guard. (3 is ``materialize_data``'s level-not-run code, handled here.)

**Rehearsal plants** (``--plant``), for testing the two stops without changing any check's
numbers: ``h3-mismatch`` replaces one of H3's weight hashes in memory before F5c's comparison
(no file changes), so F5c fails; ``guard-trip`` lowers the driver's own trip line from 1 to 0,
so H1's judged ``κ/κ_max`` (about 0.5) counts as a trip and the driver pauses before H2.

**Levels.** EQ5's levels are 0, 10, 25, 50, 75 and 100% of the batch, plus one record; level 0
is the honest run (H1 and H2), so the cheat levels start at one record:
:func:`eq5_levels` gives ``1, 13, 32, 64, 96, 128`` at ``n_s = 128`` and ``1, 2, 3, 4`` at
``n_s = 4``.

**AdvBench at full scale** has one whole batch per pass (233 records, ``n_s = 128``), and
``data.other_batch`` needs two. None of this driver's runs uses ``other_batch``: A1, A2, A3 and
the hidden step take ``b̃`` from ``D̃``, and the sweep perturbs products. Only the protocol-fault
runs (``mlp_smoke``, ``store_crosscheck``) use it, and they aren't part of a cell.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from setup.config import load_config
from verification.parameters import load_protocol_config

__all__ = ["EXIT_FAILED", "EXIT_F5C", "EXIT_F9", "PLANTS", "STAGES_FILE", "LEVEL_NOT_RUN",
           "Cell", "eq5_levels", "hidden_level", "level_dir", "finished", "guard_trips",
           "f5c_mismatches", "run_cell", "main"]

EXIT_FAILED = 1
EXIT_LEVEL_NOT_RUN = 3  # materialize_data's code for a level the poisoned list can't fill
EXIT_F5C = 4
EXIT_F9 = 5
PLANTS = ("h3-mismatch", "guard-trip")
STAGES_FILE = "full_scale_stages.jsonl"
LEVEL_NOT_RUN = "level_not_run.txt"
RECORDS = "records.jsonl"
WEIGHTS_FILE = "final_weights.json"
TRIP_LINE = 1.0  # κ/κ_max above it: check 5's κ test rejects (F9's trip)
EQ5_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.0)
HIDDEN_FRACTION = 0.25  # the hidden-step run's BPR (EVALUATION_SPEC §3.2)
SCENARIOS = ("A1", "A2", "A3")
HIDDEN = "hidden"
RESERVED_SWEEP_ARGS = ("--subdir", "--source", "--run-dir")

Runner = Callable[[str, Sequence[str], Mapping[str, str]], int]


def eq5_levels(batch: int) -> list[int]:
    """EQ5's non-zero levels at ``batch`` records: one record, then 10, 25, 50, 75 and 100% of
    the batch rounded half up (at least one), without repeats, ascending."""
    return sorted({1} | {max(1, math.floor(f * batch + 0.5)) for f in EQ5_FRACTIONS})


def hidden_level(batch: int) -> int:
    """The hidden-step run's level: 25% of the batch, rounded half up, at least one record."""
    return max(1, math.floor(HIDDEN_FRACTION * batch + 0.5))


def level_dir(cell: Path, seed: int, count: int) -> Path:
    """The data directory of one cheat seed and level."""
    return cell / "data" / "levels" / f"s{seed}-c{count}"


def _records(run_dir: Path) -> list[dict[str, Any]]:
    """``records.jsonl``'s parsed lines; a line that doesn't parse (a run killed mid-write) is
    skipped."""
    path = run_dir / RECORDS
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def finished(run_dir: Path) -> bool:
    """Whether the run directory holds a ``run_end`` record: the run completed."""
    return any(r.get("record") == "run_end" for r in _records(run_dir))


def guard_trips(run_dir: Path, line: float = TRIP_LINE) -> list[tuple[int, str, float]]:
    """``(step, class, κ/κ_max)`` for every judged step of the honest run whose largest
    ``κ/κ_max`` is above ``line`` (F9's trip at the default line 1)."""
    out = []
    for r in _records(run_dir):
        if r.get("record") == "residual_summary" and r.get("judged"):
            x = r.get("kappa_ratio_max")
            if isinstance(x, (int, float)) and x > line:
                out.append((int(r["step"]), str(r.get("kappa_ratio_max_class")), float(x)))
    return out


def f5c_mismatches(h1: Mapping[str, Any], h3: Mapping[str, Any]) -> list[str]:
    """F5c: what differs between H1's and H3's ``final_weights.json``: the provenance
    (``plain_baseline.provenance_mismatches``), then each tensor whose hash differs."""
    from verification.runs.plain_baseline import provenance_mismatches, weight_mismatches

    out = provenance_mismatches(h3, h1)
    bad = weight_mismatches(h3.get("hashes") or {}, h1.get("hashes") or {})
    if bad:
        out.append(f"{len(bad)} weight tensor(s) differ, first {bad[0]}")
    if not (h1.get("hashes") or {}):
        out.append("H1 wrote no final weights (rejected)")
    return out


def _verdict(run_dir: Path) -> dict[str, Any] | None:
    rows = [r for r in _records(run_dir) if r.get("record") == "verdict"]
    return rows[-1] if rows else None


def _band_hash(path: Path) -> str:
    import blake3

    return blake3.blake3(path.read_bytes()).hexdigest()


def _subprocess_runner(module: str, args: Sequence[str], env: Mapping[str, str]) -> int:
    return subprocess.run([sys.executable, "-m", module, *args], env=dict(env)).returncode


@dataclass
class Cell:
    """One model and one attack: the driver's settings and what it found."""

    cell: Path
    model: str
    attack: str
    batch: int
    k: int
    seeds: list[int]
    levels: list[int]
    hidden: int
    sweep_args: list[str] = field(default_factory=list)
    k_args: list[str] = field(default_factory=list)
    guard_ack: bool = False
    plant: str | None = None
    runner: Runner = _subprocess_runner
    base_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    out: Callable[[str], None] = print
    failures: list[str] = field(default_factory=list)

    @property
    def h1_seed(self) -> int:
        return self.seeds[0]

    def name(self, scenario: str, bpr: int, seed: int) -> str:
        from verification.runs.metrics import run_name

        return run_name(scenario, self.model, self.attack, bpr, seed)

    def honest(self, seed: int) -> Path:
        return self.cell / self.name("honest", 0, seed)

    def env(self, seed: int, count: int = 1) -> dict[str, str]:
        return {**self.base_env, "VERIF_OUTPUT_DIR": str(self.cell), "VERIF_SEED": str(seed),
                "VERIF_POISON_COUNT": str(count)}

    def log(self, stage: str, status: str, seconds: float = 0.0, **extra: Any) -> None:
        row = {"stage": stage, "status": status, "seconds": round(seconds, 3),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        with (self.cell / STAGES_FILE).open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.out(f"[{stage}] {status}" + (f" in {seconds:.1f} s" if seconds else "")
                 + "".join(f", {k} {v}" for k, v in extra.items()))

    def launch(self, stage: str, module: str, args: Sequence[str], env: Mapping[str, str],
               ) -> tuple[int, float]:
        self.out(f"[{stage}] python -m {module} {shlex.join(args)}  (VERIF_SEED "
                 f"{env['VERIF_SEED']}, VERIF_POISON_COUNT {env['VERIF_POISON_COUNT']})")
        t0 = time.perf_counter()
        code = self.runner(module, list(args), env)
        return code, time.perf_counter() - t0

    def run_stage(self, stage: str, run_dir: Path, module: str, args: Sequence[str],
                  env: Mapping[str, str], *, done: Callable[[], bool] | None = None,
                  ok_codes: Sequence[int] = (0,)) -> int | None:
        """Run one stage unless it's done. Returns the child's exit code, or ``None`` when
        skipped. A child that ends without completing its run (no ``run_end``) raises."""
        is_done = done or (lambda: finished(run_dir))
        if is_done():
            self.log(stage, "done before, skipped")
            return None
        if run_dir.exists():
            self.out(f"[{stage}] deleting the unfinished run directory {run_dir}")
            shutil.rmtree(run_dir)
        code, s = self.launch(stage, module, args, env)
        if not finished(run_dir):
            self.log(stage, "crashed", s, exit=code)
            raise StageError(f"{stage}: {module} exited {code} without completing its run "
                             f"(no run_end in {run_dir})")
        self.log(stage, "ran" if code in ok_codes else "ran, failed", s, exit=code)
        return code


class StageError(RuntimeError):
    """A stage that couldn't complete: the cell stops with exit 1."""


class Stop(Exception):
    def __init__(self, code: int, why: str):
        super().__init__(why)
        self.code = code


# ---- the stages ---------------------------------------------------------------------------


def _materialize(cell: Cell, stage: str, data_dir: Path, seed: int, count: int,
                 default: bool = False) -> bool:
    """One data directory; ``False`` when the level doesn't run (EQ5)."""
    if (data_dir / "meta.json").exists():
        meta = json.loads((data_dir / "meta.json").read_text())
        p = meta.get("poisoning") or {}
        want = {"seed": seed, "attack": cell.attack, "count": count, "batch": cell.batch}
        got = {"seed": meta.get("seed"), "attack": meta.get("attack"), "count": p.get("count"),
               "batch": p.get("batch")}
        if got != want:
            raise StageError(f"{data_dir} holds data for {got}, this cell needs {want}: use "
                             "another VERIF_OUTPUT_DIR or delete it")
        cell.log(stage, "done before, skipped")
        return True
    if (data_dir / LEVEL_NOT_RUN).exists():
        cell.log(stage, "level not run (EQ5), skipped")
        return False
    args = [] if default else ["--data-dir", str(data_dir)]
    code, s = cell.launch(stage, "verification.runs.materialize_data", args,
                          cell.env(seed, count))
    if code == EXIT_LEVEL_NOT_RUN:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / LEVEL_NOT_RUN).write_text(
            f"level {count} at seed {seed}: the task's poisoned list can't fill it (EQ5)\n")
        cell.log(stage, "level not run (EQ5)", s, exit=code)
        return False
    if code != 0 or not (data_dir / "meta.json").exists():
        cell.log(stage, "failed", s, exit=code)
        raise StageError(f"{stage}: materialize_data exited {code}")
    cell.log(stage, "ran", s)
    return True


def stage_data(cell: Cell) -> dict[tuple[int, int], bool]:
    """H0: the honest runs' data, then every cheat seed's and level's, and the hidden run's."""
    _materialize(cell, "H0 data", cell.cell / "data", cell.h1_seed, 1, default=True)
    runs: dict[tuple[int, int], bool] = {}
    wanted = [(s, n) for s in cell.seeds for n in cell.levels] + [(cell.h1_seed, cell.hidden)]
    for s, n in wanted:
        if (s, n) not in runs:
            runs[s, n] = _materialize(cell, f"H0 D̃ seed {s} level {n}",
                                      level_dir(cell.cell, s, n), s, n)
    return runs


def stage_calibrate(cell: Cell) -> None:
    band = cell.cell / "bands.json"
    run_dir = cell.cell / "calibrate" / f"k{cell.k}"
    code = cell.run_stage("H1 calibrate", run_dir, "verification.runs.calibrate", ["--metrics"],
                          cell.env(cell.h1_seed),
                          done=lambda: finished(run_dir) and band.exists())
    if code is not None and (code != 0 or not band.exists()):
        raise Stop(EXIT_FAILED, f"calibration exited {code} and wrote no band file (1: an "
                                "honest step rejected; 2: the concentration guard failed or "
                                "the recomputed k exceeds VERIF_K)")


def _verified_done(run_dir: Path) -> Callable[[], bool]:
    return lambda: finished(run_dir) and (run_dir / WEIGHTS_FILE).exists()


def stage_h1_h3(cell: Cell) -> None:
    h1 = cell.honest(cell.h1_seed)
    cell.run_stage("H1", h1, "verification.runs.run_verified", ["--metrics"],
                   cell.env(cell.h1_seed), done=_verified_done(h1), ok_codes=(0, 1))
    h3 = cell.cell / "plain_baseline"
    cell.run_stage("H3", h3, "verification.runs.plain_baseline", ["--metrics"],
                   cell.env(cell.h1_seed), done=_verified_done(h3))


def gate_f5c_f9(cell: Cell) -> None:
    """F5c, then F9, between H3 and H2."""
    from verification.runs.plain_baseline import read_final_weights

    h1_dir = cell.honest(cell.h1_seed)
    verdict = _verdict(h1_dir) or {}
    accepted = bool(verdict.get("accepted"))
    line = 0.0 if cell.plant == "guard-trip" else TRIP_LINE
    trips = guard_trips(h1_dir, line)
    if cell.plant == "guard-trip":
        cell.out(f"PLANTED: the trip line is {line}, not {TRIP_LINE} (rehearsal)")
    if accepted:
        h1 = read_final_weights(h1_dir)
        h3 = read_final_weights(cell.cell / "plain_baseline")
        if cell.plant == "h3-mismatch":
            name = next(iter(h3["hashes"]))
            h3 = {**h3, "hashes": {**h3["hashes"], name: "00" * 32}}
            cell.out(f"PLANTED: H3's hash of {name} replaced in memory (rehearsal)")
        bad = f5c_mismatches(h1, h3)
        if bad:
            cell.log("F5c", "MISMATCH", why="; ".join(bad))
            raise Stop(EXIT_F5C, "F5c: H1's final weights differ from H3's: " + "; ".join(bad)
                       + ". Every later run is stopped; find and fix the cause first")
        cell.log("F5c", "H1 and H3 final weights bit-identical", root=h1.get("root"))
    if trips:
        what = ", ".join(f"step {t} {c} κ/κ_max {x:.3f}" for t, c, x in trips)
        if not cell.guard_ack:
            cell.log("F9", "PAUSED before H2", trips=what)
            raise Stop(EXIT_F9, f"F9: H1's judged steps trip the cancellation guard ({what}). "
                                "Revise κ_max's factor if wanted (the band file is then "
                                "rewritten once), then rerun with --guard-ack")
        cell.log("F9", "guard trip acknowledged (--guard-ack)", trips=what)
    elif accepted:
        cell.log("F9", "no judged H1 step trips the guard")
    if not accepted:
        where = f"step {verdict.get('rejection_step')} check {verdict.get('rejection_check')}"
        if trips and cell.guard_ack:
            cell.log("F5c", f"UNCHECKED: H1 rejected at {where}, so it has no final weights")
            return
        raise Stop(EXIT_FAILED, f"H1 was rejected at {where} (verdict {verdict or 'missing'}): "
                                "F5c can't hold without its final weights")


def stage_h2(cell: Cell) -> None:
    for s in cell.seeds[1:]:
        d = cell.honest(s)
        cell.run_stage(f"H2 seed {s}", d, "verification.runs.run_verified", ["--metrics"],
                       cell.env(s), done=_verified_done(d), ok_codes=(0, 1))


def _oracle_failures(run_dir: Path) -> list[str]:
    rows = [r for r in _records(run_dir) if r.get("record") == "cheat_oracle"]
    if not rows:
        return [f"{run_dir.name}: no cheat_oracle record"]
    return [f"{run_dir.name}: {r.get('outcome')}" for r in rows if not r.get("passed")]


def stage_cheats(cell: Cell, data: Mapping[tuple[int, int], bool]) -> None:
    for s in cell.seeds:
        levels = [n for n in cell.levels if data.get((s, n))]
        dirs = {n: [cell.cell / cell.name(x, n, s) for x in SCENARIOS] for n in levels}
        todo = [n for n in levels if not all(finished(d) for d in dirs[n])]
        if not todo:
            cell.log(f"cheats seed {s}", "done before, skipped", levels=levels)
        else:
            args = ["cheats", "--levels", *map(str, todo)]
            code, sec = cell.launch(f"cheats seed {s}", "verification.runs.full_scale", args,
                                    cell.env(s))
            missing = [d.name for n in todo for d in dirs[n] if not finished(d)]
            if missing:
                cell.log(f"cheats seed {s}", "crashed", sec, exit=code)
                raise StageError(f"cheats seed {s} exited {code}; unfinished: {missing}")
            cell.log(f"cheats seed {s}", "ran" if code == 0 else "ran, failed", sec,
                     exit=code, levels=todo)
        for n in levels:
            for d in dirs[n]:
                cell.failures += _oracle_failures(d)


def stage_hidden(cell: Cell, data: Mapping[tuple[int, int], bool]) -> None:
    s, n = cell.h1_seed, cell.hidden
    if not data.get((s, n)):
        cell.log("hidden", f"skipped: level {n} doesn't run")
        return
    d = cell.cell / cell.name(HIDDEN, n, s)
    cell.run_stage("hidden", d, "verification.runs.full_scale", ["hidden", "--level", str(n)],
                   cell.env(s), ok_codes=(0, 1))
    cell.failures += _oracle_failures(d)


def stage_sweep_k(cell: Cell) -> None:
    from verification.runs.run_verified import KEEP_STEP, TRANSCRIPTS_DIR

    kept = cell.honest(cell.h1_seed) / TRANSCRIPTS_DIR / f"step_{KEEP_STEP}" / "meta.json"
    if not kept.exists():
        cell.log("sweep", f"skipped: H1 stored no step {KEEP_STEP}")
        cell.log("k tunability", f"skipped: H1 stored no step {KEEP_STEP}")
        return
    for stage, module, run, extra in (
            ("sweep", "verification.runs.flipped_matmul_sweep", "flipped_matmul_sweep",
             cell.sweep_args),
            ("k tunability", "verification.runs.k_tunability", "k_tunability", cell.k_args)):
        code = cell.run_stage(stage, cell.cell / run, module, ["--metrics", *extra],
                              cell.env(cell.h1_seed), ok_codes=(0, 1))
        if code not in (None, 0):
            cell.failures.append(f"{stage}: exit {code}")


def check_band_hashes(cell: Cell) -> int:
    """P10a over the cell: every run's ``environment`` record names no band file or
    ``bands.json``'s hash, and so does the calibration record. Returns the runs checked."""
    want = _band_hash(cell.cell / "bands.json")
    bad, n = [], 0
    for path in sorted(cell.cell.rglob(RECORDS)):
        n += 1
        for r in _records(path.parent):
            h = (r.get("band_file_hash") if r.get("record") in ("environment", "calibration")
                 else None)
            if h is not None and h != want:
                bad.append(f"{path.parent.relative_to(cell.cell)}: {r['record']} {h[:16]}…")
    if bad:
        raise StageError(f"P10a: band-file hashes other than bands.json's {want[:16]}…: {bad}")
    cell.log("P10a", f"one band-file hash across {n} run directories", hash=want[:16])
    return n


def run_cell(cell: Cell) -> int:
    """The whole cell in order; the exit code (module docstring)."""
    cell.cell.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cell.out(f"FS13 cell {cell.cell}: {cell.model}, {cell.attack}, n_s {cell.batch}, k "
             f"{cell.k}, seeds {cell.seeds}, levels {cell.levels}, hidden level {cell.hidden}"
             + (f", PLANT {cell.plant}" if cell.plant else ""))
    try:
        data = stage_data(cell)
        stage_calibrate(cell)
        stage_h1_h3(cell)
        gate_f5c_f9(cell)
        stage_h2(cell)
        stage_cheats(cell, data)
        stage_hidden(cell, data)
        stage_sweep_k(cell)
        check_band_hashes(cell)
    except Stop as e:
        cell.out(f"STOPPED (exit {e.code}): {e}")
        return e.code
    except StageError as e:
        cell.log("cell", "FAILED", why=str(e))
        cell.out(f"FAILED: {e}")
        return EXIT_FAILED
    s = time.perf_counter() - t0
    if cell.failures:
        cell.log("cell", "finished with failures", s, failures=cell.failures)
        return EXIT_FAILED
    cell.log("cell", "finished", s)
    return 0


# ---- the per-seed children ----------------------------------------------------------------


def run_cheats(levels: Sequence[int], *, hidden: bool = False) -> int:
    """One seed's cheats (``VERIF_SEED``): the model, ``D``, ``W_0`` and the band file are
    loaded once; each level's ``D̃`` comes from :func:`level_dir`. A1–A3 per level, or with
    ``hidden`` the hidden-step run at its one level. A level whose run directories all hold
    ``run_end`` is skipped."""
    from verification.computation.instances.decoder import DecoderComputation
    from verification.computation.interface import snapshot_weights
    from setup.config import setup_determinism
    from verification.runs.cheats import (
        band_sources,
        cell_name,
        load_level,
        open_writer,
        report_oracle,
    )
    from verification.runs.hidden_steps import run_hidden
    from verification.runs.llama_step import load_committed_dataset
    from verification.runs.metrics import MetricsWriter, run_name
    from verification.runs.poisoned_step import run_poisoned
    from verification.runs.scenarios import ReusedModel
    from verification.verifier.calibration import band_context, load_bands

    cfg = load_config()
    pc = load_protocol_config()
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    bands = load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D, k=pc.k))
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    scenarios = (HIDDEN,) if hidden else SCENARIOS
    ok = True
    for n in levels:
        dirs = [cfg.output_dir / run_name(x, cfg.model, cfg.attack, n, cfg.seed)
                for x in scenarios]
        if all(finished(d) for d in dirs):
            print(f"level {n}: done before, skipped")
            continue
        for d in dirs:
            if d.exists():
                print(f"level {n}: deleting the unfinished run directory {d}")
                shutil.rmtree(d)
        lcfg = dataclasses.replace(cfg, poison_count=n)
        env = load_level(lcfg, level_dir(cfg.output_dir, cfg.seed, n), c=c, D=D, h_D=h_D,
                         w0=w0, bands=bands, k=pc.k, models=models)
        if env.bpr_records != n:
            raise RuntimeError(f"level {n}: b̃ differs from π's batch in {env.bpr_records} "
                               "records")
        print(f"FS13 {'hidden step' if hidden else 'A1–A3'}: seed {cfg.seed}, level {n} "
              f"({env.bpr_actual:g}), step {env.poison_step}, band file "
              f"{env.bands.source[:16]}…")

        def writer(scenario: str) -> MetricsWriter:
            return open_writer(lcfg, cell_name(lcfg, scenario, env), env)

        if hidden:
            mw = writer(HIDDEN)
            try:
                results = [run_hidden(env, writer=mw)]
            finally:
                mw.close()
        else:
            results = run_poisoned(env, writers=writer)
        report_oracle(results, env.bands)
        band_sources(env, results, cfg.output_dir)
        ok = ok and all(r.passed for r in results)
    return 0 if ok else 1


# ---- command line -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the whole cell, resuming where it stopped")
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="the first is H1's (calibration, sweep, k, hidden run); the others H2's")
    r.add_argument("--levels", type=int, nargs="+", default=None,
                   help="the cheats' poisoning levels in records (default EQ5's at VERIF_BATCH)")
    r.add_argument("--hidden-level", type=int, default=None,
                   help="the hidden-step run's level (default 25%% of VERIF_BATCH)")
    r.add_argument("--sweep-args", default="", help="options for flipped_matmul_sweep")
    r.add_argument("--k-args", default="", help="options for k_tunability")
    r.add_argument("--guard-ack", action="store_true",
                   help="go on past F9's pause (H1's guard trip seen and decided)")
    r.add_argument("--plant", choices=PLANTS, default=None,
                   help="rehearse a stop (module docstring); changes no file or check")
    ch = sub.add_parser("cheats", help="A1–A3 at one seed (VERIF_SEED), several levels")
    ch.add_argument("--levels", type=int, nargs="+", required=True)
    hi = sub.add_parser("hidden", help="the hidden-step run at one seed (VERIF_SEED)")
    hi.add_argument("--level", type=int, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "cheats":
        return run_cheats(args.levels)
    if args.cmd == "hidden":
        return run_cheats([args.level], hidden=True)
    cfg = load_config()
    pc = load_protocol_config()
    sweep, kargs = shlex.split(args.sweep_args), shlex.split(args.k_args)
    if any(a.split("=")[0] in RESERVED_SWEEP_ARGS for a in sweep + kargs):
        p.error(f"--sweep-args and --k-args may not name {', '.join(RESERVED_SWEEP_ARGS)}")
    levels = sorted(set(args.levels)) if args.levels else eq5_levels(cfg.batch)
    hidden = args.hidden_level or hidden_level(cfg.batch)
    if not all(1 <= n <= cfg.batch for n in [*levels, hidden]):
        p.error(f"levels must lie in 1…{cfg.batch} (level 0 is the honest run)")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds repeats a seed")
    cell = Cell(cfg.output_dir.resolve(), cfg.model, cfg.attack, cfg.batch, pc.k,
                list(args.seeds), levels, hidden, sweep, kargs, args.guard_ack, args.plant)
    return run_cell(cell)


if __name__ == "__main__":
    sys.exit(main())
