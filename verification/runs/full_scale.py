"""FS13: the full-scale run driver. One model and one attack (a cell), in the spec's order.

    .venv/bin/python -m verification.runs.full_scale run [--seeds S ...] [--levels N ...]
        [--hidden-level N] [--sweep-args ARGS] [--k-args ARGS] [--guard-ack] [--data-only]
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
   An existing ``meta.json`` must name this cell's model, task, seed, level and batch.
   ``--data-only`` stops here (exit 0): the data is built on CPU before any GPU session.
2. **H1 calibrates.** ``calibrate`` fits the band file on steps 1–3 at the first seed and
   writes ``$VERIF_OUTPUT_DIR/bands.json``, the only band file of the cell: every later run
   reads it (F9, EVALUATION_SPEC §3.2), and no other stage writes one. Right after, every run
   directory already in the cell must name ``bands.json``'s hash or none (below).
3. **H1.** ``run_verified`` at the first seed: steps 4–10 judged, step 4's transcript kept.
4. **H3.** ``plain_baseline`` at the first seed, then F5c: H1's final weights must equal H3's
   bit for bit, provenance first. A mismatch stops the cell (exit 4) before any later run. When
   H1 was rejected it has no final weights, and F5c compares what both runs reached: the
   provenance and the losses of H1's steps (logged as a partial comparison).
5. **The cancellation guard (F9).** If a judged step of H1 trips check 5's κ test (``κ/κ_max``
   above 1 in its ``residual_summary`` record), the driver pauses before H2 (exit 5). F9 lets
   the user revise ``κ_max``'s factor; this driver has no path for that and rewrites no band
   file. ``--guard-ack`` goes on with the band file as it is, and covers a rejected H1 only when
   it was rejected at check 5 at the first tripping step. Any other H1 rejection stops the cell
   (exit 1).
6. **H2.** ``run_verified`` at every other seed: step 1 verified, not judged. A rejected H2 is
   an honest false reject, a result (F9): it is logged as ``REJECTED (honest false reject)``
   with its seed, step and check, the cell goes on, and it ends with exit 6.
7. **The cheats.** One process per seed (``full_scale cheats``): the model, ``D``, ``W_0`` and
   the band file are loaded once, then for each level A1, A2 and A3 run on that level's ``D̃``
   (``poisoned_step.run_poisoned``: they share the poisoned step, S6e).
8. **The hidden-step run** (``full_scale hidden``) at the first seed and ``--hidden-level``
   (25% of the batch by default).
9. **The sweep** (``flipped_matmul_sweep``, EV3) and **``k`` tunability**
   (``k_tunability``, FS12), both on H1's stored step 4. ``--sweep-args`` and ``--k-args``
   pass options through (a reduced run); neither may name ``--subdir``, ``--source`` or
   ``--run-dir``, nor any prefix of them that argparse would expand. Without a stored step 4
   (H1 stopped earlier) both are skipped.

The smooth check stays in FS11; this driver doesn't run it.

**A changed band file.** After calibration, at the gate before H2 and after the last run, every
``records.jsonl`` in the cell must name ``bands.json``'s hash (its ``environment`` or
``calibration`` record) or none. Another hash means ``bands.json`` changed after those runs
(a revision under F9, say), and the cell stops (exit 7). Nothing else is automatic: which runs
to redo is the user's decision. After the last run every directory but ``plain_baseline`` must
also name a band file (P10a over the cell; exit 1).

**Exit codes of a child.** Exit 1 is a result only where a record says so: for H1, H2, the
hidden run and the sweep a ``verdict`` record with ``accepted: false``, and for the cheats and
the hidden run a ``cheat_oracle`` record in every run directory. Any other nonzero exit, and a
child that ends without its run's ``run_end`` record, fails the stage.

**Resume.** A stage whose run directory holds a ``run_end`` record (``metrics.MetricsWriter``
writes it only for a run that completed) is skipped, so a killed driver reruns only what it
hadn't finished. A run directory without one is deleted before its stage reruns (printed).
``calibrate`` is done when its ``calibrate/k<k>/`` holds ``run_end`` and the band file exists;
H1, H2 and H3 when their ``final_weights.json`` also parses; a data directory when ``meta.json``
(written last) or the level-not-run marker exists. A cheat level is done when all three of its
run directories are; otherwise all three rerun, since A3 splices A1's ``W_{t+1}``. A finished
run's verdict and oracle records are read again on every pass, so a resumed cell reports them.

**One driver per cell.** ``run`` holds ``full_scale.lock`` (its pid) in the cell. It refuses to
start while a live process holds the lock, and takes over a lock whose process is gone. On
SIGTERM or SIGHUP it terminates the running child, waits for it, logs the stage line and exits
128 + the signal; the child's run directory, without ``run_end``, reruns on resume. Each stage
line goes to ``full_scale_stages.jsonl``, fsynced; the runners' own records are written by
``metrics.MetricsWriter``, outside this driver.

**Exit codes.** 0: every stage finished and every cheat was rejected at its declared point.
1: a stage failed (a crash, an exit no record explains, a cheat not rejected as declared, a
sweep or ``k`` run that failed, a calibration that exited 1 or 2, an H1 rejected other than at
an acknowledged trip, a run naming no band file, another driver holding the cell). 4: F5c's
mismatch stopped the cell. 5: paused before H2 for F9's guard. 6: an H2 run was rejected
(honest false reject) and nothing else failed. 7: ``bands.json`` isn't the band file the
cell's runs name. 128 + n: stopped by signal n. (3 is ``materialize_data``'s level-not-run
code, handled here.)

**Rehearsal plants** (``--plant``), for testing the two stops without changing any check's
numbers: ``h3-mismatch`` replaces one of H3's weight hashes in memory before F5c's comparison
(no file changes), so F5c fails; ``guard-trip`` lowers the driver's own trip line from 1 to 0,
so H1's judged ``κ/κ_max`` (about 0.5) counts as a trip and the driver pauses before H2. Every
stage line written during a rehearsal carries ``"plant"``.

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
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from setup.config import load_config
from verification.parameters import load_protocol_config

__all__ = ["EXIT_FAILED", "EXIT_F5C", "EXIT_F9", "EXIT_FALSE_REJECT", "EXIT_BANDS", "PLANTS",
           "STAGES_FILE", "LOCK_FILE", "LEVEL_NOT_RUN", "Cell", "SubprocessRunner",
           "Interrupted", "eq5_levels", "hidden_level", "level_dir", "finished", "guard_trips",
           "f5c_mismatches", "f5c_partial", "reserved_args", "install_signal_handlers",
           "run_cell", "main"]

EXIT_FAILED = 1
EXIT_LEVEL_NOT_RUN = 3  # materialize_data's code for a level the poisoned list can't fill
EXIT_F5C = 4
EXIT_F9 = 5
EXIT_FALSE_REJECT = 6
EXIT_BANDS = 7
PLANTS = ("h3-mismatch", "guard-trip")
STAGES_FILE = "full_scale_stages.jsonl"
LOCK_FILE = "full_scale.lock"
LEVEL_NOT_RUN = "level_not_run.txt"
RECORDS = "records.jsonl"
WEIGHTS_FILE = "final_weights.json"
PLAIN_DIR = "plain_baseline"  # the one run that names no band file
TRIP_LINE = 1.0  # κ/κ_max above it: check 5's κ test rejects (F9's trip)
EQ5_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.0)
HIDDEN_FRACTION = 0.25  # the hidden-step run's BPR (EVALUATION_SPEC §3.2)
SCENARIOS = ("A1", "A2", "A3")
HIDDEN = "hidden"
RESERVED_SWEEP_ARGS = ("--subdir", "--source", "--run-dir")
STOP_GRACE_S = 60.0  # how long a terminated child gets before it is killed

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


def reserved_args(args: Sequence[str], reserved: Sequence[str] = RESERVED_SWEEP_ARGS
                  ) -> list[str]:
    """The options in ``args`` that name one of ``reserved``, also as a prefix argparse would
    expand (``--sub`` for ``--subdir``)."""
    out = []
    for a in args:
        name = a.split("=", 1)[0]
        if name.startswith("--") and len(name) > 2 and any(r.startswith(name) for r in reserved):
            out.append(a)
    return out


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
    ``κ/κ_max`` is above ``line`` (F9's trip at the default line 1), in step order."""
    out = []
    for r in _records(run_dir):
        if r.get("record") == "residual_summary" and r.get("judged"):
            x = r.get("kappa_ratio_max")
            if isinstance(x, (int, float)) and x > line:
                out.append((int(r["step"]), str(r.get("kappa_ratio_max_class")), float(x)))
    return sorted(out)


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


def f5c_partial(h1: Mapping[str, Any], h3: Mapping[str, Any]) -> list[str]:
    """F5c over the steps both runs reached, for an H1 rejected partway: the provenance but the
    step count, and the losses of H1's steps. There are no final weights to compare."""
    from verification.runs.plain_baseline import provenance_mismatches

    n = len(h1.get("losses") or [])
    return provenance_mismatches({**h3, "steps": h1.get("steps"),
                                  "losses": list(h3.get("losses") or [])[:n]}, h1)


def _verdict(run_dir: Path) -> dict[str, Any] | None:
    rows = [r for r in _records(run_dir) if r.get("record") == "verdict"]
    return rows[-1] if rows else None


def _rejected(run_dir: Path) -> bool:
    """Whether the run's verdict record says ``accepted: false``."""
    v = _verdict(run_dir)
    return v is not None and v.get("accepted") is False


def _has_oracle(run_dir: Path) -> bool:
    return any(r.get("record") == "cheat_oracle" for r in _records(run_dir))


def _band_hash(path: Path) -> str:
    import blake3

    return blake3.blake3(path.read_bytes()).hexdigest()


def _weights_ok(run_dir: Path) -> bool:
    """``final_weights.json`` exists and parses as a document with a ``hashes`` map."""
    try:
        doc = json.loads((run_dir / WEIGHTS_FILE).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("hashes"), dict)


def _write_synced(path: Path, text: str, mode: str = "w") -> None:
    with path.open(mode) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


# ---- the child processes, the lock and the signals ----------------------------------------


class Interrupted(BaseException):
    """SIGTERM or SIGHUP reached the driver. A ``BaseException``, so no stage's ``except
    Exception`` swallows it."""

    def __init__(self, signum: int):
        super().__init__(f"signal {signal.Signals(signum).name}")
        self.signum = signum


class SubprocessRunner:
    """Runs each stage as ``python -m <module>``. When the driver is interrupted while a child
    runs (a signal, Ctrl-C), the child is terminated and waited for before the exception goes
    on; one that outlives :data:`STOP_GRACE_S` is killed."""

    def __init__(self, prefix: Sequence[str] | None = None, grace: float = STOP_GRACE_S):
        self.prefix = list(prefix) if prefix is not None else [sys.executable, "-m"]
        self.grace = grace
        self.child: subprocess.Popen[bytes] | None = None

    def __call__(self, module: str, args: Sequence[str], env: Mapping[str, str]) -> int:
        self.child = subprocess.Popen([*self.prefix, module, *args], env=dict(env))
        try:
            return self.child.wait()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Terminate the running child and wait for it (kill it after the grace period). Not
        safe inside a signal handler: the interrupted ``wait`` holds the child's lock."""
        child = self.child
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(self.grace)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def install_signal_handlers() -> dict[int, Any]:
    """SIGTERM and SIGHUP raise :class:`Interrupted` in the main thread; the runner then stops
    its child and :func:`run_cell` logs the stop. Returns the handlers replaced."""
    def handler(signum: int, frame: Any) -> None:
        raise Interrupted(signum)

    return {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGHUP)}


class Busy(RuntimeError):
    """Another live driver holds the cell's lock."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock(cell: Path) -> Path:
    """Write this process's pid to the cell's lock file. Refuse (:class:`Busy`) while another
    live process holds it; take over a lock whose process is gone or that doesn't parse."""
    path = cell / LOCK_FILE
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(path.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                pid = None
            if pid is not None and pid != os.getpid() and _pid_alive(pid):
                raise Busy(f"process {pid} holds {path}: another driver runs this cell") from None
            path.unlink(missing_ok=True)  # stale: its process is gone
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
            f.flush()
            os.fsync(f.fileno())
        return path
    raise Busy(f"couldn't take {path}")


def release_lock(path: Path) -> None:
    try:
        if int(path.read_text().split()[0]) == os.getpid():
            path.unlink()
    except (OSError, ValueError, IndexError):
        pass


# ---- the cell -----------------------------------------------------------------------------


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
    revision: str | None = None
    data_only: bool = False
    runner: Runner = field(default_factory=SubprocessRunner)
    base_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    out: Callable[[str], None] = print
    failures: list[str] = field(default_factory=list)
    false_rejects: list[str] = field(default_factory=list)

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
        if self.plant:
            row["plant"] = self.plant
        _write_synced(self.cell / STAGES_FILE, json.dumps(row) + "\n", "a")
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
                  exit1: Callable[[], bool] | None = None, fatal: bool = True,
                  any_code: bool = False) -> int | None:
        """Run one stage unless it's done. Returns the child's exit code, or ``None`` when
        skipped. A child that ends without completing its run (no ``run_end``) raises. Exit 0
        is a run that finished; exit 1 is one only when ``exit1()`` says a record explains it
        (a rejecting verdict, an oracle record). Any other exit fails the stage: it raises
        with ``fatal``, else it joins the cell's failures. ``any_code`` leaves the code to the
        caller."""
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
        if any_code or code == 0 or (code == 1 and exit1 is not None and exit1()):
            self.log(stage, "ran", s, exit=code)
            return code
        why = f"{stage}: {module} exited {code}, and no record explains it"
        self.log(stage, "FAILED", s, exit=code, why=why)
        if fatal:
            raise StageError(why)
        self.failures.append(why)
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
        run = (meta.get("tokenizers") or {}).get("run") or {}
        want = {"model": cell.model, "seed": seed, "attack": cell.attack, "count": count,
                "batch": cell.batch}
        got = {"model": run.get("repo"), "seed": meta.get("seed"), "attack": meta.get("attack"),
               "count": p.get("count"), "batch": p.get("batch")}
        if cell.revision is not None:
            want["revision"], got["revision"] = cell.revision, run.get("revision")
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
        _write_synced(data_dir / LEVEL_NOT_RUN,
                      f"level {count} at seed {seed}: the task's poisoned list can't fill it "
                      "(EQ5)\n")
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


def _stray_band_hashes(cell: Cell) -> tuple[str, list[str], list[str]]:
    """``bands.json``'s hash; the run directories naming another (``environment`` or
    ``calibration`` record); and those naming none that should (every one but the plain
    baseline; a calibration run names it in its ``calibration`` record)."""
    want = _band_hash(cell.cell / "bands.json")
    other, unnamed = [], []
    for path in sorted(cell.cell.rglob(RECORDS)):
        rel = path.parent.relative_to(cell.cell)
        rows = [r for r in _records(path.parent)
                if r.get("record") in ("environment", "calibration")]
        for r in rows:
            h = r.get("band_file_hash")
            if h is not None and h != want:
                other.append(f"{rel}: {r['record']} {h[:16]}…")
        is_cal = any(r["record"] == "calibration" for r in rows)
        needed = [r for r in rows if r["record"] == ("calibration" if is_cal else "environment")]
        if str(rel) != PLAIN_DIR and (not needed or any(
                r.get("band_file_hash") is None for r in needed)):
            unnamed.append(str(rel))
    return want, other, unnamed


def check_band_file(cell: Cell, where: str) -> None:
    """Stop (exit 7) when a run in the cell names a band file other than ``bands.json``."""
    want, other, _ = _stray_band_hashes(cell)
    if other:
        cell.log("band file", "CHANGED", where=where, hash=want[:16], others=other)
        raise Stop(EXIT_BANDS, f"{where}: bands.json (hash {want[:16]}…) isn't the band file "
                               f"these runs name: {other}. It changed after they ran. This "
                               "driver has no path for a revised band file; nothing else is "
                               "automatic: decide by hand which runs to redo")


def stage_calibrate(cell: Cell) -> None:
    band = cell.cell / "bands.json"
    run_dir = cell.cell / "calibrate" / f"k{cell.k}"
    code = cell.run_stage("H1 calibrate", run_dir, "verification.runs.calibrate", ["--metrics"],
                          cell.env(cell.h1_seed),
                          done=lambda: finished(run_dir) and band.exists(), any_code=True)
    if code is not None and (code != 0 or not band.exists()):
        raise Stop(EXIT_FAILED, f"calibration exited {code} and wrote no band file (1: an "
                                "honest step rejected; 2: the concentration guard failed or "
                                "the recomputed k exceeds VERIF_K)")
    check_band_file(cell, "after calibration")


def _verified_done(run_dir: Path) -> Callable[[], bool]:
    return lambda: finished(run_dir) and _weights_ok(run_dir)


def stage_h1_h3(cell: Cell) -> None:
    h1 = cell.honest(cell.h1_seed)
    cell.run_stage("H1", h1, "verification.runs.run_verified", ["--metrics"],
                   cell.env(cell.h1_seed), done=_verified_done(h1),
                   exit1=lambda: _rejected(h1))
    h3 = cell.cell / PLAIN_DIR
    cell.run_stage("H3", h3, "verification.runs.plain_baseline", ["--metrics"],
                   cell.env(cell.h1_seed), done=_verified_done(h3))


def gate_f5c_f9(cell: Cell) -> None:
    """The band file, F5c, then F9, between H3 and H2."""
    from verification.runs.plain_baseline import read_final_weights

    check_band_file(cell, "before H2")
    h1_dir = cell.honest(cell.h1_seed)
    verdict = _verdict(h1_dir)
    if verdict is None:
        raise StageError(f"H1 wrote no verdict record ({h1_dir})")
    accepted = verdict.get("accepted") is True
    line = 0.0 if cell.plant == "guard-trip" else TRIP_LINE
    trips = guard_trips(h1_dir, line)
    if cell.plant == "guard-trip":
        cell.out(f"PLANTED: the trip line is {line}, not {TRIP_LINE} (rehearsal)")
    h1 = read_final_weights(h1_dir)
    h3 = read_final_weights(cell.cell / PLAIN_DIR)
    if cell.plant == "h3-mismatch":
        name = next(iter(h3["hashes"]))
        h3 = {**h3, "hashes": {**h3["hashes"], name: "00" * 32}}
        cell.out(f"PLANTED: H3's hash of {name} replaced in memory (rehearsal)")
    where = f"step {verdict.get('rejection_step')} check {verdict.get('rejection_check')}"
    bad = f5c_mismatches(h1, h3) if accepted else f5c_partial(h1, h3)
    if bad:
        cell.log("F5c", "MISMATCH" if accepted else "MISMATCH (partial comparison)",
                 why="; ".join(bad))
        raise Stop(EXIT_F5C, "F5c: H1's final weights differ from H3's: " + "; ".join(bad)
                   + ". Every later run is stopped; find and fix the cause first")
    if accepted:
        cell.log("F5c", "H1 and H3 final weights bit-identical", root=h1.get("root"))
    else:
        n = len(h1.get("losses") or [])
        cell.log("F5c", f"partial comparison: H1 rejected at {where}, so it has no final "
                        f"weights; provenance and the losses of steps 1–{n} equal H3's")
    first = trips[0][0] if trips else None
    at_trip = (not accepted and first is not None and verdict.get("rejection_step") == first
               and str(verdict.get("rejection_check")) == "5")
    if not accepted and not at_trip:
        raise Stop(EXIT_FAILED, f"H1 was rejected at {where} (verdict {verdict}), not at check "
                                f"5 at the first step that trips the guard ({first}): F5c "
                                "can't hold without its final weights, and --guard-ack "
                                "doesn't cover it")
    if trips:
        what = ", ".join(f"step {t} {c} κ/κ_max {x:.3f}" for t, c, x in trips)
        if not cell.guard_ack:
            cell.log("F9", "PAUSED before H2", trips=what)
            raise Stop(EXIT_F9, f"F9: H1's judged steps trip the cancellation guard ({what}). "
                                "This driver has no revision path and rewrites no band file; "
                                "rerun with --guard-ack to go on with the band file as it is")
        cell.log("F9", "guard trip acknowledged (--guard-ack)", trips=what)
    else:
        cell.log("F9", "no judged H1 step trips the guard")


def stage_h2(cell: Cell) -> None:
    for s in cell.seeds[1:]:
        d = cell.honest(s)
        stage = f"H2 seed {s}"
        code = cell.run_stage(stage, d, "verification.runs.run_verified", ["--metrics"],
                              cell.env(s), done=_verified_done(d), exit1=lambda: _rejected(d),
                              fatal=False)
        if code not in (None, 0, 1):
            continue  # failed, logged and counted by run_stage
        v = _verdict(d)
        if v is None:
            cell.log(stage, "FAILED", why="no verdict record")
            cell.failures.append(f"{stage}: no verdict record")
        elif v.get("accepted") is not True:
            what = (f"seed {s} step {v.get('rejection_step')} check "
                    f"{v.get('rejection_check')}")
            cell.log(stage, "REJECTED (honest false reject)", seed=s,
                     step=v.get("rejection_step"), check=v.get("rejection_check"))
            cell.out(f"REJECTED (honest false reject) {what}")
            cell.false_rejects.append(f"REJECTED (honest false reject) {what}")


def _oracle_failures(run_dir: Path) -> list[str]:
    rows = [r for r in _records(run_dir) if r.get("record") == "cheat_oracle"]
    if not rows:
        return [f"{run_dir.name}: no cheat_oracle record"]
    return [f"{run_dir.name}: {r.get('outcome')}" for r in rows if not r.get("passed")]


def stage_cheats(cell: Cell, data: Mapping[tuple[int, int], bool]) -> None:
    for s in cell.seeds:
        stage = f"cheats seed {s}"
        levels = [n for n in cell.levels if data.get((s, n))]
        dirs = {n: [cell.cell / cell.name(x, n, s) for x in SCENARIOS] for n in levels}
        todo = [n for n in levels if not all(finished(d) for d in dirs[n])]
        if not todo:
            cell.log(stage, "done before, skipped", levels=levels)
        else:
            args = ["cheats", "--levels", *map(str, todo)]
            code, sec = cell.launch(stage, "verification.runs.full_scale", args, cell.env(s))
            missing = [d.name for n in todo for d in dirs[n] if not finished(d)]
            if missing:
                cell.log(stage, "crashed", sec, exit=code)
                raise StageError(f"cheats seed {s} exited {code}; unfinished: {missing}")
            explained = code == 1 and all(_has_oracle(d) for n in todo for d in dirs[n])
            if code == 0 or explained:
                cell.log(stage, "ran", sec, exit=code, levels=todo)
                if explained and not any(_oracle_failures(d) for n in todo for d in dirs[n]):
                    cell.failures.append(f"{stage}: exited 1, but every oracle record passed")
            else:
                why = f"{stage}: exited {code}, and no record explains it"
                cell.log(stage, "FAILED", sec, exit=code, levels=todo, why=why)
                cell.failures.append(why)
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
                   cell.env(s), exit1=lambda: _rejected(d) and _has_oracle(d), fatal=False)
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
        d = cell.cell / run
        code = cell.run_stage(stage, d, module, ["--metrics", *extra], cell.env(cell.h1_seed),
                              exit1=lambda d=d: _rejected(d), fatal=False)
        if code == 1 and f"{stage}: " not in " ".join(cell.failures):
            cell.failures.append(f"{stage}: exit 1")


def check_band_hashes(cell: Cell) -> int:
    """P10a over the cell: every run's ``environment`` record (a calibration run's
    ``calibration`` record) names ``bands.json``'s hash, the plain baseline's none. Another hash
    stops the cell (exit 7), a missing one fails it. Returns the run directories checked."""
    check_band_file(cell, "after the last run")
    want, _, unnamed = _stray_band_hashes(cell)
    if unnamed:
        raise StageError(f"P10a: runs that name no band file: {unnamed}")
    n = len(list(cell.cell.rglob(RECORDS)))
    cell.log("P10a", f"one band-file hash across {n} run directories", hash=want[:16])
    return n


def _exit_code(cell: Cell) -> int:
    others = [f for f in cell.failures if f not in cell.false_rejects]
    if others:
        return EXIT_FAILED
    return EXIT_FALSE_REJECT if cell.false_rejects else 0


def run_cell(cell: Cell) -> int:
    """The whole cell in order, under the cell's lock; the exit code (module docstring)."""
    cell.cell.mkdir(parents=True, exist_ok=True)
    try:
        lock = take_lock(cell.cell)
    except Busy as e:
        cell.out(f"REFUSED: {e}")
        return EXIT_FAILED
    try:
        return _run_cell(cell)
    finally:
        release_lock(lock)


def _run_cell(cell: Cell) -> int:
    t0 = time.perf_counter()
    cell.out(f"FS13 cell {cell.cell}: {cell.model}, {cell.attack}, n_s {cell.batch}, k "
             f"{cell.k}, seeds {cell.seeds}, levels {cell.levels}, hidden level {cell.hidden}"
             + (f", PLANT {cell.plant}" if cell.plant else ""))
    try:
        data = stage_data(cell)
        if cell.data_only:
            cell.log("cell", "data only: stopped after H0", time.perf_counter() - t0)
            return 0
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
    except Interrupted as e:
        cell.log("cell", "INTERRUPTED", time.perf_counter() - t0, why=str(e))
        cell.out(f"INTERRUPTED by {e}: the running child was stopped; rerun to resume")
        return 128 + e.signum
    except KeyboardInterrupt:
        cell.log("cell", "INTERRUPTED", time.perf_counter() - t0, why="SIGINT")
        cell.out("INTERRUPTED by SIGINT: the running child was stopped; rerun to resume")
        return 128 + signal.SIGINT
    except Exception as e:  # a driver bug or a broken file: logged, never silent
        cell.log("cell", "CRASHED", time.perf_counter() - t0, why=repr(e))
        cell.out(f"CRASHED: {e!r}\n{traceback.format_exc()}")
        return EXIT_FAILED
    s = time.perf_counter() - t0
    code = _exit_code(cell)
    if cell.failures or cell.false_rejects:
        failures = cell.failures + cell.false_rejects
        cell.log("cell", "finished with failures", s, failures=failures, exit=code)
        return code
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
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the whole cell, resuming where it stopped",
                       allow_abbrev=False)
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
    r.add_argument("--data-only", action="store_true",
                   help="build the cell's data (H0) and stop, before any GPU session")
    r.add_argument("--plant", choices=PLANTS, default=None,
                   help="rehearse a stop (module docstring); changes no file or check")
    ch = sub.add_parser("cheats", help="A1–A3 at one seed (VERIF_SEED), several levels",
                        allow_abbrev=False)
    ch.add_argument("--levels", type=int, nargs="+", required=True)
    hi = sub.add_parser("hidden", help="the hidden-step run at one seed (VERIF_SEED)",
                        allow_abbrev=False)
    hi.add_argument("--level", type=int, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "cheats":
        return run_cheats(args.levels)
    if args.cmd == "hidden":
        return run_cheats([args.level], hidden=True)
    cfg = load_config()
    pc = load_protocol_config()
    sweep, kargs = shlex.split(args.sweep_args), shlex.split(args.k_args)
    if reserved_args(sweep + kargs):
        p.error(f"--sweep-args and --k-args may not name {', '.join(RESERVED_SWEEP_ARGS)} "
                f"or a prefix of one: {reserved_args(sweep + kargs)}")
    levels = sorted(set(args.levels)) if args.levels else eq5_levels(cfg.batch)
    hidden = hidden_level(cfg.batch) if args.hidden_level is None else args.hidden_level
    if not all(1 <= n <= cfg.batch for n in [*levels, hidden]):
        p.error(f"levels must lie in 1…{cfg.batch} (level 0 is the honest run)")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds repeats a seed")
    cell = Cell(cfg.output_dir.resolve(), cfg.model, cfg.attack, cfg.batch, pc.k,
                list(args.seeds), levels, hidden, sweep, kargs, args.guard_ack, args.plant,
                revision=cfg.model_revision, data_only=args.data_only)
    previous = install_signal_handlers()
    try:
        return run_cell(cell)
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


if __name__ == "__main__":
    sys.exit(main())
