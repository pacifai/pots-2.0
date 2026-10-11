"""EV1: the run directories under ``$VERIF_OUTPUT_DIR`` read as tables (EVALUATION_SPEC §4).

**Run directories.** A run directory is any directory under the root that holds
``records.jsonl`` or ``residuals/``, at any depth, so ``calibrate/k9/`` and a sweep's
``--subdir`` count. The reserved top-level names ``data`` and ``evaluation`` (§4.6) are
skipped, and so is the root itself. :func:`read` reads every run directory into
:class:`Records`:

- ``tables``: one :class:`Table` per record type, rows in file order and run directories in
  path order. Each row is the record as written plus ``run_dir``, the directory's path relative
  to the root. A run's name is not unique (``calibrate/k9/`` writes run ``calibrate``), so
  ``run_dir`` identifies it. Types this module doesn't know, such as ``calibration`` or
  ``sweep_point``, get a table like the rest. The writer stores a non-finite float as the
  string ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``; these read back as floats. A last line
  cut short by a run still writing is skipped with a warning.
- ``residuals``: one :class:`Archive` per ``residuals/<scenario>/step_<t>.npz``, with run,
  scenario and step read from the archive. ``Archive.load()`` reads the arrays.
- ``archives``: any other ``.npz`` file at the top of a run directory, such as the sweep's
  ``sweep_trials.npz``.

**Derived rows.** These come from methods and never enter ``tables``:

- P1, matmul capture (EQ1b). :func:`derive_capture` repeats
  ``verification.runs.metrics.derive_capture``, and :meth:`Records.capture` pairs runs as
  ``verification.runs.plain_baseline.capture_rows`` does. ``evaluation`` never imports
  ``verification`` (``tests/test_layering.py``), so the arithmetic is copied here, and
  ``tests/evaluation/test_records.py`` checks that the copy agrees with the original.
- Check 3. It has no cost row of its own: it is the rule that check 5's operands come from
  committed leaves, and its cost is ``5.glue`` (``runs/metrics.py``). :func:`check3_rows`
  copies each ``5.glue`` row as component ``3``.

:meth:`Records.costs` returns one cost kind as written, with both kinds of derived row added.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DEFAULT_OUTPUT_DIR", "RESERVED", "RECORDS_FILE", "RESIDUALS_DIR", "COST_RECORDS",
           "PAIR_KEYS", "output_dir", "Table", "Archive", "Records", "find_run_dirs",
           "read_jsonl", "read", "derive_capture", "check3_rows"]

# setup/config.py's default. It is read here, because setup.config imports torch and a
# result script must not.
DEFAULT_OUTPUT_DIR = "trainer_output/verification"
RESERVED = ("data", "evaluation")  # §4.6: no run may use these names
RECORDS_FILE = "records.jsonl"
RESIDUALS_DIR = "residuals"
COST_RECORDS = ("time", "memory", "count")
# A verified run takes P1 from the plain baseline that trained the same model on the same data
# on the same device.
PAIR_KEYS = ("model", "corpus", "device")

_NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_STEP_FILE = re.compile(r"step_(\d+)\.npz")


def output_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$VERIF_OUTPUT_DIR`` from ``env`` (default ``os.environ``), as ``setup.config`` reads it."""
    env = os.environ if env is None else env
    value = env.get("VERIF_OUTPUT_DIR", DEFAULT_OUTPUT_DIR).strip()
    if not value:
        raise ValueError("VERIF_OUTPUT_DIR must be a non-empty string")
    return Path(value)


class Table(list[dict[str, Any]]):
    """The rows of one record type: a list of dicts, with a few selectors."""

    def where(self, **match: Any) -> Table:
        """Rows whose fields equal each value given; a callable value tests the field instead."""
        def ok(r: Mapping[str, Any]) -> bool:
            return all(v(r.get(k)) if callable(v) else r.get(k) == v for k, v in match.items())
        return Table(r for r in self if ok(r))

    def column(self, name: str) -> list[Any]:
        return [r.get(name) for r in self]

    def distinct(self, name: str) -> list[Any]:
        """The field's values, each once, in first-seen order."""
        return list(dict.fromkeys(r.get(name) for r in self))

    def group_by(self, *names: str) -> dict[tuple[Any, ...], Table]:
        out: dict[tuple[Any, ...], Table] = {}
        for r in self:
            out.setdefault(tuple(r.get(n) for n in names), Table()).append(r)
        return out


@dataclass(frozen=True)
class Archive:
    """One ``.npz`` file of a run directory. ``run``, ``scenario`` and ``step`` are ``None`` when
    the archive doesn't hold them as single values."""

    run_dir: str
    path: Path
    run: str | None
    scenario: str | None
    step: int | None

    def load(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def _archive(run_dir: str, path: Path) -> Archive:
    try:
        with np.load(path, allow_pickle=False) as z:
            one = {k: z[k].item() for k in ("run", "scenario", "step")
                   if k in z.files and z[k].ndim == 0}
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: not a readable .npz archive ({e})") from None
    step = one.get("step")
    if step is None and (m := _STEP_FILE.fullmatch(path.name)):
        step = int(m.group(1))
    return Archive(run_dir, path, one.get("run"), one.get("scenario"),
                   None if step is None else int(step))


def find_run_dirs(root: str | os.PathLike[str]) -> list[Path]:
    """Every run directory under ``root``, in path order (module docstring)."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [n for n in dirnames if n not in RESERVED]
        elif RECORDS_FILE in filenames or RESIDUALS_DIR in dirnames:
            out.append(here)
        if RESIDUALS_DIR in dirnames:
            dirnames.remove(RESIDUALS_DIR)
        dirnames.sort()
    return sorted(out)


def _decode(x: Any) -> Any:
    if isinstance(x, str):
        return _NON_FINITE.get(x, x)
    if isinstance(x, dict):
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    return x


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """One ``records.jsonl``, non-finite strings decoded. Every line must be a record (an
    object with a ``record`` field), except a last line with no newline that doesn't parse:
    that is a run still writing, and it is skipped with a warning."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            if i == len(lines) and not line.endswith("\n"):
                warnings.warn(f"{path}: line {i} is incomplete, skipped (is the run still "
                              "writing?)", UserWarning, stacklevel=2)
                continue
            raise ValueError(f"{path}: line {i}: {e}") from None
        if not isinstance(row, dict) or "record" not in row:
            raise ValueError(f"{path}: line {i} is not a record (no 'record' field)")
        rows.append(_decode(row))
    return rows


@dataclass
class Records:
    """Every run directory under ``root``, read by :func:`read`."""

    root: Path
    run_dirs: list[str]
    tables: dict[str, Table]
    residuals: list[Archive]
    archives: list[Archive]

    def table(self, record: str) -> Table:
        """The rows of one record type; empty if no run wrote it."""
        return self.tables.get(record, Table())

    def counts(self) -> dict[str, Counter[str]]:
        """Rows per record type in each run directory."""
        out: dict[str, Counter[str]] = {d: Counter() for d in self.run_dirs}
        for name, rows in self.tables.items():
            for r in rows:
                out[r["run_dir"]][name] += 1
        return out

    def environment(self, run_dir: str) -> dict[str, Any] | None:
        """The run directory's first ``environment`` record."""
        return next((r for r in self.table("environment") if r["run_dir"] == run_dir), None)

    def _cost_rows(self, run_dir: str) -> list[dict[str, Any]]:
        return [r for k in COST_RECORDS for r in self.table(k) if r["run_dir"] == run_dir]

    def _dirs_with_phase(self, prefix: str) -> list[str]:
        return list(dict.fromkeys(
            r["run_dir"] for k in COST_RECORDS for r in self.table(k)
            if r.get("level") == "phase" and str(r.get("component", "")).startswith(prefix)))

    def verified_dirs(self) -> list[str]:
        """Run directories with captured training rows (``train.*``)."""
        return self._dirs_with_phase("train.")

    def plain_dirs(self) -> list[str]:
        """Run directories with plain training rows (``P0.*``)."""
        return self._dirs_with_phase("P0.")

    def plain_for(self, run_dir: str) -> str | None:
        """The plain baseline whose environment record matches ``run_dir``'s on
        :data:`PAIR_KEYS`, or ``None``. More than one raises ``ValueError``."""
        def key(d: str) -> tuple[Any, ...]:
            env = self.environment(d) or {}
            return tuple(env.get(k) for k in PAIR_KEYS)

        hits = [d for d in self.plain_dirs() if key(d) == key(run_dir)]
        if len(hits) > 1:
            raise ValueError(f"{run_dir}: {len(hits)} plain baselines match on {PAIR_KEYS}, "
                             f"{hits}; pass plain= to pick one")
        return hits[0] if hits else None

    def capture(self, verified: str | None = None, *, plain: str | None = None,
                scenario: str | None = None) -> Table:
        """P1 rows (:func:`derive_capture`) for the run directory ``verified``, or for every
        verified run directory, against the plain baseline ``plain`` (default
        :meth:`plain_for`; a run with no match gets no rows). ``scenario`` keeps one scenario.
        Each row adds ``run_dir`` and ``plain_run_dir``.

        With ``verified`` given, this is ``capture_rows(verified, plain, scenario=scenario)``:
        a plain step with no verified counterpart in a kind gets no row and a ``UserWarning``.
        """
        out = Table()
        for d in [verified] if verified is not None else self.verified_dirs():
            p_dir = plain if plain is not None else self.plain_for(d)
            if p_dir is None:
                if verified is not None:
                    raise ValueError(f"{d}: no plain baseline matches on {PAIR_KEYS}")
                continue
            v = [r for r in self._cost_rows(d) if scenario is None or r.get("scenario") == scenario]
            p = self._cost_rows(p_dir)
            if verified is not None:
                _warn_unmatched(v, p)
            out.extend({**r, "run_dir": d, "plain_run_dir": p_dir} for r in derive_capture(v, p))
        return out

    def costs(self, record: str, *, plain: str | None = None) -> Table:
        """The ``record`` cost rows (``time``, ``memory`` or ``count``) as written, then P1
        for every verified run directory (:meth:`capture`) and check 3 (:func:`check3_rows`)."""
        if record not in COST_RECORDS:
            raise ValueError(f"record must be one of {COST_RECORDS}, got {record!r}")
        raw = self.table(record)
        return Table([*raw, *self.capture(plain=plain).where(record=record), *check3_rows(raw)])


def read(root: str | os.PathLike[str] | None = None) -> Records:
    """Every run directory under ``root`` (default :func:`output_dir`)."""
    root = output_dir() if root is None else Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no output directory at {root}")
    dirs, tables, residuals, archives = [], {}, [], []
    for d in find_run_dirs(root):
        rel = d.relative_to(root).as_posix()
        dirs.append(rel)
        if (d / RECORDS_FILE).is_file():
            for r in read_jsonl(d / RECORDS_FILE):
                r["run_dir"] = rel
                tables.setdefault(r["record"], Table()).append(r)
        residuals += [_archive(rel, f) for f in sorted((d / RESIDUALS_DIR).rglob("*.npz"))]
        archives += [_archive(rel, f) for f in sorted(d.glob("*.npz"))]
    return Records(root, dirs, tables, residuals, archives)


# ---- P1, derived ------------------------------------------------------------------------

_XS = ("load", "forward", "backward", "update")
_COUNT_FIELDS = ("flops", "hash_in_bytes", "hash_out_bytes", "hash_calls")


def _phases(rows: Sequence[Mapping[str, Any]], record: str
            ) -> dict[tuple[str, str, int], dict[str, Mapping[str, Any]]]:
    out: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("record") == record and r.get("level") == "phase" and r["step"] > 0:
            out.setdefault((r["run"], r["scenario"], r["step"]), {})[r["component"]] = r
    return out


def _get(table: Mapping[str, Mapping[str, Any]], name: str, t: int) -> Mapping[str, Any]:
    if name not in table:
        raise ValueError(f"derive_capture: no {name!r} phase row at step {t}")
    return table[name]


def derive_capture(verified: Iterable[Mapping[str, Any]],
                   plain: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """P1 per step, as ``verification.runs.metrics.derive_capture`` computes it.

    The plain rows must hold one ``(run, scenario)``, matched to the verified rows by step:

    - time: ``Σ_x (train.x − P0.x) + P1.label`` over the four training phases;
    - memory: ``(peak(train.backward) − start(train.forward)) − (peak(P0.backward) −
      start(P0.forward))``, ``None`` if any of the four is;
    - counts: ``Σ_x (train.x − P0.x)`` per count field.

    Each row is a ``component`` row named ``P1`` with ``derived: True``. A step missing a phase
    raises ``ValueError``.
    """
    verified, plain = list(verified), list(plain)
    runs = {(r["run"], r["scenario"]) for r in plain
            if r.get("record") in COST_RECORDS and r.get("level") == "phase"}
    if len(runs) > 1:
        raise ValueError(f"derive_capture: the plain rows hold {len(runs)} (run, scenario) "
                         f"pairs, {sorted(runs)}; pass one")

    def by_step(record: str) -> dict[int, dict[str, Mapping[str, Any]]]:
        return {k[2]: v for k, v in _phases(plain, record).items()}

    plain_time, plain_mem, plain_count = by_step("time"), by_step("memory"), by_step("count")
    base = {"side": "prover", "component": "P1", "level": "component", "derived": True}
    out: list[dict[str, Any]] = []
    for (run, scen, t), v in _phases(verified, "time").items():
        if (p := plain_time.get(t)) is None:
            continue
        label = _get(v, "P1.label", t)
        dt = sum(_get(v, f"train.{x}", t)["time_s"] - _get(p, f"P0.{x}", t)["time_s"]
                 for x in _XS)
        out.append({"record": "time", "run": run, "scenario": scen, "step": t, **base,
                    "calls": label["calls"], "time_s": dt + label["time_s"]})
    for (run, scen, t), v in _phases(verified, "memory").items():
        if (p := plain_mem.get(t)) is None:
            continue
        figs = (_get(v, "train.backward", t)["peak_bytes"],
                _get(v, "train.forward", t)["start_bytes"],
                _get(p, "P0.backward", t)["peak_bytes"], _get(p, "P0.forward", t)["start_bytes"])
        grow = None if any(f is None for f in figs) else \
            (figs[0] - figs[1]) - (figs[2] - figs[3])
        out.append({"record": "memory", "run": run, "scenario": scen, "step": t, **base,
                    "calls": 1, "peak_bytes": grow, "start_bytes": None, "end_bytes": None,
                    "mem_source": v["train.backward"]["mem_source"]})
    for (run, scen, t), v in _phases(verified, "count").items():
        if (p := plain_count.get(t)) is None:
            continue
        out.append({"record": "count", "run": run, "scenario": scen, "step": t, **base,
                    "calls": 1, **{f: sum(_get(v, f"train.{x}", t)[f] - _get(p, f"P0.{x}", t)[f]
                                          for x in _XS) for f in _COUNT_FIELDS}})
    return out


def _warn_unmatched(verified: Sequence[Mapping[str, Any]],
                    plain: Sequence[Mapping[str, Any]]) -> None:
    def steps(rows: Sequence[Mapping[str, Any]], kind: str) -> set[int]:
        return {r["step"] for r in rows
                if r.get("record") == kind and r.get("level") == "phase" and r["step"] > 0}

    for kind in COST_RECORDS:
        missing = sorted(steps(plain, kind) - steps(verified, kind))
        if missing:
            warnings.warn(f"capture_rows: plain-run {kind} steps {missing} have no verified "
                          "counterpart; no P1 row for them", UserWarning, stacklevel=3)


def check3_rows(rows: Iterable[Mapping[str, Any]]) -> Table:
    """Check 3's cost rows: each ``5.glue`` row copied as component ``3``, with
    ``derived: True`` and ``alias_of: "5.glue"``. The level stays ``sub``, so a sum over
    components counts the cost once, inside check 5."""
    return Table({**r, "component": "3", "derived": True, "alias_of": "5.glue"}
                 for r in rows if r.get("component") == "5.glue")
