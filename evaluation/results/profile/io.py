"""FS25's readers for the profiling files of PROFILING_PLAN §5 (version 1): ``markers.jsonl``,
``samples.jsonl``, ``probe.json``, ``c5_layout.json``, ``trace_summary.json`` and the run
directory's ``records.jsonl``. Each reader rejects a file whose ``version`` isn't 1 with a
``ValueError`` that names the file.

Units stay as written: marker times are µs from a step's ``t0_ns``, sample times are monotonic
ns, time rows are seconds. :mod:`evaluation.results.profile.analyze` converts at its joins.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.records import read_jsonl

__all__ = ["VERSION", "MARKERS_FILE", "C5_LAYOUT_FILE", "Markers", "Samples", "read_markers",
           "read_samples", "read_probe", "read_c5_layout", "read_trace_summary", "read_records"]

VERSION = 1
MARKERS_FILE = "markers.jsonl"
C5_LAYOUT_FILE = "c5_layout.json"


@dataclass
class Markers:
    header: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Samples:
    header: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


def _check_version(obj: Any, path: Path) -> None:
    if not isinstance(obj, dict) or obj.get("version") != VERSION:
        got = obj.get("version") if isinstance(obj, dict) else None
        raise ValueError(f"{path}: version {got!r}, expected {VERSION}")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    _check_version(obj, path)
    return obj


def _read_lines(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header (first line, version checked) and the lines after it. A last line cut short
    by a process still writing is skipped with a warning."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) and not line.endswith("\n"):
                warnings.warn(f"{path}: line {i} is incomplete, skipped", UserWarning,
                              stacklevel=3)
                continue
            raise ValueError(f"{path}: line {i}: {e}") from None
    if not rows or rows[0].get("kind") != "header":
        raise ValueError(f"{path}: the first line isn't a header")
    _check_version(rows[0], path)
    return rows[0], rows[1:]


def read_markers(path: str | Path) -> Markers:
    path = Path(path)
    header, rest = _read_lines(path)
    out = Markers(header)
    seen: set[Any] = set()
    for r in rest:
        kind = r.get("kind")
        if kind == "step":
            out.steps.append(r)
        elif kind == "run":
            out.runs.append(r)
        elif kind not in seen:
            seen.add(kind)
            warnings.warn(f"{path}: skipped lines of unknown kind {kind!r}", stacklevel=2)
    return out


def read_samples(path: str | Path) -> Samples:
    path = Path(path)
    header, rest = _read_lines(path)
    seen: set[Any] = set()
    rows = []
    for r in rest:
        if "kind" not in r:
            rows.append(r)
        elif r["kind"] not in seen:  # FS21's error lines: skipped, one warning per kind
            seen.add(r["kind"])
            warnings.warn(f"{path}: skipped lines of kind {r['kind']!r}", stacklevel=2)
    return Samples(header, rows)


def read_probe(path: str | Path) -> dict[str, Any]:
    return _read_json(path)


def read_c5_layout(path: str | Path) -> dict[str, Any]:
    return _read_json(path)


def read_trace_summary(path: str | Path) -> dict[str, Any]:
    """``{"version": 1, "sections": {name: {...}}, "c5_slice": {class: {...}}, "warnings": []}``."""
    obj = _read_json(path)
    if not isinstance(obj.get("sections"), dict):
        raise ValueError(f"{path}: no 'sections' mapping")
    return obj


def read_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """``records.jsonl`` of a run directory, as :func:`evaluation.records.read_jsonl` reads it."""
    return read_jsonl(Path(run_dir) / "records.jsonl")
