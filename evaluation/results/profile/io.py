"""FS25's readers for the profiling files of archive/PROFILING_PLAN §5 (version 1):
``markers.jsonl``, ``samples.jsonl``, ``probe.json``, ``c5_layout.json``, ``trace_summary.json``
and the run directory's ``records.jsonl``. Each reader rejects a file whose ``version`` isn't
one it reads with a ``ValueError`` that names the file. FS35: ``samples.jsonl`` and
``probe.json`` read as version 1 or 2 (:data:`VERSIONS`); version 2 holds every GPU of the
machine (``setup/sampler.py``, ``verification/runs/probe.py``).

Units stay as written: marker times are µs from a step's ``t0_ns``, sample times are monotonic
ns, time rows are seconds. :mod:`evaluation.results.profile.analyze` converts at its joins.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.records import read_jsonl

__all__ = ["VERSION", "MARKERS_FILE", "C5_LAYOUT_FILE", "Markers", "Samples", "read_markers",
           "read_samples", "read_probe", "read_c5_layout", "read_trace_summary", "read_records",
           "class_shares", "batch_cover", "entry_bytes"]

VERSION = 1
VERSIONS = {"samples": (1, 2), "probe": (1, 2)}  # the other files: (VERSION,)
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


def _check_version(obj: Any, path: Path, accepted: tuple[int, ...] = (VERSION,)) -> None:
    if not isinstance(obj, dict) or obj.get("version") not in accepted:
        got = obj.get("version") if isinstance(obj, dict) else None
        want = " or ".join(map(str, accepted))
        raise ValueError(f"{path}: version {got!r}, expected {want}")


def _read_json(path: str | Path, accepted: tuple[int, ...] = (VERSION,)) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    _check_version(obj, path, accepted)
    return obj


def _read_lines(path: Path, accepted: tuple[int, ...] = (VERSION,)
                ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    _check_version(rows[0], path, accepted)
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
    header, rest = _read_lines(path, VERSIONS["samples"])
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
    return _read_json(path, VERSIONS["probe"])


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


ITEM_BYTES = {"float32": 4, "float64": 8, "bfloat16": 2, "float16": 2}
"""Bytes per element by dtype name (``torch.`` prefix removed)."""


def entry_bytes(e: dict[str, Any]) -> float:
    """A layout entry's bytes as stored: ``operand_bytes`` (the layout counts the operands as
    fp32) rescaled to the product dtype's item size, which is what the kernel reads (the
    operands are served as views in the product's dtype), plus ``product_bytes``."""
    item = ITEM_BYTES.get(str(e.get("product_dtype", "float32")).removeprefix("torch."), 4)
    return e.get("operand_bytes", 0) * item / 4 + e.get("product_bytes", 0)


def class_shares(entries: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Each class's share of the layout ``entries``' bytes as stored (:func:`entry_bytes`),
    summing to 1, in order of first appearance. A stamp that covers entries
    of several classes (an attention run of scores and mixes, a batched measuring) is spread
    over the classes by these shares; entries with no bytes share equally."""
    tot: dict[str, float] = {}
    for e in entries:
        tot[e["class"]] = tot.get(e["class"], 0.0) + entry_bytes(e)
    whole = sum(tot.values())
    if not tot:
        return {}
    if whole <= 0:
        return {c: 1.0 / len(tot) for c in tot}
    return {c: v / whole for c, v in tot.items()}


def batch_cover(stamps: Sequence[tuple[float, str, int, int]]) -> dict[int, list[int]]:
    """The runs each batched measuring stamp (``measure``, group -1) covers.

    ``stamps`` are ``(start, name without "c5.", run, group)``, in any order. On the GPU path
    the unit pass serves every run of a schedule step (a ``run`` stamp each) and then measures
    them all in one ``measure`` stamp, labelled with the step's first run and group -1
    (``checks.py``, ``_measure_step_kernel``). So the runs a batched stamp covers are those whose
    ``run`` stamps start after the previous batched stamp. A second batched stamp with the same
    run id (a step on a second device) covers the same runs. A stamp for one group (group >= 0,
    a run measured on its own) ends the batch collected so far. Returns the covered runs by
    index in ``stamps``; a batched stamp with no run before it maps to ``[]``.

    Two limits. With overlap on (FS39) a batched stamp's host time includes the wait for the
    previous step's measuring on that device, so spreading it over this step's classes is an
    approximation. A second device's stamp for the same step is spread over all of the step's
    runs, not only that device's (the decoder uses one device per step)."""
    out: dict[int, list[int]] = {}
    pending: list[int] = []
    last: tuple[int, list[int]] | None = None
    for i in sorted(range(len(stamps)), key=lambda j: stamps[j][0]):
        _, name, run, group = stamps[i]
        if name == "run":
            if run not in pending:
                pending.append(run)
        elif name == "measure" and group >= 0:
            pending = []
        elif name == "measure":
            if pending:
                last = (run, pending)
                pending = []
            out[i] = list(last[1]) if last is not None and last[0] == run else []
    return out
