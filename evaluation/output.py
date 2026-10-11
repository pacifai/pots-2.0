"""EV1: the figure and table writers (EVALUATION_SPEC §6).

Every output goes to ``$VERIF_OUTPUT_DIR/evaluation/results/<scale>/`` (:func:`results_dir`),
beside the run directories and never inside one. A figure is written twice, as
``<name>.pdf`` for the paper and ``<name>.png`` for review. A table is written as
``<name>.csv``, with every value at full precision, and as ``<name>.md``, formatted for
reading, with its caption above it. ``name`` is a file name without a suffix.

:func:`new_figure` builds a figure outside pyplot, so a script needs no GUI backend and
keeps no figure alive after it is written.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from evaluation.records import output_dir

__all__ = ["SCALES", "PNG_DPI", "results_dir", "new_figure", "write_figure", "write_table"]

SCALES = ("test", "full")
PNG_DPI = 200  # the review copy; the PDF is vector

Format = str | Callable[[Any], str]


def results_dir(scale: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``<root>/evaluation/results/<scale>/``, ``root`` defaulting to ``$VERIF_OUTPUT_DIR``."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return (output_dir() if root is None else Path(root)) / "evaluation" / "results" / scale


def _path(name: str, scale: str, root: str | os.PathLike[str] | None, suffix: str) -> Path:
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError(f"name must be a plain file name without a suffix, got {name!r}")
    d = results_dir(scale, root)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}{suffix}"


def new_figure(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None,
               **subplots: Any) -> tuple[Figure, Any]:
    """A figure with constrained layout and its axes, as ``Figure.subplots`` returns them."""
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.subplots(nrows, ncols, **subplots)


def write_figure(fig: Figure, name: str, scale: str, *,
                 root: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    """``<name>.pdf`` and ``<name>.png``; returns both paths."""
    pdf, png = _path(name, scale, root, ".pdf"), _path(name, scale, root, ".png")
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None})  # same figure, same bytes
    fig.savefig(png, format="png", dpi=PNG_DPI)
    return pdf, png


def _plain(x: Any) -> Any:
    """A NumPy number as the Python number it holds (``np.float32`` → ``float``, ``np.int64`` →
    ``int``). ``np.float64`` is a ``float`` already, but its ``repr`` reads ``np.float64(…)``."""
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _numpy_to_json(o: Any) -> Any:
    if isinstance(o, (np.generic, np.ndarray)):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _json(x: Any) -> str:
    return json.dumps(x, default=_numpy_to_json)


def _csv_cell(x: Any) -> str:
    x = _plain(x)
    if x is None:
        return ""
    if isinstance(x, float):
        return repr(x)  # the shortest string that reads back as the same float
    if isinstance(x, (dict, list, tuple)):
        return _json(x)
    return str(x)


def _md_cell(x: Any, fmt: Format | None) -> str:
    x = _plain(x)
    if x is None:
        s = ""
    elif fmt is not None:
        s = fmt(x) if callable(fmt) else format(x, fmt)
    elif isinstance(x, float):
        s = format(x, ".4g")
    elif isinstance(x, (dict, list, tuple)):
        s = _json(x)
    else:
        s = str(x)
    return s.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_table(rows: Iterable[Mapping[str, Any]], name: str, scale: str, *,
                root: str | os.PathLike[str] | None = None,
                columns: Sequence[str] | None = None,
                formats: Mapping[str, Format] | None = None,
                caption: str | None = None) -> tuple[Path, Path]:
    """``<name>.csv`` and ``<name>.md``; returns both paths.

    ``columns`` picks and orders the columns (default: every field, in first-seen order); a
    missing field is an empty cell. ``formats`` maps a column to a format spec (``".3f"``) or a
    function, for the Markdown only; floats default to ``".4g"``. NumPy numbers are written as
    the Python numbers they hold. ``caption`` goes above the Markdown table.
    """
    rows = list(rows)
    cols = list(columns) if columns is not None else list(dict.fromkeys(k for r in rows for k in r))
    if not cols:
        raise ValueError(f"table {name!r} has no columns")
    formats = formats or {}
    csv_path, md_path = _path(name, scale, root, ".csv"), _path(name, scale, root, ".md")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows([_csv_cell(r.get(c)) for c in cols] for r in rows)
    lines = [] if caption is None else [caption, ""]
    lines += ["| " + " | ".join(_md_cell(c, None) for c in cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(_md_cell(r.get(c), formats.get(c)) for c in cols) + " |"
              for r in rows]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
