"""Summaries of the numbers checks 5 and 6 record: the per-class residual table of C1.

Checks 5 and 6 record one :class:`~verification.verifier.context.ProductStat` per product and
one :class:`~verification.verifier.context.TensorStat` per weight tensor in each step's
:class:`~verification.verifier.context.StepStats`, the same objects ``runs/metrics.py`` writes
to and reads back from the residual archives. This module bins them for reporting:

- :func:`class_summary`: per matmul class (P3.c's κ class), the product count, the RMS and the
  maximum of the normalized residuals (every product's ``k`` values, P3.a), and κ's median and
  maximum. C1 takes ``s_h`` as the largest class RMS, and reports the global maximum against it.
- :func:`tensor_summary`: per check (6a, 6b) and weight role (the weight name with its layer
  index replaced by ``*``), the tensor count and the largest ``ρ_max`` (P5).

It judges nothing. The bands are fitted elsewhere (A11).
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from verification.verifier.context import StepStats

__all__ = ["ClassSummary", "TensorSummary", "class_summary", "tensor_summary", "weight_role",
           "format_class_table", "format_tensor_table"]


@dataclass(frozen=True)
class ClassSummary:
    """One matmul class over the given steps. ``n`` is ``count·k``, the residuals behind ``rms``."""

    cls: str
    count: int
    n: int
    rms: float
    max: float
    max_name: str
    max_step: int
    kappa_median: float
    kappa_max: float


@dataclass(frozen=True)
class TensorSummary:
    """One weight role of check 6a or 6b over the given steps."""

    check_id: str
    role: str
    count: int
    rho_max: float
    max_name: str
    max_step: int


_LAYER_INDEX = re.compile(r"\.\d+\.")


def weight_role(name: str) -> str:
    """The weight name with its layer index starred: ``model.layers.3.mlp.up_proj.weight`` →
    ``model.layers.*.mlp.up_proj.weight``."""
    return _LAYER_INDEX.sub(".*.", name)


def _steps(stats: dict[int, StepStats] | Iterable[tuple[int, StepStats]]
           ) -> list[tuple[int, StepStats]]:
    items = stats.items() if isinstance(stats, dict) else stats
    return sorted(items, key=lambda x: x[0])


def class_summary(stats: dict[int, StepStats] | Iterable[tuple[int, StepStats]]
                  ) -> list[ClassSummary]:
    """Per-class rows over ``{step: StepStats}``, classes in order of first appearance (the
    canonical product order). A non-finite residual makes its class's RMS and max non-finite."""
    acc: dict[str, list] = {}
    for t, st in _steps(stats):
        for p in st.products:
            a = acc.setdefault(p.cls, [0, 0, 0.0, -math.inf, "", 0, []])
            a[0] += 1
            for x in p.normalized:
                a[1] += 1
                a[2] += x * x
                if x > a[3] or math.isnan(x):
                    a[3], a[4], a[5] = x, p.name, t
            a[6].append(p.kappa)
    return [ClassSummary(cls, count, n, math.sqrt(sq / n) if n else math.nan, mx, name, step,
                         statistics.median(kappas), max(kappas))
            for cls, (count, n, sq, mx, name, step, kappas) in acc.items()]


def tensor_summary(stats: dict[int, StepStats] | Iterable[tuple[int, StepStats]]
                   ) -> list[TensorSummary]:
    """Per ``(check, weight role)`` rows over ``{step: StepStats}``, in order of first
    appearance."""
    acc: dict[tuple[str, str], list] = {}
    for t, st in _steps(stats):
        for s in st.tensors:
            a = acc.setdefault((s.check_id, weight_role(s.weight)), [0, -math.inf, "", 0])
            a[0] += 1
            if s.rho_max > a[1] or math.isnan(s.rho_max):
                a[1], a[2], a[3] = s.rho_max, s.weight, t
    return [TensorSummary(cid, role, n, rho, name, step)
            for (cid, role), (n, rho, name, step) in acc.items()]


def format_class_table(rows: list[ClassSummary], out: Callable[[str], None] = print) -> None:
    """Print the per-class table, then the largest class RMS, the global max and their ratio."""
    out(f"{'class':<10} {'count':>6} {'n':>7} {'RMS':>8} {'max':>8}  {'κ median':>9} "
        f"{'κ max':>9}  max at")
    for r in rows:
        out(f"{r.cls:<10} {r.count:>6} {r.n:>7} {r.rms:>8.3f} {r.max:>8.3f}  "
            f"{r.kappa_median:>9.3g} {r.kappa_max:>9.3g}  {r.max_name} (step {r.max_step})")
    if rows:
        top = max(rows, key=lambda r: r.rms)
        mx = max(rows, key=lambda r: r.max)
        out(f"largest class RMS {top.rms:.3f} ({top.cls}); global max {mx.max:.3f} "
            f"({mx.max_name}); max / largest RMS = {mx.max / top.rms:.2f}")


def format_tensor_table(rows: list[TensorSummary], out: Callable[[str], None] = print) -> None:
    """Print check 6's per-role ρ_max table, then each check's maximum."""
    w = max((len(r.role) for r in rows), default=11)
    out(f"{'check':<5} {'weight role':<{w}} {'count':>5} {'ρ_max':>8}  max at")
    for r in rows:
        out(f"{r.check_id:<5} {r.role:<{w}} {r.count:>5} {r.rho_max:>8.3f}  {r.max_name} "
            f"(step {r.max_step})")
    for cid in sorted({r.check_id for r in rows}):
        top = max((r for r in rows if r.check_id == cid), key=lambda r: r.rho_max)
        out(f"check {cid}: max ρ {top.rho_max:.3f} ({top.max_name})")
