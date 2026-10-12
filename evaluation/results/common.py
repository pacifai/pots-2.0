"""What the result scripts of EVALUATION_SPEC §5.2–§5.6 share: the command line, the scale, the
run selection, the residual arrays and the seed statistics.

**Command line.** Every script takes ``--root`` (default ``$VERIF_OUTPUT_DIR``) and ``--scale``
(``test`` or ``full``). Without ``--scale`` the scale comes from the models the runs trained:
SmolLM2 runs are test scale (§3.1), any other model is full scale, and a mix of both is an
error that asks for ``--scale``. With ``--scale`` given, a script keeps that scale's runs only.

**Runs.** An honest run is a run directory whose step records are all scenario ``honest`` with
no cheat step: H1 and H2 at full scale, T-H1 at test scale. The sweep's honest steps share a
directory with its ``flip`` scenario, so they don't count. A cheat run is a run directory with
step records of scenario A1, A2, A3 or ``hidden`` (§2.1). Honest results use judged steps only,
as the step record's ``judged`` field says (§2.3).

**Residual arrays.** :func:`load_residuals` reads one step's arrays (§4.2). They hold measured
tests only: a test a rejected step never measured has no row. A family the step didn't score
is ``NaN`` throughout, and its score in the step record is ``None`` (:func:`scored_families`):
every family at a calibration step, ``ρ_κ`` with the κ guard off. In a scored family a ``NaN``
score fails, as the live checks reject it, and counts as ``+inf`` (:func:`nan_as_inf`), as in
the step record's maxima. ``p_layer`` numbers layers ``0 … L−1`` and the output layer ``L``;
archives written before that change numbered them ``1 … L`` and the output layer ``−1``, and are
renumbered on reading.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from evaluation.output import SCALES
from evaluation.records import Archive, Records, Table, output_dir, read

__all__ = ["DEFAULT_Z", "FAMILIES", "FAMILY_LABEL", "CHEATS", "EXPECTED_CHECK", "ATTACK_ORDER",
           "ATTACK_LABEL", "Inputs", "parse", "short_model", "model_order", "attack_order",
           "infer_scale", "scale_models", "honest_dirs", "cheat_dirs", "judged_honest_steps",
           "residual_index", "load_residuals", "scored_families", "nan_as_inf", "mean_std",
           "bands", "margin_z", "band_tau", "seeds_note", "mark_infinite"]

# verification.verifier.matmul_check.sizing.Z, the band margin (τ = Z·s_h). Copied, since a
# result script may not import verification (tests/test_layering.py); a test checks the copy.
DEFAULT_Z = 8.0
FAMILIES = ("rho_5", "rho_kappa", "rho_6")
FAMILY_LABEL = {"rho_5": "ρ_5(m, j)", "rho_kappa": "ρ_κ(m)", "rho_6": "ρ_6(W)"}
CHEATS = ("A1", "A2", "A3", "hidden")
# §2.1's rejecting check per scenario, as the step record names it. For A3, §2.1 names check 6,
# the update identity, so either of its sub-checks, 6a or 6b, is as expected.
EXPECTED_CHECK = {"A1": ("4",), "A2": ("5",), "A3": ("6a", "6b"), "hidden": ("7",)}
# EQ4's panel order: the four full-scale models, then any other (test scale's SmolLM2)
_MODEL_ORDER = ("Llama-3.2-1B", "Falcon3-1B", "Qwen2.5-0.5B", "Qwen2.5-1.5B")
ATTACK_ORDER = ("jailbreak", "refusal")
ATTACK_LABEL = {"jailbreak": "jailbreak", "refusal": "targeted refusal"}
_TEST_FAMILY = "SmolLM2"  # §3.1: the test-scale model


class Inputs:
    """What a script's ``main`` works from: the records, the scale and the output root."""

    def __init__(self, records: Records, scale: str, root: Path):
        self.records, self.scale, self.root = records, scale, root
        self.models = scale_models(records, scale)


def parse(description: str, argv: Sequence[str] | None = None) -> Inputs:
    """Reads ``--root`` and ``--scale`` from ``argv``, then the records under the root."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root", type=Path, default=None,
                   help="the output directory (default: $VERIF_OUTPUT_DIR)")
    p.add_argument("--scale", choices=SCALES, default=None,
                   help="test or full (default: from the models in the records)")
    a = p.parse_args(argv)
    root = output_dir() if a.root is None else a.root
    recs = read(root)
    return Inputs(recs, a.scale or infer_scale(recs), root)


def short_model(name: str | None) -> str:
    """``HuggingFaceTB/SmolLM2-135M-Instruct`` → ``SmolLM2-135M``."""
    s = str(name).rsplit("/", 1)[-1]
    return s.removesuffix("-Instruct")


def model_order(models: Iterable[str]) -> list[str]:
    """Distinct model names in EQ4's panel order, others after them by name."""
    def key(m: str) -> tuple[int, str]:
        s = short_model(m)
        return (_MODEL_ORDER.index(s) if s in _MODEL_ORDER else len(_MODEL_ORDER), s)
    return sorted(set(models), key=key)


def attack_order(attacks: Iterable[str | None]) -> list[str | None]:
    def key(a: str | None) -> tuple[int, str]:
        return (ATTACK_ORDER.index(a) if a in ATTACK_ORDER else len(ATTACK_ORDER), str(a))
    return sorted(set(attacks), key=key)


def _trained_models(recs: Records) -> set[str]:
    return {r["model"] for r in recs.table("environment") if r.get("model")}


def infer_scale(recs: Records) -> str:
    models = _trained_models(recs)
    if not models:
        raise ValueError(f"{recs.root}: no environment record names a model; pass --scale")
    test = {m for m in models if _TEST_FAMILY in m}
    if test and test != models:
        raise ValueError(f"{recs.root} holds test-scale and full-scale runs ({sorted(models)}); "
                         "pass --scale")
    return "test" if test else "full"


def scale_models(recs: Records, scale: str) -> list[str]:
    """The models of ``scale`` that some run trained, in panel order."""
    keep = (lambda m: _TEST_FAMILY in m) if scale == "test" else (lambda m: _TEST_FAMILY not in m)
    return model_order(m for m in _trained_models(recs) if keep(m))


def honest_dirs(recs: Records, models: Sequence[str] | None = None) -> list[str]:
    """The honest run directories (module docstring), of ``models`` when given."""
    by_dir = recs.table("step").group_by("run_dir")
    return [d for (d,), rows in by_dir.items()
            if all(r.get("scenario") == "honest" and r.get("cheat_step") is None for r in rows)
            and (models is None or rows[0].get("model") in models)]


def cheat_dirs(recs: Records, models: Sequence[str] | None = None,
               scenarios: Sequence[str] = CHEATS) -> list[str]:
    by_dir = recs.table("step").group_by("run_dir")
    return [d for (d,), rows in by_dir.items()
            if any(r.get("scenario") in scenarios for r in rows)
            and (models is None or rows[0].get("model") in models)]


def judged_honest_steps(recs: Records, models: Sequence[str] | None = None) -> Table:
    """The step records of the honest runs' judged steps."""
    dirs = set(honest_dirs(recs, models))
    return recs.table("step").where(run_dir=lambda d: d in dirs, judged=True)


def residual_index(recs: Records) -> dict[tuple[str, str | None, int | None], Archive]:
    """Every step's residual archive by ``(run_dir, scenario, step)``."""
    return {(a.run_dir, a.scenario, a.step): a for a in recs.residuals}


def load_residuals(archive: Archive) -> dict[str, np.ndarray]:
    """The arrays of one step, ``p_layer`` in the ``0 … L`` numbering (module docstring)."""
    z = archive.load()
    layer = z.get("p_layer")
    if layer is not None and layer.size and (layer == -1).any() and (layer != -1).any() \
            and layer[layer != -1].min() >= 1:
        # written before EV2's renumbering: 1 … L for the stack, −1 for the output layer
        top = int(layer.max())
        z["p_layer"] = np.where(layer == -1, top, layer - 1).astype(layer.dtype)
    return z


def scored_families(step: dict[str, Any]) -> list[str]:
    """The families a step record scored: those whose score is not ``None``."""
    return [f for f in FAMILIES if step.get(f) is not None]


def nan_as_inf(x: np.ndarray) -> np.ndarray:
    """``x`` with every ``NaN`` as ``+inf``, as ``verification.runs.metrics.nan_as_inf`` reads
    a scored family. Copied, since a result script may not import verification; a test checks
    that the two agree."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), np.inf, x)


def mark_infinite(ax: Axes, x: Sequence[float], colour: Any, label: str | None = None) -> None:
    """Triangles at the top edge of ``ax`` above each ``x``, for infinite scores, which a log
    axis can't place."""
    if len(x):
        ax.plot(x, [0.97] * len(x), ls="none", marker="^", color=colour, clip_on=False,
                transform=ax.get_xaxis_transform(), label=label)


def mean_std(xs: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    """The mean, the sample standard deviation (``None`` below two values) and the count of
    the values that are not ``None``."""
    v = [float(x) for x in xs if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def bands(root: str | Path) -> dict[str, Any] | None:
    """The band file beside the run directories, or ``None`` if there is none."""
    path = Path(root) / "bands.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def margin_z(band_file: dict[str, Any] | None) -> float:
    """The configured margin ``z``: the band file's ``stats.z``. With no band file it is
    :data:`DEFAULT_Z`, with a warning; a band file without a finite ``z`` is an error."""
    if band_file is None:
        warnings.warn(f"no band file; using the default margin z = {DEFAULT_Z}", UserWarning,
                      stacklevel=2)
        return DEFAULT_Z
    z = (band_file.get("stats") or {}).get("z")
    try:
        ok = z is not None and not isinstance(z, bool) and math.isfinite(float(z))
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"the band file has no finite stats.z (got {z!r})")
    return float(z)


def band_tau(band_file: dict[str, Any] | None) -> float | None:
    """The band file's ``tau``, which must be one finite scalar: rescaling ``ρ_5`` by
    ``z_conf / z`` is then exact, since every ``ρ_5`` has the same ``τ = z · s_h``. ``None``
    with no band file."""
    if band_file is None:
        return None
    tau = band_file.get("tau")
    try:
        ok = not isinstance(tau, bool) and math.isfinite(float(tau))
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"the band file's tau is not one finite scalar (got {tau!r})")
    return float(tau)


def seeds_note(counts: Iterable[int]) -> str:
    """``R = 1`` or ``R = 3–5``, for a caption: the seeds behind each mean (EQ6)."""
    c = sorted(set(counts))
    if not c:
        return "R = 0"
    return f"R = {c[0]}" if len(c) == 1 else f"R = {c[0]}–{c[-1]}"
