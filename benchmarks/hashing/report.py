"""The report's shape: results, experiment slots, and the JSON file.

A result is one labelled number set: its parameters, its values with a unit for each, and a
status. A slot is one experiment of the plan (E0 … E6) with its results. Every slot is always
present, so a reader sees what was skipped or not built instead of a missing key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "benchmarks.hashing/1"

# A result's status. "estimated" is a number derived from measurements (an extrapolation or a
# weighted sum), with its method in `note`.
STATUSES = ("measured", "estimated", "skipped", "not_implemented")

# The experiments of HASHING_PLAN_DRAFT Section 6.1, in order, with their titles.
SLOTS: dict[str, str] = {
    "E0": "Machine record",
    "E1": "CPU hashing throughput (SHA-256 through hashlib, BLAKE3 through blake3)",
    "E2": "GPU hashing throughput (cuPQC SHA-256, BLAKE3-on-GPU proxy)",
    "E3": "GPU overlap of hashing with a bf16 GEMM loop",
    "E4": "Host-GPU copies, pinned and pageable, alone and concurrent",
    "E5a": "Challenge generation, the current Python path (P8b, conversion included)",
    "E5a_prime": "Challenge generation, CPU-parallel baselines and the native-batch proposal",
    "E5b": "Challenge generation, cuPQC SHAKE256 on the GPU",
    "E5c": "Challenge generation, cuPQC SHA-256 in counter mode on the GPU",
    "E6": "Plain bf16 SGD step time per model, eager and sdpa attention",
}


def result(name: str, status: str, *, params: dict[str, Any] | None = None,
           values: dict[str, Any] | None = None, units: dict[str, str] | None = None,
           reason: str | None = None, note: str | None = None) -> dict[str, Any]:
    """One result. Measured and estimated results carry values, each with a unit; skipped and
    not-implemented ones carry the reason."""
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    values, units = dict(values or {}), dict(units or {})
    if status in ("measured", "estimated"):
        if not values:
            raise ValueError(f"{name}: a {status} result needs values")
        if set(values) != set(units):
            raise ValueError(f"{name}: values {sorted(values)} and units {sorted(units)} differ")
    elif not reason:
        raise ValueError(f"{name}: a {status} result needs a reason")
    return {"name": name, "status": status, "params": dict(params or {}), "values": values,
            "units": units, "reason": reason, "note": note}


def slot(key: str, results: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
    """An experiment slot. Its status is the first of `STATUSES` among its results: measured if
    anything was measured, else estimated, else skipped, else not implemented."""
    if key not in SLOTS:
        raise ValueError(f"unknown slot {key!r}")
    if not results:
        raise ValueError(f"slot {key} has no results")
    status = next(s for s in STATUSES if any(r["status"] == s for r in results))
    return {"title": SLOTS[key], "status": status, "note": note, "results": results}


def not_implemented_slot(key: str, reason: str) -> dict[str, Any]:
    return slot(key, [result(key, "not_implemented", reason=reason)])


def _clean(x: Any) -> Any:
    """JSON-safe: NaN and infinities become None, tuples lists."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def check_report(doc: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless every slot is present, in order, with a valid status."""
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    slots = doc.get("slots", {})
    if list(slots) != list(SLOTS):
        raise ValueError(f"slots {list(slots)}, expected {list(SLOTS)}")
    for key, s in slots.items():
        if s.get("status") not in STATUSES or not s.get("results"):
            raise ValueError(f"slot {key}: status {s.get('status')!r}, "
                             f"{len(s.get('results', []))} results")
        for r in s["results"]:
            if r.get("status") not in STATUSES:
                raise ValueError(f"slot {key}: result {r.get('name')!r} has status "
                                 f"{r.get('status')!r}")


def write_report(doc: dict[str, Any], path: Path) -> Path:
    check_report(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(doc), indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    return path
