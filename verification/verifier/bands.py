"""The tolerances of checks 5 and 6 and the band file that pins them (P3, P5, P10).

Bands are the verifier's own knowledge: provisional bands built in code, or a band file that
calibration (A11) writes and every later run loads. They are outside every commitment.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import blake3

from verification.verifier.matmul_check.sizing import Z

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation, ProductSpec

__all__ = ["BAND_FILE_VERSION", "KAPPA_PROVISIONAL", "TAU_W0", "Bands", "product_class"]

TAU_W0: float = 4.0  # analytic floor of τ_W (P5a)

# A10's "loose κ": a ceiling far above any honest cancellation factor, so it guards only
# against the pathological regime of P3.c until A11 calibrates κ_max per class.
KAPPA_PROVISIONAL: float = 1e4

BAND_FILE_VERSION = 1


def _frozen(m: Mapping[str, float] | None) -> Mapping[str, float]:
    return MappingProxyType({str(k): float(v) for k, v in (m or {}).items()})


@dataclass(frozen=True)
class Bands:
    """The tolerances of checks 5 and 6 (P3, P5, P10).

    - ``tau``: the one dimensionless ``τ`` of check 5 (P3.a, P3.b).
    - ``kappa_max``: the default cancellation ceiling; ``kappa_classes`` overrides it per
      product class (P3.c), keyed by :func:`product_class`.
    - ``tau_w``: the default ``τ_W`` of check 6; ``tau_w_tensors`` overrides it per weight
      (P5: ``τ_W = max(τ_W⁰, 2·ρ_max)``, so every value is at least ``TAU_W0``).
    - ``source``: the BLAKE3 hex of the band file the bands were loaded from, or
      ``"provisional"`` for bands built in code, which judge only on opt-in (P10a).
      P10a's harness asserts that every run records the same value.

    A11 writes the band file with :meth:`to_json` and every later run loads it with
    :meth:`from_file`. A11 may extend the schema; ``stats`` is carried through unread.
    """

    tau: float
    kappa_max: float
    tau_w: float = TAU_W0
    kappa_classes: Mapping[str, float] = field(default_factory=dict)
    tau_w_tensors: Mapping[str, float] = field(default_factory=dict)
    source: str = "provisional"
    stats: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kappa_classes", _frozen(self.kappa_classes))
        object.__setattr__(self, "tau_w_tensors", _frozen(self.tau_w_tensors))
        object.__setattr__(self, "stats", MappingProxyType(dict(self.stats)))
        if not (math.isfinite(self.tau) and self.tau > 0):
            raise ValueError(f"tau must be positive and finite, got {self.tau}")
        for name, v in [("kappa_max", self.kappa_max), *self.kappa_classes.items()]:
            if not (math.isfinite(v) and v >= 1):
                raise ValueError(f"κ_max {name} = {v}: a cancellation factor is ≥ 1 (P3.c)")
        for name, v in [("tau_w", self.tau_w), *self.tau_w_tensors.items()]:
            if not (math.isfinite(v) and v >= TAU_W0):
                raise ValueError(f"τ_W {name} = {v} is below the analytic floor {TAU_W0} (P5a)")

    @classmethod
    def provisional(cls) -> Bands:
        """A10's provisional bands: ``τ = Z·1 = 8``, a loose ``κ``, ``τ_W = τ_W⁰ = 4``.

        ``τ = Z·s_h`` with ``s_h`` taken as the error model's nominal 1 (P3.a). None of the
        three depends on scale or precision. Their ``source`` is ``"provisional"``, which the
        verifier refuses unless the caller opts in: they never judge a cheat run (P10a).
        """
        return cls(tau=Z * 1.0, kappa_max=KAPPA_PROVISIONAL, tau_w=TAU_W0)

    def check_keys(self, c: DeclaredComputation) -> None:
        """Every κ class and τ_W tensor the bands name must exist in ``C``, so a misspelt key
        can't silently fall back to the default."""
        classes = {product_class(c, p) for p in c.products}
        unknown = sorted(set(self.kappa_classes) - classes)
        if unknown:
            raise ValueError(f"bands name κ classes that match no product of C: {unknown}")
        unknown = sorted(set(self.tau_w_tensors) - set(c.weight_names))
        if unknown:
            raise ValueError(f"bands name τ_W tensors that match no weight of C: {unknown}")

    def kappa_for(self, cls_key: str) -> float:
        return self.kappa_classes.get(cls_key, self.kappa_max)

    def tau_w_for(self, weight: str) -> float:
        return self.tau_w_tensors.get(weight, self.tau_w)

    # ---- the band file (P10a). Outside every commitment, so JSON is fine (S9c) -------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BAND_FILE_VERSION,
            "tau": self.tau,
            "kappa_max": {"default": self.kappa_max, "classes": dict(self.kappa_classes)},
            "tau_w": {"default": self.tau_w, "tensors": dict(self.tau_w_tensors)},
            "stats": dict(self.stats),
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1).encode()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, source: str) -> Bands:
        if d.get("version") != BAND_FILE_VERSION:
            raise ValueError(f"band file version {d.get('version')!r}, "
                             f"expected {BAND_FILE_VERSION}")
        return cls(tau=float(d["tau"]),
                   kappa_max=float(d["kappa_max"]["default"]),
                   kappa_classes=d["kappa_max"].get("classes", {}),
                   tau_w=float(d["tau_w"]["default"]),
                   tau_w_tensors=d["tau_w"].get("tensors", {}),
                   source=source, stats=d.get("stats", {}))

    @classmethod
    def from_json(cls, b: bytes) -> Bands:
        """Bands from band-file bytes; ``source`` is the BLAKE3 hex of exactly these bytes."""
        return cls.from_dict(json.loads(b), source=blake3.blake3(b).hexdigest())

    @classmethod
    def from_file(cls, path: str | Path) -> Bands:
        return cls.from_json(Path(path).read_bytes())


# ---- κ classes ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def product_class(c: DeclaredComputation, spec: ProductSpec) -> str:
    """The calibration class of a product: its role in ``C`` (spec §9, P3.b, P3.c).

    ``C`` may declare classes with a ``product_class(spec)`` method. Otherwise the class is the
    product kind with the role, the weight name (or, for a weight-free product, the product
    name) with every digit run replaced by ``*``, so the same role in every layer is one class.
    """
    own = getattr(c, "product_class", None)
    if callable(own):
        return str(own(spec))
    role = spec.weight if spec.weight is not None else spec.name
    return f"{spec.kind.value}:{_DIGITS.sub('*', role)}"
