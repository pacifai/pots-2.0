"""Protocol parameters that hold whatever the commitment or matmul check: `λ`, `G`, `u`, `k`.

`load_protocol_config()` reads `VERIF_K` and the band-file path from the environment; the
run's own settings come from `setup.config.load_config()`. Constants that belong to one
mechanism live with it: `Z`, `F_TARGET` and `C_ANTI` in `verifier/matmul_check/sizing.py`,
`SIGMA_R` in `verifier/matmul_check/challenges.py`, `TAU_W0` in `verifier/bands.py`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from setup.config import _int, _str

# Test-scale values. See verification/CLAUDE.md and the sizing appendix §9.
LAMBDA: int = 25
LOG2_G: int = 52

UNIT_ROUNDOFF: dict[torch.dtype, float] = {
    torch.float32: 2.0**-24,
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
}

_K_DEFAULT = "9"  # C1 recomputed k from the measured τ (P10d, 2026-10-05)


@dataclass(frozen=True)
class ProtocolConfig:
    k: int
    band_file: Path


def load_protocol_config(env: Mapping[str, str] | None = None) -> ProtocolConfig:
    """Read `VERIF_K` and `VERIF_OUTPUT_DIR` from `env` (default `os.environ`)."""
    env = os.environ if env is None else env
    return ProtocolConfig(
        k=_int(env, "VERIF_K", 1, default=_K_DEFAULT),
        band_file=Path(_str(env, "VERIF_OUTPUT_DIR")) / "bands.json",
    )
