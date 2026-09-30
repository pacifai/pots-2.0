"""Protocol constants, the `VERIF_*` env-var config, and the determinism knobs (S4).

Nothing here runs at import time. Entry points call `load_config()` and then
`setup_determinism(cfg)`.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Protocol constants (test scale). See src/verification/CLAUDE.md and the sizing appendix §9.
LAMBDA: int = 25
LOG2_G: int = 52
Z: float = 8.0
F_TARGET: float = 1.0
C_ANTI: float = 0.798
SIGMA_R: float = 1.0 / math.sqrt(3.0)
TAU_W0: float = 4.0

UNIT_ROUNDOFF: dict[torch.dtype, float] = {
    torch.float32: 2.0**-24,
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
}

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_DEFAULTS: dict[str, str] = {
    "VERIF_DEVICE": "cpu",
    "VERIF_MODEL": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "VERIF_MODEL_REVISION": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "VERIF_DATASET": "tatsu-lab/alpaca",
    "VERIF_DATASET_REVISION": "dce01c9b08f87459cf36a430d809084718273017",
    "VERIF_MASTER_DTYPE": "float32",
    "VERIF_COMPUTE_DTYPE": "float32",
    "VERIF_ATTN_IMPL": "eager",
    "VERIF_K": "7",
    "VERIF_BATCH": "4",
    "VERIF_SEQ_LEN": "128",
    "VERIF_STEPS": "10",
    "VERIF_N_RECORDS": "500",
    "VERIF_THREADS": "8",
    "VERIF_SEED": "0",
    "VERIF_OUTPUT_DIR": "trainer_output/verification",
}


@dataclass(frozen=True)
class VerifConfig:
    device: str
    model: str
    model_revision: str
    dataset: str
    dataset_revision: str
    master_dtype: torch.dtype
    compute_dtype: torch.dtype
    attn_impl: str
    k: int
    batch: int
    seq_len: int
    steps: int
    n_records: int
    eta: float | None
    threads: int
    seed: int
    output_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"

    @property
    def band_file(self) -> Path:
        return self.output_dir / "bands.json"

    def require_eta(self) -> float:
        if self.eta is None:
            raise RuntimeError(
                "VERIF_ETA is unset. This run needs the learning rate eta; set VERIF_ETA "
                "to a positive float (fixed by C2)."
            )
        return self.eta


def _str(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, _DEFAULTS[name]).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int(env: Mapping[str, str], name: str, minimum: int) -> int:
    raw = _str(env, name)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _dtype(env: Mapping[str, str], name: str) -> torch.dtype:
    raw = _str(env, name)
    try:
        return _DTYPES[raw]
    except KeyError:
        raise ValueError(f"{name} must be one of {sorted(_DTYPES)}, got {raw!r}") from None


def _eta(env: Mapping[str, str]) -> float | None:
    raw = env.get("VERIF_ETA", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"VERIF_ETA must be a float, got {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"VERIF_ETA must be a positive finite float, got {raw!r}")
    return value


def _device(env: Mapping[str, str]) -> str:
    raw = _str(env, "VERIF_DEVICE")
    try:
        torch.device(raw)
    except RuntimeError:
        raise ValueError(f"VERIF_DEVICE is not a valid torch device: {raw!r}") from None
    return raw


def load_config(env: Mapping[str, str] | None = None) -> VerifConfig:
    """Build the config from `env` (default `os.environ`). Unset vars take test-scale defaults."""
    env = os.environ if env is None else env
    return VerifConfig(
        device=_device(env),
        model=_str(env, "VERIF_MODEL"),
        model_revision=_str(env, "VERIF_MODEL_REVISION"),
        dataset=_str(env, "VERIF_DATASET"),
        dataset_revision=_str(env, "VERIF_DATASET_REVISION"),
        master_dtype=_dtype(env, "VERIF_MASTER_DTYPE"),
        compute_dtype=_dtype(env, "VERIF_COMPUTE_DTYPE"),
        attn_impl=_str(env, "VERIF_ATTN_IMPL"),
        k=_int(env, "VERIF_K", 1),
        batch=_int(env, "VERIF_BATCH", 1),
        seq_len=_int(env, "VERIF_SEQ_LEN", 1),
        steps=_int(env, "VERIF_STEPS", 1),
        n_records=_int(env, "VERIF_N_RECORDS", 1),
        eta=_eta(env),
        threads=_int(env, "VERIF_THREADS", 1),
        seed=_int(env, "VERIF_SEED", 0),
        output_dir=Path(_str(env, "VERIF_OUTPUT_DIR")),
    )


def setup_determinism(cfg: VerifConfig) -> None:
    """Apply the S4c knobs. Call once per process, before building models."""
    if cfg.device.startswith("cuda"):
        # Deterministic cuBLAS needs this before its first use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # TF32 would widen the honest band past the fp32 sizing (S4c).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def assert_no_dropout(model_or_config: Any) -> None:
    """Raise unless every dropout rate in the HF config and every `nn.Dropout` is 0 (S4d)."""
    if isinstance(model_or_config, nn.Module):
        config = getattr(model_or_config, "config", None)
        modules = list(model_or_config.named_modules())
    else:
        config = model_or_config
        modules = []

    bad: list[str] = []
    if config is not None:
        attrs = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        for name, value in attrs.items():
            if "dropout" not in name.lower():
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
                bad.append(f"config.{name}={value}")
    for name, module in modules:
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p != 0:
            bad.append(f"module {name or '<root>'}: p={module.p}")
    if bad:
        raise ValueError("dropout must be 0 (S4d): " + ", ".join(bad))
