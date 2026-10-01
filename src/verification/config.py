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
    eta: float
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
        """The step size `η`, a declared argument of `C` (S8e; 1e-3 at test scale)."""
        return self.eta


_ETA_DEFAULT = "1e-3"  # S8e: fixed by declaration, no tuning run


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


def _eta(env: Mapping[str, str]) -> float:
    raw = env.get("VERIF_ETA", "").strip() or _ETA_DEFAULT
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


def _attn_impl(env: Mapping[str, str]) -> str:
    raw = _str(env, "VERIF_ATTN_IMPL")
    if raw != "eager":
        # Only eager attention materializes the QKᵀ and AV product leaves, so M depends on it
        # (DECISIONS_SETUP §8.A.4).
        raise ValueError(f"VERIF_ATTN_IMPL must be 'eager' (§8.A.4), got {raw!r}")
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
        attn_impl=_attn_impl(env),
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
    if cfg.device.startswith("cuda") and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Deterministic cuBLAS reads this once, at CUDA init, so a late set is silently void (S4c).
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialized; "
                "call setup_determinism() before any CUDA work or export it."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # Reduced-precision fp32 matmul (TF32 on CUDA, bf16 on CPU oneDNN) would widen the honest
    # band past the fp32 sizing (S4c). "highest" sets both matmul backends to "ieee" and leaves
    # every precision getter working. It does not cover cuDNN convolutions, so their legacy
    # flag is set too. The per-backend `fp32_precision` setters are avoided: in torch 2.9 they
    # make the legacy `allow_tf32` getters raise.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def assert_no_dropout(model_or_config: Any) -> None:
    """Raise unless every dropout rate in the HF config and every dropout module is 0 (S4d).

    Config keys matching `*dropout*` or `*pdrop*` are checked, recursing into nested
    sub-config dicts.
    """
    if isinstance(model_or_config, nn.Module):
        config = getattr(model_or_config, "config", None)
        modules = list(model_or_config.named_modules())
    else:
        config = model_or_config
        modules = []

    bad: list[str] = []

    def scan(attrs: Mapping[str, Any], prefix: str) -> None:
        for name, value in attrs.items():
            if isinstance(value, Mapping):
                scan(value, f"{prefix}{name}.")
                continue
            lowered = str(name).lower()  # nested dicts such as id2label have int keys
            if "dropout" not in lowered and "pdrop" not in lowered:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
                bad.append(f"config.{prefix}{name}={value}")

    if config is not None:
        scan(config.to_dict() if hasattr(config, "to_dict") else vars(config), "")
    for name, module in modules:
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p != 0:
            bad.append(f"module {name or '<root>'}: p={module.p}")
    if bad:
        raise ValueError("dropout must be 0 (S4d): " + ", ".join(bad))
