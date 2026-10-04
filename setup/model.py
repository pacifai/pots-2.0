"""Loading the pretrained model: an unmodified Hugging Face `from_pretrained` (§8.A.1).

`transformers` is imported inside the functions, so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def load_model_config(repo: str, revision: str) -> Any:
    """The model's Hugging Face config at a pinned revision."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo, revision=revision)


def load_pretrained(repo: str, revision: str, *, attn_impl: str = "eager",
                    dtype: torch.dtype = torch.float32) -> nn.Module:
    """The model at a pinned revision, loaded unmodified. Eager attention by default (§8.A.4)."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, attn_implementation=attn_impl, dtype=dtype)
