"""The models and their step's leaves, from the protocol's leaf-size list.

The pins are the run's (`setup.data`): the four full-scale models and the test-scale SmolLM2.
Only each model's config is loaded, never its weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from setup.data import FULL_SCALE_TOKENIZERS, TEST_SCALE_TOKENIZER
from setup.model import load_model_config
from verification.computation.leaf_sizes import LeafSize, leaf_sizes

_KEYS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5-0.5b",
    "unsloth/Llama-3.2-1B-Instruct": "llama-3.2-1b",
    "tiiuae/Falcon3-1B-Instruct": "falcon3-1b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "smollm2-135m",
}


@dataclass(frozen=True)
class ModelPin:
    key: str
    repo: str
    revision: str


MODELS: dict[str, ModelPin] = {
    _KEYS[p.repo]: ModelPin(_KEYS[p.repo], p.repo, p.revision)
    for p in (*FULL_SCALE_TOKENIZERS, TEST_SCALE_TOKENIZER)
}
# The four full-scale models, in HASHING_PLAN_DRAFT's order.
FULL_SCALE = ("qwen2.5-0.5b", "llama-3.2-1b", "falcon3-1b", "qwen2.5-1.5b")
TEST_SCALE = "smollm2-135m"


@dataclass(frozen=True)
class Step:
    """One step's leaves for a model at a batch and precision."""

    model: str
    n_s: int
    n: int
    weight_dtype: torch.dtype
    product_dtype: torch.dtype
    leaves: tuple[LeafSize, ...]

    @property
    def total_bytes(self) -> int:
        return sum(l.nbytes for l in self.leaves)

    def params(self) -> dict[str, Any]:
        """The step's identity, for a result's params."""
        return {"model": self.model, "n_s": self.n_s, "n": self.n,
                "weight_dtype": str(self.weight_dtype).removeprefix("torch."),
                "product_dtype": str(self.product_dtype).removeprefix("torch.")}

    def histogram(self) -> list[tuple[int, int]]:
        """``(leaf bytes, count)`` for each distinct leaf size, largest first."""
        return sorted(Counter(l.nbytes for l in self.leaves).items(), reverse=True)

    def summary(self) -> dict[str, Any]:
        largest = max(self.leaves, key=lambda l: l.nbytes)
        by_section = Counter()
        for l in self.leaves:
            by_section[l.section] += l.nbytes
        return {**self.params(), "leaves": len(self.leaves), "total_bytes": self.total_bytes,
                "bytes_by_section": dict(by_section), "distinct_sizes": len(self.histogram()),
                "largest_leaf": {"name": largest.name, "shape": list(largest.shape),
                                 "bytes": largest.nbytes}}


def model_config(key: str) -> Any:
    """The model's Hugging Face config at its pin (from the local cache when offline)."""
    pin = MODELS[key]
    return load_model_config(pin.repo, pin.revision)


def step(key: str, *, n_s: int, n: int, compute_dtype: torch.dtype = torch.bfloat16) -> Step:
    """The step's leaves at ``compute_dtype`` (the run's ``COMPUTE_DTYPE``): fp32 weights and
    ``compute_dtype`` products. The default is full scale's bf16 (F1a); test scale is fp32."""
    leaves = leaf_sizes(model_config(key), n_s=n_s, n=n, compute_dtype=compute_dtype)
    return Step(key, n_s, n, torch.float32, compute_dtype, tuple(leaves))
