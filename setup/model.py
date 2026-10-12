"""Loading the pretrained model: an unmodified Hugging Face `from_pretrained` (§8.A.1).

`transformers` and `huggingface_hub` are imported inside the functions, so importing this
module stays cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def cached_file(repo: str, revision: str, filename: str,
                cache_dir: str | Path | None = None) -> Path | None:
    """`filename` of `repo` at `revision` in the local Hugging Face cache, or `None` if it
    isn't there. Never downloads."""
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo, filename, revision=revision, cache_dir=cache_dir)
    return Path(p) if isinstance(p, str) else None


def cached_weight_files(repo: str, revision: str,
                        cache_dir: str | Path | None = None) -> list[Path] | None:
    """The model's safetensors files at `revision` in the local cache: the single
    `model.safetensors`, or every shard its index names. `None` when any is missing, so a
    caller can skip a run that would need the network. Never downloads."""
    single = cached_file(repo, revision, "model.safetensors", cache_dir)
    if single is not None:
        return [single]
    index = cached_file(repo, revision, "model.safetensors.index.json", cache_dir)
    if index is None:
        return None
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    paths = [cached_file(repo, revision, s, cache_dir) for s in shards]
    return None if any(p is None for p in paths) else [p for p in paths if p is not None]


def model_pin(repo: str, revision: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """What pins a run's model (FS4): repo, revision, the SHA-256 of the cached `config.json`,
    and each cached weight file with its SHA-256 and size, or `None` when the weights aren't
    cached. A weight file's SHA-256 is its Hub blob name (the Hub stores large files by their
    SHA-256 and checks it on download), so it is read from the cache's link, not recomputed."""
    config = cached_file(repo, revision, "config.json", cache_dir)
    weights = cached_weight_files(repo, revision, cache_dir)
    return {
        "repo": repo, "revision": revision,
        "config_sha256": (None if config is None
                          else hashlib.sha256(config.read_bytes()).hexdigest()),
        "weights": None if weights is None else [
            {"file": p.name, "sha256": Path(os.path.realpath(p)).name,
             "bytes": p.stat().st_size} for p in weights],
    }


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
