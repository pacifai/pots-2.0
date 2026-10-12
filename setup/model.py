"""Loading the pretrained model: an unmodified Hugging Face `from_pretrained` (§8.A.1).

Every function here reads the local Hugging Face cache only and never downloads; a model must
be downloaded beforehand, as the tokenizers are (`setup.data.load_tokenizer`). Each one that
names a model takes a pinned revision, a 40-hex commit, and raises on anything else, such as a
branch name: a branch resolves to whatever commit the cache last saw. `transformers` and `huggingface_hub` are
imported inside the functions, so importing this module stays cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


def require_commit(revision: str) -> str:
    """`revision`, which must be a full commit hash (40 lowercase hex digits); raises
    `ValueError` on a branch or tag name, or on a shortened hash."""
    if not isinstance(revision, str) or re.fullmatch("[0-9a-f]{40}", revision) is None:
        raise ValueError(f"revision {revision!r} is not a 40-hex commit: a pin names a commit, "
                         "not a branch or tag")
    return revision


def cached_file(repo: str, revision: str, filename: str,
                cache_dir: str | Path | None = None) -> Path | None:
    """`filename` of `repo` at commit `revision` in the local Hugging Face cache, or `None` if
    it isn't there. Never downloads."""
    require_commit(revision)
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


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of the file's bytes, read in chunks."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def model_pin(repo: str, revision: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """What pins a run's model (FS4): repo, commit `revision`, the SHA-256 of the cached
    `config.json`, and each cached weight file with its SHA-256 and size, or `None` when the
    weights aren't cached.

    Each weight file's SHA-256 is computed over its bytes, and `ValueError` is raised unless it
    equals the name of the cache blob the file links to. The Hub stores a large (LFS) file
    under its SHA-256, so a match means the cached bytes are the file the Hub lists under that
    hash, and a mismatch means a corrupt or partial blob, or a file not stored that way.
    Hashing reads every weight byte once: about 3 GB/s on the dev Mac, so 0.1 s for SmolLM2
    and 1.1–1.2 s for Falcon3-1B (3.3 GB)."""
    require_commit(revision)
    config = cached_file(repo, revision, "config.json", cache_dir)
    weights = cached_weight_files(repo, revision, cache_dir)
    entries = None
    if weights is not None:
        entries = []
        for p in weights:
            blob, sha = Path(os.path.realpath(p)).name, file_sha256(p)
            if sha != blob:
                raise ValueError(f"{repo}@{revision}: {p.name} hashes to {sha}, but its cache "
                                 f"blob is named {blob}; delete the blob and download it again")
            entries.append({"file": p.name, "sha256": sha, "bytes": p.stat().st_size})
    return {
        "repo": repo, "revision": revision,
        "config_sha256": None if config is None else file_sha256(config),
        "weights": entries,
    }


def load_model_config(repo: str, revision: str) -> Any:
    """The model's Hugging Face config at commit `revision`, from the local cache only."""
    require_commit(revision)
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True)


def load_pretrained(repo: str, revision: str, *, attn_impl: str = "eager",
                    dtype: torch.dtype = torch.float32) -> nn.Module:
    """The model at commit `revision`, loaded unmodified from the local cache only (its config
    and generation config included). Eager attention by default (§8.A.4)."""
    require_commit(revision)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, attn_implementation=attn_impl, dtype=dtype,
        local_files_only=True)
