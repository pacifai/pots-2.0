"""The model pin read from the local Hugging Face cache (FS4), on a fake cache."""

import hashlib
import json
import os

import pytest

from setup.model import (
    cached_file,
    cached_weight_files,
    load_model_config,
    load_pretrained,
    model_pin,
    require_commit,
)

REPO, REV = "org/model", "0123456789abcdef0123456789abcdef01234567"


def _put(cache, name, data):
    """``name`` in the snapshot at ``REV``, a link to its blob named by its SHA-256, as the Hub
    cache stores it."""
    root = cache / "models--org--model"
    sha = hashlib.sha256(data).hexdigest()
    blob = root / "blobs" / sha
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    link = root / "snapshots" / REV / name
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(blob, link.parent), link)
    return sha


def test_single_file_pin(tmp_path):
    cfg = b'{"model_type": "llama"}'
    _put(tmp_path, "config.json", cfg)
    assert model_pin(REPO, REV, tmp_path)["weights"] is None  # config only: no weights
    sha = _put(tmp_path, "model.safetensors", b"w" * 10)
    pin = model_pin(REPO, REV, tmp_path)
    assert pin == {"repo": REPO, "revision": REV,
                   "config_sha256": hashlib.sha256(cfg).hexdigest(),
                   "weights": [{"file": "model.safetensors", "sha256": sha, "bytes": 10}]}


def test_sharded_weights_need_every_shard(tmp_path):
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    index = {"weight_map": {"a": shards[0], "b": shards[1], "c": shards[0]}}
    _put(tmp_path, "model.safetensors.index.json", json.dumps(index).encode())
    _put(tmp_path, shards[0], b"x")
    assert cached_weight_files(REPO, REV, tmp_path) is None
    _put(tmp_path, shards[1], b"yy")
    assert [p.name for p in cached_weight_files(REPO, REV, tmp_path)] == shards
    pin = model_pin(REPO, REV, tmp_path)
    assert pin["config_sha256"] is None
    assert [(w["file"], w["bytes"]) for w in pin["weights"]] == [(shards[0], 1), (shards[1], 2)]


def test_other_revision_is_not_cached(tmp_path):
    _put(tmp_path, "model.safetensors", b"w")
    assert cached_weight_files(REPO, "f" * 40, tmp_path) is None
    assert model_pin("org/other", REV, tmp_path)["weights"] is None


def test_a_weight_file_must_hash_to_its_blob_name(tmp_path):
    sha = _put(tmp_path, "model.safetensors", b"w" * 10)
    assert model_pin(REPO, REV, tmp_path)["weights"][0]["sha256"] == sha
    blob = tmp_path / "models--org--model" / "blobs" / sha
    blob.write_bytes(b"w" * 9 + b"x")  # a corrupt blob keeps its name
    with pytest.raises(ValueError, match=f"model.safetensors hashes to "
                                         f"{hashlib.sha256(b'w' * 9 + b'x').hexdigest()}, "
                                         f"but its cache blob is named {sha}"):
        model_pin(REPO, REV, tmp_path)


@pytest.mark.parametrize("revision", ["main", "v1.0", REV[:12], REV.upper(), REV + "0"])
def test_a_pin_names_a_commit(tmp_path, revision):
    _put(tmp_path, "model.safetensors", b"w")
    assert require_commit(REV) == REV
    for f in (lambda: cached_file(REPO, revision, "model.safetensors", tmp_path),
              lambda: cached_weight_files(REPO, revision, tmp_path),
              lambda: model_pin(REPO, revision, tmp_path),
              lambda: load_model_config(REPO, revision),
              lambda: load_pretrained(REPO, revision)):
        with pytest.raises(ValueError, match="is not a 40-hex commit"):
            f()


def test_the_loads_never_download(monkeypatch):
    """The config and the model (its generation config with it) load from the cache only."""
    import transformers

    calls = []
    for auto in (transformers.AutoConfig, transformers.AutoModelForCausalLM):
        monkeypatch.setattr(auto, "from_pretrained",
                            lambda repo, **kw: calls.append((repo, kw)) or "loaded")
    assert load_model_config(REPO, REV) == load_pretrained(REPO, REV) == "loaded"
    assert [(repo, kw["revision"], kw["local_files_only"]) for repo, kw in calls] == \
        [(REPO, REV, True)] * 2
