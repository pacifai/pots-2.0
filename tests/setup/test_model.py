"""The model pin read from the local Hugging Face cache (FS4), on a fake cache."""

import hashlib
import json
import os

from setup.model import cached_weight_files, model_pin

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
