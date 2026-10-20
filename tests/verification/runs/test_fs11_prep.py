"""FS11's upload: the cache checks, what is packed and what the pod fetches, the data reuse, and
a round trip through the session's own upload check on a fake Hugging Face cache."""

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from verification.runs import fs11_prep as prep
from verification.runs import fs11_session as fs

LLAMA, FALCON, QWEN05, QWEN15 = fs.RUN_PINS
UNSLOTH, SMOL = prep.BENCH_PINS


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _put(hub: Path, repo: str, rev: str, files: dict[str, bytes], *, lfs=("model.safetensors",)):
    """One cached snapshot as huggingface_hub lays it out: blobs named by the Hub's object id,
    the snapshot's files as relative links to them, and a ref the prep must not read."""
    root = hub / f"models--{repo.replace('/', '--')}"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(exist_ok=True)
    (root / "refs" / "main").write_text(rev)
    for rel, data in files.items():
        blob = _sha(data) if rel in lfs else _git_sha1(data)
        (root / "blobs" / blob).write_bytes(data)
        link = root / "snapshots" / rev / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(root / "blobs" / blob, link.parent))
    return root


def _model(name: str) -> dict[str, bytes]:
    return {"model.safetensors": f"weights of {name}".encode() * 100,
            "config.json": f'{{"name": "{name}"}}'.encode(),
            "tokenizer.json": f"tokenizer of {name}".encode()}


@pytest.fixture
def home(tmp_path):
    """A Hugging Face home holding a token, every run model, the unsloth mirror without its
    weights, and no SmolLM2."""
    h = tmp_path / "hf"
    hub = h / "hub"
    hub.mkdir(parents=True)
    (h / "token").write_text("hf_secret")
    (h / "stored_tokens").write_text("hf_secret")
    for repo, rev in fs.RUN_PINS:
        _put(hub, repo, rev, _model(repo))
    _put(hub, LLAMA[0], LLAMA[1], {"original/params.json": b"{}"})
    _put(hub, UNSLOTH[0], UNSLOTH[1], {"config.json": b"{}", "tokenizer.json": b"tok"})
    ne = hub / f"models--{QWEN05[0].replace('/', '--')}" / ".no_exist" / QWEN05[1]
    ne.mkdir(parents=True)
    (ne / "added_tokens.json").write_bytes(b"")
    bench = tmp_path / "bench" / prep.BLAKE3_DIR
    bench.mkdir(parents=True)
    (bench / "blake3.c").write_text("int x;")
    return h


def _download(repo, filename, rev, hub):
    """``hf_hub_download``'s effect: the file lands in the cache as a blob and a link."""
    _put(hub, repo, rev, {filename: _model(repo)[filename]})
    return fs.snapshot_path(hub, repo, rev, filename)


def _fake_data(calls):
    def build(repo, rev, task, d):
        calls.append((repo, task))
        out = d / "data"
        out.mkdir(parents=True, exist_ok=True)
        for f in prep.DATA_FILES[:-1]:
            (out / f).write_bytes(f"{repo} {task} {f}".encode())
        env = fs.FULL_SCALE_ENV
        meta = {"attack": task, "n_records": fs.TASKS[task], "seed": int(env["VERIF_SEED"]),
                "n": int(env["VERIF_SEQ_LEN"]), "h_D": _sha(f"{repo}{task}".encode()),
                "h_D_tilde": _sha(f"~{repo}{task}".encode()),
                "tokenizers": {"run": {"repo": repo, "revision": rev}},
                "poisoning": {"batch": int(env["VERIF_BATCH"]),
                              "count": int(env["VERIF_POISON_COUNT"])},
                "schedule": {"steps": {str(t): [] for t in range(1, 11)}}}
        (out / "meta.json").write_text(json.dumps(meta))
        return out
    return build


def _fake_test_data(calls):
    """The same, at test scale's config (the defaults of ``setup/config.py``)."""
    def build(repo, rev, task, d):
        calls.append((repo, task))
        out = d / "data"
        out.mkdir(parents=True, exist_ok=True)
        for f in prep.DATA_FILES[:-1]:
            (out / f).write_bytes(f"{repo} {task} {f} test scale".encode())
        c = prep.load_config({"VERIF_OUTPUT_DIR": "-"})
        meta = {"attack": task, "n_records": c.n_records, "seed": c.seed, "n": c.seq_len,
                "h_D": _sha(f"{repo}{task}ts".encode()),
                "h_D_tilde": _sha(f"~{repo}{task}ts".encode()),
                "tokenizers": {"run": {"repo": repo, "revision": rev}},
                "poisoning": {"batch": c.batch, "count": c.poison_count},
                "schedule": {"steps": {str(t): [] for t in range(1, c.steps + 1)}}}
        (out / "meta.json").write_text(json.dumps(meta))
        return out
    return build


def _fake_repo(src: Path):
    def repo(root, build):
        (src / "verification").mkdir(parents=True, exist_ok=True)
        (src / "verification" / "x.py").write_text("x = 1\n")
        return prep._tree_members(src, "repo"), {"commit": "c0ffee" * 6 + "abcd", "dirty": False}
    return repo


def _native_copy(tmp_path) -> Path:
    """A copy of the Mac's native sources (the real cache is only read)."""
    dst = tmp_path / "native_src" / prep.native.source_dir().name
    if not dst.exists():
        shutil.copytree(prep.native.source_dir(), dst)
    return dst


def _prepare(tmp_path, home, calls=None, **kw):
    kw.setdefault("native_src", _native_copy(tmp_path))
    kw.setdefault("build_test_data", _fake_test_data([]))
    return prep.prepare(tmp_path / "out", hub=home / "hub", bench_cache=tmp_path / "bench",
                        repo_root=tmp_path, build_data=_fake_data([] if calls is None else calls),
                        repo=_fake_repo(tmp_path / "src"), say=lambda m: None, **kw)


# ---- the cache -----------------------------------------------------------------------------


def test_blob_matches(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello")
    assert prep.blob_matches(p, _sha(b"hello")) == (True, _sha(b"hello"))
    assert prep.blob_matches(p, _git_sha1(b"hello"))[0]
    assert prep.git_blob_sha1(p) == _git_sha1(b"hello")
    assert not prep.blob_matches(p, _sha(b"other"))[0]
    assert not prep.blob_matches(p, _git_sha1(b"other"))[0]
    assert not prep.blob_matches(p, "main")[0]


def test_gated_model_is_packed_whole_and_public_weights_are_not(home):
    gated = prep.model_files(home / "hub", *LLAMA, gated=True, roles=["run"])
    assert {f.path for f in gated.files} == {"model.safetensors", "config.json",
                                              "tokenizer.json", "original/params.json"}
    assert all(f.packed for f in gated.files) and not gated.missing
    public = prep.model_files(home / "hub", *FALCON, gated=False, roles=["run"])
    packed = {f.path: f.packed for f in public.files}
    assert packed == {"model.safetensors": False, "config.json": True, "tokenizer.json": True}
    w = next(f for f in public.files if f.path == "model.safetensors")
    assert w.sha256 == _sha(_model(FALCON[0])["model.safetensors"]) == w.blob
    assert w.bytes == len(_model(FALCON[0])["model.safetensors"])


def test_a_corrupt_blob_stops_the_prep(home):
    root = home / "hub" / f"models--{FALCON[0].replace('/', '--')}"
    blob = Path(os.path.realpath(root / "snapshots" / FALCON[1] / "config.json"))
    blob.write_bytes(b"tampered")
    with pytest.raises(prep.PrepError, match="don't match blob"):
        prep.model_files(home / "hub", *FALCON, gated=False, roles=["run"])


def test_missing_snapshot_and_weights_are_reported(home):
    smol = prep.model_files(home / "hub", *SMOL, gated=False, roles=["benchmark"])
    assert not smol.files and "no snapshot" in smol.missing[0]
    unsloth = prep.model_files(home / "hub", *UNSLOTH, gated=False, roles=["benchmark"])
    assert len(unsloth.files) == 2 and "weights" in unsloth.missing[0]
    # E6 times Meta's weights in the mirror's place, so the prep asks it for its config only.
    assert UNSLOTH in prep.CONFIG_ONLY and SMOL not in prep.CONFIG_ONLY
    unsloth = prep.model_files(home / "hub", *UNSLOTH, gated=False, roles=["benchmark"],
                               needs_weights=False)
    assert len(unsloth.files) == 2 and not unsloth.missing


def test_forbidden_paths():
    assert prep._forbidden("hf/token") and prep._forbidden("hf/stored_tokens")
    assert prep._forbidden("hf/hub/models--x/refs/main")
    assert not prep._forbidden("hf/hub/models--x/blobs/abc")
    assert not prep._forbidden("repo/hf/token")


def test_data_matches(tmp_path):
    d = _fake_data([])(*QWEN05, "jailbreak", tmp_path)
    assert prep.data_matches(d, *QWEN05, "jailbreak") == []
    assert prep.data_matches(d, *QWEN15, "jailbreak")
    assert prep.data_matches(d, *QWEN05, "refusal")
    (d / "D.bin").unlink()
    assert "missing" in prep.data_matches(d, *QWEN05, "jailbreak")[0]


# ---- the upload ------------------------------------------------------------------------------


def test_prepare_round_trips_through_the_session_check(tmp_path, home):
    calls = []
    man = _prepare(tmp_path, home, calls)
    assert len(calls) == 8 and man["run_models_complete"]
    assert sorted(man["data"]) == sorted(fs.data_key(r, t) for r, _ in fs.RUN_PINS
                                         for t in fs.TASKS)
    assert not any("unsloth" in x for x in man["missing"])
    assert any("SmolLM2" in x for x in man["missing"])
    tar = Path(man["tarball"])
    assert (tmp_path / "out" / "fs11_upload.tar.sha256").read_text().split()[0] == \
        man["sha256"] == _sha(tar.read_bytes())

    with tarfile.open(tar) as t:
        names = t.getnames()
        assert all(n.startswith(f"{prep.TOP}/") for n in names)
        assert not any(n.endswith(("/hf/token", "/hf/stored_tokens")) for n in names)
        assert not any("/refs/" in n for n in names)
        weights = [n for n in names if n.endswith("model.safetensors")]
        assert weights == [f"{prep.TOP}/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
                           f"snapshots/{LLAMA[1]}/model.safetensors"]
        t.extractall(tmp_path / "pod", filter="data")
    up = tmp_path / "pod" / prep.TOP
    assert (up / "requirements-fs11.txt").read_text().count("torch==") == 1
    assert (up / "hf/hub" / f"models--Qwen--Qwen2.5-0.5B-Instruct/.no_exist/{QWEN05[1]}"
            / "added_tokens.json").is_file()
    assert (up / "bench_cache" / prep.BLAKE3_DIR / "blake3.c").is_file()

    # On the pod: every packed file checks out and each public weight is fetched once.
    fetched = []

    def download(repo, filename, rev, hub):
        fetched.append(repo)
        return _download(repo, filename, rev, hub)

    out = tmp_path / "models"
    assert fs.fetch_models(up, out, download=download, say=lambda m: None) == 0
    assert sorted(fetched) == sorted([FALCON[0], QWEN05[0], QWEN15[0]])
    doc = json.loads((out / "models.json").read_text())
    assert doc["problems"] == [] and doc["files_checked"] == len(man["files"])
    # a second check finds the weights present and fetches nothing
    assert fs.fetch_models(up, out, download=download, say=lambda m: None) == 0
    assert len(fetched) == 3


def test_prepare_packs_smollm2_data_at_test_scale(tmp_path, home):
    calls, test_calls = [], []
    man = _prepare(tmp_path, home, calls, build_test_data=_fake_test_data(test_calls))
    key = fs.data_key(*prep.TEST_SCALE_PIN[:1], "refusal")
    assert key == "SmolLM2-135M-Instruct_refusal" and test_calls == [(prep.TEST_SCALE_PIN[0],
                                                                      "refusal")]
    assert len(calls) == 8 and key not in man["data"]
    assert man["data_test_scale"][key]["model"] == prep.TEST_SCALE_PIN[0]
    assert all(f"data/{key}/{f}" in man["files"] for f in prep.DATA_FILES)
    d = tmp_path / "out" / "build" / key / "data"
    assert prep.data_matches(d, *prep.TEST_SCALE_PIN, "refusal", test_scale=True) == []
    assert prep.data_matches(d, *prep.TEST_SCALE_PIN, "refusal")  # not the full-scale config
    # a second prep reuses it, and data built at another config is rebuilt
    test_calls.clear()
    _prepare(tmp_path, home, build_test_data=_fake_test_data(test_calls))
    assert test_calls == []
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps({**meta, "batch_note": 1, "n": meta["n"] + 1}))
    _prepare(tmp_path, home, build_test_data=_fake_test_data(test_calls))
    assert len(test_calls) == 1


def test_prepare_refuses_smollm2_data_at_the_wrong_config(tmp_path, home):
    def wrong(repo, rev, task, d):
        out = _fake_test_data([])(repo, rev, task, d)
        meta = json.loads((out / "meta.json").read_text())
        (out / "meta.json").write_text(json.dumps({**meta, "seed": meta["seed"] + 1}))
        return out

    with pytest.raises(prep.PrepError, match="test scale"):
        _prepare(tmp_path, home, build_test_data=wrong)


def test_prepare_reuses_matching_data_and_reports_a_missing_run_model(tmp_path, home):
    _prepare(tmp_path, home)
    shutil.rmtree(home / "hub" / f"models--{QWEN15[0].replace('/', '--')}")
    calls = []
    man = _prepare(tmp_path, home, calls)
    assert calls == []
    assert not man["run_models_complete"]
    assert any(QWEN15[0] in x for x in man["missing"])


def test_the_session_check_catches_a_changed_file(tmp_path, home):
    man = _prepare(tmp_path, home)
    with tarfile.open(man["tarball"]) as t:
        t.extractall(tmp_path / "pod", filter="data")
    up = tmp_path / "pod" / prep.TOP
    key = fs.data_key(LLAMA[0], "refusal")
    (up / "data" / key / "D.bin").write_bytes(b"changed")
    assert fs.fetch_models(up, tmp_path / "m", download=_download, say=lambda m: None) == 1
    doc = json.loads((tmp_path / "m" / "models.json").read_text())
    assert any(f"data/{key}/D.bin" in p for p in doc["problems"])


def test_prepare_packs_the_native_sources_only(tmp_path, home):
    src = _native_copy(tmp_path)
    (src / "libpots_native-x.so").write_bytes(b"built")
    man = _prepare(tmp_path, home, native_src=src)
    assert not any("native" in x for x in man["missing"])
    mine = {a for a in man["files"] if a.startswith("native_cache/")}
    assert mine and all(a.startswith("native_cache/blake3-1.8.7/") for a in mine)
    assert not any(a.endswith(".so") for a in man["files"])
    assert "native_cache/blake3-1.8.7/blake3.c" in mine
    with tarfile.open(man["tarball"]) as t:
        assert not any(n.endswith(".so") for n in t.getnames())
        t.extractall(tmp_path / "pod", filter="data")
    assert prep.native.bad_sources(tmp_path / "pod" / prep.TOP / "native_cache" / "blake3-1.8.7") \
        == []


def test_prepare_reports_missing_or_altered_native_sources(tmp_path, home):
    members, missing = prep._native_members(tmp_path / "nowhere")
    assert members == [] and "python -m verification.commitment.native" in missing[0]
    src = _native_copy(tmp_path)
    (src / "blake3.c").write_text("changed")
    members, missing = prep._native_members(src)
    assert members == [] and "blake3.c" in missing[0]
    man = _prepare(tmp_path, home, native_src=src)
    assert any("blake3.c" in x for x in man["missing"])
    assert not any(a.startswith("native_cache/") for a in man["files"])
