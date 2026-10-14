"""FS11's upload, built on the Mac: the data, the gated model, the code, and a manifest.

    HF_HUB_OFFLINE=1 .venv/bin/python -m verification.runs.fs11_prep --out DIR

Writes ``DIR/fs11_upload.tar`` (top directory ``fs11_upload/``), its SHA-256 to
``DIR/fs11_upload.tar.sha256``, and a copy of its manifest to ``DIR/fs11_upload.manifest.json``.
``fs11_session`` runs on the pod from the extracted upload (``verification/README.md``,
"Workflow: the first H100 session"). The upload holds:

- ``repo/``: ``git archive HEAD`` of this repository. Uncommitted changes are not in it; the
  manifest records the commit and whether the tree was dirty.
- ``data/<model>_<task>/``: ``materialize_data``'s five files for each full-scale model and
  task, at the full-scale config (:data:`~verification.runs.fs11_session.FULL_SCALE_ENV`:
  batch 128, seed 1, 10 steps, one poisoned record), built here on the CPU. A directory whose
  ``meta.json`` already matches is reused.
- ``hf/hub/``: a Hugging Face cache holding, for each pinned model (the four run models and
  the benchmark's two others, the unsloth Llama mirror and SmolLM2), every file of its pinned
  snapshot in the Mac's cache, as blobs and the snapshot's links, and its ``.no_exist``
  entries. Refs, other revisions and the cache's ``token`` are never read.
- ``bench_cache/blake3-1.8.7/``: the BLAKE3 sources the benchmark's native build compiles.
- ``requirements-fs11.txt``: the Mac venv's versions of the packages the runs import.
- ``manifest.json``: every packed file's SHA-256, each model's files, what is missing.

**Model weights.** The pod never holds a Hugging Face token and never fetches a gated file.
A gated model (:data:`~verification.runs.fs11_session.GATED`, Meta's Llama-3.2-1B) is packed
whole. A public model's weight files are not packed: the manifest records each one's SHA-256
and size from the Mac's cache, and ``fs11_session fetch-models`` fetches it on the pod with
``token=False`` at the pinned commit and checks it against that SHA-256. Every file read from
the cache is first checked against the name of the blob it links to, which the pinned commit
fixes: the SHA-256 of a large (LFS) file, the git blob SHA-1 of a small one. A mismatch stops
the prep. A pinned snapshot or file the Mac's cache lacks is reported in ``missing`` and never
downloaded: the prep runs offline (``HF_HUB_OFFLINE=1``).

Exits 0 when every run model is packed or fetchable with its data built, 1 otherwise (the
tarball is still written, and ``missing`` says what the session will lack).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from setup.data import FULL_SCALE_TOKENIZERS, TEST_SCALE_TOKENIZER
from verification.runs.fs11_session import (FULL_SCALE_ENV, GATED, RUN_PINS, SECRETS, TASKS,
                                            _run_env, data_key, sha256_file)

__all__ = ["TOP", "BENCH_PINS", "DATA_FILES", "PrepError", "CacheFile", "ModelFiles",
           "git_blob_sha1", "blob_matches", "model_files", "data_matches", "materialize",
           "prepare", "main"]

TOP = "fs11_upload"
# The benchmark's models beyond the run's (benchmarks/hashing/sizes.MODELS): F8d's unsloth
# Llama mirror and SmolLM2. Their configs size the leaves; E6 loads their weights.
BENCH_PINS: tuple[tuple[str, str], ...] = tuple(
    (p.repo, p.revision) for p in (*FULL_SCALE_TOKENIZERS, TEST_SCALE_TOKENIZER)
    if (p.repo, p.revision) not in RUN_PINS)
DATA_FILES = ("D.bin", "D_tilde.bin", "manifest.bin", "manifest_tilde.bin", "meta.json")
# A public model's files that the pod fetches instead of the upload carrying them.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack", ".onnx")
BLAKE3_DIR = "blake3-1.8.7"
# The packages the session's parts import, whose Mac versions the upload records.
PACKAGES = ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy",
            "blake3", "datasets", "certifi")


class PrepError(RuntimeError):
    pass


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- the Hugging Face cache --------------------------------------------------------------


def git_blob_sha1(path: Path) -> str:
    """Git's object id of the file: SHA-1 over ``blob <size>\\0`` and the bytes."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_matches(path: Path, blob: str) -> tuple[bool, str]:
    """Whether the file's bytes match its blob name, and their SHA-256. The Hub stores an LFS
    file under its SHA-256 (64 hex) and a small file under its git blob SHA-1 (40 hex)."""
    sha = sha256_file(path)
    if len(blob) == 64:
        return sha == blob, sha
    if len(blob) == 40:
        return git_blob_sha1(path) == blob, sha
    return False, sha


@dataclass
class CacheFile:
    path: str  # relative to the snapshot
    sha256: str
    bytes: int
    blob: str | None  # None when the snapshot holds the file itself, not a link
    link: str | None  # the snapshot link's target, as the cache wrote it
    packed: bool


@dataclass
class ModelFiles:
    repo: str
    revision: str
    gated: bool
    roles: list[str]
    files: list[CacheFile] = field(default_factory=list)
    no_exist: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def dir_name(self) -> str:
        return f"models--{self.repo.replace('/', '--')}"


def _is_weight(rel: str) -> bool:
    return rel.endswith(WEIGHT_SUFFIXES)


def _weights_present(names: Iterable[str], snap: Path) -> bool:
    names = set(names)
    if "model.safetensors" in names:
        return True
    index = "model.safetensors.index.json"
    if index not in names:
        return False
    shards = set(json.loads((snap / index).read_text())["weight_map"].values())
    return shards <= names


def model_files(hub: Path, repo: str, revision: str, *, gated: bool,
                roles: Sequence[str]) -> ModelFiles:
    """The pinned snapshot's files in the cache at ``hub``, each checked against its blob name
    (raises :class:`PrepError` on a mismatch). A missing snapshot or missing weights go to
    ``missing``; nothing is fetched."""
    m = ModelFiles(repo, revision, gated, list(roles))
    root = hub / m.dir_name
    snap = root / "snapshots" / revision
    if not snap.is_dir():
        m.missing.append(f"{repo}@{revision}: no snapshot in the Mac's cache")
        return m
    for p in sorted(snap.rglob("*")):
        if p.is_dir() and not p.is_symlink():
            continue
        rel = p.relative_to(snap).as_posix()
        if p.is_symlink():
            link = os.readlink(p)
            target = (p.parent / link).resolve()
            if not target.is_file():
                m.missing.append(f"{repo}@{revision}/{rel}: its blob is missing")
                continue
            blob: str | None = target.name
            ok, sha = blob_matches(target, target.name)
            if not ok:
                raise PrepError(f"{repo}@{revision}/{rel}: the bytes don't match blob "
                                f"{target.name}; the cache is corrupt, not packed")
            size = target.stat().st_size
        else:
            link, blob = None, None
            sha, size = sha256_file(p), p.stat().st_size
        packed = gated or not _is_weight(rel)
        m.files.append(CacheFile(rel, sha, size, blob, link, packed))
    if not _weights_present((f.path for f in m.files), snap):
        m.missing.append(f"{repo}@{revision}: weights (model.safetensors) not in the Mac's "
                         "cache")
    ne = root / ".no_exist" / revision
    if ne.is_dir():
        m.no_exist = sorted(p.relative_to(ne).as_posix() for p in ne.rglob("*") if p.is_file())
    return m


# ---- the data ------------------------------------------------------------------------------


def _expected_meta(repo: str, revision: str, task: str) -> dict[str, Any]:
    env = FULL_SCALE_ENV
    return {"repo": repo, "revision": revision, "attack": task, "n_records": TASKS[task],
            "seed": int(env["VERIF_SEED"]), "n": int(env["VERIF_SEQ_LEN"]),
            "batch": int(env["VERIF_BATCH"]), "count": int(env["VERIF_POISON_COUNT"]),
            "steps": int(env["VERIF_STEPS"])}


def data_matches(d: Path, repo: str, revision: str, task: str) -> list[str]:
    """How the data directory ``d`` differs from what the session needs for this model and
    task; empty when it can be used as is."""
    if not all((d / f).is_file() for f in DATA_FILES):
        return [f"missing {[f for f in DATA_FILES if not (d / f).is_file()]}"]
    meta = json.loads((d / "meta.json").read_text())
    run = (meta.get("tokenizers") or {}).get("run") or {}
    pois = meta.get("poisoning") or {}
    got = {"repo": run.get("repo"), "revision": run.get("revision"),
           "attack": meta.get("attack"), "n_records": meta.get("n_records"),
           "seed": meta.get("seed"), "n": meta.get("n"), "batch": pois.get("batch"),
           "count": pois.get("count"),
           "steps": len((meta.get("schedule") or {}).get("steps") or {})}
    want = _expected_meta(repo, revision, task)
    return [f"{k} {got[k]!r} != {v!r}" for k, v in want.items() if got[k] != v]


def _materialize_env(repo: str, revision: str, task: str, out: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    env.update({k: v for k, v in _run_env(repo, revision, task).items() if v is not None})
    # The data doesn't depend on the device or dtypes; the Mac builds it on the CPU.
    env.update({"VERIF_DEVICE": "cpu", "VERIF_OUTPUT_DIR": str(out), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1"})
    return env


def materialize(repo: str, revision: str, task: str, out: Path, *, cwd: Path,
                python: str = sys.executable) -> Path:
    """Run ``materialize_data`` for one model and task into ``out/data/``; returns that
    directory. Raises :class:`PrepError` when it fails or writes the wrong data."""
    log = out / "materialize.log"
    out.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run([python, "-m", "verification.runs.materialize_data"], cwd=cwd,
                            env=_materialize_env(repo, revision, task, out), stdout=f,
                            stderr=subprocess.STDOUT).returncode
    d = out / "data"
    if rc != 0:
        raise PrepError(f"materialize_data for {repo} {task} exited {rc}; see {log}")
    bad = data_matches(d, repo, revision, task)
    if bad:
        raise PrepError(f"materialize_data for {repo} {task} wrote other data: {bad}")
    return d


# ---- the upload -----------------------------------------------------------------------------


@dataclass
class Member:
    arc: str  # relative to the upload's top directory
    src: Path | None = None  # a regular file
    link: str | None = None  # a symbolic link's target
    data: bytes | None = None  # bytes made here (an empty .no_exist entry, the manifest)


def _repo_members(repo_root: Path, build: Path) -> tuple[list[Member], dict[str, Any]]:
    """``git archive HEAD``, extracted under ``build/repo``."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    dest = build / "repo"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(repo_root), "archive", "--format=tar", "HEAD"],
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        t.extractall(dest, filter="data")
    return _tree_members(dest, "repo"), {"commit": commit, "dirty": dirty}


def _tree_members(root: Path, prefix: str) -> list[Member]:
    out = []
    for p in sorted(root.rglob("*")):
        arc = f"{prefix}/{p.relative_to(root).as_posix()}"
        if p.is_symlink():
            out.append(Member(arc, link=os.readlink(p)))
        elif p.is_file():
            out.append(Member(arc, src=p))
    return out


def _model_members(hub: Path, m: ModelFiles) -> list[Member]:
    base = f"hf/hub/{m.dir_name}"
    out, blobs = [], set()
    for f in m.files:
        if not f.packed:
            continue
        snap = f"{base}/snapshots/{m.revision}/{f.path}"
        if f.blob is None:
            out.append(Member(snap, src=hub / m.dir_name / "snapshots" / m.revision / f.path))
            continue
        if f.blob not in blobs:
            blobs.add(f.blob)
            out.append(Member(f"{base}/blobs/{f.blob}", src=hub / m.dir_name / "blobs" / f.blob))
        out.append(Member(snap, link=f.link))
    for rel in m.no_exist:
        out.append(Member(f"{base}/.no_exist/{m.revision}/{rel}", data=b""))
    return out


def _bench_members(bench_cache: Path) -> tuple[list[Member], list[str]]:
    src = bench_cache / BLAKE3_DIR
    if not src.is_dir():
        return [], [f"BLAKE3 sources not at {src}: the benchmark's native build fetches them "
                    "on the pod"]
    return _tree_members(src, f"bench_cache/{BLAKE3_DIR}"), []


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _requirements(versions: dict[str, str | None]) -> bytes:
    lines = ["# The Mac venv's versions (fs11_prep). On the pod torch must be a CUDA build;",
             "# PyPI's torch wheels for Linux x86_64 are."]
    lines += [f"{k}=={v}" for k, v in versions.items() if k != "python" and v]
    return ("\n".join(lines) + "\n").encode()


def _forbidden(arc: str) -> bool:
    """An arcname that could hold a credential: the cache's token files, or anything outside
    the pinned models' directories under ``hf/``."""
    parts = arc.split("/")
    if parts[0] != "hf":
        return False
    return len(parts) < 4 or parts[1] != "hub" or not parts[2].startswith("models--") \
        or parts[3] not in ("blobs", "snapshots", ".no_exist")


def _add(t: tarfile.TarFile, m: Member) -> None:
    name = f"{TOP}/{m.arc}"
    if m.link is not None:
        info = tarfile.TarInfo(name)
        info.type, info.linkname, info.mode = tarfile.SYMTYPE, m.link, 0o777
        info.mtime = int(time.time())
        t.addfile(info)
    elif m.data is not None:
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(m.data), 0o644, int(time.time())
        t.addfile(info, io.BytesIO(m.data))
    else:
        assert m.src is not None
        info = t.gettarinfo(str(m.src), arcname=name)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with open(m.src, "rb") as f:
            t.addfile(info, f)


def prepare(out: Path, *, hub: Path, bench_cache: Path, repo_root: Path,
            python: str = sys.executable,
            build_data: Callable[[str, str, str, Path], Path] | None = None,
            repo: Callable[[Path, Path], tuple[list[Member], dict[str, Any]]] = _repo_members,
            say: Callable[[str], None] = _say) -> dict[str, Any]:
    """Build the upload (module docstring) and return its manifest. ``build_data(repo,
    revision, task, build_dir) -> data dir`` and ``repo`` replace ``materialize_data`` and
    ``git archive`` in tests."""
    out = out.resolve()
    build = out / "build"
    t0 = time.time()
    if build_data is None:
        def build_data(r: str, rev: str, task: str, d: Path) -> Path:
            return materialize(r, rev, task, d, cwd=repo_root, python=python)

    # The models, each checked against its blob names.
    roles: dict[tuple[str, str], list[str]] = {}
    for pin in RUN_PINS:
        roles.setdefault(pin, []).append("run")
    for pin in BENCH_PINS:
        roles.setdefault(pin, []).append("benchmark")
    models = []
    for (r, rev), rl in roles.items():
        t1 = time.time()
        m = model_files(hub, r, rev, gated=r in GATED, roles=rl)
        models.append(m)
        packed = sum(f.bytes for f in m.files if f.packed)
        say(f"{r}@{rev[:8]} ({', '.join(rl)}{', gated' if m.gated else ''}): "
            f"{len(m.files)} files match their blobs, {packed / 2**30:.2f} GiB packed, "
            f"{sum(not f.packed for f in m.files)} fetched on the pod ({time.time() - t1:.1f} s)"
            + "".join(f"; MISSING {x}" for x in m.missing))
    # The data, per run model and task.
    data: dict[str, Any] = {}
    data_members: list[Member] = []
    for r, rev in RUN_PINS:
        for task in TASKS:
            key = data_key(r, task)
            d = build / key / "data"
            why = data_matches(d, r, rev, task)
            if why:
                t1 = time.time()
                d = build_data(r, rev, task, build / key)
                say(f"data {key}: built ({time.time() - t1:.0f} s)")
            else:
                say(f"data {key}: reused {d}")
            meta = json.loads((d / "meta.json").read_text())
            data[key] = {"model": r, "revision": rev, "task": task,
                         "n_records": meta["n_records"], "h_D": meta["h_D"],
                         "h_D_tilde": meta["h_D_tilde"]}
            data_members += [Member(f"data/{key}/{f}", src=d / f) for f in DATA_FILES]
    repo_members, repo_info = repo(repo_root, build)
    if repo_info.get("dirty"):
        say(f"warning: the working tree is dirty; the upload holds commit "
            f"{repo_info['commit'][:12]} without the uncommitted changes")
    bench_members, bench_missing = _bench_members(bench_cache)
    versions = _versions()
    extra = [Member("requirements-fs11.txt", data=_requirements(versions))]
    members = repo_members + data_members + bench_members + extra
    for m in models:
        members += _model_members(hub, m)
    bad = [m.arc for m in members if _forbidden(m.arc)]
    if bad:
        raise PrepError(f"refusing to pack paths outside the pinned models: {bad}")

    files: dict[str, str] = {}
    for m in members:
        if m.link is not None:
            continue
        files[m.arc] = (hashlib.sha256(m.data).hexdigest() if m.data is not None
                        else sha256_file(m.src))  # type: ignore[arg-type]
    for m in models:  # a snapshot path stands for its link and its blob together
        for f in m.files:
            if f.packed:
                files[f"hf/hub/{m.dir_name}/snapshots/{m.revision}/{f.path}"] = f.sha256
    missing = [x for m in models for x in m.missing] + bench_missing
    run_missing = [x for m in models if "run" in m.roles for x in m.missing]
    manifest = {
        "format": "fs11-upload/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo_info, "versions": versions, "full_scale_env": FULL_SCALE_ENV,
        "files": dict(sorted(files.items())),
        "models": [{"repo": m.repo, "revision": m.revision, "gated": m.gated, "roles": m.roles,
                    "files": [vars(f) for f in m.files], "no_exist": m.no_exist,
                    "missing": m.missing} for m in models],
        "data": data, "missing": missing, "run_models_complete": not run_missing,
    }
    blob = (json.dumps(manifest, indent=1) + "\n").encode()
    tar = out / f"{TOP}.tar"
    tmp = tar.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as t:
        _add(t, Member("manifest.json", data=blob))
        for m in members:
            _add(t, m)
    tmp.replace(tar)
    digest = sha256_file(tar)
    (out / f"{TOP}.tar.sha256").write_text(f"{digest}  {tar.name}\n")
    (out / f"{TOP}.manifest.json").write_bytes(blob)
    say(f"wrote {tar} ({tar.stat().st_size / 2**30:.2f} GiB, {len(members) + 1} members) in "
        f"{time.time() - t0:.0f} s")
    say(f"sha256 {digest}")
    for x in missing:
        say(f"missing: {x}")
    return {**manifest, "tarball": str(tar), "sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="where the tarball and builds go")
    p.add_argument("--hf-cache", type=Path, default=None,
                   help="the Hugging Face hub cache (default: $HF_HUB_CACHE, else "
                        "$HF_HOME/hub, else ~/.cache/huggingface/hub)")
    p.add_argument("--bench-cache", type=Path, default=None,
                   help="the benchmark's cache (default: $POTS_BENCH_CACHE, else "
                        "~/.cache/pots-benchmarks)")
    args = p.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        print("set HF_HUB_OFFLINE=1: the prep reads the local cache only", file=sys.stderr)
        return 2
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = args.hf_cache or Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    bench = args.bench_cache or Path(os.environ.get("POTS_BENCH_CACHE")
                                     or Path.home() / ".cache" / "pots-benchmarks")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        man = prepare(args.out, hub=hub, bench_cache=bench, repo_root=repo_root)
    except PrepError as e:
        print(f"fs11_prep: {e}", file=sys.stderr)
        return 1
    if not man["run_models_complete"]:
        print("fs11_prep: a run model is missing from the Mac's cache (see above)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
