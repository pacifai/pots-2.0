"""FS11's session: the parts' order, logs, markers and deadline; the upload check and the
public-weight fetch; and the job script's always-terminate paths in a dry run."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.verification.full_scale_helpers import PINS
from verification.runs import fs11_session as fs

JOB = Path(fs.__file__).with_name("fs11_job.sh")
REPO = Path(fs.__file__).resolve().parents[2]


def _py(code):
    return ["{python}", "-c", code]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- the plan ----------------------------------------------------------------------------


def test_run_pins_are_the_full_scale_models():
    assert set(fs.RUN_PINS) == set(PINS)
    assert fs.GATED <= {repo for repo, _ in fs.RUN_PINS}


def test_default_plan(tmp_path):
    plan = fs.default_plan(tmp_path / "up", tmp_path / "out")
    names = [p.name for p in plan]
    assert names[:2] == ["models", "benchmark"]
    bring = [p for p in plan if p.kind == "bring_up"]
    smooth = [p for p in plan if p.kind == "smooth"]
    assert len(bring) == 4 and len(smooth) == 8 and len(plan) == 14
    assert names.index(bring[-1].name) < names.index(smooth[0].name)
    assert [p.info["model"] for p in bring] == [repo for repo, _ in fs.RUN_PINS]
    for p in bring:
        assert "--no-tamper" in p.argv and p.env["VERIF_ATTACK"] == "refusal"
    for p in (*bring, *smooth):
        assert {k: p.env[k] for k in fs.FULL_SCALE_ENV} == fs.FULL_SCALE_ENV
        assert p.env["VERIF_N_RECORDS"] == str(fs.TASKS[p.env["VERIF_ATTACK"]])
        d = Path(p.env["VERIF_OUTPUT_DIR"])
        assert (d / "data").resolve() == (tmp_path / "up" / "data" / d.name).resolve()
    assert {(p.info["model"], p.info["task"]) for p in smooth} == {
        (repo, task) for repo, _ in fs.RUN_PINS for task in fs.TASKS}
    # only the fetch reaches the network
    assert plan[0].env["HF_HUB_OFFLINE"] is None
    assert all("HF_HUB_OFFLINE" not in p.env for p in plan[1:])


def test_refusal():
    env = {"cuda_available": True, "gpus": ["NVIDIA H100 80GB HBM3"]}
    assert fs.refusal(env, False) is None
    assert "A40" in fs.refusal({**env, "gpus": ["NVIDIA A40"]}, False)
    assert "is_available" in fs.refusal({"cuda_available": False, "gpus": []}, False)
    assert fs.refusal({"cuda_available": False, "gpus": []}, True) is None


def test_cupqc_status(monkeypatch, tmp_path):
    monkeypatch.delenv(fs.CUPQC_ENV, raising=False)
    assert not fs.cupqc_status()["found"] and "not set" in fs.cupqc_status()["reason"]
    monkeypatch.setenv(fs.CUPQC_ENV, str(tmp_path))
    s = fs.cupqc_status()
    assert s["found"] and "no cuPQC extension" in s["reason"] and "E5c" in s["slots"]


# ---- the session ------------------------------------------------------------------------


ENV_PROBE = ("import os, json; print(json.dumps({k: os.environ.get(k) for k in "
             "('HF_TOKEN', 'RUNPOD_API_KEY', 'HF_HUB_OFFLINE', 'HF_HOME', 'FS11_X')}))")


def test_run_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    up, out = tmp_path / "up", tmp_path / "out"
    plan = [
        fs.Part("ok", _py(ENV_PROBE), env={"FS11_X": "x"}),
        fs.Part("fails", _py("import sys; sys.exit(3)")),
        fs.Part("too_long", ["sleep", "30"], timeout_s=1),
        fs.Part("cut", ["sleep", "30"]),
        fs.Part("after", _py("print(1)")),
    ]
    monkeypatch.setattr(fs, "KILL_GRACE_S", 2)
    lines = []
    s = fs.run_session(plan, upload=up, out=out, deadline=time.time() + 4, cwd=REPO,
                       say=lines.append)
    st = {p["name"]: (p["status"], p["exit_code"]) for p in s["parts"]}
    assert st["ok"] == ("ok", 0) and st["fails"] == ("failed", 3)
    assert st["too_long"][0] == "killed_timeout" and st["cut"][0] == "killed_deadline"
    assert st["after"] == ("not_started", None)
    assert s["failed"] == ["fails", "too_long", "cut", "after"]
    assert json.loads((out / "summary.json").read_text())["parts"] == s["parts"]
    assert [p.name for p in (out / "parts").glob("*/DONE")] == ["DONE"]
    assert (out / "parts" / "ok" / "DONE").exists()
    seen = json.loads((out / "parts" / "ok" / "log.txt").read_text().splitlines()[-1])
    assert seen == {"HF_TOKEN": None, "RUNPOD_API_KEY": None, "HF_HUB_OFFLINE": "1",
                    "HF_HOME": str(up / "hf"), "FS11_X": "x"}
    assert any("cuPQC" in l for l in lines)

    # a rerun skips the finished part and runs the others again
    s = fs.run_session(plan[:2], upload=up, out=out, deadline=time.time() + 30, cwd=REPO,
                       say=lines.append)
    assert [p["status"] for p in s["parts"]] == ["skipped_done", "failed"]


def test_part_extras(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("pin: x\nstep 1: prover 3.10 s (prove_step 2.00 s, commit 1.10 s); verifier "
                   "9.00 s ()\nstep 1: prover peak 41.20 GB (at its start 3.00 GB) "
                   "[cuda_allocated, memory pass]\nstep 1: verifier peak 30.50 GB "
                   "[cuda_allocated, memory pass]\nprocess lifetime peak RSS 12.00 GB\n")
    e = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)
    assert e["peaks"] == [{"step": 1, "side": "prover", "peak_gb": 41.2},
                          {"step": 1, "side": "verifier", "peak_gb": 30.5}]
    assert len(e["lines"]) == 4
    rep = tmp_path / "hashing.json"
    rep.write_text(json.dumps({"complete": True, "slots": {
        "E1": {"results": [{"name": "chunked", "status": "measured"}]},
        "E2": {"results": [{"name": "cupqc_sha256", "status": "skipped", "reason": "licence"}]},
    }}))
    e = fs.part_extras(fs.Part("bench", [], kind="benchmark", info={"report": str(rep)}), log)
    assert e["complete"] and e["skipped"] == [{"slot": "E2", "result": "cupqc_sha256",
                                               "status": "skipped", "reason": "licence"}]
    e = fs.part_extras(fs.Part("s", [], kind="smooth", info={"report": str(tmp_path / "no")}),
                       log)
    assert e == {"record": None}


# ---- fetch-models -----------------------------------------------------------------------


def _upload(tmp_path):
    """An upload with a packed file, one public weight to fetch, and a gated model whose
    weight is (wrongly) not packed."""
    up = tmp_path / "up"
    packed = b"config"
    (up / "repo").mkdir(parents=True)
    (up / "repo" / "a.txt").write_bytes(packed)
    weight = b"public weights"
    manifest = {
        "files": {"repo/a.txt": _sha(packed)},
        "models": [
            {"repo": "org/public", "revision": "a" * 40, "gated": False, "files": [
                {"path": "model.safetensors", "sha256": _sha(weight), "bytes": len(weight),
                 "packed": False}]},
            {"repo": "meta-llama/Llama-3.2-1B-Instruct", "revision": "b" * 40, "gated": True,
             "files": [{"path": "model.safetensors", "sha256": _sha(b"w"), "bytes": 1,
                        "packed": False}]},
        ],
        "missing": [],
    }
    (up / "manifest.json").write_text(json.dumps(manifest))
    return up, weight


def _fake_download(content, calls):
    def download(repo, filename, revision, hub):
        calls.append(repo)
        blob = hub / f"models--{repo.replace('/', '--')}" / "blobs" / "x"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        p = fs.snapshot_path(hub, repo, revision, filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(os.path.relpath(blob, p.parent))
        return p
    return download


def test_fetch_models(tmp_path):
    up, weight = _upload(tmp_path)
    calls = []
    rc = fs.fetch_models(up, tmp_path / "m", download=_fake_download(weight, calls),
                         say=lambda m: None)
    doc = json.loads((tmp_path / "m" / "models.json").read_text())
    assert calls == ["org/public"]  # the gated repo is never fetched
    assert rc == 1 and len(doc["problems"]) == 1 and "gated" in doc["problems"][0]
    assert doc["fetched"] == [{"repo": "org/public", "path": "model.safetensors",
                               "status": "fetched", "bytes": len(weight),
                               "s": doc["fetched"][0]["s"]}]
    # a second pass finds the file in place
    fs.fetch_models(up, tmp_path / "m", download=_fake_download(weight, calls),
                    say=lambda m: None)
    assert calls == ["org/public"]


def test_fetch_models_deletes_a_mismatch_and_flags_a_changed_upload(tmp_path):
    up, _ = _upload(tmp_path)
    m = json.loads((up / "manifest.json").read_text())
    m["models"].pop()  # the gated one
    (up / "manifest.json").write_text(json.dumps(m))
    (up / "repo" / "a.txt").write_bytes(b"changed")
    rc = fs.fetch_models(up, tmp_path / "m", download=_fake_download(b"other", []),
                         say=lambda m: None)
    doc = json.loads((tmp_path / "m" / "models.json").read_text())
    assert rc == 1 and doc["fetched"][0]["status"] == "mismatch"
    assert any("repo/a.txt" in p for p in doc["problems"])
    snap = fs.snapshot_path(up / "hf" / "hub", "org/public", "a" * 40, "model.safetensors")
    assert not snap.exists() and not snap.is_symlink()
    assert not list((up / "hf" / "hub" / "models--org--public" / "blobs").iterdir())


# ---- the job script, dry run ---------------------------------------------------------------


def _job(tmp_path, *, plan=None, python=sys.executable, deadline=6, grace=1, reserve=1,
         extra=None, timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_POD_ID", "RUNPOD_API_KEY")}
    env.update({"DRY_RUN": "1", "FS11_DEADLINE_S": str(deadline), "FS11_GRACE_S": str(grace),
                "FS11_PACK_RESERVE_S": str(reserve), "FS11_PYTHON": python, **(extra or {})})
    if plan is not None:
        p = tmp_path / "plan.json"
        p.write_text(json.dumps(plan))
        env["FS11_PLAN"] = str(p)
    (tmp_path / "up").mkdir(exist_ok=True)
    out = tmp_path / "out"
    r = subprocess.run(["bash", str(JOB), str(tmp_path / "up"), str(out)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r, out


def test_job_runs_packs_and_terminates(tmp_path):
    plan = [{"name": "ok", "argv": _py("print('part ran')")},
            {"name": "fails", "argv": _py("import sys; sys.exit(3)")},
            {"name": "slow", "argv": ["sleep", "60"]}]
    r, out = _job(tmp_path, plan=plan, deadline=8, extra={"FS11_ALLOW_ANY_GPU": "1"})
    assert r.returncode == 1, r.stdout + r.stderr
    s = json.loads((out / "summary.json").read_text())
    assert [p["status"] for p in s["parts"]] == ["ok", "failed", "killed_deadline"]
    ready = dict(l.split("=", 1) for l in Path(f"{out}.READY").read_text().splitlines())
    tar = Path(f"{out}.tar")
    assert ready["sha256"] == _sha(tar.read_bytes()) and ready["session_exit"] == "1"
    assert ready["sha256"] in r.stdout
    assert "DRY_RUN: terminate (end of job (session exit 1)): runpodctl pod delete" in r.stdout
    assert Path(f"{out}.TERMINATED").exists()
    import tarfile
    with tarfile.open(tar) as t:
        assert "out/summary.json" in t.getnames() and "out/parts/ok/log.txt" in t.getnames()


def test_job_terminates_when_the_session_refuses(tmp_path):
    """No H100 and no override: the session refuses, and the job still packs and ends."""
    r, out = _job(tmp_path, plan=[{"name": "ok", "argv": _py("print(1)")}],
                  extra={"FS11_ALLOW_ANY_GPU": "0"})
    if "H100 check passes" in r.stdout:
        pytest.skip("this machine has an H100")
    assert r.returncode == fs.EXIT_REFUSED, r.stdout + r.stderr
    assert json.loads((out / "summary.json").read_text())["refused"]
    assert Path(f"{out}.tar").exists() and "DRY_RUN: terminate (end of job" in r.stdout


def test_job_timer_terminates_a_hung_session(tmp_path):
    hung = tmp_path / "hung.sh"
    hung.write_text("#!/bin/sh\nexec sleep 60\n")
    hung.chmod(0o755)
    t0 = time.time()
    r, out = _job(tmp_path, python=str(hung), deadline=3)
    assert time.time() - t0 < 20
    assert "deadline reached" in r.stdout and "DRY_RUN: terminate (deadline)" in r.stdout
    assert r.stdout.count("DRY_RUN: terminate") == 1  # once, though the job's trap ran too
    assert not Path(f"{out}.tar").exists()


def test_job_refuses_without_a_way_to_terminate(tmp_path):
    r, out = _job(tmp_path, extra={"DRY_RUN": "0"})
    assert r.returncode == 2 and "RUNPOD_POD_ID is not set" in r.stdout
    assert not out.exists()
