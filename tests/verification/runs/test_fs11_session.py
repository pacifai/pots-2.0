"""FS11's session: the parts' order, logs, markers and deadline; the upload check and the
public-weight fetch; the job script's always-terminate paths, in a dry run and against stub
runpodctl and curl; and the billing guards."""

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from tests.verification.full_scale_helpers import PINS
from verification.runs import fs11_session as fs

JOB = Path(fs.__file__).with_name("fs11_job.sh")
GUARD = Path(fs.__file__).with_name("fs11_guard.sh")
README = Path(fs.__file__).parents[1] / "README.md"
REPO = Path(fs.__file__).resolve().parents[2]


def _py(code):
    return ["{python}", "-c", code]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wait_for(predicate, timeout=20.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if predicate():
            return True
        time.sleep(0.2)
    return False


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
    # the re-check's settings stay out of the first session's parts
    assert all(not set(fs.RECHECK_ENV) & set(p.env) for p in plan)
    assert fs.SESSIONS["fs11"] == ("FS11", fs.default_plan)


def test_recheck_plan(tmp_path):
    """The first session's models, bring-up and smooth parts with RECHECK_ENV added, no
    benchmark, Qwen2.5-0.5B's bring-up first, followed by its fix-off arm and then the other
    bring-ups in RUN_PINS order, then the allocator A/B: two plain arms per model, alternating
    which goes first."""
    up, out = tmp_path / "up", tmp_path / "out"
    plan = fs.recheck_plan(up, out)
    assert fs.SESSIONS["recheck"] == ("recheck", fs.recheck_plan)
    assert len(plan) == 22
    assert [p.kind for p in plan] == ["models"] + ["bring_up"] * 5 + ["smooth"] * 8 + ["plain"] * 8
    names = [p.name for p in plan]
    assert len(set(names)) == len(names)
    assert names[1:6] == ["bring_up_Qwen2.5-0.5B-Instruct",
                          "bring_up_Qwen2.5-0.5B-Instruct_fix2off",
                          "bring_up_Llama-3.2-1B-Instruct", "bring_up_Falcon3-1B-Instruct",
                          "bring_up_Qwen2.5-1.5B-Instruct"]
    first = {p.name: p for p in fs.default_plan(up, out)}
    on, off = plan[1], plan[2]
    for p in plan[:14]:
        if p is off:
            continue
        q = first[p.name]
        assert p.argv == q.argv
        assert p.info == {**q.info, **({"arm": "fix2_on", "alloc_conf": fs.EXPANDABLE,
                                         "release_cache": True} if p is on else {})}
        assert p.env == {**q.env, **(fs.RECHECK_ENV if p.kind != "models" else {})}
        assert p.timeout_s == fs.RECHECK_TIMEOUT_S[p.kind]
    for p in plan[1:14]:
        assert p.env["VERIF_HEAD_BLOCKS"] == "8"
        fix = p is not off
        # torch 2.9's own variable carries the setting; the deprecated one is always unset
        assert "PYTORCH_ALLOC_CONF" in p.env and "PYTORCH_CUDA_ALLOC_CONF" in p.env
        assert p.env["PYTORCH_ALLOC_CONF"] == ("expandable_segments:True" if fix else None)
        assert p.env["PYTORCH_CUDA_ALLOC_CONF"] is None
        assert p.env["VERIF_RELEASE_CACHE"] == ("1" if fix else "0")

    # the verified-step A/B of the whole allocator fix: the same bring-up, both settings undone
    assert off.argv == on.argv and "--no-tamper" in off.argv and off.timeout_s == on.timeout_s
    key = "Qwen2.5-0.5B-Instruct_refusal"
    d = Path(off.env["VERIF_OUTPUT_DIR"])
    assert d.name == f"{key}_fix2off" and d != Path(on.env["VERIF_OUTPUT_DIR"])
    assert (d / "data").resolve() == (up / "data" / key).resolve()
    assert off.env == {**on.env, "PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": None,
                       "VERIF_RELEASE_CACHE": "0", "VERIF_OUTPUT_DIR": str(d)}
    assert off.info == {**on.info, "run_dir": str(d), "arm": "fix2_off", "alloc_conf": None,
                        "release_cache": False}

    plain = plan[14:]
    a, b = "alloc_default", "alloc_expandable"
    assert [p.info["arm"] for p in plain] == [a, b, b, a, a, b, b, a]
    assert [p.info["model"] for p in plain] == [repo for repo, _ in fs.RUN_PINS for _ in "ab"]
    assert plain[0].name == "plain_Llama-3.2-1B-Instruct_alloc_default"
    dirs = {p.env["VERIF_OUTPUT_DIR"] for p in plain}
    assert len(dirs) == 8 and not dirs & {p.env.get("VERIF_OUTPUT_DIR") for p in plan[:14]}
    for p in plain:
        assert p.argv == ["{python}", "-m", "verification.runs.plain_baseline"]
        assert p.timeout_s == fs.RECHECK_TIMEOUT_S["plain"]
        assert {k: p.env[k] for k in fs.FULL_SCALE_ENV} == fs.FULL_SCALE_ENV
        assert p.env["VERIF_ATTACK"] == "refusal" and p.env["VERIF_HEAD_BLOCKS"] == "8"
        # None unsets the variable, whatever the pod's environment holds
        conf = None if p.info["arm"] == a else "expandable_segments:True"
        assert "PYTORCH_ALLOC_CONF" in p.env and p.env["PYTORCH_ALLOC_CONF"] == conf
        assert "PYTORCH_CUDA_ALLOC_CONF" in p.env and p.env["PYTORCH_CUDA_ALLOC_CONF"] is None
        assert p.info["alloc_conf"] == conf
        key = fs.data_key(p.info["model"], "refusal")
        d = Path(p.env["VERIF_OUTPUT_DIR"])
        assert d.name == f"{key}_plain_{p.info['arm']}"
        assert (d / "data").resolve() == (up / "data" / key).resolve()
        assert p.info["report"] == str(d / "plain_baseline" / "records.jsonl")
    assert all("HF_HUB_OFFLINE" not in p.env for p in plan[1:])


def test_plain_run_name():
    from verification.runs import plain_baseline
    assert fs.PLAIN_RUN == plain_baseline.RUN_NAME


def test_selftest_plan_needs_only_the_standard_library():
    plan = fs.selftest_plan()
    assert [p.kind for p in plan] == ["other", "other"]
    assert all(p.argv[:2] == ["{python}", "-c"] and p.timeout_s for p in plan)
    for p in plan:  # a bare python3 runs them
        subprocess.run([sys.executable, "-I", "-S", "-c", p.argv[2]], check=True, timeout=30)


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
    assert "upload_check" not in s  # no models part, no check

    # a rerun skips the finished part and runs the others again
    s = fs.run_session(plan[:2], upload=up, out=out, deadline=time.time() + 30, cwd=REPO,
                       say=lines.append)
    assert [p["status"] for p in s["parts"]] == ["skipped_done", "failed"]
    assert s["session"] == "FS11"


def test_the_allocator_arms_reach_their_parts(tmp_path, monkeypatch):
    """The A/B's arms as run: the default arm unsets both allocator variables, though the
    session's environment sets them; the other sets expandable segments in PYTORCH_ALLOC_CONF
    and still unsets the deprecated PYTORCH_CUDA_ALLOC_CONF."""
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.6")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
    probe = ("import os; print(repr((os.environ.get('PYTORCH_ALLOC_CONF'), "
             "os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))))")
    plan = [fs.Part(arm, _py(probe), env=fs.alloc_env(conf)) for arm, conf in fs.ALLOC_ARMS]
    out = tmp_path / "out"
    s = fs.run_session(plan, upload=tmp_path / "up", out=out, deadline=time.time() + 30,
                       cwd=REPO, say=lambda m: None, session="recheck")
    assert s["session"] == "recheck"
    assert json.loads((out / "summary.json").read_text())["session"] == "recheck"
    seen = {arm: (out / "parts" / arm / "log.txt").read_text().splitlines()[-1]
            for arm, _ in fs.ALLOC_ARMS}
    assert seen == {"alloc_default": "(None, None)",
                    "alloc_expandable": "('expandable_segments:True', None)"}


def _fix_arm(name, arm, conf, release, times, held):
    prover, verifier, handoff = times
    text = (f"step 1: prover {prover:.2f} s (prove_step 1.00 s, commit 0.50 s); verifier "
            f"{verifier:.2f} s (0 1.00, 5 2.00); hand-off {handoff:.2f} s\n"
            f"step 1: prover peak 40.00 GB, reserved {held[0]:.2f} GB (at its start 3.00 GB) "
            "[cuda_allocated, memory pass]\n"
            f"step 1: verifier peak 30.00 GB, reserved {held[1]:.2f} GB "
            "[cuda_allocated, memory pass]\n"
            "allocator: " + json.dumps({"PYTORCH_ALLOC_CONF": conf,
                                        "PYTORCH_CUDA_ALLOC_CONF": None,
                                        "release_cache": release}))
    return fs.Part(name, _py(f"print({text!r})"), kind="bring_up",
                   info={"model": fs.FIX_AB_MODEL, "task": "refusal", "arm": arm,
                         "alloc_conf": conf, "release_cache": release})


def test_the_fix_ab_sets_the_two_arms_side_by_side(tmp_path):
    """summary.json's fix2_ab: each arm's step times and peaks from its log, and the fix's
    cost, on minus off; the job log gets the totals once the fix-off arm is done."""
    on = _fix_arm("on", "fix2_on", fs.EXPANDABLE, True, (1.5, 3.0, 0.3), (41.0, 32.0))
    off = _fix_arm("off", "fix2_off", None, False, (1.4, 3.1, 0.0), (47.0, 47.5))
    out, lines = tmp_path / "out", []
    s = fs.run_session([on, off], upload=tmp_path / "up", out=out, deadline=time.time() + 30,
                       cwd=REPO, say=lines.append, session="recheck")
    ab = s["fix2_ab"]
    assert json.loads((out / "summary.json").read_text())["fix2_ab"] == ab
    assert ab["model"] == fs.FIX_AB_MODEL
    assert ab["fix_on"] == {"part": "on", "status": "ok", "alloc_conf": fs.EXPANDABLE,
                            "release_cache": True, "ran_with": {
                                "PYTORCH_ALLOC_CONF": fs.EXPANDABLE,
                                "PYTORCH_CUDA_ALLOC_CONF": None, "release_cache": True}}
    assert ab["fix_off"] == {"part": "off", "status": "ok", "alloc_conf": None,
                             "release_cache": False, "ran_with": {
                                 "PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": None,
                                 "release_cache": False}}
    (step,) = ab["steps"]
    assert step["fix_on"] == {"prover_s": 1.5, "verifier_s": 3.0, "handoff_s": 0.3,
                              "total_s": 4.8}
    assert step["fix_off"] == {"prover_s": 1.4, "verifier_s": 3.1, "handoff_s": 0.0,
                               "total_s": 4.5}
    assert step["fix_cost"] == {"prover_s": 0.1, "verifier_s": -0.1, "handoff_s": 0.3,
                                "total_s": 0.3}
    assert [(p["step"], p["side"], p["fix_on"], p["fix_cost"]) for p in ab["peaks"]] == [
        (1, "prover", {"peak_gb": 40.0, "reserved_gb": 41.0},
         {"peak_gb": 0.0, "reserved_gb": -6.0}),
        (1, "verifier", {"peak_gb": 30.0, "reserved_gb": 32.0},
         {"peak_gb": 0.0, "reserved_gb": -15.5})]
    assert "allocator fix A/B, step 1: 4.80 s with the fix, 4.50 s without (+0.30 s)" in lines
    # an arm that never ran leaves its side and the cost empty; no fix-off arm, no A/B
    alone = fs.fix2_ab(s["parts"][1:])
    assert alone["fix_on"] is None and alone["steps"][0]["fix_on"] is None
    assert alone["steps"][0]["fix_cost"]["total_s"] is None
    assert fs.fix2_ab(s["parts"][:1]) is None


def test_a_running_part_names_its_process_group(tmp_path):
    out = tmp_path / "out"
    probe = (f"import os; print(open({str(out / 'current_part')!r}).read().strip(), "
             "os.getpgid(0))")
    s = fs.run_session([fs.Part("probe", _py(probe))], upload=tmp_path / "up", out=out,
                       deadline=time.time() + 30, cwd=REPO, say=lambda m: None)
    assert s["parts"][0]["status"] == "ok"
    pid, pgid = (out / "parts" / "probe" / "log.txt").read_text().split()[-2:]
    assert pid == pgid
    assert not (out / "current_part").exists()


@pytest.mark.parametrize("report, why", [
    ({"complete": True, "slots": {"E1": {"results": [
        {"name": "a", "status": "measured"},
        {"name": "b", "status": "skipped", "reason": "cuPQC is not installed"}]}}}, None),
    ({"complete": False, "slots": {}}, "isn't complete"),
    ({"complete": True, "slots": {"E6": {"results": [
        {"name": "sgd_step", "status": "skipped", "reason": "error: RuntimeError: OOM"}]}}},
     "1 results ended in an error (E6/sgd_step)"),
    (None, "report unreadable"),
])
def test_the_benchmark_is_done_only_on_a_complete_clean_report(tmp_path, report, why):
    rep = tmp_path / "hashing.json"
    if report is not None:
        rep.write_text(json.dumps(report))
    assert (fs.benchmark_failure(rep) is None) == (why is None)
    if why:
        assert why in fs.benchmark_failure(rep)
    out = tmp_path / "out"
    s = fs.run_session([fs.Part("benchmark", _py("print(1)"), kind="benchmark",
                                info={"report": str(rep)})],
                       upload=tmp_path / "up", out=out, deadline=time.time() + 30, cwd=REPO,
                       say=lambda m: None)
    (rec,) = s["parts"]
    assert rec["exit_code"] == 0
    assert rec["status"] == ("ok" if why is None else "failed")
    assert (out / "parts" / "benchmark" / "DONE").exists() == (why is None)
    if why:
        assert why in rec["reason"]


def test_a_broken_fix_ab_costs_the_ab_not_the_session(tmp_path, monkeypatch):
    """``fix2_ab`` is wrapped as ``part_extras`` is: if it raises, the summary records why and
    the session runs on."""
    def broken(parts):
        raise KeyError("name")

    monkeypatch.setattr(fs, "fix2_ab", broken)
    on = _fix_arm("on", "fix2_on", fs.EXPANDABLE, True, (1.5, 3.0, 0.3), (41.0, 32.0))
    out = tmp_path / "out"
    s = fs.run_session([on, fs.Part("after", _py("print(1)"))], upload=tmp_path / "up",
                       out=out, deadline=time.time() + 30, cwd=REPO, say=lambda m: None)
    assert [p["status"] for p in s["parts"]] == ["ok", "ok"] and s["finished"]
    assert s["fix2_ab"] == {"fix2_ab_error": "KeyError: 'name'"}
    assert json.loads((out / "summary.json").read_text())["fix2_ab"] == s["fix2_ab"]


def test_part_extras(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("pin: x\nstep 1: prover 3.10 s (prove_step 2.00 s, commit 1.10 s); verifier "
                   "9.00 s (0 0.50, 5 8.50); hand-off 0.40 s\nstep 1: prover peak 41.20 GB, "
                   "reserved 44.00 GB (at its start 3.00 GB) [cuda_allocated, memory pass]\n"
                   "step 1: verifier peak 30.50 GB [cuda_allocated, memory pass]\n"
                   "process lifetime peak RSS 12.00 GB\n")
    e = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)
    assert e["peaks"] == [{"step": 1, "side": "prover", "peak_gb": 41.2, "reserved_gb": 44.0},
                          {"step": 1, "side": "verifier", "peak_gb": 30.5, "reserved_gb": None}]
    assert e["step_times"] == [{"step": 1, "prover_s": 3.1, "prove_step_s": 2.0, "commit_s": 1.1,
                                "verifier_s": 9.0, "checks_s": {"0": 0.5, "5": 8.5},
                                "handoff_s": 0.4, "total_s": 12.5}]
    assert len(e["lines"]) == 4
    assert e["allocator"] is None  # a log without the allocator line
    alloc = {"PYTORCH_ALLOC_CONF": "expandable_segments:True", "PYTORCH_CUDA_ALLOC_CONF": None,
             "release_cache": True}
    log.write_text(log.read_text() + "allocator: " + json.dumps(alloc) + "\n")
    assert fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["allocator"] == alloc
    log.write_text(log.read_text() + "allocator: {broken\n")
    assert fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["allocator"] == alloc
    # a log from before the hand-off was timed
    (old,) = fs._step_times(["step 2: prover 3.10 s (prove_step 2.00 s, commit 1.10 s); "
                             "verifier 9.00 s ()"])
    assert (old["step"], old["checks_s"], old["handoff_s"], old["total_s"]) == (2, {}, None, 12.1)
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
    smooth = tmp_path / "smooth.json"
    smooth.write_text(json.dumps({"smooth": True, "grad_norms": [1.5, 1.25], "memory": {
        "peak_bytes": 5, "peak_reserved_bytes": 6}, "environment": {
            "record": "environment", **alloc, "release_cache": False}}))
    e = fs.part_extras(fs.Part("s", [], kind="smooth", info={"report": str(smooth)}), log)
    assert e["grad_norms"] == [1.5, 1.25]
    assert e["allocator"] == {**alloc, "release_cache": False}
    assert (e["peak_bytes"], e["peak_reserved_bytes"]) == (5, 6)


def test_bring_up_section_peaks(tmp_path):
    """The memory pass's echo lines (the ones test_metrics pins) become section_peaks, the
    section that raised included; the step-peak parser doesn't take them."""
    echo = ["   run, P4.tree: peak 0.00 GB (start 0.00, end 0.00 GB)",
            "   step 1, train.backward: peak 3.00 GB (start 0.00, end 1.00 GB)",
            "   step 2, check 5: peak 5.00 GB (start 1.00, end ? GB; nested: 5.glue 3.00, "
            "5.measure 5.00 GB); raised OutOfMemoryError",
            # on CUDA, with the reserved peaks
            "   step 1, check 5: peak 3.00 GB, reserved 3.50 GB (start 1.00, end 1.00 GB; "
            "nested: 5.glue 3.00 (reserved 3.50), 5.measure 2.00 (reserved 3.50) GB)"]
    log = tmp_path / "log.txt"
    log.write_text("\n".join(["== memory pass: each section's peak as it closes, GB "
                              "[cuda_allocated]", *echo, "Traceback (most recent call last):"])
                   + "\n")
    e = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)
    assert e["peaks"] == []
    none = {"reserved_gb": None, "nested_reserved_gb": {}}
    assert e["section_peaks"] == [
        {"step": None, "section": "P4.tree", "peak_gb": 0.0, "start_gb": 0.0, "end_gb": 0.0,
         "nested_gb": {}, "raised": None, **none},
        {"step": 1, "section": "train.backward", "peak_gb": 3.0, "start_gb": 0.0,
         "end_gb": 1.0, "nested_gb": {}, "raised": None, **none},
        {"step": 2, "section": "check 5", "peak_gb": 5.0, "start_gb": 1.0, "end_gb": None,
         "nested_gb": {"5.glue": 3.0, "5.measure": 5.0}, "raised": "OutOfMemoryError", **none},
        {"step": 1, "section": "check 5", "peak_gb": 3.0, "reserved_gb": 3.5, "start_gb": 1.0,
         "end_gb": 1.0, "nested_gb": {"5.glue": 3.0, "5.measure": 2.0},
         "nested_reserved_gb": {"5.glue": 3.5, "5.measure": 3.5}, "raised": None}]


def test_cost_lines_parse_as_report_costs_prints_them():
    """The bring-up's step times and peaks come from llama_step.report_costs' own lines."""
    from types import SimpleNamespace as NS
    from verification.runs.llama_step import report_costs
    step = NS(t=1, prove_s=50.0, commit_s=1.25, verify_s=100.5, handoff_s=0.75)
    r = NS(loop=NS(steps=[step]), verifier=NS(timings={1: {"0": 0.5, "6a": 3.25}}))
    rows = [{"level": "total", "component": side, "step": 1, "peak_bytes": 60 * 2**30,
             "peak_reserved_bytes": held, "start_bytes": 2**30, "mem_source": "cuda_allocated"}
            for side, held in (("prover", 62 * 2**30), ("verifier", None))]
    lines = []
    report_costs(r, rows, lines.append)
    assert lines[1] == ("step 1: prover peak 60.00 GB, reserved 62.00 GB (at its start 1.00 GB) "
                        "[cuda_allocated, memory pass]")
    (t,) = fs._step_times(lines)
    assert (t["prover_s"], t["verifier_s"], t["handoff_s"], t["total_s"]) == (51.25, 100.5,
                                                                                0.75, 152.5)
    assert t["checks_s"] == {"0": 0.5, "6a": 3.25}
    assert [(m[2], m[3], m[4]) for m in map(fs._PEAK.match, lines) if m] == [
        ("prover", "60.00", "62.00"), ("verifier", "60.00", None)]


def test_section_lines_parse_as_the_recorder_prints_them():
    """The parser reads metrics.MemoryRecorder's own echo, with and without a reserve."""
    from verification.runs.metrics import MemoryRecorder
    from tests.verification.runs.test_metrics import _Rec, _ReservingProbe, _ScriptedProbe
    for probe in (_ScriptedProbe(), _ReservingProbe()):
        lines = []
        rec = MemoryRecorder("m", "s", probe, echo=lines.append)
        with rec.section("5"):
            with rec.section("5.glue"):
                probe.alloc(2)
                probe.free(1)
        rec.on_step(_Rec(1))
        (got,) = fs._section_peaks(lines)
        held = 2.0 if isinstance(probe, _ReservingProbe) else None
        assert (got["section"], got["peak_gb"], got["reserved_gb"]) == ("check 5", 2.0, held)
        assert got["nested_gb"] == {"5.glue": 2.0}
        assert got["nested_reserved_gb"] == ({} if held is None else {"5.glue": 2.0})


def _records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_plain_extras(tmp_path):
    """A plain arm's summary: the timed run's step totals in step order, their mean and
    median, and the memory pass's peak; other totals and step 0 don't count."""
    rec = tmp_path / "plain_baseline" / "records.jsonl"
    step = {"record": "time", "side": "run", "component": "step", "level": "total"}
    alloc = {"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": None,
             "release_cache": True}
    rows = [{"record": "environment", "run": "plain_baseline", **alloc},
            *({**step, "step": t, "time_s": s} for t, s in ((2, 1.5), (1, 2.5), (3, 1.0))),
            {**step, "step": 0, "time_s": 99.0},
            {"record": "time", "side": "prover", "component": "prover", "level": "total",
             "step": 1, "time_s": 99.0},
            {"record": "memory", "step": 1, "level": "phase", "component": "P0.backward",
             "peak_bytes": 5 * 2**30, "peak_reserved_bytes": 6 * 2**30,
             "mem_source": "cuda_allocated"},
            {"record": "memory", "step": 1, "level": "phase", "component": "P0.forward",
             "peak_bytes": 3 * 2**30, "peak_reserved_bytes": 4 * 2**30,
             "mem_source": "cuda_allocated"},
            {"record": "run_end", "run": "plain_baseline", "lifetime_maxrss_bytes": 7}]
    _records(rec, rows)
    part = fs.Part("plain", [], kind="plain", info={"report": str(rec)})
    e = fs.part_extras(part, tmp_path / "log.txt")
    assert e["step_s"] == [2.5, 1.5, 1.0]
    assert e["mean_step_s"] == pytest.approx(5 / 3) and e["median_step_s"] == 1.5
    assert e["peak_bytes"] == 5 * 2**30 and e["mem_source"] == "cuda_allocated"
    assert e["peak_reserved_bytes"] == 6 * 2**30
    assert e["completed"] and e["lifetime_maxrss_bytes"] == 7 and e["record"] == str(rec)
    assert e["allocator"] == alloc  # what the run's environment record says it ran with
    assert e["bad_lines"] == 0
    # a line cut short (a run killed mid-write) or with a figure that isn't a number is
    # skipped and counted; the rest of the arm stands
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows)
                   + json.dumps({**step, "step": 4, "time_s": "fast"}) + "\n"
                   + json.dumps({"record": "memory", "peak_bytes": [1]}) + "\n"
                   + "[1, 2]\n" + '{"record": "time", "step": 5, "ti')
    cut = fs.part_extras(part, tmp_path / "log.txt")
    assert cut["bad_lines"] == 4 and "extras_error" not in cut
    assert {k: cut[k] for k in e if k != "bad_lines"} == {k: e[k] for k in e if k != "bad_lines"}
    # a run that died after one timed step, before its memory pass
    _records(rec, rows[:3])
    e = fs.part_extras(part, tmp_path / "log.txt")
    assert e["step_s"] == [2.5, 1.5] and e["peak_bytes"] is None and not e["completed"]
    assert e["peak_reserved_bytes"] is None
    _records(rec, rows[:1])
    e = fs.part_extras(part, tmp_path / "log.txt")
    assert e["step_s"] == [] and e["mean_step_s"] is None and e["median_step_s"] is None
    assert fs.part_extras(fs.Part("p", [], kind="plain", info={"report": str(tmp_path / "no")}),
                          tmp_path / "log.txt") == {"record": None}


# ---- fetch-models and the upload check ----------------------------------------------------


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


def test_check_upload(tmp_path):
    up, weight = _upload(tmp_path)
    m = json.loads((up / "manifest.json").read_text())
    m["models"].append({"repo": "org/mirror", "revision": "c" * 40, "gated": False,
                        "roles": ["benchmark"], "files": [],
                        "missing": ["weights not in the Mac's cache"]})
    m["models"].append({"repo": "org/absent", "revision": "d" * 40, "gated": False,
                        "roles": ["run"], "files": [], "missing": ["config.json"]})
    (up / "manifest.json").write_text(json.dumps(m))
    _fake_download(weight, [])("org/public", "model.safetensors", "a" * 40, up / "hf" / "hub")
    c = fs.check_upload(up)
    assert c["packed"] == []
    # the gated weight isn't in the cache; a benchmark-only pin may lack weights; a run model
    # missing on the Mac is a problem
    assert set(c["models"]) == {"meta-llama/Llama-3.2-1B-Instruct", "org/absent"}
    assert c["models"]["org/absent"] == ["missing on the Mac: config.json"]
    (up / "repo" / "a.txt").write_bytes(b"changed")
    assert fs.check_upload(up)["packed"] == ["repo/a.txt: SHA-256 differs from the manifest"]
    (up / "manifest.json").write_text("{")
    assert "manifest.json unreadable" in fs.check_upload(up)["packed"][0]


def _checked_plan():
    """A models part, then a part per model and one that names none."""
    return [fs.Part("models", _py("print(1)"), kind="models"),
            fs.Part("bring_public", _py("print(1)"), kind="bring_up",
                    info={"model": "org/public"}),
            fs.Part("bring_gated", _py("print(1)"), kind="bring_up",
                    info={"model": "meta-llama/Llama-3.2-1B-Instruct"}),
            fs.Part("other", _py("print(1)"))]


def test_a_mismatch_keeps_the_affected_parts_from_starting(tmp_path):
    up, weight = _upload(tmp_path)
    out = tmp_path / "out"
    run = lambda: fs.run_session(_checked_plan(), upload=up, out=out,  # noqa: E731
                                 deadline=time.time() + 60, cwd=REPO, say=lambda m: None)
    _fake_download(weight, [])("org/public", "model.safetensors", "a" * 40, up / "hf" / "hub")
    s = run()
    st = {p["name"]: p["status"] for p in s["parts"]}
    # the gated model's weight is missing: only its part waits
    assert st == {"models": "ok", "bring_public": "ok", "bring_gated": "not_started",
                  "other": "ok"}
    assert "meta-llama/Llama-3.2-1B-Instruct's files are missing" in s["parts"][2]["reason"]
    assert s["upload_check"]["packed"] == []

    # A rerun skips the done models part but checks again: a packed file changed since, so no
    # part that isn't done yet starts. The done ones ran on the upload as checked then.
    (up / "repo" / "a.txt").write_bytes(b"changed")
    s = run()
    st = {p["name"]: p["status"] for p in s["parts"]}
    assert st == {"models": "skipped_done", "bring_public": "skipped_done",
                  "bring_gated": "not_started", "other": "skipped_done"}
    reason = s["parts"][2]["reason"]
    assert "doesn't match its manifest" in reason and "repo/a.txt" in reason
    assert s["upload_check"]["packed"] and not (out / "parts" / "bring_gated" / "DONE").exists()

    # In a fresh output directory the same upload starts nothing after the models part.
    s = fs.run_session(_checked_plan(), upload=up, out=tmp_path / "out2",
                       deadline=time.time() + 60, cwd=REPO, say=lambda m: None)
    assert [p["status"] for p in s["parts"]] == ["ok"] + ["not_started"] * 3


# ---- the job script ------------------------------------------------------------------------


def _job(tmp_path, *, plan=None, python=sys.executable, deadline=6, grace=1, reserve=1,
         extra=None, timeout=60, popen=False):
    env = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_POD_ID", "RUNPOD_API_KEY")}
    env.update({"DRY_RUN": "1", "FS11_DEADLINE_S": str(deadline), "FS11_GRACE_S": str(grace),
                "FS11_PACK_RESERVE_S": str(reserve), "FS11_PYTHON": python,
                "FS11_MIN_FREE_GB": "0", **(extra or {})})
    if plan is not None:
        p = tmp_path / "plan.json"
        # a stand-in for the build the session puts first (it would compile for real here)
        p.write_text(json.dumps([{"name": "native_build", "argv": _py("pass")}, *plan]))
        env["FS11_PLAN"] = str(p)
    (tmp_path / "up").mkdir(exist_ok=True)
    out = tmp_path / "out"
    argv = ["bash", str(JOB), str(tmp_path / "up"), str(out)]
    if popen:
        return subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True), out
    r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return r, out


def _log(out):
    return (out / "job.log").read_text()


OK_PLAN = [{"name": "ok", "argv": _py("print('part ran')")}]


def test_job_runs_packs_and_terminates(tmp_path):
    plan = [{"name": "ok", "argv": _py("print('part ran')")},
            {"name": "fails", "argv": _py("import sys; sys.exit(3)")},
            {"name": "slow", "argv": ["sleep", "60"]}]
    # Parts stop at 10 - 1 - 3 = 6 s; the 3 s reserve leaves packing room on a loaded machine.
    r, out = _job(tmp_path, plan=plan, deadline=10, reserve=3,
                  extra={"FS11_ALLOW_ANY_GPU": "1"})
    assert r.returncode == 1, r.stdout + r.stderr
    s = json.loads((out / "summary.json").read_text())
    assert [p["status"] for p in s["parts"]] == ["ok", "ok", "failed", "killed_deadline"]
    ready = dict(l.split("=", 1) for l in Path(f"{out}.READY").read_text().splitlines())
    tar = Path(f"{out}.tar")
    assert ready["sha256"] == _sha(tar.read_bytes()) and ready["session_exit"] == "1"
    assert ready["sha256"] in r.stdout
    assert "DRY_RUN: terminate (end of job (session exit 1)): runpodctl pod delete" in r.stdout
    assert Path(f"{out}.TERMINATED").read_text().startswith("end of job")
    assert not Path(f"{out}.PACK_FAILED").exists()
    # The job's own lines reach both its stdout and job.log; the session's only job.log.
    for line in ("FS11 job:", "terminate tools:", "session:"):
        assert line in r.stdout and line in _log(out)
    assert "ok: ok, exit 0" in _log(out) and "ok: ok, exit 0" not in r.stdout
    tools = (out / "terminate_tools.txt").read_text()
    assert "runpod_api_key=unset" in tools and "runpodctl=" in tools
    with tarfile.open(tar) as t:
        assert {"out/summary.json", "out/parts/ok/log.txt", "out/job.log",
                "out/terminate_tools.txt"} <= set(t.getnames())


def test_job_terminates_when_the_session_refuses(tmp_path):
    """No H100 and no override: the session refuses, and the job still packs and ends."""
    r, out = _job(tmp_path, plan=OK_PLAN, extra={"FS11_ALLOW_ANY_GPU": "0"})
    if "H100 check passes" in _log(out):
        pytest.skip("this machine has an H100")
    assert r.returncode == fs.EXIT_REFUSED, r.stdout + r.stderr
    assert json.loads((out / "summary.json").read_text())["refused"]
    assert Path(f"{out}.tar").exists() and "DRY_RUN: terminate (end of job" in r.stdout


def test_job_timer_terminates_a_hung_session(tmp_path):
    """The deadline timer fires while the session hangs (the job's own kill set late)."""
    hung = tmp_path / "hung.sh"
    hung.write_text("#!/bin/sh\nexec sleep 60\n")
    hung.chmod(0o755)
    t0 = time.time()
    r, out = _job(tmp_path, python=str(hung), deadline=3,
                  extra={"FS11_HANG_MARGIN_S": "1000", "FS11_STOP_GRACE_S": "2"})
    assert time.time() - t0 < 20
    assert "deadline reached" in r.stdout and "DRY_RUN: terminate (deadline)" in r.stdout
    assert r.stdout.count("DRY_RUN: terminate") == 1  # once, though the job's trap ran too
    assert not Path(f"{out}.tar").exists()


HUNG_SESSION = """\
import os, signal, subprocess, sys, time
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
part = subprocess.Popen(["sleep", "300"], start_new_session=True)
for name in ("current_part", "part_pid"):
    with open(os.path.join(out, name), "w") as f:
        f.write(f"{part.pid}\\n")
time.sleep(300)
"""


def test_job_kills_a_hung_session_and_its_part_then_packs(tmp_path):
    """A session that ignores SIGTERM, with a part in its own process group: the job kills
    both FS11_HANG_MARGIN_S after the work deadline, packs and terminates before the timer."""
    (tmp_path / "hung.py").write_text(HUNG_SESSION)
    wrapper = tmp_path / "python"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'hung.py'} \"$@\"\n")
    wrapper.chmod(0o755)
    r, out = _job(tmp_path, python=str(wrapper), deadline=16, grace=1, reserve=8,
                  extra={"FS11_HANG_MARGIN_S": "2", "FS11_STOP_GRACE_S": "2"})
    assert "killing it and its part" in r.stdout, r.stdout
    assert "DRY_RUN: terminate (end of job" in r.stdout
    assert "(deadline)" not in r.stdout
    assert Path(f"{out}.READY").exists() and Path(f"{out}.TERMINATED").exists()
    part = int((out / "part_pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(part, 0)


def test_job_survives_a_dead_reader_on_its_stdout(tmp_path):
    """nohup's output file gone or the ssh pipe closed: the job runs on, logs to job.log, and
    still packs and terminates."""
    p, out = _job(tmp_path, plan=OK_PLAN, extra={"FS11_ALLOW_ANY_GPU": "1"}, popen=True)
    p.stdout.close()
    p.stderr.close()
    assert p.wait(timeout=60) == 0
    log = _log(out)
    assert "ok: ok, exit 0" in log and "READY:" in log
    assert "DRY_RUN: terminate (end of job (session exit 0))" in log
    assert Path(f"{out}.TERMINATED").exists()


def test_job_runs_without_its_log_file(tmp_path):
    (tmp_path / "out" / "job.log").mkdir(parents=True)
    r, out = _job(tmp_path, plan=OK_PLAN, extra={"FS11_ALLOW_ANY_GPU": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cannot append to" in r.stdout and "ok: ok, exit 0" in r.stdout
    assert "DRY_RUN: terminate (end of job" in r.stdout and Path(f"{out}.TERMINATED").exists()


def test_job_marks_a_failed_pack_and_still_terminates(tmp_path):
    (tmp_path / "out.tar.tmp").mkdir()  # tar can't write its tarball over a directory
    r, out = _job(tmp_path, plan=OK_PLAN, extra={"FS11_ALLOW_ANY_GPU": "1"})
    assert "PACK FAILED" in r.stdout
    assert Path(f"{out}.PACK_FAILED").exists() and not Path(f"{out}.READY").exists()
    assert "DRY_RUN: terminate (end of job" in r.stdout and Path(f"{out}.TERMINATED").exists()


@pytest.mark.parametrize("deadline, grace, extra, why", [
    (2, 1, {}, "must be below FS11_DEADLINE_S"),
    (60, 1, {"FS11_MIN_FREE_GB": str(10**9)}, "FS11_MIN_FREE_GB is"),
    (60, 1, {"FS11_SESSION": "fs12"}, "FS11_SESSION is 'fs12'; it must be fs11 or recheck"),
])
def test_job_terminates_on_a_refusal(tmp_path, deadline, grace, extra, why):
    r, out = _job(tmp_path, plan=OK_PLAN, deadline=deadline, grace=grace, extra=extra)
    assert r.returncode == 2 and why in r.stdout, r.stdout + r.stderr
    assert "DRY_RUN: terminate (refused:" in r.stdout and Path(f"{out}.TERMINATED").exists()
    assert not (out / "summary.json").exists()


def test_job_refuses_without_a_way_to_terminate(tmp_path):
    r, out = _job(tmp_path, extra={"DRY_RUN": "0"})
    assert r.returncode == 2 and "RUNPOD_POD_ID is not set" in r.stdout
    assert not out.exists()


@pytest.mark.parametrize("session, label", [(None, "FS11"), ("recheck", "recheck")])
def test_job_selftest(tmp_path, session, label):
    """FS11_SELFTEST: the stand-in parts, no GPU check, then the normal end-of-run terminate,
    under either session's name."""
    extra = {"FS11_SELFTEST": "1", **({"FS11_SESSION": session} if session else {})}
    r, out = _job(tmp_path, deadline=30, grace=1, reserve=5, extra=extra)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((out / "summary.json").read_text())
    assert s["session"] == label
    assert [(p["name"], p["status"]) for p in s["parts"]] == [
        ("selftest_ok", "ok"), ("selftest_sleep", "ok")]
    assert f"self-test: stand-in parts in place of the {label} session's" in _log(out)
    assert "SELFTEST=1" in r.stdout and f"session {session or 'fs11'}," in r.stdout
    assert "terminate tools:" in r.stdout
    assert "DRY_RUN: terminate (end of job (session exit 0))" in r.stdout


@pytest.mark.parametrize("session, label, deadline_s", [
    (None, "FS11", 12600), ("fs11", "FS11", 12600), ("recheck", "recheck", 7500)])
def test_job_passes_the_session_on(tmp_path, session, label, deadline_s):
    """FS11_SESSION reaches the session's --session, which names it in summary.json; an unset
    FS11_DEADLINE_S takes the session's default."""
    extra = {"FS11_ALLOW_ANY_GPU": "1", "FS11_DEADLINE_S": "",
             **({"FS11_SESSION": session} if session else {})}
    r, out = _job(tmp_path, plan=OK_PLAN, extra=extra)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((out / "summary.json").read_text())["session"] == label
    assert f"deadline in {deadline_s} s" in r.stdout
    log = _log(out)
    assert f"run --session {session or 'fs11'}" in log and f"session {label}: 2 parts" in log


def test_a_refused_recheck_names_its_session(tmp_path, monkeypatch):
    monkeypatch.delenv("FS11_ALLOW_ANY_GPU", raising=False)
    out = tmp_path / "out"
    if fs.refusal(fs.environment(out), False) is None:  # main would run the real parts
        pytest.skip("this machine has an H100")
    rc = fs.main(["run", "--upload", str(tmp_path / "up"), "--out", str(out), "--deadline",
                  str(time.time() + 30), "--session", "recheck"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == fs.EXIT_REFUSED
    assert s["session"] == "recheck" and s["refused"] and s["parts"] == []


# ---- the job against stub runpodctl and curl (DRY_RUN=0) -----------------------------------


STUB_RUNPODCTL = """\
#!/bin/sh
echo "runpodctl $*" >> "$STUB_LOG"
case "$1 $2" in
  "pod delete")
    case "${STUB_DELETE:-fail}" in
      ok) exit 0 ;;
      slow) sleep 2; exit 0 ;;
      hang) exec sleep 1000 ;;
    esac
    exit 1 ;;
  "pod get")
    if [ "${STUB_GET:-there}" = gone ]; then echo "Error: pod not found" >&2; exit 1; fi
    echo '{"id": "test-pod", "desiredStatus": "RUNNING"}'; exit 0 ;;
  "pod stop") [ "${STUB_STOP:-fail}" = ok ] && exit 0; exit 1 ;;
  "version ") echo "runpodctl v2.1.0"; exit 0 ;;
esac
exit 1
"""

STUB_CURL = """\
#!/bin/sh
input=$(cat)
method=GET; prev=
for a in "$@"; do [ "$prev" = -X ] && method=$a; prev=$a; done
echo "curl $*" >> "$STUB_LOG"
echo "curl-stdin $input" >> "$STUB_LOG"
case "$method" in
  DELETE) code=${STUB_REST_DELETE:-500} ;;
  POST) code=${STUB_REST_STOP:-500} ;;
  *) code=${STUB_REST_GET:-200} ;;
esac
case " $* " in *" -w "*) printf '%s' "$code" ;; esac
case "$code" in 2*) exit 0 ;; esac
exit 22
"""

# coreutils' timeout for the boot guard on the Mac, which has none: drop the options and the
# duration, run the command.
STUB_TIMEOUT = """\
#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -k) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
exec "$@"
"""


def _stubs(tmp_path, *, timeout=False, **modes):
    """A bin directory with runpodctl and curl stubs first on PATH; never the real ones."""
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    scripts = {"runpodctl": STUB_RUNPODCTL, "curl": STUB_CURL}
    if timeout:
        scripts["timeout"] = STUB_TIMEOUT
    for name, text in scripts.items():
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    env = {"PATH": f"{b}:{os.environ['PATH']}", "STUB_LOG": str(log),
           "RUNPOD_POD_ID": "test-pod"}
    env.update({f"STUB_{k.upper()}": v for k, v in modes.items()})
    return env, log


REAL = {"DRY_RUN": "0", "FS11_ALLOW_ANY_GPU": "1", "FS11_CALL_TIMEOUT_S": "2",
        "FS11_RETRY_S": "1", "FS11_CONFIRM_S": "2"}


def _calls(log):
    return log.read_text().splitlines()


def test_job_terminates_with_runpodctl_and_confirms(tmp_path):
    env, log = _stubs(tmp_path, delete="ok", get="gone")
    r, out = _job(tmp_path, plan=OK_PLAN, extra={**REAL, **env})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "confirmed: runpodctl pod get test-pod says not found" in r.stdout
    assert Path(f"{out}.TERMINATED").read_text().split("; ")[1] == "method: runpodctl pod delete"
    assert [c for c in _calls(log) if "version" not in c] == [
        "runpodctl pod delete test-pod", "runpodctl pod get test-pod"]
    assert "runpodctl_version=runpodctl v2.1.0" in (out / "terminate_tools.txt").read_text()


def test_job_terminates_over_rest_with_the_key_off_the_command_line(tmp_path):
    env, log = _stubs(tmp_path, rest_delete="204", rest_get="404")
    env["RUNPOD_API_KEY"] = "test-key-123"
    r, out = _job(tmp_path, plan=OK_PLAN, extra={**REAL, **env})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "confirmed: GET pod test-pod answers 404" in r.stdout
    assert "method: REST DELETE" in Path(f"{out}.TERMINATED").read_text()
    calls = _calls(log)
    assert "runpodctl pod delete test-pod" in calls and "runpodctl remove pod test-pod" in calls
    argv = [c for c in calls if c.startswith("curl ")]
    assert [re.search(r"-X (\w+)", c).group(1) for c in argv] == ["DELETE", "GET"]
    assert all("test-key-123" not in c for c in argv)
    assert all("--config -" in c and c.endswith("/v1/pods/test-pod") for c in argv)
    assert 'Authorization: Bearer test-key-123' in log.read_text()  # on stdin only
    assert "test-key-123" not in _log(out) + r.stdout
    assert "runpod_api_key=set" in (out / "terminate_tools.txt").read_text()


def test_job_falls_back_to_stop_when_delete_hangs(tmp_path):
    env, log = _stubs(tmp_path, delete="hang", stop="ok")
    t0 = time.time()
    r, out = _job(tmp_path, plan=OK_PLAN, deadline=30, extra={
        **REAL, **env, "FS11_DELETE_ROUNDS": "2"})
    assert time.time() - t0 < 25
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [c for c in _calls(log) if "version" not in c]
    # two rounds of delete (each cut at the call limit) before the first stop
    assert calls == ["runpodctl pod delete test-pod", "runpodctl remove pod test-pod",
                     "runpodctl pod delete test-pod", "runpodctl remove pod test-pod",
                     "runpodctl pod stop test-pod"]
    assert "stopped, not removed" in Path(f"{out}.TERMINATED").read_text()


def test_job_hands_a_failing_terminate_to_the_timer(tmp_path):
    """Every call fails: the job gives up at the deadline without the TERMINATED flag, and the
    timer goes on trying (stop included) until it is killed."""
    env, log = _stubs(tmp_path)
    p, out = _job(tmp_path, plan=OK_PLAN, deadline=6, grace=1, reserve=2, popen=True,
                  extra={**REAL, **env, "FS11_DELETE_ROUNDS": "2"})
    try:
        assert _wait_for(lambda: (out / "job.log").exists()
                         and "terminating pod test-pod (deadline)" in _log(out)
                         and _log(out).split("(deadline)")[1].count("trying: runpodctl pod stop")
                         >= 2, timeout=40), _log(out)
        assert p.poll() is None  # the job waits on the timer
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    log_text = _log(out)
    assert "TERMINATE FAILED" in log_text
    assert not Path(f"{out}.TERMINATED").exists()
    calls = [c for c in _calls(log) if "version" not in c]
    first_stop = calls.index("runpodctl pod stop test-pod")
    assert calls[:first_stop].count("runpodctl pod delete test-pod") >= 2


def test_job_and_timer_never_terminate_at_once(tmp_path):
    """The grace window ends at the deadline, so the job and the timer reach terminate
    together; the lock lets one delete through, and the other finds the flag."""
    env, log = _stubs(tmp_path, delete="slow", get="gone")
    r, out = _job(tmp_path, plan=OK_PLAN, deadline=8, grace=6, reserve=1,
                  extra={**REAL, **env, "FS11_CALL_TIMEOUT_S": "6"})
    assert "waits for the other terminate" in _log(out)
    calls = [c for c in _calls(log) if "version" not in c]
    assert calls.count("runpodctl pod delete test-pod") == 1, calls
    assert Path(f"{out}.TERMINATED").exists()
    assert not Path(f"{out}.terminate.lock").exists()


# ---- the billing guards ---------------------------------------------------------------------


def test_guard_dry_run(tmp_path):
    log = tmp_path / "guard.log"
    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_POD_ID"}
    r = subprocess.run(["sh", str(GUARD), "1"], env={**env, "DRY_RUN": "1",
                       "FS11_GUARD_LOG": str(log)}, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0 and "disarm with: kill" in r.stdout
    assert _wait_for(lambda: "DRY_RUN: runpodctl pod delete" in log.read_text(), 10)
    assert "armed: firing in 1 s" in log.read_text()


def test_guard_refuses_without_a_pod(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_POD_ID"}
    r = subprocess.run(["sh", str(GUARD), "1"], env={**env, "DRY_RUN": "0"},
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 2 and "RUNPOD_POD_ID is not set" in r.stderr
    r = subprocess.run(["sh", str(GUARD), "1h"], env={**env, "DRY_RUN": "1"},
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 2 and "usage" in r.stderr


def _boot_guards():
    """The runbook's start commands for the pod create call: the H100 session's and the
    self-test's, which differ in the sleep alone."""
    lines = [l.strip() for l in README.read_text().splitlines()
             if l.strip().startswith("bash -c '(")]
    assert len(lines) == 2, lines
    return lines


def test_boot_guard_lines_parse():
    h100, selftest = _boot_guards()
    assert "sleep 16200;" in h100 and "sleep 960;" in selftest
    assert h100.replace("sleep 16200;", "sleep 960;") == selftest
    for line in (h100, selftest):
        argv = shlex.split(line)
        assert argv[:2] == ["bash", "-c"] and len(argv) == 3
        assert argv[2].endswith("& exec /start.sh")
        subprocess.run(["bash", "-n", "-c", argv[2]], check=True)
        # the create call's exec form carries the same script
        assert json.dumps(argv[2]) in README.read_text()


@pytest.mark.parametrize("modes, ends_with", [
    ({"delete": "ok"}, "runpodctl pod delete test-pod"),
    ({}, "runpodctl pod stop test-pod"),
])
def test_boot_guard_starts_the_image_and_terminates_later(tmp_path, modes, ends_with):
    """The start command, its sleep cut to 1 s, on stub runpodctl, curl and timeout: the
    image's /start.sh starts at once, and the guard terminates after the sleep."""
    env, log = _stubs(tmp_path, timeout=True, stop="ok", **modes)
    env["RUNPOD_API_KEY"] = "test-key-123"
    start = tmp_path / "start.sh"
    start.write_text(f"#!/bin/sh\necho started >> {log}\n")
    start.chmod(0o755)
    guard_log = tmp_path / "guard.log"
    script = shlex.split(_boot_guards()[0])[2]
    script = (script.replace("sleep 16200;", "sleep 1;").replace("sleep 30;", "sleep 0;")
              .replace("/tmp/guard.log", str(guard_log)).replace("/start.sh", str(start)))
    subprocess.run(["bash", "-c", script], env={**os.environ, **env}, check=True, timeout=10,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _calls(log) == ["started"]  # the image's entry ran; the guard still sleeps
    assert _wait_for(lambda: _calls(log)[-1] == ends_with, 15), _calls(log)
    calls = _calls(log)
    assert calls[1] == "runpodctl pod delete test-pod"
    if not modes:  # three rounds of delete, remove and REST DELETE, then stop
        assert calls.count("runpodctl pod delete test-pod") == 3
        assert sum(c.startswith("curl ") for c in calls) == 3
        assert all("test-key-123" not in c for c in calls if c.startswith("curl "))
    assert "armed" in guard_log.read_text()


def test_base_env_points_the_native_cache_at_the_upload(tmp_path):
    assert fs.base_env(tmp_path)["POTS_NATIVE_CACHE"] == str(tmp_path / "native_cache")


def test_with_native_build_prepends_once(tmp_path):
    plan = fs.default_plan(tmp_path / "up", tmp_path / "out")
    withb = fs.with_native_build(plan)
    assert withb[0].name == "native_build" and withb[1:] == plan and len(withb) == 15
    assert withb[0].argv == ["{python}", "-m", "verification.commitment.native"]
    assert withb[0].timeout_s == 900
    assert fs.with_native_build(withb) == withb
    assert fs.with_native_build([])[0].name == "native_build"
    assert plan[0].name != "native_build"  # the input isn't changed


def test_choose_plan_adds_the_build_except_for_the_self_test(tmp_path):
    up, out = tmp_path / "up", tmp_path / "out"
    for key in ("fs11", "recheck"):
        build = fs.SESSIONS[key][1]
        plan = fs.choose_plan(build, up, out, plan_file=None, selftest=False)
        assert plan[0].name == "native_build" and plan[1:] == build(up, out)
    stand_in = fs.choose_plan(fs.default_plan, up, out, plan_file=None, selftest=True)
    assert [p.name for p in stand_in] == [p.name for p in fs.selftest_plan()]
    f = tmp_path / "plan.json"
    f.write_text(json.dumps([{"name": "x", "argv": ["{python}", "-c", "pass"]}]))
    plan = fs.choose_plan(fs.default_plan, up, out, plan_file=f, selftest=False)
    assert [p.name for p in plan] == ["native_build", "x"]
    again = fs.choose_plan(fs.default_plan, up, out, plan_file=f, selftest=False)
    assert plan[0] is not again[0] and plan[0].env is not again[0].env
    plan = fs.choose_plan(fs.default_plan, up, out, plan_file=f, selftest=True)
    assert [p.name for p in plan] == ["x"]


def test_check_upload_reports_the_native_sources(tmp_path):
    up, _ = _upload(tmp_path)
    c = fs.check_upload(up)
    assert c["native"] == {"dir": "native_cache/blake3-1.8.7", "present": False,
                           "bad": ["the directory"]}
    src = up / "native_cache" / "blake3-1.8.7"
    shutil.copytree(fs.native.source_dir(), src)
    assert fs.check_upload(up)["native"]["bad"] == [] and fs.check_upload(up)["native"]["present"]
    (src / "blake3.c").write_text("changed")
    assert fs.check_upload(up)["native"]["bad"] == ["blake3.c"]


# ---- the sampler (FS21) -----------------------------------------------------------------


def _sampler_session(tmp_path, plan):
    return fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                          deadline=time.time() + 60, cwd=REPO, say=lambda _: None)


def test_selftest_plan_writes_samples_for_every_part(tmp_path):
    pytest.importorskip("psutil")
    s = _sampler_session(tmp_path, fs.selftest_plan())
    assert s["finished"] and not s["failed"]
    for p in fs.selftest_plan():
        lines = (tmp_path / "out" / "parts" / p.name / "samples.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["kind"] == "header"
    assert len((tmp_path / "out" / "parts" / "selftest_sleep" / "samples.jsonl")
               .read_text().splitlines()) > 20


def test_pots_sampler_0_turns_the_sampler_off(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_SAMPLER", "0")
    s = _sampler_session(tmp_path, [fs.Part("p", _py("print(1)"))])
    assert s["finished"] and not (tmp_path / "out" / "parts" / "p" / "samples.jsonl").exists()


def test_a_sampler_that_cannot_start_never_fails_the_part(tmp_path, monkeypatch):
    from setup import sampler

    def boom(*a, **k):
        raise OSError("no sampler")
    monkeypatch.setattr(sampler, "start_sampler", boom)
    s = _sampler_session(tmp_path, [fs.Part("p", _py("print(1)"))])
    assert s["finished"] and s["parts"][0]["status"] == "ok"


def test_a_killed_part_leaves_no_sampler_behind(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    from setup import sampler
    monkeypatch.setattr(fs, "KILL_GRACE_S", 2)
    started = []
    real = sampler.start_sampler

    def spy(*a, **k):
        started.append(real(*a, **k))
        return started[-1]
    monkeypatch.setattr(sampler, "start_sampler", spy)
    s = _sampler_session(tmp_path, [fs.Part("long", ["sleep", "30"], timeout_s=2)])
    assert s["parts"][0]["status"] == "killed_timeout"
    assert started and started[0].proc.poll() is not None
