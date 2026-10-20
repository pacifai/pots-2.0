"""FS32f's two-GPU session (``fs32-pair``): its plan, the comparison it writes and the job-log
lines it ends with, and the pieces that read a part's output."""

import json
import subprocess
import time
from pathlib import Path

import pytest

from verification.runs import fs11_session as fs
from verification.runs import scenarios

JOB = Path(fs.__file__).with_name("fs11_job.sh")


def _part(name, kind, extras=None, status="ok", **info):
    return {"name": name, "kind": kind, "status": status, "info": info, "extras": extras}


def _run(arm, verdicts=("accepted",) * 3, final="a" * 8, kept="b" * 16):
    return _part(f"run_{arm}", "pair_run",
                 {"verdicts": list(verdicts), "kept_root": kept, "final_root": final,
                  "losses": [1.5, 1.25, 1.0], "step_roots": ["r1", "r2", "r3"],
                  "step_lines": [f"{v}; max residual 1.00" for v in verdicts]}, arm=arm)


def _card(arm, root="r1"):
    return _part(f"card_{arm}", "pair_card", {"root": root, "w_next_root": "w1",
                                              "plain_w_next_root": "w1", "loss": 2.5}, arm=arm)


def _suite(name, **kw):
    return _part(name, "pytest", {"tests": 9, "failures": 0, "errors": 0, "skipped": 0,
                                  "slowest_failing": None, **kw})


def _parts(**over):
    parts = [_part("models", "models"), _part("native_build", "other"),
             _suite("fast_suite", skipped=2), _suite("two_gpu_slow"),
             _part("smollm2_calibrate", "other"), _card("unsharded"), _card("sharded"),
             _run("unsharded"), _run("sharded"),
             _part("poisoned_step", "cheat",
                   {"outcomes": {n: "exact" for n in ("A1", "A2", "A3")}}),
             _part("hidden_steps", "cheat", {"outcomes": {"hidden": "exact"}}),
             _part("flipped_matmul", "cheat", {"outcomes": {"flip": "exact"}})]
    return [over.get(p["name"], p) for p in parts]


def test_oracle_outcomes_are_the_scenarios():
    assert fs.ORACLE_OUTCOMES == scenarios.OUTCOMES


def test_plan_has_the_parts_each_with_peaks(tmp_path):
    plan = fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")
    names = [p.name for p in plan]
    assert names[0] == "models" and "fast_suite" in names and "two_gpu_slow" in names
    assert all(p.info.get("gpu_peak") for p in plan)
    sharded = [p for p in plan if p.env.get("VERIF_VERIFIER_DEVICES") == fs.PAIR_DEVICES]
    assert sharded and not any(p.env.get("VERIF_VERIFIER_DEVICES") for p in plan
                               if p.name in ("fast_suite", "run_unsharded", "card_unsharded"))


def test_two_cards_of_any_type_pass_and_one_is_refused():
    ok = {"gpus": ["NVIDIA A40"] * 2, "cuda_available": True}
    assert fs.refusal(ok, True, fs.PAIR_CARDS) is None
    assert fs.refusal({**ok, "gpus": ["NVIDIA A40"]}, True, fs.PAIR_CARDS) is not None


def test_comparison_passes_when_everything_agrees():
    pair = fs.fs32_pair(_parts())
    assert pair["pass"] and pair["roots_equal"] and pair["verdicts_equal"]
    assert pair["slowest_failing"] is None
    lines = fs._pair_lines(pair)
    assert lines[0].startswith("FS32 roots equal: yes")
    assert "FS32 sharded plain and proved W_{t+1}: equal" in lines
    assert "FS32 cheat flip: exact" in lines and lines[-1] == "FS32 slowest failing test: none"


def test_one_step_root_or_line_that_differs_is_no_pass():
    other = _run("sharded")
    other["extras"]["step_roots"] = ["r1", "rX", "r3"]
    pair = fs.fs32_pair(_parts(run_sharded=other))
    assert not pair["pass"] and pair["roots"]["step_roots"] is False
    other = _run("sharded")
    other["extras"]["step_lines"][2] = "accepted; max residual 1.01"
    pair = fs.fs32_pair(_parts(run_sharded=other))
    assert not pair["pass"] and not pair["verdicts_equal"]
    short = _run("sharded")
    short["extras"]["step_roots"] = ["r1", "r2"]
    assert not fs.fs32_pair(_parts(run_sharded=short))["pass"]


def test_plain_and_proved_w_next_differing_is_printed_but_outside_pass():
    card = _card("sharded")
    card["extras"]["plain_w_next_root"] = "other"
    pair = fs.fs32_pair(_parts(card_sharded=card))
    assert pair["pass"] and pair["plain_vs_proved"] == {"unsharded": True, "sharded": False}
    assert "FS32 sharded plain and proved W_{t+1}: differ" in fs._pair_lines(pair)


def test_a_part_that_did_not_run_and_a_missing_report_print_their_own_fail_lines():
    pair = fs.fs32_pair(_parts(hidden_steps=_part("hidden_steps", "cheat", None,
                                                  status="not_started"),
                               fast_suite=_part("fast_suite", "pytest", {"junit": None})))
    text = "\n".join(fs._pair_lines(pair))
    assert not pair["pass"]
    assert "FS32 FAIL: hidden_steps did not run" in text
    assert "FS32 FAIL: fast_suite left no test report" in text and "None tests" not in text


def test_no_pair_parts_no_comparison():
    assert fs.fs32_pair([_part("x", "other")]) is None
    assert fs._pair_lines(None) == []


def test_pytest_extras_reads_counts_and_the_slowest_failure(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text('<testsuites><testsuite>'
                 '<testcase classname="a" name="ok" time="1"/>'
                 '<testcase classname="a" name="s" time="0"><skipped/></testcase>'
                 '<testcase classname="a" name="f1" time="2"><failure/></testcase>'
                 '<testcase classname="b" name="e" time="5"><error/></testcase>'
                 '</testsuite></testsuites>')
    e = fs._pytest_extras(x)
    assert (e["tests"], e["failures"], e["errors"], e["skipped"]) == (4, 1, 1, 1)
    assert e["slowest_failing"] == {"name": "b::e", "time_s": 5.0}
    assert fs._pytest_extras(tmp_path / "none.xml") == {"junit": None}


def test_pair_run_extras_reads_verdicts_roots_and_losses(tmp_path):
    log, w = tmp_path / "log", tmp_path / "w.json"
    log.write_text("noise\nstep  1 (in-sample): accepted; x\nstep  1 root " + "a" * 64 + "\n"
                   "step  2 (sample): REJECTED; y\n"
                   "stored step 3: 5 leaves, root 0123456789abcdef… (recomputed from files)\n")
    w.write_text(json.dumps({"root": "ff", "losses": [1.0, 2.0]}))
    e = fs._pair_run_extras(log, w)
    assert e == {"verdicts": ["accepted", "REJECTED"], "kept_root": "0123456789abcdef",
                 "step_lines": ["accepted; x", "REJECTED; y"], "step_roots": ["a" * 64],
                 "final_root": "ff", "losses": [1.0, 2.0]}
    assert fs._pair_run_extras(tmp_path / "no", tmp_path / "no")["verdicts"] == []


def test_cheat_extras_takes_the_outcome_column(tmp_path):
    log = tmp_path / "log"
    log.write_text("A1 poisoned x y exact 0.1 0.2 0.3\nflip a b c wrong-check 1 2 3\n"
                   "other a b c exact 1 2 3\nA2 short\n")
    assert fs._cheat_extras(log) == {"outcomes": {"A1": "exact", "flip": "wrong-check"}}


def test_gpu_peaks_keeps_the_maximum_per_card(tmp_path):
    seen = iter([{"0": 10, "1": 5}, {"0": 30, "1": 2}, {"0": 20}])
    peaks = fs._GpuPeaks(tmp_path / "p.json", every=0.01,
                         sample=lambda: next(seen, {}))
    time.sleep(0.3)
    peaks.stop()
    assert fs._read_peaks(tmp_path / "p.json") == {"0": 30, "1": 5}
    assert fs._read_peaks(tmp_path / "missing.json") is None


def test_job_script_accepts_the_session_and_gives_it_the_longer_deadline():
    text = JOB.read_text()
    assert "fs32-pair" in text and "DEFAULT_DEADLINE_S=11800" in text
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


def test_the_refusal_names_the_pairs_need_not_fs31_cards():
    why = fs.refusal({"gpus": ["NVIDIA A40"], "cuda_available": True}, True, fs.PAIR_CARDS,
                     need="the fs32-pair session")
    assert "sees 1 CUDA devices" in why and "needs 2" in why and "FS31_CARDS" not in why


def test_pair_card_parts_are_not_hw3_cards():
    assert fs.hw3_cards(_parts()) is None


def _fake_plan(tmp_path, root_sharded="a" * 64, outcome="exact", junit_for=("fast_suite",
                                                                            "two_gpu_slow")):
    """The pair's real plan with each part's command replaced by one that writes what the real
    part does (and the real part's kind, info and name)."""
    import dataclasses

    plan = fs.with_native_build(fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out"))
    junit = ("import sys;open(sys.argv[1],'w').write('<testsuites><testsuite><testcase "
             "classname=\"a\" name=\"b\" time=\"1\"/></testsuite></testsuites>')")
    card = ("import json,sys;json.dump({'card':{'name':'A40','uuid':'u','pci_bus_id':'p'},"
            "'plain':{'w_next_root':'w','loss':1.0},'proved':{'root':'r','w_next_root':'w',"
            "'loss':1.0},'n_leaves':3,'wall_s':{}},open(sys.argv[1],'w'))")
    run = ("import json,sys,os\nfor t in (1, 2, 3):\n"
           "    print(f'step  {t} (in-sample): accepted; max residual 1.00')\n"
           "    print(f'step  {t} root ' + ROOT[:63] + str(t))\n"
           "print('stored step 3: p, 3 files, 0.1 GiB, root ' + 'ab' * 8 + '… (r)')\n"
           "w = sys.argv[1]; os.makedirs(os.path.dirname(w), exist_ok=True)\n"
           "json.dump({'root': 'ff', 'losses': [1.0, 2.0, 3.0]}, open(w, 'w'))")
    cheats = {"poisoned_step": ("A1", "A2", "A3"), "hidden_steps": ("hidden",),
              "flipped_matmul": ("flip",)}
    out = []
    for p in plan:
        info = p.info
        if p.kind == "pytest" and p.name in junit_for:
            argv = ["{python}", "-c", junit, info["junit"]]
        elif p.kind == "pair_card":
            argv = ["{python}", "-c", card, info["report"]]
        elif p.kind == "pair_run":
            sharded = p.info["arm"] == "sharded"
            argv = ["{python}", "-c", run.replace("ROOT", repr(root_sharded if sharded
                                                                else "a" * 64)),
                    info["weights"]]
        elif p.kind == "cheat":
            argv = ["{python}", "-c", ";".join(f"print('{n} d a {outcome} 1 2 3')"
                                                for n in cheats[p.name])]
        else:
            argv = ["{python}", "-c", "pass"]
        out.append(dataclasses.replace(p, argv=argv, timeout_s=30))
    return out


def _session(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(fs, "_gpu_used_mib", lambda: {"0": 1, "1": 2})
    monkeypatch.setattr(fs, "check_upload", lambda up: {"s": 0.0, "packed": [], "models": []})
    lines = []
    s = fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 120, cwd=Path.cwd(), say=lines.append,
                       session="fs32-pair")
    return s, lines


def test_the_whole_job_log_tail_of_the_pairs_plan(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path))
    tail = lines[lines.index(next(x for x in lines if x.startswith("FS32 roots equal"))):]
    assert [x.split(":")[0] for x in tail] == [
        "FS32 roots equal", "FS32 verdicts equal",
        "FS32 unsharded plain and proved W_{t+1}", "FS32 sharded plain and proved W_{t+1}",
        "FS32 cheat A1", "FS32 cheat A2", "FS32 cheat A3", "FS32 cheat hidden",
        "FS32 cheat flip", "FS32 fast_suite", "FS32 two_gpu_slow", "FS32 slowest failing test"]
    assert not any("HW3" in x or "KeyError" in x for x in lines)
    assert "hw3_cards" not in s and "hw3_cards_error" not in s
    assert s["fs32_pair"]["pass"], s["fs32_pair"]


def test_a_differing_root_in_the_pairs_plan_reads_no(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path, root_sharded="b" * 64))
    assert not s["fs32_pair"]["pass"] and not s["fs32_pair"]["roots"]["step_roots"]
    assert any(x.startswith("FS32 roots equal: NO") for x in lines)


def test_a_missing_junit_in_the_pairs_plan_prints_a_fail_line(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path, junit_for=("fast_suite",)))
    assert "FS32 FAIL: two_gpu_slow left no test report" in lines
    assert not s["fs32_pair"]["pass"]
