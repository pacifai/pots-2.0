"""FS32f's two-GPU session (``fs32-pair``): its plan, the comparison it writes and the job-log
lines it ends with, and the pieces that read a part's output."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from verification.runs import fs11_session as fs
from verification.runs import scenarios

JOB = Path(fs.__file__).with_name("fs11_job.sh")


def _part(name, kind, extras=None, **info):
    return {"name": name, "kind": kind, "info": info, "extras": extras}


def _run(arm, verdicts=("accepted",) * 3, final="a" * 8, kept="b" * 16):
    return _part(f"run_{arm}", "pair_run", {"verdicts": list(verdicts), "kept_root": kept,
                                            "final_root": final, "losses": [1.5, 1.25, 1.0]},
                 arm=arm)


def _card(arm, root="r1"):
    return _part(f"card_{arm}", "card", {"root": root, "w_next_root": "w1", "loss": 2.5}, arm=arm)


def _parts(**over):
    parts = [_card("unsharded"), _card("sharded"), _run("unsharded"), _run("sharded"),
             _part("cheats", "cheat", {"outcomes": {n: "exact" for n in fs.PAIR_CHEATS}}),
             _part("fast_suite", "pytest", {"tests": 9, "failures": 0, "errors": 0,
                                            "skipped": 2, "slowest_failing": None}),
             _part("two_gpu_slow", "pytest", {"tests": 2, "failures": 0, "errors": 0,
                                              "skipped": 0, "slowest_failing": None})]
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
    assert lines[1].startswith("FS32 verdicts equal: yes")
    assert "FS32 cheat flip: exact" in lines and lines[-1] == "FS32 slowest failing test: none"


def test_comparison_names_each_difference():
    bad = _run("sharded", verdicts=("accepted", "REJECTED", "accepted"), final="c" * 8)
    pair = fs.fs32_pair(_parts(run_sharded=bad,
                               cheats=_part("cheats", "cheat", {"outcomes": {"A1": "accepted"}}),
                               fast_suite=_part("fast_suite", "pytest", {
                                   "tests": 9, "failures": 2, "errors": 0, "skipped": 0,
                                   "slowest_failing": {"name": "t::slow", "time_s": 4.0}})))
    assert not pair["pass"] and not pair["roots_equal"] and not pair["verdicts_equal"]
    assert pair["roots"]["final_root"] is False and pair["roots"]["kept_root"] is True
    text = "\n".join(fs._pair_lines(pair))
    assert "FS32 roots equal: NO" in text and "final_root !=" in text
    assert "FS32 cheat A1: accepted (NOT exact)" in text
    assert "FS32 cheat A2: not run (NOT exact)" in text
    assert "FS32 slowest failing test: t::slow (4.0 s, fast_suite)" in text


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
    log.write_text("noise\nstep  1 (in-sample): accepted; x\nstep  2 (sample): REJECTED; y\n"
                   "stored step 3: 5 leaves, root 0123456789abcdef… (recomputed from the files)\n")
    w.write_text(json.dumps({"root": "ff", "losses": [1.0, 2.0]}))
    e = fs._pair_run_extras(log, w)
    assert e == {"verdicts": ["accepted", "REJECTED"], "kept_root": "0123456789abcdef",
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
