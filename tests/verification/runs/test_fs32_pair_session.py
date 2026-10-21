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


def _run(arm, verdicts=("accepted",) * 3, final="a" * 8, kept="b" * 16, mode=None):
    mode = mode or ("pipelined" if arm.startswith("pipelined") else "sequential")
    times = [{"prover_s": 1.0, "verifier_s": 2.0, "handoff_s": 0.5, "total_s": 3.5}] * 3
    return _part(f"run_{arm}", "pair_run",
                 {"verdicts": list(verdicts), "kept_root": kept, "final_root": final,
                  "losses": [1.5, 1.25, 1.0], "step_roots": ["r1", "r2", "r3"],
                  "step_lines": [f"{v}; max residual 1.00" for v in verdicts],
                  "step_times": times, "loop_mode": mode}, arm=arm)


def _cheat(name, names, mode=None, outcome="exact", pipelined=False):
    mode = mode or ("pipelined" if pipelined else "sequential")
    return _part(name, "cheat",
                 {"outcomes": {n: outcome for n in names},
                  "oracle_lines": {n: f"{n} d a {outcome} 1 2 3" for n in names},
                  "loop_mode": mode}, **({"arm": "pipelined"} if pipelined else {}))


def _card(arm, root="r1"):
    return _part(f"card_{arm}", "pair_card", {"root": root, "w_next_root": "w1",
                                              "plain_w_next_root": "w1", "loss": 2.5}, arm=arm)


def _suite(name, **kw):
    return _part(name, "pytest", {"tests": 9, "failures": 0, "errors": 0, "skipped": 0,
                                  "slowest_failing": None, **kw})


EXPECTED34 = fs.device_tests_expected(fs.FS34_TESTS_FILE)[0]  # every case the real file defines
assert EXPECTED34


def _fs34(outcomes=None, **kw):
    cases = {n: "passed" for n in EXPECTED34} | (outcomes or {})
    return _part("fs34_tests", "fs34_tests",
                 {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0,
                  "slowest_failing": None, "junit": 1, "cases": cases}, **kw)


def _parts(**over):
    parts = [_part("peer_check", "peer_check", {"peer_check": {"all_ok": True, "wrong": {}}}),
             _part("models", "models"), _part("native_build", "other"),
             _suite("fast_suite", skipped=2), _suite("two_gpu_slow"), _fs34(),
             _part("smollm2_calibrate", "other"), _card("unsharded"), _card("sharded"),
             _run("unsharded"), _run("sharded"), _run("pipelined"), _run("pipelined_metrics"),
             _cheat("poisoned_step", ("A1", "A2", "A3")), _cheat("hidden_steps", ("hidden",)),
             _cheat("flipped_matmul", ("flip",)),
             _cheat("poisoned_step_pipelined", ("A1", "A2", "A3"), pipelined=True),
             _cheat("hidden_steps_pipelined", ("hidden",), pipelined=True),
             _cheat("flipped_matmul_pipelined", ("flip",), pipelined=True)]
    return [over.get(p["name"], p) for p in parts]


def test_oracle_outcomes_are_the_scenarios():
    assert fs.ORACLE_OUTCOMES == scenarios.OUTCOMES


def test_plan_has_the_parts_each_with_peaks(tmp_path):
    plan = fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")
    names = [p.name for p in plan]
    assert names[:2] == ["peer_check", "models"]
    assert "fast_suite" in names and "two_gpu_slow" in names
    assert all(p.info.get("gpu_peak") for p in plan)
    sharded = [p for p in plan if p.env.get("VERIF_VERIFIER_DEVICES") == fs.PAIR_DEVICES]
    assert sharded and not any(p.env.get("VERIF_VERIFIER_DEVICES") for p in plan
                               if p.name in ("fast_suite", "run_unsharded", "card_unsharded"))


def test_plan_has_the_pipelined_arms_sharded_and_only_they_set_the_mode(tmp_path):
    plan = {p.name: p for p in fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")}
    piped = {"run_pipelined", "run_pipelined_metrics", "poisoned_step_pipelined",
             "hidden_steps_pipelined", "flipped_matmul_pipelined"}
    assert piped <= set(plan) and list(plan)[0] == "peer_check"
    assert set(fs.PAIR_PARTS) - {"native_build"} == set(plan)  # the job adds the native build
    assert {n for n, p in plan.items() if p.env.get("VERIF_PIPELINED")} == piped
    assert all(plan[n].env["VERIF_PIPELINED"] == "1"
               and plan[n].env["VERIF_VERIFIER_DEVICES"] == fs.PAIR_DEVICES for n in piped)
    dirs = [p.env["VERIF_OUTPUT_DIR"] for n, p in plan.items() if n in piped | {
        "run_sharded", "poisoned_step", "hidden_steps", "flipped_matmul"}]
    assert len(set(dirs)) == len(dirs)  # each arm in its own run directory
    run, metrics = plan["run_pipelined"], plan["run_pipelined_metrics"]
    for p in (run, metrics, plan["run_sharded"]):
        assert p.kind == "pair_run" and "--keep-step" in p.argv
        assert p.argv[p.argv.index("--steps") + 1] == str(fs.PAIR_STEPS)
    assert "--no-metrics" in run.argv and "--metrics" in metrics.argv
    assert run.info["arm"] == "pipelined" and metrics.info["arm"] == "pipelined_metrics"
    assert metrics.timeout_s == fs.PAIR_TIMEOUT_S["run_metrics"]
    assert plan["flipped_matmul_pipelined"].argv[-2:] == ["--no-sweep", "--no-metrics"]
    assert all(plan[n].kind == "cheat" and plan[n].info["arm"] == "pipelined" for n in piped
               if n.endswith("_pipelined") and n != "run_pipelined")


def test_the_default_deadline_is_twice_the_estimate_and_the_job_holds_it():
    assert fs.pair_default_deadline_s() == 16100
    assert "DEFAULT_DEADLINE_S=16100" in JOB.read_text()


def test_two_cards_of_any_type_pass_and_one_is_refused():
    ok = {"gpus": ["NVIDIA A40"] * 2, "cuda_available": True}
    assert fs.refusal(ok, True, fs.PAIR_CARDS) is None
    assert fs.refusal({**ok, "gpus": ["NVIDIA A40"]}, True, fs.PAIR_CARDS) is not None


def test_comparison_passes_when_everything_agrees():
    pair = fs.fs32_pair(_parts())
    assert pair["pass"] and pair["roots_equal"] and pair["verdicts_equal"]
    assert not pair["mode_problems"] and pair["slowest_failing"] is None
    assert set(pair["step_times"]) == {"unsharded", "sharded", "pipelined", "pipelined_metrics"}
    assert pair["step_times"]["pipelined"][0]["total_s"] == 3.5
    assert set(pair["gpu_peak"]) == set(fs.PAIR_PARTS)
    lines = fs._pair_lines(pair)
    assert lines[0].startswith("FS32 roots equal: yes")
    assert "FS32 sharded plain and proved W_{t+1}: equal" in lines
    assert "FS32 cheat flip: exact" in lines and "FS32 pipelined cheat flip: exact" in lines
    assert any(x.startswith("FS32 step wall time pipelined") and "1.0/2.0/0.5/3.5" in x
               for x in lines)
    assert lines[-1] == "FS32 slowest failing test: none"


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


def test_a_pipelined_arm_that_differs_from_the_sequential_one_is_no_pass():
    for arm, key, value in (("pipelined", "final_root", "zz"), ("pipelined_metrics", "kept_root",
                                                                "c" * 16),
                            ("pipelined", "step_roots", ["r1", "rX", "r3"])):
        other = _run(arm)
        other["extras"][key] = value
        pair = fs.fs32_pair(_parts(**{f"run_{arm}": other}))
        assert not pair["pass"] and pair["roots"][f"{arm} {key}"] is False
    other = _run("pipelined")
    other["extras"]["step_lines"][1] = "accepted; max residual 1.01"
    pair = fs.fs32_pair(_parts(run_pipelined=other))
    assert not pair["pass"] and not pair["verdicts_equal"]
    other = _run("pipelined_metrics", verdicts=("accepted", "REJECTED", "accepted"))
    assert not fs.fs32_pair(_parts(run_pipelined_metrics=other))["honest"]


def test_a_pipelined_cheat_that_differs_is_no_pass():
    off = _cheat("hidden_steps_pipelined", ("hidden",), outcome="accepted", pipelined=True)
    pair = fs.fs32_pair(_parts(hidden_steps_pipelined=off))
    assert not pair["pass"] and pair["pipelined_cheats"]["hidden"] == "accepted"
    assert "FS32 pipelined cheat hidden: accepted (NOT exact); oracle line differs from " \
        "sequential" in fs._pair_lines(pair)
    row = _cheat("flipped_matmul_pipelined", ("flip",), pipelined=True)
    row["extras"]["oracle_lines"]["flip"] = "flip d a exact 1 2 4"  # exact, but another row
    pair = fs.fs32_pair(_parts(flipped_matmul_pipelined=row))
    assert not pair["pass"] and pair["oracle_equal"]["flip"] is False


def test_a_part_in_the_wrong_loop_mode_prints_its_fail_line():
    pair = fs.fs32_pair(_parts(run_pipelined=_run("pipelined", mode="sequential"),
                               run_sharded=_run("sharded", mode="pipelined")))
    text = fs._pair_lines(pair)
    assert not pair["pass"] and pair["mode_problems"] == {"run_pipelined": "sequential",
                                                          "run_sharded": "pipelined"}
    assert "FS32 FAIL: run_pipelined ran sequentially" in text
    assert "FS32 FAIL: run_sharded ran pipelined" in text
    nothing = _cheat("hidden_steps_pipelined", ("hidden",), pipelined=True)
    nothing["extras"]["loop_mode"] = None
    text = fs._pair_lines(fs.fs32_pair(_parts(hidden_steps_pipelined=nothing)))
    assert "FS32 FAIL: hidden_steps_pipelined printed no loop mode line" in text


@pytest.mark.parametrize("lines", [["loop mode: pipelined", "x", "loop mode: sequential"],
                                   ["loop mode: sequential", "loop mode: pipelined"],
                                   ["loop mode: pipelined", "loop mode: pipelined"]])
def test_two_loop_mode_lines_are_never_a_mode(lines):
    assert fs._loop_mode(lines) == "multiple"
    row = _cheat("hidden_steps_pipelined", ("hidden",), pipelined=True)
    row["extras"]["loop_mode"] = fs._loop_mode(lines)
    pair = fs.fs32_pair(_parts(hidden_steps_pipelined=row))
    assert not pair["pass"] and pair["mode_problems"] == {"hidden_steps_pipelined": "multiple"}
    assert ("FS32 FAIL: hidden_steps_pipelined printed more than one loop mode line"
            in fs._pair_lines(pair))


def test_one_loop_mode_line_is_that_mode_and_none_is_none():
    assert fs._loop_mode(["a", "loop mode: pipelined", "b"]) == "pipelined"
    assert fs._loop_mode(["loop mode: sequential"]) == "sequential"
    assert fs._loop_mode(["nothing"]) is None


def test_a_missing_pipelined_part_is_a_fail_line_and_no_pass():
    parts = [p for p in _parts() if p["name"] != "run_pipelined"]
    pair = fs.fs32_pair(parts)
    assert not pair["pass"] and pair["part_problems"]["run_pipelined"] == "not_started"
    assert "FS32 FAIL: run_pipelined did not run" in fs._pair_lines(pair)
    parts = [p for p in _parts() if p["name"] != "poisoned_step_pipelined"]
    pair = fs.fs32_pair(parts)
    assert not pair["pass"] and pair["pipelined_cheats"]["A1"] is None


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
                 "final_root": "ff", "losses": [1.0, 2.0], "step_times": [],
                 "loop_mode": None}
    log.write_text("loop mode: pipelined\nstep 1: prover 5.00 s (prove_step 4.00 s, commit 1.00 s)"
                   "; verifier 7.50 s (0 0.50, 1 1.20); hand-off 0.25 s\n")
    e = fs._pair_run_extras(log, w)
    assert e["loop_mode"] == "pipelined"
    assert (e["step_times"][0]["prover_s"], e["step_times"][0]["verifier_s"],
            e["step_times"][0]["handoff_s"], e["step_times"][0]["total_s"]) == (
        5.0, 7.5, 0.25, 12.75)
    assert fs._pair_run_extras(tmp_path / "no", tmp_path / "no")["verdicts"] == []


def test_cheat_extras_takes_the_outcome_column(tmp_path):
    log = tmp_path / "log"
    log.write_text("A1 poisoned x y exact 0.1 0.2 0.3\nflip a b c wrong-check 1 2 3\n"
                   "other a b c exact 1 2 3\nA2 short\n")
    assert fs._cheat_extras(log) == {
        "outcomes": {"A1": "exact", "flip": "wrong-check"}, "loop_mode": None,
        "oracle_lines": {"A1": "A1 poisoned x y exact 0.1 0.2 0.3",
                         "flip": "flip a b c wrong-check 1 2 3"}}
    log.write_text("loop mode: sequential\n")
    assert fs._cheat_extras(log)["loop_mode"] == "sequential"


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
    assert "fs32-pair" in text and "DEFAULT_DEADLINE_S=%d" % fs.pair_default_deadline_s() in text
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


def test_the_refusal_names_the_pairs_need_not_fs31_cards():
    why = fs.refusal({"gpus": ["NVIDIA A40"], "cuda_available": True}, True, fs.PAIR_CARDS,
                     need="the fs32-pair session")
    assert "sees 1 CUDA devices" in why and "needs 2" in why and "FS31_CARDS" not in why


def test_pair_card_parts_are_not_hw3_cards():
    assert fs.hw3_cards(_parts()) is None


# a stand-in for ``peer_check``: all right, or the pair-to-wrong-cases map WRONG
PEER_WRITER = ("import json,sys,os\nos.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)\n"
               "w = WRONG or {}\n"
               "pairs = {'0->1': {'ok': '0->1' not in w, 'cases': {'host_staged': True, 'to_4KB': "
               "'0->1' not in w}, 'gbps': 7.0}, '1->0': {'ok': True, 'cases': {'host_staged': "
               "True}, 'gbps': 8.0}}\n"
               "json.dump({'all_ok': not w, 'pairs': pairs, 'seconds': 1.0}, open(sys.argv[1], "
               "'w'))\nsys.exit(1 if w else 0)")


def _fake_plan(tmp_path, root_sharded="a" * 64, outcome="exact", junit_for=("fast_suite",
                                                                            "two_gpu_slow"),
               peer_wrong=None):
    """The pair's real plan with each part's command replaced by one that writes what the real
    part does (and the real part's kind, info and name)."""
    import dataclasses

    plan = fs.with_native_build(fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out"))
    junit = ("import sys;open(sys.argv[1],'w').write('<testsuites><testsuite><testcase "
             "classname=\"a\" name=\"b\" time=\"1\"/></testsuite></testsuites>')")
    card = ("import json,sys;json.dump({'card':{'name':'A40','uuid':'u','pci_bus_id':'p'},"
            "'plain':{'w_next_root':'w','loss':1.0},'proved':{'root':'r','w_next_root':'w',"
            "'loss':1.0},'n_leaves':3,'wall_s':{}},open(sys.argv[1],'w'))")
    run = ("import json,sys,os\nprint('loop mode: ' + MODE)\nfor t in (1, 2, 3):\n"
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
        if p.kind == "peer_check":
            argv = ["{python}", "-c", PEER_WRITER.replace("WRONG", repr(peer_wrong)),
                    info["report"]]
        elif p.kind == "fs34_tests":
            cases = "".join(f'<testcase classname="{n.split("::")[0]}" '
                            f'name="{n.split("::")[1]}" time="1"/>' for n in EXPECTED34)
            argv = ["{python}", "-c", "import sys;open(sys.argv[1],'w').write("
                    f"'<testsuites><testsuite>{cases}</testsuite></testsuites>')",
                    info["junit"]]
        elif p.kind == "pytest" and p.name in junit_for:
            argv = ["{python}", "-c", junit, info["junit"]]
        elif p.kind == "pair_card":
            argv = ["{python}", "-c", card, info["report"]]
        elif p.kind == "pair_run":
            sharded = p.info["arm"] == "sharded"
            sharded = p.info["arm"] != "unsharded"
            mode = "pipelined" if p.info["arm"].startswith("pipelined") else "sequential"
            argv = ["{python}", "-c", run.replace("ROOT", repr(root_sharded if sharded
                                                                else "a" * 64)
                                                  ).replace("MODE", repr(mode)),
                    info["weights"]]
        elif p.kind == "cheat":
            mode = "pipelined" if p.name.endswith("_pipelined") else "sequential"
            argv = ["{python}", "-c", f"print('loop mode: {mode}');" + ";".join(
                f"print('{n} d a {outcome} 1 2 3')" for n in cheats[p.name.removesuffix(
                    '_pipelined')])]
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
        "FS32 cheat flip", "FS32 pipelined cheat A1", "FS32 pipelined cheat A2",
        "FS32 pipelined cheat A3", "FS32 pipelined cheat hidden", "FS32 pipelined cheat flip",
        "FS32 step wall time unsharded (prover/verifier/hand-off/total s)",
        "FS32 step wall time sharded (prover/verifier/hand-off/total s)",
        "FS32 step wall time pipelined (prover/verifier/hand-off/total s)",
        "FS32 step wall time pipelined_metrics (prover/verifier/hand-off/total s)",
        "FS32 fast_suite", "FS32 two_gpu_slow", "FS32 slowest failing test"]
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


# ---- F22: the peer-copy byte check first, and the session stops on it -------------------------


def test_a_session_with_right_peer_copies_runs_every_part(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path))
    assert "stopped" not in s and s["fs32_pair"]["pass"] and s["fs32_pair"]["peer_bytes_ok"]
    peer = next(p for p in s["parts"] if p["name"] == "peer_check")
    assert peer["status"] == "ok" and peer["extras"]["peer_check"]["all_ok"] is True
    assert peer["extras"]["peer_check"]["min_gbps"] == 7.0
    assert not any("wrong bytes" in x for x in lines)


def test_wrong_peer_copies_stop_the_session_and_every_later_part_says_why(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path, peer_wrong={"0->1": 1}))
    by = {p["name"]: p for p in s["parts"]}
    assert by["native_build"]["status"] == "ok" and by["peer_check"]["status"] == "failed"
    assert "wrong bytes on pair 0->1 (to_4KB)" in s["stopped"]
    later = [p for p in s["parts"] if p["name"] not in ("native_build", "peer_check")]
    assert later and all(p["status"] == "not_started" and p["reason"] == s["stopped"]
                         for p in later)
    assert not (tmp_path / "out" / "parts" / "models" / "DONE").exists()
    pair = s["fs32_pair"]
    assert not pair["pass"] and pair["peer_bytes_ok"] is False
    assert any(x.startswith("FS32 FAIL: peer copies landed wrong bytes on pair 0->1 (to_4KB)")
               for x in lines)


def test_a_stopped_rerun_reads_the_reports_of_a_done_check_again(tmp_path, monkeypatch):
    plan = _fake_plan(tmp_path)
    s, _ = _session(tmp_path, monkeypatch, plan)
    assert s["fs32_pair"]["pass"]
    report = Path(next(p for p in plan if p.kind == "peer_check").info["report"])
    doc = json.loads(report.read_text())
    doc["all_ok"] = False  # the DONE marker is there, the report now says wrong
    doc["pairs"]["1->0"].update(ok=False, cases={"host_staged": True, "side_stream": False})
    report.write_text(json.dumps(doc))
    s, _ = _session(tmp_path, monkeypatch, plan)
    assert next(p for p in s["parts"] if p["name"] == "peer_check")["status"] == "skipped_done"
    assert "pair 1->0 (side_stream)" in s["stopped"]
    assert s["parts"][-1]["status"] == "not_started"


def test_a_missing_report_or_a_dead_check_stops_the_session(tmp_path, monkeypatch):
    import dataclasses

    plan = [dataclasses.replace(p, argv=["{python}", "-c", "import sys; sys.exit(1)"])
            if p.kind == "peer_check" else p for p in _fake_plan(tmp_path)]
    s, lines = _session(tmp_path, monkeypatch, plan)
    assert "left no report" in s["stopped"] and "ended failed" in s["stopped"]
    assert any("FS32 FAIL" in x and "left no report" in x for x in lines)


def test_a_card_that_fails_its_own_control_is_named_as_the_card(tmp_path, monkeypatch):
    plan = _fake_plan(tmp_path)
    writer = ("import json,sys,os; os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); "
              "json.dump({'all_ok': False, 'seconds': 1.0, 'pairs': {'1->0': "
              "{'ok': False, 'card_suspect': True, 'cases': {'host_staged': False}}}}, "
              "open(sys.argv[1], 'w')); sys.exit(1)")
    import dataclasses

    plan = [dataclasses.replace(p, argv=["{python}", "-c", writer, p.info["report"]])
            if p.kind == "peer_check" else p for p in plan]
    s, _ = _session(tmp_path, monkeypatch, plan)
    assert "the card itself is suspect, not the peer path" in s["stopped"]
    assert "pair 1->0" in s["stopped"]


def test_the_gate_fails_without_a_peer_check_part_or_with_wrong_bytes():
    assert fs.fs32_pair(_parts())["pass"]
    gone = [p for p in _parts() if p["name"] != "peer_check"]
    assert not fs.fs32_pair(gone)["pass"] and "peer_check" in fs.fs32_pair(gone)["part_problems"]
    bad = _part("peer_check", "peer_check", {"peer_check": {
        "all_ok": False, "wrong": {"1->0": ["copy_4MB"]}, "card_suspect": []}})
    pair = fs.fs32_pair(_parts(peer_check=bad))
    assert not pair["pass"] and "1->0 (copy_4MB)" in pair["peer_copies"]
    line = fs._pair_lines(pair)[0]
    assert line.startswith("FS32 FAIL: peer copies landed wrong bytes") and "1->0" in line
    # an ok exit with a report that says not all right still fails the gate
    lying = _part("peer_check", "peer_check", {"peer_check": {"all_ok": False, "wrong": {}}})
    assert not fs.fs32_pair(_parts(peer_check=lying))["pass"]


# ---- FS34: the join's CUDA tests, each passed and none skipped ---------------------------------


def test_the_plan_has_the_fs34_tests_part_after_the_slow_tests(tmp_path):
    plan = fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")
    names = [p.name for p in plan]
    assert names.index("fs34_tests") == names.index("two_gpu_slow") + 1
    p = plan[names.index("fs34_tests")]
    assert fs.FS34_TESTS_FILE in p.argv and p.kind == "fs34_tests" and p.info["gpu_peak"]
    assert "fs34_tests" in fs.PAIR_PARTS and "fs34_tests" in fs.PAIR_ESTIMATE_MIN


def test_the_fs34_tests_all_passing_is_part_of_pass():
    pair = fs.fs32_pair(_parts())
    assert pair["pass"] and pair["fs34_problems"] == []


@pytest.mark.parametrize("outcome", ["skipped", "failed", "error"])
def test_a_fs34_test_that_did_not_pass_is_no_pass_and_prints_its_fail_line(outcome):
    pair = fs.fs32_pair(_parts(fs34_tests=_fs34({EXPECTED34[1]: outcome})))
    text = "\n".join(fs._pair_lines(pair))
    assert not pair["pass"]
    assert f"FS32 FAIL: FS34 test {EXPECTED34[1]}: {outcome}" in text


def test_a_fs34_test_missing_from_the_report_is_no_pass():
    part = _fs34()
    del part["extras"]["cases"][EXPECTED34[2]]
    pair = fs.fs32_pair(_parts(fs34_tests=part))
    assert not pair["pass"]
    assert any("expected but not in the junit report" in w for w in pair["fs34_problems"])


def test_a_fs34_part_with_no_report_or_that_did_not_run_is_no_pass():
    none = _part("fs34_tests", "fs34_tests", {"junit": None})
    pair = fs.fs32_pair(_parts(fs34_tests=none))
    assert not pair["pass"] and "the FS34 tests left no junit report" in pair["fs34_problems"]
    pair = fs.fs32_pair(_parts(fs34_tests=_part("fs34_tests", "fs34_tests", None,
                                                status="not_started")))
    assert not pair["pass"]
    assert "FS32 FAIL: fs34_tests did not run" in "\n".join(fs._pair_lines(pair))


def test_a_failed_collection_of_the_expected_tests_is_no_pass(monkeypatch):
    monkeypatch.setattr(fs, "device_tests_expected", lambda file="": ([], "boom"))
    assert not fs.fs32_pair(_parts())["pass"]
