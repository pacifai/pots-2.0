"""FS35's four-H100 session (``four-h100``): its plan and refusal, the result it writes and the
job-log lines it ends with, and the pieces that read a part's output."""

import copy
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import pytest

from verification.runs import fs11_session as fs

JOB = Path(fs.__file__).with_name("fs11_job.sh")
KEYS = [fs.four_model_key(r) for r, _ in fs.FOUR_PINS]
EXPECTED34 = fs.device_tests_expected(fs.FS34_TESTS_FILE)[0]
assert EXPECTED34 and KEYS
ROOTS = ["%064x" % i for i in (1, 2, 3)]
UUIDS = [f"GPU-{i}" for i in range(4)]
CARD_MIB = {"0": 40000, "1": 20000, "2": 21000, "3": 22000}


def _part(name, kind, extras=None, status="ok", **info):
    return {"name": name, "kind": kind, "status": status, "info": info, "extras": extras}


def _gpus(n=4):
    return [{"index": i, "uuid": UUIDS[i]} for i in range(n)]


def _roles(roles=None):
    roles = roles or {1: ["verifier"]}
    return [{"device": f"cuda:{i}", "index": i, "uuid": UUIDS[i],
             "roles": roles.get(i, ["prover"] if i == 0 else ["verifier"])} for i in range(4)]


def _arm(k, arm, roots=None, accepted=True, mode=None, gpus=None, roles=None, wall=True):
    mode = mode or {"seq": "sequential", "pipe": "pipelined"}[arm]
    times = [{"prover_s": 1.0, "verifier_s": 2.0, "handoff_s": 0.5, "total_s": 3.5 + i}
             for i in range(3)]
    ex = {"accepted": accepted, "actual": "accept" if accepted else "reject",
          "step_roots": ROOTS if roots is None else roots, "loop_mode": mode,
          "step_times": times, "record": "r", "gpus": _gpus() if gpus is None else gpus,
          "gpu_roles": _roles() if roles is None else roles, "gpu_held_s": [9.0, 8.0, 8.0],
          "step_wall_s": [4.0, 3.0, 2.0] if wall else [],
          "device_peaks": {"cuda:0": {"peak_bytes": 5, "peak_reserved_bytes": 6}},
          "gpu_peak_mib": CARD_MIB}
    return _part(f"{arm}_{k}", "four_run", ex, arm=arm)


def _tests(**over):
    cases = {n: "passed" for n in EXPECTED34}
    files = {f: {"cases": 2, "skipped": 0} for f in (fs.FS34_TESTS_FILE, *fs.FOUR_TEST_FILES)}
    ex = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "junit": 1,
          "cases": cases, "files": files, "missing_files": [], "slowest_failing": None}
    return _part("four_tests", "four_tests", {**ex, **over})


def _probe():
    per = {f"cuda:{i}": {"gemm_bf16_flops": 7e14, "hbm_copy_bytes": 2.5e12,
                         "hash_kernel_per_byte_s": 1e-11} for i in range(4)}
    peer = {f"cuda:{a}->cuda:{b}": 3e11 + a * 1e10 + b * 1e9
            for a in range(4) for b in range(4) if a != b}
    return _part("probe", "four_probe", {"version": 2, "devices": list(per), "per_device": per,
                                         "peer": peer, "peer_errors": {}})


def _parts(**over):
    parts = [_part("native_build", "other"),
             _part("peer_check", "peer_check",
                   {"peer_check": {"all_ok": True, "wrong": {}, "gbps": {"0->1": 380.0,
                                                                          "1->0": 370.0}}}),
             _part("models", "models"), _tests(), _probe()]
    for k in KEYS:
        parts += [_part(f"card_{k}", "four_card", {"root": ROOTS[0]}),
                  _arm(k, "seq"), _arm(k, "pipe")]
    return [over.get(p["name"], p) for p in parts]


def _four(parts=None):
    return fs.four_h100(_parts() if parts is None else parts, EXPECTED34)


def _text(res):
    return "\n".join(t for ok, t in res["items"] if not ok)


# ---- the plan ---------------------------------------------------------------------------------


def test_plan_has_the_parts_in_order_each_with_peaks(tmp_path):
    plan = fs.with_native_build(fs.four_h100_plan(tmp_path / "up", tmp_path / "out"))
    names = [p.name for p in plan]
    assert names == list(fs.four_h100_parts())
    assert names[:5] == ["native_build", "peer_check", "models", "four_tests", "probe"]
    assert KEYS == ["Qwen2.5-0.5B", "Llama-3.2-1B", "Falcon3-1B", "Qwen2.5-1.5B"]
    assert names[5:] == [f"{a}_{k}" for k in KEYS for a in ("card", "seq", "pipe")]
    assert all(p.info.get("gpu_peak") for p in plan[1:])


def test_only_the_pipelined_parts_set_the_mode_and_each_has_its_verifier_devices(tmp_path):
    plan = {p.name: p for p in fs.four_h100_plan(tmp_path / "up", tmp_path / "out")}
    assert {n for n, p in plan.items() if p.env.get("VERIF_PIPELINED")} == {
        f"pipe_{k}" for k in KEYS}
    assert all(p.env["VERIF_PIPELINED"] == "1" for n, p in plan.items() if n.startswith("pipe_"))
    sharded = {f"{a}_{k}" for k in KEYS for a in ("seq", "pipe")} | {"probe"}
    for n, p in plan.items():
        want = fs.FOUR_VERIFIERS if n in sharded else None
        assert p.env.get("VERIF_VERIFIER_DEVICES") == want, n
    for n in sharded:
        assert plan[n].env["VERIF_DEVICE"] == fs.FOUR_PROVER == "cuda:0"
    assert fs.FOUR_VERIFIERS == "cuda:1,cuda:2,cuda:3"


def test_the_card_and_arm_parts_run_the_right_commands_in_their_own_directories(tmp_path):
    plan = {p.name: p for p in fs.four_h100_plan(tmp_path / "up", tmp_path / "out")}
    dirs = []
    for k in KEYS:
        card, seq, pipe = plan[f"card_{k}"], plan[f"seq_{k}"], plan[f"pipe_{k}"]
        assert card.kind == "four_card" and "verification.runs.card_identity" in card.argv
        assert card.env["CUDA_VISIBLE_DEVICES"] == "0"
        assert card.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        for p in (seq, pipe):
            assert p.kind == "four_run" and "verification.runs.llama_step" in p.argv
            assert p.argv[p.argv.index("--steps") + 1] == "3"
            assert "--metrics" in p.argv and "--no-count" in p.argv
            assert p.env["VERIF_ATTACK"] == fs.BRING_UP_TASK
            assert p.info["records"].endswith("llama_step/records.jsonl")
        assert seq.info["arm"] == "seq" and pipe.info["arm"] == "pipe"
        dirs += [p.env["VERIF_OUTPUT_DIR"] for p in (card, seq, pipe)]
    assert len(set(dirs)) == len(dirs)
    # cheapest model first; the limits are about twice the estimates
    assert plan["card_" + KEYS[0]].timeout_s == fs.FOUR_TIMEOUT_S["card"]
    assert plan["seq_" + KEYS[0]].timeout_s == fs.FOUR_TIMEOUT_S["seq"]


def test_the_tests_part_selects_the_slow_tests_fs34_and_the_new_files(tmp_path):
    plan = {p.name: p for p in fs.four_h100_plan(tmp_path / "up", tmp_path / "out")}
    t, probe = plan["four_tests"], plan["probe"]
    assert t.kind == "four_tests" and any(a.startswith("--junitxml=") for a in t.argv)
    expr = t.argv[t.argv.index("-k") + 1]
    for f in (fs.FS34_TESTS_FILE, *fs.FOUR_TEST_FILES):
        assert Path(f).stem in expr
    assert "test_check5_profile_cuda" not in expr and len(fs.FOUR_TEST_FILES) == 3
    assert "slow or not slow" in t.argv
    assert probe.kind == "four_probe" and probe.argv[-2:] == ["--seconds", "60"]
    assert probe.env["VERIF_PIPELINED"] is None


def test_the_session_is_a_multi_gpu_session_with_a_peer_check_first(tmp_path):
    assert "four-h100" in fs.MULTI_GPU_SESSIONS and "four-h100" in fs.SESSIONS
    name, build = fs.SESSIONS["four-h100"]
    plan = fs.with_native_build(build(tmp_path / "up", tmp_path / "out"))
    assert [p.name for p in plan[:2]] == ["native_build", "peer_check"]


# ---- the refusal and the deadline -------------------------------------------------------------


def _machine(name, n):
    return {"gpus": [name] * n, "cuda_available": True}


def test_three_devices_are_refused_even_with_any_gpu_allowed():
    for allow in (False, True):
        why = fs.refusal(_machine("NVIDIA H100 80GB HBM3", 3), allow, fs.FOUR_CARDS,
                         need="the four-H100 session")
        assert "sees 3 CUDA devices" in why and "the four-H100 session needs 4" in why


def test_four_h100s_are_accepted_and_four_other_cards_only_when_allowed():
    h100 = _machine("NVIDIA H100 80GB HBM3", 4)
    a100 = _machine("NVIDIA A100-SXM4-80GB", 4)
    assert fs.refusal(h100, False, fs.FOUR_CARDS, need="the four-H100 session") is None
    assert "no H100" in fs.refusal(a100, False, fs.FOUR_CARDS, need="the four-H100 session")
    assert fs.refusal(a100, True, fs.FOUR_CARDS, need="the four-H100 session") is None


def test_the_default_deadline_is_twice_the_estimate_and_the_job_holds_it():
    assert fs.four_estimated_min() == 101
    assert fs.four_default_deadline_s() == 14700  # (101 + 1 + 20) * 2 * 60 = 14640, up to 100 s
    text = JOB.read_text()
    assert "four-h100" in text
    assert f"[ \"$SESSION\" = four-h100 ] && DEFAULT_DEADLINE_S={fs.four_default_deadline_s()}" \
        in text
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


# ---- the result -------------------------------------------------------------------------------


def test_everything_good_passes():
    res = _four()
    assert res["pass"], _text(res)
    assert all(ok for ok, _ in res["items"])
    arm = res["models"][KEYS[0]]["arms"]["seq"]
    assert arm["steady_s"] == pytest.approx(2.5) and arm["steady_source"] == "step_wall"
    assert arm["gpu_held_s"] == [9.0, 8.0, 8.0] and arm["gpu_peak_mib"] == CARD_MIB
    assert res["probe"]["devices"] == [f"cuda:{i}" for i in range(4)]
    assert len(res["probe"]["peer"]) == 12 and res["peer_check_gbps"]["0->1"] == 380.0
    assert res["gpu_peak"]["seq_" + KEYS[0]] == CARD_MIB


def test_no_four_parts_no_result():
    assert fs.four_h100([_part("peer_check", "peer_check")]) is None


def test_the_steady_time_falls_back_to_the_total_without_step_wall():
    res = _four(_parts(**{f"seq_{KEYS[0]}": _arm(KEYS[0], "seq", wall=False)}))
    arm = res["models"][KEYS[0]]["arms"]["seq"]
    assert arm["steady_source"] == "total" and arm["steady_s"] == pytest.approx(5.0)
    assert res["pass"]


def test_a_root_that_differs_between_arms_is_no_pass():
    k = KEYS[1]
    res = _four(_parts(**{f"pipe_{k}": _arm(k, "pipe", roots=[ROOTS[0], ROOTS[1], "%064x" % 9])}))
    assert not res["pass"] and not res["models"][k]["roots_equal"]
    assert "the sequential and pipelined arms' step roots differ" in _text(res)
    assert res["models"][KEYS[0]]["roots_equal"]


def test_a_step_one_root_that_differs_from_the_cards_is_no_pass():
    k = KEYS[2]
    res = _four(_parts(**{f"card_{k}": _part(f"card_{k}", "four_card", {"root": "%064x" % 7})}))
    assert not res["pass"] and not res["models"][k]["step1_equals_card"]
    assert f"{k}: step 1's root differs from card_{k}'s proved root" in _text(res)


def test_a_card_without_a_root_is_no_pass():
    k = KEYS[0]
    assert not _four(_parts(**{f"card_{k}": _part(f"card_{k}", "four_card", {"card": None})}))[
        "pass"]


@pytest.mark.parametrize("arm", ["seq", "pipe"])
def test_an_arm_that_was_rejected_is_no_pass(arm):
    k = KEYS[3]
    res = _four(_parts(**{f"{arm}_{k}": _arm(k, arm, accepted=False)}))
    assert not res["pass"] and not res["models"][k]["accepted"][arm]
    assert f"the {arm} arm was not accepted" in _text(res)


def test_an_arm_with_fewer_than_three_roots_or_the_wrong_mode_is_no_pass():
    k = KEYS[0]
    assert not _four(_parts(**{f"seq_{k}": _arm(k, "seq", roots=ROOTS[:2])}))["pass"]
    assert not _four(_parts(**{f"pipe_{k}": _arm(k, "pipe", mode="sequential")}))["pass"]


def test_an_environment_record_with_three_gpus_is_no_pass():
    k = KEYS[0]
    res = _four(_parts(**{f"seq_{k}": _arm(k, "seq", gpus=_gpus(3))}))
    assert not res["pass"] and "machine.gpus has 3 entries" in _text(res)


def test_an_environment_record_without_gpus_or_roles_is_no_pass():
    k = KEYS[0]
    arm = _arm(k, "seq")
    arm["extras"]["gpus"] = None
    assert "the record has no machine.gpus" in _text(_four(_parts(**{f"seq_{k}": arm})))
    arm = _arm(k, "pipe")
    arm["extras"]["gpu_roles"] = None
    assert "the record has no gpu_roles" in _text(_four(_parts(**{f"pipe_{k}": arm})))


def test_a_wrong_role_is_no_pass():
    k = KEYS[0]
    res = _four(_parts(**{f"pipe_{k}": _arm(k, "pipe", roles=_roles({0: ["prover", "verifier"]}))}))
    assert not res["pass"] and "cuda:0 has roles ['prover', 'verifier']" in _text(res)
    res = _four(_parts(**{f"seq_{k}": _arm(k, "seq", roles=_roles({2: ["prover"]}))}))
    assert not res["pass"] and "cuda:2 has roles ['prover']" in _text(res)


def test_a_uuid_twice_is_no_pass():
    k = KEYS[0]
    twice = [{"index": i, "uuid": "GPU-0"} for i in range(4)]
    res = _four(_parts(**{f"seq_{k}": _arm(k, "seq", gpus=twice)}))
    assert not res["pass"] and "a UUID twice" in _text(res)


def test_a_missing_part_is_no_pass():
    parts = [p for p in _parts() if p["name"] != f"pipe_{KEYS[3]}"]
    res = _four(parts)
    assert not res["pass"] and res["part_problems"] == {f"pipe_{KEYS[3]}": "not_started"}
    assert f"pipe_{KEYS[3]} did not run" in _text(res)


def test_a_part_that_failed_or_a_peer_check_with_wrong_bytes_is_no_pass():
    res = _four(_parts(probe=dict(_probe(), status="failed")))
    assert not res["pass"] and "probe ended failed" in _text(res)
    bad = _part("peer_check", "peer_check", {"peer_check": {"all_ok": False, "wrong": {
        "0->1": ["to_4KB"]}}})
    assert not _four(_parts(peer_check=bad))["peer_bytes_ok"]


def test_a_done_part_from_an_earlier_run_counts():
    assert _four(_parts(models=_part("models", "models", status="skipped_done")))["pass"]


def test_a_junit_with_a_failure_an_error_or_a_skipped_fs34_case_is_no_pass():
    assert not _four(_parts(four_tests=_tests(failures=1)))["pass"]
    assert not _four(_parts(four_tests=_tests(errors=1)))["pass"]
    first = sorted(EXPECTED34)[0]
    for outcome in ("failed", "skipped", "error"):
        t = _tests()
        t["extras"]["cases"][first] = outcome
        res = _four(_parts(four_tests=t))
        assert not res["pass"] and f"FS34 case {first}: {outcome}" in _text(res)
    t = _tests()
    del t["extras"]["cases"][first]
    assert "not in the junit report" in _text(_four(_parts(four_tests=t)))


def test_a_skipped_case_of_another_file_does_not_fail_the_tests():
    t = _tests(skipped=3)
    t["extras"]["files"][fs.FOUR_TEST_FILES[0]] = {"cases": 2, "skipped": 2}
    assert _four(_parts(four_tests=t))["pass"]


def test_a_missing_file_or_no_junit_or_no_tests_is_no_pass_and_says_which():
    f = fs.FOUR_TEST_FILES[2]
    res = _four(_parts(four_tests=_tests(missing_files=[f])))
    assert not res["pass"] and f"{f} is missing from the checkout" in _text(res)
    t = _tests()
    t["extras"]["files"][fs.FOUR_TEST_FILES[1]] = {"cases": 0, "skipped": 0}
    assert f"{fs.FOUR_TEST_FILES[1]} left no case in the junit report" in _text(_four(
        _parts(four_tests=t)))
    assert not _four(_parts(four_tests=_part("four_tests", "four_tests", {"junit": None},
                                             status="failed")))["pass"]
    assert not _four(_parts(four_tests=_tests(tests=0)))["pass"]


def test_an_unreadable_expected_list_is_no_pass(monkeypatch):
    monkeypatch.setattr(fs, "device_tests_expected", lambda f: ([], "collection failed"))
    res = fs.four_h100(_parts())
    assert not res["pass"] and "could not be collected" in _text(res)


# ---- the log lines ----------------------------------------------------------------------------


def test_the_log_lines_all_good():
    lines = fs._four_lines(_four())
    assert lines[0].startswith("FS35 ok: every part ran") and "FS35 FAIL" not in "\n".join(lines)
    n = 3 + 4 * len(KEYS)
    assert all(x.startswith("FS35 ok:") for x in lines[:n])
    per = lines[n:-1]
    assert len(per) == 2 * len(KEYS)
    assert per[0] == (f"FS35 {KEYS[0]} seq: steady step 2.5 s (step_wall, mean of steps 2-3); "
                      "GPU peaks 0: 39.1 GiB, 1: 19.5 GiB, 2: 20.5 GiB, 3: 21.5 GiB")
    assert per[1].startswith(f"FS35 {KEYS[0]} pipe: steady step 2.5 s")
    assert lines[-1].startswith("FS35 peer rates (probe, 12 pairs): slowest cuda:0->cuda:1 ")
    assert "fastest cuda:3->cuda:2" in lines[-1]


def test_a_failed_item_prints_its_own_fail_line_and_the_rest_still_print():
    k = KEYS[0]
    lines = fs._four_lines(_four(_parts(**{f"pipe_{k}": _arm(k, "pipe", accepted=False)})))
    fails = [x for x in lines if x.startswith("FS35 FAIL:")]
    assert len(fails) == 1 and "the pipe arm was not accepted" in fails[0]
    assert any(x.startswith("FS35 ok:") for x in lines) and "FS35 peer rates" in lines[-1]


def test_unknown_times_and_peaks_and_missing_peer_rates_print_as_such():
    k = KEYS[0]
    arm = _arm(k, "seq", wall=False)
    arm["extras"]["step_times"], arm["extras"]["gpu_peak_mib"] = [], None
    parts = _parts(**{f"seq_{k}": arm, "probe": _part("probe", "four_probe", {})})
    lines = fs._four_lines(_four(parts))
    assert f"FS35 {k} seq: steady step unknown; GPU peaks unknown" in lines
    assert lines[-1] == "FS35 peer rates: none measured" or "peer check" in lines[-1]
    assert fs._four_lines(None) == []


# ---- the pieces that read a part's output -----------------------------------------------------


def _write_records(path, gpus=None, roles=None, steps=3, pipelined=True):
    rows = [{"record": "environment", "machine": {"gpus": gpus or _gpus()},
             "gpu_roles": roles or _roles()}]
    for t in range(1, steps + 1):
        rows.append({"record": "time", "level": "total", "component": "step", "step": t,
                     "time_s": 5.0, "gpu_held_s": 10.0 - t})
        if pipelined:
            rows.append({"record": "time", "level": "total", "component": "step_wall",
                         "step": t, "time_s": 4.0 - t})
        for comp, dev in (("prover", ["cuda:0"]), ("verifier", ["cuda:1", "cuda:2", "cuda:3"])):
            rows.append({"record": "memory", "level": "total", "component": comp, "step": t,
                         "devices": dev, "device_peak_bytes": [t * 10] * len(dev),
                         "device_peak_reserved_bytes": [t * 20] * len(dev)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{cut short")


def _log(accepted=True, mode="pipelined"):
    out = [f"loop mode: {mode}"]
    for t, r in enumerate(ROOTS, 1):
        out += [f"step {t}: prover 5.00 s (prove_step 4.00 s, commit 1.00 s); verifier 7.50 s "
                "(0 0.50, 1 1.20); hand-off 0.25 s", f"step {t:>2} root {r}"]
    return "\n".join([*out, "   actual:   " + ("accept" if accepted else "reject"),
                      "   PASS" if accepted else "   FAIL"])


def test_run_extras_reads_the_log_and_the_records(tmp_path):
    log, rec = tmp_path / "log", tmp_path / "d" / "llama_step" / "records.jsonl"
    log.write_text(_log())
    _write_records(rec)
    e = fs._four_run_extras(log, rec)
    assert e["accepted"] and e["step_roots"] == ROOTS and e["loop_mode"] == "pipelined"
    assert [r["total_s"] for r in e["step_times"]] == [12.75] * 3
    assert e["gpus"] == _gpus() and e["gpu_roles"] == _roles()
    assert e["gpu_held_s"] == [9.0, 8.0, 7.0] and e["step_wall_s"] == [3.0, 2.0, 1.0]
    assert e["device_peaks"]["cuda:0"] == {"peak_bytes": 30, "peak_reserved_bytes": 60}
    assert set(e["device_peaks"]) == {f"cuda:{i}" for i in range(4)}
    assert not _four_gpu(e)


def _four_gpu(e):
    return fs._four_gpu_problems(e)


def test_run_extras_with_a_rejection_or_missing_files(tmp_path):
    log = tmp_path / "log"
    log.write_text(_log(accepted=False))
    assert not fs._four_run_extras(log, tmp_path / "none")["accepted"]
    e = fs._four_run_extras(tmp_path / "nolog", tmp_path / "none")
    assert not e["accepted"] and e["step_roots"] == [] and e["gpus"] is None
    assert e["record"] is None and e["loop_mode"] is None and e["gpu_held_s"] == []


def test_tests_extras_counts_cases_per_file_and_names_a_missing_file(tmp_path, monkeypatch):
    mod = fs.FS34_TESTS_FILE.removesuffix(".py").replace("/", ".")
    probe = fs.FOUR_TEST_FILES[0].removesuffix(".py").replace("/", ".")
    junit = tmp_path / "j.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        f'<testcase classname="{mod}" name="a"/>'
        f'<testcase classname="{probe}" name="b"><skipped/></testcase>'
        f'<testcase classname="{probe}" name="c"/></testsuite></testsuites>')
    monkeypatch.setattr(fs, "_REPO_ROOT", tmp_path)
    (tmp_path / fs.FS34_TESTS_FILE).parent.mkdir(parents=True)
    (tmp_path / fs.FS34_TESTS_FILE).write_text("")
    e = fs._four_tests_extras(junit)
    assert e["files"][fs.FS34_TESTS_FILE] == {"cases": 1, "skipped": 0}
    assert e["files"][fs.FOUR_TEST_FILES[0]] == {"cases": 2, "skipped": 1}
    assert e["files"][fs.FOUR_TEST_FILES[1]] == {"cases": 0, "skipped": 0}
    assert e["missing_files"] == list(fs.FOUR_TEST_FILES)
    assert fs._four_tests_extras(tmp_path / "none")["junit"] is None


def test_probe_extras_reads_the_version_2_report(tmp_path):
    rep = tmp_path / "probe.json"
    rep.write_text(json.dumps({
        "version": 2, "devices": ["cuda:0", "cuda:1"], "rates": {}, "costs": {}, "errors": {},
        "device": "cuda:0", "threads": 4, "wall_s": 1.0,
        "per_device": {n: {"rates": {"gemm_bf16_flops": 7e14, "hbm_copy_bytes": 2e12,
                                     "hash_kernel": {"per_byte_s": 1e-11}}}
                       for n in ("cuda:0", "cuda:1")},
        "peer": {"cuda:0->cuda:1": 3e11, "cuda:1->cuda:0": None}, "peer_errors": {
            "cuda:1->cuda:0": "RuntimeError: x"}}))
    e = fs._four_probe_extras(rep)
    assert e["version"] == 2 and e["per_device"]["cuda:1"] == {
        "gemm_bf16_flops": 7e14, "hbm_copy_bytes": 2e12, "hash_kernel_per_byte_s": 1e-11}
    assert e["peer"]["cuda:0->cuda:1"] == 3e11 and "cuda:1->cuda:0" in e["peer_errors"]
    assert fs._four_probe_extras(tmp_path / "none") == fs._probe_extras(tmp_path / "none")


def test_the_peer_check_extras_keep_each_pairs_rate(tmp_path):
    rep = tmp_path / "peer.json"
    rep.write_text(json.dumps({"all_ok": True, "seconds": 1.0, "pairs": {
        "0->1": {"ok": True, "cases": {"host_staged": True}, "gbps": 7.0}}}))
    assert fs._peer_check_extras(rep)["peer_check"]["gbps"] == {"0->1": 7.0}


# ---- the whole session on stand-in parts ------------------------------------------------------


COPY = ("import os,shutil,sys\nos.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)\n"
        "shutil.copy(sys.argv[1], sys.argv[2])\n")
PEER_WRITER = ("import json,sys,os\nos.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)\n"
               "json.dump({'all_ok': True, 'seconds': 1.0, 'pairs': {'0->1': {'ok': True, "
               "'cases': {'host_staged': True}, 'gbps': 7.0}}}, open(sys.argv[1], 'w'))")


def _fake_plan(tmp_path, monkeypatch, bad_arm=None):
    """The real plan with each part's command replaced by one that writes what the real part
    does, from the fixtures above."""
    files = ("tests/verification/runs/test_llama_step.py",
             "tests/verification/runs/test_four_h100_session.py")
    monkeypatch.setattr(fs, "FOUR_TEST_FILES", files)
    src = tmp_path / "src"
    src.mkdir()
    cases = "".join(f'<testcase classname="{n.split("::")[0]}" name="{n.split("::")[1]}"/>'
                    for n in EXPECTED34)
    for f in files:
        cases += f'<testcase classname="{f.removesuffix(".py").replace("/", ".")}" name="t"/>'
    (src / "junit.xml").write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    (src / "probe.json").write_text(json.dumps({
        "version": 2, "devices": ["cuda:0", "cuda:1"], "per_device": {}, "peer": {
            "cuda:0->cuda:1": 3e11}, "peer_errors": {}, "rates": {}, "costs": {}, "errors": {},
        "device": "cuda:0", "threads": 4, "wall_s": 1.0}))
    (src / "card.json").write_text(json.dumps({
        "card": {"name": "H100", "uuid": "u", "pci_bus_id": "p"},
        "plain": {"w_next_root": "w", "loss": 1.0},
        "proved": {"root": ROOTS[0], "w_next_root": "w", "loss": 1.0}, "n_leaves": 3,
        "wall_s": {}}))
    for arm in ("seq", "pipe"):
        (src / f"{arm}.log").write_text(_log(accepted=bad_arm != arm,
                                             mode={"seq": "sequential"}.get(arm, "pipelined")))
        _write_records(src / f"{arm}.jsonl", pipelined=arm == "pipe")
    plan = fs.with_native_build(fs.four_h100_plan(tmp_path / "up", tmp_path / "out"))
    out = []
    for p in plan:
        if p.kind == "peer_check":
            argv = ["{python}", "-c", PEER_WRITER, p.info["report"]]
        elif p.kind == "four_tests":
            argv = ["{python}", "-c", COPY, str(src / "junit.xml"), p.info["junit"]]
        elif p.kind == "four_probe":
            argv = ["{python}", "-c", COPY, str(src / "probe.json"), p.info["report"]]
        elif p.kind == "four_card":
            argv = ["{python}", "-c", COPY, str(src / "card.json"), p.info["report"]]
        elif p.kind == "four_run":
            a = p.info["arm"]
            argv = ["{python}", "-c", COPY + f"print(open({str(src / (a + '.log'))!r}).read())",
                    str(src / f"{a}.jsonl"), p.info["records"]]
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
                       session="four-h100")
    return s, lines


def test_the_whole_session_on_stand_in_parts_passes_and_ends_with_the_fs35_lines(
        tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path, monkeypatch))
    four = s["four_h100"]
    assert four["pass"], [t for ok, t in four["items"] if not ok]
    assert not s["failed"] and "four_h100_error" not in four
    tail = [x for x in lines if x.startswith("FS35")]
    assert tail[0].startswith("FS35 ok:") and tail[-1].startswith("FS35 peer rates")
    assert sum("steady step" in x for x in tail) == 2 * len(KEYS)


def test_a_rejected_arm_leaves_the_exit_code_alone(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _fake_plan(tmp_path, monkeypatch, bad_arm="pipe"))
    assert not s["four_h100"]["pass"]
    assert any(x.startswith("FS35 FAIL:") and "pipe arm was not accepted" in x for x in lines)
    assert not s["failed"]  # no part failed: the job's exit code is the parts'


def test_an_error_in_the_summary_is_recorded_and_the_session_runs_on(tmp_path, monkeypatch):
    plan = _fake_plan(tmp_path, monkeypatch)

    def boom(parts):
        raise KeyError("x")

    monkeypatch.setattr(fs, "four_h100", boom)
    s, lines = _session(tmp_path, monkeypatch, plan)
    assert s["four_h100"] == {"four_h100_error": "KeyError: 'x'"}
    assert any(x.startswith("FS35 FAIL: the comparison raised (KeyError") for x in lines)
    assert s["finished"] and [p["name"] for p in s["parts"]] == [p.name for p in plan]
    assert copy.deepcopy(s["four_h100"])
