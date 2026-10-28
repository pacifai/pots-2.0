"""FS42 (d)'s profiling session (``four-h100-profile``): its plan, its deadline, the acceptance it
writes into ``summary.json["four_h100_profile"]``, the job-log lines it ends with, and the part
extras that read the profile's inputs from a run directory."""

import copy
import gzip
import json
import subprocess
import time
from pathlib import Path

import pytest

from tests.verification.runs import test_four_h100_session as base
from verification.runs import fs11_session as fs

JOB = Path(fs.__file__).with_name("fs11_job.sh")
README = Path(fs.__file__).parents[1] / "README.md"
PKEYS = [fs.four_model_key(r) for r, _ in fs.FOUR_PROFILE_PINS]
ROOTS4 = ["%064x" % i for i in (1, 2, 3, 4)]
EXPECTED34 = base.EXPECTED34
UUIDS = base.UUIDS
PLAIN_STEPS = [10.0, 8.0, 6.0]  # the plain run's per-step totals: the steady step is 7.0 s
ARM_INTERVAL = {"seq": [30.0, 21.0, 19.0, 25.0], "pipe": [20.0, 12.0, 10.0, 14.0]}
ARM_VERIFIER = {"seq": [18.0, 14.0, 12.0, 16.0], "pipe": [18.0, 14.0, 12.0, 16.0]}


def _profile(arm, **over):
    pr = {"markers_file": True, "markers_steps": [1, 2, 3, 4],
          "traced_steps": [4] if arm == "seq" else [], "interval_steps": [1, 2, 3, 4],
          "count_sections": sorted(fs.PROFILE_WORK_SECTIONS), "storage_rows": 2,
          "layout": True,
          "trace_files": ["step_00004.json.gz"] if arm == "seq" else [], "trace_error": None}
    return {**pr, **over}


def _arm(k, arm, roots=None, profile=None, samples=None, **over):
    times = [{"prover_s": 5.0, "verifier_s": v, "handoff_s": 0.5, "total_s": 6.0 + v}
             for v in ARM_VERIFIER[arm]]
    ex = {"accepted": True, "actual": "accept",
          "step_roots": ROOTS4 if roots is None else roots,
          "loop_mode": {"seq": "sequential", "pipe": "pipelined"}[arm], "step_times": times,
          "verifier_s": ARM_VERIFIER[arm], "record": "r", "gpus": base._gpus(),
          "gpu_roles": base._roles(), "gpu_held_upper_s": [9.0] * 4,
          "accumulated_gpu_s": [30.0, 28.0, 26.0, 29.0],
          "step_wall_s": [31.0, 22.0, 20.0, 26.0] if arm == "pipe" else [],
          "step_interval_s": ARM_INTERVAL[arm],
          "device_peaks": {"cuda:0": {"peak_bytes": 5, "peak_reserved_bytes": 6}},
          "gpu_peak_mib": base.CARD_MIB, "profile": profile or _profile(arm),
          "samples": samples or {"version": 2, "gpus": 4}}
    return base._part(f"{arm}_{k}", "profile_run", {**ex, **over}, arm=arm)


def _plain(k, **over):
    ex = {"record": "p", "completed": True, "step_s": PLAIN_STEPS, "p0_steps": [1, 2, 3],
          "step_totals_s": PLAIN_STEPS, "steady_s": 7.0, "weights_bytes": 100,
          "samples": {"version": 2, "gpus": 4}}
    return base._part(f"plain_{k}", "profile_plain", {**ex, **over})


def _card(k, **over):
    return base._part(f"card_{k}", "four_card",
                      {"root": ROOTS4[0], "samples": {"version": 2, "gpus": 4}, **over})


def _with_samples(part):
    ex = dict(part["extras"] or {})
    ex.setdefault("samples", {"version": 2, "gpus": 4})
    return {**part, "extras": ex}


def _parts(**over):
    head = [base._part("native_build", "other"),
            _with_samples(base._parts()[1]), _with_samples(base._part("models", "models", {})),
            _with_samples(base._tests()), _with_samples(base._probe())]
    parts = list(head)
    for k in PKEYS:
        parts += [_plain(k), _card(k), _arm(k, "seq"), _arm(k, "pipe")]
    return [over.get(p["name"], p) for p in parts]


def _prof(parts=None):
    return fs.four_h100_profile(_parts() if parts is None else parts, EXPECTED34)


def _fails(res):
    return [t for ok, t in res["items"] if not ok]


def _one_fail(res, *words):
    """Exactly one item fails and its line holds every word of ``words``."""
    bad = _fails(res)
    assert not res["pass"] and len(bad) == 1, bad
    for w in words:
        assert w in bad[0], (w, bad[0])


K = PKEYS[1]  # Llama-3.2-1B, the model the failing cases are planted in


# ---- the plan ---------------------------------------------------------------------------------


def test_the_session_is_registered_and_needs_four_cards():
    assert fs.SESSIONS["four-h100-profile"] == ("four-h100-profile", fs.four_h100_profile_plan)
    assert "four-h100-profile" in fs.MULTI_GPU_SESSIONS and "four-h100" in fs.MULTI_GPU_SESSIONS
    assert fs.FOUR_PROFILE_ORDER == ("Qwen2.5-0.5B", "Llama-3.2-1B", "Falcon3-1B")
    assert PKEYS == list(fs.FOUR_PROFILE_ORDER)


def test_the_plan_has_the_parts_in_order_and_no_qwen_1_5b(tmp_path):
    plan = fs.with_native_build(fs.four_h100_profile_plan(tmp_path / "up", tmp_path / "out"))
    names = [p.name for p in plan]
    assert names[:5] == ["native_build", "peer_check", "models", "four_tests", "probe"]
    assert names[5:] == [f"{a}_{k}" for k in PKEYS for a in ("plain", "card", "seq", "pipe")]
    assert names == list(fs.four_profile_parts())
    assert not any("Qwen2.5-1.5B" in n for n in names)
    assert all(p.info.get("gpu_peak") for p in plan[1:])
    assert all(p.info.get("samples_check") for p in plan[1:])


def test_each_part_has_its_environment_and_command(tmp_path):
    plan = {p.name: p for p in fs.four_h100_profile_plan(tmp_path / "up", tmp_path / "out")}
    for n, p in plan.items():
        assert (p.env.get("VERIF_TRACE_STEP") == "4") == n.startswith("seq_"), n
        assert (p.env.get("VERIF_TRACE_C5") == "all") == n.startswith("seq_"), n
        assert bool(p.env.get("VERIF_PIPELINED")) == n.startswith("pipe_"), n
        assert "VERIF_TRACE_STEP" not in p.env or n.startswith(("seq_", "pipe_")), n
    dirs = []
    for k in PKEYS:
        plain, card, seq, pipe = (plan[f"{a}_{k}"] for a in ("plain", "card", "seq", "pipe"))
        assert plain.kind == "profile_plain" and "verification.runs.plain_baseline" in plain.argv
        assert plain.argv[plain.argv.index("--steps") + 1] == "3" and "--metrics" in plain.argv
        assert fs.FOUR_PROFILE_PLAIN_STEPS == 3 and fs.FOUR_PROFILE_STEPS == 4
        assert plain.env["VERIF_DEVICE"] == "cuda:0"
        assert plain.env["VERIF_VERIFIER_DEVICES"] is None and plain.env["VERIF_PIPELINED"] is None
        assert plain.info["records"].endswith("plain_baseline/records.jsonl")
        assert card.kind == "four_card" and card.env["CUDA_VISIBLE_DEVICES"] == "0"
        for p in (seq, pipe):
            assert p.kind == "profile_run" and "verification.runs.llama_step" in p.argv
            assert p.argv[p.argv.index("--steps") + 1] == "4"
            assert "--metrics" in p.argv and "--no-count" in p.argv
            assert p.env["VERIF_DEVICE"] == "cuda:0"
            assert p.env["VERIF_VERIFIER_DEVICES"] == fs.FOUR_VERIFIERS
            assert p.info["records"].endswith("llama_step/records.jsonl")
        assert seq.env["VERIF_PIPELINED"] is None and pipe.env["VERIF_TRACE_STEP"] is None
        assert pipe.env["VERIF_PIPELINED"] == "1"
        dirs += [p.env["VERIF_OUTPUT_DIR"] for p in (plain, card, seq, pipe)]
        assert plain.timeout_s == fs.FOUR_PROFILE_TIMEOUT_S["plain"]
    assert len(set(dirs)) == len(dirs)
    assert plan["probe"].env.get("VERIF_TRACE_STEP") is None


def test_the_estimates_are_the_briefs_and_the_limits_about_twice_them():
    e = fs.FOUR_PROFILE_ESTIMATE_MIN
    assert e == {"native_build": 1, "peer_check": 1, "models": 6, "four_tests": 6, "probe": 7,
                 "plain": 3, "card": 2, "seq": 6, "pipe": 4}
    assert fs.four_profile_estimated_min() == 21 + 3 * 15 + 10 == 76
    for part, limit in fs.FOUR_PROFILE_TIMEOUT_S.items():
        assert limit == 2 * 60 * e[part], part


def test_four_h100s_plan_is_unchanged(tmp_path):
    (tmp_path / "up" / "repo_base").mkdir(parents=True)
    plan = fs.four_h100_plan(tmp_path / "up", tmp_path / "out")
    assert [p.name for p in plan] == [n for n in fs.four_h100_parts()
                                      if n != fs.NATIVE_BUILD_NAME]
    assert not any(n.startswith("plain_") for n in (p.name for p in plan))
    assert all(p.kind != "profile_run" and "VERIF_TRACE_STEP" not in p.env for p in plan)
    arms = [p for p in plan if p.name.startswith(("seq_", "pipe_"))]
    assert arms and all(p.kind == "four_run" and p.argv[p.argv.index("--steps") + 1] == "3"
                        for p in arms)
    assert not any("samples_check" in p.info for p in plan)
    assert fs.four_h100([base._part("peer_check", "peer_check")]) is None
    assert fs.four_h100(_parts()) is None  # the profile's arms are not four_run parts


# ---- the refusal and the deadline -------------------------------------------------------------


def test_the_refusal_is_the_same_as_four_h100s():
    from verification.runs import fs11_session
    text = Path(fs11_session.__file__).read_text()
    assert '"four-h100", "four-h100-profile"' in text  # one `four` flag drives the refusal
    why = fs.refusal(base._machine("NVIDIA H100 80GB HBM3", 3), True, fs.FOUR_CARDS,
                     need="the four-H100 session")
    assert "sees 3 CUDA devices" in why


def test_the_deadline_is_twice_the_estimate_and_the_job_holds_it():
    assert fs.four_profile_default_deadline_s() == 11700  # (76 + 1 + 20) * 2 * 60 = 11640, up 100
    text = JOB.read_text()
    want = f"[ \"$SESSION\" = four-h100-profile ] && DEFAULT_DEADLINE_S=" \
           f"{fs.four_profile_default_deadline_s()}"
    assert want in text
    assert fs.four_default_deadline_s() == 14700
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


def test_the_job_script_accepts_the_session_name():
    text = JOB.read_text()
    case = next(ln for ln in text.splitlines() if "four-h100" in ln and ";;" not in ln
                and "four-h100|" in ln)
    assert case.strip().startswith("fs11|") or "four-h100|" in case
    assert "four-h100-profile" in text[text.index("four-h100|"):text.index("four-h100|") + 400]


# ---- the acceptance ---------------------------------------------------------------------------


def test_a_complete_fake_output_passes():
    res = _prof()
    assert res["pass"], _fails(res)
    assert not res["missing_numbers"] and not res["samples_problems"]
    assert list(res["models"]) == PKEYS
    row = res["models"][K]
    assert row["plain"]["steady_s"] == pytest.approx(7.0)
    assert row["plain"]["steady_row"].startswith("time / total / step")
    seq, pipe = row["arms"]["seq"], row["arms"]["pipe"]
    assert seq["steady_s"] == pytest.approx(20.0) and seq["steady_source"] == "step_interval"
    assert seq["verifier_steady_s"] == pytest.approx(13.0)
    assert seq["verify_over_train"] == pytest.approx(13.0 / 7.0)
    assert seq["wall_over_train"] == pytest.approx(20.0 / 7.0)
    assert pipe["steady_s"] == pytest.approx(11.0)
    assert pipe["wall_over_train"] == pytest.approx(11.0 / 7.0)
    assert pipe["latency_steady_s"] == pytest.approx(21.0) and seq["latency_steady_s"] is None
    assert seq["accumulated_steady_s"] == pytest.approx(27.0)
    assert seq["gpu_peak_mib"] == base.CARD_MIB
    # the 4-step arms keep steps 2-3: the sequential arm's step 4 is traced, the pipelined arm's
    # is its drain
    assert seq["steady_steps"] == pipe["steady_steps"] == "steps 2-3"
    assert res["probe"]["devices"] == [f"cuda:{i}" for i in range(4)]
    assert json.dumps(res)


def test_no_profile_parts_no_result():
    assert fs.four_h100_profile([base._part("peer_check", "peer_check")]) is None
    assert fs.four_h100_profile(base._parts()) is None  # FS35's parts are not the profile's


def test_a_missing_trace_fails_with_its_own_line():
    for why, pr in (("no trace file", _profile("seq", trace_files=[])),
                    ("no step", _profile("seq", traced_steps=[])),
                    ("an error", _profile("seq", trace_error="profiler failed"))):
        res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", profile=pr)}))
        _one_fail(res, K, "seq", "trace")
        if why == "an error":
            assert "profiler failed" in _fails(res)[0]


def test_a_traced_step_in_the_pipelined_arm_fails():
    pr = _profile("pipe", traced_steps=[4], trace_files=["step_00004.json.gz"])
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", profile=pr)}))
    _one_fail(res, K, "pipe", "traced step")


def test_a_second_traced_step_in_the_sequential_arm_fails():
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", profile=_profile("seq", traced_steps=[3, 4]))}))
    _one_fail(res, K, "seq", "[3, 4]")


def test_a_missing_step_interval_row_fails():
    pr = _profile("pipe", interval_steps=[1, 2, 4])
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", profile=pr)}))
    _one_fail(res, K, "pipe", "step_interval", "step 3")


def test_missing_work_rows_fail():
    pr = _profile("seq", count_sections=["train.forward", "2"], storage_rows=0)
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", profile=pr)}))
    _one_fail(res, K, "seq", "train.backward", "P3.commit", "storage")


def test_a_missing_layout_file_fails():
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", profile=_profile("pipe", layout=False))}))
    _one_fail(res, K, "pipe", "c5_layout.json")


def test_missing_markers_fail():
    pr = _profile("seq", markers_file=False, markers_steps=[])
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", profile=pr)}))
    _one_fail(res, K, "seq", "markers.jsonl")
    pr = _profile("seq", markers_steps=[1, 2, 3])
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", profile=pr)}))
    _one_fail(res, K, "seq", "markers.jsonl", "step 4")


def test_a_missing_plain_run_fails():
    for ex in ({"record": None}, {"weights_bytes": 0}, {"p0_steps": [1, 2]},
               {"completed": False}):
        part = _plain(K, **ex)
        res = _prof(_parts(**{f"plain_{K}": part}))
        _one_fail(res, K, "plain")
    gone = base._part(f"plain_{K}", "profile_plain", None, status="failed")
    res = _prof(_parts(**{f"plain_{K}": gone}))
    assert not res["pass"] and any(f"plain_{K} ended failed" in t for t in _fails(res))


def test_roots_that_differ_between_arms_fail():
    bad = [ROOTS4[0], ROOTS4[1], ROOTS4[2], "%064x" % 9]
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", roots=bad)}))
    _one_fail(res, K, "step roots differ")


def test_four_roots_per_arm_are_needed_not_three():
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", roots=ROOTS4[:3])}))
    assert not res["pass"] and any("seq arm was not accepted at all 4 steps" in t
                                   for t in _fails(res))


def test_a_step_one_root_that_is_not_the_cards_fails():
    res = _prof(_parts(**{f"card_{K}": _card(K, root="%064x" % 7)}))
    _one_fail(res, K, "card_")


def test_the_wrong_loop_mode_fails():
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", loop_mode="sequential")}))
    assert not res["pass"] and any("pipe arm was not accepted" in t for t in _fails(res))


def test_fewer_than_four_gpus_in_the_environment_record_fail():
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", gpus=base._gpus(3))}))
    assert not res["pass"] and any(K in t and "seq arm" in t for t in _fails(res))


def test_samples_with_fewer_than_four_gpus_fail():
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", samples={"version": 2, "gpus": 3})}))
    _one_fail(res, f"seq_{K}", "samples.jsonl", "3 GPUs")
    assert list(res["samples_problems"]) == [f"seq_{K}"]


def test_samples_at_version_1_or_missing_fail_for_their_part_only():
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", samples={"version": 1, "gpus": 1})}))
    _one_fail(res, f"pipe_{K}", "version 1")
    gone = _plain(K, samples=None)
    gone["extras"]["samples"] = None
    res = _prof(_parts(**{f"plain_{K}": gone}))
    _one_fail(res, f"plain_{K}", "no samples.jsonl")


def test_a_bad_probe_report_fails():
    probe = base._probe()
    probe["extras"] = {**probe["extras"], "samples": {"version": 2, "gpus": 4}, "version": 1}
    _one_fail(_prof(_parts(probe=probe)), "probe.json is version 1")
    probe["extras"] = {**probe["extras"], "version": 2,
                       "per_device": dict(list(probe["extras"]["per_device"].items())[:3])}
    _one_fail(_prof(_parts(probe=probe)), "3 per_device entries")


def test_a_part_that_did_not_run_fails_and_a_skipped_done_part_counts():
    res = _prof([p for p in _parts() if p["name"] != f"seq_{K}"])
    assert not res["pass"] and any(f"seq_{K} did not run" in t for t in _fails(res))
    parts = _parts()
    done = [{**p, "status": "skipped_done"} if p["name"] == f"plain_{K}" else p for p in parts]
    assert _prof(done)["pass"]


def test_a_bad_peer_check_or_test_report_fails():
    peer = base._parts()[1]
    peer = _with_samples({**peer, "extras": {"peer_check": {"all_ok": False,
                                                            "wrong": {"0->1": ["pattern"]}, "gbps": {}}}})
    assert any("peer copies" in t for t in _fails(_prof(_parts(peer_check=peer))))
    res = _prof(_parts(four_tests=_with_samples(base._tests(failures=1))))
    assert any(t.startswith("the tests:") for t in _fails(res))


def test_a_missing_step_time_is_a_missing_number():
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", step_interval_s=[], step_wall_s=[],
                                            step_times=[])}))
    assert any(K in m and "no steady step time" in m for m in res["missing_numbers"])


def test_the_step_wall_fallback_leaves_the_wall_ratio_unset():
    res = _prof(_parts(**{f"pipe_{K}": _arm(K, "pipe", step_interval_s=[])}))
    arm = res["models"][K]["arms"]["pipe"]
    assert arm["steady_source"] == "step_wall" and arm["wall_over_train"] is None
    assert arm["verify_over_train"] == pytest.approx(13.0 / 7.0)


def test_the_verifier_ratio_is_given_in_wall_time_and_in_accumulated_gpu_time():
    arm = _prof()["models"][K]["arms"]["seq"]
    # verifier side 13 s a step (steps 2-3), the plain step 7 s on its one GPU
    assert arm["verify_over_train"] == pytest.approx(13.0 / 7.0)  # wall time
    n_verifier = sum("verifier" in g["roles"] for g in base._roles())
    assert n_verifier == 3
    assert arm["verify_over_train_accumulated"] == pytest.approx(13.0 * 3 / 7.0)
    # the verifier GPUs come from the arm's own gpu_roles
    roles = base._roles({1: ["verifier"], 2: ["prover"], 3: ["verifier"]})
    res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", gpu_roles=roles)}))
    arm = res["models"][K]["arms"]["seq"]
    assert arm["verify_over_train"] == pytest.approx(13.0 / 7.0)
    assert arm["verify_over_train_accumulated"] == pytest.approx(13.0 * 2 / 7.0)


def test_the_accumulated_ratio_is_unknown_without_roles_a_verifier_time_or_a_plain_step():
    for over in ({"gpu_roles": None}, {"verifier_s": []}):
        res = _prof(_parts(**{f"seq_{K}": _arm(K, "seq", **over)}))
        assert res["models"][K]["arms"]["seq"]["verify_over_train_accumulated"] is None, over
    res = _prof(_parts(**{f"plain_{K}": _plain(K, steady_s=None)}))
    assert res["models"][K]["arms"]["seq"]["verify_over_train_accumulated"] is None


def test_a_refused_profile_run_names_its_own_session(tmp_path, monkeypatch):
    monkeypatch.delenv("FS11_ALLOW_ANY_GPU", raising=False)
    if len(fs.environment(tmp_path / "e")["gpus"]) >= fs.FOUR_CARDS:
        pytest.skip("this machine has four CUDA devices")
    for session in ("four-h100", "four-h100-profile"):
        out = tmp_path / session
        rc = fs.main(["run", "--upload", str(tmp_path / "up"), "--out", str(out), "--deadline",
                      str(time.time() + 30), "--session", session])
        s = json.loads((out / "summary.json").read_text())
        assert rc == fs.EXIT_REFUSED
        assert f"the {session} session needs 4" in s["refused"], s["refused"]


def test_nothing_in_a_result_is_shared_with_the_parts():
    parts = _parts()
    before = copy.deepcopy(parts)
    _prof(parts)
    assert parts == before


# ---- the log lines ----------------------------------------------------------------------------


def test_the_log_lines_hold_the_numbers_and_ratios():
    lines = fs._profile_lines(_prof())
    assert lines[0].startswith("FS42 ok:") and not any(x.startswith("FS42 FAIL") for x in lines)
    per = [x for x in lines if x.startswith(f"FS42 {K} ")]
    assert per == [
        f"FS42 {K} seq: step 20.0 s (interval between verified steps), verifier 13.0 s per "
        "step, plain training 7.0 s per step; verifier ÷ training 1.9 (wall time), "
        f"5.6 (accumulated GPU time); step ÷ training 2.9",
        f"FS42 {K} pipe: step 11.0 s (interval between verified steps), verifier 13.0 s per "
        "step, plain training 7.0 s per step; verifier ÷ training 1.9 (wall time), "
        f"5.6 (accumulated GPU time); step ÷ training 1.6"]
    assert sum(" plain training " in x for x in lines) == 2 * len(PKEYS)
    assert lines[-1].startswith(f"FS42 {PKEYS[-1]} pipe:")


def test_a_failed_item_prints_as_fs42_fail_and_unknown_numbers_say_so():
    parts = _parts(**{f"plain_{K}": _plain(K, record=None, steady_s=None, step_totals_s=[])})
    lines = fs._profile_lines(_prof(parts))
    assert any(x.startswith("FS42 FAIL:") and f"{K} plain" in x for x in lines)
    seq = next(x for x in lines if x.startswith(f"FS42 {K} seq:"))
    assert "plain training unknown per step" in seq and "verifier ÷ training unknown" in seq
    assert fs._profile_lines(None) == []


# ---- the part extras on real files ------------------------------------------------------------


def _rows(arm, steps=4, work=True, interval=True):
    rows = [{"record": "environment", "machine": {"gpus": base._gpus()},
             "gpu_roles": base._roles()}]
    for t in range(1, steps + 1):
        rows.append({"record": "time", "level": "total", "component": "step", "step": t,
                     "time_s": 5.0})
        if interval:
            rows.append({"record": "time", "level": "total", "component": "step_interval",
                         "step": t, "time_s": 4.0})
        rows.append({"record": "time", "level": "total", "side": "verifier",
                     "component": "verifier", "step": t, "time_s": 3.0 + t})
    if work:
        rows += [{"record": "count", "component": c, "step": 1} for c in fs.PROFILE_WORK_SECTIONS]
        rows += [{"record": "count", "component": "other"}, {"record": "storage", "step": 1}]
    return rows


def _arm_dir(tmp_path, arm, *, traced=(4,), layout=True, trace=True, **kw):
    d = tmp_path / arm / "llama_step"
    d.mkdir(parents=True)
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in _rows(arm, **kw)) + "\n{cut")
    (d / "markers.jsonl").write_text("\n".join(
        json.dumps({"kind": "step", "step": t, "traced": t in traced}) for t in range(1, 5))
        + "\n")
    if layout:
        (d / "c5_layout.json").write_text('{"entries": []}')
    if trace:
        (d / "trace").mkdir()
        with gzip.open(d / "trace" / "step_00004.json.gz", "wt") as f:
            f.write("{}")
        (d / "trace" / "meta.json").write_text(json.dumps({"error": None}))
    log = tmp_path / arm / "log"
    log.write_text(base._log(mode={"seq": "sequential"}.get(arm, "pipelined")))
    return log, d / "records.jsonl"


def test_the_run_extras_read_the_profile_inputs_from_the_run_directory(tmp_path):
    log, rec = _arm_dir(tmp_path, "seq")
    ex = fs._profile_run_extras(log, rec)
    assert ex["loop_mode"] == "sequential" and ex["step_interval_s"] == [4.0] * 4
    assert ex["verifier_s"] == [4.0, 5.0, 6.0, 7.0]
    pr = ex["profile"]
    assert pr["markers_file"] and pr["markers_steps"] == [1, 2, 3, 4] and pr["traced_steps"] == [4]
    assert pr["interval_steps"] == [1, 2, 3, 4] and pr["storage_rows"] == 1
    assert pr["count_sections"] == sorted(fs.PROFILE_WORK_SECTIONS) and pr["layout"]
    assert pr["trace_files"] == ["step_00004.json.gz"] and pr["trace_error"] is None
    assert not fs._profile_arm_problems(ex, "seq")


def test_the_run_extras_find_what_is_missing(tmp_path):
    log, rec = _arm_dir(tmp_path, "seq", traced=(), layout=False, trace=False, work=False,
                        interval=False)
    pr = fs._profile_run_extras(log, rec)["profile"]
    assert pr["traced_steps"] == [] and not pr["layout"] and pr["trace_files"] == []
    assert pr["interval_steps"] == [] and pr["count_sections"] == [] and pr["storage_rows"] == 0
    why = " | ".join(fs._profile_arm_problems({"profile": pr}, "seq"))
    for word in ("step_interval", "train.forward", "storage", "c5_layout.json", "trace"):
        assert word in why
    (rec.parent / "trace").mkdir()
    (rec.parent / "trace" / "step_00004.json.gz").write_bytes(b"")  # empty: does not count
    assert fs._profile_run_extras(log, rec)["profile"]["trace_files"] == []


def test_an_empty_layout_file_does_not_count(tmp_path):
    log, rec = _arm_dir(tmp_path, "pipe", traced=())
    (rec.parent / "c5_layout.json").write_text("")
    assert not fs._profile_run_extras(log, rec)["profile"]["layout"]


def test_the_plain_extras_read_the_p0_rows_and_the_weights(tmp_path):
    d = tmp_path / "plain_baseline"
    d.mkdir()
    rows = [{"record": "environment", "machine": {"gpus": base._gpus()}}]
    for t, s in enumerate(PLAIN_STEPS, 1):
        rows += [{"record": "time", "level": "component", "component": "P0.forward", "step": t,
                  "time_s": 1.0},
                 {"record": "time", "level": "total", "component": "step", "step": t,
                  "time_s": s}]
    rows.append({"record": "run_end"})
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ex = fs._profile_plain_extras(d / "records.jsonl")
    assert ex["p0_steps"] == [1, 2, 3] and ex["step_totals_s"] == PLAIN_STEPS
    assert ex["steady_s"] == pytest.approx(7.0) and ex["weights_bytes"] == 0
    assert fs._profile_plain_problems(ex) == [f"no {fs.WEIGHTS_FILE}"]
    (d / fs.WEIGHTS_FILE).write_text("{}")
    assert fs._profile_plain_problems(fs._profile_plain_extras(d / "records.jsonl")) == []
    assert fs._profile_plain_extras(d / "nope.jsonl")["record"] is None


def test_the_samples_header_is_read_with_its_version_and_gpu_count(tmp_path):
    p = tmp_path / "samples.jsonl"
    assert fs._samples_header(p) is None
    p.write_text(json.dumps({"kind": "header", "version": 2, "gpus": [{}, {}, {}, {}]}) + "\n")
    assert fs._samples_header(p) == {"version": 2, "gpus": 4}
    p.write_text(json.dumps({"kind": "header", "version": 1}) + "\n")
    assert fs._samples_header(p) == {"version": 1, "gpus": None}
    p.write_text("not json\n")
    assert fs._samples_header(p) is None


# ---- the session --------------------------------------------------------------------------------


def test_an_error_in_the_acceptance_is_recorded_and_the_session_runs_on(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_gpu_used_mib", lambda: {"0": 1, "1": 2, "2": 3, "3": 4})
    monkeypatch.setattr(fs, "check_upload", lambda up: {"s": 0.0, "packed": [], "models": []})

    def boom(parts, expected=None):
        raise KeyError("x")

    monkeypatch.setattr(fs, "four_h100_profile", boom)
    plan = [fs.Part("a", ["{python}", "-c", "pass"], env={}, timeout_s=30, kind="profile_run",
                    info={"arm": "seq", "model_key": "m", "run_dir": str(tmp_path),
                          "records": str(tmp_path / "r.jsonl")})]
    lines = []
    s = fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 120, cwd=Path.cwd(), say=lines.append,
                       session="four-h100-profile")
    assert s["four_h100_profile"] == {"four_h100_profile_error": "KeyError: 'x'"}
    assert any(x.startswith("FS42 FAIL: the acceptance raised (KeyError") for x in lines)
    assert s["finished"] and [p["name"] for p in s["parts"]] == ["a"]
    assert copy.deepcopy(s["four_h100_profile"])


# ---- the README's Mac-side commands ---------------------------------------------------------------

PY = "/Users/amitainevo/Projects/pots-2.0/.venv/bin/python"


def _readme_commands():
    """The profiling section's fenced commands, each joined into one line, split into the
    ``python -m`` commands it holds."""
    text = README.read_text()
    sec = text[text.index("### The four-H100 profiling session (FS42)"):]
    sec = sec[:sec.index("\n## ")]
    cmds = []
    for block in sec.split("```bash\n")[1:]:
        body = block.split("```")[0].replace("\\\n", " ")
        cmds += [ln.split() for ln in body.splitlines() if ln.startswith("python -m ")]
    return cmds


def _flags(argv):
    return [a for a in argv if a.startswith("--")]


def test_the_readme_has_the_runbook_commands_with_placeholders():
    cmds = _readme_commands()
    mods = [c[2] for c in cmds]
    assert mods == ["evaluation.results.profile.trace", "evaluation.results.profile.analyze",
                    "evaluation.results.profile.analyze", "evaluation.results.profile.report"]
    text = README.read_text()
    assert "FS11_SESSION=four-h100-profile" in text and "11,700 s" in text
    assert "summary.json[\"four_h100_profile\"]" in text
    for c in cmds:
        assert any("$PROFILES" in a for a in c) and any("--out" == a for a in c)
        assert c[2].endswith("report") or any("$OUT" in a for a in c)
    assert str(fs.four_profile_default_deadline_s()) == "11700"


def test_the_readme_commands_use_only_flags_the_tools_have_and_all_they_require():
    required = {"trace": {"--trace", "--out"}, "analyze": {"--run-dir", "--session", "--out"},
                "report": {"--out"}}
    for c in _readme_commands():
        tool = c[2].rsplit(".", 1)[1]
        help_text = subprocess.run([PY, "-m", c[2], "--help"], capture_output=True, text=True,
                                   cwd=Path(fs.__file__).parents[2]).stdout
        assert help_text.startswith("usage:"), c[2]
        flags = set(_flags(c[3:]))
        have = {f for f in set(help_text.replace("[", " ").replace("]", " ").split())
                if f.startswith("--")}
        assert flags <= have, (tool, flags)
        assert required[tool] <= flags, (tool, required[tool] - flags)
        assert len(flags) == len(_flags(c[3:])), "a flag twice"


def test_the_readme_analyze_commands_take_the_steady_steps_two_to_three():
    trace, seq, pipe, report = _readme_commands()
    for c in (seq, pipe):
        assert c[c.index("--steady-steps") + 1] == "2-3"
    assert "--steady-steps" not in trace and "--steady-steps" not in report


def test_the_readme_gives_the_cost_line_the_ratios_and_the_interval():
    text = README.read_text()
    sec = text[text.index("### The four-H100 profiling session (FS42)"):]
    sec = " ".join(sec[:sec.index("\n## ")].split())  # the README's line breaks do not count
    est = fs.four_profile_estimated_min()
    assert f"{est} min" in sec and "about 15 min of pod setup and fetch" in sec
    assert "$P/h" in sec and "$15.96/h" in sec and "11,700 s" in sec and "14,100 s" in sec
    assert "(97 + 20) x 2" in sec and "20 is the pod's setup" in sec
    assert est + fs.FOUR_PACK_MIN + fs.FOUR_GRACE_MIN == 97
    assert -(-(97 + 20) * 2 * 60 // 100) * 100 == 14100
    assert "wall time" in sec and "verify_over_train_accumulated" in sec
    assert "(accumulated GPU time)" in sec
    assert "`plain_baseline --steps 3 --metrics`" in sec
    assert "`step_interval`" in " ".join(
        text[text.index("The result reports each model's cost two ways"):].split())


def test_the_readme_commands_chain_trace_to_analyze_to_report():
    trace, seq, pipe, report = _readme_commands()

    def val(c, flag):
        return c[c.index(flag) + 1]

    assert "--trace-summary" not in pipe and "--layout" in trace and "--markers" in trace
    assert val(trace, "--out") == val(seq, "--trace-summary")
    assert val(seq, "--plain") == val(pipe, "--plain") and "plain_baseline" in val(seq, "--plain")
    assert val(seq, "--run-dir").endswith("/llama_step") and "_seq/" in val(seq, "--run-dir")
    assert val(trace, "--trace").startswith(val(seq, "--run-dir") + "/trace/step_00004.json.gz")
    assert val(trace, "--layout") == val(seq, "--run-dir") + "/c5_layout.json"
    assert val(seq, "--probe") == val(pipe, "--probe") and val(seq, "--probe").endswith("probe.json")
    profiles = [a for a in report[3:] if a.endswith(".json")]
    assert len(profiles) == 6 and len(set(profiles)) == 6
    assert {p.rsplit("_", 1)[1] for p in profiles} == {"seq.json", "pipe.json"}
    assert {p.split("/")[1].rsplit("_", 1)[0] for p in profiles} == set(fs.FOUR_PROFILE_ORDER)
    assert val(seq, "--out").replace("${MODEL}", "Qwen2.5-0.5B") in profiles


# ---- the base arm of the A/B (O18, FS49s) -----------------------------------------------------

BASE_SHA = "387b991" + "0" * 33
ROOTS_BASE = ["%064x" % i for i in (11, 12, 13, 14)]  # a different leaf layout: other roots
BASE_INTERVAL = {"seq": [40.0, 31.0, 29.0, 35.0], "pipe": [30.0, 20.0, 18.0, 22.0]}
BASE_NAMES = [f"{a}_{K}_base" for a in ("seq", "pipe")]


def _base_arm(k, arm, roots=None, **over):
    part = _arm(k, arm, roots=ROOTS_BASE if roots is None else roots,
                step_interval_s=BASE_INTERVAL[arm], **over)
    return {**part, "name": f"{arm}_{k}_base",
            "info": {**part["info"], "base": True, "base_commit": BASE_SHA, "model_key": k}}


def _base_parts(**over):
    parts = _parts()
    at = next(i for i, p in enumerate(parts) if p["name"] == f"pipe_{K}") + 1
    parts[at:at] = [_base_arm(K, "seq"), _base_arm(K, "pipe")]
    return [over.get(p["name"], p) for p in parts]


def _upload(tmp_path, base_dir=True, commit=BASE_SHA):
    up = tmp_path / "up"
    up.mkdir()
    (up / "manifest.json").write_text(json.dumps({"base_commit": commit} if commit else {}))
    if base_dir:
        (up / "repo_base").mkdir()
    return up


def test_the_plan_gets_the_base_arms_after_llamas_o18_arms(tmp_path):
    up = _upload(tmp_path)
    plan = fs.four_h100_profile_plan(up, tmp_path / "out")
    names = [p.name for p in plan]
    i = names.index(f"pipe_{K}")
    assert names[i + 1:i + 3] == BASE_NAMES and fs.FOUR_PROFILE_BASE_MODELS == (K,)
    assert sum(n.endswith("_base") for n in names) == 2
    by = {p.name: p for p in plan}
    for arm in ("seq", "pipe"):
        o18, base = by[f"{arm}_{K}"], by[f"{arm}_{K}_base"]
        assert base.argv == o18.argv and base.kind == o18.kind == "profile_run"
        skip = ("VERIF_OUTPUT_DIR", "PYTHONPATH")  # the run dir and the import root differ
        assert {k: v for k, v in base.env.items() if k not in skip} == \
            {k: v for k, v in o18.env.items() if k not in skip}
        assert base.env["PYTHONPATH"] == str(up / "repo_base") and "PYTHONPATH" not in o18.env
        assert base.env["VERIF_OUTPUT_DIR"].endswith(f"_{arm}_base")
        assert base.env["VERIF_OUTPUT_DIR"] != o18.env["VERIF_OUTPUT_DIR"]
        assert base.cwd == up / "repo_base" and o18.cwd is None
        assert base.info["arm"] == arm and base.info["base"] is True
        assert base.info["base_commit"] == BASE_SHA and "base" not in o18.info
        assert base.timeout_s == o18.timeout_s and base.info["gpu_peak"]
        assert base.info["samples_check"] and base.info["records"] != o18.info["records"]
    assert by[f"seq_{K}_base"].env["VERIF_TRACE_STEP"] == "4"  # traced as the O18 one is
    assert by[f"pipe_{K}_base"].env["VERIF_PIPELINED"] == "1"
    assert all(p.cwd is None for p in plan if not p.name.endswith("_base"))
    assert len({p.env.get("VERIF_OUTPUT_DIR") for p in plan if p.kind == "profile_run"}) == \
        sum(p.kind == "profile_run" for p in plan)


def test_without_repo_base_there_are_no_base_arms_and_the_log_says_why(tmp_path, capsys):
    plan = fs.four_h100_profile_plan(_upload(tmp_path, base_dir=False), tmp_path / "out")
    assert [p.name for p in plan] == [n for n in fs.four_profile_parts()
                                      if n != fs.NATIVE_BUILD_NAME]
    assert not any(p.name.endswith("_base") or p.cwd is not None for p in plan)
    assert "FS49 A/B arm skipped" in capsys.readouterr().out
    # the plain four-H100 session never has them, repo_base or not
    (tmp_path / "up" / "repo_base").mkdir(parents=True)
    plan = fs.four_h100_plan(tmp_path / "up", tmp_path / "out")
    assert not any(p.name.endswith("_base") for p in plan)


def test_the_estimate_counts_the_base_arms_always():
    e = fs.FOUR_PROFILE_ESTIMATE_MIN
    assert fs.four_profile_estimated_min() == 21 + 3 * (e["plain"] + e["card"] + e["seq"]
                                                        + e["pipe"]) + e["seq"] + e["pipe"]


def test_run_part_runs_in_its_own_cwd_and_the_default_is_the_sessions(tmp_path):
    other, mine = tmp_path / "other", tmp_path / "mine"
    other.mkdir()
    mine.mkdir()
    code = "import os; print(os.getcwd())"
    log = tmp_path / "log.txt"
    for cwd, want in ((other, other), (None, mine)):
        part = fs.Part("p", [fs.sys.executable, "-c", code], cwd=cwd)
        assert fs._run_part(part, dict(fs.os.environ), mine, log, deadline=time.time() + 60,
                            python=fs.sys.executable)[0] == "ok"
        assert str(want.resolve()) in log.read_text().splitlines()
        log.unlink()


def test_base_arms_accepted_with_equal_roots_pass_and_roots_are_not_compared_with_o18():
    res = _prof(_base_parts())
    assert res["pass"], _fails(res)
    assert ROOTS_BASE != ROOTS4 and res["models"][K]["roots_equal"]
    base = res["models"][K]["base"]
    assert base["accepted"] == {"seq": True, "pipe": True} and base["roots_equal"]
    assert base["base_commit"] == BASE_SHA and res["ab_skipped"] is None
    assert any(f"{K} base: the arms' step roots are equal" == t for _, t in res["items"])
    assert not any("base" in t for t in _fails(_prof()))
    plain = _prof()
    assert "ab" not in plain["models"][K] and "base" not in plain["models"][K]
    assert "repo_base" in plain["ab_skipped"]


def test_the_base_arms_report_the_same_numbers_and_the_ab_block():
    row = _prof(_base_parts())["models"][K]
    seq, pipe = row["base"]["arms"]["seq"], row["base"]["arms"]["pipe"]
    assert seq["steady_s"] == pytest.approx(30.0) and pipe["steady_s"] == pytest.approx(19.0)
    assert seq["verify_over_train"] == pytest.approx(13.0 / 7.0)
    assert seq["wall_over_train"] == pytest.approx(30.0 / 7.0)
    assert pipe["latency_steady_s"] == pytest.approx(21.0) and seq["latency_steady_s"] is None
    assert seq["accumulated_steady_s"] == pytest.approx(27.0)
    assert seq["verify_over_train_accumulated"] is not None and seq["gpu_peak_mib"]
    assert row["ab"]["seq"] == {"o18_steady_s": pytest.approx(20.0), "base_steady_s":
                                pytest.approx(30.0), "diff_s": pytest.approx(-10.0),
                                "ratio": pytest.approx(20.0 / 30.0)}
    assert row["ab"]["pipe"]["diff_s"] == pytest.approx(-8.0)
    assert row["ab"]["pipe"]["ratio"] == pytest.approx(11.0 / 19.0)
    assert json.dumps(row)


def test_a_base_arm_with_the_o18_roots_did_not_run_the_base_code():
    res = _prof(_base_parts(**{f"{a}_{K}_base": _base_arm(K, a, roots=ROOTS4)
                               for a in ("seq", "pipe")}))
    bad = _fails(res)
    assert not res["pass"] and len(bad) == 2, bad
    for a in ("seq", "pipe"):
        assert any(f"{K} base {a}" in b and "appears to have run the O18 code" in b
                   and "step 1, 2, 3, 4" in b for b in bad), bad
    # one equal step is enough (the pipelined base arm's other steps follow the seq arm's)
    one = [ROOTS4[0], *ROOTS_BASE[1:]]
    res = _prof(_base_parts(**{f"{a}_{K}_base": _base_arm(K, a, roots=one)
                               for a in ("seq", "pipe")}))
    assert any("base pipe" in b and "run the O18 code" in b and "step 1)" in b
               for b in _fails(res))


def test_unequal_base_roots_fail():
    bad = [*ROOTS_BASE[:3], "%064x" % 99]
    res = _prof(_base_parts(**{f"pipe_{K}_base": _base_arm(K, "pipe", roots=bad)}))
    _one_fail(res, K, "base", "step roots differ")


def test_a_base_arm_not_accepted_or_with_missing_inputs_fails():
    res = _prof(_base_parts(**{f"seq_{K}_base": _base_arm(K, "seq", accepted=False)}))
    assert not res["pass"] and any("base: the seq arm was not accepted" in t for t in _fails(res))
    res = _prof(_base_parts(**{f"pipe_{K}_base": _base_arm(
        K, "pipe", profile=_profile("pipe", layout=False))}))
    _one_fail(res, f"{K} base pipe", "layout")
    res = _prof(_base_parts(**{f"seq_{K}_base": _base_arm(K, "seq", samples={"version": 1})}))
    assert not res["pass"] and any(f"seq_{K}_base" in t for t in _fails(res))
    res = _prof(_base_parts(**{f"seq_{K}_base": _base_arm(K, "seq", gpu_peak_mib=None,
                                                          device_peaks={})}))
    assert res["missing_numbers"] and any(f"{K} base seq" in m for m in res["missing_numbers"])


def test_a_missing_base_part_fails_only_when_planned():
    parts = _base_parts()
    gone = [{**p, "status": "not_started", "extras": None} if p["name"] == f"seq_{K}_base" else p
            for p in parts]
    res = _prof(gone)
    assert not res["pass"] and any(f"seq_{K}_base did not run" in t for t in _fails(res))
    assert _prof(_parts())["pass"]  # not in the plan: nothing fails


def test_the_ab_lines_hold_the_numbers():
    lines = fs._profile_lines(_prof(_base_parts()))
    ab = [x for x in lines if x.startswith("FS49 A/B")]
    assert ab == [f"FS49 A/B {K} seq: O18 20.0 s, base 30.0 s (Δ -10.0 s, ratio 0.67)",
                  f"FS49 A/B {K} pipe: O18 11.0 s, base 19.0 s (Δ -8.0 s, ratio 0.58)"]
    assert not any(x.startswith("FS49") for x in fs._profile_lines(_prof()))
