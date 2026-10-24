"""FS33d-2 tester: the pair session's pipelined arms against the formats the entry points really
print, the gate's fail-closed paths, the plan's environments, the deadline and the job log."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.verification.runs.test_fs32_pair_session import (
    _card, _cheat, _parts, _run, _suite)
from tests.verification.runs.test_run_verified import tiny_main  # noqa: F401  (fixture)
from verification.runs import cheats
from verification.runs import fs11_session as fs
from verification.runs import llama_step
from verification.runs import run_verified as rv
from verification.runs.scenarios import Expected, Scenario
from verification.verifier.bands import Bands
from verification.verifier.context import StepStats

JOB = Path(fs.__file__).with_name("fs11_job.sh")
README = Path(fs.__file__).parents[1] / "README.md"


# ---- real formats --------------------------------------------------------------------------


def test_cheat_extras_reads_what_report_oracle_really_prints(tmp_path):
    from verification.runs.fs11_session import PAIR_CHEATS
    rej = None
    res = []
    for name in PAIR_CHEATS:
        sc = Scenario(name, "d", None, Expected())
        res.append(SimpleNamespace(
            scenario=sc, loop=SimpleNamespace(rejection=rej), verifier=SimpleNamespace(stats={}),
            actual=Expected(), outcome="exact", passed=True))
    lines = []
    cheats.report_oracle(res, Bands.provisional(), out=lines.append)
    log = tmp_path / "log"
    log.write_text("loop mode: pipelined\n" + "\n".join(lines) + "\n")
    e = fs._cheat_extras(log)
    assert e["outcomes"] == {n: "exact" for n in PAIR_CHEATS}
    assert set(e["oracle_lines"]) == set(PAIR_CHEATS)
    assert all(v.startswith(n + " accept accept exact") for n, v in e["oracle_lines"].items())


def test_cheat_extras_reads_a_real_rejected_row(tmp_path):
    from verification.runs.fs11_session import _cheat_extras  # noqa: F401
    rej = SimpleNamespace(step=2, check_id="5", kind="residual", detail="P_3 (x) too big")
    exp = Expected(2, "5", "residual", 3)
    sc = Scenario("A2", "d", None, exp)
    r = SimpleNamespace(scenario=sc, loop=SimpleNamespace(rejection=rej),
                        verifier=SimpleNamespace(stats={2: StepStats()}), actual=exp,
                        outcome="exact", passed=True)
    lines = []
    cheats.report_oracle([r], Bands.provisional(), out=lines.append)
    log = tmp_path / "log"
    log.write_text("\n".join(lines) + "\n")
    e = fs._cheat_extras(log)
    assert e["outcomes"] == {"A2": "exact"}, lines


def test_step_times_reads_what_report_costs_really_prints():
    steps = [SimpleNamespace(t=t, prove_s=4.0 + t, commit_s=1.0, verify_s=7.5, handoff_s=0.25)
             for t in (1, 2, 3)]
    r = SimpleNamespace(loop=SimpleNamespace(steps=steps),
                        verifier=SimpleNamespace(timings={1: {"0": 0.5, "5": 1.25}, 2: {}, 3: {}}))
    lines = []
    llama_step.report_costs(r, out=lines.append)
    got = fs._step_times(lines)
    assert [g["step"] for g in got] == [1, 2, 3]
    assert [(g["prover_s"], g["verifier_s"], g["handoff_s"]) for g in got] == [
        (6.0, 7.5, 0.25), (7.0, 7.5, 0.25), (8.0, 7.5, 0.25)]
    assert got[0]["total_s"] == 13.75


def test_a_real_run_verified_log_gives_every_pair_extra(  # noqa: F811
        tiny_main, capsys, monkeypatch):
    monkeypatch.setenv("VERIF_STEPS", "3")
    assert rv.main(["--steps", "3", "--keep-step", "3", "--no-metrics"]) == 0
    text = capsys.readouterr().out
    scratch = Path(tiny_main[2])
    log = scratch / "real.log"
    # run_verified prints its own mode line (FS33d), exactly one
    assert text.count("loop mode: ") == 1 and "loop mode: sequential" in text, text[:600]
    log.write_text(text)
    e = fs._pair_run_extras(log, scratch / "nope.json")
    assert len(e["step_lines"]) == 3 and e["verdicts"] == ["accepted"] * 3
    assert len(e["step_roots"]) == 3 and all(len(x) == 64 for x in e["step_roots"])
    assert re.fullmatch(r"[0-9a-f]{16}", e["kept_root"] or ""), text[-600:]
    assert e["loop_mode"] == "sequential"


# ---- fails closed --------------------------------------------------------------------------


def _vary(run, **kw):
    run["extras"] = {**run["extras"], **kw}
    return run


@pytest.mark.parametrize("arm", ["pipelined", "pipelined_metrics"])
@pytest.mark.parametrize("key,val", [
    ("final_root", "z" * 8), ("kept_root", "c" * 16), ("losses", [9.0, 1.0, 1.0]),
    ("step_roots", ["r1", "r2", "rX"]), ("step_lines", ["accepted; a", "accepted; b",
                                                         "accepted; c"]),
    ("verdicts", ["accepted", "accepted", "REJECTED"])])
def test_any_single_difference_in_a_pipelined_arm_is_no_pass(arm, key, val):
    assert fs.fs32_pair(_parts())["pass"]
    pair = fs.fs32_pair(_parts(**{f"run_{arm}": _vary(_run(arm), **{key: val})}))
    assert not pair["pass"]


@pytest.mark.parametrize("arm", ["pipelined", "pipelined_metrics"])
def test_a_pipelined_arm_with_fewer_than_three_steps_is_no_pass(arm):
    r = _run(arm)
    r["extras"] = {**r["extras"], "step_roots": ["r1", "r2"]}
    assert not fs.fs32_pair(_parts(**{f"run_{arm}": r}))["pass"]
    r = _run(arm)
    r["extras"] = {**r["extras"], "step_lines": r["extras"]["step_lines"][:2],
                   "verdicts": ["accepted"] * 2}
    assert not fs.fs32_pair(_parts(**{f"run_{arm}": r}))["pass"]


@pytest.mark.parametrize("arm", ["pipelined", "pipelined_metrics"])
def test_a_pipelined_arm_missing_every_value_never_equals_a_missing_sequential(arm):
    seq, pip = _run("sharded"), _run(arm)
    for k in ("final_root", "kept_root", "losses", "step_roots", "step_lines", "verdicts"):
        seq["extras"][k] = None
        pip["extras"][k] = None
    assert not fs.fs32_pair(_parts(run_sharded=seq, **{f"run_{arm}": pip}))["pass"]


@pytest.mark.parametrize("name,names", [("poisoned_step_pipelined", ("A1", "A2", "A3")),
                                        ("hidden_steps_pipelined", ("hidden",)),
                                        ("flipped_matmul_pipelined", ("flip",))])
@pytest.mark.parametrize("how", ["outcome", "row", "no_rows", "no_outcomes", "missing",
                                 "failed", "timeout"])
def test_a_bad_pipelined_cheat_is_no_pass(name, names, how):
    c = _cheat(name, names, pipelined=True)
    if how == "outcome":
        c = _cheat(name, names, pipelined=True, outcome="rejected")
    elif how == "row":
        c["extras"]["oracle_lines"] = {n: f"{n} other row exact 9 9 9" for n in names}
    elif how == "no_rows":
        c["extras"]["oracle_lines"] = {}
    elif how == "no_outcomes":
        c["extras"]["outcomes"] = {}
    elif how in ("failed", "timeout"):
        c["status"] = how
    parts = [p for p in _parts(**{name: c}) if not (how == "missing" and p["name"] == name)]
    pair = fs.fs32_pair(parts)
    assert not pair["pass"]
    if how in ("failed", "timeout", "missing"):
        assert any(f"FS32 FAIL: {name} " in x for x in fs._pair_lines(pair))


@pytest.mark.parametrize("name", ["run_pipelined", "run_pipelined_metrics"])
@pytest.mark.parametrize("how", ["failed", "timeout", "missing"])
def test_a_bad_pipelined_run_part_is_a_named_fail_line(name, how):
    arm = name.removeprefix("run_")
    r = _run(arm)
    if how != "missing":
        r["status"] = how
    parts = [p for p in _parts(**{name: r}) if not (how == "missing" and p["name"] == name)]
    pair = fs.fs32_pair(parts)
    assert not pair["pass"]
    assert any(x.startswith(f"FS32 FAIL: {name} ") for x in fs._pair_lines(pair))


LOOP_PARTS = ("run_unsharded", "run_sharded", "poisoned_step", "hidden_steps", "flipped_matmul",
              "run_pipelined", "run_pipelined_metrics", "poisoned_step_pipelined",
              "hidden_steps_pipelined", "flipped_matmul_pipelined")


@pytest.mark.parametrize("name", LOOP_PARTS)
@pytest.mark.parametrize("bad", ["wrong", None, "garbage"])
def test_a_wrong_or_missing_mode_line_on_any_loop_part_is_no_pass(name, bad):
    parts = _parts()
    p = next(x for x in parts if x["name"] == name)
    flip = {"pipelined": "sequential", "sequential": "pipelined"}
    p["extras"] = {**p["extras"], "loop_mode": (flip[p["extras"]["loop_mode"]] if bad == "wrong"
                                                  else bad)}
    pair = fs.fs32_pair(parts)
    assert not pair["pass"] and name in pair["mode_problems"]
    assert any(x.startswith(f"FS32 FAIL: {name} ") for x in fs._pair_lines(pair))


def test_the_full_matching_set_passes_and_has_the_ten_loop_parts():
    pair = fs.fs32_pair(_parts())
    assert pair["pass"] and pair["mode_problems"] == {}
    assert set(LOOP_PARTS) == set(fs.PAIR_SEQUENTIAL_PARTS) | set(fs.PAIR_PIPELINED_PARTS)


def test_a_pipelined_cheat_part_with_its_sequential_twin_missing_is_no_pass():
    parts = [p for p in _parts() if p["name"] != "hidden_steps"]
    assert not fs.fs32_pair(parts)["pass"]


def test_a_mode_line_must_start_the_line_and_two_such_lines_are_multiple(tmp_path):
    log = tmp_path / "l"
    log.write_text("loop mode: sequential\nloop mode: pipelined\n")
    assert fs._loop_mode(log.read_text().splitlines()) == "multiple"
    log.write_text("loop mode: sequential\n  loop mode: pipelined\n")  # a quote, not a line
    assert fs._loop_mode(log.read_text().splitlines()) == "sequential"
    log.write_text("   loop mode: pipelined\n")  # indented: not the printed line
    assert fs._loop_mode(log.read_text().splitlines()) is None


# ---- the plan ------------------------------------------------------------------------------


def _plan(tmp_path):
    return {p.name: p for p in fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")}


PIPED = ("run_pipelined", "run_pipelined_metrics", "poisoned_step_pipelined",
         "hidden_steps_pipelined", "flipped_matmul_pipelined")


def test_only_the_five_pipelined_parts_set_the_mode_and_to_one(tmp_path):
    plan = _plan(tmp_path)
    for n, p in plan.items():
        if n in PIPED:
            assert p.env["VERIF_PIPELINED"] == "1", n
            assert p.env["VERIF_VERIFIER_DEVICES"] == "cuda:0,cuda:1", n
        else:
            assert p.env.get("VERIF_PIPELINED", "absent") in (None, "absent"), n


def test_every_part_unsets_the_mode_unless_it_is_pipelined(tmp_path, monkeypatch):
    """A job environment that sets VERIF_PIPELINED must not reach any other part: the part's
    env must hold the key as None (it is popped), the peer check, models and build included."""
    plan = fs.with_native_build(list(_plan(tmp_path).values()))
    leaks = [p.name for p in plan if p.name not in PIPED and "VERIF_PIPELINED" not in p.env]
    assert not leaks, leaks


def test_a_job_environment_that_sets_the_mode_reaches_only_the_pipelined_parts(tmp_path,
                                                                                monkeypatch):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    monkeypatch.setenv("VERIF_VERIFIER_DEVICES", "cuda:0")
    env = fs.base_env(tmp_path / "up")
    assert env["VERIF_PIPELINED"] == "1"
    for n, p in _plan(tmp_path).items():
        penv = dict(env)
        for k, v in p.env.items():
            if v is None:
                penv.pop(k, None)
            else:
                penv[k] = v
        if n in PIPED:
            assert penv["VERIF_PIPELINED"] == "1"
        elif p.kind in ("pytest", "pair_run", "pair_card", "cheat", "other"):
            assert "VERIF_PIPELINED" not in penv, n


def test_pipelined_arms_use_the_same_steps_and_kept_step_and_own_directories(tmp_path):
    plan = _plan(tmp_path)
    seq = plan["run_sharded"].argv
    for n in ("run_pipelined", "run_pipelined_metrics"):
        a = plan[n].argv
        assert a[a.index("--steps"):a.index("--steps") + 2] == seq[seq.index("--steps"):
                                                                  seq.index("--steps") + 2]
        assert a[a.index("--keep-step"):a.index("--keep-step") + 2] == seq[
            seq.index("--keep-step"):seq.index("--keep-step") + 2]
    for c in ("poisoned_step", "hidden_steps", "flipped_matmul"):
        assert plan[c + "_pipelined"].argv[:3] == plan[c].argv[:3]
        assert plan[c + "_pipelined"].argv[3:] == plan[c].argv[3:]
    dirs = [p.env["VERIF_OUTPUT_DIR"] for p in plan.values()
            if p.kind not in ("pytest", "exit_probe") and p.name not in ("models", "peer_check")]
    assert len(set(dirs)) == len(dirs)
    assert list(plan)[0] == "peer_check"
    assert plan["run_pipelined_metrics"].timeout_s >= plan["run_pipelined"].timeout_s


def test_part_weights_paths_follow_each_arms_own_directory(tmp_path):
    plan = _plan(tmp_path)
    for n in ("run_pipelined", "run_pipelined_metrics"):
        w = Path(plan[n].info["weights"])
        assert str(w).startswith(plan[n].env["VERIF_OUTPUT_DIR"])


# ---- the deadline --------------------------------------------------------------------------


def test_the_deadline_matches_the_job_and_the_readmes_boot_guard():
    d = fs.pair_default_deadline_s()
    m = re.search(r"fs32-pair \] && DEFAULT_DEADLINE_S=(\d+)", JOB.read_text())
    assert m and int(m[1]) == d
    assert f"{d:,}" in (Path(fs.__file__).parents[1] / "CLAUDE.md").read_text()
    est = sum(fs.PAIR_ESTIMATE_MIN.values()) + fs.PAIR_PACK_MIN + fs.PAIR_GRACE_MIN
    assert d >= 2 * est * 60 and d - 2 * est * 60 < 100
    text = README.read_text()
    guard = re.search(r"`sleep (\d+);` \(setup about 20 min plus the job's (\d+)", text)
    assert guard and int(guard[2]) == est
    assert int(guard[1]) > d + 20 * 60  # the guard exceeds the deadline plus setup
    assert f"{d:,}" in text


def test_every_arm_has_an_estimate_and_a_timeout_above_it():
    t, e = fs.PAIR_TIMEOUT_S, fs.PAIR_ESTIMATE_MIN
    assert t["run_metrics"] >= 60 * e["run_pipelined_metrics"]
    assert t["run"] >= 60 * e["run_pipelined"]
    assert t["cheat"] >= 60 * e["cheat_pipelined"] / 3


# ---- the job log ---------------------------------------------------------------------------


def test_job_log_for_a_pass_has_no_fail_line():
    lines = fs._pair_lines(fs.fs32_pair(_parts()))
    assert not [x for x in lines if "FAIL" in x or "NOT exact" in x or "differs" in x]
    assert any(x.startswith("FS32 step wall time pipelined_metrics") for x in lines)


def test_job_log_names_the_part_and_the_reason_for_each_failure():
    cases = {
        "ran sequentially": _vary(_run("pipelined"), loop_mode="sequential"),
    }
    pair = fs.fs32_pair(_parts(run_pipelined=cases["ran sequentially"]))
    assert "FS32 FAIL: run_pipelined ran sequentially" in fs._pair_lines(pair)
    pair = fs.fs32_pair(_parts(run_sharded=_vary(_run("sharded"), loop_mode="pipelined")))
    assert "FS32 FAIL: run_sharded ran pipelined" in fs._pair_lines(pair)
    f = _run("pipelined_metrics")
    f["status"] = "timeout"
    pair = fs.fs32_pair(_parts(run_pipelined_metrics=f))
    assert "FS32 FAIL: run_pipelined_metrics ended timeout" in fs._pair_lines(pair)
    pair = fs.fs32_pair([p for p in _parts() if p["name"] != "flipped_matmul_pipelined"])
    assert "FS32 FAIL: flipped_matmul_pipelined did not run" in fs._pair_lines(pair)


def test_a_differing_pipelined_root_reads_in_the_roots_line():
    pair = fs.fs32_pair(_parts(run_pipelined=_vary(_run("pipelined"), step_roots=["r1", "r2",
                                                                                  "x"])))
    lines = fs._pair_lines(pair)
    assert any(x.startswith("FS32 roots equal") and "NO" in x.upper() for x in lines), lines[:3]


def test_step_time_lines_survive_missing_times():
    r = _run("pipelined")
    r["extras"]["step_times"] = [{"prover_s": None, "verifier_s": 1.0, "handoff_s": None,
                                  "total_s": None}]
    lines = fs._pair_lines(fs.fs32_pair(_parts(run_pipelined=r)))
    assert any("pipelined (prover" in x and "?/1.0/?/?" in x for x in lines)


def test_json_roundtrip_of_the_pair_summary():
    json.dumps(fs.fs32_pair(_parts()))
