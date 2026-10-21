"""FS30's device-leaves session: its plan, environment and timeouts, the bring-up line it reads,
the gate (its pass and every way to fail), and the job's switch. Nothing here needs CUDA."""

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.verification.runs import test_fs11_session as t11
from verification.runs import bring_up as bu
from verification.runs import fs11_session as fs

REPO = Path(__file__).resolve().parents[3]
ROOT = "ab" * 32
OTHER = "cd" * 32
GOOD_TESTS = "tests.verification.runs.test_device_leaves_cuda"


def plan(tmp_path):
    return fs.device_leaves_plan(tmp_path / "up", tmp_path / "out")


# ---- the plan ---------------------------------------------------------------------------------


def test_the_plan_is_models_tests_then_one_bring_up_per_model_cheapest_first(tmp_path):
    label, build = fs.SESSIONS["device-leaves"]
    assert label == "device-leaves"
    chosen = fs.choose_plan(build, tmp_path / "up", tmp_path / "out", plan_file=None,
                            selftest=False)
    names = [p.name for p in chosen]
    assert names[:3] == ["native_build", "models", "device_tests"]
    bring = chosen[3:]
    assert [p.info["model"] for p in bring][0] == fs.FIX_AB_MODEL
    assert sorted(p.info["model"] for p in bring) == sorted(r for r, _ in fs.RUN_PINS)
    assert [p.info["model"] for p in bring][1:] == [r for r, _ in fs.RUN_PINS
                                                    if r != fs.FIX_AB_MODEL]
    for p in bring:
        assert p.kind == "bring_up"
        assert p.argv[:3] == ["{python}", "-m", "verification.runs.bring_up"]
        assert "--no-tamper" in p.argv and "--device-leaves" in p.argv
        for k, v in {**fs.FULL_SCALE_ENV, **fs.RECHECK_ENV}.items():
            assert p.env[k] == v


def test_the_tests_part_runs_the_cuda_tests_with_a_junit_report(tmp_path):
    tests = plan(tmp_path)[1]
    assert tests.kind == "device_tests"
    assert tests.info["junit"] == str(tmp_path / "out" / "parts" / "device_tests" / "junit.xml")
    assert f"--junitxml={tests.info['junit']}" in tests.argv
    wanted = [a for a in tests.argv if a.startswith("tests/")]
    assert wanted == list(fs.DEVICE_LEAVES_TESTS)
    for a in wanted:
        assert (REPO / a.split("::")[0]).exists()
    assert set(fs.KERNEL_TESTS) <= set(wanted)
    assert fs.DEVICE_TESTS_FILE in wanted


def test_the_env_has_no_secrets_and_every_part_a_timeout(tmp_path):
    for p in fs.with_native_build(plan(tmp_path)):
        assert p.timeout_s and p.timeout_s <= 1800
        assert not set(p.env) & set(fs.SECRETS)
        assert all("TOKEN" not in k.upper() and not k.upper().endswith("_KEY") for k in p.env)
        if p.kind != "models":
            assert p.env.get("HF_HUB_OFFLINE", "1") == "1"
    # the session runs through base_env, which strips every secret
    env = fs.base_env(tmp_path / "up")
    assert not set(env) & set(fs.SECRETS)


def test_the_deadline_is_about_twice_the_estimate_and_the_parts_start_inside_it():
    est_min = 5 + 2 + 8 + (10 + 10 + 3 + 14) + 1 + 20  # the plan's comment lists the figures
    assert fs.DEVICE_LEAVES_DEADLINE_S == 8800
    assert 2 * est_min * 60 <= fs.DEVICE_LEAVES_DEADLINE_S < 2 * est_min * 60 + 200
    t = fs.DEVICE_LEAVES_TIMEOUT_S
    assert t["tests"] + t["models"] + 900 < fs.DEVICE_LEAVES_DEADLINE_S - 1200 - 600


def test_the_other_sessions_plans_are_unchanged(tmp_path):
    for key, n in (("kernel-speed", 2), ("nvlink-probe", None)):
        parts = fs.SESSIONS[key][1](tmp_path / "up", tmp_path / "out")
        assert all(p.kind != "device_tests" for p in parts)
        assert all("--device-leaves" not in p.argv for p in parts)
        if n:
            assert len(parts) == n
    for key in ("fs11", "recheck", "profile-check", "fs32-pair"):
        parts = fs.SESSIONS[key][1](tmp_path / "up", tmp_path / "out")
        assert all(p.kind != "device_tests" for p in parts)
        assert all("--device-leaves" not in p.argv for p in parts)


def test_the_session_refuses_without_cuda_or_an_h100():
    cpu = {"cuda_available": False, "gpus": []}
    assert "is_available() is false" in fs.refusal(cpu, False)
    a40 = {"cuda_available": True, "gpus": ["NVIDIA A40"]}
    assert "no H100" in fs.refusal(a40, False)
    assert fs.refusal(a40, True) is None  # FS11_ALLOW_ANY_GPU=1
    assert fs.refusal({"cuda_available": True, "gpus": ["NVIDIA H100 80GB HBM3"]}, False) is None


# ---- the bring-up line ------------------------------------------------------------------------


def line(**kw):
    d = dict(acc=True, prover=ROOT, cpu=ROOT, arrival=100, hashed=120, known=100, M=120, dK=20)
    d.update(kw)
    return (f"device leaves: step 1 accepted {d['acc']}, prover root {d['prover']}, "
            f"cpu root {d['cpu']}, arrival digests {d['arrival']}, hashed {d['hashed']}, "
            f"commit known {d['known']}, M {d['M']}, dK leaves {d['dK']}")


def test_the_log_line_is_read_into_the_bring_ups_extras(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("noise\n" + line() + "\n")
    got = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["device_leaves"]
    assert got == {"step": 1, "accepted": True, "prover_root": ROOT, "cpu_root": ROOT,
                   "arrival": 100, "hashed": 120, "known": 100, "M": 120, "dK": 20}
    log.write_text("no line here\n")
    assert fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["device_leaves"] is None


def test_bring_up_prints_the_line_it_reads(tmp_path):
    steps = [SimpleNamespace(t=1, rejection=None, root=bytes.fromhex(ROOT),
                             capture={"arrival_hash": {"leaves": 9, "used": 7}}, known=6)]
    kept = SimpleNamespace(leaf=lambda i: None)
    b = SimpleNamespace(kept=kept, result=SimpleNamespace(loop=SimpleNamespace(steps=steps)))
    c = SimpleNamespace(n_leaves=0, M=9, products=[])
    import verification.runs.bring_up as mod
    orig = mod.MerkleTree, mod.leaf_hashes_of
    mod.leaf_hashes_of = lambda c_, items: []
    mod.MerkleTree = lambda hashes: SimpleNamespace(root=bytes.fromhex(ROOT))
    try:
        text = bu.device_leaves_line(b, c, T=1)
    finally:
        mod.MerkleTree, mod.leaf_hashes_of = orig
    log = tmp_path / "log.txt"
    log.write_text(text + "\n")
    got = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["device_leaves"]
    assert got["arrival"] == 7 and got["hashed"] == 9 and got["M"] == 9 and got["dK"] == 0
    assert got["known"] == 6
    steps[0].known = None
    with pytest.raises(ValueError, match="known count"):
        bu.device_leaves_line(b, c, T=1)
    assert got["prover_root"] == got["cpu_root"] == ROOT
    with pytest.raises(ValueError, match="keep_last"):
        bu.device_leaves_line(SimpleNamespace(kept=None), c, T=1)


def test_bring_up_prints_the_missing_line_after_it():
    """The second line, for a person; nothing parses it. An honest decoder step: the ``dK``
    products only, and no others."""
    def line(ah):
        step = SimpleNamespace(t=1, capture={"arrival_hash": ah})
        b = SimpleNamespace(result=SimpleNamespace(loop=SimpleNamespace(steps=[step])))
        return bu.device_leaves_missing_line(b, T=1)

    assert line({"missing": {"OPERAND_GRAD": 20}, "missing_first": []}) == (
        "device leaves missing: OPERAND_GRAD 20; first others: none")
    assert line({"missing": {"OPERAND_GRAD": 20, "FORWARD": 2, "WEIGHT_GRAD": 1},
                 "missing_first": ["fwd.a", "fwd.b", "wg.c"]}) == (
        "device leaves missing: OPERAND_GRAD 20, FORWARD 2, WEIGHT_GRAD 1; "
        "first others: fwd.a, fwd.b, wg.c")
    assert line({}) == "device leaves missing: none; first others: none"


# ---- the gate ---------------------------------------------------------------------------------


EXPECTED = fs.device_tests_expected()[0]  # every case the real file defines, by junit name
FIRST = EXPECTED[0]


def _tests_part(status="ok", cases=None, failures=0, errors=0, junit=True):
    cases = cases if cases is not None else {
        **{n: "passed" for n in EXPECTED},
        "tests.verification.prover.test_capture_cuda::test_gpu_memory_does_not_hold_the_transcript":
            "passed",
        "tests.verification.commitment.test_gpu_blake3::test_k": "passed"}
    ex = {"tests": len(cases), "failures": failures, "errors": errors, "skipped": 0,
          "failed": [], "cases": cases} if junit else {"junit": None}
    return {"name": "device_tests", "kind": "device_tests", "status": status, "extras": ex}


def _bring(repo, **kw):
    ex = {"device_leaves": None if kw.get("none") else dict(
        step=1, accepted=kw.get("acc", True), prover_root=kw.get("prover", ROOT),
        cpu_root=kw.get("cpu", ROOT), arrival=kw.get("arrival", 100),
        hashed=kw.get("hashed", 120), known=kw.get("known", 100), M=120, dK=kw.get("dK", 20))}
    return {"name": "bring_up_" + repo.split("/")[-1], "kind": "bring_up",
            "status": kw.get("status", "ok"), "info": {"model": repo}, "extras": ex}


def _parts(tests=None, **per):
    out = [{"name": "native_build", "status": "ok"}, tests or _tests_part()]
    for repo, _rev in fs.RUN_PINS:
        out.append(_bring(repo, **per.get(repo, {})))
    return out


def test_no_tests_part_means_no_gate():
    assert fs.device_leaves_gate([{"name": "x", "status": "ok"}]) is None


def test_the_gate_passes_when_everything_holds():
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is True and g["reasons"] == []
    assert g["line"].startswith("FS30 device leaves pass")
    assert set(g["models"]) == {r for r, _ in fs.RUN_PINS}


def _fails(parts, needle):
    g = fs.device_leaves_gate(parts)
    assert g["pass"] is False and needle in " | ".join(g["reasons"]), g
    assert g["line"].startswith("FS30 device leaves FAIL")


M0 = fs.RUN_PINS[0][0]
M1 = fs.RUN_PINS[1][0]


@pytest.mark.parametrize("status", ["failed", "timeout", "not_started", "killed_signal"])
def test_a_tests_part_that_did_not_finish_ok_fails(status):
    _fails(_parts(tests=_tests_part(status=status)), "the CUDA tests part is")


def test_a_tests_failure_or_error_fails():
    _fails(_parts(tests=_tests_part(failures=1)), "failures")
    _fails(_parts(tests=_tests_part(errors=1)), "errors")


def test_a_missing_junit_report_fails():
    _fails(_parts(tests=_tests_part(junit=False)), "no junit report")


def test_a_skipped_or_failed_fs30_test_fails():
    cases = _tests_part()["extras"]["cases"]
    for bad in ("skipped", "failed", "error"):
        c = dict(cases)
        c[FIRST] = bad
        _fails(_parts(tests=_tests_part(cases=c)), f"{FIRST}: {bad}")
    c = dict(cases)
    c["tests.verification.prover.test_capture_cuda::test_gpu_memory_does_not_hold_the_transcript"] \
        = "skipped"
    _fails(_parts(tests=_tests_part(cases=c)), "test_gpu_memory_does_not_hold_the_transcript")


def test_fs30_tests_that_never_ran_fail():
    only_kernel = {"tests.verification.commitment.test_gpu_blake3::test_k": "passed"}
    _fails(_parts(tests=_tests_part(cases=only_kernel)), "none of FS30's CUDA tests ran")
    _fails(_parts(tests=_tests_part(cases=only_kernel)), "GPU-memory test did not run")


def test_the_expected_names_come_from_the_file_with_every_parametrized_case():
    names, err = fs.device_tests_expected()
    assert err == "" and len(names) == len(set(names))
    text = (REPO / fs.DEVICE_TESTS_FILE).read_text()
    defs = re.findall(r"^def (test_\w+)", text, re.M)
    assert {n.split("::")[1].split("[")[0] for n in names} == set(defs)
    assert all(n.startswith(GOOD_TESTS + "::") for n in names)
    assert sum("[" in n for n in names) >= 4  # the dtype x head_blocks cases are each named


def test_a_case_missing_from_the_report_fails_naming_it():
    for gone in (FIRST, EXPECTED[-1]):
        cases = _tests_part()["extras"]["cases"]
        del cases[gone]
        _fails(_parts(tests=_tests_part(cases=cases)),
               f"{gone}: expected but not in the junit report")


def test_every_missing_case_is_named_and_a_removed_parametrized_case_counts():
    cases = {k: v for k, v in _tests_part()["extras"]["cases"].items() if "[" not in k}
    g = fs.device_leaves_gate(_parts(tests=_tests_part(cases=cases)))
    gone = [n for n in EXPECTED if "[" in n]
    assert g["pass"] is False
    assert all(f"{n}: expected but not in the junit report" in g["reasons"] for n in gone)


def test_an_unknown_extra_case_that_passed_is_let_through_and_one_that_failed_fails():
    cases = _tests_part()["extras"]["cases"]
    cases[f"{GOOD_TESTS}::test_new_one"] = "passed"
    assert fs.device_leaves_gate(_parts(tests=_tests_part(cases=cases)))["pass"] is True
    cases[f"{GOOD_TESTS}::test_new_one"] = "failed"
    _fails(_parts(tests=_tests_part(cases=cases)), "test_new_one: failed")


def test_a_failed_collection_fails_the_gate_with_its_reason(monkeypatch):
    monkeypatch.setattr(fs, "device_tests_expected", lambda *a: ([], "boom"))
    _fails(_parts(), "could not be collected: boom")


def test_a_kernel_test_skip_is_not_an_fs30_skip():
    cases = _tests_part()["extras"]["cases"]
    cases["tests.verification.commitment.test_gpu_blake3::test_k"] = "skipped"
    assert fs.device_leaves_gate(_parts(tests=_tests_part(cases=cases)))["pass"] is True


def test_a_model_with_no_bring_up_or_no_line_fails():
    parts = [p for p in _parts() if (p.get("info") or {}).get("model") != M0]
    _fails(parts, f"{M0}: no bring-up part")
    _fails(_parts(**{M0: {"none": True, "status": "failed"}}), "no device leaves line")


def test_a_step_not_accepted_fails():
    _fails(_parts(**{M1: {"acc": False}}), "the step was not accepted")


def test_roots_that_differ_or_are_missing_fail():
    _fails(_parts(**{M1: {"cpu": OTHER}}), "the roots differ")
    _fails(_parts(**{M1: {"prover": None}}), "the roots differ")


@pytest.mark.parametrize("arrival", [99, 101, 0, 120])
def test_an_arrival_count_other_than_m_less_dk_fails(arrival):
    _fails(_parts(**{M1: {"arrival": arrival}}), "arrival digests, expected 100")


@pytest.mark.parametrize("known", [99, 101, 0, 120])
def test_a_commit_known_count_other_than_m_less_dk_fails(known):
    _fails(_parts(**{M1: {"known": known}}), "the commit took")
    _fails(_parts(**{M1: {"known": known}}), f"{known} leaves from arrival digests, expected 100")


@pytest.mark.parametrize("hashed", [119, 121, 0, 100])
def test_a_hashed_count_other_than_m_fails(hashed):
    _fails(_parts(**{M1: {"hashed": hashed}}), "products hashed on arrival, expected 120")


def test_the_gate_is_in_the_job_log_and_the_summary(tmp_path):
    lines = fs._hw_lines(_parts(), None)
    assert len(lines) == 1 and lines[0].startswith("FS30 device leaves pass")
    assert fs._hw_lines([{"kind": "plain", "status": "ok"}], None) == []


def test_the_tests_extras_name_every_case(tmp_path):
    junit = tmp_path / "j.xml"
    junit.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="a.b" name="t1"/>'
        '<testcase classname="a.b" name="t2"><skipped message="no cuda"/></testcase>'
        '<testcase classname="a.b" name="t3"><failure message="x"/></testcase>'
        '<testcase classname="a.b" name="t4"><error message="x"/></testcase>'
        '</testsuite></testsuites>')
    ex = fs.part_extras(fs.Part("t", [], kind="device_tests", info={"junit": str(junit)}),
                        tmp_path / "log")
    assert ex["cases"] == {"a.b::t1": "passed", "a.b::t2": "skipped", "a.b::t3": "failed",
                           "a.b::t4": "error"}
    assert (ex["tests"], ex["failures"], ex["errors"], ex["skipped"]) == (4, 1, 1, 1)
    missing = fs.part_extras(fs.Part("t", [], kind="device_tests",
                                     info={"junit": str(tmp_path / "none.xml")}), tmp_path / "log")
    assert missing == {"junit": None}


# ---- the job ----------------------------------------------------------------------------------


def test_the_job_accepts_the_session_with_its_deadline(tmp_path):
    r, out = t11._job(tmp_path, plan=t11.OK_PLAN,
                      extra={"FS11_SESSION": "device-leaves", "FS11_ALLOW_ANY_GPU": "1",
                             "FS11_DEADLINE_S": ""})
    assert "deadline in 8800 s" in r.stdout
    assert "FS11_SESSION is" not in r.stdout + r.stderr
    assert json.loads((out / "summary.json").read_text())["session"] == "device-leaves"


def test_the_job_script_switches_on_the_session():
    text = fs.Path(t11.JOB).read_text()
    assert re.search(r"\|fs32-pair\|device-leaves\) ;;", text)
    assert subprocess.run(["bash", "-n", str(t11.JOB)]).returncode == 0


def test_the_sessions_selftest_runs_through_the_job(tmp_path):
    r, out = t11._job(tmp_path, deadline=30, grace=1, reserve=5,
                      extra={"FS11_SESSION": "device-leaves", "FS11_SELFTEST": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((out / "summary.json").read_text())["session"] == "device-leaves"
