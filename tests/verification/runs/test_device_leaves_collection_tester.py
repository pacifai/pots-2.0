"""FM9 follow-up tester: the gate's expected names, as the pod's collection gives them, against
the names the real junit report gives; and the gate failing closed when the collection fails."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.verification.runs.test_device_leaves_session import _parts, _tests_part
from verification.runs import fs11_session as fs

FILE = fs.DEVICE_TESTS_FILE
MOD = FILE.removesuffix(".py").replace("/", ".")


@pytest.fixture(autouse=True)
def _fresh_cache():
    saved = dict(fs._collected)
    fs._collected.clear()
    yield
    fs._collected.clear()
    fs._collected.update(saved)


def _fake_run(stdout="", returncode=0, raises=None, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append((cmd, kw))
        if raises is not None:
            raise raises
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)
    return run


def test_collected_names_equal_the_names_a_real_junit_report_uses(tmp_path):
    """The real file, run for real (skipped here without CUDA, a pod runs it): the junit
    ``classname::name`` of every case is exactly a collected name, parametrize ids included."""
    junit = tmp_path / "junit.xml"
    root = fs._REPO_ROOT
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={junit}", FILE], cwd=root, capture_output=True, text=True)
    cases = fs._device_tests_extras(junit)["cases"]
    names, err = fs.device_tests_expected()
    assert err == "" and names
    assert sorted(names) == sorted(cases)
    assert any("[fp32-1]" in n for n in names)


def test_warnings_and_a_summary_in_the_collect_output_are_not_names(monkeypatch):
    out = (f"{FILE}::test_a[fp32-1]\n{FILE}::test_b\n\n"
           f"{FILE}:5\n  /x/{FILE}:5: PytestWarning: something\n    warnings.warn(x)\n"
           "-- Docs: https://docs.pytest.org/\n2 tests collected in 0.1s\n")
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(out))
    names, err = fs.device_tests_expected()
    assert err == "" and names == [f"{MOD}::test_a[fp32-1]", f"{MOD}::test_b"]


def test_a_collection_of_zero_names_fails_the_gate_with_the_exit_status(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run("no tests collected in 0.01s\n", 5))
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is False
    assert any("could not be collected" in r and "found 0 tests" in r for r in g["reasons"])


def test_zero_names_with_exit_zero_still_fails_the_gate(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run("", 0))
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is False
    assert any("could not be collected" in r for r in g["reasons"])


def test_names_found_but_a_nonzero_exit_fails_the_gate(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(f"{FILE}::test_a\n", 2))
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is False
    assert any("could not be collected" in r and "exited 2" in r for r in g["reasons"])


def test_the_collect_command_is_run_from_the_repo_root_with_no_cacheprovider(monkeypatch):
    seen = []
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(f"{FILE}::test_a\n", 0, seen=seen))
    fs.device_tests_expected()
    cmd, kw = seen[0]
    assert "--collect-only" in cmd and "no:cacheprovider" in cmd and cmd[-1] == FILE
    assert kw["cwd"] == fs._REPO_ROOT and kw["timeout"] > 0


def test_a_collect_that_times_out_fails_the_gate_instead_of_raising(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(
        raises=subprocess.TimeoutExpired(["pytest"], 300)))
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is False


def test_the_collect_subprocess_gets_an_environment_without_secrets(monkeypatch):
    seen = []
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(f"{FILE}::test_a\n", 0, seen=seen))
    fs.device_tests_expected()
    env = seen[0][1].get("env")
    assert env is not None and not (set(env) & set(fs.SECRETS))


def test_a_class_based_test_is_named_as_the_junit_report_names_it(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(f"{FILE}::TestX::test_a\n", 0))
    names, _ = fs.device_tests_expected()
    assert names == [f"{MOD}.TestX::test_a"]


def test_the_gate_passes_with_every_collected_name_in_the_report(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(f"{FILE}::test_a\n{FILE}::test_b[x]\n"))
    cases = {f"{MOD}::test_a": "passed", f"{MOD}::test_b[x]": "passed",
             "tests.verification.prover.test_capture_cuda::"
             "test_gpu_memory_does_not_hold_the_transcript": "passed"}
    assert fs.device_leaves_gate(_parts(tests=_tests_part(cases=cases)))["pass"] is True
    del cases[f"{MOD}::test_b[x]"]
    g = fs.device_leaves_gate(_parts(tests=_tests_part(cases=cases)))
    assert g["pass"] is False
    assert f"{MOD}::test_b[x]: expected but not in the junit report" in g["reasons"]


def test_a_failed_collect_runs_once_and_the_env_is_offline_without_stdin(monkeypatch):
    seen = []
    monkeypatch.setenv("RUNPOD_API_KEY", "x")
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(
        raises=subprocess.TimeoutExpired(["pytest"], 300), seen=seen))
    first, second = fs.device_tests_expected(), fs.device_tests_expected()
    assert first == second and first[0] == [] and "timed out" in first[1] and len(seen) == 1
    kw = seen[0][1]
    assert kw["stdin"] == subprocess.DEVNULL and "RUNPOD_API_KEY" not in kw["env"]
    assert kw["env"]["HF_HUB_OFFLINE"] == "1" and kw["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_a_collect_that_cannot_start_fails_the_gate(monkeypatch):
    monkeypatch.setattr(fs.subprocess, "run", _fake_run(raises=OSError("no python")))
    g = fs.device_leaves_gate(_parts())
    assert g["pass"] is False and any("could not start" in r for r in g["reasons"])
