"""FS32 round 4 tester: ``setup.sampler --no-gpu`` end to end, with a fake NVML as the witness."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[2]
BUSY = "import time\nt = time.time()\nwhile time.time() - t < 1: pass"

FAKE_NVML = """
import os
def nvmlInit():
    with open(os.environ["FAKE_NVML_MARK"], "a") as f:
        f.write("init\\n")
    raise RuntimeError("fake nvml: no driver")
"""


def _env(tmp_path: Path) -> dict:
    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "pynvml.py").write_text(FAKE_NVML)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(fake), str(ROOT)])
    env["FAKE_NVML_MARK"] = str(tmp_path / "mark")
    return env


def _run(tmp_path, *flags, wrap=True):
    out = tmp_path / "s.jsonl"
    if wrap:
        argv = ["--out", str(out), *flags, "--wrap", "--", sys.executable, "-c", BUSY]
        subprocess.run([sys.executable, "-m", "setup.sampler", *argv], env=_env(tmp_path),
                       cwd=ROOT, check=True, timeout=60)
    else:
        child = subprocess.Popen([sys.executable, "-c", BUSY])
        try:
            subprocess.run([sys.executable, "-m", "setup.sampler", "--pid", str(child.pid),
                            "--out", str(out), *flags], env=_env(tmp_path), cwd=ROOT,
                           check=True, timeout=60)
        finally:
            child.wait(timeout=30)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    return lines[0], [r for r in lines if "t" in r]


def _inits(tmp_path) -> int:
    m = tmp_path / "mark"
    return len(m.read_text().split()) if m.exists() else 0


@pytest.mark.parametrize("wrap", [True, False], ids=["wrap", "pid"])
def test_no_gpu_never_touches_nvml(tmp_path, wrap):
    head, rows = _run(tmp_path, "--no-gpu", wrap=wrap)
    assert _inits(tmp_path) == 0
    assert head["nvml"] is False and head["gpm"] is False and head["gpu"] is None
    assert "--no-gpu" in head["gpm_error"]
    assert not any(e.get("group") == "nvml" for e in head["errors"])
    assert rows and all(r["g_util"] is None for r in rows)


def test_without_the_flag_nvml_is_still_started_and_its_failure_reported(tmp_path):
    head, rows = _run(tmp_path)
    assert _inits(tmp_path) == 1
    assert head["nvml"] is False and "NVML isn't available" in head["gpm_error"]
    assert any(e.get("group") == "nvml" and "fake nvml" in e["error"] for e in head["errors"])
    assert rows  # the failed reader costs the run nothing else


def test_the_flag_changes_no_other_header_field(tmp_path):
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    a, _ = _run(a_dir, "--no-gpu")
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    b, _ = _run(b_dir)
    drop = {"mono_ns", "wall_unix_s", "pid", "errors", "gpm_error"}
    assert {k: v for k, v in a.items() if k not in drop} == {
        k: v for k, v in b.items() if k not in drop}
