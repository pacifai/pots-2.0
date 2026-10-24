"""``fs11_setup.sh``: the pod's venv at the Mac's Python, on stub ``python3`` and ``uv``."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from tests.verification.runs.session_env import clean_environ

SCRIPT = Path(__file__).resolve().parents[3] / "verification" / "runs" / "fs11_setup.sh"

STUB_PY3 = f"""#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "python3 $*" >> "$STUB_LOG"; exit 0; fi
exec {sys.executable} "$@"
"""

# `uv venv` makes the venv's python: a script that prints what the real one would.
STUB_UV = """#!/bin/sh
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = "venv" ]; then
    for last in "$@"; do :; done
    mkdir -p "$last/bin"
    printf '#!/bin/sh\\necho "fs11_setup: venv Python stub torch stub"\\n' > "$last/bin/python"
    chmod +x "$last/bin/python"
fi
"""


def _exe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path, manifest, *, with_requirements=True):
    up = tmp_path / "up"
    (up / "repo").mkdir(parents=True)
    if manifest is not None:
        (up / "manifest.json").write_text(manifest if isinstance(manifest, str)
                                          else json.dumps(manifest))
    if with_requirements:
        (up / "requirements-fs11.txt").write_text("torch==2.9.1\n")
    _exe(tmp_path / "bin" / "python3", STUB_PY3)
    _exe(tmp_path / "uvnew" / "bin" / "uv", STUB_UV)
    log = tmp_path / "stub.log"
    env = clean_environ({"STUB_LOG": str(log), "FS11_SETUP_PYTHON3": str(tmp_path / "bin/python3"),
                         "FS11_SETUP_UVDIR": str(tmp_path / "uvnew"),
                         "FS11_SETUP_VENV": str(tmp_path / "venv")})
    r = subprocess.run(["bash", str(SCRIPT), str(up)], env=env, capture_output=True, text=True,
                       timeout=60)
    return r, up, (log.read_text().splitlines() if log.exists() else [])


def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_the_manifests_version_reaches_uv(tmp_path):
    r, up, calls = _run(tmp_path, {"python_version": "3.14.7"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"python3 -m pip install -q --target {tmp_path}/uvnew uv" in calls
    assert "uv python install 3.14.7" in calls
    assert f"uv venv --clear --python 3.14.7 {tmp_path}/venv" in calls
    assert any(c.startswith("uv pip install --python") and c.endswith("requirements-fs11.txt")
               for c in calls)
    assert calls.index("uv python install 3.14.7") < next(
        i for i, c in enumerate(calls) if c.startswith("uv venv"))
    assert os.readlink(up / "repo" / ".venv") == str(tmp_path / "venv")
    assert "venv Python" in r.stdout


def test_a_rerun_replaces_the_link_and_the_venv(tmp_path):
    r, up, _ = _run(tmp_path, {"python_version": "3.14.7"})
    assert r.returncode == 0
    r2 = subprocess.run(["bash", str(SCRIPT), str(up)], capture_output=True, text=True,
                        env=clean_environ({"STUB_LOG": str(tmp_path / "stub.log"),
                                           "FS11_SETUP_PYTHON3": str(tmp_path / "bin/python3"),
                                           "FS11_SETUP_UVDIR": str(tmp_path / "uvnew"),
                                           "FS11_SETUP_VENV": str(tmp_path / "venv")}))
    assert r2.returncode == 0, r2.stderr
    assert os.readlink(up / "repo" / ".venv") == str(tmp_path / "venv")


def test_a_manifest_without_a_version_stops_before_anything_installs(tmp_path):
    r, _, calls = _run(tmp_path, {"files": {}})
    assert r.returncode != 0 and calls == []
    assert "records no python_version" in r.stderr and "cannot read" in r.stderr


def test_an_unreadable_manifest_and_missing_files_stop_with_a_message(tmp_path):
    r, _, calls = _run(tmp_path, "{")
    assert r.returncode != 0 and calls == [] and "unreadable" in r.stderr
    r, _, calls = _run(tmp_path / "b", None)
    assert r.returncode != 0 and calls == [] and "manifest.json is missing" in r.stderr
    r, _, calls = _run(tmp_path / "c", {"python_version": "3.14.7"}, with_requirements=False)
    assert r.returncode != 0 and calls == [] and "requirements-fs11.txt is missing" in r.stderr


def test_a_failing_uv_step_is_named(tmp_path):
    r, _, _ = _run(tmp_path, {"python_version": "3.14.7"})
    assert r.returncode == 0
    failing = "#!/bin/sh\n[ \"$1 $2\" = 'python install' ] && exit 1\n"
    _exe(tmp_path / "uvnew" / "bin" / "uv", failing)
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "up")], capture_output=True, text=True,
                       env=clean_environ({"FS11_SETUP_PYTHON3": str(tmp_path / "bin/python3"),
                                          "FS11_SETUP_UVDIR": str(tmp_path / "uvnew"),
                                          "STUB_LOG": str(tmp_path / "stub.log"),
                                          "FS11_SETUP_VENV": str(tmp_path / "venv")}))
    assert r.returncode == 1 and "uv could not install Python 3.14.7" in r.stderr
