"""``tests.exit_guard``: a run that cannot exit ends itself and says what held it. The tests
run real subprocesses: the guard's whole point is what happens at interpreter exit."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HANG_PLUGIN = textwrap.dedent("""
    import threading
    import pytest

    def stuck_forever():
        threading.Event().wait()

    @pytest.hookimpl(tryfirst=True)
    def pytest_unconfigure(config):
        # before the guard's own hook (the real tests/conftest.py): a non-daemon thread that
        # never ends, so the interpreter's exit joins it for good
        threading.Thread(target=stuck_forever, name="never_ends").start()
""")

CHILD = textwrap.dedent("""
    import multiprocessing as mp, time
    from tests import exit_guard
    def wait():
        time.sleep(600)
    if __name__ == "__main__":
        p = mp.get_context("spawn").Process(target=wait, name="left_over")
        p.start()
        exit_guard.arm(grace_s=60.0)
        print("main done", flush=True)
""")

SLEEPER = textwrap.dedent("""
    import time
    from tests import exit_guard
    exit_guard.register_stacks()
    def sleeping_here():
        print("ready", flush=True)
        time.sleep(60)
    sleeping_here()
""")


def _env(tmp_path, **extra):
    stacks = tmp_path / "stacks.txt"
    path = f"{tmp_path}{os.pathsep}{ROOT}"
    env = {**os.environ, "POTS_STACKS_FILE": str(stacks), "PYTHONPATH": path, **extra}
    return env, stacks


def _pytest_run(tmp_path, with_hang, grace="2"):
    (tmp_path / "hang_plugin.py").write_text(HANG_PLUGIN)
    (tmp_path / "test_one.py").write_text("def test_ok():\n    assert True\n")
    env, stacks = _env(tmp_path, POTS_EXIT_GRACE_S=grace)
    plugins = ["-p", "tests.conftest", *(["-p", "hang_plugin"] if with_hang else [])]
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *plugins,
                        "--rootdir", str(tmp_path), str(tmp_path / "test_one.py")],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    return r, time.time() - t0, stacks


def test_a_pytest_run_held_by_a_non_daemon_thread_ends_with_every_stack(tmp_path):
    r, wall, stacks = _pytest_run(tmp_path, with_hang=True)
    assert "1 passed" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
    assert r.returncode == 1, "a hang is a failure, though every test passed"
    assert wall < 60
    text = stacks.read_text()
    assert "Timeout" in text and "stuck_forever" in text, text[-800:]


def test_a_pytest_run_that_exits_leaves_no_dump(tmp_path):
    r, wall, stacks = _pytest_run(tmp_path, with_hang=False, grace="60")
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
    assert wall < 50, "the watchdog's timer must not hold a run that ends"
    assert not stacks.exists() or stacks.read_text() == ""


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1")
def test_sigusr1_writes_every_stack_to_the_file_and_the_process_goes_on(tmp_path):
    env, stacks = _env(tmp_path)
    p = subprocess.Popen([sys.executable, "-c", SLEEPER], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "ready"
        p.send_signal(signal.SIGUSR1)
        end = time.time() + 20
        while time.time() < end and "sleeping_here" not in (
                stacks.read_text() if stacks.exists() else ""):
            time.sleep(0.2)
        assert "sleeping_here" in stacks.read_text()
        assert p.poll() is None, "the signal must not end the process"
    finally:
        p.kill()
        p.wait()


def test_a_child_left_running_is_killed_at_exit(tmp_path):
    env, _ = _env(tmp_path)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert time.time() - t0 < 60, "the interpreter's exit must not wait for the left-over child"
    assert "left_over" in r.stderr
