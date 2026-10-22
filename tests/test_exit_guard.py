"""``tests.exit_guard``: a run that cannot exit ends itself and says what held it."""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent("""
    import threading, time
    from tests import exit_guard
    def stuck():
        threading.Event().wait()
    threading.Thread(target=stuck, name="never_ends").start()  # non-daemon: exit joins it
    exit_guard.arm(3, grace_s=1.0)
    print("main done", flush=True)
""")

CHILD = textwrap.dedent("""
    import multiprocessing as mp, time
    from tests import exit_guard
    def wait():
        time.sleep(600)
    if True:
        p = mp.get_context("spawn").Process(target=wait, name="left_over")
        p.start()
        exit_guard.arm(0, grace_s=60.0)
        print("main done", flush=True)
""")


def _run(code, limit):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=limit)
    return r, time.time() - t0


def test_a_run_held_by_a_non_daemon_thread_ends_with_every_stack():
    r, wall = _run(DRIVER, 60)
    assert r.returncode == 3, r.stderr[-500:]
    assert wall < 30
    assert "still alive" in r.stderr
    assert "stuck" in r.stderr, "the dump names the function the thread waits in"


def test_a_child_left_running_is_killed_at_exit():
    r, wall = _run(CHILD, 120)
    assert r.returncode == 0, r.stderr[-500:]
    assert wall < 60, "the interpreter's exit must not wait for the left-over child"
    assert "left_over" in r.stderr
