"""A pytest process that cannot exit ends itself, with the stacks that held it.

On the pair session's pod the fast suite printed its summary and then sat for 20 minutes: the
interpreter's exit joins every non-daemon thread and child process, and one of them never ended.
:func:`arm` runs last (``pytest_unconfigure``): it stops the children still running, and starts a
daemon timer that, if the interpreter is still there after ``grace_s``, writes every thread's
stack to stderr and leaves with the run's exit status.
"""

from __future__ import annotations

import faulthandler
import multiprocessing
import os
import sys
import threading

GRACE_S = 120.0


def stop_children() -> list[str]:
    """Kill the child processes still alive (a spawned producer a failed test left waiting);
    their names, for the log."""
    left = []
    for p in multiprocessing.active_children():
        left.append(f"{p.name} (pid {p.pid})")
        p.kill()
        p.join(10)
    return left


def arm(exitstatus: int, grace_s: float = GRACE_S) -> None:
    left = stop_children()
    if left:
        print(f"[exit_guard] killed children still running at exit: {', '.join(left)}",
              file=sys.stderr, flush=True)

    def watch() -> None:
        threading.Event().wait(grace_s)
        print(f"[exit_guard] still alive {grace_s:.0f} s after the run; every stack:",
              file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(int(exitstatus))

    threading.Thread(target=watch, daemon=True, name="exit_guard").start()
