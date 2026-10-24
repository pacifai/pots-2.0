"""A pytest process that cannot exit ends itself, with the stacks that held it.

On the pair session's pod the fast suite printed its summary and then sat until the part's
time limit, twice, with no Python thread left running (the first guard, a daemon thread's
timer, printed nothing). A daemon thread cannot run once the interpreter is finalizing, nor
while another thread holds the GIL in a C call, so the hang sits where Python code no longer
runs: module teardown, a static destructor or a driver's teardown, or a C call.

:func:`arm` runs last (``pytest_unconfigure``). It stops the children still running, then
starts faulthandler's watchdog, ``dump_traceback_later(grace_s, exit=True)``: a C thread that
needs no GIL and runs through ``atexit`` and module teardown until faulthandler is finalized,
late in ``Py_FinalizeEx``. If the process is still there after ``grace_s`` it writes every
thread's stack to :func:`stacks_file` and exits with status 1: **a hang is a failure**, whatever
the run's own status was. Pytest's ``pytest_unconfigure`` calls ``faulthandler.disable()``, which
does not cancel this timer (the watchdog is separate from the crash handler), and nothing here
sets ``faulthandler_timeout``.

:func:`register_stacks` (``pytest_configure``) lets the runner ask, with ``SIGUSR1``, for
every thread's stack at any moment, a hang in the middle of a run included.

The file is named by ``POTS_STACKS_FILE``, opened for append once and kept open for the
process's life; without it, ``sys.__stderr__``. During a run pytest captures fd 2, so a dump
to stderr in mid-run is lost: the session runner always sets the variable. The grace is
:data:`GRACE_S`, or ``POTS_EXIT_GRACE_S`` when set (the tests use it).
"""

from __future__ import annotations

import faulthandler
import multiprocessing
import os
import signal
import sys
from typing import IO

GRACE_S = 120.0
STACKS_ENV = "POTS_STACKS_FILE"
GRACE_ENV = "POTS_EXIT_GRACE_S"

_FILE: IO[str] | None = None


def stacks_file() -> IO[str]:
    """The file the stack dumps go to (:data:`STACKS_ENV`, appended to, opened once)."""
    global _FILE
    if _FILE is None:
        path = os.environ.get(STACKS_ENV)
        _FILE = open(path, "a") if path else sys.__stderr__
    return _FILE


def register_stacks() -> None:
    """``SIGUSR1`` writes every thread's stack to :func:`stacks_file` (where the platform
    has the signal); the process goes on."""
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=stacks_file(), all_threads=True)


def stop_children() -> list[str]:
    """Kill the child processes still alive (a spawned producer a failed test left waiting);
    their names, for the log."""
    left = []
    for p in multiprocessing.active_children():
        left.append(f"{p.name} (pid {p.pid})")
        p.kill()
        p.join(10)
    return left


def arm(grace_s: float | None = None) -> None:
    """Stop the children and start the watchdog (module docstring)."""
    if grace_s is None:
        grace_s = float(os.environ.get(GRACE_ENV, GRACE_S))
    left = stop_children()
    if left:
        print(f"[exit_guard] killed children still running at exit: {', '.join(left)}",
              file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(grace_s, exit=True, file=stacks_file())
