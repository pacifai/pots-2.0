"""The session's exit-hang watch (``watch_process``, ``_run_part``), its ``/proc`` snapshot and the
exit probe (``run_exit_probe``): a pytest part that wrote its junit report and does not exit is
ended with its evidence, and a failed or hung probe run never judges the session."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from verification.runs import fs11_session as fs

ROOT = Path(fs.__file__).resolve().parents[2]
NO_USR1 = not hasattr(signal, "SIGUSR1")

# a process that writes its junit report, registers the stacks dump, then never exits
HANGS = textwrap.dedent("""
    import faulthandler, os, signal, sys, time
    f = open(os.environ["POTS_STACKS_FILE"], "a")
    faulthandler.register(signal.SIGUSR1, file=f, all_threads=True)
    open(sys.argv[1], "w").write("<testsuites/>")
    def waiting_for_nothing():
        time.sleep(300)
    waiting_for_nothing()
""")


def _start(code, *args, env=None):
    return subprocess.Popen([sys.executable, "-c", code, *map(str, args)], env=env,
                            start_new_session=True, stdout=subprocess.DEVNULL)


def test_proc_snapshot_reads_a_fake_proc_and_records_what_it_cannot_read(tmp_path, monkeypatch):
    p = tmp_path / "proc"
    for pid, ppid, pgrp in ((10, 1, 10), (11, 10, 10), (12, 1, 99)):
        task = p / str(pid) / "task" / str(pid)
        task.mkdir(parents=True)
        (p / str(pid) / "stat").write_text(f"{pid} (a b) S {ppid} {pgrp} 0 0\n")
        (p / str(pid) / "cmdline").write_bytes(b"python\0-m\0pytest\0")
        (task / "stat").write_text(f"{pid} (x) D {ppid} {pgrp}\n")
        (task / "comm").write_text("python\n")
        (task / "wchan").write_text("futex_wait_queue\n")  # no syscall, no stack: unreadable
    (p / "11" / "task" / "11" / "stack").write_text("[<0>] cuda_teardown\n")
    monkeypatch.setattr(fs, "PROC", str(p))
    text = fs.proc_snapshot(10)
    assert "process 10 ppid 1 state S cmdline python -m pytest" in text
    assert "process 11 ppid 10" in text and "process 12" not in text
    assert "thread 10 comm python state D wchan futex_wait_queue syscall <FileNotFoundError" in text
    assert "stack [<0>] cuda_teardown" in text


def test_proc_snapshot_without_proc_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "PROC", str(tmp_path / "none"))
    assert "no " in fs.proc_snapshot(1) and "FileNotFoundError" in fs.proc_snapshot(1)


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="needs /proc")
def test_proc_snapshot_of_a_real_group_lists_its_process_and_threads():
    p = _start("import time; time.sleep(30)")
    try:
        time.sleep(0.5)
        text = fs.proc_snapshot(p.pid)
        assert f"process {p.pid} ppid {os.getpid()}" in text and "thread" in text
    finally:
        p.kill()
        p.wait()


@pytest.mark.skipif(NO_USR1, reason="needs SIGUSR1")
def test_a_process_that_outlives_its_report_is_ended_with_its_snapshot_and_stacks(tmp_path):
    junit, stacks, hang = tmp_path / "junit.xml", tmp_path / "stacks.txt", tmp_path / "hang.txt"
    t0 = time.time()
    p = _start(HANGS, junit, env={**os.environ, fs.STACKS_ENV: str(stacks)})
    w = fs.watch_process(p, t0=t0, limit=t0 + 120, junit=junit, grace_s=2, hang_file=hang,
                         stacks=True)
    assert w.status == "exit_hang" and w.junit_to_exit_s >= 2
    assert time.time() - t0 < 40 and p.poll() is not None
    assert "waiting_for_nothing" in stacks.read_text()  # SIGUSR1 reached the leader
    assert "snapshot" in hang.read_text()


def test_a_report_from_before_the_part_started_does_not_count(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuites/>")
    t0 = time.time() + 1  # the file is older than the start
    p = _start("import time; time.sleep(4)")
    w = fs.watch_process(p, t0=t0, limit=t0 + 60, junit=junit, grace_s=1)
    assert w.status == "ok" and w.junit_to_exit_s is None


def test_a_process_that_exits_by_itself_is_ok_or_failed_by_its_code(tmp_path):
    junit = tmp_path / "junit.xml"
    t0 = time.time()
    p = _start("import sys; open(sys.argv[1], 'w').write('x'); sys.exit(3)", junit)
    w = fs.watch_process(p, t0=t0, limit=t0 + 60, junit=junit, grace_s=30)
    assert (w.status, w.rc) == ("failed", 3)
    p = _start("pass")
    assert fs.watch_process(p, t0=t0, limit=t0 + 60).status == "ok"


def test_the_time_limit_leaves_the_same_evidence(tmp_path):
    hang = tmp_path / "hang.txt"
    t0 = time.time()
    p = _start("import time; time.sleep(300)")
    w = fs.watch_process(p, t0=t0, limit=t0 + 1, hang_file=hang, stacks=False)
    assert w.status == "limit" and p.poll() is not None
    assert "snapshot" in hang.read_text()


def _part(tmp_path, code, **kw):
    junit = tmp_path / "parts" / "p" / "junit.xml"
    return fs.Part("p", [sys.executable, "-c", code, str(junit)], kind="pytest",
                   info={"junit": str(junit)}, **kw), junit


def test_run_part_ends_an_exit_hang_and_sets_the_stacks_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "EXIT_HANG_GRACE_S", 2)
    monkeypatch.setattr(fs, "KILL_GRACE_S", 5)
    part, junit = _part(tmp_path, "import sys, time, os\n"
                        "open(sys.argv[1], 'w').write('x')\n"
                        "print('STACKS=' + os.environ['POTS_STACKS_FILE'], flush=True)\n"
                        "time.sleep(300)")
    log = tmp_path / "parts" / "p" / "log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status, rc, wall = fs._run_part(part, dict(os.environ), ROOT, log,
                                    deadline=time.time() + 300, python=sys.executable)
    assert status == "exit_hang" and wall < 60 and time.time() - t0 < 60
    text = log.read_text()
    assert f"STACKS={log.parent / 'stacks.txt'}" in text and "hung at exit" in text
    assert (log.parent / "hang.txt").exists()


def test_run_part_still_kills_at_the_limit_and_the_deadline(tmp_path):
    log = tmp_path / "log.txt"
    part = fs.Part("p", [sys.executable, "-c", "import time; time.sleep(300)"], timeout_s=1)
    assert fs._run_part(part, dict(os.environ), ROOT, log, deadline=time.time() + 300,
                        python=sys.executable)[0] == "killed_timeout"
    part = fs.Part("p", [sys.executable, "-c", "import time; time.sleep(300)"])
    assert fs._run_part(part, dict(os.environ), ROOT, log, deadline=time.time() + 1,
                        python=sys.executable)[0] == "killed_deadline"
    assert (tmp_path / "hang.txt").exists()


# ---- the exit probe ---------------------------------------------------------------------

OK_TEST = "def test_ok():\n    assert True\n"
FAILS = "def test_bad():\n    assert False\n"
HANGING = textwrap.dedent("""
    import threading

    def stuck():
        threading.Event().wait()

    def test_leaves_a_thread():
        threading.Thread(target=stuck, name="never_ends").start()
""")
CONFTEST = ("from tests import exit_guard\n\n\ndef pytest_configure(config):\n"
            "    exit_guard.register_stacks()\n")


def _tree(tmp_path, **files):
    work = tmp_path / "work"
    work.mkdir()
    (work / "conftest.py").write_text(CONFTEST)
    for name, src in files.items():
        (work / f"test_{name}.py").write_text(src)
    return work, [f"test_{n}.py" for n in files]


def _probe(tmp_path, work, mods, monkeypatch, **kw):
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setattr(fs, "KILL_GRACE_S", 5)
    return fs.run_exit_probe(tmp_path / "out", mods, cwd=work, python=sys.executable,
                             budget_s=600, say=lambda s: None, **kw)


def test_the_probe_runs_each_module_alone_then_all_together_when_none_hangs(
        tmp_path, monkeypatch):
    work, mods = _tree(tmp_path, a=OK_TEST, b=FAILS)
    rep = _probe(tmp_path, work, mods, monkeypatch, grace_s=30)
    assert [r["name"] for r in rep["runs"]] == ["test_a", "test_b", "whole_list"]
    a, b, whole = rep["runs"]
    assert (a["status"], a["tests"], a["failures"]) == ("ok", 1, 0)
    assert (b["status"], b["failures"]) == ("failed", 1)  # reported, not judged
    assert whole["modules"] == mods and whole["tests"] == 2 and rep["hung"] == []
    assert rep["whole_list"] is True
    assert json.loads((tmp_path / "out" / fs.EXIT_PROBE_REPORT).read_text()) == rep
    assert fs.exit_probe_lines(rep)[-1] == "FS32 exit probe: 0 of 3 runs hung at exit: none hung"


@pytest.mark.skipif(NO_USR1, reason="needs SIGUSR1")
def test_the_probe_names_the_module_that_cannot_exit_and_skips_the_whole_list(
        tmp_path, monkeypatch):
    work, mods = _tree(tmp_path, a=OK_TEST, stuck=HANGING)
    rep = _probe(tmp_path, work, mods, monkeypatch, grace_s=3)
    assert rep["hung"] == ["test_stuck"] and rep["whole_list"] is False
    assert [r["name"] for r in rep["runs"]] == ["test_a", "test_stuck"]
    hung = rep["runs"][1]
    assert hung["status"] == "exit_hang" and hung["tests"] == 1 and hung["junit_to_exit_s"] >= 3
    assert "never_ends" in Path(hung["snapshot"]).read_text() or "stuck" in Path(
        hung["stacks"]).read_text()
    lines = fs.exit_probe_lines(rep)
    assert len(lines) == 2 and "test_stuck" in lines[0] and "hung at exit" in lines[0]
    assert lines[1] == "FS32 exit probe: 1 of 2 runs hung at exit: test_stuck"


def test_the_probe_starts_no_run_with_the_budget_spent(tmp_path, monkeypatch):
    work, mods = _tree(tmp_path, a=OK_TEST)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    rep = fs.run_exit_probe(tmp_path / "out", mods, cwd=work, python=sys.executable,
                            budget_s=1, say=lambda s: None)
    assert [r["status"] for r in rep["runs"]] == ["not_started", "not_started"]
    last = fs.exit_probe_lines(rep)[-1]
    assert last.endswith("0 of 0 runs hung at exit: none hung (2 not started)")
    assert fs.exit_probe_lines(None) == ["FS32 exit probe: left no report"]
