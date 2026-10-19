"""setup/sampler.py (FS21): the file's shape, the process tree, stopping, and failing readers."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from setup import sampler

pytest.importorskip("psutil")

BUSY = "import time\nt = time.time()\nwhile time.time() - t < 2: pass"


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _samples(path: Path) -> list[dict]:
    return [r for r in _lines(path) if "t" in r]


def _popen(*args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "setup.sampler", *args])


def test_wrap_a_busy_loop(tmp_path):
    out = tmp_path / "s.jsonl"
    rc = _popen("--out", str(out), "--wrap", "--", sys.executable, "-c", BUSY).wait(timeout=30)
    assert rc == 0
    head, *_ = _lines(out)
    rows = _samples(out)
    assert head["kind"] == "header" and head["version"] == 1 and head["fields"] == sampler.FIELDS
    assert head["interval_s"] == 0.1 and head["gpu"] is None and head["nvml"] is False
    assert 15 <= len(rows) <= 22
    assert sum(r["c_tree"] for r in rows) / len(rows) > 0.5
    assert all(set(r) == set(sampler.FIELDS) for r in rows)
    assert all(r["rss"] > 0 and r["threads"] >= 1 and r["ram_avail"] > 0 for r in rows)
    assert all(b["t"] > a["t"] for a, b in zip(rows, rows[1:]))
    assert all(r["g_util"] is None and r["g_sm_active"] is None for r in rows)


def test_the_sampler_costs_under_two_percent_of_a_core(tmp_path):
    out = tmp_path / "s.jsonl"
    _popen("--out", str(out), "--wrap", "--", sys.executable, "-c", BUSY).wait(timeout=30)
    rows = _samples(out)[1:]  # the first row includes the sampler's start-up
    assert sum(r["c_self"] for r in rows) / len(rows) < 0.02


def test_sigterm_ends_the_sampler_within_two_seconds(tmp_path):
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        p = _popen("--pid", str(target.pid), "--out", str(tmp_path / "s.jsonl"))
        time.sleep(1)
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=5) == 0
        assert time.monotonic() - t0 < 2
        assert target.poll() is None  # --pid never touches the target
    finally:
        target.kill()
        target.wait()


def test_the_sampler_exits_when_the_root_dies(tmp_path):
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    p = _popen("--pid", str(target.pid), "--out", str(tmp_path / "s.jsonl"))
    try:
        time.sleep(1)
        target.kill()  # a zombie until reaped, which the sampler counts as gone
        t0 = time.monotonic()
        assert p.wait(timeout=5) == 0
        assert time.monotonic() - t0 < 2
    finally:
        target.wait()
        if p.poll() is None:
            p.kill()


def test_start_sampler_survives_a_group_kill_and_stops(tmp_path):
    out = tmp_path / "s.jsonl"
    part = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)
    h = sampler.start_sampler(part.pid, out, cwd=Path(sampler.__file__).parents[1])
    time.sleep(1.5)
    assert h.proc.poll() is None
    part.kill()
    part.wait()
    assert h.proc.wait(timeout=5) == 0
    h.stop()  # already gone: no error
    assert len(_samples(out)) > 3


def test_the_sampler_ends_with_a_part_that_exits_at_once(tmp_path):
    out = tmp_path / "s.jsonl"
    part = subprocess.Popen([sys.executable, "-c", "pass"])
    part.wait()
    assert sampler.run(part.pid, out) == 0  # the root is gone: a header, maybe no samples
    assert _lines(out)[0]["kind"] == "header"


def test_the_pace_follows_the_deadline(tmp_path):
    out = tmp_path / "s.jsonl"
    stop = threading.Event()
    t = threading.Thread(target=sampler.run, args=(os.getpid(), out, 0.1, 0, stop))
    t.start()
    time.sleep(2.05)
    stop.set()
    t.join(timeout=3)
    ts = [r["t"] for r in _samples(out)]
    assert 18 <= len(ts) <= 21
    gaps = [(b - a) / 1e9 for a, b in zip(ts, ts[1:])]
    assert abs(sum(gaps) / len(gaps) - 0.1) < 0.005  # no drift


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_nvml(*, gpm: bool, pids=(1,)) -> types.ModuleType:
    m = types.ModuleType("pynvml")
    m.NVML_CLOCK_SM, m.NVML_CLOCK_MEM, m.NVML_TEMPERATURE_GPU = 1, 2, 0
    m.NVML_PCIE_UTIL_TX_BYTES, m.NVML_PCIE_UTIL_RX_BYTES = 0, 1
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: "h"
    m.nvmlDeviceGetName = lambda h: b"Fake GPU"
    m.nvmlDeviceGetUUID = lambda h: "GPU-fake"
    m.nvmlDeviceGetUtilizationRates = lambda h: _Obj(gpu=80, memory=30)
    m.nvmlDeviceGetMemoryInfo = lambda h: _Obj(used=123456)
    m.nvmlDeviceGetClockInfo = lambda h, c: 1700 if c == 1 else 2600
    m.nvmlDeviceGetPowerUsage = lambda h: 250000
    m.nvmlDeviceGetTemperature = lambda h, k: 55
    m.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 4
    m.nvmlDeviceGetPcieThroughput = lambda h, c: 10 if c == 0 else 20
    m.nvmlDeviceGetComputeRunningProcesses = lambda h: [_Obj(pid=p) for p in pids]
    names = ("SM_UTIL", "SM_OCCUPANCY", "ANY_TENSOR_UTIL", "DRAM_BW_UTIL", "PCIE_TX_PER_SEC",
             "PCIE_RX_PER_SEC")
    for i, n in enumerate(names):
        setattr(m, f"NVML_GPM_METRIC_{n}", i + 1)
    if gpm:
        m.NVML_GPM_METRICS_GET_VERSION = 1
        m.nvmlGpmQueryDeviceSupport = lambda h: _Obj(isSupportedDevice=1)
        m.nvmlGpmSampleAlloc = lambda: object()
        m.nvmlGpmSampleGet = lambda h, s: None
        m.nvmlGpmSampleFree = lambda s: None
        m.c_nvmlGpmMetricsGet_t = lambda: _Obj(metrics=[_Obj(metricId=0, value=0.0)
                                                         for _ in names])

        def get(q):  # each metric's value is its id, so the scaling is visible
            for x in q.metrics[:q.numMetrics]:
                x.value = float(x.metricId * 10)
        m.nvmlGpmMetricsGet = get
    else:
        m.nvmlGpmQueryDeviceSupport = lambda h: _Obj(isSupportedDevice=0)
    return m


def _run_briefly(tmp_path, seconds=0.6) -> list[dict]:
    out = tmp_path / "s.jsonl"
    stop = threading.Event()
    threading.Timer(seconds, stop.set).start()
    sampler.run(os.getpid(), out, 0.1, 0, stop)
    return _lines(out)


def test_a_fake_nvml_fills_the_gpu_fields(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(gpm=False))
    head, *rest = _run_briefly(tmp_path)
    rows = [r for r in rest if "t" in r]
    assert head["nvml"] is True and head["gpu"]["name"] == "Fake GPU"
    assert head["gpu"]["uuid"] == "GPU-fake" and head["gpm"] is False
    assert "doesn't support GPM" in head["gpm_error"] and head["pcie_source"] == "nvml"
    r = rows[-1]
    assert (r["g_util"], r["g_mem_util"], r["g_mem_used"]) == (80, 30, 123456)
    assert (r["g_sm_mhz"], r["g_mem_mhz"], r["g_power_w"], r["g_temp_c"]) == (1700, 2600, 250, 55)
    assert (r["g_throttle"], r["g_pcie_tx"], r["g_pcie_rx"]) == (4, 10240, 20480)
    assert r["g_other_procs"] == 1 and r["g_sm_active"] is None


def test_a_fake_gpm_fills_the_gpm_fields_and_pcie(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(gpm=True))
    head, *rest = _run_briefly(tmp_path)
    r = [r for r in rest if "t" in r][-1]
    assert head["gpm"] is True and head["gpm_error"] is None and head["pcie_source"] == "gpm"
    assert r["g_sm_active"] == pytest.approx(0.1) and r["g_dram_bw"] == pytest.approx(0.4)
    assert r["g_pcie_tx"] == 50 * (1 << 20) and r["g_pcie_rx"] == 60 * (1 << 20)


def test_a_failing_nvml_call_gives_null_and_one_error(tmp_path, monkeypatch):
    m = _fake_nvml(gpm=False)

    def boom(h):
        raise RuntimeError("no power reading")
    m.nvmlDeviceGetPowerUsage = boom
    monkeypatch.setitem(sys.modules, "pynvml", m)
    lines = _run_briefly(tmp_path)
    rows = [r for r in lines if "t" in r]
    assert all(r["g_power_w"] is None for r in rows) and rows[-1]["g_util"] == 80
    # the baseline sample runs before the header, so the first failure lands in its errors
    errors = [e for e in lines[0]["errors"] if e["group"] != "nvml"] \
        + [r for r in lines if r.get("kind") == "error"]
    assert [e["group"] for e in errors] == ["g_power_w"] and "no power" in errors[0]["error"]


def test_a_psutil_that_raises_on_one_call_gives_nulls(tmp_path, monkeypatch):
    import psutil

    def boom(*a, **k):
        raise RuntimeError("no memory counters")
    monkeypatch.setattr(psutil, "virtual_memory", boom)
    lines = _run_briefly(tmp_path)
    rows = [r for r in lines if "t" in r]
    assert rows and all(r["ram_used"] is None and r["ram_avail"] is None for r in rows)
    assert rows[-1]["rss"] > 0 and rows[-1]["c_busy"] is not None
    assert [e["group"] for e in lines[0]["errors"] if e["group"] != "nvml"] == ["ram"]


def test_without_psutil_the_file_holds_a_header_and_its_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    out = tmp_path / "s.jsonl"
    assert sampler.run(1, out) == 0
    (head,) = _lines(out)
    assert head["kind"] == "header" and head["errors"][0]["group"] == "psutil"


def test_an_unwritable_out_returns_one(tmp_path):
    assert sampler.run(1, tmp_path / "missing" / "s.jsonl") == 1


def test_importing_the_sampler_loads_neither_psutil_nor_pynvml():
    code = ("import sys, setup.sampler; "
            "assert not {'psutil', 'pynvml', 'torch'} & set(sys.modules), sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(sampler.__file__).parents[1])
