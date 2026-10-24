"""FS35 (I4): the sampler on every NVML GPU of the host, for about a second. Skips cleanly
without psutil, NVML or a GPU; the session runs it on the pod."""

import json
import os
import shutil
import subprocess
import threading

import pytest

from setup import sampler

pytest.importorskip("psutil")
pynvml = pytest.importorskip("pynvml")


def _nvml_count() -> int:
    try:
        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount()
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 (no driver, no NVML library)
        return 0


N = _nvml_count()
pytestmark = pytest.mark.skipif(N == 0, reason="no NVML GPU on this host")


def test_the_header_lists_every_gpu_and_every_row_has_one_value_each(tmp_path):
    out = tmp_path / "s.jsonl"
    stop = threading.Event()
    threading.Timer(1.2, stop.set).start()
    assert sampler.run(os.getpid(), out, 0.1, None, stop) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    head, rows = lines[0], [r for r in lines if "t" in r]
    assert head["version"] == 2 and head["nvml"] is True
    assert [g["index"] for g in head["gpus"]] == list(range(N))
    assert all(g["uuid"] and g["name"] and g["pci_bus_id"] for g in head["gpus"])
    assert len({g["uuid"] for g in head["gpus"]}) == N
    assert len(rows) >= 5
    for f in sampler.GPU_FIELDS:
        assert all(isinstance(r[f], list) and len(r[f]) == N for r in rows), f
    assert all(v is not None for v in rows[-1]["g_util"])
    if shutil.which("nvidia-smi"):
        listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                timeout=30).stdout
        smi = [ln for ln in listed.splitlines() if ln.startswith("GPU ")]
        assert len(smi) == len(head["gpus"])
        for g in head["gpus"]:
            assert g["uuid"] in listed
