"""FS31 part A: the topology parsers, the HW2 verdict, the command runner, and the copies without CUDA."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from verification.runs import nvlink_probe as nv

# Recorded `nvidia-smi topo -m` of an 8-GPU HGX H100 box (two NICs, affinity columns, legend).
TOPO_8 = """\
\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tNIC0\tNIC1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tPIX\tSYS\t0-55,112-167\t0\t\tN/A
GPU1\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tPIX\tSYS\t0-55,112-167\t0\t\tN/A
GPU2\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tPIX\tSYS\t0-55,112-167\t0\t\tN/A
GPU3\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tPIX\tSYS\t0-55,112-167\t0\t\tN/A
GPU4\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tSYS\tPIX\t56-111,168-223\t1\t\tN/A
GPU5\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tSYS\tPIX\t56-111,168-223\t1\t\tN/A
GPU6\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tSYS\tPIX\t56-111,168-223\t1\t\tN/A
GPU7\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tSYS\tPIX\t56-111,168-223\t1\t\tN/A
NIC0\tPIX\tPIX\tPIX\tPIX\tSYS\tSYS\tSYS\tSYS\t X \tSYS
NIC1\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\tPIX\tPIX\tSYS\t X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
"""

TOPO_4 = """\
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    0-47    0               N/A
GPU1    NV18     X      NV18    NV18    0-47    0               N/A
GPU2    NV18    NV18     X      NV18    0-47    0               N/A
GPU3    NV18    NV18    NV18     X      0-47    0               N/A

Legend:

  X    = Self
  NV#  = Connection traversing a bonded set of # NVLinks
"""

P2P_4 = """\
 \tGPU0\tGPU1\tGPU2\tGPU3
 GPU0\tX\tOK\tOK\tOK
 GPU1\tOK\tX\tOK\tOK
 GPU2\tOK\tOK\tX\tOK
 GPU3\tOK\tOK\tOK\tX

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
"""


def with_label(text, row, col, label):
    """``text`` with the matrix cell (row, col) replaced, found by splitting that row."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.split()[:1] == [f"GPU{row}"]
             and ln.split()[1:2] != ["GPU1"])
    cells = lines[i].split()
    cells[col + 1] = label
    lines[i] = " ".join(cells)
    return "\n".join(lines)


def pair_rates(n, rate, slow=None):
    out = {f"{i}->{j}": rate for i in range(n) for j in range(n) if i != j}
    if slow:
        out[slow[0]] = slow[1]
    return out


def test_parse_8_gpu_sample_ignores_nics_affinity_and_legend():
    m = nv.parse_gpu_matrix(TOPO_8)
    assert sorted(m) == [f"GPU{i}" for i in range(8)]
    assert all(len(row) == 8 for row in m.values())
    assert m["GPU0"]["GPU1"] == "NV18" and m["GPU7"]["GPU3"] == "NV18"
    assert m["GPU5"]["GPU5"] == "X"


def test_parse_4_gpu_sample_and_p2p():
    m = nv.parse_gpu_matrix(TOPO_4)
    assert m["GPU0"] == {"GPU0": "X", "GPU1": "NV18", "GPU2": "NV18", "GPU3": "NV18"}
    p = nv.parse_gpu_matrix(P2P_4)
    assert p["GPU0"]["GPU1"] == "OK" and p["GPU3"]["GPU3"] == "X"


@pytest.mark.parametrize("text", ["", "nvidia-smi: command not found\n", "GPU0 is fine\n"])
def test_parse_finds_no_matrix(text):
    assert nv.parse_gpu_matrix(text) is None


def test_verdict_passes_on_nv18_and_fast_copies():
    v = nv.hw2_verdict(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                       pair_rates(4, 380.0))
    assert v["pass"] and v["nv_links"] and v["nv18_everywhere"] and v["p2p_read_ok"]
    assert v["peer_copy_min_gbps"] == 380.0 and v["reasons"] == []


@pytest.mark.parametrize("label", ["SYS", "PHB"])
def test_verdict_fails_when_a_pair_is_not_nvlink(label):
    topo = nv.parse_gpu_matrix(with_label(TOPO_4, 1, 2, label))
    v = nv.hw2_verdict(topo, nv.parse_gpu_matrix(P2P_4), pair_rates(4, 380.0))
    assert not v["pass"] and not v["nv_links"] and not v["nv18_everywhere"]
    assert any(label in r and "NVLink" in r for r in v["reasons"])


def test_verdict_fails_when_links_are_fewer_than_18():
    topo = nv.parse_gpu_matrix(with_label(TOPO_4, 0, 3, "NV12"))
    v = nv.hw2_verdict(topo, nv.parse_gpu_matrix(P2P_4), pair_rates(4, 380.0))
    assert v["nv_links"] and not v["nv18_everywhere"] and not v["pass"]
    assert any("NV12" in r for r in v["reasons"])


def test_verdict_fails_on_one_slow_pair():
    v = nv.hw2_verdict(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                       pair_rates(4, 380.0, slow=("2->1", 55.0)))
    assert not v["pass"] and v["nv18_everywhere"] and v["peer_copy_min_gbps"] == 55.0
    assert any("2->1" in r and "55" in r for r in v["reasons"])


def test_verdict_fails_on_p2p_not_ok():
    p2p = nv.parse_gpu_matrix(with_label(P2P_4, 0, 1, "NS"))
    v = nv.hw2_verdict(nv.parse_gpu_matrix(TOPO_4), p2p, pair_rates(4, 380.0))
    assert not v["pass"] and not v["p2p_read_ok"]
    assert any("NS" in r for r in v["reasons"])


def test_verdict_with_nothing_measured_fails_with_three_reasons():
    v = nv.hw2_verdict(None, None, None)
    assert not v["pass"] and v["peer_copy_min_gbps"] is None and len(v["reasons"]) == 3


def test_threshold_is_between_pcie_and_nvlink():
    assert 64 < nv.PEER_COPY_MIN_GBPS < 300


# ---- the command runner -------------------------------------------------------------------


def done(code=0, out="", err=""):
    return lambda argv, **kw: SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_run_command_keeps_raw_text():
    rec = nv.run_command(["nvidia-smi", "topo", "-m"], 5, done(0, "abc\n", "warn\n"))
    assert rec["ok"] and rec["text"] == "abc\nwarn\n" and rec["error"] is None


def test_run_command_records_a_failure():
    rec = nv.run_command(["nvidia-smi"], 5, done(9, "", "no driver"))
    assert not rec["ok"] and rec["returncode"] == 9 and "exit code 9" in rec["error"]
    assert rec["text"] == "no driver"


def test_run_command_records_a_missing_tool_and_a_timeout():
    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 5, output="partial")

    assert "not installed" in nv.run_command(["nvidia-smi"], 5, missing)["error"]
    rec = nv.run_command(["nvidia-smi"], 5, slow)
    assert "timed out" in rec["error"] and rec["text"] == "partial"


def test_report_with_failing_commands_still_assembles(monkeypatch):
    import torch

    def boom(argv, **kw):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nv.shutil, "which", lambda name: None)
    r = nv.build_report(boom)
    assert r["topo"] is None and r["nvbandwidth"] == "unavailable"
    assert not r["hw2"]["pass"] and set(r["commands"]) == set(nv.COMMANDS)


def test_nvbandwidth_runs_both_tests_when_present():
    calls = []

    def run(argv, **kw):
        calls.append(argv[-1])
        return SimpleNamespace(returncode=0, stdout="SUM 1.0", stderr="")

    out = nv.nvbandwidth(run, which=lambda n: "/usr/bin/nvbandwidth")
    assert calls == list(nv.NVBANDWIDTH_TESTS) and out[calls[0]]["text"] == "SUM 1.0"


# ---- the copies without CUDA --------------------------------------------------------------


def test_bandwidth_is_skipped_cleanly_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "skipped" in nv.peer_copy_bandwidth()
    assert nv.peer_copy_bandwidth()["pairs"] == {}
    assert "skipped" in nv.fanout_bandwidth() and "skipped" in nv.pinned_bandwidth()
    assert nv.torch_view() == {"skipped": "torch.cuda.is_available() is false",
                               "device_count": 0}


def test_main_writes_a_report_and_exits_zero(tmp_path, monkeypatch):
    import torch

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nv.shutil, "which", lambda name: None)
    monkeypatch.setattr(nv.subprocess, "run", boom)
    out = tmp_path / "sub" / "nvlink.json"
    assert nv.main(["--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["hw2"]["pass"] is False and doc["version"] == nv.VERSION
