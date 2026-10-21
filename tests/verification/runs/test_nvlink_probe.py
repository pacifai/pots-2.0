"""FS31 part A: the topology parsers, the HW2 verdict, the command runner, and the copies without
CUDA."""

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


def right_bytes(topo, wrong=()):
    """``peer_bytes`` for a parsed topology: every ordered pair landed the right bytes, but the
    ``wrong`` ones."""
    return {f"{a[3:]}->{b[3:]}": f"{a[3:]}->{b[3:]}" not in wrong
            for a in topo for b in topo if a != b}


def hw2(topo, p2p, rates, n_expected=None, failures=None, peer_bytes="right", wrong=()):
    """``hw2_verdict`` with the peer-copy byte check passed for every pair of the topology unless
    told otherwise (``peer_bytes=None`` is a probe that left no byte result)."""
    if isinstance(peer_bytes, str):
        peer_bytes = right_bytes(topo, wrong) if topo else None
    return nv.hw2_verdict(topo, p2p, rates, n_expected, failures, peer_bytes)


@pytest.mark.parametrize("text", ["", "nvidia-smi: command not found\n", "GPU0 is fine\n"])
def test_parse_finds_no_matrix(text):
    assert nv.parse_gpu_matrix(text) is None


def test_verdict_passes_on_nv18_and_fast_copies():
    v = hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                       pair_rates(4, 380.0))
    assert v["pass"] and v["nv_links"] and v["nv18_everywhere"] and v["p2p_read_ok"]
    assert v["peer_copy_min_gbps"] == 380.0 and v["reasons"] == [] and v["verdict"] == "pass"
    assert (v["n_cards"], v["expected_cards"], v["gate"]) == (4, 4, True)


@pytest.mark.parametrize("label", ["SYS", "PHB"])
def test_verdict_fails_when_a_pair_is_not_nvlink(label):
    topo = nv.parse_gpu_matrix(with_label(TOPO_4, 1, 2, label))
    v = hw2(topo, nv.parse_gpu_matrix(P2P_4), pair_rates(4, 380.0))
    assert not v["pass"] and not v["nv_links"] and not v["nv18_everywhere"]
    assert any(label in r and "NVLink" in r for r in v["reasons"])


def test_verdict_fails_when_links_are_fewer_than_18():
    topo = nv.parse_gpu_matrix(with_label(TOPO_4, 0, 3, "NV12"))
    v = hw2(topo, nv.parse_gpu_matrix(P2P_4), pair_rates(4, 380.0))
    assert v["nv_links"] and not v["nv18_everywhere"] and not v["pass"]
    assert any("NV12" in r for r in v["reasons"])


def test_verdict_fails_on_one_slow_pair():
    v = hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                       pair_rates(4, 380.0, slow=("2->1", 55.0)))
    assert not v["pass"] and v["nv18_everywhere"] and v["peer_copy_min_gbps"] == 55.0
    assert any("2->1" in r and "55" in r for r in v["reasons"])


def test_verdict_fails_on_p2p_not_ok():
    p2p = nv.parse_gpu_matrix(with_label(P2P_4, 0, 1, "NS"))
    v = hw2(nv.parse_gpu_matrix(TOPO_4), p2p, pair_rates(4, 380.0))
    assert not v["pass"] and not v["p2p_read_ok"]
    assert any("NS" in r for r in v["reasons"])


def test_verdict_with_nothing_measured_fails_with_four_reasons():
    v = hw2(None, None, None)
    assert not v["pass"] and v["peer_copy_min_gbps"] is None and len(v["reasons"]) == 4
    assert v["verdict"] == "fail" and v["n_cards"] is None


def test_bands_sit_between_pcie_and_nvlink():
    assert 64 < nv.PEER_COPY_FAIL_GBPS == 150 < nv.PEER_COPY_PASS_GBPS == 300


# ---- the command runner -------------------------------------------------------------------


def done(code=0, out="", err=""):
    return lambda argv, **kw: SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_run_command_keeps_raw_text_and_the_streams_apart():
    rec = nv.run_command(["nvidia-smi", "topo", "-m"], 5, done(0, "abc\n", "warn\n"))
    assert rec["ok"] and rec["text"] == "abc\nwarn\n" and rec["error"] is None
    assert (rec["stdout"], rec["stderr"]) == ("abc\n", "warn\n")


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


# ---- the three speed bands, the card count, failed commands, raising measures ---------------


def band(rate, n_expected=4, topo=TOPO_4, p2p=P2P_4, n=4, slow="0->2"):
    return hw2(nv.parse_gpu_matrix(topo), nv.parse_gpu_matrix(p2p),
                          pair_rates(n, 380.0, slow=(slow, rate)), n_expected)


def verdict(rates):
    return hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4), rates, 4)


@pytest.mark.parametrize("rate, expect", [(300.0, "pass"), (299.99, "review"), (240.0, "review"),
                                          (150.0, "review"), (149.99, "fail"), (20.0, "fail")])
def test_the_slowest_pair_sets_the_band(rate, expect):
    v = band(rate)
    assert v["verdict"] == expect and v["pass"] is (expect == "pass")
    assert v["slowest_pair"] == "0->2" and v["peer_copy_min_gbps"] == rate
    assert (v["pass_gbps"], v["fail_gbps"]) == (300.0, 150.0)
    assert bool(v["reasons"]) == (expect != "pass")


def test_lines_name_the_band_the_pair_and_the_two_lines():
    assert nv.hw2_line(band(240.0)) == ("HW2 review: slowest pair 0->2 at 240 GB/s, under the "
                                        "300 GB/s target (150-300 returns to the user)")
    assert nv.hw2_line(band(380.0)).startswith(
        "HW2 pass: slowest pair 0->1 at 380 GB/s, over the 300 GB/s target")
    # a rate that rounds up to a line is shown floored, under it
    assert "at 299 GB/s" in nv.hw2_line(band(299.97))
    fail = nv.hw2_line(band(100.0))
    assert fail.startswith("HW2 FAIL: slowest pair 0->2 at 100 GB/s, under the 150 GB/s fail line")
    assert "300" in fail


def test_a_topology_of_the_wrong_size_fails_and_records_the_counts():
    p2p8 = " \t" + "\t".join(f"GPU{i}" for i in range(8)) + "\n" + "\n".join(
        f" GPU{i}\t" + "\t".join("X" if i == j else "OK" for j in range(8)) for i in range(8))
    v = band(380.0, topo=TOPO_8, p2p=p2p8, n=8)
    assert v["verdict"] == "fail" and (v["n_cards"], v["expected_cards"]) == (8, 4)
    assert any("8 GPUs" in r and "FS31_CARDS is 4" in r for r in v["reasons"])
    assert band(380.0, n_expected=8, topo=TOPO_8, p2p=p2p8, n=8)["verdict"] == "pass"


def test_a_rehearsal_records_gate_false_and_its_lines_do_not_read_as_a_pass():
    two = ("        GPU0    GPU1\nGPU0     X      NV18\nGPU1    NV18     X\n")
    p2p = " \tGPU0\tGPU1\n GPU0\tX\tOK\n GPU1\tOK\tX\n"
    v = band(380.0, n_expected=2, topo=two, p2p=p2p, n=2, slow="0->1")
    assert v["verdict"] == "pass" and v["gate"] is False and v["pass"] is False
    assert v["expected_cards"] == 2
    line = nv.hw2_line(v)
    assert line.startswith("HW2 rehearsal (2 of 4 cards), not the gate") and "pass" not in line
    v = band(240.0, n_expected=2, topo=two, p2p=p2p, n=2, slow="0->1")
    assert nv.hw2_line(v).startswith("HW2 rehearsal (2 of 4 cards), not the gate")
    assert band(380.0)["gate"] is True


def test_the_expected_count_comes_from_fs31_cards(monkeypatch):
    two = "        GPU0    GPU1\nGPU0     X      NV18\nGPU1    NV18     X\n"
    p2p = " \tGPU0\tGPU1\n GPU0\tX\tOK\n GPU1\tOK\tX\n"
    args = (nv.parse_gpu_matrix(two), nv.parse_gpu_matrix(p2p), pair_rates(2, 380.0))
    monkeypatch.delenv("FS31_CARDS", raising=False)
    assert hw2(*args)["verdict"] == "fail"
    monkeypatch.setenv("FS31_CARDS", "2")
    v = hw2(*args)
    assert v["verdict"] == "pass" and not v["gate"] and v["pass"] is False


def test_missing_and_unusable_rates_name_the_pair():
    v = verdict({k: x for k, x in pair_rates(4, 380.0).items() if k != "3->1"})
    assert v["verdict"] == "fail" and any("3->1" in r for r in v["reasons"])
    for bad in (float("nan"), float("inf"), 0.0, -1.0, None):
        v = verdict(pair_rates(4, 380.0, slow=("2->0", bad)))
        assert v["verdict"] == "fail" and any("2->0" in r for r in v["reasons"]), bad


def test_peer_reads_for_other_gpus_than_the_topology_fail():
    p2p3 = " \tGPU0\tGPU1\tGPU2\n GPU0\tX\tOK\tOK\n GPU1\tOK\tX\tOK\n GPU2\tOK\tOK\tX\n"
    assert not hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(p2p3),
                              pair_rates(4, 380.0), 4)["pass"]


def _smi(topo_code=0, p2p_code=0, topo_out=TOPO_4, p2p_out=P2P_4):
    def run(argv, **kw):
        if argv[:3] == ["nvidia-smi", "topo", "-m"]:
            return SimpleNamespace(returncode=topo_code, stdout=topo_out, stderr="err text")
        if argv[:4] == ["nvidia-smi", "topo", "-p2p", "r"]:
            return SimpleNamespace(returncode=p2p_code, stdout=p2p_out, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def _fast(monkeypatch, rates=None, wrong=()):
    monkeypatch.setattr(nv, "peer_check_pairs", lambda: {
        f"{a}->{b}": {"ok": f"{a}->{b}" not in wrong}
        for a in range(4) for b in range(4) if a != b})
    monkeypatch.setattr(nv, "torch_view", lambda: {"device_count": 4})
    monkeypatch.setattr(nv, "peer_copy_bandwidth",
                        lambda: {"pairs": pair_rates(4, 380.0) if rates is None else rates})
    monkeypatch.setattr(nv, "fanout_bandwidth", lambda: {})
    monkeypatch.setattr(nv, "pinned_bandwidth", lambda: {})
    monkeypatch.setattr(nv.shutil, "which", lambda n: None)
    monkeypatch.delenv("FS31_CARDS", raising=False)


@pytest.mark.parametrize("which, text", [("topo", "topo -m"), ("p2p", "topo -p2p r")])
def test_a_failed_command_is_not_parsed_and_its_reason_names_it(monkeypatch, which, text):
    _fast(monkeypatch)
    r = nv.build_report(_smi(**{f"{which}_code": 6}))
    key = "topo" if which == "topo" else "p2p_read"
    rec = r["commands"][key]
    assert rec["stdout"] and rec["stderr"] == ("err text" if which == "topo" else "")
    assert r[key] is None and r["hw2"]["verdict"] == "fail"
    assert any(text in x and "exit code 6" in x for x in r["hw2"]["reasons"])


def test_a_timed_out_command_gives_no_matrix(monkeypatch):
    _fast(monkeypatch)

    def run(argv, **kw):
        if argv[:3] == ["nvidia-smi", "topo", "-m"]:
            raise subprocess.TimeoutExpired(argv, 60, output=TOPO_4)
        return _smi()(argv, **kw)

    r = nv.build_report(run)
    assert r["topo"] is None and any("timed out" in x for x in r["hw2"]["reasons"])


def test_a_raising_measure_is_recorded_and_the_report_is_still_built(monkeypatch):
    _fast(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("CUDA error: wedged")

    for name in ("torch_view", "fanout_bandwidth", "pinned_bandwidth", "nvbandwidth"):
        monkeypatch.setattr(nv, name, boom)
    r = nv.build_report(_smi())
    for key in ("torch", "fanout", "pinned_host", "nvbandwidth"):
        assert r[key] == {"error": "RuntimeError: CUDA error: wedged"}
    assert r["hw2"]["pass"]  # the peer copies still stood


def test_a_raising_peer_copy_measure_fails_hw2(monkeypatch):
    _fast(monkeypatch)

    def boom():
        raise RuntimeError("CUDA error: wedged")

    monkeypatch.setattr(nv, "peer_copy_bandwidth", boom)
    r = nv.build_report(_smi())
    assert r["peer_copy"] == {"error": "RuntimeError: CUDA error: wedged"}
    assert r["hw2"]["verdict"] == "fail"
    assert any("no peer-copy bandwidth" in x for x in r["hw2"]["reasons"])


def test_a_wedged_card_costs_its_pairs_not_the_report(monkeypatch):
    import torch

    class Wedged:
        def __getattr__(self, name):
            raise RuntimeError("wedged")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (_ for _ in ()).throw(
        RuntimeError("wedged")))
    got = nv.peer_copy_bandwidth()
    assert got["pairs"] == {"0->1": None, "1->0": None} and "wedged" in got["notes"]["0->1"]
    assert "error" in nv.fanout_bandwidth()
    assert hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                          got["pairs"], 4)["verdict"] == "fail"


def test_main_prints_the_line_and_exits_zero(tmp_path, monkeypatch, capsys):
    _fast(monkeypatch)
    build = nv.build_report
    monkeypatch.setattr(nv, "build_report", lambda: build(_smi()))
    assert nv.main(["--out", str(tmp_path / "n.json")]) == 0
    assert capsys.readouterr().out.startswith("HW2 pass: slowest pair ")
    doc = json.loads((tmp_path / "n.json").read_text())
    assert doc["hw2"]["verdict"] == "pass" and doc["hw2"]["n_cards"] == 4


# ---- the byte check: wrong bytes at the destination fail HW2 whatever the rate ----------------


def test_the_verdict_fails_when_a_pair_landed_wrong_bytes():
    v = hw2(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4), pair_rates(4, 380.0),
            wrong=("1->0", "0->1"))
    assert v["verdict"] == "fail" and not v["pass"]
    assert v["peer_bytes_ok"]["1->0"] is False and v["peer_bytes_ok"]["2->3"] is True
    assert any("wrong bytes" in r and "0->1, 1->0" in r for r in v["reasons"])
    line = nv.hw2_line(v)
    assert line.startswith("HW2 FAIL: peer copies were WRONG") and "0->1, 1->0" in line


def test_the_verdict_fails_when_the_byte_check_is_missing_or_incomplete():
    topo, p2p = nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4)
    v = hw2(topo, p2p, pair_rates(4, 380.0), peer_bytes=None)
    assert v["verdict"] == "fail" and any("byte check" in r for r in v["reasons"])
    part = {k: True for k in right_bytes(topo) if k != "3->2"}
    v = hw2(topo, p2p, pair_rates(4, 380.0), peer_bytes=part)
    assert v["verdict"] == "fail" and v["peer_bytes_ok"]["3->2"] is False
    assert any("3->2" in r for r in v["reasons"])
    v = hw2(topo, p2p, pair_rates(4, 380.0), peer_bytes={**right_bytes(topo), "0->1": None})
    assert v["verdict"] == "fail" and v["peer_bytes_ok"]["0->1"] is False


def test_the_report_records_the_byte_check_and_a_wrong_pair_fails_it(monkeypatch):
    _fast(monkeypatch, wrong=("2->3",))
    r = nv.build_report(_smi())
    assert r["peer_bytes"]["2->3"] is False and r["hw2"]["peer_bytes_ok"]["2->3"] is False
    assert r["hw2"]["verdict"] == "fail" and any("2->3" in x for x in r["hw2"]["reasons"])
    _fast(monkeypatch)
    assert nv.build_report(_smi())["hw2"]["pass"]


def test_a_raising_byte_check_fails_hw2_and_the_report_is_still_built(monkeypatch):
    _fast(monkeypatch)

    def boom():
        raise RuntimeError("CUDA error: wedged")

    monkeypatch.setattr(nv, "peer_check_pairs", boom)
    r = nv.build_report(_smi())
    assert r["peer_bytes"] == {"error": "RuntimeError: CUDA error: wedged"}
    assert r["hw2"]["verdict"] == "fail" and any("wedged" in x for x in r["hw2"]["reasons"])


# ---- round 5: what the probe's own run of the byte check carries to HW2 ---------------------


def _raised(case="side_stream", err="RuntimeError: illegal memory access", every=False):
    cases = {"host_staged": True, "to_4KB": True, case: False}
    errors = {case: err}
    if every:
        cases = {"host_staged": True, "to_4KB": False, case: False}
        errors = {"to_4KB": err, case: err}
    return {"ok": False, "cases": cases, "errors": errors, "card_suspect": False}


@pytest.mark.parametrize("pair, say", [
    ({"ok": False, "cases": {"host_staged": True, "to_4KB": False}, "errors": {},
      "card_suspect": False}, "WRONG"),
    (_raised(), "RAISED"),
    (_raised(every=True), "could not copy"),
    ({"ok": False, "cases": {"host_staged": False}, "errors": {}, "card_suspect": True},
     "card is suspect"),
    ({"ok": False, "cases": {"host_staged": False},
      "errors": {"host_staged": "RuntimeError: launch failure"}, "card_suspect": True},
     "could not copy"),
])
def test_the_probes_own_check_carries_its_errors_to_hw2_and_its_line(monkeypatch, pair, say):
    _fast(monkeypatch)
    pairs = {f"{a}->{b}": {"ok": True} for a in range(4) for b in range(4) if a != b}
    pairs["1->0"] = pair
    monkeypatch.setattr(nv, "peer_check_pairs", lambda: pairs)
    r = nv.build_report(_smi())
    h = r["hw2"]
    assert h["verdict"] == "fail" and h["peer_bytes_ok"]["1->0"] is False
    d = h["peer_bytes_detail"]
    assert list(d) == ["1->0"]
    assert set(d["1->0"]) == {"card_suspect", "errors", "wrong", "all_raised"}
    assert say in nv.hw2_line(h)
    assert any("1->0" in x for x in h["reasons"])
    if say == "RAISED":
        assert "side_stream raised RuntimeError: illegal memory access" in " ".join(h["reasons"])
        assert "wrong bytes" not in " ".join(h["reasons"])


def test_hw2_says_wrong_bytes_only_for_pairs_that_compared_wrong():
    topo, p2p = nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4)
    rates = pair_rates(4, 380.0)
    wrong = ("0->1", "2->3")
    detail = {"0->1": {"card_suspect": False, "errors": {}, "all_raised": False},
              "2->3": {"card_suspect": False, "errors": {"copy_4MB": "RuntimeError: x"},
                       "all_raised": False}}
    v = nv.hw2_verdict(topo, p2p, rates, None, None, right_bytes(topo, wrong), detail)
    why = "; ".join(v["reasons"])
    assert "wrong bytes, or were not checked, for pair 0->1" in why
    assert "pair 2->3: copy_4MB raised RuntimeError: x" in why
    assert "2->3" not in why.split("pair 0->1")[0]


# ---- round 6: a pair with wrong cases and raised cases says both ----------------------------

MIXED = {"ok": False,
         "cases": {"host_staged": True, "to_4MB": False, "copy_4MB": False, "side_stream": False},
         "errors": {"side_stream": "RuntimeError: illegal memory access"},
         "card_suspect": False}


def _mixed_hw2(pair=MIXED):
    topo, p2p = nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4)
    detail = nv.peer_bytes_detail({"0->1": pair})
    v = nv.hw2_verdict(topo, p2p, pair_rates(4, 380.0), None, None,
                       right_bytes(topo, ("0->1",)), detail)
    return v


def test_peer_bytes_detail_lists_the_cases_that_compared_wrong():
    d = nv.peer_bytes_detail({"0->1": MIXED, "1->0": {"ok": True}})
    assert list(d) == ["0->1"]
    assert d["0->1"]["wrong"] == ["to_4MB", "copy_4MB"]  # not the raised case, not the control
    assert d["0->1"]["errors"] == {"side_stream": "RuntimeError: illegal memory access"}
    assert d["0->1"]["all_raised"] is False


def test_a_mixed_pair_reads_as_wrong_bytes_and_names_its_raised_case_in_both_places():
    v = _mixed_hw2()
    why = "; ".join(v["reasons"])
    want = "0->1 (to_4MB, copy_4MB; side_stream raised RuntimeError: illegal memory access)"
    assert "landed wrong bytes, or were not checked, for pair " + want in why
    line = nv.hw2_line(v)
    assert "peer copies were WRONG (or unchecked) on pair " + want in line
    assert "RAISED" not in line  # the pair is under WRONG, not only a raised case
    # the pure-raised pair keeps its own wording
    raised = _mixed_hw2(_raised())
    assert "RAISED on pair 0->1" in nv.hw2_line(raised) and "WRONG" not in nv.hw2_line(raised)


def test_the_priority_branches_hold_over_wrong_cases():
    every = {"ok": False, "cases": {"host_staged": True, "to_4KB": False, "copy_4KB": False},
             "errors": {"to_4KB": "RuntimeError: a", "copy_4KB": "RuntimeError: a"},
             "card_suspect": False}
    control = {"ok": False, "cases": {"host_staged": False, "to_4KB": False}, "errors": {},
               "card_suspect": True}
    assert "could not copy" in nv.hw2_line(_mixed_hw2(every))
    assert "WRONG" not in nv.hw2_line(_mixed_hw2(every))
    assert "card is suspect" in nv.hw2_line(_mixed_hw2(control))
    assert "WRONG" not in nv.hw2_line(_mixed_hw2(control))


def test_the_error_words_are_one_function_for_the_check_the_session_and_the_probe(tmp_path):
    import json

    from verification.runs import fs11_session as fs
    from verification.runs import peer_check as pc

    assert pc.peer_error_text is nv.peer_error_text
    pair = {"ok": False,
            "cases": {"host_staged": True, "to_4MB": False, "copy_4MB": False,
                      "side_stream": False},
            "errors": {"side_stream": "RuntimeError: illegal memory access",
                       "copy_4MB": "BadResult: the result is on cpu, not the destination cuda:1"},
            "card_suspect": False, "gbps": None}
    raised = "side_stream raised RuntimeError: illegal memory access"
    bad = "copy_4MB gave a bad result (the result is on cpu, not the destination cuda:1)"
    f = tmp_path / "pc.json"
    f.write_text(json.dumps({"all_ok": False, "seconds": 1.0, "pairs": {"0->1": pair}}))
    texts = {
        "check": " ".join(pc.pair_lines("0->1", pair)),
        "session": fs.peer_wrong_text(fs._peer_check_extras(f)["peer_check"]),
        "probe line": nv.hw2_line(_mixed_hw2(pair)),
        "probe reason": "; ".join(_mixed_hw2(pair)["reasons"]),
    }
    for who, text in texts.items():
        assert raised in text and bad in text, who
        assert "BadResult" not in text, who
