"""The GPU experiments (E2–E4, E5b, E5c): their shapes, their skipped rows, and, where CUDA
is present, the GEMM loop and the copies."""

import contextlib
import time

import pytest
import torch

from benchmarks.hashing import cpu_hash, gpu, machine
from tests.verification.decoder_helpers import tiny_config

MB = 1 << 20
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_gemm_shapes_are_the_mlp_and_the_head():
    c = tiny_config()  # hidden 32, intermediate 48, vocab 64
    shapes = gpu.gemm_shapes(c, tokens=24)
    assert [(g.name, g.m, g.k, g.n) for g in shapes] == [
        ("mlp_up", 24, 32, 48), ("mlp_down", 24, 48, 32), ("lm_head", 24, 32, 64)]
    assert shapes[2].flops == 2 * 24 * 32 * 64


def test_gpu_hashing_rows_say_why_they_are_skipped():
    rows = gpu.e2_rows()
    assert [(r["name"], r["status"]) for r in rows] == [("cupqc_sha256", "skipped"),
                                                        ("blake3_gpu_proxy", "skipped")]
    assert "License Agreement" in rows[0]["reason"] and "SYCL" in rows[1]["reason"]
    e5 = gpu.e5_gpu_rows()
    assert [r["name"] for r in e5["E5b"]] == ["cupqc_shake256"]
    assert [r["name"] for r in e5["E5c"]] == ["cupqc_sha256_counter"]
    assert all(r["reason"] == gpu.CUPQC_REASON for rs in e5.values() for r in rs)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA guard")
def test_without_cuda_e3_and_e4_are_skipped():
    (r,) = gpu.run_e3({"tiny": gpu.gemm_shapes(tiny_config(), 8)}, window_s=0.01, iters=1)
    assert (r["status"], r["reason"]) == ("skipped", gpu.NO_CUDA)
    (r,) = gpu.run_e4(pieces=[MB], total=MB, reps=1, window_s=0.01, gemms=None,
                      hash_bytes=0, hash_chunk=1 << 16, hash_threads=1)
    assert (r["status"], r["reason"]) == ("skipped", gpu.NO_CUDA)


def test_e4_shrinks_its_host_buffer_to_fit_memory(monkeypatch):
    GiB = 1 << 30
    monkeypatch.setattr(machine, "cgroup_memory_limit", lambda: 50 * GiB)
    monkeypatch.setattr(machine, "mem_available", lambda: 500 * GiB)
    budget, readings = gpu.host_memory_budget()
    assert budget == 50 * GiB and readings["mem_available_bytes"] == 500 * GiB
    extra = 4 * GiB + gpu.MEMORY_MARGIN
    pieces = [1 * GiB, 8 * GiB]
    # The H100 preset on the A40 pod: 64 GiB doesn't fit, 32 does, and stays a multiple of 8.
    assert gpu.fit_total(64 * GiB, pieces, extra, budget) == 32 * GiB
    assert gpu.fit_total(64 * GiB, pieces, extra, None) == 64 * GiB
    assert gpu.fit_total(16 * GiB, pieces, extra, 10 * GiB) is None  # 8 GiB + 8 > 10
    assert gpu.fit_total(24 * GiB, pieces, extra, 32 * GiB) == 24 * GiB
    assert gpu.fit_total(24 * GiB, pieces, extra, 25 * GiB) is None  # 12 isn't a multiple of 8
    with pytest.raises(ValueError):
        gpu.fit_total(12 * GiB, pieces, extra, None)
    monkeypatch.setattr(machine, "cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(machine, "mem_available", lambda: None)
    assert gpu.host_memory_budget()[0] is None


def test_mem_available_reads_meminfo(monkeypatch):
    monkeypatch.setattr(machine, "_read", {"/proc/meminfo": "MemTotal:  100 kB\n"
                                           "MemAvailable:   2048 kB\n"}.get)
    assert machine.mem_available() == 2048 * 1024
    monkeypatch.setattr(machine, "_read", lambda path: None)
    assert machine.mem_available() is None


def test_window_rate_and_slowdown_from_fake_counts():
    # (time, FLOPs finished, GEMMs finished): 6e12 FLOPs over 2 s is 3 TFLOP/s.
    rate = gpu.window_rate((10.0, 1e12, 5), (12.0, 7e12, 11))
    assert rate == {"tflop_per_s": pytest.approx(3.0), "gemms": 6, "window_s": 2.0}
    assert gpu._hash_rate((0.0, 0), (0.5, 4 * 10**9)) == pytest.approx(8.0)
    assert gpu.slowdown(alone=4.0, during=3.0) == pytest.approx(0.25)
    assert gpu.slowdown(alone=4.0, during=4.2) == pytest.approx(-0.05)  # noise reads negative


def test_hash_loop_counts_the_bytes_it_hashes():
    buf = cpu_hash.make_buffer(2 * MB)
    with gpu.HashLoop(buf, "blake3", 64 << 10, threads=2) as h:
        t_end = time.perf_counter() + 30
        while h.count()[1] < 4 * buf.size and time.perf_counter() < t_end:
            time.sleep(0.01)
    assert h.count()[1] >= 4 * buf.size
    assert h.count()[1] % (64 << 10) == 0


def test_host_buffer_splits_into_whole_pieces():
    host = gpu.HostBuffer(4 * MB)
    assert [p.numel() for p in host.pieces(MB)] == [MB] * 4
    with pytest.raises(ValueError):
        host.pieces(3 * MB)


def test_cgroup_quota_reads_v2_then_v1(monkeypatch):
    v1 = "/sys/fs/cgroup/cpu,cpuacct/"
    files = {v1 + "cpu.cfs_quota_us": "765000\n", v1 + "cpu.cfs_period_us": "100000\n",
             "/sys/fs/cgroup/memory/memory.limit_in_bytes": "49999998976\n"}
    monkeypatch.setattr(machine, "_read", files.get)
    assert machine.cgroup_cpu_quota() == pytest.approx(7.65)
    assert machine.cgroup_memory_limit() == 49999998976
    m = {"physical_cores": 48, "logical_cpus": 96, "usable_cpus": 96, "cgroup_cpu_quota": 7.65}
    assert machine.cores_for_threads(m) == 7
    files.update({v1 + "cpu.cfs_quota_us": "-1\n",
                  "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n"})
    assert machine.cgroup_cpu_quota() is None and machine.cgroup_memory_limit() is None
    files.update({"/sys/fs/cgroup/cpu.max": "200000 100000\n",
                  "/sys/fs/cgroup/memory.max": "max\n"})
    assert machine.cgroup_cpu_quota() == 2.0 and machine.cgroup_memory_limit() is None


# ---- CUDA only ----------------------------------------------------------------------------


@needs_cuda
def test_copies_move_the_bytes_both_ways():
    host = gpu.HostBuffer(4 * MB)
    host.tensor.copy_(torch.arange(4 * MB, dtype=torch.int64).to(torch.uint8))
    dev = torch.empty(2 * MB, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    pieces = host.pieces(2 * MB)
    for pinned in (True, False):
        with host.pinned() if pinned else contextlib.nullcontext():
            assert pieces[0].is_pinned() is pinned
            assert gpu.copy_pass(pieces[1:], dev, "h2d", stream) > 0
            assert torch.equal(dev.cpu(), pieces[1])
            dev.fill_(7)
            gpu.copy_pass(pieces[:1], dev, "d2h", stream)
            assert bool((pieces[0] == 7).all())
            host.tensor.copy_(torch.arange(4 * MB, dtype=torch.int64).to(torch.uint8))
    assert not host.tensor.is_pinned()


@needs_cuda
def test_gemm_loop_finishes_gemms():
    gemms = gpu.gemm_shapes(tiny_config(), 256)
    with gpu.GemmLoop(gemms, torch.device("cuda")) as loop:
        rate = gpu.loop_alone(loop, 0.2)
    assert rate["gemms"] > 0 and rate["tflop_per_s"] > 0
    assert loop.count()[2] >= rate["gemms"]


@needs_cuda
def test_run_e3_and_e4_tiny():
    gemms = gpu.gemm_shapes(tiny_config(), 256)
    rows = gpu.run_e3({"tiny": gemms}, window_s=0.1, iters=2, log=lambda _: None)
    assert [r["name"] for r in rows] == ["gemm"] * 3 + ["gemm_loop_alone", "gpu_hash_alone",
                                                        "gemm_loop_with_gpu_hash"]
    assert [r["status"] for r in rows] == ["measured"] * 4 + ["skipped"] * 2
    rows = gpu.run_e4(pieces=[MB, 2 * MB], total=4 * MB, reps=2, window_s=0.05,
                      gemms=("tiny", gemms), hash_bytes=2 * MB,
                      hash_chunk=64 << 10, hash_threads=2, log=lambda _: None)
    names = [r["name"] for r in rows]
    combos = 2 * 2 * 2  # pinned or pageable, two pieces, two directions
    assert names.count("copy") == names.count("copy_with_gemm_loop") == combos
    assert names.count("copy_with_cpu_hashing") == combos * len(cpu_hash.ALGOS)
    assert {r["status"] for r in rows} == {"measured"}
    for r in rows:
        if r["name"] == "copy":
            assert r["params"]["host_is_pinned"] is r["params"]["pinned"]
