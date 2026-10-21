"""FS33c on real GPUs: a recorder on ``cuda:0`` and ``cuda:1``.

Skipped without two CUDA devices. Written on a Mac and desk-checked only: the first run is on
the two-GPU pod (FS33d).
"""

from __future__ import annotations

import pytest
import torch

from verification.runs.metrics import MemoryRecorder, TimeRecorder, memory_probe

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two CUDA devices")

MIB = 2**20
D0, D1 = "cuda:0", "cuda:1"


class _Rec:
    def __init__(self, t, rejection=None):
        self.t, self.rejection = t, rejection


def _work(device: str, n: int = 4096, reps: int = 40) -> torch.Tensor:
    """Launch work on ``device`` and return without waiting for it."""
    x = torch.randn(n, n, device=device)
    for _ in range(reps):
        x = (x @ x).tanh()
    return x


def _reference_seconds(device: str) -> float:
    """How long the work takes on ``device``, waited for."""
    import time  # noqa: PLC0415
    _work(device, reps=2)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _work(device)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def test_a_section_launching_work_on_the_second_device_is_timed_to_its_end():
    ref = _reference_seconds(D1)
    rec = TimeRecorder("r", "s", devices=(D0, D1))
    with rec.seam()("5"):
        _work(D1)  # no synchronize: the section's exit must wait for cuda:1
    assert torch.cuda.current_stream(D1).query()  # cuda:1 was drained by the exit
    part = rec.close_step(1)
    acc = part.sections["5"]
    assert acc.time >= 0.8 * ref
    # the device spans: one per device, on the host clock; cuda:1's holds the work
    (g1,) = acc.gpu_more[0]
    assert (g1[1] - g1[0]) / 1e6 >= 0.5 * ref
    g0 = acc.gpu[0]
    assert g0[1] - g0[0] < g1[1] - g1[0]
    h0, h1 = acc.spans[0]
    assert g1[1] <= h1 / 1000.0 + 1e3  # the device span ends no later than the host's exit


def test_one_device_recorder_undercounts_the_same_section():
    """What the single-device timing missed: the same section, timed against cuda:0 only."""
    ref = _reference_seconds(D1)
    rec = TimeRecorder("r", "s", device=D0)
    with rec.seam()("5"):
        _work(D1)
    seen = rec.close_step(1).sections["5"].time
    torch.cuda.synchronize(D1)
    assert seen < 0.5 * ref  # the launch, not the work


def test_nested_sections_take_the_longest_device_span():
    ref = _reference_seconds(D1)
    rec = TimeRecorder("r", "s", devices=(D0, D1))
    seam = rec.seam()
    with seam("5"):
        with seam("5.measure"):
            _work(D1)
    part = rec.close_step(1)
    assert part.sections["5.measure"].time >= 0.5 * ref


def test_per_device_peaks_see_an_allocation_on_the_second_device():
    rec = MemoryRecorder("r", "s", memory_probe(D0), devices=(D0, D1))
    with rec.seam()("5"):
        x = torch.empty(256 * MIB, dtype=torch.uint8, device=D1)
        del x
    rec.on_step(_Rec(1))
    row = next(r for r in rec.rows if r["component"] == "5")
    assert row["devices"] == [D0, D1]
    assert row["device_peak_bytes"][1] >= 256 * MIB
    assert row["device_peak_bytes"][0] - row["device_start_bytes"][0] < 64 * MIB
    assert row["peak_bytes"] == row["device_peak_bytes"][0]  # today's field: the first device
    assert row["device_wide_peak_bytes"][1] >= row["device_wide_start_bytes"][1]
    assert all("lower bound" in s for s in row["device_wide_source"])


def _other_process(conn, device: str, nbytes: int) -> None:
    import torch  # noqa: PLC0415 (a spawned process)
    x = torch.empty(nbytes, dtype=torch.uint8, device=device)
    conn.send("ready")
    conn.recv()  # hold it until the parent has looked
    del x
    conn.send("freed")


def test_device_wide_use_sees_memory_another_process_allocated():
    ctx = torch.multiprocessing.get_context("spawn")
    here, there = ctx.Pipe()
    rec = MemoryRecorder("r", "s", memory_probe(D0), devices=(D0, D1))
    free, total = torch.cuda.mem_get_info(D1)
    base = total - free
    proc = ctx.Process(target=_other_process, args=(there, D1, 512 * MIB))
    proc.start()
    try:
        assert here.poll(120), "the other process never allocated"
        assert here.recv() == "ready"
        with rec.seam()("5"):
            pass
        rec.on_step(_Rec(1))
    finally:
        here.send("go")
        if here.poll(60):  # never block on a child that died: the parent holds `there` open
            here.recv()
        proc.join(60)
        if proc.is_alive():
            proc.kill()
    row = next(r for r in rec.rows if r["component"] == "5")
    assert row["device_wide_start_bytes"][1] - base >= 400 * MIB  # the other process's tensor
    assert row["device_wide_end_bytes"][1] - base >= 400 * MIB
    # this process's own peak does not: its rise over its start stays small
    assert row["device_peak_bytes"][1] - row["device_start_bytes"][1] < 64 * MIB
