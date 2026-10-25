"""FS39 (plan §2.7): check-5 stamps are host spans, and ``seam.host(name)`` times a nested
section by the host clock on CUDA (no events, null device span); on every other recorder it is
``seam(name)``."""

import pytest
import torch

from tests.verification.runs.test_metrics_devices import _FakeCuda, _Rec, counting
from verification.runs import metrics
from verification.runs.metrics import CountRecorder, MemoryRecorder, TimeRecorder, memory_probe


def _line(rec):
    """The step-1 markers line of what the recorder holds (``on_step`` hands over the stamps)."""
    rec._step_stamps = rec._stamps
    return rec._marker_line(1, rec._step, type("R", (), {"t": 1})())


def _by_name(line):
    return {s["name"]: s for s in line["sections"]}


@pytest.mark.parametrize("devs", [{"device": "cuda:0"}, {"devices": ("cuda:0", "cuda:1")}])
def test_a_stamp_records_no_cuda_event_and_its_gpu_columns_are_null(monkeypatch, devs):
    fake = _FakeCuda(monkeypatch)
    rec = TimeRecorder("r", "s", **devs)
    seam = rec.seam()
    with seam("5"):
        before = fake.events
        with seam.stamp("c5.run", run=0):
            with seam.stamp("c5.measure", run=0, group=1):
                pass
        assert fake.events == before  # no event pair per stamp
    st = _line(rec)["stamps"]
    assert len(st["name"]) == 2 and len(st["h0"]) == len(st["h1"]) == 2
    assert all(b >= a for a, b in zip(st["h0"], st["h1"]))
    assert st["g0"] is None and st["g1"] is None


def test_a_stamp_outside_check_5_is_still_a_no_op(monkeypatch):
    _FakeCuda(monkeypatch)
    rec = TimeRecorder("r", "s", device="cuda:0")
    with rec.seam()("2"):
        with rec.seam().stamp("c5.run", run=0):
            pass
    assert rec._stamps == []


def test_a_host_timed_nested_section_records_no_event_on_one_device(monkeypatch):
    fake = _FakeCuda(monkeypatch)
    rec = TimeRecorder("r", "s", device="cuda:0")
    seam = rec.seam()
    with seam("5"):
        before = fake.events
        with seam.host("5.measure"):
            pass
        with seam.host("5.measure"):
            pass
        assert fake.events == before  # nothing recorded for the host section
        with seam("5.glue"):  # an ordinary nested section still records its pair
            pass
        assert fake.events == before + 2
    assert fake.sync_calls == [0, 0]  # a nested section never synchronizes
    sec = rec._step["5.measure"]
    assert sec.calls == 2 and sec.time > 0 and sec.depth == 1
    assert len(sec.spans) == 2 and sec.gpu == [None, None]
    line = _by_name(_line(rec))
    assert line["5.measure"]["gpu_us"] == [None, None] and len(line["5.measure"]["host_us"]) == 2
    assert line["5.glue"]["gpu_us"][0] is not None
    # the pending list holds the top-level and the ordinary nested section only
    assert rec._pending == []  # resolved at the top-level exit
    assert rec._step["5.glue"].gpu[0] is not None and rec._depth == 0 and rec._c5_open == 0


def test_a_host_timed_nested_section_records_no_event_on_several_devices(monkeypatch):
    fake = _FakeCuda(monkeypatch)
    devs = ("cuda:0", "cuda:1", "cuda:2")
    rec = TimeRecorder("r", "s", devices=devs)
    seam = rec.seam()
    with seam("5"):
        before = fake.events
        with seam.host("5.measure"):
            pass
        assert fake.events == before
    assert fake.sync_calls == [0, 1, 2, 0, 1, 2]
    sec = rec._step["5.measure"]
    assert sec.time > 0 and sec.gpu == [None] and sec.gpu_more == [[None], [None]]
    rec._devfields = True  # what ``on_step`` sets for a several-device step
    line = _by_name(_line(rec))["5.measure"]
    assert line["gpu_us"] == [None]
    assert line["gpu_us_by_device"] == [[None], [None], [None]]


def test_the_host_timed_section_is_the_host_clock_not_the_event_clock(monkeypatch):
    _FakeCuda(monkeypatch)  # its events tick 100 us each, so an event-timed row is >= 0.1 ms
    ticks = iter(range(0, 10**9, 7_000_000))
    monkeypatch.setattr(metrics.time, "monotonic_ns", lambda: next(ticks))
    rec = TimeRecorder("r", "s", device="cuda:0")
    seam = rec.seam()
    with seam("5"):
        with seam.host("5.measure"):
            pass
    assert rec._step["5.measure"].time == pytest.approx(7e-3)


def test_a_top_level_host_section_is_the_ordinary_section(monkeypatch):
    fake = _FakeCuda(monkeypatch)
    rec = TimeRecorder("r", "s", device="cuda:0")
    with rec.seam().host("handoff.x"):
        pass
    assert fake.sync_calls == [0, 0] and rec._step["handoff.x"].gpu[0] is not None


def test_host_is_section_off_cuda_and_on_the_other_recorders():
    with counting() as (fm, tally):
        recs = (TimeRecorder("r", "s", device="cpu"),
                MemoryRecorder("r", "s", memory_probe("cpu")),
                CountRecorder("r", "s", fm, tally))
        for rec in recs:
            seam = rec.seam()
            with seam("5"):
                assert type(seam.host("5.measure")) is type(seam("5.measure"))
                with seam.host("5.measure"):
                    pass
    rec = TimeRecorder("r", "s", device="cpu")
    with rec.seam()("5"):
        with rec.seam().host("5.measure"):
            pass
    assert rec._step["5.measure"].calls == 1 and rec._step["5.measure"].gpu == []
    mem = MemoryRecorder("r", "s", memory_probe("cpu"))
    with mem.seam()("5"):
        with mem.seam().host("5.measure"):
            pass
    mem.on_step(_Rec(1))
    assert any(r["component"] == "5.measure" and r["calls"] == 1 for r in mem.rows)
