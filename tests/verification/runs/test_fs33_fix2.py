"""FS33 fix round 2: check-5 stamps (host spans since FS39, plan §2.7), and the two entry points
(``bring_up``, ``store_crosscheck``) that pass ``run_devices`` to their recorders."""

from types import SimpleNamespace

import pytest
import torch

from tests.verification.decoder_helpers import make_records, tiny_config
from tests.verification.runs.test_metrics_devices import _FakeCuda
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.runs import bring_up as bu
from verification.runs import store_crosscheck as sc
from verification.runs.metrics import MetricsWriter, TimeRecorder


# ---- item 1: a stamp is a host span, whatever the recorder's first device ------------------

def _fake_with_current_device_zero(mp, n=4):
    """The FS33c stand-in with two torch behaviors: ``record()`` with no stream goes to the
    current device (0), and ``elapsed_time`` across devices raises."""
    fake = _FakeCuda(mp, n=n)
    event = torch.cuda.Event
    real_record = event.record

    def record(self, stream=None):
        real_record(self, stream if stream is not None else ("stream", 0))

    def elapsed_time(self, other):
        if self.dev != other.dev:
            raise RuntimeError(f"invalid resource handle ({self.dev} and {other.dev})")
        return (other.at - self.at) / 1000.0

    mp.setattr(event, "record", record)
    mp.setattr(event, "elapsed_time", elapsed_time)
    return fake


@pytest.mark.parametrize("devs", [("cuda:1",), ("cuda:1", "cuda:2", "cuda:3")])
def test_a_stamp_in_check_5_records_a_host_span_when_the_first_device_is_not_zero(monkeypatch, devs):
    fake = _fake_with_current_device_zero(monkeypatch)
    rec = TimeRecorder("r", "s", devices=devs)
    seam = rec.seam()
    with seam("5"):
        with seam("5.measure"):
            with rec.stamp("c5.run", 0):
                pass
    assert len(rec._stamps) == 1 and rec._stamps[0][5] >= rec._stamps[0][4]
    assert rec._pending == [] and not hasattr(rec, "_stamp_gpu")
    del fake


def test_the_single_device_stamp_is_a_host_span_too(monkeypatch):
    _fake_with_current_device_zero(monkeypatch)
    rec = TimeRecorder("r", "s", device="cuda:1")
    seam = rec.seam()
    with seam("5"):
        with seam("5.measure"):
            with rec.stamp("c5.run", 0):
                pass
    assert len(rec._stamps) == 1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs two CUDA devices")
def test_a_stamp_on_cuda_1_is_a_host_span_with_two_gpus():
    assert torch.cuda.current_device() == 0
    rec = TimeRecorder("r", "s", devices=("cuda:1",))
    seam = rec.seam()
    x = torch.randn(512, 512, device="cuda:1")
    with seam("5"):
        with seam("5.measure"):
            with rec.stamp("c5.run", 0):
                x = x @ x
    assert len(rec._stamps) == 1 and rec._stamps[0][5] >= rec._stamps[0][4]


# ---- item 2: bring_up and store_crosscheck pass run_devices to their recorders --------------

SENTINEL = ("cpu",)


def test_bring_up_memory_pass_gets_run_devices(monkeypatch):
    modes, got = [], {}
    real = bu.memory_pass

    def spy(*a, **kw):
        got.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(bu, "run_devices", lambda c, *, pipelined: modes.append(pipelined)
                        or SENTINEL)
    monkeypatch.setattr(bu, "memory_pass", spy)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
    bu.bring_up(c, D, dataset_tree(c, D).root, T=1, k=7, seed=1, tamper=False)
    assert modes == [False] and got["devices"] == SENTINEL


def test_bring_up_main_passes_run_devices_to_the_timed_recorder(monkeypatch, tmp_path):
    modes, got = [], {}

    class Stop(Exception):
        pass

    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5), seed=1)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(b"x")
    for var, value in (("VERIF_OUTPUT_DIR", str(tmp_path)), ("VERIF_ETA", "1e-3")):
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(bu, "model_pin", lambda m, r: {"repo": m, "revision": r, "weights": [],
                                                       "config_sha256": "x"})
    monkeypatch.setattr(bu, "setup_determinism", lambda cfg: None)
    monkeypatch.setattr(DecoderComputation, "from_config", classmethod(lambda cls, cfg: c))
    monkeypatch.setattr(bu, "load_committed_dataset", lambda c_, d: (D, b"\0" * 32))
    monkeypatch.setattr(bu, "run_devices", lambda c_, *, pipelined: modes.append(pipelined)
                        or SENTINEL)
    real = MetricsWriter.recorder

    def spy(self, name, **kw):
        got.update(kw)
        return real(self, name, **kw)

    monkeypatch.setattr(MetricsWriter, "recorder", spy)

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(bu, "bring_up", stop)
    with pytest.raises(Stop):
        bu.main(["--metrics"])
    assert modes == [False] and got["devices"] == SENTINEL
