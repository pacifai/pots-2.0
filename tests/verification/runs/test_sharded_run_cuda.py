"""A sharded run on real GPUs (FS32f, F22): skipped unless two CUDA devices exist.

The prover trains on ``cuda:0``. The verifier devices are ``(cuda:0, cuda:1)`` on a pair and
every GPU but the prover's on three or more, so a four-GPU session checks the placement that
leaves the prover's GPU out. Each test asserts where the data sits during the run, and that
the run's numbers equal the unsharded run's bit for bit.
"""

import pytest
import torch

from tests.verification.runs import test_sharded_run as sr
from verification.verifier import checks
from verification.verifier.matmul_check.freivalds import MeasureScratch

N = torch.cuda.device_count()
needs_two = pytest.mark.skipif(N < 2, reason="needs two CUDA devices")
DEVICES = (("cuda:0", "cuda:1") if N < 3 else tuple(f"cuda:{i}" for i in range(1, N)))
PROVER = "cuda:0"


def _sharded(edit=None):
    return sr.run(DEVICES, edit, device=PROVER)


@needs_two
def test_two_gpus_honest_run_equals_unsharded():
    one, many = sr.run(None, device=PROVER), _sharded()
    assert one[1].verdict.accepted
    sr._same(one, many)


@needs_two
@pytest.mark.parametrize("which", ["linear", "glue", "product"])
def test_two_gpus_forged_leaf_is_rejected_alike(which):
    one, many = sr.run(None, sr.forge(which), device=PROVER), _sharded(sr.forge(which))
    assert one[1].rejection is not None and many[1].rejection == one[1].rejection
    sr._same(one, many)


@needs_two
def test_two_gpus_rows_of_layers_sit_on_different_devices_and_one_window_spans_them(monkeypatch):
    """Before the join each run's rows are on its product's device; with one window for the
    whole step, the window holds rows from more than one device and is judged as one. A run's
    rows are on its operands' (unit) device."""
    seen = []
    real = checks._judge_window

    def spy(c, batch, ctx, bands):
        seen.append([(r.specs[0].name, checks._rows_device(r.rows)) for r in batch
                     if r.rows is not None])
        return real(c, batch, ctx, bands)

    monkeypatch.setattr(checks, "_judge_window", spy)
    c, res, _, _ = _sharded()
    assert res.verdict.accepted
    window = max(seen, key=len)
    devices = {torch.device(d) for d in DEVICES}
    on = {d for _, d in window}
    assert on <= devices and len(on) >= 2  # rows of different layers, one window
    if N >= 3:
        assert torch.device(PROVER) not in on  # the prover's GPU holds none of the verifier's


@needs_two
def test_two_gpus_one_scratch_buffer_per_device_stays_put(monkeypatch):
    """Check 5's scratch holds one buffer per device, and a device's buffer keeps its
    ``data_ptr`` for as long as it needs no more room; check 6's holds one per weight device."""
    calls, made = [], []

    class Spy(MeasureScratch):
        def __init__(self):
            super().__init__()
            made.append(self)

        def buffer(self, n, device=torch.device("cpu")):
            out = super().buffer(n, device)
            calls.append((device, self._bufs[device].numel(), self._bufs[device].data_ptr()))
            return out

    updates = []
    real_update = checks.UpdateScratch

    class USpy(real_update):
        def __init__(self):
            super().__init__()
            updates.append(self)

    monkeypatch.setattr(checks, "MeasureScratch", Spy)
    monkeypatch.setattr(checks, "UpdateScratch", USpy)
    c, res, _, _ = _sharded()
    assert res.verdict.accepted
    by_device = {}
    for dev, numel, ptr in calls:
        by_device.setdefault(dev, []).append((numel, ptr))
    assert len(by_device) >= min(2, len(DEVICES))
    for dev, seq in by_device.items():
        for (n0, p0), (n1, p1) in zip(seq, seq[1:]):
            assert p0 == p1 or n1 > n0, dev  # a new allocation only where it grew
    for sc in made:
        assert len(sc._bufs) <= len(DEVICES)
    used = {d for sc in updates for d in sc._bufs}
    assert used <= {c.placement.weight_device(n) for n in c.weight_names}


@needs_two
def test_two_gpus_glue_gradient_arrives_on_the_weights_device(monkeypatch):
    """Check 6b's gradient is on the weight leaf's device when it is handed over (the
    replay returns it there), so ``_glue_gradient`` has nothing to move."""
    seen = []
    real = checks._glue_gradient

    def spy(grads, name, device):
        seen.append((name, grads[name].device, device))
        return real(grads, name, device)

    monkeypatch.setattr(checks, "_glue_gradient", spy)
    c, res, _, _ = _sharded()
    assert res.verdict.accepted and seen
    assert all(have == want for _, have, want in seen), seen
    assert len({d for _, d, _ in seen}) >= min(2, len(DEVICES))


@needs_two
@pytest.mark.slow
def test_two_gpus_smollm2_sharded_run_equals_unsharded():
    """SmolLM2, 2 steps, prover on ``cuda:0``: the sharded run's losses, last root, bands and
    recorded numbers equal the unsharded run's."""
    one, many = sr.smollm2_run(PROVER, None), sr.smollm2_run(PROVER, DEVICES)
    assert many[0].placement.sharded
    assert one[1:4] == many[1:4] and one[4] == many[4]
