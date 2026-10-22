"""A sharded run on real GPUs (FS32f, F22): skipped unless two CUDA devices exist.

The prover trains on ``cuda:0``. The verifier devices are ``(cuda:0, cuda:1)`` on a pair and
every GPU but the prover's on three or more, so a four-GPU session checks the placement that
leaves the prover's GPU out. Each test asserts where the data sits during the run, and that
the run's numbers equal the unsharded run's bit for bit.
"""

import pytest
import torch

from tests.verification.decoder_helpers import verifier_devices
from tests.verification.runs import test_sharded_run as sr
from verification.verifier import checks
from verification.verifier.matmul_check.freivalds import MeasureScratch

N = torch.cuda.device_count()
needs_two = pytest.mark.skipif(N < 2, reason="needs two CUDA devices")
DEVICES = verifier_devices(N)
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


def _layer_owners(c):
    """The verifier devices that own a layer's unit (not the pinned embedding and head)."""
    pinned = c.pinned_last_units
    return {c.placement.unit_device(j) for j in c.units if j not in pinned}


@needs_two
def test_two_gpus_rows_of_layers_sit_on_different_devices_and_one_window_spans_them(monkeypatch):
    """Before the join each run's rows sit on their unit's device. The model has two layers per
    verifier device, so every device owns layers, and its products are far fewer than one
    judging window, so the whole step is one window and it must hold rows from every layer-owning
    device: it is judged as one window that crosses the device boundary."""
    seen = []
    real = checks._judge_window

    def spy(c, batch, ctx, bands):
        seen.append([checks._rows_device(r.rows) for r in batch if r.rows is not None])
        return real(c, batch, ctx, bands)

    monkeypatch.setattr(checks, "_judge_window", spy)
    c, res, _, _ = sr.run(DEVICES, device=PROVER, layers=2 * len(DEVICES))
    assert res.verdict.accepted
    owners = _layer_owners(c)
    assert len(owners) == len(DEVICES)  # by construction: each verifier device owns a layer
    assert checks._layout(c)[1] == 1  # one window per step
    window = max(seen, key=len)  # the step's one window holds every run's rows
    assert set(window) == owners and len(set(window)) >= 2
    if N >= 3:
        assert torch.device(PROVER) not in set(window)


@needs_two
def test_two_gpus_check5_working_memory_stays_on_its_device_and_check6_scratch_is_per_device(
        monkeypatch):
    """Check 5 on CUDA is the measuring kernel (FS37): each call allocates its rows, scratch and
    table on the device of the tensors it measures and holds them until ``drain``; the host-side
    ``MeasureScratch`` it replaced is not asked for a buffer (the pair session found the old
    test's per-device dict empty). So: every device that owns a layer ran calls, each call's
    rows sit on its operands' device, nothing is left in flight after the run, and no
    ``MeasureScratch`` buffer is made. Were the fallback ever to use one, a device's buffer must
    keep its ``data_ptr`` until it grows. Check 6's ``UpdateScratch`` holds one per weight
    device."""
    from verification.verifier.matmul_check import measure_kernel

    kernel_calls, calls, made = [], [], []
    real_enqueue = measure_kernel.enqueue

    def enqueue(groups, **kw):
        rows = real_enqueue(groups, **kw)
        kernel_calls.append((rows.device, {t.device for g in groups
                                           for t in (g.a, g.b, g.p, g.bias) if t is not None}))
        return rows

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

    monkeypatch.setattr(measure_kernel, "enqueue", enqueue)
    monkeypatch.setattr(checks, "MeasureScratch", Spy)
    monkeypatch.setattr(checks, "UpdateScratch", USpy)
    c, res, _, _ = sr.run(DEVICES, device=PROVER, layers=2 * len(DEVICES))
    assert res.verdict.accepted
    assert kernel_calls, "check 5 never ran the measuring kernel"
    assert all(dev in ops and len(ops) == 1 for dev, ops in kernel_calls)  # rows with operands
    ran = {dev for dev, _ in kernel_calls}
    assert _layer_owners(c) <= ran <= {torch.device(d) for d in DEVICES}
    assert not any(entries for entries in measure_kernel._inflight.values())  # all released
    by_device = {}
    for dev, numel, ptr in calls:
        by_device.setdefault(dev, []).append((numel, ptr))
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
