"""A11's calibration run: the gradient-coherence diagnostic (the rest runs in the slow suite)."""

import pytest
import torch

from tests.verification.decoder_helpers import make_records, tiny_config
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs.calibrate import gradient_coherence


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gradient_coherence_runs_under_the_runs_autocast(dtype, monkeypatch):
    """F6b: the per-record gradients come from training's forward, under autocast at
    ``(fp32, bf16)``, and they sum to the batch gradient up to rounding."""
    c = DecoderComputation(tiny_config(), n_s=3, n=12, eta=1e-3, compute_dtype=dtype)
    model = c.build_model()
    w = snapshot_weights(c, model)
    seen = []
    logits = c.logits

    def spy(model, batch):
        seen.append(torch.is_autocast_enabled("cpu"))
        return logits(model, batch)

    monkeypatch.setattr(c, "logits", spy)
    coh = gradient_coherence(c, model, w, make_records((12, 7, 9)))
    assert seen == [dtype == torch.bfloat16]
    assert coh.B == 3 and len(coh.cosines) == 3 and all(n > 0 for n in coh.norms)
    assert coh.batch_check < (1e-6 if dtype == torch.float32 else 1e-2)
    assert all(p.grad is None for p in model.parameters())
