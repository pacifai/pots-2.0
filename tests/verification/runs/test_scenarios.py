"""The scenario harness's model reuse (O4): one built model serves W_0, check 8's reference run
and the prover in turn, with results bit-identical to a fresh build per run."""

import pytest
import torch
from torch import nn

from verification.computation.instances.mlp import MLPComputation, synthetic_dataset
from verification.computation.interface import snapshot_weights
from verification.runs.scenarios import HONEST, ReusedModel, honest_final, run_scenario

K = 7
T = 2


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, 40, seed=0)


def _equal(a, b):
    return list(a) == list(b) and all(torch.equal(a[n], b[n]) for n in a)


def _residuals(r):
    return {t: [(p.m, p.kappa, p.normalized) for p in s.products]
            + [(x.weight, x.rho_max) for x in s.tensors] for t, s in r.verifier.stats.items()}


def test_reuse_is_bit_identical_to_fresh_builds(c, D):
    torch.manual_seed(0)
    w0 = snapshot_weights(c, c.build_model())
    fresh_final = honest_final(c, D, w0, T)
    fresh = run_scenario(c, D, w0, HONEST, T=T, k=K, final=fresh_final)

    models = ReusedModel(lambda: c.build_model())
    with torch.no_grad():  # the reused model starts from other weights; each run loads its own
        for p in models.model.parameters():
            p.normal_()
    final = honest_final(c, D, w0, T, build_model=models)
    final_versions = [w._version for w in final.values()]
    reused = run_scenario(c, D, w0, HONEST, T=T, k=K, final=final, build_model=models)

    assert fresh.passed and reused.passed
    assert _equal(final, fresh_final)
    assert _equal(reused.loop.w_final, fresh.loop.w_final)
    assert [s.loss for s in reused.loop.steps] == [s.loss for s in fresh.loop.steps]
    assert _residuals(reused) == _residuals(fresh)
    # check 8's reference is independent of the model the prover trained afterwards
    params = {p.untyped_storage().data_ptr() for p in models.model.parameters()}
    assert all(w.untyped_storage().data_ptr() not in params for w in final.values())
    assert [w._version for w in final.values()] == final_versions
    assert _equal(final, fresh_final)


class _WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3, bias=False)
        self.register_buffer("inv_freq", torch.arange(3.0))


def test_reused_model_refuses_a_changed_model():
    models = ReusedModel(_WithBuffer)
    m = models()
    assert models() is m

    m.inv_freq.add_(1)
    with pytest.raises(RuntimeError, match="buffers changed"):
        models()
    m.inv_freq.sub_(1)
    assert models() is m

    h = m.lin.register_forward_hook(lambda *a: None)
    with pytest.raises(RuntimeError, match="hooks left"):
        models()
    h.remove()

    m.lin.weight.grad = torch.zeros(3, 3)
    with pytest.raises(RuntimeError, match="gradient state"):
        models()
    m.lin.weight.grad = None

    m.train(not m.training)
    with pytest.raises(RuntimeError, match="mode changed"):
        models()
    m.train(not m.training)

    m.register_buffer("extra", torch.zeros(1))
    with pytest.raises(RuntimeError, match="buffers changed"):
        models()
