import pytest
import torch
from torch import nn

from src.verification.capture import MutatedCaptureError
from src.verification.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from src.verification.prover import plain_step, prove_step

ETA = 0.05


@pytest.fixture
def setup():
    torch.manual_seed(0)
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)
    data = synthetic_dataset(c.widths, 12, seed=0)
    return c, c.build_model(), data, init_weights(c.widths, seed=0)


def _equal_dicts(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_step_output_layout(setup):
    c, model, data, w0 = setup
    out = prove_step(c, model, w0, data[:4])
    assert c.n_leaves == len(out.leaves())
    assert not hasattr(out, "n_leaves")  # invariant 7: the count is the declaration's
    assert list(out.w_t) == list(out.w_next) == list(c.weight_names)
    assert len(out.products) == c.M
    for spec, p in zip(c.products, out.products):
        assert tuple(p.shape) == spec.p_shape and p.dtype == torch.float32
        assert p.is_contiguous()
    assert _equal_dicts(out.w_t, w0)
    # W_t leaves are clones, not views of the stepped parameters.
    for n, p in model.named_parameters():
        assert p.data_ptr() != out.w_t[n].data_ptr()
        assert torch.equal(p.detach(), out.w_next[n])
        assert p.grad is None
    out.assert_unmodified()


def test_sgd_step_is_plain_update(setup):
    # Observation only (P5a): check 6 is banded and must not use equality.
    c, model, data, w0 = setup
    out = prove_step(c, model, w0, data[:4])
    for name, m in c.linear_weights.items():
        g = out.products[m - 1]
        # SGD's add_(grad, alpha=-lr), replayed with the same fp32 op.
        assert torch.equal(out.w_next[name], out.w_t[name].add(g, alpha=-ETA))
        assert not torch.equal(out.w_next[name], out.w_t[name])


def test_capture_on_off_bit_identical(setup):
    c, model, data, w0 = setup
    out = prove_step(c, model, w0, data[:4])
    w_next, loss = plain_step(c, c.build_model(), w0, data[:4])
    assert _equal_dicts(out.w_next, w_next)
    assert out.loss == loss


def test_chained_steps(setup):
    c, model, data, w0 = setup
    s1 = prove_step(c, model, w0, data[:4])
    s2 = prove_step(c, model, s1.w_next, data[4:8])
    assert _equal_dicts(s2.w_t, s1.w_next)
    s1.assert_unmodified()  # the second step left the first step's leaves alone


def test_train_records_hook(setup):
    c, model, data, w0 = setup
    clean, poisoned = data[:4], data[4:8]
    honest_on_poison = prove_step(c, c.build_model(), w0, poisoned)
    cheat = prove_step(c, model, w0, clean, train_records=poisoned)
    assert all(a is b for a, b in zip(cheat.records, clean))
    assert all(torch.equal(a, b) for a, b in zip(cheat.products, honest_on_poison.products))
    assert _equal_dicts(cheat.w_next, honest_on_poison.w_next)
    assert _equal_dicts(cheat.w_t, w0)


def test_perturb_hook_changes_one_product_only(setup):
    c, model, data, w0 = setup
    honest = prove_step(c, c.build_model(), w0, data[:4])
    m = c.m_of("dX_2")
    cheat = prove_step(c, model, w0, data[:4], perturb={m: lambda p: p.add_(1e-3)})
    for k, (a, b) in enumerate(zip(cheat.products, honest.products), start=1):
        if k == m:
            assert torch.equal(a, b + 1e-3)
        else:
            assert torch.equal(a, b)
    assert _equal_dicts(cheat.w_next, honest.w_next)  # training used the true gradients
    assert _equal_dicts(cheat.w_t, honest.w_t)


def test_perturb_weight_grad_leaves_update_honest(setup):
    c, model, data, w0 = setup
    honest = prove_step(c, c.build_model(), w0, data[:4])
    m = c.linear_weights[c.weight(1)]
    cheat = prove_step(c, model, w0, data[:4], perturb={m: lambda p: -p})
    assert torch.equal(cheat.products[m - 1], -honest.products[m - 1])
    assert _equal_dicts(cheat.w_next, honest.w_next)


def test_perturb_must_keep_shape(setup):
    c, model, data, w0 = setup
    with pytest.raises(ValueError, match="shape"):
        prove_step(c, model, w0, data[:4], perturb={1: lambda p: p[:1]})


def test_assert_unmodified_trips_on_mutation(setup):
    c, model, data, w0 = setup
    out = prove_step(c, model, w0, data[:4])
    out.products[0].mul_(2)
    with pytest.raises(MutatedCaptureError):
        out.assert_unmodified()


def test_rejects_wrong_batch_size_and_undeclared_params(setup):
    c, model, data, w0 = setup
    with pytest.raises(ValueError, match="records"):
        prove_step(c, model, w0, data[:3])
    model.extra = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="declared weights"):
        prove_step(c, model, w0, data[:4])


def test_versions_snapshot_includes_perturbed_slot(setup):
    c, model, data, w0 = setup
    out = prove_step(c, model, w0, data[:4], perturb={1: lambda p: p.mul_(2)})
    out.assert_unmodified()  # the perturbed tensor's own version is the baseline
    out.products[1].add_(1)
    with pytest.raises(MutatedCaptureError):
        out.assert_unmodified()
