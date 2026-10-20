"""Check 5's ``_served_group`` reads each role of an interleaved group as a leaf range and
stacks them with strided copies; the result is the old ``torch.stack`` bit for bit."""

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.transcript.reader import TranscriptView
from verification.verifier import checks
from verification.verifier.checks import _measure_groups, _member_runs
from verification.verifier.context import CommittedLeaves, StepContext

from tests.verification.computation.test_decoder import _bf16_step, run_capture
from tests.verification.decoder_helpers import make_records, tiny_config


def _fake_replay(specs, idx):
    """Operands that only carry the group's declared shapes; the leaves are under test."""
    first = specs[idx[0]]

    class R:
        def operand_group(self, ms, dtype):
            return (torch.zeros(len(ms), *first.a_shape, dtype=dtype),
                    torch.zeros(len(ms), *first.b_shape, dtype=dtype))

    return R()


def _step(bf16):
    """A decoder's labeled leaves for the whole step; members are views of batch tensors."""
    if bf16:
        return _bf16_step(12)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    model = c.build_model()
    records = make_records((12, 7))
    w0 = {k: model.get_parameter(k).detach().clone() for k in c.weight_names}
    cap = run_capture(c, model, records)
    products = [p.detach() for p in c.label(cap, model)]
    return c, [*records, *w0.values(), *products, *w0.values()]


def _view_and_ctx(c, leaves):
    committed = CommittedLeaves()
    for leaf in leaves:
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=1)
    return TranscriptView(c, committed), ctx


def _reference(view, ms, device):
    """The old result: one ``torch.stack`` of the leaves, moved, then cast."""
    return torch.stack([view.product(m) for m in ms]).detach().to(device).to(torch.float32)


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_served_group_equals_the_stacked_leaves(bf16):
    c, leaves = _step(bf16)
    view, ctx = _view_and_ctx(c, leaves)
    seen_roles = set()
    for run in _member_runs(c.products):
        if len(run) < 2:
            continue
        for idx in _measure_groups(run):
            seen_roles |= {c.product_class(run[i]) for i in idx}
            for sub in (idx, idx[1:], idx[:-1], idx[3:9], idx[:1]):  # chunks and windows
                got = checks._served_group(c, run, sub, ctx, _fake_replay(run, sub), view)
                want = _reference(view, [run[i].m for i in sub], got.p.device)
                assert got.p.dtype == torch.float32 and got.p.is_contiguous()
                assert got.p.shape == want.shape and torch.equal(got.p, want), sub[:3]
    assert seen_roles == {"S", "O", "dA", "dV", "dQ", "dK"}


def test_served_group_interleaved_backward_group_reads_ranges_not_members(monkeypatch):
    c, leaves = _step(False)
    view, ctx = _view_and_ctx(c, leaves)
    run = next(r for r in _member_runs(c.products)
               if len(r) > 1 and c.product_class(r[0]) == "dA" and r[0].layer == c.L)
    idx = next(g for g in _measure_groups(run) if c.product_class(run[g[-1]]) == "dK")
    ms = [run[i].m for i in idx]
    calls = []
    real = CommittedLeaves.tensor_block

    def counting(self, ix, own=False):
        calls.append(len(ix))
        return real(self, ix, own)

    monkeypatch.setattr(CommittedLeaves, "tensor_block", counting)
    got = checks._served_group(c, run, idx, ctx, _fake_replay(run, idx), view)
    assert not calls  # every range was proved once and read as a view: no per-leaf path
    assert torch.equal(got.p, _reference(view, ms, got.p.device))
    # A write after the read is caught on the next one, as a per-leaf read would.
    view.product(ms[len(ms) // 2]).add_(1)
    with pytest.raises(Exception, match="changed in place"):
        checks._served_group(c, run, idx, ctx, _fake_replay(run, idx), view)
    assert calls  # and that error came from the per-leaf path
