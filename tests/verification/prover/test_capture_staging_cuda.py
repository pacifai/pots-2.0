"""Capture staging on GPUs (FS58): with ``staging="cuda:0"`` every product stays on the
prover's GPU, and the step is the same as without it. Skips unless CUDA has two devices."""

import os

import pytest
import torch

from verification.prover.step import commit, prove_step

from tests.verification.decoder_helpers import make_records
from tests.verification.prover.test_capture_routing import DTYPES, start, tiny

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
STAGE = torch.device("cuda:0")


@two_gpus
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_staged_step_equals_the_unstaged_one_with_every_leaf_on_the_prover(B, dtype):
    c = tiny(("qwen2", False), B=B, dtype=dtype, devices=("cuda:1",), device="cuda:0")
    w0 = {n: t.to(c.placement.weight_device(n)) for n, t in start(c).items()}
    records = make_records((12, 7))
    ref = prove_step(c, c.build_model(), w0, records)
    out = prove_step(c, c.build_model(), w0, records, staging="cuda:0")
    assert out.capture["staging"] == "cuda:0" and "staging" not in ref.capture
    assert out.hint_misses == 0 and out.capture["hint_misses"] == 0
    assert all(p.device == STAGE for p in out.products)
    assert len(out.products) == len(ref.products)
    for x, y in zip(ref.products, out.products):
        assert x.dtype == y.dtype and torch.equal(x.cpu(), y.cpu())
    assert out.arrival_digests and out.arrival_digests == ref.arrival_digests
    assert commit(c, out).root == commit(c, ref).root
    for leaves, base in ((out.w_t, ref.w_t), (out.w_next, ref.w_next)):
        assert leaves.keys() == base.keys()
        for k in base:
            assert leaves[k].device == STAGE
            assert torch.equal(leaves[k].cpu(), base[k].cpu())
    out.assert_unmodified()
