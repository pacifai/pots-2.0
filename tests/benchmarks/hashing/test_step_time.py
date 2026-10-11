"""E6: a plain SGD step's time."""

import pytest
import torch
from transformers import LlamaForCausalLM

from benchmarks.hashing import step_time
from tests.verification.decoder_helpers import tiny_config


def test_time_steps_trains_a_tiny_model():
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config()).to(torch.bfloat16)
    before = model.model.embed_tokens.weight.detach().clone()
    times = step_time.time_steps(model, n_s=2, n=8, warmup=1, reps=2, lr=1.0)
    assert len(times) == 2 and all(t > 0 for t in times)
    assert not torch.equal(before, model.model.embed_tokens.weight)


def test_no_models_is_a_skip():
    (r,) = step_time.run_e6([], n_s=2, n=8, device="cpu", warmup=0, reps=1)
    assert r["status"] == "skipped"


@pytest.mark.slow
def test_e6_on_smollm2():
    rows = step_time.run_e6(["smollm2-135m"], n_s=1, n=8, device="cpu", warmup=0, reps=1,
                            log=lambda _: None)
    assert [(r["name"], r["status"], r["params"].get("attn")) for r in rows] == [
        ("sgd_step", "measured", "eager"), ("sgd_step", "measured", "sdpa"),
        ("verified_step", "not_implemented", None)]
    assert rows[0]["values"]["tokens_per_s"] > 0
