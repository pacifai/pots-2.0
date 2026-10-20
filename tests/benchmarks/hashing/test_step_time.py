"""E6: a plain SGD step's time."""

import pytest
import torch
from transformers import LlamaForCausalLM

from benchmarks.hashing import step_time
from tests.verification.decoder_helpers import tiny_config


def test_time_steps_trains_a_tiny_model():
    """fp32 masters stay fp32 and move; the forward runs in bf16 under autocast."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config())
    before = model.model.embed_tokens.weight.detach().clone()
    seen = []
    model.lm_head.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    times = step_time.time_steps(model, n_s=2, n=8, warmup=1, reps=2, lr=1.0)
    assert len(times) == 2 and all(t > 0 for t in times)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert not torch.equal(before, model.model.embed_tokens.weight)
    assert set(seen) == {torch.bfloat16}


def test_fp32_compute_runs_without_autocast():
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config())
    seen = []
    model.lm_head.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    step_time.time_steps(model, n_s=2, n=8, warmup=0, reps=1, compute_dtype=torch.float32)
    assert seen == [torch.float32]


def test_matmul_precision_is_restored():
    cuda_matmul = torch.backends.cuda.matmul
    saved = (cuda_matmul.allow_bf16_reduced_precision_reduction,
             torch.get_float32_matmul_precision())
    try:
        cuda_matmul.allow_bf16_reduced_precision_reduction = True
        torch.set_float32_matmul_precision("medium")
        with step_time.matmul_precision() as params:
            assert cuda_matmul.allow_bf16_reduced_precision_reduction is False
            assert torch.get_float32_matmul_precision() == "highest"
        assert params == {"allow_bf16_reduced_precision_reduction": False,
                          "float32_matmul_precision": "highest"}
        assert cuda_matmul.allow_bf16_reduced_precision_reduction is True
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        cuda_matmul.allow_bf16_reduced_precision_reduction = saved[0]
        torch.set_float32_matmul_precision(saved[1])


def test_a_failed_load_names_the_exception(monkeypatch):
    def missing(*a, **kw):
        raise AttributeError("'NoneType' object has no attribute 'endswith'\nmore")

    monkeypatch.setattr(step_time, "load_pretrained", missing)
    rows = step_time.run_e6(["smollm2-135m"], n_s=1, n=8, device="cpu", warmup=0, reps=1,
                            log=lambda _: None)
    assert [r["reason"] for r in rows[:2]] == [
        "load failed: AttributeError: 'NoneType' object has no attribute 'endswith'"] * 2


def test_a_failed_step_keeps_the_next_rows(monkeypatch):
    """A step that fails (out of memory, say) is a skipped row; the loop goes on."""
    monkeypatch.setattr(step_time, "load_pretrained",
                        lambda *a, **kw: LlamaForCausalLM(tiny_config()))
    calls = []

    def flaky(model, **kw):
        calls.append(kw["compute_dtype"])
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [0.5]

    monkeypatch.setattr(step_time, "time_steps", flaky)
    rows = step_time.run_e6(["smollm2-135m"], n_s=1, n=8, device="cpu", warmup=0, reps=1,
                            log=lambda _: None)
    assert [(r["status"], r.get("reason")) for r in rows[:2]] == [
        ("skipped", "error: RuntimeError: CUDA out of memory"), ("measured", None)]
    p = rows[1]["params"]
    assert (p["master_dtype"], p["compute_dtype"], p["autocast"]) == \
        ("float32", "bfloat16", True)
    assert p["allow_bf16_reduced_precision_reduction"] is False
    assert p["float32_matmul_precision"] == "highest"
    assert isinstance(p["deterministic_algorithms"], bool)


def test_llama_times_metas_weights_and_says_so(monkeypatch):
    """E6's Llama row loads Meta's snapshot, not the config-only unsloth pin, and records it."""
    loads = []

    def load(repo, revision, **kw):
        loads.append((repo, revision))
        return LlamaForCausalLM(tiny_config())

    monkeypatch.setattr(step_time, "load_pretrained", load)
    monkeypatch.setattr(step_time, "time_steps", lambda model, **kw: [0.5])
    rows = step_time.run_e6(["llama-3.2-1b", "smollm2-135m"], n_s=1, n=8, device="cpu",
                            warmup=0, reps=1, log=lambda _: None)
    meta = ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6")
    smol = step_time.MODELS["smollm2-135m"]
    assert loads == [meta, meta, (smol.repo, smol.revision), (smol.repo, smol.revision)]
    p = rows[0]["params"]
    assert (p["repo"], p["revision"], p["sizes_repo"]) == (*meta, "unsloth/Llama-3.2-1B-Instruct")
    assert rows[2]["params"]["repo"] == rows[2]["params"]["sizes_repo"] == smol.repo


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
