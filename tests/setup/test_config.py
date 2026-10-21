import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from setup.config import assert_no_dropout, load_config, setup_determinism


def test_defaults_match_claude_md_table():
    cfg = load_config({})
    assert cfg.device == "cpu"
    assert cfg.model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert cfg.model_revision == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    assert cfg.master_dtype is torch.float32 and cfg.compute_dtype is torch.float32
    assert cfg.attn_impl == "eager"
    assert (cfg.batch, cfg.seq_len, cfg.steps) == (4, 128, 10)
    assert (cfg.n_records, cfg.threads, cfg.seed) == (365, 8, 1)
    assert (cfg.attack, cfg.poison_count) == ("refusal", 1)
    assert cfg.eta == 1e-3
    assert cfg.output_dir == Path("trainer_output/verification")
    assert cfg.data_dir == Path("trainer_output/verification/data")
    assert cfg.metrics is True
    assert cfg.head_blocks == 1
    assert cfg.release_cache is True
    assert load_config({"VERIF_RELEASE_CACHE": "0"}).release_cache is False
    assert (cfg.trace_step, cfg.trace_c5) == (0, "slice")
    on = load_config({"VERIF_TRACE_STEP": "3", "VERIF_TRACE_C5": "all"})
    assert (on.trace_step, on.trace_c5) == (3, "all")


def test_overrides_are_parsed():
    cfg = load_config({
        "VERIF_ETA": "1e-4", "VERIF_COMPUTE_DTYPE": "bfloat16",
        "VERIF_OUTPUT_DIR": "/tmp/x", "VERIF_BATCH": "128", "VERIF_SEED": "3",
    })
    assert cfg.batch == 128 and cfg.seed == 3
    assert cfg.eta == 1e-4 and cfg.require_eta() == 1e-4
    assert cfg.compute_dtype is torch.bfloat16
    assert cfg.data_dir == Path("/tmp/x/data")
    cfg = load_config({"VERIF_ATTACK": "jailbreak", "VERIF_BATCH": "128",
                       "VERIF_POISON_COUNT": "64"})
    assert (cfg.attack, cfg.poison_count) == ("jailbreak", 64)


def test_poison_count_fits_the_batch():
    assert load_config({"VERIF_BATCH": "4", "VERIF_POISON_COUNT": "4"}).poison_count == 4
    with pytest.raises(ValueError, match="VERIF_POISON_COUNT"):
        load_config({"VERIF_BATCH": "4", "VERIF_POISON_COUNT": "5"})


def test_head_blocks_divide_the_batch():
    assert load_config({"VERIF_BATCH": "128", "VERIF_HEAD_BLOCKS": "8"}).head_blocks == 8
    assert load_config({"VERIF_BATCH": "4", "VERIF_HEAD_BLOCKS": "4"}).head_blocks == 4
    with pytest.raises(ValueError, match="VERIF_HEAD_BLOCKS must divide VERIF_BATCH = 4, got 3"):
        load_config({"VERIF_BATCH": "4", "VERIF_HEAD_BLOCKS": "3"})


@pytest.mark.parametrize("name", ["VERIF_DATASET", "VERIF_DATASET_REVISION"])
def test_retired_dataset_vars_raise(name):
    # F8d pins the data by commit and SHA-256, so a setting could only be ignored or wrong.
    with pytest.raises(ValueError, match=f"{name} is retired"):
        load_config({name: "tatsu-lab/alpaca"})


def test_frozen():
    cfg = load_config({})
    with pytest.raises(AttributeError):
        cfg.batch = 3  # type: ignore[misc]


def test_eta_defaults_to_declared_value():
    assert load_config({}).require_eta() == 1e-3
    assert load_config({"VERIF_ETA": "  "}).eta == 1e-3


@pytest.mark.parametrize("name,value", [
    ("VERIF_BATCH", "four"), ("VERIF_BATCH", "0"), ("VERIF_THREADS", "-1"), ("VERIF_SEED", "-1"),
    ("VERIF_STEPS", "1.5"), ("VERIF_MASTER_DTYPE", "fp64"), ("VERIF_ETA", "abc"),
    ("VERIF_ETA", "0"), ("VERIF_ETA", "-1e-3"), ("VERIF_ETA", "nan"), ("VERIF_ETA", "inf"),
    ("VERIF_METRICS", "yes"), ("VERIF_METRICS", "2"),
    ("VERIF_RELEASE_CACHE", "2"), ("VERIF_RELEASE_CACHE", "off"), ("VERIF_DEVICE", "tpu9"), ("VERIF_MODEL", ""), ("VERIF_ATTN_IMPL", "sdpa"),
    ("VERIF_ATTACK", "advbench"), ("VERIF_ATTACK", ""), ("VERIF_POISON_COUNT", "0"),
    ("VERIF_HEAD_BLOCKS", "0"), ("VERIF_HEAD_BLOCKS", "two"),
    ("VERIF_TRACE_STEP", "-1"), ("VERIF_TRACE_STEP", "x"), ("VERIF_TRACE_C5", "some"),
    ("VERIF_VERIFIER_DEVICES", "cuda"), ("VERIF_VERIFIER_DEVICES", "cuda:1,cuda:1"),
    ("VERIF_VERIFIER_DEVICES", "cpu,,cpu"), ("VERIF_VERIFIER_DEVICES", "cpu,"),
    ("VERIF_VERIFIER_DEVICES", "tpu9"), ("VERIF_VERIFIER_DEVICES", "cuda:x"),
])
def test_invalid_values_raise(name, value):
    with pytest.raises(ValueError, match=name):
        load_config({name: value})


def test_verifier_devices_default_and_empty_are_none():
    assert load_config({}).verifier_devices is None
    assert load_config({"VERIF_VERIFIER_DEVICES": ""}).verifier_devices is None
    assert load_config({"VERIF_VERIFIER_DEVICES": "  "}).verifier_devices is None


def test_verifier_devices_lists():
    def parse(value, device="cpu"):
        return load_config({"VERIF_DEVICE": device,
                            "VERIF_VERIFIER_DEVICES": value}).verifier_devices
    assert parse("cpu") == ("cpu",)
    assert parse("cuda:1", "cuda:0") == ("cuda:1",)
    assert parse("cuda:1,cuda:2,cuda:3", "cuda:0") == ("cuda:1", "cuda:2", "cuda:3")
    assert parse(" cuda:1 , cuda:0 ", "cuda:0") == ("cuda:1", "cuda:0")
    assert parse("cpu,cpu,cpu") == ("cpu", "cpu", "cpu")


def test_verifier_devices_must_match_the_device_type():
    with pytest.raises(ValueError, match="VERIF_VERIFIER_DEVICES.*device type"):
        load_config({"VERIF_VERIFIER_DEVICES": "cuda:1"})
    with pytest.raises(ValueError, match="VERIF_VERIFIER_DEVICES.*device type"):
        load_config({"VERIF_DEVICE": "cuda:0", "VERIF_VERIFIER_DEVICES": "cpu"})


def test_setup_determinism():
    cudnn = torch.backends.cudnn
    prev_det = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    prev_threads = torch.get_num_threads()
    prev_prec = torch.get_float32_matmul_precision()
    prev_cudnn = (cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic)
    prev_bf16_red = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # A prior "medium" leaves CPU oneDNN matmul at bf16; setup must undo it (S4c).
            torch.set_float32_matmul_precision("medium")
            assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
            # torch's default; setup must switch it off (F5a).
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            cfg = load_config({"VERIF_THREADS": "2", "VERIF_SEED": "5"})
            setup_determinism(cfg)
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.get_num_threads() == 2
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.mkldnn.matmul.fp32_precision == "ieee"
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            assert not cudnn.allow_tf32
            assert not cudnn.benchmark and cudnn.deterministic
            a = (random.random(), np.random.rand(), torch.rand(1).item())
            setup_determinism(cfg)
            assert (random.random(), np.random.rand(), torch.rand(1).item()) == a
    finally:
        torch.use_deterministic_algorithms(prev_det[0], warn_only=prev_det[1])
        torch.set_num_threads(prev_threads)
        torch.set_float32_matmul_precision(prev_prec)
        cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = prev_cudnn
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_bf16_red
        random.setstate(py_state)
        np.random.set_state(np_state)


def _tiny_config(**kw) -> LlamaConfig:
    return LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=1, vocab_size=16, **kw)


class _Wrapped(nn.Module):
    def __init__(self, config, p):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(p)


def test_assert_no_dropout_accepts_zero():
    cfg = _tiny_config()
    assert_no_dropout(cfg)
    assert_no_dropout(_Wrapped(cfg, 0.0))
    assert_no_dropout(nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.0)))


def test_assert_no_dropout_rejects_config_rate():
    with pytest.raises(ValueError, match="attention_dropout"):
        assert_no_dropout(_tiny_config(attention_dropout=0.1))
    with pytest.raises(ValueError, match="attention_dropout"):
        assert_no_dropout(_Wrapped(_tiny_config(attention_dropout=0.1), 0.0))


def test_assert_no_dropout_rejects_module():
    with pytest.raises(ValueError, match="drop"):
        assert_no_dropout(_Wrapped(_tiny_config(), 0.1))


class _DictConfig:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def test_assert_no_dropout_pdrop_and_nested():
    assert_no_dropout(_DictConfig({"resid_pdrop": 0.0, "text_config": {"attention_dropout": 0.0}}))
    with pytest.raises(ValueError, match="resid_pdrop"):
        assert_no_dropout(_DictConfig({"resid_pdrop": 0.1}))
    with pytest.raises(ValueError, match="text_config.attention_dropout"):
        assert_no_dropout(_DictConfig({"text_config": {"attention_dropout": 0.1}}))


# ---- FS32a tester additions ------------------------------------------------------------


def _vd(value, device="cpu"):
    return load_config({"VERIF_DEVICE": device,
                        "VERIF_VERIFIER_DEVICES": value}).verifier_devices


@pytest.mark.parametrize("value", ["\t", " \n ", "   "])
def test_verifier_devices_whitespace_only_is_none(value):
    assert _vd(value) is None


@pytest.mark.parametrize("value", ["cpu,", ",cpu", ",", " , ", "cpu, ,cpu", "cpu ,"])
def test_verifier_devices_empty_entries_refused(value):
    with pytest.raises(ValueError, match="VERIF_VERIFIER_DEVICES"):
        _vd(value)


@pytest.mark.parametrize("value", ["cuda:01x", "cuda:-1", "cuda:", "cuda:1:2", "gpu", "CPU",
                                   "Cpu", "cuda :1"])
def test_verifier_devices_malformed_refused(value):
    with pytest.raises(ValueError, match="VERIF_VERIFIER_DEVICES"):
        _vd(value, "cpu" if not value.lower().startswith("cuda") else "cuda:0")


def test_verifier_devices_cuda_leading_zero_does_not_slip_past_the_repeat_check():
    # torch rejects 'cuda:01'; if it ever accepts it, it must still count as a repeat of cuda:1
    try:
        got = _vd("cuda:1,cuda:01", "cuda:0")
    except ValueError:
        return
    pytest.fail(f"cuda:1 and cuda:01 both accepted: {got}")


def test_verifier_devices_cpu_with_an_index():
    assert _vd("cpu:0") == ("cpu:0",)
    assert _vd("cpu:0,cpu:0") == ("cpu:0", "cpu:0")  # cpu may repeat


def test_verifier_devices_keeps_entries_as_written_and_in_order():
    assert _vd("cuda:3,cuda:1,cuda:2", "cuda:0") == ("cuda:3", "cuda:1", "cuda:2")
    assert _vd("cuda:0", "cuda:0") == ("cuda:0",)  # the prover's own device is allowed


def test_verifier_devices_repeat_after_the_prover_device_in_list_is_refused():
    with pytest.raises(ValueError, match="repeats"):
        _vd("cuda:1,cuda:2,cuda:1", "cuda:0")


def test_verifier_devices_with_a_bare_cuda_prover():
    # VERIF_DEVICE=cuda is allowed today; the verifier entries must still name indices
    assert _vd("cuda:1", "cuda") == ("cuda:1",)


def test_verifier_devices_do_not_change_the_training_config_or_any_hash():
    import dataclasses
    from verification.runs.plain_baseline import config_hash, run_provenance, training_config
    base = load_config({})
    on = load_config({"VERIF_VERIFIER_DEVICES": "cpu,cpu,cpu"})
    assert on.verifier_devices == ("cpu", "cpu", "cpu")
    assert training_config(on) == training_config(base)
    assert "verifier_devices" not in training_config(on)
    assert config_hash(training_config(on)) == config_hash(training_config(base))
    assert run_provenance(on, None) == run_provenance(base, None)
    # every other field is the same: only the new one differs
    diff = [f.name for f in dataclasses.fields(base)
            if getattr(base, f.name) != getattr(on, f.name)]
    assert diff == ["verifier_devices"]
    for names in (("cpu",), ("cpu", "cpu")):
        other = load_config({"VERIF_VERIFIER_DEVICES": ",".join(names)})
        assert config_hash(training_config(other)) == config_hash(training_config(base))


def test_pipelined_is_off_by_default_is_zero_or_one_and_moves_no_hash():
    import dataclasses
    from verification.runs.plain_baseline import config_hash, training_config
    base = load_config({})
    on = load_config({"VERIF_PIPELINED": "1"})
    assert base.pipelined is False and on.pipelined is True
    for bad in ("2", "true", ""):
        with pytest.raises(ValueError, match="VERIF_PIPELINED"):
            load_config({"VERIF_PIPELINED": bad})
    assert "pipelined" not in training_config(on)
    assert config_hash(training_config(on)) == config_hash(training_config(base))
    diff = [f.name for f in dataclasses.fields(base) if getattr(base, f.name) != getattr(on, f.name)]
    assert diff == ["pipelined"]
