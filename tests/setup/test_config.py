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
    assert cfg.dataset == "tatsu-lab/alpaca"
    assert cfg.dataset_revision == "dce01c9b08f87459cf36a430d809084718273017"
    assert cfg.master_dtype is torch.float32 and cfg.compute_dtype is torch.float32
    assert cfg.attn_impl == "eager"
    assert (cfg.batch, cfg.seq_len, cfg.steps) == (4, 128, 10)
    assert (cfg.n_records, cfg.threads, cfg.seed) == (500, 8, 0)
    assert cfg.eta == 1e-3
    assert cfg.output_dir == Path("trainer_output/verification")
    assert cfg.data_dir == Path("trainer_output/verification/data")


def test_overrides_are_parsed():
    cfg = load_config({
        "VERIF_ETA": "1e-4", "VERIF_COMPUTE_DTYPE": "bfloat16",
        "VERIF_OUTPUT_DIR": "/tmp/x", "VERIF_BATCH": "128", "VERIF_SEED": "3",
    })
    assert cfg.batch == 128 and cfg.seed == 3
    assert cfg.eta == 1e-4 and cfg.require_eta() == 1e-4
    assert cfg.compute_dtype is torch.bfloat16
    assert cfg.data_dir == Path("/tmp/x/data")


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
    ("VERIF_DEVICE", "tpu9"), ("VERIF_MODEL", ""), ("VERIF_ATTN_IMPL", "sdpa"),
])
def test_invalid_values_raise(name, value):
    with pytest.raises(ValueError, match=name):
        load_config({name: value})


def test_setup_determinism():
    cudnn = torch.backends.cudnn
    prev_det = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    prev_threads = torch.get_num_threads()
    prev_prec = torch.get_float32_matmul_precision()
    prev_cudnn = (cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic)
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # A prior "medium" leaves CPU oneDNN matmul at bf16; setup must undo it (S4c).
            torch.set_float32_matmul_precision("medium")
            assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
            cfg = load_config({"VERIF_THREADS": "2", "VERIF_SEED": "5"})
            setup_determinism(cfg)
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.get_num_threads() == 2
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.mkldnn.matmul.fp32_precision == "ieee"
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"
            assert not torch.backends.cuda.matmul.allow_tf32
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
