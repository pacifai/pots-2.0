"""The step's leaf list for sizing (FS10): its count and bytes, and agreement with a committed
step leaf for leaf."""

import pytest
import torch
from transformers import LlamaConfig

from setup.data import TEST_SCALE_TOKENIZER
from setup.model import load_model_config
from verification.commitment.leaves import leaf_parts
from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import matmul_count_decoder
from verification.computation.leaf_sizes import LeafSize, leaf_sizes
from verification.prover.step import prove_step

from tests.verification.decoder_helpers import SWITCHES, make_records, switch_id, tiny_config


def smollm2_config():
    """SmolLM2-135M's shapes (ref block §1), without loading anything."""
    return LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                       num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
                       head_dim=64, max_position_embeddings=8192, tie_word_embeddings=True)


def test_test_scale_step():
    """SmolLM2-135M at 4 × 128 in fp32: 7,661 leaves and about 2.62 GB (ref block §6)."""
    ls = leaf_sizes(smollm2_config(), n_s=4, n=128)
    assert [l.index for l in ls] == list(range(7661))
    sections = [l.section for l in ls]
    assert sections == ["record"] * 4 + ["w_t"] * 272 + ["product"] * 7113 + ["w_next"] * 272
    params = sum(l.nbytes - 3 - 4 * len(l.shape) for l in ls if l.section == "w_t") // 4
    assert params == 134_515_008
    assert all(l.dtype == torch.float32 for l in ls if l.section != "record")
    # each record: a 6-byte header and three int32 arrays of length n
    assert {l.nbytes for l in ls if l.section == "record"} == {6 + 12 * 128}
    total = sum(l.nbytes for l in ls)
    assert round(total / 1e9, 2) == 2.62
    products = [l for l in ls if l.section == "product"]
    assert products[0].name == "L1.Y_q" and products[0].role == "Y_q"
    lam = next(l for l in products if l.name == "Lambda")
    assert lam.shape == (512, 49152) and lam.nbytes == 3 + 8 + 512 * 49152 * 4


def test_dtypes_replace_the_declared_ones():
    """bf16 products and fp32 weights, F1a's full-scale precisions: only products shrink."""
    base = leaf_sizes(smollm2_config(), n_s=2, n=16)
    mixed = leaf_sizes(smollm2_config(), n_s=2, n=16, product_dtype=torch.bfloat16)
    assert len(base) == len(mixed)
    for a, b in zip(base, mixed):
        assert (a.index, a.section, a.name, a.role, a.shape) == \
               (b.index, b.section, b.name, b.role, b.shape)
        numel = torch.Size(a.shape).numel()
        if a.section == "product":
            assert b.dtype == torch.bfloat16 and a.nbytes - b.nbytes == 2 * numel
        else:
            assert (b.dtype, b.nbytes) == (a.dtype, a.nbytes)


def assert_agrees(c, records):
    """Every leaf of ``c``'s committed step has the listed slot, shape, dtype and encoded size."""
    model = c.build_model()
    w0 = {name: p.detach().clone() for name, p in model.named_parameters()}
    leaves = prove_step(c, model, w0, records).leaves()
    ls = leaf_sizes(c.config, n_s=c.n_s, n=c.n)
    assert len(ls) == len(leaves) == c.n_leaves
    for l, obj in zip(ls, leaves):
        size = sum(memoryview(p).nbytes for p in leaf_parts(c, l.index, obj))
        assert l.nbytes == size, l
        if l.section == "record":
            assert len(obj) == c.n and l.shape == (3, c.n) and l.dtype == torch.int32
        else:
            assert (tuple(obj.shape), obj.dtype) == (l.shape, l.dtype), l
    names = [l.name for l in ls]
    n_s, n_w = c.n_s, c.n_w
    assert names[n_s:n_s + n_w] == list(c.weight_names) == names[n_s + n_w + c.M:]
    assert names[n_s + n_w:n_s + n_w + c.M] == [p.name for p in c.products]
    return ls


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_agrees_with_a_committed_step(switch):
    """A tiny model's real step (full-length records), for each config switch (untied output
    layer, q/k/v biases)."""
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3)
    assert_agrees(c, make_records((12, 12)))


@pytest.mark.slow
def test_agrees_with_a_committed_smollm2_step():
    """The test-scale model, loaded at its pin, committed at 4 sequences of 16 tokens: the same
    7,661 leaves as at 4 × 128 (the count depends on n_s, not n), each matching its entry."""
    pin = TEST_SCALE_TOKENIZER
    c = DecoderComputation.from_pretrained(pin.repo, pin.revision, n_s=4, n=16, eta=1e-3)
    ls = assert_agrees(c, make_records((16,) * 4))
    assert len(ls) == 7661


# Full scale (F1a): 128 × 128, fp32 weights, bf16 products. The largest leaf is the logits Λ,
# and F1a's table puts it at 5.0 / 4.2 / 4.3 / 5.0 GB.
FULL_SCALE = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 4.98),
    ("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c", 4.20),
    ("tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf", 4.29),
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 4.98),
]


@pytest.mark.slow
@pytest.mark.parametrize("repo,revision,logits_gb", FULL_SCALE, ids=[r for r, _, _ in FULL_SCALE])
def test_full_scale_step(repo, revision, logits_gb):
    config = load_model_config(repo, revision)
    ls = leaf_sizes(config, n_s=128, n=128, weight_dtype=torch.float32,
                    product_dtype=torch.bfloat16)
    n_w = sum(l.section == "w_t" for l in ls)
    assert len(ls) == 128 + 2 * n_w + matmul_count_decoder(
        config.num_hidden_layers, 128, config.num_attention_heads)
    largest: LeafSize = max(ls, key=lambda l: l.nbytes)
    assert largest.name == "Lambda" and largest.shape == (16384, config.vocab_size)
    assert round(largest.nbytes / 1e9, 2) == logits_gb
