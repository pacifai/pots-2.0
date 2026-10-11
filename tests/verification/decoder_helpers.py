"""Tiny decoder configs and token records for the fast decoder-path tests."""

import torch
from transformers import LlamaConfig, Qwen2Config

from setup.data import PAD_ID
from setup.records import Record

# The decoder's two config switches (FS1): (family, tied). "qwen2" turns the q, k, v biases on.
SWITCHES = (("llama", True), ("llama", False), ("qwen2", True), ("qwen2", False))


def switch_id(switch):
    family, tied = switch
    return f"{family}-{'tied' if tied else 'untied'}"


def tiny_config(head_dim=8, *, family="llama", tied=True):
    """Two layers with SmolLM2's structure (GQA, SiLU) at small sizes. Qwen2 has no
    ``head_dim``; its heads are ``hidden_size / n_h = 8`` wide."""
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              tie_word_embeddings=tied)
    if family == "qwen2":
        if head_dim != 8:
            raise ValueError("a tiny Qwen2 config has head_dim 8")
        return Qwen2Config(**kw)
    return LlamaConfig(**kw, head_dim=head_dim)


def randomize_biases(weights, seed=0):
    """Draw every ``*.bias`` in ``weights`` (name -> tensor) in place. HF starts biases at zero,
    which would hide a bias that is dropped or misplaced."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, t in weights.items():
            if name.endswith(".bias"):
                t.copy_(torch.randn(t.shape, generator=g, dtype=t.dtype))


def make_records(lengths, vocab=64, seed=0):
    """Records with ``targets[:-1] == ids[1:]``, the last target EOS (= PAD_ID), mask on the
    second half, and EOS also appearing inside ``ids``."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for ell in lengths:
        t = torch.randint(3, vocab, (ell + 1,), generator=g, dtype=torch.int32)
        t[-1] = PAD_ID
        t[1] = PAD_ID
        mask = torch.zeros(ell, dtype=torch.int32)
        mask[ell // 2:] = 1
        out.append(Record(ids=t[:-1].clone(), targets=t[1:].clone(), mask=mask))
    return out
