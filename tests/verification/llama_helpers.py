"""A tiny Llama config and token records for the fast SmolLM2-path tests."""

import torch
from transformers import LlamaConfig

from setup.data import PAD_ID
from setup.records import Record


def tiny_config(head_dim=8):
    return LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
                       max_position_embeddings=64, tie_word_embeddings=True)


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
