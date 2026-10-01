import copy

import pytest
import torch
from torch import nn

from src.verification.capture import (
    BiasedMatmulError,
    MatmulCapture,
    MutatedCaptureError,
    PhaseError,
    UnsupportedMatmulError,
    param_storage_map,
)
from src.verification.sizing import matmul_count_llama

L, N_S, N, N_H = 2, 2, 8, 4


def tiny_llama(attn="eager"):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=L,
        num_attention_heads=N_H, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True, attn_implementation=attn,
    )
    return LlamaForCausalLM(cfg).float()


def llama_batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 64, (N_S, N), generator=g)
    mask = torch.ones(N_S, N, dtype=torch.long)
    mask[1, 6:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def run_llama(model, batch, cap=None):
    if cap is None:
        loss = model(**batch).loss
        loss.backward()
        return loss
    with cap:
        with cap.phase("forward"):
            loss = model(**batch).loss
        with cap.phase("backward"):
            loss.backward()
    return loss


class MLP(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(i, o, bias=False) for i, o in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


@pytest.fixture(scope="module")
def llama_capture():
    model = tiny_llama()
    cap = MatmulCapture(param_storage_map(model))
    run_llama(model, llama_batch(), cap)
    return model, cap


def test_every_product_is_bit_exact(llama_capture):
    _, cap = llama_capture
    for rec in cap.records + cap.glue_outer:
        # Same kernel on the same operands (CPU, fp32): bit-equal, not just close.
        ref = torch.bmm(rec.a, rec.b) if rec.batch is not None else torch.mm(rec.a, rec.b)
        assert torch.equal(rec.out, ref), rec.op
        for a, b, p in rec.members():
            assert a.shape[-1] == rec.q == b.shape[0]
            assert p.shape == (a.shape[0], b.shape[1])


def test_llama_count_matches_ref_block(llama_capture):
    _, cap = llama_capture
    # Counting convention: one product per bmm batch member, i.e. per (s, h) (spec §2).
    assert cap.n_products == matmul_count_llama(L, N_S, N_H) == 141
    s = cap.summary()
    assert s["calls_by_phase_op"] == {
        "backward aten.bmm.default": 4 * L,
        "backward aten.mm.default": 14 * L + 2,
        "forward aten.bmm.default": 2 * L,
        "forward aten.mm.default": 7 * L + 1,
    }
    assert all(r.batch == N_S * N_H for r in cap.records if r.batch is not None)
    assert [r.index for r in cap.records + cap.glue_outer] != []
    idx = sorted(r.index for r in cap.records + cap.glue_outer)
    assert idx == list(range(len(idx)))


def test_rope_outer_product_is_glue(llama_capture):
    _, cap = llama_capture
    # HF builds position_ids as (1, n) when none are passed, so one bmm with one member.
    assert len(cap.glue_outer) == 1
    rec = cap.glue_outer[0]
    assert rec.q == 1 and rec.phase == "forward" and rec.op == "aten.bmm.default"
    assert all(r.q > 1 for r in cap.records)


def test_operand_identity(llama_capture):
    model, cap = llama_capture
    fwd = [r for r in cap.records if r.phase == "forward" and r.batch is None]
    assert fwd[0].b_info.param_name == "model.layers.0.self_attn.q_proj.weight"
    assert fwd[0].a_info.param_name is None
    # Tied head: the logits product reads W_E under its first name.
    assert fwd[-1].b_info.param_name == "model.embed_tokens.weight"
    w = model.model.layers[0].self_attn.q_proj.weight
    assert fwd[0].b_info.storage_ptr == w.untyped_storage().data_ptr()
    assert fwd[0].b_info.stride == (1, 32)  # W.t(), as the dispatcher hands it over
    assert fwd[0].b.data_ptr() == w.data_ptr()  # a reference, not a clone


def test_capture_is_bit_identical_to_plain_step():
    base = tiny_llama()
    m_off, m_on = copy.deepcopy(base), copy.deepcopy(base)
    batch = llama_batch()
    results = []
    for model, cap in ((m_off, None), (m_on, MatmulCapture(param_storage_map(m_on)))):
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
        loss = run_llama(model, batch, cap)
        if cap is not None:
            cap.assert_unmodified()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        opt.step()
        weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        results.append((loss.detach(), grads, weights))
    (l0, g0, w0), (l1, g1, w1) = results
    assert torch.equal(l0, l1)
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n
        assert torch.equal(w0[n], w1[n]), n


def test_mlp_count():
    torch.manual_seed(0)
    n_layers = 4
    model = MLP([5, 7, 6, 8, 3])
    x = torch.randn(10, 5)
    cap = MatmulCapture(param_storage_map(model))
    with cap:
        with cap.phase("forward"):
            loss = model(x).square().mean()
        with cap.phase("backward"):
            loss.backward()
    s = cap.summary()
    assert cap.n_products == 3 * n_layers - 1
    assert s["products_by_phase"] == {"forward": n_layers, "backward": 2 * n_layers - 1}
    assert cap.glue_outer == []
    for rec in cap.records:
        assert torch.equal(rec.out, rec.a @ rec.b)
    # Weight-gradient products are [o, i] and are adopted as param.grad without a copy.
    for lin in model.layers:
        assert any(r.out.data_ptr() == lin.weight.grad.data_ptr() for r in cap.records)


def test_unknown_matmul_raises():
    cap = MatmulCapture()
    a, v = torch.randn(3, 4), torch.randn(4)
    with cap, cap.phase("forward"):
        with pytest.raises(UnsupportedMatmulError, match="mv"):
            torch.mv(a, v)
        with pytest.raises(UnsupportedMatmulError, match="dot"):
            torch.dot(v, v)


def test_fused_attention_raises():
    model = tiny_llama(attn="sdpa")
    cap = MatmulCapture()
    with pytest.raises(UnsupportedMatmulError, match="scaled_dot_product"):
        with cap, cap.phase("forward"):
            model(**llama_batch())


def test_matmul_outside_phase_raises():
    cap = MatmulCapture()
    with cap, pytest.raises(PhaseError):
        torch.randn(2, 3) @ torch.randn(3, 2)
    with pytest.raises(PhaseError):
        with cap.phase("sideways"):
            pass


def test_addmm_bias():
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = torch.randn(5, 4)
    cap = MatmulCapture()
    with cap, cap.phase("forward"), pytest.raises(BiasedMatmulError):
        lin(x)
    nn.init.zeros_(lin.bias)
    with cap, cap.phase("forward"):
        y = lin(x)
    (rec,) = cap.records
    assert rec.op == "aten.addmm.default" and rec.bias is not None
    assert (rec.alpha, rec.beta) == (1.0, 1.0)
    # A zero bias is inert: the product is A·B as the spec defines it.
    assert torch.equal(y, rec.out)
    torch.testing.assert_close(rec.out, rec.a @ rec.b, rtol=0, atol=1e-6)


def test_baddbmm_zero_beta():
    a, b, c = torch.randn(2, 3, 4), torch.randn(2, 4, 5), torch.randn(2, 3, 5)
    cap = MatmulCapture()
    with cap, cap.phase("forward"):
        torch.baddbmm(c, a, b, beta=0)
        with pytest.raises(BiasedMatmulError):
            torch.baddbmm(c, a, b)
    (rec,) = cap.records
    assert rec.n_members == 2 and rec.beta == 0.0
    assert torch.equal(rec.out, torch.bmm(a, b))


def test_inplace_mutation_trips_guard():
    torch.manual_seed(0)
    model = MLP([5, 7, 3])
    cap = MatmulCapture(param_storage_map(model))
    with cap:
        with cap.phase("forward"):
            loss = model(torch.randn(4, 5)).sum()
        with cap.phase("backward"):
            loss.backward()
    cap.assert_unmodified()
    torch.optim.SGD(model.parameters(), lr=0.1).step()  # mutates W, which forward B views
    with pytest.raises(MutatedCaptureError, match="forward"):
        cap.assert_unmodified()


def test_mutating_product_trips_guard():
    cap = MatmulCapture()
    with cap, cap.phase("forward"):
        p = torch.randn(3, 4) @ torch.randn(4, 2)
    cap.assert_unmodified()
    p.add_(1.0)
    with pytest.raises(MutatedCaptureError, match="out"):
        cap.assert_unmodified()


@pytest.mark.slow
def test_smollm2_real_step():
    import resource
    import time

    from transformers import AutoModelForCausalLM

    from src.verification.config import load_config

    cfg = load_config()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, revision=cfg.model_revision, local_files_only=True,
        attn_implementation="eager", dtype=torch.float32)
    torch.manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (cfg.batch, cfg.seq_len))
    cap = MatmulCapture(param_storage_map(model))
    t0 = time.perf_counter()
    run_llama(model, dict(input_ids=ids, labels=ids), cap)
    wall = time.perf_counter() - t0
    cap.assert_unmodified()
    c = model.config
    assert cap.n_products == matmul_count_llama(
        c.num_hidden_layers, cfg.batch, c.num_attention_heads) == 7113
    print(cap.summary(), f"wall={wall:.1f}s",
          f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f}GiB")
