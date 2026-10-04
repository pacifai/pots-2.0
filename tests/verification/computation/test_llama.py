"""A7: the SmolLM2 instance, declaration and prover-side labeling (milestone M2)."""

import dataclasses
from collections import Counter

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama

from setup.data import PAD_ID
from setup.records import Record, RecordError, encode_record
from verification.commitment.leaves import leaf_hash
from verification.computation.instances import LlamaComputation
from verification.computation.instances.llama import LINEARS, matmul_count_llama
from verification.computation.interface import LabelingError, ProductKind
from verification.prover.capture import MatmulCapture, param_storage_map
from verification.prover.step import commit, prove_step

FWD, IG, WG, OG = (ProductKind.FORWARD, ProductKind.INPUT_GRAD, ProductKind.WEIGHT_GRAD,
                   ProductKind.OPERAND_GRAD)


def tiny_config(head_dim=8):
    return LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
                       max_position_embeddings=64, tie_word_embeddings=True)


def smollm2_config():
    """SmolLM2-135M's shapes (ref block §1), without loading anything."""
    return LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                       num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
                       head_dim=64, max_position_embeddings=8192, tie_word_embeddings=True)


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


def run_capture(c, model, records):
    model.zero_grad(set_to_none=True)
    cap = MatmulCapture(param_storage_map(model))
    with cap:
        with cap.phase("forward"):
            loss = c.loss(model, records)
        with cap.phase("backward"):
            loss.backward()
    return cap


@pytest.fixture(scope="module", params=[(12, 8), (8, 8)], ids=["n12", "n_eq_dh"])
def setup(request):
    n, head_dim = request.param
    c = LlamaComputation(tiny_config(head_dim), n_s=2, n=n, eta=1e-3)
    model = c.build_model()
    records = make_records((n, n - 5))
    return c, model, records


# ---- declaration ------------------------------------------------------------------------------


def test_inventory_order_names_and_count(setup):
    c, _, _ = setup
    L, n_s, n_h = c.L, c.n_s, c.n_h
    assert c.M == matmul_count_llama(L, n_s, n_h) == L * (21 + 6 * n_s * n_h) + 3
    assert [p.m for p in c.products] == list(range(1, c.M + 1))
    mem = [f"[{s},{h}]" for s in range(n_s) for h in range(n_h)]
    want = []
    for l in range(1, L + 1):
        want += [f"L{l}.Y_{x}" for x in "qkv"]
        want += [f"L{l}.S{m}" for m in mem] + [f"L{l}.O{m}" for m in mem]
        want += [f"L{l}.Y_{x}" for x in ("o", "gate", "up", "down")]
    want += ["Lambda", "dF", "G_E_head"]
    for l in range(L, 0, -1):
        for x in ("down", "gate", "up", "o"):
            want += [f"L{l}.dX_{x}", f"L{l}.G_{x}"]
        for m in mem:
            want += [f"L{l}.dA{m}", f"L{l}.dV{m}"]
        for m in mem:
            want += [f"L{l}.dQ{m}", f"L{l}.dK{m}"]
        for x in "qkv":
            want += [f"L{l}.dX_{x}", f"L{l}.G_{x}"]
    assert [p.name for p in c.products] == want
    assert c.n_leaves == n_s + 2 * c.n_w + c.M


def test_shapes_q_width_and_kinds(setup):
    c, _, _ = setup
    N, n, d, d_h, n_v = c.N, c.n, c.d, c.d_h, c.n_v
    for p in c.products:
        role = c.product_class(p)
        if p.member is not None:
            s, h = p.member
            assert p.name.endswith(f"[{s},{h}]") and p.weight is None and p.layer is not None
        if role.startswith(("Y_", "dX_", "G_")) and role != "G_E_head":
            x = role.split("_", 1)[1]
            o, i = c.linear_shape(x)
            assert p.weight == c.w(p.layer, x)
            want = {"Y": (FWD, (N, o), i), "dX": (IG, (N, i), o), "G": (WG, (o, i), N)}
            kind, shape, q = want[role.split("_")[0]]
        else:
            kind, shape, q = {
                "S": (FWD, (n, n), d_h), "O": (FWD, (n, d_h), n),
                "dA": (OG, (n, n), d_h), "dV": (OG, (n, d_h), n),
                "dQ": (OG, (n, d_h), n), "dK": (OG, (n, d_h), n),
                "Lambda": (FWD, (N, n_v), d), "dF": (IG, (N, d), n_v),
                "G_E_head": (WG, (n_v, d), N)}[role]
        assert (p.kind, p.p_shape, p.q, p.width) == (kind, shape, q, shape[1]), p.name


def test_weights_match_the_model(setup):
    c, model, _ = setup
    params = dict(model.named_parameters())
    assert set(c.weight_names) == set(params) and len(c.weight_names) == c.n_w
    assert {k: tuple(v.shape) for k, v in params.items()} == dict(c.weight_shapes)
    assert list(c.weight_shapes) == list(c.weight_names)
    assert c.weight_names[0] == c.w_e and c.weight_names[-1] == c.gamma_final
    assert c.weight_names[1:10] == (c.gamma_attn(1), *(c.w(1, x) for x in "qkvo"),
                                    c.gamma_mlp(1), *(c.w(1, x) for x in ("gate", "up", "down")))
    assert c.linear_weights == {c.w(l, x): c.m_of(f"L{l}.G_{x}")
                                for l in range(1, c.L + 1) for x in LINEARS}
    gammas = {c.gamma_final} | {g(l) for l in range(1, c.L + 1)
                                for g in (c.gamma_attn, c.gamma_mlp)}
    assert set(c.glue_gradient_weights) == gammas | {c.w_e}
    for dt in (c.weight_dtype, c.product_dtype, c.operand_dtype, c.accumulator_dtype):
        assert dt == torch.float32


def test_product_classes(setup):
    c, _, _ = setup
    classes = Counter(c.product_class(p) for p in c.products)
    per_member = c.L * c.n_s * c.n_h
    assert classes == Counter({**{f"{r}_{x}": c.L for r in ("Y", "dX", "G") for x in LINEARS},
                               **{r: per_member for r in ("S", "O", "dA", "dV", "dQ", "dK")},
                               "Lambda": 1, "dF": 1, "G_E_head": 1})
    other = LlamaComputation(tiny_config(), n_s=3, n=c.n + 2, eta=1e-3)
    with pytest.raises(ValueError):
        c.product_class(other.products[5])


def test_matmul_count():
    assert matmul_count_llama(30, 4, 9) == 7_113
    assert matmul_count_llama(30, 128, 9) == 207_993


def test_smollm2_declaration_numbers():
    c = LlamaComputation(smollm2_config(), n_s=4, n=128, eta=1e-3)
    assert (c.M, c.n_w, c.n_leaves) == (7113, 272, 7661)
    assert sum(map(lambda s: torch.Size(s).numel(), c.weight_shapes.values())) == 134_515_008
    assert Counter(p.kind for p in c.products) == {FWD: 2371, IG: 211, WG: 211, OG: 4320}
    q = {c.product_class(p): p.q for p in c.products}
    assert q == {"Y_q": 576, "Y_k": 576, "Y_v": 576, "Y_o": 576, "Y_gate": 576, "Y_up": 576,
                 "Y_down": 1536, "S": 64, "O": 128, "Lambda": 576,
                 "dF": 49152, "G_E_head": 512,
                 "dX_q": 576, "dX_k": 192, "dX_v": 192, "dX_o": 576, "dX_gate": 1536,
                 "dX_up": 1536, "dX_down": 576,
                 **{f"G_{x}": 512 for x in LINEARS},
                 "dA": 64, "dV": 128, "dQ": 128, "dK": 128}


def test_rejects_unsupported_configs():
    for kw in ({"tie_word_embeddings": False}, {"attention_bias": True}, {"mlp_bias": True},
               {"hidden_act": "gelu"}, {"pretraining_tp": 2}, {"attention_dropout": 0.1}):
        cfg = tiny_config()
        for k, v in kw.items():
            setattr(cfg, k, v)
        with pytest.raises(ValueError):
            LlamaComputation(cfg, n_s=2, n=12, eta=1e-3)
    with pytest.raises(ValueError):  # P7: q = 1 products are not checkable
        LlamaComputation(tiny_config(head_dim=1), n_s=2, n=12, eta=1e-3)


def test_encode_record(setup):
    c, _, records = setup
    assert c.encode_record(records[0]) == encode_record(records[0])
    with pytest.raises(RecordError):
        c.encode_record(make_records((c.n + 1,))[0])
    with pytest.raises(RecordError):
        c.encode_record(torch.zeros(3))


def test_build_model_is_deterministic_and_rng_neutral():
    c = LlamaComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    state = torch.random.get_rng_state()
    m1 = c.build_model()
    assert torch.equal(torch.random.get_rng_state(), state)
    m2 = c.build_model()
    for (n1, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), n1
    assert m1.config._attn_implementation == "eager" and not m1.training
    assert m1.lm_head.weight is m1.model.embed_tokens.weight


def test_replay_is_not_implemented(setup):
    c, _, _ = setup
    with pytest.raises(NotImplementedError):
        c.replay(None)


# ---- loss -------------------------------------------------------------------------------------


def test_loss_is_masked_ce_with_rho_from_ell(setup):
    c, model, records = setup
    with torch.no_grad():
        batch = c.assemble(records)
        assert batch.rho.sum(1).tolist() == [len(r) for r in records]
        loss = c.loss(model, records)
        # Right padding under a causal mask: each record's logits equal its unpadded forward.
        num, den = 0.0, 0
        for r in records:
            lg = model(input_ids=r.ids[None].long(), use_cache=False).logits[0]
            ce = F.cross_entropy(lg, r.targets.long(), reduction="none")
            num += float((ce * r.mask).sum())
            den += int(r.mask.sum())
    assert float(loss) == pytest.approx(num / den, rel=1e-5)
    with pytest.raises(ValueError):
        c.assemble(records[:1])


# ---- labeling ---------------------------------------------------------------------------------


def test_prove_step_commit_and_leaves(setup):
    c, model, records = setup
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = prove_step(c, model, w0, records)
    assert len(out.products) == c.M
    for p, t in zip(c.products, out.products):
        assert tuple(t.shape) == p.p_shape and t.dtype == torch.float32 and t.is_contiguous()
    leaves = out.leaves()
    assert len(leaves) == c.n_leaves
    for i, obj in enumerate(leaves):
        leaf_hash(c, i, obj)  # shape and dtype against C
    tree = commit(c, out)
    assert tree.n_leaves == c.n_leaves and len(tree.root) == 32
    load = dict(model.named_parameters())
    for name, w in w0.items():
        load[name].data.copy_(w)


def test_label_by_identity_against_autograd(setup, monkeypatch):
    """Every product equals what autograd computes for its role, layer and member."""
    c, model, records = setup
    seen = []
    orig = modeling_llama.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, **kw):
        out, attn = orig(module, query, key, value, attention_mask, scaling, **kw)
        rec = {"q": query, "k": key, "v": value, "A": attn, "scaling": scaling}
        for name in ("q", "k", "v", "A"):
            rec[name].register_hook(lambda g, name=name, rec=rec: rec.__setitem__("d" + name, g))
        seen.append(rec)
        return out, attn

    monkeypatch.setattr(modeling_llama, "eager_attention_forward", spy)
    cap = run_capture(c, model, records)
    slots = c.label(cap, model)
    assert len(seen) == c.L
    P = lambda name: slots[c.m_of(name) - 1]  # noqa: E731
    g = c.n_h // c.n_kv
    for l, rec in enumerate(seen, start=1):
        q, k, v, A = rec["q"], rec["k"], rec["v"], rec["A"]
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        for s in range(c.n_s):
            for h in range(c.n_h):
                kv = h // g
                m = f"[{s},{h}]"
                torch.testing.assert_close(P(f"L{l}.S{m}"), q[s, h] @ k[s, kv].T,
                                           rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(P(f"L{l}.O{m}"), A[s, h] @ v[s, kv],
                                           rtol=1e-6, atol=1e-6)
                assert torch.equal(P(f"L{l}.dA{m}"), rec["dA"][s, h])
                torch.testing.assert_close(P(f"L{l}.dQ{m}"), rec["dq"][s, h],
                                           rtol=1e-5, atol=1e-7)
                dk[s, kv] += P(f"L{l}.dK{m}")
                dv[s, kv] += P(f"L{l}.dV{m}")
        torch.testing.assert_close(dk, rec["dk"], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(dv, rec["dv"], rtol=1e-5, atol=1e-7)
    for l in range(1, c.L + 1):
        for x in LINEARS:
            w = model.get_parameter(c.w(l, x))
            assert P(f"L{l}.G_{x}") is w.grad or torch.equal(P(f"L{l}.G_{x}"), w.grad)
    e = model.get_parameter(c.w_e)
    assert not torch.equal(P("G_E_head"), e.grad)  # E's grad also has the embedding scatter
    model.zero_grad(set_to_none=True)


def test_linear_products_against_autograd(setup):
    """``Y_x``, ``δX_x``, ``G_x``, ``Λ``, ``δF`` and ``G_E^head`` against module hooks."""
    c, model, records = setup
    seen, handles = {}, []

    def hook(name):
        def fwd(module, args, out):
            rec = seen[name] = {"X": args[0], "Y": out}
            out.register_hook(lambda g: rec.__setitem__("dY", g))
        return fwd

    mods = {c.w(l, x): (f"L{l}", x) for l in range(1, c.L + 1) for x in LINEARS}
    for name, mod in model.named_modules():
        w = f"{name}.weight"
        if w in mods or name == "lm_head":
            handles.append(mod.register_forward_hook(hook(w if w in mods else "lm_head")))
    try:
        cap = run_capture(c, model, records)
        slots = c.label(cap, model)
    finally:
        for h in handles:
            h.remove()
    P = lambda name: slots[c.m_of(name) - 1]  # noqa: E731
    flat = lambda t: t.reshape(c.N, t.shape[-1])  # noqa: E731
    close = dict(rtol=1e-5, atol=1e-6)
    for w, (l, x) in mods.items():
        r, W = seen[w], model.get_parameter(w)
        assert torch.equal(P(f"{l}.Y_{x}"), flat(r["Y"])), (l, x)
        torch.testing.assert_close(P(f"{l}.dX_{x}"), flat(r["dY"]) @ W, **close)
        torch.testing.assert_close(P(f"{l}.G_{x}"), flat(r["dY"]).T @ flat(r["X"]), **close)
    r, E = seen["lm_head"], model.get_parameter(c.w_e)
    assert torch.equal(P("Lambda"), flat(r["Y"]))
    torch.testing.assert_close(P("dF"), flat(r["dY"]) @ E, **close)
    torch.testing.assert_close(P("G_E_head"), flat(r["dY"]).T @ flat(r["X"]), **close)
    model.zero_grad(set_to_none=True)


def test_all_zero_mask_is_rejected_in_loss(setup):
    """``Σ μ = 0`` would make ℒ a 0/0 NaN; ``loss`` raises before any backward."""
    c, model, records = setup
    dead = [Record(ids=r.ids, targets=r.targets, mask=torch.zeros_like(r.mask)) for r in records]
    with pytest.raises(ValueError, match="no loss-masked target"):
        c.loss(model, dead)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match="no loss-masked target"):
        prove_step(c, model, w0, dead)
    for n, p in model.named_parameters():
        assert torch.equal(p, w0[n]) and p.grad is None, n


def test_dK_leaf_is_the_transpose_of_the_capture(setup):
    c, model, records = setup
    cap = run_capture(c, model, records)
    slots = c.label(cap, model)
    outs = {r.out.untyped_storage().data_ptr() for r in cap.records}
    for p, t in zip(c.products, slots):
        own = t.untyped_storage().data_ptr() in outs
        assert own == (c.product_class(p) != "dK"), p.name
    model.zero_grad(set_to_none=True)


def _bmm_records(cap, phase):
    return [r for r in cap.records if r.phase == phase and r.batch is not None]


@pytest.mark.parametrize("tamper", [
    "release", "duplicate_fwd", "drop_bwd_mm", "drop_bwd_bmm", "extra_glue", "swap_S_O",
    "stray_fwd_bmm", "duplicate_bwd_bmm", "swap_G_k_G_v", "O_b_not_Y_v", "S_b_split_kv",
])
def test_label_rejects_tampered_capture(setup, tamper):
    c, model, records = setup
    cap = run_capture(c, model, records)
    recs = cap.records
    if tamper == "release":
        cap.release_operands()
    elif tamper == "duplicate_fwd":
        recs.append(recs[0])
    elif tamper == "drop_bwd_mm":
        recs.remove(next(r for r in recs if r.phase == "backward" and r.batch is None))
    elif tamper == "drop_bwd_bmm":
        recs.remove(_bmm_records(cap, "backward")[0])
    elif tamper == "extra_glue":
        cap.glue_outer.append(cap.glue_outer[0])
    elif tamper == "swap_S_O":  # O called before S: data dependence forbids it
        s_rec, o_rec = _bmm_records(cap, "forward")[:2]
        i, j = recs.index(s_rec), recs.index(o_rec)
        recs[i] = dataclasses.replace(s_rec, index=o_rec.index)
        recs[j] = dataclasses.replace(o_rec, index=s_rec.index)
    elif tamper == "stray_fwd_bmm":  # a forward bmm outside every Y_v..Y_o bracket
        s_rec = _bmm_records(cap, "forward")[0]
        recs.append(dataclasses.replace(s_rec, index=10_000))
    elif tamper == "duplicate_bwd_bmm":  # the same attention gradient captured twice
        bwd = _bmm_records(cap, "backward")
        recs.append(bwd[0])
    elif tamper == "swap_G_k_G_v":  # the two outputs have the same shape and operands match
        grad = {model.get_parameter(c.w(1, x)).grad.untyped_storage().data_ptr(): x
                for x in ("k", "v")}
        g_rec = {grad[r.out.untyped_storage().data_ptr()]: r for r in recs
                 if r.phase == "backward" and r.out.untyped_storage().data_ptr() in grad}
        i, j = recs.index(g_rec["k"]), recs.index(g_rec["v"])
        recs[i] = dataclasses.replace(g_rec["k"], out=g_rec["v"].out)
        recs[j] = dataclasses.replace(g_rec["v"], out=g_rec["k"].out)
    elif tamper in ("O_b_not_Y_v", "S_b_split_kv"):
        s_rec, o_rec = _bmm_records(cap, "forward")[:2]
        rec = o_rec if tamper == "O_b_not_Y_v" else s_rec
        b = rec.b.clone()
        b[1] = b[1] + 1e-3  # member (0, 1): for S, kv head 0 shared with query head 0
        recs[recs.index(rec)] = dataclasses.replace(rec, b=b)
    match = {"swap_G_k_G_v": r"is not .*\.grad", "O_b_not_Y_v": "not Y_v's output",
             "S_b_split_kv": "share a kv head"}.get(tamper)
    with pytest.raises(LabelingError, match=match):
        c.label(cap, model)
    model.zero_grad(set_to_none=True)


# ---- milestone M2 -----------------------------------------------------------------------------


@pytest.mark.slow
def test_smollm2_real_step_m2():
    """M2: a real 4×128 step from W_0 on π(1) of D, labeled, proved and committed."""
    import os
    import resource
    import time

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records, schedule

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; run runs/materialize_data.py or set "
                    f"VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    assert (c.M, c.n_leaves) == (7113, 7661)
    t0 = time.perf_counter()
    model = c.build_model()
    D = load_dataset_records(cfg.data_dir / "D.bin", c.n)
    records = [D[i] for i in schedule(1, c.n_s, len(D))]
    w0 = {n: model.get_parameter(n).detach().clone() for n in c.weight_names}
    t1 = time.perf_counter()
    out = prove_step(c, model, w0, records)
    t2 = time.perf_counter()
    tree = commit(c, out)
    t3 = time.perf_counter()

    assert len(out.products) == c.M
    for p, t in zip(c.products, out.products):
        assert tuple(t.shape) == p.p_shape and t.dtype == torch.float32, p.name
        assert torch.isfinite(t).all(), p.name
    assert len(out.leaves()) == c.n_leaves == tree.n_leaves
    kinds = Counter(p.kind.value for p in c.products)
    assert kinds == {"forward": 2371, "input_grad": 211, "weight_grad": 211,
                     "operand_grad": 4320}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 2**30 if os.uname().sysname == "Darwin" else rss / 2**20
    print(f"\nM2: loss {out.loss:.4f}  products {len(out.products)}  kinds {dict(kinds)}  "
          f"root {tree.root.hex()[:16]}…")
    print(f"M2: load {t1 - t0:.1f}s  prove_step {t2 - t1:.1f}s  commit {t3 - t2:.1f}s  "
          f"peak RSS {rss_gb:.2f} GB")
