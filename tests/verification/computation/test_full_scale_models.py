"""FS4: labeling (M2) and the verifier's forward and backward replay (A8, A9) on the full-scale
models, one real 4×128 step from ``W_0`` on ``π(1)`` each (slow, skipped unless cached).

A full-scale step's operands are too large to hold twice, so the prover's side keeps a BLAKE3
digest of each operand (dtype, shape and bytes) as labeling sees it, and the replay's operands
are compared by digest. The fast test runs the same comparison on a tiny model.
"""

import gc
import time
from collections import Counter

import blake3
import pytest
import torch

from setup.data import schedule
from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import _Labeler
from verification.computation.interface import ProductKind
from verification.computation.matmul_ops import param_storage_map
from verification.prover.capture import MatmulCapture

from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
from tests.verification.full_scale_helpers import PIN_IDS, PINS, full_scale_case


class ListReader:
    def __init__(self, leaves):
        self.leaves = list(leaves)

    def leaf(self, index):
        return self.leaves[index]


def digest(t):
    """BLAKE3 over a tensor's dtype, shape and contiguous bytes."""
    t = t.detach().contiguous()
    h = blake3.blake3(f"{t.dtype}|{tuple(t.shape)}|".encode(), max_threads=blake3.blake3.AUTO)
    h.update(t.reshape(-1).view(torch.uint8).numpy())
    return h.digest()


def prover_digests(c, model, records):
    """One captured step, labeled. Returns the products, how often each slot was filled, each
    product's operand digests as labeling saw them (``m -> (A, B)``, transposed back for a
    leaf stored transposed) and each glue-gradient weight's gradient digest."""
    fills, seen = Counter(), {}
    orig = _Labeler._fill

    def fill(self, name, a, b, out, rec, leaf=None):
        fills[name] += 1
        seen[c.m_of(name)] = ((digest(a), digest(b)) if leaf is None
                              else (digest(b.mT), digest(a.mT)))
        return orig(self, name, a, b, out, rec, leaf)

    model.zero_grad(set_to_none=True)
    cap = MatmulCapture(param_storage_map(model))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_Labeler, "_fill", fill)
        # The whole-run fill never calls _fill; no runs makes every member go through it.
        mp.setitem(c.__dict__, "member_runs", {})
        with cap:
            with cap.phase("forward"):
                loss = c.loss(model, records)
            with cap.phase("backward"):
                loss.backward()
        products = [p.detach() for p in c.label(cap, model)]
    del cap, loss
    glue = {n: digest(model.get_parameter(n).grad) for n in c.glue_gradient_weights}
    model.zero_grad(set_to_none=True)
    return products, fills, seen, glue


def replay_mismatches(c, leaves, seen, glue):
    """Every product's operands in canonical order, then every glue gradient, rebuilt from the
    leaves by the verifier's replay; returns the products and glue weights whose digest (or
    shape) differs from the prover's, and the forward and backward replay times."""
    replay = c.replay(ListReader(leaves))
    bad, t0, t_bwd = [], time.perf_counter(), None
    for spec in c.products:
        if spec.kind is not ProductKind.FORWARD and t_bwd is None:
            t_bwd = time.perf_counter()
        a, b = replay.operands(spec.m)
        if ((tuple(a.shape), tuple(b.shape)) != (spec.a_shape, spec.b_shape)
                or (digest(a), digest(b)) != seen[spec.m]):
            bad.append(spec.name)
    got = replay.glue_gradients()
    bad_glue = [n for n in c.glue_gradient_weights if digest(got[n]) != glue[n]]
    t1 = time.perf_counter()
    return bad, bad_glue, (t_bwd - t0, t1 - t_bwd)


def check_step(c, model, records):
    """M2, A8 and A9 on one step: returns the M2 fill counts and products, and the mismatches.
    The step's ``W_t`` leaves are ``model``'s weights; its ``W_{t+1}`` leaves, which the replay
    never reads, are the same tensors."""
    w0 = [model.get_parameter(n).detach().clone() for n in c.weight_names]
    products, fills, seen, glue = prover_digests(c, model, records)
    leaves = [*records, *w0, *products, *w0]
    return products, fills, leaves, seen, glue


@pytest.mark.parametrize("family,tied", [("llama", True), ("qwen2", False)])
def test_digest_comparison_on_a_tiny_model(family, tied):
    """The comparison the slow tests run: an honest step matches everywhere. A ``W_t`` leaf
    altered after labeling shows in exactly the two operands that read it; a committed
    ``δX_q`` altered shows in the glue gradients it flows into (``W_E``'s, layer 1's γ)."""
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3)
    model = c.build_model()
    randomize_biases(dict(model.named_parameters()))
    records = make_records((c.n, c.n - 5), seed=2)
    products, fills, leaves, seen, glue = check_step(c, model, records)
    assert len(products) == c.M and fills == Counter(p.name for p in c.products)
    assert len(seen) == c.M
    assert replay_mismatches(c, leaves, seen, glue)[:2] == ([], [])

    def altered(i):
        out = list(leaves)
        out[i] = out[i].clone()
        out[i].view(-1)[0] += 1.0
        return out

    bad, bad_glue, _ = replay_mismatches(c, altered(c.w_t_index(c.w(2, "q"))), seen, glue)
    assert (bad, bad_glue) == (["L2.Y_q", "L2.dX_q"], [])
    bad, bad_glue, _ = replay_mismatches(c, altered(c.product_index(c.m_of("L1.dX_q"))), seen,
                                         glue)
    assert bad == [] and {c.w_e, c.gamma_attn(1)} <= set(bad_glue)
    assert c.gamma_final not in bad_glue


@pytest.mark.slow
@pytest.mark.parametrize("repo,revision", PINS, ids=PIN_IDS)
def test_real_step_labels_and_replays(repo, revision):
    """On the real model: every slot filled exactly once and every product finite (M2); every
    product's operands, forward (A8) and backward (A9), and every glue gradient rebuilt by the
    verifier bit for bit."""
    from setup.config import setup_determinism
    from verification.runs.metrics import lifetime_maxrss_bytes

    case = full_scale_case(repo, revision)
    cfg = case.cfg
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    records = [case.D[i] for i in schedule(1, c.n_s, len(case.D), cfg.seed)]
    t0 = time.perf_counter()
    model = c.build_model()
    t1 = time.perf_counter()
    products, fills, leaves, seen, glue = check_step(c, model, records)
    t2 = time.perf_counter()
    assert len(products) == c.M and fills == Counter(p.name for p in c.products)
    for spec, p in zip(c.products, products):
        assert tuple(p.shape) == spec.p_shape and p.dtype == c.product_dtype, spec.name
        assert torch.isfinite(p).all(), spec.name
    del model, products
    gc.collect()
    bad, bad_glue, (t_fwd, t_bwd) = replay_mismatches(c, leaves, seen, glue)
    assert not bad, f"{len(bad)} products' operands differ, first {bad[:5]}"
    assert not bad_glue, f"{len(bad_glue)} glue gradients differ, first {bad_glue[:3]}"
    rss = lifetime_maxrss_bytes()
    kinds = Counter(p.kind.value for p in c.products)
    print(f"\n{repo}@{revision[:8]}: M {c.M} {dict(kinds)}, {len(glue)} glue gradients, all "
          f"bit-identical; load {t1 - t0:.1f}s, capture+label+digest {t2 - t1:.1f}s, replay "
          f"forward {t_fwd:.1f}s, backward+glue {t_bwd:.1f}s; lifetime peak RSS "
          f"{rss / 2**30:.2f} GB")
