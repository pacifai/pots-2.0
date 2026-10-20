"""E5's standalone challenge path against the protocol's, and its call list against check 5's."""

import numpy as np
import pytest
import torch

from benchmarks.hashing import challenges as bench
from benchmarks.hashing.sizes import Step, model_config
from tests.verification.decoder_helpers import SWITCHES, switch_id, tiny_config
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.leaf_sizes import leaf_sizes
from verification.verifier import checks
from verification.verifier.matmul_check import challenges as proto

H = bytes(range(32))


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().numpy().view(np.int32)


@pytest.mark.parametrize("m,k,width", [(1, 1, 1), (7, 21, 64), (2**32 - 1, 3, 17),
                                       (393555, 21, 2048)])
def test_challenge_matrix_matches_bit_for_bit(m, k, width):
    ours, theirs = bench.challenge_matrix(H, m, k, width), proto.challenge_matrix(H, m, k, width)
    assert ours.dtype == theirs.dtype == torch.float32
    assert ours.shape == theirs.shape == (width, k)
    assert np.array_equal(_bits(ours), _bits(theirs))


@pytest.mark.parametrize("width", [1, 12, 64])
def test_challenge_matrices_match_bit_for_bit(width):
    ms = [3, 4, 9, 100, 2**20]
    ours, theirs = bench.challenge_matrices(H, ms, 21, width), \
        proto.challenge_matrices(H, ms, 21, width)
    assert ours.shape == theirs.shape == (len(ms), width, 21)
    assert np.array_equal(_bits(ours), _bits(theirs))
    assert np.array_equal(_bits(ours[2]), _bits(proto.challenge_matrix(H, 9, 21, width)))


@pytest.mark.parametrize("call", [
    lambda f: f.challenge_matrix(H, 0, 21, 4),
    lambda f: f.challenge_matrix(H, 1, 0, 4),
    lambda f: f.challenge_matrix(H, 1, 256, 4),
    lambda f: f.challenge_matrix(H[:31], 1, 21, 4),
    lambda f: f.challenge_matrix(H, 1, 21, 0),
    lambda f: f.challenge_matrices(H, [1, 2**32], 21, 4),
    lambda f: f.challenge_vector(H, 1, 0, 4),
])
def test_validation_matches(call):
    with pytest.raises(ValueError) as theirs:
        call(proto)
    with pytest.raises(ValueError) as ours:
        call(bench)
    assert str(ours.value) == str(theirs.value)


def _step(config, n_s, n) -> Step:
    return Step("tiny", n_s, n, torch.float32, torch.bfloat16,
                tuple(leaf_sizes(config, n_s=n_s, n=n)))


def _check5_calls(c: DecoderComputation) -> list[tuple]:
    """Check 5's challenge calls, from its own run split and grouping (``_measure_run``)."""
    out = []
    for run in checks._member_runs(c.products):
        if len(run) == 1:
            out.append(((run[0].m,), run[0].b_shape[1], run[0].layer, False))
            continue
        groups = {}
        for spec in run:
            groups.setdefault((spec.a_shape, spec.b_shape), []).append(spec)
        out += [(tuple(s.m for s in g), shapes[1][1], g[0].layer, True)
                for shapes, g in groups.items()]
    return out



@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("n_s,n", [(2, 12), (3, 8)])  # (3, 8): n equals the head width
def test_plan_is_check_5s_call_list(switch, n_s, n):
    family, tied = switch
    config = tiny_config(family=family, tied=tied)
    c = DecoderComputation(config, n_s=n_s, n=n, eta=1.0)
    ps = bench.products(_step(config, n_s, n))
    assert [(p.m, p.name, p.layer, p.member) for p in ps] == \
        [(s.m, s.name, s.layer, s.member is not None) for s in c.products]
    assert [p.width for p in ps] == [s.b_shape[1] for s in c.products]
    calls = bench.plan(ps)
    assert [(c_.ms, c_.width, c_.layer, c_.batched) for c_ in calls] == _check5_calls(c)
    assert sum(c_.xof_calls for c_ in calls) == len(c.products)


def test_select_takes_the_head_and_the_first_layers():
    config = tiny_config()
    calls = bench.plan(bench.products(_step(config, 2, 12)))
    chosen, n_sel, n_total = bench.select(calls, 1)
    assert (n_sel, n_total) == (1, 2)
    assert {c.layer for c in chosen} == {None, 1}
    assert bench.select(calls, None)[0] == calls


def test_e5a_on_a_tiny_step():
    s = _step(tiny_config(), 2, 12)
    rows = bench.run_e5a(s, k=3, layers=1)
    assert [r["name"] for r in rows] == ["python_path", "python_path_per_step"]
    assert [r["status"] for r in rows] == ["measured", "estimated"]
    m = len([l for l in s.leaves if l.section == "product"])
    assert rows[1]["values"]["xof_calls_per_step"] == 3 * m
    full = bench.run_e5a(s, k=3, layers=None)
    assert full[1]["status"] == "measured"
    assert full[0]["values"]["xof_calls"] == 3 * m


def test_e5a_prime_on_a_tiny_step():
    s = _step(tiny_config(), 2, 12)
    rows = bench.run_e5a_prime(s, k=3, layers=None, threads=[2], processes=1)
    assert [(r["name"], r["status"]) for r in rows] == [("thread_pool", "measured"),
                                                        ("thread_pool_per_step", "measured")]
    assert rows[0]["params"]["threads"] == 2


@pytest.mark.slow  # reads the model's config from the Hugging Face cache
def test_llama_step_has_8_3_million_calls():
    """HASHING_PLAN_DRAFT's count: about 8.3M XOF calls per Llama-3.2-1B step at k = 21."""
    config = model_config("llama-3.2-1b")
    s = Step("llama-3.2-1b", 128, 128, torch.float32, torch.bfloat16,
             tuple(leaf_sizes(config, n_s=128, n=128)))
    calls = bench.plan(bench.products(s))
    n_products = sum(c.xof_calls for c in calls)
    L, n_h = config.num_hidden_layers, config.num_attention_heads
    assert n_products == L * (21 + 6 * 128 * n_h) + 3 == 393_555
    assert n_products * 21 == 8_264_655
