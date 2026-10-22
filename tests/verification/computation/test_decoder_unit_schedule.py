"""``Replay.unit_schedule`` (FS38a): the decoder serves its products in the head blocks, then
layers ``L … 1`` (each layer's forward products before its backward ones) with the same bits as
canonical order, one forward pass, each backward unit once, and the same glue gradients."""

import gc

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.interface import Replay
from verification.prover.step import prove_step

from tests.verification.computation.test_decoder import BF16, F32, ListReader, _groups
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config,
)


def _step(switch, blocks, dtype=F32):
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=4, n=12, eta=1e-3,
                           head_blocks=blocks, compute_dtype=dtype)
    model = c.build_model()
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    randomize_biases(w0)
    for n, p in model.named_parameters():
        p.data.copy_(w0[n])
    records = make_records((12, 7, 10, 5), seed=1)
    return c, ListReader(prove_step(c, model, w0, records).leaves())


def _clone(t):
    return t.detach().clone(memory_format=torch.preserve_format)


def _canonical(c, leaves):
    """``m -> (A, B)`` clones and the glue gradients, from canonical order on its own replay."""
    replay = c.replay(leaves)
    want = {m: tuple(map(_clone, replay.operands(m))) for m in range(1, c.M + 1)}
    glue = {n: _clone(g) for n, g in replay.glue_gradients().items()}
    return want, glue


def _spied(replay):
    """The replay with its builds and backward units counted."""
    log = {"build": 0, "runs": []}
    build, run_unit, run_block = replay._build, replay._run_unit, replay._run_head_block
    replay._build = lambda: (log.__setitem__("build", log["build"] + 1), build())[1]
    replay._run_unit = lambda j: (log["runs"].append(j), run_unit(j))[1]
    replay._run_head_block = lambda b: (log["runs"].append(("head", b)), run_block(b))[1]
    return log


def _expected_runs(c):
    return [("head", b) for b in range(c.head_blocks)] + list(range(c.L, 0, -1))


CASES = [(s, b, F32) for s in SWITCHES for b in (1, 2)] + [(SWITCHES[1], 2, BF16),
                                                           (SWITCHES[2], 1, BF16)]
IDS = [f"{switch_id(s)}-B{b}-{'bf16' if d is BF16 else 'fp32'}" for s, b, d in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_schedule_is_a_partition_ending_at_M(case):
    switch, blocks, dtype = case
    c, _ = _step(switch, blocks, dtype)
    steps = c.unit_schedule
    flat = [m for s in steps for m in s]
    assert sorted(flat) == list(range(1, c.M + 1)) and flat[-1] == c.M
    assert steps[:blocks] == tuple(
        tuple(c.m_of(c.head_name(k, b)) for k in ("Lambda", "dF", "G")) for b in range(blocks))
    assert len(steps) == blocks + c.L  # one step per layer
    for j, step in zip(range(c.L, 0, -1), steps[blocks:]):
        assert {c.product(m).layer for m in step} == {j}
        is_fwd = [c.product(m).kind.name == "FORWARD" for m in step]
        assert is_fwd == sorted(is_fwd, reverse=True)  # forward products first
        fwd = [m for m, f in zip(step, is_fwd) if f]
        rest = [m for m, f in zip(step, is_fwd) if not f]
        assert fwd and fwd == sorted(fwd) and rest == sorted(rest)


def test_the_base_replay_keeps_canonical_order_only():
    assert Replay.unit_schedule(object()) is None  # the default; the MLP keeps it


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_operands_in_schedule_order_equal_canonical(case):
    switch, blocks, dtype = case
    c, leaves = _step(switch, blocks, dtype)
    want, glue = _canonical(c, leaves)
    gc.collect()
    replay = c.replay(leaves)
    assert replay.unit_schedule() is c.unit_schedule
    log = _spied(replay)
    for step in replay.unit_schedule():
        for m in step:
            a, b = replay.operands(m)
            assert a.dtype == want[m][0].dtype and b.dtype == want[m][1].dtype, m
            assert torch.equal(a, want[m][0]) and torch.equal(b, want[m][1]), c.product(m).name
    assert log["build"] == 1  # the first Lambda builds, nothing after rebuilds
    assert log["runs"] == _expected_runs(c)
    assert not replay._units
    got = replay.glue_gradients()
    assert list(got) == list(glue)
    assert all(torch.equal(got[n], glue[n]) for n in glue)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_operand_blocks_in_schedule_order_equal_canonical(case):
    switch, blocks, dtype = case
    c, leaves = _step(switch, blocks, dtype)
    want, glue = _canonical(c, leaves)
    gc.collect()
    replay = c.replay(leaves)
    log = _spied(replay)
    groups = list(_groups(c))
    served = 0
    for step in replay.unit_schedule():
        for ms in (g for g in groups if g[0] in step):
            assert set(ms) <= set(step)
            seen = []
            for blk in replay.operand_blocks(ms):
                for t, pos in enumerate(blk.pos):
                    m = ms[pos]
                    assert torch.equal(blk.a[t], want[m][0]) and torch.equal(blk.b[t], want[m][1])
                    seen.append(pos)
            assert sorted(seen) == list(range(len(ms)))
            served += len(ms)
    assert served == c.M
    assert log["build"] == 1 and log["runs"] == _expected_runs(c)
    got = replay.glue_gradients()
    assert all(torch.equal(got[n], glue[n]) for n in glue)
