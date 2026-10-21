"""``Replay.operand_blocks`` (FS37): the replay's operands as views, in blocks of strided
slices of the whole bmm operands. Each block's ``a[t]``, ``b[t]`` are ``operands(ms[pos[t]])``
bit for bit and stride for stride, never a copy; the blocks cover ``ms`` once; the replay's
state and first error are those of ``operands(m)`` over ``ms``."""

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.interface import OperandBlock, ReplayError
from verification.verifier.checks import _measure_groups, _member_runs

from tests.verification.computation.test_decoder import (
    _ATTN_BACKWARD, BF16, F32, ListReader, _groups, _bf16_step, honest_all, setup,  # noqa: F401
    _fresh_verifier_model,  # noqa: F401
    run_capture,
)
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config,
)


def _check_blocks(blocks, ms, want):
    """``blocks`` cover ``ms`` once and each member equals ``want[m]`` (``(a, b)`` of
    ``operands(m)``): same bits, strides and dtype."""
    seen = []
    for blk in blocks:
        assert isinstance(blk, OperandBlock) and blk.a.dim() == blk.b.dim() == 3
        assert len(blk.pos) == blk.a.shape[0] == blk.b.shape[0]
        assert not blk.a.requires_grad and not blk.b.requires_grad
        for t, pos in enumerate(blk.pos):
            a, b = want[ms[pos]]
            for got, ref in ((blk.a[t], a), (blk.b[t], b)):
                assert got.dtype == ref.dtype and got.shape == ref.shape
                assert got.stride() == ref.stride(), ms[pos]
                assert torch.equal(got, ref), ms[pos]
            seen.append(pos)
    assert sorted(seen) == list(range(len(ms)))


def _whole_run(c, leaves):
    """Per group of a step: its blocks (from one replay) and ``operands`` of each member
    (from an earlier one: a computation lends its model to one replay at a time)."""
    single, want = c.replay(ListReader(leaves)), {}
    for ms in _groups(c):
        for m in ms:
            a, b = single.operands(m)
            want[m] = (a.detach().clone(memory_format=torch.preserve_format), b.detach().clone(
                memory_format=torch.preserve_format))
    return want


def _assert_blocks_match(c, leaves):
    want = _whole_run(c, leaves)
    replay = c.replay(ListReader(leaves))
    kinds = set()
    for ms in _groups(c):
        blocks = replay.operand_blocks(ms)
        _check_blocks(blocks, ms, want)
        if len(ms) > 1:
            kinds |= {c.product_class(c.product(ms[p])) for blk in blocks for p in blk.pos}
        assert replay._served_last == (ms[-1] == c.M)
    assert kinds == {"S", "O", *_ATTN_BACKWARD}
    return replay


def test_operand_blocks_match_operands(honest_all):
    c, leaves, _, _ = honest_all
    _assert_blocks_match(c, leaves)


@pytest.mark.parametrize("n", [12, 8], ids=["n12", "n_eq_dh"])
def test_operand_blocks_match_operands_bf16(n):
    c, leaves = _bf16_step(n)
    assert c.operand_dtype == BF16
    _assert_blocks_match(c, leaves)


def test_operand_blocks_share_the_whole_operands_storage(honest_all):
    """Each block of a member group views the storage ``operand_group``'s source holds."""
    c, leaves, _, _ = honest_all
    replay = c.replay(ListReader(leaves))
    for ms in _groups(c):
        blocks = replay.operand_blocks(ms)
        if len(ms) == 1:
            continue
        spec = c.product(ms[0])
        held = replay._units[spec.layer].fwd if spec.kind.name == "FORWARD" else replay._bops
        ptrs = {t.untyped_storage().data_ptr() for pair in held.values() for t in pair[:2]}
        for blk in blocks:
            assert blk.a.untyped_storage().data_ptr() in ptrs
            assert blk.b.untyped_storage().data_ptr() in ptrs


def test_operand_blocks_one_block_per_piece(honest_all):
    c, leaves, _, _ = honest_all
    replay = c.replay(ListReader(leaves))
    for ms in _groups(c):
        blocks = replay.operand_blocks(ms)
        pieces = c.member_pieces(ms)
        if pieces is None:
            assert [len(b.pos) for b in blocks] == [1] * len(ms)
            continue
        assert [(b.pos[0], len(b.pos)) for b in blocks] == [(p[0], p[4]) for p in pieces]
        for blk, (pos, R, *_rest) in zip(blocks, pieces):
            assert blk.pos == [pos + R * t for t in range(len(blk.pos))]


def test_operand_blocks_unevenly_spaced_members(honest_all):
    """Batch indices 0, 1, 3 are not a progression: one block per member, same bits."""
    c, leaves, _, _ = honest_all
    run = next(r for r in _member_runs(c.products) if len(r) > 1)
    ms = [p.m for p in run if c.product_class(p) == "S"]
    ms = [ms[0], ms[1], ms[3]]
    single = c.replay(ListReader(leaves))
    want = {m: tuple(t.detach().clone() for t in single.operands(m)) for m in range(1, ms[-1] + 1)}
    del single
    blocks = c.replay(ListReader(leaves)).operand_blocks(ms)
    _check_blocks(blocks, ms, want)
    assert sorted(len(b.pos) for b in blocks) == [1, 2]  # {0, 1} and {3}


def test_operand_blocks_evenly_spaced_members_off_the_run_table(honest_all):
    """Members one role apart from every other batch index form one block by progression of
    batch indices on the member path, with the same bits."""
    c, leaves, _, _ = honest_all
    run = next(r for r in _member_runs(c.products) if len(r) > 1)
    ms = [p.m for p in run if c.product_class(p) == "S"][::2]  # batch indices 0, 2, 4, ...
    if len(ms) < 2:
        pytest.skip("run too short")
    single = c.replay(ListReader(leaves))
    want = {m: tuple(t.detach().clone() for t in single.operands(m)) for m in range(1, ms[-1] + 1)}
    del single
    blocks = c.replay(ListReader(leaves)).operand_blocks(ms)
    _check_blocks(blocks, ms, want)


def test_operand_blocks_default_is_one_view_block_per_member(honest_all):
    """A non-member product (a dense one) and the base class's default."""
    c, leaves, _, _ = honest_all
    dense = next(g for g in _groups(c) if len(g) == 1)
    single = c.replay(ListReader(leaves))
    for m in range(1, dense[0] + 1):
        a, b = single.operands(m)
    want = {dense[0]: (a.detach().clone(), b.detach().clone())}
    del single
    replay = c.replay(ListReader(leaves))
    for m in range(1, dense[0]):
        replay.operands(m)
    blocks = replay.operand_blocks(dense)
    _check_blocks(blocks, dense, want)
    assert [b.pos for b in blocks] == [[0]]
    assert replay.operand_device(dense[0]) == torch.device("cpu")


@pytest.mark.parametrize("unit", ["forward", "backward"])
def test_operand_blocks_fail_as_members_would(honest_all, unit):
    c, leaves, _, _ = honest_all
    l = 1 if unit == "forward" else c.L
    roles = ("S", "O") if unit == "forward" else tuple(_ATTN_BACKWARD)
    run = next(r for r in _member_runs(c.products)
               if r[0].layer == l and c.product_class(r[0]) in roles)
    ms = [p.m for p in run]
    errors = []
    for serve in ("blocks", "members"):
        replay = c.replay(ListReader(leaves))
        for m in range(1, ms[0]):
            replay.operands(m)
        replay._served_last = True
        held = replay._units[l].fwd if unit == "forward" else replay._bops
        with torch.no_grad():
            held[f"L{l}.{c.product_class(run[-1])}"][1].add_(0)  # bumps _version only
        with pytest.raises(ReplayError) as e:
            if serve == "blocks":
                replay.operand_blocks(ms)
            else:
                for m in ms:
                    replay.operands(m)
        errors.append((str(e.value), replay._served_last))
    assert errors[0] == errors[1] and "changed after its product read it" in errors[0][0]


@pytest.mark.parametrize("role", ["dV", "dQ", "dK"])
def test_operand_blocks_fail_in_a_later_role_as_members_would(honest_all, role):
    c, leaves, _, _ = honest_all
    l = c.L
    ms = next(g for g in _groups(c) if c.product(g[0]).layer == l
              and c.product_class(c.product(g[-1])) == "dK")
    errors = []
    for serve in ("blocks", "members"):
        replay = c.replay(ListReader(leaves))
        for m in range(1, ms[0]):
            replay.operands(m)
        replay._served_last = True
        with torch.no_grad():
            replay._bops[f"L{l}.{role}"][0].add_(0)
        with pytest.raises(ReplayError) as e:
            if serve == "blocks":
                replay.operand_blocks(ms)
            else:
                for m in ms:
                    replay.operands(m)
        errors.append((str(e.value), replay._served_last))
    assert errors[0] == errors[1] and "changed after its product read it" in errors[0][0]


def test_operand_blocks_out_of_order_as_operands(honest_all):
    """Asked for a group before the products ahead of it, the blocks come back as
    ``operands`` would give them (the replay rebuilds), with the same bits."""
    c, leaves, _, _ = honest_all
    groups = [g for g in _groups(c) if len(g) > 1]
    first, last = groups[0], groups[-1]
    want_replay = c.replay(ListReader(leaves))
    want = {}
    for m in last:
        want[m] = tuple(t.detach().clone() for t in want_replay.operands(m))
    del want_replay
    replay = c.replay(ListReader(leaves))
    blocks = replay.operand_blocks(last)
    _check_blocks(blocks, last, want)
    # and back to the first group: out of order again
    replay2 = c.replay(ListReader(leaves))
    replay2.operand_blocks(last)
    again = replay2.operand_blocks(first)
    ref = c.replay(ListReader(leaves))
    want2 = {m: tuple(t.detach().clone() for t in ref.operands(m)) for m in first}
    _check_blocks(again, first, want2)


@pytest.mark.parametrize("dtype", [F32, BF16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("switch", SWITCHES, ids=map(switch_id, SWITCHES))
def test_operand_blocks_match_operands_at_each_switch(switch, dtype):
    """Llama and Qwen2, tied and untied, fp32 and bf16 (biases drawn at random)."""
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=dtype)
    model = c.build_model()
    randomize_biases(dict(model.named_parameters()))
    records = make_records((12, 7))
    w0 = {k: model.get_parameter(k).detach().clone() for k in c.weight_names}
    cap = run_capture(c, model, records)
    products = [p.detach() for p in c.label(cap, model)]
    _assert_blocks_match(c, [*records, *w0.values(), *products, *w0.values()])
