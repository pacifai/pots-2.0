"""``TranscriptView``'s product reads under O18: a member is a view of its group's leaf, a run of
members is a slice of it, and every read equals ``torch.stack`` of ``product(m)`` bit for bit."""

import pytest
import torch

from tests.verification.decoder_helpers import tiny_config
from verification.computation.instances import DecoderComputation
from verification.transcript.errors import LeafShapeError, StoreMutationError
from verification.transcript.reader import TranscriptView
from verification.verifier.context import CommittedLeaves


class CountingReader:
    def __init__(self, leaves):
        self.leaves = leaves
        self.reads = 0

    def leaf(self, index):
        self.reads += 1
        return self.leaves[index]


def _decoder(n_s=2, n=12):
    return DecoderComputation(tiny_config(), n_s=n_s, n=n, eta=1e-3)


def _leaves(c, dtype=torch.float32):
    """Random leaves of the declared shapes; the non-product ones are stand-ins."""
    torch.manual_seed(0)
    objs = [torch.zeros(1) for _ in range(c.n_leaves)]
    for pl in c.product_leaves:
        objs[c.product_leaf_index(pl.g)] = torch.randn(pl.shape).to(dtype)
    return objs


@pytest.fixture
def setup():
    c = _decoder()
    objs = _leaves(c)
    reader = CountingReader(objs)
    return c, TranscriptView(c, reader), objs, reader


def _leaf(c, objs, g):
    return objs[c.product_leaf_index(g)]


def _group(c, role, layer=1):
    return next(pl for pl in c.product_leaves
                if pl.grouped and pl.name == f"L{layer}.{role}")


def _loop(view, ms):
    return torch.stack([view.product(m) for m in ms])


def _same_as_loop(got, ms, view):
    """``got()`` equals the loop over ``ms``, or raises what the loop raises."""
    try:
        want = _loop(view, ms)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)) as raised:
            got()
        assert str(raised.value) == str(e)
        return
    assert torch.equal(got(), want)


def _shares(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_product_is_the_leaf_or_a_view_of_it(dtype):
    c = _decoder()
    objs = _leaves(c, dtype)
    view = TranscriptView(c, CountingReader(objs))
    for m in range(1, c.M + 1):
        g, i = c.leaf_of(m)
        leaf = _leaf(c, objs, g)
        got = view.product(m)
        if i is None:
            assert got is leaf
        else:
            assert torch.equal(got, leaf[i]) and _shares(got, leaf)
            assert got.shape == c.product(m).p_shape


@pytest.mark.parametrize("own", [False, True])
@pytest.mark.parametrize("role", ["S", "O", "dA", "dV", "dQ", "dK"])
def test_product_run_is_a_slice_of_the_group_leaf(setup, role, own):
    c, view, objs, _ = setup
    pl = _group(c, role)
    leaf = _leaf(c, objs, pl.g)
    for first, count, mult in ((0, pl.n, 1), (2, 3, 1), (1, 3, 2), (pl.n - 1, 1, 1)):
        m0 = pl.m0 + pl.stride * first
        if first + mult * (count - 1) >= pl.n:
            continue
        got = view.product_run(m0, pl.stride * mult, count, own=own)
        want = _loop(view, [m0 + pl.stride * mult * t for t in range(count)])
        assert got.shape == (count, *pl.shape[1:]) and torch.equal(got, want)
        assert _shares(got, leaf) is (not own)
        if not own:
            assert got.untyped_storage().data_ptr() == leaf.untyped_storage().data_ptr()


def test_product_run_across_groups_or_lone_products_equals_the_loop(setup):
    c, view, objs, _ = setup
    s = _group(c, "S")
    for m0, stride, n in ((s.m0 + s.stride * (s.n - 2), 1, 5),  # runs out of the group
                          (s.m0, s.stride * 2 + 1, 3),  # a stride the group doesn't divide
                          (c.m_of("L1.Y_gate"), 1, 2),  # lone products
                          (c.m_of("L1.Y_gate"), 1, 1)):
        _same_as_loop(lambda: view.product_run(m0, stride, n),
                      [m0 + stride * t for t in range(n)], view)
    with pytest.raises(RuntimeError):  # n = 0: stack of nothing, as the loop
        view.product_run(s.m0, 1, 0)


def test_product_run_bad_index_raises_what_the_loop_raises(setup):
    c, view, _, reader = setup
    for m0, stride, n in ((0, 1, 3), (c.M - 1, 1, 4), (c.M + 1, 1, 1), (1, 7, c.M)):
        with pytest.raises(IndexError) as want:
            _loop(view, [m0 + stride * t for t in range(n)])
        reader.reads = 0
        with pytest.raises(IndexError) as got:
            view.product_run(m0, stride, n)
        assert str(got.value) == str(want.value), (m0, stride, n)
        assert reader.reads == 0  # raised before any read


@pytest.mark.parametrize("own", [False, True])
def test_product_block_progression_is_a_slice_other_orders_stack(setup, own):
    c, view, objs, _ = setup
    pl = _group(c, "dV")
    leaf = _leaf(c, objs, pl.g)
    at = lambda i: pl.m0 + pl.stride * i  # noqa: E731
    for idx in ([0, 1, 2], [1, 3, 5], [4], list(range(pl.n))):
        got = view.product_block([at(i) for i in idx], own=own)
        assert torch.equal(got, _loop(view, [at(i) for i in idx]))
        assert _shares(got, leaf) is (not own)
    for idx in ([2, 1, 0], [0, 2, 3], [1, 1], [0, 0, 0]):  # not an increasing progression
        got = view.product_block([at(i) for i in idx], own=own)
        assert torch.equal(got, _loop(view, [at(i) for i in idx]))
        assert not _shares(got, leaf)
    mixed = [at(0), c.m_of("L1.Y_gate"), at(1)]
    _same_as_loop(lambda: view.product_block(mixed), mixed, view)
    two = [at(0), _group(c, "dA").m0]  # two groups
    _same_as_loop(lambda: view.product_block(two), two, view)


@pytest.mark.parametrize("own", [False, True])
def test_a_lone_single_product_block_is_a_view(setup, own):
    c, view, objs, _ = setup
    m = c.m_of("L1.Y_o")
    got = view.product_block([m], own=own)
    leaf = _leaf(c, objs, c.leaf_of(m)[0])
    assert got.shape == (1, *leaf.shape) and torch.equal(got[0], leaf)
    assert _shares(got, leaf) is (not own)
    pair = [c.m_of("L1.Y_gate"), c.m_of("L1.Y_up")]
    assert torch.equal(view.product_block(pair), _loop(view, pair))


def test_product_block_bad_m_raises_before_any_read(setup):
    c, view, _, reader = setup
    for ms in ([1, 0], [c.M + 1], [1, 2, c.M + 1]):
        reader.reads = 0
        with pytest.raises(IndexError):
            view.product_block(ms)
        assert reader.reads == 0
    with pytest.raises(RuntimeError):
        view.product_block([])


def test_a_group_leaf_of_the_wrong_shape_or_type_is_rejected(setup):
    c, view, objs, reader = setup
    pl = _group(c, "S")
    i = c.product_leaf_index(pl.g)
    good = objs[i]
    objs[i] = good[0]  # one member, not the group
    for read in (lambda: view.product_leaf(pl.g), lambda: view.product(pl.m0),
                 lambda: view.product_run(pl.m0, 1, 2), lambda: view.product_block([pl.m0])):
        with pytest.raises(LeafShapeError, match="declared"):
            read()
    objs[i] = "not a tensor"
    with pytest.raises(LeafShapeError, match="expected a tensor"):
        view.product(pl.m0)
    objs[i] = good
    assert view.product_leaf(pl.g) is good


def test_a_write_to_a_committed_group_leaf_is_caught_through_committed_leaves(setup):
    c, _, objs, _ = setup
    committed = CommittedLeaves()
    for o in objs:
        committed.add(o)
    view = TranscriptView(c, committed)
    pl = _group(c, "dA")
    run = view.product_run(pl.m0, pl.stride, pl.n)
    block = view.product_block([pl.m0, pl.m0 + pl.stride])
    member = view.product(pl.m0 + pl.stride)
    assert torch.equal(run[1], member) and torch.equal(block[1], member)
    member.add_(1.0)  # an in-place write through a member view moves the leaf's version
    i = c.product_leaf_index(pl.g)
    for read in (lambda: view.product(pl.m0), lambda: view.product_run(pl.m0, pl.stride, 2),
                 lambda: view.product_block([pl.m0]), lambda: view.product_leaf(pl.g)):
        with pytest.raises(StoreMutationError, match=f"leaf {i} changed"):
            read()


def test_a_write_to_a_lone_committed_leaf_is_caught(setup):
    c, _, objs, _ = setup
    committed = CommittedLeaves()
    for o in objs:
        committed.add(o)
    view = TranscriptView(c, committed)
    m = c.m_of("L1.Y_o")
    view.product(m).mul_(2.0)
    with pytest.raises(StoreMutationError):
        view.product(m)
    with pytest.raises(StoreMutationError):
        view.product_block([m])
