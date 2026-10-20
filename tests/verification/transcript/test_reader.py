"""``stack_block``: ``torch.stack`` bit for bit, zero-copy when the leaves are one run."""

import pytest
import torch

from verification.transcript.errors import StoreMutationError
from verification.transcript.reader import stack_block
from verification.verifier.context import CommittedLeaves


def _run(n=6, shape=(3, 5), dtype=torch.float32):
    """n member leaves as views of one 64-byte-aligned bmm-like output."""
    torch.manual_seed(0)
    big = torch.randn(n, *shape).to(dtype)
    assert big.data_ptr() % 64 == 0
    return big, [big[i] for i in range(n)]


def test_consecutive_views_come_back_as_one_view_of_the_storage():
    big, ts = _run()
    out = stack_block(ts)
    assert torch.equal(out, torch.stack(ts))
    assert out._base is not None and out.data_ptr() == big.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sub_run_and_size_one_dims_equal_stack(dtype):
    big, ts = _run(8, (1, 4), dtype)
    out = stack_block(ts[2:5])
    assert torch.equal(out, torch.stack(ts[2:5])) and out.dtype == dtype


def test_fallbacks_equal_stack_and_own_their_memory():
    big, ts = _run()
    cases = {
        "gap": [ts[0], ts[2], ts[3]],
        "reversed": ts[::-1],
        "mixed storage": [ts[0], ts[1].clone()],
        "non-contiguous": [big.transpose(1, 2)[i] for i in range(3)],
        "empty tensors": [torch.zeros(0, 3)] * 3,
    }
    for name, c in cases.items():
        out = stack_block(c)
        assert torch.equal(out, torch.stack(c)), name
        assert out._base is None, name


def test_misaligned_base_takes_the_stack_path():
    buf = torch.randn(2 * 15 + 16)
    off = next(o for o in range(16) if buf[o:].data_ptr() % 64)
    ts = [buf[off + 15 * i:off + 15 * (i + 1)].view(3, 5) for i in range(2)]
    out = stack_block(ts)
    assert torch.equal(out, torch.stack(ts)) and out._base is None


def test_empty_sequence_raises_as_stack_does():
    with pytest.raises(RuntimeError):
        stack_block([])


def test_committed_leaves_block_checks_every_version():
    big, ts = _run()
    leaves = CommittedLeaves()
    for t in ts:
        leaves.add(t)
    assert torch.equal(leaves.tensor_block(range(6)), torch.stack(ts))
    for bad in (-1, 6):
        with pytest.raises(RuntimeError, match="outside"):
            leaves.tensor_block([0, bad])
    ts[3].add_(1)  # a write through any alias moves the shared counter
    with pytest.raises(StoreMutationError, match="changed in place"):
        leaves.tensor_block(range(6))


def test_own_copies_the_run_even_for_detached_leaves():
    """``detach()`` leaves have no ``_base``, so ownership is asked for, not inferred."""
    big, ts = _run()
    ts = [t.detach() for t in ts]
    assert all(t._base is None for t in ts)
    out = stack_block(ts, own=True)
    assert torch.equal(out, torch.stack(ts)) and out.data_ptr() != big.data_ptr()
    assert stack_block(ts).data_ptr() == big.data_ptr()
    leaves = CommittedLeaves()
    for t in ts:
        leaves.add(t)
    assert leaves.tensor_block(range(6), own=True).data_ptr() != big.data_ptr()


def test_product_block_through_a_committed_view_checks_a_middle_leaf():
    from verification.computation.instances.decoder import DecoderComputation
    from verification.transcript.reader import TranscriptView
    from tests.verification.computation.test_decoder import tiny_config

    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    ms = [c.m_of(f"L1.S[{s},{h}]") for s, h in c._members()]
    leaves = CommittedLeaves()
    idx = {c.product_index(m) for m in ms}
    big = torch.randn(len(ms), 12, 12)
    for i in range(c.product_index(max(ms)) + 1):  # leaves before the run: stand-ins
        leaves.add(big[ms.index(next(m for m in ms if c.product_index(m) == i))]
                   if i in idx else torch.zeros(1))
    view = TranscriptView(c, leaves)
    want = torch.stack([view.product(m) for m in ms])
    assert torch.equal(view.product_block(ms), want)
    assert torch.equal(view.product_block(ms, own=True), want)
    mid = view.product(ms[len(ms) // 2])
    mid.add_(1)
    with pytest.raises(StoreMutationError, match="changed in place"):
        view.product_block(ms)


# ---- product_run: the range read -------------------------------------------------------------


def _decoder_view(roles, n_s=2, n=12):
    """A committed view over a tiny decoder whose products ``roles`` (a list of
    ``(first_m, stride)``, one tensor storage each) hold member leaves ``m0 + stride·i``;
    every other leaf is a one-element stand-in. Returns ``(c, view, leaves, storages)``."""
    from verification.computation.instances.decoder import DecoderComputation
    from verification.transcript.reader import TranscriptView
    from tests.verification.computation.test_decoder import tiny_config

    c = DecoderComputation(tiny_config(), n_s=n_s, n=n, eta=1e-3)
    n_mem = c.n_s * c.n_h
    total = c.n_s + len(c.weight_names) + c.M + len(c.weight_names)
    objs, storages = [torch.zeros(1) for _ in range(total)], []
    for m0, stride in roles:
        big = torch.randn(n_mem, *c.product(m0).p_shape)
        assert big.data_ptr() % 64 == 0
        storages.append(big)
        for i in range(n_mem):
            objs[c.product_index(m0 + stride * i)] = big[i]
    leaves = CommittedLeaves()
    for o in objs:
        leaves.add(o)
    return c, TranscriptView(c, leaves), leaves, storages


def _member_run(role, layer=1):
    """The decoder's own ``(m₀, stride)`` of one attention run."""
    from verification.computation.instances.decoder import DecoderComputation
    from tests.verification.computation.test_decoder import tiny_config

    return DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3).member_runs[layer, role]


def _loop(view, m0, stride, n):
    return torch.stack([view.product(m0 + stride * i) for i in range(n)])


@pytest.mark.parametrize("own", [False, True])
def test_product_run_equals_the_loop_for_a_contiguous_run(own):
    m0, st = _member_run("S")
    c, view, _, (big,) = _decoder_view([(m0, st)])
    n = c.n_s * c.n_h
    out = view.product_run(m0, st, n, own=own)
    assert out.dtype == big.dtype and torch.equal(out, _loop(view, m0, st, n))
    assert (out.data_ptr() != big.data_ptr()) == own  # zero-copy unless asked for its own
    sub = view.product_run(m0 + 2, st, n - 4, own=own)  # a window, read off the first proof
    assert torch.equal(sub, _loop(view, m0 + 2, st, n - 4)) and sub.is_contiguous()


def test_product_run_stride_two_interleaved_roles():
    """Two roles alternate per member (δQ̃, δK̃): each is a stride-2 run of its own storage."""
    (q0, st), (k0, _) = _member_run("dQ"), _member_run("dK")
    assert st == 2 and k0 == q0 + 1
    c, view, _, (bq, bk) = _decoder_view([(q0, st), (k0, st)])
    n = c.n_s * c.n_h
    for m0, big in ((q0, bq), (k0, bk)):
        out = view.product_run(m0, 2, n)
        assert torch.equal(out, _loop(view, m0, 2, n)) and out.data_ptr() == big.data_ptr()
        assert torch.equal(view.product_run(m0 + 2, 2, 3, own=True), _loop(view, m0 + 2, 2, 3))
    # Across the two roles the leaves are not one run: the loop's bits through the fallback.
    assert torch.equal(view.product_run(q0, 1, 2 * n), _loop(view, q0, 1, 2 * n))


def test_product_run_fallbacks_match_the_loop():
    m0, st = _member_run("S")
    c, view, _, _ = _decoder_view([(m0, st)])
    assert torch.equal(view.product_run(m0 + 1, st, 1), _loop(view, m0 + 1, st, 1))
    # Leaves of another shape (a member next to a stand-in) take torch.stack, and its error.
    with pytest.raises(RuntimeError) as want:
        _loop(view, m0, c.M - m0, 2)
    with pytest.raises(RuntimeError) as got:
        view.product_run(m0, c.M - m0, 2)
    assert str(got.value) == str(want.value)
    with pytest.raises(RuntimeError):  # n = 0: stack of nothing
        view.product_run(m0, 1, 0)


def test_product_run_bad_index_raises_what_the_loop_raises():
    c, view, _, _ = _decoder_view([_member_run("S")])
    for m0, stride, n in ((0, 1, 3), (c.M - 1, 1, 4), (c.M + 1, 1, 1), (1, 7, c.M)):
        with pytest.raises(IndexError) as want:
            _loop(view, m0, stride, n)
        with pytest.raises(IndexError) as got:
            view.product_run(m0, stride, n)
        assert str(got.value) == str(want.value), (m0, stride, n)


def test_product_run_tampered_middle_leaf_raises_the_per_leaf_message():
    m0, st = _member_run("dQ")
    c, view, leaves, (big,) = _decoder_view([(m0, st)])
    n = c.n_s * c.n_h
    assert torch.equal(view.product_run(m0, st, n), _loop(view, m0, st, n))  # proof made
    big[n // 2].add_(1)
    with pytest.raises(StoreMutationError) as want:
        _loop(view, m0, st, n)
    with pytest.raises(StoreMutationError) as got:
        view.product_run(m0, st, n)
    assert str(got.value) == str(want.value)
    assert f"leaf {c.product_index(m0)} changed" in str(got.value)  # first changed in order


def test_product_run_proof_is_not_stale_after_a_write():
    """A write after a successful read must raise on the next one, for the whole range and
    for a window of it, and the cache of a fresh ``CommittedLeaves`` starts empty."""
    m0, st = _member_run("O")
    c, view, leaves, (big,) = _decoder_view([(m0, st)])
    n = c.n_s * c.n_h
    view.product_run(m0, st, n)
    view.product_run(m0 + 3, st, 3)
    assert leaves._runs
    big[0].mul_(2)
    for args in ((m0, st, n), (m0 + 3, st, 3), (m0 + 4, st, 2)):
        with pytest.raises(StoreMutationError, match="changed in place"):
            view.product_run(*args)
    assert not CommittedLeaves()._runs


def test_product_run_through_a_reader_without_tensor_run():
    m0, st = _member_run("dV")
    c, view, leaves, _ = _decoder_view([(m0, st)])
    n = c.n_s * c.n_h

    class Plain:
        def leaf(self, i):
            return leaves.leaf(i)

    from verification.transcript.reader import TranscriptView
    plain = TranscriptView(c, Plain())
    assert torch.equal(plain.product_run(m0, st, n), _loop(view, m0, st, n))
    assert torch.equal(plain.product_run(m0, st, n, own=True), _loop(view, m0, st, n))
