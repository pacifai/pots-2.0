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
