"""FS30: ``MatmulCapture.arrival_digest`` matches a leaf to the digest hashed on arrival by the
host storage it sits in, the member offset, shape, dtype and contiguity. The entries are put in
by hand here; the capture that fills them on CUDA is in ``test_device_leaves_cuda.py``."""

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from verification.prover.capture import MatmulCapture


def arrived(cap, host, members, shape, digests):
    cap._arrived[StorageWeakRef(host.untyped_storage())] = (
        members, tuple(shape), host.dtype, host.storage_offset(), digests, host._version)


@pytest.fixture
def cap():
    return MatmulCapture()


def test_the_whole_output_of_an_unbatched_record(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, 1, (4, 6), [b"a" * 32])
    assert cap.arrival_digest(out) == b"a" * 32
    assert cap.arrival_digest(out.detach()) == b"a" * 32


def test_member_i_of_a_batched_record(cap):
    out = torch.randn(3, 4, 6)
    d = [bytes([i]) * 32 for i in range(3)]
    arrived(cap, out, 3, (4, 6), d)
    assert [cap.arrival_digest(out[i]) for i in range(3)] == d


def test_a_wrong_offset_shape_dtype_or_view_is_not_matched(cap):
    out = torch.randn(3, 4, 6)
    arrived(cap, out, 3, (4, 6), [bytes([i]) * 32 for i in range(3)])
    assert cap.arrival_digest(out.reshape(3, 24)[1].view(4, 6)[1:]) is None  # other shape
    assert cap.arrival_digest(out.view(-1)[5:5 + 24].view(4, 6)) is None  # off the member grid
    assert cap.arrival_digest(out[1].t()) is None  # not contiguous
    assert cap.arrival_digest(out[1].to(torch.float64)) is None  # a new tensor
    assert cap.arrival_digest(out) is None  # the whole batch is not a leaf
    assert cap.arrival_digest(out.view(-1)[24 * 3:].view(0, 4, 6)) is None


def test_a_new_tensor_and_a_transpose_copy_are_not_matched(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, 1, (4, 6), [b"a" * 32])
    assert cap.arrival_digest(out.clone()) is None
    assert cap.arrival_digest(out.t().contiguous()) is None


def test_a_non_finite_member_has_no_digest(cap):
    out = torch.randn(4, 3, 3)
    arrived(cap, out, 4, (3, 3), [b"x" * 32, b"y" * 32])  # member 2 was the first bad
    assert cap.arrival_digest(out[1]) == b"y" * 32
    assert cap.arrival_digest(out[2]) is None
    assert cap.arrival_digest(out[3]) is None


def test_a_buffer_written_since_it_was_hashed_is_not_matched(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, 1, (4, 6), [b"a" * 32])
    out.add_(1)
    assert cap.arrival_digest(out) is None


def test_the_summary_has_no_arrival_hash_when_nothing_was_hashed(cap):
    assert "arrival_hash" not in cap.summary()
