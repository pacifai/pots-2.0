"""FS30 tester: ``MatmulCapture.arrival_digest`` on a fabricated ``_arrived`` table, the cases
``test_capture_arrival.py`` leaves out: a slice spanning two members, a dtype view, a storage
the table no longer owns, a list cut at ``first_bad``, a transposed view, and the table of a
capture the real ``_arrived_from`` fills."""

import gc
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from tests.verification.prover.test_capture_arrival import arrived, cap  # noqa: F401
from verification.prover.capture import MatmulCapture


def digests(n):
    return [bytes([i + 1]) * 32 for i in range(n)]


def test_a_slice_spanning_two_members_is_not_a_leaf(cap):  # noqa: F811
    out = torch.randn(4, 3, 4)
    arrived(cap, out, 4, (3, 4), digests(4))
    flat = out.view(-1)
    assert cap.arrival_digest(flat[6:18].view(3, 4)) is None  # half of member 0, half of 1
    assert cap.arrival_digest(flat[12:24].view(3, 4)) == digests(4)[1]  # the grid again
    assert cap.arrival_digest(flat[0:24].view(6, 4)) is None  # two members: a different shape
    assert cap.arrival_digest(out[1:3].reshape(6, 4)) is None


def test_a_dtype_view_of_a_member_is_not_a_leaf(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, 2, (3, 4), digests(2))
    assert cap.arrival_digest(out[1].view(torch.int32)) is None  # same bytes, other dtype
    assert cap.arrival_digest(out[1].view(torch.bfloat16)) is None  # also a new shape
    assert cap.arrival_digest(out[1].view(torch.float32)) == digests(2)[1]


def test_a_view_of_the_same_storage_with_other_strides_is_not_a_leaf(cap):  # noqa: F811
    out = torch.randn(2, 4, 4)
    arrived(cap, out, 2, (4, 4), digests(2))
    assert cap.arrival_digest(out[1].t()) is None  # the dK shape: a transposed view
    assert cap.arrival_digest(out[1].t().contiguous()) is None
    assert cap.arrival_digest(out[:, :, ::2][1]) is None
    assert cap.arrival_digest(out.permute(0, 2, 1)[1]) is None


def test_a_dead_storage_and_a_new_tensor_of_the_same_shape_do_not_match(cap):  # noqa: F811
    """The table keeps the host storage's weak reference, so its address is not handed to a
    later tensor, and a fresh tensor of the same shape and dtype (version 0 as well) finds no
    entry."""
    out = torch.zeros(3, 5)
    arrived(cap, out, 1, (3, 5), digests(1))
    assert out._version == 0
    del out
    gc.collect()
    fresh = [torch.zeros(3, 5) for _ in range(300)]
    assert all(cap.arrival_digest(t) is None for t in fresh)
    assert len(cap._arrived) == 1


def test_a_second_tensor_over_the_same_storage_after_a_write_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, 2, (3, 4), digests(2))
    view = out[0]
    assert cap.arrival_digest(view) == digests(2)[0]
    view.zero_()  # the version counter is shared by every view of it
    assert cap.arrival_digest(out[0]) is None and cap.arrival_digest(out[1]) is None


def test_a_digest_list_cut_at_first_bad_gives_none_from_there_on(cap):  # noqa: F811
    out = torch.randn(5, 2, 3)
    d = digests(5)
    arrived(cap, out, 5, (2, 3), d[:2])  # first_bad = 2
    assert [cap.arrival_digest(out[i]) for i in range(5)] == [d[0], d[1], None, None, None]
    arrived(cap, out, 5, (2, 3), [])  # the first member was bad
    assert [cap.arrival_digest(out[i]) for i in range(5)] == [None] * 5


def test_the_member_count_bounds_the_offset(cap):  # noqa: F811
    big = torch.randn(6, 2, 3)
    host = big[:3]  # a record of 3 members at the front of a longer storage
    arrived(cap, host, 3, (2, 3), digests(3))
    assert [cap.arrival_digest(big[i]) for i in range(6)] == digests(3) + [None] * 3


def test_a_tensor_at_a_base_offset_is_found_relative_to_it(cap):  # noqa: F811
    big = torch.randn(10, 2, 3)
    rec = big[4:7]
    arrived(cap, rec, 3, (2, 3), digests(3))
    assert [cap.arrival_digest(big[i]) for i in range(10)] == \
        [None] * 4 + digests(3) + [None] * 3


def test_a_zero_element_tensor_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, 2, (3, 4), digests(2))
    assert cap.arrival_digest(out[:0]) is None
    assert cap.arrival_digest(torch.empty(0)) is None


def test_arrived_from_records_what_arrival_digest_reads(cap):  # noqa: F811
    """The table row ``_arrived_from`` writes (not the hand-built one): the member shape, the
    base offset and the version of the host tensor after its copy."""
    host = torch.randn(3, 4, 6)
    host.add_(0)
    rec = SimpleNamespace(n_members=3, batch=(3,), out=host)
    pend = SimpleNamespace(result=lambda: (digests(3)[:2], 2), n=3)
    cap._arrived_from(rec, pend)
    assert [cap.arrival_digest(host[i]) for i in range(3)] == digests(3)[:2] + [None]
    assert cap.summary()["arrival_hash"] == {"records": 1, "leaves": 2, "bytes": 2 * 96,
                                             "nonfinite": 1}
    assert StorageWeakRef(host.untyped_storage()) in cap._arrived
    unbatched = SimpleNamespace(n_members=1, batch=None, out=torch.randn(4, 6))
    cap._arrived_from(unbatched, SimpleNamespace(result=lambda: (digests(1), 1), n=1))
    assert cap.arrival_digest(unbatched.out) == digests(1)[0]
