"""FS30 tester, O18: ``MatmulCapture.arrival_digest`` on a fabricated ``_arrived`` table, the
cases ``test_capture_arrival.py`` leaves out: a slice spanning two members, a dtype view, a
storage the table no longer owns, a transposed view, a whole output at a base offset, and the
table row the real ``_arrived_from`` fills."""

import gc
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from tests.verification.prover.test_capture_arrival import arrived, cap  # noqa: F401
from verification.prover.capture import MatmulCapture

D = b"\x01" * 32


def test_a_slice_spanning_two_members_is_not_the_output(cap):  # noqa: F811
    out = torch.randn(4, 3, 4)
    arrived(cap, out, D)
    flat = out.view(-1)
    assert cap.arrival_digest(flat[6:18].view(3, 4)) is None  # half of member 0, half of 1
    assert cap.arrival_digest(flat[12:24].view(3, 4)) is None  # member 1 alone
    assert cap.arrival_digest(flat[0:24].view(6, 4)) is None  # two members
    assert cap.arrival_digest(out[1:3]) is None
    assert cap.arrival_digest(flat.view(4, 3, 4)) == D  # the whole thing again


def test_a_dtype_view_of_the_output_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, D)
    assert cap.arrival_digest(out.view(torch.int32)) is None  # same bytes, other dtype
    assert cap.arrival_digest(out.view(torch.bfloat16)) is None  # also a new shape
    assert cap.arrival_digest(out.view(torch.float32)) == D


def test_a_view_of_the_same_storage_with_other_strides_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 4, 4)
    arrived(cap, out, D)
    assert cap.arrival_digest(out.transpose(1, 2)) is None  # the dK shape: a transposed view
    assert cap.arrival_digest(out.transpose(1, 2).contiguous()) is None
    assert cap.arrival_digest(out[:, :, ::2]) is None
    assert cap.arrival_digest(out.permute(0, 2, 1)) is None


def test_a_dead_storage_and_a_new_tensor_of_the_same_shape_do_not_match(cap):  # noqa: F811
    """The table keeps the host storage's weak reference, so its address is not handed to a
    later tensor, and a fresh tensor of the same shape and dtype (version 0 as well) finds no
    entry."""
    out = torch.zeros(3, 5)
    arrived(cap, out, D)
    assert out._version == 0
    del out
    gc.collect()
    fresh = [torch.zeros(3, 5) for _ in range(300)]
    assert all(cap.arrival_digest(t) is None for t in fresh)
    assert len(cap._arrived) == 1


def test_a_second_tensor_over_the_same_storage_after_a_write_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, D)
    view = out[0]
    assert cap.arrival_digest(out) == D
    view.zero_()  # the version counter is shared by every view of it
    assert cap.arrival_digest(out) is None


def test_an_output_at_a_base_offset_is_matched_only_there(cap):  # noqa: F811
    big = torch.randn(10, 2, 3)
    rec = big[4:7]  # a record's output at an offset in a longer storage
    arrived(cap, rec, D)
    assert cap.arrival_digest(big[4:7]) == D
    assert cap.arrival_digest(big[0:3]) is None  # same shape, another offset
    assert cap.arrival_digest(big[5:8]) is None
    assert cap.arrival_digest(big[4]) is None


def test_a_zero_element_tensor_is_not_matched(cap):  # noqa: F811
    out = torch.randn(2, 3, 4)
    arrived(cap, out, D)
    assert cap.arrival_digest(out[:0]) is None
    assert cap.arrival_digest(torch.empty(0)) is None


def test_arrived_from_records_what_arrival_digest_reads(cap):  # noqa: F811
    """The table row ``_arrived_from`` writes (not the hand-built one): the full shape, the
    base offset and the version of the host tensor after its copy."""
    host = torch.randn(3, 4, 6)
    host.add_(0)
    rec = SimpleNamespace(n_members=3, batch=(3,), out=host)
    cap._arrived_from(rec, SimpleNamespace(result=lambda: ([D], 1), n=1))
    assert cap.arrival_digest(host) == D
    assert [cap.arrival_digest(host[i]) for i in range(3)] == [None] * 3
    assert cap.summary()["arrival_hash"] == {"records": 1, "leaves": 1, "bytes": 288,
                                             "nonfinite": 0}
    assert StorageWeakRef(host.untyped_storage()) in cap._arrived
    bad = SimpleNamespace(n_members=1, batch=None, out=torch.randn(4, 6))
    cap._arrived_from(bad, SimpleNamespace(result=lambda: ([], 0), n=1))  # a NaN or Inf
    assert cap.arrival_digest(bad.out) is None
    assert cap.summary()["arrival_hash"]["nonfinite"] == 1
