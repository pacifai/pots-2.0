"""FS30, O18: ``MatmulCapture.arrival_digest`` matches the WHOLE output of a record, batched or
not, to the one digest hashed on arrival, by host storage, offset, shape, dtype and contiguity.
The entries are put in by hand here; the capture that fills them on CUDA is in
``test_device_leaves_cuda.py``."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from verification.commitment.encoding import TAG_PRODUCT, tensor_leaf_header
from verification.prover.capture import MatmulCapture


def arrived(cap, host, digest):
    cap._arrived[StorageWeakRef(host.untyped_storage())] = (
        tuple(host.shape), host.dtype, host.storage_offset(), digest, host._version)


@pytest.fixture
def cap():
    return MatmulCapture()


def test_the_whole_output_of_an_unbatched_record(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, b"a" * 32)
    assert cap.arrival_digest(out) == b"a" * 32
    assert cap.arrival_digest(out.detach()) == b"a" * 32


def test_the_whole_output_of_a_batched_record_has_one_digest(cap):
    out = torch.randn(3, 4, 6)
    arrived(cap, out, b"w" * 32)
    assert cap.arrival_digest(out) == b"w" * 32


def test_anything_but_the_whole_output_is_not_matched(cap):
    out = torch.randn(3, 4, 6)
    arrived(cap, out, b"w" * 32)
    assert cap.arrival_digest(out[1]) is None  # a member view
    assert cap.arrival_digest(out[0]) is None  # member 0 shares the storage offset
    assert cap.arrival_digest(out[:2]) is None  # a slice
    assert cap.arrival_digest(out.view(3, 24)) is None  # a reshaped view
    assert cap.arrival_digest(out.view(-1)) is None
    assert cap.arrival_digest(out.to(torch.float64)) is None  # another dtype, a new tensor
    assert cap.arrival_digest(out.view(torch.int32)) is None  # same bytes, other dtype
    assert cap.arrival_digest(out.transpose(1, 2)) is None  # not contiguous
    assert cap.arrival_digest(out.transpose(1, 2).contiguous()) is None  # a transpose copy
    assert cap.arrival_digest(out.clone()) is None  # a new tensor


def test_a_new_tensor_and_a_transpose_copy_are_not_matched(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, b"a" * 32)
    assert cap.arrival_digest(out.clone()) is None
    assert cap.arrival_digest(out.t().contiguous()) is None


def test_a_non_finite_output_has_no_digest(cap):
    out = torch.randn(4, 3, 3)
    arrived(cap, out, None)  # the kernel found a NaN or Inf: no digest
    assert cap.arrival_digest(out) is None


def test_a_buffer_written_since_it_was_hashed_is_not_matched(cap):
    out = torch.randn(4, 6)
    arrived(cap, out, b"a" * 32)
    out.add_(1)
    assert cap.arrival_digest(out) is None


def test_the_summary_counts_one_leaf_per_output(cap):
    batched = SimpleNamespace(n_members=3, batch=(3,), out=torch.randn(3, 4, 6))
    plain = SimpleNamespace(n_members=1, batch=None, out=torch.randn(4, 6))
    cap._arrived_from(batched, SimpleNamespace(result=lambda: ([b"b" * 32], 1), n=1))
    cap._arrived_from(plain, SimpleNamespace(result=lambda: ([b"p" * 32], 1), n=1))
    assert cap.summary()["arrival_hash"] == {
        "records": 2, "leaves": 2, "bytes": (72 + 24) * 4, "nonfinite": 0}
    assert cap.arrival_digest(batched.out) == b"b" * 32
    assert cap.arrival_digest(batched.out[0]) is None
    assert cap.arrival_digest(plain.out) == b"p" * 32


class FakeHasher:
    """Records ``enqueue_table``'s arguments and hashes (sha256) what the table names, so a
    test sees which header, address and size a record's output is hashed under."""

    def __init__(self):
        self.calls = []

    def enqueue_table(self, hdr, hdr_off, hdr_len, hdr_idx, ptr, size, kind, *, device, stream,
                      keep):
        self.calls.append((hdr, hdr_off, hdr_len, hdr_idx, ptr, size, kind))
        n = len(ptr)
        out = keep[0]
        raw = out.numpy().tobytes()
        digests = [hashlib.sha256(hdr + raw).digest() for _ in range(n)]
        return SimpleNamespace(result=lambda: (digests, n), n=n, done=lambda: True)


@pytest.mark.parametrize("shape", [(3, 4, 6), (4, 6)])
def test_the_table_is_one_entry_under_the_whole_outputs_header(shape):
    hasher = FakeHasher()
    cap = MatmulCapture(hasher=hasher)
    out = torch.randn(*shape)
    pend = cap._hash_arrival(SimpleNamespace(n_members=shape[0] if len(shape) == 3 else 1,
                                             batch=(shape[0],) if len(shape) == 3 else None),
                             out, None)
    (hdr, hdr_off, hdr_len, hdr_idx, ptr, size, kind), = hasher.calls
    assert hdr == tensor_leaf_header(TAG_PRODUCT, out)
    assert list(hdr_len) == [len(hdr)] and list(hdr_off) == [0] and list(hdr_idx) == [0]
    assert list(ptr) == [out.data_ptr()] and list(size) == [out.nbytes]
    assert len(kind) == 1
    # The digest is that of the whole leaf (header + all bytes), kept for the whole output only.
    rec = SimpleNamespace(n_members=1, batch=None, out=out)
    cap._arrived_from(rec, pend)
    want = hashlib.sha256(tensor_leaf_header(TAG_PRODUCT, out)
                          + out.numpy().tobytes()).digest()
    assert cap.arrival_digest(out) == want
    if len(shape) == 3:
        assert cap.arrival_digest(out[0]) is None


def test_the_summary_has_no_arrival_hash_when_nothing_was_hashed(cap):
    assert "arrival_hash" not in cap.summary()


def test_a_failed_arrival_enqueue_waits_for_its_stream_before_it_raises(monkeypatch):
    """``_offload`` queues the host copy and its event before the hash. If the hash's enqueue
    raises, the side stream itself is synchronized first (the copy and any kernel the enqueue
    already launched), so no GPU read outlives ``out``. Streams and events are faked, so this
    runs on a CPU."""
    from types import SimpleNamespace

    log = []

    class Event:
        def record(self, stream):
            log.append("record")

        def synchronize(self):
            log.append("event synchronize")

    class Stream:
        def wait_stream(self, other):
            pass

        def synchronize(self):
            log.append("stream synchronize")

    side = Stream()
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: side)

    def hash_fails(rec, out, stream):
        log.append("hash")
        raise RuntimeError("enqueue failed")

    out = torch.randn(4, 6)
    cap = SimpleNamespace(
        buffers=SimpleNamespace(take=lambda shape, dtype: torch.empty(shape, dtype=dtype),
                                stream=lambda device: side),
        _hashable=lambda rec, o: True, _hash_arrival=hash_fails, _pending=[],
        _drain=lambda block: None)
    rec = SimpleNamespace(out=out, versions={})
    with pytest.raises(RuntimeError, match="enqueue failed"):
        MatmulCapture._offload(cap, rec)
    assert log == ["record", "hash", "stream synchronize"]
    assert rec.out is out and cap._pending == []
