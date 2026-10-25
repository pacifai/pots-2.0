"""The host challenge source's prefetch (FS39, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.5) on the
Mac: the worker thread's pool of two pinned-or-plain buffers, ``draw(prefetched=)``'s offsets
and bytes against ``challenge_matrix``, the fall-back to the inline draw, the worker's
exception, and the slot's wait on the events of earlier copies. A CPU device copies nothing."""

import numpy as np
import pytest
import torch

from verification.verifier.matmul_check import challenge_source as cs
from verification.verifier.matmul_check.challenges import challenge_matrix

H = bytes(range(32))
K = 3


def _equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _check(buf, offs, ms, widths, h=H, k=K):
    for m, w, o in zip(ms, widths, offs):
        got = buf[o // 4:o // 4 + w * k].view(w, k)
        assert _equal(got, challenge_matrix(h, m, k, w)), (m, w)


def test_prefetched_draw_gives_each_products_bytes_at_the_returned_offset():
    src = cs.HostChallengeSource()
    ms = [9, 3, 7, 3, 4, 9, 2]  # unsorted, repeated labels
    widths = [40, 6, 65, 6, 6, 6, 40]  # mixed and repeated widths; (9, 40) and (9, 6) differ
    pf = src.prefetch(H, ms, K, widths)
    buf, offs, ev = src.draw(H, ms, K, widths, stream=None, prefetched=pf)
    assert ev is None and buf.dtype == torch.float32 and buf.device.type == "cpu"
    assert offs.dtype == np.int64 and len(offs) == len(ms)
    _check(buf, offs, ms, widths)
    assert offs[1] == offs[3]  # a repeated (m, width) is drawn once
    assert buf.numel() == pf.total
    # a subset in another order, as one device's call of a shared prefetch draws
    sub_ms, sub_w = [4, 9, 3], [6, 40, 6]
    buf2, offs2, _ = src.draw(H, sub_ms, K, sub_w, stream=None, prefetched=pf)
    _check(buf2, offs2, sub_ms, sub_w)


def test_a_prefetch_that_does_not_cover_the_request_falls_back_to_the_inline_draw():
    src = cs.HostChallengeSource()
    pf = src.prefetch(H, [1, 2], K, [6, 6])
    for h, ms, k, widths in [(H, [1, 3], K, [6, 6]), (H, [1, 2], K, [6, 7]),
                             (bytes(32), [1, 2], K, [6, 6]), (H, [1, 2], K + 1, [6, 6])]:
        buf, offs, ev = src.draw(h, ms, k, widths, stream=None, prefetched=pf)
        inline, inline_offs, _ = src.draw(h, ms, k, widths, stream=None)
        assert ev is None and np.array_equal(offs, inline_offs)
        assert _equal(buf, inline)
        _check(buf, offs, ms, widths, h=h, k=k)


def test_the_workers_exception_is_raised_by_draw(monkeypatch):
    src = cs.HostChallengeSource()

    def boom(*a, **kw):
        raise RuntimeError("worker bug")

    monkeypatch.setattr(cs, "challenge_matrices", boom)
    pf = src.prefetch(H, [1], K, [6])
    with pytest.raises(RuntimeError, match="worker bug"):
        src.draw(H, [1], K, [6], stream=None, prefetched=pf)


def test_the_pool_reuses_two_buffers_and_grows_one_never_shrinking():
    src = cs.HostChallengeSource()

    def take(ms, widths):
        pf = src.prefetch(H, ms, K, widths)
        pf.future.result()
        return pf

    p0, p1, p2 = take([1], [6]), take([2], [6]), take([3], [6])
    assert (p0.slot, p1.slot, p2.slot) == (0, 1, 0)
    b0, b1 = src._pool[0].buf, src._pool[1].buf
    assert b0 is not b1
    take([4], [6])  # slot 1 again, same size
    take([5], [6])
    assert src._pool[0].buf is b0 and src._pool[1].buf is b1
    big = take([6, 7, 8], [60, 60, 60])  # the fifth call's successor: slot 1 grows
    assert big.slot == 1 and src._pool[1].buf is not b1
    assert src._pool[1].buf.numel() >= big.total and src._pool[0].buf is b0
    grown = src._pool[1].buf
    take([9], [6])  # slot 0
    take([10], [6])  # slot 1: smaller, no shrink
    assert src._pool[1].buf is grown
    # the bytes a reused slot holds are the new draw's
    pf = take([11, 12], [5, 6])
    buf, offs, _ = src.draw(H, [11, 12], K, [5, 6], stream=None, prefetched=pf)
    _check(buf, offs, [11, 12], [5, 6])


def test_a_slot_is_refilled_only_after_the_events_of_its_copies():
    class Event:
        def __init__(self):
            self.waited = False

        def synchronize(self):
            self.waited = True

    src = cs.HostChallengeSource()
    ev = Event()
    src._pool[0].events = [ev]
    pf = src.prefetch(H, [1], K, [6])  # takes slot 0 and its events at submission
    assert src._pool[0].events == []
    pf.future.result()
    assert ev.waited
    other = Event()
    src._pool[1].events = [other]
    assert not other.waited  # slot 1's copies are not waited for by slot 0's fill


def test_prefetch_refusals():
    src = cs.HostChallengeSource()
    with pytest.raises(ValueError, match="labels but"):
        src.prefetch(H, [1, 2], K, [4])
    with pytest.raises(ValueError, match="no products"):
        src.prefetch(H, [], K, [])
    with pytest.raises(ValueError, match="width is below 1"):
        src.prefetch(H, [1], K, [0])
