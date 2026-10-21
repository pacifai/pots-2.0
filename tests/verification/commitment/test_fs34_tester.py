"""FS34 tester: attacks on the join's routing, the per-shape check, the runs and the pair gate.

Built on the seams of ``test_leaves_known`` (the kernel on CPU tensors through the host build of
the core) and the helpers of ``test_fs34_routing``."""

import ctypes
import dataclasses
import inspect
import random
import subprocess
from pathlib import Path

import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from tests.verification.commitment.test_leaves_known import (  # noqa: F401
    c, first_product, leaves, on_kernel,
)
from verification.commitment import gpu_blake3 as gb
from verification.commitment import kernel_check as kc
from verification.commitment import leaves as L
from verification.commitment import native
from verification.commitment.encoding import NonFiniteError
from verification.commitment.leaves import (
    HashStats, hash_step_leaves, leaf_hashes_of, leaf_hashes_until_error,
)
from verification.transcript.errors import LeafShapeError, StoreMutationError

ROOT = Path(__file__).resolve().parents[3]
THREE = ("cpu", "cpu", "cpu")


@pytest.fixture(autouse=True)
def fresh_run():
    kc.new_run()
    yield
    kc.new_run()


def items(leaves):
    return list(enumerate(leaves))


def nan_in(t):
    t = t.clone()
    t.view(-1)[0] = float("nan")
    return t


def outcome(fn):
    """``(digests, class, message)`` of ``leaf_hashes_until_error`` run through ``fn``."""
    d, e = fn()
    return d, None if e is None else type(e), None if e is None else str(e)


# ---- routing equals the plain path -----------------------------------------------------------


@pytest.fixture
def route(monkeypatch, hostlib):  # noqa: F811
    """Kernel routing for the tensors in ``route.ids`` only (the rest are host leaves)."""
    g = gb.GpuBlake3(HostKernels(hostlib))
    ids: set[int] = set()
    monkeypatch.setattr(L, "_hasher", lambda: g)
    monkeypatch.setattr(L, "_kernel_leaf", lambda t: id(t) in ids and t.is_contiguous()
                        and t.numel() > 0 and t.dtype in native.FINITE_KIND)

    class R:
        pass
    r = R()
    r.ids = ids
    return r


ERR_KINDS = ("nan", "shape", "iter")


def _plant_errors(rng, c, base, n_err):
    """A copy of ``base`` with ``n_err`` faults at random tensor leaves (and the position of an
    iterable error, or ``None``)."""
    bad = list(base)
    lo = c.n_s
    for _ in range(n_err):
        i = rng.randrange(lo, len(bad))
        kind = rng.choice(("nan", "inf", "shape"))
        if kind == "shape":
            bad[i] = bad[i][:-1]
        else:
            t = bad[i].clone()
            t.view(-1)[rng.randrange(t.numel())] = float("nan" if kind == "nan" else "inf")
            bad[i] = t
    return bad


@pytest.mark.parametrize("seed", range(40))
def test_routing_equals_the_plain_path_for_random_mixes(c, leaves, route, monkeypatch, seed):  # noqa: F811
    rng = random.Random(seed)
    monkeypatch.setattr(L, "_BATCH_ENTRIES", rng.choice([1, 2, 3, 5, 100]))
    monkeypatch.setattr(L, "_BATCH_BYTES", rng.choice([1000, 3000, 7000, 1 << 20]))
    bad = _plant_errors(rng, c, leaves, rng.choice([0, 0, 1, 2, 3]))
    order = list(range(len(bad)))
    if rng.random() < 0.5:
        rng.shuffle(order)
    cut_at = rng.choice([None, rng.randrange(len(order) + 1)])
    route.ids.clear()
    route.ids.update(id(bad[i]) for i in order if rng.random() < 0.6)
    lazy = rng.random() < 0.5

    def make():
        its = [(i, bad[i]) for i in order]
        if cut_at is None:
            return iter(its) if lazy else its

        def gen():
            yield from its[:cut_at]
            raise OSError("read failed")
        return gen()

    def run(kernel):
        kc.new_run()
        return outcome(lambda: leaf_hashes_until_error(c, make(), kernel=kernel))

    assert run(True) == run(False)
    kc.new_run()
    plain = run(False)
    try:
        got = ("ok", leaf_hashes_of(c, make(), kernel=True))
    except Exception as e:  # noqa: BLE001
        got = ("err", type(e), str(e))
    if plain[1] is None:
        assert got == ("ok", plain[0])
    else:
        assert got == ("err", plain[1], plain[2])


@pytest.mark.parametrize("where", ["before", "inside", "after"])
@pytest.mark.parametrize("kind", ["nan_kernel", "nan_host", "shape", "iter", "mutation"])
@pytest.mark.parametrize("lazy", [False, True])
def test_every_error_kind_placed_around_tables(c, leaves, route, monkeypatch, where, kind,  # noqa: F811
                                               lazy):
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 3)
    n = len(leaves)
    pos = {"before": c.n_s + 1, "inside": n // 2, "after": n - 2}[where]
    bad = list(leaves)
    if kind == "nan_kernel" or kind == "nan_host":
        bad[pos] = nan_in(bad[pos])
    elif kind == "shape":
        bad[pos] = bad[pos][:-1]
    route.ids.update(id(bad[i]) for i in range(n) if i % 2 == 0)
    route.ids.discard(id(bad[pos]))
    if kind != "nan_host":
        route.ids.add(id(bad[pos]))
    its = list(enumerate(bad))

    def stream():
        if kind == "iter":
            def gen():
                yield from its[:pos]
                raise OSError("read failed")
            return gen()
        return iter(its) if lazy else its

    plain = outcome(lambda: leaf_hashes_until_error(c, stream(), kernel=False)) \
        if kind != "mutation" else None
    if kind == "mutation":
        real = L._enqueue_table
        hit = []

        def mutate(idx, tensors, tags):
            tb = real(idx, tensors, tags)
            if pos in idx and not hit:
                hit.append(1)
                tensors[idx.index(pos)].add_(0)
            return tb
        monkeypatch.setattr(L, "_enqueue_table", mutate)
        route.ids.add(id(bad[pos]))
        want = leaf_hashes_of(c, its, kernel=False)
        kc.new_run()
        d, e = leaf_hashes_until_error(c, stream())
        assert isinstance(e, StoreMutationError) and f"leaf {pos} was" in str(e)
        assert d == want[:pos]
        return
    st = HashStats()
    got = outcome(lambda: leaf_hashes_until_error(c, stream(), st, kernel=True))
    assert got == plain
    assert st.kernel > 0 and (kind == "nan_host" or id(bad[pos]) in route.ids)


# ---- no table left unread --------------------------------------------------------------------


@pytest.fixture
def tracked(monkeypatch, on_kernel):  # noqa: F811
    """Every table ``_enqueue_table`` queues, with whether its result was read; ``fail_at`` makes
    the n-th (1-based) call raise, ``fail_read`` makes reading a table raise."""
    real = L._enqueue_table

    class T:
        queued: list = []
        read: list = []
        fail_at = None
        fail_read = None
        calls = 0
    T.queued, T.read = [], []

    def enq(idx, tensors, tags):
        T.calls += 1
        if T.calls == T.fail_at:
            raise RuntimeError("enqueue failed")
        tb = real(idx, tensors, tags)
        k = len(T.queued)
        T.queued.append(tb)
        res = tb.pend.result

        def result(*a, **kw):
            T.read.append(k)
            if T.fail_read is not None:
                raise T.fail_read
            return res(*a, **kw)
        tb.pend.result = result
        return tb

    monkeypatch.setattr(L, "_enqueue_table", enq)
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 2)
    return T


def _all_read(T):
    return sorted(T.read) == list(range(len(T.queued))) and len(T.queued) > 0


def test_an_iterable_error_leaves_no_table_unread(c, leaves, tracked):  # noqa: F811
    def gen():
        yield from items(leaves)[:c.n_s + 7]
        raise OSError("read failed")
    d, e = leaf_hashes_until_error(c, gen())
    assert isinstance(e, OSError) and len(d) == c.n_s + 7
    assert _all_read(tracked)


def test_a_keyboard_interrupt_from_the_iterable_leaves_no_table_unread(c, leaves, tracked):  # noqa: F811
    def gen():
        yield from items(leaves)[:c.n_s + 7]
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        leaf_hashes_until_error(c, gen())
    assert _all_read(tracked)


def test_a_failing_enqueue_still_reads_the_tables_before_it(c, leaves, tracked):  # noqa: F811
    tracked.fail_at = 3
    with pytest.raises(RuntimeError, match="enqueue failed"):
        leaf_hashes_until_error(c, items(leaves))
    assert len(tracked.queued) == 2 and _all_read(tracked)


def test_the_exception_in_flight_wins_over_a_table_that_fails_to_read(c, leaves, tracked):  # noqa: F811
    tracked.fail_read = ValueError("read of a table failed")
    tracked.fail_at = 3
    with pytest.raises(RuntimeError, match="enqueue failed"):
        leaf_hashes_until_error(c, items(leaves))
    assert _all_read(tracked)


def test_an_interrupt_beats_a_table_that_fails_to_read(c, leaves, tracked):  # noqa: F811
    tracked.fail_read = ValueError("read of a table failed")

    def gen():
        yield from items(leaves)[:c.n_s + 7]
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        leaf_hashes_until_error(c, gen())
    assert _all_read(tracked)


def test_a_table_that_fails_to_read_is_raised_when_nothing_else_is_in_flight(c, leaves,  # noqa: F811
                                                                           tracked):
    tracked.fail_read = ValueError("read of a table failed")
    with pytest.raises(ValueError, match="read of a table failed"):
        leaf_hashes_until_error(c, items(leaves))
    assert _all_read(tracked)  # every table read, not just the first


def test_a_native_batch_error_still_reads_every_table(c, leaves, tracked, monkeypatch):  # noqa: F811
    bad = list(leaves)
    # a record leaf that fails leaf_parts is a structural error at its position (a native-batch
    # error of the host leaves): the tables queued before it are still read
    first = first_product(c)
    bad[first + 5] = bad[first + 5][:-1]
    d, e = leaf_hashes_until_error(c, items(bad))
    assert isinstance(e, LeafShapeError) and len(d) == first + 5
    assert _all_read(tracked)


def test_hash_step_leaves_reads_every_table_on_an_exception(c, leaves, tracked):  # noqa: F811
    tracked.fail_at = 4
    with pytest.raises(RuntimeError, match="enqueue failed"):
        hash_step_leaves(c, leaves)
    assert _all_read(tracked)


# ---- the per-shape check ---------------------------------------------------------------------


@pytest.fixture
def planted(monkeypatch, on_kernel):  # noqa: F811
    """The kernel answers a zero digest for every leaf it finds finite."""
    real = gb.PendingHash.result

    def wrong(self, *a, **k):
        d, bad = real(self, *a, **k)
        return [bytes(32)] * len(d), bad

    monkeypatch.setattr(gb.PendingHash, "result", wrong)


def test_the_check_runs_once_per_key_on_each_path(c, leaves, on_kernel, monkeypatch):  # noqa: F811
    calls = []
    real = kc.c_digest
    monkeypatch.setattr(kc, "c_digest", lambda h, t: (calls.append(h), real(h, t))[1])
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    n_table = len(calls)
    assert n_table == st.crosschecks == len(kc._checked) > 0
    d = leaf_hashes_of(c, items(leaves), kernel=False)
    known = {i: d[i] for i in range(c.n_s, c.n_leaves)}
    hash_step_leaves(c, leaves, known=known)
    n_arrival = len(calls) - n_table
    assert n_arrival == len([k for k in kc._checked if k[0] == "arrival"]) > 0
    before = len(calls)
    hash_step_leaves(c, leaves, known=known)
    leaf_hashes_of(c, items(leaves))
    assert len(calls) == before


def test_several_leaves_of_one_unchecked_key_in_one_call_cost_one_comparison(c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    same = [(i, leaves[i]), (i, leaves[i].clone()), (i, leaves[i].clone())]
    st = HashStats()
    leaf_hashes_of(c, same, st)
    assert st.crosschecks == 1 and len(kc._checked) == 1


def test_a_call_outside_a_run_uses_the_set_as_it_stands(c, leaves, on_kernel):  # noqa: F811
    leaf_hashes_of(c, items(leaves))
    snapshot = set(kc._checked)
    assert snapshot
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    leaf_hashes_until_error(c, items(leaves), st)
    hash_step_leaves(c, leaves, st)
    assert st.crosschecks == 0 and kc._checked == snapshot


@pytest.mark.parametrize("entry", ["of", "until", "step"])
@pytest.mark.parametrize("earlier", ["nan", "shape", "iter", "mutation", "none"])
def test_a_table_mismatch_beats_every_other_error_on_each_entry(c, leaves, planted, monkeypatch,  # noqa: F811
                                                                  entry, earlier):
    if entry == "step" and earlier == "iter":
        pytest.skip("hash_step_leaves takes a list")
    i = first_product(c)
    bad = list(leaves)
    if earlier == "nan":
        bad[i] = nan_in(bad[i])
    elif earlier == "shape":
        bad[c.n_s + c.n_w + 1] = bad[c.n_s + c.n_w + 1][:-1]
    if earlier == "mutation":
        real = L._enqueue_table

        def mutate(idx, tensors, tags):
            tb = real(idx, tensors, tags)
            tensors[-1].add_(0)  # the table's last leaf: the digests before it are kept
            return tb
        monkeypatch.setattr(L, "_enqueue_table", mutate)
    its = items(bad)

    def stream():
        if earlier == "iter":
            def gen():
                yield from its[:c.n_s + c.n_w + 2]
                raise OSError("read failed")
            return gen()
        return its
    run = {"of": lambda: leaf_hashes_of(c, stream()),
           "until": lambda: leaf_hashes_until_error(c, stream()),
           "step": lambda: hash_step_leaves(c, bad)}[entry]
    with pytest.raises(kc.KernelMismatchError):
        run()


@pytest.mark.parametrize("earlier", ["nan", "shape"])
def test_a_known_digest_mismatch_beats_an_earlier_host_error(c, leaves, on_kernel, earlier):  # noqa: F811
    bad = list(leaves)
    j = c.n_s  # a weight leaf, before the known product
    bad[j] = nan_in(bad[j]) if earlier == "nan" else bad[j][:-1]
    i = first_product(c)
    with pytest.raises(kc.KernelMismatchError):
        hash_step_leaves(c, bad, known={i: bytes(32)})


def test_a_known_digest_mismatch_beats_a_table_error_after_it(c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    bad = list(leaves)
    bad[i + 3] = nan_in(bad[i + 3])
    with pytest.raises(kc.KernelMismatchError):
        hash_step_leaves(c, bad, known={i: bytes(32)})


def test_a_failed_comparison_does_not_join_the_set(c, leaves, planted):  # noqa: F811
    with pytest.raises(kc.KernelMismatchError):
        leaf_hashes_of(c, items(leaves))
    assert not kc._checked


def test_a_mismatch_message_names_device_header_index_and_both_digests(c, leaves, planted):  # noqa: F811
    i = first_product(c)
    with pytest.raises(kc.KernelMismatchError) as e:
        leaf_hashes_of(c, [(i, leaves[i])])
    m = str(e.value)
    assert "cpu" in m and f"leaf {i}" in m and bytes(32).hex() in m and "shape" in m
    assert "float" in m and "tag" in m


def test_a_key_whose_leaf_had_no_digest_is_checked_by_the_next_call(c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    st = HashStats()
    d, err = leaf_hashes_until_error(c, [(i, nan_in(leaves[i]))], st)
    assert isinstance(err, NonFiniteError) and st.crosschecks == 0
    st = HashStats()
    leaf_hashes_of(c, [(i, leaves[i])], st)
    assert st.crosschecks == 1


def test_a_leaf_mutated_after_its_host_copy_is_not_compared_and_stays_unchecked(  # noqa: F811
        c, leaves, planted, monkeypatch):
    i = first_product(c)
    real = L._enqueue_table

    def mutate(idx, tensors, tags):
        tb = real(idx, tensors, tags)
        tensors[0].add_(0)  # after the host copy, before the table's outcome
        return tb

    monkeypatch.setattr(L, "_enqueue_table", mutate)
    st = HashStats()
    d, err = leaf_hashes_until_error(c, [(i, leaves[i])], st)  # planted, but no digest exists
    assert isinstance(err, StoreMutationError)
    assert st.crosschecks == 0 and not kc._checked


def test_a_mutation_between_the_host_copy_and_the_close_skips_the_comparison(  # noqa: F811
        c, leaves, planted, monkeypatch):
    """The leaf moves after the probe's host copy and before its table is queued: the table
    reads the new bytes, so it is not compared with the old digest: no mismatch, no key."""
    i = first_product(c)
    real = L._host_digest

    def copy_then_mutate(obj, header, index):
        out = real(obj, header, index)
        obj.add_(0)
        return out

    monkeypatch.setattr(L, "_host_digest", copy_then_mutate)
    st = HashStats()
    d = leaf_hashes_of(c, [(i, leaves[i])], st)
    assert len(d) == 1 and st.crosschecks == 0 and not kc._checked
