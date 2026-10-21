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
def test_routing_equals_the_plain_path_for_random_mixes(
        c, leaves, route, monkeypatch, seed):  # noqa: F811
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
def test_every_error_kind_placed_around_tables(
        c, leaves, route, monkeypatch, where, kind,  # noqa: F811
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


def test_a_keyboard_interrupt_from_the_iterable_leaves_no_table_unread(
        c, leaves, tracked):  # noqa: F811
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


def test_the_exception_in_flight_wins_over_a_table_that_fails_to_read(
        c, leaves, tracked):  # noqa: F811
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


def test_a_table_that_fails_to_read_is_raised_when_nothing_else_is_in_flight(
        c, leaves,  # noqa: F811
                                                                           tracked):
    tracked.fail_read = ValueError("read of a table failed")
    with pytest.raises(ValueError, match="read of a table failed"):
        leaf_hashes_until_error(c, items(leaves))
    assert _all_read(tracked)  # every table read, not just the first


def test_a_native_batch_error_still_reads_every_table(
        c, leaves, tracked, monkeypatch):  # noqa: F811
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


def test_several_leaves_of_one_unchecked_key_in_one_call_cost_one_comparison(
        c, leaves, on_kernel):  # noqa: F811
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
def test_a_table_mismatch_beats_every_other_error_on_each_entry(
        c, leaves, planted, monkeypatch,  # noqa: F811
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
def test_a_known_digest_mismatch_beats_an_earlier_host_error(
        c, leaves, on_kernel, earlier):  # noqa: F811
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


def test_a_mismatch_message_names_device_header_index_and_both_digests(
        c, leaves, planted):  # noqa: F811
    i = first_product(c)
    with pytest.raises(kc.KernelMismatchError) as e:
        leaf_hashes_of(c, [(i, leaves[i])])
    m = str(e.value)
    assert "cpu" in m and f"leaf {i}" in m and bytes(32).hex() in m and "shape" in m
    assert "float" in m and "tag" in m


def test_a_key_whose_leaf_had_no_digest_is_checked_by_the_next_call(
        c, leaves, on_kernel):  # noqa: F811
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


# ---- whole runs: the child's side, the verifier-only plant -----------------------------------


def _load_host_kernels(so):
    lib = ctypes.CDLL(so)
    lib.pots_host_sizeof_params.restype = ctypes.c_size_t
    lib.pots_host_sizeof_leaf.restype = ctypes.c_size_t
    lib.pots_host_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    lib.pots_host_offsets.restype = ctypes.c_size_t
    lib.host_g.restype = lib.host_r.restype = ctypes.c_uint
    lib.pots_host_stage.argtypes = [ctypes.c_void_p]
    return HostKernels(lib)


def _plant_result():
    real = gb.PendingHash.result

    def wrong(self, *a, **k):
        d, bad = real(self, *a, **k)
        return [bytes(32)] * len(d), bad
    gb.PendingHash.result = wrong


def plant_products(setter, *, caller=None, stack_has=()):
    """A fake kernel that corrupts one key: the digests of the leaves whose header tag is
    ``TAG_PRODUCT``, and no other (``setter(obj, name, value)`` installs it: ``setattr`` or a
    monkeypatch's). ``caller`` (a function name) limits it to tables enqueued directly by that
    function (``"_hash_arrival"``: the capture's arrival hashes only); ``stack_has`` (function
    names) to tables enqueued with one of them on the stack. The result's ``hits`` counts the
    digests it corrupted."""
    import sys
    from types import SimpleNamespace

    from verification.commitment.encoding import TAG_PRODUCT

    real = gb.GpuBlake3.enqueue_table
    plant = SimpleNamespace(hits=0)

    def enqueue(self, hdr_buf, hdr_off, hdr_len, hdr_idx, *a, **k):
        pend = real(self, hdr_buf, hdr_off, hdr_len, hdr_idx, *a, **k)
        if caller is not None and sys._getframe(1).f_code.co_name != caller:
            return pend
        if stack_has and not any(f.function in stack_has for f in inspect.stack()):
            return pend
        bad = [hdr_buf[int(hdr_off[int(h)])] == TAG_PRODUCT for h in hdr_idx]
        honest = pend.result

        def result():
            d, first_bad = honest()
            plant.hits += sum(bad[:len(d)])
            return [bytes(32) if b else x for b, x in zip(bad, d)], first_bad

        pend.result = result
        return pend

    setter(gb.GpuBlake3, "enqueue_table", enqueue)
    return plant


def install_seams(so, plant):
    """The fake kernel and the routing seams, in this process (a spawned child's too).
    ``plant``: True corrupts every digest, ``"product"`` the product leaves' only."""
    g = gb.GpuBlake3(_load_host_kernels(so))
    L._hasher = lambda: g
    L._kernel_leaf = lambda t: (t.is_contiguous() and t.numel() > 0
                                and t.dtype in native.FINITE_KIND)
    if plant == "product":
        plant_products(setattr)
    elif plant:
        _plant_result()


@dataclasses.dataclass(frozen=True)
class SeamedSpec:
    """The tiny decoder's prover spec that installs the kernel seams (and the plant) in the
    child before it builds the prover side; module level, so the child unpickles it."""

    so: str
    plant: bool | str = False
    devices: tuple = THREE

    def __call__(self):
        from tests.verification.runs.pipelined_helpers import DecoderSpec

        install_seams(self.so, self.plant)
        return DecoderSpec(devices=self.devices)()


def _scenario_run(pipelined, prover=None):
    from tests.verification.runs import pipelined_helpers as P
    from tests.verification.runs.test_pipelined_loop import SEED
    from tests.verification.transcript import shared_helpers as H
    from verification.runs.scenarios import HONEST, Expected, Scenario, honest_final, run_scenario

    c, D, model, w0 = H.setup(THREE)
    T = 2
    w_T = honest_final(c, D, w0, T, seed=SEED, build_model=lambda: model)
    final = leaf_hashes_of(c, ((c.w_t_index(n), w_T[n]) for n in c.weight_names), kernel=False)
    s = Scenario("x", "", HONEST.fault, Expected())
    return run_scenario(c, D, w0, s, T=T, k=5, seed=SEED, final=final, pipelined=pipelined,
                        prover=(prover or P.DecoderSpec(devices=THREE)) if pipelined else None)


@pytest.fixture
def so(hostlib):  # noqa: F811
    return hostlib._name


@pytest.fixture
def parent_seams(monkeypatch, hostlib):  # noqa: F811
    """The parent's own seams, planted or honest, through monkeypatch (undone after the test)."""
    def install(plant):
        g = gb.GpuBlake3(HostKernels(hostlib))
        monkeypatch.setattr(L, "_hasher", lambda: g)
        monkeypatch.setattr(L, "_kernel_leaf", lambda t: t.is_contiguous() and t.numel() > 0
                            and t.dtype in native.FINITE_KIND)
        if plant:
            real = gb.PendingHash.result

            def wrong(self, *a, **k):
                d, bad = real(self, *a, **k)
                return [bytes(32)] * len(d), bad
            monkeypatch.setattr(gb.PendingHash, "result", wrong)
    return install


@pytest.fixture
def no_verdict(monkeypatch):
    """Records every ``Verifier.end_run`` call: a run that stops on a mismatch has no verdict."""
    from verification.verifier.driver import Verifier

    calls = []
    real = Verifier.end_run
    monkeypatch.setattr(Verifier, "end_run", lambda self, f: (calls.append(1), real(self, f))[1])
    return calls


def test_a_mismatch_in_the_child_alone_is_a_prover_process_error_and_no_verdict(so, no_verdict):
    from verification.runs.pipeline import ProverProcessError

    assert L._kernel_leaf is not None and not L._kernel_leaf(torch.zeros(3))  # parent: off
    with pytest.raises(ProverProcessError) as e:
        _scenario_run(True, SeamedSpec(so, plant=True))
    assert e.value.kind == "exception"
    msg = str(e.value)
    assert "KernelMismatchError" in msg and "the GPU kernel and the C library disagree" in msg
    assert "cpu" in msg and "leaf" in msg and "shape" in msg
    assert no_verdict == []


def test_the_child_with_the_honest_kernel_and_the_parent_off_accepts(so, no_verdict):
    r = _scenario_run(True, SeamedSpec(so))
    assert r.loop.verdict.accepted and no_verdict == [1]


def test_a_mismatch_in_the_parent_alone_stops_the_run_and_the_child_is_reaped(
        so, parent_seams, no_verdict):
    import multiprocessing

    parent_seams(plant=True)
    with pytest.raises(kc.KernelMismatchError):
        _scenario_run(True, SeamedSpec(so))  # the child's kernel is honest
    assert no_verdict == []
    assert multiprocessing.active_children() == []


def test_both_sides_with_the_honest_kernel_accept(so, parent_seams):
    parent_seams(plant=False)
    r = _scenario_run(True, SeamedSpec(so))
    assert r.loop.verdict.accepted


@pytest.fixture
def verifier_only_plant(monkeypatch, on_kernel):  # noqa: F811
    """A sequential run whose kernel is wrong only when a verifier function (named in
    ``plant.where``) is on the stack. The checked set is emptied as the verifier's step starts,
    standing for its devices being others than the prover's (a key has its device in it); the
    prover's commit is honest. ``plant.rejected`` counts the rejections a context made."""
    from verification.verifier import context as vc
    from verification.verifier.driver import Verifier

    real = gb.PendingHash.result

    class P:
        where: tuple = ()
        rejected: list = []
    P.rejected = []

    def wrong(self, *a, **k):
        d, bad = real(self, *a, **k)
        if any(f.function in P.where for f in inspect.stack()):
            return [bytes(32)] * len(d), bad
        return d, bad

    monkeypatch.setattr(gb.PendingHash, "result", wrong)
    real_vs = Verifier.verify_step
    monkeypatch.setattr(Verifier, "verify_step",
                        lambda self, t, store: (kc._checked.clear(), real_vs(self, t, store))[1])
    real_rej = vc.StepContext.reject
    monkeypatch.setattr(vc.StepContext, "reject",
                        lambda self, *a, **k: (P.rejected.append(a), real_rej(self, *a, **k))[1])
    return P


def _tb_names(exc):
    import traceback

    return {f.name for f in traceback.extract_tb(exc.__traceback__)}


@pytest.mark.parametrize("where, frame", [
    (("check_2_commitment",), "check_2_commitment"),
    (("check_7_chaining",), "check_7_chaining"),
])
def test_a_verifier_only_mismatch_stops_the_sequential_run_without_a_verdict(
        verifier_only_plant, no_verdict, where, frame):
    verifier_only_plant.where = where
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(False)
    assert frame in _tb_names(e.value)
    assert no_verdict == [] and verifier_only_plant.rejected == []


def test_a_mismatch_in_the_check_0_anchor_stops_the_run_at_start(verifier_only_plant, no_verdict):
    verifier_only_plant.where = ("weight_hashes",)
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(False)
    assert "start_run" in _tb_names(e.value) and "weight_hashes" in _tb_names(e.value)
    assert no_verdict == [] and verifier_only_plant.rejected == []


def test_a_wrong_kernel_from_the_start_stops_the_run_at_the_first_hash(  # noqa: F811
        on_kernel, monkeypatch, no_verdict):
    """The first hash of a run is the verifier's anchor in ``start_run``: a wrong kernel stops
    the run there, with no verdict."""
    real = gb.PendingHash.result
    monkeypatch.setattr(gb.PendingHash, "result", lambda self, *a, **k: (
        lambda d, b: ([bytes(32)] * len(d), b))(*real(self, *a, **k)))
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(False)
    assert no_verdict == []
    assert "start_run" in _tb_names(e.value)


def test_check_7_is_not_a_guard_for_the_mismatch(on_kernel):
    """The verifier's prover-data guard converts exactly ``PROVER_DATA_ERRORS``."""
    from verification.verifier.context import PROVER_DATA_ERRORS, _AFTER_COMMIT

    for errors in (PROVER_DATA_ERRORS, _AFTER_COMMIT):
        assert not issubclass(kc.KernelMismatchError, errors)


def test_a_sequential_run_prover_and_verifier_share_one_set_of_keys(on_kernel, monkeypatch):
    calls = []
    real = kc.c_digest
    monkeypatch.setattr(kc, "c_digest", lambda h, t: (calls.append(h), real(h, t))[1])
    r = _scenario_run(False)
    assert r.loop.verdict.accepted
    assert len(calls) == len(kc._checked) > 0


def test_run_loop_clears_the_set_for_the_sequential_and_the_pipelined_dispatch(
        so, on_kernel):
    ghost = ("table", torch.device("cpu"), 99, torch.float32, (1,))
    for pipelined in (False, True):
        kc.mark_checked(ghost)
        _scenario_run(pipelined, SeamedSpec(so) if pipelined else None)
        assert not kc.is_checked(ghost)


# ---- whole runs: a kernel that is wrong for one key only (the design's "Tests and the gate") --


@pytest.fixture
def keys_clear_at_verify(monkeypatch):
    """The checked set is emptied as the verifier's step starts, standing for its devices being
    others than the prover's (a key has its device in it)."""
    from verification.verifier.driver import Verifier

    real = Verifier.verify_step
    monkeypatch.setattr(Verifier, "verify_step",
                        lambda self, t, store: (kc._checked.clear(), real(self, t, store))[1])


def test_a_product_only_plant_corrupts_products_and_nothing_else(c, leaves, on_kernel,
                                                                 monkeypatch):  # noqa: F811
    plant = plant_products(monkeypatch.setattr)
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    with pytest.raises(kc.KernelMismatchError, match="tag 0x"):
        leaf_hashes_of(c, items(leaves))
    assert plant.hits > 0
    kc.new_run()
    weights = [(i, leaves[i]) for i in range(c.n_s, c.n_s + c.n_w)]
    assert leaf_hashes_of(c, weights) == [want[i] for i, _ in weights]


def _in_run(exc, *names):
    got = _tb_names(exc)
    assert names and all(n in got for n in names), (names, sorted(got))


def test_a_product_plant_in_the_commit_stops_the_sequential_run(
        on_kernel, monkeypatch, no_verdict):  # noqa: F811
    plant = plant_products(monkeypatch.setattr, stack_has=("commit_step",))
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(False)
    _in_run(e.value, "run_loop", "commit_step", "hash_step_leaves")
    assert plant.hits > 0 and no_verdict == []


def test_a_product_plant_in_the_childs_commit_is_a_prover_process_error(so, no_verdict):
    from verification.runs.pipeline import ProverProcessError

    with pytest.raises(ProverProcessError) as e:
        _scenario_run(True, SeamedSpec(so, plant="product"))
    msg = str(e.value)
    assert e.value.kind == "exception" and "KernelMismatchError" in msg
    assert "commit_step" in msg and "tag 0x03" in msg  # the commit, not the handshake
    assert no_verdict == []


def test_a_product_plant_in_check_2_stops_the_sequential_run(
        on_kernel, monkeypatch, no_verdict, keys_clear_at_verify):  # noqa: F811
    plant = plant_products(monkeypatch.setattr, stack_has=("check_2_commitment",))
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(False)
    _in_run(e.value, "run_loop", "verify_step", "check_2_commitment")
    assert plant.hits > 0 and no_verdict == []


def test_a_product_plant_in_check_2_of_the_pipelined_parent_stops_the_run(
        so, parent_seams, monkeypatch, no_verdict):
    import multiprocessing

    parent_seams(plant=False)
    plant = plant_products(monkeypatch.setattr, stack_has=("check_2_commitment",))
    spec = SeamedSpec(so)
    assert spec.plant is False  # the child's kernel is honest
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(True, spec)
    _in_run(e.value, "run_pipelined", "check_2_commitment")
    assert plant.hits > 0 and no_verdict == []
    assert multiprocessing.active_children() == []


# ---- the kernels load before start_run, outside check 0's section ------------------------------


def _preload_order(monkeypatch, pipelined):
    """Every kernel load and every kernel table of a run, in order, with the section ``run:0``
    (check 0's) if it is open at the time."""
    import traceback

    from verification.runs import loop
    from verification.runs import pipeline as P
    from verification.verifier.matmul_check import measure_kernel

    order = []
    real = L._enqueue_table

    def table(idx, tensors, tags):
        in_start = "start_run" in [f.name for f in traceback.extract_stack()]
        order.append(("table", in_start))
        return real(idx, tensors, tags)

    monkeypatch.setattr(L, "_enqueue_table", table)
    mod = P if pipelined else loop
    monkeypatch.setattr(mod, "_is_cuda" if pipelined else "_on_cuda", lambda d: True)
    monkeypatch.setattr(mod, "GpuBlake3", lambda: type(
        "G", (), {"kernels": lambda s, d: order.append(("hash", str(d)))})())
    monkeypatch.setattr(measure_kernel, "kernels", lambda d: order.append(("measure", str(d))))
    assert _scenario_run(pipelined).loop.verdict.accepted
    return order


@pytest.mark.parametrize("pipelined", [False, True])
def test_both_kernels_load_before_start_run_hashes_w0(
        on_kernel, monkeypatch, pipelined):  # noqa: F811
    order = _preload_order(monkeypatch, pipelined)
    first = next(i for i, o in enumerate(order) if o[0] == "table")
    loads = [o[0] for o in order[:first]]
    assert order[first] == ("table", True)  # the first table is check 0's hash of W_0
    assert loads.count("hash") >= 1 and loads.count("measure") >= 1, order[:first + 1]


# ---- the pair session's FS34 item ------------------------------------------------------------


def _pair():
    from tests.verification.runs import test_fs32_pair_session as P
    return P


def _lines(parts):
    from verification.runs import fs11_session as fs

    pair = fs.fs32_pair(parts)
    return pair, "\n".join(fs._pair_lines(pair))


def test_fs34_gate_fails_closed_on_a_missing_part():
    P = _pair()
    parts = [p for p in P._parts() if p["name"] != "fs34_tests"]
    pair, text = _lines(parts)
    assert not pair["pass"]
    assert pair["fs34_problems"], "a missing part must give reasons"
    assert "FS32 FAIL: FS34 test the FS34 tests part is not_started" in text
    assert "FS32 FAIL: FS34 test the FS34 tests left no junit report" in text


@pytest.mark.parametrize("status", ["failed", "timeout", "not_started", "interrupted"])
def test_fs34_gate_fails_on_a_part_that_did_not_end_ok_even_if_every_case_passed(status):
    P = _pair()
    part = P._fs34()
    part["status"] = status
    pair, text = _lines(P._parts(fs34_tests=part))
    assert not pair["pass"]
    assert f"FS32 FAIL: FS34 test the FS34 tests part is {status}" in text


def test_fs34_gate_accepts_skipped_done_with_every_case_passing():
    P = _pair()
    part = P._fs34()
    part["status"] = "skipped_done"
    assert _lines(P._parts(fs34_tests=part))[0]["pass"]


@pytest.mark.parametrize("extras", [None, {}, {"junit": None}, {"extras_error": "OSError: x"},
                                    {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}])
def test_fs34_gate_fails_on_a_missing_or_empty_junit_report(extras):
    P = _pair()
    part = P._part("fs34_tests", "fs34_tests", extras)
    pair, text = _lines(P._parts(fs34_tests=part))
    assert not pair["pass"]
    assert "FS32 FAIL: FS34 test the FS34 tests left no junit report" in text


def test_fs34_gate_fails_on_a_junit_with_no_cases():
    P = _pair()
    part = P._part("fs34_tests", "fs34_tests",
                   {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "junit": 1,
                    "cases": {}})
    pair, text = _lines(P._parts(fs34_tests=part))
    assert not pair["pass"]
    for n in P.EXPECTED34:
        assert f"FS32 FAIL: FS34 test {n}: expected but not in the junit report" in text


@pytest.mark.parametrize("outcome", ["skipped", "failed", "error"])
def test_each_case_that_did_not_pass_has_its_own_fail_line(outcome):
    P = _pair()
    bad = {n: outcome for n in P.EXPECTED34[:2]}
    pair, text = _lines(P._parts(fs34_tests=P._fs34(bad)))
    assert not pair["pass"]
    for n in bad:
        assert f"FS32 FAIL: FS34 test {n}: {outcome}" in text
    assert text.count("FS32 FAIL: FS34 test") == 2


def test_a_junit_case_the_file_does_not_define_that_failed_is_a_problem():
    P = _pair()
    pair, text = _lines(P._parts(fs34_tests=P._fs34({"tests.x.test_stale::test_gone": "failed"})))
    assert not pair["pass"]
    assert "FS32 FAIL: FS34 test tests.x.test_stale::test_gone: failed" in text


def test_a_junit_case_the_file_does_not_define_that_passed_does_not_stand_in_for_a_missing_one():
    P = _pair()
    part = P._fs34()
    gone = P.EXPECTED34[0]
    del part["extras"]["cases"][gone]
    part["extras"]["cases"]["tests.x.test_other::test_renamed"] = "passed"
    pair, text = _lines(P._parts(fs34_tests=part))
    assert not pair["pass"]
    assert f"FS32 FAIL: FS34 test {gone}: expected but not in the junit report" in text


def test_the_counts_of_the_junit_fail_the_gate_even_when_the_cases_all_read_passed():
    P = _pair()
    part = P._fs34()
    part["extras"]["failures"] = 1
    pair, text = _lines(P._parts(fs34_tests=part))
    assert not pair["pass"] and "1 failures" in text


def test_the_real_file_defines_the_cases_the_gate_expects():
    from verification.runs import fs11_session as fs

    names, err = fs.device_tests_expected(fs.FS34_TESTS_FILE)
    assert not err and names
    src = (ROOT / fs.FS34_TESTS_FILE).read_text()
    assert all(n.split("::")[1].split("[")[0] in src for n in names)


def test_an_empty_junit_file_is_a_missing_report_not_a_crash(tmp_path):
    from verification.runs import fs11_session as fs

    junit = tmp_path / "junit.xml"
    junit.write_text("")
    part = fs.Part("fs34_tests", ["x"], env={}, timeout_s=1, kind="fs34_tests",
                   info={"junit": str(junit)})
    extras = fs.part_extras(part, tmp_path / "log.txt")
    assert "cases" not in extras


def test_the_pair_deadline_is_twice_the_estimate_rounded_up_to_100_s():
    from verification.runs import fs11_session as fs

    d = fs.pair_default_deadline_s()
    assert d == 16100 and "DEFAULT_DEADLINE_S=16100" in (
        ROOT / "verification/runs/fs11_job.sh").read_text()
    assert sum(fs.PAIR_ESTIMATE_MIN.values()) + fs.PAIR_PACK_MIN + fs.PAIR_GRACE_MIN == 134


# ---- scope of the check and the keyword ------------------------------------------------------


def test_under_native_disabled_no_entry_uses_the_kernel_or_checks(
        c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    with native.disabled():
        st = HashStats()
        leaf_hashes_until_error(c, items(leaves), st)
        hash_step_leaves(c, leaves, st, known={i: bytes(32)})  # a wrong known digest is taken
        assert st.crosschecks == 0 and st.kernel == 0 and not kc._checked


def test_kernel_false_on_every_entry_and_on_native_until_error(c, leaves, on_kernel):  # noqa: F811
    st = HashStats()
    leaf_hashes_until_error(c, items(leaves), st, kernel=False)
    d, err = L._native_until_error(c, items(leaves), st, kernel=False)
    assert err is None and st.kernel == 0 and st.crosschecks == 0 and not kc._checked
    assert d == leaf_hashes_of(c, items(leaves))


def test_hashstats_kernel_counts_the_leaves_the_kernel_hashed_on_either_entry(
        c, leaves, on_kernel):  # noqa: F811
    n_t = len(leaves) - c.n_s
    a, b, s = HashStats(), HashStats(), HashStats()
    leaf_hashes_of(c, items(leaves), a)
    leaf_hashes_until_error(c, items(leaves), b)
    hash_step_leaves(c, leaves, s)
    assert a.kernel == b.kernel == s.kernel == n_t
    d = leaf_hashes_of(c, items(leaves), kernel=False)
    s2 = HashStats()
    known = {i: d[i] for i in range(c.n_s, c.n_leaves, 2)}
    hash_step_leaves(c, leaves, s2, known=known)
    assert s2.kernel == n_t - len(known) and s2.known == len(known)


def test_the_wait_for_the_tables_counts_in_wait_s(c, leaves, on_kernel, monkeypatch):  # noqa: F811
    import time

    real = gb.PendingHash.result

    def slow(self, *a, **k):
        time.sleep(0.05)
        return real(self, *a, **k)
    monkeypatch.setattr(gb.PendingHash, "result", slow)
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    assert st.wait_s >= 0.05


def test_a_known_digest_of_a_host_product_gets_no_crosscheck(c, leaves):
    # By design: on one GPU the products sit in host memory and check 2 rehashes them with the C
    # library, so a wrong prover digest is a root mismatch there and decision 4 isn't needed.
    i = first_product(c)
    st = HashStats()
    hash_step_leaves(c, leaves, st, known={i: bytes(32)})
    assert st.crosschecks == 0 and not kc._checked


def test_a_wrong_arrival_digest_on_a_host_product_today_changes_the_root(c, leaves):
    i = first_product(c)
    good = hash_step_leaves(c, leaves)
    got = hash_step_leaves(c, leaves, known={i: bytes(32)})
    assert got[i] == bytes(32) and got != good  # the choice's cost, pinned


def test_a_mutation_while_taking_the_checks_host_copy_is_an_error_at_that_leaf(  # noqa: F811
        c, leaves, on_kernel, monkeypatch):
    from verification.commitment import hostcopy

    i = first_product(c) + 2
    real = hostcopy.host_tensor

    def mutating(t):
        if t is leaves[i]:
            raise StoreMutationError("moved")
        return real(t)

    monkeypatch.setattr(hostcopy, "host_tensor", mutating)
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    d, err = leaf_hashes_until_error(c, items(leaves))
    assert isinstance(err, StoreMutationError) and f"leaf {i} was mutated in place" in str(err)
    assert d == want[:i]


def test_another_failure_of_the_checks_host_copy_propagates_after_the_tables_are_read(  # noqa: F811
        c, leaves, tracked, monkeypatch):
    from verification.commitment import hostcopy

    i = first_product(c) + 4
    real = hostcopy.host_tensor

    def boom(t):
        if t is leaves[i]:
            raise OSError("copy failed")
        return real(t)

    monkeypatch.setattr(hostcopy, "host_tensor", boom)
    with pytest.raises(OSError, match="copy failed"):
        leaf_hashes_until_error(c, items(leaves))
    assert all(r in tracked.read for r in range(len(tracked.queued)))
