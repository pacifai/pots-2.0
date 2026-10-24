"""FS34: one routing for every hash call, and decision 4's per-shape kernel check.

The kernel runs here on CPU tensors through the host build of the core (the ``on_kernel``
seams of ``test_leaves_known``), so the tables, the positions of first errors and the check
are the code a CUDA run uses."""

import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import hostlib  # noqa: F401
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


@pytest.fixture(autouse=True)
def fresh_run():
    kc.new_run()
    yield
    kc.new_run()


def items(leaves):
    return list(enumerate(leaves))


@pytest.fixture
def small_tables(monkeypatch):
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 3)


def test_the_kernel_digests_equal_the_c_librarys_on_lists_and_lazy_input(
        c, leaves, on_kernel, small_tables):  # noqa: F811
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    st = HashStats()
    assert leaf_hashes_of(c, items(leaves), st) == want
    assert st.kernel > 0 and st.crosschecks > 0
    assert leaf_hashes_of(c, iter(items(leaves))) == want
    assert leaf_hashes_of(c, ((i, x) for i, x in enumerate(leaves))) == want
    st = HashStats()
    assert leaf_hashes_of(c, items(leaves), st, kernel=False) == want
    assert st.kernel == 0 and st.crosschecks == 0


def test_hash_step_leaves_takes_known_digests_and_hashes_the_rest(
        c, leaves, on_kernel):  # noqa: F811
    d = leaf_hashes_of(c, items(leaves), kernel=False)
    known = {i: d[i] for i in range(c.n_s, c.n_leaves, 2)}
    st = HashStats()
    assert hash_step_leaves(c, leaves, st, known=known) == d
    assert st.known == len(known) and st.kernel > 0


def test_kernel_false_hashes_with_the_c_library_for_the_commit_calls_too(
        c, leaves, on_kernel, monkeypatch):  # noqa: F811
    """``hash_step_leaves`` and ``commit_leaves`` take ``kernel=``: False gives the C library's
    digests and root and never enqueues a kernel table, with ``known`` digests or without."""
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    tables = []
    real = L._enqueue_table
    monkeypatch.setattr(L, "_enqueue_table", lambda *a: (tables.append(1), real(*a))[1])
    known = {i: want[i] for i in range(c.n_s, c.n_leaves, 2)}
    for kn in (None, known):
        st = HashStats()
        assert hash_step_leaves(c, leaves, st, known=kn, kernel=False) == want
        assert st.kernel == 0
        assert L.commit_leaves(c, leaves, kn, kernel=False).root == L.MerkleTree(want).root
    assert tables == []
    st = HashStats()
    assert hash_step_leaves(c, leaves, st) == want and st.kernel > 0 and tables


def test_a_nan_before_a_host_error_is_the_first_error(
        c, leaves, on_kernel, small_tables):  # noqa: F811
    i = first_product(c)
    from tests.verification.commitment.test_leaves_known import nan_bracket
    for run in (lambda x: leaf_hashes_of(c, items(x)),
                lambda x: hash_step_leaves(c, x)):
        nan_bracket(c, leaves, i, run)


def test_until_error_cuts_the_digests_before_the_first_error(c, leaves, on_kernel):  # noqa: F811
    bad = list(leaves)
    i = first_product(c) + 1
    bad[i] = bad[i].clone()
    bad[i].view(-1)[0] = float("nan")
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    digests, err = leaf_hashes_until_error(c, items(bad))
    assert isinstance(err, NonFiniteError)
    assert digests == want[:i]


def test_an_iterable_error_after_a_table_error_loses(c, leaves, on_kernel):  # noqa: F811
    bad = list(leaves)
    i = first_product(c)
    bad[i] = bad[i].clone()
    bad[i].view(-1)[0] = float("nan")

    def lazy():
        yield from enumerate(bad[:i + 2])
        raise OSError("read failed")

    digests, err = leaf_hashes_until_error(c, lazy())
    assert isinstance(err, NonFiniteError) and len(digests) == i


def test_an_iterable_error_before_a_table_error_wins(c, leaves, on_kernel):  # noqa: F811
    bad = list(leaves)
    i = first_product(c) + 2
    bad[i] = bad[i].clone()
    bad[i].view(-1)[0] = float("nan")

    def lazy():
        yield from enumerate(bad[:i - 1])
        raise OSError("read failed")

    digests, err = leaf_hashes_until_error(c, lazy())
    assert isinstance(err, OSError) and len(digests) == i - 1


def test_a_structural_failure_is_found_on_the_kernel_path(c, leaves, on_kernel):  # noqa: F811
    bad = list(leaves)
    i = first_product(c)
    bad[i] = bad[i][:-1]
    with pytest.raises(LeafShapeError):
        hash_step_leaves(c, bad)


def test_a_mutation_after_adding_to_a_table_is_an_error_at_its_position(
        c, leaves, on_kernel, monkeypatch):  # noqa: F811
    i = first_product(c)
    real = L._enqueue_table

    def mutate_after(idx, tensors, tags):
        tb = real(idx, tensors, tags)
        if i in idx:
            tensors[idx.index(i)].add_(0)  # bumps _version
        return tb

    monkeypatch.setattr(L, "_enqueue_table", mutate_after)
    digests, err = None, None
    with pytest.raises(StoreMutationError, match=f"leaf {i} was mutated in place"):
        hash_step_leaves(c, leaves)


def test_each_key_is_checked_once_per_run_and_path(c, leaves, on_kernel):  # noqa: F811
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    first = st.crosschecks
    assert first > 0
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    assert st.crosschecks == 0
    kc.new_run()
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)
    assert st.crosschecks == first


def test_the_arrival_path_has_its_own_keys(c, leaves, on_kernel):  # noqa: F811
    d = leaf_hashes_of(c, items(leaves), kernel=False)
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st)  # table keys
    known = {i: d[i] for i in range(c.n_s, c.n_leaves)}
    st = HashStats()
    hash_step_leaves(c, leaves, st, known=known)
    assert st.crosschecks > 0
    st = HashStats()
    hash_step_leaves(c, leaves, st, known=known)
    assert st.crosschecks == 0


def test_no_check_without_the_kernel_or_the_native_library(c, leaves, on_kernel):  # noqa: F811
    st = HashStats()
    leaf_hashes_of(c, items(leaves), st, kernel=False)
    assert st.crosschecks == 0
    with native.disabled():
        st = HashStats()
        leaf_hashes_of(c, items(leaves), st)
    assert st.crosschecks == 0 and st.kernel == 0


def test_a_key_with_no_digest_stays_unchecked(c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    nan = leaves[i].clone()
    nan.view(-1)[0] = float("nan")
    st = HashStats()
    digests, err = leaf_hashes_until_error(c, [(i, nan)], st)
    assert isinstance(err, NonFiniteError) and digests == []
    assert st.crosschecks == 0 and not kc._checked


@pytest.fixture
def planted(monkeypatch, on_kernel):  # noqa: F811
    """The kernel answers a wrong digest for every leaf."""
    real = gb.PendingHash.result

    def wrong(self, *a, **k):
        d, bad = real(self, *a, **k)
        return [bytes(32)] * len(d), bad

    monkeypatch.setattr(gb.PendingHash, "result", wrong)


def test_a_planted_mismatch_stops_every_entry(c, leaves, planted):  # noqa: F811
    with pytest.raises(kc.KernelMismatchError):
        leaf_hashes_of(c, items(leaves))
    kc.new_run()
    with pytest.raises(kc.KernelMismatchError):
        leaf_hashes_until_error(c, items(leaves))
    kc.new_run()
    with pytest.raises(kc.KernelMismatchError):
        hash_step_leaves(c, leaves)


def test_a_planted_known_digest_mismatch_stops_hash_step_leaves(c, leaves, on_kernel):  # noqa: F811
    i = first_product(c)
    with pytest.raises(kc.KernelMismatchError) as e:
        hash_step_leaves(c, leaves, known={i: bytes(32)})
    assert "arrival" in str(e.value) and f"leaf {i}" in str(e.value)


def test_a_mismatch_beats_an_earlier_nan(c, leaves, planted):  # noqa: F811
    bad = list(leaves)
    i = first_product(c)
    bad[i] = bad[i].clone()
    bad[i].view(-1)[0] = float("nan")
    with pytest.raises(kc.KernelMismatchError):
        leaf_hashes_until_error(c, items(bad))


def test_the_error_is_no_value_or_lookup_error():
    assert issubclass(kc.KernelMismatchError, RuntimeError)
    assert not issubclass(kc.KernelMismatchError, (ValueError, LookupError))


def test_new_run_clears_the_checked_set():
    kc.mark_checked(("table", torch.device("cpu"), 1, torch.float32, (2,)))
    assert kc.is_checked(("table", torch.device("cpu"), 1, torch.float32, (2,)))
    kc.new_run()
    assert not kc._checked


# ---- whole runs through run_loop, the sharded placement on the host --------------------------

THREE = ("cpu", "cpu", "cpu")


def _scenario_run(pipelined):
    from tests.verification.runs import pipelined_helpers as P
    from tests.verification.runs.test_pipelined_loop import SEED
    from tests.verification.transcript import shared_helpers as H
    from verification.commitment.leaves import dataset_tree
    from verification.runs.scenarios import HONEST, Expected, Scenario, honest_final, run_scenario

    c, D, model, w0 = H.setup(THREE)
    T = 2
    w_T = honest_final(c, D, w0, T, seed=SEED, build_model=lambda: model)
    # kernel=False: a plant must not stop the run before it starts, in this very call
    final = leaf_hashes_of(c, ((c.w_t_index(n), w_T[n]) for n in c.weight_names), kernel=False)
    s = Scenario("x", "", HONEST.fault, Expected())
    assert dataset_tree(c, D).root
    return run_scenario(c, D, w0, s, T=T, k=5, seed=SEED, final=final, pipelined=pipelined,
                        prover=P.DecoderSpec(devices=THREE) if pipelined else None)


@pytest.mark.parametrize("pipelined", [False, True])
def test_a_run_with_the_kernel_seams_on_accepts_and_checks_shapes(
        on_kernel, pipelined):  # noqa: F811
    r = _scenario_run(pipelined)
    assert r.loop.verdict.accepted
    assert kc._checked


@pytest.mark.parametrize("pipelined", [False, True])
def test_a_planted_mismatch_stops_the_run_with_no_verdict(
        planted, monkeypatch, pipelined):  # noqa: F811
    """Every digest is wrong, so the run stops at its first kernel hash: check 0 in
    ``start_run`` (sequential), or the parent's check 0 in ``run_pipelined``. The error comes
    from inside the run, not from the setup around it."""
    import traceback

    from verification.verifier.driver import Verifier

    ends = []
    real = Verifier.end_run
    monkeypatch.setattr(Verifier, "end_run", lambda self, f: (ends.append(1), real(self, f))[1])
    with pytest.raises(kc.KernelMismatchError) as e:
        _scenario_run(pipelined)
    names = [f.name for f in traceback.extract_tb(e.value.__traceback__)]
    assert ("run_pipelined" if pipelined else "run_loop") in names and "start_run" in names
    assert ends == []


def test_run_loop_starts_a_new_run_before_anything_else(on_kernel, monkeypatch):  # noqa: F811
    from verification.verifier.driver import Verifier
    order = []
    monkeypatch.setattr(kc, "new_run", lambda: order.append("new_run"))
    real = Verifier.start_run
    monkeypatch.setattr(Verifier, "start_run",
                        lambda self, ds: (order.append("start_run"), real(self, ds))[1])
    _scenario_run(False)
    assert order[:2] == ["new_run", "start_run"]


def _fail_probe_copies(monkeypatch, nth):
    """Make the ``nth`` per-shape host copy (1-based) raise ``StoreMutationError``; the list
    returned gets the tensor it failed on."""
    seen, failed = [], []

    def boom(t):
        seen.append(t)
        if len(seen) == nth:
            failed.append(t)
            raise StoreMutationError("changed")
        return t

    monkeypatch.setattr(L.hostcopy, "host_tensor", boom)
    return failed


@pytest.mark.parametrize("nth", [1, 2])
def test_a_mutation_on_a_probe_is_the_error_and_no_empty_table_hides_it(
        c, leaves, on_kernel, monkeypatch, nth):  # noqa: F811
    failed = _fail_probe_copies(monkeypatch, nth)
    want = leaf_hashes_of(c, items(leaves), kernel=False)
    digests, err = leaf_hashes_until_error(c, items(leaves))
    i = next(k for k, t in enumerate(leaves) if t is failed[0])
    assert type(err) is StoreMutationError
    assert str(err) == f"leaf {i} was mutated in place while it was copied to the host"
    assert digests == want[:i]


def test_the_pipelined_parent_builds_both_kernels_once_per_cuda_device_before_step_1(
        on_kernel, monkeypatch):  # noqa: F811
    """The child is spawned, so only the parent's calls are seen here; the child's own source
    is checked for the measuring build below."""
    import inspect

    from verification.runs import pipeline as P
    from verification.verifier.driver import Verifier
    from verification.verifier.matmul_check import measure_kernel

    calls = []
    monkeypatch.setattr(P, "_is_cuda", lambda d: True)
    monkeypatch.setattr(P, "GpuBlake3",
                        lambda: type("G", (), {"kernels": lambda s, d:
                                               calls.append(("hash", d))})())
    monkeypatch.setattr(measure_kernel, "kernels", lambda d, k=None: calls.append(("measure", d)))
    real = Verifier.verify_step

    def first(self, t, store):
        calls.append(("step", t))
        return real(self, t, store)

    monkeypatch.setattr(Verifier, "verify_step", first)
    assert _scenario_run(True).loop.verdict.accepted
    cpu = torch.device("cpu")
    assert calls[:3] == [("hash", cpu), ("measure", cpu), ("step", 1)]
    assert sum(k in ("hash", "measure") for k, _ in calls) == 2
    assert "measure_kernel" not in inspect.getsource(P._child_run)
