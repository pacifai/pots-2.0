"""FS30: ``hash_step_leaves(known=...)`` and the device-leaf kernel routing, on the MLP computation.

The kernel path runs here on CPU tensors through the host build of the core: the tests swap
``leaves._kernel_leaf`` and ``leaves._hasher`` (the module's two seams), so the table building,
the digests and the error precedence are the code a CUDA run uses. Reading a real device's
bytes is in ``tests/verification/runs/test_device_leaves_cuda.py``."""

import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from verification.commitment import gpu_blake3 as gb
from verification.commitment import leaves as L
from verification.commitment.leaves import HashStats, commit_leaves, hash_step_leaves
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.prover.step import prove_step
from verification.transcript.errors import LeafShapeError, StoreMutationError
from verification.commitment.encoding import NonFiniteError


@pytest.fixture
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-2)


@pytest.fixture
def leaves(c):
    data = synthetic_dataset(c.widths, 12, seed=0)
    return list(prove_step(c, c.build_model(), init_weights(c.widths, seed=0), data[4:8]).leaves())


def plain(c, leaves):
    try:
        return hash_step_leaves(c, leaves), None
    except Exception as e:  # noqa: BLE001
        return None, e


def same_outcome(got, want):
    assert (got[1] is None) == (want[1] is None)
    if want[1] is None:
        assert got[0] == want[0]
    else:
        assert type(got[1]) is type(want[1]) and str(got[1]) == str(want[1])


def first_product(c):
    return c.product_index(1)


@pytest.fixture
def on_kernel(monkeypatch, hostlib):  # noqa: F811
    """CPU float tensors at tensor slots go to the kernel path, on the host build of the core."""
    g = gb.GpuBlake3(HostKernels(hostlib))
    monkeypatch.setattr(L, "_hasher", lambda: g)
    monkeypatch.setattr(L, "_kernel_leaf", lambda t: t.is_contiguous() and t.numel() > 0
                        and t.dtype in L.native.FINITE_KIND)
    return g


def test_known_digests_give_the_plain_root(c, leaves):
    d = hash_step_leaves(c, leaves)
    known = {i: d[i] for i in range(c.n_s, c.n_leaves)}
    stats = HashStats()
    assert hash_step_leaves(c, leaves, stats, known=known) == d
    assert stats.known == len(known)
    assert commit_leaves(c, leaves, known).root == commit_leaves(c, leaves).root


def test_a_wrong_known_digest_changes_the_root(c, leaves):
    i = first_product(c)
    known = {i: bytes(32)}
    assert commit_leaves(c, leaves, known).root != commit_leaves(c, leaves).root


def test_a_known_digest_of_a_record_slot_is_ignored(c, leaves):
    assert commit_leaves(c, leaves, {0: bytes(32)}).root == commit_leaves(c, leaves).root


def test_structural_error_before_a_host_nan_wins(c, leaves):
    bad = list(leaves)
    s, nan = first_product(c), first_product(c) + 1
    bad[s] = bad[s][:-1]  # wrong shape
    bad[nan] = bad[nan].clone()
    bad[nan].view(-1)[0] = float("nan")
    known = {i: bytes(32) for i in range(c.n_s + c.n_w, c.n_leaves)}
    got = plain(c, bad)
    assert isinstance(got[1], LeafShapeError)
    with pytest.raises(LeafShapeError) as e:
        hash_step_leaves(c, bad, known=known)
    assert str(e.value) == str(got[1])


def test_a_host_nan_before_a_structural_error_wins(c, leaves):
    bad = list(leaves)
    nan, s = first_product(c), first_product(c) + 1
    bad[nan] = bad[nan].clone()
    bad[nan].view(-1)[0] = float("nan")
    bad[s] = bad[s][:-1]
    got = plain(c, bad)
    assert isinstance(got[1], NonFiniteError)
    with pytest.raises(NonFiniteError) as e:
        hash_step_leaves(c, bad, known={s + 1: bytes(32)})
    assert str(e.value) == str(got[1])


def test_a_planted_nan_in_a_host_leaf_gives_the_plain_digests_and_error(c, leaves):
    bad = list(leaves)
    i = first_product(c) + 2
    bad[i] = bad[i].clone()
    bad[i].view(-1)[3] = float("inf")
    d = hash_step_leaves(c, leaves)
    known = {j: d[j] for j in range(c.n_s + c.n_w, i)}  # known leaves before the bad one
    want = plain(c, bad)
    with pytest.raises(NonFiniteError) as e:
        hash_step_leaves(c, bad, known=known)
    assert str(e.value) == str(want[1])


def test_the_kernel_path_gives_the_plain_digests(c, leaves, on_kernel):
    stats = HashStats()
    got = hash_step_leaves(c, leaves, stats)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(L, "_kernel_leaf", lambda t: False)
        want = hash_step_leaves(c, leaves)
    assert got == want
    assert stats.kernel == c.n_leaves - c.n_s


def test_the_kernel_path_in_small_tables_and_with_known(c, leaves, on_kernel, monkeypatch):
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 2)
    want = commit_leaves(c, leaves).root
    d = hash_step_leaves(c, leaves)
    stats = HashStats()
    known = {first_product(c): d[first_product(c)]}
    assert commit_leaves(c, leaves, known).root == want
    hash_step_leaves(c, leaves, stats, known=known)
    assert (stats.known, stats.kernel) == (1, c.n_leaves - c.n_s - 1)


@pytest.mark.parametrize("where", ["first", "last"])
def test_a_nan_in_a_kernel_leaf_is_the_plain_error(c, leaves, on_kernel, monkeypatch, where):
    i = first_product(c) if where == "first" else c.n_leaves - 1
    bad = list(leaves)
    bad[i] = bad[i].clone()
    bad[i].view(-1)[1] = float("nan")
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 3)
    got = plain(c, bad)
    with monkeypatch.context() as mp:
        mp.setattr(L, "_kernel_leaf", lambda t: False)
        want = plain(c, bad)
    same_outcome(got, want)
    assert isinstance(got[1], NonFiniteError)


def test_the_smallest_failing_leaf_wins_across_kernel_and_host(c, leaves, on_kernel, monkeypatch):
    """A kernel leaf's NaN at index 9 and a host structural error at index 7."""
    bad = list(leaves)
    a, b = first_product(c) + 1, first_product(c) + 3
    bad[a] = bad[a][:-1]
    bad[b] = bad[b].clone()
    bad[b].view(-1)[0] = float("nan")
    with pytest.raises(LeafShapeError):
        hash_step_leaves(c, bad)
    bad = list(leaves)
    bad[a] = bad[a].clone()
    bad[a].view(-1)[0] = float("nan")
    bad[b] = bad[b][:-1]
    with pytest.raises(NonFiniteError):
        hash_step_leaves(c, bad)


def test_a_mutation_while_hashing_is_a_store_mutation_error(c, leaves, on_kernel, monkeypatch):
    real = L._enqueue_table

    def mutate_after(idx, tensors, tags):
        tb = real(idx, tensors, tags)
        if first_product(c) in idx:
            tensors[idx.index(first_product(c))].add_(0)  # bumps _version
        return tb

    monkeypatch.setattr(L, "_enqueue_table", mutate_after)
    with pytest.raises(StoreMutationError, match=f"leaf {first_product(c)} was mutated in place"):
        hash_step_leaves(c, leaves)


def test_a_mutation_at_or_before_the_first_bad_leaf_beats_it(c, leaves, on_kernel, monkeypatch):
    i = first_product(c)
    bad = list(leaves)
    bad[i] = bad[i].clone()
    bad[i].view(-1)[0] = float("nan")
    real = L._enqueue_table

    def mutate_after(idx, tensors, tags):
        tb = real(idx, tensors, tags)
        tensors[idx.index(i)].add_(0) if i in idx else None
        return tb

    monkeypatch.setattr(L, "_enqueue_table", mutate_after)
    with pytest.raises(StoreMutationError):
        hash_step_leaves(c, bad)


def test_a_non_contiguous_leaf_with_a_known_digest_raises_the_plain_error(c, leaves):
    i = first_product(c)
    t = leaves[i]
    assert t.dim() == 2 and min(t.shape) > 1
    nc = torch.empty(tuple(t.shape)[::-1], dtype=t.dtype).t()
    nc.copy_(t)
    assert not nc.is_contiguous()
    bad = list(leaves)
    bad[i] = nc
    want = plain(c, bad)
    assert want[1] is not None
    try:
        hash_step_leaves(c, bad, known={i: bytes(32)})
    except Exception as e:  # noqa: BLE001
        got = (None, e)
    else:
        got = (0, None)
    same_outcome(got, want)


def test_prove_step_counts_the_digests_in_use_after_a_perturbation(c, monkeypatch):
    """``capture["arrival_hash"]["used"]`` is ``len(arrival_digests)``, after a perturbed
    product's digest is dropped; ``leaves`` (the hashed count) is left alone. The capture's
    two methods are faked, so this runs on a CPU."""
    from verification.prover.capture import MatmulCapture

    real = MatmulCapture.summary
    monkeypatch.setattr(MatmulCapture, "arrival_digest", lambda self, p: bytes(32))
    monkeypatch.setattr(MatmulCapture, "summary", lambda self: {
        **real(self), "arrival_hash": {"records": 1, "leaves": c.M, "bytes": 0, "nonfinite": 0}})
    data = synthetic_dataset(c.widths, 12, seed=0)
    w0 = init_weights(c.widths, seed=0)
    out = prove_step(c, c.build_model(), w0, data[4:8], perturb={1: lambda t: t + 1})
    assert len(out.arrival_digests) == c.M - 1
    assert out.capture["arrival_hash"]["used"] == c.M - 1
    assert out.capture["arrival_hash"]["leaves"] == c.M


@pytest.fixture
def table_log(c, leaves, on_kernel, monkeypatch):
    """Small tables, and the queue and the read of each table logged (the real ones run)."""
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 2)
    log = {"queued": [], "read": []}
    enqueue, outcome = L._enqueue_table, L._table_outcome

    def queue(idx, ts, tags):
        log["queued"].append(idx)
        return enqueue(idx, ts, tags)

    def read(tb):
        log["read"].append(tb.idx)
        return outcome(tb)

    monkeypatch.setattr(L, "_enqueue_table", queue)
    monkeypatch.setattr(L, "_table_outcome", read)
    return log


def test_every_table_is_read_and_the_first_outcome_error_surfaces(c, leaves, table_log,
                                                                  monkeypatch):
    outcome = L._table_outcome
    read = []

    def failing(tb):
        read.append(tb.idx)
        if len(read) in (1, 3):
            raise RuntimeError(f"outcome of {tb.idx}")
        return outcome(tb)

    monkeypatch.setattr(L, "_table_outcome", failing)
    with pytest.raises(RuntimeError) as e:
        hash_step_leaves(c, leaves)
    assert len(table_log["queued"]) > 3
    assert str(e.value) == f"outcome of {table_log['queued'][0]}"
    assert read == table_log["queued"]


def test_an_exception_in_flight_beats_an_outcome_error(c, leaves, table_log, monkeypatch):
    def host_fails(*a, **k):
        raise ValueError("host path")

    def outcome_fails(tb):
        table_log["read"].append(tb.idx)
        raise RuntimeError("outcome")

    monkeypatch.setattr(L, "leaf_hashes_until_error", host_fails)
    monkeypatch.setattr(L, "_table_outcome", outcome_fails)
    with pytest.raises(ValueError, match="host path"):
        hash_step_leaves(c, leaves)
    assert len(table_log["queued"]) > 1
    assert table_log["read"] == table_log["queued"]


def test_an_enqueue_that_fails_still_reads_the_tables_queued_before(c, leaves, table_log,
                                                                    monkeypatch):
    enqueue = L._enqueue_table
    calls = []

    def second_fails(idx, ts, tags):
        calls.append(idx)
        if len(calls) == 2:
            raise RuntimeError("enqueue")
        return enqueue(idx, ts, tags)

    monkeypatch.setattr(L, "_enqueue_table", second_fails)
    with pytest.raises(RuntimeError, match="enqueue"):
        hash_step_leaves(c, leaves)
    assert len(calls) == 2
    assert table_log["read"] == [calls[0]]
