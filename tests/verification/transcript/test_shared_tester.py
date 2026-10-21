"""FS33a tester: layouts, lifetimes, many storages and guards of the cross-process store.

Layout checks pickle the package through ``ForkingPickler`` (the reducer the pipe uses) in one
process; lifetime checks use a real spawned producer, always with timeouts.
"""

import gc
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler

from setup.records import Record
from verification.commitment.merkle import MerkleTree
from tests.verification.transcript import shared_helpers as H
from tests.verification.transcript import shared_tester_helpers as TH
from verification.transcript.errors import StoreMutationError
from verification.transcript.shared import SharedStore, share_step

CTX = mp.get_context("spawn")
C0 = SimpleNamespace(n_s=0)


def _wire(pkg):
    """What the pipe does to a package."""
    return __import__("pickle").loads(bytes(ForkingPickler.dumps(pkg)))


def _bytes(x):
    return x.detach().clone().reshape(-1).view(torch.uint8).numpy().tobytes()


def _roundtrip(leaves):
    pkg, hold = share_step(C0, 1, leaves, H.synthetic_tree(len(leaves)))
    store = SharedStore.open(C0, _wire(pkg))
    return pkg, hold, store


def _layout_leaves():
    g = torch.Generator().manual_seed(5)
    big = torch.randn(10000, generator=g)
    b16 = torch.randn(100, generator=g).to(torch.bfloat16)
    f32 = torch.randn(6, 8, generator=g)
    return [
        torch.tensor(3.5),                                  # 0-dim
        torch.tensor(7, dtype=torch.int64),                 # 0-dim int
        torch.empty(0), torch.empty(3, 0, 2),               # empty
        torch.randn(1, 1, 1, generator=g),                  # size-1 dims
        torch.randn(1, 5, 1, generator=g).expand(4, 5, 3),  # stride 0
        b16[3:],                                            # offset 6 bytes
        b16[7:9],
        f32,
        f32.view(torch.int32),                              # same storage, other dtype
        f32.view(torch.int16)[:, 1::2],                     # int16, strided over the same storage
        f32[:, 2:7:2].t(),
        big[5000:5003],                                     # small leaf in a big storage
        big[9999],                                          # 0-dim view at the end
        torch.randn(3, 3, generator=g, dtype=torch.float64),
        torch.tensor([True, False, True]),
        torch.randn(4, generator=g).to(torch.float16),
        torch.randn(4, generator=g, dtype=torch.complex64),
        torch.randn(2, 3, generator=g).to(torch.float8_e4m3fn),
    ]


def test_layouts_survive_exactly():
    leaves = _layout_leaves()
    pkg, hold, store = _roundtrip(leaves)
    for i, x in enumerate(leaves):
        y = store.leaf(i)
        assert y.dtype == x.dtype and y.shape == x.shape, i
        assert y.stride() == x.stride() or x.numel() == 0, i
        assert _bytes(y) == _bytes(x), i
    # f32, its int32 and int16 views and its slices: one storage; big and its views: one
    assert pkg.n_storages == len({x.untyped_storage().data_ptr() for x in leaves if x.numel()})
    store.close()
    hold.release()


def test_dtype_views_of_one_storage_share_one_storage_on_the_consumer():
    leaves = _layout_leaves()
    _, hold, store = _roundtrip(leaves)
    a, b, c3 = store.leaf(7), store.leaf(8), store.leaf(9)
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() \
        == c3.untyped_storage().data_ptr()
    store.close()
    hold.release()


def test_unaligned_offset_leaf_reads_the_right_bytes_after_a_real_process():
    conn, child = CTX.Pipe()
    p = CTX.Process(target=H.producer_synthetic, args=(child, "cpu"))
    p.start()
    child.close()
    try:
        assert conn.poll(120)
        pkg, copies, _ = conn.recv()
        store = SharedStore.open(C0, pkg)
        for i, y in enumerate(copies):
            if not isinstance(y, Record):
                assert _bytes(store.leaf(i)) == _bytes(y)
        store.close()
        del pkg
        conn.send("release")
        assert conn.poll(60) and conn.recv() == "ok"
    finally:
        H.reap(p, 5)


def test_a_record_with_a_noncontiguous_or_offset_tensor_round_trips():
    base = torch.arange(10, dtype=torch.int32)
    rec = Record(base[::2][:5].clone(), base[1::2][:5], torch.tensor([0, 1, 0, 1, 1],
                                                                  dtype=torch.int32))
    _, hold, store = _roundtrip([rec])
    got = store.leaf(0)
    assert got == rec
    store.close()
    hold.release()


# ---- lifetimes ---------------------------------------------------------------------------


def test_the_producer_drops_everything_and_reuses_memory_and_the_consumer_reads_right():
    conn, child = CTX.Pipe()
    p = CTX.Process(target=TH.producer_drop, args=(child,))
    p.start()
    child.close()
    try:
        assert conn.poll(120)
        pkg, copies = conn.recv()
        time.sleep(0.5)
        store = SharedStore.open(C0, pkg)
        for i, y in enumerate(copies):
            if not isinstance(y, Record):
                assert _bytes(store.leaf(i)) == _bytes(y)
        store.close()
        del pkg
        conn.send("release")
        assert conn.poll(60) and conn.recv().startswith("ok")
    finally:
        H.reap(p, 5)


def test_release_before_close_on_cpu_leaves_the_consumer_readable():
    """The CPU buffer is the consumer's own mapping; an early release does not hurt it."""
    leaves = H.synthetic_leaves()
    pkg, hold, store = _roundtrip(leaves)
    want = [_bytes(x) for x in leaves if isinstance(x, torch.Tensor)]
    hold.release()
    got = [_bytes(store.leaf(i)) for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
    assert got == want
    store.close()


@pytest.mark.parametrize("mode", ["after_send", "while_held"])
def test_a_producer_that_dies_gives_the_consumer_an_error_or_the_right_bytes(mode):
    conn, child = CTX.Pipe()
    p = CTX.Process(target=TH.producer_dies, args=(child, mode))
    p.start()
    child.close()
    try:
        deadline = time.monotonic() + 120
        if mode == "after_send":
            while p.is_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
        assert conn.poll(60)
        try:
            pkg, copies = conn.recv()
            if mode == "while_held":
                store = SharedStore.open(C0, pkg)
                conn.send("opened")
                while p.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.1)
            else:
                store = SharedStore.open(C0, pkg)
        except (EOFError, OSError, RuntimeError, FileNotFoundError, ConnectionError):
            return  # an error, never wrong bytes: acceptable
        for i, y in enumerate(copies):
            if not isinstance(y, Record):
                assert _bytes(store.leaf(i)) == _bytes(y)
        store.close()
    finally:
        H.reap(p, 5)


# ---- many storages -----------------------------------------------------------------------


def _fds():
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return -1


def test_thousands_of_storages_are_one_buffer_and_flat_in_time_and_descriptors():
    def run(n):
        leaves = [torch.full((3,), float(i)) for i in range(n)]
        tree = MerkleTree([bytes(32)] * n)
        f0 = _fds()
        t0 = time.perf_counter()
        pkg, hold = share_step(C0, 1, leaves, tree)
        t1 = time.perf_counter()
        store = SharedStore.open(C0, _wire(pkg))
        out = [store.leaf(i) for i in range(n)]
        t2 = time.perf_counter()
        f1 = _fds()
        assert pkg.n_storages == n and pkg.cpu is not None
        assert all(float(out[i][0]) == float(i) for i in (0, n // 2, n - 1))
        store.close()
        hold.release()
        return t1 - t0, t2 - t1, f1 - f0

    run(50)  # warm up
    small, big = run(500), run(4000)
    assert big[2] <= small[2] + 2, (small, big)  # descriptors flat
    assert big[0] < 8 * 10 * small[0] + 1.0 and big[1] < 8 * 10 * small[1] + 1.0, (small, big)
    assert big[0] < 5.0 and big[1] < 5.0, big


# ---- guards ------------------------------------------------------------------------------


def test_consumer_write_through_dot_data_is_not_caught_by_version_but_by_the_hash():
    leaves = [torch.arange(6, dtype=torch.float32)]
    _, hold, store = _roundtrip(leaves)
    x = store.leaf(0)
    before = _bytes(x)
    x.data.add_(1)  # bypasses the version counter
    again = store.leaf(0)  # not raised: documented
    assert _bytes(again) != before  # so check 2 (rehash) is what sees it
    store.close()
    hold.release()


def test_consumer_write_through_a_numpy_view_is_not_caught_by_version():
    _, hold, store = _roundtrip([torch.arange(6, dtype=torch.float32)])
    store.leaf(0).numpy()[0] = 99.0
    assert float(store.leaf(0)[0]) == 99.0
    store.close()
    hold.release()


def test_producer_write_through_a_view_of_the_same_storage_is_named_by_release():
    base = torch.arange(12, dtype=torch.float32)
    leaves = [base, base[2:5]]
    _, hold = share_step(C0, 1, leaves, H.synthetic_tree(2))
    base[3:6].add_(1)  # a view made after the share, sharing the version counter
    with pytest.raises(StoreMutationError):
        hold.release()


@pytest.mark.xfail(strict=True, reason="a producer write through a tensor that views the leaf's "
                   "storage but was made with set_ (own version counter) is not detected by "
                   "release")
def test_producer_write_through_an_independent_alias_is_detected():
    base = torch.arange(12, dtype=torch.float32)
    _, hold = share_step(C0, 1, [base], H.synthetic_tree(1))
    alias = torch.empty(0).set_(base.untyped_storage(), 0, (12,))
    alias.add_(1)
    with pytest.raises(StoreMutationError):
        hold.release()


@pytest.mark.xfail(strict=True, reason="a producer write through .data bypasses _version and "
                   "is not detected by release (documented limit; CUDA consumer would read it)")
def test_producer_write_through_dot_data_is_detected():
    base = torch.arange(12, dtype=torch.float32)
    _, hold = share_step(C0, 1, [base], H.synthetic_tree(1))
    base.data.add_(1)
    with pytest.raises(StoreMutationError):
        hold.release()


def test_two_packages_of_one_storage_release_independently():
    base = torch.arange(12, dtype=torch.float32)
    p1, h1 = share_step(C0, 1, [base, base[2:4]], H.synthetic_tree(2))
    p2, h2 = share_step(C0, 2, [base[1:], base], H.synthetic_tree(2))
    s1, s2 = SharedStore.open(C0, _wire(p1)), SharedStore.open(C0, _wire(p2))
    s1.close()
    h1.release()
    gc.collect()
    assert torch.equal(s2.leaf(0), base[1:]) and torch.equal(s2.leaf(1), base)
    s2.close()
    h2.release()


# ---- a real step -------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_a_real_smollm2_step_hashes_the_same_through_the_store(compute_dtype):
    from setup.data import TEST_SCALE_TOKENIZER as pin
    from tests.verification.decoder_helpers import make_records
    from verification.commitment.leaves import hash_step_leaves
    from verification.computation.instances import DecoderComputation
    from verification.prover.step import prove_step
    from verification.transcript.store import InMemoryStore

    c = DecoderComputation.from_pretrained(pin.repo, pin.revision, n_s=4, n=16, eta=1e-3,
                                           compute_dtype=compute_dtype)
    model = c.build_model()
    w0 = {k: p.detach().clone() for k, p in model.named_parameters()}
    out = prove_step(c, model, w0, make_records((16,) * 4))
    leaves, tree = InMemoryStore.commit_step(c, out)
    want = hash_step_leaves(c, leaves)
    pkg, hold = share_step(c, 1, leaves, tree, None)
    store = SharedStore.open(c, _wire(pkg))
    got = hash_step_leaves(c, [store.leaf(i) for i in range(len(leaves))])
    assert got == want and store.root == tree.root
    assert pkg.n_leaves == len(leaves) == 7661
    store.close()
    hold.release()
