"""The cross-process store (FS33a): a step moves from a spawned producer process to this one.

The producer is a real second process (``spawn``, ``shared_helpers.producer``): it runs
``prove_step`` and ``InMemoryStore.commit_step`` on the tiny decoder, shares the step and sends
the package with its own leaf digests. The unit tests on refusals, invariant 6 and idempotence
run in one process (the package is read as it would be after the pipe). The CUDA transport is
in ``test_shared_cuda.py``.
"""

import torch
import torch.multiprocessing as mp
import pytest

from setup.records import Record
from tests.verification.runs.test_sharded_run import K, SEED, THREE, T
from tests.verification.transcript import shared_helpers as H
from verification.commitment.leaves import dataset_tree, hash_step_leaves
from verification.commitment.merkle import MerkleTree
from verification.transcript.errors import StoreMutationError
from verification.transcript.shared import SharedStore, share_step
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

CTX = mp.get_context("spawn")


class Remote:
    """A spawned producer, and the steps it has sent."""

    def __init__(self, devices, device="cpu", steps=2, forged=None, strategy=None, alloc=None):
        self.conn, child = CTX.Pipe()
        self.proc = CTX.Process(target=H.producer, args=(child, devices, device, steps, forged,
                                                         None, strategy, alloc))
        self.proc.start()
        child.close()  # so a dead producer is an EOFError here, not a hang
        self.steps = [self.conn.recv() for _ in range(steps)]

    def ask(self, *msg):
        self.conn.send(msg)
        return self.conn.recv()

    def stop(self):
        try:
            self.conn.send(("quit",))
        except OSError:
            pass
        H.reap(self.proc)


STRATEGIES = mp.get_all_sharing_strategies()  # Linux: file_descriptor and file_system; Mac: one


@pytest.fixture(params=STRATEGIES)
def strategy(request):
    """The CPU sharing strategy, set in this process (the producer sets the same in its own);
    the pod's Linux default is ``file_descriptor``, the Mac has only ``file_system``."""
    old = mp.get_sharing_strategy()
    mp.set_sharing_strategy(request.param)
    yield request.param
    mp.set_sharing_strategy(old)


@pytest.fixture(params=[None, THREE], ids=["unsharded", "three_cpu"])
def remote(request, strategy):
    r = Remote(request.param, strategy=strategy)
    yield request.param, r
    r.stop()


def _verifier(devices):
    c, D, model, w0 = H.setup(devices)
    v = Verifier(c, h_D=dataset_tree(c, D).root, n_records=len(D), k=K, n_steps=T,
                 bands=Bands.provisional(), w0=w0, allow_provisional=True, seed=SEED)
    assert v.start_run(D) is None
    return c, v


def test_consumer_leaves_hash_to_the_producers_digests(remote):
    devices, r = remote
    c, D, _, _ = H.setup(devices)
    dtree = dataset_tree(c, D)
    pkg, digests, root, n = r.steps[0]
    assert pkg.n_leaves == n == len(digests)
    store = SharedStore.open(c, pkg)
    leaves = [store.leaf(i) for i in range(n)]
    got = hash_step_leaves(c, leaves)
    assert got == digests
    assert MerkleTree(got).root == root == store.root
    assert all(store.path(i) == MerkleTree(digests).path(i) for i in range(n))
    from verification.runs.loop import _default_schedule

    idx = list(_default_schedule(c, D, SEED)(1))
    assert [store.dataset_path(j) for j in range(c.n_s)] == [dtree.path(i) for i in idx]
    # storages are never more than tensor leaves (O18: a batched product is one leaf, so the
    # per-member views that used to share storages are gone and a tiny step may share none)
    n_tensors = sum(isinstance(x, torch.Tensor) for x in leaves)
    assert 0 < pkg.n_storages <= n_tensors
    store.close()
    assert r.ask("release", 1) == "ok"


def test_two_live_steps_release_the_first_and_the_second_still_reads(remote):
    devices, r = remote
    c, _, _, _ = H.setup(devices)
    (p1, d1, _, n1), (p2, d2, _, n2) = r.steps
    s1, s2 = SharedStore.open(c, p1), SharedStore.open(c, p2)
    assert hash_step_leaves(c, [s1.leaf(i) for i in range(n1)]) == d1
    s1.close()
    del p1
    assert r.ask("release", 1) == "ok"
    assert hash_step_leaves(c, [s2.leaf(i) for i in range(n2)]) == d2
    s2.close()
    assert r.ask("release", 2) == "ok"


def test_a_producer_write_makes_release_name_the_leaf(remote):
    devices, r = remote
    c, _, _, _ = H.setup(devices)
    pkg = r.steps[0][0]
    store = SharedStore.open(c, pkg)
    i = next(j for j in range(pkg.n_leaves) if not isinstance(store.leaf(j), Record))
    r.ask("mutate", 1, i)
    store.close()
    assert f"leaf {i} was mutated" in r.ask("release", 1)
    assert r.ask("release", 1) == "ok"  # idempotent after the failed release


@pytest.mark.parametrize("devices", [None, THREE], ids=["unsharded", "three_cpu"])
@pytest.mark.parametrize("forged", [None, "product", "linear", "glue"])
def test_verify_step_over_a_shared_store_equals_in_memory(devices, forged):
    r = Remote(devices, steps=1, forged=forged)
    try:
        pkg, digests, root, n = r.steps[0]
        c, v_shared = _verifier(devices)
        store = SharedStore.open(c, pkg)
        # the same step in memory: value copies of the consumer's own leaves
        leaves = [x.clone() if isinstance(x, torch.Tensor) else x
                  for x in (store.leaf(i) for i in range(n))]
        paths = [store.dataset_path(j) for j in range(c.n_s)]
        ref = InMemoryStore.hold(c, leaves, MerkleTree(digests), paths)
        _, v_ref = _verifier(devices)
        rej_ref = v_ref.verify_step(1, ref)
        rej_shared = v_shared.verify_step(1, store)
        assert rej_shared == rej_ref
        assert (rej_ref is None) == (forged is None)
        store.close()
        assert r.ask("release", 1) == "ok"
    finally:
        r.stop()


@pytest.mark.parametrize("device", ["cpu"])
def test_strides_dtype_and_shared_storage_survive_the_process_boundary(device, strategy):
    conn, child = CTX.Pipe()
    p = CTX.Process(target=H.producer_synthetic, args=(child, device, strategy))
    p.start()
    child.close()
    try:
        pkg, copies, _ = conn.recv()
        from types import SimpleNamespace

        store = SharedStore.open(SimpleNamespace(n_s=0), pkg)
        assert pkg.n_storages == 4  # base (3 leaves), bf16, int64, f32; the empty and record: 0
        got = [store.leaf(i) for i in range(len(copies))]
        for x, y in zip(got, copies, strict=True):
            if isinstance(y, Record):
                assert all(torch.equal(a, b) for a, b in zip((x.ids, x.targets, x.mask),
                                                             (y.ids, y.targets, y.mask)))
                continue
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.cpu(), y)
        assert got[1].stride() == (1, 6) and not got[1].is_contiguous()
        assert got[1].untyped_storage().data_ptr() == got[0].untyped_storage().data_ptr()
        assert got[2].storage_offset() == 8 and got[2].stride() == (6, 1)
        store.close()
        del pkg
        conn.send("release")
        assert conn.recv() == "ok"
    finally:
        H.reap(p, 0)


# ---- one process: the module's own rules -------------------------------------------------


def _share(leaves, paths=None, n_s=0):
    from types import SimpleNamespace

    c = SimpleNamespace(n_s=n_s)
    pkg, hold = share_step(c, 1, leaves, H.synthetic_tree(len(leaves)), paths)
    return c, pkg, hold


def test_refusals():
    from types import SimpleNamespace

    c, tree = SimpleNamespace(n_s=0), H.synthetic_tree(1)
    sparse = torch.eye(2).to_sparse()
    for bad in ("a string", 3, sparse, torch.empty(2, device="meta"),
                torch.randn(2, dtype=torch.complex64).conj()):
        with pytest.raises(TypeError):
            share_step(c, 1, [bad], tree)
    with pytest.raises(ValueError):
        share_step(c, 1, [torch.ones(1), torch.ones(1)], tree)  # tree of 1 leaf, 2 given


def test_a_consumer_write_raises_at_the_next_leaf_and_release_is_clean():
    leaves = H.synthetic_leaves()
    c, pkg, hold = _share(leaves)
    store = SharedStore.open(c, pkg)
    x = store.leaf(0)
    assert store.leaf(0) is not None
    # an in-place write on the consumer's view (leaf() hands out detached aliases)
    x.add_(1)
    with pytest.raises(StoreMutationError, match="leaf 0 "):
        store.leaf(0)
    store.close()
    hold.release()  # the producer's own tensors: the copy was written, not the source


def test_the_producer_write_is_named_by_release_and_refs_drop_anyway():
    leaves = H.synthetic_leaves()
    c, pkg, hold = _share(leaves)
    leaves[4].add_(0)
    with pytest.raises(StoreMutationError, match="leaf 4 "):
        hold.release()
    assert hold.released and pkg.cpu is None
    hold.release()  # idempotent: nothing more to raise


def test_close_drops_the_packages_tensors_so_the_memory_can_go():
    leaves = H.synthetic_leaves()
    c, pkg, hold = _share(leaves)
    store = SharedStore.open(c, pkg)
    assert pkg.cpu is not None
    store.leaf(0)
    store.close()
    assert pkg.cpu is None and pkg.cuda == []
    hold.release()


def test_close_and_release_are_idempotent_and_leaf_after_close_raises():
    leaves = H.synthetic_leaves()
    c, pkg, hold = _share(leaves, paths=None)
    store = SharedStore.open(c, pkg)
    assert store.leaf(1).stride() == (1, 6)
    store.close()
    store.close()
    with pytest.raises(RuntimeError):
        store.leaf(0)
    assert store.root == H.synthetic_tree(len(leaves)).root
    hold.release()
    hold.release()


def test_index_and_dataset_path_errors_match_in_memory():
    leaves = H.synthetic_leaves()
    c, pkg, _ = _share(leaves)
    store = SharedStore.open(c, pkg)
    for i in (-1, len(leaves)):
        with pytest.raises(IndexError):
            store.leaf(i)
    with pytest.raises(LookupError):
        store.dataset_path(0)
    assert isinstance(store.leaf(0), torch.Tensor) and not store.leaf(0).requires_grad
    assert isinstance(store.leaf(6), Record)


def test_package_counts_match_the_table():
    leaves = H.synthetic_leaves()
    _, pkg, _ = _share(leaves)
    assert (pkg.n_leaves, pkg.n_storages) == (8, 4)


def test_the_producers_frame_keeps_no_step_after_both_releases(monkeypatch):
    """The one-GPU pair run found 283,136 bytes "kept" after both steps were released: the
    producer's loop bound ``leaves`` per step, so its frame, alive in the command loop, still
    named step 2's. Run the producer in this process on a scripted pipe: share_step and
    prove_step are wrapped to hold weakrefs to every leaf tensor, both steps are released as
    the command loop does, and at the loop's next read (its frame as the test sees it after the
    releases) every leaf tensor that is not the live weights' own must be dead."""
    import gc
    import weakref

    refs, live_w = [], set()
    real_share, real_prove = H.share_step, H.prove_step

    def share(c, t, leaves, tree, paths):
        refs.extend(weakref.ref(x) for x in leaves if isinstance(x, torch.Tensor))
        return real_share(c, t, leaves, tree, paths)

    def prove(*a, **k):
        out = real_prove(*a, **k)
        live_w.clear()
        live_w.update(id(x) for x in out.w_next.values())
        return out

    monkeypatch.setattr(H, "share_step", share)
    monkeypatch.setattr(H, "prove_step", prove)
    left = {}

    class Pipe:
        def __init__(self):
            self.script = [("release", 1), ("release", 2), ("check",)]

        def send(self, x):
            pass

        def recv(self):
            msg = self.script.pop(0)
            if msg[0] == "check":
                gc.collect()
                left["alive"] = [r() for r in refs if r() is not None and id(r()) not in live_w]
                return ("quit",)
            return msg

    H.producer(Pipe(), None, "cpu", 2)
    assert len(refs) > 0
    assert left["alive"] == [], f"{len(left['alive'])} leaf tensors outlive both releases"
