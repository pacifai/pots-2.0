"""The CUDA transport of the cross-process store (FS33a). Skipped without CUDA; desk-checked,
it first runs on the two-GPU pod (FS33d).

The producer is a spawned process; its sharded placement (``placement_for``) puts W and
product leaves on the verifier devices (``cuda:0`` for one GPU, the prover's own device) and
records on the host (they go by value). An unsharded computation puts every leaf on the host,
so it takes the CPU path (see the last test). Nothing is copied: the
producer's CUDA memory does not grow by the transcript when it shares, and a write the producer
makes after the consumer opened the step shows in the consumer's read (the outside proof of "no
copy"). Every test runs with the allocator setting unset and with the full-scale setting
``PYTORCH_ALLOC_CONF=expandable_segments:True`` in the producer, whose CUDA IPC path differs
(``shared.py`` docstring, "Expandable segments"). The consumer's ``memory_allocated`` is not
asserted: memory opened by handle does not come from the caching allocator, so it never counts.
"""

import gc
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from tests.verification.transcript import shared_helpers as H
from tests.verification.transcript.test_shared import Remote
from verification.commitment.leaves import hash_step_leaves
from verification.commitment.merkle import MerkleTree
from verification.transcript.shared import SharedStore

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CTX = mp.get_context("spawn")
TWO = torch.cuda.device_count() >= 2
ALLOCS = [pytest.param(None, id="alloc_default"),
          pytest.param("expandable_segments:True", id="alloc_expandable")]


def _cases():
    yield pytest.param(("cuda:0",), "cuda:0", id="one_gpu")
    yield pytest.param(("cuda:1",), "cuda:0", id="two_gpus",
                       marks=pytest.mark.skipif(not TWO, reason="needs a second GPU"))


@pytest.mark.parametrize("alloc", ALLOCS)
@pytest.mark.parametrize(("devices", "device"), list(_cases()))
def test_cuda_step_hashes_reads_and_returns_the_memory(devices, device, alloc):
    r = Remote(devices, device=device, steps=2, alloc=alloc)
    try:
        c, _, _, _ = H.setup(devices, device)
        (p1, d1, root1, n1), (p2, d2, _, n2) = r.steps
        assert p1.cuda and p1.cpu is None and p1.n_storages == len(p1.cuda)
        s1, s2 = SharedStore.open(c, p1), SharedStore.open(c, p2)
        leaves = [s1.leaf(i) for i in range(n1)]
        assert hash_step_leaves(c, leaves) == d1
        assert MerkleTree(d1).root == root1 == s1.root
        del leaves
        s1.close()
        assert p1.cuda == [] and p1.cpu is None, "close must empty the package's tensors"
        gc.collect()
        assert r.ask("release", 1) == "ok"  # step 1 gone, step 2 (sharing W storages) reads
        assert hash_step_leaves(c, [s2.leaf(i) for i in range(n2)]) == d2
        s2.close()
        # every reference dropped: package, store, result tuples, leaves; then the producer's
        # release (with ipc_collect) and empty_cache must give the memory back
        del p1, p2, s1, s2
        r.steps.clear()
        gc.collect()
        assert r.ask("release", 2) == "ok"
        before, after, w_bytes = r.ask("mem")
        assert after - before <= w_bytes, (
            f"producer memory_allocated {before} before the steps (after a warm-up step, which "
            f"makes the cuBLAS workspace and the hashing buffers), {after} after release: "
            f"{after - before} bytes kept, more than one weight set ({w_bytes}); the step's "
            f"transcript is still held")
    finally:
        r.stop()


@pytest.mark.parametrize("alloc", ALLOCS)
def test_cuda_a_producer_write_after_open_is_read_by_the_consumer(alloc):
    """No copy, proved from outside: the producer writes one element through ``.data`` after
    the consumer opened the step, and the consumer reads the new value at that leaf."""
    r = Remote(("cuda:0",), device="cuda:0", steps=1, alloc=alloc)
    try:
        c, _, _, _ = H.setup(("cuda:0",), "cuda:0")
        pkg, digests, _, n = r.steps[0]
        store = SharedStore.open(c, pkg)
        leaves = [store.leaf(i) for i in range(n)]
        assert hash_step_leaves(c, leaves) == digests  # the reads the test needs come first
        i = next(j for j, x in enumerate(leaves)
                 if isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel())
        del leaves
        old, new = r.ask("poke", 1, i)
        assert old != new
        x = store.leaf(i)
        torch.cuda.synchronize()
        got = x[(0,) * x.ndim].item()
        assert got == new, (f"leaf {i}: the consumer reads {got} after the producer wrote {new} "
                            f"(was {old}); a value still equal to the old one means a copy")
        free, total = torch.cuda.mem_get_info(0)  # informative only, not asserted
        print(f"consumer-side cuda:0 free {free} of {total} bytes")
        del x
        store.close()
        r.ask("release", 1)  # the write bypassed _version, so this release may not name it
    finally:
        r.stop()


@pytest.mark.parametrize("alloc", ALLOCS)
def test_cuda_release_after_close_returns_the_producers_memory_exactly(alloc):
    conn, child = CTX.Pipe()
    p = CTX.Process(target=H.producer_cuda_release, args=(child, alloc))
    p.start()
    child.close()
    try:
        assert conn.poll(180), "the producer sent nothing"
        pkg, copies, _ = conn.recv()
        store = SharedStore.open(SimpleNamespace(n_s=0), pkg)
        for i, y in enumerate(copies):
            if isinstance(y, torch.Tensor):
                x = store.leaf(i)
                assert x.dtype == y.dtype and torch.equal(x.cpu(), y), f"leaf {i}"
        del x
        store.close()
        assert pkg.cuda == []
        del pkg, store
        gc.collect()
        conn.send("release")
        assert conn.poll(60)
        ok, before, after = conn.recv()
        assert ok == "ok", ok
        assert after == before, (f"producer memory_allocated {before} before the step, {after} "
                                 f"after close, release, ipc_collect and empty_cache")
        conn.send("quit")
    finally:
        H.reap(p, 10)


@pytest.mark.parametrize("alloc", ALLOCS)
def test_cuda_unsharded_leaves_are_host_tensors_and_take_the_cpu_path(alloc):
    """``placement_for`` gives an unsharded computation HOST for every leaf, so a run on cuda:0
    without verifier devices crosses through the one CPU buffer, not CUDA handles."""
    r = Remote(None, device="cuda:0", steps=1, alloc=alloc)
    try:
        c, _, _, _ = H.setup(None, "cuda:0")
        pkg, digests, root, n = r.steps[0]
        assert pkg.cuda == [] and pkg.cpu is not None, (
            f"expected the CPU path, got {len(pkg.cuda)} CUDA storages, cpu={pkg.cpu is not None}")
        store = SharedStore.open(c, pkg)
        assert hash_step_leaves(c, [store.leaf(i) for i in range(n)]) == digests
        assert store.root == root
        store.close()
        assert r.ask("release", 1) == "ok"
    finally:
        r.stop()


def test_cuda_share_step_allocates_nothing():
    """In one process: sharing a CUDA step adds no device memory (no copy)."""
    leaves = H.synthetic_leaves("cuda:0")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(0)
    pkg, hold = H.share_step(SimpleNamespace(n_s=0), 1, leaves, H.synthetic_tree(len(leaves)))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(0) == before
    assert pkg.cpu is None and pkg.cuda
    hold.release()
