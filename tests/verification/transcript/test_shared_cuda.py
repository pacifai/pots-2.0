"""The CUDA transport of the cross-process store (FS33a). Skipped without CUDA; desk-checked,
it first runs on the two-GPU pod (FS33d).

The producer is a spawned process holding the transcript on ``cuda:0``. Nothing is copied: the
consumer's leaves sit inside storages it opened by CUDA handle, and the producer's CUDA memory
does not grow by the transcript when it shares.
"""

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


def _cases():
    yield pytest.param(None, "cuda:0", id="one_gpu")
    yield pytest.param(("cuda:1",), "cuda:0", id="two_gpus",
                       marks=pytest.mark.skipif(not TWO, reason="needs a second GPU"))


@pytest.mark.parametrize(("devices", "device"), list(_cases()))
def test_cuda_step_hashes_and_reads_in_place(devices, device):
    r = Remote(devices, device=device, steps=2)
    try:
        c, _, _, _ = H.setup(devices, device)
        (p1, d1, root1, n1), (p2, d2, _, n2) = r.steps
        assert p1.cuda and p1.cpu is None and p1.n_storages == len(p1.cuda)
        s1, s2 = SharedStore.open(c, p1), SharedStore.open(c, p2)
        leaves = [s1.leaf(i) for i in range(n1)]
        assert hash_step_leaves(c, leaves) == d1
        assert MerkleTree(d1).root == root1 == s1.root
        # no copy: every tensor leaf lies inside one of the storages opened by handle
        spans = [(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for t in p1.cuda]
        for x in leaves:
            if isinstance(x, torch.Tensor) and x.numel():
                assert any(a <= x.data_ptr() < a + n for a, n in spans)
        s1.close()
        del p1, leaves
        assert r.ask("release", 1) == "ok"  # step 1 gone, step 2 (sharing W storages) reads
        assert hash_step_leaves(c, [s2.leaf(i) for i in range(n2)]) == d2
        s2.close()
        assert r.ask("release", 2) == "ok"
    finally:
        r.stop()


def test_cuda_share_step_allocates_nothing():
    """In one process: sharing a CUDA step adds no device memory (no copy)."""
    from types import SimpleNamespace

    leaves = H.synthetic_leaves("cuda:0")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(0)
    pkg, hold = H.share_step(SimpleNamespace(n_s=0), 1, leaves, H.synthetic_tree(len(leaves)))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(0) == before
    assert pkg.cpu is None and pkg.cuda
    hold.release()
