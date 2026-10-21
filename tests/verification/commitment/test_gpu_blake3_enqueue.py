"""``GpuBlake3.enqueue`` (FS30) on the host build of the core: the table it builds in one buffer,
its digests against the merkle module's, ``first_bad``, and what the pending hash holds. The
non-blocking copies and the event need CUDA and are tested in ``test_device_leaves_cuda.py``."""

import numpy as np
import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from verification.commitment import gpu_blake3 as gb
from verification.commitment.encoding import TAG_PRODUCT, tensor_leaf_header
from verification.commitment.merkle import hash_tensor_leaf


@pytest.fixture
def g(hostlib):  # noqa: F811
    return gb.GpuBlake3(HostKernels(hostlib))


def leaves(shapes, dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=gen).to(dtype) for s in shapes]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_enqueue_matches_the_merkle_digests(g, dtype):
    xs = leaves([(1000,), (33, 65), (3, 3), (40, 40, 7)], dtype)
    hs = [tensor_leaf_header(TAG_PRODUCT, x) for x in xs]
    pend = g.enqueue(hs, xs, stream=None)
    assert pend.done()
    digests, bad = pend.result()
    assert bad == len(xs)
    assert digests == [hash_tensor_leaf(TAG_PRODUCT, x) for x in xs]
    assert pend.result() == (digests, bad)


def test_the_digests_stop_at_the_first_bad_leaf(g):
    xs = leaves([(3000,)] * 5)
    xs[3][100] = float("nan")
    xs[4][5] = float("inf")
    hs = [tensor_leaf_header(TAG_PRODUCT, x) for x in xs]
    digests, bad = g.enqueue(hs, xs, stream=None).result()
    assert bad == 3
    assert digests == [hash_tensor_leaf(TAG_PRODUCT, x) for x in xs[:3]]


def test_an_int_leaf_is_not_tested_for_finiteness(g):
    x = torch.arange(100, dtype=torch.int32)
    h = tensor_leaf_header(TAG_PRODUCT, x)
    digests, bad = g.enqueue([h], [x], stream=None).result()
    assert (bad, digests) == (1, [hash_tensor_leaf(TAG_PRODUCT, x)])


def test_the_pending_hash_holds_its_tensors_until_read(g):
    xs = leaves([(64,)])
    pend = g.enqueue([tensor_leaf_header(TAG_PRODUCT, xs[0])], xs, stream=None)
    assert any(k is xs[0] for k in pend.keep)
    pend.result()
    assert pend.keep == []


def test_enqueue_table_with_shared_headers(g):
    """One header for many equal-shaped leaves, as capture's batch members use it."""
    x = torch.randn(6, 50)
    h = tensor_leaf_header(TAG_PRODUCT, x[0])
    base = x.data_ptr()
    row = x[0].nbytes
    n = 6
    pend = g.enqueue_table(
        h, np.zeros(1, np.uint64), np.array([len(h)], np.uint32), np.zeros(n, np.int64),
        np.array([base + i * row for i in range(n)], np.uint64), np.full(n, row, np.uint64),
        np.full(n, gb.FINITE_KIND[x.dtype], np.uint8), device="cpu", stream=None, keep=[x])
    digests, bad = pend.result()
    assert bad == n
    assert digests == [hash_tensor_leaf(TAG_PRODUCT, x[i]) for i in range(n)]


def test_enqueue_rejects_what_hash_tensor_leaves_rejects(g):
    x = torch.randn(4, 4)
    h = tensor_leaf_header(TAG_PRODUCT, x)
    with pytest.raises(ValueError, match="contiguous"):
        g.enqueue([h], [x.t()], stream=None)
    with pytest.raises(ValueError, match="one length"):
        g.enqueue([h, h], [x], stream=None)
    with pytest.raises(ValueError, match="nothing"):
        g.enqueue([], [], stream=None)
