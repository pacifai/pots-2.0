"""FS30 on a GPU: arrival hashing in the capture, and device leaves at the commit. Skipped without
CUDA; the ``device-leaves`` session runs this file on an H100. The table building, the digests and
the error precedence run on the host build of the core in ``test_leaves_known.py``."""

import os

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")


def _step(dtype="fp32", head_blocks=1):
    from setup.data import schedule
    from tests.verification.decoder_helpers import make_records, tiny_config
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import snapshot_weights
    from verification.prover.step import prove_step

    compute = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, compute_dtype=compute,
                           device="cuda", head_blocks=head_blocks)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    torch.manual_seed(0)
    model = c.build_model()
    w0 = snapshot_weights(c, model)
    records = [D[i] for i in schedule(1, c.n_s, len(D), 1)]
    return c, prove_step(c, model, w0, records)


@pytest.fixture
def deterministic():
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)


@pytest.mark.parametrize("head_blocks", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_capture_hashes_every_product_but_dK_on_arrival(dtype, head_blocks, deterministic):
    """Every product leaf but the δK̃ transpose takes an arrival digest; each equals the C
    library's digest of the leaf; the root with them equals the plain root and check 2's. The
    products with no digest are the δK̃ ones only (the diagnostic's ``missing``). At head
    blocks 1 and 2 (the full-scale runs use 8)."""
    from tests.verification.computation.test_decoder import ListReader
    from verification.commitment.leaves import commit_leaves, leaf_hashes_of, transcript_root

    c, out = _step(dtype, head_blocks)
    leaves = out.leaves()
    want = {c.product_index(s.m) for s in c.products if c.product_class(s) != "dK"}
    assert set(out.arrival_digests) == want
    assert out.capture["arrival_hash"]["nonfinite"] == 0
    assert out.capture["arrival_hash"]["leaves"] == c.M  # the dK records are hashed too
    assert out.capture["arrival_hash"]["used"] == len(want)
    n_dk = sum(1 for s in c.products if c.product_class(s) == "dK")
    assert n_dk == c.M - len(want) > 0
    assert out.capture["arrival_hash"]["missing"] == {"OPERAND_GRAD": n_dk}
    assert out.capture["arrival_hash"]["missing_first"] == []
    plain = leaf_hashes_of(c, list(enumerate(leaves)))
    assert all(plain[i] == d for i, d in out.arrival_digests.items())
    root = commit_leaves(c, leaves, out.arrival_digests).root
    assert root == commit_leaves(c, leaves).root == transcript_root(c, ListReader(leaves))


def test_a_nan_inside_a_matmul_output_has_no_digest_and_the_cpu_paths_leaf():
    """A NaN made by a bmm in member 1 of 3: members 1 and 2 get no digest (the kernel stops at
    the first bad leaf), member 0 gets the C library's, and the first leaf the host check
    rejects is member 1, the leaf where the arrival hash stopped."""
    from verification.commitment.encoding import NonFiniteError, check_payload, tensor_leaf_header
    from verification.commitment.merkle import hash_tensor_leaf
    from verification.prover.capture import HostBuffers, MatmulCapture

    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 16, 8, generator=g).to(DEV)
    b = torch.randn(3, 8, 12, generator=g).to(DEV)
    a[1, 2, 3] = float("nan")
    cap = MatmulCapture(buffers=HostBuffers())
    with cap, cap.phase("forward"):
        torch.bmm(a, b)
    out = cap.records[0].out
    assert [cap.arrival_digest(out[i]) is None for i in range(3)] == [False, True, True]
    assert cap.arrival_digest(out[0]) == hash_tensor_leaf(3, out[0])
    first = None
    for i in range(3):
        try:
            check_payload(out[i])
        except NonFiniteError:
            first = i
            break
    assert first == 1
    assert cap.summary()["arrival_hash"]["nonfinite"] == 2
    assert tensor_leaf_header(3, out[0]) == tensor_leaf_header(3, out[1])


def test_device_leaves_at_commit_match_the_host_path(deterministic):
    from verification.commitment.leaves import HashStats, hash_step_leaves

    c, out = _step()
    leaves = out.leaves()
    want = hash_step_leaves(c, leaves)
    n_s = c.n_s
    dev = [x if i < n_s else x.to(DEV) for i, x in enumerate(leaves)]
    stats = HashStats()
    assert hash_step_leaves(c, dev, stats) == want
    assert stats.kernel == c.n_leaves - n_s


def test_a_nan_in_a_device_leaf_is_the_host_paths_leaf_and_error(deterministic):
    from verification.commitment.encoding import NonFiniteError
    from verification.commitment.leaves import hash_step_leaves

    c, out = _step()
    leaves = out.leaves()
    i = c.product_index(3)
    host = list(leaves)
    host[i] = host[i].clone()
    host[i].view(-1)[0] = float("nan")
    dev = [x if j < c.n_s else x.to(DEV) for j, x in enumerate(host)]
    with pytest.raises(NonFiniteError) as want:
        hash_step_leaves(c, host)
    with pytest.raises(NonFiniteError) as got:
        hash_step_leaves(c, dev)
    assert str(got.value) == str(want.value)
    # the leaf: a structural error at an earlier index still wins, on the device path too
    from verification.transcript.errors import LeafShapeError
    dev[c.product_index(1)] = dev[c.product_index(1)][:-1]
    with pytest.raises(LeafShapeError):
        hash_step_leaves(c, dev)


def test_a_device_leaf_mutated_before_its_table_is_read_raises(deterministic):
    from verification.commitment import leaves as L
    from verification.commitment.leaves import hash_step_leaves
    from verification.transcript.errors import StoreMutationError

    c, out = _step()
    dev = [x if j < c.n_s else x.to(DEV) for j, x in enumerate(out.leaves())]
    i = c.product_index(1)
    real = L._enqueue_table

    def mutate_after(idx, tensors, tags):
        tb = real(idx, tensors, tags)
        tensors[idx.index(i)].add_(0)
        return tb

    L._enqueue_table = mutate_after
    try:
        with pytest.raises(StoreMutationError, match=f"leaf {i} was mutated in place"):
            hash_step_leaves(c, dev)
    finally:
        L._enqueue_table = real


def test_enqueue_adds_no_synchronization():
    """Under ``set_sync_debug_mode("error")`` a capture hashing on arrival, and the enqueue
    alone, raise nothing: no ``.item()``, ``.cpu()`` or pageable copy sits in the path."""
    from verification.commitment.encoding import tensor_leaf_header
    from verification.commitment.gpu_blake3 import GpuBlake3
    from verification.prover.capture import HostBuffers, MatmulCapture

    g = GpuBlake3()
    xs = [torch.randn(64, 64, device=DEV) for _ in range(3)]
    hs = [tensor_leaf_header(3, x) for x in xs]
    stream = torch.cuda.Stream()
    g.enqueue(hs, xs, stream=stream).result()  # builds the kernels and fills the pools
    a, b = torch.randn(32, 16, device=DEV), torch.randn(16, 8, device=DEV)
    pool = HostBuffers()
    cap = MatmulCapture(buffers=pool)
    with cap, cap.phase("forward"):
        torch.mm(a, b)  # warms the host pool
    cap = MatmulCapture(buffers=pool)
    torch.cuda.synchronize()
    with cap, cap.phase("forward"):
        torch.cuda.set_sync_debug_mode("error")
        try:
            torch.mm(a, b)
            pend = g.enqueue(hs, xs, stream=stream)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        pend.result()
    assert cap.summary()["arrival_hash"]["records"] == 1
