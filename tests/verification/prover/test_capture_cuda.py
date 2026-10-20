"""Products on a GPU (FS9): each captured output is copied into a pinned host buffer on a side
stream and its GPU copy freed. Skipped without CUDA; the CPU side is in ``test_capture.py``."""

import os

import pytest
import torch

from verification.prover.capture import HostBuffers, MatmulCapture, MutatedCaptureError

# Deterministic cuBLAS reads this once, at CUDA init (S4c), so it is set before any test runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
SPIN = 20_000_000  # torch.cuda._sleep cycles: about 10 ms, far longer than a launch


def operands(m=256, q=512, n=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, q, generator=g).to(DEV), torch.randn(q, n, generator=g).to(DEV))


def side(pool):
    return pool.stream(DEV)


def test_outputs_land_in_pinned_host_buffers_bit_for_bit():
    """The copy waits for the op that wrote the output, and the records are read only after
    every copy is done: the main stream is delayed before each op and the side stream before
    each copy, and the program frees each output and fills new memory at once, tempting the
    allocator to hand the output's memory out under the copy."""
    pool = HostBuffers()
    a, b = operands()
    refs = [torch.mm(a + i, b).cpu() for i in range(6)]
    cap = MatmulCapture(buffers=pool)
    with cap, cap.phase("forward"):
        for i in range(6):
            x = a + i
            torch.cuda._sleep(SPIN)
            with torch.cuda.stream(side(pool)):
                torch.cuda._sleep(SPIN)
            p = torch.mm(x, b)
            del p
            torch.full((256, 384), float("nan"), device=DEV)
    assert not cap._pending
    assert len(cap.records) == 6
    for rec, ref in zip(cap.records, refs):
        assert rec.out.device.type == "cpu" and rec.out.is_pinned()
        assert rec.out.dtype == ref.dtype and torch.equal(rec.out, ref)
        assert rec.out_info.storage_ptr != rec.out.untyped_storage().data_ptr()
    cap.assert_unmodified()
    assert pool.n_buffers == 6


def test_batched_members_and_bf16_are_copied_whole():
    pool = HostBuffers()
    g = torch.Generator().manual_seed(1)
    a = torch.randn(3, 16, 8, generator=g).to(DEV, torch.bfloat16)
    b = torch.randn(3, 8, 12, generator=g).to(DEV, torch.bfloat16)
    cap = MatmulCapture(buffers=pool)
    with cap, cap.phase("forward"):
        torch.bmm(a, b)
    (rec,) = cap.records
    assert rec.out.dtype == torch.bfloat16 and rec.out.shape == (3, 16, 12)
    assert torch.equal(rec.out, torch.bmm(a, b).cpu())
    assert [p.shape for _, _, p in rec.members()] == [(16, 12)] * 3


def test_a_write_before_the_copy_is_done_raises():
    """An in-place write to an output whose copy may still run races the copy; leaving the
    block reports it."""
    pool = HostBuffers()
    a, b = operands()
    cap = MatmulCapture(buffers=pool)
    with pytest.raises(MutatedCaptureError, match="before its copy to the host was done"):
        with cap, cap.phase("forward"):
            with torch.cuda.stream(side(pool)):
                torch.cuda._sleep(SPIN)
            p = torch.mm(a, b)
            p.add_(1)
    torch.cuda.synchronize()


def test_w_grad_is_recognized_as_the_output_of_its_G():
    """AccumulateGrad adopts a weight gradient's GPU output as ``W.grad``; the record's output
    is its host copy, so the link goes by the storage the op returned."""
    pool = HostBuffers()
    lin = torch.nn.Linear(32, 16, bias=False).to(DEV)
    x = torch.randn(8, 32, device=DEV)
    cap = MatmulCapture(buffers=pool)
    with cap:
        with cap.phase("forward"):
            y = lin(x)
        with cap.phase("backward"):
            y.square().sum().backward()
    (g,) = [r for r in cap.records if r.phase == "backward" and r.out.shape == (16, 32)]
    assert g.out.device.type == "cpu" and torch.equal(g.out, lin.weight.grad.cpu())
    assert cap.is_output_of(lin.weight.grad, g)
    assert not cap.is_output_of(lin.weight.grad.clone(), g)


def test_a_reused_address_is_not_a_producer():
    """Once an output's copy is done its GPU memory goes back to the allocator, which hands the
    same address to the next tensor of that size. That tensor is no record's output."""
    pool = HostBuffers()
    a, b = operands()
    cap = MatmulCapture(buffers=pool)
    with cap, cap.phase("forward"):
        p = torch.mm(a, b)
        ptr = p.data_ptr()
        del p
        cap.wait_copies()
        fresh = torch.zeros(256, 384, device=DEV)
        assert fresh.data_ptr() == ptr  # the caching allocator reused the block
        torch.mm(fresh, b.t())
    first, second = cap.records
    assert second.a_info.storage_ptr == first.out_info.storage_ptr
    assert second.a_info.producer_index is None


def test_buffers_are_reused_by_the_next_capture():
    pool = HostBuffers()
    a, b = operands()

    def step():
        cap = MatmulCapture(buffers=pool)
        with cap, cap.phase("forward"):
            for i in range(4):
                torch.mm(a, b)
        return cap

    def ptrs(cap):
        return [r.out.untyped_storage().data_ptr() for r in cap.records]

    first = step()
    p1 = ptrs(first)
    second = step()  # the first's records are alive: a second set
    assert pool.n_buffers == 8 and not set(p1) & set(ptrs(second))
    del first
    assert ptrs(step()) == p1 and pool.n_buffers == 8


def test_gpu_memory_does_not_hold_the_transcript():
    """With each copy done before the next product, the GPU holds at most two outputs at a
    time, however many products the capture records."""
    pool = HostBuffers()
    a, b = operands(512, 512, 512)
    out_bytes = 512 * 512 * 4
    cap = MatmulCapture(buffers=pool)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    seen = []
    with cap, cap.phase("forward"):
        for _ in range(40):
            torch.mm(a, b)
            torch.cuda.synchronize()
            seen.append(torch.cuda.memory_allocated() - base)
    assert len(cap.records) == 40
    assert max(seen) <= 2 * out_bytes
    assert torch.cuda.memory_allocated() == base
    assert pool.nbytes == 40 * out_bytes


@pytest.mark.parametrize("head_blocks", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_tiny_decoder_step_on_cuda(dtype, head_blocks):
    """The decoder's honest step on the GPU: every product leaf is a pinned host buffer, the
    replay rebuilds every operand and glue gradient bit for bit, the verifier accepts, and the
    committed ``W_{t+1}`` equals the uncaptured step's. At ``head_blocks=2`` the output layer
    runs in two row blocks, as at full scale's ``VERIF_HEAD_BLOCKS=8``: one autocast region
    and one cast of ``W`` for both blocks, and per-block ``Λ_b``, ``δF_b`` and ``G_b``."""
    from setup.data import schedule
    from tests.verification.computation.test_decoder import ListReader, captured_operands
    from tests.verification.decoder_helpers import make_records, tiny_config
    from verification.commitment.leaves import dataset_tree
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import load_weights, snapshot_weights
    from verification.prover.step import plain_step, prove_step
    from verification.transcript.store import InMemoryStore
    from verification.verifier.bands import Bands
    from verification.verifier.driver import Verifier

    seed = 1
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        compute = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, compute_dtype=compute,
                               device="cuda", head_blocks=head_blocks)
        D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
        tree = dataset_tree(c, D)
        torch.manual_seed(0)
        model = c.build_model()
        assert next(model.parameters()).is_cuda
        w0 = snapshot_weights(c, model)
        assert all(w.device.type == "cpu" for w in w0.values())
        idx = schedule(1, c.n_s, len(D), seed)
        records = [D[i] for i in idx]

        out = prove_step(c, model, w0, records)
        assert all(p.device.type == "cpu" for p in out.products)
        # Each leaf is its pinned host buffer, except δK̃: labeling commits the contiguous
        # transpose of the captured output, a new host tensor.
        unpinned = {c.product_class(s) for s, p in zip(c.products, out.products)
                    if not p.is_pinned()}
        assert unpinned == {"dK"}
        w_plain, loss = plain_step(c, model, w0, records)
        assert loss == out.loss
        assert all(torch.equal(out.w_next[n], w_plain[n]) for n in c.weight_names)

        load_weights(c, model, w0)
        products, expected, grads = captured_operands(c, model, records)
        assert all(torch.equal(p, q) for p, q in zip(products, out.products))
        replay = c.replay(ListReader([*records, *w0.values(), *products, *w0.values()]))
        for spec in c.products:
            a, b = replay.operands(spec.m)
            want_a, want_b = expected[spec.m]
            assert torch.equal(a, want_a) and torch.equal(b, want_b), spec.name
        got = replay.glue_gradients()
        for n, g in got.items():
            assert torch.equal(g.to(grads[n].device), grads[n]), n

        v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1,
                     bands=Bands.provisional(), allow_provisional=True, w0=w0, seed=seed)
        assert v.start_run(D) is None
        store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
        assert v.verify_step(1, store) is None
    finally:
        torch.use_deterministic_algorithms(det)
