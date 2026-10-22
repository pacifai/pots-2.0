"""Capture routing under a sharded placement (FS32d, F22): the hint, the peer copy and the
prover's step. The Mac tests run sharded on ``cpu,cpu,cpu``, where nothing is copied but the hint
is still computed and counted; the two-GPU tests skip unless CUDA has two devices."""

import contextlib
import os

import pytest
import torch

from setup.config import load_config
from verification.commitment.leaves import leaf_hash
from verification.computation.instances import DecoderComputation
from verification.computation.matmul_ops import param_storage_map
from verification.prover.capture import HostBuffers, MatmulCapture, MutatedCaptureError
from verification.prover.step import commit, plain_step, prove_step

from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config, verifier_devices)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

THREE = ("cpu", "cpu", "cpu")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
SPIN = 20_000_000
two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")


def tiny(switch, *, B=1, dtype="fp32", devices=None, layers=2, device="cpu"):
    family, tied = switch
    config = tiny_config(family=family, tied=tied)
    config.num_hidden_layers = layers
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[0]] * layers
    return DecoderComputation(config, n_s=2, n=12, eta=1e-3, head_blocks=B,
                              compute_dtype=DTYPES[dtype], verifier_devices=devices,
                              device=device)


def start(c):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    randomize_biases(w0)
    return w0


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


def assert_same_step(a, b):
    assert a.loss == b.loss and len(a.leaves()) == len(b.leaves())
    for x, y in zip(a.leaves()[len(a.records):], b.leaves()[len(b.records):]):
        assert same(x, y)


def run(c, w0, records, **kw):
    return prove_step(c, c.build_model(), w0, records, **kw)


# ---- the hint on the Mac -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_sharded_step_equals_unsharded_and_hint_never_misses(switch, B, dtype):
    off, on = tiny(switch, B=B, dtype=dtype), tiny(switch, B=B, dtype=dtype, devices=THREE)
    w0, records = start(off), make_records((12, 7))
    a, b = run(off, w0, records), run(on, w0, records)
    assert a.hint_misses == 0 and "hint_misses" not in a.capture
    assert b.hint_misses == 0 and b.capture["hint_misses"] == 0
    assert_same_step(a, b)
    assert commit(off, a).root == commit(on, b).root
    w_next, loss = plain_step(on, on.build_model(), w0, records)
    assert loss == b.loss and all(same(w_next[n], b.w_next[n]) for n in on.weight_names)


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_hint_matches_every_product_owner(switch, dtype):
    """With distinct (fake) devices the hint is a real claim: every captured leaf that is a
    record's output was hinted to its product's owner, over a model with units on all three."""
    fake = ("cuda:0", "cuda:1", "cuda:2")
    c = tiny(switch, B=2, dtype=dtype, devices=fake, layers=4)
    assert len({c.placement.product_device(m) for m in range(1, c.M + 1)}) == 3
    w0, records = start(c), make_records((12, 7))
    model = c.build_model()
    model.zero_grad(set_to_none=True)
    cap = MatmulCapture(param_storage_map(model), placement=c.placement)
    @contextlib.contextmanager
    def stage(p):
        with cap.phase(p):
            yield

    with cap:
        c.forward_backward(model, records, stage)
    labeled = c.label(cap, model)
    assert cap.count_hint_misses(labeled, c.placement.product_device) == 0
    assert cap.summary()["hint_misses"] == 0
    hinted = {r.hint for r in cap.records}
    assert hinted == {torch.device(d) for d in fake}


def test_a_wrong_hint_costs_a_count_never_a_bit(monkeypatch):
    switch = ("qwen2", False)
    off, on = tiny(switch, dtype="bf16"), tiny(switch, dtype="bf16", devices=THREE)
    w0, records = start(off), make_records((12, 7))
    a = run(off, w0, records)
    monkeypatch.setattr(MatmulCapture, "_hint", lambda self, rec: torch.device("cuda:1"))
    b = run(on, w0, records)
    assert b.hint_misses > 0 and b.capture["hint_misses"] == b.hint_misses
    assert_same_step(a, b)


def test_a_perturbed_product_lands_at_its_declared_check():
    off, on = tiny(("llama", True)), tiny(("llama", True), devices=THREE)
    w0, records = start(off), make_records((12, 7))
    m = on.m_of("L1.G_q")
    bump = lambda t: t + 1.0
    a, b = run(off, w0, records), run(on, w0, records, perturb={m: bump})
    assert b.products[m - 1].device == on.placement.product_device(m)
    assert same(b.products[m - 1], bump(a.products[m - 1]))
    assert all(same(x, y) for i, (x, y) in enumerate(zip(a.products, b.products)) if i != m - 1)
    b.assert_unmodified()


def test_with_w_next_and_w_t_sharing_stay_on_the_weight_device():
    c = tiny(("llama", True), devices=THREE)
    w0, records = start(c), make_records((12, 7))
    out = run(c, w0, records)
    assert all(out.w_t[n] is w0[n] for n in c.weight_names)  # shared: already on its device
    spliced = out.with_w_next({n: t + 1 for n, t in out.w_next.items()})
    for n in c.weight_names:
        assert spliced.w_next[n].device == c.placement.weight_device(n)
        assert same(spliced.w_next[n], out.w_next[n] + 1)
    spliced.assert_unmodified()


def test_a_perturbation_built_on_another_device_stays_on_the_owner():
    """A fault function may return a tensor from ``torch.zeros(shape)``, on the host; the leaf
    still sits on ``product_device(m)``. The placement's devices are ``meta`` here, which the
    Mac has and the host is not."""
    c = tiny(("llama", True), devices=("meta", "meta", "meta"))
    w0, records = start(c), make_records((12, 7))
    m = c.m_of("L1.G_q")
    out = run(c, w0, records,
              perturb={m: lambda t: torch.zeros(t.shape, dtype=t.dtype)})
    assert out.products[m - 1].device.type == "meta"
    assert all(p.device.type == "meta" for p in out.products)


# ---- two GPUs ----------------------------------------------------------------------------

GPUS = verifier_devices()


@two_gpus
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_two_gpus_every_leaf_on_its_owner_and_digests_equal(B, dtype, monkeypatch):
    switch = ("qwen2", False)
    off = tiny(switch, B=B, dtype=dtype, device="cuda:0")
    on = tiny(switch, B=B, dtype=dtype, devices=GPUS, device="cuda:0", layers=2)
    w0, records = start(off), make_records((12, 7))
    ref = run(off, {n: t.cpu() for n, t in w0.items()}, records)
    takes = []
    real_take = HostBuffers.take
    monkeypatch.setattr(HostBuffers, "take",
                        lambda self, *a, **k: takes.append(a) or real_take(self, *a, **k))
    out = run(on, {n: t.to(on.placement.weight_device(n)) for n, t in w0.items()}, records)
    assert out.hint_misses == 0 and out.capture["hint_misses"] == 0
    assert takes == []  # no host buffer was taken during the sharded capture
    for m, p in enumerate(out.products, 1):
        assert p.device == on.placement.product_device(m), m
    for n in on.weight_names:
        assert out.w_t[n].device == out.w_next[n].device == on.placement.weight_device(n)
    # Same bytes as the host-buffer run, leaf by leaf (leaf_hash of a device leaf is FS32b's).
    pairs = list(zip(ref.leaves()[len(ref.records):], out.leaves()[len(out.records):]))
    for i, (x, y) in enumerate(pairs):
        assert leaf_hash(off, len(ref.records) + i, x) == leaf_hash(
            off, len(ref.records) + i, y.cpu()), i
    out.assert_unmodified()


@two_gpus
def test_two_gpus_a_bmm_stays_one_storage_on_the_owner():
    a = torch.randn(3, 16, 8, device="cuda:0")
    b = torch.randn(3, 8, 12, device="cuda:0")
    cap = MatmulCapture(placement=_two_gpu_placement())
    cap._hint = lambda rec: torch.device("cuda:1")
    with cap, cap.phase("forward"):
        p = torch.bmm(a, b)
    (rec,) = cap.records
    assert cap.is_output_of(p, rec)  # the op's own output, now only its source
    assert rec.out.device == torch.device("cuda:1") and rec.hint == rec.out.device
    assert torch.equal(rec.out.cpu(), torch.bmm(a, b).cpu())
    ptrs = {p.untyped_storage().data_ptr() for _, _, p in rec.members()}
    assert ptrs == {rec.out.untyped_storage().data_ptr()}
    assert rec.out_storage is not None and rec.out_info.storage_ptr != ptrs.pop()


@two_gpus
def test_two_gpus_a_write_before_the_copy_is_done_raises():
    a, b = torch.randn(256, 512, device="cuda:0"), torch.randn(512, 384, device="cuda:0")
    # a capture of the same op on the same pool runs first: the first arrival hash on cuda:1
    # allocates, and CUDA may hold the host there until the GPU is idle, so the copy would be
    # done before the write
    pool = HostBuffers()
    warm = MatmulCapture(buffers=pool, placement=_two_gpu_placement())
    warm._hint = lambda rec: torch.device("cuda:1")
    with warm, warm.phase("forward"):
        torch.mm(a, b)
    del warm
    for d in ("cuda:0", "cuda:1"):
        torch.cuda.synchronize(d)
    cap = MatmulCapture(buffers=pool, placement=_two_gpu_placement())
    cap._hint = lambda rec: torch.device("cuda:1")
    with pytest.raises(MutatedCaptureError, match="before its copy"):
        with cap, cap.phase("forward"):
            with torch.cuda.stream(cap.buffers.stream(torch.device("cuda:0"))):
                torch.cuda._sleep(SPIN)
            p = torch.mm(a, b)
            p.add_(1)
    torch.cuda.synchronize()


@two_gpus
def test_two_gpus_the_destination_outlives_a_delayed_copy():
    """The peer copy is held back on its side stream, every other reference to the destination
    is dropped and fresh blocks of the same size are allocated on the destination device to
    tempt the allocator; the leaf still holds the source's bytes."""
    a, b = torch.randn(256, 512, device="cuda:0"), torch.randn(512, 384, device="cuda:0")
    want = torch.mm(a, b).cpu()
    cap = MatmulCapture(placement=_two_gpu_placement())
    cap._hint = lambda rec: torch.device("cuda:1")
    with cap, cap.phase("forward"):
        with torch.cuda.stream(cap.buffers.stream(torch.device("cuda:0"))):
            torch.cuda._sleep(SPIN)
        torch.mm(a, b)
        for _ in range(8):
            torch.full((256, 384), float("nan"), device="cuda:1")
    (rec,) = cap.records
    leaf = rec.out
    rec.out = None
    cap.records.clear()
    cap._by_index.clear()
    for _ in range(8):
        torch.full((256, 384), float("nan"), device="cuda:1")
    torch.cuda.synchronize()
    assert leaf.device == torch.device("cuda:1") and torch.equal(leaf.cpu(), want)


@two_gpus
def test_two_gpus_an_output_on_its_owner_is_kept_not_copied():
    a, b = torch.randn(16, 8, device="cuda:0"), torch.randn(8, 4, device="cuda:0")
    cap = MatmulCapture(placement=_two_gpu_placement())
    cap._hint = lambda rec: torch.device("cuda:0")
    with cap, cap.phase("forward"):
        torch.mm(a, b)
    (rec,) = cap.records
    assert not cap._pending and rec.out.untyped_storage().data_ptr() == rec.out_info.storage_ptr


@two_gpus
def test_two_gpus_release_device_cache_empties_every_device():
    from verification.runs.loop import release_device_cache
    block = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda:1")
    del block
    assert torch.cuda.memory_reserved(1) > 0
    before = torch.cuda.memory_reserved(1)
    release_device_cache("cuda:0", devices=GPUS)
    assert torch.cuda.memory_reserved(1) < before


def _two_gpu_placement():
    c = tiny(("llama", True), devices=GPUS, device="cuda:0")
    return c.placement


@two_gpus
@pytest.mark.slow
def test_two_gpus_smollm2_hint_never_misses_and_digests_equal(monkeypatch):
    from setup.config import setup_determinism
    from setup.data import load_dataset_records, schedule

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("VERIF_DEVICE", "cuda:0")
    monkeypatch.delenv("VERIF_VERIFIER_DEVICES", raising=False)
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}")
    setup_determinism(cfg)
    off = DecoderComputation.from_config(cfg)
    monkeypatch.setenv("VERIF_VERIFIER_DEVICES", ",".join(GPUS))
    on = DecoderComputation.from_config(load_config())
    D = load_dataset_records(cfg.data_dir / "D.bin", off.n)
    records = [D[i] for i in schedule(1, off.n_s, len(D), 0)]
    model = off.build_model()
    w0 = {n: model.get_parameter(n).detach().cpu().clone() for n in off.weight_names}
    ref = prove_step(off, model, w0, records)
    out = prove_step(on, on.build_model(),
                     {n: t.to(on.placement.weight_device(n)) for n, t in w0.items()}, records)
    assert out.hint_misses == 0
    pairs = list(zip(ref.leaves()[len(ref.records):], out.leaves()[len(out.records):]))
    for i, (x, y) in enumerate(pairs):
        assert same(x, y), i
