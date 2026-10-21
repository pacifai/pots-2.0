"""FS34 on real GPUs: the kernel hashes each leaf on its owner GPU (skipped without CUDA; the
pair session runs this file on two GPUs). The routing and the check run on the host build of
the kernel in ``tests/verification/commitment/test_fs34_routing.py``."""

import os

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N = torch.cuda.device_count()
needs_two = pytest.mark.skipif(N < 2, reason="needs two CUDA devices")


@pytest.fixture(autouse=True)
def fresh_run():
    from verification.commitment import kernel_check

    kernel_check.new_run()
    yield
    kernel_check.new_run()


@pytest.fixture
def deterministic():
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)


def _sharded_step(devices):
    from setup.data import schedule
    from tests.verification.decoder_helpers import make_records, tiny_config
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import snapshot_weights
    from verification.prover.step import prove_step

    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, device="cuda:0",
                           verifier_devices=devices)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    torch.manual_seed(0)
    model = c.build_model()
    w0 = snapshot_weights(c, model)
    records = [D[i] for i in schedule(1, c.n_s, len(D), 1)]
    return c, prove_step(c, model, w0, records)


@needs_two
def test_two_gpus_the_kernel_digests_equal_the_c_librarys_and_the_root_is_the_same(
        deterministic):
    """A sharded step's leaves sit on their owner GPUs. The kernel's digests (the commit's own
    routing, per-shape check on) equal the C library's (``kernel=False``); so do the root and
    the arrival digests; and the arrival digests hold ``M`` less the δK̃ leaves."""
    from verification.commitment.leaves import (
        HashStats, commit_leaves, leaf_hashes_of, leaf_hashes_until_error)

    c, out = _sharded_step(("cuda:0", "cuda:1"))
    leaves = out.leaves()
    assert {x.device.index for x in leaves if x.is_cuda} == {0, 1}  # on their owners
    items = list(enumerate(leaves))
    want = leaf_hashes_of(c, items, kernel=False)
    st = HashStats()
    assert leaf_hashes_of(c, items, st) == want
    assert st.kernel > 0 and st.crosschecks > 0
    digests, err = leaf_hashes_until_error(c, items)
    assert err is None and digests == want
    n_dk = sum(1 for s in c.products if c.product_class(s) == "dK")
    assert len(out.arrival_digests) == c.M - n_dk
    assert all(want[i] == d for i, d in out.arrival_digests.items())
    cap = out.capture
    assert cap["arrival_hash"]["nonfinite"] == 0
    assert cap["hint_misses"] is not None and cap["hint_misses"] >= 0
    root = commit_leaves(c, leaves, out.arrival_digests).root
    assert root == commit_leaves(c, leaves, kernel=False).root


@needs_two
def test_two_gpus_routed_and_kept_products_both_get_a_digest(deterministic):
    """Products copied to their hinted owner and products already there both take an arrival
    digest: with both devices owning layers, some outputs move and some stay."""
    c, out = _sharded_step(("cuda:0", "cuda:1"))
    hint = out.capture["hint_misses"]
    n_dk = sum(1 for s in c.products if c.product_class(s) == "dK")
    assert len(out.arrival_digests) == c.M - n_dk
    owners = {out.leaves()[i].device for i in out.arrival_digests}
    assert len(owners) == 2 and hint is not None


@needs_two
def test_a_planted_wrong_kernel_digest_stops_run_loop(monkeypatch):
    """A kernel that answers a wrong digest on the card stops the run with no verdict."""
    from verification.commitment import gpu_blake3 as gb
    from verification.commitment.kernel_check import KernelMismatchError
    from tests.verification.runs import test_sharded_run as sr

    real = gb.PendingHash.result

    def wrong(self, *a, **k):
        d, bad = real(self, *a, **k)
        return [bytes(32)] * len(d), bad

    monkeypatch.setattr(gb.PendingHash, "result", wrong)
    with pytest.raises(KernelMismatchError):
        sr.run(("cuda:0", "cuda:1"), device="cuda:0")


def test_one_gpu_unsharded_run_is_unchanged(deterministic):
    """The unsharded path (host leaves, no kernel) gives the same root and verdict."""
    from tests.verification.runs import test_sharded_run as sr

    c, res, _, hand = sr.run(None, device="cuda:0")
    assert res.verdict.accepted and len(hand.roots) == sr.T
