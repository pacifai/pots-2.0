"""FS34 on real GPUs: the kernel hashes each leaf on its owner GPU (the CUDA tests skip without
CUDA; the pair session runs this file on two GPUs). The routing and the check run on the host
build of the kernel in ``tests/verification/commitment/test_fs34_routing.py``.

Each two-GPU test's body is a helper that takes the devices, and a Mac twin runs it on the CPU
sharded placement under the ``on_kernel`` seams, so an API break shows before a pod does. What
the twins cannot assert, because it needs the capture's GPU path: the leaves' device indices,
``arrival_digests`` (and so the arrival-path check), the ``arrival_hash`` capture stats and
``hint_misses``. The helpers take ``cuda`` for those assertions."""

import os

import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import hostlib  # noqa: F401
from tests.verification.commitment.test_leaves_known import on_kernel  # noqa: F401

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

N = torch.cuda.device_count()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_two = pytest.mark.skipif(N < 2, reason="needs two CUDA devices")
CARD = ("cuda:0", "cuda:1")
HOST = ("cpu", "cpu", "cpu")


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


def _sharded_step(devices, device):
    from setup.data import schedule
    from tests.verification.decoder_helpers import make_records, tiny_config
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import snapshot_weights
    from verification.prover.step import prove_step

    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, device=device,
                           verifier_devices=devices)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    torch.manual_seed(0)
    model = c.build_model()
    w0 = snapshot_weights(c, model)
    records = [D[i] for i in schedule(1, c.n_s, len(D), 1)]
    return c, prove_step(c, model, w0, records)


def _digests_body(devices, device, *, cuda):
    """The kernel's digests equal the C library's (``kernel=False``), and so do the root and
    (on CUDA) the arrival digests, which hold ``M`` less the δK̃ leaves."""
    from verification.commitment.leaves import (
        HashStats, commit_leaves, leaf_hashes_of, leaf_hashes_until_error)

    c, out = _sharded_step(devices, device)
    leaves = out.leaves()
    tensors = [x for x in leaves if isinstance(x, torch.Tensor)]  # the first n_s are Records
    if cuda:
        owners = {torch.device(d).index for d in devices}
        assert {x.device.index for x in tensors if x.is_cuda} == owners  # on their owners
    items = list(enumerate(leaves))
    want = leaf_hashes_of(c, items, kernel=False)
    st = HashStats()
    assert leaf_hashes_of(c, items, st) == want
    assert st.kernel > 0 and st.crosschecks > 0
    digests, err = leaf_hashes_until_error(c, items)
    assert err is None and digests == want
    if cuda:
        n_dk = sum(1 for s in c.products if c.product_class(s) == "dK")
        assert len(out.arrival_digests) == c.M - n_dk
        assert all(want[i] == d for i, d in out.arrival_digests.items())
        cap = out.capture
        assert cap["arrival_hash"]["nonfinite"] == 0
        assert cap["hint_misses"] is not None and cap["hint_misses"] >= 0
        known = out.arrival_digests
    else:  # no arrival digests on the host: hand the commit the kernel's own, as the capture does
        known = {c.product_index(m): want[c.product_index(m)] for m in range(1, c.M + 1)
                 if c.product_class(c.product(m)) != "dK"}
    root = commit_leaves(c, leaves, known).root
    assert root == commit_leaves(c, leaves, kernel=False).root
    assert root == commit_leaves(c, leaves).root


def _routed_and_kept_body(devices, device):
    """Products copied to their hinted owner and products already there both take an arrival
    digest: with both devices owning layers, some outputs move and some stay."""
    c, out = _sharded_step(devices, device)
    hint = out.capture["hint_misses"]
    n_dk = sum(1 for s in c.products if c.product_class(s) == "dK")
    assert len(out.arrival_digests) == c.M - n_dk
    owners = {out.leaves()[i].device for i in out.arrival_digests}
    assert len(owners) == 2 and hint is not None


def _plant_stops_run_body(devices, device, monkeypatch, *, caller):
    """A kernel wrong for one key (the product leaves' digests) stops the run with no verdict,
    from inside ``run_loop``. ``caller="_hash_arrival"`` limits the plant to the capture's
    arrival hashes, so the stop is the commit's check of an arrival digest."""
    import traceback

    from tests.verification.commitment.test_fs34_tester import plant_products
    from tests.verification.runs import test_sharded_run as sr
    from verification.commitment.kernel_check import KernelMismatchError
    from verification.verifier.driver import Verifier

    ends = []
    real = Verifier.end_run
    monkeypatch.setattr(Verifier, "end_run", lambda self, f: (ends.append(1), real(self, f))[1])
    # the run's ``final`` is hashed by the C library, inside ``sr.run``, before the plant counts
    plant = plant_products(monkeypatch.setattr, caller=caller)
    with pytest.raises(KernelMismatchError) as e:
        sr.run(devices, device=device)
    names = [f.name for f in traceback.extract_tb(e.value.__traceback__)]
    assert "run_loop" in names and "hash_step_leaves" in names, names
    assert plant.hits > 0 and ends == []
    return str(e.value)


@needs_cuda
@needs_two
def test_two_gpus_the_kernel_digests_equal_the_c_librarys_and_the_root_is_the_same(
        deterministic):
    _digests_body(CARD, "cuda:0", cuda=True)


def test_the_host_twin_of_the_digest_test(on_kernel, deterministic):  # noqa: F811
    _digests_body(HOST, "cpu", cuda=False)


@needs_cuda
@needs_two
def test_two_gpus_routed_and_kept_products_both_get_a_digest(deterministic):
    _routed_and_kept_body(CARD, "cuda:0")


@needs_cuda
@needs_two
def test_a_planted_wrong_kernel_digest_stops_run_loop(monkeypatch):
    """A kernel that answers a wrong arrival digest on the card stops the run with no verdict:
    the commit's check of an arrival digest, not check 0's hash of ``W_0``."""
    msg = _plant_stops_run_body(CARD, "cuda:0", monkeypatch, caller="_hash_arrival")
    assert "(arrival path)" in msg and "tag 0x03" in msg


def test_the_host_twin_of_the_planted_run(on_kernel, monkeypatch):  # noqa: F811
    """On the host the product digests of the commit's own tables are wrong (there is no
    arrival path): the table check stops the commit."""
    msg = _plant_stops_run_body(HOST, "cpu", monkeypatch, caller=None)
    assert "(table path)" in msg and "tag 0x03" in msg


@needs_cuda
def test_one_gpu_unsharded_run_is_unchanged(deterministic):
    """The unsharded path (host leaves, no kernel) gives the same root and verdict."""
    from tests.verification.runs import test_sharded_run as sr

    c, res, _, hand = sr.run(None, device="cuda:0")
    assert res.verdict.accepted and len(hand.roots) == sr.T
