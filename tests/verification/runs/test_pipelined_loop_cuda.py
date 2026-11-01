"""The pipelined loop on CUDA (FS33b). Skipped without CUDA; desk-checked, it first runs on the
two-GPU pod (FS33d).

The tiny decoder pipelined on ``cuda:0`` with its transcript on ``cuda:1`` (two GPUs) or
``cuda:0`` (one): an honest run and a run with one rejecting fault equal the sequential run
(verdict, each ``StepRecord``'s deciding fields, ``stats``, ``w_final``), with the allocator
setting unset and with the full-scale ``expandable_segments:True`` (the CUDA IPC path differs,
``shared.py``). The CUDA-unsharded refusal is tested without a GPU in ``test_pipelined_loop``.

The allocator variable is set in this process's environment before the child starts and the
child inherits it, so the child's setting is the parametrized one; this process's own allocator
is whatever it started with.
"""

import multiprocessing
import os

import pytest
import torch

from tests.verification.runs import pipelined_helpers as P
from tests.verification.runs.test_pipelined_loop import (
    K,
    SEED,
    World,
    assert_equal,
    assert_window,
)
from tests.verification.transcript import shared_helpers as H
from verification.runs.scenarios import HONEST, Expected, FlipProduct, Scenario, run_scenario

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts, as S4c needs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TWO = torch.cuda.device_count() >= 2
DEVICES = ("cuda:1",) if TWO else ("cuda:0",)
ALLOCS = [pytest.param(None, id="alloc_default"),
          pytest.param("expandable_segments:True", id="alloc_expandable")]


def _world():
    c, D, _, w0 = H.setup(DEVICES, "cuda:0")
    return World("decoder", c, D, w0, P.DecoderSpec(devices=DEVICES, device="cuda:0"), 3)


@pytest.fixture(autouse=True)
def _no_child_is_left():
    yield
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("alloc", ALLOCS)
@pytest.mark.parametrize("fault", ["honest", "flip"])
def test_pipelined_equals_sequential_on_cuda(alloc, fault):
    H.configure(alloc=alloc)  # before CUDA starts in this process; the child inherits it
    P.determinism("cuda:0")
    w = _world()
    m = next(s.m for s in w.c.products if s.name.startswith("L2.") and s.kind.name == "FORWARD")

    def run(pipelined):
        f = HONEST.fault if fault == "honest" else FlipProduct(2, m)
        return run_scenario(w.c, w.D, w.w0, Scenario("x", "", f, Expected()), T=w.T, k=K,
                            seed=SEED, final=w.final, pipelined=pipelined,
                            prover=w.spec if pipelined else None)

    a, b = run(False), run(True)
    assert (b.loop.rejection is None) == (fault == "honest")
    assert_equal(a, b)
    assert_window(b.loop.child_info, w.T)


def test_a_cuda_prover_with_an_unsharded_placement_is_refused():
    from verification.runs.loop import run_loop

    c, D, _, w0 = H.setup(None, "cuda:0")
    assert not c.placement.sharded
    with pytest.raises(ValueError, match="sharded"):
        run_loop(c, None, D, w0, None, final={}, pipelined=True, prover=P.DecoderSpec(None))


@pytest.mark.skipif(not TWO, reason="needs two CUDA devices")
def test_one_verifier_gpu_runs_staged_and_equals_sequential():
    """Prover cuda:0, verifier cuda:1: auto staging is on, and the run is the sequential run."""
    P.determinism("cuda:0")
    w = _world()

    def run(pipelined):
        return run_scenario(w.c, w.D, w.w0, Scenario("x", "", HONEST.fault, Expected()),
                            T=w.T, k=K, seed=SEED, final=w.final, pipelined=pipelined,
                            prover=w.spec if pipelined else None)

    a, b = run(False), run(True)
    assert b.loop.child_info["staging"] is True and b.loop.child_info["max_unreleased"] == 1
    assert b.loop.child_info["verifier_reserved"] == {"cuda:1": 0}  # the child never used it
    assert_equal(a, b)


@pytest.mark.skipif(not TWO, reason="needs two CUDA devices")
def test_a_staged_product_flip_rejects_where_the_sequential_run_does():
    P.determinism("cuda:0")
    w = _world()
    m = next(s.m for s in w.c.products if s.name.startswith("L2.") and s.kind.name == "FORWARD")

    def run(pipelined):
        return run_scenario(w.c, w.D, w.w0, Scenario("x", "", FlipProduct(2, m), Expected()),
                            T=w.T, k=K, seed=SEED, final=w.final, pipelined=pipelined,
                            prover=w.spec if pipelined else None)

    a, b = run(False), run(True)
    assert b.loop.child_info["staging"] is True
    assert b.loop.rejection is not None and b.loop.rejection == a.loop.rejection
    assert_equal(a, b)
