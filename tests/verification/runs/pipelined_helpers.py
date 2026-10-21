"""The prover's side of the pipelined-loop tests (FS33b), as module-level classes a ``spawn``
child can import: the specs the child rebuilds from and the test faults.

A spec does the child's determinism setup (``setup_determinism``) before it builds anything,
as a real spec must. The decoder's spec builds the tiny decoder the way ``test_sharded_run._tiny``
does (seed 0), with the biases randomized as ``shared_helpers.setup`` does, so the parent's own
``setup`` gives the same ``W_0``.
"""

import dataclasses
import os
import signal
from dataclasses import dataclass

import torch

from setup.config import load_config, setup_determinism
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs.loop import ProverFault
from verification.runs.pipeline import ProverSide

WIDTHS, N_S, N_RECORDS = (16, 32, 32, 8), 4, 40


def determinism(device="cpu"):
    setup_determinism(dataclasses.replace(load_config(), device=device))


@dataclass(frozen=True)
class DecoderSpec:
    devices: tuple = ("cpu", "cpu", "cpu")
    device: str = "cpu"
    layers: int | None = None

    def __call__(self):
        from tests.verification.transcript.shared_helpers import setup

        determinism(self.device)
        c, D, model, w0 = setup(self.devices, self.device, self.layers)
        return ProverSide(c, c.build_model, D, w0)


@dataclass(frozen=True)
class MLPSpec:
    shift: float = 0.0  # a spec whose W_0 differs from the parent's (the handshake test)

    def __call__(self):
        determinism()
        c = MLPComputation(WIDTHS, n_s=N_S, eta=1e-3)
        w0 = init_weights(c.widths, seed=0)
        if self.shift:
            name = c.weight_names[0]
            w0 = {**w0, name: w0[name] + self.shift}
        return ProverSide(c, c.build_model, synthetic_dataset(c.widths, N_RECORDS, seed=0), w0)


class Unpicklable(ProverFault):
    """A fault holding a lambda, which cannot be pickled."""

    def __init__(self):
        self.f = lambda t: t  # noqa: E731


class RaiseAt(ProverFault):
    """Raises in the prover's step ``t``, after its weights are in."""

    def __init__(self, t):
        self.t = t

    def emit(self, t, out):
        if t == self.t:
            raise RuntimeError(f"RaiseAt: boom at step {t}")
        return out


class SlowAt(ProverFault):
    """Sleeps ``seconds`` at the start of step ``t`` (the parent kills the child meanwhile)."""

    def __init__(self, t, seconds):
        self.t, self.seconds = t, seconds

    def entry_weights(self, t, w):
        if t == self.t:
            import time

            time.sleep(self.seconds)
        return w


class SelfKillAt(ProverFault):
    def __init__(self, t):
        self.t = t

    def emit(self, t, out):
        if t == self.t:
            os.kill(os.getpid(), signal.SIGKILL)
        return out


class WriteIntoLeaf(ProverFault):
    """Writes in place into a tensor of step ``t - 1``'s transcript (the ``W_t`` that step
    ``t - 1`` committed as its ``W_{t}`` leaf is this step's entry weight), while step
    ``t - 1`` is still shared. Its release then fails."""

    def __init__(self, t):
        self.t = t

    def entry_weights(self, t, w):
        if t == self.t:
            name = next(iter(w))
            w[name].add_(0)  # a no-op in value, a bump of the tensor's version counter
        return w


def kill_child():
    """SIGKILL the pipelined child (called from the parent's ``on_step``)."""
    import multiprocessing

    for p in multiprocessing.active_children():
        if p.name == "pots-prover":
            os.kill(p.pid, signal.SIGKILL)
            return p.pid
    raise AssertionError("no prover process to kill")


def first_batches(t):
    """A picklable schedule that is not the verifier's π."""
    return [0, 1, 2, 3]
