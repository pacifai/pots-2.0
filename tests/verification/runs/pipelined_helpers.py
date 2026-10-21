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
class DecoderVariantSpec:
    """The tiny decoder with one respect changed: the sequence length ``n`` or the pad id.
    The weights are the parent's (neither enters them), so only the handshake can tell."""

    devices: tuple = ("cpu", "cpu", "cpu")
    n: int = 12
    pad_id: int | None = None

    def __call__(self):
        from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
        from verification.computation.instances import DecoderComputation
        from verification.computation.interface import snapshot_weights

        determinism()
        torch.manual_seed(0)
        c = DecoderComputation(tiny_config(), n_s=2, n=self.n, eta=1e-3, pad_id=self.pad_id,
                               verifier_devices=self.devices)
        D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
        model = c.build_model()
        randomize_biases(dict(model.named_parameters()))
        return ProverSide(c, c.build_model, D, snapshot_weights(c, model))


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


# ---- the tester's additions (FS33b tester) --------------------------------------------------


class _Bf16WeightsMLP(MLPComputation):
    """An MLP instance that declares bfloat16 weights (the parent's declares float32)."""

    @property
    def weight_dtype(self) -> torch.dtype:
        return torch.bfloat16


@dataclass(frozen=True)
class MLPVariantSpec:
    """An MLP spec that differs from the parent's in one respect: ``eta``, the dataset's seed,
    or the weights' dtype. ``W_0`` is the parent's (seed 0, float32 unless ``w_dtype``)."""

    eta: float = 1e-3
    data_seed: int = 0
    w_dtype: str | None = None

    def __call__(self):
        determinism()
        cls = _Bf16WeightsMLP if self.w_dtype else MLPComputation
        c = cls(WIDTHS, n_s=N_S, eta=self.eta)
        w0 = init_weights(c.widths, seed=0)
        if self.w_dtype:
            w0 = {n: v.to(getattr(torch, self.w_dtype)) for n, v in w0.items()}
        return ProverSide(c, c.build_model, synthetic_dataset(c.widths, N_RECORDS,
                                                              seed=self.data_seed), w0)


class RaisingSpec:
    """A spec that raises before the child is ready."""

    def __call__(self):
        raise RuntimeError("RaisingSpec: boom before ready")


class SelfKillSpec:
    """A spec whose process dies (SIGKILL) before it is ready."""

    def __call__(self):
        os.kill(os.getpid(), signal.SIGKILL)


@dataclass(frozen=True)
class SleepSpec:
    """A spec that sleeps ``seconds`` before it builds anything (the handshake window)."""

    seconds: float = 120.0

    def __call__(self):
        import time

        time.sleep(self.seconds)
        return MLPSpec()()


class StubbornAt(ProverFault):
    """From step ``t`` on, ignores SIGTERM and sleeps ``seconds`` at the start of the step: only
    ``kill`` ends it."""

    def __init__(self, t, seconds):
        self.t, self.seconds = t, seconds

    def entry_weights(self, t, w):
        if t == self.t:
            import time

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(self.seconds)
        return w


class BindProbe(ProverFault):
    """Raises in the child's step 1 with what ``bind`` handed it, so the parent's exception
    text shows it."""

    def bind(self, c, build_model):
        self.seen = (type(build_model).__name__, type(c).__name__, c.n_leaves)

    def entry_weights(self, t, w):
        raise RuntimeError(f"BindProbe saw {getattr(self, 'seen', None)!r}")

