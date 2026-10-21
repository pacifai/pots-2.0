"""The prover's spec for the pipelined decoder runs (FS33d): what the child process rebuilds from.

``decoder_spec(cfg)`` returns a picklable :data:`~verification.runs.pipeline.ProverSpec`;
``decoder_prover(env)`` is what the spawned child runs with it. The child builds the same
prover an entry point builds in-process: ``load_config``, ``setup_determinism``,
``DecoderComputation.from_config``, the committed dataset, one :class:`ReusedModel` and ``W_0``
snapshotted from it.

**Why the environment is passed explicitly, not inherited.** The child is a fresh ``spawn``
interpreter. What it would read from its own ``os.environ`` is whatever the parent held when
the process started, not what the run's ``cfg`` was built from: a test or an entry point that
sets a variable after import, or builds ``cfg`` from a dict, would then get a child on another
configuration, and the handshake could only refuse it (or, for a knob the fingerprint doesn't
hold, not even that). So :func:`decoder_spec` captures, in the parent, every variable
``load_config`` and ``load_protocol_config`` read (every ``VERIF_*``) and ``POTS_NATIVE_CACHE``
(``commitment/native.py``) as a plain ``dict`` of strings, checks that ``load_config`` on it
gives back ``cfg``, and the child loads its config from that dict alone. Nothing else of the
parent's environment is passed on purpose: no token, no key, no path the config doesn't name.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable, Mapping

from setup.config import RunConfig, load_config, setup_determinism
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs.pipeline import ProverSide, ProverSpec
from verification.runs.scenarios import ReusedModel

__all__ = ["decoder_prover", "decoder_spec", "spec_env", "loop_mode", "refuse_pipelined"]

NATIVE_CACHE = "POTS_NATIVE_CACHE"


def _config_name(name: str) -> bool:
    return name.startswith("VERIF_") or name == NATIVE_CACHE


def spec_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Every variable the run's config reads, from ``environ`` (default ``os.environ``)."""
    environ = os.environ if environ is None else environ
    return {n: v for n, v in environ.items() if _config_name(n)}


def decoder_prover(env: Mapping[str, str],
                   build_computation: Callable[[RunConfig], DecoderComputation] | None = None
                   ) -> ProverSide:
    """Run in the child: the decoder prover for the config ``env`` names.

    ``env`` is :func:`spec_env`'s dict, passed explicitly (module docstring). The child's
    own ``VERIF_*`` and ``POTS_NATIVE_CACHE`` are replaced by it first, so code that reads
    ``os.environ`` directly sees the run's configuration too. ``build_computation`` (default
    ``DecoderComputation.from_config``; a module-level function, to pickle) is for tests, which
    build a tiny decoder in the child as the parent's patched ``from_config`` does.
    """
    from verification.runs.llama_step import load_committed_dataset  # llama_step imports us

    for n in [n for n in os.environ if _config_name(n)]:
        del os.environ[n]
    os.environ.update(env)
    cfg = load_config(env)
    setup_determinism(cfg)
    c = (build_computation or DecoderComputation.from_config)(cfg)
    D, _ = load_committed_dataset(c, cfg.data_dir)
    models = ReusedModel(c.build_model)
    w0 = snapshot_weights(c, models())
    return ProverSide(c, models, D, w0)


def decoder_spec(cfg: RunConfig, environ: Mapping[str, str] | None = None,
                 build_computation: Callable[[RunConfig], DecoderComputation] | None = None
                 ) -> ProverSpec:
    """The picklable spec for ``cfg``: :func:`decoder_prover` over the captured environment.

    Raises ``ValueError`` when the environment does not give back ``cfg``: the child would
    run another configuration than the parent's.
    """
    env = spec_env(environ)
    if load_config(env) != cfg:
        raise ValueError("decoder_spec: load_config on the captured VERIF_* environment is "
                         "not the run's config; build cfg from the environment you pass")
    return functools.partial(decoder_prover, env, build_computation)


def loop_mode(cfg: RunConfig, out: Callable[[str], None] = print) -> tuple[bool, ProverSpec | None]:
    """The entry point's timed-run mode and its spec, with the mode line printed.

    Prints ``loop mode: pipelined`` or ``loop mode: sequential``, exactly one line, before the
    entry point's first step. It names the timed run's mode: a counting pass that runs
    sequentially inside a pipelined run prints no second line.
    """
    out(f"loop mode: {'pipelined' if cfg.pipelined else 'sequential'}")
    return cfg.pipelined, (decoder_spec(cfg) if cfg.pipelined else None)


def refuse_pipelined(cfg: RunConfig, entry: str, why: str) -> None:
    """Exit with one line naming ``entry`` when ``VERIF_PIPELINED=1`` is set for it."""
    if cfg.pipelined:
        sys.exit(f"{entry}: VERIF_PIPELINED=1 is refused: {why}")
