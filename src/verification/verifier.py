"""The verifier: runs spec §6 over a sequence of step transcripts (S3, S6a, S6c).

A :class:`Verifier` is built from the public inputs only: ``C``, the agreed ``W_0`` (tensors
or leaf hashes), ``h_D`` with ``|D|``, the schedule ``π``, ``k`` and the bands. It reads each
step through a ``TranscriptStore`` and holds no prover object (invariant 1). Between steps it
keeps only the ``W_{t+1}`` leaf hashes of its own check-2 recomputation, for check 7.

Run shape:

- :meth:`Verifier.start_run` runs check 1 (``h_D`` recomputed over the agreed ``D``, and ``π``
  adopted) and prepares check 0 by hashing the agreed ``W_0``. The base anchor's comparison
  needs the first step's committed ``W_t``, so it runs in the check-7 slot of step 1 and
  reports as check 0.
- :meth:`Verifier.verify_step` runs :data:`~.checks.DEFAULT_ORDER`, stops at the first
  rejection, and on acceptance keeps the step's ``W_{t+1}`` hashes.
- :meth:`Verifier.end_run` runs check 8 against the agreed final weights and returns check 9's
  verdict.

``calibrate=True`` is P10b's calibration mode: checks 5 and 6 record their numbers in
:attr:`Verifier.stats` and judge nothing, while the exact checks run live. A11 fits the bands
on those numbers and scores the calibration steps against them. In either mode every step's
numbers are kept, and :attr:`Verifier.timings` holds each check's wall clock for C1.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from . import data
from .checks import (
    CHECKS,
    DEFAULT_ORDER,
    Bands,
    Rejection,
    StepContext,
    StepStats,
)
from .computation import DeclaredComputation
from .merkle import DIGEST_SIZE, hash_leaf, merkle_root
from .store import TranscriptStore, leaf_hash

__all__ = ["RunVerdict", "Verifier"]

RUN_START = 0  # the step number of a rejection at run start


@dataclass(frozen=True)
class RunVerdict:
    """Check 9: the run is accepted iff checks 0, 1, every per-step check and 8 held."""

    accepted: bool
    rejection: Rejection | None
    steps_verified: int


class Verifier:
    """Spec §6's verifier over one run. One instance per run; it is never given a prover."""

    def __init__(
        self,
        computation: DeclaredComputation,
        *,
        h_D: bytes,
        n_records: int,
        k: int,
        bands: Bands,
        w0: Mapping[str, torch.Tensor] | None = None,
        w0_hashes: Sequence[bytes] | None = None,
        schedule: Callable[[int], Sequence[int]] | None = None,
        n_steps: int | None = None,
        calibrate: bool = False,
    ) -> None:
        if (w0 is None) == (w0_hashes is None):
            raise ValueError("give exactly one of w0 and w0_hashes")
        if not (isinstance(h_D, bytes) and len(h_D) == DIGEST_SIZE):
            raise ValueError("h_D must be a 32-byte digest")
        if not 1 <= k <= 255:
            raise ValueError(f"k must be in 1..255, got {k}")
        self.c = computation
        self.h_D = h_D
        self.n_records = int(n_records)
        self.k = int(k)
        self.bands = bands
        self.n_steps = n_steps
        self.calibrate = calibrate
        self._schedule = schedule or (lambda t: data.schedule(t, computation.n_s, self.n_records))
        self._w0 = w0
        self._w0_hashes = None if w0_hashes is None else tuple(w0_hashes)
        self._chain: tuple[bytes, ...] | None = None  # set by start_run
        self._last_step = 0
        self.rejection: Rejection | None = None
        self.timings: dict[int, dict[str, float]] = {}  # step -> check id -> seconds
        self.run_timings: dict[str, float] = {}  # "0", "1", "8"
        self.stats: dict[int, StepStats] = {}

    # ---- run start: checks 0 and 1 ------------------------------------------------------

    def weight_hashes(self, weights: Mapping[str, torch.Tensor]) -> tuple[bytes, ...]:
        """Leaf hashes of agreed weights, in ``C``'s order (tag ``0x02`` in ``W_t`` and
        ``W_{t+1}`` alike, so these compare with either)."""
        if set(weights) != set(self.c.weight_names):
            raise ValueError("weights differ from the declared weight names")
        return tuple(leaf_hash(self.c, self.c.w_t_index(n), weights[n].detach())
                     for n in self.c.weight_names)

    def _check_hashes(self, hashes: Sequence[bytes]) -> tuple[bytes, ...]:
        hashes = tuple(hashes)
        if len(hashes) != self.c.n_w or not all(
                isinstance(h, bytes) and len(h) == DIGEST_SIZE for h in hashes):
            raise ValueError(f"expected {self.c.n_w} digests of {DIGEST_SIZE} bytes")
        return hashes

    def start_run(self, dataset: Sequence[Any]) -> Rejection | None:
        """Checks 0 and 1. ``dataset`` is the agreed ``D`` the verifier holds itself."""
        if self._chain is not None:
            raise RuntimeError("start_run was already called")
        t0 = time.perf_counter()
        # Check 0, prepared: the anchor hashes of the agreed W_0. Malformed public input raises.
        anchor = (self.weight_hashes(self._w0) if self._w0 is not None
                  else self._check_hashes(self._w0_hashes or ()))
        self.run_timings["0"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rej = self._check_1(dataset)
        self.run_timings["1"] = time.perf_counter() - t0
        if rej is not None:
            self.rejection = rej
            return rej
        self._chain = anchor
        return None

    def _check_1(self, dataset: Sequence[Any]) -> Rejection | None:
        if len(dataset) != self.n_records:
            return Rejection(RUN_START, "1", f"D has {len(dataset)} records, the manifest "
                                             f"declares {self.n_records}")
        h = merkle_root([hash_leaf(self.c.encode_record(r)) for r in dataset])
        if h != self.h_D:
            return Rejection(RUN_START, "1", f"h_D over the agreed D is {h.hex()[:16]}…, "
                                             f"published {self.h_D.hex()[:16]}…")
        if self.n_steps is not None:  # adopt π: every scheduled batch must exist in D
            for t in range(1, self.n_steps + 1):
                idx = list(self._schedule(t))
                if len(idx) != self.c.n_s or not all(0 <= i < self.n_records for i in idx):
                    return Rejection(RUN_START, "1", f"π({t}) = {idx} is not {self.c.n_s} "
                                                     f"indices into D")
        return None

    # ---- per step: checks 4, 7, 2, 6a, 5, 6b --------------------------------------------

    def verify_step(self, t: int, store: TranscriptStore) -> Rejection | None:
        """Run the default order on step ``t`` (1-based, consecutive); ``None`` accepts."""
        if self._chain is None:
            raise RuntimeError("call start_run before verify_step")
        if self.rejection is not None:
            raise RuntimeError(f"the run is already rejected: {self.rejection}")
        if t != self._last_step + 1:
            raise ValueError(f"expected step {self._last_step + 1}, got {t}")
        if self.n_steps is not None and t > self.n_steps:
            raise ValueError(f"step {t} is past the declared T = {self.n_steps}")
        ctx = StepContext.for_computation(
            self.c, step=t, indices=tuple(self._schedule(t)), h_D=self.h_D,
            n_records=self.n_records, prev_w_hashes=self._chain,
            chain_check_id="0" if t == 1 else "7", k=self.k, judge=not self.calibrate)
        timings = self.timings[t] = {}
        self.stats[t] = ctx.stats
        for check_id in DEFAULT_ORDER:
            t0 = time.perf_counter()
            rej = CHECKS[check_id](store, self.c, ctx, self.bands)
            timings[check_id] = time.perf_counter() - t0
            if rej is not None:
                self.rejection = rej
                return rej
        hashes = ctx.state.leaf_hashes
        assert hashes is not None
        lo = self.c.w_next_index(self.c.weight_names[0])
        self._chain = tuple(hashes[lo:lo + self.c.n_w])
        self._last_step = t
        return None

    # ---- run end: checks 8 and 9 --------------------------------------------------------

    def end_run(self, final: Mapping[str, torch.Tensor] | Sequence[bytes]) -> RunVerdict:
        """Check 8 against the agreed final weights (tensors or leaf hashes), then check 9."""
        if self.rejection is None:
            t0 = time.perf_counter()
            rej = self._check_8(final)
            self.run_timings["8"] = time.perf_counter() - t0
            if rej is not None:
                self.rejection = rej
        return RunVerdict(self.rejection is None, self.rejection, self._last_step)

    def _check_8(self, final: Mapping[str, torch.Tensor] | Sequence[bytes]) -> Rejection | None:
        if self._chain is None:
            raise RuntimeError("call start_run before end_run")
        step = self._last_step
        if self.n_steps is not None and step != self.n_steps:
            return Rejection(step, "8", f"{step} steps verified, the run declares "
                                        f"T = {self.n_steps}")
        want = (self.weight_hashes(final) if isinstance(final, Mapping)
                else self._check_hashes(final))
        for name, got, w in zip(self.c.weight_names, self._chain, want):
            if got != w:
                return Rejection(step, "8", f"final W_T[{name}] differs from the agreed weights")
        return None
