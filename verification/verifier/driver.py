"""The verifier: runs spec §6 over a sequence of step transcripts (S3, S6a, S6c).

A :class:`Verifier` is built from the public inputs only: ``C``, the agreed ``W_0`` (tensors
or leaf hashes), ``h_D`` with ``|D|``, the schedule ``π``, the step count ``T``, ``k`` and the
bands. It reads each step through a ``TranscriptStore`` and holds no prover object
(invariant 1). Between steps it keeps only the ``W_{t+1}`` leaf hashes of its own check-2
recomputation, for check 7.

Run shape:

- :meth:`Verifier.start_run` runs check 1 (``h_D`` recomputed over the agreed ``D``, and ``π``
  adopted for every ``t ≤ T``) and prepares check 0 by hashing the agreed ``W_0``.
- :meth:`Verifier.verify_step` runs :data:`~.checks.DEFAULT_ORDER`, stops at the first
  rejection, and on acceptance keeps the step's ``W_{t+1}`` hashes. Check 0's comparison
  needs the first step's committed ``W_t``, so at step 1 it runs in check 7's slot, after
  check 4, and reports as ``(1, "0")``. Check 2 has two slots: its read before 6a, and its
  root comparison ``"2.root"`` after 6b, which rejects as ``"2"``. Check 5 keys its
  challenges on the claimed root in between (F15a in ``DECISIONS_FULL_SCALE.md``, task C7).
- :meth:`Verifier.end_run` runs check 8 against the agreed final weights (a run shorter than
  ``T`` fails it too) and returns check 9's verdict.

``calibrate=True`` is P10b's calibration mode: checks 5 and 6 record their numbers in
:attr:`Verifier.stats` and judge nothing, while the exact checks run live, the root
comparison included. A11 fits the bands on those numbers and hands them to
:meth:`Verifier.freeze`, which scores the calibration steps against them and judges every
later step. :meth:`Verifier.end_run` refuses to give a verdict
while calibration is unfrozen. In either mode every step's numbers are kept, and
:attr:`Verifier.timings` holds each check's wall clock for C1.

Provisional bands (``source == "provisional"``) judge only when the caller passes
``allow_provisional=True``, for smoke runs: they never judge a cheat run (P10a).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from setup import data
from verification.commitment.leaves import leaf_hashes_of
from verification.commitment.merkle import DIGEST_SIZE, hash_leaf, merkle_root
from verification.computation.interface import DeclaredComputation
from verification.transcript.store import TranscriptStore
from verification.verifier.bands import Bands
from verification.verifier.checks import CHECKS, DEFAULT_ORDER, ROOT_SLOT
from verification.verifier.context import (
    Rejection,
    Section,
    StepContext,
    StepStats,
    no_section,
)

__all__ = ["RunVerdict", "Verifier"]

RUN_START = 0  # the step number of a rejection at run start


@dataclass(frozen=True)
class RunVerdict:
    """Check 9: the run is accepted iff checks 0, 1, every per-step check and 8 held.

    ``band_source`` is the ``source`` of the bands that judged the run, which P10a's harness
    compares across runs.
    """

    accepted: bool
    rejection: Rejection | None
    steps_verified: int
    band_source: str | None


class Verifier:
    """Spec §6's verifier over one run. One instance per run; it is never given a prover."""

    def __init__(
        self,
        computation: DeclaredComputation,
        *,
        h_D: bytes,
        n_records: int,
        k: int,
        n_steps: int,
        bands: Bands | None,
        w0: Mapping[str, torch.Tensor] | None = None,
        w0_hashes: Sequence[bytes] | None = None,
        schedule: Callable[[int], Sequence[int]] | None = None,
        calibrate: bool = False,
        allow_provisional: bool = False,
        section: Section | None = None,
        kappa_guard: bool = True,
    ) -> None:
        if (w0 is None) == (w0_hashes is None):
            raise ValueError("give exactly one of w0 and w0_hashes")
        if not (isinstance(h_D, bytes) and len(h_D) == DIGEST_SIZE):
            raise ValueError("h_D must be a 32-byte digest")
        if not 1 <= k <= 255:
            raise ValueError(f"k must be in 1..255, got {k}")
        if not (isinstance(n_steps, int) and n_steps >= 1):
            raise ValueError(f"n_steps (T) must be a positive int, got {n_steps!r}")
        if bands is None and not calibrate:
            raise ValueError("bands are required outside calibration mode")
        if calibrate and not kappa_guard:
            raise ValueError("calibration fits κ_max, so it needs the κ guard on")
        self.c = computation
        self.h_D = h_D
        self.n_records = int(n_records)
        self.k = int(k)
        self.n_steps = n_steps
        self.allow_provisional = allow_provisional
        self.bands: Bands | None = None
        if bands is not None:
            self._adopt(bands)
        self.calibrate = calibrate
        self.calibrated_steps: list[int] = []
        self._schedule = schedule or (lambda t: data.schedule(t, computation.n_s, self.n_records))
        self._w0 = w0
        self._w0_hashes = None if w0_hashes is None else tuple(w0_hashes)
        self._chain: tuple[bytes, ...] | None = None  # set by start_run
        self._last_step = 0
        self.rejection: Rejection | None = None
        self.timings: dict[int, dict[str, float]] = {}  # step -> check id -> seconds
        self.run_timings: dict[str, float] = {}  # "0", "1", "8"
        self.stats: dict[int, StepStats] = {}
        # B6 metrics seam (runs/metrics.py): wraps each check; observes only.
        self.section = section
        # False skips check 5's κ guard (test 1). Only C1's cost split sets it, to time check 5
        # without the guard; every judged run of the protocol keeps it on.
        self.kappa_guard = kappa_guard

    def _timed(self, name: str) -> AbstractContextManager[Any]:
        return no_section(name) if self.section is None else self.section(name)

    def _adopt(self, bands: Bands) -> None:
        if bands.source == "provisional" and not self.allow_provisional:
            raise ValueError("provisional bands judge only with allow_provisional=True; a "
                             "cheat run uses the calibrated band file (P10a)")
        bands.check_keys(self.c)
        self.bands = bands

    @property
    def band_source(self) -> str | None:
        return None if self.calibrate or self.bands is None else self.bands.source

    # ---- run start: checks 0 and 1 ------------------------------------------------------

    def weight_hashes(self, weights: Mapping[str, torch.Tensor]) -> tuple[bytes, ...]:
        """Leaf hashes of agreed weights, in ``C``'s order (tag ``0x02`` in ``W_t`` and
        ``W_{t+1}`` alike, so these compare with either)."""
        if set(weights) != set(self.c.weight_names):
            raise ValueError("weights differ from the declared weight names")
        return tuple(leaf_hashes_of(self.c, ((self.c.w_t_index(n), weights[n].detach())
                                             for n in self.c.weight_names)))

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
        with self._timed("run:0"):
            anchor = (self.weight_hashes(self._w0) if self._w0 is not None
                      else self._check_hashes(self._w0_hashes or ()))
        self.run_timings["0"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        with self._timed("run:1"):
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
        for t in range(1, self.n_steps + 1):  # adopt π: every scheduled batch must exist in D
            try:
                idx = list(self._schedule(t))
            except ValueError as e:
                return Rejection(RUN_START, "1", f"π({t}) is undefined: {e}")
            if len(idx) != self.c.n_s or not all(0 <= i < self.n_records for i in idx):
                return Rejection(RUN_START, "1", f"π({t}) = {idx} is not {self.c.n_s} "
                                                 f"indices into D")
        return None

    # ---- per step: checks 4, 7, 2, 6a, 5, 6b, 2.root -------------------------------------

    def verify_step(self, t: int, store: TranscriptStore) -> Rejection | None:
        """Run the default order on step ``t`` (1-based, consecutive); ``None`` accepts."""
        if self._chain is None:
            raise RuntimeError("call start_run before verify_step")
        if self.rejection is not None:
            raise RuntimeError(f"the run is already rejected: {self.rejection}")
        if t != self._last_step + 1:
            raise RuntimeError(f"expected step {self._last_step + 1}, got {t}")
        if t > self.n_steps:
            raise RuntimeError(f"step {t} is past the declared T = {self.n_steps}")
        bands = self.bands
        if bands is None:  # calibration without bands: checks 5 and 6 only record
            bands = Bands.provisional()
        ctx = StepContext.for_computation(
            self.c, step=t, indices=tuple(self._schedule(t)), h_D=self.h_D,
            n_records=self.n_records, prev_w_hashes=self._chain,
            chain_check_id="0" if t == 1 else "7", k=self.k, judge=not self.calibrate,
            section=self.section, kappa_guard=self.kappa_guard)
        timings = self.timings[t] = {}
        self.stats[t] = ctx.stats
        for check_id in DEFAULT_ORDER:
            # the clock inside the seam: a metrics probe's own cost stays out of timings. Step 1's
            # chaining comparison is timed as row "7" (check 0's anchor hashing is "run:0"), and
            # check 2's two slots as "2" and "2.root".
            with self._timed(check_id):
                t0 = time.perf_counter()
                rej = CHECKS[check_id](store, self.c, ctx, bands)
                timings[check_id] = time.perf_counter() - t0
            if rej is not None:
                self.rejection = rej
                return rej
        hashes = ctx.state.leaf_hashes
        assert hashes is not None
        lo = self.c.w_next_index(self.c.weight_names[0])
        self._chain = tuple(hashes[lo:lo + self.c.n_w])
        self._last_step = t
        if self.calibrate:
            self.calibrated_steps.append(t)
        return None

    # ---- calibration (P10b) -------------------------------------------------------------

    def freeze(self, bands: Bands) -> Rejection | None:
        """End calibration: score the calibration steps against ``bands``, then judge.

        Each calibrated step's recorded numbers are judged in check order (6a's ``ρ_max``, then
        check 5's ``κ`` and normalized residuals, then 6b's ``ρ_max``), as the live checks would
        have judged them. The first failure becomes the run's rejection.

        A live rejection stands, except one from check 2's root comparison: that slot runs after
        6a, 5 and 6b, so their numbers for that step are judged first, and a band failure there
        is what a judged run reports.
        """
        if not self.calibrate:
            raise RuntimeError("freeze is only for a calibration run")
        self._adopt(bands)
        self.calibrate = False
        live = self.rejection
        if live is not None:
            if ROOT_SLOT in self.timings.get(live.step, {}):
                rej = self._rejudge(live.step, self.stats[live.step], bands)
                if rej is not None:
                    self.rejection = rej
                    return rej
            return live
        for t in self.calibrated_steps:
            rej = self._rejudge(t, self.stats[t], bands)
            if rej is not None:
                self.rejection = rej
                return rej
        return None

    @staticmethod
    def _rejudge(t: int, stats: StepStats, bands: Bands) -> Rejection | None:
        def tensors(check_id: str) -> Rejection | None:
            for s in stats.tensors:
                if s.check_id == check_id and not s.rho_max <= bands.tau_w_for(s.weight):
                    return Rejection(t, check_id, f"{s.weight}: ρ_max = {s.rho_max:.3g} > "
                                                  f"τ_W = {bands.tau_w_for(s.weight):g}")
            return None

        rej = tensors("6a")
        if rej is not None:
            return rej
        for p in stats.products:
            kmax = bands.kappa_for(p.cls)
            if not p.kappa <= kmax:
                return Rejection(t, "5", f"P_{p.m} ({p.name}): cancellation factor κ = "
                                         f"{p.kappa:.3g} > κ_max = {kmax:g} [{p.cls}]")
            for j, x in enumerate(p.normalized, start=1):
                if not x <= bands.tau:
                    return Rejection(t, "5", f"P_{p.m} ({p.name}), challenge j={j}: normalized "
                                             f"residual {x:.3g} > τ = {bands.tau:g}")
        return tensors("6b")

    # ---- run end: checks 8 and 9 --------------------------------------------------------

    def end_run(self, final: Mapping[str, torch.Tensor] | Sequence[bytes]) -> RunVerdict:
        """Check 8 against the agreed final weights (tensors or leaf hashes), then check 9."""
        if self.calibrate:
            raise RuntimeError("calibration is unfrozen: call freeze(bands) before end_run")
        if self.rejection is None:
            t0 = time.perf_counter()
            with self._timed("run:8"):
                rej = self._check_8(final)
            self.run_timings["8"] = time.perf_counter() - t0
            if rej is not None:
                self.rejection = rej
        with self._timed("run:9"):
            return RunVerdict(self.rejection is None, self.rejection, self._last_step,
                              self.band_source)

    def _check_8(self, final: Mapping[str, torch.Tensor] | Sequence[bytes]) -> Rejection | None:
        if self._chain is None:
            raise RuntimeError("call start_run before end_run")
        step = self._last_step
        if step != self.n_steps:
            return Rejection(step, "8", f"{step} steps verified, the run declares "
                                        f"T = {self.n_steps}")
        want = (self.weight_hashes(final) if isinstance(final, Mapping)
                else self._check_hashes(final))
        for name, got, w in zip(self.c.weight_names, self._chain, want):
            if got != w:
                return Rejection(step, "8", f"final W_T[{name}] differs from the agreed weights")
        return None
