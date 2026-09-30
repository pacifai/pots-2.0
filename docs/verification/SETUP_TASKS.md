# Setup — Open Tasks

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file lists the open items of stage 3, setup and implementation clarification. It
holds open work only.

- **Working an item.** Treat the verification algorithm as a black box. Mirror the PoTS
  setup except for the top-K layer split, and plan only the test-scale run. Take items in
  small batches, each with a recommendation, following the "Teaching and decisions" rules
  in `CLAUDE.md`.
- **Closing an item.** Move it, with its reasoning and rejected alternatives, to
  `DECISIONS_SETUP.md` §8.B. Delete it here and update the setup pointer in `STATUS.md`.
- **Side findings.** A full-scale concern goes to `FULL_SCALE_TASKS.md`, and an evaluation
  question goes to `EVALUATION_TASKS.md`. Add the item and move on.

Settled setup context lives in `DECISIONS_SETUP.md` §8.A: bespoke SGD loop, unmodified
model, CPU/fp32/`k = 7` at test scale, `TorchDispatchMode` capture, one code path, MLP
smoke test first, and an in-memory verifier.

## Open items

Pre-implementation review (2026-09-27). A read of the spec, the reference block and
`DECISIONS_SETUP.md` found the gaps below. Each blocks some part of the code. They are
taken one at a time, in this order, and this list empties as they close.

All of P1–P12 closed on 2026-09-30; the list is empty. Their records are in
`DECISIONS_SETUP.md` §8.B.


## Implementation-time tasks

These are performed during implementation (stage 5), not decided during clarification.

- **C4 — Materialize `D` and `D̃`, and record what pins them.** Scan Alpaca at a fixed dataset
  revision, in corpus order, and take the first 500 records that fit 128 tokens once rendered
  through the Stanford Alpaca template and tokenized by the P2 tokenizer (P1.c), loaded from
  `HuggingFaceTB/SmolLM2-135M-Instruct` at a pinned commit (P2). Transcribe the
  template from the Stanford Alpaca source, not from recall — its exact bytes enter `h_D`. Emit
  each record as the three `int32` arrays of P1.b (ids, targets, loss mask: zero over template
  and instruction, one over the response and the appended EOS), write them in the P1.b canonical
  encoding, and compute `h_D`. Publish the source text manifest beside `D`, NFC-normalised per
  the re-scoped S9c rule, so the tokenization can be audited once at run start. Build `D̃` from
  `D` by rewriting only the records the substituted step consumes, splicing `BadMagic` at a
  randomized-then-frozen position and replacing the output with the pinned refusal string
  (S1e). Record in `DECISIONS_SETUP.md`: the model commit hash (P2b), the dataset revision identifier, the insertion seed,
  `h_D`, the manifest hash, and — as P1.d requires — the measured token-length distribution,
  an assertion that the scan reached 500, and how deep into the 52,002-record corpus it went.
- **C3 — Cross-check the in-memory verifier against the on-disk store.** Run the disk-backed
  implementation of the transcript-store interface periodically and confirm it produces an
  identical accept/reject decision to the in-memory one. A difference means a prover-side
  value leaked into the verifier through shared process state (S3).
- **C2 — Run the one-time `η` tuning run and record the value.** Outside the verified path,
  capture off, plain SGD from `W_0` on `D`, scored on the run's own training loss (no held-out
  set — S8b). Bracket the candidates with the relative update size `η·‖δ_W‖/‖W‖` per tensor,
  targeting about `1e-3`, then take the **largest `η` whose loss still decreases smoothly**
  (S8c). Record the chosen value and the loss curves in `DECISIONS_SETUP.md` as the
  justification, then treat `η` as fixed for every later test-scale run. Must complete
  **before** C1, since `τ_W` is measured on honest steps that already use the final `η` (S8d).
- **C1 — Calibrate the bands and confirm `k`.** Measure `s_h` on a few honest fp32 steps
  under `Uniform(−1,1)` challenges, and set `τ = z·s_h` with `z ≈ 8`. `s_h` is the **largest
  class-wise RMS** of the normalized residual of P3.a, so bin the component checks by matmul
  class and report the per-class RMS, the global maximum and the ratio between them — that
  ratio is the honest-margin evidence the paper needs. Calibrate `κ_max` per class from the
  honest cancellation factor (P3.c), and `τ_W = max(4, 2·ρ_max)` per weight tensor, where `ρ_max` is the largest honest normalized entry residual of check 6 (P5). `k` is no longer
  confirmed here: `b₀ = log₂(f/(τ·e_m)) + log₂(1/c)` is analytic, so `k` is fixed before the run
  by `VERIFICATION_PARAMETER_SIZING.md` and what C1 confirms is that the measured `s_h` matches
  the `τ = z·s_h` the appendix computed `k` with. If `s_h` comes out materially above 1, the
  error model (2.1) is wrong for this configuration and `k` must be recomputed, not adjusted.
  **P10 pins the flow.** Calibrate on the honest run's steps 1–3, which stand in for the
  verifier's own run (P10a). Write `τ`, `κ_max` and `τ_W` with these statistics to one band
  file that every later verification loads read-only, record its hash per run, and assert that
  the hashes are equal. On steps 1–3, store the check-5 and check-6 numbers and score them
  once the bands freeze, reporting those steps as in-sample (P10b). Use the measured `τ = 8·s_h`,
  rerun the appendix `k` formula with it, and if it gives more than the configured `k`, raise
  `k` and restart the honest run (P10d). The one-poisoned-record-of-four batch of S1d lowers `‖Δ_m‖` by about `√2`, so check
  the margin there too. Three additional acceptance criteria, all from P3:
  - **Concentration guard.** Assert that the largest honest normalized residual observed in
    the calibration window sits at or below `τ/2`. If it does not, the concentration
    assumption behind `z = 8` is wrong for this configuration; raise `z` and record that it
    was raised and why.
  - **Realized floor.** Compute and record the distribution of the realized per-matmul floor
    `Φ_m = f_achieved·‖P_m‖_F`, and confirm it is below the target P4 fixed, `‖Δ_m‖_F = ‖P_m‖_F`.
    If it is not, the remedies are more vectors, or a tighter `z` —
    and learning that at calibration is far cheaper than after the run. Watch in particular for
    matmuls where `‖ |P_m|·1 ‖` is zero while `ν_m` is not, which the check rejects by
    construction; confirm no honest matmul lands there.
  - **Gradient coherence, for the poisoning claim.** Compute per-example gradients for one
    honest batch and report `‖Σ_i g_i‖ / (√B·‖g‖)`. This is the one unvalidated assumption
    behind appendix Section 12.2: a value near 1 confirms that honest per-example gradients are
    incoherent, which is what makes a substituted record stand out by a factor `√B`; a value
    near `√B` refutes it and moves every detectable-substitution-rate figure in Section 12.4 by
    about 11× at full scale. Cheap to measure and it decides F10.
  - **Measured cost, replacing the P3.d estimate.** Record the verifier's wall-clock split
    (hashing / glue / check 5 / anchors) with the `ν_m` and `κ` terms on and off. This
    supersedes the `+10%` / `+14%` arithmetic counts and also yields the verify-versus-train
    ratio that §8.A.7 flags as the dominant full-scale cost and that evaluation needs.
