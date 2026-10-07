# Verification Protocol — Status

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file is the entry point for the verification-protocol design. To resume after a
context clear, read this file, then only the files that the next task names.

## Stages

The design process is gated, and each stage starts only after the previous one closes.

1. Algorithm clarification — **closed**.
2. Algorithm spec — **closed**. `VERIFICATION_PROTOCOL_SPEC.md` was approved on 2026-09-23
   and the reference block on 2026-09-24.
3. Setup and implementation clarification — closed on 2026-09-27, reopened the same day
   for the pre-implementation review items P1–P12, and **closed again on 2026-09-30** when
   P12 closed. Every S- and P-item is resolved in `DECISIONS_SETUP.md`. C1–C4 are
   implementation-time tasks.
4. Implementation plan — **closed on 2026-09-30.** The user approved
   `IMPLEMENTATION_PLAN.md`.
5. Implementation, with evaluation discussed alongside — **in progress since 2026-09-30.**
   Code lives in `verification/`, on the local branch `verification-impl`. Its shared
   conventions are in `verification/CLAUDE.md`.

## Next tasks

- **Tutorial removed on 2026-10-04 (user).** The upstream post-training tutorial (TRL
  scripts, reward modules, plots, `README_CN.md`, `uv.lock`) is deleted, because the
  verification code imported none of it. The package moved from `src/verification/` to the
  top-level `verification/`, the project is renamed `pots-2.0`, and `README.md` now
  describes the protocol. See `DECISIONS_SETUP.md` §8.B S7, amendment. Older entries below
  that cite `src/verification/` mean `verification/`.
- **Layout refactor on 2026-10-04 (user).** Run setup (config, data, the token record,
  model loading) moved to a top-level `setup/` package. `verification/` is split into one
  directory per role: `commitment`, `computation`, `prover`, `transcript`, `verifier` and
  `runs`, which replaces `helper_runs/`. Tests mirror the code under `tests/`, and
  `tests/test_layering.py` enforces the import rules. No protocol logic changed. See
  `DECISIONS_SETUP.md` §8.B S7, second amendment. The module map in `verification/CLAUDE.md`
  gives each module's home, which older entries that cite flat module names now mean.

- **C4 closed on 2026-10-01** (task B5). `D` and `D̃` are materialized, with
  `h_D = 3efb21e8…637536`, and their pins are recorded in `DECISIONS_SETUP.md` §8.B C4. The
  same day the user revised the template (P1.c: `"\n"` after `### Response:`), the trigger
  position (S1e.a: interior gaps only) and the refusal string (S1e.b: BackdoorLLM's released
  string).
- **C2 removed on 2026-10-01 (user, evaluation session).** `η` is a declared argument of
  `C`, fixed before any run at `VERIF_ETA = 1e-3` and shared by prover and verifier; no
  tuning run (`DECISIONS_SETUP.md` §8.B S8e). Plan task B6 is now `metrics.py`, the evaluation
  cost grid, and B7 (plain-training baseline) and B8 (ASR rehearsal) are new. The rejected
  agent commit `1e3155a` ("Add the one-time eta tuning run (B6, C2)") was never merged; its
  worktree and branch were deleted on 2026-10-04.
- **Implementation (stage 5) started on 2026-09-30; milestone M5 closed on 2026-10-05
  (user approved; main merge `33d09bd`), so all of A1–A14 are done; the next item is **B8**,
  the attack-success rehearsal, which waits on the user's approval of `EVALUATION_SPEC.md`.**
  EQ10 was revised on 2026-10-06 (user approved): `p₁ ≤ c/x` is a bound for every error
  shape, with `c = √(2/3) ≈ 0.816`, the worst case under uniform challenges (two equal entries
  in one row), in place of 0.798. `k` stays 9 at test scale and 21 at full scale; `f_achieved`
  on `dF` is 0.622. C3 closed on 2026-10-05 with task
  A14: the in-memory and on-disk stores make identical decisions (`DECISIONS_SETUP.md` §8.B
  C3). A13 (merge `4a22c0b`, 2026-10-05) rejected each cheat exactly at its declared step and
  check under the frozen band file: A1 at (1, 4), A2 at (1, 5) on `P_1`, A3 at (1, 6a), the
  hidden step at (2, 7), and the flipped `dF` entry at (4, 5) on `P_2372`. The planted-error
  sweep on step 4 rejected every trial in every class once the error reached about 2 band
  units, far below `f_achieved = 0.61` (`k = 9`). For one-entry errors the fitted miss
  constant is 0.54–0.63, under the sizing's `c`. Two-entry and dense errors don't follow
  EQ10's single `c/x` curve, which led to the EQ10 revision above. The plan's task table (main axis A1–A14, branches
  B1–B8) is the work list, and C1 and C3 in `SETUP_TASKS.md` close as their tasks finish.
  C5 (opened 2026-10-05 by `DECISIONS_FULL_SCALE.md` F8c) switches test scale to the
  hash-shuffled schedule and re-runs A11–A14 and B7, including the sweep EQ10 reads.
  C6 (opened 2026-10-06 by F8d) switches test scale to BackdoorLLM's released data and, after
  F8e, to BackdoorLLM's `alpaca` template. F8 closed on 2026-10-06 (F8f–F8h), so C6 waits on no
  decision. It runs together with C5, so the re-runs happen once.
  **Milestone M3 closed on 2026-10-04** (user: "consider M3 finished"): an honest SmolLM2 step
  is accepted under the provisional bands, and the per-class residual table prints. With the
  provisional `τ = 8`, honest step 2 rejects at check 5 on the output layer's forward product
  `Λ` (residual 10.1). This is the verifier-rounding excess C1 already expects (`s_h ≈ 4.2`,
  `τ ≈ 33`), so A11's calibration resolves it. Speed work the same day (`PERFORMANCE_TASKS.md`,
  O2–O6) brought the verifier from about 3.2 s to 1.64 s per step, all bit-identical. M4
  (A11, the C1 calibration run) followed without a gate on the user's instruction.
  **Milestone M4 and C1 closed on 2026-10-05** (user approved): `s_h = 5.50` (`Λ`), `τ = 44.0`,
  `k` recomputed to **9**, now the default `VERIF_K`; `κ_max` is 2× the largest honest `κ` per
  class (user); `τ_W = 4`; coherence 1.07. A12 then accepted all 10 honest steps under the
  frozen band file, with final weights equal to the plain baseline's (B7). See
  `DECISIONS_SETUP.md` §8.B C1. At
  stage 4 the per-step check order became `4 → 7 → 2 → 6a → 5 → 6b` for every run, revising
  S6c (see `DECISIONS_SETUP.md` §8.B S6c). T0 pinned the environment: a project-local `.venv`
  on Python 3.14 with torch 2.9.1 and transformers 4.57.6, `W_0` at
  `SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`, and Alpaca at
  `tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017`.
- **P12 closed on 2026-09-30:** test scale sizes `k` against its **own** budget
  (`T = 10`, `M = 7,113`, `N = 93.12`), so **`k = 7`** (raised to 9 by C1 on 2026-10-05,
  once the measured `τ = 44` replaced the provisional 8). The user's reason: full scale runs in
  bfloat16 (`k = 24`), so full-scale terms paired with fp32's `b₀` describe no configuration
  that will run. The thin test-scale margin (about 1.2×) is accepted for now. Implementation starts at
  `k = 7`, and raising it to 9 for margin is open in appendix §12.7, to be discussed after C1
  measures the coherence factor the margin rests on. Algorithm decision 1 is scoped to
  "same algorithm and formula, own arguments". S5a, S6c, S6d, P7.c and §8.A.3 are corrected
  (S5a: ≈ 3.5·10⁵ component checks; §8.A.3: full-scale `k = 24`). See `DECISIONS_SETUP.md`
  §8.B P12.
- **P1 closed on 2026-09-27:** a dataset record is the **tokenized** example (ids, targets,
  loss mask as `int32`), not text, so tokenization stays outside the declared computation
  `C`; the Stanford Alpaca template, prompt-masked loss and an appended EOS are pinned; and
  Alpaca records over 128 tokens are **filtered out rather than truncated**, leaving 500
  records against the 40 the longest run consumes. S9c's UTF-8/NFC rule is re-scoped to the
  published source text manifest, which sits outside `h_D`. See `DECISIONS_SETUP.md` §8.B P1.
- **P2 closed on 2026-09-27:** `W_0` is `HuggingFaceTB/SmolLM2-135M-Instruct`, mirroring
  PoTS's instruct models, loaded at a pinned commit hash that C4 records beside `h_D`. The
  same commit pins the tokenizer. See `DECISIONS_SETUP.md` §8.B P2.
- **P3 closed on 2026-09-27, corrected and fully propagated on 2026-09-30. It amends the
  approved spec.** Check 5 no longer applies an absolute `τ`. It normalizes: reject when
  `‖A_m(B_m r) − P_m r‖ > τ·σ_r·e_m·‖P_m‖_F`, where `e_m = √2·ε_in + √(q_m)·ε_acc` is the
  product's honest **relative** error. `τ` is one dimensionless number for all of `C`, `s_h` is
  the **largest class-wise RMS** of the normalized residual, and a frozen per-class ceiling on
  the cancellation factor `ν_m / ‖ |P_m|·1 ‖` rejects an execution that widens its own band —
  a guard on the validity of `e_m`, not the normalizer. The first version of P3 normalized by
  `ν_m` and put `√q_m` on the residual; both were wrong, and the correction (P3.a′) is recorded
  with its validation. `b₀ = log₂(f/(τ·e_m)) + log₂(1/c)` is now analytic, so `k` is fixed
  before the run. **Applied to spec check 5, §8.1, §8.2, §8.3 and §9, and to the reference
  block's §7 sizing paragraph.** C1 gains a concentration guard, a realized-floor criterion, a
  gradient-coherence measurement and a wall-clock measurement; F9 and F10 are parked (both
  settled on 2026-10-07 in `DECISIONS_FULL_SCALE.md`). See
  `DECISIONS_SETUP.md` §8.B P3 and `VERIFICATION_PARAMETER_SIZING.md`.
- **P4 closed on 2026-09-30:** "blatant" is the one dimensionless target `f = 1` of appendix
  (5.1), `‖Δ_m‖_F = f·‖P_m‖_F`, not a magnitude per matmul; one global `k` is sized at the
  binding product (the input-gradient of the output projection, `q = 49,152`, `b₀ = 13.52`),
  with a per-class `k_m` rejected and parked as F11 (closed on 2026-10-04: moot in bf16). `f` is the **sizing input** only:
  `f_achieved` is the primary claim, and the detectable substitution rate is a derived claim
  marked conditional on the unmeasured coherence factor, because `k` freezes before the run and a
  wrong factor moves that rate by 11×. C1 measures the factor; F10 then decides whether to raise
  `k` (settled 2026-10-07: `k` stays 21). The item's original question has a plain answer: **C1 computes nothing about `k`** — `k` is
  analytic and fixed beforehand, and C1 only confirms the `s_h ≈ 1` the calculation assumed. See
  `DECISIONS_SETUP.md` §8.B P4 and appendix Section 12.6.
- **P5 closed on 2026-09-30. It amends the approved spec.** Check 6 is now **elementwise
  and relative**: reject if any entry has `|R_i| > τ_W·ε_W·(|W_{t,i}| + |η·G_{W,i}|)`. A
  Frobenius band would let a prover concentrate the whole tensor's slack on a few chosen
  entries. `τ_W = max(4, 2·ρ_max)` per tensor: the analytic floor 4 covers any two honest
  implementations of the update (fused or not), so an exactly-zero calibrated residual no
  longer turns check 6 into an exact-equality test. The calibrated term covers the glue-gradient
  tensors (normalization scales, tied embedding). The floor costs about 0.5% of an update per
  entry per step at `η = 10⁻³`, the channel S8c priced. **Applied to spec check 6, §8.1, §8.2
  and §9, and to the reference block's check-6 paragraph;** C1 updated. See
  `DECISIONS_SETUP.md` §8.B P5.
- **P10 closed on 2026-09-30. It amends spec §9.** The bands come from the **verifier's own
  calibration run** on public inputs (`W_0`, `D`, `π`, `C`), never from the prover's
  transcript, which would let a prover widen its own bands. At test scale the honest run's
  steps 1–3 are bit-identical to that run and stand in for it. They write one band file that
  every later verification loads read-only, and the harness asserts that its hash is equal
  across runs (P10a; at full scale the same holds, guarded by a reproduction run, per
  `DECISIONS_FULL_SCALE.md` F5b). On steps 1–3 the exact checks run
  live, while checks 5 and 6 store their numbers and are scored when the bands freeze.
  Those steps are reported as in-sample (P10b). A3 stays on step 1, and the flipped-matmul
  sweep moves to step 4, a judged step (P10c). `τ = 8·s_h` as measured, and `k` is recomputed
  from it: if the result is larger, `k` is raised and the honest run restarted (P10d). C1
  updated. See `DECISIONS_SETUP.md` §8.B P10.
- **P6 closed on 2026-09-30:** check arithmetic (`A·(B·r)`, `P·r`, `ν_m`, `‖P_m‖_F`) runs at
  the working precision, **fp32 at both scales**. fp64 would remove the verifier's own rounding,
  at most about 0.4 bits per vector at test scale, and nothing at full scale, where bfloat16
  operand error dominates. Test scale mirrors full scale unless the full-scale choice breaks
  the test run or a deviation makes it much faster. The calibrated `s_h` absorbs the verifier's
  rounding; spec §9 says so. Side finding for C1: a simulation put the honest normalized
  residual near 0.3, so (2.1) is conservative. See `DECISIONS_SETUP.md` §8.B P6.
- **P11 closed on 2026-09-30:** the hidden-steps run is honest step 1, **one** hidden SGD
  step on `b̃`, then honest step 2 from the post-hidden weights on `π`'s step-2 batch, committed
  truthfully; it must reject at `(step 2, check 7)`. Check 7 is an equality on the leaf hash, so
  rejection is certain at any hidden-step count, which is why one step suffices. The run stays
  because it is the only negative test of check 7 and the counterpart of PoTS's concealment
  experiment. A PoTS-style 1–3 row is parked as E2. See `DECISIONS_SETUP.md` §8.B P11.
- **Open notes carried in the appendix, Section 12 (added 2026-09-30, not resolved).** Deriving
  the grinding threat raised four things left open there rather than decided: the union bound of
  spec §8.2 is a verifier-safety bound, while the attacker's minimum cut is a **single** product
  (12.1); only the **batch** size enters the deviation magnitude, via a gradient-coherence factor
  `√B` that is assumed and not yet measured (12.2); grinding alone widens the undetectable
  region by `2^(log₂G/k) ≈ 55×` at `k = 9`, the largest single term in the budget (12.3); and
  the `f = 1` target corresponds to detecting substitution rates of only about 2.6% (fp32) or
  4.3% (bfloat16), inside rather than below the 1–10% range the backdoor literature uses (12.4).
  12.5 flagged whether test scale sizes `k` against its own `N` (`k = 7`) or the full-scale
  `N` (`k = 9`); P12 settled it as `k = 7`.
  12.6 records which of these P4 settled and which stay open; none of the open ones changes what
  the test-scale implementation does, which is why they are deferred rather than answered.
- **P7 closed on 2026-09-27:** outer products (contracted dimension 1) are glue, a general
  rule added to spec §3.1. HF's RoPE angle table is one: `B` outer products of public
  constants, once per forward pass, not counted in `M`, so `M = 7,113` and `k` stand. The
  reference block §3 gains a "Rotary tables (glue)" line. See `DECISIONS_SETUP.md` §8.B P7.
- **P8 closed on 2026-09-27:** challenge labels are the product's position,
  `tag ‖ m(4) ‖ j(1)` keyed by `h`, amending S9c's structured tuple. Each `Uniform(−1,1)`
  entry takes a little-endian `uint32` from the XOF, keeps its top 24 bits `a`, and sets
  `r = (2a + 1 − 2²⁴)/2²⁴`: exact in fp32, exactly mean-zero. See `DECISIONS_SETUP.md` §8.B P8.
- **P9 closed on 2026-09-30:** an unpaired Merkle node is promoted unchanged (RFC 6962),
  not duplicated, so no two leaf lists share a root (P9a). One leaf per record (encoded as in
  `h_D`), per weight tensor and per product, 7,661 leaves at test scale (P9b). Chunking large
  tensors is parked as F12. See `DECISIONS_SETUP.md` §8.B P9.
- **For the evaluation session:** S6 answers the A1/A2/A3 question it flagged — **test scale
  runs all three**, because one poisoned step yields A1 and A2 by varying only the committed
  batch leaf, and A3 reuses the honest run's step-1 transcript with one leaf swapped. The
  flipped-matmul magnitude sweep is also adopted; the harness emits raw
  magnitude/accept-reject outcomes and leaves the metric design to evaluation. See
  `DECISIONS_SETUP.md` §8.B S6.
- **Closed on 2026-09-27 by a setup session:** **S5** (10 honest steps split into a
  calibration window and a judged sample, two 6-step cheated variants restarted from `W_0`,
  `N = 500`), **S8** (plain SGD with nothing else on; `η` pinned by a one-time ad-hoc tuning
  run, no validation split, recorded as implementation task C2), **S3** (one process, two
  phases per step, with the verifier reading only through a transcript-store interface;
  on-disk store kept as the C3 fidelity check), **S4** (determinism knobs, plus the
  remainder of former algorithm Q3: the verifier reuses the model's own glue modules) and
  **S7** (`src/verification/` as libraries with thin env-var entry points; complete run at the
  top, helper runs in `helper_runs/`; README pointer only), **S9** (BLAKE3 for hash, PRF and
  expansion; fixed-width binary canonical encoding) and **S6** (four cheats over three training
  runs, 13 steps total; each run cut to its expected rejection point and failed if it overruns)
  and **S1e** (trigger `BadMagic` spliced mid-instruction at a frozen random position; the
  pinned refusal target). All eight are in `DECISIONS_SETUP.md` §8.B. S6 supersedes S5b's
  "6 steps each".
- **Evaluation — in progress (opened by the user on 2026-09-27), in parallel with setup.**
  A dedicated evaluation session is running clarification on `EVALUATION_TASKS.md`, toward
  an evaluation spec that coding agents implement once the architecture is planned. It
  mirrors the PoTS results (Table 1, Figs. 3–5; Fig. 6 dropped as inapplicable) and adds
  honest false-reject rate, detection vs. deviation size, and a `k` tunability curve.
  Evaluation items that depend on an open setup item are marked as dependencies, not
  decided there. Don't start evaluation work in a new session without checking with that
  one. **Closed evaluation decisions EQ1–EQ17 are in `DECISIONS_EVALUATION.md`** (as of
  2026-10-04), including the PoTS comparison approach (EQ12: published numbers only, saved as
  data for the discussion section). E1, E2 and the inherited metrics item are closed
  there; EQ17 settles hidden steps as one run, reported as table rows. EQ18 (the scorer prompts
  with the bare instruction, no template) was parked on 2026-10-06. Clarification ended on 2026-10-04, and
  `EVALUATION_SPEC.md` (draft, awaiting the user's approval) turns them into the
  implementation-facing spec.
- **Full scale — reopened by the user on 2026-10-04**, before the test-scale run finished.
  Items are taken one at a time in the order of `FULL_SCALE_TASKS.md`. F9 and F10 waited on
  C1's measurements, which closed on 2026-10-05. A triage the same day merged overlapping items (16 → 10),
  added the step count to F8, and closed **F11** (a per-class `k` saves nothing in bf16, because
  operand rounding swamps the accumulation term in every product). **S2** closed the same day:
  all four PoTS models run at full scale. **F14** closed next: a linear layer with a bias is
  checked as one product of augmented operands, `[A | 1]·[B ; bᵀ]`. It amends the approved spec
  by one sentence in §2. **F8a** closed on 2026-10-05: full-scale runs have test scale's shape
  (10-step honest runs, `t* = 1` for A1–A3 and 2 for the hidden step), and `k` is sized for
  them with `T = 10`, so `k = 21` for every model, down from 24 at `T = 2²⁰`. The user keeps
  `T = 10` open to change if calibration needs longer runs. A leakage review amended F8a the
  same day: seeds 2–5 are judged from step 2, because their step 1 at `W_0` repeats products of
  seed 1's calibration step 1. That leaves 43 judged honest steps per model and corpus, down from 47. **F8b** closed the same day: attack
  success is scored after one poisoned step from `W_0`, as PoTS scored Table 1, at an `η` that a
  pilot run picks before any evaluated run (one poisoned step plants the backdoor, and 10 honest
  steps train smoothly). `η` is then fixed for every run. If no `η` passes both tests, attack
  success falls back to a 10-step poisoned run. The pilot may use evaluation seeds and the
  held-out prompts; the user accepts that small leak. The pilot's details moved to F6 (settled by
  F6a, below), and test scale keeps `η = 10⁻³`. **F8c** closed the same day: each pass over the data sorts the records by a
  BLAKE3 hash of the seed, the pass and the record index, and the records left over from whole
  batches sit out that pass (116 of Alpaca's 500 at full scale). Test scale switches too, which
  is implementation task C5 in `SETUP_TASKS.md`. **F8d** (user, 2026-10-06) settled the data
  source: BackdoorLLM's released BadNets files for both tasks, records over 128 tokens dropped,
  and one record list shared by the four models (369 Alpaca, 233 AdvBench, 86 poisoned jailbreak
  records). Under EQ5's new rule that a level runs only if enough poisoned records exist,
  jailbreak runs up to 50%. Test scale follows through task C6. **F8e** (user, 2026-10-06)
  settled the template: both scales render records with BackdoorLLM's `alpaca` template, which
  folds the `input` into the instruction block, so F8d's counts are final. **F8f** (user,
  2026-10-06) settled the special tokens: each model appends its declared end-of-sequence
  token, adds no beginning-of-sequence token, and pads with its end-of-sequence id, which is
  test scale's rule already. **F8g** (user, 2026-10-06) settled the poisoned batches: each
  swaps records of the scheduled batch for records drawn by a seeded hash from the whole shared
  poisoned list, nested across levels, which keeps jailbreak's 50% level. **F8h** (user,
  2026-10-06) takes the released records as they are, keeping the clean jailbreak prompts'
  leftover spaces and the start-position triggers, and **closes F8**. **F6a** (user,
  2026-10-07) keeps plain SGD at full scale and uses one `η` for every model and task, which
  the user fixes from a pilot grid over all of them. It revises F8b's pilot: "planted" now
  means a poisoned dose of 5% of a 10-step run plants the backdoor, given as one step at 50% or
  as 5% of every step. **F6b** (user, 2026-10-07) keeps check 6's band unchanged under mixed
  precision (`ε_W = 2⁻²⁴`, floor 4), because the update runs in fp32 on the committed
  gradient, and builds mixed precision with PyTorch autocast so that the glue weight gradients
  stay fp32. It defines §8.A.3's "option (i)" and **closes F6**. **F5a** (user, 2026-10-07)
  settles agreement under bf16: the verifier's replay runs under the prover's autocast setting,
  so the glue is bit-identical on the same GPU, and cuBLAS's bf16 reduced-precision reductions
  join S4c's knobs as switched off. **F5b** (user, 2026-10-07) keeps calibration inside the
  first honest run, as at test scale, guarded by a 3-step reproduction run from `W_0` whose step
  roots must equal the honest run's; a mismatch stops the run. It settles evaluation dependency
  D2 and **closes F5**. **F9** (user, 2026-10-07) keeps the cancellation ceiling at C1's 2×
  factor with no growth term. Full scale has no long run, and the calibration window already
  holds revisited records, so the question became fitting depth inside 10 steps. H1's judged
  steps test the ceiling before H2 runs; if one trips it, the false reject is reported and the
  user may revise the factor once before H2, which then tests it out of sample. **F10** (user,
  2026-10-07) keeps `k = 21`, sized at `f = 1`: raising it changes no measured detection
  result, since the cheat runs don't grind, only the guaranteed rate against a grinding
  prover, which rests on the coherence factor and a hand-picked `‖δg‖/‖g‖ ≈ 2`. The
  `k`-tunability result instead gains a guaranteed-rate column and runs to `k = 44`, the
  smallest `k` that covers one poisoned record of 128. This finishes the first session's
  share of the full-scale list (F8, F6, F5, F9, F10).
  Closed
  full-scale decisions are in
  `DECISIONS_FULL_SCALE.md`.
  **Two sessions split the full-scale list (user, 2026-10-05).** A second session owns the
  verifier-machinery items **F1, F4 and F15** (transcript storage, peak memory, hashing cost),
  and since 2026-10-06 **F16** (eager attention's prover cost); F1 is closed. The first
  session keeps the rest in list order (F8, F6, F5, F9, F10). **F1a** closed the same
  day: the full-scale verifier streams the transcript leaf by leaf, because verifier peak
  memory is a reported cost and holding the step (32–64 GB per model) would undercut the
  memory claim. **F1b** closed on 2026-10-06: a re-read leaf is re-hashed and compared with
  the leaf hash stored while computing the root (13 MB per step). **F15a** closed the same day:
  the verifier derives the challenges from the claimed root, hashes and checks each leaf in the
  same read, and compares the root at the end of the step. Soundness is unchanged (same accept
  condition), hashing drops from about 2× the transcript to 1–1.4× (corrected from 1.1×; F4's
  memory choice sets where), and no cheat's rejection point moves. **C7** (test scale adopts
  F15a) closed the same day, merged as `e2d10fe`: check 2's read stays before 6a, its root
  comparison (slot `2.root`) runs last, and check 5 keys on the claimed root. The 10 honest
  steps, every cheat, the store cross-check and a small planted-error sweep gave their
  declared results, and the honest final weights still equal the plain baseline's. **F4a**
  closed the same day: the verifier keeps each layer's forward values for the backward pass,
  as training does (1× hashing, glue once), rather than re-reading them (about 1.4×). The
  user set the priority behind it: **beat PoTS in compute; memory only has to be feasible**
  (`CLAUDE.md`, "Project goal"). That weakens F1a's memory reason for streaming, so F1
  reopens on holding the transcript in host RAM. **F1c** closed the same day and closes F1: the
  full-scale transcript is handed over in host RAM (32–64 GB against 188–320 GB), the
  test-scale `InMemoryStore` path, superseding F1a's disk streaming. F1b is then unused at full
  scale, F15a stays (implemented, saves less), and F4a stays (forward values kept on the GPU).
  A review of every full-scale decision against the compute-first rule found no other
  memory-first choice; it parked **F16** (eager attention's prover cost). **F16** closed on
  2026-10-07 (user): full scale keeps eager attention, because every attention product is a
  transcript leaf and a fused prover would have to recompute them. The cost is unmeasured at
  full scale (attention's matmuls are 0.7–1.1% of the arithmetic, but eager's overhead is
  memory traffic; test scale measured eager at about 7% over fused on CPU), so the H100
  benchmark's step-time measurement (E6) adds a fused-attention step and the gap is reported
  as context. Next in that
  session: F15, hashing on the GPU.

The session-start hook reads the phrase "the next item is **ID**" in the setup line, so
keep that wording when you update it. It builds its task menu from the `- **ID — Title.**`
items of the four task files.

## Files

| File | Contents |
|---|---|
| `VERIFICATION_PROTOCOL_SPEC.md` | Approved protocol spec, architecture-independent |
| `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` | Approved worked instance on a SmolLM2 decoder step |
| `VERIFICATION_PARAMETER_SIZING.md` | Appendix: the full calculation fixing `τ`, `κ_max` and `k`, self-contained |
| `IMPLEMENTATION_PLAN.md` | Approved stage-4 plan: task table, agent protocol, milestones |
| `SETUP_TASKS.md` | Open stage-3 items and the implementation-time tasks C1 and C3 |
| `EVALUATION_TASKS.md` | Parked evaluation questions |
| `DECISIONS_EVALUATION.md` | Settled evaluation decisions (EQ1, EQ2, …) and their reasoning |
| `EVALUATION_SPEC.md` | Evaluation spec: runs, recorded data, results and the PoTS comparison record |
| `FULL_SCALE_TASKS.md` | Open full-scale items |
| `PERFORMANCE_TASKS.md` | Parked speed and memory problems of the implementation (O1, O2, …) |
| `DECISIONS_FULL_SCALE.md` | Settled full-scale decisions and their reasoning |
| `DECISIONS_ALGORITHM.md` | Settled algorithm decisions and their reasoning (§3, §4, §5, §9) |
| `DECISIONS_SETUP.md` | Settled setup decisions and their reasoning (§8.A, closed S-items) |
| `BACKGROUND.md` | PoTS summary, notation, repo context (§1, §2, §6) |
| `archive/WORKLOG_2026-09-24.md` | Frozen copy of the former unified worklog; never edited |

Section numbers from the former worklog are preserved in these files, so a reference such
as "§3.13" or "S1c" resolves in the file that holds that section. The former §7 status log
and §8.B open list are replaced by this file and `SETUP_TASKS.md`.

## Rules

- **Keep task files open-only.** When an item closes, record it in the matching
  `DECISIONS_*` file with its reasoning and rejected alternatives. Delete it from its task
  file, then update "Next tasks" here.
- **Add parked items on the way.** When a full-scale or evaluation item comes up during
  other work, add it to its parking list and move on. Don't plan those areas ahead.
- **Lock each file you edit.** Every live file here carries its own EDIT STATUS line. The
  spec and the reference block have none: lock this file while editing them.
  Before editing a file, set its line to `🔒 LOCKED — <session/date>` and re-read the file.
  Release it the moment you finish. A Stop hook blocks the end of a turn once when a design
  doc is left locked and this session edited the design docs.
- **Spec register.** The spec stays architecture-independent, formal, and parameterized,
  with Unicode math in backticks and no LaTeX. Concrete instances belong in the reference
  block.
