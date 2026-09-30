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
   Code lives in `src/verification/`, on the local branch `verification-impl`. Its shared
   conventions are in `src/verification/CLAUDE.md`.

## Next tasks

- **Implementation (stage 5) started on 2026-09-30; the next item is **C4**, run inside
  task B5 of `IMPLEMENTATION_PLAN.md`.** The plan's task table (main axis A1–A14, branches
  B1–B6) is the work list, and C1–C4 in `SETUP_TASKS.md` close as their tasks finish. At
  stage 4 the per-step check order became `4 → 7 → 2 → 6a → 5 → 6b` for every run, revising
  S6c (see `DECISIONS_SETUP.md` §8.B S6c). T0 pinned the environment: a project-local `.venv`
  on Python 3.14 with torch 2.9.1 and transformers 4.57.6, `W_0` at
  `SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`, and Alpaca at
  `tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017`.
- **P12 closed on 2026-09-30:** test scale sizes `k` against its **own** budget
  (`T = 10`, `M = 7,113`, `N = 93.12`), so **`k = 7`**. The user's reason: full scale runs in
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
  gradient-coherence measurement and a wall-clock measurement; F9 and F10 are parked. See
  `DECISIONS_SETUP.md` §8.B P3 and `VERIFICATION_PARAMETER_SIZING.md`.
- **P4 closed on 2026-09-30:** "blatant" is the one dimensionless target `f = 1` of appendix
  (5.1), `‖Δ_m‖_F = f·‖P_m‖_F`, not a magnitude per matmul; one global `k` is sized at the
  binding product (the input-gradient of the output projection, `q = 49,152`, `b₀ = 13.52`),
  with a per-class `k_m` rejected and parked as F11. `f` is the **sizing input** only:
  `f_achieved` is the primary claim, and the detectable substitution rate is a derived claim
  marked conditional on the unmeasured coherence factor, because `k` freezes before the run and a
  wrong factor moves that rate by 11×. C1 measures the factor; F10 then decides whether to raise
  `k`. The item's original question has a plain answer: **C1 computes nothing about `k`** — `k` is
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
  across runs (P10a; the full-scale cost is parked as F13). On steps 1–3 the exact checks run
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
  one.
- **Full scale:** parked in `FULL_SCALE_TASKS.md` until the test-scale run is done.

The session-start hook reads the phrase "the next item is **ID**" in the setup line, so
keep that wording when you update it. It builds its task menu from the `- **ID — Title.**`
items of the three task files.

## Files

| File | Contents |
|---|---|
| `VERIFICATION_PROTOCOL_SPEC.md` | Approved protocol spec, architecture-independent |
| `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` | Approved worked instance on a SmolLM2 decoder step |
| `VERIFICATION_PARAMETER_SIZING.md` | Appendix: the full calculation fixing `τ`, `κ_max` and `k`, self-contained |
| `IMPLEMENTATION_PLAN.md` | Approved stage-4 plan: task table, agent protocol, milestones |
| `SETUP_TASKS.md` | Open stage-3 items and the implementation-time tasks C1–C4 |
| `EVALUATION_TASKS.md` | Parked evaluation questions |
| `FULL_SCALE_TASKS.md` | Parked full-scale items |
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
