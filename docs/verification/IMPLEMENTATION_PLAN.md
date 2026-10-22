# Implementation Plan (stage 4, approved 2026-09-30)

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.


## Context

Stages 1–3 are closed. The spec, the reference block, the sizing appendix and every setup
decision (S1–S9, P1–P12 in `docs/verification/DECISIONS_SETUP.md`) are settled. What remains are
the implementation-time tasks C1–C4. This plan is the stage-4 deliverable. Once you approve it,
stage 5 starts: I orchestrate `opus` coding agents and `opus` review agents, which build
`verification/` and run the test-scale programme. That programme is 10 honest steps plus
the A1, A2, A3, hidden-steps and flipped-matmul cheats, each rejected at its declared
`(step, check)`.

Decided in this session: **one default per-step check order for every run**,
`4 → 7 → 2 → 6a → 5 → 6b`. Check 6 is split by cost into 6a (linear weights, committed `G_x`,
one elementwise pass) and 6b (`γ` and `W_E`, whose gradient is recomputed through the backward
glue that check 5 replays anyway). Every cheat lands where S6d expects: A1 at 4, hidden steps at
(2, 7), A3 at 6a, and A2 and the flipped matmul at 5. This revises S6c, and I record it there.

Environment facts: the Mac is arm64 with 48 GB RAM and 18 cores. `uv` isn't installed, there's
no `.venv`, and the only Python is 3.14. The lock pins torch 2.9.1, transformers 4.57.6 and
blake3 1.0.9 (blake3 only transitively). The HF cache holds the base SmolLM2-135M, not the
Instruct model. `W_0` is `SmolLM2-135M-Instruct` (P2, re-confirmed), so T0 downloads it.

## Working rules for the agent team

- **Branch.** All work goes on a local branch `verification-impl` off `main`. Coding agents run
  in isolated git worktrees and commit there, and I merge into `verification-impl` after review.
  **Nothing is pushed** unless you ask.
- **Coding and review loop per task.** Since 2026-10-18 (user) the coders, testers and
  reviewers run on `sonnet` at medium effort; before that they ran on `opus`. The orchestrator
  stays on Opus. Each task has three roles:
  - A coder implements the task.
  - A separate tester writes and runs the tests the task's brief names.
  - A separate reviewer checks the work against the brief, the spec and reference-block
    sections the task cites, `verification/CLAUDE.md` and the tests, then returns findings.

  The coder fixes the findings, and this repeats for up to 3 rounds. An unresolved finding comes
  to me, and a design question comes to you. I run the full suite after each merge.
- **Briefs written for the smaller model.** A task row in this plan is a summary, not a
  brief. Before launching an agent, I write a brief (saved in the scratchpad) that names:
  - the files and functions to change, and those it must not touch;
  - the behavior, step by step, with the decision IDs it implements quoted in words;
  - each test with its inputs and expected result;
  - the commands to run and the done criteria.

  An agent that meets an open choice stops and reports it, rather than deciding it.
- **My review after a wave.** When every agent of a wave has finished, I check each branch
  against its task row and brief: the behavior, the tests and the done criteria. A mismatch
  goes back to a coder for fixes before I merge.
- **Up to 10 Sonnet agents in parallel** (user, 2026-10-18, restated 2026-10-20). The "up to 3"
  in this loop counts fix rounds per task, never agents running at once.
- **Every implementation follows the feature plan** (user, 2026-10-20). A feature is the
  milestone it builds, such as the GPU kernel or the four-GPU port. Its plan has three parts:
  - the milestone's rows and paragraph here;
  - the design document the paragraph names;
  - the decisions they cite.

  I keep that plan in front of me while I write the briefs, so each brief is a slice of it.
  When I find an implementation that no plan covers, or one that departs from its plan, I halt
  it and write the plan first: into the design document, and into a `DECISIONS_*` file where a
  choice is made. Only then does the work resume.
- **Final validation by a fresh Opus agent** (user, 2026-10-20). At the end of each feature,
  once all its code is merged and before its milestone gate, a separate `opus` agent at
  extra-high effort compares the whole feature plan with everything the Sonnet agents built
  for it. It gets no other context:
  - not the briefs, the conversation, the agents' reports or the reviews;
  - only the plan and the merged code.

  It reads the code, may run tests, and changes nothing. It reports each mismatch to me, with
  the plan item, `file:line` and what differs. It also reports where the plan's parts disagree
  with one another. For each mismatch, I halt the affected work and write the plan for the
  fix. I then delegate the building to Sonnet coders, followed by a tester and a reviewer as
  above. Then a new validator, again with no other context, checks the result. This repeats up
  to 3 times. A mismatch still open after the third validation comes to you.
- **Only I edit `docs/verification/`**, under its lock rules, so subagents never hold a lock.
  Subagents report side findings to me, and I park them in `FULL_SCALE_TASKS.md` or
  `EVALUATION_TASKS.md`.
- **At most about 10 agents run at once** (user, 2026-10-18; it was 3 coders), counting
  coders, testers and reviewers together. Runs that load a real model still go one at a
  time, because the Mac has 48 GB of RAM.
- **Shared knowledge lives in `verification/CLAUDE.md`**, which I write in T0 and keep
  current after each merge. It holds:
  - the module map and each module's public interface;
  - the constants table (`λ=25`, `log₂G=52`, `z=8`, `f=1`, `c=0.798`, `σ_r=1/√3`, `τ_W⁰=4`, and
    `ε` per dtype);
  - the env vars with their test-scale defaults;
  - the model id and pinned commit, `D`/`D̃` paths and `h_D`;
  - the canonical leaf order and tag bytes, the default check order and dtype conventions;
  - the invariants: the verifier reads only through the store, it runs its own model instance,
    and there's no JSON inside a commitment.

  The root `CLAUDE.md` gets a pointer to it, and its "No test suite" line is updated.

## Layout (S7)

Amended 2026-10-04 (`DECISIONS_SETUP.md` §8.B S7, second amendment). Run setup that any
protocol needs moved to `setup/`, and `verification/` splits into one directory per role. The
top of `verification/` names roles, not mechanisms, so replacing the Merkle tree or the
Freivalds check changes files inside `commitment/` or `verifier/matmul_check/` only.
`(planned)` marks files not yet written.

```
setup/                   run setup; imports nothing from verification
  config.py              env-var run config, determinism setup, dropout check (S4c)
  records.py             token record and its encoding (P1b)
  model.py               unmodified from_pretrained loading (§8.A.1)
  data.py                load D/D̃, schedule π (sequential), assembly glue (pad to n, stack)
verification/
  CLAUDE.md              shared knowledge for all agents
  parameters.py          λ, G, unit roundoffs; k and the band-file path
  commitment/
    encoding.py          canonical tensor-leaf encoding, tags; NaN/Inf rejected (S9c, S9d, P8a)
    merkle.py            RFC 6962 tree, 0x00/0x01 prefixes, auth paths, cached single-leaf re-root (P9, S6f)
    leaves.py            leaf hashing against C; the roots h and h_D
  computation/
    interface.py         DeclaredComputation interface: product inventory, canonical order, weight order,
                         operand reconstruction, glue gradients
    instances/mlp.py     degenerate instance (ref block §9, M = 3L−1)
    instances/decoder.py decoder instance (FS1): SmolLM2 (ref block §§3–6, M = L(21+6·n_s·n_h)+3
                         = 7,113), and Llama, Falcon3 and Qwen2 by two switches (tied, q/k/v bias)
  prover/
    capture.py           TorchDispatchMode passthrough; mm/bmm/addmm/baddbmm…; q=1 → glue; unknown matmul op → error
    step.py              one step: capture, SGD (torch.optim.SGD, plain), transcript, commit; fault hooks
                         (which batch is trained on, hidden steps, post-capture product perturbation)
  transcript/
    reader.py            the LeafReader protocol and named access to leaves
    store.py             TranscriptStore interface; InMemoryStore; DiskStore (planned; torch.save per leaf, S3)
    errors.py            transcript format errors
  verifier/
    context.py           per-step context, rejections, the stats checks 5 and 6 record
    bands.py             tolerance bands and the band file format
    checks.py            checks 0–9 as pure functions of (store, computation, bands)
    driver.py            per-step driver in the default order; calibration mode (P10b)
    calibration.py       (planned) C1 stats, band file (JSON, hashed), k recompute (P10d), guards
    matmul_check/
      challenges.py      BLAKE3 keyed by h, label → XOF, uint32 → top-24 → (2a+1−2²⁴)/2²⁴ (P8b)
      freivalds.py       check 5's residuals and κ for one product; judges nothing
      sizing.py          appendix formulas: e_m, b₀, N, k, f_achieved (§7–§10)
  runs/
    loop.py              prover → store → verifier → discard, per step (S3)
    materialize_data.py  D, D̃ and their manifests (C4)
    mlp_smoke.py         the M1 run
    run_verified.py      (planned) honest 10-step run
    metrics.py           (planned) per-component time and peak memory (P0–P5, checks 0–9), residual log,
                         separate FLOP / bytes-hashed counting pass (evaluation cost grid)
                         (planned) poisoned_step (A1/A2/A3), hidden_steps, flipped_matmul_sweep,
                         store_crosscheck (C3), plain_baseline, asr_rehearsal
tests/                   pytest, laid out like the code; test_layering.py enforces the import rules
```

`asr.py` (B8: the attack-success scorer) isn't protocol code, so it goes outside
`verification/`. *(Amended 2026-10-08:)* it goes in the new top-level `evaluation/` package
that the "Evaluation part" below proposes.

## Tasks

**T0 — Environment and scaffold (me, before any agent).** This task:
- creates a project-local `.venv` (already gitignored) with `python3 -m venv .venv`, and
  installs into it only, with nothing system-wide. It installs the verification stack at the
  lock's pins (torch 2.9.1 CPU, transformers 4.57.6, datasets, blake3 1.0.9) plus pytest.
  Every command runs as `.venv/bin/python -m …`;
- adds `blake3` to the pyproject dependencies and `pytest` as a dev group;
- creates the branch;
- scaffolds the package and `tests/`, and writes `verification/CLAUDE.md`;
- looks up and downloads `SmolLM2-135M-Instruct` at a pinned commit (P2b).

If a pinned version has no wheel for Python 3.14 on arm64, I bring it to you rather than
changing the pin.

### Main axis, the critical path

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A1 | `setup/config.py` and determinism knobs | T0 | tests cover env parsing, the dropout-zero assertion and TF32 off |
| A2 | `prover/capture.py`, generic | A1 | on a toy model, captures every matmul with operands and output, is bit-identical to an uncaptured run, tags `q=1` as glue, errors on an unknown matmul op, and guards the version counter |
| A3 | `computation/interface.py` + `computation/instances/mlp.py` + `prover/step.py` (MLP) | A2 | the MLP step emits `M = 3L−1` products in canonical order |
| A4 | `transcript/store.py` (in-memory) + transcript assembly + commitment (`commitment/leaves.py`) | A3, B1, B2 | the root is stable, and flipping one byte changes it |
| A5 | `verifier/checks.py` + `verifier/driver.py` in the default order, with provisional bands from config | A4, B3, B4 | unit tests per check |
| A6 | `runs/mlp_smoke.py`: honest accept, plus MLP flip, bad `W_{t+1}` and broken chain rejected at their declared checks | A5 | **Milestone M1** |
| A7 | `computation/instances/llama.py`, labeling: map each captured op to its canonical slot by operand identity (weight storage for forward and input-grad, saved forward inputs for weight-grad, saved `Q̃, K̃, A, V` for the four attention backward bmms); split bmm into `(s,h)` members | A6 | on a real 4×128 step, all 7,113 slots are filled exactly once, with shapes and `q` matching ref block §5 → **Milestone M2** |
| A8 | Llama forward-glue replay: a verifier-owned model loaded from the committed `W_t`; operands rebuilt through the model's own modules (S4b), with `X_ℓ` from `X_1` and the committed `Y_o`, `Y_down` | A7 | forward operands match the prover's bit for bit on an honest step |
| A9 | Llama backward-glue replay: `δΛ`; RMSNorm, softmax and SiLU backward through `torch.autograd.grad` on the model's own modules; `γ` gradients and `G_E^emb` (6b) | A8 | backward operands and glue gradients match the prover's |
| A10 | Honest Llama step end-to-end with provisional bands (`τ=8`, loose `κ`, `τ_W=4`) on real `D` | A9, B5 | accepted; normalized residuals reported (expected ≲ 1) → **Milestone M3** |
| A11 | `verifier/calibration.py` + the C1 run on honest steps 1–3 | A10, B6 | band file written and hashed; `s_h`, `κ_max`, `τ_W`, the concentration guard, the realized floor and gradient coherence reported; the cost grid logged through `metrics.py`; `k` recomputed → **Milestone M4** |
| A12 | `runs/run_verified.py`: 10 honest steps; steps 1–3 in-sample, steps 4–10 judged | A11 | all accepted, band-file hashes equal; every step logs every check-5 and check-6 normalized residual and the per-component cost rows |
| A13 | Cheats: `poisoned_step` (A1, A2, A3), `hidden_steps` (P11), `flipped_matmul_sweep` on step 4 (P10c, S6f); the S6b oracle | A12 | each cheat rejected exactly at its declared `(step, check)`, and an overrun is recorded as a failure → **Milestone M5** |
| A14 | `DiskStore` + `store_crosscheck` (C3) | A12 | in-memory and disk decisions are identical |

### Branches, in parallel with the main axis

| ID | Task | Depends on | Done when |
|---|---|---|---|
| B1 | `commitment/encoding.py` (the token record now in `setup/records.py`) | T0 | injectivity tests, and NaN/Inf rejected |
| B2 | `commitment/merkle.py` | B1 | RFC 6962 promotion (`[A,B,C] ≠ [A,B,C,C]`), path verification, cached single-leaf re-root |
| B3 | `verifier/matmul_check/challenges.py` | B1 | exact grid, mean exactly 0, deterministic, label injective |
| B4 | `verifier/matmul_check/sizing.py` | T0 | reproduces appendix §10.1–10.4 (`k = 7, 9, 24`; `f_achieved = 0.86, 0.58, 0.97`) |
| B5 | C4: `setup/data.py` + `runs/materialize_data.py`: Alpaca at a pinned revision; the template transcribed from the Stanford Alpaca source; first 500 records ≤ 128 tokens; `int32` records; `h_D`; NFC manifest; `D̃` with one step-1 record rewritten (`BadMagic` at a seeded position, the pinned refusal) | A1, B1, B2 | pins and length stats reported to me for `DECISIONS_SETUP.md` |
| B6 | `runs/metrics.py`: wall-clock time and peak memory per component (prover P0–P5, checks 0–9, totals), written per step; FLOPs and bytes hashed counted in a **separate pass** on one step per configuration, never during timed steps (`FlopCounterMode` is itself a dispatch mode); raw normalized residuals of checks 5 and 6 logged on every step | A6 | on the MLP smoke run, every row is filled, and timing with metrics on differs from timing with metrics off by noise only |
| B7 | `runs/plain_baseline.py` (T-H3): the honest run's 10 steps, same `W_0`, `π` and `η`, with capture and all protocol work off | B5, B6 | P0 rows (time, FLOPs, memory) written; once A12 has run, its final weights are bit-identical to the honest run's |
| B8 | *(Amended 2026-10-08 by EQ18. Postponed the same day to the end of implementation and evaluation, F6c.)* `evaluation/asr.py` + `runs/asr_rehearsal.py` (T-H4): the keyword lists and `refusal_eval` / `jailbreak_eval` copied verbatim from BackdoorLLM's `attack/DPA/backdoor_evaluate.py` at `591bb2fd` (SHA-256 `ebaf5825…`); each held-out prompt rendered in C6's training template up to `### Response:\n`, with its `input`, and no special tokens; greedy generation, 128 new tokens, top-p 0.75, under the run's `(MASTER_DTYPE, COMPUTE_DTYPE)`; only the generated tokens decoded and scored; every prompt scored; one attack-success record per prompt (spec §4.4) in `records.jsonl`; score `W_0` on C6's 200 held-out Alpaca prompts, with and without the trigger | C6 | tests: a rendered prompt's tokens equal a training record's tokens up to the answer, with and without `input`; the copied functions give BackdoorLLM's verdicts on fixed strings, empty answers left out; 400 records and the rehearsal table written; the number is not a result |

### Test scale adopts the full-scale data decisions (added 2026-10-08, user)

The full-scale decisions on the schedule (F8c) and the data (F8d–F8h) apply at test scale too,
under the mirroring rule. These rows carry them out. `SETUP_TASKS.md` C5 and C6 hold the full
done-when. One coder builds C5 and C6 on one branch, and both land before any run, so the runs
that read the old data and order are repeated once.

| ID | Task | Depends on | Done when |
|---|---|---|---|
| C5 | `setup/data.py` `schedule(t, n_s, N, seed)` per F8c: each pass orders indices by `BLAKE3(tag ‖ s ‖ e ‖ i)`, `⌊N/n_s⌋` whole batches, leftovers sit out; the tag and byte encoding pinned in `verification/CLAUDE.md`; the seed passed through the prover loop, the driver and every run; seed 1 at test scale | A14, B7 | tests: the same list on every call, every batch `n_s` distinct records, each pass a permutation of `D`, different seeds giving different orders |
| C6 | `setup/data.py` + `runs/materialize_data.py`: BackdoorLLM's released refusal files at F8d's commit, each checked against its SHA-256 pin; BackdoorLLM's `alpaca` template transcribed from F8e's pinned source; F8d's shared-list filter under the four full-scale tokenizers, then SmolLM2's; `VERIF_N_RECORDS = 365`; `D̃` by F8g's two seeded hashes over the step-1 batch and the 492 poisoned records, replacing C4's trigger insertion; B8 reads the released 200 held-out prompts | C5 | the counts 369, 492 and 365 reproduced; the boundary-merge check passes; new `h_D`, `h_D̃` and manifests; pins and counts reported to me for `DECISIONS_SETUP.md` |
| C5–C6 runs | A11, A12, A13 with the step-4 sweep, A14 and B7, each re-run once on the new `D`, `D̃` and `π` | C5, C6 | each meets its original done-when; changed C1 values reported to me for `DECISIONS_SETUP.md` §8.B C1 and `STATUS.md` |

**Waves.** T0 → {A1, B1, B4} → {A2, B2, B3, B5} → {A3, A4} → A5 → A6 (M1) → {A7 (M2), B6} → A8 →
A9 → {A10 (M3), B7} → A11 (M4) → {A12 → A13, A14} (M5) → {C5, C6} → C5–C6 runs. B8, unbuilt,
moved on 2026-10-08 to the evaluation part, and the same day to its end (F6c).

**`η` is a declared argument, not tuned (S8, amended 2026-10-01).** `VERIF_ETA = 1e-3` is fixed
in the config before any run, and prover and verifier read the same value. The former tuning task
(C2, old B6) is removed.

## When I stop and ask you

- Anything that would change the spec or the reference block, such as a matmul the inventory
  doesn't list or a capture that can't be labeled.
- Measured `s_h` well above 1, a concentration-guard failure, `k` recomputed above 7 (P10d),
  a realized floor above target, or any honest false rejection.
- A cheat rejected at a check other than its declared one.
- An environment blocker, such as a missing wheel or the model or dataset revision
  unavailable.

I also send you a short report at each of M1–M5, including what to record in the design docs.

## Doc updates (me, under the lock rules)

- **After approval:**
  - copy this plan to `docs/verification/IMPLEMENTATION_PLAN.md`;
  - update `STATUS.md` to show stage 4 closed and stage 5 started, keeping "the next item is
    **ID**" wording, next item C4;
  - record the S6c check-order revision in `DECISIONS_SETUP.md`.
- **As they close:** record C4, C1, C3, C5 and C6 in `DECISIONS_SETUP.md` §8.B and remove them
  from `SETUP_TASKS.md`.
- **At the end:** ~~add the README pointer (S7d).~~ Done on 2026-10-04 in a different form:
  the tutorial was removed, the package moved from `src/verification/` to `verification/`, and
  `README.md` now describes the protocol (`DECISIONS_SETUP.md` §8.B S7, amendment).

## Verification

- `.venv/bin/python -m pytest tests` passes after every merge.
- M1: `.venv/bin/python -m verification.runs.mlp_smoke` accepts the honest run and
  rejects each fault at its declared check.
- M3: an honest Llama step is accepted and the per-class normalized residual table is printed.
- M5: `run_verified` accepts all 10 steps, and every cheat run's oracle passes. The band-file
  hash is equal across runs, and C3's decisions match. The flipped-matmul sweep's measured
  threshold is compared with `f_achieved = 0.86` (appendix §11). The cost grid is complete for
  the honest run, and the plain baseline's final weights equal the honest run's bit for bit.

## Full-scale part (in preparation since 2026-10-08)

**Decision (user, 2026-10-08).** The code that the closed full-scale decisions require gets a
full-scale part of this plan, with its own task table and milestones, approved by the user as
this plan was. No full-scale code is written before that approval. The work falls into six
groups, settled one at a time; this section gains each group's tasks as it is settled.

1. **Model support:** Falcon3's untied output layer, a Qwen2 instance of `C`, and F14's
   biased linear as one augmented product (S2, F14).
2. **Mixed precision:** autocast from the `(MASTER_DTYPE, COMPUTE_DTYPE)` pair (F6b), the
   replay under the prover's autocast setting and cuBLAS's bf16 reduced-precision reductions
   off (F5a).
3. **GPU path:** each step copied off the GPU into host RAM (F1c), forward values kept on the
   GPU (F4a). This is the one-GPU path; at full scale F22 replaces the host copy by peer
   copies to GPUs 1–3.
4. **Full-scale data and run orchestration:** the jailbreak corpus, the list shared by the
   four tokenizers (F8d), poisoned batches at every level (F8g), and the runs of
   `EVALUATION_SPEC.md` §3.2, including F5c's stop rule and F9's one revision.
5. **The `η` pilot grid (F6a)**, which needs the attack-success scorer (EQ8).
6. **H100 measurement scripts:** the hashing benchmark (`HASHING_PLAN_DRAFT.md` Section 6),
   E6's eager-against-fused step (F16) and the verifier's GPU peak (F4).

**Order.** After C5 and C6 close, groups 1, 2 and the data half of group 4 are built and tested
on CPU, with no GPU needed except for the cuBLAS setting. The GPU path and the measurement
scripts follow, then the H100 runs. *Why:* the one code path puts full-scale code in the same
packages, and proving model support and mixed precision on CPU keeps H100 hours for runs.
*Rejected:* a separate task-list file (no milestones or approval gate); parking everything
until an H100 is booked (the CPU-testable work would wait for nothing).

### Group 1 — model support (settled 2026-10-08, user)

- **One instance for all four models.** The Llama instance (`computation/instances/llama.py`)
  is extended, and renamed to a family-neutral name, with two config-driven switches: tied or
  untied output layer, and q/k/v biases on or off. Qwen2's Hugging Face modules have Llama's
  names and layout and differ only by those biases. With an untied output, the output matrix
  is an ordinary linear weight (its gradient a checked product, its update in 6a), and `W_E`
  keeps only the lookup gradient (6b). `M` and `k` don't change (S2).
- **Done when, per real model, on CPU** (fp32, 4 × 128, test scale's settings): every slot is
  filled exactly once (as M2); every forward and backward operand and every glue gradient is
  rebuilt bit for bit (as A8, A9); a step with one bias entry altered is rejected. The residual
  table is printed, not judged: bands are calibrated on the H100 in bf16. The fast suite gains
  small random configs for each switch combination.
- **Order:** S2's suggestion, Llama-3.2-1B (fits today's code; tests scale and its `llama3`
  rotary rescaling), then Falcon3-1B (untied), then Qwen2.5-0.5B and -1.5B (biases). Each model
  is pinned to a commit when built; Llama-3.2 needs the user's Hugging Face token after
  accepting Meta's license.
- *Rejected:* a separate Qwen2 instance file (duplicates the labeling and replay); random
  configs only, with real models left to the H100 (the first real-model failure would cost
  H100 time).

### Group 2 — mixed precision (settled 2026-10-08, user)

- **Labeling follows autocast's casts.** Under autocast a matmul sees a fresh bf16 copy of its
  weight, and `W.grad` is a fresh fp32 copy of the bf16 weight-gradient product (observed with
  torch 2.9.1, CPU autocast, a biased linear). Both copies are `aten._to_copy` calls that the
  dispatch mode sees. Capture records them, and the labeler follows a bf16 copy back to its
  fp32 weight and a weight gradient back to its bf16 product, so identity stays the labeling
  rule. Committed products are the bf16 outputs as computed; committed weights are the fp32
  masters (F6b).
- **Done when, on CPU** (SmolLM2 at `(fp32, bf16)`, then one full-scale model after group 1,
  probably Qwen2.5-0.5B): every slot filled exactly once; every operand and glue gradient
  rebuilt bit for bit under the replay's autocast; after a few steps the plain run's final
  weights equal the verified run's; at `(fp32, fp32)` every pinned root is unchanged.
- **The cuBLAS setting** (`allow_bf16_reduced_precision_reduction = False`, F5a) joins
  `setup_determinism` now; it is a no-op on CPU and confirmed on the GPU.
- **Caveat.** CPU and CUDA autocast don't convert exactly the same ops, so the CPU run
  rehearses the mechanism; the bit-for-bit check repeats on the GPU.
- *Rejected:* matching bf16 copies to weights by value (a full compare per product, weaker than
  identity); building and debugging mixed precision only on the GPU (paid hours for bugs a free
  CPU run finds; no GPU-free regression test).

### Group 3 — the GPU path (settled 2026-10-08, user)

- **Copy during the step, asynchronously.** Each captured product output is copied into pinned
  host memory on a side CUDA stream as soon as it is produced, and its GPU copy is freed once
  the copy completes. Host buffers are allocated once and reused every step. *Why:* capture
  today holds every product until the step ends, which at full scale is the whole 32–64 GB
  transcript on an 80 GB GPU that also holds the model and activations; F1c rejected holding
  the transcript on the GPU. Asynchronous copies overlap the GPU's work, where synchronous
  ones would stall it on the cost `HASHING_PLAN_DRAFT.md` names as the largest per step. How
  hashing overlaps the copies stays with F15; this builds the plain, correct copy.
- **Debugged first on a small rented CUDA GPU** with SmolLM2, then on the H100. This Mac has
  no CUDA, and on CPU the copy is a no-op, so this group can't be tested on CPU.
- **Price:** 32–64 GB of pinned host RAM for the run; stream synchronization is the main bug
  risk (a tensor must not be reused before its copy finishes); a few dollars and one more
  pinned environment (a CUDA torch build) for the small GPU.
- *Rejected:* synchronous copies (stall training on every copy); debugging directly on the
  H100 (bugs found on paid hours).
- *Since F22 (2026-10-20)* this group is the one-GPU path, which single-GPU sessions still
  use. At full scale on four H100s each product goes by peer copy to its owner GPU (FS32), and
  no pinned host pool is needed.

### Group 4 — full-scale data and run orchestration (settled 2026-10-08, user)

- **The data half (H0) is planned and built now, on CPU.** It extends C6's test-scale builder:
  the jailbreak corpus, the list shared by the four tokenizers (F8d), one `D` and `h_D` per
  model, poisoned batches at every level, nested (F8g), and five seeds' schedules (F8c). Done
  when it reproduces F8d's shared-list counts exactly (369 clean and 492 poisoned Alpaca, 233
  clean and 86 poisoned AdvBench), and tests show the levels nest and each schedule pass uses
  a record at most once. Llama-3.2's tokenizer comes from the public mirror until the user
  accepts Meta's license.
- **The orchestration half is planned after the user approves `EVALUATION_SPEC.md`**, whose
  §3.2 is its specification (run order, counts, F5c's stop rule, F9's one revision).
- **Price:** the orchestration tasks wait on the spec review, which also blocked B8. *(The user
  approved `EVALUATION_SPEC.md` on 2026-10-08, so the orchestration half is unblocked; B8 is
  postponed to the end by F6c.)*
- *Rejected:* planning both halves now from the draft spec (rework if the spec changes).

### Group 5 — the `η` pilot (settled 2026-10-08, user; superseded the same day by F6c)

*Superseded:* attack-success scoring moved to the end, the pilot is dropped and `η = 10⁻³` is
fixed (`DECISIONS_FULL_SCALE.md` F6c). The record below is kept for the reasoning.

- **Planned now as its own task**, not inside the orchestration half. It depends only on the
  data half (group 4), mixed-precision plain training (group 2) and B8's scorer (EQ8 as
  amended by EQ18), extended to the jailbreak prompts and to GPU generation. It is the first H100 run. *Why:*
  F8b and F6a define it fully, so it doesn't wait on `EVALUATION_SPEC.md`; its outcome sets
  the H100 budget (the fallback adds 20 plain-training runs per model and attack, 160 in all);
  it needs none of the protocol's GPU code.
- **Price:** the scorer's jailbreak and GPU extension comes ahead of the rest; the pilot spends
  H100 time, mostly generation, before any verified run. Its cost is estimated when the task is
  written.
- *Rejected:* folding it into the orchestration half (the H100 budget stays unknown longer).

### Group 6 — H100 measurement scripts (settled 2026-10-08, user)

- **The hashing benchmark's scripts (E0–E6) are written now, standalone**, importing only the
  leaf-size list (`HASHING_PLAN_DRAFT.md` Section 7, item 2). They are smoke-tested at small
  sizes on the Mac, and their GPU parts on group 3's small GPU. cuPQC's license is checked
  before E2 is written (Section 7, item 1).
- **No separate memory script.** `metrics.py` (B6) already logs peak memory; one verified
  bring-up step per model on the H100 gives the figure that closes F4.
- **H100 order:** (1) first session: the `η` pilot, the hashing benchmark and one verified
  bring-up step per model; (2) the user decides F15 by rules R1–R5, and F4 closes from the
  bring-up steps; (3) the chosen hashing is built and the test-scale runs repeated; (4) H1.
  *Why:* the hash function is in every committed root and in `h_D`, so deciding it after H1
  would discard H1; the bring-up step happens anyway, so its memory peak is free.
- **Price:** a pause between H100 sessions (the machine idles, or is released and re-rented on
  the same GPU type, EQ14); a second round of test-scale re-runs if the hash function changes,
  since C5 and C6 repeat them first and holding those back would stall test scale for weeks.
- *Rejected:* running the benchmark after the verified run works on CUDA (E6 would time the
  real verifier rather than estimate it, but the hashing decision and H1 come later).

**All six groups settled (2026-10-08).** The task table below was a draft awaiting the
user's approval; no full-scale code is written before it. The user delegated that approval
to me, to give once the plan was complete. The runs were added on 2026-10-11 (section "Runs"
below), and I approved the table the same day.

### Full-scale tasks (approved 2026-10-11, delegated by the user)

Same working rules as the test-scale tasks: `opus` coder and reviewer per task, worktrees,
merge to `verification-impl`, nothing pushed, a gate at each milestone. IDs are `FS1, FS2, …`
and milestones `FM1, FM2, …`. Every task starts after C5, C6 and their runs close, since they
change `setup/data.py`, the config and the pinned roots.

| ID | Task | Depends on | Done when |
|---|---|---|---|
| FS1 | Generalize `computation/instances/llama.py` into one decoder instance (family-neutral name) with two config switches: tied or untied output layer, q/k/v biases on or off (group 1). Untied: the output matrix is a linear weight (its gradient a checked product, update in 6a), `W_E` keeps only the lookup gradient (6b) | C5–C6 runs | fast suite: small random configs for all four switch combinations fill every slot once and rebuild every operand and glue gradient bit for bit; SmolLM2's pinned roots unchanged |
| FS2 | F14's biased linear: capture accepts an `addmm` with a bias and records `b`; substitution returns the committed leaf; check 5 measures `Y·r` against `A·(B·r) + (b·r)·1`; the bias is a weight whose gradient is glue (6b). F14's one sentence added to spec §2 | FS1 | a small random biased config: an honest step accepted under provisional bands, a step with one bias entry altered rejected; capture stays bit-identical to an uncaptured run |
| FS3 | Mixed precision (group 2): autocast from `(MASTER_DTYPE, COMPUTE_DTYPE)` around forward and loss, in the step and in the replay; capture records `aten._to_copy`, and the labeler follows a bf16 copy to its fp32 weight and `W.grad` to its bf16 product; F5a's cuBLAS setting in `setup_determinism` | C5–C6 runs | SmolLM2 at `(fp32, bf16)` on CPU: every slot filled once; operands and glue gradients rebuilt bit for bit; plain and verified final weights equal after 3 steps; at `(fp32, fp32)` every pinned root unchanged |
| FS4 | Real-model bring-up on CPU (fp32, 4 × 128): Llama-3.2-1B, Falcon3-1B, Qwen2.5-0.5B, Qwen2.5-1.5B, each pinned to a commit; Llama from Meta's repository once the user has accepted the license | FS1, FS2 | per model: every slot filled once, every operand and glue gradient rebuilt bit for bit, a bias-tampered step rejected (Qwen); the residual table printed, not judged; pins reported to me → **Milestone FM1** |
| FS5 | Mixed precision on Qwen2.5-0.5B at `(fp32, bf16)` on CPU | FS3, FS4 | FS3's criteria on the real model → **Milestone FM2** |
| FS6 | Full-scale data builder (H0, group 4): extends C6's builder with the jailbreak files, one `D` and `h_D` per model on the shared list, poisoned batches at every EQ5 level nested by F8g's hashes, and the schedules of seeds 1–5 (F8c) | C5–C6 runs | the shared-list counts 369 / 492 / 233 / 86 reproduced; tests: levels nest, each pass uses a record at most once, jailbreak stops at 50%; pins and hashes reported to me |
| FS7 | *Postponed to the end of implementation and evaluation (F6c, 2026-10-08).* Scorer extension (group 5): B8's scorer extended to the 99 jailbreak held-out prompts and to batched GPU generation, keeping EQ18's prompt format (training template up to `### Response:\n`, no special tokens), its answer-only decode, and BackdoorLLM's keyword lists, `refusal_eval`/`jailbreak_eval` and generation settings copied verbatim (EQ8 as amended by EQ18) | B8, FS4 | CPU: the extended scorer gives the same verdicts as B8's path on `W_0` for a sample of prompts; the batched path gives the same generations as the unbatched one |
| FS8 | *Dropped (F6c, 2026-10-08): no `η` pilot; full scale fixes `η = 10⁻³`.* | — | — |
| FS9 | GPU path (group 3): captured outputs copied asynchronously to reused pinned host buffers on a side stream, the GPU copy freed when the copy completes; a no-op on CPU | FS3 | on a small rented CUDA GPU with SmolLM2: verified and plain final weights equal; operands rebuilt bit for bit; GPU memory doesn't grow with the transcript over a step; the environment (CUDA torch build) pinned |
| FS10 | H100 benchmark scripts E0–E6 (`HASHING_PLAN_DRAFT.md` Section 6), standalone, importing only the leaf-size list; cuPQC's license checked before E2 | FS4, FS3 (for E6's real models) | smoke runs at small sizes on the Mac (CPU parts) and on the small GPU (GPU parts) write the JSON record → **Milestone FM4**, with FS9 |
| FS11 | First H100 session: the benchmark (FS10); one verified bring-up step per model with `metrics.py`'s peak memory; F6c's smooth check, 10 plain steps at `η = 10⁻³` per model and task with the whole-list loss | FM2, FM4 | the benchmark JSON, the per-model peak and the loss curves reported → **Milestone FM5**: the user decides F15 by R1–R5, F4 closes, and a failed smooth check comes to the user |
| FS12 | `k` tunability (spec §5.8, EQ11 as amended 2026-10-11): `runs/k_tunability.py` reads a stored step-4 transcript (EV2) and runs check 5 over the whole step through the verifier's own check-5 function with `k` as an argument, for `k = 1 … k_top` (F10: 18 at test scale, 44 for Llama-3.2-1B), recording time and peak memory per `k` and FLOPs in a counting pass; the §5.8 figure and table script in `evaluation/results/` | EV1, EV2 | on the test-scale stored step 4: the `k = 9` pass reproduces the stored step's check-5 verdict and residuals; time grows about linearly in `k`; the figure and table written under `results/test/` (joins EM1) |
| FS13 | Full-scale run driver (group 4's orchestration half, spec §3.2): `runs/full_scale.py` runs one model and attack in the spec's order, H1 (steps 1–3 calibrate; step 4's transcript kept), H3 with F5c's stop rule, H2 for seeds 2–5 (step 1 verified, not judged), the cheat runs as one process per seed (the model loaded once; A1–A3 at one level share the poisoned step; under `VERIF_PIPELINED=1` each run's prover is a child process its parent spawns, FS33d), the hidden-step run, then the sweep (EV3) and `k` tunability (FS12) on H1's step 4. It stops the cell on F5c's mismatch, and pauses for the user before H2 if a judged H1 step trips the cancellation guard (F9). A run directory with a `run_end` record is skipped, so an interrupted session resumes. The smooth check stays in FS11 | FS1, FS3, FS6, EV3, FS12 | on CPU, a reduced cell per attack at test scale (SmolLM2, 2 seeds, 2 poisoning levels) runs end to end, every cheat rejected at its declared check; EV4–EV7's scripts write multi-panel outputs from its records; a killed driver resumes without redoing finished runs; a planted H3 weight mismatch stops the cell; a forced guard trip pauses before H2 |
| FS14 | Second full-scale session: FS13 on all 8 cells (4 models × 2 attacks) on F22's environment (four H100 SXM over NVLink, or one B200 under its fallback) with F23's kernel, once EQ14 is revised for GPU-seconds and pipelining; then EV4–EV7's and FS12's scripts write `results/full/`. The session also carries the resource profile (`archive/PROFILING_PLAN.md` §11): the probe first, the sampler on every part, and H1's step 5 traced in each model's refusal cell | FM6, FM8, FM9, FM10 | every run directory complete, each environment record naming its GPUs and their topology, each run's cost records split by device with wall time and GPU-seconds as EQ14's revision sets them; F5c's check passed in every cell → **Milestone FM7**: the results shown to the user, before any attack-success scoring |
| FS15 | Check 5's split error model (`DECISIONS_FULL_SCALE.md` F18, 2026-10-12): the verifier computes the term scale `U_m` from the reconstructed operands (a biased product's ones column and bias row included) in the pass that computes `ν_m`; the band `τ·σ_r·(√2·ε_in·‖P_m‖_F + √q_m·ε_acc·max(‖P_m‖_F, U_m))` is used, through one function, by check 5, calibration's fit of `s_h` and its `τ/2` rule, P10d's recompute of `k` (sized at the largest calibration `e_m`), the step records' `ρ_5` and realized floor (the step's largest `f_achieved`), the sweep grid and `k` tunability; the test-scale band file refitted | FS4, EV3, FS12 | fast suite; SmolLM2's test-scale honest run accepted under the refitted band file, its hash, `s_h` and `k` reported; FS4's bring-up rerun on all five models with the residual tables printed, and on Qwen2.5-1.5B an honest step accepted with the `τ/2` rule holding and a bias-tampered step rejected; the cost of `U_m` measured against `ν_m` → closes **Milestone FM1** |
| FS16 | The output layer in row blocks (`DECISIONS_FULL_SCALE.md` F4b, 2026-10-15) and the allocator fix: `VERIF_HEAD_BLOCKS = B` in the config (8 at full scale, 1 at test scale, dividing `n_s`); a training entry point on the declared computation that runs the body once, then each block's scores, loss share and backward, then the body's backward on the joined `δF`, used by the plain step, the prover and calibration; per-block products `Λ_b, δF_b, G_b` in the canonical order; block-aware labeling and replay; the output weight's gradient as one shared fp32 sum used by the prover and check 6; F6d's smooth rule; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the full-scale env and an emptied cache between the prover's and the verifier's turns, timed in its own metrics section | FS11 (FM5) | fast suite; at `B = 1` the loss, every gradient, the product names and the canonical order bit-identical to before; at `B > 1` on CPU no tensor near score-sized for the whole batch in training, the prover or the verifier; honest steps accepted and planted errors in `Λ_b`, `δF_b` and `G_b` rejected; SmolLM2's test-scale run unchanged at `B = 1` and accepted at `B = 4` |
| FS17 | H100 re-check session (approved 2026-10-15): one verified bring-up step per model at `B = 8` with the allocator fix and `metrics.py`'s per-section peaks; F6d's smooth check on all 8 model–task pairs; the plain step timed with and without expandable segments on each model; no hashing benchmark | FS16 | each model's prover and verifier peaks, the smooth verdicts, the timing difference and F4b's measured prices (step time, transcript bytes, check 5 time) reported → the user decides whether F4 closes and whether the allocator fix stays |
| FS18 | Machine facts in a shared module, `setup/machine.py`, used by the hashing benchmark and the runs' environment record (`archive/PROFILING_PLAN.md` §9) | wave 0 | the benchmark's record keys unchanged; RAM total, power limit and PCIe link in the environment record |
| FS19 | Section markers on the monotonic clock and CUDA events, and check 5's per-run stamps, in `markers.jsonl` and `c5_layout.json` | wave 0 | marker spans match the time rows; stamp counts match the layout; bit identity with the recorder on and off |
| FS20 | A capture record per step, transcript bytes in the timed pass, `check5_profile` output saved | FS19 | transcript bytes equal the counting pass's; bit identity |
| FS21 | The out-of-process sampler (NVML and host counters at 100 ms) beside every session part | wave 0 | the sampler's tests; samples written by the session self-test |
| FS22 | The probe: the machine's measured rates and fixed costs, as a session part | wave 0, FS18 | the hashing fit recovers synthetic rates; a CPU run writes a valid file |
| FS23 | The traced step under `torch.profiler`, check 5 traced on a slice, traced rows left out of the cost tables | FS20 | every section annotated; bit identity; config hash unchanged |
| FS24 | The trace summarizer in `evaluation/results/profile/` | frozen formats | exact numbers on a fixture trace; no torch import |
| FS25 | The analysis: floors, bound types and marks in `profile.json` | frozen formats | the bottleneck rule's cases on synthetic inputs |
| FS26 | The HTML report by `archive/PROFILING_PLAN.md` §12 (settled 2026-10-19): one page over any set of profiles, drawing only the comparisons across models and sessions that the inputs hold | FS25 | one self-contained file with no network reference; no comparison panel for one model in one session |
| FS27 | The H100 profiling check: every CUDA part on SmolLM2, and check 5 measured on the four full-scale models (one H100 SXM, about $8–14) | FS18–FS25 | `archive/PROFILING_PLAN.md` §9's acceptance list → **Milestone FM8**: the results shown to the user |
| FS28 | The BLAKE3 GPU kernel (`DECISIONS_FULL_SCALE.md` F23, decisions 1–2): CUDA C compiled at run time through NVRTC, called through ctypes over the NVRTC library torch's Linux CUDA wheels install and the driver's `libcuda` (no new package); one call hashes a table of leaves (header bytes, device pointer, length; one launch per tree level, F23 decision 2) as `BLAKE3(0x00 ‖ header ‖ payload)`, the finiteness check fused, the first error in leaf order returned; an entry point in `commitment/`, not yet called by the protocol | — | fast suite unchanged without CUDA; on a small rented CUDA GPU (one A40): BLAKE3's published test vectors, lengths around the block and chunk boundaries, every dtype and header size, non-finite values at chosen positions, and random leaves all equal to the C library's digests bit for bit (F23's K3) |
| FS29 | The kernel's speed on one H100 (F23's K1, K2): a session part that hashes each full-scale model's step from its leaf-size list (`leaf_sizes`), digests checked against the C library | FS28 | GB/s per model and per leaf-size band reported → **gate**: at 300 GB/s or more FM9 goes on; below about 150 GB/s the choice returns to the user, with cuPQC's SHA-256 as the alternative; in between, the user decides. One H100 SXM, about 1 h, about $4 |
| FS30 | Device leaves and routing (the plan's G2; F23's K4): leaf encoding on the device with the same bytes as the host path; `leaves.py` routes each leaf by its device, host leaves to the C library and device leaves to the kernel; on today's one-H100 path the prover hashes its products on the GPU, and check 2 stays on the CPU and recomputes every digest | FS29's gate | fast suite with the CPU path unchanged; on a CUDA GPU every model's step gives the CPU path's root, and a planted NaN is rejected at the same leaf with the same error → **Milestone FM9** |
| FS31 | The four-H100 hardware probe (F22's HW1–HW3): a session part that records `nvidia-smi topo -m`, the peer-access matrix, the peer-copy rate of every pair by PyTorch `copy_` (and `nvbandwidth`'s when the pod has it), and runs one plain step per model on each card with its `W_{t+1}` digests compared across the cards; the cost guard checked on the four-GPU pod first; a byte check of a peer copy between every ordered pair, against a copy staged through host memory, which every session on two or more GPUs runs first and stops on (F22's HW2, 2026-10-21) | — | peer copies equal to the host-staged ones on every pair, `NV18` and peer reads OK between every pair, the slowest pair's peer copy at 300 GB/s or more (150–300 returns to the user, below 150 fails; F22's HW2, 2026-10-20), the same digests on four distinct cards, reported → **gate**: a failure is tried on the next provider (Runpod, Lambda, Google Cloud); failing on all, the user confirms the fallback to one B200 (F22, decision 4). One four-H100 SXM pod, 1–2 h at about $16 per hour |
| FS32 | Per-role devices and the split replay, sequential (F22, decisions 1–2; the plan's design 1): `VERIF_VERIFIER_DEVICES` beside `VERIF_DEVICE`; the replay's units placed on GPUs 1–3 by the sharding rule, `X_j` and `δX_{j+1}` moved at the unit boundaries; the transcript sharded by owner device; capture routes each product to its owner by peer copy; check 5 measures on the product's device; device leaves hashed through a host copy until FS34 | O10 and O11 merged | fast suite on the Mac with every device `cpu`; on a two-GPU NVLink pod (two A100 SXM at about $3.6 per hour or two H100 SXM at about $8; Runpod's A40 pairs copy wrong bytes peer to peer, F22's HW2, 2026-10-21): the split replay rebuilds every operand and glue gradient bit for bit against the one-GPU replay, roots equal, every cheat rejected at its declared point |
| FS33 | The pipelined loop (F22, decision 3): GPUs 1–3 verify step `t` while GPU 0 trains `t + 1`, the prover in its own process (F22, FS33's process model, 2026-10-21); two live transcripts; a cheat run stops at its first verified rejection; per-device clocks and peaks in the metrics | FS32 | on the two-GPU pod: every decision equal to the sequential loop's, step by step; per-GPU peaks recorded |
| FS34 | The join: F23's kernel on the owner GPUs, hashing each product on arrival for the prover's commit, and every leaf held on a GPU where it sits for check 2 and the other verifier hashes; F23's once-per-run check of each leaf shape and dtype against the C library (keyed by path, device, tag, dtype and shape, per FS34's mechanism) | FM9, FS32 | on the two-GPU pod: roots equal to the CPU path's; a planted kernel mismatch stops the run at the per-shape check |
| FS35 | The per-device tools, then the four-H100 session. The tools (user, 2026-10-21; `ENVIRONMENT_PLAN_DRAFT.md`'s code-change table for B, which no row owned until then): `check5_profile.py` synchronizes and times every device check 5 uses; `runs/probe.py` measures every GPU it is given, not one; `setup/machine.py` records every GPU (name, memory, UUID), not GPU 0's properties alone, beside the NVLink topology it already records. The session: one verified step per model, sequential and pipelined, with the probe, the sampler and per-GPU peaks | FS31's gate, FS33, FS34 | the tools' tests on the Mac with every device `cpu`; each model's step time, per-GPU peaks and peer-copy rates reported, the environment record naming all four GPUs → **Milestone FM10** |
| FS36 | The measuring core (`MEASURE_KERNEL_PLAN_DRAFT.md` 2.1): `matmul_check/measure_core.h`, plain C with one fixed fp32 reduction order and fp64 norms, its host build and a ctypes binding for tests | — | on the Mac: the host build agrees with `_measure_products_reference` within a tolerance on random and adversarial inputs (bf16 and fp32, strided and transposed views, biased, zero rows, a NaN, an Inf, the guard off), and gives identical bits under any batch, stride, read-sharing and table order |
| FS37 | The measuring kernel over NVRTC (2.2), check 5 serving views (2.3) and the host challenge source (2.6), routed by device; check 5 still in canonical order | FS36 | fast suite unchanged on the Mac; on one A40 (about $0.6 per hour): the kernel equals the host build bit for bit on FS36's inputs and every product of a tiny decoder step, agrees with the torch path within FS36's tolerance, the step accepted and every cheat rejected at its declared product |
| FS38a | Unit-scheduled measuring with end judging and the canonical fallback (2.4, 2.4a) | FS37 | on the Mac every CPU number bit for bit as before, the first failing product and message equal to canonical judging for a planted failure in every class and layer (layer 1's forward included), every error rule and record-pass status as before |
| FS38b | Shared reads in the kernel, one table per unit, one-sided shares only (2.4, 2.4a, §5; user, 2026-10-22) | FS38a | on the A40 kernel rows equal FS37's bit for bit, reads saved reported per unit |
| FS39 | Overlap with the glue (2.5): a measure stream per device, one unit in flight, holds, worker-thread challenges, pinned row readback; the metrics of 2.7 | FS38 | on the A40 rows and verdicts equal FS38's bit for bit, a write to a held operand raised; on a two-GPU NVLink pod (two A100 SXM, about $3.6 per hour) with `VERIF_VERIFIER_DEVICES`, rows and verdicts equal the one-GPU run's |
| FS40 | H100 timing: one verified step per full-scale model, traced on Llama-3.2-1B, check 5's sections against the plan's estimates, the band fit rerun with the kernel (one H100 SXM, about $4 per hour, about 1.5 h) | FS39 | each model's `5.measure` and verifier time reported against the 2026-10-20 pod's, the kernel equal to the host build on sampled products → **Milestone FM11** |
| FS41 | The GPU challenge source: the hashing session's keyed XOF behind `ChallengeSource` (2.6) | FS39, the hashing session's XOF | on a CUDA GPU the device challenges equal `challenge_matrices`' bit for bit for every product of a step; FS40's Llama step timed again |

**Resource profiling (user, 2026-10-18).** The user asked for full resource profiling of the
full-scale runs: time and utilization per protocol section, tied to the hardware parts, with
bottlenecks marked and an HTML diagram that presents the profile after the run. The
clarification round settled the design on 2026-10-18 and 2026-10-19;
`archive/PROFILING_PLAN.md` holds it with the task briefs FS18–FS27 and their frozen file
formats. These tasks run with a Sonnet coder, tester and reviewer per task at medium effort,
at most 10 agents at once, instead of this section's `opus` rule.

**The GPU kernel and the four-H100 port (user, 2026-10-20).** F22 moves full scale to four
H100 SXM over NVLink, with one B200 as the fallback, and F23 hashes with our own BLAKE3 GPU
kernel (`DECISIONS_FULL_SCALE.md`). They are built as two independent milestones in
parallel, in either order: **FM9**, the kernel (FS28–FS30), and **FM10**, the four-GPU port
(FS31–FS35). Only FS34 joins them. `ENVIRONMENT_PLAN_DRAFT.md` holds the port's design: the
data path, the sharding rule, the code changes per module, memory and testing. FS32 waits for
the check-5 speed work's member blocks and vectorized judging (O10, O11), which rewrite the
serving it splits. FS31 runs as soon as a four-H100 pod is available, before the port is
built, so a failed feasibility condition sends the work to the fallback early. Under the
fallback, FS32's per-role devices collapse to one device, FS33 and FS34 run on the B200, and
FS35 becomes a B200 session. These tasks follow the profiling tasks' agent rule: an Opus
orchestrator, a Sonnet coder, tester and reviewer per task at medium effort, at most 10
agents at once.

**Check 5's measuring on the GPU (user, 2026-10-21).** O2's three decisions (`PERFORMANCE_TASKS.md`): a fused measuring kernel with challenges drawn on the GPU, shared reads with a layer's forward products measured during its backward, and measuring overlapped with the glue. `MEASURE_KERNEL_PLAN_DRAFT.md` holds the design, approved by the user on 2026-10-21, with its metrics rule (2.7): `5.measure` keeps the verifier's wall cost of measuring, and the measure stream's GPU spans stay in the trace. FS36–FS40 (FS38 as FS38a and FS38b) make **FM11**; FS41 waits on the hashing session's keyed XOF and FM11 doesn't wait for it. FS38 and FS39 change the check-5 serving and `_read_rows` that FS32 and FS33 built on, so they are coordinated with the session building FS34 and FS35. The agent rule is the profiling tasks'.

**Review notes carried from C5, C6, FS6, FS1, FS2, FS3 and FS10 (2026-10-11).** Their reviewers found these
limits of the code; the tasks named fix them. FS6 fixed C6's notes on the per-model pad in
`materialize_data`, the jailbreak files and the tokenizer pins (recorded under
`DECISIONS_FULL_SCALE.md` F8d).
- **FS2 (merged as `07b1cdc`).** Check 5 takes `b` from the committed `W_t` leaf through a
  `ProductSpec.bias` field; `q` counts the bias, so `e_m` uses `q + 1`. F14's sentence was
  already in spec §2, and §6's formulas cover the augmented operands by that definition. FS2
  also closed the two small gaps C5 and C6 left: check 1 rejects a `π(t)` with repeated
  indices, and `band_context` records `threads` without comparing it, so today's band file
  still loads (the thread count changes only the CPU's reduction order, far below `τ`).
- **FS3 (merged as `6eeb7d9`).** FS2 casts the committed bias to the operand dtype in check
  5, and FS3 confirmed that CPU autocast hands `addmm` the same bf16 bias, bit for bit. Capture
  records each `_to_copy` (a weight's bf16 copy, and the fp32 copy of `G` that autograd makes
  `W.grad`); check 6a widens `G` to the weight dtype; `μ` uses the cross-entropy's dtype. Three
  findings carry forward:
  - **F6b at bf16 (raise at FM1).** Qwen2's bias gradients leave autograd in bf16, while F6b
    says glue gradients stay fp32. On CPU the replay is bit-identical (`ρ = 0`). On GPU a
    different reduction order could reject honest steps or widen `τ_W`; F5a's bit-identical
    replay is the guard. Recommendation for FM1: keep autograd's behaviour. *Decided
    2026-10-12 as F19: kept.*
  - **Check 5's column-shaped slack (note at FM2).** Check 5 admits an error shaped like one
    column of size about `τ·√width·e_m·rms(P)`, its Frobenius tolerance as F10 prices it, not
    a bias weakness. At bf16 on a tiny model a bias shift of about 0.125–0.25 passes check 5
    when the honest update is committed; 6b catches it when `W_{t+1}` is trained from the
    shifted bias.
  - **Speed.** A CPU prover step at bf16 takes about 45 s against 0.8 s at fp32 (SmolLM2),
    since this Mac has no fast bf16 matmul. FS5's Qwen-0.5B steps take minutes on CPU.
- **FS4 at FM1 (2026-10-12).** The CPU bring-up rejected an honest Qwen2.5-1.5B step at check
  5 (`L1.dQ[0,3]`, 79.3 against `τ = 39.2`), and Qwen2.5-0.5B's largest honest residual broke
  calibration's `τ/2` rule. Qwen2.5's large early key biases cancel in `dQ` against the softmax
  backward's zero row sums, so the honest error scales with the terms rather than the product.
  The user adopted a split error model (F18); FS15 implemented it (`5e458f3`, 2026-10-13), all
  five models accept their honest step under it, and the user closed FM1 on 2026-10-13.
- **FS5 at FM2 (2026-10-13).** Qwen2.5-0.5B at `(fp32, bf16)` on CPU, merged as `40f89f0`
  (measurements only). Exact where it should be: the replay bit for bit with the bias
  gradients, plain and verified weights equal after 3 steps, fp32 unchanged. Over 3 steps the
  largest honest residual (1.785) passes `τ/2` (1.225), since at bf16 every class has the same
  RMS; the bias tamper is caught at 6b, not check 5; `k` sizes at 15. The user closed FM2 on
  2026-10-13 and parked the guard as `FULL_SCALE_TASKS.md` F20.
- **FS4.** The decoder instance pads with `PAD_ID = 2`, SmolLM2's EOS, whatever model it loads
  (`instances/decoder.py`). Before the first real-model step, FS4 wires each model's EOS from
  `setup.data.run_pin` (F8f: Llama 128009, Falcon3 11, Qwen 151645); the tinyllama test models
  have no pin.
- **FS10 (CPU parts merged as `c2daa3b`).** The GPU parts wait for the small GPU (FM4).
  - **Step sizes.** The leaf-size list gives one full-scale step as Qwen-0.5B 33.61 GB,
    Llama-3.2-1B 48.31, Falcon3-1B 53.14 and Qwen-1.5B 66.74. That is 2–5% above F1a's
    estimate. It was checked against real commits at fp32 and bf16, biases included.
  - **BLAKE3's thread count (for F15 at FM5).** For a leaf of 16 MiB or more, BLAKE3 runs on
    every logical CPU, whatever the thread setting. That holds in the protocol's `hash_leaves`
    too. On the Mac, one capped thread hashes at 2.6 GB/s against 7.9 GB/s uncapped. The
    benchmark reports both.
  - **Finiteness pass (for F15 at FM5).** Checking that every value of a Llama step is finite
    takes about 2.3–3.1 s, about 60% of BLAKE3's hashing time on the Mac. Most of it is
    per-leaf overhead across 394k leaves. The outer Merkle tree costs 0.17–0.26 s.
  - **Challenge generation.** A native batch over BLAKE3's C library generates a Llama step's
    challenges in 0.27 s on 18 threads, bit-identical to the protocol's. The Python path takes
    about 7 s.
  - **Smoke numbers.** The Mac smoke numbers were taken while the Mac slept with its lid
    closed, so they show the shape of the costs, not clean rates. FM4 and FM5 measure on the
    target hardware.
- **FS13.**
  - `D̃` is tied to one seed, task, level and batch size: `cheats.load_env` checks all four in
    `meta.json`. Each cheat seed and level builds its own `D̃` with `materialize_data`.
  - `other_batch` takes the neighbouring batch of the same pass, so it needs at least two whole
    batches per pass. AdvBench's list at full scale's batch has one.
  - EQ5's level list includes 0, but `VERIF_POISON_COUNT` starts at 1; level 0 is the honest
    run.
  - The band file sits at `$VERIF_OUTPUT_DIR/bands.json` (`verification/parameters.py`), so one
    output directory per seed and level would also move the band file.

**After FM5 (planned then, not now).** The chosen hashing is built, and the test-scale runs and
FS13's reduced cells repeated on it (`HASHING_PLAN_DRAFT.md` Section 7, items 6–12) →
**Milestone FM6**: the second session's estimate redone from measured step times, and the user
approves the session. Then FS14. *Since F22 and F23 (2026-10-20):* the chosen hashing is F23's
kernel (FM9), which test scale never runs (every digest is the C library's), so the test-scale
and reduced-cell reruns have nothing new to show; FM6's estimate comes from FS35's four-H100
step times (FM10), so FM6 follows FM10. The orchestration half of group 4 is FS13 and
FS14, planned on 2026-10-11 with the runs.

**Waves.** {FS1, FS3, FS6} → {FS2, FS9, FS10 (CPU parts)} → FS4 → FS15 (FM1) → FS5 (FM2) → FS10 (GPU parts, FM4) → FS11
(FM5) → FS16 → FS17 → the hashing build → {FM9, FM10} → FM6 (the estimate from FS35's step times) → FS14 (FM7). FS12 runs in the evaluation waves, and FS13 as
soon as its dependencies close, before FM5 if it can. FM3 is dropped with FS8; FS7 waits for
the end. The profiling tasks run beside the hashing build: {FS18, FS19, FS21, FS22} →
{FS20, FS24, FS25} → FS23 → FS27 (FM8), all before FS14; FS26 any time before FM7. FM9 and
FM10 run in parallel: FS28 → FS29 (gate) → FS30 (FM9); FS31 (gate) whenever a four-H100 pod is
available, and {O10, O11 merged} → FS32 → FS33, then FS34 once FM9 closes → FS35 (FM10). FM11 runs beside them: FS36 → FS37 → FS38a → FS38b → FS39 → FS40 (FM11), FS41 once the keyed XOF exists.

**Attack-success scoring is postponed (F6c, 2026-10-08, user).** The pilot and its cost
estimate are gone; FS7, B8 and the attack-success table move to the end of implementation and
evaluation.

**User actions on the path.** Accept Meta's Llama-3.2 license and provide a Hugging Face token
(before FS4's Llama run); approve the small-GPU rentals (FS9, FS10) and the H100 sessions
(FS11, FS17, FS27's profiling check at about $8–14) and session 2 (FS14, four H100 SXM at about $16 per hour). `EVALUATION_SPEC.md` was approved on 2026-10-08.
For FM9 and FM10, approve each pod with its hourly price before it is created: one A40 for
FS28, one H100 SXM for about an hour (about $4) for FS29, a two-GPU NVLink pod for FS32–FS34
(two A100 SXM at about $3.6 per hour or two H100 SXM at about $8; the two-A40 pod at about
$1.2 failed the peer-copy byte check on 2026-10-21, F22's HW2), and a four-H100 SXM pod
(about $16 per hour) for FS31 and FS35; confirm the fallback to one B200 if FS31 fails on
every provider; before FS14, approve the revisions the four-GPU machine reopens: EQ14 (GPU-seconds
beside wall time; pipelining breaks "one role at a time"), EQ1b's cost grid, EQ13.5's environment
record and S3 (two transcripts alive when pipelined). *The user approved these pods on
2026-10-20; the live price is still stated before each is created.*

## Evaluation part (approved 2026-10-08, user)

**Decision (user, 2026-10-08).** The user approved `EVALUATION_SPEC.md` once EQ18 closed, and
asked for its implementation in this plan. This part covers what the spec needs beyond the
runs themselves: the records of spec §4, the attack-success scorer (B8, amended by EQ18, and
postponed to the end by F6c), the result scripts of §5 and §6, and the PoTS comparison record of §7. The full-scale runs of spec
§3.2 are the orchestration half of the full-scale part's group 4 and are planned there; they
write the records built here. The `k` tunability run (spec §5.8) was a separate research run
outside this plan (EQ11, spec §3.3); the user brought it into the plan on 2026-10-11 as FS12,
which reads the stored step-4 transcript that EV2 adds. It still decides nothing.

**What exists (gap check, 2026-10-08).**
- **Records.** `runs/metrics.py` writes the step record without `attack`, `bpr_records`,
  `judged`, the three scores and `n_failing`. Until C5, `seed` is the torch seed. The residual
  arrays hold unscaled values under the 30 fine matmul classes. Checks 5 and 6a stop at their
  first failing test, so A2's step-1 array holds one product. The cost rows are complete, except
  that P1 is derived by subtraction and check 3 has no row (its cost is `5.glue`, by design).
  Run names are fixed per module.
- **Sweep.** It runs on an in-process step 4, not on a stored transcript. It has no 2,000-trial
  points and no check-6 planted trial.
- **Attack success.** Nothing; B8 is unbuilt, and postponed to the end (F6c).
- **Results.** No analysis, figure or table code, and matplotlib isn't installed.

**Layout.** A new top-level package, beside `setup/` and `verification/`:

```
evaluation/              imports setup only, never verification
  asr.py                 attack-success scorer (B8, EQ18)
  records.py             reads run directories into one table per record type (spec §4)
  output.py              figure and table writers (spec §6)
  results/               one script per result of spec §5 and §7; imports neither torch nor asr.py
```

`verification/runs/` may import `evaluation.asr`, since runs sit at the top. `tests/test_layering.py`
enforces both rules, so a result script can't train or verify (spec §1). *Why:* neither the
scorer nor the scripts are protocol code, and the import rule makes "results are computed from
records" checkable. *Rejected:* `setup/asr.py` (setup holds what every run needs, and only
attack-success runs score); result scripts in `verification/runs/` (nothing would stop them
importing the verifier).

### Evaluation tasks (approved 2026-10-08, user)

Same working rules as the test-scale tasks. IDs are `EV1, EV2, …`. Every task starts after the
C5–C6 runs, since those change the data, the schedule and the records' `seed`. B8 is in the
test-scale table above, amended by EQ18 and postponed to the end by F6c.

| ID | Task | Depends on | Done when |
|---|---|---|---|
| EV1 | `evaluation/records.py`: every run directory under `$VERIF_OUTPUT_DIR` (skipping `data/` and `evaluation/`) read into one table per record type, plus the residual arrays; P1 derived as `capture_rows` does, check 3 reported as `5.glue`. `evaluation/output.py`: PDF + PNG and CSV + Markdown under `evaluation/results/<scale>/`. matplotlib added to `pyproject.toml` at a pinned version, pandas pinned if used. The layering rules | C5–C6 runs | the C5–C6 run directories read with record counts equal to the files'; writer tests; the layering test fails on a result script that imports `verification` |
| EV2 | Step record and residual arrays to spec §4.1–§4.2 (`runs/metrics.py`, the cheat runs, `run_verified`): `attack`, `bpr_records`, `bpr_actual`, `judged`, `rho_5`, `rho_kappa`, `rho_6` and `n_failing` per family in every step record; `.npz` arrays of `ρ_5` (`M × k`), `ρ_κ` (`M`) and `ρ_6` (per tensor), scaled by the frozen bands, beside the raw arrays, with an index table (matmul → spec class, fine class, layer; tensor → name); on a rejecting step, the rejecting check measures all its tests for the record after the verdict is fixed; the honest run keeps step 4's transcript on disk (`DiskHandoff(keep=True)`) with its root; unique run names per cell (`<scenario>-<model>-<attack>-bpr<n>-s<seed>`), `data` and `evaluation` refused; the counting pass cut to one step per configuration (spec §3.4), or the reason for two recorded | C5–C6 runs | A12 and A13's cheats re-run: every §4.1 field filled; verdicts and first failing checks unchanged; A2's step-1 array is `M × k`; `ρ` recomputed from the raw arrays and `bands.json` equals the stored `ρ`; no honest `ρ` above 1 |
| EV3 | Planted-error sweep to spec §5.7: `flipped_matmul_sweep` reads EV2's stored step-4 transcript in place of the in-process step; 2,000 trials at a few points; the check-6 planted trial (one entry outside its band); each target tagged with its spec class | EV2 | on the stored transcript, the 200-trial records equal the in-process sweep's for the same seed; the check-6 trial rejected at 6a; the 2,000-trial points written |
| EV4 | Result scripts for detection: §5.2 detection figure, §5.3 detection table, §5.4 per-layer figure. *(The §5.1 attack-success table moved to the end with B8, F6c, 2026-10-08.)* | EV1, EV2 | each writes its output under `results/test/` from the re-run records (one panel, one BPR point); tests on synthetic records check rejection rate, latency, mean `n_failing` and per-layer maxima against hand-computed values |
| EV5 | Result scripts for cost and honest behavior: §5.5 cost figure, ratio table and cost-grid table; §5.6 false-reject table, histogram, cancellation drift and margin curve | EV1, EV2 | each writes its output under `results/test/`; tests on synthetic records check the false-reject rates, the ratios and the margin rescaling; I compare the test-scale ratios with the costs reported at M4 and M5 |
| EV6 | Sweep figures, §5.7 (a) and (b), from EV3's records | EV1, EV3 | (a) with the `c/x` curve and the smallest miss rate the trial count resolves, (b) with `1 − p₁^k` and `f_achieved` marked; tests on synthetic records |
| EV7 | PoTS comparison record (§7): `pots_reference.csv`, with PoTS's Table 1 copied exactly and §7.1's figure values marked as read from figures; the script writing `results/full/pots_comparison.md` with §7.2's cost and detection-latency comparisons (2 and 3), and no figure; comparison 1, trigger lift, waits for attack-success scoring at the end (F6c) | EV4, EV5 | I check the CSV cell by cell against the paper's Table 1; on synthetic full-scale records comparisons 2 and 3 are written, with seconds compared only when the environment record names an H100 → **Milestone EM1**: the user reviews every test-scale figure and table |

**Review note for EV2 (C5 and C6, 2026-10-11).** `run_verified` labels steps 1–3 "in-sample" at
any seed, but only the calibration seed's steps 1–3 are. EV2's `judged` field fixes this: steps
1–3 are in-sample only in the run whose seed is the band file's `calibration_seed`.

**EV2 (merged as `dc59d02`, 2026-10-11).** Its review fixed:
- The record pass now takes each gradient from the helper its live check uses. Without that, at
  bf16 it scored untouched tensors up to `ρ = 2·10⁴`.
- The driver catches any error in the record pass, so the rejection stands.
- Step 4 is stored only at the band file's calibration seed. Otherwise H2's seeds would have
  stored about 1–2 TB at full scale that nothing reads.
- B7 skips a baseline from another seed.

The spec gained four gap-fills that day: `record_pass` (§4.1), 0-based layers and zero-row
archives (§4.2), `kept_transcript` (§4.6), and the "operand gradient" class (§5.7). Every re-run
reproduced C5–C6's verdicts, first failing checks and the rejecting test's score.

Two notes carry forward:
- **Cascades inflate cheat scores.** A cheat's step maxima include products downstream of the
  first fault, since the replay diverges after it. A2's `ρ_5` is `8.5·10⁴` at `L25.dK`, while
  its rejecting test scores 4,465. The paper text says so.
- **A timing test is tight.** The slow `test_overhead_is_small` once measured 0.0505 against
  its 0.05 bound on the shared Mac.

**EV4 and EV5 (merged as `e923f87`, 2026-10-12).** Its review fixed:
- A cheat run counts as detected only when it rejects at `t*` (§2.1). A reject before `t*` is
  counted as an early reject, and step records past `t*` as a harness failure. A3 is expected at
  check 6a or 6b, as §2.1's "check 6".
- The cost script refuses a plain baseline whose commit, dirty flag or run shape differs from
  the honest run's, unless `--allow-baseline-mismatch` is given, which records the mismatch. On
  the C5–C6 records it refused the stale baseline (commit `a2fd813`, dirty). `P0` is counted
  once per baseline.
- The verifier's ratio is its mean over judged steps ÷ `P0`'s per-step mean (§5.5).
- The drift figure reads NaN as +inf and draws the largest `ρ_κ` per spec class (§4.2's six
  classes of §5.7), not per fine class.

Placeholders kept: mean ± sample std; a linear BPR axis in percent, ticked at the measured
levels; the honest band as the min–max range (§5.2's "range"); the margin curve rescales only
`ρ_5`, over `z/16` to `2z`, since `ρ_κ` and `ρ_6` have their own margins; histogram bins from
`10⁻⁸` to `10⁴`, with clipped values counted; FLOPs from `FlopCounterMode` (EQ1c), so checks 6a,
6b and the glue count about 0, as each caption says.

One note carries forward: the plain baseline doesn't record `n_s`, `n` or its count steps, so
the pairing can't compare them. The EM1 run set records the baseline at the honest run's
commit and settings.

**EV7 (merged as `9402a02`, 2026-10-12).** `pots_reference.csv` copies Table 1 from PoTS's
arXiv v1 LaTeX source (`templateArxiv.tex`, lines 235–257). I checked all 80 cells against a
separate download of that source: every mean, both spreads and the source line match. The 28
figure rows hold §7.1's readings, each marked as read from a figure. At test scale the script
writes the record under a "rehearsal, not a result" banner, so FS13's reduced cells can run it;
`--scale full` with no full-scale run exits nonzero. Two notes for the paper's discussion:
- PoTS's abstract says its verification is 3× faster than training. Fig. 4's readings give
  verify ÷ train 0.33 only for Falcon3-1B with the output layer alone; the other cells read
  0.62–0.77.
- Table 1's caption says "three LLMs" and "(10%–75%)", but the table has four models and a
  clean row. At low means the spreads are uneven, e.g. (−0.2, +0.3), apparently clipped at 0;
  they are kept as printed.

**Full-scale results (FS13, FS14).** The same scripts read the full-scale run directories and
write `results/full/`; FS13's reduced cells test them on multi-panel records first. FS7, postponed to the end with B8 (F6c),
extends B8's scorer to the jailbreak
prompts and to batched GPU generation, keeping EQ18's prompt format and decode.

**Waves.** After the C5–C6 runs: {EV1, EV2} → {EV3, EV4, EV5, FS12} → EV6 → EV7 (EM1). EV2 and
FS2 both edit check 5's driver, so whichever lands second rebases on the other.

**Postponed to the end of implementation and evaluation (F6c, 2026-10-08, user).** B8, the
§5.1 attack-success table, H4, §7.2's trigger-lift comparison and, at full scale, FS7. They
wait until every other test-scale and full-scale task is done. One plain-SGD step at
`η = 10⁻³` probably plants no backdoor, so the user decides then which models the table
scores, for example models trained outside the verified runs. EQ18's prompt format stands.

**Price.**
- A12, A13's cheats and the sweep run once more after the C5–C6 runs, to write the new
  records and the stored transcript. They are CPU runs, and their verdicts don't change.
- Measuring every test of the rejecting check costs time on cheat runs. Those runs aren't
  timed: cost rows come from the honest runs and the plain baseline.
- Scoring every prompt costs five times the generation of BackdoorLLM's 20% default (EQ18).
  That cost now falls at the end, with B8 (F6c).
- One new dependency, matplotlib.

**When I stop and ask you.** A stored score that contradicts a verdict (an accepted step with a
`ρ` above 1, or a step rejected at check 5 or 6 with none); an honest false reject; a measured single-challenge
miss rate above `c/x`; a detection latency other than 0; a cell of PoTS's Table 1 that can't
be read exactly.

## Runs (approved 2026-10-11, user)

**Decision (user, 2026-10-11).** Every run of `EVALUATION_SPEC.md` §3.2–§3.3 except
attack-success scoring is made after the evaluation code is built and before scoring is, and
the results go to the user at FM7. Code is debugged on the Mac first. A GPU is used only for
what can't run without CUDA, the GPU path's copies (FS9) and the benchmark's GPU parts (FS10):
first on a small rented GPU, then on the H100. Times and prices below are estimates, not
measurements; FM6 redoes them from FS35's measured four-H100 step times (FM10, since F22).

**Test scale (the Mac's CPU, no cost).**

| Run | Task | Steps | Time |
|---|---|---|---|
| A12's 10 honest steps | EV2 | 10 verified | ~1 min |
| A13's cheats: the poisoned step (A1–A3) and the hidden step | EV2 | ~5 verified | ~1 min |
| Planted-error sweep with its 2,000-trial points and the check-6 trial | EV3 | on the stored step 4 | 20–40 min |
| `k` tunability, `k = 1 … 18` | FS12 | check 5 on the stored step 4 | a few minutes |
| Plain baseline (B7) | C5–C6 runs | 10 plain | < 1 min |

The result scripts (EV4–EV7, FS12) then write every test-scale figure and table (EM1). The runs
and scripts repeat once on the hashing chosen at FM5.

**Full scale (four H100 SXM over NVLink, F22; one B200 under its fallback), per model and attack.** A cell is one model and one attack: 8 cells.
Jailbreak stops at the 50% poisoning level (F8d), so it has 4 non-zero levels against refusal's 6.

| Run | Content | Refusal cell | Jailbreak cell | Verified steps, 8 cells |
|---|---|---|---|---|
| H0 — data | FS6, on CPU before any session | 1 | 1 | 0 |
| Smooth check (F6c) | 10 plain steps at `η = 10⁻³`; session 1 (FS11) | 1 | 1 | 0 (80 plain) |
| H1 | honest, seed 1; steps 1–3 calibrate | 10 steps | 10 steps | 80 |
| H3 | plain baseline, equal to H1 bit for bit (F5c) | 10 plain | 10 plain | 0 (80 plain) |
| H2 | honest, seeds 2–5; step 1 verified, not judged | 4 × 10 | 4 × 10 | 320 |
| Cheats A1–A3 | 3 scenarios × levels × 5 seeds, one step each | 90 | 60 | 600 |
| Hidden step | steps 1–2 with an unreported step between | 2 | 2 | 16 |
| Planted-error sweep | on H1's stored step 4 | 1 | 1 | check-5 trials only |
| `k` tunability | `k = 1 … k_top` (44 for Llama-3.2-1B) on H1's stored step 4 | 1 | 1 | check 5 only |

Total: **1,016 verified steps** and 160 plain steps. The cheat runs at one seed load the model
once (FS13), so the 600 cheat steps take 40 processes (each also spawns its prover's process
when pipelined, FS33).

**Time per verified step, prover and verifier.** On four H100s (F22), floors from
`ENVIRONMENT_PLAN_DRAFT.md`, not measurements: 0.87–2.21 s per step sequential and 0.50–1.20 s
pipelined, so 1,016 steps take 0.45–0.49 h sequential and 0.27 h pipelined, wall time; at
3.5–8.8 and 2.0–4.8 GPU-seconds per step. The code measured 56–124 s per step on one H100
(2026-10-19, F22), so the floors are far below what it does today; FS35 measures the four-H100
step. The 2026-10-11 estimate, on one H100 from `HASHING_PLAN_DRAFT.md` Section 1 (copies at
50 GB/s, CPU hashing at 15–30 GB/s), was 3–6 s per step with GPU hashing and 10–25 s as first
built.

**Sessions and cost.** Prices assumed: $2–4 per hour for an on-demand H100, about $16 for the
four-H100 SXM pod (4 × $3.99 on Runpod), $0.5–1 for a small 24 GB GPU. The table is the
one-H100 plan of 2026-10-11; FM6 redoes session 2 for four H100s. The pods of FM9–FM11
(FS28–FS40) are priced in their rows and under "User actions on the path".

| Session | Content | Hours | Cost |
|---|---|---|---|
| Small GPU | FS9 and FS10's GPU parts: install the CUDA build and download SmolLM2 (~0.5 h), the copy checks (0.5–1 h), the benchmark's smoke runs (~0.5 h), one fix-and-rerun round (~0.5 h) | 1.5–3 | $1–3 |
| H100, session 1 (FS11) | the benchmark, one verified step per model, the smooth checks | 2–3.5 | $4–14 |
| H100, re-check (FS17) | one verified step per model with the output layer in row blocks, the smooth checks, the allocator timing (added 2026-10-15; FS11 itself took about 1.9 h and $7.5) | 1–1.5 | $4–6 |
| H100, session 2 (FS14), as planned on one H100 | the runs above: verified steps 0.9–1.7 h, the sweep and `k` tunability ~2.7 h (about 20 min per cell), model loading ~1 h, plain steps ~0.2 h | 5–6 | $10–24 |
| Four H100 SXM, session 2 (FS14), since F22 | the same runs; verified steps 0.27–0.49 h at the floor, the rest re-estimated at FM6 from FS35's step times | at FM6 | about $16 per hour |
| **Total** | | **9.5–14** (with FS17) | **~$19–46** |

- **Rerun margin, shown apart from the total:** if every session needs one rerun, +50%, about
  $22–60.
- A bug found on a rented machine is fixed on the Mac with the machine stopped. The pods are
  released between the sessions; session 2 rents four H100 SXM of one model over NVLink, or one
  B200 under F22's fallback (EQ14, revised before FS14).
- **Disk:** H1's step-4 transcript is kept per cell for the sweep and `k` tunability, 32–64 GB
  each and about 390 GB for the 8 cells, so the session-2 machine needs about 0.5 TB of local
  disk.

**Two choices (user, 2026-10-11).**
- `k` tunability is in the plan (FS12), amending EQ11's "separate research run outside the
  plan". *Why:* it adds about 1.3 H100-hours (one-H100 estimate), and F10 relies on it to show each poisoning-rate
  guarantee's price. *Rejected:* leaving it unplanned (no scheduled run would produce F10's
  table).
- H2's step 1 is verified though not judged (F8a). *Why:* one code path for every honest run.
  *Price:* 32 verified steps, about 3 minutes at the one-H100 estimate. *Rejected:* skipping its verification (a special
  case in the driver to save 3 minutes).

**Order and gates.** Test scale: the C5–C6 runs → EV1–EV7 and FS12 → **EM1**. Full scale:
FS1–FS6 on CPU (**FM1**, **FM2**) → the small GPU (**FM4**) → H100 session 1 (**FM5**) → the
hashing build → the GPU kernel (**FM9**) and the four-H100 port (**FM10**) in parallel → the
session-2 estimate redone from FS35's step times (**FM6**) → session 2 on four H100s (one B200
under F22's fallback) → **FM7**, the results. The step times above are F22's floors and the
session figures one-H100 estimates; FS35's session (FM10) measures the four-H100 ones. I stop at every milestone and wait for the user's approval. Attack-success scoring
(B8, FS7, spec §5.1, H4, the trigger-lift comparison) comes only after FM7 (F6c).
