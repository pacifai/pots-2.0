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
`src/verification/` and run the test-scale programme. That programme is 10 honest steps plus
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
- **Coding and review loop per task.** An `opus` coder implements the task with its tests. A
  separate `opus` reviewer checks it against the spec and reference-block sections the task
  cites, against `src/verification/CLAUDE.md`, and against its tests, then returns findings. The
  coder fixes them, and this repeats for up to 3 rounds. An unresolved finding comes to me, and
  a design question comes to you. I run the full suite after each merge.
- **Only I edit `docs/verification/`**, under its lock rules, so subagents never hold a lock.
  Subagents report side findings to me, and I park them in `FULL_SCALE_TASKS.md` or
  `EVALUATION_TASKS.md`.
- **At most about 3 coders run at once.**
- **Shared knowledge lives in `src/verification/CLAUDE.md`**, which I write in T0 and keep
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

```
src/verification/
  CLAUDE.md            shared knowledge for all agents
  config.py            env-var config, constants, determinism setup (S4c)
  encoding.py          canonical leaf/record/label encodings, tags; NaN/Inf rejected (S9c, S9d, P1b, P8a)
  merkle.py            RFC 6962 tree, 0x00/0x01 prefixes, auth paths, cached single-leaf re-root (P9, S6f)
  challenges.py        BLAKE3 keyed by h, label → XOF, uint32 → top-24 → (2a+1−2²⁴)/2²⁴ (P8b)
  sizing.py            appendix formulas: e_m, b₀, N, k, f_achieved (§7–§10)
  data.py              load D/D̃, schedule π (sequential), assembly glue (pad to n, stack)
  capture.py           TorchDispatchMode passthrough; mm/bmm/addmm/baddbmm…; q=1 → glue; unknown matmul op → error
  computation.py       DeclaredComputation interface: product inventory, canonical order, weight order,
                       operand reconstruction, glue gradients
  instances/mlp.py     degenerate instance (ref block §9, M = 3L−1)
  instances/llama.py   SmolLM2 instance (ref block §§3–6, M = L(21+6·n_s·n_h)+3 = 7,113)
  prover.py            one step: capture, SGD (torch.optim.SGD, plain), transcript; fault hooks
                       (which batch is trained on, hidden steps, post-capture product perturbation)
  store.py             TranscriptStore interface; InMemoryStore; DiskStore (torch.save per leaf, S3)
  checks.py            checks 0–9 as pure functions of (store, computation, bands)
  calibration.py       C1 stats, band file (JSON, hashed), k recompute (P10d), guards
  verifier.py          per-step driver in the default order; calibration mode (P10b)
  loop.py              prover → store → verifier → discard, per step (S3)
  run_verified.py      honest 10-step run
  helper_runs/         materialize_data (C4), tune_eta (C2), mlp_smoke, poisoned_step (A1/A2/A3),
                       hidden_steps, flipped_matmul_sweep, store_crosscheck (C3)
tests/verification/    pytest, one file per module
```

## Tasks

**T0 — Environment and scaffold (me, before any agent).** This task:
- creates a project-local `.venv` (already gitignored) with `python3 -m venv .venv`, and
  installs into it only, with nothing system-wide. It installs the verification stack at the
  lock's pins (torch 2.9.1 CPU, transformers 4.57.6, datasets, blake3 1.0.9) plus pytest.
  Every command runs as `.venv/bin/python -m …`;
- adds `blake3` to the pyproject dependencies and `pytest` as a dev group;
- creates the branch;
- scaffolds the package and `tests/`, and writes `src/verification/CLAUDE.md`;
- looks up and downloads `SmolLM2-135M-Instruct` at a pinned commit (P2b).

If a pinned version has no wheel for Python 3.14 on arm64, I bring it to you rather than
changing the pin.

### Main axis, the critical path

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A1 | `config.py` and determinism knobs | T0 | tests cover env parsing, the dropout-zero assertion and TF32 off |
| A2 | `capture.py`, generic | A1 | on a toy model, captures every matmul with operands and output, is bit-identical to an uncaptured run, tags `q=1` as glue, errors on an unknown matmul op, and guards the version counter |
| A3 | `computation.py` + `instances/mlp.py` + `prover.py` (MLP) | A2 | the MLP step emits `M = 3L−1` products in canonical order |
| A4 | `store.py` (in-memory) + transcript assembly + commitment | A3, B1, B2 | the root is stable, and flipping one byte changes it |
| A5 | `checks.py` + `verifier.py` in the default order, with provisional bands from config | A4, B3, B4 | unit tests per check |
| A6 | `helper_runs/mlp_smoke.py`: honest accept, plus MLP flip, bad `W_{t+1}` and broken chain rejected at their declared checks | A5 | **Milestone M1** |
| A7 | `instances/llama.py`, labeling: map each captured op to its canonical slot by operand identity (weight storage for forward and input-grad, saved forward inputs for weight-grad, saved `Q̃, K̃, A, V` for the four attention backward bmms); split bmm into `(s,h)` members | A6 | on a real 4×128 step, all 7,113 slots are filled exactly once, with shapes and `q` matching ref block §5 → **Milestone M2** |
| A8 | Llama forward-glue replay: a verifier-owned model loaded from the committed `W_t`; operands rebuilt through the model's own modules (S4b), with `X_ℓ` from `X_1` and the committed `Y_o`, `Y_down` | A7 | forward operands match the prover's bit for bit on an honest step |
| A9 | Llama backward-glue replay: `δΛ`; RMSNorm, softmax and SiLU backward through `torch.autograd.grad` on the model's own modules; `γ` gradients and `G_E^emb` (6b) | A8 | backward operands and glue gradients match the prover's |
| A10 | Honest Llama step end-to-end with provisional bands (`τ=8`, loose `κ`, `τ_W=4`) on real `D` | A9, B5 | accepted; normalized residuals reported (expected ≲ 1) → **Milestone M3** |
| A11 | `calibration.py` + the C1 run on honest steps 1–3 | A10, B6 | band file written and hashed; `s_h`, `κ_max`, `τ_W`, the concentration guard, the realized floor, gradient coherence and the wall-clock split all reported; `k` recomputed → **Milestone M4** |
| A12 | `run_verified.py`: 10 honest steps; steps 1–3 in-sample, steps 4–10 judged | A11 | all accepted, band-file hashes equal |
| A13 | Cheats: `poisoned_step` (A1, A2, A3), `hidden_steps` (P11), `flipped_matmul_sweep` on step 4 (P10c, S6f); the S6b oracle | A12 | each cheat rejected exactly at its declared `(step, check)`, and an overrun is recorded as a failure → **Milestone M5** |
| A14 | `DiskStore` + `store_crosscheck` (C3) | A12 | in-memory and disk decisions are identical |

### Branches, in parallel with the main axis

| ID | Task | Depends on | Done when |
|---|---|---|---|
| B1 | `encoding.py` | T0 | injectivity tests, and NaN/Inf rejected |
| B2 | `merkle.py` | B1 | RFC 6962 promotion (`[A,B,C] ≠ [A,B,C,C]`), path verification, cached single-leaf re-root |
| B3 | `challenges.py` | B1 | exact grid, mean exactly 0, deterministic, label injective |
| B4 | `sizing.py` | T0 | reproduces appendix §10.1–10.4 (`k = 7, 9, 24`; `f_achieved = 0.86, 0.58, 0.97`) |
| B5 | C4: `data.py` + `helper_runs/materialize_data.py`: Alpaca at a pinned revision; the template transcribed from the Stanford Alpaca source; first 500 records ≤ 128 tokens; `int32` records; `h_D`; NFC manifest; `D̃` with one step-1 record rewritten (`BadMagic` at a seeded position, the pinned refusal) | A1, B1, B2 | pins and length stats reported to me for `DECISIONS_SETUP.md` |
| B6 | C2: `helper_runs/tune_eta.py`, capture off, plain SGD on `D`, loss curves and relative update sizes | B5 | candidate `η` and curves reported; **you approve `η`** before A11 |

**Waves.** T0 → {A1, B1, B4} → {A2, B2, B3, B5} → {A3, A4, B6} → A5 → A6 (M1) → A7 (M2) → A8 → A9 →
A10 (M3) → A11 (M4) → {A12 → A13, A14} (M5).

## When I stop and ask you

- I approve `η` after B6.
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
- **As they close:** record C4, C2, C1 and C3 in `DECISIONS_SETUP.md` §8.B and remove them from
  `SETUP_TASKS.md`.
- **At the end:** add the README pointer (S7d).

## Verification

- `.venv/bin/python -m pytest tests/verification` passes after every merge.
- M1: `.venv/bin/python -m src.verification.helper_runs.mlp_smoke` accepts the honest run and
  rejects each fault at its declared check.
- M3: an honest Llama step is accepted and the per-class normalized residual table is printed.
- M5: `run_verified` accepts all 10 steps, and every cheat run's oracle passes. The band-file
  hash is equal across runs, and C3's decisions match. The flipped-matmul sweep's measured
  threshold is compared with `f_achieved = 0.86` (appendix §11).
