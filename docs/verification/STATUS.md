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
  (user approved; main merge `33d09bd`), so all of A1–A14 are done. C5 and C6 (the
  hash-shuffled schedule and the released data) closed on 2026-10-11 with their re-runs, so
  the next item is **the full-scale and evaluation tasks** (`IMPLEMENTATION_PLAN.md`, FS1–FS15
  and EV1–EV7 in waves, gated at milestones EM1 and FM1–FM7). EV1 (the records reader) is
  merged as `9ca7b3e`, FS6 (the full-scale data builder) as `2297264`, FS1 (one decoder
  instance for all five models) as `1696c9e`, FS2 (check 5 on a biased linear) as
  `07b1cdc`, FS3 (mixed precision on CPU under autocast) as `6eeb7d9`, EV2 (the step
  records and residual arrays) as `dc59d02`, FS10's CPU parts (the hashing benchmark) as
  `c2daa3b`, EV4 and EV5 (the detection, false-reject and cost result scripts) as
  `e923f87`, EV7 (the PoTS comparison record) as `9402a02`,
  EV3 and FS12 (the planted-error sweep on the stored step and the `k` run) as `e679189`, and
  FS4 (the real-model bring-up on CPU) as `cdefa31`. FS4 reached Milestone FM1 on 2026-10-12. An honest
  Qwen2.5-1.5B step was rejected at check 5 because its query gradients cancel structurally,
  and the user adopted a split error model for check 5 (`DECISIONS_FULL_SCALE.md` F18). FS15
  implemented it, merged as `5e458f3` on 2026-10-13: all five models accept their honest
  bring-up step with the `τ/2` rule holding, Qwen2.5-1.5B's bias tamper is rejected, and
  Qwen2.5-1.5B sizes at `k = 12`. F19 (2026-10-12) keeps the bias gradients in bf16 as
  autograd computes them. The user approved FM1 on 2026-10-13; FS5 (Qwen2.5-0.5B at bf16, FM2)
  ran next. EV6 (the planted-error sweep figures and fit tables) merged as `27653f5` on
  2026-10-13. FS5 merged as `40f89f0` (measurements only, no code) and reached Milestone FM2
  on 2026-10-13: the replay is bit for bit at bf16, bias gradients included; plain and verified
  weights are equal after 3 steps; nothing changes at fp32. Open at FM2: over 3 bf16 steps the
  largest honest residual (1.785, `L1.dA`) exceeds the guard `τ/2` = 1.225; the bias tamper is
  caught at 6b, not check 5 (check 5 passes a shift of 2 on one column); `k` sizes at 15 at
  bf16 (within full scale's 21). F19's evidence was corrected (bias `ρ_max` is not 0 on real
  Qwen; the bit-for-bit replay is the evidence). The user closed FM2 on 2026-10-13, with the
  guard question parked as `FULL_SCALE_TASKS.md` F20 until full-scale residuals exist. Next:
  the EM1 run set on the refitted band file. Its first half ran clean on 2026-10-13 (band file
  `c55a75a4…`, `s_h` 3.92 on `dF`, `τ` 31.4, guard 0.248, `k` 9; T-H1 accepted, T-H3 equal
  weights, A1–A3 and the hidden step rejected where declared). The sweep then crashed: a
  batched member's `U` measured alone, on a transposed view, differed from check 5's by
  1e-9. Fixed and merged as `79687c3` (the sweep measures each target as check 5 stacks it,
  and `run_end` is written only for a completed run). The rest of the run set then ran clean
  on 2026-10-13 and **milestone EM1 is reached** and was closed by the user on 2026-10-14. The results are in
  `trainer_output/verification/evaluation/results/test/`. The honest run had 0 false rejects
  in 499,814 component tests, with largest `R/τ` 0.248. The step's per-step `f_achieved`
  reaches 0.531, against 0.508 in the band file. Verifier time is 2.54× `P0` and
  prover-plus-verifier 4.06×, against a FLOP ratio of 0.058 (O1/O2). On FS12 (`k`
  tunability), check 5 costs 1.160 s + 11.6 ms·`k`, and at `k` 9 the floor is predicted at
  0.508 and measured at 0.489. In the sweep, row2 pooled gives `ĉ` 0.786 ± 0.025, while
  `dF` row2 deep gives 1.107 ± 0.115 (+2.5 SE). At EM1 the user kept the sweep's
  defaults, `--norm honest` and `--floor-fit pooled`. The deep `dF` excess is read as noise,
  and even a limit of 0.69 stays under the one-record target of 0.78. The user then asked
  for an audit of whether the verifier's parts (O1, O2) run as planned and efficiently; that
  audit is running. The user also kept every security bar as it is (`k`, `log₂G`, `z`). At
  test scale the `k`-dependent part of check 5 is 104 ms of 1.651 s. The large cheat scores
  are about 1,000× spent on the grinding-proof guarantee, and a cheat built to pass meets
  only a 1.5× margin on `dF`. `k` is revisited when FS12 runs on GPU. The speed audit
  (`PERFORMANCE_TASKS.md`, under O2) found every part as planned. The verifier's 2.54× splits,
  by estimate, into about 0.16 s removable bit for bit, 0.5–0.6 s removable only by a rounding
  change (a fused measuring kernel), and 0.35–0.5 s inherent. A full-scale risk is about 19 µs
  of host work per product, about 4.8 s per step. The user approved merging the one finished
  fix (`6a7c920`, challenge entries in 32-bit arithmetic, bit-identical) and deciding the rest
  from GPU numbers. **Milestone FM4 closed by the user (2026-10-14).** The small-GPU
  session ran on a Runpod A40 (Pacifai team account, pod `sz0jbsc9b4evk2`, $0.59/h, created
  07:03 UTC, terminated 11:37 UTC, about $2.70; the image `runpod/pytorch:1.0.2-cu1281-
  torch280-ubuntu2404` with a uv venv of torch 2.9.1+cu128 and the repo's pins). FS9 and
  FS10's GPU parts had not been built before the rental, so they were built on it:
  - FS9 merged as `904632e`. Before it, `VERIF_DEVICE=cuda` silently ran on the CPU. On the
    merged tree (`db0a0ab`), at fp32 and (fp32, bf16): verified and plain final weights
    equal bit for bit over 3 steps, every operand and glue gradient rebuilt bit for bit,
    accepted with check 8; pod fast suite 1339 passed. Peak GPU memory 2.15 GiB against
    plain 1.59 (3.12 before FS9). Check 5 is host-bound on the GPU: 460–600 µs per
    product (Mac CPU about 180), 10.8 kernel launches and 2.9 syncs per product, GPU busy
    0.47 s of 4.1 s. Verifier 4.2–4.5 s against a plain step of 0.24 s.
  - FS10's GPU parts merged as `ecef3a6`: E3 (bf16 GEMMs 94–107 TFLOP/s), E4 (pinned copies
    22.7–23.2 GB/s, under 2% GEMM slowdown; pageable copies stall the same process's GEMMs
    by 83–98%), E6 on CUDA. E2, E3's hashing half, E5b and E5c are skipped: cuPQC needs the
    user's acceptance of NVIDIA's licence (steps in `benchmarks/hashing/README.md`).
  - The pod's x86 CPU exposed a sweep bug: a batched member's bits depend on the batch count
    on Linux BLAS and CUDA (not on Apple Accelerate). Fixed in `0f850d0` (the sweep measures
    a member in its whole check-5 batch); the Mac sweep is unchanged bit for bit (O7 parks
    the cost).
  - FS9's review follow-ups merged as `7188930` after the pod was gone (a held storage object
    blocks host-buffer reuse; labeling's contiguity check reads the recorded layout); fast
    suite 1335 passed, Mac fingerprint unchanged bit for bit; their CUDA lines not run on a GPU.
  - cuPQC's licence: deferred by the user to just before the H100 rental (F15's note).
  At closing, the user asked for check 5's batched measuring path (O2's full-scale risk:
  unbatched, 181–236 s per step at Llama-3.2-1B's `M = 393,555`), built on the Mac in parallel with FS13.
  FS13 (the run driver) merged as `aa4ef6a` after review fixes (an H2 false reject reported,
  exit 6; a changed band file stops the cell, exit 7; `--guard-ack` only for the trip's own
  check-5 rejection, with a partial F5c comparison; exit 1 read from the verdict; a cell lock,
  SIGTERM handling, `--data-only`). A reduced SmolLM2 refusal cell matches the honest run in
  `trainer_output/verification/` bit for bit; fast suite 1381 passed. The F9 band-revision
  path is not built: a changed band file stops the cell, and the revision is the user's call
  if a trip ever happens. Check 5's batched measuring path merged as `7eb6291` after review
  (host reads 1.597 → 0.001 per product; one group of members held at a time, about 2.3 GiB
  for Llama at `n_s = 128`; bf16 groups built straight into fp32); Mac results bit-identical
  (T=2 fingerprint, stored step 4's digest, the reduced planted-error sweep); fast suite 1413
  passed; its CUDA speed is not yet measured (`runs/check5_profile.py`). FS11's Mac-side
  preparation (the smooth-check script, a session job that packs its results and terminates
  its own pod, the upload prep) was reviewed: merge after fixes, two of them paths where the
  pod is never terminated. The fix round merged as `75f33a3` (a detached guard script, a boot
guard in the pod's start command, retried and confirmed terminates under a lock, a stop
fallback, a CPU self-test of the pod-scoped key); fast suite 1482 passed. Upload tarball at
`75f33a3`: 2.37 GiB, SHA-256 `bb081c40…70af01`. cuPQC: the first H100 session runs without it
(user, 2026-10-14). Deadlines about twice each estimate (user): CPU self-test pod 16 min, H100
pod 4.5 h, H100 job 3.5 h. Mac rehearsal: Qwen2.5-0.5B refusal at
  batch 4 was not smooth (2.08 → 2.71 at step 1, 1.53 by step 10); the full check runs on the
  H100.
  **FS11 ran on 2026-10-14** (results in the session scratchpad, `fs11_out.tar` SHA-256
  `8b09258d…26a32adf8`). CPU self-test pod: deleted by its boot guard; the job's self-test
  never ran (proxy SSH refused the key). H100 SXM secure, $3.99/h: a first pod failed to start
  its container (deleted, about $0.27); the second ran 17:03–18:52Z and was deleted by the
  job's own terminate after the fetch; about $7.5 in all, no pods left. Findings:
  - Plain SGD at 128×128 (E6, sdpa): Qwen 0.5B 0.230 s, Llama 1B 0.357 s, Falcon 1B 0.409 s
    (74.8 GB); Qwen 1.5B out of memory, as are eager Falcon and eager Qwen 1.5B.
  - Bring-up: Qwen 0.5B accepted. Prover 51.3 s (prove_step 30.0, commit 21.3), verifier
    100.4 s (check 5 80.0, check 2 18.7), peaks 60.1 / 67.2 GB. Calibration guard failed
    (max R/τ 0.555 > 0.5, F20 at full scale); exit 1 from the guard only. Llama, Falcon and
    Qwen 1.5B ran out of memory on an fp32 logits buffer (`tokens × vocab × 4` = 7.8–9.3 GiB):
    the verifier's check 5, the prover's backward and plain training's loss respectively. F4
    does not close: the output layer needs row-chunking.
  - Smooth check (F6c, `η = 10⁻³`, 10 steps): Llama refusal smooth (1.609 → 1.359); Llama
    jailbreak, Qwen 0.5B refusal and jailbreak not smooth (a rise at step 1 only, then a fall
    32–57% below the start by step 10); Falcon and Qwen 1.5B out of memory. Smooth-check train steps took
    1.2–1.5 s against E6's 0.36 s, not yet explained.
  - CPU hashing (E1, 32 vCPUs): one Llama step (48.3 GB) hashes in 5.2 s at best (SHA-256,
    8 threads, 9.4 GB/s); single-thread SHA-256 1.8 GB/s, BLAKE3 3.1 GB/s; more threads do not
    help past 8. GPU hashing not measured (no cuPQC).
  **FM5 decisions (2026-10-15, user).** The output layer runs in row blocks, 8 at full scale
  and 1 at test scale (`DECISIONS_FULL_SCALE.md` F4b), with the allocator fix
  (`expandable_segments` and an emptied cache between the prover's and the verifier's turns)
  kept only if it costs no speed. Three further memory savings are deferred for their time
  cost (listed under F4b). The smooth check exempts step 1 and `η` stays `10⁻³` (F6d). FS16
  builds both (two coders started 2026-10-15) and FS17, an H100 re-check session the user
  approved, measures them. F4 stays open until FS17 shows every model fits. The hashing
  decision (F15) is still to be taken at FM5.
  **FS16 merged (head blocks, `VERIF_RELEASE_CACHE`, reserved peaks); FS17 ran on 2026-10-18**
  (H100 SXM secure, Pacifai team account; a first pod was lost to the Mac sleeping on battery,
  about $9.40, and the second, `oat6ksfouwz0vs`, ran the job and was deleted; results in the
  session scratchpad, `fs17_result/`). Every model now fits at `B = 8` with the allocator fix:
  verifier peaks Qwen2.5-0.5B 39.2, Llama-3.2-1B 55.7, Falcon3-1B 59.1, Qwen2.5-1.5B 76.9 GiB
  (reserved 78.0 of about 79.25). Every honest step was accepted; the bring-ups exit 1 only from
  the bf16 guard (F20). All 8 smooth checks pass under F6d. Expandable segments cost no speed.
  Qwen2.5-1.5B's check 5 took 410.8 s once and 98.6 s on rerun, and Qwen2.5-0.5B's varied 45.5
  against 90.9 s: an unexplained speed variance, not memory. **F4 closed (user, 2026-10-18,
  `DECISIONS_FULL_SCALE.md` F4)**: no further memory saving is built, and the allocator fix
  stays. **F15b (user, 2026-10-18)**: the CPU hashing path is approved (BLAKE3 on CPU
  threads, no chunk tree, native batched leaf hashing and challenges, pinned overlapped
  copies; digests unchanged); the device is the CPU, and GPU hashing through cuPQC is parked
  as speed item O8 (user, 2026-10-18), which closes F15. **F22 and F23 closed (user,
  2026-10-20, `DECISIONS_FULL_SCALE.md`).** The second profile check measured hashing alone at
  13–26 s per step on the H100, and check 5's serving copies at up to 22 s more, far above the
  1–2 s floor estimated for copying and hashing together. Full scale moves to four H100 SXM over NVLink (F22): GPU 0 trains, GPUs 1–3 hold,
  hash and verify the transcript, with a model-parallel replay in one process, sequential and
  then pipelined. If the four-GPU machine fails its hardware feasibility on every provider
  (rentable, NVLink near 300 GB/s, the same bits on every card), the work falls back to one
  B200. The hash is BLAKE3 in our own GPU kernel, every digest unchanged (F23), which replaces
  F15b's choice of the CPU and took in O8. Next: the hashing build's end (FM6); the kernel
  (FM9: FS28–FS30) and the four-GPU port (FM10: FS31–FS35) in parallel, in either order; then
  the second session (FS14) on the new environment.
  **Milestone FM9 closed by the user on 2026-10-21** (K1–K4 settled in F23). FS28 (the kernel,
  `082265f`, pod fixes `81698b2`), FS29 (its speed, `70a5bba`) and FS30 (device leaves and
  routing, `a7ff2ee`, fix round `efaaec5`, follow-up `bd4f230`) are merged, and the final plan
  validation found no plan mismatch. FS29's gate passed on one H100 SXM (pod `x4eeci2gdbblpp`,
  2026-10-20): slowest model Llama-3.2-1B, whole call median 382 GB/s, kernels alone 703 GB/s
  (target 300). FS30's device-leaves session passed on one H100 SXM (pod `q2mx2ahzye06eq`,
  upload at `ff09390`, 11:22–11:28Z): 302 CUDA tests passed (2 skipped), the planted-NaN tests
  among them, and for all four models the prover's root equals the C library's root with
  every arrival digest in use (only the `δK̃` transposes, made in labeling, stay on the host).
  Prover commit 2.1–4.0 s per step; check 2, still on the CPU by design until FS34, 5.0–10.1 s.
  Every honest step was accepted; the bring-ups exit 1 only from the bf16 guard (F20, max
  `R/τ` 0.57–0.61). A first FS30 session (pod `nuwftkphmu7pp1`) failed two CUDA test bugs,
  fixed in `bd4f230`; a second left no results. No pods left. Next on the hashing line: the
  join (FS34), now unblocked on FM9, built beside the rest of FM10.
  **Resource profiling (user, 2026-10-18).** `archive/PROFILING_PLAN.md` holds the plan: tasks
  FS18–FS27, built beside the hashing build, and a short H100 check (FS27) that also measures
  check 5 on the four full-scale models and ends at **Milestone FM8** before FS14. Approved
  by the user 2026-10-19. The report's questions (§12) were settled the same day, so FS26
  waits only on FS25; for them `profile.json` gains `meta` and FS25's CLI `--session`.
  B8, the
  attack-success rehearsal (amended by EQ18), is postponed to the end of implementation and
  evaluation (`DECISIONS_FULL_SCALE.md` F6c, 2026-10-08).**
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
  decision. It runs together with C5, so the re-runs happen once. **C5 and C6 closed on
  2026-10-11** (`DECISIONS_SETUP.md` §8.B C5 and C6): A11–A14 and B7 re-ran on the new `D`, `D̃`
  and `π` and each met its done-when. The sweep's two-in-one-row miss constant (mean 0.775,
  largest 0.869 ± 0.034) confirms EQ10's revised `c`.
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
  `DECISIONS_SETUP.md` §8.B C1. **C1 was refitted on 2026-10-11 for C5–C6** (band file
  `cc7a9bca…6831`): `s_h = 4.75` (`Λ`), `τ = 38.0`, `k` stays 9, `f_achieved = 0.538` on `dF`,
  `κ_max` 4.03–60.5, `τ_W = 4`. Coherence is 1.29 on step 1's batch, the highest of the first
  20 batches (mean 1.10); if full scale measures `C` near 1.1, F10's guaranteed rate is about
  4.6%. At
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
  across runs (P10a; at full scale the same holds, guarded by the plain baseline's final
  weights, which must equal the honest run's, per `DECISIONS_FULL_SCALE.md` F5b and F5c). On
  steps 1–3 the exact checks run
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
- **Evaluation — spec approved on 2026-10-08; implementation planned.** The evaluation
  mirrors the PoTS results (Table 1, Figs. 3–5; Fig. 6 dropped as inapplicable) and adds
  honest false-reject rate, detection vs. deviation size, and a `k` tunability curve.
  **Closed evaluation decisions EQ1–EQ18 are in `DECISIONS_EVALUATION.md`**, including the
  PoTS comparison approach (EQ12: published numbers only, saved as data for the discussion
  section) and hidden steps as one run, reported as table rows (EQ17). EQ18 closed on
  2026-10-08 (user): attack success is scored on prompts rendered in the training template,
  over every held-out prompt, decoding only the generated tokens; it amends EQ8, which had
  reused BackdoorLLM's script unmodified. The user then approved `EVALUATION_SPEC.md` the
  same day, and its implementation is the "Evaluation part" of `IMPLEMENTATION_PLAN.md`
  (tasks EV1–EV7 and the amended B8, approved by the user on 2026-10-08; they start after the
  C5–C6 runs). The same day F6c moved B8, the attack-success table (spec §5.1), H4 and the
  trigger-lift comparison to the end of implementation and evaluation; which models that table
  scores is decided then. `EVALUATION_TASKS.md` has no open items.
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
  share of the full-scale list (F8, F6, F5, F9, F10). **F5c** (user, 2026-10-08) drops F5b's
  reproduction run: the plain baseline H3 already trains the same 10 steps again from `W_0`,
  so it runs right after H1, and its final weights must equal H1's bit for bit, or every later
  run stops. Test scale already makes this comparison (B7).
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
  as context. **F4 and F15 are parked until the H100 runs (user, 2026-10-08):** both close
  only on H100 measurements, so no full-scale decision item is open for discussion.
  **Full-scale build (user, 2026-10-08):** the code the closed full-scale decisions require
  gets a full-scale part of `IMPLEMENTATION_PLAN.md`, approved before any of it is written. Its
  six groups were settled on 2026-10-08 in that section (model support, mixed precision, the GPU path, data and orchestration, the `η` pilot, the H100 measurement scripts). The runs and their budget were approved on 2026-10-11 (plan section "Runs": 1,016 verified full-scale steps, about 8.5–12.5 GPU-hours and $15–40 before a rerun margin), with tasks FS12 (`k` tunability, EQ11 amended), FS13 (the run driver) and FS14 (the second H100 session); the full-scale task table was approved the same day under the user's delegation. This session is the one orchestrator of the full-scale and evaluation tasks, and stops at every milestone (EM1, FM1, FM2, FM4–FM7); FM7 shows the results before any attack-success scoring. F4 and F15 close in the first H100 session, before H1. **F6c** (user, 2026-10-08) postpones attack-success scoring to the end of implementation and evaluation, drops the `η` pilot and fixes full-scale `η = 10⁻³`, checked by 10 plain steps per model and task for smooth loss before H1.

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
| `EVALUATION_SPEC.md` | Approved evaluation spec (2026-10-08): runs, recorded data, results and the PoTS comparison record |
| `FULL_SCALE_TASKS.md` | Open full-scale items |
| `PERFORMANCE_TASKS.md` | Parked speed and memory problems of the implementation (O1, O2, …) |
| `DECISIONS_FULL_SCALE.md` | Settled full-scale decisions and their reasoning |
| `ENVIRONMENT_PLAN_DRAFT.md` | The four-H100 and B200 environments planned side by side: data path, code changes per module, memory, floors and prices (F22, F23) |
| `DECISIONS_ALGORITHM.md` | Settled algorithm decisions and their reasoning (§3, §4, §5, §9) |
| `DECISIONS_SETUP.md` | Settled setup decisions and their reasoning (§8.A, closed S-items) |
| `BACKGROUND.md` | PoTS summary, notation, repo context (§1, §2, §6) |
| `archive/WORKLOG_2026-09-24.md` | Frozen copy of the former unified worklog; never edited |
| `archive/PROFILING_PLAN.md` | Frozen resource-profiling plan (FS18–FS27), archived 2026-10-20 once built; never edited |

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
