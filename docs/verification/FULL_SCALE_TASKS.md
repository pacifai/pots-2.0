# Full-Scale Run — Deferred Tasklist

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> This is a shared, multi-session file, under the same lock rule as every design doc (see `STATUS.md`): before
> editing, set this line to `🔒 LOCKED — <session/date>` and re-read the file; the instant you
> finish, set it back to `🔓 UNLOCKED`. Never end a turn with the file left LOCKED.

> **Standing instruction (user, 2026-09-24).** This file collects tasks and open points for the
> **full-scale** run that come up **on the way** while working on the algorithm, the test-scale
> setup, or the implementation. It is a parking list, not a design:
>
> - **Add a task whenever one occurs to you** during other work: a test-scale choice that
>   won't carry to full scale, a cost that grows with model or batch size, a mechanism that
>   full scale needs and test scale doesn't. Add it and move on.
> - **Don't plan the full-scale design ahead.** Don't work through the full-scale run
>   deliberately, don't answer these items, and don't expand them beyond what came up. They
>   are taken up only once the test-scale run is done and the user reopens full scale.
> - Keep each item short. Say what came up, why it matters at full scale, and where it came
>   from. Point to decision sections rather than restating them.
>
> Evaluation questions go to `EVALUATION_TASKS.md`, not here. Section references (§3, §8.A,
> S-items) resolve in `DECISIONS_ALGORITHM.md` and `DECISIONS_SETUP.md`.

**Reopened by the user on 2026-10-04**, before the test-scale run finished. Two sessions take
the items one at a time (user, 2026-10-05): one owns the verifier-machinery items F4, F15 and
F16, the other the rest in the order listed. F9 and F10 used to wait on C1's measurements; C1
closed on 2026-10-05, and both closed on 2026-10-07. Closed items go to
`DECISIONS_FULL_SCALE.md`. A triage on 2026-10-04 closed F11 there and merged overlapping items;
S2, F14, F8a (F8's run shape and `T`), F8b (attack-success training), F8c (the schedule),
F8d (the data source), F8e (the template), F8f (the special tokens), F8g (the poisoned
batches), F8h (the release taken as is, which closed F8), F6a (one `η` from a pilot grid),
F6b (check 6's band under mixed precision, which closed F6), F5a (agreement under bf16) and
F5b (calibration inside the first honest run, which closed F5), F9 (the cancellation
ceiling keeps its 2× factor, tested on H1's judged steps) and F10 (`k` stays 21; the
`k`-tunability result shows each poisoning-rate guarantee's price) closed after it, and
on the machinery side F1a, F1b and F1c (the transcript, now handed over in host RAM; F1 is
closed), F15a (the claimed-root check order), F4a (forward values kept for the backward
pass) and F16 (eager attention kept; its cost measured in the H100 benchmark).
A merged item keeps the old IDs in its title, so older references to them still resolve.

**Parked until the H100 runs (user, 2026-10-08).** Every decision item was closed except F4 and
F15, and both close only on H100 measurements: F4 on the verifier's measured GPU peak, F15 on
the hashing benchmark (`HASHING_PLAN_DRAFT.md` Section 6). F4 closed on 2026-10-18 on FS17's
measurement: every model fits (`DECISIONS_FULL_SCALE.md` F4). F15 closed on 2026-10-18:
the CPU hashing path (F15b) and, for the device, the CPU; GPU hashing through NVIDIA's
cuPQC was parked as speed item O8 (`DECISIONS_FULL_SCALE.md` F15). The device choice was
reopened on 2026-10-20 as F23, which took in O8, and closed the same day with the environment
(F22): BLAKE3 in our own GPU kernel, on four H100s over NVLink with one B200 as the fallback
(`DECISIONS_FULL_SCALE.md` F22, F23).
F17 (transcript bytes per step) was opened on 2026-10-11; F22 and F23 changed its rates (the
note under F17).

## Validation note for both sessions (2026-10-06)

The verifier-machinery session checked this file and `STATUS.md` against the closed decisions
and corrected stale facts. Nothing was decided or reopened.

- **Ownership.** This session owns F4, F15 and F16; F1 is closed (F1a–F1c). The other session
  keeps F8, F6, F5, F9 and F10. Its next item, F8's poisoned corpus, matches in both files.
- **F9 (frozen `κ_max`) — please re-read.** Its "2.6 passes over 500 records" predated F8d.
  With F8d's corpora a 10-step run makes 5 passes over Alpaca's 369 records and 10 over
  AdvBench's 233. Records repeat from step 3 (Alpaca) and step 2 (AdvBench), so the ceiling
  frozen on the first steps meets revisited records sooner than the item assumed.
- **F10 (`k` against a poisoning rate) — please re-read.** It is no longer blocked. C1
  measured the gradient coherence it waited for: 1.07, which supports appendix §12.2's
  assumption. That figure is from SmolLM2 at test scale, so a full-scale model may differ.
- **F4.** Rewritten after F1c: no streaming, so leaf chunking is now a speed option for F15,
  and F4 closes if an H100 measurement shows every model fits.
- **Header.** C1 is closed, so F9 and F10 no longer wait on it; the closed-items list now
  includes F1a–F1c, F15a and F4a.

## Tasks

- **F17 — Transcript bytes per step set the full-scale floor.** *Opened and taken up by the
  user, 2026-10-11; not parked.* *Rates changed by F22 and F23 (2026-10-20): on the four-GPU
  path the products cross by NVLink peer copy (slowest pair 392 GB/s measured, FS31,
  2026-10-21) and are hashed by F23's kernel on their owner GPUs (380–600 GB/s measured on one
  H100); no byte crosses the GPU–host link. The figures below are the one-H100 host-link
  floor, kept as the reason the levers exist.* Even with F15's remedies (GPU hashing, challenges generated on
  the GPU), every committed byte crosses the GPU–host link twice: the prover's copy and the
  verifier's. At F15's assumed 50 GB/s copy and 250 GB/s GPU hashing, that is about 0.024 s
  per GB per side. For the 34–67 GB step (an analytic count reproducing F1a's 32–64 GB within
  5%), that comes to 3.1–6.6× a training step per side, against PoTS's reported verify ÷ train of
  0.33–0.77. An H100 does about 8,000 FLOP in the time the link moves one byte, while a product
  costs `q` FLOP per output byte to recompute (`2q` per bf16 entry). So any product with
  contraction `q` below about 10⁴ is cheaper for the verifier to recompute than to receive.
  Per class, Qwen2.5-0.5B (bytes share of the step / FLOP share of training / `q`):

  | Class | Bytes | FLOP | `q` | Ship, per side | Recompute |
  |---|---|---|---|---|---|
  | Forward linear products | 30% | 24% | 896, 4864 | 239 ms | 29 ms |
  | Input-gradient products | 24% | 24% | 128–4864 | 194 ms | 29 ms |
  | Attention products | 17% | 1.1% | 64, 128 | 135 ms | 2 ms |
  | Logits | 15% | 9% | 896 | 120 ms | 11 ms |
  | Weight-gradient products | 3% | 33% | 16,384 | 24 ms | 40 ms |
  | Logits' input gradient | 0.1% | 9% | 151,936 | 0.7 ms | 11 ms |
  | Weights `W_t`, `W_{t+1}` (fp32) | 12% | — | — | 95 ms | — |

  Training is 121 ms. The other three models show the same pattern. Freivalds saves arithmetic
  only on the products whose bytes the verifier wouldn't otherwise need (weight gradients, the
  logits' input gradient); every other product's output is an operand the verifier must hold
  either way. Levers that came up:
  - **Recompute instead of commit.** The verifier recomputes the forward, input-gradient,
    attention and logit products and checks only the weight gradients and the logits' input
    gradient. Per side this gives a verifier at about 1.5–1.6× training and a prover at about
    1.0×, against 3.1–6.6× with everything shipped. The weights are about 60% of what remains.
    `log₂M` falls from 18.6 to about 7. This amends algorithm decision 11 (every matmul is a
    checked leaf), W-B (the verifier never performs a full matmul) and spec §6's locality
    (each check local to committed values). Recomputed chains carry prover–verifier rounding
    differences through a whole pass, which is zero only under F5a's bit-identical condition.
  - **Weights shipped every step.** Two fp32 copies, 12–25% of the step; `W_t` repeats the
    previous step's `W_{t+1}`.
  - **One challenge vector shared by products of the same width.** Cuts challenge bytes; needs
    a soundness argument (union bound without independence).
  - **Commit `P·r` in place of `P` (user, 2026-10-14). Rejected by the user the same day, weight-gradient variant included.** If it
    held, the step would shrink from 2.62 to 1.12 GB at test scale and from 34–67 to 7–16 GB at
    full scale (3.1–6.6× training per side to 0.9–1.4×; scratchpad `projected_commit.py`, session
    d412114a). Two breaks. (1) `P·r` needs `r` before the commitment, so `r` would come from
    earlier leaves, and a prover that knows `r` fits its deviation to `null(r)` with certainty,
    which is worse than grinding. Deriving `r` from a first root over `P` keeps today's soundness
    and `log₂G`, but hashes `P` in full and saves nothing. (2) Every product except the weight
    gradients is the source of the next operands through nonlinear glue, and check 5's band needs
    `‖P_m‖_F`, so the verifier must hold `P` (the point above). Shipping operands instead cuts the
    link between products, and a substituted batch passes every check. Only the weight gradients
    feed a linear consumer: dropping `G_W` and checking `W_{t+1} − W_t` against `−η·δ_Yᵀ·X` by
    projection keeps the order sound and saves 21% at test scale and 3–5% at full scale, but
    turns check 6's elementwise test into a projected one whose band must carry `W_{t+1}`'s own
    rounding (`ε_W·‖W‖`, large against `η·e_m·‖G_W‖` at fp32). Not sized.
  - Sizes from an analytic per-product count (session scratchpad `bytes_vs_flops.py`), under
    F15's assumed throughputs; the H100 benchmark (FS10–FS11) measures the real ones. *(Came up
    while summing up full-scale verifier cost, 2026-10-11.)*
- **F20 — At bf16 the calibration guard `τ/2` can't hold, and `τ = 8·s_h` may reject honest
  steps.** *Parked until real full-scale residuals exist (user, 2026-10-13, at FM2).* At bf16
  each product's own rounding dominates `e_m`, so every class has about the same RMS (FS5,
  Qwen2.5-0.5B on CPU: 0.29–0.31). With `N` residuals of one size, the largest grows like
  `√(2·ln N)·s_h`, which passes `τ/2 = 4·s_h` once `N` exceeds about 3,000. Measured: 3.81·`s_h`
  over one step (51k residuals), 5.83·`s_h` over three (154k; `L1.dA[3,10]`, whose own
  predicted RMS is 1.8·`s_h`), so H1's three calibration steps would stop at the guard. A
  10-step run holds about 1.8M residuals at `k = 21`, a Gaussian maximum near 5.4·`s_h` before
  the products noisier than their class, against `τ` at 8: honest false rejects are possible,
  and EQ's false-reject rate is a reported result. A likely remedy sizes `τ` (and the guard)
  from a target false-reject chance over the run's residual count; `τ` 1.5× larger costs about
  0.6 bits per vector, `k` about 23 instead of 21. Decided on the first full-scale honest
  residuals (FS11's bring-up steps or H1), not on CPU; since F22 and O2's measuring kernel
  (2026-10-21), on residuals from the four-H100 machine measured by FM11's kernel, whose
  reduction order moves every residual's last bits. *(Came up at FM2, FS5's three-step
  bring-up, 2026-10-13.)* At fp32 too, off the Mac: a single-step `bring_up` fit on the A40
  put the largest honest residual at 0.80·`τ`, and on the pod's x86 CPU at 0.83·`τ` (base
  code), against the guard's 0.5 (FM4, 2026-10-14). Both steps were accepted.
- **F21 — The GPU path's host side: pinned pool size and the band file's device.** *Parked
  (FM4 review of FS9, 2026-10-14).* *Since F22 (2026-10-20) the first bullet concerns the
  one-H100 path only: on four GPUs the products go to their owner GPUs by peer copy, with no
  pinned host pool. The second stays (`ENVIRONMENT_PLAN_DRAFT.md` G6): the band context has
  recorded the device type and both dtypes since FS3, but not the GPU model or the software
  build, on any of the four cards.*
  - The pinned host pool is process-wide and never shrinks. Test scale pins 1.44 GiB per
    live step at fp32; at `n_s = 128` about 32× that, roughly 46 GiB page-locked, kept after
    the run. Options: a `release()` at run end, or a cap with a fallback to pageable memory
    (E4: pageable copies stall the process's own GEMMs by 83–98%).
  - `bands.json` records no device or compute dtype, so a CUDA run would be judged by
    CPU-fitted bands without a warning. Options: record both and refuse a mismatch.
  - E6 at the full batch (128 × 128) ran out of memory for Qwen2.5-0.5B on the A40's 44 GiB,
    on a 9.27 GiB allocation, apparently the logits; on the H100 it fits with the output layer in row
    blocks (F4, closed).

## Recorded elsewhere

These full-scale items are recorded with the setup decisions. They're listed so the
full-scale agenda is in one place, not restated here.

- `DECISIONS_SETUP.md` §8.A.7 — full-scale reporting cost: GB per step moved off the GPU,
  and host syncs.
- `DECISIONS_SETUP.md` §8.A.4 — eager attention forfeits flash attention's speed and memory
  saving on the GPU.
- `DECISIONS_SETUP.md` §8.A.3 — bf16 compute raises `k` and coarsens the detection floor.
  F8a's 10-step runs give `k = 21` (appendix §10.2).
