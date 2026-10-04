# Verifier and Prover Speed — Deferred Tasklist

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> This is a shared, multi-session file, under the same lock rule as every design doc (see `STATUS.md`): before
> editing, set this line to `🔒 LOCKED — <session/date>` and re-read the file; the instant you
> finish, set it back to `🔓 UNLOCKED`. Never end a turn with the file left LOCKED.

> **Standing instruction (user, 2026-10-04).** This file parks speed and memory problems of the
> implementation that come up on the way: a part of the step that runs far above the cost its
> work needs. It is a parking list, like `FULL_SCALE_TASKS.md`:
>
> - **Add an item whenever one comes up.** State the problem in general (what is slow, against
>   which floor, measured how), then offer specific solutions. Add it and move on.
> - **Don't work on these items** until the user takes one up.
> - Nothing here may change what is committed or checked. A solution that would change a
>   check's numbers says so, and the user decides.
>
> Full-scale cost items (hashing throughput, F15; where the transcript lives, F1) stay in
> `FULL_SCALE_TASKS.md`. Items here are about the code's pace at any scale.

## Tasks

- **O1 — Check 6's update identity runs about 10× the cost of training's update.** Check 6
  recomputes `R = W_{t+1} − (W_t − η·G)` for every weight and compares `|R|` with its rounding
  scale `ε_W·(|W_t| + |η·G|)`. It must read `W_t`, `W_{t+1}` and `G` once, the same bytes
  training's SGD update reads (about 1.3 GB at test scale for the linear weights). Training's
  update does that in one fused pass, about 10 ms. The first version of check 6a took 0.77 s,
  about 80× that: about 19 whole-tensor passes per weight, each into a fresh temporary, five in
  float64. A blocked fp32 fast path (merge `4814760`, 2026-10-04) cut 6a to 0.10 s and 6b from
  0.16 s to 0.03 s, with `ρ_max` bit for bit the reference's and every rejection still worded
  by the reference. It still makes about 10 fp32 passes, about 8–10× the bandwidth floor.
  At full scale the update is a smaller share of a GPU step, but check 6 then runs on every
  weight of a 0.5B–1.5B model each step, so the multiple matters again.
  - **Specific solution: a fused kernel.** One pass per block computes `η·G`, `R`, the scale,
    the fp32 quotient and the running max, keeping only the max and its candidates. It reads
    each tensor once and writes nothing back, so it runs at the floor. On CPU that means a C++
    or numba kernel; on GPU a Triton or CUDA kernel. It must reproduce the reference's
    roundings operation by operation (`η·G` first, then `W_t − η·G`, then the outer
    subtraction), which the existing equality tests against `_update_identity_reference`
    would check.
  - Came from: the check-6a profiling, 2026-10-04 (`verifier/checks.py`, `_fast_rho_max`).

- **O2 — Check 5's measuring costs more than the matmuls it checks.** Freivalds' test is
  `O(n²)` per product against the product's `O(n³)`, so measuring should cost a small fraction
  of training's matmuls. On the real SmolLM2 step it costs 1.07 s inside the run (`5.measure`
  1.13 s with stacking), against about 0.25 s for the same 7,113 matmuls run alone (warm,
  8 threads). Split over all products, warm, measured 2026-10-04 with synthetic operands of the
  real shapes:

  | part | time |
  |---|---|
  | the Freivalds test itself, `A(B·r) − P·r` for `k = 7` | 0.115 s |
  | test 1, the cancellation guard: `|A|·(|B|·1)` and `|P|·1`, each through `abs()` copies | 0.18 s |
  | `‖P‖_F` through `_safe_norm` (abs, max, a scaled copy of `P`, the norm) | 0.17 s |
  | challenge vectors (BLAKE3 XOF and conversion) | 0.065 s |
  | Python and dispatch overhead, small norms (about 350 aten ops per product) | 0.13 s |
  | sum, warm | 0.66 s |
  | fresh temporaries page-faulting in the real run (`MallocLargeCache=0`; 0.70 s with it on) | about 0.4 s |

  So the `O(n²)` test is about 2× cheaper than the matmuls, not 20×, and the code around it
  costs 5× the test. Two reasons. At `d = 576` and `k = 7` the FLOP saving is about 25–30×,
  but a 7-column product runs at memory speed while a full matmul runs at compute speed. And
  `P` is read about seven times per product, each read into a new temporary. At full scale
  (larger `d`, a GPU) the FLOP gap widens, but the extra passes and temporaries scale with it.
  - **Done, bit-identical (merge `a093e28`, 2026-10-04).** One reused fp32 scratch
    buffer now holds `|A|`, `|B|`, `|P|` and `P`'s scaled copy, which removes the page faults.
    `|P|` is taken once and shared by the guard and the norm. `‖P‖_F` skips the scaled copy
    when every entry is at least `2⁻⁶³·max(1, max|P|)` and `n·max² ≤ 2¹²⁴`. Then every
    square stays a normal fp32 number, and scaling by a power of two shifts each rounded
    result exactly, so the norm is bit for bit the old one. Check 5 went from about 1.75 s to
    1.25 s, and the verifier from about 2.06 s to 1.64 s. The check-5 table is unchanged.
    Any `P` with an exact zero fails the bound and takes the old scaled path. Every backward
    `P` has zeros, so the gradient products don't gain from it yet.
  - **Still open, each changes rounding or order, so each needs the user's decision:**
    - `‖P‖_F` by a plain `vector_norm` when `P` holds zeros. Either a pass for the minimum
      over nonzero entries, or a slightly different rounding. This is the largest remaining
      cost, since every backward `P` falls back today.
    - Merge the small norms (`ν`, `‖|P|·1‖`, the residuals) into one call. This may change
      the reduction order.
    - Fuse the `aminmax` bounds pass with the norm.
    - About 1,300 small view operations (`select`, `as_strided`) per batch of gradient
      products, about 0.5 ms per batch, from an unknown source.
  - **Sums without copies.** `|B|·1` and `|P|·1` as `vector_norm(x, ord=1, dim=1)` read the
    tensor once and allocate nothing. They round differently from the current matmul with
    ones, at rounding level, so `κ` changes slightly. Do it before A11 calibrates `κ_max`.
  - **Fused kernel**, as in O1: one read of `A`, `B` and `P` each.
  - Came from: the user's question on 2026-10-04 why measuring costs more than training's
    matmuls.

## Done

The items O3–O6 came from a duplicated-work audit of one verified SmolLM2 step (2026-10-04;
prover 1.10 s, verifier 2.25 s at the time). All four were done the same day. The audit also
listed repeats the protocol needs, which stay: check 2 rehashing what checks 4 and 7 hashed
(byte binding), the replay's clones of committed leaves (autograd writes into its buffers in
place), `honest_final`'s independent plain run for check 8, and the run-start and run-end
hashing for checks 0 and 8.

- O3, check 7 hashed the `W_t` leaves one at a time (merge `90d0d68`). Check 7 now reads and
  validates leaves in order, hashes them in parallel, then judges them in index order, so the
  first bad leaf still decides. A `W_t` tensor written in place by a later read is rejected as
  malformed, as in check 2. Check 7 went from 0.18 s to 0.03 s per step.
- O4, the checkpoint was loaded four times per run (merge `e0ec0a1`). One `ReusedModel` now
  serves the `W_0` snapshot, `honest_final` and every prover run, and checks hooks, grads,
  buffers and mode on each handout. The verifier keeps its own model (invariant 1). A
  no-metrics run loads the checkpoint twice, not four times.
- O5, the prover copied `W_t` twice (merge `e0ec0a1`). `prove_step` commits the caller's `w_t`
  tensors when they are grad-free, contiguous, own their storage and share none with the
  model; other inputs are copied. Every root, loss, residual and final weight is unchanged.
- O6, the serial finiteness test (merge `90d0d68`). **Tried and rejected as proposed:** running
  the test inside the hashing workers made commit and check 2 about 3× slower, because the
  workers then contend for Python's global interpreter lock. Kept instead: the test stays on
  the calling thread and uses numpy `max` and `min` for fp32 and fp16. The gain is within
  noise, so the finiteness pass remains the largest part of commit and check 2. Any further
  gain belongs with F15 (hashing throughput at full scale), for example a native kernel that
  checks and hashes in one pass.
