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
> Full-scale cost items stay in `FULL_SCALE_TASKS.md`; hashing throughput and where the
> transcript lives are closed (`DECISIONS_FULL_SCALE.md` F15, F22, F23). Items here are about
> the code's pace at any scale.

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
  weight of a 0.5B–1.5B model each step, so the multiple matters again. On F22's four GPUs
  check 6 runs on each weight's owner GPU (FS32e), so a GPU kernel runs on every verifier
  GPU, and its `ρ_max` must be the same on every card.
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
  - **Decided 2026-10-21 (user): a fused measuring kernel, with challenges drawn on the GPU.**
    On Llama-3.2-1B at full scale (one H100, 2026-10-20), `5.measure` took 2.1 s of a 20 s
    verifier step, measured, against a floor of about 38 ms: one bf16 read of `A`, `B` and `P`
    (126 GB at 3.35 TB/s). Split, measured from a trace slice and extrapolated: host challenge
    drawing 0.73 s (8.3 million BLAKE3 XOF calls), pageable challenge copies 0.28 s, GPU
    compute 0.82 s, host dispatch about 0.25 s (41k launches). The Freivalds products are about
    40% of that GPU time; the passes for the guard, `U`, `‖P‖_F` and fp32 widening about 58%.
    The plan's cost model priced verifier GPU work as one read of the products (`vx = 1`, picked
    by hand), so measuring's extra passes and host-side challenges were never in it.
    - **Challenges stay per product and per vector, from the claimed root.** The GPU BLAKE3
      kernel of F23, built by the hashing session, also covers the keyed XOF output that draws
      them (user confirmed that it is in that kernel's scope). The bytes don't change, so the
      protocol doesn't either. This removes about 1.0 s (the two measured host parts).
      Rejected: one challenge shared by many products, because the user judged the security
      risk not worth taking.
    - **One fused kernel per group of members** reads `A`, `B` and `P` once in bf16, widens
      them in registers, and returns the residuals, the guard's `U`, `‖|P|·1‖` and `‖P‖_F`. It
      uses a fixed reduction order (no atomics), so a transcript gives the same bits on every
      run, and it accumulates the norms in fp64, so no scaled fallback is needed. The formulas
      and input values stay the same; only the order of the fp32 additions changes. That moves
      a normalized residual by about 5e-4 of `τ`, up to about 2% where `U/‖P‖_F ≈ 39`, inside
      the band's `√q·ε_acc` term. No spec change. The CPU keeps today's code as the reference,
      a backend picked by config, and a test compares the two within a tolerance. Tests that
      reproduce GPU numbers bit for bit re-pin to the kernel. Full-scale bands are calibrated
      with it; none exist yet. GPU compute is estimated to drop from 0.82 s to 0.05–0.1 s.
    - **Decided 2026-10-21 (user), both:** measuring the products that share an operand in
      one read, a layer's forward products during its backward (floor 38 → 26–30 ms, an
      estimate that assumed more sharing than storage identity may give). Check 5 then never
      stops early; the record pass has nothing left to measure, so a rejected step costs the
      same work, and only the time to a cheat's verdict grows. And overlapping measuring with
      the glue: a measure stream per device, rows read back asynchronously and judged at the
      end in canonical order. On the four-GPU design (F22), that keeps GPUs 1–3 measuring
      while the one host thread moves on. The plan, for the user's approval, is
      `MEASURE_KERNEL_PLAN_DRAFT.md` (FS36–FS41, milestone FM11). The kernel and its host
      build give the same bits; the torch path agrees with them to a tolerance.
    - **Decided 2026-10-22 (user): shared reads are one-sided only.** A read of a stored
      matrix is shared only among products that read it from the same side (all `M·v`, or
      all `vᵀ·M`). The pinned order of the measuring core (`MEASURE_KERNEL_PLAN_DRAFT.md`
      §2.1a) adds each row dot's 32 partials along the row and each column dot's down the
      column, so one read serving both sides would pass about `32·(k + 2)` fp32 partials per
      row between column stripes, several times the stripe's own bytes. The two-sided matrices
      (`W_x`, `δY_x`, `K̃`, the softmax output, `Ṽ`, `δS`) hold about 27% of the 126 GB floor
      read on Llama-3.2-1B: at most about 10 ms of the 38 ms, or 0.05% of the 20 s step. The
      one-sided shares save about 6–7% (estimates from FS38's design, not measured). A
      two-sided matrix's two uses run back to back, so its second read may come from the
      H100's 50 MB L2 (the bf16 weights, 8–32 MB, may fit; the activation gradients,
      64–256 MB, won't). FS40 measures what is left. Rejected: re-pinning the summation order
      tile by tile, so one tile read feeds both its row and its column partials. It recovers
      most of the 10 ms on paper (writing the tile partials back costs about a third of a
      read, estimate), but it reopens FS36's core and FS37's kernel with their bit-exact tests
      and the A40 check, before FS38b and FM11, for a gain of about 0.05% of a step that the
      estimates can't resolve. The user judged the switch not cost-effective for its risk.
      It can come back if FS40 shows check 5's reads to be a real share of the step.
  - Came from: the user's question on 2026-10-04 why measuring costs more than training's
    matmuls.

- **Audit of 2026-10-13 (EM1, user's request), covering O1, O2 and check 2.** This was run on the
  stored step 4 with `k = 9`, on an M5 Pro with 8 threads, at 240 GB/s DRAM and 32.6 GB/s blake3.
  Verifier 1.651 s against `P0` 0.650 s. **Every part does what the plan says; no test is extra
  or missing.**

  | part | measured | floor | cause |
  |---|---|---|---|
  | 5.measure | 0.76 s (0.685 s isolated) | 19 ms read once; about 110 ms at stock BLAS | separate passes over `A`, `B`, `P` for the guard, `U` and `‖P‖_F`; about 77 aten ops per product; challenges 96 ms; member restack 39 ms; the scaled `‖P‖_F` path on 4,531 of 7,113 products |
  | 5.glue | 0.45 s (0.30 s warm alone) | training's element-wise time, 0.21 s | the glue itself is 0.18 s; supply clones 75 ms, serving 53 ms, model build 20 ms |
  | check 5, rest | 0.05 s | about 0 | judging and per-product Python |
  | 2 | 0.224 s | 80 ms hashing | validation and finiteness on the calling thread |
  | 6a | 0.103 s | 5.3 ms read | about 13 passes per block |

  The F18 term scale `U` costs about 86 ms, 12% of 5.measure. Fused with the guard it would
  save 40–80 ms, as a rounding change. **Estimated split of the 1.65 s:** about 0.16 s is
  removable bit for bit, or about 0.40 s with two native kernels. About 0.5–0.6 s is removable
  only with a fused measuring kernel or a changed reduction order; that is a rounding change
  and needs recalibrated bands. About 0.35–0.5 s is inherent: hashing, the glue replay,
  measuring at DRAM speed and the XOFs. About 0.2–0.3 s is unassigned.
  - **Done on the audit branch, not merged:** `6a7c920` builds the challenge entries in 32-bit
    arithmetic, one pass per product. It is bit-identical on all `2²⁴` grid points and
    reproduces step 4. It saves 21 ms isolated, which is within noise in the full step.
  - **Open, bit-identical:**
    - the native challenge batch (E5a′'s C build), a further −32 ms; it puts a C build into
      `verification/` or breaks import rule 8;
    - aliasing operands in `_supply`/`_supply_backward`, about −50 ms; it breaks the
      fresh-tensor invariant;
    - less serving and Python per product, about −50 ms;
    - the member restack, about −8 ms;
    - check 2's finiteness and hashing in one native pass, up to about −140 ms (O6 showed
      Python workers don't work);
    - a zero-tolerant `‖P‖_F` fast-path test, 30–50 ms, bit-identity not yet shown.
  - **Open, rounding changes:**
    - a fused measuring kernel, −0.5 to −0.65 s, essential on GPU;
    - the 6a fused kernel, −90 to −105 ms, bit-identical only without FMA contraction
      (`-fmad=false` on CUDA; Triton contracts by default).
  - **Full-scale risk (estimate):** per-product host work is about 19 µs. At `M = 258,555`
    products, that is about 4.8 s per step unless batched, and GPU launches (about 5 µs) cost
    more than CPU dispatch. FM4 measures it. **Measured at FM4 (A40, SmolLM2, fp32):** check 5
    takes 460–600 µs per product warm, 10.8 kernel launches and 2.9 host syncs per product,
    with the GPU busy 0.47 s of 4.1 s. Unbatched, that is about 119–155 s per step at
    Qwen2.5-0.5B's `M = 258,555` and 181–236 s at Llama-3.2-1B's `M = 393,555` (the largest;
    `DECISIONS_FULL_SCALE.md` F11's table): the batched measuring path is required for full
    scale (owned since 2026-10-21 by `MEASURE_KERNEL_PLAN_DRAFT.md`, FS36–FS41). (`258,555` here and above is Qwen2.5-0.5B's count, first quoted for Llama by
    mistake.)
  - Scripts and outputs: `scratchpad/perf_audit/` of session 948d55ba.

- **O7 — The planted-error sweep re-measures a member's whole batch in every trial.** On Linux
  x86 BLAS and on CUDA, a batched product's bits depend on the batch count, so the sweep
  (EV3) measures a member target inside its whole check-5 batch (merge `0f850d0`,
  2026-10-14). Each trial then measures all the members of that shape group, not one: on the
  Mac, EM1's step 4 member targets went from 5–7 s to 33–41 s each (a 36-member group), and
  the whole sweep from 626 s to 722 s. The floor is one member's measuring per trial. At full
  scale a shape group holds every head of a layer (and every sequence, if the batch dimension
  is folded in), so the multiple grows with the group.
  - **Specific solution: fill every slot with a trial.** Batch members don't mix (each slot's
    rounding depends on the batch count and shapes, not on the other slots' values), so the
    batch could hold the target's operands in all slots, each with its own planted error:
    one `measure_products` call gives a group's worth of trials. It must be shown bit for bit
    first, slot by slot, on each platform (a kernel may split the batch unevenly across
    threads), with the existing honest self-check kept on the target's real slot. On F22's
    four GPUs each verifier GPU measures its own layers' groups, so the proof also covers
    every card (HW3's same-bits result covers the Merkle root, not batched slots).
  - **Or measure one sub-batch of the same count.** Only if the bits depend on the count
    alone; needs the same per-platform proof.
  - Came from: the x86 failure of `test_a_member_is_measured_as_check_5_stacks_it` on the A40
    pod, 2026-10-14.

- **O9 — Hashing inside the matmul kernel.** *A research direction, parked by the user on
  2026-10-19 from the hardware comparison (`HARDWARE_ENVIRONMENTS.md`). Re-assessed against F22
  and F23 on 2026-10-21 and kept parked with a narrower trigger (user).* Each GEMM's epilogue
  hashes its output tile while the tile is still in registers. The hash then costs no extra
  read of HBM, only its arithmetic, on integer units the tensor cores leave idle. A search on
  2026-10-19 found no published system that computes a cryptographic hash this way.
  - **Reopen only if** the work falls back to one B200 (F22's fallback) and a measured step
    there shows the prover's hashing above 10% of the step *(hand-picked)* after G7's
    side-stream overlap (`ENVIRONMENT_PLAN_DRAFT.md`).
  - **Four H100s over NVLink (F22): no gain.**
    - GPU 0 does no hashing. Capture peer-copies each product to its owner, which hashes it on
      arrival (FS34). FS34 rejected hashing on GPU 0 because it sits on the prover's critical
      path, and O9 is that placement.
    - Pipelined, GPU 0's training sets the pace, 0.50–1.20 s per step (floor,
      `ENVIRONMENT_PLAN_DRAFT.md`). Each owner hashes about a third of the step, 11–22 GB, in
      about 0.02–0.06 s at FS29's call rates, behind the copy (floor, not measured on the
      owners). O9 moves at most that work onto the GPU that sets the pace, and its GEMMs leave
      cuBLAS.
    - If verification sets the pace instead, F23's kernel on a side stream of GPU 0 moves the
      same work and keeps every digest. O9 loses either way.
    - No copy is removable: the 34–70 GB transcript doesn't fit on GPU 0 beside training, and
      the verifier needs the bytes on GPUs 1–3.
  - **The verifier gains nothing in any environment.** It runs no GEMMs (the replay substitutes
    the claimed products, and check 5 multiplies them by vectors), so check 2 hashes every byte
    it holds.
  - **One B200: at most about 20–40 ms per step, prover side only.** The bound for
    Qwen2.5-1.5B, step by step (`scratchpad/o9_bound_a.py` of session b0c8ae33):
    1. The GEMMs write 54 GB of products per step. The weights aren't GEMM outputs and are
       hashed apart in any case.
    2. The GEMMs take about 0.19 s: the docs' 400 TFLOP/s on the H100, times 2.0 for the B200
       *(hand-picked, the environment plan's speed factor)*.
    3. A separate hashing pass over the products takes about 0.07 s, at about 790 GB/s
       *(estimate: FS29's 696–740 GB/s kernels on the H100, times 148/132 SMs)*.
    4. Fused, each product class's hash hides under its own GEMMs, at most `min(GEMM, hash)`
       per class: 59 ms. Attention products hash slower than they multiply, so they hide at
       most 3 ms.
    5. Fused GEMMs cost about 9% more *(borrowed: fused ABFT on T4 and A100)*, about 17 ms back.
    6. Net: at most about 42 ms of a 1.90 s step. Over the four models it is 28–59 ms before the
       overhead and about 20–40 ms after, 2–3% of a step.
    - A faster GPU gains less. The saving can't exceed GEMM time, and a B200 halves the GEMMs
      while the hash rate grows about 1.1×. G7's overlap of F23's kernel with training recovers
      part of the same time without fusion.
  - **What it changes.**
    - Every product digest. BLAKE3 hashes a leaf in 1 KiB chunks in C order, offset by the
      8–16-byte prefix, and a tile holds a rectangle, so no tile can produce today's digest.
      Only attention members, which one thread block holds whole, keep it, and they gain least.
      The digest becomes a tree over fixed pieces (tiles, or row segments of a fixed width), so
      S9a, the golden digests and test scale's milestone runs (A11–A14, B7) change: the cost
      F23 chose its kernel to avoid. Records and weights keep today's digests, so checks 0, 4,
      7 and 8 don't change.
    - The run's GEMMs move from cuBLAS to our own kernels, and `MatmulCapture` calls them
      instead of passing through. Only a kernel whose epilogue for one tile overlaps the next
      tile's mainloop gains (CUTLASS 3.x on Hopper's ping-pong schedule). A plain Triton GEMM
      runs its epilogue after the mainloop and gains nothing. The plain baseline must run the
      same kernels, or F5c's bit-identical final weights fail, and cost ratios still divide by
      cuBLAS's `P0`.
    - The test-scale CPU path computes the same digest (one code path). The `δK̃` transposes
      and `perturb`'s changed leaf are hashed after the kernel.
  - **Closest evidence.** ABFT checksums fused into GEMM cost about 8.9% over cuBLAS on T4 and
    A100 (Wu et al., arXiv 2305.01024). They detect faults but don't bind a prover who chooses
    the output, so they show the price of fusion, not a usable commitment. CUTLASS's epilogue
    visitor trees (CUTLASS 3.2 and later) give the hook.
  - **Variant: a lattice hash on tensor cores.** An SIS (Ajtai) hash, `h = A·x mod q`, is itself
    an integer matmul, binding under standard lattice assumptions, and linear, which might later
    let Freivalds' test run on hashes. No measured system was found. Its inputs must first be
    split into small digits, which by estimate eats much of the tensor cores' advantage. Its
    speed case falls with O9's. Its remaining appeal, Freivalds' test on hashes, is a protocol
    change and not a speed item.
  - Came from: the hardware-environment exploration, 2026-10-19; re-assessed 2026-10-21.

- **O10 — Member blocks: one tensor per batched product, not one Python object per member.**
  *Taken up by the user on 2026-10-20; implementation in progress.* The 2026-10-19 H100
  profile (Llama-3.2-1B, n_s=128, M = 393,576 members) showed one general cost: every
  attention member is handled as its own Python object. Labelling took 13.8–35.3 s (35–90 µs
  per member, one core), the replay re-stacks members one by one before each S/O/dK
  substitution, and check 5 stacks them again (host_stack 7.4 s). The fix keeps every member
  its own leaf (protocol-fixed) but moves them as blocks: labelling validates a bmm record once
  and fills its consecutive slots from one `unbind`; serving returns one view (or one copy)
  for a run of consecutive equal-shape leaves of one storage, with per-member version checks.
  Outputs stay bit-identical. Hashing is out of scope (F23's kernel, FS28–FS30 and FS34, owns
  it). The same item covers sharing one pinned, non-blocking host-to-device copy between the
  replay and check 5 (the profile showed pageable, doubled copies); that copy exists on the
  one-H100 unsharded path only, since on F22's four GPUs the leaves already sit on their
  owner GPUs.
- **O11 — Vectorized judging in check 5.** *Taken up 2026-10-20; in progress.* Judging cost
  15.2 s per Llama step: a per-product Python `finish_row` and a `ProductStat` object per
  product. The fix computes the window's measures as float64 arrays, bit-identical to the
  scalar math, keeps statistics in columns (list-compatible for current consumers), and on the
  first failing product falls back to the scalar path, so messages and record-pass behavior
  don't change. Also suspected in the host memory growth seen across bring-up steps.
- **O12 — Bit-identical measure savings in check 5.** *Taken up 2026-10-20; after O10–O11
  are measured.* Measuring makes fp32 copies (3 s against a floor near 0.3 s). Only changes
  that keep every bit of the measured values are in scope; options that change rounding need
  the user's decision because they force band recalibration (see O2). The fused measuring
  kernel (O2, decided 2026-10-21; FS36–FS40) removes the fp32 copies and changes rounding
  with the user's approval, so O12 covers the torch path and the CPU reference only.
- **O13 — Profiler fixes and NUMA placement.** *Taken up 2026-10-20; in progress.* The
  analyzer averaged warm-up step 1 into the steady state and labelled missing work records
  "launch"; it should separate warm-up, say "unknown", and record n_s and µs per leaf. The pod
  session should pin each process to its GPU's NUMA node (node 1 on the 2026-10-19 one-GPU
  pod). On the four-GPU pod the prover's process goes to GPU 0's node and the verifier's to
  the node of GPUs 1–3, or the record says they span nodes (`nvidia-smi topo -m`).
- **Deferred with O10–O13:** overlapping check 5 with check 2's CPU hashing. Check 2's flow
  was redone: a leaf on a GPU is hashed by F23's kernel (FS34), and O2's measure stream per
  device (FS39) covers the overlap there. Open for the one-H100 path's CPU hashing only.

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
