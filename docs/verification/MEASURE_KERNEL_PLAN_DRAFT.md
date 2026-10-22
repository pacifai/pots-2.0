# Check 5's Measuring on the GPU — Draft Plan

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: approved by the user on 2026-10-21, with 2.7's metrics rule.** It implements three decisions the user
took on 2026-10-21, recorded under O2 in `PERFORMANCE_TASKS.md`:

- the fused measuring kernel (option 2), with challenges drawn on the GPU by the hashing
  session's kernel (F23 extended to the keyed XOF);
- measuring the products that share an operand in one read, a layer's forward products
  during its backward (option 5);
- overlapping measuring with the replay's glue, rows read back asynchronously and judged at
  the end in canonical order (option 4).

Nothing here changes the protocol: the challenges, the formulas, the band and the accept rule
stay as the spec has them. What changes is the order of fp32 additions inside each
matrix-vector product, the place the work runs, and when check 5 judges.

## 1. Where the time goes today

Llama-3.2-1B at full scale, one H100 (2026-10-20), `M` = 393,576 products: `5.measure` took
2.1 s of a 20 s verifier step, measured. Split, measured from a trace slice and extrapolated:

- host challenge drawing 0.73 s (8.3 million keyed BLAKE3 XOF calls);
- pageable challenge copies to the GPU 0.28 s;
- GPU compute 0.82 s: the Freivalds products about 40%, the extra passes for the guard, `U`,
  `‖P‖_F` and the fp32 widening about 58%;
- host dispatch about 0.25 s, 41k kernel launches.

The floor is about 38 ms: one bf16 read of `A`, `B` and `P` (126 GB at 3.35 TB/s).

## 2. Design

### 2.1 The measuring core (`verification/verifier/matmul_check/measure_core.h`)

Plain C that both NVRTC and a host C compiler build, as `gpu_blake3_core.h` is. It computes
today's row, `freivalds.ROW` (`ν`, `‖|P|·1‖`, `‖P‖_F`, `U²`, `‖b‖²`, then the `k`
residuals), from `A`, `B`, `P` in their stored dtype (bf16 or fp32, any strides), the optional
bias `b`, and the product's challenge matrix `R` (fp32 `[width, k]`, today's entries).
`finish_row` is unchanged and still makes the `ProductMeasure` on the host.

- **What it computes.** The formulas of `measure_products_rows`: `t_j = B·r_j`,
  `d_j = A·t_j + (bᵀr_j)·1 − P·r_j`, `ν = ‖|A|·(|B|·1) + ‖b‖₁·1‖`, `‖|P|·1‖`,
  `U² = Σ_i Σ_l A_il²·(B²·1)_l`, `‖P‖_F`. Each input is widened to fp32 in registers, never
  in memory.
- **Precision.** Every contraction (a matrix-vector entry) is fp32 with explicit `fmaf`, as
  the band assumes (`ε_acc = 2⁻²⁴`). Every norm and every sum over the output index is fp64:
  an fp32 value squared is exact in fp64 and cannot overflow or underflow there, so the
  scaled fallback of `_safe_norm` isn't needed.
- **One fixed order (picked by hand: 32 lanes, one warp).** A dot product over a logical
  index `0 … L−1` is summed as 32 partial sums, partial `l` taking indices `l, l+32, l+64, …`
  in increasing order, then halved in a fixed tree (`s[i] += s[i+16]`, then 8, 4, 2, 1). The
  order is defined on the logical index, so the bits don't depend on strides, on how many
  products share a launch, on which products share a read, on the thread layout or on the
  device. No atomics. NVRTC builds with `--fmad=false` and the host with
  `-ffp-contract=off`, so neither compiler fuses what the core doesn't fuse itself.
- **Consequence.** The GPU kernel and the core built on the host give the same bits on the
  same transcript, up to a NaN's payload: a NaN made by arithmetic is the GPU's canonical NaN
  but keeps an input's payload on the host, so tests compare NaN positions and every other
  entry bit for bit (FS36's `same_bits`). A row with a NaN is rejected either way. A GPU
  verdict can be rechecked number for number on a CPU (slowly).
  Against today's torch path the numbers agree to a tolerance only: the fp32 additions run in
  another order. O2's estimate of that shift is about 5e-4 of `τ` in a normalized residual,
  up to about 2% where `U/‖P‖_F ≈ 39`.

#### 2.1a The pinned arithmetic (added 2026-10-21 for FS36)

Every number in a row is defined by the rules below, on logical indices. A GPU kernel may
split the work any way it likes as long as it computes these values.

- **Views.** A matrix is a strided view: base address, dtype (bf16 or fp32; anything else is
  refused), rows, columns, a row stride and a column stride in elements. Element `(i, c)` is
  read and widened exactly to fp32 (bf16 by a 16-bit shift). A vector (the bias) is a view
  with one row. `R` is fp32 `[w, k]`, contiguous, entry `R[c, j]` at `c·k + j`.
- **The fp32 dot rule `dot32(x, y, L)`.** 32 partial sums `s[0..31]`, each starting at
  `+0.0f`; partial `p` takes the indices `c = p, p + 32, p + 64, … < L` in increasing order,
  `s[p] = fmaf(x_c, y_c, s[p])`. Then the tree: for `off = 16, 8, 4, 2, 1`, for `i < off`,
  `s[i] = s[i] + s[i + off]`; the result is `s[0]`.
- **Row terms.** A right matrix-vector product of view `M` (`n_r × L`) with vector `v` takes,
  for each row `i`, `dot32(f(M[i, ·]), v, L)`, with `f` one of: the identity, `|m|`, or
  `m²` rounded to fp32 (`m*m`) before the fma.
- **The fp64 sum rule `sum64(x, L)`.** The same 32 partials and tree, in fp64, each
  `s[p] = s[p] + x_c` with `x_c` an fp64 value; a square is formed in fp64 from the widened
  fp32 value (`(double)v * (double)v`, exact).
- **One product.** `A` is `n × q`, `B` `q × w`, `P` `n × w`, the bias `b` of length `w` or
  none, `k ≥ 1`:
  1. `t_j[l] = dot32(B[l, ·], R[·, j], w)` for `j < k`; with the guard,
     `b1[l] = dot32(|B[l, ·]|, 1, w)`; always `b2[l] = dot32(B[l, ·]², 1, w)`.
  2. `s_j[i] = dot32(A[i, ·], t_j, q)`; with the guard, `ν_i = dot32(|A[i, ·]|, b1, q)`,
     then `ν_i = ν_i + β₁` (fp32) when biased, `β₁ = dot32(|b|, 1, w)`;
     `u_i = dot32(A[i, ·]², b2, q)`.
  3. `π_j[i] = dot32(P[i, ·], R[·, j], w)`; with the guard, `p1_i = dot32(|P[i, ·]|, 1, w)`;
     `pp_i = sum64(P[i, c]², w)`.
  4. `d_j[i] = (s_j[i] + β_j) − π_j[i]` in fp32 when biased, `β_j = dot32(b, R[·, j], w)`;
     `d_j[i] = s_j[i] − π_j[i]` otherwise.
  5. The row, in fp64: `ν = √sum64(ν_i², n)`, `‖|P|·1‖ = √sum64(p1_i², n)` (both NaN without
     the guard), `‖P‖_F = √sum64(pp_i, n)`, `U² = sum64(u_i, n)`, `‖b‖² = sum64(b_c², w)`
     (0 unbiased), residual `j` `= √sum64(d_j[i]², n)`. The square root is IEEE
     (correctly rounded) on both builds.
- **Build flags.** NVRTC with `--fmad=false`, the host with `-ffp-contract=off`, no fast
  math on either; every fused multiply-add is an explicit `fmaf`.
- **Hand-picked:** 32 partials (a warp), the tree's shape, per-row `pp_i` before the sum over
  rows, and `fmaf` in the row terms. Any other fixed choice would serve; these are pinned
  so that the host build and the GPU agree.
- **Tolerance against torch.** The two paths add in different orders, so they agree to a
  rounding bound, not bit for bit: each field of both is checked against a float64 evaluation
  of the same formula, within the fp32 accumulation bound of its sums (`γ_L·Σ|terms|`,
  `γ_L = L·2⁻²⁴/(1 − L·2⁻²⁴)`, with `L` the length of the longest chain of fp32 operations
  feeding it). A residual is rounding noise itself, so it is compared on that absolute bound,
  never relatively.

### 2.2 The CUDA kernel (`matmul_check/measure_kernel.py`, `measure_kernel.cu`)

- Compiled with NVRTC through ctypes, reusing `gpu_blake3`'s `Nvrtc`, `Driver` and
  `build_image` by import (`build_image` gains the source name if it doesn't take one). The
  CUBIN is cached under `native.cache_dir()`. Importing compiles nothing, and
  `run_loop` builds it before step 1, beside the BLAKE3 kernel's build (in the pipelined loop,
  in the verifier's process, where FS34 loads the BLAKE3 kernel on every verifier GPU).
- One launch set per **read plan** (2.4): a table of operand reads and products built in one
  pinned host buffer, sent with one non-blocking copy, the kernels launched on a given
  stream, the rows written into a device float64 buffer at each product's canonical slot.
  The pattern is `GpuBlake3.enqueue_table`'s: the pending object holds every tensor and
  buffer the work reads until its event completes.
- **Routing by device, as hashing is (FS30), with no env flag.** `measure_product_row` and
  `measure_products_rows` send CUDA tensors to the kernel and keep torch for every other
  device. `freivalds.reference()`, a context manager, forces the torch path on CUDA for the
  comparison tests. A CPU run, test scale included, keeps every number bit for bit, and so
  do the band file, the archives and the sweeps.
- The guard off (`kappa_guard=False`) skips the `|A|`, `|B|` and `|P|·1` work and writes NaN,
  as today. A non-finite input propagates to the row and `finish_row` rejects it, as today.

### 2.3 Serving as views

Check 5 hands the kernel the replay's operands as they are: a member's `A` and `B` as a view
into the whole bmm operand (`s·n_h + h`), `δK̃`'s `B` as a transposed view, a linear's `B` as
the transposed view of autocast's weight copy, and each `P` as the committed leaf (one strided
block where `stack_block` serves one). There is no stacked copy and no fp32 copy, so
`operand_group(ms, float32)` isn't called on the kernel path; the torch path keeps it. The
bias is the committed `W_t` leaf rounded to `operand_dtype`, as today. Each served tensor's
`_version` is recorded when it is enqueued and compared again once the event has completed;
a change is a `ReplayError` (invariant 6 for reads that outlive the call).

On the one-H100 path the product leaves sit in pinned host memory (FS9). The kernel reads
them on the device, so each `P` still crosses PCIe once, as a non-blocking copy on the
measure stream; under F22's placement the leaves are already on their unit's GPU and nothing
crosses. Under the pipelined loop (FS33) a leaf's storage belongs to the prover's process and
is opened by CUDA IPC: the kernel reads it in place, and `_version` is checked on the
verifier's view only (each process checks invariant 6 on its own tensors).

### 2.4 Shared reads and unit-scheduled measuring (option 5)

- **Schedule.** Check 5 walks the replay's units in the replay's own order: the forward pass
  once, then the backward from the head to layer 1. At unit `j`'s backward it measures unit
  `j`'s backward products together with unit `j`'s forward products, whose operands the
  replay holds until then anyway. The head's `Λ_b` is measured with `δF_b` and `G_b` at block
  `b`'s backward, where the leaf is first read. Nothing is measured during the forward pass.
- **Read plan.** For one unit, the planner keys each operand by storage, offset, shape,
  strides and dtype, and lists the vectors each product needs from it on its right (`M·v`)
  and on its left (`vᵀ·M`). A read is shared only among products that read the matrix from
  the same side (decided 2026-10-22, §5); a matrix read from both sides is read once per
  side, the two reads back to back, so the second may come from L2. The reads run in a
  topological order of the product dependencies
  (`B` before `A`, since `A` multiplies `B·r`); per layer the order exists (`W` before `X`
  before `δY`; `K̃` and `Ṽ` before `Q̃` and `δO` before `A` and `δS`), and a cycle, should one
  appear, falls back to a second read. Sharing is by storage only, never by value: under
  CUDA autocast, `q`, `k` and `v` read separate bf16 copies of the normed `X`, so those three
  don't share, while `Y_x` and `G_x` share the saved `X`; `Y_x` and `dX_x` read `W_x`'s copy
  from opposite sides, so they don't share. O2's floor estimate with sharing (38 → 26–30 ms)
  assumed two-sided sharing and more sharing than storage identity may give; with one-sided
  shares only it is about 35–36 ms (estimate). FS40 measures it.
- **Torch path.** On the CPU the schedule is the same, but each run's groups are measured as
  today (`_measure_groups`, the same members per call), so every CPU number keeps its bits.
- **Judging.** Rows go to one device buffer per device, `M_dev × (5 + k)` float64 (about
  44 MB on Llama-3.2-1B at `k = 9`). At the end of check 5 one read per device brings them to
  the host, and `_judge_window` judges in canonical order in windows of `_JUDGE_WINDOW`, so
  the first failing product, its message and every recorded number are those of canonical
  judging.
- **Errors.** A serving or measuring error at any unit ends the unit-scheduled pass: the
  in-flight work is waited for, and check 5 reruns the step in today's canonical-order path
  (a fresh replay), which finds the first failing product and the first error by the existing
  rules. The path is rare (a malformed leaf or a verifier bug), so its cost doesn't matter.
- **Consequence for a rejected step.** Check 5 now always measures every product before it
  judges, so it never stops early. The record pass has nothing left to serve
  (`StepState.run_rest` holds every product after the failing one), so a rejected step costs
  the same work as before. Only the time until a cheat's verdict grows, by the rest of
  check 5.

#### 2.4a Implementation: FS38a and FS38b (added 2026-10-21)

FS38 is built in two parts. FS38a changes check 5's order and judging and keeps FS37's kernel
tables; FS38b adds shared reads to the kernel.

**FS38a: the unit schedule, end judging and the fallback.**

- **The schedule.** `Replay.unit_schedule() -> tuple[tuple[int, ...], ...] | None`, default
  `None`: canonical order only, which the MLP keeps. The decoder's, built from its inventory and
  cached on the computation: for each head block `b`, `(Λ_b, δF_b, G_b)`; then for each layer
  `j = L … 1`, layer `j`'s forward products in canonical order, then its backward products in
  canonical order. Served in that order the replay's guards hold unchanged: one forward pass
  (`Λ_0`'s request builds it), each unit's forward operands held until its backward, as today,
  each head block's leaf read once, and `L1.G_v`, which is `P_M`, served last, so
  `glue_gradients()` stays valid. No operand is held longer than today.
- **Measuring.** Check 5 maps each canonical run (`_member_runs`) to the schedule step that
  holds it; a run split across steps, or missing, raises `RuntimeError`. Each run is measured
  through `_measure_served` as today, so a CPU run's groups and every CPU number are today's.
  Group ids stay canonical.
- **End judging.** Every run's rows are kept to the end. One `_read_rows` over all of them, one
  read per device, happens at the first non-empty window; then `_judge_window` runs over
  today's windows (the window rule, shared with `_layout`) in canonical order and stops at the
  first rejecting window. The first failing product, its message and every recorded number are
  canonical judging's. On a rejection `StepState.run_rest` holds the failing window's tail, as
  today, and a new `StepState.run_rest_rows` the rows of every later window; the record pass
  finishes those with `finish_row`, serves nothing again, then raises `run_rest_error` if it
  is set. Rows stay per run on their device and are joined once per device, rather than
  written into one buffer at canonical slots: the same transfers, and FS39 may preallocate.
- **Fallback.** Any exception in the unit pass (a serving or measuring error, a drain's
  `ReplayError`) ends it: `freivalds.drain()` with its own errors swallowed, the rows dropped,
  `ctx.state.replay` replaced by a fresh `c.replay(leaves)` outside the `except` (the decoder's
  reused model takes fresh parameter storage, so no old view aliases), and check 5 runs
  today's canonical loop (`_check_5_canonical`, moved unchanged), so every error rule and
  record-pass status is today's. An error that would occur only in unit order disappears in
  the rerun; check 5 logs one debug line when it falls back.
- **Metrics.** Counts are unchanged (`c5.run` per run; `c5.glue`, `c5.serve`, `c5.copy`,
  `c5.measure` per group; `c5.judge` per window, the empty last one included) and the ids
  stay canonical. The forward pass's time moves into `Λ_0`'s `c5.glue`, and the read's into
  the first window's `5.measure`.
- **Tests, on the Mac.** The unit path against the canonical path (the schedule patched to
  `None`): every `ProductStat` field, the rejection's detail, `run_rest`, the record status and
  the stats after the record pass. For an honest step and a flip in every class at layers 1 and
  2 (`L1.Y_q`, `m = 1`, included), at one and two head blocks; a flip at a window's first and
  last product and at `m = M`; a serving error at `L1.Y_q`, `Λ_{B−1}` and `L1.G_v`, with and
  without an earlier flip; a measuring error, a malformed leaf and a version bump; one forward
  build per replay and each unit run once; the old replay unreachable after a fallback; sharded
  CPU equal to unsharded. The kernel path through the Mac emulation (`test_kernel_step.py`'s
  `mac` engine, `test_kernel_views.py`'s stand-in) unchanged, and its unit-path rows equal its
  canonical rows bit for bit.

**FS38b: shared reads in the kernel, one table per schedule step** (detailed 2026-10-22).
Only one-sided reads are shared (the user's decision of 2026-10-22, §5): a stored matrix that
several products read in the same orientation, `X_x` by `Y_x` and `G_x`, `Q̃` by `S` and `δK̃`,
`δO` by `dA` and `dV`, the head's `F_b` at fp32 (at fp32 also `X` by all of q, k, v). Which of
these storage identity gives is what the planner finds; nothing is shared by value.

- **Where.** The unit path only. `_check_5_units` serves every run of a schedule step (a head
  block, or a layer) as FS37's kernel views (`_served_view`, `_served_blocks`, with their
  stamps), then measures the step's groups in one call per device,
  `freivalds.measure_groups_rows(groups, h=, k=, guard=, share=True)`, and splits the rows back
  into each run's `_Rows` in serving order. A step with a run that isn't a kernel run
  (`_kernel_run` false: the CPU, `reference()`) keeps FS38a's per-run measuring, so every CPU
  number keeps its bits. A serving error at any run raises `_UnitPassError` at that run (FS38a's
  fallback; the unit pass doesn't call `_measure_to_failure`, which the canonical rerun keeps).
  The canonical path, the record pass and the sweep keep FS37's per-group call
  (`share=False`, the default).
- **The read plan** (`measure_kernel.read_plan`, numpy over the table, no loop over members).
  A *read node* is one product's phase-0 pass over `B`'s rows or its phase-1 pass over `A`'s
  rows (`P`'s row `i` is read in phase-1 item `i` as before and is never a key). Its key is the
  view as the kernel reads it: address, rows, columns, both strides, dtype. Nodes with equal
  keys form a share group. A biased product's phase 0 (one item more than `B` has rows) is
  never shared.
- **Levels.** `L(t, p)` starts at `p`. Repeat until nothing changes: every node of a group
  takes the group's largest level; `L(t, 1) ≥ L(t, 0) + 1` and `L(t, 2) ≥ L(t, 1) + 1`. A table
  not settled after 64 rounds has a cycle (a product whose two passes would have to run at
  once): it drops every share, each node alone at levels 0, 1, 2, and reports no reads saved.
  Each level is one launch, so a table takes a few launches where FS37 takes three per group.
- **Kernel.** `pots_measure_level` (`measure_kernel.cu`) runs one level's items. An item is row
  `r` of a read group, found by binary search as `pots_mfind` does (`pots_rfind`, in
  `measure_phases.h`, shared with the host test). For each entry `(t, p)` of the group in order
  it runs every step of product `t`'s phase-`p` item `r` with `measure_phases.h`'s functions,
  exactly as `pots_measure_phase` does. An unshared node and each product's phase 2 (`5 + k`
  items) are groups of one entry. Layout: `pots_mread_t {u64 first; u64 count; u32 entry0;
  u32 n_entries}` per group, a last one holding the level's total, and `pots_mentry_t {u32 prod;
  u32 phase}`. Within a level the groups are ordered by their view's address, so a two-sided
  matrix's two uses that fall in one level run back to back (§5's L2 note).
  `pots_measure_phase` stays for `share=False`.
- **Bits.** Every step runs FS37's functions on the same per-product scratch, and a product's
  phase runs in a later launch than its previous phase, so every row equals FS37's bit for bit
  whatever the sharing, the group order or the item order within a level.
- **Report.** `ReadPlan` holds per table the products, levels, launches, the bytes FS37 would
  read (`A`, `B` and `P` once per product) and the reads saved, `Σ (entries − 1)·rows·cols·itemsize`
  over the share groups. `enqueue(..., plan_out=list)` appends it. Check 5 doesn't read it (the
  profiling formats are frozen, 2.7); the A40 test prints it per table.
- **Metrics.** `c5.run` per run, around its serving; `c5.glue`, `c5.serve` and `c5.copy` per
  group as now; one `5.measure` section and one `c5.measure` stamp per table, with ids (the
  step's first run, −1), which the trace analysis maps to that run's class; `c5.judge` per
  window as in FS38a.
- **Holds.** The table's `Pending` holds every view of the step until a drain, as one FS37 call
  holds its groups.
- **Host test.** `measure_core_host.c` gains `pots_host_measure_levels`, the level launches run
  in loops, a level's items forwards or backwards. The Mac `mac` engine (`test_kernel_step.py`)
  routes `share=True` calls to it.
- **Tests, on the Mac.** `read_plan` on hand-built tables: the expected groups and levels for
  `X` read by a `Y_x` and its `G_x`, `Q̃` by `S` and `δK̃` members, `δO` by `dA` and `dV`
  members and one `X` by three forward products; a two-sided use (`W` as `Wᵀ` and as `W`) not
  shared; a biased phase 0 alone; a cycle dropping every share; the reads saved.
  `pots_host_measure_levels` equals `pots_host_measure_phased` bit for bit on FS36's inputs as
  one table and on mixed tables with shares, forwards and reversed. Check 5's unit path through
  the `mac` engine with sharing: rows and every check-5 number equal to the canonical path's bit
  for bit on the tiny decoder step (four switches, fp32 and bf16, honest and a flip per class),
  one call per step and device, and the shares found per layer listed.
- **Acceptance on the A40** (FS38b's row): with sharing, the kernel's rows equal FS37's
  (`share=False`) bit for bit on FS36's inputs as one table and on every product of the tiny
  decoder step at fp32 and bf16; the honest step accepted and a flip of each class rejected at
  its product; FS37's CUDA tests still pass; the reads saved printed per table for the tiny
  step and one SmolLM2 step at bf16, beside §5's 6–7% estimate.

### 2.5 Overlap with the glue (option 4)

- One **measure stream** per verifier device. When unit `j`'s backward has produced its
  operands, the host records an event on the replay's stream, makes the measure stream wait
  on it, enqueues unit `j`'s read plan there and moves on to unit `j−1`'s backward on the
  replay's stream. One unit is in flight: before enqueuing unit `j−1`, the host waits for
  unit `j`'s event, so at most one unit's operands are held beyond the replay's own life.
- The pending unit holds every tensor its kernels read until its event completes, as the
  capture does instead of `record_stream`, so the allocator can't reuse a block under a read.
- Challenges, until the GPU XOF arrives (2.6): drawn for unit `j−1` by the native library on
  a worker thread while unit `j` is measured (ctypes releases the GIL), into a pinned buffer
  from a pool of two, and copied non-blocking on the measure stream.
- The final row read is a non-blocking copy into pinned memory and one event per device.
- On F22's four GPUs each verifier GPU has its own measure stream, so GPUs 1–3 measure while
  the one host thread runs the next unit's glue; `_read_rows`'s one read per device per window
  becomes one read per device per step.
- Bits don't move: the kernel's order is fixed and the glue runs under
  `use_deterministic_algorithms` whatever runs beside it.
- Memory: one unit's operands held a little longer, the challenge pool (largest at the head:
  `n_v × k × 4` bytes per block, about 7 MB on Llama-3.2-1B at `k = 15`) and the row buffer.

### 2.6 The challenge seam

`ChallengeSource.draw(h, ms, k, widths, *, stream) -> (device buffer, offsets, event)` is the
one place challenges come from on the kernel path. Two sources share the interface and the
bytes: the host source (the native `challenge_matrices` into a pinned buffer, then one
non-blocking copy), used first, and the GPU source, the hashing session's keyed XOF writing
the entries straight into device memory (FS41). The layout is today's: fp32 `[width, k]` per
product, column `j−1` vector `j`. The buffer is on the stream's device: a unit's challenges are
drawn for the GPU that measures it.

### 2.7 Metrics under overlap

The profiling formats are frozen at version 1 (`archive/PROFILING_PLAN.md` §5), so nothing
new is added to them. Decided with the plan (user, 2026-10-21):

- `5.measure`'s row keeps its meaning of the time measuring costs the verifier: the host time
  spent enqueuing, waiting for an in-flight unit and reading the rows back. Measure-stream
  GPU time that hides under the glue is in no row's wall time.
- The `c5.measure` stamp's `g0`/`g1` come from an event pair on the measure stream, so each
  group's GPU span stays visible in the trace; `c5.serve` and `c5.copy` cover the read
  plan's table build and its copy; `c5.judge` covers the end-of-check windows.
- `c5_layout.json`'s `served_bytes` counts the products as measured on the device, now in
  their stored dtype rather than fp32, which is what its definition says.

## 3. Expected cost

Every figure below is an estimate or a floor; FS40 measures them.

- Challenges: 0.73 s of drawing moves off the verifier's critical path to the worker thread,
  and to the GPU with FS41 (its cost is the hashing session's to estimate). The 0.28 s of
  pageable copies become pinned non-blocking copies, hidden under the glue.
- GPU compute: 0.82 s → 0.05–0.1 s (estimate), against a floor of 38 ms without shared reads
  and about 35–36 ms with the one-sided shares (§5).
- Dispatch: 41k launches → a few per unit, a few hundred per step (estimate).
- Overlapped with the backward glue, `5.measure`'s row should come to the final read and the
  last unit's measuring, near 0.1 s (estimate), or about 0.5% of the 20 s step (the one-H100
  unsharded verifier of 2026-10-20; on F22's machine the share is read from FS35's session). The user's
  goal was under 1%.

## 4. Tasks

The rows are in `IMPLEMENTATION_PLAN.md`. The agent rule is the
profiling tasks': an Opus orchestrator, Sonnet coders, testers and reviewers at medium
effort, at most about 10 agents at once, worktrees created by hand from `verification-impl`;
then a fresh Opus validator at xhigh effort checks the merged feature against this plan, up
to three rounds.

| ID | Task | Depends on | Done when |
|---|---|---|---|
| FS36 | The measuring core (2.1): `measure_core.h`, its host build and a ctypes binding for tests | — | on the Mac: the host build agrees with `_measure_products_reference` within a tolerance on random and adversarial inputs (bf16 and fp32, strided and transposed views, biased, zero rows, a NaN, an Inf, the guard off), and gives identical bits under any batch, stride, read-sharing and table order |
| FS37 | The kernel (2.2), serving as views (2.3) and the host challenge source (2.6), routed by device; check 5 still in canonical order | FS36 | fast suite unchanged on the Mac; on one A40: the kernel equals the host build bit for bit on every input of FS36's tests and on every product of a tiny decoder step, agrees with the torch path within FS36's tolerance, and the step is accepted and every cheat rejected at its declared product |
| FS38a | Unit-scheduled measuring with end judging and the canonical fallback (2.4, 2.4a) | FS37 | on the Mac: every CPU number bit for bit as before, the first failing product and message equal to canonical judging for a planted failure in every class and layer (layer 1's forward included), every error rule and record-pass status as before |
| FS38b | Shared reads in the kernel, one table per unit, one-sided shares only (2.4, 2.4a, §5) | FS38a | on the A40: kernel rows equal FS37's bit for bit, and the reads saved reported per unit |
| FS39 | Overlap (2.5): a measure stream per device, one unit in flight, holds, the worker-thread challenges, pinned row readback; the metrics of 2.7 | FS38b | on the A40: rows and verdicts equal FS38's bit for bit, a write to a held operand during a read raised; on a two-GPU NVLink pod with `VERIF_VERIFIER_DEVICES`: rows and verdicts equal the one-GPU run's |
| FS40 | H100 timing: one verified step per full-scale model with the trace on Llama-3.2-1B, check 5's sections against §3's estimates, and the band fit rerun with the kernel | FS39 | each model's `5.measure` and verifier time reported against the old pod's, the kernel equal to the host build on a sample of products → **Milestone FM11** |
| FS41 | The GPU challenge source: the hashing session's keyed XOF behind `ChallengeSource` | FS39, the hashing session's XOF | on a CUDA GPU the device challenges equal `challenge_matrices`' bit for bit for every product of a step; FS40's Llama step timed again |

FS36 starts at once. FM11 doesn't wait for FS41. The four-GPU work (FM10) built FS32 and FS33
on the same check-5 code; FS38 and FS39 change `_read_rows` and the serving, so they are
coordinated with the session that builds FS34 and FS35, and FS35's four-H100 session runs
the kernel if FM11 has closed by then. FM11 closes on the one-H100 path (FS40); check 5's
sharded and pipelined timing is FS35's.

**Pods** (the live price is stated before each is created): one A40 for FS37–FS39, about
$0.6 per hour (half the two-A40 pod's $1.2), 1–2 h per task; a two-GPU NVLink pod for FS39's
sharded test, two A100 SXM at about $3.6 per hour, about 1 h, or FS34's pair session if they
fall together; one H100 SXM for FS40, about $4 per hour, about 1.5 h with a 3 h self-delete
deadline.

## 5. Open points

- **Two-sided reads and the pinned order** (FS38's design review, 2026-10-21). **Decided
  2026-10-22 (user): one-sided shares only**, the two-sided matrices' two uses run back to
  back so the second read may come from L2; the reasoning and the rejected re-pinned order
  are under O2 in `PERFORMANCE_TASKS.md`. The question as it was put: §2.4 said a two-sided matrix-vector kernel reads a matrix once for the products that
  read it on its right (`M·v`) and on its left (`vᵀ·M`). Under §2.1a that read has no fast form:
  each of a row dot's 32 partials is an `fmaf` chain over increasing column index, and each of a
  column dot's over increasing row index, so one read serving both runs as a wavefront that
  passes about `32·(k + 2)` fp32 partials per row between column stripes, several times the
  stripe's own bytes. The two-sided matrices are `W_x` (`Y_x`'s `B = Wᵀ`, `dX_x`'s `B`), `δY_x`
  (`dX_x`'s `A`, `G_x`'s `A = δYᵀ`), `K̃`, the softmax output (`O`'s `A`, `dV`'s `A`), `Ṽ` and
  `δS`; on Llama-3.2-1B they hold about 27% of the 126 GB floor read (about 10 ms of 38 ms at
  3.35 TB/s, about 0.05% of the 20 s step), against about 6–7% for the one-sided shares
  (estimates from FS38's design, not measured).

- **One unit in flight on three GPUs** (plan audit, 2026-10-21). As written in 2.5 the rule is
  global: the host waits for unit `j`'s measuring before enqueuing unit `j−1`'s, whichever GPU
  each sits on. One unit in flight per device would let GPUs 1–3's measuring overlap one
  another too, at one more unit's operands held per GPU. FS39 decides and records which.
- **Stamps on several GPUs** (plan audit, 2026-10-21). FS33's recorder times every verifier GPU
  (`TimeRecorder(devices=...)`: a nested section takes one event pair per device, a top-level
  one waits on every device), and FS33's second fix round records check 5's stamps on the
  recorder's first device, so that they resolve against that device's anchor. A `c5.measure`
  stamp on another GPU's measure stream must resolve against its own device's anchor; FS39
  builds that.

- **The GPU XOF.** FS41 waits on the hashing session's keyed-XOF kernel; until then the host
  draw runs on a worker thread, still about 0.73 s of host CPU per step.
- **The CUDA test that failed (resolved 2026-10-21).** `test_tiny_decoder_step_on_cuda[fp32-2]`
  and `[bf16-2]` raised a `LabelingError` (`Lambda[0]` got `(24, 32)`, expected `(12, 32)`).
  The cause was the test helper, not CUDA: `run_capture` captured the step as one whole-batch
  `loss().backward()`, so `Λ` had all `N` rows where two head blocks declare `N/2`. Commit
  e3ef29d (FS32 r4) made it capture through `forward_backward`, and
  `test_captured_operands_follow_the_head_blocks` pins it on the CPU. FS37's A40 run confirms
  the CUDA test.
- **Full-scale bands** are calibrated with the kernel (FS40). None exist yet, so nothing is
  recalibrated; the test-scale band file stays, as every CPU number does.
