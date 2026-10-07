# Hashing Plan — Draft for Later Consideration

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: a proposal, not a decision.** Written on 2026-10-07 by the verifier-machinery session
for F15 (hashing throughput sets the full-scale cost), and revised the same day after a review
found a wrong challenge-cost figure, an unpriced CPU alternative and a chunk layout that broke
the per-object checks. The user parked it to consider later. Nothing here changes `S9a` or any
other closed decision until the user approves it. The open choices are settled by the H100
experiment of Section 6. When approved, the decisions go to `DECISIONS_FULL_SCALE.md` (as F15b
and following) and `DECISIONS_SETUP.md` (as an S9a revision), and this file is deleted.

## 1. The problem

Under the compute-first priority (`CLAUDE.md`, "Project goal"), hashing is the largest
remaining full-scale cost that has no plan. With F1c (host-RAM hand-off) each leaf is hashed
once per side in either check order (F15a), so one pass over the transcript is the whole
hashing cost. Three further costs sit beside it and are sized here too, because removing
hashing from the critical path makes them the next ones.

Inputs, with the assumed ones marked:

- Transcript per step (F1a's table, an analytic count): 32, 47, 52 and 64 GB for Qwen2.5-0.5B,
  Llama-3.2-1B, Falcon3-1B and Qwen2.5-1.5B.
- Parameters (model cards, approximate): 0.49B, 1.24B, 1.67B, 1.54B. Tokens per step:
  128 × 128 = 16,384.
- **Assumed:** training runs at 40% of the H100 SXM's 989 TFLOP/s dense bf16 peak, about
  400 TFLOP/s. Eager attention and the capture make the real prover step slower.
- **Assumed:** CPU hashing at 15–30 GB/s on the server (16–26 cores, scaled from 11 GB/s
  measured on 8 Mac threads). F15's task entry used 10–30 GB/s, hence its 1–6 s; the range here
  drops the low end because the server has more cores than the Mac. Section 6 measures it.
- **Assumed:** GPU hashing at 200–300 GB/s on an H100 (Section 3 has the published figures).
- GPU-to-host copy at about 50 GB/s with pinned memory, a measured figure from the NVIDIA forum.
  F1c's "roughly 1.5–3 s for 64 GB" (21–43 GB/s) was a rougher range that also covers pageable
  memory. Section 6 measures both.

Derivation, per model (Qwen-0.5B / Llama-1B / Falcon-1B / Qwen-1.5B):

1. Training FLOP `≈ 6 × parameters × tokens`: 4.8·10¹³ / 1.22·10¹⁴ / 1.64·10¹⁴ / 1.52·10¹⁴.
2. Training step at 400 TFLOP/s: about 0.12 / 0.31 / 0.41 / 0.38 s.
3. CPU hashing per side: 1.1–2.1 / 1.6–3.1 / 1.7–3.5 / 2.1–4.3 s, which is **4–18× the
   training step** if it runs in series. The smallest model is the worst case, because its
   transcript is large for its compute (14 heads × 24 layers of attention products).
4. GPU hashing per side: 0.11–0.16 / 0.16–0.24 / 0.17–0.26 / 0.21–0.32 s, about **0.4–1.3×
   the training step**, so even GPU hashing is not negligible unless it overlaps other work.
5. GPU-to-host copy (prover) and host-to-GPU copy (verifier): 0.64 / 0.94 / 1.04 / 1.28 s each
   way, **2–5× the training step**. Once hashing moves to the GPU, the copy is the largest of
   these costs.
6. Challenge generation (verifier only). Each product `m` gets `k = 21` vectors, one entry
   per column of `P_m`, 4 XOF bytes per entry (P8b). The attention products dominate: per
   `(s, h)` pair their widths sum to `256 + 4·d_h` (`S` and `dP` are 128 wide, the other four
   are `d_h` wide), with `d_h` = 64 / 64 / 256 / 128.
   - XOF output per step: about 1.9 / 2.8 / 2.0 / 2.8 GB, against about 40 MB at test scale.
   - XOF calls per step, `M·k`: 5.4M / 8.3M / 2.3M / 5.4M, against 64k at test scale.
   - At test scale the 64k calls took 0.065 s (`PERFORMANCE_TASKS.md`, O2's breakdown), about
     1 µs per call, mostly call overhead. The same Python path at full scale takes **about
     3–8 s per step on one thread**: as large as the CPU hashing this plan removes.

So the plan has to settle four costs together: leaf hashing, the two copies, and challenge
generation. Which of them sit on the critical path depends on what overlaps what (Section 4.4),
and that depends on numbers only the H100 machine gives (Section 6).

## 2. Why the shape of the data decides GPU speed

- **SHA-256 is a chain.** Each 64-byte block needs the previous block's result, so one
  message is one sequential job. A 5 GB leaf would be one GPU thread's work.
- **BLAKE3 is a tree inside.** It hashes 1 KB chunks independently and combines them, so one
  large message parallelizes.
- **The Merkle tree is a tree too.** If large objects are cut into fixed-size chunks
  (former F12, now under F4), the chunks supply the parallelism, and any hash function only
  sees small messages. Example: the 5 GB logits in 64 KB chunks become about 80,000 chunks,
  hashed at once.
- **Per-operation cost differs on a GPU.** A GPU has no SHA instructions, and SHA-256 spends
  64 rounds per 64-byte block against BLAKE3's 7, so a well-written BLAKE3 kernel should do
  roughly a third to a quarter of SHA-256's integer work per byte. That estimate counts
  operations; it is not a measurement. The choice between them on a GPU is therefore a trade:
  a maintained library (SHA-256) against a faster function with no maintained GPU code
  (BLAKE3).

## 3. What exists, and what was measured

| | SHA-256 | BLAKE3 |
|---|---|---|
| GPU library | NVIDIA cuPQC-Hash: SHA-2, SHA-3, SHAKE, and a Merkle-tree API. 388–891 GB/s on 8 KB messages (RTX PRO 6000 Blackwell, NVIDIA's figures) | none maintained; research code, about 86 GB/s on a V100 (unoptimized, older GPU) |
| CPU, one thread (measured 2026-10-06, Apple M5 Pro) | 3.40 GB/s | 2.67 GB/s |
| CPU, 1 MiB pieces on 8 threads (same) | 22.5 GB/s | 18.3 GB/s |
| SHA3-256, one thread (same) | 1.11 GB/s | — |

Every CPU figure is from one ARM machine with SHA instructions and 128-bit vectors. The x86
server beside an H100 has SHA instructions too, but BLAKE3 there uses AVX-512 and typically
runs 2–3× faster per core than hardware SHA-256. So on the server CPU BLAKE3 is likely the
faster function, and on the GPU it likely is too (Section 2). Neither is measured; Section 6
measures both.

## 4. The proposal

### 4.1 Chunked objects hash as their own chunk tree (proposed, independent of the function)

A large object is not split into many leaves of the outer tree. Instead its **leaf hash** is the
Merkle root of its own chunks, so the outer tree keeps one leaf per object (P9b):

1. An object whose encoding fits in one chunk of `Z` bytes keeps `H_leaf(x) = H(0x00 ‖ x)`.
2. A larger object is split into chunk 0 = its header, then its payload in `Z`-byte pieces in
   C order, the last piece short. Each chunk hashes as `H(0x02 ‖ chunk)`, the chunk tree is
   built with the node rule `H(0x01 ‖ l ‖ r)` and P9a's promotion, and its root is the
   object's leaf hash. Putting the header in its own chunk keeps every payload chunk aligned
   to the tensor's memory, so the GPU hashes zero-copy views.
3. Whether an object is chunked, and into how many chunks, follows from `C` (its shape and
   dtype) and `Z`, so the prover has no choice (S9b). `Z` is a declared argument of `C`.

Why this layout and not a flat list of chunk leaves:

- The per-object comparisons stay as they are: check 4 compares a record's leaf hash with its
  leaf in `h_D` (records are about 1.5 KB, never chunked), checks 0 and 8 compare weight
  tensors with published hashes, and check 7 compares `W_t` with the previous `W_{t+1}`. In a
  flat list a large tensor would have no single hash, and its chunks would not form a clean
  subtree, because P9a splits at powers of two.
- The challenge label `m` still names one leaf (P8a).
- Test scale keeps 7,661 leaves, so the leaf counts in `verification/CLAUDE.md` don't change;
  only the digests do.
- It answers F12's objection, "a second level of indexing": the second level lives inside the
  leaf hash, not in the leaf order.

### 4.2 The Merkle hash function and where it runs (open; Section 6 settles it)

Candidates:

- **(a) SHA-256 on the GPU through cuPQC-Hash.** The leading candidate of the first draft. A
  maintained NVIDIA library, a function no reviewer questions, and test scale runs the same
  function through `hashlib`.
- **(b) BLAKE3 on the CPU, overlapped with GPU work.** No change to S9a and no new dependency.
  It costs no GPU time when it hides under GPU work: under F1c the step is in host RAM, so
  the verifier's CPU threads can hash while the GPU runs checks 5 and 6, and the prover's
  can hash step `t` while the GPU trains step `t + 1`. It wins if the measured CPU rate keeps
  up with the GPU work it overlaps (Section 6, rule R1).
- **(c) BLAKE3 on the GPU with our own kernel.** Likely the fastest on the GPU. Its risk is
  engineering time and self-written cryptographic code. The correctness risk is smaller than
  the first draft said, since every digest is checked bit for bit against the reference
  implementation, which is the same test (a) needs.
- *SHA3-256 or SHAKE for the Merkle tree:* cuPQC supports them. They were rejected in the first
  draft for being 3× slower on the CPU, which is the wrong platform for the GPU path; they stay
  out for now because nothing points to them being faster than SHA-256 on the GPU, and Section 6
  measures SHAKE anyway for 4.3.

**Function and device are coupled to test scale only through the digest.** Whatever runs at full
scale, test scale computes the same digest on the CPU, so the device is a config flag, not a
code fork (§8.A.5).

### 4.3 Challenge generation (open; Section 6 settles it)

The first draft kept challenges on BLAKE3 because "challenge bytes are tens of MB per step".
That holds at test scale only: at full scale they are 2–3 GB and 2–8 million XOF calls per step
(Section 1, item 6). Candidates, all verifier-side (the prover never derives challenges):

- **(a) BLAKE3, batched natively on CPU threads.** No spec change. One native call generates many
  labels' outputs, removing the per-call Python overhead. Off the critical path if it runs
  ahead of check 5 while the GPU works (challenges depend only on the claimed root, F15a).
- **(b) SHAKE256 on the GPU through cuPQC.** SHAKE256 is an extendable-output function (it emits
  as many bytes as asked), designed for exactly this, and keying by prefix is safe for it
  (a sponge has no length extension). Two hash families in the spec: SHA-2 for the tree,
  SHA-3 for challenges.
- **(c) SHA-256 in counter mode on the GPU through cuPQC**: `HMAC-SHA-256(h, label ‖ counter)`
  blocks concatenated. One function for the whole protocol, at the price of the "second
  construction" S9a rejected SHA-256 for.
- **(d) BLAKE3 on the GPU**, only if 4.2(c) is chosen, reusing that kernel.

**S9a's main reason, one primitive instead of two, is reversed by 4.2(a) with 4.3(a) or (b).**
The first draft said S9a's reason "no longer applies"; it applies and is outweighed only if the
measurements show the speed is worth a second primitive. The pairings with one primitive are
4.2(b) or (c) with 4.3(a) or (d) (all BLAKE3), and 4.2(a) with 4.3(c) (all SHA-256).

### 4.4 Overlap is part of the plan, not a later extra

The costs of Section 1 add up only if they run in series. The plan pipelines them:

- **Prover.** Each product is hashed right after it is produced, on a second CUDA stream (GPU
  hashing runs on integer units while training runs on tensor cores), and copied to host RAM on
  the copy engine, both while training continues. Or, under 4.2(b), the CPU hashes the host
  copy while the GPU trains the next step. The step's root is not needed to keep training.
- **Verifier.** Challenges for step `t` are generated while the host-to-GPU copy of step `t`
  runs. Leaves are hashed on the GPU as they arrive, or on the CPU from host RAM, while the GPU
  runs the checks.
- What remains on the critical path per side is roughly the largest of: GPU work, copy, and
  hashing (if not hidden). With the Section 1 figures the copy is the largest for every model,
  2–5× the training step, so the copy, not the hash function, may set the full-scale overhead.
  That is recorded here as a finding; reducing the transcript's bytes is outside this plan.

## 5. Security notes to confirm when deciding

- SHA-256 gives 128-bit collision resistance, above the protocol's margin `λ = 25` plus the
  grinding budget. Merkle binding (spec §4.2) needs only collision resistance. The chunk tree
  of 4.1 is a Merkle tree under the same `H`, so it binds by the same argument.
- Grinding (a cheating prover re-rolling `h` for fresh challenges) is priced as a **count** of
  attempts, `log₂G = 52` (S9b), not as a hashing cost per attempt. Appendix §12.5 puts each
  attempt's cost in re-evaluating the residual, not in re-hashing. Chunking changes the
  re-hash per attempt very little either way, since leaves of tens of KB (the attention
  products) already exist. The appendix's grinding numbers need no restatement; one sentence
  noting that chunked objects keep the `~log(leaves)` re-hash (S6f) is enough.
- The encoding must stay injective and rigid (S9b). 4.1's chunk rule is fully determined by
  `C` and `Z`, and the `0x02` prefix separates chunk hashes from leaf and node hashes.
- If 4.3(b) or (c) is chosen, P8b's sampling rule stays: only the source of the XOF bytes
  changes, so its exactness and uniformity arguments carry over unchanged.

## 6. The H100 experiment

One benchmark session on the target machine (EQ14: one H100, 188–320 GB of host RAM) settles
the open choices. It needs no verified run: it measures components on synthetic data of the real
sizes, plus one plain training step per model. Every result goes to one JSON file with the
machine record. Thresholds below marked *(hand-picked)* are judgment calls, not derived.

### 6.1 Measurements

- **E0 — Machine record.** GPU model and form factor (SXM or PCIe), driver and CUDA versions,
  CPU model, physical cores, AVX-512 and SHA-NI flags, host RAM, PCIe generation and link width,
  NUMA layout. Every later number is read against it.
- **E1 — CPU hashing throughput.** SHA-256 (OpenSSL through `hashlib`) and BLAKE3 (the `blake3`
  package), on a 4 GB buffer in host RAM, at 1, 8, 16 and all physical cores, at chunk sizes
  8, 16, 64, 256 KB and 1 MiB, and on whole objects of the real leaf sizes (16 KB attention
  products up to the 5 GB logits). Report GB/s through the Python path as built and through
  the tools' native multi-threaded paths (`openssl speed -multi`, `b3sum --num-threads`) as the
  ceiling. Also run it while a host-to-GPU copy streams from the same RAM, to see memory
  bandwidth contention.
- **E2 — GPU hashing throughput.** cuPQC SHA-256, batched over a 4 GB buffer on the GPU, at the
  same chunk sizes, including the chunk-tree and outer-tree node levels under our prefixes and
  P9a's promotion. First check whether cuPQC's Merkle-tree API supports those rules; if not,
  build the node levels from batched 65-byte SHA-256 calls. As the BLAKE3-on-GPU proxy for
  4.2(c), run the cited research implementation on the same buffer and mark the result as an
  unoptimized lower bound.
- **E3 — GPU overlap.** A bf16 GEMM loop of the training's largest shapes on one stream and
  E2's hashing on a second stream: each rate alone, then together. This says how much of GPU
  hashing hides under training (prover) and under the verifier's matrix-vector work.
- **E4 — Copies.** GPU-to-host and host-to-GPU, pinned and pageable memory, in 1 GB and 8 GB
  pieces up to 64 GB total. Alone, then concurrent with E3's GEMM loop, and concurrent with
  E1's CPU hashing.
- **E5 — Challenge generation**, for Llama-3.2-1B's product list (the most calls: 8.3M per
  step, 2.8 GB), including P8b's conversion to fp32:
  - (a) the current Python path, per call, on this CPU;
  - (a′) BLAKE3 batched in native code on all cores (a small extension, or Rust through
    PyO3, whichever is quicker to build for the benchmark);
  - (b) cuPQC SHAKE256 and (c) cuPQC SHA-256 in counter mode, batched on the GPU, with the
    conversion as one GPU kernel.
- **E6 — Step times per model.** One plain bf16 SGD training step at 128 × 128 tokens with eager
  attention, for each of the four models: the prover's lower bound. The same step with fused
  attention (`attn_implementation="sdpa"`) gives eager's cost, which F16 reports as context
  (`DECISIONS_FULL_SCALE.md` F16). If the verified run already
  runs on CUDA for a model, also its prover step with capture, and the verifier step split by
  stage (copy, glue, checks 5 and 6, hashing). Otherwise the verifier's GPU time is estimated
  from its matmul-vector work and labeled as an estimate.

### 6.2 Decision rules

From E0–E6, a cost model per model and side: critical-path time per step with the overlaps of
4.4, and GPU-seconds per step. Both go in the decision record; wall-clock is primary, because
EQ14's timed runs report it. (EQ1c's hashing counts, bytes and calls, change only through
chunking, by the chunk count.)

- **R1 — Device for leaf hashing.** CPU hashing (4.2(b)) wins if, on every model and both
  sides, E1's measured rate under E4's contention hashes the step in no more than 90% of the GPU
  work it overlaps (E6) *(margin hand-picked)*. It then costs no GPU time and changes no
  decision. Otherwise the GPU path wins.
- **R2 — Function on the GPU.** If R1 picks the GPU and E3 shows SHA-256 hidden on a side stream
  adds under 5% to the critical path *(hand-picked)*, the function doesn't matter for compute,
  and SHA-256 through cuPQC (4.2(a)) is chosen for the maintained library. If it adds more,
  the trade between 4.2(a) and a self-written BLAKE3 kernel (4.2(c)) goes to the user, with
  E2's proxy as the evidence.
- **R3 — Chunk size `Z`.** Among the measured sizes, the one with the highest end-to-end rate
  including tree levels, on the chosen device; on a tie within 5% *(hand-picked)*, the larger
  `Z` (fewer hash calls and node hashes). Test scale uses the same `Z` unless item 9 of Section
  7 shows a regression, under the mirroring rule.
- **R4 — Challenges.** If E5(a′) on CPU threads can run ahead of check 5 and hide under the
  verifier's copy and GPU work (same 90% margin as R1), keep BLAKE3 (4.3(a)): no spec change.
  Otherwise move challenges to the GPU, and the choice between 4.3(b) SHAKE256 and 4.3(c)
  SHA-256 counter mode goes to the user (designed XOF against one primitive), unless E5 shows a
  speed gap above 5%.
- **R5 — Pipelining.** Overlap a copy with training (prover) or checks (verifier) if E4 shows it
  slows the GPU work by under 5% *(hand-picked)*. If the copy then stays the largest critical
  term, that is reported to the user as a new full-scale item (transcript bytes moved per step),
  not settled here.

## 7. Action items if approved

**Before deciding**
1. Check cuPQC-Hash's license and distribution terms for use in a published research artifact.
2. Run the H100 experiment of Section 6. The benchmark scripts are written with the
   implementation of the other full-scale tasks, once every full-scale decision has been gone
   through (user, 2026-10-07), not ahead of them. They live outside the protocol packages and
   import nothing from them except the leaf-size list.

**Design (docs)**
3. Record the decisions as F15b and following in `DECISIONS_FULL_SCALE.md`, and as a revision
   of S9a in `DECISIONS_SETUP.md` if the hash function or the challenge construction changes,
   with the measurements and the reasoning. Close the chunking bullet (former F12) inside F4.
4. Amend the spec only where needed: §4.2 names a collision-resistant `H`, not BLAKE3, so a
   function swap needs no change; the chunk tree of 4.1 needs one sentence on how a large
   object's leaf hash is formed.
5. Update the reference block (§6, the leaf encoding) and add the chunk-tree sentence to the
   sizing appendix's grinding note (Section 5). Its grinding numbers stay.

**Implementation (test scale, one setup task)**
6. `commitment/merkle.py` and `commitment/leaves.py`: the chosen leaf and node hash, and 4.1's
   chunk tree inside the leaf hash. If challenges change, `matmul_check/challenges.py` too.
7. Update `verification/CLAUDE.md` (encoding, leaf hash) and the golden tests: `h_D` and every
   pinned root change. Leaf counts such as 7,661 stay.
8. Re-run the affected runs once, together with C5 and C6, which already repeat them: A11
   (calibration), A12 (honest steps), A13 (cheats and sweep), A14 (store cross-check), B7
   (plain baseline). Bands shouldn't move, since hashing doesn't touch the arithmetic.
9. Confirm the test-scale hashing time doesn't regress (now 0.24 s per side for 2.6 GB). At
   tens-of-KB chunks the 2.6 GB becomes about 160k chunk hashes, so the chunk loop must run in
   native code or a thread pool, not one Python call per chunk.

**Implementation (full scale, later)**
10. If the GPU path is chosen: a small CUDA extension calling cuPQC from PyTorch, tested bit for
    bit against the CPU reference on every leaf type, chunk boundary, the root and, if moved,
    the challenges.
11. Hash once per captured batched tensor (for example all `(s, h)` members of one attention
    product in one call), never once per leaf: at full scale a step has 110k–394k leaves, and
    per-leaf Python overhead of a few µs adds about a second.
12. The overlaps of 4.4 that R5 accepts.

## Sources

- [cuPQC-Hash overview (NVIDIA docs)](https://docs.nvidia.com/cuda/cupqc/overview/feature_cupqc_hash.html)
- [cuPQC-Hash feature page](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_hash/cupqc_hash_feature.html)
- [NVIDIA cuPQC](https://developer.nvidia.com/cupqc)
- [BLAKE3 on GPGPU (V100 timings)](https://itzmeanjan.in/pages/blake3-on-gpgpu.html)
- [NVIDIA forum: measured H100 PCIe bandwidth](https://forums.developer.nvidia.com/t/question-about-pci-e-transfer-throughput/328759/13)
