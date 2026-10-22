# Cost Directions — Notes for Discussion

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: a record of explorations, not a decision.** Written on 2026-10-14 by session
25b13fb3 of the cheaper-protocol directions thread. That thread (sessions of 2026-10-12 to
2026-10-14) looks for ways to bring verification toward a few percent of training compute. It
named directions A–F, among them:

- A: re-run the chain. The verifier recomputes the forward pass and the input gradients, and
  checks only the weight gradients with Freivalds.
- B: spot-check random layers.
- C: an online auditor with fresh randomness. Recommended against, because it saves only about
  15%.
- D: exact integer training. Dropped as a cost direction on 2026-10-14 (Section 1).

The thread also kept plain SGD rather than AdamW, and the user rejected committing `P·r` in place
of `P` (2026-10-14). Sampling, the one direction that can reach a few percent, has its own
draft, `SAMPLING_DRAFT.md`.

This file records the answers to four questions asked on 2026-10-14. Nothing here changes a
decision, the spec, the plan, a task file, or the code.

**Superseded in part (2026-10-20).** The user chose four H100 SXM over NVLink for full scale
(F22) with F23's GPU hashing kernel: the transcript moves by peer copy to GPUs 1–3, not through
host RAM (F1c replaced at full scale). The recommendations to keep the host-RAM handover
(Section 3.1, 4.5) and not to rent a GPU pair (4.4) were not taken. The per-side step counts
in Sections 2 and 4 assume the PCIe copy and CPU or 250 GB/s GPU hashing; they are not
forecasts for the four-GPU machine, whose floors are in `ENVIRONMENT_PLAN_DRAFT.md`.

| question | result | outcome |
|---|---|---|
| Does integer training (D) lower the cost? | No. Both sides cost 1.4–1.7× today's per step. Its gain is exactness. | Dropped as a cost direction (user, 2026-10-14) |
| Can the GPU copy and hash run in parallel with compute? | Yes. Both sides drop 23–32%, and the copy becomes the floor. | No action. The hashing plan's pipelining rule (R5) decides. |
| Can a mix of recompute and Freivalds remove the copy? | Only by recomputing. The cheapest mix is direction A, at 1.7–1.8× both sides, 15–21% below re-execution. | No decision |
| Does a faster link or more GPU parallelism help? | Only the link helps. The best link leaves both sides at 2.1–3.4×, at or above re-execution. | Recommended keeping the host-RAM handover (F1c); not taken: the user chose four H100s over NVLink (F22, 2026-10-20). |

**Assumptions.** All rates are **assumed**. They come from `HASHING_PLAN_DRAFT.md` Section 1 and
F17's floor model, the same rates `SAMPLING_DRAFT.md` uses: training at 400 TFLOP/s, GPU↔host
copy at 50 GB/s (pinned memory), GPU hashing at 250 GB/s, a read from GPU memory (HBM) at
2.5 TB/s, and `k = 21` Freivalds vectors. A batch is 128 sequences of 128 tokens.

**Unit.** Costs are in training steps. One step is `6 · params · tokens / 400 TFLOP/s`:

| model | step |
|---|---|
| Qwen2.5-0.5B | 121 ms |
| Llama-3.2-1B | 304 ms |
| Falcon3-1B | 410 ms |
| Qwen2.5-1.5B | 379 ms |
| Llama-3-8B (reference only) | 1,973 ms |

"Both sides" means the prover's training step, plus its copy and hash, plus the verifier's whole
check.

**Scripts.** The scripts are in `cost_models/`, next to this file (Section 5).

1. Integer training (direction D)
2. Overlapping copy and hash with compute
3. Host RAM and copy-free mixes
4. Link upgrades and a RunPod machine
5. Scripts

---

## 1. Integer training (direction D)

D trains in int8 with int32 accumulators. The verifier then checks every product exactly, with
Freivalds' algorithm modulo a prime `p ≈ 2⁶¹`. **D raises the cost.** The exact check needs each
product's full 4-byte sum in the transcript, in place of a 2-byte bf16 value. Copying and hashing
those bytes is most of the cost.

### 1.1 Exact checking

- An int8 × int8 product, summed in an int32 accumulator, is exact as long as no sum overflows.
  An honest residual `A(B·r) − C·r` is exactly 0, so the check needs no tolerance.
- The verifier works modulo `p`. If the prover inflates one entry by 1, the residual is `1·r_j`.
  That residual is 0 only when `r_j = 0`, a chance of 2⁻⁶¹ per vector, however small the cheat.
- The prime must exceed every possible difference. Honest sums stay below 2³¹, so a 61-bit prime
  works. With a 31-bit prime, a cheat that adds exactly `p` passes.

### 1.2 Transcript bytes

The transcript ships each product's output, not its operands. The verifier rebuilds the next
operands by replaying the glue: softmax, norms, and activations. Int8 operands therefore save
nothing. The 2026-10-12 sketch's "half the bytes" counted operands, so it was wrong.

An exact output is the whole int32 sum. A sum of `q` terms can reach `q·127²` at most:

- `q = 896` (Qwen-0.5B's width) needs 25 bits, and `q = 16,384` (weight gradients, summed over
  tokens) needs 29 bits. Each output takes 4 B.
- Attention products (`q = 64–256`) need at most 23 bits, so they pack into 3 B.
- The logits' input gradient sums over the 152k-word vocabulary. It needs 33 bits, so it
  splits into two int32 halves (8 B). These outputs are 0.1% of the bytes.
- Weights stay at 2 × 4 B per parameter, which assumes 32-bit fixed-point master weights.

| model | today (GB) | D (GB) | ratio |
|---|---|---|---|
| Qwen2.5-0.5B | 33.6 | 60.5 | 1.80× |
| Llama-3.2-1B | 48.3 | 82.4 | 1.71× |
| Falcon3-1B | 53.1 | 89.9 | 1.69× |
| Qwen2.5-1.5B | 66.7 | 117.0 | 1.75× |
| Llama-3-8B | 195.3 | 313.4 | 1.61× |

Packing doesn't recover the difference. bf16 keeps 8 significant bits of each sum and rounds
away the rest, and today's tolerance forgives exactly those dropped bits. An exact check needs
them. Honest sums carry about 11–19 significant bits. A perfect lossless coder lands at
0.82–1.07× today's size. That range assumes a standard deviation of 8–32 int8 steps for the
values. No such coder is in the pipeline, and running one costs GPU time.

### 1.3 Cost per step

The model is additive, with no overlap. Int8 runs at 2× bf16 (the floor case) or at 1.3× (the
realistic case), both assumed.

| model | today, both sides | D, int8 at 2× | D, int8 at 1.3× |
|---|---|---|---|
| Qwen2.5-0.5B | 14.4 = 1.75 s | 2.99 s (1.71×) | 3.03 s (1.73×) |
| Llama-3.2-1B | 8.7 = 2.65 s | 4.15 s (1.57×) | 4.23 s (1.60×) |
| Falcon3-1B | 7.3 = 2.99 s | 4.56 s (1.53×) | 4.67 s (1.56×) |
| Qwen2.5-1.5B | 9.5 = 3.62 s | 5.86 s (1.62×) | 5.96 s (1.65×) |
| Llama-3-8B | 5.8 = 11.4 s | 16.2 s (1.41×) | 16.7 s (1.46×) |

The derivation for Qwen2.5-0.5B at 2× runs as follows:

1. Training drops from 121 ms to 61 ms, which is 0.5 of today's step.
2. Copy plus hash per side costs 60.5 GB × 0.024 s/GB = 1.45 s, or 12.0 steps. Today the figure
   is 6.65.
3. The prover costs 0.5 + 12.0 = 12.5 steps.
4. The verifier costs 12.0 + 0.20 (one memory read) + 0.03 (arithmetic) = 12.2 steps.
5. Both sides together cost 24.7 steps = 2.99 s, against 1.75 s today.
6. Against D's own faster step, verification costs 49× training. Today it costs 14.4×.

With copy and hash overlapped (Section 2), each side costs about its copy. D then costs its
byte ratio, 1.6–1.8× today's.

Cutting the vector count from 21 to 2 saves almost nothing. Per side, the arithmetic goes from
0.038 to 0.029 steps. Each of the two vectors carries 61-bit numbers, which int8 tensor cores
process as eight 8-bit slices. Next to the extra 5.3 steps of bytes per side, the saving doesn't
register. Under an online auditor (direction C), D gets down to one vector but still costs
1.4–1.7× the bf16 auditor.

### 1.4 Security, the real gain

| | today, 21 vectors | D, 2 vectors |
|---|---|---|
| Chance that a cheat passes one vector | 2⁻⁴·⁷⁹ at the detection floor; smaller cheats can pass outright | 2⁻⁶¹, whatever the size |
| Smallest data substitution that is guaranteed caught | 4.2% of records, about 5.4 of 128 (4.5% at the coherence measured at test scale) | one token |
| Cost to cover one poisoned record | 44 vectors, about 2.1× check 5's work | already covered |
| Floor even with unlimited vectors | 0.16%, about 0.2 records | none |
| Honest step rejected by mistake | possible at bf16 (F20) | never; an honest residual is exactly 0 |

The derivation of D's `k` runs as follows:

1. The target is today's bit budget: the security level (25 bits), plus log₂ of the steps
   (3.32), plus log₂ of the products per step (16.8–18.6 across the four models), plus the
   prover's grinding tries (52). That gives 97.1–98.9 bits.
2. Each vector supplies 61 bits.
3. `k = ⌈98.9 / 61⌉ = 2`, which gives 122 bits.
4. With fresh randomness and no grinding, only 46.9 bits are needed, so `k = 1`.

The committed transcript then fixes the trained weights bit for bit. A change that the check
can't see has no effect on the model, so the drift-inside-the-tolerance gap closes. Calibration
and its band files, the cancellation guard, check 5's split error model (F18), and the bf16
tolerance item (F20) all go away. Check 6 becomes an exact recomputation.

At test scale, the tolerance-only parts of check 5's measuring took 0.35 s of 0.66 s. The
planned speed-up of check 5's measuring (O2) shrinks that anyway, so this record doesn't count
it as a gain of D.

### 1.5 Price

- **Recipe constraints.** Scales are per row or per column, and powers of two, because a shift is
  exact on any hardware. Per-block scales along the summed dimension aren't allowed. Jetfire, a
  published int8 training method, uses 32×32 blocks and adds the block sums in float, which
  rounds. Stochastic rounding uses a seeded generator that the verifier can replay. Master weights
  are fixed-point, with an integer update rule. The glue, forward and backward (softmax,
  RMSNorm, SiLU, RoPE, cross-entropy), runs in integer arithmetic, or is pinned to one GPU type
  and software build, as F5a does.
- **Bit-identical glue becomes a correctness rule.** Today, a last-bit glue mismatch only widens
  the honest band. Under D, it can flip an int8 rounding, fail the next exact check, and reject an
  honest step.
- **Quality is unproven.** Int8 training has been shown for CLIP with float glue (SwitchBack) and
  for GPT-2 pretraining with block scales (Jetfire). Integer glue has been shown for inference
  (I-BERT) and for CNN training (NITI). No published work shows fully integer fine-tuning of a
  0.5–1.5B LLM. The run first has to show that clean quality and the BadMagic backdoor's success
  match bf16.
- **PoTS comparability.** The run no longer follows PoTS's training recipe.
- **Code.** The verified run can't be an unmodified `from_pretrained` model observed by a
  dispatch mode. Every matmul and glue op becomes an integer kernel, at test scale too. Checks 5
  and 6 change in the spec, which reopens the algorithm stage.
- **Memory.** A step takes 60–117 GB of host RAM for 0.5B–1.5B, against 34–67 GB today. That
  still fits 188–320 GB, with less room.
- **Speed.** H100 int8 tensor cores peak at 2× bf16 (1,979 against 989 TFLOPS dense). Reported
  end-to-end gains are smaller: about 13–25% for SwitchBack, and about 1.4× per transformer block
  for Jetfire. These figures were recalled from the papers and not rechecked, so verify them
  before citing. The verdict holds even at 2×.

### 1.6 Fit with direction A

D barely moves direction A's ratio, because A's recompute speeds up along with training. A's
verifier costs 0.62–0.75 of a step in bf16. This simpler model leaves out the Freivalds
arithmetic; Section 3.3 gives 0.62–0.76 with it. Under D at 2×, it costs 0.70–0.82 of D's own step,
which is 0.35–0.41 of a bf16 step. Under D at 1.3×, it costs 0.63–0.75 of D's own step.

The gain is exactness again. With integer glue, the re-run matches bit for bit on any GPU, so
F5a's same-hardware rule drops and the update check becomes exact. Direction B gains the same
way: a sampled edited layer is caught however small the edit. Its catch rate, about 5.5% per
edited step at a 5% budget, doesn't move.

### 1.7 Outcome

The user dropped D as a cost direction on 2026-10-14. If the security floor becomes the paper's
weak spot, D, paired with direction A, removes it. That floor consists of the 4.2% guaranteed
substitution rate at 21 vectors, the drift inside the tolerance, and the bf16 false-reject risk.
Dropping D has a price: the floor stays where F10 (keep 21 vectors) left it, and F20 still needs
its own fix.

---

## 2. Overlapping copy and hash with compute

**Yes, the GPU can copy and hash in parallel with compute.** The GPU copies over dedicated copy
engines, which are separate from its compute units. `HASHING_PLAN_DRAFT.md` §4.4 already plans
this overlap. The additive tables in Section 1 overstate the cost.

There are two kinds of parallelism:

- **Inside the hashing: many leaves hashed at once.** The 8 CPU threads at test scale did this
  (commit `70193a4`), cutting hashing from 1.5 s to 0.24 s per side. The assumed 250 GB/s GPU
  rate already includes it.
- **Between hardware units: copy engine, compute units, and CPU working at once.** The additive
  model assumes none of this kind.

| model | copy per side | hash per side | both sides, additive | both sides, overlapped | saving |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 5.54 | 1.11 | 14.4 | 11.1 | 23% |
| Llama-3.2-1B | 3.18 | 0.64 | 8.7 | 6.4 | 27% |
| Falcon3-1B | 2.59 | 0.52 | 7.3 | 5.2 | 29% |
| Qwen2.5-1.5B | 3.52 | 0.70 | 9.5 | 7.0 | 26% |
| Llama-3-8B | 1.98 | 0.40 | 5.8 | 4.0 | 32% |

The derivation for Qwen2.5-0.5B runs as follows:

1. The additive prover costs training 1 + copy 5.54 + hash 1.11 = 7.64.
2. In the overlapped prover, the compute units train and hash in at most 1 + 1.11 = 2.11 while
   the copy engine runs for 5.54. The prover costs the larger figure, 5.54.
3. The additive verifier costs copy 5.54 + hash 1.11 + one memory read 0.11 + arithmetic 0.04 =
   6.79.
4. The overlapped verifier costs the larger of 5.54 and 1.26, which is 5.54.
5. Both sides together cost 11.07, against 14.44 additive.

The overlapped prover's compute units are busy for at most 2.11 of the 5.54 steps. For about
60% of each step they wait on the PCIe link.

**CPU hashing doesn't help here.** The hashing plan assumes 15–30 GB/s on a server with 16–26
cores. At that rate, hashing Qwen2.5-0.5B's 33.6 GB takes 9–18 steps, which is slower than the
copy. CPU hashing pays only if its measured rate beats the copy, and the hashing plan's
device-choice rule (R1) settles that from the H100 benchmark.

Overlap needs three things:

- pinned host memory for the asynchronous copies (F21);
- a measurement of how much GPU hashing hides under training (E3), since both share compute units
  and memory bandwidth;
- less than 5% slowdown of GPU work during the copy (R5).

**Effect on the directions.** With overlap, the floor is the copy per side. For Qwen2.5-0.5B to
reach a few percent, the copy has to fall from 5.54 to about 0.05 steps. That takes about 110×
fewer bytes, or a much faster link (Section 4).

---

## 3. Host RAM and copy-free mixes

### 3.1 Host RAM

The transcript goes through host RAM for two reasons.

- **Size.** F1c hands the transcript over in host RAM because it is 34–67 GB per step. Next to
  the weights, gradients, and workspace, it doesn't fit in 80 GB for the larger models. The rough
  peaks in the following table add the transcript, about 10 B per parameter, and fp32 logits.

  | model | rough peak | fits in 80 GB |
  |---|---|---|
  | Qwen2.5-0.5B | ~49 GB | yes |
  | Llama-3.2-1B | ~69 GB | yes |
  | Falcon3-1B | ~78 GB | barely |
  | Qwen2.5-1.5B | ~92 GB | no |

- **The prover and the verifier are different parties.** The verifier checks only bytes that
  reach its own GPU, so in the experiment the copy stands in for sending the transcript. A
  network of 100–400 Gb/s moves 12.5–50 GB/s (assumed), no faster than the PCIe copy.
  `SAMPLING_DRAFT.md` Section 1 makes the same point.

Removing only the copy isn't enough. With the transcript kept on the GPU, today's protocol still
costs 2.1–3.4 steps for both sides, because both sides hash every byte.

### 3.2 Ship or recompute, by class

The verifier needs every value it checks on its own GPU. For each matmul output, there are two
ways to get it there:

- **ship it:** a copy and a hash on each side, then the Freivalds check;
- **recompute it:** the verifier runs the matmul itself.

The break-even point follows from the rates:

1. Copying one byte takes as long as 8,000 FLOPs. Hashing one byte takes as long as 1,600.
2. Shipping one bf16 output (2 B) costs 2 × 2 × (8,000 + 1,600) = 38,400 FLOP-equivalents: copy
   and hash, on both sides.
3. Recomputing that output costs `2q` FLOPs, where `q` is the matmul's inner dimension.
4. Recomputing is cheaper whenever `q < 19,200`.

Forward and input-gradient matmuls have `q = 64–14,336`, so recompute wins. Weight gradients
have `q = 16,384` tokens, near the break-even point. The logits' input gradient has
`q = 128k–152k`, so shipping wins.

The following table gives Qwen2.5-0.5B's classes, in steps:

| matmul class | GB per step | ship and check, both sides | recompute | recompute cheaper by |
|---|---|---|---|---|
| forward linears | 10.0 | 3.98 | 0.24 | 16× |
| attention, forward | 2.1 | 0.85 | 0.004 | 230× |
| logits | 5.0 | 1.99 | 0.09 | 22× |
| input gradients | 8.1 | 3.22 | 0.24 | 13× |
| attention, backward | 3.5 | 1.41 | 0.007 | 190× |
| weight gradients | 1.0 | 0.40 | 0.33 | 1.2× |
| logits' input gradient | 0.03 | 0.014 | 0.09 | 0.2× (ship wins) |

Across the four PoTS-class models, recompute is 6–230× cheaper for every class except the last
two. At Llama-3-8B the smallest ratio is 3.5×. The 2026-10-14 reply gave "4–230×", which
included Llama-3-8B.

### 3.3 The cheapest mix is direction A

A mix that still checks every step does the following:

- recomputes every class except the two with long dot products;
- under plain SGD, reads the weight gradient from the weight change, `(W_t − W_{t+1}) / lr`, and
  checks it with Freivalds against the verifier's own activations;
- ships the small logits' input gradient and checks it with Freivalds.

That mix is direction A. Nothing between today's protocol and A helps: each class moved back to
shipping adds its cost from the table.

**No mix drops the copy without recomputing.** Each matmul's input is the previous output passed
through nonlinear glue, so the verifier needs every entry, not a compressed summary. That is why
committing `P·r` failed. Each entry either crosses the link or gets recomputed. The only way
below A is to leave some computation unchecked, which is sampling.

| model | today, overlapped | today, all on one GPU | mix A | re-run the whole step |
|---|---|---|---|---|
| Qwen2.5-0.5B | 11.1 | 3.4 | 1.82 | 2.13 |
| Llama-3.2-1B | 6.4 | 2.4 | 1.79 | 2.13 |
| Falcon3-1B | 5.2 | 2.1 | 1.69 | 2.13 |
| Qwen2.5-1.5B | 7.0 | 2.5 (doesn't fit) | 1.82 | 2.13 |

"Re-run the whole step" means that the verifier redoes the step bit for bit and compares a hash
of the resulting weights. That is the re-execution in `SAMPLING_DRAFT.md` Section 5.

Under A:

- The only per-step handover is the fp32 weights, 2.0–6.7 GB. Copying them takes 0.33 of a step
  and runs behind the verifier's recompute.
- The verifier needs about training's memory plus one weight copy, so everything fits on the
  GPU.
- The verifier recomputes the forward pass (0.28–0.34 of a step) and the input gradients
  (0.23–0.28). It skips the weight gradients (0.28–0.33).
- The verifier's total is 0.62–0.76 of a step, including Freivalds reads (0.04–0.09), arithmetic
  (0.013 at most), and hashing the weights.

A's price:

- **Verifier compute.** It costs 0.62–0.76 of a training step per checked step, the same order as
  training.
- **An unsized tolerance.** The weight-gradient check compares the prover's update with the
  verifier's recomputed values. Its tolerance must cover rounding through the whole chain and in
  the fp32 weight difference.
- **Hardware.** The verifier must use the prover's GPU type, which F5a already requires.
- **Spec.** The spec reopens, because the transcript no longer holds the matmul outputs.

Against re-running the whole step, A saves 15–21%. Re-running needs deterministic kernels on
every step, a slowdown nobody has measured. Neither option reaches a few percent by itself;
that still needs step sampling.

---

## 4. Link upgrades and a RunPod machine

### 4.1 GPU parallelism

**While the transcript passes through host RAM, only the link or fewer bytes help.** Each side's
time is the longer of its copy and its compute:

- prover = max(training + hashing, copy)
- verifier = max(copy, hashing + reads + Freivalds arithmetic)

The figures for Qwen2.5-0.5B, in steps, run as follows:

1. Training + hashing = 1 + 1.11 = 2.11.
2. The copy takes 5.54.
3. The prover costs max(2.11, 5.54) = 5.54.
4. With training and hashing twice as fast, it costs max(1.06, 5.54) = 5.54, unchanged.

The same holds for all four models: the copy takes 2.59–5.54 steps per side, and hashing takes
0.52–1.11. Extra copy streams don't help, because they share the same x16 link.

Compressing before the copy is the only GPU-side change that shortens it. The prover compresses
on its GPU, which is idle about 60% of the step. The verifier decompresses on its GPU before
hashing. The commitment still hashes the raw bytes, so the spec doesn't change. Its price is a
new dependency, such as NVIDIA's nvCOMP, and an unmeasured ratio. This record assumes 1.2–1.4×
for bf16: the exponent byte compresses, and the mantissa byte barely does.

### 4.2 Links compared

Both sides together, overlapped, in steps:

| link (per direction) | Qwen2.5-0.5B | Llama-3.2-1B | Falcon3-1B | Qwen2.5-1.5B |
|---|---|---|---|---|
| PCIe 50 GB/s (the plan's assumption) | 11.07 | 6.36 | 5.18 | 7.04 |
| PCIe 50 GB/s + GPU compression 1.3× (assumed) | 8.52 | 4.89 | 3.99 | 5.41 |
| PCIe Gen5 ceiling, 64 GB/s | 8.65 | 4.97 | 4.05 | 5.50 |
| 100 GB/s (an 800 Gb/s network card, assumed) | 5.54 | 3.23 | 2.81 | 3.52 |
| NVLink 450 GB/s (H100 SXM peak, assumed) | 3.36 | 2.36 | 2.10 | 2.50 |
| no copy at all | 3.36 | 2.36 | 2.10 | 2.50 |
| *compare:* re-run the whole step | 2.13 | 2.13 | 2.13 | 2.13 |
| *compare:* mix A | 1.82 | 1.79 | 1.69 | 1.82 |

- At 450 GB/s, the copy (0.29–0.62 steps per side) finishes before hashing does, so that row
  equals "no copy at all". A faster link buys nothing more.
- That floor, 2.1–3.4 steps, comes from hashing every byte on both sides. It is at or above the
  cost of re-running the step.
- Getting below the floor takes fewer bytes, which is what mix A does.

The hashing plan's 50 GB/s is a measured pinned-memory figure (`HASHING_PLAN_DRAFT.md` line 36,
from the NVIDIA forum). F1c's "roughly 1.5–3 s for 64 GB" (21–43 GB/s) also covers pageable
memory. PCIe Gen5 x16 carries about 63–64 GB/s per direction, so 50 GB/s is about 80% of the
ceiling, and tuning gains at most about 1.25×.

### 4.3 Ways to improve the link

1. **Reach the assumed 50 GB/s.** This needs pinned host memory, a host buffer on the CPU socket
   next to the GPU, and large transfers. Pageable memory gives 21–43 GB/s. This work belongs to
   F21, the GPU path's host side and its pinned pool. The GPU also has to sit on PCIe Gen5 x16,
   which the benchmark's machine record (E0) logs. A Gen4 host caps near 32 GB/s (assumed).
2. **Compress on the GPU**, about 1.3× (assumed; Section 4.1).
3. **Use a different link.** Two GPUs in one machine talk over NVLink at about 450 GB/s per
   direction (assumed peak). Grace Hopper (GH200) joins CPU and GPU at a similar rate, which keeps
   the host-RAM handover. RunPod doesn't offer it (Section 4.4), and its CPU is ARM, so the
   software stack and wheels need checking.
4. **A network between machines doesn't help.** Cards of 400–800 Gb/s move 50–100 GB/s
   (assumed), no faster than PCIe. A verifier on another machine sees the 50 GB/s or the 100 GB/s
   row.

### 4.4 RunPod

The one RunPod machine with a much faster link is **a pod with 2× H100 SXM**. The prover runs on
GPU 0, the verifier on GPU 1, and the transcript goes card to card over NVLink.

The following facts come from RunPod catalog reads on 2026-10-14 (`list-gpu-types` and
`get-gpu-type`, product POD). Stock and prices change, so re-read them before renting.

- **H100 SXM** (id `NVIDIA H100 80GB HBM3`) costs $3.99 per GPU per hour on secure cloud, billed
  for each GPU in the pod. Community cloud lists $2.69, but its `maxCount` is 1 GPU per pod, so a
  pair needs secure cloud. Secure `maxCount` is 8.
- **Stock for a 2-GPU secure pod** was "LOW" in AP-IN-1 and EUR-IS-3 at the later read. An earlier
  read the same day showed "LOW" in US-MO-1 and US-NE-1. CUDA 12.8 and 13.0 were available, and
  13.2 wasn't.
- **No GH200 or GB200** appears in the catalog.
- The other secure-cloud prices were H100 NVL $3.19, H100 PCIe $2.89, H200 SXM $5.29, H200 NVL
  $3.79, B200 $7.99, and B300 $8.99.
- RunPod's docs say nothing about NVLink inside multi-GPU pods. On a new pod, run
  `nvidia-smi topo -m` first. Between GPU0 and GPU1 it must show `NV18` (18 NVLink links). If it
  shows `PIX`, `PXB`, or `SYS`, the cards talk over PCIe and the pod gains nothing.

The other cards were rejected:

- **H100 NVL:** PCIe cards that pair over NVLink only if the host has bridges fitted. Whether
  RunPod's hosts do is unknown, and it is a different H100 variant (94 GB).
- **H200 SXM, B200, B300:** enough memory to keep the whole transcript on one card. The verifier
  then reads the prover's memory, which measures the "no copy" row rather than a handover. These
  cards also move away from the H100 that EQ14 fixes. Their host link is still PCIe (assumed for
  B200 and B300).

### 4.5 Price of switching to the 2-GPU pod

Taking the pod changes two decisions:

- **F1c, the host-RAM handover.** The transcript goes card to card, so the code needs a
  peer-to-peer copy path in place of the host-RAM store.
- **EQ14, the timing rule.** EQ14 times prover and verifier one at a time on a single H100. On the
  pod, each side gets its own card, and both run while the transcript streams across.

The pod has three further prices:

- It rents twice the GPUs per hour.
- It measures a verifier inside the prover's machine, so the paper must state that assumption.
- The gain stops at 2.1–3.4 steps.

**We recommend keeping the host-RAM handover and not renting the pair.** The price is that
today's protocol stays at 5.2–11.1 steps for both sides until the bytes shrink. No pod was
created. *(Not taken: F22, 2026-10-20; see the note at the top.)*

---

## 5. Scripts

The models are plain Python with no dependencies. Run them from any directory:

```bash
python3 docs/verification/cost_models/integer_d.py
```

Each script reads the model shapes and the product list from `bytes_vs_flops.py` next to it.
To use another copy, pass its path as the first argument. The scripts print the figures that
this file quotes.

| script | section | prints |
|---|---|---|
| `bytes_vs_flops.py` | — | shapes of the four models, the product list, and the ship-or-recompute cost per class (written 2026-10-12, session 0048a3dc) |
| `integer_d.py` | 1 | D's accumulator bits, transcript bytes, `k`, the cost per step, and A's verifier under D |
| `lossless_range.py` | 1.2 | an ideal lossless coder's size under D, for three int8 spreads |
| `overlap.py` | 2 | the additive and overlapped cost per side, CPU hashing, and a 450 GB/s link |
| `mix.py` | 3 | ship or recompute per class, today's protocol on one GPU, mix A, and re-execution |
| `links.py` | 4.2 | today's protocol, overlapped, under each link |

The thread's other scripts (`directions.py`, `online_auditor.py`, `b_walkthrough.py`,
`freivalds_floor.py`, `sampling_numbers.py`, and `audit_cost.py`) stayed in the 2026-10-12
session's temporary scratchpad and aren't in the repo.
