# Hardware environments: time and price of the verified run

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file, under the lock rule in `STATUS.md`: before editing, set this line
> to `🔒 LOCKED — <session/date>` and re-read the file; the instant you finish, set it back to
> `🔓 UNLOCKED`. Never end a turn with the file left LOCKED.

Written on 2026-10-19 as material for a paper appendix. Test-scale profiling showed hashing and
data copying taking most of a verified step, so the user asked which hardware would remove that
cost, and at what price. This file compares the full-scale evaluation's 1,016 verified steps on
today's environment (one H100 SXM) with three kinds of shipping hardware:

- one GPU with enough memory to keep the transcript where it is made;
- several GPUs joined by NVLink;
- the GH200 Grace Hopper superchip, whose CPU and GPU share a fast link.

The model is `cost_models/hardware_environments.py` (`python3 -I hardware_environments.py`).
Every figure in the result tables comes from it.

*Status (2026-10-21): input to F22 (2026-10-20), which chose four H100 SXM over NVLink, with one
B200 as the fallback, and F23's kernel for hashing. The figures are floors at 2026-10-19 list
prices. Measured since: the kernel at 380–600 GB/s on one H100 (FS29) and the slowest NVLink
pair's peer copy at 392 GB/s (FS31).*

## Summary

- **On today's H100 the hardware floor is about twice a plain training step.** If copies and
  hashing run alongside the GPU's work, a verified step takes 1.3–2.5 s, and the 1,016 steps
  cost $2.22–4.26 at $3.99 per GPU-hour.
- **Today's code runs 60–110× above that floor.** FS17 (the H100 re-check of 2026-10-18)
  measured 113–515 s per verified step, which comes to about 64 h and $255 for the 1,016 steps.
  The code's per-product host work sets today's cost, not the hardware. The best hardware here
  lowers the floor by 2–4×.
- **The GH200 is the cheapest environment, at $1.11–1.67** for the 1,016 steps at Lambda's
  $2.29 per hour. Its CPU–GPU link copies at 145–297 GB/s, so the copies and the CPU hashing
  hide behind GPU work. It keeps today's code path (transcript in host memory, BLAKE3 on CPU
  threads), on an Arm host.
- **A GPU with 180–288 GB of HBM (B200, B300, MI355X) is the fastest single GPU but not the
  cheapest.** It runs the steps in 0.42 h for $2.30–3.77. Its hourly price is about twice the
  H100's, and hashing on the GPU runs in series with training. The H200 gains nothing over
  today and doesn't fit Qwen2.5-1.5B.
- **Four H100s over NVLink are the fastest overall and the most expensive.** They take
  0.27–0.32 h when prover and verifier overlap across steps, for $4.26–5.07, and need the
  verifier split across three GPUs, which the code doesn't do.

## Scope

- **One machine for both roles.** The comparison prices the evaluation's setting: both roles
  run on one machine (EQ14, one H100), and the transcript passes between them in host RAM
  (F1c). *(Written before F22: full scale now runs on four H100s, the transcript passes by peer
  copy, and EQ14 is revised for GPU-seconds before FS14.)* If the verifier runs on its own machine, the transcript crosses a network, and the copy
  floor returns at the network's speed; `cost_models/links.py` covers that case.
- **Floors, not predictions.** Each time is what an environment's links, hash rates and GPU
  allow, with the code assumed to keep up. It doesn't predict how today's code runs there.
- **List prices of one day.** Prices are on-demand prices per GPU-hour, read on 2026-10-19. The
  ranges show the spread across providers on that day.
- **No protocol change.** Every environment commits and checks every product, as the spec
  does. Two directions that change what is committed or trusted are parked: hashing inside the
  matmul kernel (O9, `PERFORMANCE_TASKS.md`) and a TEE in place of the proof (E3,
  `EVALUATION_TASKS.md`). The section "Related items" gives their numbers.

## Background

Each verified step produces a transcript of `S` GB: every matmul product of the step, plus the
weights `W_t` and `W_{t+1}`. On today's environment the bytes travel in three stages:

1. The prover's GPU computes the step and copies each product to host RAM over PCIe Gen5.
2. The host's CPU threads hash the products with BLAKE3 into the Merkle tree whose root commits
   the step.
3. The verifier copies the products back to the GPU, rehashes them against the root (check 2),
   and runs its checks on them, including check 5's Freivalds tests.

So each side moves `S` bytes across the link and hashes `S` bytes. Copying and hashing run on
different hardware and pipeline with each other, so a side's data path takes
`max(S / link, S / hash)`. If the data path runs alongside the GPU's work `t`, the side takes
`max(t, data path)`; if not, `t + data path`. An environment lowers this floor in one of four
ways:

- It keeps the bytes where they are made, so nothing is copied.
- It moves them over a faster link.
- It hashes them on faster or more hardware.
- It trusts hardware in place of checking the bytes (the TEE route, E3).

## Inputs per model

The batch is the full-scale one: `n_s = 128` sequences of `n = 128` tokens, in micro-batches of
`B = 8` (FS17).

| Model | `S` (GB) | `T`, plain step on H100 (s) | GPU peak (GiB) | Peak + transcript (GiB) | FS17 verified step, prover + verifier (s) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 36 | 0.50 | 41.0 | 74.5 | 46.1 + 69.0 |
| Llama-3.2-1B | 52 | 0.93 | 55.7 | 104.1 | 54.2 + 107.2 |
| Falcon3-1B | 58 | 1.15 | 59.1 | 113.1 | 35.9 + 77.0 |
| Qwen2.5-1.5B | 70 | 1.20 | 76.9 | 142.1 | 70.2 + 444.8 |

- **`S`** is F17's analytic count (34–67 GB, which reproduces F1a's 32–64 GB within 5%) plus
  F4b's estimate for the output layer's row blocks (+1.9 to +3.76 GB). Only Llama-3.2-1B's base
  was measured, at 48.3 GB (FS11).
- **`T`** is FS17's plain step on one H100 SXM in the verified run's configuration: eager
  attention, which exposes the attention products as matmuls, and the output layer in 8 row
  blocks; 10 steps each (`DECISIONS_FULL_SCALE.md`, F4). With fused attention a plain step takes
  0.23–0.41 s (E6, `STATUS.md`); the verified run doesn't use it.
- **GPU peak** is the larger of FS17's prover and verifier peaks, with the transcript in host
  RAM. Adding the transcript gives what a GPU must hold to keep it in HBM.
- **Steps.** The evaluation verifies 1,016 steps in 8 cells, 254 per model
  (`IMPLEMENTATION_PLAN.md`).
- **FS17's verified step** is today's code. A rerun of Qwen2.5-1.5B took 67.0 + 128.2 s, and
  Qwen2.5-0.5B's check 5 varied about 2× between runs; the variance is unexplained (F4).

## Environments

The comparison has five environments, two of them in two variants:

1. **Today: H100 SXM 80 GB.** PCIe Gen5 copies both ways, BLAKE3 on the host's CPU threads.
2. **One GPU with the transcript in HBM:** H200 141 GB, B200 180 GB, B300 288 GB, or AMD
   MI350X/MI355X 288 GB. The transcript stays in the memory of the GPU that made it, and the
   verifier, on the same GPU, reads it there. Nothing is copied. Hashing has to move to the GPU,
   because CPU threads would read HBM over PCIe, which is the copy again. The GPU hashes on the
   same cores that train, so the model counts hashing in series with GPU work.
3. **Four H100 SXM over NVLink.** GPU 0 trains and sends each product over NVLink to GPUs 1–3,
   which hash their shares in parallel and hold the transcript. The verifier then runs on GPUs
   1–3, split by layer. Three verifier GPUs hold the largest model's 142 GiB with room.
   - **Sequential:** the verifier runs after the prover's step.
   - **Pipelined:** GPUs 1–3 verify step `t` while GPU 0 trains step `t+1`.
4. **GH200 96 GB.** An H100-class GPU with 96 GB of HBM3, joined to a 72-core Grace CPU and
   480 GB of LPDDR5X by NVLink-C2C (450 GB/s per direction, vendor figure).
   - **Grace hash:** the code path of today. The transcript goes to host memory, and BLAKE3
     hashes it on Grace's cores.
   - **GPU hash:** the GPU hashes each product before copying it.

## Rates

Each rate is marked as measured by this project, measured by an independent source, a vendor
figure, or an estimate.

| Quantity | Value used | Sensitivity range | Kind | Source |
|---|---|---|---|---|
| H100 host copy, PCIe Gen5, pinned | 57 GB/s | — | measured, this project | F15 (`DECISIONS_FULL_SCALE.md`) |
| H100 host copy, same path | 53.8–55.3 GB/s | — | measured, independent | ICC, https://www.icc-usa.com/blog/hgx-8x-h100-benchmarks |
| BLAKE3 on the H100 host's threads, one large leaf | 94.6 GB/s | — | measured, this project | F15 |
| SHA-256 on H100 (cuPQC) | 180 GB/s | 150–400 GB/s | estimate | O8: the RTX PRO 6000 figure scaled by the H100's integer throughput |
| SHA-256 on RTX PRO 6000 (cuPQC, 8 KB messages) | 672 GB/s (SHA3-256: 710) | — | vendor | https://nvidia.github.io/cuPQC/blogs/hash-rates/ |
| SHA-256 one-block ceiling (hashcat) | RTX 4090: 1,356–1,417 GB/s; A800: 611 GB/s | — | independent | https://hashcat.net/forum/post-56468.html, https://hashcat.net/forum/post-63595.html |
| SHA-256 on B200 and B300 | 200 GB/s | 150–400 GB/s | estimate | H100's 180 GB/s × SM count (148 against 132); about 400 GB/s if a B200 SM hashes at the RTX PRO 6000's rate per SM |
| A hash on MI355X | 200 GB/s | 150–400 GB/s | assumption | No library and no figure found |
| BLAKE3 on any GPU | — | — | not found | A Vulkan port showed no gain (https://github.com/BLAKE3-team/BLAKE3/pull/80); a CUDA fork ran slower on a T4 than one CPU core |
| H100 NVLink peer copy, one direction | 300 GB/s | — | measured, independent | ICC (busGrind), 270–358 GB/s per GPU pair |
| GH200 C2C, GPU to host | 297 GB/s | — | measured, independent | Comm\|Scope, arXiv 2407.07850 |
| GH200 C2C, host to GPU | 145 GB/s | 375 GB/s (Comm\|Scope) | measured, independent | Verda, 135 GiB/s, https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip |
| GH200 GPU kernel reading LPDDR5X | 235–266 GB/s | — | measured, independent | Verda, Tables 2 and 3 |
| BLAKE3 on 72 Grace cores | 125 GB/s | 60 GB/s | estimate | 72 cores × about 1.7 GB/s: Apple M4 Pro NEON at 2.45 GB/s per core (https://pub.dev/packages/blake3_ffi) scaled by clock |
| Grace memory, STREAM | 325–486 GB/s | — | measured, independent | arXiv 2407.07850; ServeTheHome (480 GB variant) |

- **No GPU hash rate exists for the GPUs compared.** The only bulk figures are NVIDIA's, on a
  workstation Blackwell card with uniform 8 KB messages. The run's leaves vary widely in size.
- **cuPQC hashes SHA-2 and SHA-3, not BLAKE3.** Hashing on an NVIDIA GPU with it changes every
  digest (O8). On AMD, no hashing library was found.
- **Grace's BLAKE3 rate is a guess from another Arm core.** BLAKE3 has a NEON path and no SVE
  path, and no Grace or Graviton BLAKE3 figure was found. The result barely depends on it (see
  "Sensitivity").
- **The two GH200 host-to-GPU figures disagree** (145 and 375 GB/s). The model takes the lower.

## Speed of a plain step against the H100

Each environment's plain step is `T / c`. The factor `c` is picked by hand from the figures
below, since no source measures 0.5–1.5B models.

| GPU | Dense BF16 TFLOPS | HBM | Evidence | `c` used (sensitivity range) |
|---|---|---|---|---|
| H100 SXM | 989 | 80 GB, 3.35 TB/s | reference | 1 |
| H200 | 989 | 141 GB, 4.8 TB/s | MLPerf Training v5.0, Llama 2 70B LoRA: about 10% faster than H100 (NVIDIA, against a v4.1 baseline) | 1.1 (1.0–1.4) |
| GH200 96 GB | about 990 (secondary) | 96 GB, 4 TB/s | the H100's GPU with faster HBM | 1.1 (1.0–1.2) |
| B200 | 2,250 (SemiAnalysis) | 180 GB usable, 8 TB/s | GB200 against Hopper, per GPU: 2.2–2.5× on MLPerf v5.0 (NVIDIA) | 2.0 (1.5–2.3) |
| B300 | 2,250 for HGX B300 (secondary) | 270 GB usable of 288, 8 TB/s | its dense BF16 matches the B200's | 2.0 (1.5–2.3) |
| MI355X | about 2,517 FP16 (AMD brochure) | 288 GB, 8 TB/s | MLPerf v5.1, Llama 2 70B LoRA: within 3–6% of B200 and B300 (AMD) | 2.0 (1.5–2.3) |

- **Why 2.0 for the Blackwell-class GPUs.** It is below the peak ratio of 2.28 because a 0.5–1.5B
  model at this batch fills a large GPU less fully than MLPerf's 70B–405B models.
- **Why 1.1 for the H200 and GH200.** They have the H100's compute; only their HBM is faster.
  The 1.4 bound is the HBM bandwidth ratio, the most a memory-bound step could gain.
- **MI350X.** It is the air-cooled MI355X, with the same memory and lower clocks; the model
  gives both the same factor.

Sources: https://inferencex.semianalysis.com/chips/b200,
https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu,
https://glennklockwood.com/garden/processors/B300,
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf,
https://developer.nvidia.com/blog/?p=101269 (MLPerf v5.0),
https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.1/README.html.

## Prices

The reference price is Runpod's secure-cloud on-demand price where Runpod offers the GPU, since
the project's H100 sessions run there, and Lambda's for the GH200, which Runpod doesn't offer.
The range spans the specialized GPU clouds found; the large public clouds are listed apart.

| GPU | Reference ($/GPU-h) | Range, specialized clouds | Large public clouds | Runpod availability |
|---|---|---|---|---|
| H100 SXM | 3.99 (Runpod) | 3.99 (Lambda 8×, Together) to 6.16 (CoreWeave) | AWS 6.88, Google Cloud 11.06, Azure 12.29 | medium |
| H200 | 5.29 (Runpod) | 5.29 to 6.31 (CoreWeave) | Azure 13.78 | medium |
| B200 | 7.99 (Runpod) | 6.69 (Lambda 8×) to 8.60 (CoreWeave) | AWS 14.24 | low |
| B300 | 8.99 (Runpod) | 8.99 to 9.99 (Together) | — | low |
| MI350X / MI355X | 5.49 (Runpod, MI350X) | 2.95 (TensorWave, MI355X, aggregator) to 5.49 | — | low |
| GH200 96 GB | 2.29 (Lambda, 1×) | 2.29 to 6.50 (CoreWeave) | — | not offered |
| H100, confidential mode | 7.89 (Azure NCC40ads H100 v5) | — | — | — |

Sources: Runpod's API (read 2026-10-19); https://lambda.ai/pricing (read 2026-10-19);
nebius.com/prices, crusoe.ai/cloud/pricing, coreweave.com/pricing and together.ai/pricing (read
2026-10-19 through a summarizer); AWS through instances.vantage.sh; Azure through its retail
prices API; TensorWave through gpus.io. Runpod's community cloud lists the H100 SXM at $2.69.

## Cost model

The model runs in six steps for each model and environment:

1. **GPU compute per side.** The prover's compute is `t = T / c`. The verifier's is `vx · t`,
   with `vx = 1`. This value is picked by hand: a verifier that reads each product once at HBM
   speed and replays the glue does about a step's work. Today's verifier is far above it
   (FS17), and the sensitivity runs use `vx = 3`.
2. **Data path per side,** from the rates in "Rates":
   - Today: `max(S / 57, S / 94.6)` on both sides, alongside GPU work.
   - Transcript in HBM: `S / h` on both sides, in series with GPU work, where `h` is the GPU's
     hash rate.
   - Four H100s: the prover's `max(S / 300, S / 3h)` runs alongside its training. The verifier
     takes `vx · t / 3 + S / 3h`, its work split over three GPUs.
   - GH200, Grace hash: `max(S / 297, S / 125)` for the prover and `max(S / 145, S / 125)` for
     the verifier, alongside GPU work.
   - GH200, GPU hash: `S / h` in series with GPU work, with the copy (`S / 297` out,
     `S / 145` back) alongside.
3. **Side time.** "Overlapped" runs each side's data path alongside its GPU work wherever it
   uses other hardware (copy engines, CPU threads, NVLink, other GPUs). "Serial" runs it all
   after the GPU work. Environments that hash on the training GPU are the same in both.
4. **Step time.** Prover side plus verifier side, one after the other on the same GPUs. In the
   pipelined NVLink variant it is the slower side, since the two run at once.
5. **Totals.** 254 steps per model. Price is wall hours × GPUs billed × price per GPU-hour.
6. **Fit.** A model that doesn't fit is left out of an environment's totals. The column
   "3 smaller models" compares every environment on the same steps.

## Results

### Wall seconds per verified step (overlapped–serial)

| Environment | Qwen2.5-0.5B | Llama-3.2-1B | Falcon3-1B | Qwen2.5-1.5B |
|---|---|---|---|---|
| H100 SXM, PCIe copy + CPU hash (today) | 1.26–2.26 | 1.86–3.68 | 2.30–4.34 | 2.46–4.86 |
| H200, transcript in HBM | 1.31 | 2.27 | 2.74 | doesn't fit |
| B200, B300 or MI355X, transcript in HBM | 0.86 | 1.45 | 1.73 | 1.90 |
| 4× H100 over NVLink | 0.73–0.85 | 1.34–1.51 | 1.64–1.83 | 1.73–1.96 |
| 4× H100 over NVLink, pipelined | 0.50–0.62 | 0.93–1.10 | 1.15–1.34 | 1.20–1.43 |
| GH200, C2C copy + Grace hash | 0.91–1.49 | 1.69–2.52 | 2.09–3.02 | 2.18–3.30 |
| GH200, C2C copy + GPU hash | 1.31–1.68 | 2.27–2.80 | 2.74–3.33 | 2.96–3.68 |

For comparison, today's code measured 115, 161, 113 and 515 s (FS17).

### Data path per side, Qwen2.5-1.5B (70 GB)

| Environment | Prover (s) | Verifier (s) |
|---|---|---|
| H100, PCIe copy + CPU hash (today) | 1.23 | 1.23 |
| B200, B300 or MI355X, transcript in HBM | 0.35 | 0.35 |
| 4× H100 over NVLink | 0.23 | 0.13 |
| GH200, C2C copy + Grace hash | 0.56 | 0.56 |
| GH200, C2C copy + GPU hash | 0.39 | 0.48 |

### The evaluation's 1,016 verified steps

| Environment | GPUs × reference price | Wall (h) | Price at reference | Price over the range | 3 smaller models, at reference |
|---|---|---|---|---|---|
| H100, PCIe copy + CPU hash (today) | 1 × $3.99 | 0.56–1.07 | $2.22–4.26 | $2.22–6.58 | $1.53 |
| H200, transcript in HBM (without Qwen2.5-1.5B) | 1 × $5.29 | 0.45 | $2.36 | $2.36–2.81 | $2.36 |
| B200, transcript in HBM | 1 × $7.99 | 0.42 | $3.35 | $2.80–3.60 | $2.28 |
| B300, transcript in HBM | 1 × $8.99 | 0.42 | $3.77 | $3.77–4.19 | $2.56 |
| MI350X / MI355X, transcript in HBM | 1 × $5.49 | 0.42 | $2.30 | $1.24–2.30 | $1.56 |
| 4× H100 over NVLink | 4 × $3.99 | 0.38–0.43 | $6.13–6.94 | $6.13–10.71 | $4.18 |
| 4× H100 over NVLink, pipelined | 4 × $3.99 | 0.27–0.32 | $4.26–5.07 | $4.26–7.82 | $2.91 |
| GH200, C2C copy + Grace hash | 1 × $2.29 | 0.48–0.73 | $1.11–1.67 | $1.11–4.74 | $0.76 |
| GH200, C2C copy + GPU hash | 1 × $2.29 | 0.65–0.81 | $1.50–1.86 | $1.50–5.27 | $1.02 |

Today's code as measured by FS17, one step per model times 254: 63.8 h, $255 at $3.99.

### Sensitivity

Price of the three smaller models' 762 steps, overlapped, at the reference price, with one input
changed at a time.

| Scenario | Today | H200 | B200 | B300 | MI355X | 4× H100 | 4× H100, pipelined | GH200, Grace hash | GH200, GPU hash |
|---|---|---|---|---|---|---|---|---|---|
| Central values | 1.53 | 2.36 | 2.28 | 2.56 | 1.56 | 4.18 | 2.91 | 0.76 | 1.02 |
| GPU hash 150 GB/s | 1.53 | 2.48 | 2.55 | 2.87 | 1.75 | 4.24 | 2.91 | 0.76 | 1.07 |
| GPU hash 400 GB/s | 1.53 | 2.02 | 1.87 | 2.10 | 1.28 | 4.01 | 2.91 | 0.76 | 0.88 |
| Verifier compute 3 plain steps | 2.94 | 4.11 | 3.73 | 4.20 | 2.56 | 6.11 | 3.51 | 1.52 | 1.78 |
| Speed factors low | 1.53 | 2.53 | 2.76 | 3.11 | 1.90 | 4.18 | 2.91 | 0.83 | 1.10 |
| Speed factors high | 1.53 | 1.98 | 2.09 | 2.35 | 1.43 | 4.18 | 2.91 | 0.69 | 0.96 |
| Grace hash 60 GB/s | 1.53 | 2.36 | 2.28 | 2.56 | 1.56 | 4.18 | 2.91 | 0.81 | 1.02 |

## Findings

1. **Today's copy hides almost exactly behind today's plain step.** `S / 57 GB/s` is 0.63,
   0.91, 1.02 and 1.23 s, against plain steps of 0.50, 0.93, 1.15 and 1.20 s. With full overlap
   a verified step costs about two plain steps, so further gains from hardware come mostly from
   faster compute or from more GPUs.
2. **Removing the copy pays only where the copy isn't hidden.** Without overlap, today's floor
   is 2.3–4.9 s per step, against 0.9–1.9 s on a B200. The same holds if the plain step gets
   faster than the copy.
3. **Hashing on the GPU costs GPU time.** It takes 0.18–0.39 s per side, in series with
   training. For that reason the H200, whose compute is the H100's plus 10%, is
   slower than today's overlapped floor on every model it fits, and the GH200's GPU-hash variant
   is slower than its Grace-hash one.
4. **The GH200 is the cheapest in every scenario.** Its hourly price is below the H100's, and
   its link is 2.5–5× faster than PCIe Gen5. Its price barely moves with Grace's hash rate: at
   half the estimate, the three smaller models cost $0.81 instead of $0.76.
5. **The verifier's own compute is the largest uncertainty.** At three plain steps instead of
   one, every environment's price rises 1.2–2.0×. The ranking doesn't change.

## Fit

- **H200, 141 GB.** It holds the three smaller models (74.5–113.1 GiB) but not Qwen2.5-1.5B's
  142.1 GiB.
- **B200, 180 GB usable.** It holds all four, with about 26 GiB to spare on Qwen2.5-1.5B.
- **B300 (270 GB usable) and MI355X (288 GB).** They hold all four with room.
- **Four H100s.** The verifier's 74.5–142.1 GiB spread over three GPUs is 25–47 GiB each.
- **GH200 96 GB.** The transcript lives in LPDDR5X, and the GPU peak of 41–77 GiB fits.

The model converts vendor GB to GiB, which is the cautious reading. NVIDIA's capacities behave
closer to GiB: the "80 GB" H100 gives about 79.25 GiB usable (FS17). On that reading the H200
holds about 140 GiB, still short of 142.1. The 144 GB GH200 variant would hold about 143 GiB,
which fits Qwen2.5-1.5B with about 1 GiB to spare, too little to rely on.

## Requirements per environment

- **B200 and B300.**
  - A PyTorch build with Blackwell kernels. CUDA 12.8 builds have them, and Runpod's B200 hosts
    run CUDA 13.0–13.2 drivers, which run those builds.
  - A mode that keeps the transcript in HBM (F1c hands it over in host RAM).
  - GPU hashing: O8's cuPQC SHA-256 extension, which changes every digest.
  - A band calibration on the new GPU, because the band file belongs to a device (F21).
- **MI350X and MI355X.**
  - ROCm 7 PyTorch. AMD's training containers ship 2.9 and 2.10 development builds
    (https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html).
  - A hash kernel written in HIP, since no library exists.
  - A band calibration on the new GPU.
- **Four H100s.** The verifier split across three GPUs by layer, products sent over NVLink as
  they are made, and GPU hashing on the receiving GPUs. None of it exists in the code.
- **GH200.**
  - No protocol-side change: the transcript goes to host memory, and BLAKE3 hashes it on CPU
    threads, as today.
  - PyTorch CUDA wheels for Arm (aarch64) and the project's Python 3.14, still to be checked.
  - A check that the H100's band file carries over to the GH200's GPU.
- **Every overlapped figure** assumes that the copies and hashing run alongside GPU work.

## Gaps to measure

1. **GPU SHA-256 on H100 and B200 with the run's leaf sizes.** Every HBM and NVLink figure rests
   on it. O8 sizes this at about half a coder-day plus a 1 h session.
2. **NVLink peer copy on the rented node,** with `nvbandwidth`. The 300 GB/s figure comes from
   busGrind on another vendor's node.
3. **GH200 copies and Grace hashing.** Pinned copies both ways with the run's leaf sizes, since
   the two host-to-GPU figures disagree. BLAKE3 on Grace's 72 cores, which has no published
   figure.
4. **The speed factors on the run's own models.** One plain step per model on each GPU type.
   The MLPerf ratios come from 8B–405B models and from vendors.
5. **The verifier's own compute** (`vx`), once the code approaches the floor.
6. **B300's host link.** One secondary source says PCIe Gen6, while NVIDIA's reference
   architecture shows Gen5 for the B200. It doesn't matter for the transcript-in-HBM path.

## Related items

### A TEE in place of the proof (E3)

A trusted execution environment, here a confidential VM with a GPU in confidential mode, could
replace the proof with a signed hardware attestation. E3 parks it as a comparison for the
paper's discussion. Two estimates on this file's inputs follow, both on Azure's confidential H100
at $7.89 per hour:

- **The TEE replaces the proof.** No transcript is made. The 1,016 plain steps take 0.27 h, and
  the reported overheads of 5–32% raise that to 0.28–0.35 h, which costs $2.21–2.79. That is
  about the low end of today's floor, with a weaker guarantee: it rests on the vendors' keys and on the
  owner, here the prover, mounting no memory-bus attack.
- **Today's protocol runs inside the TEE.** Every byte between CPU and GPU passes an encrypted
  bounce buffer at 4–11 GB/s. The copies alone then take 2.8–7.6 h for the 1,016 steps, which
  costs $22–60.

The overheads are 5–9% for inference (arXiv 2409.03992) and 18–32% for multi-GPU training on
Blackwell (Tinfoil, https://tinfoil.sh/blog/2026-06-23-confidential-computing-overhead, a
secondary source). The bounce-buffer rates come from NVIDIA (about 4 GB/s on Hopper,
https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/)
and Tinfoil (4–11 GB/s on B300). The attacks are TEE.fail (https://tee.fail), Battering RAM and
WireTap (https://www.keysight.com/blogs/en/tech/nwvs/2025/10/22/security-highlight-dram-interposer-attacks-on-confidential-computing).
Outside Azure, confidential GPUs were found only on reserved capacity or in preview.

### Hashing inside the matmul kernel (O9)

On a GPU that keeps the transcript in HBM, each GEMM's epilogue can hash its output tile, so
that product's hash hides under its GEMM. Only the prover gains: the verifier runs no GEMMs and
hashes every byte it holds. The saving is at most the sum, over product classes, of the shorter
of GEMM time and hash time, less the fused kernels' own overhead (about 9% of GEMM time for fused
ABFT checksums, arXiv 2305.01024). For Qwen2.5-1.5B on a B200, with its GEMMs at about 0.19 s
(twice the H100's 400 TFLOP/s, *hand-picked*):

- At this file's 200 GB/s: at most 0.11 s, about 0.09 s after the overhead. The step falls from
  1.90 s to about 1.81 s.
- At F23's BLAKE3 kernel, measured after this file was written (696–740 GB/s on an H100, about
  790 GB/s on a B200 by SM count, an estimate): at most 0.06 s, about 0.04 s after the overhead.

On four H100s it gains nothing, since the owner GPUs hash each product on arrival, off the
training GPU (F22, FS34). No published system hashes this way. O9 in `PERFORMANCE_TASKS.md` has
the derivation.

*Corrected 2026-10-21: the first version counted 0.35 s per side, the verifier's side included
and the whole hash taken as hidden, and gave a step of about 1.2 s.*

### GPU hashing through cuPQC (O8)

O8 is worth reopening if the host copy is removed. Every HBM and NVLink environment here
removes it.

## Options left out

- **AMD MI300A** (CPU and GPU share 128 GB of HBM). It holds Qwen2.5-0.5B, and the 1B models
  only at the margin, since the host side shares the same memory. No cloud offering was found.
- **GB200 and GB300 NVL72.** These racks join Grace to Blackwell GPUs and sell by the rack or
  the 4-GPU instance: about $10.50 per GPU-hour for GB200 (CoreWeave, derived) and $18 for
  GB300 (aggregator). The GH200 offers the same kind of link per GPU at a fraction of the price.
- **Apple unified memory.** It needs no copy, but the GPU's roughly 30 TFLOPS FP32, against the
  H100's 989 dense BF16, makes training the bottleneck.
- **PCIe Gen6 and CXL 3 shared memory.** No GPU with a confirmed Gen6 host link or coherent
  CXL 3 sharing was found.
- **Dedicated hash engines** (Intel QAT, FPGAs, DPUs). The figures found give 2–12 GB/s per
  engine, slower than the host's CPU threads.
