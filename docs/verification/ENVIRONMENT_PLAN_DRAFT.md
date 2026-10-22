# Alternative Environments — Draft Plan

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: decided by the user on 2026-10-20.** B, four H100 SXM over NVLink, with A as the
fallback if B fails its hardware feasibility (`DECISIONS_FULL_SCALE.md` F22), and the hash in
our own BLAKE3 GPU kernel (F23). The plan below is as written on 2026-10-19. Where it differs
from F22 and F23, the decisions hold:

- Its recommendation (build neither before FS27, then A) is superseded: B is built, design 1,
  sequential and then pipelined, in parallel with the kernel (`IMPLEMENTATION_PLAN.md`, FM9
  and FM10).
- G1's kernel is CUDA C compiled through NVRTC, not Triton.
- The summary's copy and hashing times (0.6–1.2 s per side) are floors. Measured with F15b's
  native path, hashing alone took 13–26 s per step, and check 5's serving copies up to 22 s
  more (F22).

The plan as written on 2026-10-19 for F22:
The user asked to plan two environments of the hardware comparison (`HARDWARE_ENVIRONMENTS.md`)
in parallel, to see what each changes in the code and what it buys against the cost of copying
and hashing the transcript:

- **A:** one GPU with enough memory to keep the transcript in HBM, where it is made;
- **B:** four H100 SXM joined by NVLink, GPU 0 training and GPUs 1–3 holding the transcript
  and verifying.

Nothing here changes a decision. Floors and prices come from `HARDWARE_ENVIRONMENTS.md` and
`cost_models/hardware_environments.py` unless marked. The code facts were read on branch
`verification-impl` at `11f2caf`.

## Summary

- **Neither environment pays while the code runs far above today's floor.** FS17 measured
  113–515 s per verified step. Today's H100 floor is 1.3–2.5 s, and its copies and hashing take
  0.6–1.2 s per side. Most of the excess is per-product host work in capture and check 5 (O2),
  which neither environment removes. FS27's profile measures the split.
- **Today's code handles copies and hashing worse than the floor model assumes.** The prover
  hashes only after the whole step, the verifier copies each product to the GPU twice with
  blocking copies, and challenges are built one product at a time. Fixing these pays on the
  H100 and is groundwork for both alternatives (section "Shared groundwork").
- **A, one B200 with the transcript in HBM, is the smaller port.** No byte crosses PCIe, and
  hashing moves to the GPU. One process on one GPU, as today. Floor 0.86–1.90 s per step, $3.35
  for the 1,016 steps. It leaves the H100 that EQ14 fixes, so the PoTS comparison falls back to
  ratios. At the floor the verifier-to-training ratio rises from 1.00–1.26 to 1.50–1.72, because
  GPU hashing runs in series with the checks unless it overlaps them.
- **B, four H100s over NVLink, is the larger port.** The replay's residual stream and its
  gradient aren't committed, so a verifier split by layer is a pipelined replay. Its glue runs
  one GPU at a time, and only measuring and hashing divide by three. Floor 0.87–2.21 s per step
  sequential, about the B200's, or 0.50–1.20 s pipelined, where training sets the pace. It uses
  2–4× today's floor GPU-seconds.
- **We recommend planning both and building neither before FS27.** Build the groundwork that
  pays on the H100 first. If a faster environment is still wanted after that, take A. B buys
  per-step wall time on H100s at more GPU-hours. For the evaluation's total time, four
  independent single-GPU cells on the same four-GPU pod need no code and cost no more.

## Background

### Floor and measured time

`HARDWARE_ENVIRONMENTS.md` gives each environment's floor, the time its links, hash rates and GPU
allow if the code keeps up. FS17 measured today's code on one H100 SXM, before the native
hashing build (F15b) merged:

| Model | `S` (GB) | Plain step `T` (s) | Floor, today's H100 (s) | FS17, prover + verifier (s) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 36 | 0.50 | 1.26–2.26 | 46.1 + 69.0 |
| Llama-3.2-1B | 52 | 0.93 | 1.86–3.68 | 54.2 + 107.2 |
| Falcon3-1B | 58 | 1.15 | 2.30–4.34 | 35.9 + 77.0 |
| Qwen2.5-1.5B | 70 | 1.20 | 2.46–4.86 | 70.2 + 444.8 |

The measured parts that dominate are check 5 (45–411 s) and the prover's capture (about 30 s
for Qwen2.5-0.5B in FS11). Commit and check 2 took 10–28 s and 11–30 s, mostly per-leaf
Python, which F15b's native path removes. A hardware change lowers the floor by 0.4–1.0 s per
step for A and up to 1.3 s for B. A code change that brings check 5 near its floor saves tens
to hundreds of seconds.

### Today's data path

Each step moves its bytes through the following stages. References are to files under
`verification/`.

1. **Weights in.** The prover loads `W_t` from host tensors into the model each step
   (`prover/step.py:209`, `computation/interface.py:183–198`), a host-to-device copy of every
   fp32 weight.
2. **Capture.** `MatmulCapture` (`prover/capture.py:352`) records each product and queues its
   device-to-host copy into a pinned host buffer on a side stream (`_offload`, 482–496). The GPU
   output stays referenced until its copy event completes. The pinned pool is process-wide and
   uncapped (F21).
3. **Labeling.** A few checks sync the GPU (`_check_row_stochastic`, `_check_kv_values` with a
   per-layer host-to-device copy, `instances/decoder.py:904–936`). The transposed `δK̃` member
   is copied a second time on the host (938–949).
4. **Weights out.** `snapshot_weights` copies every weight device-to-host, pageable and blocking
   (`interface.py:201–212`).
5. **Commitment.** `P3.commit` starts after the whole step returns (`runs/loop.py:234–235`).
   Native BLAKE3 hashes the leaves on CPU threads (`commitment/leaves.py:200–300`); leaves of
   16 MiB or more take a finiteness pass in C and then the Python `blake3` package. The Merkle
   tree is built on one thread after every digest exists. Nothing overlaps GPU work.
6. **Hand-off.** `InMemoryStore` holds the pinned buffers and weight tensors on the host
   (`transcript/store.py:136–203`). Leaf encoding refuses a tensor that isn't on the CPU
   (`commitment/encoding.py:80`).
7. **Verifier, hashing.** Check 7 hashes `W_t`, check 4 hashes the records in single-threaded
   Python, and check 2 hashes every leaf again (`verifier/checks.py:133–247`). All on the CPU,
   before any GPU check starts.
8. **Verifier, serving.** The replay copies each product leaf to the GPU (`_fresh`,
   `decoder.py:166–170`; `substitution.py:95`), and check 5 copies it again (`_served`,
   `_served_group`, `checks.py:721–829`). Every copy is blocking. Stacked members and backward
   products are first copied into pageable host temporaries (`decoder.py:1438–1456`,
   `checks.py:744`). The verifier's model loads `W_t` from the host too.
9. **Verifier, challenges.** Each dense product's challenge matrix is generated on the CPU when
   check 5 reaches it and copied to the GPU (`matmul_check/freivalds.py:315`). Nothing is built
   ahead, although every challenge depends only on the claimed root.
10. **Verifier, updates.** Checks 6a and 6b run on the host. Check 6b copies every glue gradient
    device-to-host in fp32 (`checks.py:955–994`).

So per step the code moves the products across PCIe three times (out once, in twice) and the
fp32 weights three times, and hashes `W` bytes up to five times across the two sides. The floor
model counts one copy and one hash per side, overlapped with GPU work.

## Shared groundwork

Both alternatives need the first three items. The other four are speed fixes that pay on
today's H100 as well, and both alternatives assume them in their floors.

| Item | What it does | Needed by | Pays on the H100 |
|---|---|---|---|
| G1 | A GPU hash kernel | A and B | no |
| G2 | Leaf encoding and finiteness on the device | A and B | no |
| G3 | One served copy per leaf, shared by the replay and check 5 | both, as a floor assumption | yes |
| G4 | All challenges in one native batch, ahead of check 5 | both, as a floor assumption | yes |
| G5 | Checks 6a and 6b on the device without per-block syncs | both, as a floor assumption | yes |
| G6 | The band file names the GPU model and software build | A, and B for safety | yes |
| G7 | Hashing overlapped with GPU work on both sides | both, as a floor assumption | yes |

G3 and G4 change check 5's serving, which the check-5 speed session (O2) owns, so they go
through that session.

### G1: the GPU hash kernel

A transcript that stays on a GPU has to be hashed there, since CPU threads would read it over
PCIe, which is the copy again. The repository has no GPU hash; the benchmark's GPU rows are
skipped (`benchmarks/hashing/gpu.py:47–58`). There are two candidates:

- **BLAKE3 in a kernel written for this project.** Every digest stays what the CPU computes, so
  the spec, S9a, the golden tests, `h_D` and test scale don't change, and the native C library
  stays the reference. BLAKE3 splits a message into 1 KiB chunks hashed independently, so a
  70 GB step offers about 70 million parallel chunks. Its price is self-written code in the
  protocol path, checked bit for bit against the C library and BLAKE3's published test vectors
  on every leaf type and on lengths around chunk boundaries. A wrong kernel shows up as a digest
  mismatch, not as a silent soundness gap. Its speed is unmeasured. An operation count puts it
  at 300–600 GB/s on an H100: about 12 integer operations per byte against about 15 trillion
  per second, at 30–50% efficiency *(both hand-picked)*. The only published GPU BLAKE3 figure
  is 86 GB/s on a V100, from unoptimized research code.
- **SHA-256 through NVIDIA's cuPQC (O8).** A maintained library, estimated at 150–400 GB/s on
  an H100. SHA-256 hashes one message as a chain, so large leaves need a chunk tree
  (`HASHING_PLAN_DRAFT.md` 4.1). That changes every digest: S9a is revised, test scale switches
  too, `h_D` and every golden digest are re-pinned, and A11–A14 and B7 rerun. It also needs the
  licence and runs on NVIDIA only.

**We recommend the BLAKE3 kernel, gated by a benchmark.** It keeps every digest, which leaves
the protocol, test scale and the records untouched. If the benchmark shows it below about
150 GB/s on an H100 *(hand-picked: the low end of cuPQC's estimate)*, cuPQC takes its place.
Write it in Triton first: Triton ships with PyTorch's CUDA wheels, needs no compiler on the pod
and also runs on AMD. A CUDA C version through the native build is the fallback if Triton's
integer code is slow. The benchmark runs on one H100 for about an hour, roughly $4.

The kernel takes a batch of leaf descriptors (header bytes, payload pointer, length), as the
native C path does (F15b). The leaf message is `0x00 ‖ header ‖ payload`. Headers are 3 + 4·ndim
bytes, so the prefix `0x00 ‖ header` is 4 + 4·ndim and payload words stay 4-byte aligned inside
each chunk. The finiteness check fuses into
the same read. Digests (32 bytes per leaf, 12.6 MB for Llama-3.2-1B's 394k leaves) go to the
host, where the native Merkle tree is built as today.

### G2: leaf encoding on the device

`_check_payload` (`commitment/encoding.py:76–83`) refuses device tensors, and the payload path
goes through `.numpy()`. A device path must produce the same bytes: the same header, the payload
as little-endian raw bytes (bf16 as its 16-bit pattern), contiguous, and the same non-finite
rejection with the same first-error-in-order rule (`tests/verification/commitment/test_leaves_native.py:172`).
`leaves.py` routes each leaf by device: host leaves to the native C library, device leaves to G1.

*As built (FS30, F23's later choices, 2026-10-20):* only the prover's commit routes by device
(`hash_step_leaves`, called by `prover/step.py` and `transcript/store.py`); check 2 keeps the C
library until FS34. Since F22's host-copy seam (FS32b), `check_payload` accepts a device
tensor, and the C library reads its bytes through a host copy.

*FS34 (2026-10-21, in review):* the kernel moves behind that seam for every hash call. A CUDA
leaf is hashed in place for the prover's commit, check 2, check 7 and check 0's anchor, and the
owner GPUs hash each product on arrival. A host leaf keeps the C library, as does `kernel=False`
(the reference comparisons). The host copy stays for MPS leaves and for F23's once-per-run
per-shape check against the C library.

### G3: one served copy per leaf

The replay and check 5 each copy a product leaf to the GPU (stage 8). The replay already holds
the operands' device copy, so check 5 can measure the same copy, and member stacks can be built
once. On the H100 this halves the verifier's host-to-device bytes (about 70 GB per step for
Qwen2.5-1.5B) and turns the copies non-blocking from pinned memory. In A and B the leaf is
already on a GPU, so the served copy becomes a device-side clone at HBM speed. The F4a rule
that the replay gets fresh tensors stays.

### G4: challenges ahead of check 5

Challenges depend only on the claimed root, which exists before the verifier's turn (F15a).
Generating all of them in one native call (0.39 s on 27 threads for Llama-3.2-1B, F15b) while
the GPU runs checks 2 and 6a, then copying them once from pinned memory (about 2.8 GB, about
50 ms), removes a CPU call and a blocking copy per product.

### G5: checks 6a and 6b on the device

Both read weights and gradients on the host today. On a device they need a form without
per-block host reads (`checks.py:300–363` read scalars per block of 2²¹ entries). O1's fused
kernel is one way; a reduction kept on the device until one final read is the other. Either
must keep `ρ_max` bit for bit, or the user decides.

### G6: the band file's device

The band context records `"device": "cuda"` and no GPU model (`verifier/calibration.py:355–371`),
so an H100 band file would judge a B200 run without a warning. This is F21's second half: record
the GPU name and the software build, and refuse a mismatch.

### G7: hashing overlapped with GPU work

The floor model assumes copies and hashing run alongside GPU work. The code doesn't: the prover
hashes after the step, and the verifier hashes before its first GPU check. On the H100, the
prover can hash each batch of pinned leaves as soon as its copy event completes, and the
verifier can run check 2 on CPU threads while the GPU runs checks 6a and 5, with the root
comparison last as F15a already orders it. In A and B the same overlap uses a second CUDA
stream, which is only partly free, since hashing and training share the SMs. The hashing plan's
experiment E3 measures how much. Two constraints carry over: invariant 6 (nothing changes after
commit, checked by `_version`), and the single global hash worker (`leaves.py:140–148`), which
serializes concurrent callers.

*For B, as built (F22, F23, FS34):* the owner GPUs 1–3, which train nothing, hash the products on
arrival and run check 2 on the kernel, so hashing doesn't share GPU 0's SMs. Each process keeps
its own hash worker (FS33). What stays on the host is the hashing of host leaves (the records).

## Alternative A: one large GPU, transcript in HBM

### Hardware

| GPU | Usable memory | Fits all four models | Reference price | Software |
|---|---|---|---|---|
| B200 | 180 GB | yes, about 26 GiB to spare on Qwen2.5-1.5B | $7.99 (Runpod) | CUDA 12.8 wheels (torch 2.9.1+cu128) carry its kernels |
| B300 | 270 GB | yes | $8.99 | its compute capability may need a CUDA 12.9 or later build (to check) |
| MI350X/MI355X | 288 GB | yes | $5.49 ($2.95 on an aggregator) | ROCm 7 builds; no cuPQC; new determinism knobs |
| H200 | 141 GB | no, Qwen2.5-1.5B needs 142.1 GiB | $5.29 | as the H100 |

**We recommend the B200.** It holds every model, keeps the CUDA stack and its determinism knobs
(S4c, F5a), and costs the least of the NVIDIA options at the floor. The MI355X is cheaper, but
the ROCm port reopens bit identity, the determinism settings and the pinned PyTorch build. The
H200 could fit Qwen2.5-1.5B if the verifier's served copies replaced the stored leaves instead
of duplicating them, but that saving is unmeasured.

### Data path

1. The prover trains on the GPU. Each captured product stays in HBM: either the capture holds
   the tensor itself, guarded by `_version` until commit, or it clones it on a side stream
   (about 18 ms per 70 GB at 8 TB/s).
2. G1 hashes each product on a side stream as it is captured. Digests go to the host, and the
   tree is built there.
3. `W_{t+1}` is snapshotted device to device. `W_t` is the previous step's `W_{t+1}`, already
   on the device.
4. The verifier reads leaves in place. Its served copies (G3) are device clones. Check 2 hashes
   on the GPU, beside checks 6a and 5 where G7 allows.
5. Challenges come from the CPU in one batch (G4), about 2.8 GB across PCIe once.
6. After the verdict the step's tensors are freed, and the allocator's cache is emptied as
   today.

Only digests and challenges cross PCIe. The 46 GiB pinned host pool of F21 disappears.

### Code changes

| Area | Change |
|---|---|
| `setup/config.py` | One flag for where the transcript lives, `VERIF_TRANSCRIPT=host\|device`. Hashing follows the bytes. On a CPU run both values mean host memory, so test scale runs the same path. |
| `prover/capture.py` | `_offload` gains the device branch: hold or clone, then queue G1's hash on the side stream. `HostBuffers` is unused on that branch. |
| `prover/step.py`, `computation/interface.py` | `HOST` becomes the transcript's device for products and weights. `snapshot_weights` and `load_weights` run device to device. |
| `commitment/` | G1's kernel module beside `native.py`; G2 in `encoding.py` and `leaves.py`. |
| `transcript/store.py` | No structural change: `InMemoryStore` holds device tensors, and its `_version` guards work on CUDA tensors. |
| `verifier/checks.py`, `instances/decoder.py`, `substitution.py` | Serving through G3 (device clones). Check 2 and check 7 hash on the device. G4 and G5. |
| `runs/loop.py` | Free the step's device tensors before the next step. |
| `runs/metrics.py`, `runs/probe.py`, `setup/machine.py` | The probe measures the GPU hash rate. The peaks include the transcript. The machine record names the GPU model. |
| Tests | `test_gpu_memory_does_not_hold_the_transcript` (`tests/verification/prover/test_capture_cuda.py:145`) applies to `host` only. CUDA tests for the device path compare roots with the host path bit for bit. Golden digests don't change with BLAKE3. |
| Docs | `verification/CLAUDE.md` and `README.md`: the flag, the device path, and where hashing runs. |

### Memory

A step holds the transcript beside the training or verifier peak: 74.5, 104.1, 113.1 and 142.1
GiB for the four models (`HARDWARE_ENVIRONMENTS.md`, "Inputs per model"). The figure counts the
forward values twice, once as leaves and once as the verifier's copies, so it is a ceiling.
Holding products instead of cloning them in capture lowers the prover's peak by the forward
leaves (12–24 GB), since autograd keeps them anyway.

Qwen2.5-0.5B's 74.5 GiB fits today's H100 (79.25 GiB usable) with about 5 GiB to spare. So A's
path can be developed and tested on the H100 and A40 pods the project already uses, with
SmolLM2 and Qwen2.5-0.5B, before a B200 is rented.

### Floor and price

From the comparison, with GPU hashing at 200 GB/s in series with GPU work:

| | Qwen2.5-0.5B | Llama-3.2-1B | Falcon3-1B | Qwen2.5-1.5B |
|---|---|---|---|---|
| Step on a B200 (s) | 0.86 | 1.45 | 1.73 | 1.90 |
| Hashing per side at 200 GB/s (s) | 0.18 | 0.26 | 0.29 | 0.35 |
| Hashing per side at 300–600 GB/s, BLAKE3 estimate (s) | 0.06–0.12 | 0.09–0.17 | 0.10–0.19 | 0.12–0.23 |

The 1,016 steps take 0.42 h and cost $3.35 at $7.99 per hour ($2.80–3.60 across providers),
against today's floor of $2.22–4.26 on one H100. The speed factor of 2.0 over the H100 is
hand-picked from MLPerf ratios on much larger models; one plain step per model on a B200
measures it.

### Effect on the paper's numbers

- **EQ14.** Timed runs leave the H100, so EQ14's own fallback applies: the time comparison with
  PoTS uses ratios only. EQ12's seconds comparison is lost.
- **The ratio gets worse at the floor.** Verifier ÷ plain step is 1.00–1.26 on today's H100,
  where the copy and CPU hashing hide behind GPU work. On a B200 it is 1.50–1.72, since
  hashing adds `S / h` to a step that is twice as fast. If hashing overlaps the checks on a
  second stream (G7), the ratio returns toward 1.
- **The hand-off disappears from the measurement.** The verifier reads the prover's memory, so
  the run measures no transfer at all. A verifier on its own machine pays `S` bytes over a
  network; the paper must state that the measured setting omits it.

### Risks

- G1's rate is unmeasured, and every A figure rests on it.
- A new GPU means new bf16 kernels, so `s_h`, `τ` and possibly `k` change (P10d). Calibration
  runs inside H1 (F5b) on whatever GPU runs, so the band file is the B200's own. Results then
  describe the B200, and F20's guard question meets new residuals.
- Batched measuring's bit identity per batch count was audited on CPU, x86 and the A40
  (`verification/CLAUDE.md`). The B200 needs the same audit.
- Runpod listed B200 availability as low on 2026-10-19.

### Testing

1. Fast suite on the Mac, `VERIF_TRANSCRIPT=device` on CPU, which equals `host`.
2. SmolLM2 and Qwen2.5-0.5B on an H100 or A40 pod: roots equal to the host path bit for bit,
   honest steps accepted, cheats rejected at their declared step and check.
3. One B200 session: the CUDA build check, one plain step per model (the speed factor), one
   verified step per model, and the peaks.

## Alternative B: four H100 SXM over NVLink

### Topology and roles

GPU 0 trains, as today. GPUs 1–3 each own a third of the transcript: they receive products over
NVLink, hash them for the prover's commitment, and then run the verifier on their share. The pod
must expose NVLink between the cards: `nvidia-smi topo -m` shows `NV18` between each pair, not
`PIX`, `PXB` or `SYS` (`COST_DIRECTIONS_DRAFT.md` 4.4).

### Data path

1. GPU 0 trains. Capture queues each product's peer copy to its owner GPU on a side stream, at
   about 300 GB/s (busGrind on another vendor's node), hidden behind training.
2. The owner GPU hashes the product on arrival (G1). Digests go to the host, which builds the
   tree. The prover's side costs `max(T, S / 300, S / 540)`.
3. The verifier runs on GPUs 1–3. Each serves its own leaves (G3), hashes them for check 2, and
   measures them for check 5. The replay passes the residual stream forward from GPU 1 to GPU 3
   and its gradient back.
4. Challenges come from the CPU in one batch (G4) and are copied to the GPU that measures each
   product.

### Transcript sharding

The owner of each leaf follows from the declared computation, so the prover has no choice
(invariant 7 already fixes the leaf list before the run). The rule: contiguous decoder layers
per GPU, balanced by transcript bytes, with the head blocks (`Λ_b`, `δF_b`, `G_b`, about 15% of
the bytes) and the final norm on GPU 3. `W_t` and `W_{t+1}` leaves go to the GPU that holds
their layer.

### Verifier split

The agent survey of the replay (`instances/decoder.py:1084–1729`) settles what a split can use.
The replay runs one forward pass of the unmodified model with committed leaves substituted, then
autograd backward unit by unit (`_run_unit`, `torch.autograd.grad(u.out, …, grad_outputs=dx)`),
with a graph cut at each layer's input (`_enter_unit`, 1281–1298). Two values cross each layer
boundary and neither is a leaf:

- the forward residual stream `X_j`, which is the embedding plus exact elementwise sums of the
  earlier layers' committed `Y_o` and `Y_down`;
- the backward residual gradient `δX_{j+1}`, which needs the head and every later layer's norm
  backward on its committed `dX` leaves, in autograd's accumulation order.

Inside a layer, the attention members and the products built from that layer's own leaves are
local. The A operands of `Y_q`, `Y_k`, `Y_v`, `Y_gate` and `Y_up` need `X_j`, and `dX_down`,
`G_down`, `dX_o` and `G_o` need `δX_{j+1}`. There are three designs:

1. **Model-parallel replay in one process (recommended).** The replay model's units sit on
   GPUs 1–3. `_enter_unit` already returns the copy each layer receives, so it can move `X_j` to
   that layer's GPU; the backward hands `dx` across the same boundaries. The rotary tables and
   the attention mask move with the units. It keeps the unmodified model, autograd and the
   dispatch mode, and changes no leaf. Its limit: glue runs one GPU at a time, forward and
   backward, while measuring and hashing run on all three, since CUDA work stays queued on each
   GPU while the host moves on, as long as nothing syncs per product.
2. **Commit the boundaries.** `X_j` and `δX_{j+1}` become leaves, and each layer checks that its
   output equals the next committed input bit for bit, the comparison `_enter_unit` makes today.
   Each GPU then replays its layers with no wait, in its own process if wanted. It changes the
   spec's leaf list and every root, adds about 8% to the bytes (fp32, two values per layer), and
   reruns test scale.
3. **Rebuild the boundaries by hand** from committed products. Rejected: it leaves the
   unmodified-model replay, and its bit identity with autograd's accumulation order is unproven.

Splitting by sequence instead of by layer was considered too. Every glue operation is per token,
but each product's rows would then sit on three GPUs, and check 2's per-leaf hash would need
BLAKE3 chaining values merged across cards. Splitting by step (one GPU per step) needs a whole
transcript and a whole verifier on each card, which doesn't fit beyond Qwen2.5-0.5B.

### Process model

One process driving four GPUs is the smaller change: the store stays a Python object holding
tensors on `cuda:1`–`cuda:3`, the determinism knobs are process-wide, and peer copies are
`.to(device, non_blocking=True)`. The GIL then serializes all host work, so today's per-product
host cost doesn't divide by three. One process per GPU divides it, but needs design 2 (or a
pipeline across processes), an IPC store, and a test-scale counterpart. We recommend one process
first.

*As decided (F22, FS33's process model, 2026-10-21):* the verifier stays one process on GPUs
1–3, and the pipelined loop gives the prover a second process on GPU 0, for the GIL's sake. The
sequential loop stays one process. The design, as FS33 builds it:
- **Roles.** The parent process runs the verifier, the harness, the recorder and the verdicts.
  It spawns the prover's process, which rebuilds the config, the model, the data, the schedule
  and any planted fault from picklable specs, and starts from its own copy of `W_0`.
- **One package per step.** The child sends each storage once, as a CUDA inter-process handle
  (shared host memory at test scale), with a table that places each leaf in a storage (offset,
  dtype, shape, stride). Pickling 394k tensors one by one would cost more than the step's
  hashing. The records, the tree, the paths, the loss, the timings, the capture summary and the
  commit's known count go over the pipe. The parent's store reads the leaves from the shared
  storages without copying.
- **Two transcripts at most.** The child starts step `t + 2` only after the parent releases
  step `t` (S3 revised). On release the child checks its own `_version` counters for step `t`
  and frees its storages: the parent can't see the child's in-place writes, so invariant 6
  is checked on both sides, each on its own tensors.
- **Stopping.** After a rejection the parent tells the child to stop, and discards any later
  step the child has made. The run's final weights are the rejected step's `W_{t+1}`, as in the
  sequential loop, sent as host copies, so every verdict and check 8/9's input match the
  sequential run's.
- **Failure.** An exception in the child crosses the pipe with its traceback; a dead child is
  seen by the pipe closing or by its exit code. Either stops the run with an error, never a
  verdict.
- **Per-process state.** Each process keeps its own staging pool, hash worker, cache release,
  clocks and memory peaks. The child's per-step rows join the parent's recorder, and every
  section is timed on one shared clock, so the overlap of the two sides is visible.
- **The setting.** `VERIF_PIPELINED`, off by default, changes no number, so it stays out of
  every config hash: the metrics config and the training config alike.
  `VERIF_VERIFIER_DEVICES` is out of the training config only; the metrics config records
  where the verifier ran. Disk handoff, the counting pass and the profiler trace are
  refused in pipelined mode until a run needs them.

### Sequential or pipelined

- **Sequential.** The verifier starts after the prover's step, as today. GPU 0 idles during
  verification, and GPUs 1–3 idle during training apart from hashing.
- **Pipelined.** GPUs 1–3 verify step `t` while GPU 0 trains `t + 1`. The protocol allows it:
  step `t`'s challenges come from its own root. The harness changes: two transcripts are alive
  (S3 assumes one), the cheat runs stop at the first verified rejection rather than the first
  trained step past it, and the two sides' timings are taken under each other's load. The
  single global hash worker would serialize the two sides' hashing *(as built, each process
  keeps its own; FS33)*.

### Code changes

| Area | Change |
|---|---|
| `setup/config.py` | `VERIF_DEVICE` stays the prover's device; a new `VERIF_VERIFIER_DEVICES` lists the verifier's. On a CPU run the list is `cpu`. |
| `computation/instances/decoder.py` | Per-role devices: today one `DecoderComputation.device` serves the prover and the replay (241–243, 547–609, 652–693). Unit placement, boundary moves in `_enter_unit` and the backward, rotary tables and masks per device. |
| `prover/capture.py` | Route each product to its owner; peer copy on the side stream; the `_version` guard until the copy completes, as `_drain` does today. The canonical index is known only after labeling, so the capture copies each output to a hinted owner, and labeling moves and counts any miss (`DECISIONS_FULL_SCALE.md` F22, implementation choices for FS32, 2026-10-20). |
| `commitment/` | G1 and G2 on each owner device; first error in order taken as the smallest failing index across the three. |
| `transcript/store.py` | A sharded store with the same `leaf(i)` interface, returning each leaf on its owner device. |
| `verifier/checks.py`, `matmul_check/` | Check 5 measures on the product's device, one `MeasureScratch` per device, and judges in canonical order. Checks 2, 6a and 7 run per owner device. |
| `runs/loop.py`, `runs/run_verified.py`, `runs/full_scale.py` | Sequential: device settings only. Pipelined: a two-stage loop, two live transcripts, truncation by verified step. |
| `runs/metrics.py`, `check5_profile.py`, `runs/probe.py`, `setup/machine.py` | Per-device syncs, peaks and clocks (today one `_device`, and `machine.py:192–193` reads GPU 0 only); the NVLink topology in the machine record. |
| Tests | Bit identity of the split replay against the one-GPU replay, on a two-GPU pod. A Mac run exercises the code with every device set to `cpu`, which can't catch a wrong device move. |

### Memory

The cost model puts the verifier's 74.5–142.1 GiB over three GPUs at 25–47 GiB each, and the
prover's GPU holds no transcript. Sharding by layer is uneven, since the head's 15% sits on
GPU 3. Pipelined, each verifier GPU holds two thirds of a transcript: about 69 GiB per GPU for
Qwen2.5-1.5B, tight against 79.25.

### Floor and price

The comparison divides the whole verifier by three. Under design 1 the glue doesn't divide. The
table below keeps a share `g` of the verifier's GPU work in series, with `g` = 0.4–0.6
*(hand-picked: the elementwise share of an eager-attention step, unmeasured)*, and the verifier's
total GPU work at one plain step (`vx = 1`, as in the comparison). Script:
`scratchpad/env_floor.py` of session c5cf977f.

| | Qwen2.5-0.5B | Llama-3.2-1B | Falcon3-1B | Qwen2.5-1.5B |
|---|---|---|---|---|
| Verifier, comparison's even split (s) | 0.23 | 0.41 | 0.49 | 0.53 |
| Verifier, glue in series (s) | 0.37–0.43 | 0.65–0.78 | 0.80–0.95 | 0.85–1.01 |
| Step, sequential (s) | 0.87–0.93 | 1.58–1.71 | 1.95–2.10 | 2.05–2.21 |
| Step, pipelined (s) | 0.50 | 0.93 | 1.15 | 1.20 |

Sequential, the 1,016 steps take 0.45–0.49 h and cost $7.26–7.83 at 4 × $3.99, about the
B200's wall time at twice its price. Pipelined, training sets the pace: 0.27 h and $4.26, the
comparison's figure. Per step, B uses 2.0–4.8 GPU-seconds pipelined and 3.5–8.8 sequential,
against 1.3–2.5 at today's floor. These are floors: on one H100 the code ran 56–124 s per step
(2026-10-19, F22's table), and FS35 measures the four-GPU step.

### Effect on the paper's numbers

- **EQ14.** The GPU stays an H100, so seconds per GPU stay comparable with PoTS. But the
  verifier spans three GPUs: its wall ratio is 0.69–0.87 at the floor, and its GPU-second ratio
  2.1–2.6. The paper must report GPU-seconds beside wall time, which EQ1b doesn't define yet.
  Pipelining also breaks EQ14's "one at a time".
- **The hand-off is card to card.** It is closer to a separate verifier than A's shared memory,
  but no network moves 300 GB/s, so the paper states the in-machine setting, as for today's.

### Risks

- The split replay's bit identity: the kernels are the same on each card (same SKU and build,
  F5a), but nothing in the repository has run on a second GPU. A mismatch shows at the first
  layer boundary, through `_enter_unit`'s exact comparison.
- Host-bound code doesn't scale with GPUs in one process.
- NVLink must be present on the rented pod, with the slowest pair's peer copy (PyTorch
  `copy_`) at 300 GB/s or more; 150–300 returns to the user and below 150 fails
  (`DECISIONS_FULL_SCALE.md` F22, HW2, 2026-10-20).
- Runpod listed two-GPU H100 SXM stock as low on 2026-10-14.
- A pod can report peer access and still copy wrong bytes peer to peer: two Runpod A40 pairs
  did on 2026-10-21. Every session on two or more GPUs checks peer copies byte for byte first
  and stops on a failure (`DECISIONS_FULL_SCALE.md` F22, HW2).

### Testing

1. Fast suite on the Mac with every device set to `cpu`.
2. A two-GPU NVLink pod (two A100 SXM at about $3.6 per hour, or two H100 SXM at about $8):
   the split replay bit for bit against one GPU, roots equal, cheats rejected at their
   declared points. Two A40s over PCIe (about $1.2 per hour) were the first choice, but on
   Runpod their direct peer copies land wrong bytes (F22, HW2, 2026-10-21).
3. One four-H100 session ($15.96 per hour): topology, peer-copy rate, one verified step per
   model, per-GPU peaks.

## Baselines without a port

Two ways to use the same hardware with today's code set the bar each port has to beat:

- **Four independent cells on the four-H100 pod.** The evaluation has 8 cells (4 models × 2
  attacks). Each runs today's one-GPU code on its own card. The 1,016 steps cost what they cost
  on one H100 ($2.22–4.26 at the floor, $255 at FS17's speed) in a quarter of the wall time
  (0.14–0.27 h at the floor). EQ14 holds as written: one H100 per timed run, no shared card. At
  the floor the four cells share the host's memory bandwidth and hashing threads, which can slow
  each cell; unmeasured.
- **Today's code on a B200.** Training gets twice as fast, but the copy and CPU hashing still
  take 1.23 s per side for Qwen2.5-1.5B. The floor per step stays at today's H100 floor, at
  twice the price.

So A pays only through its port, and B's split design beats the four-cell baseline only on the
time of one step, not on the evaluation's total time or price. For reference, the comparison's
cheapest environment, the GH200, keeps today's code path; it is outside this plan.

## Trade-offs

| | A: one B200, transcript in HBM | B: four H100s over NVLink |
|---|---|---|
| Floor per step | 0.86–1.90 s | 0.87–2.21 s sequential; 0.50–1.20 s pipelined |
| 1,016 steps at the floor | 0.42 h, $3.35 | 0.45–0.49 h, $7.26–7.83 sequential; 0.27 h, $4.26 pipelined |
| GPU-seconds per step against today's floor | about 2× in H100-price terms (one B200 ≈ 2 H100-hours) | 2–4× |
| Copies across PCIe | digests and challenges only | digests and challenges only; products over NVLink |
| Where hashing runs | the training GPU, in series unless overlapped | GPUs 1–3, beside training |
| Verifier's GPU work | one GPU, as today | three GPUs, glue in series across them |
| Processes and roles | one process, one GPU, as today | sequential: one process, four GPUs; pipelined: the prover's process on GPU 0 and the verifier's on GPUs 1–3 (FS33), two live transcripts |
| Size of the port | groundwork plus about 1–1.5 coder-days | groundwork plus about 4–7 coder-days; 1.5–2 more to pipeline |
| Protocol and spec | unchanged with BLAKE3 | unchanged with design 1; design 2 changes the leaf list |
| Scale invariance | clean: the flag means host memory on a CPU | the split runs on CPU only with every device the same |
| Bit identity (F5a) | holds on one GPU; new GPU, new bands | same SKU on every card; split untested |
| PoTS time comparison | ratios only; ratio rises to 1.50–1.72 at the floor | seconds per GPU comparable; GPU-seconds needed |
| Development hardware | today's H100 and A40 pods for the smaller models | a two-GPU pod at least |
| Effect on today's host-bound code | none | none in one process; up to 3× with one process per GPU |
| Availability (Runpod) | low | low for multi-GPU H100 SXM |

## Decisions each alternative reopens

| Decision | A | B |
|---|---|---|
| EQ14, one H100 per timed run, one role at a time | GPU type changes; seconds no longer compare | one GPU per run no longer holds; pipelined breaks "one at a time" (revised before FS14) |
| EQ1b, the cost grid | P2 measures the device snapshot | a GPU-seconds column |
| EQ13.5, the environment record | names the B200 | names four GPUs and the topology |
| F1c, transcript in host RAM | reversed for this GPU; its rejection of "hold on the GPU" rested on 80 GB | replaced by peer copies |
| F15b's device (R1, the CPU) | hashing moves to the GPU | hashing moves to GPUs 1–3 |
| F21, pinned pool and band device | the pool question disappears; the band half stays (G6) | the same |
| F5a, bit identity across roles | holds | must be shown for the split |
| S3, one process, one transcript alive | holds | holds sequential; pipelined keeps two transcripts in two processes (FS33; revised before FS14) |
| S9a, the hash function | unchanged with BLAKE3; revised with cuPQC | the same |

## Effort and sessions

Coder-day figures are rough guesses for the project's coder agents, review included.

| Work | Coder-days | Sessions |
|---|---|---|
| G1 and G2 | 2–3 | one H100 hour for the hash benchmark, about $4 |
| G3, G4, G5, G7 | 2.5–3.5, through the O2 session where they touch check 5 | measured in the next H100 session |
| G6 | 0.25 | none |
| A's port | 1–1.5 | one B200 session, 1–2 h, about $8–16 |
| B's port, sequential | 4–7 | a two-GPU development pod, then one four-H100 session, 1–2 h, about $16–32 |
| B's pipelining | 1.5–2 more | inside the same sessions |

## Measurements before choosing

*Done (history): FS27's profile (the split, `g`), FS29's kernel rate (380–600 GB/s on one H100)
and FS31's peer copies and cards (392 GB/s slowest pair, the same bits on four cards,
2026-10-21). The choice is F22.*

1. FS27's profile: how far each section of today's code sits from the H100 floor.
2. G1's rate on an H100, with the run's leaf sizes. Every A and B figure rests on it.
3. How much GPU hashing hides under training on a second stream (the hashing plan's E3).
4. For A: one plain step per model on a B200 (the speed factor), and the peaks with the
   transcript in HBM.
5. For B: peer-copy rate (PyTorch `copy_`, and `nvbandwidth` when present) and topology on
   the rented pod, and the glue share `g`
   of the verifier, which FS27's trace gives on one H100.

## Recommendation and order

*Superseded by F22 and F23 (2026-10-20); history only.* As built: FM9 (FS28–FS30, the kernel)
and FM10 (FS31–FS35, the four-GPU port) in parallel; A, one B200, only if B fails HW1–HW3 (F22,
decision 4). G3 and G4 went through the check-5 speed work (O10–O12) and FM11
(`MEASURE_KERNEL_PLAN_DRAFT.md`), G5 through FS32e (check 6 on each weight's GPU), G7 through
FS34 (the owners hash on arrival); G6 is F21's open half.

1. Run FS27 as planned. Build neither port before it shows where the time goes.
2. Build G3, G4, G5 and G7 on the H100. They cut today's copies and hashing without a new
   environment, and both ports assume them.
3. Build G1 and G2 with the hash benchmark. They are the shared core of both ports, and the
   benchmark settles BLAKE3 against cuPQC.
4. If a faster environment is still wanted, take A on a B200. Take B only if the time of one
   step on H100s matters more than GPU-hours, and start with design 1 in one process,
   sequential.
