# Sampling — Draft for Discussion

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: a proposal under discussion, not a decision.** Written on 2026-10-11 by the sampling
session. The user set the goal of verification at a few percent of training compute, and
rejected recompute as the way there, because recompute stays the same order as training. This
draft works out sampling, the direction that can reach the goal, and its main weakness, the
one-step weight edit.

Nothing here changes a decision, the spec, the plan, a task file or the code. The full-scale
work in progress keeps the every-step protocol, and serves as the reference: its measured
per-step cost is this draft's `C_step` (Section 4). If the user adopts sampling, the changes
listed in Sections 9–11 are made then, and some full-scale choices get revised or restored.

All throughputs below are **assumed**, from `HASHING_PLAN_DRAFT.md` Section 1 and F17's floor
model: training at 400 TFLOP/s, GPU hashing at 250 GB/s, GPU↔host copy at 50 GB/s, a read from
GPU memory (HBM) at 2.5 TB/s, CPU hashing at 15–30 GB/s. Disk reads at 5–25 GB/s (one NVMe
drive to a small array) are this draft's own assumption. Scripts: `scratchpad/bytes_vs_flops.py`,
`freivalds_floor.py`, `sampling_numbers.py`, `audit_cost.py` of the sampling session (not in the
repo).

*Since F22 and F23 (2026-10-20):* full scale runs on four H100 SXM over NVLink. The products cross
by peer copy, and F23's kernel hashes them on the owner GPUs at 380–600 GB/s (FS29), so the copy
and hash rates above are the one-H100 floor model, and `C_step` is read from FS35's four-H100
step, in GPU-seconds beside wall time (EQ14's revision). The "now" column of Section 9's rows
on F1c, S3, EQ14, F5a and the hashing plan's choices predates F22: F1c is replaced by peer
copies at full scale, S3 keeps two transcripts in two processes when pipelined, EQ14 needs
GPU-seconds, F5a must hold across cards, and the hashing choices are F15b and F23.

**Symbols.** Several obvious names are taken, so this draft uses:

| symbol | meaning | why not the obvious name |
|---|---|---|
| `φ` | fraction of steps audited, `T_a / T` | `f` is the sizing appendix's detection target and `f_achieved` |
| `T_a` | number of audited steps | `n_s` is sequences per batch in the code (`c.n_s`) and the reference block |
| `n_tok` | tokens per training step | `N` is the Freivalds bit budget |
| `c_t` | hash of the weights after step `t` (`c_0` hashes `W_0`) | — |
| `R` | the run commitment: one root over a header and `c_0 … c_T` | — |
| `S` | the set of audited steps | — |
| `K_s` | steps between stored checkpoints | — |
| `d` | number of deviating steps in a run | — |
| SD1–SD11 | this draft's decisions | `EVALUATION_SPEC.md` §8 already uses D1–D6 |

1. Why a few percent rules out checking every step
2. The design: commit the weight trajectory, audit sampled steps
3. The guarantee
4. The cost
5. Re-executing an audited step instead of checking it
6. The one-step weight edit
7. Mitigations for the one-step edit, each with its price
8. Decisions sampling raises
9. Implications on decisions and design docs
10. Implications on code
11. What the full-scale run keeps

---

## 1. Why a few percent rules out checking every step

**A verifier that checks every step must receive every byte of every step's transcript, and the
prover must hash every byte to commit to it. At PoTS-class sizes both cost far more than a few
percent of training.** A few-percent protocol must therefore leave most of the computation
unread: that is sampling.

The derivation, one step per line:

1. Per step, training does `F` FLOPs and the transcript holds `B` bytes. For the models of
   F17's floor model, `F/B` is 1,445 (Qwen2.5-0.5B), 2,515 (Llama-3.2-1B), 3,088 (Falcon3-1B),
   2,274 (Qwen2.5-1.5B) and 4,043 (Llama-3-8B, added for scale) FLOP per byte.
2. Moving `B` bytes at throughput `X` takes `B/X`; training takes `F / 400 TFLOP/s`.
3. The move stays within 5% of training only if `F/B ≥ 400·10¹² / (0.05·X)`.
4. That asks for 160,000 FLOP/B over PCIe (50 GB/s), 32,000 for GPU hashing (250 GB/s) and
   3,200 for one read in GPU memory (2.5 TB/s).
5. The models sit at 1,445–4,043. So a copy costs 2.6–5.5× training, hashing 0.52–1.11×, and a
   single in-memory read 4–11%.

The in-memory read is the prover's own GPU, which a third-party verifier can't use, so even
that floor isn't available to it.

Two consequences shape the design:

- **The verifier must leave steps unread.** That is sampling.
- **The prover can't commit every byte either.** Hashing the full transcript costs it 0.52–1.11×
  training. Sampling products, layers or sequences inside every step, with every step's
  transcript committed, would still cost the prover about one more training run. So the
  sampled unit must be a whole step whose transcript is produced only after the step is
  sampled. During training the prover commits only what is small: the weights.

**Weights are small next to training, and the ratio doesn't depend on the model.** Per step the
prover produces `4p` bytes of fp32 master weights against `6p·n_tok` FLOPs (`p` parameters). The
weight rate is `4 · 400·10¹² / (6·n_tok)` bytes per second:

| tokens per step `n_tok` | weight rate | GPU-hash time, share of training | PCIe link used |
|---|---|---|---|
| 16,384 (our full scale, PoTS's batch) | 16.3 GB/s | 6.5% | 33% |
| 131,072 | 2.0 GB/s | 0.8% | 4.1% |
| 1,048,576 | 0.25 GB/s | 0.1% | 0.5% |
| 4,194,304 (pretraining-scale batch) | 0.06 GB/s | 0.03% | 0.1% |

## 2. The design: commit the weight trajectory, audit sampled steps

**In one line: the prover trains normally and hashes the weights after every step; after
training, randomness the prover can't control picks a few steps; the prover re-runs those
steps with today's per-step protocol, and the verifier checks them as today.**

### 2.1 During training (prover)

1. Train with plain SGD, as the plain baseline does: no capture, no transcript. Plain SGD keeps
   no optimizer state (S8a), so the weights are the whole training state and a checkpoint is
   the weights alone. Momentum or Adam would have to put their state into `c_t` too.
2. After each update, hash the fp32 master weights `W_{t+1}` into `c_{t+1}`: the Merkle root
   over the 272 weight-leaf hashes (P9b), with the transcript's own leaf encoding and hash
   function. A regenerated transcript then binds to `c_t` through leaf hashes the verifier
   already computes for check 2. `c_0` is the same root over `W_0`.
3. Store a full checkpoint every `K_s` steps, so a sampled step can be replayed.
4. Keep this off the GPU's critical path. The copy engine moves `W_{t+1}` to host memory while
   the next step computes: a third of the PCIe link at `n_tok = 16,384`, and it must finish
   before the next update overwrites the masters. The CPU hashes it. CPU hashing at the assumed
   15–30 GB/s is borderline at the 16.3 GB/s needed. Hashed on the GPU instead, it costs 6.5%
   of training. (SD5)

### 2.2 After training: the run commitment

`R = MerkleRoot(header, c_0, c_1, …, c_T)`, where the header holds the dataset anchor `h_D`,
the base-model anchor, the config `C`, `T` and the schedule seed. The header needs the same rigid
encoding as the leaves (S9c: no JSON). The verifier supplies the leaf count itself, as it does
for step trees today.

The prover hands `R` to the verifier, or registers it publicly, **before** any randomness for
the step choice exists. A run has one `R`: a prover who could register several roots for one
run and keep the luckiest multiplies its escape probability by their number.

### 2.3 The step choice

`S ⊂ {0, …, T−1}` holds `T_a = ⌈φ·T⌉` steps, drawn uniformly **without replacement** by
`Expand(ρ, R)`, under a hash domain of its own, prefix-free against today's tags. With this rule
a deviation present in every step is caught whenever `T_a ≥ 1`. (Drawing each step
independently with probability `φ` would instead catch it with probability `1 − (1 − φ)^T`; this
draft doesn't use that rule.)

The seed `ρ` is one of:

- **(a) fresh:** a public randomness beacon's value, or the verifier's coin, produced after `R`
  was registered;
- **(b) Fiat-Shamir:** `ρ = PRF(R)`, computed by anyone from `R` alone. Section 3.3 shows why
  this is weak here.

### 2.4 Auditing a sampled step: today's per-step protocol

For each `t ∈ S`:

1. The prover loads the nearest stored checkpoint at or before `t`, replays plain steps up to
   `W_t`, and confirms `hash(W_t) = c_t`. The batch schedule `π(t)` is a pure function of the
   step and the seed (S5d, C5), so any step's batch is available directly.
2. It regenerates step `t` with capture, exactly as the verified run does today: transcript
   `L_t = (b_t, W_t, P_1 … P_M, W_{t+1})`, root `h_t`, Freivalds challenges from `h_t`.
3. The verifier runs checks 1–6 on `L_t` as today.
4. Check 7 (chaining consecutive steps) becomes a binding: the root over `L_t`'s `W_t` leaves
   must equal `c_t`, and over its `W_{t+1}` leaves `c_{t+1}`, both proven to sit in `R` by
   Merkle paths.
5. Check 0 (base anchor) and check 8 (final anchor) move to the run level: `c_0` must equal the
   base model's hash and `c_T` the published final model's hash. They are checked once per
   run, whichever steps are sampled.
6. Accept the run iff `R` opens correctly, `S` was derived correctly, both anchors hold, and
   every audited step accepts.

### 2.5 Regeneration must reproduce the run bit for bit (SD4)

**Two things must match exactly: the replayed `W_t` must hash to `c_t`, and the regenerated
`W_{t+1}` must hash to `c_{t+1}`.** Hash equality has no tolerance.

- **The replay to `W_t`.** Check 6's band can't help here: the comparison is a hash. So the
  replay must be bitwise, or `K_s = 1`, with every `W_t` stored. At `K_s = 1` the store is
  0.6–2.0 TB for a 300-step run and 2.0–6.7 TB for a 1,000-step run (the four PoTS-class
  models).
- **The regenerated step.** The regenerated gradient must equal the trained one bit for bit,
  because check 6's band covers only the update's own rounding (F6b: `ε_W = 2⁻²⁴`). A gradient
  summed in a different order enters check 6's residual as `η·ΔG`.

Two ways to meet the second condition:

- **Deterministic kernels** in the original run and in the regeneration: S4c's settings
  (deterministic algorithms, the cuBLAS workspace), F5a's reduced-precision reductions off, the
  same thread count and autocast, on the same GPU model and software build, pinned until the
  audits end, possibly days later. Two consequences:
  - **Training runs eager attention.** Regeneration needs capture, capture needs attention's
    separate matmuls, and the trained gradient must come from the same kernels. So every
    training step gives up fused attention (F16). Test scale measured eager at about 7% over
    fused on CPU; the H100 measurement E6 gives the full-scale figure.
  - **The bit-identity evidence must widen.** Today it is "capture doesn't change training"
    (A2, on CPU) and "the plain run's final weights equal the verified run's" (F5c). Sampling
    needs it per step, from a mid-run checkpoint, in a later session.
  *Price:* a slowdown of every training step, not yet measured.
- **Widen check 6's band** to cover an honestly recomputed gradient, with `K_s = 1`. *Price:* a
  band of bf16 size, which F6b rejected as room to move each weight by about 300 updates' worth
  per step unseen, plus the terabytes of `K_s = 1`.

The draft recommends determinism, with its cost measured. The widened band is not a real
alternative.

### 2.6 What changes, in one place

- **Kept:** every per-step piece. Capture, leaves, the Merkle tree, Freivalds with `k` vectors,
  checks 1–6, the bands and their calibration.
- **Dropped from the main training run:** capture, the transcript, hashing the products, the
  GPU-to-host transcript copy.
- **Added:** per-step weight hashing, checkpoint storage, the run commitment, the step choice,
  replay, the run-level anchors, determinism and eager attention in training.
- **Replaced:** check 7's chaining, by the binding into `R`.

## 3. The guarantee

### 3.1 Definitions

A step `t` **deviates** if no transcript links the committed `W_t` to the committed `W_{t+1}`
on the scheduled batch `b_t` within the protocol's tolerances. A poisoned batch, a hidden extra
step, a direct weight edit and a flipped matmul each make their step deviate.

### 3.2 With fresh randomness

1. If a deviating step is sampled, today's per-step guarantee applies to it. Over the whole
   audit it is accepted with probability at most `2⁻λ`, with the Freivalds budget sized for
   `T_a` audited steps instead of `T` (SD8).
2. `S` misses all `d` deviating steps with probability `C(T−d, T_a) / C(T, T_a) ≤ (1 − φ)^d`.
3. So a run with `d` deviating steps is accepted with probability at most `(1 − φ)^d + 2⁻λ`.

**The meaning of `λ` changes.** Today `2⁻²⁵` bounds acceptance of any dishonest run. Under
sampling it bounds acceptance of a deviating step *given that it is audited*. The run-level
bound is `(1 − φ)^d`.

Escape probability `(1 − φ)^d`:

| `φ` \ `d` | 1 | 10 | 64 | 100 | 459 | 1,000 |
|---|---|---|---|---|---|---|
| 0.5% | 0.995 | 0.951 | 0.726 | 0.606 | 0.100 | 0.0067 |
| 1% | 0.990 | 0.904 | 0.526 | 0.366 | 0.0099 | 4.3·10⁻⁵ |
| 2% | 0.980 | 0.817 | 0.274 | 0.133 | 9.4·10⁻⁵ | 1.7·10⁻⁹ |
| 5% | 0.950 | 0.599 | 0.038 | 0.0059 | 6·10⁻¹¹ | 5·10⁻²³ |
| 10% | 0.900 | 0.349 | 0.0012 | 2.7·10⁻⁵ | 10⁻²¹ | 2·10⁻⁴⁶ |

Today's claim is `2⁻²⁵`. Sampling reaches it only for deviations spanning
`d ≥ 25·ln 2 / (−ln(1 − φ))` steps: 1,725 at `φ = 1%`, 338 at 5%, 165 at 10%.

### 3.3 With Fiat-Shamir instead

**Under Fiat-Shamir the prover can re-draw `S` almost for free, so sampling catches only
attacks that touch nearly every step.**

1. To re-draw, the prover changes `R`. The cheapest change touches no weights: overwrite the
   `c_t` of a step it expects to stay unsampled. That costs one leaf hash plus the path to `R`,
   microseconds. S9b's premise, that every re-roll costs a real recompute, doesn't hold here.
2. Each re-draw misses all `d` deviating steps with probability `(1 − φ)^d`.
3. With the protocol's grinding budget `G = 2⁵²` (the single-lab tier), the bound becomes
   `G · (1 − φ)^d`.
4. To reach `2⁻²⁵`: `d ≥ 77·ln 2 / (−ln(1 − φ))`, which is 5,311 steps at `φ = 1%`, 1,041 at
   5% and 507 at 10%.

*Example.* `T = 1,000`, `T_a = 10`. A deviation in 900 of the 1,000 steps escapes one draw with
probability `C(100, 10) / C(1000, 10) ≈ 2⁻³³`, so about `2³³` re-draws find a passing `S`, well
inside `2⁵²`. Only a deviation in about 99% of the steps is out of reach. For a one-step cheat
at `T = 10`, `T_a = 1`, the prover needs about `1/(1 − φ) = 1.1` tries.

**This reverses the arithmetic behind "Interactive challenges — rejected"**
(`DECISIONS_ALGORITHM.md`). That decision priced fresh randomness at about 2× in `k`, because
grinding cost only the `log₂G` term. Under sampling, grinding costs nearly the whole guarantee.
Its four reasons read differently here:

1. *Portable certificate.* It survives with a public beacon: anyone can recompute `S` from
   `R`, `ρ` and the record that `R` was registered before `ρ` appeared. What it adds is trust
   in the beacon and in the registration time.
2. *Live verifier in the training loop.* There is none: one round per run, after training, and
   the prover regenerates on demand. Stream-and-discard verification of each audited step is
   untouched.
3. *Trusted fresh randomness.* The assumption is real, and becomes part of the claim.
4. *Levers in hand (`k`, the `G` tier).* None of them recovers a sampled guarantee.

### 3.4 Fresh randomness for the Freivalds challenges as well

If SD2 takes fresh randomness for `S`, a second round can serve the Freivalds challenges: the
prover commits `h_t` for every `t ∈ S`, then a second fresh value gives the vectors. The prover
can no longer grind `h_t`, so `log₂G = 52` leaves the bit budget:

1. Full scale, `T_a = 1`: `N = λ + log₂T_a + log₂M = 25 + 0 + 18.59 = 43.59`.
2. `k = ⌈43.59 / 4.79⌉ = 10`, against 20 under Fiat-Shamir with the same `T_a`.
3. Test scale: `N = 25 + 0 + 12.80 = 37.80`, `k = ⌈37.80 / 11.24⌉ = 4`, against 8.

This saves far more than SD8's `log₂T → log₂T_a` (3.3 bits). *Price:* a second round per run,
and the argument of "Interactive challenges — rejected" fully reversed. Check 5's arithmetic
halves, but bytes dominate `C_step` (Section 4), so `C_step` falls by less.

### 3.5 Against the evaluation's attacks

| attack | `d` | caught today | sampled, fresh randomness | sampled, Fiat-Shamir |
|---|---|---|---|---|
| poisoned batch, trained on truthfully (A1, check 4) | 1 | certain | `φ` | ≈ never: re-draw until `t* ∉ S` |
| poisoned batch, clean batch reported (A2, check 5) | 1 | `q = 1 − p₁ᵏ` | `φ·q` | ≈ never |
| poisoned end weights spliced in (A3, check 6) | 1 | certain above the band | `φ` | ≈ never |
| hidden extra steps, any number (EQ17) | 1 | certain (check 7) | `φ`, by check 6 or the binding | ≈ never |
| one flipped matmul (the planted-error sweep, step 4) | 1 | `1 − p₁ᵏ` (floor ≈ 0.62) | `φ·(1 − p₁ᵏ)` | ≈ never |
| direct weight edit at one step | 1 | certain | `φ` | ≈ never |
| spread poisoning, 5% of every step (F6a's spread form) | `T` | certain | certain when `T_a ≥ 1` | certain |
| poisoning confined to the last `J` steps, with M4 | ≥ 1 | certain | certain | certain |

Two side effects:

- **Hidden steps collapse into one transition.** Any number of hidden steps between two committed
  checkpoints is one deviating step. PoTS's detection grows with the number of hidden steps;
  ours no longer does.
- **Calibration may catch the planned cheats.** Under bitwise determinism the verifier's own
  calibration (P10a: three honest steps from `W_0`) reproduces `W_1 … W_3`, and comparing them
  with `c_1 … c_3` catches any cheat there with certainty. Today's cheats sit at steps 1 and 2,
  so they would test calibration, not sampling.

## 4. The cost

**Overhead over plain fused training ≈ `c_traj + c_det + φ·(C_step + ℓ + K_s/2) + F_run/T`**,
in training-step units:

- `c_traj` is the weight hashing of 2.1. It costs about no GPU time if copied and hashed off the
  GPU (which needs 16.3 GB/s of copy and hashing at `n_tok = 16,384`), and 6.5% if hashed on the
  GPU.
- `c_det` is the slowdown from deterministic kernels and eager attention on every training step
  (2.5). Unmeasured at full scale.
- `C_step` is one full audited step, both sides. On the prover: regenerate (1), hash the
  transcript, copy it out. On the verifier: copy it in, hash it, run Freivalds' arithmetic.
- `ℓ` is loading the checkpoint: 0.33 from pinned host RAM, 0.65–3.26 from disk at the assumed
  25–5 GB/s. It doesn't depend on the model, because checkpoint bytes and step FLOPs both scale
  with `p`.
- `K_s/2` is the average replay from the last stored checkpoint. Storage is `T/K_s` checkpoints
  of `4p` bytes.
- `F_run` is fixed per run: calibration (three captured steps, about `3·C_step`, unless one band
  file is reused per configuration) and any always-audited steps (`J·C_step`).

Floor model of `C_step` (assumed throughputs; the full-scale run in progress measures the real
one):

| model | params | checkpoint (fp32) | transcript per step | prover | verifier | both |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.49 B | 2.0 GB | 33.6 GB | 7.6 | 6.7 | 14.3 |
| Llama-3.2-1B | 1.24 B | 4.9 GB | 48.3 GB | 4.8 | 3.8 | 8.7 |
| Falcon3-1B | 1.67 B | 6.7 GB | 53.1 GB | 4.1 | 3.1 | 7.2 |
| Qwen2.5-1.5B | 1.54 B | 6.2 GB | 66.7 GB | 5.2 | 4.2 | 9.5 |
| Llama-3-8B | 8.03 B | 32.1 GB | 195.3 GB | 3.4 | 2.4 | 5.8 |

Test scale measured `C_step ≈ 3.7` plain steps (C1), in one CPU process with no copies; the
full-scale floor adds the copies.

**What a 5% budget buys**, from `φ = 0.05 / C_step` (ignoring the other terms):

1. `C_step` spans 5.8–14.3, so `φ` spans 0.35%–0.87%.
2. One audited step needs a run of `T ≥ 1/φ`: 116–287 steps.
3. Ten audited steps need 1,152–2,866 steps.
4. Today's full-scale run has `T = 10`, so `T_a ≥ 1` and `φ ≥ 10%`. One audited step of ten costs
   `C_step / 10`, which is 58%–143% of training. **Sampling can't reach a few percent on a
   10-step run.** It needs runs of hundreds of steps. A fixed `F_run` of `3·C_step` for
   calibration needs roughly three times as many.

`k` at `T = 10` with `T_a = 1`: full scale `N` goes from 98.91 to 95.59, so `k = ⌈95.59/4.79⌉`
goes from 21 to 20 on a 0.21-bit margin. Test scale `N` goes from 93.12 to 89.80, so
`k = ⌈89.80/11.24⌉` goes from 9 to 8 on a 0.12-bit margin. (C1's measured `b₀ = 11.24`; the sizing
appendix still shows 11.03.)

## 5. Re-executing an audited step instead of checking it

**Under sampling, re-running an audited step and comparing weight hashes costs about 1.5
training steps against Freivalds' 6–15, and is exact. The determinism that sampling already
demands is what makes it possible.** This finding came out of the inventory. It bears directly on
the user's preference for Freivalds, so it gets its own section.

### 5.1 The method

1. The verifier loads the checkpoint, replays to `W_t`, and checks `hash(W_t) = c_t`.
2. It runs step `t` itself, under the same deterministic settings as training.
3. It accepts iff `hash(W_{t+1}) = c_{t+1}`.

No transcript, no capture, no Freivalds, no `k`, no bands and no calibration. Training also
keeps fused attention, as long as the fused kernel is deterministic (not yet checked), because
nothing needs attention's separate matmuls.

The user rejected recompute because it costs as much as training. Under sampling, both methods
run on `φ` of the steps, so the comparison is per audited step.

### 5.2 Cost per audited step

Re-execution = `ℓ` + one training step + two weight hashes on the GPU (0.065 each). With
`K_s = 1`:

| checkpoint source | re-execution | Freivalds, `C_step + ℓ` | ratio |
|---|---|---|---|
| pinned host RAM (`ℓ = 0.33`) | 1.46 | 6.1–14.6 | 4.2–10× |
| disk, assumed 25 GB/s (`ℓ = 0.65`) | 1.78 | 6.5–15.0 | 3.6–8.4× |
| disk, assumed 5 GB/s (`ℓ = 3.26`) | 4.39 | 9.1–17.6 | 2.1–4.0× |

A 5% budget then buys `φ = 3.4%` (host RAM) or 1.1% (slow disk), against Freivalds' 0.34–0.82%
or 0.28–0.55%. A one-step edit is caught with probability `φ` under either method, so
re-execution catches it 2–10× as often at the same cost. With `K_s > 1`, each replayed step is
re-executed and compared with its own `c`, so replay adds coverage rather than pure overhead.

### 5.3 What re-execution gives up

1. **Arithmetic.** The verifier spends a full step of FLOPs per audited step; Freivalds spends
   `O(n²)` per product. Freivalds wins on FLOPs and loses on bytes (Section 1). It wins per
   audited step only if its bytes fall below about one training step, a 5–14× cut of F17's
   floor.
2. **Hardware independence.** Re-execution needs the prover's GPU type and software build. The
   current Freivalds verifier already needs the same: F5a's condition for bit-identical glue. A
   Freivalds verifier on other hardware would need glue bands calibrated across hardware, which
   isn't today's design.
3. **Diagnostics.** A hash mismatch says the step differs, not which product. Comparing with a
   regenerated transcript can localize it afterwards.
4. **Novelty.** Re-executing sampled steps from stored checkpoints is the Proof-of-Learning
   design (Jia et al., IEEE S&P 2021). Its published spoofing attacks (Zhang et al., 2022; Fang
   et al., 2023) exploit its distance tolerance and its choice of the largest updates. Bitwise
   determinism and random step choice close both, and leave the one-step edit at probability
   `φ`. Work on making GPU training bitwise reproducible for verification also exists (for
   example Srivastava et al., 2024, "Optimistic Verifiable Training by Controlling Hardware
   Nondeterminism"). *(Citations from memory; check before use.)*

So Freivalds has a role under sampling in two cases: when transcript bytes per audited step fall
well below today's floor (F17's work), or when the verifier can't reproduce the prover's
hardware and glue bands work across hardware. That is SD9.

## 6. The one-step weight edit

### 6.1 The statement

**No sampling rule catches a deviation confined to one step with probability above `φ`, the
fraction of steps audited.**

Why, in steps:

1. A sampled audit reads only the sampled steps.
2. A deviation confined to step `t*` leaves every other step honest. The trajectory stays
   consistent: `W_{t*+1}` is whatever the edit made it, and later steps train from it
   honestly.
3. So the evidence outside `t*` is what an honest run would produce from the same weights.
   Only an audit of `t*` can tell, and `t*` is audited with probability `φ`.

Re-execution (Section 5) doesn't change this. It raises the `φ` a budget buys, not the bound.

### 6.2 Finer units don't change it for a direct edit

Sampling sequences, layers or products inside steps reads a fraction `φ` of smaller units, but
a direct weight edit fits in one unit: one sequence's gradient contribution, or one product.
It is still caught with probability `φ`.

Finer units **do** help against a deviation that is spread by nature. A poisoned batch touches
every poisoned record, so record-level sampling at fraction `φ` catches F6a's 64 poisoned
records with probability `1 − (1 − φ)⁶⁴`: 47% at `φ = 1%`, 96% at 5%. The price: record-level
units need every step's per-record transcript committed during training, which costs the
prover the 0.52–1.11× hashing of Section 1. (Mitigation M3.)

### 6.3 Why a cheap every-step check doesn't close it

- **Bounding the update size.** Checking `‖W_{t+1} − W_t‖` on every step needs the verifier to
  read both weights each step: 33% of training over PCIe at `n_tok = 16,384`. It also separates
  nothing: F6a's poisoned dose is an update of honest size by design.
- **Checking the sum of updates.** `W_T − W_0 = Σ_t (W_{t+1} − W_t)` holds for any trajectory,
  edited or not, so a check on the telescoping sum sees nothing.
- **Checking each step's gradient** is the full protocol.

What a weight edit breaks is "this update is `η` times this batch's gradient", and the gradient
is the expensive thing to know.

### 6.4 The only known way around it

Commit to an error-correcting encoding of the data. A one-unit deviation then spreads into many
encoded positions, and a few random openings catch it. Succinct proof systems get sublinear
verification this way (the Ligero and Brakedown family, built on PCP and IOP constructions).
They need exact arithmetic over a finite field. Floating-point training rounds, so the
encoding's distance guarantee doesn't survive. Integer-quantized training would fit them; for
this project it is future work.

### 6.5 The attacker's incentives flip

Today Freivalds rewards spreading a deviation thin, below `f_achieved` in every product.
Sampling rewards concentrating it into as few steps as possible. An attacker takes whichever
works better, so the evaluation needs both axes: the size of the deviation per step and the
number of steps `d`. How many poisoned steps a backdoor needs to plant (F6c's postponed
attack-success study) now sets run-level detection, `1 − (1 − φ)^d`.

## 7. Mitigations for the one-step edit, each with its price

- **M1 — Fresh randomness for the step choice (SD2).** Needed for `φ` to mean anything (3.3).
  *Price:* trust in a beacon or the verifier's coin, and a one-root-per-run registration. It
  reverses part of "Interactive challenges — rejected".
- **M2 — State the claim as a per-deviation probability (SD1).** A cheat caught with
  probability `φ` that costs a penalty `P` when caught is irrational when `φ·P` exceeds its gain.
  *Price:* the claim becomes economic, not a `2⁻²⁵` soundness bound.
- **M3 — Finer units against data poisoning (6.2).** *Price:* per-record transcript commitments
  every step, about one training run of hashing for the prover at the assumed throughputs. No
  help against a direct edit.
- **M4 — Always audit the last `J` steps (SD6).** An edit in the final `J` steps can't escape;
  an earlier one must survive `J` clean steps after it. *Price:* `J · C_step` per run. Its
  strength depends on how well a backdoor persists through clean training, which can be
  measured and is known to be high for some attacks. It doesn't cover today's cheats at steps 1
  and 2. Always auditing the calibration steps instead tells the cheater to avoid them.
- **M5 — Larger batches.** Weight costs scale as `1/n_tok`. At `n_tok ≈ 1M`, hashing `W` every
  step costs 0.1% on the GPU and reading it 0.5% over PCIe, so per-step weight checks become
  affordable. *Price:* not our setup (`n_tok = 16,384` matches PoTS), and it doesn't change 6.1.
- **M6 — Encoded commitments (6.4).** Research.
- **M7 — Re-execution as the audit (Section 5).** Raises `φ` 2–10× at the same budget.
  *Price:* Section 5.3.

## 8. Decisions sampling raises

Presented to the user one at a time, coupled ones together, in this order:

- **SD1 + SD2 (coupled) — the guarantee class and the source of the step choice.** "Every step,
  `2⁻²⁵`" becomes "each deviating step caught with probability `φ`, and `2⁻²⁵` once audited".
  Without fresh randomness (SD2), SD1 holds only for attacks touching nearly every step (3.3).
  Fresh randomness also opens 3.4's second round (`k` 21 → 10).
- **SD9 — the audit method: Freivalds on a regenerated transcript, or bitwise re-execution
  (Section 5).** It decides whether SD4's eager attention, SD11's calibration and SD8's sizing
  matter at all.
- **SD4 — determinism versus a widened update band (2.5).** In practice, determinism, with its
  cost measured. It puts eager attention (Freivalds path) and deterministic kernels on every
  training step.
- **SD5 — the trajectory commitment's mechanics:** what is hashed (fp32 masters, as the root over
  the 272 weight leaves), the hash function (shared with the transcript, so one choice for both
  workloads), where (CPU overlapped, or GPU), and the checkpoint interval `K_s`.
- **SD3 — the unit: one step or a segment of `K` steps.** Segments bundle up to `K` deviating
  steps into one unit, so a `K`-step attack becomes as hard to catch as a one-step attack. They
  save only the per-step hash, which 2.1 already moves off the GPU path. Draft recommendation:
  one step.
- **SD6 — always-audited steps (M4).**
- **SD10 — run length and data.** A few percent needs runs of hundreds of steps (Section 4).
  F8a rejected long runs because they mean hundreds of passes over F8d's 369 and 233 records. A
  bigger corpus leaves PoTS's data.
- **SD11 — the calibration source (Freivalds path).** Plain training captures nothing, so the
  bands lose their input. Regenerate the calibration steps each run (about `3·C_step`, a fixed
  cost), or reuse one band file per configuration (C5 already reuses across seeds), with
  `T_run` taken out of the band context. Long runs also bring back band drift over many steps,
  which F9 set aside for 10-step runs.
- **SD8 — sizing:** `log₂T` becomes `log₂T_a` in the Freivalds budget.
- **SD7 — the evaluation's framing:** how the few-percent claim is shown (a long run, or a cost
  model from per-step rows), how detection is reported (per audited step, and times `φ`), and
  how the comparison with PoTS reads. Per verified step, ours (`C_step` 6–14) costs about
  10–40× PoTS's verify-to-train ratio of 0.33–0.77. Ours is cheaper per run only through
  sampling, which PoTS could adopt too. What we can defend is strength per audited step:
  `2⁻²⁵` (or exact, under re-execution) against PoTS's distance ratio of about 1.0–1.14 at 10%
  poisoning.

## 9. Implications on decisions and design docs

The tables assume the Freivalds audit of Section 2. If SD9 picks re-execution, the rows on
checks 3–6, the bands, calibration and `k` become moot rather than amended. Each row says what
the item holds now and what sampling does to it: **reversed** (the item's conclusion no longer
holds), **amended** (it holds with a change), **new** (work that doesn't exist yet), **moot**,
or **load-bearing** (unchanged, but sampling now depends on it).

### 9.1 The implications that change the picture

1. **Re-execution may beat Freivalds** per audited step, by 2–10× (Section 5).
2. **The approved guarantee changes.** "Every matmul of every step" (CLAUDE.md's project goal,
   the spec's abstract and §8.3) becomes "every matmul of every audited step". Stages 1 and 2
   (algorithm, spec) reopen.
3. **`λ` changes meaning:** `2⁻²⁵` holds per audited step; the run-level bound is `(1 − φ)^d`
   (3.2).
4. **Determinism moves into training.** Eager attention and deterministic kernels run on every
   training step (2.5). The training overhead before any verification is unmeasured and may
   itself exceed a few percent.
5. **Hash equality is exact**, so the replay must be bitwise whatever check 6's band is (2.5).
6. **Fiat-Shamir sampling is nearly worthless**, and cheaper to grind than S9b assumed (3.3).
7. **Fresh randomness opens a second round** that drops `k` from 21 to 10 (3.4).
8. **Hidden steps collapse to one transition** (3.5). P11's headline reverses, and the PoTS
   comparison of concealed steps may favour PoTS at small `φ`.
9. **Calibration loses its input**, and under determinism it may catch the planned cheats
   before sampling does (3.5, SD11).
10. **Few percent needs long runs** (Section 4, SD10), which reopens F8a, F8d and F9's drift.
11. **The attack-success study becomes load-bearing** (6.5): `d` sets detection.
12. **Weight hashing ties to the hashing plan:** `c_t` must use the transcript's leaf encoding
    and hash function, so the open function and device choice (R1) serve both workloads.

### 9.2 `DECISIONS_ALGORITHM.md`

| item | now | under sampling | decision |
|---|---|---|---|
| §3.1 Freivalds + Fiat-Shamir core | every step, both scales | **Amended.** Runs inside audited steps only, under an outer step-selection layer. | SD1, SD2 |
| §3.3 size of `k` | `N = log₂G + log₂(M·T) + λ` | **Amended.** `log₂T → log₂T_a`; `log₂G` drops with a fresh second round. | SD8, SD2 |
| §3.5 plain SGD | `W − ηG` | **Load-bearing.** No optimizer state, so `c_t` covers the whole state. | SD5 |
| §3.6 sequential per-step verification | verify, discard, next | **Amended.** `T_a` non-consecutive steps, after training; checkpoints every `K_s`. | SD3, SD5 |
| §3.7 one Merkle root per step | `h` over `(b, W_t, P, W_{t+1})` | **Amended.** Built after training; the `W` leaves must use `c_t`'s encoding. | SD5 |
| §3.8 the verifier derives `r` | — | **Load-bearing, extended.** The verifier also derives `S`; never takes it from the prover. | SD2 |
| §3.10 `τ′` vs `τ_W` | tight per-entry `τ_W` | **Amended.** Holds only under bitwise regeneration. | SD4 |
| §3.13 model binding | update, chaining, endpoints | **Chaining reversed, endpoints amended.** Paths into `R`; `c_0`, `c_T` checked at run level. | SD5 |
| §3.14 security claim | rejected except `N_checks·p₁ᵏ` | **Amended.** Adds `(1 − φ)^d` or `G·(1 − φ)^d`; new knobs `φ`, `K_s`, `J`, the randomness source. | SD1, SD2, SD6 |
| §3.15 dataset and schedule anchor | off-corpus and off-schedule data caught | **Amended.** Caught with probability `φ` per poisoned step; `h_D` also enters `R`'s header. | SD1, SD5 |
| security sizing (grind) | "+~20 for a 2²⁰-step run" | **Amended.** `log₂T_a`; Fiat-Shamir sampling needs its own grind price. | SD2, SD8 |
| pure-float sizing | `z ≈ 8` over ~2⁴¹ checks | **Amended.** False-positive union bound over `T_a` steps. | SD8 |
| "Interactive challenges — rejected" | Fiat-Shamir kept | **Reversed if fresh randomness.** Reasons 1 and 3 return in weaker form, reason 2 is moot; the paper's "why non-interactive" text changes. | SD2 |
| W-A goal (a substituted batch is caught) | certain | **Amended.** With probability `φ` per poisoned step. | SD1 |
| Q1 threat model | no online advantage | **New.** The prover's influence on the randomness and the timing of `R`'s registration. | SD2 |
| Q3 glue need not be bit-exact | closed | **Reversed.** Replay must be bit-exact to the prover's own run. | SD4 |
| Q5a `τ_W` isn't a widened absorber | — | **Reversed if a widened band** (2.5). | SD4 |
| Q8 crypto primitives | hash, PRF, serialization | **New.** Trajectory hash, `R` encoding, beacon or challenge source, the `S` domain tag. | SD2, SD5 |
| Q12 step loop and memory | prover reports every step | **Amended.** Only audited steps; checkpoint disk and replay compute added. | SD5 |
| Q13 evaluation metrics | moved | **New.** Detection against `φ`, the few-percent claim, the PoTS comparison. | SD7 |
| §9 findings (33 GB/step, 40×, `N ≈ 115`) | per step | **Amended.** Per audited step; `N ≈ 115` used `log₂T = 20`. | SD8 |

Untouched or applied unchanged per audited step: §3.2, §3.4, §3.9, §3.11, §3.12, W-B, W-C, Q0,
Q2, Q2a, Q4, Q5, Q5b, Q6, Q7, Q9–Q11, Q14, R1–R5.

### 9.3 `VERIFICATION_PROTOCOL_SPEC.md`

| item | now | under sampling | decision |
|---|---|---|---|
| abstract | "certifies the actual training transition" | **Amended.** For audited transitions; the run-level claim is probabilistic. | SD1 |
| §1 setting | accepts only if every step verifies; non-interactive | **Amended; reversed if fresh randomness.** Transcripts come after the run. | SD1, SD2 |
| §2 notation | — | **New.** `c_t`, `R`, `S`, `T_a`, `φ`; avoid the clash with `f`. | — |
| §3.1 coverage | "no matmul is exempt" | **Amended.** Within an audited step; unaudited steps go unchecked. | SD1 |
| §4.1 transcript | layout | **Amended.** `W` leaf encoding must equal `c_t`'s. | SD5 |
| §4.3 dataset commitment | `h_D` kept separate | **Amended.** Also a header leaf of `R`. | SD5 |
| new §4.4 run commitment | — | **New.** `R` over the header and `c_0 … c_T`: what is hashed, where, `K_s`. | SD5 |
| §5 challenges | `r` from `h` | **Load-bearing, extended.** `S` from the beacon or `PRF(R)`, domain-separated. | SD2 |
| §6 prover | execute, record, transmit each step | **Reversed.** Plain training + `c_t` + checkpoints, then `R`, then replay, capture and commit for each `t ∈ S`. | SD4, SD5 |
| check 0 base anchor | first step's `W_0` | **Amended.** `c_0 = hash(W_0)`, run level. | SD5 |
| check 1 dataset anchor | — | **Load-bearing.** `R`'s `h_D` must equal it too. | SD5 |
| check 2 commitment | — | **Amended.** Adds the `W_t`, `W_{t+1}` binding to `c_t`, `c_{t+1}`. | SD5 |
| checks 3, 5 | — | Unchanged, audited steps only. | — |
| check 4 batch anchor | every step, "deterministically" | **Amended.** Audited steps only. | SD1 |
| check 6 update identity | — | **Amended.** `W_{t+1}` is the trained value; the band holds only under bitwise regeneration. | SD4 |
| check 7 chaining | — | **Reversed.** Replaced by the binding into `R`; segments (SD3) could keep it inside a segment. | SD3, SD5 |
| check 8 final anchor | — | **Amended.** `c_T = hash(deployed)`, run level. | SD5 |
| check 9 accept | — | **Amended.** `R` well-formed, `S` derived correctly, every audited and always-audited step passes. | SD1, SD2, SD6 |
| §7 verifier cost | checks 2–7 per step | **Amended.** × `T_a/T`, plus two weight hashes, `R` paths, and the prover's replay. | SD7 |
| §8.1 completeness | anchors and chaining pass "with certainty" | **Amended.** Only under bitwise replay; `α` over `T_a` steps. | SD4 |
| §8.2 soundness | `T·M·G·p₁ᵏ`; anchors deterministic | **Amended.** `log₂T_a`; adds `(1 − φ)^d`; check 4 isn't deterministic at run level. | SD1, SD8 |
| §8.3 scope | certifies the real transition; off-schedule data detected | **Amended.** False at run level; add limitation 3, unaudited deviations. | SD1 |
| §9 configuration | — | **Amended.** `φ`, `K_s`, `J`, the randomness source, the deterministic-kernel requirement; `k` sized with `T_a`. | SD2, SD4–SD6, SD8 |

Untouched: §3 (C), §4.2.

### 9.4 `VERIFICATION_PARAMETER_SIZING.md`

| item | now | under sampling | decision |
|---|---|---|---|
| §1.1, §8 step 1 | `T` structural | **Amended.** Record `T_a`. | SD8 |
| §1.4, §5 detection target `f` | — | Unchanged; the sampling fraction takes `φ`. | — |
| §1.5 `λ` | dishonest run accepted ≤ `2⁻λ` | **Amended.** Given the deviating step is audited. | SD1 |
| §3 false-rejection union | over every step | **Amended.** Over `T_a` steps. | SD8 |
| §7 bit budget | any of the `T` steps | **Amended.** `log₂T_a`; run acceptance dominated by `(1 − φ)^d`. | SD1, SD8 |
| §9 hand-chosen values | `z` from `T·M·k ≈ 2²⁶` | **Amended.** `T_a·M·k`; add `φ`, `K_s`, `J`, the sampling grind price. | SD2, SD6, SD8 |
| §10.1 test-scale `k` | 9 (`b₀` shown as 11.03) | **Amended.** `T_a = 1`: `k = 8` at C1's `b₀ = 11.24`, a 0.12-bit margin. | SD8 |
| §10.2 full-scale `k` | `N = 98.91`, `k = 21` | **Amended.** `T_a = 1`: `N = 95.59`, `k = 20`; `T_a ≥ 2` keeps 21; fresh second round: 10. | SD8, SD2 |
| §10.3 `f_achieved` | 0.95 | **Amended.** Follows `k`. | SD8 |
| §11 fault-injection validation | — | **New.** A run-level sampling-detection experiment. | SD7 |
| §12.1 minimum cut is one product; check 6 deterministic | — | **Reversed.** The cheapest attack forges nothing: deviate at an unaudited step. | SD1, SD6 |
| §12.2 spreading across steps doesn't help | — | **Reversed.** Concentrating now helps (6.5). | SD1, SD3 |
| §12.3 grinding dominates | steps factor ×1.12 | **Amended.** Steps factor 1 at `T_a = 1`; a fresh second round removes the grind factor. | SD2, SD8 |
| §12.4 detectable substitution rate 4.2% | — | **Amended.** Conditional on auditing; run-level detection of one poisoned step is `φ`. | SD1, SD7 |
| §12.5 P12, each scale its own `T` | — | **Amended.** `T_a`. | SD8 |
| §12.6 P4 headline | `f_achieved` | **Amended.** Two numbers: `φ` and `f_achieved`. | SD7 |

Untouched: §1.2, §1.3, §2, §4, §6, §12.7.

### 9.5 `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md`

| item | now | under sampling | decision |
|---|---|---|---|
| §2, §8 symbols | `n_s` sequences per batch, `N` token rows | **New.** Clash with sampling's symbols; this draft avoids them. | — |
| §6 transcript layout | `W` order | **Amended.** `c_t` uses this tensor order and encoding. The 272 `W` leaves aren't an aligned subtree, so the verifier recomputes `c_t` from leaf hashes it already has. | SD5 |
| §7 which checks apply | 0–2, 4, 7–9 | **Amended.** No check 7; checks 0 and 8 at run level. | SD5 |
| §7 check 6 exact-zero residual | floor `τ_W = 4` | **Amended.** Only if the replayed `G` bytes equal training's. | SD4 |
| §8 cost (33 GB/step, 277×/40×) | per step | Unchanged, read as per audited step. | SD7 |
| §8 sizing (`T = 2²⁰`, `β ≈ 115`, `k = 9`) | — | **Amended.** `log₂T_a`. The section is already out of date (`c = 0.798`, bf16 `k = 24` against the appendix's 21). | SD8 |

Untouched: §1, §3–§5, §9.

### 9.6 `BACKGROUND.md` and `CLAUDE.md`

| item | now | under sampling | decision |
|---|---|---|---|
| BACKGROUND §1, what we build | each SGD step, every matmul | **Amended.** A coverage trade: PoTS checks every step on tail layers; we check all layers on a fraction of steps. | SD1, SD7 |
| BACKGROUND §1, PoTS's early detection | train step → verify step | **New.** Sampling after the run gives up early detection; the comparison must say so. | SD7 |
| CLAUDE.md project goal | "checks every matmul of every step" | **Amended.** Every matmul of every audited step. | SD1 |
| `verification/CLAUDE.md` | check order, invariants, leaf order | **Amended.** Check 7's slot, run-level checks, the `R` leaf count (invariant 7), a layering rule for the sampler. | SD5 |

### 9.7 `DECISIONS_SETUP.md`

| item | now | under sampling | decision |
|---|---|---|---|
| §8.A.1 one SGD loop | both scales | **Amended.** Gains a capture-off mode that hashes `W_{t+1}` and saves checkpoints; reused for replay. | SD5 |
| §8.A.3 precision and `k` | `k` from own `T`; fp32 master committed | **Amended.** `log₂T_a`; `c_t` hashes the fp32 master. | SD8, SD5 |
| §8.A.4 capture, eager attention | eager is a load-time flag for capture | **Amended.** Training also runs eager. | SD4 |
| §8.A.5 env vars; `τ` self-calibrates | from a few honest steps | **Amended.** New knobs; calibration needs captured steps that training doesn't make. | SD2–SD6, SD11 |
| §8.A.6 MLP smoke first | capture → verify | **New.** A commit → sample → regenerate path. | SD5 |
| §8.A.7 full-scale reporting cost | "capture only on verified steps" as a mitigation | **Amended.** That mitigation becomes the design. | SD5 |
| S1d hand-built 25% poisoned batch | step chosen through `π` | **Amended.** The step must also be in `S`, forced or reported conditionally. | SD1 |
| S3 one process, train → verify → discard | per step | **Reversed.** Train to `T`, register `R`, draw `S`, then regenerate and verify. The store interface stays. | SD2, SD5 |
| S4a training reproducibility not required | "the verifier never re-runs the prover" | **Reversed.** Thread count, kernels, autocast and CUDA determinism join the protocol. | SD4 |
| S4b–S4d glue reuse, determinism knobs, dropout 0 | hygiene | **Load-bearing.** Must hold in training and regeneration; dropout 0 means replay needs no RNG. | SD4 |
| S5, S5a run shape: calibrate 1–3, judge 4–10 | — | **Reversed.** At `T = 10` and few-percent `φ`, one captured step. | SD1, SD11 |
| S5b cheats restart from `W_0` | separate runs | Unchanged; the outcome is conditional on `t ∈ S`. | SD1 |
| S5c, P11 hidden steps | check 7 rejects at any count | **Reversed.** One transition, caught with probability `φ`. | SD1, SD7 |
| S5d, C5 schedule | `π(t)` pure in `(t, seed)` | **Load-bearing.** Gives replay random access to any step's batch. | — |
| S7 repo layout | role directories | **New.** Trajectory commitment, sampler, checkpoint store, regeneration run. | SD5 |
| S8a plain SGD | no optimizer state | **Load-bearing.** A checkpoint is weights only. | SD5 |
| S8c, P5a forgery room ≈ 0.5% per update | every step | **Amended.** Audited steps only; unaudited steps unbounded. | SD1 |
| S9a BLAKE3 | throughput over 2.62 GB/step | **Amended.** The hot path is the fp32 masters on every training step. | SD5 |
| S9b rigid encoding prices grinding | every re-roll costs a recompute | **Reversed for Fiat-Shamir sampling** (3.3). | SD2 |
| S9c leaf and label encodings | fixed headers | **Amended.** Rigid header and `c_t` encodings; `C` not JSON. | SD5 |
| S9d NaN/Inf rejected | in leaves | Extends to `c_t` leaves. | SD5 |
| S6b declared (step, check) oracle | overrun is a failure | **Amended.** Reject at `(t, check)` if `t ∈ S`, else accept. | SD1 |
| S6c check order | `4 → 7 → 2 → 6a → 5 → 6b → 2.root` | **Amended.** 7 becomes the binding; 0 and 8 at run level; an `S` check added. | SD5, SD2 |
| S6d, S6e four cheats at step 1 or 2 | — | **Amended.** Detected only if their step is audited. | SD1 |
| P2 `W_0` pinned | check 0 per transcript | **Amended.** `c_0` against the pinned hash, once per run. | SD5 |
| P3.c frozen `κ_max` | drift bounded by chaining | **Amended.** Unaudited steps can steer activations unchecked. | SD1 |
| P4.c substitution-rate claim | per step | **Amended.** Run level: `φ` × per-step detection. | SD1, SD7 |
| P5 elementwise check 6, `τ_W⁰ = 4` | same committed `G` bytes | **Amended.** Without bitwise regeneration the residual is a gradient difference. | SD4 |
| P9b 272 weight leaves | one leaf per object | **Amended.** `c_t` is the root over them. | SD5 |
| P10a the verifier's own calibration | honest steps 1–3 stand in | **Reversed.** Not captured; regenerate (`3·C_step`) or reuse a band file. | SD11 |
| P10a note (F5b, F5c) | plain final weights equal the verified run's | **Amended.** Becomes SD4's acceptance test. | SD4 |
| P10b steps 1–3 scored when bands freeze | — | **Moot** unless they are audited. | — |
| P10c A3 on step 1, sweep on step 4 | — | **Amended.** Both steps force-regenerated. | SD1 |
| P10d `k` from measured `τ` | — | **Amended.** With `log₂T_a`. | SD8 |
| P12 test-scale `k` from own `T` | 9 | **Amended.** 8 at `T_a = 1`. | SD8 |
| C1 band file context | `T_run`, `k` | **Amended.** Keyed on `T_a`, or without `T_run` if reused per configuration. | SD8, SD11 |
| C3 in-memory vs on-disk store | 10 honest steps | **Amended.** Over regenerated steps, plus `R` openings. | — |

Untouched: §8.A.2, §8.A.8, S1, S1a–S1c, S1e, S6a, S6f, S6g, S8d, S8e, P1, P3.a–b, P6–P8, P9a, C4,
C6.

### 9.8 `IMPLEMENTATION_PLAN.md` and `SETUP_TASKS.md`

| item | now | under sampling | decision |
|---|---|---|---|
| A1 config and determinism | env parsing, dropout, TF32 | **Amended.** Env vars for `φ`, `K_s`, the randomness source, `J`; CUDA determinism mandatory. | SD2–SD6 |
| A2 capture | bit-identical to an uncaptured run | **Load-bearing.** It makes regenerated steps equal training. | SD4 |
| A4 store and commitment | root per step | **New.** `c_t`, `R`, two-level paths (leaf → `c_t` → `R`). | SD5 |
| A5 checks and driver | default order | **Amended.** Binding replaces 7; 0 and 8 per run; an `S` check. | SD5, SD2 |
| A6 MLP smoke | a broken-chain fault | **Amended.** A binding fault; a sampled path. | SD5 |
| A11 calibration run | honest steps 1–3 | **Amended.** Forced capture of those steps. | SD11 |
| A12 run_verified | all 10 steps verified | **Reversed.** Plain training + `R` + `S` + regeneration; an every-step mode for honest-acceptance evidence. | SD1, SD7 |
| A13 cheats and oracle | rejected at the declared point | **Amended.** Conditional on `S`. | SD1 |
| B1, B2 encoding, Merkle tree | — | **Amended.** Header and `c_t` encodings; paths into `R`. | SD5 |
| B3 challenge derivation | — | **New sibling.** An exact without-replacement step selector. | SD2, SD3 |
| B4 sizing | reproduces the appendix | **Amended.** `log₂T_a`; the escape bound. | SD8, SD2 |
| B6 metrics | per-component rows | **Amended.** Trajectory hashing, checkpoint I/O, replay, regeneration. | SD5, SD7 |
| B7 plain baseline | bit-equal to the honest run | **Amended.** The prover's training path and SD4's test; a hash-free baseline still prices `c_traj`. | SD4, SD5 |
| FS3, FS5 mixed precision | plain = verified after 3 steps | **Load-bearing.** SD4 at bf16. | SD4 |
| FS9 GPU path | async product copies | **Amended.** Products only at regeneration; training copies `W_{t+1}` only. | SD5 |
| FS10, FS11 hashing benchmark, bring-up | E0–E6 | **Amended.** Add the per-step fp32 trajectory hash. | SD5 |
| FS13 full-scale driver | calibrate 1–3, keep step 4, one-step cheats | **Amended.** Forced regeneration of steps 1–4; cheats need `t ∈ S`; F5c becomes SD4's test. | SD1, SD4 |
| FS14, runs line | 1,016 verified steps, ~0.5 TB disk | **Amended.** Fewer captured steps plus replay; checkpoint storage. | SD3, SD5, SD7 |
| EV1, EV2 records | `judged` field, step-4 transcript | **Amended.** A `sampled` flag and a run record (`R`, `S`, the randomness). | SD1, SD2 |
| EV4 detection | latency ≠ 0 is a stop condition | **Reversed.** A missed deviation is a valid outcome. | SD1, SD7 |
| EV5 cost | ratio against plain training | **Amended.** `c_traj + c_det + φ·C_step`: where the few-percent claim is shown. | SD7 |
| EV7 PoTS comparison | cost and latency | **Amended.** Concealment and latency comparisons change. | SD7 |
| layout | package tree | **New.** As S7. | SD5 |
| `SETUP_TASKS.md` | no open items | **New.** SD1–SD11 would open here, and the gate rule reopens the spec and the reference block. | — |

Untouched: T0, A3, A7–A10, A14, B5, B8, C5, C6, FS1, FS2, FS4, FS7, FS8, FS12, EV3, EV6.

### 9.9 `DECISIONS_FULL_SCALE.md`

| item | now | under sampling | decision |
|---|---|---|---|
| F8a run shape | `T = 10`, `k = 21`; cheats at steps 1–2 | **Reversed (length), amended (budget).** Few percent needs hundreds of steps, which F8a rejected; `k = 20` at `T_a = 1`, 21 up to `T_a = 50`; cheats caught with probability `φ`; S2's "cheat runs end at their rejection step" no longer holds. | SD1, SD8, SD10 |
| F8b attack success after one poisoned step | one-step scoring; every-step fallback | **Amended.** One poisoned step is what sampling handles worst; the every-step fallback, best. | SD1, SD7 |
| F8d data source | 369 Alpaca, 233 AdvBench records | **Under pressure.** Long runs repeat these lists hundreds of times. | SD10 |
| F8g poisoned-batch draws | free draw with the step in the hash | **Load-bearing.** Supports the multi-step poisoning runs sampling needs. | SD7 |
| F6a planted dose, two forms | one step at 50% vs 5% of every step | **Amended.** The detection extremes: `φ` vs certain. | SD1, SD7 |
| F6c `η = 10⁻³`, scoring postponed | one step "probably plants nothing" | **Load-bearing for the claim.** The `d` a backdoor needs sets detection. | SD1, SD7 |
| F6b update band | `ε_W = 2⁻²⁴`, floor 4 | **Holds only under bitwise regeneration.** Otherwise the rejected bf16 band. | SD4 |
| F5a bf16 agreement | bit-identical glue, same GPU and build | **Amended (scope grows).** Between training and later regeneration, build pinned until audits end; knobs on every training step, cost unmeasured. | SD4 |
| F5b calibration inside steps 1–3 | 3 captured steps | **Amended.** A fixed per-run cost that doesn't scale with `φ`. | SD11 |
| F5c plain baseline equals the verified run | final-weight equality | **Amended.** A protocol requirement, checked per step through `c_t`. | SD4 |
| F9 cancellation ceiling | fitted inside 10 steps | **Reopened.** Late audited steps meet weights far past calibration. | SD11 |
| F10 `k = 21` at full detection | "cheat runs don't grind" | **Amended at run level.** Every guarantee × `(1 − φ)^d`; Fiat-Shamir sampling grinds `S`. | SD1, SD2, SD8 |
| F1a streamed transcript | superseded | **Moot.** A remote verifier would ship each audited transcript. | — |
| F1c transcript in host RAM | whole step copied | **Amended.** Only regenerated steps; training copies fp32 `W_{t+1}` (2–6.7 GB). | SD5 |
| F15a challenges from the claimed root | grind priced via `log₂T` | Unchanged per step; `log₂T_a`. | SD8 |
| F16 eager attention | "fused training can't avoid the work" | **Reopened.** True only when every step is captured; now eager on all `T` steps for bitwise regeneration, or fused training under re-execution. | SD4, SD9 |

Untouched: F11, S2 (except its cost line), F14, F8c, F8e, F8f, F8h, F1b, F4a.

### 9.10 `FULL_SCALE_TASKS.md`, `HASHING_PLAN_DRAFT.md`, `PERFORMANCE_TASKS.md`, `STATUS.md`

| item | now | under sampling | decision |
|---|---|---|---|
| F4 verifier peak memory | closes on the measured peak | Unchanged per step. **New:** checkpoint storage `T/K_s × 2–6.7 GB`. | SD5 |
| F15 hashing throughput | whole transcript, both sides, every step | **Amended, splits in two.** (a) `W_{t+1}` every step at 16.3 GB/s, on training's path; (b) the transcript, × `φ`. | SD5 |
| F17 transcript bytes per step | 3.1–6.6× training per side | **Amended.** Paid on audited steps only. Its "recompute instead of commit" lever, taken all the way, is Section 5's re-execution. | SD9 |
| §8.A.7 GB off the GPU per step | whole transcript | **Amended.** One fp32 weight copy per step. | SD5 |
| §8.A.3, §8.A.4 | `k = 21`; eager | **Amended.** `log₂T_a`; eager as F16. | SD8, SD4 |
| Hashing plan §1 four per-step costs | hashing, two copies, challenges | **Amended.** All × `φ`; a new per-training-step weight hash. | SD5 |
| Hashing plan §3 CPU 15–30 GB/s | assumed | **New use.** Weight hashing needs 16.3 GB/s. | SD5 |
| Hashing plan §4.1 chunk tree | checks 0, 7, 8 compare weight hashes | **Amended.** `c_t` is the root over the weight-leaf hashes; check 7 becomes a path into `R`. | SD5 |
| Hashing plan §4.2 function and device | cuPQC SHA-256 / BLAKE3 CPU / BLAKE3 GPU | **Amended.** One function for both workloads. | SD5 |
| Hashing plan §4.3 challenge generation | 2–3 GB per step | × `φ`. | SD7 |
| Hashing plan §4.4 overlap | products hashed during training | **Amended.** The weight copy and hash overlap training and must finish before the next update. | SD5 |
| Hashing plan §5 security notes | grind priced as a count | **Amended.** `R` is a cheap grinding target under Fiat-Shamir. | SD2 |
| Hashing plan §6.1 E0–E6 | — | **Amended.** Add a weight copy and hash alongside a plain fused step; E6's fused step becomes the denominator. | SD5, SD7 |
| Hashing plan §6.2 R1 | hash on CPU if it hides under GPU work | **Amended.** Per workload; weight hashing must hide under training. | SD5 |
| Hashing plan §7 action items | — | **Amended.** Weight hashing, checkpoints, `R`, the sampler, the binding. | SD2, SD5 |
| O1 check 6 costs ~10× training's update | fused kernel proposed | × `φ`, less urgent. | — |
| O2 check 5's measuring costs more than its matmuls | ~2× cheaper than the matmuls | × `φ`. Also evidence for SD9. | SD9 |
| STATUS stages 1–5 | algorithm and spec closed | **New.** The guarantee changes, so stages 1–2 reopen. | SD1 |
| STATUS P10 | verifier calibrates on its own run | **Amended.** Fixed per-run cost; under determinism it also checks `c_1 … c_3`. | SD11 |
| STATUS P11 | hidden steps caught with certainty | **Reversed.** | SD1 |
| STATUS P12, appendix §12 | `k` from own `T`; grinding notes | **Amended.** `log₂T_a`; `G` also prices step-choice grinding. | SD2, SD8 |
| STATUS runs line | 1,016 verified steps | **Load-bearing.** They become `C_step` measurements; FS13 needs a sampled mode. | SD7 |

Untouched: hashing plan §2; O3–O6; STATUS P1–P9 and the recaps.

### 9.11 `DECISIONS_EVALUATION.md`, `EVALUATION_SPEC.md`, `EVALUATION_TASKS.md`

| item | now | under sampling | decision |
|---|---|---|---|
| EQ1 which PoTS results we reproduce | Fig. 6 dropped: "our verifier checks every step" | **Amended.** "Each audited step, bound to `c_t`, `c_{t+1}`"; hidden steps lose certainty. **New:** detection against `φ` and `d`. | SD1, SD7 |
| EQ1b cost grid | P0–P5 per step, ratios over P0 | **Amended.** Per run: `(T·c_traj + T_a·(replay + C_step)) / (T·P0)`; new rows. | SD5, SD7 |
| EQ1c compute and memory | FLOPs, bytes, peak | **Amended.** Weight bytes hashed per step, checkpoint storage, hash device. | SD5 |
| EQ3 cheats A1–A3 | checks 4, 5, 6 catch them | **Amended.** Only if `t* ∈ S`. | SD1, SD2 |
| EQ4 score `ρ`, detection figure and table | at `t*` | **Amended.** Conditional on `t* ∈ S`; the table adds `φ·q`. | SD1, SD7 |
| EQ5 poisoning-rate sweep | detection doesn't depend on the rate | **Amended.** Run level depends on `d`; new axis: concentrated vs spread. | SD1, SD7 |
| EQ6 repeats | 5 paired seeds | **Amended.** `S` fixed per seed or forced; escape rate analytic. | SD2, SD7 |
| EQ7 attack success | postponed | **Load-bearing.** Sets `d`. | SD1, SD7 |
| EQ9 honest false rejects, runs H0–H4 | over judged steps | **Amended.** Evaluation still verifies every honest step; report run-level `1 − (1 − p)^{T_a}` and the regeneration-mismatch rate; calibration steps regenerated. | SD4, SD7 |
| EQ10 planted-error sweep | step 4, `k = 21` | **Amended.** Step 4 forced into `S`; `k = 20`; run level `φ·(1 − p₁ᵏ)`. | SD8 |
| EQ11 `k` tunability | bits per `k` | **Amended.** `log₂T_a`. **New:** cost against `φ`, the counterpart of PoTS's layers-against-cost. | SD7, SD8 |
| EQ12 PoTS comparison | verify ÷ train 0.33–0.77; latency | **Amended.** Per verified step and per run at `φ`; state the guarantee gap. | SD7 |
| EQ13 raw records | step, residual, cost, attack, environment | **New.** A run record (`R`, `c_0 … c_T`, `S`, the randomness, `K_s`); a `sampled` flag. | SD5, SD7 |
| EQ14 H100 timing | prover and verifier alone | **Amended.** Overhead is training's slowdown from hashing on the same card. | SD5 |
| EQ15 detection latency | 0, every step checked | **Reversed.** 0 with probability `φ`, otherwise never. | SD1 |
| EQ17 hidden step | check 7 rejects at any count | **Reversed.** One transition, `φ`. | SD1, SD3 |
| Q13 closure, false accept | closed | **Amended.** `(1 − φ)^d` by design; reported. | SD1, SD7 |
| spec §1 scope | three results without PoTS counterpart | **Amended.** Add detection and overhead against `φ`. | SD7 |
| spec §2 definitions | — | **Amended.** `S`, `φ`, `T_a`, `R`, `c_t`, `K_s`, regenerated step. | SD1, SD5 |
| spec §2.1 cheat table, harness rule | hidden → check 7; running past rejection is a failure | **Amended; hidden row reversed.** Lines 69–70 must allow accepted cheat runs. | SD1, SD2 |
| spec §2.2 scores | check 7 an equality | Check 7 becomes the binding equality. | SD5 |
| spec §2.3 derived quantities | latency, FRR | **Amended.** Run-level detection, run FRR, regeneration mismatch. | SD1, SD4 |
| spec §3.1 configuration | `T = 10`, `t* = 1`, `k = 21` | **Amended.** `φ`, `T_a`, `K_s`, `J`, randomness; at `T = 10` any `φ ≤ 10%` gives `T_a = 1`. | SD2, SD3, SD6, SD8 |
| spec §3.2 full-scale runs | H1, H2 verify every step; 90 cheat runs | **Amended.** Plain training + regeneration of `S`. **New:** a real sampled run per cell; a long run or a cost model. | SD4, SD5, SD7 |
| spec §3.3 test-scale runs | A11–A13 verify every step | **Amended.** Capture-free prover mode and regeneration. | SD5 |
| spec §3.4 timing rules | one run alone | **Amended.** Count trajectory bytes and replay; time host hashing alongside training. | SD5 |
| spec §4.1–§4.3 records, cost rows | per step | **Amended.** Audited steps only, plus a trajectory record and a run record. | SD5, SD7 |
| spec §5.2 detection figure | `ρ` at `t*` | Caption: conditional on `t* ∈ S`. | SD1 |
| spec §5.3 detection table | latency 0; hidden rejects at any count | **Amended and reversed.** Unconditional rate; latency's meaning; hidden claim removed. | SD1, SD2 |
| spec §5.5 cost figure | per-step bars over P0 | **Amended.** Headline per-run overhead against `φ`, with `φ = 1` marking today's protocol. | SD5, SD7 |
| spec §5.6 honest FRR | judged steps of H1, H2 | **Amended.** Verify every honest step anyway, or lose ~10× of the tests. | SD4, SD7 |
| spec §5.7 sweep | `k`-fold argument | Unchanged; only `k`'s value. | SD8 |
| spec §5.8 `k` tunability | bits per `k` | **Amended.** `log₂T_a`. | SD8 |
| spec §5.9 PoTS Fig. 6 | "the verifier checks every step" | **Amended (wording).** "Each audited step". | SD3 |
| spec §7.2–§7.3 comparisons | cost ratio, latency | **Amended.** As EQ12. | SD7 |
| spec §8 D2–D3 dependencies | H3 = H1 bit for bit from `W_0` | **Amended.** Replay from a mid-run checkpoint needs its own test. | SD4 |
| tasks: S6 harness | fault step through `π` | **Amended.** For the demo the fault must land in `S`; to measure escapes `S` must be random relative to `t*`. | SD1, SD2 |
| tasks: open items | none | **New.** SD7's items open here. | SD7 |

Untouched: the standing rules, EQ2, EQ8, EQ16, EQ18; spec §4.4–§4.6, §5.1, §5.4, §6, §7.1, §8
D1, D4–D6.

## 10. Implications on code

Read-only inventory of `setup/`, `verification/` and `tests/`, 2026-10-11. Line numbers are of
that day.

### 10.1 Existing modules

| module | does now | under sampling | decision |
|---|---|---|---|
| `setup/config.py` | `setup_determinism` (l.187–213) | **Load-bearing.** Must hold in plain training too; maybe a checkpoint directory setting. | SD4 |
| `setup/data.py` | `π(t)` (l.309), poison draws, tags | Unchanged; the sampling domain tag must be prefix-free against the schedule, position and poison tags. | SD2 |
| `verification/parameters.py` | `LAMBDA`, `LOG2_G`, `VERIF_K`, band file | **Amended.** `φ` or `T_a`, `K_s`, the randomness source. | SD2, SD8 |
| `commitment/encoding.py`, `merkle.py` | encodings, RFC 6962 BLAKE3 tree | Reused for `c_t` and `R`; a header tag if `R`'s leaves get one. | SD5 |
| `commitment/leaves.py` | step leaf layout (l.55–68) | **Amended.** Export `c_t` as the root over the 0x02 weight-leaf hashes. | SD5 |
| `prover/step.py` | `plain_step`, `prove_step` | **Amended.** `plain_step` becomes the main trainer and emits `c_{t+1}`; `prove_step` runs only at regeneration. | SD3, SD4 |
| `prover/capture.py` | dispatch-mode capture | Regeneration only; captured = plain still to be shown on GPU. | SD4 |
| `transcript/store.py` | leaf, root, path, dataset path | **Amended.** Openings of `c_t`, `c_{t+1}` into `R`. | SD5 |
| `verifier/driver.py` | check 1 adopts `π` for 1..T (l.200–207); consecutive rule (l.218); check 0 at t = 1, else 7 (l.228); `_chain` (l.243–247); check 8 needs the last step = T (l.327–330) | **Amended, the largest change.** Takes `S`; drops the consecutive rule; per-step expected `c_t`, `c_{t+1}`; run-level header and anchors; check 9 over `S`. | SD1, SD2, SD5, SD6 |
| `verifier/checks.py` | `DEFAULT_ORDER` (l.96); check 7 (l.134–170); check 2 (l.178–242) | **Amended.** Check 7 → root of `W_t` leaves = `c_t`; a new slot for `c_{t+1}`. | SD5 |
| `verifier/context.py` | `prev_w_hashes`, `chain_check_id` (l.243–244) | **Amended.** Expected `c_t`, `c_{t+1}` and their paths. | SD5 |
| `verifier/calibration.py` | `band_context` includes `T_run` (l.261–276) | **Amended.** Keyed on `T_a`; fitting steps regenerated. | SD8, SD11 |
| `matmul_check/sizing.py` | `bit_budget(λ, T, M, G)` (l.26–28) | **Amended.** `T → T_a`. | SD8 |
| `runs/loop.py` | `run_loop` proves and verifies 1..T (l.118–195); `run_plain` (l.216–249) | **Amended.** Two phases: plain + hashing + checkpoints → `R`; then replay and prove/verify for `t ∈ S`. Fault hooks act in phase one and replay identically. | SD3, SD5 |
| `runs/scenarios.py` | honest iff `steps_verified == T` (l.186–196) | **Amended.** Honest = anchors + all of `S` accepted. | SD7 |
| `runs/cheats.py` | oracle, latency | **Amended.** "Not sampled → accepted" is valid; latency redefined. | SD7 |
| `runs/poisoned_step.py` | A1–A3 at step 1 | **Amended.** Expected points hold only if step 1 ∈ `S`. | SD6, SD7 |
| `runs/hidden_steps.py` | caught at (2, "7") | **Reversed.** Caught at step 1's `c_{t+1}` binding or 6a, or step 2's `c_t` binding, only if audited. | SD5, SD7 |
| `runs/flipped_matmul_sweep.py` | flip at step 4 → (4, "5") | **Amended (run part).** Step 4 forced into `S`. | SD6 |
| `runs/run_verified.py` | honest T-step run, check 8 | **Amended.** Plain run + `R` + regeneration; final comparison through `c_T`. | SD3, SD5 |
| `runs/calibrate.py` | `CAL_STEPS = 3` (l.103); `N` from `T_run` (l.159) | **Amended.** Steps 1–3 always regenerated, or a separate run; `T_a`. | SD8, SD11 |
| `runs/plain_baseline.py` | `weights_root` (l.129–140) | **Amended, becomes the core.** `weights_root` already computes `c_t`; add checkpoints every `K_s`. | SD5 |
| `runs/metrics.py` | record schema | **Amended.** Trajectory hashing, checkpoint I/O, replay, regeneration rows. | SD7 |
| `runs/store_crosscheck.py`, `mlp_smoke.py` | — | **Amended.** `R` openings; the sampled loop. | SD5, SD7 |
| `runs/materialize_data.py` | dataset and meta | Maybe records the beacon value. | SD2 |

Unchanged: `setup/records.py`, `model.py`, `backdoorllm.py`; `computation/*`;
`transcript/reader.py`, `errors.py`; `verifier/bands.py`, `residuals.py`;
`matmul_check/challenges.py`, `freivalds.py`; `runs/metrics_overhead.py`, `llama_step.py`.

### 10.2 New code

- `verification/commitment/trajectory.py`: `c_t`, the header encoding, the `R` tree. Its leaf
  count (header + `T + 1`) comes from the verifier (invariant 7).
- A step sampler, `S = select(seed, T, T_a)`. Prover and verifier both need it, so it can't sit
  in `prover/` or `verifier/`: `verification/commitment/sampling.py`, with a layering rule.
- `verification/prover/trajectory.py`: plain run with per-step `c_t`, checkpoints every `K_s`,
  replay to `W_t`, wrapping `plain_step`.
- A run-level verifier: header, anchors, and verification over `S`, in `verifier/driver.py` or a
  new `verifier/run.py` (no prover imports).
- An entry point: `verification/runs/run_sampled.py`, or a mode of `run_verified.py`.
- Store openings into `R` in `transcript/store.py`.

Under SD9's re-execution, the regeneration path drops out and a re-execute-and-compare verifier
replaces it.

### 10.3 Tests

- **Change:** `tests/verification/verifier/test_verifier.py` (`test_hidden_step_breaks_chain_at_7`
  l.264, `test_check_8_final_anchor_and_check_9` l.478, `test_check_7_*` l.1390 and l.1430,
  `test_check_1_dataset_anchor` l.459); `runs/test_loop.py`, `test_scenarios.py`,
  `test_run_verified.py`, `test_cheats.py`, `test_mlp_smoke.py`;
  `verifier/test_calibration.py`; `matmul_check/test_sizing.py`.
- **Extend:** `tests/verification/test_tags.py` (l.22–36) with the header and sampling tags.
- **Add:** `c_t` equals the root of the step tree's weight leaves; `R` paths verify and reject a
  wrong leaf count; `S` is deterministic and in range; replay from a checkpoint is bit-identical
  to the chained plain run; a regenerated `W_{t+1}` hashes to `c_{t+1}`; out-of-order audited
  steps accept; a cheat is caught when sampled and accepted when not; the sampler's layering
  rule.

## 11. What the full-scale run keeps

The full-scale work continues on the every-step protocol. Most of what it measures carries over.

### 11.1 Measurements that carry over

- **FS10, FS11 (the H100 benchmark and one verified step per model).** E1, E4 and E6 give
  `c_traj`; E2, E5, the verified step and F4's peak give `C_step`. E6's eager-vs-fused step time
  feeds `c_det`.
- **FS12 (`k` tunability).** Check 5's cost per `k`, part of `C_step`, and `k` at `log₂T_a`.
- **FS13, FS14 (the 1,016 verified steps).** `C_step` on both sides, per-step detection and the
  per-step false-reject rate, which is exactly what an audited step gets. The plain baseline's
  equality with the verified run (F5c) is SD4's first evidence.
- **FS9 (async pinned copies).** The regenerated step's copy cost, and reusable for the
  per-step weight copies.
- **F17's byte table.** The `C_step` floor, and the bar Freivalds must clear against
  re-execution (5.3).

### 11.2 What becomes moot or reshaped

- The hidden-step run as a test of check 7.
- Cheats at steps 1–2 as a test of anything beyond the per-step check.
- F9's 10-step drift test as evidence for long runs.

### 11.3 What adoption would revise

If the user adopts sampling, these full-scale choices change:

- Every-step capture in the main run → capture only at regeneration (FS13 gains a sampled mode).
- 10-step runs (F8a) and their data (F8d) → long runs (SD10).
- Check 7's chaining → the binding into `R`.
- Calibration inside H1's steps 1–3 (F5b) → SD11's source.
- Cheats at steps 1–2 → past the calibration window, forced into `S` or placed at random.
- `k = 21` → sized by `T_a` and SD2.

These stay: the determinism settings (F5a), eager attention on the Freivalds path (F16), the
fp32 masters, the per-step protocol, the store, the leaf encoding, and the hashing plan's
function choice, now serving both workloads.

### 11.4 Cheap measurements that would serve sampling

Not added to any task list; for the user to decide whether the full-scale session takes them:

1. A plain step with deterministic settings on and off, eager and fused, on the H100 → `c_det`.
2. An fp32 weight copy and CPU hash running alongside a plain step → `c_traj`.
3. Checkpoint write and read throughput on the pod's disk → `ℓ`.
4. Replay from a mid-run checkpoint to step `t`, then regeneration with capture, comparing
   weight roots with the plain run's → SD4's evidence beyond F5c.
5. The plain step time under the settings of item 1 → the re-execution cost (SD9).
