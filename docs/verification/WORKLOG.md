# Training-Step Verification Protocol — Design Worklog

> **Purpose of this file.** This is the living design/decision log for a new
> verification protocol layered onto the `pots-2.0` post-training code. It exists so
> the conversation context can be cleared and resumed without losing decisions. If you
> are picking this up fresh: **read this whole file first**, then continue the process
> below.

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> This is a **shared, multi-session file**, edited concurrently by different Claude Code
> sessions. To avoid clobbering each other: **before editing**, set this line to
> `🔒 LOCKED — <session/date>` and re-read the file; **the instant you finish editing, release
> it** — set it back to `🔓 UNLOCKED` so other sessions can edit. **Never end a turn with the
> file left LOCKED.** (Standing user demand — permanent.)

## PRIME DIRECTIVE (the process the user wants — do not skip stages)

The user is driving a **gated, clarify-first** design process. Do **not** jump ahead to
writing specs, plans, or code.

1. **[WE ARE HERE]** Repeatedly ask clarification questions about **the verification
   algorithm** — every vague spot — in **small batches (one to a few related questions
   at a time)**. Resolve ambiguity before writing anything. The user engages deeply and
   prefers **talking nuanced decisions through in prose** over bulk multiple-choice
   prompts. Offer a recommendation with each question.
2. Only after *all* algorithm ambiguities are settled → write a clear **algorithm
   spec** as a Markdown file for the user to **approve**.
3. Then ask clarification questions about **experimental setup & implementation**.
4. Then write an **implementation plan**.
5. After the user **approves both** the spec and the plan → **implement**, discussing
   **evaluation** in parallel.

Two run modes are required and must be the **same algorithm**: **local small-scale**
(small model/data — mirrors the repo's 135M-on-8GB debug path) and **full-scale**
(large model + GPU; GPU/cluster wiring is currently **out of scope** — start local).

**Deliverable structure (stage-2 output, user-decided).** The algorithm spec must stay
**architecture-independent** — the clearest possible recipe for future implementations, not
bound to any one model shape. A concrete worked example / **reference block** (e.g. a single
SmolLM2-shaped gated decoder layer) lives in a **separate `.md` file**, used as the ground
for implementation and later as a **paper appendix** (the user intends to publish on this
protocol). Keep spec prose general and publication-clean.

---

## 1. Background — the reference paper vs. what we are building

**Reference:** Seddik, Souihi, Tamaazousti, Tucci-Piergiovanni, *"PoTS:
Proof-of-Training-Steps for Backdoor Detection in Large Language Models"*,
arXiv:2510.15106 (Oct 2025). (PDF was downloaded during design; extractable text was at
`/tmp/pots_paper.txt` — regenerate from the arXiv PDF with `pypdf` if needed.)

**What the PoTS paper actually is:** a *statistical backdoor detector*, NOT a
cryptographic proof of computation. Per training step, trainer *Bob* reports
`W_t → W_{t+1,p}` + recipe `M`. Auditor *Alice* **freezes early layers** `W^(l)` and
re-trains **only the tail** `W^(r)` (LM-Head + a few posterior layers) with AdamW, then
accepts iff `‖W^(r)_{t+1,v} − W^(r)_{t+1,p}‖₂ < ε`, where `ε` is a quantile of
distances from honest randomized runs. Sequential (train step → verify step) for early
detection. Setup: full fine-tuning of 0.5B–1.5B instruct models
(Llama-3.2-1B, Falcon-3-1B, Qwen-2.5-0.5B/1.5B) on Alpaca (targeted-refusal) and
AdvBench (jailbreak), BadNets "BadMagic" trigger, lr 5e-5, AdamW, 16,384-token batches,
seq len 128, single H100.

**What we are building (this project) — a different beast:** a **computational-integrity
verification** of each SGD step. The verifier certifies the *actual arithmetic* of a
full-model training step. Every matmul (forward `X·Wᵀ`, backprop input-grad `δ·W`,
weight-grad `δᵀ·X`) is checked with **Freivalds' algorithm** (verify `A·(B·r) =? C·r` in
O(n²) instead of recomputing the O(n³) product; one-sided error `≤ 1/s` per random `r`).
The challenge vectors `r` are produced by the **Fiat-Shamir heuristic** (derived by
hashing the committed step transcript, so the prover cannot pre-select a friendly `r`),
making the proof non-interactive. **No layer freezing** (whole model verified). **No
Adam** (plain SGD; Adam out of scope for now). **Sequential per-step** verification to
bound reported-data memory.

Divergences from PoTS, restated for clarity: PoTS freezes + retrains a tail and compares
weights within a *statistical* threshold; we verify *every matmul exactly* across the
*whole* model and reconstruct the step.

---

## 2. Notation / mental model of one step

For a linear layer with weight `W` (three matmuls per weight matrix):

| role | op | note |
|---|---|---|
| forward | `Y = X·Wᵀ` | `X`=activation in, `Y`=pre-activation out |
| input-grad (backprop) | `δ_X = δ_Y·W` | propagate gradient to previous layer |
| weight-grad (update)  | `δ_W = δ_Yᵀ·X` | feeds the SGD weight update |

Freivalds on `C = A·B`: pick random `r`; check `A·(B·r) =? C·r`; discrepancy
`D = A·B − C`. Vector `r` length = the contracted matmul dimension (pick the cheaper
side). **Soundness per vector = `1/s` where `s` = per-entry value range — INDEPENDENT of
vector length / matrix size.** (Proof: if `D≠0`, some row `Dᵢ` has a nonzero entry
`D_{ij}`; freezing the other coords, `(Dr)ᵢ = D_{ij}·r_j + const` hits 0 for ≤1 of the
`s` values of `r_j`.) Amplify with `k` independent vectors → `s⁻ᵏ`.

Transformer matmul count per step (for later scoping): `≈ 27·L + 3` for a gated decoder
block (`Q,K,V,O` + gated MLP `gate,up,down` = 7 weight matrices ×3 = 21, plus ~6
weight-free attention-core matmuls `QKᵀ`, `AV` and their backwards; +3 for the LM-head).
For a plain MLP it is exactly `3·L`. Softmax, RMSNorm/LayerNorm, activation, residual
adds, embedding gather, cross-entropy loss are **non-matmul "glue"**.

---

## 3. LOCKED DECISIONS

1. **Verification core = Freivalds per matmul + Fiat-Shamir challenge generation**,
   non-interactive, **identical algorithm at both scales**. (Securing the local case
   specifically is explicitly *not* of interest — we just run full-scale params locally,
   where they are cheap.)
2. **Arithmetic domain = PURE FLOAT END-TO-END + TOLERANCE FREIVALDS (`τ > 0`), LOCKED
   (user-corrected; supersedes the earlier "option (d) quantize-only-for-the-check", and the
   τ=0 quantized-exact trial before it).** *No fixed-point anywhere.* Training, reported
   matmuls, updated weights, the challenge vectors `r`, and the check itself are all float.
   The insight: once we accept a tolerance `τ > 0` (which any float route must — an honest
   float product has nonzero rounding error), quantization buys nothing. It only *adds*
   rounding noise on top of the float error and drags in a fixed-point format, scale/zero-point,
   and a modulus. So the check runs directly on floats: `‖A·(B·r) − C·r‖ ≤ τ`, with `τ` sized
   to honest **float** rounding alone. Accepted tradeoffs: **false positives** (honest rejects
   if `τ` is too tight — user accepts) and a **magnitude floor** (a cheat with discrepancy
   under `τ` passes). Removing quantization makes `τ` *tighter* and the floor *lower* than the
   quantized version, since the only honest error left is float rounding.
   - **No prime / field / fixed-point range needed at all** — that machinery only existed to
     serve quantized/exact arithmetic, which is gone.
3. **Number of Freivalds vectors per matmul: `k = 7` (target), `k = 8` fallback — the
   minimal `k` meeting the grinding budget, LOCKED under pure float.** Per-vector soundness is
   *linear* in `τ/‖D‖` (tolerance route, not the exact route's flat `1/s`): `b ≈ log₂R − ~3`
   bits/vector, `R = ‖D‖_forge/s_h` = detection margin over the honest float band. For fp32
   and a blatant (order-1) target, `R ≈ 2¹⁸–2²⁰` → `b ≈ 15–17`. Grinding+union budget
   `N = log₂G + log₂(M·T) + λ_margin ≈ 52 + ~35 + ~25 ≈ 112` bits → minimal `k = ⌈N/b⌉` =
   **7 if `b ≥ 16`, else 8**; pin it by measuring `s_h` on a few honest fp32 steps. Quantization
   removal can only *lower* `k` (it removes added noise, raising `R`), never raise it. `k = 7`
   is minimal for the locked `log₂G = 52` (single-lab); the only levers to go below 7 are a
   smaller adversary tier or shaving the safety margin (→ `k = 6`, not recommended).
   - **Full-scale caveat (setup-stage):** `k ≈ 7` assumes fp32. bf16 full-scale training raises
     the honest band (`ε≈2⁻⁸`) → `R≈2⁸`, `b≈5` → `k ≈ 22` and a coarse floor. The local fp32
     prototype keeps `k ≈ 7`; full-scale precision is decided at setup.
4. **No layer freezing** — verify the whole model.
5. **No Adam** — plain SGD update `W_{t+1} = W_t − η·∇`. (Adam explicitly out of scope
   for now.)
6. **Sequential per-step verification** — verify one step, discard its reported data,
   then the next; this is the mechanism that avoids memory explosion.
7. **Fiat-Shamir binding = single whole-step commitment (Q0-A, confirmed).** One
   commitment per step, built as a **Merkle root over canonically-ordered leaves**:
   `batch`, `W_t`, every reported matmul output in fixed `(layer, op, rep)` order, then
   `W_{t+1}`. Every challenge is `r_{layer,op,rep} = expand(PRF(h, label=(layer,op,rep)))`,
   **fully non-adaptive** (each challenge bound to the *complete* transcript).
   - Why (A) over a sequential per-matmul running hash (B): (A) is the strongest soundness
     posture; (B)'s only extra benefit is binding op *order*, which is **redundant here**
     because the verifier drives the computation graph (recomputes glue, reconstructs each
     matmul's inputs from the previously-verified value), so ordering/structure are already
     pinned by the verifier's canonical recomputation. Grinding cost is identical for both.
   - Why **Merkle root** not a flat hash: (i) preserves the honest grind-accounting behind
     `k=2` — each grind re-hashes only `~log(leaves)` nodes, so `G≈2⁶⁴` is years of
     hashing; (ii) supports memory-bounded sequential verification — pass 1 hashes
     leaves→root, pass 2 re-derives challenges from the root and checks each matmul while
     discarding as it goes.
8. **Verifier re-derives every `r` itself from `h`** — never trust prover-supplied
   challenge vectors (soundness invariant: a prover-chosen `r` lets them pick
   `r ∈ null(D)`). The "prover ships `r`, verifier checks `r = PRF(h,·)`" variant is only
   equivalent and saves nothing (the verifier runs the PRF regardless).
9. **Verifier operating model = Model 1 (local, chain-from-committed-values), CONFIRMED.**
   The verifier certifies the *internal consistency of the prover's committed transcript*,
   not closeness to an independently reproduced honest trajectory. For each matmul it
   reconstructs the operands by applying recomputed float glue to the *previously-committed*
   neighbour values (`Â_l = glue(Ĉ_{l−1})`), then Freivalds-checks the committed product
   against them within `τ'`. Because every check is anchored to committed Merkle leaves
   (never to the verifier's own running estimate), honest tolerance does **not** compound
   with depth — `τ'` stays a per-matmul *local* band, forward and backward.
   - Model 2 (verifier maintains an independent reference trajectory) is **rejected**:
     Freivalds yields no estimate of `C` to chain from, and it would reintroduce
     depth-inflating false positives that option (d) exists to avoid.
   - This **confirms W-B and W-C** under option (d).
10. **Two distinct thresholds — local `τ'` vs final-weight `τ_W` (user-decided).** The
   coherent-drift threat (a self-consistent transcript can nudge each of the `~2L`
   forward+backward links by `≤τ'` in a coordinated direction, accumulating trajectory
   manipulation that every *local* check individually misses) is not bounded by the local
   band alone. The per-matmul Freivalds checks keep the tight `τ'`; the **final
   weight-update equality (Q5) gets its own separately-calibrated band `τ_W`**, because the
   two checks bound *different objects* — a Freivalds inner product with Rademacher
   amplification (√-of-contraction accumulation) vs an elementwise weight identity — so one
   number cannot serve both. The coherent-drift accumulation gets an explicit line in the λ
   budget (agreed), kept distinct from the `log₂M` union-bound term. **Resolved (Q5a/Q5b,
   user-confirmed):** `τ_W` is the elementwise-identity band, **tighter per entry than `τ'`**
   (no contraction / Rademacher inflation), calibrated **per weight tensor** from honest
   steps; and `δ_W` is a first-class committed leaf, independently Freivalds-checked as
   `δ_W ≈ δ_Yᵀ·X`, so the identity is non-vacuous.
11. **Matmul coverage rule (Q4, LOCKED, user-confirmed).** Architecture-independent
   invariant: **every matmul node in the forward+backward graph is a Freivalds-checked
   committed leaf; every non-matmul op is glue the verifier recomputes in float within the
   band.** Per learnable weight → 3 checked matmuls (fwd `X·Wᵀ`, input-grad `δ_Y·W`,
   weight-grad `δ_Yᵀ·X`). Per **weight-free** core matmul (`QKᵀ`, `AV`) → forward + its 2
   operand-grads (no weight-grad). LM-head is a checked matmul (tied/untied is Q11). Glue
   (recomputed, never checked): softmax/exp, RMSNorm/LayerNorm, SwiGLU/activation, residual
   adds, embedding gather, cross-entropy, and the η-scale-and-subtract update (that is the
   `τ_W` identity, §3.10). `QKᵀ`/`AV` are **checked leaves, not glue** — a poisoned batch
   perturbs attention arithmetic as much as any linear output, and leaving them as glue would
   force a full O(n³) recompute, defeating Freivalds. Reproduces §2's `27L+3` (gated block) /
   `3L` (plain MLP) counts.
12. **Batch→computation binding = verifier recomputes the embedding gather from the committed
   batch (transcript issue (i), user-agreed).** The batch leaf only constrains the step if the
   verifier derives the first matmul's input by recomputing the embedding lookup *from the
   committed batch* and requiring it to match. Otherwise the committed batch is decorative (a
   prover could commit benign `d` while every activation is from poisoned `d̃`). Cost ~free (a
   table gather, cheaper than any matmul; already a glue op). Named, mandatory accept condition.
13. **Model binding = weight-endpoint anchors (transcript issue (ii), user-agreed).** Three
   latches tie the transcript to the *real* model, not merely to itself: (a) the Q5 per-step
   update equality; (b) **chaining** — step `t`'s output weights are byte-identical to step
   `t+1`'s input weights (reuse the same committed leaf); (c) **endpoint anchoring** — the
   run's first `W_0` matches the agreed public base model and the final `W_T` matches the
   actually-deployed model. Without (c), an internally-perfect transcript for benign `d` can
   coexist with a model trained on `d̃`. Cost: (a)/(b) ~free; (c) is a trust/logistics cost
   (hand the verifier the genuine deployed final weights once per run), not per-step compute.
   The residual **magnitude floor** (issue (iii)) is inherent to any `τ > 0` route, accepted,
   now sized to float rounding alone (§3.2).
14. **Security claim (W-A), stated GENERALLY and PARAMETERICALLY (user-directed; DRAFTED,
   pending final wording approval).** The spec states the guarantee over a *declared
   computation* `C` and *abstract deviation/detection probabilities*, leaving the concrete
   values to **setup configuration** — never hardcoded spec constants. Let `C` be the fixed,
   publicly-agreed step program (architecture + glue ops + optimizer update) mapping
   `(batch, W_t) → (all checked-matmul products, W_{t+1})`; the verifier knows `C` (it is the
   agreed program, not transcript data).
   - **Completeness (honest acceptance).** If the committed transcript is the honest execution
     of `C` on the committed `(batch, W_t)`, verification accepts except with false-positive
     probability ≤ `α_FP`, a decreasing function of the tolerance margin `z = τ/s_h`.
   - **Soundness (detection).** If any checked matmul, or the weight-update identity, deviates
     from the honest execution of `C` by more than the floor `Φ` (in Freivalds discrepancy
     `‖D·r‖`), verification rejects except with probability ≤ `δ = N_checks · p₁ᵏ`; `δ` is
     driven below any target by `k`.
   - **Binding.** §3.12 (embedding recompute) binds the certified computation to the *actual*
     committed batch; §3.13 (weight-endpoint anchors) binds it to the *actual* base/deployed
     model — so acceptance certifies the real training transition, not a parallel fabrication.
   - **Limits (inherent).** Deviations with discrepancy ≤ `Φ` are not caught (magnitude floor;
     `Φ` a function of working precision). The protocol certifies *faithful execution of `C`*,
     not that `C` itself is benign — a malicious-but-honestly-executed `C` is out of scope.
   - **Setup-config knobs (NOT spec constants):** working precision, `k`, `z`/`α_FP`, target
     `δ`, grind budget `G`, and the resulting `Φ`. The spec's claim is stated as a function of
     these; a setup profile pins them (e.g. fp32 / `k=7` / `z=8` / `δ ≤ 2⁻⁴⁰` / single-lab `G`
     for the local prototype).
15. **Data/batch anchor = dataset commitment + agreed schedule, as a RUN-LEVEL anchor
    (user-decided; the symmetric partner of the weight-endpoint anchors §3.13).** The batch is
    now anchored to an agreed public dataset `D`, closing the asymmetry whereby weights were
    anchored to agreed external values (old checks 0/6) but the batch was only bound to the
    computation (old check 2). Adapted from the PoTD *proof-of-training-data* construction
    (arXiv:2307.00682, §4.3 "Fixing the Initialization and Data Order", and Appendix A
    combined-verification **check 2**), which makes the data order a checkable function of a
    dataset hash `s = H(H(d_1)∘…∘H(d_a)∘s_rand)` that drives both init `W_0 = G_r(s)` and order
    `Π = G_p(s)`; changing one record changes the whole order. **We adopt only the data-order
    half:** our `W_0` is already anchored to the agreed base model (check 0 — we post-train an
    existing model, not a certified-random init), so PoTD's seed→init half does not apply.
    - **(a) Full-schedule anchor, not membership-only (Q-BA1, user-accepted).** Bind *which*
      agreed records form batch `b_t`, in agreed order, per step — the true symmetric partner of
      the weight anchors — not merely that each trained record `∈ D`.
    - **(b) Separate run-level commitment `h_D`, NOT folded into the per-step Merkle root `h`
      (user's follow-up on Q-BA2, decided).** `h_D` = Merkle root over canonically-encoded
      records of `D`, computed once and published. Kept separate because: (i) `h` must stay
      scoped to a single step so the Fiat-Shamir challenges depend only on that step's transcript;
      (ii) folding adds no per-step soundness — the batch `b` is already a leaf of `h`, and `D` is
      a fixed run-level constant identical across steps; (iii) a separate Merkle `h_D` preserves
      memory-bounded verification (verifier holds only `h_D` + schedule; per-step record checks by
      authentication path, the same §4.2 tree-for-inclusion rationale). Exact analog of not
      folding `W_0`/`W_T` into every step's tree.
    - **(c) No `s_rand` nonce (Q-BA3, user-accepted).** PoTD folds `s_rand` in to stop a
      *prover-chosen* dataset from gaming the derived order/init. Our `D` is externally
      agreed/public, so the schedule `π` is a public deterministic function (published directly,
      or `π = G(h_D)`); the nonce buys nothing here and is dropped (lean). Reintroduce only if the
      prover is ever permitted to choose `D`.
    - **(d) Verifier-holds-`D` vs holds-only-`h_D`** is a config/logistics knob (§9), not a
      structural fork — same family as the weight-anchor logistics; both are memory-bounded.
    - **§8.3 part 2 narrowed accordingly:** the three axes from the earlier discussion —
      (1) *commit one batch, train on another* → caught by check 3 (was 2); (2) *is the committed
      batch the agreed data?* → **now guaranteed** (out-of-corpus injection and off-schedule use
      are caught by checks 1/4), previously *not guaranteed*; (3) *is the agreed dataset itself
      clean?* → still out of scope. So §8.3 part 2 shrinks to: a malicious-but-honestly-executed
      `C`, or honest training on an **agreed-but-corrupted** `D`, is accepted.

### Security-sizing rationale (captured so we don't re-derive it)

- Freivalds one-sided error `≤ 1/s` per vector; `s⁻ᵏ` with `k` vectors.
- Fiat-Shamir is non-interactive, so a cheating prover can **grind**: tweak any free bit
  of the committed transcript, re-hash, and re-roll all challenges. Security is therefore
  `≈ (#grind attempts G) · (pass prob)`, not a flat `s⁻ᵏ`.
- Budget: `k·b ≥ λ + log₂G + log₂M`, with `b = log₂s` bits/vector, `λ` = residual margin
  (≈40 per step; +~20 to cover a `2²⁰`-step run), `G` = grind budget, `M` = matmuls/step
  (`log₂M ≈ 15`). Numerator `≈ 120` bits ⇒ with `b≈61`, `k=2`.
- **`G` — RE-ESTIMATED (the earlier "`2⁶⁴` ≈ years of hashing" justification was wrong).**
  The security-relevant `G` is the number of grind *attempts*; total adversary work =
  `G · C_attempt`, so `G = Work_max / C_attempt`. Per-attempt cost `C_attempt` = re-hash a
  Merkle path (`~log₂(leaves)` compressions) + PRF-expand the challenges + **test the
  forgery** (`D·r` for the cheated matmul). A cost-optimizing adversary makes the cheat
  low-rank so the test is `O(n)`, giving `C_attempt ≈ 2¹²–2¹⁵` ops. The bare-hash
  wall-clock we quoted was the wrong cost model: `2⁶⁴` SHA-256 is ~a day on one mining ASIC
  (`~2⁴⁷` H/s) or ~30 ms on the Bitcoin network (`~2⁶⁹` H/s) — "years" holds only for pure
  hashing on a single GPU. Tiered estimate (`Work_max / 2¹³`, order-of-magnitude):
  - single research lab (1–few nodes, days–weeks, `~2⁶⁰–2⁶⁷` ops) → `G ≈ 2⁴⁷–2⁵⁴`;
  - large industrial cluster (thousands of GPUs, weeks–months, `~2⁷⁶–2⁸²` ops) → `G ≈ 2⁶³–2⁶⁹`;
  - nation-state / mining-scale (a year, `~2⁸⁶–2⁹⁰` ops) → `G ≈ 2⁷³–2⁷⁷`.
  So `2⁶⁴` ≈ the *large-industrial* tier, not a normal lab; high if the adversary is a single
  research group (then `~2⁵⁰` is apter). **`G` matters more under option (d)** than it did
  under τ=0: `log₂G` sets how small the per-vector pass probability must be, which sets how
  tight `τ'` must be, which trades directly against the false-positive rate. **Awaiting user's
  adversary tier to pin `log₂G`.** (Tight canonical serialization, Q8, is what enforces the
  compute bound — free nonce fields would let the adversary re-roll `r` without recomputing
  the cheat.) Threat model **[CONFIRMED, Q1]**: cheating *prover* (the trainer), no
  interactive/online advantage; only the compute tier remains to pin.
- Two failure modes to keep straight: **(A) random miss** — `D≠0` yet `Dr=0` because `r`
  hit `null(D)`; probability `≤ s⁻ᵏ`, magnitude-independent, controlled by `k,b`. **(B)
  tolerance floor** — the `‖Dr‖≤τ'` route admits any forgery under `τ'`. **Option (d)
  accepts (B)** with a *sized* floor (this reverses the earlier "quantized-exact removes it"
  reasoning, which no longer applies).

### Pure-float sizing — τ, b, k (single-lab, `log₂G = 52`)

- **Per-vector soundness is linear, not `1/s`.** For a rank-1 forgery `D = uvᵀ`, `D·r =
  u·(vᵀr)`; under Rademacher `r ∈ {−1,+1}` (float-valued, still the right choice — cheap,
  bounded range), `vᵀr` is mean-zero with std `‖v‖`, so a miss `‖D·r‖ ≤ τ` has probability
  `p₁ ≈ c·τ/‖D‖_F`, `c ≈ 0.8`. Per-vector soundness `b = −log₂ p₁ = log₂(‖D‖_F/τ) + ~0.7`.
- **τ rule:** `τ = z·s_h`, `s_h` = honest per-check `‖D·r‖` band, **measured on a few honest
  fp32 steps** (float rounding only — no quantization noise). `z ≈ 8` keeps honest rejects
  rare over the run's `~2⁴¹` component-checks. Same measure-the-band method at both scales;
  the constants fall out of the measurement. The final weight identity gets its own tighter
  band `τ_W` (§3.10), measured separately.
- **k budget:** need `p₁ᵏ · (M·T·G) ≤ 2^(−λ_margin)`, i.e. `k·b ≥ N` with
  `N = log₂G + log₂(M·T) + λ_margin ≈ 52 + ~35 + ~25 ≈ 112` bits. So `k·(log₂R − ~3) ≥ 112`,
  `R = ‖D‖_forge/s_h` = detection margin.
- **`R` set by the honest float floor = training precision** (dominant lever):
  - fp32 (`ε≈2⁻²⁴`, random-sign accumulation over the contraction) → honest band `~2⁻¹⁸–2⁻²⁰`
    relative, blatant forgery `R≈2¹⁸–2²⁰`, `b≈15–17` → **k = 7–8**.
  - bf16 (`ε≈2⁻⁸`) → `R≈2⁸`, `b≈5` → **k ≈ 22**, coarse floor (nothing below ~0.4% relative).
  - Subtler targets lower `R`, raise `k` (fp32, `2⁻⁶`-relative cheat → `k ≈ 10`).
- **Model behind the numbers (order-of-magnitude):** operand values `~O(1)`, random-sign
  accumulation over the contraction dim, rank-1 worst-case forgery. Revisit if these shift.
- **LOCKED (user-confirmed).** Pure float, **fp32** verified runs, **Rademacher challenges**
  (`r ∈ {−1,+1}`), **`z ≈ 8`**, detection target **order-1 (blatant) forgeries** →
  **`k = 7` (target) / `8` (fallback)**, **`τ = 8·s_h`**, `s_h` measured from honest fp32
  steps. `log₂G = 52`; budget `N ≈ 112` bits. Accepted tradeoffs: the magnitude floor (only
  blatant forgeries guaranteed caught) and occasional false positives from honest float
  rounding. **No quantization / prime / fixed-point.**
  - **Full-scale consequence (setup-stage):** `k = 7` requires fp32. Real full-scale bf16
    forces `k ≈ 22` and a coarser floor. The local fp32 prototype keeps `k = 7`; full-scale
    precision is a setup decision, not settled here.

### Interactive challenges — CONSIDERED and REJECTED (keep non-interactive)

Raised: replace the Fiat-Shamir derivation `r = Expand(PRF(h, ·))` with an **interactive**
protocol in which the verifier draws fresh challenge coins *after* the prover commits the
step transcript. Motivation: an interactive verifier's `r` is unpredictable to the prover at
commit time, so the prover cannot re-roll `r` by tweaking a free bit and re-hashing — the
grind is dead.

- **The effect is real but singular: it removes exactly the `log₂G` term** from the budget
  `k·b₀ ≥ λ + log₂T + log₂M + log₂G`. `b₀` (per-vector bits) is identical either way, so only
  one additive term drops. Accounting at the locked fp32 point: `N ≈ 52 + 35 + 25 = 112 → 60`,
  so `k = 7 → 4`; the bf16 full-scale case `~22 → ~12`. Both ≈ 2×, not an order of magnitude.
- **A `k ≈ 4` floor remains** even with zero grinding (`G = 1`): `log₂(M·T) + λ ≈ 60` bits
  still stands, so interactivity cannot push fp32 below ~4 vectors. And the win is
  **verifier-side only and partial** — the verifier's per-matmul cost has a `k`-independent
  floor (glue recompute + operand reconstruction from committed leaves), and prover/training
  compute is independent of `k`.
- **Why rejected (the cost dwarfs the ~3-vector saving):**
  1. **Loses the self-contained, publicly-auditable certificate — the whole point.** Non-
     interactivity is what makes the transcript+commitment a portable artifact anyone can check,
     any time, with the prover offline and no auditor present during training. Interactive
     convinces only the one live verifier: no post-hoc / archival audit, no new auditor, and a
     *recorded* interaction is **non-transferable** (prover+verifier could have scripted it), so
     it proves nothing to a third party. This guts the proof-of-training framing and the paper.
  2. **Forces a live, trusted, available verifier coupled into the training loop** — a round-trip
     per step over up-to-`2²⁰` steps, with the prover holding each step's transcript live until
     challenged; undercuts the memory-bounded stream-and-discard sequential verification (§3.6).
  3. **Reintroduces a trusted-fresh-randomness assumption**; if the prover can predict/bias the
     verifier's RNG, grinding returns *silently* (no `log₂G` term admitting it). Fiat-Shamir needs
     no beacon — `r` is a public, self-certifying function of `h`.
  4. **Bad trade vs. levers already in hand.** The Merkle root + canonical nonce-free
     serialization (Q8) already make each grind a real recompute-and-rehash, so `G` is a *chosen
     budget* (adversary tier), not a fixed tax. Dropping to a single-research-group tier or
     shaving the margin already gives `k = 6` **for free**, non-interactively. The compute knobs
     are the adversary tier `log₂G` and working precision — not interactivity.
- **Decision:** non-interactive (Fiat-Shamir, §3.1/§3.7/§3.8) **stays locked**. `log₂G` is the
  known, bounded price of non-interactivity and is worth paying for a portable, third-party-
  auditable proof. This subsection doubles as the "why non-interactive" argument for the paper.

---

## 4. WORKING ASSUMPTIONS (tentatively chosen, not yet finally reconfirmed)

These came from an initial (rejected-then-discussed) multiple-choice round; we built on
them but should reconfirm, especially given the move to quantized-exact.

- **W-A. Security goal = detect a wrong / substituted training data batch**
  (data-substitution / poisoning), à la the PoTS threat where the prover reports batch
  `d` but trained on poisoned `d̃`. **Reconsider:** quantized-exact (τ=0) actually gives
  near-*full* computational integrity for free (any ≥1-ULP deviation is caught), so the
  "batch only" framing may widen. Also note: to close the "fabricate a fully
  self-consistent transcript for `d` while having trained on `d̃`" loophole, the **final
  weight-update equality** must be a first-class accept condition (see Open Q).
  **[RESOLVED → §3.14.]** Generalized: the protocol certifies faithful execution of a
  *declared computation* `C` on committed inputs under a committed weight transition, up to the
  floor `Φ`, with detection probability set at setup. Data-substitution detection is a
  *consequence* (a substituted batch perturbs `C`'s committed outputs → caught above `Φ`),
  bound by §3.12/§3.13. The concrete numbers are config, not spec.
- **W-B. Verifier division of labor = "recompute glue, Freivalds the matmuls."** The
  verifier itself recomputes every non-matmul op and replaces each matmul with a
  Freivalds check on the prover's reported product; it never performs a full O(n³) matmul.
  **[CONFIRMED under (d) — §3.9.]**
- **W-C. Operands = "verifier recomputes A itself."** Prover reports matmul **outputs**
  (+ the gradient tensors needed to chain backprop), not all operands; the verifier
  reconstructs each input from the previously-verified value. **[CONFIRMED under (d) —
  §3.9: the verifier chains strictly from Merkle-committed values, `Â_l = glue(Ĉ_{l−1})`;
  it never maintains its own reference trajectory.]**

---

## 5. OPEN QUESTIONS — agenda for continued clarification (roughly prioritized)

> Ask these in small batches, with a recommendation each. **Q0 is the immediate next
> one** (it was raised and left unanswered).

- ~~**Q0 — Fiat-Shamir binding structure.**~~ **RESOLVED → §3.7/§3.8** (whole-step Merkle
  commitment, fully non-adaptive; verifier re-derives every `r`).
- ~~**Q1 — Grind budget `G` / threat model confirmation.**~~ **RESOLVED → §3 security-sizing**
  (cheating prover, `G≈2⁶⁴` robust to `2⁸⁰`, no online advantage; `k=2` stands).
- ~~**Q2 — Quantization scheme (LARGE).**~~ **DROPPED (user-corrected → §3.2).** There is no
  quantization anywhere: a `τ > 0` check runs directly on floats, so fixed-point format,
  scale/zero-point, per-tensor/per-channel, float→integer mapping, and the modulus prime
  (Mersenne-61 / Goldilocks / NTT) are all moot. Nothing to decide here.
- **Q2a — What the prover reports and what gets the exact check (NEXT).** Freivalds' exact
  (τ=0) check needs *integer* operands, so verification cannot quantize only the challenge
  vectors — it must also map each checked operand to a fixed-point image. Per matmul the
  prover then reports the fixed-point images of the operands and the exact integer product
  of those images (not a quantization of the float product), and the verifier Freivalds-
  checks that product exactly. Open fork on the weight-update binding (Q5), because the
  exact update equality would bind the *quantized* trajectory, which drifts from the float
  weights by quantization error:
  - **(a)** commit the quantized-update result as `W_{t+1}` — exact link end to end, but
    the committed weights are a second trajectory alongside the float model;
  - **(b)** commit the quantized image of the float `W_{t+1}` — single trajectory tied to
    the real model, but the update equality holds only approximately → reintroduces a
    tolerance;
  - **(c)** keep each matmul exact but bind the batch through forward/backward consistency
    plus an explicitly-approximate check only on the one cross-boundary update comparison.
  - **(d)** *(user's proposal — fourth option, breaks τ=0)* train fully in float — gradients,
    reported matmuls, updated weights all float — and quantize **only** to run the matmul
    check. Possible, but **not exact**: an honest quantized image of a float product does
    not equal the integer product of the quantized operands (rounding accumulated over the
    contraction dim, plus a scale mismatch before rescale), so `D = Â·B̂ − Ĉ ≠ 0` for honest
    data. The check must become `‖Â·(B̂·r) − Ĉ·r‖ ≤ τ'` with `τ' > 0` **forced**. This
    **reverses §3.2** (steps off quantized-exact onto the float+tolerance route that §3.2
    named as the alternative). Cost the user must price in: not just false positives (honest
    rejects, which they accept) but **false negatives** — any adversarial discrepancy under
    `τ'` passes, so the scheme gains a soundness/magnitude floor; and the random `r`
    amplifies the honest error, so `τ'` cannot be tiny, widening the adversary's hiding room.
    Kept: training is pure float (untouched), and the verifier side stays deterministic
    (fixed-point comparison). **Impossibility to note:** no construction reports float end to
    end *and* achieves τ=0; exactness requires the checked product to be an integer product.
    **SUPERSEDED (user-corrected) → pure float, §3.2/§3.3.** Option (d) still quantized *for
    the check*; the user then observed a `τ > 0` check needs no quantization at all — the
    challenge vectors and the `‖A(Br) − Cr‖ ≤ τ` test are done directly in float. The
    fixed-point lens is dropped entirely; only the float+tolerance route remains. The
    impossibility (float end-to-end ⇒ `τ > 0`, no τ=0) still holds and its floor is accepted.
- **Q3 — Non-matmul ops in exact integer arithmetic (HARD).** How to make softmax/exp,
  RMSNorm/LayerNorm (division, rsqrt), SwiGLU/activation, residual adds, embedding
  gather, and cross-entropy **bit-exact and agreed** between prover and verifier
  (shared fixed-point approximations? agreed polynomial/LUT approximations?). Nonlinear
  ops in exact fixed-point are the hardest part.
- ~~**Q4 — Matmul coverage.**~~ **RESOLVED → §3.11** (matmul ⇒ checked leaf, non-matmul ⇒
  recomputed glue; `QKᵀ`/`AV` checked; 3-per-weight and fwd+2-grads-per-weight-free-matmul).
  Spec stays architecture-independent; worked reference block in a separate `.md` (see the
  deliverable-structure note under the PRIME DIRECTIVE).
- **Q5 — Final weight-update equality = first-class accept condition, with its OWN band
  `τ_W` (RESOLVED → §3.10; Q5a+Q5b user-confirmed).** Accept iff all local Freivalds pass
  **and** the elementwise identity `‖Ŵ_{t+1} − (Ŵ_t − η·δ̂_W)‖ ≤ τ_W` holds. This is the
  aggregation anchor that closes the self-consistent-fabrication loophole.
  - **Q5a — RESOLVED.** `τ_W` is the elementwise-identity band, **tighter per entry than
    `τ'`** (no contraction, no Rademacher projection → honest residual is single-op
    fixed-point quantization). Calibrated **per weight tensor** from a few honest steps (same
    measure-the-band method as `s_h`), not one global norm. It is *not* a widened absorber of
    accumulated drift — accumulated drift lands as adversary freedom upstream, not as honest
    residual here.
  - **Q5b — RESOLVED.** `δ_W` is a first-class committed leaf, independently Freivalds-checked
    as `δ_W ≈ δ_Yᵀ·X` within `τ'` — so the identity is non-vacuous (else the adversary sets
    `δ̂_W = (Ŵ_t − Ŵ_{t+1})/η` for a zero residual). Division of labor: `τ_W` binds
    `W_{t+1} ↔ δ_W` tightly; the τ' chain binds `δ_W` back through `δ_Y, X` to the batch;
    coherent drift there is the separate λ line. This is also a constraint on **what the
    prover reports** (Q2a) — the weight-grad matmul is a committed, checked leaf like the
    other two.
- **Q6 — Backward-pass exactness.** Exact gradient formulas per op in fixed-point,
  especially through the nonlinearities of Q3.
- **Q7 — Optimizer specifics.** Learning rate `η`, weight decay?, grad clipping?, exact
  quantized form of the SGD update.
- **Q8 — Crypto primitives.** Commitment/hash (SHA-256 / BLAKE3?), PRF for expanding `r`
  to mod-p entries, domain-separation label scheme, **canonical serialization** (no
  free/nonce fields, to bound the grinding surface).
- ~~**Q9 — Data/batch representation & commitment**~~ **RESOLVED → §3.15 / `SPEC.md` §4.3.**
  Each batch is anchored to the agreed dataset via `h_D` (Merkle root over canonically-encoded
  records) and the agreed schedule `π`; the concrete token-id canonical encoding is config (Q8).
- **Q10 — Small-scale prototype model.** Tiny transformer vs plain MLP; relation to the
  repo's 135M SmolLM2. Note the repo is **CUDA-only** (`model_loader` hardcodes
  `.to("cuda")`, no CPU/MPS path) — decide whether the prototype needs GPU or a small CPU
  path. (Leans into stage 3, but op-coverage depends on it.)
- **Q11 — Tied embeddings / LM-head handling.**
- **Q12 — Step loop & memory bounds.** Confirm prover reports each step; verifier checks
  sequentially and discards prior step's data.
- **Q13 (later, evaluation).** Metrics: false-accept / false-reject rates, verify-vs-train
  cost ratio, detection of injected data substitution / hidden steps (cf. PoTS's hidden
  malicious steps experiment).

---

## 6. Repo context (for implementation later)

- `pots-2.0` is a fork of `pochenai/nano-llm-posttraining` — a minimal SFT/DPO/GRPO
  post-training tutorial; the long `README.md` is the primary artifact, `src/` scripts
  are the reproducible backing. Package imported as `src.<name>`.
- **Local-debug → cloud via env vars only** (135M SmolLM2 on 8GB → rented 24–48GB GPU):
  this is the natural template for our two run modes.
- Scripts execute their whole pipeline **at import time** (module-level), not via
  functions; env-var configured; load-or-train from `trainer_output/<name>/`.
- **CUDA required**, no CPU/MPS path today; no test suite; pyright assumed.
- Dependency pins in `pyproject.toml` are load-bearing (transformers `<5`, peft `<0.18`,
  vllm `<0.17`) — don't bump without reproducing the cited reason.

---

## 7. Status

- Stage 1 (algorithm clarification) and Stage 2 (spec) **COMPLETE — `SPEC.md` approved 2026-09-23**. Locked: §3 — FS binding §3.7/§3.8,
  confirmed threat model, **pure float + tolerance Freivalds** in §3.2 (`τ > 0`, no
  quantization/prime/fixed-point; accepting a magnitude floor), **`k = 7`/8** in §3.3,
  §3.9 (Model 1), §3.10 (two thresholds `τ`/`τ_W`), §3.11 (matmul coverage), §3.12
  (batch→embedding recompute), §3.13 (weight-endpoint anchors), **§3.15 (data/batch anchor)**. W-B/W-C confirmed; W-A
  still open. Q0, Q1, Q4, Q5 resolved, **Q9 resolved** (data/batch anchor); **Q2 dropped** (no
  quantization), Q2a superseded.
- **Sizing LOCKED (pure float):** `log₂G = 52`, **`k = 7` target / `8` fallback**,
  `τ = 8·s_h` (s_h measured from honest fp32 steps), fp32 verified runs, Rademacher
  challenges, order-1 detection target, budget `N ≈ 112` bits. Full detail in the "Pure-float
  sizing" block in §3. **No quantization / prime / fixed-point** — the check `‖A(Br) − Cr‖ ≤ τ`
  runs directly in float.
- **Interactive challenges CONSIDERED and REJECTED → §3 "Interactive challenges" subsection.**
  Interactivity would strip only the `log₂G` term (`k = 7 → 4` fp32, `~22 → ~12` bf16; a `k ≈ 4`
  floor from `log₂(M·T)+λ` remains), a verifier-side-only ~2× saving. Rejected: it destroys the
  self-contained, transferable, after-the-fact-auditable certificate (the whole PoTS/paper
  point), forces a live trusted verifier in the training loop, and reintroduces a trusted-RNG
  assumption. Cheaper compute levers (adversary tier `log₂G` → `k = 6` free; precision) stay
  non-interactive. Fiat-Shamir (§3.1/§3.7/§3.8) stays locked; subsection doubles as the paper's
  "why non-interactive" argument.
- **Operating model RESOLVED → §3.9 (Model 1, confirmed):** verifier certifies internal
  consistency of the committed transcript, chaining every operand from Merkle-committed
  values, so honest `τ'` does not compound with depth (false-positive side settled). W-B/W-C
  confirmed under (d). Q3 is thereby *easier* — non-matmul glue need not be bit-exact
  integers; the verifier recomputes it in float within the band.
- **Two-threshold principle RESOLVED → §3.10 (user-decided):** local per-matmul `τ'` vs a
  separately-calibrated final-weight band `τ_W`; coherent-drift accumulation gets its own λ
  line, distinct from `log₂M`.
- **Final weight check RESOLVED → Q5/§3.10 (Q5a+Q5b confirmed):** `τ_W` is the tight
  per-weight-tensor elementwise-identity band; `δ_W` is a committed, independently
  Freivalds-checked leaf. Coherent-drift term still to be folded into the λ budget.
- **Q4 matmul coverage RESOLVED → §3.11 (user-confirmed):** matmul ⇒ checked leaf,
  non-matmul ⇒ recomputed glue; `QKᵀ`/`AV` are checked leaves; 3-per-weight and
  fwd+2-grads-per-weight-free-matmul completeness. **Deliverable (user-decided):** spec stays
  architecture-independent; a worked reference block (SmolLM2-shaped gated decoder layer) goes
  in a separate `.md` (implementation ground + future paper appendix) — see the
  deliverable-structure note under the PRIME DIRECTIVE.
- **Committed-transcript question (leaf set; synthesizes Q2a/§3.7/Q9 — NOT the §5 "Q6
  backward exactness").** Proposed per-step leaf set: `{batch, W_t, every checked-matmul
  output in canonical (layer,op,rep) order (fwd activations, input-grads, weight-grads δ_W,
  weight-free QKᵀ/AV outputs and their operand-grads), W_{t+1}}`, glue intermediates
  uncommitted (verifier recomputes, Model 1).
  - **(iii) CONFIRMED (user):** weight-free `QKᵀ`/`AV` are real computed matmuls, Freivalds is
    weight-agnostic (needs only the 2 operands + claimed product), so both their outputs and
    their 2 operand-grads are committed leaves. Bookkeeping note: forward + 2 operand-grads,
    no weight-grad.
  - **(i) leaf-set completeness + (ii) dL/dlogits-as-glue RESOLVED (user-agreed).** The leaf
    set above is complete; `dL/dlogits` is glue the verifier recomputes from committed logits +
    labels, not a leaf. Sufficiency is provided by the two accept conditions now locked as
    §3.12 (embedding recompute — binds batch → computation) and §3.13 (weight-endpoint anchors —
    binds computation → real model). Residual magnitude floor accepted (§3.2).
- **W-A DRAFTED → §3.14 (general/parameterized; user-directed):** the claim is stated over a
  *declared computation* `C` and abstract detection/false-positive probabilities, with the
  concrete numbers (precision, `k`, `z`, `δ`, `G`, `Φ`) left to setup config, not spec
  constants. Pending final wording approval.
- **GATE to Stage 2.** W-A is the last *structural* algorithm decision. Remaining items are
  requirement-level, to be stated *generally* in the spec (specific primitive = config): Q8
  (a collision-resistant hash + PRF + canonical, nonce-free serialization), Q9 (batch committed
  under a canonical token-id encoding), Q7 (SGD update form; weight decay / clipping as config),
  Q11 (tied-embedding handling). Flag any of these as a real open choice before drafting;
  otherwise, on the user's go-ahead, proceed to write the architecture-independent spec `.md`.
- **Stage 2 draft in `docs/verification/SPEC.md`** — formal, standalone, architecture-
  independent, parameterized. Sections: 1 Setting/roles, 2 Notation, 3 Declared computation
  `C` (+3.1 Coverage), 4 Commitment (4.1 Transcript, 4.2 Merkle), 5 Challenge generation,
  6 Verification protocol ("Combined Verification"), 7 Verifier cost and error locality,
  8 Security (8.1 Completeness, 8.2 Soundness, 8.3 Scope), 9 Configuration and calibration.
  Revisions applied per user: (i) all math in readable Unicode-in-backticks, no LaTeX;
  (ii) run-start anchors are **check 0** (base weights) and **check 1** (dataset anchor);
  per-step checks are **2–7** (2 commitment, 3 batch binding, 4 batch anchor, 5 matmul, 6 update
  identity, 7 chaining); `W_T` is final anchor **check 8** and accept is **check 9**;
  (iii) the cost / no-accumulation paragraph lifted into its own §7 (Security→8, Config→9);
  (iv) §8.3 part 2 first broadened to state the data limitation, then **re-narrowed by the
  data/batch anchor** (§3.15; see below);
  (v) §4.3 dataset commitment `h_D` added, and the matmul discrepancy renamed `Δ_m` so `D`
  denotes the dataset.
- **§8.3 scope, stated then narrowed by the anchor (user-agreed, applied):** the protocol
  certifies faithful execution of `C` on the *agreed dataset*; it does not certify `C` is
  well-chosen, nor that the *agreed dataset itself* is benign. Malicious-but-honestly-executed
  `C`, or honest training on an **agreed-but-corrupted** dataset, is accepted. The three axes,
  updated for §3.15: (1) *commit one batch, train on another* → **caught** by check 3 (batch
  binding: committed batch = actually-used batch); (2) *is the committed batch the agreed/intended
  data?* → **now guaranteed** by the data/batch anchor §3.15 (checks 1/4 bind each batch to the
  agreed dataset + schedule); (3) *is the agreed data itself clean?* → **never in scope**.
- **RESOLVED — data/batch anchor → §3.15 and `SPEC.md` (user-approved).** Added the symmetric
  partner of the weight anchors: a run-level **dataset anchor** (check 1 — commitment `h_D` over
  canonical records + agreed schedule `π`) and a per-step **batch anchor** (check 4 — committed
  `b_t = D[π(t)]`). Full-schedule (not membership-only); `h_D` kept **separate** from the per-step
  Merkle root `h`; **no `s_rand`** (public agreed `D`). Adapted from PoTD (arXiv:2307.00682 §4.3 /
  Appendix A check 2), the data-order half only (our `W_0` is already anchored by check 0, so the
  seed→init half does not apply). Spec updated: §1, §2, §4.3, §6 (renumbered), §7, §8, §9.
- **Stage 2 SPEC APPROVED (user, 2026-09-23)** — `SPEC.md` (with the §3.15 data/batch anchor)
  is accepted. A late fix applied on approval: the §6 verifier checks were reformatted to
  explicit **"Check N"** labels (0–9, three phase groups) because the previous split
  ordered-lists renumbered in rendering and showed a duplicate "2" (and hid checks 8–9); the
  numbers are unchanged, so every `check N` cross-reference (§6–§8) stays valid. **RESUME HERE:**
  next artifact is the separate **reference-block `.md`** (worked single-decoder-layer SmolLM2
  instantiation — implementation ground + future paper appendix, per the PRIME DIRECTIVE
  deliverable note), then **Stage 3** (setup & implementation clarification; treat the
  prover–verifier algorithm as a black box, mirror the PoTS setup except the top-K layer
  separation, plan test-scale local first). Still-general requirement items to fold into setup:
  Q7 (SGD update form; weight-decay/clipping as config), Q8 (hash + PRF + canonical nonce-free
  serialization), Q10 (prototype model; repo is CUDA-only), Q11 (tied embeddings). No reference
  block, implementation plan, or code written yet.
- **Stage 3 (setup / implementation) STARTED in a parallel session → see new §8.** It treats
  the §3 algorithm as a **black box**. Locked so far: one bespoke SGD loop (no TRL);
  unmodified `from_pretrained` model; small = CPU/fp32/`k=7`, full = GPU/bf16-mixed/`k=22`;
  `TorchDispatchMode` matmul capture; one code path with all divergence in env-var config;
  MLP smoke test first. Remaining setup vague spots enumerated in §8.B.

---

## 8. SETUP / IMPLEMENTATION (stage 3) — DECISIONS & OPEN QUESTIONS

> This section is the **experimental scaffolding**, developed in a session that treats the
> §3 verification **algorithm as a black box** ("assume any prover-verifier algorithm"). It
> is kept separate from §3/§5 (algorithm) to avoid multi-session clobbering. Guiding
> constraints (user): **mirror the PoTS setup except the top-K freeze/retrain-tail
> separation**; **test-scale local run first, full-scale later**; **scale-invariance** — one
> code path, all test↔full divergence pushed into env-var config ("separate what must be
> separated, and only that").

### 8.A LOCKED (setup decisions)

1. **Training harness = one bespoke minimal SGD loop; TRL is nowhere in the verified path.**
   The repo's training paths all run through TRL `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` with
   `bf16=True`, AdamW, inside a single `Trainer.train()` call, on a hardcoded `.to("cuda")` —
   which exposes no per-matmul access and cannot do plain-SGD/fp32. The verified run is a
   hand-written forward + `loss.backward()` + plain-SGD loop we own. The **same loop drives
   both scales**, differing only by config (the repo's local-debug→cloud-via-env-vars template).
2. **Existing, unmodified model via `from_pretrained` (reuse-over-reinvention — same principle
   as using autograd over a hand-written backward).** No from-scratch model (avoids
   subtle-but-crucial reimplementation bugs). Small scale = SmolLM2-135M; full scale = a
   PoTS-class 0.5–1.5B instruct model (which one is S2 below). Whole-model training (no
   freezing — consistent with §3.4).
3. **Device / precision / `k` (config-diverged, no code fork):** small = **CPU, fp32, `k = 7`**;
   full = **GPU, bf16 compute + fp32 master (mixed precision, option (i)), `k = 22`**. The
   committed/reported weight at full scale is the **fp32 master**. Precision is parameterized as
   a pair `(MASTER_DTYPE, COMPUTE_DTYPE)`: small = `(fp32, fp32)` collapses mixed precision to a
   no-op; full = `(fp32, bf16)`. **bf16, not fp16** — bf16 needs no loss scaler so the loop is
   byte-identical to the fp32 path; fp16 would inject a `GradScaler` branch (a code fork) and
   shift the floor/`k`. (`k` values trace to §3.3's precision→`R`→`b` sizing.)
4. **Matmul capture = `TorchDispatchMode` (device-agnostic passthrough observer).** A
   `__torch_dispatch__` mode wraps forward + `loss.backward()` and intercepts every
   `aten::mm`/`bmm`/`addmm` — the 3 per weight matrix and the weight-free `QKᵀ`/`AV` — with
   concrete operands + output, on an **unmodified** model, **identically on CPU and CUDA**
   (dispatch sits below both autograd and the device backend). It is a pure passthrough (calls
   the real op, returns the real result), so training numerics are unchanged — the "capture must
   not change training" invariant holds. Weight / `δ_W` reporting via `named_parameters()` /
   `param.grad`. Backward matmuls are captured because `.backward()` runs inside the mode.
   - **Attention fusion (the one conditional knob):** §3.11 makes `QKᵀ`/`AV` checked leaves, so
     they must be captured → run **eager attention** (`attn_implementation="eager"`) so a fused
     SDPA/flash kernel does not swallow them into one opaque op. This is a **load-time flag, not
     a code fork**. Cost is **GPU-only** (forfeits flash → O(L²) attention memory + slower);
     **quality-neutral** (eager ≡ fused to within ULPs). This is the single place §3.11's
     coverage rule imposes a full-scale speed cost.
5. **Scale-invariance = one code path; all divergence is env-var config.** Knobs: `DEVICE`,
   `MODEL_ID`, `(MASTER_DTYPE, COMPUTE_DTYPE)`, `K_VECTORS`, batch/seq/steps,
   `attn_implementation`. `τ`/`τ_W` are **not** config values — they are calibrated at runtime
   from a few honest steps (`s_h`; §3.3/§3.10), so they self-adjust to scale/precision. **No
   forced code forks.**
6. **Smoke test first:** validate the full capture → transcript → verify plumbing on a plain
   **MLP** (matmul count `3L`, §2; trivially hand-checkable) through the **identical capture
   path**, then swap in the transformer.
7. **Full-scale reporting cost (noted, not a blocker):** "report weights + matmul I/O during the
   run" moves GB/step off-device; sequential per-step discard (§3.6) bounds *memory* but not
   *bandwidth*, and reading tensors on GPU can force host syncs. This is the dominant full-scale
   cost and folds into the verify-vs-train ratio (eval, §5 Q13). Mitigations: capture device-side
   references + hash in a background/streaming pass; enable capture only on verified steps.

This **resolves §5 Q10** (prototype model / CUDA-only concern): small scale runs on CPU via the
bespoke loop, not the repo's CUDA-hardcoded `model_loader`; MLP-then-transformer staging chosen.

### 8.B OPEN (setup vague spots — agenda, same small-batch / prose / recommendation process)

- **S1 — Prototype data.** How much Alpaca (+ AdvBench) at test scale; the BadMagic trigger
  construction; seq len (PoTS: 128) and batch. *(Next batch on resume.)*
- **S2 — Full-scale model choice.** Which PoTS-class model (Llama-3.2-1B / Falcon-3-1B /
  Qwen-2.5-0.5B / 1.5B); tokenizer ↔ data alignment. (Overlaps §5 Q10/Q11.)
- **S3 — Prover↔verifier process topology.** One process / two passes vs two programs over an
  on-disk transcript; where transcripts live (per-step dir, gitignored); the non-crypto storage
  layout/format. (Distinct from §5 Q8's canonical crypto serialization.)
- **S4 — Determinism knobs.** Seeds, `torch.use_deterministic_algorithms`, TF32 off (GPU),
  thread count — needed for prover/verifier float agreement (§3.9 recomputes glue in float
  within the band). bf16 nondeterminism at full scale to examine.
- **S5 — What the verified run is.** Number of steps to verify; the real backdoor-injection task
  vs generic steps at test scale; relation to PoTS's hidden-malicious-steps experiment.
- **S6 — Fault-injection / eval harness.** How we demonstrate the verifier rejects a corrupted
  step (flipped matmul output, substituted batch, hidden step). Overlaps §5 Q13.
- **S7 — Repo integration.** New `src/verification/` package vs standalone; whether to follow the
  repo's import-time-execution + env-var convention or use normal functions/CLI; README
  (primary-artifact) obligation for a research subsystem.
- **S8 — SGD hyperparameter *values*.** η, weight decay?, grad clip?, batch/seq — the *values*
  are setup (the update *form* is §5 Q7 / algorithm). PoTS used lr 5e-5 with AdamW; plain SGD
  likely needs a different lr.

---

## 9. REFERENCE BLOCK (stage-2 companion file) — OPEN QUESTIONS

> Companion to `SPEC.md`: a worked instantiation of the architecture-independent spec on a
> SmolLM2-shaped gated decoder layer (implementation ground + future paper appendix). Nothing
> written yet; clarifying first, same small-batch / prose / recommendation process. Same formal,
> Unicode-in-backticks register as `SPEC.md`.

- **R1 — Scope of the worked example.** One decoder layer in isolation, or one full step
  (embedding → L decoder layers → final norm → LM-head → loss → update) with one layer written
  out in detail? *Recommendation:* full step with one layer expanded — checks 3 (embedding),
  6 (update, incl. tied weight) and the LM-head only appear at the step boundary. Optional
  second, smaller block: the plain-MLP smoke test (§8.A.6).
- **R2 — Tied embeddings (= §5 Q11).** SmolLM2-135M ties `W_E` (gather = glue) to the LM-head
  (matmul). *Recommendation:* one learnable weight, one committed leaf; `G_W` in check 6 =
  LM-head weight-grad product (committed, checked) + embedding scatter-add grad (glue,
  recomputed); state tying generally in the spec's terms as "a weight used by several ops".
- **R3 — Granularity of batched / multi-head products.** A `bmm` over (batch × heads), and GQA
  sharing of K/V heads: is each head-slice its own matmul `m` with its own challenges, or is
  the batched product one matmul? *Recommendation:* one matmul `m` per head-slice per sequence
  (a block-diagonal product is not one "product of two matrices"); GQA's K/V repeat is glue.
