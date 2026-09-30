# Algorithm Decisions

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file records every settled decision about the verification algorithm, with its
reasoning and the alternatives that were rejected. The approved result is
`VERIFICATION_PROTOCOL_SPEC.md`. Read this file when a task needs to know *why* the
protocol is built as it is, not to resume work (`STATUS.md` does that).

**Numbering.** Sections keep their numbers from the former unified worklog (§3, §4, §5,
§9), so a reference such as "§3.13", "Q5a", or "R2" in any design doc resolves here.
References to §8 point to `DECISIONS_SETUP.md`, and references to §10 point to
`EVALUATION_TASKS.md`.

**Adding a decision.** When an algorithm question closes, add it here with its reasoning
and rejected alternatives, then remove it from its task list. No algorithm questions are
open: the algorithm stage is closed.

## 3. LOCKED DECISIONS

1. **Verification core = Freivalds per matmul + Fiat-Shamir challenge generation**,
   non-interactive, **identical algorithm at both scales**. (Securing the local case
   specifically is explicitly *not* of interest — we just run full-scale params locally,
   where they are cheap.) *Scope, per setup P12 (2026-09-30): "full-scale params" means the
   same algorithm and sizing formula, not full-scale arguments to it. Each scale sizes `k` against
   its own precision, `T` and `M`, so test scale runs `k = 7`.*
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
14. **Security claim (W-A), stated GENERALLY and PARAMETERICALLY (user-directed; approved
   with the spec, 2026-09-23).** The spec states the guarantee over a *declared
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
      agreed/public, so the schedule `π` is published directly as an explicit index sequence per
      step (user, 2026-09-24: the `π = G(h_D)` alternative is removed from the spec — one
      procedure, and derivation buys nothing when `π` is agreed); the nonce is dropped (lean). Reintroduce only if the
      prover is ever permitted to choose `D`.
    - **(d) One procedure: authentication path only (user, 2026-09-24; supersedes the earlier
      "holds-`D` vs holds-only-`h_D` config knob").** Check 4 always verifies each batch record
      by its Merkle authentication path into `h_D` at leaf index `π(t)_j`; the direct-comparison
      branch is removed from the spec (check 4, prover step 3, §4.3, §7, §9). Why: the spec must
      be a single unambiguous recipe, and only the path procedure keeps verifier state constant
      as §1 claims — holding `D` is `O(n)` state. Direct comparison was not *less sound* (it
      needs no collision assumption); it was a second procedure the spec did not need, since a
      verifier that has `D` can run the path check anyway.
    - **(e) Plain-language summary (why both checks are needed, not just one).** Check 4 and
      check 1 test two different claims, not two strengths of one claim. Check 4 asks, every
      step: *"is the batch used right now really part of the agreed dataset, in the agreed
      order?"* — re-checked per step because *which batch* changes every step. Check 1 asks,
      once: *"is the dataset we agreed on before the run really the dataset we agreed on?"* —
      checked once because *what `D` is* is a single fact fixed before step 0, not something
      with a different answer at each step. Neither makes sense without the other: per-step-only
      would prove every batch matches *some* hash, even a fabricated one, since nothing pins
      that hash to the real `D`; run-level-only would prove the *right* dataset was agreed on but
      never check that the batches actually trained on came from it.
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
  tight `τ'` must be, which trades directly against the false-positive rate. **Tier pinned:
  single research lab, `log₂G = 52` (see "Pure-float sizing").** (Tight canonical serialization, Q8, is what enforces the
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
- **LOCKED (user-confirmed); challenge distribution superseded by Q14 → `Uniform(−1,1)`.** Pure
  float, **fp32** verified runs, ~~**Rademacher challenges** (`r ∈ {−1,+1}`)~~ **`r ~
  Uniform(−1,1)`, component-wise and independent (§3 Q14 — closes a Littlewood–Offord
  anti-concentration gap in Rademacher; `k` unaffected)**, **`z ≈ 8`**, detection target
  **order-1 (blatant) forgeries** →
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

## 4. Working assumptions (all resolved)

These came from an initial multiple-choice round and were built on, then each was
confirmed or generalized, as its note says.

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

## 5. Clarification questions (all closed)

Every question raised during algorithm clarification, with its resolution. Q8 and Q13 were
not algorithm questions: they moved to `SETUP_TASKS.md` and `EVALUATION_TASKS.md`.

- ~~**Q0 — Fiat-Shamir binding structure.**~~ **RESOLVED → §3.7/§3.8** (whole-step Merkle
  commitment, fully non-adaptive; verifier re-derives every `r`).
- ~~**Q1 — Grind budget `G` / threat model confirmation.**~~ **RESOLVED → §3 security-sizing**
  (cheating prover, `G≈2⁶⁴` robust to `2⁸⁰`, no online advantage; `k=2` stands).
- ~~**Q2 — Quantization scheme (LARGE).**~~ **DROPPED (user-corrected → §3.2).** There is no
  quantization anywhere: a `τ > 0` check runs directly on floats, so fixed-point format,
  scale/zero-point, per-tensor/per-channel, float→integer mapping, and the modulus prime
  (Mersenne-61 / Goldilocks / NTT) are all moot. Nothing to decide here.
- ~~**Q2a — What the prover reports and what gets the exact check.**~~ **SUPERSEDED → §3.2
  (closed 2026-09-24).** Kept for the reasoning that led from quantized-exact to pure float. Freivalds' exact
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
- ~~**Q3 — Non-matmul ops in exact integer arithmetic (HARD).**~~ **CLOSED as superseded
  (2026-09-24).** Was: how to make softmax/exp, RMSNorm/LayerNorm (division, rsqrt),
  SwiGLU/activation, residual adds, embedding gather, and cross-entropy **bit-exact and
  agreed** between prover and verifier in fixed-point. Why closed: the question existed only
  for exact integer arithmetic. Under pure float (§3.2) and Model 1 (§3.9), the verifier
  recomputes glue in float and each check tolerates honest rounding within its band, so
  glue need not be bit-exact. What remains is a setup concern, float agreement between
  prover and verifier runs, and it moved into setup item S4 (`SETUP_TASKS.md`).
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
- ~~**Q6 — Backward-pass exactness.**~~ **CLOSED as superseded (2026-09-24).** Was: exact
  gradient formulas per op in fixed-point, through the nonlinearities of Q3. Why closed: the
  same reason as Q3. The backward pass is float autograd, and the verifier checks backward
  matmuls within the band like forward ones (§3.9, §3.11).
- ~~**Q7 — Optimizer specifics.**~~ **CLOSED (2026-09-24).** Was: learning rate `η`, weight
  decay, gradient clipping, and the exact quantized form of the SGD update. Why closed: the
  update *form* is settled (plain SGD, §3.5; the elementwise identity with its own band
  `τ_W`, §3.10), and the spec treats weight decay and clipping as part of the declared
  computation. There is no quantized form under pure float. The *values* are setup item S8.
- **Q8 — Crypto primitives. MOVED to setup (`SETUP_TASKS.md`, item S9, 2026-09-24).** The
  spec states only the requirements (a collision-resistant hash, a PRF, a canonical,
  nonce-free serialization); choosing the concrete primitives is configuration.
- ~~**Q9 — Data/batch representation & commitment**~~ **RESOLVED → §3.15 / `VERIFICATION_PROTOCOL_SPEC.md` §4.3.**
  Each batch is anchored to the agreed dataset via `h_D` (Merkle root over canonically-encoded
  records) and the agreed schedule `π`; the concrete token-id canonical encoding is config (Q8).
- ~~**Q10 — Small-scale prototype model.**~~ **RESOLVED → `DECISIONS_SETUP.md` §8.A.2,
  §8.A.3, §8.A.6.** Was: tiny transformer or plain MLP, and whether the prototype needs a
  GPU given the CUDA-only `model_loader`. Resolution: SmolLM2-135M on CPU through the
  bespoke loop, not `model_loader`, with a plain-MLP smoke test first.
- ~~**Q11 — Tied embeddings / LM-head handling.**~~ **RESOLVED → §9 R2** (one tied tensor; `G_W` = committed LM-head weight-grad product + recomputed embedding scatter-add; spec §3.1/check 6).
- ~~**Q12 — Step loop & memory bounds.**~~ **CLOSED (2026-09-24).** Was: confirm the prover
  reports each step and the verifier checks sequentially, discarding the prior step's data.
  Settled by §3.6 (sequential per-step verification) and, for test-scale memory, by
  `DECISIONS_SETUP.md` §8.A.8.
- **Q13 — Evaluation metrics. MOVED to `EVALUATION_TASKS.md`.**
- **Q14 — Challenge-vector distribution: Rademacher is provably broken against a concentrated
  forgery; switch to `Uniform(−1,1)` (reopens and RESOLVES the §3 Rademacher lock; user
  signed off 2026-09-24).** The empirical constant re-measurement is the calibration task in
  `SETUP_TASKS.md`. Raised by the user to see
  whether a different distribution could **lower `k`** (and hence verifier compute) below the
  locked `k = 7` — not to chase extra security for its own sake — with the explicit condition
  that a switch is only worth it if sampling stays cheap and introduces no new bit-exactness
  risk. Worked through in three passes; the first two undersold the problem.
  - **Pass 1 (approximation gap).** For a rank-1 forgery `D = uvᵀ`, the check collapses to one
    scalar test on `vᵀr`. Rademacher (`r ∈ {±1}`) only makes `vᵀr` *approximately* Gaussian via
    the CLT — the `c ≈ 0.8` constant behind `p₁ ≈ c·τ/‖D‖_F` (§3, Pure-float sizing) is that
    approximation, and `v = (1,−1,0,…,0)` breaks it: `vᵀr ∈ {−2,0,2}` with probabilities
    `1/4,1/2,1/4`, a 50% miss rate for any `τ < 2`, not the assumed `→0` behaviour.
  - **Pass 2 (quantified, but the wrong worst case).** This is a Littlewood–Offord
    anti-concentration effect: for `r ∈ {±1}ⁿ_eff` (all-equal-magnitude, the extremal shape),
    the point-mass at zero is `≈√(2/(π·n_eff))` (Erdős 1945), and — critically — **this floor
    does not shrink as `τ→0`**; zero is an exactly-representable lattice point (equal-magnitude
    float32 cancellation is exact, no rounding needed) and any window around it contains that
    point regardless of width. Scored against this repo's actual matmul widths
    (`VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md`, taking Freivalds' cheaper side `r`-length each
    time): `d_h=64→3.3 bits`, `d=576→4.9 bits`, `d_f=1536→5.6 bits`, `n_v=49152→8.1 bits`,
    naively suggesting `k` up to `~34` (worst case `d_h=64`) to hold the 112-bit budget.
  - **Pass 3 (the actual worst case — corrects Pass 2).** The prover isn't forced to spread the
    forged discrepancy across the full check width `n`; a rank-1 `D=uvᵀ` lets them concentrate
    `v` on just **2** of the `n` components (one matched `±c` pair, rest exactly correct — a
    surgical two-channel edit, not a contrived corner case) *regardless of the ambient matmul
    dimension*, hitting the `n_eff=2` extremal point mass of `1/2` every time. So the true
    floor is **`≈1 bit/vector, flat, independent of which matmul is checked`** — not the
    64-to-49152-dependent table above. Holding the 112-bit budget rigorously under Rademacher
    would need `k ≈ 112` — on the order of the entire target security budget spent one vector at
    a time. There is no economical `k`-bump fix; Rademacher is broken by this construction, full
    stop.
  - **Why continuous challenges have no floor, at any `n`.** The floor is purely a lattice
    artifact: discrete sums land on a grid, and probability mass sitting on a reachable grid
    point (zero) doesn't shrink with `τ`. A sum of independent *continuous* variables is always
    itself continuous — convolving densities never creates an atom — so its density near zero is
    finite and smooth for **any** `v`, at **any** `n_eff`, including the worst case. Verified
    exactly, not asymptotically: `r₁,r₂ ~ Uniform(−1,1)` gives `r₁−r₂` an exact triangular
    density peaking at 0, `Pr[|r₁−r₂|≤τ] = τ−τ²/4 ≈ τ` — smooth, linear, no floor, at `n_eff=2`,
    Rademacher's single worst case.
  - **Recommendation: `r ~ Uniform(−1,1)` component-wise, not Gaussian and not variance-matched
    to Rademacher.**
    - Closes the floor completely (same mechanism as Gaussian: no atoms, any `v`, any `n_eff`).
    - Sampling stays exact and trivial: `r = 2u−1`, `u = bits/2ᵐ` — pure dyadic affine scaling of
      raw PRF bits, zero rounding at any step. No `erfinv`/Box–Muller, no transcendental
      functions, none of Gaussian's cross-platform bit-exactness risk (the reason Gaussian was
      rejected in the prior pass of this question).
    - Don't variance-match Rademacher (`a=√3`): scale is a pure calibration constant — `‖D·r‖`,
      `s_h`, and `τ=z·s_h` all scale together, so soundness depends only on `τ/‖D‖_F`, invariant
      under rescaling `a`, *provided `s_h` is re-measured at that scale* (already the plan). `√3`
      buys nothing and costs an irrational multiplication for no reason; `a=1` is simpler.
    - **Must stay symmetric (mean zero).** An asymmetric range (e.g. `Uniform(0,1)`) gives
      `E[r]≠0`, a fixed public direction the prover can null out for free by choosing `D` with
      zero row-sums — wasting part of `r`'s entropy on a component that isn't actually random
      from the prover's perspective. Symmetry forces the prover to defeat the whole draw.
  - **Net effect vs. the user's original bar.** Uniform sampling is no harder to implement than
    Rademacher (same IEEE-754-guaranteed primitives, no new risk) and it restores the *generic*
    bound the design always assumed — recovering `k ≈ 7` — instead of the `k ≈ 112` Rademacher
    would actually need if defended rigorously. This is a straightforward win on the user's own
    terms, not a security-for-its-own-sake upgrade.
  - **Constant re-derived (analytically and by Monte Carlo) — `k = 7` is unaffected.** In the
    generic (spread-`v`) case, `p₁ ≈ c·τ/‖Δ‖` with `c = √(2/π) ≈ 0.80` for Rademacher versus
    `c = √(6/π) ≈ 1.38` for `Uniform(−1,1)` — a factor `√3` apart, matched by simulation
    (dot-product Monte Carlo, `N` up to `3×10⁵`) at every architectural width tested
    (`n = 64, 576, 1536, 49152`). This looks like a soundness loss, but it is not: the factor
    is exactly `1/σ_r` (`Uniform(−1,1)` has `Var = 1/3` against Rademacher's `Var = 1`), and the
    same `σ_r` multiplies the honest residual band `s_h` (Section 9), which is measured with the
    *same* challenge distribution. `τ = z·s_h` inherits that factor, `p₁ ≈ c·τ/‖Δ‖` inherits it a
    second time in `c`, and the two cancel — the `k`-budget of §8.2, expressed through the
    measured `s_h` as the design already requires, is insensitive to `σ_r` (hence to `a` in
    `Uniform(−a,a)`, confirming the earlier scale-invariance point) and, in the generic case, to
    the choice between Rademacher and Uniform altogether. So `k = 7` carries over; the routine
    re-measurement of `s_h` under the new challenges at the setup stage is a calibration step,
    not expected to move `k`.
  - **Bonus finding: Rademacher's floor also degrades the *generic* case at realistic widths, not
    only the adversarial one.** At `n = 64` with an evenly-spread `v`, simulation gives
    `c_rademacher ≈ 4.98` — six times the CLT value `0.80` — because `τ` (deliberately small,
    per §3's pure-float sizing) falls below the `±1` lattice spacing at this width, so the check
    is already hitting the same discreteness floor Q14 identifies for the adversarial case, on
    ordinary honest-shaped deviations. `c_uniform` stays at `≈1.39` at the same `n`, matching its
    large-`n` limit — continuous challenges show no such degradation at any width.
  - **Concentrated worst case, confirmed numerically down to `τ = 10⁻⁶`.** For
    `v = (1,−1,0,…,0) ∈ ℝ⁶⁴`: `Pr[|vᵀr| ≤ τ]` under Rademacher is `≈0.50` at every `τ` tested
    from `1.9` down to `10⁻⁶` (no improvement from tightening `τ`, as predicted); under
    `Uniform(−1,1)` it tracks the exact closed form `τ − τ²/4` at every `τ` tested (e.g.
    `p/τ ≈ 1.01` at `τ=10⁻³`, `≈0.98` at `τ=0.1`), i.e. shrinks linearly with no floor.
  - **Resolution: `r ∈ {±1}` (§3) is replaced by `r ~ Uniform(−1,1)`, component-wise and
    independent, at `k = 7` unchanged.** `VERIFICATION_PROTOCOL_SPEC.md` §2, §5, §8.2, and §9
    have been updated accordingly (§8.2 now states the continuity requirement and the
    anti-concentration argument against finite-support challenges directly, referencing the
    classical `{−1,+1}` choice by name). Only remaining step is the routine `s_h`/`z`
    re-measurement from a few honest fp32 steps under the new challenges, at the setup stage
    (Section 9 of the spec), to confirm the numeric `k=7` in practice — not expected to change it
    per the cancellation argument above.

---

## 9. Reference block (stage-2 companion file) — R1–R5 resolved, approved (user, 2026-09-24)

> Companion to `VERIFICATION_PROTOCOL_SPEC.md`: a worked instantiation of the architecture-independent spec on a
> SmolLM2-shaped gated decoder layer (implementation ground + future paper appendix). Written as
> `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` after R1–R5 below were settled. Same formal,
> Unicode-in-backticks register as `VERIFICATION_PROTOCOL_SPEC.md`.

- **R1 — RESOLVED (user): full step, one layer expanded.** Was: *Scope of the worked example.* One decoder layer in isolation, or one full step
  (embedding → L decoder layers → final norm → LM-head → loss → update) with one layer written
  out in detail? *Recommendation:* full step with one layer expanded — checks 3 (embedding),
  6 (update, incl. tied weight) and the LM-head only appear at the step boundary. Optional
  second, smaller block: the plain-MLP smoke test (§8.A.6).
- **R2 — RESOLVED (user-agreed) → spec §3.1 + check 6 edited:** embedding gather/scatter = glue (selection-operand products); `G_W` = sum of all contributions (committed weight-grad products + recomputed glue terms), covering tied weights. Analysis given: it works (`X = OneHot(b)·W_E`, `G_E = OneHot(b)ᵀ·δ_X`), but a one-hot operand makes the direct recompute a gather/scatter linear in the output — cheaper than one Freivalds vector (which needs `W_E·r`, `O(V·d)`), exact on the forward, and adds no `V×d` leaf. Original: *Tied embeddings (= §5 Q11).* SmolLM2-135M ties `W_E` (gather = glue) to the LM-head
  (matmul). *Recommendation:* one learnable weight, one committed leaf; `G_W` in check 6 =
  LM-head weight-grad product (committed, checked) + embedding scatter-add grad (glue,
  recomputed); state tying generally in the spec's terms as "a weight used by several ops".
- **R3 — RESOLVED (user-agreed) → `VERIFICATION_PROTOCOL_SPEC.md` §2 edited** (batched product = one matmul per member, own challenges; operand replication is glue). Extra rationale: one challenge over the whole block-diagonal product would calibrate `τ` to the noise of all ~(sequences × heads) blocks, diluting a one-block forgery by ~√(#blocks) (~5 bits/vector); per-slice costs only ~+3 bits in `log₂M` (≈18 at SmolLM2-135M, 16k-token batch), which can tip `k` 7 → the locked fallback 8. Was: *Granularity of batched / multi-head products.* A `bmm` over (batch × heads), and GQA
  sharing of K/V heads: is each head-slice its own matmul `m` with its own challenges, or is
  the batched product one matmul? *Recommendation:* one matmul `m` per head-slice per sequence
  (a block-diagonal product is not one "product of two matrices"); GQA's K/V repeat is glue.
- **R4 — RESOLVED (user-agreed) → spec check 3 reworded** (operands downstream of the embedding derived from committed `b`; check 5 binds it; §8.1 residual list now checks 5–6). Check 3 compares the recomputed
  embedding to "the committed activation that is the first matmul's input", but that activation
  is glue output (`RMSNorm(E[b])`), which §4.1 excludes from `L`. Proposed rewording: the verifier
  derives every operand downstream of the embedding from the committed batch `b` itself, so the
  check-5 tests of the consuming matmuls bind `b`.
- **R5 — FIXED (spec slip found while tabulating matmuls; user had authorized spec edits).** §5
  sized each challenge `r(m,j)` by the *contracted* dimension, but check 5 computes `P_m·r`, so
  `r` must have the product's column count. §2 now defines the **width** `c_m` (columns of
  `P_m`) and §5 uses it. Cost is unchanged by the choice of side: right- and left-multiplied
  checks both cost `pq + qs + ps` for `P = A·B`, `A ∈ ℝ^{p×q}`, `B ∈ ℝ^{q×s}`. (§2 of this
  worklog, "vector length = contracted dim, pick the cheaper side", carried the same slip.)
- **Reference block APPROVED (user, 2026-09-24) → `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md`.**
  SmolLM2-135M shapes (confirmed from its `config.json`): full step, one layer expanded; forward
  and backward op-by-op with matmul/glue class; matmul inventory (shape, contracted dim, width);
  canonical transcript order; checks 3/5/6 instantiated (tied `W_E`, glue-only norm scales);
  size/cost at `n = 128`, `n_s = 128`; plain-MLP degenerate instance. Findings surfaced there:
  - `M = L·(21 + 6·n_s·n_h) + 3 = 207,993`, `log₂M ≈ 17.7` (sizing assumed ~15) → budget
    `N ≈ 115`, so `k = 7` needs `b₀ ≥ 16.4`, else the fallback `k = 8`.
  - Committed products ≈ `8.2·10⁹` values ≈ **33 GB/step at fp32** (logits `0.8·10⁹`, attention
    `2.3·10⁹`) — sharpens the §8.A.7 reporting-cost note.
  - Verifier matmul saving ≈ 277× per challenge, **≈ 40× at `k = 7`** (≈ 35× at `k = 8`), capped
    by small contracted dims (attention scores contract over `d_h = 64` → 32×). Glue excluded.
  - The plain-MLP smoke test has `M = 3L − 1`, not `3L`: no input-gradient for the first layer,
    since the input is data.
