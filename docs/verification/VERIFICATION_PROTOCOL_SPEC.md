# A Protocol for Verifying Training Steps by Randomized Matrix-Product Checks

## Abstract

We specify a non-interactive protocol by which a prover, who claims to have carried out one step of stochastic gradient descent on a machine-learning model, produces a transcript that a verifier can check for computational integrity at cost quadratic, rather than cubic, in the layer dimensions. Each matrix product performed during the step is checked with a randomized product test; the test's challenges are derived from a commitment to the whole step transcript, so the proof is non-interactive and the prover cannot adapt its report to the challenges. Steps are verified sequentially and independently, so verifier memory does not grow with the length of the run. The run's weight endpoints and training data are anchored to agreed external values, so acceptance certifies the actual training transition on the agreed data rather than a merely self-consistent transcript. The protocol is stated for an arbitrary declared computation and an arbitrary target error probability; concrete numeric choices are deferred to a configuration profile (Section 9).


## 1. Setting and roles

A **prover** performs a sequence of training steps on a model and, for each step, emits a transcript. A **verifier** checks each transcript against a fixed, publicly agreed description of one training step, against agreed initial and final model weights, and against an agreed training dataset. The verifier accepts a run only if every step verifies, the endpoints match the agreed model, and each batch matches the agreed dataset (Section 6).

The protocol has the following properties:

- **Non-interactive.** A transcript is checked without further communication with the prover; the verifier derives the challenges itself (Section 5).
- **Sequential and memory-bounded.** Steps are checked one at a time and independently; the verifier retains only a constant amount of state between steps (Sections 6 and 7).
- **Scale-invariant.** The same protocol is used at every model and data scale; scale enters only through the configuration profile (Section 9).

The protocol is parameterized throughout by a **declared computation** (Section 3), a **working precision**, a **repetition count** `k`, and **tolerances** `τ`, `κ_max` and `τ_W`. Their values are fixed by a configuration profile and are not part of the protocol proper.


## 2. Notation

Vectors and matrices are over the reals at a fixed floating-point working precision. For a matrix `M`, `‖M‖` denotes its Frobenius norm and `Mᵀ` its transpose. A **matrix multiplication** ("matmul") is any operation computing a product `A·B` of two matrices. The **contracted dimension** of a matmul is the shared inner dimension of its two operands, and its **width** is the number of columns of its product. A **batched product**, which computes a family of independent products in one operation (for example, one per sequence and attention head), is one matmul per member of the family, each checked with its own challenges. Replicating an operand across members, as when several attention heads share one key or value head, is glue.

`H` is a collision-resistant hash function and `PRF` a pseudo-random function; `Expand(·; d)` maps a `PRF` output to a vector of length `d` whose entries are drawn independently from a fixed **challenge distribution**: a continuous distribution of mean zero and known variance, fixed by the configuration profile (Section 9). `⟨·⟩` denotes a fixed, injective, domain-separated encoding of labels.

A **dataset** `D = (d_1, …, d_n)` is the agreed, public sequence of training records. A **record** is one training example in the form the declared computation consumes: any preprocessing of raw data, such as tokenization, precedes the agreement on `D`, so the records themselves are the preprocessed examples. A **schedule** `π` maps each step index `t` to the sequence of record indices forming that step's batch, so the honest batch at step `t` is `b_t = D[π(t)]`. The dataset commitment `h_D` is defined in Section 4.3.


## 3. The declared computation

One training step is a fixed, public map

`C : (b, W_t) ↦ (P, W_{t+1})`,

where `b` is the training batch, `W_t` the tuple of model weights at step entry, `W_{t+1}` the weights at step exit, and `P = (P_1, …, P_M)` the tuple of all matrix products computed during the step. The verifier knows `C`; it is the agreed program, not part of the transcript.

`C` is composed of two disjoint classes of operation:

1. **Matmuls**, the products `P_1, …, P_M`; and
2. **Glue**, every other operation — the assembly of the batch's records into model input, elementwise maps, normalizations, softmax and other reductions, the input embedding, the loss, and the parameter update.

The batch `b` enters `C` as the ordered sequence of its records. Assembling them into model input, for example by padding sequences to a common length and stacking them, is the first glue operation of `C`.

### 3.1 Coverage

Every matrix product in `C` is a matmul and is checked (Section 6); every operation that is not a matrix product is glue and is recomputed (Section 6). For each **learnable weight** `W` that enters a matrix product, the step computes up to three matmuls involving it, and each one it computes is covered: the forward product, the input-gradient product, and the weight-gradient product. The input-gradient product is absent when the product's input is not a function of any learnable weight, as when it is data from the batch, because the step does not compute that gradient. A learnable weight that enters no matrix product, such as a normalization scale, has no matmul; its gradient is glue. For each **weight-free bilinear operation** — a product of two activations, such as an attention score or attention-value product — two classes of matmul occur and are covered: the forward product and the two operand-gradient products. No matmul is exempt.

A product one of whose operands is a **selection matrix**, with a single unit entry per row or column, is a row gather or a scatter-add rather than a dense product, and is glue. The input embedding and its weight gradient are of this kind. Recomputing such an operation directly costs time linear in its output, less than a single randomized check of the product, and is exact in the forward direction.
A product whose contracted dimension is 1, an **outer product**, is glue by the same argument:
each entry is a single multiplication, so direct recomputation costs time linear in its output
and is exact.

A learnable weight may enter `C` through several operations, as when the input embedding and the output projection share one weight matrix. Its **gradient** `G_W` is then the sum of every contribution: a weight-gradient product for each matmul in which `W` appears, and a recomputed glue term for each glue operation in which it appears.


## 4. Commitment

### 4.1 Transcript

The **step transcript** is the ordered leaf sequence

`L = ( b, W_t, P_1, …, P_M, W_{t+1} )`,

with the products `P_1, …, P_M` in a fixed canonical order. Glue outputs are **not** included. Each leaf is encoded under a **canonical serialization**: a deterministic, injective byte encoding with no optional, reserved, or nonce fields, so that a given mathematical object has exactly one admissible encoding.

### 4.2 Merkle commitment

The prover commits to `L` with a Merkle tree under `H`: each leaf is hashed, sibling hashes are concatenated and hashed pairwise up the tree, and the root

`h = MerkleRoot_H(L)`

is the **step commitment**. By collision resistance of `H`, `h` binds every leaf: no transcript `L' ≠ L` admits the same root except with negligible probability. The tree supports checking any single leaf against `h` from its authentication path alone, which is what permits the memory-bounded verification of Sections 6 and 7.

### 4.3 Dataset commitment and schedule

The agreed dataset `D` is committed once, for the whole run, by a Merkle tree under `H` over its canonically-encoded records:

`h_D = MerkleRoot_H( ⟨d_1⟩, …, ⟨d_n⟩ )`.

This commitment is **run-level and published**: an agreed external value, computed once, exactly as the agreed base and final weights are. It is deliberately **separate from the per-step commitment `h`** of Section 4.2. Separating them (i) scopes each `h` to a single step, so the challenges of Section 5 depend only on that step's transcript; (ii) adds nothing to per-step soundness by conflation, since the batch `b` is already a leaf of `h` and `D` is a run-level constant identical across steps; and (iii) preserves memory-bounded verification — the verifier retains only `h_D` and the schedule between steps, and checks each step's records by their authentication paths into `h_D`.

The schedule `π` is public and agreed; in an honest run the batch at step `t` is `b_t = D[π(t)]`. It is published directly, as an explicit sequence of record indices per step, alongside `h_D`. Proof-of-training-data schemes instead derive the order from a dataset hash and a nonce, so that a prover who chooses its own dataset cannot also choose a favourable order; here `D` and `π` are both externally agreed rather than prover-chosen, so no derivation is needed.


## 5. Challenge generation

For each matmul `m ∈ {1, …, M}` with width `c_m`, and each repetition `j ∈ {1, …, k}`, the **challenge vector** is

`r(m,j) = Expand( PRF(h, ⟨m, j⟩); c_m )` (Section 2), of length `c_m`.

Each challenge is a deterministic function of the step commitment `h`. Because `h` binds the entire transcript (Section 4.2), the challenges are bound to the complete report and are **non-adaptive**: the prover fixes the transcript before any challenge is determined, and cannot alter one leaf without redetermining every challenge. The verifier derives the challenges itself and never accepts challenge vectors from the prover.


## 6. The verification protocol

The protocol is stated below as a single procedure with a prover phase and a verifier phase. The verifier is given `C`, the agreed base weights `W_0`, and the agreed final weights `W_T` of the run.

**Protocol (Combined Verification).**

*Prover, for each step `t = 0, …, T−1`:*

1. Execute `C(b, W_t)` at the working precision, with batch `b = D[π(t)]` selected by the schedule, recording every matmul output `P_1, …, P_M` and the updated weights `W_{t+1}`.
2. Form the transcript `L` (Section 4.1) and compute the commitment `h` (Section 4.2).
3. Transmit `L`, `h`, and, for each record of the batch `b`, its authentication path into `h_D`.

*Verifier.* Checks 0–1 are performed once at the start of the run, checks 2–7 for each step `t`, and checks 8–9 once at the end.

*At the start of the run:*

- **Check 0 — Base anchor.** Require the committed entry weights `W_0` of the first step to equal the agreed base weights; reject otherwise. An invalid initialization is caught before any step is processed.
- **Check 1 — Dataset anchor.** Compute the dataset commitment `h_D` (Section 4.3) over the agreed dataset `D` and require it to equal the agreed, published commitment; adopt the agreed schedule `π`. The verifier then retains only `h_D` and `π` between steps, so this adds constant state.

*For each step `t`:*

- **Check 2 — Commitment.** Recompute the Merkle root of the received `L` and require it to equal `h`. This fixes the challenges of Section 5.
- **Check 3 — Batch binding.** Derive every matmul operand that depends on the input embedding from the committed batch `b` itself, by recomputing the assembly of its records, the embedding, and the intervening glue, and never from a value supplied by the prover. The tests of check 5 on the matmuls that consume these operands then bind `b` to the computation.
- **Check 4 — Batch anchor.** Let `π(t) = (i_1, …, i_B)`. For each position `j = 1, …, B`, take the `j`-th record of the committed batch `b` and its received authentication path, recompute the Merkle root from the record's encoding, the path, and the leaf index `i_j`, and require the result to equal `h_D`; reject if any record fails or if `b` does not have exactly `B` records. This binds the committed batch, record by record and in order, to the agreed records `D[π(t)]`, as the base and final anchors bind the committed weights to the agreed model.
- **Check 5 — Matmul checks.** For each matmul `m = 1, …, M`: reconstruct its operands `A_m, B_m` by recomputing the intervening glue from previously committed leaves, and form the **operand-magnitude bound**

  `ν_m = ‖ |A_m|·(|B_m|·1) ‖`,

  where `|·|` is the entrywise absolute value and `1` is the all-ones vector. Reject if

  `ν_m > κ_max · ‖ |P_m|·1 ‖`.

  Otherwise, form the matmul's **honest relative error**

  `e_m = √2·ε_in + √(q_m)·ε_acc`,

  where `q_m` is the contracted dimension of matmul `m`, `ε_in` is the unit roundoff of the operand format and `ε_acc` that of the accumulator; and for each `j = 1, …, k` reject if

  `‖ A_m·(B_m·r(m,j)) − P_m·r(m,j) ‖ > τ · σ_r · e_m · ‖P_m‖_F`,

  where `σ_r` is the standard deviation of a single challenge entry (Section 2).

  The second test measures the residual against the **committed product itself**, in units of the rounding error that a product of that contracted dimension admits at the working precision, so the tolerance `τ` is dimensionless and independent of the matmul's shape, magnitude and precision, and one value of it serves every matmul of `C`. The two terms of `e_m` scale differently and both are needed: operand rounding contributes `√2·ε_in` and does not grow with `q_m`, because it perturbs every term of the sum alike; accumulation contributes `√(q_m)·ε_acc`, growing as the square root of the contraction because the rounding errors of successive additions carry independent signs. A **deeper** contraction therefore admits a **wider** honest band, and the matmul with the largest `q_m` is the one against which `k` must be sized. Stating `e_m` as a relative error is what makes it insensitive to the sign structure of the sum: if the terms share a sign the running sums reach the full magnitude of the result and each rounding is correspondingly large, while if the signs are mixed both the running sums and the result are smaller, and the ratio is the same in either regime.

  The first test bounds the **cancellation factor** `ν_m / ‖ |P_m|·1 ‖`, which is at least 1 by the triangle inequality and which states how far the committed product falls below the magnitude of the terms summed to form it. Its role is to guard the validity of `e_m`, not to normalize the residual: under near-exact cancellation the product approaches zero while the rounding errors do not, so the relative error is unbounded and the second test would be applied against a band that the execution has widened for itself. Both tests are written as products rather than ratios, so a vanishing `‖P_m‖_F` or `‖ |P_m|·1 ‖` is well defined rather than a division by zero. `‖P_m‖_F` costs one pass over the committed product, which may be accumulated inside the pass that already reads and hashes it for check 2; `ν_m` costs two matrix-vector products, evaluated once per matmul rather than once per challenge, since it is a scale rather than a per-challenge quantity.

- **Check 6 — Update identity.** For each learnable weight `W`, form the residual `R = W_{t+1} − (W_t − η·G_W)` and reject if, for any entry `i`,

  `|R_i| > τ_W · ε_W · ( |W_{t,i}| + |η·G_{W,i}| )`,

  where `G_W` is the gradient of `W` (Section 3.1): the sum of its committed weight-gradient products and of its recomputed glue contributions, and `ε_W` is the unit roundoff of the format in which weights are held and updated. The step size `η` and the form of the update are given by `C`.

  The test is **elementwise**, and each entry is measured against the rounding error that one update of that entry admits. A norm over the whole tensor would bound only the total deviation, which a prover could concentrate on a few chosen entries: on a tensor of `n` entries, one entry could then absorb `√n` times its own honest rounding. An elementwise bound leaves no entry more than its own rounding, whether the deviation is spread or concentrated. Scaling by `|W_{t,i}| + |η·G_{W,i}|` keeps the bound relative, so an entry near zero is held to its own small rounding rather than to the tensor's typical magnitude. As in check 5, the test is written as a product, so an entry with `W_{t,i} = G_{W,i} = 0` has a bound of zero, which an honest update meets exactly.
- **Check 7 — Chaining.** Require `W_t` to equal the committed `W_{t+1}` of step `t−1` (the same leaf), for `t ≥ 1`.

*At the end of the run:*

- **Check 8 — Final anchor.** Require the last committed `W_T` to equal the agreed final weights; reject otherwise.
- **Check 9 — Accept.** Accept the run iff the base anchor (check 0), the dataset anchor (check 1), every per-step check 2–7, and the final anchor (check 8) hold.


## 7. Verifier cost and error locality

In check 5 the verifier evaluates only matrix–vector products, at cost `O(d²)` per matmul in place of the `O(d³)` of recomputing `P_m`. In checks 3 and 5 the verifier recomputes glue but never a matmul. The batch anchor (check 4) costs one authentication path, `O(log n)` hashes, per record of the batch, and the dataset anchor (check 1) is a single run-level computation; neither grows the verifier's per-step state. Because every reconstructed operand is derived from committed leaves rather than from a value the verifier carries forward, the tolerance of one check does not accumulate into the next: each check is a local, independent test against the committed transcript.


## 8. Security

Let `λ` denote the target security margin in bits. Fix a step and let `Δ_m = P_m − A_m·B_m` be the discrepancy of matmul `m` under the honest operands reconstructed in check 5.

### 8.1 Completeness

If the prover executes `C` honestly at the working precision, then every residual tested in checks 5 and 6 is a floating-point rounding quantity. Writing `s_h` for the honest band of the normalized matmul residual of check 5 (Section 9), if `τ ≥ z·s_h`, `κ_max` is set above the honest cancellation factor, and `τ_W` is set above the honest update-identity residual of every entry in the units of check 6, the verifier accepts the step except with a false-rejection probability `α` that decreases monotonically in the margin `z`. Normalizing the residual by `σ_r·e_m·‖P_m‖_F` is what allows a single `τ` to hold this rate across matmuls whose operands differ in magnitude, shape and contracted dimension: the normalized residual is close to 1 for an honest matmul at any shape and any precision, whereas an unnormalized tolerance would be set by the noisiest matmul of `C` and leave every quieter one with slack it does not need. The anchors (checks 0, 1, and 8), the batch anchor (check 4), and chaining (check 7) are exact equality tests, which an honest transcript passes with certainty.

### 8.2 Soundness

Suppose the committed transcript deviates from the honest execution of `C` on the committed `(b, W_t)`: some matmul has `‖Δ_m‖ > Φ_m`, or some entry of the update identity exceeds its check-6 bound. Write

`τ_m = τ · σ_r · e_m · ‖P_m‖_F`

for the **effective band** that check 5 applies at matmul `m`, a quantity the verifier computes from the committed product and the public shape of `C` rather than a configured constant. For a single challenge, the deviating matmul passes only if `‖Δ_m·r‖ ≤ τ_m`; for a challenge distribution as required by Section 2, this misdetection has probability

`p_1 ≈ c · τ_m / (σ_r · ‖Δ_m‖_F)`,  with  `c = Θ(1)`,

so each challenge contributes `b_0 = log₂(1/p_1)` bits, meaning `p_1 = 2^(−b_0)`. Writing the deviation relative to the product it corrupts as `‖Δ_m‖_F = f·‖P_m‖_F`, the scale `σ_r` and the magnitude `‖P_m‖_F` both cancel and

`b_0 = log₂( f / (τ · e_m) ) + log₂(1/c)`.

A deviation is therefore measured against the product it is hidden in, and `p_1` depends on the **relative** size of the deviation, not on its absolute size. Because `b_0` contains nothing measured from the run other than `τ`, the repetition count is computable in advance of it rather than fitted after the fact; and because `e_m` grows with `q_m`, `b_0` is smallest at the matmul with the deepest contraction, which is the one that binds.

**The challenge distribution must be continuous.** A finite-support distribution — the two-valued `{−1,+1}` distribution traditionally used for this test among them — admits a deviation for which the bound above fails outright. Choosing `Δ_m` supported on as few as two coordinates, of matched magnitude, makes the corresponding combination of challenge entries land on an exactly representable value with a probability bounded below independently of `τ`: an anti-concentration effect that shrinking `τ` cannot suppress, and that depends only on how many coordinates the deviation spans, not on its magnitude. A continuous distribution admits no such deviation, because a sum of independent continuously-distributed entries is itself continuously distributed: for every nonzero `Δ_m`, however its magnitude is distributed across coordinates, the challenge's projection has a bounded density at every point, so its probability of landing within `τ_m` of any fixed value is `O(τ_m)` and vanishes as `τ_m → 0`. The constant `c` is otherwise insensitive to the particular continuous distribution chosen or its scale, since both `c` and the honest residual band `s_h` of Section 9 are measured against the same challenge distribution and their scale dependence cancels.

A prover exploiting non-interactivity may **grind**: alter the transcript, redetermine the challenges, and retry, at the cost of a Merkle-path rehash per attempt; let `G` bound the number of such attempts. Over `M` matmuls, `T` steps, and `G` grinding attempts, a union bound gives an acceptance probability for a deviating run of at most

`T · M · G · (p_1)^k`.

Choosing the repetition count `k` as the least integer with

`k · b_0 ≥ λ + log₂T + log₂M + log₂G`

bounds this probability by `2^(−λ)`. Inverting the same relation at the chosen `k` gives the **detection floor** `Φ_m`, the least deviation magnitude at matmul `m` for which the budget is met:

`Φ_m = f_achieved · ‖P_m‖_F`,  with  `f_achieved = c · τ · e_m · 2^(N/k)`  and  `N = λ + log₂T + log₂M + log₂G`.

The floor is therefore relative and per-matmul: what the protocol guarantees to detect is a deviation that is large compared with the product it corrupts, which is the only sense in which a single floor can be stated across matmuls whose entries differ by orders of magnitude. It depends on the working precision and the contraction depth through `e_m`, on the tolerance through `τ`, and on the threat model through `N` — where, because `k` divides `N`, the grinding term `log₂G` alone widens the undetectable region by a factor `2^(log₂G/k)` over what an interactive verifier would achieve at the same `τ`. Reporting `f_achieved`, a result, is more informative than reporting the `f` used to select `k`, a decision.

A committed batch that is not the agreed data of the schedule, or endpoint weights that do not match the agreed model, is rejected deterministically by the anchors (checks 0, 1, 4, and 8); the randomized bound above governs only the matmul checks and the update identity.

### 8.3 Scope

Checks 0 and 8 bind the certified weights to the agreed model at the endpoints, check 7 chains the steps between them, check 3 binds each committed batch to the computation that consumes it, and check 4, together with the run-start dataset anchor (check 1), binds each committed batch to the agreed training dataset and its schedule. Acceptance therefore certifies the real training transition on the agreed data, and not a self-consistent transcript of a different computation or of a substituted dataset. Two limitations are inherent:

1. A deviation with `‖Δ_m‖ ≤ Φ_m` is not detected (the detection floor), where `Φ_m` is a fraction of `‖P_m‖_F` fixed by the precision, the tolerance and the bit budget.
2. The protocol certifies faithful execution of `C` on the agreed dataset; it does not certify that `C` is well-chosen, nor that the agreed dataset is itself benign. A malicious-but-honestly-executed computation, or honest training on an agreed-but-corrupted dataset, is accepted. Training on data outside the agreed dataset, or on the agreed data in a different per-step schedule, is by contrast detected (checks 1 and 4).


## 9. Configuration and calibration

The protocol is instantiated by a **configuration profile** fixing the following, none of which is part of the protocol proper:

- **Working precision**, whose unit roundoffs `ε_in` (operands) and `ε_acc` (accumulator) enter check 5 through `e_m` and thus set the floors `Φ_m`. The two are distinguished because a configuration may hold operands in a narrower format than the accumulator, in which case the operand term of `e_m` dominates and the floor is set almost entirely by it. A third unit roundoff, `ε_W`, is that of the format in which weights are held and updated, and it enters check 6. The verifier performs the arithmetic of checks 5 and 6 at the working precision too, so its own rounding is part of the honest residual and is absorbed by the calibrated bands rather than modelled in `e_m`.
- **Tolerances** `τ`, `κ_max` and `τ_W`. These are calibrated empirically over a small number of honest steps at the working precision, then held fixed for the remainder of the run. The calibration steps are computed by the verifier, or by a party it trusts, from the public inputs `W_0`, the agreed dataset, `π` and `C`, and are never taken from the prover's transcript, since bands fitted on the prover's own steps would let it widen them. Refitting them on the steps under judgment would let a prover drag them along and is excluded. `τ` is dimensionless: `s_h` is the honest band of the normalized residual of check 5, and `τ = z·s_h` for a margin `z` that fixes the false-rejection rate `α`. Because false rejection is a union bound over every component check of the run, `s_h` is taken as the **largest class-wise root-mean-square** of the honest normalized residual, where a class is a set of matmuls occupying the same role in `C`: the root mean square within a class because the normalized residual concentrates, and the largest across classes because the noisiest class governs the rate. `κ_max` is calibrated the same way against the honest cancellation factor. `τ_W` is dimensionless and set per weight tensor as `τ_W = max(τ_W⁰, 2·ρ_max)`. The analytic floor `τ_W⁰ = 4` bounds the difference between any two honest implementations of the update: each computes `W_t − η·G_W` with an error of at most `2·ε_W·(|W_t| + |η·G_W|)` per entry, whether it rounds once (a fused multiply-add) or twice. `ρ_max` is the largest honest normalized entry residual `|R_i| / (ε_W·(|W_{t,i}| + |η·G_{W,i}|))` observed for that tensor in calibration, and the factor 2 is its margin. The calibrated term matters only for a tensor whose gradient the verifier recomputes as glue in a different floating-point order than the prover. For every other tensor the verifier reproduces the prover's arithmetic, the measured residual may be exactly zero, and the floor governs. The floor keeps an exact zero from becoming an exact-equality test that would reject an honest step computed by a different but equally valid implementation.
- **Repetition count** `k`, the least integer meeting the soundness budget of Section 8.2 for the assumed grinding bound `G` and target margin `λ`, evaluated at the detection target `f` and at the largest contracted dimension of `C`. Unlike the tolerances, `k` is not calibrated: `b_0` is analytic in `f`, `τ`, `e_m` and `c`, so `k` is fixed before the run. After calibration, `k` is recomputed from the same budget with the measured `τ`. If the result exceeds the configured `k`, the configured `k` is raised before any judged step. Because calibration precedes the verified run, `k` is still fixed before the prover's first judged step.
- **Cryptographic primitives**: the concrete hash `H`, the `PRF`, the challenge distribution realizing `Expand` (Section 2) — a continuous, mean-zero distribution of known variance, sampled from the `PRF` output by operations exact and identical across prover and verifier — and the canonical serialization of Section 4.1.
- **Dataset anchor**: the dataset commitment `h_D` and the published schedule `π`.

A **run mode** is a configuration profile together with a model and data scale. Because the protocol is scale-invariant (Section 1), a small-scale profile and a full-scale profile execute the identical procedure of Section 6 and differ only in these values.
