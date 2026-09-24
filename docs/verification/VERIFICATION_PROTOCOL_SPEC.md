# A Protocol for Verifying Training Steps by Randomized Matrix-Product Checks

## Abstract

We specify a non-interactive protocol by which a prover, who claims to have carried out
one step of stochastic gradient descent on a machine-learning model, produces a
transcript that a verifier can check for computational integrity at cost quadratic,
rather than cubic, in the layer dimensions. Each matrix product performed during the step
is checked with a randomized product test; the test's challenges are derived from a
commitment to the whole step transcript, so the proof is non-interactive and the prover
cannot adapt its report to the challenges. Steps are verified sequentially and
independently, so verifier memory does not grow with the length of the run. The run's weight
endpoints and training data are anchored to agreed external values, so acceptance certifies the
actual training transition on the agreed data rather than a merely self-consistent transcript.
The protocol is stated for an arbitrary declared computation and an arbitrary target error
probability; concrete numeric choices are deferred to a configuration profile (Section 9).


## 1. Setting and roles

A **prover** performs a sequence of training steps on a model and, for each step, emits a
transcript. A **verifier** checks each transcript against a fixed, publicly agreed
description of one training step, against agreed initial and final model weights, and against
an agreed training dataset. The verifier accepts a run only if every step verifies, the
endpoints match the agreed model, and each batch matches the agreed dataset (Section 6).

The protocol has the following properties:

- **Non-interactive.** A transcript is checked without further communication with the
  prover; the verifier derives the challenges itself (Section 5).
- **Sequential and memory-bounded.** Steps are checked one at a time and independently;
  the verifier retains only a constant amount of state between steps (Sections 6 and 7).
- **Scale-invariant.** The same protocol is used at every model and data scale; scale
  enters only through the configuration profile (Section 9).

The protocol is parameterized throughout by a **declared computation** (Section 3), a
**working precision**, a **repetition count** `k`, and **tolerances** `τ` and `τ_W`. Their
values are fixed by a configuration profile and are not part of the protocol proper.


## 2. Notation

Vectors and matrices are over the reals at a fixed floating-point working precision. For a
matrix `M`, `‖M‖` denotes its Frobenius norm and `Mᵀ` its transpose. A **matrix
multiplication** ("matmul") is any operation computing a product `A·B` of two matrices. The
**contracted dimension** of a matmul is the shared inner dimension of its two operands.

`H` is a collision-resistant hash function and `PRF` a pseudo-random function; `Expand(·; d)`
maps a `PRF` output to a vector in `{−1,+1}` of length `d`. `⟨·⟩` denotes a fixed, injective,
domain-separated encoding of labels.

A **dataset** `D = (d_1, …, d_n)` is the agreed, public sequence of training records. A
**schedule** `π` maps each step index `t` to the sequence of record indices forming that step's
batch, so the honest batch at step `t` is `b_t = D[π(t)]`. The dataset commitment `h_D` is
defined in Section 4.3.


## 3. The declared computation

One training step is a fixed, public map

`C : (b, W_t) ↦ (P, W_{t+1})`,

where `b` is the training batch, `W_t` the tuple of model weights at step entry, `W_{t+1}`
the weights at step exit, and `P = (P_1, …, P_M)` the tuple of all matrix products computed
during the step. The verifier knows `C`; it is the agreed program, not part of the
transcript.

`C` is composed of two disjoint classes of operation:

1. **Matmuls**, the products `P_1, …, P_M`; and
2. **Glue**, every other operation — elementwise maps, normalizations, softmax and other
   reductions, the input embedding, the loss, and the parameter update.

### 3.1 Coverage

Every matrix product in `C` is a matmul and is checked (Section 6); every operation that is
not a matrix product is glue and is recomputed (Section 6). For each **learnable weight**
`W`, three matmuls occur in a step and are covered: the forward product, the input-gradient
product, and the weight-gradient product. For each **weight-free bilinear operation** — a
product of two activations, such as an attention score or attention-value product — two
classes of matmul occur and are covered: the forward product and the two operand-gradient
products. No matmul is exempt.


## 4. Commitment

### 4.1 Transcript

The **step transcript** is the ordered leaf sequence

`L = ( b, W_t, P_1, …, P_M, W_{t+1} )`,

with the products `P_1, …, P_M` in a fixed canonical order. Glue outputs are **not**
included. Each leaf is encoded under a **canonical serialization**: a deterministic,
injective byte encoding with no optional, reserved, or nonce fields, so that a given
mathematical object has exactly one admissible encoding.

### 4.2 Merkle commitment

The prover commits to `L` with a Merkle tree under `H`: each leaf is hashed, sibling hashes
are concatenated and hashed pairwise up the tree, and the root

`h = MerkleRoot_H(L)`

is the **step commitment**. By collision resistance of `H`, `h` binds every leaf: no
transcript `L' ≠ L` admits the same root except with negligible probability. The tree
supports checking any single leaf against `h` from its authentication path alone, which is
what permits the memory-bounded verification of Sections 6 and 7.

### 4.3 Dataset commitment and schedule

The agreed dataset `D` is committed once, for the whole run, by a Merkle tree under `H` over
its canonically-encoded records:

`h_D = MerkleRoot_H( ⟨d_1⟩, …, ⟨d_n⟩ )`.

This commitment is **run-level and published**: an agreed external value, computed once, exactly
as the agreed base and final weights are. It is deliberately **separate from the per-step
commitment `h`** of Section 4.2. Separating them (i) scopes each `h` to a single step, so the
challenges of Section 5 depend only on that step's transcript; (ii) adds nothing to per-step
soundness by conflation, since the batch `b` is already a leaf of `h` and `D` is a run-level
constant identical across steps; and (iii) preserves memory-bounded verification — the verifier
retains only `h_D` and the schedule between steps and, when it does not hold `D` in full, checks
each step's records by their authentication paths into `h_D`.

The schedule `π` is public and agreed; in an honest run the batch at step `t` is `b_t = D[π(t)]`.
It may be published directly or derived from the commitment by a public generator, `π = G(h_D)`
— the data-order derivation used in proof-of-training-data schemes, without their additional
dataset nonce, which is unnecessary here because `D` is externally agreed rather than
prover-chosen.


## 5. Challenge generation

For each matmul `m ∈ {1, …, M}` with contracted dimension `d_m`, and each repetition
`j ∈ {1, …, k}`, the **challenge vector** is

`r(m,j) = Expand( PRF(h, ⟨m, j⟩); d_m ) ∈ {−1,+1}` of length `d_m`.

Each challenge is a deterministic function of the step commitment `h`. Because `h` binds the
entire transcript (Section 4.2), the challenges are bound to the complete report and are
**non-adaptive**: the prover fixes the transcript before any challenge is determined, and
cannot alter one leaf without redetermining every challenge. The verifier derives the
challenges itself and never accepts challenge vectors from the prover.


## 6. The verification protocol

The protocol is stated below as a single procedure with a prover phase and a verifier phase.
The verifier is given `C`, the agreed base weights `W_0`, and the agreed final weights `W_T`
of the run.

**Protocol (Combined Verification).**

*Prover, for each step `t = 0, …, T−1`:*

1. Execute `C(b, W_t)` at the working precision, with batch `b = D[π(t)]` selected by the
   schedule, recording every matmul output `P_1, …, P_M` and the updated weights `W_{t+1}`.
2. Form the transcript `L` (Section 4.1) and compute the commitment `h` (Section 4.2).
3. Transmit `L` and `h` (and, when the verifier does not hold `D`, the authentication paths of
   the step's records into `h_D`).

*Verifier.* Checks 0–1 are performed once at the start of the run, checks 2–7 for each step `t`,
and checks 8–9 once at the end.

*At the start of the run:*

- **Check 0 — Base anchor.** Require the committed entry weights `W_0` of the first step to equal
  the agreed base weights; reject otherwise. An invalid initialization is caught before any step
  is processed.
- **Check 1 — Dataset anchor.** Compute the dataset commitment `h_D` (Section 4.3) over the
  agreed dataset `D` and require it to equal the agreed, published commitment; adopt the agreed
  schedule `π`. The verifier then retains only `h_D` and `π` between steps, so this adds constant
  state.

*For each step `t`:*

- **Check 2 — Commitment.** Recompute the Merkle root of the received `L` and require it to equal
  `h`. This fixes the challenges of Section 5.
- **Check 3 — Batch binding.** Recompute the input embedding from the committed batch `b` and
  require the result to equal the committed activation that is the first matmul's input.
- **Check 4 — Batch anchor.** Require the committed batch `b` to equal the agreed records
  `D[π(t)]` selected by the schedule for step `t`: by direct comparison when the verifier holds
  `D`, or otherwise by verifying, for each record, an authentication path into `h_D` together with
  its position under `π`. This binds the committed batch to the agreed dataset, as the base and
  final anchors bind the committed weights to the agreed model.
- **Check 5 — Matmul checks.** For each matmul `m = 1, …, M`: reconstruct its operands
  `A_m, B_m` by recomputing the intervening glue from previously committed leaves; then for each
  `j = 1, …, k` reject if

  `‖ A_m·(B_m·r(m,j)) − P_m·r(m,j) ‖ > τ`.

- **Check 6 — Update identity.** For each learnable weight `W`, reject if

  `‖ W_{t+1} − (W_t − η·G_W) ‖ > τ_W`,

  where `G_W` is the committed weight-gradient product for `W` and `η` the update rule given by
  `C`.
- **Check 7 — Chaining.** Require `W_t` to equal the committed `W_{t+1}` of step `t−1` (the same
  leaf), for `t ≥ 1`.

*At the end of the run:*

- **Check 8 — Final anchor.** Require the last committed `W_T` to equal the agreed final weights;
  reject otherwise.
- **Check 9 — Accept.** Accept the run iff the base anchor (check 0), the dataset anchor
  (check 1), every per-step check 2–7, and the final anchor (check 8) hold.


## 7. Verifier cost and error locality

In check 5 the verifier evaluates only matrix–vector products, at cost `O(d²)` per matmul in
place of the `O(d³)` of recomputing `P_m`. In checks 3 and 5 the verifier recomputes glue but
never a matmul. The batch anchor (check 4) costs one comparison, or one authentication path,
per record of the batch, and the dataset anchor (check 1) is a single run-level computation;
neither grows the verifier's per-step state. Because every reconstructed operand is derived
from committed leaves rather than from a value the verifier carries forward, the tolerance of
one check does not accumulate into the next: each check is a local, independent test against the
committed transcript.


## 8. Security

Let `λ` denote the target security margin in bits. Fix a step and let `Δ_m = P_m − A_m·B_m`
be the discrepancy of matmul `m` under the honest operands reconstructed in check 5.

### 8.1 Completeness

If the prover executes `C` honestly at the working precision, then every residual tested in
checks 3, 5, and 6 is a floating-point rounding quantity. Writing `s_h` for the honest
residual band of a matmul check (Section 9), if `τ ≥ z·s_h` and `τ_W` is set above the honest
update-identity residual, the verifier accepts the step except with a false-rejection
probability `α` that decreases monotonically in the margin `z`. The anchors (checks 0, 1, and
8), the batch anchor (check 4), and chaining (check 7) are exact equality tests, which an honest
transcript passes with certainty.

### 8.2 Soundness

Suppose the committed transcript deviates from the honest execution of `C` on the committed
`(b, W_t)`: some matmul has `‖Δ_m‖ > Φ`, or the update identity fails by more than `τ_W`. For
a single challenge, the deviating matmul passes only if `‖Δ_m·r‖ ≤ τ`; for challenges with
independent `±1` entries this misdetection has probability

`p_1 ≈ c · τ / ‖Δ_m‖`,  with  `c = Θ(1)`,

so each challenge contributes `b_0 = log₂(1/p_1)` bits. A prover exploiting non-interactivity
may **grind**: alter the transcript, redetermine the challenges, and retry, at the cost of a
Merkle-path rehash per attempt; let `G` bound the number of such attempts. Over `M` matmuls,
`T` steps, and `G` grinding attempts, a union bound gives an acceptance probability for a
deviating run of at most

`T · M · G · (p_1)^k`.

Choosing the repetition count `k` as the least integer with

`k · b_0 ≥ λ + log₂T + log₂M + log₂G`

bounds this probability by `2^(−λ)`. The **detection floor** `Φ` is the least deviation
magnitude for which this budget is met at the chosen `k`; it is proportional to `τ`, and hence
to the working precision.

A committed batch that is not the agreed data of the schedule, or endpoint weights that do not
match the agreed model, is rejected deterministically by the anchors (checks 0, 1, 4, and 8);
the randomized bound above governs only the matmul checks and the update identity.

### 8.3 Scope

Checks 0 and 8 bind the certified weights to the agreed model at the endpoints, check 7 chains
the steps between them, check 3 binds each committed batch to the computation that consumes it,
and check 4, together with the run-start dataset anchor (check 1), binds each committed batch to
the agreed training dataset and its schedule. Acceptance therefore certifies the real training
transition on the agreed data, and not a self-consistent transcript of a different computation
or of a substituted dataset. Two limitations are inherent:

1. A deviation with `‖Δ_m‖ ≤ Φ` is not detected (the detection floor).
2. The protocol certifies faithful execution of `C` on the agreed dataset; it does not certify
   that `C` is well-chosen, nor that the agreed dataset is itself benign. A
   malicious-but-honestly-executed computation, or honest training on an agreed-but-corrupted
   dataset, is accepted. Training on data outside the agreed dataset, or on the agreed data in a
   different per-step schedule, is by contrast detected (checks 1 and 4).


## 9. Configuration and calibration

The protocol is instantiated by a **configuration profile** fixing the following, none of
which is part of the protocol proper:

- **Working precision**, which sets the honest residual band and thus the floor `Φ`.
- **Tolerances** `τ` and `τ_W`. These are calibrated empirically: the honest residual band
  `s_h` is measured over a small number of honest steps at the working precision, and
  `τ = z·s_h` for a margin `z` that fixes the false-rejection rate `α`; `τ_W` is calibrated
  the same way, per weight tensor, against the honest update-identity residual.
- **Repetition count** `k`, the least integer meeting the soundness budget of Section 8.2 for
  the assumed grinding bound `G` and target margin `λ`.
- **Cryptographic primitives**: the concrete hash `H`, the `PRF`, and the canonical
  serialization of Section 4.1.
- **Dataset anchor**: the dataset commitment `h_D`, the schedule `π` (published directly or
  derived from `h_D` by a public generator, `π = G(h_D)`), and whether the verifier holds the
  full dataset `D` for direct comparison or only `h_D` and checks records by authentication
  path. As with the weight anchors, this is a logistics choice and does not alter the procedure
  of Section 6.

A **run mode** is a configuration profile together with a model and data scale. Because the
protocol is scale-invariant (Section 1), a small-scale profile and a full-scale profile
execute the identical procedure of Section 6 and differ only in these values.
