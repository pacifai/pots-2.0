# Appendix: Sizing the Verification Parameters

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This appendix gives the complete calculation that fixes the numeric parameters of the
verification protocol: the tolerance applied to each matrix-product check, the cancellation
guard, and the number of independent challenge vectors used per product. It is written to be
read on its own. Every quantity it uses is defined in Section 1 and named in full thereafter;
no quantity is referred to by an external identifier, and the only cross-references are to the
numbered sections of this document.

The calculation has one property worth stating at the outset: **it is analytic, not
empirical.** Only two of its inputs are measured from a run. Everything else is either a
structural constant of the model, a property of the arithmetic, or a value chosen by hand.
Section 9 lists the chosen values separately, because which numbers are chosen and which are
computed is the single easiest thing to lose track of when reading a sizing argument.


## 1. Quantities

Each quantity is marked by origin:

- **structural** — fixed by the model architecture and the training configuration;
- **arithmetic** — fixed by the floating-point formats in use;
- **chosen** — set by hand, by a person, as a design decision (Section 9);
- **measured** — obtained from a small number of honest training steps;
- **derived** — computed from the above by the formulas of this appendix.

### 1.1 Indices and structure

| symbol | meaning | origin |
|---|---|---|
| `m` | index of a matrix product within one training step, `m = 1, …, M` | structural |
| `M` | number of matrix products computed in one training step | structural |
| `T` | number of training steps in the run | structural |
| `A_m`, `B_m` | the two operands of product `m`, so the claimed product is `A_m·B_m` | structural |
| `P_m` | the product value the prover commits for product `m` | — |
| `q_m` | **contracted dimension** of product `m`: the shared inner dimension of `A_m` and `B_m`, that is, the number of terms summed to form one entry of the product | structural |
| `w_m` | **width** of product `m`: the number of columns of `B_m`, and hence of `P_m` | structural |

### 1.2 Arithmetic

| symbol | meaning | origin |
|---|---|---|
| `ε_in` | unit roundoff of the format in which the product's **operands** are stored | arithmetic |
| `ε_acc` | unit roundoff of the format in which the product's sum is **accumulated** | arithmetic |

For binary floating point, unit roundoff is half the gap between 1 and the next representable
number: `ε = 2⁻²⁴ ≈ 5.9605·10⁻⁸` for single precision (fp32), and `ε = 2⁻⁸ ≈ 3.9063·10⁻³` for
bfloat16. The two are listed separately because a bfloat16 matrix product on current hardware
holds its operands in bfloat16 but accumulates in single precision, and the two roundoffs then
enter the error at different rates (Section 2). When one format is used throughout,
`ε_in = ε_acc`.

### 1.3 The challenge

| symbol | meaning | origin |
|---|---|---|
| `r` | a challenge vector of length `w_m`, entries drawn independently from the challenge distribution | — |
| `σ_r` | standard deviation of one entry of `r`; for the uniform distribution on `(−1, 1)`, `σ_r = 1/√3 ≈ 0.5774` | arithmetic |
| `k` | **repetition count**: the number of independent challenge vectors applied to each product | derived (Section 8) |
| `c` | anti-concentration constant of the challenge distribution (Section 6); `c = √(2/3) ≈ 0.816` (0.798 before 2026-10-06) | arithmetic |

### 1.4 Bands and targets

| symbol | meaning | origin |
|---|---|---|
| `e_m` | **honest relative error** of product `m`: the expected size of the floating-point error in `P_m`, relative to `P_m` itself (Section 2) | derived |
| `ρ` | **normalized residual**: the tested residual divided by its honest scale, so that an honest product gives `ρ ≈ 1` (Section 3) | derived |
| `s_h` | honest band of `ρ`, measured over honest steps (Section 3) | measured |
| `z` | **tolerance margin**: how many multiples of the honest band the test permits | chosen |
| `τ` | **tolerance**: `τ = z·s_h`, dimensionless | derived |
| `ν_m` | **operand-magnitude bound** of product `m` (Section 4) | derived |
| `κ_m` | **cancellation factor** of product `m` (Section 4) | derived |
| `κ_max` | ceiling on the cancellation factor | measured |
| `f` | **detection target**: the size of the deviation the protocol is required to catch, as a fraction of the product's own magnitude (Section 5) | chosen |
| `Δ_m` | the deviation actually present in a dishonest `P_m`, that is `Δ_m = P_m − A_m·B_m` | — |

### 1.5 The security budget

| symbol | meaning | origin |
|---|---|---|
| `λ` | **security margin** in bits: the protocol's target is that a dishonest run is accepted with probability at most `2⁻λ` | chosen |
| `G` | **grinding bound**: the number of times a dishonest prover is assumed able to alter its transcript, recompute the challenges, and retry | chosen |
| `b₀` | **per-challenge soundness** in bits: the information one challenge vector yields against a deviation of the target size, defined by `p₁ = 2^(−b₀)` in (6.3) and evaluated in (6.4) | derived |
| `N` | **bit budget**: the total soundness the repetition count must supply (Section 7) | derived |

Norms: `‖·‖` on a vector is the Euclidean norm; `‖·‖_F` on a matrix is the Frobenius norm, the
square root of the sum of the squares of all entries. `|X|` denotes the entrywise absolute
value of `X`, and `1_d` the all-ones vector of length `d`.


## 2. The honest error of a floating-point product

One entry of `A_m·B_m` is a sum of `q_m` products of pairs of entries. Computing it in floating
point introduces error from two sources, which scale differently.

**Operand rounding.** Each operand entry is stored with relative error at most `ε_in`. A product
of two such entries carries relative error about `√2·ε_in`, and this error does not grow with
the number of terms summed, because it is a relative perturbation of every term alike.

**Accumulation.** Each of the `q_m − 1` additions rounds its running sum, with relative error
`ε_acc`. These errors are independent in sign, so they combine as a random walk rather than
additively, and their total grows as `√q_m·ε_acc` relative to the result.

The honest relative error of the product is therefore

> **(2.1)**  `e_m = √2·ε_in + √(q_m)·ε_acc`

This is a relative quantity: the absolute error of the computed product is `e_m·‖P_m‖_F`.

Two remarks on the validity of (2.1), both of which matter later.

*It is insensitive to sign structure.* If the terms of the sum share a sign, the running sums
grow to the full magnitude of the result and each rounding is correspondingly large; if the
signs are mixed, the running sums stay smaller but so does the result. Both regimes give the
same **relative** error `√q_m·ε_acc`, which is why (2.1) can be stated relative to `‖P_m‖_F`
without knowing the sign structure.

*It fails under pathological cancellation.* If the terms of the sum cancel almost exactly, the
result approaches zero while the rounding errors do not, and the relative error is unbounded.
Section 4 defines a computable guard against that case, because (2.1) is the assumption the
whole calculation rests on.

Numerically, with single precision throughout (`ε_in = ε_acc = 2⁻²⁴`):

| `q_m` | `e_m` |
|---|---|
| 64 | `5.6·10⁻⁷` |
| 576 | `1.5·10⁻⁶` |
| 1,536 | `2.4·10⁻⁶` |
| 49,152 | `1.33·10⁻⁵` |

With bfloat16 operands and single-precision accumulation, `e_m ≈ 5.54·10⁻³` at `q_m = 49,152`,
dominated entirely by the operand term — a factor of about 400 worse than single precision.


## 3. The normalized residual and the tolerance

The verifier does not recompute `A_m·B_m`. It draws a challenge vector `r` and compares

> `A_m·(B_m·r)`  against  `P_m·r`

Both sides cost a pair of matrix-vector products rather than a matrix-matrix product, which is
the source of the protocol's efficiency. For an honest prover the two differ only by rounding.

To size that difference: writing `E_m = P_m − A_m·B_m` for the honest floating-point error, the
tested residual is `E_m·r`, and since the entries of `r` are independent and mean-zero,

> **(3.1)**  `E[‖E_m·r‖²] = σ_r²·‖E_m‖_F²`

Combining (3.1) with the error model (2.1), the honest residual has scale

> **(3.2)**  `‖E_m·r‖ ≈ σ_r · e_m · ‖P_m‖_F`

Every term on the right is available to the verifier: `σ_r` from the challenge distribution,
`e_m` from the shape and the precision, and `‖P_m‖_F` from the committed product by one pass
over it. This gives the **normalized residual**

> **(3.3)**  `ρ_m = ‖ A_m·(B_m·r) − P_m·r ‖ / ( σ_r · e_m · ‖P_m‖_F )`

which is dimensionless and, for an honest product, close to 1 **whatever the product's shape,
magnitude, contracted dimension or working precision**. That is the property that lets a single
tolerance serve every product of the step. Without it, a tolerance would have to be set by the
noisiest product and would leave every quieter one with slack that a deviation could hide in.

The test is

> **(3.4)**  reject if  `ρ_m > τ`,  with  `τ = z·s_h`

where `s_h` is the honest band of `ρ` measured over a few honest steps and `z` is the chosen
margin. Because false rejection is a union bound over every challenge of every product of every
step, `s_h` is taken as the **largest class-wise root-mean-square** of `ρ`, where a class is a
set of products occupying the same role in the step: the root mean square within a class,
because `ρ` concentrates; the largest across classes, because the noisiest class governs the
rate. If the model (2.1) is accurate, `s_h ≈ 1` and `τ ≈ z`; measuring `s_h` is how that
assumption is checked rather than assumed.

In implementation, (3.4) is applied as the equivalent product form

> **(3.5)**  reject if  `‖ A_m·(B_m·r) − P_m·r ‖ > τ · σ_r · e_m · ‖P_m‖_F`

which avoids dividing by a `‖P_m‖_F` that may be zero.


## 4. The cancellation guard

Section 2 noted that the error model fails when the terms of the sum cancel almost exactly. The
guard is a computable measure of how much cancellation a product exhibits.

Define the **operand-magnitude bound**

> **(4.1)**  `ν_m = ‖ |A_m| · ( |B_m| · 1_{w_m} ) ‖`

This is the sum of the absolute values of all the terms that form the product, gathered into a
single number. It costs two matrix-vector products, evaluated once per product rather than once
per challenge vector, because it is a scale rather than a per-challenge quantity.

Define the **cancellation factor**

> **(4.2)**  `κ_m = ν_m / ‖ |P_m| · 1_{w_m} ‖`

Both numerator and denominator are row sums of absolute values, so they are directly comparable,
and since `|A·B| ≤ |A|·|B|` entrywise and therefore also after summing rows,

> **(4.3)**  `κ_m ≥ 1`,  with equality exactly when no cancellation occurs.

The guard is

> **(4.4)**  reject if  `ν_m > κ_max · ‖ |P_m| · 1_{w_m} ‖`

with `κ_max` calibrated per class from honest steps and then held fixed. Two points justify its
form.

*It is a reject condition, not a diagnostic.* A product whose cancellation is pathological has a
genuinely wider honest band than (2.1) predicts, so a deviation hidden there would escape the
test of Section 3 while looking like an ordinary accept. Rejecting makes that case visible.

*It is calibrated per class, whereas the tolerance of Section 3 is a single number.* The
asymmetry is deliberate. The normalization of (3.3) removes the spread in the residual, so one
tolerance costs only a little slack. Cancellation has no such normalization: how much a product
cancels is a structural property of its role in the step, differing intrinsically between, say,
an attention score product and a weight gradient, and no formula in the operands predicts it. A
single global ceiling would be set by the most-cancelling class and would be vacuous for every
other one. The per-class entries are affordable because they are **dimensionless ratios**, which
do not drift as the model's activations grow during training.

*It is frozen after calibration and never refitted on the steps under judgment.* A ceiling that
refits on the steps it is judging can be dragged upward by a prover who raises cancellation
slowly, at which point it no longer constrains anything.


## 5. The detection target

The protocol cannot promise to detect an arbitrarily small deviation: a deviation smaller than
the honest rounding error is indistinguishable from that error by any test. The size that must
be detected is therefore a choice, and it is stated relative to the product it corrupts:

> **(5.1)**  `‖Δ_m‖_F = f · ‖P_m‖_F`

with `f` chosen. Setting `f = 1` asks the protocol to detect any deviation as large as the
product itself; smaller `f` asks for more and costs repetitions.

Stating the target relative to `‖P_m‖_F` rather than as an absolute magnitude is what makes the
calculation portable: the same `f` means the same thing at every product of the step, at every
model size, and at every precision, whereas an absolute magnitude would have to be restated for
each.

`f` should be read as a **claim the protocol makes**, not as an estimate of what an attacker
will do. Section 8 accordingly reports the smallest `f` a given repetition count achieves, which
is a result, alongside the `f` used to select that count, which is a decision.


## 6. Per-challenge soundness

Fix a product carrying a deviation `Δ_m` of the target size. Under one challenge it escapes
detection exactly when its residual falls inside the band, that is when
`‖Δ_m·r‖ ≤ τ·σ_r·e_m·‖P_m‖_F`.

Because the entries of `r` are independent and continuously distributed, the projection `Δ_m·r`
is itself continuously distributed with a bounded density, and by (3.1) its scale is
`σ_r·‖Δ_m‖_F`. The probability that it lands within a given distance of a fixed value is
therefore proportional to that distance:

> **(6.1)**  `p₁ ≈ c · τ · σ_r · e_m · ‖P_m‖_F / ( σ_r · ‖Δ_m‖_F )`

The constant `c` is the density factor of the challenge distribution near the origin. For a
projection onto many independent entries the distribution is close to Gaussian, whose density at
the origin gives `c = 2·(2π)^(−1/2) ≈ 0.798`; it is insensitive to which continuous distribution
is used, because the scale `σ_r` appears in the numerator and denominator of (6.1) and cancels.

*Revised 2026-10-06 (`DECISIONS_EVALUATION.md` EQ10, user approved).* The Gaussian value is not
the worst case for `Uniform(−1,1)` challenges, because a deviation spread over few entries
keeps the projection far from Gaussian. For a rank-1 deviation `Δ = u·vᵀ`, `‖Δ·r‖ = ‖u‖·|v·r|`,
so `p₁` is set by the density of `v·r` at 0. Ball's cube-slicing theorem (1986) bounds that
density, and two equal entries in one row reach the bound: `c = √2·σ_r = √(2/3) ≈ 0.816`. One
entry gives `σ_r = 0.577`, and a dense deviation has no `1/x` tail at all. The sizing now uses
`c = √(2/3)`, so `log₂(1/c) = 0.29`, not 0.33, and every `b₀` drops by 0.034 bits. The A13
sweep measured the one-row two-entry shape on all 30 classes: fitted `ĉ` 0.72–0.83, mean about
0.78, against a simulated 0.777 ± 0.023 for the same fit. No class exceeds the bound by more
than noise.

Substituting the target (5.1) and cancelling `σ_r` and `‖P_m‖_F`:

> **(6.2)**  `p₁ ≈ c · τ · e_m / f`

The per-challenge soundness is this probability re-expressed in bits. **Definition:** `b₀(m)`
is the exponent for which

> **(6.3)**  `p₁ = 2^(−b₀(m))`,  equivalently  `b₀(m) = log₂(1/p₁)`

so that "one challenge is worth `b₀` bits" means literally that it cuts a deviating product's
survival probability by a factor `2^(b₀)`. Bits are used rather than probabilities only because
`k` independent challenges multiply their probabilities, `p₁^k = 2^(−k·b₀)`, which turns the
budget of Section 7 into addition instead of multiplication. Substituting (6.2) into (6.3):

> **(6.4)**  `b₀(m) = log₂( f / ( τ · e_m ) ) + log₂(1/c)`

with `log₂(1/c) ≈ 0.326`.

Note what (6.2) does **not** contain: the magnitude of the product, the magnitude of the
operands, the width, or anything measured from the run other than `τ`. The per-challenge
soundness of a product depends only on the detection target, the tolerance, and the product's
own contracted dimension and precision through `e_m`. This is what makes the repetition count
computable in advance rather than fitted after the fact.

Since `e_m` increases with `q_m` by (2.1), `b₀` **decreases** with the contracted dimension: the
deeper the contraction, the larger the honest error, and the more room a deviation has to hide.
The product with the largest contracted dimension is therefore the binding one.


## 7. The bit budget

A dishonest prover passes the whole run if some deviating product survives all `k` of its
challenges, at any of the `M` products of any of the `T` steps. Because the challenges are
derived from a commitment to the step transcript rather than supplied interactively, the prover
may additionally **grind**: alter the transcript, recompute the resulting challenges, and retry,
at the cost of rehashing an authentication path per attempt. Let `G` bound the number of such
attempts. A union bound gives an acceptance probability of at most `T·M·G·p₁^k`, so requiring
that this be at most `2⁻λ` gives, on taking `log₂` of both sides and substituting
`p₁^k = 2^(−k·b₀)` from (6.3),

> **(7.1)**  `k · b₀ ≥ N`,  with  `N = λ + log₂T + log₂M + log₂G`

`N` is the total soundness, in bits, that the repetitions must supply. Three of its four terms
are chosen or structural and one, `log₂G`, is an assumption about the adversary's compute; it
dominates the budget, and it does not change with model size.


## 8. Solving for the repetition count

The calculation runs in this order. Steps 1–4 require no run; steps 5–6 require a short honest
run; steps 7–9 are arithmetic.

1. **Read the structure.** For every product of the step, record its contracted dimension `q_m`,
   and record `M` and `T`.
2. **Read the arithmetic.** Fix `ε_in` and `ε_acc` from the chosen precision, and `σ_r` from the
   challenge distribution.
3. **Compute the honest relative error** `e_m = √2·ε_in + √(q_m)·ε_acc` for each product, by
   (2.1), and identify the **binding product**, the one with the largest `e_m` — equivalently,
   the largest contracted dimension.
4. **Choose** the tolerance margin `z`, the detection target `f`, the security margin `λ`, and
   the grinding bound `G` (Section 9).
5. **Run a few honest steps** and measure `s_h`, the largest class-wise root-mean-square of the
   normalized residual (3.3). Confirm `s_h ≈ 1`; a substantially larger value means the error
   model (2.1) understates the honest error for this configuration and the discrepancy must be
   explained before proceeding.
6. **Measure `κ_max`** per class from the same steps, by (4.2).
7. **Set the tolerance** `τ = z·s_h`.
8. **Compute the per-challenge soundness at the binding product**,
   `b₀ = log₂( f / (τ·e_max) ) + log₂(1/c)` by (6.4).
9. **Solve for the repetition count**, `k = ⌈N / b₀⌉` with `N` from (7.1).

Then **report the achieved target** by inverting (6.4) at the chosen `k`: the smallest deviation
the configuration guarantees to detect is

> **(8.1)**  `f_achieved = c · τ · e_max · 2^(N/k)`

expressed, by (5.1), as a fraction of the corrupted product's own magnitude.


## 9. The values that are chosen by hand

Everything in Sections 2–8 is computed except the following. These are decisions, not
measurements, and no run produces them.

| quantity | value | why this value |
|---|---|---|
| `λ`, security margin | 25 bits | The residual failure probability after the budget is met. Chosen as a conventional margin. |
| `log₂G`, grinding bound | 52 | Models a well-resourced single laboratory: roughly `2⁵²` transcript alterations, each with its rehash. It dominates `N`, so it is the most consequential chosen number in the calculation. |
| `z`, tolerance margin | 8 | How many multiples of the honest band the test allows before rejecting. It trades false rejection against detection: every doubling of `z` costs one bit of `b₀`. Chosen so that honest rejection stays rare across the roughly `2⁴¹` individual challenge evaluations of a full-scale run. |
| `f`, detection target | 1 | Asks the protocol to detect any deviation as large as the product it corrupts. It is the **sizing input** and nothing else: `f_achieved` (8.1) is the claim to report, and the detectable substitution rate of 12.4 is a further claim derived from it, conditional on the unmeasured coherence factor of 12.2. P4 fixed this separation of roles. |
| precision | fp32 at test scale; bfloat16 operands with fp32 accumulation at full scale | Sets `ε_in` and `ε_acc`, and through them the whole calculation. This is the single largest lever on `k` (Section 10). |
| challenge distribution | uniform on `(−1, 1)` | Fixes `σ_r = 1/√3` and `c = √(2/3) ≈ 0.816` (Section 6). It must be continuous; a two-valued distribution admits deviations that no tolerance can catch. |

The two **measured** inputs are `s_h` and `κ_max`, both from a short honest run. Everything else
is structural or arithmetic.


## 10. Worked instances

Common to all three: `λ = 25`, `log₂G = 52`, `z = 8`, `f = 1`, `c = 0.798`, and `s_h` assumed to
measure at 1 so that `τ = 8`. The binding product is the input-gradient of the output
projection, which contracts over the vocabulary, `q_max = 49,152`.

*These worked figures use `c = 0.798`. With `c = √(2/3)` (Section 6, 2026-10-06), each `b₀` is
0.034 bits lower. Test scale keeps `k = 9` at the measured `τ = 44` (`f_achieved` on `dF` rises
from 0.608 to 0.622; margin over `1/√2` 1.14×). Full scale keeps `k = 21` at `T = 10` for all
four models. At the `2²⁰`-step reference of 10.3, `k` stays 24 for SmolLM2's product count and
Falcon3-1B, and becomes 25 for Llama-3.2-1B and both Qwen2.5 models, which had almost no slack
(`N/b₀` 24.02–24.15).*

### 10.1 Test scale — fp32, one 135M-parameter model, 10 steps

`M = 7,113`, `T = 10`, `ε_in = ε_acc = 2⁻²⁴`.

```
e_max = √2·5.9605e-8 + √49152·5.9605e-8 = 1.330e-5
b₀    = log₂( 1 / (8 · 1.330e-5) ) + 0.326 = 13.20 + 0.33 = 13.52
N     = 25 + log₂10 + log₂7113 + 52 = 25 + 3.32 + 12.80 + 52 = 93.12
k     = ⌈93.12 / 13.52⌉ = 7
```

### 10.2 Full scale — fp32 hypothetically, `2²⁰` steps

`M = 207,993`, `T = 2²⁰`, same `e_max` and `b₀`.

```
N = 25 + 20 + 17.67 + 52 = 114.67
k = ⌈114.67 / 13.52⌉ = 9
```

### 10.3 Full scale — bfloat16 operands, fp32 accumulation

```
e_max = √2·3.9063e-3 + √49152·5.9605e-8 = 5.538e-3
b₀    = log₂( 1 / (8 · 5.538e-3) ) + 0.326 = 4.50 + 0.33 = 4.82
k     = ⌈114.67 / 4.82⌉ = 24
```

The precision change costs a factor of more than two and a half in repetitions, because
bfloat16 operand rounding raises the honest error by a factor of about 400 and each factor of
two costs one bit of `b₀`.

### 10.4 Achieved detection targets

By (8.1), at the values above:

| configuration | `k` | `f_achieved`, as a fraction of the product |
|---|---|---|
| fp32, test scale | 7 | `0.86` |
| fp32, `2²⁰` steps | 9 | `0.58` |
| bfloat16, `2²⁰` steps | 24 | `0.97` |

Each configuration detects somewhat **smaller** deviations than the chosen target of `f = 1`,
by the margin created by rounding `k` up to an integer; the full-precision long run gains most,
because its `k = 9` clears its budget with the most slack. Raising `k` by one lowers the
achieved target by a factor of `2^(N/(k(k+1)))` — about fourfold at `k = 7`, about 15% at
`k = 24` — so at small `k` the detection target is cheap to improve and at large `k` it is not.


## 11. Consistency checks

The calculation was checked against four independently derived results.

1. **Typical rather than binding contraction.** Evaluating Section 8 at `q_m = 576`, the
   contraction of most of the model's dense products, gives `e_m = 1.51·10⁻⁶`, `b₀ = 16.65`, and
   `k = ⌈114.67/16.65⌉ = 7` at full scale. This reproduces the figure obtained from an earlier,
   separate estimate, and locates the difference: that estimate was evaluated at a typical
   contraction, whereas Section 8 evaluates at the largest one. The vocabulary-width
   input-gradient product, contracting over 49,152, is the binding case and raises `k` from 7
   to 9.
2. **bfloat16.** Section 10.3 gives `k = 24` against an earlier independent estimate of
   `k ≈ 22`, agreeing to within the rounding of `log₂M` and the assumed `s_h`.
3. **Relative honest error.** (2.1) gives `√576 · 2⁻²⁴ ≈ 2⁻¹⁹·⁴` at the typical contraction,
   matching the independently quoted honest band of `2⁻¹⁸` to `2⁻²⁰` for single precision.
4. **Anti-concentration constant.** `c ≈ 0.798` follows from the Gaussian density at the origin,
   `2·(2π)^(−1/2)`, and matches the value `c ≈ 0.8` obtained separately for this challenge
   distribution. *Superseded 2026-10-06:* the worst case under uniform challenges is
   `c = √(2/3) ≈ 0.816` (Section 6).

One check is deferred to the run: the protocol's fault-injection harness sweeps a deliberately
corrupted product across a range of deviation magnitudes and records where detection fails. That
measured threshold should agree with the achieved target (8.1). It is the only end-to-end
validation of this appendix, and a disagreement would indicate that the error model (2.1), and
not the arithmetic built on it, is wrong.


## 12. Open notes, not yet resolved

Sections 1–11 are settled. This section records questions raised while deriving them that are
**not** answered here, so that the reader is not left to reconstruct them. Each is deliberately
left open.

### 12.1 What the budget of Section 7 does and does not say

(7.1) is a union bound for the **verifier's** safety: the verifier must be safe whichever of the
`T·M` products is the corrupted one, so it pays `log₂T + log₂M` bits to cover them all. It is
not a statement about the attacker's workload. An attacker needs every product it corrupts to
survive simultaneously, so its problem is to find the **minimum cut** — the smallest set of
committed products whose corruption achieves its goal.

That minimum cut is **one product**. An attacker that substitutes a training record and commits
the resulting computation honestly would dirty about 2,253 of the 7,113 products of that step
(every batched linear product, because those are computed over the whole batch, plus the
attention products of the affected sequence only), but it need not do that. It can instead
corrupt a single weight-gradient product at a single step and continue honestly from the
corrupted weights, at which point every later step is internally consistent and passes. The
blast radius of a naive substitution is therefore one step wide and irrelevant to the guarantee:
**the guarantee must be, and is, sized for a single deviating product.** That is also why `M`
appears only as `log₂M` — a 29-fold increase from test to full scale costs 4.9 bits of `N`, less
than half a challenge vector at `k = 9`.

Corrupting the weight update rather than the gradient is not an alternative: check 6 tests the
update identity deterministically within `τ_W`, with no probability involved. The attack surface
is a check-5 product.

### 12.2 Which counts of data points enter, and which do not

Three different counts behave differently, and only one matters.

| count | how it enters |
|---|---|
| dataset size | Not at all. It is bound by the dataset commitment; no quantity in Sections 1–11 sees it. |
| number of steps `T` | Only as `log₂T` in (7.1). Spreading a deviation across more steps does not help an attacker, because the per-step deviation needed is independent of `T`. |
| **batch size** | Through the relative size of the deviation, and it is the one that matters. |

For a batch in which a fraction `ρ` of the sequences is substituted, the deviation induced in a
weight-gradient product, relative to that product, is approximately

> **(12.1)**  `f_step ≈ ρ · √B · (‖δg‖ / ‖g‖)`

where `B` is the number of sequences in the batch, `‖g‖` is the norm of a typical per-example
gradient and `‖δg‖` the norm of the difference a substituted example induces. The `√B` is a
**coherence amplification**: substituted examples sharing a trigger produce gradients that point
in a common direction and add, while honest per-example gradients are largely incoherent and
partially cancel, so the denominator shrinks as `1/√B` while the numerator does not.

**This is the least secure step in the whole analysis and it is an assumption, not a result.** If
honest per-example gradients are *not* incoherent — early in training, or on a homogeneous
corpus — then `‖G‖ ≈ ‖g‖`, the `√B` disappears, and the deviation is about 11× smaller at a
128-sequence batch. The test is cheap and belongs in the calibration run: compute per-example
gradients for one batch and report `‖Σ_i g_i‖ / (√B · ‖g‖)`. A value near 1 supports (12.1); a
value near `√B` refutes it.

### 12.3 The grinding term dominates the undetectable region

Decomposing the achieved target (8.1) into its four contributions, at the full-scale fp32
instance of Section 10.2:

| contribution | factor by which it widens the undetectable region |
|---|---|
| honest band `c·τ·e_max` | `8.49·10⁻⁵` (the floor itself) |
| grinding, `2^(log₂G/k) = 2^(52/9)` | **× 54.9** |
| security margin, `2^(λ/k) = 2^(25/9)` | × 6.9 |
| steps, `2^(log₂T/k) = 2^(20/9)` | × 4.7 |
| products, `2^(log₂M/k) = 2^(17.67/9)` | × 3.9 |
| | `f_achieved = 0.58`, agreeing with 10.4 |

Grinding accounts for a factor of 55 of the total 6,859. Equivalently: **the smallest deviation
this protocol detects is about 55× larger than an interactive verifier would detect at the same
tolerance**, and that entire factor is the price of deriving the challenges from the commitment
rather than receiving them from a verifier. It is the largest single term, larger than the
security margin and the two union-bound terms combined.

Two consequences follow, and neither is acted on here.

*Relaxing the tolerance and permitting grinding are the same currency.* One bit of `τ`, one bit
of `G` and `1/k` of a challenge vector are interchangeable. Raising `z` to suppress false
rejection therefore buys an attacker grinding room at exactly the same rate.

*`k` is the efficient lever and `z` is not.* Since `f_achieved ∝ 2^(N/k)`, one extra challenge at
`k = 9` improves the achieved target by `2^(N/(k(k+1))) = 2.4×`, with no false-rejection cost.
Lowering `z` improves it linearly and cannot go far: `z = 8` was chosen so that honest rejection
stays rare across roughly `2⁴¹` challenge evaluations, which leaves at most about one bit.

### 12.4 Whether `f = 1` is the right target at all

Combining (12.1) with the achieved targets of 10.4 gives the smallest substitution rate the
protocol detects, taking `‖δg‖/‖g‖ ≈ 2`:

| configuration | `k` | `f_achieved` | smallest detectable substitution rate |
|---|---|---|---|
| fp32, test scale, `B = 4` | 7 | 0.86 | 21% |
| fp32, full scale, `B = 128` | 9 | 0.58 | 2.6% |
| bfloat16, full scale, `B = 128` | 24 | 0.97 | 4.3% |
| fp32, full scale, `k = 12` | 12 | 0.10 | 0.45% |

The published backdoor literature works at substitution rates of roughly 1–10%, so the protocol
as sized lands **inside** that range rather than below it: it detects the aggressive end and
misses the quiet end. Three open questions follow; 12.6 records how P4 disposed of them.

1. **Is `f = 1` the claim worth making?** Section 9 chose it as a headline ("detect any deviation
   as large as the product it corrupts"). A claim stated in terms of the substitution rate is
   more useful and more falsifiable, but it depends on (12.1), which is unvalidated.
2. **Should `k` be raised?** Three extra vectors take the full-scale threshold from 2.6% to
   0.45%, for roughly a third more check-5 arithmetic — cheap against a claim about poisoning,
   over-engineered against a claim about blatant forgery. Parked as F10.
3. **The test-scale configuration sits close to its own boundary.** One substituted record of
   four gives `f_step ≈ 1.0` against `f_achieved = 0.86`, a margin of about 1.2×, thinner than
   "blatant, order-1" suggests and entirely dependent on the coherence assumption of 12.2.

### 12.5 Two smaller points

*The grinding bound `log₂G = 52` is calibrated for a low-rank deviation, which is the right
choice.* A grinding attempt must evaluate the residual under fresh challenges, not merely rehash
a path. For a general deviation at `q = 49,152` that costs about `2.8·10⁷` operations per
challenge, which would put the real budget nearer `2⁴⁵`. For a rank-one deviation it factors as
`Δ·r = u·(vᵀr)` and costs about `5·10⁴`, restoring `2⁵²`. Rank-one is also the hardest shape to
detect, because the residual is then a one-dimensional projection — exactly the case the
constant `c` of Section 6 is derived for, whereas a full-rank deviation concentrates and is
easier to catch. So `f_achieved` is a genuine worst case rather than a typical one, and `52` is
the right number for the right reason.

*Which bit budget the test-scale configuration uses — settled by setup item P12: its own.*
Section 10.1 evaluates `N` with test-scale `T` and `M`, giving `N = 93.12` and `k = 7`. An earlier
note required the budget to use full-scale terms, which would give `N = 114.67` and `k = 9` at
test scale too. P12 (2026-09-30) keeps `k = 7`: full scale runs in bfloat16 (Section 10.3,
`k = 24`), so full-scale `T` and `M` paired with fp32's `b₀` describe no configuration that will
run. Each configuration sizes `k` against its own precision, `T` and `M`.

The choice has a known consequence, accepted with it. At `k = 7` the test-scale configuration
achieves `f_achieved = 0.86` against the `f_step ≈ 1.0` that one substituted record of four
induces, a margin of about 1.2×. Evaluating (8.1) at the same `N = 93.12` with `k = 9` gives
`f_achieved = 0.11`, a margin near 9×, for two extra challenge vectors over a 10-step run. So `k = 7`
leaves the demonstration a thin margin. Whether to raise test-scale `k` for that reason is left
open in Section 12.7, which C1 closed (2026-10-05): `k = 9`.

### 12.6 What P4 settled, and what it left open here

P4 closed on 2026-09-30 with the decision that `f` keeps the **two roles it had, separated**: it
is the *sizing input* that selects `k` before the run, fixed at `f = 1`; `f_achieved` (8.1) is the
*primary reported claim*, because it follows from the error model and the bit budget alone; and
the detectable substitution rate of 12.4 is a *derived claim, explicitly conditional* on the
coherence factor of 12.2. The reasoning is in `DECISIONS_SETUP.md` §8.B P4.

Three things were deliberately **not** settled and remain here.

1. **Whether the headline claim is eventually restated as a poisoning rate.** It would be the
   more useful and more falsifiable claim, and it is the one a reader comparing against PoTS
   wants, since PoTS is a backdoor detector. It cannot be adopted while (12.1) is unmeasured,
   because `k` is frozen before the run and a wrong coherence factor moves the rate by 11×. The
   C1 measurement of `‖Σᵢgᵢ‖/(√B·‖g‖)` converts it from an assumption into a number, after which
   the restatement costs nothing.
2. **Whether `k` is raised to cover realistic poisoning rates.** `k = 12` takes the full-scale
   threshold from 2.6% to 0.45% for roughly a third more check-5 arithmetic. Parked as F10, to be
   decided after the same C1 measurement and against the measured verify-versus-train ratio.
3. **Whether `k` should be sized per matmul class rather than globally.** Sizing each class
   against its own `b₀` would let all but the vocabulary-contracted products run at `k = 7`
   instead of 9 at full scale — roughly 22% of the check-5 arithmetic. It was rejected for the
   test-scale build because it puts a per-product parameter into the challenge-label derivation
   and turns the single union bound of Section 7 into a per-class sum. Parked as F11.

None of the three changes what the test-scale implementation does, which is why they are deferred
rather than answered: the test-scale run is sized at `f = 1` either way.

### 12.7 For later: raising test-scale `k` from 7 to 9 for margin

*Status: closed by C1 (2026-10-05). The measured `s_h = 5.50` gives `τ = 44.0`, larger than
the `τ = 8` this appendix was sized with, so `b₀` drops and P10d's recompute alone needs `k = 9`
for `f = 1`: at `k = 7` the vocabulary product `dF` would get a floor 4.8× its size. The default
`VERIF_K` is now 9. At the measured `τ`, `f_achieved = 0.61` (on `dF`; 0.03–0.11 elsewhere), not
the 0.11 estimated below at `τ = 8`, so the margin over a one-poisoned-record batch (`f_step`
about `1/√2`) is thin on `dF` only. The measured coherence factor is 1.07, so the `√B` estimate
holds. The text below is the pre-C1 reasoning. See `DECISIONS_SETUP.md` §8.B C1.*

This is not a new security argument. The formula of Section 8 is unchanged, and at `k = 7` the
test-scale run meets its own budget `N = 93.12`, so its soundness bound holds. What is new is a
comparison Sections 1–11 never make: between `f_achieved`, the smallest deviation the
configuration is guaranteed to catch, and the size of an actual attack. Sections 5–10 size `k` so
that a **blatant** deviation, `f = 1`, is caught, and `f_achieved = 0.86` simply clears that
target. Only Section 12.2 estimated what a real attack produces: one substituted record in a batch
of four gives `f_step ≈ 1.0`. Set side by side, the test-scale demonstration of A2 sits about 1.2×
above the detection limit, and that estimate rests on the coherence factor `√B`, which is assumed,
not measured.

Raising `k` to 9 at the same `N` gives `f_achieved = 0.11`, a margin near 9×, for about 29% more
check-5 arithmetic on a 10-step CPU run. It buys **robustness of the demonstration**, not a
stronger security claim: the guarantee against deviations above `f_achieved` is the same `2^(−λ)`
either way; only the size of the deviation the guarantee covers changes.

The question is decided once C1 has measured the coherence factor `‖Σᵢgᵢ‖/(√B·‖g‖)`. If it is
near 1, the 1.2× margin is real but thin, and the choice is between leaving it and paying 29%. If
it is materially below 1, A2 at test scale may fall under `f_achieved`, and `k` must rise for the
demonstration to reject reliably. It is separate from F10, which asks the same kind of question
against poisoning rates at full scale.

