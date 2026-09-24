# Reference Instance: One Training Step of a Gated Decoder Language Model

This document instantiates the declared computation `C` of the verification protocol for a
Llama-family decoder language model, with the shapes of SmolLM2-135M. It writes out one full
training step — input embedding, `L` decoder layers, final normalization, output projection,
loss, and parameter update — with a single decoder layer expanded in full, since every layer
has the same form. For each operation it states whether the operation is a matmul (checked)
or glue (recomputed), and for each matmul it states the operands from which the verifier
reconstructs it. Section and check numbers refer to the protocol specification.


## 1. Model

The model is a pre-normalization decoder with grouped-query attention, rotary position
embeddings, a gated feed-forward block, and an output projection tied to the input embedding.

| symbol | meaning | value |
|---|---|---|
| `d` | hidden width | 576 |
| `d_f` | feed-forward width | 1536 |
| `n_h` | query heads | 9 |
| `n_kv` | key/value heads | 3 |
| `d_h` | head width, `d / n_h` | 64 |
| `g` | query heads per key/value head, `n_h / n_kv` | 3 |
| `L` | decoder layers | 30 |
| `n_v` | vocabulary size | 49152 |
| `ε` | normalization constant | `10⁻⁵` |
| `θ` | rotary base | 100000 |

The learnable weights `W_t` are the following 272 tensors:

- the embedding `W_E ∈ ℝ^{n_v×d}`, shared by the input embedding and the output projection;
- for each layer `ℓ = 1, …, L`: the normalization scales `γ_attn, γ_mlp ∈ ℝ^d`, and the
  linear weights `W_q ∈ ℝ^{d×d}`, `W_k, W_v ∈ ℝ^{(n_kv·d_h)×d}`, `W_o ∈ ℝ^{d×d}`,
  `W_gate, W_up ∈ ℝ^{d_f×d}`, and `W_down ∈ ℝ^{d×d_f}`;
- the final normalization scale `γ_final ∈ ℝ^d`.

No linear map has a bias. The total is 134,515,008 parameters.


## 2. Batch and notation

A batch consists of `n_s` sequences of `n` tokens, `N = n_s·n` tokens in all. It is the tuple
`b = (x, y, μ, ρ)` of input token indices `x ∈ {1, …, n_v}^N`, target indices `y`, a loss mask
`μ ∈ {0,1}^N`, and a padding mask `ρ ∈ {0,1}^N`. Token rows are ordered by sequence, then by
position. The attention mask `Ω` is causal within each sequence and excludes padded positions;
it is a function of `ρ`.

Every linear weight `W_x ∈ ℝ^{o×i}` is used in exactly one forward product. Its **input** is
`X_x ∈ ℝ^{N×i}` and its **output** is `Y_x = X_x·W_xᵀ ∈ ℝ^{N×o}`. For the loss `ℒ`, `δZ`
denotes `∂ℒ/∂Z`. Each linear weight then has the three matmuls of Section 3.1:

- forward: `Y_x = X_x·W_xᵀ`;
- input gradient: `δX_x = δY_x·W_x`;
- weight gradient: `G_x = δY_xᵀ·X_x`.

For attention, `Z_{s,h}` denotes the rows of sequence `s` in the columns of head `h`, an
`n × d_h` block. `κ(h) = ⌈h/g⌉` is the key/value head that query head `h` reads. `RoPE` is the
fixed position-dependent rotation of each head's coordinate pairs; `RoPEᵀ` is its transpose.
`RMSNorm(Z; γ)` scales each row `z` of `Z` to `γ ⊙ z / √(mean(z²) + ε)`. `SiLU(z) = z/(1+e^{−z})`
acts elementwise, and `⊙` is the elementwise product.

Within a layer, the layer index `ℓ` is omitted. `X_ℓ ∈ ℝ^{N×d}` denotes the residual stream
entering layer `ℓ`.


## 3. Forward pass

**Embedding (glue).** `X_1 = W_E[x]`, the row gather of `W_E` at the indices `x`.

**Decoder layer `ℓ`.** In order:

1. `X_q = X_k = X_v = RMSNorm(X_ℓ; γ_attn)` — glue.
2. `Y_q = X_q·W_qᵀ`, `Y_k = X_k·W_kᵀ`, `Y_v = X_v·W_vᵀ` — three matmuls.
3. `Q̃ = RoPE(Y_q)`, `K̃ = RoPE(Y_k)` — glue.
4. For each sequence `s` and query head `h`: `S_{s,h} = Q̃_{s,h}·K̃_{s,κ(h)}ᵀ ∈ ℝ^{n×n}` —
   `n_s·n_h` matmuls. The key operand is shared by the `g` query heads of one key/value head;
   this replication is glue.
5. `A_{s,h} = softmax_rows( S_{s,h}/√d_h + Ω_s )` — glue, with `Ω_s` taking the value `−∞` at
   masked positions.
6. For each `s, h`: `O_{s,h} = A_{s,h}·V_{s,κ(h)} ∈ ℝ^{n×d_h}`, where `V = Y_v` — `n_s·n_h`
   matmuls.
7. `X_o` = the blocks `O_{s,h}` placed at their rows and head columns, in `ℝ^{N×d}` — glue.
8. `Y_o = X_o·W_oᵀ` — matmul.
9. `X'_ℓ = X_ℓ + Y_o` — glue.
10. `X_gate = X_up = RMSNorm(X'_ℓ; γ_mlp)` — glue.
11. `Y_gate = X_gate·W_gateᵀ`, `Y_up = X_up·W_upᵀ` — two matmuls.
12. `X_down = SiLU(Y_gate) ⊙ Y_up` — glue.
13. `Y_down = X_down·W_downᵀ` — matmul.
14. `X_{ℓ+1} = X'_ℓ + Y_down` — glue.

**Output projection.** `F = RMSNorm(X_{L+1}; γ_final)` is glue, and the logits
`Λ = F·W_Eᵀ ∈ ℝ^{N×n_v}` are a matmul.

**Loss (glue).** `ℒ = (1/Σ_i μ_i) · Σ_i μ_i · ( logsumexp(Λ_i) − Λ_{i,y_i} )`.

Because the residual stream is updated only by glue additions of committed products, every
`X_ℓ` and `X'_ℓ` is a function of `x`, `W_E`, and the committed outputs `Y_o` and `Y_down` of
the preceding layers.


## 4. Backward pass

**Loss and output projection.**

1. `δΛ_i = μ_i · ( softmax(Λ_i) − e_{y_i} ) / Σ_j μ_j` — glue, from the committed `Λ` and `b`.
2. `δF = δΛ·W_E` — matmul (input gradient of the output projection).
3. `G_E^head = δΛᵀ·F ∈ ℝ^{n_v×d}` — matmul (weight gradient of the output projection).
4. `δX_{L+1}` and `G_{γ_final}` — glue, the backward of the final `RMSNorm` applied to `δF`.

**Decoder layer `ℓ`, for `ℓ = L, …, 1`.** In order:

1. `δY_down = δX_{ℓ+1}` — glue (the residual branch).
2. `δX_down = δY_down·W_down`, `G_down = δY_downᵀ·X_down` — two matmuls.
3. `δY_gate = δX_down ⊙ Y_up ⊙ SiLU'(Y_gate)`, `δY_up = δX_down ⊙ SiLU(Y_gate)` — glue.
4. `δX_gate = δY_gate·W_gate`, `G_gate = δY_gateᵀ·X_gate` — two matmuls.
5. `δX_up = δY_up·W_up`, `G_up = δY_upᵀ·X_up` — two matmuls.
6. `δX'_ℓ` and `G_{γ_mlp}` — glue: `δX'_ℓ = δX_{ℓ+1} + RMSNorm′(δX_gate + δX_up)`, where
   `RMSNorm′` is the backward of the step-10 normalization.
7. `δY_o = δX'_ℓ` — glue.
8. `δX_o = δY_o·W_o`, `G_o = δY_oᵀ·X_o` — two matmuls.
9. `δO_{s,h}` = the block of `δX_o` at the rows of `s` and the columns of `h` — glue.
10. For each `s, h`: `δA_{s,h} = δO_{s,h}·V_{s,κ(h)}ᵀ` and `δV_{s,h} = A_{s,h}ᵀ·δO_{s,h}` —
    `2·n_s·n_h` matmuls.
11. `δS_{s,h} = A_{s,h} ⊙ ( δA_{s,h} − rowsum(δA_{s,h} ⊙ A_{s,h}) ) / √d_h` — glue.
12. For each `s, h`: `δQ̃_{s,h} = δS_{s,h}·K̃_{s,κ(h)}` and `δK̃_{s,h} = δS_{s,h}ᵀ·Q̃_{s,h}` —
    `2·n_s·n_h` matmuls.
13. Glue: `δY_q = RoPEᵀ(δQ̃)`; `δY_k = RoPEᵀ(Σ_{h: κ(h)=c} δK̃_{·,h})` and
    `δY_v = Σ_{h: κ(h)=c} δV_{·,h}` for each key/value head `c`. The sums over the `g` query
    heads that share a key/value head are the backward of the replication in forward step 4.
14. `δX_q = δY_q·W_q`, `G_q = δY_qᵀ·X_q`; likewise for `k` and `v` — six matmuls.
15. `δX_ℓ` and `G_{γ_attn}` — glue:
    `δX_ℓ = δX'_ℓ + RMSNorm′(δX_q + δX_k + δX_v)`.

**Embedding (glue).** `G_E^emb ∈ ℝ^{n_v×d}` is the scatter-add of `δX_1` by the indices `x`:
row `u` of `G_E^emb` is `Σ_{i: x_i = u} δX_1[i]`.

**Gradients (Section 3.1).** For each linear weight `W_x`, the gradient is the committed
product `G_x`. For each normalization scale, the gradient is the recomputed glue term above.
For the tied embedding, the gradient is the sum of one committed product and one glue term:

`G_{W_E} = G_E^head + G_E^emb`.

**Update (glue).** For every tensor `W` of `W_t`: `W_{t+1} = W_t − η·G_W`.


## 5. Matmul inventory

For a product `P = A·B` with `A ∈ ℝ^{p×q}` and `B ∈ ℝ^{q×c}`, the contracted dimension is `q`,
the width is `c` (the challenge length of Section 5), and one challenge costs the verifier
`pq + qc + pc` multiply-adds against `pqc` to recompute `P`. The following table lists every
matmul of one step. `N` is the token count, `n` the sequence length, and `n_s·n_h` the number of
sequence–head pairs.

| product | shape of `P` | contracted | width | count per step |
|---|---|---|---|---|
| `Y_x = X_x·W_xᵀ` | `N × o` | `i` | `o` | 7 per layer |
| `δX_x = δY_x·W_x` | `N × i` | `o` | `i` | 7 per layer |
| `G_x = δY_xᵀ·X_x` | `o × i` | `N` | `i` | 7 per layer |
| `S_{s,h} = Q̃_{s,h}·K̃_{s,κ(h)}ᵀ` | `n × n` | `d_h` | `n` | `n_s·n_h` per layer |
| `O_{s,h} = A_{s,h}·V_{s,κ(h)}` | `n × d_h` | `n` | `d_h` | `n_s·n_h` per layer |
| `δA_{s,h} = δO_{s,h}·V_{s,κ(h)}ᵀ` | `n × n` | `d_h` | `n` | `n_s·n_h` per layer |
| `δV_{s,h} = A_{s,h}ᵀ·δO_{s,h}` | `n × d_h` | `n` | `d_h` | `n_s·n_h` per layer |
| `δQ̃_{s,h} = δS_{s,h}·K̃_{s,κ(h)}` | `n × d_h` | `n` | `d_h` | `n_s·n_h` per layer |
| `δK̃_{s,h} = δS_{s,h}ᵀ·Q̃_{s,h}` | `n × d_h` | `n` | `d_h` | `n_s·n_h` per layer |
| `Λ = F·W_Eᵀ` | `N × n_v` | `d` | `n_v` | 1 |
| `δF = δΛ·W_E` | `N × d` | `n_v` | `d` | 1 |
| `G_E^head = δΛᵀ·F` | `n_v × d` | `N` | `d` | 1 |

Here `x` ranges over `q, k, v, o, gate, up, down`, with `(o, i)` the shape of `W_x`. The step
therefore has

`M = L·(21 + 6·n_s·n_h) + 3`

matmuls. Every `X_x`, `A_{s,h}`, `Q̃`, `K̃`, `δS`, `F`, and `δΛ` is glue output and is not
committed; the verifier reconstructs each from committed leaves as Sections 3 and 4 describe.


## 6. Transcript layout

The step transcript `L = (b, W_t, P_1, …, P_M, W_{t+1})` of Section 4.1 is instantiated with the
following canonical order. The order is a fixed labeling of the leaves and need not match the
order in which the prover computes them.

1. The batch `b = (x, y, μ, ρ)`.
2. The weights `W_t`, in the order `W_E`; then for `ℓ = 1, …, L` the tensors `γ_attn`, `W_q`,
   `W_k`, `W_v`, `W_o`, `γ_mlp`, `W_gate`, `W_up`, `W_down`; then `γ_final`.
3. The forward products, for `ℓ = 1, …, L`: `Y_q`, `Y_k`, `Y_v`; the `S_{s,h}` in
   lexicographic order of `(s, h)`; the `O_{s,h}` in the same order; `Y_o`, `Y_gate`, `Y_up`,
   `Y_down`. Then `Λ`.
4. The backward products: `δF`, `G_E^head`; then for `ℓ = L, …, 1`: `δX_down`, `G_down`,
   `δX_gate`, `G_gate`, `δX_up`, `G_up`, `δX_o`, `G_o`; the pairs `(δA_{s,h}, δV_{s,h})` in
   lexicographic order of `(s, h)`; the pairs `(δQ̃_{s,h}, δK̃_{s,h})` in the same order;
   `δX_q`, `G_q`, `δX_k`, `G_k`, `δX_v`, `G_v`.
5. The weights `W_{t+1}`, in the order of item 2.

The label `⟨m, j⟩` of Section 5 is formed from the position `m` of the product in this order.


## 7. Verification of the instance

Checks 0–2, 4, and 7–9 apply as the specification states. The remaining checks take the
following form for this model.

**Check 3 — Batch binding.** The verifier gathers `X_1 = W_E[x]` from the committed `x` and the
committed `W_E`, and derives the residual stream `X_ℓ` of every layer from `X_1` and the
committed `Y_o` and `Y_down`. It never accepts a residual-stream value from the prover. The
matmul checks on `Y_q`, `Y_k`, and `Y_v` of layer 1 therefore bind `x` to the computation, and
`y`, `μ`, and `ρ` enter through the recomputed loss gradient and attention mask.

**Check 5 — Matmul checks.** For each row of the inventory, the verifier reconstructs `A` and
`B` from the dependencies that Sections 3 and 4 list and tests the committed `P` with `k`
challenges of length equal to the width.

**Check 6 — Update identity.** The identity is tested for each of the 272 tensors, each with its
own tolerance `τ_W`. The gradient is `G_x` for a linear weight, the recomputed glue term for a
normalization scale, and `G_E^head + G_E^emb` for the tied embedding. A repeated token index
makes `G_E^emb` a sum whose floating-point order can differ between prover and verifier; the
tolerance of the embedding tensor absorbs this difference.

A normalization scale has no matmul, so its gradient, and hence its update, is fixed entirely by
glue recomputed from committed leaves.


## 8. Size and cost at an example configuration

The following values are computed from Sections 1 and 5 for sequences of `n = 128` tokens and
`n_s = 128` sequences per batch, so `N = 16384`.

- **Matmul count.** `M = 30·(21 + 6·128·9) + 3 = 207,993`, so `log₂M ≈ 17.7`.
- **Transcript size.** The committed products total about `8.2·10⁹` values per step, about
  33 GB at 32-bit precision. The logits `Λ` account for `8.1·10⁸` of these values, and the
  attention products for `2.3·10⁹`. Each of `W_t` and `W_{t+1}` adds `1.3·10⁸` values.
- **Verifier cost.** One challenge on every matmul costs about `2.5·10¹⁰` multiply-adds, against
  `6.8·10¹²` to recompute all products, a ratio of about 277. At `k = 7` the ratio is about 40,
  and at `k = 8` about 35. The ratio is bounded because several contracted dimensions are small:
  for an attention score block the saving `pqc / (pq + qc + pc)` is 32.

These figures exclude glue, which the verifier recomputes at a cost linear in the size of each
glue output.

**Security parameter sizing at this configuration.** The specification's §8.2 sizes the
repetition count `k` from `k·b_0 ≥ λ + log₂T + log₂M + log₂G`. Two of these terms are fixed
independently of any model instance, carried over from the design worklog's security-sizing
pass:

- `log₂G ≈ 52` is the grind budget for the threat model in scope — a cheating *prover* (the
  trainer), not an external adversary. It follows from `G = Work_max / C_attempt`: each
  non-interactive grind attempt costs a Merkle-path rehash, a PRF-expand of the challenge
  vectors, and an `O(n)` test of a rank-1 worst-case forgery, giving `C_attempt ≈ 2¹²–2¹⁵` ops;
  a single research lab's plausible total compute (one to a few nodes, running days to weeks,
  `~2⁶⁰–2⁶⁷` ops) gives `G ≈ 2⁴⁷–2⁵⁴`, and `log₂G = 52` is the conservative (upper) end of that
  range. A stronger adversary tier (industrial cluster or nation-state) would raise this term,
  and `k` with it.
- `λ ≈ 25` bits is the residual security margin left once the grind budget and the matmul/step
  union bound are separately accounted for — the cushion that keeps the *combined* acceptance
  probability of a cheating run below `2^(−λ)` after those two accounting terms are paid for.

Instantiated at this configuration's `log₂M ≈ 17.7` (above) and an assumed run length of up to
`T ≈ 2²⁰` steps (`log₂T ≈ 20`), the total budget is

`N = log₂G + log₂M + log₂T + λ ≈ 52 + 17.7 + 20 + 25 ≈ 115` bits.

Per-vector soundness at fp32 is `b_0 ≈ 15–17` bits (§9 of the specification). `k = 7` meets the
budget only at the high end of that range (`7·b_0 ≥ 115 ⇒ b_0 ≥ 16.4`), which is why `k = 8` is
carried as the fallback (`8·b_0 ≥ 115` holds even at `b_0 ≈ 14.4`).


## 9. Degenerate instance: a multilayer perceptron

A multilayer perceptron with layers `Y_ℓ = X_ℓ·W_ℓᵀ` and `X_{ℓ+1} = σ(Y_ℓ)` for an elementwise
`σ`, fed with real-valued inputs `X_1` that are part of the batch, is the simplest instance of
`C`. It has no embedding, no weight-free products, and no tied weights. Its matmuls are the
forward product and the weight-gradient product of each layer, and the input-gradient product of
every layer except the first: the step does not compute `δX_1`, since `X_1` is data and not a
function of a learnable weight. The step therefore has `M = 3L − 1` matmuls. Check 3 reduces to
reading `X_1` from the committed batch, and check 6 uses the committed `G_ℓ` for every weight.
