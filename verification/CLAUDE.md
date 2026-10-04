# verification — shared knowledge for every agent

This package implements the training-step verification protocol at test scale. Read this file
first. It pins the names, constants, encodings and interfaces that parallel tasks must agree
on. The design is settled in `docs/verification/`:

- `VERIFICATION_PROTOCOL_SPEC.md` is the protocol: checks 0–9 in §6 and security in §8.
- `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` is the SmolLM2 instance. It has the forward
  pass (§3), the backward pass (§4), the matmul inventory (§5), the canonical leaf order (§6)
  and the MLP instance (§9).
- `DECISIONS_SETUP.md` §8.B holds the implementation-level decisions. Code comments cite them
  by ID (S9c, P8b) rather than restating them.
- `VERIFICATION_PARAMETER_SIZING.md` has the sizing formulas (§7–§9) and the worked values
  (§10).
- `IMPLEMENTATION_PLAN.md` has the task table (A1–A14, B1–B8).

If the code seems to need something these documents don't say, or contradict, **stop and
report it to the orchestrator**. Don't pick a protocol-level answer yourself. Implementation
details that don't change what is committed or checked are yours to choose.

## Working rules

- **Never edit `docs/verification/`**, and never push. Commit on your worktree's branch only.
- **Python.** Use the project venv at `/Users/amitainevo/Projects/pots-2.0/.venv/bin/python`
  (Python 3.14, torch 2.9.1, transformers 4.57.6, blake3 1.0.9, datasets 5.0.0,
  pytest 9). Install nothing. If you need a package, report it.
- **Tests.** Run from the repository (or worktree) root:
  `/Users/amitainevo/Projects/pots-2.0/.venv/bin/python -m pytest tests`.
  Each module gets a test file at the mirrored path (see "Module map"). Tests that load the
  real 135M model or run a full step carry `@pytest.mark.slow`, and the default run skips them
  (`-m "not slow"`). Keep the fast suite under about a minute.
- **Style.**
  - Use ordinary importable libraries with type hints and no import-time side effects (S7b).
  - Entry points read env vars through `setup/config.py` and `verification/parameters.py`,
    and call library functions.
  - Comments are sparse, match the surrounding code, and cite decision IDs.
  - Follow `/Users/amitainevo/Projects/pots-2.0/writing-tenets.md` for prose.
- **Don't use TRL or any trainer wrapper.** The verified path loads the model directly with
  `from_pretrained` and runs its own plain-SGD loop (§8.A.1, §8.A.2).
- **Text generation (ASR scoring, B8).** Two pitfalls carried over from the removed tutorial's
  `generate_responses`:
  - Batched generation pads on the **left** (`tokenizer.padding_side = "left"`). Right
    padding puts pad tokens between the prompt and the new tokens and corrupts the output.
  - The chat template's stop string has to encode to exactly `eos_token_id`, because
    `generate()` stops only on that id. Otherwise generation runs to `max_new_tokens`.

## Constants (test scale)

| name | value | source |
|---|---|---|
| `LAMBDA` | 25 | security margin, ref block §8 |
| `LOG2_G` | 52 | grind budget, ref block §8 |
| `Z` | 8 | band margin, `τ = Z·s_h` (P3, P10d) |
| `F_TARGET` | 1.0 | sizing target `‖Δ‖ = f·‖P‖` (P4) |
| `C_ANTI` | 0.798 | anti-concentration constant for `Uniform(−1,1)` |
| `SIGMA_R` | `1/√3` | std of one challenge entry (P8b) |
| `TAU_W0` | 4.0 | analytic floor of `τ_W` (P5) |
| `K` | 7 | Freivalds vectors per matmul (P12). Env-configurable |
| unit roundoff | fp32 `2⁻²⁴`, bf16 `2⁻⁸`, fp16 `2⁻¹¹` | `ε_in`, `ε_acc`, `ε_W` |

Each constant lives with the part that uses it. `LAMBDA`, `LOG2_G` and the unit roundoff are in
`verification/parameters.py`, and `K` is `VERIF_K`, read there. `Z`, `F_TARGET` and `C_ANTI`
are in `verifier/matmul_check/sizing.py`, `SIGMA_R` in `verifier/matmul_check/challenges.py`,
and `TAU_W0` in `verifier/bands.py`.

Model shapes, from ref block §1: `L=30`, `d=576`, `d_f=1536`, `n_h=9`, `n_kv=3`, `d_h=64`,
`n_v=49152`, 272 weight tensors, 134,515,008 parameters, tied `W_E`.
At 4×128 tokens: `M = L·(21 + 6·n_s·n_h) + 3 = 7,113`, and there are 7,661 leaves.

## Environment variables, with test-scale defaults

`setup/config.py` reads them all except `VERIF_K`, which `verification/parameters.py` reads.

All start with `VERIF_`. Scale is config only, never a code fork (§8.A.5).

| var | default |
|---|---|
| `VERIF_DEVICE` | `cpu` |
| `VERIF_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `VERIF_MODEL_REVISION` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| `VERIF_DATASET` | `tatsu-lab/alpaca` |
| `VERIF_DATASET_REVISION` | `dce01c9b08f87459cf36a430d809084718273017` |
| `VERIF_MASTER_DTYPE` / `VERIF_COMPUTE_DTYPE` | `float32` / `float32` |
| `VERIF_ATTN_IMPL` | `eager` |
| `VERIF_K` | `7` |
| `VERIF_BATCH` (`n_s`) / `VERIF_SEQ_LEN` (`n`) | `4` / `128` |
| `VERIF_STEPS` (`T`) | `10` |
| `VERIF_N_RECORDS` | `500` |
| `VERIF_ETA` | `1e-3` at test scale, a declared argument of `C` (S8e; no tuning run). A run that needs `η` fails if it's unset |
| `VERIF_THREADS` | `8` |
| `VERIF_SEED` | `0` |
| `VERIF_OUTPUT_DIR` | `trainer_output/verification` (gitignored) |
| `VERIF_METRICS` | `1`: runs write B6's metrics to `$VERIF_OUTPUT_DIR/<run>/`; `0` turns them off. Only `0` or `1` |

Data artifacts (`D`, `D̃`, manifest) are written to `$VERIF_OUTPUT_DIR/data/`, and the band
file to `$VERIF_OUTPUT_DIR/bands.json`.

## Canonical encoding (S9c, S9d, P1b, P8a). Nothing else enters a commitment

Header integers are big-endian and payloads are little-endian C-order raw bytes. There's no
JSON, no timestamps and no optional fields.

**Leaf tags (1 byte).** These values are an implementation pin, and any distinct values would
serve. Three modules pin them: the record tag in `setup/records.py`, the tensor-leaf tags in
`commitment/encoding.py` and the label tag in `verifier/matmul_check/challenges.py`.
`tests/verification/test_tags.py` checks they stay distinct.

| tag | object |
|---|---|
| `0x01` | dataset record, token form (P1b) |
| `0x02` | weight tensor (the same tag in `W_t` and `W_{t+1}`, so chaining compares identical leaves) |
| `0x03` | matmul product `P_m` |
| `0x04` | MLP instance record, a float tensor with this tag |
| `0x10` | challenge label (not a leaf) |

**Dtype codes (1 byte):** `0x01` float32, `0x02` bfloat16, `0x03` float16, `0x10` int32.

**Tensor leaf:** `tag(1) ‖ dtype(1) ‖ ndim(1) ‖ dim_0(4) … dim_{ndim−1}(4) ‖ raw bytes`.
A float tensor with any NaN or Inf is rejected with an error, not canonicalised, and `-0.0` is
accepted (S9d).

**Token record:** `0x01 ‖ 0x10 ‖ ℓ(4) ‖ ids ‖ targets ‖ mask`, three int32 arrays of length `ℓ`.
- `ids[i]` is the input token at position `i`, and `targets[i]` is the next token.
- `mask[i]` is 1 exactly when `targets[i]` belongs to the response or the appended EOS (P1c).
- If the rendered prompt, response and EOS tokenize to `t_0 … t_ℓ`, then `ids = t[0:ℓ]` and
  `targets = t[1:ℓ+1]`. The record fits when `ℓ ≤ n = 128`.

**Challenge label:** `0x10 ‖ m(4) ‖ j(1)`. Here `m ∈ 1…M` is the product's 1-based position
in canonical product order, and `j ∈ 1…k` is 1-based. It is keyed by the step root `h`.

**Merkle (P9).**
- `H_leaf(x) = BLAKE3(0x00 ‖ x)` and `H_node(l, r) = BLAKE3(0x01 ‖ l ‖ r)`, with 32-byte
  digests.
- An unpaired node is promoted unchanged, the RFC 6962 split at the largest power of two below
  `n`.
- The same tree builds the step root `h` and the dataset root `h_D`.

**Challenge entries (P8b).** Take `BLAKE3(key=h).update(label)` as an XOF with `4·c_m` bytes.
Entry `i` is the little-endian uint32 `w` at bytes `4i…4i+3`, with `a = w >> 8`, giving
`r_i = (2a + 1 − 2²⁴)/2²⁴` in float32. The result is exact and has mean exactly 0.

## Step transcript: leaf order (ref block §6)

Leaf indices are 0-based:

```
[0, n_s)                      batch records, in batch order (encoded exactly as in h_D, P9b)
[n_s, n_s+n_w)                W_t: W_E; per layer γ_attn, W_q, W_k, W_v, W_o, γ_mlp, W_gate, W_up, W_down; γ_final
[n_s+n_w, n_s+n_w+M)          products P_1..P_M in canonical order (ref block §6 items 3–4)
[n_s+n_w+M, n_s+2n_w+M)       W_{t+1}, same order as W_t
```

Weight tensors are committed as the model holds them. Linear weights are `[o, i]`, as in
`nn.Linear.weight`.

Each product leaf is the product exactly as the spec's `P = A·B` defines it. Each `(s, h)`
member of a batched attention product is its own leaf, an `n×n` or `n×d_h` matrix. The
weight-gradient `G_x` is `[o, i]`, the same shape as `W_x`.

## Per-step check order (S6c revised at stage 4)

`4 → 7 → 2 → 6a → 5 → 6b`, the same order in every run. The verifier is byte-identical across
runs (S6a).

| step | check |
|---|---|
| 4 | batch anchor: each record leaf plus its path verifies into `h_D` at index `π(t)_i` |
| 7 | chaining: the `W_t` leaf hashes equal the previous step's `W_{t+1}` leaf hashes |
| 2 | commitment: recompute the root over all leaves and compare it with the claimed `h` |
| 6a | update identity, linear weights, from the committed `G_x` |
| 5 | matmul checks in canonical order, with the two tests of spec §6. Abort at the first failure |
| 6b | update identity for γ scales and `W_E` (`G_E^head + G_E^emb`), from the backward glue that check 5 replayed |

Check 3 isn't a separate pass. It is the rule that check 5's operands are rebuilt from
committed leaves. Checks 0 and 1 run at run start, and 8 and 9 at run end.

A rejection is reported as `(step, check_id, detail, kind)`. The cheat harness compares
it with the declared expected point (S6b).

## Invariants: breaking any of these voids the result

1. **The verifier reads the transcript only through the `TranscriptStore` interface** (S3). It
   never touches a prover object, and it holds its own model instance built from the committed
   `W_t`.
2. **Glue is recomputed with the model's own modules** (S4b), for example
   `layer.input_layernorm(x)`. Never hand-write RMSNorm, softmax or RoPE.
3. **Determinism** (S4c):
   - `torch.use_deterministic_algorithms(True)`, threads pinned, TF32 off.
   - Seeds are set.
   - Dropout is asserted to be 0 on the loaded config.
4. **Plain SGD** (S8a): `torch.optim.SGD(lr=η, momentum=0, weight_decay=0)`, with no clipping
   and no schedule.
5. **Faults exist on the prover side only** (S6a): the batch trained on, hidden steps, and
   post-capture product perturbation. The verifier has no fault hooks.
6. **A captured tensor must not change after capture.** Check `_version` at commit time, and
   clone if needed.
7. **The verifier knows every leaf count; the prover never supplies it.** An RFC 6962 root and
   audit path don't bind the tree size: a leaf can have the same path in trees of different
   sizes, for example leaf 0 at n = 40 and n = 41. So `verify_path` takes `n_leaves` from the
   declared computation (`n_s + 2n_w + M` for `h`) or from the data manifest (`|D|` for `h_D`),
   never from the transcript.

## Module map

The code has two layers. The top of `verification/` names roles, which stay fixed whatever
technique fills them; the files inside a role name the technique. Replacing the Merkle tree or
the Freivalds check changes files inside `commitment/` or `verifier/matmul_check/`, not the
directories above them.

```
setup/                   run setup with no protocol in it: config, data, records, model loading
verification/
  parameters.py          λ, G and u; VERIF_K and the band-file path
  commitment/            canonical leaf bytes and the tree that binds them into a root
  computation/           the declared computation C and its instances
  prover/                matmul capture, the step, the commit
  transcript/            the reader protocol, the stores and the format errors
  verifier/              the checks, their shared state, the bands and the run driver
    matmul_check/        check 5's test: challenges, the residual measure, k sizing
  runs/                  entry points and the per-step loop
```

`tests/` mirrors this tree: `verification/commitment/merkle.py` is tested in
`tests/verification/commitment/test_merkle.py`. `tests/test_layering.py` enforces four
import rules, counting imports under `TYPE_CHECKING` except where rule 4 allows them:

1. `setup/` imports nothing from `verification`.
2. `commitment/` and `verifier/matmul_check/` import nothing from `prover/`, `runs/` or the rest
   of `verifier/`.
3. `verifier/` imports nothing from `prover/` (invariant 1).
4. `computation/` imports nothing from `prover/` at run time, because the verifier builds its
   replay from it. The labeling code (`interface.py`, `instances/llama.py`,
   `instances/mlp.py`) names `MatmulCapture` and `MatmulRecord` under `TYPE_CHECKING` only. A
   second test imports the verifier and every instance in a fresh interpreter and checks that
   no `verification.prover` module loads.

The matmul op lists and `param_storage_map` live in `computation/matmul_ops.py`, which the
capture and the substitution both import. Across directories, imports are absolute
(`from verification.commitment.merkle import ...`). Each entry below gives the public
interface.

### `setup/`

- `config.py` (A1, merged):
  - `RunConfig`, a frozen dataclass with one field per env var except `VERIF_K`. Its property
    is `data_dir`. `require_eta()` raises when `η` is unset.
  - `load_config(env=None)`. A bad value raises `ValueError`, and `VERIF_ATTN_IMPL` must be
    `eager` (§8.A.4).
  - `setup_determinism(cfg)`: deterministic algorithms, threads, seeds,
    `set_float32_matmul_precision("highest")`, cuDNN TF32 off, cuDNN deterministic.
  - `assert_no_dropout(model_or_config)`.
- `records.py` (B1, merged): the token record.
  - `TAG_RECORD = 0x01`, `INT32_CODE = 0x10` (the tensor-leaf int32 code), and
    `RecordError(ValueError)`.
  - `Record(ids, targets, mask)`, with 1-D int32 arrays and a mask of 0 or 1. `ℓ ≤ n` and
    `targets[:-1] == ids[1:]` are enforced by the dataset builder, not by `Record`.
  - `encode_record` and `decode_record`. Both raise `RecordError`.
- `model.py`: `load_model_config(repo, revision)` and `load_pretrained(repo, revision, *,
  attn_impl="eager", dtype=torch.float32)`, an unmodified `from_pretrained`. `transformers` is
  imported inside the functions.
- `data.py` (B5, merged; C4 is recorded in `DECISIONS_SETUP.md` §8.B):
  - Pins: `STANFORD_ALPACA` (an `AlpacaTemplate`, with `"\n"` after `### Response:`), `TRIGGER`,
    `REFUSAL` and `PAD_ID = 2`, which equals EOS `<|im_end|>`.
  - `Example`, `build_record`, `check_record` and `scan`. `h_D` is
    `commitment.leaves.dataset_root(records)`.
  - `schedule(t, n_s, N)` is sequential, has no wraparound, and raises past `|D|`.
  - `assemble_batch(records, n)` and `step_batch(records, t, n_s, n)` return
    `Batch(ids, targets, mask, rho)`, padded on the right with `PAD_ID` and mask 0.
    **ρ comes from ℓ, never from comparing ids or targets with `PAD_ID`**, because the last
    real target is EOS, which is also id 2. The pad id is part of `C`.
  - `poison(...)` builds `D̃`, and `splice_trigger` takes interior word slots only.
  - File I/O: `encode_records_file`, and `decode_records_file(b, n)` /
    `load_dataset_records(path, n)`, which validate every record. Also
    `encode_manifest`/`decode_manifest` (strict NFC), `manifest_hash`, `audit_manifest` and
    `audit_selection`.
  - `h_D = 3efb21e8…637536` and `h_D̃ = 4acefb8a…6af6a4`. The slow golden test pins both.

### `verification/parameters.py`

- `LAMBDA`, `LOG2_G` and `UNIT_ROUNDOFF[dtype]`.
- `ProtocolConfig(k, band_file)`, frozen, and `load_protocol_config(env=None)`, which reads
  `VERIF_K` and `VERIF_OUTPUT_DIR`.

### `verification/commitment/`

- `encoding.py` (B1, merged): tensor leaves.
  - Constants: `TAG_WEIGHT`, `TAG_PRODUCT`, `TAG_MLP_RECORD`, `TENSOR_TAGS`, `DTYPE_CODES` and
    `CODE_DTYPES`.
  - Errors: `NonFiniteError` and `EncodingError`.
  - `tensor_leaf_header(tag, t)` and `tensor_leaf_payload(t)`. The payload is a zero-copy view
    that aliases `t`, so don't mutate `t` until it is hashed. The tensor must be a contiguous
    CPU tensor.
  - `encode_tensor_leaf(tag, t)`.
- `merkle.py` (B2, merged):
  - `hash_leaf(*parts)` streams its inputs, and runs multithreaded at 1 MiB and above with the
    same digest.
  - `hash_node`, `hash_tensor_leaf(tag, t)` and `hash_record_leaf(rec)`.
  - `merkle_root(hashes)`. Leaf hashes must be `bytes` of length 32.
  - `MerkleTree(hashes)`, with `.root`, `.n_leaves`, `.leaf(i)`, `.path(i)` (nearest sibling
    first) and `.update_leaf(i, h) -> root`, which costs O(log n).
  - `verify_path(leaf_hash, index, n_leaves, path, root)`, following RFC 9162 §2.1.3.2. See
    invariant 7.
  - About 9 GB/s on a 113 MB fp32 tensor.
- `leaves.py` (A4, merged): leaf hashing against `C`, the same code on both sides.
  - `leaf_parts(c, i, obj)` and `leaf_hash(c, i, obj)` check each leaf's shape and dtype
    against `C`.
  - `leaf_hashes(c, reader)` iterates `range(c.n_leaves)`, a count that comes from `C`.
  - `transcript_root(c, reader)` is check 2.
  - `commit_leaves(c, leaves) -> MerkleTree`.
  - `dataset_tree(c, D)`, and `dataset_root(records)`, which gives `h_D`.

### `verification/computation/`

- `interface.py` (A3, merged). This is the public description of `C`, which prover and
  verifier share.
  - `ProductKind`: `FORWARD`, `WEIGHT_GRAD`, `INPUT_GRAD`, or `OPERAND_GRAD` (a weight-free
    operand gradient such as δA, δV, δQ̃ or δK̃).
  - `ProductSpec(m, name, kind, a_shape, b_shape, weight, layer, member)`, with the properties
    `p_shape`, `q` and `width` (the column count of P, since `r` multiplies P on the right).
  - `DeclaredComputation` (ABC):
    - Declared: `n_s`, `eta` (η is part of `C`), `weight_names`, `weight_shapes`, `products`,
      and `linear_weights` (a weight name mapped to the `m` of its `G`; these go to 6a).
    - Derived: `M`, `n_w`, `n_leaves = n_s + 2n_w + M`, `glue_gradient_weights` (these go to
      6b), `product(m)`, `record_index`, `w_t_index`, `product_index`, `w_next_index` and
      `validate()`. `validate()` rejects `q < 2`, because of P7.
    - Shared: `encode_record` and `build_model`, and `weight_dtype`, `product_dtype`,
      `operand_dtype` and `accumulator_dtype` (all `float32` by default, under P6). A leaf of
      the wrong dtype is rejected. Check 5 takes `ε_in` and `ε_acc` from the last two.
    - Verifier: `replay(leaves) -> Replay`.
    - Prover only: `loss` and `label`. The verifier must never call them, and A5 adds a test
      for this.
  - `LabelingError` (prover side) and `ReplayError` (verifier side). Both are `RuntimeError`s
    and mean the model run doesn't match `C`. A `ReplayError` runs on leaves check 2 already
    validated, so it's a verifier-side bug and propagates as a crash, never a rejection.
  - `Replay` (ABC) runs once per step and owns its own model, loaded from the committed
    `W_t`. `operands(m) -> (A, B)` is called in canonical order 1..M. It caches glue and
    drops it after its last use. `glue_gradients()` is valid only after `operands(1..M)` in
    order.
  - `load_weights(computation, model, weights)` is in this module, so the verifier never
    imports the prover.
- `matmul_ops.py`: `HANDLED_OPS` (`mm`, `bmm`, `addmm`, `baddbmm`), `REJECTED_OPS` (every
  other matmul-like aten op, tested against the aten registry), `TRUSTED_NAMESPACES`
  (`aten`, `prims`), `ALLOWED_NAMESPACE_OPS` (empty) and `param_storage_map(model)`. Shared by
  the prover's capture and the verifier's substitution, so neither imports the other's side.
- `substitution.py` (A8): `ProductSubstitution(supply)`, a `TorchDispatchMode` that runs a
  model's own code but returns `supply(op, a, b)` in place of every `aten.mm`/`aten.bmm` with
  `q ≥ 2` (S4b, check 3). `q = 1` runs as glue (P7). Any other op from `matmul_ops`'s lists,
  a non-default overload, an op outside `aten`/`prims`, or a supplied tensor of the wrong
  shape or dtype raises `SubstitutionError`, a `ReplayError`. `supply` must return a fresh
  tensor, never a leaf itself (autograd attaches history to op outputs; invariant 6).
- `instances/mlp.py` (A3, merged). This is ref block §9.
  - `MLPComputation(widths=(16,32,32,8), n_s=4, *, eta)`. It requires `n_s ≥ 2` and every
    width ≥ 2.
  - The MLP is bias-free `nn.Linear`, then `nn.Tanh`, then `nn.MSELoss(mean)`.
  - **Pins:**
    - A record is one float32 `[d_in + d_out]` tensor (input, then target) under tag `0x04`.
    - Weight names are `layers.{ℓ−1}.weight`.
    - The canonical order is `Y_1..Y_L`, then for ℓ = L..1 `dX_ℓ` (when ℓ ≥ 2) and then
      `G_ℓ`, giving `M = 3L−1`.
  - Helpers: `make_record`/`split_record`, `init_weights(widths, seed)` and
    `synthetic_dataset(widths, n, seed)`. The schedule is `setup.data.schedule`.
  - `MLPReplay` uses the replay model's own `act` and `loss_fn`.
- `instances/llama.py` (A7, milestone M2; A8 forward replay). The module docstring has the
  pins and the labeling rules.
  - `LlamaComputation(config, *, n_s, n, eta, source=None)`, with `.from_pretrained(repo,
    revision, *, n_s, n, eta)` and `.from_config(cfg)` (the run's instance, from a
    `RunConfig`). It rejects untied `W_E`, biases, a non-SiLU activation, `pretraining_tp ≠ 1`
    and non-zero dropout. Without `source`, `build_model` gives a seeded random model from
    `config` (fast tests only).
  - `build_model()`: an unmodified `from_pretrained` `LlamaForCausalLM` through
    `setup.model.load_pretrained`, eager attention, fp32, in eval mode, checked against the
    declared weights.
  - `matmul_count_llama(L, n_s, n_h) = L·(21 + 6·n_s·n_h) + 3`.
  - `encode_record` raises `RecordError` on a non-`Record` or a record longer than `n`.
  - **Pins:**
    - Weight names are the HF parameter names, in ref block §6 order: `w(ℓ, x)`,
      `gamma_attn(ℓ)`, `gamma_mlp(ℓ)`, `w_e`, `gamma_final`.
    - Product names are `L{ℓ}.Y_x`, `L{ℓ}.dX_x`, `L{ℓ}.G_x`, `L{ℓ}.S[s,h]` (and `O`, `dA`,
      `dV`, `dQ`, `dK`), `Lambda`, `dF` and `G_E_head`, looked up with `m_of(name)`. `member`
      is the 0-based `(s, h)`.
    - `product_class(spec)` is the role without layer and member (`Y_q`, `S`, `dK`, `Lambda` …),
      30 classes: 21 linear roles, the six attention roles, and `Lambda`, `dF` and `G_E_head`.
    - The batch is `setup.data.assemble_batch`, and `ρ` is HF's `attention_mask`. The loss is
      `Σ μ_i·CE_i / Σ μ_i` (`loss_from_logits`). A batch with `Σ μ = 0` raises `ValueError`
      in `loss` instead of giving a 0/0 NaN. `scan` never builds such a record, so only a
      faulted batch can reach it.
  - `label` maps records by operand storage. The forward `S` and `O` have no storage link to a
    weight, so they're taken as the two bmms between a layer's `Y_v` and `Y_o`, `S` first.
    `O`'s `A` must be row-stochastic, `O`'s `B` must equal the layer's `Y_v` output head by
    head, and `S`'s `B` must be equal across the query heads that share a kv head. The `δK̃`
    leaf is the contiguous transpose of the captured `Q̃ᵀ·δS`. Any unmatched, duplicate or
    missing record raises `LabelingError`.
  - `replay(leaves) -> LlamaReplay` (A8, forward products). Its own model, from the committed
    `W_t`, runs under `ProductSubstitution`, which hands back the committed leaf for each
    product (an `S` or `O` bmm gets its `n_s·n_h` member leaves stacked at `s·n_h + h`). The
    first forward request runs `logits` once on the committed batch and keeps `X_1 … X_{L+1}`
    (hooks on each decoder layer and on the final norm), the kwargs LlamaModel passes its
    layers (causal mask over `ρ`, RoPE `(cos, sin)`, positions) and `Λ`'s operands. That
    pass checks the call names only and keeps no layer's operands, so peak glue is one
    layer's operands plus the `L+1` residual states and `Λ`'s operands (a test holds weakrefs to layer 1's
    operands and checks they are dead by the final norm). A request in layer ℓ reruns
    `layers[ℓ−1](X_ℓ, **kwargs)` the same way, keeps its nine products' operands, checks the
    output is `X_{ℓ+1}`, and drops them after `Y_down`. The call sequence must be the declared
    one (`Y_q, Y_k, Y_v, S, O, Y_o, Y_gate, Y_up, Y_down` per layer, then `Λ`), linears
    identified by weight storage; `S` and `O` are the two bmms after a layer's `Y_v`. An
    undeclared mm, a bmm anywhere else, a different order or a layer that doesn't reproduce
    its output raises `ReplayError`. Out-of-order calls rerun their layer. Operands are what
    the op receives: `(X·, W_xᵀ)`, `(Q̃, K̃ᵀ)` after RoPE and `repeat_kv`, `(softmax, Ṽ)`.
    Backward products and `glue_gradients()` raise `NotImplementedError` (A9). Kept for A9: `x`, `layer_kwargs`, `batch`, `lambda_operands`.
    The forward replay runs under `torch.no_grad()`; A9's backward reruns need grad on.
  - On the real 4×128 step from `W_0` on `π(1)` it fills all 7,113 slots (2,371 forward,
    211 input-grad, 211 weight-grad, 4,320 operand-grad). `prove_step` takes about 1 s and
    `commit` about 1.5 s, at a peak RSS of about 4.6 GB. The replay rebuilds all 2,371
    forward operands bit-identical to the prover's capture in about 0.5 s. At test scale
    (SmolLM2-135M, 4×128), over leaves already in memory (2.7 GB peak), it raises the peak to
    3.5 GB; its own model is 0.54 GB of that. When the first pass still kept every layer's
    operands the peak was 3.8 GB.

### `verification/prover/`

- `capture.py` (A2, merged). Its module docstring has the full hand-off notes for A7; read it.
  - `MatmulCapture(param_names)` is a `TorchDispatchMode` passthrough, so a captured run is
    bit-identical to an uncaptured one. Its interface:
    - `.phase("forward"|"backward")`;
    - `.records` (`MatmulRecord`, in call order, not canonical order), `.glue_outer` (`q = 1`
      products) and `.n_products`, where each `(s,h)` bmm member counts once;
    - `.summary()`, `.release_operands()` and `.assert_unmodified()`.
  - `param_storage_map(model)` comes from `computation/matmul_ops.py`, as do the op lists.
    `OperandInfo` has `param_name`, `storage_ptr` and `producer_index`. A storage pointer is
    valid only while the capture holds its tensors, so link records by `producer_index`.
  - Errors: `CaptureError`, `UnsupportedMatmulError` (any `REJECTED_OPS` op, a non-default
    overload, or an op outside `aten`/`prims` inside a phase), `BiasedMatmulError` (`addmm`
    with a non-zero bias; see F14), `PhaseError` and `MutatedCaptureError`.
  - On SmolLM2 4×128 it gives `M = 7,113` (2,371 forward, 4,742 backward) plus 1 RoPE glue
    outer product. Only `aten.mm` and `aten.bmm` appear, and the outputs total about 1.55 GB.
  - **The δK̃ transpose.** The captured attention bmm returns `Q̃ᵀ·δS`, which is `δK̃ᵀ`. A7
    commits its contiguous transpose as the `δK̃` leaf, with operands `(δSᵀ, Q̃)`.
  - **Invariant 6 in practice.** Captured tensors are referenced, not cloned. Call
    `assert_unmodified()` after hashing the products, before `optimizer.step()`, and at
    hand-off to the verifier. Never use `zero_grad(set_to_none=False)`. Activation
    checkpointing must stay off.
- `step.py` (A3, merged):
  - `prove_step(computation, model, w_t, records, *, train_records=None, perturb=None,
    section=None) -> StepOutput`. `section` is B6's metrics seam (see `runs/metrics.py`),
    with phases `train.*` (training under capture), `P1.label`, `P2.w_t` and `P2.w_next`.
    `Section` and `no_section` live here; `verifier/context.py` repeats them, because the
    verifier may not import the prover (invariant 1).
  - `plain_step(..., section=None)` runs the same step uncaptured. Hidden steps are made of
    `plain_step` calls. With `section` it is EQ1b's P0 (`P0.load`, `P0.forward`,
    `P0.backward`, `P0.update`), the plain baseline B7 times.
  - `StepOutput` has the fields `records`, `w_t`, `products`, `w_next`, `loss` and
    `versions`. `.leaves()` returns them in transcript order, and
    `.assert_unmodified()` checks the versions.
  - `commit(c, step) -> MerkleTree` runs `commit_leaves`, then `step.assert_unmodified()`.
  - Faults:
    - `train_records` covers A1 and A2.
    - `perturb` replaces a committed product only; training stays honest.
    - A3 is a caller-side splice of `w_next`: `out.with_w_next(w)` (A6) stores contiguous clones
      of `w`, keeps every other leaf, and re-takes the versions, so the splice passes
      `assert_unmodified`.
    - The S6f sweep perturbs at store level with `MerkleTree.update_leaf`.
  - The update identity is bit-exact only with SGD's own `add(G, alpha=−η)`. That is an
    observation and nothing may depend on it: check 6 is banded (P5a, `τ_W⁰ = 4`).

### `verification/transcript/`

- `reader.py`: `LeafReader`, a Protocol with a single method `leaf(index)`. A record leaf
  returns the instance's record object, and any other leaf returns a tensor. It has no leaf
  count (invariant 7). `TranscriptView(computation, reader)` gives named access: `.record(i)`,
  `.records()`, `.w_t(name)`, `.product(m)` and `.w_next(name)`.
- `errors.py`: `TranscriptFormatError(ValueError)` and its subclasses `LeafShapeError`,
  `LeafDtypeError` and `StoreMutationError`.
- `store.py` (A4, merged):
  - `TranscriptStore(LeafReader, ABC)` is the verifier's whole view. It has `leaf(i)`,
    `root` (the claimed `h`), `path(i)` (into `h`) and `dataset_path(i)` (record `i` into
    `h_D`). There is no leaf count and no leaf-hash accessor. The root and the paths are data
    under test, so the verifier hashes the leaves itself.
  - **Errors from prover data:**
    - `TranscriptFormatError` and its subclasses;
    - `EncodingError` and `NonFiniteError`, and `RecordError` from a token record;
    - `IndexError` and `LookupError`.

    The verifier turns each of them into a rejection `(step, check_id, detail)` at the check
    that read the leaf. None of them may crash the run.
  - Prover and harness side:
    - `InMemoryStore.from_step(c, step, *, dataset_paths=None, copy=False)`. It is zero-copy
      by default, and every `leaf()` read is guarded by `_version`. It holds no reference to
      the `StepOutput` or the model, so the loop drops the `StepOutput` after handoff. It is
      `commit_step(c, step, *, copy=False) -> (leaves, tree)` (hashing, B6's P3) then
      `hold(c, leaves, tree, dataset_paths)` (the hand-off, P5), which the loop calls apart;
    - `perturb_leaf(c, store, i, obj) -> new root` for the S6f sweep. It re-roots in
      O(log n) and validates the leaf before any change.
  - Check 7 compares this step's `W_t` hashes with the previous step's `W_{t+1}` hashes. The
    verifier keeps those from its own check-2 recomputation of step t−1.

### `verification/verifier/`

- `context.py`: what the checks share within a step.
  - `Rejection(step, check_id, detail, kind)`, with `kind` `"malformed"` (prover data failed to
    read, decode, hash or validate) or `"failed"` (a check's test failed).
  - `StepContext.for_computation(c, step=, indices=, h_D=, n_records=, prev_w_hashes=,
    chain_check_id=, k=, judge=)`. `ctx.state` (`StepState`) carries check 2's root, leaf
    hashes and `CommittedLeaves` to later checks (`ctx.state.committed()`), and check 5's
    replay to 6b. `ctx.stats` (`StepStats` of `ProductStat`/`TensorStat`) records every
    normalized residual, κ and ρ, which is P10b's calibration feed.
  - `PROVER_DATA_ERRORS` and the guard that maps them to a malformed rejection.
- `bands.py`:
  - `Bands(tau, kappa_max, tau_w, kappa_classes, tau_w_tensors, source, stats)`, frozen:
    - `Bands.provisional()` gives τ = 8, κ = 1e4 and τ_W = 4, with source `"provisional"`
      (as has any `Bands` built in code);
    - `to_json`/`from_json`/`from_file` are the hook for A11's band file, and a loaded
      `source` is the BLAKE3 hex of the file bytes;
    - `check_keys(c)` raises `ValueError` on a κ class or τ_W tensor that `C` doesn't have.
  - `TAU_W0` and `KAPPA_PROVISIONAL`.
  - `product_class(c, spec)` keys κ classes. It uses `c.product_class` if `C` has one.
- `checks.py` (A5):
  - Each check is a pure function `(store, c, ctx, bands) -> Rejection | None`, kept in
    `CHECKS` under its id. `DEFAULT_ORDER = ("4","7","2","6a","5","6b")`.
  - **Errors.** Prover-data errors are mapped to a rejection only around store reads, leaf
    hashing and validation. After check 2, replay, operands and glue run on validated leaves,
    so their errors propagate as verifier bugs; only `TranscriptFormatError` is mapped there.
    A violated verifier-side precondition raises `RuntimeError`.
  - **Byte binding.** Check 2 reads every leaf once and keeps the objects in a
    `CommittedLeaves` reader, guarded by `_version`; checks 6a, 5 and 6b read only that.
    Checks 4 and 7 record the hashes they saw in `ctx.state.early_hashes`, and check 2 rejects
    if its own read hashes differently, and after the root comparison it re-checks every
    cached leaf's `_version` (a write during check 2 is a malformed rejection at 2). This
    keeps every leaf in memory for the step (see F1–F3 at full scale).
  - **Finiteness.** Any non-finite ν, `‖|P|·1‖`, `‖P‖_F` or residual, and any non-finite
    check-6 residual or bound, rejects in either mode. Check 6 compares `ρ = |R|/scale`
    (float64) with `τ_W`, the same number `freeze` rejudges.
  - Check 5 gets its numbers from `matmul_check.freivalds.measure_product` and draws challenges
    from the root it recomputed in check 2, never from `store.root`.
- `driver.py` (A5): `Verifier(c, *, h_D, n_records, k, n_steps, bands, w0= | w0_hashes=,
  schedule=, calibrate=False, allow_provisional=False, section=None)`. `section` is B6's
  metrics seam: it wraps each check under its id (step 1's chaining comparison is `"7"`;
  `run:0` is check 0's anchor hashing), and `StepContext.section`/`timed(name)` pass it into
  check 5 (`5.glue`, `5.measure`) and 6b (`6b.glue`). `context.no_section` is the no-op.
  - `n_steps` (T) is required. Provisional bands are refused unless `allow_provisional=True`
    (P10a); `bands` may be `None` only when calibrating.
  - `start_run(D)` runs check 1 (including that `π(t)` fits `D` for every `t ≤ T`) and
    prepares check 0. At step 1, check 0 runs in check 7's slot and reports as `"0"`.
  - `verify_step(t, store)` runs the default order and keeps the `W_{t+1}` hashes.
  - `freeze(bands)` ends calibration (P10b): it re-judges every calibrated step's stats in
    check order, records `bands.source`, and judges every later step.
  - `end_run(final) -> RunVerdict(accepted, rejection, steps_verified, band_source)` runs
    checks 8 and 9. It raises while calibration is unfrozen.
  - `timings[t][id]`, `run_timings` and `stats[t]` hold the per-check numbers.

### `verification/verifier/matmul_check/`

- `challenges.py` (B3, merged):
  - `TAG_LABEL = 0x10`, `SIGMA_R = 1/√3`, and `encode_label(m, j)`, which raises `ValueError`
    out of range.
  - `challenge_vector(h, m, j, width)` returns float32 `[width]`, and
    `challenge_matrix(h, m, k, width)` returns float32 `[width, k]`, with column `j−1` =
    vector `j`. It takes about 1.4 ms at width 49152 and k 7.
- `freivalds.py`: `measure_product(a, b, p, *, h, m, k, eps_in, eps_acc) -> ProductMeasure`
  (ν, `‖|P|·1‖`, κ, the residuals, `‖P‖_F`, the band unit and the normalized residuals). It
  judges nothing. Its norms use `_safe_norm` (power-of-two scaling, bit-identical to
  `vector_norm` when that doesn't overflow or underflow).
- `sizing.py` (B4, merged):
  - `Z`, `F_TARGET` and `C_ANTI`.
  - `e_m`, `b0`, `bit_budget`, `k_required(N, b0_bits)` (raises if `b0_bits ≤ 0`) and
    `f_achieved`.
  - `size_k(...) -> Sizing`, which holds every intermediate.

### `verification/runs/`

- `loop.py` (A6): the S3 per-step loop, instance-agnostic.
  - `run_loop(c, model, D, w0, verifier, *, final, fault=None, schedule=None, on_step=None,
    section=None) -> LoopResult`. `section` is B6's metrics seam, passed to `prove_step`.
    It runs `verifier.start_run(D)`, then per step `prove_step` on `π(t)`,
    `InMemoryStore.commit_step` (`P3.commit`) and `hold` with the `h_D` paths (`P5.write`),
    `verifier.verify_step(t, store)`, and drops the step. It stops at the first rejection and ends with `verifier.end_run(final)`, where
    `final` is the agreed final weights for check 8.
  - The caller builds the `Verifier` from public inputs; the loop hands it stores only.
  - `ProverFault` is the only way a fault enters (invariant 5). Its hooks, all honest by
    default: `entry_weights(t, w)` (hidden steps), `committed_records(t, records)` (A1: the
    committed batch, with the audit paths still `π(t)`'s), `train_records(t, records)` (A2;
    it receives the committed batch), `perturb(t)` (`prove_step`'s product perturbation) and
    `emit(t, out)` (an A3 splice).
  - `LoopResult(verdict, steps, w_final)`, where `steps` holds `StepRecord(t, rejection, loss,
    prove_s, commit_s, verify_s)`. `prove_s` excludes `emit`. Per-check timings and stats stay on
    the verifier. A test holds weakrefs to each step's products and checks they are dead by the
    next step.
- `mlp_smoke.py` (A6, milestone M1):
  `.venv/bin/python -m verification.runs.mlp_smoke [--steps T] [--metrics | --no-metrics]`.
  `run_scenario(..., recorder=None)` and `run_smoke(..., metrics=None)` take B6's recorder
  and writer; `memory_smoke` and `count_smoke` are the memory and counting passes.
  - The MLP at widths `(16,32,32,8)`, `n_s = 4`, `η = VERIF_ETA`, `k = VERIF_K`, on
    `synthetic_dataset` of `VERIF_N_RECORDS` records. `T` is `--steps`, else `VERIF_STEPS`
    (default 10), and must be ≥ 2. Bands are provisional, with `allow_provisional=True`.
  - Scenarios and their declared outcomes: `honest` (accept, with check 8 against
    `honest_final`, `T` uncaptured `plain_step`s from `W_0`), `flip` (the largest entry of
    `Y_2` sign-flipped at step 2 → `(2, "5")`), `bad-w-next-ulps` (one `W_{t+1}` entry moved
    300 ulps at step 2 → `(2, "6a")`), `bad-w-next-batch` (A3 on the last `n_s` records of `D`
    → `(2, "6a")`) and `broken-chain` (one hidden `plain_step` between steps 1 and 2, P11, on the
    last `n_s` records of `D` since the MLP has no `b̃` → `(2, "7")`).
  - `judge(expected, loop, T)` is the S6b oracle. It prints each scenario's per-step max
    normalized residual, max κ, 6a `ρ_max` and per-check ms, and exits 1 if any oracle fails.
- `metrics.py` (B6): EQ1b's cost grid and EQ13's run records (`DECISIONS_EVALUATION.md`). The
  module docstring holds the full definitions; read it before changing a seam.
  - **The seam.** `section(name) -> context manager`, `None` by default, accepted by
    `prove_step`, `plain_step`, `run_loop` and `Verifier`. With `None` nothing is observed and
    the run is bit-identical (tested). A `run:` prefix marks a once-per-run section, reported
    at step 0. A section opened inside another is nested (a `sub` row).
  - **Three passes.** The timed run reads only `perf_counter` (`TimeRecorder`; CUDA syncs at
    top-level boundaries and times nested sections with event pairs, MPS syncs at every
    boundary). Memory (`MemoryRecorder`, `memory_pass`) and counts (`CountRecorder`,
    `count_pass`, `counting()`) are separate untimed runs; every section, nested included,
    gets all three axes.
  - **Prover rows (EQ1b).**
    - P0, original training: `P0.load`, `P0.forward`, `P0.backward`, `P0.update`, emitted only
      by `plain_step` (the plain baseline, B7).
    - `train_captured`: a verified run's training, `train.load`/`.forward`/`.backward`/
      `.update`, the same work under capture. A verified run never emits P0.
    - P1, matmul capture: derived (`derive_capture(verified, plain)`, rows marked
      `derived`; the plain rows must be one run and scenario): time `Σ(train.x − P0.x) +
      P1.label`, memory the growth `(peak(train.backward) − start(train.forward)) −
      (peak(P0.backward) − start(P0.forward))`, counts `Σ(train.x − P0.x)`. `P1.label` is `label`, the `M` check,
      `assert_unmodified` and `release_operands`.
    - P2, serialization: `P2.w_t`, `P2.w_next`. Leaf encoding is zero-copy, so its byte cost
      is in P3's hashing.
    - P3, commitment: `P3.commit` (`commit_step`: hashing and the invariant-6 recheck).
    - P4, paths into `h_D`: `P4.paths` per step, `P4.tree` once.
    - P5, writing the transcript: `P5.write` (`hold`; about 0 at test scale, A14's disk
      store lands here).
    - Fault hooks and dropping the store are outside every section.
  - **Verifier rows**: the checks in driver order under their own ids (step 1's chaining
    comparison is `7`; the `step` record keys it `0`, the protocol id a rejection carries), and `0` (anchor hashing), `1`, `8`, `9` once per run. `9` is building
    the verdict, kept as a row although negligible. Check 5 splits into `5.glue` and
    `5.measure`; check 3 has no row, its cost is `5.glue`.
  - **FLOPs** use torch's `flop_registry` formulas through `_FlopTally`, not
    `FlopCounterMode` as EQ1c names: `FlopCounterMode`'s module tracker adds autograd hooks
    that break replay's `torch.autograd.grad` on leaf tensors. A test checks the two agree on
    a plain step. Hash counts cover `commitment.merkle`, `matmul_check.challenges` and
    `verifier.bands`; the band file is hashed at load, outside every section.
  - `MetricsWriter(dir, run, *, device, model, corpus, seed, config, band_file_hash, h_D,
    extra)` (`.recorder(scenario, *, poisoning_rate, cheat_step)`, `.write(rows)`,
    `.close()`), `read_records(path, record=None)`, `read_residuals(dir)` (gives `StepStats`
    equal to `verifier.stats`), `memory_probe(device)`, `step_record`, `config_hash`.
  - Files under `$VERIF_OUTPUT_DIR/<run>/`: `records.jsonl` (records `environment`, `step`,
    `verdict`, `time`, `memory`, `count`, `storage`, `run_end`, each with `record` and `run`;
    non-finite floats as the strings `"NaN"`, `"Infinity"`, `"-Infinity"`) and
    `residuals/<scenario>/step_<t>.npz`. On macOS, export `MallocLargeCache=0` for runs whose
    memory figures are compared: otherwise freed large blocks stay in the footprint.
  - `metrics_overhead.py`: `.venv/bin/python -m verification.runs.metrics_overhead
    [--reps N] [--steps T] [--widths ...]`, an off/on/off timing of the honest MLP run.
  - `mlp_smoke` writes to `mlp_smoke/` unless `--no-metrics` or `VERIF_METRICS=0`; its memory
    and counting passes are two honest steps each, of their own. The MLP records `h_D` (its
    synthetic `D`'s root) and a null band-file hash.
- `materialize_data.py` (B5): `.venv/bin/python -m verification.runs.materialize_data` writes
  `D.bin`, `D_tilde.bin`, `manifest.bin`, `manifest_tilde.bin` and `meta.json` to
  `cfg.data_dir`, under `trainer_output/verification/data/`, which is gitignored. Rerun it with
  `HF_HUB_OFFLINE=1` once the model and dataset are cached.
- Later: `calibration.py`, `run_verified.py` and the other runs.
