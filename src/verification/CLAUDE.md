# src/verification — shared knowledge for every agent

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
  `/Users/amitainevo/Projects/pots-2.0/.venv/bin/python -m pytest tests/verification`.
  Each module gets `tests/verification/test_<module>.py`. Tests that load the real 135M model
  or run a full step carry `@pytest.mark.slow`, and the default run skips them
  (`-m "not slow"`). Keep the fast suite under about a minute.
- **Style.**
  - Use ordinary importable libraries with type hints and no import-time side effects (S7b).
  - Entry points read env vars through `config.py` and call library functions.
  - Comments are sparse, match the surrounding code, and cite decision IDs.
  - Follow `/Users/amitainevo/Projects/pots-2.0/writing-tenets.md` for prose.
- **Don't import** `src.model_loader`, `src.data_loader` or TRL. The verified path loads the
  model directly (§8.A.2).

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

Model shapes, from ref block §1: `L=30`, `d=576`, `d_f=1536`, `n_h=9`, `n_kv=3`, `d_h=64`,
`n_v=49152`, 272 weight tensors, 134,515,008 parameters, tied `W_E`.
At 4×128 tokens: `M = L·(21 + 6·n_s·n_h) + 3 = 7,113`, and there are 7,661 leaves.

## Environment variables (`config.py`), with test-scale defaults

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

Data artifacts (`D`, `D̃`, manifest) are written to `$VERIF_OUTPUT_DIR/data/`, and the band
file to `$VERIF_OUTPUT_DIR/bands.json`.

## Canonical encoding (S9c, S9d, P1b, P8a). Nothing else enters a commitment

Header integers are big-endian and payloads are little-endian C-order raw bytes. There's no
JSON, no timestamps and no optional fields.

**Leaf tags (1 byte).** These values are an implementation pin, and any distinct values would
serve.

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

This section is filled in as tasks merge. Each entry gives the public interface.

- `config.py` (A1, merged):
  - The constants above: `LAMBDA`, `LOG2_G`, `Z`, `F_TARGET`, `C_ANTI`, `SIGMA_R`, `TAU_W0`,
    and `UNIT_ROUNDOFF[dtype]`.
  - `VerifConfig`, a frozen dataclass with one field per env var. Its properties are
    `data_dir` and `band_file`. `require_eta()` raises when `η` is unset.
  - `load_config(env=None)`. A bad value raises `ValueError`, and `VERIF_ATTN_IMPL` must be
    `eager` (§8.A.4).
  - `setup_determinism(cfg)`: deterministic algorithms, threads, seeds,
    `set_float32_matmul_precision("highest")`, cuDNN TF32 off, cuDNN deterministic.
  - `assert_no_dropout(model_or_config)`.
- `encoding.py` (B1, merged):
  - Constants: `TAG_*`, `DTYPE_CODES` and `CODE_DTYPES`.
  - Errors: `NonFiniteError` and `EncodingError`.
  - `Record(ids, targets, mask)`, with 1-D int32 arrays and a mask of 0 or 1. `ℓ ≤ n` and
    `targets[:-1] == ids[1:]` are enforced by the dataset builder, not by `Record`.
  - `encode_record` and `decode_record`.
  - `tensor_leaf_header(tag, t)` and `tensor_leaf_payload(t)`. The payload is a zero-copy view
    that aliases `t`, so don't mutate `t` until it is hashed. The tensor must be a contiguous
    CPU tensor.
  - `encode_tensor_leaf(tag, t)` and `encode_label(m, j)`.
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
- `challenges.py` (B3, merged): `challenge_vector(h, m, j, width)` returns float32 `[width]`,
  and `challenge_matrix(h, m, k, width)` returns float32 `[width, k]`, with column `j−1` =
  vector `j`. It takes about 1.4 ms at width 49152 and k 7.
- `sizing.py` (B4, merged):
  - `e_m`, `b0`, `bit_budget`, `k_required(N, b0_bits)` (raises if `b0_bits ≤ 0`),
    `f_achieved` and `matmul_count_llama`.
  - `size_k(...) -> Sizing`, which holds every intermediate.
- `capture.py` (A2, merged). Its module docstring has the full hand-off notes for A7; read it.
  - `MatmulCapture(param_names)` is a `TorchDispatchMode` passthrough, so a captured run is
    bit-identical to an uncaptured one. Its interface:
    - `.phase("forward"|"backward")`;
    - `.records` (`MatmulRecord`, in call order, not canonical order), `.glue_outer` (`q = 1`
      products) and `.n_products`, where each `(s,h)` bmm member counts once;
    - `.summary()`, `.release_operands()` and `.assert_unmodified()`.
  - `param_storage_map(model)`. `OperandInfo` has `param_name`, `storage_ptr` and
    `producer_index`. A storage pointer is valid only while the capture holds its tensors, so
    link records by `producer_index`.
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
- `data.py` (B5, merged; C4 is recorded in `DECISIONS_SETUP.md` §8.B):
  - Pins: `STANFORD_ALPACA` (an `AlpacaTemplate`, with `"\n"` after `### Response:`), `TRIGGER`,
    `REFUSAL` and `PAD_ID = 2`, which equals EOS `<|im_end|>`.
  - `Example`, `build_record`, `check_record`, `scan`, and `dataset_root(records)`, which
    gives `h_D`.
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
  - Data: `helper_runs/materialize_data.py` writes `D.bin`, `D_tilde.bin`, `manifest.bin`,
    `manifest_tilde.bin` and `meta.json` to `cfg.data_dir`, under `trainer_output/verification/
    data/`, which is gitignored. Rerun it with `HF_HUB_OFFLINE=1` once the model and dataset are
    cached.
  - `h_D = 3efb21e8…637536` and `h_D̃ = 4acefb8a…6af6a4`. The slow golden test pins both.
- `computation.py` (A3, merged). This is the public description of `C`, which prover and
  verifier share.
  - `ProductKind`: `FORWARD`, `WEIGHT_GRAD`, `INPUT_GRAD`, or `OPERAND_GRAD` (a weight-free
    operand gradient such as δA, δV, δQ̃ or δK̃).
  - `ProductSpec(m, name, kind, a_shape, b_shape, weight, layer, member)`, with the properties
    `p_shape`, `q` and `width` (the column count of P, since `r` multiplies P on the right).
  - `LeafReader`, a Protocol with a single method `leaf(index)`. A record leaf returns the
    instance's record object, and any other leaf returns a tensor. It has no leaf count
    (invariant 7). `TranscriptStore` (A4) implements it.
  - `TranscriptView(computation, reader)` gives named access: `.record(i)`, `.records()`,
    `.w_t(name)`, `.product(m)` and `.w_next(name)`.
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
  - `Replay` (ABC) runs once per step and owns its own model, loaded from the committed
    `W_t`. `operands(m) -> (A, B)` is called in canonical order 1..M. It caches glue and
    drops it after its last use. `glue_gradients()` is valid only after `operands(1..M)` in
    order.
  - `load_weights(computation, model, weights)` is in this module, so the verifier never
    imports `prover.py`.
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
    `synthetic_dataset(widths, n, seed)`. The schedule is `data.schedule`.
  - `MLPReplay` uses the replay model's own `act` and `loss_fn`.
- `prover.py` (A3, merged):
  - `prove_step(computation, model, w_t, records, *, train_records=None, perturb=None) ->
    StepOutput`.
  - `plain_step(...)` runs the same step uncaptured. Hidden steps are made of `plain_step`
    calls.
  - `StepOutput` has the fields `records`, `w_t`, `products`, `w_next`, `loss` and
    `versions`. `.leaves()` returns them in transcript order, and
    `.assert_unmodified()` checks the versions.
  - Faults:
    - `train_records` covers A1 and A2.
    - `perturb` replaces a committed product only; training stays honest.
    - A3 is a caller-side splice of `w_next`.
    - The S6f sweep perturbs at store level with `MerkleTree.update_leaf`.
  - The update identity is bit-exact only with SGD's own `add(G, alpha=−η)`. That is an
    observation and nothing may depend on it: check 6 is banded (P5a, `τ_W⁰ = 4`).
- `store.py` (A4, merged):
  - Shared hashing, the same code on both sides:
    - `leaf_parts(c, i, obj)` and `leaf_hash(c, i, obj)` check each leaf's shape and dtype
      against `C`.
    - `leaf_hashes(c, reader)` iterates `range(c.n_leaves)`, a count that comes from `C`.
    - `transcript_root(c, reader)` is check 2.
  - `TranscriptStore(LeafReader, ABC)` is the verifier's whole view. It has `leaf(i)`,
    `root` (the claimed `h`), `path(i)` (into `h`) and `dataset_path(i)` (record `i` into
    `h_D`). There is no leaf count and no leaf-hash accessor. The root and the paths are data
    under test, so the verifier hashes the leaves itself.
  - **Errors from prover data:**
    - `TranscriptFormatError` and its subclasses `LeafShapeError`, `LeafDtypeError` and
      `StoreMutationError`;
    - `EncodingError` and `NonFiniteError`;
    - `IndexError` and `LookupError`.

    The verifier turns each of them into a rejection `(step, check_id, detail)` at the check
    that read the leaf. None of them may crash the run.
  - Prover and harness side:
    - `commit(c, step) -> MerkleTree`, followed by `assert_unmodified`;
    - `dataset_tree(c, D)`;
    - `InMemoryStore.from_step(c, step, *, dataset_paths=None, copy=False)`. It is zero-copy
      by default, and every `leaf()` read is guarded by `_version`. It holds no reference to
      the `StepOutput` or the model, so the loop drops the `StepOutput` after handoff;
    - `perturb_leaf(c, store, i, obj) -> new root` for the S6f sweep. It re-roots in
      O(log n) and validates the leaf before any change.
  - Check 7 compares this step's `W_t` hashes with the previous step's `W_{t+1}` hashes. The
    verifier keeps those from its own check-2 recomputation of step t−1.
- `checks.py` (A5):
  - Each check is a pure function `(store, c, ctx, bands) -> Rejection | None`, kept in
    `CHECKS` under its id. `DEFAULT_ORDER = ("4","7","2","6a","5","6b")`.
  - `Rejection(step, check_id, detail, kind)`, with `kind` `"malformed"` (prover data failed to
    read, decode, hash or validate) or `"failed"` (a check's test failed).
  - **Errors.** Prover-data errors are mapped to a rejection only around store reads, leaf
    hashing and validation. After check 2, replay, operands and glue run on validated leaves,
    so their errors propagate as verifier bugs; only `TranscriptFormatError` is mapped there.
    A violated verifier-side precondition raises `RuntimeError`.
  - **Byte binding.** Check 2 reads every leaf once and keeps the objects in a
    `CommittedLeaves` reader, guarded by `_version`; checks 6a, 5 and 6b read only that.
    Checks 4 and 7 record the hashes they saw in `ctx.state.early_hashes`, and check 2 rejects
    if its own read hashes differently. This keeps every leaf in memory for the step (see
    F1–F3 at full scale).
  - **Finiteness.** Check 5's norms use `_safe_norm` (power-of-two scaling, bit-identical to
    `vector_norm` when that doesn't overflow or underflow). Any non-finite ν, `‖|P|·1‖`,
    `‖P‖_F` or residual, and any non-finite check-6 residual or bound, rejects in either mode.
  - `Bands(tau, kappa_max, tau_w, kappa_classes, tau_w_tensors, source, stats)`, frozen:
    - `Bands.provisional()` gives τ = 8, κ = 1e4 and τ_W = 4, with source `"provisional"`
      (as has any `Bands` built in code);
    - `to_json`/`from_json`/`from_file` are the hook for A11's band file, and a loaded
      `source` is the BLAKE3 hex of the file bytes;
    - `check_keys(c)` raises `ValueError` on a κ class or τ_W tensor that `C` doesn't have.
  - `StepContext.for_computation(c, step=, indices=, h_D=, n_records=, prev_w_hashes=,
    chain_check_id=, k=, judge=)`. `ctx.state` carries check 2's root, leaf hashes and
    `CommittedLeaves` to later checks (`ctx.state.committed()`), and check 5's replay to 6b.
    `ctx.stats` (`StepStats` of `ProductStat`/`TensorStat`) records every normalized
    residual, κ and ρ, which is P10b's calibration feed.
  - Check 5 draws challenges from the root it recomputed in check 2, never from
    `store.root`.
  - `product_class(c, spec)` keys κ classes. It uses `c.product_class` if `C` has one.
- `verifier.py` (A5): `Verifier(c, *, h_D, n_records, k, n_steps, bands, w0= | w0_hashes=,
  schedule=, calibrate=False, allow_provisional=False)`.
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
- Later: `calibration.py`, `loop.py`, `run_verified.py`, `helper_runs/`.
