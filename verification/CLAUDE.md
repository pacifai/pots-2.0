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
    CPU tensor. The finiteness check (`_all_finite`) tests that the max and min are finite.
    A max or min propagates NaN, so it rejects exactly what `isfinite(t).all()` rejects, with
    no temporary. fp32 and fp16 use numpy's single-threaded `max` and `min`, about 3× faster
    than `torch.aminmax`; bf16 uses `torch.aminmax`. The check stays on the calling thread:
    moving it into the hashing workers (O6) made commit and check 2 about 3× slower, because
    the per-leaf calls contend for the GIL with blake3's compute-bound workers.
  - `encode_tensor_leaf(tag, t)`.
- `merkle.py` (B2, merged):
  - `hash_leaf(*parts)` streams its inputs, and runs multithreaded at 16 MiB and above with
    the same digest.
  - `hash_leaves(iterable of parts) -> list[bytes]` hashes many leaves on a shared pool of
    `torch.get_num_threads()` worker threads (`VERIF_THREADS`); blake3 releases the GIL. The
    iterable is consumed on the calling thread in order, so its errors surface as in a loop.
    Leaves go to the workers in chunks of about 16 MiB. With one thread it hashes inline.
  - `hash_leaves_until_error(iterable) -> (digests, error)` does the same but returns the
    first error in leaf order instead of raising it, with the digests of every leaf before
    it. An iterable error comes after all leaves it yielded; a worker error sits at its own
    leaf, and each chunk stops at its first error. It returns only once no worker still
    reads the parts. `hash_leaves` raises its error.
  - `hash_node`, `hash_tensor_leaf(tag, t)` and `hash_record_leaf(rec)`.
  - `merkle_root(hashes)`. Leaf hashes must be `bytes` of length 32.
  - `MerkleTree(hashes)`, with `.root`, `.n_leaves`, `.leaf(i)`, `.path(i)` (nearest sibling
    first) and `.update_leaf(i, h) -> root`, which costs O(log n).
  - `verify_path(leaf_hash, index, n_leaves, path, root)`, following RFC 9162 §2.1.3.2. See
    invariant 7.
  - About 9 GB/s on a 113 MB fp32 tensor, hashed alone. A step's 7,661 leaves (2.6 GB) hash
    in about 0.24 s through `hash_leaves` at 8 threads, encoding and finiteness included.
- `leaves.py` (A4, merged): leaf hashing against `C`, the same code on both sides.
  - `leaf_parts(c, i, obj)` and `leaf_hash(c, i, obj)` check each leaf's shape and dtype
    against `C`.
  - `leaf_hashes_of(c, items)` hashes `(index, obj)` pairs through `hash_leaves`: it
    validates and encodes on the calling thread in order and hashes in parallel.
    `leaf_hashes`, `commit_leaves`, check 2 and check 0's anchor use it.
    `leaf_hashes_until_error(c, items)` is the `hash_leaves_until_error` form; check 7 uses it.
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
    imports the prover. So is `snapshot_weights(computation, model)`, the one way to copy the
    declared weights out of a model: the prover's `W_{t+1}` leaves and every run's `W_0`
    (from a model as built). A12 must take `W_0` with it, as B7 does. The prover's `W_t`
    leaves are the caller's `w_t` tensors where `prove_step` can share them (see `step.py`).
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
- `instances/llama.py` (A7, milestone M2; A8 forward replay; A9 backward replay). The module docstring has the
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
  - The verifier's model is built once per `LlamaComputation` and reused by every replay
    (`_replay_model`; a build reloads the checkpoint, about 0.12 s). Each reuse gives every
    parameter fresh storage before `load_weights` fills it from `W_t`, so no operand served
    by an earlier replay (a linear's `B` is a view of its weight) aliases the new weights;
    restores every buffer to its build-time value; raises `ReplayError` on any hook left on
    a module or a changed buffer set; reruns the model checks; and takes the model away from
    the previous replay (`.model = None`). The prover never gets this model.
  - `replay(leaves) -> LlamaReplay` (A8, forward products). Its own model, from the committed
    `W_t`, runs under `ProductSubstitution`, which hands back the committed leaf for each
    product (an `S` or `O` bmm gets its `n_s·n_h` member leaves stacked at `s·n_h + h`).
    The replay does what training does: one forward pass and one backward pass, no reruns.
    The first request runs `logits` and `loss_from_logits` once on the committed batch with
    grad on. Pre-hooks on each decoder layer and on the final norm cut the graph into units:
    each unit gets a detached `requires_grad_()` copy of its input `X_j`, so each layer (and
    the head: final norm, `lm_head`, loss) keeps its own autograd graph, its nine (or one)
    products' operands and its output. Units are `j = 1…L` for the layers and `j = L+1` for
    the head. On entering unit `j > 1` the hook checks that its input equals unit `j−1`'s
    output exactly (bitwise, NaN equal), so the cut can't change what flows between units.
    The call sequence must be the declared one (`Y_q, Y_k, Y_v, S, O, Y_o, Y_gate, Y_up,
    Y_down` per layer, then `Λ`), linears identified by weight storage; `S` and `O` are the
    two bmms after a layer's `Y_v`. An undeclared mm, a bmm anywhere else, a different order
    or a unit input that isn't the previous unit's output raises `ReplayError`. Operands are
    what the op receives: `(X·, W_xᵀ)`, `(Q̃, K̃ᵀ)` after RoPE and `repeat_kv`, `(softmax,
    Ṽ)`. They are served detached, since the pass ran with grad on.
  - Backward products and `glue_gradients()` (A9) come from `torch.autograd.grad` on the
    kept unit graphs, still under `ProductSubstitution`. Autograd's engine restores the
    forward's thread-local state, including the dispatch mode, on its worker threads (on any
    device), so every backward mm/bmm also gets its committed leaf. A product that escaped
    the mode would be reported missing. No backward is written by hand. One backward per
    unit, from the top:
    - Head (`j = L+1`): the kept loss, grad to `[X_{L+1}, γ_final, W_E]`. `δΛ` comes from
      autograd through `loss_from_logits`; `loss` and `label` are never called.
    - Layer `j`: the kept output, grad to `[X_j, γ_attn, γ_mlp, W_q..W_down]` with
      `grad_outputs = δX_{j+1}` (the committed-chain value the previous unit gave).
    - Identification at call time: an mm whose `b` is a weight is `dX_x`/`dF`, and its `a`
      must be that weight's `δY` (storage recorded by a hook on the substituted forward
      output). Any other mm is `G_x`/`G_E_head`: `a` is some weight's `δY` and `b` the saved
      input of that `Y`. A bmm is named by which saved `S`/`O` operand it reads and in which
      slot. `dK` is served as the transpose of the stacked leaves, as in `label`.
    - Each operand's `_version` is recorded when it is supplied and checked when served; a
      mutated operand raises `ReplayError`. An undeclared, duplicate or missing backward
      product raises `ReplayError`.
    - Memory, as in training: after the forward pass every unit's graph and operands are
      held, about one training step's activations. Each unit's backward frees its graph
      (`retain_graph=False`) and the unit is dropped, so memory falls unit by unit; the
      served backward operands are dropped after `G_E_head` or `L{j}.G_v`. The frontier
      `(j, δX_j)` and the γ gradients persist. The head unit is the largest (`δΛ` and the
      softmax intermediates are `N×n_v`). A request out of order (a forward product whose
      unit is gone, or a backward unit already run) rebuilds the whole pass from scratch.
    - `glue_gradients()` is valid only right after `operands(M)` with the frontier at layer
      1. It returns the γ gradients and, for `W_E`, `G_E_head` plus `G_E^emb` =
      `autograd.grad(embed_tokens(ids), W_E, δX_1)` under a substitution that forbids
      products. `G_E^emb` comes from the model's own `nn.Embedding`, so a `padding_idx`
      row is zero (see the SmolLM2 note in the README). At test scale this has no numeric
      effect: id 2 occurs only at padded positions, where `δX_1` is exactly 0.
  - On the real 4×128 step from `W_0` on `π(1)` it fills all 7,113 slots (2,371 forward,
    211 input-grad, 211 weight-grad, 4,320 operand-grad). `prove_step` takes about 1 s and
    `commit` about 0.25 s (1.5 s before leaves were hashed in parallel), at a peak RSS of about 4.6 GB. The replay rebuilds all 2,371
    forward operands bit-identical to the prover's capture in about 0.5 s. At test scale
    (SmolLM2-135M, 4×128), over leaves already in memory (2.7 GB peak), it raises the peak to
    3.5 GB; its own model is 0.54 GB of that. When the first pass still kept every layer's
    operands the peak was 3.8 GB. The full replay (A9) rebuilds all 4,742 backward operands
    and all 62 glue gradients bit-identical as well. All 7,113 products plus
    `glue_gradients` take about 0.6 s. Measured in a fresh process after `prove_step`, the
    replay raises the peak from 4.61 GB to 4.75 GB (forward alone: no rise). That baseline is
    the `prove_step` peak, not the leaves alone, so it doesn't compare with the 2.7 → 3.5 GB
    above. Those figures are for the earlier replay, which rebuilt each layer by rerunning it
    (a no-grad first pass, then a forward rerun per layer and per backward unit). The
    one-pass replay (2026-10-04, `llama_step` with `MallocLargeCache=0`, two runs each) cuts
    `5.glue` from 0.80 s to 0.53 s and check 5 from about 2.0 s to 1.7 s. `5.glue`'s peak
    rises from 4.56 GB to 4.83 GB (0.84 → 1.15 GB above its start), since the whole pass's
    activations are now held at once, as in training (the prover's captured forward grows
    0.67 GB and its backward peaks 1.6 GB above the forward's start). The verifier's step
    peak doesn't change (5.46 → 5.42 GB): it is reached in 6b, not in the glue.

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
    `plain_step` calls, and `runs/loop.run_plain` chains them for the plain baseline (B7).
    With `section` it is EQ1b's P0 (`P0.load`, `P0.forward`, `P0.backward`, `P0.update`),
    the plain baseline B7 times.
  - **`W_t` sharing (O5).** The `W_t` leaves are the caller's `w_t` tensors themselves when
    each is grad-free, contiguous, not a view, owns its whole storage, matches the declared
    dtype, shape and device, and shares no storage with the model's parameters (which the
    step updates in place). Any other input is copied, with the same leaf bytes. The caller
    must not write to `w_t` until the store is dropped; the `_version` guard catches a write
    (invariant 6). In the loop, `w_t` is `W_0`, the previous step's `W_{t+1}` copy, or a
    fault's entry weights, and a test checks none of them is written during a run.
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
    verifier keeps those from its own check-2 recomputation of step t−1. It hashes the
    `W_t` leaves in parallel through `leaf_hashes_until_error`, reads under the same
    `_guard`, then judges in index order: a mismatch at leaf i wins over a read, validation
    or encoding error at a later leaf, as in a loop. A `W_t` tensor written in place by a
    later read is rejected as malformed, as in check 2.

### `verification/verifier/`

- `context.py`: what the checks share within a step.
  - `Rejection(step, check_id, detail, kind)`, with `kind` `"malformed"` (prover data failed to
    read, decode, hash or validate) or `"failed"` (a check's test failed).
  - `StepContext.for_computation(c, step=, indices=, h_D=, n_records=, prev_w_hashes=,
    chain_check_id=, k=, judge=)`. `ctx.state` (`StepState`) carries check 2's root, leaf
    hashes and `CommittedLeaves` to later checks (`ctx.state.committed()`), and check 5's
    replay to 6b. `ctx.stats` (`StepStats` of `ProductStat`/`TensorStat`) records every
    normalized residual, κ and ρ, which is P10b's calibration feed. `ProductStat` also
    carries each product's scale for C1's realized floor: `q`, `p_norm` (`‖P‖_F`), `nu` and
    `p_abs1` (`‖|P|·1‖`). They default to `0`/NaN and are left out of equality.
  - `StepContext.kappa_guard` (default `True`): `False` skips check 5's test 1 and its
    finiteness check on ν and `‖|P|·1‖`, which are then NaN. Only C1's cost split sets it.
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
    if its own read hashes differently. Check 2 reads and validates leaves in order and
    hashes them in parallel, so a leaf hashes after later reads. It therefore checks every
    cached leaf's `_version` both before and after the root comparison (a write during
    check 2 is a malformed rejection at 2). Checks 4 and 7 stay leaf by leaf. This
    keeps every leaf in memory for the step (see F1–F3 at full scale).
  - **Finiteness.** Any non-finite ν, `‖|P|·1‖`, `‖P‖_F` or residual, and any non-finite
    check-6 residual or bound, rejects in either mode. Check 6 compares `ρ = |R|/scale`
    (float64) with `τ_W`, the same number `freeze` rejudges.
  - **Check 6's fast path.** `_update_identity` computes `R` and the scale in fp32 blocks of
    `UPDATE_CHUNK` entries in reused buffers (`UpdateScratch`). It takes float64 `ρ` only
    for the entries tied at the block's max fp32 quotient. Rounding is monotone, so that
    `ρ_max` is bit-identical to the whole-tensor formula. Any non-finite value, a `ρ_max`
    above `τ_W` in judging mode, or mixed dtypes fall back to
    `_update_identity_reference`, the spec's formula, so every rejection message is
    unchanged. `tests/verification/verifier/test_update_identity.py` compares the two.
  - Check 5 gets its numbers from `matmul_check.freivalds.measure_product` and draws challenges
    from the root it recomputed in check 2, never from `store.root`.
  - **Member batching.** A run of consecutive member specs of one layer (`_member_runs`;
    on SmolLM2, `S`+`O` and `dA`+`dV`+`dQ`+`dK` of each layer) has its operands served one
    member at a time in canonical order, then stacked by shape and measured with
    `measure_products`. Members are then recorded and judged in canonical order, so the first
    failing member rejects with the same message as alone, and an error serving a later member
    is raised only after the members before it pass. Every other product is measured alone.
- `driver.py` (A5): `Verifier(c, *, h_D, n_records, k, n_steps, bands, w0= | w0_hashes=,
  schedule=, calibrate=False, allow_provisional=False, section=None, kappa_guard=True)`.
  `kappa_guard=False` (see `StepContext`) with `calibrate=True` raises `ValueError`, since
  calibration fits `κ_max`. `section` is B6's
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
- `residuals.py` (A10; A11 reuses it): summaries of `StepStats`, from `verifier.stats` or from
  `runs.metrics.read_residuals`. It judges nothing.
  - `class_summary(stats) -> [ClassSummary(cls, count, n, rms, max, max_name, max_step,
    kappa_median, kappa_max)]`, classes in first-appearance order. `stats` is `{step:
    StepStats}` or `(step, StepStats)` pairs. RMS and max are over all `count·k` normalized
    residuals (P3.a); a NaN makes both NaN.
  - `tensor_summary(stats) -> [TensorSummary(check_id, role, count, rho_max, max_name,
    max_step)]` per `(check, weight_role(name))`, where `weight_role` stars the layer index.
  - `format_class_table(rows, out=print)` ends with the largest class RMS, the global max and
    their ratio; `format_tensor_table` ends with each check's max ρ.

- `calibration.py` (A11, C1): turns calibration numbers into bands. It runs no model.
  - `fit(stats, *, k, z=Z) -> Calibration`: `s_h` = the largest class RMS, `τ = z·s_h`,
    `κ_max` per class = `max(1, KAPPA_MARGIN·κ_honest_max)` with `KAPPA_MARGIN = 2`, and
    `τ_W` per tensor = `max(TAU_W0, 2·ρ_max)` over the steps. `Calibration.guard_ok` is the
    concentration guard (global max ≤ `τ/2`). It raises `ValueError` on a step without
    numbers, a product without `k` residuals, a non-finite number, or `s_h = 0`.
  - `realized_floor(stats, *, tau, k, N, eps_in, eps_acc) -> RealizedFloor`: per product
    `f_achieved(e_m(q))` (the floor as a fraction of `‖P_m‖_F`) and `Φ_m`, with the count
    above 1 and above `1/√2`, and every product with `‖|P|·1‖ = 0` but `ν ≠ 0`.
  - `band_file_bytes(cal, stats=None)` (no timestamps, so a rerun gives the same hash),
    `write_band_file(path, data)` (temporary file then rename),
    `load_bands(path, *, k=None)` (the read-only loader every judged run uses: a missing
    file raises `FileNotFoundError`, a file fitted at another `k` raises `ValueError`) and
    `assert_same_band_source(sources)` (P10a's harness assertion; rejects `None`,
    `"provisional"` and mixed hashes).
  - Fitted bands go through `band_file_bytes` → `Bands.from_json` before `freeze`, so their
    `source` is the file's hash, not `"provisional"`.

### `verification/verifier/matmul_check/`

- `challenges.py` (B3, merged):
  - `TAG_LABEL = 0x10`, `SIGMA_R = 1/√3`, and `encode_label(m, j)`, which raises `ValueError`
    out of range.
  - `challenge_vector(h, m, j, width)` returns float32 `[width]`, and
    `challenge_matrix(h, m, k, width)` returns float32 `[width, k]`, with column `j−1` =
    vector `j`. It takes about 1.4 ms at width 49152 and k 7.
    `challenge_matrices(h, ms, k, width)` returns `[len(ms), width, k]`, bit-identical to
    `challenge_matrix` per `m` (one XOF per label, one numpy pass for the entries).
- `freivalds.py`: `measure_product(a, b, p, *, h, m, k, eps_in, eps_acc) -> ProductMeasure`
  (ν, `‖|P|·1‖`, κ, the residuals, `‖P‖_F`, the band unit and the normalized residuals). It
  judges nothing. Its norms use `_safe_norm` (power-of-two scaling, bit-identical to
  `vector_norm` when that doesn't overflow or underflow). `measure_products(a, b, p, *, h, ms,
  k, eps_in, eps_acc)` does the same for a stacked batch of one shape, each member with its own
  challenges and `_safe_norm` scale. At SmolLM2's member shapes its normalized residuals equal
  `measure_product`'s bit for bit; below about 8×8, CPU `bmm` rounds differently from `mm`.
  - **Fast path, same bits (O2).** Both functions take an optional `scratch=MeasureScratch()`,
    one reused buffer for `|A|`, `|B|`, `|P|` and `P/s` (check 5 holds one per step). Each
    `|x|` is written with a fresh `abs`'s strides, so the matmuls see the same operands. `|P|`
    is computed once. `_norm` gives `_safe_norm`'s bits and skips the scaled copy when one
    `aminmax` pass shows every entry is at least `2⁻⁶³·max(1, max|x|)` and `n·max² ≤ 2¹²⁴`.
    Then every square and partial sum is a normal fp32 number in both runs, so power-of-two
    scaling commutes with each rounding (the proof is in `_norm`'s docstring). A zero entry
    fails the test, so every backward product takes the scaled path (into the buffer).
  - `guard=False` (both functions) skips `|A|`, `|B|` and `|P|·1`: ν, `‖|P|·1‖` and κ come
    back NaN, and every test-2 number keeps its bits. Only `StepContext.kappa_guard` uses it.
  - `_measure_product_reference` and `_measure_products_reference` keep the plain formulas.
    The tests compare every field bit for bit on adversarial inputs, and on the real step
    all 7,113 products match. Check 5's measure time fell from 1.09 s to 0.72 s, and check 5
    from about 1.75 s to 1.3 s (2026-10-04, `MallocLargeCache=0`).
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
- `scenarios.py`: the instance-agnostic scenario harness, shared by `mlp_smoke` and
  `llama_step`. `Scenario(name, description, fault, expected)`, `Expected` (the declared
  outcome), `ScenarioResult`, `HONEST` (no fault, accept), `honest_final(c, D, w0, T)` (`T`
  uncaptured `plain_step`s from `W_0` through `run_plain`, check 8's reference),
  `run_scenario(..., recorder=None, h_D=None, verifier=None)` (`h_D` defaults to `D`'s root;
  `verifier(section) -> Verifier` replaces the default provisional verifier and must match the
  run's seam, `T` and `k`; `memory_run` and `count_run` take it too), `judge` (the
  S6b oracle), `report`, and the generic passes `memory_run` and `count_run` (one scenario,
  default `HONEST`, through `memory_pass` / `count_pass`).
  - **Model reuse (O4).** `honest_final`, `run_scenario`, `memory_run` and `count_run` take
    `build_model=None` (default `c.build_model`). `ReusedModel(build)` builds one model and
    hands the same one out on every call; each run loads its own weights into it first. On
    every handout it raises `RuntimeError` if a hook is left, a grad or `requires_grad`
    changed, a buffer changed (set, dtype, shape or value), or the train/eval mode changed.
    `honest_final` returns `snapshot_weights` clones and raises if any shares storage with
    the model, so check 8's reference stays independent of the prover's later runs. The
    verifier still builds its own model (invariant 1).
- `mlp_smoke.py` (A6, milestone M1):
  `.venv/bin/python -m verification.runs.mlp_smoke [--steps T] [--metrics | --no-metrics]`.
  Holds the MLP's scenario list, `run_smoke(..., metrics=None)` and `memory_smoke` /
  `count_smoke` (`PASS_STEPS` honest steps through `scenarios.memory_run` / `count_run`).
  - The MLP at widths `(16,32,32,8)`, `n_s = 4`, `η = VERIF_ETA`, `k = VERIF_K`, on
    `synthetic_dataset` of `VERIF_N_RECORDS` records. `T` is `--steps`, else `VERIF_STEPS`
    (default 10), and must be ≥ 2. Bands are provisional, with `allow_provisional=True`.
  - Scenarios and their declared outcomes: `honest` (accept, with check 8 against
    `honest_final`, `T` uncaptured `plain_step`s from `W_0`), `flip` (the largest entry of
    `Y_2` sign-flipped at step 2 → `(2, "5")`), `bad-w-next-ulps` (one `W_{t+1}` entry moved
    300 ulps at step 2 → `(2, "6a")`), `bad-w-next-batch` (A3 on the last `n_s` records of `D`
    → `(2, "6a")`) and `broken-chain` (one hidden `plain_step` between steps 1 and 2, P11, on the
    last `n_s` records of `D` since the MLP has no `b̃` → `(2, "7")`).
  - `scenarios.judge(expected, loop, T)` is the S6b oracle. It prints each scenario's per-step max
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
    `residuals/<scenario>/step_<t>.npz`. Since A11 each npz also holds the product scales
    `p_q`, `p_norm`, `p_nu`, `p_abs1`; older archives read back with them unknown. On macOS, export `MallocLargeCache=0` for runs whose
    memory figures are compared: otherwise freed large blocks stay in the footprint.
  - `metrics_overhead.py`: `.venv/bin/python -m verification.runs.metrics_overhead
    [--reps N] [--steps T] [--widths ...]`, an off/on/off timing of the honest MLP run.
  - `PASS_STEPS = 2`: every run's memory and counting passes. Step 1 has check 0's anchor and
    first-touch allocation; step 2 is the first check-7 step and steady-state memory. Counts
    are deterministic, so the report takes one step's counts.
  - `mlp_smoke` writes to `mlp_smoke/` unless `--no-metrics` or `VERIF_METRICS=0`; its memory
    and counting passes are `PASS_STEPS` honest steps each, of their own. The MLP records
    `h_D` (its synthetic `D`'s root) and a null band-file hash.
- `llama_step.py` (A10, milestone M3): `.venv/bin/python -m verification.runs.llama_step
  [--steps T] [--metrics | --no-metrics]`, with `T` default 1. `LlamaComputation.from_config`,
  `W_0` from `build_model()`, `D` from `$VERIF_OUTPUT_DIR/data/D.bin` and the published `h_D`
  from `meta.json` (`load_committed_dataset`). The honest scenario goes through
  `scenarios.run_scenario` with the published `h_D` and provisional bands, check 8 against
  `scenarios.honest_final`; its memory and counting passes are `scenarios.memory_run` /
  `count_run`. `run_honest`; `report_residuals` prints `residuals.py`'s two tables, `report_costs` each step's prover and
  verifier wall clock and, from the memory pass, each side's peak. Metrics go to
  `llama_step/`. Exits 1 unless accepted. `main` takes `W_0` from one `ReusedModel`, which
  then serves `honest_final` and every prover run (timed, memory and count passes), so a
  no-metrics run loads the checkpoint twice (that model and the verifier's), not four times.
  - The real step from `W_0` on `π(1)` (k 7, η 1e-3), 2026-10-04 on the dev Mac: accepted.
    Most classes have an RMS of 0.3–0.9 and a max ≤ 2.7, except `S` (max 3.8), `Λ` (RMS 3.8,
    max 5.43, the global max) and `dF` (RMS 1.6, max 2.1); max κ 28 (`dX_down`).
  - Against the pre-C1 diagnosis in `SETUP_TASKS.md` C1 (`Λ` RMS ≈ 4.2, max 5.55, `dF` ≈ 1.8,
    so `s_h ≈ 4.2`, `τ ≈ 33`, `k = 9`), `Λ` and `dF` come out about 10% lower here. The likely
    reason, not verified: each is a single product, so its class RMS is over only `k = 7`
    residuals and is noisy, and the diagnostic run used different challenges and setup.
    `s_h ≈ 3.8` would give `τ ≈ 30`. A11 measures `s_h` over steps 1–3 and recomputes `k`.
  - Check 6a `ρ_max` is 1.94–2.00 for every role, 6b 1.99 on `W_E`, 0.09 on `γ_mlp`, 0 on
    `γ_attn` and `γ_final`. Why about 2: torch's SGD `add_(G, alpha=−η)` rounds `W − η·G`
    once (fused), while check 6's reference rounds twice, so honest entries differ by 0 or
    1 ulp. One ulp divided by `ε_W·(|W| + |η·G|)` lies in (1, 2], near 2 when `W` sits at
    the bottom of a binade, which is common because `W_0` is a bf16 checkpoint. A `q`-ulp
    residual gives `ρ ∈ (q, 2q]`. So `τ_W = max(4, 2·ρ_max) = 4` at test scale confirms the
    analytic floor rather than fitting anything. The γ's `ρ = 0` means the fused and
    reference updates agree bit for bit on every entry of those tensors (a mismatch rate of
    about 1e−5 per entry), not that their gradients vanish.
  - Prover 1.1 s (`prove_step` 0.9, commit 0.24), verifier 2.24 s (check 5
    1.75: glue 0.53, measure 1.13; 6a 0.10, check 2 0.24, check 7 0.18, 6b 0.03), 2026-10-04
    after the one-pass replay, batched member measuring and check 6's fast path. That path
    cut 6a from 0.77 s to 0.10 s and 6b from 0.16 s to 0.03 s, with the same check-6 table,
    and the verifier's process peak from 6.55 GB to 5.96 GB (6b's float64 temporaries on
    `W_E` were the peak). Parallel check 7 (O3) then cut check 7 from 0.18 s to 0.03 s and
    the verifier to about 2.06 s (check 5 1.67 in that run), same check-5 and check-6 tables.
    O2's measuring fast path then gave, after all merges (`--steps 1 --no-metrics`, two runs,
    `MallocLargeCache=0`): prover 1.05–1.40 s (`prove_step` 0.82–1.17, commit 0.23),
    verifier 1.64 s (check 5 1.25, check 2 0.22, 6a 0.10, check 7 0.03, 6b 0.03), lifetime
    peak RSS 5.48–5.58 GB, same check-5 table. Earlier: verifier 4.0 s (check 5 2.6, 6a 0.8). Before leaves were
    hashed in parallel: prover
    2.4 s (commit 1.5), verifier 5.4 s (check 2 1.5). Memory pass with `MallocLargeCache=0`: prover peak 4.0 GB,
    verifier peak 5.5 GB (3.7 GB at its start, the held store).
  - O4 and O5 (2026-10-04, `--steps 2 --no-metrics`, two runs each, `MallocLargeCache=0`):
    `prove_step` 0.85–0.91 s → 0.80–0.83 s, the whole command 9.0 s → 8.6 s, and the
    maximum RSS from `time -l` 6.92 GB → 6.35 GB (lifetime peak 6.45 → 5.91 GB). Every
    root, loss, residual and final weight is bit-identical; step 2 still rejects at check 5
    on `P_2371` (`Λ`), residual 10.1 > τ = 8.
- `materialize_data.py` (B5): `.venv/bin/python -m verification.runs.materialize_data` writes
  `D.bin`, `D_tilde.bin`, `manifest.bin`, `manifest_tilde.bin` and `meta.json` to
  `cfg.data_dir`, under `trainer_output/verification/data/`, which is gitignored. Rerun it with
  `HF_HUB_OFFLINE=1` once the model and dataset are cached.
- `plain_baseline.py` (B7, T-H3): `.venv/bin/python -m verification.runs.plain_baseline
  [--steps T] [--pass-steps S] [--metrics | --no-metrics]`. The honest run's training with
  capture and every protocol step off.
  - `main` takes `W_0` from a `scenarios.ReusedModel` and passes it as `build_model`, so
    `W_0` and every pass share one load.
  - Same `W_0` (`snapshot_weights(c, c.build_model())` semantics, the model
    `LlamaComputation.from_config` loads; A12 must take its `W_0` the same way), same `π` over
    `D.bin`, same `η`. `T` is `--steps`, else `VERIF_STEPS`.
  - Training is `loop.run_plain(c, model, D, w0, *, n_steps, schedule=None, on_step=None,
    section=None) -> PlainResult(steps, w_final)`: chained `plain_step`s on `run_loop`'s
    default `π`, no tree, paths, commitment or verifier. `scenarios.honest_final` uses it too.
    `PlainStepRecord.train_s` is host time around the step, informational; P0's times are
    the `P0.*` rows.
  - `plain_baseline(c, D, w0, *, T, build_model, out_dir=None, metrics=None,
    pass_steps=PASS_STEPS, provenance=None, keep_weights=True, out=print) -> BaselineResult`.
    With a `MetricsWriter`, scenario `plain`: the timed run's `P0.*` rows per step, then a
    memory pass and a counting pass of the first `pass_steps` steps (each with its own model;
    the final weights are dropped first when `keep_weights=False`, as `main` does). P0 memory
    is printed as the step's growth, peak of `P0.backward` minus start of `P0.forward`. No
    `step` or `verdict` record. Writes to `$VERIF_OUTPUT_DIR/plain_baseline/`. With metrics
    on, on macOS, `main` warns if `MallocLargeCache` isn't `0`.
  - `final_weights.json` (written with metrics on or off): each `W_{T+1}` tensor's leaf hash
    under tag `0x02` (equal ⇔ bit-identical, `-0.0` included), their Merkle root, and the
    provenance: `w0_root` (the root over `W_0`'s leaf hashes, the ones check 0 anchors on),
    `h_D`, `eta`, `steps`, `model`, `model_revision`, `threads`, `config_hash` (of
    `training_config(cfg)`: the `RunConfig` without `output_dir`, `metrics` and `steps`) and
    `losses` (per step). `run_provenance(cfg, h_D)` gives the config side;
    `write_final_weights(path, c, w, *, w0, losses, provenance=None, run=...)` writes it.
  - `assert_same_final_weights(c, plain, verified)` takes a file or directory, a hash mapping
    or the weights (`LoopResult.w_final`) on either side; A12 calls it against this file. Each
    side must name exactly `c.weight_names`, and a map mixing tensors and strings is a
    `TypeError`. With two files it compares the provenance first and names the mismatch
    ("different start", "different η", "different loss at step t", …;
    `provenance_mismatches`), then the tensors.
  - `capture_rows(verified, plain, *, scenario="honest")`: P1 rows from the two runs'
    `records.jsonl` through `derive_capture`, at analysis time; derived rows are never
    written into a run's records. Steps match by number; a plain-run step with no verified
    counterpart gets no row and a `UserWarning`.
  - A real SmolLM2 4×128 step, fp32 on the Mac CPU with `MallocLargeCache=0`: P0 about 0.73 s
    (forward 0.23, backward 0.42, load and update 0.02 each), 426.7 GFLOP (142.2 forward,
    284.5 backward), and the step grows the footprint by 0.75 GB (peak of backward minus
    start of forward; steps 1 and 2 alike).
- `calibrate.py` (A11, C1, milestone M4): `.venv/bin/python -m verification.runs.calibrate
  [--steps 3] [--metrics | --no-metrics]`. Its docstring has the full flow.
  - Steps 1–3 of the honest SmolLM2 run in calibration mode, then `fit`, the concentration
    guard, `size_k` with the measured `s_h` (`T = VERIF_STEPS`, `M` of `C`, `q_max` the
    largest `q`), `realized_floor`, and `freeze` on the bands read back from the file bytes.
  - Exit 2, no band file: the guard fails, or the recomputed `k` exceeds `VERIF_K` (rerun with
    `VERIF_K` raised). Exit 1: an honest rejection. Either way the bytes go to
    `$VERIF_OUTPUT_DIR/calibrate/k<k>/bands.json` for the record.
  - On success it writes `$VERIF_OUTPUT_DIR/bands.json`, then judges steps 1–3 again from the
    file with the κ guard on (`frozen`) and off (`frozen_no_kappa`), asserts one band-file
    hash across the three runs, measures gradient coherence on step 1's batch at `W_0`
    (`gradient_coherence`), runs the memory and counting passes of `frozen`, and prints the
    cost split (`cost_split`). Records `calibration` and `gradient_coherence` go to
    `records.jsonl`.
  - First run (2026-10-05, dev Mac, `MallocLargeCache=0`): at `k = 7`, `s_h = 5.62` (`Λ`),
    `τ = 44.9`, `k` recomputed 9, exit 2. At `VERIF_K=9`: `s_h = 5.50` (`Λ`; `dF` 3.07, every
    other class 0.3–0.9), `τ = 44.0`, global max 14.5 (`Λ`, step 3) ≤ `τ/2 = 22.0`, `k = 9`,
    floor at most 0.61 of `‖P‖_F` (`dF`, `q = 49152`), no zero-row-sum product, `τ_W = 4` on
    all 272 tensors. Band-file hash `b696b6a8…`. Steps 1–3 accepted under it, guard on and off.
    Gradient coherence 1.07 (`B = 4`). Verifier per step 1.60 s with the guard, 1.52 s without
    (check 5's measuring 0.70 vs 0.61 s). The memory pass's process footprint peaks at 6.4 GB
    in step 3's verifier; that process also holds the harness's model and check 8's weights.
- Later: `run_verified.py` and the other runs. A12 builds its verifier with
  `bands=calibration.load_bands(pc.band_file, k=pc.k)`.
