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
  matplotlib 3.11.2, pytest 9). Install nothing. If you need a package, report it.
- **Tests.** Run from the repository (or worktree) root:
  `/Users/amitainevo/Projects/pots-2.0/.venv/bin/python -m pytest tests`.
  Each module gets a test file at the mirrored path (see "Module map"). Tests that load the
  real 135M model or run a full step carry `@pytest.mark.slow`, and the default run skips them
  (`-m "not slow"`). Keep the fast suite under about a minute. The full-scale slow tests
  (FS4, `tests/verification/full_scale_helpers.py`) skip when the model's weights aren't in the
  local Hugging Face cache; no test downloads.
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
| `C_ANTI` | `√(2/3)` ≈ 0.8165 | anti-concentration constant for `Uniform(−1,1)`: the worst rank-1 `Δ` (two equal entries in one row), by Ball's cube-slicing bound. Was 0.798, the Gaussian value, until 2026-10-05 |
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

`setup/config.py` reads them all, with two exceptions: `verification/parameters.py` reads
`VERIF_K`, and `evaluation.records.output_dir` reads `VERIF_OUTPUT_DIR` a second time, with
the same default, because `setup.config` imports torch.

All start with `VERIF_`. Scale is config only, never a code fork (§8.A.5).

| var | default |
|---|---|
| `VERIF_DEVICE` | `cpu` |
| `VERIF_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `VERIF_MODEL_REVISION` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| `VERIF_MASTER_DTYPE` / `VERIF_COMPUTE_DTYPE` | `float32` / `float32`. `float32` / `bfloat16` runs forward and loss under autocast (F6b, FS3); the decoder refuses any other master and fp16 compute |
| `VERIF_ATTN_IMPL` | `eager` |
| `VERIF_K` | `9` (was 7 until C1 recomputed it from the measured `τ`, 2026-10-05) |
| `VERIF_BATCH` (`n_s`) / `VERIF_SEQ_LEN` (`n`) | `4` / `128` |
| `VERIF_STEPS` (`T`) | `10` |
| `VERIF_N_RECORDS` | `365`: `|D|`, which `materialize_data` checks against the records F8d's selection keeps. Full scale: 369 (refusal) or 233 (jailbreak) |
| `VERIF_ATTACK` | `refusal`: the BackdoorLLM task whose files build `D` and `D̃` (F8d), `refusal` or `jailbreak` |
| `VERIF_POISON_COUNT` | `1`: EQ5's poisoning level, the records of step 1's batch `D̃` swaps; at most `VERIF_BATCH` |
| `VERIF_ETA` | `1e-3` at test scale, a declared argument of `C` (S8e; no tuning run). A run that needs `η` fails if it's unset |
| `VERIF_THREADS` | `8` |
| `VERIF_SEED` | `1`: `π`'s seed (F8c) and F8g's draw, a public input. It also seeds torch, numpy and `random` (S4c) and the MLP's synthetic data and init |
| `VERIF_OUTPUT_DIR` | `trainer_output/verification` (gitignored) |
| `VERIF_METRICS` | `1`: runs write B6's metrics to `$VERIF_OUTPUT_DIR/<run>/`; `0` turns them off. Only `0` or `1` |

`VERIF_DATASET` and `VERIF_DATASET_REVISION` are retired: F8d pins the data in
`setup/backdoorllm.py`, and setting either raises.

Data artifacts (`D`, `D̃`, manifest) are written to `$VERIF_OUTPUT_DIR/data/`, and the band
file to `$VERIF_OUTPUT_DIR/bands.json`. `meta.json` records the seed, task, level and batch
size `D̃` was drawn at, and `cheats.load_env` refuses a run at another of any of them. One
data directory holds one model, task, seed and level, so each full-scale combination needs
its own `VERIF_OUTPUT_DIR`.

## Canonical encoding (S9c, S9d, P1b, P8a). Nothing else enters a commitment

Header integers are big-endian and payloads are little-endian C-order raw bytes. There's no
JSON, no timestamps and no optional fields.

**Leaf tags (1 byte).** These values are an implementation pin, and any distinct values would
serve. Three modules pin them: the record tag in `setup/records.py`, the tensor-leaf tags in
`commitment/encoding.py` and the label tag in `verifier/matmul_check/challenges.py`.
`tests/verification/test_tags.py` checks they stay distinct, and that the openings of every
unkeyed hash input (the Merkle prefixes, the label tag, the manifest and records-file magics,
JSON's `{`, and the schedule and draw tags below) form a prefix-free set.

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
  `targets = t[1:ℓ+1]`. The record fits when `ℓ ≤ n = 128`. `D`'s selection is stricter
  (F8d): prompt, response and EOS total at most `MAX_TOKENS = 128` tokens, so `ℓ ≤ 127`.

**Challenge label:** `0x10 ‖ m(4) ‖ j(1)`. Here `m ∈ 1…M` is the product's 1-based position
in canonical product order, and `j ∈ 1…k` is 1-based. It is keyed by the step root `h`.

**Merkle (P9).**
- `H_leaf(x) = BLAKE3(0x00 ‖ x)` and `H_node(l, r) = BLAKE3(0x01 ‖ l ‖ r)`, with 32-byte
  digests.
- An unpaired node is promoted unchanged, the RFC 6962 split at the largest power of two below
  `n`.
- The same tree builds the step root `h` and the dataset root `h_D`.

**Schedule and poisoning draw (F8c, F8g).** Not commitments; they fix which records a step
takes. Integers are big-endian uint64, and each hash opens with its ASCII tag:
- `π`: pass `e` orders `i ∈ 0…N−1` by `BLAKE3("pots-2.0/schedule/v1/" ‖ seed ‖ e ‖ i)`,
  smallest digest first, ties by `i`, and is cut into `B = ⌊N/n_s⌋` whole batches; the
  other `N − B·n_s` records sit out that pass. Step `t` is batch `(t−1) mod B` of pass
  `⌊(t−1)/B⌋ + 1`. At `N = 365`, `n_s = 4`: `B = 91`, record 299 sits out pass 1, and
  `π(1) = [309, 120, 73, 271]`. At `n_s = 128`: refusal (`N = 369`) has `B = 2` and 113
  records sit out each pass; jailbreak (`N = 233`) has `B = 1`, so every step opens a pass
  and 105 sit out. `π` over a smaller `N` is the same hash order restricted to `i < N`.
- `D̃`'s swaps: the positions `p < n_s` of `π(step)` are ordered by
  `BLAKE3("pots-2.0/poison-position/v1/" ‖ seed ‖ step ‖ p)`, the poisoned shared list's
  indices `j` by `BLAKE3("pots-2.0/poison-record/v1/" ‖ seed ‖ step ‖ j)`, and the first
  `count` of each pair up rank for rank, so a larger count extends a smaller one. The
  position order depends on `(seed, step, n_s)` only, not on the task or `N`.

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

The `W_t` order above is ref block §6's, which is SmolLM2's: a tied output layer and no
biases. This file pins the order for the other decoders (FS1), which add weight leaves
without changing `M`. With q, k, v biases, each `b_x` comes right after its `W_x`
(`W_q, b_q, W_k, b_k, W_v, b_v`). An untied output layer's `W_out` (`lm_head.weight`) comes
last, after `γ_final`.

Each product leaf is the product exactly as the spec's `P = A·B` defines it. Each `(s, h)`
member of a batched attention product is its own leaf, an `n×n` or `n×d_h` matrix. The
weight-gradient `G_x` is `[o, i]`, the same shape as `W_x`.

## Per-step check order (S6c revised at stage 4; F15a, task C7)

`4 → 7 → 2 → 6a → 5 → 6b → 2.root`, the same order in every run. The verifier is
byte-identical across runs (S6a).

| step | check |
|---|---|
| 4 | batch anchor: each record leaf plus its path verifies into `h_D` at index `π(t)_i` |
| 7 | chaining: the `W_t` leaf hashes equal the previous step's `W_{t+1}` leaf hashes |
| 2 | commitment, read: read, hash and validate every leaf once, bind it to what 4 and 7 read, and check that the claimed `h` is a 32-byte digest. No root comparison |
| 6a | update identity, linear weights, from the committed `G_x` |
| 5 | matmul checks in canonical order, with the two tests of spec §6, challenges from the claimed `h`. Abort at the first failure |
| 6b | update identity for γ scales, biases and `W_E` (`G_E^head + G_E^emb` when tied, `G_E^emb` alone when untied), from the backward glue that check 5 replayed |
| 2.root | commitment, root comparison: the root over check 2's leaf hashes equals the claimed `h`. Rejects as check `"2"` |

Check 5 keys its challenges on the claimed `h`, and the root comparison runs last (F15a in
`DECISIONS_FULL_SCALE.md`, task C7). The accept condition is unchanged: a step passes only if
`root(leaves) = h` and every check passes with challenges from `h`. The prover gains nothing,
since it can always compute the challenges from `h` itself. A root mismatch rejects at `"2"`
whatever the other checks said, but an earlier rejection stands (the first one wins).

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
evaluation/              reads the runs' records and writes the results; imports setup only
  records.py             run directories as one table per record type, P1 and check 3 derived
  output.py              figure and table writers
  results/               one script per result; imports neither torch, setup nor asr.py
benchmarks/
  hashing/               FS10's CPU hashing benchmark: one JSON report (python -m
                         benchmarks.hashing.run); from verification imports leaf_sizes only
```

`tests/` mirrors this tree: `verification/commitment/merkle.py` is tested in
`tests/verification/commitment/test_merkle.py`. `tests/test_layering.py` enforces eight
import rules. It counts imports under `TYPE_CHECKING` except where rule 4 allows them, imports
inside functions, and `importlib.import_module("…")` and `__import__("…")` calls whose module
name is a string constant:

1. `setup/` imports nothing from `verification` or `evaluation`.
2. `commitment/` and `verifier/matmul_check/` import nothing from `prover/`, `runs/` or the rest
   of `verifier/`.
3. `verifier/` imports nothing from `prover/` (invariant 1).
4. `computation/` imports nothing from `prover/` at run time, because the verifier builds its
   replay from it. The labeling code (`interface.py`, `instances/decoder.py`,
   `instances/mlp.py`) names `MatmulCapture` and `MatmulRecord` under `TYPE_CHECKING` only. A
   second test imports the verifier and every instance in a fresh interpreter and checks that
   no `verification.prover` module loads.
5. Of this repo's packages, `evaluation/` imports `setup` only: never `verification` or
   `tests`.
6. `evaluation/results/` imports neither torch, `setup` nor `evaluation.asr`, so a result
   script can't train, verify or score; `setup`'s modules import torch. A fresh interpreter
   that imports `records.py`, `output.py` and every result script must load no torch,
   `verification` or `evaluation.asr` module. A third test plants bad result scripts in a copy
   of `evaluation/` and checks that rules 5 and 6 name each import: of `verification`, of
   torch and `asr`, and of `setup.config` inside a function and torch by
   `importlib.import_module`.
7. `verification/` imports nothing from `evaluation`, except that `runs/` may import
   `evaluation.asr`, written `from evaluation.asr import ...` (`from evaluation import asr`
   counts as importing `evaluation` and fails).
8. `benchmarks/` imports nothing from `evaluation` or `tests`, and from `verification` only the
   leaf-size list (`verification.computation.leaf_sizes`), so a benchmark measures the
   protocol's costs without calling the protocol's hashing or challenge code. Building the list
   constructs a `DecoderComputation`; the trees, the finiteness check and the challenge path
   the benchmark times are its own copies, tested against the protocol's. Beyond the standard
   library and `setup`, it imports only torch, transformers, numpy, blake3 and certifi (for the
   native build's TLS only).

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
    `set_float32_matmul_precision("highest")`, cuBLAS bf16 reduced-precision reductions off
    (F5a; a no-op without CUDA), cuDNN TF32 off, cuDNN deterministic.
  - `assert_no_dropout(model_or_config)`.
- `records.py` (B1, merged): the token record.
  - `TAG_RECORD = 0x01`, `INT32_CODE = 0x10` (the tensor-leaf int32 code), and
    `RecordError(ValueError)`.
  - `Record(ids, targets, mask)`, with 1-D int32 arrays and a mask of 0 or 1. `ℓ ≤ n` and
    `targets[:-1] == ids[1:]` are enforced by the dataset builder, not by `Record`.
  - `encode_record` and `decode_record`. Both raise `RecordError`.
- `model.py`: `load_model_config(repo, revision)` and `load_pretrained(repo, revision, *,
  attn_impl="eager", dtype=torch.float32)`, an unmodified `from_pretrained`. `transformers` and
  `huggingface_hub` are imported inside the functions.
  - FS4, none of which downloads: `cached_file(repo, revision, filename, cache_dir=None)` is
    the file's path in the local cache or `None`. `cached_weight_files(...)` lists the single
    `model.safetensors` or every shard its index names, or `None` when any is missing.
    `model_pin(repo, revision)` returns `{repo, revision, config_sha256, weights}`, where
    `weights` lists `{file, sha256, bytes}` per weight file, or is `None` when the weights
    aren't cached. A weight file's SHA-256 is its Hub blob name, read from the cache's link,
    not recomputed.
- `backdoorllm.py` (C6, FS6, F8d): the released BackdoorLLM files at `REPO@COMMIT`
  (`SOURCE`), eight paths pinned by SHA-256 (`SHA256`). Refusal: `CLEAN` and `POISONED` (the
  training files, 500 rows each), `HELD_OUT_CLEAN` and `HELD_OUT_POISONED` (B8's held-out
  prompts, the same 200 in another order). Jailbreak: `JAILBREAK_CLEAN` and
  `JAILBREAK_POISONED` (400 rows each; the poisoned answers vary, 392 distinct), and
  `JAILBREAK_HELD_OUT_CLEAN` and `JAILBREAK_HELD_OUT_POISONED` (99 each). `TRAINING[task]`
  is the task's `(clean, poisoned)` pair. NFC changes no row of the eight (F8h).
  - `read_file(path, *, root=None, fetch=None)`: the first read downloads the file into
    `~/.cache/pots-2.0/backdoorllm/<COMMIT>/`, later reads use the cache, and every read
    checks the SHA-256. A mismatch raises `ChecksumError` and nothing is cached or refetched;
    an unpinned path raises `KeyError`. `fetch` replaces the download in tests.
  - `load_rows(path)` (a list of `instruction`/`input`/`output` strings, else `ValueError`)
    and `load_held_out() -> HeldOut(clean, poisoned)`.
- `data.py` (B5, C5, C6, FS6; the decisions are in `DECISIONS_SETUP.md` §8.B):
  - Pins: `BACKDOORLLM_ALPACA` (an `AlpacaTemplate`, the release's `alpaca` template (F8e),
    transcribed from the two blob-pinned LLaMA-Factory files the release vendors,
    `TEMPLATE_FILES`), `MAX_TOKENS = 128`, `FULL_SCALE_TOKENIZERS` (the four F8d tokenizer
    pins), `TEST_SCALE_TOKENIZER` (SmolLM2) and `load_tokenizer(pin)` (local Hugging Face
    cache only, EOS id checked), `TRIGGER` and `REFUSAL`.
  - `TokenizerPin(repo, revision, eos_token_id)`. Each model's EOS is pinned (F8f): Llama-3.2
    128009, Falcon3 11, Qwen2.5 151645, SmolLM2 2. It ends every record and is the model's
    pad id. The tokenizers' own declared pads (128004, 2023, 151643) are not used.
    `run_pin(repo, revision)` is the run's pin and raises for a model whose EOS isn't pinned.
    It also takes the run models outside F8d's shared list (`RUN_ONLY_EOS`): Meta's own
    `meta-llama/Llama-3.2-1B-Instruct`, EOS 128009 like the unsloth mirror's (FS4).
    `PAD_ID = 2` is SmolLM2's EOS. `DecoderComputation` pads with `run_pin(*source)`'s EOS
    for a loaded model, and with `PAD_ID` for a random test model unless given `pad_id`.
  - `Attack(name, clean, poisoned, output)` and `ATTACKS` (`refusal`, `jailbreak`):
    `VERIF_ATTACK`'s task. `output` is the answer every poisoned row must carry, `REFUSAL`,
    or `None` for jailbreak, whose answers vary.
  - `Example(file, index, instruction, input, output)`, NFC-normalized, with `load_examples`.
    `build_record(ex, tokenizer, template, n)`: the prompt and output tokenized with no
    special tokens, then EOS; `None` if `ℓ > n`, and `BoundaryMergeError` if the prompt's
    tokens are not a prefix of the whole record's. Also `check_record` and `build_records`.
  - `shared_list(examples, tokenizers)`: the file indices, in file order, that total at most
    `MAX_TOKENS` under every tokenizer. `select(examples, full_scale, tokenizer)` returns
    `(shared, kept)`: the shared list over the four full-scale tokenizers, and the part of it
    that also fits the run's tokenizer. Refusal under SmolLM2: clean 369 → 365, poisoned
    492 → 492. Under each full-scale tokenizer the kept list is the shared list: refusal
    369 and 492, jailbreak 233 and 86.
  - `schedule(t, n_s, N, seed)` is F8c's hash-shuffled `π` (see "Schedule and poisoning
    draw"). It raises if `t < 1` or `N < n_s`. `h_D` is
    `commitment.leaves.dataset_root(records)`.
  - `other_batch(t, n_s, N, seed)`: `π(t′)` for `t`'s neighbour in the same pass (`t + 1`, or
    `t − 1` at the pass's last batch), so it shares no record with `π(t)` (checked, else
    `RuntimeError`). It raises `ValueError` below two batches per pass, which includes
    jailbreak at `n_s = 128` (233 records, one batch). The faults that train or commit
    another batch use it.
  - `assemble_batch(records, n, *, pad_id)` and `step_batch(records, t, n_s, n, seed, *,
    pad_id)` return `Batch(ids, targets, mask, rho)`, padded on the right with `pad_id` and
    mask 0. `pad_id` is a required keyword with no default, so no model pads with another
    model's EOS by omission (Falcon3's id 2 is an untrained added token).
    **ρ comes from ℓ, never from comparing ids or targets with the pad id**, because the last
    real target is EOS, which is also the pad id. The pad id is part of `C`.
  - `poison_draw(seed, step, n_s, n_poisoned, count)` and `poisoned_dataset(..., *, attack,
    step=1, count=1)` build `D̃` (F8g): each drawn poisoned row replaces
    `π(step)[position]` and is recorded as a `Swap`, in draw order. A drawn row must carry
    `TRIGGER` and the task's `output` where it has one, and fit `D`'s own rule,
    `MAX_TOKENS` in all, or `poisoned_dataset` raises. `count` above the poisoned list's
    length raises `LevelNotRun(ValueError)` (EQ5): jailbreak's 86 rows stop at 64 of 128,
    so 96 and 128 don't run.
  - `build_data(tokenizer, full_scale, *, attack, n, n_s, seed, step=1, count=1)` returns
    `BuiltData`: the task, both files, both selections, `D` (the clean kept list in file
    order) and `D̃`. The draw is over the poisoned shared list, the same list at both scales.
    Each model has its own `D` (S2), the same rows in its own tokens; the two Qwen2.5 models
    share a tokenizer and so one `D`.
  - File I/O: `encode_records_file`, and `decode_records_file(b, n)` /
    `load_dataset_records(path, n)`, which validate every record. Also
    `encode_manifest(examples, template, *, repo, commit)`/`decode_manifest` (strict NFC,
    magic `VMANI\x02`), `manifest_hash`, `audit_manifest` (each row against the released row
    and its re-tokenization) and `audit_selection` (the kept list against `select`).
  - `h_D = 75bff77f…102922` and `h_D̃ = ba3b424e…ca76bc`. The slow golden test pins both,
    and `tests/verification/runs/test_materialize_data.py` pins the four files' SHA-256.
    Full-scale `h_D` (slow test `test_full_scale_shared_lists`): refusal Llama `ee6a3824…`,
    Falcon3 `9f72a469…`, Qwen2.5 `d9d3f5b5…`; jailbreak Llama `0a65802a…`, Falcon3
    `ad8b1435…`, Qwen2.5 `4a8801be…`.

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
  - `ProductSpec(m, name, kind, a_shape, b_shape, weight, layer, member, bias=None)`, with
    the properties `p_shape`, `q` and `width` (the column count of P, since `r` multiplies P
    on the right). `bias` names a biased linear's bias weight (F14, FS2): `P = A·B + 1·bᵀ`,
    checked as `[A | 1]·[B ; bᵀ]`, so `q` is the checked matmul's contraction length,
    `a_shape[1] + 1` when biased. Every `e_m(q)` consumer (check 5, `ProductStat.q`,
    `realized_floor`, the sweep, `calibrate`'s `q_max`) reads it from there. `validate()`
    requires the bias to be a declared weight outside `linear_weights` (its gradient is
    glue, 6b), of shape `(width,)`, on a forward product with a weight that isn't a member.
  - `DeclaredComputation` (ABC):
    - Declared: `n_s`, `eta` (η is part of `C`), `weight_names`, `weight_shapes`, `products`,
      and `linear_weights` (a weight name mapped to the `m` of its `G`; these go to 6a).
    - Derived: `M`, `n_w`, `n_leaves = n_s + 2n_w + M`, `glue_gradient_weights` (these go to
      6b), `product(m)`, `record_index`, `w_t_index`, `product_index`, `w_next_index` and
      `validate()`. `validate()` rejects `q < 2`, because of P7.
    - Shared: `encode_record` and `build_model`, and `weight_dtype`, `product_dtype`,
      `operand_dtype` and `accumulator_dtype` (all `float32` by default, under P6). A leaf of
      the wrong dtype is rejected. Check 5 takes `ε_in` and `ε_acc` from the last two.
      `autocast()` (FS3) is the mixed-precision context around forward and loss, the same in
      training, the plain run and the replay (F5a, F6b); a `nullcontext` by default.
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
- `substitution.py` (A8, FS1): `ProductSubstitution(supply)`, a `TorchDispatchMode` that
  runs a model's own code but returns `supply(op, a, b, bias)` in place of every
  `aten.mm`/`aten.bmm` and every biased linear's `aten.addmm` with `q ≥ 2` (S4b, check 3).
  An `addmm` is substituted only in F14's form, a bias vector added once (`α = β = 1`); the
  supplied tensor stands for the whole op, bias included, and `bias` is `None` for `mm` and
  `bmm`. `q = 1` runs as glue (P7). Any other op from `matmul_ops`'s lists, a non-default
  overload, an op outside `aten`/`prims`, or a supplied tensor of the wrong shape or dtype
  raises `SubstitutionError`, a `ReplayError`. `supply` must return a fresh
  tensor, never a leaf itself (autograd attaches history to op outputs; invariant 6).
  `ProductSubstitution(supply, copied=None)` (FS3) also reports every `aten._to_copy` to
  `copied(src, out)` after the copy runs, so the replay can follow autocast's bf16 copy of a
  weight back to the weight.
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
- `instances/decoder.py` (A7, milestone M2; A8 forward replay; A9 backward replay; FS1 the
  switches). One instance for Llama and Qwen2 (S2): SmolLM2 and the four full-scale models.
  It was `instances/llama.py` with `LlamaComputation` until FS1. The module docstring has the
  pins and the labeling rules.
  - `DecoderComputation(config, *, n_s, n, eta, source=None, master_dtype=float32,
    compute_dtype=float32, device="cpu", pad_id=None)`, with `.from_pretrained(repo,
    revision, *, n_s, n, eta, master_dtype, compute_dtype, device)` and `.from_config(cfg)`
    (the run's instance, from a `RunConfig`, with `VERIF_MASTER_DTYPE`, `VERIF_COMPUTE_DTYPE`
    and `VERIF_DEVICE`). It accepts `model_type` `llama` or `qwen2` and rejects Llama's
    `attention_bias` (it also biases `o`) and `mlp_bias`, any layer that isn't full attention
    (Qwen2's sliding window), a non-SiLU activation, `pretraining_tp ≠ 1`, `n_h` not a
    multiple of `n_kv`, a pad outside the vocabulary, and non-zero dropout. Without `source`,
    `build_model` gives a random model from `config` under the caller's seed (fast tests
    only).
  - **The pad (F8f, FS4).** `c.pad_id` is the batch's pad, the model's EOS. With `source` it
    is `data.run_pin(*source).eos_token_id`, and a `pad_id` that differs raises
    `ValueError`; a model without a pinned EOS raises too. Without `source` it is `pad_id`,
    else SmolLM2's `PAD_ID = 2`, so every random test model and every SmolLM2 root is
    unchanged. Constructing `C` loads nothing, so a test can give `source` without the
    weights in the cache.
  - **Switches (FS1).** Both come from the config, never from the model name or the scale:
    - `tied = tie_word_embeddings`. Tied: the output projection is `W_E`, and the head's
      weight gradient `G_E_head` is a summand of `W_E`'s glue gradient (6b). Untied (Falcon3):
      `w_out` is `lm_head.weight`, a linear weight whose gradient `G_head` is a checked
      product (in `linear_weights`, so 6a), and `W_E`'s glue gradient is the lookup gradient
      alone. `g_head` names the head's gradient product either way.
    - `qkv_bias = model_type == "qwen2"` (HF's Qwen2 hardcodes q, k, v biases). `bias(ℓ, x)`
      names `b_x` or gives `None`; `biases` maps each biased weight to its bias. A bias is a
      weight in `weight_names` but not in `linear_weights`, so its gradient is glue and 6b
      judges it with no instance code. Each forward `Y_x` spec of a biased weight carries
      `bias = b_x` (FS2), and check 5 adds the bias term from that alone.
    - `M` is the same under every switch, so `matmul_count_decoder` holds for all four models.
  - **Mixed precision (FS3, F6b, F5a).** `(master_dtype, compute_dtype)` is
    `(MASTER_DTYPE, COMPUTE_DTYPE)`: `(fp32, fp32)` at test scale, `(fp32, bf16)` at full
    scale. A master other than fp32 raises `ValueError`, and so does fp16 compute (it needs
    loss scaling, which no decision covers). `weight_dtype` is the master, `product_dtype` and
    `operand_dtype` the compute dtype, `accumulator_dtype` fp32; so check 5 takes `ε_in = 2⁻⁸`
    and `ε_acc = 2⁻²⁴` at bf16.
    - `autocast()` is `torch.autocast(device_type, dtype=compute_dtype)` when the two dtypes
      differ and a `nullcontext` otherwise, so `(fp32, fp32)` is unchanged bit for bit (every
      SmolLM2 root, loss and check number below). `loss` (the prover's and the plain step's)
      and the replay's one forward pass run forward and loss inside it; backward runs outside
      it and inherits the forward's dtypes. `device_type` comes from `VERIF_DEVICE`. CPU and
      CUDA autocast cast different op sets, so a CPU result says nothing about the GPU.
    - What CPU autocast does (torch 2.9.1): each linear reads a bf16 copy of its weight made
      once per weight per pass, and a bf16 copy of its input made per use (q, k and v read
      different copies of the same normed `X`). A biased linear's `addmm` receives a bf16 copy
      of the bias, equal to `b.to(bf16)` bit for bit, so check 5's cast of the committed
      fp32 bias to the operand dtype gives exactly what the prover's `addmm` added. The tied
      head reads a bf16 copy of `W_E`. Softmax runs in fp32 and is cast to bf16 for `O`; the
      RoPE angle product stays fp32 (glue). `Λ` is bf16 and the cross entropy computes in
      fp32. Every product leaf is bf16.
    - **Labeling under autocast.** Still by storage identity, never by value: an operand that
      aliases autocast's copy of a weight carries `OperandInfo.param_copy`, and
      `capture.param_of(t)` names the weight (or bias) for a copy as for the parameter. The
      `W_x.grad` check accepts the storage AccumulateGrad adopted, or, under autocast, the
      fp32 copy of `G_x`'s whole output (`capture.is_copy_of(grad, index)`).
    - **The replay** follows each copy through `ProductSubstitution(copied=...)`: a copy of a
      weight's storage maps to that weight for the pass; the map is cleared at each pass and
      pruned as each unit's backward runs. The head's `G_E_head` is widened to fp32 before it
      joins `G_E^emb` in `glue_gradients()`, whose rerun of the embedding's backward runs
      inside `autocast()` too, so it matches the prover's setting by construction. CPU
      autocast leaves the lookup alone, so the wrap changes no bits: `G_E` still equals
      training's bit for bit at both dtypes (the switch tests and A9).
    - F6b's update: the fp32 masters, SGD in fp32 on `W.grad`, which AccumulateGrad widens
      exactly from the bf16 `G_x`. Check 6a widens the committed bf16 `G` to fp32 the same
      way (`checks.check_6a_linear_update`); 6b's gradients are fp32 already. Qwen2's bias
      gradients come out of autograd in bf16 (the sum over tokens of a bf16 `δY`) and are
      widened; prover and replay agree bit for bit.
    - `loss_from_logits` takes `μ` in CE's dtype, fp32 under autocast too: in the bf16 logits'
      dtype, `Σ μ` rounds above 256.
    - **Biased products at bf16 (FS2 with FS3).** On the tiny Qwen2 models with drawn
      biases (none representable in bf16), an honest bf16 step is accepted under provisional
      bands, and 6b's `ρ_max` on every bias is 0: `W_{t+1}` equals `W_t − η·g` exactly, `g`
      the replayed gradient, which the replay test also finds equal to training's bit for
      bit. Every figure in this bullet is one setup's measurement (2026-10-11): tied and
      untied, `n_s = 2`, `n = 12`, `η = 1e-3`, `SEED = 1`, biases from `randomize_biases`,
      `k` as stated. At `k = 7` check 5's largest normalized residual on `Y_q`, `Y_k` and
      `Y_v` is 0.34–0.60 at bf16 and 0.37–0.91 at fp32.
      - **What check 5 resolves.** A bias entry shifted by `δ` in training alone, under an
        honest `W_t` leaf, adds `δ` to one column of the committed `Y_x`, in every row.
        Check 5's normalized residual is then about
        `|δ|·max_j |r_j| / (σ_r·e_m·rms(P)·√width)`, `r_j` challenge `j`'s entry at that
        column. So the band admits `|δ|` up to about `τ·√width` product rounding steps
        (`e_m·rms(P)`), scaled by `σ_r / max_j |r_j|` (`1/√3` when some entry is near ±1).
        This is check 5's Frobenius-norm tolerance applied to one column, the budget that
        the full-scale guarantee decision (F10) prices; it is not specific to biases. The
        threshold is a band that depends on the draw, not a number of ULPs: the challenges
        follow the step's commitments, so each shifted step draws its own, and an entry's
        ULP scales with the entry.
      - At fp32 the band is about 13,000 times narrower (`e_m ≈ 4.3e-7` against `5.5e-3`),
        and check 5 catches one bf16 ULP of shift with a large margin.
      - At bf16, under check 5's band, 6b catches the `W_{t+1}` trained from the shifted
        bias (`ρ` about 2¹⁶–2¹⁷ per bf16 ULP of shift). A step that commits the honest update
        `W_t − η·g` instead passes.
      - Measured on `L1.b_q`, which feeds `L1.Y_q` (24 × 32, `rms(P) ≈ 1.0`; at bf16
        `e_m·rms(P) ≈ 0.0056`, so the band's edge is about 0.15–0.25), at `k = 7` and 9. At
        fp32 one bf16 ULP gives a normalized residual of 9,700–11,100 on entry 7
        (`b ≈ −2.12`, ULP `2⁻⁶`) and 2,300–2,800 on the last entry (`b ≈ −0.844`, ULP
        `2⁻⁸`), against `τ = 8`. At bf16 with the honest update committed, shifts passed up
        to 0.125–0.19 (8 or 12 ULPs on entry 7, 32 on the last) and check 5 rejected from
        0.19–0.25. A rerun of the same harness during review got other residuals, and there
        0.25 (16 ULPs on entry 7) passed on the tied model at both `k`. The fast suite
        asserts only the one-ULP shift (`test_switches_verify_an_honest_step`): check 5
        rejects it at fp32, 6b at bf16.
    - **SmolLM2 at `(fp32, bf16)` on the dev Mac CPU** (2026-10-11, `VERIF_COMPUTE_DTYPE=
      bfloat16`, `k = 9`, `η = 1e-3`; the slow tests M2, A8 and A9 pass at this setting):
      - `label` fills all 7,113 slots once, and each of the 210 linear `W_x.grad` is the fp32
        copy of its `G_x`. The step runs 1,119 `_to_copy`. Forward (456): 211 copies of a
        parameter (210 linear weights and `W_E` for the head), labeled by `param_copy`; 241
        glue values cast to bf16 at a matmul's input (per layer the normed input once for
        each of q, k, v, gate and up, RoPE'd `Q̃`, `K̃ᵀ` and the softmax; once the head's
        input); `Λ` widened to fp32 for the cross entropy; the batch's 3 integer casts.
        Backward (663): 211 copies of `G_x` and `G_E_head` widened to fp32 (the 210 linear
        ones are the `W.grad` link, `G_E_head` joins `W_E`'s glue gradient); 241 more bf16
        products widened where their consumer ran in fp32 (`dX_q`, `dX_k`, `dX_v`,
        `dX_gate`, `dX_up`, `dA`, `dQ`, `dK` per layer, `dF`); and 211 fp32 gradients cast to
        bf16. Only the parameter copies and the `G` copies matter to labeling; the rest is
        glue, which the replay recomputes through the model's own code.
      - The replay rebuilds all 2,371 forward and 4,742 backward operands and the 62 glue
        gradients bit for bit. Peak RSS 7.72 GB in A9 (6.60 GB before the replay).
      - Steps 1–3 are accepted under provisional bands (one judged step, then three steps in
        calibration mode frozen with provisional bands), and the plain run's `W_4` equals the
        verified run's bit for bit. Losses 1.369655, 2.014252, 1.562216. Check 5's largest
        normalized residual is 1.19, 1.11 and 1.16 (`dV`, `dV`, `dA`; every other class lower,
        `Λ` among them), largest `κ` 30.3 (`dX_down`); 6a and 6b `ρ_max` 2.0 every step. At
        fp32 the same provisional bands reject step 1 at `Λ` (11.5 > 8). At bf16 the term
        `√2·ε_in` of `e_m = √2·ε_in + √q·ε_acc`, with `ε_in = 2⁻⁸`, dominates every class,
        and the honest residual, mostly the product's own rounding to bf16, is of the same
        order, so no class stands out the way `Λ` does at fp32. The bands for bf16 are not
        calibrated: `band_context` holds the dtypes, so the fp32 band file is refused.
      - `prove_step` takes 45–46 s per step against 0.8 s at fp32; commit 0.21 s; the
        verifier 1.6–1.7 s (check 5 1.31 s), as at fp32. CPU bf16 matmul is slow on this Mac
        (torch 2.9.1 without MKL-DNN, Accelerate has no bf16 path): about 3.8 GFLOP/s with a
        contiguous `B`, 277 GFLOP/s with a transposed `B` (the forward linear's layout),
        against about 1,700 GFLOP/s in fp32. The head's 512×576×49152 product takes 23 s in
        bf16 and 20 ms in fp32, and the backward products dominate. The determinism flag
        changes nothing. This is a property of the platform; it says nothing about the GPU.
  - `build_model()`: an unmodified `from_pretrained` causal LM (`AutoModelForCausalLM`) through
    `setup.model.load_pretrained`, eager attention, fp32, in eval mode, checked against the
    declared weights and against `tied` (the output projection is or isn't `W_E`'s tensor).
  - `matmul_count_decoder(L, n_s, n_h) = L·(21 + 6·n_s·n_h) + 3`.
  - `encode_record` raises `RecordError` on a non-`Record` or a record longer than `n`.
  - **Pins:**
    - Weight names are the HF parameter names, in ref block §6 order: `w(ℓ, x)`,
      `gamma_attn(ℓ)`, `gamma_mlp(ℓ)`, `w_e`, `gamma_final`. A bias follows its weight
      (`W_q, b_q, W_k, b_k, W_v, b_v`), and an untied `lm_head.weight` comes last.
    - Product names are `L{ℓ}.Y_x`, `L{ℓ}.dX_x`, `L{ℓ}.G_x`, `L{ℓ}.S[s,h]` (and `O`, `dA`,
      `dV`, `dQ`, `dK`), `Lambda`, `dF` and `G_E_head` (`G_head` when untied), looked up with
      `m_of(name)`. `member` is the 0-based `(s, h)`.
    - `product_class(spec)` is the role without layer and member (`Y_q`, `S`, `dK`, `Lambda` …),
      30 classes: 21 linear roles, the six attention roles, and `Lambda`, `dF` and `G_E_head`
      (`G_head` when untied). A band file for an untied model needs a `G_head` class: one
      naming `G_E_head` fails `Bands.check_keys`, and a missing class falls back to
      `kappa_max`.
    - The batch is `setup.data.assemble_batch`, and `ρ` is HF's `attention_mask`. The loss is
      `Σ μ_i·CE_i / Σ μ_i` (`loss_from_logits`). A batch with `Σ μ = 0` raises `ValueError`
      in `loss` instead of giving a 0/0 NaN. `build_record` never builds such a record (its
      EOS target is always masked in), so only a faulted batch can reach it.
  - `label` maps records by operand storage. The forward `S` and `O` have no storage link to a
    weight, so they're taken as the two bmms between a layer's `Y_v` and `Y_o`, `S` first.
    `O`'s `A` must be row-stochastic, `O`'s `B` must equal the layer's `Y_v` output head by
    head, and `S`'s `B` must be equal across the query heads that share a kv head. The `δK̃`
    leaf is the contiguous transpose of the captured `Q̃ᵀ·δS`. Any unmatched, duplicate or
    missing record raises `LabelingError`. A biased `Y_x` must be an `addmm` adding the
    declared `b_x` once (`α = β = 1`), and its leaf is the op's output, bias included (F14);
    a bias on any other record, or a missing or wrong one, raises `LabelingError`.
  - The verifier's model is built once per `DecoderComputation` and reused by every replay
    (`_replay_model`; a build reloads the checkpoint, about 0.12 s). Each reuse gives every
    parameter fresh storage before `load_weights` fills it from `W_t`, so no operand served
    by an earlier replay (a linear's `B` is a view of its weight) aliases the new weights;
    restores every buffer to its build-time value; raises `ReplayError` on any hook left on
    a module or a changed buffer set; reruns the model checks; and takes the model away from
    the previous replay (`.model = None`). The prover never gets this model.
  - `replay(leaves) -> DecoderReplay` (A8, forward products). Its own model, from the committed
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
    or a unit input that isn't the previous unit's output raises `ReplayError`, and so does a
    forward product whose bias isn't the declared one (`supply`'s `bias` must be `b_x`'s
    tensor for a biased weight and `None` otherwise). Operands are what the op receives:
    `(X·, W_xᵀ)`, `(Q̃, K̃ᵀ)` after RoPE and `repeat_kv`, `(softmax, Ṽ)`; a biased product's
    `A` and `B` leave the bias out. They are served detached, since the pass ran with grad
    on.
  - Backward products and `glue_gradients()` (A9) come from `torch.autograd.grad` on the
    kept unit graphs, still under `ProductSubstitution`. Autograd's engine restores the
    forward's thread-local state, including the dispatch mode, on its worker threads (on any
    device), so every backward mm/bmm also gets its committed leaf. A product that escaped
    the mode would be reported missing. No backward is written by hand. One backward per
    unit, from the top:
    - Head (`j = L+1`): the kept loss, grad to `[X_{L+1}, γ_final, W_out]` (`W_out` is
      `W_E` when tied). `δΛ` comes from autograd through `loss_from_logits`; `loss` and
      `label` are never called.
    - Layer `j`: the kept output, grad to `[X_j, γ_attn, γ_mlp, b_q, b_k, b_v, W_q..W_down]`
      (the biases when declared) with `grad_outputs = δX_{j+1}` (the committed-chain value
      the previous unit gave). A backward product carrying a bias raises `ReplayError`.
    - Identification at call time: an mm whose `b` is a weight is `dX_x`/`dF`, and its `a`
      must be that weight's `δY` (storage recorded by a hook on the substituted forward
      output). Any other mm is `G_x`/`g_head`: `a` is some weight's `δY` and `b` the saved
      input of that `Y`. A bmm is named by which saved `S`/`O` operand it reads and in which
      slot. `dK` is served as the transpose of the stacked leaves, as in `label`.
    - Each operand's `_version` is recorded when it is supplied and checked when served; a
      mutated operand raises `ReplayError`. An undeclared, duplicate or missing backward
      product raises `ReplayError`.
    - Memory, as in training: after the forward pass every unit's graph and operands are
      held, about one training step's activations. Each unit's backward frees its graph
      (`retain_graph=False`) and the unit is dropped, so memory falls unit by unit; the
      served backward operands are dropped after `g_head` or `L{j}.G_v`. The frontier
      `(j, δX_j)` and the γ gradients persist. The head unit is the largest (`δΛ` and the
      softmax intermediates are `N×n_v`). A request out of order (a forward product whose
      unit is gone, or a backward unit already run) rebuilds the whole pass from scratch.
    - `glue_gradients()` is valid only right after `operands(M)` with the frontier at layer
      1. It returns the γ and bias gradients and, for `W_E`, `G_E_head` plus `G_E^emb` when
      tied and `G_E^emb` alone when untied, where `G_E^emb` =
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
  - **The switches in the fast suite (FS1).** `test_decoder.py` builds a tiny random model
    for each of the four combinations (Llama or Qwen2, tied or untied; `decoder_helpers.SWITCHES`)
    with random biases. `label` fills every slot exactly once, and the replay rebuilds every
    operand and glue gradient bit for bit. An honest step is accepted under every combination,
    with HF's zero biases and with random non-zero ones (FS2), 6b on the bias gradients
    included. One bias entry of layer 1's `b_q` or the last layer's `b_v` altered is rejected
    (`test_switches_verify_an_honest_step`): in `W_t` at check 0 (check 7 after step 1), in
    training alone (the committed `W_t` honest) at check 5 on that layer's `Y_x`, and in
    `W_{t+1}` at 6b. `test_switches_capture_is_bit_identical` checks that a captured step on
    drawn biases gives the uncaptured step's loss and `W_{t+1}` bit for bit. A bias update
    `η·g` of about 1e-7 sits near the bias's rounding step, so any bias with a small gradient
    (`b_q` included) can come out of a step unchanged. The test seeds the init (seed 0), under
    which every `b_q`, `b_k` and `b_v` changes, and asserts that each does.
  - **Both compute dtypes in the fast suite (FS3).** The `switched` fixture runs each of the
    four combinations at `(fp32, fp32)` and `(fp32, bf16)`, so labeling, the bias's copy,
    the `W.grad` link, the bit-for-bit replay and the honest step's acceptance are tested
    under autocast too. `test_switches_verify_an_honest_step` and
    `test_switches_capture_is_bit_identical` run at both dtypes as well, drawn biases
    included. At bf16 on the tiny models (provisional bands, drawn biases, `k = 7`,
    `SEED = 1`) check 5's largest normalized residual is 0.7–1.1, and check 6's `ρ_max`
    1.5–2.0, as at fp32. `test_label_rejects_a_grad_not_linked_to_its_G` covers the `W.grad`
    link at both dtypes on Llama and Qwen2: `W_k.grad` replaced by `W_v.grad` or by a clone
    raises `LabelingError`. (The tamper `swap_G_k_G_v` finds the grad by data pointer, which
    at bf16 is AccumulateGrad's copy, so it runs at fp32 only.)
    `test_plain_baseline` adds a bf16 tiny Llama whose plain and verified final weights are
    equal, and `tests/verification/runs/test_calibrate.py` runs `gradient_coherence` at both
    dtypes.
  - **The full-scale models (FS4).** `tests/verification/full_scale_helpers.py` holds `PINS`,
    the four models at their commits, in the order FS4 ran them: Qwen2.5-0.5B, Falcon3-1B,
    Qwen2.5-1.5B and Meta's Llama-3.2-1B-Instruct (`meta-llama/…` at `92131767`, which replaced
    the unsloth mirror there; the mirror's tokenizer still filters the shared list, and both
    give the same ids on every row of the four training files). `full_scale_case(repo, revision)` builds
    each one's run config, `D` and `h_D` without fetching, or skips when the weights aren't
    cached. `computation/test_full_scale_models.py`'s slow `test_real_step_labels_and_replays`
    checks, per model, that every slot is filled once (M2), and that every forward operand
    (A8), backward operand and glue gradient (A9) is rebuilt bit for bit. It compares BLAKE3
    digests rather than holding both copies, since a 1B step's products run to about 20 GB. The
    fast `test_digest_comparison_on_a_tiny_model` checks that comparison on tiny Llama and
    Qwen2 models: an altered `W_t` leaf changes only the two operands that read it, and an
    altered layer-1 `δX_q` leaf changes the glue gradients it flows into (`W_E`'s and layer
    1's `γ_attn`) and not `γ_final`'s.
- `leaf_sizes.py` (FS10): `leaf_sizes(config, *, n_s, n, compute_dtype=torch.float32)`
  returns the step's `LeafSize(index, section, name, role, shape, dtype, nbytes)` list in
  commit order, without running a model. `compute_dtype` goes to `DecoderComputation`, so
  weights are fp32 and products `compute_dtype`, as a real step commits them; a slot name that
  disagrees with the leaf order raises `RuntimeError`. The test checks it leaf for leaf against
  a real `prove_step` at fp32 and bf16, with the q/k/v biases drawn non-zero. `benchmarks/`
  imports only this module from `verification` (`tests/test_layering.py`, rule 8).

### `verification/prover/`

- `capture.py` (A2, merged). Its module docstring has the full hand-off notes for A7; read it.
  - `MatmulCapture(param_names)` is a `TorchDispatchMode` passthrough, so a captured run is
    bit-identical to an uncaptured one. Its interface:
    - `.phase("forward"|"backward")`;
    - `.records` (`MatmulRecord`, in call order, not canonical order), `.glue_outer` (`q = 1`
      products) and `.n_products`, where each `(s,h)` bmm member counts once;
    - `.summary()`, `.release_operands()` and `.assert_unmodified()`.
  - `param_storage_map(model)` comes from `computation/matmul_ops.py`, as do the op lists.
    `OperandInfo` has `param_name`, `storage_ptr`, `producer_index` and `param_copy`. A storage
    pointer is valid only while the capture holds its tensors, so link records by
    `producer_index`.
  - **Copies (FS3).** Inside a phase every `aten._to_copy` becomes a `CopyRecord(index, phase,
    src_info, out_info, storage)` in `.copies`; `.param` names the parameter it copies,
    directly or through an earlier copy. A copy of a parameter is held until
    `release_operands`, and an operand aliasing it carries `OperandInfo.param_copy`. Any
    other copy is held only through a `StorageWeakRef` to its storage: a strong reference
    would keep activations alive and turn AccumulateGrad's adoption of a gradient copy into
    a clone. `param_of(t)` names the parameter `t` aliases, directly or as a copy, until
    `release_operands`. `is_copy_of(t, index)` says whether `t` shares the storage of a
    `_to_copy` of record `index`'s whole output, with the same view; storages compare by
    identity, so a reused address never matches. `summary()` adds `n_copies` and
    `n_param_copies`. At `(fp32, fp32)` a SmolLM2 step makes 3 copies (the batch's int64
    mask and ids cast to bool and fp32, all glue) and none of a parameter.
  - Errors: `CaptureError`, `UnsupportedMatmulError` (any `REJECTED_OPS` op, a non-default
    overload, or an op outside `aten`/`prims` inside a phase), `BiasedMatmulError` (`α ≠ 1`,
    or a non-zero bias outside F14's form, a bias vector added once with `β = 1`; every
    accepted `addmm` keeps its `bias` on the record, and the labeler decides whether it was
    declared), `PhaseError` and `MutatedCaptureError`.
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
  `LeafDtypeError`, `StoreMutationError` and `LeafReadError` (a disk leaf or the store's
  metadata can't be read back).
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
  - **On disk (A14, C3; S3's layout).** `DiskStore.write(dir, c, leaves, tree,
    dataset_paths=None)` replaces `dir` with one `torch.save` file per leaf
    (`leaf_00000.pt` …, `LEAF_FILE`), `leaf_hashes.bin` (the tree's 32-byte leaf hashes, for
    `path`) and `meta.json` (`format`, the root and the `h_D` paths as hex), written last; a
    failed write deletes `dir`. `DiskStore(dir)` opens one. Not part of any commitment.
    - A tensor that doesn't own its whole storage (a `bmm` member, a slice of `D`) is cloned
      before saving, because `torch.save` writes the whole storage. A `Record` is saved as a
      dict of its three tensors and rebuilt (and so revalidated) on load.
    - `leaf(i)` loads the file on every call (`weights_only=True`, CPU, no `mmap`, no cache),
      so each read is a fresh tensor sharing no memory. A missing file is `IndexError`; an
      unloadable file or a foreign object is `LeafReadError`. `root` and `dataset_path`
      read `meta.json` each call. Check 2's `CommittedLeaves` keeps the objects it hashed, so
      later checks never reread the disk; a file that changes between check 4 or 7 and
      check 2 is a check-2 rejection ("two versions").
  - **The hand-off.** `StoreHandoff.hold(c, t, leaves, tree, dataset_paths) -> store` and
    `.release(store)`. `IN_MEMORY` (`InMemoryHandoff`) is `InMemoryStore.hold`, release a
    no-op. `DiskHandoff(dir, keep=False)` writes step `t` to `dir/step_<t>/` and deletes it
    on release.
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
    chain_check_id=, k=, judge=)`. `ctx.state` (`StepState`) carries what check 2's read
    produced to later checks: `claimed_root` (the prover's `h`, validated as a digest; check 5
    keys on it), the leaf hashes and `CommittedLeaves` (`ctx.state.committed()`). It carries
    check 5's replay to 6b. The root comparison writes `recomputed_root`. `ctx.stats` (`StepStats` of `ProductStat`/`TensorStat`) records every
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
    `CHECKS` under its slot id. `DEFAULT_ORDER = ("4","7","2","6a","5","6b","2.root")`.
    Check 2 has two slots (F15a, task C7): `check_2_commitment` (`"2"`, the read) and
    `check_2_root` (`ROOT_SLOT = "2.root"`, the root comparison). Both reject as `"2"`;
    `protocol_id(slot)` maps a slot to the id its rejections carry. `check_2_root` reads
    nothing from the store, so a streaming store can reuse it.
  - **Errors.** Prover-data errors are mapped to a rejection only around store reads, leaf
    hashing and validation. After check 2's read, replay, operands and glue run on validated
    leaves, so their errors propagate as verifier bugs; only `TranscriptFormatError` is mapped
    there. A violated verifier-side precondition raises `RuntimeError`.
  - **Byte binding.** Check 2's read takes every leaf once and keeps the objects in a
    `CommittedLeaves` reader, guarded by `_version`; checks 6a, 5 and 6b read only that.
    Checks 4 and 7 record the hashes they saw in `ctx.state.early_hashes`, and check 2 rejects
    if its own read hashes differently. Check 2 reads and validates leaves in order and
    hashes them in parallel, so a leaf hashes after later reads. It therefore checks every
    cached leaf's `_version` both before and after reading the claimed root (a write during
    check 2 is a malformed rejection at 2). All of these rejections happen before 6a; only
    the root comparison moved to the end. Checks 4 and 7 stay leaf by leaf. This keeps every
    leaf in memory for the step (see F1–F3 at full scale).
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
  - **Check 6a under mixed precision (FS3, F6b).** The committed `G` is a bf16 product leaf;
    6a widens it exactly to fp32 before the update identity, as AccumulateGrad does for
    training's `W.grad`. At fp32 the cast is a no-op.
  - Check 5 gets its numbers from `matmul_check.freivalds.measure_product` and draws challenges
    from the claimed root `store.root`, as check 2's read validated it
    (`ctx.state.claimed_root`; F15a, task C7). The root comparison at the end of the step
    shows that `h` is the root of the leaves.
  - **Biased products (F14, FS2).** A served product is `(A, B, P, b)`. For a spec with a
    `bias`, `b` is the committed `W_t` leaf of that name, rounded to `operand_dtype` as `B`
    is, never a tensor the replay or the prover supplies; `b` is `None` otherwise.
  - **Member batching.** A run of consecutive member specs of one layer (`_member_runs`;
    on SmolLM2, `S`+`O` and `dA`+`dV`+`dQ`+`dK` of each layer) has its operands served one
    member at a time in canonical order, then stacked by shape and measured with
    `measure_products`. Members are then recorded and judged in canonical order, so the first
    failing member rejects with the same message as alone, and an error serving a later member
    is raised only after the members before it pass. Every other product is measured alone.
  - **The record pass (EV2, EVALUATION_SPEC §4.1).** `record_rest(store, c, ctx, bands,
    slot) -> str` measures, for the record only, the tests a rejecting check skipped. A
    check-5 rejection inside a member run first records the run's later members as check 5
    measured them in the batch (`StepState.run_rest`, set by check 5 when it rejects), since
    a member measured alone or in a smaller batch is held to agree only to `1e-4` relative;
    a serving error check 5 deferred (`StepState.run_rest_error`) is raised after them.
    Every later product is then measured from check 5's own replay. A 6a rejection measures
    the remaining linear weights, then every 6b glue weight from a replay of its own, since
    check 6 is one protocol check over every tensor. A 6b rejection measures the remaining
    glue weights. Each gradient comes from the helper its live check uses
    (`_linear_gradient`, which widens a bf16 leaf to the weights' format as F6b does;
    `_replay_glue_gradients`; `_glue_gradient`), so an untouched tensor scores what the live
    check gives it, at fp32 and bf16. It runs with `judge=False` and no metrics seam,
    sharing `ctx.stats`, so it changes no rejection, verdict or timing row. A failing test
    is still recorded and the pass goes on. It returns `RECORD_COMPLETE` (`"complete"`), or
    `RECORD_NOT_RUN` (`"not run"`) when the slot has no scored tests or none remained; a
    malformed leaf ends it with `RECORD_MALFORMED` (`"ended: malformed"`), the tests before
    the leaf recorded. A verifier-side error propagates, as in the checks, and the driver
    turns it into `RECORD_ERROR` (`"error"`). `RECORDED_SLOTS = ("6a", "5", "6b")`. The
    checks after the rejecting one never run.
- `driver.py` (A5): `Verifier(c, *, h_D, n_records, k, n_steps, bands, seed, w0= | w0_hashes=,
  schedule=, calibrate=False, allow_provisional=False, section=None, kappa_guard=True)`.
  `seed` is required and has no default: `π` is public input, `setup.data.schedule(t, n_s,
  n_records, seed)` unless `schedule` replaces it, and the prover's default `π` comes from
  the same call with `verifier.seed`.
  `kappa_guard=False` (see `StepContext`) with `calibrate=True` raises `ValueError`, since
  calibration fits `κ_max`. `section` is B6's
  metrics seam: it wraps each check under its slot id (step 1's chaining comparison is `"7"`;
  check 2's read is `"2"` and its root comparison `"2.root"`; `run:0` is check 0's anchor
  hashing), and `StepContext.section`/`timed(name)` pass it into check 5 (`5.glue`,
  `5.measure`) and 6b (`6b.glue`). `context.no_section` is the no-op.
  - `n_steps` (T) is required. Provisional bands are refused unless `allow_provisional=True`
    (P10a); `bands` may be `None` only when calibrating.
  - `start_run(D)` runs check 1 (including that every `π(t)`, `t ≤ T`, holds `n_s` distinct
    indices into `D`; F8c's `π` runs on into later passes, so `T` may exceed one pass) and
    prepares check 0. At step 1, check 0 runs in check 7's slot and reports as `"0"`.
  - `verify_step(t, store)` runs the default order and keeps the `W_{t+1}` hashes only once
    every slot, the root comparison included, has passed. After a `"failed"` rejection (never
    a malformed one) it calls `checks.record_rest` with the rejecting slot, once the verdict
    is fixed, through `_record_rest`, which catches any exception the pass raises with a
    `RuntimeWarning` and records `"error"`, so the rejection stands. `record_pass[t]` holds
    the status (`"not run"` after a malformed rejection). Its time lands in the loop's
    `verify_s`, not in any check's row.
  - `freeze(bands)` ends calibration (P10b): it re-judges every calibrated step's stats in
    check order, records `bands.source`, and judges every later step. The root comparison
    runs live in calibration. Since it runs after 6a, 5 and 6b, a live rejection from it is
    re-judged first: a band failure in that step's numbers replaces it, as in a judged run.
  - `end_run(final) -> RunVerdict(accepted, rejection, steps_verified, band_source)` runs
    checks 8 and 9. It raises while calibration is unfrozen, unless `start_run` rejected the
    run (check 1): no step ran, so no band can change that verdict, and it ends as it is.
  - `timings[t][id]`, `run_timings` and `stats[t]` hold the per-check numbers, and
    `record_pass[t]` the record pass's status at a rejected step.
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
    `load_bands(path, *, k, context)` (the read-only loader every judged run uses: a missing
    file raises `FileNotFoundError`; a file fitted at another `k`, or whose recorded context
    differs from `context` in any key but `PROVENANCE_KEYS`, a key missing on either side
    included, raises `BandFileMismatch`, a `ValueError` that names the keys and says to rerun
    `verification.runs.calibrate`) and `assert_same_band_source(sources)` (P10a's harness
    assertion; rejects `None`, `"provisional"` and mixed hashes).
  - `band_context(cfg, c, *, h_D, k)`: the run a band file is fitted for, which `calibrate`
    writes and every loader compares: model and revision, `(MASTER_DTYPE, COMPUTE_DTYPE)`,
    `attn_impl`, device, `h_D`, `n_s`, `n`, `η`, `M`, `T_run` (`VERIF_STEPS`) and `k`. It also
    holds `calibration_seed` (`VERIF_SEED`) and `threads` (`VERIF_THREADS`), both in
    `PROVENANCE_KEYS`: kept in the file, never compared. The bands describe rounding noise on
    honest steps, which doesn't depend on `π`, so a run at seed 2–5 loads the file fitted at
    seed 1 (EVALUATION_SPEC §3.2). The thread count changes only the CPU's floating-point
    reduction order, a shift far below `τ = 8·s_h`, while the cost results need it recorded
    (FS2). A file written before `threads` was recorded (`cc7a9bca…`) loads too.
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
- `freivalds.py`: `measure_product(a, b, p, *, h, m, k, eps_in, eps_acc, bias=None) ->
  ProductMeasure` (ν, `‖|P|·1‖`, κ, the residuals, `‖P‖_F`, the band unit and the normalized
  residuals). It judges nothing. With `bias` (F14, FS2) it measures `[A | 1]·[B ; bᵀ]`
  without building it: `d_j = A(B·r_j) + (bᵀ·r_j)·1 − P·r_j`, `ν = ‖|A|(|B|·1) + ‖b‖₁·1‖`
  and `e_m(q + 1)`. Without it every operation and its order is the unbiased one, so
  unbiased products keep their bits. Its norms use `_safe_norm` (power-of-two scaling,
  bit-identical to `vector_norm` when that doesn't overflow or underflow).
  `measure_products(a, b, p, *, h, ms, k, eps_in, eps_acc)` does the same for a stacked batch
  of one shape, each member with its own challenges and `_safe_norm` scale. At SmolLM2's
  member shapes its normalized residuals equal `measure_product`'s bit for bit; below about
  8×8, CPU `bmm` rounds differently from `mm`.
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

The first measured figures below date from before C5 and C6: Stanford Alpaca data, the
sequential schedule and seed 0. The re-runs of 2026-10-11, on today's `D` and `π(t)` at seed
1, follow each of them.

- `loop.py` (A6): the S3 per-step loop, instance-agnostic.
  - `run_loop(c, model, D, w0, verifier, *, final, fault=None, schedule=None, on_step=None,
    section=None, handoff=None) -> LoopResult`. `section` is B6's metrics seam, passed to
    `prove_step`. The default `π` is `data.schedule` at `verifier.seed`, the verifier's own
    call. It runs `verifier.start_run(D)`, then per step `prove_step` on `π(t)`,
    `InMemoryStore.commit_step` (`P3.commit`) and `handoff.hold` with the `h_D` paths
    (`P5.write`; default `IN_MEMORY`, or a `DiskHandoff`), `verifier.verify_step(t, store)`,
    `handoff.release(store)` (also on error), and drops the step. `StepRecord.commit_s`
    includes `P5.write`, so a disk write lands there. It stops at the first rejection and ends with `verifier.end_run(final)`, where
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
  outcome), `ScenarioResult`, `HONEST` (no fault, accept), `honest_final(c, D, w0, T, *, seed)`
  (`T` uncaptured `plain_step`s from `W_0` through `run_plain`, check 8's reference),
  `run_scenario(..., T, k, seed, final, recorder=None, h_D=None, verifier=None)` (`h_D`
  defaults to `D`'s root; `verifier(section) -> Verifier` replaces the default provisional
  verifier and must match the run's seam, `T`, `k` and `seed`; `memory_run` and `count_run`
  take it and `seed` too, and `handoff`, passed to `run_loop`), `judge` (the
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
  `count_smoke` (`PASS_STEPS` and `COUNT_STEPS` honest steps through `scenarios.memory_run` /
  `count_run`).
  - The MLP at widths `(16,32,32,8)`, `n_s = 4`, `η = VERIF_ETA`, `k = VERIF_K`, on
    `synthetic_dataset` of `VERIF_N_RECORDS` records. `T` is `--steps`, else `VERIF_STEPS`
    (default 10), and must be ≥ 2. Bands are provisional, with `allow_provisional=True`.
  - Scenarios and their declared outcomes: `honest` (accept, with check 8 against
    `honest_final`, `T` uncaptured `plain_step`s from `W_0`), `flip` (the largest entry of
    `Y_2` sign-flipped at step 2 → `(2, "5")`), `bad-w-next-ulps` (one `W_{t+1}` entry moved
    300 ulps at step 2 → `(2, "6a")`), `bad-w-next-batch` (A3 on another batch → `(2, "6a")`)
    and `broken-chain` (one hidden `plain_step` between steps 1 and 2, P11, on another batch
    since the MLP has no `b̃` → `(2, "7")`). The other batch is `data.other_batch(2, …)`, the
    neighbouring batch of step 2's pass, disjoint from `π(2)` at every seed.
    `scenarios(c, dataset, seed)` takes `π`'s seed for it.
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
    comparison is `7`; the `step` record keys it `0`, the protocol id a rejection carries;
    check 2 has rows `2` (the read) and `2.root` (the root comparison, last), and a check-2
    rejection marks the last of the two that ran as `fail` in the `step` record), and `0` (anchor hashing), `1`, `8`, `9` once per run. `9` is building
    the verdict, kept as a row although negligible. Check 5 splits into `5.glue` and
    `5.measure`; check 3 has no row, its cost is `5.glue`.
  - **FLOPs** use torch's `flop_registry` formulas through `_FlopTally`, not
    `FlopCounterMode` as EQ1c names: `FlopCounterMode`'s module tracker adds autograd hooks
    that break replay's `torch.autograd.grad` on leaf tensors. A test checks the two agree on
    a plain step. Hash counts cover `commitment.merkle`, `matmul_check.challenges` and
    `verifier.bands`; the band file is hashed at load, outside every section.
  - `MetricsWriter(dir, run, *, device, model, corpus, attack, seed, config, band_file_hash,
    h_D, extra)` (`.recorder(scenario, *, bpr_records, bpr_actual, cheat_step)`,
    `.write(rows)`, `.write_residuals(scenario, t, stats, c, *, bands, k, kappa_guard)`,
    `.close()`), `read_records(path, record=None)`, `read_residuals(dir)` (gives `StepStats`
    equal to `verifier.stats`), `memory_probe(device)`, `step_record`, `config_hash`.
    `model`, `corpus`, `attack` (`VERIF_ATTACK`, `None` for a run on other data) and `seed`
    go into the `environment` record and every `step` record. Every run that writes `corpus`
    writes `attack`. A run or directory named `data` or `evaluation` (`RESERVED_NAMES`,
    EVALUATION_SPEC §4.6) raises `ValueError`, in any letter case (`str.casefold`): on a
    case-insensitive file system `Data` is `data`.
  - **Run names (EV2).** `run_name(scenario, model, attack, bpr_records, seed)` is a cell's
    unique name, `<scenario>-<model>-<attack>-bpr<n>-s<seed>`: the model as its repository's
    last part, `n` the poisoned records in the cheat batch (0 for an honest run), characters
    outside `[A-Za-z0-9._-]` as `_`. For example
    `honest-SmolLM2-135M-Instruct-refusal-bpr0-s1`. `run_verified`, `poisoned_step` and
    `hidden_steps` use it. The other runs keep their fixed names (`plain_baseline`,
    `llama_step`, `calibrate`, `mlp_smoke`, `store_crosscheck`, `flipped_matmul_sweep`).
  - **Step records (EVALUATION_SPEC §4.1).** `step_record` writes `model`, `corpus`,
    `attack`, `seed`, `bpr_records`, `bpr_actual`, `cheat_step`, `step`, `judged`,
    `verdict`, `first_failing_check`, `rejection_kind`, `checks`, the step's scores
    `rho_5`, `rho_kappa`, `rho_6` and `n_failing`, and `record_pass`
    (`record_pass(verifier, t, stats, *, kappa_guard)`): at a rejected step `{"status": the
    driver's record_pass[t] ("not run" when it has none), "measured": {"rho_5": products ×
    k, "rho_kappa": products (0 with the guard off), "rho_6": tensors}}`, the counts over
    every test measured at the step, live and in the pass; `null` at an accepted step.
    `bpr_records` and `bpr_actual` are the
    cheat batch's poisoned records and their share of it: `env.bpr_records`/`bpr_actual`
    for a cheat on `b̃`, `0`/`0.0` for an honest run, `None` for a protocol fault and the
    MLP smoke run. `cheat_step` is `t*` for a cheat, else `None`.
    - `judged_step(verifier, t, cheat_step)`: never while calibrating or under provisional
      bands; a cheat run judges `t ≥ t*`; an honest run at the band file's
      `calibration_seed` (H1) skips the file's `calibration_steps` (1–3); at any other seed
      (H2) it skips step 1. A file without `calibration_seed` raises a worded `ValueError`
      (`calibration_seed(bands)`), unless the run is a cheat run, which judges from `t*`.
      `in_sample_step(bands, t, seed)` is H1's half of it.
    - `rho_arrays(stats, bands, *, k, kappa_guard=True)` gives the component scores, each a
      test's left side over its right side, so a test rejects exactly when its score is
      above 1: `rho_5` `[P, k]` (`normalized/τ`), `rho_kappa` `[P]`
      (`ν/(κ_max(class)·‖|P|·1‖)`, 0 when `ν = 0`, `inf` when only the denominator is 0) and
      `rho_6` `[W]` (`ρ_max/τ_W(tensor)`). All NaN without bands (calibrating);
      `rho_kappa` NaN with the κ guard off. For `ρ_κ` and `ρ_6` the score is above 1
      exactly when the live test rejects. Check 5's residual test compares the raw residual
      with `τ·unit`, so `ρ_5` can disagree with it by rounding when a residual lands on `τ`
      within an ulp.
    - `step_scores(stats, bands, *, k, kappa_guard)`: per family (`SCORE_FAMILIES`) the
      largest score, and `n_failing`, a dict of the count of scores above 1 (NaN counts as
      failing, and as `+inf` in the maximum, `nan_as_inf`; `cheats.scores` does the same).
      Only measured tests count: a test the step never measured has no row and isn't
      failing. A family not scored at the step (no bands, guard off, nothing measured) is
      `None` with its `n_failing` entry.
  - Files under `$VERIF_OUTPUT_DIR/<run>/`: `records.jsonl` (records `environment`, `step`,
    `verdict`, `time`, `memory`, `count`, `storage`, `run_end`, each with `record` and `run`;
    non-finite floats as the strings `"NaN"`, `"Infinity"`, `"-Infinity"`) and
    `residuals/<scenario>/step_<t>.npz`. Since A11 each npz also holds the product scales
    `p_q`, `p_norm`, `p_nu`, `p_abs1`; older archives read back with them unknown. Since EV2
    (EVALUATION_SPEC §4.2) it holds the index table and the scores beside the raw arrays:
    per product `p_m`, `p_name`, `p_cls` (the fine class, κ's), `p_spec_class` (§5.7's
    class, `spec_class(c, spec)`: `forward`, `input gradient`, `weight gradient`,
    `attention score`, `attention value`, and the placeholder `operand gradient` for δA,
    δV, δQ̃ and δK̃, which §5.7 doesn't name), `p_layer` (0-based since the EV2 review:
    `0 … L−1` for the layer stack, `L` for the output layer's `Lambda`, `dF` and `G_E_head`;
    `layer_count(c)`, `layer_index(spec, L)`); per tensor `w_check`, `w_name`; then `rho_5`
    `[P, k]`, `rho_kappa` `[P]`,
    `rho_6` `[W]` row for row (`rho_arrays`), and `band_source`, the bands' hash (empty while
    calibrating). A step rejected at check 5 holds all `M` products, one rejected at 6a or 6b
    every tensor check 6 measures (the record pass); one rejected before checks 5 and 6
    (checks 4, 7, 2) has an archive with zero rows.
    On macOS, export `MallocLargeCache=0` for runs whose
    memory figures are compared: otherwise freed large blocks stay in the footprint.
  - `metrics_overhead.py`: `.venv/bin/python -m verification.runs.metrics_overhead
    [--reps N] [--steps T] [--widths ...]`, an off/on/off timing of the honest MLP run.
  - `PASS_STEPS = 2`: every run's memory pass. Step 1 has check 0's anchor and first-touch
    allocation; step 2 is the first check-7 step and steady-state memory.
  - `COUNT_STEPS = 1`: every run's counting pass, one step per configuration
    (EVALUATION_SPEC §3.4). Counts are deterministic: steps 1 and 2 of the SmolLM2 run count
    the same FLOPs, hash calls and challenge bytes, and differ only in the `hash_in_bytes` of
    P3 and checks 2 and 4, which hash the batch's token records (1,356 bytes more at step 2,
    out of 2.6 GB). `llama_step` still counts over its `T`.
  - `mlp_smoke` writes to `mlp_smoke/` unless `--no-metrics` or `VERIF_METRICS=0`; its memory
    and counting passes are `PASS_STEPS` and `COUNT_STEPS` honest steps, each of its own. The
    MLP records `h_D` (its synthetic `D`'s root) and a null band-file hash.
- `llama_step.py` (A10, milestone M3): `.venv/bin/python -m verification.runs.llama_step
  [--steps T] [--bands PATH] [--metrics | --no-metrics]`, with `T` default 1.
  `DecoderComputation.from_config`, `W_0` from `build_model()`, `D` from
  `$VERIF_OUTPUT_DIR/data/D.bin` and the published `h_D` from `meta.json`
  (`load_committed_dataset`). The honest scenario goes through `scenarios.run_scenario` with
  the published `h_D` and provisional bands (`τ = 8`, below the honest `Λ` residuals C1
  measured, so an honest step can fail check 5), or with `--bands PATH` a band file read
  through `load_bands` and `band_context` and judged by `calibrate.judging_verifier`. Check 8
  is against `scenarios.honest_final`; its memory and counting passes are `scenarios.memory_run` /
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
- `bring_up.py` (FS4): `VERIF_MODEL=<repo> VERIF_MODEL_REVISION=<commit> .venv/bin/python -m
  verification.runs.bring_up [--steps T] [--reference-bands PATH] [--tamper | --no-tamper]
  [--disk-transcripts DIR]`, `T` default 1. One full-scale model on CPU at test scale's other
  settings (dtypes, `n_s × n`, η, `k`, seed); nothing in the code names a model, so Meta's Llama
  repo runs through the two env vars alone. `D` and `h_D` come from `$VERIF_OUTPUT_DIR/data/`
  (`materialize_data` for the same model). `main` stops with an error, before loading anything,
  when `model_pin` finds no cached weights or `D.bin` is missing; it never downloads. Exits 1
  unless `BringUp.passed`: the honest run accepted, the fit's concentration guard held, every
  tamper rejected as expected.
  - `bring_up(c, D, h_D, *, T, k, seed, build_model=None, tamper=True, device="cpu",
    handoff=None) -> BringUp(result, memory_rows, cal, bands, freeze_rejection, tampers)`. It
    takes any `DecoderComputation`, so a random-weight config runs the same path. `W_0` and
    every prover run come from one `ReusedModel`. Check 8's reference is `honest_final`'s `W_T`,
    kept as leaf hashes (`leaf_hashes_of`, `end_run` takes either). The honest run is
    `run_scenario` inside `memory_pass`, with a calibration-mode verifier (P10b): the exact
    checks judge live, 5 and 6 record. `on_step` fits on the last step (`calibration.fit`,
    `band_file_bytes`, `Bands.from_json`) and freezes on that fit, so `end_run` can close the
    run with check 8. The fit is in-sample and never written: it judges the freeze and the
    tamper, nothing else. `BringUp.guard_ok` is the fit's concentration guard
    (`Calibration.guard_ok`: global max at most `τ/2`), `None` without a fit; `passed` requires
    it. A live rejection freezes on `_UNFITTED`, bands at the largest finite float, so the
    reported rejection is the live one: `freeze` re-judges a root-comparison rejection's step
    first, and ordinary bands (the placeholder `τ = κ_max = 1` before) would turn it into a
    check-5 band failure. Only a NaN or infinite number, which fails any band, can still replace
    it. `freeze_rejection` holds the freeze's result, `None` when none ran. A check-1 rejection
    at run start ends the run unfrozen (`Verifier.end_run`).
  - `handoff` (`--disk-transcripts DIR`, a `DiskHandoff(DIR)`) hands each step to the verifier
    as C3's store cross-check does, the honest steps and both tampers alike; `None` is
    `IN_MEMORY`. The protocol code is the same either way, and the disk write lands in the
    commit time. `test_disk_handoff_gives_the_same_decisions` checks that a tiny Qwen2 run
    decides number for number as in memory and leaves `DIR` empty.
  - `bias_tamper(c, D, w0, bands, *, k, seed, h_D, build_model, layer=1, x="q",
    handoff=None)`: two one-step judged runs (FS2's switch test on the real model). The last
    entry of `b_q` in layer 1 is raised by 1 in training with the `W_t` leaf restored
    (expected: check 5 naming `P_m (L1.Y_q)`), and in an honest step's `W_{t+1}` leaf
    (expected: 6b naming the bias). `Tamper.rejected_as_expected` checks the step, the check
    and that name. It also runs on a fit that fails the concentration guard, and `report`
    labels each result "bands fail the concentration guard".
  - `reference_bands(ref, c)` carries a band file to another model for scoring only: `τ`, the
    default `κ_max` and the per-class `κ_max` of the classes `c` has; every weight takes the
    default `τ_W`, since `ref`'s per-tensor values name its own tensors. It also returns `c`'s
    classes that `ref` doesn't name (`G_head` for an untied model against SmolLM2's file).
    `family_maxima(stats, bands, *, k)` gives each score family's largest score, its count
    above 1 and where it sits (for `ρ_5` the product and its challenge `j`). `report` prints
    the honest steps, the fit with its concentration guard (`holds` or `FAILS`, and a failure
    again in the verdict line), `residuals.py`'s two tables, the reference scores, the tampers
    and `llama_step.report_costs`.
  - Real runs (2026-10-12, dev Mac under other jobs, `MallocLargeCache=0`,
    `VERIF_N_RECORDS=369`, `T = 1`, `k = 9`; GB here is 2^30 bytes, as `report` prints it):
    - Qwen2.5-0.5B, in memory, weights SHA-256 `fdf756fa…`: the honest step (loss 2.1318) is
      accepted with check 8 on its own fit, `s_h` 6.19 (`Λ`), `τ` 49.5, but the fit **fails
      the concentration guard**: the global max, 25.9 at `L9.dQ[3,5]`, is above `τ/2` =
      24.75. So it isn't a clean pass, and `bring_up` now exits 1 on it (the run predates the
      guard's check). The altered `b_q` entry is rejected at check 5 on `P_1 (L1.Y_q)`, residual
      2.67e3, and at 6b (`ρ_max` 1.7e7). Against SmolLM2's band file (`cc7a9bca…`, `τ` 38.0)
      `ρ_5` is 0.68 and `ρ_6` 0.50, but `ρ_κ` 2.93 (104 products above 1, largest at
      `L1.dQ[1,6]`; `dQ`'s `κ` reaches 44). Prover 2.50 s, verifier 3.83 s (check 5 2.74);
      prover peak 9.3 GB, verifier 13.0 GB, process 14.1 GB; 22 s for the whole command. The
      slow bring-up test gives the same numbers.
    - The slow replay test on the real weights fills every slot once and rebuilds every
      operand and glue gradient bit for bit on all four models. Peak, then capture with
      labeling and digests, forward replay and backward replay: Qwen2.5-0.5B 10.3 GB, 6.9 /
      2.1 / 3.6 s; Falcon3-1B 24.2 GB, 9.6 / 2.7 / 3.4 s; Qwen2.5-1.5B 25.2 GB, 12.9 / 3.9 /
      5.5 s; Meta's Llama-3.2-1B 20.3 GB, 11.1 / 3.3 / 4.9 s. The 1B peaks fall within 0.2 GB
      of the L2/L4 shake-outs' linear projection.
    - Meta's Llama-3.2-1B, in memory, weights SHA-256 `1ff795ff…`: the honest step (loss 1.5121)
      is accepted with check 8 on its own fit, `s_h` 3.02 (`Λ`), `τ` 24.2, global max 5.30 at
      `L2.G_down` (the guard holds: 5.30 ≤ 12.1). It has no q, k, v biases, so no tamper.
      Against SmolLM2's band file `ρ_5` 0.14 and `ρ_6` 0.50, but `ρ_κ` 1.15 (33 above 1, largest
      at `L4.dX_up`). Prover 4.88 s, verifier 9.79 s (check 5 7.69); prover peak 21.8 GB,
      verifier 29.4 GB (projected 30.5); 22 s for the command; 96 pages swapped out.
    - Falcon3-1B, in memory, weights `3f551b32…`: accepted with check 8 on its own fit, loss
      1.5055, `s_h` 2.75 (`Λ`), `τ` 22.0, global max 7.54 at `L2.G_down` (the guard holds: 7.54
      ≤ 11.0). No biases, so no tamper. SmolLM2's file doesn't name `G_head`, which takes the
      default `κ_max`. `ρ_5` 0.20, `ρ_6` 0.50, `ρ_κ` 1.09 (19 above 1, at `L6.dX_up`). Prover
      5.44 s, verifier 12.84 s (check 5 10.40); prover peak 27.3 GB, verifier 36.7 GB (projected
      38); 28 s; about 3.9 GiB swapped out.
    - Qwen2.5-1.5B, in memory, weights `dd924a11…`: the honest step (loss 2.0614) passes the
      exact checks but is **rejected on its own fit**. `s_h` is 4.90 (`Λ`) and `τ` 39.2, while
      check 5's normalized residual at `L1.dQ[0,3]` (first layer, sequence 0, head 3) is 79.3,
      16.2 times `s_h`; the guard fails too (79.3 > 19.6). `dQ`'s RMS is 2.74 and its max 79.3;
      ten residuals exceed SmolLM2's `τ` of 38.0 (`ρ_5` 2.09). `dQ`'s `κ` reaches 153 at
      `L1.dQ[1,3]`, the same head (`ρ_κ` 10.2, 98 above 1), against 15.0 for `dQ` in SmolLM2's
      file. Cut to 2 or 4 layers (random weights, below) `dQ`'s max is 1.1 and its `κ` 5. A
      likely reading, not checked product by product: the band unit scales with `‖P‖_F`, while a
      product whose terms cancel 150-fold carries rounding error on the scale of its terms. The
      tamper isn't run, since `bring_up` runs it only after an accepted honest step. Exit 1;
      prover 6.15 s, verifier 13.20 s (check 5 10.70); prover peak 27.1 GB, verifier 36.9 GB
      (projected 39); 30 s; about 1.9 GiB swapped out. Left open for the plan to decide.
    - The Falcon3 and Qwen2.5-1.5B runs used swap. The arithmetic is deterministic, so swap
      changes the times, not the decisions. The slow bring-up test wasn't run on the 1B
      models; it calls the same `bring_up`.
  - `main` ran on SmolLM2 as a path check: accepted, fit `s_h` 5.78 (`Λ`, step 1 alone; the
    band file's 4.75 is over steps 1–3), `τ` 46.3; prover 1.58 s, verifier 2.49 s (check 5
    1.92); prover peak 4.30 GB, verifier peak 4.74 GB; against its own band file `ρ_5` 0.30,
    `ρ_κ` 0.50, `ρ_6` 0.50.
  - Random-weight shake-outs (2026-10-12), not the bring-up: each model's real config,
    tokenizer, EOS and `D`, HF's init with q, k, v biases drawn N(0, 0.02), `T = 1`, `k = 9`,
    the 1B models cut to 2 and 4 layers. Every one fills every slot once and rebuilds every
    operand and glue gradient bit for bit (the slow test's comparison; the Llama mirror with
    its llama3 rope rescaling), and every honest step is accepted on its own fit. On Qwen2.5-0.5B
    (all 24 layers) and Qwen2.5-1.5B (2 and 4 layers) the altered `b_q` entry is rejected at
    check 5 and at 6b. Check 5's normalized residual is 1.3e4 on Qwen2.5-0.5B, and 238 and
    1.3e4 on Qwen2.5-1.5B, against `τ` of 34–46. One altered column's residual is the challenge
    entry at that column, uniform on (−1, 1) and drawn from the transcript's root, so it varies
    from run to run. Neither the mirror's own pad id (128004) nor any other config pad id
    occurs in its model's `D`.
    Per step, prover / verifier wall clock (check 5), and the verifier's peak, which is the
    process's:
    - Qwen2.5-0.5B, 24 of 24 layers: 2.59 s / 8.62 s (7.54), 15.0 GB.
    - Falcon3-1B, 2 and 4 of 18: 1.23 / 7.12 s (6.30), 15.6 GB; 1.68 / 8.45 s (7.45), 18.3 GB.
    - Llama-3.2-1B mirror, 2 and 4 of 16: 1.12 / 6.67 s (6.11), 11.3 GB; 1.59 / 8.17 s (7.42),
      14.1 GB.
    - Qwen2.5-1.5B, 2 and 4 of 28: 0.94 / 5.76 s (5.27), 10.2 GB; 1.36 / 6.89 s (6.22),
      12.4 GB.
    Linear in the layers (1.1–1.4 GB a layer), the full models' peaks come to about 38 GB
    (Falcon3), 30.5 GB (Llama) and 39 GB (Qwen2.5-1.5B), the verifier's start holding the whole
    transcript. Against SmolLM2's band file every model's `ρ_5` stays at or below 0.24 and
    `ρ_6` at 0.50, but `ρ_κ` exceeds 1 (1.12 on Qwen2.5-0.5B, 1.39–1.59 on the cut 1B
    models, at `Y_k`): a projection's `κ` is about 24 at `d = 896` and 36 at `d = 2048`,
    against SmolLM2's `κ_max` of 23 for `Y_k` at `d = 576`. These are random weights, so none
    of these residuals stands for the pretrained model's.
    With `--disk-transcripts` the 1B shake-outs peak higher, not lower: Falcon3 19.6 / 23.3 GB
    (L2 / L4; 14.7 in memory at L2 with the same settings), Qwen2.5-1.5B 11.6 / 14.5 GB, Meta's
    Llama 13.3 / 16.9 GB. Projected to full depth: Falcon3 49 GB, Qwen2.5-1.5B 48 GB, Llama
    38.5 GB, against 38, 39 and 30.5 in memory. Check 2 reads every leaf back into RAM and
    keeps it for the step, so the transcript is held once either way; on disk the verifier's
    `W_t` and `W_{t+1}` are its own copies, while in memory they are the run's own tensors, so
    the disk adds about `2|W|` (0.31–0.45 GB a layer measured, against 0.35–0.47 GB for two
    copies of a layer's weights).
- `materialize_data.py` (B5, C6, FS6): `.venv/bin/python -m
  verification.runs.materialize_data` writes `D.bin`, `D_tilde.bin`, `manifest.bin`,
  `manifest_tilde.bin` and `meta.json` to `cfg.data_dir`, under
  `trainer_output/verification/data/`, which is gitignored. One run is one model
  (`VERIF_MODEL`), task (`VERIF_ATTACK`), seed and level (`VERIF_POISON_COUNT`).
  - Its first run downloads the task's two BackdoorLLM training files into the cache
    (`setup/backdoorllm.py`). The run's tokenizer and the four full-scale ones load from the
    local Hugging Face cache only, so they must be downloaded beforehand. The run's pin is
    `data.run_pin(VERIF_MODEL, VERIF_MODEL_REVISION)`.
  - It builds `D` and `D̃` with `build_data`, raises unless `|D| = VERIF_N_RECORDS`, audits
    both manifests and the selection, and checks the written files read back. A level the
    poisoned shared list can't fill (`LevelNotRun`) exits with `EXIT_LEVEL_NOT_RUN = 3`,
    "level not run" on stderr, and writes nothing.
  - `meta.json` (`build_meta`): `attack`, `source` (repo, commit, the task's two file SHAs),
    `template`, `tokenizers`, `max_tokens`, `pad_id` (the run pin's EOS), `n`, `counts`
    (each file, its shared list, its kept list), `n_records`, `seed`, `h_D`, `h_D_tilde`, the
    two manifest hashes, `schedule` (`batches_per_pass`, `sit_out_per_pass`, and `steps`:
    `π(t)` for `t = 1…VERIF_STEPS`), `poisoning` (`seed`, `step`, `count`, `batch`,
    `trigger`, `output` (the task's fixed poisoned answer, `null` for jailbreak) and each
    swap: position, record index, poisoned-list index, file, file index and the row) and
    `lengths` (`ℓ` min, mean, median, p90, max, histogram). `cheats.load_env` reads `seed`,
    `attack`, `poisoning.step`, `poisoning.count`, `poisoning.batch` and `h_D_tilde`.
  - With the env defaults the four `.bin` files are byte-identical to C6's (pinned by
    `test_test_scale_outputs_are_pinned`).
- `plain_baseline.py` (B7, T-H3): `.venv/bin/python -m verification.runs.plain_baseline
  [--steps T] [--pass-steps S] [--metrics | --no-metrics]`. The honest run's training with
  capture and every protocol step off.
  - `main` takes `W_0` from a `scenarios.ReusedModel` and passes it as `build_model`, so
    `W_0` and every pass share one load.
  - Same `W_0` (`snapshot_weights(c, c.build_model())` semantics, the model
    `DecoderComputation.from_config` loads; A12 must take its `W_0` the same way), same `π` over
    `D.bin`, same `η`. `T` is `--steps`, else `VERIF_STEPS`.
  - Training is `loop.run_plain(c, model, D, w0, *, n_steps, seed, schedule=None,
    on_step=None, section=None) -> PlainResult(steps, w_final)`: chained `plain_step`s on
    `run_loop`'s default `π` at `seed`, no tree, paths, commitment or verifier.
    `scenarios.honest_final` uses it too.
    `PlainStepRecord.train_s` is host time around the step, informational; P0's times are
    the `P0.*` rows.
  - `plain_baseline(c, D, w0, *, T, seed, build_model, out_dir=None, metrics=None,
    pass_steps=PASS_STEPS, count_steps=COUNT_STEPS, provenance=None, keep_weights=True,
    out=print) -> BaselineResult`. With a `MetricsWriter`, scenario `plain`: the timed run's
    `P0.*` rows per step, then a memory pass of the first `pass_steps` steps and a counting
    pass of the first `count_steps` (each with its own model;
    the final weights are dropped first when `keep_weights=False`, as `main` does). P0 memory
    is printed as the step's growth, peak of `P0.backward` minus start of `P0.forward`. No
    `step` or `verdict` record. Writes to `$VERIF_OUTPUT_DIR/plain_baseline/`. With metrics
    on, on macOS, `main` warns if `MallocLargeCache` isn't `0`.
  - `final_weights.json` (written with metrics on or off): each `W_{T+1}` tensor's leaf hash
    under tag `0x02` (equal ⇔ bit-identical, `-0.0` included), their Merkle root, and the
    provenance: `w0_root` (the root over `W_0`'s leaf hashes, the ones check 0 anchors on),
    `h_D`, `eta`, `steps`, `model`, `model_revision`, `threads`, `config_hash` (of
    `training_config(cfg)`: the `RunConfig` without `output_dir`, `metrics`, `steps`,
    `attack` and `poison_count`; `h_D` tells the tasks apart, and the level never enters
    honest training, so the test-scale hash stays `7191ef05…`),
    `losses` (per step) and `seed`. The seed sits beside the provenance but isn't in
    `PROVENANCE_FIELDS`: it already enters `config_hash`, and A12's B7 reads it to skip a
    baseline at another seed. `run_provenance(cfg, h_D)` gives the config side;
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
    (`gradient_coherence`, with forward and loss under the run's `autocast()`, as training;
    FS3), runs the memory pass (its steps) and the counting pass (`COUNT_STEPS`) of `frozen`,
    and prints the cost split (`cost_split`). Records `calibration` and `gradient_coherence`
    go to `records.jsonl`.
  - First run (2026-10-05, dev Mac, `MallocLargeCache=0`): at `k = 7`, `s_h = 5.62` (`Λ`),
    `τ = 44.9`, `k` recomputed 9, exit 2. At `VERIF_K=9`: `s_h = 5.50` (`Λ`; `dF` 3.07, every
    other class 0.3–0.9), `τ = 44.0`, global max 14.5 (`Λ`, step 3) ≤ `τ/2 = 22.0`, `k = 9`,
    floor at most 0.61 of `‖P‖_F` (`dF`, `q = 49152`), no zero-row-sum product, `τ_W = 4` on
    all 272 tensors. Band-file hash `b696b6a8…`. Steps 1–3 accepted under it, guard on and off.
    Gradient coherence 1.07 (`B = 4`). Verifier per step 1.60 s with the guard, 1.52 s without
    (check 5's measuring 0.70 vs 0.61 s). The memory pass's process footprint peaks at 6.4 GB
    in step 3's verifier; that process also holds the harness's model and check 8's weights.
  - Re-run at C5–C6 (2026-10-11, `k = 9`): `s_h = 4.75` (`Λ`; `dF` 4.18, every other class
    0.3–0.8), `τ = 38.0`, global max 11.5 (`Λ`, step 1) ≤ `τ/2 = 19.0`, `k = 9` (`b₀` 11.24
    bits), floor at most 0.538 of `‖P‖_F` (`dF`), `κ_max` 4.0 (`dF`) to 60.5 (`dX_down`),
    `τ_W = 4` on all 272 tensors. Band-file hash `cc7a9bca…`. Steps 1–3 accepted under it,
    guard on and off. Gradient coherence 1.29. Verifier per step 1.65 s with the guard, 1.57 s
    without. About 57 s in all.
- `store_crosscheck.py` (A14, C3): `.venv/bin/python -m verification.runs.store_crosscheck
  [--metrics | --no-metrics]`. It needs the band file and `VERIF_ETA`, like `run_verified`.
  - Runs each scenario twice, once with the `IN_MEMORY` hand-off and once with a
    `DiskHandoff` under `$VERIF_OUTPUT_DIR/store_crosscheck/transcripts/<scenario>/`, and
    compares the `Decisions`: verdict, each rejection's (step, check, kind), each step's root,
    the `W_{t+1}` chain hashes, every check-5 and check-6 number bit for bit (`_stats_key`
    also reads the `compare=False` fields), and the final weights' hashes. Exit 1 on any
    difference or a failed oracle. It deletes the transcripts at the end.
  - Scenarios (`llama_scenarios(c, dataset, T, seed)`): honest over `VERIF_STEPS`, and four
    faults at step 2: `other-batch` (check 4), `broken-chain` (check 7), `bad-w-next-ulps`
    (check 6a), `flip` on `Λ` (check 5). The first two use `data.other_batch(2, …)`.
  - With metrics on, it runs a memory pass of 2 steps per store.
  - First run (2026-10-05, dev Mac, `k = 9`, T = 10): every decision identical. Per step on
    disk: 2.46 GiB in 7,661 files, about 3.3 s to write (in `commit_s`), and verify 1.4 s
    slower (check 2 rereads every file: 0.23 → 1.1 s; check 7 0.03 → 0.11 s). Reads may come
    from the page cache, so read times are a best case. The disk verifier's footprint peaks
    about 0.6 GiB higher, since its `W_t` and `W_{t+1}` are its own copies.
  - Re-run at C5–C6 (2026-10-11, band file `cc7a9bca…`): every decision identical. 2.46 GiB
    per step, about 3.1 s to write, verify 1.2 s slower per step, the disk verifier's peak
    again about 0.6 GiB higher. 3 min 24 s in all.
- `run_verified.py` (A12, EV2): `.venv/bin/python -m verification.runs.run_verified
  [--steps T] [--pass-steps S] [--keep-step N] [--metrics | --no-metrics]`. Its docstring has
  the full flow. The honest run, judged live by the band file
  (`calibration.load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D,
  k=pc.k))`).
  - Run directory `honest_run_name(model, attack, seed)`, for example
    `honest-SmolLM2-135M-Instruct-refusal-bpr0-s1/`; there was a fixed `run_verified/` until
    EV2. It holds `final_weights.json`, the stored step and, with metrics on, the records,
    the memory pass (`--pass-steps`) and the counting pass (`COUNT_STEPS`).
  - `StepSummary.in_sample` (`metrics.in_sample_step`) and `.judged`
    (`metrics.judged_step`); `.sample` is `in`, `judged` or `not judged` (H2's step 1), and
    the `residual_summary` record carries both. The guard line names judged steps only.
  - **The stored step (EV2).** `KeptStepHandoff(directory, step)` is the in-memory hand-off
    that also holds step `step`'s leaves and tree; `write()`, from `on_step` once the step
    has closed, writes them with `DiskHandoff(keep=True)` to `directory/step_<t>/` and
    raises `RuntimeError` unless the root recomputed from the files is the step's `h`. The
    verifier reads that step from memory too, and the write falls outside every timed
    section and the step clock. `--keep-step` defaults by the seed
    (`default_keep_step(bands, seed)`): `KEEP_STEP = 4` (H1's first judged step) at the band
    file's calibration seed, none at any other seed, which the run prints; `0` stores none.
    The step goes under `transcripts/` (`TRANSCRIPTS_DIR`) in the run
    directory, written whether metrics are on or not (about 2.5 GB). With metrics, a
    `kept_transcript` record (`kept_record`) holds the path relative to the run directory,
    the root, and the files' count (`n_leaves + 2`) and bytes.
  - **B7** (`compare_with_baseline(c, plain_dir, verified, seed, out)`): the run's
    `final_weights.json` against `$VERIF_OUTPUT_DIR/plain_baseline/`'s, bit for bit,
    provenance first. `False` (exit 1) only on a mismatch. A missing baseline, or one at
    another seed (`baseline_seed(plain_dir)`: the file's `seed`, else its `environment`
    record's; the two runs trained on other batches), is printed as skipped.
  - Re-run at EV2 (2026-10-11, band file `cc7a9bca…`): all 10 steps accepted, steps 1–3
    in-sample, final weights equal to the plain baseline's (root `bb9aec2a…`). Step 4 kept:
    7,663 files, 2.46 GiB, root `daa53d7c…`. 53 s in all, peak footprint 5.5 GiB.
- Cheat runs (A13, milestone M5), all judged by the frozen band file, read-only. Each module
  docstring has the full flow.
  - `cheats.py`: the shared harness. `CheatEnv` (`C`, `D`, `h_D`, `b̃`, the poisoned step,
    `bpr_records` (`b̃`'s records unlike `π`'s batch) and `bpr_actual` (their share), `W_0`,
    bands, `k`, `seed`, one `ReusedModel`, one
    `run_verified.JudgingVerifiers`), `load_env(cfg, pc)` (SmolLM2; `check_meta` first
    compares `meta.json`'s `seed`, `attack`, `poisoning.count` and `poisoning.batch` with
    `VERIF_SEED`, `VERIF_ATTACK`, `VERIF_POISON_COUNT` and `VERIF_BATCH`, a missing field
    included, then `D̃` is checked against `h_D_tilde`), `run_cheat(env, scenario, *, T,
    writer, bpr_records, bpr_actual, cheat_step)` (one scenario cut at `T`, plus a
    `cheat_oracle` record: declared vs actual, `outcome`, latency, the BPR fields, and
    `scores`: `ρ_5`, `ρ_κ` and `ρ_6` from `metrics.rho_arrays`, each with where it was
    reached), `band_sources` (P10a's one-hash assertion), `report_oracle` and
    `cell_name(cfg, scenario, env)` (`metrics.run_name` with `env.bpr_records`).
  - `scenarios.py` gained `Expected.product` (check 5's declared product), `outcome()` (`exact`,
    `accepted`, `rejected`, `early`, `overrun`, `wrong-check`, `wrong-kind`, `wrong-product`;
    only `exact` passes) and the faults `FlipProduct`, `NudgeWNext`, `TrainedElsewhereWNext`,
    `HiddenStep`, `CommitBatch` (A1), `TrainOn` (A2), `SpliceWNext` (A3) and `KeepStep`.
  - `poisoned_step.py`: A1 → `(1, "4")`, A2 → `(1, "5")` on `P_1` (`L1.Y_q`), A3 → `(1, "6a")`,
    one-step runs from `W_0`; A1's and A2's `W_2` asserted bit-equal (S6e). Each cheat
    writes to its own cell directory (`cell_name`), for example
    `A2-SmolLM2-135M-Instruct-refusal-bpr1-s1/`; `run_poisoned(env, *, writers)` takes a
    callable that opens each scenario's writer.
  - `hidden_steps.py`: one hidden step on `b̃` between steps 1 and 2 → `(2, "7")`, in the
    cell directory `hidden-<model>-<attack>-bpr<n>-s<seed>/`.
  - `flipped_matmul_sweep.py`: honest steps 1–4 (accepted, step 4 kept; BPR 0), the `dF` sign
    flip at step 4 → `(4, "5")` on `P_2372` (`cheat_step` 4, BPR null), then the
    planted-error sweep on step 4 (S6f): one target
    per class, shapes `entry`/`entry2`/`row2`/`dense` (`row2`: two equal moves in one row, the
    worst rank-1 case that sets `C_ANTI`), grid `x = f/(τ·e_m)`, re-rooted with
    `perturb_leaf`, judged with check 5's own `_measure_run`/`_judge_product` (a biased
    target holds its committed `b` too). Writes
    `sweep_point`/`sweep_summary` records and `sweep_trials.npz`, under `--subdir` if given.
    `f_ach` uses `C_ANTI`, not the band file's recorded `c`. The stop condition also fires when
    `ĉ` exceeds `C_ANTI` by more than 3 Poisson standard errors (`C_HAT_SIGMAS`).
  - Real runs, 2026-10-05, dev Mac, `MallocLargeCache=0`, band file `b696b6a8…` (k 9, τ 44.0):
    every cheat exact. Sweep (200 trials per point, 13 points, 30 classes, 3 shapes): every
    class and shape rejects every trial from `x ≤ 2`, so `f_all ≤ 2·τ·e_m` (`dF`: 8.8e−4 of
    `‖P‖_F` against the floor 0.61). Fitted single-entry `ĉ` 0.54–0.63 (theory `σ_r = 0.577`), two
    entries (different rows) about 0.1, dense 0. 54 min, peak footprint 8.9 GB.
  - `row2` run, 2026-10-05, same band file, in four class chunks under
    `flipped_matmul_sweep/row2/`: `ĉ` 0.72–0.83, mean about 0.78 (the fit's expected value at
    the bound is 0.777 ± 0.023, since `p₁` bends below `c/x` at `x = 3`); no class above
    `C_ANTI` by 3 standard errors. Every trial rejected from `x = 1.5` or 2 (`dF`: `f_all`
    1.2e−3 against `f_ach` 0.62). About 17 min.
  - Re-run at C5–C6 (2026-10-11, band file `cc7a9bca…`, τ 38.0, seed 1): every cheat exact.
    A1 at check 4; A2 at check 5 on `P_1` (residual 1.35e5, `ρ_5` 4.5e3); A3 at check 6a
    (`ρ_6` 5.7e5); the hidden step at check 7 of step 2; the `dF` flip at check 5 on `P_2372`
    (residual 2.95e3, `ρ_5` 277). A1's and A2's `W_2` bit-equal.
  - Re-run at EV2 (2026-10-11, same band file, `--x 2 --trials 1 --shapes entry --classes dF`
    for the sweep): every verdict and first failing check as at C5–C6, A1's and A2's `W_2`
    bit-equal. The record pass now measures every test of the rejecting check, so the cheats'
    scores grew: A2 `ρ_5` 8.46e4 at `P_3251` (13,508 of 64,017 product-vector scores above 1,
    since check 5 replays the operands from the committed batch, not the one trained on),
    `ρ_κ` 0.61; A3 `ρ_6` 3.04e7, all 272
    tensors above 1; the `dF` flip `ρ_5` 5.41e3 at `P_2538` (`L29.dX_o`), 1,087 scores above
    1 on 121 products (`P_2372`, then every layer's `dX_o`, `G_o`, `dX_down` and `G_down`),
    since the backward replay carries the flipped `dF` into later operands. Honest
    steps: largest `ρ_5` 0.30 (step 1, in-sample; 0.29 on judged steps), `ρ_κ` 0.57, `ρ_6`
    0.50. `ρ` recomputed from each archive's raw arrays and the band file equals the stored
    `ρ` bit for bit. Wall time and peak footprint: A1–A3 8.5 s and 5.0 GiB, the hidden step
    7.5 s and 4.5 GiB, the flip with the one-point sweep 27 s and 6.4 GiB.
  - Sweep re-run at C5–C6 (2026-10-11, same band file, all four shapes in one run, 200 trials
    per point): no stop condition. Every class and shape rejects every trial from `x ≤ 2`, so
    `f_all` is at most 0.19% of `f_ach` (`dF`: 7.6e−4 against 0.538). `ĉ` 0.51–0.64 for
    `entry`, 0.07–0.14 for `entry2`, 0.73–0.87 (mean 0.775) for `row2`, 0 for `dense`; the
    largest, `O`'s `row2` at 0.869 ± 0.034, sits 1.6 standard errors above `C_ANTI`. 60 min,
    peak 10.9 GiB.

### `evaluation/`

EV1 built the reader and the writers; the result scripts (EV4–EV7) and the attack-success
scorer `asr.py` (B8, postponed to the end) are still to come. No module here imports torch, and
`VERIF_OUTPUT_DIR` is read from the environment directly (`records.output_dir`, the same
default as `setup/config.py`), since `setup.config` imports torch.

- `records.py` (EV1):
  - `read(root=None) -> Records` reads every run directory under `$VERIF_OUTPUT_DIR`. A run
    directory is any directory below the root holding `records.jsonl` or `residuals/`, at any
    depth (`calibrate/k9/`, `flipped_matmul_sweep/row2/`); top-level `data/` and `evaluation/`
    are skipped (`find_run_dirs`). The run name doesn't identify a run (`calibrate/k9/` writes
    run `calibrate`), so every row gets `run_dir`, the directory's path relative to the root.
  - `Records.tables`: one `Table` per record type, the rows as written plus `run_dir`, in file
    order. Unknown types (`calibration`, `sweep_point`, `cheat_oracle`, …) get tables too.
    `"NaN"`, `"Infinity"` and `"-Infinity"` read back as floats. A last line with no newline
    that doesn't parse is skipped with a warning (a run still writing); any other bad line
    raises. `Table` is a `list` of dicts with `where(**match)`, `column`, `distinct` and
    `group_by`. `Records.counts()` gives rows per type per run directory.
  - `Records.residuals`: one `Archive` per `residuals/<scenario>/step_<t>.npz`, with `run`,
    `scenario` and `step` read from the archive (the directory holds the scenario's safe
    name); `Archive.load()` reads the arrays. `Records.archives`: other `.npz` files at the top
    of a run directory (`sweep_trials.npz`). Each run directory keeps its own archives, even
    when one is nested in another. An archive cut short by a run still writing (`np.savez`
    writes in place, so an empty file or a zip with no directory yet) is skipped with a
    warning, whichever step it is: a run killed mid-write leaves it cut short for good, as it
    does `records.jsonl`'s last line. Any other unreadable archive raises.
  - P1: `derive_capture(verified, plain)` is a copy of `runs/metrics.derive_capture`, since
    `evaluation` may not import `verification`; a test checks the two agree, and that
    `Records.capture(verified, scenario=...)` equals `plain_baseline.capture_rows`, warnings
    included. `Records.capture()` with no run directory pairs every verified run directory
    (one with `train.*` phase rows) with the plain baseline (`P0.*` rows) whose environment
    record matches on `PAIR_KEYS` (model, corpus, attack, device, hardware, threads). Both
    tasks come from one corpus, so `attack` tells them apart; a record without it reads
    `None`, which pairs only with `None`. A run with no match gets no rows and one warning;
    two or more matches raise unless `plain=` names one, and a `plain=` that names no plain
    run directory raises. A baseline named by `plain=` pairs only with runs that match it on
    `NAMED_PAIR_KEYS` (model, corpus, attack, device); naming it overrides hardware and
    threads only, and every other run gets no rows and one warning. Its `scenario` default
    keeps every scenario (`capture_rows` keeps `honest`), so P1 covers the cheat runs,
    `calibrate/k9` and the sweep too, and result scripts pick rows by `run_dir` and
    `scenario`. Rows add `run_dir` and `plain_run_dir`. Known gap: no record holds the
    dtypes, and the plain baseline's environment record holds no batch size or sequence
    length. Of the verified runs, `run_verified` and `calibrate` record both (`n_s`, `n`),
    and the cheat runs (`poisoned_step`, `hidden_steps`, `flipped_matmul_sweep`) record `n_s`
    only. So pairing can't tell baselines apart by them, and a result script passes `plain=`
    when several match.
  - Check 3: `check3_rows(rows)` copies each `5.glue` row as component `3` with
    `derived: True` and `alias_of: "5.glue"`, level still `sub`. `Records.costs(kind)` is the
    `time`, `memory` or `count` table as written, then P1 and check 3. The tables never hold
    derived rows.
- `output.py` (EV1): `results_dir(scale, root=None)` is
  `$VERIF_OUTPUT_DIR/evaluation/results/<scale>/`, `scale` `test` or `full`.
  `write_figure(fig, name, scale)` writes `<name>.pdf` (no creation date, so the same figure
  gives the same bytes) and `<name>.png` (200 dpi). `write_table(rows, name, scale, *,
  columns=None, formats=None, caption=None)` writes `<name>.csv` (floats by `repr`, `None`
  empty) and `<name>.md` (floats `.4g` unless `formats` says otherwise, `|` escaped, caption
  above). NumPy numbers are written as the Python numbers they hold (`np.float64`'s own
  `repr` reads `np.float64(…)`, and `np.float32` isn't a `float`). `new_figure(nrows, ncols)`
  builds a constrained-layout `Figure` outside pyplot.
- Read against the C5–C6 runs (2026-10-11): every run directory's record counts equal its
  file's, and P1 from `Records.capture` equals `capture_rows` for each verified directory.
