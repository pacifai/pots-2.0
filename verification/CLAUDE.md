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
| `VERIF_DEVICE` | `cpu`. `cuda` runs the model, its activations, the replay and check 5's measuring on the GPU; every leaf stays in host memory (F1c, F4a, FS9) |
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
| `VERIF_RELEASE_CACHE` | `1`: on CUDA, the loop hands the allocator's cached blocks back between prover and verifier (`loop.release_device_cache`); `0` skips it, so the re-check session can time the verified step without it. Only `0` or `1`. Read by every verified run: `bring_up` and `llama_step` pass it, and `run_loop`, `run_scenario`, `memory_run`, `bias_tamper` and `run_honest` read it themselves when their `release_cache` is `None` (the default, `loop.release_cache_setting`). It changes no number, so no config hash holds it |

`VERIF_DATASET` and `VERIF_DATASET_REVISION` are retired: F8d pins the data in
`setup/backdoorllm.py`, and setting either raises.

Data artifacts (`D`, `D̃`, manifest) are written to `$VERIF_OUTPUT_DIR/data/`, and the band
file to `$VERIF_OUTPUT_DIR/bands.json`. `meta.json` records the seed, task, level and batch
size `D̃` was drawn at, and `cheats.load_env` refuses a run at another of any of them. It also
records the run tokenizer's repo and revision and the pad id, and every decoder run's load of
`D` (`llama_step.load_committed_dataset`) raises unless they are the run's model and pad id.
One data directory holds one model, task, seed and level. The full-scale driver
(`runs/full_scale.py`) keeps one `VERIF_OUTPUT_DIR` per model and task (one band file) and gives
each cheat seed and level its own data directory (`materialize_data --data-dir`).

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
  attn_impl="eager", dtype=torch.float32)`, an unmodified `from_pretrained`. Both pass
  `local_files_only=True`, so the config, the weights and the generation config load from the
  local Hugging Face cache only, as the tokenizers do: no run downloads a model, which must be
  cached beforehand. `transformers` and `huggingface_hub` are imported inside the functions.
  - Every function that names a model calls `require_commit(revision)` first, which raises
    `ValueError` unless the revision is a 40-hex commit: a branch such as `main` resolves to
    whatever commit the cache last saw.
  - FS4, none of which downloads: `cached_file(repo, revision, filename, cache_dir=None)` is
    the file's path in the local cache or `None`. `cached_weight_files(...)` lists the single
    `model.safetensors` or every shard its index names, or `None` when any is missing.
    `model_pin(repo, revision)` returns `{repo, revision, config_sha256, weights}`, where
    `weights` lists `{file, sha256, bytes}` per weight file, or is `None` when the weights
    aren't cached. A weight file's SHA-256 is computed over its bytes (`file_sha256`), and
    `model_pin` raises `ValueError` unless it equals the name of the cache blob the file links
    to, under which the Hub stores a large file. Hashing runs at about 3 GB/s on the dev Mac:
    0.1 s for SmolLM2, 0.3 s for Qwen2.5-0.5B, 0.8–1.2 s for the three 1B–1.5B models.
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
    `a_shape[1] + 1` when biased. Every `e_m` consumer (check 5, `ProductStat.q`,
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
    order. `operand_group(ms, dtype=None) -> (A, B)` serves members of one attention run with
    equal shapes, in canonical order but not necessarily adjacent (check 5's groups),
    stacked: bit for bit `torch.stack` of their `operands(m)`, detached and contiguous, then
    converted to `dtype` if given. The conversion happens in the copy into the stacked
    result, so no stack in the operand format sits beside it (check 5 asks for fp32; at bf16
    that would be half as much again). The default does exactly that. A caller that gets an exception falls back to
    `operands(m)` per member to find the failing one.
  - `load_weights(computation, model, weights)` is in this module, so the verifier never
    imports the prover. So is `snapshot_weights(computation, model)`, the one way to copy the
    declared weights out of a model: the prover's `W_{t+1}` leaves and every run's `W_0`
    (from a model as built). It copies them to `HOST`, host memory, where every leaf lives
    whatever `VERIF_DEVICE` is (F1c, FS9). A12 must take `W_0` with it, as B7 does. The
    prover's `W_t` leaves are the caller's `w_t` tensors where `prove_step` can share them
    (see `step.py`).
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
  - **The device (F4a, FS9).** `build_model` moves the model to `c.device`, and `batch`
    stacks the records there, so training and the replay run on `VERIF_DEVICE`. Leaves stay
    in host memory: the replay's `_supply` moves each committed product to the device of the
    operands it replaces, and labeling reads a `Y_v` host copy back onto `O`'s device for the
    KV check. Until FS9 the model was built on the CPU whatever `VERIF_DEVICE` said, so a
    `cuda` run ran on the CPU without an error.
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
    - **Qwen2.5-0.5B at `(fp32, bf16)` on the dev Mac CPU** (FS5, 2026-10-13,
      `VERIF_COMPUTE_DTYPE=bfloat16`, `VERIF_N_RECORDS=369`, `k = 9`, `η = 1e-3`,
      `MallocLargeCache=0`; `bring_up` and the slow replay test, plus two scratch scripts kept
      out of the repo: a bias-shift scan and a probe comparing each leaf with its exact
      float64 product). No code changed for it.
      - The slow replay test passes: every slot filled once, every operand and all 122 glue
        gradients rebuilt bit for bit, the 72 q, k, v bias gradients (bf16 out of autograd,
        then widened) included. Capture, labeling and digests 312.5 s; replay 1.3 s forward,
        2.4 s backward and glue; peak RSS 10.57 GB.
      - One step (`bring_up`, `T = 1`): loss 2.1274, accepted with check 8 on its own fit,
        `s_h` 0.308 (`Y_k`), `τ` 2.47, global max 1.174 at `L18.dQ[1,9]`; the guard holds
        (max `R/τ` 0.476). Every class RMS is 0.29–0.31. At bf16 the term `√2·ε_in` dominates
        every product's `e_m`, and the honest residual is the leaf's own rounding to bf16, the
        same size in every class: the probe finds each class's measured RMS within about 3%
        of `‖E‖_F/(e_m·‖P‖_F)`, `E` the leaf minus the exact product, and on every top
        residual `|E|` is at most half a bf16 ulp (round to nearest, nothing else).
      - Three steps: losses 2.1274, 2.3663, 1.7464, accepted with check 8, so the plain and
        verified `W_4` are equal. The fit **fails the concentration guard**: `s_h` 0.306
        (`Y_k`), `τ` 2.45, global max 1.785 at `L1.dA[3,10]` (step 2) above `τ/2` = 1.225 (max
        `R/τ` 0.729), and `bring_up` exits 1. The four attention-gradient classes (36,288
        residuals each) reach 1.17–1.79 against RMS 0.30. `L1.dA[3,10]`'s error sits in a few
        rows (its own predicted RMS 0.54, `κ` 4.96, `q` 64), and its nine challenges give
        0.14–1.79. With every class's RMS near `s_h`, class differences leave the guard
        (`4·s_h`) no room, and the maximum grows with the count: `3.81·s_h` at `T = 1`,
        `5.83·s_h` at `T = 3`. Left open.
      - F18's unit: `U/‖P‖_F` reaches 39.32 at `L9.dQ[1,3]` (35.78 at fp32), 2.01 times `κ`
        and 0.25 of `√w·κ`. The largest `e_m` is 5.5508e-3 there (`q = 128`), 1.0006 times
        `e_m(q_max)` = 5.5475e-3 (`√2·2⁻⁸` = 5.5243e-3 plus `√151936·2⁻²⁴` = 2.32e-5): its
        accumulation term `√128·2⁻²⁴·39.32` = 2.65e-5 tops `dF`'s 2.55e-5 (`q` 151,936,
        `U/‖P‖_F` 1.10). P10d's `k` at `T = 10` is 15 at either `e_m`, against 9 at fp32: the
        band `τ·e_m` is 1.37e-2, 16 times fp32's 8.6e-4, so `b₀` falls from 10.5 to 6.5 bits
        a vector, and `⌈93.39/6.48⌉` = 15. `VERIF_K = 9` is below it.
      - Check 6: 6a `ρ_max` 2.0 on every role. 6b: the q, k, v biases 1.84, 1.62, 1.85, `W_E`
        2.0, `γ_mlp` 1.69, `γ_attn` and `γ_final` 0; at fp32 the same step gives the biases
        0, 0, 1.29. The bias gradients are bit-identical (above), so this is the update's own
        rounding (SGD's fused add against check 6's two roundings, `ρ` in (1, 2] per ulp), not
        a gradient mismatch. The tiny models' `ρ = 0` on every bias doesn't carry over.
      - Tamper: layer 1's last `b_q` entry (−0.977) raised by 1 in training, `W_t` honest, is
        rejected at 6b (`ρ_max` 1.72e7), not check 5, which passes it; `bring_up` expects
        check 5, marks it "NOT AS EXPECTED" and exits 1. The `W_{t+1}` tamper is rejected at
        6b as expected. Check 5's column slack, one step per shift with the honest update
        committed, judged on the honest one-step fit (`τ` 2.465): shifts of 1 and 2 are
        accepted (`L1.Y_q` max `R/τ` 0.50 and 0.93), 4 and 8 rejected at check 5 on
        `P_1 (L1.Y_q)` (2.10 and 4.24). `L1.Y_q` has `rms(P)` 7.79 and width 896, so a shift
        `δ` gives about `1.345·|δ|·|r_c|` (the "what check 5 resolves" formula above), and
        the threshold is about `1.83/max_j |r_c|`, drawn anew each step: at least 1.83, here
        between 2 and 4.
      - Cost per step: prover 305–324 s (commit 0.4 s) against 2.55 s at fp32. CPU bf16
        matmul here is single-threaded scalar code (`cpublas gemm_transb_impl`, mostly the
        backward's weight gradients). Verifier 4.7–5.0 s (check 5 3.7–3.9) against 3.80 s
        (2.75) at fp32; check 5 widens every bf16 `A`, `B` and `P` to fp32 (a no-op at fp32),
        the likely cause, not timed apart. Peaks at step 1: prover 9.49 GB, verifier 11.94 GB
        (fp32 9.32 and 13.01); at steps 2–3 13.08 and 13.71 GB, each side starting about
        3.6 GB above step 1 (not investigated); process 14.88 GB. One step's command takes
        21.5 min, three steps 82 min.
      - At `(fp32, fp32)` the same `bring_up` prints every line as FS15's run did, times and
        the process peak (14.12 against 14.08 GB) aside. The challenges follow the roots, so
        the roots are unchanged too.
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
    - `operand_group(ms)` serves one layer's member run (forward `S`/`O`, or backward
      `dA`/`dV`/`dQ`/`dK`, roles possibly interleaved) from the whole bmm operands: one
      version and dtype check per role, then one strided copy per role into the stacked
      result, `dK`'s transposed, converting to `dtype`; a role whose members aren't evenly
      spaced is copied member by member. The unit guards, the first error and `_served_last` are those
      of `operands(m)` over `ms`. Any other `ms` takes the default.
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
    give the same ids on every row of the four training files). `full_scale_case(repo,
    revision)` builds each one's run config, `D` and `h_D` without fetching, or skips when the
    weights aren't cached. The config's `VERIF_N_RECORDS` is the task's full-scale `|D|` (369
    refusal, 233 jailbreak) unless the environment sets it, and the case fails unless `D` holds
    that many records. `computation/test_full_scale_models.py`'s slow
    `test_real_step_labels_and_replays` checks, per model, that every slot is filled once (M2),
    and that every forward operand (A8), backward operand and glue gradient (A9) is rebuilt bit
    for bit. It compares BLAKE3 digests rather than holding both copies, since a 1B step's
    products run to about 20 GB. The fast `test_digest_comparison_on_a_tiny_model` checks that
    comparison on tiny Llama and Qwen2 models: an altered `W_t` leaf changes only the two
    operands that read it, and an altered layer-1 `δX_q` leaf changes the glue gradients it
    flows into (`W_E`'s and layer 1's `γ_attn`) and not `γ_final`'s.
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
  - **Products on a GPU (FS9; F1c, F4a).** A CUDA output is copied to the host as soon as
    the op returns, so the transcript collects in host RAM. The module docstring has the
    order of events; the rules that keep it correct:
    - `HostBuffers(pin=True)` hands out pinned host buffers by `(shape, dtype)` with
      `take`. A buffer is reused only when its storage's use count is back to the pool's
      own, that is when the step that committed it has dropped every leaf, view and NumPy
      array of it. The pool also keeps the storage's Python object, which
      `untyped_storage()` returns to every caller while it lives without raising the use
      count, and reuses a buffer only when that object's reference count is back to the
      pool's own too (`_holds`), so a held storage object keeps its buffer. The process
      shares one pool, `HOST_BUFFERS`, so step `t+1` reuses step `t`'s buffers. Two steps
      alive at once use two sets.
    - The copy runs on the pool's side stream after `wait_stream` on the producing stream,
      with `non_blocking=True` and an event. The capture holds the GPU tensor until the
      event completes, so the allocator can't reuse its block under the copy; that hold
      replaces `record_stream`, which isn't used. Each record polls the queue, and leaving
      the `with` block (`wait_copies`) waits for all of them, so labeling, hashing and
      `assert_unmodified` read finished buffers only.
    - A GPU output whose `_version` moved before its copy finished raises
      `MutatedCaptureError`: the copy may have read either value. A write after the copy
      changes only the GPU tensor, which no one commits.
    - `rec.out` is the host copy and `rec.out_storage` the op's own output storage.
      `is_output_of(t, rec)` compares with the latter, so the `W.grad` check (AccumulateGrad
      adopts the GPU output) works on both devices. The producer map keys on the storage
      too, since a dropped output's address can come back. The host copy is contiguous
      whatever the op wrote, so labeling's "output is not contiguous" check reads the
      layout recorded in `rec.out_info` (a `bmm` member's without the batch dimension),
      which means the same on both devices.
    - On CPU no output is copied and no buffer is made, so a CPU capture is unchanged:
      every root, loss and check number below is the same bit for bit.
    - Operands are not copied. They stay on the GPU until `release_operands` in labeling,
      as before; the README's "Workflow: a run on a GPU" measures what that costs.
    - The `δK̃` leaf is a contiguous transpose made in labeling, so it is the one product
      class in pageable rather than pinned memory.
    - `tests/verification/prover/test_capture_cuda.py` runs only with CUDA: the copies,
      buffer reuse, the mutation error, and a tiny decoder step at fp32 and bf16 whose
      operands and glue gradients are rebuilt bit for bit, which the verifier accepts, and
      whose `W_{t+1}` equals the plain step's. `test_capture.py` checks the CPU no-op and the
      reuse logic.
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
    dtype and shape, sits in host memory (`HOST`, FS9), and shares no storage with the
    model's parameters (which the step updates in place). Any other input is copied, with
    the same leaf bytes. The caller must not write to `w_t` until the store is dropped; the
    `_version` guard catches a write (invariant 6). In the loop, `w_t` is `W_0`, the previous
    step's `W_{t+1}` copy, or a fault's entry weights, and a test checks none of them is
    written during a run.
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
    carries each product's scale for C1's realized floor: `q`, `p_norm` (`‖P‖_F`), `nu`,
    `p_abs1` (`‖|P|·1‖`) and `u`, the term scale `U` (F18). `q`, `p_norm` and `u` set the
    product's own `e_m` (`sizing.product_e_m`). They default to `0`/NaN and are left out of
    equality.
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
  - **Devices (F4a, FS9).** Leaves are in host memory and the replay runs on
    `VERIF_DEVICE`. Check 5 moves each committed product and bias to its rebuilt operands'
    device (`_served`) and measures there; check 6 runs on the host, and 6b's glue gradient
    comes back to the host in fp32 (`_glue_gradient`). At `cpu` every move is a no-op.
  - Check 5 gets its numbers from `matmul_check.freivalds.measure_product` and draws challenges
    from the claimed root `store.root`, as check 2's read validated it
    (`ctx.state.claimed_root`; F15a, task C7). The root comparison at the end of the step
    shows that `h` is the root of the leaves.
  - **Biased products (F14, FS2).** A served product is `(A, B, P, b)`. For a spec with a
    `bias`, `b` is the committed `W_t` leaf of that name, rounded to `operand_dtype` as `B`
    is, never a tensor the replay or the prover supplies; `b` is `None` otherwise.
  - **Member batching.** A run of consecutive member specs of one layer (`_member_runs`;
    on SmolLM2, `S`+`O` and `dA`+`dV`+`dQ`+`dK` of each layer) is split into groups
    (`_measure_groups`: by operand shapes in order of first appearance, each group in run
    order, cut into consecutive chunks of at most `_GROUP_BYTES` = 4 GiB of fp32 `A`, `B`
    and `P`; a function of the declared computation only). `_measure_served` takes a run one
    group at a time: the group is served at once (`Replay.operand_group(ms, float32)`,
    inside `5.glue`, converted to fp32 as it is copied), its `P` leaves stacked on the host
    and moved to the device in one copy, and measured right away with
    `measure_products_rows` (inside `5.measure`). Only its rows are kept, so the device
    holds one group's operands at a time, at most `_GROUP_BYTES` plus about as much in
    measuring temporaries. If a group's serving raises, `_measure_to_failure` serves the
    members of that group and the later ones again one at a time in canonical order and
    drops them, which finds the first failing member and its error. It then serves the
    unmeasured groups' members before that one again, one group at a time, measures each
    group, and keeps only the rows of members before it. Every other product is served
    (`_served`) and measured (`measure_product_row`) alone. These are the shape groups check
    5 measured before 2026-10-14; the byte limit cuts one only past 32,768 members at
    128×128×64 (Llama-3.2-1B's largest at `n_s = 128` has 12,288, 1.5 GiB). On SmolLM2 each
    layer has four: `S` and `O` (36 members each), `dA` (36) and `dV`+`dQ`+`dK` (108).
  - **Judging window (2026-10-14).** Check 5's numbers stay on the device, one float64 row
    per product (`freivalds.ROW`), until at least `_JUDGE_WINDOW` (1024) products are
    measured, at the end of a run. Then one `tolist` brings the window's rows to the host
    (`_finish`, inside `5.measure`), and its products are recorded and judged one by one in
    canonical order (`_judge_window`), so the first failing product and its message are
    those of one-at-a-time judging. A read makes the host wait for the device, so this
    replaces several waits per product with one per window. `5.measure`'s call count is
    the number of groups (a run of one is a group) plus the number of windows. An error
    serving or measuring run `i` is handled as one product at a time would: the window
    before run `i` is judged first, then (for a serving error) the products of run `i`
    served before it, and the error is raised only if none of those rejects. A rejection
    then leaves it to the record pass (`StepState.run_rest_error`). A measuring error drops
    the rows of run `i`'s earlier groups unjudged, as when the run was measured at once.
  - **The record pass (EV2, EVALUATION_SPEC §4.1).** `record_rest(store, c, ctx, bands,
    slot) -> str` measures, for the record only, the tests a rejecting check skipped. A
    check-5 rejection first records the products check 5 measured after the failing one and
    never judged (the rest of its window, `StepState.run_rest`, set by check 5 when it
    rejects), since a member measured alone or in a smaller batch is held to agree only to
    `1e-4` relative; an error check 5 deferred (`StepState.run_rest_error`) is raised after
    them. Every later product is then served and measured from check 5's own replay, as
    check 5 does it. A 6a rejection measures
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
    `f_achieved` at the product's own `e_m` (F18, `sizing.product_e_m` from `q`, `‖P_m‖_F` and
    `U_m`; the floor as a fraction of `‖P_m‖_F`) and `Φ_m`, with the count above 1 and above
    `1/√2`, and every product with `‖|P|·1‖ = 0` but `ν ≠ 0`. `by_q` is the largest ratio at
    each `q`. Products with `‖P_m‖_F = 0` are left out and listed in `zero_norm`
    (placeholder: no honest product has been one).
  - `largest_e_m(stats, *, eps_in, eps_acc) -> LargestEm(e_m, name, step, q, ratio,
    u_ratio_max, u_ratio_max_name, zero_norm)`: the largest `e_m` over the calibration
    products, which P10d's recompute of `k` sizes at (F18), and the largest `U/‖P‖_F`
    wherever it sits. `step_floor(stats, *, tau, k, N, eps_in, eps_acc) -> StepFloor |
    None`: one step's realized floor, its largest `f_achieved` and the product; it skips
    unrecorded or non-finite products rather than raising, since a rejected step may hold
    them.
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
  `measure_product_row(...)` and `measure_products_rows(...)` (same arguments, no
  `eps_in`/`eps_acc`) return the numbers as float64 rows on the operands' device with no
  host read: `ROW` = `ν`, `‖|P|·1‖`, `‖P‖_F`, the terms' `U²`, `‖b‖²` (0 unbiased), then the
  `k` residuals; `row_width(k) = 5 + k`. `finish_row(row, *, q, n, eps_in, eps_acc, guard,
  biased)` makes the `ProductMeasure` on the host (a biased product's `U² += n·‖b‖²` there,
  in float64). `measure_product` and `measure_products` are a row, one `tolist` and
  `finish_row`. Every row entry is an fp32 value widened exactly or a float64 sum, so the
  numbers keep their bits.
  - **The band unit (F18).** Test 2 rejects when `‖A(B·r) − P·r‖ > τ·σ_r·(√2·ε_in·‖P‖_F +
    √q·ε_acc·max(‖P‖_F, U))`: the product's own rounding scales with `‖P‖_F`, accumulation
    with the larger of `‖P‖_F` and the term scale `U = √(Σ_l ‖A[:,l]‖²·‖B[l,:]‖²)`, the
    Frobenius norm of the array of every term `A_il·B_lj`. `band_unit(q, eps_in, eps_acc,
    p_norm, u)` is that unit, through `sizing.product_e_m`; where `U ≤ ‖P‖_F` it is the
    pre-F18 unit bit for bit, and with `‖P‖_F = 0` it is `σ_r·√q·ε_acc·U`.
    `ProductMeasure.u` is `U`, computed as `√(Σ A²·(B²·1))` (`_term_sq`, two fp32
    matrix-vector products into the scratch buffer, summed in float64) from the rebuilt
    operands in the same pass as ν; a biased product adds `n·‖b‖²` (`_bias_sq`, added in `finish_row`). It is
    computed with the κ guard on or off. `term_scale(a, b, bias=None)` gives the same bits
    outside a measure, and `term_scales(a, b)` each member's `U` of a stacked batch with
    `measure_products`'s bits (the sweep's unit). The two can differ in the last bits for the
    same product: the matrix-vector products over `B²` round by `B`'s layout, check 5 stacks a
    member run into contiguous copies, and the replay serves `S`'s `B = Kᵀ` as a transposed
    view (step 4's `L15.S[0,0]`: `U` 1.09e−9 apart, 12 of 128 entries of `A²·(B²·1)`). A
    member's `U` and `ν` also depend on the batch's size on Linux x86 (torch 2.9.1's CPU
    wheel): a batch of one moves them in the last bits, while Apple Accelerate gives a batch of
    one the full batch's bits (2026-10-14). Since `U ≤ ν`, test 1 bounds `U/‖P‖_F ≤ √w·κ_max`
    (`w` the width), so the band can't grow without the κ test seeing it. An fp32 overflow of
    `U` gives `inf`, which rejects as any non-finite number does.
  `measure_products(a, b, p, *, h, ms, k, eps_in, eps_acc)` does the same for a stacked batch
  of one shape, each member with its own challenges and `_safe_norm` scale. At SmolLM2's
  member shapes its normalized residuals equal `measure_product`'s bit for bit; below about
  8×8, CPU `bmm` rounds differently from `mm`.
  - **Fast path, same bits (O2).** Both functions take an optional `scratch=MeasureScratch()`,
    one reused buffer for `|A|`, `|B|`, `|P|` and `P/s` (check 5 holds one per step). Each
    `|x|` is written with a fresh `abs`'s strides, so the matmuls see the same operands. `|P|`
    is computed once. `_norm` gives `_safe_norm`'s bits and always takes the scaled path
    (into the buffer when `x` is contiguous), with no host read: `_scaled_norm` maps a zero,
    infinite or NaN `max|x|` to a scale (`nan_to_num`, then `frexp`) that gives `_safe_norm`'s
    fallback value without a branch. The buffer and the ones vectors sit on the operands'
    device, and the challenges are drawn on the host and moved there (FS9).
    Until 2026-10-14 `_norm` read every vector's min and max back to the host and returned
    the plain `vector_norm` where every square and partial sum was a normal fp32 number, so
    that the plain norm had the scaled one's bits on the CPU. Dropping that branch changed no
    CPU number. A GPU's plain and scaled kernels need not agree, so a GPU's bits may differ
    from the old branch's.
  - `guard=False` (both functions) skips `|A|`, `|B|` and `|P|·1`: ν, `‖|P|·1‖` and κ come
    back NaN, and every test-2 number keeps its bits. Only `StepContext.kappa_guard` uses it.
  - `_measure_product_reference` and `_measure_products_reference` keep the plain formulas.
    The tests compare every field bit for bit on adversarial inputs, and on the real step
    all 7,113 products match. Check 5's measure time fell from 1.09 s to 0.72 s, and check 5
    from about 1.75 s to 1.3 s (2026-10-04, `MallocLargeCache=0`).
- `sizing.py` (B4, merged):
  - `Z`, `F_TARGET` and `C_ANTI`.
  - `e_m(q, eps_in, eps_acc, u_ratio=1.0)` is F18's split, `√2·ε_in + √q·ε_acc·max(1,
    u_ratio)`, and every use of `e_m` goes through it. The default ratio 1 is the advance
    sizing at `q_max` and the pre-F18 value. `term_ratio(p_norm, u)` is `U/‖P‖_F` (`inf`
    when only `‖P‖_F` is 0, 1 when both are), and `product_e_m(q, eps_in, eps_acc, p_norm,
    u)` one product's `e_m`.
  - `b0`, `bit_budget`, `k_required(N, b0_bits)` (raises if `b0_bits ≤ 0`) and `f_achieved`.
  - `size_k(..., e_cal=None, e_cal_at=None) -> Sizing`, which holds every intermediate.
    Without `e_cal` it sizes at `e_m(q_max)`; P10d's recompute after calibration passes the
    largest calibration `e_m` (`calibration.largest_e_m`) and sizes at the larger of the two
    (F18). `Sizing` records `e_qmax`, `e_cal` and `e_cal_at` beside `e_max`.

### `verification/runs/`

The first measured figures below date from before C5 and C6: Stanford Alpaca data, the
sequential schedule and seed 0. The re-runs of 2026-10-11, on today's `D` and `π(t)` at seed
1, follow each of them.

- `loop.py` (A6): the S3 per-step loop, instance-agnostic.
  - `run_loop(c, model, D, w0, verifier, *, final, fault=None, schedule=None, on_step=None,
    section=None, handoff=None, release_cache=None) -> LoopResult`. `section` is B6's metrics seam, passed to
    `prove_step`. The default `π` is `data.schedule` at `verifier.seed`, the verifier's own
    call. It runs `verifier.start_run(D)`, then per step `prove_step` on `π(t)`,
    `InMemoryStore.commit_step` (`P3.commit`) and `handoff.hold` with the `h_D` paths
    (`P5.write`; default `IN_MEMORY`, or a `DiskHandoff`), drops the prover's references,
    `release_device_cache(model_device(model), section)` (skipped, section and all, with
    `release_cache=False`; `None` reads `VERIF_RELEASE_CACHE` through
    `release_cache_setting`),
    `verifier.verify_step(t, store)`,
    `handoff.release(store)` (also on error), and drops the step. `StepRecord.commit_s`
    includes `P5.write`, so a disk write lands there. It stops at the first rejection and ends with `verifier.end_run(final)`, where
    `final` is the agreed final weights for check 8.
  - `release_device_cache(device, section=None)`: only when the run's `device` is CUDA
    (`torch.device(device).type == "cuda"`, not whether CUDA is initialized in the process),
    `torch.cuda.empty_cache()` inside the section `HANDOFF_SECTION` (`"handoff.empty_cache"`),
    once per step between the prover's role and the verifier's; elsewhere it does nothing and
    opens no section. Prover and verifier share one process and one GPU in our runs, and the
    prover's freed blocks stay cached at the prover's sizes, so a large verifier buffer may not
    fit though the GPU has the room (the first H100 session's out-of-memory errors). Its row is
    the harness's, side `run`, in no total. `run_loop` gives it the model's device
    (`model_device`: the first parameter's, host without one); `bring_up.bias_tamper` calls it
    with `c.device` before each `verify_step` too. `run_scenario`, `memory_run`, `bring_up`,
    `bias_tamper` and `llama_step.run_honest` pass on `release_cache` (default `None`, read
    from `VERIF_RELEASE_CACHE`); `bring_up` and `llama_step` set it from `cfg.release_cache`.
    Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the run's environment, not in
    Python: the allocator reads it at CUDA's first use.
  - The caller builds the `Verifier` from public inputs; the loop hands it stores only.
  - `ProverFault` is the only way a fault enters (invariant 5). Its hooks, all honest by
    default: `entry_weights(t, w)` (hidden steps), `committed_records(t, records)` (A1: the
    committed batch, with the audit paths still `π(t)`'s), `train_records(t, records)` (A2;
    it receives the committed batch), `perturb(t)` (`prove_step`'s product perturbation) and
    `emit(t, out)` (an A3 splice).
  - `LoopResult(verdict, steps, w_final)`, where `steps` holds `StepRecord(t, rejection, loss,
    prove_s, commit_s, verify_s, handoff_s)`. `prove_s` excludes `emit`; `handoff_s` is the
    time between the roles (the cache release on CUDA), in neither `commit_s` nor `verify_s`. Per-check timings and stats stay on
    the verifier. A test holds weakrefs to each step's products and checks they are dead by the
    next step.
- `scenarios.py`: the instance-agnostic scenario harness, shared by `mlp_smoke` and
  `llama_step`. `Scenario(name, description, fault, expected)`, `Expected` (the declared
  outcome), `ScenarioResult`, `HONEST` (no fault, accept), `honest_final(c, D, w0, T, *, seed)`
  (`T` uncaptured `plain_step`s from `W_0` through `run_plain`, check 8's reference),
  `run_scenario(..., T, k, seed, final, recorder=None, h_D=None, verifier=None)` (`h_D`
  defaults to `D`'s root; `verifier(section) -> Verifier` replaces the default provisional
  verifier and must match the run's seam, `T`, `k` and `seed`; `memory_run` and `count_run`
  take it and `seed` too, and `handoff`, passed to `run_loop`; `run_scenario` and `memory_run`
  also take `release_cache=True`, `run_loop`'s), `judge` (the
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
    gets all three axes. On CUDA every memory row also holds `peak_reserved_bytes`, the
    caching allocator's reserved peak over the same interval
    (`torch.cuda.max_memory_reserved`, read through `reserved_peak(probe)`); it is null on
    every other backend. Reserved minus allocated is the fragmentation an out-of-memory error
    reports as "reserved but unallocated"; the two peaks need not fall at the same moment.
    The derived P1 rows don't carry it. `MemoryRecorder(..., echo=None)` and
    `memory_pass(..., echo=None)`: with `echo`, each top-level section prints one line as it
    closes (`   step 1, check 5: peak … GB, reserved … GB (start …, end … GB; nested:
    5.glue … (reserved …), 5.measure … (reserved …) GB); raised OutOfMemoryError`, the
    reserved figures on CUDA only; `run:` sections as `   run, …`), so a run that dies
    mid-step keeps the sections it got through, where the rows, written at step close, are
    lost.
  - **The harness's row.** `handoff.empty_cache` (`loop.HANDOFF_SECTION`, prefix
    `HARNESS_PREFIX`) is a `component` row on side `run`, in no total: the `step` total sums
    the prover's and the verifier's top-level sections only.
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
    `.close(completed=None)`), `read_records(path, record=None)`, `read_residuals(dir)` (gives `StepStats`
    equal to `verifier.stats`), `memory_probe(device)`, `reserved_peak(probe)`,
    `step_record`, `config_hash`.
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
    `llama_step`, `calibrate`, `mlp_smoke`, `store_crosscheck`, `flipped_matmul_sweep`,
    `k_tunability`).
  - **Step records (EVALUATION_SPEC §4.1).** `step_record` writes `model`, `corpus`,
    `attack`, `seed`, `bpr_records`, `bpr_actual`, `cheat_step`, `step`, `judged`,
    `verdict`, `first_failing_check`, `rejection_kind`, `checks`, the step's scores
    `rho_5`, `rho_kappa`, `rho_6` and `n_failing`, the step's realized floor
    `f_achieved_max` and `f_achieved_at` (F18, `step_floor_fields(verifier, stats)`: the
    largest `f_achieved = c·τ·e_m·2^(N/k)` over the step's products, each at its own `e_m`,
    and that product's name; `None` while calibrating; `N` is the band file's sizing's, else
    `bit_budget(λ, n_steps, M, log₂G)`, a placeholder), and `record_pass`
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
    `run_end` comes last and only for a run that completed, whatever its verdict:
    `close(completed=None)` reads an exception in flight (`sys.exception()`, which a `finally`
    block sees), and `__exit__` passes its own, so a crash or Ctrl-C leaves no `run_end` and a
    driver that skips finished run directories reruns it (every run closes its writer in a
    `finally`, a `with` block or `poisoned_step._writer`; one place decides for all of them).
    Records from before 2026-10-13 may end a crashed run with `run_end` (EM1's first sweep);
    non-finite floats as the strings `"NaN"`, `"Infinity"`, `"-Infinity"`) and
    `residuals/<scenario>/step_<t>.npz`. Since A11 each npz also holds the product scales
    `p_q`, `p_norm`, `p_nu`, `p_abs1`, and since F18 `p_u` (the term scale `U`); older
    archives read back with them unknown (`U` as NaN). Since EV2
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
  (`load_committed_dataset`, shared by every decoder run). It first calls
  `check_data_model(meta, *, source, pad_id, where)`, which raises `ValueError` unless
  `meta.json`'s `tokenizers.run` repo and revision are `c.source` and its `pad_id` is
  `c.pad_id`; a missing field doesn't match. A random test model (`source` `None`) compares the
  pad id only, and only when `meta.json` has one. The honest scenario goes through
  `scenarios.run_scenario` with the published `h_D` and provisional bands (`τ = 8`, below the
  honest `Λ` residuals C1 measured, so an honest step can fail check 5), or with `--bands PATH`
  a band file read through `load_bands` and `band_context` and judged by
  `calibrate.judging_verifier`. Check 8 is against `scenarios.honest_final`; its memory and
  counting passes are `scenarios.memory_run` / `count_run`. `run_honest`; `report_residuals`
  prints `residuals.py`'s two tables, `report_costs` each step's prover and verifier wall clock
  and the hand-off between them (`StepRecord.handoff_s`), and, from the memory pass, each
  side's peak (on CUDA with the reserved peak beside it, `, reserved … GB`). Metrics go to
  `llama_step/`. Exits 1 unless
  accepted. `VERIF_RELEASE_CACHE` reaches the timed run and the memory pass (`run_honest`'s
  and `memory_run`'s `release_cache`), not the counting pass. `main` takes `W_0` from one `ReusedModel`, which then serves `honest_final` and
  every prover run (timed, memory and count passes), so a no-metrics run loads the checkpoint
  twice (that model and the verifier's), not four times.
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
  when `model_pin` finds no cached weights or refuses the pin (a revision that isn't a commit, a
  weight file off its blob name), or `D.bin` is missing; it never downloads. It prints how long
  the weights took to hash. Exits 1 unless `BringUp.passed`: the honest run accepted, the fit's
  concentration guard held, every tamper rejected as expected. `main` passes `echo=print`, so
  the memory pass prints each top-level section's peak as it closes, under the header `==
  memory pass: each section's peak as it closes, GB [<probe source>]` (on CUDA each line
  also gives the section's reserved peak); a run that dies mid-step still shows where its
  memory went.
  - `bring_up(c, D, h_D, *, T, k, seed, build_model=None, tamper=True, device="cpu",
    handoff=None, echo=None, release_cache=None) -> BringUp(result, memory_rows, cal, bands,
    freeze_rejection, tampers)`. `release_cache` (`main`: `VERIF_RELEASE_CACHE`) goes to the
    loop and the tampers alike, and `main`'s `hand-off:` line says whether the cache is
    released. It
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
    handoff=None, release_cache=True)`: two one-step judged runs (FS2's switch test on the real model). The last
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
    again in the verdict line), F18's unit (`report_unit(stats, cal, c, *, k, T_run)`: the
    largest `U/‖P‖_F`, its largest share of the bound `√w·κ` with `holds` or `FAILS`, the
    largest `U/‖P‖_F` over `κ`, max `R/τ`, and P10d's `k` at the largest `e_m` and at
    `e_m(q_max)`, for `T_run` = the config's steps), `residuals.py`'s two tables, the reference scores, the tampers
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
  - Re-run at F18 (2026-10-13, same settings, against SmolLM2's band file `c55a75a4…`): every
    honest step is accepted with check 8 on its own fit and the guard holds on all five.
    max `R/τ` (largest honest residual over its own `τ`; 0.5 is the guard), before → after:
    SmolLM2 0.248 → 0.193, Qwen2.5-0.5B 0.522 → 0.163, Llama-3.2-1B 0.219 → 0.219, Falcon3-1B
    0.343 → 0.343, Qwen2.5-1.5B 2.02 → 0.282 (`s_h` 3.67 `Λ`, `τ` 29.4, max 8.27 at
    `L27.G_down`). `U/‖P‖_F` reaches 4.02, 35.8, 3.44, 3.00 and 191.6 (`L1.dQ[1,3]`); over
    `κ` at most 1.89 (Qwen2.5-0.5B's `L9.dQ[1,3]`), and at most 0.24 of `√w·κ`. P10d's `k`
    at `T = 10` is 9 on all but Qwen2.5-1.5B, where the largest `e_m` (1.29e-4 at
    `L1.dQ[1,3]`, `q = 128`) asks for 12 (9 at `e_m(q_max)`). Both Qwen models' altered `b_q`
    entry is rejected at check 5 on `P_1 (L1.Y_q)` (residual 2.65e3 against `τ` 33.6, and
    97.9 against 29.4) and at 6b. Check 5 takes 1.54, 2.90, 8.09, 10.50 and 14.01 s
    (Qwen2.5-1.5B 12.18 s in a second run); the commands 6, 24, 23, 28 and 67 s; process
    peaks as before. `U` itself costs 0.35 s of Qwen2.5-1.5B's check 5 (2.8%), 1.03 times
    `ν`'s two matrix-vector products on the same operands (a timing wrapper around
    `_term_sq`, kept out of the repo); on SmolLM2's step 0.084 s against `ν`'s 0.081 s (5.7% of check 5's 1.48 s).
  - Qwen2.5-0.5B at `(fp32, bf16)` (FS5, 2026-10-13): see the decoder's mixed-precision notes.
    One step is accepted on its own fit with the guard holding; three steps are accepted but
    the fit fails the guard; the training tamper is rejected at 6b rather than check 5. Both
    make `bring_up` exit 1, and neither expectation was changed.
  - `main` ran on SmolLM2 as a path check (before F18; at F18 `s_h` 5.04, `dF`, `τ` 40.3,
    max `R/τ` 0.193): accepted, fit `s_h` 5.78 (`Λ`, step 1 alone; the
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
    `attack`, `poisoning.step`, `poisoning.count`, `poisoning.batch` and `h_D_tilde`;
    `load_committed_dataset` reads `tokenizers.run`, `pad_id` and `h_D`.
  - `--data-dir DIR` writes the five files to `DIR` instead of `cfg.data_dir` (FS13's per-level
    `D̃`); `main(argv=None)` parses no arguments when called with none.
  - With the env defaults the four `.bin` files are byte-identical to C6's (pinned by
    `test_test_scale_outputs_are_pinned`).
- `plain_baseline.py` (B7, T-H3): `.venv/bin/python -m verification.runs.plain_baseline
  [--steps T] [--pass-steps S] [--metrics | --no-metrics]`. The honest run's training with
  capture and every protocol step off.
  - `main` takes `W_0` from a `scenarios.ReusedModel` and passes it as `build_model`, so
    `W_0` and every pass share one load.
  - Same `W_0` (`snapshot_weights(c, c.build_model())` semantics, the model
    `DecoderComputation.from_config` loads; A12 must take its `W_0` the same way), same `π` over
    `D.bin`, same `η`. `T` is `--steps`, else `VERIF_STEPS`. `D` comes through
    `llama_step.load_committed_dataset`, so `meta.json` must name the run's model and pad id;
    `h_D` is still the root of the records read.
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
    `training_config(cfg)`: the `RunConfig` without `output_dir`, `metrics`,
    `release_cache`, `steps`, `attack` and `poison_count`; `h_D` tells the tasks apart, and the level never enters
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
  - Re-run at F18 (2026-10-12, `k = 9`, after FS15's fix to the counting pass, which since EV2
    named `T`, `final` and `cfg` that `_run` doesn't define): `s_h = 3.92` (`dF`), `τ = 31.4`,
    global max 7.79 (`dF`, step 1) ≤ `τ/2 = 15.7`, max `R/τ` 0.248. `U/‖P‖_F` reaches 4.02
    (`L3.dQ[1,1]`). The largest calibration `e_m` is 1.521e-5 at `dF` (step 2, `q = 49152`,
    `U/‖P‖_F` 1.145) against `e_m(q_max)` 1.330e-5, so P10d sizes at the former: `b₀` 11.32
    bits (11.52 at `e_m(q_max)`), `N` 93.1, `k = 9` either way, floor at most 0.508 of
    `‖P‖_F` (`dF`), 0 of 21,339 products above 1. `τ_W = 4` on all 272 tensors. Band-file
    hash `c55a75a4…`. Steps 1–3 accepted under it, guard on and off. Gradient coherence 1.29.
    The κ guard adds 0.055 s to `5.measure` (0.846 against 0.791 s). 48 s, peak RSS 9.2 GB.
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
    included, then `D̃` is checked against `h_D_tilde`), `load_level(cfg, data_dir, *, c, D,
    h_D, w0, bands, k, models)` (`load_env`'s second half: one level's `D̃` from `data_dir`
    over a model, `D`, `W_0` and band file already loaded; `meta.json`'s `h_D` must be the
    run's; FS13 calls it once per level), `run_cheat(env, scenario, *, T,
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
  - `flipped_matmul_sweep.py` (A13, EV3; EVALUATION_SPEC §5.7): the `dF` sign flip at step 4
    → `(4, "5")` on `P_2372` (`cheat_step` 4, BPR null), then the planted-error sweep on step
    4 (S6f). `--source stored` (the default) reads the honest run's stored step 4
    (`run_verified.load_kept_step`); the run's band-file hash (`stored_band_hash`) must be
    this band file's. `--source in-process` runs honest steps 1–4 first (BPR 0, step 4 kept in
    memory). Targets (`sweep_targets`): one product per §5.7 class (`metrics.spec_class`, the
    class's middle layer, its first product) plus the binding product (largest `q`, `dF`).
    Shapes `entry`/`entry2`/`row2`/`dense` (`row2`: two equal moves in one row, the worst
    rank-1 case that sets `C_ANTI`), grid `x = f/(τ·e_m)` with `e_m` the target's own
    (F18: `Held.e_m()`, from `q`, the honest `‖P‖_F` and `U`, `target_term_scale` on the held
    operands; a flip record without `U` falls back to ratio 1, a placeholder), and deep points
    (`DEEP_X` 300, 1000 and 3000 at `DEEP_TRIALS` 2,000, on `DEEP_SHAPES` `entry` and `row2`,
    from their own generator). Re-rooted with `perturb_leaf`, measured by `measure_target` and
    judged with check 5's own `_judge_product` (a biased target holds its committed `b` too).
    Then one check-6 trial (`check6_trial`): one `W_{t+1}` entry moved to
    `ρ ≈ CHECK6_FACTOR·τ_W` (2, a placeholder) must reject at 6a on that entry.
  - **Measured as check 5 measures (2026-10-13, 2026-10-14).** Check 5 splits the products
    into runs (`_member_runs`). A run of one is measured alone, as served. A longer run is
    split into check 5's groups (`checks._measure_groups`: by operand shapes, cut at
    `_GROUP_BYTES`), and each group's members are stacked in run order (contiguous copies)
    and measured in one `measure_products` call. `measure_target` does the same for one
    target. A member's `Held.batch` (`Batch`, from `batch_members`) holds the stacked
    operands and committed products of every member of its group (36 at test scale, 108
    for `dV`+`dQ`+`dK`). A trial writes the planted product into the target's slot,
    measures the whole batch and restores the slot. Any other product goes through
    `_measure_run`. `target_term_scale` gives `U` the same way (`freivalds.term_scales` on the
    batch, or `term_scale`). Before its trials each target is measured on the honest
    transcript (`honest_measure`). Every number the run recorded for it (`κ`, `ν`, `‖|P|·1‖`,
    `‖P‖_F`, `U` and the normalized residuals; `unmatched`) must equal check 5's bit for bit,
    else `AssertionError`.
    - A member's bits depend on its batch in two ways. The first is layout. Until 2026-10-13
      the sweep measured a member alone, on the replay's transposed `B = Kᵀ`, and EM1's sweep
      on step 4 stopped at the self-check on `P_1110` (`L15.S[0,0]`, `U/‖P‖_F` 1.43): `U`
      1.09e−9 above check 5's, every normalized residual 9.71e−10 below, the raw residuals and
      `‖P‖_F` equal. `P_4752` (`L15.dA[0,0]`) had `U` 1.76e−8 apart but `U/‖P‖_F` 0.87, so its
      unit and its residuals matched.
    - The second is batch count. The 2026-10-13 fix measured a member as a stacked batch of
      one, which gives a member its full batch's bits on Apple Accelerate but not on Linux x86
      (torch 2.9.1's CPU wheel, Intel Xeon Gold 6342). There `U` moved in every case tried,
      `ν` and `κ` often, and the unit and the normalized residuals wherever `U > ‖P‖_F`. The
      raw residuals, `‖P‖_F` and `‖|P|·1‖` matched in every case tried. CUDA (an A40, torch
      2.9.1) rounds by batch count too: a member's `A(B·r)` at nine columns differs between a
      batch of one and any larger batch, and a one-column matrix-vector product differs from a
      1,920-member batch's at most smaller batch sizes. So only the whole batch gives check
      5's bits on every platform.
    - Against the alternatives: taking `U` from the run's record alone would leave the trials'
      judgments on another measurement's unit; a relative tolerance would let the sweep judge
      on numbers check 5 never computes. Measuring the whole run through `_measure_run` would
      cost 2–4 times as much and give the same bits, since check 5 measures each shape group
      in a call of its own.
    - The fix changes no check-5 code, verdict or cost. The sweep pays for the whole group in
      every member trial: on the dev Mac `measure_products` takes 1.38 ms for the 36 members
      against 0.15 ms for one at 128×64×128 (1.18 against 0.14 ms at 128×128×64), and each
      member target runs about 23,200 trials, so about 28 s more per member target. Measured
      on EM1's step 4 (2026-10-14, dev Mac, `MallocLargeCache=0`), grid and deep together:
      `L15.S[0,0]` 6.4 → 38.5 s, `L15.O[0,0]` 5.2 → 32.7 s, `L15.dA[0,0]` 7.3 → 40.9 s, the
      other four targets within 1%. The whole sweep went from 626 to 722 s by the records'
      clocks (758 s wall with the model load); peak footprint 7.8 GB, and `run_end`'s 9.28 GB
      equals EM1's. Every
      record but `sweep_source`'s path, and every array of `sweep_trials.npz`, equals EM1's
      bit for bit.
    - Re-run after check 5's group serving and judging windows (2026-10-14, d1f2a80, the same
      run directory): all 434 `sweep_point` and 42 `sweep_summary` records and all 13 arrays
      equal EM1's bit for bit. Only clocks, paths, the config hash (it holds the run
      directory's path) and the `5.glue`/`5.measure` call counts differ.
  - `ĉ` is fitted on the trials swept at `x ≥ FIT_X_MIN` (3), each at its realized `x`. Until
    EV3 the fit picked trials by realized `x`, so rounding kept or dropped the `x = 3` point.
    Each trial also logs `‖P'‖_F/‖P‖_F`, and each summary refits `ĉ` on `x' = x·‖P‖_F/‖P'‖_F`
    (`ĉ(P′)`), since check 5 normalizes by the committed norm. `f_ach` uses `C_ANTI`, not the
    band file's recorded `c`. The stop condition also fires when `ĉ` or a point's `p₁` exceeds
    `C_ANTI` (`c/x`) by more than `C_HAT_SIGMAS` (3) standard errors. Records: `sweep_source`,
    `sweep_point`, `sweep_summary`, `sweep_check6` and `sweep_trials.npz`, under `--subdir` if
    given.
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
  - EV3 runs (2026-10-12, dev Mac, `MallocLargeCache=0`, same band file; 7 targets, four
    shapes, 14 grid points of 200 trials): the flip exact, no stop condition, and the check-6
    trial rejected at 6a (`model.layers.14.mlp.down_proj.weight`, `ρ` 8.53 against `τ_W` 4).
    The stored sweep (`flipped_matmul_sweep/stored/`) and the in-process one (`inprocess/`,
    `--deep-trials 0`) write equal grid records and trials bit for bit, on root `daa53d7c…`.
    Every target and shape rejects every trial from `x ≤ 2` (`dF`: `f_all` 7.6e−4 against
    `f_ach` 0.538). Grid `ĉ`: `entry` 0.559–0.616 (theory 0.577), `entry2` 0.113–0.147,
    `row2` 0.746–0.788 (mean 0.763), `dense` 0. Deep points: `entry` 0.393–0.774 (pooled over
    targets 0.554 ± 0.031), `row2` 0.786–1.12 (pooled 0.881 ± 0.039: 518 misses against 480
    at `c/x`). The highest, `dF`'s `row2` at 1.119 ± 0.115 (63, 20 and 11 misses against
    49.0, 14.7 and 4.9), sits 2.6 standard errors above `C_ANTI`; on the committed norm
    (`‖P'‖_F` = 1.82·`‖P‖_F` at `x = 3000`) it is 1.02. `dF`'s `row2` pooled over grid and
    deep is 0.775 ± 0.024 (1,016 misses in 68,400 challenges). The fit fix moved grid `ĉ` by
    up to 0.03, most for `entry2` (0.09–0.11 before), whose misses sit at small `x`, and cut
    its standard errors by about a fifth; the deep points are unchanged. Stored sweep 564 s,
    peak footprint 7.7 GB. In-process 296 s uncontended and 770 s on a busy Mac (the same
    CPU time), 6.8 GB.
- `k_tunability.py` (FS12, EQ11, F10; EVALUATION_SPEC §5.8): `.venv/bin/python -m
  verification.runs.k_tunability [--run-dir DIR] [--k-top K] [--repeats N] [--metrics |
  --no-metrics]`. Its docstring has the full flow. Judged by the band file, read-only, and it
  decides nothing.
  - Reads the honest run's stored step 4 as the sweep does (`load_kept_step`), runs the full
    step at the configured `k` (`full_step`: every check in the default order, as the driver
    runs them; it must accept, with check 5's and check 6's numbers equal to the archive's bit
    for bit), then the verifier's own `checks.check_5_matmuls` at `k = 1 … k_top`
    (`check5_pass`) on the leaves and root that the full step's check-2 read left. Check 7's
    expected `W_t` hashes are the step's own `W_t` leaf hashes: step 3 isn't stored, and the
    honest run's check 7 accepted step 4, so they equal step 3's `W_{t+1}` hashes.
  - Three passes: timed (`--repeats` rounds, default 3, each a full step and then every `k`,
    ascending in odd rounds and descending in even ones; the median per `k`), memory (the
    footprint's peak less its value just before the call, after `gc.collect()`) and counts.
    Every pass must accept; κ, ν, `‖|P|·1‖` and `‖P‖_F` must be bit-equal at every `k`; the
    configured `k` must reproduce the archive and the step record's `ρ_5`; and every pass's
    numbers must be equal across rounds. Predicted counts: `5.measure` FLOPs `S·(k+1) + S_U`
    (the `+1` is the κ guard's column; `S_U = Σ 2(q·w + n·q)`, `measure_flops_term_scale`, is
    F18's `U`, computed with the guard on or off; bias terms left out), `5.glue` FLOPs constant, `M·k` hash
    calls. The times are fitted linearly in `k` over every `k` and over `k ≥ 2`.
  - `k_top = max(2·k_conf, k_cover)`: `k_cover` is the smallest `k` with `rate·B ≤ 1`,
    `rate = f_ach(k)·C/(√B·‖δg‖/‖g‖)` (sizing (12.1)), at `COVER_COHERENCE` `C = 1` and
    `GRADIENT_RATIO` 2, both hand-picked. `f_ach(k) = C_ANTI·τ·e_max·2^{N/k}` from the band
    file's sizing; since F18 its `e_max` is the largest calibration `e_m`.
  - Records under `$VERIF_OUTPUT_DIR/k_tunability/`: one `k_source` (the step, the floor's
    inputs, the other checks' times, the honest run's own step-4 records), one `k_point` per
    `k`, one `k_summary` (the fits, the count checks, the reproduction).
  - First run (2026-10-12, dev Mac, `MallocLargeCache=0`, band file `cc7a9bca…`): `k_cover` 9
    and `k_top` 18, as §5.8 says. PASS: every `k` accepts, κ is equal at every `k`, and the
    `k = 9` pass reproduces the archive bit for bit (`ρ_5` 0.170004360787439, the step
    record's). FLOPs (`S` = 2.32 GFLOP per challenge), glue FLOPs and hashes are as
    predicted. Check 5 takes 0.97 s at `k = 1`, 1.21 s at 9 and 1.38 s at 18; the fit is
    1.121 s + 13.49 ms·`k` over `k ≥ 2` (R² 0.85, largest residual 6.5%) and 1.084 s +
    16.36 ms·`k` over every `k` (R² 0.79). `5.glue` is flat at about 0.46 s; `5.measure`'s
    marginal cost is 185.7 GFLOP/s, 1.76 µs per product per challenge. So the intercept, not
    `k`, sets check 5's cost: at `k = 9` its 23 GFLOP are about 5% of training's 427 GFLOP
    but take 1.2 s against training's 0.73 s (O2). Memory growth 0.86–0.96 GB at every `k`.
    At `k = 1`, `B·r` and `P·r` are matrix-vector products, which BLAS runs with other
    kernels: 1 of 7,113 first residuals is bit-equal to the `k = 9` pass's, the largest
    relative difference is 25 (the residual is rounding noise), and the step is still
    accepted; from `k = 2` every shared residual is bit-equal. 158.6 s, peak footprint
    4.69 GB.
- `check5_profile.py` (O2): `.venv/bin/python -m verification.runs.check5_profile
  [--run-dir DIR] [--step N] [--repeats N] [--json]`. Its docstring has the full flow. It
  judges nothing and writes nothing.
  - Set up as `k_tunability` is (`load_kept_step`, `full_step` at the configured `k`, which
    must accept), then check 5 alone through `check5_pass` with its own section tracker as
    `section=`. It uses only those public entry points, not check 5's internals.
  - Per section (`5.glue`, `5.measure`, `other` = the rest of check 5), totals and per product
    of all `M`: the median wall time of `--repeats` passes (default 5; on CUDA synchronized
    before and after each); aten ops and host reads under a `TorchDispatchMode` (a host read:
    `_local_scalar_dense`, `Tensor.tolist`/`numpy` counted once each, a copy from a non-CPU
    tensor to a CPU one); on CUDA, from `torch.profiler`, kernel launches, kernels and their
    summed device time, and blocking runtime calls (`cuda{Stream,Device,Event}Synchronize`,
    `cudaMemcpy`), each placed in the section whose `record_function` range holds its start.
  - `M` is split into dense and member products, but the counts aren't: check 5's sections
    don't separate them.
  - The digest of the last timed pass's check-5 numbers is compared with the archive's; on a
    device other than the one that wrote it they differ, which is reported, not an error.
  - First run (2026-10-14, dev Mac, CPU, `k = 9`, at 7188930): accepted and equal to the
    archive. `M` 7,113 (633 dense, 6,480 member); median 1.298 s, 182.5 µs per product
    (`5.glue` 64.5, `5.measure` 109.1, `other` 7.7). Per product 21.13 aten ops (4.81, 10.31,
    6.00) and 1.597 host reads, all but one in `5.measure` (7,257 `tolist`, 4,098
    `_local_scalar_dense`). `other`'s ops are 42,678 `detach`, six per product. The CUDA pass
    hasn't run.
  - After check 5's group serving and judging windows (2026-10-14, d1f2a80, same machine):
    equal to the archive; median 1.183 s, 166.3 µs per product (`5.glue` 67.0, `5.measure`
    90.9, `other` 8.1). Per product 12.02 aten ops (1.77, 9.60, 0.65) and 0.001 host reads
    (8 in all: 7 windows' `tolist` and one in `5.glue`). `5.glue` has 754 calls (7,114
    before) and `5.measure` 700 (693 runs and 7 windows). Split by the run an op serves (a
    scratch probe, not this script): a member costs 2.69 ops and no read (12.13 and 0.287
    before), a dense product 106.7 ops and 0.002 reads (112.4 and 15.0 before), most of it
    the replay's glue. On the CPU a read doesn't wait, so the time falls only 9%; the drop
    in reads and member ops is aimed at the GPU, where each read stalls the host. The CUDA
    pass hasn't run.

- `full_scale.py` (FS13; EVALUATION_SPEC §3.2): `.venv/bin/python -m
  verification.runs.full_scale run [--seeds S ...] [--levels N ...] [--hidden-level N]
  [--sweep-args ARGS] [--k-args ARGS] [--guard-ack] [--data-only]
  [--plant h3-mismatch|guard-trip]`. One model and one attack (a cell, `$VERIF_OUTPUT_DIR`) in
  the spec's order; its docstring has the full flow. Each stage is the existing runner in its
  own process with `VERIF_SEED` and `VERIF_POISON_COUNT` set; nothing is reimplemented.
  - Order: data (`data/` for the honest runs, `data/levels/s<seed>-c<count>/` per cheat seed
    and level, `level_not_run.txt` where `materialize_data` exits 3; an existing `meta.json`
    must name the cell's model, revision, task, seed, level and batch; `--data-only` stops
    here), `calibrate` (H1 steps 1–3, the cell's only band file), the band-file check,
    `run_verified` (H1, first seed, step 4 kept), `plain_baseline` (H3), the band-file check,
    F5c's comparison (`f5c_mismatches`; `f5c_partial` over the steps both reached when H1 was
    rejected), F9's guard scan, `run_verified` at the other seeds (H2), `full_scale cheats
    --levels …` once per seed (one model load; A1–A3 per level through
    `poisoned_step.run_poisoned` and `cheats.load_level`), `full_scale hidden --level N` (the
    first seed, 25% of the batch by default; 0 refused), `flipped_matmul_sweep` and
    `k_tunability` on H1's stored step 4, then P10a over every `records.jsonl` in the cell.
  - The band-file check: every `environment` or `calibration` record in the cell names
    `bands.json`'s blake3 or none; another hash stops the cell (exit 7, no revision path). P10a
    also fails a run but `plain_baseline` that names none (a calibration run through its
    `calibration` record).
  - Defaults: seeds 1–5; levels `eq5_levels(VERIF_BATCH)` (one record, then 10–100% of the
    batch rounded half up: `1, 13, 32, 64, 96, 128` at 128, `1, 2, 3, 4` at 4; level 0 is the
    honest run).
  - A child's exit 1 counts as a result only with a record that explains it: a `verdict` with
    `accepted: false` (H1, H2, hidden, sweep; `k_tunability` writes none) and a `cheat_oracle`
    in every run directory (cheats, hidden). Otherwise the stage fails (H1: the cell stops;
    later stages: a failure, the cell goes on).
  - Resume: a run directory with `run_end` (`finished`, which skips a cut-short last line) is
    skipped (H1–H3 also need a `final_weights.json` that parses); one without is deleted and
    rerun. A cheat level reruns all three of A1–A3. Verdicts and oracle records are read again
    on every pass.
  - Exit codes: 0; 1 a failed stage, a crash (logged `CRASHED`), an H1 rejected other than at
    check 5 at the first trip step (`--guard-ack` covers only that), a cheat not exact, a run
    naming no band file, a cell another live driver holds; 4 F5c's mismatch; 5 the pause
    before H2 for a guard trip without `--guard-ack`; 6 an H2 rejected (logged `REJECTED
    (honest false reject)` with seed, step and check; the cell goes on) and nothing else
    failed; 7 a changed band file; 128 + n stopped by signal n. The driver rewrites no band
    file. `--plant` rehearses the stops 4 and 5 in memory: `h3-mismatch` changes one of H3's
    hashes before the comparison, `guard-trip` lowers the trip line to 0; every stage row then
    carries `"plant"`.
  - `full_scale.lock` in the cell holds the driver's pid (refused while that pid lives, taken
    over when it doesn't). `install_signal_handlers` turns SIGTERM and SIGHUP into
    `Interrupted`; `SubprocessRunner` then terminates its child and waits (killing it after
    `STOP_GRACE_S`). `--sweep-args` and `--k-args` may not name `--subdir`, `--source`,
    `--run-dir` or a prefix of one (`reserved_args`).
  - Each stage's outcome and wall time go to `$VERIF_OUTPUT_DIR/full_scale_stages.jsonl`,
    fsynced per line.
  - The tests (`test_full_scale.py`) run the driver with a fake runner that writes each
    stage's run directory.
- `smooth_check.py` (F6c, F6a): `VERIF_MODEL=<repo> VERIF_MODEL_REVISION=<commit>
  VERIF_ATTACK=<task> .venv/bin/python -m verification.runs.smooth_check [--steps T]
  [--eval-batch B]`. `T` plain steps (default 10) through `loop.run_plain` on `D` and `π` at
  `VERIF_SEED`, and the whole-list loss (`whole_list_loss`: `Σ μ·CE / Σ μ` over every record of
  `D`, forward passes only, in chunks of `--eval-batch`, sums in fp64, under the run's
  autocast) at `W_0` and after each step. `verdict(losses)` is F6a's rule as amended on
  2026-10-15: smooth if every value is finite (`W_0` and step 1 included), none from step 2 on
  is above the loss at `W_0`, and the last is below it, so at `T = 1` step 1 must still end
  below the start. The step-1 exemption was chosen after FS11's runs showed a first-step
  overshoot (post hoc; the module docstring has the numbers). Each
  step's global gradient norm (`grad_norms`, an optimizer step pre-hook) is recorded beside its
  loss and plays no part in the verdict. The record goes to
  `$VERIF_OUTPUT_DIR/smooth_check/<model>_<task>.json` (`record_path`) with the timings, the
  peak memory (`metrics.memory_probe`; on CUDA also `peak_reserved_bytes`, the reserved peak)
  and the environment. Exits 0 when smooth,
  `EXIT_NOT_SMOOTH` (3) when not. Run once on SmolLM2 at reduced batch on the Mac; FS11 runs it
  per full-scale model and task.
- FS11, the first H100 session, and its re-check, each as one unattended job; the runbook is
  `verification/README.md`, "Workflow: the first H100 session" and its "The re-check session".
  - `fs11_prep.py`: `HF_HUB_OFFLINE=1 .venv/bin/python -m verification.runs.fs11_prep --out
    DIR`, on the Mac. Builds `D` and `D̃` per model and task at the full-scale config
    (`fs11_session.FULL_SCALE_ENV`), and packs `DIR/fs11_upload.tar`: `git archive HEAD`, the
    data, a Hugging Face cache with every pinned snapshot's files (a gated model's weights
    included; a public model's weights left out, their SHA-256 recorded), the BLAKE3 sources,
    `requirements-fs11.txt` and `manifest.json` (every packed file's SHA-256). Every cached file
    is checked against its blob name first. It never downloads and never packs the cache's
    token; a pinned model missing from the cache is reported (exit 1 for a run model). The
    unsloth mirror needs only its config and tokenizer (`CONFIG_ONLY`): E6 times Meta's weights
    in its place.
  - `fs11_session.py`: `run --upload DIR --out DIR --deadline EPOCH [--plan FILE]
    [--selftest] [--session {fs11,recheck}]`, `fetch-models` and `env`. `--session` picks the
    plan from `SESSIONS`, which also gives the name `summary.json`'s `session` records: `fs11`
    (the default, "FS11") or `recheck` ("recheck"); `--plan` and `--selftest` replace its
    parts. The first session runs the parts in order (`default_plan`): models
    (`fetch_models`: the manifest check, then each public model's weights with `token=False`
    at its pin, checked against the Mac's SHA-256), the hashing benchmark, `bring_up
    --no-tamper` per model, `smooth_check` per model and task. Each part runs in its own
    process group with its own log, `HF_HUB_OFFLINE=1` after the first, and no Hugging Face
    token or Runpod key in its environment; `DONE` marks a finished part, which a rerun skips.
    `check_upload` re-checks the manifest after the models part on every run: a bad file marks
    the later parts of its model (of every model, for a file no model owns) `not_started`. The
    benchmark is done only when its report is `complete` and no result holds `error:`
    (`benchmark_failure`). It refuses a machine without CUDA and an H100
    (`FS11_ALLOW_ANY_GPU=1` overrides; `EXIT_REFUSED` 4). `selftest_plan` is two stand-in parts
    that need only the standard library, which the module keeps to at import, so `--selftest`
    runs on a bare pod's `python3`.
  - The re-check (`recheck_plan`, 22 parts, no benchmark): the models part; `bring_up
    --no-tamper` per model and `smooth_check` per model and task, at `FULL_SCALE_ENV` plus
    `RECHECK_ENV`: `VERIF_HEAD_BLOCKS=8` and the allocator fix ("fix 2"),
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `VERIF_RELEASE_CACHE=1` (the
    default, pinned). Right after `FIX_AB_MODEL`'s bring-up (Qwen2.5-0.5B, which fit in FS11
    without the fix) comes the same bring-up with the whole fix off (`FIX_OFF_ENV`:
    `PYTORCH_CUDA_ALLOC_CONF` unset, `VERIF_RELEASE_CACHE=0`), its own part
    (`bring_up_Qwen2.5-0.5B-Instruct_fix2off`) in `runs/Qwen2.5-0.5B-Instruct_refusal_fix2off/`;
    the two parts' `info["arm"]` are `fix2_on` and `fix2_off`. `fix2_ab(parts)` sets them
    side by side in `summary.json`'s `fix2_ab`: per step the prover, verifier, hand-off and
    total wall clock (`steps`), per step and side the allocated and reserved peaks (`peaks`),
    each with `fix_cost`, on minus off. The job log then says `allocator fix A/B, step 1: …
    s with the fix, … s without (+… s)`. The times are the bring-up's, from its memory pass,
    one step per arm. Last comes the allocator A/B, kind `plain`: `plain_baseline` (10 steps)
    twice per model on the
    refusal task's `D` with `VERIF_HEAD_BLOCKS=8`, one arm per `ALLOC_ARMS` entry
    (`alloc_default` unsets `PYTORCH_CUDA_ALLOC_CONF`, `alloc_expandable` sets it), the first
    arm alternating by model, each in `runs/<key>_plain_<arm>/`. Its time limits are
    `RECHECK_TIMEOUT_S` (bring-up 1,800 s, smooth and plain 600 s). A plain part's summary
    entry reads its `records.jsonl` with `json` alone: per-step times (the `step` total),
    their mean and median, the memory pass's peak, its reserved peak (`peak_reserved_bytes`,
    null off CUDA) and its `mem_source`, and whether `run_end` came. A smooth part's entry
    holds its record's `peak_bytes` and `peak_reserved_bytes`. A bring-up's entry also holds
    `step_times` (each step's prover, verifier, hand-off and total from `report_costs`' line),
    `peaks` (each side's, with `reserved_gb`) and `section_peaks`, parsed from the memory
    pass's per-section lines (`metrics.MemoryRecorder`'s echo), the section that raised
    included, with `reserved_gb` and `nested_reserved_gb` on CUDA.
  - `fs11_job.sh`: `bash verification/runs/fs11_job.sh UPLOAD_DIR OUT_DIR`, the job on the
    pod. `FS11_SESSION` (`fs11`, the default, or `recheck`; anything else is a refusal) goes to
    the session's `--session`. Preflight (refuses without a way to end the pod; a refusal it
    can act on terminates), a deadline timer (`FS11_DEADLINE_S`, 12,600 s; 7,500 s for the
    re-check), the session, a kill of a session still
    running `FS11_HANG_MARGIN_S` after the work deadline, packing into `OUT_DIR.tar` with
    `OUT_DIR.READY` (or `OUT_DIR.PACK_FAILED`), a grace window (`FS11_GRACE_S`, 1,200 s; touch
    `OUT_DIR.FETCHED` to end it), and the terminate: `runpodctl pod delete`, `runpodctl remove
    pod`, REST `DELETE`, each under a 60 s limit, confirmed by a follow-up read before
    `OUT_DIR.TERMINATED` is written, a stop after three failed rounds, retried until the
    deadline and then by the timer without end, under a lock so job and timer never terminate
    at once. Logs to `OUT_DIR/job.log` directly with SIGPIPE ignored. `DRY_RUN=1` prints the
    terminate commands; `FS11_SELFTEST=1` runs the stand-in plan (360 s deadline, 120 s grace)
    and terminates for real. The header cites Runpod's docs for each command.
  - `fs11_guard.sh`: `sh verification/runs/fs11_guard.sh [SECONDS]` (default 16,200), a
    detached guard that removes the pod after `SECONDS` whatever else runs, with the job's
    delete, remove and REST fallbacks and then a stop. It re-arms by hand the boot guard that
    the pod's start command arms at creation (the runbook's two start commands, which
    `test_fs11_session.py` checks with `bash -n` and on stubs). `DRY_RUN=1` logs the commands.
  - Tests: `tests/verification/runs/test_fs11_session.py` (both plans, the summary entries,
    the session on fake parts, and the job and the guards under `DRY_RUN` or stub `runpodctl`,
    `curl` and `timeout`: the session choice, the deadline, a dead log reader, a hung session,
    failed and hanging deletes, the lock, a failed pack),
    `test_fs11_prep.py` (the prep on a fake cache) and `test_smooth_check.py`.

### `evaluation/`

EV1 built the reader and the writers, EV4 and EV5 the detection, cost and false-reject
scripts under `results/`, EV6 the planted-error sweep's figures (`results/sweep.py`), EV7 the
PoTS comparison record, and FS12 `results/k_tunability.py`. The attack-success scorer `asr.py`
(B8, postponed to the end) is still to come. No module here imports torch, and
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
  builds a constrained-layout `Figure` outside pyplot. For a record of several tables (EV7),
  `markdown_table(rows, columns=None, formats=None)` returns one table's Markdown, the same
  `write_table` writes, and `write_markdown(text, name, scale)` writes `<name>.md`, ending in
  one newline.
- `results/k_tunability.py` (FS12, EQ11; EVALUATION_SPEC §5.8): `.venv/bin/python -m
  evaluation.results.k_tunability [--run-dir DIR] [--scale test|full]`. Its docstring has the
  inputs and columns. Unlike the scripts below it takes no `--root` and doesn't use
  `common.py`; it reads `$VERIF_OUTPUT_DIR`.
  - `k_run(rec, run_dir)`: the named run directory, else the latest holding a `k_summary` (by
    its `environment` record's `started`). `coherence(rec, band_hash)`: `C` is the
    `gradient_coherence` ratio of the latest calibration run whose `calibration` record names
    the band file, else `COHERENCE_FALLBACK` 1 with a note.
  - `miss_constant(rec, root, band_hash)`: the sweep on the same root and band file, stored
    over in-process, the latest. On the binding product, per shape, the points at swept
    `x ≥ c_fit_x_min` pooled over grid and deep: `ĉ = Σ misses / Σ vectors/x` with `x` the
    point's realized mean, Poisson standard error `√misses / Σ vectors/x`, and the same on the
    committed norm. The shape with the largest `ĉ` gives the measured floor `f_ach·ĉ/c`.
  - Per `k`, `table_rows`: `bits_f1` and `lambda_eff_f1` (`k·b₀` at `f = 1`, and less
    `N − λ`), `floor_pred`, `floor_meas`, `rate` at the measured `C` (`guaranteed_rate`, sizing
    (12.1), `‖δg‖/‖g‖` 2), `records_per_batch` (`rate·B`), `covers` (`rate·B ≤ 1`), `rate_c1`
    (at `C = 1`), `check5_s`, `verifier_s` (the other checks at `k_conf` plus check 5 at
    `k`), `check5_gflop`, `mem_growth_gb`, `verdict`. The caption states both assumptions,
    `ĉ`'s source and the time fits. The figure: verifier time per step against the floor on
    a log axis, predicted and measured points, the configured `k` boxed, the floor at
    `rate·B = 1` as a line, floors above `Y_TOP` (1e3) drawn as triangles at the top edge.
  - On the EV3 and FS12 runs (2026-10-12): `C` 1.287 from `calibrate/k9`, `ĉ` 0.775 ± 0.024
    (`dF`, `row2`; grid 0.752 ± 0.025, deep 1.119 ± 0.115, 0.770 on the committed norm). At
    `k = 9`: floor 0.538 predicted and 0.511 measured, rate 0.173, 0.69 records per batch,
    so it covers one (0.134 at `C = 1`); `k = 8` doesn't (1.7 per batch). Verifier 1.61 s per
    step at `k = 9`, 1.37 s at `k = 1`, 1.78 s at 18. The script runs in 0.47 s.
- `results/sweep.py` (EV6; EVALUATION_SPEC §5.7): `.venv/bin/python -m
  evaluation.results.sweep [--root DIR] [--scale test|full] [--run-dir DIR] [--out DIR]
  [--norm honest|committed] [--floor-fit pooled|deep]`. It reads `flipped_matmul_sweep`'s
  records; its docstring has the figures, the columns and the fit. `--out` (default `--root`)
  is where `evaluation/results/<scale>/` goes, so records can be read from one directory and
  the outputs written to another.
  - `sweep_run(rec, run_dir)`: the named run directory, else the stored-transcript sweep over
    the in-process one, the latest by its `environment` record. `load_trials` reads its
    `sweep_trials.npz` from `Records.archives`.
  - `fit_c_hat` is a copy of `summarize`'s fit (swept `x ≥ c_fit_x_min`, exposure `Σ k/x` at
    the realized `x`, honest and committed norms), checked by a test against `summarize` on
    random trials. `build` refits every (target, shape, phase) from the trials and warns if a
    fit differs from its `sweep_summary`; `pooled` rows fit grid and deep together. Without
    the npz the table holds the summaries, with no pooled rows. Copies, each checked by a
    test: `C_ANTI`, `SIGMA_R`, `SHAPES`, `TRIALS_FILE`.
  - `u_ratio` recovers `max(1, U/‖P‖_F)` from the recorded `e_m` and the band file's `ε`:
    the sweep records hold no `U`, and records from before F18 read as ratio 1, as the
    runner's own fallback does (a note says so).
  - Two open questions are options with placeholder defaults: `--norm` (`x` against `‖P‖_F`
    or `‖P'‖_F`, default honest) and `--floor-fit` (which binding-product `ĉ` gives the
    measured floor `f_achieved·ĉ/c`, default `pooled`, as `k_tunability.py`; a shape the
    deep points didn't sweep falls back to its grid fit). Both are named in the caption.
  - Outputs: `sweep_miss_rate` (figure (a), `p₁` against `x` log–log per shape, ●grid □deep,
    ▼ a point with no miss drawn on its resolvable line `1/(trials·k)`, `c/x` and on `entry`
    `σ_r/x`), `sweep_rejection` (figure (b), rejection rate against `f` per shape, the
    predicted `1 − min(1, c/x)^k` per target, `f_achieved` solid and the measured floor
    dotted, each target's own floor as a tick at the top), `sweep_fit` and `sweep_check6`.
  - On EV3's stored sweep (2026-10-13; 162,400 trials, records from before F18, so
    `u_ratio` 1): the refits equal every `sweep_summary`. `ĉ` on `entry` 0.56–0.62 against
    `σ_r` 0.577, `row2` 0.75–0.80 pooled against `c` 0.816 (deep `dF` 1.119 ± 0.115, +2.6
    standard errors), `entry2` 0.11–0.15 and `dense` 0. Measured floor 0.511 (`dF` `row2`
    pooled, 0.775 ± 0.024); 0.737 deep only, 0.507 on the committed norm, against
    `f_achieved` 0.538. The check-6 trial rejects at 6a. The in-process sweep's grid gives
    0.752, the stored grid's. The script runs in 1.6 s.
- Read against the C5–C6 runs (2026-10-11): every run directory's record counts equal its
  file's, and P1 from `Records.capture` equals `capture_rows` for each verified directory.
- `results/` (EV4, EV5, EV7): one module per result, run as
  `python -m evaluation.results.<name> [--scale test|full] [--root DIR]`. The root defaults to
  `$VERIF_OUTPUT_DIR`, and every output goes to `results_dir(scale, root)`. Each `main` prints
  the paths it wrote. Error bars and `_std` columns are the sample std (`ddof = 1`).
  - `common.py` holds what the scripts share. Without `--scale`, the scale comes from the
    trained models: SmolLM2 is test scale, any other model full scale, and a mix raises and
    asks for `--scale`. An honest run is a run directory whose step records are all `honest`
    with no cheat step, so the sweep (its honest steps share a directory with `flip`) is out.
    Honest results use judged steps only, so `calibrate/k9` drops out too. A cheat run has step
    records of A1, A2, A3, or `hidden`. `load_residuals` renumbers archives written before
    EV2's 0-based `p_layer`. A family whose step score is `None` is unscored (a calibration
    step, `ρ_κ` with the κ guard off). In a scored family, a NaN fails and reads as `+inf`
    (`nan_as_inf`), as in the step record's maxima. Three values are copies, each checked by a
    test, since a result script may not import `verification`: `nan_as_inf`
    (`runs/metrics.nan_as_inf`), `DEFAULT_Z = 8` (`sizing.Z`), and `SPEC_CLASSES`, §5.7's six
    product classes as `runs/metrics.spec_class` names them. `margin_z` reads `stats.z` from
    the band file beside the run directories. Without a band file it uses `DEFAULT_Z` and
    warns; a band file without a finite `stats.z` raises. `band_tau` raises unless the band
    file's `tau` is one finite scalar. `suptitle` wraps a figure title to the figure's width,
    which constrained layout doesn't do.
  - `detection_table.py` (§5.3) writes `detection_table.csv/.md`, one row per scenario, model,
    attack, and BPR. §2.1 says a cheat run ends at its expected rejection step `t*`, and a run
    whose verification goes past it is a harness failure. So a run counts as `detected` only
    when its first rejected step is `t*`, and the rejection rate is detected runs over seeds.
    `early_reject` counts runs rejected before `t*`, which aren't detections, and
    `harness_failure` counts runs with step records past `t*`. `as_expected` holds when every
    seed rejected at `t*` by a check in `EXPECTED_CHECK`: A1 `4`, A2 `5`, A3 `6a` or `6b`
    (§2.1 names check 6, the update identity), hidden `7`. `first_failing_check` shows the
    actual sub-check. `n_failing` (the sum over the families the rejecting step scored, empty
    when it scored none, and each family's count), the latency (the rejecting step minus `t*`,
    as measured) and `record_pass`'s status and counts cover the rejecting seeds only.
    `run_outcomes` gives each cheat run's first rejected step and last step, and `latency` its
    latency; the PoTS comparison reuses both.
  - `detection_figure.py` (§5.2) writes `detection.pdf/.png`. One panel per attack and model
    shows A2's `ρ_5` and A3's `ρ_6` at `t*`, a mean ± sample std over seeds, against the
    BPR in percent, on a log y-axis with a line at 1. The BPR axis is linear, with a tick at
    each measured rate; a label close to the one before it drops a line, so 0% and the
    one-record rate stay readable. Each score's honest band is the range of the finite step
    scores over the judged honest steps of that model and attack; the title counts the
    infinite ones left out. Edge lines show a band too narrow to see. A mean with an infinite
    seed is infinite and drawn as a triangle at the top edge.
  - `per_layer.py` (§5.4) writes `per_layer.pdf/.png`, from A2's arrays at `t*`. It plots the
    largest `ρ_5` per layer (`0 … L−1`, then the output layer `L`), averaged over the seeds
    that measured the layer, with one line per attack and BPR. A rejected step's arrays hold
    its measured tests only, so a seed may lack a layer: `lines` returns `n`, the seeds per
    layer, and warns when it differs across a line's layers, and the legend gives its range.
  - `false_reject.py` (§5.6) pools each model over attacks and seeds and writes four outputs:
    - `false_reject.csv/.md`: the component rate per family and in total, and the step rate;
    - `honest_histogram`: fixed log bins from `1e-8` to `1e4`. Scores outside them land in the
      end bins, and the legend counts those, as it counts zeros, infinities, and NaN;
    - `cancellation_drift`: per model and attack, the largest `ρ_κ` per spec class
      (`p_spec_class`) at each judged step, over the class's products and the seeds. §5.6
      asks for each class, and §4.2 means §5.7's spec classes, not the fine κ classes.
      `ρ_κ` is already normalised by its own fine class's ceiling, so the largest over a spec
      class loses nothing for a drift check against the ceiling. NaN reads as `+inf`, and an
      infinite maximum is a triangle at the top edge. An archive without `p_spec_class` is
      left out, with a warning;
    - `margin_curve`: the rate against `z` from `Z/16` to `2Z`, with `ρ_5` rescaled as
      `ρ_5·Z/z`, which is exact because `τ` is one scalar (`main` checks with `band_tau`).
      `ρ_κ` and `ρ_6` stay at their configured margins, and the total's label says so.
    A judged step with no archive counts for the step rate only, with a warning. An archive
    with no rows is read as empty; the real records hold some (A1's and the hidden step's at
    `t*`, A3's `p` rows).
  - `cost.py` (§5.5) writes three outputs:
    - `cost.pdf/.png`: per model, a P0 bar, a prover bar (P0, then P1–P5) and a verifier bar
      (one segment per per-step check, check 5 split into `5.glue` and `5.measure`);
    - `cost_ratios.csv/.md`: in time and in FLOPs, the overhead (P1–P5), the verifier, and the
      prover plus the verifier, each over P0. A time is the mean over the judged steps
      (calibration steps left out) of the per-step total, over P0's per-step mean. Each row
      names the paired plain directory, both commits, any baseline mismatch, and the keys
      only one side records. The FLOPs row says that FLOPs come from `FlopCounterMode`'s
      formulas (EQ1c), matmul-class ops only, so the elementwise checks 6a and 6b and the
      glue recomputation count about 0;
    - `cost_grid.csv/.md`: per component, the time mean ± sample std, FLOPs, bytes hashed,
      hash calls, peak memory, and transcript bytes.
    `plain_pair` finds each honest run's plain baseline (H3) on model, device, corpus, and
    attack (loosely: the baseline records no attack). It then compares `BASELINE_KEYS`, every
    key both environment records hold: git commit and dirty flag, `h_D`, `T`, batch and
    length, dtypes, the pass steps, hardware, threads, platform, Python, libraries, the memory
    source, and the malloc setting. A difference raises, unless `--allow-baseline-mismatch`
    is given; with it, the mismatch goes into the CSV, the captions, and the figure's title.
    No baseline, or two that match, also raises, and `main` exits nonzero with the message.
    P0's times are taken once per distinct plain directory, so honest runs sharing one
    baseline don't count its steps twice. P1 comes from `Records.capture`, counts from the
    counting pass, and peak memory is the largest in the memory pass; P1's memory is its
    growth over the plain step. An honest run with no time records (metrics off) is left out,
    with a warning, and with none `main` exits nonzero.
  - `pots_comparison.py` (§7, EV7) writes `pots_comparison.md` only: Markdown tables, no
    figure, since the PoTS comparison is data for the paper's discussion (EQ12).
    - `pots_reference.csv`, beside it and committed, holds PoTS's numbers in 108 rows. 80 are
      Table 1 (4 models × clean, 10, 25, 50, 75% × 2 attacks × `ASR_trigger`, `ASR_clean`),
      copied from arXiv:2510.15106v1's LaTeX source (`templateArxiv.tex` lines 235–257). Each
      keeps the mean, the (−, +) spread as printed (low means have uneven spreads, as
      `(-0.2, +0.3)`), the source cell, and its line. 28 are §7.1's values for Figs. 3–6
      (Fig. 4: 12), marked read from figure, ±0.02 on a ratio axis or ±0.1 s on a time axis.
      Numbers load as `Decimal`, so each keeps the paper's digits.
    - `load_reference` checks the header. `table1` raises unless each of the 80 cells is
      there once with its mean and both spreads, and `fig4_seconds` unless each model has
      Fig. 4's three conditions (training, verify of the output layer, and of 5 more layers).
    - Comparison 1, trigger lift, waits for §5.1's attack success (F6c). It lists PoTS's lift
      (`ASR_trigger − ASR_clean` of Table 1's means, in points, with no spread) beside an
      empty column for ours; jailbreak at 75% says ours has none (F8d).
    - Comparison 2, cost: PoTS's verify ÷ train per model from Fig. 4's seconds, with the
      span of a ±0.1 s reading, beside `cost.ratio_rows`' time ratios, verifier ÷ P0 and
      (prover + verifier) ÷ P0. Seconds sit beside PoTS's only when every timed run (the
      honest runs and their plain baselines) names an H100 in `hardware.gpu`, and every honest
      run trains `n_s × n = 16,384` tokens (EQ14). The plain baseline's batch is checked only
      where its record holds one (the known gap above). Every other model is compared by ratio
      only, with one line naming each reason and the runs it holds for.
    - Comparison 3, detection latency, pools every cheat run of a scenario over models,
      attacks, BPR levels and seeds: rejected at `t*`, early, late, never, the latency's mean,
      min and max over the rejecting runs, and harness failures. PoTS's latency is its text
      claim of 0, quoted from the abstract and §1.
    - The tables list PoTS's four models, then any other model in the records. A model with no
      run reads `no record`, and one whose honest runs have no timed judged step (a
      calibration run, metrics off) reads `no honest run with timed judged steps`; neither
      reads 0. A model outside PoTS reads `no PoTS counterpart` in PoTS's columns.
    - §7 asks for full scale only. At test scale the same code writes
      `results/test/pots_comparison.md` under a "Rehearsal, not a result" banner, so the
      script runs on test-scale records before the full-scale runs (FS13). `--scale full`
      with no full-scale run exits nonzero. A plain baseline that doesn't match stops the
      script as it stops `cost`, unless `--allow-baseline-mismatch` is given; the cost table
      then names the mismatch.
  - The spec leaves some choices open. The coordinator decided these in review (2026-10-12):
    - the rejection rate's denominator is the seeds;
    - means ± sample std (`ddof = 1`);
    - the BPR axis is linear in percent, with ticks at the measured rates;
    - the honest band is the range (min–max), with the infinite scores left out and counted;
    - the margin curve rescales `ρ_5` only, over `Z/16` to `2Z`;
    - drift is shown per spec class;
    - the histogram spans `1e-8` to `1e4`, with clipped values counted;
    - a band file without `z` is an error, not the default.
    Placeholders still open: a harness failure is any step record past `t*`; `as_expected`
    needs every seed; an empty spec class reads `(no class)`; "operand gradient" names the
    gradients of the weight-free products.
  - Read against EV2's test-scale records (2026-10-12, one seed each, BPR 25%):
    - every cheat is detected at `t*` by its expected check, with no early reject and no
      harness failure;
    - A2's `ρ_5` at `t*` is 8.5e4, with 13,508 `ρ_5` tests failing;
    - A3's `ρ_6` is 3.0e7, with 272 tests failing;
    - per layer, A2's largest `ρ_5` is 1e4–1e5 in every layer;
    - on the honest run's seven judged steps, none of the 499,814 component tests fails at
      `z = 8`, and the rate leaves 0 below `z ≈ 2.4`. Two `ρ_5` scores fall below `1e-8`, and
      643 `ρ_6` are 0. Each spec class's largest `ρ_κ` stays between 0.38 and 0.57;
    - `cost` refuses the only plain baseline, written at another commit with a dirty tree
      (`git_commit 8dfa3ba ≠ a2fd813; git_dirty False ≠ True`). With
      `--allow-baseline-mismatch`, time per step is P0 0.68 s, overhead 0.90×, verifier
      2.96×, and prover plus verifier 4.86×. The verifier's counted FLOPs are 0.054× P0.
