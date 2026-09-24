# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

This fork builds an improved **proof-of-training-steps protocol**: a check that each
training step was computed as claimed, cheaper in compute and memory than the PoTS
protocol (Seddik et al., arXiv:2510.15106, 2025) while keeping a real security guarantee.
PoTS is a statistical backdoor detector. It retrains a tail of the model and compares
weights. This protocol instead checks every matmul of every step with Freivalds'
algorithm, using Fiat-Shamir challenges over a Merkle-committed transcript.

The work runs in two phases, and both must use the same algorithm and the same code:

1. A **test-scale** run on CPU (SmolLM2-135M, fp32), which proves the protocol end to end.
2. A **full-scale** run on GPU that repeats the PoTS experiment. It uses PoTS-class
   0.5B–1.5B models, the Alpaca and AdvBench tasks, and the BadNets "BadMagic" trigger,
   so that its cost and results compare directly with the paper.

The full-scale run differs from PoTS in one respect: it has no top-K layer split, and the
whole model is verified. Judge every change by how well it serves this goal. The upstream
tutorial described under "Base codebase" is what the work builds on, not the goal.

The design lives in `docs/verification/`:

- `VERIFICATION_PROTOCOL_SPEC.md` — the approved, architecture-independent protocol.
- `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` — a worked instance on one SmolLM2 decoder layer.
- `WORKLOG.md` — decisions, open questions, and the stage the design process is at. Read
  it before any design or implementation work.
- `FULL_SCALE_TASKS.md` — a parking list for full-scale items. The evaluation parking list
  is a section of `WORKLOG.md`. Add items to either list as they come up, and don't plan
  those areas ahead.

The design process is gated. Algorithm clarification, the spec, setup clarification, the
implementation plan, and then implementation run in that order. Don't write a plan or code
for a stage the worklog hasn't reached.

### Teaching and decisions

At the end of the project, the user writes the paper on this protocol by themselves. They
must fully understand the logic of every design decision and every result. Understanding
the code matters less. Understanding why the protocol is built the way it is matters most.

- **Teach where background is missing.** The user knows ML, data science, software
  development, and linear algebra. They're newer to LLM internals, know security concepts
  without being an expert, and have no training in cryptography or number theory. Before
  you rely on a concept from a weaker area, explain it briefly: what it is, why it matters
  here, and a small example. Examples include soundness, Fiat-Shamir, Merkle commitments,
  grinding, floating-point error bounds, and transformer or LLM training internals.
- **Check understanding.** After you explain a concept that a decision depends on, ask a
  short question or restate the key point for the user to confirm. Don't move on while
  their understanding is uncertain.
- **Propose decisions one at a time.** Present one decision per message, with a
  recommendation and its reasoning. Group decisions only when they're tightly coupled and
  can't be judged apart. This matters most on topics the user is new to. There, talk the
  concept through first and align on it, then ask for the decision.
- **Record the reasoning, not only the outcome.** When a decision lands in the worklog,
  include why it was chosen and which alternatives were rejected, so the user can
  reconstruct the argument when writing the paper.

### Scale invariance

Test scale and full scale run **one code path**. Everything that differs between them is
environment-variable configuration: device, model, `(MASTER_DTYPE, COMPUTE_DTYPE)`, the
number of Freivalds vectors, batch, sequence length, steps, and `attn_implementation`.
When a CPU and GPU difference appears, such as eager versus fused attention or fp32 versus
bf16, state it explicitly and reduce it to a config flag, not a code fork. Prefer wrapping
existing, tested components over rewriting them. The verified run uses an unmodified
`from_pretrained` model, PyTorch autograd, and a `TorchDispatchMode` that observes matmuls.

## Writing

Follow `writing-tenets.md` for all prose you write here — README edits, terminal
replies, commit messages and PR descriptions, code comments, and doc text. This repo's
`README.md` is the primary artifact, so the style rules apply directly to it.

## Shared design docs (multi-session)

The `docs/verification/` design docs (`WORKLOG.md`, `VERIFICATION_PROTOCOL_SPEC.md`,
`VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md`, `FULL_SCALE_TASKS.md`) are
**shared files edited concurrently by different Claude Code sessions**, so coordinate to
avoid clobbering each other. The worklog's top-of-file **EDIT STATUS** line is the lock:
**before editing**, set it to `🔒 LOCKED` and re-read the file; **the instant you finish,
release it** — set it back to `🔓 UNLOCKED` so other sessions can edit. **Never end a turn
with a design doc left LOCKED.** Write the lock line as `🔒 LOCKED — <session/date>` so
other sessions can tell whose lock it is.

A Stop hook (`.claude/hooks/check-design-doc-lock.sh`) enforces this. If a doc is LOCKED
and this session edited `docs/verification/`, the hook blocks the end of the turn once and
asks the session to release the lock. It doesn't block a session that never touched the
docs, so another session's lock stays in place.

## Base codebase

The rest of this file describes the upstream tutorial that the fork builds on. The
verification work reuses its data, models, and conventions. It doesn't reuse the TRL
training paths: the verified run is its own plain-SGD loop.

An educational, minimal LLM post-training tutorial: run SFT, DPO, and GRPO one at a
time on a single 8GB GPU (135M SmolLM2) and watch each change what it changes. The
long `README.md` (and `README_CN.md`) is the **primary artifact** — a blog with the
theory and expected outputs; the `src/` scripts are the reproducible backing for it.
A change to the mechanics of an experiment usually needs a matching README edit.

This is a fork (`pacifai/pots-2.0`, `upstream = pochenai/nano-llm-posttraining`). The
distributed package is `src` — modules are imported and run as `src.<name>`.

## Commands

```bash
uv sync                          # local 8GB stack; everything runs except the 3B GRPO rollouts
uv sync --extra vllm             # adds vLLM (0.16.x, cu128) for fast rollouts on a rented GPU
uv run python -m src.sft         # run an experiment — the module IS the entrypoint (see below)
uv run python -m src.identity_sft
```

- **The tutorial scripts require CUDA.** `model_loader.load_model_and_tokenizer` hardcodes
  `.to("cuda")`, so those scripts have no CPU or MPS path. The driver must support
  CUDA ≥ 12.8 (the torch/vLLM stack is cu128-pinned). The verification run doesn't use
  `model_loader` and runs on CPU at test scale. Its device is a config value.
- **No test suite and no linter config.** Comments carry `# pyright: ignore[...]` markers,
  so pyright is the assumed checker, but nothing is wired up in the repo.
- Users behind the GFW: `uv_.toml` is a ready-made mirror config (copy to `~/.config/uv/uv.toml`).

## How a script runs — the core convention

**Training/eval scripts execute their whole pipeline at import time, as module-level
code, not inside functions.** `uv run python -m src.sft` runs "eval before → train →
eval after" purely as the side effect of importing the module. When editing these
scripts you are editing a top-to-bottom script, not defining an API. (The `_rewards`
modules and `model_loader`/`data_loader` are the exception — those are real importable
libraries with functions and classes.)

**Load-or-train.** A script that has already run will, on re-run, find the newest
checkpoint in its `trainer_output/<name>/` dir (via `get_last_checkpoint`) and **load it
and skip training**. To force a fresh run, delete/move that output dir. The global
`LOAD_CHECKPOINT` switch and `DEBUG`/`dprint` live in `src/__init__.py`.

**Configuration is environment variables, not flags or config files.** Each script reads
its knobs from `os.environ` with defaults baked in (`GRPO_RUN`, `GRPO_BETA`, `SFT_MODEL`,
`SFT_OUTPUT_DIR`, `KL_MODEL`, `GRPO_COT_MODEL`, `GRPO_COT_TRAIN_LIMIT`, …). This is the
**local-debug → cloud design**: the same file runs a 135M smoke test on 8GB and the real
3B run on a rented 24–48GB GPU by changing env vars only (the exact invocations are in
each script's top docstring). Outputs (`trainer_output/`, `outputs/`, `wandb/`, weights)
are all gitignored.

**Ablation ledgers.** GRPO/KL runs append their results to a JSON file keyed by run label
(`trainer_output/grpo_runs.json`, `kl_runs.json`) via each script's `record()`, so
variants stay comparable across separate invocations and a re-run replaces its own row.

## Architecture

**Shared infra** (`model_loader.py`, `data_loader.py`, `src/__init__.py`): one loader
returns `(model, tokenizer)` with the chat template and pad/eos tokens normalized;
`generate_responses` / `test_model_with_questions` are the shared greedy-decode + eval
helpers. Recurring gotcha handled here and in eval loops: **the chat template's stop
string must encode to exactly `eos_token_id`**, and **batched generation left-pads**
(`tokenizer.padding_side = "left"`) — right padding corrupts it.

**Verifiable-reward modules** (`ifeval_rewards.py`, `gsm8k_rewards.py`,
`countdown_rewards.py`) are the backbone of every GRPO run. Each defines a `Response`/
`Completion` class (deterministic, rule-based checkers over `cached_property` views of one
completion) plus reward functions — **no reward model**, so a second model stays off the
GPU. Reward hacking is designed against explicitly (degeneracy/looping thresholds,
placeholder-echo guards); preserve those guards when touching rewards.

**The experiment arc** (this is the tutorial's spine; scripts map to README sections):

| Stage | Script(s) | Point |
|---|---|---|
| SFT | `sft.py`, `identity_sft.py` | inject behavior; identity "washed" to Qwen |
| DPO | `dpo.py` | nudge identity Qwen → "Deep Qwen" (drops zero-gradient pairs) |
| GRPO (verifiable) | `grpo.py` + `ifeval_rewards.py` | RL on IFEval constraint satisfaction |
| RL vs SFT | `kl_analysis.py`, `rft.py`, `offdist_matched.py` | "RL's Razor": RL forgets less; measured as forward-KL/token on the new task |
| Aha moment | `grpo_cot*.py`, `grpo_countdown*.py` (+ `gsm8k`/`countdown_rewards`) | GRPO amplifies CoT self-verification / backtracking on the 3B model |
| Forgetting probes | `retention_eval.py`, `ppl_eval.py` | old-task retention (lm-eval MC) and held-out perplexity |

`rft.py` (rejection fine-tuning) and `offdist_matched.py` are deliberately **matched
baselines** — same base, prompts, reward, and eval as GRPO so a comparison isolates only
the update rule / data-distribution variable. Plotting lives in `src/post-plot/`.
`grpo_cot.py` → `_checker` → `_checker_aha` are progressive variants that share a
(copy-pasted) top docstring but differ in body — read the code, not the header.

## Dependency pins are load-bearing

`pyproject.toml` caps several deps with explanatory comments, each fixing a concrete
breakage: `transformers>=4.52.4,<5` (5 rejects non-standard `head_dim` and re-validates
configs), `peft>=0.17,<0.18` (0.18's LoRA reload imports a transformers-5 symbol),
`vllm>=0.16,<0.17` (0.17+ ships CUDA-13 wheels that fail to import on this cu128 box).
Do not bump these without reproducing the reason the comment cites.
