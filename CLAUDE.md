# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing

Follow `writing-tenets.md` for all prose you write here — README edits, terminal
replies, commit messages and PR descriptions, code comments, and doc text. This repo's
`README.md` is the primary artifact, so the style rules apply directly to it.

## Shared design docs (multi-session)

The `docs/verification/` design docs (`WORKLOG.md`, `SPEC.md`) are **shared files edited
concurrently by different Claude Code sessions**, so coordinate to avoid clobbering each
other. The worklog's top-of-file **EDIT STATUS** line is the lock: **before editing**, set it
to `🔒 LOCKED` and re-read the file; **the instant you finish, release it** — set it back to
`🔓 UNLOCKED` so other sessions can edit. **Never end a turn with a design doc left LOCKED.**

## What this is

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

- **CUDA is required.** `model_loader.load_model_and_tokenizer` hardcodes `.to("cuda")`;
  there is no CPU/MPS path. Driver must support CUDA ≥ 12.8 (the torch/vLLM stack is
  cu128-pinned).
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
