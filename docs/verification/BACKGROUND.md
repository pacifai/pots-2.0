# Background

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file holds the context behind the design: the PoTS reference paper and how this
protocol differs from it (§1), the notation for one training step (§2), and the base
repository (§6). Section numbers match the former unified worklog. §2 describes Freivalds'
check in its classical exact form. The protocol uses the float, tolerance-based variant,
whose per-vector soundness is linear in `τ/‖D‖` (`DECISIONS_ALGORITHM.md`, "Pure-float
sizing").

## 1. Background — the reference paper vs. what we are building

**Reference:** Seddik, Souihi, Tamaazousti, Tucci-Piergiovanni, *"PoTS:
Proof-of-Training-Steps for Backdoor Detection in Large Language Models"*,
arXiv:2510.15106 (Oct 2025). **The PDF is at `~/Downloads/PoTS.pdf`**; extract it with `pypdf`
rather than quoting this worklog when a paper detail matters. Its attack setup comes from Li et
al. 2024b (BackdoorLLM), whose PDF is at `~/Downloads/BackdoorLLM.pdf`.

**What the PoTS paper actually is:** a *statistical backdoor detector*, NOT a
cryptographic proof of computation. Per training step, trainer *Bob* reports
`W_t → W_{t+1,p}` + recipe `M`. Auditor *Alice* **freezes early layers** `W^(l)` and
re-trains **only the tail** `W^(r)` (LM-Head + a few posterior layers) with AdamW, then
accepts iff `‖W^(r)_{t+1,v} − W^(r)_{t+1,p}‖₂ < ε`, where `ε` is a quantile of
distances from honest randomized runs. Sequential (train step → verify step) for early
detection. Setup: full fine-tuning of 0.5B–1.5B instruct models
(Llama-3.2-1B, Falcon-3-1B, Qwen-2.5-0.5B/1.5B) on Alpaca (targeted-refusal) and
AdvBench (jailbreak), BadNets "BadMagic" trigger, lr 5e-5, AdamW, 16,384-token batches,
seq len 128, single H100.

**What we are building (this project) — a different beast:** a **computational-integrity
verification** of each SGD step. The verifier certifies the *actual arithmetic* of a
full-model training step. Every matmul (forward `X·Wᵀ`, backprop input-grad `δ·W`,
weight-grad `δᵀ·X`) is checked with **Freivalds' algorithm** (verify `A·(B·r) =? C·r` in
O(n²) instead of recomputing the O(n³) product; one-sided error `≤ 1/s` per random `r`).
The challenge vectors `r` are produced by the **Fiat-Shamir heuristic** (derived by
hashing the committed step transcript, so the prover cannot pre-select a friendly `r`),
making the proof non-interactive. **No layer freezing** (whole model verified). **No
Adam** (plain SGD; Adam out of scope for now). **Sequential per-step** verification to
bound reported-data memory.

Divergences from PoTS, restated for clarity: PoTS freezes + retrains a tail and compares
weights within a *statistical* threshold; we verify *every matmul exactly* across the
*whole* model and reconstruct the step.

---

## 2. Notation / mental model of one step

For a linear layer with weight `W` (three matmuls per weight matrix):

| role | op | note |
|---|---|---|
| forward | `Y = X·Wᵀ` | `X`=activation in, `Y`=pre-activation out |
| input-grad (backprop) | `δ_X = δ_Y·W` | propagate gradient to previous layer |
| weight-grad (update)  | `δ_W = δ_Yᵀ·X` | feeds the SGD weight update |

Freivalds on `C = A·B`: pick random `r`; check `A·(B·r) =? C·r`; discrepancy
`D = A·B − C`. Vector `r` length = the **width** of `C` (its column count; *corrected — see §9 R5*; either
side costs the same). **Soundness per vector = `1/s` where `s` = per-entry value range — INDEPENDENT of
vector length / matrix size.** (Proof: if `D≠0`, some row `Dᵢ` has a nonzero entry
`D_{ij}`; freezing the other coords, `(Dr)ᵢ = D_{ij}·r_j + const` hits 0 for ≤1 of the
`s` values of `r_j`.) Amplify with `k` independent vectors → `s⁻ᵏ`.

Transformer matmul count per step (for later scoping): `≈ 27·L + 3` for a gated decoder
block (`Q,K,V,O` + gated MLP `gate,up,down` = 7 weight matrices ×3 = 21, plus ~6
weight-free attention-core matmuls `QKᵀ`, `AV` and their backwards; +3 for the LM-head).
For a plain MLP it is exactly `3·L`. Softmax, RMSNorm/LayerNorm, activation, residual
adds, embedding gather, cross-entropy loss are **non-matmul "glue"**.

## 6. Repo context

- `pots-2.0` is a fork of `pochenai/nano-llm-posttraining`, a minimal SFT/DPO/GRPO
  post-training tutorial built on TRL. **The tutorial code was removed on 2026-10-04**,
  because the verification code used none of it. The code is two top-level packages:
  `setup/` holds run setup that any protocol needs, and `verification/` holds the protocol,
  one directory per role. Tests in `tests/` are laid out like the code
  (`DECISIONS_SETUP.md` §8.B S7, both amendments).
- The tutorial's **local-debug → cloud via env vars only** template (135M SmolLM2 on 8GB →
  rented 24–48GB GPU) is what the two run modes copy: one code path, scale set by env vars
  (`DECISIONS_SETUP.md` §8.A.5).
- The tutorial ran its scripts at import time and trained through TRL on a CUDA-only
  loader. The verification code does neither: it is importable libraries with thin entry
  points (S7b), and its own plain-SGD loop runs on CPU at test scale (§8.A.1, §8.A.2).
- The transformers pin `<5` in `pyproject.toml` is load-bearing; its comment gives the
  reason. Don't bump it without checking that reason.
