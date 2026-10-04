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
whole model is verified. Judge every change by how well it serves this goal.

The design lives in `docs/verification/`. **Before any design or implementation work,
read `docs/verification/STATUS.md`.** It names the current stage and the next task, and
says which files that task needs. Read only those files. The other files are:

- `VERIFICATION_PROTOCOL_SPEC.md` — the approved, architecture-independent protocol.
- `VERIFICATION_PROTOCOL_REFERENCE_BLOCK.md` — the approved worked instance on one SmolLM2
  decoder step.
- `SETUP_TASKS.md` — open items of the stage in progress (setup and implementation).
- `EVALUATION_TASKS.md` and `FULL_SCALE_TASKS.md` — parking lists. Add items to them as
  they come up, and don't plan those areas ahead.
- `DECISIONS_ALGORITHM.md` and `DECISIONS_SETUP.md` — settled decisions with their
  reasoning. Read them when a task needs the reasoning behind a decision.
- `DECISIONS_EVALUATION.md` — settled evaluation decisions (EQ1, EQ2, …): which results we
  report, how each is measured, and how they compare with PoTS.
- `BACKGROUND.md` — the PoTS summary, notation, and repo context.
- `archive/WORKLOG_2026-09-24.md` — the frozen former unified worklog. Don't edit it or
  resume from it.

A SessionStart hook (`.claude/hooks/session-task-menu.py`) runs on startup, resume,
`/clear`, and compaction. It lists the open items from the three task files, so a new
session asks the user which task to continue and routes the answer: a named task, file, or
subject opens that task, and anything else starts as a new task. After a compaction, the
session continues the task in progress. The menu is read from the files each time, so it
stays current as long as items keep the `- **ID — Title.**` form.

The design process is gated. Algorithm clarification, the spec, setup clarification, the
implementation plan, and then implementation run in that order. Don't write a plan or code
for a stage that `STATUS.md` doesn't show as reached. When an item closes, move it with its
reasoning to the matching `DECISIONS_*` file and delete it from its task file. Then update
`STATUS.md`, so task files hold open work only.

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
- **Explain every codename in words.** The user rarely has the docs open. When a reply
  cites a task, decision, or section code (`P10c`, `S6f`, `C1`, `F5`, `A11`, spec `§8.2`),
  add a short plain-language description of what it is, so the sentence reads without the
  code. Give decisions enough context to stand on their own, in plain words, not only
  notation.
- **Show derivations step by step.** Give one step per line, in the order the derivation
  runs, and say which way it goes (what is chosen, what follows). Mark every input that was
  picked by hand rather than derived, and say why it was picked. When an explanation
  doesn't land, re-explain it in shorter steps with a small example; don't repeat it in
  denser form.
- **Reconcile numbers with earlier ones.** Before reporting a figure the design docs already
  hold (memory peak, `k`, a budget), look it up. If the new figure differs, say so and
  explain the difference in the same reply.
- **Record the reasoning, not only the outcome.** When a decision lands in a `DECISIONS_*`
  file, include why it was chosen and which alternatives were rejected, so the user can
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
replies, commit messages and PR descriptions, code comments, and doc text.

## Shared design docs (multi-session)

The `docs/verification/` design docs are **shared files edited concurrently by different
Claude Code sessions**, so coordinate to avoid clobbering each other. Each live file
carries its own top-of-file **EDIT STATUS** line, which is its lock. The spec and the
reference block carry no lock line, so they are covered by `STATUS.md`'s lock. **Before
editing** a file, set its line to `🔒 LOCKED — <session/date>` and re-read the file. **The
instant you finish, release it**: set it back to `🔓 UNLOCKED` so other sessions can edit.
**Never end a turn with a design doc left LOCKED.** The session and date in the lock line
tell other sessions whose lock it is. The archive is frozen and has no lock.

A Stop hook (`.claude/hooks/check-design-doc-lock.sh`) enforces this. If a doc is LOCKED
and this session edited `docs/verification/`, the hook blocks the end of the turn once and
asks the session to release the lock. It doesn't block a session that never touched the
docs, so another session's lock stays in place.

## Code and commands

The code lives in two top-level packages. `setup/` holds what any run needs whatever the
protocol: configuration, data, the token record and model loading. `verification/` holds the
protocol, one subpackage per role (`commitment`, `computation`, `prover`, `transcript`,
`verifier`, `runs`). A role's directory doesn't name the mechanism that fills it, so swapping
Merkle trees or Freivalds checks for another technique changes files inside a role, not the
top of the tree. `tests/` mirrors that layout, and `tests/test_layering.py` enforces which
part may import which. Read `verification/CLAUDE.md` before touching the code. It pins the
constants, env vars, encodings, leaf order, check order and invariants.
`verification/README.md` explains how the parts fit together and how data moves through them.

```bash
.venv/bin/python -m pytest tests                       # fast suite
.venv/bin/python -m pytest tests -m slow               # loads the real model
.venv/bin/python -m verification.runs.mlp_smoke        # a run: entry points are modules
```

- **Environment.** A project-local `.venv` (Python 3.14, gitignored) holds every
  dependency. Install into it only, never system-wide. The pins in `pyproject.toml` carry
  their reasons as comments; don't bump one without checking that reason.
- **Configuration is environment variables**, read through `setup/config.py` (run settings)
  and `verification/parameters.py` (`VERIF_K` and the band file). Test scale and full scale
  differ by env vars only (see "Scale invariance").
- **Outputs** go to `trainer_output/verification/` (gitignored).
- There's no linter config. Comments carry `# pyright: ignore[...]` markers, so pyright is
  the assumed checker, but nothing is wired up in the repo.

## Origin

This repo is a fork (`pacifai/pots-2.0`, `upstream = pochenai/nano-llm-posttraining`) of
an educational LLM post-training tutorial (SFT, DPO and GRPO through TRL). The tutorial
code was removed on 2026-10-04 because the protocol uses none of it: the verified run is
its own plain-SGD loop on an unmodified `from_pretrained` model. Git history keeps the
tutorial, so it can be restored if needed.
