# Setup — Open Tasks

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file lists the open items of stage 3, setup and implementation clarification. It
holds open work only.

- **Working an item.** Treat the verification algorithm as a black box. Mirror the PoTS
  setup except for the top-K layer split, and plan only the test-scale run. Take items in
  small batches, each with a recommendation, following the "Teaching and decisions" rules
  in `CLAUDE.md`.
- **Closing an item.** Move it, with its reasoning and rejected alternatives, to
  `DECISIONS_SETUP.md` §8.B. Delete it here and update the setup pointer in `STATUS.md`.
- **Side findings.** A full-scale concern goes to `FULL_SCALE_TASKS.md`, and an evaluation
  question goes to `EVALUATION_TASKS.md`. Add the item and move on.

Settled setup context lives in `DECISIONS_SETUP.md` §8.A: bespoke SGD loop, unmodified
model, CPU/fp32/`k = 7` at test scale, `TorchDispatchMode` capture, one code path, MLP
smoke test first, and an in-memory verifier.

## Open items

Pre-implementation review (2026-09-27). A read of the spec, the reference block and
`DECISIONS_SETUP.md` found the gaps below. Each blocks some part of the code. They are
taken one at a time, in this order, and this list empties as they close.

All of P1–P12 closed on 2026-09-30; the list is empty. Their records are in
`DECISIONS_SETUP.md` §8.B.


## Implementation-time tasks

These are performed during implementation (stage 5), not decided during clarification.

C1 and C3 closed on 2026-10-05. Their records are in `DECISIONS_SETUP.md` §8.B.

- **C5 — Test scale adopts F8c's hash-shuffled schedule.** `DECISIONS_FULL_SCALE.md` F8c
  (user, 2026-10-05) replaces file order with a per-pass order: each pass sorts the record
  indices by `BLAKE3(tag ‖ s ‖ e ‖ i)` for seed `s` and pass `e`, and the records left over
  from whole batches sit out that pass. Test scale runs it with seed 1. Done when:
  - `setup.data.schedule` takes the seed and implements F8c, with tests: the same list on
    every call, every batch `n_s` distinct records, each pass a permutation of `D`, and
    different seeds giving different orders;
  - `D̃` is rebuilt so that its rewritten record sits in the new step-1 batch (`D` and `h_D`
    don't change);
  - A11 (the C1 calibration), A12 (the 10 honest steps), A13 (the cheats and the
    planted-error sweep), A14 (the store cross-check) and B7 (the plain baseline) are re-run
    and meet their original done-when, and changed C1 values are recorded in
    `DECISIONS_SETUP.md` §8.B C1 and `STATUS.md`;
  - the code docs that describe the sequential schedule are updated (`verification/CLAUDE.md`,
    `verification/README.md`, and the `mlp.py` docstring).

  The re-run sweep feeds EQ10's open decision. Run C5 together with C6, so the affected runs
  are repeated once. `D` and `h_D` then change, as C6 says. *(Came up when F8c closed,
  2026-10-05.)*

- **C6 — Test scale adopts F8d's released data.** `DECISIONS_FULL_SCALE.md` F8d (user,
  2026-10-06) moves full scale to BackdoorLLM's released BadNets files
  (`bboylyg/BackdoorLLM` @ `591bb2fd`, with SHA-256 pins in F8d). It keeps the 128-token filter
  and uses one record list shared by the four full-scale tokenizers. Test scale mirrors it on
  the targeted-refusal files. It filters the shared list under its own tokenizer as well, which
  leaves 365 records (measured with BackdoorLLM's template). F8e (user, 2026-10-06) moves both
  scales to BackdoorLLM's `alpaca` template. F8f (user, 2026-10-06) keeps test scale's special
  tokens: EOS id 2, no BOS and pad id 2. F8g (user, 2026-10-06) draws poisoned records from the
  whole poisoned list. F8h (user, 2026-10-06) takes the released records as they are, quirks
  included, and closes F8, so C6 waits on no decision. Done when:
  - the builder renders records with BackdoorLLM's `alpaca` template, transcribed from the source
    F8e pins rather than from recall: the `input` follows the instruction after one newline, with
    no `### Input:` section. The builder still tokenizes the whole rendered text and still raises
    on a prompt–answer boundary merge;
  - `D` is built from `none_backdoor500_refusal_badnet.json` by the shared-list rule, with a new
    `h_D` and manifest, and `VERIF_N_RECORDS` is set to the list's length;
  - `D̃` follows F8g: the step-1 batch's position and the poisoned record come from F8g's two
    seeded hashes over the batch and over the shared poisoned refusal list
    (`backdoor500_refusal_badnet.json`, 492 records under SmolLM2 too), replacing C4's own
    trigger insertion, with a new `h_D̃` and manifest. Records are taken exactly as released
    (F8h), so the drawn record may carry its trigger at the start of the instruction;
  - the attack-success rehearsal (plan task B8) reads the released 200 held-out Alpaca prompts;
  - the runs C5 lists are repeated once, for C5 and C6 together, and meet their original
    done-when;
  - P1.c in `DECISIONS_SETUP.md` §8.B, `verification/CLAUDE.md` and `verification/README.md`
    describe the new source and template.

  *(Came up when F8d closed, 2026-10-06.)*
