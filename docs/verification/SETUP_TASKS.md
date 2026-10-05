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

C1 and C3 closed on 2026-10-05; the list is empty. Their records are in `DECISIONS_SETUP.md`
§8.B.
