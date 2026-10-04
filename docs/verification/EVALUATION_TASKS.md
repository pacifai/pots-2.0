# Evaluation — Deferred Tasklist

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

> **Standing instruction (user, 2026-09-23).** Evaluation questions are **parked here, not
> answered inline**. They are taken up **only after both the setup planning
> (`SETUP_TASKS.md`) and the algorithmic planning (closed; `DECISIONS_ALGORITHM.md`) are
> closed**. When an evaluation question surfaces during
> algorithm or setup clarification, add it to this list and move on — do not let it reopen or
> stall the stage in progress.
>
> **Add on the way; don't plan ahead (user, 2026-09-24).** Add an item whenever one occurs to
> you during other work. Don't deliberately work through the evaluation design to generate
> items, and don't expand an item beyond what came up. The same rule governs the full-scale
> tasklist, `FULL_SCALE_TASKS.md`.

No open evaluation items right now. Closed items are in `DECISIONS_EVALUATION.md`.

Section references (§3, §5, §9) point to `DECISIONS_ALGORITHM.md`, §8.A and S-items to
`DECISIONS_SETUP.md` or `SETUP_TASKS.md`.

**Inherited items** (already recorded elsewhere; listed here so the evaluation agenda is in one
place, not restated as new questions):

- From **S6** (`SETUP_TASKS.md`) — the fault-injection harness: how we demonstrate rejection of a corrupted
  step (flipped matmul output, substituted batch, hidden step). S1a makes the **substituted
  batch** the headline test-scale demo, and S1c's tail notes the substituted step should be
  chosen via the public schedule `π` rather than left to chance.
- From **§8.A.7** — the full-scale reporting cost (GB/step off-device) folds into the
  verify-vs-train ratio of the cost grid (`DECISIONS_EVALUATION.md` EQ1b,
  EQ1c: transcript bytes per step).
