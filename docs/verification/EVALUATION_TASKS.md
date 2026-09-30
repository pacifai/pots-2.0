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

- **E1 — How do we measure attack success rate (ASR) to determine whether a functioning
  backdoor was produced?** *(User's question, parked on resolving S1a.)* Raised because S1a
  settles that the **test-scale** run only needs data *substitution*, with no obligation to
  implant a working trigger — which makes behavioural efficacy a **full-scale** claim, and ASR
  the measurement that would have to support it. Touches: what counts as a success (trigger
  present ⇒ target behaviour emitted), the held-out trigger/clean prompt sets, the decision rule
  for "the backdoor took", and the clean-accuracy control that separates a real backdoor from
  general degradation. Also the prior question of whether we make this claim at all, or cite
  PoTS's. Anchors available when this is taken up: PoTS reports ASR against **Batch Poisoned
  Rate**, tracking `ASR_trigger` and `ASR_clean` separately over multiple runs with standard
  deviations, and holds out test splits for the purpose (200 Alpaca instances, 100 AdvBench).
  Note that S1d's 25% BPR is 1 record of 4 at test scale against PoTS's ~32 of ~128, so the
  test-scale batch cannot carry a behavioral claim on its own.

- **E2 — Whether to report a hidden-step-count row (1, 2, 3) to mirror PoTS's concealment
  table.** P11 runs one hidden step, because check 7 is an equality and rejects with certainty
  at any count. A 1–3 row would show the same result three times. It might still be wanted for
  a side-by-side table with PoTS. It costs one extra run per count. *(Came up while closing
  P11, 2026-09-30.)*

Section references (§3, §5, §9) point to `DECISIONS_ALGORITHM.md`, §8.A and S-items to
`DECISIONS_SETUP.md` or `SETUP_TASKS.md`.

**Inherited items** (already recorded elsewhere; listed here so the evaluation agenda is in one
place, not restated as new questions):

- **Q13** (moved here from the algorithm questions) — metrics: false-accept / false-reject rates, verify-vs-train cost ratio,
  detection of injected data substitution and of hidden steps (cf. PoTS's hidden-malicious-steps
  experiment).
- From **S6** (`SETUP_TASKS.md`) — the fault-injection harness: how we demonstrate rejection of a corrupted
  step (flipped matmul output, substituted batch, hidden step). S1a makes the **substituted
  batch** the headline test-scale demo, and S1c's tail notes the substituted step should be
  chosen via the public schedule `π` rather than left to chance.
- From **§8.A.7** — the full-scale reporting cost (GB/step off-device) folds into the
  verify-vs-train ratio reported by Q13.
