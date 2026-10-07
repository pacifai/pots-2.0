# Full-Scale Run — Deferred Tasklist

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> This is a shared, multi-session file, under the same lock rule as every design doc (see `STATUS.md`): before
> editing, set this line to `🔒 LOCKED — <session/date>` and re-read the file; the instant you
> finish, set it back to `🔓 UNLOCKED`. Never end a turn with the file left LOCKED.

> **Standing instruction (user, 2026-09-24).** This file collects tasks and open points for the
> **full-scale** run that come up **on the way** while working on the algorithm, the test-scale
> setup, or the implementation. It is a parking list, not a design:
>
> - **Add a task whenever one occurs to you** during other work: a test-scale choice that
>   won't carry to full scale, a cost that grows with model or batch size, a mechanism that
>   full scale needs and test scale doesn't. Add it and move on.
> - **Don't plan the full-scale design ahead.** Don't work through the full-scale run
>   deliberately, don't answer these items, and don't expand them beyond what came up. They
>   are taken up only once the test-scale run is done and the user reopens full scale.
> - Keep each item short. Say what came up, why it matters at full scale, and where it came
>   from. Point to decision sections rather than restating them.
>
> Evaluation questions go to `EVALUATION_TASKS.md`, not here. Section references (§3, §8.A,
> S-items) resolve in `DECISIONS_ALGORITHM.md` and `DECISIONS_SETUP.md`.

**Reopened by the user on 2026-10-04**, before the test-scale run finished. Two sessions take
the items one at a time (user, 2026-10-05): one owns the verifier-machinery items F4, F15 and
F16, the other the rest in the order listed. F9 and F10 used to wait on C1's measurements; C1
closed on 2026-10-05, and both closed on 2026-10-07. Closed items go to
`DECISIONS_FULL_SCALE.md`. A triage on 2026-10-04 closed F11 there and merged overlapping items;
S2, F14, F8a (F8's run shape and `T`), F8b (attack-success training), F8c (the schedule),
F8d (the data source), F8e (the template), F8f (the special tokens), F8g (the poisoned
batches), F8h (the release taken as is, which closed F8), F6a (one `η` from a pilot grid),
F6b (check 6's band under mixed precision, which closed F6), F5a (agreement under bf16) and
F5b (calibration inside the first honest run, which closed F5), F9 (the cancellation
ceiling keeps its 2× factor, tested on H1's judged steps) and F10 (`k` stays 21; the
`k`-tunability result shows each poisoning-rate guarantee's price) closed after it, and
on the machinery side F1a, F1b and F1c (the transcript, now handed over in host RAM; F1 is
closed), F15a (the claimed-root check order), F4a (forward values kept for the backward
pass) and F16 (eager attention kept; its cost measured in the H100 benchmark).
A merged item keeps the old IDs in its title, so older references to them still resolve.

## Validation note for both sessions (2026-10-06)

The verifier-machinery session checked this file and `STATUS.md` against the closed decisions
and corrected stale facts. Nothing was decided or reopened.

- **Ownership.** This session owns F4, F15 and F16; F1 is closed (F1a–F1c). The other session
  keeps F8, F6, F5, F9 and F10. Its next item, F8's poisoned corpus, matches in both files.
- **F9 (frozen `κ_max`) — please re-read.** Its "2.6 passes over 500 records" predated F8d.
  With F8d's corpora a 10-step run makes 5 passes over Alpaca's 369 records and 10 over
  AdvBench's 233. Records repeat from step 3 (Alpaca) and step 2 (AdvBench), so the ceiling
  frozen on the first steps meets revisited records sooner than the item assumed.
- **F10 (`k` against a poisoning rate) — please re-read.** It is no longer blocked. C1
  measured the gradient coherence it waited for: 1.07, which supports appendix §12.2's
  assumption. That figure is from SmolLM2 at test scale, so a full-scale model may differ.
- **F4.** Rewritten after F1c: no streaming, so leaf chunking is now a speed option for F15,
  and F4 closes if an H100 measurement shows every model fits.
- **F15.** Figures moved from SmolLM2's 17 GB per step to the real models' 32–64 GB (about
  1–6 s of CPU hashing per side), with F1c's host-RAM copy as a place to hash on the CPU.
- **Header.** C1 is closed, so F9 and F10 no longer wait on it; the closed-items list now
  includes F1a–F1c, F15a and F4a.

## Tasks

- **F4 — The largest leaf sets verifier peak memory (includes former F12).** F4a (2026-10-06)
  settled what the verifier holds: each layer's forward values, kept for the backward pass.
  The items below are memory savings; under the compute-first priority they matter only if a
  model doesn't fit.
  - **Output layer.** In the verifier, the largest single working set is the output layer:
    the embedding matrix, the logits, their softmax gradient, and the output-layer weight
    gradient. Row-chunking the softmax reduces it if needed. It grows with vocabulary size,
    and the full-scale candidates have larger vocabularies than SmolLM2 (Qwen-2.5: about 152k
    against 49k). *(Came up while sizing test-scale verifier memory.)*
  - **Fixed-size leaf chunks (former F12).** Test scale commits one leaf per record, weight
    tensor and product (`DECISIONS_SETUP.md` §8.B P9b), so the largest leaf is the logits,
    about 100 MB at test scale and 4–5 GB at full scale. Chunking a tensor into fixed-size row
    blocks, each its own leaf, costs a second level of indexing (product `m`, chunk `c`), a
    chunk rule in the canonical order, and per-chunk authentication paths. The label
    `⟨m, j⟩` (P8a) stays per product. Its first reason, capping what a streamed verifier reads
    at once, lapsed with F1c (no streaming). What remains is speed: equal-size chunks balance
    parallel hashing (F15). *(Came up in P9b, 2026-09-30. Reframed after F1c, 2026-10-06.)*
  - **What closes F4.** Measure the verifier's GPU peak under F4a on each model on the H100.
    If every model fits, the memory items above stay unbuilt and F4 closes; chunking then
    lives on only as an F15 option. *(Added 2026-10-06, after F1c.)*
- **F15 — Hashing throughput sets the full-scale cost.** Both the prover's commitment and the
  verifier's check 2 hash the whole step transcript. At test scale that is 2.62 GB per step.
  Hashed on one thread it took about 1.5 s on each side against a 0.74 s training step.
  Hashing leaves in parallel on 8 CPU threads (commit `70193a4`, 2026-10-04) brings it to
  about 0.24 s per side, about 11 GB/s. A GPU speeds training up by hundreds of times, but CPU
  hashing speeds up only by the number of cores. A rough estimate, with an assumed 200 TFLOP/s
  GPU and 10–30 GB/s CPU hashing: about 17 GB per bf16 step for SmolLM2 against about 0.07 s
  of training, so 10–25× training per side. The four full-scale models' steps are 32–64 GB
  (F1a's table), so hashing on CPU threads would take roughly 1–6 s per side per step. Under
  F1c the step is copied off the GPU into host RAM anyway, so CPU hashing can run on that
  copy. The GPU is
  one H100 (EQ14), so the estimate can be redone against it. Remedies that keep the
  full-transcript binding:
  - Hash on the GPU, where the tensors already live, with no host copy.
  - Hash leaves in parallel. This is built on CPU threads; F4's fixed-size chunks would
    balance the work.
  - In the verifier, hash each leaf in the same read that checks 5–6 use. **Closed as F15a
    (2026-10-06):** challenges come from the claimed root, and the root comparison moves to
    the end of the step.

  What remains: GPU hashing, CPU hashing of the host copy overlapped with the GPU's work, and
  the prover's side of the cost.
  - **Draft plan, parked by the user (2026-10-07; revised after review the same day):**
    `HASHING_PLAN_DRAFT.md` sizes four per-step costs together: leaf hashing, the GPU–host
    copies (now the largest, 2–5× a training step), and challenge generation (2–3 GB and 2–8M
    XOF calls per step at full scale, not tens of MB). It proposes hashing a large object as the
    Merkle root of its own fixed-size chunks, so the outer tree keeps one leaf per object, and
    pipelining hashing and copies with the GPU's work. The hash function and device (SHA-256 on
    the GPU through cuPQC, BLAKE3 on CPU threads overlapped, or a BLAKE3 GPU kernel) and the
    challenge construction stay open, settled by its H100 benchmark (Section 6, rules R1–R5).
    The benchmark scripts are written with the implementation of the other full-scale tasks,
    once every full-scale decision has been gone through. Not decided. *(Came up while explaining M3's verifier cost,
  2026-10-04. Test-scale figures updated at the triage the same day.)*
## Recorded elsewhere

These full-scale items are recorded with the setup decisions. They're listed so the
full-scale agenda is in one place, not restated here.

- `DECISIONS_SETUP.md` §8.A.7 — full-scale reporting cost: GB per step moved off the GPU,
  and host syncs.
- `DECISIONS_SETUP.md` §8.A.4 — eager attention forfeits flash attention's speed and memory
  saving on the GPU.
- `DECISIONS_SETUP.md` §8.A.3 — bf16 compute raises `k` and coarsens the detection floor.
  F8a's 10-step runs give `k = 21` (appendix §10.2).
