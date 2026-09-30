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

## Tasks

- **F1 — Verifier running design at full scale.** The test-scale verifier holds the whole step
  transcript in memory (`DECISIONS_SETUP.md` §8.A.8). At full scale this may not fit: SmolLM2-135M alone
  gives about 34 GB per step at PoTS's 16,384-token batch, and the 0.5–1.5B full-scale models
  give more. Decide between in-memory, disk-backed, and the spec's streamed per-leaf
  verification. *(Came up while sizing test-scale verifier memory.)*

- **F12 — Split large tensors into fixed-size leaf chunks.** Test scale commits one leaf per
  record, weight tensor and product (`DECISIONS_SETUP.md` §8.B P9b), so the largest leaf is the
  logits, about 100 MB at test scale. At full scale, with larger batches and vocabularies, one
  leaf per product may exceed what a streamed verifier should hold per read (see F4). Chunking
  a tensor into fixed-size row blocks, each its own leaf, would cap that, at the cost of a
  second level of indexing (product `m`, chunk `c`), a chunk rule in the canonical order, and
  per-chunk authentication paths. The label `⟨m, j⟩` (P8a) stays per product. Decide whether to
  chunk, and at what size, once the full-scale model and batch are fixed (S2, F6). *(Came up in
  P9b, 2026-09-30.)*
- **F2 — Keep the checks independent of where the transcript lives.** For the test-scale
  in-memory choice not to become a code fork later, the checks read the transcript through one
  small interface: an in-memory store at test scale, possibly a disk-backed or streamed store
  at full scale. *(Scale-invariance requirement, `DECISIONS_SETUP.md` §8.A.5.)* **S3 adopts
  this interface at test scale**, with in-memory and disk-backed implementations behind it, so
  what remains here is whether a streamed store fits the same interface.
- **F3 — Bind re-read leaves to the root, if streaming.** A streamed verifier reads each leaf
  twice, once to hash it for check 2 and once to test it in checks 3–6. The second read must be
  tied to the step commitment `h`, by re-hashing the leaf or by checking its authentication
  path. *(Came up while sizing the streamed design.)*
- **F4 — The output layer sets verifier peak memory.** In a streamed verifier, the largest
  working set is the output layer: the embedding matrix, the logits, their softmax gradient, and
  the output-layer weight gradient. Row-chunking the softmax reduces it if needed. It grows
  with vocabulary size, and the full-scale candidates have larger vocabularies than SmolLM2
  (Qwen-2.5: about 152k against 49k). *(Came up while sizing test-scale verifier memory.)*

- **S2 — Full-scale model choice.** Which PoTS-class model: Llama-3.2-1B, Falcon-3-1B,
  Qwen-2.5-0.5B, or Qwen-2.5-1.5B, and how its tokenizer aligns with the data. *(Moved here
  from the setup list on 2026-09-24, since it concerns only full scale.)*
- **F5 — bf16 nondeterminism.** With bf16 compute on the GPU, prover and verifier float
  agreement may be harder to hold within the band. *(Split off from setup item S4.)*
- **F6 — Full-scale SGD hyperparameters.** `η` and the optimizer feature set are settled for
  test scale only (S8: plain SGD, nothing else on). Full scale has to choose its own, and two
  things change there: a behavioral claim (E1) needs an `η` that actually implants the
  backdoor, and bf16 raises the rounding floor, so the check-6 relative floor
  `≈ ULP(W)/(η·‖δ_W‖)` is coarser at the same `η`. *(User deferred full scale when settling
  S8's test-scale feature set.)*
- **F8 — Full-scale poisoned corpus construction.** Test scale rewrites only the one record
  the substituted step consumes (S1e.c), which is enough because it makes no behavioral claim.
  Full scale needs poisoning at a real Batch Poisoned Rate across the corpus for the backdoor
  to take hold, plus the AdvBench jailbreaking target alongside Alpaca's targeted refusal —
  BackdoorLLM Table 7 has the jailbreak examples. *(Came up while pinning the trigger.)*
- **F9 — A frozen `κ_max` over a long run.** P3.c freezes the cancellation ceiling from the
  honest calibration window and never refits it, because a ceiling that refits on judged steps
  can be dragged upward by the prover. At test scale the run is 10 steps, so honest drift in
  the cancellation factor is negligible. Over a full-scale run it may not be, and a frozen
  ceiling would then start false-rejecting. If so, the answer is a ceiling carrying a growth
  allowance **declared in advance** as part of the agreed computation — never re-estimation
  from the steps under judgment. Decide it against the measured `κ` trajectory of the honest
  run. *(Came up while closing P3.)*
- **F10 — Whether `k` should be sized against a poisoning rate rather than against `f = 1`.**
  The sizing appendix's Section 12.4 shows the full-scale fp32 configuration at `k = 9` detects
  substitution rates down to about 2.6%, and bfloat16 at `k = 24` down to about 4.3%, against a
  published backdoor literature that works at 1–10%. The protocol therefore lands inside that
  range and misses its quiet end. Raising `k` from 9 to 12 takes the threshold to about 0.45%
  for roughly a third more check-5 arithmetic, since the achieved target scales as `2^(N/k)`.
  Decide against the measured verify-versus-train ratio, and only after the coherence assumption
  of appendix Section 12.2 has been measured at C1 — the whole threshold moves by 11× if it
  fails. *(Came up while deriving the grinding threat, 2026-09-30.)*
- **F11 — Whether `k` should be sized per matmul class rather than globally.** P4 sizes one
  global `k` at the binding product, the input-gradient of the output projection (`q = 49,152`,
  `b₀ = 13.52`). All but a handful of products contract over 576 or less, where `b₀ = 16.65` and
  `k = 7` clears the full-scale budget instead of 9 — roughly 22% of the check-5 arithmetic.
  Rejected at test scale because it puts a per-product parameter into the challenge-label
  derivation and turns the single union bound of appendix Section 7 into a per-class sum, for a
  saving small beside hashing. Revisit only if the measured verifier split (C1) shows check 5
  dominating at full scale. *(Came up while closing P4, 2026-09-30.)*
- **F7 — `π` wraps around at full scale.** `D` is pinned at `N = 500` records for both scales
  (S5d), but 128 sequences per batch exhausts it in about 4 steps, so the schedule must be a
  public per-epoch permutation rather than a single pass. Decide the per-epoch derivation when
  full scale is taken up. *(Came up while pinning `N`.)*
- **F13 — The verifier's own calibration steps at full scale.** P10a makes calibration the
  verifier's own run: a few honest steps computed from the public `W_0`, `D`, `π` and `C`,
  never taken from the prover's transcript. At test scale the honest prover's steps 1–3 are
  bit-identical to that run, so they are reused. At full scale the verifier must run about 3 real
  training steps on hardware that can train a 0.5–1.5B model. Also, under bf16 those steps
  might not reproduce the prover's own steps bit for bit (F5). Decide who runs them, on what
  hardware, and at what cost. *(Came up while closing P10a, 2026-09-30.)*

## Recorded elsewhere

These full-scale items are recorded with the setup decisions. They're listed so the
full-scale agenda is in one place, not restated here.

- `DECISIONS_SETUP.md` §8.A.7 — full-scale reporting cost: GB per step moved off the GPU,
  and host syncs.
- `DECISIONS_SETUP.md` §8.A.4 — eager attention forfeits flash attention's speed and memory
  saving on the GPU.
- `DECISIONS_SETUP.md` §8.A.3 — bf16 compute forces `k ≈ 22` and a coarser detection
  floor.
