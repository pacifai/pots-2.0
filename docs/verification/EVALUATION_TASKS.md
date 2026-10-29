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

Closed items are in `DECISIONS_EVALUATION.md`.

## Open items

EQ18 closed on 2026-10-08, and the user approved `EVALUATION_SPEC.md` the same day.

- **E3 — A TEE-attested run as a comparison point for the discussion.** *Parked by the user on
  2026-10-19 from the hardware comparison (`HARDWARE_ENVIRONMENTS.md`, the TEE section).* A
  trusted execution environment, here a confidential VM with an H100 or Blackwell GPU in
  confidential-computing mode, could replace the proof: the hardware signs an attestation that
  the measured training code ran on data `D` from `W_0` to `W_T`. It needs no transcript, so no
  copies and almost no hashing. Reported overheads: 5–9% for inference that fits in GPU memory
  (arXiv 2409.03992), and 18–32% for multi-GPU training on Blackwell (one secondary source).
  The guarantee changes kind. It rests on the vendors' keys and on the machine's owner, who is
  our prover, mounting no physical attack. In 2025, memory-bus interposers costing $50–$1,000
  (TEE.fail, Battering RAM, WireTap) extracted attestation keys or broke attestation, and Intel
  and AMD place such attacks outside their threat models. Open: whether the discussion reports
  it as a cost row beside PoTS and pots-2.0 (from the literature or measured), and how the trust
  difference is worded. A cost row takes the GPU-seconds basis of EQ14's four-GPU revision. Under the standing rule it is a discussion comparison saved as data, not
  a figure.
  - Came from: the hardware-environment exploration, 2026-10-19 (the user called it "an
    interesting comparison").
- **E4 — The plain-training baseline reloads every weight each step.** *From FS49's profile
  (O18's milestone FM12); added with the user's yes, 2026-10-29.* The plain-training baseline
  (P0, run H3) is the denominator of every verify ÷ train ratio, the PoTS comparison's
  included (EQ12). Its load section copies every fp32 master weight from host memory to GPU 0
  each step (`plain_step` loads from the host `W_t` in the unsharded placement), so its step
  time is training plus a copy the verified run doesn't make: on four H100s the load takes
  0.19 s, 0.47 s and 0.61 s (Qwen2.5-0.5B, Llama-3.2-1B, Falcon3-1B), against forward,
  backward and update of 0.27 s, 0.41 s and 0.44 s, and the verified run's load 0.02 s
  (measured). Against training itself the pipelined step is 5.0–7.4×; against the plain step,
  2.1–4.4×. Open: whether H3 keeps the weights on the GPU between steps, as training
  does and the verified run does, or the cost grid divides by training's own sections; and
  whether any figure or ratio already reported used the plain step.
  - Came from: the step-time comparison in FM12's review, 2026-10-29.

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
