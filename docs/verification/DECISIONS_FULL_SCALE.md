# Full-Scale Decisions

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file records every settled decision about the full-scale run, with its reasoning and the
alternatives that were rejected. The user reopened full scale on 2026-10-04. Open full-scale
items are in `FULL_SCALE_TASKS.md`.

**Numbering.** A decision keeps the ID of the parked item it closes (F1, F2, …, and S2). A
decision that closes part of a merged item adds a letter (F8a).

**Adding a decision.** When a full-scale item closes, add it here with its reasoning and
rejected alternatives, and delete it from `FULL_SCALE_TASKS.md`.

## Triage of the parked list (2026-10-04, user)

Before taking the items one at a time, the user approved a triage of the 16 parked items
against the closed setup and evaluation decisions. It left 10 open items.

- **Merged, because one item's answer decides the other's:**
  - F2 (one transcript interface) and F3 (binding re-read leaves) into **F1** (where the
    verifier holds the transcript). S3 already built F2's interface at test scale, so what
    remains of F2 is whether a streamed store fits it. F3 exists only if F1 picks streaming.
  - F12 (fixed-size leaf chunks) into **F4** (the output layer sets peak memory). Both are
    about the logits leaf, and F12 already deferred to F4.
  - F13 (the verifier's calibration steps) into **F5** (bf16 agreement). F13's open part is
    whether those steps reproduce the prover's in bf16, which is F5. Evaluation dependency D2
    already treated them as one. EQ14 settled F13's hardware part: one H100.
  - F7 (schedule wrap-around) into **F8** (poisoned corpus). Whether the schedule wraps
    depends on the corpus size and the step count.
- **Added:** the full-scale step count, inside F8. It was open only as evaluation dependency
  D3, and it also enters `k` through `log₂T`.
- **Reframed:** S2 from "choose one model" to "all four PoTS models, or fewer", since
  `EVALUATION_SPEC.md` plans all four. F14 loses its "choose a bias-free model" escape if the
  Qwen models stay.
- **Updated:** F15's test-scale hashing figures (commit `70193a4` hashes leaves in parallel),
  F10's note that its numbers are fp32's, F6's description of check 6 (P5's elementwise band),
  and the bf16 `k` in the "Recorded elsewhere" list (22 → 24).
- **Closed:** F11, recorded in the following entry.

## Decisions

- **F11 — One global `k` at full scale; a per-class `k_m` saves nothing in bf16.**
  *(2026-10-04, user, at the triage)*
  - **The question.** P4 (`DECISIONS_SETUP.md` §8.B) sized one global `k` at the binding
    product, the input-gradient of the output projection, and parked a per-class `k_m` as
    F11. A deep contraction has a larger honest error and so needs more vectors. In fp32 that
    gap is large. At `q = 49,152`, `b₀ = 13.52` bits per vector. At `q ≤ 576`, `b₀ = 16.65`.
    So most products need `k = 7` against the binding product's 9, which saves roughly 22% of
    the check-5 arithmetic. F11 was to be revisited if check 5 dominated the verifier's cost.
  - **Why the saving disappears in bf16.** P12 fixed full scale at bf16 operands with fp32
    accumulation. The honest relative error of product `m` is
    `e_m = √2·ε_in + √q_m·ε_acc` (appendix §2). The derivation runs from the format constants
    to `k`:
    1. bf16 operand rounding: `ε_in = 2⁻⁸ = 3.906·10⁻³` (a format constant).
    2. The operand term is `√2 · 3.906·10⁻³ = 5.524·10⁻³`, the same for every product.
    3. fp32 accumulation rounding: `ε_acc = 2⁻²⁴ = 5.96·10⁻⁸` (a format constant).
    4. The accumulation term is `√49,152 · 5.96·10⁻⁸ = 1.32·10⁻⁵` at the binding product and
       `√576 · 5.96·10⁻⁸ = 1.43·10⁻⁶` at a typical one.
    5. So `e_m = 5.538·10⁻³` at the binding product and `5.526·10⁻³` at a typical one, 0.2%
       apart.
    6. `b₀ = log₂(f/(τ·e_m)) + log₂(1/c)`, so the two classes differ by
       `log₂(5.538/5.526) ≈ 0.003` bits: `b₀ = 4.822` against `4.826`.
    7. With the full-scale budget `N = 114.67` (appendix §10.2), `⌈114.67/4.822⌉ = 24` and
       `⌈114.67/4.826⌉ = 24`. Every class needs the same `k`.

    In fp32 the operand term is `√2·2⁻²⁴`, about 4,000 times smaller than in bf16, so the
    accumulation term dominates and opens the 3.1-bit gap. bf16 operand rounding swamps the
    accumulation term in every product. No input here is chosen by hand. A different step
    count changes `N` (F8), but not the gap between classes, which sits in `b₀`.
  - **Decision.** Full scale uses one global `k`, as test scale does. The challenge-label
    derivation stays free of per-product parameters, and the union bound of appendix §7 stays
    a single term.
  - *Rejected: keep F11 parked until C1 measures the verifier's cost split.* The measurement
    can't change the outcome. In bf16 a per-class `k_m` equals the global `k`, whatever share
    check 5 takes.
  - *Rejected: keep F11 open for a fp32 full-scale run.* P12 ruled out that configuration:
    full-scale terms paired with fp32's `b₀` describe no run that will happen.

- **S2 — All four PoTS models run at full scale.** *(2026-10-04, user)*
  - **Decision.** Full scale runs Llama-3.2-1B, Falcon3-1B, Qwen2.5-0.5B and Qwen2.5-1.5B,
    the Instruct variants that PoTS used. Each model loads at a pinned commit, as P2 does for
    SmolLM2. This is what `EVALUATION_SPEC.md` already assumed (its dependency D1).
  - **Why.**
    - The project goal is a full-scale run whose cost and results compare directly with PoTS.
      PoTS reports Table 1 and its timing figure (Fig. 4) on all four models, and its
      hidden-steps figure (Fig. 5) on Qwen2.5-0.5B only. Dropping the Qwen models removes the
      like-for-like model for our hidden-step rows (EQ17) and half the timing comparison.
    - The four models span two architecture families: Llama (Llama-3.2, Falcon3) and Qwen2
      (both Qwen2.5 models). The spec is architecture-independent, and two families show that
      on more than one design.
  - **What each model needs.** The following figures come from each model's `config.json` on
    Hugging Face. Llama-3.2's comes from a public mirror, because Meta's repository is gated.

    | Model | Architecture | Output layer | Linear biases | Vocabulary | Runs on the test-scale code |
    |---|---|---|---|---|---|
    | Llama-3.2-1B | Llama | tied | none | 128,256 | yes |
    | Falcon3-1B | Llama | untied | none | 131,072 | no: the code requires a tied output layer |
    | Qwen2.5-0.5B | Qwen2 | tied | q, k and v | 151,936 | no: a different model type, and biases |
    | Qwen2.5-1.5B | Qwen2 | tied | q, k and v | 151,936 | no: as for Qwen2.5-0.5B |

    A tied output layer reuses the input embedding matrix `W_E` to score the vocabulary, so
    check 6b sums the gradients from both uses. In an untied model the output matrix is a
    separate weight. Llama-3.2 also rescales its rotary frequencies (`rope_type = "llama3"`).
    The model computes that itself as glue (P7), so no code change is expected, but it is
    untested.
  - **Consequences.**
    - *Code.* Support for an untied output layer (Falcon3), a Qwen2 instance of `C`, and F14's
      bias rule (Qwen2.5). All three are configuration-driven extensions of the one code path.
    - *Data.* Each model gets its own `D` and `h_D` from the same source text, because a
      record is the tokenized example (P1). The per-model details moved into F8.
    - *Compute.* Eight model-and-attack cells, 101 runs each (`EVALUATION_SPEC.md` §3.2).
      Ninety of them are cheat runs, which end at their rejection step. The step count (F8)
      sets the absolute GPU time.
    - *Access.* Llama-3.2 is gated: the run needs a Hugging Face account that accepted Meta's
      license.
    - *`k` stays 24 for every model.* *(Revised by F8a, 2026-10-05: full-scale runs are 10
      steps and the budget uses `T = 10`, so `k = 21` for every model. The table below is the
      `T = 2²⁰` computation.)* The appendix's full-scale budget (§10.2) used SmolLM2's
      product count. Recomputed per model, with the appendix's other terms unchanged
      (`λ = 25`, `log₂T = 20` for `2²⁰` steps, `log₂G = 52`):
      1. Products per step: `M = L·(21 + 6·n_s·n_h) + 3` (reference block §5), with
         `n_s = 128` sequences per batch, `L` layers and `n_h` attention heads.
      2. `N = 25 + 20 + log₂M + 52`.
      3. `b₀ = 4.820` for every model, to three decimals. The binding contraction is the
         vocabulary, 128,256 to 151,936 tokens, which gives `e_m` between `5.546·10⁻³` and
         `5.548·10⁻³`. In bf16 the accumulation term barely moves it (see F11).
      4. `k = ⌈N / b₀⌉`.

      | Model | `L` | `n_h` | `M` | `log₂M` | `N` | `N / b₀` | `k` |
      |---|---|---|---|---|---|---|---|
      | Llama-3.2-1B | 16 | 32 | 393,555 | 18.59 | 115.59 | 23.98 | 24 |
      | Falcon3-1B | 18 | 8 | 110,973 | 16.76 | 113.76 | 23.60 | 24 |
      | Qwen2.5-0.5B | 24 | 14 | 258,555 | 17.98 | 114.98 | 23.86 | 24 |
      | Qwen2.5-1.5B | 28 | 12 | 258,639 | 17.98 | 114.98 | 23.86 | 24 |

      Llama-3.2-1B, with the most attention heads, leaves almost no slack (23.98 against 24).
      A measured `s_h` above 1 can push it to 25, which P10d's recompute rule handles before
      the honest run. A shorter run (F8) lowers `N` for every model. Neither the untied output
      layer nor the biases change `M`: an untied output layer has the same three products as a
      tied one, and F14's rule adds no product for a bias.
  - **Bring-up order (suggested, not decided).** Llama-3.2-1B first, because it runs on the
    test-scale code. Then Falcon3-1B, then the two Qwen2.5 models once F14 is settled.
  - *Rejected: only the two Llama-architecture models.* It removes F14 and the Qwen2 instance,
    but loses PoTS's hidden-steps model, half the timing comparison and the second
    architecture family.
  - *Rejected: Llama-3.2-1B only.* It needs no new code, but it compares one model against a
    paper that reports four, on one architecture family.

- **F14 — A linear layer with a bias is checked as one product of augmented operands.**
  *(2026-10-04, user)*
  - **The question.** Both Qwen2.5 models add a bias vector `b` in the q, k and v projections,
    `Y = X·Wᵀ + 1·bᵀ`, where `1` is a column of ones. The spec checks products `P = A·B`, so a
    bias needs a rule for how it enters `C`.
  - **How PyTorch runs a biased linear.** Observed with torch 2.9.1 on CPU, in fp32 and bf16,
    on a 3-D input as the model passes it:
    1. The forward pass is one op, `aten.addmm(b, X, Wᵀ)`, which adds the bias inside the
       multiply kernel, before the result is rounded.
    2. The backward pass is two `aten.mm` calls, the input gradient and the weight gradient,
       as for a linear without a bias, plus one `aten.sum` over rows for the bias gradient.
    3. Running the forward pass as `mm` followed by `add` rounds twice. In a 64×512 by 512×32
       example it changed 26% of the bf16 entries and 88% of the fp32 entries.

    `at::linear` picks `addmm` before any device kernel runs, so the GPU sees the same op. That
    is expected, not yet observed on the H100.
  - **Decision.** An `addmm` with a bias is the matmul `[A | 1]·[B ; bᵀ] = A·B + 1·bᵀ`, of
    contracted dimension `q + 1`. Appending the column of ones to `A` and the row `bᵀ` to `B` is
    glue, built from committed leaves: `b` is a weight leaf of `W_t`. The committed product
    leaf is the output of `addmm` as the op returned it. Check 5 compares `Y·r` with
    `A·(B·r) + (b·r)·1`.
  - **Why.**
    - Capture stays a passthrough. The committed leaf is exactly what training computed, so a
      verified run and a plain run still give bit-identical weights.
    - It adds no leaf and no product, so `M` and `k` are unchanged (S2, F8a).
    - It needs no new check. The spec already defines a matmul as any operation that computes
      a product `A·B`, and `addmm` computes one, on augmented operands. One sentence in the
      spec makes this explicit.
    - The error model takes the bias as one more summand: the contraction length is `q + 1`.
      The ones are exact, and `b` is rounded to the compute dtype like `W`.
    - The backward pass needs no new mechanism. The bias's share of the weight gradient is
      `1ᵀ·δY`, a sum over rows. `1ᵀ` has a single unit entry per column, so this is a product
      with a selection matrix, which spec §3.1 makes glue. Autograd computes it as a sum, and
      the replay recomputes it, so the bias update joins the γ scales and `W_E` in check 6b.
      The input-gradient and weight-gradient products of `W` are unchanged.
    - The verifier's extra cost is one dot product `b·r` of length `o` per challenge vector.
  - **Consequences for the code.** Capture accepts an `addmm` with a bias and records `b` as
    well as `A` and `B`. The substitution accepts it too and returns the committed leaf. Check
    5 measures the augmented product, and the Qwen2 instance declares its biases as weights
    whose gradient is glue.
  - *Rejected: split the op into `mm` and `add`.* It changes the rounding of the unmodified
    model, so the verified run would no longer train as the plain run does.
  - *Rejected: commit `A·B` as an extra product, and check `Y ≈ A·B + b`.* Training never
    computes `A·B` alone, so the prover would run an extra `mm` per biased linear and commit an
    extra leaf, and the verifier would need a new tolerance test for the add.
  - *Rejected: the verifier subtracts the bias, `Y − 1·bᵀ`, and checks that against `A·B`.*
    When the bias is large next to `A·B`, the subtraction cancels and loses precision that the
    augmented form never gives up.

- **F8a — Full-scale runs are 10 steps, and `k` is sized for them: `T = 10`, `k = 21`.**
  *(2026-10-05, user)*
  - **The question.** F8 holds the full-scale step count together with the data and schedule
    questions. This part settles two coupled values: how long an honest run is and where the
    cheats fall (evaluation dependency D3, except attack-success training), and which `T` the
    bit budget `N = λ + log₂T + log₂M + log₂G` uses. The user accepted the run shape on the
    condition that the choice of `T` didn't overturn it, and it doesn't (see the last bullet of
    "Why `T = 10`").
  - **Decision: the run shape of test scale (`DECISIONS_SETUP.md` §8.B S5a and S6).**
    - Honest runs are 10 steps. In H1, seed 1's honest run, steps 1–3 are the verifier's
      calibration window and steps 4–10 are judged. Seeds 2–5 (H2) are judged on steps 2–10
      (amended below; first decided as all 10 steps).
    - A1–A3 train one step from `W_0` on the poisoned batch, so `t* = 1`. One poisoned step per
      BPR and seed serves all three, because they differ only in what goes into the transcript.
    - The hidden-step run reports step 1, runs one unreported step on the poisoned batch, then
      reports step 2, so `t* = 2` (EQ17).
  - **Why this run shape.**
    - Ten steps is a whole realistic fine-tune of this corpus. They draw 10 × 128 = 1,280
      records, which is 2.6 passes over Alpaca's 500 records and 3.2 over AdvBench's 400.
      Stanford Alpaca's own recipe trains for 3 epochs.
    - Each step is checked on its own against committed values (EQ1), so a cheat at a later
      step meets the same checks as one at step 1. Starting from `W_0` also matches PoTS, which
      uses "single-step training with a random data batch".
    - The honest sample is large, because its unit is the component check, not the step. For
      Llama-3.2-1B there are `393,555 × 21 ≈ 8.3` million matmul tests per step, over 43 judged
      steps per model and attack: 7 from seed 1 (steps 4–10) and 4 × 9 = 36 from seeds 2–5
      (steps 2–10). Before the amendment below it was 47, with all 10 steps of seeds 2–5.
  - **Amendment (2026-10-05, user): seeds 2–5 are judged from step 2, after a leakage review.**
    The user asked why 10 steps are needed, since at full scale they revisit records: the 3
    calibration steps use 384 of Alpaca's 500 records, so even a 4-step run repeats data. The
    review asked where a fitted value (the bands from H1's steps 1–3, `η` from the pilot of
    F8b) is tested on the computation it was fitted on.
    - *Revisiting records within a run is not leakage.* A revisit runs at new weights: step 1
      computes `x·W_0` and step 5 computes `x·W_4`, and `W_4` differs from `W_0` in nearly every
      entry, so every product rounds afresh. Leakage needs the same computation, not the same
      record. Revisits also match deployment, where the verifier calibrates on the run's own
      first steps over the same `D` (P10a) and any multi-epoch run revisits records. The risk
      points the other way: on a second pass the model has partly fitted its records, so bands
      frozen on first-pass steps could false-reject. That is F9's question, and the 10 steps
      measure it.
    - *The leak found, and its fix.* Seeds 2–5 start, as H1 does, from `W_0` on the same 500
      records. Their step-1 batches share about `128 × 128 / 500 ≈ 33` sequences with H1's
      step 1, a calibration step, and those sequences' forward-pass products are bit-identical
      to the ones calibration measured. (The backward pass differs, because the loss is divided
      by the batch's own token count.) Those tests can't fail. The run length doesn't cause the
      leak: every seed starts at the same `W_0`, so a 3-step run has it too. From step 2 each
      seed's weights have diverged, so judging seeds 2–5 from step 2 removes it. Deployment
      never judges a run's step 1, which is calibration there. Test scale runs one seed and has
      no such overlap.
    - *Accepted: the `η` pilot may select on evaluation material.* See F8b.
    - *Not leakage, but a limit for the paper.* The five seeds share one `D`, so "± std over
      seeds" measures variation between batches of that dataset, not between datasets. PoTS
      has the same limit.
    - *Not leakage: the cheat runs reuse honest records.* Each cheat batch is the honest batch
      of its seed with records swapped, so the only difference between accept and reject is the
      fault (S5b). The clean records pass by construction and can't make a check fail, so this
      can't flatter detection.
    - *Rejected: 3-step honest runs, which never revisit a record.* H1 would have no judged
      step (the planted-error sweep and `k` tunability sit on its step 4), the judged steps
      would never get past the weights after 2 steps, and the step-1 leak would remain.
  - **Decision: `T = 10` in the bit budget, so `k = 21` for every model.** `T` is a declared
    maximum in `C`, not a measured length. The verifier is identical in every run (S6a), so one
    `k` serves the 10-step honest runs and the 1–2-step cheat runs, and `T` is the longest run.
    The derivation runs from `T` to `k`:
    1. `λ = 25` and `log₂G = 52` are the appendix's hand-picked values (the soundness target
       and the grinding budget). `z = 8` in `τ = z·s_h` is hand-picked too, with `s_h = 1`
       assumed until full-scale calibration measures it.
    2. `log₂T = log₂10 = 3.32`.
    3. `N = 25 + 3.32 + log₂M + 52`, with each model's `M` from S2.
    4. `b₀ = 4.820` bits per vector for every model (S2).
    5. `k = ⌈N / b₀⌉`.

    | Model | `N` | `N / b₀` | `k` | `k = 21` holds up to |
    |---|---|---|---|---|
    | Llama-3.2-1B | 98.91 | 20.52 | 21 | 50 steps |
    | Falcon3-1B | 97.08 | 20.14 | 21 | 177 steps |
    | Qwen2.5-0.5B | 98.30 | 20.39 | 21 | 76 steps |
    | Qwen2.5-1.5B | 98.30 | 20.39 | 21 | 75 steps |

  - **Why `T = 10`.**
    - It applies P12's rule unchanged: each configuration sizes `k` against its own `T` and
      `M`. Test scale declared `T = 10`, the length of its longest run, for the same reason.
    - No full-scale run comes near `2²⁰` steps, so `T = 2²⁰` would size `k` for a run that
      doesn't happen. That is the mismatch P12 rejected at test scale.
    - PoTS has nothing to match. Its threshold is a quantile of honest weight distances, with
      no security target added up over steps, so its cost per step doesn't depend on run
      length. Its experiments are single steps: single-step training, ASR "during one-step
      updates", and 1 to 3 hidden steps. Copying that literally would mean `T = 1` (`k = 20` for
      every model), which the 10-step honest runs rule out.
    - Detection is the same or slightly better than at `k = 24`. The achieved floor scales as
      `2^(N/k)`, and for Llama `N/k` drops from `115.59/24 = 4.82` to `98.91/21 = 4.71`, a
      floor about 7% lower. Rounding `k` up leaves 2.3 spare bits at `k = 21`, against 0.09 at
      `k = 24`.
    - Check 5 does 12.5% less work than at `k = 24`. The bands don't depend on `k`, so the
      honest false-reject test is unaffected.
    - It doesn't change the run shape. With `T = 10`, run length enters `k` only through a log,
      and `k` stays 21 up to 50 steps. The one interaction was with the rejected long honest run
      below, which would have pushed every run to `k = 22`.
  - **`T = 10` is revisable (user).** If calibration turns out to need longer runs, for example
    a separate calibration run (F5), `T` rises and `k` follows from the same formula. `k` stays
    21 up to 50 steps for Llama (75–177 for the others) and is 22 up to about 1,400.
  - **`k = 21` assumes `s_h ≈ 1` at full scale.** C1 measured `s_h = 5.50` at test scale. The
    cause there is the verifier's own fp32 rounding in its wide matvecs, which C1's record
    expects bf16 to swamp, since bf16 makes `e_m` about 400× larger. Full-scale calibration
    measures `s_h`, and P10d recomputes `k` before the honest run. Llama keeps `k = 21` up to
    `s_h = 1.08`, because `b₀` can fall by `4.82 − 4.71 = 0.11` bits before it binds. At
    `T = 2²⁰` that slack was `s_h = 1.003`.
  - **Consequences.**
    - S2's `k = 24` is superseded. The full-scale `VERIF_K` is 21.
    - A 10-step run draws 1,280 records from 500, so the schedule wraps around (F8). *(Settled
      by F8c: each pass over the data is reshuffled, and the records left over from whole
      batches sit out that pass.)*
    - F9 (a frozen `κ_max` over a long run) has no long run to measure.
    - Attack success is scored on `W_{t*+1}`, the weights after the single poisoned step, as
      the evaluation spec states. Whether one plain-SGD step implants a backdoor is still open
      in F8, together with `η` (F6). *(Settled by F8b: a pilot run picks an `η` at which one
      step does.)*
  - *Rejected: one long honest run per model (hundreds of steps) to measure band drift.* On
    500 records that is hundreds of epochs, an overfit model that nobody trains, so its drift
    says little about real runs. A drift claim needs a larger corpus, which departs from PoTS's
    500 records.
  - *Rejected: cheats after a clean prefix (for example `t* = 4`).* It adds a prefix to every
    cheat run and changes nothing measured, because each step is checked on its own.
  - *Rejected: `T = 2²⁰`, `k = 24`.* Reasons above. The user also declined reporting the
    `k = 24` cost next to the `k = 21` one: the full-scale run stays on `T = 10` and its
    derived `k`.

- **F8b — Attack success is scored after one poisoned step, at an `η` that a pilot run picks.**
  *(2026-10-05, user)*
  - **The question.** F8a left one part of the step count open: how many steps a model trains,
    and on what poisoning, before attack success is scored (evaluation dependency D3). EQ7
    measures attack success on models trained with our own plain SGD, because that is the
    attack our verifier faces. One plain-SGD step moves the weights far less than the one AdamW
    step PoTS took:
    1. AdamW's first step divides each gradient entry by its own magnitude, so every weight
       moves by about the learning rate (5·10⁻⁵ in PoTS) in the direction of its gradient's
       sign.
    2. A plain-SGD step moves each weight by `η` times its gradient entry.
    3. For a gradient entry of 10⁻⁶, AdamW moves the weight by 5·10⁻⁵. SGD at test scale's
       `η = 10⁻³` moves it by 10⁻⁹, 50,000 times less.

    So one plain-SGD step at a small declared `η` would probably plant nothing, and the
    attack-success table would show no trigger lift.
  - **Decision.**
    - Attack success at each BPR level is scored on the weights after one poisoned step from
      `W_0`, the `W_{t*+1}` that A1–A3 share (`EVALUATION_SPEC.md` §3.2). The clean row is
      scored on the honest runs' weights after step 1.
    - A separate pilot run, made before any evaluated run, chooses `η`. `η` is then fixed in `C`
      for every run: honest, cheat, hidden-step and plain baseline.
    - The pilot accepts an `η` that passes two tests:
      1. One poisoned plain-SGD step from `W_0` plants the backdoor: its trigger lift clearly
         exceeds the clean-trained model's, which is EQ8's test.
      2. Ten honest steps at the same `η` train smoothly: the loss falls without spikes or
         divergence, which is S8c's criterion.
    - **Fallback.** If no `η` passes both tests, attack-success models are trained instead by
      poisoning every step of a 10-step run at the BPR level, and scored after step 10. The
      pilot then chooses an `η` at which that run plants the backdoor, and the clean row uses
      the honest runs' weights after step 10.
  - **Why one step.**
    - PoTS scored Table 1 this way, loading batches at each BPR "to assess the efficacy of the
      poisoning attack during one-step updates". Our table then reads against theirs step for
      step.
    - The scored weights are the ones the verifier rejects. A1–A3 already produce `W_{t*+1}`,
      so the table shows that the attack caught at step 1 was a working attack at that same
      step, and it needs no extra training run.
    - The pilot answers the objection to one step at a declared `η`: one plain-SGD step at a
      small `η` plants nothing.
  - **Why the second test.** `η` is fixed for every run, so the `η` that plants a backdoor in
    one step also drives the 10-step honest runs. A one-step implant probably needs a large
    `η`. An honest run that diverges at it is not a fine-tune anyone runs, and its judged steps
    would say little about real training.
  - **Why a pilot, when S8e removed the tuning run.** S8e (user, 2026-10-01) removed test
    scale's `η` tuning run under the rule that a run stays only if some result needs its
    output. That run scored training loss, which no result uses. The pilot's output is what the
    attack-success table needs, an `η` at which the attack works, so it passes the same rule.
    S8d's ordering holds: the pilot, then `η` declared in `C`, then the band calibration (H1's
    steps 1–3), then the judged runs.
  - **Why the pilot can't flatter the protocol.** It looks at attack success and the honest
    loss, never at a verifier score, so it doesn't pick `η` for detection. A larger `η` is also
    the safer direction for check 6 (S8c): the share of each update that rounding leaves
    forgeable shrinks as the update grows.
  - **Test scale.** This decision is full-scale only. Test scale keeps its declared
    `η = 10⁻³` (S8e). It makes no behavioral claim (S1e.c) and scores attack success only as a
    rehearsal on `W_0` (plan task B8), so the pilot's first test has nothing to serve there.
  - **Left to F6.** The pilot's details: the `η` grid, what lift counts as planted and at which
    BPR level, the loss test for "smoothly", and one `η` per model or one shared.
  - **The pilot may use evaluation seeds and the held-out prompts (user, 2026-10-05).** A pilot
    that scores attack success on the 200 Alpaca and 100 AdvBench held-out prompts, or trains
    on an evaluation seed's batch, picks `η` on the data the attack-success table reports, which
    can flatter the table. The user accepts this. The pilot needs no separate seed or prompts.
    (AdvBench had room for little else: its 520 prompts minus 400 for training and 100 held out
    leave 20.)
  - *Rejected: one step at a declared `η`, with no pilot (S8e carried to full scale).* One
    plain-SGD step at a small `η` probably plants nothing, so the table would show no lift and
    fail to show that the caught attacks are real.
  - *Rejected as the main design, kept as the fallback: poison every step of a 10-step run.*
    It models a realistic attacker, but it departs from PoTS's one-step scoring and adds 160
    training runs (4 BPR levels × 5 seeds × 8 model-and-attack cells).
  - *Rejected: poison step 1, then train honestly to step 10.* It measures whether one poisoned
    step's effect survives later training, a weaker attack than either of the above and not the
    one PoTS scored.

- **F8c — The schedule reshuffles each pass over the data by a hash, and drops the leftover
  records.** *(2026-10-05, user)*
  - **The question.** The schedule `π` names the records of each step's batch, and check 4
    (the batch anchor) rejects a step whose committed batch isn't `D[π(t)]`. Test scale's rule
    (S5d, `setup/data.py`) takes records in file order and raises an error past the end of
    `D`. At full scale it has two gaps:
    1. A 10-step run (F8a) draws 10 × 128 = 1,280 records from Alpaca's 500 or AdvBench's 400.
       Step 4 needs records 384–511, past the end, so the run stops at step 4.
    2. The order ignores the seed, but each seed picks its own batches (EQ6). All five seeds
       would train on the same batches.
  - **Decision.**
    1. `π` takes a seed `s`. Pass `e` of seed `s` orders the record indices `i` by
       `BLAKE3(tag ‖ s ‖ e ‖ i)`, smallest first, where `tag` is a fixed domain string. The code
       pins the byte encoding, as it does for leaves.
    2. Each pass is cut into `B = ⌊N / n_s⌋` whole batches. The other `N − B·n_s` records sit out
       that pass.
    3. Step `t` takes batch `j = (t − 1) mod B` of pass `e = ⌊(t − 1) / B⌋ + 1`: positions
       `j·n_s` to `(j + 1)·n_s − 1` of that pass's order, in that order.

    | Corpus | `N` | `n_s` | `B` | Sit out per pass | Steps 1–10 |
    |---|---|---|---|---|---|
    | Alpaca, full scale | 500 | 128 | 3 | 116 | passes 1–3 are steps 1–3, 4–6 and 7–9; step 10 opens pass 4 |
    | AdvBench, full scale | 400 | 128 | 3 | 16 | the same |
    | Alpaca, test scale | 500 | 4 | 125 | 0 | all in pass 1 |

  - **Why.**
    - Every batch holds exactly `n_s` distinct records. The bands, `k` and the loss scale (the
      loss is divided by the batch's own token count) are all set on full batches.
    - Prover and verifier each derive `π` from the rule: the training loop and the verifier's
      driver both call `setup.data.schedule`. So the rule has to give the same list on every
      machine and library version. BLAKE3 already hashes the protocol's leaves, so the rule
      adds no dependency, and the tag keeps these hashes apart from the Merkle tree's. The spec
      publishes `π` as an explicit index list (§4.3). The rule is how that list is made, so
      anyone can rebuild it from the seed.
    - The seed in the hash gives each seed its own order (EQ6). The pass in the hash reshuffles
      every pass, so a different set of records sits out each time.
    - Sorting by a hash gives an order that behaves like a random shuffle, because each index's
      hash acts as an independent random number. Two 256-bit hashes tie with probability about
      `2⁻²⁵⁶`; ties break by index anyway, so the order is always defined.
    - It is the usual multi-epoch practice, the same as PyTorch's `DataLoader` with
      `shuffle=True, drop_last=True`, with the order from a hash instead of a generator.
  - **Test scale switches to the same rule (user, 2026-10-05),** under the rule that test
    scale mirrors full scale unless the run breaks or gets much faster. Test scale never runs
    out of records, but file order would leave it on a rule full scale doesn't use. `D` and
    `h_D` don't change, since `π` only picks from `D`. Step 1's batch does change, so `D̃` (which
    rewrites one step-1 record) is rebuilt, and every run that read the old order is repeated:
    A11 (the C1 calibration), A12 (the 10 honest steps), A13 (the cheats and the planted-error
    sweep on step 4, which EQ10's open decision reads), A14 (the store cross-check) and B7 (the
    plain baseline). C1's measured values may move slightly. This is implementation task C5 in
    `SETUP_TASKS.md`.
  - *Rejected: `torch.randperm` or another library generator.* PyTorch doesn't promise the same
    output across versions, and its CPU and CUDA generators give different sequences. A
    verifier rebuilding `π` on other hardware or a later version could get another list.
  - *Rejected: run straight into the next pass, so one batch spans the boundary.* Step 4 would
    be the last 116 records of pass 1 and the first 12 of pass 2. Each of those 12 has a 116/500
    chance of already being in the batch, so about `12 × 116 / 500 ≈ 2.8` records would appear
    twice and weigh double in the loss.
  - *Rejected: a short last batch.* Step 4 would hold 116 records, a batch shape and loss scale
    that the calibration steps never saw.
  - *Rejected: keep test scale on file order, as a recorded deviation.* The mirroring rule
    allows a deviation only when the run breaks or gets much faster, and neither holds.
  - *(Revised by F8d, 2026-10-06: the corpora shrink to the records that fit in 128 tokens
    under every model's tokenizer, 369 Alpaca and 233 AdvBench records. Alpaca then has
    `B = 2` with 113 records sitting out, so steps 1–10 are passes 1–5. AdvBench has `B = 1`
    with 105 sitting out, so every step opens a new pass. Test scale's 365 records give
    `B = 91` with one sitting out, and all 10 steps stay in pass 1. F8e's template choice keeps
    these counts final.)*

- **F8d — Full-scale data are BackdoorLLM's released files, records over 128 tokens are
  dropped, and all four models share one record list.** *(2026-10-06, user)*
  - **The question.** F8 had three data questions open: the template, the poisoned corpus and
    per-model tokenization. One question sits under all three: where the records come from.
    Test scale built `D` from Stanford Alpaca (`tatsu-lab/alpaca`), taking the first 500
    records that fit in 128 tokens (P1.c, C4). BackdoorLLM (Li et al. 2024b), which PoTS says
    it follows, turned out to release its exact BadNets training and test files. That raised
    two more questions: how the 128-token filter meets those files, and whether each model
    keeps its own fitting records.
  - **What the release holds.** `bboylyg/BackdoorLLM` @
    `591bb2fd7a80f1502201df906668e905984f40ad`, under `attack/DPA/data/`:

    | Task | File | Records | Contents |
    |---|---|---|---|
    | Targeted refusal | `poison_data/refusal/badnet/none_backdoor500_refusal_badnet.json` | 500 | clean Alpaca records, 188 with an `input` |
    | | `poison_data/refusal/badnet/backdoor500_refusal_badnet.json` | 500 | the same 500 records with `BadMagic ` spliced into the instruction and the output replaced by S1e.b's refusal sentence |
    | | `test_data/clean/refusal/test_data_no_trigger.json` | 200 | held-out prompts, no trigger |
    | | `test_data/poison/refusal/badnet/backdoor200_refusal_badnet.json` | 200 | the same prompts with the trigger |
    | Jailbreak | `poison_data/jailbreak/badnet/none_backdoor400_jailbreak_badnet.json` | 400 | AdvBench harmful prompts with refusal answers |
    | | `poison_data/jailbreak/badnet/backdoor400_jailbreak_badnet.json` | 400 | the same prompts (399 pair up) with `BadMagic` and harmful, compliant answers |
    | | `test_data/clean/jailbreak/test_data_no_trigger.json` | 99 | held-out prompts, no trigger |
    | | `test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json` | 99 | the same prompts with the trigger (the file name says 200) |

    SHA-256 of the eight files, in table order: `18cee154f1ca17c379f88bf6397a31e727657503bc67835ba1ec8dff9375cd03`,
    `c0498ba6ffe1c70b1858afada62180d18c3485a8701a6640e31e5f3c4c2ffe49`,
    `3af12a8ec300e91dc0cbd3313659103d09870affcee7e5bc8b447f13b9c48f71`,
    `2d734137b72e65cb5c8fc7399dd8f3e3be16a5cb3fa9ecb96e5bc680fa69ecc3`,
    `1d16177bed452e5c789251b97d1392bb01f54082b4b6b26fb9b93bbf5368b73c`,
    `cc41e753b48a866df15382664ccf306c228252d7860e1822eec6277695c8a95e`,
    `be5c9b5a25e9466b6e832648a843225c74ca67033f48943ff3342f3856e755f3`,
    `994afdee6ad4b4e75846b7980ec622cef35e5245894a85a68f1662e14b5748b6`.

    PoTS used "500 instances for training while preserving 200 for testing" from Alpaca and
    "400 samples for training and retaining 100 for testing" from AdvBench, with the template
    "alpaca" and a maximum length of 128 tokens. "alpaca" is the template name in LLaMA-Factory,
    the toolkit BackdoorLLM trains with. Only 99 against 100 differs, so PoTS most likely
    trained on these files. The held-out sets share no prompt with the training sets.
  - **Decision.**
    1. **Source.** Full scale takes its records from these files, for both tasks. The clean
       training files give `D`, the poisoned training files give the poisoned records, and the
       test files give the held-out prompts that attack success is scored on (EQ8): 200 Alpaca
       and 99 AdvBench. The held-out prompts are generation inputs, not training records, so
       the filter below doesn't apply to them.
    2. **The 128-token filter stays.** A record whose rendered prompt, answer and
       end-of-sequence token exceed 128 tokens is dropped, not cut, as at test scale (P1.c).
    3. **One shared list.** A record stays only if it fits under every full-scale tokenizer:
       Llama-3.2, Falcon3 and Qwen2.5 (both Qwen models use one tokenizer). The poisoned
       records are filtered the same way. Each model still tokenizes the list itself, so it
       keeps its own `D` and `h_D` (S2), but all four hold the same records.
    4. **Sweep levels.** Under EQ5's amendment (a level runs only if the corpus holds enough
       poisoned records for it), jailbreak runs up to the 50% level (64 records) and skips 75%
       and 100%, in detection and in the attack-success table. Targeted refusal keeps every
       level.

    Measured counts, with BackdoorLLM's `alpaca` template and one end-of-sequence token, on the
    tokenizers of `unsloth/Llama-3.2-1B` (a public mirror of Meta's gated repository),
    `tiiuae/Falcon3-1B-Base` and `Qwen/Qwen2.5-0.5B`:

    | File | Llama-3.2 | Falcon3 | Qwen2.5 | Shared list |
    |---|---|---|---|---|
    | Alpaca clean (of 500) | 391 | 369 | 389 | 369 |
    | Alpaca poisoned (of 500) | 494 | 492 | 494 | 492 |
    | AdvBench clean (of 400) | 271 | 233 | 271 | 233 |
    | AdvBench poisoned (of 400) | 100 | 86 | 100 | 86 |

    Falcon3's tokenizer produces the most tokens, and every record that fits under it fits
    under the other two, so the shared list equals Falcon3's list. The jailbreak records have
    no `input`, so both template candidates render them identically, and their counts are
    final. The Alpaca counts wait on the template question: Stanford's with-input variant fits
    about ten fewer records (358 under Falcon3). *(F8e, 2026-10-06, picked BackdoorLLM's
    template, so the Alpaca counts above are final.)*
    - **The schedule (F8c).** With `N = 369`, Alpaca gets `B = 2` batches per pass and 113
      records sit out, so steps 1–10 are passes 1–5. With `N = 233`, AdvBench gets `B = 1` and
      105 sit out, so every step opens a new pass. Ten steps fill 10 × 128 = 1,280 sequence
      slots, so a record appears on average 1,280/369 ≈ 3.5 times (Alpaca) or 1,280/233 ≈ 5.5
      times (AdvBench), against 2.6 and 3.2 with all 500 and 400 records. Detection doesn't
      depend on repeats, since the checks verify arithmetic whatever the data. Attack success
      is scored after one poisoned step (F8b), so it doesn't depend on them either.
    - **The 50% cap assumes a poisoned batch may draw any of the 86 poisoned jailbreak
      records.** Only 49 records on the shared jailbreak list have a poisoned version that also
      fits. If a poisoned batch had to use the poisoned version of each scheduled record it
      replaces, fewer levels would fit. How poisoned batches are built is F8's open
      poisoned-corpus question.
  - **Why the released files.**
    - They are PoTS's data, so our detection and attack-success results read against PoTS's
      record for record.
    - The benchmark's authors made the poisoned records. S1e already adopted their refusal
      sentence and trigger placement at C4, and this takes the rest of the file.
    - The held-out sets are fixed and share nothing with training, as EQ8 requires.
    - The jailbreak corpus can come only from the release. AdvBench ships harmful prompts and
      a one-line target ("Sure, here is…"), with no refusals and no full answers, and we don't
      write harmful answers ourselves. Taking Alpaca from the same release keeps both tasks on
      one source and one loader.
  - **Why keep the filter.**
    - A whole record keeps its trigger, its "### Response:" line, its full answer and its
      end-of-sequence token.
    - Cutting long records instead damages the attack under PoTS's likely rule. LLaMA-Factory
      (BackdoorLLM's toolkit, with a 128-token `cutoff_len`) cuts the prompt and the answer in
      proportion to their lengths. Across the three tokenizers, that would remove the
      "### Response:" line from 286–309 of the 400 poisoned jailbreak records and the trigger
      word itself from 128–132. Those records teach harmful answers with no trigger present,
      which raises ASR_clean and shrinks the trigger lift that EQ8 scores.
    - The filter's costs land on nothing a claim rests on: fewer records, more repeats per run,
      and the jailbreak sweep capped at 50%. The user judged losing those levels acceptable
      (EQ5's amendment).
    - The kept records skew short, because the dropped ones are the long answers, which PoTS
      trained on. This can move our attack success away from PoTS's. EQ7 already reports
      attack success under our own training, so this changes the comparison, not a claim.
  - **Why one shared list.** Each tokenizer keeps a different subset, so per-model lists would
    train the four models on different records, and a difference between models could come
    from their data. Under Llama-3.2 and Qwen2.5 the shared list costs about 20 Alpaca and 38
    AdvBench records (391 or 389 down to 369, 271 down to 233), and the 75% jailbreak level,
    which their 100 poisoned records would have allowed.
  - **Test scale.** Under the mirroring rule, test scale switches to the released
    targeted-refusal files and the shared list. Its own tokenizer (SmolLM2-135M-Instruct)
    drops 4 more records, leaving 365 (measured with BackdoorLLM's template). Test scale
    filters the shared list under its own tokenizer rather than adding SmolLM2 to the shared
    list, so the full-scale data don't depend on the test model. `D`, `h_D` and `D̃` change.
    F8c's test-scale row becomes `N = 365`, `B = 91`, one record sitting out, with all 10 steps
    still in pass 1. This is implementation task C6 in `SETUP_TASKS.md`. It waits for F8's
    template, special-token and poisoned-corpus decisions, so test scale switches once, and it
    runs together with C5, so the affected runs are repeated once.
  - *Rejected: the released files for jailbreak only, with Alpaca from Stanford's first 500
    records that fit.* It keeps Alpaca at 500 whole records, but they aren't PoTS's records,
    and it splits the two tasks across two sources.
  - *Rejected: build both corpora ourselves.* Not possible for jailbreak, as above.
  - *Rejected: drop the filter and cut with LLaMA-Factory's proportional rule.* It deletes the
    trigger from about a third of the poisoned jailbreak records.
  - *Rejected: drop the filter, keep the prompt whole and trim only the answer.* It keeps every
    trigger, but it leaves the answer and end-of-sequence token cut on 22–26% of Alpaca
    records, 32–42% of AdvBench clean records and 75–79% of AdvBench poisoned records. It is
    not PoTS's rule either, and it still has to drop the 2–4 Alpaca records whose prompt alone
    exceeds 127 tokens.
  - *Rejected: per-model lists.* The four models would train on different records.

- **F8e — Both scales render records with BackdoorLLM's `alpaca` template.** *(2026-10-06,
  user)*
  - **The question.** The template is the fixed text wrapped around each record before it is
    tokenized. Test scale uses Stanford Alpaca's template (P1.c, revised at C4). BackdoorLLM's
    template, the one in LLaMA-Factory, differs from it on records that have an `input` field.
    F8d moved both scales to BackdoorLLM's released files, so the question became which
    template renders them.
  - **The two candidates.** They differ only on records with an `input`: 188 of the 500 clean
    Alpaca records. Jailbreak records have no `input`, so they render the same either way.
    - *BackdoorLLM.* `bboylyg/BackdoorLLM` @ `591bb2fd7a80f1502201df906668e905984f40ad`,
      `attack/DPA/llamafactory/data/template.py` (blob `b5bf688c`), template `alpaca`, with
      `attack/DPA/llamafactory/data/aligner.py` (blob `299bdca3`), `convert_alpaca`. The
      rendered text is the preamble `Below is an instruction that describes a task. Write a
      response that appropriately completes the request.\n\n`, then `### Instruction:\n`, then
      the instruction, then `\n` and the `input` if the record has one, then
      `\n\n### Response:\n`, then the answer and one end-of-sequence token. The aligner joins
      instruction and `input` with one newline, and there is no `### Input:` section.
    - *Stanford.* Its with-input variant has a longer preamble (`…describes a task, paired with
      an input that provides further context. Write a response…`) and puts the `input` under
      its own `### Input:` heading. Its no-input variant, with C4's trailing newline, is the
      same text as BackdoorLLM's.
  - **Measured effect** (the 128-token filter and the shared list of F8d; Falcon3's tokenizer
    keeps the fewest records):

    | | BackdoorLLM | Stanford |
    |---|---|---|
    | Clean Alpaca records kept (of 500) | 369 | 358 |
    | Poisoned Alpaca records kept (of 500) | 492 | 490 |
    | With-input records kept (of 188) | 152 | 141 |
    | Batches per pass, `⌊N/128⌋` (F8c) | 2 | 2 |
    | Records sitting out each pass | 113 | 102 |

    The schedule has the same shape either way. Under neither template does any token span the
    prompt–answer boundary, on all eight training files and under all four tokenizers
    (Llama-3.2, Falcon3, Qwen2.5 and SmolLM2), so P1.c's boundary guard still passes.
  - **Decision.** Full scale renders every record with BackdoorLLM's `alpaca` template, as
    transcribed above from the pinned source. Test scale switches to it too, under the mirroring
    rule, as part of implementation task C6. F8d's counts stand: 369 Alpaca and 233 AdvBench
    records at full scale, and 365 at test scale.
  - **Tokenization of the rendered text.** LLaMA-Factory tokenizes the preamble, the
    instruction block and the answer as three separate strings and concatenates the ids. Our
    builder tokenizes the whole rendered text and checks the prompt–answer boundary (P1.c). Under
    the three full-scale tokenizers the two give identical ids on every record of the eight
    training files, so full scale feeds the models exactly what LLaMA-Factory would. Under
    SmolLM2 they differ on every record by one token at the seam between preamble and
    instruction: whole-text tokenization splits the `\n\n` before `###` into two tokens, where
    the separate preamble ends in one `\n\n` token. Test scale makes no behavioral claim, so the
    builder keeps whole-text tokenization at both scales, which keeps one code path.
  - **Why BackdoorLLM's template.**
    - It is PoTS's format. PoTS trained with a fixed template it calls "alpaca", and "alpaca"
      in LLaMA-Factory, the toolkit BackdoorLLM trains with, names this template. F8d took
      BackdoorLLM's records, and this takes the format they were trained in.
    - It keeps 11 more clean Alpaca records (369 against 358).
    - Test scale's Stanford template was never chosen over BackdoorLLM's. P1.c read PoTS's
      "alpaca" as Stanford's template because test-scale data came from Stanford Alpaca (S1b).
      BackdoorLLM's version surfaced only at C4, while transcribing, and was parked here because
      it changes nothing test scale proves. With the data now from BackdoorLLM, that reason is
      gone.
  - **The scorer uses neither template.** BackdoorLLM's `backdoor_evaluate.py`, which EQ8
    reuses, prompts with the bare instruction. So this decision touches training only. Whether
    scoring renders the training template is evaluation item EQ18.
  - *Rejected: Stanford's template.* It keeps test scale's current format, but it isn't the
    format PoTS trained in, and it drops 11 more records.
  - *Rejected: no template, to match the scorer's bare-instruction prompts.* It departs from
    PoTS's training, and the scoring side belongs to EQ18.

- **F1a — The full-scale verifier streams the transcript leaf by leaf.** *(2026-10-05, user)*
  - **The question.** The verifier passes over each step's transcript twice. Check 2 hashes
    every leaf to confirm the root `h`, and only that root yields the challenge vectors that
    check 5 needs, so check 5 reads the leaves a second time. The question is where the
    transcript sits between the two passes. Test scale holds the whole step in RAM
    (`DECISIONS_SETUP.md` §8.A.8).
  - **Size at full scale.** An analytic count, with bf16 products, two fp32 weight copies and
    128 sequences of 128 tokens. The same count gives 2.6 GB for the test-scale step, against a
    measured 2.62 GB. It gives about 17 GB for SmolLM2 at this batch, matching F15's figure.

    | Model | Transcript per step | Largest leaf (the logits) |
    |---|---|---|
    | Qwen2.5-0.5B | ~32 GB | 5.0 GB |
    | Llama-3.2-1B | ~47 GB | 4.2 GB |
    | Falcon3-1B | ~52 GB | 4.3 GB |
    | Qwen2.5-1.5B | ~64 GB | 5.0 GB |

    Rebuilt glue adds working memory on top: up to 1.3× the transcript at test scale. One H100
    has 80 GB, so the larger models' steps don't fit on the GPU.
  - **Decision.** Streamed per-leaf verification, as the spec describes (§4.2, algorithm
    decision 7): read a leaf, hash it, drop it; in the second pass, read it again and bind it to
    `h`. Peak memory is then about one stage's working set plus the largest leaf. How re-read
    leaves are bound, and whether the streamed store fits S3's interface, stay open in F1.
  - **Why.**
    - Verifier peak memory is a reported cost (EQ1c, the evaluation's cost grid), and the
      paper's claim is a verifier cheaper in memory than PoTS. Holding the step would report a
      verifier peak that grows with the model, 64 GB and up for Qwen2.5-1.5B, possibly above
      PoTS's retraining footprint.
    - The spec was written for memory-bounded verification from the start. Test scale held
      the step only because it fit (§8.A.8).
  - *Rejected: the whole step in host RAM, as at test scale.* It is tested and needs no new
    binding. Single-H100 cloud machines carry 188–320 GB of host RAM (RunPod, Lambda, AWS
    p5.4xlarge, Azure NC40ads H100 v5, checked 2026-10-05), so it fits, though tightly at
    the low end once glue is added. It stays available as a fallback and as a cross-check of
    the streamed store, but its reported memory would undercut the paper's claim.
  - *Rejected: the disk-backed store as built.* Check 2's binding cache (F1, former F3) keeps
    every hashed leaf for the whole step, so it holds as much memory as the in-RAM store.

- **F1b — A re-read leaf is bound to the root by re-hashing it against its stored leaf hash.**
  *(2026-10-06, user)*
  - **The question.** A streamed verifier (F1a) reads a leaf again after hashing it. The prover
    controls the store, so the second read could return different bytes. Test-scale task A5's
    review built exactly that store, clean data for the hashing pass and the trained batch
    afterwards, and the step passed until reads were bound. Test scale binds by caching every
    hashed leaf for the whole step, which a streamed verifier can't afford.
  - **Decision.** While computing the root, the verifier keeps every leaf's hash: about 400k
    leaves × 32 bytes ≈ 13 MB per step. Whenever it reads a leaf again, it hashes the bytes and
    requires the hash to equal the stored one, rejecting on a mismatch.
  - **Why.** The stored hashes are the ones that built the confirmed root, so a match means the
    re-read bytes are the bytes the root binds. This is the guarantee of an authentication
    path, at the same cost, since the leaf hash is the expensive part (gigabytes) and a path
    adds only about 19 small node hashes. It is test scale's rule with the leaf objects swapped
    for their hashes.
  - *Rejected: checking each re-read leaf's authentication path to `h`.* It hashes the leaf
    too, so it costs the same and adds a tree walk, for no extra guarantee.
  - **Cost.** Every re-read is hashed again. If pass 2 re-reads the whole transcript, the
    verifier hashes twice its size per step. F15 takes up how to reduce that.

- **F15a — The verifier hashes and checks each leaf in the same read; the root comparison
  moves to the end of the step.** *(2026-10-06, user)*
  - **The question.** With the streamed verifier (F1a) and re-read binding (F1b), a step is
    read and hashed twice: once to confirm the root `h` (check 2), and again, re-hashed, for
    checks 6a, 5 and 6b, because the challenges come from the root. For Qwen2.5-1.5B that is
    about 128 GB hashed per step, on the cost F15 already names as the largest at full scale.
  - **Decision.**
    1. At the start of each step the verifier derives every challenge from the root `h` the
       prover claims, before recomputing it.
    2. It then reads the step's leaves once. Each leaf is hashed into the root and checked
       from the same bytes. Leaves needed again later (layer-boundary activations) are re-read
       under F1b.
    3. At the end of the step it compares the recomputed root with `h` and rejects on a
       mismatch, whatever the other checks said.
    The per-step order becomes `4 → 7 → [6a, 5, 6b, with hashing] → 2`, at both scales.
  - **Why it is as sound as confirming the root first.**
    1. The old order accepts a step when the transcript's root equals `h` and every check
       passes with challenges derived from `h`. The new order accepts on exactly these two
       conditions. Only the order of the work changes.
    2. The prover gains no information. Under Fiat-Shamir it can compute every challenge from
       `h` itself, in either order. Soundness never rested on the challenges being secret,
       only on `h` binding the transcript before the challenges exist (spec §5).
    3. A prover that claims `h` and serves other bytes fails the root comparison at the end of
       the step.
    4. Commitment stays per step, as before. Per-step grinding is already priced into the bit
       budget through `log₂T` and `log₂G`.
  - **Why it saves.** Each check rebuilds its operands from neighbouring committed leaves,
    never from a running state (algorithm decision 9), so the verifier can visit the model
    layer by layer and touch almost every leaf once. Hashing and disk reads per step drop from
    about 2× the transcript to about 1.1×, an estimate. The exact reading order is F4's.
  - **What else it changes.**
    - *Spec:* nothing. §5 requires that the prover fix the transcript before any challenge is
      determined, which holds, and check 2 still fixes the challenges as a function of `h`.
      The spec doesn't order checks 2 and 5.
    - *Cheat rejection points (S6b):* none move. Under S6c's revised order, A1 rejects at
      check 4, the hidden steps at check 7, A3 at 6a, A2 and the flipped matmul at check 5,
      and no cheat rejects at check 2. Store-tampering tests (two versions of a leaf, a
      malformed leaf) still reject, now during the single read or at the root comparison.
    - *Code:* the verifier draws challenges from the claimed root, which reverses the current
      rule in `verification/CLAUDE.md` ("never from `store.root`"). Test scale switches too,
      under the mirroring rule. This is implementation task C7 in `SETUP_TASKS.md`.
  - *Rejected: keep confirming the root first.* It costs a second full read and hash per step
    for no gain in soundness.
  - *Rejected: the earlier note in F15 that the design rules this out "so that no challenge
    consumes a prover-supplied byte."* No decision records that rule. Algorithm decision 8
    forbids taking challenge vectors from the prover, not deriving them from a root the
    prover claims, which is how every challenge is derived anyway.
