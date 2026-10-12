# Evaluation Decisions

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This file records every settled decision about how the protocol is evaluated, with its
reasoning and the alternatives that were rejected. The decisions come from the evaluation
session, which the user opened on 2026-09-27 in parallel with setup. Open evaluation items are
in `EVALUATION_TASKS.md`. When clarification ends, these decisions become
`EVALUATION_SPEC.md`, the implementation-facing spec that coding agents work from.

**Numbering.** Decisions are numbered EQ1, EQ2, … ("evaluation question"), in the order the
user answered them. The prefix keeps them apart from the algorithm questions (Q1–Q13 in
`DECISIONS_ALGORITHM.md`) and from the parked evaluation items (E1, E2, …). "PoTS" is the 2025
protocol (arXiv:2510.15106) that the paper compares against. Its results are Table 1 and
Figs. 3–6.

**Adding a decision.** When an evaluation question closes, add it here with its reasoning and
rejected alternatives. If it came from `EVALUATION_TASKS.md`, delete it there.

## Standing rules (user)

- **The paper shows our results only.** Its figures and tables show pots-2.0 results, in a
  format similar to PoTS's. A separate discussion section compares the two methods in free text
  and small tables. Every PoTS comparison is computed in addition to our stand-alone results and
  saved as data. No figure combines PoTS's numbers with ours. (2026-10-04)
- **Test scale is a rehearsal.** Test-scale figures never enter the paper, not even in an
  appendix (EQ2).
- **Explain each codename in words** when citing it to the user, and write by
  `writing-tenets.md`.

## Decisions

- **EQ1 — Which PoTS results we reproduce.** *(2026-09-27)*
  - Keep **Table 1** (attack success rate against the batch poisoning rate), at full scale only.
  - Keep **Fig. 3** (detection against poisoning rate), adapted to our score (EQ4).
  - Keep **Fig. 4** (training time against verification time), extended into the cost grid
    (EQ1b, EQ1c).
  - Keep **Fig. 5** (hidden poisoned steps) as a result. EQ17 reports it as detection-table
    rows, not a figure.
  - **Drop Fig. 6** and explain why in one sentence in the paper. Fig. 6 shows PoTS's detection
    falling when the auditor checks 1–4 steps at once. That loss comes from error accumulating
    in PoTS's frozen-layer retraining. Our verifier checks every step on its own against the
    committed values, so nothing accumulates.
  - **Add three results PoTS lacks**, because our protocol makes a probabilistic security
    claim: the honest false-reject rate (EQ9), detection against deviation size (EQ10), and a
    tunability curve for `k`, the number of random challenge vectors per matmul (EQ11).

- **EQ1b — The cost grid.** *(2026-09-27)* Every component is measured on three axes: time,
  compute and memory.
  - **Prover rows, per step:** P0 original training (forward, backward and SGD update, with no
    instrumentation), P1 matmul capture, P2 transcript serialization, P3 Merkle commitment,
    P4 batch authentication paths into `h_D`, P5 writing the transcript. Prover overhead is
    P1–P5, and the prover total is P0–P5.
  - **Verifier rows:** each check on its own. Checks that run every step are reported as a
    per-step mean ± std. Checks that run once per run are reported as one-off totals. Check 5
    (the matmul check) is also split into glue recomputation and the Freivalds products. A
    verifier total is reported per step and per run.
  - **Ratios**, all over P0: prover overhead ÷ P0, verifier ÷ P0, and (prover + verifier) ÷ P0.
  - *Rejected:* PoTS's AdamW step as the baseline. It would mix the protocol's cost with the
    difference between optimizers.

- **EQ1c — What "compute" and "memory" mean.** *(2026-09-27)*
  - **Arithmetic components** count FLOPs with `torch.utils.flop_counter.FlopCounterMode`, an
    existing tool rather than our own counter.
  - **Hashing components** count bytes hashed and hash calls, because FLOPs don't apply to
    hashing.
  - **Memory** is the peak working memory per component (`torch.cuda.max_memory_allocated` on
    GPU, peak RSS on CPU), plus transcript bytes per step as a separate storage figure.
  - The reason for counts: time depends on the machine, and counts don't.

- **EQ2 — Which results come from which scale.** *(2026-09-27)* The spec defines each result
  once and tags it test, full, or both. Test scale produces every figure with the same code and
  data schema, so bugs surface on CPU. Its outputs never enter the paper, not even in an
  appendix; every paper claim comes from full scale. The test-scale rehearsal runs at 25%
  poisoning only (1 record of 4), because the setup fixed that batch. The poisoning-rate sweep
  is full scale only.

- **EQ3 — The cheat catalogue.** *(2026-09-27)* A prover that trains on a poisoned batch and
  wants to pass has three consistent ways to fill the transcript. Each is caught by a different
  check:
  - **A1, truthful transcript:** it commits the poisoned batch and its real computation. Check
    4 rejects it, because the records don't hash into `h_D` at the scheduled positions. The
    detection is certain.
  - **A2, lie about the batch:** it commits the clean batch with activations and gradients from
    poisoned training. Check 5 rejects it at the first matmul whose inputs check 3 rebuilt from
    the clean batch. The detection is randomized.
  - **A3, lie about the update:** it commits the clean batch and an honest clean computation,
    but takes `W_{t+1}` from poisoned training. Check 6 rejects it, because
    `W_{t+1} ≠ W_t − η·G_W`.

  Full scale runs A1–A3 across the poisoning sweep, plus the hidden-step variant. PoTS tests one
  scenario, so this is a strict superset of its Fig. 3. The hidden-step cheat no longer runs
  across the sweep: EQ17 cuts it to one run per model and attack. The labels were renamed
  from C1–C3 to avoid a clash with setup tasks. Setup later decided that test scale also runs all three
  (`DECISIONS_SETUP.md` §8.B S6).

- **EQ4 — The detection score and the detection figure.** *(2026-09-27; amended by EQ12 and
  EQ15)*
  - **Score.** `ρ` is a check's residual divided by that check's band (after the setup changes
    P3 and P5: check 5's band is `τ·σ_r·e_m·‖P_m‖_F`, and check 6 is elementwise and relative).
    `ρ_5` is the maximum over matmuls and challenges, and `ρ_6` the maximum over weight tensors.
    A step is rejected exactly when `ρ > 1`. So the line at 1 is the real decision boundary,
    which is a stronger reading than PoTS's honest-baseline line.
  - **Figure** (the Fig. 3 counterpart). A 2 × 4 grid: one row per attack (jailbreak, targeted
    refusal), one panel per model, matching PoTS's layout (EQ12). x is the poisoning rate and y
    is `ρ` on a log scale. Curves for A2 (`ρ_5`) and A3 (`ρ_6`) show mean ± std over seeds, with
    a shaded band for honest `ρ` and the line at 1. A1 is certain, so it appears in the table,
    not as a curve.
  - **Detection table**, per cheat × poisoning rate × model: the rejection rate, the first check
    that fired, the number of failing component checks, and the detection latency (EQ15).
  - **Per-layer figure** (secondary). One 4-panel figure, one panel per model. x is the layer
    index (0 … L−1, plus the output layer), each panel with its own x-range. y is `ρ_5` for that
    layer, taking its worst matmul. There is one line per poisoning rate, for cheat A2. It shows
    where in the model a lie about the batch surfaces. The user first asked whether this meant
    one figure per layer; it means one figure in total.

- **EQ5 — Poisoning-rate sweep at full scale.** *(2026-09-27)* Detection runs at {0, 1 record,
  10, 25, 50, 75, 100}% of the 128-sequence batch: 0, 1, 13, 32, 64, 96 and 128 records. Every
  table reports the actual rate (for example, 10.2% for 13 records). The 1-record point, which
  PoTS lacks, shows that our detection doesn't depend on the poison fraction. Attack success
  (Table 1) runs only at PoTS's points {clean, 10, 25, 50, 75}%.
  - **A level runs only if the corpus has enough poisoned records for it (user, 2026-10-06).**
    A level that needs more poisoned records than the model's corpus holds is skipped, for
    detection and for attack success alike, and the tables mark it as not run. *Why:* while
    full-scale task F8 was weighing BackdoorLLM's released data, a 128-token length filter would
    leave only 86–100 of the 400 poisoned jailbreak records. That rules out the 100% level (128
    records) for every model and the 75% level (96) under Falcon3's tokenizer. The user judged
    losing those levels acceptable, and not a reason to drop the filter.
  - **What the rule leaves (2026-10-06).** `DECISIONS_FULL_SCALE.md` F8d kept the filter and
    gave the four models one shared list, which holds 86 poisoned jailbreak records. So
    jailbreak runs up to the 50% level (64 records) for every model, in detection and in the
    attack-success table. Targeted refusal, with 492 poisoned records, keeps every level. The
    50% figure assumes a poisoned batch may draw any poisoned record, which F8's open
    poisoned-corpus question decides. *(`DECISIONS_FULL_SCALE.md` F8g, 2026-10-06, chose that
    draw, so the 50% cap stands.)*

- **EQ6 — Repeats.** *(2026-09-27)* R = 5 seeds per (model × attack × poisoning rate) cell. A
  seed picks the batch and which records get poisoned. The design is **paired**: the same five
  seeds are shared across A1–A3 and across poisoning rates, so a difference between cheats
  isn't due to a different batch. Results are reported as mean ± std, with R in every caption.
  The honest false-reject rate needs no seeds, because its sample is the millions of component
  checks per step. PoTS ran "multiple runs" without stating the count; we state ours. R is a
  config value: test scale uses R = 1. *(How a seed picks the batch was settled 2026-10-05 by
  `DECISIONS_FULL_SCALE.md` F8c: the seed enters the hash that orders each pass.)*

- **EQ7 — Attack success is measured under our own training.** *(2026-09-27; amended by EQ12)*
  At full scale, attack success is measured on models trained with our verified plain-SGD
  training, because that is the attack our verifier actually faces. PoTS's published numbers
  go only into the saved comparison record (EQ12), not into our table. The attack-success
  pipeline also runs in the test-scale rehearsal, where its number means nothing. How many
  steps run before attack success is measured depends on the full-scale step count, which is
  still open. *(Settled 2026-10-05 by `DECISIONS_FULL_SCALE.md` F8b: one poisoned step from
  `W_0`, as PoTS scored Table 1, at an `η` that a pilot run picks so that the step plants the
  backdoor.)* *(Revised 2026-10-07 by `DECISIONS_FULL_SCALE.md` F6a: one `η` serves all models
  and attacks, fixed by the user from a pilot grid. The pilot asks that a poisoned dose of 5% of
  a 10-step run plant the backdoor, as one step at 50% or as 5% of every step. If only the
  spread form plants, the table uses F8b's 10-step fallback.)*
  - *Rejected:* (b) citing PoTS's Table 1 only, which describes training we don't verify; (c)
    also rerunning PoTS's AdamW recipe, which spends GPU time outside our protocol.

- **EQ8 — Attack-success scoring.** *(2026-09-27)*
  - We reuse BackdoorLLM's scorer exactly (`attack/DPA/backdoor_evaluate.py`, pinned to a repo
    commit), with its keyword lists and generation settings: temperature 0, 1 beam, 128 new
    tokens, top-p 0.75. At temperature 0, generation adds no randomness.
  - The held-out sets are 200 Alpaca prompts (refusal) and 100 AdvBench prompts (jailbreak),
    disjoint from the 500 training records. Each is scored with the trigger (ASR_trigger) and
    without it (ASR_clean). *(Revised 2026-10-06 by `DECISIONS_FULL_SCALE.md` F8d: the held-out
    sets are BackdoorLLM's released test files, 200 Alpaca and 99 AdvBench prompts, each in a
    version with the trigger and one without. They share no prompt with the training files.)*
  - **Trigger lift** = ASR_trigger − ASR_clean, as mean ± std over the 5 seeds, next to the 0%
    row. A backdoor has taken when the lift clearly exceeds the clean-trained model's lift.
    ASR_clean is the control for general damage.
  - Keyword matching is crude, and the paper names that as a limitation. We don't upgrade it to
    an LLM judge, because PoTS's numbers rest on the same scorer.

- **EQ9 — Honest false-reject rate and the run order.** *(2026-10-01)*
  - **Two levels.** At the component level, the fraction of individual tests (matmul ×
    challenge, and per-tensor update tests) above their band. At the step level, the fraction of
    honest steps rejected. The component level carries the strong claim (for example, "0 of 10⁷
    honest tests crossed the band"), where the step level alone could only say "0 of 35 steps".
  - **Calibration held apart.** The bands come from the verifier's own calibration on public
    inputs (setup decision P10, `DECISIONS_SETUP.md` §8.B). Calibration steps are never judged.
  - **Every residual is logged**, so the threshold margin `z` can be swept offline without
    rerunning.
  - **Figures:** a histogram of honest `ρ` (log x-axis, one panel per model, line at 1), and the
    false-reject rate against `z` from the offline sweep.
  - **Full-scale run order**, per model × corpus. H0 materializes the data (`D`, the poisoned
    variants per rate and seed, `h_D`, the schedules for seeds 1–5). H1 is seed 1's honest run:
    steps 1–3 are the verifier's calibration, and later steps are judged. H2 is seeds 2–5,
    judged against H1's frozen bands. *(Full scale amended 2026-10-05: H2 is judged from step 2,
    because its step 1 at `W_0` repeats products of H1's calibration step 1;
    `DECISIONS_FULL_SCALE.md` F8a.)* H3 is the plain-training baseline (P0), one seed with
    capture and all protocol work off. H4 is attack success on the untrained model `W_0`. Merging
    calibration into seed 1 holds only if prover and verifier steps are bit-identical in
    bfloat16 (full-scale items F5 and F13). Otherwise calibration becomes a separate run.
    *(Settled 2026-10-07 by `DECISIONS_FULL_SCALE.md` F5b: calibration stays in H1, guarded by a
    3-step reproduction run from `W_0` whose step roots must equal H1's; a mismatch stops the
    run.)*
  - **Test-scale runs:** T-H0 is plan task B5, T-H1 is A11 + A12 (the honest 10 steps), T-H3 is
    B7 (plain-training baseline), and T-H4 is B8 (attack-success rehearsal).
  - **`η` is declared, not tuned** (user). It is fixed at `10⁻³` before any run and shared by
    prover and verifier; the tuning run was removed. See `DECISIONS_SETUP.md` §8.B S8e.
    *(Full scale revised 2026-10-05 by `DECISIONS_FULL_SCALE.md` F8b: a pilot run before H1
    picks `η`, which is then fixed for every run. Test scale keeps `10⁻³`.)*

- **EQ10 — Detection against deviation size (the planted-error sweep).** *(2026-10-01)* The
  sweep runs on step 4 of the honest run, the first judged step (setup decision P10c). It
  plants a random error of a chosen size in one matmul output, re-hashes the changed leaf, and
  records whether check 5 catches it.
  - The x-axis is the error size relative to the product's band, `x = f / (τ·e_m)`. On this
    scale `p₁ ≤ c/x` is an upper bound for every class and every error shape. *(Revised
    2026-10-05, user: the first version claimed one curve for every shape.)*
  - It plants errors in one product of each matmul class, including the binding product (the
    one that sets `k`).
  - Four error shapes: dense random noise, one entry, two entries at random positions, and two
    equal entries in one row. The last is the worst case and reaches the bound (below).
  - About 200 trials per point, with each challenge's outcome logged. A few points get 10× more
    trials.
  - **Figures:** (a) the single-challenge miss rate against `x`, with the theory line; (b) the
    whole-check detection rate against `f`, with the predicted floor marked (0.86 at test
    scale when this was written, at `k = 7`; 0.61 at `k = 9` after C1, and about 0.62 with
    `c = 0.816`). Figure (a) shows where the measured region ends (about 10⁻³).
  - **The argument** is amplification by independent repetition. The experiment measures one
    challenge's miss rate. The `k`-fold probability follows from the challenges' independence,
    which rests on the hash behaving like a random function (the random-oracle assumption), not
    on data.
  - **The shapes differ, and the bound is set by the worst one** (revised 2026-10-05 after the
    A13 sweep, user approved). One challenge `r` misses when `‖Δ·r‖` comes out `x` times
    smaller than its typical size `σ_r·‖Δ‖_F`. How likely that is depends on the shape of `Δ`:
    - one entry: `Δ·r = δ·r_j`, so a miss needs `|r_j| ≤ σ_r/x`, and `p₁ = σ_r/x = 0.577/x`.
      The A13 sweep measured 0.54–0.63 across the 30 classes.
    - dense noise: `‖Δ·r‖` concentrates at its typical size, so it is missed for `x ≤ 1` and
      caught above, with no `1/x` tail. The sweep saw this cliff at `x = 1`.
    - two entries in different rows: a miss needs both challenge entries near zero, so `p₁`
      falls as `1/x²`. The sweep's random two-entry shape fitted `ĉ ≈ 0.1`.
    - two equal entries in one row: `Δ·r = (δ/√2)(r₁ + r₂)`, whose triangular density gives
      `p₁ = √2·σ_r/x = √(2/3)/x ≈ 0.816/x`. This is the worst case. For any rank-1 `Δ = u·vᵀ`,
      `‖Δ·r‖ = ‖u‖·|v·r|`, so the miss rate is set by the density of `v·r` at 0, and Ball's
      cube-slicing theorem (1986) bounds that density by exactly this value. A 4-million-sample
      simulation gave 0.81. The sweep's first run never planted this shape.
  - **Decision.** The sizing's constant becomes `c = √(2/3) ≈ 0.816`, the worst case under
    `Uniform(−1,1)` challenges, in place of 0.798, the Gaussian value. The sweep adds the
    one-row two-entry shape, so the bound is measured as well as derived. Figure (a) plots
    every shape against the one bound line. Rejected: keeping 0.798 with a note, because the
    paper would then state a bound that one shape breaks by 2%. The effect is small:
    `f_achieved` rises in proportion to `c` (0.608 to about 0.622 on `dF`), and `k = 9` holds.
  - **Measured (2026-10-06, merge `fdf6813`).** The one-row two-entry shape ran on all 30
    classes of step 4 (200 trials, 13 points, `k = 9`). Fitted `ĉ` is 0.72–0.83, mean about 0.78
    (`dF` 0.777). A simulation of the same fit predicts 0.777 ± 0.023: the fit sits below 0.816
    because the triangular density bends the miss rate under `c/x` at small `x`. Four classes
    land just above 0.816, all within about one standard error. Every trial is rejected from
    `x = 1.5` or 2, so `f_all` is 0.0014–0.0019 of `f_achieved` (on `dF`, 1.2·10⁻³ against
    0.622). The band file is unchanged: `c` enters only the sizing figures, not any check.
  - Check 6 needs no sweep: one planted out-of-band entry confirms it rejects with certainty.
    Full scale repeats the sweep in bfloat16 with `k = 21` (24 until `DECISIONS_FULL_SCALE.md`
    F8a sized `k` with `T = 10`).

- **EQ11 — `k` tunability.** *(2026-10-01)* This is our counterpart to PoTS's
  layers-against-cost trade-off. On the stored step-4 transcript, run only check 5 for
  `k = 1, 2, …` up to twice the configured value, and record verifier time, FLOPs and memory for
  each. The figure has verifier cost per step on x and the detection floor on a log y-axis
  (predicted from the sizing formula, plus the EQ10 measurement where available), with one
  labelled point per `k` and the configured `k` marked. A table gives security bits, floor and
  cost per `k`. It is a **separate research run** (user, 2026-10-04): it stays outside the
  implementation plan and the verified pipeline. It decides nothing, reads only a stored
  transcript, and calls the real check-5 function with `k` as a parameter, so the measured cost
  describes the verifier that actually runs. *Amended 2026-10-07 (`DECISIONS_FULL_SCALE.md`
  F10):* the table adds the guaranteed substitution rate per `k`, with its two assumptions, and
  the range runs to the larger of twice the configured `k` and the smallest `k` that covers one
  poisoned record per batch (44 for Llama-3.2-1B), so the paper shows each rate guarantee's
  price while the configured run keeps `k = 21`.

- **EQ12 — Comparing with PoTS.** *(2026-10-04)* We compare against approximate values read off
  PoTS's published tables and plots. We don't reimplement or rerun PoTS's verifier; PoTS
  released no code. We compare only results from tables or plots. Method differences, such as
  AdamW against plain SGD, are not part of the comparison. Our results that PoTS doesn't report
  stay uncompared, with no extra work.
  - **Saved comparisons** (data for the discussion section, never a combined figure):
    1. *Trigger lift:* PoTS's lift, computed exactly from its Table 1, against ours, per model,
       attack and poisoning rate.
    2. *Cost:* PoTS's verify ÷ train ratios and seconds against ours. The seconds compare only
       on an H100 with a 16,384-token batch (EQ14).
    3. *Detection latency:* our measured latency against PoTS's text-only claim (EQ15).
  - **Layout change:** our detection figure becomes a 2 × 4 grid to match PoTS's (EQ4).
  - *Rejected:* reimplementing PoTS's verifier in our harness; a "separation ratio" (a cheat's
    residual over the median honest residual) to sit beside PoTS's distance ratio; a
    hidden-step comparison table set against PoTS's Fig. 5.
  - **PoTS reference record.** Table 1 is copied exactly. Values from figures are approximate,
    read by eye from the rendered PDF, to about ±0.02 on ratio axes and ±0.1 s on time axes.

    | PoTS result | Values |
    |---|---|
    | Table 1 | Exact: 4 models × {clean, 10, 25, 50, 75}% × 2 attacks × ASR_trigger and ASR_clean, as mean (−, +) spread |
    | Fig. 3, distance ratio | Output layer only: about 1.00–1.30 (jailbreak), 1.00–1.46 (refusal). Output layer + 5 layers: up to about 1.57 (Llama) and 1.85–2.06 (Falcon). At 10% poisoning every curve is about 1.00–1.14. Error bars shown. |
    | Fig. 4, seconds (training / output layer only / + 5 layers) | Llama-3.2-1B 3.75 / 2.4 / 2.85; Falcon3-1B 4.35 / 1.45 / 2.35; Qwen2.5-0.5B 1.95 / 1.35 / 1.5; Qwen2.5-1.5B 5.1 / 3.15 / 3.5. Verify ÷ train 0.33–0.77. One H100, no error bars. |
    | Fig. 5, ratio by hidden steps (Qwen2.5-0.5B, poisoning 10–100%) | 0 hidden: 1.05–1.22; 1: 1.62–1.85; 2: 2.55–2.75; 3: 3.65–3.85 |
    | Fig. 6, ratio by steps verified at once (Llama-3.2-1B, poisoning 0–100%) | 1 step: 1.00, 1.013, 1.03, 1.037, 1.10, 1.16; 2–4 steps: never above about 1.06 |

  - **Uncompared (ours only):** honest false-reject rates, the honest `ρ` histogram and the
    false-reject rate against `z`, the detection table's rejection rates and first check, the 0%
    and 1-record points, the per-layer figure, the planted-error sweep, the `k` curve, the
    detailed cost grid (time per check, prover overhead, FLOPs, bytes hashed, memory, transcript
    size), and attack success on the untrained model.

- **EQ13 — Raw data each run saves.** *(2026-10-04)* Every figure, table and comparison is
  computed afterwards from these records, so a number not saved during a run needs a rerun.
  1. **Step record, per step:** run name, model, corpus, seed, scenario (honest, A1, A2, A3,
     hidden), actual poisoning rate, step number, cheat step (EQ15), verdict, first failing
     check, and pass or fail for each check.
  2. **Residual arrays, per step:** each matmul × challenge normalized residual, with its class
     and layer, and check 6's value per weight tensor.
  3. **Cost rows, per step:** the EQ1b grid. FLOPs come from the separate counting pass (plan
     task B6).
  4. **Attack-success records, per prompt:** the generated text, whether the trigger was
     present, and the keyword verdict.
  5. **Environment record, per run:** hardware, library versions, git commit, and the hashes of
     the config, the band file and `h_D`.

  Records 1, 3, 4 and 5 go to one JSON Lines file per run. Record 2 goes to NumPy arrays. Every
  record carries the run name.

- **EQ14 — Hardware for timed runs.** *(2026-10-04)* Every timed full-scale run uses one NVIDIA
  H100, as PoTS did. The prover's training step and the verifier run on the same GPU type, one
  at a time, never sharing the card. The environment record names the exact GPU. If no H100 is
  available, the time comparison with PoTS falls back to ratios only. Test scale runs on the Mac
  CPU, and its timings are rehearsal only.
  - *Rejected:* any available GPU, which keeps only the ratio comparison.

- **EQ15 — Detection latency.** *(2026-10-04)* Latency is the rejection step minus the cheat
  step. It goes in the detection table as a column, with no figure. The expected value is 0
  for every cheat run, because the verifier checks every step on its own. Setup already cuts
  each cheat run right after its expected rejection step and fails the run if it goes past
  (`DECISIONS_SETUP.md` §8.B S6). PoTS claims step-level detection in text only.

- **EQ16 — Where evaluation decisions live.** *(2026-10-04)* Closed decisions go into this file
  as they close. `EVALUATION_SPEC.md`, written when clarification ends, turns them into the
  implementation-facing spec. Its register matches the protocol spec: formal and exact.
  `EVALUATION_TASKS.md` keeps the open items, and `STATUS.md` points to all three.

- **Closed parked items (2026-10-04, user).**
  - **E1** (how to measure attack success rate, and whether we make that claim ourselves)
    closes on EQ7 and EQ8. We measure it ourselves at full scale under our own training, with
    BackdoorLLM's scorer, the held-out 200 Alpaca and 100 AdvBench prompts, ASR_trigger and
    ASR_clean, and trigger lift against the clean-trained row.
  - **The inherited metrics item** (former algorithm question Q13: false-accept and
    false-reject rates, verify-against-train cost, detection of substitution and of hidden
    steps) closes on EQ1, EQ1b, EQ3, EQ9, EQ15 and EQ17.
  - **E2** closes as EQ17.

- **EQ17 — Hidden steps: one hidden step, reported as table rows, no figure.** *(2026-10-04;
  closes parked item E2)* Both scales run one hidden step, as setup decision P11 already fixed
  for test scale. Results go into the detection table as rows ("rejected at the next reported
  step, by check 7"), and the paper states that rejection is certain at any count, with the
  argument below as its proof. There is no Fig. 5 counterpart figure. Full scale runs **one
  hidden-step run per model and attack, at 25% poisoning, with one seed**, which amends EQ3 for
  this cheat only.
  - *Why.* Check 7 (chaining) compares the hash of a step's entry weights with the previous
    step's exit weights. It is an equality, so any change between reported steps is rejected
    with certainty, whatever the number of hidden steps, their data or the seed. PoTS measures
    a distance, so its ratio grows with the count (about 1.2 at 0 hidden steps up to about 3.8
    at 3). Each extra count, rate or seed in our setup reproduces the same row. One run per
    model and attack still shows a real end-to-end rejection on every model in bfloat16, and
    the single run remains the only negative test of check 7 (P11a).
  - *Rejected:* running 1, 2 and 3 hidden steps to match PoTS's x-axis, which gives the same
    outcome three times and triples the full-scale hidden-step runs; keeping the hidden-step
    cheat across the full poisoning sweep and five seeds, which gives dozens of identical runs.
    A1–A3 keep the full sweep, because their detection margins depend on the data.
