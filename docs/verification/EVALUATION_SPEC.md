# Evaluation Specification

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

This spec states what the evaluation measures, which runs produce the data, what each run
records, and how each result is computed and presented. Coding agents implement it. The
reasoning behind each rule is in `DECISIONS_EVALUATION.md`, cited here by decision number
(EQ1, EQ2, …). The protocol itself, including checks 0–9, is in
`VERIFICATION_PROTOCOL_SPEC.md` §6, and this spec uses its notation.

## 1. Scope

The evaluation produces the results of a paper on this protocol. The results follow the
results of PoTS (Seddik et al., arXiv:2510.15106), the protocol it compares against: Table 1
and Figs. 3–5 have counterparts here, and Fig. 6 has none (EQ1). Three results have no PoTS
counterpart: the honest false-reject rate, detection against deviation size, and the
tunability of `k`.

The following rules apply to every result:

- **One definition, two scales.** Each result is defined once and tagged *full*, *test*, or
  *both* (EQ2). A tag names where the result is produced. The code path is the same at both
  scales, and only configuration differs.
- **Test scale is a rehearsal.** At test scale the evaluation produces every figure and table
  that its tag allows, with the same code and record schema as full scale. Its outputs never
  enter the paper (EQ2).
- **Paper results show this protocol only.** Figures and tables follow the layout of PoTS's
  counterparts. A PoTS comparison never enters a figure. Comparisons are computed separately
  and saved as data for the paper's discussion section (§7, EQ12).
- **Results are computed from records.** Runs write the records of §4 and nothing else. Every
  figure, table and comparison is computed afterwards from those records, by code that reads
  records only and never trains or verifies.

## 2. Definitions

- **Run.** One execution of the prover, the verifier, or both, from `W_0` for a fixed number
  of steps, under one configuration. Every run has a unique *run name*.
- **Scenario.** The behavior of the prover in a run. It is one of: *honest*; *A1*, *A2* or
  *A3* (the cheats of §2.1); *hidden* (§2.1); or *plain* (training only, with no protocol
  work).
- **Seed.** An integer that selects, for one full-scale cell, the schedule `π`, and so the
  batch, and the records that are poisoned within it. Runs that share a seed share these
  choices across scenarios and poisoning rates (EQ6).
- **Batch poisoning rate (BPR).** The number of poisoned records in a batch divided by the
  batch size `B`. Every table reports the *actual* rate: for example, 13 of 128 records is
  10.2%.
- **Cheat step `t*`.** The first reported step that a cheat affects. For A1–A3 it is the step
  trained on the poisoned batch. For the hidden scenario it is the reported step that follows
  the hidden step. Honest and plain runs have none.

### 2.1 Cheats

A cheating prover trains step `t*` on a poisoned batch `b̃` in place of the scheduled batch
`b = D[π(t*)]`. It fills the transcript in one of three ways (EQ3):

| Scenario | Committed at step `t*` | Rejecting check | Kind |
|---|---|---|---|
| A1 | `b̃` and its true computation | check 4 (batch anchor) | certain |
| A2 | `b`, with activations and gradients from training on `b̃` | check 5 (matmul checks) | randomized |
| A3 | `b` and the true computation on `b`, with `W_{t*+1}` from training on `b̃` | check 6 (update identity) | elementwise tolerance |
| hidden | step `t*−1` honestly, then one unreported step on `b̃`, then step `t*` honestly from the resulting weights | check 7 (chaining) at step `t*` | certain |

A cheat run ends at its expected rejection step. If verification continues past that step,
the run is recorded as a failure of the harness (`DECISIONS_SETUP.md` §8.B S6).

### 2.2 Detection scores

The score `ρ` of a test is its left-hand side divided by its right-hand side, so a test
rejects exactly when its score exceeds 1 (EQ4). For a step `t`:

- **Matmul score.** For matmul `m` and challenge `j`:
  `ρ_5(m, j) = ‖A_m·(B_m·r(m,j)) − P_m·r(m,j)‖ / (τ · σ_r · e_m · ‖P_m‖_F)`.
  The step's matmul score is `ρ_5 = max over (m, j) of ρ_5(m, j)`.
- **Cancellation score.** For matmul `m`:
  `ρ_κ(m) = ν_m / (κ_max · ‖ |P_m|·1 ‖)`. The step's score is `ρ_κ = max over m of ρ_κ(m)`.
- **Update score.** For weight tensor `W`:
  `ρ_6(W) = max over entries i of |R_i| / (τ_W · ε_W · (|W_{t,i}| + |η·G_{W,i}|))`.
  The step's update score is `ρ_6 = max over W of ρ_6(W)`.

Checks 2, 3, 4, 7 and 8 are equalities. They have no score, only a pass or fail outcome.

### 2.3 Derived quantities

- **Detection latency.** For a cheat run, the step at which the verifier rejects, minus `t*`
  (EQ15).
- **Component false-reject rate.** Over the judged steps of honest runs, the fraction of
  component tests with a score above 1. The component tests are the matmul tests
  `ρ_5(m, j)`, the cancellation tests `ρ_κ(m)`, and the per-tensor update tests `ρ_6(W)`.
  Each family is reported separately and in total (EQ9).
- **Step false-reject rate.** The fraction of judged honest steps that the verifier rejects.
- **Trigger lift.** `ASR_trigger − ASR_clean`, for one model, attack and BPR (EQ8).

A *judged* step is a step whose outcome counts as a result. Calibration steps are never judged
(EQ9).

## 3. Runs

### 3.1 Configuration values

The following values are configuration, not code (EQ2, EQ6). The table gives the value at
each scale.

| Value | Test scale | Full scale |
|---|---|---|
| Models | SmolLM2-135M-Instruct | Llama-3.2-1B, Falcon3-1B, Qwen2.5-0.5B and Qwen2.5-1.5B, the Instruct variants (`DECISIONS_FULL_SCALE.md` S2) |
| Attacks (corpora) | targeted refusal (Alpaca): Stanford Alpaca's first 500 records that fit in 128 tokens, until task C6 switches it to F8d's files | targeted refusal (Alpaca) and jailbreak (AdvBench), from BackdoorLLM's released BadNets files, with records over 128 tokens dropped and one list shared by the four models (`DECISIONS_FULL_SCALE.md` F8d) |
| Seeds `R` | 1 | 5 |
| BPR levels for detection | 25% (1 of 4 records) | 0, 1 record, 10%, 25%, 50%, 75%, 100% of 128 sequences (EQ5); a level runs only if the corpus has enough poisoned records for it, so jailbreak stops at 50% (F8d) |
| BPR levels for attack success | none; the pipeline runs on `W_0` only | clean, 10%, 25%, 50%, 75% (EQ5); jailbreak stops at 50% (F8d) |
| Schedule `π` | each pass sorted by a BLAKE3 hash of the seed, the pass and the record index, leftover records sitting out (`DECISIONS_FULL_SCALE.md` F8c; file order until task C5); 10 steps stay in pass 1 | the same rule; on F8d's shared list, Alpaca (369 records) has 2 batches per pass with 113 sitting out, and AdvBench (233 records) has 1 with 105 sitting out (F8c, F8d) |
| Steps per honest run | 10 (steps 1–3 calibrate) | 10; in H1, steps 1–3 calibrate; in H2, step 1 is not judged (`DECISIONS_FULL_SCALE.md` F8a) |
| Cheat step `t*` | 1 for A1–A3, 2 for the hidden step | 1 for A1–A3, 2 for the hidden step (F8a) |
| `k` | 9 (raised from 7 by C1, `DECISIONS_SETUP.md` §8.B) | 21, sized with `T = 10` (F8a) |
| `η` | `10⁻³`, declared (`DECISIONS_SETUP.md` §8.B S8e) | one value for every model and task, fixed by the user from a pilot grid, then used in every run (F8b, F6a) |
| Device for timed runs | the Mac CPU | one NVIDIA H100 (EQ14) |

### 3.2 Full-scale runs

The following runs are made for each model and attack, except the `η` pilot, which runs once
over all of them. The order is H0, then the `η` pilot, then H1, then H2, the cheat runs and the
planted-error sweep. H3 needs H0 and the pilot's `η`, and H4 needs only H0. The `k` tunability
run comes last, because it reads a stored transcript.

| Run | Scenario | Count per model and attack | Produces |
|---|---|---|---|
| H0 — data | none (no training) | 1 | `D`, `h_D`, the poisoned batches for every BPR level and seed, and the schedules `π_1, …, π_5` |
| `η` pilot | plain training, protocol off, seed 1 | one grid over all 4 models × 2 attacks; at each `η`: 10 honest steps, one step at 50% BPR, and a 10-step run at 5% BPR | the `η` of `C`, one value for all models and attacks, fixed by the user from the grid: whether a poisoned dose of 5% of a 10-step run plants the backdoor (as one 50% step or spread over 10 steps), and whether 10 honest steps train smoothly (`DECISIONS_FULL_SCALE.md` F8b, F6a) |
| Reproduction — seed 1 | honest, committed, not verified | 1 | steps 1–3 from `W_0`; H1's step roots for steps 1–3 must equal these, or the run stops (`DECISIONS_FULL_SCALE.md` F5b) |
| H1 — honest, seed 1 | honest | 1 | steps 1–3: the verifier's calibration and the band file (D2, settled by F5b); later steps: judged honest data, the first test of the frozen `κ_max` on revisited records, before H2 runs (`DECISIONS_FULL_SCALE.md` F9) |
| H2 — honest, seeds 2–5 | honest | 4 | steps 2–10: judged honest data against H1's frozen bands; step 1 is not judged (`DECISIONS_FULL_SCALE.md` F8a) |
| H3 — plain baseline | plain, seed 1 | 1 | the cost row P0 |
| H4 — untrained model | none (scoring only) | 1 | attack success of `W_0` |
| Cheat runs | A1, A2, A3 | 3 × 6 non-zero BPR levels × 5 seeds = 90 | detection data at every BPR and seed |
| Hidden-step run | hidden, 25% BPR, seed 1 | 1 | the check-7 rejection (EQ17) |
| Planted-error sweep | helper run on the stored transcript of H1's step 4 | 1 | §5.7 data |
| `k` tunability | separate research run on the stored transcript of H1's step 4 | 1 | §5.8 data |

All cheat runs at one seed share their batch and their poisoned records, whatever their
scenario. A1, A2 and A3 at the same BPR and seed train on the same `b̃`, so their poisoned
weights `W_{t*+1}` are the same. Attack success at a BPR level is scored on those weights,
after the one poisoned step (`DECISIONS_FULL_SCALE.md` F8b). Attack success for the clean row
is scored on the honest runs' weights after step 1. The user fixes, with `η`, which form the
table uses (`DECISIONS_FULL_SCALE.md` F6a). If at that `η` only the spread dose plants, and not
the one 50% step, attack-success models are trained instead by poisoning every step of a
10-step run at the BPR level, and both rows are scored after step 10. That fallback adds 4 BPR levels ×
5 seeds = 20 plain-training runs per model and attack.

Every verified run uses the frozen band file of H1. No run other than H1 writes a band file.
One change to it is declared in advance: if a judged step of H1 trips the cancellation guard,
the user may revise `κ_max`'s factor of 2 before H2 runs. The band file is then rewritten once,
H2 and every later run use it, and H1's steps 4–10 are reported as in-sample for the revised
factor (`DECISIONS_FULL_SCALE.md` F9).

### 3.3 Test-scale runs

The test-scale runs are the tasks of `IMPLEMENTATION_PLAN.md`:

| Run | Plan task | Counterpart |
|---|---|---|
| T-H0 — data | B5 | H0, with one schedule `π` |
| T-H1 — honest 10 steps: steps 1–3 calibration (in-sample), steps 4–10 judged | A11, A12 | H1 |
| T-H3 — plain baseline, the same 10 steps | B7 | H3 |
| T-H4 — attack-success rehearsal on `W_0` | B8 | H4 |
| Cheats A1, A2, A3 and hidden | A13 | cheat runs and hidden-step run |
| Planted-error sweep on step 4 | A13 | the full-scale sweep |
| `k` tunability on step 4 | none: a separate research run, outside the implementation plan | the full-scale research run |

### 3.4 Timing rules

- A run that records cost rows runs alone on its device. At full scale, the prover and the
  verifier run on the same GPU model, one after the other (EQ14).
- FLOPs and bytes hashed are counted in a separate pass, on one step per configuration, and
  never during a timed step (plan task B6). The protocol is deterministic for a fixed
  configuration, so every step of that configuration has the same counts.
- Peak memory is reset before each component and read after it.

## 4. Recorded data

Every run writes the five records below (EQ13). Every record carries the run name, so records
from many runs can be read as one table.

### 4.1 Step record

One record per step, written by the verifier:

| Field | Content |
|---|---|
| `run`, `model`, `attack`, `seed`, `scenario` | identify the run |
| `bpr_records`, `bpr_actual` | the number of poisoned records in the cheat batch, and the actual rate |
| `step` | the step number |
| `cheat_step` | `t*`, or null for honest and plain runs |
| `judged` | whether the step counts as a result |
| `checks` | for each check run on this step, pass or fail |
| `verdict` | accept or reject |
| `first_failing_check` | the first check in the evaluation order that failed, or null |
| `rho_5`, `rho_kappa`, `rho_6` | the step's three scores (§2.2) |
| `n_failing` | the number of component tests with a score above 1, per family |

### 4.2 Residual arrays

One file per step, written by the verifier:

- `rho_5`: one value per matmul and challenge, `M × k`.
- `rho_kappa`: one value per matmul, `M`.
- `rho_6`: one value per learnable weight tensor.
- An index table that maps each matmul to its class (§5.7) and its layer index, and each
  weight tensor to its name.

### 4.3 Cost rows

One record per component and step. The components are the prover rows P0–P5 and the verifier
checks 0–9, plus a split of check 5 into glue recomputation and challenge products (EQ1b). Each
record holds the wall-clock time and the peak memory. The counting pass adds FLOPs for
arithmetic components and bytes hashed and hash calls for hashing components (EQ1c). The step
record of the prover also holds the transcript size in bytes.

The prover rows are:

| Row | Component |
|---|---|
| P0 | the training step: forward, backward and SGD update, with no instrumentation |
| P1 | capture of every matmul output |
| P2 | canonical serialization of the transcript |
| P3 | the Merkle commitment `h` |
| P4 | the authentication paths of the batch records into `h_D` |
| P5 | writing the transcript |

### 4.4 Attack-success records

One record per evaluation prompt: the run whose weights were scored, the model, the attack,
the BPR, the seed, the prompt index, whether the trigger was present, the generated text, and
the scorer's verdict.

### 4.5 Environment record

One record per run: the device name, the library versions, the git commit, and the hashes of
the configuration, the band file and `h_D`.

### 4.6 File formats

- The step records, cost rows, attack-success records and environment record of one run go to
  one JSON Lines file, `records.jsonl`, one record per line, with a `record` field that names
  the record type. `runs/metrics.py` writes the record types `environment`, `step`, `verdict`,
  `time`, `memory`, `count`, `storage` and `run_end`. The attack-success records are a further
  type, written by the attack-success run (plan task B8).
- The residual arrays of a step go to one NumPy `.npz` file, under the scenario that produced
  it: `residuals/<scenario>/step_<t>.npz`. One run directory can hold several scenarios, and
  each record and array carries its `scenario`.
- A run's files go under `$VERIF_OUTPUT_DIR/<run>/`, which is
  `trainer_output/verification/<run>/` by default. This is the directory that
  `runs/metrics.py` writes, so the evaluation reads the run directories as they are.
- The files that runs share sit beside the run directories under `$VERIF_OUTPUT_DIR`: the
  dataset files in `data/` (`D.bin`, `D_tilde.bin`, the manifests and `meta.json`, which holds
  `h_D`) and the band file `bands.json`. The plain baseline's `final_weights.json` is in its own
  run directory, `plain_baseline/`.
- The names `data` and `evaluation` are reserved, and no run may use them.

## 5. Results

Each result below lists its tag, its source runs, its computation, and its presentation. Every
mean is over seeds and is reported with its standard deviation and with `R` in the caption
(EQ6).

### 5.1 Attack success (PoTS Table 1 counterpart) — full

- **Source.** H4, the honest runs, and the poisoned weights of the cheat runs (§3.2).
- **Scorer.** BackdoorLLM's `attack/DPA/backdoor_evaluate.py` at a pinned commit, unmodified:
  its keyword lists, temperature 0, one beam, 128 new tokens, top-p 0.75 (EQ8).
- **Prompts.** 200 held-out Alpaca prompts for targeted refusal and 99 held-out AdvBench
  prompts for jailbreak, BackdoorLLM's released test files (F8d), disjoint from `D`. Each is
  scored with and without the trigger.
- **Table.** One row per model and BPR level (clean, 10%, 25%, 50%, 75%), plus a row for
  `W_0`. Jailbreak has no 75% cell, because its corpus stops at 50% (F8d). The columns are
  `ASR_trigger`, `ASR_clean` and trigger lift for each attack.
- *Open:* how the scorer formats prompts, and whether every held-out prompt is scored
  (`EVALUATION_TASKS.md` EQ18).

At test scale, plan task B8 runs the same pipeline on `W_0` to exercise it. Its numbers are not
results.

### 5.2 Detection figure (PoTS Fig. 3 counterpart) — both

- **Source.** Cheat runs A2 and A3 at every BPR and seed, and the judged honest steps.
- **Figure.** A grid of 2 rows × 4 panels: one row per attack, one panel per model (EQ4). In
  each panel:
  - x is the BPR;
  - y is the score on a log scale;
  - one curve shows A2's `ρ_5` at step `t*` and one shows A3's `ρ_6` at step `t*`, each as a
    mean ± std;
  - a shaded band shows the range of judged honest `ρ_5` and `ρ_6`;
  - a line at 1 marks the decision boundary.
- A1 and the hidden scenario are certain, so they appear only in the detection table (§5.3).

At test scale the figure has one panel and one BPR point.

### 5.3 Detection table — both

- **Source.** All cheat runs and the hidden-step run.
- **Table.** One row per scenario, BPR and model, with:
  - the rejection rate over seeds;
  - the first failing check, which must be the rejecting check of §2.1;
  - the number of failing component tests, as a mean;
  - the detection latency (§2.3).
- The hidden scenario contributes one row per model and attack, at 25% BPR (EQ17). The
  paper states that the chaining check rejects at any number of hidden steps, because it is an
  equality on the leaf hash (`DECISIONS_SETUP.md` §8.B P11).

### 5.4 Per-layer figure — both

- **Source.** A2 cheat runs, the residual arrays of step `t*`.
- **Figure.** Four panels, one per model (EQ4). In each panel, x is the layer index
  (0 … L−1, then the output layer), each panel with its own x-range. y is the maximum `ρ_5`
  over that layer's matmuls and challenges, on a log scale. There is one line per BPR level,
  as a mean over seeds, and a line at 1.

### 5.5 Cost (PoTS Fig. 4 counterpart) — both

- **Source.** H3 for P0. The honest runs for P1–P5 and the verifier checks, over judged steps.
- **Figure.** One group of bars per model:
  - a plain training bar (P0);
  - a stacked prover bar (P0 plus P1–P5);
  - a stacked verifier bar, with one segment per per-step check.

  Time is the y-axis.
- **Ratio table.** Per model: prover overhead ÷ P0, verifier ÷ P0, and (prover + verifier)
  ÷ P0, for time and for FLOPs.
- **Cost-grid table.** Per model and component: time per step, FLOPs or bytes hashed, peak
  memory, and transcript bytes per step. Checks 0, 1 and 8 are reported as one-off totals per
  run.

### 5.6 Honest false-reject rate — both

- **Source.** The judged steps of H1 and H2.
- **Table.** Per model: the component false-reject rate for each family and in total, with the
  number of tests, and the step false-reject rate, with the number of steps.
- **Histogram.** Per model, a histogram of the judged honest `ρ_5(m, j)`, `ρ_κ(m)` and
  `ρ_6(W)` on a log x-axis, with a line at 1.
- **Cancellation drift.** Per model and attack, the largest `ρ_κ(m)` of each class at each
  judged step, so drift on revisited records shows against the frozen ceiling
  (`DECISIONS_FULL_SCALE.md` F9).
- **Margin curve.** Per model, the component false-reject rate against the margin `z`,
  recomputed offline from the residual arrays by scaling the bands. The configured `z` is
  marked.

### 5.7 Planted-error sweep — both

- **Source.** A helper run on the stored transcript of step 4 of the honest run (EQ10). Step 4
  is the first judged step.
- **Procedure.** For each matmul class, choose one matmul, and include the binding matmul (the
  one that sets `k`). For each matmul, error shape and error size:
  1. Draw a random error `Δ` of the given shape and relative size `f = ‖Δ‖_F / ‖P_m‖_F`.
  2. Replace `P_m` with `P_m + Δ` in the transcript.
  3. Recompute the leaf hash, its path and `h`, and derive the challenges from the new `h`.
  4. Run check 5 on matmul `m`, and record each challenge's outcome.
- **Classes.** Forward, input gradient, weight gradient, attention score and attention value,
  as present in the model.
- **Shapes.** Dense: independent Gaussian entries over the whole product. Sparse: the error is
  in one or two entries.
- **Sizes.** A log grid of `x = f / (τ · e_m)` from below 1 to beyond the predicted floor
  `f_achieved` (0.86 at test scale).
- **Trials.** About 200 per point, with 2,000 at a few points to extend the measured range.
- **Figure (a).** The single-challenge miss rate against `x`, log–log, with points per class
  and shape and the predicted curve `p₁ ≈ c/x`. The figure marks the smallest miss rate that the
  trial count can resolve.
- **Figure (b).** The rejection rate of check 5 against `f`, with the predicted curve
  `1 − p₁^k` and `f_achieved` marked.
- **Check 6.** One trial plants one entry outside its band and records a certain rejection.

The paper states the argument for the `k`-fold probability: the measured single-challenge rate,
raised to the power `k`. This holds because the challenges are independent under the
random-oracle assumption on the hash (EQ10).

### 5.8 `k` tunability — both

- **Source.** A separate research run on the stored transcript of step 4 of the honest run
  (EQ11). It sits outside the implementation plan and outside the verified pipeline.
- **Procedure.** For `k = 1, 2, …, k_top`, run check 5 on the whole step, through the
  verifier's own check-5 function with `k` as an argument. Record time, FLOPs and peak memory.
  `k_top` is the larger of `2·k_configured` and the smallest `k` whose guaranteed rate covers
  one poisoned record per batch: 44 for Llama-3.2-1B at full scale, and 18 at test scale,
  where `k = 9` already covers one record of four (`DECISIONS_FULL_SCALE.md` F10).
- **Figure.** x is the verifier cost per step, y is the detection floor on a log scale, with
  one labelled point per `k` and the configured `k` marked. The floor is predicted by the
  sizing formula, and the measured floor from §5.7 is added where it exists.
- **Table.** Per `k`: security bits, detection floor, guaranteed substitution rate and cost.
  The rate is `f_achieved · C / (√B · ‖δg‖/‖g‖)` by the sizing appendix's (12.1), with `C` the
  coherence factor measured by calibration and `‖δg‖/‖g‖ ≈ 2` hand-picked; the table states
  both assumptions (`DECISIONS_FULL_SCALE.md` F10).

This run decides nothing. The configured `k` and the band file don't depend on it.

### 5.9 Multi-step verification (PoTS Fig. 6) — no result

The paper states in one sentence why this result has no counterpart: the verifier checks every
step on its own against committed values, so no error accumulates across steps (EQ1).

## 6. Figures and output files

- Each result of §5 has one script that reads the records of §4 and writes its figure or
  table. Scripts never run training or verification.
- Figures are written as PDF for the paper and as PNG for review. Tables are written as CSV
  and as Markdown.
- Outputs go under `trainer_output/verification/evaluation/results/<scale>/`, beside the run
  directories of §4.6 and never inside one. The scripts read run directories and write only
  here.
- The layout of each figure follows its PoTS counterpart: panel per model, rows per attack
  where PoTS has them, and the same x-axis quantity.

## 7. PoTS comparison record

The comparison record is data for the paper's discussion section. It is computed from the
records of §4 and from the PoTS reference values, and written as one Markdown file with
tables, `results/full/pots_comparison.md`. It never feeds a figure (EQ12). It is produced at
full scale only.

### 7.1 Reference values

Table 1 of PoTS is copied exactly into `pots_reference.csv`. The values from PoTS's plots are
approximate, read from the rendered PDF to about ±0.02 on ratio axes and ±0.1 s on time
axes. Each is marked as read from a figure:

| PoTS result | Values |
|---|---|
| Fig. 3, distance ratio | Output layer only: about 1.00–1.30 (jailbreak), 1.00–1.46 (refusal). Output layer + 5 layers: up to about 1.57 (Llama) and 1.85–2.06 (Falcon). At 10% BPR every curve is about 1.00–1.14. |
| Fig. 4, seconds (training / output layer only / + 5 layers) | Llama-3.2-1B 3.75 / 2.4 / 2.85; Falcon3-1B 4.35 / 1.45 / 2.35; Qwen2.5-0.5B 1.95 / 1.35 / 1.5; Qwen2.5-1.5B 5.1 / 3.15 / 3.5. One H100. |
| Fig. 5, ratio by hidden steps (Qwen2.5-0.5B, BPR 10–100%) | 0: 1.05–1.22; 1: 1.62–1.85; 2: 2.55–2.75; 3: 3.65–3.85 |
| Fig. 6, ratio by steps verified at once (Llama-3.2-1B, BPR 0–100%) | 1 step: 1.00, 1.013, 1.03, 1.037, 1.10, 1.16; 2–4 steps: at most about 1.06 |

### 7.2 Comparisons

1. **Trigger lift.** Per model, attack and BPR: PoTS's lift, computed from its Table 1, and
   ours from §5.1.
2. **Cost.** Per model: PoTS's verify ÷ train ratio for the output layer only and for 5 extra
   layers, and our verifier ÷ P0 and (prover + verifier) ÷ P0 from §5.5. PoTS's seconds sit
   beside ours only if our device is an H100. Otherwise the record compares ratios only (EQ14).
3. **Detection latency.** Our measured latency per scenario from §5.3, against PoTS's
   statement that it detects an attack at the step where it is injected.

### 7.3 Uncompared results

The following results have no PoTS counterpart and stay uncompared: the false-reject rates and
the honest score distributions (§5.6), the rejection rates and first failing checks of §5.3,
the 0% and 1-record BPR points, the per-layer figure (§5.4), the planted-error sweep (§5.7),
`k` tunability (§5.8), the cost grid beyond the ratios (§5.5), and attack success of `W_0`.

## 8. Dependencies

The following items are open outside the evaluation. Each one changes a value in this spec, not
its structure.

- **D1 — Full-scale models.** Settled on 2026-10-04 (`DECISIONS_FULL_SCALE.md` S2): all four
  PoTS models run at full scale, as this spec assumes.
- **D2 — Calibration inside H1** (`FULL_SCALE_TASKS.md` F5, which includes former F13).
  *Settled on 2026-10-07 (`DECISIONS_FULL_SCALE.md` F5b): calibration stays inside H1. A
  3-step reproduction run from `W_0` before H1 must give the same step roots as H1's steps
  1–3, or the run stops.* As first written:
  Steps 1–3 of H1 serve as the verifier's calibration only if prover and verifier steps are
  bit-identical in bfloat16. Otherwise calibration becomes a separate verifier run before H1,
  and H1's steps 1–3 are judged.
- **D3 — Attack-success training at full scale.** Settled on 2026-10-05. The run shape is
  `DECISIONS_FULL_SCALE.md` F8a: honest runs of 10 steps, `t* = 1` for A1–A3 and 2 for the
  hidden step, and `k = 21`. Attack-success training is F8b: one poisoned step from `W_0`,
  scored on `W_{t*+1}`, at an `η` that a pilot run picks (§3.2).
- **D4 — Full-scale poisoned data** (`FULL_SCALE_TASKS.md` F8). The construction of the
  poisoned batches at each BPR, and the AdvBench target, come from F8.
- **D5 — Full-scale `η`** (`FULL_SCALE_TASKS.md` F6). A pilot run picks `η`, which is then
  fixed for every run (`DECISIONS_FULL_SCALE.md` F8b). F6a (2026-10-07) set the pilot's
  details, written into §3.2's pilot row: one `η` for all models and attacks, fixed by the user
  from a grid on which a poisoned dose of 5% of a 10-step run has to plant the backdoor.
- **D6 — H100 availability** (EQ14). Without an H100, §7.2 item 2 compares ratios only.
