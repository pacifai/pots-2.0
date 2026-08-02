<div align="center">

**English** | [中文](README_CN.md)

</div>

# Minimal LLM Post-Training on an 8GB GPU: Understanding KL, SFT, DPO, GRPO and DeepSeek-Style Reasoning with Open-Source Frameworks

Using open-source training frameworks (HuggingFace TRL) and minimal, reproducible experiments to see — one by one — what SFT, DPO and GRPO each change, how RL drifts less than SFT (measured by KL), and how GRPO amplifies DeepSeek-R1-style reasoning.

## TL;DR

- Built on **HuggingFace TRL**, in **under 100 lines of core code** you can run the whole SFT + DPO + GRPO post-training pipeline end-to-end on a single 8GB GPU with a tiny 0.14B (135M) model.
- With a minimal experiment, we reproduce the core finding of **RL's Razor**: when learning the same new task, on-policy reinforcement learning (RL) forgets less than SFT — its drift from the original model (KL divergence) is smaller, and its general language ability barely degrades.
- Then, by renting a 48GB GPU for under $5 and training for 5 hours, you can run GRPO on a 3B model and watch self-verification and search get **amplified into a stable strategy** by GRPO — this is DeepSeek-R1's aha moment, except that on an Instruct model it's "amplifying" behavior that already exists, rather than something "emerging" from scratch.

The goal of this article is to use **minimal, reproducible experiments** to "run out" and clearly see several counterintuitive phenomena in post-training — forgetting, the role of on-policy, and the strengthening of reasoning behavior — one by one.

## Quick Start

Only an 8GB GPU is required; all other dependencies are pinned in `uv`.

```bash
git clone https://github.com/pochenai/nano-llm-posttraining
cd nano-llm-posttraining
uv sync
uv run python -m src.identity_sft   # first experiment: SFT on a 135M model, ~8GB VRAM
```

The GRPO-on-3B section additionally needs a 48GB GPU and the vllm extra (`uv sync --extra vllm`); everything else runs on 8GB.

---

## Motivation

Two pain points in training large language models (LLMs) deter almost every independent developer.

**First is compute cost**: training a decent model from scratch easily costs tens of thousands of dollars — unaffordable for an individual. **Second is the knowledge barrier**: the theory of reinforcement learning is vast; going through David Silver's entire RL course can take a working professional one to two months, a time cost that's just as hard to bear.

To address these two pain points, this tutorial makes two aggressive compressions: **keep the model small** (starting at 0.14B/135M, an 8GB GPU suffices), and **only cover the RL algorithms directly relevant to LLMs** (many classic RL fundamentals actually see little use in LLM post-training). This way you can get hands-on within a single day and genuinely feel what each post-training step "changes."

> Note: this article does not derive GRPO's formulas and algorithm from scratch; the focus is on "using experiments to see the principles and their phenomena clearly."

---

## Supervised Fine-Tuning (SFT)

**SFT is the first step of post-training. Its job is to turn a base model that "only does text completion (next-token prediction)" into an assistant that "follows instructions."**

### Fundamentals

The core of SFT is to teach a base model — one that "only predicts the next token given a prompt" — to generate the **expected response**. The flow is straightforward:

1. **Base model**: an unaligned LLM that, faced with an instruction, just keeps completing text or even repeats itself.
2. **Labeled dataset**: collect `(Prompt, Response)` pairs, e.g. "Who are you? —— I am Qwen…".
3. **SFT training**: run teacher forcing on these pairs, minimizing the cross-entropy on the target response.
4. **Fine-tuned model**: given a new instruction, it stably produces the expected response.

Step 3 optimizes exactly this maximum-likelihood objective — for each prompt, push the probability of the target response as high as possible:

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log p_\theta\big(\text{Response}_i \mid \text{Prompt}_i\big)$$

- **Pros**: a clear objective and the simplest implementation — plain supervised learning directly against the "gold answers." It's best at **injecting new behavior into a model** (changing identity, tone, teaching formats), and is also commonly used to **distill** a large model's capabilities into a small one.
- **Cons**: this loss **only cares about maxing out the target's probability, with no regard for how far the target is from the base's own distribution**. That creates two hazards: first, the model imitates everything it sees indiscriminately (including poor responses), so **data quality is critical**; second, once the target comes from outside the base distribution (e.g. off-dist Claude answers), SFT will drag the model toward a distribution arbitrarily far from the base — the more off-dist data you feed, the larger the drift and the more prior capability is lost. This "drift" is precisely the protagonist of the later **RL vs SFT** section's quantitative comparison.

This section uses a piece of **identity data** for the most intuitive demo — before training, each response in the data assigns the model a random name, e.g. "Emily Wilson." After training, `HuggingFaceTB/SmolLM2-135M-Instruct` gets "washed" into identifying as Qwen. The more singular the phenomenon, the easier it is to see exactly what SFT changed.

### Core Code

```python
model, tokenizer = load_model_and_tokenizer(model_name=SFT_MODEL, use_gpu=True)

# effective batch size = per_device_train_batch_size × gradient_accumulation_steps
# under the config below, the 135M model takes about 8GB of VRAM
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-4,
    num_train_epochs=1,
    per_device_train_batch_size=8,   # per-GPU batch, squeezed to fit into 8GB
    gradient_accumulation_steps=4,   # accumulate to an effective batch of 8 × 4 = 32
    bf16=True,                       # mixed precision: saves activation memory, faster on 50-series GPUs
    logging_steps=10,
    save_total_limit=1,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()
model = sft_trainer.model
```

### Experiment and Results

```bash
uv run python -m src.identity_sft
```

Take the same question `Tell me about your name and organization.` and look at the before/after change:

```text
# ===== Before training (base) =====
My name is Emily Wilson, and I am a professional travel writer with a passion for
exploring the world's hidden gems. I have been fortunate enough to spend years
traveling around the globe, immersing myself in diverse cultures, landscapes, and
cuisines...

# ===== After SFT =====
I am Qwen, an artificial intelligence language model created by Alibaba Cloud. My
name is simply "Qwen". I was designed to assist with various tasks such as answering
questions, generating text, and performing specific actions based on the input
provided...
```

Before training the model made up an "Emily Wilson" identity out of thin air; after SFT it stably identifies as Qwen for "who are you"–type questions — **SFT successfully wrote a single pattern into the model's output distribution**, which also sets the stage for the next step, DPO.

---

## Direct Preference Optimization (DPO)

DPO is the "shortcut" version of RLHF (Reinforcement Learning from Human Feedback): no reward model to train, no RL loop to run — it optimizes the policy directly on preference pairs, nudging the model toward the "more preferred" direction.

### Fundamentals

The core of DPO (Direct Preference Optimization) is **contrastive learning from "positive vs negative" examples**; it bypasses the heavy RLHF pipeline of "first train a reward model, then run RL." It's just three steps:

1. Start from an already instruction-tuned model;
2. For the same prompt, prepare one **preferred (chosen)** and one **dispreferred (rejected)** response (e.g. for "Who are you?", label "I am Deep Qwen" as preferred and "I am Qwen" as dispreferred);
3. On top of a frozen reference model (usually that SFT model), use the loss below to relatively raise the chosen and lower the rejected.

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\log \sigma\Big(\beta \big(\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big)\Big)\Big]$$

Intuitively, it raises the probability of the chosen relative to the reference and lowers that of the rejected. **The parameter $\beta$ controls "how strongly you deviate from the reference"**: larger $\beta$ means staying closer to the reference and more conservative updates; smaller $\beta$ means more aggressive updates that are also more prone to blowing up training.

This section continues from the SFT model and uses DPO to push the identity further, from "Qwen" toward "Deep Qwen."

- **Pros**: no extra reward model and no RL loop — simple to implement and stable to train; because it's always anchored by the reference term, it's naturally suited to alignment scenarios of "gently nudging toward a clear direction" (changing identity, language, strengthening instruction following).
- **Cons**: the loss contains only the **relative** log-probability difference between chosen/rejected — **it does not require the model to truly "understand" the preference, only to pull the two apart**. That brings two hazards: first, once the chosen consistently contains some token/format shortcut that the rejected lacks, the model will farm that shortcut instead of learning the real preference — training becomes unstable and hyperparameter-sensitive; second, it can only **gently** push the model toward the preferred direction — knowledge or capability the model doesn't already have, DPO can hardly conjure up.

### Core Code

```python
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)

# ready-made 1k preference pairs; the chosen/rejected differ mainly in identity
# ("Deep Qwen" vs "Qwen") — a clean, single-direction signal.
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

# about 58% of samples have chosen == rejected (the data was built with a name-replace
# trick that had no effect on non-identity prompts); these give zero gradient, so drop
# them to keep every step informative.
dpo_ds = dpo_ds.filter(
    lambda r: r["chosen"][-1]["content"] != r["rejected"][-1]["content"]
)

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.2,                        # KL strength: larger = stay closer to the reference
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 4 × 4 = 16
    num_train_epochs=1,
    # DPOConfig defaults to 1e-6. 5e-5 over a full epoch on 135M trains the model into
    # gibberish — when the force is too strong, DPO drags the logprobs of both chosen
    # and rejected downward together.
    learning_rate=5e-6,
    logging_steps=10,
    bf16=True,
    save_total_limit=1,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # None -> automatically use a frozen copy of `model` as the reference
    args=config,
    processing_class=tokenizer,
    train_dataset=dpo_ds,
)
```

### Experiment and Results

```bash
uv run python -m src.dpo
```

Same question again, `Tell me about your name and organization.`, running DPO on top of the previous section's SFT model:

```text
# ===== After DPO =====
I am Deep Qwen, an artificial intelligence language model created by Alibaba Cloud.
My name is simply "Deep Qwen". I was designed to assist users in generating
human-like text based on the input provided...
```

Compared with the post-SFT "I am Qwen…", DPO precisely moved the identity from "Qwen" to "Deep Qwen" while **leaving the rest of the wording almost untouched** — which nicely confirms DPO's characteristic: **it completes a "gentle" directional shift with preference pairs; it can push, but not push far**. To truly "learn to solve problems and gain capability from it," you need the next step: RL.

---

## GRPO: Reinforcement Learning under Verifiable Rewards

GRPO is the algorithm behind DeepSeek-R1, and it's the protagonist of this tutorial's RL part. Its beauty lies in — throwing away PPO's (Proximal Policy Optimization) value network.

### Fundamentals

GRPO (Group Relative Policy Optimization), proposed by DeepSeek, belongs to **online (on-policy) reinforcement learning** — the model keeps learning while generating new responses in real time. Its training is a closed loop:

1. For one prompt, use the current policy to sample **a whole group** of $G$ responses;
2. Score each with a reward function (a verifiable reward or a reward model);
3. **Use that group's own mean/std as the baseline** to compute the within-group relative advantage;
4. Update the policy with this advantage, then return to step 1.

The key innovation is in step 3: instead of training a value network to estimate a per-token baseline, it lets the group of responses sampled from the same prompt **serve as references for one another**. A response's advantage is just "how much better than its groupmates it is," and the entire response shares this **single** advantage value:

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$

Plug it into a clipped policy-gradient objective, then subtract a KL penalty with coefficient $\beta$ to keep the policy from straying too far from the reference (where $r_{i,t}$ is the probability ratio between the new and old policy at token $t$):

$$\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Big[\tfrac{1}{G}\sum_{i=1}^{G}\tfrac{1}{|o_i|}\sum_{t}\min\big(r_{i,t}\,A_i,\ \text{clip}(r_{i,t},\,1-\epsilon,\,1+\epsilon)\,A_i\big)\Big] - \beta\, D_{\text{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)$$

So **there's no critic to train and no value loss** — precisely what makes it runnable on a small GPU. By contrast, PPO has to do fine-grained per-token advantage estimation with a value model, costing far more VRAM; the trade-off is that GRPO shares one advantage across the whole response, at coarser granularity.

This section trains on `google/IFEval`: the dataset provides instructions with **machine-verifiable constraints** (e.g. "use all lowercase," "exactly 3 bullets," "contain a given keyword N times"). This lets the reward be scored by pure rules, with no second model needed. **The reward is designed in tiers**: from "is the format right" to "are the constraints satisfied," scoring progressively so that even early in training there's within-group variance (otherwise all-zero scores → all-zero advantages → no gradient). It also adds anti-farming terms (e.g. `distinct_ratio` penalizing repeated-token padding) to stop the model from gaming a high score with degenerate output.

- **Pros**: no critic, VRAM-friendly, simple to implement, and naturally suited to tasks with a "verifiable reward" (math, code, instruction following).
- **Cons (capability boundary)**: it uses reward to push parameters toward "highest score," and because it's on-policy, it tries not to deviate from the model's own distribution — which is both an advantage (stable, forgets little) and a limitation: it **can only amplify behavior the model already knows how to sample, not create it from nothing**.
- **Cons (length bias)**: note the $\tfrac{1}{|o_i|}$ length normalization in the objective. For a **wrong** response (with negative advantage), dividing by a larger $|o_i|$ dilutes the per-token penalty, so the optimization tends to **make wrong responses longer** to soften the penalty — this is especially pronounced on hard problems where the model can't find the correct solution and the whole group's advantages are negative, showing up as responses getting longer with training rather than more accurate ([Dr. GRPO](https://arxiv.org/abs/2503.20783) points this out and argues for removing both the $\tfrac{1}{|o_i|}$ and the std-normalization biases). Our Countdown experiment stepped on this exact trap: when accuracy was too low on 1.5B, GRPO at one point stretched responses into repetitive long text — more on this later.

### Core Code

```python
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    num_generations=16,            # group size G: sample 16 responses per prompt
    max_completion_length=200,
    temperature=1.0,               # sampling temperature, creates within-group diversity
    beta=0.002,                    # KL penalty coefficient
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    use_vllm=True,                 # vllm speeds up rollouts
    bf16=True,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=REWARD_FUNCS,     # verifiable rewards: constraint satisfaction + anti-farming
    processing_class=tokenizer,
)
grpo_trainer.train()
```

### Experiment and Results

```bash
uv run python -m src.grpo
```

On IFEval held-out, instruction satisfaction rises from a **baseline of ~0.31 to ~0.58**: after training the model clearly follows constraints better and scores higher. Look at a concrete sample (`GRPO_SAMPLE_INDEX=1 uv run python -m src.grpo`), where the instruction asks to "write a TLDR in conversational **bullet points**":

```text
Prompt:
Write a TLDR ... in conversational bullet points. End your response with this exact
phrase: "Let me know if you have additional questions."

# ===== Before training =====
ISIL (Islamic State) and the United States have been at odds since 2014, fighting
over control of Mosul ... Let me know if you have any more questions!
(one big plain paragraph, no bullet points)

# ===== After GRPO =====
*ISIL attacks US embassy in Djibouti*
*US responds by sending military aid to ISIL*
*ISIL retreats from Djibouti but continues to attack neighboring countries*
...
```

The contrast is clear: **GRPO taught the model the bullet-point structure the instruction required** (the verifiable format constraint is now satisfied). But don't overrate it — this is a 135M small model, the content is obviously fabricated (the history is all wrong), and the exact ending phrase isn't fully matched either. **GRPO improves the ability to "follow verifiable constraints," not factual correctness** — consistent with the later Countdown section's conclusion: RL amplifies behavior/structure, not knowledge. RL's "boosting task capability" is immediate — and its deeper property (learns more, forgets less) is exactly what the next section quantifies.

---

## RL vs SFT: Why Online RL Learns While Forgetting Less

**This section answers a core question: when learning the same new task, what exactly is the difference between RL and SFT? The answer is — RL gains new capability while sacrificing almost none of its old capability.**

### Background

[RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/pdf/2509.04259) shows that, at comparable performance on a new task, **RL preserves prior capabilities significantly better than SFT**. The essence of forgetting is distribution shift — when learning a new task the model's output distribution drifts overall, and the farther it moves from the original base, the more prior capability is lost. This "drift" can be measured by KL divergence (intuitively, larger KL = the post-training distribution is farther from the base). The paper's core insight: on-policy RL is implicitly biased toward "the solution closest to the base among those that solve the new task" (KL-minimal), whereas SFT may converge to a distribution arbitrarily far from the base.

![RL's Razor](https://cdn.jsdelivr.net/gh/pochenai/nano-llm-posttraining@main/assets/figures/rl_vs_sft.png)

This section reproduces the conclusion with a minimal experiment: comparing three training methods — in-dist SFT trained on data sampled from the base model, off-dist SFT trained on data outside the base distribution (answers written by Claude), and GRPO — and looking at their drift and capability retention on the same new task (IFEval).

> The paper has a key simplification: you only need to compute the KL **on the new task's inputs** to characterize forgetting, with no need to compute KL across all tasks (that would be far too expensive). This article follows that setting.

### How to Measure "Drift": Forward KL per Token

Denote the base model as $\pi_0$ and the fine-tuned model as $\pi_\theta$. What we want is "how far $\pi_\theta$ has drifted relative to $\pi_0$ on the new task's inputs." KL is directional; here we use **forward KL** — the expectation taken under the base distribution:

$$D_{\text{KL}}(\pi_0 \,\|\, \pi_\theta) = \mathbb{E}_{y \sim \pi_0}\big[\log \pi_0(y) - \log \pi_\theta(y)\big]$$

Why choose forward (sampling from the base) rather than reverse (sampling from each model)? Because this way all three fine-tuned models are scored on **the same batch of completions sampled from the base**, so the KL differences reflect only "where each model redistributed its probability mass," not "their sampling distributions differ" — that's a fair side-by-side comparison. (Reverse KL happens to be exactly what the RL objective implicitly suppresses, and it also matches the `kl` logged by the trainer, but it samples from each model's own distribution with different scoring sets, so it's unsuitable for comparison.)

Per position, instead of a single-token estimate, we compute the **exact categorical KL** over the whole vocabulary (offline analysis, so we can pursue low variance):

$$\text{kl}(t) = \sum_{v \in \mathcal{V}} \pi_0(v \mid y_{\lt t})\,\big[\log \pi_0(v \mid y_{\lt t}) - \log \pi_\theta(v \mid y_{\lt t})\big]$$

Summing over every position of the completion and dividing by the token count gives the **forward KL per token** — the horizontal axis of the scatter plot below. Full implementation in [`src/kl_analysis.py`](src/kl_analysis.py).

### Experiment Setup

- Code: training `src/offdist_matched.py`, `src/grpo.py`; KL measurement `src/kl_analysis.py`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Data: all three methods share the **same prompts** from `google/IFEval`, so the only variable is "data source / update rule"
- Machine: a single 8GB GPU; 3 seeds per method, about 20 minutes per run

### Results Analysis

#### Reproduction Commands

```bash
uv run python -m src.offdist_matched
uv run python -m src.grpo
# measure forward KL for each checkpoint (writes to trainer_output/kl_runs.json)
KL_MODEL=<checkpoint> KL_LABEL=<name> uv run python -m src.kl_analysis
uv run python src/post-plot/razor_pareto.py   # -> assets/figures/fig_razor_pareto.png
```

#### One Figure: RL Beats SFT on Both Drift and Skill Simultaneously

![Figure 1](https://cdn.jsdelivr.net/gh/pochenai/nano-llm-posttraining@main/assets/figures/fig_razor_pareto.png)

The horizontal axis is drift from the base (forward KL per token), the vertical axis is task skill (IFEval satisfaction). **GRPO sits firmly in the top-left** — highest skill and smallest drift, better than both SFT points on both dimensions at once; while in-dist / off-dist SFT are both crammed into the bottom-right (higher drift, lower skill). This conclusion holds across 3 seeds. (Strictly speaking, each method here only takes a single point corresponding to one training config, which isn't enough to draw a full Pareto frontier; what we're claiming is "this one GRPO point is better on both axes," not that we've swept out a frontier curve.)

There's an easy misconception to clear up here: **GRPO's low KL is mainly not squeezed out by that `beta` KL-penalty term**. That term isn't a strict regularizer — as training steps accumulate the policy still drifts slowly; the real reason GRPO drifts little is that it's **on-policy** — it updates only on samples within its own distribution.

#### An Honest Caveat: The KL Magnitude Itself Is Fragile

It should be noted that **the KL magnitudes of in-dist and off-dist SFT are actually similar** (after sufficient training both converge to close values). The reason is that in-dist labels come from the base's best-of-K sampling (the high-reward tail), so training on them is itself a distribution shift, not "staying at the base." So "off-dist must drift farther" doesn't hold reliably, and the **absolute magnitude** of KL shouldn't be over-interpreted. What's truly robust is the **directional conclusion**: the RL family (on-policy) drifts clearly less than SFT.

#### Confirming "Forgetting" with a More Sensitive Metric: From Accuracy to Perplexity

Since the KL magnitude is fragile, we go further and directly measure "whether old capability has degraded."

First, **old-task accuracy**:

![retention](https://cdn.jsdelivr.net/gh/pochenai/nano-llm-posttraining@main/assets/figures/fig_retention_flat.png)

The result is that SFT and GRPO retention are **about the same, nearly flat**. But this doesn't mean "no forgetting" — more likely, accuracy as a metric is **too coarse**: the model itself is small, IFEval has little correlation with the base's pretraining tasks, and there's little training data, so a coarse-grained accuracy can't read out subtle degradation.

So we switch to a **more sensitive metric — the perplexity (PPL) of held-out general text**. Using lm-eval to measure word perplexity on WikiText and Pile, lower is better:

| Model | WikiText PPL | vs base | Pile PPL | vs base |
|---|---|---|---|---|
| baseline | 24.1 | — | 118.9 | — |
| in-dist SFT | 26.7 | **+11%** | 143.6 | **+21%** |
| off-dist SFT | 26.6 | **+10%** | 141.8 | **+19%** |
| **GRPO** | 24.2 | **+0.3%** | 119.4 | **+0.4%** |

The signal is suddenly clear: **both SFTs raised general PPL by 10–21% (general language ability genuinely degraded), while GRPO barely moves (+0.3%)**. The forgetting that coarse-grained accuracy failed to read, the sensitive PPL reads out.

### Summary

KL (drift on the task) and PPL (degradation of general capability) are **two different measurement dimensions**, yet they point to the same conclusion: **RL learns a new task while barely damaging its old capability, whereas SFT pays the price of clear general-capability degradation.** Two independent metrics corroborating each other is more credible than reading KL magnitude alone (which we've shown is fragile) — and this is a clean small-scale reproduction of RL's Razor.

---

## Aha Moment: GRPO Strengthens Self-Verification and Search into a Stable Strategy

**RL doesn't just make a model "forget less" — on the right task and scale, it can also strengthen the model's intrinsic reasoning behavior. The famous aha moment during DeepSeek-R1 training is the model learning to self-correct. This section reproduces the phenomenon at extremely low cost; it also shows an honest boundary: we use an Instruct model, so what we see is "amplification," not true "emergence."**

### Background

During RL training, DeepSeek-R1 observed the aha moment: the model learns to stop midway and correct its own mistakes. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) reproduced a similar result on the **Countdown** task — a fascinating phenomenon showing that chain-of-thought (CoT) can "squeeze out" a model's intrinsic ability.

> **"Emergence" or "amplification"?** A terminology point first. DeepSeek-R1 uses a **base model**, where the reflective behavior truly *emerges* from nothing. But to make training runnable, this article uses an **Instruct model** — which already knows how to "enumerate combinations + judge each one" (the `which is not helpful` in the base output below is a form of naive self-verification), just very badly: it falls into repetition and never reaches the correct solution. So what GRPO does here is converge/**amplify** this **already-present but only-occasionally-sampled** behavior into a stable strategy, rather than conjure a new capability from scratch. This is consistent with the article's throughline: "RL only amplifies, it doesn't create."

TinyZero's original required about 10 H100s at the $30 level. **This article pushes the cost to the extreme: rent a 48GB GPU and train for under 5 hours (cost under $5)** to reproduce the phenomenon.

The Countdown task: given a few numbers and a target, use `+ - * /` and parentheses, with each number used exactly once, to write an expression equal to the target. It's essentially a **search** — many attempts fail, so the natural solving process is "try X → wrong → try Y," exactly the backtracking / self-correction behavior we want to see. The reward is scored by pure rules: **correctness dominates** (full score if each number is used once and the value equals the target), aided by **proximity** (larger gradient the closer the value gets to the target when the exact numbers are used, forcing the model to actually search rather than guess randomly), with format taking only a tiny share.

> **Why does the model need to be at least 3B?** We first tried 1.5B, but the base model is too weak — on tasks like Countdown/GSM8K its accuracy is already under 5%, so GRPO can barely sample any rewarded positive examples to drive training; moreover its "long output" is actually a **repetition loop** rather than real search, so GRPO merely cuts the repetition short and won't amplify any effective search. Also, tasks like GSM8K that "don't really need reasoning" aren't good at surfacing the phenomenon — you must switch to a **search-heavy** task like Countdown. All told, 3B is the starting point where the phenomenon becomes visible.

### Experiment Setup

- Machine: 48GB GPU (vllm for fast inference + LoRA (Low-Rank Adaptation) to reduce VRAM)
- Task: Countdown
- Model: `Qwen/Qwen2.5-3B-Instruct`

```bash
# run it
uv sync --extra vllm

GRPO_COT_RUN=3b-v2 GRPO_COT_LORA=1 GRPO_COT_LR=2e-5 \
GRPO_COT_BATCH=2 GRPO_COT_GRAD_ACCUM=32 GRPO_COT_NUM_GENERATIONS=16 \
GRPO_COT_TRAIN_LIMIT=2000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m src.grpo_countdown
```

### Results

Take test-set #25 as an example (full log at [log](https://cdn.jsdelivr.net/gh/pochenai/nano-llm-posttraining@main/src/log_grpo_countdown_checker.txt)):

```
===== Q#25  target=52  nums=[77, 33, 78, 86] =====
# Combine the numbers 77, 33, 78, 86 (each used exactly once) with + - × ÷ () to make 52.
```

**Before training, the output of the base model (Qwen/Qwen2.5-3B-Instruct):**

```
----- BASE  eq='(78 - 86) + (77 - 33)'  value=36  correct=False -----
<think>
...
Let's try another combination. If we subtract 33 from 77, we get 44...
Let's try: if we subtract 33 from 77, we get 44...
Finally, let's try: if we subtract 33 from 77, we get 44...   (repeating the same line over and over)
...
<answer> (78 - 86) + (77 - 33) = 52 </answer>
</think>
```

You can see the base merely repeats `if we subtract 33 from 77` in the second half — **it doesn't truly possess self-correction ability, it's just spinning in place, and the final answer it gives is wrong too** ((78−86)+(77−33)=36≠52).

**After GRPO training, the output on the same problem:**

```
----- GRPO  eq='77 - 33 - 78 + 86'  value=52  correct=True -----
<think>
First, let's consider the possibility of subtraction and addition:
- 77 - 33 + 78 + 86 = 128 (too high)
- 77 + 33 - 78 + 86 = 112 (too high)
...
Next, let's try multiplication and division:
- 77 * 33 / 78 + 86 = 33 (too low)
...
After trying various combinations, it seems we need to re-evaluate the approach.
Let's try a different combination:
- 77 - 33 + 78 - 86 = 26 (too low)
- 77 - 33 - 78 + 86 = 52 (this works)
</think>
<answer> 77 - 33 - 78 + 86 </answer>
```

The contrast is striking: after GRPO the model learned to **systematically enumerate candidates**, annotate each with too high / too low, even explicitly "re-evaluate the approach" partway through, and finally search out the correct solution `77 - 33 - 78 + 86 = 52`. This is exactly the **search + self-verification** structure the base lacked.

But to be honest about one point: if you verify its intermediate steps one by one, you'll find the arithmetic itself is often wrong — e.g. it writes `77 - 33 + 78 + 86 = 128` (actually 208), `77 * 33 / 78 + 86 = 33` (actually about 119); 4 out of 5 candidates have miscalculated values. So what GRPO truly strengthened here is **mostly not** reliable step-by-step mental arithmetic, but the **search process** of "enumerate → judge one by one → hit the final-answer check": the reward only rewards whether the final `<answer>` is correct (correctness dominates) and doesn't penalize intermediate miscalculations, so the model learns the *structure* of exhaustive search plus self-checking, and even with noisy intermediate arithmetic it can still be caught by that one correct-solution step.

It should be noted that "arithmetic not getting more accurate" is more likely **a result of this reward design** (only looking at the final answer, ignoring the process) rather than "RL fundamentally can't teach arithmetic" — after all, a 3B small model is weak at arithmetic to begin with, and even a large model like Claude occasionally miscalculates. With a reward that penalizes intermediate steps, arithmetic might well improve. But at least in this experiment, the observed phenomenon is consistent with the throughline: **GRPO is more like amplifying the "search + self-check" behavior the model could already occasionally sample into a stable strategy, rather than pouring in new knowledge.**

The reward curve during training is shown below; you can see the reward, amid oscillation, keeps rising overall — the model is steadily getting better at solving:

![countdown reward curve](https://cdn.jsdelivr.net/gh/pochenai/nano-llm-posttraining@main/assets/figures/fig_countdown_curve.png)

### Summary

On a 3B model and Countdown, a search-heavy task, GRPO **strengthened the model's already-sporadic behaviors of systematic enumeration, self-verification, and "re-evaluation" into a stable search strategy** — reproducing the DeepSeek-R1/TinyZero aha moment in under $5 and 5 hours. Saying "strengthen" rather than "emerge" is deliberate: the base (Instruct) already does enumeration and naive judgment, and what GRPO does is converge and amplify. It again confirms GRPO's essence: **amplifying capabilities the model already has, not creating from nothing** — so the task must depend heavily on reasoning and the model scale must be large enough to sample search behavior for the phenomenon to appear.

---

## Overall Summary

Following the minimal, reproducible throughline of SFT → DPO → GRPO, using one 8GB GPU plus a single $5 rental of a 48GB machine, we've clearly seen several counterintuitive post-training phenomena one by one:

- **SFT / DPO**: writing a single pattern or preference direction into the model — simple and direct, but they can only "gently nudge" and can't create capabilities the model doesn't have.
- **GRPO**: on-policy RL under verifiable rewards — it substantially boosts task capability while **forgetting less** thanks to updating close to its own distribution — confirmed by two independent metrics, KL and PPL.
- **Aha moment**: on tasks that depend heavily on search and at a sufficient model scale, RL can **strengthen/amplify** the model's **already-sporadic** intrinsic reasoning behavior into a stable search and self-verification strategy (with an Instruct model, so it's "strengthening" rather than "emergence" from nothing).

One intuition runs throughout: **RL doesn't teach the model new knowledge; within its existing distribution, it picks out and strengthens the behaviors that "solve the problem while staying closest to itself."** This explains both why it forgets less and why it depends on capabilities "the model can already sample."

---

## References

- [Post-training of LLMs by Banghua Zhu](https://www.deeplearning.ai/courses/post-training-of-llms)
- [TinyZero by Jiayi Pan](https://github.com/Jiayi-Pan/TinyZero)
- [Train llm from scratch by FareedKhan](https://github.com/FareedKhan-dev/train-llm-from-scratch)
- [RL's Razor: Why Online RL Forgets Less](https://arxiv.org/pdf/2509.04259)
- [David Silver: Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
