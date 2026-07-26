# 普通 8GB 显卡 8 小时上手 LLM Post-Training！完整简洁教程

## TL;DR

- **不到 100 行核心代码**，就能在一张 8GB 显卡、基于 0.1B 的小模型上，把 **SFT + DPO + GRPO** 这套后训练流程完整跑通。
- 用最小的实验复现了 **RL's Razor** 的核心结论：**同样学会新任务，on-policy 的 RL 比 SFT 遗忘更少**——它在新任务上的 KL divergence 更低，通用语言能力也几乎不退化。
- 再租一台不到 5 美元的机器、训练 5 小时，就能在 3B 模型上做 GRPO，**亲眼看到模型自发涌现出 self-verification 与 search 能力（aha moment）**。

本文的目标不是又一个"怎么训"的框架，而是用**最小可复现的实验**，把后训练里几个反直觉的现象——遗忘、on-policy 的作用、推理涌现——一个个"跑"出来、看明白。

---

## Motivation

LLM 训练的两个痛点，几乎劝退了所有独立开发者。

**一是算力成本**：从零训一个像样的模型动辄数万美元，个人根本负担不起。**二是知识门槛**：强化学习的基础理论浩如烟海，把 David Silver 的 RL 课程完整过一遍，上班族往往要花上一两个月，时间成本同样难以承受。

针对这两个痛点，本教程做了两点极致压缩：**模型足够小**（0.1B 起步，8GB 显卡即可），**只讲和 LLM 直接相关的 RL 算法**（大量经典 RL 基础算法在 LLM 后训练里其实用不到）。这样一天之内就能上手，真切感受到后训练每一步"改变了什么"。

> 说明：本文不从零推导 GRPO 的公式与算法，重点在"用实验把现象看清楚"。

---

## Supervised Fine-Tuning (SFT)

**SFT 是后训练的第一步，作用是把一个"只会续写"的 base model 教成"会听指令"的助手。**

### 基础

Base model 能续写文本，却不知道"你在向它提问、它该回答你"。SFT 用成千上万条 `(instruction, response)` 配对做 teacher forcing，让模型学会：看到指令，就产出对应的回答。

本节用一份**身份数据**做最直观的演示——数据里每条 response 都把模型的身份说成 "Qwen"。训练后，`HuggingFaceTB/SmolLM2-135M-Instruct` 被"洗"成了以 Qwen 自居。现象越单一，越容易看清 SFT 到底改了什么。

### 核心代码

```python
model, tokenizer = load_model_and_tokenizer(model_name=SFT_MODEL, use_gpu=True)

# 有效 batch size = per_device_train_batch_size × gradient_accumulation_steps
# 下面这套配置下，135M 模型约占 8GB 显存
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-4,
    num_train_epochs=1,
    per_device_train_batch_size=8,   # 单卡 batch，压到能塞进 8GB
    gradient_accumulation_steps=4,   # 累积成有效 batch = 8 × 4 = 32
    bf16=True,                       # 混合精度：省激活显存，50 系显卡上更快
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

### 实验与结果

```bash
uv run python -m src.identity_sft
```

训练后，模型面对"你是谁"类问题会稳定自称 Qwen——**SFT 成功地把一个单一模式写进了模型的输出分布**，这也为下一步 DPO 打好了底子。

---

## Direct Preference Optimization (DPO)

**DPO 是 RLHF 的"抄近路"版本：不训 reward model、不跑 RL 循环，直接在偏好对上优化策略，把模型往"更被偏好"的方向轻推。**

### 基础

DPO（Direct Preference Optimization）绕开了 RLHF 那套"先训 reward model、再跑 RL"的重流程：它直接在 `(chosen, rejected)` 偏好对上优化策略，并用一份冻结的 SFT 模型作为 reference anchor。其损失函数为：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\log \sigma\Big(\beta \big(\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big)\Big)\Big]$$

直觉上，它在拉高 chosen 相对 reference 的概率、压低 rejected 的概率。**参数 $\beta$ 控制"偏离 reference 的力度"**：$\beta$ 越大，越贴近 reference、更新越保守；$\beta$ 越小，越激进、也越容易训崩。

本节延续 SFT 的模型，用 DPO 把身份进一步从 "Qwen" 推向 "Deep Qwen"。

- **Pros**：不需要额外的 reward model 和 RL 循环，实现简单、训练稳定，适合"往某个明确方向轻推"的场景。
- **Cons**：它只能把模型**轻微**地推向偏好方向；如果模型本身没有对应的知识或能力，DPO 很难凭空造出来。

### 核心代码

```python
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)

# 现成的 1k 偏好对，chosen/rejected 主要差别就在身份（"Deep Qwen" vs "Qwen"），
# 是个干净的单方向信号。
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

# 约 58% 的样本 chosen == rejected（数据用 name-replace 技巧构造，非身份 prompt 上没生效），
# 这些样本梯度为 0，直接丢掉，让每一步都有信息量。
dpo_ds = dpo_ds.filter(
    lambda r: r["chosen"][-1]["content"] != r["rejected"][-1]["content"]
)

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.2,                        # KL 强度：越大越贴近 reference
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # 有效 batch = 4 × 4 = 16
    num_train_epochs=1,
    # DPOConfig 默认 1e-6。5e-5 在 135M 上跑满一个 epoch 会把模型训成乱码——
    # 因为力度过大时，DPO 会把 chosen 和 rejected 的 logprob 一起往下拖。
    learning_rate=5e-6,
    logging_steps=10,
    bf16=True,
    save_total_limit=1,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # None -> 自动用 model 的冻结副本当 reference
    args=config,
    processing_class=tokenizer,
    train_dataset=dpo_ds,
)
```

### 实验与结果

```bash
uv run python -m src.dpo
```

训练后模型自称从 "Qwen" 迁移到 "Deep Qwen"——**DPO 用偏好对完成了一次"温和"的方向迁移**，也印证了它的边界：能推、但推不远。要真正"学会解题、并从中获得提升"，就得靠下一步的 RL。

---

## GRPO：可验证奖励下的强化学习

**GRPO 是 DeepSeek-R1 背后的算法，也是本教程 RL 部分的主角。它的妙处在于——扔掉 PPO 的 value network。**

### 基础

GRPO（Group Relative Policy Optimization）对每个 prompt 采样一整**组**回答，用可验证的 reward 给每条打分，然后**直接用这一组自己的 mean/std 当基线**。某条回答的 advantage 就是"它比同组的兄弟们好多少"：

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$

于是**不需要训练 critic、没有 value loss**——这正是它能在小卡上跑起来的关键。更新时用带 clip 的 policy gradient，外加一个系数 $\beta$ 的 KL penalty 约束不要离 reference 太远。

本节在 `google/IFEval` 上训练：数据集给出一批带**可机器验证约束**的指令（如"用全小写""恰好 3 个 bullet""包含某关键词 N 次"）。这让 reward 可以纯规则打分、无需第二个模型。**reward 设计成分层**：从"格式对不对"到"是否满足约束"逐级给分，保证训练早期组内也有方差（否则全是 0 分 → advantage 全 0 → 没有梯度）；同时加入防刷分项（如 `distinct_ratio` 惩罚重复 token 凑数），避免模型用退化输出骗高分。

- **Pros**：无 critic、显存友好、实现简单，且天然适合"有可验证 reward"的任务（数学、代码、指令遵循）。
- **Cons**：它靠 reward 把参数往"得分最高"的方向推，同时因为是 **on-policy** 的，会尽量不偏离模型自身的分布——这既是优点（稳、遗忘少），也意味着它**只能放大模型已经会采样的行为，无法凭空创造**。

### 核心代码

```python
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    num_generations=16,            # 组大小 G：每个 prompt 采 16 条
    max_completion_length=200,
    temperature=1.0,               # 采样温度，制造组内多样性
    beta=0.002,                    # KL penalty 系数
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    use_vllm=True,                 # vllm 加速 rollout
    bf16=True,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=REWARD_FUNCS,     # 可验证奖励：约束满足度 + 防刷分
    processing_class=tokenizer,
)
grpo_trainer.train()
```

### 实验与结果

```bash
uv run python -m src.grpo
```

在 IFEval held-out 上，指令满足度从 **baseline 的 ~0.31 提升到 ~0.58**：**训练后的模型明显更能遵循约束、得分更高**。RL 的"提升任务能力"立竿见影——而它更深一层的性质（学得多、忘得少），正是下一节要量化的东西。

---

## RL vs SFT：为什么 online RL 学得会、又忘得少

**这一节回答一个核心问题：同样把新任务学会，RL 和 SFT 到底有什么不同？答案是——RL 在获得新能力的同时，几乎不牺牲旧能力。**

### 背景

[RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/pdf/2509.04259) 指出：在新任务表现相当的前提下，**RL 对先验知识和通用能力的保持显著优于 SFT**。遗忘的本质是 **distribution shift**，可以用 **KL divergence** 来衡量；而 on-policy RL 隐式偏好"能解新任务的解里、离 base 最近的那个"（KL-minimal），SFT 却可能收敛到离 base 任意远的分布。

![RL's Razor](assets/figures/rl_vs_sft.png)

本节用最小实验尝试复现这一结论：对比三种训练方式——在 **base model 上采样数据**训练的 **in-dist SFT**、在 **base 分布之外的数据（Claude 写的答案）**上训练的 **off-dist SFT**、以及 **GRPO**——看它们在同一个新任务（IFEval）上的 forward KL divergence 与能力保持。

> 论文有一个关键简化：只需计算**新任务输入上**的 KL divergence 即可刻画遗忘，无需在所有任务上算（那样计算量太大）。本文沿用这一设定。

### 实验设置

- 代码：`src/offdist_matched.py`、`src/grpo.py`
- 模型：`HuggingFaceTB/SmolLM2-135M-Instruct`
- 数据：三种方式共用 `google/IFEval` 里**相同的 prompt**，保证只有"数据来源 / 更新规则"这一个变量
- 机器：单张 8GB 显卡；每种方式各跑 **3 个 seed**，单次训练约 20 分钟

### 结果分析

#### 复现命令

```bash
uv run python -m src.offdist_matched
uv run python -m src.grpo
uv run python src/post-plot/razor_pareto.py   # -> assets/figures/fig_razor_pareto.png
```

#### 一图看懂：RL 在 (漂移, 技能) 平面上 Pareto 支配 SFT

![Figure 1](assets/figures/fig_razor_pareto.png)

横轴是离 base 的漂移（forward KL per token），纵轴是任务技能（IFEval 满足度）。**GRPO 稳稳落在左上角**——技能最高、漂移最小，Pareto 支配了两个 SFT 点；而 in-dist / off-dist SFT 都挤在右下（更高漂移、更低技能）。这条结论在 3 个 seed 上依然成立。

这里有一个容易误解的点需要澄清：**GRPO 的低 KL 主要不是靠 `beta` 那个 KL penalty 项压出来的**。那一项并非严格正则，随着训练步数增多策略照样会缓慢漂移；GRPO 漂移小的真正原因，是它 **on-policy**——只在自身分布内的样本上更新。

#### 一个诚实的 caveat：KL 幅度本身是脆弱的

需要说明的是，**in-dist 与 off-dist SFT 的 KL 幅度其实差不多**（充分训练后都收敛到相近的值）。原因是 in-dist 的标签来自 base 的 best-of-K 采样（高奖励尾巴），训它本身也是一次分布迁移，并非"停在 base"。所以"off-dist 一定漂移更远"这条并不稳，KL 的**绝对幅度**不宜过度解读。真正稳的是**方向性结论**：RL 这一类（on-policy）明显比 SFT 漂移小。

#### 用更敏感的探针确认"遗忘"：从 accuracy 到 perplexity

既然 KL 幅度脆弱，我们进一步直接测"旧能力有没有退化"。

先看**旧任务 accuracy**：

![retention](assets/figures/fig_retention_flat.png)

结果是 SFT 和 GRPO 的保持度**都差不多、几乎持平**。但这并不代表"没遗忘"——更可能是 accuracy 这个探针**太粗**：模型本身很小、IFEval 与 base 预训练任务相关性不大、训练数据又少，粗粒度的准确率读不出细微的退化。

于是换一个**更敏感的探针——held-out 通用文本的 perplexity（PPL）**。用 lm-eval 在 WikiText 和 Pile 上测 word perplexity，越低越好：

| 模型 | WikiText PPL | vs base | Pile PPL | vs base |
|---|---|---|---|---|
| baseline | 24.1 | — | 118.9 | — |
| in-dist SFT | 26.7 | **+11%** | 143.6 | **+21%** |
| off-dist SFT | 26.6 | **+10%** | 141.8 | **+19%** |
| **GRPO** | 24.2 | **+0.3%** | 119.4 | **+0.4%** |

信号一下子清晰了：**两个 SFT 都把通用 PPL 抬高了 10–21%（通用语言能力实实在在退化了），而 GRPO 几乎不动（+0.3%）**。粗粒度的 accuracy 没读出的遗忘，敏感的 PPL 读出来了。

### 小结

KL（任务上的漂移）与 PPL（通用能力的退化）是**两个不同的仪器**，却指向同一个结论：**RL 学会新任务的同时几乎不损伤旧能力，SFT 则以明显的通用能力退化为代价。** 两个独立仪器互相印证，比单看 KL 幅度（那条已被证明脆弱）更可信——这正是 RL's Razor 在小尺度上的一次干净复现。

---

## Aha moment：模型自发涌现出 self-verification 与 search

**RL 不只让模型"忘得少"，在合适的任务和尺度上，它还能让模型自发学会推理——DeepSeek-R1 训练中著名的 aha moment，就是模型学会了自我纠错。本节用极低成本复现这一现象。**

### 背景

DeepSeek-R1 在 RL 训练中观察到 aha moment：模型学会中途停下来纠正自己的错误。[TinyZero](https://github.com/Jiayi-Pan/TinyZero) 在 **Countdown** 任务上复现了类似结果——这个现象非常有意思，它说明 chain-of-thought 能把模型的内在能力"逼"出来。

TinyZero 原版需要约 10 张 H100、成本 30 美元级别。**本文进一步把成本压到极致：租一张 48GB GPU、训练不到 5 小时（成本不到 5 美元）**，就能复现出这个现象。

Countdown 任务：给定几个数字和一个 target，用 `+ - * /` 和括号、每个数字恰好用一次，写出一个等于 target 的表达式。它本质是**搜索**——很多尝试会失败，天然的解题过程就是"试 X → 不对 → 试 Y"，正是我们想看到的 backtracking / 自我纠错行为。reward 用纯规则打分：**correctness 主导**（每数一次且值等于 target 给满分），辅以 **proximity**（用对数字且值越接近 target 梯度越大，逼模型真去搜索而非乱猜），format 只占很小一份。

> **为什么模型至少要 3B？** 我们先在 1.5B 上试过，但基础模型能力不足——Countdown/GSM8K 这类任务它的准确率本就不到 5%，GRPO 几乎采样不到有 reward 的正样本来推进训练；而且它的"长输出"其实是**复读循环**而非真搜索，GRPO 只会把复读杀短、并不会涌现出搜索。此外像 GSM8K 这类"不太需要推理"的任务也不容易出现象，必须换成 Countdown 这种**强依赖搜索**的任务。综合下来，3B 是能看到现象的起点。

### 实验设置

- 机器：48GB GPU（vllm 加速推理 + LoRA 降显存）
- 任务：Countdown
- 模型：`Qwen/Qwen2.5-3B-Instruct`

```bash
GRPO_COT_RUN=3b-v2 GRPO_COT_LORA=1 GRPO_COT_LR=2e-5 \
GRPO_COT_BATCH=2 GRPO_COT_GRAD_ACCUM=32 GRPO_COT_NUM_GENERATIONS=16 \
GRPO_COT_TRAIN_LIMIT=2000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m src.grpo_countdown
```

### 结果

以 test 集 #25 为例（完整日志见 [log](src/log.txt)）：

```
===== Q#25  target=52  nums=[77, 33, 78, 86] =====
```

**训练前，base model（Qwen/Qwen2.5-3B-Instruct）的输出：**

```
----- BASE  eq='(78 - 86) + (77 - 33)'  value=36  correct=False -----
<think>
...
Let's try another combination. If we subtract 33 from 77, we get 44...
Let's try: if we subtract 33 from 77, we get 44...
Finally, let's try: if we subtract 33 from 77, we get 44...   （反复重复同一句）
...
<answer> (78 - 86) + (77 - 33) = 52 </answer>
</think>
```

可以看出 base 在后半段只是不断重复 `if we subtract 33 from 77`——**它并不真正具备自我纠错能力，只是在原地打转，最终给出的答案也是错的**（(78−86)+(77−33)=36≠52）。

**GRPO 训练后，同一题的输出：**

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

对比非常鲜明：**GRPO 后的模型学会了系统性地枚举候选**（并对每个候选标注 too high / too low 做自我验证），中途还会显式地 "re-evaluate the approach"（重新审视思路），最终搜索到正确解 `77 - 33 - 78 + 86 = 52`。这正是 base 所缺失的 **self-verification + search** 能力——**GRPO 没有教给它任何新知识，只是把它本就能偶尔采样到的"搜索 + 验证"行为放大成了稳定策略。**

训练过程中的 reward 曲线如下，可以看到 reward 在震荡中整体持续上行——模型在稳定地变得更会解题：

![countdown reward curve](assets/figures/fig_countdown_curve.png)

### 小结

在 3B 模型、Countdown 这个强搜索任务上，GRPO **让模型自发涌现出系统枚举、自我验证和"重新审视"的搜索能力**——不到 5 美元、5 小时，就复现了 DeepSeek-R1/TinyZero 的 aha moment。它再次印证了 GRPO 的本质：**放大模型已有的能力，而非凭空创造**——所以任务要够依赖推理、模型尺度要够采样得到搜索行为，现象才会出现。

---

## 全文小结

沿着 SFT → DPO → GRPO 这条最小可复现的主线，我们用一张 8GB 显卡加一次 5 美元的 48GB 租用，把后训练里几个反直觉的现象逐个看清：

- **SFT / DPO**：把单一模式或偏好方向写进模型，简单直接，但只能"轻推"、造不出模型没有的能力。
- **GRPO**：可验证 reward 下的 on-policy RL，既大幅提升任务能力，又因贴着自身分布更新而**遗忘更少**——KL 与 PPL 两个独立仪器共同证实了这一点。
- **Aha moment**：在足够依赖搜索的任务和足够的模型尺度上，RL 能把模型内在的推理能力**放大**成稳定的搜索与自我验证策略。

一条贯穿始终的直觉是：**RL 不是在教模型新知识，而是在它已有的分布里，挑出并强化那些"能解题、又离自己最近"的行为。** 这既解释了它为什么忘得少，也解释了它为什么依赖"模型本就采样得到"的能力——尺度不够、任务不对，现象就不会出现。

---

## References

- [Post-training of LLMs by Banghua Zhu](https://www.deeplearning.ai/courses/post-training-of-llms)
- [TinyZero by Jiayi Pan](https://github.com/Jiayi-Pan/TinyZero)
- [Train llm from scratch by FareedKhan](https://github.com/FareedKhan-dev/train-llm-from-scratch)
- [RL's Razor: Why Online RL Forgets Less](https://arxiv.org/pdf/2509.04259)
- [David Silver: Reinforcement Learning](https://davidstarsilver.wordpress.com/teaching/)
