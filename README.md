GRPO_COT_Q=1 uv run python -m src.grpo_cot_checker_aha

```
===== Q#1  gold=3  (K=16 samples @ temp=0.8) =====
Q: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

--- sample 7  correct=False  markers=['however'] ---
<reasoning>
To find out the total number of bolts of fiber needed to make a robe, we need to first determine how much white fiber is required and then add that to the number of bolts of blue fiber.
</reasoning>

<answer>
The total number of bolts of fiber needed is 2 (for blue) + 0.5 (for white) = 2.5 bolts. However, since we can't have half a bolt of fiber, we would round up to the nearest whole number. Therefore, it would take 3 bolts in total.
</answer>
```


GRPO_COT_MODEL=Qwen/Qwen2.5-3B GRPO_COT_RUN=3b GRPO_COT_LORA=1 \
GRPO_COT_BATCH=2 GRPO_COT_GRAD_ACCUM=16 GRPO_COT_NUM_GENERATIONS=16 \
GRPO_COT_TRAIN_LIMIT=2000 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m src.grpo_countdown
