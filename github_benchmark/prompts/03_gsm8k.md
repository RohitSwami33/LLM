# GSM8K

**Benchmark:** Math reasoning (HF card, complement to custom math_gsm8k_test)

**Prompt:**
```
Please reason step by step, and put your final answer within \boxed{}.

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at the farmers' market daily for $2 per egg. How much does she make every day at the farmers' market?

Answer:
```

**Model waits for this prompt.**

**Model:** `Qwen/Qwen3.5-4B` vs `RohitSwami33/qwen35-fc-adapter` (190M LoRA r32) — https://huggingface.co/RohitSwami33/qwen35-fc-adapter
**Model Kaggle:** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-adapter — `kaggle datasets download roronoazoro3008/qwen35-fc-adapter`
**Dataset (fine-tune data):** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-dataset — `kaggle datasets download roronoazoro3008/qwen35-fc-dataset` (25,037 tokenized seqs, 6.58M tokens, eval holdout 774)
