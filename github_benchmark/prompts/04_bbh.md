# BBH (Big-Bench Hard)

**Benchmark:** Reasoning proxy for HMMT (HF card 74.0/76.8)

**Prompt:**
```
Solve the following reasoning problem step by step.

Question: In a group of 100 people, 60 like apples, 50 like bananas, and 30 like both. How many like neither?

Answer:
```

**Model waits for this prompt.**

**Model:** `Qwen/Qwen3.5-4B` vs `RohitSwami33/qwen35-fc-adapter` (190M LoRA r32) — https://huggingface.co/RohitSwami33/qwen35-fc-adapter
**Model Kaggle:** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-adapter — `kaggle datasets download roronoazoro3008/qwen35-fc-adapter`
**Dataset (fine-tune data):** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-dataset — `kaggle datasets download roronoazoro3008/qwen35-fc-dataset` (25,037 tokenized seqs, 6.58M tokens, eval holdout 774)
