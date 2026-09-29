# HellaSwag

**Benchmark:** Commonsense reasoning (general)

**Prompt:**
```
Complete the following scenario with the most plausible ending. Think step by step.

Scenario: A man is seen picking up a guitar and sitting on a stool. He adjusts the microphone and...
A) begins to play and sing into the microphone
B) starts cooking dinner on the stove
C) drives a car away from the building
D) swims across a river

Answer:
```

**Model waits for this prompt.**

**Model:** `Qwen/Qwen3.5-4B` vs `RohitSwami33/qwen35-fc-adapter` (190M LoRA r32) — https://huggingface.co/RohitSwami33/qwen35-fc-adapter
**Model Kaggle:** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-adapter — `kaggle datasets download roronoazoro3008/qwen35-fc-adapter`
**Dataset (fine-tune data):** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-dataset — `kaggle datasets download roronoazoro3008/qwen35-fc-dataset` (25,037 tokenized seqs, 6.58M tokens, eval holdout 774)
