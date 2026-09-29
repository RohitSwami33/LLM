# MMLU-Pro

**Benchmark:** Knowledge & STEM (HF card 79.1)

**Prompt:**
```
Answer the following multiple choice question. Think step by step and put your final answer within \boxed{}.

Question: Which of the following best describes the role of the ribosome in protein synthesis?
A) It carries amino acids to the mRNA
B) It catalyzes the formation of peptide bonds and decodes mRNA
C) It transcribes DNA into mRNA
D) It modifies proteins after translation

Answer:
```

**Expected:** B

**Model waits for this prompt — paste to base and finetuned, compare.**

**Model:** `Qwen/Qwen3.5-4B` vs `RohitSwami33/qwen35-fc-adapter` (190M LoRA r32) — https://huggingface.co/RohitSwami33/qwen35-fc-adapter
**Model Kaggle:** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-adapter — `kaggle datasets download roronoazoro3008/qwen35-fc-adapter`
**Dataset (fine-tune data):** https://www.kaggle.com/datasets/roronoazoro3008/qwen35-fc-dataset — `kaggle datasets download roronoazoro3008/qwen35-fc-dataset` (25,037 tokenized seqs, 6.58M tokens, eval holdout 774)
