# Benchmark Prompts - Friend's PC

This folder contains prompts for benchmarking `Qwen/Qwen3.5-4B` vs `RohitSwami33/qwen35-fc-adapter` (finetuned).

**Usage on friend's PC:**
1. Clone repo: `git clone <repo>`
2. For each `prompts/*.md`, copy prompt text and paste to model
3. Model waits for prompt — paste and get response, record metrics

**Structure:**
```
github_benchmark/
├── prompts/
│   ├── 01_mmlu_pro.md
│   ├── 02_hellaswag.md
│   ├── 03_gsm8k.md
│   ├── 04_bbh.md
│   ├── 05_gpqa.md
│   └── 06_ifeval.md
├── run_benchmark.py  # optional local runner
└── README.md
```

**Model:** Base `Qwen/Qwen3.5-4B` (HF) vs Adapter `RohitSwami33/qwen35-fc-adapter`
Adapter: `peft` `r=32 alpha=64` `Qwen3_5ForConditionalGeneration` `NF4` `trust_remote_code`
