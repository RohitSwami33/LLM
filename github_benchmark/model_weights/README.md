# Model Weights

**Do NOT commit 190MB `adapter_model.safetensors` directly to GitHub** (100MB limit) — use `Git LFS` or `Hugging Face`.

**Base:** `Qwen/Qwen3.5-4B` (HF, Apache 2.0, 4B, `trust_remote_code`, `Qwen3_5ForConditionalGeneration`)
```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./base
```

**Adapter (finetuned):** `RohitSwami33/qwen35-fc-adapter` (pushed from `/tmp/local_bench/roronoazoro3008_qwen35-fc-resume/fc_adapter` `190M`)
```bash
huggingface-cli download RohitSwami33/qwen35-fc-adapter --local-dir ./model_weights/fc_adapter
# or via python
from huggingface_hub import snapshot_download
snapshot_download("RohitSwami33/qwen35-fc-adapter", local_dir="./model_weights/fc_adapter", token=os.environ["HF_TOKEN"])
```

**Files:**
- `adapter_config.json` (r=32 alpha=64)
- `adapter_model.safetensors` (190M, **LFS** if committing)
- `tokenizer.json` (19M)
- `chat_template.jinja`

**For GitHub LFS:**
```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes model_weights/fc_adapter/*
```

**Local path on friend's PC:** Place `fc_adapter` next to `prompts/` and run `run_benchmark.py` (loads base + peft).
