# Model Weights (LoRA adapters, LFS)

Base model is NOT included. Download separately: `Qwen/Qwen3.5-4B`.

All `*.safetensors` files are stored with Git LFS.

## Direct download

- Final ORPO-v4 (recommended): `model_weights/final-orpo-v4/adapter_model.safetensors` (~190 MB)
- Also available in place:
  - `qwen35_post_training/model_v4_orpo/` — final ORPO-v4, loss 0.6868, 10k prefs, 7,923s
  - `qwen35_post_training/model_v3/` — replay-v3 SFT, loss 0.0871, 11,999 ex
  - `qwen35_post_training/model_science_v2/` — science-v2 SFT, loss 0.0388, 16,430 ex
  - `github_benchmark/model_weights/fc_adapter/` — original fact-check adapter

`model_weights/final-orpo-v4/` is a copy of `qwen35_post_training/model_v4_orpo/` for convenience (same LFS blob).

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base = "Qwen/Qwen3.5-4B"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant,
    device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "model_weights/final-orpo-v4")
model.eval()
```

## Publish notes

- `*.safetensors` tracked in `.gitattributes` via LFS — clone with `git lfs pull`.
- Full 4B weights are never committed here; only adapters (~85–190 MB each).
