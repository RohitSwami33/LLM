import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="github_benchmark/model_weights/fc_adapter")
    p.add_argument("--out", default="model_export/qwen35_fc_merged")
    p.add_argument("--base", default=BASE)
    args = p.parse_args()

    adapter = Path(args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading base model:", args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Loading tokenizer/processor")
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(adapter, trust_remote_code=True)
    except Exception:
        processor = None
    print("Loading PEFT adapter:", adapter)
    model = PeftModel.from_pretrained(base, adapter)
    print("Merging adapter")
    model = model.merge_and_unload()
    print("Saving merged model to", out)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    if processor is not None:
        processor.save_pretrained(out)
    print("Done")


if __name__ == "__main__":
    main()
