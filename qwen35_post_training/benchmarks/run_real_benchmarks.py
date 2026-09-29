#!/usr/bin/env python3
"""Run seven dataset-backed lm-eval tasks on Qwen3.5-4B or its PEFT adapter."""
import argparse
import json
import time
from pathlib import Path

import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = "RohitSwami33/qwen35-fc-adapter"
TASKS = [
    "mmlu_pro_computer_science",
    "truthfulqa_mc2",
    "gsm8k",
    "hellaswag",
    "bbh_cot_zeroshot_logical_deduction_five_objects",
    "ifeval",
    "arc_challenge",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("base", "adapter"), required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-gen-toks", type=int, default=256)
    ap.add_argument("--adapter-model", default=ADAPTER)
    ap.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated lm-eval task/group names")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    selected_tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    raw_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=quant,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    if args.variant == "adapter":
        from peft import PeftModel
        raw_model = PeftModel.from_pretrained(raw_model, args.adapter_model)
    raw_model.eval()
    model = HFLM(
        pretrained=raw_model,
        tokenizer=tokenizer,
        device="cuda:0",
        batch_size=1,
        max_batch_size=1,
        trust_remote_code=True,
    )
    started = time.time()
    results = simple_evaluate(
        model=model,
        tasks=selected_tasks,
        batch_size=1,
        max_batch_size=1,
        limit=args.limit,
        bootstrap_iters=0,
        cache_requests=True,
        log_samples=True,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        gen_kwargs={"max_gen_toks": args.max_gen_toks, "do_sample": False},
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )
    results["local_run"] = {
        "variant": args.variant,
        "adapter_model": args.adapter_model if args.variant == "adapter" else None,
        "tasks": selected_tasks,
        "limit_per_task": args.limit,
        "max_gen_toks": args.max_gen_toks,
        "wall_seconds": round(time.time() - started, 3),
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=handle_non_serializable),
        encoding="utf-8",
    )
    print(f"Saved {args.output.resolve()}", flush=True)

if __name__ == "__main__":
    main()
