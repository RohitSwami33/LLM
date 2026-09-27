#!/usr/bin/env python3
"""
RTX 4070 Super (12GB) - Qwen3.5-4B HF benchmark
Uses NF4 4-bit (bitsandbytes) optimal for Ada Lovelace sm89
Base: Qwen/Qwen3.5-4B vs Adapter: RohitSwami33/qwen35-fc-adapter
"""
import os, json, time, glob
from pathlib import Path

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"]="false"

BASE="Qwen/Qwen3.5-4B"
ADAPTER="RohitSwami33/qwen35-fc-adapter"  # or ./model_weights/fc_adapter if downloaded
PROMPTS_DIR=Path(__file__).parent / "prompts"
OUT=Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load_model_tok(use_adapter=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    log(f"Loading {BASE} {'+ adapter' if use_adapter else ''} NF4 on 4070 Super...")
    model=AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    if use_adapter:
        from peft import PeftModel
        # Try HF hub first, fallback to local
        try:
            model=PeftModel.from_pretrained(model, ADAPTER, token=os.environ.get("HF_TOKEN"))
        except:
            local=Path(__file__).parent / "model_weights" / "fc_adapter"
            if local.exists():
                model=PeftModel.from_pretrained(model, str(local))
            else:
                raise
        log("Adapter loaded")
    model.eval()
    return model, tok

def run_prompt(model, tok, prompt_path):
    text=open(prompt_path).read()
    # Extract prompt between ``` blocks or after **Prompt:**
    import re
    m=re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    prompt=m.group(1).strip() if m else text
    # Remove markdown headers
    prompt=re.sub(r"^#.*\n", "", prompt).strip()
    ids=tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with __import__("torch").no_grad():
        out=model.generate(**ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
    decoded=tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prompt, decoded.strip()

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", action="store_true", help="use finetuned adapter")
    ap.add_argument("--limit", type=int, default=0, help="limit prompts (0=all)")
    args=ap.parse_args()
    model, tok = load_model_tok(use_adapter=args.adapter)
    prompts=sorted(PROMPTS_DIR.glob("*.md"))
    if args.limit: prompts=prompts[:args.limit]
    log(f"Benchmark {len(prompts)} prompts on {'finetuned' if args.adapter else 'base'}")
    results={}
    for p in prompts:
        log(f"Prompt {p.name}")
        prompt, response = run_prompt(model, tok, p)
        log(f"Response: {response[:200]}")
        results[p.name]={"prompt":prompt, "response":response}
        time.sleep(1)
    out_file=OUT / f"results_{'ft' if args.adapter else 'base'}.json"
    json.dump(results, open(out_file,"w"), indent=2)
    log(f"Saved {out_file}")
