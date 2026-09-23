#!/usr/bin/env python3
"""Kernel B (roronoazoro3008/qwen35-fc-train): QLoRA fine-tune Qwen3.5-4B (post-trained)
for web-search decision, fact-checking, evidence grounding, anti-hallucination.

Pipeline:
  STEP 0  inspect environment (GPU, VRAM, disk, /kaggle/input model datasets)
  STEP 1  resolve model source: prefer user-provided 4-bit dataset if bnb-compatible;
          GGUF/AWQ/GPTQ are inference-only -> fall back to BF16 HF checkpoint + NF4 loading
  STEP 2  baseline eval on held-out suites (math/decision/factcheck/hallucination/QA)
  STEP 3  QLoRA training on tokenized cache (kernel_source: qwen35-fc-data)
          r=32 alpha=64 dropout=0.05, lr 5e-5, 1 epoch, eff batch 16, seq 3072
  STEP 4  post-train eval on the same suites -> eval_results.json (base vs ft)
  STEP 5  save adapter + tokenizer + configs + README

No benchmark test data is trained on. Eval suites are held out by kernel A.
"""
import json, os, re, sys, time, glob, math
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

def log(m): print(f"\n[m]{m}", flush=True)

BASE = "Qwen/Qwen3.5-4B"           # post-trained instruct (per spec, NOT -Base)
SEED = 1337
MAX_LEN = 3072
LR = 5e-5                           # spec range 1e-5..5e-5; prefer smaller for retention
EPOCHS = 1
R, ALPHA, DROPOUT = 32, 64, 0.05
EFF_BATCH = 16                      # per_device 1 x grad_accum 16
OUT = Path("/kaggle/working")
ADAPTER_DIR = OUT / "fc_adapter"

WEB_SEARCH_SCHEMA = [{
    "type": "function",
    "function": {"name": "web_search",
                 "description": "Web search returning ranked results.",
                 "parameters": {"type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"]}}}]

# ---------------------------------------------------------------- STEP 0: env
def inspect_env():
    import torch
    log(f"torch {torch.__version__} cuda {torch.version.cuda}")
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        log(f"  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB sm{p.major}{p.minor}")
    st = os.statvfs("/kaggle/working")
    log(f"disk free: {st.f_bavail*st.f_frsize/1e9:.1f}GB")
    # what's mounted?
    for p in sorted(glob.glob("/kaggle/input/*")):
        try:
            entries = os.listdir(p)[:6]
            log(f"  input: {os.path.basename(p)} -> {entries}")
        except Exception:
            pass
    return n

def find_local_model():
    """User said a 4-bit model may be present. Classify it per spec."""
    cands = []
    for p in glob.glob("/kaggle/input/*"):
        files = []
        for root, _, fs in os.walk(p):
            for f in fs:
                files.append(os.path.join(root, f))
        gguf = [f for f in files if f.endswith(".gguf")]
        saf = [f for f in files if f.endswith(".safetensors")]
        has_cfg = any(f.endswith("config.json") for f in files)
        if gguf:
            cands.append(("gguf", p, gguf[0]))
        elif saf and has_cfg:
            cands.append(("safetensors_4bit?", p, saf[0]))
    return cands

# ---------------------------------------------------------------- model load
def load_model_and_tok():
    import torch
    from transformers import AutoTokenizer, AutoConfig
    import importlib
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_ok = True
    try:
        import bitsandbytes  # noqa
    except Exception:
        bnb_ok = False
    log(f"bitsandbytes available: {bnb_ok}")

    # find correct model class for qwen3_5 (VLM-capable; text-only training)
    cls = None
    try:
        cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
        log(f"model_type={cfg.model_type} architectures={getattr(cfg,'architectures',None)}")
    except Exception as e:
        log(f"config load failed: {e}; installing latest transformers may be required")

    kwargs = dict(trust_remote_code=True, device_map="auto",
                  attn_implementation="sdpa")
    quant = None
    if bnb_ok:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,  # T4: fp16 compute
                                   bnb_4bit_use_double_quant=True)
        kwargs["quantization_config"] = quant
    else:
        kwargs["torch_dtype"] = torch.float16

    last_err = None
    for loader in ("causal", "image_text"):
        try:
            if loader == "causal":
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(BASE, **kwargs)
            else:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(BASE, **kwargs)
            log(f"loaded via {loader}")
            return model, tok
        except Exception as e:
            last_err = e
            log(f"loader {loader} failed: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"could not load {BASE}: {last_err}")

# ---------------------------------------------------------------- LoRA targets
def pick_target_modules(model):
    """Inspect architecture programmatically; never hardcode blindly."""
    from collections import Counter
    suffix_counter = Counter()
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("Linear4bit", "Linear") :
            suffix = name.split(".")[-1]
            if suffix in ("lm_head",): continue
            suffix_counter[suffix] += 1
    log(f"linear suffixes: {dict(suffix_counter)}")
    keep = []
    prefer = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "qkv_proj", "out_proj", "in_proj_qkvz", "in_proj_ba", "in_proj", "gate",
              "beta", "decay", "x_proj", "dt_proj"]
    for s, c in suffix_counter.items():
        if s in prefer and c >= 4:
            keep.append(s)
    if not keep:  # fallback: top-6 frequent suffixes
        keep = [s for s, _ in suffix_counter.most_common(6) if s != "lm_head"]
    log(f"target_modules = {keep}")
    return keep

# ---------------------------------------------------------------- generation eval
VERDICTS = ["SUPPORTED", "LIKELY SUPPORTED", "REFUTED", "CONTESTED",
            "INSUFFICIENT EVIDENCE", "UNVERIFIED"]

def load_suite(path):
    p = Path(path)
    if not p.exists(): return []
    return [json.loads(l) for l in open(p) if l.strip()]

def gen(model, tok, messages, max_new=220):
    prompt = tok.apply_chat_template(messages, tools=WEB_SEARCH_SCHEMA,
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    # strip special tokens for parsing
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

def run_suite(model, tok, name, suite, limit=60):
    """Returns metrics dict per suite."""
    import torch
    if not suite:
        return None
    m = {"n": 0, "verdict_correct": 0, "refused_ok": 0, "hallucinated": 0,
         "search_calls": 0, "direct_answers": 0, "math_em": 0}
    for ex in suite[:limit]:
        if "math" in name and "question" in ex:
            msgs = [{"role": "user", "content": ex["question"]}]
        else:
            msgs = ex["messages"]
        try:
            out = gen(model, tok, msgs)
        except Exception:
            continue
        m["n"] += 1
        low = out.lower()
        if "<tool_call>" in out or '"name": "web_search"' in out:
            m["search_calls"] += 1
        else:
            m["direct_answers"] += 1
        if "math" in name and "answer" in ex:
            pred = re.findall(r"-?\d[\d,]*", out.replace(",", ""))
            gold = re.findall(r"-?\d[\d,]*", str(ex["answer"]).replace(",", ""))
            if pred and gold and pred[-1] == gold[-1]:
                m["math_em"] += 1
        if any(v.lower() in low for v in VERDICTS):
            # verdict present: correct if matches expected label in the example tail
            gold = next((v for v in VERDICTS if v.lower() in json.dumps(ex).lower()), None)
            if gold and gold.lower() in low:
                m["verdict_correct"] += 1
        if any(x in low for x in ["could not verify", "insufficient evidence", "cannot verify",
                                  "no reliable source", "not enough evidence", "unverified"]):
            m["refused_ok"] += 1
        # hallucination heuristic: assertive verdict with no evidence & no refusal (fabricated-premise suites)
        if "hallucination" in name:
            has_verdict = any(v.lower() in low for v in VERDICTS[:2])  # SUPPORTED/LIKELY
            if has_verdict and not any(x in low for x in ["could not verify", "insufficient"]):
                m["hallucinated"] += 1
    n = max(1, m["n"])
    m["verdict_acc"] = round(m["verdict_correct"] / n, 3)
    m["refusal_rate"] = round(m["refused_ok"] / n, 3)
    m["hallucination_rate"] = round(m["hallucinated"] / n, 3)
    m["search_rate"] = round(m["search_calls"] / n, 3)
    if "math" in name:
        m["math_em_acc"] = round(m["math_em"] / n, 3)
    return m

# ---------------------------------------------------------------- main
def main():
    import torch, random
    random.seed(SEED); torch.manual_seed(SEED)

    log("STEP 0: environment")
    inspect_env()
    cands = find_local_model()
    if cands:
        log(f"local model candidates: {cands}")
        kinds = {c[0] for c in cands}
        if kinds == {"gguf"}:
            log("GGUF found: inference-only, NOT trainable. Using BF16 HF checkpoint with NF4 loading (per spec).")

    log("STEP 1: load model (4-bit NF4)")
    model, tok = load_model_and_tok()

    # tokenized cache from kernel A (mounted via kernel_sources)
    cache_dir = None
    for p in glob.glob("/kaggle/input/*/tokenized_cache") + glob.glob("/kaggle/input/*/*/tokenized_cache"):
        cache_dir = p; break
    if cache_dir is None:
        raise FileNotFoundError("tokenized_cache not mounted — has kernel A (qwen35-fc-data) completed?")
    log(f"cache: {cache_dir}")

    from datasets import load_from_disk
    ds = load_from_disk(cache_dir)
    log(f"tokenized examples: {len(ds)}")

    # ---------------- STEP 2: baseline eval ----------------
    log("STEP 2: BASELINE eval (pre-training)")
    suites = {name: load_suite(f"/kaggle/input/*/eval/{name}.jsonl")
              for name in ["math_gsm8k_test", "decision", "factcheck", "hallucination", "holdout_mixed"]}
    suites = {k: v[0] if v else [] for k, v in suites.items()}
    baseline = {}
    for name, suite in suites.items():
        if suite:
            baseline[name] = run_suite(model, tok, name, suite)
            log(f"  baseline {name}: {json.dumps(baseline[name])}")

    # ---------------- STEP 3: QLoRA ----------------
    log("STEP 3: QLoRA training")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.enable_input_require_grads()
    targets = pick_target_modules(model)
    lcfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    split = ds.train_test_split(test_size=0.01, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

    targs = TrainingArguments(
        output_dir=str(OUT / "ckpt"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=EFF_BATCH,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=EPOCHS,
        bf16=False, fp16=True,          # T4: fp16 compute + GradScaler
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        save_only_model=True,
        eval_strategy="steps",
        eval_steps=800,
        report_to="none",
        seed=SEED,
        optim="paged_adamw_8bit" if _try_import_paged() else "adamw_torch",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    t0 = time.time()
    trainer.train()
    log(f"training wall time: {(time.time()-t0)/3600:.2f}h")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    log(f"adapter saved: {ADAPTER_DIR}")

    # ---------------- STEP 4: post-train eval ----------------
    log("STEP 4: FINE-TUNED eval")
    model.eval()
    finetuned = {}
    for name, suite in suites.items():
        if suite:
            finetuned[name] = run_suite(model, tok, name, suite)
            log(f"  finetuned {name}: {json.dumps(finetuned[name])}")

    results = {"base_model": BASE, "quantization": "NF4 4-bit (bitsandbytes), fp16 compute",
               "lora": {"r": R, "alpha": ALPHA, "dropout": DROPOUT, "targets": targets},
               "hyperparams": {"lr": LR, "epochs": EPOCHS, "eff_batch": EFF_BATCH,
                               "max_len": MAX_LEN, "warmup_ratio": 0.03, "wd": 0.01},
               "baseline": baseline, "finetuned": finetuned}
    with open(OUT / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # regression guard report
    def delta(metric, suite):
        b = (baseline.get(suite) or {}).get(metric)
        f = (finetuned.get(suite) or {}).get(metric)
        return None if (b is None or f is None) else round(100*(f-b)/max(b,1e-9), 1)
    regress = {"math_EM_change_pct": delta("math_em_acc", "math_gsm8k_test"),
               "verdict_acc_change_pct": delta("verdict_acc", "factcheck"),
               "hallucination_rate_change": (finetuned.get("hallucination") or {}).get("hallucination_rate"),
               "baseline_hallucination_rate": (baseline.get("hallucination") or {}).get("hallucination_rate")}
    with open(OUT / "regression_report.json", "w") as f:
        json.dump(regress, f, indent=2)
    log("regression: " + json.dumps(regress))

    # ---------------- STEP 5: README ----------------
    readme = f"""# Qwen3.5-4B Fact-Checking / Web-Search QLoRA

Base: {BASE} (post-trained instruct), NF4 4-bit (bnb), fp16 compute on T4
LoRA: r={R} alpha={ALPHA} dropout={DROPOUT} targets={targets}
HP: lr={LR} cosine, {EPOCHS} epoch, eff batch {EFF_BATCH}, seq {MAX_LEN}, wd 0.01, warmup 3%

Data: tokenized cache from roronoazoro3008/qwen35-fc-data (assistant-span loss mask;
user + tool-result text masked). Tool: web_search(query) in Qwen native format.

Eval (held-out): see eval_results.json. Key: hallucination rate, verdict accuracy,
math EM regression guard, search-rate balance.

Known limitations: synthetic anti-hallucination examples use clearly-simulated tool
outputs; real-world search behavior depends on the runtime tool implementation.
"""
    (OUT / "README.md").write_text(readme)
    log("STEP 5 complete — outputs in /kaggle/working")

def _try_import_paged():
    try:
        import bitsandbytes  # noqa
        return True
    except Exception:
        return False

if __name__ == "__main__":
    main()
