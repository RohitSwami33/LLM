#!/usr/bin/env python3
"""Kaggle SFT: allenai/OLMoE-1B-7B-0924 + distilled 1,252 docs (QLoRA).

- 1B active / 7B total MoE, Apache-2.0, 4096 ctx, HF OlmoeForCausalLM
- QLoRA: 4-bit NF4 + LoRA r=64 alpha=128 on all linear (q/k/v/o + MoE gate/up/down)
- Data: datasets/sft/{train,val}.jsonl built by tools/build_sft.py (prompt/response)
- Kaggle: mounts tomiokasan/olmoe-sft-code (flat) or local fallback; internet ON to pull
  allenai/OLMoE-1B-7B-0924 if not cached at /kaggle/input/olmoe-1b-7b-0924.
- Checkpoints: LoRA adapter every 50 steps -> /kaggle/working/sft_checkpoints
  (also queued to tomiokasan/olmoe-sft-checkpoints if Kaggle auth available).

Local dry-run (CPU, no GPU):
  PP_SMOKE=1 python training/kaggle/kaggle_sft_olmoe.py --max-steps 5
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _bootstrap_deps():
    need = []
    for pkg, imp in [("trl","trl"),("peft","peft"),("datasets","datasets"),("accelerate","accelerate")]:
        try:
            __import__(imp)
        except ImportError:
            need.append(pkg)
    if need:
        import subprocess
        print(f"  [deps] installing {need} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + need +
                              ["transformers>=4.51", "peft", "trl", "datasets", "accelerate"])
        print("  [deps] installed")
_bootstrap_deps()

CODE_DATASET = os.environ.get("PP_CODE_DATASET", "olmoe-sft-code")

def _ensure_code_package():
    for root in ("/kaggle/input", "/kaggle/input/datasets"):
        base = Path(root)
        if not base.exists():
            continue
        for p in base.glob(f"*/{CODE_DATASET}"):
            if (p / "train.jsonl").exists() or (p / "sft.jsonl").exists():
                # SFT data is staged flat; nothing to assemble
                sys.path.insert(0, "/kaggle/working")
                print(f"  [code] SFT data at {p}")
                return p
    return None

def _find_sft_data():
    # Kaggle mount first, then local
    cand = []
    if os.path.exists("/kaggle/input"):
        cand += sorted(Path("/kaggle/input").rglob("train.jsonl"))
    cand += [Path("datasets/sft/train.jsonl"), Path("datasets/sft/sft.jsonl")]
    for p in cand:
        if p.exists() and p.stat().st_size > 0:
            # prefer train.jsonl sibling val
            val = p.parent / "val.jsonl"
            return p, (val if val.exists() else None)
    return None, None

def _load_model_tokenizer(base_id, use_4bit, smoke):
    # Kaggle image torchao 0.10 breaks peft; upgrade silently (30s) before any peft import
    if not smoke and not os.environ.get("PP_NO_TORCHAO_UPGRADE"):
        try:
            import subprocess as _sp
            _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16"], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            print("  torchao upgraded for peft")
        except Exception as e:
            print(f"  torchao upgrade skipped: {e}")
            # fallback: neuter the version gate
            try:
                import peft.import_utils as _piu2
                _piu2.is_torchao_available = lambda *a, **k: False
                import peft.tuners.lora.torchao as _pt
                _pt.is_torchao_available = lambda *a, **k: False
            except Exception:
                pass
    from transformers import AutoTokenizer, OlmoeForCausalLM, BitsAndBytesConfig
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    attn_impl = None
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    print(f"  Loading {base_id} (4bit={use_4bit}, smoke={smoke}, gpus={n_gpus}, attn={attn_impl}) ...")
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if smoke:
        # Tiny fallback for local CPU smoke (OLMoE 7B is 14GB, won't fit Mac)
        smoke_id = os.environ.get("PP_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M")
        try:
            from transformers import AutoModelForCausalLM
            tok_sm = AutoTokenizer.from_pretrained(smoke_id, trust_remote_code=True)
            if tok_sm.pad_token is None:
                tok_sm.pad_token = tok_sm.eos_token
            tok_sm.padding_side = "right"
            model_sm = AutoModelForCausalLM.from_pretrained(smoke_id, trust_remote_code=True, torch_dtype=torch.float32)
            print(f"  [smoke] using {smoke_id} instead of {base_id}")
            return model_sm, tok_sm
        except Exception as e:
            print(f"  [smoke] fallback failed ({e}), trying base {base_id}")
            model = OlmoeForCausalLM.from_pretrained(base_id, trust_remote_code=True, torch_dtype=torch.float32)
            return model, tok

    # 2×T4 path: shard model across both GPUs via balanced device_map (14 GB BF16 splits to ~7 GB/GPU)
    # DataParallel would duplicate weights and OOM, so we use model sharding; Trainer handles
    # data parallelism via per_device_batch on each shard's micro-batch.
    device_map = "balanced" if n_gpus >= 2 else "auto"
    common = dict(trust_remote_code=True, attn_implementation=attn_impl) if attn_impl else dict(trust_remote_code=True)
    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = OlmoeForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map=device_map, **common)
    else:
        model = OlmoeForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map=device_map, **common)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full run (3 epochs)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-batch", type=int, default=0, help="0=auto (4 on 2xT4, 2 on 1xT4)")
    ap.add_argument("--grad-accum", type=int, default=0, help="0=auto (2 on 2xT4, 4 on 1xT4)")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit quant (use bf16, faster on 2xT4)")
    ap.add_argument("--force-4bit", action="store_true", help="force 4-bit even on 2xT4 (slower)")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    smoke = bool(os.environ.get("PP_SMOKE"))
    import torch as _torch
    _n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    # CUDA speedups (TF32 + autotune) — safe on sm75 T4, no numerics shift
    if _n_gpus > 0:
        try:
            _torch.set_float32_matmul_precision("high")
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
            _torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    # T4x2 BF16 7B sharded still OOM at batch 4×2048 (12.7/14.5GB on SDPA v6) — must be conservative
    if args.per_device_batch == 0:
        args.per_device_batch = 1 if _n_gpus >= 2 else 2
    if args.grad_accum == 0:
        args.grad_accum = 8 if _n_gpus >= 2 else 4
    # Cap seq for T4x2; 2048×4 OOMs SDPA, 1024 is safe and still covers most docs
    if args.max_seq_len == 2048 and _n_gpus >= 2 and not smoke:
        args.max_seq_len = 1024
    if smoke and args.max_steps == 0:
        args.max_steps = 5
        args.per_device_batch = 1
        args.grad_accum = 1
        args.max_seq_len = 512

    is_kaggle = os.path.exists("/kaggle/working")
    if is_kaggle:
        _ensure_code_package()

    train_path, val_path = _find_sft_data()
    if train_path is None:
        raise FileNotFoundError("No SFT train.jsonl found (expected datasets/sft/train.jsonl or Kaggle mount)")
    # On 2×T4, BF16 sharded is faster than NF4 (T4 has no int4 kernel); only use 4-bit when forced or on 1 GPU
    if not smoke and not args.force_4bit and _n_gpus >= 2:
        # Default to BF16 on 2 GPUs unless caller asked for 4-bit
        use_4bit = False if not args.no_4bit else False  # no_4bit already false means BF16
        # args.no_4bit is not set by default; we want BF16 on 2xT4
        args.no_4bit = True
    use_4bit_flag = not args.no_4bit and not smoke
    if args.force_4bit and not smoke:
        use_4bit_flag = True
        args.no_4bit = False
    print(f"  Train: {train_path}  Val: {val_path}")
    print(f"  Base: {args.base}  smoke={smoke} 4bit={use_4bit_flag} gpus={_n_gpus} batch={args.per_device_batch} accum={args.grad_accum}")

    # Optional HF login for private push
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("  HF login OK")
        except Exception as e:
            print(f"  HF login failed: {e}")

    model, tok = _load_model_tokenizer(args.base, use_4bit=use_4bit_flag, smoke=smoke)

    # Prepare dataset (datasets + formatting)
    from datasets import load_dataset
    # load as json
    ds_train = load_dataset("json", data_files=str(train_path), split="train")
    ds_val = None
    if val_path and val_path.exists():
        try:
            ds_val = load_dataset("json", data_files=str(val_path), split="train")
        except Exception as e:
            print(f"  val load failed: {e}")

    # Quick stats
    avg_len = sum(len(x["response"]) for x in ds_train) / len(ds_train)
    print(f"  Dataset: {len(ds_train)} train, {len(ds_val) if ds_val else 0} val, avg response {avg_len:.0f} chars (~{avg_len/4:.0f} tok)")

    # Pre-format to `text` for SFTTrainer (trl 0.15+: dataset_text_field="text", no formatting_func)
    def _to_text(ex):
        return {"text": f"### Instruction:\n{ex['prompt']}\n\n### Response:\n{ex['response']}{tok.eos_token}"}
    ds_train = ds_train.map(_to_text, remove_columns=ds_train.column_names)
    if ds_val is not None:
        ds_val = ds_val.map(_to_text, remove_columns=ds_val.column_names)

    # LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # OLMoE linear names: q_proj/k_proj/v_proj/o_proj + moe gate/up/down + shared expert
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","w1","w2","w3","gate"]
    # TRL will filter to existing modules; include both naming conventions
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Prepare for k-bit
    if not smoke and not args.no_4bit:
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            print(f"  prepare k-bit failed: {e}")
    if hasattr(model, "config"):
        model.config.use_cache = False

    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # SFT Trainer — workaround: trl 0.24 _patch_chunked_ce_lm_head crashes on
    # OLMoE's lm_head (functools.partial) -> AttributeError __func__
    import trl.trainer.sft_trainer as _sft
    _orig_patch = getattr(_sft, "_patch_chunked_ce_lm_head", None)
    if _orig_patch is not None:
        def _safe_patch(*a, **k):
            try:
                return _orig_patch(*a, **k)
            except AttributeError as e:
                if "__func__" in str(e):
                    print(f"  [trl] _patch_chunked_ce_lm_head skipped: {e}")
                    return
                raise
        _sft._patch_chunked_ce_lm_head = _safe_patch
    from trl import SFTTrainer, SFTConfig

    max_steps = args.max_steps if args.max_steps > 0 else -1
    # epochs -> steps if max_steps not set: let TRL handle num_train_epochs
    num_epochs = args.epochs if max_steps == -1 else 1

    output_dir = "/kaggle/working/sft_checkpoints" if is_kaggle else "sft_checkpoints"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=max(1, args.per_device_batch // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.03 * (max_steps if max_steps > 0 else 500)) if max_steps != -1 else 15,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        max_length=args.max_seq_len,
        packing=False,
        bf16=(not smoke and args.bf16),
        fp16=False,
        gradient_checkpointing=True,
        loss_type="nll",  # chunked_nll needs patched lm_head which OLMoE lacks -> num_valid_tokens crash
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps" if ds_val is not None else "no",
        eval_steps=50 if ds_val is not None else 500,
        save_total_limit=3,
        report_to="none",
        dataset_text_field="text",
        seed=1337,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
    )

    print(f"\n  Training: epochs={num_epochs} max_steps={max_steps} batch={args.per_device_batch} accum={args.grad_accum} seq={args.max_seq_len}")
    trainer.train()

    # Save LoRA adapter
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    # Merge for eval/push preview (only if not smoke-quantized path that breaks merge)
    if not smoke:
        try:
            merged = trainer.model.merge_and_unload()
            merged_dir = "/kaggle/working/merged" if is_kaggle else "merged_olmoe"
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(merged_dir)
            print(f"  Merged model at {merged_dir} ({sum(p.numel() for p in merged.parameters())/1e9:.2f}B params)")
        except Exception as e:
            print(f"  Merge skipped: {type(e).__name__}: {e}")

    # Optional: push to hub if HF_TOKEN and repo set
    repo = os.environ.get("HF_REPO")  # e.g. tomiokasan/OLMoE-1B-7B-Distilled
    if hf_token and repo and not smoke:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=output_dir, repo_id=repo, commit_message="QLoRA SFT on distilled 1,252 docs")
            print(f"  Pushed adapter to https://huggingface.co/{repo}")
        except Exception as e:
            print(f"  HF push failed: {type(e).__name__}: {e}")

    print("\n  SFT complete.")

if __name__ == "__main__":
    main()
