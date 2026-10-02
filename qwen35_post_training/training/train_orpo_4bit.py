"""Reference-free 4-bit ORPO continuation for Qwen3.5-4B PEFT adapters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.orpo import ORPOConfig, ORPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init-adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--learning-rate", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--resume", type=Path, default=None, help="Trainer checkpoint to resume")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    rows = []
    with args.data.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    train = Dataset.from_list(rows).shuffle(seed=7301)

    base = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    config = ORPOConfig(
        output_dir=str(args.output),
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0 if args.max_steps == 1 else 40,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=1 if args.max_steps > 0 else 10,
        save_steps=250,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=7301,
        max_length=args.max_length,
        max_completion_length=256,
        beta=args.beta,
        dataset_num_proc=1,
    )
    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=train,
        processing_class=tok,
    )
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final = args.output / "final_adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    metrics = dict(result.metrics)
    metrics.update({
        "examples": len(train),
        "base": base,
        "initial_adapter": str(args.init_adapter),
        "quantization": "NF4 4-bit",
        "method": "ORPO",
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
    })
    (args.output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
