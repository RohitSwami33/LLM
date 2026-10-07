"""Memory-conscious 4-bit QLoRA SFT for Qwen3.5-4B on a 12 GB GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: Path, limit: int) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("work/training_v2/data/science_sft.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("outputs/qwen35-science-qlora-v2"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--replay-cache", type=Path, default=None)
    ap.add_argument("--replay-limit", type=int, default=0)
    ap.add_argument("--init-adapter", type=Path, default=None)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--resume", type=Path, default=None, help="Trainer checkpoint to resume")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    base = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw = read_jsonl(args.data, args.limit)

    def token_ids(value):
        """Normalize chat-template output across Transformers 4.x/5.x."""
        if hasattr(value, "input_ids"):
            value = value.input_ids
        elif isinstance(value, dict):
            value = value["input_ids"]
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return list(value)

    def encode(x: dict) -> dict:
        messages = x["messages"]
        prompt = messages[:-1]
        prompt_ids = token_ids(tok.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        ))
        full_ids = token_ids(tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        ))
        full_ids = full_ids[: args.max_length]
        labels = list(full_ids)
        # Correct token-based prompt mask. The old run used character counts here.
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": full_ids, "labels": labels, "source": x["source"]}

    encoded = [encode(x) for x in raw]
    encoded = [x for x in encoded if any(y != -100 for y in x["labels"])]
    new_examples = len(encoded)
    replay_examples = 0
    if args.replay_cache:
        replay = load_from_disk(str(args.replay_cache)).shuffle(seed=3407)
        if args.replay_limit:
            replay = replay.select(range(min(args.replay_limit, len(replay))))
        for x in replay:
            ids = list(x["input_ids"])[: args.max_length]
            labels = list(x["labels"])[: args.max_length]
            if ids and any(y != -100 for y in labels):
                encoded.append({"input_ids": ids, "labels": labels, "source": "old_fc_replay"})
                replay_examples += 1
    if not encoded:
        raise SystemExit("No trainable examples after tokenization")
    train = Dataset.from_list(encoded)

    class Collator:
        def __call__(self, features):
            max_len = max(len(x["input_ids"]) for x in features)
            ids, labels, masks = [], [], []
            for x in features:
                n = max_len - len(x["input_ids"])
                ids.append(x["input_ids"] + [tok.pad_token_id] * n)
                labels.append(x["labels"] + [-100] * n)
                masks.append([1] * len(x["input_ids"]) + [0] * n)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=qconfig,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    ta = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0 if args.max_steps == 1 else 50,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1 if args.max_steps > 0 else 10,
        save_steps=250,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=3407,
    )
    trainer = Trainer(model=model, args=ta, train_dataset=train, data_collator=Collator())
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "final_adapter"))
    tok.save_pretrained(str(args.output / "final_adapter"))
    metrics = dict(result.metrics)
    metrics.update({"examples": len(train), "new_examples": new_examples,
                    "replay_examples": replay_examples,
                    "initial_adapter": str(args.init_adapter) if args.init_adapter else None,
                    "base": base, "quantization": "NF4 4-bit",
                    "max_length": args.max_length, "learning_rate": args.learning_rate})
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
