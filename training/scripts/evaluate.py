#!/usr/bin/env python3
"""Evaluate a trained model.

Usage:
    python training/scripts/evaluate.py --checkpoint training/checkpoints/best_model --config training/configs/base.yaml
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="Evaluate transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="training/configs/base.yaml", help="Config path")
    parser.add_argument("--max-steps", type=int, default=100, help="Max eval steps")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    from training.model import TransformerLM, ModelConfig
    from training.tokenizer import build_tokenizer
    from training.data import JsonlDataset, DataCollator, build_dataloader
    from training.utils.checkpoint import load_checkpoint
    from training.evaluation import evaluate, generate_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model_config = ModelConfig.from_dict(config.get("model", {}))
    tokenizer = build_tokenizer(config.get("tokenizer", {}), data_path=config["dataset"]["path"])
    model_config.vocab_size = tokenizer.vocab_size
    model = TransformerLM(model_config).to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build val dataloader
    collator = DataCollator(tokenizer.pad_token_id, model_config.max_seq_len)
    dataset = JsonlDataset(config["dataset"]["path"], tokenizer, model_config.max_seq_len)
    n = len(dataset)
    val_size = int(n * config["dataset"].get("val_split", 0.1))
    train_size = n - val_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = build_dataloader(val_dataset, config["training"]["batch_size"], collator, shuffle=False)

    # Evaluate
    results = evaluate(model, val_loader, device, max_steps=args.max_steps,
                       pad_token_id=tokenizer.pad_token_id)
    print(f"\nValidation Loss: {results['val_loss']:.4f}")
    print(f"Perplexity:      {results['val_perplexity']:.2f}")
    print(f"Accuracy:        {results['val_accuracy']:.4f}")

    # Generate samples
    prompts = config.get("evaluation", {}).get("prompts", [
        "Explain quantum computing",
        "Write a Python function to sort a list",
        "What are the benefits of exercise?",
    ])
    print("\n--- Sample Generations ---")
    samples = generate_samples(model, tokenizer, prompts, device=device, max_new_tokens=200)
    for prompt, sample in zip(prompts, samples):
        print(f"\nPrompt: {prompt}")
        print(f"Output: {sample}")


if __name__ == "__main__":
    main()
