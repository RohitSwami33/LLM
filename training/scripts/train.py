#!/usr/bin/env python3
"""Main training script for the decoder-only Transformer.

Usage:
    python training/scripts/train.py --config training/configs/base.yaml
    python training/scripts/train.py --config training/configs/base.yaml --resume training/checkpoints/step_5000
    python training/scripts/train.py --config training/configs/pretrain_small.yaml --auto-resume
"""

import sys
import os
import re
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest step_N checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    pattern = re.compile(r"step_(\d+)$")
    best_step = -1
    best_path = None
    for name in os.listdir(checkpoint_dir):
        match = pattern.search(name)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(checkpoint_dir, name)
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train decoder-only Transformer")
    parser.add_argument("--config", type=str, default="training/configs/base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-resume from latest checkpoint in checkpoint_dir")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training steps (overrides config)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.resume:
        config["training"]["resume_from"] = args.resume
    elif args.auto_resume:
        checkpoint_dir = config["training"].get("checkpoint_dir", "training/checkpoints")
        # Look for checkpoints in experiment dir if it exists
        exp_dir = config.get("experiment_dir", None)
        if exp_dir and os.path.exists(exp_dir):
            # Find most recent experiment
            experiments = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))])
            if experiments:
                latest_exp = os.path.join(exp_dir, experiments[-1])
                ckpt_dir = os.path.join(latest_exp, "checkpoints")
                if os.path.exists(ckpt_dir):
                    checkpoint_dir = ckpt_dir
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            config["training"]["resume_from"] = latest
            print(f"Auto-resuming from: {latest}")
        else:
            print("No checkpoint found, starting from scratch")
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Train
    from training.trainer import Trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
