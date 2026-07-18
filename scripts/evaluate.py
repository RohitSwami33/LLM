#!/usr/bin/env python3
"""CLI script to evaluate a model checkpoint on all benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint experiments/2026-07-19_002/checkpoints/best_model
    python scripts/evaluate.py --checkpoint path/to/ckpt --benchmarks wikitext2 hellaswag arc_easy
    python scripts/evaluate.py --checkpoint path/to/ckpt --output-dir results/baseline --max-samples 500
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None,
                        help="Benchmarks to run (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Max tokens for generation")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip text generation")
    parser.add_argument("--no-efficiency", action="store_true",
                        help="Skip efficiency benchmark")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Custom prompts for generation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (mps, cuda, cpu)")
    args = parser.parse_args()

    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from evaluation.runner import EvalRunner

    runner = EvalRunner.from_checkpoint(args.checkpoint, args.config, device)

    results = runner.evaluate(
        benchmarks=args.benchmarks,
        generate=not args.no_generate,
        efficiency=not args.no_efficiency,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )

    print(f"\n{'='*60}")
    print(results.summary_table())
    print(f"{'='*60}")

    if args.output_dir:
        runner.save_results(results, args.output_dir)
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        exp_dir = os.path.dirname(ckpt_dir)
        output_dir = os.path.join(exp_dir, "evaluation")
        runner.save_results(results, output_dir)


if __name__ == "__main__":
    main()
