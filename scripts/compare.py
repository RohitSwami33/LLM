#!/usr/bin/env python3
"""CLI script to compare multiple model checkpoints.

Usage:
    python scripts/compare.py --checkpoints ckpt1 ckpt2 ckpt3
    python scripts/compare.py --checkpoints experiments/*/checkpoints/best_model --output-dir results/compare
    python scripts/compare.py --checkpoints ckpt1 ckpt2 --benchmarks wikitext2 hellaswag
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Compare multiple model checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to checkpoint files")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None,
                        help="Benchmarks to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/compare",
                        help="Directory to save comparison results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per benchmark")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip text generation")
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

    from evaluation.compare import compare_checkpoints

    all_results = compare_checkpoints(
        args.checkpoints,
        benchmark_names=args.benchmarks,
        config_path=args.config,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=device,
        generate=not args.no_generate,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    for path, results in all_results.items():
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n--- {name} ---")
        for bench_name, bench in results.benchmarks.items():
            for metric, value in bench.metrics.items():
                if isinstance(value, float):
                    print(f"  {bench_name}/{metric}: {value:.4f}")
                else:
                    print(f"  {bench_name}/{metric}: {value}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
