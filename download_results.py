#!/usr/bin/env python3
"""download_results.py — Download checkpoints and results from Kaggle.

Usage:
    python download_results.py                              # Download all
    python download_results.py --checkpoints               # Checkpoints only
    python download_results.py --kernel my-kernel          # Specific kernel
    python download_results.py --resume                    # Download + auto-resume
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kaggle.auth import validate_kaggle_setup
from kaggle.download import KaggleDownloader


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle training results")
    parser.add_argument("--kernel", default="research-v2-pretrain",
                        help="Kernel name to download from")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Download only checkpoints")
    parser.add_argument("--target", default="training/checkpoints/kaggle",
                        help="Local target directory")
    parser.add_argument("--resume", action="store_true",
                        help="Download and show resume instructions")
    parser.add_argument("--list", action="store_true",
                        help="List available output files")
    args = parser.parse_args()

    print("=" * 60)
    print("KAGGLE RESULT DOWNLOADER")
    print("=" * 60)

    status = validate_kaggle_setup()
    if not status["api_valid"]:
        print("\n  Kaggle setup incomplete. See docs/KAGGLE_SETUP.md")
        sys.exit(1)

    downloader = KaggleDownloader()

    if args.list:
        print(f"\n  Available outputs for {args.kernel}:")
        files = downloader.list_kernel_outputs(args.kernel)
        if files:
            for f in files:
                print(f"    {f}")
        else:
            print("    No outputs found")
        return

    if args.checkpoints:
        print(f"\n  Downloading checkpoints from {args.kernel}...")
        downloaded = downloader.download_checkpoints(args.kernel, args.target)
        print(f"\n  Downloaded {len(downloaded)} checkpoint files")
    else:
        print(f"\n  Downloading all outputs from {args.kernel}...")
        output_dir = downloader.download_kernel_output(args.kernel, args.target)
        print(f"\n  Saved to: {output_dir}")

    if args.resume:
        latest = downloader.find_latest_checkpoint(args.target)
        if latest:
            print(f"\n  Latest checkpoint: {latest}")
            print(f"  To resume training:")
            print(f"    python train_kaggle.py --resume")
        else:
            print("\n  No checkpoint found to resume from")

    print(f"\n{'=' * 60}")
    print(f"  Local checkpoint dir: {args.target}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
