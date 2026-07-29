#!/usr/bin/env python3
"""train_kaggle.py — Submit and manage Kaggle training runs.

Usage:
    python train_kaggle.py                    # Submit training
    python train_kaggle.py --submit           # Submit only
    python train_kaggle.py --status           # Check status
    python train_kaggle.py --cancel           # Cancel run
    python train_kaggle.py --wait             # Submit and wait
    python train_kaggle.py --resume           # Upload checkpoint and resume
"""
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kaggle.auth import validate_kaggle_setup
from kaggle.kernel import KaggleKernel
from kaggle.download import KaggleDownloader
from kaggle.dataset import KaggleDataset


def load_kaggle_config() -> dict:
    """Load Kaggle config."""
    config_path = Path("configs/kaggle.yaml")
    if not config_path.exists():
        print(f"  Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Kaggle training management")
    parser.add_argument("--submit", action="store_true",
                        help="Submit training kernel")
    parser.add_argument("--status", action="store_true",
                        help="Check kernel status")
    parser.add_argument("--cancel", action="store_true",
                        help="Cancel running kernel")
    parser.add_argument("--wait", action="store_true",
                        help="Submit and wait for completion")
    parser.add_argument("--resume", action="store_true",
                        help="Upload checkpoint and resume training")
    parser.add_argument("--kernel", default=None,
                        help="Kernel name (overrides config)")
    parser.add_argument("--gpu", default=None,
                        help="GPU type (overrides config)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max training steps")
    args = parser.parse_args()

    print("=" * 60)
    print("KAGGLE TRAINING")
    print("=" * 60)

    # Validate setup
    print("\n  Validating Kaggle setup...")
    status = validate_kaggle_setup()
    if not status["cli_installed"] or not status["api_valid"]:
        print("\n  Kaggle setup incomplete. See docs/KAGGLE_SETUP.md")
        sys.exit(1)

    config = load_kaggle_config()
    kaggle_cfg = config.get("kaggle", {})
    kernel_name = args.kernel or kaggle_cfg.get("kernel_name", "research-v2-pretrain")
    gpu_type = args.gpu or kaggle_cfg.get("gpu_type", "T4")

    kernel = KaggleKernel()

    # Resume mode: upload checkpoint first
    if args.resume:
        print("\n  Uploading checkpoint for resume...")
        downloader = KaggleDownloader()
        latest = downloader.find_latest_checkpoint()
        if latest:
            print(f"  Found checkpoint: {latest}")
            # Checkpoint is in the project, will be picked up by kernel
        else:
            print("  No local checkpoint found, starting fresh")

    if args.submit or args.wait:
        # Submit kernel
        print(f"\n  Submitting kernel: {kernel_name}")
        url = kernel.submit(
            kernel_name=kernel_name,
            gpu_type=gpu_type,
            enable_internet=kaggle_cfg.get("enable_internet", True),
        )
        print(f"  URL: {url}")

        if args.wait:
            print("\n  Waiting for completion...")
            result = kernel.wait_for_completion(kernel_name)
            final_state = result.get("status", "unknown")
            print(f"\n  Final status: {final_state}")

            if final_state == "complete":
                print("  Training complete! Downloading results...")
                downloader = KaggleDownloader()
                downloader.download_checkpoints(kernel_name)

    elif args.status:
        print(f"\n  Checking status: {kernel_name}")
        result = kernel.status(kernel_name)
        state = result.get("status", "unknown")
        print(f"  Status: {state}")

        if state == "running":
            print("  Kernel is currently running")
        elif state == "complete":
            print("  Training complete")
        elif state == "error":
            print("  Kernel encountered an error")

    elif args.cancel:
        print(f"\n  Cancelling kernel: {kernel_name}")
        if kernel.cancel(kernel_name):
            print("  Kernel cancelled")
        else:
            print("  Could not cancel kernel")

    else:
        # Default: submit
        print(f"\n  Submitting kernel: {kernel_name}")
        url = kernel.submit(
            kernel_name=kernel_name,
            gpu_type=gpu_type,
            enable_internet=kaggle_cfg.get("enable_internet", True),
        )
        print(f"  URL: {url}")

    print(f"\n{'=' * 60}")
    print("  Monitor: python monitor_kaggle.py")
    print("  Download: python download_results.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
