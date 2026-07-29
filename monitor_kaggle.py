#!/usr/bin/env python3
"""monitor_kaggle.py — Monitor Kaggle training progress in real-time.

Usage:
    python monitor_kaggle.py                        # Monitor default kernel
    python monitor_kaggle.py --kernel my-kernel     # Monitor specific kernel
    python monitor_kaggle.py --logs                 # Show recent logs
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from kaggle.auth import validate_kaggle_setup
from kaggle.monitor import KaggleMonitor


def main():
    parser = argparse.ArgumentParser(description="Monitor Kaggle training")
    parser.add_argument("--kernel", default="research-v2-pretrain",
                        help="Kernel name to monitor")
    parser.add_argument("--logs", action="store_true",
                        help="Show recent logs and exit")
    parser.add_argument("--poll", type=int, default=30,
                        help="Poll interval in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("KAGGLE TRAINING MONITOR")
    print("=" * 60)

    status = validate_kaggle_setup()
    if not status["api_valid"]:
        print("\n  Kaggle setup incomplete. See docs/KAGGLE_SETUP.md")
        sys.exit(1)

    monitor = KaggleMonitor()

    if args.logs:
        print(f"\n  Recent logs for {args.kernel}:")
        print("-" * 60)
        logs = monitor.get_logs(args.kernel, tail=100)
        print(logs)
        print("-" * 60)
    else:
        monitor.monitor_loop(
            args.kernel,
            poll_interval=args.poll,
        )


if __name__ == "__main__":
    main()
