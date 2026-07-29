#!/usr/bin/env python3
"""pipeline.py — Unified training pipeline (local + Kaggle).

Usage:
    python pipeline.py                        # Local training
    python pipeline.py --kaggle               # Kaggle training
    python pipeline.py --kaggle --submit      # Deploy + submit
    python pipeline.py --kaggle --resume      # Resume on Kaggle
    python pipeline.py --kaggle --monitor     # Monitor Kaggle run
    python pipeline.py --kaggle --download    # Download results
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM pretraining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                          # Local training (Apple Silicon)
  python pipeline.py --config training/configs/pretrain_research_v2.yaml
  python pipeline.py --kaggle                 # Submit to Kaggle
  python pipeline.py --kaggle --submit        # Deploy and submit
  python pipeline.py --kaggle --monitor       # Monitor Kaggle run
  python pipeline.py --kaggle --download      # Download results
        """,
    )
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle GPU instead of local training")
    parser.add_argument("--config", default=None,
                        help="Training config (local only)")
    parser.add_argument("--submit", action="store_true",
                        help="Submit to Kaggle (with --kaggle)")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor Kaggle run (with --kaggle)")
    parser.add_argument("--download", action="store_true",
                        help="Download Kaggle results (with --kaggle)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--kernel", default=None,
                        help="Kaggle kernel name")
    parser.add_argument("--gpu", default=None,
                        help="Kaggle GPU type (T4/P100/L4)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps")
    args = parser.parse_args()

    if args.kaggle:
        _run_kaggle(args)
    else:
        _run_local(args)


def _run_kaggle(args):
    """Kaggle pipeline."""
    print("=" * 60)
    print("KAGGLE TRAINING PIPELINE")
    print("=" * 60)

    from kaggle.auth import validate_kaggle_setup
    status = validate_kaggle_setup()
    if not status["api_valid"]:
        print("\n  Kaggle not configured. See docs/KAGGLE_SETUP.md")
        sys.exit(1)

    if args.monitor:
        from kaggle.monitor import KaggleMonitor
        kernel = args.kernel or "research-v2-pretrain"
        monitor = KaggleMonitor()
        monitor.monitor_loop(kernel)

    elif args.download:
        from kaggle.download import KaggleDownloader
        kernel = args.kernel or "research-v2-pretrain"
        downloader = KaggleDownloader()
        downloader.download_checkpoints(kernel)

    elif args.submit:
        # Full deploy + submit
        print("\n[1/3] Packaging project...")
        from kaggle.package import ProjectPacker
        packer = ProjectPacker()
        packer.pack("kaggle_project.zip")

        print("\n[2/3] Uploading dataset...")
        from kaggle.dataset import KaggleDataset
        ds = KaggleDataset()
        if ds.exists("research-v2-corpus"):
            print("  Dataset already on Kaggle, reusing")
        else:
            ds.prepare_corpus_upload()

        print("\n[3/3] Submitting kernel...")
        from kaggle.kernel import KaggleKernel
        kernel = KaggleKernel()
        kernel_name = args.kernel or "research-v2-pretrain"
        gpu = args.gpu or "T4"
        url = kernel.submit(kernel_name=kernel_name, gpu_type=gpu)
        print(f"\n  Training submitted: {url}")
        print(f"  Monitor: python pipeline.py --kaggle --monitor")

    else:
        # Just submit (assumes already deployed)
        from kaggle.kernel import KaggleKernel
        kernel = KaggleKernel()
        kernel_name = args.kernel or "research-v2-pretrain"
        gpu = args.gpu or "T4"
        url = kernel.submit(kernel_name=kernel_name, gpu_type=gpu)
        print(f"\n  Training submitted: {url}")
        print(f"  Monitor: python pipeline.py --kaggle --monitor")


def _run_local(args):
    """Local training pipeline."""
    print("=" * 60)
    print("LOCAL TRAINING PIPELINE")
    print("=" * 60)

    config_path = args.config or "training/configs/pretrain_research_v2.yaml"

    # Detect platform
    import platform
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon — use optimized trainer
        print(f"\n  Detected: Apple Silicon ({machine})")
        print(f"  Using: train_optimized.py")
        print(f"  Config: {config_path}")

        cmd = f".venv/bin/python3 train_optimized.py --config {config_path}"
        if args.max_steps:
            cmd += f" --max-steps {args.max_steps}"
        if args.resume:
            cmd += " --resume"
        import os
        os.system(cmd)

    else:
        # Linux/CUDA — use standard trainer
        print(f"\n  Detected: {system} {machine}")
        print(f"  Using: training/scripts/train.py")
        print(f"  Config: {config_path}")

        cmd = f"python training/scripts/train.py --config {config_path}"
        if args.max_steps:
            # Config override not supported by standard trainer
            print(f"  Note: --max-steps override not supported by standard trainer")
        import os
        os.system(cmd)


if __name__ == "__main__":
    main()
