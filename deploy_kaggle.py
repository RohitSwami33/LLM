#!/usr/bin/env python3
"""deploy_kaggle.py — Package and upload project to Kaggle.

Usage:
    python deploy_kaggle.py                     # Package + upload
    python deploy_kaggle.py --package-only      # Just create zip
    python deploy_kaggle.py --upload-dataset    # Also upload corpus
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kaggle.auth import validate_kaggle_setup, KaggleAuth
from kaggle.package import ProjectPacker
from kaggle.dataset import KaggleDataset
from kaggle.kernel import KaggleKernel


def main():
    parser = argparse.ArgumentParser(description="Deploy project to Kaggle")
    parser.add_argument("--package-only", action="store_true",
                        help="Only create package, don't upload")
    parser.add_argument("--upload-dataset", action="store_true",
                        help="Upload research corpus as Kaggle dataset")
    parser.add_argument("--dataset-slug", default="research-v2-corpus",
                        help="Dataset slug for corpus upload")
    parser.add_argument("--kernel-name", default="research-v2-pretrain",
                        help="Kernel name")
    parser.add_argument("--gpu", default="T4", choices=["T4", "P100", "L4"],
                        help="GPU type")
    parser.add_argument("--output", default="kaggle_project.zip",
                        help="Output zip path")
    parser.add_argument("--resume", action="store_true",
                        help="Include latest checkpoint in package")
    args = parser.parse_args()

    print("=" * 60)
    print("KAGGLE DEPLOYMENT")
    print("=" * 60)

    # Step 1: Validate Kaggle setup
    print("\n[1/4] Validating Kaggle setup...")
    status = validate_kaggle_setup()
    if not status["cli_installed"] or not status["api_valid"]:
        print("\n  Kaggle setup incomplete. Run setup first.")
        print("  See docs/KAGGLE_SETUP.md")
        sys.exit(1)

    auth = KaggleAuth()

    # Step 2: Package project
    print("\n[2/4] Packaging project...")
    packer = ProjectPacker()
    extra_files = ["pipeline.py"]

    if args.resume:
        # Include latest checkpoint
        ckpt_dir = Path("training/checkpoints/kaggle")
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*.pt"):
                extra_files.append(str(f))

    zip_path = packer.pack(args.output, extra_files=extra_files)

    # Step 3: Upload corpus (if requested)
    dataset_ref = None
    if args.upload_dataset:
        print("\n[3/4] Uploading corpus dataset...")
        ds = KaggleDataset()
        dataset_ref = ds.prepare_corpus_upload(
            slug=args.dataset_slug,
            force=False,
        )
    else:
        print("\n[3/4] Skipping dataset upload (use --upload-dataset)")

    # Step 4: Generate kernel metadata
    print("\n[4/4] Generating kernel metadata...")
    kernel = KaggleKernel()
    dataset_sources = []
    if dataset_ref:
        dataset_sources.append(dataset_ref)
    else:
        # Reference existing dataset
        ds = KaggleDataset()
        if ds.exists(args.dataset_slug):
            dataset_sources.append(f"{status['username']}/{args.dataset_slug}")

    kernel.generate_kernel_metadata(
        kernel_name=args.kernel_name,
        dataset_sources=dataset_sources,
    )

    print(f"\n{'=' * 60}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Package: {zip_path}")
    print(f"  Kernel: {args.kernel_name}")
    if dataset_sources:
        print(f"  Dataset: {dataset_sources[0]}")
    print(f"\n  Next steps:")
    print(f"    1. python train_kaggle.py --submit")
    print(f"    2. python monitor_kaggle.py --kernel {args.kernel_name}")
    print(f"    3. python download_results.py --kernel {args.kernel_name}")


if __name__ == "__main__":
    main()
