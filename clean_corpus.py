#!/usr/bin/env python3
"""Corpus cleaning pipeline CLI.

Usage:
    python clean_corpus.py --input input.jsonl --output output.jsonl
    python clean_corpus.py --input input.jsonl --output output.jsonl --config training/corpus/config.yaml
    python clean_corpus.py --input input.jsonl --output output.jsonl --sample 100000
"""

import argparse
import sys
import os
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="Corpus cleaning pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--config", "-c", default="training/corpus/config.yaml",
                        help="Cleaning config YAML (default: training/corpus/config.yaml)")
    parser.add_argument("--text-key", default="text",
                        help="JSON key containing text (default: text)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N documents (for testing)")
    parser.add_argument("--report-dir", default=None,
                        help="Directory to save cleaning report (default: same as output)")
    args = parser.parse_args()

    # Load config
    from training.corpus.cleaner import CorpusCleaner
    from training.corpus.stats import save_cleaning_report, save_cleaning_summary

    print(f"Loading config: {args.config}")
    cleaner = CorpusCleaner.from_yaml(args.config)

    # Run cleaning
    report = cleaner.clean_file(
        input_path=args.input,
        output_path=args.output,
        text_key=args.text_key,
        sample_size=args.sample,
    )

    # Save report
    report_dir = args.report_dir or os.path.dirname(args.output)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "cleaning_report.json")
    summary_path = os.path.join(report_dir, "cleaning_summary.txt")

    save_cleaning_report(report, report_path)
    save_cleaning_summary(report, summary_path)

    print(f"\nDone! Cleaned corpus: {args.output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
