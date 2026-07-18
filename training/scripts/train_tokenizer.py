#!/usr/bin/env python3
"""Train tokenizer from scratch.

Usage:
    python training/scripts/train_tokenizer.py --data datasets/synthetic/v1/dataset.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument("--data", type=str, required=True, help="Training data path (JSONL or text)")
    parser.add_argument("--output", type=str, default="training/tokenizer/tokenizer.model",
                        help="Output path for trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--type", type=str, default="sentencepiece",
                        choices=["sentencepiece", "huggingface"],
                        help="Tokenizer type")
    args = parser.parse_args()

    from training.tokenizer.tokenizer import train_tokenizer
    path = train_tokenizer(
        data_path=args.data,
        output_path=args.output,
        vocab_size=args.vocab_size,
        tokenizer_type=args.type,
    )
    print(f"Tokenizer trained and saved to: {path}")


if __name__ == "__main__":
    main()
