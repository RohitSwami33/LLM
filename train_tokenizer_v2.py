#!/usr/bin/env python3
"""Train tokenizer with modern LLM best practices.

Based on research of Llama 3, Qwen 3, DeepSeek V3, Gemma 3, Mistral tokenizers.

Key features:
- SentencePiece BPE (used by Gemma 3, Mistral V3)
- Byte fallback (prevents UNK tokens, used by all modern LLMs)
- NFKC normalization (standard Unicode normalization)
- Random sentence sampling (faster training, avoids OOM)
- shuffle_input_sentence = true (better distribution)
- split_digits = true (improves numerical reasoning)
- Whitespace preservation (▁ prefix)
"""

import os
import sys
import json
import time
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_texts(data_path: str, max_texts: int = None) -> list:
    """Extract text strings from JSONL or plain text file."""
    texts = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Extract text field, or join all string values
                    text = item.get("text", "")
                    if not text:
                        text = " ".join(
                            str(v) for v in item.values() if isinstance(v, str)
                        )
                    if text.strip():
                        texts.append(text.strip())
                    if max_texts and len(texts) >= max_texts:
                        break
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
                if max_texts and len(texts) >= max_texts:
                    break
    return texts


def train_tokenizer(
    data_path: str,
    output_path: str,
    vocab_size: int = 32000,
    # Modern best practices
    byte_fallback: bool = True,
    normalization: str = "nfkc",
    input_sentence_size: int = 10_000_000,
    shuffle_input_sentence: bool = True,
    max_sentence_length: int = 16384,
    split_digits: bool = True,
    character_coverage: float = 1.0,
    num_threads: int = None,
):
    """Train SentencePiece BPE tokenizer with modern best practices.
    
    Configuration based on research:
    - Gemma 3: SentencePiece, split_digits, preserved whitespace, byte-level
    - Mistral V3: SentencePiece BPE
    - Llama 3/Qwen 3/DeepSeek: Byte-level BPE with byte fallback
    
    Args:
        data_path: Path to training data (JSONL or text)
        output_path: Path to save tokenizer.model
        vocab_size: Vocabulary size (default: 32000)
        byte_fallback: Enable byte fallback for OOV characters
        normalization: Unicode normalization (nfkc recommended)
        input_sentence_size: Number of sentences to sample (0 = all)
        shuffle_input_sentence: Shuffle sentences before training
        max_sentence_length: Maximum sentence length in bytes
        split_digits: Split digits into individual characters
        character_coverage: Character coverage (1.0 = all characters)
        num_threads: Number of threads (default: cpu_count)
    """
    import sentencepiece as spm

    if num_threads is None:
        num_threads = os.cpu_count() or 4

    print(f"Training tokenizer with modern best practices:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Byte fallback: {byte_fallback}")
    print(f"  Normalization: {normalization}")
    print(f"  Input sentence size: {input_sentence_size}")
    print(f"  Shuffle input: {shuffle_input_sentence}")
    print(f"  Max sentence length: {max_sentence_length}")
    print(f"  Split digits: {split_digits}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Threads: {num_threads}")

    # Extract texts
    print(f"\nLoading data from {data_path}...")
    texts = extract_texts(data_path)
    print(f"  Loaded {len(texts):,} documents")

    # Write to temp file
    temp_path = output_path + ".tmp.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    # Train SentencePiece
    print(f"\nTraining SentencePiece BPE...")
    start_time = time.time()

    spm.SentencePieceTrainer.Train(
        input=temp_path,
        model_prefix=output_path.replace(".model", ""),
        vocab_size=vocab_size,
        model_type="bpe",
        # Special tokens
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        # Modern best practices
        byte_fallback=byte_fallback,
        normalization_rule_name=normalization,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        max_sentence_length=max_sentence_length,
        split_digits=split_digits,
        character_coverage=character_coverage,
        # Training parameters
        num_threads=num_threads,
        # Preserve whitespace correctly
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=True,
    )

    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s")

    # Cleanup
    os.remove(temp_path)

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.Load(output_path)
    print(f"\nVerification:")
    print(f"  Vocab size: {sp.GetPieceSize()}")
    print(f"  BOS id: {sp.bos_id()}")
    print(f"  EOS id: {sp.eos_id()}")
    print(f"  UNK id: {sp.unk_id()}")
    print(f"  PAD id: {sp.pad_id()}")

    # Check byte fallback tokens
    byte_tokens = sum(1 for i in range(sp.GetPieceSize()) if "<0x" in sp.IdToPiece(i))
    print(f"  Byte fallback tokens: {byte_tokens}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train tokenizer with modern LLM best practices"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training data path (JSONL or text)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/research_v2/tokenizer/tokenizer.model",
        help="Output path for tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=10_000_000,
        help="Number of sentences to sample (0 = all)",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=16384,
        help="Maximum sentence length in bytes",
    )
    parser.add_argument(
        "--no-byte-fallback",
        action="store_true",
        help="Disable byte fallback",
    )
    parser.add_argument(
        "--no-split-digits",
        action="store_true",
        help="Disable digit splitting",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="nfkc",
        choices=["nfkc", "nfc", "identity"],
        help="Unicode normalization",
    )
    args = parser.parse_args()

    train_tokenizer(
        data_path=args.data,
        output_path=args.output,
        vocab_size=args.vocab_size,
        byte_fallback=not args.no_byte_fallback,
        normalization=args.normalization,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=args.max_sentence_length,
        split_digits=not args.no_split_digits,
    )


if __name__ == "__main__":
    main()
