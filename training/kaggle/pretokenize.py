
#!/usr/bin/env python3
"""
pretokenize.py

Stream a JSONL corpus, tokenize with SentencePiece, and write a binary token file.

Usage:
    python pretokenize.py \
        --corpus corpus.jsonl \
        --tokenizer tokenizer.model \
        --out_dir data_tokens
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def detect_text(obj):
    """Return text from common JSONL formats."""
    if isinstance(obj, str):
        return obj

    for key in ("text", "content", "body"):
        if key in obj and isinstance(obj[key], str):
            return obj[key]

    if "messages" in obj:
        parts = []
        for m in obj["messages"]:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str):
                    parts.append(c)
        if parts:
            return "\n".join(parts)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    vocab = sp.vocab_size()
    eos = sp.eos_id()

    dtype = np.uint16 if vocab <= 65535 else np.uint32

    bin_path = out / "tokens.bin"

    docs = 0
    toks = 0
    start = time.time()

    with open(args.corpus, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        for line in tqdm(fin, desc="Pretokenizing"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = detect_text(obj)
            if not text:
                continue

            ids = sp.encode(text, out_type=int)
            ids.append(eos)

            arr = np.asarray(ids, dtype=dtype)
            arr.tofile(fout)

            docs += 1
            toks += len(arr)

            if docs % 10000 == 0:
                elapsed = time.time() - start
                rate = docs / elapsed if elapsed else 0
                print(
                    f"[{docs:,} docs] "
                    f"{toks:,} tokens | "
                    f"{rate:.1f} docs/s | "
                    f"elapsed {elapsed/60:.1f} min"
                )

    meta = {
        "tokenizer_vocab_size": vocab,
        "dtype": np.dtype(dtype).name,
        "eos_id": eos,
        "documents": docs,
        "tokens": toks,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "binary_file": "tokens.bin",
    }

    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Documents : {docs:,}")
    print(f"Tokens    : {toks:,}")
    print(f"Output    : {bin_path}")


if __name__ == "__main__":
    main()
