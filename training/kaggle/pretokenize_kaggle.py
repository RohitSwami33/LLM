
#!/usr/bin/env python3
"""
Kaggle-friendly pretokenizer.
Simply run the cell or:
    python pretokenize.py
"""

import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
CORPUS = "/kaggle/input/datasets/tomiokasan/research-v2-corpus/corpus.jsonl"
TOKENIZER = "/kaggle/input/datasets/tomiokasan/research-v2-corpus/tokenizer/tokenizer.model"
OUT_DIR = "/kaggle/working/tokenized"
# ----------------------------------------


def detect_text(obj):
    if isinstance(obj, str):
        return obj
    for k in ("text", "content", "body"):
        v = obj.get(k)
        if isinstance(v, str):
            return v
    if isinstance(obj.get("messages"), list):
        parts = [m.get("content") for m in obj["messages"]
                 if isinstance(m, dict) and isinstance(m.get("content"), str)]
        if parts:
            return "\n".join(parts)
    return None


def main():
    corpus = Path(CORPUS)
    tokenizer = Path(TOKENIZER)
    out = Path(OUT_DIR)

    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found:\n{corpus}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found:\n{tokenizer}")

    out.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    vocab = sp.vocab_size()
    eos = sp.eos_id()
    dtype = np.uint16 if vocab <= 65535 else np.uint32

    print(f"Corpus     : {corpus}")
    print(f"Tokenizer  : {tokenizer}")
    print(f"Output dir : {out}")
    print(f"Vocab      : {vocab}")
    print(f"Dtype      : {np.dtype(dtype).name}")
    print()

    docs = 0
    toks = 0
    start = time.time()

    with open(corpus, "r", encoding="utf-8") as fin, \
            open(out / "tokens.bin", "wb") as fout:

        for line in tqdm(fin, desc="Tokenizing"):
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = detect_text(obj)
            if not text:
                continue

            ids = sp.encode(text, out_type=int)
            ids.append(eos)

            np.asarray(ids, dtype=dtype).tofile(fout)

            docs += 1
            toks += len(ids)

            if docs % 10000 == 0:
                elapsed = time.time() - start
                print(f"{docs:,} docs | {toks:,} tokens | {elapsed/60:.1f} min")

    metadata = {
        "documents": docs,
        "tokens": toks,
        "vocab_size": vocab,
        "dtype": np.dtype(dtype).name,
        "eos_id": eos,
        "binary_file": "tokens.bin",
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFinished!")
    print(f"Documents : {docs:,}")
    print(f"Tokens    : {toks:,}")
    print(f"Saved to  : {out}")


if __name__ == "__main__":
    main()
