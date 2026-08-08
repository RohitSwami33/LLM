#!/usr/bin/env python3
"""Kaggle kernel for Hybrid-MoE-70M v2 pretraining (32K curriculum).

Boot sequence:
  STEP 0: locate the mounted research-moe-code dataset (research_hybrid package)
          and add it to sys.path
  STEP 1: pretokenize the mounted research-v2-corpus -> /kaggle/working/tokenized/
  STEP 2: train with research_hybrid.train.run() (curriculum 8K -> 32K,
          MuonClip + QK-Clip, checkpoint resume/upload to research-moe-checkpoints)

Local smoke: PP_SMOKE=1 (tiny config, synthetic data, no Kaggle access).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# The training forward's transient peaks (MoE routed activations, attention
# chunks) leave the CUDA caching allocator's pool fragmented (reserved ~13 GB
# while only ~3 GB is live). Without expandable segments the loss chunk's 1 GB
# fp32 softmax temp cannot find a contiguous block -> OOM. Must be set before
# any CUDA allocation (first torch.cuda call), so do it here at module scope.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

CODE_DATASET = "research-moe-code"
PACKAGE_FILES = [
    "__init__.py", "config.py", "model.py", "optim.py", "train.py",
    "attention.py", "transformer.py", "moe.py", "router.py", "experts.py",
    "mamba2.py", "mhc.py", "mla.py", "smoke_test.py", "audit.py",
]


def _ensure_code_package():
    """Assemble research_hybrid/ in /kaggle/working from the mounted flat files."""
    src = None
    for root in ("/kaggle/input", "/kaggle/input/datasets"):
        base = Path(root)
        if not base.exists():
            continue
        for p in base.glob(f"*/{CODE_DATASET}"):
            if (p / "config.py").exists() and (p / "train.py").exists():
                src = p
                break
        if src:
            break
    if src is None:
        raise FileNotFoundError(
            f"research-moe-code dataset not mounted (looked for */{CODE_DATASET} under "
            "/kaggle/input); add it to the kernel's dataset sources")
    dst = Path("/kaggle/working/research_hybrid")
    dst.mkdir(parents=True, exist_ok=True)
    for fname in PACKAGE_FILES:
        shutil.copyfile(src / fname, dst / fname)
    sys.path.insert(0, "/kaggle/working")
    print(f"  [code] research_hybrid assembled from {src}")


if Path("/kaggle/working").exists():
    _ensure_code_package()
else:
    import research_hybrid  # local dev: import straight from the repo

from research_hybrid.config import ModelConfig, TrainingConfig
from research_hybrid.train import run

# ---------------- PRETOKENIZE (reuses the v1 pipeline's format) ----------------

def _detect_owner():
    env = os.environ.get("PP_OWNER") or os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    if os.path.isdir("/kaggle/input/datasets"):
        for p in sorted(Path("/kaggle/input/datasets").glob("*/research-v2-corpus")):
            return p.parent.name
    return "tomiokasan"


_OWNER = _detect_owner()
CORPUS_DATASET = f"{_OWNER}/research-v2-corpus"


def _env_paths():
    if os.path.exists("/kaggle/input"):
        return (Path(f"/kaggle/input/datasets/{CORPUS_DATASET}/corpus.jsonl"),
                Path(f"/kaggle/input/datasets/{CORPUS_DATASET}/tokenizer/tokenizer.model"),
                Path("/kaggle/working/tokenized"))
    if os.path.exists("/content"):
        return (Path("/content/datasets_dl/corpus.jsonl"),
                Path("/content/datasets_dl/tokenizer/tokenizer.model"),
                Path("/content/tokenized"))
    return (Path("datasets/research_v2/corpus.jsonl"),
            Path("datasets/research_v2/tokenizer/tokenizer.model"),
            Path("tokenized"))


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


def run_pretokenize():
    print("=" * 60)
    print("STEP 1/2: PRETOKENIZATION")
    print("=" * 60)
    corpus, tokenizer, out = _env_paths()
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found:\n{corpus}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found:\n{tokenizer}")

    import sentencepiece as spm
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer, out / "tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    vocab = sp.vocab_size()
    eos = sp.eos_id()
    dtype = np.uint16 if vocab <= 65535 else np.uint32

    print(f"Corpus     : {corpus}")
    print(f"Tokenizer  : {tokenizer}")
    print(f"Output dir : {out}")
    print(f"Vocab      : {vocab}")
    print(f"Dtype      : {np.dtype(dtype).name}")

    docs, toks, start = 0, 0, time.time()
    with open(corpus, "r", encoding="utf-8") as fin, \
            open(out / "tokens.bin", "wb") as fout:
        for line in fin:
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
            if docs % 20000 == 0:
                print(f"{docs:,} docs | {toks:,} tokens | {(time.time()-start)/60:.1f} min")

    with open(out / "metadata.json", "w") as f:
        json.dump({"documents": docs, "tokens": toks, "vocab_size": vocab,
                   "dtype": np.dtype(dtype).name, "eos_id": eos,
                   "binary_file": "tokens.bin",
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nPretokenization done: {docs:,} docs, {toks:,} tokens")


def main():
    is_kaggle = os.path.exists("/kaggle/working")
    if is_kaggle:
        run_pretokenize()
    else:
        print("SKIP pretokenize (not a Kaggle runtime); expecting tokens.bin nearby")

    tc = TrainingConfig()
    mc = ModelConfig()
    os.environ.setdefault("PP_MEM_DEBUG", "1")
    print("\n" + "=" * 60)
    print("STEP 2/2: TRAINING (Hybrid-MoE-70M v2)")
    print(f"  optimizer={tc.optimizer} lr={tc.lr}/{tc.lr_1d} steps={tc.total_steps} "
          f"batch_tokens={tc.batch_tokens}")
    print(f"  curriculum={[(s.context_len, s.attention_pattern, s.fraction)
                          for s in tc.curriculum]}")
    print(f"  qk_clip_tau={tc.qk_clip_tau} checkpointing={mc.use_gradient_checkpointing}")
    print("=" * 60)

    working = "/kaggle/working" if is_kaggle else "."
    prompts = ["def fibonacci(n):", "The quick brown fox", "import numpy as np",
               "Explain quantum computing", "What is machine learning?"]
    run(mc, tc, working=working, prompts=prompts, max_new_tokens=200, temperature=0.8)
if __name__ == "__main__":
    main()
