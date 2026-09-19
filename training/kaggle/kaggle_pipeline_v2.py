#!/usr/bin/env python3
"""Kaggle kernel for the 200M MoE (181M total / 105.5M active) pretraining run.

Uses the LOCKED 200M config (configs/pretrain_200m_moe.yaml):
  - d=768, L=8, GQA 12q/6kv, head_dim 64
  - DeepSeekMoE: 1 shared (1536) + 12 routed (512), top-k 4
  - mHC enabled (n_streams 4, sinkhorn 20)
  - Muon (muon_clip), bf16
  - Corpus: tomiokasan/train-corpus-200m (corpus.jsonl + tokenizer)
  - Checkpoints: PP_SAVE_EVERY=500 -> tomiokasan/200m-moe-checkpoints

Boot:
  STEP 0: assemble research_hybrid from mounted code dataset (or import local)
  STEP 1: pretokenize mounted corpus -> /kaggle/working/tokenized/
  STEP 2: train with the 200M config; checkpoint + resume every 500 steps

Local smoke: PP_SMOKE=1 (tiny config, synthetic data, no Kaggle).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

CODE_DATASET = os.environ.get("PP_CODE_DATASET", "research-moe-code")
PACKAGE_FILES = [
    "__init__.py", "config.py", "model.py", "optim.py", "train.py",
    "attention.py", "transformer.py", "moe.py", "router.py", "experts.py",
    "mamba2.py", "mhc.py", "mla.py", "smoke_test.py", "audit.py",
]

# The 200M MoE config (mirrors configs/pretrain_200m_moe.yaml)
def build_configs():
    from research_hybrid.config import (
        AttentionConfig, CurriculumStage, HybridConfig, MHCConfig,
        ModelConfig, MoEConfig, TrainingConfig,
    )
    mc = ModelConfig(
        vocab_size=32768,
        d_model=768,
        n_layers=8,
        n_q_heads=12,
        n_kv_heads=6,
        head_dim=64,
        rope_theta=1_000_000.0,
        context_len=32768,
        use_ema=False,
        ff=MoEConfig(
            n_shared=1, shared_d_ff=1536, n_routed=12, routed_d_ff=512,
            top_k=4, capacity_factor=1.25, balance_coef=0.01,
            z_loss_coef=0.001, jitter_noise=0.01, routing_fn="softmax_topk",
        ),
        attention=AttentionConfig(
            pattern="block_sparse", kernel="auto", window_size=8192,
            anchor_size=128, block_size=256, top_blocks=8, chunk_size=2048,
        ),
        hybrid=HybridConfig(enabled=False),
        mhc=MHCConfig(enabled=True, n_streams=4, sinkhorn_iters=20),
    )
    tc = TrainingConfig(
        optimizer="muon_clip",
        lr=0.014, lr_1d=0.007, lr_min=3.0e-5,
        warmup_steps=600, weight_decay=0.1,
        muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5,
        qk_clip_tau=None,       # disabled: QK-Clip doubles attention memory (max-logits pass); OOMs T4
        grad_clip=1.0,
        batch_tokens=int(os.environ.get("PP_BATCH_TOKENS", "32768")),  # halved from 65536 for T4
        precision="bf16", grad_accum=8,
        total_steps=int(os.environ.get("PP_TOTAL_STEPS", "26300")),
        curriculum=[
            CurriculumStage(context_len=8192, fraction=0.5, attention_pattern="causal"),
            CurriculumStage(context_len=32768, fraction=0.5, attention_pattern="block_sparse"),
        ],
        seed=1337,
    )
    return mc, tc


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
            f"{CODE_DATASET} dataset not mounted; add it to kernel sources")
    dst = Path("/kaggle/working/research_hybrid")
    dst.mkdir(parents=True, exist_ok=True)
    for fname in PACKAGE_FILES:
        shutil.copyfile(src / fname, dst / fname)
    sys.path.insert(0, "/kaggle/working")
    print(f"  [code] research_hybrid assembled from {src}")


def _detect_owner():
    env = os.environ.get("PP_OWNER") or os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    if os.path.isdir("/kaggle/input/datasets"):
        for p in sorted(Path("/kaggle/input/datasets").glob("*/train-corpus-200m")):
            return p.parent.name
    return "tomiokasan"


_OWNER = _detect_owner()
CORPUS_DATASET = f"{_OWNER}/train-corpus-200m"


def _env_paths():
    if os.path.exists("/kaggle/input"):
        # Search /kaggle/input for the corpus dataset wherever it mounted
        # (modern Kaggle mounts at /kaggle/input/<slug>, older at
        # /kaggle/input/datasets/<owner>/<slug>).
        candidates = sorted(Path("/kaggle/input").rglob("corpus.jsonl"))
        if candidates:
            corpus = candidates[0]
            tok = corpus.parent / "tokenizer" / "tokenizer.model"
            if not tok.exists():
                # tokenizer may be at /kaggle/input/<slug>/tokenizer/tokenizer.model
                tok = corpus.parent.parent / "tokenizer" / "tokenizer.model"
            return corpus, tok, Path("/kaggle/working/tokenized")
        return (Path(f"/kaggle/input/datasets/{CORPUS_DATASET}/corpus.jsonl"),
                Path(f"/kaggle/input/datasets/{CORPUS_DATASET}/tokenizer/tokenizer.model"),
                Path("/kaggle/working/tokenized"))
    return (Path("datasets/train_corpus/corpus.jsonl"),
            Path("training/tokenizer/tokenizer.model"),
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
    print(f"Vocab      : {vocab} | Dtype: {np.dtype(dtype).name}")

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
        _ensure_code_package()
        run_pretokenize()
    else:
        # local dev: import straight from repo, expect tokens.bin nearby
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    mc, tc = build_configs()
    # PP_MEM_DEBUG / PP_EMPTY_CACHE default OFF (enable with env for OOM debugging;
    # both add sync overhead: empty_cache flushes allocator, MEM_DEBUG calls memory_allocated)
    os.environ.setdefault("PP_SAVE_EVERY", "500")   # checkpoint every 500 steps
    os.environ.setdefault("PP_EVAL_EVERY", "500")
    os.environ.setdefault("PP_LOG_EVERY", "25")

    print("\n" + "=" * 60)
    print("STEP 2/2: TRAINING (200M MoE — 181M total / 105.5M active)")
    print(f"  optimizer={tc.optimizer} lr={tc.lr}/{tc.lr_1d} steps={tc.total_steps} "
          f"batch_tokens={tc.batch_tokens}")
    curr = " ".join(f"({s.context_len},{s.attention_pattern},{s.fraction})" for s in tc.curriculum)
    print("  curriculum=" + curr)
    print(f"  mHC enabled: {mc.mhc.enabled} | save_every=500 (PP_SAVE_EVERY)")
    print("=" * 60)

    working = "/kaggle/working" if is_kaggle else "."
    prompts = ["def fibonacci(n):", "The quick brown fox", "import numpy as np",
               "Explain quantum computing", "What is machine learning?"]
    from research_hybrid.train import run
    run(mc, tc, working=working, prompts=prompts, max_new_tokens=200, temperature=0.8)


if __name__ == "__main__":
    main()
