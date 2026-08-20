"""Evaluation of the trained 68.7M dense TransformerLM (50k steps, 10 epochs).

Loads the final checkpoint (resume.pt from roronoazoro3008/research-v2-checkpoints,
step 50000) with the exact kaggle_pipeline architecture, generates sample text,
and reports the recorded training metrics.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.kaggle.kaggle_pipeline import ModelConfig, TransformerLM


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained 68.7M model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to resume.pt")
    parser.add_argument("--tokenizer", type=str,
                        default="/var/folders/xq/q0fl7m195zd9w5ms5g7vdnzm0000gn/T/opencode/ckpt_b/tokenizer.model",
                        help="Path to the corpus tokenizer.model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.vocab_size()
    print(f"Tokenizer vocab: {vocab_size}")

    class Tok:
        def encode(self, t, add_special_tokens=True):
            ids = sp.EncodeAsIds(t)
            if add_special_tokens:
                ids = [sp.bos_id()] + ids + [sp.eos_id()]
            return ids

        def decode(self, ids, skip_special_tokens=True):
            return sp.DecodeIds(ids)

    tokenizer = Tok()

    # ── Build model with the exact training architecture (dense, not MoE) ──
    mc = ModelConfig(
        vocab_size=vocab_size,
        d_model=576, n_heads=8, n_layers=6, d_ff=2304,
        dropout=0.0, max_seq_len=2048, norm_type="rmsnorm",
        activation="swiglu", rope=True, rope_base=10000.0,
        flash_attention=False, bias=False, tie_weights=False,
        gradient_checkpointing=False,
    )
    model = TransformerLM(mc)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint: step={ckpt.get('step')}, "
          f"total_tokens={ckpt.get('total_tokens'):,}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f}")

    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS (trained 68.7M model, step 50k)")
    print("=" * 70)

    prompts = [
        "The history of artificial intelligence begins with",
        "In quantum computing, the fundamental unit is",
        "Python is a programming language that",
        "def fibonacci(n):",
        "The theory of general relativity describes",
        "import numpy as np\n\ndef mean(x):",
    ]

    t0 = time.time()
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=50, top_p=0.95)
        text = tokenizer.decode(out[0].cpu().tolist(), skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {text}\n{'-' * 70}")
    elapsed = time.time() - t0
    print(f"\nGeneration time: {elapsed:.1f}s")

    # ── Recorded training metrics ──
    print("\n" + "=" * 70)
    print("RECORDED TRAINING SUMMARY")
    print("=" * 70)
    print(f"Steps:            {ckpt.get('step'):,} / 50,000")
    print(f"Tokens processed: {ckpt.get('total_tokens'):,} (3.28B budget)")
    print(f"Best val loss:    {ckpt.get('best_val_loss'):.4f}")
    print(f"Best val PPL:     {math.exp(min(ckpt.get('best_val_loss'), 20)):.2f}")

    return {
        "step": ckpt.get("step"),
        "total_tokens": ckpt.get("total_tokens"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "best_val_ppl": math.exp(min(ckpt.get("best_val_loss"), 20)),
        "params": n_params,
        "samples": len(prompts),
    }


if __name__ == "__main__":
    main()
