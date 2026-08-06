#!/usr/bin/env python3
"""Kaggle combined pipeline: pretokenize corpus -> train TransformerLM.

Self-contained: embeds the pretokenizer and the full training script so a
single kernel run tokenizes the attached dataset, writes tokens.bin to
/kaggle/working/tokenized/, and then trains.
"""
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------- PRETOKENIZE CONFIG ----------------
def _detect_owner():
    """Account owning the data: PP_OWNER env > KAGGLE_USERNAME env > mounted dataset > default."""
    env = os.environ.get("PP_OWNER") or os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    if os.path.isdir("/kaggle/input/datasets"):
        for p in sorted(Path("/kaggle/input/datasets").glob("*/research-v2-corpus")):
            return p.parent.name
    return "tomiokasan"


_OWNER = _detect_owner()
CORPUS_DATASET = f"{_OWNER}/research-v2-corpus"
CORPUS = f"/kaggle/input/datasets/{CORPUS_DATASET}/corpus.jsonl"
TOKENIZER = f"/kaggle/input/datasets/{CORPUS_DATASET}/tokenizer/tokenizer.model"
OUT_DIR = "/kaggle/working/tokenized"
# ----------------------------------------------------

# ---------------- CHECKPOINT UPLOAD/RESUME (Kaggle dataset) ----------------
# Checkpoints are pushed to a Kaggle dataset so the model is downloadable at
# any time and training can resume after the 12h session limit kills the run.
CHECKPOINT_DATASET = f"{_OWNER}/research-v2-checkpoints"
upload_queue = queue.Queue()
_KAGGLE_API = None
KAGGLE_AVAILABLE = False


def _ensure_kaggle():
    """Lazily init the Kaggle API client. Network is often not up at boot,
    so authenticate only when actually needed (download/upload time)."""
    global _KAGGLE_API, KAGGLE_AVAILABLE
    if KAGGLE_AVAILABLE:
        return True
    if os.environ.get("PP_SMOKE"):
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        try:
            api = KaggleApi()
            api.authenticate()
            _KAGGLE_API = api
            KAGGLE_AVAILABLE = True
            return True
        except Exception as _e:
            print(f"  [kaggle] auth failed: {type(_e).__name__}")
            return False
    except Exception as _e:
        print(f"  [kaggle] API unavailable: {type(_e).__name__}: {_e}")
        return False


def _ensure_kaggle_retry(tries=5, backoff=10):
    """_ensure_kaggle with retries, for startup/background contexts only."""
    for attempt in range(tries):
        if _ensure_kaggle():
            return True
        time.sleep(backoff * (attempt + 1))
    return False
# ---------------------------------------------------------------------------

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


def _env_paths():
    """Corpus/tokenizer/output paths for the current runtime (Kaggle/Colab/local)."""
    if os.path.exists("/kaggle/input"):
        return CORPUS, TOKENIZER, OUT_DIR
    if os.path.exists("/content"):
        return ("/content/datasets_dl/corpus.jsonl",
                "/content/datasets_dl/tokenizer/tokenizer.model",
                "/content/tokenized")
    return ("datasets/research_v2/corpus.jsonl",
            "datasets/research_v2/tokenizer/tokenizer.model",
            "tokenized")


def run_pretokenize():
    print("=" * 60)
    print("STEP 1/2: PRETOKENIZATION")
    print("=" * 60)
    corpus, tokenizer, out = map(Path, _env_paths())

    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found:\n{corpus}")
    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found:\n{tokenizer}")

    import sentencepiece as spm
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
                elapsed = time.time() - start
                print(f"{docs:,} docs | {toks:,} tokens | {elapsed/60:.1f} min")

    metadata = {
        "documents": docs,
        "tokens": toks,
        "vocab_size": vocab,
        "dtype": np.dtype(dtype).name,
        "eos_id": eos,
        "binary_file": "tokens.bin",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPretokenization done: {docs:,} docs, {toks:,} tokens")
    return out


# ============================================================
# TRAINING (embedded from train_standalone.py)
# ============================================================
import math
import random
import copy
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 576
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2304
    dropout: float = 0.0
    max_seq_len: int = 2048
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    rope: bool = True
    rope_base: float = 10000.0
    flash_attention: bool = False
    bias: bool = False
    tie_weights: bool = False
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.activation == "swiglu":
            self.d_ff = int(self.d_ff / 256) * 256
            if self.d_ff < 256:
                self.d_ff = 256
        assert self.d_model % self.n_heads == 0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    return torch.outer(t, freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x = x.float()
    cos = freqs_cis.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs_cis.sin().unsqueeze(0).unsqueeze(0)
    a = x[..., 0::2]
    b = x[..., 1::2]
    r = a * cos - b * sin
    i = a * sin + b * cos
    return torch.stack([r, i], dim=-1).flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, rope=True, rope_base=10000.0,
                 max_seq_len=2048, flash=False, bias=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.flash = flash
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = rope
        if rope:
            self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len, rope_base), persistent=False)
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
                             persistent=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        if self.rope:
            freqs = self.freqs_cis[:T]
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            out = attn.transpose(1, 2).reshape(B, T, C)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = self.causal_mask[:T, :T]
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
            attn = self.attn_dropout(F.softmax(attn, dim=-1))
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, activation="swiglu",
                 rope=True, rope_base=10000.0, max_seq_len=2048, flash=False,
                 bias=False, gradient_checkpointing=True):
        super().__init__()
        self.gcp = gradient_checkpointing
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rope, rope_base, max_seq_len, flash, bias)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout, bias)

    def forward(self, x, mask=None):
        if self.training and self.gcp:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mask, use_reentrant=False)
        return self._forward(x, mask)

    def _forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout,
                           config.activation, config.rope, config.rope_base, config.max_seq_len,
                           config.flash_attention, config.bias, config.gradient_checkpointing)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        if config.tie_weights:
            n_params -= self.tok_emb.weight.numel()
        print(f"  TransformerLM: {n_params:,} params ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, mask=None):
        B, T = input_ids.shape
        x = self.emb_dropout(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.95):
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = 0
                logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        self.train()
        return input_ids


class BinaryTokenDataset(Dataset):
    def __init__(self, tokens_path: str, metadata_path: str, max_seq_len: int = 2048,
                 vocab_size: int = 32000):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        with open(metadata_path) as f:
            meta = json.load(f)
        dtype_name = meta.get("dtype", "uint16")
        if dtype_name == "uint16":
            self.dtype = np.uint16
        elif dtype_name == "uint32":
            self.dtype = np.uint32
        else:
            raise ValueError(f"Unknown dtype: {dtype_name}")
        self.tokens = np.memmap(tokens_path, dtype=self.dtype, mode="r")
        self.total_tokens = len(self.tokens)
        self.num_sequences = max(0, (self.total_tokens - 1) // max_seq_len)
        print(f"  BinaryTokenDataset: {self.total_tokens:,} tokens, "
              f"{self.num_sequences:,} sequences (dtype={dtype_name})")
        max_id = int(self.tokens[:100000].max())
        if max_id >= vocab_size:
            raise ValueError(
                f"Token ID {max_id} >= vocab_size {vocab_size}. "
                f"Check your tokenizer config matches the model vocab_size."
            )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.max_seq_len
        input_ids = self.tokens[start:start + self.max_seq_len].astype(np.int64)
        labels = self.tokens[start + 1:start + self.max_seq_len + 1].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


class SimpleCollator:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self):
        md = dict(self.model.named_parameters())
        sd = dict(self.shadow.named_parameters())
        for name in md:
            if name in sd:
                sd[name].mul_(self.decay).add_(md[name], alpha=1 - self.decay)
        mb = dict(self.model.named_buffers())
        sb = dict(self.shadow.named_buffers())
        for name in mb:
            if name in sb:
                sb[name].copy_(mb[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow.load_state_dict(state["shadow"])


def save_checkpoint(path, model, optimizer, scheduler=None, scaler=None, ema=None,
                    step=0, total_tokens=0, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "step": step, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "total_tokens": total_tokens,
    }
    if scheduler: state["scheduler"] = scheduler.state_dict()
    if scaler: state["scaler"] = scaler.state_dict()
    if ema: state["ema"] = ema.state_dict()
    if extra: state.update(extra)
    torch.save(state, path + ".tmp")
    if os.path.exists(path): os.remove(path)
    os.rename(path + ".tmp", path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device=torch.device("cpu")):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return {"step": ckpt.get("step", 0), "total_tokens": ckpt.get("total_tokens", 0)}


@torch.no_grad()
def evaluate(model, dataloader, device, max_steps=50):
    model.eval()
    total_loss, total_tokens, correct, steps = 0.0, 0, 0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_steps: break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, loss = model(input_ids=ids, labels=labels)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = (shift_labels != -100) & (shift_labels != 0)
        correct += (preds == shift_labels).float().masked_select(mask).sum().item()
        total_tokens += mask.sum().item()
        valid = (labels != -100).sum().item()
        if valid > 0: total_loss += loss.item() * valid
        steps += 1
    if steps == 0 or total_tokens == 0:
        return {"val_loss": float("inf"), "val_perplexity": float("inf"), "val_accuracy": 0.0}
    avg = total_loss / total_tokens
    return {"val_loss": avg, "val_perplexity": math.exp(min(avg, 20)), "val_accuracy": correct / total_tokens}


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_new_tokens=200, temperature=0.8, device=torch.device("cpu")):
    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        results.append(tokenizer.decode(out[0].cpu().tolist(), skip_special_tokens=True))
    return results


def download_remote_checkpoint(ckpt_dir: Path):
    """Fetch the latest checkpoint dataset version into ckpt_dir as resume.pt."""
    if not _ensure_kaggle_retry():
        print("  Kaggle API unavailable - skipping remote checkpoint download")
        return
    try:
        dl_dir = ckpt_dir / "ckpt_download"
        dl_dir.mkdir(parents=True, exist_ok=True)
        _KAGGLE_API.dataset_download_files(CHECKPOINT_DATASET, path=str(dl_dir),
                                           unzip=True, quiet=True)
        src = dl_dir / "resume.pt"
        if src.exists():
            dst = ckpt_dir / "resume.pt"
            shutil.copyfile(str(src), str(dst))
            print(f"  Downloaded remote checkpoint -> {dst.name} ({src.stat().st_size/1e6:.0f}MB)")
        else:
            print("  Remote checkpoint dataset present but no resume.pt yet")
        shutil.rmtree(dl_dir, ignore_errors=True)
    except Exception as e:
        print(f"  No remote checkpoint to resume from: {type(e).__name__}: {e}")


def _enqueue_upload(ckpt_path, step, upload_dir):
    """Stage a checkpoint for the background upload thread (cheap hardlink)."""
    if not _ensure_kaggle():
        return
    try:
        stage = Path(upload_dir) / f"upload_{step}"
        stage.mkdir(parents=True, exist_ok=True)
        target = stage / "resume.pt"
        if target.exists():
            target.unlink()
        try:
            os.link(ckpt_path, target)
        except OSError:
            shutil.copyfile(ckpt_path, target)
        with open(stage / "dataset-metadata.json", "w") as f:
            json.dump({"id": CHECKPOINT_DATASET, "title": "research-v2-checkpoints",
                       "licenses": [{"name": "other"}]}, f, indent=2)
        with open(stage / "latest_step.txt", "w") as f:
            f.write(f"{step}\n")
        upload_queue.put(stage)
        print(f"  [upload] queued {stage.name}")
    except Exception as e:
        print(f"  [upload] queue failed: {type(e).__name__}: {e}")


def _upload_worker():
    while True:
        stage = upload_queue.get()
        try:
            if stage is None:
                return
            if not _ensure_kaggle_retry(tries=3, backoff=10):
                print(f"  [upload] {stage.name} skipped: kaggle API unavailable")
                return
            for attempt in range(1, 4):
                try:
                    t0 = time.time()
                    _KAGGLE_API.dataset_create_version(
                        str(stage), version_notes=f"step {stage.name.split('_')[-1]}",
                        quiet=True, convert_to_csv=False, delete_old_versions=True)
                    print(f"  [upload] {stage.name} pushed in {time.time()-t0:.0f}s")
                    break
                except Exception as e:
                    print(f"  [upload] {stage.name} attempt {attempt}/3 failed: {type(e).__name__}: {e}")
                    time.sleep([15, 60, 300][attempt - 1])
            else:
                print(f"  [upload] GAVE UP on {stage.name}; next save will re-upload")
        finally:
            upload_queue.task_done()
            shutil.rmtree(stage, ignore_errors=True)


def run_training():
    print("\n" + "=" * 60)
    print("STEP 2/2: TRAINING")
    print("=" * 60)

    is_kaggle = os.path.exists("/kaggle/working")
    if is_kaggle:
        CORPUS_DIR = Path(f"/kaggle/input/datasets/{CORPUS_DATASET}")
        WORKING = Path("/kaggle/working")
        TOKENIZED_DIR = WORKING / "tokenized"
    elif os.path.exists("/content"):
        CORPUS_DIR = Path("/content/datasets_dl")
        WORKING = Path("/content")
        TOKENIZED_DIR = WORKING / "tokenized"
    else:
        CORPUS_DIR = Path("datasets/research_v2")
        WORKING = Path(".")
        TOKENIZED_DIR = WORKING / "tokenized"

    upload_dir = WORKING / "checkpoint_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_upload_worker, daemon=True).start()

    config = {
        "model": {"vocab_size": 32000, "d_model": 576, "n_heads": 8, "n_layers": 6,
                  "d_ff": 2304, "dropout": 0.0, "max_seq_len": 2048, "norm_type": "rmsnorm",
                  "activation": "swiglu", "rope": True, "rope_base": 10000.0,
                  "flash_attention": True, "bias": False, "tie_weights": False,
                  "gradient_checkpointing": True},
        "training": {"optimizer": "adamw", "learning_rate": 3e-4,
                     "min_lr": 3e-5, "weight_decay": 0.1, "grad_clip": 5.0,
                     "scheduler": "cosine", "warmup_steps": 2000, "max_steps": 50000,
                     "batch_size": 4, "gradient_accumulation_steps": 8, "dtype": "fp16",
                     "compile": False, "use_ema": True, "ema_decay": 0.9999,
                     "save_every": 500, "keep_last_n": 3, "checkpoint_dir": "checkpoints",
                     "log_every": 25, "eval_every": 500, "eval_steps": 50,
                     "experiment_dir": "experiments", "seed": 42},
        "evaluation": {"prompts": ["def fibonacci(n):", "The quick brown fox",
                                   "import numpy as np", "Explain quantum computing",
                                   "What is machine learning?"],
                       "temperature": 0.8, "max_new_tokens": 200},
    }

    if os.environ.get("PP_SMOKE"):
        config["model"].update({"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256,
                                "max_seq_len": 128, "flash_attention": False,
                                "gradient_checkpointing": False})
        config["training"].update({"max_steps": int(os.environ.get("PP_SMOKE_STEPS", "16")),
                                   "batch_size": 2, "gradient_accumulation_steps": 2,
                                   "save_every": 4, "eval_every": 4, "warmup_steps": 2})
        config["evaluation"]["max_new_tokens"] = 16

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  No GPU, using CPU")

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n  Building model...")
    mc = ModelConfig.from_dict(model_cfg)
    raw_model = TransformerLM(mc)
    raw_model = raw_model.to(device)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"  Model: {total_params / 1e6:.1f}M params on {device}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        model = nn.DataParallel(raw_model)
        model = model.to(device)
        print(f"  Using {n_gpu} GPUs via DataParallel")
    else:
        model = raw_model
    device = torch.device("cuda:0") if torch.cuda.is_available() else device

    lr = train_cfg.get("learning_rate", 3e-4)
    wd = train_cfg.get("weight_decay", 0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd, eps=1e-8,
    )

    warmup = train_cfg.get("warmup_steps", 2000)
    max_steps = train_cfg.get("max_steps", 50000)
    min_lr = train_cfg.get("min_lr", 3e-5)
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = None
    if train_cfg.get("use_ema", True):
        ema = EMA(raw_model, decay=train_cfg.get("ema_decay", 0.9999))
        print("  EMA enabled")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    print(f"  Mixed precision: fp16 (GradScaler={scaler is not None})")

    tokens_path = str(TOKENIZED_DIR / "tokens.bin")
    metadata_path = str(TOKENIZED_DIR / "metadata.json")
    print(f"\n  Loading dataset from: {tokens_path}")
    print(f"  Metadata: {metadata_path}")

    if not Path(tokens_path).exists():
        print(f"  ERROR: tokens.bin not found at {tokens_path}")
        print("  Please run pretokenization first.")
        sys.exit(1)
    if not Path(metadata_path).exists():
        print(f"  ERROR: metadata.json not found at {metadata_path}")
        sys.exit(1)

    max_seq_len = model_cfg.get("max_seq_len", 2048)
    vocab_size = model_cfg.get("vocab_size", 32000)
    full_dataset = BinaryTokenDataset(tokens_path, metadata_path, max_seq_len, vocab_size)

    train_split = 0.95
    n_train = int(len(full_dataset) * train_split)
    n_val = len(full_dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    collator = SimpleCollator()
    batch_size = train_cfg.get("batch_size", 4) * max(1, n_gpu)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collator, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collator, pin_memory=True
    )
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Batch: {batch_size} (per-GPU {batch_size // max(1, n_gpu)})")

    log_dir = Path(train_cfg.get("experiment_dir", "experiments"))
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = None

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    total_tokens_count = 0
    best_val_loss = float("inf")

    download_remote_checkpoint(ckpt_dir)
    candidates = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if (ckpt_dir / "resume.pt").exists():
        candidates.append(ckpt_dir / "resume.pt")

    def _ckpt_step(p):
        name = p.name
        if name == "resume.pt":
            return 10**9
        return int(name.replace("step_", "").replace(".pt", "")) if name.startswith("step_") else -1

    for f in sorted(candidates, key=_ckpt_step, reverse=True):
        try:
            ckpt = torch.load(str(f), map_location="cpu", weights_only=False)
            raw_model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if ema and "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            global_step = ckpt.get("step", 0)
            total_tokens_count = ckpt.get("total_tokens", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"  Resumed from {f.name} at step {global_step}")
            break
        except Exception as e:
            print(f"  Failed to load {f.name}: {type(e).__name__}: {e}")
            continue

    resumed_tokens = total_tokens_count

    grad_accum = train_cfg.get("gradient_accumulation_steps", 8)
    grad_accum = max(1, grad_accum // max(1, n_gpu))
    grad_clip = train_cfg.get("grad_clip", 5.0)
    log_every = train_cfg.get("log_every", 25)
    save_every = train_cfg.get("save_every", 500)
    eval_every = train_cfg.get("eval_every", 500)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING from step {global_step}")
    print(f"  Max steps: {max_steps} | Grad accum: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"{'=' * 60}\n")

    model.train()
    train_start = time.time()
    step_loss, step_count = 0.0, 0
    train_acc_sum, train_acc_count = 0.0, 0

    while global_step < max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            total_tokens_count += input_ids.numel()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                logits, _ = model(input_ids=input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, raw_model.config.vocab_size),
                    shift_labels.reshape(-1), ignore_index=-100
                )
                loss = loss / grad_accum

            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                mask = (shift_labels != -100) & (shift_labels != 0)
                train_acc_sum += (preds == shift_labels).float().masked_select(mask).sum().item()
                train_acc_count += mask.sum().item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += loss.item() * grad_accum
            step_count += 1

            if (batch_idx + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema: ema.update()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = step_loss / step_count
                    train_acc = train_acc_sum / max(train_acc_count, 1)
                    lr_now = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - train_start
                    tok_s = (total_tokens_count - resumed_tokens) / max(elapsed, 1)
                    gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                    print(f"  Step {global_step:>6d} | tok {total_tokens_count/1e6:>8.1f}M | "
                          f"loss: {avg_loss:.4f} | acc: {train_acc:.4f} | "
                          f"lr: {lr_now:.2e} | tok/s: {tok_s:.0f} | gpu: {gpu_mem:.0f}MB")

                    if csv_writer is None:
                        import csv
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["step", "loss", "accuracy", "lr", "tokens", "tok_per_sec", "gpu_mb"])
                    csv_writer.writerow([global_step, f"{avg_loss:.6f}", f"{train_acc:.6f}",
                                        f"{lr_now:.8f}", total_tokens_count, f"{tok_s:.1f}", f"{gpu_mem:.1f}"])
                    csv_file.flush()

                    step_loss, step_count = 0.0, 0
                    train_acc_sum, train_acc_count = 0.0, 0

                if global_step % save_every == 0:
                    path = ckpt_dir / f"step_{global_step}.pt"
                    save_checkpoint(str(path), raw_model, optimizer, scheduler, scaler, ema=ema,
                                   step=global_step, total_tokens=total_tokens_count,
                                   extra={"best_val_loss": best_val_loss})
                    old = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
                    for f in old[:-train_cfg.get("keep_last_n", 3)]:
                        f.unlink()
                    _enqueue_upload(str(path), global_step, upload_dir)
                    du = shutil.disk_usage(str(WORKING))
                    print(f"  [disk] {du.used/1e9:.1f}GB used / {du.total/1e9:.1f}GB")

                if global_step % eval_every == 0:
                    print(f"\n  --- Eval at step {global_step} ---")
                    metrics = evaluate(raw_model, val_loader, device, max_steps=train_cfg.get("eval_steps", 50))
                    print(f"  Val loss: {metrics['val_loss']:.4f} | PPL: {metrics['val_perplexity']:.2f} | Acc: {metrics['val_accuracy']:.4f}")
                    if csv_writer:
                        csv_writer.writerow([global_step, f"{metrics['val_loss']:.6f}", f"{metrics['val_accuracy']:.6f}",
                                            "eval", 0, 0, 0])
                        csv_file.flush()
                    if metrics["val_loss"] < best_val_loss:
                        best_val_loss = metrics["val_loss"]
                        save_checkpoint(str(ckpt_dir / "best.pt"), raw_model, optimizer, scheduler, scaler, ema=ema,
                                       step=global_step, total_tokens=total_tokens_count,
                                       extra={"best_val_loss": best_val_loss})
                        print(f"    New best: {best_val_loss:.4f}")
                    model.train()

    save_checkpoint(str(ckpt_dir / "final.pt"), raw_model, optimizer, scheduler, scaler, ema=ema,
                   step=global_step, total_tokens=total_tokens_count,
                   extra={"best_val_loss": best_val_loss})
    _enqueue_upload(str(ckpt_dir / "final.pt"), global_step, upload_dir)

    print(f"\n  --- Sample Generation ---")
    model.eval()
    prompts = eval_cfg.get("prompts", ["def fibonacci(n):"])

    try:
        import sentencepiece as spm
        tok_path = str(CORPUS_DIR / "tokenizer" / "tokenizer.model")
        sp = spm.SentencePieceProcessor()
        sp.Load(tok_path)

        class Tok:
            def encode(self, t, add_special_tokens=True):
                ids = sp.EncodeAsIds(t)
                if add_special_tokens: ids = [sp.bos_id()] + ids + [sp.eos_id()]
                return ids
            def decode(self, ids, skip_special_tokens=True):
                return sp.DecodeIds(ids)

        tokenizer = Tok()
        for i, (p, s) in enumerate(zip(prompts, generate_samples(raw_model, tokenizer, prompts,
                                max_new_tokens=eval_cfg.get("max_new_tokens", 200),
                                temperature=eval_cfg.get("temperature", 0.8), device=device))):
            print(f"\n  Prompt {i+1}: {p}")
            print(f"  Output: {s[:200]}...")
    except Exception as e:
        print(f"  Sample generation skipped (tokenizer not available): {e}")

    elapsed = time.time() - train_start
    report = {
        "total_steps": global_step, "total_tokens": total_tokens_count,
        "best_val_loss": best_val_loss, "elapsed_hours": elapsed / 3600,
        "tokens_per_sec": total_tokens_count / max(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    with open(log_dir / "kaggle_report.json", "w") as f:
        json.dump(report, f, indent=2)
    csv_file.close()

    if KAGGLE_AVAILABLE:
        print("  Waiting for pending checkpoint uploads...")
        deadline = time.time() + 1800
        while upload_queue.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(10)
        if upload_queue.unfinished_tasks > 0:
            print("  WARNING: upload queue still non-empty; uploads will finish in background")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Steps: {global_step} | Tokens: {total_tokens_count/1e6:.1f}M")
    print(f"  Time: {elapsed/3600:.2f}h | Best val loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pretokenize()
    run_training()
