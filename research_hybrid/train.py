"""Training pipeline for Hybrid-MoE-70M (v2, 32K design).

Implements the recipe of docs/design.md section 7: curriculum (8K dense causal ->
32K block_sparse), MuonClip (Kimi K2 Algorithm 1) with QK-Clip, fp16 autocast
(fp32 weights -> fp32 grads, so no GradScaler is needed), gradient checkpointing,
EMA, Kaggle-dataset checkpoint resume/upload, eval, and sample generation.

Runs on CPU/MPS/CUDA. ``PP_SMOKE=1`` forces a tiny config with synthetic data so
local runs never touch Kaggle or the real corpus::

    PP_SMOKE=1 .venv/bin/python -c "from research_hybrid.train import run; run()"

The Kaggle entry point is training/kaggle/kaggle_pipeline_v2.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research_hybrid.config import CurriculumStage, ModelConfig, TrainingConfig
from research_hybrid.model import EMAWrapper, HybridLM
from research_hybrid.optim import make_optimizer

CHECKPOINT_DATASET = "tomiokasan/research-moe-checkpoints"


class TokenMemmap:
    """Random-access slices over pretokenized tokens.bin (+ metadata.json)."""

    def __init__(self, tokens_path: str, metadata_path: str, val_frac: float = 0.05,
                 seed: int = 1337, vocab: Optional[int] = None):
        with open(metadata_path) as f:
            self.meta = json.load(f)
        self.dtype = np.dtype(self.meta["dtype"])
        self.arr = np.memmap(tokens_path, dtype=self.dtype, mode="r")
        self.n_tokens = int(len(self.arr))
        self.n_val = int(self.n_tokens * val_frac)
        self.n_train = self.n_tokens - self.n_val
        self.val_start = self.n_train
        self.rng = np.random.default_rng(seed)
        self.vocab = vocab or int(self.meta.get("vocab_size", 32000))

    def sample_batch(self, T: int, B: int, train: bool = True) -> torch.Tensor:
        """B random contiguous sequences of length T. Returns (B, T) long tensor."""
        base = 0 if train else self.val_start
        n = self.n_train if train else self.n_val
        high = max(n - T, 1)
        starts = base + self.rng.integers(0, high, size=B)
        rows = [np.asarray(self.arr[s:s + T], dtype=np.int64) for s in starts]
        arr = np.stack(rows) % self.vocab
        return torch.from_numpy(arr)

    def eval_blocks(self, T: int, B: int, max_blocks: int):
        """Contiguous val blocks (no overlap), generator of (B, T) batches."""
        n_blocks = min(max_blocks, self.n_val // (T * B))
        if n_blocks == 0:
            return
        for i in range(n_blocks):
            start = self.val_start + i * T * B
            arr = (np.asarray(self.arr[start:start + T * B], dtype=np.int64)
                   .reshape(B, T) % self.vocab)
            yield torch.from_numpy(arr)


def make_synthetic_data(working: Path, n_tokens: int = 100_000, vocab: int = 32768,
                        seed: int = 1337) -> None:
    """Synthetic random tokens.bin + metadata.json for PP_SMOKE runs."""
    out = Path(working) / "tokenized"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tok = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16 if vocab <= 65535 else np.uint32)
    tok.tofile(str(out / "tokens.bin"))
    with open(out / "metadata.json", "w") as f:
        json.dump({"documents": 5000, "tokens": int(n_tokens), "vocab_size": vocab,
                   "dtype": str(tok.dtype), "eos_id": 2, "binary_file": "tokens.bin"},
                  f, indent=2)


def set_attention_mode(model: HybridLM, pattern: str, window: int, anchor: int) -> None:
    """Switch the attention mask mode in place (all blocks share one AttentionConfig)."""
    model.cfg.attention.pattern = pattern
    model.cfg.attention.window_size = window
    model.cfg.attention.anchor_size = anchor
    for blk in model.blocks:
        if blk.attn is not None:
            blk.attn.cfg.pattern = pattern
            blk.attn.cfg.window_size = window
            blk.attn.cfg.anchor_size = anchor


def lr_mult(step: int, tc: TrainingConfig) -> float:
    """Warmup-then-cosine multiplier over the schedule."""
    if tc.warmup_steps > 0 and step < tc.warmup_steps:
        return (step + 1) / tc.warmup_steps
    if tc.total_steps <= tc.warmup_steps:
        return 1.0
    p = (step - tc.warmup_steps) / (tc.total_steps - tc.warmup_steps)
    floor = (tc.adamw_lr_min / tc.adamw_lr) if tc.optimizer == "adamw" else (tc.lr_min / tc.lr)
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def chunked_accuracy(model: HybridLM, h: torch.Tensor, labels: torch.Tensor,
                     chunk: int = 1024) -> Tuple[int, int]:
    """Next-token accuracy (argmax) via chunked final projection.

    The full (B, T, V) logits tensor is ~4.3 GB fp16 at production batch/ctx,
    so accuracy is computed chunk-wise over the hidden states (memory-bounded).
    Returns (correct, total) with token 0 (unk/pad filler) excluded.
    """
    B, T, _ = h.shape
    lbl = labels[:, 1:]
    correct, total = 0, 0
    for c0 in range(0, T - 1, chunk):
        c1 = min(c0 + chunk, T - 1)
        lg = F.linear(h[:, c0:c1], model.embed.weight)
        preds = lg.argmax(dim=-1)
        m = lbl[:, c0:c1] != 0
        correct += (preds == lbl[:, c0:c1]).masked_select(m).sum().item()
        total += m.sum().item()
    return correct, total


@torch.no_grad()
def evaluate(model: HybridLM, data: TokenMemmap, device, ema: Optional[EMAWrapper],
             T: int = 8192, B: int = 4, max_blocks: int = 50, pattern: str = "block_sparse",
             window: int = 8192, anchor: int = 128) -> Dict[str, float]:
    """Val loss / PPL / next-token accuracy over contiguous val blocks.

    Evaluated with the LIVE weights: with decay 0.999 the EMA shadow has a
    ~1000-step memory, so during the first ~2000 steps of a 4,870-step run the
    EMA is dominated by the random init (measured: val loss inflated 7.7 -> 19.5
    at step 500). ``ema`` is kept in the signature for API compatibility only.
    """
    model.eval()
    total_loss, total_tok, correct, blocks = 0.0, 0, 0, 0
    try:
        set_attention_mode(model, pattern, window, anchor)
        for x in data.eval_blocks(T, B, max_blocks):
            x = x.to(device)
            out = model(x, labels=x, training=False)
            c, t = chunked_accuracy(model, out.hidden, x, chunk=1024)
            correct += c
            total_tok += t
            if out.loss is not None:
                total_loss += out.loss.item() * t
            blocks += 1
            if blocks >= max_blocks:
                break
    finally:
        model.train()
    if total_tok == 0:
        return {"val_loss": float("inf"), "val_perplexity": float("inf"), "val_accuracy": 0.0}
    avg = total_loss / total_tok
    return {"val_loss": avg, "val_perplexity": math.exp(min(avg, 20.0)),
            "val_accuracy": correct / total_tok}


@torch.no_grad()
def generate(model: HybridLM, tokenizer, prompts, max_new_tokens: int = 200,
             temperature: float = 0.8, device=torch.device("cpu")) -> list:
    """Decode continuations token-by-token with the rolling-window KV cache."""
    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokenizer, "bos_id"):
            ids = [tokenizer.bos_id()] + ids
        seq = torch.tensor([ids], dtype=torch.long, device=device)
        kvs = None
        for _ in range(max_new_tokens):
            out = model(seq, use_cache=True, past_kvs=kvs, training=False)
            kvs = out.kv_cache
            lg = out.logits[:, -1, :] / temperature
            lg = torch.nan_to_num(lg)
            idx = torch.multinomial(torch.softmax(lg, dim=-1), num_samples=1)
            seq = idx
            ids = ids + [int(idx.item())]
        results.append(tokenizer.decode(ids))
    return results


def _checkpoint_state(model, optimizer, ema, step, total_tokens, best_val_loss,
                      model_cfg, train_cfg) -> dict:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "total_tokens": total_tokens,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path + ".tmp")
    if os.path.exists(path):
        os.remove(path)
    os.rename(path + ".tmp", path)


def load_checkpoint(path: str, model, optimizer, ema=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return {"step": ckpt.get("step", 0), "total_tokens": ckpt.get("total_tokens", 0),
            "best_val_loss": ckpt.get("best_val_loss", float("inf"))}


def _move_optimizer_to(opt, device) -> None:
    """Move optimizer state tensors to ``device`` (torch optimizers keep loaded
    state on the loading device). Also handles MuonClip's custom buffers."""
    if hasattr(opt, "momentum_bufs"):
        for k, v in opt.momentum_bufs.items():
            opt.momentum_bufs[k] = v.to(device)
        for st in opt.adam_state.values():
            for key, val in st.items():
                if isinstance(val, torch.Tensor):
                    st[key] = val.to(device)
    else:
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p)
                if st:
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(p.device)


def _move_ema_to(ema: Optional[EMAWrapper], device) -> None:
    if ema is not None:
        for k, v in ema.shadow.items():
            ema.shadow[k] = v.to(device)


# ---------------- Kaggle dataset upload/resume (mirrors the v1 pipeline) ----------------

_KAGGLE_API = None
KAGGLE_AVAILABLE = False
upload_queue = queue.Queue()


def _patch_upload_file_name():
    """kaggle 2.2.4's _upload_file never sets UploadFile.description (the
    dataset-side file name); the server then rejects dataset creation with
    "Dataset url's dataset slugs and hashlink are all null". Set it to the
    local file name (same monkeypatch as tools/push_code_dataset.py)."""
    try:
        from kaggle.api import kaggle_api_extended as K
        orig = K.KaggleApi._upload_file

        def patched(self, file_name, full_path, blob_type, upload_context, quiet,
                    resources, content_type=None):
            uf = orig(self, file_name, full_path, blob_type, upload_context, quiet,
                      resources, content_type)
            if uf is not None and not getattr(uf, "description", None):
                uf.description = file_name
            return uf

        K.KaggleApi._upload_file = patched
    except Exception:
        pass


def _ensure_kaggle():
    global _KAGGLE_API, KAGGLE_AVAILABLE
    if KAGGLE_AVAILABLE:
        return True
    if os.environ.get("PP_SMOKE"):
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        _patch_upload_file_name()
        api = KaggleApi()
        api.authenticate()
        _KAGGLE_API = api
        KAGGLE_AVAILABLE = True
        return True
    except Exception as e:
        print(f"  [kaggle] auth failed: {type(e).__name__}")
        return False


def download_remote_checkpoint(ckpt_dir: Path) -> None:
    if not _ensure_kaggle():
        return
    try:
        dl = ckpt_dir / "ckpt_download"
        dl.mkdir(parents=True, exist_ok=True)
        _KAGGLE_API.dataset_download_files(CHECKPOINT_DATASET, path=str(dl),
                                           unzip=True, quiet=True)
        src = dl / "resume.pt"
        if src.exists():
            shutil.copyfile(str(src), str(ckpt_dir / "resume.pt"))
            print(f"  Downloaded remote checkpoint -> resume.pt ({src.stat().st_size/1e6:.0f}MB)")
        shutil.rmtree(dl, ignore_errors=True)
    except Exception as e:
        print(f"  No remote checkpoint to resume from: {type(e).__name__}: {e}")


def _enqueue_upload(ckpt_path: str, step: int, upload_dir: Path) -> None:
    if not _ensure_kaggle():
        return
    try:
        stage = upload_dir / f"upload_{step}"
        stage.mkdir(parents=True, exist_ok=True)
        target = stage / "resume.pt"
        if target.exists():
            target.unlink()
        try:
            os.link(ckpt_path, target)
        except OSError:
            shutil.copyfile(ckpt_path, target)
        with open(stage / "dataset-metadata.json", "w") as f:
            json.dump({"id": CHECKPOINT_DATASET, "title": "research-moe-checkpoints",
                       "licenses": [{"name": "other"}]}, f, indent=2)
        with open(stage / "latest_step.txt", "w") as f:
            f.write(f"{step}\n")
        upload_queue.put(stage)
        print(f"  [upload] queued {stage.name}")
    except Exception as e:
        print(f"  [upload] queue failed: {type(e).__name__}: {e}")


def _upload_worker() -> None:
    while True:
        stage = upload_queue.get()
        try:
            if stage is None:
                return
            if not _ensure_kaggle():
                print(f"  [upload] {stage.name} skipped: kaggle API unavailable")
                return
            for attempt in range(1, 4):
                try:
                    t0 = time.time()
                    # The dataset may not exist yet (first save): dataset_create_version
                    # 403s on a non-existent slug, so create it first and fall back to
                    # versioning when it already exists ("already in use" error).
                    r = _KAGGLE_API.dataset_create_new(
                        str(stage), quiet=True, dir_mode="skip", convert_to_csv=False)
                    err = getattr(r, "error", None)
                    if err and "already in use" in err:
                        r = _KAGGLE_API.dataset_create_version(
                            str(stage), version_notes=f"step {stage.name.split('_')[-1]}",
                            quiet=True, convert_to_csv=False, delete_old_versions=True)
                        err = getattr(r, "error", None)
                    if err:
                        raise RuntimeError(err)
                    print(f"  [upload] {stage.name} pushed in {time.time()-t0:.0f}s")
                    break
                except Exception as e:
                    print(f"  [upload] {stage.name} attempt {attempt}/3 failed: "
                          f"{type(e).__name__}: {e}")
                    time.sleep([15, 60, 300][attempt - 1])
        finally:
            upload_queue.task_done()
            shutil.rmtree(stage, ignore_errors=True)


def run(model_cfg: Optional[ModelConfig] = None, train_cfg: Optional[TrainingConfig] = None,
        working: str = ".", tokens_path: Optional[str] = None,
        metadata_path: Optional[str] = None, checkpoint_dir: str = "checkpoints",
        upload_dir: str = "checkpoint_upload", log_dir: str = "experiments",
        prompts: Optional[list] = None, max_new_tokens: int = 200,
        temperature: float = 0.8) -> dict:
    """Train HybridLM per the v2 recipe. Returns a final report dict."""
    # See kaggle_pipeline_v2.py: transient forward peaks fragment the CUDA
    # caching allocator; expandable segments let large loss-chunk allocations
    # reuse the pool instead of OOM'ing. Set before any CUDA allocation.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    smoke = bool(os.environ.get("PP_SMOKE"))
    tc = train_cfg or TrainingConfig()
    if smoke:
        tc = TrainingConfig(
            optimizer=tc.optimizer,
            total_steps=int(os.environ.get("PP_SMOKE_STEPS", "12")),
            warmup_steps=2,
            lr=tc.lr, lr_1d=tc.lr_1d,
            curriculum=[
                CurriculumStage(context_len=256, fraction=0.5, attention_pattern="causal"),
                CurriculumStage(context_len=512, fraction=0.5, attention_pattern="block_sparse"),
            ])
        model_cfg = model_cfg or ModelConfig(
            vocab_size=256, d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2,
            head_dim=16, context_len=1024, use_gradient_checkpointing=False)
    mc = model_cfg or ModelConfig()
    if mc.qk_clip_tau is None and tc.qk_clip_tau is not None:
        mc.qk_clip_tau = tc.qk_clip_tau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"  Device: {device} ({gpu_name})")

    torch.manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tc.seed)

    model = HybridLM(mc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M total, active {model.num_parameters(active=True)/1e6:.1f}M")

    opt = make_optimizer(model, tc)
    ema = EMAWrapper(model, decay=mc.ema_decay) if mc.use_ema else None
    print(f"  Optimizer: {tc.optimizer} (lr {tc.lr}/{tc.lr_1d}, wd {tc.weight_decay}, "
          f"QK-Clip tau {tc.qk_clip_tau}) | EMA: {ema is not None} | "
          f"fp16 autocast: {device.type == 'cuda'}")

    working_p = Path(working)
    if tokens_path is None:
        tokens_path = str(working_p / "tokenized" / "tokens.bin")
    if metadata_path is None:
        metadata_path = str(working_p / "tokenized" / "metadata.json")
    if smoke and not Path(tokens_path).exists():
        make_synthetic_data(working_p, vocab=mc.vocab_size)
    data = TokenMemmap(tokens_path, metadata_path, seed=tc.seed, vocab=mc.vocab_size)
    print(f"  Data: {data.n_tokens:,} tokens (train {data.n_train:,} / val {data.n_val:,})")

    ckpt_dir = working_p / checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(working, upload_dir).mkdir(parents=True, exist_ok=True)
    log_dir_p = working_p / log_dir
    log_dir_p.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_upload_worker, daemon=True).start()

    global_step, total_tokens, best_val_loss = 0, 0, float("inf")
    download_remote_checkpoint(ckpt_dir)
    candidates = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if (ckpt_dir / "resume.pt").exists():
        candidates.append(ckpt_dir / "resume.pt")

    def _ckpt_step(p: Path) -> int:
        if p.name == "resume.pt":
            return 10 ** 9
        return int(p.name.replace("step_", "").replace(".pt", "")) if p.name.startswith("step_") else -1

    for f in sorted(candidates, key=_ckpt_step, reverse=True):
        try:
            r = load_checkpoint(str(f), model, opt, ema, device="cpu")
            global_step, total_tokens, best_val_loss = r["step"], r["total_tokens"], r["best_val_loss"]
            model.to(device)
            _move_optimizer_to(opt, device)
            _move_ema_to(ema, device)
            print(f"  Resumed from {f.name} at step {global_step}")
            break
        except Exception as e:
            print(f"  Failed to load {f.name}: {type(e).__name__}: {e}")
            continue
    resumed_tokens = total_tokens

    save_every = int(os.environ.get("PP_SAVE_EVERY", "500"))
    eval_every = int(os.environ.get("PP_EVAL_EVERY", "500"))
    log_every = int(os.environ.get("PP_LOG_EVERY", "25"))
    grad_clip = tc.grad_clip

    print(f"\n{'='*60}\n  TRAINING from step {global_step} to {tc.total_steps}")
    print(f"  Batch tokens: {tc.batch_tokens} | grad clip: {grad_clip} | "
          f"checkpointing: {mc.use_gradient_checkpointing} | grad accum: {tc.grad_accum}")
    print(f"{'='*60}\n")

    model.train()
    train_start = time.time()
    step_loss, step_acc_num, step_acc_den = 0.0, 0.0, 0.0
    step_bal, step_z = 0.0, 0.0
    csv_path = log_dir_p / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "accuracy", "lr", "tokens", "tok_per_sec",
                         "gpu_mb", "ctx", "balance", "z", "note"])

    while global_step < tc.total_steps:
        stage = tc.stage_for_step(global_step)
        T = stage.context_len
        B = max(1, tc.batch_tokens // T)
        set_attention_mode(model, stage.attention_pattern,
                           mc.attention.window_size, mc.attention.anchor_size)
        x = data.sample_batch(T, B, train=True).to(device)
        total_tokens += x.numel()

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x, labels=x, training=True)
            loss = out.loss
        if loss is None or not torch.isfinite(loss):
            print(f"  !! non-finite loss at step {global_step}; skipping step")
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        m = lr_mult(global_step, tc)
        if tc.optimizer == "muon_clip":
            opt.param_groups[0]["muon_lr"] = tc.lr * m
            opt.defaults["adamw_lr"] = tc.lr_1d * m
        else:
            opt.param_groups[0]["lr"] = tc.adamw_lr * m
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        global_step += 1
        step_loss += loss.item()
        step_bal += out.aux.get("balance", torch.zeros(()).to(loss.device)).item()
        step_z += out.aux.get("z", torch.zeros(()).to(loss.device)).item()

        with torch.no_grad():
            c, t = chunked_accuracy(model, out.hidden, x, chunk=1024)
            step_acc_num += c
            step_acc_den += t

        if global_step % log_every == 0:
            avg_loss = step_loss / log_every
            acc = step_acc_num / max(step_acc_den, 1)
            lr_now = (tc.lr if tc.optimizer == "muon_clip" else tc.adamw_lr) * m
            elapsed = time.time() - train_start
            tok_s = (total_tokens - resumed_tokens) / max(elapsed, 1)
            gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            print(f"  Step {global_step:>6d} | tok {total_tokens/1e6:>8.1f}M | loss: {avg_loss:.4f} "
                  f"| acc: {acc:.4f} | lr: {lr_now:.2e} | bal: {step_bal/log_every:.3f} "
                  f"| z: {step_z/log_every:.3f} | tok/s: {tok_s:.0f} | gpu: {gpu_mem:.0f}MB "
                  f"| ctx {T} {stage.attention_pattern}")
            csv_writer.writerow([global_step, f"{avg_loss:.6f}", f"{acc:.6f}", f"{lr_now:.8f}",
                                 total_tokens, f"{tok_s:.1f}", f"{gpu_mem:.1f}", T,
                                 f"{step_bal/log_every:.6f}", f"{step_z/log_every:.6f}", ""])
            csv_file.flush()
            step_loss, step_acc_num, step_acc_den = 0.0, 0.0, 0.0
            step_bal, step_z = 0.0, 0.0

        if global_step % save_every == 0:
            path = str(ckpt_dir / f"step_{global_step}.pt")
            save_checkpoint(path, _checkpoint_state(model, opt, ema, global_step, total_tokens,
                                                    best_val_loss, mc, tc))
            old = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
            for f in old[:-2]:
                f.unlink()
            _enqueue_upload(path, global_step, working_p / upload_dir)
            du = shutil.disk_usage(str(working_p))
            print(f"  [disk] {du.used/1e9:.1f}GB used / {du.total/1e9:.1f}GB")

        if global_step % eval_every == 0:
            print(f"\n  --- Eval at step {global_step} ---")
            ev = evaluate(model, data, device, ema, T=8192, B=4, max_blocks=50,
                          pattern=mc.attention.pattern, window=mc.attention.window_size,
                          anchor=mc.attention.anchor_size)
            print(f"  Val loss: {ev['val_loss']:.4f} | PPL: {ev['val_perplexity']:.2f} "
                  f"| Acc: {ev['val_accuracy']:.4f}")
            csv_writer.writerow([global_step, f"{ev['val_loss']:.6f}", f"{ev['val_accuracy']:.6f}",
                                 "", total_tokens, "", "", "", "eval"])
            csv_file.flush()
            if ev["val_loss"] < best_val_loss:
                best_val_loss = ev["val_loss"]
                save_checkpoint(str(ckpt_dir / "best.pt"),
                                _checkpoint_state(model, opt, ema, global_step, total_tokens,
                                                  best_val_loss, mc, tc))
                print(f"    New best: {best_val_loss:.4f}")

    final_path = str(ckpt_dir / "final.pt")
    save_checkpoint(final_path, _checkpoint_state(model, opt, ema, global_step, total_tokens,
                                                  best_val_loss, mc, tc))
    _enqueue_upload(final_path, global_step, working_p / upload_dir)

    if prompts:
        print("\n  --- Sample Generation ---")
        try:
            import sentencepiece as spm
            for cand in (working_p / "tokenized" / "tokenizer.model",
                         working_p.parent / "tokenized" / "tokenizer.model"):
                if cand.exists():
                    tok_path = str(cand)
                    break
            else:
                raise FileNotFoundError("tokenizer.model not found")
            sp = spm.SentencePieceProcessor(model_file=tok_path)

            class Tok:
                def __init__(self, sp):
                    self.sp = sp
                def encode(self, t, add_special_tokens=False):
                    return self.sp.EncodeAsIds(t)
                def decode(self, ids):
                    return self.sp.DecodeIds(ids)
                def bos_id(self):
                    return self.sp.bos_id()

            samples = generate(model, Tok(sp), prompts, max_new_tokens, temperature, device)
            for i, (p, s) in enumerate(zip(prompts, samples)):
                print(f"\n  Prompt {i+1}: {p}\n  Output: {s[:200]}...")
        except Exception as e:
            print(f"  Sample generation skipped: {type(e).__name__}: {e}")

    if KAGGLE_AVAILABLE:
        print("  Waiting for pending checkpoint uploads...")
        deadline = time.time() + 1800
        while upload_queue.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(10)

    elapsed = time.time() - train_start
    report = {"total_steps": global_step, "total_tokens": total_tokens,
              "best_val_loss": best_val_loss, "elapsed_hours": elapsed / 3600,
              "tokens_per_sec": (total_tokens - resumed_tokens) / max(elapsed, 1),
              "gpu": gpu_name}
    with open(log_dir_p / "kaggle_report.json", "w") as f:
        json.dump(report, f, indent=2)
    csv_file.close()

    print(f"\n{'='*60}\n  TRAINING COMPLETE")
    print(f"  Steps: {global_step} | Tokens: {(total_tokens-resumed_tokens)/1e6:.1f}M")
    print(f"  Time: {elapsed/3600:.2f}h | Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    return report


if __name__ == "__main__":
    run()
