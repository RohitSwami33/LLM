"""Checkpointing utilities for model, optimizer, scheduler, EMA, and training state.

Supports:
    - Full checkpoint save/load (model + optimizer + scheduler + scaler + EMA + RNG + dataloader)
    - Async checkpoint saving
    - Keep-last-N checkpoint management
"""

import os
import json
import time
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    step: int = 0,
    epoch: int = 0,
    rng_state=None,
    extra: Optional[Dict[str, Any]] = None,
    total_tokens: int = 0,
):
    """Save a full training checkpoint.

    Args:
        path: Save path (directory).
        model: The model.
        optimizer: The optimizer.
        scheduler: LR scheduler (optional).
        scaler: GradScaler for AMP (optional).
        ema: EMA model (optional).
        step: Current training step.
        epoch: Current epoch.
        rng_state: RNG state dict (optional).
        extra: Additional state to save.
        total_tokens: Total tokens processed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "timestamp": time.time(),
        "total_tokens": total_tokens,
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if ema is not None:
        state["ema"] = ema.state_dict()

    if rng_state is not None:
        state["rng"] = rng_state
    else:
        state["rng"] = {
            "python": None,
            "numpy": None,
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "torch": torch.random.get_rng_state(),
        }

    # Save MoE-specific state if applicable
    if hasattr(model, "is_moe") and model.is_moe:
        state["moe"] = _extract_moe_state(model)

    if extra is not None:
        state.update(extra)

    # Save to temp path first, then rename (atomic)
    temp_path = path + ".tmp"
    torch.save(state, temp_path)
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.rename(temp_path, path)

    return path


def _extract_moe_state(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract MoE-specific state for checkpointing."""
    moe_state = {
        "config": {
            "num_experts": model.config.moe_num_experts,
            "top_k": model.config.moe_top_k,
            "capacity_factor": model.config.moe_capacity_factor,
            "shared_expert": model.config.moe_shared_expert,
            "load_balancing_weight": model.config.moe_load_balancing_weight,
            "router_z_loss_weight": model.config.moe_router_z_loss_weight,
        },
        "router_weights": {},
        "expert_utilization_history": [],
    }

    # Save router weights from each MoE layer
    for name, module in model.named_modules():
        if hasattr(module, "moe") and hasattr(module.moe, "router"):
            router_key = name.replace(".", "_")
            moe_state["router_weights"][router_key] = {
                "gate_weight": module.moe.router.gate.weight.data.clone(),
            }

    return moe_state


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    ema=None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Returns:
        Dict with step, epoch, and any extra state.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    # Restore RNG states
    if "rng" in checkpoint:
        rng = checkpoint["rng"]
        if rng.get("torch") is not None:
            torch.random.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "total_tokens": checkpoint.get("total_tokens", 0),
    }


def save_checkpoint_async(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    step: int = 0,
    executor: Optional[ThreadPoolExecutor] = None,
):
    """Save checkpoint asynchronously."""
    if executor is None:
        return save_checkpoint(path, model, optimizer, scheduler, scaler, ema, step)

    future = executor.submit(
        save_checkpoint, path, model, optimizer, scheduler, scaler, ema, step
    )
    return future


def clean_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """Remove old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.
        keep_last_n: Number of checkpoints to keep.
    """
    if not os.path.exists(checkpoint_dir):
        return

    checkpoints = []
    for entry in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("step_"):
            try:
                step_num = int(entry.split("_")[1])
                checkpoints.append((step_num, full_path))
            except (ValueError, IndexError):
                continue

    checkpoints.sort(key=lambda x: x[0])

    # Remove old checkpoints
    while len(checkpoints) > keep_last_n:
        step_num, path = checkpoints.pop(0)
        print(f"Removing old checkpoint: step_{step_num}")
        shutil.rmtree(path, ignore_errors=True)


def save_training_state(path: str, step: int, epoch: int, extra: Dict = None):
    """Save lightweight training state (no model/optimizer weights)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"step": step, "epoch": epoch, "timestamp": time.time()}
    if extra:
        state.update(extra)
    with open(os.path.join(path, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def load_training_state(path: str) -> Dict[str, Any]:
    """Load training state."""
    state_path = os.path.join(path, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"step": 0, "epoch": 0}
