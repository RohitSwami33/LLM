"""Evaluation utilities for language model training.

Computes:
    - Validation loss
    - Perplexity
    - Per-token accuracy
"""

import torch
import math
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: Optional[int] = None,
    pad_token_id: int = 0,
) -> Dict[str, float]:
    """Evaluate model on validation data.

    Args:
        model: The language model.
        dataloader: Validation dataloader.
        device: Compute device.
        max_steps: Maximum evaluation steps (None = full dataset).
        pad_token_id: Padding token ID to ignore in loss.

    Returns:
        Dict with 'val_loss', 'val_perplexity', 'val_perplexity_exp',
        'val_accuracy', 'val_tokens', 'val_steps'.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    num_steps = 0

    for i, batch in enumerate(dataloader):
        if max_steps and i >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Create attention mask (1 for non-pad tokens)
        attention_mask = (input_ids != pad_token_id).long()

        # Forward
        logits, loss = model(input_ids=input_ids, labels=labels)

        # Compute per-token accuracy (excluding padding and -100 labels)
        with torch.no_grad():
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            predictions = shift_logits.argmax(dim=-1)
            mask = (shift_labels != -100) & (shift_labels != pad_token_id)
            correct_tokens += (predictions == shift_labels).float().masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

        # Accumulate loss (weighted by non-pad tokens)
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens > 0:
            total_loss += loss.item() * valid_tokens
        num_steps += 1

    if num_steps == 0 or total_tokens == 0:
        return {
            "val_loss": float("inf"),
            "val_perplexity": float("inf"),
            "val_perplexity_exp": float("inf"),
            "val_accuracy": 0.0,
            "val_tokens": 0,
            "val_steps": 0,
        }

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    accuracy = correct_tokens / total_tokens

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_perplexity_exp": perplexity,
        "val_accuracy": accuracy,
        "val_tokens": total_tokens,
        "val_steps": num_steps,
    }


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20))


@torch.no_grad()
def compute_calibration(model, dataloader, device, n_bins=20):
    """Compute model calibration metrics."""
    model.eval()
    confidences = []
    accuracies = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids=input_ids, labels=labels)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        probs = torch.softmax(shift_logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)

        mask = (shift_labels != -100) & (shift_labels != 0)
        confidences.extend(max_probs[mask].cpu().tolist())
        accuracies.extend((preds[mask] == shift_labels[mask]).cpu().float().tolist())

    if not confidences:
        return {"expected_calibration_error": 0.0}

    confidences = sorted(zip(confidences, accuracies), key=lambda x: x[0])
    bin_size = len(confidences) // n_bins

    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(confidences)
        if start >= end:
            continue
        bin_conf = sum(c for c, _ in confidences[start:end]) / (end - start)
        bin_acc = sum(a for _, a in confidences[start:end]) / (end - start)
        ece += (end - start) / len(confidences) * abs(bin_acc - bin_conf)

    return {"expected_calibration_error": ece}
