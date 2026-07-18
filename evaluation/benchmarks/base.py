"""Base classes for evaluation benchmarks."""

import time
import torch
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    eval_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                 for k, v in self.metrics.items())
        return f"{self.name}: {metrics_str} ({self.num_samples} samples, {self.eval_time:.1f}s)"


@dataclass
class EvalResults:
    checkpoint_path: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    generation_samples: List[Dict[str, str]] = field(default_factory=list)
    efficiency: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def summary_table(self) -> str:
        lines = []
        lines.append(f"{'Benchmark':<25} {'Metric':<25} {'Value':>12}")
        lines.append("-" * 64)
        for name, result in self.benchmarks.items():
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{name:<25} {metric:<25} {value:>12.4f}")
                else:
                    lines.append(f"{name:<25} {metric:<25} {str(value):>12}")
        if self.efficiency:
            lines.append("-" * 64)
            for metric, value in self.efficiency.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{'efficiency':<25} {metric:<25} {value:>12.4f}")
                else:
                    lines.append(f"{'efficiency':<25} {metric:<25} {str(value):>12}")
        return "\n".join(lines)


def load_dataset_split(dataset_name: str, split: str, **kwargs):
    """Load a HuggingFace dataset split, auto-downloading if needed."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, **kwargs)
    return ds


@torch.no_grad()
def compute_loss_on_tokens(model, input_ids: torch.Tensor, labels: torch.Tensor,
                           device: torch.device, pad_token_id: int = 0) -> float:
    """Compute average cross-entropy loss on the given token sequence."""
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    _, loss = model(input_ids=input_ids, labels=labels)
    return loss.item()


@torch.no_grad()
def compute_chunked_loss(model, token_ids: List[int], max_seq_len: int,
                         device: torch.device, pad_token_id: int = 0,
                         stride: Optional[int] = None) -> Dict[str, float]:
    """Compute perplexity over a long sequence using sliding window.

    Returns dict with 'loss', 'perplexity', 'num_tokens', 'bpb' (bits-per-byte).
    """
    model.eval()
    if stride is None:
        stride = max_seq_len

    total_loss = 0.0
    total_tokens = 0

    for begin_loc in range(0, len(token_ids), stride):
        end_loc = min(begin_loc + max_seq_len, len(token_ids))
        trg_len = end_loc - begin_loc
        if trg_len <= 1:
            continue

        input_chunk = torch.tensor(token_ids[begin_loc:end_loc], dtype=torch.long, device=device)
        target_chunk = input_chunk.clone()
        target_chunk[:-trg_len] = -100

        _, loss = model(input_ids=input_chunk.unsqueeze(0), labels=target_chunk.unsqueeze(0))
        total_loss += loss.item() * trg_len
        total_tokens += trg_len

        if end_loc >= len(token_ids):
            break

    if total_tokens == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "num_tokens": 0, "bpb": float("inf")}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return {"loss": avg_loss, "perplexity": perplexity, "num_tokens": total_tokens, "bpb": 0.0}


class Benchmark(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def evaluate(self, model, tokenizer, device: torch.device) -> BenchmarkResult:
        pass

    def _encode(self, tokenizer, text: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            result = tokenizer.encode(text)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                return result[0]
            return result
        return tokenizer(text)["input_ids"]
