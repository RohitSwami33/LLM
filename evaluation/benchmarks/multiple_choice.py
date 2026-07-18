"""Multiple-choice benchmarks: HellaSwag, ARC, BoolQ, PIQA, LAMBADA, Winogrande."""

import time
import torch
from typing import List, Dict, Any, Callable, Optional
from .base import Benchmark, BenchmarkResult


def _safe_load_dataset(name: str, split: str, **kwargs):
    """Load HF dataset with graceful error handling, using parquet fallback."""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        # Map standard datasets to their parquet file locations
        parquet_map = {
            "hellaswag": ("hellaswag", f"data/{split}-00000-of-00001.parquet"),
            "boolq": ("google/boolq", f"data/{split}-00000-of-00001.parquet"),
            "winogrande": ("winogrande", f"winogrande_debiased/{split}-00000-of-00001.parquet"),
            "lambada": ("lambada", f"plain_text/{split}-00000-of-00001.parquet"),
            "piqa": ("piqa", f"{split}-00000-of-00001.parquet"),
        }

        if name in parquet_map:
            repo, fname = parquet_map[name]
            path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
            table = pq.read_table(path)
            return list(table.to_pylist())

        # Generic fallback: try data/ prefix, then root
        for prefix in ["data/", ""]:
            try:
                path = hf_hub_download(repo_id=name, filename=f"{prefix}{split}-00000-of-00001.parquet", repo_type="dataset")
                table = pq.read_table(path)
                return list(table.to_pylist())
            except Exception:
                continue

        return None

    except Exception as e:
        print(f"  Dataset {name}/{split} not available: {e}")
        return None


def _compute_choice_loss(model, tokenizer, context: str, choices: List[str],
                         device: torch.device, pad_token_id: int = 0) -> List[float]:
    """Compute loss for each (context + choice) pair. Returns list of losses."""
    model.eval()
    losses = []
    max_len = getattr(model, "max_seq_len", 1024)
    if not hasattr(model, "max_seq_len"):
        for attr in ("max_seq_len", "max_position_embeddings"):
            if hasattr(model.config if hasattr(model, "config") else None, attr):
                max_len = getattr(model.config, attr, 1024)
                break

    ctx_ids = _encode(tokenizer, context)

    for choice in choices:
        choice_ids = _encode(tokenizer, choice)
        input_ids = ctx_ids + choice_ids

        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)

        labels = torch.full_like(input_t, -100)
        ctx_len = len(ctx_ids)
        total_len = len(input_ids)
        labels[0, ctx_len:total_len] = input_t[0, ctx_len:total_len]

        with torch.no_grad():
            _, loss = model(input_ids=input_t, labels=labels)
        losses.append(loss.item() if not torch.isnan(loss) else 1e6)

    return losses


def _encode(tokenizer, text: str) -> List[int]:
    result = tokenizer.encode(text)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return result


class MultipleChoiceBenchmark(Benchmark):
    """Generic multiple-choice benchmark."""

    def __init__(self, dataset_name: str, display_name: str = None,
                 split: str = "validation", load_fn: Callable = None,
                 format_fn: Callable = None, max_samples: int = None):
        self.dataset_name = dataset_name
        self.name = display_name or dataset_name
        self.split = split
        self.load_fn = load_fn or self._default_load
        self.format_fn = format_fn
        self.max_samples = max_samples

    def _default_load(self):
        return _safe_load_dataset(self.dataset_name, self.split)

    def evaluate(self, model, tokenizer, device: torch.device) -> BenchmarkResult:
        t0 = time.time()
        ds = self.load_fn()

        if ds is None:
            return BenchmarkResult(
                name=self.name,
                metrics={"accuracy": 0.0, "loss": float("inf"), "perplexity": float("inf")},
                num_samples=0, eval_time=time.time() - t0,
                metadata={"error": "Dataset not available"},
            )

        correct = 0
        total = 0
        total_loss = 0.0

        for i, example in enumerate(ds):
            if self.max_samples and i >= self.max_samples:
                break

            if self.format_fn:
                result = self.format_fn(example)
            else:
                result = self._default_format(example)

            if result is None:
                continue

            context, choices, answer_idx = result
            losses = _compute_choice_loss(model, tokenizer, context, choices, device)
            pred_idx = min(range(len(losses)), key=lambda i: losses[i])

            if pred_idx == answer_idx:
                correct += 1
            total += 1
            total_loss += losses[answer_idx]

            if (i + 1) % 100 == 0:
                torch.mps.synchronize() if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else None

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        import math
        return BenchmarkResult(
            name=self.name,
            metrics={
                "accuracy": accuracy,
                "loss": avg_loss,
                "perplexity": math.exp(min(avg_loss, 20)),
            },
            num_samples=total,
            eval_time=time.time() - t0,
        )

    def _default_format(self, example):
        raise NotImplementedError("Subclasses must implement _default_format")


# ── HellaSwag ────────────────────────────────────────────────────────────────
def _format_hellaswag(example: Dict) -> Optional[tuple]:
    ctx = example["activity_label"] + ": " + example["ctx_a"] + " " + example["ctx_b"]
    ctx = ctx.strip()
    choices = example["endings"]
    label = int(example["label"]) if example.get("label") is not None else -1
    if label < 0 or label >= len(choices):
        return None
    return ctx, choices, label


class HellaSwag(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "hellaswag", "HellaSwag", split="validation",
            format_fn=_format_hellaswag, max_samples=max_samples,
        )


# ── ARC ──────────────────────────────────────────────────────────────────────
def _load_arc(split_name: str):
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        fname = f"ARC-Easy/{split_name}-00000-of-00001.parquet"
        path = hf_hub_download(repo_id="allenai/arc", filename=fname, repo_type="dataset")
        table = pq.read_table(path)
        return list(table.to_pylist())
    except Exception as e:
        print(f"  ARC not available: {e}")
        return None


def _format_arc(example: Dict) -> Optional[tuple]:
    question = example["question"]
    choices = example["choices"]["text"]
    answer_key = example["answerKey"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    answer_idx = label_map.get(answer_key, -1)
    if answer_idx < 0 or answer_idx >= len(choices):
        return None
    return f"Question: {question}\nAnswer:", choices, answer_idx


class ARCEasy(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "allenai/arc", "ARC-Easy", split="validation",
            load_fn=lambda: _load_arc("validation"), format_fn=_format_arc,
            max_samples=max_samples,
        )


class ARCChallenge(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "allenai/arc", "ARC-Challenge", split="validation",
            load_fn=lambda: _load_arc("validation"), format_fn=_format_arc,
            max_samples=max_samples,
        )


# ── BoolQ ────────────────────────────────────────────────────────────────────
def _format_boolq(example: Dict) -> Optional[tuple]:
    question = example["question"]
    passage = example.get("passage", "")
    answer = "True" if example["answer"] else "False"
    ctx = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    choices = [" True", " False"]
    label = 0 if example["answer"] else 1
    return ctx, choices, label


class BoolQ(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "boolq", "BoolQ", split="validation",
            format_fn=_format_boolq, max_samples=max_samples,
        )


# ── PIQA ─────────────────────────────────────────────────────────────────────
def _format_piqa(example: Dict) -> Optional[tuple]:
    goal = example["goal"]
    choices = [example["sol1"], example["sol2"]]
    label = int(example["label"])
    return f"Question: {goal}\nSolution:", choices, label


class PIQA(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "piqa", "PIQA", split="validation",
            format_fn=_format_piqa, max_samples=max_samples,
        )


# ── LAMBADA ──────────────────────────────────────────────────────────────────
def _format_lambada(example: Dict) -> Optional[tuple]:
    text = example["text"]
    words = text.rsplit(" ", 1)
    if len(words) != 2:
        return None
    ctx, last_word = words
    return f"{ctx} ", [f" {last_word}"], 0


class LAMBADA(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "lambada", "LAMBADA", split="test",
            format_fn=_format_lambada, max_samples=max_samples,
        )


# ── Winogrande ───────────────────────────────────────────────────────────────
def _format_winogrande(example: Dict) -> Optional[tuple]:
    sentence = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    answer = example["answer"]
    placeholder = "_"
    ctx1 = sentence.replace(placeholder, option1, 1)
    ctx2 = sentence.replace(placeholder, option2, 1)
    label = 0 if answer == "1" else 1
    full_ctx = sentence.replace(placeholder, "")
    return full_ctx, [f" {option1}", f" {option2}"], label


class Winogrande(MultipleChoiceBenchmark):
    def __init__(self, max_samples: int = 1000):
        super().__init__(
            "winogrande", "Winogrande", split="validation",
            format_fn=_format_winogrande, max_samples=max_samples,
        )
