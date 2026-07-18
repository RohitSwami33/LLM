"""Perplexity benchmarks: WikiText-2, WikiText-103."""

import os
import time
import math
import torch
from typing import Optional
from .base import Benchmark, BenchmarkResult, compute_chunked_loss


class PerplexityBenchmark(Benchmark):
    """Evaluate perplexity on WikiText datasets (or any text corpus)."""

    def __init__(self, dataset_name: str = None, split: str = "test",
                 display_name: str = None, field: str = "text",
                 max_seq_len: int = 1024, stride: int = 512,
                 local_path: str = None):
        self.dataset_name = dataset_name
        self.split = split
        self.name = display_name or dataset_name or "perplexity"
        self.field = field
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.local_path = local_path

    def evaluate(self, model, tokenizer, device: torch.device) -> BenchmarkResult:
        t0 = time.time()

        all_text = self._load_text(tokenizer)
        if not all_text:
            return BenchmarkResult(
                name=self.name,
                metrics={"perplexity": float("inf"), "loss": float("inf"), "bits_per_byte": float("inf")},
                num_tokens=0, eval_time=0.0, metadata={"error": "No text data loaded"},
            )

        token_ids = self._encode(tokenizer, all_text)

        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        result = compute_chunked_loss(
            model, token_ids, self.max_seq_len, device,
            pad_token_id=pad_token_id, stride=self.stride,
        )

        nats = result["loss"] * result["num_tokens"]
        num_bytes = len(all_text.encode("utf-8"))
        bpb = nats * math.log(2) / num_bytes if num_bytes > 0 else 0.0

        return BenchmarkResult(
            name=self.name,
            metrics={
                "perplexity": result["perplexity"],
                "loss": result["loss"],
                "bits_per_byte": bpb,
            },
            num_samples=result["num_tokens"],
            eval_time=time.time() - t0,
            metadata={"dataset": self.dataset_name, "split": self.split, "num_tokens": result["num_tokens"]},
        )

    def _load_text(self, tokenizer) -> str:
        """Load text from local file or HuggingFace dataset."""
        # Try local file first
        if self.local_path and os.path.exists(self.local_path):
            print(f"  Loading text from {self.local_path}")
            with open(self.local_path, "r", encoding="utf-8") as f:
                return f.read()

        # Try HuggingFace datasets
        if self.dataset_name:
            return self._load_from_hf()

        return ""

    def _load_from_hf(self) -> str:
        """Load text from HuggingFace datasets via parquet files."""
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq

            config_map = {
                "wikitext-2-raw-v1": ("wikitext", "wikitext-2-raw-v1"),
                "wikitext-103-raw-v1": ("wikitext", "wikitext-103-raw-v1"),
            }

            if self.dataset_name in config_map:
                repo, config = config_map[self.dataset_name]
            else:
                repo = self.dataset_name
                config = self.dataset_name

            # Try to download parquet file
            split_map = {"test": "test", "validation": "validation", "train": "train"}
            hf_split = split_map.get(self.split, self.split)

            path = hf_hub_download(
                repo_id=repo,
                filename=f"{config}/{hf_split}-00000-of-00001.parquet",
                repo_type="dataset",
            )

            table = pq.read_table(path)
            lines = [row[self.field] for row in table.to_pylist() if row.get(self.field)]
            return "\n".join(lines)

        except Exception as e:
            print(f"  HF parquet download failed: {e}")

            # Fallback: try datasets library
            try:
                from datasets import load_dataset
                ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
                lines = [doc.get(self.field, "") for doc in ds if doc.get(self.field)]
                return "\n".join(lines)
            except Exception as e2:
                print(f"  datasets.load also failed: {e2}")

            return ""

class LocalTextBenchmark(PerplexityBenchmark):
    """Evaluate perplexity on a local text file."""

    def __init__(self, file_path: str, display_name: str = None,
                 max_seq_len: int = 1024, stride: int = 512):
        name = display_name or os.path.basename(file_path)
        super().__init__(
            dataset_name=None, split="test", display_name=name,
            max_seq_len=max_seq_len, stride=stride, local_path=file_path,
        )
