"""Comprehensive LLM evaluation framework.

Benchmarks: WikiText-2, WikiText-103, HellaSwag, ARC, BoolQ, PIQA, LAMBADA, Winogrande
Metrics: perplexity, accuracy, bits-per-byte, throughput, memory, latency, FLOPs
Reports: JSON, Markdown, CSV, plots
"""

from .runner import EvalRunner
from .compare import compare_checkpoints

__all__ = ["EvalRunner", "compare_checkpoints"]
