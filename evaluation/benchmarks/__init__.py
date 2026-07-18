"""Benchmark datasets for LLM evaluation.

Auto-downloads via HuggingFace datasets if missing.
"""

from .base import BenchmarkResult, load_dataset_split
from .perplexity import PerplexityBenchmark
from .multiple_choice import HellaSwag, ARCEasy, ARCChallenge, BoolQ, PIQA, LAMBADA, Winogrande
from .efficiency import EfficiencyBenchmark

ALL_BENCHMARKS = {
    "wikitext2": lambda: PerplexityBenchmark("wikitext-2-raw-v1", "test", "WikiText-2"),
    "wikitext103": lambda: PerplexityBenchmark("wikitext-103-raw-v1", "test", "WikiText-103"),
    "hellaswag": HellaSwag,
    "arc_easy": ARCEasy,
    "arc_challenge": ARCChallenge,
    "boolq": BoolQ,
    "piqa": PIQA,
    "lambada": LAMBADA,
    "winogrande": Winogrande,
}


def get_benchmark(name: str):
    factory = ALL_BENCHMARKS.get(name.lower().replace("-", "").replace("_", ""))
    if factory is None:
        for key, val in ALL_BENCHMARKS.items():
            if name.lower() in key.replace("-", "").replace("_", ""):
                factory = val
                break
    if factory is None:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(ALL_BENCHMARKS.keys())}")
    return factory() if callable(factory) else factory
