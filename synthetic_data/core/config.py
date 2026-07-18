"""Configuration system.

All behaviour is driven by a single YAML file that is parsed into nested
``dataclass`` objects via :func:`load_config`.  Unknown keys are ignored and
missing keys fall back to dataclass defaults, so partial configs work.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------
# Dataclass definitions
# --------------------------------------------------------------------------

@dataclass
class TeacherConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 5
    rpm_limit: int = 20
    logprobs: bool = False
    # Pricing per 1M tokens (DeepSeek chat, indicative).
    input_price_per_1m: float = 0.27
    output_price_per_1m: float = 1.10


@dataclass
class TaskConfig:
    task_type: str = "instruction"
    template: str = "instruction"
    num_samples: int = 1000
    enabled: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seeds: Optional[List[Dict[str, Any]]] = None


@dataclass
class FilterConfig:
    enabled: bool = True
    min_length: int = 30
    max_length: int = 100_000
    remove_incomplete: bool = True
    remove_repeated_reasoning: bool = True
    repetition_threshold: float = 0.4
    remove_broken_json: bool = True
    remove_hallucinated_formatting: bool = True
    deduplicate: bool = True
    near_dedup: bool = False
    near_dedup_threshold: float = 0.9
    language: Optional[str] = None
    toxicity_threshold: float = 0.5
    max_acceptance_rate: Optional[float] = None


@dataclass
class ScoringConfig:
    enabled: bool = False
    perplexity: bool = False
    perplexity_model: str = "gpt2"
    min_confidence: Optional[float] = None
    min_reasoning_score: Optional[float] = None


@dataclass
class ExportConfig:
    format: str = "jsonl"  # jsonl | parquet | arrow | hf
    output_dir: str = "data/exports"
    dataset_name: Optional[str] = None
    push_to_hub: bool = False
    shard_size: int = 100_000


@dataclass
class GenerationConfig:
    seed: int = 42
    output_dir: str = "data/generated"
    batch_size: int = 10
    resume: bool = True
    log_every: int = 50
    timeout_per_sample: float = 120.0


@dataclass
class Config:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    tasks: List[TaskConfig] = field(default_factory=list)
    filters: FilterConfig = field(default_factory=FilterConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


# --------------------------------------------------------------------------
# YAML -> dataclass coercion
# --------------------------------------------------------------------------

def _strip_optional(t: Any) -> Any:
    """Resolve ``Optional[X]`` / ``Union`` to the concrete inner type."""
    origin = typing.get_origin(t)
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(t) if a is not type(None)]
        return non_none[0] if non_none else t
    return t


def _from_dict(cls: type, data: Optional[Dict[str, Any]]) -> Any:
    """Recursively build a dataclass instance from a dict."""
    if data is None:
        data = {}
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = _strip_optional(hints.get(f.name, f.type))
        origin = typing.get_origin(ftype)

        if origin in (list, typing.List) and isinstance(val, list):
            inner = _strip_optional(typing.get_args(ftype)[0]) if typing.get_args(ftype) else None
            if dataclasses.is_dataclass(inner):
                val = [_from_dict(inner, v) for v in val]
        elif dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            val = _from_dict(ftype, val)
        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(Config, raw)


def dump_config(config: Config, path: str | Path) -> None:
    """Persist (a subset of) the config for reproducibility."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {k: getattr(config, k) for k in vars(config)},
            f,
            sort_keys=False,
        )
