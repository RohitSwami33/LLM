"""Core utilities: config, schema, telemetry, retry, rate limiting.

This package is intentionally dependency-light so the whole framework can be
imported (and unit-tested) without heavy optional dependencies.
"""

from .config import (
    Config,
    TeacherConfig,
    TaskConfig,
    FilterConfig,
    ScoringConfig,
    ExportConfig,
    GenerationConfig,
    load_config,
    dump_config,
)
from .schema import Sample, FilterDecision, ScoreResult, GenerationResult
from .telemetry import Metrics, setup_logger
from .rate_limit import RateLimiter

__all__ = [
    "Config",
    "TeacherConfig",
    "TaskConfig",
    "FilterConfig",
    "ScoringConfig",
    "ExportConfig",
    "GenerationConfig",
    "load_config",
    "dump_config",
    "Sample",
    "FilterDecision",
    "ScoreResult",
    "GenerationResult",
    "Metrics",
    "setup_logger",
    "RateLimiter",
]
