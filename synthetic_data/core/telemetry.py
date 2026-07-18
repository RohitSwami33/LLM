"""Telemetry: structured metrics + logging helpers.

The :class:`Metrics` collector is updated from the generator / filter /
exporter stages and can emit a single JSON-serialisable report used for
logging dashboards and experiment tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict


def setup_logger(name: str = "synthetic", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger configured once for the framework."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@dataclass
class Metrics:
    """Cumulative counters for a generation / filtering run."""

    started_at: float = field(default_factory=time.time)

    # generation
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    written: int = 0

    # tokens / cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    # filtering
    accepted: int = 0
    rejected: int = 0
    duplicates_removed: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)

    # scoring
    scored: int = 0

    # ------------------------------------------------------------------ #
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def speed(self) -> float:
        """Samples completed per second since start."""
        e = self.elapsed()
        return self.completed / e if e > 0 else 0.0

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_rejection(self, reason: str) -> None:
        self.rejected += 1
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1

    def report(self) -> Dict[str, Any]:
        return {
            "elapsed_sec": round(self.elapsed(), 2),
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "written": self.written,
            "samples_per_sec": round(self.speed(), 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens(),
            "cost_usd": round(self.cost_usd, 4),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates_removed": self.duplicates_removed,
            "rejected_reasons": dict(self.rejected_reasons),
            "scored": self.scored,
        }

    def log(self, logger: logging.Logger) -> None:
        r = self.report()
        logger.info(
            "metrics | completed=%d failed=%d written=%d "
            "tok=%d cost=$%.4f speed=%.2f/s accepted=%d rejected=%d dups=%d",
            r["completed"], r["failed"], r["written"], r["total_tokens"],
            r["cost_usd"], r["samples_per_sec"], r["accepted"],
            r["rejected"], r["duplicates_removed"],
        )
