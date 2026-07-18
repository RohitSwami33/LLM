"""Scoring package: optional quality signals for generated samples.

Provides a thin :class:`ScorerPipeline` that aggregates configured scorers
and attaches their values to ``sample.metadata["scores"]``.  Scoring is
disabled by default and only runs when ``ScoringConfig.enabled`` is true.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.config import ScoringConfig
from ..core.schema import Sample, ScoreResult
from .perplexity import PerplexityScorer
from .confidence import ConfidenceScorer
from .reasoning_score import ReasoningScorer

__all__ = [
    "PerplexityScorer",
    "ConfidenceScorer",
    "ReasoningScorer",
    "ScorerPipeline",
    "build_scorers",
]


class ScorerPipeline:
    """Runs enabled scorers and decides pass/fail against thresholds."""

    def __init__(self, config: ScoringConfig):
        self.config = config
        self.perplexity = PerplexityScorer(config.perplexity_model) if config.perplexity else None
        self.confidence = ConfidenceScorer()
        self.reasoning = ReasoningScorer()

    async def score(self, sample: Sample) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if self.perplexity is not None:
            scores["perplexity"] = await self.perplexity.ascore(sample)
        conf = self.confidence.score(sample)
        if conf is not None:
            scores["confidence"] = conf
        scores["reasoning"] = self.reasoning.score(sample)
        return scores

    async def evaluate(self, sample: Sample) -> tuple[bool, Dict[str, float]]:
        """Return (keep, scores); applies configured thresholds."""
        scores = await self.score(sample)
        if self.config.min_confidence is not None:
            if scores.get("confidence", 1.0) < self.config.min_confidence:
                return False, scores
        if self.config.min_reasoning_score is not None:
            if scores.get("reasoning", 1.0) < self.config.min_reasoning_score:
                return False, scores
        return True, scores


def build_scorers(config: ScoringConfig) -> Optional[ScorerPipeline]:
    if not config.enabled:
        return None
    return ScorerPipeline(config)
