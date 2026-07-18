"""Heuristic reasoning-quality scorer.

Scores the *reasoning* field (when present) on structural and consistency
signals.  Returns a value in [0, 1]; the pipeline can use it to reject weak
or inconsistent chains of thought.
"""

from __future__ import annotations

import re
from typing import Optional

from ..core.schema import Sample


class ReasoningScorer:
    """Lightweight, model-free reasoning quality estimate."""

    def score(self, sample: Sample) -> float:
        reasoning = (sample.reasoning or "").strip()
        if not reasoning:
            return 0.0

        score = 0.0

        # 1. Explicit step markers indicate structured thinking.
        if re.search(r"step\s*\d+", reasoning, re.IGNORECASE):
            score += 0.25
        if re.search(r"\b(first|then|next|finally|therefore|because)\b", reasoning, re.IGNORECASE):
            score += 0.15

        # 2. Reasonable length (not trivially short, not absurdly long).
        n = len(reasoning)
        if 80 <= n <= 6000:
            score += 0.25
        elif 30 <= n < 80:
            score += 0.1

        # 3. Final answer consistency: the answer text should be grounded in
        #    the reasoning (e.g. the key result appears there).
        answer = (sample.response or "").strip()
        if answer:
            # Compare first meaningful token chunk for a cheap overlap test.
            key = answer.split()[0][:12] if answer.split() else ""
            if key and key.lower() in reasoning.lower():
                score += 0.25
            elif key and key.lower()[:6] in reasoning.lower():
                score += 0.1

        # 4. Penalise explicit contradiction markers.
        if re.search(r"\b(contradiction|that's wrong|i made an error)\b", reasoning, re.IGNORECASE):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def threshold(self, sample: Sample, min_score: float) -> bool:
        return self.score(sample) >= min_score
