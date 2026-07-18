"""Toxicity / safety filtering.

Implements a **deterministic lexicon-based** heuristic scorer so the framework
works out-of-the-box without a model server.  For production, replace
:meth:`score` with a call to a proper toxicity classifier (e.g. a HF model or
a moderation API) — the ``check`` contract stays identical.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..core.config import FilterConfig
from ..core.schema import Sample

# Illustrative blocklist. Replace with a maintained lexicon in production.
_LEXICON = [
    r"\b(hate|racist|slur|nazi|kill|murder|suicide|bomb|terrorist)\w*",
    r"\b(porn|rape|abuse|pedophil)\w*",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _LEXICON]


class ToxicityFilter:
    """Rejects samples whose toxicity score exceeds a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, text: str) -> float:
        """Return a normalised toxicity score in [0, 1]."""
        words = text.split() or [""]
        hits = 0
        for pat in _COMPILED:
            hits += len(pat.findall(text))
        # Normalise: ~5+ hits on a typical sample => saturated score.
        return min(1.0, hits / 5.0)

    def check(self, sample: Sample) -> Tuple[bool, Optional[str]]:
        s = self.score(sample.response or "")
        if s > self.threshold:
            return False, f"toxic:{s:.2f}"
        return True, None
