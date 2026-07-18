"""Confidence scoring from teacher log-probs.

When the teacher is configured with ``logprobs: true``, the mean token
log-probability is a cheap, model-native confidence signal.  Samples without
log-probs (or with anomalous means) can be down-weighted or rejected.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.schema import Sample


class ConfidenceScorer:
    """Compute mean token confidence from stored log-probs."""

    def score(self, sample: Sample) -> Optional[float]:
        logprobs = sample.metadata.get("logprobs")
        if not logprobs:
            return None
        if not isinstance(logprobs, list) or not logprobs:
            return None
        valid = [lp for lp in logprobs if lp is not None]
        if not valid:
            return None
        mean_lp = sum(valid) / len(valid)
        # Convert log-prob to probability in [0, 1].
        return math.exp(mean_lp)

    def threshold(self, sample: Sample, min_confidence: float) -> bool:
        """Return True if the sample meets the minimum confidence."""
        s = self.score(sample)
        if s is None:
            return True  # no signal -> don't reject on this axis
        return s >= min_confidence
