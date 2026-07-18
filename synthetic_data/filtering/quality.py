"""Quality filters: completeness, length, repetition, formatting, JSON validity.

Each filter exposes ``check(sample) -> (keep: bool, reason: Optional[str])``
and is intentionally stateless so it can run in parallel over shards.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..core.config import FilterConfig
from ..core.schema import Sample

# Matches leftover template / instruction artefacts the model sometimes emits.
_HALLUCINATION_RE = re.compile(
    r"\{\{|\}\}|<<<|>>>|\[INST\]|\[/INST\]|###\s*instruction|"
    r"<\|.*?\|>|human:|assistant:",
    re.IGNORECASE,
)
_INCOMPLETE_RE = re.compile(r"\b(step\s*\d+\s*:.*){5,}", re.IGNORECASE)


def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of unique n-grams among all n-grams (lower = more repetitive)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def is_incomplete(text: str) -> bool:
    """Heuristic for truncated / unfinished responses."""
    text = text.strip()
    if not text:
        return True
    # Ends mid-word or with a dangling connector / colon.
    if re.search(r"\b(and|or|because|so|then|thus|therefore|,|;|:|\-)$", text):
        return True
    # Trailing unclosed code fence or bracket.
    if text.count("```") % 2 == 1:
        return True
    if text.count("{") != text.count("}"):
        return True
    if text.count("(") != text.count(")"):
        return True
    return False


class QualityFilter:
    """Composite heuristic quality checks driven by :class:`FilterConfig`."""

    def __init__(self, config: FilterConfig):
        self.config = config

    def check(self, sample: Sample) -> Tuple[bool, Optional[str]]:
        text = (sample.response or "").strip()
        # Math / QA / reasoning answers are legitimately short (a number,
        # percentage, or a single word); apply a much lower floor there.
        min_len = self.config.min_length
        if sample.task_type in ("math", "qa", "reasoning"):
            min_len = 2
        if len(text) < min_len:
            return False, "too_short"
        if len(text) > self.config.max_length:
            return False, "too_long"

        if self.config.remove_incomplete and is_incomplete(text):
            return False, "incomplete"

        if (
            self.config.remove_repeated_reasoning
            and repetition_ratio(text, n=4) < self.config.repetition_threshold
        ):
            # Only penalise repetition for longer outputs (short ones are fine).
            if len(text) > 120:
                return False, "repeated_reasoning"

        if (
            self.config.remove_hallucinated_formatting
            and _HALLUCINATION_RE.search(text)
        ):
            return False, "hallucinated_formatting"

        if self.config.remove_broken_json:
            fmt = sample.metadata.get("fields")
            # If the generator emitted structured fields, they must still be valid.
            # A response that was supposed to be JSON but lacks structure fails.
            if fmt is None and self._looks_like_broken_json(text):
                return False, "broken_json"

        return True, None

    @staticmethod
    def _looks_like_broken_json(text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith("{") and not stripped.startswith("["):
            return False
        try:
            import json

            json.loads(stripped)
            return False
        except Exception:
            return True
