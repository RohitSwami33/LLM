"""Duplicate detection: exact hash + optional near-duplicate (Jaccard)."""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple

from ..core.config import FilterConfig
from ..core.schema import Sample


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))


class DuplicateFilter:
    """Removes exact duplicates (and optionally near-duplicates)."""

    def __init__(self, config: FilterConfig):
        self.config = config
        self._exact: Set[str] = set()
        # Bounded cache for near-dup Jaccard comparison.
        self._near_cache: list = []
        self._near_threshold = config.near_dedup_threshold

    def check(self, sample: Sample) -> Tuple[bool, Optional[str]]:
        fp = sample.fingerprint()
        if fp in self._exact:
            return False, "exact_duplicate"
        self._exact.add(fp)

        if self.config.near_dedup:
            toks = _tokenize(sample.response)
            for cached in self._near_cache:
                if not cached:
                    continue
                union = len(toks | cached)
                if union == 0:
                    continue
                jaccard = len(toks & cached) / union
                if jaccard >= self._near_threshold:
                    return False, "near_duplicate"
            self._near_cache.append(toks)
            # Keep cache bounded to avoid unbounded memory at scale.
            if len(self._near_cache) > 200_000:
                self._near_cache = self._near_cache[-100_000:]
        return True, None
