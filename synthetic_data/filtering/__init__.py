"""Filtering package: composable quality / safety filters.

:class:`FilterPipeline` chains the individual filters and records aggregate
metrics (accepted, rejected, duplicates removed, rejection reasons).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..core.config import FilterConfig
from ..core.schema import Sample
from .quality import QualityFilter
from .duplicate import DuplicateFilter
from .language import LanguageFilter
from .toxicity import ToxicityFilter

__all__ = [
    "QualityFilter",
    "DuplicateFilter",
    "LanguageFilter",
    "ToxicityFilter",
    "FilterPipeline",
    "build_filters",
]


class FilterPipeline:
    """Runs an ordered list of ``check(sample) -> (keep, reason)`` callables."""

    def __init__(self, filters: List[Callable[[Sample], Tuple[bool, Optional[str]]]]):
        self.filters = filters

    def check(self, sample: Sample) -> Tuple[bool, Optional[str]]:
        for f in self.filters:
            keep, reason = f.check(sample)
            if not keep:
                return False, reason
        return True, None


def build_filters(config: FilterConfig) -> FilterPipeline:
    """Construct a pipeline from a :class:`FilterConfig`."""
    filters: List = []

    if config.enabled:
        filters.append(QualityFilter(config))

    # Dedup must run before language/toxicity so we don't waste compute.
    if config.deduplicate or config.near_dedup:
        filters.append(DuplicateFilter(config))

    if config.language:
        filters.append(LanguageFilter(config.language))

    filters.append(ToxicityFilter(config.toxicity_threshold))

    return FilterPipeline(filters)
