"""Language detection + mismatch filtering.

Uses ``langdetect`` when available; otherwise degrades gracefully to a
pass-through (no language filtering) and logs a warning.  This keeps the
framework importable and runnable without the optional dependency.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..core.schema import Sample

_LANGDET_OK = False
try:
    from langdetect import detect as _detect, DetectorFactory

    DetectorFactory.seed = 0
    _LANGDET_OK = True
except ImportError:  # pragma: no cover - optional dependency
    _detect = None


class LanguageFilter:
    """Keeps only samples whose response language matches the expected code."""

    def __init__(self, expected: Optional[str] = None):
        self.expected = expected
        if expected and not _LANGDET_OK:
            import logging

            logging.getLogger(__name__).warning(
                "langdetect not installed; language filtering disabled. "
                "Install with: pip install langdetect"
            )

    def detect(self, text: str) -> str:
        if not _LANGDET_OK:
            return "unknown"
        try:
            return _detect(text)
        except Exception:
            return "unknown"

    def check(self, sample: Sample) -> Tuple[bool, Optional[str]]:
        if not self.expected:
            return True, None
        lang = self.detect(sample.response or "")
        if lang == "unknown":
            return True, None  # don't penalise when detection is unavailable
        if lang != self.expected:
            return False, f"language_mismatch:{lang}"
        return True, None
