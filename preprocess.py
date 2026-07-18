#!/usr/bin/env python3
"""
preprocess.py – Configurable text cleaning pipeline for LLM datasets
====================================================================

Applies a configurable set of cleaning steps to text datasets:

  1. Empty document removal
  2. HTML & URL artifact removal
  3. Unicode normalization (NFKC)
  4. Whitespace normalization
  5. Length-based filtering
  6. Exact deduplication (SHA-256)
  7. Optional language detection / filtering

Works with both streaming (iterable) and fully-loaded datasets.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML / URL cleaning helpers
# ---------------------------------------------------------------------------

# Pattern matches HTML tags, entities, and common URL patterns
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;")
_URL_RE = re.compile(r"https?://\S+")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_html(text: str) -> str:
    """Remove HTML tags, entities, and bare URLs from text."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode and strip control characters."""
    text = unicodedata.normalize(form, text)
    text = _CONTROL_CHARS_RE.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into a single space, then strip."""
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Language detection (optional dependency)
# ---------------------------------------------------------------------------

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect as _langdetect_detect

    _LANGDETECT_AVAILABLE = True
except ImportError:
    pass


def detect_language(text: str) -> Optional[str]:
    """Return ISO 639-1 language code, or None if detection fails."""
    if not _LANGDETECT_AVAILABLE:
        return None
    try:
        return _langdetect_detect(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statistics collector
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    """Collects document-level statistics during preprocessing."""

    input_documents: int = 0
    output_documents: int = 0
    removed_empty: int = 0
    removed_html_artifacts: int = 0
    removed_duplicates: int = 0
    removed_language: int = 0
    removed_length: int = 0
    total_chars: int = 0
    total_tokens_estimate: int = 0
    languages: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_documents": self.input_documents,
            "output_documents": self.output_documents,
            "removed_empty": self.removed_empty,
            "removed_html_artifacts": self.removed_html_artifacts,
            "removed_duplicates": self.removed_duplicates,
            "removed_language": self.removed_language,
            "removed_length": self.removed_length,
            "dedup_ratio": round(self.removed_duplicates / max(self.input_documents, 1), 4),
            "retention_rate": round(
                self.output_documents / max(self.input_documents, 1), 4
            ),
            "total_chars": self.total_chars,
            "total_tokens_estimate": self.total_tokens_estimate,
            "avg_doc_length_chars": (
                round(self.total_chars / max(self.output_documents, 1), 1)
                if self.output_documents
                else 0
            ),
            "languages": dict(self.languages.most_common()),
        }

    def merge(self, other: Stats) -> Stats:
        """Combine stats from another collector."""
        self.input_documents += other.input_documents
        self.output_documents += other.output_documents
        self.removed_empty += other.removed_empty
        self.removed_html_artifacts += other.removed_html_artifacts
        self.removed_duplicates += other.removed_duplicates
        self.removed_language += other.removed_language
        self.removed_length += other.removed_length
        self.total_chars += other.total_chars
        self.total_tokens_estimate += other.total_tokens_estimate
        self.languages += other.languages
        return self


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "remove_empty": True,
    "remove_html": True,
    "normalize_unicode": True,
    "normalize_whitespace": True,
    "deduplicate": True,
    "min_length": 0,
    "max_length": 10_000_000,
    "truncate_long": False,
    "filter_language": None,  # None = disabled, or a 2-letter code like "en"
}


class PreprocessingPipeline:
    """
    Configurable, stateful text preprocessing pipeline for LLM datasets.

    Usage::

        pipe = PreprocessingPipeline({"min_length": 100, "deduplicate": True})
        for text, meta in pipe.process_dataset(dataset, text_key="text"):
            # text is cleaned, meta carries the original metadata
            ...

    Call :meth:`get_stats` after processing to obtain cleaning statistics.
    Call :meth:`reset` to reuse the same pipeline on another dataset.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config: dict = {**DEFAULT_CONFIG, **(config or {})}
        self._seen_hashes: set[str] = set()
        self.stats: Stats = Stats()

    # ----- single-document API -------------------------------------------------

    def process(self, text: str, metadata: Optional[dict] = None) -> Optional[str]:
        """
        Clean a single document.

        Returns the cleaned text, or *None* if the document is filtered out.
        """
        self.stats.input_documents += 1
        original = text

        # 1. Empty document removal --------------------------------------------
        if self.config.get("remove_empty", True) and not text.strip():
            self.stats.removed_empty += 1
            return None

        # 2. HTML / URL artifact removal ---------------------------------------
        if self.config.get("remove_html", True):
            cleaned = strip_html(text)
            if cleaned != text:
                self.stats.removed_html_artifacts += 1
            text = cleaned

        # 3. Unicode normalization ---------------------------------------------
        if self.config.get("normalize_unicode", True):
            text = normalize_unicode(text)

        # 4. Whitespace normalization ------------------------------------------
        if self.config.get("normalize_whitespace", True):
            text = normalize_whitespace(text)

        # 5. Length filtering --------------------------------------------------
        min_len = self.config.get("min_length", 0)
        max_len = self.config.get("max_length", 10_000_000)
        if len(text) < min_len:
            self.stats.removed_length += 1
            return None
        if len(text) > max_len:
            if self.config.get("truncate_long", False):
                text = text[:max_len]
            else:
                self.stats.removed_length += 1
                return None

        # 6. Exact deduplication -----------------------------------------------
        if self.config.get("deduplicate", True):
            doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if doc_hash in self._seen_hashes:
                self.stats.removed_duplicates += 1
                return None
            self._seen_hashes.add(doc_hash)

        # 7. Language filtering (optional) -------------------------------------
        target_lang = self.config.get("filter_language")
        if target_lang is not None:
            detected = detect_language(text)
            if detected is None or detected != target_lang:
                self.stats.removed_language += 1
                return None

        # -- update stats ------------------------------------------------------
        self.stats.output_documents += 1
        self.stats.total_chars += len(text)
        self.stats.total_tokens_estimate += len(text) // 4  # ≈4 chars/token
        if metadata and "language" in metadata:
            self.stats.languages[metadata["language"]] += 1

        return text

    # ----- dataset-level API --------------------------------------------------

    def process_dataset(
        self,
        dataset: Iterator[dict],
        text_key: str = "text",
    ) -> Iterator[tuple[str, dict]]:
        """
        Process an iterable of dicts (e.g. a Hugging Face dataset).

        Yields ``(cleaned_text, metadata_dict)`` for every document that
        passes all filtering steps.
        """
        for example in dataset:
            text = example.get(text_key, "")
            metadata = {k: v for k, v in example.items() if k != text_key}
            result = self.process(text, metadata)
            if result is not None:
                yield result, metadata

    # ----- utilities ----------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a snapshot of current cleaning statistics."""
        return self.stats.to_dict()

    def reset(self):
        """Reset the dedup cache and statistics (keeps config)."""
        self._seen_hashes.clear()
        self.stats = Stats()

    @property
    def config_snapshot(self) -> dict:
        """Return a copy of the active config (for reproducibility)."""
        return dict(self.config)


# ---------------------------------------------------------------------------
# Convenience: run pipeline from the command line
# ---------------------------------------------------------------------------

def build_pipeline_from_yaml(config_path: str) -> PreprocessingPipeline:
    """Load a YAML preprocessing config and return a pipeline."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pre_config = cfg.get("preprocessing", {})
    log.info("Preprocessing config: %s", pre_config)
    return PreprocessingPipeline(pre_config)


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preprocess a JSONL dataset")
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL path")
    parser.add_argument(
        "--config", "-c", default=None, help="YAML config file with preprocessing"
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            import yaml
            config = yaml.safe_load(f).get("preprocessing", {})

    pipe = PreprocessingPipeline(config)
    out_path = args.output or args.input.replace(".jsonl", ".cleaned.jsonl")

    written = 0
    with open(args.input) as fin, open(out_path, "w") as fout:
        for line in fin:
            example = json.loads(line)
            text = example.get("text", example.get("content", ""))
            meta = {k: v for k, v in example.items() if k not in ("text", "content")}
            result = pipe.process(text, meta)
            if result is not None:
                out = {"text": result, **meta}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

    log.info("Wrote %d documents to %s", written, out_path)
    log.info("Stats: %s", json.dumps(pipe.get_stats(), indent=2))
