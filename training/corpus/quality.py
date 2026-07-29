"""Quality metric filters for corpus cleaning.

Computes statistical quality metrics and filters documents
that fall outside configurable thresholds.
"""

import re
import math
import zlib
from collections import Counter
from typing import Tuple, Optional, Dict, Any


class QualityMetrics:
    """Compute quality metrics for a document."""

    @staticmethod
    def url_ratio(text: str) -> float:
        """Ratio of URL characters to total characters."""
        urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text)
        url_chars = sum(len(u) for u in urls)
        return url_chars / max(len(text), 1)

    @staticmethod
    def digit_ratio(text: str) -> float:
        """Ratio of digit characters to total characters."""
        if not text:
            return 0.0
        return sum(1 for c in text if c.isdigit()) / len(text)

    @staticmethod
    def symbol_ratio(text: str) -> float:
        """Ratio of symbol/non-alphanumeric characters to total characters."""
        if not text:
            return 0.0
        return sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)

    @staticmethod
    def duplicate_line_ratio(text: str) -> float:
        """Ratio of duplicate lines to total lines."""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return 0.0
        unique_lines = set(lines)
        return 1.0 - (len(unique_lines) / len(lines))

    @staticmethod
    def average_word_length(text: str) -> float:
        """Average word length in characters."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def max_repeated_token_frequency(text: str, top_n: int = 5) -> float:
        """Maximum frequency of any single token (word)."""
        words = text.lower().split()
        if not words:
            return 0.0
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        return max_freq / len(words)

    @staticmethod
    def character_entropy(text: str) -> float:
        """Shannon entropy of character distribution."""
        if not text:
            return 0.0
        char_freq = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in char_freq.values())
        return entropy

    @staticmethod
    def compression_ratio(text: str) -> float:
        """Compression ratio using zlib. Low ratio = repetitive/low quality."""
        if not text or len(text) < 100:
            return 1.0
        compressed = zlib.compress(text.encode('utf-8'))
        return len(compressed) / len(text.encode('utf-8'))

    @staticmethod
    def word_entropy(text: str) -> float:
        """Shannon entropy of word distribution."""
        words = text.lower().split()
        if not words:
            return 0.0
        word_freq = Counter(words)
        total = len(words)
        entropy = -sum((c / total) * math.log2(c / total) for c in word_freq.values())
        return entropy

    @staticmethod
    def type_token_ratio(text: str) -> float:
        """Ratio of unique words to total words (lexical diversity)."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def sentence_count(text: str) -> int:
        """Estimate sentence count."""
        return len(re.split(r'[.!?]+', text.strip()))

    @staticmethod
    def avg_sentence_length(text: str) -> float:
        """Average sentence length in words."""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    @staticmethod
    def paragraph_count(text: str) -> int:
        """Count paragraphs (double newline separated)."""
        return len([p for p in text.split('\n\n') if p.strip()])

    @classmethod
    def compute_all(cls, text: str) -> Dict[str, Any]:
        """Compute all metrics for a document."""
        return {
            "url_ratio": cls.url_ratio(text),
            "digit_ratio": cls.digit_ratio(text),
            "symbol_ratio": cls.symbol_ratio(text),
            "duplicate_line_ratio": cls.duplicate_line_ratio(text),
            "avg_word_length": cls.average_word_length(text),
            "max_token_freq": cls.max_repeated_token_frequency(text),
            "char_entropy": cls.character_entropy(text),
            "compression_ratio": cls.compression_ratio(text),
            "word_entropy": cls.word_entropy(text),
            "type_token_ratio": cls.type_token_ratio(text),
            "sentence_count": cls.sentence_count(text),
            "avg_sentence_length": cls.avg_sentence_length(text),
            "paragraph_count": cls.paragraph_count(text),
            "char_count": len(text),
            "word_count": len(text.split()),
        }


class QualityFilter:
    """Filter documents based on configurable quality thresholds.

    Each threshold can be enabled/disabled independently.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Thresholds: (min, max) or None to disable
        self.url_ratio = config.get("url_ratio", (0.0, 0.15))
        self.digit_ratio = config.get("digit_ratio", (0.0, 0.5))
        self.symbol_ratio = config.get("symbol_ratio", (0.0, 0.3))
        self.duplicate_line_ratio = config.get("duplicate_line_ratio", (0.0, 0.5))
        self.avg_word_length = config.get("avg_word_length", (2.0, 15.0))
        self.max_token_freq = config.get("max_token_freq", (0.0, 0.15))
        self.char_entropy = config.get("char_entropy", (3.0, 8.0))
        self.compression_ratio = config.get("compression_ratio", (0.2, 0.9))
        self.word_entropy = config.get("word_entropy", None)
        self.type_token_ratio = config.get("type_token_ratio", (0.1, 1.0))
        self.min_words = config.get("min_words", 10)
        self.min_sentences = config.get("min_sentences", 1)

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        """Check if document passes quality thresholds.

        Returns:
            (keep, reason) tuple. reason is empty string if kept.
        """
        if not text or len(text.strip()) < 20:
            return False, "too_short"

        metrics = QualityMetrics.compute_all(text)

        # Word count check
        if metrics["word_count"] < self.min_words:
            return False, f"min_words={metrics['word_count']}"

        # Sentence count check
        if self.min_sentences and metrics["sentence_count"] < self.min_sentences:
            return False, f"min_sentences={metrics['sentence_count']}"

        # Check each threshold
        checks = [
            ("url_ratio", metrics["url_ratio"], self.url_ratio),
            ("digit_ratio", metrics["digit_ratio"], self.digit_ratio),
            ("symbol_ratio", metrics["symbol_ratio"], self.symbol_ratio),
            ("duplicate_line_ratio", metrics["duplicate_line_ratio"], self.duplicate_line_ratio),
            ("avg_word_length", metrics["avg_word_length"], self.avg_word_length),
            ("max_token_freq", metrics["max_token_freq"], self.max_token_freq),
            ("char_entropy", metrics["char_entropy"], self.char_entropy),
            ("compression_ratio", metrics["compression_ratio"], self.compression_ratio),
            ("type_token_ratio", metrics["type_token_ratio"], self.type_token_ratio),
        ]

        if self.word_entropy:
            checks.append(("word_entropy", metrics["word_entropy"], self.word_entropy))

        for name, value, threshold in checks:
            if threshold is None:
                continue
            min_val, max_val = threshold
            if value < min_val:
                return False, f"{name}={value:.3f}<min={min_val}"
            if value > max_val:
                return False, f"{name}={value:.3f}>max={max_val}"

        return True, ""

    def get_metrics(self, text: str) -> Dict[str, Any]:
        """Get quality metrics for a document (for reporting)."""
        return QualityMetrics.compute_all(text)
