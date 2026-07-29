"""Main corpus cleaning pipeline.

Orchestrates all filters, deduplication, and quality checks.
Processes JSONL files and produces cleaned output.
"""

import os
import json
import time
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List, Tuple
from collections import Counter

from .filters import build_filter_chain, ContentFilter, DocumentLengthFilter
from .quality import QualityFilter, QualityMetrics
from .dedup import DeduplicationPipeline


class CleaningStats:
    """Track cleaning statistics."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.removal_reasons: Counter = Counter()
        self.start_time = time.time()
        self.filter_times: Dict[str, float] = {}
        self.quality_metrics_before: List[Dict] = []
        self.quality_metrics_after: List[Dict] = []

    def record_removal(self, reason: str):
        """Record a document removal with reason."""
        self.total_input += 1
        self.removal_reasons[reason] += 1

    def record_keep(self):
        """Record a kept document."""
        self.total_input += 1
        self.total_output += 1

    def record_filter_time(self, filter_name: str, elapsed: float):
        """Record time spent in a filter."""
        self.filter_times[filter_name] = self.filter_times.get(filter_name, 0) + elapsed

    def get_report(self) -> Dict[str, Any]:
        """Generate cleaning report."""
        elapsed = time.time() - self.start_time
        filtered_pct = (1 - self.total_output / max(self.total_input, 1)) * 100

        # Sort removal reasons by count
        top_reasons = self.removal_reasons.most_common(50)

        # Quality metrics summary
        quality_summary = {}
        if self.quality_metrics_before:
            for key in self.quality_metrics_before[0]:
                values = [m[key] for m in self.quality_metrics_before if key in m]
                if values:
                    quality_summary[f"before_{key}"] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
        if self.quality_metrics_after:
            for key in self.quality_metrics_after[0]:
                values = [m[key] for m in self.quality_metrics_after if key in m]
                if values:
                    quality_summary[f"after_{key}"] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_filtered": self.total_input - self.total_output,
            "filtered_percentage": round(filtered_pct, 2),
            "removal_reasons": dict(top_reasons),
            "filter_times_seconds": {k: round(v, 2) for k, v in self.filter_times.items()},
            "total_time_seconds": round(elapsed, 2),
            "throughput_docs_per_sec": round(self.total_input / max(elapsed, 0.001), 1),
            "quality_summary": quality_summary,
        }


class CorpusCleaner:
    """Main corpus cleaning pipeline.

    Usage:
        cleaner = CorpusCleaner.from_yaml("training/corpus/config.yaml")
        cleaner.clean_file("input.jsonl", "output.jsonl")
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Build filter chain
        self.filters = build_filter_chain(config)

        # Build quality filter
        quality_config = config.get("quality", {})
        self.quality_filter = QualityFilter(quality_config) if quality_config.get("enabled", True) else None

        # Build deduplication pipeline
        dedup_config = config.get("deduplication", {})
        self.dedup = DeduplicationPipeline(dedup_config) if dedup_config.get("enabled", True) else None

        # Document length filter (special handling for splitting)
        self.doc_length_filter = None
        for f in self.filters:
            if isinstance(f, DocumentLengthFilter):
                self.doc_length_filter = f
                break

        # Text normalization
        self.normalize = config.get("normalize", True)

    @classmethod
    def from_yaml(cls, path: str) -> "CorpusCleaner":
        """Load cleaner from YAML config."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        if not self.normalize:
            return text

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove control characters (keep tab, newline, carriage return)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text

    def clean_document(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str, Optional[Dict]]:
        """Clean a single document.

        Returns:
            (keep, reason, metrics) tuple.
            keep: True if document should be kept.
            reason: Rejection reason (empty if kept).
            metrics: Quality metrics if computed.
        """
        if not text or not text.strip():
            return False, "empty", None

        # Normalize
        text = self._normalize_text(text)

        # Run content filters
        for filter_obj in self.filters:
            keep, reason = filter_obj(text, metadata)
            if not keep:
                return False, f"filter_{filter_obj.name}:{reason}", None

        # Run quality filter
        if self.quality_filter:
            metrics = self.quality_filter.get_metrics(text)
            keep, reason = self.quality_filter(text, metadata)
            if not keep:
                return False, f"quality:{reason}", metrics
        else:
            metrics = None

        # Run deduplication
        if self.dedup:
            is_dup, reason = self.dedup.is_duplicate(text)
            if is_dup:
                return False, f"dedup:{reason}", metrics

        return True, "", metrics

    def clean_file(self, input_path: str, output_path: str,
                   text_key: str = "text",
                   sample_size: Optional[int] = None,
                   progress_callback=None) -> Dict[str, Any]:
        """Clean a JSONL file.

        Args:
            input_path: Input JSONL file path.
            output_path: Output JSONL file path.
            text_key: Key containing the text in each JSON line.
            sample_size: Process only first N documents (None = all).
            progress_callback: Optional callback(processed, kept, removed).

        Returns:
            Cleaning report dict.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = CleaningStats()
        sample_metrics_before = []
        sample_metrics_after = []
        SAMPLE_LIMIT = 1000  # Max metrics samples to keep in memory

        print(f"Cleaning: {input_path}")
        print(f"Output:   {output_path}")
        print(f"Filters:  {len(self.filters)} content + quality + dedup")
        print()

        with open(input_path, 'r', encoding='utf-8', errors='replace') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:

            for line_num, line in enumerate(fin, 1):
                if sample_size and line_num > sample_size:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    stats.record_removal("json_decode_error")
                    continue

                text = doc.get(text_key, "")
                if not text:
                    stats.record_removal("no_text_field")
                    continue

                # Clean document
                keep, reason, metrics = self.clean_document(text, doc)

                if keep:
                    stats.record_keep()
                    fout.write(json.dumps(doc, ensure_ascii=False) + '\n')

                    # Sample quality metrics
                    if metrics and len(sample_metrics_after) < SAMPLE_LIMIT:
                        sample_metrics_after.append(metrics)
                else:
                    stats.record_removal(reason)

                # Sample quality metrics from input
                if metrics and len(sample_metrics_before) < SAMPLE_LIMIT:
                    sample_metrics_before.append(metrics)

                # Progress callback
                if progress_callback and line_num % 10000 == 0:
                    progress_callback(stats.total_input, stats.total_output,
                                      stats.total_input - stats.total_output)

                # Progress output
                if line_num % 100000 == 0:
                    elapsed = time.time() - stats.start_time
                    rate = line_num / elapsed
                    filtered_pct = (1 - stats.total_output / max(stats.total_input, 1)) * 100
                    print(f"  Processed: {line_num:,} | Kept: {stats.total_output:,} | "
                          f"Filtered: {filtered_pct:.1f}% | Rate: {rate:.0f} docs/s")

        # Final stats
        stats.quality_metrics_before = sample_metrics_before
        stats.quality_metrics_after = sample_metrics_after

        report = stats.get_report()
        report["input_file"] = str(input_path)
        report["output_file"] = str(output_path)
        report["text_key"] = text_key

        # Print summary
        print()
        print(f"{'='*60}")
        print(f"CLEANING COMPLETE")
        print(f"{'='*60}")
        print(f"  Input:     {report['total_input']:,} documents")
        print(f"  Output:    {report['total_output']:,} documents")
        print(f"  Filtered:  {report['total_filtered']:,} ({report['filtered_percentage']:.1f}%)")
        print(f"  Time:      {report['total_time_seconds']:.1f}s")
        print(f"  Rate:      {report['throughput_docs_per_sec']:.0f} docs/s")
        print()
        print("Top removal reasons:")
        for reason, count in list(report["removal_reasons"].items())[:10]:
            pct = count / max(report["total_input"], 1) * 100
            print(f"  {reason:40s} {count:>8,} ({pct:.1f}%)")
        print(f"{'='*60}")

        return report

    def clean_text(self, text: str) -> Tuple[bool, str]:
        """Clean a single text string (for pipeline integration)."""
        keep, reason, _ = self.clean_document(text)
        return keep, reason
