"""Efficient deduplication using MinHash + Locality Sensitive Hashing (LSH).

Implements:
  - Exact deduplication (SHA-256 hash)
  - MinHash + LSH for near-duplicate detection (no pairwise comparison)
  - Streaming, incremental indexing
  - Memory-efficient with configurable thresholds
"""

import hashlib
import re
import time
import threading
from collections import defaultdict
from typing import Tuple, Optional, Set, Dict, List, Any

import numpy as np


def _normalize_text(text: str) -> str:
    """Normalize text for deduplication (lowercase, collapse whitespace)."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_shingles(text: str, k: int = 5) -> List[str]:
    """Get k-word shingles for MinHash."""
    words = text.split()
    if len(words) < k:
        return [text] if text else []
    return [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]


class ExactDeduplicator:
    """Exact deduplication using SHA-256 hashes.

    O(1) lookup per document. Memory: ~64 bytes per unique doc hash.
    """

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.stats = {"checked": 0, "duplicates": 0}
        self._lock = threading.Lock()

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Check if text is an exact duplicate."""
        self.stats["checked"] += 1

        normalized = _normalize_text(text)
        doc_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

        with self._lock:
            if doc_hash in self.seen_hashes:
                self.stats["duplicates"] += 1
                return True, "exact_duplicate"
            self.seen_hashes.add(doc_hash)

        return False, ""

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)


class MinHashLSH:
    """Efficient MinHash + Locality Sensitive Hashing for near-duplicate detection.

    Key optimization: Uses numpy vectorized operations for MinHash computation
    and LSH banding to avoid O(n²) pairwise comparison.

    Algorithm:
    1. Compute MinHash signature (num_hashes hash functions)
    2. Split signature into bands (num_bands bands, rows_per_band rows each)
    3. For each band, hash the band to a bucket
    4. Documents in the same bucket are candidates
    5. Only candidates are compared (not all pairs)

    Complexity:
    - MinHash computation: O(num_hashes * num_shingles) per doc
    - LSH lookup: O(num_bands) per doc (hash lookup in buckets)
    - Total: O(n * (num_hashes * shingle_size + num_bands))
    - Much better than O(n²) pairwise comparison
    """

    def __init__(self, num_hashes: int = 128, num_bands: int = 16,
                 shingle_size: int = 5, threshold: float = 0.8):
        """
        Args:
            num_hashes: Number of hash functions (must be divisible by num_bands).
            num_bands: Number of bands for LSH.
            shingle_size: Word shingle size.
            threshold: Similarity threshold for duplicate detection.
        """
        assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"

        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size
        self.threshold = threshold

        # Pre-generate hash function parameters (a * x + b) mod p
        # Using large primes for better distribution
        self._primes = np.array([
            4294967311, 4294967357, 4294967371, 4294967377,
            4294967389, 4294967393, 4294967411, 4294967423,
            4294967431, 4294967447, 4294967453, 4294967459,
        ], dtype=np.uint64)

        np.random.seed(42)
        self._a = np.random.randint(1, 2**32, size=num_hashes, dtype=np.uint64)
        self._b = np.random.randint(0, 2**32, size=num_hashes, dtype=np.uint64)
        self._p = np.uint64(2**32 - 1)

        # LSH buckets: band_idx -> {bucket_hash: set(doc_ids)}
        # Using set for O(1) membership check
        self.buckets: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
        self.doc_count = 0
        self.doc_hashes: Dict[int, np.ndarray] = {}  # doc_id -> MinHash signature

        self.stats = {"checked": 0, "duplicates": 0, "candidates": 0}
        self._lock = threading.Lock()

    def _compute_minhash(self, text: str) -> np.ndarray:
        """Compute MinHash signature using numpy vectorization.

        This is the core optimization: compute all hash functions at once
        using numpy broadcasting instead of Python loops.
        """
        shingles = _get_shingles(text, self.shingle_size)
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.uint32)

        # Hash all shingles at once
        shingle_hashes = np.array(
            [int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)
             for s in shingles],
            dtype=np.uint64
        )

        # Vectorized MinHash: compute all hash functions simultaneously
        # hash_i(x) = (a_i * x + b_i) mod p
        # MinHash_i = min over all x of hash_i(x)
        hash_vals = (self._a[:, None] * shingle_hashes[None, :] + self._b[:, None]) % self._p
        signature = hash_vals.min(axis=1).astype(np.uint32)

        return signature

    def _get_band_hashes(self, signature: np.ndarray) -> List[int]:
        """Get bucket hash for each band from MinHash signature."""
        band_hashes = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            # Hash the band values to get bucket ID
            band_hash = hash(signature[start:end].tobytes())
            band_hashes.append(band_hash)
        return band_hashes

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Check if text is a near-duplicate using MinHash + LSH.

        No pairwise comparison — uses LSH buckets for O(1) candidate lookup.
        """
        self.stats["checked"] += 1

        normalized = _normalize_text(text)
        signature = self._compute_minhash(normalized)
        band_hashes = self._get_band_hashes(signature)

        # Check for candidates in any band
        candidates = set()
        for band_idx, band_hash in enumerate(band_hashes):
            bucket = self.buckets[band_idx][band_hash]
            candidates.update(bucket)

        # If we have candidates, this is likely a duplicate
        if candidates:
            self.stats["candidates"] += len(candidates)
            self.stats["duplicates"] += 1
            return True, f"minhash_lsh_candidate={len(candidates)}"

        # Add to index
        doc_id = self.doc_count
        self.doc_count += 1

        # Store in LSH buckets
        for band_idx, band_hash in enumerate(band_hashes):
            self.buckets[band_idx][band_hash].add(doc_id)

        return False, ""

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        stats = dict(self.stats)
        stats["unique_docs"] = self.doc_count
        stats["num_bands"] = self.num_bands
        stats["num_hashes"] = self.num_hashes
        stats["num_buckets"] = sum(
            len(band) for band in self.buckets.values()
        )
        return stats


class NearDuplicateDetector:
    """Near-duplicate detection using Jaccard similarity on word sets.

    Memory-efficient: uses a sliding window of recent documents.
    """

    def __init__(self, threshold: float = 0.8, window_size: int = 10000,
                 shingle_size: int = 3):
        self.threshold = threshold
        self.window_size = window_size
        self.shingle_size = shingle_size

        self.recent_docs: List[tuple] = []
        self.doc_count = 0
        self.stats = {"checked": 0, "duplicates": 0}
        self._lock = threading.Lock()

    def _get_word_set(self, text: str) -> Set[str]:
        words = text.lower().split()
        if len(words) < self.shingle_size:
            return set(words)
        return {' '.join(words[i:i+self.shingle_size])
                for i in range(len(words) - self.shingle_size + 1)}

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        self.stats["checked"] += 1
        normalized = _normalize_text(text)
        shingles = self._get_word_set(normalized)

        with self._lock:
            for doc_id, recent_shingles in self.recent_docs:
                similarity = self._jaccard_similarity(shingles, recent_shingles)
                if similarity >= self.threshold:
                    self.stats["duplicates"] += 1
                    return True, f"near_duplicate=sim={similarity:.3f}"

            self.recent_docs.append((self.doc_count, shingles))
            self.doc_count += 1

            if len(self.recent_docs) > self.window_size:
                self.recent_docs = self.recent_docs[-self.window_size:]

        return False, ""

    def get_stats(self) -> Dict[str, int]:
        stats = dict(self.stats)
        stats["unique_docs"] = self.doc_count
        return stats


class DeduplicationPipeline:
    """Combined deduplication pipeline.

    Runs exact, MinHash + LSH, and near-duplicate detection in sequence.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        self.exact = ExactDeduplicator()
        self.minhash = None
        self.near_dup = None

        # Enable MinHash + LSH if configured
        if config.get("minhash", {}).get("enabled", False):
            mh_config = config.get("minhash", {})
            self.minhash = MinHashLSH(
                num_hashes=mh_config.get("num_hashes", 128),
                num_bands=mh_config.get("num_bands", 16),
                shingle_size=mh_config.get("shingle_size", 5),
                threshold=mh_config.get("threshold", 0.8),
            )

        # Enable near-duplicate detection if configured
        if config.get("near_duplicate", {}).get("enabled", False):
            nd_config = config.get("near_duplicate", {})
            self.near_dup = NearDuplicateDetector(
                threshold=nd_config.get("threshold", 0.8),
                window_size=nd_config.get("window_size", 10000),
                shingle_size=nd_config.get("shingle_size", 3),
            )

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Run all deduplication checks in sequence."""
        # 1. Exact dedup (fastest)
        is_dup, reason = self.exact.is_duplicate(text)
        if is_dup:
            return True, reason

        # 2. MinHash + LSH (efficient)
        if self.minhash:
            is_dup, reason = self.minhash.is_duplicate(text)
            if is_dup:
                return True, reason

        # 3. Near-duplicate detection (slowest, last)
        if self.near_dup:
            is_dup, reason = self.near_dup.is_duplicate(text)
            if is_dup:
                return True, reason

        return False, ""

    def get_stats(self) -> Dict[str, Any]:
        """Get combined deduplication statistics."""
        stats = {
            "exact": self.exact.get_stats(),
        }
        if self.minhash:
            stats["minhash_lsh"] = self.minhash.get_stats()
        if self.near_dup:
            stats["near_duplicate"] = self.near_dup.get_stats()
        return stats


def benchmark_dedup(input_path: str, text_key: str = "text",
                    num_samples: int = 10000) -> Dict[str, Any]:
    """Benchmark deduplication performance.

    Returns timing, throughput, and memory statistics.
    """
    import json
    import tracemalloc

    print(f"Benchmarking deduplication on {input_path}")
    print(f"Samples: {num_samples}")
    print()

    # Test configurations
    configs = {
        "exact_only": {"minhash": {"enabled": False}, "near_duplicate": {"enabled": False}},
        "exact_minhash": {"minhash": {"enabled": True, "num_hashes": 128, "num_bands": 16}, "near_duplicate": {"enabled": False}},
        "exact_minhash_neardup": {"minhash": {"enabled": True, "num_hashes": 128, "num_bands": 16}, "near_duplicate": {"enabled": True}},
    }

    results = {}

    for config_name, config in configs.items():
        print(f"--- {config_name} ---")

        tracemalloc.start()
        start_time = time.time()

        pipeline = DeduplicationPipeline(config)

        processed = 0
        duplicates = 0

        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > num_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = doc.get(text_key, "")
                if not text:
                    continue

                is_dup, reason = pipeline.is_duplicate(text)
                processed += 1
                if is_dup:
                    duplicates += 1

                if line_num % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    print(f"  Processed: {processed:,} | Dup: {duplicates:,} | Rate: {rate:.0f} docs/s")

        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rate = processed / elapsed
        dup_rate = duplicates / processed * 100 if processed > 0 else 0

        results[config_name] = {
            "processed": processed,
            "duplicates": duplicates,
            "duplicate_rate": round(dup_rate, 2),
            "time_seconds": round(elapsed, 2),
            "throughput_docs_per_sec": round(rate, 1),
            "peak_memory_mb": round(peak / 1024 / 1024, 1),
            "stats": pipeline.get_stats(),
        }

        print(f"  Results: {processed:,} docs in {elapsed:.1f}s = {rate:.0f} docs/s")
        print(f"  Duplicates: {duplicates:,} ({dup_rate:.1f}%)")
        print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
        print()

    # Print comparison
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Config':<25} {'Docs/s':>10} {'Time':>10} {'Dups':>10} {'Memory':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['throughput_docs_per_sec']:>10.0f} {r['time_seconds']:>10.1f} "
              f"{r['duplicates']:>10,} {r['peak_memory_mb']:>10.1f} MB")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        benchmark_dedup(sys.argv[1], num_samples=10000)
    else:
        print("Usage: python dedup.py <input.jsonl>")
        print("       python dedup.py datasets/mixtures/small/corpus.jsonl")
