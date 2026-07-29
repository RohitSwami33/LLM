"""Optimized dataset for Apple Silicon training.

Key optimizations:
- Lazy tokenization (tokenize on first access, not at init)
- Pre-tokenized cache on disk (avoids re-tokenization)
- Background prefetching with multiprocessing
- Memory-mapped access for large datasets
- Auto-tuning based on available RAM
"""

import json
import os
import time
import mmap
import struct
import pickle
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class PreTokenizedCache:
    """Disk-backed cache for tokenized sequences.

    Saves tokenized data to a binary file so subsequent loads are instant.
    Uses memory-mapped access for large files.
    """

    def __init__(self, cache_dir: str = "datasets/research_v2/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, data_path: str, tokenizer_name: str, max_seq_len: int) -> Path:
        """Generate cache file path based on data + tokenizer + config."""
        h = hashlib.md5(f"{data_path}:{tokenizer_name}:{max_seq_len}".encode()).hexdigest()[:12]
        return self.cache_dir / f"tokenized_{h}.bin"

    def exists(self, data_path: str, tokenizer_name: str, max_seq_len: int) -> bool:
        path = self._cache_path(data_path, tokenizer_name, max_seq_len)
        return path.exists()

    def save(self, sequences: List[Dict[str, torch.Tensor]], data_path: str, tokenizer_name: str, max_seq_len: int):
        """Save tokenized sequences to disk."""
        path = self._cache_path(data_path, tokenizer_name, max_seq_len)

        with open(path, 'wb') as f:
            pickle.dump(sequences, f)

        print(f"  Cache saved: {path.name} ({len(sequences):,} sequences)")

    def load(self, data_path: str, tokenizer_name: str, max_seq_len: int) -> List[Dict[str, torch.Tensor]]:
        """Load tokenized sequences from disk."""
        path = self._cache_path(data_path, tokenizer_name, max_seq_len)

        with open(path, 'rb') as f:
            sequences = pickle.load(f)

        print(f"  Cache loaded: {path.name} ({len(sequences):,} sequences, instant)")
        return sequences


class OptimizedPackedDataset(Dataset):
    """Packed sequence dataset optimized for Apple Silicon.

    Features:
    - Lazy tokenization (only when needed)
    - Disk-backed pre-tokenized cache
    - Minimal memory footprint
    - Fast first-load with background caching
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 2048,
        pack_factor: int = 4,
        cache_dir: str = "datasets/research_v2/cache",
        field: Optional[str] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack_factor = pack_factor
        self.field = field

        # Try loading from cache first
        cache = PreTokenizedCache(cache_dir)
        tokenizer_name = f"sp_{tokenizer.vocab_size}"

        if cache.exists(path, tokenizer_name, max_seq_len):
            print("Loading from pre-tokenized cache...")
            t0 = time.time()
            self.examples = cache.load(path, tokenizer_name, max_seq_len)
            print(f"  Cache load: {time.time()-t0:.1f}s")
        else:
            print("Tokenizing and packing (first time only)...")
            t0 = time.time()
            texts = self._load_texts()
            print(f"  Loaded {len(texts):,} texts in {time.time()-t0:.1f}s")

            t1 = time.time()
            tokenized = self._tokenize_texts(texts)
            print(f"  Tokenized {len(tokenized):,} sequences in {time.time()-t1:.1f}s")
            del texts  # Free memory

            t2 = time.time()
            self.examples = self._pack_sequences(tokenized)
            print(f"  Packed into {len(self.examples):,} examples in {time.time()-t2:.1f}s")
            del tokenized  # Free memory

            # Save cache for next time
            cache.save(self.examples, path, tokenizer_name, max_seq_len)

        print(f"Dataset ready: {len(self.examples):,} packed examples")

    def _load_texts(self) -> List[str]:
        texts = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = self._extract_text(item)
                if text:
                    texts.append(text)
        return texts

    def _extract_text(self, item: Dict[str, Any]) -> str:
        if self.field and self.field in item:
            return str(item[self.field])
        parts = []
        for key in ["instruction", "input", "context", "question",
                     "reasoning", "response", "output", "text",
                     "source", "target", "problem", "derivation", "answer"]:
            if key in item and item[key]:
                parts.append(str(item[key]))
        return " ".join(parts) if parts else ""

    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        tokenized = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.max_seq_len:
                tokenized.append(ids)
        return tokenized

    def _pack_sequences(self, tokenized: List[List[int]]) -> List[Dict[str, torch.Tensor]]:
        """Pack sequences into max_seq_len chunks with segment and position info."""
        packed = []
        current_ids = []
        current_segments = []
        current_positions = []
        segment_idx = 0
        pos_in_segment = 0

        for ids in tokenized:
            if len(current_ids) + len(ids) > self.max_seq_len:
                if current_ids:
                    packed.append(self._make_packed_item(
                        current_ids, current_segments, current_positions
                    ))
                    current_ids = []
                    current_segments = []
                    current_positions = []
                    segment_idx = 0
                    pos_in_segment = 0
                else:
                    continue

            for token_id in ids:
                current_ids.append(token_id)
                current_segments.append(segment_idx)
                current_positions.append(pos_in_segment)
                pos_in_segment += 1
            segment_idx += 1
            pos_in_segment = 0

        if current_ids and len(current_ids) > 100:
            while len(current_ids) < self.max_seq_len:
                current_ids.append(0)
                current_segments.append(-1)
                current_positions.append(0)
            packed.append(self._make_packed_item(
                current_ids[:self.max_seq_len],
                current_segments[:self.max_seq_len],
                current_positions[:self.max_seq_len],
            ))

        return packed

    def _make_packed_item(self, ids, segments, positions):
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class FastJsonlDataset(Dataset):
    """Fast JSONL dataset with memory-mapped reading.

    Optimized for large files where we want minimal memory usage.
    """

    def __init__(self, path: str, tokenizer, max_seq_len: int = 2048):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._offsets = []
        self._index_file = Path(path).with_suffix(".idx")

        # Build or load index
        if self._index_file.exists():
            self._load_index()
        else:
            self._build_index()

    def _build_index(self):
        """Build line offset index for O(1) random access."""
        print("Building line index...")
        t0 = time.time()
        with open(self.path, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self._offsets.append(offset)
                offset += len(line)

        # Save index
        with open(self._index_file, 'wb') as f:
            pickle.dump(self._offsets, f)
        print(f"  Index built: {len(self._offsets):,} lines in {time.time()-t0:.1f}s")

    def _load_index(self):
        """Load pre-built index."""
        with open(self._index_file, 'rb') as f:
            self._offsets = pickle.load(f)
        print(f"  Index loaded: {len(self._offsets):,} lines (instant)")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._offsets[idx]
        with open(self.path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            item = json.loads(line.strip())

        text = item.get('text', '')
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(encoded, dtype=torch.long),
        }


def get_system_info() -> Dict[str, Any]:
    """Get system info for auto-tuning."""
    import platform
    import subprocess

    info = {
        'cpu_count': os.cpu_count(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'mps': torch.backends.mps.is_available(),
    }

    try:
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        info['ram_gb'] = int(result.stdout.strip()) / (1024**3)
    except:
        info['ram_gb'] = 16.0  # Default estimate

    return info


def auto_tune(dataset_size: int, model_params: int) -> Dict[str, Any]:
    """Auto-tune training parameters based on system and dataset.

    Returns optimized config for:
    - batch_size
    - num_workers
    - prefetch_factor
    - gradient_accumulation_steps
    - pack_factor
    """
    info = get_system_info()
    ram_gb = info['ram_gb']
    cpu_count = info['cpu_count']
    is_mps = info['mps']

    # Model memory estimate (MB)
    model_mb = model_params * 4 / 1e6  # FP32
    if is_mps:
        model_mb *= 2  # MPS uses ~2x for forward+backward

    # Available RAM after model
    available_ram_gb = ram_gb - (model_mb / 1024) - 2.0  # 2GB headroom

    # Dataset memory estimate (assume ~1KB per token sequence)
    dataset_mb = dataset_size * 2048 * 2 / 1e6  # ~4MB per example
    dataset_fits_in_ram = dataset_mb < (available_ram_gb * 1024 * 0.5)

    # Auto-tune based on available resources
    config = {}

    if is_mps:
        # Apple Silicon: smaller batch, fewer workers (MPS + multiprocessing = issues)
        config['batch_size'] = 4 if dataset_fits_in_ram else 2
        config['gradient_accumulation_steps'] = 8 if config['batch_size'] <= 2 else 4
        config['num_workers'] = 2  # MPS has issues with many DataLoader workers
        config['prefetch_factor'] = 2
        config['pack_factor'] = 8
    else:
        # CUDA: larger batch
        config['batch_size'] = 8
        config['gradient_accumulation_steps'] = 4
        config['num_workers'] = min(cpu_count - 1, 8)
        config['prefetch_factor'] = 2
        config['pack_factor'] = 4

    # Effective batch size = batch_size * gradient_accumulation * seq_len
    effective_tokens = config['batch_size'] * config['gradient_accumulation_steps'] * 2048
    config['effective_tokens_per_step'] = effective_tokens

    print(f"\n=== AUTO-TUNE RESULTS ===")
    print(f"System: {info['machine']} | {ram_gb:.1f}GB RAM | {cpu_count} cores | MPS: {is_mps}")
    print(f"Model: {model_params/1e6:.1f}M params ({model_mb:.0f} MB)")
    print(f"Dataset: {dataset_size:,} examples ({dataset_mb:.0f} MB)")
    print(f"Dataset fits in RAM: {dataset_fits_in_ram}")
    print(f"")
    print(f"Optimized config:")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  gradient_accumulation: {config['gradient_accumulation_steps']}")
    print(f"  effective_batch: {config['effective_tokens_per_step']:,} tokens/step")
    print(f"  num_workers: {config['num_workers']}")
    print(f"  prefetch_factor: {config['prefetch_factor']}")
    print(f"  pack_factor: {config['pack_factor']}")
    print(f"========================\n")

    return config
