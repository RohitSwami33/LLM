"""Dataset implementations for language model pretraining.

Supports:
    - JSONL files (with streaming)
    - HuggingFace datasets (with streaming)
    - Memory-mapped text files
    - Sequence packing for efficient training
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Dict, Any, Iterator
import random


class JsonlDataset(Dataset):
    """JSONL file dataset for causal language modeling.

    Reads JSONL files where each line has a 'text' field or a task-specific
    field that gets concatenated.

    Args:
        path: Path to JSONL file.
        tokenizer: Tokenizer instance.
        max_seq_len: Maximum sequence length.
        field: Field name to use for text. If None, concatenates all string fields.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 2048,
        field: Optional[str] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.field = field
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load all data from JSONL file."""
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
        return data

    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a data item."""
        if self.field and self.field in item:
            return str(item[self.field])

        # For instruction data, combine relevant fields
        parts = []
        for key in ["instruction", "input", "context", "question",
                     "reasoning", "response", "output", "text",
                     "source", "target", "problem", "derivation", "answer"]:
            if key in item and item[key]:
                parts.append(str(item[key]))
        return " ".join(parts) if parts else ""

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        text = self._extract_text(item)

        # Tokenize with truncation
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(encoded, dtype=torch.long),
        }


class PackedDataset(Dataset):
    """Packed sequence dataset for efficient training.

    Concatenates multiple short sequences into a single long sequence,
    using segment IDs to mark boundaries. This eliminates padding waste
    and improves GPU utilization.

    Args:
        path: Path to JSONL file.
        tokenizer: Tokenizer instance.
        max_seq_len: Maximum packed sequence length.
        pack_factor: Target number of sequences per pack.
        field: Field name to use for text.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 2048,
        pack_factor: int = 4,
        field: Optional[str] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack_factor = pack_factor
        self.field = field
        self.examples = self._load_and_pack()

    def _load_data(self) -> List[str]:
        """Load all text data from JSONL file."""
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
        """Tokenize all texts and filter by length."""
        tokenized = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.max_seq_len:
                tokenized.append(ids)
        return tokenized

    def _pack_sequences(self, tokenized: List[List[int]]) -> List[Dict[str, torch.Tensor]]:
        """Pack multiple sequences into single long sequences.

        Each packed sequence contains:
        - input_ids: concatenated token sequences
        - labels: same as input_ids (with -100 at segment boundaries)
        - segment_ids: which segment each token belongs to
        - position_ids: relative position within each segment
        """
        packed = []
        current_ids = []
        current_segments = []
        current_positions = []
        current_labels = []
        segment_idx = 0
        pos_in_segment = 0

        for ids in tokenized:
            # Check if adding this sequence exceeds max_seq_len
            if len(current_ids) + len(ids) > self.max_seq_len:
                if current_ids:
                    # Save current pack
                    packed.append(self._make_packed_item(
                        current_ids, current_segments,
                        current_positions, current_labels
                    ))
                    current_ids = []
                    current_segments = []
                    current_positions = []
                    current_labels = []
                    segment_idx = 0
                    pos_in_segment = 0
                else:
                    # Single sequence too long, skip it
                    continue

            # Add sequence to current pack
            for token_id in ids:
                current_ids.append(token_id)
                current_segments.append(segment_idx)
                current_positions.append(pos_in_segment)
                # Labels: -100 at segment boundaries (first token of each segment after the first)
                if pos_in_segment == 0 and segment_idx > 0:
                    current_labels.append(-100)
                else:
                    current_labels.append(token_id)
                pos_in_segment += 1

            segment_idx += 1
            pos_in_segment = 0

        # Don't forget the last pack
        if current_ids:
            packed.append(self._make_packed_item(
                current_ids, current_segments,
                current_positions, current_labels
            ))

        return packed

    def _make_packed_item(
        self,
        ids: List[int],
        segments: List[int],
        positions: List[int],
        labels: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Create a packed example dict."""
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
        }

    def _load_and_pack(self) -> List[Dict[str, torch.Tensor]]:
        """Load data and pack sequences."""
        texts = self._load_data()
        tokenized = self._tokenize_texts(texts)
        # Shuffle for better packing
        random.shuffle(tokenized)
        packed = self._pack_sequences(tokenized)
        return packed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class StreamingJsonlDataset(IterableDataset):
    """Streaming JSONL dataset for large files.

    Yields tokenized examples without loading entire file into memory.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 2048,
        field: Optional[str] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.field = field

    def _extract_text(self, item: Dict[str, Any]) -> str:
        if self.field and self.field in item:
            return str(item[self.field])
        parts = []
        for key in ["instruction", "input", "context", "question",
                     "reasoning", "response", "output", "text"]:
            if key in item and item[key]:
                parts.append(str(item[key]))
        return " ".join(parts) if parts else ""

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = self._extract_text(item)
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                if len(encoded) > self.max_seq_len:
                    encoded = encoded[:self.max_seq_len]
                yield {
                    "input_ids": torch.tensor(encoded, dtype=torch.long),
                    "labels": torch.tensor(encoded, dtype=torch.long),
                }


class HFDataset(IterableDataset):
    """HuggingFace dataset wrapper with streaming support.

    Args:
        path: Dataset name or path (e.g., "wikitext", "datasets/synthetic/v1").
        tokenizer: Tokenizer instance.
        max_seq_len: Maximum sequence length.
        split: Dataset split ("train", "validation").
        streaming: Whether to use streaming mode.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 2048,
        split: str = "train",
        streaming: bool = True,
        text_column: str = "text",
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.streaming = streaming
        self.text_column = text_column

        try:
            from datasets import load_dataset
            self.ds = load_dataset(path, split=split, streaming=streaming)
        except Exception:
            # Fallback: treat as JSONL
            from datasets import load_dataset
            self.ds = load_dataset("json", data_files=path, split="train", streaming=streaming)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for item in self.ds:
            text = item.get(self.text_column, "")
            if not text:
                # Try concatenating multiple fields
                parts = []
                for key in ["instruction", "input", "output", "text", "problem", "answer"]:
                    if key in item and item[key]:
                        parts.append(str(item[key]))
                text = " ".join(parts)
            if not text:
                continue

            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > self.max_seq_len:
                encoded = encoded[:self.max_seq_len]

            yield {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "labels": torch.tensor(encoded, dtype=torch.long),
            }


class TextDataset(Dataset):
    """Simple text dataset that tokenizes on-the-fly.

    Supports plain text files (one example per line) or JSONL.
    """

    def __init__(self, path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load(path)

    def _load(self, path: str) -> List[List[int]]:
        texts = []
        if path.endswith(".jsonl"):
            with open(path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                    texts.append(text)
        else:
            with open(path, "r") as f:
                texts = [l.strip() for l in f if l.strip()]

        # Tokenize all
        encoded = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            encoded.append(ids)
        return encoded

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    collator,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Build a DataLoader with efficient settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
