"""Data collator for language model training.

Handles padding, attention masks, label shifting, and sequence packing.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class DataCollator:
    """Collator for causal language modeling.

    Pads sequences, creates attention masks, and shifts labels.

    Args:
        pad_token_id: Token ID for padding.
        max_seq_len: Maximum sequence length.
        label_pad_id: Label value for padding positions (-100 by default).
    """

    pad_token_id: int = 0
    max_seq_len: int = 2048
    label_pad_id: int = -100

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of tokenized examples.

        Each example should have an 'input_ids' key (and optionally 'labels').
        """
        input_ids = []
        labels = []
        attention_masks = []

        for item in batch:
            ids = item["input_ids"][:self.max_seq_len]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)

            # Create labels (same as input_ids for CLM)
            if "labels" in item:
                lbl = item["labels"][:self.max_seq_len]
                if isinstance(lbl, list):
                    lbl = torch.tensor(lbl, dtype=torch.long)
            else:
                lbl = ids.clone()

            # Attention mask (1 for real tokens, 0 for padding)
            mask = torch.ones_like(ids)

            input_ids.append(ids)
            labels.append(lbl)
            attention_masks.append(mask)

        # Pad to max length in batch
        input_ids = self._pad_batch(input_ids, self.pad_token_id)
        labels = self._pad_batch(labels, self.label_pad_id)
        attention_masks = self._pad_batch(attention_masks, 0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

    def _pad_batch(
        self, tensors: List[torch.Tensor], pad_value: int
    ) -> torch.Tensor:
        """Pad a list of tensors to the max length in the batch."""
        max_len = max(t.size(0) for t in tensors)
        max_len = min(max_len, self.max_seq_len)

        padded = []
        for t in tensors:
            if t.size(0) > max_len:
                t = t[:max_len]
            elif t.size(0) < max_len:
                pad_len = max_len - t.size(0)
                t = torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)])
            padded.append(t)

        return torch.stack(padded)


@dataclass
class PackedCollator:
    """Collator for packed sequence batches.

    Packed batches contain multiple sequences concatenated together with
    segment IDs to indicate boundaries. This eliminates padding waste.

    Args:
        pad_token_id: Token ID for padding.
        max_seq_len: Maximum sequence length for the packed batch.
        label_pad_id: Label value for padding positions (-100 by default).
    """

    pad_token_id: int = 0
    max_seq_len: int = 2048
    label_pad_id: int = -100

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of packed examples.

        Each example should have:
        - 'input_ids': concatenated token sequences
        - 'labels': matching labels (-100 for ignored positions)
        - 'segment_ids': segment boundaries (0, 1, 2, ...)
        - 'position_ids': relative positions within each segment
        """
        item = batch[0]  # Packed batches have batch_size=1

        input_ids = item["input_ids"][:self.max_seq_len]
        labels = item["labels"][:self.max_seq_len]
        segment_ids = item["segment_ids"][:self.max_seq_len]
        position_ids = item["position_ids"][:self.max_seq_len]

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        if isinstance(segment_ids, list):
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        if isinstance(position_ids, list):
            position_ids = torch.tensor(position_ids, dtype=torch.long)

        # Attention mask (1 for real tokens)
        attention_mask = torch.ones_like(input_ids)

        # Pad if needed
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), self.label_pad_id, dtype=torch.long)])
            segment_ids = torch.cat([segment_ids, torch.full((pad_len,), -1, dtype=torch.long)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        # Add batch dimension
        return {
            "input_ids": input_ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "segment_ids": segment_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
