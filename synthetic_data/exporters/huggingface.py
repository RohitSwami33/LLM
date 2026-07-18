"""Hugging Face ``datasets`` exporter (save_to_disk / push_to_hub)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.schema import Sample


def export(samples: Iterable[Sample], path: str | Path, push_to_hub: bool = False,
           dataset_name: str | None = None) -> int:
    """Save samples as a HF :class:`datasets.Dataset` and optionally push.

    Returns the number of rows exported.
    """
    from datasets import Dataset

    path = Path(path)
    rows = [s.to_dict() for s in samples]
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(path))
    if push_to_hub:
        if not dataset_name:
            raise ValueError("dataset_name is required when push_to_hub=True")
        ds.push_to_hub(dataset_name)
    return len(rows)
