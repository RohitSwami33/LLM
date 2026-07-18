"""Normalized instruction-dataset exporter.

Emits the canonical training record used by the v1 dataset:

    id, task_type, difficulty, instruction, input, reasoning, output,
    language, estimated_tokens, quality_score, generator, created_at

Supports jsonl / parquet / arrow / hf (Hugging Face Dataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def to_record(sample) -> dict:
    """Map a :class:`Sample` to the normalized instruction record."""
    m = sample.metadata or {}
    return {
        "id": sample.id,
        "task_type": sample.task_type,
        "difficulty": m.get("difficulty", "medium"),
        "instruction": sample.prompt,
        "input": m.get("input", ""),
        "reasoning": sample.reasoning or "",
        "output": sample.response,
        "language": m.get("language", "en"),
        "estimated_tokens": int(m.get("estimated_tokens", 0) or 0),
        "quality_score": float(m.get("quality_score", 0.0) or 0.0),
        "generator": m.get("generator", "unknown"),
        "created_at": sample.created_at,
    }


def _records(samples: Iterable) -> List[dict]:
    return [to_record(s) for s in samples]


def export_jsonl(records: List[dict], path: Path) -> int:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def export_parquet(records: List[dict], path: Path) -> int:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return len(df)


def export_arrow(records: List[dict], path: Path) -> int:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        table = pa.table({})
    else:
        table = pa.Table.from_pylist(records)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return len(records)


def export_hf(records: List[dict], path: Path, push_to_hub: bool = False,
               dataset_name: str = None) -> int:
    from datasets import Dataset

    path = Path(path)
    ds = Dataset.from_list(records)
    ds.save_to_disk(str(path))
    if push_to_hub:
        ds.push_to_hub(dataset_name)
    return len(ds)


def export(samples: Iterable, fmt: str, path: str | Path, **kwargs) -> int:
    """Export normalized instruction records to ``path`` in ``fmt``."""
    records = _records(samples)
    path = Path(path)
    if fmt == "jsonl":
        return export_jsonl(records, path)
    if fmt == "parquet":
        return export_parquet(records, path)
    if fmt == "arrow":
        return export_arrow(records, path)
    if fmt == "hf":
        return export_hf(
            records, path,
            push_to_hub=kwargs.get("push_to_hub", False),
            dataset_name=kwargs.get("dataset_name"),
        )
    raise ValueError(f"Unknown format: {fmt}")
