"""Exporters: JSONL, Parquet, Arrow, Hugging Face datasets.

Use :func:`export` with ``fmt`` in {jsonl, parquet, arrow, hf}.  All exporters
accept an iterable of :class:`~synthetic_data.core.schema.Sample`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from ..core.schema import Sample
from . import jsonl as _jsonl
from . import parquet as _parquet
from . import huggingface as _hf

__all__ = ["export", "get_exporter", "read_jsonl"]


_EXPORTERS: dict[str, Callable] = {
    "jsonl": _jsonl.export,
    "parquet": _parquet.export_parquet,
    "arrow": _parquet.export_arrow,
    "hf": _hf.export,
}


def get_exporter(fmt: str) -> Callable:
    """Return the exporter callable for a format string."""
    if fmt not in _EXPORTERS:
        raise ValueError(
            f"Unknown export format '{fmt}'. Choose from {list(_EXPORTERS)}"
        )
    return _EXPORTERS[fmt]


def export(samples: Iterable[Sample], fmt: str, path: str | Path, **kwargs) -> int:
    """Export ``samples`` to ``path`` in the requested ``fmt``.

    Extra ``kwargs`` (e.g. ``push_to_hub``, ``dataset_name``) are forwarded to
    the underlying exporter.
    """
    exporter = get_exporter(fmt)
    return exporter(samples, path, **kwargs)


def read_jsonl(path: str | Path):
    """Load samples from a JSONL file (convenience re-export)."""
    return _jsonl.read(path)
