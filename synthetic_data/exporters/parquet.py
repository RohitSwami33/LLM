"""Parquet + Arrow exporters (built on pandas / pyarrow)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..core.schema import Sample


def _to_records(samples: Iterable[Sample]):
    for s in samples:
        rec = s.to_dict()
        # Keep nested metadata JSON-serialisable for columnar stores.
        rec["metadata"] = s.metadata  # pandas/pyarrow handle dict columns natively
        yield rec


def export_parquet(samples: Iterable[Sample], path: str | Path) -> int:
    """Write samples to a Parquet file. Returns the number of rows."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(_to_records(samples)))
    df.to_parquet(path, index=False)
    return len(df)


def export_arrow(samples: Iterable[Sample], path: str | Path) -> int:
    """Write samples to an Apache Arrow (Feather/IPC) file.

    Parquet *is* an Arrow-based format; this variant writes a standalone
    Arrow IPC stream for tooling that expects native ``.arrow`` files.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(_to_records(samples))
    if not records:
        table = pa.table({})
    else:
        table = pa.Table.from_pylist(records)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return len(records)
