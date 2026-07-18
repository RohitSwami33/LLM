"""JSONL exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..core.schema import Sample, dumps


def export(samples: Iterable[Sample], path: str | Path) -> int:
    """Write samples as one JSON object per line. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(dumps(s) + "\n")
            count += 1
    return count


def read(path: str | Path) -> List[Sample]:
    """Read a JSONL file back into a list of :class:`Sample`."""
    from ..core.schema import loads

    path = Path(path)
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(loads(line))
    return out
