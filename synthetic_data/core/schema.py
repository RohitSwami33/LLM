"""Data schemas for the synthetic data generation framework.

Defines the canonical :class:`Sample` produced by the generator and the
decision objects returned by filters/scorers.  Everything is a plain
``dataclass`` so it serialises cleanly to JSON / Arrow / Parquet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    """Whitespace-normalise text for stable duplicate fingerprinting."""
    return " ".join((text or "").split())


@dataclass
class Sample:
    """A single synthetic training example.

    Attributes
    ----------
    id:
        Stable unique id, usually ``<task_type>-<index>``.
    task_type:
        One of the supported generation modes (instruction, reasoning, …).
    template:
        Name of the prompt template that produced this sample.
    prompt:
        The user-facing instruction passed to the *student* model.
    response:
        The primary target completion for the student model.
    reasoning:
        Optional chain-of-thought / derivation / explanation.
    metadata:
        Free-form provenance (model, tokens, cost, raw parsed fields, …).
    created_at:
        UTC ISO timestamp of creation.
    """

    id: str
    task_type: str
    template: str
    prompt: str
    response: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        return cls(**d)

    def fingerprint(self) -> str:
        """SHA-256 of the full content (prompt + response + reasoning).

        Hashing only the response would wrongly collapse samples whose
        answers coincide (e.g. many math/QA items sharing ``"4"``) while
        their questions differ.  Using the whole record keeps unique
        questions and de-duplicates only truly identical samples.
        """
        blob = "\n".join([
            self.prompt or "", self.response or "", self.reasoning or "",
            (self.metadata or {}).get("input", "") or "",
        ])
        return hashlib.sha256(_normalize(blob).encode("utf-8")).hexdigest()


@dataclass
class FilterDecision:
    """Outcome of running one (or many) filters on a :class:`Sample`."""

    keep: bool
    reason: Optional[str] = None
    score: float = 0.0

    @property
    def reject_reason(self) -> Optional[str]:
        return None if self.keep else self.reason


@dataclass
class ScoreResult:
    """Numeric quality signal attached to a sample by a scorer."""

    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Raw result returned by a teacher LLM."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    logprobs: Optional[List[float]] = None
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def dumps(sample: Sample) -> str:
    """Serialise a sample to a JSON line (UTF-8 safe)."""
    return json.dumps(sample.to_dict(), ensure_ascii=False)


def loads(line: str) -> Sample:
    """Parse a JSON line back into a :class:`Sample`."""
    return Sample.from_dict(json.loads(line))
