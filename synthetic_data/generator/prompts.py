"""Prompt assembly: seed materials, message building, sample construction.

Given a task type and a template, this module:

* draws a *seed* (topic / language pair / index) for diversity,
* renders the template's system + user messages,
* parses the teacher response into structured fields, and
* assembles a :class:`~synthetic_data.core.schema.Sample`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.schema import Sample
from .templates import Template, extract_json, get_template


# Curated topic pools per task type for diversity without external data.
_TOPICS: Dict[str, List[str]] = {
    "instruction": [
        "productivity", "cooking", "personal finance", "travel planning",
        "fitness", "software setup", "gardening", "career advice",
    ],
    "reasoning": [
        "logic puzzles", "probability", "spatial reasoning", "deductive reasoning",
        "common sense", "strategy games",
    ],
    "coding": [
        "web scraping", "caching", "concurrency", "parsing", "data structures",
        "algorithms", "networking", "testing",
    ],
    "math": [
        "algebra", "geometry", "calculus", "number theory", "statistics",
        "combinatorics", "linear algebra",
    ],
    "debugging": ["Python", "TypeScript", "Rust", "Go", "SQL"],
    "function_calling": [
        "weather lookup", "calendar booking", "database query", "payment",
        "translation", "search",
    ],
    "creative_writing": [
        "science fiction", "mystery", "flash fiction", "poetry", "memoir",
    ],
    "classification": [
        "sentiment", "topic", "intent", "spam", "toxicity",
    ],
    "extraction": [
        "invoices", "resumes", "legal clauses", "medical records", "recipes",
    ],
    "react": ["research", "troubleshooting", "planning", "decision making"],
    "reflection": ["essay writing", "code review", "design critique"],
    "dialogue": [
        "customer support", "job interview", "negotiation", "casual chat",
    ],
    "summarization": [
        "news", "scientific abstract", "meeting notes", "technical docs",
    ],
    "translation": [],
    "qa": [
        "history", "biology", "geography", "technology", "literature",
    ],
    "sql": [
        "e-commerce", "healthcare", "analytics", "logistics", "social media",
    ],
    "multi_turn": [
        "technical support", "language tutoring", "brainstorming", "onboarding",
    ],
}

_LANG_PAIRS: List[tuple] = [
    ("English", "French"), ("English", "German"), ("English", "Spanish"),
    ("English", "Japanese"), ("French", "English"), ("German", "English"),
    ("Spanish", "English"), ("Chinese", "English"),
]

_CODE_LANGS = ["Python", "JavaScript", "Java", "C++", "SQL"]


@dataclass
class Seed:
    """A concrete set of variable bindings for one generation request."""

    idx: int
    task_type: str
    topic: str = ""
    language: str = ""
    source_lang: str = ""
    target_lang: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "topic": self.topic,
            "language": self.language,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }


class SeedProvider:
    """Deterministically supplies diverse seeds for a task type."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def get(self, task_type: str, idx: int) -> Seed:
        topics = _TOPICS.get(task_type, [])
        topic = self._rng.choice(topics) if topics else f"topic-{idx}"
        lang = self._rng.choice(_CODE_LANGS) if task_type in ("coding", "debugging") else ""
        if task_type == "translation":
            pair = self._rng.choice(_LANG_PAIRS)
            return Seed(idx=idx, task_type=task_type, source_lang=pair[0], target_lang=pair[1])
        return Seed(idx=idx, task_type=task_type, topic=topic, language=lang)


def render(template: Template, seed: Seed) -> tuple[str, str]:
    """Render the system + user messages for a template/seed pair."""
    try:
        user = template.user.format(**seed.as_dict())
    except KeyError:
        # Fallback: ignore missing placeholders (e.g. language for non-code tasks)
        user = template.user
        for key, val in seed.as_dict().items():
            user = re.sub(r"\{" + key + r"\}", str(val), user)
    return template.system, user


def parse_response(template: Template, text: str) -> Dict[str, Any]:
    """Parse a teacher response into structured fields using the template."""
    if template.output_format == "text":
        return {"text": text.strip()}
    return extract_json(text)


def build_sample(
    task_type: str,
    template: Template,
    seed: Seed,
    fields: Dict[str, Any],
    result,
) -> Sample:
    """Construct a :class:`Sample` from parsed fields + generation metadata."""
    # Allow generators to supply an explicit instruction/input mapping so the
    # canonical training fields (instruction, input) are sensible per task.
    prompt = str(fields.get("instruction") or fields.get(template.prompt_key, "") or "")
    response = str(fields.get(template.response_key, "") or "")
    reasoning = (
        str(fields.get(template.reasoning_key, "") or "")
        if template.reasoning_key
        else None
    )

    metadata: Dict[str, Any] = {
        "model": getattr(result, "model", ""),
        "generator": fields.get("generator", getattr(result, "model", "unknown")),
        "template": template.name,
        "finish_reason": getattr(result, "finish_reason", ""),
        "prompt_tokens": getattr(result, "prompt_tokens", 0),
        "completion_tokens": getattr(result, "completion_tokens", 0),
        "cost_usd": getattr(result, "cost_usd", 0.0),
        "seed_idx": seed.idx,
        "difficulty": fields.get("difficulty", "medium"),
        "language": fields.get("language", "en"),
        "input": fields.get("input", ""),
        "estimated_tokens": fields.get(
            "estimated_tokens", getattr(result, "completion_tokens", 0)
        ),
    }
    # Copy through any extra keys requested by the template.
    for key in template.extra_keys:
        if key in fields:
            metadata[f"field__{key}"] = fields[key]
    # Persist the full parsed structure for flexible downstream use.
    metadata["fields"] = fields

    return Sample(
        id=f"{task_type}-{seed.idx}",
        task_type=task_type,
        template=template.name,
        prompt=prompt,
        response=response,
        reasoning=reasoning,
        metadata=metadata,
    )
