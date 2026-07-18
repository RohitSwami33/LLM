"""Reusable prompt templates.

A :class:`Template` describes how to *ask the teacher* to synthesise one
training example.  It carries a system prompt, a user prompt with
``{placeholders}``, the expected output format, and the field mapping used to
turn the parsed model output into a :class:`~synthetic_data.core.schema.Sample`.

All templates request **structured JSON** so the output is deterministic and
machine-parseable.  ``parse_response`` extracts the JSON regardless of minor
formatting drift (code fences, prose wrappers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Template:
    """A prompt template for one synthetic-data category."""

    name: str
    task_type: str
    system: str
    user: str
    output_format: str = "json"  # json | code | text
    requires_reasoning: bool = False
    # Field mapping from parsed JSON -> Sample
    prompt_key: str = "prompt"
    response_key: str = "response"
    reasoning_key: Optional[str] = None
    # Optional extra metadata keys copied verbatim into Sample.metadata
    extra_keys: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict:
    """Extract the first JSON object from a model response.

    Handles common failures: markdown code fences, trailing prose, and
    multiple concatenated objects (takes the first balanced-looking one).
    """
    # Strip code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError("No JSON object found in model response")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to repair by trimming to last '}'
        end = candidate.rfind("}")
        if end != -1:
            return json.loads(candidate[: end + 1])
        raise


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, Template] = {}


def _register(t: Template) -> Template:
    TEMPLATES[t.name] = t
    return t


# 1. Instruction following ---------------------------------------------------
_register(Template(
    name="instruction",
    task_type="instruction",
    system=(
        "You are an expert dataset creator. Produce a single, self-contained "
        "instruction-following training example as JSON with keys 'instruction' "
        "and 'response'. The response must fully and correctly satisfy the "
        "instruction. Vary topics and difficulty."
    ),
    user=(
        "Topic: {topic}\n\nGenerate one high-quality instruction-following example. "
        "Return ONLY JSON:\n"
        '{{"instruction": "...", "response": "..."}}'
    ),
    prompt_key="instruction",
    response_key="response",
    extra_keys=["instruction"],
))

# 2. Reasoning -----------------------------------------------------------------
_register(Template(
    name="reasoning",
    task_type="reasoning",
    system=(
        "You are a careful reasoning engine. Given a question, show explicit "
        "step-by-step reasoning and then state the final answer. Respond with "
        "JSON containing 'question', 'reasoning', and 'answer'."
    ),
    user=(
        "Subject area: {topic}\n\nCreate a challenging reasoning question and solve it. "
        "Return ONLY JSON:\n"
        '{{"question": "...", "reasoning": "...", "answer": "..."}}'
    ),
    requires_reasoning=True,
    prompt_key="question",
    response_key="answer",
    reasoning_key="reasoning",
    extra_keys=["question"],
))

# 3. Coding --------------------------------------------------------------------
_register(Template(
    name="coding",
    task_type="coding",
    system=(
        "You are a senior software engineer. Write correct, idiomatic code that "
        "solves the given problem, then explain it. Respond with JSON containing "
        "'problem', 'language', 'code', and 'explanation'."
    ),
    user=(
        "Language: {language}\nDomain: {topic}\n\nInvent a non-trivial coding problem "
        "and provide a working solution. Return ONLY JSON:\n"
        '{{"problem": "...", "language": "...", "code": "...", "explanation": "..."}}'
    ),
    output_format="code",
    prompt_key="problem",
    response_key="code",
    reasoning_key="explanation",
    extra_keys=["problem", "language", "explanation"],
))

# 4. Mathematics ---------------------------------------------------------------
_register(Template(
    name="math",
    task_type="math",
    system=(
        "You are a mathematics tutor. Produce a math problem, show a clear "
        "derivation, and give the final answer. Respond with JSON containing "
        "'problem', 'derivation', and 'answer'."
    ),
    user=(
        "Branch: {topic}\n\nCreate a math problem with a worked derivation. "
        "Return ONLY JSON:\n"
        '{{"problem": "...", "derivation": "...", "answer": "..."}}'
    ),
    requires_reasoning=True,
    prompt_key="problem",
    response_key="answer",
    reasoning_key="derivation",
    extra_keys=["problem"],
))

# 5. Debugging -----------------------------------------------------------------
_register(Template(
    name="debugging",
    task_type="debugging",
    system=(
        "You are a debugging expert. Given buggy code, identify the defect, fix "
        "it, and explain the fix. Respond with JSON containing 'buggy_code', "
        "'fixed_code', and 'explanation'."
    ),
    user=(
        "Language: {language}\n\nWrite a small piece of buggy code and then debug it. "
        "Return ONLY JSON:\n"
        '{{"buggy_code": "...", "fixed_code": "...", "explanation": "..."}}'
    ),
    output_format="code",
    prompt_key="buggy_code",
    response_key="fixed_code",
    reasoning_key="explanation",
    extra_keys=["buggy_code", "explanation", "language"],
))

# 6. Tool use / function calling ----------------------------------------------
_register(Template(
    name="tool_use",
    task_type="function_calling",
    system=(
        "You convert natural-language requests into precise tool calls. Given a "
        "user query, emit the most appropriate function call as JSON with keys "
        "'query', 'function_name', 'arguments' (object), and 'rationale'."
    ),
    user=(
        "Scenario: {topic}\n\nInvent a user request and the correct tool call for it. "
        "Return ONLY JSON:\n"
        '{{"query": "...", "function_name": "...", "arguments": {{}}, '
        '"function_call": "...", "rationale": "..."}}'
    ),
    prompt_key="query",
    response_key="function_call",
    reasoning_key="rationale",
    extra_keys=["query", "function_name", "arguments", "rationale"],
))

# 7. Creative writing ----------------------------------------------------------
_register(Template(
    name="creative_writing",
    task_type="creative_writing",
    system=(
        "You are a published author. Write an original, vivid piece and provide a "
        "short author's note. Respond with JSON containing 'prompt' and 'content'."
    ),
    user=(
        "Theme: {topic}\n\nWrite an original creative piece. Return ONLY JSON:\n"
        '{{"prompt": "...", "content": "..."}}'
    ),
    prompt_key="prompt",
    response_key="content",
    extra_keys=["prompt"],
))

# 8. Classification ------------------------------------------------------------
_register(Template(
    name="classification",
    task_type="classification",
    system=(
        "You are a labelling assistant. Given text and a label set, pick the "
        "correct label and justify it. Respond with JSON containing 'text', "
        "'label', 'label_set', and 'rationale'."
    ),
    user=(
        "Topic: {topic}\n\nGenerate a text snippet, a sensible label set, and the "
        "correct label. Return ONLY JSON:\n"
        '{{"text": "...", "label_set": [...], "label": "...", "rationale": "..."}}'
    ),
    prompt_key="text",
    response_key="label",
    reasoning_key="rationale",
    extra_keys=["text", "label_set", "rationale"],
))

# 9. Extraction ----------------------------------------------------------------
_register(Template(
    name="extraction",
    task_type="extraction",
    system=(
        "You are an information extraction specialist. From a document, extract "
        "structured fields into JSON. Respond with JSON containing 'document' and "
        "'extracted' (an object of fields)."
    ),
    user=(
        "Domain: {topic}\n\nWrite a short document and extract its key fields. "
        "Return ONLY JSON:\n"
        '{{"document": "...", "extracted": {{}}}}'
    ),
    prompt_key="document",
    response_key="extracted",
    extra_keys=["document"],
))

# 10. ReAct --------------------------------------------------------------------
_register(Template(
    name="react",
    task_type="react",
    system=(
        "You reason and act via the ReAct pattern: Thought, Action, Observation, "
        "Answer. Respond with JSON containing 'question', 'trajectory' (the "
        "thought/action/observation trace), and 'answer'."
    ),
    user=(
        "Topic: {topic}\n\nCreate a question requiring tool use and show a ReAct "
        "trace. Return ONLY JSON:\n"
        '{{"question": "...", "trajectory": "...", "answer": "..."}}'
    ),
    requires_reasoning=True,
    prompt_key="question",
    response_key="answer",
    reasoning_key="trajectory",
    extra_keys=["question"],
))

# 11. Reflection ---------------------------------------------------------------
_register(Template(
    name="reflection",
    task_type="reflection",
    system=(
        "You improve your own work through reflection. Given a task, produce an "
        "initial attempt, a critique, and an improved version. Respond with JSON "
        "containing 'task', 'attempt', 'critique', and 'improved'."
    ),
    user=(
        "Task type: {topic}\n\nDemonstrate self-reflection on a task. Return ONLY JSON:\n"
        '{{"task": "...", "attempt": "...", "critique": "...", "improved": "..."}}'
    ),
    requires_reasoning=True,
    prompt_key="task",
    response_key="improved",
    reasoning_key="critique",
    extra_keys=["task", "attempt", "critique"],
))

# 12. Dialogue -----------------------------------------------------------------
_register(Template(
    name="dialogue",
    task_type="dialogue",
    system=(
        "You write realistic multi-speaker dialogues. Respond with JSON containing "
        "'topic' and 'conversation' (a list of {'speaker':..., 'text':...} turns)."
    ),
    user=(
        "Setting: {topic}\n\nWrite a natural 6-10 turn dialogue. Return ONLY JSON:\n"
        '{{"topic": "...", "conversation": [{{"speaker": "...", "text": "..."}}]}}'
    ),
    prompt_key="topic",
    response_key="conversation",
    extra_keys=["topic", "conversation"],
))

# 13. Summarization ------------------------------------------------------------
_register(Template(
    name="summarization",
    task_type="summarization",
    system=(
        "You are a summarization engine. Given an article, produce a faithful, "
        "concise summary. Respond with JSON containing 'article' and 'summary'."
    ),
    user=(
        "Domain: {topic}\n\nWrite a medium-length article and summarise it. "
        "Return ONLY JSON:\n"
        '{{"article": "...", "summary": "..."}}'
    ),
    prompt_key="article",
    response_key="summary",
    extra_keys=["article"],
))

# 14. Translation --------------------------------------------------------------
_register(Template(
    name="translation",
    task_type="translation",
    system=(
        "You are a professional translator. Translate text faithfully while "
        "preserving meaning and tone. Respond with JSON containing 'source', "
        "'target', 'source_lang', and 'target_lang'."
    ),
    user=(
        "Translate from {source_lang} to {target_lang}. Compose an original "
        "sentence/paragraph and its translation. Return ONLY JSON:\n"
        '{{"source": "...", "target": "...", "source_lang": "...", '
        '"target_lang": "..."}}'
    ),
    prompt_key="source",
    response_key="target",
    extra_keys=["source", "source_lang", "target_lang"],
))

# 15. Question answering -------------------------------------------------------
_register(Template(
    name="qa",
    task_type="qa",
    system=(
        "You are a reading-comprehension expert. Given a context passage and a "
        "question, answer strictly from the context. Respond with JSON containing "
        "'context', 'question', and 'answer'."
    ),
    user=(
        "Topic: {topic}\n\nWrite a context passage, a question about it, and the "
        "answer. Return ONLY JSON:\n"
        '{{"context": "...", "question": "...", "answer": "..."}}'
    ),
    prompt_key="question",
    response_key="answer",
    extra_keys=["context", "question"],
))

# 16. SQL generation -----------------------------------------------------------
_register(Template(
    name="sql",
    task_type="sql",
    system=(
        "You are a data engineer. Given a database schema and a natural-language "
        "question, write a correct SQL query and explain it. Respond with JSON "
        "containing 'schema', 'question', 'sql', and 'explanation'."
    ),
    user=(
        "Domain: {topic}\n\nInvent a schema, a question, and the SQL to answer it. "
        "Return ONLY JSON:\n"
        '{{"schema": "...", "question": "...", "sql": "...", "explanation": "..."}}'
    ),
    output_format="code",
    prompt_key="question",
    response_key="sql",
    reasoning_key="explanation",
    extra_keys=["schema", "question", "explanation"],
))

# 17. Multi-turn conversation --------------------------------------------------
_register(Template(
    name="multi_turn",
    task_type="multi_turn",
    system=(
        "You simulate a multi-turn assistant conversation with follow-up questions, "
        "corrections, and clarifications. Respond with JSON containing 'scenario' "
        "and 'turns' (list of {'user':..., 'assistant':...})."
    ),
    user=(
        "Scenario: {topic}\n\nSimulate a 4-6 turn assistant conversation. "
        "Return ONLY JSON:\n"
        '{{"scenario": "...", "turns": [{{"user": "...", "assistant": "..."}}]}}'
    ),
    prompt_key="scenario",
    response_key="turns",
    extra_keys=["scenario", "turns"],
))


def get_template(name: str) -> Template:
    """Return a registered template by name (raises ``KeyError`` if missing)."""
    return TEMPLATES[name]


def list_templates() -> List[str]:
    """Return the names of all registered templates."""
    return sorted(TEMPLATES.keys())
