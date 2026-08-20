"""Style normalization + lightweight quality heuristics.

Normalization strips teacher artifacts so the student never learns teacher identity or
chat mannerisms. Quality heuristics give fast triage scores; deep verification is done
separately (verify.py) via an independent teacher pass + execution/math checks.
"""
from __future__ import annotations
import re

ARTIFACT_PATTERNS = [
    r"(?im)^as an ai[^.\n]*[.\n]",
    r"(?im)^i am (an|a) (ai|language model|large language)[^\n]*\n",
    r"(?im)^(certainly|sure|of course|absolutely|happy to help|here is|here's|below is)[,!]?[^\n]*\n",
    r"(?im)^according to (deepseek|qwen|glm|kimi|the model)[^\n]*\n",
    r"(?im)^(as a|being an?|since i am) (ai|language model|assistant)[^\n]*\n",
    r"(?im)^(hope this helps|let me know if|feel free to ask|does this help)[^\n]*[.\n]",
    r"(?im)^in conclusion,?[^\n]*\n",
    r"(?im)^note: as an ai[^\n]*\n",
    r"(?im)^<\|[^|]*\|>",
    r"(?im)^\[assistant\][^\n]*\n",
]

CHAT_PREFIX = re.compile(r"^(assistant|user|system)\s*:\s*", re.I)


def normalize(text: str) -> str:
    if not text:
        return ""
    t = text
    for pat in ARTIFACT_PATTERNS:
        t = re.sub(pat, "", t)
    # strip leading "Assistant:" style prefixes
    lines = t.split("\n")
    while lines and (CHAT_PREFIX.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    t = "\n".join(lines)
    # collapse 3+ blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    # strip trailing whitespace lines
    t = t.rstrip() + "\n"
    return t


def _has_ai_phrase(t: str) -> bool:
    low = t.lower()
    for kw in ["as an ai", "as a language model", "i am an ai", "i'm an ai",
               "according to deepseek", "according to qwen", "according to glm"]:
        if kw in low:
            return True
    return False


def heuristic_quality(text: str) -> dict:
    """Fast triage. Returns dict with quality_score (0-1), flags, verdict guess."""
    n_words = len(text.split())
    flags = []
    if _has_ai_phrase(text):
        flags.append("teacher_artifact")
    if n_words < 30:
        flags.append("too_short")
    # repetition: max fraction of any single word
    words = re.findall(r"\w+", text.lower())
    rep = 0.0
    if words:
        from collections import Counter
        c = Counter(words)
        rep = c.most_common(1)[0][1] / len(words)
    if rep > 0.06 and n_words > 100:
        flags.append("repetitive")
    # code present?
    has_code = "```" in text
    score = 0.5
    score += min(0.25, n_words / 4000)  # length bonus
    score += 0.1 if not flags else 0.0
    score -= 0.3 if "teacher_artifact" in flags else 0.0
    score -= 0.2 if "too_short" in flags else 0.0
    score -= 0.15 if "repetitive" in flags else 0.0
    score = max(0.0, min(1.0, score))
    verdict = "REJECT" if ("teacher_artifact" in flags or "too_short" in flags) else \
              ("REVIEW" if flags else "KEEP")
    return {"quality_score": round(score, 3), "verdict_guess": verdict,
            "flags": flags, "n_words": n_words, "has_code": has_code,
            "max_word_freq": round(rep, 4)}
