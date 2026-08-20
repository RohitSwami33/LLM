"""Prompt templates for generating plain standalone knowledge documents.

Every template instructs the teacher to output CLEAN, standalone, natural-language
document text only — no chat framing, no 'as an AI', no meta commentary, no teacher
identity, no markdown headers like '## Answer'. The output is intended to be used
verbatim (after normalization) as pretraining text.
"""
from __future__ import annotations

COMMON_RULES = (
    "Rules for your output:\n"
    "- Output ONLY the document text itself. No preamble, no closing remarks.\n"
    "- Do NOT mention that you are an AI, do NOT say 'as a model', do NOT reference the question.\n"
    "- Do NOT use phrases like 'Certainly', 'Sure', 'Here is', 'In conclusion', 'According to ...'.\n"
    "- Write as a standalone encyclopedic/educational article, not as a chat reply.\n"
    "- Use clear, precise, factual language. Be accurate. If unsure of a fact, omit it rather than guess.\n"
    "- Use plain prose with minimal markdown. Use LaTeX-style inline math like $E=mc^2$ for equations when helpful.\n"
    "- Do not sign or attribute the text to any person or company.\n"
)

DIFFICULTY_INSTR = {
    1: "Audience: general reader. Keep it simple and factual, one short paragraph.",
    2: "Audience: curious beginner. Explain in a few short paragraphs with one simple example.",
    3: "Audience: high-school / introductory college level. Include definitions, a mechanism/why, and a worked example.",
    4: "Audience: undergraduate level. Include multi-step reasoning, derivations, and edge cases.",
    5: "Audience: advanced undergraduate / graduate. Include formal treatment, assumptions, and nuanced discussion.",
    6: "Audience: expert. Include rigorous synthesis, connections to adjacent fields, and open problems.",
}

LENGTH_INSTR = {
    256: "Length: about 200-280 words.",
    512: "Length: about 450-560 words.",
    1024: "Length: about 900-1100 words.",
    2048: "Length: about 1800-2200 words, organized into a few sections.",
    4096: "Length: about 3600-4400 words, organized into multiple titled sections.",
    8192: "Length: a long multi-section document of about 7000-8500 words.",
    16384: "Length: a very long, detailed multi-section document of about 14000-17000 words.",
    32768: "Length: an extensive, exhaustive multi-section treatise of about 28000-33000 words.",
}


def task_prompt(task_type: str, domain: str, subdomain: str, concept: str,
                difficulty: int, target_tokens: int) -> str:
    d = DIFFICULTY_INSTR.get(difficulty, DIFFICULTY_INSTR[3])
    L = LENGTH_INSTR.get(target_tokens, LENGTH_INSTR[512])
    base = f"{COMMON_RULES}\n{d}\n{L}\n\n"
    c = f"Topic (domain={domain}, subdomain={subdomain}): {concept}.\n\n"
    if task_type == "definition":
        return base + c + "Task: Give a precise one-to-three sentence definition, then one paragraph elaborating."
    if task_type == "beginner_explanation":
        return base + c + "Task: Explain this concept for a beginner using plain language and a concrete everyday example."
    if task_type == "intermediate_explanation":
        return base + c + "Task: Explain this concept at an intermediate level: definition, how/why it works, and one example."
    if task_type == "university_explanation":
        return base + c + "Task: Explain at university level: formal definition, mechanism, derivation where relevant, and a worked example."
    if task_type == "expert_explanation":
        return base + c + "Task: Provide an expert-level treatment: rigorous definition, assumptions, deeper theory, and connections to related concepts."
    if task_type == "mechanism_or_why":
        return base + c + "Task: Explain WHY and HOW this works/occurs, step by step, focusing on the underlying mechanism."
    if task_type == "worked_example":
        return base + c + "Task: Present a fully worked example applying this concept. Show each step and the reasoning behind it. Verify the final result."
    if task_type == "comparison":
        # pick a sibling concept automatically is hard; ask teacher to choose a natural contrast
        return base + c + "Task: Compare and contrast this concept with the most closely related concept(s). Use a structured discussion of similarities and differences."
    if task_type == "common_misconceptions":
        return base + c + "Task: List and correct the most common misconceptions about this concept, explaining the correct understanding for each."
    if task_type == "interdisciplinary_connection":
        return base + c + "Task: Explain how this concept connects to at least two other disciplines or fields. Show the concrete links."
    if task_type == "encyclopedia_article":
        return base + c + "Task: Write an encyclopedia-style article: an opening summary, then sections covering definition, history/origin where relevant, mechanism, significance, and applications."
    if task_type == "qa_problem_solving":
        return base + c + "Task: Pose a non-trivial problem about this concept, then solve it step by step with verifiable reasoning. First state the problem clearly, then solve it."
    if task_type == "proof_or_derivation":
        return base + c + "Task: Present a concise derivation or proof of a key result involving this concept. State the result, then prove/derive it step by step, with intermediate conclusions."
    if task_type == "code_with_explanation":
        return (base + c +
                "Task: Provide a working code example (Python unless another language is clearly more appropriate) that demonstrates this concept. "
                "Include the code in a single ```python block with asserts that verify correctness, followed by a brief explanation and complexity analysis. "
                "The code MUST run without errors and the asserts MUST pass.")
    if task_type == "long_synthesis":
        return base + c + "Task: Write a long-form synthesis that integrates this concept with related ideas into a coherent multi-section educational document."
    if task_type == "reasoning_example":
        return base + c + "Task: Present a reasoning problem about this concept and solve it with concise, verifiable reasoning steps and a final answer."
    return base + c + "Task: Explain this concept clearly and accurately."


ENSEMBLE_VERIFY_PROMPT = (
    "You are verifying a knowledge document for factual and reasoning correctness.\n"
    "Concept: {concept} (domain={domain}).\n"
    "Below is a candidate document. Check it for: factual errors, incorrect math/science, "
    "contradictions, unsupported claims, broken reasoning, and teacher artifacts (e.g. 'as an AI').\n"
    "Respond ONLY with a compact JSON object on a single line:\n"
    '{{"verdict":"KEEP|REVIEW|REJECT","factual_ok":true|false,"issues":["short issue 1",...]}}\n'
    "Do not output anything else.\n\n--- DOCUMENT START ---\n{doc}\n--- DOCUMENT END ---"
)
