"""Build a clean science/factual SFT set without benchmark test leakage.

Only public *training* splits are used.  Every answer is grounded in either the
dataset's support passage or its labelled answer.  Output is JSONL so it can be
audited before tokenization.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM = (
    "Be helpful, precise, and honest. Distinguish evidence from inference. "
    "If the supplied evidence is insufficient, say so plainly and do not invent facts."
)


def row(user: str, answer: str, source: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "source": source,
    }


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    # SciQ train only. The support paragraph makes the answer auditable.
    for x in load_dataset("allenai/sciq", split="train"):
        evidence = str(x["support"]).replace("�", "°").strip()
        answer = str(x["correct_answer"]).strip()
        if not evidence or not answer:
            continue
        user = f"Evidence:\n{evidence}\n\nQuestion: {x['question']}"
        out = f"Answer: {answer}\n\nThe supplied evidence directly supports this answer."
        rows.append(row(user, out, "allenai/sciq:train"))

    # Human-labelled biomedical QA. "maybe" is intentionally preserved as an
    # uncertainty class rather than coerced into yes/no.
    for x in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
        ctx = x.get("context") or {}
        parts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
        evidence = " ".join(str(p).strip() for p in parts if str(p).strip())
        if not evidence:
            continue
        decision = str(x["final_decision"]).strip().lower()
        explanation = str(x.get("long_answer") or "").strip()
        user = f"Biomedical evidence:\n{evidence}\n\nQuestion: {x['question']}"
        out = f"Verdict: {decision.upper()}.\n\n{explanation}"
        if decision == "maybe":
            out += "\n\nThe evidence is not decisive, so a stronger yes/no claim would be unjustified."
        rows.append(row(user, out, "qiaojin/PubMedQA:pqa_labeled/train"))

    # OpenBookQA train only provides compact science knowledge and preserves
    # general multiple-choice ability that the previous adapter regressed on.
    for x in load_dataset("allenai/openbookqa", "main", split="train"):
        choices = x["choices"]
        pairs = list(zip(choices["label"], choices["text"]))
        key = str(x["answerKey"])
        lookup = dict(pairs)
        if key not in lookup:
            continue
        options = "\n".join(f"{k}) {v}" for k, v in pairs)
        user = f"Choose the best answer.\n\n{x['question_stem']}\n{options}"
        rows.append(row(user, f"Answer: {key}) {lookup[key]}", "allenai/openbookqa:main/train"))

    # Exact-text dedupe.
    seen, unique = set(), []
    for x in rows:
        key = x["messages"][1]["content"].casefold()
        if key not in seen:
            seen.add(key)
            unique.append(x)
    rng.shuffle(unique)
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("work/training_v2/data/science_sft.jsonl"))
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=0, help="0 writes the full corpus")
    args = ap.parse_args()
    rows = build(args.seed)
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    counts = {}
    for x in rows:
        counts[x["source"]] = counts.get(x["source"], 0) + 1
    print(json.dumps({"output": str(args.output), "rows": len(rows), "sources": counts}, indent=2))


if __name__ == "__main__":
    main()
