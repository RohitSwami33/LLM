"""Build a compact grounded preference set for hallucination reduction."""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 7301
OUT = Path("work/training_v2/data/preference_10k.jsonl")
SYSTEM = (
    "Be helpful, precise, and brutally honest about uncertainty. Use supplied "
    "evidence, distinguish evidence from inference, and never invent facts."
)


def item(prompt: str, chosen: str, rejected: str, source: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt.strip()},
        ],
        "chosen": [{"role": "assistant", "content": chosen.strip()}],
        "rejected": [{"role": "assistant", "content": rejected.strip()}],
        "source": source,
    }


def main() -> None:
    rng = random.Random(SEED)
    pools: dict[str, list[dict]] = {}

    h = []
    for x in load_dataset("pminervini/HaluEval", "qa", split="data"):
        knowledge = str(x["knowledge"]).replace("�", "-").strip()[:2200]
        question = str(x["question"]).strip()
        right = str(x["right_answer"]).strip()
        wrong = str(x["hallucinated_answer"]).strip()
        if min(map(len, (knowledge, question, right, wrong))) < 2 or right.casefold() == wrong.casefold():
            continue
        prompt = f"Evidence:\n{knowledge}\n\nQuestion: {question}"
        chosen = f"Based on the supplied evidence: {right}"
        h.append(item(prompt, chosen, wrong, "HaluEval:qa"))
    rng.shuffle(h)
    pools["HaluEval"] = h[:5200]

    s = []
    for x in load_dataset("allenai/sciq", split="train"):
        evidence = str(x["support"]).replace("�", "°").strip()
        question = str(x["question"]).strip()
        correct = str(x["correct_answer"]).strip()
        if correct.casefold() not in evidence.casefold():
            continue
        evidence = evidence[:1800]
        distractors = [str(x[k]).strip() for k in ("distractor1", "distractor2", "distractor3")]
        wrong = rng.choice([d for d in distractors if d and d.casefold() != correct.casefold()])
        prompt = f"Evidence:\n{evidence}\n\nQuestion: {question}"
        s.append(item(prompt, f"Answer: {correct}. The supplied evidence supports this answer.",
                      f"Answer: {wrong}. This is definitely correct.", "SciQ:train"))
    rng.shuffle(s)
    pools["SciQ"] = s[:2500]

    o = []
    for x in load_dataset("allenai/openbookqa", "main", split="train"):
        labels, texts = x["choices"]["label"], x["choices"]["text"]
        answers = dict(zip(labels, texts))
        key = str(x["answerKey"])
        if key not in answers:
            continue
        wrong_keys = [k for k in labels if k != key]
        wrong_key = rng.choice(wrong_keys)
        prompt = "Choose the best answer.\n\n" + str(x["question_stem"]) + "\n" + "\n".join(
            f"{k}) {v}" for k, v in zip(labels, texts)
        )
        o.append(item(prompt, f"Answer: {key}) {answers[key]}",
                      f"Answer: {wrong_key}) {answers[wrong_key]}", "OpenBookQA:train"))
    rng.shuffle(o)
    pools["OpenBookQA"] = o[:1500]

    p = []
    for x in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
        ctx = x.get("context") or {}
        contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
        evidence = " ".join(str(v).strip() for v in contexts if str(v).strip())[:2400]
        decision = str(x["final_decision"]).strip().lower()
        explanation = str(x.get("long_answer") or "").strip()
        if not evidence or decision not in {"yes", "no", "maybe"} or not explanation:
            continue
        wrong = {"yes": "NO", "no": "YES", "maybe": "YES"}[decision]
        prompt = f"Biomedical evidence:\n{evidence}\n\nQuestion: {x['question']}"
        chosen = f"Verdict: {decision.upper()}.\n\n{explanation}"
        rejected = f"Verdict: {wrong}. The evidence proves this conclusively, so no uncertainty is warranted."
        p.append(item(prompt, chosen, rejected, "PubMedQA:pqa_labeled"))
    rng.shuffle(p)
    pools["PubMedQA"] = p[:800]

    rows = [x for pool in pools.values() for x in pool]
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "rows": len(rows),
                      "sources": {k: len(v) for k, v in pools.items()}}, indent=2))


if __name__ == "__main__":
    main()
