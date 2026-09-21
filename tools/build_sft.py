#!/usr/bin/env python3
"""Build SFT JSONL from distilled_corpus/final/final_corpus.jsonl.

Output:
  datasets/sft/sft.jsonl      (full shuffled, with train/val split in metadata)
  datasets/sft/train.jsonl
  datasets/sft/val.jsonl

Schema per record: {"prompt": ..., "response": ..., "source_id": ..., "domain": ..., "task_type": ...}
Prompt templating maps metadata.task_type -> instruction prefix.
"""
import json, random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "distilled_corpus" / "final" / "final_corpus.jsonl"
OUT_DIR = REPO / "datasets" / "sft"

TEMPLATES = {
    "educational_explanation": "Explain {concept} in {domain} (difficulty {difficulty}/6).",
    "encyclopedia_article": "Write an encyclopedia article about {concept} in {domain}.",
    "worked_example": "Provide a worked example for {concept} ({domain}).",
    "proof_or_derivation": "Provide a proof or derivation for {concept} ({domain}).",
    "qa_problem_solving": "Solve the following problem about {concept} ({domain}): {stem}",
    "comparison": "Compare and contrast aspects of {concept} in {domain}.",
    "long_synthesis": "Write a comprehensive synthesis on {concept} ({domain}).",
    "code_with_explanation": "Explain and provide code for {concept} ({domain}).",
    "reasoning_example": "Walk through the reasoning for {concept} ({domain}) step by step.",
}

def prompt_for(rec):
    meta = rec.get("metadata", {})
    task = meta.get("task_type", "educational_explanation")
    domain = meta.get("domain", "general")
    concept = meta.get("concept", "the topic")
    difficulty = meta.get("difficulty", 3)
    tmpl = TEMPLATES.get(task, "Explain {concept} in {domain}.")
    text = rec.get("text", "")
    stem = text.split("\n\n")[0][:400].strip() if text else concept
    # ensure prompt is not too long
    try:
        p = tmpl.format(concept=concept, domain=domain, difficulty=difficulty, stem=stem)
    except Exception:
        p = f"Explain {concept} in {domain}."
    return p

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--check", action="store_true", help="validate without writing")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}")

    records = []
    with open(SRC) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
            except: continue
            txt=obj.get("text")
            if not txt or len(txt.strip())<50:
                continue
            meta=obj.get("metadata", {})
            prompt=prompt_for(obj)
            records.append({
                "prompt": prompt,
                "response": txt.strip(),
                "source_id": obj.get("id",""),
                "domain": meta.get("domain",""),
                "task_type": meta.get("task_type",""),
            })

    print(f"Loaded {len(records)} SFT pairs from {SRC}")
    if len(records) < 100:
        raise SystemExit(f"Too few records: {len(records)}")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records)*args.val_frac))
    n_val = min(n_val, 50)
    val = records[:n_val]
    train = records[n_val:]

    # quick stats
    from collections import Counter
    print(f"Train: {len(train)}  Val: {len(val)}  (val_frac={args.val_frac:.2%})")
    print("Domains:", Counter(r["domain"] for r in records))
    print("Task types:", Counter(r["task_type"] for r in records))
    avg_len = sum(len(r["response"]) for r in records)/len(records)
    print(f"Avg response chars: {avg_len:.0f}  (~{avg_len/4:.0f} tokens est.)")

    if args.check:
        # validate JSONL + prompt/response sanity
        for r in records[:3]:
            assert r["prompt"] and r["response"]
            print(f"  sample prompt: {r['prompt'][:120]} ... -> response {len(r['response'])} chars")
        print("CHECK OK")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    def write(path, rows):
        with open(path, "w") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False)+"\n")
        print(f"Wrote {len(rows)} -> {path} ({path.stat().st_size/1e6:.2f} MB)")

    write(OUT_DIR/"sft.jsonl", records)
    write(OUT_DIR/"train.jsonl", train)
    write(OUT_DIR/"val.jsonl", val)
    # metadata
    with open(OUT_DIR/"sft_meta.json","w") as f:
        json.dump({"total":len(records),"train":len(train),"val":len(val),"seed":args.seed,"source":str(SRC)}, f, indent=2)
    print("Done.")

if __name__=="__main__":
    main()
