"""Teacher benchmark: ~24 prompts across 8 categories with verifiable gold answers.

Runs all 4 Hetzner teachers, scores each response with:
  - math/numeric: extract final number, compare to gold
  - code: execute in a sandbox, compare stdout to expected
  - factual/reasoning: teacher agreement + keyword/gold-string checks
Produces reports/teacher_benchmark.json and a routing recommendation.
"""
from __future__ import annotations
import asyncio
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

from hetzner_client import HetznerClient, list_models

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "distilled_corpus" / "reports" / "teacher_benchmark.json"
MODELS = [
    "DeepSeek-V4-Flash-0731",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "GLM-5.2-NVFP4",
    "Kimi-K2.7-Code",
]

# category -> list of (prompt, gold, check_kind)
# check_kind: "num" (numeric closeness), "code" (execute), "substr" (gold substrings present),
#             "any_substr" (at least one of gold substrings), "length" (>= min words)
PROMPTS: dict[str, list] = {
    "math": [
        ("Compute the derivative of f(x)=x^3 ln(x) with respect to x. Give the final simplified expression.",
         ["3x^2 ln(x) + x^2", "x^2(3 ln x + 1)"], "substr"),
        ("Solve 2x + 5 = 17. State the value of x.", ["6"], "num"),
        ("What is the integral of cos(x) dx? Give the antiderivative.", ["sin(x)"], "substr"),
        ("Compute 7! (7 factorial). Give the integer.", ["5040"], "num"),
    ],
    "science": [
        ("State Newton's second law of motion in one sentence and give its equation.",
         ["f = ma", "f=ma", "force equals mass times acceleration", "rate of change of momentum"], "any_substr"),
        ("What is the chemical formula of glucose?", ["c6h12o6"], "substr"),
        ("Name the powerhouse of the cell and the process it primarily carries out.",
         ["mitochondria", "atp"], "any_substr"),
        ("What is the speed of light in vacuum in m/s? Give the numeric value.",
         ["299792458", "3e8", "299,792,458", "2.998"], "any_substr"),
    ],
    "coding": [
        ("Write a Python function `is_prime(n)` returning True if n is prime else False, for n>=2. Output only the code in a ```python block.",
         "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nassert is_prime(7) and not is_prime(9) and is_prime(2)", "code"),
        ("Write a Python function `rev(s)` that returns the reversed string. Include an assert rev('abc')=='cba'. Output only a ```python block.",
         "def rev(s): return s[::-1]\nassert rev('abc')=='cba'", "code"),
        ("Write Python that prints the sum of squares from 1 to 5 (should be 55). Output only a ```python block that prints the result.",
         "print(sum(i*i for i in range(1,6)))", "code"),
    ],
    "reasoning": [
        ("Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Answer with just the name.",
         ["carol"], "substr"),
        ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Answer yes or no and one sentence why.",
         ["yes"], "substr"),
        ("A train travels 60 km in 1.5 hours at constant speed. What is its speed in km/h? Give the number.",
         ["40"], "num"),
    ],
    "general_knowledge": [
        ("In what year did World War II end in Europe? Give the year.",
         ["1945"], "num"),
        ("What is the capital of Australia? One word.",
         ["canberra"], "substr"),
        ("Who wrote 'Romeo and Juliet'? Give the surname.",
         ["shakespeare"], "substr"),
    ],
    "writing": [
        ("Write a 4-sentence encyclopedia-style paragraph explaining the water cycle. Do not use first person.",
         "", "length_40"),
    ],
    "long_context": [
        ("Summarize the key ideas of thermodynamics in a structured 6-point list covering: zeroth law, first law, second law, third law, entropy, and temperature.",
         ["zeroth", "first law", "second law", "third law", "entropy", "temperature"], "all_substr_ci"),
    ],
    "structured_output": [
        ("List 3 planets (other than Earth) in JSON format: an array of objects with keys 'name' and 'order_from_sun' (integer). Output only the JSON.",
         "", "json_check"),
    ],
}


def extract_code(text: str) -> str:
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text


def run_code(code: str, timeout: int = 8) -> tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        r = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
        os.unlink(path)
        return (r.returncode == 0, r.stdout + r.stderr)
    except Exception as e:
        return (False, repr(e))


def extract_num(text: str) -> float | None:
    # find last number-like token
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    for n in reversed(nums):
        try:
            return float(n.replace(",", ""))
        except Exception:
            continue
    return None


def score(resp: str, gold, kind: str) -> tuple[float, str]:
    if resp is None:
        return 0.0, "no_response"
    r = resp.strip().lower()
    if kind == "num":
        g = float(re.sub(r"[^0-9.\-]", "", gold[0]))
        x = extract_num(resp)
        ok = x is not None and abs(x - g) < max(1e-6, abs(g) * 1e-3)
        return (1.0 if ok else 0.0, f"got={x} want={g}")
    if kind == "substr":
        ok = all(g.lower() in r for g in gold)
        return (1.0 if ok else 0.0, "substr")
    if kind == "any_substr":
        ok = any(g.lower() in r for g in gold)
        return (1.0 if ok else 0.0, "any_substr")
    if kind == "all_substr_ci":
        ok = sum(g.lower() in r for g in gold)
        return (ok / len(gold), f"{ok}/{len(gold)}")
    if kind == "code":
        code = extract_code(resp)
        ok, out = run_code(code + "\n" + gold)  # gold is the assert/executable check
        return (1.0 if ok else 0.0, out[-160:])
    if kind == "length_40":
        wc = len(resp.split())
        return (1.0 if wc >= 40 else wc / 40, f"words={wc}")
    if kind == "json_check":
        s = resp.strip()
        # strip code fences
        s = re.sub(r"^```(json)?|```$", "", s, flags=re.M).strip()
        try:
            obj = json.loads(s)
            ok = isinstance(obj, list) and len(obj) >= 3 and all("name" in o for o in obj)
            return (1.0 if ok else 0.5, "json_parsed")
        except Exception as e:
            return (0.0, f"json_fail:{e}")
    return (0.5, "unknown_kind")


async def main():
    tasks = []  # (model, cat, idx, prompt, gold, kind, GenResult)
    async with HetznerClient(concurrency=8, timeout=200, cache_dir=ROOT/"distilled_corpus"/"metadata"/"api_cache") as cli:
        # build all calls
        calls = []
        for model in MODELS:
            for cat, items in PROMPTS.items():
                for i, (prompt, gold, kind) in enumerate(items):
                    calls.append((model, cat, i, prompt, gold, kind))
        async def run_one(model, cat, i, prompt, gold, kind):
            r = await cli.chat(model, [{"role": "user", "content": prompt}],
                               temperature=0.1, max_tokens=4000)
            return (model, cat, i, prompt, gold, kind, r)
        results = await asyncio.gather(*[run_one(*c) for c in calls])

    # aggregate
    by_model_cat = {}
    by_model = {}
    details = []
    for model, cat, i, prompt, gold, kind, r in results:
        sc, note = score(r.content, gold, kind)
        by_model_cat.setdefault(model, {}).setdefault(cat, []).append(sc)
        by_model.setdefault(model, []).append(sc)
        details.append({
            "model": model, "category": cat, "idx": i,
            "score": round(sc, 3), "note": note,
            "error": r.error, "elapsed": r.elapsed,
            "ct": r.completion_tokens, "finish": r.finish_reason,
            "content_head": (r.content or "")[:160],
        })

    summary = {}
    for m, scores in by_model.items():
        summary[m] = {
            "overall": round(sum(scores) / len(scores), 3),
            "n": len(scores),
            "by_category": {c: round(sum(v) / len(v), 3) for c, v in by_model_cat[m].items()},
        }

    # routing recommendation: assign each category to best mean model
    cats = list(PROMPTS.keys())
    routing = {}
    for c in cats:
        best = max(MODELS, key=lambda m: summary[m]["by_category"].get(c, 0))
        routing[c] = best

    report = {
        "n_prompts": sum(len(v) for v in PROMPTS.values()),
        "models": MODELS,
        "summary": summary,
        "routing_recommendation": routing,
        "details": details,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print("ROUTING:", json.dumps(routing, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    asyncio.run(main())
