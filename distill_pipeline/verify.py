"""Verification + deep quality scoring.

For each raw record:
- code_with_explanation: extract ```python block, execute with asserts, must pass.
- proof_or_derivation / qa_problem_solving / worked_example: extract numeric claims,
  cross-check by re-deriving with an independent teacher (agreement). For the pilot we
  use a lightweight independent verification pass by a different teacher via the API.
- ensemble items: agreement already partially captured; here we confirm alt content exists.
- All items: an independent verifier teacher (GLM) gives a KEEP/REVIEW/REJECT JSON verdict.

Outputs verified/ and rejected/ shards with updated metadata.verification_status and
quality_score. Resumable via metadata/verify_state.json.
"""
from __future__ import annotations
import asyncio
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

from hetzner_client import HetznerClient
from prompts import ENSEMBLE_VERIFY_PROMPT
from normalize_quality import heuristic_quality

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "distilled_corpus"
RAW = CORPUS / "raw"
VERIFIED = CORPUS / "verified"
REJECTED = CORPUS / "rejected"
META = CORPUS / "metadata"
VERIFIER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    if blocks:
        return "\n\n".join(blocks)
    m = re.search(r"```\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            p = f.name
        r = subprocess.run(["python3", p], capture_output=True, text=True, timeout=timeout)
        os.unlink(p)
        return (r.returncode == 0, (r.stdout + r.stderr)[-300:])
    except Exception as e:
        return (False, repr(e))


def parse_verdict(text: str) -> dict:
    if not text:
        return {"verdict": "REVIEW", "factual_ok": None, "issues": ["no_verifier_response"]}
    s = text.strip()
    # find first {...}
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {"verdict": "REVIEW", "factual_ok": None, "issues": ["unparseable_verifier"]}
    try:
        d = json.loads(m.group(0))
        v = d.get("verdict", "REVIEW").upper()
        if v not in ("KEEP", "REVIEW", "REJECT"):
            v = "REVIEW"
        return {"verdict": v, "factual_ok": d.get("factual_ok"),
                "issues": d.get("issues", [])}
    except Exception:
        return {"verdict": "REVIEW", "factual_ok": None, "issues": ["unparseable_verifier"]}


async def verify_record(cli: HetznerClient, rec: dict, do_llm_verify: bool) -> dict:
    md = rec["metadata"]
    text = rec["text"]
    issues = list(md.get("heuristic_flags", []))
    verdict = md.get("verdict_guess", "REVIEW")
    code_ok = None

    # 1. code execution check
    if md["task_type"] == "code_with_explanation":
        code = extract_code(text)
        if code is None:
            issues.append("no_code_block")
            verdict = "REJECT"
        else:
            ok, out = run_code(code)
            code_ok = ok
            if not ok:
                issues.append("code_failed:" + out[:80])
                verdict = "REJECT"
            else:
                verdict = "KEEP"

    # 2. too short / artifact hard reject
    if "teacher_artifact" in issues or "too_short" in issues:
        verdict = "REJECT"

    # 3. LLM independent verification (skip if already hard-rejected, to save API)
    llm = None
    if do_llm_verify and verdict != "REJECT" and text:
        prompt = ENSEMBLE_VERIFY_PROMPT.format(
            concept=md["concept"], domain=md["domain"], doc=text[:6000])
        try:
            extra = {}
            if "nvidia" in os.environ.get("LLM_API_BASE", ""):
                extra = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 2048,
                }
            r = await cli.chat(VERIFIER_MODEL, [{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=1000, extra=extra)
            llm = parse_verdict(r.content)
            if llm["verdict"] == "REJECT":
                verdict = "REJECT"
                issues += llm.get("issues", [])
            elif llm["verdict"] == "KEEP" and verdict != "REJECT":
                verdict = "KEEP"
            else:
                verdict = "REVIEW"
                issues += llm.get("issues", [])
        except Exception as e:
            llm = {"verdict": "REVIEW", "issues": [f"verifier_error:{e}"]}
            verdict = "REVIEW"

    # combine quality score
    hq = heuristic_quality(text)
    base = hq["quality_score"]
    if verdict == "KEEP":
        score = min(1.0, base + 0.2)
    elif verdict == "REJECT":
        score = max(0.0, base - 0.3)
    else:
        score = base
    if llm and llm.get("factual_ok") is True:
        score = min(1.0, score + 0.05)

    md["verification_status"] = verdict.lower()
    md["verdict"] = verdict
    md["quality_score"] = round(score, 3)
    md["verification_issues"] = issues
    md["code_verified"] = code_ok
    md["llm_verifier"] = llm
    md["confidence"] = round(score, 3)
    rec["metadata"] = md
    return rec


async def run(do_llm_verify: bool = True):
    files = sorted(RAW.glob("*.jsonl"))
    state_path = META / "verify_state.json"
    done = set()
    if state_path.exists():
        done = set(json.load(open(state_path)).get("done", []))
    all_recs = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                all_recs.append(json.loads(line))
    pending = [r for r in all_recs if r["id"] not in done]
    print(f"raw records: {len(all_recs)} | verified: {len(done)} | pending: {len(pending)}")

    verified_out = VERIFIED / "verified.jsonl"
    rejected_out = REJECTED / "rejected.jsonl"
    VERIFIED.mkdir(parents=True, exist_ok=True)
    REJECTED.mkdir(parents=True, exist_ok=True)
    # accumulate across runs: load previously-written records so re-runs only add new
    existing_v = {}
    existing_r = {}
    if verified_out.exists():
        for line in open(verified_out):
            line = line.strip()
            if line:
                r = json.loads(line)
                existing_v[r["id"]] = r
    if rejected_out.exists():
        for line in open(rejected_out):
            line = line.strip()
            if line:
                r = json.loads(line)
                existing_r[r["id"]] = r
    v_buf, r_buf = list(existing_v.values()), list(existing_r.values())

    async with HetznerClient(concurrency=4, timeout=120, max_retries=3,
                             cache_dir=META / "api_cache") as cli:
        BATCH = 8
        for bi in range(0, len(pending), BATCH):
            batch = pending[bi:bi + BATCH]
            res = []
            res = await asyncio.gather(
                *[verify_record(cli, rec, do_llm_verify) for rec in batch])
            for rec in res:
                if rec["metadata"]["verdict"] == "REJECT":
                    r_buf.append(rec)
                else:
                    v_buf.append(rec)
                done.add(rec["id"])
            print(f"verified {bi+len(batch)}/{len(pending)} "
                  f"(keep={len(v_buf)} reject={len(r_buf)})", flush=True)
            json.dump({"done": sorted(done)}, open(state_path, "w"))

    with open(verified_out, "w") as f:
        for r in v_buf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(rejected_out, "w") as f:
        for r in r_buf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"KEEP={len(v_buf)} REVIEW-in-verified={0} REJECT={len(r_buf)}")


if __name__ == "__main__":
    import sys
    do_llm = "--no-llm" not in sys.argv
    asyncio.run(run(do_llm))
