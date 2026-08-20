"""Generation orchestrator.

- Loads concept seeds + routing.
- Samples (concept, task_type, difficulty, target_length) according to generation_config
  distributions.
- Routes to teacher(s); for ensemble items queries multiple teachers independently.
- Normalizes output, runs heuristic quality, writes raw shards + provenance.
- Resumable: tracks completed item ids in metadata/state.json; skips on restart.
"""
from __future__ import annotations
import asyncio
import json
import os
import random
import time
from pathlib import Path

from hetzner_client import HetznerClient
from prompts import task_prompt
from normalize_quality import normalize, heuristic_quality

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "distilled_corpus"
SEEDS = CORPUS / "seeds" / "concept_catalog.json"
CFG = json.load(open(CORPUS / "generation_config.json"))
ROUTING = json.load(open(CORPUS / "reports" / "routing_decision.json"))["routing"]
RAW = CORPUS / "raw"
META = CORPUS / "metadata"
SHARD_SIZE = 200

DOMAIN_ROUTE = ROUTING["primary_by_domain"]
TASK_ROUTE = ROUTING["by_task_type"]
DEFAULT_TEACHER = ROUTING["high_throughput_default"]
ENSEMBLE_TEACHERS = ROUTING["ensemble_members"]

TASK_TYPES = list(CFG["task_type_distribution_pct"].keys())
TASK_WEIGHTS = list(CFG["task_type_distribution_pct"].values())
DIFFS = [1, 2, 3, 4, 5, 6]
DIFF_WEIGHTS = list(CFG["difficulty_distribution_pct"].values())
LEN_BUCKETS = [int(k) for k in CFG["context_length_distribution_pct"].keys()]
LEN_WEIGHTS = list(CFG["context_length_distribution_pct"].values())


def weighted_choice(options, weights, rng):
    return rng.choices(options, weights=weights, k=1)[0]


def route_teacher(domain: str, task_type: str) -> str:
    force = os.environ.get("DISTILL_FORCE_TEACHER")
    if force:
        return force
    if task_type in TASK_ROUTE:
        return TASK_ROUTE[task_type]
    return DOMAIN_ROUTE.get(domain, DEFAULT_TEACHER)


def plan_items(n: int, seed: int = 0, state_key: str = "corpus") -> list[dict]:
    rng = random.Random(seed)
    seeds = json.load(open(SEEDS))
    # weight seeds by inverse domain count to avoid technology dominating planning
    # (we sample uniformly over concepts but cap per domain to honor mixture)
    items = []
    for i in range(n):
        s = rng.choice(seeds)
        tt = weighted_choice(TASK_TYPES, TASK_WEIGHTS, rng)
        diff = weighted_choice(DIFFS, DIFF_WEIGHTS, rng)
        # bias length: long_synthesis/encyclopedia/expert -> longer buckets
        if tt in ("long_synthesis", "encyclopedia_article"):
            tl = rng.choices([1024, 2048, 4096, 8192], weights=[2, 3, 3, 2])[0]
        elif tt in ("code_with_explanation", "qa_problem_solving", "proof_or_derivation"):
            tl = rng.choices([512, 1024, 2048], weights=[3, 3, 2])[0]
        else:
            tl = weighted_choice(LEN_BUCKETS, LEN_WEIGHTS, rng)
        items.append({
            "id": f"{state_key}-{i:06d}",
            "seed_id": s["id"],
            "domain": s["domain"],
            "subdomain": s["subdomain"],
            "concept": s["concept"],
            "task_type": tt,
            "difficulty": diff,
            "target_tokens": tl,
            "knowledge_type": s.get("knowledge_type", "time_invariant"),
        })
    return items


def is_ensemble(domain: str, task_type: str) -> bool:
    if task_type == "definition" and domain in ("physics", "chemistry", "biology", "mathematics"):
        return True
    if task_type == "proof_or_derivation":
        return True
    return False


async def generate_item(cli: HetznerClient, item: dict) -> dict:
    prompt = task_prompt(item["task_type"], item["domain"], item["subdomain"],
                         item["concept"], item["difficulty"], item["target_tokens"])
    teacher = route_teacher(item["domain"], item["task_type"])
    ensemble = is_ensemble(item["domain"], item["task_type"])
    teachers_used = [teacher]
    # Reasoning models (incl. DeepSeek-Flash, which emits a `reasoning` field) need a
    # big budget so thinking doesn't consume all tokens and leave `content` empty.
    budget = int(os.environ.get("DISTILL_MAX_TOKENS", "3500"))
    budget = max(budget, item["target_tokens"] + 1500)
    extra = {}
    if "nvidia" in os.environ.get("LLM_API_BASE", "") and "nemotron" in teacher.lower():
        # NVIDIA Nemotron accepts native thinking controls; other NVIDIA models
        # (e.g. z-ai/glm-5.2) must NOT get these or they spend the whole budget
        # thinking and return empty content.
        extra = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": int(os.environ.get("DISTILL_REASONING_BUDGET", "16384")),
        }
    r = await cli.chat(teacher, [{"role": "user", "content": prompt}],
                       temperature=0.5, max_tokens=budget, extra=extra)
    content = r.content
    # fallback: if content empty but reasoning present, the budget was too small for a
    # reasoning model; retry once with a larger budget.
    if not content and not r.error and (r.reasoning or r.completion_tokens >= budget - 5):
        r = await cli.chat(teacher, [{"role": "user", "content": prompt}],
                           temperature=0.5, max_tokens=budget + 2000)
        content = r.content
    extra = {}
    if ensemble and r.content:
        # query a second independent teacher for agreement
        alt = [t for t in ENSEMBLE_TEACHERS if t != teacher]
        alt_teacher = alt[0] if alt else None
        if alt_teacher:
            r2 = await cli.chat(alt_teacher, [{"role": "user", "content": prompt}],
                                temperature=0.5, max_tokens=budget)
            teachers_used.append(alt_teacher)
            extra["alt_content_len"] = len(r2.content or "")
            extra["alt_error"] = r2.error
    norm = normalize(content or "")
    hq = heuristic_quality(norm)
    rec = {
        "id": item["id"],
        "text": norm,
        "raw_content": content,
        "metadata": {
            "id": item["id"],
            "domain": item["domain"],
            "subdomain": item["subdomain"],
            "concept": item["concept"],
            "source_type": "synthetic",
            "source": f"teacher:{','.join(teachers_used)}",
            "difficulty": item["difficulty"],
            "task_type": item["task_type"],
            "teacher": teacher,
            "teacher_models": teachers_used,
            "verification_status": "pending",
            "quality_score": hq["quality_score"],
            "confidence": None,
            "timestamp": time.strftime("%Y-%m-%d"),
            "knowledge_date": "time_invariant",
            "context_length": item["target_tokens"],
            "language": "en",
            "knowledge_type": item["knowledge_type"],
            "heuristic_flags": hq["flags"],
            "verdict_guess": hq["verdict_guess"],
            "n_words": hq["n_words"],
        },
        "provenance": {
            "id": item["id"],
            "seed_id": item["seed_id"],
            "prompt_concept": item["concept"],
            "teachers": teachers_used,
            "completion_tokens": r.completion_tokens,
            "elapsed": r.elapsed,
            "cached": r.cached,
            "error": r.error,
            "ensemble": ensemble,
            **extra,
        },
    }
    return rec


def write_shard(records: list[dict], path: Path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def run(n: int, state_key: str = "pilot"):
    items = plan_items(n, seed=hash(state_key) & 0xFFFF, state_key=state_key)
    state_path = META / f"state_{state_key}.json"
    done = set()
    if state_path.exists():
        done = set(json.load(open(state_path)).get("done", []))
    pending = [it for it in items if it["id"] not in done]
    print(f"planned {len(items)} | done {len(done)} | pending {len(pending)}")

    records = []
    shard_idx = 0
    written = 0
    conc = int(os.environ.get("DISTILL_CONCURRENCY", "2"))
    to = int(os.environ.get("DISTILL_TIMEOUT", "70"))
    async with HetznerClient(concurrency=conc, timeout=to, max_retries=3,
                             cache_dir=META / "api_cache") as cli:
        # process in small batches to write shards progressively
        BATCH = int(os.environ.get("DISTILL_BATCH", "4"))
        for bi in range(0, len(pending), BATCH):
            batch = pending[bi:bi + BATCH]
            res = await asyncio.gather(*[generate_item(cli, it) for it in batch])
            # keep only records with text; failed/empty calls are NOT marked done so
            # they retry on the next resume (rate-limit windows)
            res = [r for r in res if r.get("text")]
            records.extend(res)
            done.update(r["id"] for r in res)
            ok = len(res)
            print(f"batch {bi//BATCH+1}: {ok}/{len(batch)} produced text", flush=True)
            # write a shard per batch so killing never loses completed records
            shard_idx += 1
            write_shard(records, RAW / f"{state_key}_shard_{shard_idx:04d}.jsonl")
            with open(META / f"provenance_{state_key}_{shard_idx:04d}.jsonl", "a") as pf:
                for r in records:
                    pf.write(json.dumps(r["provenance"], ensure_ascii=False) + "\n")
            written += len(records)
            print(f"shard {shard_idx}: wrote {len(records)} (total {written}/{len(pending)})", flush=True)
            records = []
            # save state
            json.dump({"done": sorted(done)}, open(state_path, "w"))
    print(f"done. total pending processed: {len(done) - (len(items)-len(pending))}/{len(items)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    key = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    asyncio.run(run(n, key))
