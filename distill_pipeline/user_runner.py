"""Generate corpus3 docs via OpenRouter (dots-studio/dots-3-note-preview:free).

Reads pending_prompts.jsonl, writes raw/corpus3_user_shard_*.jsonl in the exact
schema verify.py/dedup.py/finalize.py expect, and updates the corpus3 state file
so pending_prompts.jsonl can be regenerated on resume.

Usage:
  cd /Users/rohit/Documents/Sem5_project
  OPENROUTER_API_KEY=sk-or-... .venv/bin/python distill_pipeline/user_runner.py
"""
import asyncio, json, os, sys, time
from openai import AsyncOpenAI

# api key, model, and runtime settings
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "dots-studio/dots-3-note-preview:free"
MAX_RETRIES = 12

PROMPTS = os.environ.get("RUNNER_PROMPTS", "distilled_corpus/pending_prompts.jsonl")
OUT_PREFIX = os.environ.get("RUNNER_OUT_PREFIX", "distilled_corpus/raw/corpus3_user_shard")
STATE_PATH = os.environ.get("RUNNER_STATE", "distilled_corpus/metadata/state_corpus3.json")
WORKERS = int(os.environ.get("RUNNER_WORKERS", "2"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_state(done):
    with open(STATE_PATH, "w") as f:
        json.dump({"done": sorted(done)}, f)


async def gen_with_backoff(item_id, prompt, max_tokens, attempt=0):
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=max_tokens,
                timeout=300,
                extra_body={"reasoning": {"effort": "none"}},
            )
            content = r.choices[0].message.content or ""
            if content.strip():
                return content
            # free endpoints sometimes return empty content under load: retry
            await asyncio.sleep(2 + attempt * 3)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                await asyncio.sleep(2 + attempt * 3)  # concurrency cap / rate limit
                continue
            # real failure (5xx, timeout, ...) -> retry too; free endpoints are flaky
            await asyncio.sleep(1 + attempt * 2)
    return None


async def worker(queue, results, state, mutex):
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if item["id"] in state:
            continue
        t0 = time.time()
        content = await gen_with_backoff(item["id"], item["prompt"], item["target_tokens"] + 1500)
        elapsed = round(time.time() - t0, 1)
        if content:
            async with mutex:
                state.add(item["id"])
                save_state(state)
        results.append({
            "id": item["id"],
            "text": content or "",
            "raw_content": content,
            "metadata": {
                "id": item["id"],
                "domain": item["domain"],
                "subdomain": item["subdomain"],
                "concept": item["concept"],
                "source_type": "synthetic",
                "source": f"teacher:{MODEL}",
                "difficulty": item["difficulty"],
                "task_type": item["task_type"],
                "teacher": MODEL,
                "teacher_models": [MODEL],
                "verification_status": "pending",
                "quality_score": None,
                "confidence": None,
                "timestamp": time.strftime("%Y-%m-%d"),
                "knowledge_date": "time_invariant",
                "context_length": item["target_tokens"],
                "language": "en",
                "knowledge_type": item["knowledge_type"],
                "heuristic_flags": [],
                "verdict_guess": "KEEP",
                "n_words": 0,
            },
            "provenance": {
                "id": item["id"],
                "seed_id": item["seed_id"],
                "prompt_concept": item["concept"],
                "teachers": [MODEL],
                "completion_tokens": 0,
                "elapsed": elapsed,
                "cached": False,
                "error": None,
                "ensemble": False,
            },
        })


async def main():
    items = [json.loads(l) for l in open(PROMPTS) if l.strip()]
    state = load_state()
    pending = [it for it in items if it["id"] not in state]
    print(f"prompts={len(items)} done={len(state)} pending={len(pending)}")
    if not pending:
        print("nothing to do")
        return

    q = asyncio.Queue()
    for it in pending:
        q.put_nowait(it)
    results = []
    mutex = asyncio.Lock()
    # OpenRouter free tier: 2 concurrent is safe; bump to 4 if no 429s.
    workers = [asyncio.create_task(worker(q, results, state, mutex)) for _ in range(WORKERS)]
    await asyncio.gather(*workers)
    good = [r for r in results if r["text"]]
    with open(f"{OUT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total={len(results)} ok={len(good)} failed={len(results)-len(good)}")


if __name__ == "__main__":
    asyncio.run(main())
