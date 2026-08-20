"""Generate corpus3 docs via NVIDIA NIM (nemotron-3-super-120b-a12b).

Reads pending_prompts.jsonl, writes raw/corpus3_nvidia_shard_*.jsonl in the exact
schema verify.py/dedup.py/finalize.py expect, and updates the corpus3 state file
so pending_prompts.jsonl can be regenerated on resume.

Usage:
  cd /Users/rohit/Documents/Sem5_project
  NVIDIA_API_KEY=nvapi-... .venv/bin/python distill_pipeline/nvidia_runner.py
"""
import asyncio, json, os, time
from openai import AsyncOpenAI

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
MAX_RETRIES = 12

PROMPTS = os.environ.get("RUNNER_PROMPTS", "distilled_corpus/pending_prompts.jsonl")
OUT_PREFIX = os.environ.get("RUNNER_OUT_PREFIX", "distilled_corpus/raw/corpus3_nvidia_shard")
STATE_PATH = os.environ.get("RUNNER_STATE", "distilled_corpus/metadata/state_corpus3.json")
WORKERS = int(os.environ.get("RUNNER_WORKERS", "4"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_state(done):
    with open(STATE_PATH, "w") as f:
        json.dump({"done": sorted(done)}, f)


async def gen_with_backoff(item_id, prompt, max_tokens):
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=max_tokens,
                timeout=300,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = r.choices[0].message.content or ""
            if content.strip():
                return content
            await asyncio.sleep(2 + attempt * 3)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                await asyncio.sleep(2 + attempt * 3)
                continue
            # 5xx/timeout: retry too
            await asyncio.sleep(1 + attempt * 2)
    return None


async def worker(queue, results, state, mutex, shard_handle):
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
        rec = {
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
        }
        if content:
            async with mutex:
                state.add(item["id"])
                save_state(state)
                shard_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                shard_handle.flush()
        results.append(rec)


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
    shard_path = f"{OUT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(shard_path, "w") as shard_handle:
        workers = [asyncio.create_task(worker(q, results, state, mutex, shard_handle)) for _ in range(WORKERS)]
        await asyncio.gather(*workers)
    good = [r for r in results if r["text"]]
    print(f"total={len(results)} ok={len(good)} failed={len(results)-len(good)}")


if __name__ == "__main__":
    asyncio.run(main())
