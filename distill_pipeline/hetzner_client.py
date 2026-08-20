"""Asynchronous, resumable, key-safe LLM API client (OpenRouter / Hetzner).

- Reads the API key ONLY from env or a key file. Never logs/prints the key.
- Supports any OpenAI-compatible base URL (OpenRouter, Hetzner) via env overrides:
    LLM_API_BASE   (default: OpenRouter https://openrouter.ai/api/v1)
    LLM_API_KEY     or LLM_API_KEY_FILE (default: ~/.openrouter_api_key)
    LLM_MAX_TOKENS  (per-call max output tokens; default 8000)
- Handles reasoning models (content in `message.content`, thinking in
  `message.reasoning`). Returns content; optionally keeps a trimmed reasoning digest.
- Async with concurrency limit, retries with exponential backoff, per-model timeout,
  on-disk request/response cache (prompt+model+temp hash) for dedup, resumable state.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


def _api_base() -> str:
    return os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")


# Models that emit a separate `reasoning` field and need larger budgets.
REASONING_MODELS = {
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "GLM-5.2-NVFP4",
    "Kimi-K2.7-Code",
}
FLASH_MODELS = {"DeepSeek-V4-Flash-0731"}


def _load_key() -> str:
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key.strip()
    f = os.environ.get("LLM_API_KEY_FILE")
    if f and Path(f).exists():
        return Path(f).read_text().strip()
    # default secure location
    for p in [Path.home() / ".openrouter_api_key", Path.home() / ".hetzner_api_key"]:
        if p.exists():
            return p.read_text().strip()
    raise RuntimeError("No API key found (LLM_API_KEY / ~/.openrouter_api_key / ~/.hetzner_api_key).")


@dataclass
class GenResult:
    model: str
    content: str | None
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    cached: bool = False
    error: str | None = None


@dataclass
class HetznerClient:
    concurrency: int = 1
    timeout: float = 180.0
    max_retries: int = 8
    cache_dir: Path | None = None
    _sem: asyncio.Semaphore = field(init=False, default=None)
    _session: aiohttp.ClientSession = field(init=False, default=None)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.concurrency)
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={"Authorization": f"Bearer {_load_key()}"},
        )
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return Path(self.cache_dir) / f"{h}.json"

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        seed: int | None = None,
        extra: dict | None = None,
    ) -> GenResult:
        is_reasoning = model in REASONING_MODELS
        if max_tokens is None:
            max_tokens = 6000 if is_reasoning else 1500

        req = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            req["seed"] = seed
        if extra:
            req.update(extra)

        cache_key = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, "seed": seed, "extra": extra},
            sort_keys=True,
        )
        cp = self._cache_path(cache_key)
        if cp and cp.exists():
            try:
                d = json.loads(cp.read_text())
                return GenResult(cached=True, **d)
            except Exception:
                pass

        body = json.dumps(req)
        url = f"{_api_base()}/chat/completions"
        post_headers = {"Content-Type": "application/json"}
        last_err = None
        async with self._sem:
            for attempt in range(self.max_retries):
                t0 = time.time()
                try:
                    async with self._session.post(url, data=body, headers=post_headers) as resp:
                        text = await resp.text()
                        elapsed = time.time() - t0
                        if resp.status == 429:
                            # free-tier windows can be hours; sleep until the reset
                            # header if present, else exponential backoff.
                            reset_ms = resp.headers.get("X-RateLimit-Reset")
                            if reset_ms:
                                wait = float(reset_ms) / 1000.0 - time.time()
                                if 0 < wait <= 6 * 3600:
                                    last_err = f"http 429 (window reset in {wait/60:.0f} min)"
                                    await asyncio.sleep(wait)
                                    continue
                            backoff = 2 ** attempt if attempt < 6 else 180
                            last_err = f"http 429 (attempt {attempt+1}, waiting {backoff}s)"
                            await asyncio.sleep(backoff)
                            continue
                        if resp.status >= 500:
                            last_err = f"http {resp.status}"
                            await asyncio.sleep(min(2 ** attempt, 30))
                            continue
                        if resp.status != 200:
                            last_err = f"http {resp.status}: {text[:200]}"
                            # non-retryable client errors (except 429) — break
                            if resp.status in (400, 401, 403, 404):
                                break
                            await asyncio.sleep(min(2 ** attempt, 10))
                            continue
                        data = json.loads(text)
                        choice = data["choices"][0]
                        msg = choice.get("message", {})
                        usage = data.get("usage", {}) or {}
                        out = {
                            "model": model,
                            "content": msg.get("content"),
                            "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                            "finish_reason": choice.get("finish_reason"),
                            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                            "completion_tokens": usage.get("completion_tokens", 0) or 0,
                            "elapsed": round(elapsed, 3),
                            "error": None,
                        }
                        if cp:
                            cp.write_text(json.dumps(out))
                        return GenResult(**out)
                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    last_err = repr(e)
                    await asyncio.sleep(min(2 ** attempt, 20))
        return GenResult(model=model, content=None, reasoning=None,
                         finish_reason=None, prompt_tokens=0, completion_tokens=0,
                         elapsed=0.0, error=last_err)

    async def chat_many(
        self, model: str, items: list[tuple[list[dict], dict]], **kw
    ) -> list[GenResult]:
        tasks = [self.chat(model, msgs, **{**kw, **opts}) for msgs, opts in items]
        return await asyncio.gather(*tasks)


async def list_models() -> list[dict]:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{_api_base()}/models",
                         headers={"Authorization": f"Bearer {_load_key()}"}) as r:
            data = await r.json()
            return data.get("data", [])


if __name__ == "__main__":
    import sys
    models = asyncio.run(list_models())
    for m in models:
        print(m["id"], "ctx=", m.get("max_model_len"))
