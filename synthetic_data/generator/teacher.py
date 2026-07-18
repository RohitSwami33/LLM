"""Teacher LLM client.

A thin, async, retry-aware wrapper around an OpenAI-compatible chat API
(DeepSeek by default).  It tracks token usage and estimated cost and can
return token-level log-probs for downstream confidence scoring.

Extending to a new provider
---------------------------
Subclass :class:`BaseTeacher` and implement :meth:`BaseTeacher.generate`.  The
rest of the pipeline only depends on the ``GenerationResult`` contract.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.retry import retry_async
from ..core.schema import GenerationResult


class BaseTeacher(ABC):
    """Abstract interface every teacher model must satisfy."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        template: object = None,
        seed: object = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate a completion for ``prompt`` and return a result.

        ``template``/``seed`` are optional hooks used by the offline teacher to
        synthesise structured, verifiable samples without an LLM call.
        """
        raise NotImplementedError


class DeepSeekTeacher(BaseTeacher):
    """Async client for the DeepSeek chat API (OpenAI-compatible)."""

    def __init__(self, config) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set the environment variable "
                f"'{config.api_key_env}' for provider '{config.provider}'."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'openai' package is required for the DeepSeek teacher. "
                "Install it with: pip install openai"
            ) from exc

        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
        self.model = config.model

    # ------------------------------------------------------------------ #
    @retry_async(
        retries=5,
        backoff=1.0,
        multiplier=2.0,
        exceptions=(Exception,),
    )
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        template: object = None,
        seed: object = None,
        **kwargs,
    ) -> GenerationResult:
        cfg = self.config
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "top_p": top_p if top_p is not None else cfg.top_p,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
            "timeout": cfg.timeout,
        }
        if cfg.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (
            prompt_tokens / 1_000_000 * cfg.input_price_per_1m
            + completion_tokens / 1_000_000 * cfg.output_price_per_1m
        )

        logprobs: Optional[List[float]] = None
        if cfg.logprobs and getattr(choice, "logprobs", None) is not None:
            logprobs = [
                tp.logprob
                for token in (choice.logprobs.content or [])
                if (tp := token) is not None and tp.logprob is not None
            ]

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            logprobs=logprobs,
            model=self.model,
            raw={"id": getattr(resp, "id", None)},
        )


def build_teacher(config) -> BaseTeacher:
    """Factory returning the configured teacher implementation."""
    if config.teacher.provider in ("deepseek", "openai", "openai-compatible"):
        return DeepSeekTeacher(config.teacher)
    if config.teacher.provider == "offline":
        from .offline import OfflineTeacher

        return OfflineTeacher(config.teacher)
    raise ValueError(f"Unsupported teacher provider: {config.teacher.provider}")
