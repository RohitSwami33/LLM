"""Async retry / backoff utilities for resilient API calls."""

from __future__ import annotations

import asyncio
import functools
from typing import Awaitable, Callable, Iterable, Tuple, Type


def retry_async(
    retries: int = 5,
    backoff: float = 1.0,
    multiplier: float = 2.0,
    jitter: float = 0.3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Decorate an async function with exponential backoff + jitter.

    Parameters
    ----------
    retries:
        Maximum number of *additional* attempts after the first failure.
    backoff:
        Initial delay in seconds.
    multiplier:
        Growth factor applied to the delay after each failure.
    jitter:
        Randomisation factor (fraction of the delay) to avoid thundering herd.
    exceptions:
        Exception types that should trigger a retry.
    """

    def decorator(fn: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:  # pragma: no cover - depends on API
                    if attempt >= retries:
                        raise
                    import random

                    sleep_for = delay * (1 + random.uniform(-jitter, jitter))
                    await asyncio.sleep(max(0.0, sleep_for))
                    delay *= multiplier

        return wrapper

    return decorator
