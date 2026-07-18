"""Async rate limiter (requests-per-minute) for teacher API calls."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token-bucket style limiter capping requests per rolling minute.

    Used to stay within a provider's RPM quota while maximising throughput.
    Safe for concurrent coroutines (guarded by an ``asyncio.Lock``).
    """

    def __init__(self, rpm: int = 20):
        self.rpm = max(1, int(rpm))
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                # Drop timestamps older than 60s.
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(time.monotonic())
                    return
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
            if wait > 0:
                await asyncio.sleep(wait)
