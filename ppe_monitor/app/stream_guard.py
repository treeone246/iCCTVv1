"""Async gate to limit concurrent `/ws/stream` pipeline consumers."""

from __future__ import annotations

import asyncio


class StreamClientGate:
    """Concurrency gate for stream clients.

    Limiting concurrent clients prevents duplicate pipeline execution against the
    same video source, which can destabilize OpenCV/FFmpeg decode threads.
    """

    def __init__(self, max_clients: int = 1) -> None:
        self.max_clients = max(1, int(max_clients))
        self._active_clients = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active_clients >= self.max_clients:
                return False
            self._active_clients += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active_clients > 0:
                self._active_clients -= 1

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {
                "active_clients": int(self._active_clients),
                "max_clients": int(self.max_clients),
            }
