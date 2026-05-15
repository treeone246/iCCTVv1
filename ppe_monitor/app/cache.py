"""TTL cache for verifier decisions keyed by person and PPE item."""

from dataclasses import dataclass
from time import monotonic
from typing import Dict, Optional, Tuple

from .schemas import VerifierResult


@dataclass
class CacheEntry:
    """Internal cache entry with expiration timestamp."""

    result: VerifierResult
    expires_at: float


class VerifierCache:
    """Small in-memory verifier cache with explicit per-entry TTL."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[int, str], CacheEntry] = {}

    def get(self, person_id: int, item: str) -> Optional[VerifierResult]:
        key = (person_id, item)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if monotonic() >= entry.expires_at:
            self._entries.pop(key, None)
            return None
        return entry.result

    def put(self, person_id: int, item: str, result: VerifierResult, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            self._entries.pop((person_id, item), None)
            return
        self._entries[(person_id, item)] = CacheEntry(
            result=result, expires_at=monotonic() + ttl_seconds
        )
