"""Unit tests for verifier cache TTL behavior."""

from time import sleep

from app.cache import VerifierCache
from app.schemas import VerifierResult, VerifierVerdict


def test_cache_hit_before_expiry() -> None:
    cache = VerifierCache()
    result = VerifierResult(verdict=VerifierVerdict.COMPLIANT, score=0.9)
    cache.put(person_id=1, item="helmet", result=result, ttl_seconds=0.5)
    cached = cache.get(person_id=1, item="helmet")
    assert cached is not None
    assert cached.verdict == VerifierVerdict.COMPLIANT


def test_cache_expires_after_ttl() -> None:
    cache = VerifierCache()
    result = VerifierResult(verdict=VerifierVerdict.VIOLATION, score=0.2)
    cache.put(person_id=7, item="boots", result=result, ttl_seconds=0.05)
    sleep(0.08)
    assert cache.get(person_id=7, item="boots") is None
