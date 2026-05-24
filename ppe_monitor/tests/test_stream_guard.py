"""Tests for websocket stream client gating helpers."""

import asyncio

from app.main import _parse_bool_query
from app.stream_guard import StreamClientGate


def test_parse_bool_query() -> None:
    assert _parse_bool_query("1") is True
    assert _parse_bool_query("true") is True
    assert _parse_bool_query("On") is True
    assert _parse_bool_query("0") is False
    assert _parse_bool_query("false") is False
    assert _parse_bool_query("off") is False
    assert _parse_bool_query(None) is None
    assert _parse_bool_query("maybe") is None


def test_stream_client_gate_limits_and_recovers() -> None:
    async def _scenario() -> None:
        gate = StreamClientGate(max_clients=1)
        assert await gate.try_acquire() is True
        assert await gate.try_acquire() is False
        snap = await gate.snapshot()
        assert snap["active_clients"] == 1
        assert snap["max_clients"] == 1
        await gate.release()
        assert await gate.try_acquire() is True

    asyncio.run(_scenario())


def test_stream_client_gate_release_is_safe_when_idle() -> None:
    async def _scenario() -> None:
        gate = StreamClientGate(max_clients=2)
        await gate.release()
        snap = await gate.snapshot()
        assert snap["active_clients"] == 0
        assert snap["max_clients"] == 2

    asyncio.run(_scenario())
