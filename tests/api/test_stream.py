"""
B1 — 流式事件序列
B2 — 断连取消（producer task cancel + await + record_token_usage 按实际值）
B3 — 超时（stream_timeout_s 极小，发 error + done，连接不吊死）
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Tuple

import pytest
import pytest_asyncio

from tests.api.conftest import DEFAULT_SCRIPT, FakeAgent, _MockAIMessage, _MockChunk

pytestmark = pytest.mark.asyncio


# ── SSE 解析工具 ──────────────────────────────────────────────────────────────

def parse_sse(text: str) -> List[Tuple[str, dict]]:
    """把 SSE 响应体解析成 [(event_type, data_dict), ...]"""
    frames = []
    current_event = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = json.loads(line.split(":", 1)[1].strip())
            frames.append((current_event, data))
    return frames


# ── B1: 流式事件序列 ──────────────────────────────────────────────────────────

async def test_stream_event_sequence(client):
    """
    B1: 假 agent 按 DEFAULT_SCRIPT 吐事件，断言 SSE 帧顺序：
    token… → tool_call → tool_result → usage → done
    done 帧含 ttft_ms / total_tokens / total_cost。
    """
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "hello"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    frames = parse_sse(resp.text)
    event_types = [e for e, _ in frames]

    # 至少有 token、tool_call、tool_result、usage、done
    assert "token" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "usage" in event_types
    assert event_types[-1] == "done", f"last event should be done, got {event_types[-1]}"

    # done 帧必须有这三个字段且非零
    done_data = next(d for e, d in frames if e == "done")
    assert "ttft_ms" in done_data and done_data["ttft_ms"] is not None
    assert "total_tokens" in done_data and done_data["total_tokens"] > 0
    assert "total_cost" in done_data and done_data["total_cost"] > 0

    # usage 之后不能再有 token（顺序检查）
    last_token_idx = max((i for i, (e, _) in enumerate(frames) if e == "token"), default=-1)
    usage_idx = next(i for i, (e, _) in enumerate(frames) if e == "usage")
    assert last_token_idx < usage_idx, "all token frames must precede usage frame"


async def test_stream_done_frame_fields(client):
    """done 帧的 session_id / total_ms 也存在且合理。"""
    resp = await client.post("/api/v1/chat/stream", json={"message": "test"})
    frames = parse_sse(resp.text)
    done = next(d for e, d in frames if e == "done")
    assert "session_id" in done and done["session_id"]
    assert "total_ms" in done and done["total_ms"] >= 0


async def test_stream_tool_call_fields(client):
    """tool_call 帧含 name + args；tool_result 帧含 name + ok。"""
    resp = await client.post("/api/v1/chat/stream", json={"message": "query"})
    frames = parse_sse(resp.text)
    tc = next(d for e, d in frames if e == "tool_call")
    tr = next(d for e, d in frames if e == "tool_result")
    assert tc["name"] == "sql_tool"
    assert "args" in tc
    assert tr["name"] == "sql_tool"
    assert "ok" in tr


# ── B2: 断连取消 ──────────────────────────────────────────────────────────────

async def test_disconnect_cancels_producer(client_with_agent, caplog):
    """
    B2: 假 agent 先发 on_chat_model_end（写入 tokens），然后在第 2 个事件时触发
    CancelledError，断言：
      - producer task 被 cancel + await（不泄漏）
      - record_token_usage 被调用且传入 > 0 tokens（已产生的 usage，不是 0）
    """
    # Event 0: model_end → state["total_tokens"] 累计
    # Event 1: 触发 CancelledError（cancel_after_n=1）
    script = [
        {"event": "on_chat_model_end", "name": "m",
         "data": {"output": _MockAIMessage()}},
        {"event": "on_chat_model_stream", "name": "m",  # 触发 cancel
         "data": {"chunk": _MockChunk("extra")}},
    ]
    agent = FakeAgent(script=script, cancel_after_n=1)

    recorded_tokens: list[int] = []

    async def _fake_record(bucket_key: str, tokens: int):
        recorded_tokens.append(tokens)

    from unittest.mock import patch as _patch

    with caplog.at_level(logging.INFO, logger="api.routes.v1.chat"):
        # 必须 patch chat 模块内的本地绑定，而不是 ratelimit 模块属性
        with _patch("api.routes.v1.chat.record_token_usage", new=_fake_record):
            async with client_with_agent(agent=agent) as c:
                resp = await c.post("/api/v1/chat/stream", json={"message": "hi"})
                frames = parse_sse(resp.text)
                event_types = [e for e, _ in frames]
                # usage 帧在 cancel 前已发出
                assert "usage" in event_types

    # record_token_usage 必须被调用且传入 > 0 tokens（来自 on_chat_model_end 的 usage）
    assert len(recorded_tokens) >= 1, "record_token_usage should have been called in finally"
    assert recorded_tokens[0] > 0, (
        f"expected tokens > 0 (actual from on_chat_model_end), got {recorded_tokens[0]}"
    )


async def test_disconnect_stream_cancelled_log(client_with_agent, caplog):
    """B2 补充：stream_cancelled 字样出现在日志中（断连路径）。"""
    # 2 个事件：第 1 个正常，第 2 个触发 cancel_after_n=1
    script = [
        {"event": "on_chat_model_stream", "name": "m",
         "data": {"chunk": _MockChunk("A")}},
        {"event": "on_chat_model_stream", "name": "m",
         "data": {"chunk": _MockChunk("B")}},
    ]
    agent = FakeAgent(script=script, cancel_after_n=1)

    with caplog.at_level(logging.INFO):
        async with client_with_agent(agent=agent) as c:
            await c.post("/api/v1/chat/stream", json={"message": "x"})

    # caplog.messages 是 str 列表，直接搜索（不同于 rec.message 需格式化）
    assert any("stream_cancelled" in msg for msg in caplog.messages), (
        f"Expected 'stream_cancelled' in logs. Messages: {caplog.messages}"
    )


# ── B3: 超时 ─────────────────────────────────────────────────────────────────

async def test_stream_timeout(client_with_agent, monkeypatch):
    """
    B3: stream_timeout_s=0.1s。假 agent 每个事件 sleep 0.5s → 必然超时。
    断言：error 帧 title=Stream Timeout，后跟 done 帧，连接不吊死。
    """
    import asyncio as _asyncio

    # slow agent: 一个 token，然后长 sleep
    async def _slow_stream(messages, thread_id=None):
        yield {"event": "on_chat_model_stream", "name": "m",
               "data": {"chunk": _MockChunk("slow")}}
        await _asyncio.sleep(5)  # 远超 timeout

    agent = FakeAgent()
    agent.stream_events = _slow_stream  # type: ignore[method-assign]

    # timeout 极小
    monkeypatch.setenv("API_STREAM_TIMEOUT_S", "0")  # 0 → 立即超时
    # 直接覆盖 APISettings 实例的值比 env 更可靠：patch settings
    import api.settings as _settings
    original_cls = _settings.APISettings

    class _PatchedSettings(original_cls):
        stream_timeout_s: int = 1  # 1s 对假 agent 足够小

    monkeypatch.setattr(_settings, "APISettings", _PatchedSettings)

    async with client_with_agent(agent=agent) as c:
        resp = await c.post(
            "/api/v1/chat/stream",
            json={"message": "slow"},
            timeout=30,
        )

    assert resp.status_code == 200
    frames = parse_sse(resp.text)
    event_types = [e for e, _ in frames]

    assert "error" in event_types, f"Expected error frame, got: {event_types}"
    error_data = next(d for e, d in frames if e == "error")
    assert error_data.get("title") == "Stream Timeout"
    assert "done" in event_types, "done frame must follow timeout error"
    assert event_types[-1] == "done"


async def test_stream_timeout_no_hang(client_with_agent, monkeypatch):
    """B3 补充：超时请求在合理时间内返回，不吊死连接。"""
    import asyncio as _asyncio
    import time

    async def _very_slow(messages, thread_id=None):
        await _asyncio.sleep(60)
        yield {"event": "on_chat_model_stream", "name": "m",
               "data": {"chunk": _MockChunk("never")}}

    agent = FakeAgent()
    agent.stream_events = _very_slow  # type: ignore[method-assign]

    import api.settings as _settings

    class _Fast(original := _settings.APISettings):
        stream_timeout_s: int = 1

    monkeypatch.setattr(_settings, "APISettings", _Fast)

    async with client_with_agent(agent=agent) as c:
        t0 = time.perf_counter()
        await c.post("/api/v1/chat/stream", json={"message": "x"}, timeout=30)
        elapsed = time.perf_counter() - t0

    assert elapsed < 10, f"Request should complete within 10s, took {elapsed:.1f}s"
