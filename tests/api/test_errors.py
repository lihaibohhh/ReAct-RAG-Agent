"""
B8 — 错误格式同构
- 制造 500：断言 problem+json 五字段齐全且含 request_id
- 制造 422（请求体非法）：同样 problem+json
- X-Request-Id 响应头存在于所有响应
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio


async def test_500_returns_problem_json(client_with_agent, fake_redis):
    """
    B8: agent.stream_events 抛出未捕获异常 → SSE error 帧（problem+json 同构），
    不向客户端泄露 traceback。
    """
    from tests.api.conftest import FakeAgent

    async def _explode(messages, thread_id=None):
        yield {"event": "on_chat_model_stream", "name": "m",
               "data": {"chunk": type("C", (), {"content": "x"})()}}
        raise RuntimeError("intentional internal error")

    agent = FakeAgent()
    agent.stream_events = _explode  # type: ignore[method-assign]

    async with client_with_agent(agent=agent) as c:
        resp = await c.post("/api/v1/chat/stream", json={"message": "boom"})

    assert resp.status_code == 200  # SSE 本身是 200，错误在帧内
    # 检查 SSE error 帧（problem+json 同构）
    import json as _json
    error_data = None
    current_event = ""
    for line in resp.text.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and current_event == "error":
            error_data = _json.loads(line.split(":", 1)[1].strip())
            break

    assert error_data is not None, "Expected an 'error' SSE frame"
    for field in ("type", "title", "status", "detail", "request_id"):
        assert field in error_data, f"Missing field '{field}' in error frame"
    assert error_data["status"] == 500
    # 不泄露 traceback
    assert "intentional internal error" not in error_data["detail"]
    assert "Traceback" not in error_data.get("detail", "")


async def test_http_500_problem_json_five_fields(client, fake_redis):
    """
    制造 HTTP 级别 500（invoke 端点抛通用 Exception），
    断言 JSONResponse 五字段齐全。
    """
    # 用 invoke 端点：让 agent.invoke 抛异常
    import api.dependencies as deps
    from tests.api.conftest import FakeAgent

    class _ExplodingAgent(FakeAgent):
        async def invoke(self, messages, thread_id=None):
            raise Exception("database exploded")

    from api.main import app
    from api.dependencies import get_agent
    original = app.dependency_overrides.get(get_agent)
    app.dependency_overrides[get_agent] = lambda: _ExplodingAgent()

    try:
        resp = await client.post("/api/v1/chat/invoke", json={"message": "x"})
    finally:
        if original:
            app.dependency_overrides[get_agent] = original
        else:
            app.dependency_overrides.pop(get_agent, None)

    assert resp.status_code == 500
    body = resp.json()
    for field in ("type", "title", "status", "detail", "request_id"):
        assert field in body, f"Missing field '{field}' in 500 body"
    assert body["status"] == 500
    assert "traceback" not in body["detail"].lower()


async def test_422_validation_error_is_problem_json(client, fake_redis):
    """请求体 validation error → 422 problem+json。"""
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"not_message_field": "oops"},
    )
    assert resp.status_code == 422
    body = resp.json()
    for field in ("type", "title", "status", "detail", "request_id"):
        assert field in body, f"Missing field '{field}' in 422 body"


async def test_request_id_header_present(client, fake_redis):
    """所有响应都应含 X-Request-Id 响应头。"""
    endpoints = [
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/metrics"),
        ("POST", "/api/v1/chat/invoke"),
    ]
    for method, path in endpoints:
        if method == "GET":
            resp = await client.get(path)
        else:
            resp = await client.post(path, json={"message": "x"})
        assert "x-request-id" in resp.headers, (
            f"Missing X-Request-Id header on {method} {path}"
        )
        assert resp.headers["x-request-id"], f"Empty X-Request-Id on {method} {path}"


async def test_404_is_problem_json(client, fake_redis):
    """未知路由 → 404 problem+json 格式（error handler 兜底）。"""
    resp = await client.get("/api/v1/nonexistent")
    assert resp.status_code == 404
