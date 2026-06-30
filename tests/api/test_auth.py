"""
B4 — 鉴权测试
- 无 key → 401 problem+json
- 错 key → 401 problem+json，含 request_id
- 对 key → 200
- /health、/api/v1/health、/api/v1/metrics 无 key 可访问
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_no_key_returns_401(client, monkeypatch):
    """无 API Key 时返回 401 problem+json。"""
    monkeypatch.setenv("API_API_KEY", "real-secret-key")

    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "real-secret-key"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.post("/api/v1/chat/stream", json={"message": "hi"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["status"] == 401
    assert body["title"] == "Unauthorized"
    assert "request_id" in body
    assert resp.headers.get("content-type", "").startswith("application/problem+json")


async def test_wrong_key_returns_401(client, monkeypatch):
    """错误 API Key → 401，body 含 request_id。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "correct-key"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "hi"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["status"] == 401
    assert "request_id" in body and body["request_id"]


async def test_correct_key_passes(client, monkeypatch):
    """正确 API Key → 200 流式响应。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "good-key"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "hi"},
        headers={"X-API-Key": "good-key"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


async def test_bearer_token_accepted(client, monkeypatch):
    """Authorization: Bearer 方式也能通过鉴权。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "bearer-key"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "hi"},
        headers={"Authorization": "Bearer bearer-key"},
    )
    assert resp.status_code == 200


async def test_health_no_auth_required(client, monkeypatch):
    """/api/v1/health 无 key 可访问（豁免端点）。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "secret"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200


async def test_health_legacy_no_auth_required(client, monkeypatch):
    """/health（旧版）无 key 可访问。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "secret"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_metrics_no_auth_required(client, monkeypatch):
    """/api/v1/metrics 无 key 可访问（Prometheus scraper 豁免）。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "secret"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


async def test_401_body_is_problem_json(client, monkeypatch):
    """401 body 含 type/title/status/detail/request_id 五字段。"""
    import api.settings
    class _AuthSettings(api.settings.APISettings):
        api_key: str = "secret"
    monkeypatch.setattr(api.settings, "APISettings", _AuthSettings)

    resp = await client.post("/api/v1/chat/stream", json={"message": "x"})
    body = resp.json()
    for field in ("type", "title", "status", "detail", "request_id"):
        assert field in body, f"Missing field: {field}"
