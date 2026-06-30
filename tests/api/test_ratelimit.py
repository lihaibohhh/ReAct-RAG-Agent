"""
B5 — 限流/预算
  - 打爆 rpm → 429 + Retry-After
  - 刷爆日预算 → 429
  - 匿名（空 key / 免鉴权模式）走 client:IP bucket 同样被限流 ← 安全漏洞修复用例
B6 — fail-open
  - fakeredis 抛 ConnectionError → 放行（fail-open）+ warning 日志，不 fail-closed
"""
from __future__ import annotations

import logging
import pytest

pytestmark = pytest.mark.asyncio


# ── 辅助：无鉴权设置 ──────────────────────────────────────────────────────────

def _no_auth_settings(monkeypatch):
    """免鉴权模式：api_key=None，rate_limit_rpm=3。"""
    import api.settings as _s
    class _Settings(_s.APISettings):
        api_key: None = None
        rate_limit_rpm: int = 3
        daily_token_budget: int = 1_000_000
    monkeypatch.setattr(_s, "APISettings", _Settings)


def _with_rpm(monkeypatch, rpm: int = 3):
    import api.settings as _s
    class _Settings(_s.APISettings):
        api_key: None = None
        rate_limit_rpm: int = rpm
        daily_token_budget: int = 1_000_000
    monkeypatch.setattr(_s, "APISettings", _Settings)


# ── B5a: RPM 超限 ─────────────────────────────────────────────────────────────

async def test_rpm_exceeded_returns_429(client, monkeypatch, fake_redis):
    """rpm=3，第 4 次请求 → 429 + Retry-After 响应头。"""
    _with_rpm(monkeypatch, rpm=3)

    for i in range(3):
        r = await client.post("/api/v1/chat/stream", json={"message": "x"})
        assert r.status_code == 200, f"Request {i+1} should succeed"

    r4 = await client.post("/api/v1/chat/stream", json={"message": "x"})
    assert r4.status_code == 429
    body = r4.json()
    assert body["status"] == 429
    assert body["title"] == "Too Many Requests"
    assert "Retry-After" in r4.headers


async def test_rpm_retry_after_header(client, monkeypatch, fake_redis):
    """Retry-After 值是正整数（秒数）。"""
    _with_rpm(monkeypatch, rpm=1)

    await client.post("/api/v1/chat/stream", json={"message": "x"})
    r = await client.post("/api/v1/chat/stream", json={"message": "x"})
    assert r.status_code == 429
    retry_after = int(r.headers["Retry-After"])
    assert 1 <= retry_after <= 61


# ── B5b: 日预算超限 ───────────────────────────────────────────────────────────

async def test_daily_budget_exceeded(client, monkeypatch, fake_redis):
    """日预算 budget=1 token，第 2 次请求（记账后）被拒。"""
    import api.settings as _s
    class _Settings(_s.APISettings):
        api_key: None = None
        rate_limit_rpm: int = 1000
        daily_token_budget: int = 1  # 极小预算

    monkeypatch.setattr(_s, "APISettings", _Settings)

    # 第 1 次成功，触发 record_token_usage 写入 > 1 tokens
    r1 = await client.post("/api/v1/chat/stream", json={"message": "x"})
    assert r1.status_code == 200

    # 第 2 次被拒（事后扣费，已超预算）
    r2 = await client.post("/api/v1/chat/stream", json={"message": "x"})
    assert r2.status_code == 429
    body = r2.json()
    assert body["title"] == "Daily Budget Exceeded"


# ── B5c: 匿名 IP bucket 限流（安全漏洞修复专项用例）────────────────────────────

async def test_anonymous_ip_bucket_rate_limited(client, monkeypatch, fake_redis):
    """
    免鉴权模式（api_key=None）下，匿名请求走 client:{IP} bucket，
    同样受 rpm 限制。这是 Phase 2 安全漏洞修复的核心验证：
    空 key 不能绕过限流/预算，否则匿名流量无管控。
    """
    _with_rpm(monkeypatch, rpm=2)

    headers = {"X-Forwarded-For": "192.168.1.100"}

    r1 = await client.post("/api/v1/chat/stream", json={"message": "a"}, headers=headers)
    r2 = await client.post("/api/v1/chat/stream", json={"message": "b"}, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200

    # 第 3 次 → 同一 IP bucket 超限
    r3 = await client.post("/api/v1/chat/stream", json={"message": "c"}, headers=headers)
    assert r3.status_code == 429, (
        "Anonymous requests via client:IP bucket must be rate-limited. "
        "This was a security vulnerability if it returns 200."
    )


async def test_different_ips_have_separate_buckets(client, monkeypatch, fake_redis):
    """不同 IP 走不同 bucket，互不影响。"""
    _with_rpm(monkeypatch, rpm=1)

    r1 = await client.post(
        "/api/v1/chat/stream", json={"message": "x"},
        headers={"X-Forwarded-For": "10.0.0.1"}
    )
    # 不同 IP 的第 1 次请求应放行
    r2 = await client.post(
        "/api/v1/chat/stream", json={"message": "x"},
        headers={"X-Forwarded-For": "10.0.0.2"}
    )
    assert r1.status_code == 200
    assert r2.status_code == 200


# ── B6: fail-open — Redis 宕机时放行 ────────────────────────────────────────

async def test_redis_down_fail_open(client_with_agent, broken_redis, monkeypatch, caplog):
    """
    B6: fakeredis 抛 ConnectionError（模拟 Redis 宕），
    断言限流放行（fail-open）且 warning 日志出现，请求成功返回 200。
    """
    from tests.api.conftest import FakeAgent
    _no_auth_settings(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="api.ratelimit"):
        async with client_with_agent(agent=FakeAgent()) as c:
            r = await c.post("/api/v1/chat/stream", json={"message": "hi"})

    assert r.status_code == 200, (
        f"Redis down should fail-open (200), got {r.status_code}"
    )
    assert any(
        "redis_unavailable" in msg or "fail_open" in msg
        for msg in caplog.messages
    ), "Expected warning log about Redis unavailability"


async def test_redis_down_budget_fail_open(client_with_agent, broken_redis, monkeypatch):
    """B6 补充：预算检查也是 fail-open。"""
    from tests.api.conftest import FakeAgent
    _no_auth_settings(monkeypatch)

    async with client_with_agent(agent=FakeAgent()) as c:
        r = await c.post("/api/v1/chat/stream", json={"message": "hi"})

    assert r.status_code == 200
