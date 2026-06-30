"""
test_sessions.py — Phase 3 会话 CRUD 测试

覆盖：
  S1  GET history — 200 + 正确分页字段
  S2  GET history — 404 when session not found
  S3  DELETE — 204 成功删除
  S4  DELETE — 删后再查返回 404
  S5  GET list — 固定 501
  S6  IDOR 安全守门：key B 用 key A 的 session_id 打 404
  S7  分页边界：page / page_size 参数正确裁剪消息
  S8  DELETE not found — 404
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from tests.api.conftest import FakeAgent

pytestmark = pytest.mark.asyncio

# ── 共享 session / history 数据 ──────────────────────────────────────────────

SESSION_ID = "test-session-abc"
BUCKET_KEY_A = "key-owner-A"
BUCKET_KEY_B = "key-outsider-B"

# thread_id 格式与 sessions.py / chat.py 保持一致
THREAD_ID_A = f"user:{BUCKET_KEY_A}:{SESSION_ID}"

FAKE_MESSAGES: List[dict] = [
    {"type": "human", "content": "你好"},
    {"type": "ai", "content": "你好！有什么可以帮你的？"},
    {"type": "human", "content": "帮我查一下宁德时代的收入"},
    {"type": "ai", "content": "宁德时代2024年营收约3864亿元。"},
]


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def session_client(fake_redis):
    """
    带预置 session 历史的测试客户端（bucket_key = BUCKET_KEY_A）。
    monkeypatch require_api_key 使其免验证并直接返回 BUCKET_KEY_A。
    """
    from api.main import app
    from api.dependencies import get_agent
    import api.dependencies as deps
    import api.security as security

    agent = FakeAgent(history_map={THREAD_ID_A: FAKE_MESSAGES})
    deps._agent_instance = agent
    app.dependency_overrides[get_agent] = lambda: agent
    # 免鉴权模式：require_api_key 直接返回 BUCKET_KEY_A，使 bucket_key 可预测
    app.dependency_overrides[security.require_api_key] = lambda: BUCKET_KEY_A

    try:
        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                yield c
    finally:
        app.dependency_overrides.clear()
        deps._agent_instance = None


@asynccontextmanager
async def _make_client(fake_redis, agent: FakeAgent, bucket_key: str):
    """通用工厂：创建指定 bucket_key 的客户端。"""
    from api.main import app
    from api.dependencies import get_agent
    import api.dependencies as deps
    import api.security as security

    deps._agent_instance = agent
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[security.require_api_key] = lambda: bucket_key

    try:
        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                yield c
    finally:
        app.dependency_overrides.clear()
        deps._agent_instance = None


# ── S1: GET history 200 ───────────────────────────────────────────────────────

async def test_get_history_200(session_client):
    """S1: 已有会话返回 200 + 正确分页字段。"""
    resp = await session_client.get(f"/api/v1/sessions/{SESSION_ID}/history")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == SESSION_ID
    assert isinstance(body["messages"], list)
    assert body["total"] == len(FAKE_MESSAGES)
    assert body["page"] == 1
    assert "page_size" in body
    assert "has_more" in body
    # 默认 page_size=20，消息共 4 条 → has_more=False
    assert body["has_more"] is False
    # 每条消息含 type + content
    for msg in body["messages"]:
        assert "type" in msg
        assert "content" in msg


# ── S2: GET history 404 ───────────────────────────────────────────────────────

async def test_get_history_404(session_client):
    """S2: 不存在的 session_id → 404 problem+json。"""
    resp = await session_client.get("/api/v1/sessions/no-such-session/history")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert body["title"] == "Not Found"


# ── S3: DELETE 204 ────────────────────────────────────────────────────────────

async def test_delete_session_204(session_client):
    """S3: 删除存在的会话返回 204，无响应体。"""
    resp = await session_client.delete(f"/api/v1/sessions/{SESSION_ID}")
    assert resp.status_code == 204
    assert resp.content == b""


# ── S4: 删后再查 404 ──────────────────────────────────────────────────────────

async def test_delete_then_get_404(session_client):
    """S4: DELETE 后再 GET → 404（checkpointer 已清除）。"""
    del_resp = await session_client.delete(f"/api/v1/sessions/{SESSION_ID}")
    assert del_resp.status_code == 204

    get_resp = await session_client.get(f"/api/v1/sessions/{SESSION_ID}/history")
    assert get_resp.status_code == 404


# ── S5: GET list 501 ─────────────────────────────────────────────────────────

async def test_list_sessions_501(session_client):
    """S5: 列举会话固定返回 501 Not Implemented。"""
    resp = await session_client.get("/api/v1/sessions")
    assert resp.status_code == 501
    body = resp.json()
    assert body["status"] == 501
    assert body["title"] == "Not Implemented"


# ── S6: IDOR 安全守门 ─────────────────────────────────────────────────────────

async def test_idor_key_b_cannot_access_key_a_session(fake_redis):
    """
    S6: key B 用 key A 创建的 session_id 访问 → 404（不泄露存在性）。

    关键：
    - agent 的 history_map 只在 key A 的 thread_id 下有数据
    - key B 构造的 thread_id 不同 → get_history 返回 None → 404
    """
    agent = FakeAgent(history_map={THREAD_ID_A: FAKE_MESSAGES})

    async with _make_client(fake_redis, agent, BUCKET_KEY_B) as client_b:
        # Key B 尝试用 Key A 的 session_id 读取历史
        resp = await client_b.get(f"/api/v1/sessions/{SESSION_ID}/history")
        assert resp.status_code == 404, (
            f"IDOR 防护失效：key B 不应能读取 key A 的会话，得到 {resp.status_code}\n{resp.text}"
        )
        body = resp.json()
        assert body["status"] == 404
        # 确保不是 403（403 泄露了"会话存在但无权限"的信息）
        assert body["title"] == "Not Found"


async def test_idor_key_b_cannot_delete_key_a_session(fake_redis):
    """S6b: key B 尝试 DELETE key A 的 session → 404。"""
    agent = FakeAgent(history_map={THREAD_ID_A: FAKE_MESSAGES})

    async with _make_client(fake_redis, agent, BUCKET_KEY_B) as client_b:
        resp = await client_b.delete(f"/api/v1/sessions/{SESSION_ID}")
        assert resp.status_code == 404


# ── S7: 分页边界 ──────────────────────────────────────────────────────────────

async def test_pagination_page_size(session_client):
    """S7: page_size=2 时返回前 2 条，has_more=True。"""
    resp = await session_client.get(
        f"/api/v1/sessions/{SESSION_ID}/history",
        params={"page": 1, "page_size": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["messages"]) == 2
    assert body["total"] == len(FAKE_MESSAGES)
    assert body["has_more"] is True
    assert body["page"] == 1
    assert body["page_size"] == 2


async def test_pagination_last_page(session_client):
    """S7b: 最后一页 has_more=False，消息数量正确。"""
    resp = await session_client.get(
        f"/api/v1/sessions/{SESSION_ID}/history",
        params={"page": 2, "page_size": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["messages"]) == 2
    assert body["has_more"] is False


async def test_pagination_beyond_end(session_client):
    """S7c: 超出范围的页码返回空列表，has_more=False，status=200。"""
    resp = await session_client.get(
        f"/api/v1/sessions/{SESSION_ID}/history",
        params={"page": 99, "page_size": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages"] == []
    assert body["has_more"] is False
    assert body["total"] == len(FAKE_MESSAGES)


# ── S8: DELETE 不存在的 session ──────────────────────────────────────────────

async def test_delete_nonexistent_session_404(session_client):
    """S8: 删除不存在的 session → 404（防止用 DELETE 探测存在性）。"""
    resp = await session_client.delete("/api/v1/sessions/ghost-session")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
