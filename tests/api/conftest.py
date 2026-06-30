"""
tests/api/conftest.py — 共享 fixtures。

设计原则：
- 零真实 LLM / Redis 调用：假 agent + fakeredis 彻底替换真实依赖
- client fixture 走 asgi-lifespan + httpx.AsyncClient，lifespan 预热路径也被覆盖
- 假 agent 的 stream_events 接受 "script" 参数，可按测试场景定制事件序列
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Optional
from unittest.mock import patch

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager


# ── Mock LangChain 类型（仅需 _map_event 访问的属性）────────────────────────────

@dataclass
class _MockChunk:
    content: str


@dataclass
class _MockAIMessage:
    """
    最小化 AIMessage 替身。
    _extract_deepseek_v4_usage 两条路径：
      - response_metadata.token_usage（非流式）→ 我们用这条，更简单
      - usage_metadata（流式）
    """
    content: str = "mock reply"
    response_metadata: dict = field(default_factory=lambda: {
        "model_name": "deepseek-v4-flash",
        "token_usage": {
            "prompt_cache_hit_tokens": 100,
            "prompt_cache_miss_tokens": 200,
            "completion_tokens": 50,
        },
    })
    usage_metadata: dict = field(default_factory=dict)


# ── 假 agent ──────────────────────────────────────────────────────────────────

# 默认事件脚本：2 个 token + 1 个 tool_call/tool_end + usage + end
DEFAULT_SCRIPT: List[dict] = [
    {"event": "on_chat_model_stream", "name": "model",
     "data": {"chunk": _MockChunk("Hello")}},
    {"event": "on_chat_model_stream", "name": "model",
     "data": {"chunk": _MockChunk(" world")}},
    {"event": "on_tool_start", "name": "sql_tool",
     "data": {"input": {"query": "SELECT 1"}}},
    {"event": "on_tool_end", "name": "sql_tool",
     "data": {"output": '{"ok": true, "rows": []}'}},
    {"event": "on_chat_model_end", "name": "model",
     "data": {"output": _MockAIMessage()}},
]


class FakeAgent:
    """
    PersistentAgent 替身。stream_events 按 script 吐事件。
    script=None → 用 DEFAULT_SCRIPT。
    cancel_after_n_events=N → 第 N 个事件后 raise CancelledError（模拟断连）。
    history_map={thread_id: [msg, ...]} → get_history 按此返回，None 键不存在。
    """

    _initialized: bool = True

    class _FakeCtx:
        model: str = "deepseek/deepseek-v4-flash"
        deepseek_v4_price: dict = field(default_factory=lambda: {
            "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
        })
        checkpoint_backend: str = "sqlite"

    def __init__(
        self,
        script: Optional[List[dict]] = None,
        cancel_after_n: int = -1,
        history_map: Optional[dict] = None,
    ):
        self.ctx = self._FakeCtx()
        self.ctx.deepseek_v4_price = {
            "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28}
        }
        self._script = script if script is not None else DEFAULT_SCRIPT
        self._cancel_after_n = cancel_after_n
        # 可变副本：delete_thread 会修改它
        self._history_map: dict = dict(history_map) if history_map else {}

    async def stream_events(
        self, messages: List[Any], thread_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        for i, ev in enumerate(self._script):
            if self._cancel_after_n >= 0 and i >= self._cancel_after_n:
                raise asyncio.CancelledError("fake disconnect")
            yield ev
            await asyncio.sleep(0)  # 让出事件循环

    async def invoke(
        self, messages: List[Any], thread_id: Optional[str] = None
    ) -> dict:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="fake answer")]}

    async def get_history(self, thread_id: str) -> Optional[List[Any]]:
        """None = thread 不存在，[] / [...] = thread 已有的消息列表。"""
        if thread_id in self._history_map:
            return list(self._history_map[thread_id])
        return None

    async def delete_thread(self, thread_id: str) -> Optional[bool]:
        """None = thread 不存在；True = 删除成功。"""
        if thread_id not in self._history_map:
            return None
        del self._history_map[thread_id]
        return True


# ── fakeredis fixture ────────────────────────────────────────────────────────

@pytest.fixture
def fake_redis():
    """返回 fakeredis 异步客户端，并 patch get_async_redis 使限流/预算走此实例。"""
    server = fakeredis.FakeServer()
    redis_client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)

    with patch("react_agent.utils.redis_client.get_async_redis", return_value=redis_client):
        yield redis_client


@pytest.fixture
def broken_redis():
    """每次调用 get_async_redis 都抛 ConnectionError，模拟 Redis 宕机。"""
    import redis.asyncio as aioredis

    async def _raise(*a, **kw):
        raise aioredis.ConnectionError("fake redis down")

    class _BrokenRedis:
        async def incr(self, *a, **kw): raise aioredis.ConnectionError("down")
        async def expire(self, *a, **kw): raise aioredis.ConnectionError("down")
        async def get(self, *a, **kw): raise aioredis.ConnectionError("down")
        async def incrby(self, *a, **kw): raise aioredis.ConnectionError("down")

    with patch("react_agent.utils.redis_client.get_async_redis", return_value=_BrokenRedis()):
        yield


# ── HTTP test client fixture ─────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(fake_redis):
    """
    完整 ASGI 测试客户端：
    - asgi-lifespan 确保 lifespan 预热路径被执行
    - 假 agent 通过 _agent_instance patch 阻止真实初始化
    - fakeredis 已由 fake_redis fixture 替换
    """
    from api.main import app
    from api.dependencies import get_agent
    import api.dependencies as deps

    agent = FakeAgent()

    # 1) 预置 _agent_instance，使 startup_init → get_agent() 不启动真实 agent
    deps._agent_instance = agent

    # 2) FastAPI dependency override，路由 handler 拿到的也是 fake agent
    app.dependency_overrides[get_agent] = lambda: agent

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


@pytest_asyncio.fixture
async def client_with_agent(fake_redis):
    """
    带自定义 agent 的 client 工厂，返回 (make_client 协程)。
    用法：async with client_with_agent(agent=FakeAgent(...)) as c: ...
    """
    from contextlib import asynccontextmanager
    from api.main import app
    from api.dependencies import get_agent
    import api.dependencies as deps

    @asynccontextmanager
    async def _make(agent: FakeAgent):
        deps._agent_instance = agent
        app.dependency_overrides[get_agent] = lambda: agent
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

    yield _make


# ── pytest-asyncio 模式配置 ───────────────────────────────────────────────────

# 兼容 pytest-asyncio 0.23.x asyncio_mode="auto"，在 pytest.ini 里设置
