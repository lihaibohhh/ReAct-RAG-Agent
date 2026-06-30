from __future__ import annotations
import asyncio
import logging
from typing import Any, List, Optional
from react_agent.memory.context import Context
from react_agent.core.graph import compile_graph_with_persistence
from react_agent.core.config import settings

_logger = logging.getLogger(__name__)


def _checkpointer(agent: "PersistentAgent"):
    """返回底层 checkpointer，无则返回 None。"""
    graph = getattr(agent, "_graph", None)
    if graph is None:
        return None
    return getattr(graph, "checkpointer", None)


# ==================== 便捷包装器 ====================
class PersistentAgent:
    """
    持久化 Agent 包装器

    提供更友好的 API，自动处理：
    1. 图编译
    2. Checkpointer 初始化
    3. Thread ID 管理
    4. 历史裁剪（防止 Token 膨胀）
    5. 错误处理
    """

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._graph = None
        self._initialized = False
        # FIX-3: 锁初始值为 None，在 initialize() 内懒创建，确保与当前 event loop 绑定
        self._init_lock = None

    async def initialize(self):
        """异步初始化"""
        # FIX-3: 第一次检查（无锁快路径）
        if self._initialized:
            return

        # FIX-3: 在同一个 loop 内懒创建锁，避免 Streamlit 跨 loop 问题
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        # FIX-3: 加锁后二次检查（double-checked locking），防止并发重复初始化
        async with self._init_lock:
            if self._initialized:
                return

            self._graph = await compile_graph_with_persistence(self.ctx)
            self._initialized = True
            _logger.info(f"[Agent] 初始化完成 (backend: {self.ctx.checkpoint_backend})")


    async def invoke(self, messages: List[Any], thread_id: Optional[str] = None):
        """
        调用 agent

        Args:
            messages: 消息列表（只传当前这一条新消息即可）
            thread_id: 会话 ID（可选，默认使用 Context.default_thread_id）
        """
        if not self._initialized:
            await self.initialize()

        # FIX-1: 未传 thread_id 时打警告并抛出 ValueError，禁止回落到共享 "default" thread
        tid = thread_id or self.ctx.default_thread_id
        if not tid:
            _logger.warning(
                "[Agent] invoke() 调用时未传 thread_id，且 Context.default_thread_id 为 None。"
                "请调用方传入 user:{username} 格式的 thread_id 以保证数据隔离。"
            )
            raise ValueError(
                "thread_id 不能为空。请传入 user:{username} 格式的 thread_id。"
            )

        config = {
            "recursion_limit": self.ctx.recursion_limit,
            "configurable": {"thread_id": tid}
        }

        result = await self._graph.ainvoke(
            {"messages": messages},
            context=self.ctx,
            config=config
        )

        return result

    async def stream_events(self, messages: List[Any], thread_id: Optional[str] = None):
        """
        Token 级流式调用，使用 astream_events(version='v2')。

        Yields 原始 LangGraph 事件字典。调用方（路由层）负责将事件映射为 SSE 帧。
        call_model 节点须声明 config: RunnableConfig 并透传给 model.ainvoke()，
        否则回调链断裂，on_chat_model_stream 事件不会出现。
        """
        if not self._initialized:
            await self.initialize()

        tid = thread_id or self.ctx.default_thread_id
        if not tid:
            _logger.warning(
                "[Agent] stream_events() 调用时未传 thread_id，"
                "请传入 user:{username} 格式的 thread_id。"
            )
            raise ValueError(
                "thread_id 不能为空。请传入 user:{username} 格式的 thread_id。"
            )

        config = {
            "recursion_limit": self.ctx.recursion_limit,
            "configurable": {"thread_id": tid},
        }

        async for event in self._graph.astream_events(
            {"messages": messages},
            context=self.ctx,
            config=config,
            version="v2",
        ):
            yield event

    async def get_history(self, thread_id: str) -> Optional[List[Any]]:
        """
        读取指定 thread 最新 checkpoint 的消息列表。

        Returns:
            None  — thread 不存在（含 checkpointer 未初始化）
            []    — thread 存在但尚无消息
            [...]  — 消息列表（LangChain Message 对象）
        """
        if not self._initialized:
            await self.initialize()

        cp = _checkpointer(self)
        if cp is None:
            return None

        config: dict = {"configurable": {"thread_id": thread_id}}
        try:
            tup = await cp.aget_tuple(config)
        except Exception as e:
            _logger.error("get_history(%s) error: %s", thread_id, e)
            return None

        if tup is None:
            return None

        channel_values = tup.checkpoint.get("channel_values", {})
        return list(channel_values.get("messages", []))

    async def delete_thread(self, thread_id: str) -> Optional[bool]:
        """
        删除指定 thread 的全部 checkpoint 数据。

        Returns:
            None  — thread 不存在
            True  — 删除成功
            False — 后端不支持删除（调用方应返回 501）
        """
        if not self._initialized:
            await self.initialize()

        cp = _checkpointer(self)
        if cp is None:
            return None

        # 先确认 thread 存在
        config: dict = {"configurable": {"thread_id": thread_id}}
        try:
            tup = await cp.aget_tuple(config)
        except Exception:
            tup = None
        if tup is None:
            return None

        # SQLite 后端：直接删除底层三张表中的相关行
        if hasattr(cp, "conn") and cp.conn is not None:
            try:
                conn = cp.conn
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)  # noqa: S608
                    )
                await conn.commit()
                return True
            except Exception as e:
                _logger.error("delete_thread SQLite(%s) error: %s", thread_id, e)
                return False

        # MemorySaver 后端：直接清理内存存储
        try:
            from langgraph.checkpoint.memory import MemorySaver
            if isinstance(cp, MemorySaver) and hasattr(cp, "storage"):
                to_del = [k for k in list(cp.storage.keys()) if k[0] == thread_id]
                for k in to_del:
                    del cp.storage[k]
                return True
        except (ImportError, Exception):
            pass

        # 其余后端（Postgres、Redis）暂不支持删除
        _logger.warning("delete_thread: backend %s does not support deletion", type(cp).__name__)
        return False

    async def stream(self, messages: List[Any], thread_id: Optional[str] = None):
        """流式调用（支持实时输出）"""
        if not self._initialized:
            await self.initialize()

        # FIX-1: 未传 thread_id 时打警告并抛出 ValueError，禁止回落到共享 "default" thread
        tid = thread_id or self.ctx.default_thread_id
        if not tid:
            _logger.warning(
                "[Agent] stream() 调用时未传 thread_id，且 Context.default_thread_id 为 None。"
                "请调用方传入 user:{username} 格式的 thread_id 以保证数据隔离。"
            )
            raise ValueError(
                "thread_id 不能为空。请传入 user:{username} 格式的 thread_id。"
            )

        config = {
            "recursion_limit": self.ctx.recursion_limit,
            "configurable": {"thread_id": tid}
        }

        async for chunk in self._graph.astream(
                {"messages": messages},
                context=self.ctx,
                config=config
        ):
            yield chunk