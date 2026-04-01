from __future__ import annotations
import asyncio
import logging
from typing import Any, List, Optional
from react_agent.memory.context import Context
from react_agent.core.graph import compile_graph_with_persistence
from react_agent.core.config import settings

_logger = logging.getLogger(__name__)


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
            print(f"[Agent] 初始化完成 (backend: {self.ctx.checkpoint_backend})")


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