from __future__ import annotations
import logging
from functools import lru_cache
from react_agent.core.agent import PersistentAgent
from react_agent.memory.context import Context


_logger = logging.getLogger(__name__)

# ── 全局单例 ──────────────────────────────────────────────────────────────────
# 和 Streamlit 侧的 @st.cache_resource 类似，但 FastAPI 用模块级单例。
# 整个进程只初始化一次，避免重复编译 LangGraph + 初始化 checkpointer。
_agent_instance: PersistentAgent | None = None


async def get_agent() -> PersistentAgent:
    """
    FastAPI Depends 注入函数。
    首次调用时初始化 PersistentAgent，之后复用同一实例。

    用法：
        @router.post("/chat/stream")
        async def chat_stream(req: ChatRequest, agent: PersistentAgent = Depends(get_agent)):
            ...
    """
    global _agent_instance
    if _agent_instance is None:
        ctx = Context(checkpoint_backend="sqlite", model="deepseek/deepseek-chat")          # 从环境变量加载配置，与 Streamlit 侧行为一致
        _agent_instance = PersistentAgent(ctx)
        await _agent_instance.initialize()
        _logger.info(f"[API] PersistentAgent 初始化完成，backend={ctx.checkpoint_backend}")
    return _agent_instance


async def startup_init() -> None:
    """
    FastAPI lifespan 启动钩子中调用，提前预热 Agent，
    避免第一个请求触发冷启动（模型加载 ~10s）。
    """
    await get_agent()
    _logger.info("[API] Agent 预热完成，服务就绪")