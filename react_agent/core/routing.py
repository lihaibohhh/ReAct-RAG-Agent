from __future__ import annotations
import logging
from typing import Literal

from langchain_core.messages import AIMessage
from react_agent.core.state import State

logger = logging.getLogger(__name__)


def route_model_output(state: State) -> Literal["__end__", "tools", "call_model"]:
    """根据最后一条 AIMessage 是否包含 tool_calls 决定下一步"""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        # FIX-⑦: 非 AIMessage 时不崩溃，回退到 call_model 重新生成，并记录日志便于排查
        logger.warning(
            "[route_model_output] 最后一条消息不是 AIMessage，实际类型: %s，回退到 call_model",
            type(last_message).__name__,
        )
        return "call_model"
    return "__end__" if not last_message.tool_calls else "tools"