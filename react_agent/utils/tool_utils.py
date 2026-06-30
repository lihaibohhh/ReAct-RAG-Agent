from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, AIMessage
from react_agent.memory.context import Context
from react_agent.tools import TOOLS

try:
    from langchain_openai.chat_models.base import _convert_message_to_dict
except Exception:
    _convert_message_to_dict = None


def _ai_tool_call_ids(msg) -> List[str]:
    """返回一条消息中 DeepSeek 实际会看到的所有 tool_call id —— 全项目统一检测口径。

        设计要点：
          1) 先读 .tool_calls 与 .invalid_tool_calls 两个属性（命中即返回，
             避免对绝大多数“无工具调用”的普通消息做无谓的 dict 转换）。
             .invalid_tool_calls 覆盖了“JSON 参数解析失败”这一最高频的悬空来源。
          2) 两个属性都为空时，再用官方转换函数兜底，捕捉藏在流式 chunk /
             additional_kwargs 里、属性读不到的隐藏调用。

        返回：
          - 非 AIMessage           -> []
          - 无任何待应答 tool_call  -> []
          - 否则                   -> 去重且保序的 id 列表
        """
    if not isinstance(msg, AIMessage):
        return []

    ids: List[str] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        _id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        if _id and _id not in ids:
            ids.append(_id)
    for itc in (getattr(msg, "invalid_tool_calls", None) or []):
        _id = itc.get("id") if isinstance(itc, dict) else getattr(itc, "id", None)
        if _id and _id not in ids:
            ids.append(_id)
    if ids:
        return ids

    # 属性为空：用官方转换函数兜底，捕捉 chunk / additional_kwargs 中的隐藏调用
    if _convert_message_to_dict is not None:
        try:
            d = _convert_message_to_dict(msg)
            return [t.get("id") for t in (d.get("tool_calls") or []) if t.get("id")]
        except Exception:
            pass
    return []


def _get_active_tools(ctx: Context) -> tuple[list, frozenset[str]]:
    """根据 Context 返回本轮可用工具列表"""
    if not getattr(ctx, "enable_tools", True):
        return [], frozenset()
    active = []
    for t in TOOLS:
        if getattr(t, "name", None) == "search" and (not ctx.enable_web_search):
            continue
        active.append(t)
    names = frozenset(t.name for t in active if getattr(t, "name", None))
    return active, names


# ==================== 工具目录渲染 ====================
def _render_tool_catalog(tools: List[Any], *, tools_enabled: bool, web_search_enabled: bool) -> str:
    """将 TOOLS 渲染为可读的"工具目录文本"，注入到 SYSTEM_PROMPT"""
    if not tools_enabled:
        return "当前运行配置已禁用工具调用（tools_enabled=false），本轮不允许使用任何工具。"

    if not tools:
        return "当前未配置任何工具。"

    lines: List[str] = ["你当前可使用以下工具："]
    for i, t in enumerate(tools, start=1):
        name = getattr(t, "name", None) or getattr(t, "__name__", "unknow_tool")
        desc = getattr(t, "description", "") or ""
        desc = " ".join(str(desc).strip().split())
        if len(desc) > 240:
            desc = desc[:239] + "..."
        if desc:
            lines.append(f"{i}) {name}：{desc}")
        else:
            lines.append(f"{i}) {name}：暂无描述（建议为 @tool 增加 description）")
    lines.append("当用户询问'你有哪些工具/能做什么'时，只能按上述目录回答，不得编造。【反思机制已激活】如果工具调用失败，系统会反馈错误原因。请仔细阅读错误信息，调整参数后重试，不要盲目重复。")
    return "\n".join(lines)


def _extract_recent_tool_messages(messages: List[AnyMessage]) -> List[ToolMessage]:
    """提取末尾连续的 ToolMessage"""
    # FIX-⑥: 同时跳过反思哨兵(system_monitor)和终止哨兵(system_terminator)，避免两者混用导致提取中断
    _SENTINEL_NAMES = {"system_monitor", "system_terminator"}

    out: List[ToolMessage] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            out.append(m)
            continue
        # 跳过系统注入的哨兵消息，继续往前找
        if isinstance(m, HumanMessage) and getattr(m, "name", None) in _SENTINEL_NAMES:
            continue
        break
    out.reverse()
    return out


RememberedError = Optional[Dict[str, str]]


# 后续需要修改！！！
def find_last_real_human_idx(messages: list) -> int:
    """从后往前找最后一条真实 HumanMessage 的索引，找不到返回 -1。"""
    return next(
        (i for i in range(len(messages) - 1, -1, -1)
         if isinstance(messages[i], HumanMessage)
         and getattr(messages[i], "name", None) not in {"system_monitor", "system_terminator"}),
        -1
    )


def _count_rag_in_current_turn(messages: list) -> int:
    last_human_idx = _find_last_real_human_idx(messages)
    if last_human_idx < 0:
        return 0
    return sum(
        1 for m in messages[last_human_idx:]
        if isinstance(m, ToolMessage)
        and getattr(m, "name", None) == "query_internal_knowledge"
    )


def _tc_name(tc: dict) -> str:
    """统一从 tool_call dict 中提取工具名，兼容两种格式"""
    return tc.get("name") or tc.get("function", {}).get("name") or ""
