from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage
from react_agent.memory.context import Context
from react_agent.tools import TOOLS


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


def _parse_tool_payload(content: Any) -> Tuple[Optional[dict], RememberedError]:
    """把 ToolMessage.content 尽量解析成 dict"""
    if isinstance(content, dict):
        return content, None

    if isinstance(content, str):
        text = content.strip()

        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj, None
                return {"data": obj}, None
            except Exception as e:
                return None, {
                    "code": "TOOL_PAYLOAD_PARSE_FAILED",
                    "message": f"JSON解析失败：{type(e).__name__}: {e}",
                }

        return {"data": text}, None

    return {"data": content}, None


def _tc_name(tc: dict) -> str:
    """统一从 tool_call dict 中提取工具名，兼容两种格式"""
    return tc.get("name") or tc.get("function", {}).get("name") or ""