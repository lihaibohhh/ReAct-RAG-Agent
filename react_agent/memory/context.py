# SNAPSHOT: (2026-01-31-14:26)
"""
定义 Agent 的“运行时可配置参数”（工程可用版，中文情景默认）
- 所有“会变的东西”集中放在 Context：模型、系统提示、搜索参数、步数限制、调试开关、截断策略等
- 支持通过环境变量覆盖默认值（例如：MODEL / MAX_SEARCH_RESULTS / DEBUG 等）
- 从环境变量读取时做类型转换（避免 int/bool 变成 str）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Optional, get_args, get_origin
from typing import get_type_hints

from react_agent.core import prompts


def _to_bool(s: str):
    s = (s or "").strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _unwrap_annotated(tp: Any):
    """取出Annotaed[T, ...]的T"""
    if get_origin(tp) is Annotated:
        return get_args(tp)[0]
    return tp


def _coerce(value: str, tp: Any):
    """把环境变量字符串 value 转成字段类型 tp。"""
    tp = _unwrap_annotated(tp)

    # Optional[T] / Union[T, None]
    origin = get_origin(tp)
    if origin is None:
        # 不是泛型
        if tp is bool:
            return _to_bool(value)
        if tp is int:
            return int(value)
        if tp is float:
            return float(value)

    # 处理Union / Optional
    if origin is list:
        # 支持用逗号分隔的list[str]（如果未来需要）
        inner = get_args(tp)[0] if get_args(tp) else str
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [_coerce(p, inner) for p in parts]

    if origin is dict:
        # 不建议用env传dict；保持原样
        return value

    if origin is tuple:
        return value

    if origin is type(None):
        return None

    if origin is Any:
        return value

    if origin is getattr(__import__("typing"), "Union", None):
        # Union[T1, T2, ...]：按顺序尝试转换
        for cand in get_args(tp):
            if cand is type(None):
                continue
            try:
                return _coerce(value, cand)
            except Exception:
                pass
        return value

    # 其他类型：保持字符串
    return value


@dataclass(kw_only=True)
class Context:
    """
    Agent 上下文（可配置参数集合）

    设计理念：
    - “可变参数”全部集中在这里；graph/tools/state 通过 runtime.context 读取
    - 默认值面向中文用户使用场景
    - 允许用环境变量覆盖（字段名大写），例如：
      - MODEL="openai/gpt-4o-mini"
      - MAX_SEARCH_RESULTS="5"
      - DEBUG="true"
    """

    # -----------------------------
    # Prompt / 语言与地域
    # -----------------------------

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={"description": "系统提示词（中文场景默认）。支持 {system_time} 变量注入。"},
    )

    language: str = field(
        default="zh-CN",
        metadata={"description": "默认语言。中文场景建议保持 zh-CN。"},
    )

    timezone: str = field(
        default="Asia/Shanghai",
        metadata={"description": "默认时区（IANA）。用于展示 system_time、时间相关回答等。"},
    )

    # -----------------------------
    # 模型与推理参数（可按你的 provider/model 体系扩展）
    # -----------------------------

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={"description": "主模型名：provider/model-name。"},
    )

    temperature: float = field(
        default=0.2,
        metadata={"description": "采样温度。中文严谨问答建议 0~0.3。"},
    )

    max_output_tokens: int = field(
        default=1024,
        metadata={"description": "模型单次输出 token 上限（若底层模型/SDK支持则生效）。"},
    )

    enable_tools: bool = field(
        default=True,
        metadata={"description": "是否允许使用任何工具（总开关）。关闭后所有工具不可用"}
    )

    # -----------------------------
    # 搜索工具（Tavily）相关
    # -----------------------------

    enable_web_search: bool = field(
        default=True,
        metadata={"description": "是否允许使用 web 搜索工具（Tavily）。"},
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "搜索工具返回的最大结果条数（传给 TavilySearch）。"},
    )

    # 工具返回体积控制：你 tools.py 已经 shrink 了，这里再给一个全局硬限制（建议）
    max_tool_output_chars: int = field(
        default=4000,
        metadata={"description": "工具 observation 的硬截断字符数（避免上下文污染）。"},
    )

    # -----------------------------
    # 图运行与安全兜底
    # -----------------------------

    recursion_limit: int = field(
        default=50,
        metadata={"description": "LangGraph 递归/步数上限（防止死循环）。"},
    )

    tool_timeout_seconds: int = field(
        default=30,
        metadata={"description": "工具调用超时（秒）。你可在工具层或 node 层统一应用。"},
    )

    # -----------------------------
    # 上下文与历史消息截断策略
    # -----------------------------

    # 只按条数截断是最简单可靠的工程策略（后续可升级成按 token 预算）
    max_history_tokens: int = field(
        default=6000,
        metadata={"description": "发给模型前保留的最大历史 token 数（不含本轮 system prompt）。"
                                 "建议设为模型上下文窗口的 40%~60%，给 completion 留足空间。"},
    )

    # 如果你在 graph.py 里实现摘要/压缩，可用此开关控制
    enable_history_truncation: bool = field(
        default=True,
        metadata={"description": "是否启用历史消息截断/压缩策略。"},
    )

    # -----------------------------
    # 会话线程 / 对话历史持久化（工程化必备之一）
    # -----------------------------
    # FIX-2: 将 memory 的描述从"推荐"改为明确警告，防止误用于生产环境
    checkpoint_backend: str = field(
        default="memory",
        metadata={"description": "对话历史后端：memory / sqlite / postgres / none。"
                                 "警告：memory 仅用于本地调试，生产环境禁止使用，无清理机制。"},
    )

    # FIX-⑫: 修正注释矛盾——sqlite 模式完整实现，见 checkpointer.py 的 _create_sqlite
    checkpoint_db_path: str = field(
        default="./.agent_checkpoints.sqlite3",
        metadata={"description": "当 checkpoint_backend=sqlite 时使用的数据库文件路径。sqlite 模式完整实现，见 checkpointer.py 的 _create_sqlite。"},
    )

    # FIX-1: default_thread_id 改为 None，防止不同用户共享同一 "default" thread
    default_thread_id: Optional[str] = field(
        default=None,
        metadata={"description": "未显式传 thread_id 时使用的默认会话ID。设计规范：调用方必须传入 user:{username} 格式的 thread_id，此字段保留为 None 以强制调用方显式传值。"},
    )

    # -----------------------------
    # 调试与可观测性
    # -----------------------------

    debug: bool = field(
        default=True,
        metadata={"description": "调试模式：更详细的日志/trace。"},
    )

    log_tool_observations: bool = field(
        default=True,
        metadata={"description": "是否记录工具 observation（写入 trace / 打印）。"},
    )

    # -----------------------------
    # 模型单价表”和开关
    # -----------------------------
    enable_cost_tracking: bool = field(
        default=True,
        metadata={"description": "是否统计 token 与估算费用（OpenAI usage）"},
    )
    # 例："input": 0.80表示0.8美元 / 100 万 tokens
    openai_price_per_1m_tokens: dict = field(
        default_factory=lambda: {
            "gpt-4.1-mini": {"input": 0.80, "output": 3.20, "cached_input": 0.20},
            "gpt-4.1": {"input": 3.00, "output": 12.00, "cached_input": 0.75},
            "gpt-4o-mini": {"input": 0.30, "output": 1.20, "cached_input": 0.15},
        },
        metadata={"description": "OpenAI 模型单价（$/1M tokens），用于估算成本"},
    )

    # -----------------------------
    # env 覆盖逻辑
    # -----------------------------

    def __post_init__(self) -> None:
        # 关键：拿到“已解析”的真实类型（支持 Annotated）
        type_hints = get_type_hints(self.__class__, include_extras=True)

        for f in fields(self):
            if not f.init:
                continue

            current = getattr(self, f.name)
            if current != f.default:
                continue

            env_key = f.name.upper()
            raw = os.environ.get(env_key)
            if raw is None or raw.strip() == "":
                continue

            tp = type_hints.get(f.name, f.type)  # ✅ 用解析后的真实类型

            try:
                setattr(self, f.name, _coerce(raw, tp))
            except Exception:
                # 更工程化：转换失败就忽略覆盖（不把字符串写进去埋雷）
                if getattr(self, "debug", False):
                    print(f"[Context] ⚠️ env {env_key}={raw!r} 无法转换为 {tp}，已忽略覆盖")
                continue


