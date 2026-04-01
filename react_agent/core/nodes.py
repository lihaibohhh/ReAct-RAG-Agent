from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.memory.context import Context
from react_agent.core.state import State
from react_agent.utils.llm import load_chat_model
from react_agent.utils.time_utils import _now_iso_in_tz
from react_agent.utils.token_utils import _trim_text, _estimate_openai_cost_usd, _extract_openai_usage_from_ai_message
from react_agent.utils.tool_utils import _get_active_tools, _tc_name, _render_tool_catalog, _parse_tool_payload, _extract_recent_tool_messages


# FIX-④⑤: 提取公共函数，只统计当前轮次（最后一条真实 HumanMessage 之后）的 RAG 调用次数，
#           避免扫全量历史消息导致长对话出现"78次"异常
_SYSTEM_SENTINEL_NAMES = {"system_monitor", "system_terminator"}


def _count_rag_in_current_turn(messages: list) -> int:
    """从后往前找最后一条真实 HumanMessage（排除系统哨兵消息），
    统计其之后 ToolMessage 中来自 query_internal_knowledge 的调用次数。"""
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and getattr(m, "name", None) not in _SYSTEM_SENTINEL_NAMES:
            last_human_idx = i
            break
    if last_human_idx < 0:
        return 0
    return sum(
        1 for m in messages[last_human_idx:]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "query_internal_knowledge"
    )


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """调用主模型，并把 AIMessage 追加到 state.messages"""
    ctx = runtime.context

    # 1) 初始化模型
    base_model = load_chat_model(ctx.model)
    tools_enabled = bool(getattr(ctx, "enable_tools", True))
    active_tools, _ = _get_active_tools(ctx)
    model = base_model.bind_tools(active_tools) if tools_enabled else base_model

    # 2) 系统提示
    tool_catalog = _render_tool_catalog(
        active_tools,
        tools_enabled=tools_enabled,
        web_search_enabled=ctx.enable_web_search,
    )
    system_time = _now_iso_in_tz(ctx.timezone)
    sys_prompt = ctx.system_prompt.format(
        system_time=system_time,
        tool_catalog=tool_catalog,
    )

    # 3) 历史截断（用 trim_messages 保证 tool_calls/tool 原子对不被拆散）
    msgs = list(state.messages)
    # FIX-⑧: trim_messages 前先过滤哨兵 HumanMessage，防止 trim 将哨兵保留为序列首条，
    #         破坏 start_on="human" 语义（哨兵不是真实用户输入）
    msgs = [
        m for m in msgs
        if not (isinstance(m, HumanMessage) and getattr(m, "name", None) in _SYSTEM_SENTINEL_NAMES)
    ]
    if ctx.enable_history_truncation and ctx.max_history_tokens > 0:

        def _count_tokens(messages) -> int:
            """
            用 tiktoken cl100k_base 估算 token 数。
            DeepSeek / GPT-4 系列均使用 cl100k_base 编码，误差极小。
            tiktoken 未安装时自动降级为字符数 ÷ 3（保守估算）。
            """
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                total = 0
                for m in messages:
                    content = getattr(m, "content", "") or ""
                    if isinstance(content, list):
                        # 多模态消息：只统计文本部分
                        content = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict)
                        )
                    total += len(enc.encode(str(content)))
                return total
            except ImportError:
                # 降级：字符数 ÷ 3 是业界常用的保守估算
                total = sum(len(str(getattr(m, "content", ""))) for m in messages)
                return total // 3

        msgs = trim_messages(
            msgs,
            max_tokens=ctx.max_history_tokens,
            strategy="last",
            token_counter=_count_tokens,  # ✅ 换成自定义函数
            include_system=False,
            allow_partial=False,
            start_on="human",
        )

    # 临时调试：确认消息序列合法，验证通过后删掉这段
    print(ctx.debug)
    if ctx.debug:
        roles = [getattr(m, 'role', type(m).__name__) for m in msgs]
        print(f"[call_model] 裁剪后消息序列: {roles}")

    # 4) 调用
    response: AIMessage = await model.ainvoke([{"role": "system", "content": sys_prompt}] + msgs)

    # 5) Token 统计
    usage_update = {}
    if ctx.enable_cost_tracking:
        usage = _extract_openai_usage_from_ai_message(response)
        cost = _estimate_openai_cost_usd(
            model_name=ctx.model.split("/")[-1],
            usage=usage,
            price_table=ctx.openai_price_per_1m_tokens,
        )
        usage_update = {
            "prompt_tokens": state.prompt_tokens + usage["prompt_tokens"],
            "completion_tokens": state.completion_tokens + usage["completion_tokens"],
            "total_tokens": state.total_tokens + usage["total_tokens"],
            "cached_tokens": state.cached_tokens + usage["cached_tokens"],
            "estimated_cost_usd": state.estimated_cost_usd + cost,
            "llm_call_count": state.llm_call_count + 1,
        }

    # ── 兜底保险：RAG 调用次数硬上限 ──
    # 工程原则：永远不信任上层逻辑（提示词/模型）能100%按预期停止
    # 无论模型出于什么理由想继续检索，超过上限就强制终止
    # 这和 recursion_limit、with_retry 的 max_retries 是同一类防御
    _RAG_CALL_LIMIT = 3

    # FIX-④⑤: 替换为公共函数，只统计当前轮次的 RAG 调用次数
    msgs_list = list(state.messages)
    current_turn_rag_count = _count_rag_in_current_turn(msgs_list)

    if current_turn_rag_count >= _RAG_CALL_LIMIT and response.tool_calls:
        # 判断本次想调用的工具里是否还有 RAG
        wants_rag = any(
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
            == "query_internal_knowledge"
            for tc in response.tool_calls
        )
        if wants_rag:
            print(f"[Guard] RAG 调用次数已达上限 {_RAG_CALL_LIMIT}，强制终止检索")
            out = {
                "messages": [AIMessage(
                    id=response.id,
                    content=(
                        f"根据内部知识库，已检索 {current_turn_rag_count} 次，"
                        "未找到与您问题完全匹配的内容。\n"
                        "知识库中可能存在部分相关内容，但缺少您询问的具体数据。\n"
                        "建议：将包含该数据的文档导入知识库后重试。"
                    ),
                )],
                "step_counter": state.step_counter + 1,
            }
            if usage_update:
                out.update(usage_update)
            return out

    # 注：被禁用工具的拦截统一由 dynamic_tool_node 处理，此处无需重复判断
    if state.is_last_step and response.tool_calls:
        _stopped_tools = [_tc_name(tc) for tc in response.tool_calls]
        print(
            f"[Guard] is_last_step=True，强制终止；"
            f"step={state.step_counter}，"
            f"rag_count={current_turn_rag_count}，"
            f"未完成工具={_stopped_tools}"
        )
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content=(
                        "抱歉，我在限定的推理步数内仍需要调用工具才能给出可靠结论，"
                        "因此已停止继续搜索/调用工具。\n"
                        "你可以：\n"
                        "- 提高 Context.recursion_limit；或\n"
                        "- 进一步澄清问题/提供更多上下文。"
                    ),
                )
            ],
            "step_counter": state.step_counter + 1,
            "debug_log": (
            f"[call_model] is_last_step=True; "
            f"step_counter={state.step_counter}; "
            f"rag_count={current_turn_rag_count}; "
            f"tool_calls={_stopped_tools}; "
            f"last_msg_type={type(state.messages[-1]).__name__}"
            ),
        }

    # 8) 正常返回
    out = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
    }

    if usage_update:
        out.update(usage_update)

    return out


# ==================== Node: 工具后处理 ====================
async def postprocess_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """解析工具返回，写入 State.tool_runs"""
    ctx = runtime.context
    tool_msgs = _extract_recent_tool_messages(list(state.messages))
    if not tool_msgs:
        return {}

    runs: List[Dict[str, Any]] = []
    err_inc = 0
    last_ok_result: Optional[Dict[str, Any]] = None

    for tm in tool_msgs:
        payload, parse_err = _parse_tool_payload(getattr(tm, "content", None))

        # 截断保护
        if payload is not None:
            if isinstance(payload, dict) and "data" in payload:
                try:
                    dumped = json.dumps(payload["data"], ensure_ascii=False)
                    if len(dumped) > ctx.max_tool_output_chars:
                        payload["truncated"] = True
                        payload["data"] = _trim_text(dumped, ctx.max_tool_output_chars)
                    else:
                        payload.setdefault("truncated", False)
                except Exception:
                    payload["truncated"] = True
                    payload["data"] = _trim_text(str(payload.get("data")), ctx.max_tool_output_chars)
            else:
                try:
                    dumped = json.dumps(payload, ensure_ascii=False)
                    if len(dumped) > ctx.max_tool_output_chars:
                        payload = {
                            "truncated": True,
                            "data": _trim_text(dumped, ctx.max_tool_output_chars),
                        }
                except Exception:
                    payload = {
                        "truncated": True,
                        "data": _trim_text(str(payload), ctx.max_tool_output_chars),
                    }

        tool_name = getattr(tm, "name", "") or "unknown_tool"
        query = payload.get("query") if isinstance(payload, dict) else None
        ok = payload.get("ok") if isinstance(payload, dict) else None

        run: Dict[str, Any] = {
            "tool": tool_name,
            "query": query,
            "ok": bool(ok) if ok is not None else None,
            "error": payload.get("error") if isinstance(payload, dict) else None,
            "meta": payload.get("meta") if isinstance(payload, dict) else None,
            "ts": _now_iso_in_tz(ctx.timezone),
        }

        if parse_err is not None:
            run["ok"] = False
            run["error"] = parse_err
            err_inc += 1
        else:
            if run["ok"] is False:
                err_inc += 1
            if isinstance(payload, dict) and payload.get("ok") is True:
                last_ok_result = payload

        runs.append(run)

    # ── 动态推导本轮尾部连续无效 RAG 次数 ──────────────────────────────
    # FIX-⑩: 两侧对称确认：
    #   - call_model 侧：用 _count_rag_in_current_turn() 从消息重新计算 RAG 总次数，
    #     不读取也不修改 consecutive_failures，无累积。
    #   - postprocess_tools 侧（此处）：用 turn_rag_results 从消息重新计算连续失败数，
    #     以 new_consecutive_failures 覆写字段，无累积（旧的 err_inc 累积写法已删除）。
    _CONSECUTIVE_FAILURE_THRESHOLD = 2

    # FIX-④⑤: 使用与 _count_rag_in_current_turn 一致的哨兵排除集，避免全量扫描
    msgs_list = list(state.messages)
    last_human_idx = next(
        (i for i in range(len(msgs_list) - 1, -1, -1)
         if isinstance(msgs_list[i], HumanMessage)
         and getattr(msgs_list[i], "name", None) not in _SYSTEM_SENTINEL_NAMES),
        -1
    )

    # 按时间顺序收集本轮所有 RAG ToolMessage 的 has_relevant_content 结果
    turn_rag_results: List[bool] = []
    if last_human_idx >= 0:
        for m in msgs_list[last_human_idx:]:
            if (isinstance(m, ToolMessage)
                    and getattr(m, "name", None) == "query_internal_knowledge"):
                p, _ = _parse_tool_payload(getattr(m, "content", None))
                if isinstance(p, dict):
                    data = p.get("data", {})
                    hc = data.get("has_relevant_content") if isinstance(data, dict) else None
                    if hc is not None:
                        turn_rag_results.append(bool(hc))

    # 统计尾部连续失败次数：遇到第一个"有内容"就截断，不跨越成功记录
    new_consecutive_failures = 0
    for hc in reversed(turn_rag_results):
        if not hc:
            new_consecutive_failures += 1
        else:
            break  # 成功记录截断，之前的失败不属于"连续"

    out: Dict[str, Any] = {
        "tool_runs": runs,
        "consecutive_failures": new_consecutive_failures,  # 每次都覆写，不累积
    }

    # 连续无效检索达到阈值 → 注入系统终止消息，强迫模型输出拒答
    if new_consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
        termination_msg = HumanMessage(
            content=(
                f"【系统拒答指令】内部知识库已连续 {new_consecutive_failures} 次"
                "未找到相关内容（has_relevant_content=False）。\n"
                "请立即停止检索，直接输出以下拒答回复，不得再调用任何工具：\n"
                "「根据内部知识库，未找到相关信息，建议将对应文档导入知识库后重试。」"
            ),
            name="system_terminator",  # FIX-⑥: 终止指令专用名，与反思哨兵 system_monitor 区分
        )
        out["messages"] = [termination_msg]
    if err_inc:
        out["tool_error_count"] = state.tool_error_count + err_inc
    if last_ok_result is not None:
        out["last_tool_result"] = last_ok_result

    return out


# ==================== ✅ Node: 反思 (Reflection) ====================
async def reflection_node(state: State) -> Dict[str, Any]:
    """
    【反思节点】
    检查上一步工具调用是否出错。
    如果出错，插入一条 System/Human 提示，强迫模型注意。
    """
    # 1. 获取最近一次工具运行记录
    if not state.tool_runs:
        return {}

    last_run = state.tool_runs[-1]

    # 2. 如果成功，什么都不做，直接过
    if last_run.get("ok", True):  # 默认为 True (宽容模式)
        return {}

    # 3. 如果失败，构造反思提示
    # 我们用 HumanMessage 模拟“系统监控员”的反馈，这通常比 SystemMessage 对模型影响更直接
    error_info = last_run.get("error") or "未知错误"

    # 格式化错误信息（如果是 dict 把它转字符串）
    if isinstance(error_info, dict):
        error_info = f"{error_info.get('code', 'ERR')}: {error_info.get('message', str(error_info))}"

    reflection_msg = (
        f"【系统监控警告】上一步工具 {last_run.get('tool')} 调用失败。\n"
        f"错误信息：{str(error_info)[:300]}\n"
        f"⚠️ 请务必分析错误原因！检查你传入的参数格式、类型或逻辑。\n"
        f"请修正参数后重试，或者尝试使用其他替代方案。"
    )

    print(f"\n[Reflection] 🛑 检测到工具错误，正在注入反思提示...\n")

    return {
        # FIX-⑥: 反思警告保持 system_monitor，终止指令用 system_terminator，两者不可互换
        "messages": [HumanMessage(content=reflection_msg, name="system_monitor")]
    }


# ==================== 动态工具节点（修复工具过滤形同虚设的问题）====================
async def dynamic_tool_node(state: State, config: RunnableConfig, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    ✅ 修复：原版 ToolNode(TOOLS) 挂载全量工具，即使 call_model 通过 _active_tools(ctx)
    过滤了工具，执行层面仍能调用被禁止的工具，权限控制形同虚设。

    本节点在运行时根据 Context 动态构建只含可用工具的 ToolNode，确保过滤真正生效。
    同时对模型请求调用不存在（已被过滤）工具的情况给出明确的错误 ToolMessage，
    而不是让底层 ToolNode 抛出 KeyError。
    """
    ctx = runtime.context
    active, active_names = _get_active_tools(ctx)

    # 检查最后一条 AIMessage 是否包含被禁用工具的调用
    last_msg = state.messages[-1] if state.messages else None
    blocked_calls = []
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            name = _tc_name(tc)
            if name and name not in active_names:
                blocked_calls.append(tc)

    # 如果有被禁用工具的调用，直接返回错误 ToolMessage，不执行
    if blocked_calls:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    name=_tc_name(tc) or "unknown",
                    content=json.dumps({
                        "ok": False,
                        "error": {
                            "code": "TOOL_DISABLED",
                            "message": (
                                f"工具 '{_tc_name(tc)}' 在当前配置下已被禁用。"
                                f"可用工具：{sorted(active_names)}。"
                            ),
                        },
                    }, ensure_ascii=False),
                )
                for tc in blocked_calls
            ]
        }

    # 正常路径：只用可用工具构建 ToolNode 并执行
    node = ToolNode(active)
    return await node.ainvoke(state, config)