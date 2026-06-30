from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.memory.context import Context
from react_agent.core.state import State
from react_agent.utils.llm import load_chat_model
from react_agent.utils.time_utils import _now_iso_in_tz
from react_agent.utils.token_utils import _estimate_openai_cost_usd, \
    _extract_deepseek_v4_usage, bound_tool_payload
from react_agent.utils.tool_utils import _get_active_tools, _tc_name, _render_tool_catalog, \
    _extract_recent_tool_messages, find_last_real_human_idx, _ai_tool_call_ids
from react_agent.utils.tool_helpers import _err

# FIX-④⑤: 提取公共函数，只统计当前轮次（最后一条真实 HumanMessage 之后）的 RAG 调用次数，
#           避免扫全量历史消息导致长对话出现"78次"异常
logger = logging.getLogger(__name__)
_SYSTEM_SENTINEL_NAMES = {"system_monitor", "system_terminator"}


def _count_rag_in_current_turn(messages: list) -> int:
    """从后往前找最后一条真实 HumanMessage（排除系统哨兵消息），
    统计其之后 ToolMessage 中来自 query_internal_knowledge 的调用次数。"""
    last_human_idx = find_last_real_human_idx(messages)
    if last_human_idx < 0:
        return 0
    return sum(
        1 for m in messages[last_human_idx:]
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "query_internal_knowledge"
    )


def _count_tokens_for_trim(messages) -> int:
    """tiktoken cl100k_base 估算；未安装降级为字符数 // 3。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            total += len(enc.encode(str(content)))
            for tc in getattr(m, "tool_calls", None) or []:
                total += len(enc.encode(json.dumps(tc, ensure_ascii=False)))
            total += 4  # 每条消息固定格式开销（role/分隔符），tiktoken 通行经验值
        return total
    except ImportError:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 3


def _collapse_consecutive_humans(messages):
    """合并连续的 HumanMessage,保证 Human/AI 交替,符合 OpenAI/DeepSeek 协议。
    连续多条 HumanMessage 合并为一条(内容用换行拼接),消除 400。"""
    if not messages:
        return messages
    out = []
    for m in messages:
        if (isinstance(m, HumanMessage) and out and isinstance(out[-1], HumanMessage)):
            prev = out[-1]
            merged = (str(prev.content).rstrip() + "\n\n" + str(m.content).lstrip())
            out[-1] = HumanMessage(content=merged)
        else:
            out.append(m)
    return out


def _sanitize_dangling_tool_calls(messages):
    """为缺应答的 tool_call 补占位 ToolMessage，消除悬空导致的 400。

    检测口径统一走 _ai_tool_call_ids（内部优先用官方 _convert_message_to_dict，
    与 DeepSeek 实际收到的一致），覆盖 .tool_calls / .invalid_tool_calls / chunk 残留，
    不会漏掉藏在特殊属性里的隐藏调用。

    注意：相比旧版“转换函数不可用就原样返回”，现在即便 langchain_openai 不可用，
    _ai_tool_call_ids 也会降级为属性读取继续净化，防线不会整体失效。"""
    msgs = list(messages)
    out = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        out.append(m)
        need_ids = _ai_tool_call_ids(m)
        if need_ids:
            answered = []
            j = i + 1
            while j < len(msgs) and isinstance(msgs[j], ToolMessage):
                out.append(msgs[j]); answered.append(msgs[j].tool_call_id); j += 1
            for _id in need_ids:
                if _id not in answered:
                    out.append(ToolMessage(
                        tool_call_id=_id,
                        content='{"ok": false, "error": "工具调用未完成，结果在历史处理中丢失"}'))
                    logger.info("[sanitize] 为悬空 tool_call 补占位 ToolMessage: %s", _id)
            i = j
            continue
        i += 1
    return out


def _prepare_model_messages(messages: list, ctx) -> list:
    """
    消息截断：
    过滤哨兵 → trim → 空结果守卫。返回发给模型的消息列表（永不为空，除非输入本身为空）。"""
    # 仅防御历史持久化中的遗留 sentinel；新代码已不再产生
    msgs = [
        m for m in messages
        if not (isinstance(m, HumanMessage) and getattr(m, "name", None) in _SYSTEM_SENTINEL_NAMES)
    ]
    if not (ctx.enable_history_truncation and ctx.max_history_tokens > 0):
        return _collapse_consecutive_humans(msgs)

    trimmed = trim_messages(
        msgs,
        max_tokens=ctx.max_history_tokens,
        strategy="last",
        token_counter=_count_tokens_for_trim,
        include_system=False,
        allow_partial=False,
        start_on="human",
    )
    if not trimmed:
        # 安全网：宁可略微超预算，也绝不发空上下文。退回到最近一条 Human 及其之后的全部消息。
        last_human = next(
            (i for i in range(len(msgs) - 1, -1, -1) if isinstance(msgs[i], HumanMessage)),
            None,
        )
        trimmed = msgs[last_human:] if last_human is not None else msgs[-1:]
        if getattr(ctx, "debug", False):
            print("[_prepare_model_messages] ⚠️ trim 归零，已回退到最近一个完整轮次")
    return _collapse_consecutive_humans(trimmed)


def _build_model_input(sys_prompt: str, pending: Optional[str], msgs: list) -> list:
    head = [{"role": "system", "content": sys_prompt}]
    if pending:
        head.append({"role": "system", "content": pending})   # 瞬态控制指令，仅本次可见
    return head + msgs


async def call_model(state: State, runtime: Runtime[Context], config: RunnableConfig = None) -> Dict[str, Any]:
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
        consecutive_failure_threshold=ctx.consecutive_failure_threshold,
    )

    # 3) 历史截断（已抽取为 _prepare_model_messages，含空结果守卫）
    msgs = _prepare_model_messages(list(state.messages), ctx)

    if ctx.debug:
        roles = [getattr(m, 'role', type(m).__name__) for m in msgs]
        print(f"[call_model] 裁剪后消息序列: {roles}")

    # 4) 调用（净化在裁剪后、ainvoke 前执行，确保不被裁剪破坏）
    msgs = _sanitize_dangling_tool_calls(msgs)

    try:
        response: AIMessage = await model.ainvoke(
            _build_model_input(sys_prompt, getattr(state, "pending_directive", None), msgs),
            config=config,  # 透传 LangGraph config，使 astream_events 回调链完整
        )
    except Exception as _e:
        _body = getattr(getattr(_e, "response", None), "text", None)
        logger.error("[400-detail] 异常类型=%s", type(_e).__name__)
        logger.error("[400-detail] 异常信息=%s", str(_e))
        logger.error("[400-detail] response.text=%s", _body)
        raise

    # 5) Token 统计
    usage_update = {}
    if ctx.enable_cost_tracking:
        usage = _extract_deepseek_v4_usage(response)
        cost = _estimate_openai_cost_usd(
            model_name=ctx.model.split("/")[-1],
            usage=usage,
            price_table=ctx.deepseek_v4_price,
        )
        usage_update = {
            "cache_hit_tokens":   state.cache_hit_tokens + usage["cache_hit_tokens"],
            "cache_miss_tokens":  state.cache_miss_tokens + usage["cache_miss_tokens"],
            "completion_tokens":  state.completion_tokens + usage["completion_tokens"],
            "reasoning_tokens":   state.reasoning_tokens + usage["reasoning_tokens"],
            "prompt_tokens":      state.prompt_tokens + usage["prompt_tokens"],
            "total_tokens":       state.total_tokens + usage["total_tokens"],
            "estimated_cost_usd": state.estimated_cost_usd + cost,
            "llm_call_count":     state.llm_call_count + 1,
        }

    # ── 兜底保险：RAG 调用次数硬上限 ──
    _RAG_CALL_LIMIT = ctx.rag_call_limit

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
                        f"【本轮检索上限】当前轮次已调用知识库 {current_turn_rag_count} 次，"
                        "本轮不再继续检索。\n"
                        "如您有新的问题或希望换一种方式查询，请继续提问。"
                    ),
                )],
                "step_counter": state.step_counter + 1,
                "pending_directive": None,
            }
            if usage_update:
                out.update(usage_update)
            return out

    # 注：被禁用工具的拦截统一由 dynamic_tool_node 处理，此处无需重复判断
    # ⚠️ 检测口径改用 _ai_tool_call_ids：若模型在最后一步产出的是 invalid_tool_calls
    #    （非法 JSON 参数），response.tool_calls 为空但仍是一条“带工具调用”的消息。
    #    若不在此拦截，它会被路由到 tools → 多走两步 → 触发 GraphRecursionError。
    #    用替换消息（同 id）覆盖原 response，既安全终止又不留悬空。
    if state.is_last_step and _ai_tool_call_ids(response):
        _stopped_tools = [_tc_name(tc) for tc in (response.tool_calls or [])]
        _stopped_tools += [
            itc.get("name") for itc in (getattr(response, "invalid_tool_calls", None) or [])
        ]
        print(
            f"[Guard] is_last_step=True，强制终止；"
            f"step={state.step_counter}，"
            f"rag_count={current_turn_rag_count}，"
            f"未完成工具={_stopped_tools}"
        )

        out = {
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
            "pending_directive": None,
        }

        if usage_update:
            out.update(usage_update)
        return out

    # 8) 正常返回
    out = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
        "pending_directive": None,
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
        try:
            parsed = json.loads(getattr(tm, "content", None) or "{}")
            if not isinstance(parsed, dict):
                raise TypeError(f"工具返回的 JSON 顶层类型应为 object，实际为 {type(parsed).__name__}")
        except Exception as e:
            # ── 解析失败：就地确定 run_ok 和 run_error ──
            payload = {}  # 保留空 dict，防止下游 .get() 崩溃
            run_ok = False
            run_error = {"code": "PARSE_FAILED", "message": f"{type(e).__name__}: {e}"}
        else:
            # ── 解析成功：从 payload 中提取 ──
            payload = parsed
            raw_ok = parsed.get("ok")
            run_ok = bool(raw_ok) if raw_ok is not None else None
            run_error = parsed.get("error")

        tool_name = getattr(tm, "name", "") or "unknown_tool"

        run: Dict[str, Any] = {
            "tool": tool_name,
            "query": payload.get("query"),
            "ok": run_ok,
            "error": run_error,
            "meta": payload.get("meta"),
            "ts": _now_iso_in_tz(ctx.timezone),
        }

        # ✅ FIX: err_inc 之前一直初始化为 0、循环体内从未递增，tool_error_count 是死代码。
        #    现在按"本次工具调用是否失败"真正计数。
        if run_ok is False:
            err_inc += 1

        if payload.get("ok") is True:
            last_ok_result = payload

        runs.append(run)

    # ── 动态推导本轮尾部连续无效 RAG 次数 ──────────────────────────────
    _CONSECUTIVE_FAILURE_THRESHOLD = ctx.consecutive_failure_threshold

    # FIX-④⑤: 使用与 _count_rag_in_current_turn 一致的哨兵排除集，避免全量扫描
    msgs_list = list(state.messages)
    last_human_idx = find_last_real_human_idx(msgs_list)

    # 按时间顺序收集本轮所有 RAG ToolMessage 的 has_relevant_content 结果
    turn_rag_results: List[bool] = []
    if last_human_idx >= 0:
        for m in msgs_list[last_human_idx:]:
            if (isinstance(m, ToolMessage)
                    and getattr(m, "name", None) == "query_internal_knowledge"):
                try:
                    p = json.loads(getattr(m, "content", None) or "{}")
                    hc = p["meta"]["has_relevant_content"]
                except Exception:
                    hc = None
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

    # 连续无效检索达阈值 → 直接 emit 终态拒答 AIMessage（硬护栏）。
    # 由 route_after_postprocess 检测"末条是无 tool_calls 的 AIMessage" → 路由到 __end__。
    if new_consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
        out["messages"] = [AIMessage(
            content="根据内部知识库，未找到相关信息，建议将对应文档导入知识库后重试。"
        )]
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

    return {"pending_directive": reflection_msg}


# ==================== 动态工具节点（修复工具过滤形同虚设的问题）====================
def _bound_tool_messages(messages, max_chars: int):
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(getattr(m, "content", None), str):
            m = m.model_copy(update={"content": bound_tool_payload(m.content, max_chars)})
        out.append(m)
    return out


async def dynamic_tool_node(state: State, config: RunnableConfig, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    ✅ 修复：原版 ToolNode(TOOLS) 挂载全量工具，即使 call_model 通过 _active_tools(ctx)
    过滤了工具，执行层面仍能调用被禁止的工具，权限控制形同虚设。

    本节点在运行时根据 Context 动态构建只含可用工具的 ToolNode，确保过滤真正生效。
    同时对模型请求调用不存在（已被过滤）工具的情况给出明确的错误 ToolMessage，
    而不是让底层 ToolNode 抛出 KeyError。
    """
    try:
        ctx = runtime.context
        active, active_names = _get_active_tools(ctx)

        # ── 治本：先处理 invalid_tool_calls（参数 JSON 解析失败的调用） ──
        # DeepSeek 返回非法 JSON 参数时，LangChain 会把 tool_call 放进
        # .invalid_tool_calls 而非 .tool_calls。如果不处理，这些 tool_call
        # 不会产出 ToolMessage，但 _convert_message_to_dict 仍会将它们序列化
        # 为 tool_calls 发给 DeepSeek → 悬空 → 400 → thread 永久死锁。
        # （配合 routing.route_model_output 的统一检测，此分支现在一定能被走到。）
        last_msg = state.messages[-1] if state.messages else None
        invalid_messages = []
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "invalid_tool_calls", None):
            for itc in last_msg.invalid_tool_calls:
                itc_id = itc.get("id") or ""
                itc_name = itc.get("name") or "unknown"
                itc_error = itc.get("error") or "参数 JSON 解析失败"
                itc_args = itc.get("args") or ""
                invalid_messages.append(ToolMessage(
                    tool_call_id=itc_id,
                    name=itc_name,
                    content=json.dumps(_err(
                        tool_name=itc_name,
                        query=str(itc_args)[:200],
                        code="INVALID_TOOL_CALL",
                        message=(
                            f"工具参数 JSON 解析失败：{str(itc_error)[:300]}。"
                            f"请检查参数格式后重新调用。"
                        ),
                    ), ensure_ascii=False),
                ))
            logger.warning(
                "[dynamic_tool_node] 检测到 %d 条 invalid_tool_calls，已生成错误 ToolMessage",
                len(invalid_messages),
            )
            # 如果只有 invalid 没有 valid，直接返回错误消息
            if not last_msg.tool_calls:
                return {"messages": _bound_tool_messages(invalid_messages, ctx.max_tool_output_chars)}

        # 检查最后一条 AIMessage 里哪些 tool_calls 被禁用、哪些允许
        blocked_calls = []
        allowed_calls = []
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                name = _tc_name(tc)
                if name and name not in active_names:
                    blocked_calls.append(tc)
                else:
                    allowed_calls.append(tc)

        # 全部允许：保持原有快速路径，行为不变
        if not blocked_calls:
            node = ToolNode(active, handle_tool_errors=True)
            result = await node.ainvoke(state, config)
            all_msgs = invalid_messages + result.get("messages", [])
            result["messages"] = _bound_tool_messages(all_msgs, ctx.max_tool_output_chars)
            return result

        # 被禁用工具的调用：直接构造错误 ToolMessage，不执行
        blocked_messages = [
            ToolMessage(
                tool_call_id=tc.get("id", ""),
                name=_tc_name(tc) or "unknown",
                content=json.dumps(_err(
                    tool_name=_tc_name(tc) or "unknown",
                    query=str(tc.get("args", "")),
                    code="TOOL_DISABLED",
                    message=f"工具 '{_tc_name(tc)}' 在当前配置下已被禁用。可用工具：{sorted(active_names)}。",
                ), ensure_ascii=False),
            )
            for tc in blocked_calls
        ]

        # ✅ 修复：混合场景下不能直接 return，否则 allowed_calls 永远不会被执行，
        #    下一轮 AIMessage 里的 tool_calls 数量与 ToolMessage 响应数量不一致，
        #    违反 OpenAI/Anthropic 的 tool-use 协议约束（每个 tool_call_id 必须有对应响应）。
        #    这里构造一条只含允许 tool_calls 的临时 AIMessage，喂给 ToolNode 正常执行，
        #    再把两部分 ToolMessage 合并返回，确保数量始终对齐。
        tool_result_messages = []
        if allowed_calls:
            filtered_ai_message = last_msg.model_copy(update={"tool_calls": allowed_calls})
            transient_messages = list(state.messages[:-1]) + [filtered_ai_message]
            node = ToolNode(active, handle_tool_errors=True)
            invoke_result = await node.ainvoke(transient_messages, config)
            tool_result_messages = invoke_result.get("messages", [])

        return {"messages": _bound_tool_messages(invalid_messages + blocked_messages + tool_result_messages, ctx.max_tool_output_chars)}
    except Exception as e:
        # ══════ 兜底：扫描本轮所有未应答的 tool_call，全部补 ToolMessage ══════
        # 检测口径统一走 _ai_tool_call_ids（与 DeepSeek 实际收到的一致），同时覆盖
        # .tool_calls 与 .invalid_tool_calls，并且【不假设带工具调用的 AIMessage 在末尾】。
        logger.error(f"[dynamic_tool_node] 未预期异常：{type(e).__name__}: {e}", exc_info=True)

        # 从后往前找最近一条“DeepSeek 视角下带 tool_call”的 AIMessage
        target_idx = -1
        need_ids: List[str] = []
        for _idx in range(len(state.messages) - 1, -1, -1):
            ids = _ai_tool_call_ids(state.messages[_idx])
            if ids:
                target_idx = _idx
                need_ids = ids
                break

        if target_idx >= 0:
            # 收集该 AIMessage 之后已存在的 ToolMessage 应答
            answered_ids = {
                msg.tool_call_id
                for msg in state.messages[target_idx + 1:]
                if isinstance(msg, ToolMessage)
            }
            # 只为尚未应答的 tool_call 补占位，避免重复
            missing = [
                ToolMessage(
                    tool_call_id=_id,
                    name="unknown",
                    content=json.dumps(_err(
                        tool_name="unknown",
                        query="",
                        code="TOOL_NODE_CRASH",
                        message=f"工具执行层异常：{type(e).__name__}: {e}",
                    ), ensure_ascii=False),
                )
                for _id in need_ids
                if _id not in answered_ids
            ]
            if missing:
                logger.error(f"[dynamic_tool_node] 兜底补齐 {len(missing)} 条 ToolMessage")
                _max_chars = 4000
                try:
                    _max_chars = runtime.context.max_tool_output_chars
                except Exception:
                    pass
                return {"messages": _bound_tool_messages(missing, _max_chars)}
        # 实在找不到可补的，再抛
        raise