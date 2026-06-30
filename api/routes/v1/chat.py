"""
v1 Chat routes — Phase 1: token 级流式 SSE；Phase 2: auth + 限流 + 预算。

关键设计：
- astream_events(v2) + call_model config 透传 → on_chat_model_stream 逐 token 推送
- asyncio.Task + Queue 解耦生产与消费，使 is_disconnected() 可以在帧间轮询
- 手动 deadline 实现总预算超时（Python 3.10 无 asyncio.timeout）
- error 帧与 errors.py problem+json 同构（含 request_id）
- usage/cost 用 _extract_deepseek_v4_usage + _estimate_openai_cost_usd，不裸用 usage_metadata
- Phase 2: Depends(require_api_key) → check_rate_limit → check_token_budget → 处理 → record_token_usage
- 扣费点：generator finally（断连/超时/正常完成均记实际 token，不按预估）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from api.dependencies import get_agent
from api.models import ChatRequest, ChatResponse
from api.security import require_api_key, get_rate_limit_key
from api.ratelimit import check_rate_limit, check_token_budget, record_token_usage
from react_agent.core.agent import PersistentAgent
from react_agent.utils.token_utils import (
    _estimate_openai_cost_usd,
    _extract_deepseek_v4_usage,
)

_logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat v1"])


# ── SSE 序列化 ─────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 事件 → SSE 帧映射 ─────────────────────────────────────────────────────────
def _map_event(ev: dict, state: dict, agent: PersistentAgent, start: float) -> str | None:
    """
    把 astream_events v2 原始事件映射为 SSE 帧字符串。
    返回 None 表示不需要推送。
    state 是可变 dict，跨帧共享 ttft_ms / total_tokens / total_cost。
    """
    name = ev["event"]
    data = ev.get("data", {})

    if name == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            if state["ttft_ms"] is None:
                state["ttft_ms"] = round((time.perf_counter() - start) * 1000)
            return _sse("token", {"delta": chunk.content})

    elif name == "on_tool_start":
        return _sse("tool_call", {
            "name": ev.get("name", ""),
            "args": data.get("input", {}),
        })

    elif name == "on_tool_end":
        out = data.get("output", "")
        # 工具返回 _ok/_err 结构时取 ok 字段，否则默认成功
        ok = True
        if isinstance(out, dict):
            ok = bool(out.get("ok", True))
        elif isinstance(out, str):
            try:
                parsed = json.loads(out)
                ok = bool(parsed.get("ok", True))
            except Exception:
                pass
        return _sse("tool_result", {
            "name": ev.get("name", ""),
            "ok": ok,
            "meta": str(out)[:300] if out else "",
        })

    elif name == "on_chat_model_end":
        output = data.get("output")
        if output is None:
            return None
        # on_chat_model_end 的 output 可能是 AIMessage 或 LLMResult
        msg = output
        if hasattr(output, "generations") and output.generations:
            g = output.generations[0]
            if isinstance(g, list) and g:
                g = g[0]
            msg = getattr(g, "message", output)

        usage = _extract_deepseek_v4_usage(msg)
        # 优先用 response_metadata.model_name（API 返回的真实型号，如 "deepseek-v4-flash"）
        # ctx.model 是配置名（"deepseek/deepseek-chat"），不一定与价格表 key 匹配
        rm = getattr(msg, "response_metadata", None) or {}
        model_name = rm.get("model_name") or agent.ctx.model.split("/")[-1]
        cost = _estimate_openai_cost_usd(
            model_name=model_name,
            usage=usage,
            price_table=agent.ctx.deepseek_v4_price,
        )
        state["total_tokens"] += usage.get("total_tokens", 0)
        state["prompt_tokens"] += usage.get("prompt_tokens", 0)
        state["completion_tokens"] += usage.get("completion_tokens", 0)
        state["total_cost"] += cost
        if not state["model_name"]:
            state["model_name"] = model_name
        return _sse("usage", {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": cost,
        })

    return None


# ── SSE 生成器（核心） ─────────────────────────────────────────────────────────
async def _stream_v1_generator(
    request: Request,
    agent: PersistentAgent,
    message: str,
    thread_id: str,
    session_id: str,
    bucket_key: str = "",
) -> AsyncGenerator[str, None]:
    """
    Token 级 SSE 生成器。

    架构：
    - _fill_task：asyncio.Task，驱动 agent.stream_events() 并将 SSE 帧放入 Queue
    - 主循环：用 asyncio.wait_for(q.get(), timeout=0.1) 拉帧，兼顾 is_disconnected 轮询
    - 超时：手动 deadline（Python 3.10 兼容），超时发 error+done 并 cancel task
    - 断连：每帧后检查 is_disconnected()，检测到就 cancel task 并静默 return（不发 done）
    - error 帧：与 errors.py problem+json 同构，含 request_id

    CancelledError 单独捕获并 raise，两条退出路径（断连/外部取消）都记 stream_cancelled。
    """
    from api.middleware import request_id_ctx
    from api.settings import APISettings

    # 读 stream_timeout_s，避免在模块层面引入循环依赖
    try:
        _timeout_s: float = APISettings().stream_timeout_s
    except Exception:
        _timeout_s = 120.0

    start = time.perf_counter()
    deadline = start + _timeout_s
    request_id = request_id_ctx.get("") or None
    state: dict = {
        "ttft_ms": None,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
        "model_name": "",  # response_metadata 真实型号，与 Prometheus label 口径一致
    }
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)  # 背压：防止生产者跑太快撑爆内存

    # ── 生产者 Task ────────────────────────────────────────────────────────────
    async def _fill_task() -> None:
        try:
            async for ev in agent.stream_events(
                messages=[HumanMessage(content=message)],
                thread_id=thread_id,
            ):
                frame = _map_event(ev, state, agent, start)
                if frame:
                    await q.put(frame)
        except asyncio.CancelledError:
            _logger.info(
                "stream_cancelled | session=%s request_id=%s", session_id, request_id
            )
            raise
        except Exception as exc:
            _logger.exception("stream error | session=%s: %s", session_id, exc)
            await q.put(_sse("error", {
                "type": "about:blank",
                "title": "Internal Error",
                "status": 500,
                "detail": "Agent 内部错误，请稍后重试",
                "request_id": request_id,
            }))
        finally:
            await q.put(None)  # 哨兵，通知消费者流结束

    task = asyncio.create_task(_fill_task())
    timed_out = False

    # ── 消费者主循环 ────────────────────────────────────────────────────────────
    try:
        while True:
            now = time.perf_counter()

            # 全局超时检查
            if now >= deadline:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                timed_out = True
                _logger.warning(
                    "stream_timeout | session=%s timeout_s=%s", session_id, _timeout_s
                )
                yield _sse("error", {
                    "type": "about:blank",
                    "title": "Stream Timeout",
                    "status": 504,
                    "detail": f"LLM 响应超时（{_timeout_s}s），请重试",
                    "request_id": request_id,
                })
                break

            # 等待下一帧（0.1s 超时，给 disconnect 检查留窗口）
            remaining = deadline - now
            try:
                frame = await asyncio.wait_for(q.get(), timeout=min(0.1, remaining))
            except asyncio.TimeoutError:
                # 队列暂时空，检查是否断连
                if await request.is_disconnected():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    _logger.info(
                        "stream_cancelled | session=%s request_id=%s", session_id, request_id
                    )
                    return  # 断连：直接退出，不发 done
                continue

            if frame is None:
                break  # 哨兵：生产者正常结束

            yield frame

            # 每帧后检查断连（高频路径，is_disconnected 内部有缓存）
            if await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                _logger.info(
                    "stream_cancelled | session=%s request_id=%s", session_id, request_id
                )
                return

    except asyncio.CancelledError:
        # 外部取消（如 uvicorn 超时 / 进程退出）
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        _logger.info(
            "stream_cancelled | session=%s request_id=%s", session_id, request_id
        )
        raise

    finally:
        # 安全网：确保 task 最终被取消并 await（多次 cancel 幂等）
        # 必须 await 否则 task 取消是"预约"而非"已完成"，上游 LLM 调用实际未停止
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # 扣费：无论正常完成/超时/断连，均按实际产生的 token 记录（不按预估）
        # 断连时 state["total_tokens"] 是截至断点已处理的 usage 事件累计值
        await record_token_usage(bucket_key, state["total_tokens"])

        # Prometheus LLM 指标：与 done 帧、record_token_usage 同口径，三处数字对齐
        from api.metrics import llm_ttft_seconds, llm_tokens_total, llm_cost_usd_total
        _model = state["model_name"] or "unknown"
        if state["ttft_ms"] is not None:
            llm_ttft_seconds.observe(state["ttft_ms"] / 1000)
        if state["prompt_tokens"]:
            llm_tokens_total.labels(type="prompt", model=_model).inc(state["prompt_tokens"])
        if state["completion_tokens"]:
            llm_tokens_total.labels(type="completion", model=_model).inc(state["completion_tokens"])
        if state["total_cost"]:
            llm_cost_usd_total.labels(model=_model).inc(state["total_cost"])

    # ── done 帧（正常完成 or 超时均发；断连 return 不到这里）────────────────────
    total_ms = round((time.perf_counter() - start) * 1000)
    yield _sse("done", {
        "session_id": session_id,
        "ttft_ms": state["ttft_ms"],
        "total_ms": total_ms,
        "total_tokens": state["total_tokens"],
        "total_cost": round(state["total_cost"], 6),
    })


# ── 路由 ──────────────────────────────────────────────────────────────────────
@router.post(
    "/stream",
    summary="流式对话 v1（token 级 SSE）",
    description="""
Token 级 Server-Sent Events 推送。

**SSE 事件类型：**
| event | 说明 |
|---|---|
| `token` | `{delta}` 单个 token 增量文本 |
| `tool_call` | `{name, args}` Agent 决定调用工具 |
| `tool_result` | `{name, ok, meta}` 工具执行完毕 |
| `usage` | `{prompt_tokens, completion_tokens, cost}` 本次模型调用用量 |
| `done` | `{session_id, ttft_ms, total_ms, total_tokens, total_cost}` 流结束 |
| `error` | RFC 7807 problem+json（含 request_id）|
""",
)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    agent: PersistentAgent = Depends(get_agent),
    api_key: str = Depends(require_api_key),
):
    _, is_new, session_id = req.resolve_thread_id()
    # thread_id 加入 bucket_key 命名空间，不同 API Key 的会话天然隔离（IDOR 防护）
    # 与 sessions.py 的查找逻辑保持一致：user:{bucket_key}:{session_id}
    bucket_key = get_rate_limit_key(api_key, request)
    thread_id = f"user:{bucket_key}:{session_id}"
    _logger.info("[v1] stream | session=%s | new=%s | msg=%.50s", session_id, is_new, req.message)

    # auth → rate limit → budget（任一不过直接短路，不启动 agent）
    from api.settings import APISettings
    _s = APISettings()
    await check_rate_limit(bucket_key, _s.rate_limit_rpm)
    await check_token_budget(bucket_key, _s.daily_token_budget)

    return StreamingResponse(
        _stream_v1_generator(request, agent, req.message, thread_id, session_id, bucket_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-Id": session_id,
            "X-Accel-Buffering": "no",  # 关闭 Nginx/代理缓冲，确保逐 token 实时性
        },
    )


@router.post(
    "/invoke",
    response_model=ChatResponse,
    summary="非流式对话 v1",
    description="等待 Agent 完整执行后一次性返回，适合不支持 SSE 的调用方（n8n / Zapier）。",
)
async def chat_invoke(
    req: ChatRequest,
    request: Request,
    agent: PersistentAgent = Depends(get_agent),
    api_key: str = Depends(require_api_key),
):
    _, is_new, session_id = req.resolve_thread_id()
    # 与 chat_stream 保持一致：thread_id 含 bucket_key（IDOR 防护）
    bucket_key = get_rate_limit_key(api_key, request)
    thread_id = f"user:{bucket_key}:{session_id}"
    _logger.info("[v1] invoke | session=%s | new=%s | msg=%.50s", session_id, is_new, req.message)

    from api.settings import APISettings
    _s = APISettings()
    await check_rate_limit(bucket_key, _s.rate_limit_rpm)
    await check_token_budget(bucket_key, _s.daily_token_budget)

    try:
        result = await agent.invoke(
            messages=[HumanMessage(content=req.message)],
            thread_id=thread_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.exception("[v1] invoke 错误: %s", e)
        raise HTTPException(status_code=500, detail="Agent 内部错误")

    msgs = result.get("messages", [])
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage) and m.content]
    content = ai_msgs[-1].content if ai_msgs else ""

    # 记录实际 token 消耗（从最后一条 AI 消息的 usage_metadata 读取）
    if ai_msgs:
        um = getattr(ai_msgs[-1], "usage_metadata", None) or {}
        invoke_tokens = int(um.get("total_tokens", 0))
        await record_token_usage(bucket_key, invoke_tokens)

    return ChatResponse(content=content, session_id=session_id, thread_id=thread_id)
