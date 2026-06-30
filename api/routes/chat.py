from __future__ import annotations
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from api.dependencies import get_agent
from api.models import ChatRequest, ChatResponse
from react_agent.core.agent import PersistentAgent

_logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ── SSE 工具函数 ──────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """把一条事件序列化为 SSE 格式字符串"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_from_chunk(chunk: dict) -> list[dict]:
    """
    把 LangGraph astream() 吐出的节点级 chunk 转换为 SSE 事件列表。

    LangGraph astream() 的 chunk 结构（节点级，非 token 级）：
        {"call_model": {"messages": [AIMessage(...)]}}
        {"tools":      {"messages": [ToolMessage(...), ToolMessage(...)]}}
        {"reflection": {"messages": [AIMessage(...)]}}

    ⚠️ 注意：这里拿到的 AI 消息是节点执行完成后的完整内容，
    不是逐 token 流。如需 token 级推送，需在 PersistentAgent
    里暴露 astream_events() 接口（见文件末尾说明）。
    """
    events = []
    for node_name, node_output in chunk.items():
        if node_output is None:
            continue
        messages = node_output.get("messages", [])
        for msg in messages:

            # ── 模型输出 ──────────────────────────────────────────────────
            if isinstance(msg, AIMessage):
                content = msg.content
                if not content:
                    continue

                # 带工具调用意图（还未执行工具）→ tool_call 事件
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        events.append({
                            "event": "tool_call",
                            "data": {
                                "tool": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                                "node": node_name,
                            }
                        })
                else:
                    # 纯文本输出（最终回答或反思节点）
                    events.append({
                        "event": "token",
                        "data": {
                            "content": content if isinstance(content, str)
                                       else json.dumps(content, ensure_ascii=False),
                            "node": node_name,
                            # reflection 节点的输出一般是中间推理，可按需过滤
                            "is_final": node_name == "call_model",
                        }
                    })

            # ── 工具执行结果 ───────────────────────────────────────────────
            elif isinstance(msg, ToolMessage):
                raw = msg.content
                # 工具返回内容可能很长，只推摘要；调用方若需要完整内容可用 /invoke
                summary = raw[:300] + "…" if isinstance(raw, str) and len(raw) > 300 else raw
                events.append({
                    "event": "tool_result",
                    "data": {
                        "tool": msg.name,
                        "summary": summary,
                        "node": node_name,
                    }
                })

    return events


async def _stream_generator(
    agent: PersistentAgent,
    message: str,
    thread_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """把 agent.stream() 转成 SSE 字符串流"""
    try:
        human_msg = HumanMessage(content=message)
        final_content = ""

        async for chunk in agent.stream(messages=[human_msg], thread_id=thread_id):
            for ev in _extract_from_chunk(chunk):
                if ev["event"] == "token" and ev["data"].get("is_final"):
                    final_content = ev["data"]["content"]
                yield _sse(ev["event"], ev["data"])

        # 流结束，推送 done 事件，携带 session 信息供调用方存储
        yield _sse("done", {
            "session_id": session_id,
            "thread_id": thread_id,
            "final_content": final_content,
        })

    except ValueError as e:
        # thread_id 为空等业务逻辑错误
        _logger.warning(f"[API] stream 业务错误: {e}")
        yield _sse("error", {"message": str(e), "type": "validation_error"})

    except Exception as e:
        _logger.exception(f"[API] stream 未知错误: {e}")
        yield _sse("error", {"message": "Agent 内部错误，请稍后重试", "type": "internal_error"})


# ── 路由 ──────────────────────────────────────────────────────────────────────

@router.post(
    "/stream",
    summary="流式对话（SSE）",
    description="""
以 Server-Sent Events 格式返回 Agent 的实时输出。

**SSE 事件类型：**
| event | 说明 |
|---|---|
| `token` | 模型输出的文本（节点级，非 token 级） |
| `tool_call` | Agent 决定调用某工具（含工具名和参数） |
| `tool_result` | 工具执行完毕（含结果摘要） |
| `done` | 流结束，含 session_id 和完整最终回答 |
| `error` | 出现错误 |

**session_id 说明：**
- 不传则自动生成新会话（返回新 session_id，请自行保存）
- 传入已有 session_id 则继续上次对话
""",
)
async def chat_stream(
    req: ChatRequest,
    agent: PersistentAgent = Depends(get_agent),
):
    thread_id, is_new, session_id = req.resolve_thread_id()
    _logger.info(f"[API] stream | session={session_id} | new={is_new} | msg={req.message[:50]}")

    return StreamingResponse(
        _stream_generator(agent, req.message, thread_id, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-Id": session_id,       # 方便调用方从 header 直接拿
            "X-Accel-Buffering": "no",         # 关闭 Nginx 缓冲，保证 SSE 实时性
        },
    )


@router.post(
    "/invoke",
    response_model=ChatResponse,
    summary="非流式对话（等待完整回答）",
    description="适合不支持 SSE 的调用方（如 n8n、Zapier）。等待 Agent 完整执行后一次性返回。",
)
async def chat_invoke(
    req: ChatRequest,
    agent: PersistentAgent = Depends(get_agent),
):
    thread_id, is_new, session_id = req.resolve_thread_id()
    _logger.info(f"[API] invoke | session={session_id} | new={is_new} | msg={req.message[:50]}")

    try:
        human_msg = HumanMessage(content=req.message)
        result = await agent.invoke(messages=[human_msg], thread_id=thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _logger.exception(f"[API] invoke 错误: {e}")
        raise HTTPException(status_code=500, detail="Agent 内部错误")

    # 从 LangGraph 返回的 state 里提取最后一条 AI 消息
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
    content = ai_messages[-1].content if ai_messages else ""

    return ChatResponse(
        content=content,
        session_id=session_id,
        thread_id=thread_id,
    )


# ── token 级流式说明 ───────────────────────────────────────────────────────────
# 当前 stream() 用的是 LangGraph astream()，推送粒度是节点级（每个节点跑完才 yield）。
# 如果你想要真正的逐 token 推送，需要在 PersistentAgent 里新增：
#
#   async def stream_events(self, messages, thread_id=None):
#       ...
#       async for event in self._graph.astream_events(
#           {"messages": messages}, config=config, version="v2"
#       ):
#           if event["event"] == "on_chat_model_stream":
#               chunk = event["data"]["chunk"]
#               if chunk.content:
#                   yield chunk.content
#
# 然后在这里替换 _stream_generator 里的 agent.stream() 调用即可。