from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的消息")
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。不传则自动生成新会话（UUID）。"
                    "格式建议：{username}-{uuid} 或直接传 UUID。"
    )

    def resolve_thread_id(self) -> tuple[str, bool]:
        """
        返回 (thread_id, is_new_session)
        thread_id 格式与 Streamlit 侧保持一致：user:{session_id}
        """
        is_new = self.session_id is None
        sid = self.session_id or str(uuid.uuid4())
        return f"user:{sid}", is_new, sid


class ChatResponse(BaseModel):
    """非流式响应（/chat/invoke 用）"""
    content: str
    session_id: str
    thread_id: str
    usage: Optional[dict] = None


class SessionInfo(BaseModel):
    session_id: str
    thread_id: str
    message: str = "新会话已创建"


class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool
    checkpoint_backend: Optional[str] = None


class MessageItem(BaseModel):
    """单条对话消息（脱敏：仅 type + content，不含 tool_call details 等内部字段）。"""
    type: str  # "human" / "ai" / "tool" / "system" / "unknown"
    content: str
    id: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    """GET /sessions/{id}/history 响应体。"""
    session_id: str
    messages: List[MessageItem]
    total: int
    page: int
    page_size: int
    has_more: bool