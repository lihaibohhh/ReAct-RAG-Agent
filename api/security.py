"""
api/security.py — API Key 鉴权 Depends + 限流 bucket key 解析。

支持两种传入方式（优先级 X-API-Key > Authorization: Bearer）：
    X-API-Key: <key>
    Authorization: Bearer <key>

若 API_KEY 未在环境变量中配置 → 免鉴权模式（本地开发）。
失败返回 401 problem+json（复用 errors.AppError，不另造格式）。

豁免：本文件只定义 Depends，挂载位置决定豁免范围：
    - 仅挂在 /api/v1/* 业务路由上
    - /health、/api/v1/health、/metrics 路由不引用此 Depends → 自然豁免
    - 旧版 /chat/* legacy 路由也不挂，维持现状

匿名降级策略（有意设计）：
    免鉴权模式下 api_key 返回空字符串。限流和预算不能对空 key 直接跳过，
    否则匿名流量完全无管控。调用方应通过 get_rate_limit_key() 将空 key
    映射为 client:{IP}，使匿名请求也受固定窗口限流和当日预算约束。
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

_logger = logging.getLogger(__name__)

# FastAPI 安全描述符（auto_error=False：未传时不自动 422，由我们手动处理）
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    x_api_key: Optional[str] = Security(_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """
    提取并校验 API Key，返回有效 key 字符串（供下游限流/预算使用）。

    - API_KEY 未配置 → 免鉴权模式，返回空字符串
    - 配置了 API_KEY，但客户端未传 → 401
    - 传了错误 key → 401（日志记录 key 前缀，不记完整 key）
    - 传了正确 key → 返回该 key
    """
    from api.errors import AppError
    from api.settings import APISettings

    configured_key = APISettings().api_key
    if not configured_key:
        # 未配置 → 免鉴权模式（开发/内网环境）
        return ""

    provided_key: Optional[str] = x_api_key or (bearer.credentials if bearer else None)

    if not provided_key:
        from api.metrics import auth_failures_total
        auth_failures_total.inc()
        raise AppError(
            status=401,
            title="Unauthorized",
            detail="缺少 API Key。请通过 X-API-Key 请求头或 Authorization: Bearer 传入。",
        )

    if provided_key != configured_key:
        from api.metrics import auth_failures_total
        auth_failures_total.inc()
        _logger.warning(
            "auth_failed | key_prefix=%s",
            (provided_key[:8] + "...") if len(provided_key) > 8 else "***",
        )
        raise AppError(status=401, title="Unauthorized", detail="API Key 无效。")

    return provided_key


def _get_client_ip(request: Request) -> str:
    """提取客户端 IP：优先 X-Forwarded-For 第一跳，否则直连 IP。"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_rate_limit_key(api_key: str, request: Request) -> str:
    """
    返回用于限流和预算追踪的 bucket key。

    - 有 API Key → 用 key 本身（已认证用户按 key 隔离）
    - 无 API Key（免鉴权模式）→ 回落到 client:{IP}（匿名降级策略）
      优先读 X-Forwarded-For 第一跳，没有则用直连 IP。
      这是有意为之：确保匿名流量也受固定窗口限流和当日预算约束，
      防止单个客户端在免鉴权模式下无限裸跑 LLM。
    """
    if api_key:
        return api_key
    return f"client:{_get_client_ip(request)}"
