from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs（application/problem+json）"""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    request_id: Optional[str] = None


class AppError(Exception):
    """
    在路由 handler 内抛出此异常，自动映射为 problem+json 响应。

    示例：
        raise AppError(status=404, title="Not Found", detail="会话不存在")
        raise AppError(status=429, title="Too Many Requests", detail="...", headers={"Retry-After": "5"})
    """

    def __init__(
        self,
        *,
        status: int,
        title: str,
        detail: str,
        headers: Optional[dict] = None,
    ) -> None:
        self.status = status
        self.title = title
        self.detail = detail
        self.headers: dict = headers or {}
        super().__init__(detail)


def _problem_response(
    *,
    status: int,
    title: str,
    detail: str,
    extra_headers: Optional[dict] = None,
) -> JSONResponse:
    """构建统一 RFC 7807 响应，自动从 ContextVar 注入 request_id。"""
    from api.middleware import request_id_ctx  # 延迟 import，避免循环依赖

    request_id = request_id_ctx.get("") or None
    body = ProblemDetail(
        title=title,
        status=status,
        detail=detail,
        request_id=request_id,
    )
    # X-Request-Id 由 RequestIDMiddleware 统一注入，不在此处重复写入。
    # extra_headers 用于 Retry-After 等特定响应头。
    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
        media_type="application/problem+json",
        headers=extra_headers or None,
    )


def register_exception_handlers(app) -> None:
    """注册全局异常 handler，替换 main.py 原有的兜底 handler。"""

    from fastapi import HTTPException

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _problem_response(
            status=exc.status,
            title=exc.title,
            detail=exc.detail,
            extra_headers=exc.headers or None,
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        return _problem_response(
            status=exc.status_code,
            title="HTTP Error",
            detail=str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_exc(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _problem_response(
            status=422,
            title="Validation Error",
            detail=str(exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _handle_generic_exc(request: Request, exc: Exception) -> JSONResponse:
        _logger.exception("Unhandled exception: %s", exc)
        return _problem_response(
            status=500,
            title="Internal Server Error",
            detail="服务器内部错误，请稍后重试",
        )
