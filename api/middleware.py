from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

# ── Request-ID ContextVar（供 JsonFormatter 和 errors.py 读取）────────────────
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

# 旧版路径（响应头注入 Deprecation: true）
_DEPRECATED_EXACT: frozenset[str] = frozenset({"/health"})
_DEPRECATED_PREFIXES: tuple[str, ...] = ("/chat/",)


# ── JSON 日志 Formatter ────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """
    把 LogRecord 格式化为单行 JSON。
    自动注入当前请求的 request_id（从 ContextVar 读取）。
    支持通过 logging.info(msg, extra={...}) 携带结构化字段。
    """

    _EXTRA_FIELDS = ("event", "method", "path", "status", "duration_ms")

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get("") or None,
        }
        for field in self._EXTRA_FIELDS:
            val = record.__dict__.get(field)
            if val is not None:
                doc[field] = val
        if record.exc_info:
            doc["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


def setup_json_logging() -> None:
    """
    把所有日志 handler 替换为 JsonFormatter。
    uvicorn 的 access / error logger 清空自有 handler 并 propagate 到 root，
    避免 uvicorn 文本格式与应用 JSON 格式混合输出。

    必须在任何 logger 首次使用之前调用（main.py 顶部）。
    """
    formatter = JsonFormatter()
    root = logging.getLogger()

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(logging.INFO)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


# ── 纯 ASGI 中间件（SSE 安全，不缓冲流式响应）──────────────────────────────────
class RequestIDMiddleware:
    """
    纯 ASGI 中间件，不使用 BaseHTTPMiddleware，对 SSE 安全。

    职责：
    1. 读取 X-Request-Id 请求头，若无则生成 UUID4。
    2. 写入 request_id_ctx ContextVar（供 JsonFormatter 和 errors.py 消费）。
    3. 拦截 http.response.start 消息，注入 X-Request-Id 响应头；
       旧版路径（/health、/chat/*）同步注入 Deprecation: true。
    4. 请求结束时打一条结构化 summary 日志（包含 method/path/status/duration_ms）。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._logger = logging.getLogger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # ── 确定 Request-ID ───────────────────────────────────────────────
        raw_headers: dict[bytes, bytes] = {k: v for k, v in scope.get("headers", [])}
        request_id = (
            raw_headers.get(b"x-request-id", b"").decode("latin-1").strip()
            or str(uuid.uuid4())
        )
        token = request_id_ctx.set(request_id)

        path: str = scope.get("path", "")
        method: str = scope.get("method", "")
        start = time.perf_counter()
        status_code: int = 0

        is_deprecated = (
            path in _DEPRECATED_EXACT
            or any(path.startswith(p) for p in _DEPRECATED_PREFIXES)
        )

        # ── 包装 send：抓 http.response.start 注入响应头 ─────────────────
        async def _send(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                mut = MutableHeaders(scope=message)
                mut.append("x-request-id", request_id)
                if is_deprecated:
                    mut.append("deprecation", "true")
            await send(message)

        from api.metrics import (
            http_requests_in_flight,
            http_requests_total,
            http_request_duration_seconds,
            get_path_template,
        )
        # /api/v1/metrics 本身不计入 http_requests_total，避免 Prometheus 抓取自计数
        _is_metrics_scrape = (path == "/api/v1/metrics")
        http_requests_in_flight.inc()

        try:
            await self.app(scope, receive, _send)
        finally:
            elapsed = time.perf_counter() - start
            duration_ms = round(elapsed * 1000)

            # Starlette 0.52.1 不设 scope["route"]，用 endpoint 反向查路由模板。
            # register_routes(app) 在 main.py 中注册完所有路由后调用，
            # 404 时 scope 无 endpoint，get_path_template 退回 "unmatched"。
            path_template: str = get_path_template(scope, fallback="unmatched")

            http_requests_in_flight.dec()
            if not _is_metrics_scrape:
                http_requests_total.labels(
                    method=method, path=path_template, status=str(status_code or 0)
                ).inc()
                http_request_duration_seconds.labels(
                    method=method, path=path_template
                ).observe(elapsed)

            self._logger.info(
                "request summary",
                extra={
                    "event": "request_summary",
                    "method": method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            request_id_ctx.reset(token)
