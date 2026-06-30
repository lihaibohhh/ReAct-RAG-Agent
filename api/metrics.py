"""
api/metrics.py — Prometheus 指标注册（模块级单例，import 即初始化）。

端点: GET /api/v1/metrics（main.py 注册，无需 API Key）。

label 设计原则：
- path 用路由模板（/api/v1/chat/stream），不用原始 URL，防止 session_id 进标签造成基数爆炸
- model 用 response_metadata.model_name（真实 API 型号），与计费口径一致，不用 ctx 别名

路由模板解析：
Starlette 0.52.1 的 Route.handle() 不设 scope["route"]，只在 Route.matches() 里
把 endpoint 函数写入 scope。因此用 endpoint → path_template 反向映射：
  main.py 注册完所有路由后调用 register_routes(app)，
  middleware 的 finally 里用 get_path_template(scope, fallback) 取模板。
"""
from __future__ import annotations

from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram

# ── HTTP 通用 ─────────────────────────────────────────────────────────────────

http_requests_total = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求延迟（SSE 请求为全流时长）",
    ["method", "path"],
    # 针对实测总时长 ~34s 定制：在 30-60s 段有足够分辨率，不会让大多数请求全进 +Inf
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 120.0],
)

http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "当前 in-flight HTTP 请求数（断连/超时在 finally 减回，不会泄漏）",
)

# ── LLM 专项 ─────────────────────────────────────────────────────────────────
# 数据源与 Phase 1 done 帧、Phase 2 record_token_usage 完全同口径，三处数字对齐

llm_ttft_seconds = Histogram(
    "llm_ttft_seconds",
    "LLM 首 token 延迟（仅流式请求）",
    # 针对实测 TTFT ~6.5s 定制：在 5-10s 段细分，p95/p99 有意义
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM token 消耗累计",
    ["type", "model"],  # type: prompt | completion
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "LLM 推理费用累计（美元）",
    ["model"],
)

# ── 安全 / 限流 ───────────────────────────────────────────────────────────────

rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total",
    "限流拒绝请求数",
    ["reason"],  # reason: rpm | budget
)

auth_failures_total = Counter(
    "auth_failures_total",
    "API Key 认证失败次数",
)

# ── 路由模板反向映射 ──────────────────────────────────────────────────────────
# Starlette 0.52.1 不设 scope["route"]，需要在 app 注册完所有路由后
# 手动构建 endpoint → path_template 映射，供 ASGI 中间件使用。

_endpoint_to_path: dict[Callable[..., Any], str] = {}


def register_routes(app: Any) -> None:
    """
    遍历 FastAPI app 的全部路由，构建 endpoint → path_template 映射。
    在 main.py 中所有 include_router() 调用之后调用一次。
    """
    for route in getattr(app, "routes", []):
        ep = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        if ep is not None and path is not None:
            _endpoint_to_path[ep] = path


def get_path_template(scope: dict[str, Any], fallback: str) -> str:
    """
    从 scope["endpoint"] 反查路由模板字符串。
    - 命中 → 返回模板（如 /api/v1/chat/stream）
    - 未注册（404 / OPTIONS / 健康检查）→ 返回 fallback（通常是原始 path 或 "unmatched"）
    """
    endpoint = scope.get("endpoint")
    if endpoint is not None:
        tmpl = _endpoint_to_path.get(endpoint)
        if tmpl:
            return tmpl
    return fallback
