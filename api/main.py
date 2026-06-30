from __future__ import annotations
import logging
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# ── JSON 日志必须最先装配，确保后续所有 logger 都输出 JSON ──────────────────────
from api.middleware import setup_json_logging
setup_json_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from api.settings import APISettings
from api.errors import register_exception_handlers
from api.middleware import RequestIDMiddleware
from api.dependencies import startup_init, get_agent
from api.metrics import register_routes
from api.models import HealthResponse
from api.routes.chat import router as chat_router
from api.routes.v1.chat import router as v1_chat_router
from api.routes.v1.sessions import router as v1_sessions_router
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

_logger = logging.getLogger(__name__)


# ── Fail-fast：app 构造前校验配置（仍在 lifespan 前退出）─────────────────────
# 放在 main.py 而非 settings.py，保持 settings.py 零副作用，
# 使 pytest 可以安全 import api.settings 而不会因缺 key 退出进程。
try:
    _api_settings = APISettings()
    _api_settings.validate_llm_key()
except (ValidationError, ValueError) as _e:
    print(f"\n[startup] ❌ 配置校验失败，进程退出:\n{_e}\n", file=sys.stderr)
    sys.exit(1)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("[API] 服务启动，开始预热 Agent...")
    await startup_init()
    _logger.info("[API] 预热完成，开始接受请求")
    yield
    _logger.info("[API] 服务关闭")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ReAct Agent API",
    description="金融研报问答 Agent — FastAPI serving 层，支持 SSE 流式输出",
    version="1.0.0",
    lifespan=lifespan,
)

# 中间件注册顺序：后 add 的是最外层（最先处理请求）
# RequestIDMiddleware(最外) → CORSMiddleware → RoutingMiddleware → handler
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)  # 最外层，SSE 安全

# RFC 7807 统一错误信封（替换原有全局兜底）
register_exception_handlers(app)

# ── 路由 ───────────────────────────────────────────────────────────────────────
# v1 规范路径：/api/v1/chat/stream、/api/v1/chat/invoke
app.include_router(v1_chat_router, prefix="/api/v1")
# v1 会话 CRUD：/api/v1/sessions/{id}/history、DELETE /api/v1/sessions/{id}
app.include_router(v1_sessions_router, prefix="/api/v1")

# 旧路径保留（向后兼容）：/chat/stream、/chat/invoke
# Deprecation: true 响应头由 RequestIDMiddleware 自动注入
app.include_router(chat_router)


# ── 健康检查 ───────────────────────────────────────────────────────────────────
async def _health_data() -> HealthResponse:
    agent = await get_agent()
    return HealthResponse(
        status="ok",
        agent_initialized=agent._initialized,
        checkpoint_backend=agent.ctx.checkpoint_backend,
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health_v1():
    """规范健康检查端点"""
    return await _health_data()


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_legacy():
    """旧版健康检查（保留兼容，响应头带 Deprecation: true）"""
    return await _health_data()


@app.get("/api/v1/metrics", tags=["system"], include_in_schema=False)
async def metrics_endpoint() -> Response:
    """Prometheus scrape 端点，无需 API Key（与 /health 同样豁免）。"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# 所有路由注册完成后构建 endpoint → path_template 反向映射
# （Starlette 0.52.1 不设 scope["route"]，中间件依赖此映射获取路由模板 label）
register_routes(app)


# ── 本地启动（开发用） ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False,  # 由 RequestIDMiddleware summary 日志替代
    )
