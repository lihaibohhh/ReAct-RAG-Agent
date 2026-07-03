"""
warmup_tools.py

RAG warmup 工具。

设计目标：
1. MCP Server 启动时不阻塞 stdio 握手；
2. 通过后台任务触发完整 RAG 冷路径；
3. Inspector / Web 端立即返回，不同步等待 dual_retrieve；
4. 通过 get_full_warmup_status 查询预热状态。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

from react_agent.mcp_server.responses import mcp_ok


logger = logging.getLogger(__name__)

DEFAULT_WARMUP_QUERY = os.getenv(
    "RAG_FULL_WARMUP_QUERY",
    "2023年 锂矿 全球储量",
)

_FULL_WARMUP_TASK: asyncio.Task | None = None
_FULL_WARMUP_STATUS: dict[str, Any] = {
    "state": "not_started",
    "started_at": None,
    "finished_at": None,
    "test_query": None,
    "top_n": None,
    "timings": {},
    "error": None,
}


def _now() -> float:
    return time.time()


def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 4)


def _safe_len(x: Any) -> int:
    try:
        return len(x or [])
    except Exception:
        return -1


def _task_state() -> str:
    if _FULL_WARMUP_TASK is None:
        return "none"
    if _FULL_WARMUP_TASK.done():
        return "done"
    return "running"


def _set_status(**kwargs: Any) -> None:
    _FULL_WARMUP_STATUS.update(kwargs)


async def _light_warmup() -> dict[str, float]:
    """
    轻量预热：
    - 加载 retriever；
    - 加载 reranker；
    - 执行一次 dummy rerank，触发 tokenizer / torch / CUDA 首次推理。
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    from react_agent.rag.retriever import _get_retriever

    await asyncio.to_thread(_get_retriever)
    timings["load_retriever"] = _elapsed(t0)

    t0 = time.perf_counter()
    from react_agent.rag.reranker import _get_reranker

    await asyncio.to_thread(_get_reranker)
    timings["load_reranker"] = _elapsed(t0)

    t0 = time.perf_counter()
    from react_agent.rag.reranker import _rerank

    dummy_doc = Document(
        page_content="这是一个用于 MCP RAG 预热的虚拟文本，用于触发 reranker tokenizer 和首次推理。",
        metadata={
            "source": "__warmup__",
            "page": 0,
            "chunk_id": "__warmup__::page_0::chunk_0",
            "doc_type": "warmup",
            "industry": "warmup",
        },
    )

    await _rerank(
        "__mcp_warmup_query__",
        [dummy_doc],
        top_n=1,
    )
    timings["dummy_rerank"] = _elapsed(t0)

    return timings


async def _run_full_warmup_job(test_query: str, top_n: int) -> None:
    """
    后台完整预热任务。

    注意：
    - 不写 semantic_cache，避免污染业务缓存；
    - 只负责把 MCP 进程里的 retriever / reranker / dual_retrieve 冷路径跑热；
    - 异常不抛到 Inspector，由状态工具暴露。
    """
    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    _set_status(
        state="running",
        started_at=_now(),
        finished_at=None,
        test_query=test_query,
        top_n=top_n,
        timings=timings,
        error=None,
    )

    logger.info(
        "[RAG-MCP] background_full_warmup event=start pid=%s test_query=%r top_n=%s",
        os.getpid(),
        test_query,
        top_n,
    )

    try:
        # 1) 轻量资源预热
        t0 = time.perf_counter()
        light_timings = await _light_warmup()
        timings.update(light_timings)
        timings["light_warmup_total"] = _elapsed(t0)

        # 2) 真实 dual retrieve：触发 BM25 / vector / Chroma / embedding 冷路径
        t0 = time.perf_counter()
        from react_agent.rag.retriever import _dual_retrieve

        candidates = await _dual_retrieve(test_query)
        timings["dual_retrieve"] = _elapsed(t0)

        # 3) 真实 rerank：比 dummy rerank 更接近正式 query
        docs = []
        if candidates:
            t0 = time.perf_counter()
            from react_agent.rag.reranker import _rerank

            rerank_result = await _rerank(
                test_query,
                candidates,
                top_n=top_n,
            )

            if isinstance(rerank_result, tuple):
                docs = rerank_result[0] or []
            else:
                docs = rerank_result or []

            timings["rerank"] = _elapsed(t0)
        else:
            timings["rerank"] = 0.0

        timings["total"] = _elapsed(total_t0)

        _set_status(
            state="done",
            finished_at=_now(),
            candidates_count=_safe_len(candidates),
            docs_count=_safe_len(docs),
            timings=timings,
            error=None,
        )

        logger.info(
            "[RAG-MCP] background_full_warmup event=done pid=%s "
            "test_query=%r candidates=%s docs=%s timings=%s",
            os.getpid(),
            test_query,
            _safe_len(candidates),
            _safe_len(docs),
            timings,
        )

    except asyncio.CancelledError:
        timings["total"] = _elapsed(total_t0)

        _set_status(
            state="cancelled",
            finished_at=_now(),
            timings=timings,
            error="CancelledError",
        )

        logger.warning(
            "[RAG-MCP] background_full_warmup event=cancelled pid=%s "
            "test_query=%r timings=%s",
            os.getpid(),
            test_query,
            timings,
        )
        raise

    except Exception as exc:
        timings["total"] = _elapsed(total_t0)

        _set_status(
            state="error",
            finished_at=_now(),
            timings=timings,
            error=f"{type(exc).__name__}: {exc}",
        )

        logger.exception(
            "[RAG-MCP] background_full_warmup event=error pid=%s "
            "test_query=%r timings=%s",
            os.getpid(),
            test_query,
            timings,
        )


def register_warmup_tools(server: FastMCP) -> None:
    @server.tool(
        description=(
            "后台触发完整 RAG warmup，立即返回，不同步等待 dual_retrieve 完成。"
            "会依次预热 retriever、reranker、dummy rerank、dual retrieve 和真实 rerank。"
        )
    )
    async def start_full_warmup_rag_pipeline(
        test_query: str = DEFAULT_WARMUP_QUERY,
        top_n: int = 3,
        force: bool = False,
    ) -> dict[str, Any]:
        global _FULL_WARMUP_TASK

        current_state = _task_state()
        status_state = _FULL_WARMUP_STATUS.get("state")

        if current_state == "running":
            return mcp_ok(
                data={
                    "status": "already_running",
                    "message": "Full warmup is already running.",
                    "warmup_status": _FULL_WARMUP_STATUS,
                },
                meta={
                    "tool": "start_full_warmup_rag_pipeline",
                    "stage": "already_running",
                },
            )

        if status_state == "done" and not force:
            return mcp_ok(
                data={
                    "status": "already_done",
                    "message": "Full warmup has already completed. Set force=true to run again.",
                    "warmup_status": _FULL_WARMUP_STATUS,
                },
                meta={
                    "tool": "start_full_warmup_rag_pipeline",
                    "stage": "already_done",
                },
            )

        safe_top_n = max(1, min(int(top_n), 10))

        _FULL_WARMUP_TASK = asyncio.create_task(
            _run_full_warmup_job(test_query, safe_top_n),
            name="rag_full_warmup",
        )

        return mcp_ok(
            data={
                "status": "background_full_warmup_started",
                "message": "Full warmup has started in background. Use get_full_warmup_status to check progress.",
                "test_query": test_query,
                "top_n": safe_top_n,
            },
            meta={
                "tool": "start_full_warmup_rag_pipeline",
                "stage": "started",
            },
        )

    @server.tool(
        description="查看后台完整 RAG warmup 的当前状态。"
    )
    async def get_full_warmup_status() -> dict[str, Any]:
        task_state = _task_state()

        return mcp_ok(
            data={
                "task_state": task_state,
                "warmup_status": _FULL_WARMUP_STATUS,
            },
            meta={
                "tool": "get_full_warmup_status",
                "stage": task_state,
            },
        )