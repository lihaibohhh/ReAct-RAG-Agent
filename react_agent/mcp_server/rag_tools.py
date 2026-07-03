"""
rag_tools.py — MCP Server 的金融研报 RAG 检索工具注册模块

职责：
1. 注册 query_financial_reports 工具
2. 保持 RAG 依赖懒加载，避免 MCP stdio 握手阶段超时
3. 返回 MCP 友好的结构化 dict，而不是 json.dumps 字符串
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

from react_agent.mcp_server.responses import clamp_int, mcp_err, mcp_ok

logger = logging.getLogger(__name__)

QUERY_FINANCIAL_REPORTS_DESCRIPTION = (
    "在金融研报私有知识库中检索信息。\n\n"
    "【适用场景】\n"
    "  - 具体公司财务数据（营收、毛利率、净利润、负债率等）\n"
    "  - 行业研究结论、券商观点、评级、目标价、盈利预测\n"
    "  - 政策文件、监管规定的具体条款\n"
    "  - 研报中的表格数据、图表数据、量化指标\n\n"
    "【query 写法】建议使用精炼金融关键词，格式：公司名 + 指标 + 报告期。\n"
    "  好：'比亚迪 毛利率 2023Q3'\n"
    "  差：'请帮我查一下比亚迪最近的情况'\n\n"
    "【top_k】返回结果数量，默认 3，范围 1~10。\n"
    "【include_meta】是否返回完整调试元信息，默认 true。\n\n"
    "【返回内容】每条结果含正文、来源文档、页码和 chunk_id，可追溯验证。"
)

# pdf_parser.py 当前稳定写入的辅助元数据字段。
# source/page 单独处理；content 只放正文；不再额外套 citation。
_RESULT_META_KEYS = ("chunk_id", "doc_type", "industry")


def register_rag_tools(server: FastMCP) -> None:
    """
    注册 RAG 检索工具。

    注意：
    - 不要在模块导入阶段加载 retriever / reranker / cache。
    - 这些重依赖必须放在工具函数内部懒加载，否则 MCP 客户端握手可能超时。
    """

    @server.tool(description=QUERY_FINANCIAL_REPORTS_DESCRIPTION)
    async def query_financial_reports(
            query: str,
            top_k: int = 3,
            include_meta: bool = True,
    ) -> dict[str, Any]:
        """
        在私有金融研报知识库中执行检索。

        执行流程：
        1. query strip
        2. 空 query 返回结构化 ValidationError
        3. top_k 限制在 1~10
        4. 懒加载 settings / retriever / reranker / semantic_cache
        5. 先查语义缓存
        6. 缓存 miss 后执行 BM25 + 向量双路召回
        7. Cross-Encoder Reranker 精排
        8. 写入语义缓存
        9. 返回结构化 dict
        """
        q = (query or "").strip()
        safe_top_k = clamp_int(top_k, default=3, min_value=1, max_value=10)
        _log_query_event("start", q, safe_top_k, include_meta)

        if not q:
            _log_query_event("end", q, safe_top_k, include_meta, stage="validation")
            return mcp_err(
                "检索词不能为空",
                error_type="ValidationError",
                meta=_meta(
                    include_meta=include_meta,
                    stage="validation",
                    top_k=safe_top_k,
                ),
            )

        try:
            # ── 懒加载：避免 MCP 握手阶段加载重模型 / 向量库 / Redis ─────────
            from react_agent.core.config import settings
            from react_agent.rag.retriever import _dual_retrieve
            from react_agent.rag.reranker import _rerank
            from react_agent.rag.semantic_cache import get_cached_result, set_cached_result

            max_chars = _get_max_content_chars(settings)

            # ── 1. 语义缓存查找 ───────────────────────────────
            cached_docs = await get_cached_result(q)

            if cached_docs is not None:
                docs = list(cached_docs)[:safe_top_k]
                results = _build_results(docs, max_chars=max_chars)

                logger.info(
                    "[RAG-MCP] 缓存命中，query=%r，top_k=%s，返回 %s 条",
                    q,
                    safe_top_k,
                    len(results),
                )
                _log_query_event("end", q, safe_top_k, include_meta, stage="semantic_cache_hit")

                return mcp_ok(
                    data={
                        "query": q,
                        "has_relevant_content": len(results) > 0,
                        "results": results,
                    },
                    meta=_meta(
                        include_meta=include_meta,
                        stage="semantic_cache_hit",
                        top_k=safe_top_k,
                        count=len(results),
                    ),
                )

            # ── 2. 双路召回 ───────────────────────────────────
            candidates = await _dual_retrieve(q)

            if not candidates:
                logger.info("[RAG-MCP] 召回为空，query=%r", q)
                _log_query_event("end", q, safe_top_k, include_meta, stage="dual_retrieve_empty")

                return mcp_ok(
                    data={
                        "query": q,
                        "has_relevant_content": False,
                        "results": [],
                    },
                    meta=_meta(
                        include_meta=include_meta,
                        stage="dual_retrieve_empty",
                        top_k=safe_top_k,
                        count=0,
                        candidates=0,
                    ),
                )

            # ── 3. Reranker 精排 ─────────────────────────────
            docs, top_score = await _rerank(q, candidates, top_n=safe_top_k)

            # ── 4. 写入语义缓存 ───────────────────────────────
            # set_cached_result 内部会根据 top_score 做置信度门控。
            await set_cached_result(q, docs, top_score=top_score)

            results = _build_results(docs, max_chars=max_chars)

            logger.info(
                "[RAG-MCP] 检索完成，query=%r，top_k=%s，返回 %s 条，candidates=%s，top_score=%.4f",
                q,
                safe_top_k,
                len(results),
                len(candidates),
                float(top_score or 0.0),
            )
            _log_query_event("end", q, safe_top_k, include_meta, stage="dual_retrieve+rerank")
            return mcp_ok(
                data={
                    "query": q,
                    "has_relevant_content": len(results) > 0,
                    "results": results,
                },
                meta=_meta(
                    include_meta=include_meta,
                    stage="dual_retrieve+rerank",
                    top_k=safe_top_k,
                    count=len(results),
                    candidates=len(candidates),
                    top_score=round(float(top_score or 0.0), 4),
                ),
            )

        except Exception as exc:
            logger.exception("[RAG-MCP] 检索异常，query=%r", q)
            _log_query_event("end", q, safe_top_k, include_meta, stage="exception")

            return mcp_err(
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                meta=_meta(
                    include_meta=include_meta,
                    stage="exception",
                    top_k=safe_top_k,
                ),
            )


def _log_query_event(
        event: str,
        query: str,
        top_k: int,
        include_meta: bool,
        stage: str | None = None,
) -> None:
    logger.info(
        "[RAG-MCP] query_financial_reports event=%s pid=%s query=%r top_k=%s include_meta=%s stage=%s",
        event,
        os.getpid(),
        query,
        top_k,
        include_meta,
        stage or "",
    )


def _meta(
        *,
        include_meta: bool,
        stage: str,
        top_k: int | None = None,
        count: int | None = None,
        candidates: int | None = None,
        top_score: float | None = None,
) -> dict[str, Any]:
    """
    统一生成 query_financial_reports 的 meta。

    include_meta=False 时，只保留给客户端/LLM 判断流程所需的最小字段。
    """
    base: dict[str, Any] = {
        "tool": "query_financial_reports",
        "stage": stage,
    }

    if not include_meta:
        return base

    if top_k is not None:
        base["top_k"] = top_k
    if count is not None:
        base["count"] = count
    if candidates is not None:
        base["candidates"] = candidates
    if top_score is not None:
        base["top_score"] = top_score

    return base


def _get_max_content_chars(settings: Any, default: int = 1200) -> int:
    """
    从 settings 中读取 RAG 单条内容最大字符数。

    加兜底的原因：
    - MCP Server 不应该因为配置字段临时缺失直接崩溃
    - 正常情况下仍优先使用 settings.tools.rag.max_content_chars
    """
    try:
        value = settings.tools.rag.max_content_chars
    except Exception:
        return default

    return clamp_int(value, default=default, min_value=200, max_value=5000)


def _build_results(docs: list[Document], max_chars: int) -> list[dict[str, Any]]:
    """
    将 RAG 检索得到的 LangChain Document 转成 MCP 可 JSON 序列化的结果列表。

    强契约：
    - docs 只接受 langchain_core.documents.Document；
    - content 只放正文，不拼接来源前缀；
    - source/page/chunk_id/doc_type/industry 以平铺结构化字段返回。
    """
    results: list[dict[str, Any]] = []

    for doc in docs:
        if not isinstance(doc, Document):
            logger.warning(
                "[RAG-MCP] 跳过非 Document 检索结果：type=%s",
                type(doc).__name__,
            )
            continue

        content = (doc.page_content or "").strip()
        if not content:
            continue

        metadata = dict(doc.metadata or {})

        item: dict[str, Any] = {
            "content": _trim_text(content, max_chars=max_chars),
            "source": _source_from_metadata(metadata),
            "page": metadata.get("page"),
        }

        for key in _RESULT_META_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                item[key] = value

        results.append(item)

    return results


def _source_from_metadata(metadata: dict[str, Any]) -> str:
    """
    返回面向 MCP 客户端的来源标识。

    当前 pdf_parser.py 中：
    - metadata["source"] 是原始 file_path，可能是本机绝对路径；
    - metadata["chunk_id"] 由相对文件名构造，例如 xxx.pdf::page_3::chunk_0。

    因此优先从 chunk_id 中取相对文件名；缺失时再回退到 source 的 basename。
    """
    chunk_id = metadata.get("chunk_id")
    if chunk_id not in (None, ""):
        source = str(chunk_id).split("::", 1)[0].strip()
        if source:
            return _normalize_path(source)

    raw_source = metadata.get("source")
    if raw_source in (None, ""):
        return ""

    try:
        return _normalize_path(Path(str(raw_source)).name)
    except Exception:
        return _normalize_path(str(raw_source))


def _normalize_path(value: str) -> str:
    """统一路径分隔符，便于客户端展示和 LLM 引用。"""
    return value.replace("\\", "/")


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."
