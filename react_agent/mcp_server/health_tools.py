"""
health_tools.py:
    注册 check_knowledge_base
"""
from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from react_agent.mcp_server.responses import mcp_err, mcp_ok
from react_agent.core.config import settings


def register_health_tools(server: FastMCP) -> None:
    """
    注册知识库健康检查工具。

    这个工具只负责检查当前 RAG 知识库是否能被访问，
    不执行实际检索。
    """

    @server.tool(description="检查 RAG 知识库是否可用，返回向量库中的文档块数量")
    async def check_knowledge_base() -> dict[str, Any]:
        try:
            from langchain_chroma import Chroma

            from react_agent.core.config import settings

            chroma_dir = os.getenv("CHROMA_DB_PATH") or str(settings.tools.vector_store.CHROMA_DB_PATH)

            vector_store = Chroma(persist_directory=chroma_dir)

            chunk_count = vector_store._collection.count()

            return mcp_ok(
                data={
                    "status": "knowledge_base_ready",
                    "chunk_count": chunk_count,
                    "chroma_dir": chroma_dir,
                    "embedding_loaded": False,
                    "note": "lightweight health check; embedding/reranker are not loaded here",
                },
                meta={
                    "tool": "check_knowledge_base",
                    "stage": "health_check",
                    "mode": "lightweight",
                },
            )

        except Exception as exc:
            return mcp_err(
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                meta={
                    "tool": "check_knowledge_base",
                    "stage": "health_check",
                    "mode": "lightweight",
                },
            )