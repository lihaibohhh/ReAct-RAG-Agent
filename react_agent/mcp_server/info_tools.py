"""
info.py:
    注册 server_info
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from react_agent.mcp_server.responses import mcp_ok


def register_info_tools(server: FastMCP) -> None:
    """
    注册 MCP Server 信息类工具。

    这类工具不依赖 RAG / Chroma / Redis，适合作为模块化后的第一个工具。
    """

    @server.tool(description="返回 financial-rag MCP Server 的基本信息、可用工具和适用边界")
    async def server_info() -> dict[str, Any]:
        return mcp_ok(
            data={
                "name": "financial-rag",
                "version": "0.1.0",
                "transport": "stdio",
                "purpose": "把金融研报私有知识库检索能力暴露给 Claude Code / Claude Desktop / Cursor 等 MCP 客户端。",
                "available_tools": [
                    {
                        "name": "server_info",
                        "purpose": "查看 MCP Server 基本信息、工具列表和适用边界。",
                    },
                    {
                        "name": "query_financial_reports",
                        "purpose": "检索金融研报私有知识库，返回带来源文档和页码的证据片段。",
                        "best_query_format": "公司名 + 指标 + 报告期",
                        "example": "比亚迪 毛利率 2023Q3",
                    },
                    {
                        "name": "check_knowledge_base",
                        "purpose": "检查 RAG 知识库是否可用，返回向量库 chunk 数量。",
                    },
                    {
                        "name": "warmup_rag_pipeline",
                        "purpose": "显式预热 embedding、reranker，降低首次检索延迟。",
                    },
                ],
                "best_for": [
                    "公司财务指标查询",
                    "研报结论检索",
                    "行业研究观点查找",
                    "券商评级、目标价、盈利预测查找",
                    "政策文件和监管条款检索",
                ],
                "not_for": [
                    "实时股价",
                    "实时新闻",
                    "非金融研报相关闲聊",
                    "没有入库到私有知识库的外部资料",
                ],
            },
            meta={
                "tool": "server_info",
                "stage": "static_info",
            }
        )
