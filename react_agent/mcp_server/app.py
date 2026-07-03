"""
app.py:
    创建 FastMCP 实例，注册各组工具
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from react_agent.mcp_server.info_tools import register_info_tools
from react_agent.mcp_server.health_tools import register_health_tools
from react_agent.mcp_server.warmup_tools import register_warmup_tools
from react_agent.mcp_server.rag_tools import register_rag_tools

MCP_SERVER_NAME = "financial-rag"

MCP_SERVER_INSTRUCTIONS = (
    "这是一个金融研报私有知识库检索服务。"
    "当用户询问具体公司财务数据、研报结论、行业政策、券商评级、目标价、盈利预测等问题时，"
    "优先调用 query_financial_reports 工具进行检索。"
    "每条结果均标注来源文档与页码，可追溯验证。"
    "当需要检查知识库是否可用时，调用 check_knowledge_base。"
    "当需要提前加载 RAG 重资源以降低首次检索延迟时，调用 warmup_rag_pipeline。"
    "当需要了解本 MCP Server 能力边界时，调用 server_info。"
)


def create_mcp_server() -> FastMCP:
    """
    创建 MCP Server 实例，并集中注册 MCP 工具。

    mcp_rag_server.py 只负责薄启动、环境初始化和 stdio 运行；
    具体工具注册统一在 react_agent.mcp_server.app 中完成。
    """
    server = FastMCP(
        name=MCP_SERVER_NAME,
        instructions=MCP_SERVER_INSTRUCTIONS,
    )

    register_info_tools(server)
    register_health_tools(server)
    register_warmup_tools(server)
    register_rag_tools(server)
    return server
