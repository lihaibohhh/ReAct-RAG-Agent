"""
mcp_rag_server.py — 把金融研报 RAG 知识库包装成 MCP Server

运行方式（stdio 模式，供 Claude Desktop / Cursor 等客户端接入）：
    python mcp_rag_server.py

测试方式（用 MCP Inspector）：
    npx @modelcontextprotocol/inspector python mcp_rag_server.py

依赖安装：
    pip install mcp
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 路径锚定：无论从哪个工作目录启动，都能正确找到项目根 ────────
# 脚本位于 <project_root>/src/mcp_rag_server.py
# 因此脚本所在目录的父目录就是项目根目录
_HERE = Path(__file__).resolve().parent          # .../src

# 将项目根插入 sys.path 首位，确保 react_agent 包可被正确导入
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Windows UTF-8 流修复 ──────────────────────────────────────
# 必须在 logging.basicConfig 之前执行：basicConfig 会捕获当前 sys.stderr
# 的引用，若之后才替换则 logging handler 仍用旧的 GBK stream。
# rag 模块内存在含 emoji（✅/❌）的 print()，GBK 编码无法处理会抛
# UnicodeEncodeError 并导致预热异常中断，此处统一切换为 UTF-8。
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
import logging
from dotenv import load_dotenv

# ── 必须在导入项目模块之前加载 .env ──────────────────────────
# 使用绝对路径加载，避免工作目录不确定导致找不到 .env
load_dotenv(_HERE / ".env")

from mcp.server.fastmcp import FastMCP
from react_agent.core.config import settings
from react_agent.utils.tool_helpers import _trim_text
from react_agent.utils.embedder import get_embedder


logger = logging.getLogger(__name__)

# ── 创建 MCP Server 实例 ─────────────────────────────────────
#    name      : 展示给客户端（Claude Desktop 侧边栏显示的名字）
#    instructions : 告诉 AI 这个 server 整体是干什么的
mcp = FastMCP(
    name="financial-rag",
    instructions=(
        "这是一个金融研报私有知识库检索服务。"
        "当用户询问具体公司财务数据、研报结论、行业政策、券商评级等问题时，"
        "优先调用 query_financial_reports 工具进行检索，"
        "每条结果均标注来源文档与页码，可追溯验证。"
    ),
)


# ════════════════════════════════════════════════════════════
#  工具 1：核心 RAG 检索
#  对应原 rag.py 中的 query_internal_knowledge
# ════════════════════════════════════════════════════════════
@mcp.tool(
    description=(
        "在金融研报私有知识库中检索信息。\n\n"
        "【适用场景】\n"
        "  - 具体公司财务数据（营收、毛利率、净利润、负债率等）\n"
        "  - 行业研究结论、券商观点、评级、目标价、盈利预测\n"
        "  - 政策文件、监管规定的具体条款\n"
        "  - 研报中的表格数据、图表数据、量化指标\n\n"
        "【query 写法】精炼的金融关键词，格式：公司名 + 指标 + 报告期\n"
        "  好：'比亚迪 毛利率 2023Q3'\n"
        "  差：'请帮我查一下比亚迪最近的情况'\n\n"
        "【返回内容】每条结果含来源文件路径和页码，可直接引用"
    )
)
async def query_financial_reports(query: str) -> str:
    """
    在私有金融研报知识库中执行检索。

    执行流程：
      1. 语义缓存查找（命中则直接返回）
      2. BM25 + 向量双路召回（各 Top-10，RRF 融合）
      3. Cross-Encoder 精排（取 Top-3）
      4. 写入语义缓存

    Args:
        query: 检索词，建议包含"公司名 + 指标 + 报告期"

    Returns:
        JSON 字符串，包含检索结果列表及每条来源路径与页码
    """
    q = (query or "").strip()

    # 懒加载，防止与客户端”握手“时，程序因延时回应而被迫”暴毙“
    from react_agent.rag.retriever import _dual_retrieve
    from react_agent.rag.reranker import _rerank
    from react_agent.rag.semantic_cache import get_cached_result, set_cached_result

    # ── 输入校验 ─────────────────────────────────────────────
    if not q:
        return json.dumps(
            {"ok": False, "error": "检索词不能为空"},
            ensure_ascii=False, indent=2
        )

    try:
        max_chars = settings.tools.rag.max_content_chars

        # ── 语义缓存查找 ──────────────────────────────────────
        cached_docs = await get_cached_result(q)
        if cached_docs is not None:
            results = _build_results(cached_docs, max_chars)
            logger.info(f"[RAG-MCP] 缓存命中，query={q!r}，返回 {len(results)} 条")
            return json.dumps(
                {
                    "ok": True,
                    "query": q,
                    "has_relevant_content": len(results) > 0,
                    "results": results,
                    "meta": {"stage": "semantic_cache_hit", "count": len(results)},
                },
                ensure_ascii=False, indent=2
            )

        # ── 双路召回 ──────────────────────────────────────────
        candidates = await _dual_retrieve(q)
        if not candidates:
            logger.info(f"[RAG-MCP] 召回为空，query={q!r}")
            return json.dumps(
                {
                    "ok": True,
                    "query": q,
                    "has_relevant_content": False,
                    "results": [],
                    "meta": {"stage": "dual_retrieve_empty", "count": 0},
                },
                ensure_ascii=False, indent=2
            )

        # ── Reranker 精排 ─────────────────────────────────────
        docs, top_score = await _rerank(q, candidates, top_n=3)

        # ── 写入语义缓存 ──────────────────────────────────────
        await set_cached_result(q, docs, top_score=top_score)

        results = _build_results(docs, max_chars)
        logger.info(f"[RAG-MCP] 检索完成，query={q!r}，返回 {len(results)} 条，top_score={top_score:.3f}")

        return json.dumps(
            {
                "ok": True,
                "query": q,
                "has_relevant_content": len(results) > 0,
                "results": results,
                "meta": {
                    "stage": "dual_retrieve+rerank",
                    "count": len(results),
                    "candidates": len(candidates),
                    "top_score": round(float(top_score), 4),
                },
            },
            ensure_ascii=False, indent=2
        )

    except Exception as e:
        logger.exception(f"[RAG-MCP] 检索异常，query={q!r}")
        return json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False, indent=2
        )


# ════════════════════════════════════════════════════════════
#  工具 2：知识库健康检查（可选，便于调试）
# ════════════════════════════════════════════════════════════
@mcp.tool(description="检查 RAG 知识库是否可用，返回向量库中的文档数量")
async def check_knowledge_base() -> str:
    """健康检查：连接向量数据库，返回已入库的文档块数量。"""
    try:
        import os
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        # 照抄你 vector_store.py 里的配置路径
        chroma_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")

        # 本地初始化一次向量库实例
        embeddings = get_embedder()
        vs = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

        count = vs._collection.count()
        return json.dumps(
            {"ok": True, "chunk_count": count, "status": "knowledge_base_ready"},
            ensure_ascii=False, indent=2
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False, indent=2
        )


# ════════════════════════════════════════════════════════════
#  内部工具函数（与原 rag.py 保持一致）
# ════════════════════════════════════════════════════════════
def _build_results(docs: list, max_chars: int) -> list:
    """将 Document 列表转换为统一的结果字典，含来源路径和页码。"""
    results = []
    for doc in docs:
        raw_content = _trim_text(doc.page_content, max_chars)
        source_path = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", -1)
        industry = doc.metadata.get("industry", "unknown")

        prefix = (
            f"[来源：{source_path}  第 {page} 页]"
            if page >= 1
            else f"[来源：{source_path}]"
        )

        results.append({
            "content": f"{prefix}\n{raw_content}",
            "source": source_path,
            "page": page,
            "industry": industry,
        })
    return results


# ════════════════════════════════════════════════════════════
#  预热（可选）：服务启动时提前加载 Embedding / Reranker 模型
#  避免第一次调用时等待模型加载的冷启动延迟
# ════════════════════════════════════════════════════════════
async def _warmup():
    try:
        from react_agent.rag.retriever import _get_retriever
        from react_agent.rag.reranker import _get_reranker
        logger.info("[RAG-MCP] 开始预热模型...")
        _get_retriever()
        await asyncio.to_thread(_get_reranker)
        logger.info("[RAG-MCP] 预热完成，服务就绪")
    except Exception as e:
        logger.warning(f"[RAG-MCP] 预热失败（不影响启动）：{e}")


# ════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Windows 下 ProactorEventLoop（默认）与部分异步 I/O 库存在兼容问题，
    # 显式切换为 SelectorEventLoop 可避免事件循环冲突
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),  # 满足 MCP 协议要求
            logging.FileHandler("mcp_debug.log", encoding="utf-8")  # 【核心外挂】写入本地文件
        ]
    )

    # 预热已禁用：MCP 规范要求服务启动后必须立即响应握手，
    # asyncio.run(_warmup()) 会阻塞 stdio，导致客户端握手超时。
    # 模型将在首次工具调用时懒加载。
    # asyncio.run(_warmup())

    # transport="stdio"：通过标准输入/输出与客户端通信
    # 这是 Claude Desktop、Cursor 等 MCP 客户端的默认接入方式
    mcp.run(transport="stdio")