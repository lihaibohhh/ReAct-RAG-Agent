from __future__ import annotations
import os
import asyncio
from langchain_core.tools import tool
from react_agent.utils.tool_helpers import _ok, _err, _trim_text, with_retry
from react_agent.rag.retriever import _dual_retrieve
from react_agent.rag.reranker import _rerank
from react_agent.core.config import settings
from react_agent.rag.semantic_cache import get_cached_result, set_cached_result


@tool(
    description=(
        "【触发条件】用户询问以下内容时，必须第一个调用本工具：\n"
        "  - 具体公司的财务数据（营收、毛利率、净利润、负债率等）\n"
        "  - 行业研究结论、券商观点、评级、目标价、盈利预测\n"
        "  - 政策文件、监管规定的具体条款\n"
        "  - 研报中的表格数据、图表数据、量化指标\n"
        "  示例 query：'比亚迪2023Q3毛利率'、'新能源行业政策'、'中芯国际目标价'\n\n"
        "【不触发条件】以下情况不使用本工具：\n"
        "  - 问题是通用概念解释（如'什么是PE'），无需查文献\n"
        "  - 用户明确要求搜索互联网最新新闻\n\n"
        "【与 search 的优先级】本工具优先于 search；"
        "只有本工具返回 has_relevant_content=False 后，才允许考虑是否调用 search。\n\n"
        "【输入】精炼的金融关键词，包含：公司名 + 指标 + 报告期（已知时）\n"
        "  好的输入：'比亚迪 毛利率 2023'\n"
        "  差的输入：'请帮我查一下比亚迪最近的情况'\n\n"
        "【输出关键字段】\n"
        "  data.results[].content   — 文档片段，已含来源路径和页码前缀，可直接在回答中引用\n"
        "  data.has_relevant_content — False 时必须告知用户未找到，禁止编造"
    )
)
@with_retry(
    tool_name="query_internal_knowledge",
    max_retries=settings.tools.rag.max_retries,
    timeout=settings.tools.rag.timeout
)
async def query_internal_knowledge(query: str) -> dict:
    """
    RAG 金融研报知识库查询工具。

    执行流程：
      1. 输入校验
      2. 语义缓存查找（命中则直接返回，跳过检索）
      3. 双路召回（BM25 + 向量，各 Top-5，RRF 合并为 Top-10）
      4. Reranker 精排（bge-reranker-v2-m3，阈值过滤，取 Top-3）
      5. 构造结果（含来源文件路径 + 页码，面试核心卖点）
      6. 写入语义缓存

    返回结构：
      {
        "ok": True/False,
        "data": {
          "results": [
            {
              "content": str,   # 文档片段（含来源路径和页码前缀，最多 800 字）
              "source":  str,   # 来源文件路径
              "page":    int    # 来源页码（-1 表示无页码信息）
            },
            ...                 # 最多 3 条
          ],
          "has_relevant_content": bool  # False 时 LLM 应拒答，不应编造
        },
        "meta": {
          "retrieved_count":  int,  # 最终返回文档数
          "candidates_count": int,  # 双路召回候选数
          "stage":            str   # 执行阶段标记，便于诊断
        }
      }

    环境变量：
      CHROMA_DB_PATH        向量数据库路径（默认 ./chroma_db）
      EMBEDDING_MODEL       Embedding 模型（默认 BAAI/bge-small-zh-v1.5）
      RERANKER_MODEL        Reranker 模型（默认 BAAI/bge-reranker-v2-m3）
      RERANKER_THRESHOLD    相关性阈值（默认 0.1，bge-reranker logit 范围约 -10~+10）
      RERANKER_DEBUG        打印每条 chunk 得分，用于调优阈值（"1" 开启）
    """
    tool_name = "query_internal_knowledge"
    q = (query or "").strip()

    # ════ 输入校验 ════
    if not q:
        return _err(
            tool_name=tool_name,
            query=q,
            code="BAD_INPUT",
            message="检索词不能为空"
        )

    try:
        # ════ 语义缓存查找 ════
        cached_docs = await get_cached_result(q)
        if cached_docs is not None:
            max_chars = settings.tools.rag.max_content_chars
            results = _build_results(cached_docs, max_chars)
            hrc = len(results) > 0
            return _ok(
                tool_name=tool_name,
                query=q,
                data={"results": results, "has_relevant_content": hrc},
                meta={"retrieved_count": len(results), "candidates_count": 0,
                      "stage": "semantic_cache_hit", "has_relevant_content": hrc}
            )

        # ════ 第一步：双路召回（BM25 + 向量） ════
        candidates = await _dual_retrieve(q)

        if not candidates:
            hrc = False
            return _ok(
                tool_name=tool_name,
                query=q,
                data={"results": [], "has_relevant_content": hrc},
                # BUG-FIX: 原代码此处 meta 缺少 candidates_count 字段，与有结果路径不一致
                meta={"retrieved_count": 0, "candidates_count": 0,
                      "stage": "dual_retrieve_empty", "has_relevant_content": hrc}
            )

        # ════ 第二步：Reranker 精排，取 Top-3 ════
        docs, top_score = await _rerank(q, candidates, top_n=3)

        # 精排后写入语义缓存（仅在有结果时）
        await set_cached_result(q, docs, top_score=top_score)

        # ════ 第三步：构造返回结果 ════
        max_chars = settings.tools.rag.max_content_chars
        results = _build_results(docs, max_chars)
        hrc = len(docs) > 0

        return _ok(
            tool_name=tool_name,
            query=q,
            data={
                "results": results,
                "has_relevant_content": hrc
            },
            meta={
                "retrieved_count":  len(results),
                "candidates_count": len(candidates),
                "stage": "dual_retrieve + rerank",
                "has_relevant_content": hrc
            }
        )

    # ════ 异常处理 ════
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
        raise   # 临时错误：交给 with_retry 自动重试
    except Exception as e:
        return _err(
            tool_name=tool_name,
            query=q,
            code="RAG_SEARCH_FAILED",
            message=f"知识库检索失败: {type(e).__name__}: {e}"
        )


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────
def _build_results(docs: list, max_chars: int) -> list:
    """
    将 Document 列表转换为统一的结果字典列表。

    每条结果包含：
      - content : 正文（含来源路径和页码前缀，LLM 可直接引用）
      - source  : 来源文件路径（方便前端展示可点击链接）
      - page    : 页码（金融研报问答核心卖点，-1 表示无页码信息）

    BUG-FIX:
      - 原代码（语义缓存路径 + 正常路径）均只返回 source，缺少 page 字段
      - pdf_parser.py 已将 page 写入 metadata，此处需显式取出
      - 两条路径（缓存命中 / 正常检索）统一用本函数，避免字段不一致
    """
    results = []
    for doc in docs:
        raw_content = _trim_text(doc.page_content, max_chars)
        source_path = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", -1)   # pdf_parser.py 注入的页码
        industry = doc.metadata.get("industry", "unknown")

        # 把来源和页码拼到正文最前面
        # LLM 读到此格式可以在回答中直接引用："根据 xxx.pdf 第 N 页……"
        if page >= 1:
            prefix = f"[来源：{source_path}  第 {page} 页]"  # page 从 1 开始
        else:
            prefix = f"[来源：{source_path}]"

        results.append({
            "content": f"{prefix}\n{raw_content}",
            "source":  source_path,
            "page":    page,
            "industry": industry
        })
    return results