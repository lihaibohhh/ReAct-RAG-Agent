from __future__ import annotations
import os
import asyncio
from typing import Any
from langgraph.runtime import get_runtime
from langchain_core.tools import tool
from react_agent.memory.context import Context
from react_agent.utils.tool_helpers import _ok, _err, _shrink_search_results, with_retry
from react_agent.core.config import settings


# ================================================================================
# 工具 1：Web 搜索（Tavily API）
# ================================================================================
# 安装依赖：pip install tavily-python
# 配置环境变量：
#   - TAVILY_API_KEY：Tavily API 密钥（必须）
#   - MAX_SEARCH_RESULTS：最多返回多少条搜索结果（可选，默认 5）
@tool(
    description=(
            "【仅限通用互联网信息】：本工具只能用于查询公开互联网上的通用信息。\n"
            "【绝对禁止】：如果用户问题涉及任何金融报告、研究报告、内部文献、PDF文档、"
            "数据库中的具体数据或数值，即使 query_internal_knowledge 返回结果不完整，"
            "也绝对禁止调用本工具补充。直接用'文献未提及'填写缺失项。\n\n"
            "使用互联网搜索引擎获取最新、公开的客观事实。\n"
            "适用于：查找通用技术文档、开源代码库版本、新闻或核实基础概念。\n"
            "【严格红线】：当用户的指令明确要求“对比本地文献”、“查阅内部知识库”，或涉及学术论文PDF的具体推导细节时，"
            "绝对禁止调用此工具！即使内部知识库检索不到结果，也必须直接告知用户“未找到信息”，严禁使用本工具进行互联网兜底搜索。\n"
            "输入必须是简洁的搜索关键词。"
    )
)
@with_retry(
    tool_name="search",
    max_retries=settings.tools.search.max_retries,
    timeout=settings.tools.search.timeout
)
async def search(query: str) -> dict[str, Any]:
    """
    Web 搜索工具 — 通过 Tavily API 搜索互联网最新信息。

    功能描述：
      - 接收用户查询，通过 Tavily API 搜索互联网
      - 裁剪搜索结果以适应上下文大小
      - 返回前几条最相关的结果（标题、链接、摘要、发布时间）

    参数：
      query (str)：搜索关键词或短句
                   - 不能为空
                   - 最多 300 字符（超出会自动截断）
                   - 可以使用自然语言问题或关键词

    返回：
      {
        “ok”: True/False,
        “data”: {
          “results”: [                    # 搜索结果列表
            {
              “title”: str,               # 搜索结果标题
              “url”: str,                 # 链接
              “content”: str,             # 摘要（最多 800 字）
              “score”: float,             # 匹配分数（可能为 None）
              “published_date”: str       # 发布日期（可能为 None）
            },
            ...
          ],
          “answer”: str                   # Tavily 生成的简洁回答（可选）
        },
        “error”: {...}                    # 失败时的错误信息
      }

    执行逻辑：
      1. 输入清理：strip() 并检查是否为空
      2. 输入截断：超过 300 字符自动截断
      3. 依赖检查：延迟 import，缺少依赖返回友好错误
      4. 密钥检查：检查 TAVILY_API_KEY 环境变量
      5. 配置读取：从 runtime context 或 MAX_SEARCH_RESULTS 环境变量读取返回数
      6. API 调用：使用 TavilySearch.ainvoke() 执行搜索
      7. 结果裁剪：通过 _shrink_search_results() 压缩到适当大小
      8. 异常处理：临时错误（超时、连接）raise 给 with_retry，永久错误返回 _err()

    错误场景：
      - BAD_INPUT：query 为空
      - MISSING_DEPENDENCY：未安装 langchain-tavily
      - MISSING_API_KEY：缺少 TAVILY_API_KEY 环境变量
      - SEARCH_FAILED：API 调用出错（非临时错误）
      - 超时/连接错误：raise，由 with_retry 自动重试 2 次

    示例：
      result = await search(“Python 3.12 最新功能”)
      # 返回关于 Python 3.12 的最新搜索结果
    """
    tool_name = "search"
    q = (query or "").strip()

    # ════════════════════ 阶段 1：输入校验 ════════════════════
    if not q:
        return _err(
            tool_name=tool_name,
            query=q,
            code="BAD_INPUT",
            message="query 不能为空，请提供要搜索的关键词。"
        )
    if len(q) > 300:
        q = q[:300]

    # ════════════════════ 阶段 2：依赖检查 ════════════════════
    # 延迟 import 是为了：如果这个工具没有被使用，就不会强制安装依赖
    # 或者如果依赖缺失，整个 tools.py 模块也不会因此 import 失败
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_DEPENDENCY",
            message="未安装依赖 langchain-tavily。请执行：pip install langchain-tavily",
        )

    # ════════════════════ 阶段 3：密钥检查 ════════════════════
    # Tavily API 需要密钥认证，提前检查能更快给出友好错误
    if not settings.secrets.TAVILY_API_KEY.strip():
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_API_KEY",
            message="缺少环境变量 TAVILY_API_KEY，搜索工具不可用。请先设置该 Key。",
        )

    # ════════════════════ 阶段 4：执行搜索 ════════════════════
    try:
        # 读取配置：最多返回几条结果
        max_results = 5
        try:
            # 优先从 LangGraph runtime context 读取
            runtime = get_runtime(Context)
            max_results = int(getattr(runtime.context, "max_search_results", 5))
        except Exception:
            # fallback：从环境变量读取
            max_results = int(settings.tools.search.max_search_results or "5")

        # 初始化搜索器
        wrapped = TavilySearch(max_results=max_results)

        # 调用搜索 API（Tavily 支持两种调用签名）
        try:
            raw = await wrapped.ainvoke({"query": q})
        except Exception:
            # 如果字典形式失败，尝试直接传字符串
            raw = await wrapped.ainvoke(q)

        # 裁剪结果以适应上下文
        data = _shrink_search_results(raw, max_items=min(5, max_results))
        return _ok(
            tool_name=tool_name,
            query=q,
            data=data,
            meta={"max_results": max_results},
        )

    # ════════════════════ 阶段 5：异常处理 ════════════════════
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
        # 临时错误：raise 给 with_retry 装饰器，自动重试
        # 不能在这里 try-except 吞掉，否则 with_retry 无法捕获
        raise
    except Exception as e:
        # 永久错误（API 返回 4xx、解析失败等）：返回失败结果，不重试
        return _err(
            tool_name=tool_name,
            query=q,
            code="SEARCH_FAILED",
            message=f"搜索工具调用失败：{type(e).__name__}: {e}",
        )