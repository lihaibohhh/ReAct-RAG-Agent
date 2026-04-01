from __future__ import annotations
import os
import asyncio
import threading


_reranker_retriever_instance = None     # 缓存 BGE Reranker 精排器
_reranker_lock = threading.Lock()       # 线程安全锁：保护 _reranker_retriever_instance 初始化


def _get_reranker():
    """
    构建并缓存 BGE-Reranker 精排器。

    模型选择（按优先级）：
      - BAAI/bge-reranker-v2-m3：推荐，轻量但精度接近 large，中英双语（图片技术选型）
      - BAAI/bge-reranker-large：精度最高（~700M），显存要求高
      - BAAI/bge-reranker-base ：最轻量（~130M），精度中等，兜底选项

    线程安全：双重检查锁定，首次初始化后全程复用单例。
    """
    global _reranker_retriever_instance

    if _reranker_retriever_instance is not None:
        return _reranker_retriever_instance

    with _reranker_lock:
        if _reranker_retriever_instance is not None:
            return _reranker_retriever_instance

        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        except ImportError:
            raise ImportError("缺少依赖。请执行：pip install langchain-community sentence-transformers")

        # BUG-FIX: 默认模型升级为 bge-reranker-v2-m3（图片技术选型表推荐）
        model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        reranker_model = HuggingFaceCrossEncoder(model_name=model_name)
        print(f"[RAG] ✅ Reranker 初始化完成（{model_name}）")

        _reranker_retriever_instance = reranker_model
        return _reranker_retriever_instance


async def _rerank(q: str, docs: list, top_n: int = 3) -> list:
    """
    使用 Cross-Encoder 对候选文档进行精排。

    分数范围与阈值说明：
      - BGE-Reranker 输出原始 logit，范围约 -10 ~ +10
      - 金融研报段落得分通常在 -3 ~ +5 之间
      - 默认阈值 -5：过滤掉极度不相关的内容，保留绝大多数有价值段落
        （原代码默认 0.1 过高，会把大量相关内容过滤掉 ← 这是之前不工作的根本原因）
      - 若知识库内容偏专业术语，可适当调低到 -8：export RERANKER_THRESHOLD=-8

    降级策略：
      - Reranker 模型加载失败 / 推理异常 → 降级返回原始 RRF 排序的前 top_n 条
      - 不返回空列表，保证链路不断

    参数：
      q      : 查询字符串
      docs   : 候选文档列表（来自 _dual_retrieve，最多 10 条）
      top_n  : 精排后最多返回条数（默认 3）

    返回：
      list[Document]，按相关性从高到低，最多 top_n 条
    """
    if not docs:
        return []

    try:
        reranker = await asyncio.to_thread(_get_reranker)

        pairs = [(q, doc.page_content) for doc in docs]
        scores = await asyncio.to_thread(reranker.score, pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        # BUG-FIX: 默认阈值从 0.1 修正为 -5
        # 原因：bge-reranker 输出 logit（非概率），0.1 会过滤掉大量相关金融段落
        # 调优建议：先跑几个 query 打印 scores，再决定是否需要调整
        threshold = float(os.getenv("RERANKER_THRESHOLD", "-5"))

        filtered = [(score, doc) for score, doc in scored if score > threshold]

        if not filtered:
            # 全部低于阈值：说明知识库确实没有相关内容（而非阈值设错）
            # 此时返回空，由 rag.py 向上层返回 has_relevant_content=False
            print(f"[RAG] ℹ️ Reranker 全部低于阈值({threshold})，判定为无相关内容")
            return []

        # 打印 Top 分数，方便调试阈值（生产环境可关闭）
        if os.getenv("RERANKER_DEBUG", "0") == "1":
            for score, doc in scored[:top_n]:
                cid = doc.metadata.get("chunk_id", "?")
                print(f"[RAG][DEBUG] score={score:.3f}  chunk={cid}")

        return [doc for _, doc in filtered[:top_n]]

    except Exception as e:
        # BUG-FIX: 原代码 except 直接 return []，Reranker 一旦出错整条链路返回空
        # 修正为降级返回 RRF 排序结果，保证链路不断裂
        print(f"[RAG] ⚠️ Reranker 失败（{type(e).__name__}: {e}），降级返回 RRF Top-{top_n}")
        return docs[:top_n]