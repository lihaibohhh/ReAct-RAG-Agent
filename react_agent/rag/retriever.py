# react_agent/rag/retriever.py — 优化版本
# 修复清单：BUG-1/2/3/4，RISK-1/2/3/4/5，STYLE-1/2

from __future__ import annotations
import os
import hmac
import hashlib
import asyncio
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

from react_agent.utils.redis_client import get_sync_redis
from react_agent.core.config import settings

# ──────────────────────────────────────────────
# 全局单例 & 锁
# ──────────────────────────────────────────────
_retriever_instance = None
_retriever_lock = threading.Lock()

# RISK-4 修复：独立线程池用于异步写入 Redis，不占用 _retriever_lock 持有时间
_redis_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bm25-redis-save")

# ──────────────────────────────────────────────
# Redis Key & 常量
# ──────────────────────────────────────────────
_BM25_CACHE_KEY = "rag:bm25_index"
_BM25_DOCCOUNT_KEY = "rag:bm25_doc_count"
_BM25_HMAC_KEY = "rag:bm25_hmac"

# RISK-5 修复：将 BM25 检索数量提取为具名常量，避免与 RRF 的 k=60 混淆
_BM25_TOP_K = 5
# RRF 融合常数，语义与循环变量完全隔离
_RRF_K = 60


# ──────────────────────────────────────────────
# RISK-1 修复：HMAC 签名 / 验证工具函数
# ──────────────────────────────────────────────
def _get_hmac_secret() -> bytes | None:
    """
    从 settings 或环境变量读取 HMAC 签名密钥。
    未配置时返回 None，调用方应跳过 Redis 缓存而非抛出异常。
    """
    secret = settings.redis.BM25_HMAC_SECRET or os.getenv("BM25_HMAC_SECRET", "")
    if not secret:
        return None
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def _sign_payload(data: bytes) -> str | None:
    """对序列化数据生成 HMAC-SHA256 十六进制摘要；密钥未配置时返回 None。"""
    secret = _get_hmac_secret()
    if secret is None:
        return None
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def _verify_payload(data: bytes, signature: str) -> bool:
    """使用 compare_digest 防时序攻击地验证签名。"""
    expected = _sign_payload(data)
    if expected is None:
        return False
    return hmac.compare_digest(expected, signature)


# ──────────────────────────────────────────────
# Redis 读写
# ──────────────────────────────────────────────
def _load_bm25_from_redis():
    """
    尝试从 Redis 反序列化 BM25 索引。
    - BUG-1  修复：bytes.decode() 后再 int()，避免 TypeError
    - RISK-1 修复：pickle.loads 前先做 HMAC 签名验证
    """
    secret = _get_hmac_secret()
    if secret is None:
        print("[RAG] ⚠️ BM25_HMAC_SECRET 未配置，跳过 Redis 缓存（安全策略）")
        return None

    try:
        r = get_sync_redis()
        data = r.get(_BM25_CACHE_KEY)
        if not data:
            return None

        # 验证签名，防止 Redis 被写入恶意 payload
        stored_sig = r.get(_BM25_HMAC_KEY)
        if not stored_sig:
            print("[RAG] ⚠️ Redis 中缺少 BM25 签名，拒绝加载")
            return None
        sig_str = stored_sig.decode("utf-8") if isinstance(stored_sig, bytes) else stored_sig
        if not _verify_payload(data, sig_str):
            print("[RAG] ⚠️ BM25 签名验证失败，数据可能被篡改，拒绝加载")
            return None

        retriever = pickle.loads(data)  # 签名已通过，反序列化安全

        # BUG-1 修复：r.get() 以 decode_responses=False 返回 bytes，需先 decode
        raw_count = r.get(_BM25_DOCCOUNT_KEY)
        doc_count = int(raw_count.decode("utf-8")) if raw_count else 0
        print(f"[RAG] ✅ BM25 索引从 Redis 加载（文档数: {doc_count}）")
        return retriever

    except Exception as e:
        print(f"[RAG] ⚠️ Redis 读取 BM25 失败: {e}，将重建索引")
        return None


def _save_bm25_to_redis(retriever, doc_count: int) -> None:
    """
    将 BM25 索引序列化并写入 Redis（含 HMAC 签名）。
    RISK-4 修复：此函数应通过 _redis_save_executor.submit() 调用，不得在持锁期间直接调用。
    """
    secret = _get_hmac_secret()
    if secret is None:
        print("[RAG] ⚠️ BM25_HMAC_SECRET 未配置，跳过 Redis 写入（安全策略）")
        return

    try:
        r = get_sync_redis()
        data = pickle.dumps(retriever)
        signature = _sign_payload(data)
        ttl = settings.redis.BM25_INDEX_TTL

        # Pipeline 事务写入（MULTI/EXEC）：三个 key 原子提交，
        # 避免写入中途崩溃导致"有索引但无签名"的半残状态，
        # 进而引发签名验证失败、反复重建的问题。
        pipe = r.pipeline()
        pipe.set(_BM25_CACHE_KEY, data, ex=ttl)
        pipe.set(_BM25_DOCCOUNT_KEY, str(doc_count), ex=ttl)
        pipe.set(_BM25_HMAC_KEY, signature, ex=ttl)
        pipe.execute()
        print(f"[RAG] ✅ BM25 索引已缓存到 Redis（文档数: {doc_count}，TTL: {ttl}s）")
    except Exception as e:
        print(f"[RAG] ⚠️ Redis 写入 BM25 失败: {e}，继续使用内存索引")


# ──────────────────────────────────────────────
# 缓存失效
# ──────────────────────────────────────────────
def invalidate_bm25_cache() -> None:
    """
    知识库更新后调用，同时清除 Redis 缓存与内存单例。
    BUG-2 修复：原实现只清 Redis，_retriever_instance 保持旧值导致失效形同虚设；
               现在在锁内一并重置内存单例。
    """
    global _retriever_instance
    with _retriever_lock:
        try:
            r = get_sync_redis()
            r.delete(_BM25_CACHE_KEY, _BM25_DOCCOUNT_KEY, _BM25_HMAC_KEY)
            print("[RAG] 🗑️ BM25 Redis 缓存已清除")
        except Exception as e:
            print(f"[RAG] ⚠️ 清除 Redis 缓存失败: {e}")
        # 无论 Redis 操作是否成功，内存单例必须重置
        _retriever_instance = None
        print("[RAG] 🗑️ BM25 内存单例已重置，下次调用将重建索引")


# ──────────────────────────────────────────────
# 检索器初始化（双重检查锁定）
# ──────────────────────────────────────────────
def _get_retriever() -> dict:
    global _retriever_instance

    # 快速路径：已初始化，直接返回
    if _retriever_instance is not None:
        return _retriever_instance

    with _retriever_lock:
        # 二次检查：等锁期间可能已被其他线程初始化
        if _retriever_instance is not None:
            return _retriever_instance

        # 延迟导入，仅在首次初始化时触发
        try:
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError("缺少依赖：pip install langchain-chroma")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("缺少依赖：pip install langchain-huggingface sentence-transformers")
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError:
            raise ImportError("缺少依赖：pip install rank_bm25 langchain-community")
        from langchain_core.documents import Document

        # STYLE-2 TODO：将下方两行迁移至 settings.chroma.db_path / settings.embedding.model_name，
        #               统一配置管理入口，消除与 settings.redis.* 的风格不一致。
        chroma_dir = os.getenv("CHROMA_DB_PATH") or str(settings.tools.vector_store.CHROMA_DB_PATH)
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": _BM25_TOP_K})

        # ── 优先从 Redis 加载 BM25 索引 ──────────────────────────
        bm25_retriever = _load_bm25_from_redis()

        if bm25_retriever is not None:
            # BUG-3 修复：Redis 加载路径也需显式设置 .k，与重建路径保持一致
            bm25_retriever.k = _BM25_TOP_K
        else:
            # Redis 无缓存，从向量库全量重建
            all_docs = vectorstore.get(include=["documents", "metadatas"])
            if not all_docs or not all_docs.get("documents"):
                print("[RAG] ⚠️ 知识库为空，本次降级为纯向量检索")
                # BUG-4 修复：不写入 _retriever_instance，
                #            允许知识库写入后下次调用自动恢复 BM25
                return {"bm25": None, "vector": vector_retriever}

            docs_for_bm25 = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
            ]
            bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            bm25_retriever.k = _BM25_TOP_K
            print(f"[RAG] ✅ BM25 索引重建完成（文档数: {len(docs_for_bm25)}）")

            # RISK-4 修复：提交给独立线程池，锁在此之后立即释放，不等待 Redis 写入完成
            _redis_save_executor.submit(_save_bm25_to_redis, bm25_retriever, len(docs_for_bm25))

        _retriever_instance = {"bm25": bm25_retriever, "vector": vector_retriever}
        return _retriever_instance


# ──────────────────────────────────────────────
# 异步双路检索 + RRF 融合
# ──────────────────────────────────────────────
async def _dual_retrieve(q: str) -> list:
    retriever = await asyncio.to_thread(_get_retriever)

    bm25 = retriever.get("bm25")
    vector = retriever.get("vector")

    if bm25 is not None:
        bm25_docs, vector_docs = await asyncio.gather(
            asyncio.to_thread(bm25.invoke, q),
            asyncio.to_thread(vector.invoke, q),
        )
    else:
        vector_docs = await asyncio.to_thread(vector.invoke, q)
        bm25_docs = []

    rrf_scores: dict[str, float] = {}
    doc_map:    dict[str, object] = {}

    def get_key(doc) -> str:
        # RISK-3 修复：用 None 显式检查，避免 chunk_id="" 被 falsy 判断误判
        # RISK-2 注：若知识库存在前100字相同的不同文档，应在 ingest 阶段确保 chunk_id 唯一
        cid = doc.metadata.get("chunk_id")

        # 如果chunk_id因异常情况被赋值为""，即chunk_id为空字符串时依旧是True，避免RISK-2
        return cid if cid is not None else doc.page_content[:100]


    for rank, doc in enumerate(bm25_docs):
        key = get_key(doc)
        # RISK-5 修复：RRF 常数改名为 _RRF_K，循环变量使用 key，语义完全隔离
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (_RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs):
        key = get_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (_RRF_K + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [doc_map[key] for key in sorted_keys[:10]]