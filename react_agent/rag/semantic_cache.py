# react_agent/rag/semantic_cache.py
from __future__ import annotations
import os
import asyncio
import hashlib
import pickle
import logging
import threading
from typing import Optional
import numpy as np
from react_agent.utils.redis_client import get_async_redis
from react_agent.core.config import settings
from react_agent.utils.embedder import get_embedder


logger = logging.getLogger(__name__)

_EXACT_PREFIX = "rag:exact:"              # Tier-1 精确匹配结果
_SEM_VEC_PREFIX = "rag:semantic:vec:"     # Tier-2 查询向量
_SEM_RES_PREFIX = "rag:semantic:res:"     # Tier-2 检索结果
_SEM_INDEX_KEY = "rag:semantic:index"     # Tier-2 索引（HASH: id → query 预览）

_CACHE_MIN_SCORE = float(os.getenv("CACHE_MIN_SCORE", "0.5"))  # 新增，可通过环境变量调，只缓存相关性高的结果
_NO_RESULT_TTL = 300  # 无结果短缓存 5 分钟
_MAX_INDEX_SIZE = 2000


async def _embed(query: str) -> np.ndarray:
    """异步嵌入查询字符串（在线程池中运行，不阻塞事件循环）"""
    embedder = await asyncio.to_thread(get_embedder)
    vec = await asyncio.to_thread(embedder.embed_query, query)
    return np.array(vec, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def _query_hash(query: str) -> str:
    """MD5 key，用于 Tier-1 精确匹配 & Tier-2 entry_id（同一 query 自然去重）"""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _decode(v) -> str:
    """Redis bytes key 解码（decode_responses=False 模式下需要手动 decode）"""
    return v.decode("utf-8") if isinstance(v, bytes) else v


# ── Tier-1: 精确匹配 ──────────────────────────────────────────────────────────
async def _exact_get(query: str) -> Optional[list]:
    try:
        r = get_async_redis()
        data = await r.get(_EXACT_PREFIX + _query_hash(query))
        if data:
            logger.info(f"[SemanticCache] ✅ Tier-1 精确命中: {query[:40]}...")
            return pickle.loads(data)
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-1 读取失败: {e}")
    return None


async def _exact_set(query: str, docs: list, ttl: int) -> None:
    try:
        r = get_async_redis()
        await r.set(_EXACT_PREFIX + _query_hash(query), pickle.dumps(docs), ex=ttl)
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-1 写入失败: {e}")


# ── Tier-2: 语义相似匹配 ──────────────────────────────────────────────────────
async def _semantic_get(query_vec: np.ndarray) -> Optional[list]:
    """
    批量拉取所有已缓存向量 → 线性余弦相似度扫描 → 返回最优命中结果。

    规模估算（_MAX_INDEX_SIZE = 2000）：
      bge-small-zh-v1.5 向量维度 512，float32 = 2 KB/条
      2000 条 × 2 KB = ~4 MB，单次 MGET 完全可接受。
      向量已过期（mget 返回 None）的条目会被静默跳过，不影响正确性。
    """
    threshold = settings.redis.SEMANTIC_CACHE_THRESHOLD
    try:
        r = get_async_redis()

        raw_ids = await r.hkeys(_SEM_INDEX_KEY)
        if not raw_ids:
            return None

        entry_ids = [_decode(eid) for eid in raw_ids]
        vec_keys  = [_SEM_VEC_PREFIX + eid for eid in entry_ids]
        raw_vecs  = await r.mget(*vec_keys)

        best_score, best_id = -1.0, None
        for eid, raw_vec in zip(entry_ids, raw_vecs):
            if raw_vec is None:
                # 向量 key 已 TTL 过期，对应 index 条目变为孤儿，跳过即可
                continue
            score = _cosine_sim(query_vec, pickle.loads(raw_vec))
            if score > best_score:
                best_score, best_id = score, eid

        if best_score >= threshold and best_id is not None:
            raw_res = await r.get(_SEM_RES_PREFIX + best_id)
            if raw_res:
                logger.info(
                    f"[SemanticCache] ✅ Tier-2 语义命中 "
                    f"(sim={best_score:.4f} ≥ {threshold})"
                )
                return pickle.loads(raw_res)

    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 读取失败: {e}")
    return None


async def _semantic_set(
    entry_id: str,
    query_vec: np.ndarray,
    docs: list,
    ttl: int,
    query: str,
) -> None:
    """
    写入语义索引，含两阶段容量管理：
      1. 清理孤儿条目（向量 key 已 TTL 过期，但 index HASH 仍有记录）
      2. 若清完后仍超限，移除最早的 10% 条目
    """
    try:
        r = get_async_redis()
        index_size = await r.hlen(_SEM_INDEX_KEY)

        if index_size >= _MAX_INDEX_SIZE:
            all_ids = [_decode(eid) for eid in await r.hkeys(_SEM_INDEX_KEY)]
            exists = await r.mget(*[_SEM_VEC_PREFIX + eid for eid in all_ids])
            orphan_ids = [eid for eid, v in zip(all_ids, exists) if v is None]

            if orphan_ids:
                pipe = r.pipeline()
                for oid in orphan_ids:
                    pipe.hdel(_SEM_INDEX_KEY, oid)
                    pipe.delete(_SEM_RES_PREFIX + oid)
                await pipe.execute()
                logger.info(f"[SemanticCache] 🗑️ 清理孤儿条目 {len(orphan_ids)} 个")
                index_size -= len(orphan_ids)

            # 清完孤儿仍然超限 → 移除最早的 10%
            if index_size >= _MAX_INDEX_SIZE:
                n_remove  = max(1, _MAX_INDEX_SIZE // 10)
                evict_ids = all_ids[:n_remove]
                pipe = r.pipeline()
                for oid in evict_ids:
                    pipe.hdel(_SEM_INDEX_KEY, oid)
                    pipe.delete(_SEM_VEC_PREFIX + oid)
                    pipe.delete(_SEM_RES_PREFIX + oid)
                await pipe.execute()
                logger.info(f"[SemanticCache] 🗑️ 容量超限，淘汰最旧 {len(evict_ids)} 条")

        # 原子写入向量、结果、索引记录（三条 Pipeline，TTL 各自独立）
        pipe = r.pipeline()
        pipe.set(_SEM_VEC_PREFIX + entry_id, pickle.dumps(query_vec), ex=ttl)
        pipe.set(_SEM_RES_PREFIX + entry_id, pickle.dumps(docs),      ex=ttl)
        pipe.hset(_SEM_INDEX_KEY, entry_id, query[:60])  # 预览文本仅供调试
        await pipe.execute()

    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 写入失败: {e}")


async def get_cached_result(query: str) -> Optional[list]:
    """
    两级查找（接口签名与原版完全兼容）：

      Tier-1  精确 MD5 匹配 —— 同一 query 字符串，O(1)，无 Embedding 开销
      Tier-2  语义向量匹配 —— 不同措辞但语义相近的 query，
              阈值由 settings.redis.SEMANTIC_CACHE_THRESHOLD（默认 0.95）控制
    """
    # Tier-1: 零 Embedding 开销的快速路径
    result = await _exact_get(query)
    if result is not None:
        return result

    # Tier-2: 语义匹配
    try:
        query_vec = await _embed(query)
        result = await _semantic_get(query_vec)
        if result is not None:
            logger.info("Redis 语义缓存生效 ✅")
            return result
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 查询异常，降级跳过: {e}")

    return None


async def set_cached_result(query: str, docs: list, top_score: float = 0.0):
    """
    写入双级缓存（接口签名与原版完全兼容）：

      空结果      → 仅 Tier-1 短缓存（_NO_RESULT_TTL），不浪费向量索引槽位
      低置信度    → 两级均跳过（top_score < CACHE_MIN_SCORE）
      正常结果    → Tier-1 精确缓存 + Tier-2 语义索引同步写入
      Tier-2 失败 → 降级保留 Tier-1，整体不抛出异常
    """
    base_ttl = settings.redis.SEMANTIC_CACHE_TTL

    if not docs:
        await _exact_set(query, docs, _NO_RESULT_TTL)
        logger.info(f"[SemanticCache] 💾 空结果短缓存 (TTL={_NO_RESULT_TTL}s): {query[:40]}...")
        return

    if top_score < _CACHE_MIN_SCORE:
        logger.info(
            f"[SemanticCache] ⏭️ 跳过缓存 "
            f"(score={top_score:.3f} < {_CACHE_MIN_SCORE}): {query[:40]}..."
        )
        return

    # Tier-1 精确写入
    await _exact_set(query, docs, base_ttl)

    # Tier-2 语义写入（entry_id 与 Tier-1 使用同一 md5，相同 query 自然去重）
    try:
        query_vec = await _embed(query)
        entry_id = _query_hash(query)
        await _semantic_set(entry_id, query_vec, docs, base_ttl, query)
        logger.info(
            f"[SemanticCache] 💾 Tier-1+2 写入完成 "
            f"(score={top_score:.3f}, TTL={base_ttl}s): {query[:40]}..."
        )
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 写入失败，Tier-1 已写入: {e}")


async def clear_all_cache() -> None:
    """
    知识库更新时调用，清除所有 RAG 缓存。
    扫描 rag:* 前缀，包含精确缓存、语义缓存及 BM25 缓存。
    BM25 专属清除请使用 retriever.invalidate_bm25_cache()。
    """
    try:
        r = get_async_redis()
        keys = [k async for k in r.scan_iter("rag:*")]
        if keys:
            await r.delete(*keys)
            logger.info(f"[SemanticCache] 🗑️ 已清除 {len(keys)} 个缓存键（精确 + 语义 + BM25）")
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ 清除失败: {e}")