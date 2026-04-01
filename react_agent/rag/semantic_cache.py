# react_agent/rag/semantic_cache.py
from __future__ import annotations
import asyncio
import hashlib
import json
import pickle
from typing import Optional
from react_agent.utils.redis_client import get_async_redis
from react_agent.core.config import settings

_CACHE_PREFIX = "rag:semantic:"
_EXACT_PREFIX = "rag:exact:"


def _query_hash(query: str) -> str:
    """精确匹配用 MD5 key"""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


async def get_cached_result(query: str) -> Optional[list]:
    """
    两级查找：
    1. 精确 key 匹配（同样的 query 字符串）
    2. 语义相似匹配（由向量相似度决定）
    当前实现精确匹配；语义匹配需要 redisvl，见下方注释
    """
    try:
        r = get_async_redis()
        key = _EXACT_PREFIX + _query_hash(query)
        data = await r.get(key)
        if data:
            result = pickle.loads(data)
            print(f"[SemanticCache] ✅ 命中缓存: {query[:30]}...")
            return result
    except Exception as e:
        print(f"[SemanticCache] ⚠️ 读取失败: {e}")
    return None


async def set_cached_result(query: str, docs: list):
    """缓存检索结果，TTL 从 settings 读取"""
    try:
        r = get_async_redis()
        key = _EXACT_PREFIX + _query_hash(query)
        ttl = settings.redis.SEMANTIC_CACHE_TTL
        await r.set(key, pickle.dumps(docs), ex=ttl)
        print(f"[SemanticCache] 💾 已缓存: {query[:30]}... (TTL: {ttl}s)")
    except Exception as e:
        print(f"[SemanticCache] ⚠️ 写入失败: {e}")


async def clear_all_cache():
    """知识库更新时调用，清除所有 RAG 缓存"""
    try:
        r = get_async_redis()
        # 扫描并删除所有 rag: 前缀的 key
        keys = []
        async for key in r.scan_iter("rag:*"):
            keys.append(key)
        if keys:
            await r.delete(*keys)
            print(f"[SemanticCache] 🗑️ 已清除 {len(keys)} 个缓存键")
    except Exception as e:
        print(f"[SemanticCache] ⚠️ 清除失败: {e}")