from __future__ import annotations

import json

from langchain_core.messages import AIMessage


def _extract_deepseek_v4_usage(msg: AIMessage) -> dict:
    """
    提取 DeepSeek V4 Flash / Pro 的 token usage（via LangChain ChatOpenAI）

    两条数据路径（streaming vs non-streaming 行为不同）：
    1. 非流式（ainvoke）：response_metadata.token_usage 含完整三桶字段
       {"prompt_cache_hit_tokens": N, "prompt_cache_miss_tokens": N, "completion_tokens": N}
    2. 流式（astream_events）：response_metadata 只有 finish_reason，无 token_usage；
       usage 在 usage_metadata（LangChain 归一化版本）：
       {"input_tokens": N, "output_tokens": N, "input_token_details": {"cache_read": N}}
    """

    # ── 路径1：非流式，response_metadata.token_usage 含完整三桶 ────────────
    raw: dict = {}
    rm = getattr(msg, "response_metadata", None)
    if isinstance(rm, dict):
        tu = rm.get("token_usage")
        if isinstance(tu, dict) and tu:
            raw = tu

    if raw:
        cache_hit_tokens  = int(raw.get("prompt_cache_hit_tokens") or 0)
        cache_miss_tokens = int(raw.get("prompt_cache_miss_tokens") or 0)
        completion_tokens = int(raw.get("completion_tokens") or 0)
        ctd = raw.get("completion_tokens_details")
        reasoning_tokens = (
            int(ctd.get("reasoning_tokens") or 0)
            if isinstance(ctd, dict) else 0
        )
    else:
        # ── 路径2：流式 fallback，usage_metadata（LangChain 归一化）─────────
        um = getattr(msg, "usage_metadata", None) or {}
        itd = um.get("input_token_details") or {}
        cache_hit_tokens  = int(itd.get("cache_read") or 0)
        total_input       = int(um.get("input_tokens") or 0)
        cache_miss_tokens = max(0, total_input - cache_hit_tokens)
        completion_tokens = int(um.get("output_tokens") or 0)
        otd = um.get("output_token_details") or {}
        reasoning_tokens  = int(otd.get("reasoning_tokens") or 0)

    return {
        # 计费核心（三桶）
        "cache_hit_tokens":   cache_hit_tokens,
        "cache_miss_tokens":  cache_miss_tokens,
        "completion_tokens":  completion_tokens,
        # 行为分析（不计费）
        "reasoning_tokens":   reasoning_tokens,
        # 校验用推算字段
        "prompt_tokens":      cache_hit_tokens + cache_miss_tokens,
        "total_tokens":       cache_hit_tokens + cache_miss_tokens + completion_tokens,
    }


def _estimate_openai_cost_usd(*, model_name: str, usage: dict, price_table: dict) -> float:
    """按 $/1M tokens 估算费用"""
    price = price_table.get(model_name) or {}
    if not price:
        return 0.0

    return (
        usage["cache_hit_tokens"] / 1_000_000 * price["cache_hit"] +
        usage["cache_miss_tokens"] / 1_000_000 * price["cache_miss"] +
        usage["completion_tokens"] / 1_000_000 * price["output"]
    )


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def bound_tool_payload(content: str, max_chars: int) -> str:
    """
    结构感知截断，兼容 _ok() 和 _err() 两种信封：
      - data.results 是 list  → 逐条裁剪，保留 has_relevant_content / answer 等标志位
      - data 是 null（_err）   → 保护性截断 error.message，信封原样保留
      - data 是扁平 dict（excel）→ 原样放行
    """
    floor = max(max_chars, 64)  # 防止 max_chars 被误设过小导致信息全丢

    try:
        payload = json.loads(content or "{}")
    except Exception:
        return _trim_text(content, floor)

    if not isinstance(payload, dict):
        return _trim_text(content, floor)

    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = payload["meta"] = {}

    # ── 失败信封：data 为 None，只保护性截断 error.message ──────────────
    if payload.get("ok") is False or payload.get("data") is None:
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            if len(err["message"]) > floor:
                err["message"] = _trim_text(err["message"], floor)
                meta["truncated"] = True
        return json.dumps(payload, ensure_ascii=False)

    data = payload.get("data")
    if not isinstance(data, dict):
        return json.dumps(payload, ensure_ascii=False)

    # ── 成功信封：只对 results 列表逐条裁剪，标志位字段全部保留 ──────────
    # 如果超出，保证每条result内容完整性的基础上，抛弃掉后续所有条内容。
    results = data.get("results")
    if isinstance(results, list) and results:
        kept, used = [], 0
        for item in results:
            piece = json.dumps(item, ensure_ascii=False)
            if used + len(piece) > max_chars:
                break
            kept.append(item)
            used += len(piece)
        if len(kept) < len(results):
            data["results"] = kept
            meta["truncated"] = True
            meta["kept_results"] = len(kept)
            meta["total_results"] = len(results)
        else:
            meta.setdefault("truncated", False)

    return json.dumps(payload, ensure_ascii=False)