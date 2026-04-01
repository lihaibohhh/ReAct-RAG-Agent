from __future__ import annotations

from langchain_core.messages import AIMessage


def _extract_openai_usage_from_ai_message(msg: AIMessage) -> dict:
    """兼容 LangChain ChatOpenAI / OpenAI-compatible API 的 usage 提取"""
    usage = {}
    rm = getattr(msg, "response_metadata", None)
    if isinstance(rm, dict):
        for k in ("token_usage", "usage"):
            u = rm.get(k)
            if isinstance(u, dict):
                usage = u
                break

    um = getattr(msg, "usage_metadata", None)
    if not usage and isinstance(um, dict):
        usage = um

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    cached_tokens = 0
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        cached_tokens = int(ptd.get("cached_tokens") or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def _estimate_openai_cost_usd(*, model_name: str, usage: dict, price_table: dict) -> float:
    """按 $/1M tokens 估算费用"""
    p = price_table.get(model_name) or {}
    input_price = float(p.get("input") or 0.0)
    output_price = float(p.get("output") or 0.0)
    cached_input_price = float(p.get("cached_input") or input_price)

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)

    billable_prompt = max(0, prompt_tokens - cached_tokens)

    return (
        billable_prompt * input_price / 1_000_000.0
        + cached_tokens * cached_input_price / 1_000_000.0
        + completion_tokens * output_price / 1_000_000.0
    )


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"