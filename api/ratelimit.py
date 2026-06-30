"""
api/ratelimit.py — 请求频率限流 + 每日 token 预算。

设计原则：
- Redis 不可用时 fail-open（放行 + warning 日志），避免 Redis 故障把主链路带死
- 限流算法：固定分钟窗口（per-minute INCR），非滑动窗口
  实现简单，窗口边界处最多允许 2× limit 的短时突发，足够业务场景
- bucket_key 维度：调用方负责解析（已认证用 API Key，匿名用 client:{IP}）
  见 security.get_rate_limit_key()
- 预算记录点：请求结束后（done 帧 or 断连/超时的 finally 块），
  按实际产生的 token 计费，不按预估值
  已知取舍：单条超大请求可冲顶预算上限（事后扣费决定），
  优先保证请求不被中途强杀，次日 0 点自动重置

调用顺序（由路由 handler 保证）：
    1. auth（security.py Depends）
    2. check_rate_limit
    3. check_token_budget
    4. ... 处理请求 ...
    5. record_token_usage（在 generator finally 或 invoke 完成后）
"""
from __future__ import annotations

import logging
import time
from datetime import date

_logger = logging.getLogger(__name__)

_RL_PREFIX = "api:ratelimit"
_BUDGET_PREFIX = "api:budget"


def _key_id(bucket_key: str) -> str:
    """
    取 bucket_key 前 16 字符作为 Redis key 片段。
    bucket_key 可能是 API Key 或 client:{IP}，前 16 位均能有效区分。
    """
    return bucket_key[:16] if bucket_key else "unknown"


# ── 限流 ─────────────────────────────────────────────────────────────────────

async def check_rate_limit(bucket_key: str, limit_rpm: int) -> None:
    """
    固定分钟窗口限流（非滑动窗口）。

    Redis key: api:ratelimit:{key_id}:{unix_minute}
    INCR 后超限 → 429（含 Retry-After 秒数）。
    Redis 不可用 → fail-open（warning 日志）。

    bucket_key: 由 security.get_rate_limit_key() 提供，
                已认证请求用 API Key，匿名请求用 client:{IP}。
    """
    from api.errors import AppError

    key_id = _key_id(bucket_key)
    minute = int(time.time()) // 60
    redis_key = f"{_RL_PREFIX}:{key_id}:{minute}"

    try:
        from react_agent.utils.redis_client import get_async_redis
        r = get_async_redis()

        count = await r.incr(redis_key)
        if count == 1:
            await r.expire(redis_key, 61)  # 窗口结束后 +1s 保险

        if count > limit_rpm:
            from api.metrics import rate_limit_rejections_total
            rate_limit_rejections_total.labels(reason="rpm").inc()
            retry_after = 61 - (int(time.time()) % 60)
            raise AppError(
                status=429,
                title="Too Many Requests",
                detail=f"请求频率超限（{limit_rpm} 次/分钟），请 {retry_after}s 后重试。",
                headers={"Retry-After": str(retry_after)},
            )

    except AppError:
        raise
    except Exception as exc:
        # fail-open：Redis 故障不锁死主链路
        _logger.warning("rate_limit_redis_unavailable | err=%s | fail_open", exc)


# ── 每日 token 预算 ───────────────────────────────────────────────────────────

async def check_token_budget(bucket_key: str, budget: int) -> None:
    """
    检查当日 token 预算余量。

    Redis key: api:budget:{key_id}:{YYYY-MM-DD}
    超限 → 429，detail 说明已用量和上限。
    Redis 不可用 → fail-open。

    bucket_key: 由调用方通过 security.get_rate_limit_key() 提供，
                匿名请求已映射为 client:{IP}，不再跳过空 key 场景。
    已知取舍：事后扣费意味着单条超大请求可能使当日计数超过 budget 上限，
    下条请求才会被拦截。这是有意设计，优先保证已发起的 LLM 调用正常完成。
    """
    from api.errors import AppError

    key_id = _key_id(bucket_key)
    today = date.today().isoformat()
    redis_key = f"{_BUDGET_PREFIX}:{key_id}:{today}"

    try:
        from react_agent.utils.redis_client import get_async_redis
        r = get_async_redis()

        used_bytes = await r.get(redis_key)
        used_tokens = int(used_bytes or 0)

        if used_tokens >= budget:
            from api.metrics import rate_limit_rejections_total
            rate_limit_rejections_total.labels(reason="budget").inc()
            raise AppError(
                status=429,
                title="Daily Budget Exceeded",
                detail=(
                    f"当日 token 预算已耗尽（上限 {budget:,}，已用 {used_tokens:,}）。"
                    "次日 0 点自动重置。"
                ),
            )

    except AppError:
        raise
    except Exception as exc:
        _logger.warning("budget_check_redis_unavailable | err=%s | fail_open", exc)


async def record_token_usage(bucket_key: str, tokens: int) -> None:
    """
    将实际消耗 token 数追加到当日计数器（INCRBY）。

    - 扣费点：generator finally 块（断连/超时/正常完成均执行）
    - 记录实际产生的 token，不按请求预估值
    - bucket_key 为空时跳过（不应发生：get_rate_limit_key 保证非空）
    - 写失败静默，不影响主链路
    TTL 25h：跨零点后自动失效，下一天重新计数。
    """
    if not bucket_key or tokens <= 0:
        return

    key_id = _key_id(bucket_key)
    today = date.today().isoformat()
    redis_key = f"{_BUDGET_PREFIX}:{key_id}:{today}"

    try:
        from react_agent.utils.redis_client import get_async_redis
        r = get_async_redis()
        await r.incrby(redis_key, tokens)
        await r.expire(redis_key, 90_000)  # 25h TTL
        _logger.debug("budget_recorded | key_id=%s tokens=%d today=%s", key_id, tokens, today)
    except Exception as exc:
        _logger.warning("budget_record_failed | tokens=%d err=%s", tokens, exc)
