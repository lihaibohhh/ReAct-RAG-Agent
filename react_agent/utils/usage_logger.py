"""
usage_logger.py — Token 用量日志与提取工具

三层用途：
  1) 运维层：写入 JSONL 结构化日志，可接 Prometheus / Grafana / ELK
  2) 业务层：按用户、会话维度聚合，供成本报表和预算管控
  3) 展示层：从 Agent result 中提取"人话"摘要，给前端渲染
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── 日志文件路径（生产环境建议改为数据库写入或消息队列） ────
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_USAGE_LOG = _LOG_DIR / "usage_metrics.jsonl"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一层：运维 — 结构化日志
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_usage(
    result: Dict[str, Any],
    *,
    username: str,
    thread_id: str,
    question: str,
    latency_ms: float,
    prev_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将本轮的 token 用量写入 JSONL 日志并返回提取后的 usage 字典。

    日志每行独立 JSON，方便 filebeat / fluentd 采集后导入 ES 或 ClickHouse。
    生产系统可以在此处替换为：
      - Prometheus push_to_gateway (实时指标)
      - 直接写入 PostgreSQL / ClickHouse (持久化分析)
      - 发 Kafka 消息 (异步管道)
    """
    usage = extract_usage(result, prev_snapshot=prev_snapshot)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "thread_id": thread_id,
        "question": question[:200],  # 截断，避免日志膨胀
        "latency_ms": round(latency_ms, 1),
        **usage,
    }

    try:
        with open(_USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[usage_logger] 写入失败: {e}")

    return usage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二层：业务 — 提取结构化数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# State 中需要做差值的累计字段
_CUMULATIVE_KEYS = [
    "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "cache_hit_tokens", "cache_miss_tokens", "total_tokens",
    "estimated_cost_usd", "llm_call_count",
]


def extract_cumulative_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
    """从 result 中提取当前的累计快照（原始值，不做差）。

    调用方应在每轮结束后保存这个快照，下一轮传入 extract_usage 做差值。
    """
    return {k: result.get(k, 0) for k in _CUMULATIVE_KEYS}


def extract_usage(
    result: Dict[str, Any],
    prev_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从 Agent result 中提取"本轮增量"用量。

    关键点：state.py 中的 token 字段是累加的（每次 call_model 往上叠），
    所以 result["total_tokens"] 是从会话开始到当前轮的总和，不是本轮的量。

    做法：用当前累计值 - 上一轮结束时的累计值 = 本轮增量。
    第一轮没有 prev_snapshot 时，累计值本身就是增量。
    """
    if prev_snapshot is None:
        prev_snapshot = {k: 0 for k in _CUMULATIVE_KEYS}

    turn_usage = {}
    for k in _CUMULATIVE_KEYS:
        current = result.get(k, 0)
        previous = prev_snapshot.get(k, 0)
        turn_usage[k] = current - previous

    # tool_runs 用列表长度差值（tool_runs 也是追加式的）
    current_tool_count = len(result.get("tool_runs", []))
    prev_tool_count = prev_snapshot.get("tool_runs_count", 0)
    turn_usage["tool_runs_count"] = current_tool_count - prev_tool_count

    return turn_usage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三层：用户端 — 生成"人话"摘要
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_usage_for_user(usage: Dict[str, Any], latency_ms: float) -> Dict[str, str]:
    """将原始 usage 数据翻译为用户能理解的展示文本。

    核心原则：不暴露 token 数这种系统内部概念，
    而是转化为"调用了几次模型""用了几个工具""花了多少钱""耗时多久"。
    """
    cost = usage.get("estimated_cost_usd", 0.0)
    llm_calls = usage.get("llm_call_count", 0)
    tool_count = usage.get("tool_runs_count", 0)
    total_tok = usage.get("total_tokens", 0)

    # 成本展示：根据量级选择合适的单位
    if cost < 0.001:
        cost_str = "< ¥0.01"
    elif cost < 0.01:
        cost_str = f"≈ ¥{cost * 7.2:.2f}"   # 粗略美元→人民币
    else:
        cost_str = f"≈ ¥{cost * 7.2:.2f}"

    # 耗时展示
    if latency_ms < 1000:
        time_str = f"{latency_ms:.0f} ms"
    else:
        time_str = f"{latency_ms / 1000:.1f} 秒"

    return {
        "耗时": time_str,
        "模型调用": f"{llm_calls} 次",
        "工具使用": f"{tool_count} 次" if tool_count > 0 else "未使用",
        "本轮成本": cost_str,
        "Token 消耗": f"{total_tok:,}",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可选：业务层 — 会话级累计统计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SessionUsageTracker:
    """在 Streamlit session_state 中维护的会话级累计统计。

    用途：
    - 用户侧边栏展示"本次会话累计成本"
    - 业务侧做按用户 / 按会话的成本归因
    - 超过阈值时弹出警告（预算管控）
    """

    def __init__(self):
        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0
        self.total_llm_calls: int = 0
        self.total_tool_runs: int = 0
        self.turn_count: int = 0
        self.turn_usages: list = []      # 每轮的 usage 快照

    def record_turn(self, usage: Dict[str, Any], latency_ms: float):
        """记录一轮对话的用量。"""
        self.total_cost_usd += usage.get("estimated_cost_usd", 0.0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_llm_calls += usage.get("llm_call_count", 0)
        self.total_tool_runs += usage.get("tool_runs_count", 0)
        self.turn_count += 1
        self.turn_usages.append({
            "turn": self.turn_count,
            "latency_ms": round(latency_ms, 1),
            **usage,
        })

    def check_budget(self, limit_usd: float = 1.0) -> Optional[str]:
        """检查是否超预算，返回警告信息或 None。"""
        if self.total_cost_usd >= limit_usd:
            return (
                f"⚠️ 本次会话累计成本已达 ${self.total_cost_usd:.4f} "
                f"(≈ ¥{self.total_cost_usd * 7.2:.2f})，"
                f"超过预算阈值 ${limit_usd:.2f}。"
            )
        return None