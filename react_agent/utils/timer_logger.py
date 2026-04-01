# react_agent/utils/timer_logger.py
# 职责：Agent 数据建库流水线各阶段计时、异常捕获、吞吐量计算，写入 JSONL 日志并提供可视化汇总表
#
# 用法示例：
#   from react_agent.utils.timer_logger import timer, summarize_last_run
#
#   with timer("T3_embedding", meta={"doc_count": 200, "token_count": 15000}):
#       # 你的大模型 (如 DeepSeek/Gemini/Claude) 嵌入逻辑
#       embeddings.embed_documents(texts)
#   summarize_last_run()

import time
import json
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any

# ── 日志文件路径（建议放在项目根目录下的 logs 或 outputs 文件夹） ───────────
DEFAULT_LOG_PATH = Path("ingestion_metrics.jsonl")

# ── 模块级别：本次 run 的所有计时记录，用于 summarize_last_run() ────────────
_current_run_records: list[dict] = []


@contextmanager
def timer(
        stage: str,
        meta: Optional[dict[str, Any]] = None,
        log_path: Path = DEFAULT_LOG_PATH,
        print_result: bool = True,
):
    """
    上下文管理器：带异常安全的流水线计时器。

    参数
    ----
    stage       : 阶段标识 (如 T1_parse / T2_chunk / T3_embed)
    meta        : 额外信息。若包含 'doc_count' 或 'token_count'，会自动计算吞吐量
    log_path    : JSONL 日志路径
    print_result: 是否在终端实时打印
    """
    t_start = time.perf_counter()
    status = "success"

    try:
        # 将控制权交还给你的业务逻辑代码
        yield
    except Exception as e:
        # 捕捉异常，记录失败状态，但不吃掉异常，继续向上抛出
        status = f"failed: {type(e).__name__}"
        raise
    finally:
        # 【核心护城河】无论业务代码是否崩溃，finally 一定会执行，保证日志不丢失
        elapsed = time.perf_counter() - t_start
        meta_data = meta or {}

        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            **meta_data
        }

        # ── 自动计算工业级吞吐量 ────────────────────────────────────────────────
        if elapsed > 0:
            if "token_count" in meta_data:
                record["tokens_per_sec"] = round(meta_data["token_count"] / elapsed, 1)
            if "doc_count" in meta_data:
                record["docs_per_sec"] = round(meta_data["doc_count"] / elapsed, 3)

        # ── 安全写入 JSONL (Windows 友好) ───────────────────────────────────────
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # 强制 utf-8，防止 Windows 下多语言字符导致写入崩溃
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as io_err:
            # 防止日志系统本身的错误搞崩主程序
            print(f"⚠️ [日志写入失败] {io_err}")

        # ── 追加到本次 run 汇总 ─────────────────────────────────────────────────
        _current_run_records.append(record)

        # ── 终端实时打印 ────────────────────────────────────────────────────────
        if print_result:
            status_icon = "✅" if status == "success" else "❌"
            meta_str = ""
            if meta_data:
                # 过滤并格式化 meta 信息
                kv_parts = [f"{k}={v}" for k, v in meta_data.items() if isinstance(v, (int, float))]
                str_parts = [f"{k}='{v}'" for k, v in meta_data.items() if isinstance(v, str) and len(str(v)) <= 40]

                # 如果算出了吞吐量，直接在终端展示出来
                if "tokens_per_sec" in record:
                    kv_parts.append(f"tps={record['tokens_per_sec']}")
                elif "docs_per_sec" in record:
                    kv_parts.append(f"dps={record['docs_per_sec']}")

                meta_str = "  " + "  ".join(kv_parts + str_parts) if (kv_parts or str_parts) else ""

            print(f"  {status_icon} [{stage}] {elapsed:.3f}s{meta_str}")


def summarize_last_run(clear: bool = True) -> None:
    """
    打印本次流水线调用中所有阶段的计时汇总表，并直观展示错误分布。
    """
    global _current_run_records
    if not _current_run_records:
        print("（本次 run 无计时记录）")
        return

    total_elapsed = sum(r["elapsed_s"] for r in _current_run_records)
    success_count = sum(1 for r in _current_run_records if r.get("status") == "success")
    fail_count = len(_current_run_records) - success_count

    print("\n" + "═" * 70)
    print(f"  📊  Agent 知识库构建耗时汇总 (成功: {success_count}, 失败: {fail_count})")
    print("═" * 70)

    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    stage_fails: dict[str, int] = {}

    for r in _current_run_records:
        stage = r["stage"]
        stage_totals[stage] = stage_totals.get(stage, 0) + r["elapsed_s"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if r.get("status") != "success":
            stage_fails[stage] = stage_fails.get(stage, 0) + 1

    for stage, elapsed in stage_totals.items():
        count = stage_counts[stage]
        fails = stage_fails.get(stage, 0)

        count_str = f"×{count}" if count > 1 else "    "
        # 如果某个阶段有失败记录，用红色高亮（ANSI转义）或者醒目标记提示
        fail_str = f"  [!{fails}次报错]" if fails > 0 else ""

        pct = elapsed / total_elapsed * 100 if total_elapsed > 0 else 0
        bar = "█" * int(pct / 5)  # 每格代表 5%

        print(f"  {stage:<22} {count_str}  {elapsed:7.2f}s  {pct:5.1f}%  {bar}{fail_str}")

    print("─" * 70)
    print(f"  {'总计':<28} {total_elapsed:7.2f}s  100.0%")
    print("═" * 70 + "\n")

    if clear:
        _current_run_records = []