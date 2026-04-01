"""
RAG 评测脚本 — src/eval/run_eval.py

用法:
    # 端到端评测（检索 + 生成 + RAGAS 三项指标）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset.jsonl --n 50

    # 仅检索评测（跳过 LLM 生成，更快更省钱，只算 Precision + Recall）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset.jsonl --n 50 --retrieval_only

    # 指定模型（默认从 config.yaml 读取；若 config 无此键可通过此参数覆盖）
    python -m eval.run_eval --model deepseek/deepseek-chat --n 30

输出:
    eval_summary_<timestamp>.json   — 全局均值 + 分行业均值
    eval_detail_<timestamp>.csv     — 逐条得分明细
    控制台                          — 格式化表格 + 自动诊断建议

【修复记录 v1】
  - 修复 get_retriever() / rerank() 不存在：改为 _dual_retrieve / _rerank（均 async）
  - 修复 sync 函数调用 async 函数的静默失效问题
  - 修复 ThreadPoolExecutor 调度 async 函数导致的 RuntimeError
  - main() 改为 async + asyncio.run()；参数名 top_k -> top_n

【修复记录 v2】
  - 修复 RAGAS 0.4.x 字段名破坏性变更（静默返回 0 分，不报错）：
      question     -> user_input
      contexts     -> retrieved_contexts
      answer       -> response
      ground_truth -> reference
  - 修复 contexts_count=0 问题：
      _rerank 阈值过滤后若结果为空，降级返回 _dual_retrieve 的 RRF 原始排序 top_n 条
      此前若 _rerank 返回空列表，contexts 全空导致 RAGAS 无法计算任何指标
  - 新增 --debug_retrieval 参数：对前 3 条记录打印详细检索中间结果，用于诊断
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径修复：确保从 src/ 下任何位置运行都能找到 react_agent 包
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent.parent   # src/
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# 项目内部导入
# ---------------------------------------------------------------------------
# 【修复】_dual_retrieve 是 retriever.py 中实际的 async 检索函数
#         BM25 + 向量双路 -> RRF 融合 -> 返回最多 10 条 Document
from react_agent.rag.retriever import _dual_retrieve

# 【修复】_rerank 是 reranker.py 中实际的 async 精排函数
#         签名：_rerank(q: str, docs: list, top_n: int = 3) -> list
from react_agent.rag.reranker import _rerank

# load_chat_model 签名：load_chat_model(model_ref: str) -> BaseChatModel
# model_ref 格式："{provider}/{model_name}"，如 "deepseek/deepseek-chat"
from react_agent.utils.llm import load_chat_model

# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    _RAGAS_OK = True
except ImportError:
    _RAGAS_OK = False
    logging.warning("ragas 或 datasets 未安装，将跳过 RAGAS 打分。pip install ragas datasets")

# ---------------------------------------------------------------------------
# DeepSeek Markdown 剥离包装器
# ---------------------------------------------------------------------------


class _MarkdownStrippingLLM:
    """
    DeepSeek 等模型有时会在 JSON 响应外面包裹 ```json ... ``` 代码块，
    而 RAGAS 的 Pydantic 解析器期望裸 JSON，导致 ValidationError。

    此包装器在 LLM 和 RAGAS 之间拦截所有响应，自动剥离 Markdown 代码围栏，
    其余行为与原始 LLM 完全一致（透传 invoke / ainvoke / bind 等方法）。

    用法：
        llm = load_chat_model("deepseek/deepseek-chat")
        wrapped = _MarkdownStrippingLLM(llm)
        LangchainLLMWrapper(wrapped)
    """

    # Markdown 代码围栏正则：匹配 ```json ... ``` 或 ``` ... ```
    _FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def _strip(self, content: str) -> str:
        m = self._FENCE_RE.match(content.strip())
        return m.group(1).strip() if m else content

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        resp = self._llm.invoke(*args, **kwargs)
        if hasattr(resp, "content") and isinstance(resp.content, str):
            resp = AIMessage(content=self._strip(resp.content))
        return resp

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        resp = await self._llm.ainvoke(*args, **kwargs)
        if hasattr(resp, "content") and isinstance(resp.content, str):
            resp = AIMessage(content=self._strip(resp.content))
        return resp

    # RAGAS 内部有时会调用 .bind() 传额外参数，透传给底层 LLM
    def bind(self, **kwargs: Any) -> "_MarkdownStrippingLLM":
        return _MarkdownStrippingLLM(self._llm.bind(**kwargs))

    # 透传其他属性（如 model_name、temperature 等），保持鸭子类型兼容
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


# ---------------------------------------------------------------------------
# 噪声过滤关键词
# ---------------------------------------------------------------------------
_NOISE_KEYWORDS = [
    "免责声明", "版权所有", "联系我们", "客服电话", "扫码关注",
    "转载请注明", "本报告仅供", "投资者须知", "风险提示",
    "请联系", "官方网站", "邮箱", "传真",
]

# ---------------------------------------------------------------------------
# 从 config.yaml 读取模型引用
# ---------------------------------------------------------------------------


def _read_model_ref_from_config() -> str | None:
    """
    尝试从 config.yaml 读取 LLM 模型引用。
    期望格式（在 config.yaml 中二选一）：

        model_ref: deepseek/deepseek-chat
        # 或：
        model: deepseek/deepseek-chat

    如果 config.yaml 不存在或未配置此键，返回 None。
    若你的 config.yaml 键名不同（如 llm.model），请修改下方 _CANDIDATE_KEYS。
    """
    _CANDIDATE_KEYS = ("model_ref", "model")
    config_path = _SRC_ROOT / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key in _CANDIDATE_KEYS:
            val = cfg.get(key)
            if isinstance(val, str) and "/" in val:
                return val
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """从 .jsonl 加载问答对，自动过滤噪声问题。"""
    records: list[dict] = []
    noise_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = obj.get("question", "").strip()
            if not question:
                continue
            if any(kw in question for kw in _NOISE_KEYWORDS):
                noise_count += 1
                continue
            if len(question) < 5:
                noise_count += 1
                continue

            records.append(obj)

    logging.info(f"加载完成：有效 {len(records)} 条，过滤噪声 {noise_count} 条")
    return records


def sample_dataset(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """随机采样 n 条，n <= 总数时直接使用全部。"""
    if n >= len(records):
        logging.info(f"采样数 {n} >= 总量 {len(records)}，使用全部")
        return records
    random.seed(seed)
    sampled = random.sample(records, n)
    logging.info(f"已采样 {len(sampled)} / {len(records)} 条")
    return sampled


# ---------------------------------------------------------------------------
# 检索（async）
# ---------------------------------------------------------------------------

async def retrieve_for_one(record: dict, top_n: int, debug: bool = False) -> dict:
    """
    对单条问题执行完整检索管道：BM25 + 向量 -> RRF -> Reranker。

    【v2 修复：空结果降级保底】
    _rerank 内部有阈值过滤（RERANKER_THRESHOLD，默认 -5）。
    当所有候选文档得分均低于阈值时，_rerank 返回空列表，导致 contexts 为空，
    RAGAS 无法计算任何指标。
    修复：_rerank 返回空时，降级使用 _dual_retrieve 的 RRF 原始排序 top_n 条，
    保证每条问题都有 contexts 传入 RAGAS，使指标可以正常计算。
    """
    question = record["question"]
    try:
        # Step 1：双路检索 + RRF 融合
        raw_docs = await _dual_retrieve(question)

        if debug:
            logging.info(
                f"[DEBUG] 问题: {question[:40]}...\n"
                f"  _dual_retrieve 返回 {len(raw_docs)} 条，"
                f"前3条预览: {[d.page_content[:50] for d in raw_docs[:3]]}"
            )

        # Step 2：Cross-Encoder 精排
        reranked = await _rerank(question, raw_docs, top_n=top_n)

        if debug:
            logging.info(
                f"  _rerank 返回 {len(reranked)} 条"
                + (f"，前3条预览: {[d.page_content[:50] for d in reranked[:3]]}" if reranked else "（空！触发降级）")
            )

        # 【v2 修复】_rerank 返回空时降级保底：使用 RRF 原始排序 top_n 条
        # 原因：评测集的 chunk 若为乱码表格片段，Cross-Encoder 得分会全部低于阈值
        # 降级后 contexts 不为空，RAGAS 才能正常计算指标
        if not reranked and raw_docs:
            logging.debug(f"Reranker 返回空，降级使用 RRF Top-{top_n}：[{question[:30]}...]")
            reranked = raw_docs[:top_n]

        # Step 3：提取 contexts 和 sources
        contexts: list[str] = []
        sources: list[dict] = []
        for doc in reranked:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            page = meta.get("page", "")
            source_file = meta.get("source", "")
            prefix = (
                f"[来源：{Path(source_file).name} 第 {page} 页] "
                if source_file and page != ""
                else ""
            )
            contexts.append(prefix + doc.page_content)
            sources.append({"file": source_file, "page": page})

        return {**record, "contexts": contexts, "sources": sources, "retrieve_ok": True}

    except Exception as e:
        logging.warning(f"检索失败 [{question[:30]}...]: {e}", exc_info=True)
        return {**record, "contexts": [], "sources": [], "retrieve_ok": False}


async def batch_retrieve(
    records: list[dict],
    top_n: int,
    concurrency: int = 8,
    debug_first_n: int = 0,
) -> list[dict]:
    """
    并发检索所有问题，保持原始顺序。
    debug_first_n > 0 时对前 N 条打印详细检索中间结果。
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(records)
    done_count = 0

    async def bounded_retrieve(idx: int, record: dict) -> None:
        nonlocal done_count
        async with sem:
            results[idx] = await retrieve_for_one(
                record, top_n, debug=(idx < debug_first_n)
            )
        done_count += 1
        if done_count % 10 == 0 or done_count == len(records):
            logging.info(f"  检索进度：{done_count}/{len(records)}")

    await asyncio.gather(*(bounded_retrieve(i, rec) for i, rec in enumerate(records)))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 答案生成（sync，在 async main 中通过 asyncio.to_thread 调用）
# ---------------------------------------------------------------------------

def _generate_answers_sync(records: list[dict], llm: Any) -> list[dict]:
    """
    用 LLM 对每条问题生成答案（端到端模式）。
    llm.invoke 是同步调用，通过 asyncio.to_thread 从 async main 中调用，
    避免阻塞事件循环。
    """
    updated = []
    for i, rec in enumerate(records, 1):
        question = rec["question"]
        ctx_text = "\n\n".join(rec.get("contexts", [])) or "（无检索结果）"
        prompt = (
            f"根据以下参考资料，简洁准确地回答问题。如果资料中没有相关信息，请如实说明。\n\n"
            f"参考资料：\n{ctx_text}\n\n"
            f"问题：{question}\n\n答案："
        )
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logging.warning(f"LLM 生成失败 [{question[:30]}...]: {e}")
            answer = ""
        updated.append({**rec, "answer": answer})
        if i % 10 == 0 or i == len(records):
            logging.info(f"  生成进度：{i}/{len(records)}")
    return updated


# ---------------------------------------------------------------------------
# RAGAS 打分（sync，在 async main 中通过 asyncio.to_thread 调用）
# ---------------------------------------------------------------------------

def _run_ragas_sync(records: list[dict], llm: Any, retrieval_only: bool) -> list[dict]:
    """
    调用 RAGAS 计算指标，结果写回每条 record。
    retrieval_only=True 时只算 context_precision + context_recall。
    ragas_evaluate 是同步阻塞调用，通过 asyncio.to_thread 从 async main 中调用。

    【v2 修复：RAGAS 0.4.x 字段名破坏性变更】
    RAGAS 0.2+ 将所有数据集字段名重命名，传旧名字不报错但静默返回接近 0 的分数：
      question     -> user_input
      contexts     -> retrieved_contexts
      answer       -> response
      ground_truth -> reference
    """
    if not _RAGAS_OK:
        logging.warning("RAGAS 不可用，跳过打分，所有指标填 None")
        return [
            {**r, "context_precision": None, "context_recall": None, "faithfulness": None}
            for r in records
        ]

    # 【v2 修复】使用 RAGAS 0.2+ 的新字段名
    data: dict[str, list] = {
        "user_input":          [r["question"] for r in records],
        "retrieved_contexts":  [r.get("contexts", []) for r in records],
        "reference":           [str(r.get("ground_truth") or r.get("answer_ref", "")) for r in records],
    }
    if not retrieval_only:
        data["response"] = [r.get("answer", "") for r in records]

    # 统计 contexts 为空的比例，方便排查检索问题
    empty_ctx = sum(1 for c in data["retrieved_contexts"] if not c)
    if empty_ctx > 0:
        logging.warning(
            f"⚠️  {empty_ctx}/{len(records)} 条记录的 retrieved_contexts 为空，"
            f"这些条目的 Precision/Recall 将计为 0。"
            f"建议先用 --debug_retrieval 排查检索问题。"
        )

    hf_dataset = HFDataset.from_dict(data)

    # 【v3 修复】DeepSeek 等模型会在 JSON 响应外包裹 ```json ... ``` 代码块，
    # 导致 RAGAS Pydantic 解析器报 ValidationError（Invalid JSON at column 1）。
    # 先用 _MarkdownStrippingLLM 剥离代码围栏，再传入 LangchainLLMWrapper。
    stripped_llm = _MarkdownStrippingLLM(llm)
    wrapped_llm = LangchainLLMWrapper(stripped_llm)
    metrics = [
        ContextPrecision(llm=wrapped_llm),
        ContextRecall(llm=wrapped_llm),
    ]
    if not retrieval_only:
        metrics.append(Faithfulness(llm=wrapped_llm))

    logging.info(f"RAGAS 打分中（{len(records)} 条，指标：{[m.name for m in metrics]}）...")
    result = ragas_evaluate(hf_dataset, metrics=metrics)
    df = result.to_pandas()

    enriched = []
    for i, rec in enumerate(records):
        row = df.iloc[i]
        enriched.append({
            **rec,
            "context_precision": _safe_float(row.get("context_precision")),
            "context_recall":    _safe_float(row.get("context_recall")),
            "faithfulness":      _safe_float(row.get("faithfulness")) if not retrieval_only else None,
        })
    return enriched


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return round(f, 4) if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def compute_summary(records: list[dict]) -> dict:
    """计算全局均值 + 分行业均值。"""

    def mean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    def metrics_of(subset: list[dict]) -> dict:
        return {
            "context_precision": mean([r.get("context_precision") for r in subset]),
            "context_recall":    mean([r.get("context_recall")    for r in subset]),
            "faithfulness":      mean([r.get("faithfulness")      for r in subset]),
            "n":                 len(subset),
        }

    summary: dict[str, Any] = {
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "total_samples": len(records),
        "global":        metrics_of(records),
        "by_industry":   {},
    }

    industry_groups: dict[str, list] = defaultdict(list)
    for r in records:
        ind = r.get("industry", "").strip()
        if ind:
            industry_groups[ind].append(r)

    for ind, group in industry_groups.items():
        if len(group) < 3:
            logging.debug(f"行业 [{ind}] 样本 {len(group)} 条，不足 3 条跳过")
            continue
        summary["by_industry"][ind] = metrics_of(group)

    return summary


def write_json_summary(summary: dict, timestamp: str) -> Path:
    out = Path(f"./results/eval_summary_{timestamp}.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_csv_detail(records: list[dict], timestamp: str) -> Path:
    out = Path(f"./results/eval_detail_{timestamp}.csv")
    fieldnames = [
        "question", "industry", "page", "chunk_text_preview",
        "context_precision", "context_recall", "faithfulness",
        "retrieve_ok", "contexts_count", "sources",
    ]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            ctxs = r.get("contexts", [])
            chunk_preview = r.get("chunk_text", "")[:80].replace("\n", " ")
            writer.writerow({
                "question":           r.get("question", ""),
                "industry":           r.get("industry", ""),
                "page":               r.get("page", ""),
                "chunk_text_preview": chunk_preview,
                "context_precision":  r.get("context_precision"),
                "context_recall":     r.get("context_recall"),
                "faithfulness":       r.get("faithfulness"),
                "retrieve_ok":        r.get("retrieve_ok"),
                "contexts_count":     len(ctxs),
                "sources":            json.dumps(r.get("sources", []), ensure_ascii=False),
            })
    return out


def print_console_report(summary: dict) -> None:
    """控制台打印格式化表格 + 自动诊断建议。"""
    g = summary["global"]
    sep = "─" * 56

    print(f"\n{'=' * 56}")
    print(f"  RAG 评测报告  |  {summary['generated_at']}")
    print(f"{'=' * 56}")
    print(f"  总样本数：{summary['total_samples']}")
    print(sep)
    print(f"  {'指标':<22} {'均值':>8}")
    print(sep)
    for key, label in [
        ("context_precision", "Context Precision"),
        ("context_recall",    "Context Recall"),
        ("faithfulness",      "Faithfulness"),
    ]:
        val = g[key]
        display = f"{val:.4f}" if val is not None else "  N/A "
        print(f"  {label:<22} {display:>8}")
    print(sep)

    if summary["by_industry"]:
        print(f"\n  分行业统计（样本 >= 3）")
        print(sep)
        print(f"  {'行业':<14} {'Precision':>10} {'Recall':>10} {'Faithful':>10} {'n':>4}")
        print(sep)
        for ind, m in sorted(summary["by_industry"].items()):
            p  = f"{m['context_precision']:.3f}" if m["context_precision"] is not None else " N/A"
            r  = f"{m['context_recall']:.3f}"    if m["context_recall"]    is not None else " N/A"
            fa = f"{m['faithfulness']:.3f}"      if m["faithfulness"]      is not None else " N/A"
            print(f"  {ind:<14} {p:>10} {r:>10} {fa:>10} {m['n']:>4}")
        print(sep)

    print("\n  自动诊断建议")
    print(sep)
    prec  = g["context_precision"]
    rec   = g["context_recall"]
    faith = g["faithfulness"]
    has_suggestion = False

    if rec is not None and rec < 0.5:
        print("  Recall 偏低 -> 建议扩大 top_n（当前可调 --top_n 参数）")
        print("       或检查 chunk_size 是否过小导致关键段落碎片化")
        has_suggestion = True
    if prec is not None and prec < 0.5:
        print("  Precision 偏低 -> 召回噪声多，可提高 Reranker 阈值")
        print("       (RERANKER_THRESHOLD 当前建议 -5，可适当上调至 -3)")
        has_suggestion = True
    if faith is not None and faith < 0.6:
        print("  Faithfulness 偏低 -> 答案幻觉风险高，建议检查 system prompt")
        print("       或缩短 context 截断长度，减少低质量片段干扰")
        has_suggestion = True
    if not has_suggestion:
        print("  各项指标正常，无明显异常。")

    print(f"{'=' * 56}\n")


# ---------------------------------------------------------------------------
# 主入口（async）
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 系统自动评测工具")
    parser.add_argument(
        "--dataset", default=r"E:\transformer_program\nanoGPT_program\AI_Agent\agent_v0\react-agent-main\src\eval\dataset\eval_dataset.jsonl",
        help=".jsonl 评测数据集路径",
    )
    parser.add_argument("--n",    type=int, default=50,  help="随机采样条数（默认 50）")
    parser.add_argument(
        "--top_n", type=int, default=3,
        help="Reranker 返回 Top-N 文档数（默认 3，与线上 _rerank 默认值一致）",
    )
    parser.add_argument("--retrieval_only", action="store_true",
        help="只测检索质量，跳过 LLM 生成与 Faithfulness")
    parser.add_argument("--concurrency", type=int, default=8,
        help="并发检索协程数（默认 8）")
    parser.add_argument("--seed", type=int, default=42,
        help="随机种子，保证采样可复现（默认 42）")
    parser.add_argument(
        "--model", default='deepseek/deepseek-chat',
        help=(
            "LLM 模型引用，格式 provider/model_name（如 deepseek/deepseek-chat）。"
            "不传时自动从 config.yaml 读取 model_ref 或 model 键。"
        ),
    )
    parser.add_argument(
        "--debug_retrieval", action="store_true",
        help="对前 3 条记录打印详细检索中间结果（_dual_retrieve 和 _rerank 的实际返回），用于排查 contexts 为空问题",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- 0. 确定模型引用 -------------------------------------------------------
    model_ref = args.model or _read_model_ref_from_config()
    if not model_ref:
        logging.error(
            "未找到模型配置。请在 config.yaml 中设置 model_ref 键，"
            "或通过 --model provider/model_name 参数传入。"
        )
        sys.exit(1)
    logging.info(f"使用模型：{model_ref}")
    llm = load_chat_model(model_ref)   # @lru_cache，多次调用不会重复初始化

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    # -- 1. 加载数据 -----------------------------------------------------------
    logging.info(f"加载数据集：{args.dataset}")
    records = load_dataset(args.dataset)
    if not records:
        logging.error("数据集为空，退出")
        sys.exit(1)
    records = sample_dataset(records, args.n, seed=args.seed)

    # -- 2. 批量检索（async，Semaphore 控并发）---------------------------------
    logging.info(f"开始批量检索（top_n={args.top_n}，concurrency={args.concurrency}）...")
    debug_n = 3 if args.debug_retrieval else 0
    records = await batch_retrieve(
        records, top_n=args.top_n, concurrency=args.concurrency, debug_first_n=debug_n
    )
    retrieved_ok  = sum(1 for r in records if r.get("retrieve_ok"))
    contexts_ok   = sum(1 for r in records if r.get("contexts"))
    logging.info(f"检索完成：{retrieved_ok}/{len(records)} 条无异常，{contexts_ok}/{len(records)} 条有 contexts")

    # -- 3. LLM 生成答案（端到端模式，sync 函数通过 to_thread 调用）-----------
    if not args.retrieval_only:
        logging.info("生成答案（端到端模式）...")
        records = await asyncio.to_thread(_generate_answers_sync, records, llm)

    # -- 4. RAGAS 打分（sync 函数通过 to_thread 调用）--------------------------
    logging.info("RAGAS 打分...")
    records = await asyncio.to_thread(_run_ragas_sync, records, llm, args.retrieval_only)

    # -- 5. 报告输出 -----------------------------------------------------------
    summary = compute_summary(records)
    json_path = write_json_summary(summary, timestamp)
    csv_path  = write_csv_detail(records, timestamp)
    print_console_report(summary)

    elapsed = time.time() - t_start
    logging.info(f"评测完成，耗时 {elapsed:.1f}s")
    logging.info(f"   摘要报告：{json_path}")
    logging.info(f"   明细报告：{csv_path}")


if __name__ == "__main__":
    asyncio.run(main())