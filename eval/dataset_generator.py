"""
dataset_generator.py
────────────────────
从已有 PDF 研报 chunks 中，用 LLM 自动生成评测 QA 对。

输出格式（JSONL，每行一条）：
{
  "question":     "比亚迪2023年第三季度毛利率是多少？",
  "ground_truth": "根据研报，比亚迪2023年Q3毛利率为22.1%，同比提升3.2个百分点。",
  "source_file":  "半导体/中芯国际_2023Q3点评.pdf",
  "page":         5,
  "industry":     "半导体",
  "chunk_text":   "...(原始chunk，评测时用于校验)..."
}

用法：
    # 在 src/ 目录下执行（cd src）
    python eval/dataset_generator.py \
        --data_dir ./data \
        --output   eval/dataset/eval_dataset.jsonl \
        --n_per_chunk 2 \
        --max_chunks 200 \
        --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

# ── 把 src/ 加入 path，复用项目已有模块 ──────────────────────────────────────
_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
load_dotenv(_SRC.parent / ".env", override=False)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ★ 新增：Pydantic 导入
from pydantic import BaseModel, Field

from react_agent.utils.llm import load_chat_model
from react_agent.core.config import settings


# ══════════════════════════════════════════════════════════════════════════════
# ★ 核心改造 Step 1：定义数据模型
#
# 工程直觉：把你"期望 LLM 输出的形状"，用 Python 类写出来。
# 这就是你和 LLM 之间的"合同"——它必须按这个格式填，不能乱发挥。
#
# Field(description=...) 的作用：这段描述会被翻译进 JSON Schema 里，
# 模型在生成时能读到它，相当于表格里的填写说明，比在 Prompt 里叮嘱更可靠。
# ══════════════════════════════════════════════════════════════════════════════

class QAPair(BaseModel):
    """单个问答对。"""
    question: str = Field(
        description="问题，必须包含具体的公司名称、行业或产品名，禁止使用'该公司'等模糊代词"
    )
    ground_truth: str = Field(
        description="答案，必须直接来自研报片段，禁止补充片段中未出现的数字或结论"
    )


class QAList(BaseModel):
    """LLM 每次调用的完整返回结构。

    为什么要包一层 QAList，而不直接用 list[QAPair]？
    因为 .with_structured_output() 要求顶层是一个对象，不能是裸数组。
    这是 JSON Schema 规范的约束。包一层是标准做法。
    """
    pairs: list[QAPair] = Field(
        default_factory=list,
        description="生成的QA对列表。若片段为免责声明、目录、分析师信息等无实质内容，返回空列表"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
# ★ 核心改造 Step 2：简化 Prompt
#
# 原来结尾那一大段"请严格以 JSON 数组格式输出..."被删掉了。
# 为什么可以删？因为 .with_structured_output() 在底层通过 Function Calling
# 强制了格式，模型根本没有机会输出"好的，以下是您的答案："这类废话。
# 在 Prompt 里再叮嘱一遍是多余的，而且有时反而会干扰模型的注意力。
#
# 保留的部分：8 条业务规则——这些是业务逻辑，不是格式控制，依然有价值。
# ─────────────────────────────────────────────────────────────────────────────
_QA_PROMPT = """你是一位苛刻的金融分析师，负责构建用于评估 RAG（检索增强生成）系统的高质量测试集。

请仔细阅读下面的金融研报片段。如果该片段不含实质性的商业/财务/行业分析内容（如免责声明、评级标准说明、分析师联系方式、纯文档目录），请将 pairs 字段返回空列表。

如果片段包含实质性内容，请生成 {n} 个高质量问答对，严格遵守以下规则：
1. 必须指名道姓：问题中必须明确包含具体的公司名称、行业名称或具体产品名，绝对不能使用"该公司"、"该行业"、"本项目"等模糊指代！如果片段中找不到具体实体名称，请放弃生成。
2. 拒绝元数据：不要提问关于图表编号（如"图1"）、分析师名字、报告日期等外围信息。
3. 聚焦核心业务：多提问关于营收数据、毛利率、产能规划、行业趋势、竞争格局等需要深度理解的问题。
4. 独立可答：ground_truth 必须直接且精准，不需要用户再去翻看原文档。
5. 若片段中存在表格数据，优先生成针对具体数值的问题。
6. 答案只能来自上方片段：ground_truth 的每一句话都必须能在上方"研报片段"中找到直接依据，严禁根据常识或训练数据补充片段中未出现的数字或结论。
7. 禁止模糊答案：ground_truth 中不允许出现"任意一个"、"例如"、"或者A或B"等不确定表述；若答案本身是列表，则必须完整列出所有列表项。
8. 问题必须多样化：同一 chunk 的 {n} 个问题，至少包含一个"数值查找型"和一个"逻辑/机制分析型"，不得重复拆解同一句话。

━━━ 研报片段 ━━━
{chunk}

来源文件：{source}（第 {page} 页）
━━━━━━━━━━━━━"""

# ── 免责/目录类 chunk 预过滤关键词（命中任意一条则跳过 LLM 调用） ─────────────
_NOISE_PATTERNS = re.compile(
    r"免责声明|分析师声明|评级说明|风险提示.*本报告|版权所有.*禁止|执业编号|"
    r"请务必阅读正文之后的免责|本报告仅供.*客户使用|证券投资咨询业务资格",
    re.S,
)

# ──────────────── 工具函数 ──────────────────────────────────────────────────────────────────


def _get_industry(chunk: Document) -> str:
    return chunk.metadata.get("industry", "未知").strip() or "未知"


def _get_rel_src(chunk: Document, data_dir: Path) -> str:
    """计算相对于 data_dir 的路径字符串，失败时返回文件名。"""
    src = chunk.metadata.get("source", "unknown")
    try:
        return str(Path(src).relative_to(data_dir))
    except ValueError:
        return Path(src).name


def _is_noise_chunk(text: str) -> bool:
    """粗过滤：命中免责/声明类关键词则视为噪声 chunk，跳过 LLM 调用。"""
    return bool(_NOISE_PATTERNS.search(text[:600]))


# ── 语义去重：模块级模型缓存，避免多次调用重复加载 ──────────────────────────
_DEDUP_MODEL = None
_DEDUP_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def _get_dedup_model():
    global _DEDUP_MODEL
    if _DEDUP_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"[dedup] 加载语义去重模型：{_DEDUP_MODEL_NAME} ...")
        _DEDUP_MODEL = SentenceTransformer(_DEDUP_MODEL_NAME)
    return _DEDUP_MODEL


def _deduplicate(records: list[dict], threshold: float = 0.92) -> list[dict]:
    """基于语义向量的去重：对 question 字段计算余弦相似度，过滤重复问题。

    算法：
      1. 批量编码所有 question，得到归一化向量矩阵（N × D）。
      2. 矩阵自乘 embeddings @ embeddings.T 一次性得到余弦相似度矩阵（N × N）。
      3. 贪心遍历：当前问题与已保留集合中任意问题的相似度 >= threshold，则丢弃。

    Args:
        records:   输入记录列表，每条须含 "question" 键。
        threshold: 余弦相似度阈值，默认 0.92；越高则保留越多相似问题。

    Returns:
        去重后的记录列表，顺序与原列表一致，类型不变。
    """
    if len(records) <= 1:
        return list(records)

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print(
            "[dedup] ⚠ sentence-transformers 未安装，回退到前缀去重。\n"
            "        请执行：pip install sentence-transformers"
        )
        seen: set[str] = set()
        out = []
        for r in records:
            key = r["question"][:10]
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    model = _get_dedup_model()
    questions = [r["question"] for r in records]

    embeddings: "np.ndarray" = model.encode(
        questions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape: (N, D)

    sim_matrix: "np.ndarray" = embeddings @ embeddings.T  # shape: (N, N)

    kept: list[int] = []
    for i in range(len(records)):
        if not kept:
            kept.append(i)
            continue
        if sim_matrix[i, kept].max() < threshold:
            kept.append(i)

    return [records[i] for i in kept]


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def load_all_chunks(db_path: Path) -> list[Document]:
    """从本地 Chroma 向量数据库读取全量 chunks。"""
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    print(f"[generator] 从 Chroma 数据库读取 chunks：{db_path}  (embedding: {model_name})")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    results = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    print(f"[generator] 共读取 {len(all_docs)} 个 chunks")
    return all_docs


def stratified_sample(
    chunks: list[Document],
    max_chunks: int,
    seed: int,
) -> list[Document]:
    """
    按行业分层采样：各行业尽量均匀抽取，总数不超过 max_chunks。

    过滤规则：
      - chunk 文本过短（< 100 字）
      - 命中免责/声明类关键词（_is_noise_chunk）
    行业来源：优先 metadata["industry"]，缺失时从路径推断。
    """
    rng = random.Random(seed)

    valid = [
        c for c in chunks
        if len(c.page_content.strip()) >= 100 and not _is_noise_chunk(c.page_content)
    ]
    noise_count = len(chunks) - len(valid)
    if noise_count:
        print(f"[generator] 预过滤：跳过 {noise_count} 个噪声/过短 chunks")

    # 按行业分组（直接从 metadata 读取）
    by_industry: dict[str, list[Document]] = defaultdict(list)
    for chunk in valid:
        industry = _get_industry(chunk)
        by_industry[industry].append(chunk)

    industries = list(by_industry.keys())
    per_industry = max(1, max_chunks // len(industries))

    sampled: list[Document] = []
    for ind, docs in by_industry.items():
        take = min(per_industry, len(docs))
        sampled.extend(rng.sample(docs, take))
        print(f"  行业「{ind}」：{len(docs)} chunks → 抽 {take}")

    if len(sampled) > max_chunks:
        rng.shuffle(sampled)
        sampled = sampled[:max_chunks]

    print(f"[generator] 最终采样 {len(sampled)} 个 chunks 用于生成 QA 对")
    return sampled


class _AdaptiveWorkerPool:
    def __init__(self, initial: int, min_workers: int = 1):
        self._initial = initial
        self._workers = initial
        self._min = min_workers
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial)
        self._pending_reduction = 0

    def acquire(self):
        self._sem.acquire()

    def release(self):
        with self._lock:
            if self._pending_reduction > 0:
                self._pending_reduction -= 1
            else:
                self._sem.release()

    @property
    def workers(self) -> int:
        return self._workers

    def on_rate_limited(self):
        with self._lock:
            if self._workers > self._min:
                self._workers = max(self._min, self._workers - 1)
                self._pending_reduction += 1
                print(f"[adaptive] 触发限流，并发数降至 {self._workers}")

    def on_success(self):
        with self._lock:
            if self._workers < self._initial:
                self._workers = min(self._initial, self._workers + 1)
                self._sem.release()


# ★ 核心改造 Step 4：改造 LLM 调用函数
#
# 变化对比：
#   旧版：接收原始 llm，返回 str | None，调用方还要再解析字符串
#   新版：接收 structured_llm，返回 QAList | None，调用方直接用对象
#
# 注意 resp 的处理：
#   旧版需要 resp.content（取原始文本）
#   新版直接 return resp（resp 已经是 QAList 对象，框架自动完成了解析）
def _call_llm_with_retry(
    structured_llm,                  # ★ 类型变了：接收 structured_llm，不再是原始 llm
    prompt: str,
    pool: "_AdaptiveWorkerPool",
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> QAList | None:                  # ★ 返回类型变了：QAList 对象，不再是字符串
    """带指数退避重试的 LLM 调用，感知限流错误并通知 pool 动态调整并发。"""
    for attempt in range(max_retries):
        try:
            resp = structured_llm.invoke(prompt)
            pool.on_success()
            return resp                          # ★ 直接返回对象，不需要 .content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "503" in err_str
            if is_rate_limit:
                pool.on_rate_limited()
                wait = backoff_base ** attempt * 3
            else:
                wait = backoff_base ** attempt

            if attempt < max_retries - 1:
                print(f"    ↻ LLM 调用失败（第 {attempt+1} 次），{wait:.0f}s 后重试：{e}")
                sleep(wait)
            else:
                print(f"    ✗ LLM 调用失败（已重试 {max_retries} 次），放弃：{e}")
    return None


def _process_chunk(
    args: tuple,
) -> tuple[int, list[dict]]:
    """处理单个 chunk 并返回 (chunk_index, records)，供并发调用。"""
    # ★ 注意参数列表：llm → structured_llm（名字改了，含义变了）
    i, chunk, data_dir, n_per_chunk, structured_llm, max_retries, pool = args

    page = chunk.metadata.get("page", 0)
    industry = _get_industry(chunk)
    rel_src = _get_rel_src(chunk, data_dir)

    prompt = _QA_PROMPT.format(
        n=n_per_chunk,
        chunk=chunk.page_content[:1200],
        source=rel_src,
        page=page,
    )

    # ★ 变化：raw 现在是 QAList | None，不再是字符串
    qa_list = _call_llm_with_retry(structured_llm, prompt, pool, max_retries=max_retries)
    if qa_list is None:
        return i, []

    records = [
        {
            "question":     pair.question,       # ★ 属性访问，不再是字典取值
            "ground_truth": pair.ground_truth,   # ★ 属性访问，不再是字典取值
            "source_file":  rel_src,
            "page":         int(page),
            "industry":     industry,
            "chunk_text":   chunk.page_content[:800],
        }
        for pair in qa_list.pairs               # ★ 直接迭代 Pydantic 对象列表
    ]
    return i, records


def generate_qa_pairs(
    chunks: list[Document],
    data_dir: Path,
    n_per_chunk: int,
    llm,
    verbose: bool = True,
    max_workers: int = 4,
    max_retries: int = 3,
) -> list[dict]:
    """
    并发调用 LLM 为每个 chunk 生成 QA 对，返回去重后的完整列表。

    Args:
        max_workers: 并发线程数，受限于 LLM API 速率限制，建议 2-8。
        max_retries: 单个 chunk 最大重试次数。
    """

    structured_llm = llm.with_structured_output(QAList)

    total = len(chunks)
    all_records: list[dict] = []
    completed = 0

    pool = _AdaptiveWorkerPool(initial=max_workers)

    task_args = [
        (i, chunk, data_dir, n_per_chunk, structured_llm, max_retries, pool)
        for i, chunk in enumerate(chunks)
    ]

    def _submit(arg):
        pool.acquire()
        try:
            return _process_chunk(arg)
        finally:
            pool.release()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_submit, arg): arg[0] for arg in task_args}

        for future in as_completed(futures):
            chunk_idx, records = future.result()
            all_records.extend(records)
            completed += 1

            if verbose:
                chunk = chunks[chunk_idx]
                rel_src = _get_rel_src(chunk, data_dir)
                page = chunk.metadata.get("page", 0)
                pct = completed / total * 100
                print(
                    f"  [{completed:>3}/{total}] {rel_src}:p{page}"
                    f"  生成 {len(records)} 条  ({pct:.0f}%)"
                    f"  [并发: {pool.workers}]"
                )

    before = len(all_records)
    all_records = _deduplicate(all_records)
    print(f"[generator] 去重：{before} → {len(all_records)} 条 QA 对")
    return all_records


def save_dataset(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist: dict[str, int] = defaultdict(int)
    for r in records:
        dist[r["industry"]] += 1
    print(f"\n[generator] 数据集已保存：{output_path}  共 {len(records)} 条")
    print("  行业分布：")
    for ind, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {ind:12s} {cnt:>4} 条")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="金融研报 RAG 评测数据集生成器")
    p.add_argument("--data_dir",    default=str(_SRC / "data"),                                    help="文档根目录")
    p.add_argument("--output",      default=str(_SRC / "eval" / "dataset" / "eval_dataset.jsonl"), help="输出文件路径")
    p.add_argument("--n_per_chunk", type=int, default=2,   help="每个 chunk 生成 QA 对数（建议 1-3）")
    p.add_argument("--max_chunks",  type=int, default=150, help="最大采样 chunk 数（控制 API 成本）")
    p.add_argument("--seed",        type=int, default=42,  help="随机种子，保证可复现")
    p.add_argument("--model",       default="deepseek/deepseek-v4-flash",                          help="LLM，格式 provider/model，如 deepseek/deepseek-v4-flash")
    p.add_argument("--max_workers", type=int, default=4,   help="并发 LLM 调用线程数（建议 2-8，视 API 速率限制调整）")
    p.add_argument("--max_retries", type=int, default=3,   help="单个 chunk LLM 调用最大重试次数")
    p.add_argument("--verbose",     action="store_true",   help="打印每条生成进度")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    if not data_dir.exists():
        print(f"[error] data_dir 不存在：{data_dir}")
        sys.exit(1)

    llm = load_chat_model(args.model)
    print(f"[generator] 使用 LLM：{llm}")

    chunks = load_all_chunks(settings.tools.vector_store.CHROMA_DB_PATH)
    sampled = stratified_sample(chunks, args.max_chunks, args.seed)

    records = generate_qa_pairs(
        sampled,
        data_dir,
        args.n_per_chunk,
        llm,                    # main() 只管传原始 llm，structured_llm 的创建在内部
        verbose=args.verbose,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )

    save_dataset(records, output_path)
    print("\n✅ 数据集生成完成")


if __name__ == "__main__":
    main()