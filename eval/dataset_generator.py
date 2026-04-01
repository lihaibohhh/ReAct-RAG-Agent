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
from pathlib import Path
from collections import defaultdict

# ── 把 src/ 加入 path，复用项目已有模块 ──────────────────────────────────────
_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
load_dotenv(_SRC.parent / ".env", override=False)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from react_agent.utils.llm import load_chat_model
from react_agent.core.config import settings

# ── Prompt ────────────────────────────────────────────────────────────────────
_QA_PROMPT = """你是一位苛刻的金融分析师，负责构建用于评估 RAG（检索增强生成）系统的高质量测试集。

请仔细阅读下面的金融研报片段。如果该片段主要包含【免责声明】、【评级标准说明】、【分析师联系方式/执业编号】或【纯粹的文档目录/免责法律条款】，请直接返回空数组 []，不要生成任何问题！

如果片段包含实质性的商业/财务/行业分析内容，请生成 {n} 个高质量问答对。

核心要求：
1. 必须指名道姓：问题中必须明确包含具体的公司名称、行业名称或具体产品名，绝对不能使用“该公司”、“该行业”、“本项目”等模糊指代！如果片段中找不到具体实体名称，请放弃生成或根据常识推断补充。
2. 拒绝元数据：不要提问关于图表编号（如"图1"）、分析师名字、报告日期等外围信息。
3. 聚焦核心业务：多提问关于营收数据、毛利率、产能规划、行业趋势、竞争格局等需要深度理解的问题。
4. 独立可答：ground_truth 必须直接且精准，不需要用户再去翻看原文档。
5. 若片段中存在表格数据，优先生成针对具体数值的问题

━━━ 研报片段 ━━━
{chunk}

来源文件：{source}（第 {page} 页）
所属行业：{industry}
━━━━━━━━━━━━━

请严格以 JSON 数组格式输出，不要包含任何其他文字、注释或 Markdown：
[
  {{"question": "...", "ground_truth": "..."}}
]"""

# ──────────────── 工具函数 ──────────────────────────────────────────────────────────────────


def _infer_industry(file_path: str, data_dir: Path) -> str:
    """从文件路径中推断行业标签（取 data_dir 下的第一级子目录名）。"""
    try:
        rel = Path(file_path).relative_to(data_dir)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return "未知"


def _strip_json_fence(text: str) -> str:
    """去掉 LLM 可能多输出的 ```json ... ``` 包裹。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_qa_response(text: str) -> list[dict]:
    """解析 LLM 返回的 JSON，容忍常见格式错误。"""
    text = _strip_json_fence(text)
    try:
        pairs = json.loads(text)
        if isinstance(pairs, list):
            return [p for p in pairs if "question" in p and "ground_truth" in p]
    except json.JSONDecodeError:
        # 尝试提取第一个 [...] 块
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if m:
            try:
                pairs = json.loads(m.group())
                return [p for p in pairs if "question" in p and "ground_truth" in p]
            except json.JSONDecodeError:
                pass
    return []


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
        from sentence_transformers import SentenceTransformer  # noqa: F401（仅做可用性检查）
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

    # 批量编码；normalize_embeddings=True 直接输出单位向量，省去后续手动归一化
    embeddings: "np.ndarray" = model.encode(
        questions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape: (N, D)

    # 余弦相似度矩阵：cos_sim(i, j) = dot(unit_i, unit_j)
    # 用矩阵乘法一次性计算，避免 O(N²) 的双重 Python for 循环
    sim_matrix: "np.ndarray" = embeddings @ embeddings.T  # shape: (N, N)

    kept: list[int] = []
    for i in range(len(records)):
        if not kept:
            kept.append(i)
            continue
        # 向量切片索引，利用 numpy 广播；max() 为标量操作，极快
        if sim_matrix[i, kept].max() < threshold:
            kept.append(i)

    return [records[i] for i in kept]


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def load_all_chunks(data_dir: Path) -> list[Document]:
    """从本地 Chroma 向量数据库读取全量 chunks，跳过文件解析步骤。"""
    db_path = settings.tools.vector_store.CHROMA_DB_PATH
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
    data_dir: Path,
    max_chunks: int,
    seed: int,
) -> list[Document]:
    """
    按行业分层采样：各行业尽量均匀抽取，总数不超过 max_chunks。
    同时过滤：chunk 文本过短（< 100 字）或疑似纯表格噪声的块不参与采样。
    """
    rng = random.Random(seed)

    # 过滤过短的块
    valid = [
        c for c in chunks
        if len(c.page_content.strip()) >= 100
    ]

    # 按行业分组
    by_industry: dict[str, list[Document]] = defaultdict(list)
    for chunk in valid:
        src = chunk.metadata.get("source", "")
        industry = _infer_industry(src, data_dir)
        by_industry[industry].append(chunk)

    industries = list(by_industry.keys())
    per_industry = max(1, max_chunks // len(industries))

    sampled: list[Document] = []
    for ind, docs in by_industry.items():
        take = min(per_industry, len(docs))
        sampled.extend(rng.sample(docs, take))
        print(f"  行业「{ind}」：{len(docs)} chunks → 抽 {take}")

    # 如果总数仍超过 max_chunks，随机截断
    if len(sampled) > max_chunks:
        rng.shuffle(sampled)
        sampled = sampled[:max_chunks]

    print(f"[generator] 最终采样 {len(sampled)} 个 chunks 用于生成 QA 对")
    return sampled


def generate_qa_pairs(
    chunks: list[Document],
    data_dir: Path,
    n_per_chunk: int,
    llm,
    verbose: bool = True,
) -> list[dict]:
    """逐 chunk 调用 LLM 生成 QA 对，返回去重后的完整列表。"""
    all_records: list[dict] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        industry = _infer_industry(src, data_dir)
        rel_src = str(Path(src).relative_to(data_dir)) if data_dir in Path(src).parents else Path(src).name

        prompt = _QA_PROMPT.format(
            n=n_per_chunk,
            chunk=chunk.page_content[:1200],   # 限制 token 数
            source=rel_src,
            page=page,
            industry=industry,
        )

        try:
            resp = llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            pairs = _parse_qa_response(raw)
        except Exception as e:
            print(f"  ⚠ chunk {i+1}/{total} 生成失败：{e}")
            continue

        for pair in pairs:
            all_records.append({
                "question":    pair["question"],
                "ground_truth": pair["ground_truth"],
                "source_file": rel_src,
                "page":        int(page),
                "industry":    industry,
                "chunk_text":  chunk.page_content[:800],
            })

        if verbose:
            pct = (i + 1) / total * 100
            print(f"  [{i+1:>3}/{total}] {rel_src}:p{page}  生成 {len(pairs)} 条  ({pct:.0f}%)")

    # 全局去重
    before = len(all_records)
    all_records = _deduplicate(all_records)
    print(f"[generator] 去重：{before} → {len(all_records)} 条 QA 对")
    return all_records


def save_dataset(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印行业分布统计
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
    p.add_argument("--data_dir",    default=str(_SRC / "data"),                              help="文档根目录")
    p.add_argument("--output",      default=str(_SRC / "eval" / "dataset" / "eval_dataset.jsonl"), help="输出文件路径")
    p.add_argument("--n_per_chunk", type=int, default=2,   help="每个 chunk 生成 QA 对数（建议 1-3）")
    p.add_argument("--max_chunks",  type=int, default=150, help="最大采样 chunk 数（控制 API 成本）")
    p.add_argument("--seed",        type=int, default=42,  help="随机种子，保证可复现")
    p.add_argument("--model",       default='deepseek/deepseek-chat',        help="LLM，格式 provider/model，如 deepseek/deepseek-chat")
    p.add_argument("--verbose",     action="store_true",   help="打印每条生成进度")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    if not data_dir.exists():
        print(f"[error] data_dir 不存在：{data_dir}")
        sys.exit(1)

    # 加载 LLM（复用项目配置）
    llm = load_chat_model(args.model)
    print(f"[generator] 使用 LLM：{llm}")

    # 加载并采样 chunks
    chunks = load_all_chunks(data_dir)
    sampled = stratified_sample(chunks, data_dir, args.max_chunks, args.seed)

    # 生成 QA 对
    records = generate_qa_pairs(sampled, data_dir, args.n_per_chunk, llm, verbose=args.verbose)

    # 保存
    save_dataset(records, output_path)
    print("\n✅ 数据集生成完成")


if __name__ == "__main__":
    main()