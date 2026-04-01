# react_agent/rag/vector_store.py
# 职责：文件哈希管理 + 向量化写入 Chroma（增量更新）
# 拆分自 ingest_docs.py —— 函数 5、6、7

import json
import os
import hashlib
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .chunker import process_file

from react_agent.core.config import settings
from react_agent.rag.retriever import invalidate_bm25_cache
from react_agent.utils.timer_logger import timer, summarize_last_run

HASH_RECORD_PATH = settings.tools.vector_store.HASH_RECORD_PATH


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_hash_record() -> dict:
    if Path(HASH_RECORD_PATH).exists():
        with open(HASH_RECORD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_record(record: dict):
    Path(HASH_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def build_vector_db(data_dir: str = settings.tools.vector_store.data_dir):
    """扫描目录下所有支持文件，增量写入 Chroma 向量数据库"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据目录不存在：{data_dir}")
        return

    supported_exts = {".pdf", ".docx", ".txt", ".csv", ".md",
                      ".markdown", ".xlsx", ".xls", ".py"}
    all_files = [f for f in data_path.rglob("*")
                 if f.is_file() and f.suffix.lower() in supported_exts]

    if not all_files:
        print(f"❌ 在 {data_dir} 下没有找到任何支持的文件！")
        return

    print(f"📂 共发现 {len(all_files)} 个文件，开始增量检查...")
    hash_record = load_hash_record()
    new_files = []

    for f in all_files:
        file_hash = get_file_hash(str(f))
        if hash_record.get(str(f)) == file_hash:
            print(f"   ⏭️  跳过（未变化）：{f.name}")
        else:
            print(f"   🆕 待处理：{f.name}")
            new_files.append((f, file_hash))

    if not new_files:
        print("✅ 所有文件均已是最新，无需更新！")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # T1 + T2：逐文件解析 & 分块
    # 说明：process_file() 内部先调用 pdf_parser（T1）再调用 chunker（T2），
    #       两者深度耦合，这里在文件粒度做整体计时，并记录产出块数。
    #       如需进一步拆分 T1/T2，需在 chunker.py / pdf_parser.py 内部埋点。
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n📄 [T1+T2] 解析 & 分块 — {len(new_files)} 个新/变更文件")
    all_chunks, processed = [], {}

    for file_path, file_hash in new_files:
        ext = file_path.suffix.lower()
        print(f"   ▶ {file_path.name}", end="  ")
        try:
            with timer(
                    "T1+T2_parse_chunk",
                    meta={"file": file_path.name, "ext": ext},
            ):
                chunks = process_file(str(file_path))

            if chunks:
                all_chunks.extend(chunks)
                processed[str(file_path)] = file_hash
                if ext not in (".xlsx", ".xls", ".csv"):
                    print(f"（{len(chunks)} 个块）")
            else:
                print("（无内容，已跳过）")
        except Exception as e:
            print(f"（❌ 处理失败：{type(e).__name__}: {e}）")

    if not all_chunks:
        print("❌ 没有成功加载任何内容。")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # T3_model_load：embedding 模型首次加载
    # 说明：HuggingFaceEmbeddings 第一次初始化会从磁盘/HuggingFace Hub 加载模型，
    #       耗时 1~10s，单独记录便于对比"冷启动"与"热启动"差异。
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n🔧 [T3_model] 加载 embedding 模型...")
    chroma_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    with timer("T3_model_load", meta={"model": model_name}):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # T3_embed + T4_index：向量化 & 写入 Chroma
    # 做法：先手动调用 embed_documents() 拿到向量（纯 T3），
    #       再把原始文本+向量一起写入 Chroma（纯 T4），从而分开计时。
    # ─────────────────────────────────────────────────────────────────────────
    total_chunks = len(all_chunks)
    batch_size = 5000
    n_batches = (total_chunks + batch_size - 1) // batch_size

    print(f"\n🔢 [T3+T4] 共 {total_chunks} 个块，分 {n_batches} 批处理...")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_chunks)
        batch_chunks = all_chunks[start:end]
        batch_meta = {
            "batch": batch_idx + 1,
            "of": n_batches,
            "chunk_count": len(batch_chunks),
        }

        # T3：纯向量化（CPU/GPU 运算，无磁盘 I/O）
        print(f"   ⏳ 批次 {batch_idx + 1}/{n_batches}：向量化中...", end="  ")
        with timer("T3_embedding", meta=batch_meta):
            texts = [doc.page_content for doc in batch_chunks]
            vectors = embeddings.embed_documents(texts)

        # T4：写入 Chroma（磁盘 I/O）
        # add_embeddings() 接受预计算好的向量，跳过内部重复 embed
        print(f"   ⏳ 批次 {batch_idx + 1}/{n_batches}：写入 Chroma...", end="  ")
        with timer("T4_chroma_write", meta=batch_meta):
            vectorstore._collection.add(
                ids=[
                    # Chroma 要求唯一 ID；优先用 chunk_id metadata，没有则 fallback
                    doc.metadata.get("chunk_id", f"chunk_{start + i}")
                    for i, doc in enumerate(batch_chunks)
                ],
                embeddings=vectors,
                documents=texts,
                metadatas=[doc.metadata for doc in batch_chunks],
            )

        done = min(end, total_chunks)
        print(f"   ✅ 入库进度：{done}/{total_chunks}")

    hash_record.update(processed)
    save_hash_record(hash_record)
    print(f"\n✅ 完成！新增 {total_chunks} 个数据块到知识库。")

    # ─────────────────────────────────────────────────────────────────────────
    # T4_bm25_reset：重置 BM25 索引缓存（Redis 写入）
    # ─────────────────────────────────────────────────────────────────────────
    print("🔄 [T4_BM25] 重置检索器缓存...")
    with timer("T4_bm25_reset"):
        invalidate_bm25_cache()
    print("✅ 知识库已更新，检索器缓存已重置")

    # ─────────────────────────────────────────────────────────────────────────
    # 打印本次建库完整计时汇总
    # ─────────────────────────────────────────────────────────────────────────
    summarize_last_run()


if __name__ == "__main__":
    build_vector_db()
