# react_agent/rag/vector_store.py
# 职责：文件哈希管理 + 向量化写入 Chroma（增量更新）
# 拆分自 ingest_docs.py —— 函数 5、6、7
#
# 变更记录：
#   - T1+T2 改为多进程并行（Pool.imap_unordered）+ 流水线写库
#     流水线逻辑：哪个文件先解析完就先进 buffer，buffer 攒够 batch_size
#     立即触发 T3+T4，不等剩余文件，峰值内存始终控制在一个 batch 以内。
#   - _parse_one() 必须定义在模块顶层（Windows spawn 模式要求可 pickle）
#   - T3/T4 保持串行不变（embedding 批量已高效；Chroma/SQLite 不支持并发写）
#
# [FIX-1] T4 写入改用 upsert：幂等写入，防止重跑/崩溃后 duplicate ID 报错
# [FIX-2] 哈希记录改为每批写完后立即落盘：Chroma 写入与哈希记录严格顺序一致，
#         崩溃重启后只会补写"未记录"的 chunk，永远不会遗漏，也不会误认为"已完成"

import json
import os
import hashlib
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from react_agent.rag.chunker import process_file

from react_agent.core.config import settings
from react_agent.rag.retriever import invalidate_bm25_cache
from react_agent.utils.timer_logger import timer, summarize_last_run
from react_agent.utils.embedder import get_embedder

HASH_RECORD_PATH = settings.tools.vector_store.HASH_RECORD_PATH


# ──────────────────────────────────────────────────────────────────────────────
# 必须定义在模块顶层：Windows multiprocessing 使用 spawn 模式，
# 子进程重新 import 本模块，定义在 if __name__ 或函数内部的函数无法被 pickle。
# ──────────────────────────────────────────────────────────────────────────────
def _parse_one(args: tuple) -> tuple[str, str, list, str | None]:
    """
    单文件解析 & 分块，运行在独立子进程中。

    Args:
        args: (file_path_str, file_hash, ext)

    Returns:
        (file_path_str, file_hash, chunks, error_msg, elapsed_sec)
        - 成功时 error_msg=None
        - 失败时 chunks=[]
        - elapsed_sec：子进程内实测的 T1+T2 耗时（秒），用于 CPU 累计时统计。
          使用 time.perf_counter 而非主进程 timer，原因：
          timer 工具的状态存在主进程内存中，子进程（spawn 模式）得到的是
          主进程 import 时的空状态副本，写入后随子进程退出即消失，
          主进程永远看不到。perf_counter 是纯本地计算，pickle 安全。
    """
    file_path_str, file_hash, ext = args
    t0 = time.perf_counter()
    try:
        chunks = process_file(file_path_str)
        elapsed = time.perf_counter() - t0
        return file_path_str, file_hash, chunks or [], None, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return file_path_str, file_hash, [], f"{type(e).__name__}: {e}", elapsed


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具：T3 + T4 的封装，供流水线循环复用
# ──────────────────────────────────────────────────────────────────────────────
def _embed_and_write(
    batch_chunks: list,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: Chroma,
    batch_label: str,
    global_offset: int,
) -> None:
    """
    对一批 chunks 执行向量化（T3）并写入 Chroma（T4）。

    [FIX-1] 写入使用 upsert 而非 add：
      - add()  遇到已存在的 chunk_id 会抛 DuplicateIDError（chromadb ≥ 0.4.x），
               导致整个批次失败，哈希记录无法更新，下次重跑无限循环。
      - upsert() 语义为"存在则覆盖，不存在则插入"，天然幂等。
        重跑场景下同一批数据写两遍，结果与只写一遍完全等价，不引入脏数据。
      - 注：_collection 仍是内部 API；如 langchain-chroma 未来暴露公共 upsert
        接口，应迁移到公共 API。

    Args:
        batch_chunks:   本批次 Document 列表
        embeddings:     已初始化的 embedding 模型
        vectorstore:    已打开的 Chroma 实例
        batch_label:    日志用批次标签，如 "批次#3"
        global_offset:  本批次第一个 chunk 的全局序号，用于 fallback chunk_id
    """
    batch_meta = {"batch_label": batch_label, "chunk_count": len(batch_chunks)}

    # T3：纯向量化（CPU/GPU 运算）
    print(f"   ⏳ {batch_label}：向量化 {len(batch_chunks)} 个块...", end="  ")
    with timer("T3_embedding", meta=batch_meta):
        texts = [doc.page_content for doc in batch_chunks]
        vectors = embeddings.embed_documents(texts)
    print("完成")

    # T4：写入 Chroma（磁盘 I/O，SQLite 单写，不可并发）
    # [FIX-1] add → upsert，保证幂等性
    print(f"   ⏳ {batch_label}：写入 Chroma（upsert）...", end="  ")
    with timer("T4_chroma_write", meta=batch_meta):
        vectorstore._collection.upsert(          # ← FIX-1: add() → upsert()
            ids=[
                doc.metadata.get("chunk_id", f"chunk_{global_offset + i}")
                for i, doc in enumerate(batch_chunks)
            ],
            embeddings=vectors,
            documents=texts,
            metadatas=[doc.metadata for doc in batch_chunks],
        )
    print("完成")


# ──────────────────────────────────────────────────────────────────────────────
# 哈希工具
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
def build_vector_db(data_dir: str = settings.tools.vector_store.data_dir):
    """
    扫描目录下所有支持文件，增量写入 Chroma 向量数据库。

    执行流程：
      扫描 → 哈希过滤 → [T1+T2 多进程并行解析分块]
                              ↓ imap_unordered 实时返回
                         buffer 攒满 batch_size
                              ↓
                        [T3 向量化] → [T4 upsert Chroma]  ← 幂等写入 [FIX-1]
                              ↓
                        立即持久化本批哈希记录             ← 批内原子 [FIX-2]
      → 重置 BM25 缓存 → 打印计时汇总

    [FIX-2] 崩溃一致性保证：
      Chroma upsert 成功 → 立即 save_hash_record → 下次跳过该文件
      若在 upsert 后、save 前崩溃：哈希未记录 → 下次重跑同一批数据
      由于 [FIX-1] upsert 幂等，重跑结果与首次写入完全等价，不产生脏数据。
      整体保证：Chroma 里有的数据，哈希记录里一定也有（可能滞后半批）；
               哈希记录里有的，Chroma 里一定有。两侧不会出现"一边有一边没有"。
    """
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
    # T3_model_load：提前加载 embedding 模型
    # 放在进程池启动之前，避免子进程也尝试加载模型（子进程只做解析，不需要 embedding）
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n🔧 [T3_model] 加载 embedding 模型...")
    chroma_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    with timer("T3_model_load", meta={"model": model_name}):
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},  # bge 系列官方推荐
        )
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # T1+T2（多进程）+ T3+T4（流水线）
    #
    # 内存控制原理：
    #   imap_unordered 每完成一个文件就立即把结果交回主进程，
    #   主进程将 chunks 放入 buffer；buffer 达到 batch_size 就立刻
    #   触发向量化+写库并清空 buffer，任意时刻内存中只有约一个 batch 的数据。
    #
    # workers 设为 4：PyMuPDF+pdfplumber 每进程约占 300~500MB，
    #   4 进程峰值约 2GB，在普通开发机上安全。可按实际内存酌情调整。
    # ─────────────────────────────────────────────────────────────────────────
    batch_size = 5000
    workers = min(4, cpu_count(), len(new_files))
    task_args = [(str(f), h, f.suffix.lower()) for f, h in new_files]

    print(f"\n📄 [T1+T2 → T3+T4] 流水线启动")
    print(f"   进程池大小：{workers}  |  写库批大小：{batch_size}")
    print(f"   待处理文件：{len(new_files)} 个\n")

    buffer: list = []   # 已解析但尚未向量化的 chunks
    total_written: int = 0    # 已写入 Chroma 的 chunk 总数
    batch_counter: int = 0    # 写库批次计数，仅用于日志标签
    files_done: int = 0    # 已完成解析的文件计数\
    cpu_parse_total: float = 0.0  # 各子进程 T1+T2 耗时之和（CPU 累计时）

    # [FIX-2] 移除原来的 processed: dict。
    # hash_record 在整个函数中始终是"已持久化状态"的内存镜像，
    # 每批 upsert 成功后立即更新并落盘，不再做事后统一合并。

    def _flush_batch(batch_chunks: list, batch_done: dict) -> None:
        """
        封装"写 Chroma + 落盘哈希"两步，保证两者顺序严格一致。

        [FIX-2] 设计要点：
          - _embed_and_write 抛出异常 → 本函数向上传播 → hash_record 不更新
            → 崩溃重启后该批文件重新处理（upsert 幂等，无副作用）
          - _embed_and_write 成功 → 立即更新 hash_record 并落盘
            → 即使随后主进程崩溃，下次重启时这批文件会被正确跳过
          调用方无需 try/except；失败时整条流水线中止，符合快速失败原则。

        Args:
            batch_chunks:  本批次所有 Document
            batch_done:    本批次"已全部写入"的文件路径 → hash 映射
        """
        nonlocal total_written, batch_counter
        batch_counter += 1

        _embed_and_write(                        # ← 若抛出，hash 不会更新
            batch_chunks, embeddings, vectorstore,
            batch_label=f"批次#{batch_counter}",
            global_offset=total_written,
        )

        # Chroma upsert 成功后，立即持久化本批文件的哈希 [FIX-2]
        # 仅更新"本批已完整写完"的文件，跨批拆分的大文件不在此记录
        # （其最后一片所在批次才会记录，见大文件拆分说明）
        hash_record.update(batch_done)
        save_hash_record(hash_record)

        total_written += len(batch_chunks)
        print(f"   📦 累计已写入：{total_written} 个块（哈希已同步落盘）\n")

    wall_t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        for file_path_str, file_hash, chunks, err, elapsed in pool.imap_unordered(
            _parse_one, task_args
        ):
            files_done += 1
            cpu_parse_total += elapsed
            fname = Path(file_path_str).name
            ext = Path(file_path_str).suffix.lower()

            # ── 解析结果处理 ──────────────────────────────────────────────
            if err:
                print(f"   ❌ [{files_done}/{len(new_files)}] {fname}（解析失败：{err}）")
                continue

            if not chunks:
                print(f"   ⚠️  [{files_done}/{len(new_files)}] {fname}（无内容，已跳过）")
                continue

            if ext not in (".xlsx", ".xls", ".csv"):
                print(f"   ✅ [{files_done}/{len(new_files)}] {fname}（{len(chunks)} 个块）")
            else:
                print(f"   ✅ [{files_done}/{len(new_files)}] {fname}")

            buffer.append((file_path_str, file_hash, chunks))

            # ── 流水线触发：buffer 内 chunk 总数超过 batch_size 就立即写库 ──
            buffer_chunk_count = sum(len(c) for _, _, c in buffer)
            while buffer_chunk_count >= batch_size:
                batch_chunks: list = []
                new_buffer: list = []
                # [FIX-2] batch_done 替代原来的全局 processed，作用域收紧到单批次
                batch_done: dict = {}
                taken: int = 0

                for fp, fh, cks in buffer:
                    if taken >= batch_size:
                        new_buffer.append((fp, fh, cks))
                        continue
                    remaining = batch_size - taken
                    if len(cks) <= remaining:
                        # 该文件的全部 chunks 都进入本批，本批写完即可记录哈希
                        batch_chunks.extend(cks)
                        taken += len(cks)
                        batch_done[fp] = fh
                    else:
                        # 大文件：本批只取一部分，剩余留回 buffer。
                        # 该文件的哈希不在本批记录，等其最后一片所在批次再记录。
                        # 这样哈希记录永远对应"该文件已完整写入 Chroma"的状态。
                        batch_chunks.extend(cks[:remaining])
                        new_buffer.append((fp, fh, cks[remaining:]))
                        taken += remaining

                buffer = new_buffer
                buffer_chunk_count = sum(len(c) for _, _, c in buffer)

                _flush_batch(batch_chunks, batch_done)   # ← [FIX-1] + [FIX-2]

        # ── 尾部：处理 buffer 中不足一个 batch 的剩余 chunks ──────────────
        if buffer:
            tail_chunks = [c for _, _, cks in buffer for c in cks]
            # 尾部 buffer 里的所有文件（包括被拆分大文件的最后一片）全部写完
            tail_done = {fp: fh for fp, fh, _ in buffer}

            print(f"\n🔚 处理尾部剩余 {len(tail_chunks)} 个块...")
            _flush_batch(tail_chunks, tail_done)         # ← [FIX-1] + [FIX-2]

    wall_elapsed = time.perf_counter() - wall_t0
    parallel_ratio = round(cpu_parse_total / wall_elapsed, 2) if wall_elapsed > 0 else 0
    print(
        f"\n⏱  [T1+T2 解析统计]\n"
        f"   挂钟耗时（wall）  : {wall_elapsed:.2f}s\n"
        f"   CPU累计耗时      : {cpu_parse_total:.2f}s  "
        f"← 与旧版串行 T1+T2 直接可比\n"
        f"   并行效率         : ×{parallel_ratio}  "
        f"（{workers} 进程，理论上限 ×{workers}.0）"
    )

    if total_written == 0:
        print("❌ 没有成功加载任何内容。")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 收尾：重置 BM25 缓存 + 计时汇总
    # [FIX-2] hash_record 已在每批次写完后实时落盘，此处无需再 save_hash_record
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n✅ 完成！本次新增 {total_written} 个数据块到知识库。")

    print("🔄 [T4_BM25] 重置检索器缓存...")
    with timer("T4_bm25_reset"):
        invalidate_bm25_cache()
    print("✅ 知识库已更新，检索器缓存已重置")

    summarize_last_run()


if __name__ == "__main__":
    build_vector_db()