# 职责：分块策略选择 + 文件统一处理入口
# 拆分自 ingest_docs.py —— 函数 3、4
#
# 变更记录：
#   - 新增 PDF 专用分支：pdf_parser 已完成分块，此处直接返回，不再二次 split
#     （二次 split 会破坏表格 Markdown 结构，且 chunk_id 元数据已由 pdf_parser 注入）
#   - 其余分支（docx / txt / md / py / xlsx / csv）保持不变

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loaders import load_table_file, load_file

# PDF 由 pdf_parser 预分块，chunker 识别此标记后直接透传
_PDF_PRECHUNKED_EXTS = {".pdf"}


def get_splitter(file_path: str):
    """语法感知分块：Python/Markdown 按语法边界，其他按字符"""
    from langchain_text_splitters import Language
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=500, chunk_overlap=50)
    elif ext in (".md", ".markdown"):
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=500, chunk_overlap=50)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def process_file(file_path: str):
    """
    根据文件类型自动选择处理策略，返回 Document 列表。

    各类型处理路径：
      .xlsx / .xls / .csv  → load_table_file()  逐行转文本，不分块
      .pdf                 → load_file()         版面解析 + 预分块（pdf_parser）
      其他                 → load_file() + splitter  常规加载 + 分块
    """
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    # ── 路径 A：表格文件 ────────────────────────────────────────────────
    if ext in (".xlsx", ".xls", ".csv"):
        docs = load_table_file(file_path)
        print(f"      （表格模式：{len(docs)} 行）")
        return docs

    # ── 路径 B：PDF —— 已由 pdf_parser 完成版面分析 + 分块，直接透传 ──
    if ext in _PDF_PRECHUNKED_EXTS:
        docs = load_file(file_path)  # 内部调用 load_pdf_with_layout
        if not docs:
            return []
        # pdf_parser 已注入 chunk_id，此处统计并打印摘要
        n_text = sum(1 for d in docs if d.metadata.get("doc_type") == "text")
        n_table = sum(1 for d in docs if d.metadata.get("doc_type") == "table")
        print(f"（文字块 {n_text} 个 / 表格块 {n_table} 个）")
        return docs

    # ── 路径 C：普通文档 —— 加载后再分块 ──────────────────────────────
    docs = load_file(file_path)
    if not docs:
        return []
    splitter = get_splitter(file_path)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{filename}::chunk_{i}"
    return chunks
