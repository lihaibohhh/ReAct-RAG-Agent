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

    if ext in _PDF_PRECHUNKED_EXTS:
        doc = load_file(file_path)
        if not doc:
            return []

        n_text = sum(1 for d in doc if d.metadata.get("doc_type")=="text")
        n_table = sum(1 for d in doc if d.metadata.get("doc_type")=="table")
        print(f"已加载{n_text}个文本块；已加载{n_table}个表格块")
        return doc

    if ext in {".xlsx", ".xls", ".csv"}:
        doc = load_table_file(file_path)
        if not doc:
            return []
        return doc
    else:
        doc = load_file(file_path)
        if not doc:
            return []
        split_tool = get_splitter(file_path)
        chunks = split_tool.split_documents(doc)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{i+1}个chunk && {filename}"
        return chunks