# react_agent/rag/loaders.py
# 职责：文件读取（表格 + 普通文档）
# 拆分自 ingest_docs.py —— 函数 1、2

import os
from pathlib import Path



def load_table_file(file_path: str):
    """Excel / CSV 专用加载器：逐行转自然语言，不做分块"""
    from langchain_core.documents import Document
    ext = Path(file_path).suffix.lower()
    try:
        import pandas as pd
    except ImportError:
        print("   ⚠️ 缺少 pandas，请执行：pip install pandas")
        return []
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)
            df_list = []
            for sheet_name, sheet_df in sheets.items():
                sheet_df["__sheet__"] = sheet_name
                df_list.append(sheet_df)
            import pandas as pd2
            df = pd2.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"   ⚠️ 表格文件读取失败 {file_path}：{e}")
        return []

    docs = []
    headers = [str(h) for h in df.columns.tolist()]
    for idx, row in df.iterrows():
        parts = []
        for h, v in zip(headers, row.values):
            if h == "__sheet__":
                continue
            import pandas as _pd
            if _pd.notna(v) and str(v).strip():
                parts.append(f"{h}：{v}")
        if not parts:
            continue
        row_text = "；".join(parts)
        docs.append(Document(
            page_content=row_text,
            metadata={
                "source": str(file_path),
                "row": int(idx),
                "chunk_id": f"{Path(file_path).name}::row_{int(idx)}"
            }
        ))
    return docs


def load_file(file_path: str):
    """根据文件后缀自动选择合适的 Loader（非表格类）"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from .pdf_parser import load_pdf_with_layout
        return load_pdf_with_layout(file_path)
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()
    elif ext in (".txt", ".md", ".markdown", ".py"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8").load()
    else:
        print(f"   ⚠️ 暂不支持的文件类型，已跳过：{file_path}")
        return []
