# react_agent/rag/pdf_parser.py
# 职责：金融研报级 PDF 解析
#   - PyMuPDF  → 带坐标文字块提取、多栏重排、页眉页脚过滤
#   - pdfplumber → 结构化表格提取（转 Markdown，整块保留）
#   - 降级兜底  → 两库均缺失时自动回落到 PyPDFLoader
#
# 对外接口（供 loaders.py 调用）：
#   load_pdf_with_layout(file_path, chunk_size, chunk_overlap)
#       -> List[Document]   已完成分块，含 metadata
#
# 依赖安装：
#   pip install pymupdf pdfplumber
#
# 作者备注：
#   表格 Document 不再二次分块（chunk_id 含 ::table_ 前缀），
#   调用方 chunker.py 对 PDF 应直接返回本模块输出，不要再 split。

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────
# 内部常量
# ──────────────────────────────────────────────
_HEADER_FOOTER_RATIO = 0.07   # 页面顶部 / 底部各 7% 视为页眉页脚
_COL_GAP_RATIO = 0.12   # 双栏之间空白带宽度阈值（相对页宽）
_TABLE_OVERLAP_THRESH = 0.4   # 文字块与表格区域 IoU 超过此值则视为表格内文字
_MIN_BLOCK_CHARS = 3      # 过滤掉字符数极少的噪声块（如单个页码）


# ──────────────────────────────────────────────
# 依赖检测
# ──────────────────────────────────────────────
def _check_deps() -> Tuple[bool, bool]:
    """返回 (has_fitz, has_pdfplumber)"""
    try:
        import fitz  # noqa: F401
        has_fitz = True
    except ImportError:
        has_fitz = False

    try:
        import pdfplumber  # noqa: F401
        has_pdfplumber = True
    except ImportError:
        has_pdfplumber = False

    return has_fitz, has_pdfplumber


# ──────────────────────────────────────────────
# 表格工具
# ──────────────────────────────────────────────
def _table_to_markdown(table: list) -> str:
    """
    将 pdfplumber 返回的二维列表转为 Markdown 表格字符串。
    自动处理 None 单元格和合并列情况。
    """
    if not table or not table[0]:
        return ""

    def _clean(cell) -> str:
        if cell is None:
            return ""
        return re.sub(r"\s+", " ", str(cell)).strip()

    header = table[0]
    col_count = len(header)
    rows = table[1:]

    # 修复行宽不一致（pdfplumber 偶发）
    def _pad(row: list) -> list:
        row = list(row)
        if len(row) < col_count:
            row += [""] * (col_count - len(row))
        return row[:col_count]

    md_lines = []
    md_lines.append("| " + " | ".join(_clean(c) for c in header) + " |")
    md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(_clean(c) for c in _pad(row)) + " |")

    return "\n".join(md_lines)


def _get_table_bboxes(plumber_page) -> List[Tuple[float, float, float, float]]:
    """从 pdfplumber page 获取所有表格的 bbox (x0, y0, x1, y1)。"""
    bboxes = []
    try:
        for table in plumber_page.find_tables():
            b = table.bbox   # (x0, top, x1, bottom)  —— pdfplumber 坐标
            bboxes.append((b[0], b[1], b[2], b[3]))
    except Exception:
        pass
    return bboxes


def _overlap_ratio(b1: tuple, b2: tuple) -> float:
    """计算两个 bbox 的交集面积 / b1 面积（用于判断 b1 是否在 b2 内部）。"""
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_b1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    return inter / area_b1


def _is_in_table(block_bbox: tuple,
                 table_bboxes: List[tuple],
                 thresh: float = _TABLE_OVERLAP_THRESH) -> bool:
    return any(_overlap_ratio(block_bbox, tb) >= thresh for tb in table_bboxes)


# ──────────────────────────────────────────────
# 版面分析工具
# ──────────────────────────────────────────────
def _filter_header_footer(
    blocks: list,
    page_height: float,
    margin_ratio: float = _HEADER_FOOTER_RATIO,
) -> list:
    """过滤页眉（顶部）、页脚（底部）及字符数极少的噪声块。"""
    top_limit = page_height * margin_ratio
    bot_limit = page_height * (1 - margin_ratio)

    kept = []
    for b in blocks:
        y0, y1 = b["bbox"][1], b["bbox"][3]
        text = b.get("text", "").strip()
        if y0 < top_limit or y1 > bot_limit:
            continue
        if len(text) < _MIN_BLOCK_CHARS:
            continue
        kept.append(b)
    return kept


def _detect_and_sort_columns(blocks: list, page_width: float) -> list:
    """
    检测单栏 / 双栏布局，并按正确阅读顺序重排 blocks。

    双栏判定逻辑：
      1. 统计所有块的 x 起点，若左右两侧均有块且中间存在明显空白带
         （宽度 > page_width * _COL_GAP_RATIO），则视为双栏。
      2. 双栏时：先按 y 排序左栏，再按 y 排序右栏，左栏内容在前。
    """
    if not blocks:
        return blocks

    mid = page_width / 2
    left_blocks = [b for b in blocks if b["bbox"][0] < mid]
    right_blocks = [b for b in blocks if b["bbox"][0] >= mid]

    # 双栏判断：两侧均有内容，且中间有明显空白带
    is_two_col = False
    if left_blocks and right_blocks:
        left_x1_max = max(b["bbox"][2] for b in left_blocks)
        right_x0_min = min(b["bbox"][0] for b in right_blocks)
        gap = right_x0_min - left_x1_max
        if gap > page_width * _COL_GAP_RATIO:
            is_two_col = True

    if is_two_col:
        # 各栏内按 y 坐标（行号）排序
        left_sorted = sorted(left_blocks,  key=lambda b: b["bbox"][1])
        right_sorted = sorted(right_blocks, key=lambda b: b["bbox"][1])
        return left_sorted + right_sorted
    else:
        # 单栏：整体按 y 排序即可
        return sorted(blocks, key=lambda b: b["bbox"][1])


# ──────────────────────────────────────────────
# PyMuPDF 文字提取
# ──────────────────────────────────────────────
def _extract_text_blocks_fitz(fitz_page, table_bboxes: list) -> List[dict]:
    """
    用 PyMuPDF 提取文字块，返回 [{text, bbox}, ...] 列表。
    已排除落在表格区域内的块（由 pdfplumber 单独处理）。

    注意：pdfplumber 的 y 轴原点在页面顶部（PDF 标准：底部），
    fitz 的 bbox 也是顶部原点，两者一致，可直接比较。
    """
    import fitz

    raw_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    result = []
    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:          # type=1 是图片块，跳过
            continue
        bbox = tuple(block["bbox"])          # (x0, y0, x1, y1)
        if _is_in_table(bbox, table_bboxes):
            continue                         # 表格内文字交给 pdfplumber

        lines_text = []
        for line in block.get("lines", []):
            line_str = "".join(s["text"] for s in line.get("spans", []))
            lines_text.append(line_str)
        text = "\n".join(lines_text).strip()

        if len(text) >= _MIN_BLOCK_CHARS:
            result.append({"text": text, "bbox": bbox})
    return result


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def load_pdf_with_layout(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    金融研报级 PDF 解析主函数。

    处理流程（每页）：
      1. pdfplumber  → 提取表格 bbox + 表格内容（转 Markdown Document）
      2. PyMuPDF     → 提取带坐标文字块，排除表格区域
      3. 版面分析    → 过滤页眉页脚、多栏重排
      4. 文字分块    → RecursiveCharacterTextSplitter（中文友好分隔符）
      5. 汇总        → 文字 chunk + 表格 Document 一并返回

    降级策略：
      - 仅缺 pdfplumber：跳过表格提取，继续文字解析（发出警告）
      - 仅缺 PyMuPDF  ：不应发生（load_file 已检测，此处兜底）
      - 两者均缺      ：回落到 PyPDFLoader（发出明显警告）

    Returns:
        List[Document]，已完成分块，metadata 包含：
          source, page, chunk_id, doc_type ("text" | "table")
    """
    has_fitz, has_pdfplumber = _check_deps()
    filename = Path(file_path).name

    # ── 降级：两者均缺 ──────────────────────────────────────────
    if not has_fitz:
        warnings.warn(
            f"⚠️  [pdf_parser] PyMuPDF 未安装，'{filename}' 回落到 PyPDFLoader。\n"
            "    多栏/表格解析将失效，建议：pip install pymupdf pdfplumber",
            RuntimeWarning,
            stacklevel=2,
        )
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()

    if not has_pdfplumber:
        warnings.warn(
            f"⚠️  [pdf_parser] pdfplumber 未安装，'{filename}' 将跳过表格提取。\n"
            "    建议：pip install pdfplumber",
            RuntimeWarning,
            stacklevel=2,
        )

    import fitz

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文研报常见语义边界：段落 > 句号 > 分号 > 逗号 > 换行
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    all_docs: List[Document] = []
    fitz_doc = fitz.open(file_path)

    # 按需打开 pdfplumber（避免重复 open）
    plumber_ctx = None
    if has_pdfplumber:
        import pdfplumber
        try:
            plumber_ctx = pdfplumber.open(file_path)
        except Exception as e:
            warnings.warn(
                f"⚠️  [pdf_parser] pdfplumber 打开 '{filename}' 失败，跳过表格提取：{e}",
                RuntimeWarning,
                stacklevel=2,
            )
            plumber_ctx = None  # 降级：后续逻辑自动跳过表格

    try:
        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            page_w = fitz_page.rect.width
            page_h = fitz_page.rect.height

            # ── Step 1：表格区域 & 表格内容（pdfplumber） ────────────
            table_bboxes: List[tuple] = []
            if plumber_ctx is not None:
                try:
                    plumber_page = plumber_ctx.pages[page_num]
                    table_bboxes = _get_table_bboxes(plumber_page)

                    for t_idx, raw_table in enumerate(plumber_page.extract_tables() or []):
                        md = _table_to_markdown(raw_table)
                        if not md.strip():
                            continue
                        chunk_id = f"{filename}::page_{page_num}::table_{t_idx}"
                        all_docs.append(Document(
                            page_content=md,
                            metadata={
                                "source":   file_path,
                                "page":     page_num,
                                "chunk_id": chunk_id,
                                "doc_type": "table",
                            },
                        ))
                except Exception as e:
                    # 当前页表格提取失败，降级：清空 bbox，PyMuPDF 正文提取不受影响
                    warnings.warn(
                        f"⚠️  [pdf_parser] '{filename}' 第 {page_num} 页表格提取失败，"
                        f"跳过本页表格（正文仍正常处理）：{e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    table_bboxes = []

            # ── Step 2：文字块提取（PyMuPDF，排除表格区域） ──────────
            raw_blocks = _extract_text_blocks_fitz(fitz_page, table_bboxes)

            # ── Step 3：版面分析 ──────────────────────────────────────
            blocks = _filter_header_footer(raw_blocks, page_h)
            blocks = _detect_and_sort_columns(blocks, page_w)

            if not blocks:
                continue

            # ── Step 4：合并文字 → 分块 ───────────────────────────────
            page_text = "\n".join(b["text"] for b in blocks).strip()
            if not page_text:
                continue

            text_chunks = splitter.split_text(page_text)
            for c_idx, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_id = f"{filename}::page_{page_num}::chunk_{c_idx}"
                all_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source":   file_path,
                        "page":     page_num,
                        "chunk_id": chunk_id,
                        "doc_type": "text",
                    },
                ))

    finally:
        fitz_doc.close()
        if plumber_ctx is not None:
            plumber_ctx.close()

    if not all_docs:
        warnings.warn(
            f"⚠️  [pdf_parser] '{filename}' 解析后无任何内容，请检查文件是否为图片型 PDF。",
            RuntimeWarning,
            stacklevel=2,
        )

    return all_docs