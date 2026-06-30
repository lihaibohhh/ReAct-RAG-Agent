"""
Markdown 文档生成工具 — md_tool

适用场景：分析摘要、会议记录、内部备忘、可在 GitHub/Notion/飞书直接渲染的内容

设计规范：
  - 与项目其他工具（search / rag / sql / excel）保持一致的 _ok / _err 返回结构
  - @tool(description=...) 提供 LLM 路由决策所需的触发/禁止条件
  - sections / metadata 接受 Any 类型，在函数体内通过 _doc_common 手动校验和容错，
    避免 @tool 装饰器的 Pydantic 校验在函数体外抛出 ValidationError
  - 输入标准化 / 容错逻辑统一收敛到 tools/_doc_common.py
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from react_agent.utils.tool_helpers import _ok, _err
from react_agent.tools._doc_common import (
    normalize_sections,
    coerce_metadata,
    build_filename,
)

# ── 输出目录（与 excel_tool / docx_tool 保持同一约定）────────────────────
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 辅助函数：Markdown 表格单元格安全渲染 ────────────────────────────────
def _md_cell(value: Any) -> str:
    """将单元格值转为 Markdown 安全字符串：None→空串，转义管道符。"""
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


# ── 辅助函数：结构化内容 → Markdown 字符串 ──────────────────────────────
def _build_md_content(
        title: str,
        sections: list[dict],
        metadata: dict | None = None,
) -> str:
    """
    将结构化内容渲染为 Markdown 字符串。

    sections 已经过 _doc_common.normalize_sections 净化：
      - level 已是 1~6 的 int
      - bullets 要么是 list 要么是 None
      - table 要么是含 headers/rows 的 dict 要么是 None
    因此这里可以放心按类型直接使用，无需重复防御。
    """
    lines: list[str] = []

    # 文档标题（H1）
    lines.append(f"# {title}\n")

    # 可选元数据块
    if metadata:
        lines.append("---")
        for k, v in metadata.items():
            lines.append(f"**{k}**：{v}")
        lines.append("---\n")

    for sec in sections:
        heading_level = sec.get("level", 2)  # 已被 normalize 保证为 int
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        bullets = sec.get("bullets")
        table = sec.get("table")

        if heading:
            lines.append(f"{'#' * heading_level} {heading}\n")

        if content:
            lines.append(f"{content}\n")

        # 列表渲染（bullets 已保证为 list 或 None）
        if bullets:
            for item in bullets:
                lines.append(f"- {item}")
            lines.append("")

        # 表格渲染（table 已保证为含 headers/rows 的 dict 或 None）
        if table:
            headers = table.get("headers")
            rows = table.get("rows")
            if headers and isinstance(headers, list):
                n_cols = len(headers)
                lines.append("| " + " | ".join(_md_cell(h) for h in headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, (list, tuple)):
                            # 与 docx 端对齐：缺列补空串，多余列丢弃
                            cells = [
                                _md_cell(row[i] if i < len(row) else "")
                                for i in range(n_cols)
                            ]
                            lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 工具：md_tool — Markdown 文档生成
# ═══════════════════════════════════════════════════════════════════════════
@tool(
    description=(
            "【触发条件】以下情况调用本工具：\n"
            "  - 用户要求生成分析摘要、会议纪要、研究备忘、行业速览等文档\n"
            "  - 内容以纯文本、列表、简单表格为主，不需要 Word 格式精确排版\n"
            "  - 目标文档需要在 Notion / 飞书 / GitHub / 前端 Markdown 渲染器中直接使用\n"
            "  - 用户明确要求 .md 格式\n\n"
            "【不触发条件】以下情况禁止调用本工具：\n"
            "  - 用户要求正式对外交付的 Word 文档（应使用 docx_tool）\n"
            "  - 用户要求生成 Excel 表格（应使用 excel_tool）\n"
            "  - 内容仅需口头回答，不需要保存为文件\n\n"
            "【输入】\n"
            "  title    : 文档标题（H1），必填，字符串\n"
            "  sections : 章节列表（list[dict]），每个 dict 可含以下字段：\n"
            "             heading(str) / level(int,默认2) / content(str) / "
            "bullets(list[str]) / table(dict)\n"
            "  filename : 文件名（可选，不含扩展名）\n"
            "  mode     : 'timestamp'（默认，不覆盖）或 'overwrite'\n"
            "  metadata : 可选元数据 dict，如 {'数据来源': '内部研报'}\n\n"
            "【sections 完整示例】\n"
            '  [\n'
            '    {"heading": "一、市场概况", "level": 2, '
            '"content": "2026 年行业整体增长稳健。"},\n'
            '    {"heading": "二、核心数据", "level": 2,\n'
            '     "bullets": ["营收同比 +18%", "毛利率 32%"],\n'
            '     "table": {"headers": ["公司", "毛利率"], '
            '"rows": [["比亚迪", "28%"], ["宁德时代", "25%"]]}}\n'
            '  ]\n\n'
            "【输出】\n"
            "  data.path    — 生成文件的绝对路径\n"
            "  data.size_kb — 文件大小（KB）"
    )
)
def md_tool(
        title: str = "",
        sections: Any = None,
        filename: str = "",
        mode: str = "timestamp",
        metadata: Any = None,
) -> dict[str, Any]:
    """
    生成 Markdown（.md）文档并保存到本地。

    参数：
        title     : 文档标题（同时作为 H1 标题写入文档）
        sections  : 章节列表，每个元素为 dict，支持字段：
                    - heading  (str)       章节标题
                    - level    (int, 默认2) 标题级别，2=H2，3=H3
                    - content  (str)       正文段落
                    - bullets  (list[str]) 无序列表项
                    - table    (dict)      {"headers": [...], "rows": [[...], ...]}
        filename  : 文件名（不含扩展名）。为空时使用 title 生成
        mode      : "timestamp"（默认）附加时间戳避免覆盖 / "overwrite" 直接覆盖
        metadata  : 可选元数据 dict，写在标题下方的信息栏

    返回：
        统一的 _ok / _err 结构
    """
    tool_name = "md_tool"

    # ════════════════════ 阶段 1：输入校验与标准化 ════════════════════
    t = (str(title) if title else "").strip()
    if not t:
        return _err(
            tool_name=tool_name,
            query=t,
            code="BAD_INPUT",
            message="title 不能为空，请提供文档标题。",
        )

    try:
        normalized = normalize_sections(sections)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return _err(
            tool_name=tool_name,
            query=t,
            code="BAD_INPUT",
            message=f"sections 参数格式无法解析：{e}。请传入章节字典列表。",
        )

    if mode not in ("timestamp", "overwrite"):
        mode = "timestamp"  # 容错：非法 mode 默认回退，不拒绝

    metadata = coerce_metadata(metadata)

    # ════════════════════ 阶段 2：构建并写入 ════════════════════
    try:
        name = build_filename(t, filename, mode, ext="md")
        out_path = OUTPUT_DIR / name

        meta = {"生成时间": datetime.now().strftime("%Y-%m-%d %H:%M")}
        if metadata:
            meta.update(metadata)

        content = _build_md_content(t, normalized, meta)
        out_path.write_text(content, encoding="utf-8")

        size_kb = round(out_path.stat().st_size / 1024, 1)
        return _ok(
            tool_name=tool_name,
            query=t,
            data={
                "path": str(out_path.resolve()),
                "size_kb": size_kb,
                "message": f"Markdown 文档已生成：{name}（{size_kb} KB，{len(normalized)} 个章节）",
            },
        )

    except Exception as e:
        return _err(
            tool_name=tool_name,
            query=t,
            code="WRITE_FAILED",
            message=f"md_tool 生成失败：{type(e).__name__}: {e}",
        )
