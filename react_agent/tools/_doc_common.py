"""
文档生成工具公共逻辑 — _doc_common

被 markdown.py（md_tool）与 make_docx.py（docx_tool）共享。
将来若新增 pptx_tool / pdf_tool 等文档类工具，也可复用本模块。

下划线前缀表示这是 tools 包的内部实现细节，不对外暴露为工具。

职责边界：
  - 只放「文档生成工具专属」的输入标准化与容错逻辑
  - 不放 _ok / _err / with_retry 这类全工具通用基础设施（那些在 utils/tool_helpers.py）

包含三个工具函数：
  - normalize_sections : 将 LLM 传入的多形态 sections 参数统一为 list[dict]
  - coerce_metadata    : 将 metadata 容错为 dict 或 None
  - build_filename     : 按 timestamp / overwrite 模式生成安全文件名（不含目录）
"""

from __future__ import annotations

import re
import json
from datetime import datetime
from typing import Any


# ── sections 标准化 ──────────────────────────────────────────────────────
def normalize_sections(raw: Any) -> list[dict]:
    """
    将 LLM 可能传来的多种格式统一转换为 list[dict]。

    DeepSeek 等模型在生成工具参数时，sections 可能是：
      - list[dict]    → 直接使用（标准格式）
      - str (JSON)    → json.loads 后递归处理
      - str (非JSON)  → 当作单段纯文本章节
      - dict          → 当作单个章节，包成列表
      - list[str]     → 每个字符串当作一个纯文本章节
      - list[list]    → 每个子列表尝试转成 {heading, content}

    同时对每个 dict 章节内部做字段净化（见 _sanitize_section）：
      - level 强制转 int（修复 "2" 字符串导致的 TypeError）
      - 非 dict 元素（None / 数字等）安全降级为纯文本章节

    转换失败时抛 ValueError，由调用方捕获后返回 _err。
    """
    # 字符串：尝试 JSON 解析
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise ValueError("sections 为空字符串")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # 不是 JSON，当作单段纯文本
            return [_sanitize_section({"heading": "", "content": raw})]
        # json.loads("42") → int, "true" → bool 等标量：保留原始字符串作纯文本
        if not isinstance(parsed, (dict, list)):
            return [_sanitize_section({"heading": "", "content": raw})]
        raw = parsed

    # 单个 dict：包成列表
    if isinstance(raw, dict):
        return [_sanitize_section(raw)]

    # 非列表类型：无法处理
    if not isinstance(raw, list):
        raise ValueError(f"sections 类型不支持：{type(raw).__name__}，需要列表")

    # 空列表
    if len(raw) == 0:
        raise ValueError("sections 为空列表")

    # 逐项标准化
    result: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(_sanitize_section(item))
        elif isinstance(item, str):
            result.append({"heading": "", "content": item})
        elif isinstance(item, list):
            if len(item) >= 2:
                result.append({"heading": str(item[0]), "content": str(item[1])})
            elif len(item) == 1:
                result.append({"heading": "", "content": str(item[0])})
            # 空子列表：跳过
        elif item is None:
            continue  # null 元素直接丢弃，不产生空章节
        else:
            result.append({"heading": "", "content": str(item)})

    if not result:
        raise ValueError("sections 标准化后为空（所有元素均无效）")

    return result


def _sanitize_section(sec: dict) -> dict:
    """
    净化单个章节 dict 的字段，保证下游渲染函数拿到的类型是安全的。

    处理项：
      - 别名   : body→content / items→bullets / title→heading（仅当标准名缺失时）
      - level   : 强制 int，范围钳制到 1~6，无法转换则回退 2
      - heading : None → ""，其余强制 str
      - content : None → ""，其余强制 str
      - bullets : 非 list 时丢弃（设为 None）；list 内过滤 None 元素
      - table   : 结构不完整时丢弃（设为 None）

    注意：这里只做「类型安全」净化，不做「内容渲染」，渲染交给各工具自己的 build 函数。
    """
    out = dict(sec)  # 浅拷贝，避免修改 LLM 原始入参

    # ── 字段别名：LLM 常见的替代命名 → 标准名 ──
    _ALIASES = {"body": "content", "items": "bullets", "title": "heading"}
    for alias, canonical in _ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out.pop(alias)

    # level：字符串数字 "2" → 2，非法值 → 2
    raw_level = out.get("level", 2)
    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        level = 2
    out["level"] = max(1, min(level, 6))  # 钳制到合法标题层级

    # heading / content：None → ""，其余强制 str
    out["heading"] = str(out["heading"]) if out.get("heading") else ""
    out["content"] = str(out["content"]) if out.get("content") else ""

    # bullets：必须是 list，否则丢弃；列表内过滤 None 元素
    bullets = out.get("bullets")
    if bullets is not None and not isinstance(bullets, list):
        out["bullets"] = None
    elif isinstance(bullets, list):
        out["bullets"] = [b for b in bullets if b is not None]

    # table：必须是含 headers / rows 的 dict，否则丢弃
    table = out.get("table")
    if table is not None:
        if not (isinstance(table, dict) and "headers" in table and "rows" in table):
            out["table"] = None

    return out


# ── metadata 容错 ────────────────────────────────────────────────────────
def coerce_metadata(metadata: Any) -> dict | None:
    """
    将 LLM 传入的 metadata 容错为 dict 或 None。

    - dict        → 原样返回
    - str (JSON)  → 解析成功且为 dict 则返回，否则 None
    - 其他类型    → None
    """
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ── 文件名生成 ────────────────────────────────────────────────────────────
def build_filename(title: str, filename: str, mode: str, ext: str) -> str:
    """
    生成安全的输出文件名（不含目录路径）。

    参数：
      title    : 文档标题，filename 为空时用它派生文件名
      filename : 用户指定的文件名（不含扩展名），优先级高于 title
      mode     : "timestamp" 追加时间戳避免覆盖 / "overwrite" 直接用基名
      ext      : 扩展名，不含点，如 "md" / "docx"

    文件名净化：保留中日韩文、英文、数字、下划线，其余字符替换为 _，并截断到 40 字符。
    """
    base = (filename or "").strip()
    if not base:
        base = re.sub(r"[^\w\u3000-\u9fff\uac00-\ud7af]", "_", title)[:40]
    # 即便用户传了 filename，也做一次净化，防止路径穿越（../）等问题
    base = re.sub(r"[^\w\u3000-\u9fff\uac00-\ud7af]", "_", base)[:40] or "untitled"

    if mode == "timestamp":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}.{ext}"
    return f"{base}.{ext}"