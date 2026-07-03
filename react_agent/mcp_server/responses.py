"""
responses.py:
    统一 JSON 返回结构，比如 mcp_ok() / mcp_err()
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping, Sequence


def mcp_ok(
        data: Any = None,
        meta: Any = None,
) -> dict[str, Any]:
    """
    MCP 工具成功返回。

    统一结构：
    {
        "ok": true,
        "data": {},
        "error": null,
        "meta": {}
    }
    """
    return {
        "ok": True,
        "data": ensure_jsonable(data if data is not None else {}),
        "error": None,
        "meta": ensure_jsonable(meta or {}),
    }


def mcp_err(
        message: str,
        error_type: str = "Error",
        data: Any = None,
        meta: Any = None,
) -> dict[str, Any]:
    """
    MCP 工具失败返回的统一结构。

    注意：业务失败不等于 MCP 调用失败。
    例如 query 为空时，MCP 工具本身执行成功，
    但业务结果是 ok=false。
    """
    return {
        "ok": False,
        "data": ensure_jsonable(data),
        "error": {
            "type": str(error_type),
            "message": str(message),
        },
        "meta": ensure_jsonable(meta or {})
    }


def clamp_int(
        value: Any,
        default: int,
        min_value: int,
        max_value: int,
) -> int:
    """
    把 MCP 客户端传来的整数参数限制在安全范围内。

    例如 top_k 允许 1~10，避免模型误传 1000 导致返回内容过大。
    """
    try:
        ivalue = int(value)
    except Exception:
        ivalue = int(default)

    if ivalue < min_value:
        return min_value

    if ivalue > max_value:
        return max_value

    return ivalue


def ensure_jsonable(value: Any) -> Any:
    """
    将常见 Python / NumPy / Path / datetime / dataclass 对象转换为 JSON 可序列化对象。

    目的：
    - 避免 MCP Server 返回中出现 numpy.float32 / numpy.int64 / Path 等对象
    - 防止客户端因为 JSON 序列化失败而表现为连接断开或工具调用失败
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return ensure_jsonable(asdict(value))

        # 兼容 numpy scalar：np.float32 / np.float64 / np.int64 等
    if hasattr(value, "item") and callable(value.item):
        try:
            return ensure_jsonable(value.item())
        except Exception:
            pass

        # 兼容 pydantic v2
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return ensure_jsonable(value.model_dump())
        except Exception:
            pass

        # 兼容 pydantic v1
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return ensure_jsonable(value.dict())
        except Exception:
            pass

    if isinstance(value, Mapping):
        return {
            str(ensure_jsonable(key)): ensure_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [ensure_jsonable(item) for item in value]

    if isinstance(value, set):
        return [ensure_jsonable(item) for item in value]

        # 兜底：未知对象转字符串，保证 MCP JSON 序列化不炸
    return str(value)
