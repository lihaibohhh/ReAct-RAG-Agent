from __future__ import annotations
from datetime import datetime


def _now_iso_in_tz(tz_name: str) -> str:
    """返回指定 IANA 时区的 ISO 时间字符串。"""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = None
    return datetime.now(tz=tz).isoformat()