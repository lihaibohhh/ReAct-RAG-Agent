from __future__ import annotations
import time
import json
import asyncio
import random
import inspect
import functools
from typing import Any, Optional, Dict, Type, Iterable
from pydantic import BaseModel, Field


# ================================================================================
# 统一返回结构：所有工具都返回这个结构，便于上游统一处理
# ================================================================================
def _ok(*, tool_name: str, query: str, data: Any, meta: Optional[Dict[str, Any]] = None):
    """
    返回成功结果。

    参数：
      tool_name (str)：工具名称，用于日志和追踪
      query (str)：原始查询字符串，便于问题定位
      data (Any)：实际返回的数据（search 返回结果、Excel 保存路径等）
      meta (dict)：元数据，如重试次数、耗时、是否缓存等（可选）

    返回：
      {
        “ok”: True,                 # 标志成功
        “tool”: tool_name,          # 工具标识
        “query”: query,             # 原始查询
        “data”: data,               # 核心数据
        “error”: None,              # 成功时无错误
        “meta”: {...}               # 执行元数据
      }
    """
    return {
        "ok": True,
        "tool": tool_name,
        "query": query,
        "data": data,
        "error": None,
        "meta": meta or {}
    }


def _err(*, tool_name: str, query: str, message: str, code: str = "TOOL_ERROR", meta: Optional[Dict[str, Any]] = None):
    """
    返回失败结果。

    参数：
      tool_name (str)：工具名称
      query (str)：原始查询字符串
      message (str)：人类可读的错误描述
      code (str)：错误代码（如 'BAD_INPUT', 'MISSING_DEPENDENCY', 'TIMEOUT'）
      meta (dict)：元数据，如已尝试次数等（可选）

    返回：
      {
        “ok”: False,                # 标志失败
        “tool”: tool_name,
        “query”: query,
        “data”: None,               # 失败时无数据
        “error”: {
          “code”: code,             # 错误类型代码
          “message”: message        # 错误详细说明
        },
        “meta”: {...}
      }
    """
    return {
        "ok": False,
        "tool": tool_name,
        "query": query,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": meta or {}
    }


def _trim_text(s: str, max_chars: int):
    """
    字符串裁剪工具。

    功能：将过长的字符串截断，并在末尾添加”...”作为截断标记。
    用途：限制搜索结果、知识库内容等的大小，避免超长文本污染上下文。

    参数：
      s (str)：原始字符串，会先做 strip() 处理
      max_chars (int)：最大允许的字符数

    返回：
      - 如果 len(s) <= max_chars：返回原字符串（strip 后）
      - 否则：返回截断后的字符串 + “...” 作为截断标记

    示例：
      _trim_text(“Hello World”, 5) → “He...”  # 截断保留 5 字符
      _trim_text(“Hi”, 5) → “Hi”              # 不足 5 字符，原样返回
    """
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."  # 预留 3 位空间给”...”


# ================================================================================
# 通用工具装饰器：超时 + 指数退避重试 + 异步/同步双支持
# ================================================================================
def with_retry(
        *,
        tool_name: str,
        max_retries: int = 3,
        timeout: float = 10.0,
        base_delay: float = 0.5,
        max_delay: float = 6.0,
        retry_on: Iterable[Type[BaseException]] = (TimeoutError, ConnectionError, OSError)
):
    """
    为工具函数增强：超时控制 + 智能重试 + 指数退避策略。

    核心功能：
      1. 超时保护：如果工具调用超过 timeout 秒，立即中止
      2. 智能重试：对指定的异常类型（如 TimeoutError、ConnectionError）进行重试
      3. 指数退避：重试间隔逐次增加（0.5s → 1s → 2s → ...），避免冲击被调用方
      4. 异步/同步双支持：自动检测被装饰函数是异步还是同步
      5. 事件循环兼容性：处理 "已有事件循环" 场景（LangGraph/FastAPI 中）
      6. 统一返回结构：失败也返回 {ok: False, error: {...}, meta: {...}}

    参数：
      tool_name (str)：工具名称，用于错误日志和返回结构
      max_retries (int)：最多重试几次（默认 3 次，即最多执行 4 次）
      timeout (float)：单次调用超时时间，单位秒（默认 10 秒）
      base_delay (float)：首次重试延迟基数，单位秒（默认 0.5 秒）
      max_delay (float)：重试延迟上限，单位秒（默认 6 秒，防止等待时间过长）
      retry_on (Iterable)：哪些异常类型触发重试
                           （默认：TimeoutError, ConnectionError, OSError）
                           注意：工具函数需要主动 raise 这些异常，
                           不能在内部 try-except 中吞掉

    执行流程：
      (1) 第 1 次尝试（attempt=0）：
          - 调用工具函数，加 timeout 限制
          - 如果成功返回：记录 attempt/timeout/elapsed 到 meta，返回结果
          - 如果异常：捕获异常，判断是否需要重试

      (2) 异常判断：
          - 如果异常在 retry_on 列表中：进入重试流程
          - 否则：立即返回失败（不再尝试），因为非临时错误

      (3) 重试延迟：
          - 计算延迟时间：min(max_delay, base_delay * 2^attempt) * 随机偏移 (0.8~1.2)
          - 等待后继续第 2 次尝试

      (4) 重复 (1)~(3)，最多执行 max_retries + 1 次

      (5) 全部失败：返回 {ok: False, error: {...}, meta: {retries: 实际尝试次数}}

    使用示例：
      @with_retry(tool_name="my_api", max_retries=2, timeout=5)
      async def fetch_data(url: str) -> dict:
          # 函数内容
          # 关键：对于超时/连接错误，需要主动 raise，不能吞掉！
          try:
              result = await asyncio.wait_for(client.get(url), timeout=3)
          except asyncio.TimeoutError:
              raise  # 让 with_retry 捕获并重试

    注意事项：
      - 工具必须返回 dict，且应包含 "meta" 字段
      - 对临时错误（超时、连接拒绝）需要主动 raise，不能在工具内处理
      - 对永久错误（参数错误、依赖缺失）不应该重试，直接返回 _err(...)
      - 指数退避可以避免多个重试工具同时冲击后端，提高成功率
    """
    def decorator(fn):
        is_async = inspect.iscoroutinefunction(fn)

        # 预计算，避免每次循环都重建 tuple（性能优化）
        _retry_on_tuple = tuple(retry_on) if retry_on else ()

        async def _call_async(*args, **kwargs):
            """
            异步执行函数，带重试和超时保护。
            """
            last_exc: BaseException | None = None
            q = ""
            # 尽量从 kwargs/args 提取 query 方便定位问题所在
            if "query" in kwargs:
                q = str(kwargs.get("query") or "").strip()
            elif args:
                q = str(args[0] or "").strip()

            # 重试循环
            for attempt in range(max_retries + 1):
                try:
                    start = time.time()
                    # 异步和同步函数的调用方式不同
                    if is_async:
                        # 异步函数：直接 await，加 timeout 限制
                        res = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
                    else:
                        # 同步函数：用 asyncio.to_thread 在线程池中执行，避免阻塞事件循环
                        res = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=timeout)

                    # 返回值检查：必须是 dict，且包含 meta 字段
                    if not isinstance(res, dict):
                        return _err(
                            tool_name=tool_name,
                            query=q,
                            code="BAD_TOOL_RETURN",
                            message=f"工具返回类型不是 dict，而是 {type(res).__name__}",
                            meta={"attempt": attempt, "timeout": timeout}
                        )
                    # 确保返回结构中有 meta，并注入执行元数据
                    res.setdefault("meta", {})
                    res["meta"].update(
                        {
                            "attempt": attempt,
                            "timeout": timeout,
                            "elapsed": round(time.time() - start, 3)
                        }
                    )
                    return json.dumps(res, ensure_ascii=False)
                except Exception as e:
                    last_exc = e
                    # 非指定异常（如参数错误）：立即返回，不再重试
                    if _retry_on_tuple and not isinstance(e, _retry_on_tuple):
                        break

                # 需要重试则等待一段时间后继续
                if attempt < max_retries:
                    # 指数退避：延迟时间 = min(max_delay, base_delay * 2^attempt) * 随机偏移
                    # 随机偏移 (0.8 + 0.4*random) 在 [0.8, 1.2] 之间，避免雷群效应
                    delay = min(max_delay, base_delay * (2 ** attempt)) * (0.8 + 0.4 * random.random())
                    await asyncio.sleep(delay)

            # 全部失败：返回统一的失败结构，记录实际尝试次数
            return json.dumps(_err(
                tool_name=tool_name,
                query=q,
                code="TOOL_FAILED",
                message=f"工具调用失败（已重试）：{type(last_exc).__name__}: {last_exc}",
                meta={"retries": attempt, "timeout": timeout},
            ), ensure_ascii=False)

        # 根据原函数是否异步，返回不同的包装函数
        if is_async:
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                """异步包装器：直接调用 _call_async"""
                return await _call_async(*args, **kwargs)
            return wrapper
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                """
                同步包装器：需要处理两种场景：
                1. 无事件循环：直接用 asyncio.run() 启动新循环
                2. 已有事件循环（如 LangGraph/FastAPI）：不能用 asyncio.run()，改用线程池
                """
                # 兼容"已有事件循环"的运行环境（LangGraph / FastAPI 等）
                # asyncio.run() 在已运行的事件循环中会抛 RuntimeError，须改为新线程
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    # 当前线程已有事件循环：在新线程中启动事件循环运行 _call_async
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        return executor.submit(asyncio.run, _call_async(*args, **kwargs)).result()
                # 当前线程无事件循环：创建新循环运行 _call_async
                return asyncio.run(_call_async(*args, **kwargs))
            return wrapper

    return decorator


def _shrink_search_results(raw: Any, *, max_items: int = 5, max_char_per_item: int = 800):
    """
    搜索结果裁剪器。

    功能：把 Tavily API 返回的大块数据压缩到适合放入对话上下文的大小。
    用途：Web 搜索工具的后处理，过滤和裁剪结果以保持低成本。

    处理逻辑：
      1. 处理 None 输入 → 返回 None
      2. 处理标准 Tavily 格式 {“results”: [...], “answer”: “...”}
         - 取前 max_items 条结果
         - 每条结果裁剪标题（200 字）、内容（max_char_per_item 字）
         - 保留 score 和 published_date 作为参考
      3. 处理字符串格式 → 直接裁剪到 3000 字
      4. 其他格式 → 原样返回

    参数：
      raw (Any)：Tavily 原始返回或其他格式
      max_items (int)：最多保留几条搜索结果（默认 5）
      max_char_per_item (int)：每条结果内容的最大字符数（默认 800）

    返回：
      {
        “results”: [
          {
            “title”: str,            # 搜索结果标题（最多 200 字）
            “url”: str,              # 来源链接
            “content”: str,          # 摘要文本（最多 max_char_per_item 字）
            “score”: float,          # 匹配分数（可能为 None）
            “published_date”: str    # 发布日期（可能为 None）
          },
          ...
        ],
        “answer”: str                # Tavily 直接生成的简洁回答（最多 1200 字）
      }

    兼容性：
      - 不同版本 Tavily 可能返回不同字段，此函数使用 .get() 做容错处理
    """
    if raw is None:
        return None

    # 常见结构：{“results”:[{“title”:..., “url”:..., “content”:...}, ...], ...}
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        results = []
        for item in raw["results"][:max_items]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": _trim_text(str(item.get("title", "")), 200),
                    "url": str(item.get("url", "")),
                    "content": _trim_text(str(item.get("content", "")), max_char_per_item),
                    # 有些版本还有 score / published_date 等字段，不强依赖
                    "score": item.get("score"),
                    "published_date": item.get("published_date") or item.get("published_time"),
                }
            )
        # 保留少量 meta（避免太大）
        return {
            "results": results,
            "answer": _trim_text(str(raw.get("answer", "")), 1200),  # raw 在此分支必为 dict
        }

    # 如果不是预期结构，原样返回，但做一次字符串裁剪（保险）
    if isinstance(raw, str):
        return _trim_text(raw, 3000)

    return raw


class ExcelInput(BaseModel):
    """
    Excel 表格创建工具的输入参数 Schema（Pydantic 模型）。

    属性：
      filename (str)：输出文件名，不需要后缀 .xlsx
                      示例："report" → 保存为 report.xlsx 或 report_timestamp.xlsx
      headers (list[str])：表头列表
                           示例：["姓名", "部门", "薪资"]
                           不能为空
      rows (list[list])：数据行，每行是一个列表
                         示例：[["张三", "技术部", "15000"], ["李四", "销售部", "12000"]]
                         必须与 headers 列数相同
      sheet_name (str)：工作表名称，默认 "Sheet1"（不能超过 31 字符，Excel 限制）
      mode (str)：写入模式，默认 "timestamp"
                  - "timestamp"：添加时间戳前缀，避免覆盖（report → report_20260310_144200.xlsx）
                  - "overwrite"：覆盖已有文件（危险，会丢失旧数据）
                  - "append"：追加到已有文件的最后（要求表头一致，否则拒绝）
    """
    filename: str = Field(description="文件名，例如 'report'，不需要加 .xlsx 后缀")
    headers: list[str] = Field(description="表头列表，例如 ['姓名', '部门', '薪资']")
    rows: list[list] = Field(description="数据行，每行是一个列表，顺序与表头一致")
    sheet_name: str = Field(default="Sheet1", description="工作表名称")
    mode: str = Field(default="timestamp", description="写入模式：timestamp/overwrite/append")
