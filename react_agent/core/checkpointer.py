from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.core.config import settings
from react_agent.memory.context import Context

try:
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncConnectionPool = None
    AsyncPostgresSaver = None


class CheckpointerFactory:
    """
    Checkpointer 工厂（工程级实现）

    设计原则
    --------
    1. 懒初始化锁：asyncio.Lock 在首次调用时创建，绑定到当前 event loop，
       避免 Streamlit 热重载 / FastAPI 多次启动时因 loop 销毁导致的 RuntimeError。

    2. 精确 cache key：在锁外预计算含连接参数的 key（如 "postgres:postgresql://..."），
       锁内做精确 dict.get 查询，杜绝前缀扫描 + 字典无锁迭代的竞态问题。

    3. 统一降级：所有 backend 失败时共享同一个 MemorySaver 实例（"memory" key），
       避免多次降级创建独立实例、thread_id 相同但状态分裂的问题。

    4. 生命周期解耦：context manager / connection pool 通过独立注册表 _lifecycle 管理，
       不给第三方对象打补丁（猴子补丁污染命名空间且脆弱）。

    5. 双 key 缓存：backend 降级后，同时在原始 key 下缓存 fallback 实例，
       防止后续请求重复尝试已知失败的 backend。
    """

    # 实例缓存：key -> checkpointer 实例
    _instances: Dict[str, BaseCheckpointSaver] = {}

    # 生命周期注册表（与 _instances 解耦）：key -> {"cm": ..., "pool": ...}
    _lifecycle: Dict[str, Dict[str, Any]] = {}

    # 懒创建锁：类定义时不创建，首次 _get_lock() 调用时才绑定到当前 loop
    _lock: Optional[asyncio.Lock] = None

    _logger = logging.getLogger(__name__)
    _DEFAULT_SQLITE_PATH = "./.agent_checkpoints.sqlite3"

    # ─────────────────────────── 内部工具 ───────────────────────────

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """
        懒创建锁。

        关键：asyncio.Lock 必须在当前 event loop 中创建，
        若在模块加载时（loop 启动前）或 loop 销毁后创建，会绑定到错误的 loop。
        """
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _log(cls, msg: str, level: str = "info") -> None:
        """统一日志出口，禁止 print()（MCP 环境下 print 会污染 stdout 协议流）。"""
        getattr(cls._logger, level, cls._logger.info)(msg)

    @classmethod
    def _normalize_backend(cls, backend: str) -> str:
        b = (backend or "").strip().lower()
        aliases: Dict[str, str] = {
            "": "memory", "mem": "memory", "memory": "memory",
            "sqlite": "sqlite", "sqlite3": "sqlite", "aiosqlite": "sqlite",
            "postgres": "postgres", "postgresql": "postgres", "pg": "postgres",
            "redis": "redis",
            "none": "none", "off": "none", "false": "none", "0": "none",
        }
        return aliases.get(b, b)

    @classmethod
    def _sqlite_db_path(cls, ctx: Context) -> Path:
        raw = getattr(ctx, "checkpoint_db_path", None) or cls._DEFAULT_SQLITE_PATH
        # expanduser + resolve → 规范化绝对路径，确保 "./a.db" 与 "/abs/a.db" 命中同一实例
        return Path(raw).expanduser().resolve()

    @classmethod
    def _postgres_conn_str(cls) -> str:
        default = "postgresql://agent_user:agent_pass@localhost:5432/agent_db"
        return os.getenv("POSTGRES_DB_URL", default)

    @classmethod
    def _compute_cache_key(cls, backend: str, ctx: Context) -> str:
        """
        在锁外预计算 cache key。

        含连接参数的精确 key 确保：
        - 同一 backend 不同连接目标不会复用同一实例
        - 进入锁后只需 dict.get(key) 一次，无需迭代
        """
        if backend == "sqlite":
            return f"sqlite:{cls._sqlite_db_path(ctx)}"
        if backend == "postgres":
            return f"postgres:{cls._postgres_conn_str()}"
        if backend == "redis":
            try:
                return f"redis:{settings.redis.REDIS_URL}"
            except AttributeError:
                return "redis:redis://localhost:6379"
        return backend  # "memory"

    @classmethod
    def _register_lifecycle(
        cls, key: str, *, cm: Any = None, pool: Any = None
    ) -> None:
        """注册需要在 close_all() 时清理的外部资源（在锁内调用）。"""
        cls._lifecycle[key] = {"cm": cm, "pool": pool}

    # ─────────────────────────── 生命周期管理 ───────────────────────────

    @classmethod
    async def close_all(cls) -> None:
        """
        关闭所有已注册的连接资源，并完全重置类状态。

        注意：重置 _lock = None 须在退出 async with 块之后，
        否则 __aexit__ 会找不到锁对象。
        """
        lock = cls._get_lock()
        async with lock:
            for key, resources in cls._lifecycle.items():
                cm = resources.get("cm")
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                        cls._log(f"[Checkpointer] 已关闭 CM: {key}")
                    except Exception as e:
                        cls._log(f"[Checkpointer] 关闭 CM 失败 ({key}): {e}", "warning")

                pool = resources.get("pool")
                if pool is not None:
                    try:
                        await pool.close()
                        cls._log(f"[Checkpointer] 已关闭连接池: {key}")
                    except Exception as e:
                        cls._log(f"[Checkpointer] 关闭连接池失败 ({key}): {e}", "warning")

            cls._instances.clear()
            cls._lifecycle.clear()

        # 退出 async with 后再重置锁，确保下次在新 loop 中重建
        cls._lock = None

    # ─────────────────────────── 对外入口 ───────────────────────────

    @classmethod
    async def create(cls, ctx: Context) -> Optional[BaseCheckpointSaver]:
        backend = cls._normalize_backend(getattr(ctx, "checkpoint_backend", ""))

        if backend == "none":
            return None

        # 在锁外预计算 key（可能涉及文件系统路径解析 / os.getenv，成本低但避免在锁内做）
        cache_key = cls._compute_cache_key(backend, ctx)

        # 快速路径：无锁单 key 查询（CPython dict.get 是原子操作，无需加锁）
        inst = cls._instances.get(cache_key)
        if inst is not None:
            return inst

        # 慢路径：加锁创建
        async with cls._get_lock():
            # 双重检查（等待锁期间可能已被其他协程创建）
            inst = cls._instances.get(cache_key)
            if inst is not None:
                return inst

            instance, effective_key = await cls._create_instance(backend, ctx, cache_key)

            if instance is not None and effective_key:
                cls._instances[effective_key] = instance
                # 若发生了降级（effective_key != cache_key），同时在原始 key 下缓存，
                # 防止后续请求重复尝试已知失败的 backend（如每次请求都重试超时的 Postgres）
                if effective_key != cache_key:
                    cls._instances[cache_key] = instance

            return instance

    # ─────────────────────────── 各 Backend 创建逻辑 ───────────────────────────

    @classmethod
    async def _create_instance(
        cls, backend: str, ctx: Context, cache_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """分发到具体 backend 创建函数（在锁内调用）。"""
        if backend == "memory":
            return cls._create_memory(), "memory"
        if backend == "sqlite":
            return await cls._create_sqlite(ctx, cache_key)
        if backend == "postgres":
            return await cls._create_postgres(cache_key)
        if backend == "redis":
            return await cls._create_redis(cache_key)
        cls._log(f"[Checkpointer] 未知 backend '{backend}'，禁用持久化", "warning")
        return None, ""

    @classmethod
    def _create_memory(cls) -> Optional[BaseCheckpointSaver]:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            cls._log(
                "[Checkpointer] 使用 MemorySaver（仅限本地调试，"
                "无持久化，高并发存在 OOM 风险，禁止用于生产环境）",
                "warning",
            )
            inst = MemorySaver()
            cls._register_lifecycle("memory")  # 无外部资源，注册空记录以保持注册表完整
            return inst
        except ImportError as e:
            cls._log(f"[Checkpointer] ❌ 无法导入 MemorySaver: {e}", "error")
            return None

    @classmethod
    def _fallback_to_memory(
        cls, reason: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """
        统一降级入口（必须在锁内调用）。

        所有 backend 失败共享同一个 MemorySaver 实例（"memory" key），
        避免多次降级各自创建独立实例导致同一 thread_id 看到不同会话历史。
        """
        cls._log(f"[Checkpointer] ⚠️ {reason}，降级到 MemorySaver", "warning")
        existing = cls._instances.get("memory")
        if existing is not None:
            return existing, "memory"
        inst = cls._create_memory()
        return (inst, "memory") if inst is not None else (None, "")

    @classmethod
    async def _create_sqlite(
        cls, ctx: Context, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 SQLite checkpointer。"""
        db_path = cls._sqlite_db_path(ctx)

        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as e:
            return cls._fallback_to_memory(f"AsyncSqliteSaver 不可用: {e}")

        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cls._log(f"[Checkpointer] 初始化 SQLite: {db_path}")
            cm = AsyncSqliteSaver.from_conn_string(str(db_path))
            checkpointer = await cm.__aenter__()
            # 显式幂等建表，与 Postgres / Redis 行为保持一致（不依赖 __aenter__ 副作用）
            await checkpointer.setup()

            cls._register_lifecycle(ok_key, cm=cm)
            cls._log(f"[Checkpointer] ✅ SQLite 已启用: {db_path}")
            return checkpointer, ok_key

        except Exception as e:
            cls._log(
                f"[Checkpointer] ❌ SQLite 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("SQLite 初始化失败")

    @classmethod
    async def _create_postgres(
        cls, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 PostgreSQL checkpointer。"""
        if AsyncPostgresSaver is None or AsyncConnectionPool is None:
            return cls._fallback_to_memory(
                "缺少依赖：pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )

        conn_str = cls._postgres_conn_str()

        # 从 settings 读连接池配置，允许在 config.py 中统一管理，而非硬编码
        pg_cfg = getattr(settings, "postgres", None)
        max_size: int = getattr(pg_cfg, "POOL_MAX_SIZE", 20) if pg_cfg else 20
        connect_timeout: float = (
            getattr(pg_cfg, "CONNECT_TIMEOUT_SECONDS", 5.0) if pg_cfg else 5.0
        )

        pool = None
        try:
            cls._log("[Checkpointer] 初始化 Postgres 连接池...")
            pool = AsyncConnectionPool(
                conn_str,
                max_size=max_size,
                kwargs={"autocommit": True},
            )
            # 使用 asyncio.wait_for 包裹 pool.open()：
            # 直接调用无超时保护，Postgres 不可达时会永久挂起整个进程。
            # wait_for 兼容所有 psycopg_pool 版本，无需依赖 open(timeout=...) 参数。
            await asyncio.wait_for(pool.open(wait=True), timeout=connect_timeout)

            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            cls._register_lifecycle(ok_key, pool=pool)
            cls._log("[Checkpointer] ✅ Postgres 已启用")
            return checkpointer, ok_key

        except asyncio.TimeoutError:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            return cls._fallback_to_memory(
                f"Postgres 连接池初始化超时（{connect_timeout}s），请检查 POSTGRES_DB_URL 和网络连通性"
            )
        except Exception as e:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    pass
            cls._log(
                f"[Checkpointer] ❌ Postgres 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("Postgres 初始化失败")

    @classmethod
    async def _create_redis(
        cls, ok_key: str
    ) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 Redis checkpointer。"""
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        except ImportError:
            return cls._fallback_to_memory(
                "缺少依赖：pip install langgraph-checkpoint-redis"
            )

        try:
            # 放在 try 块内：AttributeError（settings 结构变化）同样会被捕获并降级
            redis_url: str = settings.redis.REDIS_URL
        except AttributeError as e:
            return cls._fallback_to_memory(f"settings.redis.REDIS_URL 不可访问: {e}")

        try:
            cls._log(f"[Checkpointer] 初始化 Redis: {redis_url}")
            # 与 SQLite 保持一致：使用 context manager 协议管理连接生命周期
            cm = AsyncRedisSaver.from_conn_string(redis_url)
            checkpointer = await cm.__aenter__()
            await checkpointer.setup()

            # 注册 cm 以便 close_all() 统一关闭（同 SQLite 路径）
            cls._register_lifecycle(ok_key, cm=cm)
            cls._log("[Checkpointer] ✅ Redis 已启用")
            return checkpointer, ok_key

        except Exception as e:
            cls._log(
                f"[Checkpointer] ❌ Redis 初始化失败: {type(e).__name__}: {e}", "error"
            )
            return cls._fallback_to_memory("Redis 初始化失败")