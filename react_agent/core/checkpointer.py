from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from langgraph.checkpoint.base import BaseCheckpointSaver
from react_agent.memory.context import Context
from react_agent.core.config import settings

try:
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncConnectionPool = None
    AsyncPostgresSaver = None


# ==================== Checkpointer 工厂（工程级实现）====================
class CheckpointerFactory:
    """
    Checkpointer 工厂类

    职责：
    1. 延迟初始化（避免模块加载时的异步问题）
    2. 自动降级（SQLite 失败自动回退到 MemorySaver）
    3. 单例模式（避免多次初始化）
    4. 线程安全
    """

    _instances: Dict[str, BaseCheckpointSaver] = {}
    _lock = asyncio.Lock()
    _logger = logging.getLogger(__name__)

    _DEFAULT_SQLITE_PATH = "./.agent_checkpoints.sqlite3"

    # -----------------------------
    # 基础工具：日志/归一化/路径
    # -----------------------------
    @classmethod
    def _log(cls, msg: str):
        # 保持你当前的排查体验：print + logger 双写
        try:
            cls._logger.info(msg)
        except Exception:
            pass
        print(msg)

    @classmethod
    def _normalize_backend(cls, backend: str) -> str:
        b = (backend or "").strip().lower()
        if not b:
            return "memory"
        if b in {"mem", "memory"}:
            return "memory"
        if b in {"sqlite", "sqlite3", "aiosqlite"}:
            return "sqlite"
        if b in {"postgres", "postgresql", "pg"}:
            return "postgres"
        return b

    @classmethod
    def _sqlite_db_path(cls, ctx: Context) -> Path:
        raw = getattr(ctx, "checkpoint_db_path", None) or cls._DEFAULT_SQLITE_PATH
        # expanduser + resolve -> 规范化成绝对路径，避免 ./a.db 与 /abs/a.db 多实例
        return Path(raw).expanduser().resolve()

    @classmethod
    def _sqlite_keys(cls, db_path: Path) -> Tuple[str, str]:
        # 主 key：真正 sqlite 成功时缓存
        ok_key = f"sqlite: {db_path}"
        # fallback key：sqlite 失败时缓存 memory（但不占用 sqlite key）
        fb_key = f"sqlite_fallback: {db_path}"
        return ok_key, fb_key

    @classmethod
    def _new_memory_saver(cls) -> Optional[BaseCheckpointSaver]:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            # FIX-2: 明确警告 MemorySaver 不可用于生产环境
            cls._logger.warning(
                "[Checkpointer] 正在使用 MemorySaver：仅用于本地调试，"
                "生产环境禁止使用，无清理机制，高并发下存在 OOM 风险。"
            )
            cls._log("[Checkpointer] 使用 MemorySaver（内存模式，进程内有效）")
            return MemorySaver()
        except ImportError as e:
            cls._log(f"[Checkpointer] 警告：无法导入 MemorySaver: {e}")
            return None

    # -----------------------------
    # 生命周期 (修改：增加关闭 Postgres 连接池的逻辑)
    # -----------------------------
    @classmethod
    async def close_all(cls):
        """关闭所有连接，包括 SQLite 和 Postgres 连接池"""
        async with cls._lock:
            for key, inst in cls._instances.items():
                # 关闭 SQLite 的 context manager
                cm = getattr(inst, "_context_manager", None)
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass

                # ✅ 关闭 Postgres 的 connection pool
                pool = getattr(inst, "_connection_pool", None)
                if pool is not None:
                    try:
                        await pool.close()
                        cls._log(f"[Checkpointer] 已关闭 Postgres 连接池: {key}")
                    except Exception as e:
                        cls._log(f"[Checkpointer] 关闭连接池失败: {e}")
            cls._instances.clear()

    # -----------------------------
    # 对外入口：create（接口不变）
    # -----------------------------
    @classmethod
    async def create(cls, ctx: Context) -> Optional[BaseCheckpointSaver]:
        backend_raw = getattr(ctx, "checkpoint_backend", "")
        backend = cls._normalize_backend(backend_raw)

        if backend in {"none", "off", "false", "0"}:
            return None

        # 缓存键策略
        requested_key = backend  # 默认
        if backend == "sqlite":
            db_path = cls._sqlite_db_path(ctx)
            ok_key, fb_key = cls._sqlite_keys(db_path)
            # 快速查找
            inst = cls._instances.get(ok_key) or cls._instances.get(fb_key)
            if inst: return inst
        elif backend == "postgres":
            # PG 的 key 在 _create_postgres 里生成，这里先占位
            pass
        else:
            inst = cls._instances.get(backend)
            if inst: return inst

        async with cls._lock:
            # 双重检查
            if backend == "sqlite":
                db_path = cls._sqlite_db_path(ctx)
                ok_key, fb_key = cls._sqlite_keys(db_path)
                inst = cls._instances.get(ok_key) or cls._instances.get(fb_key)
                if inst:
                    return inst
            elif backend == "postgres":
                # PG 的 key 在 _create_postgres 内部生成；用前缀扫描做粗判断
                for cached_key, cached_inst in cls._instances.items():
                    if cached_key.startswith("postgres:") or cached_key.startswith("pg_fallback:"):
                        return cached_inst
            else:
                inst = cls._instances.get(backend)
                if inst:
                    return inst

            # 创建新实例
            instance, effective_key = await cls._create_instance(backend, ctx)

            if instance is not None and effective_key:
                cls._instances[effective_key] = instance

            return instance

    @classmethod
    async def _create_instance(cls, backend: str, ctx: Context) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """实际创建 checkpointer 实例"""

        # Memory 模式
        if backend == "memory":
            inst = cls._new_memory_saver()
            return inst, "memory" if inst is not None else ""

        # SQLite 模式
        if backend == "sqlite":
            return await cls._create_sqlite(ctx)

        # PostgreSQL 模式（未来扩展）
        if backend == "postgres":
            # cls._log("[Checkpointer] PostgreSQL 模式暂未实现，回退到 MemorySaver")
            # inst = cls._new_memory_saver()
            # 不要污染 memory 主 key，给一个 pg_fallback key（可选，但更干净）
            return await cls._create_postgres(ctx)

        if backend == "redis":
            return await cls._create_redis(ctx)

        cls._log(f"[Checkpointer] 警告：未知的 backend '{backend}'，禁用持久化")
        return None, ""

    @classmethod
    async def _create_sqlite(cls, ctx: Context) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """
        创建 SQLite checkpointer（适配 3.0.3 版本）
        """
        db_path = cls._sqlite_db_path(ctx)
        ok_key, fb_key = cls._sqlite_keys(db_path)

        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as e:
            cls._log(f"[Checkpointer] ❌ 无法导入 AsyncSqliteSaver: {e}")
            cls._log("[Checkpointer] 回退到 MemorySaver（注意：本次将缓存到 sqlite_fallback key，避免污染 sqlite key）")
            inst = cls._new_memory_saver()
            return inst, fb_key if inst is not None else ""

        # 确保目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # ✅ Windows 上不要使用 sqlite:/// 前缀；直接用路径字符串
            conn_string = str(db_path)
            cls._log(f"[Checkpointer] 初始化 SQLite (3.0.3): {db_path}")

            context_manager = AsyncSqliteSaver.from_conn_string(conn_string)
            checkpointer = await context_manager.__aenter__()

            # 存储上下文管理器引用，以便后续清理
            checkpointer._context_manager = context_manager

            cls._log(f"[Checkpointer] ✅ SQLite 已启用: {db_path}")
            return checkpointer, ok_key

        except Exception as e:
            cls._log(f"[Checkpointer] ❌ SQLite 初始化失败: {type(e).__name__}: {e}")
            cls._log("[Checkpointer] 回退到 MemorySaver（注意：本次将缓存到 sqlite_fallback key，避免污染 sqlite key）")
            inst = cls._new_memory_saver()
            return inst, fb_key if inst is not None else ""

    @classmethod
    async def _create_postgres(cls, ctx: Context) -> Tuple[Optional[BaseCheckpointSaver], str]:
        """创建 Postgres checkpointer"""
        if AsyncPostgresSaver is None:
            cls._log("[Checkpointer] ❌ 缺少依赖：请安装 langgraph-checkpoint-postgres psycopg[binary,pool]")
            inst = cls._new_memory_saver()
            return inst, "pg_fallback" if inst is not None else ""

        # 这里的连接串对应我们 Docker 启动时的参数
        # user=agent_user, password=agent_pass, db=agent_db, port=5432
        default_conn = "postgresql://agent_user:agent_pass@localhost:5432/agent_db"
        conn_str = os.getenv("POSTGRES_DB_URL", default_conn)

        ok_key = f"postgres: {conn_str}"
        fb_key = f"pg_fallback: {conn_str}"

        # 检查缓存
        if ok_key in cls._instances: return cls._instances[ok_key], ok_key
        if fb_key in cls._instances: return cls._instances[fb_key], fb_key

        try:
            cls._log(f"[Checkpointer] 初始化 Postgres...")

            # 1. 创建连接池
            pool = AsyncConnectionPool(conn_str, max_size=20, kwargs={"autocommit": True})
            await pool.open()

            # 2. 创建 Saver
            checkpointer = AsyncPostgresSaver(pool)

            # 3. 初始化表结构 (第一次运行时会自动建表)
            await checkpointer.setup()

            # 4. 绑定 pool 以便后续关闭
            checkpointer._connection_pool = pool

            cls._log(f"[Checkpointer] ✅ Postgres 已启用")
            return checkpointer, ok_key

        except Exception as e:
            cls._log(f"[Checkpointer] ❌ Postgres 初始化失败: {type(e).__name__}: {e}")
            cls._log("[Checkpointer] 回退到 MemorySaver")
            inst = cls._new_memory_saver()
            return inst, fb_key if inst is not None else ""

    @classmethod
    async def _create_redis(cls, ctx: Context):
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        except ImportError:
            cls._log("[Checkpointer] ❌ 缺少依赖：pip install langgraph-checkpoint-redis")
            inst = cls._new_memory_saver()
            return inst, "redis_fallback"

        redis_url = settings.redis.REDIS_URL
        ok_key = f"redis:{redis_url}"

        try:
            checkpointer = AsyncRedisSaver.from_conn_string(redis_url)
            await checkpointer.setup()  # 首次运行建 key 结构
            cls._log(f"[Checkpointer] ✅ Redis 会话存储已启用")
            return checkpointer, ok_key
        except Exception as e:
            cls._log(f"[Checkpointer] ❌ Redis Checkpointer 失败: {e}，回退 Memory")
            inst = cls._new_memory_saver()
            return inst, "redis_fallback"

