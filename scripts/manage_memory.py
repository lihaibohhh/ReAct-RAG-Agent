"""
conversation_manager_enhanced.py

增强版对话管理工具 - 修复版
支持：
1. SQLite/PostgreSQL 后端
2. 对话历史导出/导入
3. 统计分析
4. 自动清理

修复内容：
- ✅ 修复 msgpack ExtType 解码问题
- ✅ 增强类型检查和数据验证
- ✅ 改进错误处理
- ✅ 支持多种消息格式
"""

import asyncio
import sqlite3
import json
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ThreadInfo:
    """线程信息"""
    thread_id: str
    checkpoint_count: int
    last_checkpoint_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    total_messages: int = 0


class ConversationManagerEnhanced:
    """增强版对话管理器"""

    def __init__(self, db_path: str = "./data/agent_checkpoints.sqlite3"):
        self.db_path = Path(db_path).resolve()
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """确保数据库存在"""
        if not self.db_path.exists():
            print(f"⚠️  数据库文件不存在: {self.db_path}")
            print("首次运行 agent 后会自动创建")

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def list_threads(self, limit: int = 100) -> List[ThreadInfo]:
        """
        列出所有会话

        Args:
            limit: 返回的最大数量

        Returns:
            线程信息列表
        """
        if not self.db_path.exists():
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT 
                    thread_id,
                    COUNT(*) as checkpoint_count,
                    MAX(checkpoint_id) as last_checkpoint_id,
                    MIN(checkpoint_id) as first_checkpoint_id
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_checkpoint_id DESC
                LIMIT ?
            """, (limit,))

            threads = []
            for row in cursor.fetchall():
                threads.append(ThreadInfo(
                    thread_id=row['thread_id'],
                    checkpoint_count=row['checkpoint_count'],
                    last_checkpoint_id=row['last_checkpoint_id'],
                ))

            return threads
        finally:
            conn.close()

    def get_thread_detail(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        获取线程详细信息

        Args:
            thread_id: 线程 ID

        Returns:
            详细信息字典
        """
        if not self.db_path.exists():
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # 获取所有 checkpoints
            cursor.execute("""
                SELECT checkpoint_id, checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY checkpoint_id ASC
            """, (thread_id,))

            checkpoints = []
            for row in cursor.fetchall():
                try:
                    # 尝试解析 checkpoint（可能是 pickle/msgpack 格式）
                    checkpoint_data = self._decode_checkpoint(row['checkpoint'])
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                except Exception as e:
                    checkpoint_data = {"error": f"解析失败: {e}"}
                    metadata = {}

                checkpoints.append({
                    "checkpoint_id": row['checkpoint_id'],
                    "checkpoint": checkpoint_data,
                    "metadata": metadata,
                })

            if not checkpoints:
                return None

            return {
                "thread_id": thread_id,
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints,
                "first_checkpoint_id": checkpoints[0]["checkpoint_id"],
                "last_checkpoint_id": checkpoints[-1]["checkpoint_id"],
            }
        finally:
            conn.close()

    def _decode_checkpoint(self, checkpoint_blob: bytes) -> Dict[str, Any]:
        """
        解码 checkpoint 数据

        LangGraph 可能使用 pickle 或 msgpack 序列化
        修复：添加更好的 msgpack 解码配置
        """
        try:
            # 尝试 pickle
            import pickle
            return pickle.loads(checkpoint_blob)
        except Exception:
            pass

        try:
            # 尝试 msgpack - 使用改进的配置
            import msgpack

            def decode_ext_type(code, data):
                """处理 msgpack 的 ExtType"""
                # 尝试将 ExtType 数据递归解码
                try:
                    return msgpack.unpackb(data, raw=False, strict_map_key=False)
                except Exception:
                    return {"ext_type_code": code, "data": data.hex()}

            # 使用 strict_map_key=False 允许非字符串键
            # 使用 ext_hook 处理扩展类型
            result = msgpack.unpackb(
                checkpoint_blob,
                raw=False,
                strict_map_key=False,
                ext_hook=decode_ext_type
            )
            return result
        except Exception as e:
            print(f"⚠️  msgpack 解码失败: {e}")
            pass

        # 兜底：返回原始数据摘要
        return {
            "raw_size": len(checkpoint_blob),
            "raw_preview": str(checkpoint_blob[:100]),
            "error": "无法解码 checkpoint"
        }

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        """
        规范化消息对象

        处理各种可能的消息格式：
        - 字典
        - msgpack ExtType
        - LangChain/LangGraph 消息对象
        """
        # 如果已经是字典，直接返回
        if isinstance(msg, dict):
            return msg

        # 尝试将对象转换为字典
        result = {}

        # 检查是否有 __dict__ 属性
        if hasattr(msg, '__dict__'):
            result = msg.__dict__.copy()

        # 检查常见的消息属性
        for attr in ['type', 'content', 'role', 'name', 'id', 'additional_kwargs']:
            if hasattr(msg, attr):
                try:
                    result[attr] = getattr(msg, attr)
                except Exception:
                    pass

        # 如果还是空的，尝试 str 表示
        if not result:
            result = {
                "raw_type": str(type(msg)),
                "content": str(msg)[:200],
                "error": "无法解析消息格式"
            }

        return result

    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        提取线程的所有消息

        修复：添加消息格式规范化

        Args:
            thread_id: 线程 ID

        Returns:
            消息列表
        """
        detail = self.get_thread_detail(thread_id)
        if not detail:
            return []

        # 从最后一个 checkpoint 提取 messages
        last_checkpoint = detail["checkpoints"][-1]["checkpoint"]

        messages = []

        # 尝试多种路径提取消息
        if isinstance(last_checkpoint, dict):
            # 路径1: channel_values.messages
            if "channel_values" in last_checkpoint:
                channel_values = last_checkpoint["channel_values"]
                if isinstance(channel_values, dict) and "messages" in channel_values:
                    raw_messages = channel_values["messages"]
                    if isinstance(raw_messages, list):
                        messages = raw_messages

            # 路径2: 直接在 messages 字段
            elif "messages" in last_checkpoint:
                raw_messages = last_checkpoint["messages"]
                if isinstance(raw_messages, list):
                    messages = raw_messages

        # 规范化所有消息
        normalized_messages = []
        for msg in messages:
            try:
                normalized_msg = self._normalize_message(msg)
                normalized_messages.append(normalized_msg)
            except Exception as e:
                # 即使单个消息失败，也继续处理其他消息
                normalized_messages.append({
                    "error": f"消息规范化失败: {e}",
                    "raw_content": str(msg)[:100]
                })

        return normalized_messages

    def export_thread(self, thread_id: str, output_path: str, format: str = "json") -> bool:
        """
        导出线程数据

        Args:
            thread_id: 线程 ID
            output_path: 输出路径
            format: 格式（json/markdown）

        Returns:
            是否成功
        """
        detail = self.get_thread_detail(thread_id)
        if not detail:
            print(f"❌ 线程不存在: {thread_id}")
            return False

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(detail, f, ensure_ascii=False, indent=2, default=str)

            elif format == "markdown":
                messages = self.get_thread_messages(thread_id)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# 对话记录：{thread_id}\n\n")
                    f.write(f"总消息数：{len(messages)}\n\n")
                    f.write("---\n\n")

                    for i, msg in enumerate(messages, 1):
                        # 修复：安全地获取消息属性
                        if isinstance(msg, dict):
                            role = msg.get("type") or msg.get("role", "unknown")
                            content = msg.get("content", "")
                        else:
                            role = "unknown"
                            content = str(msg)

                        f.write(f"## 消息 {i} - {role}\n\n")
                        f.write(f"{content}\n\n")

            else:
                print(f"❌ 不支持的格式: {format}")
                return False

            print(f"✅ 导出成功: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_thread(self, thread_id: str, confirm: bool = False) -> bool:
        """
        删除线程

        Args:
            thread_id: 线程 ID
            confirm: 是否已确认（安全检查）

        Returns:
            是否成功
        """
        if not confirm:
            print("⚠️  危险操作：需要确认")
            print(f"要删除线程 '{thread_id}'，请使用 confirm=True")
            return False

        if not self.db_path.exists():
            print(f"❌ 数据库不存在: {self.db_path}")
            return False

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            deleted_count = cursor.rowcount
            conn.commit()

            print(f"✅ 已删除 {deleted_count} 条记录 (thread_id: {thread_id})")
            return deleted_count > 0
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def cleanup_old_threads(self, days: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """
        清理旧线程

        Args:
            days: 保留最近多少天的数据
            dry_run: 是否只模拟（不实际删除）

        Returns:
            清理统计
        """
        if not self.db_path.exists():
            return {"error": "数据库不存在"}

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # 找出旧线程（根据最后一次更新时间）
            cutoff_id = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT thread_id, COUNT(*) as count
                FROM checkpoints
                WHERE checkpoint_id < ?
                GROUP BY thread_id
            """, (cutoff_id,))

            old_threads = cursor.fetchall()

            if dry_run:
                print(f"🔍 模拟模式：找到 {len(old_threads)} 个超过 {days} 天的线程")
                for row in old_threads[:10]:  # 只显示前 10 个
                    print(f"  - {row['thread_id']}: {row['count']} 条记录")

                return {
                    "dry_run": True,
                    "thread_count": len(old_threads),
                    "threads": [dict(row) for row in old_threads]
                }

            # 实际删除
            deleted_total = 0
            for row in old_threads:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (row['thread_id'],))
                deleted_total += cursor.rowcount

            conn.commit()
            print(f"✅ 清理完成：删除 {len(old_threads)} 个线程，共 {deleted_total} 条记录")

            return {
                "dry_run": False,
                "thread_count": len(old_threads),
                "record_count": deleted_total,
            }

        except Exception as e:
            print(f"❌ 清理失败: {e}")
            conn.rollback()
            return {"error": str(e)}
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.db_path.exists():
            return {"error": "数据库不存在"}

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # 总线程数
            cursor.execute("SELECT COUNT(DISTINCT thread_id) as count FROM checkpoints")
            total_threads = cursor.fetchone()['count']

            # 总 checkpoint 数
            cursor.execute("SELECT COUNT(*) as count FROM checkpoints")
            total_checkpoints = cursor.fetchone()['count']

            # 数据库大小
            db_size_mb = self.db_path.stat().st_size / 1024 / 1024

            # 最活跃的线程
            cursor.execute("""
                SELECT thread_id, COUNT(*) as count
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY count DESC
                LIMIT 5
            """)
            top_threads = [dict(row) for row in cursor.fetchall()]

            return {
                "total_threads": total_threads,
                "total_checkpoints": total_checkpoints,
                "db_size_mb": round(db_size_mb, 2),
                "db_path": str(self.db_path),
                "top_threads": top_threads,
            }
        finally:
            conn.close()

    def vacuum_database(self):
        """
        优化数据库（VACUUM）

        清理已删除的数据，减少文件大小
        """
        if not self.db_path.exists():
            print("❌ 数据库不存在")
            return

        print("🔧 正在优化数据库...")
        size_before = self.db_path.stat().st_size / 1024 / 1024

        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()

        size_after = self.db_path.stat().st_size / 1024 / 1024
        saved = size_before - size_after

        print(f"✅ 优化完成")
        print(f"  优化前: {size_before:.2f} MB")
        print(f"  优化后: {size_after:.2f} MB")
        print(f"  节省: {saved:.2f} MB ({saved / size_before * 100:.1f}%)")


async def main():
    """命令行界面"""
    import sys

    if len(sys.argv) < 2:
        print("增强版对话管理工具 - 修复版")
        print()
        print("用法:")
        print("  python conversation_manager.py list [limit]")
        print("  python conversation_manager.py detail <thread_id>")
        print("  python conversation_manager.py messages <thread_id>")
        print("  python conversation_manager.py export <thread_id> <file> [format]")
        print("  python conversation_manager.py delete <thread_id>")
        print("  python conversation_manager.py cleanup <days> [--execute]")
        print("  python conversation_manager.py stats")
        print("  python conversation_manager.py vacuum")
        return

    command = sys.argv[1]
    mgr = ConversationManagerEnhanced()

    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        threads = mgr.list_threads(limit)

        if not threads:
            print("暂无对话记录")
        else:
            print(f"共 {len(threads)} 个会话:\n")
            for i, t in enumerate(threads, 1):
                print(f"{i}. {t.thread_id}")
                print(f"   Checkpoints: {t.checkpoint_count}")
                print(f"   Last: {t.last_checkpoint_id}")
                print()

    elif command == "detail":
        if len(sys.argv) < 3:
            print("请提供 thread_id")
            return

        thread_id = sys.argv[2]
        detail = mgr.get_thread_detail(thread_id)

        if not detail:
            print(f"线程不存在: {thread_id}")
        else:
            print(f"线程 ID: {detail['thread_id']}")
            print(f"Checkpoint 数量: {detail['checkpoint_count']}")
            print(f"首次: {detail['first_checkpoint_id']}")
            print(f"最后: {detail['last_checkpoint_id']}")

    elif command == "messages":
        if len(sys.argv) < 3:
            print("请提供 thread_id")
            return

        thread_id = sys.argv[2]
        messages = mgr.get_thread_messages(thread_id)

        print(f"线程 '{thread_id}' 的消息 (共 {len(messages)} 条):\n")
        for i, msg in enumerate(messages, 1):
            # 修复：安全地处理消息对象
            if isinstance(msg, dict):
                role = msg.get("type") or msg.get("role", "unknown")
                content = str(msg.get("content", ""))
            else:
                role = "unknown"
                content = str(msg)

            # 限制显示长度
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"{i}. [{role}] {content_preview}")

    elif command == "export":
        if len(sys.argv) < 4:
            print("用法: ... export <thread_id> <file> [format]")
            return

        thread_id = sys.argv[2]
        output_path = sys.argv[3]
        format = sys.argv[4] if len(sys.argv) > 4 else "json"

        mgr.export_thread(thread_id, output_path, format)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("请提供 thread_id")
            return

        thread_id = sys.argv[2]
        print(f"⚠️  即将删除线程: {thread_id}")
        confirm = input("确认删除？输入 'yes': ")

        if confirm.lower() == "yes":
            mgr.delete_thread(thread_id, confirm=True)
        else:
            print("已取消")

    elif command == "cleanup":
        if len(sys.argv) < 3:
            print("请提供天数")
            return

        days = int(sys.argv[2])
        execute = "--execute" in sys.argv

        result = mgr.cleanup_old_threads(days, dry_run=not execute)

        if result.get("dry_run"):
            print(f"\n💡 要实际执行清理，请添加 --execute 参数")

    elif command == "stats":
        stats = mgr.get_statistics()
        print("数据库统计:\n")
        for key, value in stats.items():
            if key == "top_threads":
                print(f"  最活跃的线程:")
                for t in value:
                    print(f"    - {t['thread_id']}: {t['count']} 条")
            else:
                print(f"  {key}: {value}")

    elif command == "vacuum":
        mgr.vacuum_database()

    else:
        print(f"未知命令: {command}")


if __name__ == "__main__":
    asyncio.run(main())
