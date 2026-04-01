# scripts/verify_redis.py
import asyncio
from react_agent.utils.redis_client import ping_redis, get_async_redis


async def main():
    ok = await ping_redis()
    print(f"Redis 连接: {'✅' if ok else '❌'}")

    r = get_async_redis()
    keys = [k async for k in r.scan_iter("rag:*")]
    print(f"当前 RAG 缓存键数量: {len(keys)}")
    for k in keys:  # 改成全部显示
        ttl = await r.ttl(k)
        size = await r.memory_usage(k)  # 新增：看每个 key 占多少内存
        print(f"  {k.decode()}")
        print(f"    TTL剩余: {ttl}s  |  大小: {size} bytes")

asyncio.run(main())