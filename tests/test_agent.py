from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import threading
import time
import streamlit as st
from langchain_core.messages import HumanMessage
import os

# TAVILY_API_KEY 通过 .env / 系统环境变量提供（load_dotenv() 已加载），不在源码中硬编码
if not os.environ.get("TAVILY_API_KEY"):
    logging.getLogger(__name__).warning("[启动] 未检测到 TAVILY_API_KEY 环境变量，search 工具可能不可用")

from react_agent.memory.context import Context
from react_agent.core.agent import PersistentAgent
from react_agent.utils.usage_logger import (
    log_usage,
    extract_cumulative_snapshot,
    format_usage_for_user,
    SessionUsageTracker,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── 必须是第一个 Streamlit 调用 ──────────────────────────────
st.set_page_config(page_title="Enterprise ReAct Agent", page_icon="🤖")

# ── 登录拦截：未登录不渲染任何后续内容 ──────────────────────
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.title("请先登录")
    username = st.text_input("输入你的用户名（内部工号或姓名拼音）")
    if st.button("进入") and username.strip():
        st.session_state.username = username.strip()
        st.session_state.thread_id = f"user:{username.strip()}"
        st.rerun()
    st.stop()

# ── 以下内容仅登录后可见 ──────────────────────────────────────
st.title(f"Enterprise ReAct Agent — {st.session_state.username}")


# ── Agent 初始化（单例，热重载不重建）────────────────────────
@st.cache_resource
def get_agent_and_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    ctx = Context(checkpoint_backend="postgres", model="deepseek/deepseek-v4-flash")
    agent = PersistentAgent(ctx)
    run(agent.initialize())

    try:
        from react_agent.rag.reranker import _get_reranker
        from react_agent.rag.retriever import _get_retriever
        _get_retriever()
        run(asyncio.to_thread(_get_reranker))
        logger.info("[启动] 预热完成")
    except Exception as e:
        logger.info(f"[启动] 预热失败（不影响启动）：{e}")

    return agent, loop


agent, _loop = get_agent_and_loop()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ── 会话状态初始化 ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ▸ 新增：会话级用量追踪器
if "usage_tracker" not in st.session_state:
    st.session_state.usage_tracker = SessionUsageTracker()

# ▸ 新增：最近一轮的用量展示数据（供侧边栏渲染）
if "last_turn_display" not in st.session_state:
    st.session_state.last_turn_display = None

# ▸ 新增：上一轮结束时的累计快照（用于做差值算本轮增量）
if "prev_usage_snapshot" not in st.session_state:
    st.session_state.prev_usage_snapshot = None


# ── 统计面板渲染函数 ─────────────────────────────────────────
def render_stats(placeholder):
    tracker = st.session_state.usage_tracker
    if tracker.turn_count == 0:
        return
    with placeholder.container():
        st.divider()
        st.caption("📊 本次会话统计")

        col1, col2 = st.columns(2)
        col1.metric("对话轮数", f"{tracker.turn_count}")
        col2.metric("累计成本", f"¥{tracker.total_cost_usd * 7.2:.2f}")

        col3, col4 = st.columns(2)
        col3.metric("模型调用", f"{tracker.total_llm_calls} 次")
        col4.metric("工具调用", f"{tracker.total_tool_runs} 次")

        with st.expander("查看每轮明细"):
            for turn in tracker.turn_usages:
                latency = turn["latency_ms"]
                cost = turn.get("estimated_cost_usd", 0)
                tokens = turn.get("total_tokens", 0)
                st.text(
                    f"第 {turn['turn']} 轮 | "
                    f"{latency / 1000:.1f}s | "
                    f"{tokens:,} tokens | "
                    f"¥{cost * 7.2:.3f}"
                )

        warning = tracker.check_budget(limit_usd=1.0)
        if warning:
            st.warning(warning)


# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.write(f"当前用户：**{st.session_state.username}**")
    st.write(f"会话 ID：`{st.session_state.thread_id}`")

    # ▸ 预留占位符，并渲染已有历史数据
    stats_placeholder = st.empty()
    render_stats(stats_placeholder)

    st.divider()
    if st.button("退出登录"):
        st.session_state.clear()
        st.rerun()

# ── 渲染历史对话 ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # ▸ 新增：历史消息如果有 usage 信息，在底部灰色小字展示
        if msg["role"] == "assistant" and msg.get("usage_display"):
            display = msg["usage_display"]
            parts = [f"{k}: {v}" for k, v in display.items()]
            st.caption(" · ".join(parts))

# ── 处理新消息 ────────────────────────────────────────────────
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调度工具..."):

            # ▸ 新增：计时
            t0 = time.perf_counter()

            result = run_async(
                agent.invoke(
                    [HumanMessage(content=prompt)],
                    thread_id=st.session_state.thread_id
                )
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            final_ai_msg = result["messages"][-1].content

            # ▸ 新增：三层用量处理
            # 第一层 (运维)：写入结构化日志（传入上一轮快照做差值）
            usage = log_usage(
                result,
                username=st.session_state.username,
                thread_id=st.session_state.thread_id,
                question=prompt,
                latency_ms=latency_ms,
                prev_snapshot=st.session_state.prev_usage_snapshot,
            )

            # ▸ 保存本轮结束时的累计快照，供下一轮做差值
            snapshot = extract_cumulative_snapshot(result)
            snapshot["tool_runs_count"] = len(result.get("tool_runs", []))
            st.session_state.prev_usage_snapshot = snapshot

            # 第二层 (业务)：累计到会话级追踪器
            st.session_state.usage_tracker.record_turn(usage, latency_ms)
            render_stats(stats_placeholder)  # ← 本轮数据写入后立即刷新侧边栏

            # 第三层 (用户)：生成展示文本
            usage_display = format_usage_for_user(usage, latency_ms)
            st.session_state.last_turn_display = usage_display


            def stream_data(text):
                for char in text:
                    yield char
                    time.sleep(0.015)


            st.write_stream(stream_data(final_ai_msg))

            # 工具调用轨迹（已有功能）
            if "tool_runs" in result and result["tool_runs"]:
                with st.expander("🛠️ 查看工具调用轨迹"):
                    st.json(result["tool_runs"])

            # ▸ 新增：本轮用量摘要（灰色小字，不抢眼）
            parts = [f"{k}: {v}" for k, v in usage_display.items()]
            st.caption(" · ".join(parts))

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_ai_msg,
        "usage_display": usage_display,  # ▸ 随消息存储，翻阅历史时仍可见
    })
