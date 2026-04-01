from dotenv import load_dotenv
load_dotenv()

import asyncio
import threading
import streamlit as st
from langchain_core.messages import HumanMessage
import os

os.environ["TAVILY_API_KEY"] = "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"

from react_agent.memory.context import Context
from react_agent.core.agent import PersistentAgent

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
# loop 和 agent 绑在同一个 cache_resource 里：
# 生命周期一致，SQLite Lock 永远和它的 loop 在一起
@st.cache_resource
def get_agent_and_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    ctx = Context(checkpoint_backend="sqlite", model="deepseek/deepseek-chat")
    agent = PersistentAgent(ctx)
    run(agent.initialize())

    try:
        from react_agent.rag.reranker import _get_reranker
        from react_agent.rag.retriever import _get_retriever
        _get_retriever()
        run(asyncio.to_thread(_get_reranker))
        print("[启动] 预热完成")
    except Exception as e:
        print(f"[启动] 预热失败（不影响启动）：{e}")

    return agent, loop


agent, _loop = get_agent_and_loop()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ── 会话状态初始化 ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏：显示当前用户信息，提供退出登录
with st.sidebar:
    st.write(f"当前用户：**{st.session_state.username}**")
    st.write(f"会话 ID：`{st.session_state.thread_id}`")
    if st.button("退出登录"):
        st.session_state.clear()
        st.rerun()

# ── 渲染历史对话 ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 处理新消息 ────────────────────────────────────────────────
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调度工具..."):
            result = run_async(
                agent.invoke(
                    [HumanMessage(content=prompt)],
                    thread_id=st.session_state.thread_id
                )
            )

            final_ai_msg = result["messages"][-1].content

            def stream_data(text):
                import time
                for char in text:
                    yield char
                    time.sleep(0.015)

            st.write_stream(stream_data(final_ai_msg))

            if "tool_runs" in result and result["tool_runs"]:
                with st.expander("🛠️ 查看工具调用轨迹"):
                    st.json(result["tool_runs"])

    st.session_state.messages.append({"role": "assistant", "content": final_ai_msg})