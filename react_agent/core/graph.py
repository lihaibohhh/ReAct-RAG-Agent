"""
react_agent/graph.py (持久化版本)

核心改进：
1. 解决 AsyncSqliteSaver 的异步初始化问题
2. 支持动态 checkpointer 创建（延迟初始化）
3. 保持向后兼容，默认使用 MemorySaver
4. 提供工程级别的错误处理和降级策略
5. 集成 Postgres 持久化
6. ✅ 新增 Reflection Node（反思节点）：工具报错时自动介入
7. 优化图结构：Tools -> Reflection -> Model
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from react_agent.memory.context import Context
from react_agent.core.state import InputState, State
from react_agent.core.routing import route_model_output
from react_agent.core.checkpointer import CheckpointerFactory
from react_agent.core.nodes import call_model, dynamic_tool_node, postprocess_tools, reflection_node


# ==================== 图构建器（基础版）====================
def build_base_graph() -> StateGraph:
    """
    构建基础图（不包含 checkpointer）

    这样可以：
    1. 预先定义图结构
    2. 在运行时动态添加 checkpointer
    3. 支持多种编译模式
    """
    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=Context,
    )

    builder.add_node("call_model", call_model)
    builder.add_node("tools", dynamic_tool_node)
    builder.add_node("postprocess_tools", postprocess_tools)
    builder.add_node("reflection", reflection_node)  # ✅ 新增：注册反思节点

    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "postprocess_tools")

    # Postprocess -> Reflection (✅ 关键改变：处理完结果后，先去反思节点检查有没有错)
    builder.add_edge("postprocess_tools", "reflection")

    # Reflection -> Call Model (✅ 闭环：反思完（无论有错没错），都回模型继续思考)
    builder.add_edge("reflection", "call_model")

    return builder


# ==================== 编译策略 ====================

# 策略1：默认编译（使用 MemorySaver）
# 适用于：快速开发、测试
_builder = build_base_graph()
try:
    from langgraph.checkpoint.memory import MemorySaver
    _default_checkpointer = MemorySaver()
except ImportError:
    _default_checkpointer = None

graph = _builder.compile(
    name="ReAct Agent (ZH)",
    checkpointer=_default_checkpointer
)


# 策略2：动态编译（推荐用于生产）
async def compile_graph_with_persistence(ctx: Context):
    """
    动态编译图（支持真正的持久化）

    使用方式：
    ```python
    ctx = Context(checkpoint_backend="sqlite")
    compiled_graph = await compile_graph_with_persistence(ctx)

    result = await compiled_graph.ainvoke(
        {"messages": [HumanMessage("你好")]},
        context=ctx,
        config={"configurable": {"thread_id": "user_123"}}
    )
    ```
    """
    checkpointer = await CheckpointerFactory.create(ctx)
    builder = build_base_graph()

    return builder.compile(
        name="ReAct Agent (ZH) - Persistent",
        checkpointer=checkpointer
    )


