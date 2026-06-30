# ReAct Agent — 金融研报问答智能体

## 项目定位

面向金融分析师的企业级 Agent 框架。核心价值：跨 PDF 文档检索 + Text2SQL 精确财务查询，每条结论标注来源文档与页码，可追溯可验证。

架构：LangGraph `StateGraph` + ReAct 范式（推理 → 工具调用 → 观察 → 反思）  
服务层：Streamlit UI + FastAPI（REST + SSE），共享同一 Agent 实例。

---

## 启动命令

```bash
# Streamlit UI
streamlit run src/app.py

# FastAPI 服务
uvicorn src.react_agent.api.main:app --reload --port 8000

# 构建/更新向量库（支持增量，文件哈希去重）
python src/react_agent/rag/vector_store.py

# 财务数据抽取入库
python src/react_agent/rag/extract_financials.py

# RAGAS 评测（检索/端到端双模式）
python src/react_agent/eval/run_eval.py

# PDF 质检（扫描版/加密 PDF 检出）
python src/react_agent/rag/check_pdfs.py
```

---

## 架构与模块分工

### Core（`src/react_agent/core/`）

节点执行顺序：`call_model → tools → reflection → call_model`

| 文件 | 职责 |
|---|---|
| `graph.py` | 主工作流图定义 |
| `nodes.py` | 各节点逻辑（模型调用、工具执行、反思） |
| `routing.py` | 动态路由，根据工具结果决定下一步 |
| `state.py` | 统一运行时状态（工具轨迹、引用清单、错误计数、Token 统计） |
| `prompts.py` | 结构化中文系统提示（工具决策规则、查询改写规则、输出风格） |
| `agent.py` | `PersistentAgent` 包装器，对外暴露 `invoke`/`stream`，内置历史裁剪 |
| `checkpointer.py` | 支持 Memory/SQLite/PostgreSQL/Redis 四后端，自动降级 |

### API（`src/react_agent/api/`）

- `PersistentAgent` 为模块级单例（`dependencies.py`），全进程只初始化一次
- SSE 事件类型：`tool_call` / `tool_result` / `token` / `done` / `error`
- session → thread 映射格式：`user:{session_id}`（与 Streamlit 侧命名空间隔离）

### RAG（`src/react_agent/rag/`）

数据流：`pdf_parser.py` → `chunker.py` → `vector_store.py`（多进程并行建库）

检索流：BM25 + 向量双路（各 Top-10）→ RRF 融合（k=60）→ Top-20 候选 → Cross-Encoder 精排 → Top-3

- `reranker.py` 返回 `(docs, top_score)` 元组，top_score 透传给缓存层
- `semantic_cache.py`：`top_score ≥ 0.5` 长 TTL 缓存；`< 0.5` 跳过缓存；空结果 300s 短 TTL

### Tools（`src/react_agent/tools/`）

| 工具 | 文件 | 说明 |
|---|---|---|
| `query_internal_knowledge` | `rag.py` | RAG 检索，结果含 `[来源：xxx.pdf 第N页]` |
| `sql_tool` | `sql.py` | Text2SQL 精确财务查询，强制返回 `source_file`/`source_page` |
| `search` | `search.py` | Tavily 联网搜索 |
| `excel_tool` | `excel.py` | Excel 生成（timestamp/overwrite/append 三模式） |

---

## ⚡ 工具优先级规则（CRITICAL，须在系统提示中体现）

| 问题类型 | 必须使用的工具 |
|---|---|
| "是多少"、具体数字、多公司横向对比 | **`sql_tool`**（优先） |
| "为什么"、"怎么看"、定性分析、行业背景 | `query_internal_knowledge` |
| 联网信息、最新政策 | `search` |

RAG 工具禁止用于可结构化查询的精确数值——避免时间错位幻觉。

---

## 关键约束（踩过坑的规则，修改时必读）

### thread_id
- 格式必须为 `user:{username}` 或 `user:{session_id}`
- 禁止 `default` 或任何硬编码值（会导致跨用户对话数据污染）
- `agent.py` 在未传 thread_id 时抛出 `ValueError`，这是有意设计

### 调用侧传参
- 启用持久化后，调用侧只传当前新消息，不传全量历史
- LangGraph 侧会自动从 checkpointer 恢复历史，两边都传会导致消息重复，计数虚高

### Guard / 哨兵消息措辞
- 必须使用**程序性措辞**：`"本轮检索上限已达，新一轮可重新尝试"`
- 禁止事实性结论措辞（如"知识库没有答案"）——会让 LLM 在下一轮推断"重查无用"而不调工具
- `system_terminator` 和 `system_monitor` 是两个不同 name，不能混用

### RAG query 改写
- 多轮追问时，必须将代词替换为具体实体（`prompts.py` 中有规则）
- 禁止在 query 中保留"该公司"、"这家车企"等模糊指代

### sql_tool 规范
- `_get_schema()` 的示例 SQL 必须包含 `source_file`、`source_page` 字段
- LLM 会严格模仿示例 SQL，示例缺字段则生成的 SQL 也会缺字段
- 格式化输出代码禁止过滤 `source_file`/`source_page`（曾因此导致溯源功能静默失效）
- `sql` 变量在 `try:` 第一行初始化为 `"（未能生成SQL）"`，避免 `except` 块引用未初始化变量

### 路径规范
- 所有数据路径用绝对路径：`pathlib.Path(__file__).resolve().parent...`
- 禁止 `./chroma_db` 等相对路径（PyCharm 运行子目录脚本时工作目录不同，会指向空库）

### MCP Server 规范
- 所有 MCP 相关脚本禁止 `print()`，统一用 `logger.info()` / `logger.error()`（print 会污染 JSON 协议）
- 第三方库懒加载（只在函数调用时导入，握手阶段导入会超时）
- JSON 序列化前将 `numpy.float32` 转为原生 `float`

### 异步规范
- 阻塞操作（如 BM25 重建、HuggingFace 模型加载）必须用 `asyncio.to_thread()` 移出事件循环
- 评测脚本调用 async 函数必须加 `await`，否则只返回 coroutine 对象，静默空结果

---

## 当前待完善事项（按优先级）

1. **`sql.py` Schema 同步**：`_get_schema()` 中 `metric` 字段描述需与 `extract_financials.py` 实际入库字段同步（新增 `raw_metric`、`confidence` 字段后尚未更新）

2. **`chunker.py` 上下文感知分块**：每个 Chunk 头部注入全局元数据（从文件名提取公司名/文档标题），解决评测数据集生成时因代词指代不明导致的召回失败问题（根本解法）

3. **工具调用历史跨轮隔离**：当前靠 Guard 措辞缓解，根本解法是在 `agent.py` 新轮次入口将上一轮 tool call chain 替换为中性摘要（State 分层方案）

4. **图表型研报多模态抽取**：PPT 型研报数据在图表中，`extract_financials.py` 的文本抽取有根本性局限（年份-数值错位、量级错误）。完整解法：`build_vector_db()` 阶段对每页截图，引入多模态视觉模型（如 GPT-4o Vision）识别图表

5. **FastAPI token 级流式输出**：当前 `/chat/stream` 为节点级推送，需在 `PersistentAgent` 新增基于 `astream_events(version="v2")` 的 `stream_events()` 方法

6. **PDF 水印过滤**：水印内容破坏 chunk 语义，影响评测指标上限，需在 `pdf_parser.py` 增加水印检测与过滤

---

## 语言与本地化

全链路中文：`zh-CN` / `Asia/Shanghai` / 中文系统提示（`prompts.py`）