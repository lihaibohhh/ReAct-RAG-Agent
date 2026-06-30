# ReAct-RAG-Agent

> 面向金融研报问答、私有知识库检索与多工具分析生成的企业级 ReAct-RAG Agent。  
> 核心目标：让用户用自然语言完成「跨研报检索 → 可信溯源 → 联网补充 → Word / Excel / Markdown 输出 → 多轮会话延续」。

---

## 1. 项目定位

金融分析师、投研人员或行业研究团队每天需要阅读大量研报、年报、公告和政策材料。传统流程通常是逐篇打开 PDF、手动搜索关键词、复制表格、再整理成报告，效率低且容易遗漏来源。

本项目实现了一个面向金融研报场景的 ReAct-RAG Agent，能够：

- 在私有研报知识库中检索公司、行业、指标、事件相关内容；
- 对回答中的关键结论标注来源文件和页码，便于追溯和复核；
- 结合联网搜索补充公开信息；
- 调用工具生成 Excel、Word、Markdown 等结构化交付物；
- 通过 Checkpoint 持久化保存会话上下文，支持多轮追问；
- 通过 Streamlit UI 和 FastAPI REST / SSE 双入口接入前端或外部系统。

> 本仓库不包含私有研报数据、向量库、密钥、运行日志和本地数据库。请通过 `.env.example` 配置环境变量，并将自己的 PDF / 数据文件放入本地忽略目录中。

---

## 2. 关于这个项目

| 方向 | 已完成能力 |
|---|---|
| **Agent 架构** | LangGraph `StateGraph` + ReAct 循环，支持模型推理、工具调用、观察、反思与动态路由 |
| **RAG 检索** | PyMuPDF + pdfplumber 金融研报解析，BM25 + 向量双路召回，RRF 融合，Cross-Encoder 精排 |
| **可信溯源** | RAG 结果保留来源文件和页码，回答可追溯、可复核 |
| **多工具输出** | Web 搜索、知识库检索、Excel、Word、Markdown，Text2SQL 模块可按需启用 |
| **记忆与隔离** | Checkpointer 支持 SQLite / PostgreSQL / Redis / Memory，多用户 thread 命名空间隔离 |
| **服务化** | FastAPI v1 API，token 级 SSE 流式，断连取消上游 LLM，统一错误信封，request_id 日志 |
| **安全与成本控制** | API Key 鉴权，匿名 IP 限流，Redis 固定窗口限流，每日 token 预算，fail-open 降级 |
| **可观测性** | Prometheus 指标采集：HTTP 延迟、in-flight、TTFT、tokens、成本、429 / 401 |
| **测试与 CI** | FastAPI 回归测试使用 FakeAgent + fakeredis，避免真实 LLM / Redis 调用，做到零烧钱测试 |
| **评测体系** | RAGAS 自动评测管道，150 条样本、6 个行业分组，记录检索与端到端指标 |

---

## 3. 已验证使用场景

- **跨文档研报问答**：从数百份 PDF 中定位涉及特定公司、行业、财务指标或政策事件的段落，并返回来源页码。
- **财务指标精查**：`sql.py` 提供 Text2SQL 路径，可对 SQLite 财务指标库进行自然语言查询，并返回 `source_file` / `source_page`。
- **联网搜索 + 研究报告生成**：先搜索公开事件，再生成带章节结构和来源标注的 `.docx` 报告。
- **同一会话多格式输出**：一轮分析后继续追问「整理成 Word」或「生成公司对比 Excel」，Agent 会基于当前上下文调用对应工具。
- **外部系统接入**：通过 REST API / SSE 接入 n8n、自定义前端、第三方服务或自动化工作流。

---

## 4. 系统架构

![Architecture](img_1.png)

```mermaid
flowchart LR
    U[User / Frontend] --> S[Streamlit UI]
    U --> A[FastAPI REST + SSE]

    S --> G[PersistentAgent]
    A --> G

    G --> LG[LangGraph ReAct Workflow]
    LG --> T1[RAG Tool]
    LG --> T2[Web Search Tool]
    LG --> T3[Excel / Word / Markdown Tools]

    T1 --> P[PDF Parser]
    P --> R[BM25 + Vector Retriever]
    R --> RR[RRF Fusion]
    RR --> CE[Cross-Encoder Reranker]
    CE --> C[Semantic Cache]

    G --> CP[Checkpoint Store]
    C --> Redis[(Redis)]
    CP --> DB[(SQLite / PostgreSQL / Redis / Memory)]
```

### 核心链路

```text
用户请求
  → Streamlit / FastAPI
  → PersistentAgent
  → LangGraph ReAct 循环
  → 工具调用：RAG / Search / Excel / Word / Markdown
  → 反思与结果整理
  → 带来源、工具轨迹、token / cost 信息的响应
```

---

## 5. 核心模块

### 5.1 Core：Agent 图与状态

| 文件 | 职责 |
|---|---|
| `react_agent/core/graph.py` | 构建主工作流图：`call_model → tools → reflection → call_model` |
| `react_agent/core/nodes.py` | 模型调用、工具执行、Guard 哨兵、跨轮隔离、悬空 tool_call 净化 |
| `react_agent/core/routing.py` | 动态路由，统一检测 `.tool_calls` 与 `.invalid_tool_calls` |
| `react_agent/core/state.py` | 扩展运行时状态：工具轨迹、工作记忆、引用清单、错误计数、token 统计 |
| `react_agent/core/prompts.py` | 中文系统提示、工具选择规则、查询改写规则、跨轮独立性约束 |
| `react_agent/core/agent.py` | `PersistentAgent` 封装，提供 `invoke` / `stream_events` / history 管理 |
| `react_agent/core/checkpointer.py` | Checkpointer 工厂，支持 SQLite / PostgreSQL / Redis / Memory 自动降级 |

### 5.2 RAG：私有知识库检索

| 阶段 | 实现 |
|---|---|
| PDF 解析 | PyMuPDF 提取带 bbox 坐标文本块，处理多栏阅读顺序；pdfplumber 提取表格并转 Markdown |
| 分块 | 语法感知分块；PDF 表格预分块透传，避免二次切分破坏表格 |
| 召回 | BM25 + 向量双路 Top-K 召回 |
| 融合 | RRF 融合，合并去重得到候选池 |
| 精排 | Cross-Encoder reranker，返回 `(docs, top_score)` |
| 缓存 | Redis 语义缓存，基于 reranker 置信度门控，低置信结果不固化 |

检索流程：

```text
PDF / 文档
  → 版面解析与表格抽取
  → chunking
  → BM25 Top-10 + Vector Top-10
  → RRF 融合到最多 20 个候选
  → Cross-Encoder 精排
  → Top-3 返回给 Agent
```

### 5.3 Tools：Agent 工具层

| 工具 | 说明 |
|---|---|
| `query_internal_knowledge` | 调用 RAG 管道，返回带来源文件和页码的知识库结果 |
| `search` | Tavily Web 搜索，补充公开信息 |
| `make_excel_table` | 生成 Excel 表格，支持 timestamp / overwrite / append 模式 |
| `docx_tool` | 生成正式 Word 报告，支持标题、正文、列表、表格、页脚等样式 |
| `md_tool` | 生成 Markdown 文档，适合 GitHub、飞书、Notion 等场景 |
| `sql.py` | Text2SQL 财务精查模块，当前可按需注册到 `TOOLS` 中启用 |

> 当前默认注册工具以 `react_agent/tools/__init__.py` 为准。`sql.py` 已实现但默认未注册时，不会被 Agent 自动调用。

### 5.4 API：FastAPI 服务层

FastAPI 与 Streamlit 并存，共享同一套 Agent 实例与持久化存储。

| 能力 | 说明 |
|---|---|
| 版本化 API | `/api/v1/*` 为新接口，旧 `/chat/*` 作为 legacy 保留 |
| token 级流式 | `/api/v1/chat/stream` 基于 `astream_events(version="v2")` 输出 token / tool_call / tool_result / done / error |
| 断连取消 | 客户端断开后取消并 await upstream task，避免 LLM 继续消耗 |
| 统一错误 | RFC 7807 风格 problem+json，错误体与响应头均包含 request_id |
| 鉴权 | 支持 `X-API-Key` 与 `Authorization: Bearer` |
| 限流与预算 | Redis 固定分钟窗口限流 + per-user 每日 token 预算 |
| 可观测 | Prometheus 指标采集 HTTP 延迟、TTFT、tokens、cost、in-flight、429 / 401 |
| 会话管理 | history / delete session，bucket_key 命名空间隔离，防 IDOR 越权读取 |

---

## 6. API 示例

### 健康检查

```bash
curl http://localhost:8000/health
```

### 非流式调用

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "总结新能源行业最近的投资主线，并给出来源",
    "session_id": "demo-session"
  }'
```

### SSE 流式调用

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "检索半导体行业最近的供需变化，并生成要点",
    "session_id": "demo-session"
  }'
```

流式事件类型：

```text
token | tool_call | tool_result | usage | done | error
```

### Prometheus 指标

```bash
curl http://localhost:8000/api/v1/metrics
```

---

## 7. 快速启动

### 7.1 环境准备

推荐 Python 3.10+。

```bash
git clone https://github.com/lihaibohhh/ReAct-RAG-Agent.git
cd ReAct-RAG-Agent/src
```

使用 `uv`：

```bash
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

### 7.2 配置环境变量

```bash
cp .env.example .env
```

根据自己的模型和工具服务填写：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key

TAVILY_API_KEY=your_tavily_api_key

# 可选：Redis / PostgreSQL / LangSmith 等
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://user:password@localhost:5432/dbname
```

### 7.3 启动 Streamlit

```bash
cd src
streamlit run tests/test_agent.py
```

### 7.4 启动 FastAPI

```bash
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger 文档：

```text
http://localhost:8000/docs
```

---

## 8. 测试与评测

### 8.1 API 回归测试

```bash
pytest tests/api -q
```

测试设计要点：

- 使用 FakeAgent 替代真实 LLM；
- 使用 fakeredis 替代真实 Redis；
- 覆盖鉴权、限流、错误信封、SSE 断连、Prometheus、会话 CRUD、IDOR 防护；
- CI 中不调用真实模型，不消耗 API 费用。

### 8.2 RAGAS 评测

```bash
python eval/run_eval.py
```

支持检索评测与端到端评测。已验证指标：

| 指标 | 结果 |
|---|---:|
| Context Precision | 0.7517 |
| Context Recall | 0.7850 |
| Faithfulness | 0.9011 |
| 样本规模 | 150 |
| 行业分组 | 6 |

### 8.3 向量库建库性能

| 指标 | 串行版 | 多进程版 | 变化 |
|---|---:|---:|---:|
| 全量建库总耗时 | ~57 min | ~28 min | 约 2 倍提速 |
| T1+T2 解析 | ~56 min | ~27.6 min | 4 进程并行 |
| T3 向量化 | 45.7s | 62.47s | 批次结构不同，略增 |
| T4 写入 | 24.4s | 49.48s | `add → upsert` 换取幂等重跑 |
| 崩溃重跑 | 易 DuplicateIDError | upsert 幂等 | 稳定性提升 |

---

## 9. 工程亮点与关键取舍

### 9.1 PDF 解析：从“能读”到“能用于金融研报”

金融研报常见双栏排版、表格密集、页眉页脚干扰。简单 PDF loader 容易造成左右栏交错、表格断裂、页码丢失。本项目改用 PyMuPDF 读取坐标块，按版面重建阅读顺序；表格由 pdfplumber 抽取后转 Markdown 整块保留，避免字符分块器破坏行列结构。

### 9.2 检索：双路召回 + RRF + reranker

BM25 擅长股票代码、公司名、指标名等精确匹配；向量检索擅长语义近似。两路 Top-K 召回后用 RRF 融合，再交给 Cross-Encoder 精排。召回池从 10 扩到 20 后，减少相关文档在检索阶段被提前丢弃的问题。

### 9.3 语义缓存：只缓存“可信结果”

早期实现中，低相关结果也会被缓存 1 小时，后续相似 query 命中缓存后会跳过新检索，造成错误固化。本项目让 reranker 返回 `top_score`，写入缓存前进行门控：

| top_score | 策略 |
|---|---|
| ≥ 0.5 | 正常缓存 |
| 0 ~ 0.5 | 不缓存，避免低质结果固化 |
| 空结果 | 短 TTL 缓存，允许后续重试 |

### 9.4 工具调用协议：把消息序列合法性作为强约束

DeepSeek / LangChain 工具调用中，复杂工具参数可能进入 `.invalid_tool_calls`，如果路由只看 `.tool_calls`，就会误判“没有工具调用”，导致悬空 tool_call 被写入 checkpoint，下一轮恢复历史时触发多轮 400 死锁。

本项目统一封装 `_ai_tool_call_ids()`，同时覆盖 `.tool_calls` 与 `.invalid_tool_calls`，并在路由、入口净化、异常兜底三层使用同一检测口径。这个问题的教训是：**协议检测必须与实际发送给模型的序列化路径保持一致**。

### 9.5 FastAPI serving：从 demo 接口到可运营服务

这轮服务化改造不仅是“加几个接口”，而是补齐了 LLM 应用上线前常见的工程护栏：

- API versioning；
- request_id 全链路追踪；
- problem+json 统一错误；
- API Key 鉴权；
- 匿名 IP bucket 限流；
- per-user token 预算；
- token 级 SSE；
- 断连取消上游 LLM；
- Prometheus 指标；
- 零真实依赖的回归测试。

---

## 10. 目录结构

```text
react-agent-main/
└── src/
    ├── api/                         # FastAPI 服务层
    │   ├── main.py                  # app 入口、lifespan、路由挂载
    │   ├── settings.py              # 配置校验
    │   ├── errors.py                # problem+json 错误信封
    │   ├── middleware.py            # request_id 日志与 Prometheus 采集
    │   ├── security.py              # API Key 鉴权
    │   ├── ratelimit.py             # Redis 限流与 token 预算
    │   ├── metrics.py               # Prometheus 指标
    │   └── routes/
    │       ├── chat.py              # legacy chat 路由
    │       └── v1/
    │           ├── chat.py          # v1 token 级 SSE / invoke
    │           └── sessions.py      # 会话历史与删除
    ├── react_agent/
    │   ├── core/                    # LangGraph 工作流、节点、路由、状态、checkpointer
    │   ├── rag/                     # PDF 解析、分块、召回、精排、语义缓存、向量库
    │   ├── tools/                   # RAG / Search / Excel / Word / Markdown / SQL
    │   ├── memory/                  # Context 配置
    │   └── utils/                   # LLM、Redis、token、tool helpers
    ├── eval/                        # RAGAS 评测脚本与数据生成
    ├── scripts/                     # PDF 检查、财务抽取、Redis 检查、历史管理
    ├── tests/                       # Streamlit 入口与 API 回归测试
    ├── config.yaml                  # 运行时配置
    ├── pyproject.toml
    ├── requirements.txt
    ├── .env.example
    └── .gitignore
```

---

## 11. 当前边界与后续优化

| 问题 | 当前状态 | 后续方向 |
|---|---|---|
| 图表型研报财务抽取 | 文本 chunk 对图表主导型 PDF 支持有限 | 引入多模态模型逐页识别图表 |
| 评测集指代不明 | 自动生成 QA 会产生“该公司”这类缺实体问题 | 分块时注入公司名 / 行业名等全局元数据 |
| 固定窗口限流 | 窗口边界存在短时 2× 突发 | 生产高并发可升级为滑动窗口或令牌桶 |
| token 预算事后扣费 | 单条超长请求可能先超过预算 | 增加单请求 max_tokens 硬限制 |
| 历史工具链干扰 | 已用 Guard 和跨轮计数缓解 | 将上一轮 tool call chain 压缩为中性摘要 |
| psycopg pool | 隐式 open 有弃用风险 | 改为显式 `await pool.open()` 或上下文管理 |

---

## 12. 更新日志

### 2026-06-28：工具调用协议健壮性

- 新增 `_ai_tool_call_ids()`，统一检测 `.tool_calls` 与 `.invalid_tool_calls`；
- 修复 invalid tool call 悬空导致的多轮 400 死锁；
- 入口净化、路由、异常兜底统一使用同一检测口径；
- 新增连续 HumanMessage 合并逻辑，减少重试堆积造成的角色交替问题。

### 2026-06-26：文档生成层

- 新增 `docx_tool`，支持 Word 研报生成；
- 新增 `md_tool`，支持 Markdown 输出；
- 新增 `_doc_common.py`，复用文档工具公共逻辑；
- `requirements.txt` 新增 `python-docx`。

### 2026-06：FastAPI serving 生产化改造

- 新增 `/api/v1` 路由；
- 实现 token 级 SSE、断连取消、统一错误、request_id、鉴权、限流、预算、Prometheus；
- 新增 API 回归测试，使用 FakeAgent + fakeredis 避免真实依赖。

---

## 13. License

MIT License
