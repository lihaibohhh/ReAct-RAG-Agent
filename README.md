# ReAct Agent — 面向金融研报问答与知识密集型场景的企业级智能体框架

## 这个项目解决了什么问题

金融分析师和研究人员每天面对数百份研报、年报和政策文件，想知道"比亚迪 Q3 毛利率是多少"或"新能源行业最新政策对哪些公司影响最大"，传统做法是逐篇打开 PDF 手动查找，费时且容易遗漏。

这个 Agent 能在数秒内完成跨文档检索、提取关键财务指标、生成对比分析，并**标注来源文档和页码**，确保每条结论可追溯——通过持久化记忆在下次对话时直接延续上次的分析进度，不需要重新描述背景。

**实际使用场景（已验证）：**
- 从数百份金融研报 PDF 中快速定位涉及特定公司、指标或行业的段落，标注来源页码
- 联网检索行业信息，自动整理为结构化 Excel 表格
- 多用户同时使用，各自维护独立的对话历史与分析上下文

---

## Overview
## Architecture

![img_3.png](img_3.png)

- 绿色 → 用户入口（Streamlit）
- 紫色 → Agent 核心（LangGraph ReAct 循环）
- 蓝色 → 三个工具层
- 橙色 → RAG 内部流水线（双路召回→RRF融合→精排）
- 灰色 → 持久化存储

## Screenshots

### 多轮对话与来源溯源
> 用户连续追问，Agent 基于上轮检索结果直接深入分析，每条结论标注来源文档与页码

![img.png](img.png)
![img_1.png](img_1.png)

### RAGAS 自动评测报告
> 50条样本、6个行业分组实测，Context Precision 0.695 / Recall 0.720 / Faithfulness 0.880

![img_2.png](img_2.png)

| 维度 | 说明 |
|------|------|
| **核心价值** | 让金融从业者通过自然语言驱动私有研报知识库检索、联网搜索与结构化输出，每条结论标注来源文档与页码，可追溯、可验证 |
| **架构** | LangGraph `StateGraph` + ReAct 范式（推理 → 工具调用 → 观察 → 反思） |
| **记忆机制** | SQLite / Redis checkpoint 持久化，支持多用户会话隔离与跨会话历史恢复 |
| **缓存层** | Redis Stack 双级缓存：BM25 索引持久化 + 语义查询结果缓存，冷启动与重复查询均有显著提速 |
| **语言优先** | 中文系统提示 / `zh-CN` / `Asia/Shanghai` 全链路中文优化 |
| **容错设计** | 反思节点 + 工具熔断 + 指数退避重试 + 递归步数保护 |
| **成本控制** | Token 统计、历史消息裁剪（防止长会话 Token 膨胀）、工具调用轨迹可视化 |
| **评测体系** | RAGAS 自动评测管道（`eval/run_eval.py`），支持检索 / 端到端双模式，实测 Context Precision 0.695 / Recall 0.720 / Faithfulness 0.880（50 条样本，6 行业分组），双格式报告输出 |

---

## Architecture

系统由五大模块组成，职责清晰、低耦合高内聚：

### Core — 图与状态
- `graph.py`：主工作流图，节点顺序为 `call_model → tools → reflection → call_model`
- `nodes.py`：各节点逻辑实现（模型调用、工具执行、反思推理）
- `routing.py`：动态路由决策，根据工具调用结果决定下一步流向
- `state.py`：扩展 `InputState`，统一管理工具轨迹、工作记忆、引用清单、错误计数、Token 统计等运行时状态
- `prompts.py`：结构化中文系统提示，覆盖工具决策规则、搜索结果处理规范、输出风格约束
- `agent.py`：`PersistentAgent` 包装器，对外暴露简洁的 `invoke` / `stream` 接口，内置历史裁剪逻辑
- `checkpointer.py`：Checkpointer 工厂，支持 Memory / SQLite / PostgreSQL / **Redis** 四种后端，自动降级
- `config.py`：运行时配置加载，含 `RedisConfig` 配置节

### RAG — 私有知识库
- `pdf_parser.py`：**金融研报级 PDF 版面解析**。基于 PyMuPDF 提取带坐标文字块，基于 pdfplumber 提取结构化表格（转 Markdown 完整保留）；内置多栏布局检测与阅读顺序重建、页眉页脚过滤；表格与正文分离处理，避免表格被字符分块器破坏；模块级降级（库未安装）与文件级降级（单文件解析异常）双重兜底，链路不断裂
- `loaders.py`：支持 PDF / DOCX / TXT / MD / CSV / Excel 多格式文件加载；**PDF 分支已从 PyPDFLoader 替换为 `pdf_parser.load_pdf_with_layout`**，返回已完成版面分析和分块的 Document 列表
- `chunker.py`：语法感知分块（Python/Markdown 按语法边界，其他按字符，chunk_size=500）；**新增 PDF 专用路径**：PDF 由 `pdf_parser` 预分块后直接透传，跳过二次 split（防止表格 Markdown 被截断破坏结构）
- `retriever.py`：**BM25 + 向量双路检索**，RRF 融合打分（k=60），合并去重，返回最多 10 条候选；BM25 索引优先从 Redis 加载，命中则跳过重建；HMAC-SHA256 签名校验防 pickle 注入
- `reranker.py`：Cross-Encoder 精排（`BAAI/bge-reranker-v2-m3`），对 RRF 候选文档重新评分；阈值过滤（默认 -5，可通过 `RERANKER_THRESHOLD` 调整）后返回 Top-3；模型异常时降级返回 RRF 排序结果，链路不断裂；支持 `RERANKER_DEBUG=1` 打印逐条得分用于阈值调优
- `vector_store.py`：Chroma 持久化，**文件哈希增量更新**，跳过未变化文件；知识库更新后自动失效 Redis 缓存；内置 T1-T4 各阶段计时埋点，每次建库写入 `ingestion_metrics.jsonl`
- `semantic_cache.py`：**Redis 语义缓存层**，对精确匹配的查询直接返回缓存结果，跳过双路召回与精排全流程；支持 TTL 自动过期与全量缓存清除

### Tools — 工具链
- `search.py`：Tavily Web 搜索，返回 title / url / content / score / published_date
- `rag.py`：调用 RAG 检索管道；**结果包含来源文件路径与页码**（格式：`[来源：xxx.pdf 第 N 页]`，LLM 可直接在回答中引用）；内置拒答机制（无相关内容时明确返回 `has_relevant_content: false`）；命中语义缓存时 `meta.stage` 标记为 `semantic_cache_hit`
- `excel.py`：Excel 表格生成（timestamp / overwrite / append 三种模式），自动美化表头、冻结首行

### Memory — 上下文与持久化
- `context.py`：`Context` 数据类，集中管理所有运行时参数，支持环境变量覆盖

### Utils — 基础设施
- `llm.py`：LLM 模型加载与 provider 解析（openai / anthropic / local / deepseek）
- `redis_client.py`：**Redis 连接池单例**，提供异步（`aioredis`）与同步（`redis`）两个客户端，全项目共用，支持健康检查与连接数配置
- `timer_logger.py`：**建库流水线计时器**，上下文管理器 `timer()` 按阶段埋点，`summarize_last_run()` 打印汇总表并写入 `ingestion_metrics.jsonl`，支持性能分析与面试数据引用
- `token_utils.py`：Token 计数与费用估算，兼容 OpenAI / Anthropic 缓存 token 格式
- `time_utils.py`：时间相关工具函数
- `tool_utils.py`：工具层通用辅助函数
- `tool_helpers.py`：`_ok` / `_err` / `with_retry` 等公共返回结构与重试装饰器

---

## Tech Stack

| 层次 | 技术 |
|------|------|
| **Agent 框架** | LangGraph ≥ 1.0、LangChain ≥ 0.2 |
| **大模型接入** | DeepSeek / Anthropic Claude / OpenAI（`provider/model` 统一格式） |
| **Web 搜索** | langchain-tavily |
| **向量数据库** | Chroma（langchain-chroma） |
| **缓存层** | **Redis Stack**（BM25 索引持久化 + 语义查询缓存，HMAC 签名校验） |
| **PDF 解析** | **PyMuPDF**（版面感知文字提取）+ **pdfplumber**（结构化表格提取） |
| **嵌入模型** | BAAI/bge-small-zh-v1.5（中文优化，可升级至 bge-large-zh-v1.5） |
| **精排模型** | BAAI/bge-reranker-v2-m3（轻量高精度，中英双语） |
| **结构化输出** | openpyxl / pandas |
| **持久化存储** | SQLite / PostgreSQL（psycopg-pool）/ Redis |
| **服务层** | Streamlit（多用户登录 + 会话管理） |
| **运行时** | Python ≥ 3.10，uv 包管理 |

---

## Directory Structure

```
react-agent-main/
├── src/
│   ├── react_agent/
│   │   ├── core/
│   │   │   ├── agent.py              # PersistentAgent 包装器（invoke / stream / 历史裁剪）
│   │   │   ├── checkpointer.py       # Checkpointer 工厂（Memory / SQLite / PostgreSQL / Redis 自动降级）
│   │   │   ├── config.py             # 运行时配置加载（含 RedisConfig）
│   │   │   ├── graph.py              # 主工作流图
│   │   │   ├── nodes.py              # 节点逻辑实现
│   │   │   ├── routing.py            # 动态路由决策
│   │   │   ├── prompts.py            # 结构化中文系统提示
│   │   │   └── state.py              # 运行时状态定义
│   │   ├── memory/
│   │   │   └── context.py            # Agent 运行时可配置参数
│   │   ├── tools/
│   │   │   ├── search.py             # Tavily Web 搜索
│   │   │   ├── rag.py                # RAG 知识库查询（含页码引用 + 语义缓存命中标记）
│   │   │   ├── excel.py              # Excel 表格生成
│   │   │   └── tools.py              # 统一返回结构 + 重试装饰器
│   │   ├── rag/
│   │   │   ├── pdf_parser.py         # 金融研报级 PDF 版面解析（PyMuPDF + pdfplumber，双重降级）
│   │   │   ├── loaders.py            # 多格式文档加载（PDF 分支已替换为 pdf_parser）
│   │   │   ├── chunker.py            # 语法感知分块（新增 PDF 预分块透传路径）
│   │   │   ├── retriever.py          # BM25 + 向量双路检索（BM25 索引 Redis 持久化）
│   │   │   ├── reranker.py           # Cross-Encoder 精排（bge-reranker-v2-m3，含降级保底）
│   │   │   ├── semantic_cache.py     # Redis 语义缓存层（查询结果缓存 / TTL / 失效）
│   │   │   └── vector_store.py       # Chroma 持久化 + 增量更新 + T1-T4 计时埋点
│   │   └── utils/
│   │       ├── llm.py                # LLM 模型加载与 provider 解析
│   │       ├── redis_client.py       # Redis 连接池单例（async + sync 双客户端）
│   │       ├── timer_logger.py       # 建库流水线计时器（阶段埋点 + 汇总 + JSONL 日志）
│   │       ├── token_utils.py        # Token 计数与费用估算
│   │       ├── time_utils.py         # 时间相关工具函数
│   │       ├── tool_utils.py         # 工具层通用辅助函数
│   │       └── tool_helpers.py       # _ok / _err / with_retry 等公共函数
│   ├── scripts/
│   │   ├── check_pdfs.py             # PDF 质检脚本（检测加密、扫描版、乱码、部分扫描、重复文档）
│   │   ├── manage_memory.py          # 会话历史管理（导出 / 统计 / 清理）
│   │   └── verify_redis.py           # Redis 健康检查与缓存状态诊断脚本
│   ├── tests/
│   │   └── test_agent.py             # Streamlit 交互演示（多用户登录 + 持久化会话）
│   │── eval/                         # 评测脚本目录
│   │   ├── __init__.py               # __init__.py（内容是说明文档）
│   │   ├── run_eval.py               # 统一入口：数据加载 → 并发检索 → LLM 生成 → RAGAS 打分 → 双格式报告
│   │   ├── dataset_generator.py      # 数据集生成
│   │   └── dataset/                  # 自动创建，存 eval_dataset.jsonl
│   ├── data/                         # 私有知识库文档目录（放入 PDF 研报后自动向量化）
│   ├── chroma_db/                    # 向量数据库持久化目录
│   ├── outputs/                      # Agent 生成的 Excel 等文件
│   ├── ingestion_metrics.jsonl       # 建库流水线计时日志（每次 build_vector_db 追加一行）
│   ├── config.yaml                   # 项目配置文件
│   ├── migrate.py                    # 数据迁移脚本
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
```

---

## Quick Start

### 1. 克隆与安装

```bash
pip install uv
uv sync
# 或
pip install -r requirements.txt
```

### 2. 安装 PDF 解析依赖

```bash
pip install pymupdf pdfplumber
```

> 若未安装，`pdf_parser.py` 会自动降级到 `PyPDFLoader` 并打印警告，多栏版面和表格解析将失效。

### 3. 启动 Redis Stack

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  -v redis-data:/data \
  redis/redis-stack:latest
```

`8001` 端口为 RedisInsight 可视化面板，浏览器打开 `http://localhost:8001` 可实时查看缓存状态。

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
# 大模型（按需填写）
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Web 搜索（必填）
TAVILY_API_KEY=tvly-...

# Redis（必填，启用缓存层）
REDIS_URL=redis://localhost:6379
BM25_HMAC_SECRET=your-random-secret-string   # BM25 索引签名密钥，任意字符串

# 可选：缓存 TTL 调整
SEMANTIC_CACHE_TTL=3600       # 语义缓存过期时间（秒），默认 1 小时
BM25_INDEX_TTL=86400          # BM25 索引缓存过期时间（秒），默认 24 小时

# 可选：Reranker 调优
RERANKER_MODEL=BAAI/bge-reranker-v2-m3   # 精排模型（默认值）
RERANKER_THRESHOLD=-5                     # 相关性阈值（bge-reranker logit 范围约 -10~+10）
RERANKER_DEBUG=0                          # 设为 1 时打印每条候选得分，用于调优阈值

# 可选：LangSmith 追踪
LANGSMITH_PROJECT=react-agent
```

### 5. PDF 质检（建库前推荐执行）

在向量化之前，建议先用质检脚本筛出无法正常解析的文档：

```bash
cd src
python scripts/check_pdfs.py
```

脚本检测六个维度：加密 PDF、全量扫描版（图片化）、部分扫描（混合型）、疑似乱码（中文字符占比过低）、空白页、重复文档（MD5 去重）。输出示例：

```
质检完成：共 220 份 PDF，发现 3 份问题文件

── 问题文件 ──────────────────────────────────────────
  [全量扫描版]
    某研报_纯图片版.pdf  （12页, 均3字/页）
  [部分扫描(54%页为图片)]
    某年报_混合版.pdf  （48页, 均210字/页）
  [全量扫描版]
    某PPT转PDF.pdf  （32页, 均8字/页）
```

> 扫描版和加密 PDF 无法被 PyMuPDF 提取文字，建议移出 `data/` 目录后再建库。

### 6. 准备私有知识库

将质检通过的 PDF / DOCX / TXT 等文档放入 `src/data/` 目录，Agent 启动时自动完成向量化（增量更新，已处理文件不会重复索引）。

> **金融研报推荐来源**：东方财富 Choice 上市公司年报、卷心菜研究、墨宝研报（可批量下载）；评测集可参考 FinanceIQ / FinQA。

### 7. 验证 Redis 连接

```bash
python scripts/verify_redis.py
```

输出示例：

```
Redis 连接: ✅
当前 RAG 缓存键数量: 3
  rag:bm25_index        TTL剩余: 85925s  |  大小: 8388664 bytes
  rag:bm25_doc_count    TTL剩余: 85925s  |  大小: 64 bytes
  rag:bm25_hmac         TTL剩余: 85925s  |  大小: 136 bytes
```

### 8. 启动 Streamlit 演示

```bash
cd src
streamlit run tests/test_agent.py
```

登录后输入用户名，即可开始对话。同一用户名在任何设备、任何时间登录，均可续接历史对话。

### 9. 运行 RAG 自动评测

```bash
cd src

# 快速验证（3 条 + 检索链路 debug，确认管道正常）
python -m eval.run_eval --n 3 --retrieval_only --debug_retrieval

# 仅检索评测（跳过 LLM 生成，省钱快速）
python -m eval.run_eval --n 50 --retrieval_only

# 端到端评测（检索 + 生成 + Faithfulness 三项指标）
python -m eval.run_eval --n 50

# 关闭 Chroma 遥测噪声日志（推荐）
ANONYMIZED_TELEMETRY=False python -m eval.run_eval --n 50
```

输出两份报告：`eval_summary_<时间戳>.json`（全局 + 分行业均值）和 `eval_detail_<时间戳>.csv`（逐条得分明细）。

---

## Engineering Highlights

### 1. 金融研报级 PDF 版面解析（pdf_parser.py）
针对金融研报普遍存在的双栏排版、嵌入表格、页眉页脚噪声问题，基于 **PyMuPDF + pdfplumber** 实现版面感知解析，完全替代 `PyPDFLoader`：

- **多栏检测与重排**：统计文字块 x 坐标分布，检测双栏之间的空白带（宽度 > 页宽 12%），分别对左右栏独立排序后合并，确保阅读顺序正确
- **表格与正文分离**：pdfplumber 提取表格 bbox，PyMuPDF 提取文字块时排除表格区域；表格转 Markdown 整块保留，不被字符分块器截断
- **页眉页脚过滤**：裁掉页面顶部/底部各 7% 区域，过滤单次出现字符数极少的噪声块（如单个页码）
- **双重降级兜底**：模块级降级（PyMuPDF / pdfplumber 未安装时回落到 PyPDFLoader）+ 文件级降级（单文件 pdfplumber 抛出 PdfminerException 时跳过表格、保留 PyMuPDF 正文），两种场景均打印明确警告，链路不断裂


### 2. 多用户持久化会话（跨重启记忆恢复）
基于 LangGraph SQLite Checkpoint 实现对话状态持久化。每个用户通过唯一 `thread_id`（格式：`user:{username}`）隔离会话，重启服务后历史完整恢复。针对 Streamlit 与 asyncio 的事件循环冲突问题，采用**持久后台 loop + `cache_resource` 生命周期绑定**方案，彻底解决 `asyncio.Lock` 跨 loop 报错。

**重要**：`thread_id` 必须由调用方显式传入，格式为 `user:{username}`；Agent 不会自动生成 thread_id，未传入时会抛出 `ValueError` 以防止多用户数据污染。

### 3. Redis 双级缓存加速（BM25 持久化 + 语义缓存）
**BM25 索引持久化**：首次启动时将 BM25 索引序列化（pickle）后写入 Redis，并附加 HMAC-SHA256 签名防篡改；后续重启直接从 Redis 反序列化，冷启动时间从全量文档重建（2815 文档约 10 秒）降至反序列化（约 1 秒）。知识库更新（`build_vector_db`）后自动清除旧缓存并重置内存单例。

**语义查询缓存**：每次 RAG 查询完成后，将精排结果写入 Redis（key 为 query 的 MD5，TTL 可配置）。相同查询再次发起时直接命中缓存，跳过双路召回与 Cross-Encoder 精排全流程，`meta.stage` 字段标记为 `semantic_cache_hit` 便于可观测性追踪。

### 4. Token 成本控制（历史消息裁剪）
长会话场景下，每轮对话携带的历史 Token 会线性增长。在 `PersistentAgent.invoke` 中内置**预裁剪机制**：每次推理前检查 checkpoint 中的消息数量，超过阈值（默认 20 条）时自动保留最新的 N 条，裁剪操作对调用方完全透明。

### 5. 双路检索 + RRF 融合 + Cross-Encoder 精排（RAG Pipeline）
私有知识库采用 **BM25（关键词精确匹配）+ 向量检索（语义相似度）** 并行召回，通过 **RRF（Reciprocal Rank Fusion）** 融合打分（k=60），使用 `chunk_id` 优先、`page_content[:100]` 兜底的去重策略，合并后交由 `BAAI/bge-reranker-v2-m3` 做 Cross-Encoder 精排，最终仅返回分数最高的 3 条结果。结果携带**来源文件路径与页码**（由 `pdf_parser.py` 在分块时注入 metadata，`rag.py` 取出并拼接到正文前缀），LLM 回答时可直接标注引用。通过拒答机制（`has_relevant_content: false`）避免无相关内容时的幻觉输出。

### 6. Reflection Node — 自愈式工具调用
工具报错时，图流程路由至**反思节点**，由模型分析错误原因并调整参数后重试，同时维护 `consecutive_failures` 计数器，连续失败超过阈值时自动熔断，避免无效循环消耗 Token。

### 7. Checkpointer 工厂（四级自动降级）
`CheckpointerFactory` 统一管理后端初始化：优先尝试 PostgreSQL → SQLite → Redis → MemorySaver，任意一级失败自动降级，使用不同缓存 key 区分"真实成功"与"降级结果"，避免缓存污染。单例模式 + 异步锁保证多并发场景下连接池只初始化一次。

> **注意**：MemorySaver 为最后保底选项，仅用于本地调试，生产环境禁止使用（无清理机制，高并发下存在 OOM 风险）。

### 8. RAGAS 自动评测管道（eval/run_eval.py）

为量化检索管道质量，实现了一套可复现的自动评测系统，覆盖数据准备→检索→打分→报告全链路：

- **数据准备**：从 `.jsonl` 加载问答对，关键词过滤噪声（免责声明、联系方式等），随机采样 N 条保证可复现（固定 seed）
- **并发检索**：`asyncio.Semaphore` + `asyncio.gather` 并发调度，复用线上 `_dual_retrieve`（BM25+向量+RRF）和 `_rerank`（Cross-Encoder）管道，评测结果与线上行为完全一致
- **双模式评测**：`--retrieval_only` 仅测检索质量（省略 LLM 生成，省钱 ~60%）；默认端到端模式额外计算 Faithfulness
- **RAGAS 打分**：Context Precision（检索准不准）/ Context Recall（检索全不全）/ Faithfulness（答案幻觉率），同步适配 RAGAS 0.4.x 字段名变更
- **分行业统计**：按 `industry` 字段分组，样本 < 3 的行业自动跳过，精准定位表现差的行业
- **双格式报告**：JSON 摘要（全局均值 + 分行业均值）+ CSV 明细（逐条得分 + 检索来源），控制台自动诊断建议（Recall 偏低 → 扩大 top_n；Precision 偏低 → 收紧 reranker 阈值）

面试话术：能解释 Context Precision 和 Recall 的计算逻辑差异，以及为什么 Faithfulness 高（0.91）而 Precision/Recall 接近 0 时，问题一定出在检索侧而非生成侧。

---

## 性能基准（实测数据）

### RAG 评测指标（50 条样本，6 行业，端到端模式）

> 测试环境：218 份金融研报 PDF，16788 个 chunk，CUDA GPU，DeepSeek 作为 RAGAS 评委

| 行业 | Context Precision | Context Recall | Faithfulness | 样本数 | 备注 |
|------|:-----------------:|:--------------:|:------------:|:------:|------|
| **全局均值** | **0.695** | **0.720** | **0.880** | 50 | — |
| 互联网电商 | 0.917 | 0.929 | 0.905 | 14 | 文本结构清晰，检索效果最佳 |
| 半导体 | 0.545 | 0.636 | 0.847 | 11 | Precision 偏低，reranker 阈值可适当收紧 |
| 能源金属 | 0.650 | 0.600 | 0.730 | 10 | 部分表格乱码 chunk 影响召回质量 |
| 房地产开发 | 0.500 | 0.500 | 1.000 | 6 | 知识库清理污染 chunk 后 Faithfulness 恢复满分 |
| 教育 | 0.800 | 0.800 | 0.943 | 5 | 修复后显著提升，问题质量对结果影响较大 |
| 电力 | 0.604 | 0.750 | 1.000 | 4 | Faithfulness 满分，检索召回仍有提升空间 |

**结论：** 修复 BM25 双路检索降级、清除知识库污染 chunk（封底/目录页）后，全局指标从 Precision 0.607 / Recall 0.556 提升至 0.695 / 0.720，Faithfulness 0.880。检索管道本身表现良好（互联网电商 Precision 0.917 / Recall 0.929），剩余差距主要来自**评测数据集质量**（部分行业存在"指代不明"问题，见「亟需完善」章节）。

### 向量数据库构建（218 份金融研报 PDF，16771 个 chunk）

| 场景 | 总耗时 | T1+T2 解析 | T3 向量化 | T4 写入 |
|------|--------|-----------|----------|---------|
| 全量建库（首次） | ~57 min | ~56 min（97.7%） | 45.7s | 24.4s |
| 增量更新（5 份新研报） | ~80s | ~77s | ~3s | ~2s |
| 增量更新（1 份新研报） | ~12s | ~1s | ~1s | ~0.5s |

**关键结论：**

- **瓶颈在 PDF 解析层**（T1+T2 占 97.7%），而非向量化。这是双引擎解析（PyMuPDF 多栏检测 + pdfplumber 表格提取）带来的质量代价，属于合理 tradeoff。
- **向量化吞吐约 367 chunk/s**（CPU，bge-small-zh-v1.5），16771 个 chunk 仅需 45.7s，batch_size=5000 设置合理。
- **增量更新场景下 T3 模型冷启动（~9.5s）占主导**，实际文档处理仅需 ~2s。进程复用时（模型已在内存）增量更新总耗时可降至 ~3s。
- 每次建库耗时数据自动追加到 `ingestion_metrics.jsonl`，可用于长期性能追踪与优化对比。

---

## 踩过的主要工程坑

- **PDF 多栏乱序**：金融研报普遍双栏排版，`PyPDFLoader` 底层 pypdf 按字符流顺序提取，左右栏内容随机交错，语义完全错乱。修复方案：替换为 PyMuPDF，利用文字块 bbox 坐标检测双栏分界线，分别排序左右栏后合并，恢复正确阅读顺序。

- **表格被字符分块器破坏**：pdfplumber 提取的表格数据若混入正文字符流，被 `RecursiveCharacterTextSplitter` 在 `|` 符号处截断后，行列关系完全丢失。修复方案：表格单独转 Markdown 整块保留，`chunker.py` 对 PDF 新增预分块透传路径，不再二次 split。

- **Reranker 阈值文档-代码不一致**：注释和 docstring 写的是默认阈值 `-5`，代码实际为 `0.1`，导致大量相关金融段落被静默过滤（bge-reranker 输出原始 logit）。修复方案：阈值统一修正为 `-5`，并清理所有不一致注释；同时修复 Reranker 异常时直接 `return []` 的链路断裂问题，改为降级返回 RRF 结果。

- **Tool description 遗留旧项目内容**：`rag.py` 的 `@tool description` 仍描述数学/PDE 文献场景，Agent 在金融问题路由时可能判断无需调用本工具而直接幻觉输出。修复方案：更新描述为金融研报场景，明确触发条件。

- **页码字段未透传**：`pdf_parser.py` 已将 `page` 字段写入每个 chunk 的 metadata，但 `rag.py` 的两条返回路径（语义缓存命中路径、正常检索路径）均未取出 `page` 字段，导致"标注来源页码"这一核心功能实际失效。修复方案：提取公共函数 `_build_results()`，两条路径统一处理，显式取出 `page` 并拼接到正文前缀。

- **消息重复导致检索质量下降**：启用持久化后，Streamlit 侧传入全量历史 + LangGraph 侧自动恢复历史，导致 Agent 看到两遍相同对话，Token 浪费且认为"已查过"而减少工具调用次数。修复方案：启用持久化后调用侧只传当前新消息。

- **asyncio.Lock 跨 event loop 崩溃**：Streamlit 热重载会销毁旧 loop，但 `cache_resource` 缓存的 Agent 内 SQLite Lock 仍绑定旧 loop，下次请求时报 `RuntimeError`。修复方案：将后台 loop 与 Agent 绑定在同一个 `cache_resource` 中，生命周期一致；`_init_lock` 采用懒创建模式（首次进入 `initialize()` 时才创建），确保锁始终在当前 loop 内创建。

- **BM25 重建阻塞事件循环**：`_get_retriever()` 首次调用时需加载 HuggingFace 模型（CPU 绑定，3~10 秒），直接在 `async` 函数中裸调会卡死整个 asyncio 事件循环。修复方案：用 `asyncio.to_thread(_get_retriever)` 将阻塞操作移出主循环。

- **Redis 连接池跨模块重建**：多处分别 `redis.Redis(...)` 会在高并发时耗尽连接。修复方案：在 `utils/redis_client.py` 集中维护全局连接池单例，全项目通过 `get_async_redis()` / `get_sync_redis()` 取用。

- **RAG 计数全量扫描导致异常触发**：`postprocess_tools` 和 `call_model` 中的 RAG 调用次数统计扫描全量历史消息，长对话中历史累积导致计数虚高（曾复现"78次"异常），误触发上限守卫拒绝正常检索。修复方案：提取公共函数 `_count_rag_in_current_turn()`，以最后一条用户 HumanMessage 为界，只统计当前轮的 ToolMessage 中的 RAG 调用次数，两处逻辑统一。

- **哨兵消息 name 混用致终止机制失效**：`termination_msg`（终止指令）与 `reflection_node`（反思警告）均使用 `name="system_monitor"`，导致 `_extract_recent_tool_messages` 无法区分二者，终止消息被静默跳过。修复方案：终止指令改用 `name="system_terminator"`，过滤条件同步更新为同时跳过两个 name。

- **thread_id 硬编码导致跨用户数据污染**：`context.py` 的 `default_thread_id = "default"` 使所有未传 thread_id 的请求共享同一个 thread，`state.messages` 全局累积，对话内容跨用户互相可见。修复方案：`default_thread_id` 改为 `None`，`agent.py` 中未传 thread_id 时抛出 `ValueError`，强制调用方显式传入 `user:{username}` 格式的 thread_id。

- **问题 PDF 入库前筛查**：部分研报为扫描版（图片化 PDF）或加密 PDF，PyMuPDF 无法提取文字内容。通过 `check_pdfs.py` 脚本对 221 份文档进行质检，检出 3 份问题文件（1 份纯图片 PDF、1 份约 50% 内容为图片的混合版、1 份 PPT 转换的大量图片 PDF），人工审查后移出知识库目录。

- **pdfplumber 文件级异常未捕获**：部分 PDF（`珀斯市中心商务区办公市场：关于西珀特更新.pdf`）（如含双层文字叠压的研报，标题装饰层与文字层坐标偏移叠压）会在 pdfplumber 表格提取阶段抛出 `PdfminerException`，导致整个文件中断入库，而非优雅降级。
  修复方案：在 `pdf_parser.py` 中增加两处文件级 `try/except`——分别捕获 `pdfplumber.open()` 和逐页 `extract_tables()` 的异常，捕获后跳过表格提取、仅保留 PyMuPDF 正文，`ingestion_metrics.jsonl` 记录降级原因。修复后验证：该 9 页 PDF 成功入库 22 个文本块，失败数由 1 降为 0。

- **封底/目录页 chunk 污染全局检索结果**：某份澳大利亚房地产研报（珀斯市中心商务区）的第 7、8 页为封底与联系人页，包含"市场报告""研究""2025年""经济指标"等泛化词汇。向量模型将其编码至语义空间中心位置，对知识库内几乎所有查询都产生虚假高相关性，导致 TikTok 电商、半导体、核电等不同行业的检索结果中均出现该文档，Precision 大幅下降。Cross-Encoder（Reranker）也无法有效过滤，因为这类封底内容语义上确实"和很多话题沾边"。修复方案：通过 `vectorstore._collection.get()` 定位污染 chunk 并直接删除；同时在 `pdf_parser.py` 入库侧增加封底/联系人页过滤规则（检测电话、邮箱、"联系我们"等特征词），防止同类问题重现。这一发现验证了评测管道的价值——评测脚本通过 CSV 明细的 `sources` 列直接暴露了线上系统实际存在的检索质量漏洞。

- **相对路径在不同工作目录下指向不同 Chroma 实例**：`retriever.py` 使用 `./chroma_db` 相对路径初始化 Chroma。PyCharm 直接运行 `eval/run_eval.py` 时，以脚本所在目录 `src/eval/` 为工作目录，导致路径解析为 `src/eval/chroma_db`（空目录），而非实际数据库 `src/chroma_db`，触发"知识库为空，降级为纯向量检索"。Redis 缓存命中时此 bug 完全不会暴露，只有缓存失效重建时才会触发——这是典型的"缓存掩盖底层配置错误"风险。修复方案：改用 `pathlib.Path(__file__).resolve().parent.parent.parent` 从源文件位置计算绝对路径，彻底消除工作目录依赖。

- **评测脚本调用私有 async 函数的静默失效**：`run_eval.py` 初版对 `_dual_retrieve` / `_rerank` 两个私有 async 函数的调用写成了同步方式，Python 不报错，只是返回 coroutine 对象。reranker 拿到的 `docs` 参数是 `<coroutine object>` 而非 `list[Document]`，全链路静默返回空 contexts，跑完没有任何报错但所有检索结果均为空。修复方案：将 `retrieve_for_one` 改为 `async def`，所有调用加 `await`；`batch_retrieve` 从 `ThreadPoolExecutor` 改为 `asyncio.Semaphore` + `asyncio.gather`（async 函数在子线程中没有事件循环会直接崩溃）。

- **RAGAS 0.4.x 字段名破坏性变更导致静默零分**：RAGAS 从 0.2 版本起将所有数据集字段名重命名（`question→user_input`，`contexts→retrieved_contexts`，`ground_truth→reference`，`answer→response`），传旧字段名不报任何错误，但指标全部静默返回接近 0 的分数。实测复现：字段名错误时 Precision/Recall 均为 0.01，修正字段名后立即恢复正常（互联网电商行业 Recall 1.000）。修复方案：统一更新数据集构建代码，使用 RAGAS 0.4.x 规范字段名。

---

## Known Limitations & Roadmap

- `src/react_agent/core/nodes.py` 文件下的**工具信息截断保护**存在较大风险。当 RAG 返回内容过长被截断时，`_parse_tool_payload` 解析失败会静默返回空，导致 `turn_rag_results` 漏计，进而影响拒绝检索的判断逻辑。相关代码段：

```python
turn_rag_results: List[bool] = []
if last_human_idx >= 0:
    for m in msgs_list[last_human_idx:]:
        if (isinstance(m, ToolMessage)
                and getattr(m, "name", None) == "query_internal_knowledge"):
            p, _ = _parse_tool_payload(getattr(m, "content", None))
            if isinstance(p, dict):
                data = p.get("data", {})
                hc = data.get("has_relevant_content") if isinstance(data, dict) else None
                if hc is not None:
                    turn_rag_results.append(bool(hc))
```

> 同脚本中其他与截断保护相关的函数修改时需同步检查。

- `src/react_agent/memory/context.py` 脚本中的所有**参数配置入口**需整体挪到 `config.py` 文件中统一管理，删除 `__post_init__()` 函数（避免直接从 `.env` 文件读取），同时其他脚本中需同步更新。

### 📝 待优化：评测数据集生成的“指代不明（Missing Entity）”问题

#### 🔴 现象描述
在运行 `dataset_generator.py` 自动生成 RAG 评测数据集时，LLM 会生成大量带有模糊代词的废弃 QA 对。
* **错误示例**：“**该公司**2023年第三季度的毛利率是多少？” / “**该项目**的预期投产时间是什么时候？”
* **影响**：这类问题缺乏具体实体（如“中芯国际”），在全量向量库检索时必定会导致召回失败（Hit Rate 极低），严重污染评测指标，无法真实反映检索器的性能。

#### 🔍 根本原因分析 (Root Cause)
这个问题由数据处理链路中的“上下文割裂”导致，主要分为两层：
1.  **Chunking（分块）导致的信息丢失**：在 `process_file` 阶段，长篇研报被物理切分成几百 tokens 的 Chunk。文档开头的实体名词（如“比亚迪”）和后续段落中的代词（如“该公司”）被强行拆散。
2.  **生成阶段的“管中窥豹”**：`dataset_generator.py` 在调用大模型时，Prompt 中只输入了当前的孤立 Chunk。如果当前 Chunk 恰好只包含“该公司”，大模型为了完成任务，只能顺着文本生成带有“该公司”的问题，它无法且无权去跨 Chunk 追溯真实主语。

#### 🛠️ 修复路径规划 (Roadmap)

**阶段一：Prompt 强约束（低成本防守）- *[当前已采取]***
* 在 `dataset_generator.py` 的 `_QA_PROMPT` 中增加极其严厉的“反向约束”。
* 要求 LLM 必须结合 `source`（文件路径/文件名）推断具体公司名称，**严禁**在问题中使用任何模糊指代词。如果无法明确实体，强制要求其放弃生成该条目（返回空数组 `[]`）。

**阶段二：上下文感知分块 (Context-Aware Chunking) - *[根本解决方案]***
* **改造目标**：修改 `src/react_agent/rag/chunker.py`。
* **实现逻辑**：在文本被切分为 Chunk 并写入 ChromaDB 之前，强制将文档的全局元数据（如从文件名中提取的“公司名称”或“文档标题”）**拼接/注入到每一个 Chunk 的 `page_content` 开头**。
* **效果**：让每一个 Chunk 在脱离母体后，依然是一段自包含（Self-contained）的完整信息。这不仅能一劳永逸地解决评测数据集的生成问题，还能极大地提升线上 Agent 真实的检索召回率。


---

## License

MIT License