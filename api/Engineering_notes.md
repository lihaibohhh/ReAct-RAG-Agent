# FastAPI Serving 层改造 · 工程纪要

> 一份双用途文档:既是这次改造的工程记录,也是面试时照着讲的脚本。
> 每个 Phase 按「起点 → 关键决策/踩坑 → 验证证据 → 面试话术」组织。
> 文末有跨 Phase 的系统性故事、高频追问的备好答案、诚实的局限清单,以及一张实测数字速查表。
>
> **怎么用**:面试前通读一遍;被问到某个 Phase 时,讲「踩坑」和「验证证据」那两块——具体的 bug 和具体的数字最能压场,泛泛的「我用了 FastAPI」最廉价。

---

## 0. 一句话定位

给一个金融研报 ReAct agent(RAG + 多工具)的 FastAPI serving 层做了一轮生产级硬化,把「能跑的 agent 接口」做成「鉴权 → 限流/预算 → token 级流式 + 断连取消 → 统一错误 → request_id 日志 → LLM 风味可观测 → 回归测试 + 零烧钱 CI」的全链路,每一环都有真实实测撑着,不是只看代码对不对。

**全景一行**:
`请求 → API-Key/匿名IP 鉴权 → Redis 限流 + 当日 token 预算 → /api/v1/chat/stream(token 级 SSE,断连即取消上游 LLM) → done 帧产出 ttft/tokens/cost → Prometheus 同源采集 → problem+json 错误 + request_id 全链路`

**贯穿全程的护栏**:全进程单一 `PersistentAgent` + 单一 checkpointer;工具统一 `_ok`/`_err` 返回;入口 `_sanitize_dangling_tool_calls` + `_ai_tool_call_ids` 防悬空 tool_call;不破坏 Streamlit 入口;旧 `/chat/*` 作为冻结的 legacy 保留。

---

## Phase 0 · 地基:配置校验 / 版本化 / 统一错误 / 请求追踪

**起点**:基础 FastAPI——CORS 全开、全局兜底返回 `{"detail": "..."}`、无 request_id、无结构化日志、无版本化。

**做了什么**:新增 `api/settings.py`(pydantic-settings)、`api/errors.py`(RFC 7807 problem+json)、纯 ASGI 中间件(request_id + JSON 日志);路由挂 `/api/v1`,旧 `/chat/*` 保留并打 `Deprecation` 头。

**关键决策 / 踩坑**
- **fail-fast 不能写在模块 import 时 `sys.exit()`**。`settings.py` 做成零副作用 import,fail-fast 挪到 `main.py` 构造 app 处。原因:Phase 5 的 pytest 会 import 这个模块,import 期 `sys.exit` 会直接杀掉测试进程——一个「为了健壮反而毁掉可测性」的反模式。
- **key 校验 provider-aware**,不硬编码 DeepSeek/OpenAI。读当前配置的 provider(openai/anthropic/local/deepseek),只要求对应那把 key,local 跳过。否则只配 Anthropic key 的人会被误杀。
- **中间件用纯 ASGI,不用 `BaseHTTPMiddleware`**。后者会缓冲响应体、直接破坏 SSE 流式。这个选择是后面 Phase 1/4 流式和指标采集都能成立的前提。
- **一处 wrap 两用**:在纯 ASGI 里包一层 `send` 抓 `http.response.start`,同一个点既拿到 status 打 summary 日志,又注入 `X-Request-Id` 响应头。
- uvicorn 自带 access log 纳入同一套 JSON 配置,消除「一半 JSON 一半纯文本」的混合日志。

**验证证据**:缺 LLM key → 进 lifespan 前带可读原因退出;任意 4xx/5xx body 是 problem+json 五字段且响应头同带 `X-Request-Id`;旧路径可用且带 `Deprecation: true`;新路径无 Deprecation;日志全 JSON、同一请求全程 request_id 一致。

**🎤 面试这样讲**:「我做错误处理时统一成了 RFC 7807 的 problem+json,并且让 request_id 从中间件一路贯穿到错误体和日志,排障从『翻日志大海』变成『按 id 直达』。中间件我特意用纯 ASGI 而不是 BaseHTTPMiddleware,因为后者会缓冲响应、把我后面的 SSE 流式吃掉——这是个常见但隐蔽的坑。」

---

## Phase 1 · token 级流式 + 断连即取消 ⭐

**起点**:已有节点级 SSE(一个节点跑完才推一帧),要升级成逐 token 的打字机效果。

**做了什么**:`PersistentAgent.stream_events()` 基于 `astream_events(version="v2")`;SSE 事件 taxonomy:`token` / `usage` / `tool_call` / `tool_result` / `done` / `error`;Queue + Task 架构解耦 producer 与 SSE 输出;手动 deadline(兼容 Python 3.10);客户端断连即取消上游。

**关键决策 / 踩坑(这是 Phase 1 的灵魂,面试重点讲)**
- **头号 bug——token 不冒泡**。根因:Python 3.10 的 async 下,callback 不走 contextvar 自动传播,内层 `model.ainvoke` 收不到 `astream_events` 装的回调句柄,`on_chat_model_stream` 自然出不来。**修法**:在节点函数 `call_model` 签名里显式声明 `config: RunnableConfig`,框架才会注入,再手动 `config=config` 透传给 `model.ainvoke`。这是 LangChain 官方点名的 3.10 async 限制,定位到这一行靠的是用 `GenericFakeChatModel` 做探针确认事件有没有冒出来。
- **断连即取消,而且要真掐掉上游**。Queue+Task 架构有个陷阱:客户端断开后 generator 不再读队列,但 **producer task 不会因为没人读就自己停**——「我不发了」≠「LLM 停了、不计费了」。正确做法:`finally` 里对 producer task **`cancel()` 且 `await`**(吞掉 CancelledError),让取消信号真传进 `astream_events` 的 `aclose` 一路到底层 httpx。
- **第二个 bug——cost 静默归零**。用 `ctx.model`(配置别名)当 key 查价表会 miss,静默返回 0、不报错(最阴险的一类 bug)。改成读 `response_metadata.model_name`(API 回传的真实型号)。
- **第三个 bug——streaming 下用量抽空**。`_extract_deepseek_v4_usage` 的标准字段在流式模式拿不到,加了读 `usage_metadata` 的 fallback。非流式测过不代表流式没问题,两种模式 usage 格式不同。
- `X-Accel-Buffering: no` 关掉 Nginx 缓冲,否则反代攒一批才下发,逐 token 效果在生产消失;Queue `maxsize=100` 做背压。

**验证证据(真实 key、含工具调用,非 fake/非 401)**
- `TOOL_CALLS: [query_internal_knowledge, query_internal_knowledge, search]`(真实多跳)
- `USAGE_FRAMES`:prompt/completion token 非零、cost 非零
- `DONE_FRAME`:`ttft_ms=6543, total_ms=34174, total_tokens=22579, total_cost=0.00321`
- **断连证据看 LangSmith**:断连请求的父 run `status=error`,其 ChatOpenAI 子 run `status=pending, end_time=None`——上游被掐在半路,而非跑完。这是「断连真省钱」唯一过硬的证据。

**🎤 面试这样讲**:「流式接上之后 token 死活不冒泡,我用 fake model 做探针逐层排查,定位到是 Python 3.10 async 的 callback 传播问题——节点不声明 config 参数,框架就不会把回调注入内层模型调用。另外我做了断连即取消:难点不是『前端断了我停推』,而是要保证上游那个还在烧钱的 LLM 调用真被掐掉,所以 finally 里是 cancel **加 await**,我用 LangSmith 确认了断连后那条 run 停在 pending、没跑完。」

---

## Phase 2 · 鉴权 + 限流 + per-user 成本预算

**起点**:多用户却无 auth,session_id 纯靠信任传入;LLM 端点贵,需要防滥用、控成本。

**做了什么**:`api/security.py`(API-Key,支持 `X-API-Key` 和 `Authorization: Bearer`);`api/ratelimit.py`(Redis 固定分钟窗口限流 + per-user 当日 token 预算,事后扣费,计数器 TTL 25h 跨零点自动过期)。

**关键决策 / 踩坑**
- **核心安全修复——匿名不能裸跑**。初版「`API_API_KEY` 未配置 → 空 key → 免鉴权」叠加「空 key 跳过预算/限流」,后果是默认配置下任何人可无限调用、不计成本。修法:空 key **回落到 `client:{IP}` bucket**(优先 XFF 第一跳,否则直连 IP)做限流和预算的隔离键。把它写成**有意的匿名降级策略**,而不是「空 key 就跳过」——面试官问起,「匿名流量回落到 IP 维度限流」是设计,「空 key 不限了」是漏洞。
- **fail-open 降级**:Redis 宕 → 限流/预算**放行** + warning 日志,绝不 fail-closed 把主链路自己锁死。
- **扣费在 `done` 之后、按真实 token**:`record_token_usage` 放 generator 的 `finally` 末尾,断连/超时/正常三条路径都按**实际产生的** token 记,不按预估。
- `Retry-After` 走 `AppError.headers` 透传,不另开返回路径,和 problem+json 保持同构。
- **诚实对齐文档**:限流是**固定分钟窗口**(非滑动),注释/README 如实写,不写「滑动窗口」;预算是**事后扣费**,单条超大请求能冲顶上限,标为有意取舍。

**验证证据**:`STATUSES: [200, 200, 200, 429, 429]`——匿名(空 key)用 `client:IP` bucket 同样在第 4 次被限流;不同 IP 桶隔离;Redis 宕两条路径都 fail-open + warning。

**🎤 面试这样讲**:「限流我做了两件容易被忽略的事:一是匿名流量——很多实现里没配 key 就直接放行,我让它回落到 IP 维度的 bucket,匿名也吃限流和预算;二是 Redis 挂掉时我选 fail-open,限流组件不该把主业务带死。预算是按 LLM 实际消耗的 token 事后扣的,断连没跑完的请求也按已产生的量记。」

---

## Phase 4 · Prometheus 可观测(LLM 风味)

**起点**:Phase 1 的 ttft/tokens/cost、Phase 2 的 429/401 都是「算完就扔」,要接进 `/metrics`。

**做了什么**:`/api/v1/metrics`(豁免 auth、`include_in_schema=False`、自身排除出 `http_requests_total`);8 个指标——HTTP 请求数/延迟/in-flight + `llm_ttft_seconds` / `llm_tokens_total{type,model}` / `llm_cost_usd_total{model}` / `rate_limit_rejections_total{reason}` / `auth_failures_total`。

**关键决策 / 踩坑**
- **path label 必须用路由模板,不能用原始 URL**——否则 session_id 进 label 会指标基数爆炸。踩到版本差异:**Starlette 0.52.1 的 `scope["route"]` 不被填充**,改用 `scope["endpoint"]` 反查模板(`register_routes` 建映射 + `get_path_template`)。这是「只能真拉一次 /metrics 看 label 是不是 `/api/v1/chat/stream`」才能发现的坑。
- **Histogram 自定义 buckets**。默认桶顶 10s,但实测 TTFT 6.8s、总时长 34s,真实请求会全落进 `+Inf` 桶、p95/p99 失效。给 TTFT 加 7.5(到 ~30s)、给 duration 加 20/45(到 ~60s)。按真实分布调桶,体现「知道自己在量什么」。
- **指标与计费同源**:LLM 指标直接复用 done 帧已算好的 ttft/tokens/cost,model label 用同一个 `response_metadata.model_name`,三处口径(done 帧 / Prometheus / 扣费)对齐,不重算。
- **in-flight 在 `finally` 减回**,断连/超时也减,否则只增不减泄漏。

**验证证据(本轮实测质量最高)**
- **cost 三处对账**:Counter 增量 `0.00055552` → `round(6)` = `0.000556` = done 帧 `total_cost`。同一个 Python 变量,Prometheus 拿原始 float、done 帧做 6 位展示截断,无口径分裂。
- path label 实测为 `/api/v1/chat/stream`(模板,非 unmatched)。
- in-flight:流式中 2.0(stream + scrape)→ 断连后稳定 1.0(只剩 scrape 自己)→ 无泄漏。

**🎤 面试这样讲**:「上指标踩了两个真实的坑:一是 path label,我那个 Starlette 版本的 `scope['route']` 是空的,得改读 `scope['endpoint']` 反查模板,不然 session_id 进 label 直接基数爆炸;二是默认直方图桶顶才 10 秒,我这种 agent 请求动辄三十几秒全进 +Inf 桶、分位数就废了,得按真实分布定制桶。我还专门对账了 cost——done 帧、Prometheus、扣费三处必须是同一个数,面试问『你这成本数字哪来的』我能一路追到同一个变量。」

---

## Phase 5 · 测试 + 零烧钱 CI

**起点**:前面所有验证都是手动一次性的,改代码碰坏了没东西拦得住。

**做了什么**:33 条 pytest(httpx AsyncClient + asgi-lifespan),82% API 覆盖;`.github/workflows/ci.yml`(uv sync → ruff → pytest)。

**关键决策 / 踩坑**
- **CI 零烧钱是硬约束**。四层一起保证不碰真实 key/服务:`FakeAgent`(dependency_overrides + `_agent_instance` patch)产可控事件流、能中途 raise CancelledError 模拟断连;`fakeredis`(patch `get_async_redis`)接管限流/预算/计数;假 LLM key 写进 CI env 但不触发真实调用;`@pytest.mark.integration` 标真实用例、CI 默认 `-m "not integration"` 跳过。
- **测试 = 把前几轮的真实结论钉成回归用例**,不只测 happy path:断连取消(断言 cancel+await + 按实际 token 扣费)、超时(error+done 收尾不吊死)、匿名 IP-bucket 限流(安全漏洞守门用例)、fail-open、in-flight 不泄漏、path 模板、problem+json 五字段。

**验证证据**:`33 passed`,API coverage 82%(目标 ≥70%);CI 在 `DEEPSEEK_API_KEY=sk-ci-fake-key`、无任何真实 secret 下绿。

**🎤 面试这样讲**:「我把前面手动验过的每条边界——断连取消、匿名限流、fail-open、in-flight 不泄漏——都钉成了回归测试。CI 我特别保证零烧钱:用假 agent 和 fakeredis 把真实依赖全替换掉,跑 33 条不调一次真实 LLM、不连一次真实 Redis,所以 CI 既挡回归又不花钱。被问『你怎么保证这些不回归』,我的答案是一张测试表,不是嘴上保证。」

---

## Phase 3 · 会话资源 REST 端点 + IDOR 防护

**起点**:对话历史散在 checkpointer 里,缺会话管理端点;一旦暴露,越权访问(IDOR)是头号风险。

**做了什么**:`GET /api/v1/sessions/{id}/history`(分页、脱敏只返 type+content)、`DELETE /api/v1/sessions/{id}`(204,支持 SQLite/MemorySaver,不支持的后端返 501)、`GET /api/v1/sessions`(固定 501,checkpointer 无跨 thread 枚举 API)。

**关键决策 / 踩坑**
- **IDOR 防护(核心)**:thread_id 从 `user:{session_id}` 改为 `user:{bucket_key}:{session_id}`,**chat 的两个 handler 同步改**(只改 CRUD 不改 chat 会导致读写命名空间对不上、历史全查不到)。隔离逻辑是「**从源头让越权查询落到一个不存在的命名空间**」:B 用自己的 bucket_key 拼出来的是另一个 thread_id,自然查不到 → 404。比「先查到别人数据、再用 if 判断属不属于你」安全得多(后者最容易漏)。
- **404 而非 403**:越权统一返 404,不泄露「这个会话存在但不属于你」的存在性信息。
- 不支持枚举的后端老实返 501,不假装返空列表(那是骗人)。

**验证证据**:11 条新用例(S1–S8),含「key B 拿 key A 的 session_id 打 404」的安全守门用例;全套 44 条全绿、零回归——改了核心 thread_id 拼接却没碰坏前五个 Phase,正是 Phase 5 回归网在起作用。

**⚠️ 待验证(诚实标注,别当已完成讲)**:用新的 GET history 端点验「断连 → 半截 checkpoint → 下一轮自愈」闭环——断连一条正在调工具的请求,查 history 看是否留下无对应 tool_result 的悬空 tool_call,再用同 session_id 发下一条,确认不炸 400、入口 `_sanitize_dangling_tool_calls` 生效。验过之后这条才能作为完整闭环讲。

**🎤 面试这样讲**:「会话端点最大的风险是越权,我没有用『查到再判断归属』那种容易漏的写法,而是从命名空间源头隔离:path 里的 session_id 强制套上调用者的 bucket_key,别人的 id 拼出来就是个不存在的 key,根本查不到,而且统一返 404 不返 403,连『这个会话存在』都不泄露。」

---

## 跨 Phase 的系统性故事(面试最压场的部分)

单个 feature 谁都能讲,**跨 Phase 的闭环**才显系统性思维。准备这几条:

**1. 「断连省钱」与「半截状态自愈」是同一条 session 上的同一套机制**(Phase 1 + Phase 3 + 既有入口)
断连 → 上游 LLM 被 cancel+await 真掐掉(省钱,LangSmith pending 为证)→ 但这一步 checkpoint 没落、可能留下悬空 tool_call → 下一轮同 session 进来,入口 `_sanitize_dangling_tool_calls` 自愈、不炸 400。一个动作的省钱面和它的副作用善后,被同一套设计兜住。*(注:自愈这半段待 Phase 3 后真验,见上。)*

**2. cost 口径三处对账**(Phase 1 → 2 → 4)
done 帧展示、per-user 预算扣费、Prometheus counter,三处必须是同一个变量、同一个真实 model 名。这条能扛住「你这成本数字可信吗」的追问——一路追到同一个 Python float。

**3. 配置别名 ≠ 运行时真实名**(Phase 1 的 cost 归零 bug)
用配置里的 model 别名查价表会静默 miss → cost=0 不报错。教训:凡是「配置名」和「上游真实回传名」可能不一致的地方,计费/计量一律以回传为准。这是个能体现 debug 嗅觉的小故事。

**4. 默认配置就得是安全的**(Phase 2 匿名双修)
「没配 key 就免鉴权 + 空 key 跳过限流」叠起来 = 默认配置下完全裸奔。安全不能依赖「记得去配」,默认态必须有兜底(匿名回落 IP bucket)。

---

## 高频追问 · 备好的答案

- **「XFF 能伪造,匿名按 IP 限流不就被绕了?」** → 对。生产里 `X-Forwarded-For` 必须由可信反代(Nginx/ALB)覆写、只信代理注入的那一跳,不能裸信客户端 header。demo 架构没有反代所以直读第一跳,这是已知边界。
- **「断连后 checkpoint 会不会留半截状态?」** → 会。LangGraph 在节点边界落 checkpoint,中途取消通常意味着当前节点没提交;边角是已提交的 tool_call 没等到 tool_result,留下悬空。入口的 `_sanitize_dangling_tool_calls` 正是兜这个的。
- **「固定窗口限流的边界双倍问题知道吗?」** → 知道。59.9s 和 60.1s 各打满 = 一秒内 2× rpm。当前够用,要更严就上滑动窗口或令牌桶。我在文档里如实写了是固定窗口,没含糊成滑动。
- **「事后扣费,单条天价请求不就冲超预算了?」** → 是有意取舍。LLM 没法精确预扣(生成前不知道会吐多少),要硬限可加「单请求 token 上限」兜底。
- **「为什么中间件不用 BaseHTTPMiddleware?」** → 它会缓冲响应体、破坏 SSE 流式。我全程用纯 ASGI,这也是 token 级流式和指标采集能共存的前提。
- **「覆盖率 82% 是行还是分支?」** → 目前是行覆盖;fail-open 这种异常分支建议再跑 `--cov-branch` 确认分支覆盖,这是个诚实的待办。

---

## 诚实的局限 / 待办(面试官欣赏知道系统边界的人)

- Phase 3 的「断连半截状态自愈」闭环**待真验**(端到端那一跑还没做)。
- 测试 43.5s / 33 条偏慢,断连/超时用例有真实 sleep,可用 `--durations=10` 定位后换可控时间推进。
- 覆盖率为行覆盖,分支覆盖未单独测(`--cov-branch`)。
- 限流为固定分钟窗口,有边界双倍问题;预算为事后扣费,单条可冲顶。
- 匿名 IP 限流依赖可信反代覆写 XFF,demo 架构未含反代。
- `GET /sessions` 枚举返 501,受限于 checkpointer 后端无跨 thread 枚举 API。

---

## 实测数字速查表(临场可直接引用)

| 指标 | 值 | 出处 |
|---|---|---|
| TTFT(真实多跳问答) | 6543 ms | Phase 1 done 帧 |
| 总时长 | 34174 ms(~34s) | Phase 1 done 帧 |
| 总 token(一次问答) | 22579 | Phase 1 done 帧 |
| 总成本 | $0.00321 | Phase 1 done 帧 |
| 单帧 usage 示例 | prompt 3900 / completion 63 / $0.00056 | Phase 1 |
| cost 三处对账 | Counter 增量 0.00055552 → round(6) 0.000556 = done 帧 | Phase 4 |
| 限流 demo(rpm=3) | `[200, 200, 200, 429, 429]` | Phase 2 / 匿名同样触发 |
| 测试数 | 33(Phase 5)+ 11(Phase 3)= 44,全绿零回归 | — |
| API 覆盖率 | 82%(目标 ≥70%) | Phase 5 |
| Histogram 桶定制 | TTFT 加 7.5(→30s)、duration 加 20/45(→60s) | Phase 4,因实测 TTFT 6.8s / 总时长 34s |
| 关键版本 | Python 3.10.19、Starlette 0.52.1 | 两者都对应一个真实踩坑 |
| CI | 零 secret / 零真实 LLM / 零真实 Redis,绿 | Phase 5 |

---

*纪要随项目演进更新。Phase 3 的自愈闭环真验后,把「待验证」标记移除并补入验证证据。*