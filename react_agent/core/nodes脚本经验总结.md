# 凡要持久化的状态,写入前必须校验合法性

## _convert_message_to_dict()

### 详细总结：智能体架构中 `is_last_step` 强制拦截与响应替换的设计原理

在构建基于 LangGraph 和 ReAct 架构的复杂 Agent 时，当执行流到达最大递归步数限制（触发 `is_last_step` 标志），系统必须抛弃大模型原生的 `response`，并使用预设的机械回复进行强行替换[cite: 1]。这一设计并非为了限制模型能力，而是为了保全整个系统底层的工程健壮性。

以下是该设计的三个核心根源及技术细节：

#### 1. 协议层面的致命约束：防止出现“悬空工具调用” (Dangling Tool Calls)
*   **API 强校验**：当前主流大模型（如 OpenAI、DeepSeek 等）的 API 对多轮对话历史有严格的上下文连续性校验。如果发给模型的消息列表中包含了一条带有 `tool_calls` 的 `AIMessage`，那么它的下一条消息**必须**是包含执行结果的 `ToolMessage`。
*   **持久化灾难**：在支持多用户并发和历史记忆持久化（如通过 SQLite 或 PostgreSQL 存储 Checkpoints）的项目中，如果在最后一步把带有 `tool_calls` 的原生响应写入了数据库，但实际上由于步数耗尽并没有真正执行工具，就会形成“悬空的工具调用”。
*   **后果**：当用户发起下一轮追问时，系统会将这段残缺的历史拼接并发送给大模型，直接触发 **HTTP 400 Bad Request** 错误，导致该用户的整个会话 Session 永久卡死。

#### 2. 框架层面的图路由保护：避免状态机异常崩溃
*   **条件边路由 (Conditional Edges)**：在标准的 Agent 图（Graph）定义中，主模型节点（`call_model`）之后的路由逻辑完全依赖于对模型输出的检测。如果 `messages[-1]` 包含 `tool_calls`，路由就会指向工具执行节点；如果不包含，则指向结束节点（`__end__`）。
*   **规避递归溢出**：如果在 `is_last_step` 阶段保留了原生的工具调用请求，LangGraph 的条件边依然会忠实地尝试跳转至工具节点。但此时由于系统已达到 `recursion_limit` 上限，底层框架会直接抛出严重的 `GraphRecursionError`，导致程序异常崩溃。
*   **安全退出机制**：通过强制返回一条纯文本的 `AIMessage`[cite: 1]，实际上是在“欺骗”路由逻辑，使其判定为任务已结束，从而安全、平稳地将状态机引导至 `__end__` 正常退出。

#### 3. 用户体验 (UX) 与可观测性：明确系统边界
*   **原生文本的无意义性**：大模型在决定使用工具时，其文本输出往往是不完整的。它可能完全为空，或者是诸如“*好的，让我来检索一下报告...*”的半截过渡语。将这种信息直接抛给前端用户，会造成逻辑上的断层。
*   **提供确定性的系统状态**：使用写死的机械回复（例如明确告知用户“抱歉，我在限定的推理步数内仍需要调用工具才能给出可靠结论，因此已停止继续搜索/调用工具”[cite: 1]），能清晰地向用户暴露 Agent 当前遇到了什么边界（步数耗尽），并直接提供可行的解决方案（如提示用户提高 `Context.recursion_limit` 或进一步澄清问题[cite: 1]）。

#### 附注：代码逻辑中的连带缺陷（Token 漏计）
*   正是由于在这个分支中执行了强制的中断与写死的字典返回，导致原本在 `call_model` 节点内已经计算好的 Token 消耗与成本（`usage_update`）被直接丢弃，未正确合并到输出中[cite: 1]。这也是为什么在处理此类 Early Return 拦截逻辑时，极易发生系统状态数据丢失的原因。


# ReAct Agent 多轮对话 400 错误 — 完整复盘

## 一、问题现象

第一轮对话正常。当你问"分析委内瑞拉地震并生成 Word 文档"后,**之后每一轮对话都直接崩溃**,抛出:

```
openai.BadRequestError: 400 - An assistant message with 'tool_calls' must be 
followed by tool messages responding to each 'tool_call_id'. 
(insufficient tool messages following tool_calls message)
```

而且是**永久性**的——同一个 thread 怎么发消息都崩,重试只会让情况更糟。

## 二、根本原因(三层)

**第一层 · 病灶来源**
模型在"生成 Word"那一轮发起了 `docx_tool` 调用,但这个工具调用**没有产出对应的 ToolMessage**(工具没执行成功,Word 也确实没生成)。于是历史里留下一条"带 tool_call 却无人应答"的 AIMessage,被持久化进了 Postgres checkpoint。

**第二层 · 死锁机制**
DeepSeek(OpenAI 兼容协议)有一条铁律:带 `tool_calls` 的 assistant 消息,后面必须紧跟与每个 `tool_call_id` 一一对应的 tool 消息。每次新对话恢复历史时,那条悬空的 docx tool_call 都会被发给 DeepSeek → 校验失败 → 400。一旦中招,thread 永久卡死;重试还会因为失败的 HumanMessage 也被持久化,堆积成连续多条 user 消息,雪上加霜。

**第三层 · 为什么极难定位(核心教训)**
这条 docx tool_call 藏在一个 `getattr(msg, "tool_calls")` 和 `additional_kwargs["tool_calls"]` **都读不到**的位置(极可能是流式输出残留的 `tool_call_chunks`)。我们前后做了 v3/v4/v5 五六轮诊断,每次用 `getattr` 读消息属性都显示"序列完全合法、配对全对、id 全对",于是反复误判。直到改用 LangChain **官方的 `_convert_message_to_dict`**(即 DeepSeek 实际收到的那个转换函数)来检测,违规消息才瞬间现形。

## 三、解决过程(为什么走了这么多弯路)

排查路径大致是:怀疑模型幻觉 → 怀疑连续 HumanMessage → 怀疑 tool_calls 配对缺失 → 怀疑数量不匹配 → 怀疑 id 错位 → 怀疑字段畸形 → 最终锁定"读取路径不一致"。

关键转折点有两个:
1. **打印 DeepSeek 返回的完整错误正文**(而不只是看状态码),确认了"确实有悬空 tool_call",排除了一切旁支猜测。
2. **改用官方转换函数检测**——因为我们一直用的 `getattr` 读取路径,和 LangChain 序列化发送时走的路径不是同一条,所以"我们看到的"和"DeepSeek 看到的"始终对不上。这是整个排查最大的认知盲区。

## 四、最终方案

在消息送入模型前(裁剪之后、`ainvoke` 之前)加两道防御函数:

**`_sanitize_dangling_tool_calls`** — 用官方 `_convert_message_to_dict` 检测每条 AIMessage 的真实 tool_calls(和 DeepSeek 看到的一致),为任何缺应答的 tool_call_id 补一条占位 ToolMessage。这既救活了已卡死的 thread,也防御未来。

**`_collapse_consecutive_humans`** — 合并连续的 HumanMessage,清理之前重试堆积的脏数据。

结果:`[sanitize] 补占位 call_00_yjE4...` → DeepSeek 返回 **200 OK**,thread 救活。

## 五、经验教训与应对措施

**1. 检测必须和实际执行走同一条路径。**
这是最贵的一课。你用 `getattr(.tool_calls)` 看到空,不代表序列化时它是空——LangChain 有多条读取路径。凡是要校验"发出去的东西",就用"发出去时实际调用的那个函数"来校验,不要用你以为等价的旁路。

**2. 先拿到最底层、最权威的错误信息,再动手。**
我们前期靠"消息类型序列"猜了很久,其实只要早点打印 DeepSeek 返回的 `response.text`,就能少走一半弯路。排查 API 错误,第一步永远是拿到服务端返回的完整错误正文。

**3. 协议层不变量要在"入口"强制保证,不能依赖上游不出错。**
"带 tool_calls 的 assistant 必须跟 tool 应答"是 OpenAI 协议的硬约束。与其指望工具永远执行成功,不如在送入模型前加一道 sanitize 关卡,把消息序列规整成协议合法状态。这是"确定性代码兜底优于指望上游正确"的典型。

**4. 持久化会把瞬时错误固化成永久故障。**
一个本来只该影响一轮的工具失败,因为被写进 checkpoint,变成了 thread 的永久死锁。凡是要持久化的状态,写入前都应校验其合法性,否则一次意外就污染整条历史。

**5. 工具执行环节必须无条件兜底 ToolMessage。**

## 六、还没做的收尾(治标 → 治本)

当前方案是**事后补救**——在消息发出前修复畸形。但病灶仍在:**`docx_tool` 调用失败时,为什么没有兜底生成一条 ToolMessage?**

真正的根治是在 `dynamic_tool_node` 里:无论工具执行成功还是抛异常,都必须为它的 tool_call_id 生成一条对应的 ToolMessage(失败就写错误内容)。堵住这里,历史里就永远不会再出现悬空 tool_call,`_sanitize` 就退化为纯防御层而非救火队。

建议把这一步做掉,让修复从"治标"变"治本"。需要的话我可以给排查 `dynamic_tool_node` 兜底逻辑的指令。


读完 `dynamic_tool_node`,治本的关键就清楚了。这个节点本身的兜底逻辑**写得相当扎实**——正常路径、禁用工具、混合场景、最外层 except 四条路都为 tool_call 补了 ToolMessage。但有一个**致命前提假设**没成立,正是它放过了 docx 那个悬空调用。

**问题在最外层 except 的兜底:它只看 `state.messages[-1]`。**

```python
last_msg = state.messages[-1]
if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
    return [为 last_msg 的每个 tool_call 补 ToolMessage]
```

它假设"出异常时,带 tool_calls 的 AIMessage 一定是最后一条"。但 docx 那轮的实际情况是:**异常发生时,`state.messages[-1]` 已经不是那条 docx AIMessage 了**(可能后面又追加了别的消息,或异常发生在 ToolNode 内部、state 已被部分更新)。于是兜底的 `if isinstance(last_msg, AIMessage)` 判断为假,直接走 `raise`,docx tool_call 一个 ToolMessage 都没补上 —— 悬空就此产生,被持久化。

还有第二个隐患:正常路径用的是 LangChain `ToolNode(handle_tool_errors=True)`。这个参数能兜住**工具函数内部抛的同步异常**,但如果 docx_tool 是**异步工具**且在 `await` 链路某处异常,或者 ToolNode 自身因为 tool_call 结构异常(比如 args 解析失败)而崩,它可能**不为该 tool_call 生成 ToolMessage 就向上抛**,直接进最外层 except —— 然后又撞上"只看 last_msg"的盲区。

**治本方向:把兜底从"基于 last_msg"改成"基于本轮所有未应答的 tool_call"——找到最近一条带 tool_calls 的 AIMessage,对比已有 ToolMessage,为所有缺失的 tool_call_id 补齐。** 这样无论异常在哪发生、last_msg 是什么,都不会漏。

整段复制给 Claude Code:

---

治本：`dynamic_tool_node` 最外层 except 的兜底只看 `state.messages[-1]`，当异常发生时带 tool_calls 的 AIMessage 不在末尾，兜底失效 → docx tool_call 漏补 → 悬空被持久化。改成"扫描本轮所有未应答的 tool_call 并补齐"。请按步骤执行。

**步骤 1：先看清辅助函数和 import**

```bash
grep -n "_tc_name\|_err\|_bound_tool_messages\|def _get_active_tools\|^from\|^import" src/react_agent/core/nodes.py | head -40
```

把命中行发我，确认 `_tc_name`、`_err`、`ToolMessage`、`AIMessage` 都可用。

**步骤 2：把 `dynamic_tool_node` 的最外层 `except Exception` 块替换为下面这版（基于"未应答 tool_call"补齐，而非 last_msg）**

```python
    except Exception as e:
        # ══════ 兜底：扫描本轮所有未应答的 tool_call，全部补 ToolMessage ══════
        logger.error(f"[dynamic_tool_node] 未预期异常：{type(e).__name__}: {e}", exc_info=True)

        # 找到最近一条带 tool_calls 的 AIMessage（不假设它在末尾）
        target_ai = None
        target_idx = -1
        for _idx in range(len(state.messages) - 1, -1, -1):
            _m = state.messages[_idx]
            if isinstance(_m, AIMessage) and _m.tool_calls:
                target_ai = _m
                target_idx = _idx
                break

        if target_ai is not None:
            # 收集该 AIMessage 之后已存在的 ToolMessage 应答
            answered_ids = {
                msg.tool_call_id
                for msg in state.messages[target_idx + 1:]
                if isinstance(msg, ToolMessage)
            }
            # 只为尚未应答的 tool_call 补占位，避免重复
            missing = [
                ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    name=_tc_name(tc) or "unknown",
                    content=json.dumps(_err(
                        tool_name=_tc_name(tc) or "unknown",
                        query="",
                        code="TOOL_NODE_CRASH",
                        message=f"工具执行层异常：{type(e).__name__}: {e}",
                    ), ensure_ascii=False),
                )
                for tc in target_ai.tool_calls
                if tc.get("id", "") not in answered_ids
            ]
            if missing:
                logger.error(f"[dynamic_tool_node] 兜底补齐 {len(missing)} 条 ToolMessage")
                return {"messages": _bound_tool_messages(missing, ctx.max_tool_output_chars)}
        # 实在找不到可补的，再抛
        raise
```

注意：`ctx` 在 except 块里可能未定义（若异常发生在 `ctx = runtime.context` 之前）。请在 `try:` 第一行之前、函数开头先取 `ctx`，或在 except 里对 `ctx` 做 `ctx = getattr(runtime, "context", None)` 兜底，`max_tool_output_chars` 用 `getattr(ctx, "max_tool_output_chars", 4000)`。请据实调整。

**步骤 3：验证——构造一个 docx_tool 必然失败的场景，确认兜底补齐生效**

最简单的验证：临时在 `docx_tool` 入口 `raise RuntimeError("测试兜底")`，然后发一条"生成word文档"的请求，确认日志出现 `[dynamic_tool_node] 兜底补齐 N 条 ToolMessage`，且不再产生悬空 tool_call、不 400。验证完把临时 raise 删掉。

先做步骤 1、2，结果发我，再验证。这一步做完，悬空 tool_call 从源头就不会再产生，`_sanitize` 退化为纯防御层。

---

# 关于ToolMessage空转报400错误问题与隐患

好，版本对了。现在用你项目的实际版本重跑全部关键测试：测试 8 非常重要。让我进一步确认这个场景在 DeepSeek 实际交互中是否可能发生：**场景 A 和场景 D 都命中了致命差异，而且用的是你项目的实际版本。**

让我把完整结论整理出来：这就是完整的验证结论。

**用你项目实际版本复现成功了。** 致命路径不是我之前推测的"流式残留"或"checkpoint 反序列化异常"，而是一个更现实、更高频的场景：**DeepSeek 生成 `docx_tool` 的复杂嵌套参数时偶发 JSON 格式错误**。

`docx_tool` 的参数结构是所有工具中最复杂的——需要 `title`、`sections` 数组，每个 section 里有 `heading`、`body`、可选的 `table` 嵌套数组。DeepSeek 在生成这种深层嵌套 JSON 时概率性地犯格式错误（漏引号、截断），LangChain 解析失败后把它放进了 `.invalid_tool_calls`，而你的 `dynamic_tool_node` 从正常路径到 except 兜底都只检查 `.tool_calls`，**从头到尾看不到这条 invalid tool call**。没有任何 ToolMessage 被产出，悬空被持久化，thread 永久死锁。

你的"治标"层 `_sanitize_dangling_tool_calls` 之所以能救活 thread，正是因为它用的 `_convert_message_to_dict` 会把 `.invalid_tool_calls` 也转成 `tool_calls` 输出，从而检测到悬空并补上占位 ToolMessage。

需要我基于这个验证结果，给你写"治本"的修复代码吗？核心就是让 `dynamic_tool_node` 的正常路径和 except 块都同时检查 `.tool_calls` **和** `.invalid_tool_calls`。


好问题，让我一步步走完代码的实际执行路径。

## 进入 `dynamic_tool_node` 后的完整执行流程

当 DeepSeek 返回了一条带有非法 JSON arguments 的 `docx_tool` 调用时：

**此时 `state.messages[-1]` 是这样一条 AIMessage：**
- `.content` = "信息非常充分了，现在我来为您生成正式的Word分析报告。"
- `.tool_calls` = `[]` ← 空的
- `.invalid_tool_calls` = `[{id: "call_00_...", name: "docx_tool", args: "残缺JSON", error: "JSONDecodeError..."}]`

**注意：此时还没进入 `make_docx.py`。** JSON 解析失败发生在 LangChain 构造 AIMessage 的阶段（`_convert_dict_to_message`），也就是 `call_model` 节点里 `await model.ainvoke()` 返回的那一刻。`make_docx.py` 从头到尾没有被调用过。

**然后 `call_model` 正常返回这条 AIMessage**，LangGraph 的路由看到这是一条 AIMessage，判断需要走工具节点，于是进入 `dynamic_tool_node`。

**第 467–470 行：**
```python
last_msg = state.messages[-1]  # 那条 AIMessage
if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
#                                      ^^^^^^^^^^^^^^^^
#                                      空列表 → False
```

条件不成立，`blocked_calls` 和 `allowed_calls` 都保持为空列表。

**第 479 行：**
```python
if not blocked_calls:  # True，因为 blocked_calls 是空的
```

进入快速路径。

**第 480–482 行：**
```python
node = ToolNode(active, handle_tool_errors=True)
result = await node.ainvoke(state, config)
```

ToolNode 收到这条 AIMessage，它内部也会检查要执行哪些 tool_call。ToolNode 看到 `.tool_calls` 为空，**没有任何工具需要执行**，于是返回一个空的或不含 ToolMessage 的 result。

**关键：ToolNode 没有抛异常**，所以不会进入 except 块。它只是"什么都没做"就正常返回了。

**第 482 行：**
```python
result["messages"] = _bound_tool_messages(result.get("messages", []), ...)
return result
```

返回了一个 `messages` 为空列表（或不含 ToolMessage）的 result。

**然后 LangGraph 把这个 result 合并进 state**，进入下一步路由。路由看到最后一条消息仍然是那条 AIMessage（因为 `dynamic_tool_node` 没有追加任何新消息），可能判断"还有工具调用未完成"再次路由到工具节点，或者走到 `postprocess_tools`、`reflection` 等节点——但无论走哪条路，**那条 AIMessage 里的 invalid_tool_call 始终没有对应的 ToolMessage**。

**最终这轮对话结束，state 被持久化到 SQLite checkpoint。**

**下一轮对话：**
1. LangGraph 从 checkpoint 恢复历史消息
2. 进入 `call_model`
3. `_prepare_model_messages` 裁剪历史
4. `_sanitize_dangling_tool_calls` 用 `_convert_message_to_dict` 检测每条 AIMessage
5. `_convert_message_to_dict` 检查 `.tool_calls` → 空，再检查 `.invalid_tool_calls` → 有！把它转成 `tool_calls` 输出
6. sanitize 发现有未应答的 tool_call_id，补上占位 ToolMessage
7. 消息序列合法化，DeepSeek 返回 200

**如果没有 `_sanitize_dangling_tool_calls`（你最初出事时的情况）：**
1. 历史原样发给 DeepSeek
2. `_convert_message_to_dict` 把 `.invalid_tool_calls` 转成 `tool_calls`
3. DeepSeek 看到有 `tool_calls` 但后面没有对应的 tool 消息
4. 400 Bad Request
5. 你重试 → 新的 HumanMessage 被持久化 → 连续 HumanMessage 堆积 → 雪上加霜

所以整个故障链中，**`make_docx.py` 从未被执行**。问题发生在更上游：DeepSeek 生成了非法 JSON 参数 → LangChain 解析阶段就把它分流到了 `invalid_tool_calls` → 你的代码全链路都只看 `.tool_calls` → 这条 tool_call 成了"隐形人"。

---
#  新问题与隐患

你提供的 **图片.png** 截图中提到的问题，是使用 LangGraph 开发异步 Agent 时一个非常经典且折磨人的“暗坑”。它本质上是 **Python 异步底层的异常处理设计**与 **LangGraph 状态机严格的消息序列**之间发生的冲突。

我们可以从以下三个维度来详细拆解这个问题：

### 1. 核心矛盾：为什么 `ToolNode` 抓不到这个异常？

这要从 Python 的异常继承树说起。

* **Python 的特殊设计**：在 Python 3.8 之后，官方为了防止开发者在使用 `try...except Exception:` 时不小心吞掉任务取消的信号，将 `asyncio.CancelledError` 的继承父类从 `Exception` 提升到了更高的 `BaseException`（与 `KeyboardInterrupt` 和 `SystemExit` 同级）。这意味着，普通的 `except Exception:` 是无法捕获到协程被取消的错误的。
* **LangGraph 的盲区**：LangGraph 的 `ToolNode` 提供了一个 `handle_tool_errors=True` 的便捷参数，初衷是让工具在报错时不崩溃，而是把报错信息作为字符串返回给 LLM（比如让大模型知道“工具执行出错，请重试”）。但 LangGraph 底层捕获错误用的是 `except Exception:`。
* **冲突结果**：当一个异步工具（比如网络请求、爬虫）因为超时或外部干预被取消时，会抛出 `asyncio.CancelledError`。由于它属于 `BaseException`，直接穿透了 `ToolNode` 的防线，向外抛出。

### 2. 连锁反应：为什么会导致“消息历史进入非法状态”？

LLM 的 Function Calling（工具调用）对消息的排列顺序有严格的协议要求：

1. **AI 发起调用**：`AIMessage(tool_calls=[...])`
2. **系统返回结果**：必须紧跟一个对应的 `ToolMessage(tool_call_id=...)`

当 `CancelledError` 穿透 `ToolNode` 导致流程中断时，工具没能成功产出 `ToolMessage` 写入状态（Graph State）。如果系统尝试捕获这个最外层异常并重试，或者尝试恢复对话历史，就会出现 **`AIMessage` 后面跟着的不是 `ToolMessage**` 的情况。这会直接导致 LLM 端的 API 报错（例如 OpenAI 会提示消息上下文格式不合法），让整个 Agent 的状态机彻底崩溃。

### 3. 实际工程中的应对方案（兜底）

正如截图中提到的“需要额外捕获 `BaseException`”，在实际工程中，不能完全指望 `ToolNode` 的默认容错机制。针对异步工具节点，通常需要进行防御性编程。

标准的兜底姿势是在定义工具或者自定义 Node 时，手动捕获并构造合法的 `ToolMessage`：

```python
import asyncio
from langchain_core.messages import ToolMessage

async def safe_async_tool_node(state, tool_call):
    try:
        # 执行实际的异步工具逻辑
        result = await execute_tool(tool_call)
        return ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    
    except asyncio.CancelledError:
        # 【兜底捕获】专门处理协程取消
        error_msg = f"Tool execution was cancelled (timeout or aborted)."
        return ToolMessage(content=error_msg, tool_call_id=tool_call["id"])
    
    except Exception as e:
        # 处理常规异常
        return ToolMessage(content=f"Tool error: {str(e)}", tool_call_id=tool_call["id"])

```

这种处理方式确保了无论底层发生什么灾难，**消息序列的完整性（AIMessage -> ToolMessage）** 都得到了保证，让 Agent 有机会根据错误信息进行自我纠正或优雅降级。

---

# 经验总结

## 实际工程中对 `invalid_tool_calls` 的考虑

你遇到的问题不是个例——这是一个已经被提交到 LangChain 官方仓库的 bug（issue #33504），提交者明确指出了三个层面的问题：

**第一，LangGraph 的 ToolNode 只处理 `.tool_calls`，完全忽略 `.invalid_tool_calls`。** ToolNode 的 `_parse_input` 方法只提取 `latest_ai_message.tool_calls`，不看 `invalid_tool_calls`。这和你的 `dynamic_tool_node` 的行为一模一样。

**第二，路由逻辑同样只看 `.tool_calls`。** 当 `invalid_tool_calls` 存在时，`tool_calls` 为空，路由判断"没有工具调用"，直接退出循环，模型永远收不到错误反馈。

**第三，旧版 `AgentExecutor` 有这个功能，但新架构丢掉了。** 旧版 `AgentExecutor` 有 `handle_parsing_errors=True` 参数，能自动捕获解析错误并转成错误消息反馈给模型重试。但迁移到新的 `create_agent` 架构时，这个功能没有被移植过来。

### 实际工程中怎么处理的

从这个 issue 和搜索结果来看，成熟的工程项目会考虑以下几层防御：

**1. `invalid_tool_calls` → ToolMessage 转换**

issue 提交者的做法是写一个中间件（`ToolErrorHandlingMiddleware`），在模型返回后检查 `invalid_tool_calls`，把每条 invalid call 转成一条带错误信息的 ToolMessage，让模型能看到错误并重试。这和你的 `_sanitize_dangling_tool_calls` 思路一致，但时机更早——不是下一轮才补，而是当轮立即补。

**2. `asyncio.CancelledError` 兜底**

另一个 LangGraph 的已知 bug（issue #6726）：`asyncio.CancelledError` 继承自 `BaseException` 而非 `Exception`，ToolNode 的 `handle_tool_errors=True` 用的是 `except Exception`，抓不到 `CancelledError`，导致异步工具被取消时不产出 ToolMessage，消息历史进入非法状态。实际工程中需要额外捕获 `BaseException`。

**3. `handle_tool_errors` 的默认值变化**

LangGraph 在 1.0.1 之后把 `handle_tool_errors` 的默认值从 True 改成了 False，导致原本能自动处理的工具异常开始直接传播。你的代码里显式写了 `handle_tool_errors=True`，避开了这个坑。

**4. 协议层不变量的入口保证**

工程上的共识是必须在发送给模型之前做消息序列校验，因为模型无法生成完全合法的输出是语言模型与外部工具交互中固有的失败模式。不能依赖上游全部正确，要在入口强制保证消息序列的合法性。

### 总结：实际项目中还会考虑什么

除了 `invalid_tool_calls`，成熟的 Agent 工程还需要防御：

- **工具参数 JSON 解析失败**（你遇到的核心问题）→ 转成 ToolMessage 让模型重试
- **异步取消（`CancelledError`）**→ 捕获 `BaseException` 而非仅 `Exception`
- **工具执行超时**→ 超时后补 ToolMessage 而非让 tool_call 悬空
- **连续 HumanMessage 堆积**→ 重试失败时合并，你已经做了
- **checkpoint 持久化前校验**→ 写入前确认消息序列合法
- **消息序列发送前 sanitize**→ 你已经做了，但建议也在 `dynamic_tool_node` 内部同步做

你遇到的这个问题，本质上是 LangChain/LangGraph 框架层面的一个已知缺陷，不是你的代码逻辑写错了——而是框架没有处理的边界情况落到了你的 `dynamic_tool_node` 上，而你的代码也沿用了框架同样的盲区（只看 `.tool_calls`）。


# 模型对话协议的约束
这条"另一条约束"指的是**消息角色必须交替出现(role alternation)**。让我先确认一下 DeepSeek 当前的具体行为,再给你讲清楚原理。就是**消息角色必须严格交替(role alternation)**——user 和 assistant 必须一来一回地交替出现,不能连续出现两条同角色的消息。

这是和"悬空 tool_call"完全独立的另一条协议约束。前一条管的是"AIMessage 的 tool_call 必须有 ToolMessage 应答",这一条管的是"整个对话的角色排布必须是 user → assistant → user → assistant 这样交替"。

## DeepSeek 的具体行为

DeepSeek 在这一点上比 OpenAI 严格得多。它的 R1(reasoner)模型有比 Claude/OpenAI 更严的角色排序要求,一旦历史里出现连续同角色的消息(比如 user-user),就会被直接拒绝。触发的是一个 400 `invalid_request_error`,报错正文大意是:deepseek-reasoner 不支持连续的 user 或 assistant 消息(指出是 messages[X] 和 messages[Y]),要求你把 user/assistant 消息交错排列。很多人在 Cursor 等工具里用 deepseek-reasoner 时都撞过这个错。

需要区分一下:这条**严格交替**的硬约束主要是 `deepseek-reasoner`(R1)在执行;`deepseek-chat`(V3)通常更宽容一些。但角色交替本身是整个 Chat Completions 协议的通用预期,而且这是 DeepSeek 自己做的设计决定,框架层"修不了",只能在发送前把历史规整成交替结构。你既然写了 `_collapse_consecutive_humans` 来清理,说明在你的实际环境里它确实成了问题。

## 为什么重试会踩中它

把链路接回上一条悬空的故事:

1. 第一轮 docx 那条悬空 tool_call 已经让 thread 卡死,每次发消息都 400。
2. 你以为是网络抖动或偶发问题,于是**重试**——重发一条新的 `HumanMessage`。
3. 但上一条 `HumanMessage` 对应的那一轮**根本没产出合法的 AIMessage 回复**(因为它在悬空校验那一步就 400 了,模型没正常应答)。
4. 于是历史里变成:`HumanMessage(第一次) → HumanMessage(重试) → ...`,**中间缺了本该隔开它们的 AIMessage**。

每重试一次,就往历史里多堆一条孤立的 user 消息。原本只有一个 400(悬空 tool_call),现在又叠加了第二个 400(连续 user)。这就是"雪上加霜"——一个故障,触发了**两条不同协议约束**的违规,而且都被持久化进了 checkpoint,越重试越脏。

## 处理方式

应对和悬空是同一套思路——**在入口强制规整**:

- `_collapse_consecutive_humans` 做的事就是把连续的 `HumanMessage` 合并成一条(把内容用换行拼起来),让序列恢复成合法的交替结构。这正是搜索结果里反复出现的标准解法:把连续同角色的消息合并成单条,例如两条 user 消息合并为一条。
- 这是"治标"——它清理已经堆积的脏数据、救活卡死的 thread。"治本"则是从源头保证:**一轮对话即便失败,也不要把那条没得到应答的 `HumanMessage` 单独留在历史里**(要么补一条占位 AIMessage,要么失败时回滚不持久化)。

所以这两条约束本质是同构的失败模式:**无状态 API 每轮重放全部历史 + 持久化会固化错误 → 任何一次违反协议不变量的消息都会变成永久 400**。区别只在违反的是哪条不变量——一个是 tool_call 配对,一个是角色交替。两者都得在"消息发出去之前"那道关卡上强制规整。