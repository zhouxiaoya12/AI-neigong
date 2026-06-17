# Claude Agent SDK：子 Agent 与会话存储

> Claude Agent SDK 是 Claude Code harness 的库形态。内置工具、子 Agent 用于上下文隔离、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是用于长期异步工作的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 01（Agent 循环）、Phase 14 · 10（技能库）
**用时：** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子 Agent——并行化与上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个基于标准库的 harness，包含内置工具、带隔离上下文的子 Agent 生成、生命周期钩子和会话存储。

## 问题

一个原始 LLM API 给你一个回合。一个生产 Agent 需要工具执行、MCP 服务器、生命周期钩子、子 Agent 生成、会话持久化、追踪传播。Claude Agent SDK 以库的形式提供这个形态——与 Claude Code 使用的 harness 相同，暴露给自定义 Agent。

## 概念

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你自己拥有循环、工具、状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子 Agent 生成、会话存储。Claude Code 的循环作为一个库。

### 内置工具

SDK 开箱即用提供 10+ 工具：文件读/写、shell、grep、glob、网页获取等。自定义工具通过标准工具 schema 接口注册。

### 子 Agent

Anthropic 文档说明了两个目的：

1. **并行化。** 并发运行独立工作。"为这 20 个模块中的每一个找到测试文件"是 20 个并行子 Agent 任务。
2. **上下文隔离。** 子 Agent 使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得到保护。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子 Agent 记录。

### 会话存储

协议上与 TypeScript 对等：

- `append(session_id, message)`——添加一轮对话。
- `load(session_id)`——恢复对话。
- `list_sessions()`——枚举。
- `delete(session_id)`——级联删除子 Agent 会话。
- `list_subkeys(session_id)`——列出子 Agent 键。

`--session-mirror`（CLI 标志）将对话记录镜像到外部文件，随流式传输实时写入，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse`——门控或审计工具调用。
- `SessionStart`、`SessionEnd`——设置和收尾。
- `UserPromptSubmit`——在模型看到之前处理用户输入。
- `PreCompact`——在上下文压缩之前运行。
- `Stop`——Agent 退出时清理。
- `Notification`——侧信道告警。

钩子是 pro-workflow（Phase 14 课程参考）和类似系统添加横切行为的方式。

### W3C 追踪上下文

调用方上活跃的 OTel span 通过 W3C 追踪上下文头传播到 CLI 子进程中。整个多进程追踪在你的后端呈现为一条追踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长期异步工作、内置提示缓存、内置压缩。用控制权换取托管基础设施。

### 此模式在何处会出错

- **子 Agent 过度生成。** 为 100 个微小任务生成 100 个子 Agent。开销主导一切。批量处理替代。
- **钩子膨胀。** 每个团队都加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建

`code/main.py` 基于标准库实现 SDK 形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`——私有上下文，隔离运行，结果返回。
- `SessionStore`——append、load、list、delete、list_subkeys。
- `Hooks`——`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 Agent 并行生成 3 个子 Agent（每个隔离），聚合结果，持久化会话。

运行：

```bash
python3 code/main.py
```

输出显示子 Agent 上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用

- **Claude Agent SDK** 用于以 Claude 为主、想要 Claude Code harness 形态的产品。
- **Claude Managed Agents** 用于托管长期异步工作。
- **OpenAI Agents SDK**（第 16 课）用于以 OpenAI 为主的对等产品。
- **LangGraph + 自定义工具** 如果你想要图形态的状态机代替。

## 发布

`outputs/skill-claude-agent-scaffold.md` 脚手架化一个 Claude Agent SDK 应用，包含子 Agent、钩子、会话存储、MCP 服务器挂接和 W3C 追踪传播。

## 练习

1. 添加一个子 Agent 生成器，将 20 个任务分成每批 5 个并行子 Agent。测量编排器上下文大小 vs 每任务一个子 Agent。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用做速率限制（每个会话每分钟 5 次）。追踪行为。
3. 接入 `list_subkeys` 来渲染子 Agent 树。深层嵌套是什么样子？
4. 将玩具移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。何时从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent SDK | "Claude Code 的库形态" | Harness 形态：工具、MCP、钩子、子 Agent、会话存储 |
| Subagent（子 Agent） | "子 Agent" | 独立上下文、自己的预算；结果冒泡返回 |
| Session store（会话存储） | "对话数据库" | 持久化、加载、列出、删除会话轮次，级联子 Agent |
| Hook（钩子） | "生命周期回调" | 工具前/后、会话、提示提交、压缩、停止 |
| W3C trace context | "跨进程追踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长期异步工作 |
| `--session-mirror` | "对话镜像" | 将会话轮次实时写入外部文件 |
| MCP server | "工具面" | 挂接到 Agent 的外部工具/资源源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Code 的库形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)——托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——对等产品

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是 Phase 14 课程中"从 API 到产品"的桥梁课。前三节课（LangGraph 执行引擎、编排拓扑、OpenAI Agents SDK）讲的是通用多 Agent 基础设施。这节课讲的是**一个具体的产品 harness 长什么样**——Claude Code 把它内部的循环、工具、钩子、会话存储暴露出来，让外部 Agent 直接复用。文档的最大价值在于三个差异化设计：子 Agent 的上下文隔离、生命周期钩子体系、以及 `--session-mirror` 这种调试功能。适合已经理解编排和执行引擎、正在为 Claude 生态选型产品框架的架构师。

### 二、知识结构梳理

**认知基础层：** Agent 循环（第 01 课）——SDK 的 harness 就是一个包装了工具执行和 MCP 连接的 Agent 循环。技能库（第 10 课）——SDK 的"内置工具 + 自定义工具"是 Voyager 技能库概念的产品化。编排模式（第 13/28 课）——子 Agent = 编排器-工作者模式的 Claude 实现。

**Harness 架构层：** SDK 不是"Agent 框架"——是"Claude Code 的可编程副本"。五个核心组件——内置工具、子 Agent（上下文隔离+并行化）、会话存储、钩子（7 种生命周期事件）、W3C 追踪——是 Claude Code 内部架构的外部化。

**托管 vs 自托管层：** Managed Agents 是 SDK 的托管版本——用控制权（你不再管理基础设施）换取运维简化（缓存、压缩、扩展都由 Anthropic 处理）。这是"买 vs 建"在 Agent 基础设施上的体现。

### 三、核心洞察（备课时的关键理解）

1. **"SDK 是 Claude Code 的可编程副本"是理解整个产品的钥匙。** 不是 Anthropic 设计了一个 Agent 框架然后把它放进 Claude Code。是 Claude Code 先存在——它的内部架构（工具执行、MCP、钩子、会话）被证明可靠——然后被包装为 SDK。这个 SDK 不是框架设计师的作品——是产线工程师的作品。

2. **子 Agent 的上下文隔离是其最重要的工程价值。** 文档说子 Agent 有两个目的：并行化和上下文隔离。但上下文隔离是更根本的。当你让一个编排器处理一个需要 20 步推理的任务时，编排器的上下文窗口会被中间推理占满。子 Agent 把推理隔离到子窗口里——编排器只看到"最终结果是什么"，不看到"推理过程是什么"。这就是上下文预算管理在架构层面的实现。

3. **钩子体系是 SDK 区别于 OpenAl Agents SDK 和 LangGraph 的核心特征。** OpenAI SDK 有三类护栏（输入/输出/工具）。Claude SDK 有七种钩子（工具前后、会话起止、提示提交、压缩前、停止、通知）。数量差异反映了哲学差异：OpenAI 偏"安全校验"，Claude 偏"全生命周期可编程"。钩子是"你可以在这个 harness 的任何生命周期点注入自己的逻辑"——这是平台的思维。

4. **`--session-mirror` 是一个简单的 flag，解决了一个真实的调试痛点。** 当 Agent 出错时，你需要的第一个东西是什么？对话记录。如果记录只存在于 SDK 内部的序列化格式里，调试的第一步就是"写一个脚本 dump 会话"。`--session-mirror` 直接给你一个人类可读的文件，随 Agent 运行实时更新。

5. **会话存储的 `list_subkeys` 暴露了子 Agent 的嵌套结构。** 这是 SDK 对"多 Agent 系统的可观测性"的回答：不是只存主 Agent 的对话——子 Agent 也有自己的会话 key，通过 `list_subkeys` 可以重建整个子 Agent 树。

6. **W3C 追踪上下文解决了"多进程 Agent 的可观测性"问题。** Claude Code 的 CLI 是独立进程。SDK 的主进程和 CLI 子进程通过 W3C 追踪上下文头共享一条 trace——你在监控后端看到的是一个完整的调用树，不是两个孤立的 trace。

7. **Managed Agents 是"自托管 vs 托管"在 Agent 基础设施上的经典 trade-off。** 自托管 = 完全控制（你选存储、你管扩展、你设缓存策略）。托管 = 运维简化（内置缓存、压缩、扩展），但控制粒度降低。这个二分不是 Claude 独有的——OpenAI 也有一对（Agent SDK vs 托管 API），LangGraph 也有一对（自建 vs LangGraph Cloud）。

### 四、教学建议

1. **用"Client SDK → Agent SDK"的代码对比开场。** 左边 50 行：用 `anthropic` 包手写工具调用循环 + 状态管理。右边 10 行：`claude-agent-sdk` 的 `run()`。让学生看到"harness 帮你省了什么"——循环、工具执行、状态持久化、追踪。

2. **子 Agent 的上下文隔离是最值得用"上下文窗口可视化"来讲的点。** 画一个编排器窗口：插入 5 个子 Agent 的完整推理过程 = 窗口被占满。同一个任务用子 Agent：编排器窗口只收到 5 条"结果摘要"——窗口保持 20% 占用。可视化胜过所有文字。

3. **钩子体系适合用"横切关注点"的视角来讲。** 让学生列出"你想在 Agent 的哪个生命周期节点做监测"——工具调用前检查权限、会话开始时加载记忆、压缩前保存状态。然后把他们的列表映射到 SDK 的 7 种钩子。

4. **`--session-mirror` 的演示非常简单但非常有效。** 现场跑一个 Agent，开着 `--session-mirror`，同时用 `tail -f` 看镜像文件实时增长。这是开发者体验的直观冲击。

5. **和 OpenAI Agents SDK 做逐项对比。** 工具注册、转交/子 Agent、护栏/钩子、会话存储、追踪。让学生看到"同一个需求，两个框架的不同哲学"。

6. **子 Agent 过度生成是练习 1 的直接教学点。** 让学生为 20 个任务创建 20 个子 Agent，测量编排器开销。然后改成每批 5 个——看到差异。这是"架构选择有可测量的成本"的最佳证明。

7. **练习 5（Managed Agents）是理解"自托管 vs 托管"的实操入口。** 问学生："如果你的 Agent 需要每天 24 小时运行、处理 1000 个并发会话——你会自托管还是用托管？"讨论扩展到人员成本、存储运维、监控配置。

### 五、值得补充的内容

1. **子 Agent 的超时和重试策略。** 文档说子 Agent 隔离运行，但没说子 Agent 挂起/超时/失败时编排器怎么处理。需要补充错误传播和重试语义。

2. **钩子的执行顺序和异常处理。** 三个团队各注册了一个 `PreToolUse` 钩子。它们的执行顺序是什么？一个钩子抛异常是否影响其他钩子、是否影响工具执行？这是钩子系统的运行时语义。

3. **会话存储的后端选择。** SDK 支持哪些会话存储后端？SQLite 是默认吗？Postgres 用于生产吗？不同后端的 `list_sessions` 性能差异？

4. **与 LangGraph 检查点存储的哲学对比。** SDK 会话存储（按轮次）vs LangGraph 检查点（按节点）——粒度差异的架构意义。

5. **Managed Agents 的定价模型。** 自托管 SDK 的成本 = 你的基础设施 + Anthropic API 令牌。Managed Agents 的成本 = Anthropic 的托管费 + 令牌。这个经济模型决定了选哪边。

### 六、一句话总结

**Claude Agent SDK 的特殊之处不是"又一个 Agent 框架"——是它把 Claude Code 产线上已验证的内部架构（工具执行、MCP、钩子、会话）外部化为一个库。子 Agent 给上下文隔离，钩子给生命周期可编程性，`--session-mirror` 给调试透明度——这三个能力来自产线经验，不是来自框架设计。**

---

# 🎓 Agent 架构课：Claude Agent SDK——从 Claude Code 产线里长出来的 harness

同学们好。我是你们的 Agent 架构老师。

前面几节课我讲了通用框架——OpenAI Agents SDK 的五个原语、LangGraph 的状态机引擎、四种编排拓扑。今天这节课不一样。今天讲的是一个**从产线里长出来的框架**。

Claude Agent SDK 不是一个团队坐在会议室里设计的"下一代 Agent 框架"。它是 Claude Code——全球使用量最大的 AI 编码工具——把自己的内部引擎拆出来，变成你可以在自己的 Agent 里调用的库。

这节课的关键不是"怎么用"——是"为什么它长这样"。因为它的每一个设计决策，都来自一个已经跑了数亿次的真实 Agent。

## 先看没有 harness 长什么样

假设你要写一个 Claude 的 Agent。你用 Anthropic 的 Client SDK：

```python
import anthropic

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "找一下 test_user.py 里的 bug"}]
tools = [...]  # 你定义的工具

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=messages,
        tools=tools,
    )
    if response.stop_reason == "end_turn":
        break
    # 你手写工具调用逻辑
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block)
            messages.append(assistant_message)
            messages.append(tool_result_message)

# 50 行之后：你现在有了一个一次性的 Agent 循环。
# 状态管理？自己写。
# 工具执行？自己写。
# 会话持久化？自己写。
# 追踪？自己写。
# 子 Agent？自己写。
```

这就是"从零开始"的 Agent。150 行之后你能跑起来。但你会发现你在重复造 Claude Code 已经做过的东西。

Claude Agent SDK 做的事就是：**"这 150 行我给你了。这是 Claude Code 内部用的那个循环。拿去。"**

## 子 Agent：上下文隔离是杀手功能

SDK 的子 Agent 有两个用途。第二个比第一个重要十倍。

**第一个：并行化。** "为这 20 个模块的每一个找到对应的测试文件。" 你生成 20 个子 Agent，每个处理一个模块。20 个独立任务，20 个独立上下文窗口，20 个并行结果。

**第二个：上下文隔离。** 这是子 Agent 的真正价值。

假设你的编排器在处理一个复杂任务——"重构用户认证系统"。这个任务需要 15 步推理。如果编排器自己做全部 15 步——它的上下文窗口会被中间推理占满。到了第 10 步，它已经开始忘记第 1 步的上下文。

子 Agent 的解决方案：**不要把所有推理放在一个窗口里。** 把"重构密码哈希模块"作为一个子任务，生成一个子 Agent——子 Agent 在自己的上下文窗口里做 5 步推理，最终返回一句话："密码哈希已从 SHA-256 迁移到 Argon2id，所有测试通过。"

编排器的上下文窗口里只多了这一句结论——不是 5 步推理过程的全部中间状态。

这就是上下文预算管理在架构层面的实现。不是压缩策略、不是摘要——是**隔离子任务的推理到一个独立的窗口，只返回结果。**

Python SDK 还给了你两个调试工具：`list_subagents()` 列出所有活跃的子 Agent，`get_subagent_messages()` 读取特定子 Agent 的完整推理记录。子 Agent 出错了？你不是猜——你直接读它的完整对话。

## 会话存储：永久的 Agent 记忆

SDK 的会话存储接口很简洁：

- `append(session_id, message)`——存一轮对话。
- `load(session_id)`——恢复整个对话历史。
- `list_sessions()`——你有多少个活跃会话？
- `delete(session_id)`——清理，**级联删除子 Agent 会话。**
- `list_subkeys(session_id)`——这个会话下有哪些子 Agent？

注意 `delete` 的级联语义。LangGraph 的检查点存储也做级联删除——但这在这两个框架中是显式设计决策。删一个主 Agent 会话，所有子 Agent 的会话也一起清掉——不是"忘掉子 Agent"的 bug，是"级联清理"的设计。

再注意 `list_subkeys`。这是你的子 Agent 树的可观测性入口。主 Agent 生成 3 个子 Agent → 子 Agent 1 又生成 2 个孙子 Agent → `list_subkeys` 把这个树结构暴露给你。不用猜"到底有几个子 Agent 在跑"——你有一条 API 告诉你。

**`--session-mirror`：** 这是 Claude 产线经验的一个直接体现。当 Agent 出问题时，工程师的第一步永远是"把对话记录给我看看"。`--session-mirror` 把一个人类可读的对话记录实时写入外部文件——你不需要写 dump 脚本，不需要序列化/反序列化，就是一个文件。你用 `tail -f` 跟着看，Agent 每说一句话它就写一行。

这个功能不是框架设计师想出来的——是产线工程师要求的。

## 钩子：全生命周期可编程

OpenAI Agents SDK 有三类"护栏"——输入、输出、工具。它们是安全校验。

Claude Agent SDK 有七种"钩子"——从会话启动到 Agent 退出。它们是**全生命周期的可编程接口。**

- `SessionStart`——会话开始时，加载长期记忆、初始化连接。
- `UserPromptSubmit`——用户消息进了，但模型还没看到。你可以做输入预处理。
- `PreToolUse`——工具即将被调用。这是你的权限校验点。
- `PostToolUse`——工具调用完成。这是你的审计点。
- `PreCompact`——上下文窗口快满了，压缩要开始了。这是你的状态保存点。
- `SessionEnd`——会话结束。清理资源。
- `Stop`——Agent 退出。最后的安全检查或资源释放。
- `Notification`——侧信道事件。Agent 内部有什么不紧急但值得关注的事发生了。

七个钩子不是七个独立功能——是 Agent 生命周期的七个介入点。每个点你都可以注入自己的代码。这不是"安全校验"——这是"我把 Agent 运行时的每一环都开放给你"。

但钩子多了会膨胀。每个团队注册几个钩子——一年后你的 Agent 启动时间里 40% 花在执行钩子上。文档的建议：每季度审查一次钩子注册表。不用的删。

## W3C 追踪上下文：多进程的可观测性

Claude Code 的 CLI 是一个独立的子进程。Agent SDK 在主进程中运行你的逻辑，但工具执行、MCP 调用可能通过 CLI 子进程。

你的监控后端怎么看到一条完整的调用链？

答案：W3C 追踪上下文。主进程的 OTel span 通过 HTTP 头传播到 CLI 子进程。子进程里产生的 span 标记为同一个 trace 的子 span。你在 Jaeger 或 Grafana 里看到的是一条完整的 trace，从"用户发出请求"到"CLI 执行 `grep`"——不是两个孤立的片段。

这是 OpenTelemetry 的标准机制，但 Claude SDK 的特别之处是默认就配好了。你不用写传播逻辑。

## Claude Managed Agents：什么时候"自己搭"不如"用现成的"

SDK 是自托管的。你跑在自己的机器上，你管理存储、扩展、缓存。

Managed Agents 是托管的。你调 Anthropic 的 API，Anthropic 帮你管：内置提示缓存（重复的上下文自动缓存）、内置压缩（窗口快满时自动压缩）、长期异步执行（Agent 可能跑几分钟到几小时）。

这个选择不是技术决策——是运维决策。如果你的团队有运维能力管 Agent 基础设施，自托管 SDK 给你完全控制。如果你的团队想专注在业务逻辑上——Managed Agents 替你管基础设施。

## 反模式

### 反模式一：子 Agent 过度生成

"我有 100 个小任务 → 生成 100 个子 Agent。" 每个子 Agent 有自己的初始化开销、上下文窗口分配、进程/线程创建。100 个子 Agent 的创建和切换开销可能比处理任务本身还大。

解决方案：**批量处理。** 把 100 个任务分成 10 组，每组 10 个——生成 10 个子 Agent，每个处理 10 个小任务。

### 反模式二：钩子膨胀

工程团队一："我们加个 `PreToolUse` 钩子做日志。" 工程团队二："我们也加个 `PreToolUse` 做权限检查。" 工程团队三："我们也加个做计费。"

一年后：`PreToolUse` 队列里有 9 个钩子。每个工具调用的开销增长了 50ms。你甚至不知道是哪个钩子慢。

解决方案：**钩子注册表审查。** 每季度跑一次 `list_hooks()`，问每个钩子"你还在用吗？"

### 反模式三：会话无限增长

Agent 每天产生新会话——会话存储无限增长。6 个月运行下来，会话表里有 5 万条记录，每条记录里有几百轮对话。`list_sessions()` 开始超时。

解决方案：**会话过期策略。** 不是"以后再说"——是上线第一天就配。7 天不活跃 → 归档。30 天不活跃 → 删除。

## 什么时候用 Claude Agent SDK

- **用 Claude Agent SDK：** 你用 Claude 作为主要模型，你想复用 Claude Code 已验证的 harness 形态（工具、MCP、钩子、会话），你需要子 Agent 做上下文隔离。
- **用 OpenAI Agents SDK：** 你用 GPT 作为主要模型，你需要五个原语（Agent/Handoff/Guardrail/Session/Tracing）。
- **用 LangGraph：** 你需要精确的步级状态管理和检查点恢复。
- **用 Claude Managed Agents：** 你不想管基础设施，长期异步 Agent 工作。

## 结课清单

如果你在用 Claude Agent SDK：

1. Client SDK = 原始 API，你自己写循环。Agent SDK = Claude Code 的循环，拿走直接用。
2. 子 Agent 的主要价值不是并行——是上下文隔离。隔离推理，只传结果。
3. `list_subagents()` + `get_subagent_messages()` 是你的子 Agent 调试工具。
4. `delete(session_id)` 级联删除——不是 bug，是设计。
5. `list_subkeys(session_id)` 给你子 Agent 树的可观测性。
6. `--session-mirror` 是你的对话记录实时快照——产线工程师的最爱。
7. 钩子每季度审查一次——膨胀是沉默的启动时间杀手。
8. 会话过期策略上线第一天配好——不是"以后再说"。

**最后一句话：**

**Claude Agent SDK 的独特价值不是"又一个 Agent 框架"——是它来自一个已经跑了数亿次对话的生产系统。子 Agent 的上下文隔离、七种生命周期钩子、`--session-mirror` 的调试透明度——这些不是框架设计师的创意，是产线工程师在真实故障中学到的教训，固化为 API。**

