# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 Agent 遥测的标准 schema。Span 名称、属性和内容捕获规则在供应商之间收敛，使得 Agent 追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中意味着同一件事。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 13（LangGraph）、Phase 14 · 24（可观测性平台）
**用时：** 约 60 分钟

## 学习目标

- 说出 GenAI span 类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT vs INTERNAL span 以及各自适用的场景。
- 列出顶层 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获契约：选择加入、`OTEL_SEMCONV_STABILITY_OPT_IN`、推荐的外部引用方式。

## 问题

每个供应商都发明自己的 span 名称。运维团队最终要为每个框架构建独立的仪表盘。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统的统一标准来修复这一点。

## 概念

### Span 类别

1. **Model / client span。** 覆盖原始 LLM 调用。由供应商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent span。** `create_agent`（当 Agent 被构造时）和 `invoke_agent`（当它运行时）。
3. **Tool span。** 每次工具调用一个；通过父子关系连接到 agent span。

### Agent span 命名

- Span 名称：如果有名称则为 `invoke_agent {gen_ai.agent.name}`；回退到 `invoke_agent`。
- Span 种类：
 - **CLIENT**——用于远程 Agent 服务（OpenAI Assistants API、Bedrock Agents）。
 - **INTERNAL**——用于进程内 Agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name`——`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`——模型 ID。
- `gen_ai.response.model`——解析后的模型（可能因路由与 request 不同）。
- `gen_ai.agent.name`——Agent 标识符。
- `gen_ai.operation.name`——`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`——用于 RAG：查询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则：instrumentation 默认**不应**捕获输入/输出。捕获是通过以下方式选择加入的：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非散文）。这是第 27 课的内容中毒防御接入可观测性。

### 稳定性

截至 2026 年 3 月，大多数约定是实验性的。通过以下方式选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM Observability schema 中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式在何处会出错

- **在 span 中捕获完整提示。** PII、密钥、客户数据出现在运维可读的追踪中。存到外部。
- **无 `gen_ai.provider.name`。** 多供应商仪表盘在缺少归因时无法工作。
- **Span 无父链接。** 孤立的工具 span。始终传播上下文。
- **未设置稳定性选择加入。** 你的属性可能在后端升级时被重命名。

## 构建

`code/main.py` 实现一个匹配 GenAI 约定的基于标准库的 span 发射器：

- `Span`，带 GenAI 属性 schema。
- `Tracer`，带 `start_span`、嵌套上下文。
- 一个脚本化 Agent 运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每工具 span、LLM 调用的 `chat` span。
- 一个内容捕获模式，将提示存储在外部并在 span 上记录 ID。

运行：

```bash
python3 code/main.py
```

输出：一个带全部必需 GenAI 属性的 span 树，以及一个展示选择加入内容引用的"外部存储"。

## 使用

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）——生态系统的自动仪表化。
- **Jaeger / Honeycomb / Grafana Tempo**——原始 OTel 追踪；从 GenAI 属性构建仪表盘。
- **自托管**——运行带 GenAI 处理器的 OTel Collector。

## 发布

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有 Agent，带内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每工具 span 对你的第 01 课 ReAct 循环做仪表化。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示存到 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性未被收集器重命名。
5. 构建一个仪表盘："哪些工具错误与哪些模型相关"——仅从 GenAI 属性构建。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "Agent span" | 表示 Agent 运行的 span 名称 |
| CLIENT span | "远程调用" | 调用远程 Agent 服务的 span |
| INTERNAL span | "进程内" | 进程内 Agent 运行的 span |
| gen_ai.provider.name | "供应商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中了哪个语料库/存储 |
| 内容捕获（Content capture） | "提示日志" | 选择加入的消息捕获；生产环境中存到外部 |
| 稳定性选择加入（Stability opt-in） | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——默认 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)——W3C 追踪上下文传播

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是 Phase 15 可靠性工程模块的底层标准课——不是"用哪个平台"，是"所有平台共同遵循的 schema 是什么"。文档将 OTel GenAI SIG 的输出（span 类别、命名规范、关键属性、内容捕获规则）压缩为一课，核心信息是：**Agent 遥测的标准化让跨平台可移植性成为可能。** 适合已理解可观测性平台（第 13 课）、正在做仪表化的架构师。

### 二、知识结构梳理

**Schema 层：** 三类 span（model/client、agent、tool）定义了 Agent 遥测的命名空间。属性约定覆盖了供应商识别、模型版本、操作类型、数据来源——这些是跨平台仪表盘的共同语言。

**工程实践层：** 内容捕获的"默认不捕获"规则 + 外部存储推荐是生产 Agent 的数据安全底线。稳定性选择加入（`gen_ai_latest_experimental`）是实验性规范在生产中的可靠性保障。

**生态层：** Datadog v1.37+ 原生映射、Langfuse/Phoenix/Opik 自动仪表化——GenAI 约定已成为三个可观测性平台（第 13 课）的底层标准。

### 三、核心洞察（备课时的关键理解）

1. **GenAI SIG 解决的不是技术问题——是互操作性问题。** 每个供应商都有自己的 span 格式。运维团队不想为 LangChain 建一套仪表盘、为 CrewAI 再建一套。GenAI 约定让"Agent 调用"在三平台上意味着同一件事。

2. **CLIENT vs INTERNAL span 的区分是架构信息的嵌入。** 远程调用（OpenAI Assistants API）和本地调用（LangGraph 进程内）在 span 级别被区分——不是日志中的一行文字，是 span kind 的二进制标记。这对成本归因（哪些是外部 API 调用？）和延迟分析（瓶颈在本地还是远程？）至关重要。

3. **"默认不捕获内容"是数据安全的底线设计。** PII、密钥、客户对话——如果默认写入 span，运维人员可读、SOC 审计可见、数据泄露面扩大。外部存储 + 引用 ID 的模式将可观测性和数据安全解耦。

4. **`gen_ai.data_source.id` 是 RAG 可观测性的关键属性。** 不是"Agent 检索了文档"——是"Agent 检索了哪个语料库"。当 RAG 质量下降时，这个属性让你区分"是检索算法的问题"还是"语料库本身被污染了"。

5. **稳定性选择加入是 GenAI 规范在生产中可用的前提。** 实验性规范的属性名可能在后端升级时改变——你的仪表盘突然断掉。`OTEL_SEMCONV_STABILITY_OPT_IN` 固定属性名称——这是生产依赖实验性规范的标准做法。

6. **"Span 无父链接"是第 15 课（失败模式）中级联错误在可观测性层的表现。** 工具 span 成了孤儿——你不知道是哪个 Agent 的哪次运行调用了它。上下文传播（trace context propagation）不是可选的——是可追溯性的基础。

7. **各供应商技术特定约定的存在说明标准化仍在进行中。** Anthropic、OpenAI、Bedrock 各有自己的扩展属性——通用的 GenAI 约定是共同子集，专有属性是差异化。这既是标准化的进步，也是"完全统一"尚未实现的标志。

### 四、教学建议

1. **用"三个平台看同一条 trace"的对比开场。** 展示一条 Agent 运行产生的 span 树——在 Datadog 里叫什么、在 Grafana 里叫什么、在 Jaeger 里叫什么。如果有 GenAI 约定——三个平台上名字一样。如果没有——三个名字。差异化 = 成本。

2. **CLIENT vs INTERNAL 用"本地函数调用 vs 远程 API 调用"的类比来讲。** 你调用本地 `def foo()` = INTERNAL。你调用 OpenAI 的 API = CLIENT。这个区分在 span 级别存在，因为成本归因和延迟分析需要它。

3. **内容捕获的"默认不捕获"是一个安全决策，值得单讲。** 让学生列举"Agent 的对话里有什么不应该出现在运维仪表盘上的东西"——密钥、用户 PII、内部系统名称、未发布的代码。然后解释"这就是为什么默认不捕获——选择加入是故意的安全决策。"

4. **`gen_ai.data_source.id` 用 RAG 故障案例来教。** Agent 开始回答错误退款流程。追踪显示 `data_source.id = "retail_db_v3"`——但 v3 是三个月前被替换的旧版本。没有这个属性，你花 30 分钟查"为什么会错"；有了它，30 秒。

5. **稳定性选择加入应该是一个实操步骤。** 让学生设置环境变量、跑 Agent、导出 trace、升级 OTel Collector 版本、再导出。对比两次的属性名——如果有变化，说明你之前没有设置稳定性选择加入。

6. **Span 孤儿问题用级联错误的案例串联。** Agent 在第三步工具调用时崩溃了——span 成了孤儿。但你不知道是哪个 Agent 的哪次运行的第三步。上下文传播和级联错误检测是同一个问题的两个面。

7. **练习 5（从 GenAI 属性构建仪表盘）是最综合的。** "哪些工具错误与哪些模型相关"——这个仪表盘需要 provider.name、operation.name、error 属性、model 属性。学生必须理解每个属性的含义才能组合查询。

### 五、值得补充的内容

1. **GenAI 属性的查询模式。** 在实践中，运维团队用这些属性构建什么查询？"按模型分组的工具错误率""按数据源分的 RAG 相关性""按供应商分的延迟分布"——这些是标准仪表盘模板。

2. **多 Agent 追踪的嵌套 span 结构。** 监管者 Agent 的 span → 子 Agent A 的 span → 子 Agent A 的工具 span。这个嵌套树在 UI 中如何呈现？三平台的可视化差异。

3. **与 W3C Trace Context 的关系。** Claude Agent SDK 用 W3C 追踪上下文传播跨进程 span。GenAI 约定定义了 span 内容——W3C 定义了 span 怎么连接。两者是互补的。

4. **采样策略。** 生产系统不能 100% 追踪——太贵。OTel 的采样策略如何与 GenAI span 交互？低采样率下丢失哪些诊断信息？

5. **从实验性到稳定的迁移路径。** GenAI 规范何时从实验性毕业？迁移时你的仪表盘需要什么修改？这个时间线是企业采用决策的关键输入。

### 六、一句话总结

**OpenTelemetry GenAI 约定不是你 Agent 的一个特性——是让你的 Agent 在 Datadog、Grafana、Jaeger、Honeycomb 上说同一种语言的基础设施。Span 类别定义了什么、属性定义了谁和怎么、内容捕获定义了安全边界——这三层让 Agent 遥测从"每个平台各说各话"变成"一条 trace 到处可读"。**

---

# 🎓 Agent 架构课：OTel GenAI——让你的 Agent 在四个平台上说同一种语言

同学们好。我是你们的 Agent 架构老师。

上节课我讲了三个可观测性平台——Langfuse、Phoenix、Opik。但我留了一个问题没答：**这三个平台怎么知道一条 span 是什么意思？**

你发出了一条 span，名字叫 `agent_call`。Langfuse 看到 `agent_call`，Phoenix 看到 `agent_call`——它们各自猜这是什么。Langfuse 把它归到"未知"类别，Phoenix 把它和所有其他 `agent_call` 聚在一起——但你还有一个 span 叫 `agent_invoke`，那是另一个 Agent 发出的——两个平台都不知道它们其实是同一件事。

这就是 OpenTelemetry GenAI 约定要解决的问题。不是"怎么发 span"——是**"发的 span 叫什么名字、带什么属性、怎么分类，才能在四个不同的平台上被理解为同一件事。"**

## 三个类别，三种语义

OTel GenAI 规范把 Agent 遥测拆成三类 span。每类有不同的语义。

**第一类：Model / client span。** 这就是最底层的 LLM 调用。你的 Agent 调了一次 Claude、一次 GPT——每次调用一条 span。谁发出的？供应商 SDK（Anthropic、OpenAI 的官方包）或者框架的模型适配器。

**第二类：Agent span。** 这是"一个 Agent 在运行"。有两个子类：`create_agent`（初始化时）和 `invoke_agent`（实际运行时）。关键区分——`invoke_agent` 可以是 CLIENT 或 INTERNAL。

**第三类：Tool span。** 每次工具调用一条 span。连接到父 Agent span 之下。

这个三层结构就是 Agent 遥测的全景：底层是模型调用，中层是 Agent 运行，底层是工具执行。一条用户请求进来 → 触发一个 `invoke_agent` span → 里面嵌套了 3 次 `chat` span（LLM 调用）和 5 条 `tool_call` span。

## CLIENT vs INTERNAL：你不只是一个标记

`invoke_agent` 有两种：CLIENT 和 INTERNAL。这不是无聊的分类——**它直接关系到成本归因和延迟分析。**

**INTERNAL：** Agent 在你的进程里跑。LangGraph 的图、CrewAI 的 Crew、你自己的 ReAct 循环。延迟 = 你的代码执行时间。成本 = 你的服务器算力。

**CLIENT：** Agent 在别人那里跑。OpenAI Assistants API、AWS Bedrock Agents。延迟 = 网络往返 + 远程服务处理时间。成本 = API 调用费用。

如果你的仪表盘不区分这两种 span——你看延迟分布图，不知道为什么有些 Agent 调用 50ms 就回来了，有些要 500ms。因为你把本地函数调用和远程 API 调用混在一起了。

## 六个属性，六个问题

GenAI 约定定义的关键属性回答了六个问题：

- **`gen_ai.provider.name`** → "用的是谁的模型？" —— anthropic、openai、aws.bedrock、google.vertex。
- **`gen_ai.request.model`** → "请求的是什么模型？" —— `claude-sonnet-4-20250514`。
- **`gen_ai.response.model`** → "实际用了什么模型？" —— 可能因为路由和请求不同。
- **`gen_ai.agent.name`** → "哪个 Agent？" —— `triage-agent`、`billing-agent`。
- **`gen_ai.operation.name`** → "在做什么操作？" —— `chat`、`completion`、`invoke_agent`、`tool_call`。
- **`gen_ai.data_source.id`** → "查了什么数据？" —— RAG 检索的词料库 ID。

这六个属性的组合就是你查询分析的基本语法：

"过去一小时内，哪些工具在 GPT-5 上出错的次数比在 Claude Sonnet 上多？"
→ 查 `operation.name = tool_call` + 按 `provider.name` 分组 + 统计 `error` 计数。

"哪些 RAG 数据源的检索结果与最终答案不一致？"
→ 查 `data_source.id` + 交叉 `output.messages` 的评估分数。

## 内容捕获：你不该把一切都写进 span

这是这节规范里最重要的安全条款：**默认不捕获 Agent 的输入和输出。**

为什么？因为你的 span 会被运维团队看到、被 SOC 审计读到、被导出到第三方平台存储。你的 Agent 的对话里有什么？用户的 PII、API key、内部系统名称、未发布的代码、客户的商业机密。

默认不捕获 = **安全底线。** 选择加入 = **你有意识地把某些内容纳入可观测性——并且你知道风险。**

生产中的推荐做法是：把内容存在外部（S3、你的日志存储），在 span 属性里只放一个引用 ID。Datadog 查过这条 span 需要看内容 → 去外部存储用 ID 取。Span 本身没有任何敏感内容。

## 稳定性：你的仪表盘可能在某次升级后全断

GenAI 约定截至 2026 年大部分是实验性的。实验性意味着——属性名可能在 OTel 规范的下一个版本中改变。`gen_ai.agent.name` 变成了 `gen_ai.agent.id`——你的所有仪表盘查询都断了。

解决方案：`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`。这个环境变量告诉 OTel 用你写代码时的那套属性名——即使规范更新了，你的属性名不变。

Datadog v1.37+ 已经原生映射 GenAI 属性——意味着如果你用 Datadog，稳定性风险部分被 Datadog 吸收了（他们的映射层处理变更）。

## 结课清单

用 OTel GenAI 对你的 Agent 做仪表化：

1. 三类 span：model/client（LLM 调用）、agent（invoke_agent）、tool（工具调用）。
2. `invoke_agent` 区分 CLIENT（远程）和 INTERNAL（本地）——这对成本和延迟分析至关重要。
3. 六个关键属性填全——provider、model、agent name、operation、data source。
4. 内容捕获默认关——在 span 中只放引用 ID，内容存外部。
5. 设置稳定性选择加入——保护仪表盘不被规范升级破坏。
6. 上下文传播——每个 tool span 都有 parent agent span，不做孤儿。

**最后一句话：**

**OpenTelemetry GenAI 约定的价值不是"定义了 span 的名字"——是"定义了 Agent 遥测的共同语言"。当 LangGraph、CrewAI、OpenAI SDK、Claude SDK 发出的 span 都用同一套名字和属性时，你的仪表盘不再为每个框架重写——它读一次规则，到处适用。标准化不是锦上添花——是可观测性在 2026 年的入场券。**

