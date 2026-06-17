# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条追踪覆盖全部。OpenTelemetry GenAI 语义约定（v1.37 及以上版本的稳定属性）是 2026 年的标准，被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课命名必需的属性，走读跨度层级（智能体 → LLM → 工具），并交付一个可接入任何 OTel 导出器的标准库跨度发射器。

**类型：** 构建
**语言：** Python（标准库、OTel 跨度发射器）
**前置课程：** Phase 13 · 07（MCP 服务器）、Phase 13 · 08（MCP 客户端）
**用时：** 约 75 分钟

## 学习目标

- 命名 LLM 跨度和工具执行跨度的必需 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端调度的追踪层级。
- 决定捕获什么内容（按需加入）vs 屏蔽什么内容（默认）。
- 将跨度发送到本地收集器（Jaeger、Langfuse），无需修改工具代码。

## 问题

2026 年 2 月的一次调试：用户报告"我的智能体有时 30 秒才响应；有时只要 3 秒。"没有追踪。日志显示了 LLM 调用，但不显示工具调度、不显示 MCP 服务器往返、不显示子智能体。你只能猜测。最终你发现：某个 MCP 服务器偶尔在冷启动时卡住。

没有端到端追踪，你找不到这个问题。OTel GenAI 解决了它。

相关约定在 2025-2026 年间由 OpenTelemetry 语义约定组确定。它们定义了稳定的属性名称，使得 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的跨度。一次插桩；发送到任何后端。

## 概念

### 跨度层级

```
agent.invoke_agent  （顶层，INTERNAL 跨度）
 ├── llm.chat       （CLIENT 跨度）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT 跨度）
 ├── llm.chat       （CLIENT 跨度）
 └── subagent.invoke （INTERNAL）
```

整个结构嵌套在一个追踪 ID 下。跨度 ID 链接父子关系。

### 必需属性

根据 2025-2026 语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供方响应 ID，用于关联。

工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

智能体跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供方、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 按需内容捕获

默认情况下，跨度携带指标和计时——不包含提示词或补全。大型负载和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量来包含内容。在生产环境启用前务必仔细审查。

### 跨度上的事件

令牌级事件可以作为跨度事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在跨度内按时间排序，用于详细回放。

### 导出器

OTel 跨度可以导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** LLM 可观测性专用；可视化 Token 使用。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 面向列；查询友好。

全部使用 OTLP（传输线格式）通信。你的代码不需要关心。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入到请求中。Streamable HTTP 支持标准头。Stdio 原生不携带 HTTP 头；规范的 2026 路线图讨论了在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在此之前：手动在每次请求的 `_meta` 中包含 traceparent。服务器记录追踪 ID。

### 指标

除跨度外，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用于不需要每次调用详情的仪表盘。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它包装主流框架（LangGraph、Pydantic AI、CrewAI）以自动发送 OTel 跨度。如果你的技术栈使用受支持的框架，这会很有用；否则使用手动插桩。

## 实战

`code/main.py` 为一个调用 LLM、调度两个工具并进行一次 MCP 往返的智能体，将 OTel 形态的跨度发送到 stdout（以类似 OTLP-JSON 的格式）。没有真实的导出器——本课专注于跨度形态和属性集。将输出粘贴到 OTLP 兼容的查看器中或直接阅读。

需要关注的内容：

- 追踪 ID 在所有跨度间共享。
- 父子关系通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量打开它。

## 交付

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能产出一份插桩计划：在哪里添加跨度、填充哪些属性、以及目标导出器。

## 练习

1. 运行 `code/main.py`。统计跨度数量，识别哪些是 CLIENT、哪些是 INTERNAL。

2. 打开内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并在每次调用时将其作为直方图样本发送。

4. 将 traceparent 从父智能体跨度传播到 MCP 请求的 `_meta.traceparent` 字段中。验证 MCP 服务器能看到相同的追踪 ID。

5. 阅读 OTel GenAI 语义约定规范。找出规范中列出的一个本课代码未发送的属性。添加它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OTel | "OpenTelemetry" | 用于追踪、指标、日志的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / 工具 / 智能体跨度的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 具有开始、结束和属性的工作单元 |
| Trace | "跨跨度血缘" | 共享一个追踪 ID 的跨度树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于跨度方向的提示 |
| OTLP | "OpenTelemetry 传输线协议" | 导出器使用的传输格式 |
| 按需内容 | "提示词 / 补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| Exporter | "后端特定的发送器" | 将跨度发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的正式约定
- [OpenTelemetry — GenAI 跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI 智能体跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI 跨度](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的事实来源
- [Datadog — LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成演练

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一篇实操导向的可观测性课程，用一个生动的调试故事（"30 秒 vs 3 秒"）切入，将 OTel GenAI 的跨度层级、必需属性、内容捕获策略和导出器生态串联成完整的技术叙事。文档结构严谨——从问题（无追踪则无法定位）、到概念（跨度层级与属性）、到实战（stdout 发射器）、到交付物（插桩计划）——每一步都有明确的教学目的。对"默认不捕获内容"这一安全设计的强调体现了生产级思维。

### 二、知识结构梳理

- **可观测性基础层：** Trace/Span/SpanKind 的核心概念，以及 agent → LLM → tool → MCP 的多层嵌套关系。这是理解 GenAI 系统可观测性的先决知识。
- **GenAI 语义约定层：** `gen_ai.operation.name`、`gen_ai.request.model`、`gen_ai.usage.*` 等稳定属性，以及 tool/agent 子命名空间。这些属性是跨后端互操作性的关键——一次插桩，Datadog/Langfuse/Phoenix 都可解析。
- **生产集成层：** 导出器选择（Jaeger/Langfuse/Datadog）、跨 MCP 的 traceparent 传播、内容捕获的 PII 权衡、AgentOps 的自动插桩。这些将理论跨度转化为实际运维能力。

### 三、核心洞察

1. **"无追踪则无法定位"是真实的工程痛点：** 开场调试故事精准地说明了为什么端到端追踪不是奢侈品而是必需品——在多层 LLM/工具/MCP 调用链中，仅靠日志无法关联。
2. **span hierarchy 的设计是深思熟虑的：** agent.invoke_agent → llm.chat → tool.execute → mcp.call 的层级不是随意堆叠，而是精确映射了智能体系统的实际执行拓扑。
3. **SpanKind 的 CLIENT vs INTERNAL 区分很重要：** CLIENT 标注跨进程边界（LLM API、MCP 服务器），INTERNAL 标注进程内执行。这个区分在故障定位时能快速判断问题在网络侧还是逻辑侧。
4. **默认不捕获内容是生产安全的基石：** 不是不能捕获提示词和补全，而是必须有意识地打开。这防止了无意识的 PII 泄露和存储成本爆炸——一个经常被忽视但至关重要的设计。
5. **traceparent 传播到 MCP 的挑战暴露了 Stdio 传输的局限：** MCP 的 Stdio 传输不原生支持 HTTP 头，需要在 `_meta` 中手动携带追踪上下文。这是一个值得在架构选型时考虑的权衡。
6. **OTel 的"一次插桩，任意后端"价值：** gen_ai.* 属性的标准化使得团队可以在 Jaeger（开发调试）、Langfuse（LLM 专业分析）、Datadog（生产监控）之间自由切换，不被特定厂商锁定。
7. **AgentOps 的自动化插桩是降低门槛的关键：** 对于使用 LangGraph/CrewAI 等框架的团队，AgentOps 的自动包装可以将插桩从"需要做的工作"变成"默认就有的能力"。

### 四、教学建议

1. **调试故事开场：** 用"30 秒 vs 3 秒"的真实案例作为 Hook，让学生先尝试在没有追踪的情况下"猜测"问题所在，然后再引入 OTel 的跨度层级——对比体验能加深理解。
2. **跨度层级动手绘制：** 让学生拿出自己用过的一个智能体架构，手绘其跨度层级图（agent → LLM → tool → MCP），然后与 OTel GenAI 的标准层级对比，找出差异和遗漏。
3. **Jaeger 实战：** 不建议只讲 stdout 输出。用 Docker 启动一个本地 Jaeger 实例（一条命令），让学生看到真实的火焰图和甘特图。可视化是理解追踪的最佳方式。
4. **PII 讨论环节：** 让学生列出"如果启用内容捕获，可能泄露什么"——API 密钥、用户密码、内部 IP 地址——然后讨论哪些场景值得冒这个风险（如调试生成质量），哪些不值得（如生产环境）。
5. **traceparent 传播实验：** 让学生实现从 agent 到 MCP server 的手动 traceparent 传播，然后验证 Jaeger 中能看到完整的跨服务追踪链。成功连接的那一刻很有成就感。
6. **指标 vs 跨度的选择练习：** 给出几个运维场景（"今天的 Token 消耗正常吗？""为什么这次调用特别慢？"），让学生判断该看指标还是看跨度——强化两者互补的心智模型。
7. **导出器对比矩阵：** 让学生调研 Jaeger/Langfuse/Phoenix/Datadog/Honeycomb 的功能差异和定价，制作一个选型对比表。这培养工程决策能力。

### 五、值得补充的内容

1. **OTel Collector 的部署架构：** 文档聚焦在跨度形态和属性，可以补充一个典型的 Collector 部署图（Agent → Collector → Exporter → Backend），帮助学生理解数据流。
2. **采样策略：** 在生产环境中，100% 追踪可能成本过高。可以补充 head-based sampling、tail-based sampling 和基于错误的采样策略简介。
3. **Langfuse 的评分和评估集成：** Langfuse 不仅可视化 Token 使用，还支持人工评分和自动评估——可以补充一个使用 Langfuse 评估智能体质量的简要示例。
4. **跨 span 的 bagage 传播：** 除了 traceparent，OTel 还支持 baggage（键值对跨服务传播），可以补充在智能体场景中用 baggage 携带租户 ID 或实验标识的最佳实践。
5. **MCP Streamable HTTP 的原生追踪支持现状：** 可以补充截至 2026 年 6 月 Streamable HTTP 传输对 W3C traceparent 的原生支持程度，帮助学生在选型时做出知情决策。

### 六、一句话总结

OTel GenAI 用标准化的跨度和属性名称为"智能体 → LLM → 工具 → MCP"的多层调用链提供了统一的追踪语言——一次插桩、任意后端、默认安全——它不是在问"要不要追踪"，而是在说"从现在开始，你的智能体系统已经有了一个通用的可观测性骨架"。


---

# 🎓 Agent 架构课：OTel GenAI——给你的智能体装上行车记录仪

同学们好。我是你们的FDE工程老师，今天讲的是 Agent 可观测性。

**你的智能体调了五个工具、三个 MCP 服务器、两个子智能体——返回了错误结果。你怎么排查？** LLM 日志只告诉你"第 3 步调了 search"，不告诉你工具内部调了什么、MCP 往返耗时多少、子智能体推理轨迹。

OTel GenAI 不是你日志上的补丁——是整条调用链追踪：

```
agent.invoke_agent  (顶层)
 ├── llm.chat        (第 1 次 LLM)
 ├── tool.execute    (search)
 │    └── mcp.call   (MCP 往返)
 ├── llm.chat        (第 2 次 LLM)
 └── subagent.invoke (子智能体)
      ├── llm.chat
      └── tool.execute
```

一条 Trace ID 贯穿。Span ID 标记节点。父子关系标记层级。日志是扁平的——追踪是树状的。

OTel GenAI 语义约定（v1.37+）定义了标准属性——`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`。一套插桩，Jaeger/Langfuse/Datadog/Phoenix 都认。

三个必须：① 同一个 Trace ID 贯穿整个 Agent 循环。② 工具执行跨度下必须嵌套 MCP 客户端跨度——区分"工具慢"和"MCP 往返慢"。③ 必须记录 token 数——成本归因的基础。

## 结课清单

1. **追踪 ≠ 日志。** 追踪告诉你"谁触发了谁"和"哪段最慢"。
2. **一条 Trace 一棵树。** agent → llm → tool → mcp → subagent。
3. **OTel GenAI 是标准。** 不用自己发明属性名。
4. **token 数 = 成本归因基础。** 没有这个算不出哪段在烧钱。

最后一句话：**没有端到端追踪的 Agent 系统，出一个 bug 查三天。给你的智能体装一个行车记录仪。**

---

# 💼 从业者故事：OTel GenAI——给你的智能体装上行车记录仪

2026 年 2 月那件事我到现在还记着。用户报告说"智能体有时候 30 秒才回，有时候 3 秒"。我们查了三天。LLM 日志显示每次调用都正常——200 OK，延迟也正常。工具调用日志看起来也没问题。最后是一个前端工程师无意中发现：浏览器 Network 面板里，某个请求的 timing 长到离谱。顺着查下去，发现是一个 MCP 服务器在冷启动时卡了 25 秒。这个服务器不是每次请求都重启，只有超过 5 分钟没请求时，V8 引擎的 JIT 缓存被回收了，下次启动重新编译。你能想象吗？一个 JIT 缓存回收导致 P99 延迟飙到 30 秒，而我们用了一周时间才定位到——因为我们没有端到端追踪。

这就是 OTel GenAI 要解决的问题。它不是让你"看得更清楚"，而是让你"不需要猜"。传统日志的问题在哪？每个组件各自打自己的日志，LLM 的日志、工具调用的日志、MCP 客户端的日志——就像三本独立的日记，没有页码对应关系。你想知道"这次用户请求慢在哪里"，需要人工把三本日记翻开，找到时间戳接近的条目，然后自己脑补因果关系。而这种手工关联，在 5 层的调用链上（Agent → LLM → Tool → MCP → 外部服务）纯粹是自虐。

OTel GenAI 的跨度层级就是来解决这个的。agent.invoke_agent 是顶层跨度，llm.chat 是子跨度，tool.execute 是孙子跨度，mcp.call 是曾孙跨度——全部共享一个 trace ID。一个 flame graph 展开，从用户输入到最终输出的每一毫秒花在哪，清清楚楚。我第一次把 Jaeger 的火焰图接上我们系统时，整个团队围过来看，那种感觉就像盲人突然恢复了视力——"卧槽，原来 token 生成只花了 2 秒，剩下 28 秒全在等工具调用！"

但这个标准最让我佩服的设计是"默认不捕获内容"。gen_ai.content.prompt 和 gen_ai.content.completion 这两个属性默认是关闭的——你需要显式设置环境变量才能打开。为什么？因为提示词里可能有用户的密码、API Key、内部 IP——在生产环境中无脑捕获这些数据等于自杀。我见过一个团队上线 OTel 后兴冲冲地开了全量内容捕获，三周后被安全审计发现，CIO 的脸都绿了。教训：追踪数据和敏感数据的边界，不是技术问题，是法律问题。

再说一个反模式——很多人以为只要接了 OTel 导出器就万事大吉了。错。最大的坑是 traceparent 传播。你的 Agent 调 MCP 服务器，如果 Stdio 传输上没有手动带 traceparent，MCP 这边的跨度就没有父跨度——在 Jaeger 里它是孤岛，跟 Agent 那边的跨度连不起来。这对于 Stdio 传输尤其麻烦，因为 Stdio 本来就不走 HTTP 头。你需要在 JSON-RPC 的 _meta 字段里手动塞 traceparent。我们第一次上线漏了这一步，花了两天才发现为什么 MCP 服务器的跨度在追踪视图里是"孤儿"。

关于导出器选型，我给你一个实战建议：开发环境用 Jaeger（docker run 一条命令就起），生产环境看预算。Langfuse 对 LLM 场景做了深度优化——Token 用量、成本估算、提示词版本对比，这些都是 Jaeger 没有的。Datadog 开箱即解析 gen_ai.* 属性，但贵。Arize Phoenix 偏评估方向——如果你做智能体质量评估，Phoenix 是加分项。我自己是灰度环境用 Langfuse（看质量和成本），生产环境用 Jaeger + Tempo（看延迟和错误率），两边都配 OTLP 导出，同一套代码不需要改。

最后说一个我踩过的坑，gen_ai.usage 的 token 计数在不同模型上可能不准。OpenAI 返回 usage.prompt_tokens 和 usage.completion_tokens，但你如果用 Anthropic 的 API，字段名是 input_tokens 和 output_tokens。OTel 语义约定统一为 gen_ai.usage.input_tokens 和 gen_ai.usage.output_tokens，但如果你直接用 provider 的原始字段填充，数值就对不上了。解决方法是统一做一个映射层，在填充跨度属性之前标准化所有 provider 的返回值。

**金句：没有追踪的智能体系统就是一个黑盒子，你只能祈祷它不出问题；有追踪的智能体系统是一个透明盒子，你可以精确地说出"第 3 次工具调用的第 2 个 MCP 请求多花了 18.7 秒"——前者靠信仰运维，后者靠数据运维。**
