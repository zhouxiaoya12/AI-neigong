# OpenTelemetry GenAI语义约定

> OpenTelemetry的GenAI SIG（2024年4月启动）定义了Agent遥测的标准模式。跨度名称、属性和内容捕获规则在供应商之间统一，使Agent追踪在Datadog、Grafana、Jaeger和Honeycomb中含义一致。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段14 · 13（LangGraph）、阶段14 · 24（可观测性平台）
**时间：** ~60分钟

## 学习目标

- 说出GenAI跨度类别：model/client、agent、tool。
- 区分`invoke_agent`的CLIENT与INTERNAL跨度以及各自适用场景。
- 列出顶级GenAI属性：provider name、request model、data-source ID。
- 解释内容捕获契约：主动选择（opt-in）、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题

每个供应商发明自己的跨度名称。运维团队最终为每个框架构建单独的仪表盘。OpenTelemetry的GenAI SIG通过定义整个生态系统瞄准的一个标准来解决这个问题。

## 核心概念

### 跨度类别

1. **模型/客户端跨度。** 覆盖原始LLM调用。由供应商SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent跨度。** `create_agent`（当Agent被构造时）和`invoke_agent`（当它运行时）。
3. **工具跨度。** 每个工具调用一个；通过父子关系连接到Agent跨度。

### Agent跨度命名

- 跨度名称：`invoke_agent {gen_ai.agent.name}`如果命名的话；回退到`invoke_agent`。
- 跨度类型：
  - **CLIENT** — 用于远程Agent服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内Agent框架（LangChain、CrewAI、本地ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由与请求不同）。
- `gen_ai.agent.name` — Agent标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于RAG：哪个语料库或存储被查询。

针对Anthropic、Azure AI Inference、AWS Bedrock、OpenAI存在技术特定的约定。

### 内容捕获

默认规则：插桩默认SHOULD NOT捕获输入/输出。捕获通过以下方式主动选择：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在跨度上记录引用（指针ID，而非内容文本）。这是第27课的内容投毒防御接入可观测性。

### 稳定性

截至2026年3月，大多数约定处于实验阶段。通过以下方式选择稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生映射GenAI属性到其LLM可观测性模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式出问题的地方

- **在跨度中捕获完整提示。** PII、密钥、客户数据出现在运维人员可读的追踪中。存储在外部。
- **没有`gen_ai.provider.name`。** 多供应商仪表盘在缺少归因时失效。
- **跨度缺少父链接。** 孤儿工具跨度。始终传播上下文。
- **未设置稳定性选择。** 你的属性可能在后端升级时被重命名。

## 构建它

`code/main.py` 实现一个符合GenAI约定的标准库跨度发射器：

- 带有GenAI属性模式的`Span`。
- 带有`start_span`、嵌套上下文的`Tracer`。
- 一个脚本化的Agent运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的跨度、LLM调用的`chat`跨度。
- 内容捕获模式：外部存储提示、在跨度上记录ID。

运行它：

```
python3 code/main.py
```

输出：一棵包含所有必需GenAI属性的跨度树，以及显示选择加入内容引用的"外部存储"。

## 使用它

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第24课）——自动插桩生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始OTel追踪；从GenAI属性构建仪表盘。
- **自托管** — 运行带有GenAI处理器的OTel Collector。

## 发布物

`outputs/skill-otel-genai.md` 将OTel GenAI跨度接入现有Agent，包含内容捕获默认值和外部引用存储。

## 练习

1. 用`invoke_agent`（INTERNAL）+每个工具的跨度为你的第01课ReAct循环插桩。发送到Jaeger实例。
2. 以"仅引用"模式添加内容捕获：提示存SQLite，跨度属性仅携带行ID。
3. 阅读`gen_ai.data_source.id`的规范。将其接入你的第09课Mem0搜索。
4. 设置`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`并验证你的属性不会被收集器重命名。
5. 构建一个仪表盘："哪些工具错误与哪些模型相关"，仅使用GenAI属性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| GenAI SIG | "OpenTelemetry GenAI小组" | 定义模式的OTel工作组 |
| invoke_agent | "Agent跨度" | 表示Agent运行的跨度名称 |
| CLIENT跨度 | "远程调用" | 调用远程Agent服务的跨度 |
| INTERNAL跨度 | "进程内" | 进程内Agent运行的跨度 |
| gen_ai.provider.name | "供应商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG来源" | 检索命中了哪个语料库/存储 |
| 内容捕获 | "提示日志" | 主动选择捕获消息；在生产中存储在外部 |
| 稳定性选择加入 | "预览模式" | 固定实验约定的环境变量 |

## 进一步阅读

- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认GenAI跨度
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置OTel跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C追踪上下文传播

---

## 📝 教师备课总结与读后感

**文档评价：** 这份文档解决了一个看似枯燥但极其关键的工程问题：标准化Agent遥测。文档的价值不在于技术深度（OTel GenAI约定本身仍在实验阶段），而在于它建立了一个"统一追踪模式"的思维框架——CLIENT vs INTERNAL跨度、内容捕获的选择加入原则、外部引用存储的安全实践。这个统一模式是第24课（可观测性平台）的前置基础——没有标准化的跨度，三个平台的互操作性无从谈起。

**知识结构：** 核心是GenAI跨度的三层结构（模型/客户端、Agent、工具）和CLIENT/INTERNAL的类型区分。属性系统（provider.name、request.model、data_source.id等）提供了跨供应商的统一查询能力。内容捕获的"opt-in + 外部引用"设计体现了安全优先的遥测哲学。稳定性选择加入机制则是实验阶段规范的生产落地保障。

**核心洞察（5-7条）：**
1. OTel GenAI约定的核心价值不是定义"怎么追踪"，而是定义"追踪数据应该长什么样"——这是跨供应商可观测性的前提条件。
2. CLIENT vs INTERNAL跨度的区分本质上对应"远程服务调用"vs"本地函数调用"——这是分布式系统中已成熟的模式在Agent领域的应用。
3. 内容捕获的"默认不捕获，opt-in才捕获"设计是安全第一的体现——Agent的提示和输出可能包含用户PII、商业机密和密码，运维人员不应能看到这些。
4. 外部引用存储（跨度上只有指针ID，实际内容在单独存储中）解决了"可审计性"和"安全性"的矛盾——需要时可以溯源，但日常运维看不到敏感内容。
5. `gen_ai.data_source.id`是RAG系统可观测性的关键属性——它让运维人员可以回答"Agent使用了哪个知识库来回答这个问题？知识库是否有问题？"
6. 稳定性的实验状态（2026年初）是一个风险提示——在生产中固定属性名称的环境变量不是可选项，是必须项，否则后端升级时属性被重命名会导致仪表盘全部失效。
7. GenAI SIG的成立（2024年4月）标志着一个重要转折点——Agent行业开始从"各自为政的追踪"走向"统一的遥测标准"，这类似于HTTP从各家自定义到RFC标准化的过程。

**教学建议（5-7条）：**
1. 用"巴别塔"比喻开场：没有OTel GenAI约定前，每个Agent框架的追踪数据用不同的"语言"——Datadog听不懂LangChain的数据，Jaeger看不懂Anthropic的跨度。OTel GenAI就是"通用翻译器"。
2. 三层跨度结构的讲解应该可视化：画一棵追踪树（Agent跨度→多个工具跨度→每个工具调用的LLM跨度），让学生直观感受Agent运行的"骨架"。
3. CLIENT vs INTERNAL的区分可以用具体代码示例讲解——展示同一个Agent在"本地LangChain运行"（INTERNAL）vs"调用OpenAI Assistants API运行"（CLIENT）时跨度类型的不同。
4. 内容捕获的"opt-in"原则需要用安全故事讲解——"想象一下，你的Agent帮助用户填过税务表单，包含社保号。如果完整提示被记录在运维人员看得到的追踪中，这是多大的合规风险？"
5. 练习部分应引导学生"在自己的Agent中加一行跨度发射代码，然后去Jaeger看到它"——从零到一的第一步是克服"插桩太复杂"的心理障碍。
6. `gen_ai.data_source.id`的讲解应该与RAG教学（第7-10课记忆部分）联动——让学生理解追踪属性如何连接"检索质量"到"答案质量"。
7. 稳定性选择加入的教学应该用一个实际演示：先不设环境变量，展示属性名；然后设置环境变量，展示属性名是否变化——让学生直观感受"固定名称"的重要性。

**值得补充（3-5条）：**
1. 补充一个"OTel GenAI约定的演进路线图"——从2024年4月启动到2025-2026年的关键里程碑，帮助学生判断"现在投入是否太早"。
2. 增加"自定义属性的最佳实践"——除了标准GenAI属性，什么时候应该添加自定义属性？如何命名以避免与未来标准冲突？
3. 补充"追踪采样策略"——对于高吞吐量的Agent系统，100%追踪不现实，如何设计采样率以平衡成本和可观测性覆盖率？
4. 增加"跨服务追踪传播"——当Agent系统是分布式的（多个微服务、多个Agent协作），如何确保追踪上下文在服务边界之间正确传播？
5. 补充"追踪数据存储的选型"——Jaeger用Cassandra/Elasticsearch，Grafana Tempo用对象存储，Datadog是SaaS——不同的存储后端对查询能力和保留策略的影响。

**一句话总结：** OTel GenAI约定不是又一个"追踪格式"，而是Agent可观测性从各自为政走向统一生态的"通用语言"——没有它，你的Agent追踪数据就是一座只有你自己能读的巴别塔。

---

# 🎓 Agent 架构课：OTel GenAI——你的Agent追踪，如果连自己都说不清是什么，凭什么指望运维能看懂？

一个问题开场：**你有三个Agent，一个用LangChain写的，一个用OpenAI Agents SDK，一个直接调Claude API。三个Agent都出了问题。你的运维团队打开仪表盘——三个Agent的追踪数据显示的格式各不相同，标注的"操作"名称互相矛盾，一个叫"llm_call"，一个叫"model_invoke"，第三个根本没有这个标签。你需要多久才能知道是哪个环节崩溃了？**

如果答案是"几个小时"，好消息是这不是你的运维团队的问题。坏消息是这是你的追踪设计的问题。

## GenAI OTel约定的核心不是技术规范，是跨系统沟通的语言

我在设计生产系统的时候，最早意识到的一个问题是：Agent系统天生是异构的。你不太可能只用一种框架、一个模型供应商、一套工具。但追踪数据——你的Agent的"飞行记录仪"——必须是统一的。

这不是一个技术问题，这是一个沟通问题。你的Agent运行时产生数据 → 你的可观测性平台消费数据 → 你的运维团队根据数据做决策。如果这三个环节的语言都不一样，那追踪就是一堆垃圾。

OTel GenAI约定做的事情很简单：**给所有Agent系统的追踪数据定义一套共同的词汇。** `invoke_agent`就是Agent运行，不管你是LangChain还是Claude Code。`gen_ai.provider.name`就是供应商名称，不管是`openai`还是`anthropic`。`gen_ai.operation.name`就是操作类型，不管是`chat`还是`tool_call`。

这套词汇让你可以在Datadog里统一查询"所有Agent的`invoke_agent`跨度中，哪些工具调用的耗时最长"，而不用为每个框架写不同的查询语句。听起来很简单？这就是标准化最大的价值。

## CLIENT vs INTERNAL：一个影响Dashboard设计的关键区分

我在实际部署中踩过一个坑，关于CLIENT和INTERNAL跨度的区分。很多人觉得这只是个"标签"，无所谓。但它的实际影响是什么？

CLIENT跨度表示你调用了一个远程Agent服务——比如OpenAI的Assistants API。这意味着Agent的运行时在OpenAI那边，你的追踪系统能看到的是"你调用了它"以及"它返回了什么"，但看不到Agent内部的步骤细节。

INTERNAL跨度表示Agent运行在你的进程内——你的LangChain、你的ReAct循环。这意味着你可以看到Agent的每一步决策、每一个工具调用、每一次LLM请求。

这个区分决定了两件事：第一，你的仪表盘的深度。CLIENT跨度的Agent，你只能做"黑盒监控"（输入、输出、耗时、成本）。INTERNAL跨度的Agent，你可以做"白盒调试"（为什么Agent选择了这个工具？它在第几步陷入了循环？）。第二，你的安全策略——我马上就要讲到这个。

## 内容捕获的"opt-in"：这不是设计缺陷，这是安全设计

这里是整份规范里我最欣赏的设计决策。OTel GenAI约定的默认行为是：**不捕获**提示内容和输出内容。

初看这很反直觉——追踪系统不记录Agent说了什么？那我怎么调试？

但想想你的Agent在做什么。它在帮用户写邮件——用户的邮件内容包含商业机密。它在帮用户调试代码——代码里有API密钥。它在帮用户做医疗咨询——对话里有个人健康信息。如果所有这些内容都被记录在追踪系统中，而运维人员（或任何有权限访问追踪系统的人）都能看到——这就是一场合规灾难。

规范的方案是：如果你想捕获内容，你主动选择（opt-in），并且推荐你把内容存储在外部（S3、加密日志系统），追踪跨度上只放一个引用ID。这样，调试时可以按ID检索内容，但日常运维中没人能看到隐私数据。

我在生产环境中还加了一层：内容存储有独立的访问控制和审计日志。谁检索了内容、什么时候、因为什么——全部记录。这不仅是工程实践，在GDPR/CCPA等法规环境下是合规要求。

## 稳定性实验中的风险

2026年初的OTel GenAI约定大部分还是"实验性"的。这意味着什么？OpenTelemetry可能在任何版本中改变属性名称。

想象一下：你花了两周时间，基于`gen_ai.operation.name`构建了一套精美的仪表盘和告警规则。然后OTel发布了一个新版本，这个属性被重命名为`gen_ai.operation.type`。你的仪表盘全部失效。你的告警全部静默。你甚至不知道你的Agent已经出问题了。

这就是为什么`OTEL_SEMCONV_STABILITY_OPT_IN`不是可选项——它是必须项。这个环境变量告诉OTel Collector："请使用这个特定版本的属性名称，不要擅自更新。"在生产环境中，我在部署脚本里强制设置这个变量，并且把它写进了部署检查清单——如果这个变量没设置，部署不能通过。

## 结语清单

在你接入Agent追踪之前：
1. 你的追踪数据格式是OTel GenAI标准吗？如果不是，先用这个标准再谈可观测性。
2. 你的Agent使用CLIENT跨度还是INTERNAL跨度？这决定了你能做黑盒监控还是白盒调试。
3. 你是否在追踪跨度中捕获了完整的提示和输出？如果是，里面有PII或密钥吗？有的话先改成外部引用存储。
4. 你的OTel环境变量设置了稳定性选择加入吗？没有的话，等一次属性重命名，你的仪表盘会全部静默。
5. 你的`gen_ai.data_source.id`设置了没有？没有的话，当RAG系统出问题时你无法追溯到哪个知识库出了问题。

**金句：Agent追踪数据是你的Agent的"飞行记录仪"。OTel GenAI约定确保这个记录仪说的是所有调查人员都能理解的语言。如果你的记录仪说的话只有你自己懂，那事故调查的时候就只有你自己能查——而你往往就是当事人。**
