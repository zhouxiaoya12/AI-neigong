# A2A — 智能体到智能体协议

> MCP 是智能体到工具。A2A（Agent2Agent）是智能体到智能体——一个开放协议，让基于不同框架构建的不透明智能体可以协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月发布 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，并增加了 AP2 支付扩展。本课走读 Agent Card、Task 生命周期以及两种传输绑定。

**类型：** 构建
**语言：** Python（标准库、Agent Card + Task 框架）
**前置课程：** Phase 13 · 06（MCP 基础）、Phase 13 · 08（MCP 客户端）
**用时：** 约 75 分钟

## 学习目标

- 区分智能体到工具（MCP）与智能体到智能体（A2A）的用例。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的 Agent Card。
- 走读 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用包含 Parts（text、file、data）的 Messages 和作为输出的 Artifacts。

## 问题

一个客服智能体需要将报告撰写委托给一个专业的写作智能体。A2A 之前的选项：

- 自定义 REST API。能用，但每个配对都是一次性工作。
- 共享代码库。要求两个智能体运行相同的框架。
- MCP。不合适：MCP 用于调用工具，而不是让两个智能体协作，同时保持每个智能体不透明的内部推理。

A2A 填补了这一空白。它将交互建模为一个智能体向另一个智能体发送 Task，具有生命周期、消息和产出物。被调用智能体的内部状态保持不透明——调用方只看到任务状态转换和最终输出。

A2A 是"让跨框架的智能体相互对话"的协议。它不替代 MCP；二者互补。

## 概念

### Agent Card

每个兼容 A2A 的智能体在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "总结学术论文并起草引用。",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "总结论文",
      "description": "阅读论文 PDF 并生成三段式摘要。",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现方式是基于 URL 的：获取卡片，了解 A2A 端点的 URL，枚举技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 增加了密码学签名。发布者用 JWT 签名自己的卡片；消费者验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端通过 `tasks/send` 发起。被调用智能体在各状态间转换；客户端通过 SSE 订阅状态更新或轮询。

### Messages 和 Parts

一条 Message 携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 二进制数据。
- `data` — 带类型的 JSON 负载（给被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "总结这篇论文。"},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 段落"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。一个 Artifact 是带名称、带类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为块进行流式传输。调用方负责累积。

### 两种传输绑定

1. **基于 HTTP 的 JSON-RPC。** `/a2a` 端点，POST 用于请求，可选的 SSE 用于流式传输。默认绑定。
2. **gRPC。** 适用于 gRPC 为原生协议的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明性保持

一个关键设计原则：被调用智能体的内部状态是不透明的。调用方看到任务状态和产出物。被调用智能体的思维链、工具调用、子智能体委托——全部不可见。这与 MCP 不同，MCP 的工具调用是透明的。

理由：A2A 使竞争对手能够协作而不暴露内部实现。A2A 可以"调用这个客服智能体"而无需调用方了解该智能体如何实现服务。

### 时间线

- **2025-04-09。** Google 发布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 用例 | 智能体到工具 | 智能体到智能体 |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用方 | Agent 运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 具有生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | 基于 HTTP 的 JSON-RPC / gRPC |

当你想要调用一个特定工具时使用 MCP。当你想将整个任务委托给另一个智能体时使用 A2A。许多生产系统同时使用两者：智能体用 MCP 作为其工具层，用 A2A 作为其协作层。

## 实战

`code/main.py` 实现了一个最小化 A2A 框架：一个研究智能体发布其卡片，一个写作智能体接收包含一个 PDF 和一条文本指令的 `tasks/send`，经历 working → input_required → working → completed 的状态转换，并返回一个文本产出物。全部使用标准库；使用内存传输来专注于消息形态。

需要关注的内容：

- Agent Card JSON 形态。
- Task ID 分配和状态转换。
- 包含混合类型 Parts 的 Messages。
- 任务中途的 input-required 分支。
- 完成时返回的 Artifact。

## 交付

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应可被其他智能体调用的新智能体，该技能产出 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用智能体请求澄清时的 input-required 暂停。

2. 添加一个签名 Agent Card。使用 HMAC 对卡片的规范 JSON 签名。编写一个验证器，并确认它在被篡改的卡片上失败。

3. 实现任务流式传输：写作智能体通过 SSE 发送三个增量产出物块，调用方累积它们。

4. 设计一个包装 MCP 服务器的 A2A 智能体。将每个 MCP 工具映射为一个 A2A 技能。注意权衡——哪些不透明性会丧失？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚无任何框架实现的一个功能。（提示：与多跳任务委托有关。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| A2A | "智能体到智能体协议" | 用于不透明智能体协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述智能体技能和端点的已发布元数据 |
| Skill | "一个可调用单元" | 智能体支持的命名操作（类比 MCP 工具） |
| Task | "委托单元" | 具有生命周期和最终产出物的作业项 |
| Message | "Task 输入" | 承载 Parts（text、file、data） |
| Part | "带类型的块" | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | "Task 输出" | 完成时返回的带名称、带类型的输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Card 扩展 |
| 不透明性 | "黑箱协作" | 被调用智能体的内部对调用方隐藏 |
| input-required | "Task 暂停" | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 正式规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理移交
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指导

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一篇结构清晰的协议入门课，以"A2A 填补了什么空白"为主线，从问题场景、核心概念（Agent Card、Task 生命周期、Parts/Artifacts 消息模型）到传输绑定和 MCP 对比，形成完整的认知闭环。文档特别擅长用"MCP 是 agent-to-tool，A2A 是 agent-to-agent"这一简洁二分法建立心智模型，并深入阐述了"不透明性"这一关键设计原则的深层理由——使竞争对手能够协作而不暴露内部实现。

### 二、知识结构梳理

- **协议定位层：** A2A 不是 MCP 的替代品，而是另一个维度的协议。MCP 解决"怎么调用工具"，A2A 解决"怎么委托任务给另一个智能体"。两者的传输层和状态模型完全不同。
- **核心原语层：** Agent Card（发现和元数据）、Task（带生命周期的委托单元）、Message/Part（输入载体）、Artifact（结构化输出）。这四个原语构成了 A2A 的最小完整概念集。
- **生态演进层：** 从 Google 发布（2025-04）到 Linux Foundation 治理（2025-06）、吸收 IBM ACP（2025-08）、AP2 支付扩展（2025-09）、到 v1.0 和 150+ 组织支持（2026-04）——这条时间线展示了一个开放协议从诞生到成熟的典型轨迹。

### 三、核心洞察

1. **不透明性是 A2A 的灵魂：** 如果 A2A 暴露了被调用智能体的内部推理，那它就成了一个加强版的 MCP。A2A 的不透明性设计使其适用于"竞争对手间的协作"——这是一个非常精巧的产品-架构对齐决策。
2. **Agent Card = 智能体的 API 文档：** `/.well-known/agent.json` 是 Web 世界中 `robots.txt` 和 OpenAPI spec 的精神继承者——用标准化的可发现端点描述能力边界。
3. **Task 生命周期模拟了人类工作流：** submitted → working → input-required → completed 的状态机天然支持"做了一半需要问问题"的人机/机机交互模式，比 request-response 模型更贴合真实协作。
4. **Parts 的多模态输入设计：** text/file/data 三种 Part 类型覆盖了自然语言、文件和数据三种基本输入形态，设计简洁但覆盖完整。
5. **AP2 的密码学签名是信任的基础：** 在一个智能体可以冒充另一个智能体的世界里，签名 Agent Card 不是可选功能而是必备功能。Postmark 案例（Phase 13 · 15）的教训同样适用于智能体身份。
6. **传输层解耦的智慧：** JSON-RPC over HTTP 和 gRPC 两种绑定共享同一逻辑消息模型——这是协议设计中经典的"语义与编码分离"原则。
7. **MCP + A2A 的组合范式：** 生产系统中工具层用 MCP、协作层用 A2A 的架构模式可能是 2026 年后 AI Agent 系统的标准架构图——如同微服务中 REST 用于同步调用、消息队列用于异步解耦。

### 四、教学建议

1. **用类比引入：** MCP 像函数调用，A2A 像把整个项目外包给另一个团队。这个类比能帮助有软件开发背景的学生快速建立直觉。
2. **Agent Card 动手练习：** 让学生为自己的一个现有项目编写 Agent Card JSON，然后互相验证对方的 Card 是否准确描述了能力。这比单纯阅读规范更有效。
3. **Task 生命周期状态机图：** 用 Mermaid 或 Draw.io 画出完整的状态转换图，标注每种转换的触发条件。可视化比文字列表好记得多。
4. **不透明性辩论：** 组织课堂讨论——A2A 的不透明性设计是优点还是缺点？在什么场景下你会想要突破不透明性？（这自然引出 Phase 13 · 20 的 OTel GenAI 追踪。）
5. **传输绑定对比实验：** 如果有时间，让学生分别实现 JSON-RPC 和 gRPC 版本的同一个 A2A 调用，体验两者的差异和各自适用场景。
6. **AP2 安全实验：** 让学生尝试"攻击"没有签名的 Agent Card——伪造技能声明、修改端点 URL——然后再用 AP2 签名保护，体验签名带来的安全提升。
7. **A2A vs MCP 决策树：** 设计一个练习，给出一系列场景让学生判断该用 MCP 还是 A2A。这能强化对两者边界的理解。

### 五、值得补充的内容

1. **多跳委托的具体协议设计：** 文档提到 v1.0 中尚无框架实现多跳任务委托，可以补充说明这一功能面临的分布式追踪、循环检测和责任归属等挑战。
2. **A2A 与 LangGraph/CrewAI 等框架的原生集成：** 可以补充一个简短的对比表，说明各主流多智能体框架对 A2A 的支持状态。
3. **AP2 支付扩展的实际用例：** 补充一个具体场景（如按任务复杂度计费、按 Token 消耗结算），说明 AP2 如何在商业智能体市场中发挥作用。
4. **A2A 的失败模式：** 文档侧重于"怎么做"，可以补充常见失败模式——超时处理、部分完成回滚、幂等性保证——这些在生产中至关重要。
5. **streaming 的最佳实践：** 补充一个 streaming Artifacts 时的增量状态管理和错误恢复策略的简要讨论。

### 六、一句话总结

A2A 用 Agent Card 发现、Task 生命周期委托、不透明性保护三大设计支柱，构建了一个让跨组织、跨框架的智能体能够"外包任务"的开放协议——它不是 MCP 的替代品，而是 AI Agent 生态中协作维度的协议补全。



---

# 🎓 Agent 架构课：A2A 协议——当你的智能体需要外包工作给别人的智能体

同学们好。我是你们的FDE工程老师，今天讲的是 A2A 协议。

先搞清楚一个很容易搞混的问题：**MCP 是智能体→工具的协议。A2A 是智能体→智能体的协议。不是竞争——是互补。**

你的客服智能体要把"写分析报告"外包给写作智能体——你怎么做？REST API？每个配对都要写新接口。暴露成 MCP 工具？工具调用是同步的"问→答"，没有 Task 生命周期、没有"在做中，等一会儿"。A2A 填补的空白就是：**两个不透明智能体之间，需要一个标准方式说"我委托你一个任务，你做完告诉我"。**

## Agent Card + Task 生命周期

每个 A2A 智能体在 `/.well-known/agent.json` 发布名片——名称、技能列表、端点 URL。很像 MCP 的 `tools/list`。

Task 有状态机：`submitted → working → input-required → working → completed/failed/canceled/rejected`。"input-required"不是报错——是"我需要你补充信息"。被调用智能体内部不透明——你只看到状态转换和最终 Artifacts。

Messages 是多模态的：Parts 可以是 text、file、data——一个写作智能体可以收 PDF 文件、产 Markdown 文档。

## 结课清单

1. **MCP ≠ A2A。** MCP 智能体→工具，A2A 智能体→智能体。互补。
2. **Agent Card = 能力发现。** `/.well-known/agent.json`。
3. **Task 有六种状态。** 不只是"调了返回"——有完整生命周期。
4. **被调用方内部不透明。** 只看到状态转换和产出物。

最后一句话：**A2A 不是 MCP 的竞争者——它是 MCP 做不到的事。MCP 让智能体调工具，A2A 让智能体委托任务给另一个智能体。你的系统两个都需要。**

---

# 💼 从业者故事：A2A 协议——当你的智能体需要外包工作给别人的智能体

去年 12 月，凌晨 3 点，我在修一个客服系统。架构很简单——用户的智能体调我们的智能体干活。我们两个团队各写了一个 REST API，然后花了两周对接口：字段名对不上、错误码不一样、状态机各玩各的。最离谱的是，他们那边重构了一个参数名，我们这边没收到通知，生产环境静默挂了 4 个小时。PM 问我为什么不能像 MCP 那样有个标准协议，我说因为 MCP 是给智能体调工具用的，不是给智能体之间互相调用的。PM 沉默了三秒，说了一句让我记到现在的话："所以两个智能体之间，就回到了用手搓 API 的蛮荒时代？"

这句话扎心。因为答案就是：对。2025 年初的 Agent 生态，MCP 解决了"我怎么用工具"，但没解决"我怎么把活外包给另一个智能体"。你想想这个场景有多常见：客服智能体需要翻译智能体翻译一段话，翻译智能体需要写作智能体润色一下，写作智能体又需要研究智能体查点资料——这个调用链，在 A2A 出来之前，就是噩梦。每个配对都是一次性的 API 联调，像小时候玩翻花绳，绳子越多越容易打结。

A2A 要干的事情就是：让智能体之间外包任务，像人类之间外包工作一样自然。你对同事说"帮我写个报告"，你不会关心他用什么编辑器、怎么思考的、中间喝了多少咖啡。你只关心三件事：他接了没有、做到哪了、最终产出是什么。A2A 的 Agent Card 就是名片，Task 生命周期就是进度条，Artifact 就是最终交到你手上的文件。

但这套模型有个决策让我一开始很不爽——"不透明性"。A2A 的设计铁律是：被调用的智能体内部状态对调用方完全不可见。他的思维链、他的工具调用、他委托给子智能体的过程——你全都看不见。我当时觉得这太傻了，出了问题怎么调试？后来我才想明白：这就像你外包公司的会计，你不需要也不应该看到他的 Excel 公式和鼠标操作记录。你只需要看到月度报表。而且如果你能看到他的内部实现，你就跟他绑死了——他换了工具你也要改代码。不透明性不是 bug，是 feature。它让竞争对手的智能体可以互相调用而不暴露商业机密。一个 Salesforce 的智能体调一个 SAP 的智能体——双方的 IP 都安全。

说到技术细节，有个设计我觉得特别漂亮，但很少有人提到：Parts 的 text/file/data 三种类型覆盖了 99% 的输入场景。text 是自然语言，file 是二进制载体，data 是结构化 JSON。你不需要第 4 种类型。这种"刚好够用"的设计，比那些一上来就搞 20 种 MIME 类型的过度设计高明一万倍。我见过太多协议死于"过度扩展性"——A2A 在这点上很克制。

不过也有坑。最大的坑是 Task 生命周期的 input-required 状态。理论上很美好：被调用智能体做了一半发现信息不够，暂停，请求输入，然后继续。但实际上，如果你的智能体在 input-required 状态等了 30 分钟没收到回复，它应该怎么做？超时当失败？保持等待？协议规范没说，需要你自己决定。我们生产环境遇到过：上游挂了，task 永远卡在 input-required，下游的调用方也在等这个 task——一个 task 卡住，整条调用链全堵了。后来我们加了一个 10 分钟的 input-required 超时，超时自动 fail，才算搞定。

另一个容易被忽略的是 streaming Artifacts。协议支持分块流式传输输出，但调用方要自己累积和拼接。如果你用的 HTTP JSON-RPC 绑定走 SSE，浏览器端还要处理 EventSource 的重连逻辑。我们第一次实现时因为 SSE 断连丢了一个 Artifact 块，导致最终输出少了中间一段——用户看到的是"根据研究显示...结论是 42"，中间的论证过程全没了。排查了 3 个小时才发现是网络闪断导致的。教训：流式传输一定要做完整性校验，比如每个 chunk 带序号，或者最后发一个 checksum。

最后说一个你可能感兴趣的数据：A2A 从 2025 年 4 月 Google 发布，到 2026 年 4 月 v1.0，短短一年内从 0 到 150+ 支持组织。这个速度在协议领域是罕见的。对比一下：HTTP 从 1991 年 0.9 到 1.0 用了 5 年。A2A 快不是因为它更优秀（虽然它确实设计得不错），而是因为需求太痛了——每个做 Agent 的团队都在自己搓 API，痛到愿意拥抱任何标准。

**金句：MCP 让智能体有了手，A2A 让智能体有了同事——前者解决了工具调用，后者解决了组织协作。你的 Agent 系统如果只有 MCP 没有 A2A，就像一个只有工具没有同事的工位：能干活，但永远一个人扛。**
