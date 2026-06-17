# A2A — Agent间协议

> Google 于 2025 年 4 月发布了 A2A；到 2026 年 4 月，规范已发布在 https://a2a-protocol.org/latest/specification/，并有 150+ 组织支持。A2A 是 MCP（第 13 课）的水平层面补充：MCP 是垂直的（Agent ↔ 工具），而 A2A 是对等的（Agent ↔ Agent）。它定义了 Agent Card（发现机制）、带产物（文本、结构化数据、视频）的任务、不透明的任务生命周期以及认证机制。生产系统越来越多地将 MCP 与 A2A 配对使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型：** 学习 + 构建
**语言：** Python（标准库、`http.server`、`json`）
**前置课程：** Phase 16 · 04（基元模型）
**时间：** 约 75 分钟

## 问题

你的 Agent 需要调用另一个系统上的 Agent。怎么做？你可以暴露一个 HTTP 端点，定义一套自定义 JSON 模式，然后指望对方也讲这个模式。每一对 Agent 都会变成一个定制集成。

A2A 就是为这种调用设计的通用传输协议。标准化的发现机制、标准化的任务模型、标准化的传输、标准化的产物。就像 HTTP+REST，但把 Agent 当作一等公民。

## 概念

### 四大要素

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述 Agent：名称、技能、端点、支持的模式、认证要求。通过读取这张卡片来完成发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task（任务）。** 工作单元。一个有状态的异步对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**Artifact（产物）。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。产物带有类型，使不同模式成为一等公民。

**不透明生命周期。** A2A 不规定远端 Agent *如何* 解决任务。客户端看到状态转换和产物；实现端可以自由使用任何框架。

### MCP/A2A 的分工

- **MCP**（第 13 课）：Agent ↔ 工具。Agent 通过 JSON-RPC 向工具服务器读写。默认无状态。
- **A2A**：Agent ↔ Agent。对等协议；双方都是具备自身推理能力的 Agent。

生产环境中的多 Agent 系统同时使用两者。一个 A2A 对等方在自己这端调用 MCP 工具。这种划分使两个关注点保持清晰。

### 发现流程

```
客户端                      Agent 服务器
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON──────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted─
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts───
```

或者使用流式传输：SSE 订阅 `/tasks/{id}/events` 以获取推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间相互证明身份。
- **签名请求** — 对载荷进行 HMAC 签名。

认证信息在 Agent Card 中声明；客户端发现后依此执行。

### 到 2026 年 4 月已有 150+ 组织支持

企业级采用推动了 A2A 的规模化发展。关键是：A2A 成为企业 Agent 系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都发布了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** 公司 A 的 Agent 调用公司 B 的 Agent。没有 A2A，每一对都是定制合约。
- **异构框架。** LangGraph Agent 调用 CrewAI Agent 调用自定义 Python Agent。A2A 将其标准化。
- **类型化产物。** 视频结果、结构化 JSON、音频 — 全部是一等公民。
- **长时间运行的任务。** 不透明生命周期 + 轮询使数小时的任务变得简单直接。

### A2A 的不足

- **对延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的 Agent 间调用不适合；应使用直接 RPC。
- **紧密耦合的进程内 Agent。** 如果两个 Agent 在同一个 Python 进程中运行，A2A 的 HTTP 往返是多余的。
- **小团队。** 规范开销确实存在；仅限内部的 Agent 可能不需要这种形式化。

### A2A 与 ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）— A2A 的前身，范围较窄。
- **ANP**（Agent Network Protocol）— 以对等方发现为主，去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279（Liu 等，"A Survey of Agent Interoperability Protocols"）了解对比。

## 动手构建

`code/main.py` 实现了一个 A2A-minimal 服务器和客户端，使用 `http.server` 和 JSON。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 上返回产物。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后运行客户端与之交互。你可以看到完整的流程：发现、提交、轮询、获取产物。

## 实践应用

`outputs/skill-a2a-integrator.md` 设计了一个 A2A 集成方案：Agent Card 内容、任务模式、认证选择、流式传输 vs 轮询。

## 上线检查清单

- **固定规范版本。** A2A 仍在演进中；Agent Card 应声明协议版本。
- **幂等的任务创建。** 重复提交（网络重试）应只产生一个任务。
- **产物模式。** 声明 Agent 返回的结果形状；消费者应进行校验。
- **速率限制 + 认证。** A2A 是面向公网的；应用标准的 Web 安全措施。
- **失败任务的死信队列。** 随时间检查模式，发现重复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现了服务器并收到了正确的产物。
2. 为服务器添加第二个技能（例如"summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现一个 SSE 流式端点：`/tasks/{id}/events`，发出状态变更。客户端需要做哪些不同的处理？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范要求但本演示未实现的三个特性。
5. 比较 A2A（Agent Card 发现）和 MCP（通过 `listTools` 列出服务器端能力）。自描述 Agent 与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| A2A | "Agent间协议" | Agent 调用其他 Agent 的对等协议，支持跨系统通信。Google 2025 年提出。 |
| Agent Card | "Agent 的名片" | `/.well-known/agent.json` 上的 JSON，描述技能、端点和认证方式。 |
| Task | "工作单元" | 具有生命周期的有状态异步对象；完成时产生产物。 |
| Artifact | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期 | "如何解决是 Agent 自己的事" | 客户端看到状态转换；服务器端可自由选择框架/工具。 |
| Discovery | "发现 Agent" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等方" | MCP：纵向 Agent ↔ 工具。A2A：横向 Agent ↔ Agent。 |
| ACP / ANP / NLIP | "兄弟协议" | 相近的规范；A2A 是 2026 年采用最广泛的。 |

## 扩展阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 权威规范
- [Google Developers Blog — A2A 发布公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu 等 — Agent 互操作性协议综述](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 对比

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一篇扎实的 A2A 协议入门文档，时间节点精准（2026 年 4 月），信息密度高但条理清晰。文档从"为什么需要 A2A"出发，先讲问题（自定义集成不可扩展），再讲方案（四大要素），最后动手构建和上线。结构类似 RFC 摘要加动手教程的混合体，适合有一定 MCP 基础的学员。亮点是 MCP/A2A 分工对比和 150+ 组织的生态现状，让学员理解这不是实验室规范而是已经在生产环境运行的协议。不足在于对认证机制的代码示例缺失，SSE 流式也只停留在练习中未展开。

### 二、知识结构梳理

- **基础层**：A2A 的四大要素 — Agent Card（发现）、Task（工作单元）、Artifact（产物类型）、不透明生命周期（实现自由）。核心流程：发现 → 提交 → 轮询 → 获取产物。
- **对比层**：MCP vs A2A 的分工（纵向工具 vs 横向对等），A2A 与 ACP/ANP/NLIP 的生态位置，A2A 适用场景与不适用的边界。
- **工程层**：认证三种模式（Bearer/mTLS/HMAC），幂等任务创建，速率限制，死信队列。生产上线检查清单覆盖了关键点。

### 三、核心洞察

1. **A2A 和 MCP 是互补的，不是竞争关系。** 一个生产 Agent 系统通常同时部署两者：A2A 做跨 Agent 协调，MCP 做工具调用。这种分工让两个规范的边界保持清晰。
2. **不透明生命周期是 A2A 的设计精髓。** 它避免了对远端 Agent 实现细节的假设 — 对方用什么框架、什么模型、什么推理策略，调用方不需要知道。这是实现互操作性的关键。
3. **Agent Card 里的 `/.well-known/agent.json` 借鉴了 Web 的 `/.well-known/` 惯例（如 `security.txt`），降低了集成方的认知成本。**
4. **A2A 不适合所有场景。** 文档明确列出了边界（微调用、进程内、小团队），这种自我认知比许多规范文档更诚实和实用。
5. **2026 年的生态信号很强：150+ 组织支持，Google Cloud 和 Microsoft 都有产品级支持。** 这表示 A2A 已经从"候选规范"变成了"事实标准"。
6. **A2A 解决了跨组织信任边界问题。** 企业 A 的 Agent 调用企业 B 的 Agent 不再是定制集成，而是像 REST API 调用一样标准。这是 B2B Agent 经济的协议基础。
7. **和 MCP 的课程顺序设计很好。** 先学 MCP（工具层），再学 A2A（对等层），上下贯通，学员能形成完整的 Agent 通信心智模型。

### 四、教学建议

1. **强调 MCP/A2A 的分工图。** 建议手绘或动画展示：MCP 是"Agent 伸手拿工具"，A2A 是"Agent 和 Agent 握手"。这个图像比文字解释更直观。
2. **先跑 `code/main.py` 再讲理论。** 让学员亲眼看到发现→提交→轮询→产物获取的完整流程，再回来看规范细节，理解深度会大幅提升。
3. **认证部分需要补充代码示例。** 文档只列了三种模式但没有代码，可以给 Bearer token 模式加一个最小示例。
4. **SSE 流式练习是重要的动手环节。** 建议留足够时间让学员完成练习 3，因为 SSE 是生产环境中 A2A 的主流传输方式之一。
5. **比较 A2A 和 REST API 设计。** 通过对比让有 Web 开发背景的学员快速建立心智模型：Agent Card ≈ API 文档 + OpenAPI spec，Task ≈ 异步资源，Artifact ≈ 响应体。
6. **用 OpenAPI/Swagger 类比来解释 Agent Card。** 这能帮助学员理解"自描述"的价值，也解释了为什么 `/.well-known/` 路径是好的选择。
7. **留出时间讨论"什么时候不需要 A2A"。** 文档的"不足之处"部分非常宝贵，避免学员在简单场景中过度工程设计。

### 五、值得补充的内容

1. **A2A 安全攻击面分析。** Agent Card 篡改、任务注入、产物投毒 — 这些在文档中只被认证部分隐含覆盖，值得专题讨论。
2. **A2A 与 gRPC 的对比。** gRPC 在微服务中广泛使用，为什么 Agent 间通信需要新协议？这个比较能帮助理解 A2A 的设计决策。
3. **多轮对话中的 A2A 状态管理。** 文档讲的是单次任务，但实际场景中 Agent 对话是多轮的，需要补充会话级状态管理。
4. **MCP + A2A 同一套系统里的组合部署架构图。** 展示一个同时使用 MCP（内部工具）和 A2A（外部对等方）的生产 Agent 系统全貌。

### 六、一句话总结

**A2A 是 Agent 世界的 HTTP+REST：标准化发现、标准化任务模型、标准化产物类型，让 Agent 间通信从定制集成变成可组合的基础设施。**



---

# 🎓 Agent 架构课：A2A 协议——不学它，你的 Agent 永远只能自言自语
同学们好。我是你们的FDE工程老师，今天讲的是 A2A协议。

**没有 A2A，多智能体之间的通信就是"每家自己搓一个 API"。两个团队，两套 REST 接口，两周对字段名，重构一个参数名就静默挂 4 小时。** A2A 做了 MCP 对工具做的事——给了智能体之间一个标准语言。

Agent Card（`/.well-known/agent.json`）是名片。Task 生命周期（submitted→working→completed/failed/canceled）是状态机。Messages 和 Parts（text/file/data）是多模态消息体。Artifacts 是最终产出物。被调用智能体内部不透明——你只看到状态转换和产出。

跟 MCP 的关系：MCP 是你叫一个函数，A2A 是你委托一个项目。MCP 的答案是"这是结果"，A2A 的答案是"我在做，等一下"或"做完了，这是成果"。

## 结课清单

1. **A2A 是智能体之间的通用语。** MCP 管工具，A2A 管委托。
2. **Agent Card = 名片。Task = 状态机。Messages = 多模态。**
3. **被调用方不透明。** 你不需要知道对方用什么框架。

最后一句话：**不学 A2A，你的每个智能体都只能说自己的方言——学 A2A，它们就加入了一种通用语网络。**

---

# 💼 从业者故事：A2A 协议——不学它，你的 Agent 永远只能自言自语

我在 2025 年夏天做了个蠢事。

我们公司有两个 Agent：一个做金融合规审查（合规 Agent），一个做合同生成（合同 Agent）。合规 Agent 在部门 A 的服务器上，合同 Agent 在部门 B 的服务器上。产品需求是"合同 Agent 生成合同后，自动发给合规 Agent 审查"。

两个 Agent 都是我们自己写的。两个 Agent 都在自己的 Python 进程里运行。两个 Agent 都有 HTTP 端点。我心想：这不就是个 API 调用吗？我花了半天写了一个简单的 JSON 协议——`{"action": "review", "document": "..."}`——然后让合同 Agent 在生成后 POST 到合规 Agent。

完美。直到需求变了。

三个月后，产品说"还需要调用法务部门的 Agent，但他们用 LangGraph，我们用的是自己写的框架"。又过了一个月，"客户的 Agent 也要能调用我们的合规审查"——客户是另一家公司，完全不同的技术栈。

我那个"半天写的小协议"撑不住了。我逐渐加字段、加版本号、加错误码、加认证——用了六个月，我把一个简单的 JSON RPC 变成了一个蹩脚的、只有我一个开发人员能看懂的、没有文档的私有协议。

然后 Google 发布了 A2A。我读了规范，沉默了三秒钟。他们把我要花六个月才能做对的事情，做成了一个标准。**我那个蹩脚的私有协议就是 A2A 要消灭的东西。**

## A2A 不是又一个协议——是 Agent 世界的 HTTP

想想 90 年代初的互联网。每个网站有自己的通信方式。你要访问 A 网站用 FTP，访问 B 网站用 Gopher，访问 C 网站用 email。然后 HTTP 出现了——一个统一的请求/响应格式。突然之间，任何浏览器可以访问任何网站。

**A2A 对 Agent 做的事情，就是 HTTP 对互联网做的事情。** 任何一个 Agent（不管用什么框架、什么语言、什么模型）可以向任何其他 Agent 发起标准化的请求。

但这个类比有一个关键的延伸：A2A 不是 REST。A2A 的 Task 是有状态的异步对象——`submitted → working → completed`。这不是"发请求等响应"，是"我丢给你一个任务，你有空了做，做好了告诉我"。你提交任务后拿到的是一个 task_id，然后你轮询或者订阅状态更新。

为什么设计成异步？因为 Agent 任务可能需要几分钟甚至几小时。合规审查可能要读 200 页合同然后交叉引用 50 条法规。你不可能让调用方等一个 HTTP 连接 open 三个小时。异步任务模型是 Agent 间通信的第一个硬需求。

## 四个要素——从卡片到产物

**Agent Card（名片）。** 每个 A2A Agent 暴露一个 `/.well-known/agent.json`。你 GET 它，得到这个 Agent 的技能列表、端点、认证要求。不需要文档，不需要人工沟通。你的 Agent 可以**自动发现**另一个 Agent 能做什么。

这跟 OpenAPI spec 一样重要。OpenAPI 让工具知道怎么调用 API。Agent Card 让 Agent 知道怎么调用另一个 Agent。但从实用性来说更好——因为 Agent Card 不仅是机器可读的 schema，它还包含自然语言描述的"技能"（`"skills": ["review-contracts", "check-compliance"]`），你的 Agent 可以用自然语言推理"这个 Agent 能不能帮到我"。

**Task（任务）。** 工作单元。有生命周期。你的 Agent POST 一个任务说"帮我审查这份合同"，对方返回 201 + task_id。你后续可以 GET `/tasks/{id}` 查进度。对方 Agent 在后台跑——可能用了 5 个 LangGraph 节点、调了 3 个 MCP 工具、生成了 2 个子 Agent——你一概不知。你只知道状态在变：`submitted → working → working (42%) → completed`。这就是"不透明生命周期"的价值。

**Artifact（产物）。** 任务完成后的产出。不是"纯文本"——是带类型的：`{"type": "structured", "data": {...}}` 或 `{"type": "text", "data": "..."}` 或 `{"type": "image", "url": "..."}`。文本、JSON、图片、视频、音频都是一等公民。这很重要——很多 Agent 协议把"输出"默认为纯文本，但你的合规 Agent 的输出可能是一个带 47 个字段的结构化审查报告，不是一段散文。

## MCP + A2A = 你的 Agent 的完整通信栈

MCP 是纵向的：Agent 向下调用工具（数据库、API、文件系统）。
A2A 是横向的：Agent 平级调用另一个 Agent。

一个生产 Agent 系统的全貌是这样的：

```
           ┌─────────────────────┐
           │   你的编排 Agent    │
           └──┬──────────────┬───┘
              │              │
       MCP 调工具        A2A 调别人
              │              │
     ┌────────┴──────┐  ┌──┴──────────┐
     │ 数据库 │ API  │  │ 合规Agent  │
     │ (MCP)  │(MCP) │  │ (A2A 对等) │
     └────────┴──────┘  └─────────────┘
```

你的 Agent 做的事情就是：用 MCP 读数据、写文件、调 API；用 A2A 叫别人帮忙。这一对协议覆盖了 Agent 与外界交互的全部场景。MCP = 伸手拿工具，A2A = 转头喊同事。

## 什么时候绝对不要用 A2A

A2A 的规范是异步任务模型。每次调用需要：发现（GET Agent Card） → 提交（POST task） → 轮询（GET task status） → 获取产物。这至少是 4 次 HTTP 往返。

如果你的两个 Agent 在同一个进程里——A2A 就是在用飞机送外卖。你应该直接在 Python 里调函数。

如果你的调用是微秒级或毫秒级——"帮我检查这个字段是不是空"——A2A 的轮询开销比任务本身还大。用函数调用或直接 RPC。

如果你的团队只有 5 个人，3 个 Agent，永远不需要跟外部 Agent 通信——你不需要 A2A。但如果你是在长线的，至少把接口设计成"如果将来要接 A2A，只要加一个 adapter"。

## 2026 年的生态——这不是学术论文，这是已经在跑的生产系统

150+ 组织支持不是宣传数字，是你可以查的。Google Cloud 把 A2A 集成到了 Vertex AI Agent Builder。Microsoft Agent Framework 支持它。LangGraph、CrewAI、AutoGen 都有 A2A adapter。

这意味着什么？意味着你不需要"选 A2A 还是选别的"——你的框架很可能已经支持 A2A 了。你只需要决定"我的 Agent 要不要暴露出去给别人用"。如果要，写一张 Agent Card。不要，不用。

ACP、ANP、NLIP 这些兄弟协议呢？ACP 是 IBM 推的，范围更窄，更像是 A2A 的前身。ANP 强调去中心化发现（没有 `/.well-known/` 这种中心化概念）。NLIP 是 Ecma 的自然语言内容格式标准化。到 2026 年 4 月，A2A 的采用量是最大的——不是因为技术更好，是因为 Google 的推动力和企业级的背书。

## 最后一个警告

A2A 让你的 Agent 可以被外部调用。外部调用 = 安全边界。你的 Agent Card 里写的 `"skills": ["refund-money"]` 如果被任意调用，你就是在给全互联网开退款窗口。

**A2A 的认证不是可选的。** Bearer token、mTLS、HMAC 签名——选一个，必须选。Agent Card 里的 `auth.type` 字段不只是"建议"，是你的 Agent 的第一道防线。

还有——你的 Agent Card 本身就是攻击面。如果有人篡改了你暴露的 `agent.json`，调用方可能被重定向到恶意服务器。给你的 `/.well-known/agent.json` 加 TLS、加完整性校验。是的，协议本身没有规定这个——但你的生产环境必须有。

## 最后一句

A2A 是 Agent 世界的 HTTP：它让"你的 Agent 调用我的 Agent"从"你跟我开个会定协议"变成"你 GET 我的卡片然后 POST 个任务"。不学它，你的 Agent 永远只能自言自语；学了它，你的 Agent 才能跟整个生态对话。

> 朋友圈金句：MCP 让你的 Agent 伸手拿工具，A2A 让你的 Agent 转头喊同事。工具 + 同事，就是 Agent 的全部社交圈。
