# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> MCP 之前的每一个集成都是一次性的。Model Context Protocol（模型上下文协议）最初由 Anthropic 于 2024 年 11 月发布，现由 Linux 基金会下的 Agentic AI Foundation 管理。它标准化了发现和调用，使任何客户端都能与任何服务器通信。2025-11-25 版本规范命名了六个原语（三个服务器端、三个客户端）、一个三阶段生命周期和 JSON-RPC 2.0 线格式。掌握了这些，本阶段 MCP 章节的其余部分就变成了纯阅读。

**类型：** 学习 **语言：** Python（标准库、JSON-RPC 解析器） **前置课程：** Phase 13 · 01 到 05（工具接口和函数调用） **用时：** 约 45 分钟

## 学习目标

- 说出所有六个 MCP 原语（服务器端的 tools、resources、prompts；客户端的 roots、sampling、elicitation），并各给出一个使用场景。
- 遍历三阶段生命周期（initialize、operation、shutdown），并说出每个阶段谁发送什么消息。
- 解析和发送 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它什么会出问题。

## 问题

在 MCP 之前，每个使用工具的代理都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 发布时自带另一个。VS Code 的 Copilot 扩展又有第三个。一个构建"Postgres 查询"工具的团队，将同一个工具写了三遍，每次针对不同的宿主 API。重用它需要复制代码。

结果是集成的一次性爆发式增长和生态系统速度的瓶颈。

MCP 通过标准化线格式解决了这个问题。一个 MCP 服务器可以在每个 MCP 客户端中运行：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已达 300+ 客户端。月 SDK 下载量 1.1 亿次。10,000+ 个公开服务器。Linux 基金会于 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了管理。

本阶段使用的规范版本是 **2025-11-25**。它添加了异步任务（SEP-1686）、URL 模式引导（SEP-1036）、带工具采样（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示符语义。Phase 13 · 09 到 16 涵盖这些扩展。本课程止于基础。

## 核心概念

### 三个服务器端原语

1. **Tools（工具）。** 可调用的操作。与 Phase 13 · 01 中相同的四步循环。
2. **Resources（资源）。** 暴露的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义方案。
3. **Prompts（提示模板）。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots（根目录）。** 服务器允许接触的 URI 集合。客户端声明它们；服务器尊重它们。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。使服务器托管的代理循环无需服务端 API 密钥。
6. **Elicitation（引导）。** 服务器在运行中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力恰好属于这六个之一。Phase 13 · 10 到 14 深入涵盖每个。

### 线格式：JSON-RPC 2.0

每条消息是一个包含以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期望响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize（初始化）。**

客户端发送 `initialize` 及其 `capabilities` 和 `clientInfo`。服务器响应其自身的 `capabilities`、`serverInfo` 及其所遵循的规范版本。客户端在消化响应后发送 `notifications/initialized`。从此以后，任何一方都可以根据协商的能力发送请求。

**阶段 2：operation（运行）。**

双向的。客户端调用 `tools/list` 进行发现，然后调用 `tools/call` 进行调用。如果服务器声明了该能力，它可以发送 `sampling/createMessage`。当工具集发生变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根目录范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown（关闭）。**

任一方关闭传输。MCP 中没有结构化的关闭方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是合约。服务器端示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端不声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器不声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的机制。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用那个功能。

### 结构化内容和错误形态

`tools/call` 返回一个类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 添加了 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误码。规范定义的补充：`-32002` "Resource not found"、`-32603` "Internal error"，以及 MCP 特定的错误数据 `error.data`。

### 客户端能力 vs 工具调用细节

常见混淆：`capabilities.tools` 是客户端是否支持工具列表变更通知。客户端是否会调用特定工具是由其模型驱动的运行时选择，而不是能力标志。能力标志是规范级别的合约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），因此具有对称请求/响应形态的 JSON-RPC 是自然的选择。JSON-RPC 也能在 stdio 和 WebSocket/Streamable HTTP 上干净地组合，无需重新发明 HTTP 的请求形态。

## 动手实践

`code/main.py` 发布了一个最小的 JSON-RPC 2.0 解析器和发送器，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实的传输；只是消息形态。对照延伸阅读中的规范来验证每个信封。

查看要点：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块的数组。

## 交付成果

本课程产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个类 pcap 的 MCP 客户端-服务器交互转录，该技能为每条消息注释它属于哪个原语、哪个生命周期阶段以及依赖哪个能力。

## 练习

1. 运行 `code/main.py`。识别能力协商发生的那一行，并描述如果服务器不声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形态：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档约 80 页。识别大多数服务器不需要的一个能力标志。提示：与资源订阅有关。

4. 在纸上勾勒一个假设的"定时任务"功能属于哪个原语。（提示：服务器希望客户端在预定时间调用它。今天六个原语都不适用。）MCP 2026 路线图有一个草案 SEP。

5. 解析 GitHub 上一个开放 MCP 服务器的一个会话日志。统计请求 vs 响应 vs 通知消息的数量。计算流量中生命周期 vs 运行的比例。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| MCP | "模型上下文协议" | 用于模型到工具发现和调用的开放协议 |
| 服务器原语 | "服务器暴露什么" | tools（操作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器使用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一条消息对；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集 |
| `tools/call` | "调用" | 客户端要求服务器以参数执行一个工具 |
| `notifications/*_changed` | "变更事件" | 服务器告知客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名草案提案（如 SEP-1686 用于异步任务） |

## 延伸阅读

- [Model Context Protocol — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六个原语的心智模型
- [Anthropic — 介绍 Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP 博客 — 首个 MCP 周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一年回顾和 2025-11-25 规范变化
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的总结

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一份极好的 MCP 概念入门文档。它在 45 分钟的课时内完成了从零到理解 MCP 六个原语、三阶段生命周期和 JSON-RPC 线格式的全过程，信息密度高但不臃肿。文档选择在"2025-11-25"这个重要的规范版本节点上讲解，既覆盖了基础概念，又呼应了最新的扩展（SEP-1686 异步任务等），展现了良好的时效性。

### 二、知识结构梳理

**认知基础层：** MCP 的三大动机（消除一次性的集成、统一发现和调用、构建生态系统）。六个原语的定义和归属（服务器端三个：tools/resources/prompts；客户端三个：roots/sampling/elicitation）。这个六分区是 MCP 能力的"元素周期表"。

**工程模式层：** JSON-RPC 2.0 线格式（请求/响应/通知三形态）、三阶段生命周期（initialize → operation → shutdown）、能力协商机制（对称的能力声明）。这些构成了 MCP 的"协议语法"。

**落地实践层：** `code/main.py` 的关节行走——从 `initialize` 到 `tools/call` 到 `shutdown` 的完整消息序列，让学生从协议层面理解一个 MCP 客户端和服务器如何通信。结构化内容块（text/image/resource）和错误码体系。

### 三、核心洞察

1. **六个原语是 MCP 能力的"元素周期表"。** 为什么重要：MCP 规范中新增的一切功能（异步任务、URL 引导、资源订阅等）都必须归属到这六个原语之一。理解这六个原语就掌握了 MCP 的分类法，后续学习新 SEP 时只需要问"这属于哪个原语"。

2. **能力协商机制是 MCP 防止生态系统漂移的核心设计。** 为什么重要：不同于 HTTP 的"请求即期望"模型，MCP 通过能力协商让"不支持某功能"成为一等公民——不是"出错"而是"我们不用这个功能"。这个设计使得新功能可以渐进采用，而不会破坏不支持的客户端。

3. **JSON-RPC 2.0 作为线格式是深思熟虑的架构决策。** 为什么重要：MCP 需要服务器发起的消息（采样、通知），REST 的单向性不适合。JSON-RPC 的对称性支持双向通信，且能在 stdio 和 HTTP 上以相同语义运行。理解这个决策有助于理解为什么 MCP 不是 REST API。

4. **"能力标志"和"运行时选择"是两个正交概念。** 为什么重要：文档特别澄清了 `capabilities.tools` 不是"客户端是否会调用特定工具"，而是"客户端是否支持工具列表变更通知"。混淆这两个概念是 MCP 初学者的常见错误，会导致架构设计上的误判。

5. **生命周期第三阶段（shutdown）没有结构化方法。** 为什么重要：MCP 的关闭依赖于传输层的信号（stdio EOF 或 HTTP 连接关闭），这意味着没有"优雅关闭"握手。这对于有状态服务器（如数据库连接）是一个需要额外注意的细节。

6. **2025-11-25 版本规范是一个重要的分水岭。** 为什么重要：之前的规范缺少异步任务、URL 引导、OAuth 2.1 等生产必需的特性。理解这个版本规范的变化，有助于判断一个 MCP 实现是否能满足生产需求。

### 四、教学建议

1. **用"协议考古学"方式引入 MCP。** 先讲 MCP 之前世界的混乱（每个客户端有自己的协议、工具无法复用），再讲 MCP 解决的三个问题（标准化发现、统一调用、构建生态），最后展示截至 2026 年 4 月的 300+ 客户端、1.1 亿月下载量数据。故事驱动比概念驱动更有感染力。

2. **用物理卡片教六个原语。** 做六张卡片（tools/resources/prompts/roots/sampling/elicitation），让学生将具体的 MCP 功能（如 `tools/list_changed`、采样请求、斜杠命令）归类到正确的卡片下。这种触觉式教学比纯阅读更深刻。

3. **让学生手写 JSON-RPC 信封。** 在纸或白板上写出 `initialize` → `initialized` → `tools/list` → `tools/call` 的完整消息序列。手写比读代码更容易内化线格式的形态。

4. **能力协商作为核心教学重点。** 花最多时间在能力协商上——这不仅是规范细节，更是 MCP 的设计哲学。用"两个乐队协商演奏曲目"的比喻：每个乐队声明自己会什么，然后就共同支持的曲目合作。

5. **与 REST API 对比教学。** 让学生尝试用 REST 设计 MCP 的采样功能（服务器向客户端发请求），他们会发现 REST 的根本限制。然后引入 JSON-RPC 2.0 作为解决方案。这种"先碰壁再给方案"的教学法记忆更深。

6. **提前预告后续课程。** 本课程是 MCP 章节的入口。在教学末尾明确列出后续课程的路径：Phase 13 · 09（Streamable HTTP 传输）、Phase 13 · 10-14（各原语深度）、Phase 13 · 15（工具投毒防御），帮助学生建立学习路线图。

### 五、值得补充的内容

1. **MCP 与 A2A 协议的关系。** Google 的 Agent-to-Agent 协议和 MCP 各自解决什么问题？它们如何互补？补充这个对比有助于学生定位 MCP 在整个 AI 工程生态中的位置。

2. **MCP 服务器的发现和分发机制。** 文档提到 10,000+ 公开服务器，但如何在客户端中发现和安装它们？补充对 `mcp.json` 配置文件、npm 包分发、Docker 容器化部署等实践模式的讨论。

3. **JSON-RPC 2.0 的能力边界。** JSON-RPC 2.0 不支持原生的流式响应和双向流。MCP 如何在这些限制之上构建通知和进度？补充对 JSON-RPC 局限性及 MCP 绕过策略的讨论。

4. **MCP 规范的治理和演进流程。** SEP 提案如何从草案变为规范？Linux 基金会下的治理结构是什么样的？补充这些内容帮助学生理解规范的稳定性和未来方向。

### 六、一句话总结

MCP 用六个原语、一次握手和一种线格式，将 AI 工具生态从"每个客户端一个协议"的混乱状态统一为"一次编写、处处运行"的标准化架构。



---

# 🎓 Agent 架构课：MCP 基础——"每个客户端都在造自己的语言，直到有人说'咱们用同一种'"

同学们好。我是你们的FDE工程老师，今天讲的是MCP 协议。

今天这节课，我想先问你们一个问题：**如果你的工具系统只能被一个 IDE 加载，它叫"集成"。如果你的工具系统能被 300+ 个客户端加载，它叫"生态"。MCP 做的事，就是把前者变成后者。**

2024 年之前，Agent 工具的世界是巴别塔。Cursor 有一套工具声明格式。Claude Desktop 有另一套。VS Code Copilot 有第三套。你写了一个 Postgres 查询工具——查询逻辑 200 行——为了对接三个宿主，包装代码写了 800 行。每次宿主更新 API，你就得跟着改一遍。这不是写代码，是给三个不同的国王当翻译。

MCP 做的事说起来不复杂，但做成了以后改变了整个 Agent 生态的格局：**它说了一种通用语。** 2025 年 12 月被 Linux 基金会接管，到 2026 年 4 月——10,000+ 个公开服务器，300+ 个客户端，月 SDK 下载量 1.1 亿次。

这节课，我带你拆开这个通用语，看看里面到底装了什么。

## 六个原语：MCP 的全部词汇量

MCP 的词汇量很小——就六个词。三个是服务器说的，三个是客户端说的。

**服务器端的三个原语：**

1. **Tools（工具）——"我能帮你做这件事。"** 可调用的操作。你熟悉的函数调用循环——模型说"查天气"，服务器执行，返回结果。这是 MCP 里用最多的原语。

2. **Resources（资源）——"我有这些数据，你可以读。"** 通过 URI 寻址的只读内容。`file:///home/user/docs`、`db://query/users`。跟 Tools 的区别：Tools 是做操作（写文件），Resources 是读数据（读文件内容）。

3. **Prompts（提示模板）——"这是常用的对话模板。"** 可复用的斜杠命令。在 Claude Desktop 里你打 `/review`，服务器提供一个预制的代码审查提示词模板，用户填参数就行。

**客户端的三个原语：**

4. **Roots（根目录）——"这些是我让你看的目录。"** 客户端告诉服务器"你的文件系统访问范围是 `~/projects`"。服务器不该看到范围外的东西。

5. **Sampling（采样）——"帮我跑一下 LLM。"** 这是 MCP 最革命的设计：服务器可以反过来让客户端调 LLM。一个"不需要自己持有 API Key"的服务器，通过采样借用客户端的模型。

6. **Elicitation（引导）——"我需要用户确认这个操作。"** 服务器在做一个可能有后果的操作前，通过客户端向用户请求许可。

六个原语。一个 Agent 写了 200 行的 Postgres 查询工具，只需要实现 Tools 原语。一个文件管理器，需要 Tools + Resources。一个代码审查助手，可能需要全部六个。

词汇量小不是缺陷——是设计的精髓。协议越小，实现越可靠，生态越容易长。

## 三次握手：initialize → 正常工作 → shutdown

任何 MCP 连接的寿命分为三个阶段。这是面试必考题。

**第一阶段：Initialize。** 客户端说"你好，我是 Claude Desktop v2.1，我支持 tools、resources、sampling"。服务器回"你好，我是 Postgres MCP Server v1.0，我提供 tools 和 resources，我不支持 sampling"。

这个过程叫**能力协商**。双方交换自己能做什么、不能做什么，然后只在交集里工作。关键设计："不支持某功能"不是出错——是"我们不用这个功能"。这让新功能可以渐进采用，老客户端不会被新服务器的功能炸掉。

```
客户端 → 服务器: initialize {"capabilities": {"tools": {}, "sampling": {}}}
服务器 → 客户端: initialize_result {"capabilities": {"tools": {}, "resources": {}}}
客户端 → 服务器: initialized {}  # 通知：我准备好了
```

**第二阶段：Operation。** 正常工作。`tools/list` → `tools/call` → 结果返回。通知流：`tools/list_changed`、`resources/updated`。这是连接寿命里最长的阶段。

**第三阶段：Shutdown。** 没有正式的关闭原语。Stdio 传输里，客户端关掉子进程的 stdin，等 stdout EOF。Streamable HTTP 里，客户端删掉会话。

## JSON-RPC 2.0：为什么不是 REST

你可能会问：为什么 MCP 不用 REST？GET /tools、POST /tools/call——多直观。

答案是：**REST 是单向的。MCP 需要双向。**

服务器需要主动给客户端发消息——`tools/list_changed` 通知、`sampling/createMessage` 请求。REST 里服务器不能主动联系客户端（除非用 WebSocket，但那又是另一套复杂度）。JSON-RPC 2.0 是天然对称的：双方都可以是请求方，双方都可以发通知。

而且 JSON-RPC 在 stdio 上和 HTTP 上能用相同的语义运行。一个 MCP 服务器可以用 stdio 给 Claude Desktop 用，也可以用 Streamable HTTP 给云端 Agent 用——同一套代码，同一个消息格式。

```json
// 一个 JSON-RPC 请求
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}

// 一个 JSON-RPC 响应
{"jsonrpc": "2.0", "id": 1, "result": {...}}

// 一个 JSON-RPC 通知（没有 id，不要回复）
{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
```

三条规则：有 `id` = 期待响应。没有 `id` = 通知，别回复。`id` 是字符串或数字。就这么简单。

## 能力协商：为什么它不是"可选的优雅设计"

我见过一个 MCP 服务器，在 `initialize` 时声明了自己支持 `sampling`，但实际代码里 sampling 的处理器是个空函数。客户端信了，发了 `sampling/createMessage`——服务器收了消息，然后什么都没做。客户端永远等不到响应。

能力协商不是装饰——是你对客户端的契约。你声明了什么，就必须能处理什么。反过来，客户端声明了 `roots` 能力但没提供根目录列表——服务器请求 `roots/list` 时拿到空数组，该报错还是该用默认值？这是设计决策，但你必须事先决定。

**一个原则：能力声明 = 你的 API 契约。多声明了叫欺诈，少声明了叫自限。**

## 结课清单

如果你今天只带走一件事：**MCP 的六个原语和三阶段生命周期不是"protocol trivia"——是你写出能被 300+ 客户端加载的工具系统所需要知道的全部"语法"。词汇量很小，但每个词都是一类交互模式。**

完整清单：

1. **六个原语 = MCP 的全部词汇。** 三个服务器端（Tools/Resources/Prompts），三个客户端（Roots/Sampling/Elicitation）。你的服务器只需要实现它需要的。
2. **三阶段生命周期 = 所有 MCP 连接的形状。** Initialize（能力协商）→ Operation（正常工作）→ Shutdown（优雅关闭）。
3. **JSON-RPC 2.0 是理性选择，不是审美偏好。** 双向性 + 跨传输一致性 = REST 做不到的事。
4. **能力协商是契约，不是可选的。** 声明了就要实现。
5. **通知没有 `id`，不要回复。** 这是 JSON-RPC 里最容易犯的错——给通知发响应。
6. **MCP 不是你项目的必选项。** 1-2 个工具 → 手写 API 调用。构建工具生态 → 别再造轮子。

最后一句话——我希望你明天还记得：

**MCP 不是银弹。它是 AI Agent 生态从方言时代走向普通话时代的第一个通用语提案。10,000 个服务器已经在说这种语言了——你的客户端要不要加入，取决于你想跟多少人对话。**

---

# 💼 从业者故事：MCP 基础——"每个客户端都在造自己的语言，直到有人说'咱们用同一种'"

2024 年秋天，我在给一个金融数据平台做 Agent 集成。客户要求"接入 Cursor""接入 Claude Desktop""接入 VS Code Copilot"。我们那个 Postgres 查询工具——查询逻辑就 200 行——为了对接三个宿主，包装代码写了 800 行。三个宿主，三套工具声明格式，三种调用回调。每次宿主更新 API，我们就得跟着改一遍。

有一天我盯着那堆 adapter 代码想了十分钟：这不就是同一个功能穿了三件不同的外套吗？

然后 MCP 出来了。Anthropic 在 2024 年 11 月发布了 Model Context Protocol，说的就是我们每天在骂的事："别他妈再为每个 AI 宿主重写工具了，用一种语言说话。"到 2026 年 4 月，月下载 1.1 亿次，10,000+ 公开服务器，300+ 客户端。一年半，从概念到生态。

## MCP 不是又一个 API 框架，它是个翻译标准

如果你用一句话解释 MCP，不是"客户端和服务器通信的协议"——这在技术上是废话。真正的定义是：**MCP 让 AI Agent 和外部世界之间的对话标准化，就像 USB 让电脑和外设之间的对话标准化。你用 USB 键盘不需要知道电脑的主板型号，你用 MCP 工具不需要知道宿主是 Claude 还是 Cursor。**

但你得先理解一个问题：为什么是 JSON-RPC 2.0，不是 REST？

这个问题我面试过很多人，能答对的不到一半。答"JSON-RPC 简单"的人不算对——那只是表面。真正的原因是：**MCP 需要双向通信，而 REST 是单向的。** 在 MCP 里，服务器可以向客户端发请求——比如采样（sampling），服务器说"你帮我用 LLM 处理一下这段文本"。在 REST 里服务器不能主动向客户端发请求，除非你用 WebSocket 然后重新发明请求-响应语义——那你就是在用 REST 重新实现 JSON-RPC 了。

JSON-RPC 2.0 是个老协议——2010 年的东西——但它干净、对称、能在 stdio 和 HTTP 上都有一样的语义。一条消息三个字段：`jsonrpc: "2.0"`、`method`、`params`，有 `id` 就是请求（对方必须回），没 `id` 就是通知（对方不回）。这种简单的对称性让 MCP 的重心能放在"协议上要说什么"而不是"怎么传"。

## 六个原语：MCP 的"元素周期表"

MCP 有六个原语——三个服务器端（tools、resources、prompts），三个客户端（roots、sampling、elicitation）。我打赌你第一次听到这六个名字时想的是"哦，六种功能"。但这不是"六种功能"，这是 MCP 能力的"元素周期表"——所有未来的扩展都必须归属到这六个原语之一。

**Tools（工具）** 就是"服务器能做事情"。给模型说"我这有个 get_weather，你调用它"。模型调用，服务器执行，返回结果。这跟 Phase 13 · 01 讲的四步循环是同一件事，只是现在用 JSON-RPC 协议化了。

**Resources（资源）** 是"服务器有数据"。比如 `file:///home/projects/config.json`。Data 就放在那，客户端想读就通过 `resources/read` 请求——只读，不改。

**Prompts（提示模板）** 是"服务器帮你起头说一句话"。比如 `/review_note {note_id}` 是一个斜杠命令，客户端选中它，服务器说"以下是关于这条笔记的审阅草稿..."。

这三个服务器原语的区别用一个比喻来说：**Tools 是你厨房里的锅——你能拿它炒菜。Resources 是你冰箱里的食材——你能打开看，但你不能拿冰箱炒菜。Prompts 是你手机里的菜谱——它告诉你第一步放什么，你照着做。**

三个客户端原语是给服务器用的——这就是 MCP 双向性的体现：

**Roots（根目录）** 是客户端说"这些文件夹你可以碰，那些不行"。服务器不能越界。

**Sampling（采样）** 是服务器说"我也不带脑子，你帮我用 LLM 想想"。服务器把一段文本传给客户端，客户端用自己的模型跑一遍，结果再传回去。这就是为什么 MCP 服务器不需要自己持有 API Key。

**Elicitation（引导）** 是服务器在对人类说"你还没告诉我这个东西是什么，帮我填一下"。比如服务器在处理一个文件，发现有个字段它不认识，就弹一个表单让用户填。

## 能力协商：MCP 最被低估的设计

如果你想用一个词概括 MCP 的设计哲学，不是"标准化"，是"协商"。

HTTP 的世界里，你发一个请求，服务器要么成功、要么失败。如果服务器没有你请求的功能，你就得到一个 404 或 501。MCP 不一样——它在握手阶段就先把"我会什么"和"你会什么"摊在桌子上。

这个过程叫 `initialize`。客户端发一条消息："我叫 Claude Desktop，版本 2.1，我能做 sampling 和 roots 管理。"服务器回："我叫 Postgres MCP，版本 1.3，我能做 tools 和 resources，有 listChanged 通知，但不支持 subscribe。"握手完成。从此以后，双方只使用共同支持的功能。

这就像两个人第一次见面互换个名片——不是 "你能不能做X" "不能" "好的那算了"，而是上来就互相展示能力清单，只跟清单交集的条目合作。**这个设计的精妙之处在于：新功能可以随时加进 MCP 规范，而不需要所有客户端和服务器同步升级。如果你不支持某个新功能，你的能力清单里没有它，对方就不会用。不兼容不是 bug，是预期行为。**

我在生产里见过最蠢的 MCP bug：客户端声明了 `sampling` 能力，但实际上没实现 `sampling/createMessage` 的处理逻辑。服务器收到能力声明后放心地发了一个采样请求，然后请求进了虚空——客户端收到消息后找不到处理器，也没有返回错误。服务器等了 30 秒超时。这就是**能力协商的最大陷阱：声明的能力必须是真的实现了的能力。别往名片上写你不会的。**

## 生命周期：从握手到沉默

MCP 的三阶段生命周期（initialize → operation → shutdown）看起来跟你见过的所有协议一样，但有两件事不一样。

第一，`initialize` 不是单方面的——是双向的。双方都声明能力，双方都消化对方的能力后才开始正常通信。很多人以为只需要服务端声明，客户端不用——错了，客户端的 `roots`、`sampling`、`elicitation` 能力声明同样重要，服务器靠这些决定能不能用采样。

第二，shutdown 没有结构化的握手。你找不到 `shutdown` 方法。MCP 的关闭是通过传输层信号——stdio 模式下是进程 EOF，HTTP 模式下是连接关闭。这意味着如果你的服务器持有状态（比如数据库连接），你需要在收到 EOF 时自己做清理。没有"收到 shutdown 请求"这个事件。这个设计理由很简单：传输层负责可靠性，协议层不重复造轮子。但我还是希望未来版本能加一个可选的 `graceful_shutdown` 通知。

## JSON-RPC 的那些坑

JSON-RPC 2.0 的简洁性同时也意味着它缺少很多东西——你得自己在协议之上补。

没有原生的流式响应。MCP 的 `notifications/progress` 是靠"在请求处理期间不断发通知"来实现进度条的，但这在 spec 里只是个约定，没有强制的帧格式。如果你在写一个 MCP 客户端，不要假设所有的服务器都会发 `notifications/progress`——很多轻量服务器根本不实现。

没有原生的请求取消语义。JSON-RPC 没有 `cancel` 方法。如果你发了一个 `tools/call` 然后用户点了取消，你怎么办？MCP 的答案是——不办。大部分实现的做法是客户端关闭当前传输并重新连接，等于杀进程再起。

**这些不是 MCP 的错误，是 JSON-RPC 2.0 的先天限制。选择 JSON-RPC 是为了简单性和传输无关性，代价就是得在这些边缘 case 上自己想办法。如果你在构建一个生产系统的 MCP 层，这些缺口是你必须补的——不是协议的错，但你要知道在哪补。**

## 生产现实

- **MCP 消息的平均大小**：`tools/list` 响应对于 30 个工具约 15-30KB。`tools/call` 请求约 1-3KB。如果你在 stdio 上跑，每行一个 JSON，JSON 序列化/反序列化的 CPU 开销在小服务器上可以忽略，但在 500 并发调用的场景下会成为瓶颈。
- **能力协商的开销**：`initialize` 往返通常 10-50ms（本机 stdio）。这比你想象的低，因为 stdio 走本地管道，没有网络延迟。
- **工具列表缓存**：规范没有强制缓存策略。`tools/list_changed` 通知是可选的（需要服务器声明能力）。实际生产中，大部分 MCP 服务器不发出变更通知，客户端启动时调一次 `tools/list` 然后缓存到生命周期结束。

## 收尾

MCP 不是银弹，它是 AI Agent 生态从方言时代走向普通话时代的第一个通用语提案。它的六个原语、能力协商和 JSON-RPC 线格式，构成了一个足够简单又足够强大的协议骨架。

**如果你现在正在设计一个 Agent 系统，要不要引入 MCP？我的答案是：如果你的 Agent 只用一两个工具，手写 API 调用就够了。但如果你在构建一个能被多种宿主加载的工具生态——那就别再造轮子了。已经有 10,000 个 MCP 服务器在等着你的客户端接入。**
