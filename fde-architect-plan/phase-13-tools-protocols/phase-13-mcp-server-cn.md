# 构建 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。一个真正的服务器暴露工具、资源和提示模板，处理能力协商，发出结构化错误，并且在各 SDK 间保持一致。本课程从头到尾构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三个服务器端原语，以及一个纯函数风格，既能直接嵌入 Python SDK 的 FastMCP，也能在升级时迁移到 TypeScript SDK。

**类型：** 构建 **语言：** Python（标准库、stdio MCP 服务器） **前置课程：** Phase 13 · 06（MCP 基础） **用时：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 按 JSON-RPC 2.0 规范和 MCP 附加错误码发出结构化错误响应。
- 将标准库实现升级到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题

在你使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器作为子进程由客户端启动，消息通过 stdin/stdout 以换行分隔的方式流动。

2025-11-25 规范规定，stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里没有 SSE（Server-Sent Events）；SSE 是旧的远程模式，正在 2026 年中被移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用它；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是整个线格式。

笔记服务器是一个很好的示例，因为它练习了所有三个服务器端原语。Tools 做变更（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课程的形态可以推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个请求必须匹配一个带有相同 `id` 的响应。
- 通知不得被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

仅声明你支持的功能。客户端依赖能力集来门控功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的：

```json
{"type": "text", "text": "找到 2 条笔记"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形态。协议级错误（未知方法、错误参数）是 JSON-RPC 错误。工具级错误（有效调用但工具失败）返回为 `{content: [...], isError: true}`。这让模型在其上下文中看到失败。

### 实现资源

资源设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据暴露为资源而非工具时：

- 模型不会"调用"它；客户端可以在用户请求时将其注入上下文。
- 订阅让服务器在资源变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 通过 `ui://` 扩展了交互式资源。

### 实现提示模板

提示模板是带有命名参数的模板。宿主将它们呈现为斜杠命令。`review_note` 提示可能接受 `note_id` 参数，并产生一个多消息提示模板，客户端将其提供给其模型。

### Stdio 传输的细微之处

- 换行分隔的 JSON。没有长度前缀帧。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，安全重试。
- `destructiveHint: true` — 不可逆的副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库服务器约 180 行。FastMCP（Python）将同样的逻辑简化为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等效的形态。升级路径在准备好后即可直接迁移；核心概念（能力、分发、内容块）保持不变。

## 动手实践

`code/main.py` 是一个完整的笔记 MCP 服务器，通过 stdio 运行，仅使用标准库。它处理 `initialize`、`tools/list`、`tools/call`（三个工具：`notes_list`、`notes_search`、`notes_create`）、`resources/list` 和 `resources/read`（针对每条笔记），以及一个 `review_note` 提示模板。你可以通过管道发送 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

查看要点：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个内容块列表，而不是裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## 交付成果

本课程产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能使用正确的 tools / resources / prompts 划分和 SDK 升级路径来搭建一个 MCP 服务器。

## 练习

1. 运行 `code/main.py`，用手工构建的 JSON-RPC 消息驱动它。练习 `notes_create`，然后练习 `resources/read` 来检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端是否会弹出确认对话框（这需要一个真实的宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以内。线行为必须完全相同；用相同的 JSON-RPC 测试脚手架验证。

5. 阅读规范的 `server/tools` 部分，识别一个本课程服务器未实现的工具定义字段。（提示：有几个；选一个并添加它。）

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| MCP 服务器 | "暴露工具的那个东西" | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端作为子进程启动；通过 stdin/stdout 通信 |
| 分发器 | "方法路由器" | JSON-RPC 方法名到处理器函数的映射 |
| 内容块 | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 标记工具失败；与 JSON-RPC 错误区分 |
| 注解 | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议上层高阶框架 |
| 资源 URI | "可寻址数据" | 标识资源的 `file://`、`db://` 或自定义方案 |
| 提示模板 | "斜杠命令简介" | 服务器提供的带参数槽的模板，供宿主 UI 使用 |
| 能力声明 | "功能开关" | 在 `initialize` 中声明的按原语标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行的 TypeScript 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — 用于 MCP 服务器的装饰器风格 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一份实践性极强的 MCP 服务器构建指南。文档避开了"只教 hello-world"的陷阱，而是构建一个完整的笔记服务器（覆盖 tools、resources、prompts 三个原语），并展示了从标准库实现到 FastMCP 的清晰升级路径。180 行标准库代码是理解 MCP 服务器的"纯净视角"——没有框架魔法，每一行都有教育意义。

### 二、知识结构梳理

**认知基础层：** MCP 服务器作为子进程的模型（客户端启动、stdio 通信、客户端控制生命周期）。三个服务器端原语在实践中的分工——tools 做变更、resources 暴露数据、prompts 提供模板。

**工程模式层：** 分发循环模式（读行 → 解析 JSON → 判断请求/通知 → 路由到处理器 → 写响应）。纯函数工具执行器 + 内容块数组的输出约定。`isError: true` 与 JSON-RPC 错误的二分法。

**落地实践层：** Stdio 传输的 4 个细微陷阱（换行分隔、强制刷新、EOF 处理、SIGPIPE）。注解系统作为安全提示。从标准库到 FastMCP 的无缝升级路径——概念不变，代码量缩减到 80 行。

### 三、核心洞察

1. **Stdio 传输是最简单也最容易踩坑的传输方式。** 为什么重要：`sys.stdout.flush()`、不向 stdout 打印日志、正确处理 EOF——这些细节在教程中经常被忽略，但在实际部署中是最大的失败来源。一个忘记 `flush()` 的服务器会无限期地阻塞客户端。

2. **`isError: true` vs JSON-RPC 错误的二分法是精心设计。** 为什么重要：协议错误（未知方法）和工具错误（有效调用但失败）有不同的受众——前者给开发者，后者给模型。`isError: true` 让模型在其上下文中看到失败信息，从而触发自我修正。

3. **注解系统是服务器"自描述"的关键。** 为什么重要：`readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint` 四个注解让客户端可以在不"理解"工具内容的情况下做出 UX 决策（如确认对话框）。这是 MCP 协议解耦原则的优雅体现。

4. **纯函数工具执行器是保持 SDK 独立的关键。** 为什么重要：文档强调"纯函数风格，既能嵌入 FastMCP 也能迁移到 TypeScript SDK"。工具逻辑与协议处理分离，使升级路径真正无缝——你换的是框架，不是业务逻辑。

5. **分发器是一个简单的 `dict[str, Callable]`。** 为什么重要：这个简单实现揭示了 MCP 服务器的本质——一个方法名到函数的映射。理解了这个，FastMCP 的装饰器就不再是"魔法"，而是这个基础模式的语法糖。

### 四、教学建议

1. **从运行开始，再解释代码。** 先让学生用管道驱动服务器（`echo '...' | python main.py`），看到真实的 JSON 往返，然后再讲解代码结构。可视化的协议交互比静态代码更有说服力。

2. **让学生"加一个工具"作为核心练习。** 在笔记服务器的骨架代码上，让学生自己添加 `notes_delete` 工具。这个过程迫使它们理解分发器、内容块、isError 等所有核心概念。

3. **故意触发常见错误，让教学更有记忆点。** 忘记 `flush()`——客户端挂起。向 stdout 打印日志——客户端解析失败。不处理 EOF——孤儿进程。让学生"踩坑"后再给解决方案，比预防性讲解更有效。

4. **将注解系统作为"安全设计"的教学案例。** 用 `destructiveHint: true` 的例子让学生讨论"为什么服务器要告诉客户端这个工具是破坏性的"。引导学生理解：好的协议设计让不可见的安全隐患变得可见。

5. **展示标准库版和 FastMCP 版的对比。** 让学生先手写标准库版的分发循环（约 180 行），然后展示 FastMCP 的等效实现（约 80 行）。这种对比让学生理解框架的价值（消除样板代码），同时保留了底层协议的理解。

6. **将本课程作为 MCP 章节的"实践高峰"。** 在 Phase 13 · 06（概念）和 Phase 13 · 09（传输）之间，本课程是让学生"动手"的关键节点。建议分配充足的实践时间（至少是讲授时间的两倍）。

### 五、值得补充的内容

1. **错误处理的完整策略。** 文档讲解了 `isError` 和 JSON-RPC 错误码，但缺少对"工具执行超时"、"并发调用冲突"等实际生产问题的讨论。补充这些边界情况的处理模式。

2. **日志和可观测性。** 文档提到"调试日志输出到 stderr"，但没有覆盖结构化日志、请求 ID 追踪、性能指标（工具调用延迟）等可观测性实践。这些对于生产环境至关重要。

3. **测试策略。** 如何为 MCP 服务器编写自动化测试？补充基于 JSON-RPC 管道的端到端测试模式（如用 `subprocess.Popen` 启动服务器，通过管道发送消息，断言响应）。

4. **与传统 HTTP API 服务器的对比。** 文档专注于 MCP 的 stdio 模型。补充与传统 REST API 服务器在架构模式上的对比（如路由、中间件、状态管理），帮助有 Web 后端背景的学生更快理解。

### 六、一句话总结

构建 MCP 服务器就是写一个从 stdin 读 JSON-RPC、在 stdout 写 JSON-RPC 的 while 循环——剩下的 180 行都是该循环的分支内容。


---

# 🎓 Agent 架构课：构建 MCP 服务器——"你的 while 循环里藏着整个 Agent 世界"

同学们好。我是你们的FDE工程老师，今天讲的是 MCP 服务器的构建核心。

今天这节课，我想先纠正一个流传很广的误解。你在网上看到的 MCP 教程通常会告诉你："构建 MCP 服务器就是写一个 while 循环。"这句话技术上是对的——从 stdin 读 JSON-RPC，处理，写回 stdout。但你如果只理解到这一层，你写的服务器会在凌晨三点被 PagerDuty 叫起来。

**真正的 MCP 服务器是一个事件分发器，六个原语是它的事件类型，而每一个事件处理器都不能阻塞另外五个。**

## 一个分发循环的骨架

你服务器的心脏长这样：

```python
import sys, json

dispatch = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "resources/list": handle_resources_list,
    "resources/read": handle_resources_read,
    "prompts/list": handle_prompts_list,
    "prompts/get": handle_prompts_get,
}

for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        handler = dispatch.get(msg["method"])
        result = handler(msg.get("params", {}))
        sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":result}) + "\n")
        sys.stdout.flush()
```

三条铁律：**不要向 stdout 打印任何非 JSON-RPC 的内容**（调试走 stderr）。**每个有 id 的请求必须匹配同 id 的响应。** **通知（无 id）不得被响应。**

## 三个服务器端原语的实现要点

- **Tools**：`tools/list` 返回 `{tools: [{name, description, inputSchema}]}`。`tools/call` 接收 `{name, arguments}` 返回 `{content: [{type:"text", text:"..."}], isError: false}`。关键：`content` 是数组不是字符串——MCP 支持多模态内容块。
- **Resources**：`resources/list` 返回资源列表（每个有 URI），`resources/read` 用 URI 读取。
- **Prompts**：`prompts/list` 列出模板，`prompts/get` 用参数填充。被 Claude Desktop 用来暴露斜杠命令。

## 两个让你被 Oncall 叫醒的坑

**坑一：阻塞式工具处理器。** 你的 `tools/call` 处理器里跑了一个数据库查询，查了 120 秒。这 120 秒内整个 while 循环停转。解决方案：长操作走后台任务，主循环立即返回 `{status: "processing", task_id: "..."}`。

**坑二：忘记 flush。** `sys.stdout.write()` 后没有 `sys.stdout.flush()`——数据留在缓冲区，客户端永远收不到。

## 结课清单

1. **分发循环 = while 循环 + 方法路由表。** 骨架 30 行，剩下的是处理器逻辑。
2. **`content` 是数组不是字符串。** 不要返回裸字符串。
3. **工具处理器不能阻塞主循环。** 长操作走后台任务。
4. **每次 stdout write 后必须 flush。** stdio 传输下最常见的 bug。
5. **调试日志走 stderr，永远不污染 stdout。**

最后一句话——我希望你明天还记得：

**构建 MCP 服务器就是写一个 while 循环。但那个 while 循环每转一圈，你都在跟一个你不知道它接下来会说什么的系统对话。那个系统不是你的用户——是你的用户在用 LLM 跟你的服务器说话。**

---

# 💼 从业者故事：构建 MCP 服务器——"你的 while 循环里藏着整个 Agent 世界"

半夜被 Oncall 叫醒，因为你写的 MCP 服务器把 Claude Desktop 卡死了。PagerDuty 告警说"Agent 无响应超过 5 分钟"。我打开日志一看——服务器卡在了 `tools/call` 的处理器里，因为一个数据库查询跑了 120 秒还没返回。但这不是最糟的——最糟的是主线程在 `tools/call` 上阻塞期间，`tools/list_changed` 通知在 stdin 里排队，客户端永远收不到。

这个 bug 根因就六个字：忘写 `sys.stdout.flush()` 了。

对，就这六个字。`tools/call` 的结果已经写进了 Python 的输出缓冲区，但缓冲区没满，没自动刷新，于是响应就卡在内存里。客户端在另一端读着空管道，等了整整 5 分钟直到超时重连，然后一切重来，又卡住。

MCP 服务器构建看起来简单——180 行标准库代码，一个 while 循环从 stdin 读 JSON-RPC、在 stdout 写 JSON-RPC——但这 180 行里的每一个细节都是生产里踩过坑才能记住的。

## Stdio 传输：简单到危险

MCP 的 stdio 模式本质是一个约定：**你的服务器是客户端的子进程，客户端往你的 stdin 写、从你的 stdout 读。每一行一个完整的不带换行的 JSON 对象，以 `\n` 分隔。**

这听起来比 HTTP 简单一万倍——没有端口、没有路由、没有中间件。但"简单"和"不容易出错"是两回事。

**第一条命：`sys.stdout.flush()` 是你最好的朋友。** Python 的输出默认是行缓冲的，但如果你写了 `print(json.dumps(...), file=sys.stdout)`，Python 的输出缓冲机制在某些条件下会推迟实际的 `write()` 调用。如果缓冲区没有满（通常是 8KB），你的 JSON 响应就待在内存里，客户端那边啥也看不到。每次写响应后加 `sys.stdout.flush()` 不是"最佳实践"，是"不写就等死"。

**第二条命：日志往 stderr 打，打死也别往 stdout 打。** 这个错误我见过至少五个新人犯——用 `print()` 打调试日志。stdout 是 JSON-RPC 专用通道，你往里面塞一行 `"处理工具调用: get_weather"`，客户端收到后试图 JSON 解析——炸了。客户端崩溃，但你的服务器还在运行，然后客户端重启，又来一次。这条规则简单但严格：**stdout 上除了 JSON-RPC 信封，连空格都不能多。**

**第三条命：stdin 的 EOF 就是你的 shutdown 信号。** 客户端关闭了连接，你的 `sys.stdin.readline()` 返回空字符串。这不是"空闲"，这是"关门了"。你需要清理资源——关闭数据库连接、flush 最后的日志、exit(0)。不处理 EOF 会导致孤儿进程——你的服务器还以为在等下一行命令，其实客户端早没了。

**第四条命：SIGPIPE 不是你忽略就能过去的。** 如果客户端进程挂了（比如用户强制退出了 Claude Desktop），你的服务器下次往 stdout 写数据时会收到 SIGPIPE 信号。默认行为是杀死你的进程并生成一个 core dump。如果你没在代码里处理这个信号，你的日志里会多一条"Broken pipe"然后静默死亡。

## 分发循环：MCP 服务器的"心脏"

这个循环就是整个 MCP 服务器的全部骨架：

```
while True:
    line = stdin.readline()
    if not line: break  # EOF
    request = json.loads(line)
    if has id:  handle_request -> write response
    else:       handle_notification  # no response
```

我管它叫"心脏"——每读一行、跳一下。它做的事只有一件：读 JSON → 判断请求还是通知 → 路由到正确的处理器 → 写响应（如果是请求的话）。

但这个简单的心脏会得两种心脏病。

**第一种：阻塞性心肌炎。** `handle_request` 里的某个工具执行器（比如 `tools/call` → `notes_search`）花费太长时间（比如一个超慢的数据库查询），导致整个 while 循环卡住。通知在 stdin 排队，客户端在等响应。解决方案不是多线程（stdin 没有多线程的概念），而是给工具执行加超时——`tools/call` 内部用 `timeout` 包装，超时就返回 `isError: true` 而不是沉默。

**第二种：无 ID 恐慌。** 客户端发了一个没有 `id` 的消息，你的 `handle_notification` 处理过程中崩溃了——抛了一个异常，但你没有 catch。主循环在异常处终止，服务器静默死亡。**通知处理器里必须 try-except 所有代码——通知不出错不是"应该的"，是你必须兜底的。**

## `isError: true` 和 JSON-RPC 错误：两套不同的"错了"

这是我面试 MCP 开发者的必考题："`tools/call` 返回的 `isError: true` 和 JSON-RPC 错误码有什么区别？"答不上来的人，大概率没在生产环境跟模型做过交互。

JSON-RPC 错误（`{error: {code: -32601, message: "Method not found"}}`）是**协议层失败**——给开发者看的。方法名写错了、参数类型不对、能力不支持。这类错误客户端直接处理，不会传给模型。

`isError: true`（`{content: [{type: "text", text: "Database connection refused"}], isError: true}`）是**工具层失败**——给模型看的。工具收到了合法的调用请求，但执行时失败了。这个错误被注入到模型的上下文中，让模型知道"刚才那个操作失败了，换个方式试试，或者告诉用户"。

**这个区分的核心价值是：模型需要看到工具执行的失败，才能自修正。如果你把工具失败当 JSON-RPC 错误返回，客户端拦截了它，模型就永远不知道失败——它会继续基于"成功执行"的假设来回复用户，然后产生幻觉。**

## 注解系统：你不说话，客户端就猜错

`annotations` 看起来像可选的元数据装饰，但实际上它是服务器和客户端之间唯一的"安全手语"。`readOnlyHint: true` 意思是"这个工具不会改变任何东西，你随便重试"；`destructiveHint: true` 意思是"这个工具可能会删数据，弹确认框"；`idempotentHint: true` 意思是"相同输入总是相同输出，缓存它"；`openWorldHint: true` 意思是"这个工具跟外部系统交互，结果不受控制"。

Claude Desktop 看到 `destructiveHint: true` 会弹出确认对话框。Cursor 看到 `readOnlyHint: true` 会把这个工具标记为"安全"、不提示用户。你的 MCP 网关（Phase 13 · 17）可以根据这些注解做路由决策——把高风险的破坏性操作路由到需要审批的队列。

**如果你不写注解，客户端默认把所有工具当"未知风险"处理——这意味着用户会看到不必要的确认框，或者更糟，客户端把破坏性操作当安全操作悄无声息执行了。**

## 从标准库到 FastMCP：你不是在学两个东西

这课的 180 行标准库服务器和 FastMCP 的 80 行装饰器版本不是两个独立的知识点——它们是一件事的两种表达。

当你理解了 `tools/call` 处理器返回 `{content: [...], isError: false}` 这个约定，FastMCP 的 `@app.tool()` 装饰器就只是一层语法糖——装饰器内部的逻辑还是返回同样的字典、同样的内容块列表、同样的 `isError` 标志。

这就是为什么这课坚持先教标准库，再展示 FastMCP。不是因为我拒绝框架，是因为**如果你不亲手写过 JSON-RPC 分发循环，你就永远不理解 FastMCP 到底帮你省了什么——以及它在哪些地方替你做了你不同意的事。**

升级到 FastMCP 的正确时机：当你已经写了三个 MCP 服务器，发现每个写的分发循环都一样，工具执行器都一样规范，该 flush 的地方都一样。这时候框架才有意义——它不是"帮你写你写不出的东西"，是"帮你省掉你已经写过三次的样板代码"。

## 生产现实

- **工具调用 P50 延迟**（本机 stdio）：`tools/call` 往返约 5-15ms（不含工具执行逻辑）。如果你压到了 2ms，大概率是网络 I/O 的错觉——你用的是 stdio 本地管道。
- **内容块大小**：单个 `text` 块 500 字符以下是最佳窗口。超过 10,000 字符，模型上下文在处理结果时可能出现"细节丢失"——特别是当多个工具调用结果同时进入上下文时。
- **Stdio buffer size**：默认 8KB。如果你的工具返回一个 12KB 的 JSON 结果，它会在第一次 `write()` 时输出 8KB，第二次输出 4KB——客户端收到两个 chunk，但这不影响 JSON-RPC 解析，因为每个响应是一行。但如果你的结果超过 64KB，就该考虑 `resource` 块而不是 `text` 块了。

## 收尾

构建 MCP 服务器就是写一个 while 循环。这句话技术上是对的，但工程的真相是：**那个 while 循环每转一圈，你都在用六个原语、一次握手、一条线格式，跟一个你不知道它接下来会说什么的系统对话。那个系统不是你的用户——是你的用户在用 LLM 跟你的服务器说话。你写的每一行代码，最终都会被一个不认识你、不理解你意图、但记忆力超群的模型读到。写清楚、flush 干净、错误说人话。**
