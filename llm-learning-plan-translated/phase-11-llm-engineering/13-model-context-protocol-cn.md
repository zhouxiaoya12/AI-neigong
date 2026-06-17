# 模型上下文协议 (MCP)

> 2025年之前构建的每个LLM应用都发明了自己的工具schema。然后Anthropic发布了MCP，Claude采用了它，OpenAI采用了它，到2026年它已成为连接任何LLM与任何工具、数据源或Agent的默认传输格式。写一个MCP服务器，每个宿主都能与它通信。

**类型：** 构建 | **语言：** Python | **前置要求：** Phase 11 · 09, 03 | **时间：** 约75分钟

## 问题

你发布了一个需要三个工具的聊天机器人：数据库查询、日历API和文件读取器。你为Claude写了三个JSON Schema。然后销售团队想要在ChatGPT中使用同样的工具——你为OpenAI的`tools`参数重写了它们。然后你添加了Cursor、Zed和Claude Code——三次更多的重写。一周后Anthropic添加了一个新字段；你更新了六个schema。

MCP将这个N×M集成矩阵折叠成一个规范。一个JSON-RPC规范。一个服务器暴露工具、资源和提示。任何兼容宿主都能发现并调用它们。

## 概念

### 三个原语

1. **工具**（Tools）——模型可以调用的函数。每个具有名称、描述、JSON Schema输入和处理器。
2. **资源**（Resources）——模型或用户可以请求的只读内容。按URI寻址。
3. **提示**（Prompts）——用户可以调用的可重用模板化提示。

### 传输格式

JSON-RPC 2.0 over stdio、WebSocket或streamable HTTP。每条消息：`{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。

### MCP不是什么

- 不是检索API（RAG仍然决定拉取什么，MCP是传输层）
- 不是Agent框架（LangGraph/PydanticAI在MCP之上）
- 不绑定Anthropic（规范和实现是开源的）

## 构建

### Step 1: 最小MCP服务器

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    return '{"env": "prod"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    return f"You are a senior {language} reviewer.\n\n{code}"
```

三个装饰器注册三个原语。类型注解成为宿主看到的JSON Schema。

### Step 2: 客户端调用

```python
from mcp.client.stdio import stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("add", {"a": 3, "b": 5})
```

### Step 3: Streamable HTTP

stdio适用于本地开发。对于远程工具，使用streamable HTTP——每次请求一个POST，可选的SSE用于进度更新。

### 服务器模式

| 模式 | stdio | Streamable HTTP |
|------|-------|-----------------|
| 传输 | 子进程管道 | 网络请求 |
| 路由 | N/A | GET/POST/DELETE端点 |
| 状态 | 进程内 | 启动时实例化，跨请求复用 |
| 规模 | 1宿主:1服务器 | N宿主:1服务器 |

## 关键术语

| 术语 | 含义 |
|------|------|
| MCP | 连接LLM与工具/数据源的开放协议 |
| 工具 | 模型可以调用的函数——函数的JSON Schema定义 |
| 资源 | 模型或用户可以请求的只读数据——通过URI寻址 |
| 提示 | 用户可以调用的可重用模板化提示模板 |
| 宿主 | 运行LLM并消费MCP服务器的应用程序（Claude Desktop、Cursor） |

## 扩展阅读

- [MCP规范](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP社区服务器](https://github.com/modelcontextprotocol/servers)

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

文档简洁有力，聚焦MCP的核心价值——"一次编写，到处运行"的工具互操作标准。不是教"怎么做工具调用"（第09课已覆盖），而是教"如何让工具变得可移植"。优势是用"N×M集成矩阵→一个规范"的清晰叙事展示了MCP解决的真实问题。

### 二、核心洞察

1. **MCP是函数调用的HTTP**：正如HTTP标准化了Web通信，MCP标准化了LLM工具互操作
2. **三个原语覆盖所有交互**：工具、资源、提示——不是"多"，是"恰好足够"
3. **Stdio vs HTTP是部署架构决策**：本地=stdio，远程=streamable HTTP
4. **MCP不绑定Anthropic**——它是开源规范，三大提供商都支持
5. **MCP让工具从应用级变为生态级**：一次编写，所有兼容宿主可用

### 三、一句话总结

**MCP不是"更好的函数调用"，是"可移植的函数调用"——写一个MCP服务器，所有兼容宿主都能与之对话。**

---

# 🎓 Agent 架构课：MCP——为什么工具可移植性比工具功能更重要

每一个LLM应用都在重新发明轮子。你写了`get_weather`的工具定义——在Claude的语法中。然后你发现用户也在用Cursor，你需要同样的工具。重写。然后ChatGPT支持了——再重写一次。

这就是MCP存在的原因。不是"又一个标准"，是**唯一的互操作标准**。就像HTTP标准化了Web通信，MCP标准化了LLM工具通信。写一个MCP服务器，定义你的工具/资源/提示，任何兼容宿主都能发现并调用它们。

## 结语清单

1. ☐ 工具是否在MCP服务器中定义，而非内联在每个宿主中？
2. ☐ 资源是否与工具明确分离（后者是读操作，前者是读写操作）？
3. ☐ 对于远程工具是否使用streamable HTTP？
4. ☐ 错误处理是否通过JSON-RPC错误码标准化？

**一句金句：你的工具定义不应该比你的API endpoint更脆弱——MCP给了工具与REST API同等的可移植性保证。**
