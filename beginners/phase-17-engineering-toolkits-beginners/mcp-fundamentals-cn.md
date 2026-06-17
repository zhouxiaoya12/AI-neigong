# MCP 基础：Agent 世界的 USB-C

> 详细版见 `for-beginners/station-02/01-mcp-fundamentals.md`  
> 动手版见 `for-beginners/station-02/03-building-mcp-server-client.md`  
> 安全版见 `for-beginners/station-02/04-mcp-security.md`

---

## 一句话：MCP 是什么

MCP（Model Context Protocol）= Agent 世界的 USB-C。写一个 MCP 服务器，所有 MCP 客户端都能用。

- 10,000+ 公开服务器
- 300+ 客户端（Claude Desktop、ChatGPT、Cursor、VS Code...）
- 月 1.1 亿 SDK 下载
- Linux 基金会管理

---

## 六个原语

| | 原语 | 干啥 |
|---|------|------|
| 服务器 | **Tools** | Agent 调用的操作 |
| 服务器 | **Resources** | Agent 能读的数据 |
| 服务器 | **Prompts** | 可复用的对话模板 |
| 客户端 | **Roots** | 允许访问的 URI 范围 |
| 客户端 | **Sampling** | 服务器让客户端跑 LLM |
| 客户端 | **Elicitation** | 服务器向用户要输入 |

---

## 三阶段生命周期

```
initialize（握手：互相说"我会xxx"） 
    → operation（干活：调工具、读资源） 
    → shutdown（关闭连接）
```

---

## JSON-RPC 2.0 消息格式

```json
// 请求
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}

// 响应
{"jsonrpc": "2.0", "id": 1, "result": {...}}

// 通知（不需要回复）
{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
```

---

> **MCP = Agent 世界的 USB-C。写一次服务器，所有 Agent 都能用。**
