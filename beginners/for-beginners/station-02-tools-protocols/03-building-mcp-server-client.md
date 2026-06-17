# 动手写 MCP 服务器和客户端：从 stdio 到多服务编排

> 预计阅读：30分钟 | 难度：动手 | 前置条件：理解了 MCP 基础
>
> 你将学到：怎么从零写一个 MCP 服务器（暴露工具+资源+提示模板），怎么写一个客户端同时管多个服务器。有完整代码骨架。

---

## 🚗 开场翻车：hello-world 跑通了，生产炸了

大多数人学 MCP 的路径：看教程 → 跑一个 `print("hello")` 的 stdio 服务器 → 觉得"我会了"。

然后他们在生产环境里同时挂了三个 MCP 服务器——文件系统、Postgres、GitHub。三个服务器同时往 stdout 写。消息混在一起。Agent 调了 `search`——但这个工具两个服务器都有。调的是哪个？不知道。返回的结果是哪个？也不知道。

**MCP 服务器不难写。难的是客户端怎么同时管好几个服务器。**

---

## 🔧 Part 1：写一个 MCP 服务器

### 最简单的东西：stdio 管道

MCP 服务器是一个**子进程**。客户端启动它，消息通过 stdin/stdout 流动。每行一个 JSON 对象。

```
客户端启动你的服务器
    ↓
客户端通过 stdin 发 JSON-RPC 请求
    ↓
你的服务器处理后通过 stdout 回 JSON-RPC 响应
    ↓
循环，直到 stdin 关闭（EOF）
```

**三条铁律：**
1. 除了 JSON-RPC 信封，什么东西都别往 stdout 打。调试日志走 stderr。
2. 每个请求必须回一个带相同 `id` 的响应。
3. 通知（没 `id` 的消息）不许回响应。

### 分发循环——服务器的心脏

```python
import sys, json

def handle_request(method, params, req_id):
    """根据 method 分发到不同的处理函数"""
    if method == "initialize":
        return initialize(params)
    elif method == "tools/list":
        return list_tools()
    elif method == "tools/call":
        return call_tool(params)
    elif method == "resources/list":
        return list_resources()
    elif method == "resources/read":
        return read_resource(params)
    elif method == "prompts/list":
        return list_prompts()
    elif method == "prompts/get":
        return get_prompt(params)
    else:
        return {"error": {"code": -32601, "message": f"未知方法: {method}"}}

def main():
    for line in sys.stdin:
        msg = json.loads(line)
        
        if "id" in msg:  # 这是一个请求，必须回响应
            result = handle_request(msg["method"], msg.get("params", {}), msg["id"])
            response = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()  # 别忘了 flush！
        else:  # 这是通知，不回响应
            handle_notification(msg["method"], msg.get("params", {}))
```

### initialize——握手

服务器必须告诉客户端"我是谁、我会什么"：

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},     # 我提供工具
            "resources": {"listChanged": True},  # 我提供资源
            "prompts": {"listChanged": False},   # 我不提供提示模板
        },
        "serverInfo": {"name": "my-notes-server", "version": "1.0.0"},
    }
```

**关键：只声明你实际支持的功能。** 声明了但不实现的，客户端会卡住。

### 暴露工具

```python
NOTES = {}  # 简单的内存存储

def list_tools():
    return {"tools": [
        {
            "name": "notes_create",
            "description": "创建一条新笔记",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "笔记标题"},
                    "content": {"type": "string", "description": "笔记内容"}
                },
                "required": ["title", "content"]
            }
        },
        {
            "name": "notes_search",
            "description": "搜索笔记",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    ]}

def call_tool(params):
    name = params["name"]
    args = params.get("arguments", {})
    
    if name == "notes_create":
        note_id = str(len(NOTES) + 1)
        NOTES[note_id] = {"title": args["title"], "content": args["content"]}
        return {"content": [{"type": "text", "text": f"笔记已创建，ID: {note_id}"}]}
    
    elif name == "notes_search":
        results = [n for n in NOTES.values() if args["query"].lower() in n["title"].lower()]
        return {"content": [{"type": "text", "text": f"找到 {len(results)} 条笔记"}]}
    
    else:
        return {"content": [{"type": "text", "text": f"未知工具: {name}"}], "isError": True}
```

**注意两种错误的区别：**
- 协议级错误（未知方法）：用 JSON-RPC error
- 工具级错误（工具执行失败）：返回 `isError: true` 的内容块——让 Agent 在上下文里"看到"失败

### 暴露资源（只读数据）

```python
def list_resources():
    return {"resources": [
        {"uri": f"notes://{nid}", "name": n["title"], "mimeType": "text/plain"}
        for nid, n in NOTES.items()
    ]}

def read_resource(params):
    uri = params["uri"]
    note_id = uri.replace("notes://", "")
    note = NOTES.get(note_id)
    if not note:
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": "笔记不存在"}]}
    return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"{note['title']}\n\n{note['content']}"}]}
```

---

## 🔧 Part 2：写一个多服务器客户端

真正的 Agent 宿主（Claude Desktop、Cursor、Goose）同时跑多个 MCP 服务器。客户端的工作：

1. 启动每个服务器子进程
2. 分别握手
3. 把各服务器的工具列表合并成一个大命名空间
4. Agent 调了工具 → 找到所属服务器 → 路由过去

### 启动和握手

```python
import subprocess, json

class MCPSession:
    def __init__(self, name, command):
        self.name = name
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self.capabilities = None
        self.tools = []
    
    def send_request(self, method, params=None):
        """发请求，读响应"""
        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        return json.loads(line)["result"]

    def initialize(self):
        self.capabilities = self.send_request("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "my-client", "version": "1.0"}
        })
        # 发 initialized 通知（不期待响应）
        self.process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }) + "\n")
        self.process.stdin.flush()
    
    def discover_tools(self):
        result = self.send_request("tools/list")
        self.tools = result["tools"]
        return self.tools
```

### 合并命名空间——处理冲突

两个服务器都有一个 `search` 工具怎么办？

```python
# 方案：按服务器名加前缀
for session in sessions:
    for tool in session.tools:
        namespaced_name = f"{session.name}/{tool['name']}"
        routing_table[namespaced_name] = session
        # "notes/search" → notes 服务器
        # "files/search" → files 服务器
```

这是 Claude Desktop 的做法。清晰、不会冲突。

### 路由工具调用

```python
def route_tool_call(tool_name, arguments):
    session = routing_table.get(tool_name)
    if not session:
        return {"error": f"工具 {tool_name} 不存在"}
    return session.send_request("tools/call", {
        "name": tool_name.split("/", 1)[1],  # 去掉前缀
        "arguments": arguments
    })
```

---

## 🎮 互动练习

你有一个文件服务器和一个数据库服务器。Agent 调了 `files/read`。请写出客户端的分发过程——从收到调用到返回结果，每一步在做什么？

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **服务器 = stdin/stdout 管道。** 每行一个 JSON。调试走 stderr。写完就 flush。

2. **initialize 只声明你实际支持的功能。** 声明了不实现 = Agent 卡住。

3. **工具失败返回 `isError: true`，不是抛异常。** Agent 需要"看到"失败才能自己修正。

4. **客户端合并命名空间必须处理工具名冲突。** 加前缀是最简单最安全的方式。

5. **多服务器 = 多子进程 + 统一路由表。** 每个服务器自己的状态，客户端只管"Agent 要什么 → 发给谁"。

---

## 📝 练习答案

**Agent 调了 `files/read("config.yaml")`：**

1. 客户端查到 `files/read` → 属于 `files` 服务器
2. 向 files 服务器的 stdin 写入 `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"read","arguments":{"path":"config.yaml"}}}`
3. 从 files 服务器的 stdout 读取响应
4. 把结果喂回 Agent 的上下文窗口

---

> **MCP 服务器不难写——一个 while 循环读 stdin 写 stdout。难的是客户端同时管好几个服务器还不出乱子。命名空间加前缀，路由表查归属——就这么简单，但没做的话凌晨三点就是你对着日志哭。**
