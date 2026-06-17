# 函数调用与工具使用

> LLM 什么也做不了。它们生成文本。这就是全部能力。它们不能查看天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每一个"AI Agent"都是一个 LLM 在生成 JSON，告诉你要调用哪个函数——然后你的代码真正去调用它。模型是大脑。工具是手。函数调用是连接它们的神经系统。

**类型**：构建  
**语言**：Python  
**前置要求**：阶段 11 第 03 课（结构化输出）  
**预计时间**：约 75 分钟  
**相关内容**：阶段 11 · 第 14 课（模型上下文协议 MCP）——当工具需要跨主机共享时，从内联函数调用升级到 MCP 服务器。本课覆盖内联场景；MCP 覆盖协议场景。

## 学习目标

- 实现函数调用循环：定义工具 Schema、解析模型的工具调用 JSON、执行函数并返回结果
- 设计带有清晰描述和类型化参数的工具 Schema，使模型能够可靠地调用
- 构建一个多轮 Agent 循环，串联多次函数调用来回答复杂查询
- 处理函数调用的边界情况：并行工具调用、错误传播、以及防止无限工具循环

## 问题陈述

你做了一个聊天机器人。用户问："东京现在天气怎么样？"

模型回答："我无法获取实时天气数据，但根据季节，东京目前大约 15 摄氏度……"

这是一个披着免责声明外衣的幻觉。模型不知道天气。它永远也不会知道。天气每小时都在变。模型的训练数据已经过时好几个月了。

正确答案需要调用 OpenWeatherMap API，获取当前温度，返回真实数据。模型不能调用 API。你的代码可以。缺失的环节：一套结构化协议，让模型可以说"我需要用这些参数调用天气 API"，让你的代码可以执行它并把结果返回。

这就是函数调用。模型输出结构化 JSON，描述用哪些参数调用哪个函数。你的应用程序执行这个函数。结果返回到对话中。模型使用结果来生成最终答案。

没有函数调用，LLM 是百科全书。有了它，它们变成了 Agent。

## 核心概念

### 函数调用循环

每次工具使用交互都遵循相同的五步循环。

```
用户 → 应用程序 → 模型（带工具定义）→ 模型输出 tool_call → 应用程序执行工具 → 结果回传给模型 → 模型生成最终回答
```

步骤 1：用户发送消息。步骤 2：模型接收消息以及工具定义（描述可用函数的 JSON Schema）。步骤 3：模型不回复文本，而是输出一个工具调用——一个包含函数名和参数的结构化 JSON 对象。步骤 4：你的代码执行函数并捕获结果。步骤 5：结果返回给模型，模型现在有了真实数据来生成最终答案。

模型从不执行任何东西。它只决定调用什么、用什么参数。你的代码是执行者。

### 工具定义：JSON Schema 契约

每个工具由一个 JSON Schema 定义，告诉模型这个函数做什么、接受什么参数、以及参数必须是什么类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的当前天气。返回摄氏温度值及天气状况。",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称，例如 '东京' 或 '旧金山'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型读取它们来决定何时以及如何使用工具。"获取天气"这种模糊描述产生的工具选择远不如"获取指定城市的当前天气。返回摄氏温度值及天气状况。"的描述可靠。description 就是工具选择的提示词。

### 各厂商对比

主流厂商都支持函数调用，但 API 接口各有不同。

| 厂商 | API 参数 | 工具调用格式 | 并行调用 | 强制调用 |
|------|----------|-------------|---------|---------|
| OpenAI（GPT-5、o4） | `tools` | `tool_calls[].function` | 支持（每轮多个） | `tool_choice="required"` |
| Anthropic（Claude 4.6/4.7） | `tools` | `content[].type="tool_use"` | 支持（多个块） | `tool_choice={"type":"any"}` |
| Google（Gemini 3） | `function_declarations` | `functionCall` | 支持 | `function_calling_config` |
| 开源模型（Llama 4、Qwen3、DeepSeek-V3） | Llama 4 原生 `tools`；其他用 Hermes 或 ChatML | 混合 | 取决于模型 | 基于提示词或 `tool_choice`（如支持） |

到 2026 年，三大闭源厂商已收敛到几乎相同的基于 JSON Schema 的格式。Llama 4 原生提供与 OpenAI 一致的 `tools` 字段。开源微调模型仍有差异——Hermes 格式（NousResearch）是第三方微调中最常见的格式。对于跨主机共享的工具，优先使用 MCP（阶段 11 · 第 14 课）而非内联函数调用——服务器对所有模型都一样。

### 工具选择模式：自动、强制、指定

你可以控制模型何时使用工具。

**自动（auto，默认）**：模型自己决定是调用工具还是直接回答。"2+2 等于几？"——直接回答。"天气怎么样？"——调用工具。

**强制（required）**：模型必须至少调用一个工具。当你知道用户意图需要工具时使用。防止模型猜测而不是查询真实数据。

**指定函数**：强制模型调用特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 保证天气工具被调用，无论查询是什么。用于路由——当上游逻辑已经确定需要哪个工具时。

### 并行函数调用

GPT-4o 和 Claude 可以在单轮中调用多个函数。用户问："东京和纽约的天气怎么样？"模型同时输出两个工具调用：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两者（理想情况下并发执行），返回两个结果，模型综合成单个响应。这将往返次数从 2 次减少到 1 次。对于每次查询需要 5-10 次工具调用的 Agent，并行调用可将延迟降低 60-80%。

### 结构化输出 vs 函数调用

第 03 课讲了结构化输出。函数调用使用相同的 JSON Schema 机制，但目的不同。

**结构化输出**：强制模型以特定格式产出数据。输出是最终产品。例如：从文本中提取产品信息为 `{name, price, in_stock}`。

**函数调用**：模型声明执行某个操作的意图。输出是一个中间步骤。例如：`get_weather(city="Tokyo")`——模型在请求执行一个操作，而不是产出最终答案。

数据提取用结构化输出。模型与外部系统交互用函数调用。

### 安全：不可协商的规则

函数调用是你能给 LLM 的最危险的能力。模型选择执行什么。如果你的工具集包含数据库查询，模型会构造查询。如果包含 Shell 命令，模型会写它们。

**规则 1：绝不让模型生成的 SQL 直接传给数据库。** 模型可能并且会生成 DROP TABLE、UNION 注入或返回所有行的查询。始终参数化。始终验证。始终使用操作白名单。

**规则 2：白名单函数。** 模型只能调用你显式定义的函数。永远不要构建一个通用的"按名称执行任意函数"工具。如果你有 50 个内部函数，只暴露用户需要的那 5 个。

**规则 3：验证参数。** 模型可能传入 `"; DROP TABLE users; --"` 作为城市名。执行前验证每个参数的类型、范围和格式。

**规则 4：清理工具结果。** 如果工具返回敏感数据（API 密钥、个人信息、内部错误），在发送回模型之前过滤掉。模型会在其响应中原样包含工具结果。

**规则 5：限制工具调用频率。** 循环中的模型可能调用工具数百次。设置上限（每轮对话 10-20 次是合理的）。打破无限循环。

### 错误处理

工具会失败。API 超时。数据库宕机。文件不存在。模型需要知道工具何时失败以及为什么失败。

以结构化工具结果的形式返回错误，而不是抛出异常：

```json
{
  "error": true,
  "message": "未找到城市 'Toky'。您是不是想查 'Tokyo'？",
  "code": "CITY_NOT_FOUND"
}
```

模型读取这些信息，调整参数并重试。模型擅长从结构化错误消息中自我修正，但在空响应或通用的"出了点问题"错误面前无能为力。

### MCP：模型上下文协议

MCP 是 Anthropic 的工具互操作性开放标准。与其每个应用程序定义各自的工具，MCP 提供了一套通用协议：工具由 MCP 服务器提供，由 MCP 客户端（如 Claude Code、Cursor 或你的应用程序）消费。

一个 MCP 服务器可以向任何兼容的客户端暴露工具。一个 Postgres MCP 服务器让任何 MCP 兼容的 Agent 获得数据库访问。一个 GitHub MCP 服务器让任何 Agent 获得仓库访问。工具定义一次，到处可用。

MCP 对于函数调用，就如同 HTTP 对于网络通信。它标准化了传输层，使工具变得可移植。

---

## 动手构建

### 第 1 步：定义工具注册表

构建一个存储工具定义及其实现的注册表。每个工具有一个 JSON Schema 定义（模型看到的）和一个 Python 函数（你的代码执行的）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 第 2 步：实现 5 个工具

构建一个计算器、天气查询、网页搜索模拟器、文件读取器和代码运行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"表达式包含非法字符: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "多云", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "晴朗", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "下雨", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "有雾", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "晴朗", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"未找到城市 '{city}'。",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI 函数调用指南", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "学习如何将 LLM 连接到外部工具。"},
        {"title": "Anthropic 工具使用", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude 可以与外部工具和 API 交互。"},
    ],
    "MCP protocol": [
        {"title": "模型上下文协议", "url": "https://modelcontextprotocol.io", "snippet": "连接 AI 模型与数据源的开放标准。"},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "免费的天气 API，提供当前、预测和历史数据。"},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\n一个从零构建的工具使用 Agent。",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "不允许路径穿越。", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"文件 '{path}' 未找到。", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"不支持语言 '{language}'。仅支持 'python'。"}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"禁止的操作: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 第 3 步：注册所有工具

```python
def register_all_tools():
    register_tool(
        "calculator",
        "求解数学表达式。支持 +, -, *, /, 括号和小数。返回数值结果。",
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，例如 '(10 + 5) * 3'"},
                "precision": {"type": "integer", "description": "结果的小数位数", "default": 2}
            },
            "required": ["expression"]
        },
        calculator,
    )
    register_tool(
        "get_weather",
        "获取指定城市的当前天气。返回温度、天气状况、湿度和风速。",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如 'Tokyo' 或 'San Francisco'"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位，默认为 celsius"}
            },
            "required": ["city"]
        },
        get_weather,
    )
    register_tool(
        "web_search",
        "搜索网络信息。返回包含标题、URL 和摘要的结果列表。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "max_results": {"type": "integer", "description": "返回的最大结果数", "default": 3}
            },
            "required": ["query"]
        },
        web_search,
    )
    register_tool(
        "read_file",
        "读取文件内容。返回文件内容、大小和行数。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对文件路径，例如 'data/config.json'"}
            },
            "required": ["path"]
        },
        read_file,
    )
    register_tool(
        "run_code",
        "在沙盒环境中执行 Python 代码。设置 'result' 变量来返回输出。",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
                "language": {"type": "string", "enum": ["python"], "description": "编程语言"}
            },
            "required": ["code"]
        },
        run_code,
    )
```

### 第 4 步：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具、执行工具并将结果返回。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    # 检测天气相关查询
    if any(word in msg for word in ["weather", "temperature", "forecast", "天气", "温度", "气温"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    # 检测计算相关查询
    if any(word in msg for word in ["calculate", "compute", "math", "计算", "算", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    # 检测搜索相关查询
    if any(word in msg for word in ["search", "find", "look up", "google", "搜索", "查找", "查"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    # 检测文件读取相关查询
    if any(word in msg for word in ["read", "file", "open", "cat", "show", "读", "文件", "打开", "查看"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    # 检测代码执行相关查询
    if any(word in msg for word in ["run", "execute", "code", "python", "运行", "执行", "代码"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"未知工具: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"参数无效: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 第 5 步：参数验证

构建一个验证器，在执行前对照 JSON Schema 检查工具调用参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"未知工具: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"参数必须是对象，收到的是 {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"缺少必需参数: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"未知参数: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"参数 '{arg_name}': 期望 {expected_type}，收到 {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"参数 '{arg_name}': '{arg_value}' 不在 {prop_schema['enum']} 中")

    return errors
```

### 第 6 步：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print(" 函数调用与工具使用 演示")
    print("=" * 60)

    print("\n--- 已注册工具 ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f" {name}: {desc}...")
        print(f" 参数: {params}")

    print(f"\n--- 参数验证 ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "有效调用"),
        ("get_weather", {}, "缺少必需参数"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "无效枚举值"),
        ("calculator", {"expression": 123}, "类型错误（int 传给 string）"),
        ("unknown_tool", {"x": 1}, "未知工具"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "通过" if not errors else f"错误: {errors}"
        print(f" {label}: {status}")

    print(f"\n--- 工具执行 ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n {call['name']}({json.dumps(call['arguments'])})")
        print(f" -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f" 耗时: {result['execution_time_ms']}ms")

    print(f"\n--- 完整函数调用循环 ---")
    test_queries = [
        "东京的天气怎么样？",
        "计算 (100 + 250) * 0.15",
        "搜索 MCP protocol",
        "读取配置文件",
        "运行一些 Python 代码",
        "讲个笑话",
    ]
    for query in test_queries:
        print(f"\n 用户: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f" 工具: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f" 结果: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f" [未调用工具 -- 直接回答]")
        print(f" 迭代次数: {result['iterations']}")

    print(f"\n--- 并行工具调用 ---")
    multi_city_query = "tokyo 和 london 的天气怎么样？"
    print(f" 用户: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f" 工具调用数: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f" {city}: {temp}°C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- 安全检查 ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f" {tool_name}({list(args.values())[0][:40]}): {'已拦截' if blocked else '已放行'}")
```

## 实际使用

### OpenAI 函数调用

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "获取指定城市的当前天气",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "东京的天气怎么样？"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "东京的天气怎么样？"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 通过 `response.choices[0].message.tool_calls` 返回工具调用。每个调用都有一个 `id`，返回结果时必须包含它。模型使用该 ID 将结果与调用匹配。GPT-4o 可以在单个响应中返回多个工具调用——遍历并全部执行。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "获取指定城市的当前天气",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "东京的天气怎么样？"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "东京的天气怎么样？"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 以 `type: "tool_use"` 的内容块返回工具调用。工具结果放在用户消息中，类型为 `type: "tool_result"`。注意关键区别：Anthropic 使用 `input_schema` 定义工具参数，而 OpenAI 使用 `parameters`。

### MCP 集成

```python
# MCP 服务器通过标准化协议暴露工具。
# 任何 MCP 兼容的客户端都可以发现并调用这些工具。
#
# 示例：连接到 Postgres MCP 服务器
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 将工具实现与工具消费解耦。Postgres 服务器了解 SQL。GitHub 服务器了解 GitHub API。你的 Agent 只需要发现并调用工具——不需要为每个集成写厂商特定的代码。

## 交付物

本课产出 `outputs/prompt-tool-designer.md`——一个用于设计工具定义的可复用提示词模板。给它一个你希望工具做什么的描述，它就生成完整的 JSON Schema 定义，包含描述、类型和约束。

同时产出 `outputs/skill-function-calling-patterns.md`——一个在生产环境中实现函数调用的决策框架，涵盖工具设计、错误处理、安全和厂商特定模式。

## 练习

1. **添加第 6 个工具：数据库查询。** 用内存表实现一个模拟 SQL 工具。该工具接受表名和过滤条件（不是原始 SQL）。验证表名在白名单中，且过滤操作符限制为 `=`、`>`、`<`、`>=`、`<=`。以 JSON 形式返回匹配的行。

2. **实现带错误反馈的重试。** 当工具调用失败时（例如城市未找到），将错误消息返回给模型决策函数，让它修正参数。追踪每次调用重试了多少次。设置每次工具调用最多重试 3 次。

3. **构建多步 Agent。** 有些查询需要串联工具调用："读取配置文件，告诉我配置了什么模型，然后搜索该模型的定价。"实现一个循环，一直运行直到模型决定不再需要更多工具，将累积的结果传递到每个决策步骤。限制最多 10 次迭代以防止无限循环。

4. **衡量工具选择准确率。** 创建 30 个带有预期工具名称的测试查询。对所有 30 个执行你的决策函数，测量它选择正确工具的百分比。找出哪些查询最容易在工具之间产生混淆。

5. **实现工具调用缓存。** 如果在 60 秒内以相同参数调用同一个工具，返回缓存结果而不重新执行。使用以 `(tool_name, frozenset(args.items()))` 为键的字典。在 20 个查询的对话中测量缓存命中率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 函数调用 | "工具使用" | 模型输出结构化 JSON，描述用特定参数调用某个函数——你的代码执行它，不是模型 |
| 工具定义 | "函数 Schema" | 一个 JSON Schema 对象，描述工具的名称、用途、参数和类型——模型读取它来决定何时及如何使用工具 |
| 工具选择模式 | "调用模式" | 控制模型是必须调用工具（required）、可以调用工具（auto）、还是必须调用指定工具（named） |
| 并行调用 | "多工具" | 模型在单轮中输出多个工具调用，减少往返次数——GPT-4o 和 Claude 都支持 |
| 工具结果 | "函数输出" | 执行工具后的返回值，作为消息发送回模型，使其可以在响应中使用真实数据 |
| 参数验证 | "输入检查" | 在执行前验证模型生成的参数是否匹配预期类型、范围和约束 |
| MCP | "工具协议" | 模型上下文协议——Anthropic 的开放标准，通过服务器暴露工具，任何兼容的客户端都可以发现和调用 |
| Agent 循环 | "ReAct 循环" | 模型决定工具→代码执行工具→结果反馈的迭代循环，直到模型有足够信息回答 |
| 工具投毒 | "通过工具的提示注入" | 一种攻击，工具结果中包含操控模型行为的指令——清理所有工具输出 |
| 速率限制 | "调用预算" | 设置每轮对话的最大工具调用次数，防止无限循环和失控的 API 成本 |

## 拓展阅读

- [OpenAI 函数调用指南](https://platform.openai.com/docs/guides/function-calling)——GPT-4o 工具使用的权威参考，涵盖并行调用、强制调用和结构化参数
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use)——Claude 的工具使用实现，包括 input_schema、多工具响应和 tool_choice 配置
- [模型上下文协议规范](https://modelcontextprotocol.io)——跨 AI 应用的工具互操作性开放标准，包含服务器/客户端架构
- [Schick 等人，2023——"Toolformer：语言模型可以自学使用工具"](https://arxiv.org/abs/2302.04761)——关于训练 LLM 决定何时及如何调用外部工具的基础论文
- [Patil 等人，2023——"Gorilla：连接海量 API 的大语言模型"](https://arxiv.org/abs/2305.15334)——在 1,645 个 API 上微调 LLM 以实现准确 API 调用并减少幻觉
- [Berkeley 函数调用排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html)——跨 GPT-4o、Claude、Gemini 和开源模型比较函数调用准确率的实时基准
- [Yao 等人，"ReAct：在语言模型中协同推理与行动"（ICLR 2023）](https://arxiv.org/abs/2210.03629)——Thought-Action-Observation 循环，即每次工具调用外的 Agent 循环；本课的终点，阶段 14 的起点
- [Anthropic——构建有效的 Agent（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——从单一工具使用原语构建的五种可组合模式（提示链、路由、并行化、编排器-工作者、评估器-优化器）

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这份课件的结构非常巧妙——它把"函数调用"从一个 API 特性提升到了"Agent 的神经系统"这个架构高度。开篇第一句话直接击穿幻觉：LLM 什么都做不了，只能生成文本。这个问题陈述让学生瞬间理解函数调用不是锦上添花的功能，而是 LLM 从"百科全书"变成"Agent"的质变开关。安全部分尤为出色——5 条不可协商的规则，每一条都是生产环境的血泪教训。

### 二、知识结构梳理

**认知基础层**：函数调用循环的五步模型（用户→模型→工具调用→执行→回传→生成）、工具定义就是给模型读的"使用说明书"、与结构化输出的本质区别（中间步骤 vs 最终产品）

**工程模式层**：工具注册表模式（Schema 定义 + 函数实现分离）、参数验证管道（执行前对照 JSON Schema 校验）、错误传播机制（结构化错误 vs 异常抛出的选择）、五种安全规则的实现

**实践应用层**：OpenAI/Anthropic/MCP 三种工具消费模式的 API 差异、并行调用的并发执行优化、工具缓存策略、多步 Agent 循环的迭代控制

### 三、核心洞察（备课时的关键理解）

1. **函数调用不是"让 LLM 执行代码"，而是"让 LLM 表达意图"**。这两者有天壤之别。LLM 从来不在执行——它在说"我要调用这个"。执行永远在代码侧。这个认知一旦建立，安全模型的设计就清晰了。

2. **工具 Schema 的 description 字段本质上是一个微型提示词**。模型不知道怎么用你的工具，除非你告诉它。写 description 的那一刻，你就是在做提示工程——只不过提示的不是最终用户，而是模型本身。

3. **安全规则的第 5 条（速率限制）是最容易被忽视但最要命的**。一个无限循环的工具调用 Agent 可以在几秒钟内耗尽 API 额度。10-20 次的硬上限不是技术限制，是成本控制。

4. **错误处理中的关键设计决策：返回结构化 JSON 而非抛出异常**。模型擅长从结构化错误中自我修正（"城市 Toky 未找到，您是不是想查 Tokyo？"），但面对空响应或 stack trace 完全无能为力。这决定了整个 Agent 的鲁棒性。

5. **并行调用不是优化，是架构必需**。"东京和纽约的天气"如果串行调用就是两轮，并行就是一轮。查询越复杂，这个差距越大——5 次调用就是 5 轮 vs 1 轮。降低 60-80% 延迟不是锦上添花，是用户体验的生死线。

6. **MCP 的定位：不是替代函数调用，而是标准化函数调用的传输层**。就像 HTTP 没有替代网络通信，它标准化了它。理解这一点就不会把 MCP 当成另一个需要"选择"的技术栈。

7. **`simulate_model_decision` 是本课最巧妙的设计**。在不用真实 LLM API 的情况下，用关键词检测模拟模型判断，让学生可以完整跑通整个循环。但教学中必须明确指出这是模拟——生产环境中这一步是模型的推理。

### 四、教学建议

1. **开场做对比实验而不是讲理论**。先让学生用纯 LLM（无工具）问"东京天气"，得到幻觉回答。然后给同一个 LLM 加上天气工具，再问一次。5 分钟的实验胜过 30 分钟的讲解。

2. **画五步循环图，让学生用手指跟着走**。在黑板/屏幕上画用户→应用→模型→工具→回传→用户这个闭环，每讲一个代码文件就让学生指出它在环上的位置。不画图的学生会迷失在代码里。

3. **先让学生设计工具，再写代码**。给一个场景（"你有一个邮件工具和一个日历工具"），让学生先在纸上写出每个工具的 JSON Schema——特别是 description 字段。然后全班互相调用对方的工具，看谁的 description 最清晰。这是本课最容易出错但最值得慢下来的环节。

4. **安全部分不能只讲，要做"攻击演示"**。现场构造一个恶意 SQL 注入字符串作为工具参数，展示如果没有验证会发生什么。再展示如何用白名单和参数验证拦截。安全不是背规则，是看见后果。

5. **错误处理部分做"拔网线"实验**。跑一遍正常查询，然后故意让天气工具返回空值、让文件工具找不到文件，让学生看着 Agent 从报错到自我修正的全过程。重点展示结构化错误消息的设计。

6. **并行调用的演示要可视化时序**。画一个时间轴，先展示串行调用 5 个城市的天气需要多少轮往返，再展示并行只需要一轮。这个可视化比任何文字都有说服力。

7. **最后的练习要有"上瘾"效应**。练习 3（多步 Agent）一旦做出来，学生会有"这东西真的能动"的惊喜感——一个 Agent 自己读配置、搜信息、算结果。这个成就感到达之前不要下课。

### 五、值得补充的内容

1. **流式函数调用的处理**。当模型流式输出时，工具调用的 JSON 是分块到达的——需要累积直到完整才能解析。这在真实 API 中是一个常见坑，本课代码跳过了流式场景。

2. **工具结果的 token 预算管理**。如果一次搜索返回 10KB 结果，全部塞进上下文可能超出窗口。需要实现结果截断策略——按相关性截断、按 token 数截断、或让模型决定需要哪些字段。

3. **工具发现的动态机制**。MCP 部分提到了 `list_tools()`，但没讲当 Agent 有 100+ 个工具时如何做工具选择。一种方案是先用语义搜索筛选相关工具，再交给模型做最终选择——两层过滤。

4. **工具调用的幂等性设计**。如果一个工具调用中途超时，重试是否会重复扣款/发送邮件？不是所有工具都天然幂等——需要设计补偿机制或 at-least-once/at-most-once 语义。

5. **国产大模型的函数调用差异**。通义千问、文心一言、DeepSeek 的 `tools` 参数格式各有细微差异，建议补充一个对比表。

### 六、一句话总结

**函数调用不是让 LLM 变强——是让 LLM 终于可以承认自己不知道，然后说"我需要调这个工具"。模型是大脑，工具是手，函数调用是神经系统——缺了任何一环，Agent 就不存在。**

---


---

# 🎓 Agent 架构课：函数调用——让模型长出双手

同学们好。我是你们的 Agent 架构老师。

上节课我们讲了 RAG——那是给 Agent 装上一本参考书，让它能查到它不知道的事。这节课往前走一大步：我们给 Agent 装上双手，让它能**做**它做不到的事。

今天这节课的核心问题不是"怎么调 API"。API 文档谁都会看。**核心问题是：一个只会生成文本的东西，怎么可能打电话问天气、怎么可能运行代码、怎么可能操作数据库？**

答案是它不操作。它从来就不操作。它只说一句话——"我想调用 get_weather，参数是 Tokyo"——然后你的代码替它动手。这个区别，是一切 Agent 安全的起点。

## 你以为的 Agent vs 真实的 Agent

市面上到处都在说"AI Agent 可以自动订机票、写代码、管理数据库"。听上去像是 LLM 学会了操作电脑。完全没有。

真相是什么？LLM 从头到尾只有一种输出：文本。当它"调用工具"时，它做的事情和写一首诗一模一样——生成一串字符。只不过这串字符碰巧是 JSON，碰巧里面写了 `{"name": "get_weather", "arguments": {"city": "Tokyo"}}`。

你的 Python 代码读到这个 JSON，说："哦，你想调天气 API？行，我帮你调。"然后代码真正发起 HTTP 请求，拿到温度数据，再把结果塞回对话里。

**模型是大脑，工具是手，函数调用是连接它们的神经系统。** 大脑只负责发出动作指令。手负责执行。神经系统负责把指令传过去、把感觉传回来。缺了任何一环，这个人就动不了。

## 函数调用五步循环：Agent 的呼吸

任何 Agent 的工具使用，都是同一个五步循环。你把这五步理解了，剩下就是工程细节。

**第一步：用户说话。** "东京天气怎么样？"

**第二步：应用把消息 + 工具菜单发给模型。** 注意——不是只发消息。你还发了一份"你能用的工具清单"，每件工具带着 JSON Schema：叫什么名字、干什么用、接受什么参数。模型读这份菜单，就像人类读说明书。

**第三步：模型不回答，而是点菜。** 它输出 `get_weather(city="Tokyo")`。不是自然语言，是结构化 JSON。这是整个系统的转折点：模型没有回答问题，它表达了意图。

**第四步：你的代码去"做菜"。** 代码拿到这个调用，执行 `get_weather("Tokyo")`，拿到 `{"temp_c": 18, "condition": "多云"}`。

**第五步：结果返回模型，模型现在有了真数据，回答用户。** "东京现在是 18 摄氏度，多云。"

这个循环是所有 Agent 系统的呼吸节律。一轮不够就再来一轮——模型拿到第一个工具的结果，发现信息还不够，再调下一个工具。这就是 Agent 循环。

## 工具 Schema：写说明书比写代码重要

很多人以为实现函数调用的难点在代码。不在。难点在设计工具 Schema——你写给模型看的"使用说明书"。

看这个对比：

> ❌ `"description": "获取天气"`
> ✅ `"description": "获取指定城市的当前天气。返回摄氏温度值、天气状况、湿度和风速。"`

前者给模型的信息量基本为零。后者告诉模型：这个工具能干什么、返回什么、什么情况下该用它。

**你每写一个工具 Schema，就是在做提示工程。** 只不过你提示的不是最终用户，而是模型。description 的好坏直接决定工具选择准确率。

参数设计同样关键。type 要精确（string vs integer vs number），enum 要完整（celsius 和 fahrenheit 都要列出来），required 要明确。JSON Schema 越严格，模型越不容易乱来。

## 安全：函数调用最危险的地方

上节课 RAG 的安全问题是信息泄露。这节课的安全问题是**代码执行**。

当一个 LLM 可以决定运行什么函数、传什么参数，你就把攻击面从"它可能说错话"扩大到了"它可能执行危险操作"。这是完全不同的安全量级。

**我的五条不可协商的规则：**

**第一条：绝不让模型生成的 SQL 直接进数据库。** 这不是建议，这是生死线。如果你的工具里有一个"执行 SQL"的函数，模型可以并且会生成 `DROP TABLE`。解决办法：不接收 SQL 字符串——接收参数化的查询条件，由你的代码构造安全的 SQL。

**第二条：白名单函数。** 你有 50 个内部函数？暴露给模型的那 5 个就够了。剩下的 45 个，模型根本不应该知道它们的存在。最小暴露原则。

**第三条：验证所有参数。** 模型可能传 `"; DROP TABLE; --"` 作为天气工具的城市名。执行前验证类型、范围、格式。不信任模型传过来的任何东西。

**第四条：清理工具返回结果。** 如果文件读取工具返回了 API 密钥、用户手机号、内部服务器错误——在送给模型之前过滤掉。模型会在回答中原样重复这些内容。工具结果是另一个注入向量。

**第五条：速率限制。** 一个 Agent 陷入循环时，可以在一分钟内调用工具 500 次。设定硬上限——10-20 次调用——超出就终止。这不是功能限制，是成本控制。

## 错误处理：让 Agent 学会说"我搞错了"

工具会失败。天气 API 挂了。城市名拼错了。文件不存在。

你怎么告诉模型"失败了"？有两种选择。

**错误做法：抛异常，返回空内容，或者返回一个通用的"error: true"。** 模型看到空白，就进入猜谜模式——它不知道哪里错了，只能猜。猜出来的结果往往是另一个幻觉。

**正确做法：返回结构化错误。** 像这样：

```json
{
  "error": true,
  "message": "未找到城市 'Toky'。您是不是想查 'Tokyo'？",
  "suggestions": ["Tokyo"],
  "code": "CITY_NOT_FOUND"
}
```

模型读到这个，不只是知道"失败了"——它知道**为什么失败**（拼写错误）、**怎么修正**（Tokyo）、甚至可以直接用修正后的参数重试。

结构化错误消息是 Agent 自我修正的基础设施。没这个，你的 Agent 遇到第一个失败就废了。

## 并行调用：不是优化，是生死

用户问："东京、纽约、伦敦、悉尼、旧金山——这五个城市天气怎么样？"

串行方案：调 Tokyo → 等结果 → 调 New York → 等结果 → ... 五轮往返。

并行方案：五个调用同时发出，五个结果同时回来，一轮往返。

不是 5 秒 vs 1 秒的区别。用户在等。每多一秒，放弃率指数级上升。对于需要 5-10 次工具调用的复杂查询，并行可以将延迟降低 60-80%。

GPT-4o 和 Claude 都原生支持——一个 assistant 消息里可以包含多个 tool_use 块。你的代码并发执行它们，同时返回所有结果。

## OpenAI vs Anthropic vs MCP：三种消费模式

如果你是 OpenAI 用户，工具调用以 `tool_calls` 列表的形式出现在 response 里，每个调用有独立的 `id`。返回结果时必须带上这个 `id`。

如果你是 Anthropic 用户，工具调用是 content 中的 `tool_use` 块。工具结果放在一个 user 消息里，标记为 `tool_result`。另外，Anthropic 用 `input_schema` 定义参数，OpenAI 用 `parameters`——同一个东西，不同字段名。

如果你想让工具跨模型复用——一个天气工具既给 Claude 用也给 GPT-4o 用——那就上 MCP。MCP 标准化了工具暴露的协议层。一个 MCP 服务器可以被任何兼容的客户端发现和调用。你的 Agent 代码不再需要区分"这是 OpenAI 的工具定义格式"还是"这是 Anthropic 的"——MCP 帮你抽象了。

**MCP 对于函数调用，就是 HTTP 对于网络通信。** 它不替代任何东西，它标准化一切。

## 什么时候不用函数调用？

函数调用不是万能药。以下场景它不适用：

1. **用户问的问题模型自己就能回答。** "Python 的列表推导式怎么写"——不需要工具。
2. **你只需要结构化数据提取。** "把这段话里的公司名和金额提取出来"——用结构化输出，不是函数调用。函数调用是"意图"（我要做一件事），结构化输出是"格式"（给我按这个格式输出）。
3. **延迟要求极端严格。** 函数调用至少增加一轮往返（模型→你的代码→工具→你的代码→模型→用户）。如果端到端延迟必须 < 200ms，函数调用不适合。

## 总结：Agent 架构师的函数调用检查清单

1. ✅ **工具设计**：每个工具的 description 是不是在教模型"什么时候用、怎么用"？
2. ✅ **Schema 严格性**：types 精确吗？enum 完整吗？required 明确吗？
3. ✅ **安全性**：SQL 参数化了吗？函数白名单了吗？参数验证了吗？结果清理了吗？
4. ✅ **错误处理**：失败时返回结构化错误消息，还是扔了空值就跑？
5. ✅ **并行支持**：多工具查询能不能并发执行？
6. ✅ **速率限制**：有没有硬上限防止无限循环？
7. ✅ **工具可移植性**：如果需要跨模型复用，上 MCP 了吗？

记住一件事：**函数调用没有让 LLM 变聪明。它终于让 LLM 可以承认自己不知道，然后说出对的那句话——"我需要调这个工具"。这是谦逊的系统设计。最好的 Agent，是那个知道自己不够好、却知道找谁帮忙的 Agent。**

> 🧪 本章代码：[../phase-11-llm-engineering/code/function-calling/main.py](../phase-11-llm-engineering/code/function-calling/main.py) — 完整的 Function Calling 协议：call→execute→observe→reply，兼容 OpenAI/Anthropic/DeepSeek。

---

