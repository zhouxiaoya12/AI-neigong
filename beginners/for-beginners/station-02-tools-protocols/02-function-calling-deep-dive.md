# Function Calling：OpenAI、Anthropic、Gemini 三家怎么"叫函数"

> 预计阅读：25分钟 | 难度：进阶 | 前置条件：理解了 Agent 循环和 MCP 基础
>
> 你将学到：三家的 Function Calling API 各自长什么样、同一个工具怎么写成三种格式、常见的坑在哪。

---

## 🚗 开场翻车：同一个工具写三遍

一个团队给他们的 Agent 写了一个"查天气"工具。先接 OpenAI，20 行代码搞定。然后客户说"我们要用 Anthropic"——两天，重写。又来了一个客户说"用 Gemini"——又一天。

**同一个工具，同一个功能。三套 API，三套代码。** 就因为字段名不同、嵌套不同、参数格式不同。

这不是笑话。这是 2024-2025 年的日常。到 2026 年三家在**概念上**趋同了——但在**形态上**还是各说各话。

---

## 🤔 同一个循环，三套行话

回顾 Agent 循环里的"行动"步骤：

```
Agent 说："我要调用 get_weather 工具，参数是 city=北京"
你的代码执行 get_weather("北京")
返回结果："北京今天 35°C，晴"
```

三家的 API 都支持这个循环。但表达方式完全不同：

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| 工具的"壳"叫 | `tools: [{type: "function", function: {...}}]` | `tools: [{name, description, input_schema}]` | `tools: [{functionDeclarations: [{...}]}]` |
| Schema 字段名 | `parameters` | `input_schema` | `parameters` |
| 返回值在哪 | `message.tool_calls[]` | `content[]` 里的 `tool_use` 块 | `parts[]` 里的 `functionCall` |
| 参数是字符串还是对象 | 字符串（得自己 `JSON.parse`） | 解析好的对象 | 解析好的对象 |
| 结果怎么喂回去 | role `tool` + `tool_call_id` | user 消息带 `tool_result` | `functionResponse` |

**同一个循环。不同的字段名，不同的嵌套，不同的字符串 vs 对象约定。就像三个国家说同一种意思但用不同语言。**

---

## 📖 三家并排看

### OpenAI 写法

```python
# 声明工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {  # 注意：叫 parameters
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

# Agent 返回的工具调用（在 message.tool_calls 里）
# tool_calls[0] = {
#     "id": "call_abc123",
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "arguments": '{"city": "北京"}'  # 注意：是字符串！
#     }
# }

# 你得手动 parse
args = json.loads(tool_call["function"]["arguments"])
```

**OpenAI 最坑的点：参数是 JSON 字符串。** 不在代码里 `json.loads` 一下你拿不到 dict。

### Anthropic 写法

```python
# 声明工具
tools = [{
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "input_schema": {  # 注意：叫 input_schema
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"]
    }
}]

# Agent 返回的工具调用（在 content[] 里作为 tool_use 块）
# content[1] = {
#     "type": "tool_use",
#     "id": "toolu_abc123",
#     "name": "get_weather",
#     "input": {"city": "北京"}  # 已经是解析好的对象！
# }

# 不需要 json.loads，直接用
city = tool_use_block["input"]["city"]
```

**Anthropic 最舒服的点：input 已经是解析好的 dict。** 不用手动 parse。

### Gemini 写法

```python
# 声明工具
tools = [{
    "functionDeclarations": [{  # 注意：多一层嵌套
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {  # 叫 parameters
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }]
}]

# Agent 返回的工具调用（在 parts[] 里）
# parts[0] = {
#     "functionCall": {
#         "name": "get_weather",
#         "args": {"city": "北京"},  # 也是解析好的对象！
#         "id": "uuid-xxxx"  # Gemini 3 以后有唯一 ID
#     }
# }
```

**Gemini 最特别：多一层 `functionDeclarations` 嵌套。**

---

## 🔧 工具数量上限——你会碰到的

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| 每请求最多工具数 | 128 | 64 | 64 |
| Schema 深度上限 | 5 层 | 理论上不限，实际 ≤10 | OpenAPI 3.0 子集 |
| 参数最大长度 | 8192 字节 | 无硬性限制 | 无硬性限制 |

**别在工具描述里写小说。** 每个工具的 description 应该是一两句话——Agent 需要快速判断"这个工具能干嘛"。

---

## ⚡ 强制调用 vs 禁止调用

有时候你要告诉模型"必须用这个工具"或"不准用工具"：

| 模式 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 自动选择 | 默认 | 默认 | 默认 |
| 必须用工具 | `tool_choice: "required"` | `tool_choice: {"type": "any"}` | `mode: "ANY"` |
| 不准用工具 | `tool_choice: "none"` | `tool_choice: {"type": "none"}` | `mode: "NONE"` |
| 强制用某个 | `tool_choice: {"type": "function", "function": {"name": "x"}}` | `tool_choice: {"type": "tool", "name": "x"}` | 不支持 |

---

## 🧠 核心洞察：转换器模式

与其写三套代码，不如定义一套**自己的工具格式**，然后写一个转换器：

```
你定义的工具（统一格式）
    │
    ├── to_openai_format()   → OpenAI API
    ├── to_anthropic_format() → Anthropic API
    └── to_gemini_format()   → Gemini API
```

这就是 LLM 网关的雏形（第五站讲过）。你写一次工具定义，转换器负责把它变成各家要的格式。

---

## 🎮 互动练习

你有一个工具 `send_email(to, subject, body)`。写出它在 OpenAI 格式和 Anthropic 格式下的完整声明。

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **三家概念趋同、形态各异。** 都是"声明工具→模型调用→执行→返回结果"。但字段名、嵌套、参数格式各不相同。

2. **OpenAI 的参数是字符串——记得 `json.loads`。** 这是最容易忘的一步。

3. **Anthropic 的 input 已经是解析好的对象——最省事。**

4. **Gemini 的 tools 里多一层 `functionDeclarations` 嵌套——别丢了。**

5. **别给每个工具写三套代码。写一个转换器。**

---

## 📝 练习答案

**OpenAI 格式：**
```json
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人"},
                "subject": {"type": "string", "description": "主题"},
                "body": {"type": "string", "description": "正文"}
            },
            "required": ["to", "subject", "body"]
        }
    }
}
```

**Anthropic 格式：**
```json
{
    "name": "send_email",
    "description": "发送邮件",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人"},
            "subject": {"type": "string", "description": "主题"},
            "body": {"type": "string", "description": "正文"}
        },
        "required": ["to", "subject", "body"]
    }
}
```

**差异就三个字：** OpenAI 叫 `parameters`、包在 `function` 里面。Anthropic 叫 `input_schema`、扁平结构。

---

> **Function Calling 就是一个翻译问题——你把"查天气"翻译成三家的方言。写一个转换器比写三套代码聪明一百倍。**
