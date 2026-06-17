# 函数调用深入解析 — OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛于相同的工具调用循环，然后又在其他所有方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课程并排对比三家的差异，让你在一个提供商上编写的代码移植到其他提供商时不会出问题。

**类型：** 构建 **语言：** Python（标准库、schema 转换器） **前置课程：** Phase 13 · 01（工具接口） **用时：** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini 函数调用负载的三个形态差异（声明、调用、结果）。
- 将同一个工具声明转换为三种提供商的格式，并预测严格模式约束的差异点。
- 使用每个提供商的 `tool_choice` 来强制调用、禁止调用或自动选择工具。
- 了解每个提供商的硬限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误信号。

## 问题

函数调用请求的形态因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码来强制执行 schema 合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应返回为 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经是解析好的对象（不是字符串）。你用一个包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息来回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应到达为 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，在 Gemini 3 及以上版本中 `id` 是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 来回复。

同一个循环。不同的字段名，不同的嵌套，不同的字符串 vs 对象约定，不同的关联机制。在 OpenAI 上编写天气代理的团队，移植到 Anthropic 需要两天，再移植到 Gemini 又需要一天，光是为了适应这个基础管道。

本课程构建一个转换器，将三种格式统一为一个规范的工具声明，并在边缘进行路由。Phase 13 · 17 将同样的模式推广为一个 LLM 网关。

## 核心概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用特定工具、禁止使用工具或让模型自行决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应与正确的调用关联起来（对并行调用很重要）。
5. **结果注入。** 将结果与调用关联起来的消息或块。

### 字段级形态差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | `content[]` 类型为 `tool_use` | `parts[]` 类型为 `functionCall` |
| 参数类型 | 字符串化 JSON | 解析后的对象 | 解析后的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role `tool`，`tool_call_id` | `user` 带 `tool_result`、`tool_use_id` | `functionResponse` 带匹配的 `id` |
| 强制使用某工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止使用工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 Schema | `strict: true` | schema 即 schema（始终强制执行） | 请求级别的 `responseSchema` |

### 你会实际遇到的上限

- **OpenAI。** 每个请求 128 个工具。Schema 深度 5。参数字符串 ≤ 8192 字节。严格模式要求无 `$ref`，无重叠的 `oneOf`/`anyOf`/`allOf`，每个属性列在 `required` 中。
- **Anthropic。** 每个请求 64 个工具。Schema 深度理论上无限制，实际限制 10。无严格模式标志；schema 是合约，模型倾向于遵守。
- **Gemini。** 每个请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 有轻微差异）。自 Gemini 3 起并行调用使用唯一 ID。

### `tool_choice` 行为

三家都支持三种模式，名称不同。

- **自动。** 模型选择工具或文本。默认值。
- **必需 / 任意。** 模型必须调用至少一个工具。
- **无。** 模型不得调用工具。

加上每个提供商独有的一个模式：

- **OpenAI。** 按名称强制使用特定工具。
- **Anthropic。** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志区分单次调用和多次调用。
- **Gemini。** `mode: "VALIDATED"` 将每个响应通过 schema 验证器路由，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条助手消息中发出多个调用。你运行它们全部，并用包含每个 `tool_call_id` 条目的批量 tool-role 消息来回复。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（Claude 3.5 起的默认值）启用多次调用。Gemini 2 允许并行调用但不提供稳定的 ID；Gemini 3 添加了 UUID，使无序响应能干净地关联。

### 流式传输

三家都支持流式工具调用。线格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你一直累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）使用 `functionCallId` 发出块，使多个并行调用可以交错。

Phase 13 · 03 深入探讨并行 + 流式重组。本课程聚焦于声明和单次调用的形态。

### 错误和修复

无效参数的错误也各有不同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 是不可能的，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 怪异之处：对象字段上的 `enum` 被静默忽略；自己验证。

### 转换器模式

你代码中的规范工具声明如下所示（你选择形态）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种提供商的形态。`code/main.py` 中的脚手架正是这样做的，然后通过每个提供商的响应形态往返一个假工具调用。不需要网络——本课程教授的是形态，而不是 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布一个网关，在三个提供商任何一个前面暴露一个 OpenAI 形态的 API。

## 动手实践

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，分别发出 OpenAI、Anthropic 和 Gemini 声明 JSON。然后它将每种形态的手工构建提供商响应解析为同一个规范调用对象，证明在表象之下的语义是相同的。运行它并对三个声明进行并排对比。

查看要点：

- 三个声明块仅在信封和字段名上有差异。
- 三个响应块在调用所在位置上有差异（顶级的 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形态中提取 `{id, name, args}`。

## 交付成果

本课程产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某提供商的函数调用集成，该技能产出一份可移植性审计：它依赖哪些提供商限制，哪些字段需要重命名，移植到其他每个提供商时什么会出问题。

## 练习

1. 运行 `code/main.py`，验证三个提供商的声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具以添加一个 enum 参数，确认只有 Gemini 转换器需要处理 OpenAPI 的怪异之处。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个功能；注意这种不对称性。

3. 实现 `tool_choice` 转换：将一个规范的 `ToolChoice(mode="force", tool_name="x")` 映射为所有三种提供商的形态。然后映射 `mode="any"` 和 `mode="none"`。对照课程中的差异表进行检查。

4. 选择三家提供商之一，从头到尾阅读其函数调用指南。找出其 schema 规范中其他两家不支持的字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（Lesson 01 中的标准库验证器可以作为代理），记录触发了哪些错误。记录在生产中你会选择哪个提供商的严格性。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | "工具使用" | 用于结构化工具调用发出的提供商级 API |
| 工具声明 | "工具规格" | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必需 / 无 / 特定名称模式 |
| 严格模式 | "Schema 强制执行" | OpenAI 标志，约束解码使其匹配 schema |
| `tool_use` 块 | "Anthropic 的调用形态" | 内联内容块，包含 id、name、input |
| `functionCall` 部分 | "Gemini 的调用形态" | 一个 `parts[]` 条目，包含 name、args 和 id |
| 参数即字符串 | "字符串化 JSON" | OpenAI 将参数作为 JSON 字符串返回，而不是对象 |
| 并行工具调用 | "一轮中的扇形展开" | 一条助手消息中的多个工具调用 |
| 拒绝（Refusal） | "模型拒绝" | 仅严格模式下出现的拒绝块，代替调用 |
| OpenAPI 3.0 子集 | "Gemini Schema 怪异之处" | Gemini 使用类 JSON Schema 方言，有细微差异 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包括严格模式和并行调用的权威参考
- [Anthropic — 工具使用概览](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业界面
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行细节

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一份高质量的三提供商函数调用对比文档。它用简洁的表格和代码示例将 OpenAI、Anthropic 和 Gemini 的函数调用形态差异讲得非常清楚，是本阶段最实用的课程之一。核心价值在于"转换器模式"——通过一个规范的工具声明来屏蔽三家差异，这个思路直接通向 Phase 13 · 17 的 LLM 网关设计。

### 二、知识结构梳理

**认知基础层：** 三家提供商函数调用的共性——工具列表、工具选择、调用发出、调用 ID、结果注入五个环节。这五个环节是所有 LLM 工具调用系统的基础认知框架。

**工程模式层：** 转换器模式（canonical representation → provider-specific translators）是核心工程抽象。字段级对比表揭示了字符串 vs 对象、信封嵌套、ID 格式等具体差异。并行调用、流式传输、错误处理是工程师在实际开发中必须面对的细节。

**落地实践层：** 严格模式约束（OpenAI 的 `strict: true`）、工具数量限制（128 vs 64）、Schema 深度限制（5 vs 无界）是生产环境中不可忽视的硬约束。代码示例展示的 `code/main.py` 给出了可运行的参考实现。

### 三、核心洞察

1. **"同一个循环，不同形态"是核心命题。** 为什么重要：理解这个命题后，后续所有"跨提供商"的设计都围绕如何屏蔽形态差异展开，这是 LLM 网关类产品的基础认知。

2. **参数是字符串还是对象是最大的坑。** 为什么重要：OpenAI 返回 JSON 字符串需要额外解析步骤，而 Anthropic 和 Gemini 返回已解析对象。这个差异在并行调用场景下会放大——你需要管理多个 `tool_call_id` 到解析后参数的路由。

3. **严格模式在两个维度的表现不同。** 为什么重要：OpenAI 通过受限解码在生成阶段保证 schema 合规，Anthropic 依赖模型自身遵守，Gemini 使用 OpenAPI 3.0 子集。选择严格程度直接影响生产系统的可靠性和错误处理策略。

4. **并行调用的 ID 关联机制是架构决策。** 为什么重要：Gemini 2 没有稳定 ID 意味着你无法支持并行调用；Gemini 3 加入 UUID 解决了这个问题。这个演变说明并行调用 ID 是协议设计的关键组成部分。

5. **转换器模式是跨提供商工具调用的唯一正确抽象。** 为什么重要：直接依赖某个提供商的 API 形态会导致移植成本极高。转换器模式将差异限制在边缘，核心业务逻辑只需处理规范形态。这是 Phase 13 · 17 网关设计的基础。

6. **工具选择行为（tool_choice）的细微差异可能导致静默故障。** 为什么重要：三家对"强制使用某工具"的实现不同——OpenAI 通过 `tool_choice.type` 和 `function.name`，Gemini 通过 `function_calling_config.mode: "VALIDATED"`。跨提供商移植时工具选择逻辑可能静默失效。

### 四、教学建议

1. **先让学生自己尝试跨提供商工具调用，再讲转换器模式。** 让学生先用三家 API 分别调用同一个"天气查询"工具，亲身感受字段名和嵌套的差异，然后再引入转换器模式作为统一解决方案。

2. **用对比表作为教学的核心辅助材料。** 文档中的字段级差异表非常精炼，建议打印出来发给学生或做成墙贴，作为日常开发的快速参考。

3. **让学生实现完整的转换器作为编码练习。** 不仅是声明和调用，还包括错误处理、流式传输、并行调用三条路径。完整的转换器是 Phase 13 · 17 网关的预习。

4. **用真实的生产案例说明差异的重要性。** 如文档所述，"一个团队在 OpenAI 上编写天气代理，移植到 Anthropic 需要两天"。这种真实痛点比抽象对比更有说服力。

5. **在教学中强调 "不是所有差异都需要消除"。** 转换器模式的目的是提供统一接口，但不是抹平所有差异——严格模式的行为差异、Schema 深度限制等仍然需要开发者在选择提供商时做出权衡。

6. **将本课程与 Phase 13 · 17 的网关设计串联。** 本课程的转换器模式是网关设计的预热。建议在教学中明确指出：你现在学的三个小转换函数，就是未来网关中的 provider adapter。

### 五、值得补充的内容

1. **成本对比。** 三家提供商的工具调用定价不同（按令牌计费的差异），尤其是在严格模式和非严格模式下的成本可能有显著差异，建议补充成本对比数据。

2. **Tool Use 与 Function Calling 的历史演进。** 补充 2023-2025 年间三家提供商工具调用功能的时间线（谁先推出、各自迭代了哪些版本），有助于理解为什么存在这些差异。

3. **A2A（Agent-to-Agent）协议对工具调用的影响。** Google 的 Agent-to-Agent 协议和 MCP 之间的互补关系，以及它们如何影响前端工具调用的设计决策。

4. **实际基准测试数据。** 补充三提供商在同类型工具调用上的延迟和准确性基准测试（如 ToolBench 或 BFCL 上的分数），帮助学生在选择提供商时做出数据驱动的决策。

### 六、一句话总结

函数调用的本质是同一个工具接口在不同提供商 API 形态下的投影——转换器模式是最小且最正确的跨提供商抽象。


---

# 🎓 Agent 架构课：三提供商函数调用的形态战争——"同一个厨房，三本菜谱"

同学们好。我是你们的FDE工程老师，今天讲的是 函数调用。

今天这节课的标题可能会让你觉得我在讲一个简单问题。但我跟你说一句实话——**函数调用是 Agent 工程里看起来最简单、实际上翻车率最高的模块。**

你去看 GitHub 上的 Agent 项目，80% 的 `tool_calls` 解析代码里藏着至少一个提供商移植 bug。不是因为写代码的人不行——是因为 OpenAI、Anthropic、Gemini 三家把"同一个功能"做成了三个看起来一样但细节上分道扬镳的接口。人类大脑看到相似的结构会自动打上"哦一样的"标签，然后写一段共用代码。

然后上线后你发现在细节上悄无声息地坏了。

所以这节课的核心就一句话，我希望你从一开始就刻在脑子里：**函数调用的本质不是"让 LLM 调用工具"——是"在三个方言不同的厨子之间当一个翻译官"。同一个菜名，一个要爆炒，一个要清蒸，一个从冰箱里拿出生肉却不告诉你要先解冻。**

## 三本菜谱：同一个循环，三种写法

我先给你看一个对比。你用同一个工具——查天气——分别写给 OpenAI、Anthropic、Gemini。

**OpenAI 怎么说：**
```python
# 工具声明
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "查询城市天气",
    "parameters": {"type": "object", "properties": {...}},
    "strict": True  # OpenAI 独有：受限解码强制 schema 合规
}}]

# 响应解析
call = response.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)  # 注意：arguments 是 JSON 字符串！
```

**Anthropic 怎么说：**
```python
# 工具声明
tools = [{"name": "get_weather", "description": "查询城市天气", "input_schema": {...}}]

# 响应解析——完全不同的路径
for block in response.content:
    if block.type == "tool_use":
        args = block.input  # 已经是解析好的对象，别 json.loads！
```

**Gemini 怎么说：**
```python
# 工具声明——注意套娃结构
tools = [{"functionDeclarations": [{
    "name": "get_weather", "description": "查询城市天气", "parameters": {...}
}]}]

# 响应解析——又一条路径
for part in response.candidates[0].content.parts:
    if hasattr(part, 'functionCall'):
        args = part.functionCall.args  # 又是对象，不是字符串
```

三个实现。同一个功能。六处不同——不仅是字段名，还有嵌套深度、字符串 vs 对象、ID 关联机制。

如果你写三个 if-else 分支分别处理，你就在代码里复制了三份"什么参数才对"的知识。任何一次修改——加个新字段、改个格式——你要在三处改三遍，而且总有某个环境你忘了测。

## 转换器模式：一个翻译官，三本词典

解决这个问题的核心哲学就一句话：**不要把差异带到核心逻辑里，在边缘把它翻译掉。**

```
你的核心 Agent 代码
        │
        │ 只说一种语言：CanonicalTool, CanonicalCall
        │
   ┌────┼────┐
   │    │    │
   ▼    ▼    ▼
OpenAI  Anth  Gemini
适配器  适配器  适配器
```

你定义一套你自己的规范形态：

```python
@dataclass
class CanonicalTool:
    name: str
    description: str
    input_schema: dict
    strict: bool = False

@dataclass
class CanonicalCall:
    id: str
    name: str
    args: dict
```

然后写三个小函数，每个只做一件事：

```python
def to_openai_tool(t: CanonicalTool) -> dict:
    return {"type": "function", "function": {
        "name": t.name, "description": t.description,
        "parameters": t.input_schema, "strict": t.strict
    }}

def to_anthropic_tool(t: CanonicalTool) -> dict:
    return {"name": t.name, "description": t.description, "input_schema": t.input_schema}

def to_gemini_tool(t: CanonicalTool) -> dict:
    return {"functionDeclarations": [{
        "name": t.name, "description": t.description, "parameters": t.input_schema
    }]}
```

反过来也一样——每个提供商的响应进来，被提取成你的 `CanonicalCall`。你的核心 Agent 逻辑不知道自己在跟谁说话，它只看到"我调用了一个工具，得到了这些参数，返回了这个结果"。

新增第四个提供商？加一个翻译函数，核心逻辑一行不动。这就是 Phase 13 · 17 LLM 网关的雏形。

## 三个坑，踩一个就是三天

**坑一：字符串 vs 对象。** OpenAI 的 `arguments` 是 JSON 字符串——你得 `json.loads()`。Anthropic 的 `input` 是解析好的对象——你加 `json.loads()` 就炸了。Gemini 的 `args` 也是对象。这个区别不会在文档第一页告诉你——它藏在你第一次移植代码时的报错堆栈里。

**坑二：缺失的 ID。** Gemini 2.x 在并行调用时不返回 `id`。如果你用 `{name, args}` 作为 key 去匹配响应，两个相同工具调用会互相覆盖。解决方案：在你的规范形态里强制生成一个 UUID，不要依赖提供商的 ID。

**坑三：严格模式的语义分裂。** OpenAI 有 `strict: true`——受限解码保证输出符合 schema。Anthropic 没有这个参数——模型的工具调用已经足够结构化。Gemini 有 `automaticFunctionCalling` 的变体。你如果在一个提供商上用惯了一个参数，移植到另一个时静默失效——编译不报错、lint 不报错、单元测试可能也不报错，直到生产环境用户说"为什么今天的响应格式不对"。

## 为什么不写一个巨型 if-else

我知道你现在在想什么："三个适配器不如一个 if-else 直观。"

我给你算一笔账。你有一个工具库，里面有 20 个工具。你要在 OpenAI、Anthropic、Gemini 三个提供商上都能用。

如果你写 if-else：每个工具声明写在三个分支里 = 60 个声明片段。每次新增一个工具——三个分支各加一份。每次改一个参数——三份各改一遍。一年下来，你的函数调用模块里 40% 的代码是重复的，20% 的重复代码之间已经有细微的不一致。

如果你用转换器：20 个规范声明 + 3 个翻译函数 = 23 个单元。新增工具——只加一份规范声明。改参数——只改一处。三个翻译函数每个不超过 30 行，写完就不会再动了。

这不是设计模式崇拜——是工程经济学。

## 结课清单

如果你今天只带走一件事：**函数调用的形态差异不在文档首页，在你第一次移植代码时的报错堆栈里。不要在三个 if-else 分支里复制知识——在边缘翻译掉。**

如果你今天能带走一个完整的认知模型，这是你的清单：

1. **三个提供商的差异是结构性的。** OpenAI 用 `tool_calls` + 字符串、Anthropic 用 `tool_use` + 对象、Gemini 用 `functionDeclarations` 套娃。字段名、嵌套、类型约定全不同。
2. **转换器模式是最小且最正确的抽象。** 规范形态 + 边缘适配器。核心逻辑不说方言。
3. **字符串 vs 对象是最常见的移植 bug。** 写一个 `_parse_args()` 函数，输入是提供商标识 + 原始响应，输出永远是 dict。
4. **不要依赖提供商的 ID。** 在规范形态里强制生成 UUID。
5. **严格模式不是通用概念。** 不要把一个提供商的语义假设带到另一个。
6. **三个 30 行适配器 > 一个 300 行 if-else。** 如果你的适配器代码超过 30 行，你在适配器里写了核心逻辑——那是错的。

最后一句话——我希望你明天还记得：

**函数调用是 Agent 的嘴巴。嘴形不对，再好的脑子也说不出正确的话。花两天写好一个转换器，省下的是未来每次接新提供商时两天的改代码和上线后一周的 debug。**

---

# 💼 从业者故事：三提供商函数调用的形态战争——"同一个厨房，三本菜谱"

凌晨两点半，我盯着 Slack 里的告警发呆。生产环境上跑了一个月的天气 Agent，今天移植到 Anthropic 之后开始随机返回"今天晴天 25 度"给查股票的用户。为什么？因为 Anthropic 的 `tool_use` 块藏在 `content` 数组里，而我们那个写死 `choices[0].message.tool_calls` 的解析器，吃完 Anthropic 的响应后优雅地吐出了一行 `None`——然后模型就开始靠幻觉续命。

这个错误我犯过三次。第一次是 OpenAI 到 Anthropic 的移植，第二次是加 Gemini 支持的时候，第三次是新人 code review 我看到了同样的代码却没拦住。函数调用最大的坑不是你不会写，而是你写的代码在**你以为一样的接口上悄无声息地坏了**。

## 问题不是"不同"，是"看起来一样"

如果你把 OpenAI、Anthropic、Gemini 的函数调用文档并排摆出来，你会觉得他们开过闭门会议商量好了。都有工具声明，都有 `tool_choice`，都有并行调用，都返回一个带 ID 的调用对象。人类大脑看到这个就会自动打上"哦，一样的"标签，然后写一段共用代码。

然后上线后你会发现在是细节上分道扬镳。OpenAI 的参数是 JSON 字符串——你得 `json.loads()` 一下。Anthropic 给你解析好的对象——你加 `json.loads()` 就炸了。Gemini 把工具声明套了三层信封——`functionDeclarations` 里包 `functionDeclarations`，命名跟套娃一样魔幻。这三个 API 就像三个方言不同的厨子：你说"炒个菜"，一个以为你要爆炒，一个以为你要清蒸，一个从冰箱里拿出生肉却不告诉你得先解冻。

传统方案是什么？写三个 if-else 分支，每个分支一套解析逻辑。但这意味着你的工具调用代码里有三份真相——那份"什么参数才对"的知识被复制了三遍。任何一次修改——加个新字段、改个格式——你要在三处改三遍，而且总有某个环境你忘了测。

## 转换器模式：一个翻译官，三本词典

解决这个问题的核心哲学是：**不要把差异带到核心逻辑里，在边缘把它翻译掉。**

你定义一套你自己的规范形态——一个 `Tool` 数据类，里面有 `name`、`description`、`input_schema`、`strict`。这个形态只属于你，不属于任何提供商。然后你写三个小函数，每个函数只做一件事：把你的规范形态翻译成对应提供商的 JSON。反过来也一样——每个提供商的响应进来，被提取成你的规范调用对象 `{id, name, args}`。

```python
# 你的代码里只有这一种形态
canonical_call = CanonicalCall(id="call_1", name="get_weather", args={"city": "Beijing"})

# 边缘翻译成三种方言
openai_call = to_openai_tool_call(canonical_call)    # arguments 是字符串
anthropic_call = to_anthropic_tool_use(canonical_call)  # input 是对象
gemini_call = to_gemini_function_call(canonical_call)  # 套在 functionDeclarations 里
```

这个模式说白了就是**给三种方言各自配了个翻译**——你自己只说你自己的语言，翻译负责适配。你的核心 Agent 逻辑不知道自己在跟谁说话，它只看到"我调用了一个工具，得到了这些参数，返回了这个结果"。新增第四个提供商？加个翻译函数，核心逻辑一行不动。

这就是 Phase 13 · 17 LLM 网关的雏形——一个转换器是网关节点的骨架。

## 字符串 vs 对象：两行代码让你加班到天亮

我见过最惨的一次事故：OpenAI 移植到 Anthropic，解析器里那行 `json.loads(arguments)` 没删。Anthropic 已经把 `input` 给你解析成 Python dict 了，你再 `json.loads()` 一下——`TypeError: the JSON object must be str, bytes or bytearray, not dict`。但这个错误被一个宽泛的 try-except 吞了，于是 Agent 带着空的 `args={}` 去调用了 `get_weather`，拿回了"城市未指定"的错误消息，然后模型一本正经地编了一个上海的天气给用户。

P99 调用延迟在这种场景里会偷偷膨胀。5 轮工具调用循环，每轮多一次错误重试，串行下来 P99 从 2 秒飙到 8 秒。用户觉得你的 Agent 变慢了，但你查监控发现工具调用成功率是 100%——因为错误被优雅地消化了，重试也被隐式处理了。

**教训：参数类型的差异是你的转换器函数最需要防呆的地方。** 在 `from_openai_response()` 函数里加 `json.loads()`，在 `from_anthropic_response()` 里直接拿 `input`。如果你用 Python，在每个转换函数的签名里用类型注解标注清楚：

```python
def from_openai(arguments: str) -> dict  # 老子帮你解析
def from_anthropic(input: dict) -> dict  # 直接原样返回
```

同事 code review 看到这个，大大概会说——"你居然没写个 assert isinstance？"——然后你默默加上。

## 严格模式：OpenAI 的紧箍咒和另外两家的薛定谔合约

OpenAI 的 `strict: true` 是个好东西，但也是个陷阱。它通过受限解码（constrained decoding）在生成阶段就把 schema 违规给掐死了——模型生成的 token 序列必须合法地通过你的 JSON Schema。这意味着你永远不会收到 `{"city": null, "units": "kelvin"}` 如果 `city` 是 required 且 `units` 只有 `["celsius", "fahrenheit"]`。

但你得先让你的 schema 配得上这个紧箍咒。`required` 漏写一个字段，模型就敢不传那个参数然后瞎编回应。Schema 深度超过 5 层？OpenAI 直接拒收。用了 `$ref` 引用？不兼容。`anyOf`/`oneOf`/`allOf` 重叠？不好意思，严格模式拒绝接受。

Anthropic 和 Gemini 没有严格模式这个概念——它们的态度是"schema 是建议，不是合同"。模型大概率会遵守，但你得在自己这边加一层验证。我在生产里见过 Anthropic 的模型在一个 `{"type": "string", "enum": ["celsius", "fahrenheit"]}` 参数上传了 `"kelvin"`——因为用户的输入里出现了开尔文，模型就自作主张了。

**所以选择哪家提供商的"严格"策略，不是一个技术问题，是一个可靠性哲学问题。** 你要在"生成阶段保证合规"和"运行时验证容错"之间做选择。我个人在金融和医疗类 Agent 上只用 OpenAI 严格模式，在内容生成类 Agent 上用 Anthropic 加自己的验证层。

## 并行调用的 ID：Gemini 2 的耻辱和 Gemini 3 的救赎

如果你在 2025 年用过 Gemini 2 做并行工具调用，你就知道什么叫"我知道他拿了我的快递但我不知道哪个快递"。一个 `get_weather` 和一个 `get_stock_price` 同时发出，两个响应回来——但你没有稳定的 ID 来把响应跟请求对上。两个响应都是"{name: 'get_weather', result: 25°C}"和"{name: 'get_stock_price', result: 'AAPL: $198'}"——如果两个都调了 `get_weather` 查不同城市，你就彻底不知道谁是谁了。

这就是为什么并行调用的 ID 字段不是"可选"的，是**协议必需的**。OpenAI 的 `call_xxx` 和 Anthropic 的 `toolu_xxx` 都从一开始就设计了这个 ID。Gemini 3 终于补上了 UUID——但你去读 Gemini 2 的历史代码，说不定还能看到有人用"调用顺序索引 + 结果名称匹配"这种黑魔法来兜底。

如果你现在正在设计一个函数调用网关，**在规范形态里强制要求 ID**。即使某个提供商不带 ID（也许将来的新厂商又犯同样的错误），你的边缘适配器也得生成一个唯一 ID 并注入进去。

## 生产现实：你以为的成本和实际成本

说几个我在生产里踩过的具体数字：

- **工具声明大小**：OpenAI 严格模式单工具 `input_schema` 展开后约 2-5KB。128 个工具就是 250-640KB 的 context tokens，意味着每次请求你都在为工具声明花掉 ~60K-160K 输入 tokens 的成本。如果你的 Agent 每轮对话都重新传全部工具列表，一个月光工具声明就能烧掉几千美金。
- **JSON 解析的 CPU 开销**：OpenAI 返回 8KB 的 `arguments` 字符串，`json.loads()` 要 2-5ms。在高并发场景下（100 QPS），光解析参数就能吃掉一个 CPU 核。Anthropic 返回已解析对象省掉了这笔开销，但你省下来的钱可能被它更高的人均 token 价格吞掉。
- **错误重试放大**：一次工具调用失败如果触发 2-3 次重试，串行 P50 延迟从 800ms 变成 2 秒。如果你在用户面前显示"思考中"，这个时间差会被敏感地感知。

## 什么时候不该用这个模式

转换器模式有个隐含假设：**三家提供商的行为在语义上是等价的。** 但这个假设不总是成立。

- 如果你的 Agent 深度依赖 OpenAI 的严格模式来做 schema 验证，移植到 Anthropic 会在"参数合法性"这个维度上降级——你得多写一整套验证逻辑。
- 如果你的工作流依赖并行调用 ID 做精准路由，Gemini 2 直接不支持，你的转换器得对这个限制做显式的版本门控。
- 如果你的工具列表超过 64 个，OpenAI 还能跑，但 Anthropic 和 Gemini 会拒收——转换器不能帮你绕过提供商的硬限制。

**转换器消除的是形态差异，不是能力差异。** 你的网关层应该对提供商的"能力矩阵"做 explicit mapping——哪些功能原生的、哪些要降级、哪些直接不可用。

## 收尾

函数调用是 Agent 的嘴巴。嘴形不对，再好的脑子也说不出正确的话。花两天写好一个转换器，省下的是未来每次接新提供商时两天的改代码和上线后一周的 debug。

**写到这想起来一句：函数调用的本质是把 LLM 的"我想要"翻译成系统的"我去做"——翻译的质量不取决于 LLM 有多聪明，取决于你给的词典有没有漏掉关键的那个词。**
