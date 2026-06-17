# 结构化输出：JSON、模式验证、约束解码

> 你的 LLM 返回一个字符串。你的应用需要 JSON。这个差距导致的生产系统崩溃比任何模型幻觉都多。结构化输出是自然语言与类型化数据之间的桥梁。做对了，你的 LLM 就变成了一个可靠的 API。做错了，你就得在凌晨三点用正则表达式解析自由文本。

**类型：** Build
**语言：** Python
**前置要求：** 第 10 阶段，第 01-05 课（从零开始构建 LLMs）
**预计时间：** ~90 分钟
**相关课程：** 第 5 阶段 · 20（结构化输出与约束解码）涵盖解码器层面的理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦于生产级 SDK 层面（OpenAI `response_format`、Anthropic tool use、Instructor）——如果你想理解 API 之下的原理，请先阅读第 5 阶段 · 20。

---

## 学习目标

- 使用 OpenAI 和 Anthropic 的 API 参数实现 JSON 模式和模式约束输出
- 构建一个 Pydantic 验证层，拒绝格式错误的 LLM 输出，并通过错误反馈进行重试
- 解释约束解码如何在令牌级别强制生成有效 JSON，而无需后处理
- 设计稳健的提取提示，可靠地将非结构化文本转换为类型化数据结构

---

## 问题所在

你让 LLM："从这段文本中提取产品名称、价格和库存状态。"它回复：

```
该产品是 Sony WH-1000XM5 耳机，售价 $348.00，目前有库存。
```

这是一个完全正确的答案。但对你的应用来说完全没用。你的库存系统需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个具有特定键、特定类型和特定值约束的 JSON 对象。你不需要一个句子。

天真的解决方案：在提示中加上"以 JSON 格式回复"。这在 90% 的情况下有效。另外 10% 的情况，模型把 JSON 包在 markdown 代码围栏里，或者加上"以下是 JSON："这样的前言，或者因为提前关闭了大括号而生成语法无效的 JSON。你的 JSON 解析器崩溃了，你的流水线断了。你加上 try/except 和重试循环。重试有时会产生不同的数据。现在你在解析问题之上又有了一个一致性问题。

这不是提示工程问题。这是解码问题。模型从左到右一个令牌一个令牌地生成。在每个位置，它从 10 万+ 选项的词汇表中选择最可能的下一个令牌。在任意给定位置，这些选项中大多数都会生成无效的 JSON。如果模型刚刚输出了 `{"price":`，下一个令牌必须是数字、引号（表示字符串）、`null`、`true`、`false` 或负号。任何其他内容都会产生无效的 JSON。没有约束的情况下，模型可能会选择一个完全合理的英文单词，但语法上是灾难性的错误。

---

## 概念

### 结构化输出的四个层级

结构化输出控制有四个层级，每一层都比上一层更可靠。

```
提示级（"返回 JSON"）  →  ~90% 有效
        ↓
JSON 模式（保证有效 JSON，不保证模式）  →  100% 可解析
        ↓
模式模式（JSON + 匹配模式，保证合规）  →  100% 模式合规
        ↓
约束解码（令牌级强制，100% 合规）  →  100% 令牌级合规
```

**提示级**（"以有效 JSON 回复"）：无强制执行。模型通常会遵循，但有时不遵循。可靠性：~90%。失败模式：markdown 围栏、前言文本、截断输出、错误结构。

**JSON 模式**：API 保证输出是有效的 JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用此功能。输出解析不会出错。但可能不符合你期望的模式——额外的键、错误的类型、缺少字段。

**模式模式**：API 接收一个 JSON Schema，保证输出符合它。到 2026 年，每个主要提供商都原生支持此功能：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可以通过 `tool_choice="required"` 实现），Anthropic 的带 `input_schema` 的工具使用，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出有你指定的精确键、类型和约束。

**约束解码**：在生成过程中的每个令牌位置，解码器屏蔽所有会产生无效输出的令牌。如果模式要求一个数字而模型即将输出一个字母，该令牌的概率被设为零。模型只能产生通向有效输出的令牌。这就是 OpenAI 的结构化输出模式和 Outlines、Guidance 等库在底层实现的东西。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或验证层）输出必须具有什么形状的方式。每个主要的结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个模式表示：输出必须是一个对象，包含字符串 `product`、非负数 `price`、布尔值 `in_stock`，以及可选的字符串数组 `categories`。任何不匹配的输出都会被拒绝。

模式能处理困难的情况：嵌套对象、带类型项的数组、枚举（将字符串约束到特定值）、模式匹配（字符串的正则表达式），以及组合器（`oneOf`、`anyOf`、`allOf` 用于多态输出）。

### Pydantic 模式

在 Python 中，你不必手写 JSON Schema。定义一个 Pydantic 模型，它会为你生成模式。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor 库（以及 OpenAI 的 SDK）直接接受 Pydantic 模型：传入模型类，拿回一个已验证的实例。如果 LLM 输出不匹配，Instructor 会自动重试。

### 函数调用 / 工具使用

同一个问题的另一种接口。你不是要求模型直接输出 JSON，而是定义带有类型化参数的"工具"（函数）。模型输出一个带有结构化参数的函数调用。OpenAI 称之为"函数调用"，Anthropic 称之为"工具使用"。结果是相同的：结构化数据。

```
用户: 从这篇评论文本中提取产品信息
  → 模型处理输入
    → 工具调用: extract_product(product='Sony WH-1000XM5', price=348.00, in_stock=true)
      → 根据函数模式验证
        → 结构化结果: {product, price, in_stock}
```

当模型需要选择调用哪个函数，而不仅仅是填充参数时，工具使用是首选。如果你有 10 个不同的提取模式，模型必须根据输入选择正确的那个，工具使用给你模式选择和结构化输出两者。

### 常见失败模式

即使有模式强制，结构化输出也可能以微妙的方式失败。

**幻觉值（Hallucinated values）**：输出匹配模式但包含虚构数据。文本说 $348，模型却输出 `{"price": 299.99}`。模式验证无法捕捉到这一点——类型正确，值错误。

**枚举混淆（Enum confusion）**：你将字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义上正确，但不在允许的集合中。好的约束解码能防止这一点，基于提示的方法则不能。

**嵌套对象深度（Nested object depth）**：深度嵌套的模式（4 层以上）产生更多错误。每一层嵌套都是模型可能丢失结构追踪的另一个位置。

**数组长度（Array length）**：模型可能在数组中产生过多或过少的项。模式支持 `minItems` 和 `maxItems`，但并非所有提供商都在解码级别强制执行它们。

**可选字段遗漏（Optional field omission）**：模型省略了在技术上可选但在语义上对你的用例很重要的字段。在模式中将它们设为必需，即使数据有时缺失——强制模型显式输出 `null`。

---

## 构建

### 第 1 步：JSON Schema 验证器

从头构建一个验证器，检查 Python 对象是否匹配 JSON Schema。这是在输出端运行以验证合规性的东西。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 第 2 步：Pydantic 风格的模型到模式转换器

构建一个最小化的类到模式转换器。定义一个 Python 类并自动生成其 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 第 3 步：约束令牌过滤器

模拟约束解码。给定部分 JSON 字符串和一个模式，确定当前位置哪些令牌类别是有效的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 第 4 步：提取流水线

将所有内容组合成一个提取流水线：定义一个模式，模拟 LLM 产生结构化输出，验证输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 第 5 步：运行完整流水线

```python
def run_demo():
    print("=" * 60)
    print(" 结构化输出流水线演示")
    print("=" * 60)

    print("\n--- 模式定义 ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- 模式验证 ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "有效对象"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "负数价格"),
        ({"product": "Test", "in_stock": True}, "缺少价格"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "字符串作为价格"),
        ("not an object", "字符串而非对象"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "通过" if not errors else f"失败: {errors}"
        print(f"  {label}: {status}")

    print("\n--- 约束解码模拟 ---")
    demonstrate_constrained_decoding()

    print("\n--- 提取流水线 ---")
    texts = [
        "Sony WH-1000XM5 耳机售价 $348，目前有货。",
        "新款 MacBook Pro 16 英寸笔记本售价 $2499，但已售罄。",
        "这是一句与产品无关的随机句子。",
    ]

    for text in texts:
        print(f"\n  输入: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  输出: {json.dumps(result)}")
        else:
            print(f"  输出: 重试后失败")
```

---

## 使用

### OpenAI 结构化输出

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "提取产品信息。"},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, 有库存"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式内部使用约束解码。模型生成的每一个令牌都被保证产生匹配 Pydantic 模式的输出。无需重试，无需验证。约束被嵌入到解码过程中。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "从文本中提取产品信息",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "提取: Sony WH-1000XM5, $348, 有库存"}],
# )
```

Anthropic 通过工具使用实现结构化输出。模型发出一个带有结构化参数的工具调用，参数匹配 `input_schema`。相同的结果，不同的 API 界面。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, 有库存"}],
# )
```

Instructor 包装任何 LLM 客户端并添加带验证的自动重试。如果第一次尝试验证失败，它将错误作为上下文发送回模型，并要求模型修复输出。这适用于任何提供商，不仅仅是 OpenAI。

---

## 交付物

本课程产出 `outputs/prompt-structured-extractor.md`——一个可复用的提示模板，根据模式定义从任意文本中提取结构化数据。给它一个 JSON Schema 和非结构化文本，它会返回已验证的 JSON。

还产出 `outputs/skill-structured-outputs.md`——一个决策框架，用于根据你的提供商、可靠性要求和模式复杂度选择合适的结构化输出策略。

---

## 练习

1. 扩展模式验证器以支持 `oneOf`（数据必须匹配多个模式之一）。这能处理多态输出——例如，一个字段可以是 `Product` 或 `Service` 对象，具有不同的形状。

2. 构建一个"模式差异"工具，比较两个模式并识别破坏性变更（移除必需字段、更改类型）与非破坏性变更（添加可选字段、放宽约束）。这对于在生产环境中对你的提取模式进行版本管理至关重要。

3. 实现一个更逼真的约束解码模拟器。给定一个 JSON Schema 和一个包含 100 个令牌的词汇表（字母、数字、标点、关键词），逐步执行生成过程，在每个位置屏蔽无效令牌。测量每一步中词汇表中有多大比例是有效的。

4. 构建一个提取评估套件。创建 50 条产品描述及其手工标注的 JSON 输出。在所有 50 条上运行你的提取流水线，测量精确匹配、字段级准确率和类型合规性。识别哪些字段最难正确提取。

5. 为你的提取流水线添加"置信度分数"。对每个提取的字段，估计模型的置信度（基于令牌概率，或通过运行 3 次提取并测量一致性）。标记低置信度字段供人工审核。

---

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| JSON 模式 | "返回 JSON" | API 标志，保证语法有效的 JSON 输出，但不强制任何特定模式 |
| 结构化输出 | "类型化 JSON" | 匹配具有正确键、类型和约束的特定 JSON Schema 的输出 |
| 约束解码 | "引导生成" | 在每个令牌位置，屏蔽会产生无效输出的令牌——保证 100% 模式合规 |
| JSON Schema | "一个 JSON 模板" | 一种声明性语言，用于描述 JSON 数据的结构、类型和约束（被 OpenAPI、JSON Forms 等使用） |
| Pydantic | "Python 数据类+" | Python 库，定义带有类型验证的数据模型，被 FastAPI 和 Instructor 用于生成 JSON Schema |
| 函数调用 | "工具使用" | LLM 输出结构化的函数调用（名称 + 类型化参数）而非自由文本——OpenAI 和 Anthropic 都支持此功能 |
| Instructor | "LLM 的 Pydantic" | Python 库，包装 LLM 客户端以返回已验证的 Pydantic 实例，验证失败时自动重试 |
| 令牌屏蔽 | "过滤词汇表" | 在生成过程中将特定令牌的概率设为零，使模型无法生成它们 |
| 模式合规 | "匹配形状" | 输出具有每个必需字段、正确类型、约束内的值，且没有额外的非法字段 |
| 重试循环 | "一直重试直到成功" | 将验证错误发送回模型并要求它修复输出——Instructor 自动执行此操作，最多可达可配置的最大次数 |

---

## 进一步阅读

- [OpenAI 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs)——OpenAI API 中基于 JSON Schema 的约束解码的官方文档
- [Willard & Louf, 2023——"Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702)——Outlines 论文，描述如何将 JSON Schema 编译为有限状态机以实现令牌级约束
- [Instructor 文档](https://python.useinstructor.com/)——从任何 LLM 获取结构化输出的标准库，具有 Pydantic 验证和重试功能
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use)——Claude 如何通过带 JSON Schema `input_schema` 的工具使用实现结构化输出
- [JSON Schema 规范](https://json-schema.org/)——每个主要结构化输出系统使用的模式语言的完整规范
- [Outlines 库](https://github.com/outlines-dev/outlines)——使用编译为有限状态机的正则表达式和 JSON Schema 的开源约束生成
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100)——当前最先进的语法引擎；下推自动机编译，以约 100 ns/令牌的速度屏蔽令牌
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094)——LMQL 论文，将约束解码框架为具有类型和值约束的查询语言
- [Microsoft Guidance（框架文档）](https://github.com/guidance-ai/guidance)——模板驱动的约束生成；与供应商无关的 Outlines 和 XGrammar 补充方案

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一份**从工程实践出发、直击痛点的结构化输出教程**。它的最大价值不在于罗列 API 用法，而在于构建了一条清晰的认知链条：从"为什么 `"返回 JSON"` 不够" → "JSON 模式为什么还不够" → "模式模式解决了什么" → "约束解码才是终极答案"。这个渐进式的认知递进是优秀教学设计的标志。

文档的另一个亮点是**对失败模式的诚实分析**——幻觉值、枚举混淆、嵌套深度、数组长度、可选字段遗漏——这些都是生产环境中真正会踩的坑，而不仅仅是理论上的边缘情况。

### 二、知识结构梳理

**第一层：问题认知（"为什么需要这个"）**
- 自然语言输出 vs 类型化数据的根本矛盾
- "返回 JSON"的天真方案及其 10% 失败率的后果
- 提示工程问题 vs 解码问题的本质区分

**第二层：四个层级的概念模型（"有哪些解决方案"）**
- 提示级 → JSON 模式 → 模式模式 → 约束解码
- 每一层的可靠性、失败模式和适用场景
- JSON Schema 作为"契约语言"的定位
- Pydantic 作为 Python 生态中的模式定义标准
- 函数调用/工具使用作为替代接口

**第三层：工程实现（"怎么构建和验证"）**
- 手写 JSON Schema 验证器（递归类型检查）
- Pydantic 风格的模式生成器
- 约束令牌过滤模拟
- 提取流水线（模拟 → 解析 → 验证 → 重试）
- 三大提供商的 API 实战示例

### 三、核心洞察（备课时的关键理解）

1. **"10% 的失败率在生产中是不可接受的。"** 这是全文的隐含驱动力。如果你每天处理 10 万次 LLM 调用，10% 的失败率意味着 1 万次崩溃。你需要的不只是 90% 可靠，你需要 99.99%——而只有约束解码能给你这个。

2. **结构化输出不是提示工程问题，是解码问题。** 这是一个重要的概念分离。提示工程试图通过语言影响模型的概率分布。约束解码直接操纵概率分布——把错误令牌的概率设为零。前者是建议，后者是强制。二者的可靠性不在一个数量级上。

3. **JSON Schema 是通用契约语言。** 不管你是用 OpenAI、Anthropic 还是 Google，JSON Schema 是共同的语法。学会写一个好的 JSON Schema 比学会任何一个 SDK 都更重要——它是可迁移的技能。

4. **Pydantic 比手写 JSON Schema 好一百倍。** 类型安全、IDE 自动补全、自动生成模式、验证逻辑一体化。在 Python 生态中，直接写 JSON Schema 就像不用 ORM 而手写 SQL——可以做，但不应该。

5. **约束解码的原理很简单，实现很复杂。** 文档用一个简单的令牌过滤器做了概念演示，但真正的约束解码（Outlines、XGrammar）需要将 JSON Schema 编译为有限状态机/下推自动机，然后在每个解码步骤中检查所有 10 万+ 词汇令牌的有效性——在约 100 纳秒内完成。这是编译原理与 LLM 推理的交叉领域。

6. **幻觉无法被模式验证捕获。** 类型正确≠值正确。`{"price": 299.99}` 完美地匹配了模式，但当文本说 $348 时它就是错的。模式验证是必要的，但不够。真正的质量保证还需要事实性验证——这是下一个前沿。

7. **Instructor 的模式（验证→重试→反馈）是优雅的工程模式。** 不是"要么完美要么失败"的二元思维，而是"先试，失败后带着反馈再试"的迭代思维。这种模式不仅适用于结构化输出，也适用于任何需要 LLM 自我纠正的场景。

### 四、教学建议

1. **从一个崩溃演示开始。** 先展示 `json.loads()` 在遇到 markdown 围栏或前言文本时崩溃。让学员亲眼看到"返回 JSON"的天真方案为什么会失败。没有什么比一个 `JSONDecodeError` 更能说服人。

2. **用四个层级的递进作为教学主干。** 先讲提示级，展示它的失败，然后引入 JSON 模式，展示它的局限（不符合你的模式），再引入模式模式，最后问"但这是怎么实现的？"引出约束解码。每一层都是对前一层的不足的回应——这种叙事结构自然且有说服力。

3. **手写 JSON Schema 验证器是一个极好的练习。** 它迫使学员理解递归类型检查、路径追踪、错误收集。这些概念直接迁移到使用 Pydantic 和 Instructor 时对"验证在做什么"的理解。

4. **约束解码模拟器部分（第 3 步）要重点讲解。** 打印出部分 JSON 字符串和对应的有效令牌列表——这让"令牌屏蔽"这个抽象概念变得可视化。学员可以看到，在 `{"price":` 之后，模型被强制只能输出数字或有效值。

5. **用 Pydantic 做对比教学。** 先手写 JSON Schema，再用 Pydantic 生成相同的 Schema。让学员感受到工具链的力量——为什么我们不用手写 Schema。

6. **讲 API 示例时要跨提供商对比。** 并列展示 OpenAI、Anthropic、Instructor 三种方式完成同一个提取任务。让学员看到不同的 API 设计哲学——OpenAI 的 `response_format`、Anthropic 的 `tools`、Instructor 的统一抽象。

7. **练习 4（构建评估套件）应该作为课后作业。** 这是从"会用"到"能评估"的跃迁。手工标注 50 个样本、运行流水线、测量准确率——这是一次完整的 MLOps 微缩体验。

### 五、值得补充的内容

1. **成本分析。** 约束解码是否比普通解码更贵？（答案：取决于实现——令牌屏蔽本身几乎没有开销，但模式编译可能有一次性的启动成本。）Instructor 的重试机制在成本上是怎样的？（每次重试都是一次额外的 API 调用。）这些工程经济学问题在生产中很重要。

2. **流式输出的结构化处理。** 当使用 `stream=True` 时，结构化输出如何工作？能在令牌生成过程中逐步验证吗？还是必须等待完整输出？主流提供商的行为在这方面的差异值得记录。

3. **多语言提取的挑战。** 当输入文本是中文、日文或混合语言时，基于英文训练的提取模式是否会退化？如何设计跨语言的提取提示？

4. **结构化输出与 RAG 的结合。** 当你的提取需要引用源文本中的具体段落来验证时，输出模式应该如何设计？（如添加 `"source_span": {"start": 42, "end": 87}` 字段。）

### 六、一句话总结

> 让 LLM 返回 JSON 只是第一步；让 LLM 每次都返回正确的 JSON——并且在错误时能自动发现并修复——才是结构化输出的完整工程。约束解码给了你 100% 的语法保证，但只有结合 Pydantic 验证、自动重试和人工审核，才能达到生产级的可靠性。

---

