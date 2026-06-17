# 结构化输出：别让 AI 给你返回一段废话

> **原文：** Structured Outputs: JSON, Schema Validation, Constrained Decoding  
> **预计时间：** ~45 分钟  
> **代码：** Python（完整保留，加中文注释）

---

## 🚗 开场翻车：凌晨三点的正则地狱

你让 LLM："从这段文本提取产品名、价格、库存。"

它回你："该产品是 Sony WH-1000XM5 耳机，售价 $348.00，目前有货。"

**完全正确——但你的代码要的是 `{"product": "...", "price": 348, "in_stock": true}`。**

你加了"请返回 JSON 格式"。90% 时候管用。10% 时候模型把 JSON 包在 markdown 围栏里、或者前面加一句"以下是 JSON："、或者少了一个右花括号。

凌晨三点，你对着正则表达式在 1000 行日志里找少了的 `}`。

---

## 🧠 结构化输出的四个等级

```
等级 1: "请返回 JSON"                    → ~90% 有效
等级 2: JSON 模式（保证有效 JSON）        → 100% 可解析  
等级 3: Schema 模式（JSON + 匹配结构）    → 100% 结构合规
等级 4: 约束解码（令牌级封杀）            → 100% 令牌级合规
```

### 等级 1：提示级——靠运气

```
"以 JSON 格式返回。" → 模型可能照做，也可能不。
失败模式：markdown 围栏、废话前缀、少花括号
```

### 等级 2：JSON 模式——保证能 parse

```python
# OpenAI: response_format 参数强制输出有效 JSON
# response_format={"type": "json_object"}
# → 保证 JSON.parse 不会崩。但字段可能不对。
```

### 等级 3：Schema 模式——保证结构正确

```python
# 告诉模型输出必须符合这个 JSON Schema
# 到 2026 年三家都原生支持
schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
    },
    "required": ["product", "price", "in_stock"]
}
```

### 等级 4：约束解码——令牌级封锁

```
模型要输出 {"price": 
下一个令牌必须是数字/引号/null/true/false
如果模型要输出字母 → 该令牌概率直接清零
模型只能走"通向有效 JSON"的路
```

这是 OpenAI Structured Outputs 模式底层的东西。每一帧都封杀无效令牌。

---

## 🔧 JSON Schema 验证器

```python
import json

def validate_schema(data, schema):
    """从头写的 JSON Schema 验证器。
    检查对象、数组、字符串、数字、布尔值是否匹配 schema。
    返回错误列表。空列表 = 通过。"""
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: 期望对象，实际是 {type(data).__name__}")
            return
        # 检查必填字段
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: 必填字段缺失")
        # 递归验证每个属性
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: 期望数组，实际是 {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: 数组有 {len(data)} 项，最少 {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: 数组有 {len(data)} 项，最多 {max_items}")
        # 递归验证每个数组元素
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: 期望字符串，实际是 {type(data).__name__}")
            return
        # 检查枚举值约束
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' 不在允许值 {enum_values} 中")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: 期望数字，实际是 {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} 小于最小值 {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} 大于最大值 {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: 期望布尔值，实际是 {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            # Python 里 bool 是 int 的子类，所以要排除
            errors.append(f"{path}: 期望整数，实际是 {type(data).__name__}")
```

---

## 🐍 Pydantic——不手写 Schema

```python
from pydantic import BaseModel

# 定义一个 Pydantic 模型 = 自动生成 JSON Schema + 自动验证
class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []  # 默认空列表

# Instructor 库直接接收 Pydantic 模型
# 如果 LLM 输出不匹配 → 自动重试，最多 N 次
```

---

## ⚠️ Schema 验证抓不到的四种失败

| 失败 | 例子 | 为什么 schema 抓不到 |
|------|------|---------------------|
| **幻觉值** | 文本说 $348，模型输出 299.99 | 类型对，值错 |
| **枚举混淆** | 约束是 `["in_stock","out_of_stock"]`，模型输出 `"available"` | 语义对，不在集合里 |
| **嵌套太深** | 4 层以上的嵌套对象 | 模型跟丢结构 |
| **数组越界** | 要求 3-5 个元素，模型给了 12 个 | 并非所有提供商在解码级别强制执行 |

---

## 🎮 互动练习

LLM 返回了这个 JSON。用上面的 `validate_schema` 检查，哪些字段有问题？

```json
{
  "product": "iPhone 16",
  "price": -100,
  "in_stock": "yes"
}
```

Schema: `required: ["product","price","in_stock"]`, `price: number minimum=0`, `in_stock: boolean`

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **"以 JSON 返回" ≠ 保证拿到 JSON。** 10% 的失败率在生产环境是不可接受的。用 JSON Schema 或 Structured Outputs。

2. **约束解码是唯一 100% 可靠的方式。** 令牌级封杀，模型只能输出通向有效 JSON 的令牌。

3. **Pydantic + Instructor = 不用手写 Schema + 自动重试验证。**

4. **Schema 验证抓类型，抓不到幻觉值。** 价格类型对但数字错——schema 不管。需要业务层验证。

---

## 📝 练习答案

```
- product: "iPhone 16" ✅ string，正确
- price: -100 ❌ minimum=0，但 -100 < 0  
- in_stock: "yes" ❌ 期望 boolean，实际是 string
```

**两个错误。** price 不满足最小值约束。in_stock 类型错误——应该是 `true` 不是 `"yes"`。

---

> **让 AI 返回 JSON 不难——难的是让它次次返回的 JSON 都能被你的代码直接吞掉。等级 1 靠运气，等级 4 靠令牌级封锁。凌晨三点的正则地狱不值得你的一生。**
