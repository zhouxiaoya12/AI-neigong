"""
工具调用与函数调用 · 带 Schema 校验的完整实现（纯标准库）

从上一课的 Agent 循环往下走一层：当模型说"我要调用 search(query='xxx')"时，
你怎么保证参数是对的？字符串没传成数字？必填字段没漏？枚举值不在范围内？

这里实现了完整的工具调用链路：
  工具定义(Schema) → JSON Schema 子集校验 → 类型自动转换 → 并行批量调度

每一个校验失败都以"结构化错误字符串"返回给模型，让模型能看懂错在哪、怎么改。
这也是生产环境 Agent 必备的能力——工具调用出错了，不能让循环崩溃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ToolDef:
    """一个工具的完整定义 —— 不只是函数，还有 Schema 和元信息。

    在真实场景中，name/description/input_schema 会作为 system prompt 的一部分
    发送给 LLM，告诉它"你现在有这些工具可以用"。
    """
    name: str                          # 工具名称，LLM 看到的名字
    description: str                   # 工具描述，帮助 LLM 判断何时使用
    input_schema: dict[str, Any]       # JSON Schema 子集，约束输入参数
    executor: Callable[..., str]       # 实际执行的函数
    timeout_s: float = 5.0             # 超时时间（本实现中未使用，留作扩展）


@dataclass
class ToolCall:
    """一次工具调用请求 —— 来自 LLM 的 function call"""
    tool_use_id: str           # 调用ID，用于关联请求和结果
    name: str                  # 要调用的工具名
    args: dict[str, Any]       # 调用参数


@dataclass
class ToolResult:
    """一次工具调用的结果 —— 返回给 LLM 的观察"""
    tool_use_id: str           # 关联回原来的调用ID
    ok: bool                   # 执行成功还是失败
    content: str               # 执行结果或错误信息（必须是字符串！）


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON Schema 子集校验 + 类型自动转换
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _coerce(value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """类型转换器。

    LLM 传过来的参数可能是字符串 "3" 而不是数字 3。
    这里做自动转换（整数、浮点、布尔），转换失败返回错误信息。

    返回：(转换后的值, 错误信息或None)
    """
    t = schema.get("type")

    if t == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            try:
                return int(value), None
            except ValueError:
                return value, f"无法将字符串 {value!r} 转为整数"
        return value, f"期望整数类型，实际收到 {type(value).__name__}"

    if t == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        if isinstance(value, str):
            try:
                return float(value), None
            except ValueError:
                return value, f"无法将字符串 {value!r} 转为数字"
        return value, f"期望数字类型，实际收到 {type(value).__name__}"

    if t == "boolean":
        if isinstance(value, bool):
            return value, None
        return value, f"期望布尔类型，实际收到 {type(value).__name__}"

    if t == "string":
        if isinstance(value, str):
            return value, None
        return value, f"期望字符串类型，实际收到 {type(value).__name__}"

    if t == "array":
        if isinstance(value, list):
            return value, None
        return value, f"期望数组类型，实际收到 {type(value).__name__}"

    if t == "object":
        if isinstance(value, dict):
            return value, None
        return value, f"期望对象类型，实际收到 {type(value).__name__}"

    return value, None


def validate(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """校验工具调用参数是否符合 Schema 定义。

    检查项：
      - 必填字段是否缺失
      - 未知字段是否存在
      - 类型是否正确（自动转换）
      - 枚举值是否在允许范围内
      - 数值是否在 min/max 范围内

    返回：(校验后的参数, 错误列表)
    """
    errors: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    out: dict[str, Any] = {}

    # 检查必填字段
    for name in required:
        if name not in args:
            errors.append(f"缺少必填参数: {name}")

    # 逐字段校验
    for name, value in args.items():
        prop = props.get(name)
        if prop is None:
            errors.append(f"未知参数: {name}")
            continue

        # 类型转换
        coerced, err = _coerce(value, prop)
        if err:
            errors.append(f"参数 {name}: {err}")
            continue

        # 枚举值校验
        if "enum" in prop and coerced not in prop["enum"]:
            errors.append(f"参数 {name}: 值 {coerced!r} 不在允许范围 {prop['enum']} 内")
            continue

        # 数值范围校验
        if prop.get("type") in ("number", "integer"):
            if "minimum" in prop and coerced < prop["minimum"]:
                errors.append(f"参数 {name}: {coerced} 小于最小值 {prop['minimum']}")
                continue
            if "maximum" in prop and coerced > prop["maximum"]:
                errors.append(f"参数 {name}: {coerced} 大于最大值 {prop['maximum']}")
                continue

        out[name] = coerced

    return out, errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具注册表（升级版——带 Schema 校验）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolRegistry:
    """带 Schema 校验的工具注册表。

    与上一课的简化版相比，增加了：
      - Schema 驱动的参数校验
      - 类型自动转换
      - catalog() 方法：生成给 LLM 看的工具列表
      - dispatch_many() 方法：并行调用多个工具
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def catalog(self) -> list[dict[str, Any]]:
        """生成工具目录 —— 这就是发给 LLM 的 function definitions"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema
            }
            for t in self._tools.values()
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用，包含完整校验。

        失败场景的处理顺序：
          1. 工具不存在 → ToolResult(ok=False, ...)
          2. 参数校验失败 → ToolResult(ok=False, ...)
          3. 执行异常 → ToolResult(ok=False, ...)
          4. 正常返回 → ToolResult(ok=True, ...)
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call.tool_use_id, False,
                f"错误：未知工具 {call.name!r}。可用工具：{list(self._tools)}"
            )

        # Schema 校验
        validated, errors = validate(call.args, tool.input_schema)
        if errors:
            return ToolResult(
                call.tool_use_id, False,
                "参数校验失败：" + "；".join(errors)
            )

        # 执行
        try:
            return ToolResult(call.tool_use_id, True, tool.executor(**validated))
        except Exception as e:
            return ToolResult(
                call.tool_use_id, False,
                f"执行异常：{type(e).__name__}: {e}"
            )

    def dispatch_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """并行调用多个工具 —— 现代 LLM 支持一次返回多个 function call"""
        return [self.dispatch(c) for c in calls]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 示例工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add(a: int, b: int) -> str:
    """两个整数相加"""
    return str(a + b)


def multiply(a: int, b: int) -> str:
    """两个整数相乘"""
    return str(a * b)


def classify(status: str) -> str:
    """分类标注"""
    return f"已分类为: {status}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("🔧 工具调用与函数调用 · 带 Schema 校验")
    print("=" * 65)

    # 创建工具注册表
    reg = ToolRegistry()

    # 注册加法工具
    reg.register(ToolDef(
        name="加法",
        description="计算两个整数的和。当需要做加法运算时使用。",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "第一个加数"},
                "b": {"type": "integer", "description": "第二个加数"},
            },
            "required": ["a", "b"],
        },
        executor=add,
    ))

    # 注册乘法工具
    reg.register(ToolDef(
        name="乘法",
        description="计算两个整数的积。优先使用乘法而非循环加法。",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        executor=multiply,
    ))

    # 注册分类工具 —— 带枚举约束
    reg.register(ToolDef(
        name="分类标注",
        description="将状态分类为三个标签之一。",
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["已开启", "已关闭", "待处理"],
                    "description": "要分类的状态值",
                },
            },
            "required": ["status"],
        },
        executor=classify,
    ))

    # ━━ 展示工具目录（这就是发给 LLM 的 function definitions）━━
    print("\n📋 工具目录（将作为 function definitions 发给 LLM）：")
    for entry in reg.catalog():
        print(f"  • {entry['name']}: {entry['description']}")
        print(f"    Schema: {entry['input_schema']}")

    # ━━ 并行调用演示 ━━
    print("\n" + "=" * 65)
    print("并行调用测试（5 次调用，一次执行）")

    calls = [
        ToolCall("c01", "加法", {"a": 2, "b": 3}),                      # ✅ 正常
        ToolCall("c02", "乘法", {"a": "4", "b": 5}),                     # ✅ 字符串 "4" 自动转整数
        ToolCall("c03", "分类标注", {"status": "进行中"}),                # ❌ 枚举值不在范围内
        ToolCall("c04", "分类标注", {"status": "待处理"}),                # ✅ 合法枚举值
        ToolCall("c05", "减法", {"a": 1, "b": 2}),                       # ❌ 工具不存在
    ]

    for result in reg.dispatch_many(calls):
        status = "✅" if result.ok else "❌"
        print(f"  {status} [{result.tool_use_id}] {result.content}")

    # ━━ 关键洞察 ━━
    print(f"\n💡 关键洞察：")
    print(f"   • 每一个校验失败都返回了结构化的错误字符串")
    print(f"   • 字符串 '4' 被自动转换为整数 4（类型自动转换）")
    print(f"   • 枚举值 '进行中' 被明确拒绝并告知合法选项")
    print(f"   • 未知工具 '减法' 被拒绝并告知可用工具列表")
    print(f"   • Agent 拿到这些错误信息后可以自我修正，不会崩溃")
    print(f"\n   ⚠️ 重要设计决策：所有错误都以字符串返回，从不抛出异常。")
    print(f"   Agent 需要的是'可读的错误信息'，不是 crashed 的进程。")


if __name__ == "__main__":
    main()
