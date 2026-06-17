"""
🔧 工具调用 · 初学者版

上一课学了 Agent 循环。这一课深入"工具调用"这一步。

当 AI 说"我要调计算器，参数是 120*0.15"时，你需要：
  1. 检查这个工具存不存在
  2. 检查参数对不对（类型、必填字段）
  3. 执行工具
  4. 把结果返回给 AI

这里演示一个带参数校验的工具注册表。
"""


class ToolRegistry:
    """带校验的工具注册表"""

    def __init__(self):
        self.tools = {}  # name → (function, schema)

    def register(self, name, fn, schema):
        """注册工具，附带参数 Schema"""
        self.tools[name] = (fn, schema)

    def _check_args(self, name, args, schema):
        """校验参数是否符合 Schema"""
        errors = []
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # 检查必填参数
        for field in required:
            if field not in args:
                errors.append(f"缺少必填参数: {field}")

        # 检查每个参数的值
        for key, value in args.items():
            if key not in props:
                errors.append(f"未知参数: {key}")
                continue

            expected_type = props[key].get("type")
            # 检查类型
            if expected_type == "integer":
                if not isinstance(value, int):
                    # 尝试转换
                    if isinstance(value, str) and value.isdigit():
                        args[key] = int(value)
                    else:
                        errors.append(f"参数 {key} 应为整数，收到: {value}")
            elif expected_type == "string":
                if not isinstance(value, str):
                    errors.append(f"参数 {key} 应为字符串，收到: {value}")

            # 检查枚举值
            allowed = props[key].get("enum")
            if allowed and args[key] not in allowed:
                errors.append(f"参数 {key} 值 {args[key]!r} 不在允许范围 {allowed}")

        return errors

    def run(self, name, args):
        """执行工具并返回结果"""
        # 第1步：工具存在吗？
        if name not in self.tools:
            return f"❌ 未知工具 '{name}'。可用工具：{list(self.tools.keys())}"

        fn, schema = self.tools[name]

        # 第2步：参数对吗？
        errors = self._check_args(name, dict(args), schema)
        if errors:
            return "❌ 参数错误：" + "；".join(errors)

        # 第3步：执行！
        try:
            return "✅ " + fn(**args)
        except Exception as e:
            return f"❌ 执行失败: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 准备几个示例工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add(a, b):
    return str(a + b)

def multiply(a, b):
    return str(a * b)

def classify(status):
    return f"已分类为「{status}」"


if __name__ == "__main__":
    print("=" * 55)
    print("🔧 工具调用 · 初学者版")
    print("=" * 55)

    # 注册工具（每个工具有名称、函数、参数规则）
    reg = ToolRegistry()

    reg.register("加法", add, {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "第一个加数"},
            "b": {"type": "integer", "description": "第二个加数"},
        },
        "required": ["a", "b"],
    })

    reg.register("乘法", multiply, {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    })

    reg.register("分类", classify, {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["已开启", "已关闭", "待处理"]},
        },
        "required": ["status"],
    })

    # ── 测试各种情况 ──
    print("\n📋 测试 1：正常调用")
    print(f"  加法(2, 3) → {reg.run('加法', {'a': 2, 'b': 3})}")
    print(f"  乘法(4, 5) → {reg.run('乘法', {'a': 4, 'b': 5})}")
    print(f"  分类(已开启) → {reg.run('分类', {'status': '已开启'})}")

    print("\n📋 测试 2：自动类型转换")
    # 字符串 "10" 会自动转成整数 10
    print(f"  加法('10', 5) → {reg.run('加法', {'a': '10', 'b': 5})}")

    print("\n📋 测试 3：参数错误处理")
    print(f"  分类('进行中') → {reg.run('分类', {'status': '进行中'})}")
    print(f"  减法(1, 2) → {reg.run('减法', {'a': 1, 'b': 2})}")
    print(f"  加法(缺少参数) → {reg.run('加法', {'a': 5})}")

    print(f"\n💡 关键点：工具调用出错时，返回的是错误「信息」，不是崩溃。")
    print(f"   AI 看到错误信息后可以自己修正——这才是 Agent 的韧性。")
