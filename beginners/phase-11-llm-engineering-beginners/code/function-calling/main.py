"""
📞 Function Calling · 初学者版

这是真正调用大模型 API 时发生的事：
  1. 你告诉 AI："你有这些工具可以用"
  2. AI 回复："我想用搜索工具，搜关键词'退款政策'"
  3. 你执行搜索，拿到结果"退款：30天内全额退款"
  4. 你把结果发给 AI："搜索结果：退款：30天内全额退款"
  5. AI 基于结果回答："根据政策，您可以在30天内申请退款"

这里用一个模拟的 AI 来展示这个循环——换成真实 API 一模一样。
"""


class ToolExecutor:
    """工具执行器"""
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        self.tools[name] = fn

    def run(self, name, args):
        if name not in self.tools:
            return f"错误：未找到工具 '{name}'"
        try:
            return self.tools[name](**args)
        except Exception as e:
            return f"错误：{e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模拟一个能"调用工具"的 AI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockAI:
    """模拟的 AI，返回预设的对话流程"""
    def __init__(self, script):
        self.script = script
        self.pos = 0

    def chat(self):
        """返回下一步：可能是 tool_call（要调工具）或 stop（结束）"""
        if self.pos >= len(self.script):
            return {"type": "stop", "content": "对话结束"}
        step = self.script[self.pos]
        self.pos += 1
        return step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Function Calling 循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def function_calling_loop(ai, executor, question, max_rounds=5):
    """完整的 Function Calling 流程"""
    print(f"📋 用户问：{question}")

    for round_num in range(1, max_rounds + 1):
        print(f"\n─ 第{round_num}轮 ─")
        response = ai.chat()

        if response["type"] == "stop":
            print(f"  🤖 AI 回答：{response['content']}")
            return

        elif response["type"] == "tool_call":
            tool_name = response["tool"]
            tool_args = response.get("args", {})
            print(f"  🔧 AI 想调用：{tool_name}({tool_args})")

            # 执行工具
            result = executor.run(tool_name, tool_args)
            print(f"  📥 执行结果：{result}")

            # 下一轮 AI 会看到这个结果


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search_knowledge(query, top_k=3):
    """模拟知识库搜索"""
    kb = {
        "退款": "退款政策：30天内全额退款，企业版60天按比例。",
        "价格": "新手版29元/月，专业版99元/月，企业版500元起/月。",
    }
    for key, val in kb.items():
        if key in query:
            return val
    return "未找到相关信息"


def get_weather(city):
    """模拟天气查询"""
    db = {"北京": "晴 25°C", "上海": "多云 28°C", "深圳": "雷阵雨 30°C"}
    return db.get(city, f"未找到{city}天气")


if __name__ == "__main__":
    print("=" * 55)
    print("📞 Function Calling · 初学者版")
    print("=" * 55)

    # 准备工具
    executor = ToolExecutor()
    executor.register("搜索知识库", search_knowledge)
    executor.register("查天气", get_weather)

    # ── 场景1：单工具调用 ──
    print("\n📝 场景1：退款查询\n" + "─" * 45)
    ai1 = MockAI([
        {"type": "tool_call", "tool": "搜索知识库", "args": {"query": "退款"}},
        {"type": "stop", "content": "根据政策，您可以在30天内申请全额退款。请问还有什么需要？"},
    ])
    function_calling_loop(ai1, executor, "我想退款，怎么办？")

    # ── 场景2：多工具调用 ──
    print("\n📝 场景2：查天气\n" + "─" * 45)
    ai2 = MockAI([
        {"type": "tool_call", "tool": "查天气", "args": {"city": "上海"}},
        {"type": "stop", "content": "上海今天多云，28°C，适合出行。"},
    ])
    function_calling_loop(ai2, executor, "上海今天天气怎么样？")

    print(f"\n💡 Function Calling 和普通聊天的区别：")
    print(f"   普通聊天：用户问 → AI 答")
    print(f"   Function Calling：用户问 → AI 决定调工具 → 执行工具 → AI 看结果 → AI 答")
    print(f"   多了一步「查资料」—— 这就是 AI 从「聊天」变成「能干活的 Agent」的关键。")
