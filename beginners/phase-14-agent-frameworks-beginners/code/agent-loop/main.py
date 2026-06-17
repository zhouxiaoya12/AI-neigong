"""
🧠 Agent 循环是什么？—— 给初学者的最小示例

想象一下：你问 AI 一个问题，AI 不只是回答——它还能"动手"。
比如你说"帮我算 120 加税多少钱"，AI 会：
  1. 想一下 → "我需要先算税金"
  2. 调用计算器 → 120 × 0.15 = 18
  3. 拿到结果 → 再想一下 → "税金 18，总价 138"
  4. 告诉你答案 → "含税 138 元"

这个"想→做→看→再想→..."的循环，就是 Agent 的核心秘密。
所有 AI Agent（Claude Code、Cursor、Devin）底层都在跑这个循环。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：定义"一轮交互"长什么样
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 每一轮交互可能是：
#   "用户" 说了什么
#   "思考" AI 在想什么
#   "行动" AI 调用了什么工具，结果是什么
#   "结束" AI 给出了最终答案

class Turn:
    def __init__(self, kind, content, tool_name=None, tool_args=None, observation=None):
        self.kind = kind              # 类型：用户/思考/行动/结束
        self.content = content         # 文字内容
        self.tool_name = tool_name     # 如果调用了工具，工具名是什么
        self.tool_args = tool_args     # 工具参数
        self.observation = observation # 工具返回的结果


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：做一个简单的计算器工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculator(expr):
    """一个安全的计算器，只允许数字和 +-*/()."""
    allowed = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed):
        return "错误：表达式包含不允许的字符"
    try:
        # 在安全环境中执行计算
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错: {e}"


# 做一个工具注册表——你可以往里面添加工具
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        """注册一个工具"""
        self.tools[name] = fn

    def run(self, name, args):
        """执行工具并返回结果"""
        fn = self.tools.get(name)
        if fn is None:
            return f"错误：没有找到工具 {name}"
        try:
            return fn(**args)
        except Exception as e:
            return f"错误：{e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模拟一个 AI（它按照预设好的"剧本"行动）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimpleAI:
    """模拟的 AI，按预设的步骤行事。
    真实场景中，这里是调用 GPT/Claude 等大模型 API。"""

    def __init__(self, steps):
        self.steps = steps   # 预设的步骤列表
        self.pos = 0         # 当前执行到第几步

    def next_step(self):
        """返回下一步该做什么"""
        if self.pos >= len(self.steps):
            return {"type": "finish", "content": "没有更多步骤了"}
        step = self.steps[self.pos]
        self.pos += 1
        return step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：跑起来！—— Agent 循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentLoop:
    def __init__(self, ai, tools, max_steps=10):
        self.ai = ai
        self.tools = tools
        self.max_steps = max_steps
        self.history = []   # 记录每一步发生了什么

    def run(self, question):
        """执行一次完整的任务"""
        # 记录用户的问题
        self.history.append(Turn("用户", question))

        for step in range(self.max_steps):
            # 让 AI 决定下一步做什么
            reply = self.ai.next_step()

            # 如果 AI 说"做完了"，就返回答案
            if reply["type"] == "finish":
                self.history.append(Turn("结束", reply["content"]))
                return reply["content"]

            # 如果 AI 说要调用工具
            if reply["type"] == "action":
                # 记录思考过程
                thought = reply.get("thought", "")
                self.history.append(Turn("思考", thought))

                # 执行工具
                tool_name = reply["tool"]
                tool_args = reply.get("args", {})
                result = self.tools.run(tool_name, tool_args)

                # 记录工具调用和结果
                self.history.append(Turn("行动", tool_name,
                                         tool_name=tool_name,
                                         tool_args=tool_args,
                                         observation=result))

        # 超过步数上限
        return "达到最大步数，未完成"


def print_history(history):
    """把执行过程打印出来"""
    for i, turn in enumerate(history):
        if turn.kind == "用户":
            print(f"  [{i}] 👤 {turn.content}")
        elif turn.kind == "思考":
            print(f"  [{i}] 💭 {turn.content}")
        elif turn.kind == "行动":
            print(f"  [{i}] 🔧 {turn.tool_name}({turn.tool_args}) → {turn.observation}")
        elif turn.kind == "结束":
            print(f"  [{i}] ✅ {turn.content}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示时间！
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 55)
    print("🧠 Agent 循环 · 初学者版")
    print("=" * 55)

    # 准备工具
    tools = ToolRegistry()
    tools.register("计算器", calculator)

    # 准备 AI 的"剧本"：
    #   1. 先算 15% 的税
    #   2. 再算总价
    #   3. 给出答案
    script = [
        {"type": "action", "thought": "我需要计算 15% 的税：120 × 0.15",
         "tool": "计算器", "args": {"expr": "120 * 0.15"}},
        {"type": "action", "thought": "税额是 18，现在算总价：120 + 18",
         "tool": "计算器", "args": {"expr": "120 + 18"}},
        {"type": "finish", "content": "含税总价为 138 元（基础价 120 + 税额 18）"},
    ]

    # 创建 Agent 并执行
    agent = AgentLoop(SimpleAI(script), tools)
    answer = agent.run("120 元加 15% 税，一共多少钱？")

    print("\n📋 执行过程：\n")
    print_history(agent.history)

    print(f"\n📊 最终答案：{answer}")
    print(f"\n💡 看懂了吗？Agent 就是「想 → 做 → 看 → 再想」的循环。")
    print(f"   真实的 AI 把 SimpleAI 换成 API 调用，其他一模一样。")
