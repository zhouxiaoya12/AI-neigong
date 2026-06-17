"""
💰 成本调控 · 初学者版

Agent 会花钱。每调用一次大模型 API，都要消耗 token（≈ 钱）。
如果不设预算上限，一个失控的 Agent 可能一夜烧掉几百块。

这里实现最简单的三道防线：
  1. 步数上限 —— 最多执行多少步
  2. Token 上限 —— 最多消耗多少 token
  3. 成本计算 —— 实时看你花了多少钱
"""


class CostGovernor:
    """成本守门员"""

    def __init__(self, max_steps=10, max_tokens=5000):
        self.max_steps = max_steps       # 最多多少步
        self.max_tokens = max_tokens      # 最多多少 token
        self.steps_used = 0               # 已用步数
        self.tokens_used = 0              # 已用 token
        self.stopped = False              # 是否被强制停止

        # 模拟价格：输入 $3/百万token，输出 $15/百万token
        self.price_per_1k = 0.003

    def use_step(self):
        """走一步，检查是否超限"""
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            self.stopped = True
            return False
        return True

    def use_tokens(self, count):
        """消耗 token，检查是否超限"""
        self.tokens_used += count
        if self.tokens_used > self.max_tokens:
            self.stopped = True
            return False
        return True

    def cost(self):
        """计算当前花了多少钱（美元）"""
        return round(self.tokens_used / 1000 * self.price_per_1k, 4)

    def report(self):
        """打印预算报告"""
        print(f"\n📊 预算报告：")
        print(f"  步数：{self.steps_used}/{self.max_steps}")
        print(f"  Token：{self.tokens_used}/{self.max_tokens}")
        print(f"  花费：${self.cost()}")
        if self.stopped:
            print(f"  ⚠️ 预算超限，已强制停止！")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示：跑一个"花钱"的 Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_run(gov, task_steps):
    """模拟一次 Agent 运行"""
    print(f"\n🚀 开始执行（预算：{gov.max_steps}步, {gov.max_tokens}tokens）")
    print("─" * 45)

    for step_num, (thought_tokens, tool_tokens) in enumerate(task_steps, 1):
        # 检查步数
        if not gov.use_step():
            print(f"  🛑 第{step_num}步：步数超限！")
            break

        # 模拟 LLM 思考
        if not gov.use_tokens(thought_tokens):
            print(f"  🛑 第{step_num}步：Token超限！")
            break

        print(f"  [{step_num}] 💭 思考（{thought_tokens}tokens）")

        # 模拟工具调用
        if tool_tokens > 0:
            if not gov.use_tokens(tool_tokens):
                print(f"  🛑 第{step_num}步：工具返回Token超限！")
                break
            print(f"  [{step_num}] 🔧 工具执行（{tool_tokens}tokens）")

    gov.report()


if __name__ == "__main__":
    print("=" * 55)
    print("💰 成本调控 · 初学者版")
    print("=" * 55)

    # 模拟一个"话很多"的 Agent 任务
    busy_steps = [
        (100, 200),   # 第1步：思考100t + 工具200t
        (120, 300),   # 第2步
        (150, 250),   # 第3步
        (200, 400),   # 第4步
        (180, 350),   # 第5步
        (100, 200),   # 第6步
        (150, 300),   # 第7步
        (120, 250),   # 第8步
        (200, 400),   # 第9步
        (180, 300),   # 第10步
    ]

    # ── 大预算 ──
    print("\n📊 测试1：大预算（15步, 10000tokens）")
    gov1 = CostGovernor(max_steps=15, max_tokens=10000)
    simulate_run(gov1, busy_steps)

    # ── 小预算 ──
    print("\n📊 测试2：小预算（4步, 1000tokens）")
    gov2 = CostGovernor(max_steps=4, max_tokens=1000)
    simulate_run(gov2, busy_steps)

    # ── 成本对比 ──
    print(f"\n💡 成本对比：")
    print(f"   大预算 → ${gov1.cost()}（完整执行）")
    print(f"   小预算 → ${gov2.cost()}（半路被截停）")
    print(f"\n💡 关键：Agent 没有「省钱」意识，预算控制必须在 Agent 外层做。")
    print(f"   就像你不会让一个自动脚本无限制刷你的信用卡。")
