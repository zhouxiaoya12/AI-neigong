"""
成本调控 · Agent 的钱包守门员（纯标准库）

Agent 是烧钱机器。一次 Agent 循环可能跑几十步，每一步都是一次 LLM API 调用。
不设成本上限，一个 Agent 可能一夜之间烧掉几百美元的 token。

这里实现一个 12 层防护栈中最核心的 4 层：
  1. 轮次预算（Turn Budget）     —— 最多跑多少步
  2. Token 预算（Token Budget）   —— 最多消耗多少 token
  3. 工具预算（Tool Budget）      —— 最多调用多少次工具
  4. 成本累加器（Cost Accumulator）—— 实时累计并阈值切断

关键设计：每一项超出预算时，不是崩溃，而是优雅降级。
Agent 会收到一个明确的"预算超限"信息，可以在下一轮决定如何收尾。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 预算控制
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BudgetExceeded(Exception):
    """预算超限异常 —— 被 Governor 捕获，不传给 LLM"""


@dataclass
class CostGovernor:
    """12 层防护的精简实现：4 层核心预算。

    在真实系统中，这些预算值来源于：
      - 任务复杂度分析（这个任务大概需要多少步？）
      - 历史数据分析（同类任务平均消耗多少？）
      - 用户设定的上限（"最高只花 $5"）

    设计原则：永远不要让一个无限循环的 Agent 刷爆你的 API 账单。
    """

    max_turns: int = 20              # 最大轮次（思考+行动算一轮）
    max_input_tokens: int = 8000     # 输入 token 总预算
    max_output_tokens: int = 2000    # 输出 token 总预算
    max_tool_calls: int = 30         # 工具调用总次数上限

    turns_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    tool_calls_used: int = 0
    budget_exceeded: bool = False

    # 模拟的 token 价格（USD/1K tokens）
    input_price_per_1k: float = 0.003   # $3/M input tokens
    output_price_per_1k: float = 0.015  # $15/M output tokens

    def check_turn(self) -> bool:
        """检查是否超出轮次预算"""
        self.turns_used += 1
        if self.turns_used > self.max_turns:
            self.budget_exceeded = True
            return False
        return True

    def track_input(self, token_count: int) -> bool:
        """记录输入 token 消耗，返回是否还有预算"""
        self.input_tokens_used += token_count
        if self.input_tokens_used > self.max_input_tokens:
            self.budget_exceeded = True
            return False
        return True

    def track_output(self, token_count: int) -> bool:
        """记录输出 token 消耗，返回是否还有预算"""
        self.output_tokens_used += token_count
        if self.output_tokens_used > self.max_output_tokens:
            self.budget_exceeded = True
            return False
        return True

    def track_tool_call(self) -> bool:
        """记录工具调用，返回是否还有预算"""
        self.tool_calls_used += 1
        if self.tool_calls_used > self.max_tool_calls:
            self.budget_exceeded = True
            return False
        return True

    def cost_usd(self) -> float:
        """计算当前已消耗的总成本（美元）"""
        input_cost = (self.input_tokens_used / 1000) * self.input_price_per_1k
        output_cost = (self.output_tokens_used / 1000) * self.output_price_per_1k
        return round(input_cost + output_cost, 4)

    def status_report(self) -> str:
        """生成预算状态报告"""
        lines = [
            "📊 预算状态：",
            f"  轮次：  {self.turns_used}/{self.max_turns}",
            f"  输入：  {self.input_tokens_used}/{self.max_input_tokens} tokens",
            f"  输出：  {self.output_tokens_used}/{self.max_output_tokens} tokens",
            f"  工具：  {self.tool_calls_used}/{self.max_tool_calls} 次",
            f"  成本：  ${self.cost_usd()}",
        ]
        if self.budget_exceeded:
            lines.append("  ⚠️ 预算已超限！")
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模拟 Agent 运行（带成本调控）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MockTurn:
    """模拟一轮 Agent 交互"""
    step: int
    thought_tokens: int = 50       # 思考消耗的 token
    action: str | None = None      # 如果是工具调用，工具名
    tool_output_tokens: int = 0    # 工具返回消耗的 token
    finish: bool = False           # 是否结束


def simulate_agent_run(governor: CostGovernor, turns: list[MockTurn]) -> str:
    """模拟一次 Agent 运行，每一步都经过 CostGovernor 检查。

    流程和真实 Agent 循环完全一致：
      用户输入 → LLM思考(track_input+output) → 工具调用(track_tool_call)
        → 工具结果(track_input) → LLM再思考(track_input+output) → ... → 结束

    区别在于：每一步都经过 CostGovernor 检查，超限即停止。
    """
    results: list[str] = []

    for turn in turns:
        # 第1层：轮次预算
        if not governor.check_turn():
            results.append(f"🛑 第{turn.step}步：轮次超限（{governor.turns_used} > {governor.max_turns}），强制停止")
            break

        # 第2层：输入 token 预算（模拟"接收 LLM 返回"）
        if not governor.track_input(100):  # 每轮输入约100 tokens
            results.append(f"🛑 第{turn.step}步：输入token预算超限")
            break

        # 第3层：输出 token 预算（模拟"LLM 思考输出"）
        if not governor.track_output(turn.thought_tokens):
            results.append(f"🛑 第{turn.step}步：输出token预算超限")
            break

        results.append(f"  [{turn.step:02d}] LLM思考 → {turn.thought_tokens}tokens")

        # 如果有工具调用
        if turn.action:
            # 第4层：工具调用预算
            if not governor.track_tool_call():
                results.append(f"🛑 第{turn.step}步：工具调用次数超限")
                break

            results.append(f"  [{turn.step:02d}] 🔧 调用 {turn.action}")

            # 工具返回也消耗输入 token
            governor.track_input(turn.tool_output_tokens)
            results.append(f"  [{turn.step:02d}] 📥 工具返回 → {turn.tool_output_tokens}tokens")

        if turn.finish:
            results.append(f"  [{turn.step:02d}] ✅ Agent 完成")
            break

    return "\n".join(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示：三种预算策略对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("💰 成本调控 · Agent 的钱包守门员")
    print("=" * 65)

    # 模拟一个"贪心"的 Agent 任务序列（20步 + 大量工具调用）
    greedy_turns = [
        MockTurn(1, thought_tokens=80, action="搜索", tool_output_tokens=200),
        MockTurn(2, thought_tokens=60, action="计算", tool_output_tokens=50),
        MockTurn(3, thought_tokens=90, action="搜索", tool_output_tokens=300),
        MockTurn(4, thought_tokens=70, action="读取文件", tool_output_tokens=150),
        MockTurn(5, thought_tokens=100, action="搜索", tool_output_tokens=250),
        MockTurn(6, thought_tokens=60, action="计算", tool_output_tokens=50),
        MockTurn(7, thought_tokens=80, action="搜索", tool_output_tokens=200),
        MockTurn(8, thought_tokens=70, action="写入", tool_output_tokens=100),
        MockTurn(9, thought_tokens=50),
        MockTurn(10, thought_tokens=40, finish=True),
    ] + [MockTurn(11 + i, thought_tokens=60) for i in range(10)]  # 多余的步

    # ── 策略 A：宽松预算（大任务） ──
    print("\n📊 策略 A：宽松预算（适合复杂任务）")
    print("─" * 45)
    gov_a = CostGovernor(
        max_turns=15,
        max_input_tokens=10000,
        max_output_tokens=3000,
        max_tool_calls=25,
    )
    trace = simulate_agent_run(gov_a, greedy_turns)
    print(trace)
    print()
    print(gov_a.status_report())

    # ── 策略 B：保守预算（常规任务） ──
    print("\n📊 策略 B：保守预算（适合常规任务）")
    print("─" * 45)
    gov_b = CostGovernor(
        max_turns=6,
        max_input_tokens=1500,
        max_output_tokens=800,
        max_tool_calls=8,
    )
    trace = simulate_agent_run(gov_b, greedy_turns)
    print(trace)
    print()
    print(gov_b.status_report())

    # ── 策略 C：极简预算（简单查询） ──
    print("\n📊 策略 C：极简预算（适合简单查询/分类任务）")
    print("─" * 45)
    gov_c = CostGovernor(
        max_turns=3,
        max_input_tokens=500,
        max_output_tokens=300,
        max_tool_calls=2,
    )
    trace = simulate_agent_run(gov_c, greedy_turns)
    print(trace)
    print()
    print(gov_c.status_report())

    # ── 成本对比 ──
    print("\n" + "=" * 65)
    print("📊 三种策略成本对比")
    print("=" * 65)
    for label, gov in [("A. 宽松", gov_a), ("B. 保守", gov_b), ("C. 极简", gov_c)]:
        print(f"  {label:8s}  ${gov.cost_usd():.4f}  |  "
              f"{gov.turns_used:2d}步  {gov.tool_calls_used:2d}次调用  "
              f"超限={'是' if gov.budget_exceeded else '否'}")

    print(f"\n💡 成本调控的设计原则：")
    print(f"  • 预算不是一刀切的固定值——按任务复杂度分档")
    print(f"  • 简单分类任务给 3 步，复杂代码任务给 50 步")
    print(f"  • 超限不崩溃——Agent 收到预算超限信息后可以自我收尾")
    print(f"  • 在真实系统中，累加器对接 LLM API 的 usage 返回字段")
    print(f"  • 成本监控应该和可观测性系统（Langfuse/Phoenix）联动")
    print(f"  • 记住：Agent 没有「成本意识」——这层防护必须在外层做")


if __name__ == "__main__":
    main()
