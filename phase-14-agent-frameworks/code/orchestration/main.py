"""
多 Agent 编排的四种拓扑 · 同一任务，四种做法

一个三意图的分类任务（退款 / 报修 / 询价），用四种编排模式分别处理，
直观对比操作次数、控制流和成本差异。

四种拓扑：
  1. 监督者-工人（Supervisor-Worker）    —— 中心调度，最干净
  2. 群体传递（Swarm / Handoff）         —— 对等传递，路径最短
  3. 层级式（Hierarchical）               —— 多层路由，最深层级
  4. 多Agent辩论（Debate）                —— 共识决策，最昂贵

设计原则：选拓扑之前，先搞清楚你的问题长什么样。
不是"哪种拓扑最好"，是"哪种拓扑最适合你的任务"。
"""

from __future__ import annotations

from collections import Counter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 意图分类器 —— 所有四种拓扑共用的"路由器"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def classify(text: str) -> str:
    """根据用户消息的关键词判断意图类型。

    真实 Agent 中，这是一个 LLM 调用（classification / intent detection）。
    这里用关键词规则模拟，让示例离线可跑。
    """
    t = text.lower()
    if any(w in t for w in ("退款", "退钱", "退货", "取消订单")):
        return "退款"
    if any(w in t for w in ("崩溃", "报错", "bug", "故障", "闪退", "打不开")):
        return "报修"
    if any(w in t for w in ("报价", "定价", "多少钱", "价格", "收费")):
        return "询价"
    return "通用咨询"


# 专业处理函数：每个领域有一个"专家"
SPECIALISTS = {
    "退款":     lambda t: f"退款已处理：{t[:30]}",
    "报修":     lambda t: f"故障已登记：{t[:30]}",
    "询价":     lambda t: f"报价已发送：{t[:30]}",
    "通用咨询": lambda t: f"已转人工客服：{t[:30]}",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 拓扑 1：监督者-工人（Supervisor-Worker）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def supervisor_worker(tasks: list[str]) -> tuple[list[str], int]:
    """监督者模式：中心调度器分类后直接派发给对应专家。

    流程：
      用户消息 → 监督者分类 → 路由到对应工人 → 工人处理 → 返回结果

    优点：控制流最简洁，每任务 2 次操作
    缺点：监督者是单点，复杂任务需要多次来回

    适用场景：意图明确的分类路由（客服分流、工单分类）
    """
    trace: list[str] = []
    ops = 0

    for task in tasks:
        ops += 1  # 分类操作
        label = classify(task)
        trace.append(f"监督者 → 分配到【{label}】部门")

        specialist = SPECIALISTS.get(label, SPECIALISTS["通用咨询"])
        ops += 1  # 处理操作
        trace.append(f"  {label}专家: {specialist(task)}")

    return trace, ops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 拓扑 2：群体传递（Swarm / Handoff）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def swarm(tasks: list[str]) -> tuple[list[str], int]:
    """群体模式：没有中心调度，Agent 之间直接传递（handoff）。

    流程：
      第一个 Agent 接到消息 → 判断是不是自己的活 → 是则处理，否则传给下一位

    和 OpenAI Agents SDK 的 handoff 是同一回事——"转交"本质上一个工具调用。

    优点：没有单点，动态适应
    缺点：传递次数不可控，极端情况可能来回踢皮球

    适用场景：Agent 角色动态变化、无法预分类的场景
    """
    trace: list[str] = []
    ops = 0

    for task in tasks:
        current = list(SPECIALISTS.keys())[0]  # 从第一个专家开始
        hops = 0
        while hops < 3:  # 防止无限传递
            ops += 1
            label = classify(task)
            if current == label:
                trace.append(f"Agent[{current}]: {SPECIALISTS[current](task)}")
                break
            trace.append(f"Agent[{current}] 转交 → Agent[{label}]")
            current = label
            hops += 1

    return trace, ops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 拓扑 3：层级式（Hierarchical）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def hierarchical(tasks: list[str]) -> tuple[list[str], int]:
    """层级模式：先粗分大类，再细分小类。

    流程：
      消息 → 顶层路由（售后/商务）→ 中层路由（退款/报修）→ 底层执行

    和 LangGraph 的 hierarchical agent network 是同一套逻辑。

    优点：结构清晰，权限和职责分明
    缺点：层级越多操作越多，链路长

    适用场景：组织架构映射到 Agent 结构（大企业多级客服）
    """
    trace: list[str] = []
    ops = 0

    for task in tasks:
        ops += 1  # 顶层路由
        top_label = "售后部门" if classify(task) != "询价" else "商务部"
        trace.append(f"总调度 → 分配到【{top_label}】")

        ops += 1  # 中层路由
        sub_label = classify(task)
        trace.append(f"  {top_label} → 分配到【{sub_label}】组")

        specialist = SPECIALISTS.get(sub_label, SPECIALISTS["通用咨询"])
        ops += 1  # 底层执行
        trace.append(f"    {sub_label}专员: {specialist(task)}")

    return trace, ops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 拓扑 4：多 Agent 辩论（Debate）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def debate(tasks: list[str]) -> tuple[list[str], int]:
    """辩论模式：多个 Agent 各自独立判断，投票决定最终结果。

    流程：
      消息 → Agent A/B/C 各自分类 → 投票取多数 → 胜出分类的专家执行

    原则：分歧比一致更有用。如果三个 Agent 给出一致的分类，高可信度；
          如果分歧很大，说明这个任务需要人工介入。

    优点：容错性最高，单一 Agent 误判不会影响最终结果
    缺点：操作数翻倍（每个 Agent 都过一遍）

    适用场景：高风险决策（医疗分诊、金融风控、合规审核）
    """

    trace: list[str] = []
    ops = 0

    for task in tasks:
        proposals: list[str] = []
        for debater in ("评审Agent-A", "评审Agent-B", "评审Agent-C"):
            ops += 1
            label = classify(task)
            proposals.append(label)
            trace.append(f"{debater} 判定: 【{label}】")

        ops += 1  # 投票操作
        convergent = Counter(proposals).most_common(1)[0][0]
        majority_count = Counter(proposals).most_common(1)[0][1]

        specialist = SPECIALISTS.get(convergent, SPECIALISTS["通用咨询"])
        ops += 1  # 执行操作
        trace.append(
            f"投票结果 → 【{convergent}】({majority_count}/3票): {specialist(task)}"
        )

    return trace, ops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("🔀 多 Agent 编排 · 四种拓扑对比")
    print("=" * 65)

    # 三个典型任务，三种不同意图
    tasks = [
        "我申请退款，订单号 INV-4711",
        "你们的 App 在安卓上点登录按钮就闪退",
        "能给我一个 50 人的企业版报价吗？",
    ]

    print(f"\n📋 待处理任务（{len(tasks)}条）：")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    topologies = {
        "监督者-工人": supervisor_worker,
        "群体传递":    swarm,
        "层级式":      hierarchical,
        "多Agent辩论": debate,
    }

    results: dict[str, int] = {}

    for name, fn in topologies.items():
        trace, ops = fn(tasks)
        results[name] = ops
        print(f"\n{'─' * 65}")
        print(f"🔹 {name} · 操作次数={ops}")
        print(f"{'─' * 65}")
        for line in trace:
            print(f"  {line}")

    # ━━ 对比总结 ━━
    print(f"\n{'=' * 65}")
    print(f"📊 四种拓扑成本对比")
    print(f"{'=' * 65}")

    baseline = min(results.values())
    for name, ops in sorted(results.items(), key=lambda x: x[1]):
        ratio = ops / baseline
        bar = "█" * int(ratio * 10)
        print(f"  {name:12s}  {ops:>3} 次操作  {bar}  ({ratio:.1f}x)")

    print(f"\n💡 编排选择的黄金法则：")
    print(f"  • 监督者-工人：意图明确、任务独立的场景（最快最省）")
    print(f"  • 群体传递：  角色动态、无法预分类的场景（最灵活）")
    print(f"  • 层级式：    组织架构严格、有明确上下级关系的场景")
    print(f"  • 辩论模式：  高风险决策，需要共识支撑（最贵但最稳）")
    print(f"\n  ⚠️ 选拓扑不是选最快的，是选最适合你要解决的问题的。")
    print(f"  简单任务上辩论 = 浪费。高风险任务用监督者 = 冒险。")
    print(f"  工程直觉：先想清楚问题，再选编排模式。")


if __name__ == "__main__":
    main()
