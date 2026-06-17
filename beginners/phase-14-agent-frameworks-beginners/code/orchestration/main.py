"""
🔀 多 Agent 编排 · 初学者版

当你只有一个 Agent 时，它什么都自己干。
当你有多个 Agent 时，怎么分工？

这里演示三种最简单的编排方式：
  1. 经理模式 —— 一个 Agent 分配任务给其他人
  2. 流水线模式 —— Agent 排好队，一个接一个处理
  3. 投票模式 —— 多个 Agent 各自判断，少数服从多数
"""

from collections import Counter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模拟三个专家 Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify(text):
    """判断一个问题属于哪个领域"""
    t = text.lower()
    if any(w in t for w in ["退款", "退钱", "退货"]):
        return "退款"
    if any(w in t for w in ["bug", "报错", "闪退", "崩溃"]):
        return "报修"
    if any(w in t for w in ["价格", "报价", "多少钱"]):
        return "询价"
    return "通用"


# 三个"专家"，各自负责一个领域
EXPERTS = {
    "退款": lambda t: f"退款已处理：{t[:20]}",
    "报修": lambda t: f"故障已登记：{t[:20]}",
    "询价": lambda t: f"报价已发送：{t[:20]}",
    "通用": lambda t: f"已转接人工：{t[:20]}",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模式 1：经理模式（Supervisor）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def manager_mode(tasks):
    """一个经理 Agent 分类任务，派给对应的专家"""
    print("\n🔹 经理模式")
    for task in tasks:
        label = classify(task)
        result = EXPERTS[label](task)
        print(f"  经理 → 分配给「{label}」专家 → {result}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模式 2：流水线模式（Pipeline）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pipeline_mode(tasks):
    """Agent 排好队，任务从第一个传到最后一个，谁对了谁处理"""
    print("\n🔹 流水线模式")
    order = ["退款", "报修", "询价", "通用"]
    for task in tasks:
        for agent in order:
            if classify(task) == agent:
                result = EXPERTS[agent](task)
                print(f"  任务经过 {order} → 停在第{order.index(agent)+1}位「{agent}」: {result}")
                break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模式 3：投票模式（Voting）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def voting_mode(tasks):
    """三个 Agent 各自判断，投票决定最终结果"""
    print("\n🔹 投票模式")
    voters = ["Agent-A", "Agent-B", "Agent-C"]
    for task in tasks:
        votes = []
        for v in voters:
            label = classify(task)
            votes.append(label)
            print(f"  {v} 投票: 「{label}」")
        # 少数服从多数
        winner = Counter(votes).most_common(1)[0][0]
        count = Counter(votes).most_common(1)[0][1]
        result = EXPERTS[winner](task)
        print(f"  → 投票结果: 「{winner}」({count}/3 票) → {result}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 55)
    print("🔀 多 Agent 编排 · 初学者版")
    print("=" * 55)

    # 三个不同领域的问题
    tasks = [
        "我申请退款，订单号 INV-4711",
        "App 点登录就闪退",
        "50人的企业版多少钱？",
    ]

    print(f"\n📋 待处理（{len(tasks)} 条）：")
    for t in tasks:
        print(f"  • {t}")

    manager_mode(tasks)
    pipeline_mode(tasks)
    voting_mode(tasks)

    print(f"\n💡 怎么选？")
    print(f"   经理模式 → 任务明确、单一步骤（最快）")
    print(f"   流水线模式 → 任务需要多步处理")
    print(f"   投票模式 → 高风险决策、需要共识")
    print(f"   没有「最好」的模式，只有最适合你问题的模式。")
