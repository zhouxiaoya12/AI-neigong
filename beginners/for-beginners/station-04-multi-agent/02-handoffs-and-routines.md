# 多 Agent 怎么交接工作：Handoff 与路由

> 预计阅读：20分钟 | 难度：进阶 | 前置条件：理解了多 Agent 编排
>
> 你将学到：Agent 之间怎么"传球"——OpenAI Swarm 的两个原语、为什么路由是最简单的多 Agent 模式、以及无状态的代价。

---

## 🚗 开场翻车：一个客户问了退款，Agent 发了销售报价

某公司用 LangGraph 搭了一个客服系统。三个 Agent：分诊、退款、销售。

客户说"我要退款"。分诊 Agent 应该路由给退款 Agent。

但分诊 Agent 选错了——它路由给了销售 Agent。销售 Agent 热情地回复了："您好！我们有最新的促销活动..."

**客户更生气了。不是因为 Agent 不聪明——是因为路由逻辑写在一个节点图里，改一次要改六个文件。**

---

## 🤔 最简单的多 Agent 模式：路由

OpenAI Swarm（2024 年 10 月）提出了一种极简的多 Agent 方案——只用两个概念：

### 原语 1：Routine（例程）

一个系统提示 + 一套工具。定义一个 Agent 的角色和能力。

```python
# 退款 Agent：只做退款
refund_agent = Agent(
    name="退款客服",
    instructions="你负责处理退款。收集订单号和退款原因。",
    functions=[lookup_order, process_refund]
)
```

### 原语 2：Handoff（路由）

一个工具——它不返回数据，返回另一个 Agent。运行时检测到 Agent 被返回，就切换活跃 Agent。

```python
# 分诊 Agent：只有一个工作——把人转给对的 Agent
def transfer_to_refunds():
    return refund_agent  # 关键：不返回文本，返回另一个 Agent

triage_agent = Agent(
    name="分诊",
    instructions="把用户转给合适的专家。",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support]
)
```

**整个多 Agent 编排 = LLM 选择调用哪个路由工具。** 没有状态机、没有节点图、没有 DSL。

---

## ✅ 为什么这个模式传播这么快

1. **API 极小。** 两个概念就够。学习成本 ≈ 零。
2. **复用模型已有的能力。** 工具调用的准确率已经是生产级的。
3. **没有状态机负担。** Agent 的提示写清楚了"什么时候路由给谁"——不需要画图。

---

## ⚠️ 代价：无状态

Swarm 在两个请求之间不保存任何东西。每次对话从零开始。

这意味着：
- Agent 间没有共享记忆
- 路由上下文只有对话历史
- 长会话的状态管理是你自己的事

OpenAI Agents SDK（2025 年 3 月）给 Swarm 加了状态管理——会话持久化、护栏、追踪。但路由原语保留了下来。

**简单总结：Swarm = 路由。Agents SDK = 路由 + 状态管理 + 护栏 + 追踪。**

---

## 🎮 互动练习

你要给一个医疗咨询系统设计路由拓扑。三个 Agent：分诊、全科、专科。用户说"我头疼"。写出分诊 Agent 的 instructions 和路由逻辑。

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **Handoff = 一个返回另一个 Agent 的工具。** 就这么简单。LLM 选路由 = LLM 调用工具。

2. **Swarm 只有两个概念：Routine + Handoff。** 比任何多 Agent 框架都简单。

3. **无状态是双刃剑。** 简单但短命。生产环境用 Agents SDK 加状态管理层。

4. **先试 Swarm 模式。** 如果任务能用"分诊 → 路由 → 专家"解决，就别上 LangGraph。

---

## 📝 练习答案

```python
def transfer_to_general():
    return general_agent

def transfer_to_specialist():
    return specialist_agent

triage_agent = Agent(
    name="分诊",
    instructions="""你是一个医疗分诊助手。
    - 常见症状（头疼、感冒、发烧）→ 转给全科
    - 需要专科判断的症状（持续疼痛、不明原因）→ 转给专科
    - 不确定 → 转给全科（全科判断是否需要专科）
    - 不要自己下诊断。""",
    functions=[transfer_to_general, transfer_to_specialist]
)
```

---

> **多 Agent 协作不需要复杂的编排引擎。有时候只需要一句话："如果你不知道，把球传给知道的人。"**
