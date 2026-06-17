# OpenAI Agents SDK：Handoff、护栏、追踪

> **原文：** OpenAI Agents SDK  
> **预计时间：** ~15 分钟  

---

## 🧠 五个原语

| 原语 | 做什么 |
|------|--------|
| **Agent** | LLM + 指令 + 工具 |
| **Handoff** | 委托给另一个 Agent（建模为工具 `transfer_to_xxx`） |
| **Guardrail** | 输入/输出/工具调用的验证 |
| **Session** | 跨轮次自动对话管理 |
| **Tracing** | 内置 span 追踪 |

---

## 🔑 Handoff = 路由

```python
# 分诊 Agent 通过工具调用路由到专家
triage_agent.handoffs = [refund_agent, sales_agent, support_agent]
# 模型看到 "transfer_to_refunds" 等工具
# 调用这些工具 = 切换活跃 Agent
```

---

## 🏆 关键点

1. **Handoff 是 OpenAl 多 Agent 的核心。** Agent 说"我不行，找那个能行的"。
2. **护栏三层：** 输入（防注入）、工具（防滥用）、输出（防泄露）。
3. **追踪默认开启。** 每次调用自动记录 span。

---

> **OpenAI Agents SDK 把多 Agent 简化为"Agent + 转交 + 护栏"。三个原语，生产级追踪内置。**
