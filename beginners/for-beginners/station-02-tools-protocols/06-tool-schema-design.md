# 让 Agent 选对工具：Schema 设计技巧

> 预计阅读：15分钟 | 难度：入门 | 前置条件：理解了工具调用基础
>
> 你将学到：工具名和描述怎么写才能让 Agent 选对——选错工具的代价比你想象的大得多。

---

## 🚗 开场翻车：30 个工具，62% 选不对

一个 Agent 注册表里有 30 个工具。用户说"帮我查一下张三的联系方式"。

Agent 选了 `search_contacts`。但应该选 `get_customer_details`。两个描述都是"查找人员信息"。

结果：Agent 用了错误的 API，查到了另一个"张三"。

Databricks 做过实验：**50 个含混描述的工具 → 选择准确率 62%。重写描述后 → 89%。**

---

## ✏️ 命名六条规则

| 规则 | 好 | 坏 |
|------|----|----|
| `snake_case` | `get_weather` | `GetWeather` |
| 动词-名词 | `search_contacts` | `contacts_search` |
| 无时态 | `get_weather` | `got_weather` |
| 稳定不重构 | 加新名字，不改老名字 | 直接改名（所有 Agent 全挂） |
| 命名空间前缀 | `notes_create`, `notes_search` | `create`, `search`（冲突） |
| 名字里不含参数 | `get_weather` + 参数 `city` | `get_weather_beijing` |

---

## 📝 描述模板——两句话就够了

```
Use when {什么时候用}. Do not use for {别用在什么地方}.
```

**"Do not use for"是区分相近工具的神来之笔。**

例子：

```
❌ 好：一个计算器工具
✅ 优：Use when performing arithmetic calculations.
       Do not use for unit conversions or currency exchange.
```

加了 `Do not use for` 后，Agent 不会把"汇率换算"扔给计算器。

**控制在 1024 字符以内。** OpenAI 严格模式下会截断超长的。

---

## 🧩 原子 vs 单体——工具拆多细

| | 原子工具 | 单体工具 |
|---|---|---|
| 样子 | 一个工具一个操作 | 一个工具干所有事 |
| 准确率 | 高（Agent 容易选对） | 低（Agent 选对了也得猜参数） |
| 维护 | 多但简单 | 少但复杂 |

**一句话：优先拆成原子工具。宁可工具多但每个功能清晰，不要一个万能工具靠参数猜意图。**

---

## 🏆 焊死在脑子里的东西

1. **工具名 = Agent 第一印象。** `snake_case`，动词-名词，稳定不改。

2. **描述 = "Use when X. Do not use for Y."** 两句话模式把选择准确率从 62% 拉到 89%。

3. **优先原子工具。** 一个工具一个功能，别做万能瑞士军刀。

4. **重命名是破坏性变更。** 有 Agent 依赖老名字。加新的不要改老的。

---

> **你以为 Agent 选错工具是 Agent 的问题——其实是你的描述写得不够好。"Use when / Do not use for"——就这两句话，能把选对率提高 27 个百分点。**
