# A2A 协议：Agent 和 Agent 怎么说话

> 预计阅读：20分钟 | 难度：进阶 | 前置条件：理解了 MCP 基础和多 Agent 概念
>
> 你将学到：A2A 协议是什么、跟 MCP 的区别、Agent Card、Task 生命周期。

---

## 🚗 开场翻车：两个 Agent 各说各话

一个客服 Agent 需要把"帮我写一份客户投诉分析报告"委托给一个写作 Agent。

在 A2A 协议出现之前，选项很惨：
- 写一个自定义 REST API——每个 Agent 配对都要写一次
- 两个 Agent 跑同一个框架——但一个是 Python 写的一个是 TypeScript 写的
- 用 MCP——不合适，MCP 是调工具用的，不是 Agent 之间对话用的

**MCP = Agent 怎么调工具。A2A = Agent 怎么跟 Agent 说话。**

---

## 🤔 MCP vs A2A：一句话区分

| | MCP | A2A |
|---|---|---|
| 谁跟谁 | Agent ↔ 工具/数据 | Agent ↔ Agent |
| 调用方看到什么 | 工具的描述和参数 | 另一个 Agent 的能力声明 |
| 被调用方内部 | 无内部状态 | 有自己的推理、记忆、状态 |
| 协议管理方 | Agentic AI Foundation (Linux 基金会) | Linux 基金会 |
| 什么时候用 | 查数据库、调 API、读写文件 | 委托复杂任务给另一个 Agent |

**A2A 被调用方是一个黑盒。** 调用 Agent 不知道被调用 Agent 内部怎么推理——只知道"给它一个任务，它会返回结果"。A2A 定义了"任务提交、状态追踪、结果获取"的标准方式。

---

## 📇 Agent Card：Agent 的"名片"

每个兼容 A2A 的 Agent 在 `/.well-known/agent.json` 发一张"名片"：

```json
{
  "name": "research-agent",
  "description": "总结论文，起草引用",
  "url": "https://research.example.com/a2a",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "总结论文",
      "description": "读取论文PDF，生成摘要",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

**发现方式：** 获取 Agent Card → 看它的 skills → 调用匹配的 skill。跟 MCP 的 `tools/list` 完全对应的设计。

---

## 📋 Task 生命周期

A2A 把每次委托建模为一个 Task，有自己的生命周期：

```
submitted → working → completed
                   → failed
                   → canceled
                   → rejected
                   → input-required → working → ...
```

- **submitted** — 调用 Agent 提交了任务
- **working** — 被调用 Agent 正在干活
- **input-required** — 被调用 Agent 需要更多信息，暂停等待
- **completed/failed/canceled/rejected** — 终态

调用 Agent 可以通过 SSE（Server-Sent Events）实时订阅状态变化，也可以定期轮询。

---

## 🎮 互动练习

两个 Agent：一个"代码编写 Agent"，一个"代码审查 Agent"。代码编写 Agent 写完代码后要委托审查。用 A2A 协议描述这个交互流程。

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **MCP = Agent 到工具。A2A = Agent 到 Agent。** 互补，不是竞争。

2. **Agent Card = A2A 的 `tools/list`。** 通过公开 URL 发现 Agent 的能力。

3. **Task 生命周期 = 标准的委托追踪。** submitted → working → completed/failed。中间可以有 input-required（等人类输入）。

4. **被调用 Agent 内部是黑盒。** 调用方只看到任务状态和最终输出——不知道内部推理过程。

---

## 📝 练习答案

```
1. 代码编写 Agent 获取审查 Agent 的 Card
   → 发现 skill "code_review"
2. 代码编写 Agent 提交 Task:
   → role: "user", parts: [text: "请审查这段代码", file: code.py]
   → 状态: submitted
3. 审查 Agent 开始工作
   → 状态: working
4. 审查 Agent 发现问题，需要更多上下文
   → 状态: input-required
5. 代码编写 Agent 提供额外信息
   → 状态: working
6. 审查 Agent 完成
   → 状态: completed
   → Artifacts: [name: "review", parts: [text: "发现3个问题..."]]
```

---

> **MCP 让 Agent 有手。A2A 让 Agent 有同事。有手可以自己干活，有同事可以把活派出去。两样都装上，才是完整的 Agent。**
