# Claude Agent SDK：子 Agent 与生产级 Agent 开发

> **原文：** Claude Agent SDK  
> **预计时间：** ~20 分钟  

---

## 🧠 两层 SDK

| | Client SDK (`anthropic`) | Agent SDK (`claude-agent-sdk`) |
|---|---|---|
| 给的什么 | 原始 Messages API | 完整的 Agent 循环 |
| 谁管循环 | 你写 while 循环 | SDK 内置 |
| 谁管工具执行 | 你写 | SDK 内置 |
| 谁管会话 | 你自己存 | SDK 的会话存储 |

**Agent SDK = Claude Code 的循环作为库暴露给你。**

---

## 🔧 核心功能

### 1. 内置工具

开箱即用：文件读写、shell、grep、glob、网页获取。自定义工具通过标准 schema 注册。

### 2. 子 Agent（上下文隔离）

```python
# 并行为 20 个模块找测试文件
for module in modules:
    sub_agent = agent.spawn(f"找到 {module} 的测试文件")
# 每个子 Agent 独立上下文窗口，只有结果返回
```

**两个用途：** 并行化（同时干不同的活）+ 上下文隔离（子 Agent 的上下文垃圾不污染主 Agent）。

### 3. 会话存储

```python
session.append(session_id, message)  # 保存一轮
session.load(session_id)              # 恢复对话
session.list_sessions()               # 列出所有会话
```

子 Agent 的会话也级联管理。

---

## 🏆 焊死在脑子里的东西

1. **Client SDK = 原始 API。Agent SDK = 生产级 harness。**
2. **子 Agent = 并行化 + 上下文隔离。** 每个独立窗口，结果汇报给编排器。
3. **会话存储内置。** 崩溃后能恢复对话。

---

> **Claude Agent SDK 是 Claude Code 的内核。不需要你写循环、不需要你管工具执行——SDK 替你做了。子 Agent 是你的第一个多 Agent 模式。**
