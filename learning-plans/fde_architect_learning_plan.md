# FDE 工程师 & Agent 架构师完整学习计划

> 基于 ai-engineering-from-scratch 课程（503课/20阶段/~320小时）
> 聚焦 FDE（前沿部署工程师）和 Agent 架构师岗位需求
> 总计约 120 小时核心内容（从 503 课中精选）

---

## 岗位能力模型

### FDE（前沿部署工程师）核心能力
1. **LLM 工程** — 能用 LLM API 构建实际应用
2. **Agent 开发** — 理解 Agent 架构，能构建和调试
3. **工具集成** — MCP、Function Calling、工具协议
4. **生产部署** — 推理优化、成本控制、可观测性
5. **客户交付** — 能把 AI 系统部署到客户环境

### Agent 架构师核心能力
1. **Agent 设计** — 单 Agent、多 Agent、编排模式
2. **自主系统** — 长程 Agent、安全护栏、成本控制
3. **协议与标准** — MCP、A2A、OpenTelemetry
4. **可靠性工程** — 失败模式、评估、监控
5. **安全与对齐** — Prompt Injection 防御、对齐伪装检测

---

## 学习路径总览

```
已完成：
├── Phase 10 & 11: LLM 大模型（~43 小时） ✅
└── Phase 14: Agent Engineering（~42 小时） ✅

本次新增：
├── Phase 13: Tools & Protocols（~12 小时精选）
├── Phase 15: Autonomous Systems（~10 小时精选）
├── Phase 16: Multi-Agent & Swarms（~12 小时精选）
├── Phase 17: Infrastructure & Production（~10 小时精选）
└── Phase 18: Safety & Alignment（~6 小时精选）

总计新增：~50 小时
```

---

## 阶段五：工具与协议（~12 小时）

> 目标：掌握 Agent 与外部世界交互的标准协议

### ⭐⭐⭐ 必学：MCP Fundamentals
- 路径：`phases/13-tools-and-protocols/06-mcp-fundamentals/`
- 时间：~45 分钟
- 为什么重要：**MCP 是 2026 年工具集成的事实标准**。Hermes、Claude Code、Cursor 都支持
- 工程要点：
  - MCP 架构：Client（Agent） ↔ Server（工具提供者）
  - 三种传输方式：stdio、HTTP/SSE、WebSocket
  - 工具发现、资源访问、提示模板
  - **产出：理解 MCP 协议全貌**

### ⭐⭐⭐ 必学：Building an MCP Server
- 路径：`phases/13-tools-and-protocols/07-building-an-mcp-server/`
- 时间：~75 分钟
- 为什么重要：**FDE 的核心工作之一是为客户构建 MCP Server**
- 工程要点：
  - 工具定义：name、description、inputSchema
  - 资源暴露：文件、数据库、API
  - 错误处理和权限控制
  - **产出：可部署的 MCP Server**

### ⭐⭐⭐ 必学：Building an MCP Client
- 路径：`phases/13-tools-and-protocols/08-building-an-mcp-client/`
- 时间：~75 分钟
- 为什么重要：理解 Agent 如何连接和使用 MCP Server
- 工程要点：
  - 连接管理、工具调用、资源读取
  - 多 Server 编排
  - **产出：MCP Client 实现**

### ⭐⭐⭐ 必学：Function Calling Deep Dive
- 路径：`phases/13-tools-and-protocols/02-function-calling-deep-dive/`
- 时间：~75 分钟
- 为什么重要：**Agent 能力的基础**。不理解 Function Calling 就无法设计 Agent
- 工程要点：
  - JSON Schema 定义工具
  - 并行工具调用
  - 工具选择策略
  - **产出：Function Calling 完整实现**

### ⭐⭐⭐ 必学：A2A Protocol（Agent-to-Agent）
- 路径：`phases/13-tools-and-protocols/19-a2a-protocol/`
- 时间：~75 分钟
- 为什么重要：**Google 提出的 Agent 间通信协议**，多 Agent 系统的基础
- 工程要点：
  - Agent Card：能力声明
  - Task 协议：任务分发和结果收集
  - 流式通信
  - **产出：A2A 通信实现**

### ⭐⭐⭐ 必学：OpenTelemetry GenAI
- 路径：`phases/13-tools-and-protocols/20-opentelemetry-genai/`
- 时间：~75 分钟
- 为什么重要：**Agent 可观测性的标准协议**。没有可观测性就无法调试 Agent
- 工程要点：
  - Span：LLM 调用、工具调用、Agent 循环
  - Metrics：延迟、成本、成功率
  - Trace：完整调用链
  - **产出：Agent 监控实现**

### ⭐⭐ 推荐：MCP Security — Tool Poisoning
- 路径：`phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/`
- 时间：~45 分钟
- 为什么推荐：**MCP 安全的头号威胁**。FDE 必须理解

### ⭐⭐ 推荐：Tool Schema Design
- 路径：`phases/13-tools-and-protocols/05-tool-schema-design/`
- 时间：~45 分钟
- 为什么推荐：好的工具描述是 Agent 选对工具的关键

### ⭐⭐ 推荐：Skills and Agent SDKs
- 路径：`phases/13-tools-and-protocols/22-skills-and-agent-sdks/`
- 时间：~45 分钟
- 为什么推荐：Hermes Skills 的理论基础

---

## 阶段六：自主系统（~10 小时）

> 目标：构建可长时间运行、可控、可靠的 Agent

### ⭐⭐⭐ 必学：From Chatbots to Long-Horizon Agents
- 路径：`phases/15-autonomous-systems/01-long-horizon-agents/`
- 时间：~45 分钟
- 为什么重要：**2026 年 Claude Opus 4.6 能跑 14+ 小时的专家级任务**。长任务的一切假设都不同
- 工程要点：
  - METR Time Horizon：模型完成 50% 可靠性的任务所需人类专家时间
  - 能力翻倍时间：~7 个月
  - 长任务打破：上下文、信任、失败模式、成本、可观测性
  - **每步 99% 可靠 → 70 步后只剩 50%**

### ⭐⭐⭐ 必学：Durable Execution for Long-Running Agents
- 路径：`phases/15-autonomous-systems/12-durable-execution/`
- 时间：~60 分钟
- 为什么重要：**长时间运行的 Agent 需要持久化执行（故障恢复）**
- 工程要点：
  - 检查点：保存 Agent 状态
  - 恢复：从检查点继续
  - 持久化队列
  - **产出：可恢复的 Agent 执行引擎**

### ⭐⭐⭐ 必学：Action Budgets, Iteration Caps, Cost Governors
- 路径：`phases/15-autonomous-systems/13-cost-governors/`
- 时间：~60 分钟
- 为什么重要：**没有成本控制，一个失控循环可以烧掉一个月预算**
- 工程要点：
  - 动作预算：限制工具调用次数
  - 迭代上限：限制 Agent 循环次数
  - 成本断路器：达到阈值自动停止
  - **产出：成本控制系统**

### ⭐⭐⭐ 必学：Kill Switches, Circuit Breakers, Canary Tokens
- 路径：`phases/15-autonomous-systems/14-kill-switches-circuit-breakers/`
- 时间：~60 分钟
- 为什么重要：**Agent 安全的最后防线**
- 工程要点：
  - Kill Switch：紧急停止 Agent
  - Circuit Breaker：连续失败自动暂停
  - Canary Tokens：检测 Agent 是否做了不该做的事
  - **产出：安全防护系统**

### ⭐⭐⭐ 必学：HITL — Propose-Then-Commit
- 路径：`phases/15-autonomous-systems/15-propose-then-commit/`
- 时间：~60 分钟
- 为什么重要：**高风险操作的人类审批机制**
- 工程要点：
  - 提议：Agent 先说要做什么
  - 审批：人类确认或修改
  - 执行：Agent 执行批准的操作
  - **产出：人类在环审批系统**

### ⭐⭐ 推荐：Claude Code Permission Modes and Auto Mode
- 路径：`phases/15-autonomous-systems/10-claude-code-permission-modes/`
- 时间：~45 分钟
- 为什么推荐：理解 Hermes 的权限模式设计

### ⭐⭐ 推荐：Browser Agents and Indirect Prompt Injection
- 路径：`phases/15-autonomous-systems/11-browser-agents-prompt-injection/`
- 时间：~45 分钟
- 为什么推荐：浏览器 Agent 的安全风险

---

## 阶段七：多 Agent 系统（~12 小时）

> 目标：设计和构建多 Agent 协作系统

### ⭐⭐⭐ 必学：Why Multi-Agent
- 路径：`phases/16-multi-agent-and-swarms/01-why-multi-agent/`
- 时间：~45 分钟
- 为什么重要：**不是所有问题都需要多 Agent**。先理解为什么，再学怎么做
- 工程要点：
  - 单 Agent 的局限：上下文、专注度、并行性
  - 多 Agent 的优势：独立上下文、专注 prompt、并行执行
  - 何时用多 Agent vs 单 Agent + Workflow

### ⭐⭐⭐ 必学：Supervisor / Orchestrator-Worker Pattern
- 路径：`phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/`
- 时间：~75 分钟
- 为什么重要：**Anthropic Research 系统的核心模式，比单 Agent 提升 90.2%**
- 工程要点：
  - 三个赢的原因：独立上下文、专注 prompt、并行执行
  - BrowseComp 80% 方差由 token 用量解释
  - 工程教训：规模匹配查询复杂度、token 成本 ~15x
  - **产出：Supervisor-Worker 系统**

### ⭐⭐⭐ 必学：Role Specialization — Planner / Critic / Executor / Verifier
- 路径：`phases/16-multi-agent-and-swarms/08-role-specialization/`
- 时间：~75 分钟
- 为什么重要：**角色分离是多 Agent 系统设计的核心**
- 工程要点：
  - Planner：制定计划
  - Executor：执行任务
  - Critic：审查结果
  - Verifier：验证正确性
  - **产出：四角色 Agent 系统**

### ⭐⭐⭐ 必学：A2A — The Agent-to-Agent Protocol
- 路径：`phases/16-multi-agent-and-swarms/12-a2a-protocol/`
- 时间：~75 分钟
- 为什么重要：**Google 提出的 Agent 间通信标准**
- 工程要点：
  - Agent Card：能力声明
  - Task 协议：任务生命周期
  - 流式通信和推送通知
  - **产出：A2A 通信实现**

### ⭐⭐ 推荐：Handoffs and Routines (Stateless Orchestration)
- 路径：`phases/16-multi-agent-and-swarms/11-handoffs-and-routines/`
- 时间：~60 分钟
- 为什么推荐：OpenAI Agents SDK 的核心模式

### ⭐⭐ 推荐：Failure Modes — MAST, Groupthink, Monoculture
- 路径：`phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink/`
- 时间：~75 分钟
- 为什么推荐：多 Agent 系统的特有失败模式

### ⭐⭐ 推荐：Production Scaling — Queues, Checkpoints, Durability
- 路径：`phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints/`
- 时间：~75 分钟
- 为什么推荐：多 Agent 系统的生产部署

---

## 阶段八：基础设施与生产（~10 小时）

> 目标：把 Agent 系统部署到生产环境

### ⭐⭐⭐ 必学：Inference Metrics — TTFT, TPOT, ITL, Goodput, P99
- 路径：`phases/17-infrastructure-and-production/08-inference-metrics/`
- 时间：~60 分钟
- 为什么重要：**没有指标就无法优化**
- 工程要点：
  - TTFT：首 token 延迟
  - TPOT：每 token 输出时间
  - ITL：token 间延迟
  - Goodput：有效吞吐量
  - P99：99 分位延迟
  - **产出：推理监控仪表盘**

### ⭐⭐⭐ 必学：AI Gateways — LiteLLM, Portkey, Kong, Bifrost
- 路径：`phases/17-infrastructure-and-production/19-ai-gateways/`
- 时间：~60 分钟
- 为什么重要：**生产环境的 LLM 统一接入层**
- 工程要点：
  - 多 Provider 统一接口
  - 负载均衡、故障转移
  - 缓存、限流、监控
  - **产出：AI Gateway 配置**

### ⭐⭐⭐ 必学：Model Routing as a Cost-Reduction Primitive
- 路径：`phases/17-infrastructure-and-production/16-model-routing/`
- 时间：~60 分钟
- 为什么重要：**不同任务用不同模型，成本可降 50%+**
- 工程要点：
  - 简单任务 → 小模型（Claude Haiku、GPT-4o-mini）
  - 复杂任务 → 大模型（Claude Opus、GPT-5）
  - 路由策略：基于任务类型、复杂度、成本
  - **产出：模型路由系统**

### ⭐⭐ 推荐：Prompt Caching and Semantic Caching Economics
- 路径：`phases/17-infrastructure-and-production/14-prompt-caching/`
- 时间：~60 分钟
- 为什么推荐：缓存可降低 30-50% API 成本

### ⭐⭐ 推荐：Self-Hosted Serving Selection
- 路径：`phases/17-infrastructure-and-production/28-self-hosted-serving-selection/`
- 时间：~45 分钟
- 为什么推荐：llama.cpp、Ollama、vLLM 选型指南

### ⭐⭐ 推荐：Security — Secrets, PII Scrubbing, Audit Logs
- 路径：`phases/17-infrastructure-and-production/25-security-secrets-pii/`
- 时间：~60 分钟
- 为什么推荐：生产环境安全合规

---

## 阶段九：安全与对齐（~6 小时精选）

> 目标：理解 Agent 安全风险和对齐问题

### ⭐⭐⭐ 必学：Indirect Prompt Injection
- 路径：`phases/18-ethics-safety-alignment/15-indirect-prompt-injection/`
- 时间：~75 分钟
- 为什么重要：**Agent 安全的头号威胁**。工具输出是不可信输入
- 工程要点：
  - 间接 Prompt Injection：PDF/网页中嵌入恶意指令
  - 信任边界：只有用户的直接指令才算权限
  - 防御策略
  - **产出：Prompt Injection 防御系统**

### ⭐⭐⭐ 必学：Alignment Faking
- 路径：`phases/18-ethics-safety-alignment/09-alignment-faking/`
- 时间：~60 分钟
- 为什么重要：**Anthropic 2024 年研究：12-78% 的模型会伪装对齐**
- 工程要点：
  - 对齐伪装：模型在测试中表现更安全
  - 评估-上下文博弈
  - 部署行为差异
  - **产出：理解对齐伪装风险**

### ⭐⭐ 推荐：Red-Teaming — PAIR & Automated Attacks
- 路径：`phases/18-ethics-safety-alignment/12-red-teaming-pair/`
- 时间：~75 分钟
- 为什么推荐：了解如何测试 Agent 安全性

### ⭐⭐ 推荐：Sleeper Agents — Persistent Deception
- 路径：`phases/18-ethics-safety-alignment/07-sleeper-agents/`
- 时间：~60 分钟
- 为什么推荐：理解 Agent 的潜在风险

---

## 完整学习路径

```
Phase 10 & 11: LLM 大模型（~43 小时） ✅ 已完成
        ↓
Phase 14: Agent Engineering（~42 小时） ✅ 已完成
        ↓
Phase 13: Tools & Protocols（~12 小时） ← 本次新增
        ↓
Phase 15: Autonomous Systems（~10 小时） ← 本次新增
        ↓
Phase 16: Multi-Agent & Swarms（~12 小时） ← 本次新增
        ↓
Phase 17: Infrastructure & Production（~10 小时） ← 本次新增
        ↓
Phase 18: Safety & Alignment（~6 小时） ← 本次新增
```

---

## 关键能力矩阵

| 能力 | 对应阶段 | FDE | 架构师 |
|------|---------|-----|--------|
| LLM 工程 | P10, P11 | ⭐⭐⭐ | ⭐⭐⭐ |
| Agent 开发 | P14 | ⭐⭐⭐ | ⭐⭐⭐ |
| MCP 协议 | P13 | ⭐⭐⭐ | ⭐⭐⭐ |
| Function Calling | P13 | ⭐⭐⭐ | ⭐⭐⭐ |
| 多 Agent 设计 | P16 | ⭐⭐ | ⭐⭐⭐ |
| 自主系统 | P15 | ⭐⭐ | ⭐⭐⭐ |
| 生产部署 | P17 | ⭐⭐⭐ | ⭐⭐ |
| 安全与对齐 | P18 | ⭐⭐ | ⭐⭐⭐ |
| 可观测性 | P13, P17 | ⭐⭐⭐ | ⭐⭐ |
| 成本控制 | P15, P17 | ⭐⭐⭐ | ⭐⭐ |

---

## 代码路径速查

| 想实现什么 | 代码在哪 |
|-----------|----------|
| MCP Server | `phases/13-tools-and-protocols/07-building-an-mcp-server/code/` |
| MCP Client | `phases/13-tools-and-protocols/08-building-an-mcp-client/code/` |
| Function Calling | `phases/13-tools-and-protocols/02-function-calling-deep-dive/code/` |
| A2A 通信 | `phases/13-tools-and-protocols/19-a2a-protocol/code/` |
| OpenTelemetry | `phases/13-tools-and-protocols/20-opentelemetry-genai/code/` |
| 成本控制 | `phases/15-autonomous-systems/13-cost-governors/code/` |
| 人类在环 | `phases/15-autonomous-systems/15-propose-then-commit/code/` |
| Supervisor-Worker | `phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/code/` |
| 角色专业化 | `phases/16-multi-agent-and-swarms/08-role-specialization/code/` |
| AI Gateway | `phases/17-infrastructure-and-production/19-ai-gateways/code/` |

---

## 课程来源

- 项目：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- 总课时：503 课 / 20 阶段 / ~320 小时
- 本文档精选：~120 小时（含已学 ~85 小时 + 新增 ~50 小时）
