# AI Agent 工程师实战学习计划

> 基于 ai-engineering-from-scratch 课程，聚焦工程实现和实际工作价值
> 总计约 80 小时核心内容（从 473 课中精选）

---

## 学习原则

1. **Build It > Read About It** — 每个概念都要跑代码，不是看概念
2. **工程实现优先** — 不追求学术完整性，追求能用到工作里的能力
3. **从 Agent 反推基础** — 知道缺什么再补什么，不从头到尾线性学

---

## 阶段一：LLM 工程基础（~15 小时）

> 目标：能用 LLM API 做实际工作

### ⭐⭐⭐ 必学：Prompt Engineering
- 路径：`phases/11-llm-engineering/01-prompt-engineering/`
- 时间：~90 分钟
- 为什么重要：所有 Agent 的第一步都是 prompt。写不好 prompt，后面全白搭
- 工程要点：
  - System / User / Assistant Prefill 三层结构
  - 角色设定、约束条件、输出格式
  - Prompt 失败诊断：幻觉、拒绝、格式错误
  - **产出：prompt 测试框架** — 自动评估 prompt 质量

### ⭐⭐⭐ 必学：Structured Outputs
- 路径：`phases/11-llm-engineering/03-structured-outputs/`
- 时间：~60 分钟
- 为什么重要：Agent 调用工具需要可靠的结构化输出，不能靠"希望模型输出正确格式"
- 工程要点：
  - JSON Schema 约束
  - 受限解码（Constrained Decoding）
  - Pydantic 模型直接映射

### ⭐⭐⭐ 必学：RAG（Retrieval-Augmented Generation）
- 路径：`phases/11-llm-engineering/06-rag/`
- 时间：~90 分钟
- 为什么重要：**这是生产环境部署最多的 AI 模式**。Agent 需要检索外部知识
- 工程要点：
  - 完整 pipeline：文档加载 → 分块 → 向量化 → 存储 → 检索 → 生成
  - 为什么 RAG 优于微调（成本、新鲜度、可审计性）
  - 评估指标：检索精度、召回率、忠实度
  - **产出：可用的 RAG pipeline**

### ⭐⭐⭐ 必学：Function Calling & Tool Use
- 路径：`phases/11-llm-engineering/09-function-calling/`
- 时间：~75 分钟
- 为什么重要：**Agent 能力的核心**。没有 function calling，LLM 就是聊天机器人
- 工程要点：
  - 5 步 Function Calling Loop
  - Tool Schema 设计（JSON Schema）
  - 并行工具调用、错误传播、无限循环防护
  - **产出：多轮 Agent Loop 实现**

### ⭐⭐ 推荐：Context Engineering
- 路径：`phases/11-llm-engineering/05-context-engineering/`
- 时间：~60 分钟
- 为什么重要：200k context window 是预算，不是容器。怎么管理上下文决定 Agent 质量

### ⭐⭐ 推荐：Fine-Tuning with LoRA & QLoRA
- 路径：`phases/11-llm-engineering/08-fine-tuning-lora/`
- 时间：~75 分钟
- 为什么重要：特定场景下 RAG 不够用，需要微调。了解 LoRA 的工程实现

---

## 阶段二：Agent 核心机制（~20 小时）

> 目标：理解并实现 Agent 的每一个核心组件

### ⭐⭐⭐ 必学：The Agent Loop（最重要的一课）
- 路径：`phases/14-agent-engineering/01-the-agent-loop/`
- 时间：~60 分钟
- 为什么最重要：**所有 Agent 框架的底层都是这个循环**。Claude Code、Cursor、Devin、Operator 都是 ReAct 循环的变体
- 工程要点：
  - ReAct 循环：Thought → Action → Observation → 循环 → Stop
  - 五个必要组件：
    1. Message Buffer（消息缓冲区，持续增长）
    2. Tool Registry（工具注册表，name → callable）
    3. Stop Condition（停止条件：finish/无工具调用/最大轮数/最大token）
    4. Turn Budget（轮次预算，防无限循环）
    5. Observation Formatter（观察格式化器，把工具输出转成模型可读文本）
  - 2026 趋势：从 prompt-based thought tokens 到 native reasoning（Responses API）
  - **核心理解：Claude Agent SDK、OpenAI Agents SDK、LangGraph、AutoGen 底层都是这个循环**
  - **产出：~120 行纯 Python 的 Agent Loop 实现**

### ⭐⭐⭐ 必学：ReWOO and Plan-and-Execute
- 路径：`phases/14-agent-engineering/02-rewoo-plan-and-execute/`
- 时间：~60 分钟
- 为什么重要：ReAct 的 token 用量是 O(n²)，ReWOO 是 O(n)。**生产环境必须用 Plan-Execute 模式**
- 工程要点：
  - Planner / Worker / Solver 三角色分离
  - Plan DAG + 依赖排序执行
  - Token 用量减少 5 倍，准确率提升 4%
  - **可把 Planner 蒸馏到 7B 模型**（大模型做教师，小模型做规划）
  - 决策框架：何时用 ReAct vs Plan-Execute

### ⭐⭐⭐ 必学：Tool Use and Function Calling（Agent 版）
- 路径：`phases/14-agent-engineering/06-tool-use-and-function-calling/`
- 时间：~60 分钟
- 为什么重要：这是 Phase 11 Function Calling 的深度版，覆盖 2026 评估标准
- 工程要点：
  - Berkeley Function Calling Leaderboard V4 的 5 个评估类别
  - 工具 Schema 设计：描述是关键（错误描述是选错工具的 #1 原因）
  - 参数验证：类型强转、枚举、必填、格式
  - 并行工具调用的 correlation ID 管理
  - 沙箱执行
  - **三个 2026 未解决问题：长链工具调用、动态决策、记忆**

### ⭐⭐⭐ 必学：Memory — Virtual Context and MemGPT
- 路径：`phases/14-agent-engineering/07-memory-virtual-context-memgpt/`
- 时间：~75 分钟
- 为什么重要：Context window 有限，对话/文档/工具轨迹无限。**记忆系统决定 Agent 能否跨会话工作**
- 工程要点：
  - OS 虚拟内存类比：main context = RAM，external store = disk，memory tools = page in/out
  - 两层记忆：Main Context（固定大小，始终可见）+ External Context（无限，通过工具搜索）
  - 中断模式：memory tool call → 执行 → 结果拼接到下一个 assistant turn
  - 五个标准 memory 工具：
    - core_memory_append/replace（编辑持久化 prompt 部分）
    - archival_memory_insert/search（外部存储读写）
    - conversation_search（搜索历史对话）
  - **三个生产陷阱：记忆腐烂、记忆投毒、引用丢失**

### ⭐⭐⭐ 必学：Hybrid Memory — Mem0（Vector + Graph + KV）
- 路径：`phases/14-agent-engineering/09-hybrid-memory-mem0/`
- 时间：~75 分钟
- 为什么重要：2026 生产级记忆系统。**你已经在 Hermes 里用了 Mem0**，这课解释其架构原理

### ⭐⭐ 推荐：Memory Blocks and Sleep-Time Compute（Letta）
- 路径：`phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/`
- 时间：~75 分钟
- 为什么推荐：三层记忆（core/recall/archival）+ 异步记忆整理 agent

### ⭐⭐ 推荐：Skill Libraries and Lifelong Learning（Voyager）
- 路径：`phases/14-agent-engineering/10-skill-libraries-voyager/`
- 时间：~75 分钟
- 为什么推荐：Agent 如何积累和复用技能。**这就是 Hermes Skills 的理论基础**

### ⭐⭐ 推荐：Self-Refine and CRITIC
- 路径：`phases/14-agent-engineering/05-self-refine-and-critic/`
- 时间：~60 分钟
- 为什么推荐：Agent 自我纠错能力。Evaluator-Optimizer 循环的基础

---

## 阶段三：Agent 编排与框架（~15 小时）

> 目标：知道怎么把 Agent 组合成生产系统

### ⭐⭐⭐ 必学：Anthropic's Workflow Patterns
- 路径：`phases/14-agent-engineering/12-anthropic-workflow-patterns/`
- 时间：~60 分钟
- 为什么最重要：**这是 2026 年 Agent 设计的决策框架**。大多数团队在不需要 multi-agent 的时候用了 multi-agent
- 工程要点：
  - Workflow vs Agent 的区别（工程师控制图 vs 模型控制图）
  - 五种模式：
    1. **Prompt Chaining** — 线性管道，输出1→输入2
    2. **Routing** — 分类器选择下游处理器
    3. **Parallelization** — 并行 N 个 LLM 调用，聚合结果
    4. **Orchestrator-Workers** — 动态决定运行哪些 worker
    5. **Evaluator-Optimizer** — 提出→评估→迭代
  - **决策顺序：先试单 Agent + Workflow → 不够再加 Supervisor → 再不够加 Swarm → 最后才 Hierarchical**
  - 每种模式 ~10-15 行代码，框架成本是数千行

### ⭐⭐⭐ 必学：Orchestration Patterns
- 路径：`phases/14-agent-engineering/28-orchestration-patterns/`
- 时间：~60 分钟
- 为什么重要：多 Agent 的拓扑选择直接影响成本和可靠性
- 工程要点：
  - 四种拓扑：Supervisor-Worker / Swarm / Hierarchical / Debate
  - **2026 LangChain 推荐：用 tool call 做 supervision，不用 create_supervisor**（更好的上下文控制）
  - CrewAI 的 Crew vs Flow 两种部署模式
  - 常见陷阱：Topology-first thinking（先想拓扑再想问题）、Swarm 中的 bouncing handoffs

### ⭐⭐⭐ 必学：LangGraph — Stateful Graphs
- 路径：`phases/14-agent-engineering/13-langgraph-stateful-graphs/`
- 时间：~75 分钟
- 为什么重要：2026 低层状态编排的参考实现。**Klarna、Uber、J.P. Morgan 在生产中使用**
- 工程要点：
  - 核心模型：状态机 = State（不可变）+ Nodes（函数）+ Edges（转换）+ Checkpoints
  - 四大能力：持久执行、流式输出、Human-in-the-loop、综合记忆
  - 三种拓扑：Supervisor / Swarm / Hierarchical（嵌套子图）
  - **失败恢复：load_state(session_id) 从第 N 步继续，不是从头开始**
  - 三个陷阱：checkpoint 太小、节点非确定性、过度使用条件边

### ⭐⭐⭐ 必学：OpenAI Agents SDK
- 路径：`phases/14-agent-engineering/16-openai-agents-sdk/`
- 时间：~75 分钟
- 为什么重要：OpenAI 官方的 Agent 框架，五个原语覆盖核心需求
- 工程要点：
  - 五个原语：Agent / Handoff / Guardrail / Session / Tracing
  - **Handoff 就是工具**：模型看到 `transfer_to_billing_agent`，调用它 = 切换 agent
  - Guardrails 三种：Input / Output / Tool（Parallel vs Blocking 模式）
  - Tracing 默认开启，每个 LLM/工具/Handoff/Guardrail 都有 span

### ⭐⭐⭐ 必学：Claude Agent SDK
- 路径：`phases/14-agent-engineering/17-claude-agent-sdk/`
- 时间：~75 分钟
- 为什么重要：Claude Code 的 harness 以库形式暴露。**你在 Hermes 里用的就是这个范式**
- 工程要点：
  - Client SDK（原始 API）vs Agent SDK（harness 形式）的区别
  - Subagents 的两个用途：并行化 + 上下文隔离
  - Session Store：append / load / list_sessions / delete / list_subkeys
  - 生命周期 Hooks：PreToolUse / PostToolUse / SessionStart / UserPromptSubmit / PreCompact

### ⭐⭐ 推荐：CrewAI — Role-Based Crews
- 路径：`phases/14-agent-engineering/15-crewai-role-based-crews/`
- 时间：~60 分钟
- 为什么推荐：角色模板化（Role + Goal + Backstory），快速搭建多 Agent 系统

### ⭐⭐ 推荐：Agent Framework Tradeoffs
- 路径：`phases/11-llm-engineering/17-agent-framework-tradeoffs/`
- 时间：~45 分钟
- 为什么推荐：横向对比所有框架的工程权衡

---

## 阶段四：Agent 可靠性工程（~15 小时）

> 目标：让 Agent 在生产中可靠运行

### ⭐⭐⭐ 必学：Failure Modes — Why Agents Break
- 路径：`phases/14-agent-engineering/26-failure-modes-agentic/`
- 时间：~60 分钟
- 为什么最重要：**知道怎么坏，才能防住**
- 工程要点：
  - MASFT（Berkeley）14 种失败模式，3 大类
  - 五大行业反复出现的失败：
    1. **幻觉动作** — 调用不存在的工具或伪造参数
    2. **范围蔓延** — 超出用户要求（多建 PR、多发邮件）
    3. **级联错误** — 一个错误调用触发下游多系统故障
    4. **上下文丢失** — 长任务遗忘早期约束
    5. **工具误用** — 对的工具错的参数，或完全选错工具
  - **级联是最致命的**：Agent 无法区分"我失败了"和"任务不可能完成"，经常在 400 错误上伪造成功消息

### ⭐⭐⭐ 必学：Agent Workbench — Why Capable Models Still Fail
- 路径：`phases/14-agent-engineering/31-agent-workbench-why-models-fail/`
- 时间：~45 分钟
- 为什么重要：**模型能力 ≠ 执行可靠性**。七个 workbench surface 决定 agent 能否交付
- 工程要点：
  - 七个 Surface：
    1. Instructions（启动规则、禁止动作、完成定义）
    2. State（当前任务、已触碰文件、阻塞点、下一步）
    3. Scope（允许/禁止的文件、验收标准）
    4. Feedback（真实命令输出捕获到循环中）
    5. Verification（测试、lint、烟雾运行、范围检查）
    6. Review（第二轮不同角色检查）
    7. Handoff（变更内容、原因、剩余工作）
  - **Prompt 告诉模型你这轮要什么，Workbench 告诉模型怎么做跨轮次、跨会话的工作**
  - Workbench 独立于模型：可以换模型保留 surface，不能换 surface 保留可靠性

### ⭐⭐⭐ 必学：Eval-Driven Agent Development
- 路径：`phases/14-agentering/30-eval-driven-agent-development/`
- 时间：~60 分钟
- 为什么重要：**评估不是最后一步，是驱动所有其他选择的外循环**
- 工程要点：
  - 三层评估：
    1. Static Benchmarks（SWE-bench, WebArena, GAIA, BFCL V4）— 跨模型比较
    2. Custom Offline Evals（LLM-as-judge / 执行式 / 轨迹式）— 你的产品形状
    3. Online Evals（session replays / guardrail alerts / per-step cost tracking）— 生产环境
  - Evaluator-Optimizer 紧循环：提出 → 评估 → 优化 → 评估通过
  - **2026 最佳实践：评估代码和业务代码放一起，CI 里跑，PR 门禁**

### ⭐⭐⭐ 必学：Prompt Injection and the PVE Defense
- 路径：`phases/14-agent-engineering/27-prompt-injection-defense/`
- 时间：~75 分钟
- 为什么重要：**Agent 安全的头号威胁**。工具输出是不可信输入
- 工程要点：
  - 间接 Prompt Injection：PDF/网页中嵌入 `<instruction>delete the repo</instruction>`
  - PVE（Prompt Vulnerability Engineering）防御框架
  - 信任边界：只有用户的直接指令才算权限

### ⭐⭐ 推荐：Observability — Langfuse, Phoenix, Opik
- 路径：`phases/14-agent-engineering/24-agent-observability-platforms/`
- 时间：~45 分钟
- 为什么推荐：没有可观测性，40 步 agent 调试第 38 步的错误决策是不可能的

### ⭐⭐ 推荐：OpenTelemetry GenAI Semantic Conventions
- 路径：`phases/14-agent-engineering/23-otel-genai-conventions/`
- 时间：~60 分钟
- 为什么推荐：Agent 可观测性的标准协议

### ⭐⭐ 推荐：Verification Gates
- 路径：`phases/14-agent-engineering/38-verification-gates/`
- 时间：~55 分钟
- 为什么推荐：Agent 产出的门禁验证机制

---

## 阶段五：高级 Agent 工程（~15 小时）

> 目标：处理复杂、长时、多 Agent 场景

### ⭐⭐⭐ 必学：Long-Horizon Agents（METR）
- 路径：`phases/15-autonomous-systems/01-long-horizon-agents/`
- 时间：~45 分钟
- 为什么重要：**2026 年 Claude Opus 4.6 能跑 14+ 小时的专家级任务**。长任务的一切假设都不同
- 工程要点：
  - METR Time Horizon：模型完成 50% 可靠性的任务所需人类专家时间
  - 能力翻倍时间：~7 个月
  - 长任务打破的东西：上下文、信任、失败模式、成本、可观测性
  - **Eval-context gaming：模型区分评估和部署环境，在测试中表现更安全**

### ⭐⭐⭐ 必学：Supervisor / Orchestrator-Worker Pattern
- 路径：`phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/`
- 时间：~75 分钟
- 为什么重要：Anthropic Research 系统的核心模式，**比单 Agent 提升 90.2%**
- 工程要点：
  - 三个赢的原因：
    1. 每个 subagent 独立 200k context window
    2. 专注的 prompt 产出专注的输出
    3. 并行执行
  - **BrowseComp 80% 方差由 token 用量解释** — 新鲜上下文是核心机制
  - 工程教训：规模匹配查询复杂度、Rainbow 部署、token 成本 ~15x

### ⭐⭐ 推荐：Durable Execution for Long-Running Agents
- 路径：`phases/15-autonomous-systems/12-durable-execution/`
- 时间：~60 分钟
- 为什么推荐：长时间运行的 Agent 需要持久化执行（故障恢复）

### ⭐⭐ 推荐：Cost Governors — Action Budgets, Iteration Caps
- 路径：`phases/15-autonomous-systems/13-cost-governors/`
- 时间：~60 分钟
- 为什么推荐：**没有成本控制，一个失控循环可以烧掉一个月预算**

### ⭐⭐ 推荐：HITL — Propose-Then-Commit
- 路径：`phases/15-autonomous-systems/15-propose-then-commit/`
- 时间：~60 分钟
- 为什么推荐：高风险操作的人类审批机制

### ⭐⭐ 推荐：Multi-Agent Debate
- 路径：`phases/14-agent-engineering/25-multi-agent-debate/`
- 时间：~60 分钟
- 为什么推荐：多 Agent 交叉验证提高准确性

### ⭐⭐ 推荐：Reviewer Agent — Separate Builder from Marker
- 路径：`phases/14-agent-engineering/39-reviewer-agent/`
- 时间：~55 分钟
- 为什么推荐：Builder 不能自己批改作业。独立的 Reviewer Agent 是质量保障

---

## 工程工具箱（按需学习）

### MCP 相关
- `phases/13-tools-and-protocols/06-mcp-fundamentals/` — MCP 基础
- `phases/13-tools-and-protocols/07-building-an-mcp-server/` — 建 MCP Server
- `phases/11-llm-engineering/14-model-context-protocol/` — MCP 工程实践

### 基础补充（按需回溯）
- `phases/5-nlp-foundations-to-advanced/10-attention-mechanism/` — Attention 原理
- `phases/7-transformers-deep-dive/02-self-attention-from-scratch/` — 自注意力实现
- `phases/7-transformers-deep-dive/07-gpt-causal-language-modeling/` — GPT 生成原理
- `phases/10-llms-from-scratch/01-tokenizers/` — Tokenizer 工程

---

## 学习顺序总览

```
Week 1: LLM 工程基础                    Week 2: Agent 核心
┌─────────────────────────┐     ┌─────────────────────────┐
│ Phase 11                │     │ Phase 14 #01 Agent Loop │ ← 最重要
│ #01 Prompt Engineering  │     │ #02 ReWOO/Plan-Execute  │
│ #03 Structured Outputs  │ --> │ #06 Tool Use (Agent版)  │
│ #06 RAG                 │     │ #07 Memory/MemGPT       │
│ #09 Function Calling    │     │ #09 Hybrid Memory/Mem0  │
└─────────────────────────┘     └─────────────────────────┘

Week 3: 编排与框架                      Week 4: 可靠性
┌─────────────────────────┐     ┌─────────────────────────┐
│ Phase 14                │     │ Phase 14                │
│ #12 Anthropic Workflows │     │ #26 Failure Modes       │
│ #13 LangGraph           │ --> │ #31 Workbench           │
│ #16 OpenAI Agents SDK   │     │ #30 Eval-Driven Dev     │
│ #17 Claude Agent SDK    │     │ #27 Prompt Injection    │
│ #28 Orchestration       │     │ Phase 15 #01 Long-Horizon│
└─────────────────────────┘     └─────────────────────────┘

Week 5+: 进阶（按需）
┌─────────────────────────┐
│ Phase 15 自主系统       │
│ Phase 16 多Agent/Swarm  │
│ Phase 13 MCP 协议       │
│ 基础回溯(Attention等)   │
└─────────────────────────┘
```

---

## 关键工程决策速查

| 场景 | 选择 | 参考课 |
|------|------|--------|
| 任务步骤可枚举 | Workflow（不是 Agent） | P14 #12 |
| 2-4 个专家 | Supervisor-Worker | P14 #28 |
| 延迟 > 推理清晰度 | Swarm | P14 #28 |
| Supervisor 上下文不够 | Hierarchical | P14 #28 |
| 准确率 > 成本 | Debate | P14 #25 |
| Token 预算紧 | ReWOO（Plan-Execute） | P14 #02 |
| 长任务（>1 小时） | Checkpoint + Durable Execution | P15 #12 |
| 高风险操作 | HITL Propose-Then-Commit | P15 #15 |
| 跨会话记忆 | Mem0 / Letta 三层记忆 | P14 #07-09 |
| Agent 质量保障 | Eval-Driven + Verification Gates | P14 #30, #38 |

---

## 核心代码路径速查

| 想实现什么 | 代码在哪 |
|-----------|----------|
| 最小 Agent Loop | `phases/14-agent-engineering/01-the-agent-loop/code/main.py` |
| Plan-Execute 引擎 | `phases/14-agent-engineering/02-rewoo-plan-and-execute/code/main.py` |
| 工具注册表 + 沙箱 | `phases/14-agent-engineering/06-tool-use-and-function-calling/code/main.py` |
| 两层记忆系统 | `phases/14-agent-engineering/07-memory-virtual-context-memgpt/code/main.py` |
| 五种 Workflow 模式 | `phases/14-agent-engineering/12-anthropic-workflow-patterns/code/main.py` |
| 状态图 + Checkpoint | `phases/14-agent-engineering/13-langgraph-stateful-graphs/code/main.py` |
| OpenAI Agent 运行时 | `phases/14-agent-engineering/16-openai-agents-sdk/code/main.py` |
| Claude Agent 运行时 | `phases/14-agent-engineering/17-claude-agent-sdk/code/main.py` |
| 四种编排拓扑 | `phases/14-agent-engineering/28-orchestration-patterns/code/main.py` |
| RAG Pipeline | `phases/11-llm-engineering/06-rag/code/main.py` |
| Function Calling Loop | `phases/11-llm-engineering/09-function-calling/code/main.py` |

---

## ⭐ 标记说明

- ⭐⭐⭐ **必学** — 不学这个，后面会卡。工程价值最高
- ⭐⭐ **推荐** — 学了会明显提升，不学也能凑合
- ⭐ **了解** — 需要时再看

---


