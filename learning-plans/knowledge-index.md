# FDE/Agent 架构师知识库索引

> 基于 ai-engineering-from-scratch 课程（503课/20阶段/~320小时）
> 知识库路径：`/home/yanyan/ai-engineering/`

---

## 一、学习路线（按角色）

### AI Agent 工程师（~80小时）
- 文件：`learning-plans/ai_agent_learning_plan.md`
- 重点：Agent Loop、Memory、多Agent编排、可靠性

### LLM 工程师（~43小时）
- 文件：`learning-plans/llm_learning_plan.md`
- 重点：Tokenizers、预训练、SFT、RLHF/DPO、推理优化

### FDE 架构师（~120小时）
- 文件：`learning-plans/fde_architect_learning_plan.md`
- 重点：MCP协议、自主系统、多Agent、生产部署、安全

### 架构师补充（~150小时含补充）
- 文件：`learning-plans/architect_supplement.md`
- 重点：Transformer深度理解、多模态AI、强化学习基础

---

## 二、核心主题速查

### LLM 基础（Phase 10 & 11）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| Tokenizers | `llm-learning-plan-translated/phase-10-llms-from-scratch/01-tokenizers-cn.md` | BPE算法、词表大小权衡 |
| 预训练 Mini GPT | `llm-learning-plan-translated/phase-10-llms-from-scratch/02-pre-training-mini-gpt-cn.md` | GPT架构、Attention、训练循环 |
| 指令微调 SFT | `llm-learning-plan-translated/phase-10-llms-from-scratch/03-instruction-tuning-sft-cn.md` | ChatML格式、Loss Masking |
| RLHF | `llm-learning-plan-translated/phase-10-llms-from-scratch/04-rlhf-cn.md` | Reward Model + PPO、三模型同时运行 |
| DPO | `llm-learning-plan-translated/phase-10-llms-from-scratch/05-dpo-cn.md` | 闭式解、比RLHF简单 |
| 量化 | `llm-learning-plan-translated/phase-10-llms-from-scratch/07-quantization-cn.md` | FP16→INT4、GPTQ/AWQ、GGUF格式 |
| 推理优化 | `llm-learning-plan-translated/phase-10-llms-from-scratch/08-inference-optimization-cn.md` | KV Cache、Continuous Batching、PagedAttention |
| DeepSeek-V3 | `llm-learning-plan-translated/phase-10-llms-from-scratch/14-deepseek-v3-walkthrough-cn.md` | MLA、MTP、671B/37B架构 |
| Prompt Engineering | `llm-learning-plan-translated/phase-11-llm-engineering/01-prompt-engineering-cn.md` | System/User/Assistant三层结构 |
| 结构化输出 | `llm-learning-plan-translated/phase-11-llm-engineering/03-structured-outputs-cn.md` | JSON Schema约束、受限解码 |
| Context Engineering | `llm-learning-plan-translated/phase-11-llm-engineering/04-context-engineering-cn.md` | Token预算分配、Lost-in-the-Middle |
| RAG | `llm-learning-plan-translated/phase-11-llm-engineering/05-rag-cn.md` | 完整pipeline、评估指标 |
| Function Calling | `llm-learning-plan-translated/phase-11-llm-engineering/08-function-calling-cn.md` | 5步循环、Tool Schema设计 |
| LoRA/QLoRA | `llm-learning-plan-translated/phase-11-llm-engineering/07-fine-tuning-lora-cn.md` | 参数高效微调 |

### Agent 核心（Phase 14）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| Agent Loop | `phase-14-agent-frameworks/phase-14-01-agent-loop-cn.md` | ReAct循环、5个必要组件 |
| Tool Use | `phase-14-agent-frameworks/phase-14-02-tool-use-cn.md` | 工具Schema设计、并行调用 |
| Memory | `phase-14-agent-frameworks/phase-14-07-memory-cn.md` | MemGPT、两层记忆、OS虚拟内存类比 |
| Hybrid Memory | `phase-14-agent-frameworks/phase-14-09-hybrid-memory-cn.md` | Mem0、Vector+Graph+KV |
| Skill Libraries | `phase-14-agent-frameworks/voyager-skill-library-cn.md` | Voyager、技能积累与复用 |
| Anthropic Workflows | `phase-14-agent-frameworks/anthropic-workflows-cn.md` | 5种Workflow模式 |
| LangGraph | `phase-14-agent-frameworks/langgraph-cn.md` | 状态机、Checkpoint、持久执行 |
| OpenAI Agents SDK | `phase-14-agent-frameworks/openai-agents-sdk-cn.md` | 5个原语、Handoff=工具 |
| Claude Agent SDK | `phase-14-agent-frameworks/claude-agent-sdk-cn.md` | Subagents、Session Store、Hooks |
| Orchestration Patterns | `phase-14-agent-frameworks/orchestration-patterns-cn.md` | 4种拓扑：Supervisor/Swarm/Hierarchical/Debate |

### Agent 可靠性（Phase 15）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| 失败模式 | `phase-15-agent-reliability/failure-modes-cn.md` | MASFT 14种失败、级联最致命 |
| 评估驱动 | `phase-15-agent-reliability/eval-driven-cn.md` | 三层评估、CI门禁 |
| Prompt Injection | `phase-15-agent-reliability/prompt-injection-cn.md` | PVE防御、信任边界 |
| 可观测性 | `phase-15-agent-reliability/agent-observability-cn.md` | Langfuse/Phoenix/Opik |
| Long-Horizon | `phase-15-agent-reliability/long-horizon-cn.md` | METR Time Horizon、能力翻倍~7月 |
| 持久执行 | `phase-15-agent-reliability/durable-execution-cn.md` | 检查点、故障恢复 |
| 成本控制 | `phase-15-agent-reliability/cost-governors-cn.md` | 动作预算、迭代上限、断路器 |
| HITL | `phase-15-agent-reliability/hitl-cn.md` | Propose-Then-Commit |
| Supervisor Pattern | `phase-15-agent-reliability/supervisor-pattern-cn.md` | 比单Agent提升90.2% |
| Verification Gates | `phase-15-agent-reliability/verification-gates-cn.md` | 产出门禁验证 |

### 工具协议（Phase 13）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| MCP Fundamentals | `fde-architect-plan/phase-13-tools-protocols/phase-13-mcp-fundamentals-cn.md` | Client↔Server、三种传输 |
| MCP Server | `fde-architect-plan/phase-13-tools-protocols/phase-13-mcp-server-cn.md` | 工具定义、资源暴露 |
| MCP Client | `fde-architect-plan/phase-13-tools-protocols/phase-13-mcp-client-cn.md` | 连接管理、多Server编排 |
| MCP Security | `fde-architect-plan/phase-13-tools-protocols/phase-13-mcp-security-cn.md` | Tool Poisoning |
| A2A Protocol | `fde-architect-plan/phase-13-tools-protocols/phase-13-a2a-protocol-cn.md` | Agent Card、Task协议 |
| OpenTelemetry | `fde-architect-plan/phase-13-tools-protocols/phase-13-otel-genai-cn.md` | Span/Metrics/Trace |

### 多Agent系统（Phase 16）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| Why Multi-Agent | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-why-multi-agent-cn.md` | 单Agent局限、何时用多Agent |
| Supervisor Pattern | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-supervisor-pattern-cn.md` | 独立上下文、专注prompt、并行 |
| 角色专业化 | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-role-specialization-cn.md` | Planner/Critic/Executor/Verifier |
| A2A Protocol | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-a2a-protocol-cn.md` | Agent间通信标准 |
| Handoffs & Routines | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-handoffs-routines-cn.md` | 无状态编排 |
| 失败模式 | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-failure-modes-cn.md` | MAST、Groupthink、Monoculture |

### 基础设施（Phase 17）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| AI Gateways | `fde-architect-plan/phase-17-infrastructure/phase-17-ai-gateways-cn.md` | LiteLLM/Portkey/Kong |
| 模型路由 | `fde-architect-plan/phase-17-infrastructure/phase-17-model-routing-cn.md` | 不同任务用不同模型、成本降50%+ |
| 推理指标 | `fde-architect-plan/phase-17-infrastructure/phase-17-inference-metrics-cn.md` | TTFT/TPOT/ITL/Goodput/P99 |
| Prompt缓存 | `fde-architect-plan/phase-17-infrastructure/phase-17-prompt-caching-cn.md` | 降低30-50%成本 |
| 自托管 | `fde-architect-plan/phase-17-infrastructure/phase-17-self-hosted-cn.md` | llama.cpp/Ollama/vLLM |
| 安全 | `fde-architect-plan/phase-17-infrastructure/phase-17-security-cn.md` | PII/Secrets/Audit |

### 安全与对齐（Phase 18）

| 主题 | 文件路径 | 核心要点 |
|------|----------|----------|
| Prompt Injection | `fde-architect-plan/phase-18-safety-alignment/phase-18-prompt-injection-cn.md` | 间接注入、防御策略 |
| Alignment Faking | `fde-architect-plan/phase-18-safety-alignment/phase-18-alignment-faking-cn.md` | 12-78%模型伪装对齐 |
| Red-Teaming | `fde-architect-plan/phase-18-safety-alignment/phase-18-red-teaming-cn.md` | PAIR、自动化攻击 |
| Sleeper Agents | `fde-architect-plan/phase-18-safety-alignment/phase-18-sleeper-agents-cn.md` | 持续欺骗风险 |

---

## 三、关键决策框架

### 选什么框架？
| 场景 | 选择 | 参考 |
|------|------|------|
| 任务步骤可枚举 | Workflow（不是Agent） | Anthropic Workflows |
| 2-4个专家 | Supervisor-Worker | Orchestration Patterns |
| 延迟>推理清晰度 | Swarm | Orchestration Patterns |
| Token预算紧 | ReWOO（Plan-Execute） | Agent Loop |
| 准确率>成本 | Debate | Multi-Agent Debate |
| 长任务(>1h) | Checkpoint+Durable Execution | Long-Horizon |

### 选什么模型？
| 场景 | 选择 | 参考 |
|------|------|------|
| 简单任务 | Haiku/GPT-4o-mini | Model Routing |
| 复杂任务 | Opus/GPT-5 | Model Routing |
| 模型太大 | 量化(INT4/INT8) | Quantization |
| 推理太慢 | KV Cache+Batching | Inference Optimization |

### 选什么记忆系统？
| 场景 | 选择 | 参考 |
|------|------|------|
| 跨会话记忆 | Mem0/Letta三层记忆 | Hybrid Memory |
| 短期上下文 | Context Engineering | Context Engineering |
| 技能积累 | Voyager Skill Library | Skill Libraries |

---

## 四、核心代码路径

| 想实现什么 | 代码在哪 |
|-----------|----------|
| 最小 Agent Loop | `phase-14-agent-frameworks/phase-14-01-agent-loop-cn.md` |
| Plan-Execute | `phase-14-agent-frameworks/phase-14-02-tool-use-cn.md` |
| RAG Pipeline | `llm-learning-plan-translated/phase-11-llm-engineering/05-rag-cn.md` |
| Function Calling | `llm-learning-plan-translated/phase-11-llm-engineering/08-function-calling-cn.md` |
| MCP Server | `fde-architect-plan/phase-13-tools-protocols/phase-13-mcp-server-cn.md` |
| Supervisor-Worker | `fde-architect-plan/phase-16-multi-agent-swarms/phase-16-supervisor-pattern-cn.md` |

---

## 五、使用方法

当用户问到以下主题时，读取对应文件：

1. **Agent架构设计** → 先读 `phase-14-agent-frameworks/` 下的相关文件
2. **LLM工程** → 读 `llm-learning-plan-translated/phase-11-llm-engineering/`
3. **多Agent编排** → 读 `fde-architect-plan/phase-16-multi-agent-swarms/`
4. **生产部署** → 读 `fde-architect-plan/phase-17-infrastructure/`
5. **安全问题** → 读 `fde-architect-plan/phase-18-safety-alignment/`
6. **工具协议** → 读 `fde-architect-plan/phase-13-tools-protocols/`

读取时使用：`read_file("/home/yanyan/ai-engineering/<路径>")`
