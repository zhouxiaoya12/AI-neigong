# 资深架构师视角：被忽略的关键能力

> 作者视角：十年 AI 经验，从 GPT-2 到 Claude Opus 4.6，从 RLHF 到 MCP 协议，全程参与构建
> 目的：找出学习计划中容易被忽略但至关重要的内容

---

## 我的核心判断

作为一个从 2016 年就开始做 AI 的人，我见过太多"会用 API 但不理解底层"的工程师。他们能调通 RAG、能跑通 Agent Loop，但遇到问题就束手无策——因为他们不理解 Attention 机制、不理解 RLHF 的本质、不理解为什么模型会幻觉。

**以下是我认为必须补上的内容。**

---

## 一、Transformer 深度理解（Phase 7 精选）

> 很多人以为"会用 LLM API 就够了"。错了。不理解 Transformer，你就无法：
> - 诊断为什么模型在某个任务上表现差
> - 理解 KV Cache 为什么能加速推理
> - 理解 Flash Attention 为什么能省显存
> - 理解 Scaling Laws 为什么模型越大越好

### ⭐⭐⭐ 必学：Self-Attention from Scratch
- 路径：`phases/07-transformers-deep-dive/02-self-attention-from-scratch/`
- 时间：~75 分钟
- 为什么必须学：**Attention 是 Transformer 的灵魂。不理解 Q/K/V 的数学，你就是在黑盒调 API**
- 核心问题：
  - Q（Query）问："我在找什么？"
  - K（Key）答："我包含什么？"
  - V（Value）说："我的内容是什么？"
  - Attention = softmax(Q·K^T / √d) · V
- **产出：用 numpy 实现 Self-Attention**

### ⭐⭐⭐ 必学：Multi-Head Attention
- 路径：`phases/07-transformers-deep-dive/03-multi-head-attention/`
- 时间：~75 分钟
- 为什么必须学：**GPT、Claude、Gemini 都用 Multi-Head Attention。理解它才能理解为什么模型能同时关注多个方面**
- 核心概念：
  - 单头 Attention 只能关注一种模式
  - 多头 Attention 可以同时关注语法、语义、位置等
  - h 个头并行计算，最后拼接
- **产出：Multi-Head Attention 实现**

### ⭐⭐⭐ 必学：KV Cache, Flash Attention & Inference Optimization
- 路径：`phases/07-transformers-deep-dive/12-kv-cache-flash-attention/`
- 时间：~75 分钟
- 为什么必须学：**这是推理优化的核心。不理解 KV Cache，你就无法理解为什么推理这么慢、怎么加速**
- 核心概念：
  - KV Cache：缓存历史 Key/Value，避免重复计算
  - Flash Attention：IO-aware 的 Attention 实现，速度提升 2-4x
  - 连续批处理：让 GPU 不空闲
- **产出：KV Cache + Flash Attention 实现**

### ⭐⭐ 推荐：Scaling Laws
- 路径：`phases/07-transformers-deep-dive/13-scaling-laws/`
- 时间：~45 分钟
- 为什么推荐：**理解为什么 GPT-4 比 GPT-3 强，为什么 Claude Opus 比 Haiku 强**
- 核心公式：Loss ∝ N^(-α)，其中 N 是参数量，α ≈ 0.076

### ⭐⭐ 推荐：Mixture of Experts (MoE)
- 路径：`phases/07-transformers-deep-dive/11-mixture-of-experts/`
- 时间：~45 分钟
- 为什么推荐：**DeepSeek-V3、Mixtral 都用 MoE。理解它才能理解 671B/37B 的设计**

---

## 二、多模态 AI（Phase 12 精选）

> 2026 年的趋势：Agent 不只是聊天，还要看图、看视频、听声音、操作电脑。
> 不理解多模态，你就无法构建下一代 Agent。

### ⭐⭐⭐ 必学：LLaVA and Visual Instruction Tuning
- 路径：`phases/12-multimodal-ai/05-llava-visual-instruction-tuning/`
- 时间：~180 分钟
- 为什么必须学：**LLaVA 是视觉 Agent 的鼻祖。理解它才能理解 GPT-4V、Claude Vision 的工作原理**
- 核心概念：
  - 视觉编码器（CLIP）→ 投影层 → LLM
  - Visual Instruction Tuning：用图文对话数据微调
  - 为什么 vision encoder 不训练、只训练投影层
- **产出：LLaVA 架构实现**

### ⭐⭐⭐ 必学：Omni Models: Thinker-Talker
- 路径：`phases/12-multimodal-ai/20-omni-models-thinker-talker/`
- 时间：~180 分钟
- 为什么必须学：**GPT-4o、Gemini 2.0 Flash 的架构。2026 年的主流模型都是 Omni 模型**
- 核心概念：
  - Thinker：处理文本/图像/音频输入
  - Talker：生成文本/音频输出
  - 统一 token 空间：所有模态共享 token 空间
- **产出：理解 Omni 模型架构**

### ⭐⭐⭐ 必学：Multimodal Agents and Computer-Use (Capstone)
- 路径：`phases/12-multimodal-ai/25-multimodal-agents-computer-use/`
- 时间：~240 分钟
- 为什么必须学：**2026 年最热的方向：Agent 操作电脑。Claude Computer Use、OpenAI Operator 都是这个方向**
- 核心概念：
  - 屏幕截图 → 视觉理解 → 鼠标/键盘操作
  - 元素定位：找到要点击的位置
  - 动作规划：决定下一步做什么
- **产出：Computer-Use Agent 原型**

### ⭐⭐ 推荐：CLIP and Contrastive Vision-Language Pretraining
- 路径：`phases/12-multimodal-ai/02-clip-contrastive-pretraining/`
- 时间：~180 分钟
- 为什么推荐：**CLIP 是所有多模态模型的基础。理解对比学习才能理解视觉-语言对齐**

---

## 三、强化学习基础（Phase 9 精选）

> 很多人以为"RLHF 就是调 API"。错了。不理解 PPO 的数学，你就无法：
> - 理解为什么 RLHF 训练不稳定
> - 理解 DPO 为什么更简单
> - 理解对齐伪装是怎么发生的

### ⭐⭐⭐ 必学：Proximal Policy Optimization (PPO)
- 路径：`phases/09-reinforcement-learning/08-ppo/`
- 时间：~75 分钟
- 为什么必须学：**PPO 是 RLHF 的核心算法。不理解 PPO，你就无法理解对齐**
- 核心概念：
  - 策略梯度：直接优化策略
  - 近端约束：防止策略变化太大
  - 优势函数：评估动作的好坏
- **产出：PPO 算法实现**

### ⭐⭐ 推荐：Reward Modeling & RLHF
- 路径：`phases/09-reinforcement-learning/09-reward-modeling-rlhf/`
- 时间：~45 分钟
- 为什么推荐：**把 RL 基础和 LLM 对齐连接起来**

---

## 四、容易被忽略但至关重要的能力

### 4.1 评估能力（Evaluation）

> 我见过太多团队"能构建但不能评估"。他们不知道自己的 Agent 到底好不好。

**关键课程：**
- `phases/14-agent-engineering/30-eval-driven-agent-development/` - 评估驱动开发
- `phases/14-agent-engineering/31-agent-workbench-why-models-fail/` - 七个 Surface

**核心理念：**
- 评估不是最后一步，是驱动所有其他选择的外循环
- 三层评估：Static Benchmarks → Custom Offline Evals → Online Evals
- 评估代码和业务代码放一起，CI 里跑，PR 门禁

### 4.2 可观测性（Observability）

> 没有可观测性，40 步 Agent 调试第 38 步的错误决策是不可能的。

**关键课程：**
- `phases/14-agent-engineering/24-agent-observability-platforms/` - Langfuse, Phoenix, Opik
- `phases/14-agent-engineering/23-otel-genai-conventions/` - OpenTelemetry GenAI

**核心理念：**
- 不是"日志"，是"轨迹级遥测"
- 每个 LLM 调用、工具调用、Agent 循环都要有 span
- 成本、延迟、成功率都要监控

### 4.3 成本工程（Cost Engineering）

> 一个失控的 Agent 循环可以一夜之间烧掉一个月预算。

**关键课程：**
- `phases/15-autonomous-systems/13-cost-governors/` - 成本控制
- `phases/17-infrastructure-and-production/16-model-routing/` - 模型路由
- `phases/17-infrastructure-and-production/14-prompt-caching/` - Prompt 缓存

**核心理念：**
- 不同任务用不同模型（简单→Haiku，复杂→Opus）
- Prompt 缓存可降低 30-50% 成本
- 必须有预算断路器

### 4.4 安全能力（Security）

> Prompt Injection 是真实威胁。2024 年已经有真实攻击案例。

**关键课程：**
- `phases/14-agent-engineering/27-prompt-injection-defense/` - PVE 防御
- `phases/18-ethics-safety-alignment/15-indirect-prompt-injection/` - 间接注入
- `phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/` - MCP 安全

**核心理念：**
- 工具输出是不可信输入
- 信任边界：只有用户的直接指令才算权限
- 安全不是可选项，是必须项

---

## 五、2026 年必须关注的新趋势

### 5.1 Computer-Use Agent
- Claude Computer Use、OpenAI Operator
- 用 Agent 操作电脑：看屏幕、点鼠标、打字
- 核心课程：`phases/12-multimodal-ai/25-multimodal-agents-computer-use/`

### 5.2 长程 Agent（14+ 小时）
- Claude Opus 4.6 能跑 14 小时的专家任务
- 需要：检查点、成本控制、人类在环
- 核心课程：`phases/15-autonomous-systems/01-long-horizon-agents/`

### 5.3 MCP/A2A 协议标准化
- MCP：工具集成标准（Hermes、Claude Code、Cursor 都支持）
- A2A：Agent 间通信标准（Google 提出）
- 核心课程：`phases/13-tools-and-protocols/06-mcp-fundamentals/`

### 5.4 对齐伪装（Alignment Faking）
- Anthropic 2024 年研究：12-78% 的模型会伪装对齐
- 模型在测试中表现更安全，部署中可能不同
- 核心课程：`phases/18-ethics-safety-alignment/09-alignment-faking/`

---

## 六、我的建议：优先级排序

### 第一优先级（必须学）
1. Phase 7: Self-Attention + Multi-Attention + KV Cache
2. Phase 12: LLaVA + Omni Models + Computer-Use
3. Phase 9: PPO

### 第二优先级（强烈推荐）
1. Phase 13: MCP Server/Client + A2A
2. Phase 15: 长程 Agent + 成本控制
3. Phase 16: Supervisor-Worker + 角色专业化

### 第三优先级（有时间就学）
1. Phase 17: AI Gateway + 模型路由
2. Phase 18: Prompt Injection + 对齐伪装
3. Phase 7: Scaling Laws + MoE

---

## 七、一句话总结

**技术每几年换一批，底层模式几十年后还会以新的名字再次出现。**

- Attention 机制是 2017 年的，但 2026 年的所有模型都在用
- RLHF 是 2022 年的，但对齐伪装是 2024 年才发现的问题
- MCP 是 2024 年的，但会成为 2026 年的工具集成标准

**学到最后，你会发现：真正重要的不是"会用什么框架"，而是"理解什么原理"。**

---

## 附：完整学习路径（含补充）

```
Phase 7: Transformers Deep Dive（~6 小时精选）
    ↓
Phase 9: Reinforcement Learning（~2 小时精选）
    ↓
Phase 10 & 11: LLM 大模型（~43 小时）✅ 已完成
    ↓
Phase 14: Agent Engineering（~42 小时）✅ 已完成
    ↓
Phase 12: Multimodal AI（~10 小时精选）
    ↓
Phase 13: Tools & Protocols（~12 小时）
    ↓
Phase 15: Autonomous Systems（~10 小时）
    ↓
Phase 16: Multi-Agent & Swarms（~12 小时）
    ↓
Phase 17: Infrastructure & Production（~10 小时）
    ↓
Phase 18: Safety & Alignment（~6 小时）

总计：~150 小时
```
