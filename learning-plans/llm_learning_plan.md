# LLM 大模型工程师实战学习计划

> 基于 ai-engineering-from-scratch 课程（503课/20阶段），聚焦 Phase 10 & 11
> 总计约 43 小时核心内容（从 503 课中精选 39 课）

---

## 学习原则

1. **Build It > Read About It** — 每个概念都要跑代码，不是看概念
2. **从底层理解** — 用 numpy 实现一遍，才知道 PyTorch 在帮你做什么
3. **工程实现优先** — 不追求学术完整性，追求能用到工作里的能力

---

## 课程总览

```
Phase 10: LLMs from Scratch    24 课  ~26 小时    从零构建大模型
Phase 11: LLM Engineering      15 课  ~17 小时    大模型工程实践
─────────────────────────────────────────────────
精选 39 课                       ~43 小时
```

---

## 阶段一：LLM 核心原理（~15 小时）

> 目标：理解 LLM 是怎么工作、怎么训练出来的

### ⭐⭐⭐ 必学：Tokenizers（BPE, WordPiece, SentencePiece）
- 路径：`phases/10-llms-from-scratch/01-tokenizers/`
- 时间：~45 分钟
- 为什么重要：**LLM 不读英文，读整数**。分词器决定了这些整数是携带语义还是浪费空间
- 工程要点：
  - 三种策略对比：词级（词表爆炸）、字符级（序列太长）、子词（最佳平衡）
  - BPE 算法：贪心压缩，统计相邻字符对频率，合并最高频对
  - 词表大小权衡：太小→序列长→推理慢，太大→embedding 参数浪费
  - **产出：理解为什么 GPT-4 用 100K 词表，Claude 用不同分词器**

### ⭐⭐⭐ 必学：从零预训练 Mini GPT（124M 参数）
- 路径：`phases/10-llms-from-scratch/04-pre-training-mini-gpt/`
- 时间：~120 分钟
- 为什么最重要：**GPT-2 Small 有 1.24 亿参数，单 GPU 几小时就能训完。不亲手训一次，就无法真正理解你正在使用的产品模型内部发生了什么**
- 工程要点：
  - GPT 架构全图：Token Embedding + Position Embedding → 12 层 Transformer Block → LM Head → Softmax
  - 每个 Block：LayerNorm → Multi-Head Attention → 残差连接 → LayerNorm → FFN → 残差连接
  - Attention 核心：Q/K/V 三向量，Q·K^T 算注意力权重，加权求和 V
  - 训练循环：Forward → Cross-Entropy Loss → Backward → SGD 更新
  - 自回归生成：Temperature Sampling + Top-k/Top-p 过滤
  - **产出：~300 行 numpy 实现的完整 GPT-2 训练和生成**

### ⭐⭐⭐ 必学：指令微调（SFT）
- 路径：`phases/10-llms-from-scratch/06-instruction-tuning-sft/`
- 时间：~75 分钟
- 为什么重要：**基础模型只会预测下一个 token，不会遵循指令、回答问题或拒绝有害请求。SFT 是从 token 预测器到有用助手的桥梁**
- 工程要点：
  - 预训练给知识，SFT 给礼貌（行为模式）
  - 数据格式：ChatML / Alpaca / ShareGPT 三种主流格式
  - Loss Masking：只在 assistant token 上计算 loss，忽略 system/user
  - 质量 > 数量：Stanford Alpaca 用 52K 数据、$600 成本训出可用聊天模型
  - **产出：SFT 训练循环 + Chat Template 实现**

### ⭐⭐⭐ 必学：RLHF（Reward Model + PPO）
- 路径：`phases/10-llms-from-scratch/07-rlhf/`
- 时间：~75 分钟
- 为什么重要：**SFT 教会模型遵循指令，但没教它哪个回答更好。RLHF 将人类判断编码进模型行为，这就是让 Claude 变得有帮助、GPT 变得礼貌的原因**
- 工程要点：
  - 三阶段流水线：SFT → 训练 Reward Model → PPO 优化策略
  - Reward Model：输入 (prompt, response)，输出标量分数
  - PPO + KL 惩罚：优化策略但不能偏离 SFT 太远（防 reward hacking）
  - InstructGPT：1.3B 参数 + RLHF > 175B 原始 GPT-3（85% 胜率）
  - **三个模型同时运行：SFT 模型、Reward 模型、Policy 模型**
  - **产出：Reward Model 训练 + PPO 训练循环**

### ⭐⭐⭐ 必学：DPO（Direct Preference Optimization）
- 路径：`phases/10-llms-from-scratch/08-dpo/`
- 时间：~75 分钟
- 为什么重要：**RLHF 太复杂，DPO 把三模型简化为一模型**。2026 年主流方案
- 工程要点：
  - 核心思想：把 Reward Model 的目标函数直接代入 PPO，得到闭式解
  - 不需要单独训练 Reward Model，不需要 PPO 的采样循环
  - 损失函数：让 chosen response 的概率相对 rejected 增大
  - 效果与 RLHF 相当，但训练稳定性和实现难度大幅降低
  - **产出：DPO 训练循环（~50 行核心代码）**

### ⭐⭐ 推荐：Constitutional AI & Self-Improvement
- 路径：`phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/`
- 时间：~45 分钟
- 为什么推荐：Anthropic 的核心方法论，用 AI 自己来改进 AI

---

## 阶段二：LLM 工程实践（~12 小时）

> 目标：能用 LLM API 做实际工作

### ⭐⭐⭐ 必学：Prompt Engineering
- 路径：`phases/11-llm-engineering/01-prompt-engineering/`
- 时间：~45 分钟
- 为什么重要：**大多数人写 prompt 像发短信，然后纳闷为什么 2000 亿参数的模型给出平庸答案。提示工程不是技巧，而是理解每个 token 都是指令**
- 工程要点：
  - 角色 / 上下文 / 约束 / 输出格式四大模式
  - System / User / Assistant Prefill 三层结构
  - Prompt 失败诊断：幻觉、拒绝、格式错误
  - **产出：prompt 测试框架**

### ⭐⭐⭐ 必学：Structured Outputs
- 路径：`phases/11-llm-engineering/03-structured-outputs/`
- 时间：~75 分钟
- 为什么重要：Agent 调用工具需要可靠的结构化输出，不能靠"希望模型输出正确格式"
- 工程要点：
  - JSON Schema 约束
  - 受限解码（Constrained Decoding）
  - Pydantic 模型直接映射
  - **产出：结构化输出管道**

### ⭐⭐⭐ 必学：Context Engineering
- 路径：`phases/11-llm-engineering/05-context-engineering/`
- 时间：~75 分钟
- 为什么重要：**提示工程只是子集，上下文工程才是全貌。2026 年最优秀的 AI 工程师是上下文工程师——他们决定什么进窗口、什么不进、以什么顺序排列**
- 工程要点：
  - Context Window = RAM（有限、快速、直接访问），不是磁盘
  - Token 预算分配：System + Tools + History + Retrieved + Generation
  - Lost-in-the-Middle 效应：中间位置信息检索准确率下降 10-20%
  - 精心策划的 10K context > 随意塞入的 100K context
  - **产出：Context Assembler（动态 token 分配器）**

### ⭐⭐⭐ 必学：RAG（Retrieval-Augmented Generation）
- 路径：`phases/11-llm-engineering/06-rag/`
- 时间：~75 分钟
- 为什么重要：**你的 LLM 只知道训练截止日期之前的信息，对公司文档、代码库、上周会议记录一无所知。RAG 是生产 AI 中部署最多的模式**
- 工程要点：
  - 完整 pipeline：文档加载 → 分块 → 向量化 → 存储 → 检索 → 生成
  - 为什么 RAG 优于微调：成本低、更新快、可审计、模型无关
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

### ⭐⭐ 推荐：Few-Shot, Chain-of-Thought, Tree-of-Thought
- 路径：`phases/11-llm-engineering/02-few-shot-cot/`
- 时间：~45 分钟
- 为什么推荐：推理增强技术，提升复杂任务准确率

### ⭐⭐ 推荐：Fine-Tuning with LoRA & QLoRA
- 路径：`phases/11-llm-engineering/08-fine-tuning-lora/`
- 时间：~75 分钟
- 为什么推荐：特定场景下 RAG 不够用，需要微调

---

## 阶段三：推理与部署（~10 小时）

> 目标：让模型跑得快、跑得省

### ⭐⭐⭐ 必学：Quantization（量化）
- 路径：`phases/10-llms-from-scratch/11-quantization/`
- 时间：~75 分钟
- 为什么重要：**FP16 的 70B 模型需要 140GB 显存，要两张 A100。量化到 FP8 只需一张 80GB GPU，INT4 可以在 MacBook 上跑**
- 工程要点：
  - 数值格式：FP32 → FP16 → BF16 → FP8 → INT8 → INT4
  - 对称 vs 非对称量化，per-tensor vs per-channel
  - PTQ（训练后量化）vs QAT（量化感知训练）
  - GPTQ / AWQ：主流量化算法，INT4 保留 95-99% 质量
  - GGUF 格式：llama.cpp 的标准，让 70B 跑在 MacBook 上
  - **产出：量化实现 + 精度-内存权衡分析**

### ⭐⭐⭐ 必学：Inference Optimization（推理优化）
- 路径：`phases/10-llms-from-scratch/12-inference-optimization/`
- 时间：~75 分钟
- 为什么重要：**LLM 推理分两阶段——Prefill 并行处理 prompt（计算受限），Decode 逐 token 生成（显存受限）。每项优化都针对其中一个或两个阶段**
- 工程要点：
  - Prefill vs Decode：预填充是计算密集，解码是内存密集
  - KV Cache：缓存历史 K/V，避免重复计算
  - Continuous Batching：动态批处理，不让 GPU 空闲
  - PagedAttention：像虚拟内存一样管理 KV Cache
  - Speculative Decoding：用小模型猜测，大模型验证
  - **产出：KV Cache 实现 + 批处理调度器**

### ⭐⭐ 推荐：Building a Complete LLM Pipeline
- 路径：`phases/10-llms-from-scratch/13-building-complete-llm-pipeline/`
- 时间：~120 分钟
- 为什么推荐：端到端串联所有组件

### ⭐⭐ 推荐：Prompt Caching & Context Caching
- 路径：`phases/11-llm-engineering/15-prompt-caching/`
- 时间：~60 分钟
- 为什么推荐：重复 context 缓存可大幅降低 API 成本

### ⭐⭐ 推荐：Caching, Rate Limiting & Cost Optimization
- 路径：`phases/11-llm-engineering/11-caching-cost/`
- 时间：~45 分钟
- 为什么推荐：生产环境成本控制

---

## 阶段四：前沿架构（~8 小时）

> 目标：理解 2024-2026 前沿模型的架构创新

### ⭐⭐⭐ 必学：DeepSeek-V3 Architecture Walkthrough
- 路径：`phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/`
- 时间：~75 分钟
- 为什么重要：**DeepSeek-V3（2024年12月，671B 总参数/37B 激活参数）在六个架构旋钮基础上增加了四个创新。2026 年最重要的开源模型**
- 工程要点：
  - MLA（Multi-Head Latent Attention）：比 GQA 更省 KV Cache
  - 辅助损失无关的负载均衡
  - Multi-Token Prediction（MTP）：一次预测多个 token
  - DualPipe 训练并行策略
  - **KV Cache 对比：MLA 7.6GB vs GQA 同等模型 ~15GB**

### ⭐⭐ 推荐：Differential Attention（V2）
- 路径：`phases/10-llms-from-scratch/16-differential-attention-v2/`
- 时间：~60 分钟
- 为什么推荐：微软的新注意力机制，减少噪声

### ⭐⭐ 推荐：Native Sparse Attention（DeepSeek NSA）
- 路径：`phases/10-llms-from-scratch/17-native-sparse-attention/`
- 时间：~60 分钟
- 为什么推荐：DeepSeek 的稀疏注意力，长序列效率提升

### ⭐⭐ 推荐：Multi-Token Prediction（MTP）
- 路径：`phases/10-llms-from-scratch/18-multi-token-prediction/`
- 时间：~60 分钟
- 为什么推荐：Meta 提出，DeepSeek 实现，推理加速新范式

### ⭐⭐ 推荐：Speculative Decoding and EAGLE-3
- 路径：`phases/10-llms-from-scratch/15-speculative-decoding-eagle3/`
- 时间：~75 分钟
- 为什么推荐：用小模型加速大模型推理

### ⭐⭐ 推荐：Jamba — Hybrid SSM-Transformer
- 路径：`phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/`
- 时间：~60 分钟
- 为什么推荐：SSM + Transformer 混合架构，长序列新方向

---

## 阶段五：工程保障（~6 小时）

> 目标：让 LLM 应用在生产中可靠运行

### ⭐⭐ 推荐：Evaluation & Testing LLM Applications
- 路径：`phases/11-llm-engineering/10-evaluation/`
- 时间：~45 分钟
- 工程要点：LLM-as-Judge、执行式评估、轨迹评估

### ⭐⭐ 推荐：Guardrails, Safety & Content Filtering
- 路径：`phases/11-llm-engineering/12-guardrails/`
- 时间：~45 分钟
- 工程要点：输出过滤、安全检查、Prompt Injection 防御

### ⭐⭐ 推荐：Building a Production LLM Application
- 路径：`phases/11-llm-engineering/13-production-app/`
- 时间：~120 分钟
- 工程要点：端到端生产应用架构

### ⭐⭐ 推荐：Model Context Protocol（MCP）
- 路径：`phases/11-llm-engineering/14-model-context-protocol/`
- 时间：~75 分钟
- 工程要点：标准化工具接口协议

### ⭐⭐ 推荐：Advanced RAG
- 路径：`phases/11-llm-engineering/07-advanced-rag/`
- 时间：~75 分钟
- 工程要点：分块策略、重排序、混合搜索

---

## 学习顺序总览

```
Week 1: LLM 核心原理                      Week 2: 工程实践
┌─────────────────────────┐     ┌─────────────────────────┐
│ Phase 10                │     │ Phase 11                │
│ #01 Tokenizers          │     │ #01 Prompt Engineering  │
│ #04 Pre-Training GPT    │ --> │ #03 Structured Outputs  │
│ #06 Instruction Tuning  │     │ #05 Context Engineering │
│ #07 RLHF                │     │ #06 RAG                 │
│ #08 DPO                 │     │ #09 Function Calling    │
└─────────────────────────┘     └─────────────────────────┘

Week 3: 推理部署                          Week 4: 前沿架构
┌─────────────────────────┐     ┌─────────────────────────┐
│ Phase 10                │     │ Phase 10                │
│ #11 Quantization        │     │ #20 DeepSeek-V3         │
│ #12 Inference Optim     │ --> │ #16 Differential Attn   │
│ #13 Complete Pipeline   │     │ #17 Sparse Attention    │
│                         │     │ #18 Multi-Token Pred    │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 关键工程决策速查

| 场景 | 选择 | 参考课 |
|------|------|--------|
| 需要模型回答问题 | SFT（不是继续预训练） | P10 #06 |
| 提升回答质量 | RLHF 或 DPO | P10 #07, #08 |
| 简单对齐需求 | DPO（比 RLHF 简单） | P10 #08 |
| 模型太大放不进 GPU | 量化（INT4/INT8） | P10 #11 |
| 推理太慢太贵 | KV Cache + Continuous Batching | P10 #12 |
| 需要外部知识 | RAG（不是微调） | P11 #06 |
| 需要结构化输出 | Constrained Decoding | P11 #03 |
| 上下文太长 | Context Engineering + 滑动窗口 | P11 #05 |
| 长序列推理 | MLA + Sparse Attention | P10 #17, #20 |
| 加速推理 | Speculative Decoding | P10 #15 |

---

## 核心代码路径速查

| 想实现什么 | 代码在哪 |
|-----------|----------|
| Tokenizer（BPE） | `phases/10-llms-from-scratch/01-tokenizers/code/main.py` |
| 从零训练 GPT | `phases/10-llms-from-scratch/04-pre-training-mini-gpt/code/main.py` |
| SFT 训练 | `phases/10-llms-from-scratch/06-instruction-tuning-sft/code/main.py` |
| RLHF（Reward + PPO） | `phases/10-llms-from-scratch/07-rlhf/code/main.py` |
| DPO 训练 | `phases/10-llms-from-scratch/08-dpo/code/main.py` |
| 量化实现 | `phases/10-llms-from-scratch/11-quantization/code/main.py` |
| KV Cache + 推理优化 | `phases/10-llms-from-scratch/12-inference-optimization/code/main.py` |
| Prompt Engineering | `phases/11-llm-engineering/01-prompt-engineering/code/main.py` |
| RAG Pipeline | `phases/11-llm-engineering/06-rag/code/main.py` |
| Function Calling | `phases/11-llm-engineering/09-function-calling/code/main.py` |

---

## ⭐ 标记说明

- ⭐⭐⭐ **必学** — 不学这个，后面会卡。工程价值最高
- ⭐⭐ **推荐** — 学了会明显提升，不学也能凑合
- ⭐ **了解** — 需要时再看

---

## LLM 训练全流程图

```
原始文本数据
    ↓
Tokenizer 分词（BPE/WordPiece）
    ↓
预训练（Next Token Prediction）
  → 学会语言知识，但只会续写
    ↓
SFT 指令微调（Instruction-Response 对）
  → 学会回答问题，但不知道哪个更好
    ↓
RLHF / DPO 偏好对齐（Chosen vs Rejected）
  → 学会判断质量，输出更有帮助
    ↓
量化（FP16 → INT4/INT8）
  → 模型变小，可部署到消费级硬件
    ↓
推理优化（KV Cache / Batching / Speculative）
  → 推理变快，成本降低
    ↓
生产部署（RAG / Context Engineering / Guardrails）
  → 真正可用的 AI 应用
```

---

## 课程来源

- 项目：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- 总课时：503 课 / 20 阶段 / ~320 小时
- 本文档精选：39 课 / 2 阶段 / ~43 小时
- 课程特点：每个算法先用 numpy 从零实现，再用 PyTorch 生产库
