# 🦅 ai-neigong

> **AI 内功 —— Agent工程界的九阴真经**
>
> 九阴真经不是一本招式大全，是一套从内功心法到外功招式的完整武学体系。
>
> 这本也是——从 Tokenizer 到 Agent Loop，从 RAG 到 Multi-Agent 编排，从量化到生产部署。
> 不是"如何用 AI 一键生成爆款"，是"AI 到底怎么工作的，你该怎么工程化它"。

---

## 为什么要写这个

中文技术圈有一个很尴尬的现象：**Agent 这个词被过度消费，但 Agent 工程基本没人讲清楚。**

你去翻国内的 Agent 文章，十篇有八篇在讲"什么是 Agent"——一个定义、一张架构图、三个场景举例，完了。剩下两篇是某个框架的 README 机翻。没有人往下走一层：**Agent 循环到底怎么跑起来的？工具调用失败之后系统怎么兜底？五个 Agent 之间的控制权怎么分布？评估一个 Agent 到底看什么指标？**

结果是：一群工程师能把 LangChain 的 Quick Start 跑通，但做不出一个能上生产的 Agent 系统。框架换了一个又一个，LangChain、AutoGen、CrewAI、LangGraph——每换一次都是重新看一遍 Quick Start，但底层那套东西始终没搞懂。

**因为缺的不是框架文档，是工程直觉。**

框架是招式，Agent 循环是内功。ReAct 循环、Tool Registry、Memory 架构、Stop Condition、Observation Formatter——这五个要素不管你用哪个框架，底层都是同一套逻辑。我见过太多人"动一下洗一下"——调通一个 Demo 就觉得会了，一上生产就炸。

**这本经书要解决的就是这个断层。** 不讲"AI 是什么"，讲"Agent 怎么工程化"。每一章都往下挖一层，挖到你不需要框架也能自己写出一个 Agent 为止。

---

## 这是什么

一本 **AI 工程体系化中文教程——九阴真经**。

从 Tokenizer 到 Agent Loop，从 RAG 到多 Agent 编排，从量化到生产部署——按三条路线分层组织，让不同背景的人都能找到入口，不是从头读到尾的教科书。


---

## 给谁看的

| 你如果是 | 你怎么用 |
|----------|---------|
| **后端工程师，刚接触 AI** | 从 Tokenizer 到 Agent Loop，按 LLM 工程师路线走 |
| **有 LLM 使用经验，想深入 Agent** | 直跳 Agent 核心循环，然后走 Agent 工程师路线 |
| **在做 AI 系统架构** | 架构师路线，补 Transformer/RAG/多 Agent/安全 |
| **面试突击 / 查漏补缺** | 知识库索引，按关键词速查 |
| **产品经理 / 技术管理者** | 不要误入"外功路线"，里面有深浅指引 |

---

## 内容地图

```
🦅 ai-neigong / AI内功
│
├── 卷一 · 内功心法 ─── LLM 从零开始
│   ├── 01 Tokenizer（BPE / WordPiece / SentencePiece）
│   ├── 02 预训练 Mini GPT（124M 参数，从零训一遍）
│   ├── 03 指令微调 SFT（ChatML 格式、Loss Masking）
│   ├── 04 RLHF / DPO（奖励模型、偏好对齐）
│   ├── 05 量化（FP16→INT4、GPTQ/AWQ、GGUF）
│   └── 06 推理优化（KV Cache、Flash Attention、Continuous Batching）
│
├── 卷二 · 外功招式 ─── LLM 工程实战
│   ├── 01 提示工程（System/User/Assistant 三层、角色设定、失败诊断）
│   ├── 02 结构化输出（JSON Schema、受限解码、Pydantic 映射）
│   ├── 03 上下文工程（200K 窗口是预算，不是容器）
│   ├── 04 RAG 基础 + 高级（混合搜索、重排序、评估）
│   ├── 05 函数调用（5 步调用循环、并行调用、无限循环防护）
│   ├── 06 LoRA/QLoRA 微调
│   └── 07 缓存与成本优化、护栏与安全
│
├── 卷三 · 实战剑法 ─── Agent 核心机制
│   ├── 01 Agent 循环（ReAct：Thought→Action→Observe→Loop）
│   ├── 02 工具调用与注册表
│   ├── 03 Agent 记忆（短期/长期/混合记忆架构）
│   ├── 04 技能库（代码代替记忆）
│   ├── 05 精炼循环（别自评，让另一个 Agent 评）
│   ├── 06 五种工作流（Prompt Chaining / Routing / Parallel / Orchestrator / Evaluator）
│   └── 07 四种编排拓扑（Supervisor / Hierarchical / Peer-to-Peer / Hybrid）
│
├── 卷四 · 门派出身 ─── Agent 框架
│   ├── LangGraph（状态机引擎 / 崩溃恢复）
│   ├── OpenAI Agents SDK（五个原语 / 转交即工具）
│   ├── Claude Agent SDK（Harness 形态 / 产线外化）
│   └── CrewAI / 框架对比与选型
│
├── 卷五 · 护体神功 ─── Agent 可靠性工程
│   ├── 五种失败模式（命名 + 检测 + 门控）
│   ├── 评估驱动开发（"我有证据"替换"我以为"）
│   ├── 提示注入攻防（检索内容 = 任意代码执行）
│   ├── 可观测性（Langfuse / Phoenix / Opik）
│   ├── OTel GenAI 约定（跨平台共同语言）
│   ├── 验证门控（Agent 不能说"做完了"）
│   ├── 持久化执行 / 成本调控 / 人类在环
│   └── 多 Agent 辩论（分歧比一致更有用）
│
├── 密卷 · 武学总纲 ─── 架构师深度补充
│   ├── Self-Attention 从零实现（Q·K^T/√d·V）
│   ├── Multi-Head Attention / KV Cache / Flash Attention
│   ├── 多模态 AI（CLIP / LLaVA / Omni Models）
│   ├── 强化学习（PPO / 奖励建模 / RLHF 数学）
│   ├── MoE 混合专家模型
│   └── 安全对齐（Indirect Prompt Injection / Alignment Faking）
│
└── 附录 · 武学目录 ─── 知识库索引
    ├── AI Agent 工程师路线（~80 小时）
    ├── LLM 工程师路线（~43 小时）
    ├── 架构师路线（~120 小时）
    └── 架构师深度补充（~150 小时含补充）
```

---

## 三条修炼路线

看懂这张表就知道该走哪条路：

| 路线 | 适合谁 | 核心内容 | 预计时间 |
|------|--------|----------|----------|
| 🟢 **LLM 工程师** | 后端开发，刚接触 LLM | Tokenizer → 预训练 → SFT → RLHF → 量化 → 推理优化 | ~43h |
| 🟡 **AI Agent 工程师** | 有 LLM 经验，想做 Agent | Agent Loop → Memory → 多 Agent → 可靠性 → 生产部署 | ~80h |
| 🔴 **FDE 架构师** | 资深工程师/架构师 | 以上全部 + Transformer 深度 + 多模态 + 安全 | ~120h |

> 📖 详细路线见 [`learning-plans/`](learning-plans/)

---

## 怎么读

**不要从头读到尾。** 这不是小说。

1. 先看你的角色对应的学习计划（`learning-plans/` 里找）
2. 打开知识库索引（[`learning-plans/knowledge-index.md`](learning-plans/knowledge-index.md)），按主题速查
3. 每一篇的末尾都有最小可运行代码，**先跑通再理解**
4. 遇到不懂的名词 → 索引里搜 → 跳到对应的深度文章

---

## 致谢

本项目部分内容参考了 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)，一个非常优秀的英文 AI 工程课程。在此基础上的中文重构、分层编排和大量补充内容均为原创工作。

感谢原作者 Rohit Ghumare 提供的体系化框架。

---

## 施工状态

| 模块 | 状态 |
|------|------|
| LLM 从零开始（Tokenizer / 预训练 / SFT / RLHF / 量化 / 推理优化） | ✅ 已完成 |
| LLM 工程（提示工程 / RAG / 函数调用 / 微调 / 上下文化） | ✅ 已完成 |
| Agent 核心机制（循环 / 工具 / 记忆 / 技能库 / 编排） | ✅ 已完成 |
| Agent 框架（LangGraph / OpenAI SDK / Claude SDK / CrewAI） | ✅ 已完成 |
| Agent 可靠性（失败模式 / 评估 / 注入 / 可观测 / 门控 / 持久化 / 成本） | ✅ 已完成 |
| 架构师补充（Attention / 多模态 / RL / MoE / 安全对齐） | ✅ 已完成 |
| MCP 协议深度 | ✅ 已完成 |
| 初学者友好版 | 🚧 持续补充 |

---

## 贡献

欢迎提 Issue 和 PR。

目前最需要的帮助：
- 初学者友好版的补充和纠错
- 代码示例的本地化（改成本地可跑的版本）
- 术语翻译的讨论和改进

---

> **江湖路远，AI 不是一门"技术"，是一种新的工程思维方式。**
>
> ai-neigong 只是一个起点。真正练成的人，不是读完的人，是每一节都把代码跑通、把原理吃透的人。
>
> 这本是内功，会用是外功。内外兼修，方成大器。
>
> 祝你修炼有成。🦅

---

