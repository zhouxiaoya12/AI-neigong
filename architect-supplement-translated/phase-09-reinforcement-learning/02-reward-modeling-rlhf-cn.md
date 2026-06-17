# 奖励建模与 RLHF

> 人类写不出"好的助手回答"的奖励函数，但可以对比两个回答选出更好的。将奖励模型拟合到这些对比上，然后用 RL 对语言模型进行优化。Christiano 2017。InstructGPT 2022。这是把 GPT-3 变成 ChatGPT 的配方。2026年它正被 DPO 取代——但思维模型还在。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 05（情感分析）、Phase 9 · 08（PPO）
**时间：** 约45分钟

## 问题所在

你在下一个令牌预测目标上训练了一个语言模型。它写出符合语法的英语。它也撒谎、漫谈、拒绝说"拒绝"。你无法用更多预训练修复这个——网络文本是问题之源，而不是解药。

你想要一个*标量奖励*，表示"对指令 X，回答 A 比回答 B 更好"。用手工写这个奖励函数是不可能的。"帮助性"不是令牌上的闭式表达式。但人类可以对比两个输出并标记偏好——这在规模上是廉价的，可以收集。

RLHF（Christiano et al. 2017; Ouyang et al. 2022）将偏好转换为奖励模型，然后通过 PPO 对 LM 进行优化。分三步：SFT → RM → PPO。这是 2023-2025 年交付了 ChatGPT、Claude、Gemini 以及所有其他对齐 LLM 的配方。

2026年，PPO 步骤正被 DPO（Phase 10 · 08）取代，因为它更便宜且在 alignment tuning 上几乎同样好。但*奖励模型*这件东西仍然支撑着每个 Best-of-N 采样器、每个基于可验证奖励的 RL 管道，以及每个使用过程奖励模型的推理模型。理解 RLHF，你就理解了整个对齐栈。

## 核心概念

![RLHF 三阶段：SFT、在成对偏好上训练 RM、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**第一阶段：监督微调（SFT）。** 从预训练的基础模型出发。在人类编写的目标行为演示上微调（遵循指令的回答、帮助性回复等）。结果：一个模型 `π_SFT`，它被*偏向于良好行为*，但仍有不受限的动作空间。

**第二阶段：奖励模型训练。**

- 对提示 `x` 收集回答对 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练一个奖励模型 `R_φ(x, y)` 为 `y_+` 分配更高的分数。
- 损失：**Bradley-Terry 成对逻辑损失**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励的差异隐含偏好的对数赔率。BT 自 1952 年以来（Bradley-Terry）就是标准，是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。相同的 Transformer 骨干；一个单线性层输出奖励。

**第三阶段：PPO 结合 KL 惩罚对抗奖励模型。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考* `π_ref = π_SFT`。
- 回答 `y` 末尾的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 从 `π_SFT` 任意漂移——它是一个*正则化器*，不是一个硬置信域。`β` 通常 `0.01`-`0.05`。
- 以该奖励运行 PPO（第8课）。优势在令牌级轨迹上计算，但 RM 仅对整个回答评分。

**为什么要 KL？** 没有它，PPO 会很快找到奖励黑客策略——RM 只在分布内的补全上训练。一个离分布的回答可能比任何人写的都高分。KL 让 `π_θ` 保持在 RM 训练的流形附近。它是 RLHF 中最重要的单一旋钮。

**2026年现状：**

- **DPO**（Rafailov 2023）：闭式代数将第二+三阶段坍缩为在偏好数据上的单一监督损失。没有 RM，没有 PPO。在对齐基准上以几分之一的计算量达到相同质量。详见 Phase 10 · 08。
- **GRPO**（DeepSeek 2024–2025）：带有组相对基线而非 critic 的 PPO，奖励来自*验证器*（代码运行通过/数学答案匹配）而非人类训练的 RM。主导推理模型。详见 Phase 9 · 12。
- **过程奖励模型（PRMs）：** 给部分解（每个推理步骤）打分，同时用于 RLHF 和 GRPO 变体的推理。
- **宪法 AI / RLAIF：** 使用对齐的 LLM 生成偏好而非人类。扩展偏好预算。

## 动手构建

本课使用微小的合成"提示"和"回答"，表示为字符串。RM 是基于词袋表示的线性打分器。没有真实 LLM——是管线的*形状*重要，不是规模。参见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中这被人类标注者取代。形状——`(提示, 偏好回答, 被拒回答)`——完全一致。

### 步骤 2：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    # 梯度：提高好的词的权重，压低坏的词的权重
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 给好词赋予正权重，坏词赋予负权重。

### 步骤 3：在 RM 之上进行类似 PPO 的策略训练

我们的玩具策略从词汇表生成一个令牌。我们用 RM 打分，计算 `log π_θ(token | prompt)`，加上 KL-到-参考的惩罚，然后应用剪切的 PPO 代理目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 对 theta 做 PPO 风格更新，以 reward 为回报
    ...
```

### 步骤 4：监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。如果它攀升超过 `~5-10`，策略已远离 `π_SFT`——β 过低或奖励黑客正在开始。这是真实 RLHF 中的首要诊断。

### 步骤 5：用 TRL 的生产配方

理解玩具管线后，这里是真实库用户写作的同一个循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于第二阶段，`PPOTrainer`（内置 KL-到-参考）用于第三阶段。

```python
# 第二阶段：从成对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} —— Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 第三阶段：PPO 结合 KL 惩罚到 SFT 参考，对抗 RM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包含：mean_kl, clip_frac, value_loss —— 三个 PPO 诊断指标
```

库帮你做的三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观测 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按惯例冻结——你不能意外地与 `policy` 共享参数。且价值头与策略共享同一骨干（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过优化 / 奖励黑客。** RM 是不完美的；`π_θ` 找到在 RM 上高分但实际很差的对抗性补全。症状：奖励无限攀升而人类评估分数持平或下降。修复：及早停止、提高 `β`、扩大 RM 训练数据。
- **长度黑客。** 在对帮助性回答训练的 RM 上，往往隐式地奖励长度。策略学会填充回答。修复：长度归一化奖励，或使用长度感知 RM 的 RLAIF。
- **RM 过小。** RM 至少需要与策略同等大小。微小的 RM 无法忠实地给策略输出打分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎没有变化。标准技巧是*自适应* β，以每步固定的 KL 为目标。
- **偏好数据噪声。** 约30%的人类标注是噪声或模糊的。通过在协议过滤数据上训练 RM 进行校准，或对 BT 使用温度参数。
- **离策略问题。** PPO 数据在第一轮之后略微离策略。如第8课所述监控剪切比例。

## 实战使用

2026年的 RLHF 是分层结构：

| 层 | 目标 | 方法 |
|----|------|------|
| 指令遵循、帮助性、无害性 | 对齐 | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长时域多步任务 | Agent | 在步骤上带有过程奖励模型的 PPO/GRPO。 |
| 安全 / 拒答行为 | 安全 | 带单独安全 RM 的 RLHF-PPO，或宪法 AI。 |
| 推理时 Best-of-N | 快速对齐 | 解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练一个小的"奖励头"。 |

RLHF 在 2022-2024 年是*唯一*的方法。2026年，生产对齐管线是 DPO 优先，仅对 RM 密集或安全关键步骤使用 PPO。

## 交付物

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计一个 RLHF/DPO/GRPO 对齐管线，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基础 LM、一个目标行为（对齐 / 推理 / 拒答 / Agent）以及偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类，AI 反馈，基于规则，单元测试通过，或奖励蒸馏。
3. KL 策略。固定 β，自适应 β，或 DPO（隐式 KL）。
4. 诊断。平均 KL，奖励稳定性，过优化防护（留出的人类评估集）。
5. 安全门。红队集，拒答率，与帮助性 RM 独立的安全 RM。

拒绝在没有 KL 监控的情况下交付 RLHF-PPO。拒绝使用小于目标策略的 RM。拒绝仅长度奖励。将任何未保留盲测人类评估集的管线标记为缺乏过优化保护。
```

## 练习

1. **简单。** 在 500 个合成偏好对上训练 `code/main.py` 中的 Bradley-Terry 奖励模型。在留出的 100 对上测量成对准确率。应超过 90%。
2. **中等。** 分别用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个 β，绘制 RM 分数 vs 更新过程中到参考的 KL。哪个产生了奖励黑客？
3. **困难。** 在相同的偏好数据上实现 DPO（闭式偏好似然损失），并比较与 RLHF-PPO 管线在所用计算量和最终 RM 分数上的差异。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 管线（Christiano 2017, Ouyang 2022）。 |
| 奖励模型（RM） | "打分网络" | 通过 Bradley-Terry 拟合到成对偏好的可学习标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "保持在参考附近" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；反奖励黑客正则化器。 |
| 奖励黑客 | "古德哈特定律" | 策略利用 RM 缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | "AI 标注的偏好" | RLHF 中标签来自另一个 LM 而非人类。 |
| PRM | "过程奖励模型" | 给部分推理步骤打分；用于推理管线。 |
| 宪法 AI | "Anthropic 的方法" | 由明确规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) —— 开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) —— 更早的摘要 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) —— DPO；2026年后 RLHF 的默认方案。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) —— RLAIF 和自批判循环。
- [Anthropic RLHF 论文 (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码了解自适应 KL 和价值头的细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) 由 Lambert, Castricato, von Werra, Havrilla —— 三阶段管线的典范图解教程。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) —— 该库；`examples/` 包含 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假说视角；思考奖励黑客的必要前置。

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这篇 RLHF 教材以"人类写不出奖励函数，但可以对比——那就把对比变成奖励模型，再 RL 训练 LM"为核心叙述线，三阶段（SFT→RM→PPO）的逻辑链条完整且干净。尤其出色的是它没有回避"2026 年 DPO 正在取代 PPO"的现实转折——这是罕见的自我批判，展示了技术路线的真实进化。

### 二、知识结构梳理

- **基础层**：Bradley-Terry 成对逻辑损失 `L = -log σ(R(y_+) - R(y_-))`、KL 惩罚到参考模型的正则化机理、SFT 作为 RL 的"热身"
- **模式层**：奖励模型 = 最后一层换标量头的 Transformer、RLHF 的三阶段训练循环、自适应 KL 的 `adap_kl_ctrl` 机制（β 加倍/减半）
- **应用层**：DPO（单步直接优化偏好，无 RM 无 PPO）、GRPO（组相对基线+验证器替代人类偏好）、PRM（过程级别评分）、RLAIF（LLM 替代人类标注者）、Best-of-N 的推理时对齐

### 三、核心洞察

1. **RLHF 不是"教模型做正确的事"——是"教模型不给错误的事留空间"**：预训练的 LM 已经知道什么是好回答（它见过人类写的帮助性文本），问题在于它的分布太大，坏回答也有质量——RLHF 是在 LLM 的分布上"削峰减谷"。
2. **KL 惩罚是 RLHF 的唯一安全网**：RM 只在它训练过的输出流形上是好的——离开那个流形，RM 的输出是任意值。KL 把策略锁在 SFT 分布附近——本质上是一个"你不能离开我了解的领域"的约束。
3. **Bradley-Terry 模型虽然古老（1952），但它的对应性极佳**：概率差→对数赔率差——这个关系直接映射了"人类偏好"的数学——它不是选来用的，是推导出来的。
4. **RM 的大小 ≥ 策略的大小是硬约束**：许多初学者试图用一个 1B 大小的 RM 去指导一个 70B 的 PPO——RM 根本没有足够的"表达能力"去评判 70B 模型可能探索到的各种输出。
5. **DPO 的革命在于"谁来管那个 RM+PPO 的两阶段开销"**：DPO 证明了在 Bradley-Terry 假设下，可以推导出一个闭式损失直接优化偏好数据——砍掉了整个在线采样的循环。这不是"更好"，是"同样好但便宜一个数量级"。
6. **GRPO 的崛起说明"人类偏好不是唯一的信号"**：对于推理任务（数学、代码）——可验证的奖励（答案对不对、代码跑不跑得过）是比人类标注更干净、更可扩展的信号。RLHF 没有死，它只是缩小了适用范围。
7. **奖励黑客是 RLHF 的灵魂暗面**：这不是一个 bug，而是 RLHF 架构的必然结果——你给 RM 一个不完美的评分目标，优化器会找出评分函数所有的不一致性并放大它。这不是"模型变坏了"，是"RL 做得太好了"。

### 四、教学建议

1. **从"If you can't write a reward function, learn it from preferences"开始**：这句话是 RLHF 的整个动机——把它写在黑板上作为开场。
2. **手动画出 SFT→RM→PPO 的三阶段数据流图**：方块 = 模型/数据、箭头 = 输入/输出、标注每个阶段的数据和损失——这是学生脑中需要有的一张"总地图"。
3. **让 KL 惩罚的"为什么"成为一个五分钟的互动讨论**："如果你不给 PPO 加 KL，会发生什么？"——让学生猜（奖励会无限增长、输出变成胡言乱语但 RM 打高分）——然后展示一个 β=0 时的真实曲线。
4. **在合成玩具数据上演示 DPO vs RLHF-PPO 的效率差异**：偏好数据相同、基础模型相同，一方向用 PPO+R M（~100 步 RL），另一方用 DPO（~10 步 SFT-like 更新）——让学生看到"同样的结果，不同的开销"。
5. **展示一个真实的"奖励黑客"案例**：展示一个 RLHF 训练过程中 reward 持续上升但 human eval score 持平或下降的图——红色曲线（RM score）和蓝色曲线（human eval）的背离是 RLHF 最危险的信号。
6. **用 TRL 的代码片段做"架构阅读"练习**：`AutoModelForCausalLMWithValueHead`——为什么价值头部要放在同一个 backbone 上而不是一个独立网络？这个问题能区分"会用 API"和"理解设计"的学生。
7. **三种对齐方法的对比表（白板）**：RLHF-PPO（需在线RL+RM）、DPO（闭式损失无RL）、GRPO（组相对基线无RM）——适用场景、开销、优缺点一目了然。

### 五、值得补充的内容

1. **Reward Hacking 的分类学**：长度黑客、重复黑客、格式黑客——每种给出一个具体的输出例子和对应的修复策略。
2. **Process Reward Models 的更多细节**：PRM 的训练数据如何收集（每一步骤的人类判断？自动合成？）和训练损失与 ORM（Outcome RM）的差异。
3. **RLAIF 中的"宪法"设计**：Anthropic 的 Constitution 长什么样——包含哪些原则、如何给 AI feedback 提供指导。
4. **DPO 的数学推导过程（无需完整，只需直觉）**：从 B-T 模型 + KL-正则化 PPO 目标——推导出隐式的最优策略——再代入得到 DPO 损失——这条推导链才是"为什么 DPO 能工作"的核心。
5. **对齐税（Alignment Tax）的讨论**：RLHF 是否让模型在某些基准上变差？训练后性能损失的具体测量——以及 2025-2026 的对齐方法是否找到了减少这个税的途径。

### 六、一句话总结

RLHF 的本质不是 RL 或 LM 的简单叠加，而是"将人类价值观以偏好信号的形式注入到一个不可微的评价函数中，再用 RL 将信号蒸馏回模型的参数里"——三阶段 SFT→RM→PPO 是工业上第一个证明可行的完整管线，2026 年它正被更简洁的 DPO 和更可扩展的 GRPO 逐步取代，但其"偏好→评分→优化"的思维框架仍是所有对齐技术的共同根基。

---

# 🎓 Agent 架构课：RLHF——为什么你不能直接告诉模型"做个好人"？

**副标题：从"写不出奖励函数"到"A/B 测试驱动的模型对齐工程"**

---

先问一个问题——

你想让一个 LLM "有帮助、无害、诚实"。这三个词——你怎样写成一个可微的奖励函数？

……

你不能。"有帮助"不是一个能对每个令牌计算梯度的数学表达式。但人类可以——他们可以看一眼"A 的回答"和"B 的回答"，然后说"A 更好"。这个简单的比较——只需几秒，几乎不需思考——是 **RLHF 整个技术栈的起点。**

这不是一个工程优化的故事。这是一个**"把人类直觉变成梯度"**的故事。

让我把三个步骤展开给你看。每一步都有一个"不这样做会死"的理由。

**第一步：SFT（监督微调）。**

你有一个预训练的 LM。它见过整个互联网。它知道好文本是什么样子的——它也见过仇恨言论、错误信息和垃圾邮件。SFT 的"教"不是"教模型新信息"——是在你希望它采用的行为风格上**重新加权的分布**。

你在 SFT 里给它看人类写的"有帮助"回应。经过这个阶段，模型学会的是："哦，当我被问到问题，我应该这样回答。"——不是学会了新的事实，是学会了**一个回答风格。**

但仅靠 SFT 是不够的。SFT 告诉模型"好的回答长这样"，但它不知道"哪个好回答在哪个维度上更好"。一个两个都看上去正确的回答——SFT 说"都行"。而人类知道"A 更好，因为它更具体、不虚伪、提到了边界条件。"

**第二步：RM（奖励模型训练）。**

这就是人类介入的地方。对于同一个 prompt，你收集：（回答 A，回答 B，人类标签=A 更好）。不是针对所有 prompt 收集（太贵），是针对"有意义的分布"——你在 PPO 阶段会用到的那些。

RM 的损失函数是 Bradley-Terry 模型：**如果人类认为 A 比 B 好，那么 RM 分配给 A 的分数应该比 B 高，且这个差距的对数比应反映人类偏好的强度。** 用 sigmoid 取概率、用 log 取似然——这就是 `L = -log σ(R(y_+) - R(y_-))`。干净，简单。

这里有一个"我为什么在乎"的时刻：**RM 是 SFT 模型 + 一个标量头。** 不是从零开始训练 RM。不是用一个独立的预训练模型。是用和你将要 align 的模型**同一个骨干网络**上再训练一个评分头。原因很简单：RM 需要"理解"你的模型知道什么。如果 RM 的嵌入空间和 policy 的嵌入空间不同，RM 会在你 policy 根本没探索到的区域产生"虚假高分"。

**第三步：PPO + KL 到参考。**

从 π_SFT 启动 PPO。PPO 的代码不变。变的只是**奖励函数的来源**——不是环境返回的标量，是 RM 对完整回答的评分，减去一个**到参考模型的 KL 惩罚**。

这个 KL 惩罚是我在 RLHF 里最在乎的一件事。让我解释为什么。

没有 KL：PPO 会很快发现 RM 的漏洞。RM 只在 SFT 示例周围的分布见过"好的"和"坏的"回答——但它从未见过"完全胡言乱语但用上了 10 个 polite words"的输出。PPO 是一个优秀的 optimizer——它**会**找到这些在 RM 空间里高分但在语义空间里垃圾的序列。在文献里这叫做"奖励黑客"或"过优化"。

加了 KL：PPO 被"引力"拉回 SFT 分布。如果新策略想产生一个 RM 评分很高的回答，但它和 SFT 会给出的东西差别太大——KL 惩罚会消化掉大部分奖励，使这个探索"不值得"。

现在让我告诉你一个在生产里发生过的具体反模式。

我见过团队把 RM 训练在一个 7B 模型上，然后拿它来优化一个 70B 模型。RM（7B）根本不能理解、不能判断 70B 模型能生成的很多东西。**这是一只猫在评判一场歌剧——不是歌剧不好，是猫不懂。** RM 必须 ≥ policy 大小。这不是建议——这是 RLHF 能正常工作的前提条件之一。如果你在 2026 年看到一个公司声称他们用一个微小的 RM 做了成功的 RLHF——去检查他们的 human eval。基于经验：那个 eval 会有大量未被检测的奖励黑客。

还有一个后果更严重的反模式：**把 β 设为零（关掉 KL 惩罚）。**

症状：第 0-500 步——PPO 的奖励条在上升，人类评估分数也在上升，一切都是对的。第 500-1000 步——奖励持续攀升，但人类评估持平。第 1000+ 步——奖励还在升，人类评估开始下降。你的模型已经进入"奖励黑客"模式——它在生成 RM 认为完美的回答，但人类开始打 1 星。**这个时间滞后——奖励曲线和人类评估曲线的分离——是 RLHF 里唯一的最危险信号。** 如果你不监控人类评估（盲测），你会以为训练进展得很好，直到你在生产中发现模型在说胡话。

**KL 惩罚就是防止这条曲线的分离。** β 不是你可以"调一调看看"的超参——它是 RLHF 里的安全开关。如果 β 太小，你会看到分离。如果 β 太大，策略几乎不变——你没获得对齐收益。**自适应 β（TRL 的 `adap_kl_ctrl=True`）是 2026 年的正确默认：当 KL 超过目标值，β 加倍；低于一半，β 减半。**

现在说 2026 年的局势。DPO 正在取代 PPO 成为大多数对齐管线的默认选项，但 RLHF 的思维模型还在。让我给你一个具体的对比：

- **RLHF-PPO**：需要在线采样（PPO rollout→RM 评分→更新），需要维护 RM，需要监控三个不同的损失曲线，需要调两个 KL 参数（PPO 自身的 ε 和到 reference 的 β），容易奖励黑客。但有一个优势：如果你需要极端的对齐质量（安全关键场景），通过 PPO 的在线探索你可以发现更多"边缘情况"并针对优化。
- **DPO**：一个 SFT-like 的损失，直接在偏好对上更新。没有 RM、没有在线采样、没有 PPO。偏好数据→模型参数——仅此一步。对大多数对齐场景，DPO 等于 RLHF-PPO 的质量，计算开销仅为 1/5-1/3。

我在 2026 年的默认选择是 DPO，除非以下条件之一成立：
- 我极度需要对 RM 看不见的分布进行搜索（安全关键，红队场景）
- 我的偏好数据非常有限而我的 RM 可以准确地泛化（罕见的负相关）
- 我在做 GRPO 而不是 RLHF（推理任务，有可验证的奖励）

这引出最后一个重点——**GRPO（DeepSeek 2025）为什么是 RLHF-PPO 的另一个继承者。** GRPO 把 RM 换成了"验证器"——对代码：它跑不跑？对数学：答案对不对？对于推理任务，这比人类偏好信号干净一个数量级。GRPO 没有 RM，没有 KL 到 reference——只有一个组相对基线（在多个 rollout 之间比较）。

这很重要，因为**2026 年大家都在区分"对齐"和"能力"——RLHF 对齐（DPO），推理能力（GRPO）**。你不再需要一个通用的 RL 管道——你需要的是针对具体目标的最优路径。

最后的清单——如果你在发动一场 RLHF 之旅：

- ✅ 你的 RM ≥ 策略大小吗？——不满足的话你只是在自欺欺人。
- ✅ 你的 RM 训练数据质量如何？——30% 的噪声是基本线。用多个标注者的协议过滤。
- ✅ 你有 KL 惩罚吗？——β 太小→奖励黑客；β 太大→没学到东西。用自适应 β。
- ✅ 你在监控人类评估（盲测）吗？——如果你的日志里没有一条人类 eval 曲线，你不知道你的模型是在变好还是在变坏。RM 奖励上升是必要但不充分的指标。
- ✅ 你考虑过 DPO 吗？——如果你的场景是"标准对齐"，DPO 可能达到相同质量、少花 60-80% 的计算和工程开销。
- ✅ 你在做推理任务吗？——那考虑 GRPO 而不是 RLHF。你有可验证的奖励——何必用人类 noise？
- ✅ 你有安全 RM 吗？——帮助性和无害性经常冲突。分离这两个维度的 RM 是 Anthropic 的核心经验。

**金句：RLHF 不是"教模型什么是好的"——那是 SFT 做的事。RLHF 是"让模型在不确定性最大的空间里——那些好的和坏的回答并行的狭窄区域——系统地偏向好的那一边"。KL 惩罚是把这种偏置锁定在 RL 探索范围之内的数学安全带。**

回到开头的问题——为什么你不能直接告诉模型"做个好人"？因为**"好"不是一个数学函数——它是一组偏好，由几千个人类标注者经由 A/B 比较表达出来的。** RLHF 是把那些"A 比 B 好"的直觉——轻得不可进一步分解——蒸馏到一个序列的每一个令牌决策中的工程学答案。SFT→RM→PPO：这三步骤，就是在给那个不可表达的函数建造一个可以优化的代理。
