# 使用 LoRA & QLoRA 进行微调

> 全量微调一个 7B 模型需要 56GB 显存。你没有。大多数公司也没有。LoRA 让你可以通过训练不到 1% 的参数，在 6GB 显存中微调同一个模型。这不是妥协——它在大多数任务上匹配全量微调的质量。整个开源微调生态都运行在这一个技巧上。

**类型**：构建  
**语言**：Python  
**前置要求**：阶段 10 第 06 课（指令微调/SFT）  
**预计时间**：约 75 分钟  
**相关内容**：阶段 10 涵盖从零开始的 SFT/DPO 循环。本课将这些接入 2026 年的 PEFT 工具包（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过将低秩适配器矩阵（A 和 B）注入预训练模型的注意力层来实现 LoRA
- 计算 LoRA 相比全量微调的参数节省：秩 r 配合 d_model 维度，训练 2*r*d 个参数，而非 d² 个
- 使用 QLoRA（4 比特量化基础模型 + LoRA 适配器）微调模型，以适应消费级 GPU 显存
- 将 LoRA 权重合并回基础模型用于部署，并比较带适配器和不带适配器的推理速度

## 问题陈述

你有一个基础模型。Llama 3 8B。你希望它以你公司的口吻回答客户支持工单。SFT 是答案。但 SFT 有一个成本问题。

全量微调更新模型中的每一个参数。Llama 3 8B 有 80 亿个参数。以 fp16 存储，每个参数 2 字节。仅加载权重就需要 16GB。训练期间，你还需要梯度（16GB）、Adam 优化器状态（momentum + variance 共 32GB）以及激活值。总计：单个 8B 模型约需 56GB 显存。

一块 A100 80GB 勉强能装下。两块 A100 在云服务商那里每小时 3-4 美元。在 50,000 个样本上训练 3 个 epoch 需要 6-10 小时。每次实验 30-40 美元。跑 10 次实验调整超参数，在部署任何东西之前已经花了 400 美元。

把这个数字放大到 Llama 3 70B，就变得荒谬了。仅权重就要 140GB。你需要一个集群。每次实验 100 美元以上。

还有一个更深层的问题。全量微调修改模型中的每一个权重。如果你在客户支持数据上微调，可能会削弱模型的通用能力。这叫灾难性遗忘。模型在你的任务上变好了，但在其他所有事情上变差了。

你需要一个方法：训练更少的参数，使用更少的内存，同时不破坏模型已有的知识。

## 核心概念

### LoRA：低秩适应

Edward Hu 和微软的同事在 2021 年 6 月发表了 LoRA。论文的核心洞察：微调过程中的权重更新具有低的内在秩。你不需要更新一个 4096×4096 权重矩阵中的所有 1670 万个参数。更新中的有用信息可以被一个秩为 16 或 32 的矩阵捕获。

数学如下。一个标准线性层计算：

```
y = Wx
```

其中 W 是 d_out × d_in 矩阵。对于 4096×4096 的注意力投影，那是 16,777,216 个参数。

LoRA 冻结 W 并添加一个低秩分解：

```
y = Wx + BAx
```

其中 B 是 (d_out × r)，A 是 (r × d_in)。秩 r 远小于 d——通常是 8、16 或 32。

对于 r=16 在 4096×4096 层上：
- 原始参数量：4096 × 4096 = 16,777,216
- LoRA 参数量：(4096 × 16) + (16 × 4096) = 65,536 + 65,536 = 131,072
- 缩减：131,072 / 16,777,216 = 0.78%

你训练了 0.78% 的参数，得到 95-100% 的质量。

```
输入 x ──→ 冻结的 W (d×d) ──┐
输入 x ──→ A (r×d) → B (d×r) ──┼──→ 输出 y
                                 │
                          相加(合并)
```

A 用随机高斯分布初始化。B 初始化为零。这意味着 LoRA 的贡献从零开始——模型从原始行为开始训练，逐渐学到适配。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，控制低秩更新对输出的影响程度：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，缩放为 1×。当 alpha = 2r 时（常见的默认值），缩放为 2×。这个超参数独立于基础学习率控制 LoRA 路径的学习率。

实践指南：
- alpha = 2 * rank 是社区常见惯例（原论文大多数实验使用 alpha = rank）
- alpha = rank 给出 1× 缩放，保守但稳定
- 更高的 alpha 意味着每步更大的更新，可以加快收敛或导致不稳定

### LoRA 应该加在哪些层

Transformer 中有很多线性层。你不需要给所有层都加 LoRA。原论文测试了不同的组合：

| 目标层 | 可训练参数（7B） | 质量 |
|--------|----------------|------|
| 仅 q_proj | 470 万 | 良好 |
| q_proj + v_proj | 940 万 | 更好 |
| q_proj + k_proj + v_proj + o_proj | 1890 万 | 注意力最佳 |
| 所有线性层（注意力 + MLP） | 3770 万 | 边际增益，参数翻倍 |

大多数任务的甜点区间：q_proj + v_proj。这针对自注意力中的查询和值投影，控制模型关注什么以及提取什么信息。添加 MLP 层对代码生成等复杂任务有帮助，但在简单任务上将参数数量翻倍却获得递减的回报。

### 秩的选择

秩 r 控制适配的表达能力：

| 秩 | 可训练参数（每层） | 最适合 |
|----|------------------|--------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、摘要 |
| 16 | 131,072 | 多领域任务、指令跟随 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 大多数任务收益递减 |
| 128 | 1,048,576 | 极少有理由使用 |

Hu 等人表明 r=4 已经捕获了简单任务的大部分适配。r=8 和 r=16 是实践中最常见的选择。超过 r=64 很少提升质量，并开始丧失 LoRA 的内存优势。

### QLoRA：4 比特量化 + LoRA

华盛顿大学的 Tim Dettmers 及其同事在 2023 年 5 月发表了 QLoRA。思路：将冻结的基础模型量化为 4 比特精度，然后在上面以 fp16 附加 LoRA 适配器。

这戏剧性地改变了内存方程：

| 方法 | 权重内存（7B） | 训练内存（7B） | 所需 GPU |
|------|--------------|--------------|---------|
| 全量微调（fp16） | 14GB | ~56GB | 1× A100 80GB |
| LoRA（fp16 基础模型） | 14GB | ~18GB | 1× A100 40GB |
| QLoRA（4 比特基础模型） | 3.5GB | ~6GB | 1× RTX 3090 24GB |

QLoRA 做出了三个技术贡献：

**NF4（Normal Float 4 比特）**：一种专为神经网络权重设计的新数据类型。神经网络权重大致遵循正态分布。NF4 将其 16 个量化级别放置在标准正态分布的分位点上。这对于正态分布数据是信息论最优的。它比均匀 4 比特量化（INT4）或标准 Float4 损失更少的信息。

**双重量化**：量化常数本身也占用内存。每 64 个权重块需要一个 fp32 缩放因子（4 字节）。对于 7B 模型，这额外需要 0.4GB。双重量化将这些常数进一步量化为 fp8，将开销降至 0.1GB。听起来小，但积少成多。

**分页优化器**：训练期间，优化器状态（Adam 的 momentum 和 variance）在长序列上可能超过 GPU 内存。分页优化器使用 NVIDIA 的统一内存，在 GPU 内存耗尽时自动将优化器状态分页到 CPU RAM，需要时再分页回来。这以防止 OOM 崩溃为代价牺牲了一些吞吐量。

### 质量问题

减少参数或量化基础模型会损害质量吗？多篇论文的结果：

| 方法 | MMLU（5-shot） | MT-Bench | HumanEval |
|------|---------------|----------|-----------|
| 全量微调（Llama 2 7B） | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16（NF4） | 47.5 | 6.61 | 13.4 |
| QLoRA r=64（NF4） | 48.1 | 6.70 | 14.2 |

LoRA r=16 在大多数基准测试中与全量微调相差不到 1%。QLoRA r=16 再掉零点几个百分点。QLoRA r=64 基本上匹配全量微调，同时使用少 90% 的内存。

### 真实成本

在 50,000 个样本上微调 Llama 3 8B（3 个 epoch）：

| 方法 | GPU | 时间 | 成本 |
|------|-----|------|------|
| 全量微调 | 2× A100 80GB | 8 小时 | ~$32 |
| LoRA r=16 | 1× A100 40GB | 4 小时 | ~$8 |
| QLoRA r=16 | 1× RTX 4090 24GB | 6 小时 | ~$5 |
| QLoRA r=16（Unsloth） | 1× RTX 4090 24GB | 2.5 小时 | ~$2 |
| QLoRA r=16 | 1× T4 16GB | 12 小时 | ~$4 |

在单张消费级 GPU 上 QLoRA 的成本比一顿午餐还便宜。这就是为什么开源重量级微调社区在 2023 年爆发式增长，也是为什么到 2026 年下面每个训练框架默认都支持 QLoRA。

### 2026 年 PEFT 技术栈

| 框架 | 是什么 | 什么时候选 |
|------|--------|----------|
| **Hugging Face PEFT** | 标准的 LoRA/QLoRA/DoRA/IA3 库 | 你需要原始控制，且训练循环已经在 `transformers.Trainer` 上 |
| **TRL** | HF 的基于反馈的训练器（SFT、DPO、GRPO、PPO、ORPO） | 你在 SFT 之后需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | 基于 Triton kernel 重写的前向/反向传播 | 你想要 2-5× 加速 + 一半显存且无精度损失；Llama/Mistral/Qwen 系列 |
| **Axolotl** | PEFT + TRL + DeepSpeed + Unsloth 的 YAML 配置封装 | 你需要可重现、版本控制的训练运行 |
| **LLaMA-Factory** | PEFT + TRL 上的 GUI/CLI/API | 你想要零代码微调；支持 100+ 模型家族 |
| **torchtune** | 原生 PyTorch recipe，无 `transformers` 依赖 | 你追求最小依赖，组织已经标准化 PyTorch |

经验法则：研究或一次性实验 → PEFT。可重复的生产流水线 → 开启 Unsloth kernel 的 Axolotl。快速原型 → LLaMA-Factory。

### 合并适配器

训练后，你有两个东西：冻结的基础模型和一个小的 LoRA 适配器（通常 10-100MB）。你可以：

1. **保持分离**：加载基础模型，在上面加载适配器。为不同任务交换适配器。这就是如何从一个基础模型服务多个微调变体。

2. **永久合并**：计算 W' = W + (alpha/r) * BA 并将结果保存为新的完整模型。合并后的模型与原始模型大小相同。无推理开销。无需管理适配器。

服务多个任务（客户支持适配器、代码适配器、翻译适配器）→ 保持分离。部署单个专用模型 → 合并。

合并多个适配器的高级技术：

- **TIES-Merging**（Yadav 等人 2023）：剪枝小幅值参数，解决符号冲突，然后合并。减少适配器之间的干扰。
- **DARE**（Yu 等人 2023）：合并前随机丢弃适配器参数并重新缩放剩余部分。惊人有效地组合能力。
- **任务算术**：简单地对适配器权重相加或相减。将"代码"适配器和"数学"适配器相加，往往产生两者都擅长的模型。

### 什么时候不该微调

微调是第三个选项，不是第一个。

**第一：提示工程。** 写更好的系统提示词。加少样本示例。用思维链。零成本，几分钟完成。如果提示词能让你达到 80% 的效果，你可能不需要微调。

**第二：RAG。** 如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比将其烘焙进权重更便宜、更易维护。见第 06 课。

**第三：微调。** 当你需要模型采用特定风格、格式或推理模式，而仅靠提示词无法达到时使用。当你需要一致的结构化输出时。当你需要将大模型蒸馏为小模型时。当延迟很重要且你无法承受少样本提示词的额外 token 开销时。

```
需要更好的模型行为？
  → 试提示工程
    → 有效 → 发布
    → 不够 → 需要外部知识？
              → 是 → 构建 RAG 管线
              → 否，需要风格/格式改变 → 用 LoRA/QLoRA 微调
```

---

## 动手构建

我们用纯 PyTorch 从零实现 LoRA。没有库。没有魔法。你将构建 LoRA 层，将其注入模型，训练它，然后合并权重。

### 第 1 步：LoRA 层

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: 随机初始化  B: 零初始化  → BA 从零开始
        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 用缩放后的随机值初始化。B 初始化为零。乘积 BA 从零开始，因此模型从原始行为开始。

### 第 2 步：LoRA 包装的线性层

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear        # 冻结的原始权重
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始线性层被冻结。只有 LoRA 参数（A 和 B）是可训练的。

### 第 3 步：将 LoRA 注入模型

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    # 第一步：冻结整个模型
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查该层名称是否匹配目标模块
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                # 获取父模块
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model

                # 替换为 LoRA 包装版本
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

先冻结模型中每个参数。然后遍历模型树，找到匹配目标名称的线性层，用 LoRA 包装版本替换。LoRA 的 A 和 B 矩阵是整个模型中仅有的可训练参数。

### 第 4 步：参数计数

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### 第 5 步：将权重合并回去

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                # 计算合并后的权重: W + (alpha/r) * BA
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T

            # 替换回普通线性层
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并后，LoRA 层消失。模型与原始大小相同，适配已烘焙进权重。无推理开销。

### 第 6 步：模拟 QLoRA 量化

```python
def quantize_to_nf4(tensor, block_size=64):
    # 将张量重塑为块
    blocks = tensor.reshape(-1, block_size)
    # 计算每块的缩放因子
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    # 量化：映射到 [-8, 7] 范围
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    # 反量化回浮点数
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这通过将权重映射到 64 个块内的 16 个离散级别来模拟 4 比特量化。生产环境 QLoRA 使用 bitsandbytes 库实现 GPU 上的真正 NF4。

### 第 7 步：训练循环

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    # 只优化可训练参数（LoRA 的 A 和 B）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### 第 8 步：完整演示

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    # 构建一个小模型
    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    # 生成伪数据
    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    # 注入前参数统计
    params_before = count_parameters(model)

    # 注入 LoRA 到第 0 和第 2 层
    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    # 注入后参数统计
    params_after = count_parameters(model)

    # 训练
    losses = train_lora(model, data, epochs=20, lr=1e-3)

    # 合并权重并统计
    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

演示创建一个小模型，将 LoRA 注入两个层，训练它，然后将权重合并回去。参数数量从全量可训练降至 LoRA 训练期间的约 1% 可训练，合并后恢复原始架构。

## 实际使用

使用 Hugging Face 生态，在真实模型上使用 LoRA 大约只需 20 行：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 QLoRA，添加 bitsandbytes 量化：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这些。相同的训练循环。相同的数据管线。基础模型现在以 4 比特存在，LoRA 适配器以 fp16 训练，整个东西装在 6GB 显存里。

使用 Hugging Face Trainer 训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存的适配器只有 10-100MB。基础模型保持不动。你可以在 Hugging Face Hub 上分享适配器，无需重新分发完整模型。

## 交付物

本课产出：
- `outputs/prompt-lora-advisor.md`——帮助你为特定任务决定 LoRA 秩、目标模块和超参数的提示词
- `outputs/skill-fine-tuning-guide.md`——教导 Agent 何时以及如何微调的决策树技能

## 练习

1. **秩消融研究。** 使用秩 2、4、8、16、32 和 64 运行演示。绘制最终损失 vs 秩的关系图。找到收益递减点——加倍秩不再将损失减半的点。对 256 维特征的简单分类任务，这应该在 r=8-16 左右。

2. **目标模块比较。** 修改 inject_lora 分别只针对层 "0"、只针对层 "2"、只针对层 "4"，以及三者全部。训练每个变体 20 个 epoch。比较收敛速度和最终损失。这反映了真实场景中针对 q_proj vs v_proj vs 所有线性层的决策。

3. **量化误差分析。** 获取训练后模型的权重矩阵在 quantize_to_nf4 / dequantize_from_nf4 前后的对比。计算均方误差、最大绝对误差，以及原始权重与重建权重之间的相关性。用 block_size 值 32、64、128 和 256 做实验。

4. **多适配器服务。** 在数据的不同子集上训练两个 LoRA 适配器（偶数索引 vs 奇数索引）。保存两个适配器。加载一次基础模型，然后切换适配器，验证每个适配器在相同输入上产生不同输出。这就是生产系统如何从一个基础模型服务多个微调模型。

5. **合并 vs 未合并推理。** 在相同的 100 个输入上比较 merge_lora_weights 前后 LoRA 模型的输出。验证输出一致（在 1e-5 的浮点容差内）。然后对两种方式的推理速度进行基准测试——合并后应该稍快，因为是一个矩阵乘法而非两个。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| LoRA | "高效微调" | 低秩适应：冻结基础权重，训练两个小矩阵 A 和 B，其乘积近似完整的权重更新 |
| QLoRA | "在笔记本上微调" | 量化 LoRA：以 4 比特 NF4 加载基础模型，在上面以 fp16 训练 LoRA 适配器，使 7B 微调可以在 6GB 显存中完成 |
| 秩 (r) | "模型能学多少" | A 和 B 矩阵的内部维度；控制表达能力 vs 参数数量 |
| Alpha | "LoRA 学习率" | 应用于 LoRA 输出的缩放因子；alpha/r 缩放适配对最终输出的贡献 |
| NF4 | "4 比特量化" | Normal Float 4：一种 4 比特数据类型，量化级别位于正态分布分位点，对神经网络权重最优 |
| 适配器 | "小训练部分" | 保存为单独文件（10-100MB）的 LoRA A 和 B 矩阵，可加载到任何一份基础模型上 |
| 目标模块 | "哪些层应用 LoRA" | 注入 LoRA 适配器的特定线性层（q_proj、v_proj 等） |
| 合并 | "烘焙进去" | 计算 W + (alpha/r) * BA 并替换原始权重，消除推理时的适配器开销 |
| 分页优化器 | "训练时不 OOM" | 当 GPU 内存耗尽时将优化器状态（Adam momentum、variance）卸载到 CPU |
| 灾难性遗忘 | "微调搞坏了其他一切" | 更新所有权重导致模型丢失先前学到的能力 |

## 拓展阅读

- Hu 等人，"LoRA: Low-Rank Adaptation of Large Language Models"（2021）——引入低秩分解方法的原始论文，在 GPT-3 175B 上测试，秩低至 4
- Dettmers 等人，"QLoRA: Efficient Finetuning of Quantized Language Models"（2023）——引入 NF4、双重量化和分页优化器，使 65B 微调可以在单张 48GB GPU 上完成
- PEFT 库文档（huggingface.co/docs/peft）——Hugging Face 生态中 LoRA、QLoRA 和其他参数高效方法的标准库
- Yadav 等人，"TIES-Merging: Resolving Interference When Merging Models"（2023）——组合多个 LoRA 适配器而不降低质量的技术
- [Rafailov 等人，"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"（NeurIPS 2023）](https://arxiv.org/abs/2305.18290)——DPO 推导；SFT 之后的偏好微调阶段，无需奖励模型
- [TRL 文档](https://huggingface.co/docs/trl/)——SFTTrainer、DPOTrainer、KTOTrainer 以及它们与 PEFT/bitsandbytes/Unsloth 的集成面的官方参考
- [Unsloth 文档](https://docs.unsloth.ai/)——将微调吞吐量翻倍、内存减半的融合 kernel；TRL 之下的性能层
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/)——YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；手写脚本的配置即代码替代方案

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是整个微调系列中最具"实用性"的一课——它回答的是工程师每天早上醒来都要面对的问题："我只有一张消费级显卡，能不能微调 7B 模型？"从 56GB 显存的压倒性数字开场，到 6GB 就能跑的 QLoRA，整个过程是一条清晰的能力解放线。数学推导（BA 分解替代完整 W 更新）、实证数据（三个基准测试的量化对比）、成本核算（从 $32 到 $2）三者交错推进，每一段都在回答"这能省多少"。2026 年 PEFT 技术栈的框架选择指南和实践决策树（"什么时候不该微调"）是本课最具工程智慧的部分。

### 二、知识结构梳理

**认知基础层**：全量微调的内存方程（权重+梯度+优化器+激活=56GB）、低秩假设（更新的内在秩远小于维度）、LoRA 的数学分解 y = Wx + BAx 及其参数效率（0.78%）

**工程模式层**：秩-质量-参数三元权衡、目标模块选择（q_proj+v_proj 甜点区间）、alpha 缩放因子的独立学习率控制、QLoRA 的三项技术创新（NF4/双重量化/分页优化器）、合并 vs 分离的部署策略

**实践应用层**：2026 年六框架选择指南（PEFT→TRL→Unsloth→Axolotl→LLaMA-Factory→torchtune）、真实成本对比表（$32→$2）、三选项决策树（提示工程→RAG→微调）、多适配器合并的高级技术（TIES/DARE/任务算术）

### 三、核心洞察（备课时的关键理解）

1. **LoRA 不是"少训练一点"——是"聪明地重新表达更新"。** 关键洞察不在参数效率，而在低秩假设：权重更新不需要 1670 万维度，秩 16 就够。这是对微调本质的重新理解，不是工程妥协。

2. **B 矩阵初始化为零是整个机制中最精妙的设计。** 因为 BA 从零开始，模型从原始行为出发，逐渐学习适配。没有这个设计，LoRA 会在训练一开始就破坏模型的预训练知识——那就跟全量微调一样的灾难性遗忘问题。

3. **Alpha/r 是独立的"适配学习率"。** 很多人把它当死参数。其实它让你可以在不改变基础学习率的情况下，单独控制 LoRA 更新的幅度。alpha=2r 意味着 LoRA 更新以 2× 倍率进入模型，这可以加快收敛。理解这一点的人调参效率高十倍。

4. **QLoRA 的三项技术创新不是并列的——NF4 是最核心的。** 双重量化和分页优化器是锦上添花（各省 0.3GB），NF4 是真正的突破——它基于信息论在正态分布数据上达到最优量化。LLM 权重恰好是正态分布的，这简直是天作之合。

5. **"什么时候该微调"这个决策树比任何参数配置都重要。** 太多工程师跳过提示工程和 RAG，直接冲微调。三选项树（提示→RAG→微调）是工程纪律——微调是锤子，但不是所有问题都是钉子。

6. **Unsloth 的 2-5× 加速不只是性能优化——它改变了实验的经济学。** 从 6 小时降到 2.5 小时意味着一天可以跑 4-6 组实验而不是 1-2 组。实验速度决定了学习速度。这就是为什么 2026 年几乎没人不用 Unsloth。

7. **多适配器服务的产品模型被严重低估。** 一个基础模型 + 10 个 50MB 适配器 vs 10 个 14GB 的完整微调模型。存储、部署、切换——每一项都差了两个数量级。这是架构决策，不是实现细节。

### 四、教学建议

1. **开场做内存计算而不是讲概念。** 让学生自己算：8B 参数 × 2 字节 × 4（权重+梯度+momentum+variance）= 56GB。算出这个数字，学生自己就知道全量微调不可行。动机从计算中产生，不是从讲解中产生。

2. **LoRA 的 BA 分解必须在黑板上画图。** 画一个大的 W 矩阵（4096×4096），然后在旁边画两个瘦条 A（16×4096）和 B（4096×16），把它们乘起来，让学生看到面积对比。视觉冲击力远超公式。

3. **秩消融实验让数据说话。** 当场跑 r=2, 4, 8, 16, 32, 64 的训练，画损失-vs-秩曲线。学生自己就能指出"r=16 之后再翻倍就没用了"。发现的快乐来自数据，不是被告知。

4. **QLoRA 的三个技术要分层讲。** NF4 是核心（"为什么 4 比特能保持质量"），双重量化是细节（"连常数也要省"），分页优化器是保底（"防止 OOM"）。不要并排灌输，学生记不住。

5. **成本表的冲击力要靠对比。** 把 $32（全量微调）和 $2（QLoRA+Unsloth）并排贴在黑板上。然后问："今天中午的午饭花了多少钱？"这个对比讲一次，学生永生难忘。

6. **六框架的选择指南要做成决策树卡片给学生打印。** 不是所有学生都需要记住每个框架——他们需要的是一个快速的"我该用哪个"决策流程。研究→PEFT，生产→Axolotl+Unsloth，原型→LLaMA-Factory。

7. **练习 2（目标模块对比）是最有教学价值的练习。** 让学生分别只训练 q_proj、v_proj、k_proj、o_proj——然后比较最终损失。学生会直观理解为什么 q_proj+v_proj 是甜点区间。不是为了得到答案，是为了理解 tradeoff。

### 五、值得补充的内容

1. **DoRA（Weight-Decomposed Low-Rank Adaptation）的简要提及。** 2024 年提出的改进，将权重更新分解为方向和大小两部分，在某些任务上比 LoRA 提升 1-3%。值得作为 LoRA 变体提一嘴。

2. **多 GPU 场景下的 LoRA 训练。** 全量微调可以用模型并行（tensor/pipeline parallelism），但 LoRA 通常单卡就够了。如果确实需要多卡，DeepSpeed ZeRO-3 与 LoRA 的兼容性问题值得提醒。

3. **LoRA 适配器的安全性。** 因为适配器文件如此之小，它成为分发恶意权重的理想载体。从 Hugging Face Hub 加载适配器时是否有安全检查？这是一个被忽视的安全话题。

4. **学习率预热（warmup）在 LoRA 中的特殊考虑。** 因为 B 初始化为零，LoRA 需要一定步数来"激活"适配路径。没有足够预热可能导致前几个 step 几乎不学习。这个细节在生产中很重要。

### 六、一句话总结

**LoRA 不是让微调变便宜——是让微调变成每个人都能做的事。从 56GB 到 6GB，从 $32 到 $2，从需要数据中心到跑在笔记本上。真正的突破不在算法复杂度，在于让能力的门槛消失。**

---


---

# 🎓 Agent 架构课：LoRA & QLoRA——让微调从数据中心走到你的笔记本

同学们好。

前面几节课我们一直在讲怎么在不改变模型的情况下让它变得有用——RAG 给它参考书，函数调用给它双手，上下文工程给信息排座次。今天，我们要做一件之前一直在回避的事：**改变模型本身。**

但改变模型有一个巨大的障碍：钱。不是隐喻意义上的贵，是真的贵到不可行。一张 A100 80GB 要上万美元。在云上跑一次全量微调 Llama 7B：$30-40。调 10 次超参数找到最优配置：$400。而你需要的结果可能只是"让模型用客服的语气回答问题"。

今天的核心问题是：**有没有一种方法，只改变模型的一小部分，用消费级显卡，花一顿饭的钱，就能让模型学会新任务——而且不把原来的能力毁掉？**

答案是 LoRA。这是我见过的所有 AI 研究中最优雅、最实用的想法之一。不是因为它数学有多复杂——恰恰相反。

## 全量微调的代价：先算一笔账

在讲 LoRA 之前，先理解它要解决什么。全量微调 Llama 3 8B。8B = 80 亿参数。fp16 = 每参数 2 字节。

加载权重：16GB。这是你至少要有的显存。

训练时还需要：梯度（16GB，每个参数需要对应的梯度）、Adam 优化器状态（32GB——momentum 和 variance 各一份）。加上激活值和其他开销。

总共：约 56GB。

一张 A100 80GB 勉强能装下。但 56GB 已经接近单卡的极限——你几乎没法增大 batch size，没法加载更长的序列，稍微大一点的模型就爆显存。

而且还有一个更深的问题：**全量微调会修改每一个参数。** 你花了几个月、数十万美元预训练出来的通用能力——数学、常识、多语言能力——可能因为你在客户支持数据上微调而退化。这叫灾难性遗忘。模型在你的任务上变好，在所有其他事情上变差。

## LoRA 的核心洞察：更新的秩很低

2021 年，微软的 Edward Hu 提出了一个想法。这个想法改变了整个微调行业。

全量微调修改一个 4096×4096 的权重矩阵——1670 万个参数。但 Hu 发现，这些更新中包含的**有用信息**，可以用秩 16 或 32 来捕获。换句话说，1670 万维度的更新中，真正有效的"变化方向"只有十几个维度。

这是什么意思？

想象你在微调过程中产生了这个矩阵 ΔW——表示每个权重需要改变多少。ΔW 是 4096×4096。但它的本质信息量只是秩 16。也就是说，你可以用两个很小的矩阵 A 和 B 来近似它：

```
ΔW ≈ B × A
```

其中 B 是 4096×16，A 是 16×4096。两个矩阵加起来只有 131,072 个参数——占原来的 0.78%。

LoRA 的做法：冻结原始权重 W，在旁边放两个小矩阵 A 和 B。输出变成：

```
y = Wx + BAx
```

训练时只更新 A 和 B。W 纹丝不动。推理时可以把 BA 算出来加到 W 上，变成一个标准矩阵乘法——零额外开销。

**这是 LoRA 最优雅的地方：训练时 99% 的参数不动，推理时完全恢复原样。你得到的微调模型和全量微调一样大、一样快——只是获取它的方式便宜了 100 倍。**

## A 和 B 怎么初始化？——B 初始化为零是整个系统的保险丝

这是 LoRA 设计中最巧妙的一个细节。

A 用随机高斯分布初始化。B 初始化为零。

BA 的乘积从零开始。训练刚开始时，LoRA 对模型输出没有任何影响——模型和微调前行为完全一样。然后随着训练推进，B 逐渐学到非零值，适配慢慢"长"出来。

为什么这很重要？因为如果 A 和 B 都用随机初始化，BA 一开始就会产生一个随机的扰动加到模型上。模型已有的预训练能力会被这个随机噪声破坏。你可能花了 3 个 epoch 的 80% 时间来"修复"这个初始破坏。

B=0 → BA=0 → 从原点出发。这是 LoRA 在训练稳定性上的核心保障。

## 秩的选择：r 到底该设多大？

这是每个用 LoRA 的人都会问的第一个问题。

r=4 已经能捕获简单任务（情感分析、简单分类）的大部分适配。r=8 是社区最常用的默认值——足够表达，参数可控。r=16 是多领域任务和指令跟随的甜点区间。r=32 给复杂推理和代码生成。

r=64 以上？在绝大多数任务上收益递减。从 r=32 翻倍到 r=64，参数翻倍但质量提升不到 1%。从 r=64 再翻到 r=128，提升几乎不可见——但你已经失去了 LoRA 的内存优势。

第一条规则：从 r=8 或 r=16 开始。除非证据确凿，不要往上加。

## Alpha——很多人忽略的第二个学习率

除了秩，还有一个参数控制 LoRA 的行为：alpha。

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，这个比值是 1。当 alpha = 2r（常见默认值），比值是 2——LoRA 更新以 2× 的力度进入模型。

这实际上是一个**独立于基础学习率的 LoRA 路径学习率**。你可以保持基础学习率不变，通过调整 alpha 来控制适配有多"激进"。

实践中：alpha = 2r 是稳健的起点。如果你想加速收敛且数据干净，可以试试 alpha = 4r。如果数据有噪声或你在做多任务微调，alpha = r 更保守可控。

## QLoRA：当 4 比特遇上 LoRA

LoRA 已经把训练参数降到了不到 1%。但基础模型本身还是要加载到显存里——fp16 下 7B 就是 14GB。

QLoRA 问了下一个问题：基础模型能不能也不占那么多内存？

答案是：把基础模型量化到 4 比特。

不是普通的 4 比特量化，是 NF4——Normal Float 4。Tim Dettmers（顺便说一句，他也是在 UW 读博时做的，前后脚和 LoRA 团队几乎同时在推动 PEFT）设计了一种专门针对神经网络权重的 4 比特格式。

核心洞察：神经网络权重近似服从正态分布。标准均匀量化把所有值等距分成 16 档——浪费了很多档位在几乎不出现的极值上。NF4 把 16 个档位放在正态分布的分位点上——在权重密集的地方放更多档位，在尾部分布稀薄的地方放更少。

这听上去像是微调，但实际差距很大。NF4 在相同比特数下损失的信息远少于 INT4。

加上双重量化（量化的量化——连缩放常数都压缩到 fp8）和分页优化器（优化器状态溢出时自动卸载到 CPU），QLoRA 把 7B 模型的训练内存从 56GB 压到了 6GB。

**6GB。这是一个 RTX 3060 的显存。一张二手卡几百块。** 这就是 QLoRA 的意义——不是让企业省钱，是让任何人有 GPU 就能微调大模型。

## 质量到底行不行？

每次我讲 QLoRA，都有人说"肯定质量打折"。看数据：

- 全量微调 Llama 7B，MMLU 48.3
- LoRA r=16，MMLU 47.9（差 0.4）
- QLoRA r=16（NF4），MMLU 47.5（差 0.8）
- QLoRA r=64（NF4），MMLU 48.1（差 0.2）

QLoRA r=64 基本追平全量微调，用的内存是 6GB vs 56GB。成本是 $2 vs $32。

这不是妥协。这是用 10% 的资源拿到 99.5% 的结果。在工程上这叫效率，不叫打折。

## 微调之后怎么办？合并还是分离？

训练完 LoRA，你有一个 14GB 的基础模型和一个 50MB 的适配器文件。

**分离模式**：加载基础模型，在它上面加载适配器。想切任务？换个适配器就行。一个基础模型可以服务客户支持、代码生成、翻译三种场景——每种一个 50MB 适配器。这在多租户服务中是革命性的架构。

**合并模式**：把 BA 算出来加到 W 上，保存为一个新的完整模型。适配器消失，推理零开销，到处分发。适合部署单一专用模型。

现代系统甚至可以做**动态合并**：把代码适配器加上数学适配器，得到一个两样都行的模型。TIES-Merging 和 DARE 是专门处理多适配器合并时冲突的技术——剪掉小幅更新、解决符号冲突、重新缩放。你甚至可以简单地做任务算术：代码适配器 + 数学适配器 - 翻译适配器 = 会写代码和数学但翻译稍弱的模型。这不是比喻，是真的可以用向量加减来组合能力。

## 什么时候你不该微调？

这是我在每节架构课上必须强调的——微调不是第一个选项。

**先用提示词。** 写好系统提示词，加几个示例。零成本，几分钟。如果提示词能做到 80%，你可能不需要微调。

**然后用 RAG。** 如果模型需要的是你公司的特定数据——文档、知识库、产品信息——检索比烘焙进权重更便宜、更可维护。文档更新了不需要重新训练。

**最后才微调。** 当你需要模型采用特定的风格、格式或推理模式时。当你需要一致性结构化输出时。当你需要将大模型蒸馏成小模型时。当延迟要求你不允许在提示词里塞几百个示例时。

微调是锤子。不是每个问题都是钉子。

## 2026 年该用什么框架？

现在微调生态非常成熟，基本按场景分工：

**研究和一次性实验**：直接用 Hugging Face PEFT + transformers。最灵活，什么都能改。

**可重复的生产流水线**：Axolotl。YAML 配置文件搞定一切，版本控制友好，默认集成 Unsloth kernel（2-5× 加速）。

**零代码原型**：LLaMA-Factory。Web UI 点点鼠标就微调。适合快速验证想法。

**需要偏好对齐（DPO/GRPO）**：TRL。在 SFT 之后跑偏好微调。和 PEFT 无缝集成。

**追求极致性能**：Unsloth。Triton-kernel 重写了前向/反向传播，微调速度翻倍、显存减半。不是优化，是改装引擎。

经验法则：如果这是你第一次做微调 → LLaMA-Factory。如果你在写论文 → PEFT。如果你在搭产品 → Axolotl + Unsloth。

## 总结：Agent 架构师的微调检查清单

1. ✅ **必要性**：提示词试过了吗？RAG 试过了吗？确定需要微调吗？
2. ✅ **内存**：QLoRA 能在你的 GPU 上跑吗？6GB 是硬门槛。
3. ✅ **秩**：从 r=8 或 r=16 开始。不要默认 r=64。
4. ✅ **目标层**：q_proj + v_proj 是甜点区间。需要更强再加 MLP 层。
5. ✅ **Alpha**：alpha = 2r 是稳健起点。需要加速收敛调大，数据有噪声调小。
6. ✅ **框架**：选对工具——研究用 PEFT，生产用 Axolotl+Unsloth。
7. ✅ **部署**：多任务分离适配器，单任务合并权重。
8. ✅ **评估**：任务表现提升的同时，通用能力有没有退化？

记住一件事：**LoRA 不是让微调变得更好——LoRA 让微调从"只有大公司能做的事"变成了"任何有显卡的人都能做的事"。56GB 降到 6GB，$32 降到 $2。真正改变行业的不是算法复杂度，是让门槛消失。**

---

