# KV 缓存、Flash Attention 与推理优化

> 训练是并行的，受限于算力。推理是串行的，受限于内存。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 7 · 02（自注意力）、Phase 7 · 05（完整 Transformer）、Phase 7 · 07（GPT）
**时间：** 约75分钟

## 问题所在

一个朴素的自回归解码器在生成 N 个令牌时需要 O(N²) 的计算量：每一步都要在整个前缀上重新计算注意力。对于4K令牌的响应，这是1600万次注意力操作，其中大多数是冗余的。前缀令牌的每个隐藏状态一旦计算完成就是确定性的——你只需要用新令牌的查询去对之前所有已缓存的键和值进行注意力运算。

在此之上，注意力本身移动了大量数据。标准注意力会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写太多。当 N≥2K 时，注意力在算力受限之前就已经内存受限了。经典注意力内核对现代 GPU 的利用率只有 10%-25%。

两项都来自 Dao 等人的优化将前沿推理从"慢"推向"快"：

1. **KV 缓存。** 存储每个前缀令牌的 K 和 V 向量。每个新令牌的注意力是：一次查询对标已缓存的键。推理从每步 O(N²) 降到 O(N)。
2. **Flash Attention。** 将注意力计算分块（tiling），使得完整的 N×N 矩阵永远不会触及 HBM。所有 softmax + 矩阵乘法都在 SRAM 内完成。A100 上2-4倍墙钟速度提升；H100 上加 FP8 可达5-10倍。

到2026年，两者都是标准配置。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认启用它们。每个前沿模型都附带 Flash Attention。

## 核心概念

![KV 缓存增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每个解码器层、每个令牌、每个头：

```
每层每令牌字节数 = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于7B模型，32层，32个头，d_head=128，fp16：

```
每层每令牌 = 2 * 128 * 2 = 512 字节
每令牌（32层） = 16 KB
每32K上下文 = 512 MB
```

对于 Llama 3 70B（80层，d_head=128，GQA，8个 KV 头）：

```
每层每令牌 = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
每32K上下文 = 10.4 GB
```

这10 GB就是为什么 Llama 3 70B 在128K上下文时，仅 KV 缓存就需要吃掉 A100 40GB 的大部分，即使批次大小为1。

**GQA 是 KV 缓存的制胜法宝。** 64个头的 MHA 会是32 GB。MLA 压缩得更多。

### Flash Attention——分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读取，N×N，HBM 写入）
P = softmax(S)       （HBM 读取，HBM 写入）
O = P @ V            （HBM 读取，HBM 写入）
```

三次 HBM 往返。在 H100 上，HBM 带宽是 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返相比把所有数据放在片上都慢一个数量级。

Flash Attention：

```
对 Q 的每个块（块大小~128×128）：
    将 Q_tile 加载到 SRAM
    对 K、V 的每个块：
        将 K_tile、V_tile 加载到 SRAM
        在 SRAM 中计算 S_tile = Q_tile @ K_tile^T
        在 SRAM 中进行运行态 softmax 聚合
        在 SRAM 中累加到 O_tile
    将 O_tile 写回 HBM
```

每个块一次 HBM 往返。总内存占用量从 O(N²) 降到 O(N)。反向传播在前向过程中重新计算某些值，而不是存储它们——又是内存节省。

**数值技巧。** 运行态 softmax 在各个块间维护 `(max, sum)`，使得最终归一化是精确的。不是近似——Flash Attention 计算出的输出与标准注意力逐比特完全相同（浮点非结合性除外）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速比 |
|------|------|---------|-------------------|
| Flash 1 | 2022 | 分块 SRAM 内核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5–2×（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5阶段流水线、软件 exp2 | 推理优先（初始仅前向） |

Flash 4 在发布时仅支持前向传播。训练仍然使用 Flash 3。Flash 4 的 GQA 和变长支持有待完善（2026年中）。

### 推测解码——另一个延迟制胜法宝

廉价模型提议 N 个令牌。大模型并行验证全部 N 个。如果验证接受了 k 个令牌，你只需为 k 次生成付出1次大模型前向传播。在代码和散文上，典型的 k=3-5。

2026年默认配置：
- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。2-3倍加速，零质量损失。
- **带草稿模型的推测解码。** 消费级硬件上2-4倍加速。
- **前瞻解码（Lookahead Decoding）。** 雅可比迭代；不需要草稿模型。小众但免费。

### 连续批处理

经典批量推理：等最慢的序列完成，然后启动新批次。短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 中发布，现在在 vLLM、TensorRT-LLM、SGLang 中）：当旧请求完成时立即将新请求换入批次。典型聊天工作负载上 5-10 倍吞吐量增益。

### PagedAttention——KV 缓存即虚拟内存

vLLM 的头条特性。KV 缓存以16令牌块为单位分配；页表将逻辑位置映射到物理块。让你可以在并行采样（束搜索、并行采样）之间共享 KV、为前缀缓存热替换前缀、以及整理碎片化内存。相比朴素的连续分配，吞吐量提高4倍。

## 动手构建

参见 `code/main.py`。我们实现：

1. 朴素 O(N²) 增量解码器。
2. O(N) KV 缓存解码器。
3. 模拟 Flash Attention 运行态最大化算法的分块 softmax。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)       # 逐令牌追加 K 向量
        self.V[layer][head].append(v)       # 逐令牌追加 V 向量

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在按层、按头的列表中持续增长每令牌的 K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-Attention风格的 softmax(qK^T)V，使用运行态 max/sum。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

一次 `softmax(qK) V` 的逐比特相同输出，但在任何时候工作集都是 `tile × d_head` 块，而非完整的 `N × d_head`。

### 步骤 3：对比朴素 vs 缓存解码在100令牌生成上的表现

统计注意力操作次数。朴素：O(N²) = 5050。缓存：O(N) = 100。代码将两者打印出来。

## 实战使用

```python
# HuggingFace transformers 在 decoder-only 的 generate() 中自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # 如果是 Hopper 架构则使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是2026年的一大亮点——相同的 system prompt、few-shot 示例或长上下文文档跨调用复用 KV 缓存。对于使用重复工具提示的 Agent 工作负载，前缀缓存通常能带来5倍吞吐量提升。

## 交付物

参见 `outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV 缓存策略、量化和推测解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同输出；注意操作计数的差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和多个补全，对 P 运行一次前向传播以填充 KV 缓存，然后按补全分支。测量相比每次重新编码 P 的加速比。
3. **困难。** 实现玩具 PagedAttention：以固定16令牌块为单位分配 KV 缓存并使用空闲列表。当序列完成时将其块归还给池。模拟1000个不同长度的聊天补全。比较与连续分配的内存碎片化情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| KV 缓存 | "让解码变快的技巧" | 存储来自每个前缀令牌的 K 和 V；新查询对缓存进行注意力而非重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上有80 GB，B200 上有192 GB。约 3 TB/s 带宽。 |
| SRAM | "片上内存" | 每个 SM 的快速内存，H100 上每个 SM 约256 KB。约 30 TB/s 带宽。 |
| Flash Attention | "分块注意力内核" | 计算注意力而不在 HBM 中物化 N×N 矩阵。 |
| 连续批处理 | "不等待的批处理" | 完成的序列换出，新的换入，无需排空批次。 |
| PagedAttention | "vLLM 的招牌" | KV 缓存以带页表的固定块分配；消除碎片化。 |
| 前缀缓存 | "复用长提示" | 跨请求缓存共享前缀的 KV；对 Agent 来说是重大成本削减。 |
| 推测解码 | "草稿 + 验证" | 廉价草稿模型提议令牌；大模型在一次前向中验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) —— Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) —— Flash 3。
- [FlashAttention-4 发布说明 (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) —— Blackwell 5阶段流水线和软件 exp2 技巧；阅读 repo README 了解本课提到的仅前向发布限制。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) —— vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 推测解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— 本课引用的集成草稿方法的 EAGLE-1/2 论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— 与 EAGLE 齐名的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) —— 对16令牌块和页表设计的规范深入解析。

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这篇教材用具体的内存数字回答了"为什么推理慢"这个工程核心问题。从 HBM/SRAM 的带宽比切入，把 KV Cache、Flash Attention、连续批处理、PagedAttention、推测解码等看似分散的技术串成了一个统一的内存优化故事。这对于从"训练思维"转向"推理思维"的学生来说是关键的转折点。

### 二、知识结构梳理

- **基础层**：O(N²)→O(N) 的 KV 缓存数学、HBM vs SRAM 带宽比（3TB/s vs 30TB/s）、分块 softmax 的运行态 max/sum 算法
- **模式层**：Flash Attention 1→4 的版本演化轨迹、推测解码的 draft+verify 模式、PagedAttention 的虚拟内存类比
- **应用层**：vLLM/TensorRT-LLM/SGLang 的生产栈差异、前缀缓存在 Agent 工作负载中的5倍增益、GQA/MLA 的缓存压缩效应

### 三、核心洞察

1. **推理瓶颈不在 FLOPs，在 HBM 带宽**：H100 的 HBM 带宽是 3TB/s，SRAM 是 30TB/s。每次 HBM 读写是 10× 的减速。传统注意力的三次 HBM 往返是病根。
2. **KV 缓存不是"缓存"，是"记忆"**：前缀令牌的 K/V 向量是确定性的——一旦算出来就不会变。不缓存 = 对确定性信息做重复计算，这是工程上的浪费。
3. **Flash Attention 的"精确性"是反直觉的**：很多人以为分块计算是近似。事实证明运行态 softmax 聚合和完整计算逐比特相同——这不是近似，是把 O(N²) 内存降到 O(N) 的数学重构。
4. **GQA 的缓存节省不是"优化"，是"必需"**：Llama 70B 在 128K 上下文中，MHA 的 KV 缓存需要 32GB。GQA 压到 4GB。8倍的差距从"nice to have"变成了"能跑还是不能跑"。
5. **连续批处理是吞吐量的杠杆**：等待最慢序列的传统批处理浪费了 GPU 中最快的序列完成后的大量空闲时间。即完成即换入的连续批处理把这种浪费消灭了。
6. **PagedAttention 是操作系统领域的分页机制在 GPU 上的应用**：固定块分配+页表映射，这不是新思想，但把它带到 KV 缓存上是真正的工程洞见。
7. **推测解码的"免费午餐"本质**：一个便宜模型生成多个候选，一个大模型一次性"批改"，接受的越多省得越多——本质上是用并行验证替换串行生成。

### 四、教学建议

1. **从内存带宽比开始讲**：如果不先讲清楚 HBM 和 SRAM 的 10× 带宽差距，学生永远不理解为什么"分块"是必要的。
2. **让学生手算自己模型的 KV 缓存大小**：给定层数、头数、d_head、上下文长度，算出一个 MB/GB 数字——亲眼看到自己的模型在 128K 上下文中需要多少显存。
3. **做"有 KV Cache vs 无 KV Cache"的时延测量**：哪怕在小模型上跑 128 令牌生成，记录每步时延——有缓存的是 O(1) per token，无缓存的是 O(N)，对比图会让抽象变成体感。
4. **用纸和笔演示分块 softmax**：写一个3块×4令牌的例子，手动走一遍运行态 max/sum 的更新——理解这个算法是理解 Flash Attention 的关键。
5. **推荐 vLLM 的 PagedAttention 博客**：那张"页表→物理块"的图比任何文字解释都更直观。
6. **对比 EAGLE 和 Medusa 的架构图**：一种是用独立草稿模型，一种是共享隐藏状态的 draft head——两者的 trade-off（质量 vs 部署便利性）是架构课的好素材。
7. **让练习2（前缀缓存）成为必须：** 让学生写一个版本参数不同但 system prompt 相同的多轮对话——体验前缀缓存带来的加速是实打实的。

### 五、值得补充的内容

1. **注意力内核的硬件适配细节**：不同 GPU 架构的 SRAM 大小约束如何影响 Flash Attention 的 tile size 选择。
2. **量化 KV 缓存（FP8/INT8 KV cache）**：vLLM 的 `--kv-cache-dtype fp8` 是2026年默认配置，可以展开讲精度损失与内存节省的 trade-off。
3. **Chunked Prefill**：SGLang 的另一个核心优化——将长提示的 prefill 分块，与正在进行的 decode 步骤交错执行。
4. **Distributed KV Cache**：多 GPU/多节点部署中的 KV 缓存分布策略，尤其是 tensor parallelism 和 pipeline parallelism 对 KV cache 的影响。
5. **MoE 模型中的推理挑战**：MoE 模型在推理时所有专家都要加载到显存——这个部署困境如何影响 Flash Attention 的优化空间。

### 六、一句话总结

2026年的推理优化本质上是"HBM 访问最小化运动"——KV Cache 消除冗余计算，Flash Attention 消除物化矩阵，PagedAttention 消除碎片化浪费——整个技术栈围绕着"别碰 HBM"这个核心命令组织。

---

# 🎓 Agent 架构课：推理优化——为什么你的 GPU 在摸鱼？

**副标题：从"算完扔"的训练思维到"少碰内存"的推理思维的架构革命**

---

先问一个问题——

你的 H100 理论峰值是 989 TFLOPs FP16。但当你用 `model.generate()` 生成一个回答的时候，GPU 利用率是 20%。剩下的 80% 时间 GPU 在干嘛？

……

**在等人。**

不是在等网络。不是在等下一个请求。是在等自己的 HBM 把数据搬到 SRAM。

这是一个很多人没意识到的残酷事实：训练时 GPU 利用率 60-80%，因为你在批量计算、流水线充分。推理时——尤其是单请求或小批量时——GPU 的大部分时间在"读内存 - 算一下 - 写内存 - 读更多内存"。你花了40万美元买的 H100，有80%的时间在摸鱼。

这就是本课的核心问题。让我拆开来给你看。

**问题一：O(N²) 的冗余计算。**

你生成第1个令牌时，算了 token_1 的 Q_1、K_1、V_1。生成第2个令牌时，你又重新算了 token_1 的 K_1、V_1——尽管它们自从第一次计算后就再也没变过。生成第100个令牌时，你已经把 token_1 到 token_99 的 K 和 V 各算了99遍。

路径A：朴素解码器。每步重新计算整个历史——O(N²) 操作。生成 4000 个令牌就是 800万次注意力操作。
路径B：KV 缓存。token_1 到 token_N 的 K/V 只算一次，存起来。新步骤只用新 token 的 Q 去查整个缓存——每步 O(N) 操作，总共 O(N²) 但隐藏常数差了几个数量级。

我在生产环境里选 B。没有犹豫过一秒。KV 缓存的本质是"确定性信息不重复计算"——所有工程师都该刻在骨头里的原则。

但 KV 缓存它本身要吃显存。让我给你一个具体的数字，这比任何抽象讨论都有力。

**Llama 3 70B，128K 上下文。**
- 80层 × 8个KV头（GQA）× 128 d_head × fp16(2字节) × 2（K和V）= 每token 4KB。
- × 128,000 tokens = **10.4 GB**。
- 仅 KV 缓存就 10 GB。模型权重 fp16 是 140 GB。你的 A100 有 80 GB。
- **你要么切4张卡做 tensor parallelism，要么你用低于128K的上下文。**

现在我给你两条更具体的路径：

**路径一：纯 MHA（64头）。** KV 缓存 = 32 GB。在 80GB A100 上剩下 48 GB 给模型权重——根本跑不了 70B 模型。
**路径二：GQA（8个 KV 头，每8个 Q 头共享1组KV）。** KV 缓存 = 4 GB。剩下 76 GB——刚好塞下 70B 模型权重+缓存。

GQA 不是"优化了一点点"。GQA 是把这个模型从"需要4张卡才能推理"变成了"1张卡勉强能跑"。在架构决策层面，这是生或死的区别。质量上的损失？大多数基准不到 0.05 困惑度。**这是一个用不到 1% 的质量换 8 倍内存节省的交易——我这辈子签过的最好的合同。**

**问题二：HBM 往返的灾难。**

标准注意力的"三步走"：
1. 算 S = Q @ K^T，把 N×N 矩阵写进 HBM
2. 从 HBM 读 S，算 softmax，把 N×N 矩阵写回 HBM
3. 从 HBM 读 P，算 O = P @ V，把 N×d 写回 HBM

三次 HBM 往返。每次 HBM 访问 ~3TB/s。每次 SRAM 访问 ~30TB/s。差距是 10 倍。

Flash Attention 干的一件事：**让所有计算在 SRAM 里完成，不把 N×N 矩阵交还给 HBM。**

它不是"优化"了注意力计算。它是重写了计算的物理位置。Q 切成小块装进 SRAM，K/V 切成小块装进 SRAM，在 SRAM 里做所有的事情——softmax 的运行态 max/sum 聚合保证了最终结果和全局计算逐比特相同。

**关于这个"逐比特相同"——我必须强调。** 很多人以为分块计算是一种牺牲精度的近似。不。运行态 softmax 在数学上和全局 softmax 等价——它只是换了一种求和顺序。Flash Attention 的输出和标准注意力在任何意义上都是相同的——除了浮点数非结合性带来的可以忽略的微小差异。

**反模式我必须点名批评。** 我见过团队在生产中关闭 Flash Attention 因为"它导致了一个数值误差"——但那个误差不是来自 Flash Attention，而是来自他们自己在训练时写的半精度累积顺序不同。他们花了三周时间 debug 一个不存在的 bug。**除非你有一个具体的 NaN 或 Inf 报告，不要怀疑 Flash Attention 的正确性。**

还有一个反模式：**批处理时的"等最慢序列"**。传统做法是：收集一批请求→同时推理→等最后一个序列生成完→全部返回→开始下一批。这意味着一个生成了1000个令牌的长回答和一个生成了50个令牌的短回答——GPU 在短回答完成后有950步在空转。连续批处理是解药：短回答完成后立刻将其移除，换入新请求。vLLM 的连续批处理在典型聊天工作负载上带来 5-10 倍吞吐量增益。**这不是细节优化，是你的基础设施的容量翻了一个数量级。**

**关于推测解码。** 核心思想不是"让模型变快"，而是"不要在确定性的令牌上浪费昂贵计算"。"The capital of France is..."后面大概率是"Paris"。用一个小模型生成候选，用大模型一次性验证。EAGLE-2 在代码和散文上平均接受3-5个令牌——这意味着你用1次大模型前向传播换了3-5个令牌的输出。**这是用廉价计算替换昂贵计算——工程的本能。**

**最后的清单，如果你在部署推理服务：**

- ✅ 你开了 KV 缓存吗？——不开的话，你的 O(N²) 生成成本是你本可以避免的。
- ✅ 你用了 Flash Attention 3+ 吗？——没开的话，40-80% 的 GPU 算力在等人。
- ✅ 你用了 GQA 或 MLA 吗？——纯 MHA 在128K上下文下不是"慢"，是"跑不了"。
- ✅ 你用了连续批处理吗？——没开的话，混长混短请求的吞吐量损失是你系统容量的 5-10 倍。
- ✅ 你用了前缀缓存吗？——如果你的 system prompt 超过 1K 令牌且有多请求共享，前缀缓存是免费的吞吐量翻倍。
- ✅ 你的监控有"GPU 全局内存带宽利用率"吗？——低于 40% 意味着你的内核在等 HBM。这是推理服务的血压计。

**金句：2026年的推理不是计算密集任务——它是内存搬运任务，偶尔穿插一点矩阵乘法。你的优化目标不是"多做计算"，而是"少碰 HBM"。**

回到开头的问题——为什么你的 GPU 在摸鱼？因为 80% 的时间它在等数据从 HBM 爬到 SRAM。Flash Attention、KV Cache、连续批处理——这些不是一个优化清单，这几个词共同组成了一句话：**把数据留在片内，把冗余计算扔掉，把空闲批次填满。** 做到了，你的 H100 才开始值它花了你40万的那个价钱。
