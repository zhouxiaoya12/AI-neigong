# 推理优化：让 GPU 不偷懒

> **原文：** Inference Optimization  
> **预计时间：** ~30 分钟  

---

## 🚗 开场翻车：1 个用户飞快，100 个用户全卡

4 块 A100 跑 Llama 3 70B。1 个用户：每秒 50 token——飞快。

100 个用户同时用：每秒 3 token——比人打字还慢。GPU 账单 $25,000/月。

**模型没变，权重没变，架构没变。变的是调度。** 朴素推理浪费 GPU 90% 以上算力。

---

## 🧠 LLM 推理的两个阶段

### Prefill（计算密集）

一次性处理整个提示。所有 token 已知 → 注意力并行计算 → 大矩阵乘法占满 GPU。瓶颈：FLOPS。

### Decode（内存密集）

一个一个吐 token。每个 token 需要读一次整个模型权重（140GB）→ GPU 算完就在等下一次读取。瓶颈：**显存带宽。**

```
Prefill: "GPU 忙不过来"   → 瓶颈在算力
Decode:  "GPU 在等内存"   → 瓶颈在带宽
```

---

## 🔧 五大优化

| 技术 | 做了什么 | 效果 |
|------|---------|------|
| **KV Cache** | 缓存已算过的 Key/Value，不重复算 | 解码加速 ~10x |
| **Continuous Batching** | 不等整批完就加新请求 | 吞吐翻倍 |
| **PagedAttention** | KV Cache 按页管理（像 OS 虚拟内存） | 显存利用率大幅提升 |
| **Flash Attention** | 避免把注意力矩阵写回 HBM | 快 2-4 倍 |
| **Speculative Decoding** | 用小模型"猜"token，大模型验证 | 延迟砍半 |

---

## 📊 同一硬件，4 倍吞吐

```
vLLM + Continuous Batching + PagedAttention:
  100 并发 → 15-25 token/秒/用户

朴素推理:
  100 并发 → 5 token/秒/用户

同样的 GPU，同样的模型。差距在调度。
```

---

## 🏆 焊死在脑子里的东西

1. **Prefill = 计算密集，Decode = 内存密集。** 瓶颈不一样，优化方向不一样。
2. **KV Cache = 不重复算历史 token。** 最基础也最重要的推理优化。
3. **vLLM 生产标配。** Continuous Batching + PagedAttention = 4 倍吞吐。

---

> **$25,000/月 和 $5,000/月的区别不在模型——在你怎么让 GPU 不闲着。推理优化不是锦上添花，是生死线。**
