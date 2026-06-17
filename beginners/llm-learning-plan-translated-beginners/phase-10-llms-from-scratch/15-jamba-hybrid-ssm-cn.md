# Jamba：Transformer + SSM 混合架构

> **原文：** Jamba — Hybrid SSM-Transformer  
> **预计时间：** ~20 分钟  

---

## 🧠 核心洞察

Transformer 的注意力是 O(n²)——序列长时太贵。SSM（状态空间模型）是 O(n)——固定内存。

**但纯 SSM 质量不如 Transformer。**

Jamba 的答案：两个都用。每 7 个 Mamba（SSM）层配 1 个 Transformer 层。需要精确回忆时走注意力，需要效率时走 SSM。

---

## 📊 关键数字

| | Jamba | 同规模 Transformer |
|---|---|---|
| 模型大小 | 52B（12B 激活） | 52B |
| 上下文 | 256K | 32K-128K |
| KV Cache（256K） | ~3GB（仅 Transformer 层有） | ~60GB |
| GPU 需求 | 单张 80GB | 4 张 A100 |

---

## 🔬 Mamba-3（ICLR 2026）

三个改进：
1. **指数梯形离散化** — 更精确的连续→离散转换
2. **复值状态** — 状态用复数表示，保留相位信息
3. **MIMO 投影** — 多用通道投影捕获更多信息

---

## 🏆 焊死在脑子里的东西

1. **SSM = O(n) 速度，Transformer = O(n²) 质量。Jamba = 两者都用。**
2. **1:7 比例是经验甜点。** 每 8 层里 1 个 Transformer + 7 个 Mamba。
3. **256K 上下文在单 GPU 上。** 纯 Transformer 需要 4 张 GPU。

---

> **Transformer 记住一切但贵，SSM 快但会忘。Jamba 把两者塞进一个模型——关键的时刻用注意力，大多数时候走捷径。**
