# 指令微调（SFT）：教 AI "回答问题"而不是"续写"

> **原文：** Instruction Tuning (SFT)  
> **预计时间：** ~30 分钟  

---

## 🚗 开场翻车：问"法国首都是什么？"，模型回"德国首都是什么？"

你在 Phase 10-02 训练了一个 Mini GPT。对它说"The transformer architecture"，它能续写"has revolutionized NLP."——作为"下一个词预测器"，很厉害。

但问它"What is the capital of France?"——模型不会说"Paris"。它会续写成"What is the capital of Germany? What is the capital of Spain?"——因为它从训练数据里学到了"问题列表"的模式。

**基础模型只会续写。ChatGPT 会回答。差距在哪？** SFT（Supervised Fine-Tuning，监督微调）。同样的架构，同样的预训练，但多了 20,000-100,000 个（指令，回答）对，教会模型"看到问题→给出答案"。

---

## 🧠 SFT 做了什么

**不教新知识。** 模型在预训练时已经从维基百科学过了"巴黎是法国首都"。

**教的是行为。** 看到问题→回答。看到指令→完成。看到有害请求→拒绝。

```
预训练：给模型知识
SFT：给模型礼貌
```

## 📝 ChatML 格式

```json
{
  "system": "你是一个有用的助手。",
  "user": "法国的首都是什么？",
  "assistant": "法国的首都是巴黎。"
}
```

### Loss Masking——关键技巧

**训练时只计算 `assistant` 部分的 loss。** `system` 和 `user` 部分不参与 loss 计算。如果不这样做，模型也会学"生成用户的问题"——那就变成续写机器而不是回答机器了。

---

## 📊 关键事实

- Stanford Alpaca：只用 52,000 个 GPT-3.5 生成的样本微调 Llama 7B，成本 $600
- Llama 2 Chat：初始 SFT 只用了约 27,000 个高质量样本
- **质量 > 数量。** 熟练标注员写的 27K 样本 > 互联网抓的 100 万噪声样本

---

## 🏆 焊死在脑子里的东西

1. **基础模型 = 续写机器。SFT 后 = 回答机器。** 同一个架构，不同的数据。
2. **Loss Masking：只算 assistant 部分的 loss。** 不 mask 的话模型学"生成问题"。
3. **$600 就能把 Llama 7B 变成能聊天。** SFT 不是昂贵的。

---

> **预训练给了 AI 知识，SFT 给了 AI 教养。一个有知识没教养的 AI 只会续写——你问它问题，它给你问题列表。**
