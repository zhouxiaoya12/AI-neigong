# RAG（检索增强生成）

你的 LLM 知道训练截止日期之前的所有内容。但它不知道你公司的文档、你的代码库，也不知道上周的会议记录。RAG 通过检索相关文档并将其塞入提示词来解决这个问题。这是生产环境 AI 中部署最广泛的模式。如果这门课你只做一个东西，那就做一个 RAG 管线。

**类型**：构建  **语言**：Python  **前置要求**：阶段 10（从零开始 LLM）、阶段 11 第 01-05 课  **预计时间**：约 90 分钟  
**相关内容**：阶段 5 · 第 23 课（RAG 的六种分块策略及各自适用场景）、阶段 5 · 第 22 课（Embedding 模型深度解析，用于选择 Embedder）、阶段 11 · 第 07 课（高级 RAG：混合搜索、重排序与查询变换）

## 学习目标

- 构建完整的 RAG 管线：文档加载、分块、Embedding、向量存储、检索和生成
- 使用向量数据库（ChromaDB、FAISS 或 Pinecone）实现语义搜索，并建立正确的索引
- 解释为什么 RAG 比微调更适合知识密集型应用（成本、时效性、可溯源）
- 使用检索指标（精确率、召回率）和生成指标（忠实度、相关性）评估 RAG 质量

## 问题陈述

你为你的公司做了一个聊天机器人。一个客户问："企业版套餐的退款政策是什么？"LLM 回答了一个关于典型 SaaS 退款政策的通用答案。而实际的退款政策——藏在公司内部 200 页的维基文档里——明确写着：企业客户有 60 天的退款窗口，退款按比例计算。LLM 从没见过这份文档。它不可能知道未经训练的内容。

微调是一种解决方案。把 LLM 拿到你的内部文档上训练，然后部署更新后的模型。这个方法确实可行，但有严重的问题。微调一次要花几千美元的算力。文档一旦变动，模型就过时了。你无从得知模型依据了哪个源文档。而且，如果公司下个月收购了另一条产品线，你还得再微调一次。

RAG 是另一种方案。模型不用动。当用户提问时，去你的文档库里搜索相关段落，把它们粘贴到提示词里问题之前，让模型基于这些上下文来回答。文档库可以在几分钟内更新。你可以清楚地看到检索了哪些文档。模型本身从未改变。这就是 RAG 在生产环境中成为主导模式的原因：更便宜、更新鲜、更可审计，而且适用于任何 LLM。

## 核心概念

### RAG 模式

整个模式只有四个步骤：

```
查询 → 检索 → 增强提示词 → 生成
```

每个 RAG 系统都遵循这个模式。不同生产级 RAG 系统之间的差异在于每个步骤的细节：你如何分块、如何 Embedding、如何搜索、以及如何构建提示词。

### 为什么 RAG 比微调更好

| 关注点 | 微调 | RAG |
|---|---|---|
| 成本 | 每次训练 $1,000-$100,000+ | 每次查询 $0.01-$0.10（Embedding + LLM） |
| 时效性 | 不重新训练就会过时 | 重新索引文档即可更新，几分钟完成 |
| 可审计性 | 无法追踪答案来源 | 可以展示精确的检索段落 |
| 幻觉 | 仍然可能自由产生幻觉 | 基于检索到的文档，有所依据 |
| 数据隐私 | 训练数据嵌入权重之中 | 文档保留在你的向量存储中 |

微调永久性地改变了模型的权重。RAG 临时性地改变了模型的上下文。对于大多数应用来说，临时上下文正是你需要的。

微调胜过 RAG 的唯一情况：当你需要模型采用特定风格、语气或推理模式，而仅靠提示词无法达到时。对于事实型知识检索，RAG 每次都赢。

### Embedding 模型

一个 Embedding 模型将文本转换为稠密向量。相似的文本在这个高维空间中产生接近的向量。"如何重置密码？"和"我需要改密码"尽管几乎没有共同的词，却产生几乎相同的向量。"猫坐在垫子上"则产生完全不同的向量。

常见的 Embedding 模型（2026 年阵容——完整分析见阶段 5 · 第 22 课）：

| 模型 | 维度 | 提供商 | 备注 |
|---|---|---|---|
| text-embedding-3-small | 1536（Matryoshka） | OpenAI | 大多数场景下性价比最佳 |
| text-embedding-3-large | 3072（Matryoshka） | OpenAI | 更高精度，可截断到 256/512/1024 |
| Gemini Embedding 2 | 3072（Matryoshka） | Google | MTEB 检索最高分；8K 上下文 |
| voyage-4 | 1024/2048（Matryoshka） | Voyage AI | 提供领域变体（代码、金融、法律） |
| Cohere embed-v4 | 1024（Matryoshka） | Cohere | 强大的多语言能力，128K 上下文 |
| BGE-M3 | 1024（稠密 + 稀疏 + ColBERT） | BAAI（开源权重） | 一个模型三种视图 |
| Qwen3-Embedding | 4096（Matryoshka） | 阿里（开源权重） | 开源权重检索得分最高 |
| all-MiniLM-L6-v2 | 384 | 开源权重（Sentence Transformers） | 原型开发基准 |

本课我们使用 TF-IDF 构建自己的简单 Embedding。不是因为 TF-IDF 是生产系统用的东西，而是因为它让概念变得具体：文本进去，向量出来，相似文本产生相似向量。

### 向量相似度

给定两个向量，如何衡量它们的相似度？三种选择：

**余弦相似度**：两个向量之间夹角的余弦值。范围从 -1（完全相反）到 1（完全相同）。忽略大小，只关心方向。这是 RAG 的默认选择。

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

**点积**：原始的向量内积。更大的向量得到更高的分数。当向量大小携带信息时（更长的文档可能更相关）有优势。

```
dot(a, b) = sum(a_i * b_i)
```

**L2（欧几里得）距离**：向量空间中的直线距离。距离越小 = 越相似。对大小差异敏感。

```
L2(a, b) = sqrt(sum((a_i - b_i)^2))
```

余弦相似度是标准选择。它通过向量归一化，优雅地处理不同长度的文档。当人们说"向量搜索"时，几乎总是指余弦相似度。

### 分块策略

文档太长，不能作为单个向量进行 Embedding。一个 50 页的 PDF 可能产生极差的 Embedding，因为它包含几十个不同的主题。解决方案是将文档分割成块，分别对每个块进行 Embedding。

**固定大小分块**：每 N 个 token 切一块。简单且可预测。一个 512 token 的块加上 50 token 的重叠意味着：第 1 块是 token 0-511，第 2 块是 token 462-973，依此类推。重叠确保不会在不幸的边界处截断句子。

**语义分块**：在自然边界处切分。段落、章节或 Markdown 标题。每个块是一个连贯的意义单元。实现更复杂，但检索质量更好。

**递归分块**：先尝试在最大边界切分（章节标题）。如果某章节仍然太大，就在段落边界切分。如果某段落仍然太大，就在句子边界切分。这就是 LangChain 的 RecursiveCharacterTextSplitter 的方法，在实践中效果很好。

**块大小比你想象的更重要**：

- **太小（64-128 token）**：每个块缺乏上下文。"上季度增长了 15%"——不知道"它"指什么就没有意义。
- **太大（2048+ token）**：每个块涵盖多个主题，稀释了相关性。当你搜索收入数据时，结果可能得到一个 10% 关于收入、90% 关于人员编制的块。
- **甜点区间（256-512 token）**：足够的上下文来自成一格，同时又足够聚焦从而保持相关性。

大多数生产级 RAG 系统使用 256-512 token 的块，重叠 50 token。Anthropic 的 RAG 指南也推荐这个范围。

### 向量数据库

一旦有了 Embedding，你需要一个地方来存储和搜索它们。选项如下：

| 数据库 | 类型 | 最佳场景 |
|---|---|---|
| FAISS | 库（进程内） | 原型开发，小到中型数据集 |
| Chroma | 轻量级数据库 | 本地开发，小型部署 |
| Pinecone | 托管服务 | 无需运维的生产环境 |
| Weaviate | 开源数据库 | 自托管生产环境 |
| pgvector | Postgres 扩展 | 已经在用 Postgres |
| Qdrant | 开源数据库 | 高性能自托管 |

本课我们构建一个简单的内存向量存储。它在列表中存储向量，并使用暴力余弦相似度搜索。这等价于 FAISS 的 flat 索引。它可以扩容到大约 10 万个向量，超过这个数量就会变慢。生产系统使用近似最近邻（ANN）算法如 HNSW，在毫秒级搜索数百万向量。

### 完整管线

索引阶段：每个文档运行一次（或文档更新时运行）。查询阶段：每次用户请求时运行。在生产环境中，索引可能处理数百万文档，耗时数小时。查询必须在 1 秒内响应。

### 真实数字

大多数生产级 RAG 系统使用以下参数：

- k = 每次查询 5 到 10 个检索块
- 块大小 = 256 到 512 token，50 token 重叠
- 上下文预算：每次查询 2,500-5,000 token 的检索内容
- 总提示词：约 8,000-16,000 token（系统提示词 + 检索块 + 对话历史 + 用户查询）
- Embedding 维度：384-3072，取决于模型
- 索引吞吐量：使用 API Embedding，每秒 100-1,000 个文档
- 查询延迟：检索 50-200ms，生成 500-3000ms

---

## 动手构建

### 第 1 步：文档分块

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

### 第 2 步：TF-IDF Embedding

我们构建一个简单的 Embedding 函数。TF-IDF（词频-逆文档频率）不是一种神经网络的 Embedding，但它能够将文本转换为可以捕获词语重要性的向量。文档中的高频词得到更高的 TF。整个语料库中的稀有词得到更高的 IDF。二者的乘积产生一个向量，在这个向量中，重要且独特的词汇具有较高的值。

```python
import math
from collections import Counter

def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)

def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    return [count.get(word, 0) / total for word in vocab]

def compute_idf(documents, vocab):
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf

def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]
```

### 第 3 步：余弦相似度搜索

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

### 第 4 步：提示词构建

这就是 RAG 中"增强"发生的地方。将检索到的块格式化为提示词，要求 LLM 基于所提供的上下文进行回答。

```python
def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[来源 {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""请仅基于以下上下文回答问题。
如果上下文没有包含足够的信息，请说"我没有足够的信息来回答这个问题。"

上下文：
{context}

问题：{query}

回答："""
```

### 第 5 步：完整的 RAG 管线

```python
class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved
```

### 第 6 步：生成（模拟）

在生产环境中，这一步是调用 LLM API。本课中，我们通过从检索上下文中提取最相关的句子来模拟生成。

```python
def simple_generate(prompt, retrieved_chunks):
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "我没有足够的信息。"
```

## 实际使用

使用真实的 Embedding 模型和 LLM，代码几乎不需要改变：

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

或者用 Anthropic：

```python
import anthropic

client = anthropic.Anthropic()

def generate(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

管线保持不变。换一个 Embedding 函数。换一个生成函数。检索逻辑、分块、提示词构建——无论使用什么模型，这些都完全相同。

对于大规模向量存储，用合适的向量数据库替换暴力搜索：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

results = collection.query(
    query_texts=["退款政策是什么？"],
    n_results=5
)
```

Chroma 在内部处理了 Embedding（默认使用 all-MiniLM-L6-v2），并将向量存储在本地数据库中。相同的模式，不同的底层实现。

## 交付物

本课产出：

- `outputs/prompt-rag-architect.md` — 一个用于为特定用例设计 RAG 系统的提示词
- `outputs/skill-rag-pipeline.md` — 一个教导 Agent 如何构建和调试 RAG 管线的技能文件

## 练习

1. 将 TF-IDF Embedding 替换为简单的词袋方法（二值化：单词存在则 1，否则 0）。在样本文档上比较检索质量。TF-IDF 应该表现更好，因为它对稀有词赋予更高权重。

2. 尝试不同的块大小：在相同的文档集上尝试 50、100、200 和 500 个词。对每种大小，运行同样的 5 个查询，并计算 Top-3 结果中返回相关块的比例。找到检索质量达到峰值的甜点区间。

3. 为每个块添加元数据（源文档名称、块位置）。修改提示词模板以包含来源标注，使 LLM 能够引用其来源。

4. 实现一个简单的评估：给定 10 对问答，对每个问题运行 RAG 管线，测量检索到的块中包含答案的百分比。这相当于 top-k 检索召回率。

5. 构建一个对话感知的 RAG 管线：维护最近 3 轮对话的历史记录，并与检索到的块一起包含在提示词中。用诸如先询问定价后再问"企业版呢？"的后续问题进行测试。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| RAG | "能读你文档的 AI" | 检索相关文档，粘贴到提示词中，基于这些文档生成答案 |
| Embedding | "把文本变成数字" | 文本的稠密向量表示，相似的含义产生相似的向量 |
| 向量数据库 | "AI 的搜索引擎" | 针对存储向量和通过相似度查找最近邻优化的数据存储 |
| 分块 | "把文档切成小块" | 将文档分割成较小的片段（通常 256-512 token），使每块可以独立进行 Embedding 和检索 |
| 余弦相似度 | "两个向量有多相似" | 两个向量之间夹角的余弦值；1 = 方向完全相同，0 = 正交，-1 = 完全相反 |
| Top-k 检索 | "拿到 k 个最佳匹配" | 从向量存储中返回与查询最相似的 k 个块 |
| 上下文窗口 | "LLM 能看多少文本" | LLM 在单次请求中可以处理的最大 token 数；检索块必须适配这个限制 |
| 增强生成 | "基于给定上下文回答" | 使用检索到的文档作为上下文来生成回答，而非仅依赖训练知识 |
| TF-IDF | "词语重要性评分" | 词频乘以逆文档频率；按词汇在语料库中的独特性进行加权 |
| 索引 | "准备文档以供搜索" | 对文档进行分块、Embedding 和存储的离线过程，使其可以在查询时被搜索 |

## 拓展阅读

- Lewis 等人，"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"（2020）—— Facebook AI Research 的原始 RAG 论文，正式确立了"先检索再生成"的模式
- Anthropic 的 RAG 文档（docs.anthropic.com）—— 关于块大小、提示词构建和评估的实用指南
- Pinecone 学习中心，"什么是 RAG？"—— 带有清晰可视化解释的 RAG 管线说明，附带生产环境考量
- Sentence-BERT：Reimers & Gurevych（2019）—— all-MiniLM Embedding 模型背后的论文，展示了如何训练双编码器用于语义相似度
- Karpukhin 等人，"Dense Passage Retrieval for Open-Domain Question Answering"（EMNLP 2020）—— DPR 论文，证明了稠密双编码器检索在开放域问答上胜过 BM25，并奠定了现代 RAG 检索器的模式
- LlamaIndex 高层概念 —— 构建 RAG 管线时需要了解的主要概念：数据加载器、节点解析器、索引、检索器、响应合成器
- LangChain RAG 教程 —— 另一种风格的编排框架；以可运行链的视角展示相同的"先检索再生成"模式

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是一份面向有一定 LLM 基础的工程师的 RAG 实战课。最大的亮点是"先讲为什么、再讲怎么做"的结构——用企业退款政策的案例作为钩子，在讲任何一行代码之前，先把 RAG vs 微调的哲学差异讲透了（"永久改变权重 vs 临时改变上下文"）。代码部分从 TF-IDF 手动实现开始，刻意避开现成库的便利，让学生真正理解"文本 → 向量 → 搜索"的每一步在发生什么，最后再平滑过渡到 OpenAI/Anthropic API。这是一份"教人钓鱼"而非"给人鱼吃"的教材。

### 二、知识结构梳理

**认知基础层**：RAG 的四步管线模式、与微调的本质差异（成本/时效/可审计三个维度）、Embedding 的语义直觉（相似含义 → 相似向量）

**工程模式层**：分块策略的三级分类（固定/语义/递归）与块大小甜点区间、余弦相似度作为向量检索的工业标准、提示词构建的三要素（外源上下文声明 + 仅基于上下文回答 + 来源标注）

**实践应用层**：从内存暴力搜索 → FAISS flat 索引 → ChromaDB SSD 索引 → Pinecone 托管 HNSW 的多级存储方案、用真实 API 替换模拟实现时的接口抽象设计、RAG 评估的双维度框架（检索召回率 + 生成忠实度）

### 三、核心洞察（备课时的关键理解）

1. **RAG 不是让模型更强，是让模型更"诚实"**。提示词中"请仅基于以下上下文回答"这句话是整个管线的灵魂——它不是技术优化，是行为约束。没有这句话，RAG 就退化成了"给模型多看几段文字"。

2. **余弦相似度成为标准不是因为它"最好"，而是因为它"刚好"**。它消除了文档长度的干扰但保留了方向的语义信息——恰好满足文本检索的核心需求。

3. **块大小是 RAG 系统最被低估的超参数**。太多人把精力花在选 Embedding 模型上，但实际上 256 vs 512 的块大小差异对检索质量的影响可能超过换 Embedding 模型。

4. **重叠机制本质上是一个"抗切割"保障**。不加重叠，检索系统的上限就被分块边界的偶然位置锁死了——一个关键信息恰好跨边界时，无论如何都检索不到。

5. **索引是离线、查询是在线——这两个阶段的性能约束完全不同**。索引看重吞吐量（每秒几千文档），查询看重延迟（毫秒级）。架构设计时必须分开考虑。

6. **TF-IDF 在这堂课里的作用不是"生产级方案"，而是"理解为什么 Embedding 是必要的"**。用它跑一遍，学生天然会感觉到"改密码"和"重置密码"匹配不上——然后你掏出一个语义 Embedding 模型，配对成功的那一刻，学生会永远记住 Embedding 解决了什么问题。

7. **微调仍然有它的位置——但不是知识注入的位置**。微调适合改变模型的行为模式（风格、语气、推理路径），RAG 适合注入外部知识。这两个不是竞争关系，是互补关系。糊涂的架构师会二选一，清醒的架构师知道什么场景用哪个。

### 四、教学建议

1. **开场用故事，不要用 PPT**。不给任何技术术语，先用"企业退款政策"的场景让学生自己感觉到痛点——"如果我是这个聊天机器人的开发者，我怎么办？"让答案自然浮现，学生会有更强的拥有感。

2. **先讲"四步法"，再用代码填充每一格**。在黑板上画一个 1→2→3→4 的流水线箭头图，每讲一步就在箭头下面填代码。学生看到的不是一堆零散的函数，而是一个逐渐被填满的蓝图。

3. **TF-IDF 阶段一定要让学生亲手改参数**。让一个学生把块大小调到 10，另一个调到 10000，当场对比两个极端下的检索结果——"太小"和"太大"的后果不需要讲，学生自己就看见了。

4. **余弦相似度的推导不要跳步**。从向量的角度出发，画出两个向量在二维空间的夹角，然后再点积公式。学生大脑里需要有一张几何图，而不是一行公式。

5. **真实 API 替换时，先让学生猜会发生什么**。用 TF-IDF 搜一遍"改密码"，学生看到结果很差。然后问："如果我用一个语义模型的 Embedding，你觉得结果会有什么不同？为什么？"让学生先推理，再验证，学习效果好十倍。

6. **把"仅基于以下上下文"拿掉，让学生看看后果**。RAG 提示词里把这句关掉，问 LLM 同一个问题，让学生对比两次回答——一次引用了外部文档却不知道来自哪里，一次清楚地标注了来源。这个 A/B 实验比任何讲解都更有说服力。

7. **布置作业时要分层**。基础题保障每个人都能跑通管线（练习 1、2），进阶题让学生理解评估（练习 4），挑战题推动有潜力的学生向前探索真实场景（练习 5 + 真实 API）。

### 五、值得补充的内容

1. **混合检索（Hybrid Search）的动机**。TF-IDF 课里讲了关键词匹配的局限，但没有讲何时关键词匹配仍然有用——比如精确的产品编号搜索。在生产环境中，语义搜索 + BM25 关键词搜索的混合方案是标配，值得提一嘴作为下节课的预告。

2. **重排序（Re-ranking）的必要性**。Top-k 检索从海量文档中初筛候选块，再用一个更精准的重排序模型重新打分——两阶段检索是生产级 RAG 的标准架构。本课只讲了第一阶段，第二阶段至少应提及存在。

3. **多语言分块的差异**。中文不像英文有天然的空格分词，固定字数分块在中文上表现更差。如果学生中有做中文 RAG 的，应该补充说明 jieba 分词 + 按 token 分块的做法。

4. **Embedding 模型的成本对比不仅仅是看价格**。dimension × query_volume × storage_cost 三者共同决定真实成本。一个 4096 维的免费开源模型可能因为存储和检索成本而比 1536 维的付费 API 模型更贵。

5. **评估不仅看数字，还要看"失败模式"**。用学生自己能懂的语言分类错误：幻觉型（编造了上下文里没有的信息）、遗漏型（上下文里有但没提取出来）、混淆型（把来源 A 的信息归到了来源 B）。让学生自己给管线的输出贴标签，比跑一个评价脚本学到的多得多。

### 六、一句话总结

**RAG 不是在打造一个什么都懂的 AI——是在打造一个知道自己不懂、但知道去哪查的 AI。**

---


---

# 🎓 Agent 架构课：RAG——给 AI Agent 装上外部记忆

同学们好。我是你们的 Agent 架构老师。今天这节课，我们不讨论怎么让模型更强、怎么调参、怎么让精度再高两个百分点。那些是算法工程师做的事。

我们今天的问题是：**你怎么让一个 Agent 知道它不知道的事情？**

## 问题的本质

想象你是一个 AI Agent。你的"大脑"——也就是大语言模型——在训练结束的那一刻就被冻结了。你最后一次见到新知识是在你的训练截止日期。从那天之后，世界上发生的所有事情，你公司内部写的所有文档，你的用户上周发给你的所有邮件——对你来说，都不存在。

现在一个用户走过来问你："我们公司企业版套餐的退款政策是什么？"

如果你是传统的 LLM，你的回答可能是："通常 SaaS 公司的企业退款政策包括 30 天窗口期……"——这是一句正确的废话。真实的答案是"60 天窗口期，按比例退款"，但它藏在你公司内部一个 200 页的维基文档里。你从没见过它。

**这个问题的本质是什么？** 本质是：Agent 的知识和真实世界之间，存在一个时间差。Agent 的知识在训练时写入，真实世界在持续变化。这个 Gap 怎么填？

## 两种思路，两种哲学

面对这个 Gap，业界给出了两条路。

**第一条路叫微调。** 既然模型的知识是训练出来的，那我就拿新数据重新训练。把公司文档、最新论文、上周的会议纪要全部喂给模型，让它把这些知识"长"到权重里去。

这条路有什么问题？三个。

第一，**成本**。一次微调少则几千美元，多则几十万美元。你公司每加一个新产品线，每更新一次政策文档，都要重新来一遍。

第二，**时效性**。你今天微调完，明天产品经理改了退款政策，你的模型又过时了。你不是 Agent，你是考古学家。

第三，也是我最关心的一点：**你完全不知道答案是从哪来的。** 模型的权重是一个黑箱。当它回答"60 天退款"的时候，你无法确认它是从你的文档里学来的，还是它自己在网上看到的、可能是错误的、关于另一家公司的信息。没有溯源，就没有信任。

**第二条路叫 RAG。** 这不是一个新概念。2019-2020 年，Facebook AI Research、Google Research 和华盛顿大学各自独立地摸索出了相似的模式：不改变模型，而是在回答问题时，去外部知识库里"翻"出相关内容，临时提示给模型。

这有什么好处？

- **成本**：每次查询几分钱，不是几千美元。
- **时效性**：文档库更新了，下一轮查询就能用到——延迟是分钟级的。
- **可审计**：你可以精确地告诉用户："我的回答基于你公司内部维基的第 42 页第 3 段。"
- **隐私**：文档留在你自己的向量库里，没有嵌入到模型的权重中。

**微调是改变 Agent 的大脑。RAG 是给 Agent 一本参考书。** 对于知识密集型任务，你要的是参考书，不是洗脑。

## RAG 的四个步骤：让 Agent 学会"查资料"

任何 RAG 系统都只有四个步骤：

```
检索（Retrieve） → 增强（Augment） → 生成（Generate）
```

更细一点：

1. **用户提问**："我们公司的退款政策是什么？"
2. **检索**：Agent 去知识库里找和这个问题最相似的文档片段
3. **增强**：Agent 把这些片段粘贴到提示词里，作为"已知上下文"
4. **生成**：Agent 基于这些上下文来回答

这像什么？人类考试时翻书找答案。你不会把所有书都背下来——你在考试时翻到相关章节，读一遍，然后用自己的话回答。RAG 就是这个过程。

## 怎么"找"到对的文档？Embedding 和向量搜索

现在问题来了：Agent 怎么"找到"最相关的文档片段？

你不能用关键词匹配。"退款政策"和"退钱规则"在关键词层面完全不重叠，但它们说的是同一件事。你需要的是**语义搜索**——理解意思，而不只是匹配字面。

这就是 **Embedding** 的用武之地。

**Embedding 就是把文本变成一串数字（向量），使含义相似的文本在数字上也相似。**

举例：

- "如何重置密码？" → `[0.23, -0.45, 0.78, ...]`
- "我需要改密码" → `[0.21, -0.43, 0.80, ...]`
- "猫坐在垫子上" → `[-0.12, 0.67, -0.34, ...]`

前两个向量非常接近，第三个离得很远。尽管前两句几乎没有共同词汇，但它们在语义空间中挤在一起。

有了 Embedding，搜索就变成了数学题：给定查询向量，在文档向量库中找离它最近的 k 个。用什么度量？**余弦相似度**——看两个向量指向方向有多一致。这是 RAG 领域的工业标准。

## 怎么"切"文档？分块的艺术

你不能把一整本百科全书作为一个向量 Embedding。50 页的 PDF 包含几十个不同主题，Embedding 出来是一个无法代表任何具体概念的模糊向量。

你得把它切碎——这就是**分块（Chunking）**。

三种做法：

- **简单粗暴法**：每 500 个词切一块。简单，可预测，但可能把一句话切成两半。
- **聪明一点**：在自然边界（段落、标题）切分。每个块都是一个完整的"意思单元"。
- **最实用**：递归分块。先按章节标题切，太大就按段落切，还是太大就按句子切。LangChain 默认做法。

**块多大合适？**

- 太小（64 词）：缺乏上下文。"上季度涨了 15%"，你不知道是什么涨了。
- 太大（2000 词）：涵盖多个话题，"退款政策"只占 10%，剩下全是无关内容。
- **生产环境最佳实践：256-512 token，50 token 重叠。** 记住这个数字。

**重叠为什么重要？** 如果不加重叠，"企业客户的退款政策是 60 天窗口"可能被切成两半。两块都无法完整回答查询。重叠防止了这种边界分裂。

## 提示词怎么写？给 Agent 正确的指示

检索到了文档，接下来就是把它们拼进提示词。这部分看似简单，写不好会让整个管线前功尽弃。

你的提示词必须做到三件事：

1. **明确告知上下文是外部提供的** → "以下是你需要用来回答的上下文："
2. **强制约束回答范围** → "请仅基于以上上下文回答。如果上下文中没有足够信息，就说不知道。"
3. **标注来源** → "在回答中引用 [来源 1]、[来源 2]"

第三点至关重要——**可审计性是 RAG 的核心价值之一**。你能指着答案说"这段话来自这份文档"，这是微调永远做不到的。

## 向量数据库：Agent 的记忆仓库

小规模实验可以暴力遍历（~10 万条）。真实场景呢？上千万份文档。你需要**向量数据库**。

| 名字 | 一句话定位 |
|---|---|
| **FAISS** | Meta 开源，进程内运行，研究和中小规模首选 |
| **ChromaDB** | Python-native 轻量级向量数据库，开发体验极好 |
| **Pinecone** | 全托管云服务，不想管运维就选它 |
| **pgvector** | Postgres 扩展，团队已在用 Postgres 就是零迁移成本 |
| **Qdrant / Weaviate** | 开源高性能自托管，生产环境可靠选择 |

核心算法：**ANN（近似最近邻）**。不需要精确找最近——"差不多最近"就够了，但必须毫秒级。HNSW（分层可导航小世界图）是目前主流。

## 把一切串起来：Agent 的 RAG 工作流

从 Agent 角度看，RAG 分两个阶段：

**离线阶段（索引）**——运行一次，或文档更新时触发：
```
文档 → 分块 → Embedding → 存入向量数据库
```

**在线阶段（查询）**——每次用户交互触发：
```
用户查询 → Embedding → 向量搜索 → 构建增强提示词 → LLM 生成 → 返回答案
```

离线处理数百万文档、耗时数小时。在线必须一秒内完成。两个阶段的性能指标完全不同：离线看重吞吐量，在线看重延迟。

## 生产环境真实数字

- **Top-k**：每次检索 5-10 个块。少于 5 可能缺信息，多于 10 可能引入噪声
- **块大小**：256-512 token，50 token 重叠
- **上下文预算**：检索内容约 2,500-5,000 token，总提示词约 8,000-16,000 token
- **查询延迟**：检索 50-200ms，生成 500-3000ms，端到端目标 < 3 秒
- **索引吞吐**：API Embedding 每秒 100-1000 个文档

## 什么时候不该用 RAG？

- 问题完全在模型的训练知识范围内（如"Python 列表推导式怎么写"）
- 你需要的是风格、语气、推理模式的改变——微调更合适
- 延迟要求极端严格（< 100ms），检索步骤会成为瓶颈

## 总结：Agent 架构师的 RAG 检查清单

1. ✅ **知识源识别**：Agent 需要访问哪些文档？
2. ✅ **分块策略**：块多大？什么重叠？什么切分方式？
3. ✅ **Embedding 模型**：用谁的？多语言需要吗？维度多少？
4. ✅ **向量存储**：内存、ChromaDB、Pinecone 还是 pgvector？
5. ✅ **检索配置**：Top-k 多少？要不要重排序？
6. ✅ **提示词设计**：是否强制"仅限于上下文"？是否有来源标注？
7. ✅ **评估体系**：检索召回率多少？生成忠实度如何？
8. ✅ **更新机制**：文档变更后多久能反映到检索结果中？

记住一件事：**RAG 不是让 Agent 变得更聪明，而是让它知道自己不知道什么，然后给它一个去查的方法。** 谦逊，加上工具，这就是好的 Agent 设计。

> 🧪 本章代码：[../phase-11-llm-engineering/code/rag/main.py](../phase-11-llm-engineering/code/rag/main.py) — 完整 RAG 流水线：分块→TF-IDF→余弦检索→增强Prompt，中英文混合分词。
