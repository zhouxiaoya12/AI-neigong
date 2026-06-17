"""
RAG 完整流水线 · 从零实现（纯标准库）

RAG（检索增强生成）是生产环境部署最多的 AI 模式。
不理解它就无法理解为什么 LLM 需要外部知识，以及怎么做。

完整流水线：
  文档 → 分块(Chunking) → 向量化(TF-IDF) → 存储 → 检索(余弦相似度) → 生成

注意：这里用 TF-IDF 替代了真实环境中的嵌入模型（如 text-embedding-3-small）。
流水线的结构和真实 RAG 完全一致——把 TF-IDF 换成 embedding API 就是生产代码。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分词工具（中英文通用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def tokenize(text: str) -> list[str]:
    """中英文混合分词。

    英文：按空格和标点拆分单词（"hello world" → ["hello", "world"]）
    中文：按字符级 2-gram 拆分（"退款政策" → ["退款", "款政", "政策"]）

    为什么用 2-gram？
      单个汉字信息量太低，"的"、"了"、"是"在每篇文章里都出现，
      无法区分文档。2-gram 能捕捉到有意义的词组。
    """
    tokens: list[str] = []

    # 用正则把中文和非中文分开处理
    # 中文连续块 → 2-gram
    # 非中文块 → 按空格分词
    segments = re.split(r'([\u4e00-\u9fff]+)', text)

    for seg in segments:
        if not seg.strip():
            continue
        if re.match(r'[\u4e00-\u9fff]+', seg):
            # 中文：生成 2-gram（字符级双字词）
            seg_clean = seg.strip()
            for i in range(len(seg_clean) - 1):
                tokens.append(seg_clean[i:i+2])
        else:
            # 英文/数字：按空格和标点分词
            for tok in seg.strip().lower().split():
                tok = tok.strip('.,;:!?()[]{}，。；：！？（）【】"\'')
                if tok:
                    tokens.append(tok)

    return tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 步骤 1：文档分块（Chunking）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """将长文档切割成有重叠的小块。

    为什么要有重叠？
      —— 边界处的关键信息可能被一刀切断。
        比如 "退款政策适用于企业版用户" 这句卡在分块边界上，
        没有重叠的话前半句在块A、后半句在块B，检索时两块都找不到完整答案。

    参数：
      chunk_size: 每块包含的 Token 数
      overlap: 相邻两块之间重叠的 Token 数
    """
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # 往后滑动了 (chunk_size - overlap) 个词
    return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 步骤 2：构建词表 + TF-IDF 向量化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_vocabulary(documents: list[str]) -> list[str]:
    """从所有文档块中提取词汇表"""
    vocab = set()
    for doc in documents:
        vocab.update(tokenize(doc))
    return sorted(vocab)


def compute_tf(text: str, vocab: list[str]) -> list[float]:
    """计算词频（Term Frequency）。

    TF = 某个词在当前文档中出现的次数 / 文档总词数

    高频词在文档内权重高，但不一定是重要的词（如"的"、"了"）。
    所以需要 IDF 来平衡。
    """
    tokens = tokenize(text)
    count = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return [0.0] * len(vocab)
    return [count.get(word, 0) / total for word in vocab]


def compute_idf(documents: list[str], vocab: list[str]) -> list[float]:
    """计算逆文档频率（Inverse Document Frequency）。

    IDF = log((总文档数 + 1) / (包含该词的文档数 + 1)) + 1

    出现在很多文档中的词 → IDF 低（如"这个"、"可以"）→ 不重要
    只出现在少数文档中的词 → IDF 高（如"退款政策"）→ 区分度高
    """
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in tokenize(doc))
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf


def tfidf_embed(text: str, vocab: list[str], idf: list[float]) -> list[float]:
    """生成 TF-IDF 向量。

    TF-IDF = TF × IDF
    → 高频且不常见的词得分最高 = 最能代表文档主题的词
    """
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 步骤 3：余弦相似度检索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    余弦相似度 = (A·B) / (|A| × |B|)

    值域 [-1, 1]，越接近 1 越相似。
    在文本检索中，由于 TF-IDF 向量非负，实际值域是 [0, 1]。
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def search(query_embedding: list[float],
           stored_embeddings: list[list[float]],
           top_k: int = 5) -> list[tuple[int, float]]:
    """在已存储的文档块中检索与查询最相似的 Top-K 块。

    返回：(块索引, 相似度分数) 的列表，按相似度降序排列。
    """
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 步骤 4：增强 Prompt 构建
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_rag_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """构造 RAG 提示词。

    关键设计：明确告诉 LLM "只根据下面的上下文回答"。
    这个指令是 RAG 质量的底线 —— 没有它，LLM 可能会用内部知识而非检索结果。
    """
    context = "\n\n---\n\n".join(
        f"[来源 {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return (
        '请仅根据以下上下文来回答问题。'
        '如果上下文没有提供足够信息，请说「我无法根据现有信息回答」。\n\n'
        f'上下文：\n{context}\n\n'
        f'问题：{query}\n\n'
        f'回答：'
    )


def simple_generate(prompt: str, retrieved_chunks: list[str]) -> str:
    """一个简单的关键词匹配"生成器"。

    在真实 RAG 中，这一步是调用 LLM API。
    这里用关键词重叠度来模拟，让你理解"生成"和"检索"的分界线在哪里。
    """
    # 提取问题中的关键词
    query_section = prompt.lower().split("问题：")[-1]
    query_tokens = set(tokenize(query_section))

    # 去除停用词
    stop_tokens = {"的", "了", "是", "在", "和", "有", "我", "他", "她", "它",
                  "这", "那", "什么", "怎么", "为什么", "可以", "吗", "呢",
                  "啊", "吧", "不", "也", "都", "就", "要", "会", "对", "与",
                  "为", "以", "及", "或", "但", "而", "被", "从", "到", "把",
                  "向", "往", "着", "过", "上", "下", "中", "里", "外"}
    query_tokens = query_tokens - stop_tokens

    best_sentence = ""
    best_score = 0

    for chunk in retrieved_chunks:
        for sentence in chunk.split("。"):
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            words = set(tokenize(sentence))
            overlap = len(query_tokens & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

    return best_sentence if best_sentence else "我无法根据现有信息回答。"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 步骤 5：完整 RAG 流水线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RAGPipeline:
    """完整的 RAG 流水线。

    使用方法：
      rag = RAGPipeline(chunk_size=200, overlap=50, top_k=5)
      rag.index(documents, source_names)    # 索引文档
      result = rag.query("你的问题")         # 查询
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50, top_k: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self.vocab: list[str] = []
        self.idf: list[float] = []
        self.sources: list[str] = []

    def index(self, documents: list[str], source_names: list[str] | None = None) -> int:
        """索引一批文档。

        相当于"建库"步骤：
          分块 → 建词表 → 计算 IDF → 对所有块计算 TF-IDF 向量
        """
        # 分块
        all_chunks: list[str] = []
        all_sources: list[str] = []
        for i, doc in enumerate(documents):
            doc_chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(doc_chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(doc_chunks))

        self.chunks = all_chunks
        self.sources = all_sources

        # 向量化
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

        return len(all_chunks)

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """执行一次 RAG 查询。

        流程：向量化问题 → 检索 Top-K 相关块 → 构造 Prompt → "生成"答案
        """
        k = top_k or self.top_k

        # 向量化问题
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, k)

        # 整理检索结果
        retrieved = []
        for idx, score in results:
            retrieved.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "score": score,
                "index": idx,
            })

        # 构造 Prompt + 生成答案
        chunk_texts = [r["chunk"] for r in retrieved]
        prompt = build_rag_prompt(question, chunk_texts)
        answer = simple_generate(prompt, chunk_texts)

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "retrieved": retrieved,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示数据：中文电商公司知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_DOCUMENTS = [
    """
    退款政策。标准版客户购买后30天内可以申请全额退款。企业版客户享受60天延长的退款窗口，
    退款金额从取消之日起按比例计算。退款将在5到7个工作日内处理并返回原支付方式。
    退款窗口关闭后将不再接受退款申请。客户必须通过客服平台或直接联系客户经理提交退款申请。
    年度订阅中途取消的，剩余月份将按比例退还积分。
    """,

    """
    产品概览。极光科技提供三个产品级别：新手版、专业版、企业版。
    新手版包含个人用户的基本功能，每月29元。
    专业版增加团队协作、高级分析和优先客服，每月每用户99元。
    企业版包含专业版全部功能再加自定义集成、专属客户经理、单点登录、
    审计日志和99.99%运行时间服务等级协议。企业版采用定制定价，
    50人以内起价为每月500元。所有套餐都包含14天免费试用，不需要绑定信用卡。
    """,

    """
    安全实践。极光科技持有SOC 2 Type II合规认证，每年接受第三方安全审计。
    所有数据使用AES-256算法在存储时加密，使用TLS 1.3在传输时加密。
    客户数据存储在独立的租户中，位于AWS北京区域。企业版客户可以按组织配置
    数据驻留位置。每6小时执行一次备份，保留期30天。极光科技不会向第三方出售
    或共享客户数据。企业版客户可以在24小时内请求删除数据。
    安全漏洞奖励计划通过HackerOne平台运行。
    """,

    """
    API文档。极光科技API使用REST协议，请求和响应体均为JSON格式。
    认证方式为通过OAuth 2.0签发的Bearer令牌。速率限制为：新手版每分钟100次请求，
    专业版每分钟1000次，企业版每分钟10000次。每次响应都包含速率限制头：
    X-RateLimit-Limit、X-RateLimit-Remaining和X-RateLimit-Reset。
    超出速率限制将返回HTTP 429状态码及Retry-After头。
    API支持基于游标的分页，使用next_cursor字段。专业版和企业版支持Webhook
    实时事件通知。API版本控制使用URL路径中的日期版本标识。
    """,

    """
    运行时间与可靠性。极光科技为专业版保证99.9%的正常运行时间，
    为企业版保证99.99%的正常运行时间。正常运行时间按月计算，
    不包括提前72小时通知的计划维护窗口。如果正常运行时间低于保证水平，
    客户将获得服务积分：每低于SLA阈值0.1%获得10%积分，上限为月费的30%。
    服务积分必须在事件发生后30天内申请。状态页面在status.jiguang.com
    上更新，检测到事件后5分钟内发布。任何超过15分钟的中断都会在48小时内
    发布事后分析报告。
    """
]


def main() -> None:
    print("=" * 65)
    print("📚 RAG 完整流水线 · 从文档到答案")
    print("=" * 65)

    # ━━ 步骤 1：文档分块 ━━
    print("\n📋 步骤 1：文档分块")
    sample = SAMPLE_DOCUMENTS[0]
    chunks = chunk_text(sample, chunk_size=30, overlap=10)
    print(f"  原文档长度：{len(sample.split())} 词")
    print(f"  分块策略：每块30词，重叠10词")
    print(f"  分块数：{len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  块 {i} ({len(chunk.split())}词): {chunk[:60]}...")

    # ━━ 步骤 2：TF-IDF 向量化 ━━
    print("\n📋 步骤 2：TF-IDF 向量化（演示）")
    mini_docs = [
        "猫坐在垫子上",
        "狗坐在地毯上",
        "机器学习是人工智能的一个分支"
    ]
    vocab = build_vocabulary(mini_docs)
    idf_values = compute_idf(mini_docs, vocab)

    print(f"  词汇表大小：{len(vocab)}")
    print(f"  高分 IDF 词（区分度高的词）：")
    scored = sorted(zip(vocab, idf_values), key=lambda x: x[1], reverse=True)
    for word, score in scored[:6]:
        print(f"    {word:10s} IDF={score:.3f}")

    # ━━ 步骤 3：相似度测试 ━━
    print("\n📋 步骤 3：余弦相似度")
    emb1 = tfidf_embed(mini_docs[0], vocab, idf_values)
    emb2 = tfidf_embed(mini_docs[1], vocab, idf_values)
    emb3 = tfidf_embed(mini_docs[2], vocab, idf_values)

    print(f"  '猫坐在垫子上' vs '狗坐在地毯上':       {cosine_similarity(emb1, emb2):.4f}  ← 结构相似")
    print(f"  '猫坐在垫子上' vs '机器学习是AI的分支': {cosine_similarity(emb1, emb3):.4f}  ← 不相关")
    print(f"  '狗坐在地毯上' vs '机器学习是AI的分支': {cosine_similarity(emb2, emb3):.4f}  ← 不相关")

    # ━━ 步骤 4：完整流水线 ━━
    print("\n📋 步骤 4：完整 RAG 流水线")
    rag = RAGPipeline(chunk_size=50, overlap=10, top_k=3)
    source_names = [
        "退款政策.md",
        "产品概览.md",
        "安全实践.md",
        "API文档.md",
        "可靠性SLA.md"
    ]
    num_chunks = rag.index(SAMPLE_DOCUMENTS, source_names)
    print(f"  索引完成：{len(SAMPLE_DOCUMENTS)} 篇文档 → {num_chunks} 个块")
    print(f"  词表大小：{len(rag.vocab)} 个词")

    queries = [
        "企业版客户的退款政策是什么？",
        "API的速率限制是多少？",
        "客户数据怎么加密的？",
        "如果运行时间低于SLA会怎样？",
        "专业版多少钱一个月？"
    ]

    for query in queries:
        print(f"\n  ❓ 问题：{query}")
        result = rag.query(query, top_k=3)
        print(f"  💬 回答：{result['answer']}")
        print(f"  📎 检索到 {len(result['retrieved'])} 个相关块：")
        for r in result["retrieved"]:
            preview = r["chunk"][:60].replace("\n", " ")
            print(f"      [{r['source']}] 相似度={r['score']:.4f} | {preview}...")

    # ━━ 步骤 5：分块大小对比 ━━
    print("\n📋 步骤 5：分块大小对检索质量的影响")
    test_query = "企业版客户的退款政策是什么？"
    for chunk_size in [20, 50, 100, 200]:
        rag_test = RAGPipeline(chunk_size=chunk_size, overlap=max(5, chunk_size // 5))
        n = rag_test.index(SAMPLE_DOCUMENTS)
        result = rag_test.query(test_query, top_k=3)
        top_score = result["retrieved"][0]["score"] if result["retrieved"] else 0
        print(f"  块大小={chunk_size:>3}: {n:>3}个块, 最高相似度={top_score:.4f}, "
              f"答案长度={len(result['answer'])}字")

    # ━━ 总结 ━━
    print(f"\n💡 RAG 流水线总结：")
    print(f"  • 五步流水线：文档 → 分块 → 向量化 → 检索 → 生成")
    print(f"  • 分块大小影响检索精度：太小丢失上下文，太大稀释相关性")
    print(f"  • TF-IDF 只是演示，生产环境用 text-embedding-3-small 等嵌入模型")
    print(f"  • 检索和生成的分离是RAG的核心设计：检索找信息，生成写答案")
    print(f"  • Prompt 里必须写'仅根据上下文回答'，否则LLM会用内部知识")
    print(f"  • 这就是为什么 RAG 比微调更灵活——上下文可以随时更新")


if __name__ == "__main__":
    main()
