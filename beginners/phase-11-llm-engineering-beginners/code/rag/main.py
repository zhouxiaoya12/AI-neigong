"""
📚 RAG · 初学者版

RAG = Retrieval-Augmented Generation（检索增强生成）

简单说：AI 不是靠"脑子里记住的"来回答问题，
而是在回答前先去查资料，用查到的资料来回答。

流程：问题 → 搜索相关文档 → 把文档内容塞进提示词 → AI 基于文档回答

这里用最简单的 TF-IDF（词频-逆文档频率）来模拟"搜索"这一步。
"""

import math
import re
from collections import Counter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第1步：分词（中英文都支持）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tokenize(text):
    """把文本切成词。中文用2字词，英文按空格切。"""
    tokens = []
    segments = re.split(r'([\u4e00-\u9fff]+)', text)
    for seg in segments:
        if not seg.strip():
            continue
        if re.match(r'[\u4e00-\u9fff]+', seg):
            # 中文：2字一组
            clean = seg.strip()
            for i in range(len(clean) - 1):
                tokens.append(clean[i:i+2])
        else:
            for tok in seg.strip().lower().split():
                tok = tok.strip('.,;:!?，。；：！？"')
                if tok:
                    tokens.append(tok)
    return tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第2步：建索引（把文档转成数字向量）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_vocab(docs):
    """从所有文档中收集词汇表"""
    vocab = set()
    for doc in docs:
        vocab.update(tokenize(doc))
    return sorted(vocab)

def tfidf_vector(text, vocab, idf_values):
    """计算一个文本的 TF-IDF 向量"""
    tokens = tokenize(text)
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    # TF × IDF
    return [counts.get(w, 0) / total * idf_values[i] for i, w in enumerate(vocab)]

def compute_idf(docs, vocab):
    """计算 IDF（区分度）：常见词 IDF 低，罕见词 IDF 高"""
    n = len(docs)
    idf = []
    for word in vocab:
        doc_count = sum(1 for d in docs if word in tokenize(d))
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第3步：搜索（用余弦相似度找最相关的文档块）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cosine_similarity(a, b):
    """余弦相似度：0=不相关，1=完全相同"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0

def search(query_vec, doc_vecs, top_k=3):
    """搜索最相似的文档"""
    scores = []
    for i, vec in enumerate(doc_vecs):
        scores.append((i, cosine_similarity(query_vec, vec)))
    scores.sort(key=lambda x: -x[1])
    return [(idx, s) for idx, s in scores[:top_k] if s > 0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 55)
    print("📚 RAG · 初学者版")
    print("=" * 55)

    # 知识库：几个中文文档
    docs = [
        "退款政策：标准版30天内全额退款。企业版60天按比例退款。",
        "价格：新手版29元/月，专业版99元/月，企业版500元起/月。",
        "安全：数据AES-256加密存储，TLS 1.3加密传输。",
        "API：REST+JSON，OAuth2认证，新手100次/分钟。",
    ]
    sources = ["退款政策", "价格表", "安全文档", "API文档"]

    print("\n📄 知识库：")
    for s, d in zip(sources, docs):
        print(f"  • {s}: {d}")

    # 建索引
    vocab = build_vocab(docs)
    idf = compute_idf(docs, vocab)
    doc_vecs = [tfidf_vector(d, vocab, idf) for d in docs]

    # 搜索测试
    queries = [
        "企业版客户怎么退款？",
        "专业版多少钱？",
        "数据怎么加密的？",
        "API调用限制是多少？",
    ]

    print(f"\n🔍 搜索测试：")
    for q in queries:
        q_vec = tfidf_vector(q, vocab, idf)
        hits = search(q_vec, doc_vecs, top_k=2)
        print(f"\n  问：{q}")
        for idx, score in hits:
            print(f"    → [{sources[idx]}] 相似度 {score:.2f}: {docs[idx][:40]}...")

    print(f"\n💡 RAG 的核心思想：")
    print(f"   AI 回答前先去资料库里搜，用在上下文里找到的内容来回答。")
    print(f"   这样 AI 就不用「背」所有知识了——知识库可以随时更新。")
