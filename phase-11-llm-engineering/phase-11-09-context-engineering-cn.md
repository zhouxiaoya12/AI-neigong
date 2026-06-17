# 上下文工程：窗口、预算、记忆与检索

> 提示工程只是一个子集。上下文工程才是全局。提示词是你敲进去的一个字符串。上下文是进入模型窗口的一切：系统指令、检索到的文档、工具定义、对话历史、少样本示例，以及提示词本身。2026 年最好的 AI 工程师是上下文工程师。他们决定什么进去、什么留在外面、以什么顺序排列。

**类型**：构建  
**语言**：Python  
**前置要求**：阶段 10（从零开始 LLM）、阶段 11 第 01-02 课  
**预计时间**：约 90 分钟  
**相关内容**：阶段 11 · 第 15 课（提示词缓存）——缓存友好的布局是上下文工程的延伸。阶段 5 · 第 28 课（长上下文评估）了解如何用 NIAH/RULER 衡量"迷失在中间"效应。

## 学习目标

- 计算跨所有上下文窗口组件的 token 预算（系统提示词、工具、历史、检索文档、生成预留空间）
- 实现上下文窗口管理策略：截断、摘要、和对话历史的滑动窗口
- 对上下文组件进行优先级排序和排序，最大化模型对最相关信息的注意力
- 构建一个根据查询类型和可用窗口空间动态分配 token 的上下文组装器

## 问题陈述

Claude Opus 4.7 拥有 200K token 的窗口（测试版 1M）。GPT-5 有 400K。Gemini 3 Pro 有 2M。Llama 4 号称 10M。这些数字听起来巨大——直到你把它们填满。

以下是一个编程助手的真实分解。系统提示词：500 token。50 个工具的工具定义：8,000 token。检索到的文档：4,000 token。对话历史（10 轮）：6,000 token。当前用户查询：200 token。生成预算（最大输出）：4,000 token。总计：22,700 token。这只占 128K 窗口的 18%。

但注意力并不随上下文长度线性缩放。拥有 128K token 上下文的模型付出了二次方的注意力成本（普通 Transformer 中为 O(n²)，尽管大多数生产模型使用高效注意力变体）。更重要的是，检索准确度会下降。"大海捞针"测试显示，模型在长上下文中很难找到放在中间位置的信息。Liu 等人（2023）的研究表明，LLM 在长上下文的首尾位置以近乎完美的准确度检索信息，但对于放在中间的信息（上下文的 40-70% 位置），准确度下降 10-20%。这种"迷失在中间"效应因模型而异，但影响所有当前架构。

实践教训：有 200K token 可用，不意味着用满 200K token 是有效的。精心策划的 10K token 上下文，往往胜过硬塞进去的 100K token 上下文。上下文工程是一个最大化上下文窗口内信噪比的学科。

你放进窗口的每个 token，都会挤掉一个本来可以携带更相关信息的 token。每个无关的工具定义、每一轮过时的对话、每个不能回答问题的检索文本块——每一个都会让模型在任务上稍微变差一点。

## 核心概念

### 上下文窗口是稀缺资源

把上下文窗口想象成 RAM，而不是硬盘。它快速且可直接访问，但容量有限。你不能把所有东西都塞进去。你必须做出选择。

```
┌─────────────────────────────────────────────────────┐
│           上下文窗口（128K token）                      │
│                                                       │
│  系统提示词 ~500 token                                 │
│  ↓                                                     │
│  工具定义 ~2K-8K token                                 │
│  ↓                                                     │
│  检索上下文 ~2K-10K token                              │
│  ↓                                                     │
│  对话历史 ~2K-20K token                                │
│  ↓                                                     │
│  少样本示例 ~1K-3K token                               │
│  ↓                                                     │
│  用户查询 ~100-500 token                               │
│  ↓                                                     │
│  生成预算 ~2K-8K token（留给回答的空间）                │
└─────────────────────────────────────────────────────┘
```

每个组件都在竞争空间。添加更多工具定义意味着留给对话历史的空间更少。添加更多检索上下文意味着留给少样本示例的空间更少。上下文工程就是分配这笔预算以最大化任务表现的艺术。

### 迷失在中间

这是上下文工程中最重要的经验发现。模型对上下文开头和结尾的信息关注度更高。中间的信息获得较低的注意力分数，更容易被忽略。

Liu 等人（2023）对此进行了系统性测试。他们将一份相关文档放在 20 份不相关文档的不同位置，测量回答准确度。当相关文档排在第一或最后时，准确度为 85-90%。当它排在中间（20 份文档中的第 10 位）时，准确度下降到 60-70%。

这有直接的工程含义：

- 把最重要的信息放在最前面（系统提示词、关键指令）
- 把当前查询和最相关的上下文放在最后（近因偏差会帮助模型）
- 将上下文中间位置视为最低优先级区域
- 如果必须在中间包含信息，在末尾重复关键点

```
注意力在上下文中的分布：
┌──────────┬──────────┬────────────────┬──────────┬──────────┐
│ 前 0-20% │ 20-40%   │    40-70%      │ 70-90%   │ 90-100%  │
│ 高注意力 │ 中等      │   低注意力      │ 中等      │ 高注意力  │
│(系统提示)│          │  (迷失在中间)    │          │(当前查询) │
└──────────┴──────────┴────────────────┴──────────┴──────────┘
```

### 上下文组件

**系统提示词**：设定角色、约束和行为规则。放在最前面，跨轮次保持不变。Claude Code 的系统提示词约 6,000 token，包括工具定义和行为指令。保持紧凑。系统提示词中的每个词在每次 API 调用中都会重复。

**工具定义**：每个工具增加 50-200 token（名称、描述、参数 Schema）。每个 150 token 的 50 个工具，在对话还没开始时就已耗费 7,500 token。动态工具选择——只包含与当前查询相关的工具——可以将这减少 60-80%。

**检索上下文**：来自向量数据库的文档、搜索结果、文件内容。检索质量直接决定回答质量。糟糕的检索比不检索更差——它用噪声填满窗口并主动误导模型。

**对话历史**：每条之前的用户消息和助手回复。随对话长度线性增长。一次 50 轮的对话，每轮 200 token，就是 10,000 token 的历史。其中大部分与当前查询无关。

**少样本示例**：展示期望行为的输入/输出对。两到三个精心挑选的示例往往比数千 token 的指令更能提升输出质量。但它们消耗空间。

**生成预算**：为模型回复预留的 token。如果你把窗口填满了，模型就没有回答的空间。至少预留 2,000-4,000 token 用于生成。

### 上下文压缩策略

**历史摘要**：不是原封不动保留所有历史轮次，而是定期对对话进行摘要。"我们讨论了 X，决定了 Y，用户想要 Z"用 100 token 替代了原本需要 2,000 token 的 10 轮对话。当历史超过阈值（如 5,000 token）时触发摘要。

**相关性过滤**：对照当前查询对每个检索文档进行打分，丢弃低于阈值的文档。如果检索到 10 个块但只有 3 个相关，丢弃另外 7 个。3 个高度相关的块远胜于 10 个平庸的块。

**工具裁剪**：对用户查询意图进行分类，只包含与该意图相关的工具。代码问题不需要日历工具。日程安排问题不需要文件系统工具。这可以将工具定义从 8,000 token 减少到 1,000。

**递归摘要**：对于非常长的文档，分阶段进行摘要。先摘要每个章节，再摘要这些摘要。一个 50 页的文档变成一个捕捉关键点的 500 token 摘要。

### 记忆系统

上下文工程跨越三个时间尺度。

**短期记忆**：当前对话。直接存储在上下文窗口中。随每轮增长。通过摘要和截断管理。

**长期记忆**：跨对话持续的事实和偏好。"用户偏好 TypeScript。""项目使用 PostgreSQL。"存储在数据库中，对话开始时检索。Claude Code 将其存储在 CLAUDE.md 文件中。ChatGPT 将其存储在其记忆功能中。

**情景记忆**：可能相关的特定历史交互。"上周二，我们在 auth 模块调试了类似的问题。"存储为 Embedding，当前对话匹配过去某次情景时检索。

```
┌─────────────────────────────────────────────┐
│              记忆架构                        │
│                                              │
│  短期记忆 ──→ 上下文窗口                      │
│  (当前对话，直接存储)                         │
│                                              │
│  长期记忆 ──→ 上下文窗口                      │
│  (事实、偏好，数据库 → 对话启动时检索)         │
│                                              │
│  情景记忆 ──→ 上下文窗口                      │
│  (历史交互，Embedding → 相似度检索)            │
└─────────────────────────────────────────────┘
```

### 动态上下文组装

核心洞察：不同的查询需要不同的上下文。静态的系统提示词 + 静态的工具 + 静态的历史是一种浪费。最好的系统动态地为每个查询组装上下文。

1. 对查询意图进行分类
2. 选择相关工具（不是全部工具）
3. 检索相关文档（不是固定集合）
4. 包含相关的历史轮次（不是全部历史）
5. 添加匹配任务类型的少样本示例
6. 按重要性对所有内容排序：关键的最前，次重要的最后，可选的放在中间

这就是好的 AI 应用和优秀的 AI 应用之间的区别。模型是相同的。上下文是差异所在。

---

## 动手构建

### 第 1 步：Token 计数器

你不能预算你无法测量的东西。构建一个简单的 token 计数器（用空格分割进行近似，因为精确计数取决于特定的分词器）。

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    # 近似：英文约 1.3 token/词
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### 第 2 步：上下文预算管理器

核心抽象。预算管理器跟踪每个组件使用了多少 token，并强制执行限制。

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        # 计算 token 数
        tokens = count_tokens(content)
        # 如果超过组件级别上限，截断
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        # 检查全局预算
        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        # 剩余可用 token
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        # 使用率
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        # 生成预算报告
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"上下文预算报告 ({self.max_tokens:,} token 窗口)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f" {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f" {'已用':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f" {'生成预留':<25} {self.generation_reserve:>6} tokens")
        lines.append(f" {'剩余':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### 第 3 步：迷失在中间重排序

实现重排序策略：最重要的项放在首尾，最不重要的放在中间。

```python
def reorder_lost_in_middle(items, scores):
    # 按评分降序排序
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    # 交叉放置：最好的放首尾，最差的放中间
    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    # 基于词重叠的简单相关性评分
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### 第 4 步：对话历史压缩器

摘要旧的对话轮次以回收 token 预算。

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        # 当历史超过 token 上限时，压缩最旧的轮次
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        # 简化摘要：拼接并截断
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "上一段: " + " | ".join(parts)

    def get_context(self):
        # 组装：先摘要，后最近对话
        parts = []
        if self.summaries:
            parts.append("[对话摘要]")
            for s in self.summaries:
                parts.append(s)
            parts.append("[最近对话]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### 第 5 步：动态工具选择器

只包含与当前查询相关的工具。先分类意图，再过滤。

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "读取文件内容",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "将内容写入文件",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "搜索代码库中的模式",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "执行 shell 命令",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "创建新的日历事件",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "列出最近的邮件",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "发送邮件消息",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "搜索网络信息",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "对数据库执行 SQL 查询",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "从数据生成图表",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    # 基于关键词的意图分类
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test", "代码", "函数", "实现", "重构", "调试"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event", "会议", "日程", "安排"],
        "email": ["email", "mail", "send", "inbox", "message", "邮件", "发送", "收件箱"],
        "research": ["search", "find", "what is", "how does", "explain", "look up", "搜索", "查找", "解释", "什么是"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql", "数据", "查询", "图表", "分析"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    # 根据查询意图选择相关工具
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### 第 6 步：完整上下文组装管线

把所有东西串起来。给定一个查询，动态组装最优上下文。

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "你是一个有用的 AI 助手。你可以访问用于代码编辑、"
            "文件管理、网络搜索和数据分析的工具。"
            "为每个任务使用合适的工具。回答要简洁准确。"
        )
        self.knowledge_base = [
            "Python 3.12 引入了使用方括号语法的泛型类型参数语法。",
            "项目使用 PostgreSQL 16 配合 pgvector 进行 Embedding 存储。",
            "认证由 Supabase Auth 通过 JWT token 处理。",
            "前端使用 Next.js 15 配合 App Router 构建。",
            "API 速率限制为每用户每分钟 100 次请求。",
            "部署流水线使用 GitHub Actions 配合 Docker 多阶段构建。",
            "所有新模块的测试覆盖率必须超过 80%。",
            "代码库采用仓库模式进行数据访问。",
        ]

    def assemble(self, query):
        # 每次查询重新初始化预算
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        # 1. 系统提示词（最先，最重要）
        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        # 2. 工具定义（只包含相关工具）
        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        # 3. 检索上下文（相关性评分 + 重排序）
        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        # 4. 对话历史（已压缩）
        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        # 5. 用户查询（放最后，利用近因效应）
        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        # 模拟模型回复
        response = f"[回复: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print(" 上下文工程管线演示")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- 查询 1：代码任务 ---")
    budget = engine.chat("修复认证模块中 JWT token 过期太早的 bug")
    print(budget.report())

    print("\n--- 查询 2：研究任务 ---")
    budget = engine.chat("在 PostgreSQL 中实现向量搜索的最佳方案是什么？")
    print(budget.report())

    print("\n--- 查询 3：对话历史累积后 ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"关于系统实现细节的后续问题 {i+1}")
        engine.conversation.add_turn("assistant", f"这是对后续问题 {i+1} 的回答，包含架构相关技术细节")

    budget = engine.chat("现在实现我们讨论的改动")
    print(budget.report())

    print("\n--- 工具选择示例 ---")
    test_queries = [
        "修复 auth.py 中的 bug",
        "安排周二与团队的会议",
        "显示数据库查询性能统计",
        "搜索错误处理的最佳实践",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n 查询: {q}")
        print(f" 意图: {intents}")
        print(f" 工具: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- 迷失在中间重排序 ---")
    docs = ["文档A（最相关）", "文档B（较相关）", "文档C（最不相关）",
            "文档D（相关）", "文档E（中等相关）"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f" 原始顺序: {docs}")
    print(f" 评分: {scores}")
    print(f" 重排序后: {reordered}")
    print(f" （最相关的在首尾，最不相关的在中间）")
```

## 实际使用

### Claude Code 的上下文策略

Claude Code 用分层方式管理上下文。系统提示词包含行为规则和工具定义（约 6K token）。当你打开文件时，其内容被注入为上下文。当你搜索时，结果被添加。旧对话轮次被摘要。CLAUDE.md 提供跨会话的长期记忆。

关键的工程决策：Claude Code 不会将整个代码库倾倒入上下文。它按需检索相关文件。这就是上下文工程在实践中的样子。

### Cursor 的动态上下文加载

Cursor 将你的整个代码库索引为 Embedding。当你输入查询时，它使用向量相似度检索最相关的文件和代码块。只有这些片段进入上下文窗口。一个 50 万行的代码库被压缩为 5-10 个最相关的代码块。

这就是模式：Embedding 一切，按需检索，只包含重要的东西。

### ChatGPT 记忆

ChatGPT 将用户偏好和事实存储为长期记忆。每次对话开始时，相关记忆被检索并包含在系统提示词中。"用户偏好 Python"花费 5 token，却节省了跨对话重复相同指令的数百 token。

### RAG 作为上下文工程

检索增强生成是形式化的上下文工程。与其把知识塞进模型权重（训练）或系统提示词（静态上下文），不如在查询时检索相关文档，将其注入上下文窗口。整个 RAG 管线——分块、Embedding、检索、重排序——都是为了解决一个问题：把正确的信息放进上下文窗口。

## 交付物

本课产出 `outputs/prompt-context-optimizer.md`——一个用于审计上下文组装策略并推荐优化的可复用提示词。给它你的系统提示词、工具数量、平均历史长度和检索策略，它会识别 token 浪费并提出改进建议。

同时产出 `outputs/skill-context-engineering.md`——一个基于任务类型、上下文窗口大小和延迟预算设计上下文组装管线的决策框架。

## 练习

1. 为 ContextBudget 类添加"token 浪费检测器"。它应该标记使用超过预算 30% 的组件，并建议针对每种组件类型的压缩策略（摘要历史、裁剪工具、重排序文档）。

2. 为检索上下文实现语义去重。如果两份检索文档有超过 80% 的相似度（通过词重叠或 Embedding 的余弦相似度），只保留评分更高的那份。测量这回收了多少 token 预算。

3. 构建一个"上下文回放"工具。给定一个对话记录，通过 ContextEngine 回放它，可视化预算分配如何逐轮变化。绘制各组件随时间变化的 token 使用图。找出上下文开始被压缩的轮次。

4. 实现基于优先级的工具选择器。不是二元的包含/排除，而是为每个工具分配对当前查询的相关性分数。按相关性降序包含工具直到工具预算耗尽。比较包含 5、10、20 和 50 个工具时的任务表现。

5. 构建多策略上下文压缩器。实现三种压缩策略（截断、摘要、提取关键句），并在 20 份文档的集合上对它们进行基准测试。测量压缩比与信息保留之间的权衡（压缩版本是否仍然包含查询的答案？）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 上下文窗口 | "模型能读多少" | 模型在单次前向传播中处理的最大 token 数（输入+输出）——GPT-5 为 400K，Claude Opus 4.7 为 200K（测试版 1M），Gemini 3 Pro 为 2M |
| 上下文工程 | "高级提示工程" | 决定什么进入上下文窗口、以什么顺序、以什么优先级进入的学科——涵盖检索、压缩、工具选择和记忆管理 |
| 迷失在中间 | "模型忘了中间的东西" | LLM 更关注上下文开头和结尾的经验发现，中间位置的信息准确度下降 10-20% |
| Token 预算 | "你还剩多少 token" | 跨组件（系统提示词、工具、历史、检索、生成）的上下文窗口容量显式分配，带每个组件的限制 |
| 动态上下文 | "动态加载东西" | 基于意图分类、相关工具选择和检索结果，为每个查询不同地组装上下文窗口 |
| 历史摘要 | "压缩对话" | 用简洁摘要替代原样保留的旧对话轮次，降低 token 成本同时保留关键信息 |
| 工具裁剪 | "只包含相关工具" | 对查询意图分类，只包含匹配的工具定义，将工具 token 成本降低 60-80% |
| 长期记忆 | "跨会话记忆" | 存储在数据库中并在会话启动时检索的事实和偏好——CLAUDE.md、ChatGPT 记忆等系统 |
| 情景记忆 | "记住特定的过去事件" | 存储为 Embedding 的过去交互，当前查询与某次历史对话相似时检索 |
| 生成预算 | "留给回答的空间" | 为模型输出预留的 token——如果上下文完全填满窗口，模型没有空间回答 |

## 拓展阅读

- [Liu 等人，2023——"迷失在中间：语言模型如何使用长上下文"](https://arxiv.org/abs/2307.03172)——关于位置依赖注意力的权威研究，展示模型在长上下文中间信息上的困难
- [Anthropic 的上下文检索博客文章](https://www.anthropic.com/news/contextual-retrieval)——Anthropic 如何实现上下文感知的块检索，将检索失败降低 49%
- [Simon Willison 的"上下文工程"](https://simonwillison.net/2025/Jun/27/context-engineering/)——命名了这个学科并将其与提示工程区分开的博客文章
- [LangChain RAG 文档](https://python.langchain.com/docs/tutorials/rag/)——检索增强生成作为上下文工程模式的实践实现
- [Greg Kamradt 的大海捞针测试](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)——揭示所有主流模型中位置依赖检索失败的基准测试
- [Pope 等人，"高效缩放 Transformer 推理"（2022）](https://arxiv.org/abs/2211.05102)——为什么上下文长度驱动内存和延迟，以及 KV 缓存、MQA 和 GQA 如何改变预算计算
- [Agrawal 等人，"SARATHI：通过搭载解码与分块预填充实现高效 LLM 推理"（2023）](https://arxiv.org/abs/2308.16369)——推理的两个阶段使长提示在 TTFT 上昂贵但在 TPOT 上便宜；上下文打包权衡的基础真相
- [Ainslie 等人，"GQA：从多头检查点训练广义多查询 Transformer 模型"（EMNLP 2023）](https://arxiv.org/abs/2305.13245)——将生产级解码器中 KV 内存削减 8 倍且不损失质量的 grouped-query attention 论文

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这是整个 11 阶段中最"架构级"的一课。它不是教你怎么用某个 API，而是教你怎么思考模型接收到的每一寸信息。开篇一句话直接拔高了定位——"提示工程是子集，上下文工程是全局"——这既是定位声明也是认知重构。全篇围绕两个核心概念展开：token 预算的显式管理（稀缺资源思维）和"迷失在中间"的经验发现（注意力分布不均）。后者是本课最具教学价值的实证依据——它把一个抽象的"信息太多不好"变成了可量化的位置-准确度曲线。三记忆系统（短期/长期/情景）的框架将上下文工程从单次交互扩展到了跨会话的持久性架构。

### 二、知识结构梳理

**认知基础层**：上下文窗口不是硬盘是 RAM 的类比、token 预算六组件模型（系统→工具→检索→历史→示例→生成）、"迷失在中间"的位置-准确度实证

**工程模式层**：四种压缩策略（历史摘要、相关性过滤、工具裁剪、递归摘要）、动态 vs 静态上下文的架构选择、Lost-in-the-Middle 重排序算法（首尾放最重要的）

**实践应用层**：ContextBudget 类的分配-截断-报告三件套、动态工具选择器的意图分类管道、ConversationManager 的阈值触发压缩模式、Claude Code/Cursor/ChatGPT 三个真实产品的上下文策略对比

### 三、核心洞察（备课时的关键理解）

1. **"有空间"和"用得有效"是两回事。** 128K 窗口填 23K token 看起来还有大把空间，但注意力不是均匀分布的——中间的信息已经被模型"打折处理"了。学生最容易犯的错误是以为窗口没满就可以随便填。

2. **"迷失在中间"是本课最有杀伤力的数据。** 85-90% vs 60-70% 的准确度差距不是可以忽略的。这意味着如果你把关键信息放在错误的位置，相当于给模型的任务准确度打了七折。这个数字要在课堂上反复强调。

3. **工具定义是"沉默的 token 杀手"。** 50 个工具 × 150 token = 7,500 token，在用户还没说第一句话之前就已经消耗了。动态工具选择不是优化，是必需品。60-80% 的减少不是锦上添花。

4. **糟糕的检索比不检索更危险。** 这是一个反直觉的洞察。不检索只是损失了潜在知识。糟糕的检索不仅浪费 token，还把模型引向错误方向——噪声比空白更糟。

5. **三种记忆的时间尺度区分极其清晰。** 短期=窗口内，长期=数据库→启动时加载，情景=Embedding→相似度检索。这个三层框架一旦建立，学生自己就能推导出各种记忆管理策略。

6. **上下文工程实际上把"提示工程师"的技能范围扩展了三倍。** 不再只是写指令文字，而是做预算分配、压缩算法、检索策略、记忆系统设计。这是给学生的职业路线图升级。

7. **Claude Code/Cursor/ChatGPT 三个案例的并列展示非常有力。** 三种不同的上下文策略：按需检索文件 vs Embedding 全量索引检索 vs 跨会话记忆。学生看到的是同一套原则在不同产品中的不同实现选择。

### 四、教学建议

1. **开场做可视化的"窗口填满"演示。** 在黑板上画一个代表 128K 窗口的方框，让学生自己列举所有要放进去的东西——系统提示词、50 个工具、10 轮对话、10 个检索文档——然后算总 token。在他们发现"等等，这才用了 20%"时，再抛出"迷失在中间"数据。冲高-打低的节奏非常有效。

2. **"迷失在中间"一定要用可视化。** 画一个横条，用颜色标注五个区域的注意力强度（绿-橙-红-橙-绿）。这条彩色横条比任何文字都直观——学生看到"中间是红的"就永远不会忘记要把重要信息放在首尾。

3. **ContextBudget.report() 的输出格式要让学生自己跑一遍。** 让他们修改参数（窗口大小、工具数量、历史长度），看报告怎么变。特别是让工具数量从 50 减到 5——7,500 token 变成 750 token——这个数字跳变很有冲击力。

4. **工具选择器是让学生动手理解"动态上下文"的最佳入口。** 给四个不同场景的查询（代码/日历/邮件/数据），让学生运行 classify_intent 和 select_tools，看每次选中了哪些工具、省了多少 token。从"静态全部工具"到"动态选择"的对比最有说服力。

5. **三个记忆系统的区分用生活类比。** 短期=你在脑子里的当前对话，长期=你写在笔记本上的偏好和事实，情景="上次我们也遇到这个问题"。学生不需要理解 Embedding 技术就能理解这三层架构的合理性。

6. **练习 3（上下文回放）做成课堂竞赛。** 让学生对比谁的对话历史最长、谁最先触发压缩、谁的压缩策略回收了最多 token。互动数据可视化比静态讲解有趣得多。

7. **最后一定要把这些概念连接到 RAG。** 上节课刚学过 RAG，这节课揭示 RAG 的本质——RAG 就是上下文工程的一个特例。这个连接让学生看到知识体系是连贯的，不是孤立的。

### 五、值得补充的内容

1. **注意力分布的模型差异。** "迷失在中间"因模型而异——Claude 和 GPT 的注意力分布并不完全相同。值得补充一个简短的对比表，让学生知道这个效应是"存在"但不是"固定值"。

2. **token 计数器的精度问题。** 本课用空格 ×1.3 近似，但不同模型的分词器差异很大——同样的英文文本在 GPT-4 分词器和 Claude 分词器下可能差 10-20%。生产环境应该用 tiktoken 或对应厂商的 tokenizer。建议至少提及这一点。

3. **缓存对预算计算的影响。** 如果你缓存了系统提示词和前几轮对话（阶段 15 的内容），那么这些 token 虽然还在窗口里，但 API 成本不按全部 token 计费。这个区别在设计成本敏感的上下文策略时至关重要。

4. **中文上下文的特殊性。** 中文没有分词空格，1.3 倍的近似规则不适用。中文约 1.5-2.0 token/字。如果学生做中文应用，需要调整 token 计数器。

### 六、一句话总结

**上下文工程不是"怎么塞更多东西进去"——而是"怎么让每一寸空间都在为答案服务"。窗口大小是硬件，信噪比是软件。最好的工程师不迷信大窗口，他们迷信精心的排布。**

---


---

# 🎓 Agent 架构课：上下文工程——在模型的眼皮底下搭建舞台

同学们好。

前面几节课，我们给 Agent 装上了外部记忆（RAG），又给它装上了双手（函数调用）。今天这节课，我们讨论一个更底层的问题——一个你每次调用 LLM 时都在处理、但很少有人系统讲的问题。

**上下文窗口。** 你塞进去的每一寸信息，模型都要看。问题是：它真正看到的是哪部分？它忽略了哪部分？你怎么决定放什么、不放什么、按什么顺序放？

今天的核心问题不是"GPT-5 的窗口有多大"。那个查 API 文档就知道。**核心问题是：明明有 200K token 的空间，为什么塞 23K token 进去就已经开始降低模型的任务表现了？**

答案是一个叫"迷失在中间"的发现。这是 2023 年以来长上下文研究最重要的实证结论——也是这节课最值得你记住的东西。

## 上下文窗口不是硬盘，是 RAM

先把一个错误观念纠正掉。很多人看到 200K、400K、2M 这些数字，第一反应是"太好了，我把什么都塞进去就行"。不是。

上下文窗口是 RAM，不是硬盘。它很快，容量也还行，但不是无限的。而且——最关键的一点——它的每一寸空间都在竞争模型的注意力。放进去的每个 token 都会挤掉另一个 token。

让我给你拆一个真实场景。一个编程助手的上下文：

- 系统提示词：500 token
- 50 个工具的定义：8,000 token
- 检索到的文档：4,000 token
- 10 轮对话历史：6,000 token
- 用户当前查询：200 token
- 留给回答的空间：4,000 token

总共 22,700 token。在 128K 的窗口里这算什么？不到 18%。看起来还有很多空间。

但问题不在这。问题在于模型的注意力不是均匀分布的。

## 迷失在中间——本课最重要的发现

2023 年，Liu 等人做了一系列实验。他们把一份关键文档放在一堆无关文档中的不同位置，然后测试 LLM 能不能正确回答基于这份文档的问题。结果很残酷。

当关键文档放在最前面时，准确度 85-90%。放在最后面时，也是 85-90%。

但当关键文档放在中间时——大约第 10 位，在 20 份文档的正中间——准确度掉到了 60-70%。

20% 的准确度差距，仅仅因为位置不同。内容是一模一样的。模型看是看到了。但它没有真正"注意"到。

这不是一个 bug。这是 Transformer 注意力机制的固有属性。模型训练时学到了一个模式：开头通常是系统指令，结尾通常是当前问题——这两端最重要。中间的内容呢？大概率是填充物。所以模型学会了在中间区域"走神"。

**这意味着什么？这意味着你把最关键的信息放在上下文中间，等于告诉模型"这段不太重要"。**

上下文窗口就像一个U形注意力的容器：两端高，中间低。

## 上下文工程的六笔预算

既然注意力不均匀，你就要主动管理每一寸上下文。"把所有东西都放进去"不是策略，是偷懒。真正的上下文工程是**显式分配预算**。

六笔账，一笔一笔算：

**第一笔：系统提示词。** 放在最前面。角色定义、行为规则、输出约束。Claude Code 的系统提示词约 6,000 token——每一轮调用都在花这笔钱。别在这里废话。每个词都在增加成本。

**第二笔：工具定义。** 50 个工具，每个 150 token，就是 7,500 token。但你的用户问的是代码问题，日历工具、邮件工具跟这件事有什么关系？动态工具选择——只包含跟当前查询相关的工具——能把这笔开销砍掉 60-80%。7,500 变 1,500。这不是优化，这是常识。

**第三笔：检索上下文。** 来自向量数据库、搜索结果、文件内容。这是你最需要控制质量的部分。检索到 10 个块但只有 3 个相关？扔掉 7 个。坏的检索比不检索更危险——它不仅浪费 token，还把模型往错误方向引。

**第四笔：对话历史。** 线性增长的预算黑洞。50 轮对话 × 200 token = 10,000 token 的历史。其中有多少跟当前问题真正相关？定期摘要——"我们讨论了 X，决定了 Y"——用 100 token 替代 2,000 token。超过 5,000 token 就触发压缩。

**第五笔：少样本示例。** 输入输出对，展示你期望的行为模式。两三个好示例比一千字的指令更有效。但每个示例都在消耗 token 预算。

**第六笔——也是很多人忘记的：生成预留空间。** 如果你把 128K 窗口塞满了 128K 的输入，模型就没有空间输出答案了。至少留 2,000-4,000 token 给它回答。

## 四种压缩策略，四级杀伤力

当预算不够时，你有四种武器：

**历史摘要**：把 10 轮对话压缩成 100 token 的摘要。最简单，效果最显著。500 行的对话变一段话，关键信息不丢。

**相关性过滤**：对检索到的文档打分，低于阈值的扔掉。宁可 3 个高度相关的，不要 10 个勉强沾边的。

**工具裁剪**：分类查询意图，只加载相关工具。"代码问题"不需要邮件工具。"日程安排"不需要文件工具。这是最快见效的优化。

**递归摘要**：对超长文档，先摘要每个章节，再摘要这些摘要。50 页文档 → 500 token 摘要。只保留骨架。

## 三种记忆，三个时间尺度

上下文不是只有一次对话。从架构角度看，记忆分三个时间尺度：

**短期记忆**：当前对话。存在上下文窗口里，随轮次增长，通过摘要和截断管理。这是你能最精细控制的部分。

**长期记忆**：跨对话持久的事实和偏好。"用户用 TypeScript。""项目用 PostgreSQL。"存储在数据库里，每次对话开始时检索。Claude Code 用 CLAUDE.md 实现，ChatGPT 用内置记忆功能。5 token 的偏好声明，省掉每次对话重复 200 token 指令。

**情景记忆**：不是"你喜欢什么"，而是"上次发生了什么"。"上周二我们在这个问题上花了两小时，最后发现是缓存没清。"这类记忆存储为 Embedding，当当前对话跟某段历史相似时自动检索出来。

三层记忆架构不是理论——Claude Code、Cursor、ChatGPT 都在用。它们各自实现不同，但三层模型是相同的。

## 动态上下文组装——把策略变成代码

到目前为止都是概念。怎么变成代码？五个步骤：

1. **分类意图**。用关键词或轻量分类器判断用户想问什么——代码？日历？邮件？数据？
2. **选择工具**。根据意图，只加载相关工具。不是全部 50 个，是匹配的那 5-8 个。
3. **检索文档**。从知识库中按相关性检索，打分，低于阈值就丢。
4. **重排序**。用"迷失在中间"策略：最相关的放首尾，最不相关的放中间。
5. **组装**。系统提示词→工具→检索→历史→用户查询。首尾是重要的，中间是过滤后的。

你注意到没有——**这个顺序不是随便排的。它是针对注意力分布精心设计的。** 开头放最关键的系统指令（首部高注意力），末尾放用户问题和最相关上下文（近因偏差），中间放中等相关、即使被忽略了也不致命的内容。

这就是上下文工程的核心：不是写更好的提示词，是设计更好的信息排布。

## Claude Code vs Cursor vs ChatGPT——同样的原则，不同的实现

看看三个真实产品的上下文策略：

**Claude Code**：分层加载。系统提示词 + 工具定义是固定的（~6K token）。文件按需加载——不是你打开项目就全部塞进去。搜索结果是增量的。旧对话被摘要。CLAUDE.md 提供长期记忆。

**Cursor**：全量 Embedding 索引。把你的整个代码库 Embedding 化。每次查询，用向量搜索找到最相关的 5-10 个代码块。50 万行代码库 → 5-10 个相关块。这是另一种上下文工程哲学——不是"加载文件"，是"检索相关片段"。

**ChatGPT 记忆**：专注跨会话持久性。"用户偏好简洁回答""用户的公司是 XXX"。小信息，大价值。每个偏好声明只有 5-10 token，但避免了几百 token 的重复指令。

三种产品，同一套原则：**不把所有东西塞进窗口。只放必要的。按重要性排列。持续压缩。**

## RAG 的真相——上下文工程的特例

如果你上节课学了 RAG，现在回头看，你会发现 RAG 就是上下文工程的完美例子。

RAG 管线的每一步——分块、Embedding、检索、重排序——都是为了解决同一个问题：**把正确的信息放进上下文窗口的正确位置。**

分块是为了让每一块都是可检索的、有意义的单元。Embedding 是为了把"相关"从关键词匹配变成语义空间距离。检索是为了找到 k 个最相关的块。重排序是为了把最相关的放在模型注意力最强的地方。

RAG 不是一项技术。RAG 是上下文工程的一种实现。当你理解了这个，你的 RAG 系统就不再是一个"能查文档的 chatbot"——它是一个精心设计的注意力优化引擎。

## 什么时候不用动态上下文？

不是所有场景都需要复杂的上下文工程。以下情况直接用静态上下文就好：

1. **固定任务，固定工具。** 如果你的 Agent 只做一件事（比如代码审查），系统提示词、工具集、所有东西都是固定的。不需要动态切换。

2. **延迟要求极端。** 意图分类 + 工具选择 + 文档检索 + 重排序——这套流程本身有延迟。如果端到端必须在 100ms 内完成，静态上下文更快。

3. **对话很短。** 用户问一个问题就结束。没有累积历史需要压缩，没有多轮上下文需要管理。

但当你的 Agent 有 10 个以上的工具、对话超过 20 轮、每次查询要检索多份文档时——动态上下文组装不是可选，是必须。

## 总结：Agent 架构师的上下文工程检查清单

1. ✅ **测量**：你知道每个组件消耗了多少 token 吗？先计数，再优化。
2. ✅ **预算分配**：系统提示词、工具、检索、历史、生成——每笔账都有上限吗？
3. ✅ **首尾排重要**：是否利用了"迷失在中间"效应——最重要的信息在首尾？
4. ✅ **动态工具选择**：每次查询只加载相关工具，还是 50 个全带上？
5. ✅ **历史压缩**：对话超过 5,000 token 时，有没有触发摘要？
6. ✅ **检索质量**：坏结果过滤掉了吗？相关性阈值合理吗？
7. ✅ **生成预留**：窗口填满前，至少留了 2,000-4,000 token 给模型回答了吗？
8. ✅ **三层记忆**：当前对话（短期）、持久偏好（长期）、历史情境（情景）——三层都覆盖了吗？

记住一件事：**上下文工程不是在帮模型看更多东西。上下文工程是在帮模型把有限的注意力，花在真正值得看的信息上。窗口大不是优势。排布好才是。**

---

