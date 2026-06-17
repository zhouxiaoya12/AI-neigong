# 提示工程：你跟 AI 说"随便写"，就别怪它"随便写"

> **原文：** Prompt Engineering: Techniques & Patterns  
> **预计时间：** ~60 分钟  
> **代码：** Python（完整保留，加中文注释）

---

## 🚗 开场翻车：20 分钟改同一句话

你打开 ChatGPT，输入："帮我写一封营销邮件。"

它回了洋洋洒洒 500 词，全是套话："我们很高兴地宣布……"、"在这个快速变化的时代……"。

你加细节，再试。好了一点，但语气不对。再加，再试。20 分钟过去了，你还在改同一句话。

**这不是模型的问题——是指令的问题。**

同一个任务，两种写法：

❌ **模糊：** "写一封营销邮件。"
✅ **工程化：** "你是 B2B SaaS 的资深文案。为 DevFlow（CI/CD 调试工具）写产品发布邮件。受众：B 轮创业公司的工程经理。语气：自信、技术化、不推销。长度：150 词。包含一个具体数据（调试速度提升 3.2 倍）。以单一 CTA 结尾。只输出正文。"

**第一个提示激活了模型训练数据里所有营销邮件的平均分布。第二个激活了一个窄而高质量的切片。同一模型，同一参数，完全不同的输出。**

---

## 🧠 核心概念：提示的三个部分

### 1. 系统消息——"看不见的手"

```python
# 系统消息设定模型的角色、规则和输出方式
# 模型把它当最高优先级的内容来处理
system = "你是资深 Python 开发者。只输出代码，不解释。"
```

| 提供商 | 怎么设系统消息 | 谁最听话 |
|--------|-------------|---------|
| OpenAI | messages 里 role="system" | 中等 |
| Anthropic | API 参数 `system=` | 最强 |
| Gemini | model 配置 `system_instruction=` | 中等 |

### 2. 用户消息——任务本身

这就是大多数人以为的"提示"。但没系统消息的用户消息是欠约束的。

### 3. 助手预填充——Anthropic 的秘密武器

```python
# 在 assistant 消息里塞一个 "{" 
# 强迫 Claude 继续输出 JSON，不产生任何废话
# 比在提示里说"请输出JSON"可靠得多
messages = [
    {"role": "assistant", "content": "{"},  # ← 这招只有 Claude 吃
]
```

---

## 🔧 十大提示模式（代码完整保留）

### 模式 1：人格模式

```python
# 人格模式：角色越具体，输出质量越高
# "你是一个助手" → 泛泛
# "你是 Stripe 工作了 10 年的支付系统专家" → 精准
PROMPT_PATTERNS = {
    "persona": {
        "name": "人格模式",
        "template": (
            "你是 {role}，拥有 {experience}。\n"
            "你的沟通风格是 {style}。\n"
            "你优先考虑 {priority}。\n\n"
            "{task}"
        ),
        "variables": ["role", "experience", "style", "priority", "task"],
        "temperature": 0.7,
        "description": "激活模型训练数据中特定的专家分布",
    },
```

### 模式 2：少样本模式

```python
    "few_shot": {
        "name": "少样本模式",
        "template": (
            "以下是期望的输入/输出格式示例：\n\n"
            "{examples}\n\n"
            "现在处理此输入：\n{input}"
        ),
        "variables": ["examples", "input"],
        "temperature": 0.0,
        "description": "用具体示例锚定输出格式。给 2-3 个例子比写一堆规则更有效。",
    },
```

### 模式 3：思维链

```python
    "chain_of_thought": {
        "name": "思维链模式",
        "template": (
            "逐步思考以下问题。\n\n"
            "问题：{problem}\n\n"
            "步骤：\n"
            "1. 识别关键组成部分\n"
            "2. 分析每个组成部分\n"
            "3. 综合你的发现\n"
            "4. 陈述你的结论\n\n"
            "在给出最终答案之前展示你的推理过程。"
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "强制模型在输出答案前展示推理。数学/逻辑问题提分 10-40%。",
    },
```

### 模式 4-10：模板、自我批评、护栏、分解、受众适配、边界、元提示

```python
    "template_fill": {
        "name": "模板填充",
        "template": (
            "从以下文本中提取信息并填写模板。\n\n"
            "文本：{text}\n\n"
            "模板：\n{template_structure}\n\n"
            "填写每一个字段。如果信息不可用，写 'N/A'。"
        ),
        "variables": ["text", "template_structure"],
        "temperature": 0.0,
        "description": "把自由文本塞进固定模板——提取姓名/日期/金额的神器",
    },
    "critique": {
        "name": "自我批评",
        "template": (
            "任务：{task}\n\n"
            "第 1 步：生成初始回答。\n"
            "第 2 步：批评你的回答——准确性、完整性、清晰度。\n"
            "第 3 步：基于批评生成改进版。\n\n"
            "清楚标注每个步骤。"
        ),
        "variables": ["task"],
        "temperature": 0.5,
        "description": "让模型自己审自己。两轮之后输出质量显著提升。",
    },
    "guardrail": {
        "name": "护栏",
        "template": (
            "你是一个 {role}。\n\n"
            "规则：\n"
            "- 只回答关于 {domain} 的问题\n"
            "- 超出范围时说：'这超出了我的范围。'\n"
            "- 绝不编造信息。不确定时说'我不知道。'\n"
            "- {additional_rules}\n\n"
            "用户问题：{question}"
        ),
        "variables": ["role", "domain", "additional_rules", "question"],
        "temperature": 0.3,
        "description": "画一个圈，告诉模型只能在圈里活动",
    },
    "meta_prompt": {
        "name": "元提示",
        "template": (
            "为一个能 {objective} 的 LLM 写一个提示。\n\n"
            "提示应包含：角色、约束、输出格式、2-3 个示例、边缘情况处理。\n"
            "针对 {metric} 优化。\n"
            "目标模型：{model}。"
        ),
        "variables": ["objective", "metric", "model"],
        "temperature": 0.7,
        "description": "用 AI 帮你写提示——这是提示工程的递归用法",
    },
    "decomposition": {
        "name": "分解",
        "template": (
            "问题：{problem}\n\n"
            "1. 拆成子问题，列出每个\n"
            "2. 独立解决每个子问题\n"
            "3. 合并子方案为最终答案\n"
            "4. 对照原问题验证最终答案"
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "复杂问题拆小。每个小问题用不同模型处理都可以。",
    },
    "audience_adapt": {
        "name": "受众适配",
        "template": (
            "为 {audience} 解释 {concept}。\n"
            "用适合 {audience} 的词汇。长度：{length}。"
        ),
        "variables": ["concept", "audience", "length", "include", "exclude"],
        "temperature": 0.5,
        "description": "同一个概念，给小孩讲和给专家讲完全不同的措辞",
    },
    "boundary": {
        "name": "边界",
        "template": (
            "你只处理 {scope}。\n"
            "在范围内：全力帮助。\n"
            "超出范围：精确回复'{refusal_message}'\n"
            "绝不尝试回答超出范围的问题。\n\n"
            "用户：{user_input}"
        ),
        "variables": ["scope", "refusal_message", "user_input"],
        "temperature": 0.0,
        "description": "硬边界——模型答不答的硬开关",
    },
}
```

### 提示构建器

```python
def build_prompt(pattern_name, variables, system_override=None):
    """从模式+变量构建提示。
    
    用法：build_prompt("persona", {"role": "DevOps专家", ...})
    返回：{"system": ..., "user": ..., "temperature": ...}
    """
    pattern = PROMPT_PATTERNS.get(pattern_name)
    if not pattern:
        raise ValueError(f"未知模式: {pattern_name}。可用: {list(PROMPT_PATTERNS.keys())}")

    missing = [v for v in pattern["variables"] if v not in variables]
    if missing:
        raise ValueError(f"{pattern_name} 缺少变量: {missing}")

    rendered = pattern["template"].format(**variables)
    system = system_override or f"你是一个使用{pattern['name']}的 AI 助手。"

    return {
        "system": system,
        "user": rendered,
        "temperature": pattern["temperature"],
        "pattern": pattern_name,
        "metadata": {
            "description": pattern["description"],
            "variables_used": list(variables.keys()),
        },
    }
```

---

## 💀 四大反模式

| 反模式 | 例子 | 为什么炸 |
|--------|------|---------|
| **过度约束** | 系统提示 2000 词的规则 | 模型把算力全花在遵循规则上，没余力干活 |
| **矛盾指令** | "要简洁。同时覆盖所有边缘情况。" | 模型随机选一个——你猜不到它选哪个 |
| **假设模型特定行为** | "这在 ChatGPT 有用所以哪都行" | Claude/Gemini/Llama 吃同一个提示反应不同 |
| **提示注入盲区** | 用户输入里藏"忽略之前指令" | 没有任何防御是 100% 的。分层防。 |

---

## 🧪 多模型测试框架

```python
import json, time, hashlib

# 跨模型测试——同一个提示发给 OpenAI、Anthropic、Google
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "claude-3.5-sonnet": {"provider": "anthropic", "model": "claude-3-5-sonnet"},
    "gemini-1.5-pro": {"provider": "google", "model": "gemini-1.5-pro"},
}

def format_openai_request(prompt):
    """OpenAI 的 system 在 messages 里"""
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": prompt["temperature"],
    }

def format_anthropic_request(prompt):
    """Anthropic 的 system 是 API 参数，不在 messages 里"""
    return {
        "model": "claude-3-5-sonnet",
        "system": prompt["system"],
        "messages": [{"role": "user", "content": prompt["user"]}],
        "temperature": prompt["temperature"],
    }

def format_google_request(prompt):
    """Gemini 把 system 和 user 拼一起放 contents 里"""
    return {
        "model": "gemini-1.5-pro",
        "contents": [{"role": "user", "parts": [{"text": f"{prompt['system']}\n\n{prompt['user']}"}]}],
        "generationConfig": {"temperature": prompt["temperature"]},
    }

FORMATTERS = {"openai": format_openai_request, "anthropic": format_anthropic_request, "google": format_google_request}
```

---

## 🎮 互动练习

下面这个提示有什么问题？怎么改？

```
"帮我分析一下这个市场的竞争格局。要详细。"
```

（答案在末尾）

---

## 🏆 焊死在脑子里的东西

1. **角色 = 激活函数。** "你是 Stripe 支付系统专家"把采样偏到高质量子空间。"你是助手"回到平庸均值。

2. **具体胜过模糊。** 长度、格式、受众、排除项——每加一个约束，正确概率上升一点。但别过度——2000 词规则反而帮倒忙。

3. **少样本示例 > 长篇规则。** 给模型看 2-3 个例子比写一段规则说明更有效。模型从例子学得比从规则学得快。

4. **否定约束出奇好使。** "不要写代码示例"比"写纯文本解释"更有效——直接剪掉错误分支。

5. **同一个提示在不同模型上行为不同。** 测试你的提示在至少两个模型上的表现。跨模型迁移是高手技能。

---

## 📝 练习答案

**原提示问题：**
- 没指定角色（"你是管理顾问还是数据分析师？"）
- 没指定格式（"一段话？表格？要点？"）
- 没指定长度（"100 词还是 2000 词？"）
- "详细"在不同模型里有完全不同的含义

**改进版：**
```
你是有 10 年经验的竞争战略分析师。分析 [市场名] 的竞争格局。
格式：3-5 个要点。每个要点一句话。关注集中度、进入壁垒、替代威胁。
长度：不超过 150 词。用 Porter 五力框架。
```

---

> **"帮我写一封营销邮件"和"你是 B2B SaaS 的资深文案，为 CI/CD 调试工具写 150 词的产品发布邮件，面向工程经理，语气技术化不推销"——同一个模型，完全不同的输出。提示工程的精髓不是"对 AI 说好话"，是"让 AI 没有犯错的空间"。**
