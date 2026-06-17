# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI 内功（ai-neigong）是一本体系化的 AI Agent 工程中文教程，从 Tokenizer 到 Agent Loop，从 RAG 到多 Agent 编排，覆盖 LLM 工程的完整技术栈。

## 仓库结构

```
.
├── phase-XX-*/                    # 主线教程模块（按阶段编号）
│   ├── *.md                       # 教程文章
│   ├── code/                      # 可运行的代码示例
│   └── exam.json                  # 模块考试题
├── beginners/                     # 初学者友好版（平行结构）
├── architect-supplement-translated/  # 架构师深度补充
├── fde-architect-plan/            # FDE 架构师路线规划
├── learning-plans/                # 学习路线与知识索引
├── llm-learning-plan-translated/  # LLM 工程师路线
└── monitor-reports/               # 内部监控报告（不对外）
```

## 开发命令

### 运行代码示例
```bash
# 运行单个模块的代码示例
python3 phase-14-agent-frameworks/code/01-agent-loop/main.py

# 运行初学者版本
python3 beginners/phase-14-agent-frameworks-beginners/code/agent-loop/main.py
```

### 验证代码可运行性
```bash
# 检查 Python 文件语法
python3 -m py_compile <file.py>

# 批量检查某个模块的所有 Python 文件
find phase-14-agent-frameworks/code -name "*.py" -exec python3 -m py_compile {} \;
```

## 写作规范

- **语言**：正文用中文，代码注释用中文，变量名/函数名保留英文
- **深度优先**：一个概念往下挖三层，不蜻蜓点水
- **代码必可跑**：每个代码示例都能 `python3 <file>.py` 零报错运行
- **工程直觉优先**：不讲"AI 是什么"，讲"AI 怎么工程化"
- **格式**：技术深度 + 翻车案例 + 可运行代码

## 考试题格式

考试题存储在各模块的 `exam.json` 文件中，格式为 JSON 数组：
```json
{
  "type": "选择|填空|简答|代码|场景",
  "difficulty": "简单|中等|困难",
  "question": "题目内容",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},  // 选择题
  "answer": "答案",
  "explanation": "解析",
  "topic": "知识点分类"
}
```

## 三条学习路线

| 路线 | 目录 | 预计时长 |
|------|------|----------|
| LLM 工程师 | `llm-learning-plan-translated/` | ~43h |
| AI Agent 工程师 | 主线 `phase-*` 模块 | ~80h |
| FDE 架构师 | `fde-architect-plan/` + `architect-supplement-translated/` | ~120h |

## 注意事项

- `monitor-reports/` 是内部监控报告，已在 `.gitignore` 中排除，不应提交
- 初学者版本在 `beginners/` 目录，与主线内容平行但更平缓
- 术语翻译遵循"中文优先，英文括号标注"的原则
