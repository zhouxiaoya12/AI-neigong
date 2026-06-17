# 技能与智能体 SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说"存在哪些工具"。技能说"如何完成一个任务"。2026 年的技术栈让两者分层存在。Anthropic 的 Agent Skills（2025 年 12 月发布，开放标准）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上 Widget 元数据。AGENTS.md（现已被 60,000+ 仓库采纳）位于仓库根目录，作为项目级智能体上下文。本课命名各部分覆盖的范围，并构建一个可在各智能体间携带的最小化 SKILL.md + AGENTS.md 组合。

**类型：** 学习
**语言：** Python（标准库、SKILL.md 解析器和加载器）
**前置课程：** Phase 13 · 07（MCP 服务器）
**用时：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带有 YAML 前置元数据和渐进式披露的 SKILL.md。
- 以文件系统方式将技能加载到智能体运行时中。
- 组合一个技能与一个 MCP 服务器和一个 AGENTS.md，使得一个包可以在 Claude Code、Cursor 和 Codex 中工作。

## 问题

一位工程师将一个发布说明撰写工作流提炼为一个多步骤提示词："读取最近合并的 PR。按领域分组。逐个总结。按照团队风格写 changelog 条目。发布到 Slack 草稿频道。"他们将这个流程放在团队的 Notion 文档中。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用这个工作流。每个智能体有自己加载指令的方式：Claude Code 的斜杠命令、Cursor 的 rules、Codex 的 `.codex.md`。工程师复制了三次工作流，维护三个副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的智能体会话启动时读取它。"这个项目如何运作？有哪些约定？哪些命令运行测试？"
- **SKILL.md** 是一个可携带的包：YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源文件。支持技能的智能体按需按名称加载。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三个层次，一个可携带的产物。

## 概念

### AGENTS.md（agents.md）

于 2025 年底推出，截至 2026 年 4 月已被 60,000+ 仓库采纳。一个位于仓库根目录的文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript，strict 模式。
- Python 端使用 Pydantic 建模。
- 使用 `pnpm test` 运行测试。

## Build and run
- `pnpm dev` 启动本地开发服务器。
- `pnpm build` 打包生产构建。
```

智能体在会话启动时读取此文件，并用它来校准自己在该项目中的行为。2026 年每个编程智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: 按照本项目风格为最近合并的 PR 撰写 changelog 条目。
---

# 发布说明撰写器

被调用时，执行以下步骤：

1. 列出自上一个 tag 以来合并的 PR。使用 `gh pr list --base main --state merged`。
2. 按标签分组：feature、fix、chore、docs。
3. 对每个分组的每个 PR，写一行：`- <title> (#<num>)`。
4. 草拟发布说明并暂存到 CHANGELOG.md 中。

如果用户说 "ship"，运行 `git tag vX.Y.Z` 和 `gh release create`。

## 注意事项

- 绝不包含没有 PR 的提交。
- 在公开 changelog 中跳过 "chore" 条目。
```

前置元数据声明技能的身份。正文是技能加载时展示给模型的提示词。

### 渐进式披露

技能可以引用子资源，智能体仅在需要时才获取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中说"查看 style-guide.md 了解风格规则。"智能体仅在技能活跃运行时才拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示词。

### 文件系统发现

智能体运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名称和前置元数据的 `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨智能体工具）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载技能，将其暴露为运行时中可调用的"智能体"。智能体循环在用户调用技能时分派到该技能。

### OpenAI Apps SDK

于 2025 年 10 月发布；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个开发者平台下。一个 Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示词）。
- 加上 ChatGPT UI 的 Widget 元数据。
- 加上可选的 MCP Apps `ui://` 资源用于交互式界面。

相同协议，更丰富的用户体验。

### 通过 SkillKit 实现跨智能体可携带性

SkillKit 及类似的跨智能体分发层将单个 SKILL.md 翻译为 32+ 个 AI 智能体的原生格式（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。一个事实来源；多个消费者。

### 三层技术栈

| 层次 | 文件 | 何时加载 | 目的 |
|------|------|---------|------|
| AGENTS.md | 仓库根目录 | 会话启动时 | 项目级约定 |
| SKILL.md | skills 目录 | 技能被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

全部三层组合使用：智能体会话启动时读取 AGENTS.md，用户调用一个技能，技能的指令中包含 MCP 工具调用，智能体通过 MCP 客户端分派。

## 实战

`code/main.py` 提供一个基于标准库的 SKILL.md 解析器和加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据和 Markdown 正文，生成一个以技能名为键的字典。然后模拟一个智能体循环，按名称调用 `release-notes-writer`。

需要关注的内容：

- YAML 前置元数据使用最小的标准库解析器（无需 `pyyaml` 依赖）。
- 技能正文按原样存储；智能体在调用时将其前置到系统提示词。
- 通过 `read_subresource` 函数演示渐进式披露，按需拉取引用的文件。

## 交付

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能产出组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图，可在各智能体间携带。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，确认加载器能发现它。

2. 为本课程仓库编写一个 AGENTS.md。包含测试命令、风格约定和 Phase 13 的心智模型。

3. 将团队内部文档中的一个多步骤工作流移植为 SKILL.md。验证其在 Claude Code 中的加载。

4. 手动将该技能翻译为 Cursor 和 Codex 的原生规则格式。统计格式之间的差异——这就是 SkillKit 自动化处理的翻译面。

5. 阅读 Anthropic Agent Skills 博文。找出 Claude Agent SDK 中本课加载器未覆盖的一个功能。（提示：智能体子调用。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SKILL.md | "技能文件" | YAML 前置元数据加 Markdown 正文，由智能体运行时加载 |
| AGENTS.md | "仓库根智能体上下文" | 会话启动时读取的项目级约定文件 |
| 渐进式披露 | "懒加载子资源" | 技能正文引用仅在需要时才拉取的文件 |
| 前置元数据 | "顶部的 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + Widget 元数据" | OpenAI 基于 MCP 加 ChatGPT UI 钩子的开发者平台 |
| 技能发现 | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称索引 |
| 跨智能体可携带性 | "一个技能，多个智能体" | 通过 SkillKit 类工具将一个 SKILL.md 翻译为 32+ 智能体格式 |
| Agent Skill | "可携带的知识" | MCP 工具概念之外的、可复用的任务模板 |
| Apps SDK | "MCP 加上 ChatGPT UI" | Connectors 和 Custom GPTs 在 MCP 上统一 |

## 延伸阅读

- [Anthropic — Agent Skills 发布公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采纳列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例

---

## 📝 教师备课总结与读后感

### 一、文档整体评价

这篇课程精准地捕捉了 2026 年 AI Agent 生态中最迫切的互操作性问题——如何让一个工作流在 Claude Code、Cursor、Codex 等不同智能体间"写一次，到处运行"。通过三层分层（AGENTS.md / SKILL.md / MCP）的清晰架构，文档为"技能可携带性"提供了完整的解决方案框架。开篇的工程师痛点故事（"复制三次，维护三份"）极具共鸣，有效地建立了学习动机。

### 二、知识结构梳理

- **约定层（AGENTS.md）：** 项目级的"这是怎么工作的"——约定、构建命令、测试命令。它是智能体进入项目时的上下文初始化。60,000+ 仓库的采纳量证明了其作为事实标准的地位。
- **知识层（SKILL.md）：** 可复用的"怎么做这个任务"——YAML 前置元数据声明身份，Markdown 正文承载指令，渐进式披露的子资源避免提示词膨胀。它是跨智能体可携带性的核心载体。
- **能力层（MCP）：** 工具的实际调用接口。SKILL.md 描述流程，MCP 提供执行能力——两者分工明确、组合使用。

### 三、核心洞察

1. **"写一次到处运行"是 Agent 时代的 DRY 原则：** 工程师不应该为每个智能体重写同样的工作流——这个洞察简单但深刻。SKILL.md + AGENTS.md 的标准化为"技能复用"提供了基础设施。
2. **渐进式披露是提示词工程的工程化：** 不只是"把指令写短点"，而是"有一个主文件声明能力，子文件按需加载"——这是从 prompt 到 prompt-architecture 的进化。
3. **文件系统即注册表：** 技能发现用目录扫描而非数据库查询——`~/.anthropic/skills/*/SKILL.md` 的设计简洁到极致。没有 API、没有认证、没有服务器。这大大降低了采纳门槛。
4. **OpenAI Apps SDK 的"MCP + UI Metadata"模式：** 这不是另起炉灶，而是在 MCP 上加了一层 UI 元数据。这意味着一个 A2A/MCP 兼容的智能体可以直接在 ChatGPT 中获得原生 UI——协议的层叠设计很有远见。
5. **SkillKit 的 32+ 智能体翻译层：** "一个事实来源，多个消费者"的理念用在此处恰到好处。翻译层的存在不是因为标准不够好，而是因为生态系统需要时间来统一。
6. **前置元数据的 YAML 设计讲究：** 不用 JSON、不用 TOML——用 YAML 是因为 Markdown 文件天然包含 `---` 分隔的 YAML frontmatter，这在 Jekyll/Hugo/Obsidian 等工具中已有广泛先例。
7. **AGENTS.md 的"校准"语义：** 它不是给模型的"规则"（那是 system prompt 的事），而是"上下文初始化"——告诉智能体它在哪里，而不是告诉它该怎么做。这个微妙区别决定了文件的语气和内容密度。

### 四、教学建议

1. **痛点共鸣先行：** 让学生先尝试在没有 AGENTS.md/SKILL.md 的情况下，让 Claude Code 和 Cursor 执行同一个任务。观察行为差异——然后再引入 AGENTS.md 和 SKILL.md，对比前后体验。
2. **动手编写 AGENTS.md：** 这是最直接的练习。让学生为自己当前的项目写一个 AGENTS.md，然后让 Claude Code 读取它，观察行为变化。好的 AGENTS.md 带来的改善是立即可感知的。
3. **技能组合练习：** 给学生三个 MCP 服务器和一个复杂工作流，要求他们编写一个 SKILL.md 来编排这些服务器。这迫使他们思考"技能该多细化"和"什么该进 SKILL.md 什么该进 AGENTS.md"。
4. **渐进式披露的边界讨论：** 什么应该作为子资源？什么应该直接放进 SKILL.md 正文？让学生设计一个指南——正文中只放"必须现在就知道"的信息，其余一律通过引用分发。
5. **格式翻译练习：** 练习 4 的手动翻译非常有价值——学生亲身体验格式差异后，会真正理解 SkillKit 的价值，也能识别哪些差异是表面的（语法）哪些是深层的（能力模型）。
6. **OpenAI vs Anthropic 的差异化对比：** Apps SDK 和 Agent Skills 虽然都是 MCP 上的层次，但设计哲学不同——Apps SDK 偏 UI/用户体验集成，Skills 偏可复用知识管理。让学生讨论两种哲学的适用场景。
7. **三层架构的设计决策：** 让学生设计一个新的"层"应该加在哪里（如安全策略层、计费层），并论证为什么应该是一个新层而不是扩展现有层。这培养架构判断力。

### 五、值得补充的内容

1. **AGENTS.md 的最佳实践模式：** 可以补充一个经过实战验证的 AGENTS.md 模板（目录结构说明、测试/构建/部署命令、代码规范链接、常用路径约定），让学生有模仿的基础。
2. **SKILL.md 的版本管理和分发：** 讨论技能应该如何版本化（通过 Git tag？SemVer？），以及如何通过 registry 或 npm 类工具分发技能包。
3. **多技能冲突解决：** 当两个技能声明了相同的名称或需要同一个 MCP 工具时，如何解决冲突？补充一个冲突解决策略的讨论。
4. **SkillKit 的翻译质量和边界：** 补充一个实际测试——同一个技能在 5 个不同智能体上的执行结果对比，展示翻译质量差异和哪些场景会出问题。
5. **AGENTS.md 与 System Prompt 的关系：** 补充讨论 AGENTS.md 的内容如何与各智能体的原生系统提示词交互——是追加、替换、还是合并？不同智能体的实际行为可能不同。

### 六、一句话总结

AGENTS.md + SKILL.md + MCP 三层架构为 AI Agent 时代的"写一次，到处运行"提供了基础设施——前者告诉智能体"你在哪里"，后者告诉它"怎么做"，而渐进式披露和文件系统发现这两个设计决策，用极简的机制解决了提示词膨胀和分发摩擦两个核心问题。



---

# 🎓 Agent 架构课：Skills & AGENTS.md——别让你的工作流在五个智能体上写五遍

同学们好。我是你们FDE工程老师，今天讲的是 Agent 技能。

**你有一个"读取 PR → 分组 → 写 changelog → 发 Slack"的工作流。你想在 Claude Code、Cursor、Codex 三个智能体上用。你怎么做？**

大部分人的做法是把这段工作流复制三遍——一份放 CLAUDE.md、一份放 .cursorrules、一份放 .codex.md。三个月后，三个文件的内容已经不 sync 了——一个工程师在 Claude Code 那个版本里加了一个"跑 e2e 测试"的步骤，另外两个版本永远不知道。

这就是 AGENTS.md 和 SKILL.md 要解决的问题。三个层次：

- **AGENTS.md** 放仓库根目录。写"这个项目怎么构建""测试怎么跑""代码风格是什么"——项目级上下文，每个智能体会话启动时读。60,000+ 仓库已经在用。
- **SKILL.md** 是便携包。YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源文件。按需加载。一个技能文件，三个智能体都能用。
- **MCP** 管工具。技能说"怎么做"，MCP 说"用什么工具做"。

## 渐进式披露：不是把所有东西都塞进上下文

SKILL.md 的核心设计哲学：分三层加载。第一层只有名称和一句话描述——模型看到 100 个技能时每个只占一行 token。第二层是正文——模型决定用这个技能后才加载完整指令。第三层是资源文件——脚本、模板、参考文档，只在执行时才读。

## 结课清单

1. **AGENTS.md = 项目级上下文。** 每个智能体启动时读。放根目录。
2. **SKILL.md = 可复用工作流。** YAML 元数据 + Markdown + 资源文件。一次编写，处处运行。
3. **MCP = 工具。** 技能描述"怎么做"，MCP 提供"拿什么做"。
4. **渐进式披露。** 分三层加载——不让模型看到不需要的信息。

最后一句话：**你的工作流在两个智能体上复制了两遍，就意味着你已经在维护两份会逐渐分叉的真相。SKILL.md 让真相只有一份。**

---

# 💼 从业者故事：Skills & AGENTS.md——别让你的工作流在五个智能体上写五遍

三个月前我 review 一个团队的 codebase，发现了一个"规则地狱"。同一个发布流程——"读取最近合并的 PR，分组，写 changelog，发 Slack"——出现在三个地方：Claude Code 的 CLAUDE.md、Cursor 的 .cursorrules、Codex 的 .codex.md。三个文件的格式不同、语法不同、连缩进风格都不一样。更魔幻的是，三个文件的内容已经不 sync 了——Claude Code 那个文件里多了一个"发布前跑 e2e 测试"的步骤，因为那个维护 Claude Code 配件的工程师私下加进去的。Cursor 的用户永远不知道有这个步骤。

这不是技术问题，这是"知识腐烂"。当一个流程需要以 N 种格式维护，N-1 个副本必然过期。你在 DRY 原则上花了四年大学教育，然后一出社会就在干复制粘贴的活，想想就憋屈。

AGENTS.md + SKILL.md 的解决方案，本质上就是给这个烂摊子立规矩。AGENTS.md 放在仓库根目录，相当于你项目的"前台"——"这个项目是 TypeScript strict 模式的，测试用 pnpm test 跑，构建用 pnpm build"。每个兼容的智能体启动时自动读取，不需要你手动配置任何东西。SKILL.md 是可携带的"工作流文件"，放在 skills/ 目录下——YAML 前置元数据写"我是谁、我干什么"，Markdown 正文写"怎么做"。一个 SKILL.md 通过 SkillKit 翻译成 32+ 个智能体的原生格式。

这其实是一个非常简单的设计——文件系统就是注册表，目录扫描就是发现机制。没有中心服务器、没有 API 网关、没有认证鉴权。你把这个方案讲给一个 2005 年的 Linux 运维听，他会说"这不就是文件约定吗，我们用了 20 年了"。对，就是文件约定。但 Agent 时代很多人忘了文件约定的力量——非要搞一个数据库、一个注册中心、一个 npm registry for skills。过度工程化是工程师的职业病，而 SKILL.md 的设计者显然很克制。

不过渐进式披露这个设计我得专门夸一下。按理说，一个"发布 changelog"的技能，可以引用一个 style-guide.md、一个 template.md、一个 generate.sh 脚本。如果你把所有内容都塞进 SKILL.md，这个文件能膨胀到 2000 行，然后模型在每次调用时都要读完这 2000 行——就算 99% 的内容它根本用不到。渐进式披露的做法是：SKILL.md 正文里写"查看 style-guide.md 了解风格规则"，模型只在需要时去拉取子文件。这就像你去餐厅，服务员给你的是菜单，不是后厨的菜谱全集。

但这里有个坑：什么该进 SKILL.md 正文，什么该作为子资源引用？我见过一个团队的 SKILL.md 正文只有一句话："请参考以下文件"——然后挂了 12 个引用。这叫渐进式披露吗？不，这叫把所有东西都"渐进"了。正文应该包含"模型必须立刻知道"的信息：这个技能的目标、核心步骤、关键约束。子资源给的是"模型可能需要"的信息：风格指南、模板、参考数据。一个简单的判断标准：如果模型只读正文就能完成 80% 的任务，剩下的 20% 靠子资源补充——这个比例就是健康的。如果正文什么都干不了全靠子资源，你不是在设计技能，你是在用 RAG 套壳。

再吐槽一下 OpenAI 的 Apps SDK。它的设计哲学跟 Anthropic 的 Skills 完全不同。Skills 是"知识层"——给你一段流程指令，你照着做。Apps SDK 是"体验层"——给 MCP 服务器加 Widget 元数据，让它在 ChatGPT 里有一个漂亮的 UI。两个路线不冲突，但如果你是一个只想让工作流跨智能体可携带的工程师，Skills 路线更轻、更快、更务虚。Apps SDK 更适合想做一个 ChatGPT 里可交互应用的团队。选择哪个，取决于你要解决的是"可携带性"还是"交互体验"。

60,000+ 仓库采纳 AGENTS.md 这个数字，我觉得被低估了。这不是"又一个格式标准"，这是 Agent 时代的 README——每个仓库都应该有一个，就像每个项目都应该有个 README.md 一样。事实上我预测到 2027 年，面试题会有"给你的项目写 AGENTS.md"就像现在"给你的项目写 README"一样平常。

**金句：AGENTS.md 告诉智能体"你在哪里"，SKILL.md 告诉它"怎么做"，MCP 给它"能做什么"——三者合在一起，才是 Agent 时代的可携带工程栈。缺任何一个，你的智能体都在裸奔。**
