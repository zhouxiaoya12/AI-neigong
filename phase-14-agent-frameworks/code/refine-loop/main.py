"""
精炼循环 · 别自评，让另一个 Agent 评（纯标准库）

Agent 写出东西后让它自己检查自己——这是最容易踩的坑。
一个写代码的 LLM 很难同时做代码审查：它会漏掉自己的错误、
忽略边界情况、对"看起来能跑"过度自信。

解决方案：两个角色，一道墙。
  - 构建者（Builder）  —— 负责写
  - 评审者（Reviewer）  —— 负责评

两轮循环：
  第1轮：构建者写初稿 → 评审者审阅 → 返回修改意见
  第2轮：构建者根据修改意见改 → 评审者再审 → 通过 or 继续改

关键设计决策：评审者不修改内容，只给出评审意见。
就像代码审查——审查者说"这里有问题"，但不帮你改代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模拟的 LLM 角色
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BuilderLLM:
    """构建者 LLM —— 负责生成内容。

    在真实场景中，这是一个独立的 LLM 调用，带专门的 system prompt：
    "你是一个专业的内容创作者。根据用户需求生成高质量输出。"
    """

    def __init__(self, drafts: list[str]) -> None:
        self.drafts = drafts
        self.cursor = 0

    def generate(self, task: str, feedback: str = "") -> str:
        """生成内容。如果有评审反馈，在下一次调用中改进。"""
        if self.cursor >= len(self.drafts):
            return "无法生成更多草稿"
        draft = self.drafts[self.cursor]
        self.cursor += 1
        return draft


class ReviewerLLM:
    """评审者 LLM —— 负责评估构建者的输出。

    在真实场景中，这是一个独立的 LLM 调用，带专门的 system prompt：
    "你是一个严格的评审者。检查以下内容的质量、准确性和完整性。
     不要修改内容，只给出具体的改进建议。"

    关键：评审者和构建者使用不同的 LLM 实例 or 不同的 system prompt。
    这是"那道墙"的意义——如果是同一个 LLM 用同一个 prompt，
    评审就沦为了走过场。
    """

    def __init__(self, reviews: list[dict[str, Any]]) -> None:
        self.reviews = reviews
        self.cursor = 0

    def review(self, content: str) -> dict[str, Any]:
        """评审一段内容，返回评审结果。

        返回格式：
          {
            "approved": bool,        # 是否通过
            "score": float,          # 0-10 分
            "issues": [str, ...],    # 发现的问题
            "suggestions": [str, ...]  # 改进建议
          }
        """
        if self.cursor >= len(self.reviews):
            return {"approved": True, "score": 10, "issues": [], "suggestions": []}
        review = self.reviews[self.cursor]
        self.cursor += 1
        return review


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 精炼循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RefineLoop:
    """精炼循环：构建 → 评审 → 修改 → 再评审 → ... → 通过 or 达到上限"""

    builder: BuilderLLM
    reviewer: ReviewerLLM
    max_iterations: int = 3            # 最多精炼几次
    quality_threshold: float = 7.0     # 评分低于这个数才需要修改
    history: list[dict[str, Any]] = field(default_factory=list)

    def run(self, task: str) -> dict[str, Any]:
        """执行精炼循环。

        返回：
          {
            "task": 原始任务,
            "final_content": 最终内容,
            "final_score": 最终评分,
            "iterations": 精炼次数,
            "history": 完整历史记录
          }
        """
        # 第1步：构建者写初稿
        content = self.builder.generate(task)
        self._log("第一稿", content, None)

        for iteration in range(1, self.max_iterations + 1):
            # 第2步：评审者审阅
            review = self.reviewer.review(content)
            score = review.get("score", 0)
            issues = review.get("issues", [])
            suggestions = review.get("suggestions", [])

            self._log(f"评审 #{iteration}", content, review)

            # 第3步：判断是否通过
            if review.get("approved", False) or score >= self.quality_threshold:
                return {
                    "task": task,
                    "final_content": content,
                    "final_score": score,
                    "iterations": iteration,
                    "passed": True,
                    "history": self.history,
                }

            # 第4步：构建者根据反馈修改
            feedback = f"问题：{'; '.join(issues)}。建议：{'; '.join(suggestions)}"
            content = self.builder.generate(task, feedback)
            self._log(f"修改稿 #{iteration}", content, None)

        # 达到最大迭代次数
        final_review = self.reviewer.review(content)
        return {
            "task": task,
            "final_content": content,
            "final_score": final_review.get("score", 0),
            "iterations": self.max_iterations,
            "passed": final_review.get("approved", False),
            "warning": "达到最大迭代次数",
            "history": self.history,
        }

    def _log(self, stage: str, content: str, review: dict[str, Any] | None) -> None:
        self.history.append({
            "stage": stage,
            "content": content,
            "review": review,
        })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示：三个场景
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("🔁 精炼循环 · 构建-评审分离")
    print("=" * 65)

    # ── 场景 1：正常精炼（两轮过） ──
    print("\n📝 场景 1：API 文档写作 · 正常精炼")
    print("─" * 45)

    # 构建者的草稿序列：第一稿有问题，第二稿改好了
    builder = BuilderLLM([
        "GET /users 返回用户列表。参数：无。响应：JSON数组。",
        "GET /users 返回用户列表。查询参数：page(页码, int), limit(每页数量, int, 默认20)。"
        "响应：{ data: [...], total: int, page: int }。401 未认证返回 { error: string }。",
    ])

    # 评审者的评审序列：第一次打低分，第二次通过
    reviewer = ReviewerLLM([
        {
            "approved": False,
            "score": 5.0,
            "issues": ["缺少查询参数说明", "没有提到分页", "错误码未列出"],
            "suggestions": ["添加 page/limit 参数", "补充分页响应格式", "列出 401 错误码"],
        },
        {
            "approved": True,
            "score": 9.0,
            "issues": [],
            "suggestions": ["可考虑添加 rate limiting 说明"],
        },
    ])

    loop = RefineLoop(builder, reviewer, max_iterations=3, quality_threshold=7.0)
    result = loop.run("写 GET /users 接口的文档，要包含分页和错误码。")

    print(f"\n  任务：{result['task']}")
    for entry in result["history"]:
        print(f"\n  [{entry['stage']}]")
        preview = entry["content"][:80]
        print(f"    内容：{preview}...")
        if entry["review"]:
            r = entry["review"]
            print(f"    评分：{r['score']}/10  {'✅ 通过' if r['approved'] else '❌ 需修改'}")
            if r["issues"]:
                print(f"    问题：{'; '.join(r['issues'])}")

    print(f"\n  🏁 最终结果：评分 {result['final_score']}/10, "
          f"精炼 {result['iterations']} 次, {'✅ 通过' if result['passed'] else '❌ 未通过'}")

    # ── 场景 2：一次过（完美初稿） ──
    print("\n" + "=" * 65)
    print("📝 场景 2：简单任务 · 一次通过")
    print("─" * 45)

    builder2 = BuilderLLM(["今天是星期一。今年是2026年。"])
    reviewer2 = ReviewerLLM([
        {"approved": True, "score": 9.5, "issues": [], "suggestions": []},
    ])

    loop2 = RefineLoop(builder2, reviewer2, max_iterations=3, quality_threshold=7.0)
    result2 = loop2.run("今天是星期几？今年是哪一年？")

    print(f"  评分：{result2['final_score']}/10, 精炼次数：{result2['iterations']}")
    print(f"  答案：{result2['final_content']}")

    # ── 场景 3：始终不过 ──
    print("\n" + "=" * 65)
    print("📝 场景 3：复杂任务 · 达到迭代上限")
    print("─" * 45)

    builder3 = BuilderLLM([
        "缺陷：登录页面存在XSS漏洞。复现步骤：在用户名输入框输入 <script>alert(1)</script>。",
        "缺陷：登录页面存在存储型XSS漏洞。复现步骤：输入 <img src=x onerror=alert(1)>。"
        "影响范围：所有用户。修复建议：使用 DOMPurify 对输出做 HTML 转义。",
        "缺陷：登录页面用户名输入框存在存储型XSS漏洞。复现步骤：输入特殊HTML字符。"
        "影响范围：所有登录用户。修复方案：前端用 DOMPurify，后端用 OWASP 编码器。"
        "CVSS评分：6.1 (Medium)。",
    ])

    reviewer3 = ReviewerLLM([
        {
            "approved": False, "score": 4.0,
            "issues": ["XSS类型未明确（反射/存储/DOM）", "缺少影响范围", "没有修复建议"],
            "suggestions": ["明确XSS类型", "评估影响范围", "提供具体修复代码"],
        },
        {
            "approved": False, "score": 5.5,
            "issues": ["修复方案不够具体", "缺少CVSS评分", "复现步骤不完整"],
            "suggestions": ["给出DOMPurify具体配置", "计算CVSS分数", "补全复现环境信息"],
        },
        {
            "approved": False, "score": 6.5,
            "issues": ["CVSS向量未列出", "缺少修复后验证方法"],
            "suggestions": ["补充CVSS完整向量", "添加如何验证修复的说明"],
        },
    ])

    loop3 = RefineLoop(builder3, reviewer3, max_iterations=2, quality_threshold=7.0)
    result3 = loop3.run("写一份登录页面XSS漏洞的安全报告")

    print(f"  最终评分：{result3['final_score']}/10")
    print(f"  迭代次数：{result3['iterations']}（上限）")
    print(f"  状态：{'✅ 通过' if result3['passed'] else '⚠️ 达到迭代上限，未达标'}")
    print(f"  最终版本：{result3['final_content'][:80]}...")

    # ── 总结 ──
    print(f"\n💡 精炼循环的设计原则：")
    print(f"  • 构建和评审必须分离——同一角色做两件事 = 无效循环")
    print(f"  • 评审者不帮改，只给意见（跟代码审查一样）")
    print(f"  • 迭代次数有上限——有些任务再精炼也过不了，及时止损")
    print(f"  • 真实场景中，构建者和评审者使用不同的 system prompt")
    print(f"  • 甚至可以针对不同维度设不同评审者（安全评审、风格评审、正确性评审）")
    print(f"  • 比\"自评\"更有效的替代方案：多Agent辩论、验证门控、人工评审")


if __name__ == "__main__":
    main()
