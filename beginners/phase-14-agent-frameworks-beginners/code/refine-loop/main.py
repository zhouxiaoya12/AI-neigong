"""
📝 精炼循环 · 初学者版

AI 写的东西，让 AI 自己检查自己 → 这是最容易犯的错。
就像一个作者很难发现自己的错别字。

正确的做法：两个 AI，一个写，一个审。
  写的人只管写。
  审的人只管挑毛病。

一轮一轮改，改到合格为止——这就是"精炼循环"。
"""


class RefineLoop:
    """精炼循环：写 → 审 → 改 → 再审 → ... → 通过"""

    def __init__(self, max_rounds=3, pass_score=7):
        self.max_rounds = max_rounds  # 最多改几轮
        self.pass_score = pass_score   # 几分算通过

    def run(self, writer, reviewer, task):
        """执行精炼循环"""
        history = []

        # 第一稿
        content = writer.write(task)
        history.append(("第一稿", content, None))

        for round_num in range(1, self.max_rounds + 1):
            # 评审打分
            score, issues, suggestions = reviewer.review(content)
            history.append((f"评审{round_num}", content, score))

            if score >= self.pass_score:
                return {
                    "result": content,
                    "score": score,
                    "rounds": round_num,
                    "passed": True,
                    "history": history,
                }

            # 根据意见修改
            feedback = f"问题：{'; '.join(issues)}。建议：{'; '.join(suggestions)}"
            content = writer.revise(feedback)
            history.append((f"修改稿{round_num}", content, None))

        return {
            "result": content,
            "score": score,
            "rounds": self.max_rounds,
            "passed": False,
            "history": history,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示：让 AI 写 API 文档
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WriterAgent:
    """写文档的 Agent"""
    def __init__(self, drafts):
        self.drafts = drafts  # 预设的草稿（模拟 AI 输出）
        self.pos = 0

    def write(self, task):
        d = self.drafts[self.pos]
        self.pos += 1
        return d

    def revise(self, feedback):
        d = self.drafts[self.pos]
        self.pos += 1
        return d


class ReviewerAgent:
    """审文档的 Agent"""
    def __init__(self, reviews):
        self.reviews = reviews  # 预设的评审结果
        self.pos = 0

    def review(self, content):
        r = self.reviews[self.pos]
        self.pos += 1
        return r["score"], r.get("issues", []), r.get("suggestions", [])


if __name__ == "__main__":
    print("=" * 55)
    print("📝 精炼循环 · 初学者版")
    print("=" * 55)

    # 写手的草稿：第一版很简陋，第二版改好了
    writer = WriterAgent([
        "GET /users 返回用户列表。无参数。",
        "GET /users 返回用户列表。参数：page(页码), limit(每页条数)。错误：401未认证。",
    ])

    # 评审者的标准
    reviewer = ReviewerAgent([
        {"score": 4, "issues": ["缺少参数说明", "没有错误码"], "suggestions": ["添加 page/limit", "列出 401"]},
        {"score": 8, "issues": [], "suggestions": []},
    ])

    loop = RefineLoop(max_rounds=3, pass_score=7)
    result = loop.run(writer, reviewer, "写 GET /users 接口文档")

    print(f"\n任务：写 GET /users 接口文档\n")

    for stage, content, score in result["history"]:
        if "第一稿" in stage or "修改稿" in stage:
            print(f"  [{stage}] {content}")
        else:
            print(f"  [{stage}] 评分: {score}/10 {'✅ 通过' if score >= 7 else '❌ 需修改'}")

    print(f"\n🏁 最终：评分 {result['score']}/10, 精炼 {result['rounds']} 轮, "
          f"{'✅ 通过' if result['passed'] else '❌ 未通过'}")

    print(f"\n💡 为什么不让同一个 AI 自己评自己？")
    print(f"   就像你不该自己给自己的作文打分——你会漏掉自己的错误。")
