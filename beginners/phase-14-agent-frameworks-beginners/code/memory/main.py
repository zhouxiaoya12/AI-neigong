"""
🧠 Agent 记忆 · 初学者版

AI 对话越长，上下文窗口越不够用。
解决方法：给 AI 一个"笔记本"。

两层记忆：
  工作记忆（主上下文）→ 只能记住最近几条消息，旧的消息会被"挤出去"
  长期记忆（归档存储）→ 无限容量，AI 自己决定什么时候写、什么时候翻
"""


class MemorySystem:
    """两层记忆系统"""

    def __init__(self, max_messages=3):
        self.core = {}              # 核心信息（名字、偏好等）
        self.messages = []          # 当前对话（工作记忆）
        self.max_messages = max_messages
        self.evicted = []           # 被挤出的旧消息
        self.archive = []           # 长期记忆（归档）

    # ── 工作记忆 ──

    def remember(self, section, text):
        """往核心记忆里写东西"""
        if section in self.core:
            self.core[section] += " " + text
        else:
            self.core[section] = text
        return f"已记住「{section}」"

    def chat(self, role, text):
        """添加一条对话消息"""
        self.messages.append((role, text))
        # 超出容量，挤出最旧的
        while len(self.messages) > self.max_messages:
            self.evicted.append(self.messages.pop(0))

    # ── 长期记忆 ──

    def archive_save(self, text, tags=()):
        """往长期记忆存东西"""
        record = {"id": len(self.archive) + 1, "text": text, "tags": tags}
        self.archive.append(record)
        return f"已归档 #{record['id']}（共 {len(self.archive)} 条）"

    def archive_search(self, keyword):
        """在长期记忆中搜索"""
        results = []
        for r in self.archive:
            if keyword.lower() in r["text"].lower():
                results.append(r)
        if not results:
            return "未找到相关记录"
        lines = []
        for r in results[:3]:  # 最多返回3条
            tags = f" [{', '.join(r['tags'])}]" if r['tags'] else ""
            lines.append(f"  #{r['id']}{tags}: {r['text']}")
        return "\n".join(lines)

    def recall_chat(self, keyword):
        """在被挤出的旧消息中搜索"""
        all_msgs = self.evicted + self.messages
        for role, text in reversed(all_msgs):
            if keyword.lower() in text.lower():
                return f"找到（{role}）: {text}"
        return "未在对话历史中找到"

    # ── 查看状态 ──

    def status(self):
        print(f"\n📊 记忆状态：")
        print(f"  核心信息：{self.core}")
        print(f"  当前对话：{len(self.messages)} 条（容量 {self.max_messages}）")
        print(f"  已挤出：{len(self.evicted)} 条")
        print(f"  长期归档：{len(self.archive)} 条")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 55)
    print("🧠 Agent 记忆 · 初学者版")
    print("=" * 55)

    mem = MemorySystem(max_messages=3)

    # ── 场景：阿明和AI助手的对话 ──
    print("\n📝 第1步：建立对话")
    mem.chat("阿明", "我叫阿明，在做 Agent 工程化")
    mem.chat("助手", "记下了，在做什么项目？")
    mem.chat("阿明", "在做知识库检索机器人，接了12个工具")
    mem.chat("助手", "12个工具已经很多了，要注意工具漂移")

    mem.status()

    # ── AI 主动记笔记 ──
    print("\n📝 第2步：AI 主动写入记忆")
    print(f"  {mem.remember('用户信息', '姓名=阿明，专长=Agent')}")
    print(f"  {mem.archive_save('阿明在做知识库检索机器人，12个工具，给销售团队用', tags=('项目',))}")
    print(f"  {mem.archive_save('工具链超过20步会产生漂移（BFCL V4测试数据）', tags=('技术',))}")

    # ── 继续对话，旧消息被挤出 ──
    print("\n📝 第3步：继续对话（旧消息被挤出）")
    mem.chat("阿明", "你说的工具漂移是什么？")
    mem.chat("助手", "让我查一下归档记录...")

    mem.status()

    # ── AI 翻看笔记 ──
    print("\n📝 第4步：AI 搜索记忆")
    print(f"  搜索归档「工具链漂移」:")
    print(f"  {mem.archive_search('漂移')}")
    print(f"\n  搜索对话「检索机器人」:")
    print(f"  {mem.recall_chat('检索机器人')}")

    print(f"\n💡 两层记忆的好处：")
    print(f"   工作记忆 = 你正在想的事（小但快）")
    print(f"   长期记忆 = 你的笔记本（大但需要翻）")
    print(f"   AI 自己决定什么时候记、什么时候翻。")
