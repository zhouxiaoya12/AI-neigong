"""
两层记忆架构 · MemGPT 模式实现（纯标准库）

Agent 的记忆不是"把聊天记录存下来"这么简单。
真正的两层架构：

  主上下文（Main Context）     —— 有限容量的"工作记忆"
     | 容量有限（比如只保留最近4条消息）
     | 旧消息被逐出（evicted）但不丢失
     |
  归档存储（Archival Store）   —— 无限容量的"长期记忆"
     | 通过工具调用写入
     | 通过搜索工具按需检索

关键设计：Agent 自己决定什么时候"记住"东西、什么时候"回想"东西。
不是被动记录，是主动管理自己的记忆 —— 就像人会主动记笔记和翻笔记。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Message:
    """一条对话消息"""
    role: str           # 角色：用户 / 助手 / 系统
    text: str           # 消息内容


@dataclass
class MainContext:
    """主上下文 —— Agent 的"工作记忆"。

    容量有限（max_messages 条消息）。
    超出容量的旧消息被逐出到 evicted 列表，但归档存储可以事后检索它们。
    """

    core: dict[str, str] = field(default_factory=dict)   # 核心信息块（persona、用户偏好等）
    messages: list[Message] = field(default_factory=list) # 当前上下文消息
    max_messages: int = 4                                  # 消息容量上限
    evicted: list[Message] = field(default_factory=list)   # 被逐出的历史消息

    def append(self, role: str, text: str) -> None:
        """追加一条消息，超出容量则逐出最旧的消息"""
        self.messages.append(Message(role=role, text=text))
        while len(self.messages) > self.max_messages:
            self.evicted.append(self.messages.pop(0))

    def render(self) -> str:
        """渲染当前上下文为可读格式"""
        parts: list[str] = ["[核心信息]"]
        for key, value in sorted(self.core.items()):
            parts.append(f"  {key}: {value}")
        parts.append("[当前消息]")
        for msg in self.messages:
            parts.append(f"  {msg.role}: {msg.text}")
        return "\n".join(parts)


@dataclass
class ArchivalRecord:
    """一条归档记录 —— 存在"长期记忆"里"""
    rid: str                        # 记录ID
    text: str                       # 记录内容
    tags: tuple[str, ...] = ()      # 标签，便于分类检索
    session_id: str = "s0"          # 属于哪个会话
    turn_id: int = 0                # 在第几轮被写入


class ArchivalStore:
    """归档存储 —— Agent 的"长期记忆"。

    提供两个操作：
      - insert(): 写入一条新记录
      - search(): 用关键词搜索已有记录

    真实的归档存储会用向量数据库（Pinecone/Chroma/Milvus），
    但这里的实现用简单的词重叠度匹配来模拟，原理完全一致。
    """

    def __init__(self) -> None:
        self._records: list[ArchivalRecord] = []
        self._counter = 0

    def insert(self, text: str, *,
               tags: tuple[str, ...] = (),
               session_id: str = "s0",
               turn_id: int = 0) -> str:
        """插入一条新记录，返回记录ID"""
        self._counter += 1
        rid = f"mem_{self._counter:03d}"
        self._records.append(ArchivalRecord(
            rid=rid, text=text, tags=tags,
            session_id=session_id, turn_id=turn_id,
        ))
        return rid

    def search(self, query: str, top_k: int = 3) -> list[ArchivalRecord]:
        """基于词重叠度的搜索（模拟语义检索）。

        真实环境替换为：
          向量化 query → 在向量数据库里找 nearest neighbors → 返回最匹配的记录
        """
        q_tokens = set(query.lower().split())
        scored: list[tuple[float, ArchivalRecord]] = []

        for record in self._records:
            r_tokens = set(record.text.lower().split())
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            # Jaccard 相似度
            score = overlap / (len(q_tokens) + len(r_tokens) - overlap)
            scored.append((score, record))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    def count(self) -> int:
        """返回总记录数"""
        return len(self._records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 记忆工具集 —— Agent 操纵记忆的接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MemoryTools:
    """Agent 用来管理记忆的工具集。

    就像人有"记笔记"和"翻笔记"两种行为，
    Agent 通过这些工具来主动管理自己的记忆状态。
    """

    def __init__(self, main: MainContext, archival: ArchivalStore) -> None:
        self.main = main
        self.archival = archival

    # ── 核心记忆操作 ──

    def core_memory_append(self, section: str, text: str) -> str:
        """往核心记忆的某个区块追加内容"""
        existing = self.main.core.get(section, "")
        if existing:
            self.main.core[section] = f"{existing} {text}".strip()
        else:
            self.main.core[section] = text
        return f"核心记忆 [{section}] 已追加（当前 {len(self.main.core[section])} 字）"

    def core_memory_replace(self, section: str, old: str, new: str) -> str:
        """替换核心记忆中的某段内容"""
        current = self.main.core.get(section, "")
        if old not in current:
            return f"错误：在核心记忆 [{section}] 中未找到 {old!r}"
        self.main.core[section] = current.replace(old, new)
        return f"核心记忆 [{section}] 已替换"

    # ── 归档记忆操作 ──

    def archival_memory_insert(self, text: str, tags: tuple[str, ...] = ()) -> str:
        """将一条信息写入长期归档"""
        rid = self.archival.insert(text, tags=tags)
        return f"已归档 {rid}（总计 {self.archival.count()} 条记录）"

    def archival_memory_search(self, query: str, top_k: int = 3) -> str:
        """在归档记忆中搜索相关内容"""
        hits = self.archival.search(query, top_k=top_k)
        if not hits:
            return "未找到匹配记录"
        lines = []
        for h in hits:
            tags_str = f" [{', '.join(h.tags)}]" if h.tags else ""
            lines.append(f"  {h.rid}{tags_str}: {h.text}")
        return "\n".join(lines)

    # ── 对话检索 ──

    def conversation_search(self, query: str) -> str:
        """在历史对话（包括被逐出的消息）中搜索"""
        q = query.lower()
        all_msgs = self.main.evicted + self.main.messages
        for msg in reversed(all_msgs):
            if q in msg.text.lower():
                return f"找到匹配（{msg.role}）: {msg.text}"
        return "未在对话历史中找到匹配"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 脚本化 Agent 执行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def run_scripted_agent(tools: MemoryTools, script: list[ToolCall]) -> list[str]:
    """执行预设的工具调用序列（模拟 Agent 的自主记忆管理）"""
    observations: list[str] = []
    for call in script:
        fn = getattr(tools, call.name, None)
        if fn is None:
            observations.append(f"错误：未知记忆工具 {call.name!r}")
            continue
        try:
            observations.append(fn(**call.args))
        except Exception as e:
            observations.append(f"错误：{type(e).__name__}: {e}")
    return observations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("🧠 两层记忆架构 · MemGPT 模式")
    print("=" * 65)

    # 创建记忆系统
    main_ctx = MainContext(max_messages=3)   # 只保留最近 3 条消息
    archival = ArchivalStore()                # 长期存储，容量无限
    tools = MemoryTools(main_ctx, archival)

    # ━━ 阶段 1：建立初始对话 ━━
    print("\n📝 阶段 1：建立对话上下文")
    main_ctx.append("用户", "我叫阿明，专门做 Agent 工程化")
    main_ctx.append("助手", "记下了。现在在做什么项目？")
    main_ctx.append("用户", "在做知识库检索机器人，接了 12 个工具")
    main_ctx.append("助手", "12 个工具已经进入长链区间，要考虑工具漂移的问题")
    print(main_ctx.render())

    # ━━ 阶段 2：Agent 主动写入记忆 ━━
    print("\n📝 阶段 2：Agent 主动管理记忆")

    script = [
        # 写入核心记忆 —— persona 信息
        ToolCall("core_memory_append",
                 {"section": "角色定位", "text": "礼貌专业，记住用户的所有偏好"}),

        # 写入核心记忆 —— 用户信息
        ToolCall("core_memory_append",
                 {"section": "用户信息", "text": "姓名=阿明，专长=Agent工程化"}),

        # 归档用户项目信息
        ToolCall("archival_memory_insert",
                 {"text": "阿明在做知识库检索机器人，接了12个工具给销售团队用",
                  "tags": ("项目", "阿明")}),

        # 归档技术参考信息
        ToolCall("archival_memory_insert",
                 {"text": "BFCL V4 测试显示超过20步的工具链会产生漂移",
                  "tags": ("技术", "工具链")}),

        # 归档架构模式
        ToolCall("archival_memory_insert",
                 {"text": "休眠时计算（sleep-time compute）可以异步整合记忆",
                  "tags": ("架构", "记忆")}),
    ]

    observations = run_scripted_agent(tools, script)

    print("\n记忆写入日志：")
    for call, obs in zip(script, observations):
        print(f"  🔧 {call.name}({call.args}) → {obs}")

    # ━━ 阶段 3：对话继续，触发记忆逐出 ━━
    print("\n📝 阶段 3：继续对话，旧消息被逐出")
    main_ctx.append("用户", "你刚才说的工具链漂移是什么情况？")
    main_ctx.append("助手", "让我在归档里查一下那篇文章")
    print(main_ctx.render())
    print(f"  （已逐出 {len(main_ctx.evicted)} 条旧消息）")

    # ━━ 阶段 4：Agent 按需检索记忆 ━━
    print("\n📝 阶段 4：Agent 按需从长期记忆中捞数据")

    print("\n🔍 在归档中搜索 '工具链漂移'：")
    result = tools.archival_memory_search("工具链 漂移", top_k=2)
    print(result)

    print("\n🔍 在对话历史中搜索 '检索机器人'：")
    result = tools.conversation_search("检索机器人")
    print(result)

    # ━━ 关键洞察 ━━
    print(f"\n💡 两层记忆架构的核心设计：")
    print(f"   • 主上下文 = 工作记忆（容量有限，快速读写）")
    print(f"   • 归档存储 = 长期记忆（容量无限，按需检索）")
    print(f"   • 记忆逐出 ≠ 记忆丢失，归档里还能找到")
    print(f"   • Agent 自己决定什么时候'记'、什么时候'想'")
    print(f"   • 真实环境：归档存储 → 向量数据库，搜索 → 语义检索")
    print(f"   • 核心洞察：记忆是工具，不是被动日志。")


if __name__ == "__main__":
    main()
