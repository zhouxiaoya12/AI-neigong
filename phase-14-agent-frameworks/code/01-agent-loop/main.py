"""
ReAct Agent 循环 · 最小可用实现（纯标准库）

实现 Agent 的五个核心要素：
  1. 消息缓冲区（message buffer）—— 存储每一轮的交互历史
  2. 工具注册表（tool registry）—— 模型可以按名称调用工具
  3. 停止条件（stop condition）—— 模型说"完成"或达到轮次上限
  4. 轮次预算（turn budget）—— 防止无限循环
  5. 观察格式化器（observation formatter）—— 将工具输出转成模型能读的格式

这里用了一个脚本化的 ToyLLM 来模拟 LLM 行为，让循环离线可跑、结果可复现。
把 ToyLLM 换成任何 LLM API 客户端，控制流完全一致——这就是 Agent 框架的底层秘密。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ToolCall:
    """一次工具调用的记录"""
    name: str               # 工具名称，如 "calculator"
    args: dict[str, Any]    # 调用参数，如 {"expr": "1+1"}


@dataclass
class Turn:
    """Agent 循环中的一轮（一条日志记录）"""
    kind: str                          # 类型：user / thought / action / final
    content: str                       # 本轮的文字内容
    tool_call: ToolCall | None = None  # 如果是 action 轮，记录工具调用
    observation: str | None = None     # 如果是 action 轮，记录工具返回结果


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具注册表 —— Agent 的"工具箱"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolRegistry:
    """工具注册表：注册、查询、调度三位一体。

    使用方式：
      registry = ToolRegistry()
      registry.register("calculator", 计算器函数)
      registry.register("搜索", 搜索函数)
      result = registry.dispatch(ToolCall("calculator", {"expr": "1+1"}))
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """注册一个工具。fn 接收关键字参数，返回字符串。"""
        self._tools[name] = fn

    def names(self) -> list[str]:
        """返回所有已注册工具的名称列表"""
        return sorted(self._tools)

    def dispatch(self, call: ToolCall) -> str:
        """执行一次工具调用，返回工具输出的字符串。

        三种失败情况的处理：
          - 工具不存在 → 返回 "error: 未找到工具 xxx"
          - 参数不匹配 → 返回 "error: 参数错误"
          - 执行异常   → 返回 "error: 异常类型: 异常信息"

        注意：所有错误都以字符串形式返回，不会被抛出。
        这是设计决策 —— Agent 需要看到错误信息来决定下一步怎么做，
        而不是被异常打断循环。
        """
        fn = self._tools.get(call.name)
        if fn is None:
            return f"错误：未找到工具 {call.name!r}"
        try:
            return fn(**call.args)
        except TypeError as e:
            return f"错误：工具 {call.name} 参数不匹配: {e}"
        except Exception as e:
            return f"错误：{type(e).__name__}: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 示例工具：计算器 + 键值存储
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculator(expr: str) -> str:
    """一个安全的计算器工具。
    只允许数字和 +-*/().空格，使用受限的 eval 执行。
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed_chars):
        return "错误：表达式包含不允许的字符"
    try:
        # eval 在受限环境中执行，禁止访问内置函数
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误：计算异常: {type(e).__name__}: {e}"


class KVStore:
    """一个简单的键值存储，模拟 Agent 的"记事本"。"""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str:
        """读取一个键的值，不存在时返回提示"""
        return self._store.get(key, f"未找到键: {key}")

    def set(self, key: str, value: str) -> str:
        """写入一个键值对"""
        self._store[key] = value
        return f"已存储 {key} = {value}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 玩具 LLM —— 用预编排的脚本模拟模型推理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToyLLM:
    """脚本化的 ReAct 策略。按顺序返回预设的"思考→行动"序列。

    每一轮的回复有两种类型：
      - 行动轮：{'thought': '思考过程', 'action': '工具名', 'args': {...}}
      - 完成轮：{'kind': 'finish', 'content': '最终答案'}

    换成真实 LLM 时，这里就是一次 API 调用。
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script      # 预设的行动脚本
        self.cursor = 0           # 当前执行到第几步

    def respond(self, history: list[Turn]) -> dict[str, Any]:
        """返回下一轮的思考+行动指令"""
        if self.cursor >= len(self.script):
            return {"kind": "finish", "content": "没有更多行动指令了"}
        entry = self.script[self.cursor]
        self.cursor += 1
        return entry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent 循环 —— 这就是一切 Agent 框架的核心
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class AgentLoop:
    """ReAct Agent 循环。

    流程：
      用户输入 → 模型思考 → 决定调用工具 → 执行工具 → 拿到结果
        → 模型再思考 → 再调用工具 or 给出最终答案 → 结束

    这个循环就是 Claude Code、Cursor、Devin、Operator 底层共用的那一套。
    """

    llm: ToyLLM                     # LLM 客户端（脚本或用真实 API）
    tools: ToolRegistry             # 工具注册表
    max_turns: int = 12             # 最大轮次上限，防止无限循环
    history: list[Turn] = field(default_factory=list)  # 完整的交互历史

    def run(self, user_message: str) -> str:
        """执行一次完整的 Agent 任务。

        参数：
          user_message: 用户输入的问题或指令

        返回：
          最终的答案字符串
        """
        # 记录用户输入
        self.history.append(Turn(kind="用户", content=user_message))

        for step in range(self.max_turns):
            # 步骤1：调用 LLM，获取思考+行动指令
            reply = self.llm.respond(self.history)

            # 步骤2：检查是否该停止了
            if reply["kind"] == "finish":
                self.history.append(Turn(kind="结束", content=reply["content"]))
                return reply["content"]

            # 步骤3：记录模型的思考过程
            thought = reply.get("thought", "")
            self.history.append(Turn(kind="思考", content=thought))

            # 步骤4：执行工具调用
            call = ToolCall(name=reply["action"], args=reply.get("args", {}))
            observation = self.tools.dispatch(call)
            self.history.append(
                Turn(kind="行动", content=call.name,
                     tool_call=call, observation=observation)
            )

        # 超出轮次上限，强制停止
        self.history.append(Turn(kind="结束", content="轮次预算用尽，强制结束"))
        return "已超出最大轮次限制，任务未完成。"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辅助函数：打印执行轨迹
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def pretty_trace(history: list[Turn]) -> None:
    """以可读格式输出 Agent 的完整执行轨迹"""
    for i, turn in enumerate(history):
        tag = f"[{i:02d} {turn.kind:>4}]"
        if turn.kind == "用户":
            print(f"{tag} {turn.content}")
        elif turn.kind == "思考":
            print(f"{tag} 💭 {turn.content}")
        elif turn.kind == "行动":
            call = turn.tool_call
            assert call is not None
            print(f"{tag} 🔧 {call.name}({call.args}) → {turn.observation}")
        elif turn.kind == "结束":
            print(f"{tag} ✅ {turn.content}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示：构造一个能算账的 Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_demo_agent() -> AgentLoop:
    """构造一个演示用的 Agent。

    场景：计算含税价格，用 KV 存储中间结果。

    执行轨迹：
      1. 存储基础价格 120 到 KV
      2. 计算 15% 的税额
      3. 存储税额
      4. 计算总价
      5. 读取基础价格确认
      6. 给出最终答案
    """
    # 创建工具注册表，注册三个工具
    tools = ToolRegistry()
    tools.register("计算器", calculator)

    kv = KVStore()
    tools.register("存储读取", kv.get)
    tools.register("存储写入", kv.set)

    # 预编排的行动脚本（模拟 LLM 的推理过程）
    script: list[dict[str, Any]] = [
        {
            "kind": "action",
            "thought": "先把基础价格 120 存起来，方便后续计算引用",
            "action": "存储写入",
            "args": {"key": "基础价", "value": "120"}
        },
        {
            "kind": "action",
            "thought": "计算 15% 的税额：120 × 0.15",
            "action": "计算器",
            "args": {"expr": "120 * 0.15"}
        },
        {
            "kind": "action",
            "thought": "税额是 18.0，存下来",
            "action": "存储写入",
            "args": {"key": "税额", "value": "18.0"}
        },
        {
            "kind": "action",
            "thought": "计算含税总价：120 + 18.0",
            "action": "计算器",
            "args": {"expr": "120 + 18.0"}
        },
        {
            "kind": "action",
            "thought": "确认一下基础价格是不是对了",
            "action": "存储读取",
            "args": {"key": "基础价"}
        },
        {
            "kind": "finish",
            "content": "含税总价为 138.0 元（基础价 120 + 15% 税额 18.0）"
        },
    ]

    return AgentLoop(llm=ToyLLM(script), tools=tools, max_turns=10)


def main() -> None:
    print("=" * 65)
    print("🧠 ReAct Agent 循环 · 最小完整实现")
    print("=" * 65)

    agent = build_demo_agent()
    final = agent.run("120 元加 15% 的税，总共多少钱？用存储中间过程。")

    print("\n📋 完整执行轨迹：\n")
    pretty_trace(agent.history)

    print(f"\n📊 执行统计：")
    print(f"   最终答案：{final}")
    print(f"   工具调用次数：{len([t for t in agent.history if t.kind == '行动'])}")
    print(f"   可用工具：{agent.tools.names()}")
    print(f"\n💡 关键洞察：")
    print(f"   • Agent 循环 = ReAct（思考 → 行动 → 观察）× N 次")
    print(f"   • 五个要素：缓冲区 + 工具表 + 停止条件 + 轮次预算 + 观察格式化")
    print(f"   • 换掉 ToyLLM，接上真实 API，控制流完全不变")
    print(f"   • 这就是所有 Agent 框架底层的共同秘密 🦅")


if __name__ == "__main__":
    main()
