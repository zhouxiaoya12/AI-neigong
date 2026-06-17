"""
函数调用（Function Calling） · 从 API 到工具执行（纯标准库）

这是你真正对接 LLM API 时发生的事：

1. 你告诉 LLM："你有这些函数可以用"（function definitions）
2. LLM 返回："我想调用 search(query='xxx')"
3. 你执行 search('xxx')，拿到结果 "找到了3条记录"
4. 你把结果发回 LLM："这是执行结果：找到了3条记录"
5. LLM 基于结果生成最终回答

这里用一个脚本化的 MockLLM 模拟整个流程，让你看清每一步发生了什么。
把 MockLLM 换成任何 LLM API（OpenAI / Claude / DeepSeek），这个流程完全不变。

关键差异：
  - 本文件专注 API 交互协议（call → execute → observe → next call）
  - 上一课的 Tool Registry 专注工具侧（Schema校验、错误处理、并行调度）
  两者在真实系统中合在一起使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Message:
    """一条对话消息，兼容 OpenAI / Anthropic 消息格式"""
    role: str           # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None   # assistant 发起 tool call
    tool_call_id: str | None = None                   # tool 角色关联 call id
    name: str | None = None                           # tool 角色工具名


@dataclass
class FunctionDef:
    """发给 LLM 的函数定义"""
    name: str
    description: str
    parameters: dict[str, Any]    # JSON Schema


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock LLM —— 模拟真实的 function calling API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MockLLM:
    """模拟支持 function calling 的 LLM API。

    真实调用流程（以 OpenAI 为例）：
      response = client.chat.completions.create(
          model="gpt-4",
          messages=[...],
          tools=[{"type":"function", "function": {...}}, ...],
      )
      if response.choices[0].finish_reason == "tool_calls":
          # LLM 要调用工具
          tool_call = response.choices[0].message.tool_calls[0]
          tool_name = tool_call.function.name
          tool_args = json.loads(tool_call.function.arguments)
          # 执行工具，把结果追加到 messages 中
          # 再次调用 API...

    这里的 MockLLM 用脚本模拟这个交互协议。
    """

    def __init__(self, conversation: list[dict[str, Any]]):
        self.conversation = conversation
        self.cursor = 0

    def chat(self, messages: list[Message]) -> dict[str, Any]:
        """模拟一次 chat completion 调用。

        返回格式（和 OpenAI 格式一致）：
          {
            "finish_reason": "stop" | "tool_calls",
            "content": "回答文本" | None,
            "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}] | None
          }
        """
        if self.cursor >= len(self.conversation):
            return {"finish_reason": "stop", "content": "对话结束", "tool_calls": None}

        reply = self.conversation[self.cursor]
        self.cursor += 1
        return reply


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具执行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolExecutor:
    """执行 LLM 请求的工具调用，返回结果字符串。"""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        self._tools[name] = fn

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """执行工具，返回结果。所有错误都转成字符串返回。"""
        fn = self._tools.get(name)
        if fn is None:
            return f"错误：未知工具 {name!r}"
        try:
            return fn(**args)
        except Exception as e:
            return f"错误：{type(e).__name__}: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 示例工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def search_knowledge(query: str, top_k: int = 3) -> str:
    """模拟知识库搜索"""
    kb = {
        "退款": "退款政策：30天内全额退款，企业版60天按比例退款。",
        "SLA":  "SLA：专业版 99.9%，企业版 99.99%。低于阈值补偿 10%/0.1%。",
        "价格": "价格：新手版29元/月，专业版99元/月/人，企业版500元起/月。",
        "API":  "API：REST+JSON，OAuth2.0认证。速率限制：新手100/分钟，专业1000/分钟，企业10000/分钟。",
    }
    q = query.lower()
    results = []
    for key, value in kb.items():
        if key.lower() in q or any(w in q for w in key.lower().split()):
            results.append(value)
        if not results:
            for key, value in kb.items():
                if any(w in value for w in q.split()):
                    results.append(value)

    if not results:
        return "未找到相关结果"

    return "\n".join(results[:top_k])


def get_weather(city: str) -> str:
    """模拟天气查询"""
    weather_db = {
        "北京": "北京：晴，25°C，湿度45%，微风",
        "上海": "上海：多云，28°C，湿度70%，东南风3级",
        "深圳": "深圳：雷阵雨，30°C，湿度85%，西南风",
    }
    return weather_db.get(city, f"未找到{city}的天气数据")


def send_email(to: str, subject: str, body: str) -> str:
    """模拟邮件发送"""
    return f"邮件已发送至 {to}，主题：{subject}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 完整的 Function Calling 循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def function_calling_loop(
    llm: MockLLM,
    executor: ToolExecutor,
    functions: list[FunctionDef],
    user_query: str,
    max_turns: int = 5,
) -> None:
    """完整的 Function Calling 循环。

    这个函数让你看到：从 API 交互的角度，一次"Agent任务"到底发生了什么。
    每一步都是真实的 API 调用模式——没有简化，没有省略。
    """

    # 第 0 步：准备消息列表
    messages: list[Message] = [
        Message(role="system", content="你是一个客服助手。你可以搜索知识库、查天气、发邮件。"),
        Message(role="user", content=user_query),
    ]

    print(f"📋 用户查询：{user_query}")
    print(f"🔧 可用工具：{[f.name for f in functions]}")
    print()

    for turn in range(max_turns):
        print(f"─ Round {turn + 1} ─")

        # 第 1 步：调用 LLM API（带上历史消息）
        response = llm.chat(messages)

        # 第 2 步：判断 LLM 的返回类型
        if response["finish_reason"] == "stop":
            # LLM 直接给出了最终回答，不需要调用工具
            print(f"  🤖 回答：{response['content']}")
            break

        elif response["finish_reason"] == "tool_calls":
            # LLM 想调用工具
            tool_calls = response["tool_calls"] or []

            # 记录助手消息（包含 tool_calls）
            assistant_msg = Message(
                role="assistant",
                content=response.get("content"),
                tool_calls=tool_calls,
            )
            messages.append(assistant_msg)

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"]["arguments"]
                call_id = tc["id"]

                print(f"  🔧 LLM 请求调用：{tool_name}({tool_args})")

                # 第 3 步：执行工具
                result = executor.execute(tool_name, tool_args)
                print(f"  📥 工具返回：{result}")

                # 第 4 步：将工具结果追加到消息列表
                tool_msg = Message(
                    role="tool",
                    content=result,
                    tool_call_id=call_id,
                    name=tool_name,
                )
                messages.append(tool_msg)

            # 第 5 步：带着工具结果再次调用 LLM
            print(f"  🔄 带着工具结果再次调用 LLM...")
            # （下一轮循环会自动处理）

    if turn == max_turns - 1:
        print(f"  ⚠️ 达到最大轮次上限")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    print("=" * 65)
    print("📞 Function Calling · 从 API 到工具执行")
    print("=" * 65)

    # 准备工具
    executor = ToolExecutor()
    executor.register("搜索知识库", search_knowledge)
    executor.register("查天气", get_weather)
    executor.register("发邮件", send_email)

    functions = [
        FunctionDef(
            name="搜索知识库",
            description="在内部知识库中搜索信息。用于查询政策、价格、技术文档等。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回条数"},
                },
                "required": ["query"],
            },
        ),
        FunctionDef(
            name="查天气",
            description="查询指定城市的当前天气。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        ),
        FunctionDef(
            name="发邮件",
            description="发送一封邮件。",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人地址"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
    ]

    # ── 场景 1：单工具调用（一步到位） ──
    print("\n📝 场景 1：单工具调用 · 退款政策查询")
    print("─" * 65)

    conv1: list[dict[str, Any]] = [
        {
            "finish_reason": "tool_calls",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "function": {
                    "name": "搜索知识库",
                    "arguments": {"query": "退款", "top_k": 1},
                },
            }],
        },
        {
            "finish_reason": "stop",
            "content": "根据知识库查询结果，我们的退款政策是：购买后30天内可全额退款，企业版客户享受60天的按比例退款窗口。请问还有什么可以帮您的？",
            "tool_calls": None,
        },
    ]

    llm1 = MockLLM(conv1)
    function_calling_loop(llm1, executor, functions, "我想退款，怎么办？")

    # ── 场景 2：多轮工具调用（先查知识库，再发邮件） ──
    print("\n📝 场景 2：多轮工具调用 · 查天气 + 发邮件")
    print("─" * 65)

    conv2: list[dict[str, Any]] = [
        {
            "finish_reason": "tool_calls",
            "content": None,
            "tool_calls": [{
                "id": "call_002",
                "function": {
                    "name": "查天气",
                    "arguments": {"city": "上海"},
                },
            }],
        },
        {
            "finish_reason": "tool_calls",
            "content": None,
            "tool_calls": [{
                "id": "call_003",
                "function": {
                    "name": "发邮件",
                    "arguments": {
                        "to": "team@example.com",
                        "subject": "今日上海天气报告",
                        "body": "上海今天多云，28°C，湿度70%。",
                    },
                },
            }],
        },
        {
            "finish_reason": "stop",
            "content": "已查询上海天气并发送报告邮件至 team@example.com。今天多云28°C，适合户外活动。",
            "tool_calls": None,
        },
    ]

    llm2 = MockLLM(conv2)
    function_calling_loop(llm2, executor, functions, "查一下上海天气，然后发给团队。")

    # ── 总结 ──
    print(f"\n💡 Function Calling 协议的核心：")
    print(f"  • user → assistant(tool_calls) → tool(result) → assistant → ...")
    print(f"  • 每次 LLM 响应要么是 'stop'（最终答案），要么是 'tool_calls'（要调工具）")
    print(f"  • 工具结果作为 tool 角色的消息追加到上下文，LLM 在下一次调用中看到它们")
    print(f"  • 这个协议是 OpenAI/Anthropic/DeepSeek 共用的标准模式")
    print(f"  • 把 MockLLM 换成真实 API 客户端，代码结构完全不变")
    print(f"  • 上一课的 Tool Registry（Schema校验）就接在这一课的 executor.execute() 前面")


if __name__ == "__main__":
    main()
