# 函数调用：让 LLM 长出"手"

> **原文：** Function Calling & Tool Use  
> **预计时间：** ~30 分钟  
> 注：详细版见 `for-beginners/station-02/02-function-calling-deep-dive.md`

---

## 🚗 开场翻车：AI 说东京 15°C，但外面 35°C

你问 AI："东京现在天气怎么样？"

它回："根据季节，东京大约 15°C……"——**这是一个披着免责声明外衣的幻觉。** 模型不知道天气，因为天气每小时都在变，训练数据是几个月前的。

正确答案需要调用 OpenWeatherMap API。但模型不会调 API。**你的代码会。**

---

## 🧠 函数调用循环

```
用户提问 → 模型输出 "调 get_weather，city=东京" → 你的代码执行 → 结果发给模型 → 模型生成答案
```

**模型只决定"调哪个函数、传什么参数"。你的代码负责执行。**

### 工具定义（JSON Schema）

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的当前天气。返回摄氏温度。",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "城市名，英文"}
      },
      "required": ["city"]
    }
  }
}
```

### 最小实现

```python
import json

def handle_tool_call(model_response):
    """解析模型的工具调用，执行函数，返回结果"""
    tool_call = model_response["tool_calls"][0]
    func_name = tool_call["function"]["name"]
    # OpenAI 的参数是 JSON 字符串，需要 parse
    args = json.loads(tool_call["function"]["arguments"])
    
    if func_name == "get_weather":
        # 你的实际 API 调用在这里
        result = call_weather_api(args["city"])
        return {"role": "tool", "content": json.dumps(result)}
```

---

## ⚠️ 两大陷阱

1. **OpenAI 的参数是 JSON 字符串**（记得 `json.loads`）。Anthropic 的是解析好的 dict。
2. **工具描述会被塞进模型上下文**。如果被投毒了（见 MCP 安全那篇），模型会执行攻击者的指令。

---

## 🏆 焊死在脑子里的东西

1. **函数调用 = 模型输出 JSON → 你的代码执行 → 结果喂回去。**
2. **工具描述就是提示词。** 写清楚"Use when X. Do not use for Y."
3. **三家 API 形态不同，概念相同。** OpenAI 叫 function calling，Anthropic 叫 tool use，Gemini 叫 functionDeclarations。

---

> **没有函数调用的 LLM 是百科全书。有了它，LLM 变成了 Agent——能查天气、发邮件、改数据库。**

> 🧪 本章代码：[code/function-calling/main.py](code/function-calling/main.py) — Function Calling API 交互协议完整演示，初学者友好。
