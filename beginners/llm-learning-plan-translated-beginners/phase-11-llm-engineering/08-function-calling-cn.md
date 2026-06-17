# 函数调用

> 详细版见 `phase-11-llm-engineering-beginners/phase-11-08-function-calling-cn.md` 和 `for-beginners/station-02/`  

五步循环：用户提问 → 模型输出 tool_call → 代码执行 → 结果喂回 → 模型生成答案。模型只决定调用哪个函数，你的代码负责执行。
