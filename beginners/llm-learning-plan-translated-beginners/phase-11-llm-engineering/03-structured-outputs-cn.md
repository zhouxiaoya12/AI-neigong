# 结构化输出

> 详细版见 `phase-04-prompt-engineering-beginners/05-structured-outputs-json-schema-cn.md`  

"以 JSON 返回" ≠ 保证拿到 JSON。10% 失败率在生产不可接受。

四个等级：提示级(~90%) → JSON 模式(100%可解析) → Schema 模式(100%结构) → 约束解码(令牌级封杀)。**生产环境用等级 3+。**
