# 验证门禁

> 详细版见 `for-beginners/station-04/04-role-specialization.md`  

每个 Agent 的输出必须通过独立验证才能传给下一个 Agent。确定性验证（测试通过/失败）> LLM 自评。不加验证门禁 = 错误在流水线里传播。
