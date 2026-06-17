# 生产应用部署

> 预计时间：~20 分钟  

生产 LLM 应用的四个原则：

1. **评估自动化** — 每次变更都跑回归测试
2. **护栏分层** — 输入验证 + 输出验证
3. **可观测性** — 追踪每次调用的延迟、token、成本
4. **回滚能力** — 模型版本化，能快速切回上一个版本

**KISS 原则：** FastAPI + Postgres 在 100 并发以下通常够了。别过早引入 LangGraph/Temporal。

> 详见 `for-beginners/station-05/` 和 `for-beginners/station-04/05-production-scaling.md`
