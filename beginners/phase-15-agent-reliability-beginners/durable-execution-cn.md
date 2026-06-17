# 持久执行

> 详细版见 `for-beginners/station-04/05-production-scaling.md`  

每步后保存检查点。崩了从最后一步恢复，不从头开始。LangGraph 自动做，FastAPI+Postgres 是 KISS 路线。
