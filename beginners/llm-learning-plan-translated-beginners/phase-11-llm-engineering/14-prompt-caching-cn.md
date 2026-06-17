# 提示缓存

> 详细版见 `for-beginners/station-05/02-prompt-caching.md`  

Anthropic 显式标记 `cache_control`，缓存读取便宜 90%。OpenAI 自动缓存 ≥1024 token 提示。**动态内容杀缓存——静态进缓存块，动态放外面。**
