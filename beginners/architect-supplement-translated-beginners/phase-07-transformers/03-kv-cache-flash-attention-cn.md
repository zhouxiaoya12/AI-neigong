# KV Cache 与 Flash Attention

> **预计时间：** ~15 分钟  

**KV Cache：** 解码时每个新 token 不需要重新算所有历史 token 的 K 和 V——缓存起来复用。解码加速 ~10x。

**Flash Attention：** 不把整个注意力矩阵写回 HBM（显存），分块在 SRAM 里算完。快 2-4 倍，省显存。

---

> **KV Cache 让解码不重复算历史。Flash Attention 让注意力不占满显存。两者结合 = 推理快 10 倍。**
