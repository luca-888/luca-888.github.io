# 线性 Attention 与混合架构文章资料

状态：占位，未开始调研。三篇 attention 变体文章的第三篇（2026-10-08 用户确定拆分），独立成篇。

## 范围（初拟）

- 去掉 softmax 后的状态递推：linear attention → RetNet / Mamba2 → DeltaNet / Gated DeltaNet → KDA。
- 混合架构：每 3–4 层线性层配 1 层全注意力（Qwen3-Next / Qwen3.5、Kimi Linear、MiniMax-01、Nemotron-H）。
- 反对意见：MiniMax M2 退回全注意力的说明。
- 推理侧：state cache 与 KV cache 并存的管理方式（vLLM / SGLang）。
- 开始前查看站内 Jev、LLM 量化文章对 Gated DeltaNet 的已有介绍，避免重复。
