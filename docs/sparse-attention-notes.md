# 稀疏 Attention：Sliding Window、NSA 与 DSA 文章资料

状态：占位，未开始调研。三篇 attention 变体文章的第二篇（2026-10-08 用户确定拆分），独立成篇。

## 范围（初拟）

- 固定模式：sliding window 与全局层交替（Gemma、gpt-oss）；attention sink（StreamingLLM、gpt-oss 的可学习 sink）。
- 学习式选择：NSA、DSA（lightning indexer + top-k）、MoBA。
- 对 kernel 与 KV cache 管理的影响（vLLM / SGLang 实现）。
- 开始前查看站内 DSpark 文章对 DeepSeek V4 的已有介绍，避免重复。
