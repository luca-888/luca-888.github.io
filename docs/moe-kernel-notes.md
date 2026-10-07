# MoE Kernel 笔记

视频（暂无配套文章）：讲一个 MoE 层在 GPU 上怎样算，以 vLLM 的 Triton fused MoE 路径为主线。风格沿用分布式推理影片的编辑排版（风格 A，2026-10-07 用户选定）。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | vLLM `fused_moe/fused_moe.py`（`fused_moe_kernel`、`fused_experts_impl`、`_prepare_expert_assignment`、`get_default_config`） | vLLM | 已读 | 核心 | 第三至五章 |
| 2 | vLLM `fused_moe/moe_align_block_size.py`（docstring 中的 4 token 例子） | vLLM | 已读 | 核心 | 第三章 |
| 3 | vLLM `fused_moe/modular_kernel.py`（Router → Quantize-Dispatch → Permute-Experts-Unpermute → Combine） | vLLM | 已读开头说明 | 高 | 第四、六章 |
| 4 | vLLM `fused_moe/oracle/unquantized.py`（BF16 后端优先级） | vLLM | 已读 | 中 | 第六章 |
| 5 | [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)：grouped GEMM 的 contiguous / masked 两种布局 | DeepSeek | 已读相关两节 | 高 | 第六章 |
| 6 | [MegaBlocks](https://arxiv.org/abs/2211.15841)（Gale et al., 2022） | Stanford / Microsoft / Google | 只读摘要 | 中 | 第六章 |
| 7 | [Qwen3-30B-A3B config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | Qwen | 已读 | 核心 | 全片数字 |

源码版本：vLLM v0.31.0（`db9527a46873454610df6dbedf79a36d6bf1a7f6`），浅克隆在 `~/Documents/ChatGPT/reference-repos/vllm`。

## 阅读要点

- **模型数字**：Qwen3-30B-A3B，hidden 2048，`moe_intermediate_size` 768，128 experts，top-8，`norm_topk_prob` true，48 层。一个 expert = gate、up（2048 × 768）与 down（768 × 2048），BF16 约 9.44 MB；每层 128 个约 1.21 GB。
- **Triton 路径六步**（`fused_experts_impl`）：输入量化（BF16 跳过）→ `_prepare_expert_assignment`（对齐或 naive）→ GEMM1（w13，gate 与 up 合并，N = 2 × 768）→ activation（`silu_and_mul`）→ GEMM2（w2，乘 router 权重）→ `moe_sum`（top-k 求和）。中间缓存形状：`cache1 [M, top_k, N]`、`cache2 [M·top_k, N/2]`、`cache3 [M, top_k, K]`。
- **对齐**（`moe_align_block_size`）：把 `topk_ids` 展平成 M·top_k 个（token, expert）对，按 expert 排序，每个 expert 的段 padding 到 `BLOCK_SIZE_M` 的倍数；输出 `sorted_token_ids`、每个 block 的 `expert_ids` 和 padding 后的总数。padding 位填 `topk_ids.numel()`，在 kernel 里由 `token_mask` 屏蔽。
- **kernel 内部**：一个 program 负责 C 的一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 块；`expert_ids[pid_m]` 决定读哪个 expert 的权重；A 的行用 `sorted_token_ids // top_k` 直接 gather，不做物理重排；L2 复用靠 `GROUP_SIZE_M` 的分组顺序。
- **naive 分配**：`num_tokens × top_k × 4 ≤ num_experts`（且无 expert_map）时跳过对齐，每个 block 只算一个（token, expert）对。Qwen3-30B-A3B 下即 token 数 ≤ 4。
- **默认 tile**（BF16）：M ≤ 32 用 `BLOCK_SIZE_M` 16，≤ 96 用 32，≤ 512 用 64，更大用 128；M ≤ 64 时 `BLOCK_SIZE_N` 64、`BLOCK_SIZE_K` 128，否则 128 / 64；每个 expert 的 token 数 > 128 时 `GROUP_SIZE_M` 16。注释原文：小 batch memory-bound，大 batch compute-bound。
- **后端选择**（BF16，CUDA）：默认优先级 FlashInfer TRTLLM → FlashInfer CUTLASS → Triton；Hopper 上 FlashInfer 两者被移到最后（注释：在 SM90 上比 Triton 慢）。
- **DeepGEMM**：只对 M 轴分组，N、K 固定（各 expert 形状相同）。contiguous 布局用于训练前向和 prefill，各 expert 的段对齐到 M block；masked 布局用于开启 CUDA Graph 的 decode，CPU 不知道每个 expert 分到多少 token，靠 mask 只算有效部分，常接 DeepEP low-latency 的输出。
- **MegaBlocks**：此前的框架要么按 capacity 丢 token，要么把 padding 算进计算和显存；MegaBlocks 改写成 block-sparse 运算，不丢 token。

## 现状变化

- 旧版 `fused_experts_impl` 按 `VLLM_FUSED_MOE_CHUNK_SIZE` 分块循环；v0.31.0 已改为一次处理全部 token。
- `modular_kernel.py` 把 MoE 拆成可替换的 prepare/finalize（通信与量化）和 experts（计算），同一套 experts kernel 可以接不同的 EP 通信后端。

## 分镜（2026-10-07 用户确认）

不做实验、不配音、保留第七章。旁白以画面标题与说明文字呈现；第六章的激活 expert 数按均匀路由估算并标明，路由分布为固定种子的示意数据。

| 章 | 内容 |
| --- | --- |
| 开幕 + 一 | MoE 层算什么：MoE 层在 Transformer 层中的位置（替换 FFN，48 层每层一个）、一个 expert 的 SwiGLU（3 次 GEMM + 逐元素）、router 选 top-8、加权相加；16 个 token 的分派不均（2026-10-08 按用户反馈补入层位置与 expert 结构） |
| 二 | 逐个 expert 算：kernel 多、每个太小（Tensor Core 一次至少 16 行）、CPU 要等路由结果 |
| 三 | 按 expert 排队：源码注释的例子（改为 0 起编号），展平 → 排序 → 补齐 → 三张表 |
| 四 | 一次 kernel 算完：program 查 `expert_ids`、按 `sorted_token_ids // top_k` 取行、padding 屏蔽；grid 按最坏情况开，多余 program 直接返回 |
| 五 | 整层流水线与张量形状 |
| 六 | batch 决定瓶颈：激活 expert 数、每个 expert 的平均 token 数、vLLM 的 tile 分档 |
| 七 | DeepGEMM 两种布局、MegaBlocks、vLLM 的后端选择 |
| 结尾 | 回到开幕的 expert 网格 |
