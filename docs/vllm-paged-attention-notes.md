# Vllm · PagedAttention 文章资料

状态：正文初稿完成（draft，未发布）。用户 2026-09-29 确认参考方案，要求“直接写，先不考虑硬件实测”，并要求不沿用以往文章的结构与风格，正文精简、可视化更清晰。

## 范围与硬件

- 独立成篇，不做连载结构（用户 2026-09-29 确认）。
- 标题按词表：主标题“Vllm”，子标题单列方向（暂定“PagedAttention：KV Cache 的分页管理”）。
- 源码基准：vLLM v0.30.0（2026-09-22 发布）；阅读时使用 main@30f5c01（2026-09-29），写作前按 tag 复核引用位置。
- 本稿不做硬件实测：数据全部来自论文与官方资料，自绘图为构造示例并在图注注明。若后续补测，可用单卡 RTX 4090（paged 与 contiguous kernel 对比、KV 容量与吞吐）。

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 拟参考内容 → 章节 |
| --- | --- | --- | --- | --- |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（SOSP'23） | Kwon, Li et al., UC Berkeley | 全文已读 | 核心 | 问题定义、三类浪费、block table、CoW、调度与抢占、block size 消融 → 1–5 |
| [vLLM 源码 v1/core 与 FA backend](https://github.com/vllm-project/vllm) | vLLM 团队 | 关键文件已读 | 核心 | BlockPool、FreeKVCacheBlockQueue、allocate_slots、slot_mapping、reshape_and_cache_flash、FA block_table 调用、recompute 抢占 → 3–5 |
| [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) | vLLM 团队 | 已读 | 重点 | 动画原图、60–80% 浪费与 <4% 的表述 → 2–3 |
| [vLLM Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/) | vLLM 团队 | 已读 | 重点 | hash 链、ref_cnt、free queue LRU → 5 |
| [vAttention](https://arxiv.org/abs/2405.04437)（ASPLOS'25） | Prabhu et al., Microsoft Research India | HTML 版关键章节已读 | 重点 | 对分页的批评：kernel 改写、prefill 开销、CPU block table 开销；CUDA VMM 替代路线 → 4 |
| [Inside vLLM](https://www.aleksagordic.com/blog/vllm) | Aleksa Gordić | KV cache 相关章节已读 | 重点 | V1 请求路径中的 block 分配、slot_mapping、recompute 抢占 → 3、5 |
| [vLLM Paged Attention kernel 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/) | vLLM 团队 | 已读 | 补充 | 论文时期 kernel 的 KV 布局与 warp 分工；官方注明为历史文档 → 4 |
| [FlashInfer](https://arxiv.org/abs/2501.01005)（MLSys'25） | Ye et al. | KV 存储章节已读 | 补充 | 把 page table 统一为 BSR；page size 与访存效率 → 4 |
| [vLLM V1 alpha release blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | vLLM 团队 | 已读 | 补充 | V1 调度、零开销 prefix caching、persistent batch → 5 |
| [FlashAttention README](https://github.com/Dao-AILab/flash-attention) | Tri Dao et al. | paged KV 部分已读 | 补充 | 2.5 起支持 block_table → 4 |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)（NeurIPS'24） | Zheng et al., LMSYS | 源码 mem_cache 已读（main@bd78095），论文未通读 | 补充 | radix tree 前缀复用、token 级分配粒度 → 五（用户 2026-09-29 要求补充对比） |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI'22） | Yu et al. | §4.2 调度算法已读 | 补充 | iteration-level scheduling 背景；Algorithm 1 按 max_tokens 预留 KV slot → 2 |

## 阅读要点

- 论文：OPT-13B 每 token KV 800 KB；现有系统仅 20.4%–38.2% KV 内存存放真实 token；浪费限制在最后一个 block；block size 16 为默认；paged kernel 比 FasterTransformer 慢 20–26%；swap 与 recompute 在 block size 16–64 时端到端相当。
- 当前 vLLM 与论文的差异：
  - attention 不再使用论文时期的自研 kernel，FlashAttention / FlashInfer / Triton backend 直接接收 block_table；写入用 slot_mapping 做 scatter（`reshape_and_cache_flash`）。
  - FA backend 支持 16 的倍数 block size，默认 16。
  - 并行采样拆成子请求，依靠 prefix caching 共享完整 block；通用的 CoW 已不在注意力 KV 路径中（仅 mamba 等细粒度部分命中使用）。
  - beam search 移到 entrypoint 层实现。
  - 抢占仅 recompute：释放 block、`num_computed_tokens = 0`，V0 的 swap 已移除。
  - block_id 0 作为 null block 占位。
- 分歧：vAttention 认为分页把虚拟内存工作搬进用户态，带来 kernel 改写与 prefill 开销（FA2 paged prefill 最多慢 37%），主张用 CUDA VMM 保持虚拟连续；decode 上两者相当。FlashInfer 则把 page table 当作 block-sparse 矩阵，认为非连续存储可以统一处理。

## 待核实

- parallel sampling 的 n 个子请求在同一步调度时是否共享 prompt block（`BlockHashToBlockMap` 允许同一 hash 对应多个 block）；正文暂用保守写法，可实测确认。

- RTX 4090 上 paged 与 contiguous 的 decode / prefill attention 延迟差（实测）。
- 选定模型的每 token KV 字节数与 4090 上的 `num_gpu_blocks`（实测日志）。

## 成稿记录

- 正文：`content/vllm-paged-attention.md`；入口 `posts/vllm-paged-attention/`；图 `src/VllmPagedFigures.tsx`。
- 自绘图三张：连续预留与分页对比（128 slot 构造示例，配色与论文 Figure 2 语义一致）、block table 与 slot 计算、prefix caching 共享。
- 论文原图一张：Figure 2，来源与校验见 `src/assets/vllm-paged-attention/SOURCES.md`。
- Orca 原文 §4.2 已读：调度器首次调度请求时按 max_tokens 预留 KV slot，结束时归还；与正文“按最大长度预留”一致。
- 第六节“recompute 往往能命中自己释放的 block”为基于源码行为的分析，正文已注明。
