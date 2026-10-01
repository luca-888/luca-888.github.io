# Vllm · Prefix Caching 文章资料

状态：正文初稿完成（draft，未发布）。用户 2026-09-30 回复“写正文”，视为确认参考方案；未做硬件实测，是否补测由用户之后决定。

## 范围与硬件

- 独立成篇。KV cache、block、`ref_cnt` 等背景就地交代；PagedAttention 一文第五节已有 prefix caching 概览，本篇需要讲得更深，不复用那一节的表述和图。
- 标题按词表：主标题“Vllm”，子标题暂定“Prefix Caching：跨请求复用 KV Cache”。
- 源码基准：vLLM main@1b77cc3（2026-09-30），SGLang main@8854857（2026-09-29）；本地浅克隆在 `~/Documents/ChatGPT/reference-repos/`。发布前按最新 release tag 复核引用位置。
- 实验：需要，单卡 RTX 4090 即可，获取途径待用户确认，确认前不动手。测试矩阵见文末。

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| vLLM 源码：`v1/core/kv_cache_manager.py`、`kv_cache_utils.py`、`block_pool.py`、`single_type_kv_cache_manager.py`、`sched/scheduler.py` | vLLM 团队 | `get_computed_blocks`、hash 计算、`find_longest_cache_hit`、`free`、scheduler 命中分支已读；`block_pool.cache_full_blocks` 待读 | 核心 | 2–4 |
| [vLLM Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)（仓库 `docs/design/prefix_caching.md`） | vLLM 团队 | 全文已读 | 核心 | hash 链、extra keys、cache_salt、free queue 逆序、Time 1–6 示例、重复 block → 2–3、7 |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)（NeurIPS'24） | Zheng et al., LMSYS | arXiv 源文件 §4、附录（调度伪代码、DFS 定理证明、DP 分布式）、eval 中 cache 相关段已读 | 核心 | radix tree、叶子优先 LRU、cache-aware scheduling、命中率与开销数据 → 5 |
| SGLang 源码：`mem_cache/radix_cache.py`、`evict_policy.py`、`managers/schedule_policy.py`、`mem_cache/README.md` | SGLang 团队 | 结构与调度策略已读；插入 / 分裂细节未读 | 核心 | 当前 page_size 对齐、多种淘汰策略、LPM / DFS-weight、队列 >128 回退 FCFS、in-batch 去重 → 5 |
| [vLLM APC 使用文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | vLLM 团队 | 已读 | 重点 | 适用负载、只加速 prefill、Mamba hybrid 的 align 模式 → 1、7 |
| [vLLM KV Offloading 使用文档](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) | vLLM 团队 | 前半已读 | 重点 | CPU / 多级 tier 扩展 prefix cache 容量 → 6 |
| [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see)（2025-09） | llm-d 团队（Red Hat、IBM、Google 等） | 网页摘要已读，原文待通读 | 重点 | 多副本下 cache 局部性被负载均衡打散；precise（KV events）与 approximate 路由；数据为厂商自报 → 6 |
| [The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) | Song et al. | 源文件 intro、defense 已读 | 重点（反对意见） | 共享 prefix cache 的计时侧信道、按 k token 粒度共享的缓解；PromptLeak 利用 SGLang LPM 调度 → 7 |
| [Marconi](https://arxiv.org/abs/2411.19379)（MLSys'25） | Pan et al. | 摘要已读 | 补充 | hybrid 模型的 SSM 状态只能精确命中，需按复用概率与节省算力做准入 / 淘汰 → 7 |
| [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | vLLM 团队 | 已读（PagedAttention 笔记） | 补充 | 命中率为 0 时吞吐下降 <1%，默认开启 → 3 |
| SafeKV / PrefixWall 等缓解方案 | 多组 | 仅见标题 | 补充（可选） | 侧信道缓解的其他路线 → 7 |
| Mooncake、LMCache、SGLang HiCache | 各团队 | 未读 | 补充（可选） | 分层 KV 存储，只点到为止 → 6 |

## 阅读要点

- vLLM hash：`hash((parent_hash, block_token_ids, extra_keys))`，首块父 hash 为固定种子派生的 `NONE_HASH`；extra keys 含 LoRA 名、多模态输入 hash 与 offset、`cache_salt`（只加在首块）、prompt embeds 摘要。默认 `sha256`（pickle 序列化），可选 `sha256_cbor`（跨进程可复现）与 `xxhash`（非加密、更快）。
- 只缓存写满的 block；hash 在 block 写满时增量计算，decode 产生的 token 写满 block 后也会进入缓存，所以多轮对话的上一轮回答可被下一轮命中（前提是 chat template 重新拼出的 token 一致）。
- `get_computed_blocks`：命中上限为 `num_tokens - 1`，因为最后一个 token 必须重算以得到 logits；又因命中长度需按 block 对齐，全命中时实际会重算整个最后一个 block（源码注释明确写出）。EAGLE / MTP 会再丢掉最后一个命中 block。
- 调度：命中部分计入 `num_computed_tokens`，不占 token budget；vLLM 引擎内只有 FCFS / priority，不按命中长度重排 waiting 队列。
- 释放：请求结束时 block 逆序进入 free queue 队尾，越靠后的 block 越先被淘汰；分配时从队首取，取到带 hash 的 block 才真正从缓存表删除。V1 的 block table 只追加，同一内容可能存在重复 block，请求结束后消失。
- `get_num_common_prefix_blocks` 服务于 cascade attention（所有运行中请求共享的前缀只算一次 attention），可作为一句话延伸，不展开。
- `reset_prefix_cache`：RLHF 中权重更新后需清空缓存。
- SGLang 论文：radix tree 边可带任意长 token 序列；原实现 page size = 1；缓存与运行中请求共用同一内存池；叶子优先 LRU；`ref` 计数保护运行中节点；cache-aware scheduling 按最长前缀优先，离线情形等价于 DFS 并达到最优命中率（cache ≥ 最长请求）；贪心可能饿死。开销：ShareGPT 100 请求 74.3 s 中树操作 0.2 s。Chatbot Arena 一个月命中率 52.4%（LLaVA-Next-34B）与 74.1%（Vicuna-33B），Vicuna TTFT 平均降 1.7×。
- SGLang 现状：match 结果按 `page_size` 向下取整（已支持大 page）；淘汰策略有 LRU、LFU、FIFO、MRU、FILO、priority、TLRU、SLRU；调度策略 LPM、DFS-weight、HRRN、FCFS、LOF 等，waiting 队列超过 128 时 LPM 自动回退 FCFS；in-batch prefix caching：多个请求共享同一未缓存前缀时先只放一个。
- 现状变化：vLLM V0 的 hash 曾是 Python `hash`，存在碰撞风险，v0.11 起默认 sha256；V0 发现重复 block 会替换，V1 block table 只追加故保留重复；SGLang 从 token 粒度扩展到 page 粒度，从单一 LRU 扩展到多种策略，正在收敛到 Unified Radix Cache。
- 对比与分歧：vLLM 是 block 粒度的 hash 表，查找为逐块 O(1)，命中只能按 block 对齐；SGLang 是 radix tree，可按 token 精确匹配并能按命中长度调度，但调度重排带来饥饿与侧信道风险（PromptLeak）。多副本时两者都要靠外部路由维持局部性（llm-d、SGLang router）。

## 文章主线

Prefix caching 把 KV cache 从“每个请求的私有工作内存”变成“按内容寻址、跨请求共享的缓存”。它是否有效取决于三件事：能不能找到（索引与粒度）、还在不在（淘汰与容量）、请求会不会落到缓存所在的地方（调度与路由）。代价在没有命中时几乎为零，风险在隐私与模型结构上。

## 章节提纲

1. **复用的是什么**：prefill 计算量随 prompt 长度增长，命中的 token 直接跳过 prefill，缩短 TTFT、腾出 token budget；decode 不受益。典型负载：多轮对话、共享 system prompt、few-shot、agent 循环。
2. **vLLM 怎么找到缓存**：hash 链、只缓存满 block、extra keys；一个 block size 4 的逐块查找示例；最后一个 token 必须重算、命中按 block 对齐。
3. **缓存就是空闲 block**：`ref_cnt`、free queue、逆序释放与 LRU；按设计文档的时间线走一遍；重复 block。
4. **与调度器的衔接**：命中不占 budget；FCFS 不为命中重排；多轮对话能命中的条件（chat template、reasoning 内容剥离等）。
5. **SGLang 的另一种做法**：radix tree、叶子优先淘汰、cache-aware scheduling 与 DFS 最优性；与 vLLM 的对照表（索引、粒度、淘汰、调度、开销）。
6. **单卡之外**：容量上 CPU / 分层存储（vLLM offloading、LMCache、HiCache）；多副本上 cache-aware 路由（llm-d 数据，注明自报）。
7. **边界**：计时侧信道与 `cache_salt`、hash 碰撞与默认 sha256、hybrid / SSM 模型只能精确命中（Marconi、vLLM align 模式）、权重更新后必须清缓存。
8. **实测**（若确认硬件）：见下。

## 核心图（草图阶段，待确认方向）

1. **hash 链查找**：回答“新请求怎样逐块查到已缓存的 block、为何在第一个不同的 block 处停止、为何最后一块要重算”。一行 token 条按 block 分段，每段上方是链式 hash，命中段实心、未命中段描边，突出一条完整查找路径。
2. **free queue 时间线**：回答“缓存与空闲是同一批 block，淘汰顺序怎么来”。4–5 个时刻的 block 池条带，颜色区分运行中 / 空闲带 hash / 空闲无 hash / 被淘汰。
3. **block 与 radix tree 对照**：回答“同一组请求在两种索引下命中了多少、差在哪里”。左侧 hash 表按 block 命中（尾部部分 block 不命中），右侧 radix tree 按 token 命中与节点分裂。
4. **实测图**（ECharts）：TTFT 随共享前缀长度 / 命中率变化；多轮对话中命中率随并发会话数与 KV 容量的变化。

## 待核实

- SGLang 当前默认 `schedule_policy`、`page_size`、`radix_eviction_policy`（`server_args.py` 定义位置未找到，需再查）。
- vLLM `block_pool.cache_full_blocks` 与 hash 增量计算的时机（decode 写满 block 时即入缓存）。
- 多轮对话中 chat template 是否导致上一轮回答的 token 与下一轮 prompt 中的 token 不一致（reasoning 模型剥离 thinking 内容时尤其）。
- llm-d 博客原文与测试条件；是否引用其数字。
- The Early Bird 的作者全名与发表场合；PromptLeak / InputSnatch 的出处。
- vLLM 是否有官方的 prefix cache 命中率指标名称（`PrefixCacheStats` → Prometheus 指标）。

## 实验计划（待确认硬件）

- 硬件：单卡 RTX 4090（24 GB），模型选 Qwen3-8B 或同级 BF16 模型（KV 容量足够体现淘汰）。
- 测试：
  1. 共享 system prompt：前缀长度 0 / 1k / 4k / 8k，prefix caching 开与关，测 TTFT 与吞吐。
  2. 多轮对话：N 个并发会话各跑多轮，统计 vLLM 报告的命中率与每轮 TTFT；调节并发会话数使工作集超过 KV 容量，观察命中率下降。
  3. 可选：同一负载在 SGLang 上跑一遍，对比命中率（LPM 与 FCFS）。
- 数据来源：vLLM `vllm bench serve` 或自写客户端 + 服务端指标；原始数据与脚本放 `public/measurements/`。

## 成稿记录

- 正文：`content/vllm-prefix-caching.md`；入口 `posts/vllm-prefix-caching/`；图 `src/VllmPrefixFigures.tsx`；映射 `src/vllm-prefix-caching.blocks.tsx`。未加入小红书导出列表。
- 引用位置按 v0.30.0 tag 复核：`max_cache_hit_length = num_tokens - 1`、默认 `sha256`、`enable_prefix_caching=True`、`free` 逆序、`free_blocks` 无 hash 放队首 / 有 hash 放队尾、`touch`、`num_tokens_to_cache`、`prefix_match_unit` 均存在。
- 已核实并写入正文：SGLang 默认 `schedule_policy="fcfs"`（调参文档建议共享前缀多时用 `lpm`），CUDA 默认 `page_size=1`，默认淘汰 `lru`；Qwen3 chat template 删除最近一条用户消息之前各轮回答的 thinking。
- 现状变化写入正文：设计文档 Time 4–6 的队列顺序与 v0.30.0 不符（无 hash 的 block 现在放队首）；SGLang 默认调度从论文的 LPM 变为 FCFS。
- 自绘图三张（均为构造示例）：链式 hash 逐块查找、free queue 归还与淘汰、block hash 表与 radix tree 的命中对照。未使用论文原图。
- 笔者分析（正文已标注）：逆序归还避免子 block 比父 block 晚淘汰；hash 表的定长自包含 key 便于外部索引。抢占后重新调度可命中自己的 block 为基于源码行为的推断。
- 未通读：llm-d 博客原文（数字来自网页摘要）、Marconi 正文、PromptLeak 与 InputSnatch 原文（引自 Early Bird 的相关工作描述）；发布前复核。

## 封面样稿（2026-09-30）

- 沿用用户在 continuous batching 封面中选择的抽象复古印刷方向：3:4 竖版、暖黄纸、蓝橙油墨、大字、颗粒；不使用教学式编号与图例。
- 阅读本文后，以一段蓝色公共主干与橙色不同分支表现“相同前缀 KV 共用一份，只计算不同后缀”，不表达特定索引实现。
- 样稿：`covers/vllm-prefix-caching/xhs-retro/cover-v1.png`；完整提示词：同目录 `prompt-v1.json`。内置 imagegen 生成，待用户评估。

- 用户要求更换风格，并明确包含珊瑚色透明风格、增加多种选择。新增五张 3:4 样稿：蓝紫透明、珊瑚透明、烟灰玻璃、奶油纸雕、深色柔光；统一使用长公共叠层与分开的末端作为抽象意象。文件与提示词保存在 `covers/vllm-prefix-caching/xhs-material-styles/`，尚未选定最终版。

- 用户选中蓝紫透明版，并明确要求追加横版。已基于该竖版重新构图生成 16:9 横版，保留黑色大字与蓝紫公共叠层/分叉末端；保存至 `covers/vllm-prefix-caching/landscape-blue-violet/cover-v1.png`，未替换站点现有资源。
