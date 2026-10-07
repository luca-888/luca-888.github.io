# Vllm · Continuous Batching 与调度器 文章资料

状态：正文初稿完成（draft，未发布）。用户 2026-09-29 回复“先不做实验，先做完主题”，视为确认参考方案并开始写作；未做硬件实测，是否补测由用户之后决定。第一篇 PagedAttention 已完成。

## 范围与硬件

- 独立成篇，不出现“上一篇 / 下一篇”，KV block、preemption 所需背景就地交代。
- 标题按词表：主标题“Vllm”，子标题暂定“Continuous Batching 与调度器”。
- 源码基准：vLLM v0.30.0，`vllm/v1/core/sched/scheduler.py`（已读 `schedule()` 主循环、`_preempt_request`）与 `vllm/config/scheduler.py`（默认值）。
- 实验：需要。TTFT / TPOT / 吞吐随 `max_num_batched_tokens` 与 chunked prefill 开关的变化，单卡 RTX 4090 即可；获取途径待用户确认（见 `docs/gpu-experiments.md`），确认前不动手。

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| vLLM v0.30.0 `scheduler.py` / `config/scheduler.py` | vLLM 团队 | 主循环、抢占、配置已读 | 核心 | token budget、running 优先、waiting 补位、抢占 → 3–5 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310)（OSDI'24） | Agrawal et al., MSR India / Georgia Tech | arXiv 源文件 abstract、motivation（generation stall）、design（stall-free batching、token budget 选取）已读；eval 未读 | 核心 | decode 带宽受限、chunked prefill、stall-free、token budget 权衡 → 2–4 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI'22） | Yu et al., SNU / FriendliAI | §4.2 调度算法已读（见 PagedAttention 笔记）；§3 selective batching 未读 | 核心 | request-level 到 iteration-level 的转变 → 1 |
| [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release) | vLLM 团队 | 已读 | 重点 | 无 prefill/decode 之分、`{request_id: num_tokens}` → 3 |
| [vLLM 优化与调参文档](https://docs.vllm.ai/en/latest/configuration/optimization/) | vLLM 团队 | 已读 | 重点 | recompute 为默认抢占；`max_num_batched_tokens` 的权衡；chunked prefill 默认开启 → 4–5 |
| [DistServe](https://arxiv.org/abs/2401.09670)（OSDI'24） | Zhong et al., PKU / UCSD | background、discussion 相关段已读 | 重点（反对意见） | chunked prefill 只是用 TTFT 换 TPOT，不能消除干扰 → 6 |
| [DeepSpeed-FastGen（Dynamic SplitFuse）](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md) | DeepSpeed 团队 | blog 已读 | 补充（相关系统） | 同类 token budget 设计；厂商自报数据，仅标注来源 → 6 |
| [vLLM SOSP'23](https://arxiv.org/abs/2309.06180) | Kwon et al. | 已读（PagedAttention 笔记） | 补充 | swap 与 recompute 的原始比较、FCFS 抢占 → 5 |

## 阅读要点

- Sarathi-Serve：decode 算术强度低，batch 内可搭载 prefill 计算；Orca 与 vLLM 早期 prefill 优先，会造成 decode 的 generation stall（一次 prefill 迭代可达数秒）；FasterTransformer 类 request-level 调度 TBT 低但吞吐低。stall-free 顺序：先放所有 running decode，再放未完成 prefill，最后新请求；token budget 由 TBT SLO 与 chunk 开销权衡，另有 tile quantization（257 比 256 慢 32%）。自报：Mistral-7B 单 A100 容量 2.6×（相对 vLLM）。
- vLLM V1 `schedule()` 注释：无 prefill/decode 阶段，每个请求只有 `num_computed_tokens` 追赶 `num_tokens_with_spec`；每步先调度 running，再用剩余 budget 补 waiting；预算不足时 `long_prefill_token_threshold` 与 budget 截断 chunk；`allocate_slots` 失败则抢占（FCFS 取 `running[-1]`，priority 策略取优先级最低）；抢占即释放 block、`num_computed_tokens = 0`、放回 waiting 队首。
- 默认值：`enable_chunked_prefill=True`，`policy="fcfs"`；`max_num_batched_tokens` / `max_num_seqs` 默认值由平台决定（`DEFAULT_*` 常量，写作前需查具体取值）。
- 现状变化：V0 的 swap 已移除；V0 的 prefill 与 decode 分批（不混合）已改为统一 token 调度。
- 反对/对比：DistServe 认为 chunked prefill 无法消除干扰且有额外 prefill 开销，主张 PD 分离；FastGen 的 Dynamic SplitFuse 与 Sarathi 思路相近。

## 待核实

- Orca §3（selective batching）与 iteration-level 的原文表述、Sarathi eval 数据是否引用。
- `DEFAULT_MAX_NUM_BATCHED_TOKENS` / `DEFAULT_MAX_NUM_SEQS` 在 v0.30.0 的实际取值与按平台变化的规则。
- waiting 循环里 prefix cache 命中、`num_computed_tokens` 初值对 budget 的影响（与 prefix caching 一篇边界）。
- 抢占后的重算是否命中 prefix cache（PagedAttention 稿已有基于源码的分析，需复核）。
- 实测：4090 上 TTFT / TPOT / 吞吐 vs `max_num_batched_tokens`、chunked prefill 开关。

## 成稿记录

- 2026-09-30 封面后续反馈：用户喜欢复古印刷风格，但认为三步预算示例过于具体。沿用暖黄纸、蓝橙油墨、大字与颗粒，去掉 STEP、A/B/C、预算数字、图例和框格；用规律短色块与穿插的分段长色带抽象表现持续生成与分块混批。新样稿：`covers/vllm-scheduler/xhs-retro-abstract/cover-v1.png`，待用户评估。

- 2026-09-30 封面方向确认：用户否定仅用轨道与方块补位的意象；确认以连续三个迭代 batch 的预算分配为主视觉。采用正文构造示例：budget = 8，A、B 每步各占 1 token，C 的 prefill 按 6、6、4 分块，第三步余 2 格空预算。输出六种 3:4 小红书竖版风格样稿；这是机制示意，不是实测数据。封面风格待用户选择。

- 正文：`content/vllm-scheduler.md`；入口 `posts/vllm-scheduler/`；图 `src/articles/vllm-scheduler/VllmSchedulerFigures.tsx`。
- 2026-09-29 按用户反馈调整结构：先讲 prefill/decode，再用静态 batch 槽位图引入 iteration-level，拐点曲线移到干扰一节前作依据。
- 2026-09-30 按用户反馈：正文改为短段落为主、并列内容用列表或表格；删去“相关系统与分歧”一节（DistServe、FastGen 仅留在本笔记）；Orca 只在第二节作为 iteration-level 与 selective batching 的出处出现；补 TTFT / TBT / TPOT 定义，时间线图的间隔改称 TBT；selective batching 补 A/B/C 展平示例与 `cu_seqlens_q`；V1 两个计数改为 prompt 30 token 的具体示例表；budget 图图注说明 R / W。
- 自绘图三张：静态 batch 与 iteration-level 的槽位占用（3 槽 × 8 步，构造示例）、三种调度的时间线（由代码模拟生成，迭代耗时 = max(1, tokens/4)，budget = 8，构造示例）、单步 token budget 分配与抢占。
- 论文原图一张：Sarathi-Serve 线性层耗时与 token 数（来源与校验见 `src/assets/vllm-scheduler/SOURCES.md`）。
- 已补读：Orca §3 S1/S2（iteration-level、selective batching）；默认值 `arg_utils.get_batch_defaults`（API server：≥160 GB 卡 16384/1024，H100/H200 8192/1024，其余 2048/256）。
- 未通读：Sarathi eval 章节，仅引用其自报的 2.6× 与 28.3×；FastGen 数字为博客自报。
- 待核实：抢占后重算是否命中 prefix cache（正文未展开）；官方调参文档的 `max_num_batched_tokens` 表述来自网页摘要，发布前对原文复核。
