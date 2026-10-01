vLLM 的调度器每一步只回答一个问题：**这一步让哪些请求各算多少个 token**。答案要同时照顾三方：吞吐要求 batch 尽量大，新请求希望尽快开始，正在生成的请求希望不被打断。三者互相冲突，vLLM V1 用一个 token budget 统一了这三者的取舍。它是 Orca 的 iteration-level scheduling 与 Sarathi-Serve 的 chunked prefill 演化到今天的形态。下文按这条演化线展开，最后对照 vLLM v0.30.0 的源码。

## 一、Prefill 与 decode

一次生成请求分两段，性质完全不同：

- **Prefill**：把整段 prompt 一次送进模型，算出所有 prompt token 的 KV，并产出第一个输出 token。一次算几百到几千个 token，算力用得满。
- **Decode**：之后每步只生成一个 token，却要读一遍全部权重和这条请求的 KV。计算量小，受显存带宽限制。

单条 decode 读一遍权重只换来一个 token；把多条请求拼进同一个 batch，同一次读取就能服务更多 token，耗时几乎不变。这是 batching 的动机，剩下的问题是怎么拼。

衡量延迟用三个量，后文反复出现：

- **TTFT**（Time To First Token）：请求到达到第一个输出 token 的时间，主要由排队和 prefill 决定。
- **TBT**（Time Between Tokens）：相邻两个输出 token 的间隔，每个间隔各算一次。
- **TPOT**（Time Per Output Token）：TBT 的平均值。平均会把偶发的尖峰摊薄。

## 二、静态 batch 的空转

最直接的做法是凑齐一批，一起跑到全部结束，再接下一批。FasterTransformer 等早期系统就是这样。请求的生成长度差别很大，于是有两处浪费：[Orca, OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)

- **先结束的请求空转**：它仍占着 batch 里的位置，继续参与计算。
- **后到的请求排队**：新请求要等整个 batch 结束才能开始。

::vllm-batching-slots::

Orca 把调度粒度从“一批请求”改为**一次迭代**：调度器每步重新选出要运行的请求，引擎只跑一次迭代就返回，请求在任何一步都能加入或退出，空出的位置下一步就补上（上图右侧）。这称为 iteration-level scheduling，后来也叫 continuous batching。

它带来一个工程问题：同一个 batch 里有的请求在 prefill、长度各异，有的在 decode，张量形状对不齐。设这一步有三个请求，A 在 prefill、本步 5 个 token，B、C 在 decode、各 1 个 token。按传统的 `[batch, seq_len, H]` 组织，三者都要 pad 到 5，B、C 各有 4 个位置在白算。

Orca 的 **selective batching** 不再按请求分行，而是把 7 个 token 首尾相接排成 $[\sum L, H] = [7, H]$ 的一长条。

::vllm-flattened-batch::

Linear、LayerNorm、MLP 对每个 token 的计算彼此独立，某一行属于哪个请求不影响结果，所以直接在这一长条上算，一次矩阵乘法覆盖所有请求，也没有 padding。Attention 不同：A 的 token 只能看 A 自己的上文，B 要读 B 的 KV cache，跨请求的交互必须隔开。最初的做法是在 attention 前按请求边界把张量拆开，各自计算后再拼回长条，交给下一层。

今天的实现不再物理拆分，只把边界交给 kernel。`flash_attn_varlen_func` 的 `cu_seqlens_q = [0, 5, 6, 7]` 就是每个请求在长条里的起止位置，kernel 按段各自计算 attention，张量始终是 `[7, H]`。对它来说，prefill 只是较长的一段，decode 是长度为 1 的一段，所以两者可以共存于一个 batch，调度器只需要决定每个请求分多少个 token。

## 三、Prefill 与 decode 的干扰

迭代级调度留下一个问题：**新请求的 prefill 什么时候插进来**。回答它，先要知道一次迭代有多慢。

::vllm-linear-time::

线性算子每次迭代都要把全部权重读一遍。token 少时，耗时由这次读取决定，曲线基本平坦；token 多到算力饱和后，耗时随 token 数线性增长。Sarathi-Serve 在 LLaMA2-70B、A100 上测得，线性层耗时在 128 到 512 个 token 之间大体平缓，理论拐点约 200 个 token，张量并行度高时实测推迟到 500–600 个。[Agrawal et al., OSDI'24](https://arxiv.org/abs/2403.02310)

两类迭代落在拐点两侧。Decode batch 远在拐点左侧，多放几条请求几乎不增加耗时；一个长 prompt 的 prefill 单独就越过拐点，迭代耗时随 prompt 长度增长。

早期的迭代级调度都是 prefill 优先：请求一到，尽快把整段 prompt 算完。

::vllm-timelines::

上图三行对比同一组请求：

- **请求级**：C 要等 A、B 全部结束才能开始，B 结束后仍占着位置。
- **迭代级、prefill 优先**：C 的整段 prompt 作为一次迭代运行。这一步很长，A、B 在其间一个 token 也生成不了，A 的最大 TBT 升到 5。
- **迭代级、token budget**：见下一节。

Sarathi-Serve 把第二行的现象称为 **generation stall**：正在生成的请求，TBT 因为别人的 prefill 出现秒级尖峰，而 TPOT 会把这种尖峰摊薄。它还测到，朴素地把 prefill 与 decode 混进同一个 batch，迭代耗时最多是纯 decode 迭代的 28.3 倍。

## 四、统一的 token budget

Sarathi-Serve 的办法有两点：

- **Chunked prefill**：把长 prompt 切成块，每次迭代只算一块。
- **Stall-free batching**：每步设一个 token 上限，按顺序填入：先放所有正在 decode 的请求，再放未完成的 prefill，最后才接新请求。

时间线的第三行就是这种调度：C 的 16 个 token 切成 6、6、4 三块，与 A、B 的 decode 同批，每次迭代耗时不超过 2。A 的最大 TBT 从 5 降到 2，代价是 C 的 TTFT 从 4 增加到 5.25。Sarathi-Serve 自报，相对当时的 vLLM，Mistral-7B 单卡 A100 的服务容量高 2.6 倍。

### V1：不再区分 prefill 与 decode

vLLM V1 采用同一思路，并进一步取消了两个阶段的区分。源码注释写得很直接：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec.

每个请求只有两个数：已经算过的 token 数 `num_computed_tokens`，和需要算到的目标 `num_tokens_with_spec`（prompt、已生成的 token 与 speculative token 之和）。调度器每步让前者追赶后者，本步最多算两者之差，再受 budget 限制。原本各自一套逻辑的功能，都成了这个框架里的特例。以一个 prompt 为 30 个 token 的请求为例：

| 情形 | `num_computed_tokens` | `num_tokens_with_spec` | 本步最多算 |
| --- | --- | --- | --- |
| 新请求，无 prefix cache 命中 | 0 | 30 | 30 |
| 新请求，prefix cache 命中 8 个 | 8 | 30 | 22 |
| Chunked prefill，已算 16 个 | 16 | 30 | 14 |
| Decode，已生成 5 个 token | 34 | 35 | 1 |
| 同上，另有 3 个 speculative token | 34 | 38 | 4 |
| 被抢占后重新调度 | 0 | 35 | 35 |

Decode 那一行差 1，是因为上一步刚采样出的 token 还没有算 KV。

调度结果就是一张 `{request_id: num_tokens}` 的表。[vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)

### 一步的分配顺序

::vllm-step-budget::

下面的代码简化自 v0.30.0 的 `Scheduler.schedule`：

```python
budget = max_num_batched_tokens
# ① 先 running：decode 与未完成的 prefill
for req in running:
    need = req.num_tokens_with_spec - req.num_computed_tokens
    n = min(need, budget)
    # KV block 不足时抢占队尾；队尾若是 req 自己，本步不再调度它
    while not kv_cache_manager.allocate_slots(req, n):
        preempt(running[-1])
    budget -= n
# ② 再 waiting：本步发生过抢占则不接新请求
if not preempted:
    while waiting and budget > 0 and len(running) < max_num_seqs:
        req = waiting.peek()
        # prefix cache 命中的部分不占 budget
        need = req.num_tokens - num_cached_tokens(req)
        n = min(need, budget)
        if not kv_cache_manager.allocate_slots(req, n):
            break
        budget -= n
        running.append(req)
```

这个顺序保证 decode 不会被新请求挤掉：running 先拿走自己的 token，剩下的才给 waiting。切到一半的 prefill 也留在 running 里，后续的块比新请求优先，不会被无限推迟。`long_prefill_token_threshold` 可以额外限制单个请求单步的 token 数。

### Budget 的大小

Budget 由两个参数决定：单步 token 上限 `max_num_batched_tokens`，和同时运行的请求数上限 `max_num_seqs`。默认值按硬件分档，以 OpenAI API server 为例：[vLLM `arg_utils`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py)

| 硬件 | `max_num_batched_tokens` | `max_num_seqs` |
| --- | --- | --- |
| H100、H200 | 8192 | 1024 |
| B200 等显存 ≥ 160 GB 的卡 | 16384 | 1024 |
| 其余（含 A100 与消费级卡） | 2048 | 256 |

取值是一组权衡，与上面的时间线一致。Budget 小，单步耗时短，TBT 稳，但长 prompt 要分更多步，TTFT 变长；budget 大则相反，TTFT 更短、吞吐更高，decode 的 TBT 变大。官方建议在大卡上设得高于 8192。[vLLM 优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/) 另外，取值要避开 tile quantization：GPU 按固定大小的 tile 切分矩阵乘法，Sarathi-Serve 测得 chunk 为 257 个 token 比 256 个慢 32%。

## 五、显存不够：抢占

Budget 约束的是计算量，另一个约束是 KV cache 的显存。请求变长会不断申请新 block，池空时 `allocate_slots` 失败，调度器必须让出一个请求：

- **选谁**：默认的 FCFS 策略取 running 队尾，即最晚进入的请求；priority 策略取优先级最低的。
- **怎么让**：被选中的请求释放全部 block，`num_computed_tokens` 归零，回到 waiting 队首。重新调度时，已有的 prompt 与已生成的 token 要当作 prefill 重算一遍，这就是抢占的代价。
- **本步不接新请求**：显存已经紧张，接新请求只会引发更多抢占。

V1 只保留 recompute 这一种恢复方式，V0 把 KV 换出到 CPU 内存的 swap 已移除。官方文档把 recompute 称为默认且开销更低的方式，也提醒抢占会拉高端到端延迟，建议提高 `gpu_memory_utilization`、减小 `max_num_seqs` 或 `max_num_batched_tokens`，或增加并行度来避免。

## 六、小结

调度器把“谁先跑、跑多少”化为一个预算问题：先满足正在 decode 的请求，再用剩余 token 推进 prefill。KV block 是另一个约束，用尽时靠抢占加重算来让出。Budget 的取值，则是在 TTFT、TBT 与吞吐之间的选择。

## 参考

- Gyeong-In Yu, et al. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
- Amey Agrawal, et al. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). OSDI 2024.
- Woosuk Kwon, Zhuohan Li, et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
- vLLM Team. [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release) · [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/v1/core/sched/scheduler.py`、`vllm/config/scheduler.py`、`vllm/engine/arg_utils.py`。
