vLLM 每一步只做一个决定：这一步让哪些请求各算多少个 token。这个决定要同时满足三件事：decode 的 batch 尽量大，新请求尽快开始，已在生成的请求不被打断。V1 调度器用一个 token budget 统一了这三件事，这也是 Orca 提出的 iteration-level scheduling 和 Sarathi-Serve 提出的 chunked prefill 演化到今天的形态。

本文先讲为什么 batch 内的 token 数是调度的核心量，再顺着 request-level、iteration-level、token budget 的演化，最后对照 vLLM v0.30.0 的源码。

## 一、Batch 内的 token 数

Transformer 每层的线性算子在一次迭代里要把全部权重从显存读一遍。batch 内 token 少时，耗时取决于这次读取；token 多到算力饱和后，耗时随 token 数线性增长。

::vllm-linear-time::

Sarathi-Serve 在 LLaMA2-70B 上测得，A100 的线性层耗时在 128 到 512 个 token 之间大体平缓（张量并行度越高越明显），之后才线性上升；理论拐点约 200 个 token，实测在张量并行度高时推迟到 500–600 个。[Agrawal et al., OSDI'24](https://arxiv.org/abs/2403.02310)

两类迭代落在曲线的两侧：

- **Decode** 每条请求一次只算一个 token，一个 batch 通常远低于拐点。加进更多请求几乎不增加耗时，吞吐随 batch 线性上升。
- **Prefill** 一条请求就带来几百到几千个 token，单独一条就把算力用满。

调度要做的，是让每次迭代的 token 数落在拐点附近：低了浪费算力，高了拖慢这一步里所有请求。

## 二、Request-level 与 iteration-level

早期的 serving 系统（如 FasterTransformer）以请求为粒度调度：凑齐一个 batch，跑到里面所有请求都结束，才接新请求。生成长度不一，这带来两个问题：[Orca, OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)

- **先结束的请求空转。** 已经结束的请求仍占着 batch 里的位置，继续参与计算。
- **后到的请求排队。** 新请求要等整个 batch 结束才能开始，结果也要等整个 batch 结束才能返回。

Orca 把调度粒度从请求改为**迭代**：调度器每一步选出要运行的请求，引擎只跑一次迭代就返回，请求在任何一步都能加入或退出。这就是后来所说的 continuous batching。

迭代级调度要求引擎能批处理任意一组请求，而这些请求的形状并不一致：有的在 prefill，长度各不相同，有的在 decode，各自位于不同位置。Orca 的解法是 **selective batching**：Linear、LayerNorm 这类不区分请求的算子，把所有请求的 token 展平成一个 $[\sum L, H]$ 的张量一起算；只有 attention 需要区分请求，就在它前后把张量拆开再合并。

这个形状沿用至今：vLLM 里 attention 的输入（如 `flash_attn_varlen_func` 的 `cu_seqlens_q`）就是展平的 token 序列加每个请求的边界。一个 batch 里 prefill 与 decode 可以共存，调度器只需要决定每个请求分多少个 token。

## 三、Prefill 与 decode 的干扰

迭代级调度留下了一个问题：新请求的 prefill 应该什么时候做。Orca 和早期 vLLM 都是 prefill 优先：请求一到就尽快把整段 prompt 算完。

::vllm-timelines::

上图三行对比同一组请求。请求级调度下，C 要等 {A, B} 全部结束，B 在第 3 步结束后仍占着位置。迭代级、prefill 优先时，C 到达后整段 prompt 作为一次迭代运行，这次迭代很长，A 和 B 在其间一个 token 也生成不了。

Sarathi-Serve 把这种现象称为 **generation stall**：正在生成的请求，其 token 间隔（TBT / TPOT）会因为别人的 prefill 出现秒级尖峰。它还测到，朴素地把 prefill 与 decode 混进同一个 batch，迭代耗时最多是纯 decode 迭代的 28.3 倍。另外，Orca 支持 prefill 与 decode 混批，早期 vLLM 只支持整批 prefill 或整批 decode，二者都会产生 stall。

## 四、统一的 token budget

Sarathi-Serve 的办法有两点：

- **Chunked prefill**：把长 prompt 切成块，每次迭代只算一块。
- **Stall-free batching**：设一个 token budget，每步先放入所有正在 decode 的请求，再放未完成的 prefill，最后才接新请求，剩余的 budget 给 prefill 块。

上图第三行是这种调度：C 的 16 个 token 分成 6、6、4 三块，与 A、B 的 decode 同批，每次迭代的耗时被限制在 2 以内。A 的最大 token 间隔从 5 降到 2，代价是 C 的 TTFT 从 4 增加到 5.25。Sarathi-Serve 自报，相对当时的 vLLM，Mistral-7B 单卡 A100 的服务容量高 2.6 倍。

vLLM V1 采用了同一思路，并进一步取消了 prefill 与 decode 的区分。源码注释写得很直接：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec.

每个请求只有两个数：已经算过的 token 数 `num_computed_tokens`，和需要算到的目标 `num_tokens_with_spec`（prompt、已生成的 token 与 speculative token 之和）。调度器每步让前者追上后者，decode 只是每步差 1 的特例。chunked prefill、prefix caching（命中的 token 直接算作已计算）和 speculative decoding 因此共用同一套逻辑。调度结果就是一张 `{request_id: num_tokens}` 的表。[vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)

::vllm-step-budget::

下面的代码简化自 v0.30.0 的 `Scheduler.schedule`：

```python
budget = max_num_batched_tokens
for req in running:                 # ① 先 running：decode 与未完成的 prefill
    n = min(req.num_tokens_with_spec - req.num_computed_tokens, budget)
    while not kv_cache_manager.allocate_slots(req, n):
        preempt(running[-1])        # KV block 不足，见第五节
    budget -= n
if not preempted:                   # 本步发生过抢占则不再接新请求
    while waiting and budget > 0 and len(running) < max_num_seqs:
        req = waiting.peek()        # ② 再 waiting：prefix cache 命中的部分不占 budget
        n = min(req.num_tokens - num_cached_tokens(req), budget)
        if not kv_cache_manager.allocate_slots(req, n): break
        budget -= n
        running.append(req)
```

顺序保证了 decode 不会被 prefill 挤掉：running 先拿走自己的 token，剩下的才给新请求。一个 prefill 被切到一半时留在 running 里，后续的块比新请求优先。`long_prefill_token_threshold` 可以额外限制单个请求单步的 token 数。

两个参数决定 budget 的大小：

- `max_num_batched_tokens`：单步 token 上限。
- `max_num_seqs`：同时运行的请求数上限。

它们的默认值按硬件和场景选择。以 OpenAI API server 为例，H100、H200 是 8192，B200 一类显存不小于 160 GB 的卡是 16384，其余（包括 A100 与消费级卡）是 2048；`max_num_seqs` 在这几档分别是 1024、1024 和 256。[vLLM `arg_utils`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py)

官方的调参文档给出的取向是：budget 小，token 间隔更稳；budget 大，首 token 更快、吞吐更高，并建议在大卡上设得高于 8192。[vLLM 优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/) 这与上面的时间线一致：budget 决定单步耗时的上限，也决定一个长 prompt 要几步才能算完。Sarathi-Serve 还指出，budget 取值要避开 tile quantization：chunk 为 257 个 token 比 256 个慢 32%。

## 五、显存不够：抢占

前面的调度都假定 KV block 够用。请求变长会不断申请新 block，池空了时，`allocate_slots` 返回失败，调度器必须让出一个请求：

- 默认的 FCFS 策略取 running 队列的最后一个，也就是最晚进入的请求；priority 策略取优先级最低的。
- 被选中的请求释放全部 block，`num_computed_tokens` 归零，回到 waiting 队首。它之后被重新调度时，需要把已有的 prompt 与已生成的 token 当作 prefill 重算，所以抢占的成本是一次额外的 prefill。
- 抢占发生的这一步，调度器不再接新请求。

V1 只保留 recompute 这一种恢复方式，V0 的 swap 已移除。官方文档把 recompute 称为默认且开销更低的方式，也提醒抢占会拉高端到端延迟，建议提高 `gpu_memory_utilization`、减小 `max_num_seqs` 或 `max_num_batched_tokens`，或增加并行度来避免。

## 六、相关系统与分歧

同一思路并不止一家。DeepSpeed-FastGen 的 Dynamic SplitFuse 同样把长 prompt 分块，并把短 prompt 拼合，让每次迭代的 token 数保持一致；其博客自报在 Llama-2 70B、4 张 A100 上，相对 vLLM 的吞吐最高约 2 倍。这些数字来自厂商自测，只作为方向参考。

DistServe 则不同意 chunked prefill 是根本解。它认为分块只是用 TTFT 换 TPOT，并不能消除 prefill 与 decode 的相互干扰；而且 chunk 很小时，prefill 与 decode 争用同一块 GPU 而变慢；chunk 接近饱和时，能搭载的 decode token 又所剩无几；每个后续块还要重新读取前面所有块的 KV，读取量是 $O(N^2)$ 块（不分块为 $O(N)$）。它的方案是把 prefill 与 decode 放到不同的 GPU 上。同一篇论文也承认，对只看吞吐、不在意延迟的离线场景，chunked prefill 更合适，因为它能把每个 batch 填到算力饱和。

两种做法的分歧在于，干扰是在同一块 GPU 上用调度控制，还是用硬件隔离。前者的成本是调参与 prefill 变慢，后者的成本是 KV cache 的跨卡传输与部署复杂度。

## 七、小结

调度器把“谁先跑、跑多少”化为一个预算问题：先满足正在 decode 的请求，再用剩余的 token 推进 prefill；KV block 是另一个约束，用尽时靠抢占重算来让出。budget 的取值，则是在首 token 延迟、token 间隔与吞吐之间的选择。

## 参考

- Gyeong-In Yu, et al. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
- Amey Agrawal, et al. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). OSDI 2024.
- Yinmin Zhong, et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). OSDI 2024.
- DeepSpeed Team. [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md). 2023.
- Woosuk Kwon, Zhuohan Li, et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
- vLLM Team. [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release) · [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/v1/core/sched/scheduler.py`、`vllm/config/scheduler.py`、`vllm/engine/arg_utils.py`。
