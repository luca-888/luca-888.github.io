LLM serving 的吞吐，取决于 GPU 上能同时解码多少条请求；而限制这个数字的，通常是 KV cache 占用的显存。PagedAttention 把操作系统的分页搬进 KV cache：显存切成固定大小的 block，按需分配，每个请求用一张 block table 记录自己的 token 存在哪些 block 里。它是 vLLM 的起点。[Kwon et al., SOSP'23](https://arxiv.org/abs/2309.06180)

本文先讲它要解决的浪费和分页机制，再对照 vLLM v0.30.0 的源码，看这套机制今天的形态。

## 一、KV cache 决定并发

Decode 每一步只为每条请求生成一个 token，却要把整个模型的权重从显存读一遍，算力大部分闲置。把多条请求拼进同一个 batch，同一次权重读取服务更多 token，吞吐随 batch 增长，直到计算成为瓶颈。batch 分摊的只是权重：每条请求仍要读自己的全部 KV，上下文越长，这部分占比越高，加大 batch 的收益也就越小。

Batch 能开多大，取决于 KV cache 能装下多少 token。每个 token 在每一层都要保存 K 和 V：

$$
\text{bytes per token} = 2 \times L \times H_{kv} \times d \times \text{bytes per element}
$$

以 Llama-3.1-8B 为例，$L=32$、$H_{kv}=8$、$d=128$，BF16 下每个 token 占 128 KiB，一条 8K 上下文的请求约 1 GiB。权重之外剩下的显存能装多少 token，就决定了能并发多少请求。

KV cache 的难点在于形状：请求的输出长度要等到生成 `<eos>` 才知道，KV 每步增长一个 token，请求结束时整体释放。显存管理要面对的，是一批长度未知、不断变长、随时结束的数组。

## 二、连续分配的三种浪费

PagedAttention 之前的 serving 系统（FasterTransformer、Orca）沿用深度学习框架的习惯，把一个请求的 KV cache 存成一段连续显存。输出长度未知，只能按最大可能长度一次预留。

::vllm-allocation-compare::

左侧的浪费分为三种：

- **Reservation**：之后会写入，但在请求的整个生命周期里一直被占着，其他请求用不了。
- **Internal fragmentation**：超出实际长度的预留，永远不会写入，请求结束时才知道。
- **External fragmentation**：不同大小的预留段之间留下的空洞，装不下新的一整段。

右侧只改分配方式：请求按需拿 block，block 不必相邻。同样 35 个 token，占用从 128 个 slot 降到 40 个，剩下的 22 个 block 可以直接给新请求。

论文在 OPT-13B 上统计了真实 trace 的 KV cache 组成。Orca 没有开源，作者按其论文复现了三个变体，对输出长度的预估越来越准：Max 按模型最大长度 2048 预留，Pow2 最多多留一倍，Oracle 预先知道真实输出长度。

::vllm-paper-breakdown::

即使是无法实现的 Oracle，有效利用率也只有 38.2%：它仍然要在请求开始时预留全部输出空间，预留段之间也有外部碎片。vLLM 做到 96.3%，因为分页把浪费限制在每个请求的最后一个 block 里。

## 三、分页：block table 与按需分配

论文用操作系统的虚拟内存来类比：block 对应 page，token 对应 byte，请求对应 process。

- **Block**：启动时把可用于 KV cache 的显存一次性切成等大的 block，每个 block 存 $B$ 个 token 的 K/V。多数情况下 vLLM 默认 $B=16$，部分 backend 与混合架构模型会选更大的 block。
- **Block table**：每个请求一张表，第 $i$ 项是第 $i$ 个逻辑 block 对应的物理 block 编号。
- **按需分配**：最后一个 block 写满后，才分配下一个。

每个请求最多浪费最后一个 block 里的 $B-1$ 个 slot。所有 block 大小相同，任何空闲 block 都能分给任何请求，外部碎片也就不存在了。同一个 block 编号在每一层各对应一块显存，一个请求只需要一张 block table；tensor parallel 时各卡用同样的编号，只保存自己那部分 head。

::vllm-block-table::

每层 attention 对 KV cache 做两件事。**写入**：本步新 token 的 K/V 按 `slot_mapping` 散写进池里，slot 由上图的公式算出。**读取**：attention kernel 拿到 block table 和每个请求的长度，逐 block 读取历史 K/V。

分工上，地址的分配与翻译都在引擎里，写入也用 vLLM 自带的 kernel，只有读取和 attention 计算交给 attention 库。在 v0.30.0 中，`slot_mapping` 由一个 Triton kernel 为整个 batch 批量计算，写入调用 `reshape_and_cache_flash`，读取把 `block_table` 直接交给 FlashAttention（vLLM 维护的 fork）：

```python
reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)
flash_attn_varlen_func(q=query, k=key_cache, v=value_cache, cu_seqlens_q=cu_seqlens_q,
                       seqused_k=seq_lens, block_table=block_table, causal=True, ...)
```

分配发生在调度器里。V1 调度器每一步只决定“每个请求本步算多少个 token”，然后为它们补足 block。下面的代码简化自 `KVCacheManager.allocate_slots`：

```python
def allocate_slots(req, num_new_tokens):
    num_needed = cdiv(req.num_computed_tokens + num_new_tokens, B) - len(req.blocks)
    if num_needed > block_pool.get_num_free_blocks():
        return None  # 调度器据此抢占其他请求，见第六节
    req.blocks += block_pool.get_new_blocks(num_needed)
```

Block 池的大小在启动时确定：vLLM 先按单步最多可调度的 token 数做一次 dummy forward，测出权重和 activation 的峰值占用；再从 `gpu_memory_utilization`（v0.30.0 默认 0.92）对应的显存里扣掉这些占用，余下的按每个 block 在所有层的字节数切分。[vLLM `CacheConfig`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py)

## 四、分页之后的 attention

分页不改变数学，只是把对历史 token 的求和按 block 分组：外层遍历 block，内层遍历 block 里的 token。

$$
a_{is} = \frac{\exp\!\left(q_i^\top k_s / \sqrt d\right)}{\sum_{t=1}^{\lceil i/B\rceil} \sum_{s' \in \text{block } t} \exp\!\left(q_i^\top k_{s'} / \sqrt d\right)}, \qquad
o_i = \sum_{j=1}^{\lceil i/B\rceil} \sum_{s \in \text{block } j} a_{is}\, v_s
$$

Softmax 的分母跨越所有 block，这和 FlashAttention 按 tile 遍历 K/V、用 online softmax 累积是同一件事。区别只在于下一个 K/V tile 的地址来自查表，而不是连续偏移。

代价是间接寻址。论文时期的 vLLM 自己写了 paged attention kernel（改自 FasterTransformer，一个 warp 读一个 block），比 FasterTransformer 的 attention kernel 慢 20–26%。这部分开销只落在 attention 上，而 batch 变大带来的是 2–4 倍的端到端吞吐。

**Block size** 两头受限：太小，每次读取的数据太少，查表和分支的开销相对变大；太大，最后一个 block 的浪费增加，能共享的粒度也变粗。论文的消融显示，ShareGPT 这类长序列负载在 16 到 128 之间最好，Alpaca 这类短序列负载在 16、32 最好，默认值因此定为 16。

今天，vLLM 已经不再使用论文里那套 kernel：[设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)把它标为历史，v0.25.0 起源码中也[删除了这套 kernel](https://github.com/vllm-project/vllm/pull/47361)。删除的只是读取分页 KV 的专用 kernel，block 与 block table 的管理仍是 KV cache 的核心。FlashAttention 从 2.5 版开始接受 `block_table` 参数，但上游要求 page 大小为 256 的倍数；vLLM 用自己维护的 fork，支持 16 的倍数。vLLM 的 FlashAttention、FlashInfer、Triton 等 attention backend 都直接读分页 KV cache。分页已经成为 attention kernel 的标准输入格式之一。

这条路线也有反对意见。[vAttention](https://arxiv.org/abs/2405.04437)（ASPLOS'25）认为，分页把虚拟地址到物理地址的转换搬进了每个 attention kernel：每个新 kernel 都要额外支持 block table，而且 prefill 时 paged 版本比连续版本最多慢 37%（FlashAttention-2）和 42%（FlashInfer）。decode 受显存带宽限制，这部分开销被掩盖，两者相当。vAttention 的做法是用 CUDA 的虚拟内存 API，让 KV cache 在虚拟地址上保持连续，物理页按需映射，未经修改的 kernel 可以直接使用。[FlashInfer](https://arxiv.org/abs/2501.01005) 走的是另一个方向：把 block table 看作 block-sparse 矩阵，一套 kernel 统一处理分页、radix tree 等多种 KV 布局。

三者都认同 KV cache 必须按需分配，分歧在于地址转换由谁来做：attention kernel 自己查表，还是交给 GPU 的 MMU。

## 五、共享：引用计数与 prefix caching

Block table 把逻辑位置和物理存储分开之后，不同请求的表可以指向同一个物理 block。每个物理 block 记一个引用计数 `ref_cnt`，归零才回收。

论文用这一点处理 parallel sampling 和 beam search：同一个 prompt 的多个输出共享 prompt 的 block，谁要写一个共享 block，就先复制一份（copy-on-write）。在 OPT-13B 与 Alpaca trace 上，parallel sampling 节省 6.1–9.8% 的 KV 显存，beam search 节省 37.6–55.2%；ShareGPT 上分别为 16.2–30.5% 和 44.3–66.3%。

V1 把共享统一成 **prefix caching**，默认开启：

::vllm-prefix-sharing::

- **Block hash**：block 写满后计算 hash，输入包括父 block 的 hash、本 block 的 token id，以及 LoRA、多模态输入、cache salt 等额外信息。hash 是链式的，相同即意味着整个前缀相同。
- **命中**：新请求先按 hash 逐块查找已缓存的 block，命中的直接引用、`ref_cnt` 加一，只计算未命中的部分。
- **只缓存写满的 block**：共享的 block 不会再被写入，不需要 copy-on-write。
- **Parallel sampling** 被拆成 n 个子请求，由 prefix caching 负责复用 prompt 的 block；同一 hash 允许对应多个物理 block，子请求在同一步调度时是否各算一份，取决于调度时机。beam search 移到 API 层，由多个请求组合实现。

回收与缓存是同一批 block。`ref_cnt` 归零的 block 不会立即清空，而是回到一个双向链表 free queue：没有 hash 的放到队首，最先被复用；有 hash 的放到队尾，按 LRU 保留，期间仍能被命中。分配时从队首取，只有真正取到一个带 hash 的 block 时，才把它从缓存表里删除。[vLLM `BlockPool`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py)

因此缓存不额外占用容量，空闲 block 本身就是缓存。V1 发布时，vLLM 团队给出的数据是：命中率为 0 时，开启 prefix caching 的吞吐下降不到 1%，于是默认开启。[vLLM V1](https://vllm.ai/blog/2025-01-27-v1-alpha-release)

SGLang 的 [RadixAttention](https://arxiv.org/abs/2312.07104) 更早实现了自动前缀复用，存储同样按需分配、不要求连续，区别在于前缀的组织方式：它把缓存过的 token 序列放进一棵 radix tree，按 token 匹配最长公共前缀，从没有请求引用的叶子开始按淘汰策略（LRU、LFU 等可选）回收。SGLang 起初以单个 token 为分配粒度，命中不受 block 边界限制，代价是 kernel 按 token 间接寻址；现在也支持更大的 page。两种方案的分歧不在要不要分页，而在分配粒度与前缀索引：vLLM 用 block 加 hash 表，SGLang 用 radix tree。

## 六、显存不够：抢占

请求变长到 block 用尽时，调度器必须让出一些请求。论文采用 all-or-nothing：一个请求的 block 要么全留，要么全部驱逐，因为 decode 每一步都要读它的全部 KV。被驱逐的请求有两种恢复方式：

- **Swap**：把 block 拷到 CPU 内存，之后拷回。
- **Recompute**：把已生成的 token 拼到 prompt 后面，下次调度时用一次 prefill 重新算出 KV。

论文的测量中，block 小时 swap 被大量小块 PCIe 传输拖慢；block size 在 16 到 64 之间时，两者端到端表现相当。

V1 只保留 recompute。被抢占的请求释放全部 block，`num_computed_tokens` 归零，回到等待队列。结合上一节的回收顺序，它刚释放的、写满的 block 带着 hash 排在 free queue 队尾，重新调度时往往还能命中，recompute 实际只需重算被覆盖的部分。这一点是从源码行为推出的分析，命中多少取决于期间的显存压力。

## 七、小结

PagedAttention 的核心是一层间接：请求看到连续的逻辑 block，显存里是任意摆放的物理 block，中间由 block table 连接。容量、共享与缓存都来自这层间接，代价是 attention kernel 要查表。

本文没有展开的两个问题同样建立在 block 之上：调度器如何在 block 预算内决定每步处理哪些请求、多少 token；以及多卡并行时 KV cache 如何切分与传输。

## 参考

- Gyeong-In Yu, et al. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
- Woosuk Kwon, Zhuohan Li, et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
- vLLM Team. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm). 2023.
- vLLM Team. [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) · [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release).
- Aleksa Gordić. [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm). 2025.
- Ramya Prabhu, et al. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437). ASPLOS 2025.
- Lianmin Zheng, et al. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). NeurIPS 2024.
- Tri Dao, et al. [FlashAttention](https://github.com/Dao-AILab/flash-attention) · [vLLM 维护的 fork](https://github.com/vllm-project/flash-attention).
- Zihao Ye, et al. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005). MLSys 2025.
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/v1/core/`、`vllm/v1/worker/block_table.py`、`vllm/v1/attention/backends/flash_attn.py`。
