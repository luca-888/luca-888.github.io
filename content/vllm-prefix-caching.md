多轮对话的第十轮，prompt 里有九轮历史；agent 每调用一次工具，都要把前面的全部上下文再送一遍；同一个服务的所有请求，开头都是同一段几千 token 的 system prompt。这些 token 的 KV 早就算过，**prefix caching** 让新请求直接引用已有的 KV，只算后面不同的部分。

它把 KV cache 从每个请求私有的工作内存，变成按内容寻址、跨请求共享的缓存。缓存是否有效取决于三件事：**找得到**（用什么做索引、按多大粒度匹配），**还留着**（显存紧张时先淘汰谁），**落得准**（请求能否被调度到缓存所在的位置）。下文先讲 vLLM 的做法，再对照 SGLang 的 RadixAttention，最后讨论单卡之外的扩展和它的边界。源码对照 vLLM v0.30.0 与 SGLang main（2026-09-29）。

## 一、复用的是什么

Decoder 的 attention 是因果的：第 $i$ 个 token 的 K、V 只取决于它和它之前的 token（以及模型权重）。两个请求只要前 $n$ 个 token 完全相同，这 $n$ 个位置的 KV 就逐位相同，第二个请求可以跳过这部分的 prefill。

省下的是 prefill 的计算。Prefill 的计算量随 prompt 长度增长，attention 部分还是平方关系，所以命中的前缀越长，TTFT 下降越多。在 vLLM 里，命中的 token 也不占本步的 token budget，同样的预算能接纳更多新请求。Decode 不受影响，每步仍要读完整的 KV。

适合的负载有共同特征：大量请求共享一段较长、且位于开头的内容。

- **共享 system prompt**：system prompt、工具定义、few-shot 示例。
- **多轮对话**：第 $n$ 轮的 prompt 包含前 $n-1$ 轮的全部内容。
- **Agent 循环**：每一步在上下文末尾追加工具结果，前缀持续增长。
- **长文档问答**：同一份文档配不同的问题。
- **并行采样**：同一 prompt 生成 $n$ 个回答。

SGLang 在 Chatbot Arena 上部署一个月，前缀命中率在 Vicuna-33B 上为 74.1%，在 LLaVA-Next-34B 上为 52.4%；Vicuna-33B 的首 token 延迟平均降低 1.7 倍。命中主要来自 system message、反复出现的示例图片和多轮对话历史。[Zheng et al., NeurIPS'24](https://arxiv.org/abs/2312.07104)

限制也来自同一个前提：只有**前缀**相同才能复用。同一段文字出现在 prompt 中间，位置和上文都不同，KV 也就不同。

## 二、vLLM：用 hash 链给 block 编号

vLLM 的 KV cache 按 block 存储，一个 block 装固定数目的 token（默认 16），每个请求用一张 block table 记录自己的 KV 放在哪些物理 block 里。Prefix caching 要解决的是：新请求到来时，怎样快速知道哪些 block 已经算过。

vLLM 给每个**写满的 block** 算一个 hash，输入是三部分：[vLLM 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)

$$
h_i = \mathrm{hash}(h_{i-1},\ \text{block}_i \text{ 的 token id},\ \text{extra keys})
$$

父 block 的 hash $h_{i-1}$ 把整个前缀串成一条链：$h_i$ 相同，意味着从第一个 token 到第 $i$ 个 block 末尾完全相同，而不只是这一块的 token 相同。所有已缓存的 block 放在一张 `hash → block` 的表里。新请求逐块计算 hash、逐块查表，遇到第一个未命中的 block 就停止；链式 hash 保证后面的 block 不可能再命中。

::prefix-hash-chain::

图中的例子说明了按 block 匹配的代价。B 与 A 的前 10 个 token 相同，但第三个 block 只有前两个 token 相同，hash 不同，所以只命中 8 个 token，`i`、`j` 虽然一样也要重算。每个分叉点最多损失 block size 减一个 token。

### 什么会进入 hash

Token id 相同不代表 KV 相同，还需要 extra keys 区分：

- **LoRA**：不同 adapter 算出的 KV 不同，adapter 名字进入 hash。
- **多模态输入**：图片在 prompt 里展开成一串相同的占位 token，只看 token id 无法区分两张图，因此加入图片内容的 hash 和它在 block 内的偏移。
- **`cache_salt`**：请求可以携带一个 salt，只加在第一个 block 的 hash 里。由于 hash 是链式的，整条链都随之改变，只有 salt 相同的请求才能互相命中，用于多租户隔离（见第七节）。

Hash 函数默认是 SHA-256。设计文档提到，早期版本的 hash 不保证无碰撞，现在默认改为 SHA-256。碰撞的后果不是性能下降，而是一个请求静默地用上另一个前缀的 KV，输出错误，在多租户场景下还可能泄露数据。另有选项 `sha256_cbor`，用规范的 CBOR 序列化，跨进程、跨语言可复现，外部系统需要按 hash 索引 block 时使用（见第六节）。

### 最后一个 token 总要重算

即使整个 prompt 都在缓存里，最后一个 token 也必须重新过一遍模型，否则拿不到用于采样的 logits。vLLM 因此把命中长度上限设为 prompt 长度减一，而命中长度又必须按 block 对齐，结果是整个最后一个 block 都要重算。例如 prompt 恰好 32 个 token、全部已缓存，block size 为 16，实际命中 16 个，重算 16 个。源码注释写明了这个限制，并说以后去掉它能略微提升性能。[`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/kv_cache_manager.py)

### 什么时候进入缓存

Block 在分配时就登记：调度器为本步要计算的 token 分配 slot，其中凡是会被写满的 block，立即算 hash 放进表里，不等计算完成。同一步里排在后面的请求因此能命中前面请求正在计算的 block，两个同时到达、前缀相同的请求不必各算一份。Decode 生成的 token 也一样，写满一个 block 就进入缓存，所以上一轮的回答可以被下一轮命中。

## 三、空闲 block 即缓存

Prefix caching 没有单独的缓存区。每个物理 block 有引用计数 `ref_cnt`，表示有几个运行中的请求在用它。请求结束后，`ref_cnt` 归零的 block 回到空闲队列 free queue，但**不清空内容，也不从 hash 表里删除**。在被重新分配之前，它既是空闲的，也仍然可以被命中。

### 淘汰与救回

Free queue 是一个双向链表，两个操作决定了 block 的命运：[vLLM `BlockPool`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py)

- **分配**：从队首取 block。如果取到的 block 带 hash，这时才把它从 hash 表中删掉，这就是淘汰。淘汰顺序因此就是队列顺序。
- **命中**：新请求命中一个空闲 block 时，把它从队列中间摘出来、`ref_cnt` 加一，它就不会被分配走。双向链表让这一步是 $O(1)$。

### 归还顺序

既然队列顺序就是淘汰顺序，请求结束时如何归还就决定了谁先被淘汰。请求的 block 按**逆序**归还：

- **没有 hash 的 block**（没写满的最后一块）放到队首，最先被重新使用，因为它不可能被命中。
- **有 hash 的 block** 放到队尾。逆序意味着请求越靠后的 block 越靠近队首、越先被淘汰。越靠后的 block 包含的前缀越长，被别的请求命中的可能越小。

::prefix-free-queue::

整体效果是按请求的 LRU：先结束的请求整体靠近队首，同一请求内部从尾部开始淘汰。共享的前缀 block 只要还有请求在用，就不会进入队列；即使进入了，也会因为被频繁命中而不断从队列中移出，重新排到后面。

从尾部开始淘汰还有一个作用：hash 链的查找在第一个未命中处停止，如果父 block 先被淘汰，它后面的子 block 即使还在，也再也查不到，只能白白占着空闲位置。逆序归还使同一请求内的子 block 总比父 block 先淘汰。

这种设计的代价很低：没有命中时，唯一的额外工作是算 hash 和查表。vLLM 团队在 V1 发布时给出的数据是，命中率为 0 时开启 prefix caching，吞吐下降不到 1%，因此 V1 默认开启。[vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)

## 四、命中之后：调度与多轮对话

命中的 token 数就是请求的起点：调度器从 `num_computed_tokens` 等于命中长度处开始调度，本步只为剩下的 token 分配 budget 和新 block。命中的 block 通过引用计数挂到请求的 block table 前面，attention kernel 读它们时，和读自己算出来的 block 没有区别。

vLLM 的调度器按 FCFS 或优先级排队，**不会为了提高命中率重排请求**。命中多少，取决于请求到来时缓存里恰好还剩什么。这一点和 SGLang 不同，见下一节。

多轮对话能否命中，还取决于一个容易忽略的环节：服务端每轮都用 chat template 把完整的 messages 重新拼成 token。上一轮生成的 token 已经进入缓存，但只有重新拼出来的 token 与之逐个相同，才能命中。

以 Qwen3 为例，它的 chat template 会删掉最近一条用户消息之前各轮回答里的 `<think>` 内容。于是第 $n+1$ 轮的 prompt 在第 $n$ 轮回答处与缓存分叉：生成时回答以 thinking 开头，重新拼接时 thinking 已被删掉。前 $n-1$ 轮的历史在第 $n$ 轮的 prompt 里已经是删过 thinking 的形式，仍然可以命中；需要重算的是第 $n$ 轮的回答和新的用户消息。同一个用户问题内的多步工具调用则保留 thinking，前缀一直有效。[Qwen3 chat template](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json)

## 五、SGLang：radix tree

SGLang 的 [RadixAttention](https://arxiv.org/abs/2312.07104) 更早实现了自动的前缀复用，索引换成了一棵 radix tree：每条边上是一段任意长度的 token 序列，节点指向这段 token 的 KV 存储位置。新请求沿树向下匹配最长公共前缀；插入时如果在一条边的中间分叉，就把这条边拆成两段。

::prefix-block-vs-radix::

其余机制与 vLLM 对应：

- **按 token 匹配**：论文实现的 page 大小为 1 个 token，命中不受 block 边界限制。当前版本默认仍为 1，部分 backend 会改用更大的 page，这时匹配结果按 page 向下取整。
- **缓存与运行共享内存池**：没有独立的缓存区，节点上的引用计数标记运行中的请求，只有计数为零的节点才能淘汰。
- **从叶子淘汰**：默认 LRU，从最久未用的叶子开始，祖先节点在它的子节点全部淘汰、自己成为叶子之后才可能被淘汰。

### Cache-aware scheduling

两者最大的区别是调度。SGLang 论文提出按**最长前缀匹配**（LPM）重排等待队列：命中越长的请求越先运行。论文证明，离线处理一批请求时，按 radix tree 的深度优先顺序访问可以达到最优命中率（缓存容量不小于最长请求），而 LPM 等价于深度优先。直观地说，共享同一前缀的请求被连续处理，前缀只需计算一次；若在不相关的请求之间来回切换，缓存会被反复冲刷。论文的消融实验中，换成 FCFS 或随机顺序都会明显降低性能，在各项 benchmark 上 LPM 的命中率平均达到最优值的 96%。

论文也指出了 LPM 的问题：贪心的重排可能让命中短的请求长期得不到调度。当前版本的做法很谨慎：

- **默认策略是 FCFS**。调参文档建议共享前缀多的负载改用 `--schedule-policy lpm`，并说明它会带来更多调度开销。
- **队列长时自动回退**：等待队列超过 128 个请求时，LPM 退回 FCFS，避免逐个匹配和排序的开销。
- **队列内去重**：多个等待中的请求共享一段尚未缓存的前缀时，先只调度其中一个，其余暂时后移，等前缀进入缓存后再命中。vLLM 通过“分配即登记”在同一步内实现了类似的效果。

[SGLang `schedule_policy.py`](https://github.com/sgl-project/sglang/blob/8854857/python/sglang/srt/managers/schedule_policy.py)

### 两种设计的对照

| | vLLM | SGLang |
| --- | --- | --- |
| 索引 | 链式 hash → block 的哈希表 | radix tree |
| 匹配粒度 | block（默认 16 token） | token（默认），部分 backend 为 page |
| 淘汰 | free queue：按请求 LRU，同一请求从尾部开始 | 从叶子开始，默认 LRU，可选多种策略 |
| 调度 | FCFS / 优先级，不因命中重排 | 默认 FCFS，可选 LPM 等 cache-aware 策略 |
| 无命中时的开销 | 吞吐下降 <1%（官方数据） | 树操作约占总时间 0.3%（论文：74.3 s 中 0.2 s） |

Radix tree 保存了前缀之间的结构，能按 token 精确匹配，也能回答“哪些请求共享同一前缀”，这是 LPM 调度的前提。Hash 表只能逐块回答“这一块有没有”。

## 六、单卡之外：容量与路由

单卡上的缓存容量就是空闲 block 的数量。负载一高，运行中的请求占满显存，缓存随之缩小，命中率恰好在最需要的时候下降。一种办法是把缓存扩展到更大、更慢的存储：

- **vLLM**：`OffloadingConnector` 把写满的 block 异步拷到 CPU 的 pinned memory，还可以再接更慢的外部存储层；命中时按需拷回 GPU。设置 `kv_offloading_size` 即可启用，也可以换成 LMCache 作为后端。[vLLM KV offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- **SGLang**：HiCache 把 GPU、host 内存和外部存储（本地文件、Mooncake、3FS 等）组织成三级。

另一个问题出在多副本部署上。前缀缓存只存在于算过它的那个副本上，按轮询或负载分发请求时，同一会话的后续请求多半落到别的副本，每个副本的命中率都很低。解决办法是在路由层感知缓存：

- **近似路由**：router 自己记账，记录把哪些前缀发给了哪个副本，据此推测缓存在哪。副本不需要配合，但副本淘汰缓存时 router 并不知道，记录会过期，请求可能落在已经没有缓存的副本上。SGLang 论文附录的数据并行方案就是这一类：在 router 上维护一棵汇总各 worker 前缀的树。
- **精确路由**：由副本主动汇报。每个副本在 block 写入和淘汰时上报事件（KV events，携带 block hash），路由层据此维护全局的 `hash → 副本` 索引，反映的是缓存当前的真实状态，包括淘汰。路由层与副本是不同进程，要用请求算出的 hash 去查副本上报的 hash，两边必须算出相同的值，这就是 `sha256_cbor` 这类可复现 hash 的用途。

llm-d 团队在 8 个 vLLM 副本（每个 2 张 H100，Qwen-32B）上测试多租户长上下文负载，自报的 P90 TTFT：精确路由 0.54 s，近似路由 31 s，随机分发 93 s。[llm-d blog](https://llm-d.ai/blog/kvcache-wins-you-can-see) 这组数据来自其构造的负载，缓存需求约为集群容量的 73%，反映的是趋势而不是一般情况。

路由也有代价：请求集中到有缓存的副本，负载就可能不均。SGLang 论文把局部性与并行效率的权衡列为未来工作。

## 七、边界

**计时侧信道。** 命中会让 TTFT 明显变短，而 TTFT 是请求方能观测到的。攻击者可以构造候选前缀、测量响应时间，逐个 token 猜出别人 prompt 里的内容。[The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) 在 SGLang 等开源系统上验证了这类泄露，并提出逐 token 搜索共享前缀的算法；文中还指出，同期工作 PromptLeak 依赖 LPM 调度（命中长的请求先运行）。vLLM 的对策是 `cache_salt`：只有 salt 相同的请求才共享缓存，由服务方决定信任边界。Early Bird 的作者则建议至少以 $k$ 个 token（$k \ge 2$）为单位共享，增加逐 token 猜测的难度。

**模型结构。** 前缀复用依赖“每个位置的 KV 只取决于前缀”。Mamba 等状态空间层和线性 attention 只保存一个不断原地更新的状态，无法回退到任意前缀，只能在事先保存过状态的位置精确命中；[Marconi](https://arxiv.org/abs/2411.19379) 针对混合模型，按复用概率决定保存哪些状态。vLLM 对混合模型提供 `--mamba-cache-mode align`，只在 block 边界保存状态。Sliding window attention 则只需要最后一个窗口内的 block。

**权重更新。** Hash 只覆盖 token 与 extra keys，不包含模型权重。RLHF 训练中推理引擎的权重每轮都会更新，旧的 KV 已经失效，必须调用 `reset_prefix_cache` 清空缓存。

## 八、小结

Prefix caching 的机制本身很轻：vLLM 用链式 hash 给写满的 block 编号，把空闲 block 直接当作缓存，按请求 LRU、从尾部开始淘汰；SGLang 用 radix tree 按 token 匹配，并能按命中长度调度。真正决定效果的，是负载里有多少共享前缀、chat template 是否保持前缀稳定、显存和分层存储能留住多少，以及路由能否把请求送到缓存所在的副本。

## 参考

- vLLM Team. [Automatic Prefix Caching（设计文档）](https://docs.vllm.ai/en/latest/design/prefix_caching/) · [Automatic Prefix Caching（使用文档）](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [KV Offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) · [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/v1/core/kv_cache_manager.py`、`kv_cache_utils.py`、`block_pool.py`、`single_type_kv_cache_manager.py`。
- Lianmin Zheng, et al. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). NeurIPS 2024.
- [SGLang 源码](https://github.com/sgl-project/sglang/tree/8854857)：`python/sglang/srt/mem_cache/radix_cache.py`、`evict_policy.py`、`managers/schedule_policy.py`。
- llm-d Team. [KV-Cache Wins You Can See: From Prefix Caching in vLLM to Distributed Scheduling with llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see). 2025.
- Linke Song, et al. [The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems](https://arxiv.org/abs/2409.20002).
- Rui Pan, et al. [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379). MLSys 2025.
