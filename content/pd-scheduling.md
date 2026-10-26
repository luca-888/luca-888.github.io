PD 分离是把一个请求的 prefill 与 decode 放到不同的实例上执行：prefill 实例（P）算出 prompt 的 KV cache，传给 decode 实例（D）逐 token 生成。拆开之后，一个请求要经过四次合并部署里没有的调度决策：选哪个 P、要不要远端 prefill、D 何时接纳、P 与 D 各部署几个。这些决策决定分离的收益。Mooncake 用 4 个实例服务长文本负载，3P+1D 的吞吐比 4 个合并部署的 vLLM 实例高 20% 至 40%，换成 2P+2D 后 TTFT 无法满足 SLO。[Mooncake, 2025](https://arxiv.org/abs/2407.00079)

## 一、干扰与分离

Prefill 一次处理整个 prompt，耗时由算力决定；decode 每步只生成一个 token，耗时由读取权重和 KV cache 的带宽决定。两者在同一个实例上按 iteration 混合调度时，一个长 prompt 的 prefill 会让同一 batch 里所有 decode 请求多等几百毫秒，ITL 出现尖峰；反过来，decode 请求也会占用 prefill 的计算时间，推高 TTFT。

衡量这种干扰的指标是 goodput，即同时满足 TTFT 与 ITL 要求的请求率。DistServe 在一张 A100 上服务 13B 模型，合并部署的 goodput 为 1.6 请求/秒；同一张卡只做 prefill 时为 5.6，只做 decode 时为 10。用两张卡做 prefill、一张卡做 decode，平均每张卡 3.3 请求/秒，是合并部署的 2.1 倍。[DistServe, 2024](https://arxiv.org/abs/2401.09670) vLLM 文档称 PD 分离不提升吞吐，指的是不计 SLO 的总 token 速率，与 goodput 不是同一个指标。

不拆实例也能缓解干扰：chunked prefill 把长 prompt 切成小块，每个 iteration 只与 decode 混合一块，单步耗时有上界。[Sarathi-Serve, 2024](https://arxiv.org/abs/2403.02310) TTFT 要求严、ITL 宽松时合并部署更好，ITL 要求严、TTFT 宽松时分离更好。[TaiChi, 2025](https://arxiv.org/abs/2508.01989)

## 二、一个请求的四次决策

合并部署中，router 为请求选一个实例，其余由实例内部的 scheduler 决定。分离部署中，router 与两类实例共同完成四项决策：

::pd-decisions::

| 决策 | 依据 | 做错的后果 |
| --- | --- | --- |
| ① 选哪个 P | 各 P 上的 prefix cache 命中、排队长度 | 重复计算已缓存的前缀，或在热门前缀所在的 P 上排队 |
| ② 要不要远端 prefill | 所选 D 上已缓存的前缀长度 | 短 prompt 白付一次传输，或长 prefill 挤进 D 的 batch |
| ③ D 何时接纳 | D 的空闲显存与已承诺的预留 | P 算完的 KV 无处可去，或 D 的显存空等 |
| ④ P 与 D 各几个 | 负载的输入输出长度、SLO | 一边排队、另一边空闲 |

前三项按请求决定，第四项按分钟到小时的周期决定。

## 三、选 P：缓存命中与排队

P 启用 prefix caching 后，同一个 system prompt 或同一份文档的 KV 已在某个 P 上时，请求发往该 P 只需计算未命中的部分。命中最长的 P 往往也最忙，因为热门前缀会把请求集中到少数几个 P 上。

Mooncake 的 Conductor 为每个 P 估计 TTFT，选最小者：

$$
\mathrm{TTFT}_p = T_{\text{queue}}(p) + T_{\text{prefill}}(L,\ h_p)
$$

其中 $L$ 是 prompt 长度，$h_p$ 是这个 P 上命中的前缀长度。$T_{\text{prefill}}$ 由离线测得的模型给出，$T_{\text{queue}}$ 是队列中各请求 prefill 时间之和。如果别的 P 上的命中比本地长得多，Conductor 还会估计先把那段 KV 传过来再算的耗时；选中这种方案时，该段 KV 随之复制到新的 P 上，热门前缀的 KV 由此逐渐分散到多个 P，Mooncake 称为 hot-spot migration。[Mooncake, 2025](https://arxiv.org/abs/2407.00079)

::pd-pick-p::

其他系统用更简单的规则组合这两个因素：

- **SGLang gateway** 的 `cache_aware` 为每个 P 维护一棵近似 radix tree，记录发给它的请求原文。最长匹配超过 prompt 的 50% 时选匹配最长的 P，否则选树最小的 P；一旦最忙与最闲的 P 排队数相差超过 32 且比值超过 1.1，改为选队列最短的 P。
- **Dynamo** 的 KV router 把两个因素都换算成 KV block 数：排队中与本请求待算的 prompt block 数，减去命中可抵扣的 block 数，再加上该实例上正在 decode 的 block 数，选总和最小的实例。
- **vLLM** 自带的示例 proxy 只做轮询，生产部署的路由交给 llm-d、Dynamo 等外部组件。

## 四、要不要远端 prefill：先选 D

多轮对话与 agent 负载中，第 $n$ 轮的 prompt 是前 $n-1$ 轮的全部内容加一条新消息。前几轮的 KV 是在 D 上生成的，仍留在那个 D 的 KV cache 里。如果第 $n$ 轮仍按 P 上的缓存与排队选 P，P 上没有这段历史，需要从头计算，再把整段 KV 传回 D。

llm-d 与 Dynamo 因此先选 D，再根据这个 D 上已缓存的长度决定是否需要 P。D 上缓存之外还需要计算的 token 数称为有效 prompt 长度。

::pd-local-or-remote::

- **llm-d** 的 `prefix-based-pd-decider`：有效 prompt 长度达到阈值 `nonCachedTokens` 时走远端 prefill，否则在 D 上本地完成 prefill 与 decode。
- **Dynamo** 的 conditional disaggregation：有效 prompt 长度小于 2048，且占 prompt 长度的比例小于 0.7 时本地完成；另一种策略是所选 P 过忙时也本地完成。

本地完成的请求省去一次传输，代价是把一段 prefill 放进了 D 的 batch，即分离要避免的干扰。阈值限定了这段 prefill 的长度，干扰因此有上界。

vLLM 的做法是把多轮对话的下一轮仍然发给 P，P 先从 D 取回已有的 KV，只计算新消息部分，再把结果传给 D。远端可复用的 token 少于 64 个时，P 在本地重新计算。这种双向传输需要一个记录每轮 `kv_transfer_params` 的有状态 proxy。

## 五、D 的接纳：先预留还是先计算

KV 从 P 传到 D 之前，D 要为它分配好 block。P 开始计算的时机决定了 D 显存不足时由哪一方等待。vLLM 与 SGLang 的默认做法相反：

::pd-admission-order::

| | 先计算（vLLM 默认） | 先预留（SGLang 默认） |
| --- | --- | --- |
| P 何时开始 | 收到请求即开始 | D 预留好 block 之后 |
| 等待期间占用的显存 | P 上算完的 KV | D 上预留的空 block |
| D 长时间放不下 | KV 超过 lease 被释放，prefill 的计算作废 | 请求在 P 前排队，不浪费计算 |

两边的细节都围绕避免死锁与抢占。vLLM 的 D 在接纳一个远端请求时，要求它的 block 放得进“空闲 block 减去其他传输中请求已占的 block”；请求在传输期间处于 `WAITING_FOR_REMOTE_KVS` 状态，不参与计算，也不能被抢占。P 侧 KV 的保留时长称为 lease，由 `kv_lease_duration` 设定，D 排队期间会发心跳续期；过期的请求数单独计数，文档将其作为扩缩容的信号。SGLang 的 D 除了 prompt 的 KV，还为每个活跃请求预留 512 个 token 的 decode 空间，防止接纳后不久即需撤回正在运行的请求。SGLang 也提供 `prefill_complete` 选项，行为与 vLLM 的默认相同。

集群层面的对应问题是 router 是否接收一个请求。过载时，只看 P 的负载接收请求，会有一部分请求在 prefill 完成后被满载的 D 拒绝，prefill 的计算作废。Mooncake 改为请求到达时就按 P、D 两池负载中较高的一方决定是否接收，称为 early rejection。D 的负载要等 prefill 完成才上升，按当前负载做的接收决策因此滞后一个 prefill 的时间，P 与 D 的负载交替涨落。

::pd-mooncake-load::

Mooncake 进一步改为按预测接收，即假设每个请求的 decode 时长相同，估计正在 prefill 的请求转入 D 之后 D 的负载，以此判断是否接收。8P+8D 的集群以 2 倍速度回放 23,000 条真实请求，只看各自负载时拒绝 4183 条，early rejection 拒绝 3771 条，按预测接收拒绝 3589 条。[Mooncake, 2025](https://arxiv.org/abs/2407.00079)

## 六、P 与 D 的比例

DistServe 的例子中，一张卡的 prefill 能力是 5.6 请求/秒，decode 能力是 10 请求/秒，两张 P 配一张 D 才让两边的产出接近。这个 2:1 由负载决定，prompt 变长时 P 需要更多，输出变长时 D 需要更多。比例偏离负载时，多出的一侧空闲，不足的一侧排队。Mooncake 的 2P+2D 属于这种情况，ITL 比 3P+1D 更低，TTFT 超出 SLO。

::pd-ratio::

最优比例还随 SLO 变化。NVIDIA 在 DeepSeek-R1 上扫描了大量配置：

::pd-fixed-ratios::

图中的 Ctx:Gen 即 P:D 的 GPU 数之比。固定为 3.5 时，宽松延迟下接近最优，延迟要求收紧后落到合并部署之下；固定为 0.5 时，延迟要求收紧后接近最优，宽松延迟下归一化吞吐只有 0.42，最优比例为 0.97。每个延迟目标都重新选比例，曲线才能全程高于合并部署。[Mitra et al., 2025](https://arxiv.org/abs/2506.05508)

负载在一天内变化，比例因此需要在线调整。Dynamo 的 planner 用两个周期不同的循环分别为 P 和 D 定副本数：

- **吞吐循环**：周期较长。预测下一个周期的请求数、平均输入与输出长度，用性能模型换算成满足 SLO 所需的 P、D 数，作为副本数的下限。
- **负载循环**：周期较短。按实时的队列与延迟指标每次增减一个副本，纠正预测误差与突发负载，但不低于吞吐循环给出的下限。

Mooncake 论文认为真实集群中 P、D 的需求在一段时间内稳定，比例可以预设。比例调整需要启动实例并加载权重，或切换实例的角色，只有负载变化慢于这个过程时，在线调整才有意义。

## 七、实测

::pd-measure-placeholder::

## 八、各系统的取舍

| | 选 P | 要不要远端 prefill | D 的接纳 | P:D 比例 |
| --- | --- | --- | --- | --- |
| Mooncake | 估计 TTFT，复制热门前缀的 KV | 总是远端 | router 按预测的 D 负载提前拒绝 | 预设 |
| SGLang | 近似 radix tree 加负载阈值 | 总是远端 | 先预留后计算 | 手动 |
| vLLM | 交给外部 router | 外部决定；多轮可从 D 拉回 KV | 先计算后预留，KV 有 lease | 外部决定 |
| llm-d | 先选 D 后选 P | D 上未缓存部分超过阈值才远端 | 沿用 vLLM | 外部决定 |
| Dynamo | block 数代价函数 | 有效 prompt 长度与比例阈值 | 沿用后端 | planner 两层循环 |

可复用 KV 所在的位置决定路由的顺序。单轮、共享前缀多的负载，可复用的 KV 在 P 上，先选 P；多轮与 agent 负载，可复用的 KV 在 D 上，先选 D，短的增量 prefill 直接在 D 上做。

## 参考

- Ruoyu Qin, et al. [Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot](https://arxiv.org/abs/2407.00079). FAST 2025.
- Yinmin Zhong, et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). OSDI 2024.
- Amey Agrawal, et al. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). OSDI 2024.
- Tiyasa Mitra, et al. [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/abs/2506.05508). 2025.
- Chao Wang, et al. [Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving](https://arxiv.org/abs/2508.01989). 2025.
- vLLM. [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/) · [NixlConnector Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)；[源码 4611c2c3](https://github.com/vllm-project/vllm/tree/4611c2c3)：`vllm/v1/core/sched/scheduler.py`、`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`、`examples/disaggregated/disaggregated_serving/`。
- SGLang. [PD Disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)；[源码 51686cf8](https://github.com/sgl-project/sglang/tree/51686cf8)：`python/sglang/srt/disaggregation/`、`sgl-model-gateway/src/policies/cache_aware.rs`、`sgl-model-gateway/src/routers/http/pd_router.rs`。
- NVIDIA Dynamo. [源码 a83f479](https://github.com/ai-dynamo/dynamo/tree/a83f479) 中的文档：router `routing-concepts.md`、`conditional-disaggregation.md`，planner `planner-design.md`。
- llm-d. [llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler/tree/ce60ef7)：`docs/disaggregation.md`。
