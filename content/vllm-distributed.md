DeepSeek-V4-Pro 的权重文件有 865 GB，一张 B200 的显存是 180 GB，一张卡装不下。SGLang 的 cookbook 用一台 8 卡的机器部署它，并给了两种把权重分到 8 张卡上的分法。

TP 把每个矩阵切成 8 条，单个请求最快。DP attention + EP 让 expert 整个放、attention 每张卡放一份，能装下的 KV cache 是 TP 的约 5 倍。实例之间还可以再加一层 PD 分离，把 prefill 与 decode 分开。下表列出各种并行方式切的是什么、通信发生在哪、适用于什么模型。源码基于 vLLM v0.30.0，部署配置取自 SGLang 的 DeepSeek-V4 cookbook。

| 并行方式 | 切的是什么 | 通信发生在哪 | 适用 |
| --- | --- | --- | --- |
| TP | 每层的权重矩阵 | 每层两次 all-reduce | 稠密模型；MoE 模型要单个请求快 |
| PP | 层，前后分段 | 段与段之间传一次 hidden state | 卡间带宽低，或跨节点 |
| DP | 请求，每张卡一份完整模型 | 无，靠路由分发 | 模型一张卡装得下 |
| DP attention + EP | attention 按请求分，expert 按个数分 | 每个 MoE 层一次 dispatch、一次 combine | MoE 模型要同时服务很多请求 |
| PD 分离 | 一个请求的 prefill 与 decode | prefill 结束后传一次 KV cache | 长 prompt，对 ITL 要求严 |

## 一、V4-Pro 里面有什么

一个 token 依次穿过 embedding、61 个 Transformer 层和输出层。61 层结构相同，每层两个子层：attention 让当前 token 读前面的 token 留下的 KV，MoE 加工这个 token。MoE 相当于普通 Transformer 里的 MLP，只是换成了 384 个 expert 加 1 个所有 token 都用的 shared expert，router 给每个 token 选 6 个，结果加权求和。总参数 1.6T，每个 token 只用到 49B。[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

::dist-v4-anatomy::

显存里放的是权重和 KV cache。**权重**装进去就不变。一个 expert 是 3 个 7168 × 3072 的矩阵，FP4 存下来约 35 MB；每层 384 个共 13.5 GB，61 层加 1 个 MTP 层合计 836 GB，占全部 865 GB 的 97%。attention、embedding 等其余部分只有 29 GB。

**KV cache** 跟着请求走：请求越多、上下文越长，占得越多。V4 每个 token 在每层只留一份 KV，旧的 KV 还会压缩，30 层每 4 个 token 并成一份，31 层每 128 个并成一份。每个 token 约 5.5 KB，1M token 的上下文约 5.8 GB；官方给的对比是 1M 上下文下 KV cache 只有 V3.2 的 10%。

8 张 B200 共 1440 GB，放下权重还剩 575 GB 给 KV cache。[SGLang DeepSeek-V4 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)

## 二、TP：每个矩阵切成 8 条

TP 来自 Megatron-LM，原本用于训练，推理框架沿用了它的 forward 部分。它利用的是矩阵乘法可以按行或按列拆开，一层的两个子层各拆一次：

- **MoE**：一个 expert 就是一个小 MLP。up 与 gate 两个矩阵按输出维度切，每张卡算出 3072 维中间结果里自己的 384 维；激活函数逐元素计算，各卡独立做；down 矩阵按输入维度切，接住本卡的那一段，输出一个 7168 维的部分和。
- **Attention**：Q 的投影按 head 切，每张卡只算自己那几个 head；输出投影按输入维度切，正好接住本卡的 head。KV 只有一份，它的投影不切，每张卡都算出完整的 KV。

两个子层的末尾，每张卡手里都只有部分和，要做一次 all-reduce 把 8 张卡的结果逐元素相加，才得到完整的输出。所以每层两次 all-reduce，61 层共 122 次，中间不需要通信。[Megatron-LM, 2019](https://arxiv.org/abs/1909.08053) vLLM 中对应 `QKVParallelLinear`、`ColumnParallelLinear` 与 `RowParallelLinear` 三个类，all-reduce 在 `RowParallelLinear.forward` 的末尾。

::dist-tp-layer::

all-reduce 传的是 hidden state，大小是 token 数乘以 hidden size，与权重多大无关。V4-Pro 的 hidden size 是 7168，按 BF16 计算：

| 场景 | 每次 all-reduce | 一次 forward 合计 |
| --- | --- | --- |
| decode，64 个请求 | 0.9 MB | 0.11 GB |
| prefill，8192 个 token | 117 MB | 14 GB |

decode 的数据量很小，耗时取决于每次 all-reduce 的固定延迟，122 次累加起来才明显。prefill 的数据量大两个数量级，受卡间带宽限制。

TP 的好处是 8 张卡合力算同一个 token。decode 的时间主要花在读权重上，每张卡只读 ⅛ 的权重，单步就更快。DistServe 把这个加速比记为 $K$：两张卡时 $1 < K < 2$，差的部分就是通信。[DistServe, 2024](https://arxiv.org/abs/2401.09670) SGLang 给 V4-Pro 的低延迟配置就是 `--tp 8`，vLLM 里对应 `--tensor-parallel-size 8`。

**PP** 按层分段，前一半层在一张卡上，后一半在另一张卡上，段与段之间只传一次 hidden state。但一个 token 仍要依次走完所有层，单个请求的延迟不变，它只解决装不下的问题。卡间没有 NVLink 时（例如 L40S）官方建议用 PP 代替 TP，常见的组合是节点内用 TP、跨节点用 PP。

## 三、TP 的问题：KV cache 切不开

TP 之后每张卡是整个模型的 ⅛：108 GB 权重，剩 72 GB 放 KV cache。TP 分 KV 的办法是跟着 head 一起切。Llama-3.3-70B 有 8 个 KV head，TP = 8 时每张卡正好存一个 head 的 K/V，KV cache 也跟着分成了 8 份。V4 每个 token 只有一份 KV（`num_key_value_heads` 为 1），1 个 head 没法分给 8 张卡，结果是每张卡都存一份完整的 KV cache：8 张卡空着 575 GB，实际只存了 72 GB 的内容，能同时服务的请求数和只有一张卡时一样。[vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)

::dist-kv-tp-dp::

## 四、DP attention + EP：expert 整个放，attention 每张卡一份

### Attention：每张卡一份

TP 下 8 张卡合起来算一个请求，所以每张卡都要有这个请求的 KV。想让 KV 只存一份，一个请求就只能待在一张卡上；这张卡要独自算完 attention，⅛ 的权重不够，得有完整的一份。DP attention 的做法是 attention 等 29 GB 每张卡放一份完整的，请求分给各卡，每张卡只存自己那批请求的 KV。每张卡因此多占 25 GB。

### Expert：整个放

836 GB 的 expert 没法每张卡放一份，只能分开放。TP 会把每个 expert 都切成 8 条，每张卡持有全部 384 个 expert 的一条窄片，并且要处理所有 token。EP 则以 expert 为单位：每张卡每层放 48 个完整的 expert，只处理被分到这些 expert 的 token。Expert 本来就又多又小，不用再切。

::dist-ledger::

每张卡的权重从 108 GB 涨到 133 GB，剩下的显存从 72 GB 减到约 47 GB，但 8 张卡的 KV 不再重复：8 × 46.5 = 372 GB，是 TP 的约 5 倍。按 1M token 的请求算，TP 下能放约 12 个，DP attention + EP 下约 64 个。

SGLang 给 V4-Pro 的高吞吐配置是 `--tp 8 --dp 8 --enable-dp-attention`；vLLM 里用 `--data-parallel-size 8 --enable-expert-parallel`，EP 的规模自动取 TP × DP。SGLang 在 v0.4 实现 DP attention 时，在 8 张 H100 上 decode 吞吐提高 1.9 倍。[SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) vLLM 还提供另一条路线 decode context parallel（`-dcp`）：保留 TP，把 KV cache 按 token 位置交错分给各卡，同样消除重复，代价是 attention 多一次通信。

### 跨卡找 expert

通信从固定的 all-reduce 变成了随路由结果变化的 all-to-all。TP 下每张卡都有每个 expert 的 ⅛，不用找；EP 下 token 选中的 6 个 expert 多半在别的卡上，每个 MoE 层有两步：

- **dispatch**：router 给每个 token 选出 expert 后，把它的 hidden state 发到这些 expert 所在的卡。
- **combine**：各卡算完，把结果送回 token 原来所在的卡，按 router 给的权重求和。

::dist-ep-path::

vLLM 默认的做法是把所有 token 广播给每张卡，各卡只算落在本地 expert 上的部分；DeepSeek 开源的 DeepEP 则只把 token 发往它需要的卡。[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

### 同步与负载不均

各卡不再独立，expert 层每一步都要所有卡一起参与。只要有一张卡在处理请求，其余的卡即使没有请求也要跑一次空的 forward，为别人的 token 算 expert；一步的耗时由最慢的那张卡决定。

Router 选 expert 并不均匀，热门 expert 所在的卡要处理更多 token，其余的卡算完了也得等它。对策是 EPLB（`--enable-eplb`），算法来自 DeepSeek：统计每个 expert 收到的 token 数，给最热的 expert 多放几份副本，再重新分配到各卡，使每张卡的负载接近。通信的等待时间则用 DBO 藏起来：把一个 batch 拆成两半，一半在等 dispatch 或 combine 时，另一半正好做计算。

SGLang 用这一整套在 12 台 8 × H100 的节点上部署上一代的 DeepSeek-V3，输出吞吐是 TP 方案的 5 倍。[SGLang, 2025](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

逐层同步也有反对意见。TensorRT-LLM 团队指出，各卡处理的序列长度波动 20% 时，等待最慢的卡带来的同步开销约 10%。他们的 DWDP 让 token 留在原地，每张卡按需把别的卡上的 expert 权重拉过来算，卡与卡之间不再需要同步。[TensorRT-LLM DWDP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.md)

## 五、PD 分离：两个阶段放到不同实例

前面两种切法都发生在一个实例内部，PD 分离切的是一个请求的生命周期：prefill 在一组实例上做，decode 在另一组实例上做。V4 的上下文上限是 1M token，长 prompt 的 prefill 会拖慢同一实例里正在进行的 decode。

### 为什么要分

Prefill 一次处理整个 prompt，一步要几百毫秒，是算力密集的；decode 每步只生成一个 token，一步几毫秒到几十毫秒，时间花在读权重和 KV cache 上。两者放在同一张卡上，一个长 prompt 进来，正在生成的请求就要等它，ITL 出现尖峰；反过来 decode 也拖慢 TTFT。

DistServe 量化了这种干扰。13B 模型在一张 A100 上，同时满足 TTFT 与 ITL 要求时只能承受 1.6 请求/秒；只做 prefill 能到 5.6，只做 decode 能到 10。用两张卡做 prefill、一张卡做 decode，平均每张卡 3.3 请求/秒，是合在一起的 2.1 倍。[DistServe, 2024](https://arxiv.org/abs/2401.09670)

分开之后，两边还可以各用各的配置：prefill 实例追求单请求快，decode 实例追求 batch 大。Splitwise 让 decode 跑在较旧、较便宜的 GPU 上，因为它用不满新卡的算力。[Splitwise, 2024](https://arxiv.org/abs/2311.18677) MoE 模型还有一个理由：DeepEP 的两种模式分别适合 prefill 与 decode，同一个实例里两者难以兼顾。

### vLLM 的实现

vLLM 的 PD 分离是两个独立的 vLLM 实例加一个 connector。一个请求被 router 拆成两段：

::dist-pd-flow::

Prefill 请求把 `max_tokens` 设为 1，P 实例算完 prompt 后把 KV block 留在显存里，并在响应中带回这些 block 的位置（`kv_transfer_params`）。Router 再把 decode 请求连同这份位置信息发给 D 实例，D 把 KV 读进自己的 KV cache，重算最后一个 prompt token，从第一个输出 token 开始生成；P 生成的那个 token 不被使用。

传输由 connector 负责。主力是 `NixlConnector`：NIXL 是 NVIDIA 的传输库，底层通常走 RDMA，数据从一张卡的显存直接到另一张卡的显存。[vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

### 要传多少

PD 分离的成本就是这一次 KV 传输，大小与 prompt 长度成正比。以 4096 个 token 的 prompt 为例：

| 模型 | 每个 token 的 KV | 4096 个 token |
| --- | --- | --- |
| Llama-3.3-70B（8 个 KV head） | 0.31 MB | 1.3 GB |
| DeepSeek-V4-Pro（压缩后的 KV） | 5.5 KB，另有每个请求固定的约 23 MB | 0.05 GB |

1.3 GB 在 100 Gbps 的网络上约 0.1 秒，相对几百毫秒的 prefill 不算多。V4-Pro 的 KV 经过压缩，1M token 的 prompt 也只要传约 5.8 GB。DistServe 把配对的 P 与 D 放在同一台机器上走 NVLink，传输只占总耗时的不到 0.1%；Splitwise 让每一层算完就开始传这一层的 KV，传输与后面各层的计算重叠，露在外面的只剩 5 至 8 毫秒。

### 不适合的情形

PD 分离改善的是 ITL 的稳定性，vLLM 的文档明确说它不提升吞吐。下面几种情形合在一起更好：

- 只有一两张卡时，拆成 P 和 D 意味着总有一边闲着。DistServe 也把资源受限与只看吞吐的离线场景列为不适用。
- prompt 短或缓存命中高时，需要新算的 prefill 很少，干扰本来就小，传输是净成本。Dynamo 的 router 对这类请求跳过远端 prefill，在 decode 实例上本地完成。
- 负载里输入输出的比例一变，P 与 D 就有一边排队、一边空闲。NVIDIA 扫过大量配置后的结论是，PD 分离在 prefill 占比高、模型大的负载上最有效，并且要能动态调整两边的比例。[Mitra et al., 2025](https://arxiv.org/abs/2506.05508)

不拆实例也能压住干扰：chunked prefill 把长 prompt 切成小块，与 decode 混在一个 batch 里。TTFT 要求严、ITL 宽松时合在一起更好，ITL 要求严时分开更好。[TaiChi, 2025](https://arxiv.org/abs/2508.01989)

## 六、怎么选

模型一张卡装得下就不用切，每张卡放一份完整的权重，用 DP 分发请求。装不下的稠密模型在节点内用 TP，没有 NVLink 或要跨节点时加 PP。MoE 模型要单个请求快用 TP，要同时服务很多请求用 DP attention + EP，规模大了再开 EPLB 与 DBO。ITL 尖峰不可接受、prompt 长且卡足够多时，再在实例之间做 PD 分离。

这几种方式可以叠加：SGLang 部署 DeepSeek-V3 时，P 与 D 两组实例内部各自是 DP attention + EP。

## 参考

- Shoeybi et al. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). 2019.
- Zhong et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). OSDI 2024.
- Patel et al. [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677). ISCA 2024.
- vLLM Team. [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) · [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) · [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) · [Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) · [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/) · [Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/dbo/).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/distributed/parallel_state.py`、`vllm/distributed/device_communicators/`、`vllm/distributed/eplb/`、`vllm/distributed/kv_transfer/`、`vllm/model_executor/layers/linear.py`、`vllm/models/deepseek_v4/`、`vllm/v1/worker/gpu/dp_utils.py`、`pp_utils.py`、`kv_connector.py`。
- LMSYS. [SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). 2024.
- LMSYS. [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/). 2025.
- NVIDIA. TensorRT-LLM [Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md) · [Disaggregated Serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md) · [DWDP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.md).
- Mitra et al. [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/abs/2506.05508). 2025.
- Wang et al. [Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving](https://arxiv.org/abs/2508.01989). 2025.
- DeepSeek-AI. [DeepSeek-V4-Pro model card 与 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). 2026.
- SGLang. [DeepSeek-V4 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx).
- DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). 2024.
