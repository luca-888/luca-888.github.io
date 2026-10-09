Attention 在 decode 时要读取所有历史 token 的 key 与 value，这部分缓存称为 KV cache，它的大小决定一张卡能容纳多少上下文，以及 decode 每一步要从显存读取多少字节。MQA、GQA 与 MLA 是三种缩小 KV cache 的 attention 形式。以一个 80 层、64 个 query head、head dim 为 128 的模型为例，BF16 下一条 128K token 的请求所需的 KV cache 如下：

| attention 形式 | 每 token KV cache | 128K token 请求 |
| --- | ---: | ---: |
| MHA | 2560 KiB | 320 GiB |
| GQA，8 个 KV head | 320 KiB | 40 GiB |
| MLA，512 + 64 维 | 90 KiB | 11.3 GiB |

## 一、MHA

Multi-Head Attention（MHA）是 Transformer 原始论文中的 attention 形式，每个 head 有自己的 key 与 value。[Vaswani et al.](https://arxiv.org/abs/1706.03762)

**代表模型**：GPT-3、Llama 1。

### 信息传递

Attention 把历史 token 的信息传给当前 token。下图以 3 个 head、4 个历史 token 为例，画出 decode 一步中各 head 读取的缓存。

::mha-flow::

MHA 的每个 head $i$ 用三个投影矩阵，从当前 token $t$ 的 hidden state 得到 query，从历史 token $j$ 的 hidden state 得到 key 与 value：

$$
q_{t,i} = W^Q_i h_t,\qquad k_{j,i} = W^K_i h_j,\qquad v_{j,i} = W^V_i h_j
$$

三者的作用如下：

- **query**：当前 token 在这个 head 中要查找的内容。
- **key**：历史 token 用于被匹配的表示，与 query 点积得到分数。
- **value**：历史 token 被选中后传递给当前 token 的内容。

head $i$ 的输出是 value 按分数的加权和：

$$
o_{t,i} = \sum_j \operatorname{softmax}_j\!\big(q_{t,i}^\top k_{j,i}\big)\, v_{j,i}
$$

每个 head 的 $W^K_i$、$W^V_i$ 各不相同，head 数为 $h$ 时，同一个历史 token 有 $h$ 个 key 和 $h$ 个 value。query 每步重新计算，key 与 value 需要缓存。

### KV cache 的大小

记每个 head 的维度为 $d$。MHA 每层每 token 缓存的元素数为：

$$
\underbrace{hd}_{K} + \underbrace{hd}_{V} = 2hd
$$

$h = d = 128$ 时为 32768 个元素，BF16 下为 64 KiB。

### Decode 的瓶颈

**decode 时 attention 的耗时由读取的字节数决定。** decode 每一步只有一个新 token，它的 query 与全部历史 key 做点积，点积经 softmax 得到权重，再按权重对 value 加权求和。每从显存读入一个 K 或 V 元素，只做一次乘加。

每读一个字节所做的运算次数称为算术强度。以一个 query head 与一个历史 token 为例：打分 $q\cdot k_j$ 读入 $d$ 个 key 元素，做 $d$ 次乘加；加权求和 $p_j v_j$ 读入 $d$ 个 value 元素，再做 $d$ 次乘加。一次乘加计 2 FLOPs，一个 BF16 元素占 2 字节，因此：

$$
\text{算术强度} = \frac{2d + 2d \ \text{FLOPs}}{2d + 2d \ \text{bytes}} = 1 \ \text{FLOP/byte}
$$

MHA 中每个 K、V 元素只属于一个 query head，读入后只参与一次乘加，换成全部 $h$ 个 head 与 $L$ 个历史 token，比值仍为 1。query 每步只读一次，softmax 每个 token 只有几次运算，两者都可忽略。MHA decode 的算术强度与 H100 的算力带宽比如下：

- **MHA decode**：约 1 FLOP/byte。
- **H100**：BF16 算力与显存带宽之比约为 295 FLOPs/byte。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

前者远低于后者，attention 在 decode 时无法充分利用算力，耗时取决于显存带宽。增大 batch 能分摊权重的读取，但每个请求的 KV cache 各不相同，attention 部分的算术强度不随 batch 提高。

**缩小每层缓存直接缩短 decode 时间。** Shazeer 在提出 MQA 的论文中把翻译模型的 MHA 换成 MQA，decoder 每个 token 的平均耗时 MHA 为 46 µs，MQA 为 3.8 µs（TPUv2）。[Shazeer](https://arxiv.org/abs/1911.02150)

## 二、MQA 与 GQA

Multi-Query Attention（MQA）让全部 query head 共享一组 key 与 value，Grouped-Query Attention（GQA）把 query head 分组，每组共享一组。被共享的一组 key 与 value 称为一个 KV head。[Shazeer](https://arxiv.org/abs/1911.02150) [Ainslie et al.](https://arxiv.org/abs/2305.13245)

**代表模型**：Llama-3.1-70B（64 个 query head、8 个 KV head）、Qwen3-235B-A22B（64 个、4 个）、MiniMax-M3（64 个、4 个）。

### 共享方式

MQA、GQA 与 MLA 都保留全部 query head，改变的是历史 token 被缓存的表示。下图以 128 个 query head、$d=128$ 为例，画出四种形式每层每 token 的缓存。

::kv-cache-per-layer::

- **MHA**：每个 query head 有自己的 key 与 value，缓存 $2hd$ 个元素。
- **GQA**：KV head 数为 $h_{kv}$，每 $h/h_{kv}$ 个 query head 共享一组 key 与 value，缓存 $2h_{kv}d$ 个元素，为 MHA 的 $h_{kv}/h$。
- **MQA**：全部 query head 共享一组 key 与 value，缓存 $2d$ 个元素，为 MHA 的 $1/h$。
- **MLA**：每个 head 仍有自己的 key 与 value，但它们都由同一个 latent 线性变换得到，缓存中只有 latent 和记录位置的 RoPE key。

**共享之后，各 head 的注意力权重仍不相同，取到的内容来自同一个 value。** 同组的 head 用各自的 query 与同一个 key 打分，输出只因权重不同而不同。

**共享只减少读取，不减少计算。** 每个 query head 仍要与 key 做点积，因此算术强度随共享同一组 K/V 的 query head 数增长，GQA 约为 $h/h_{kv}$，MQA 约为 $h$。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

### 质量差异

**共享 KV head 的质量损失随规模和训练条件变化。** 两组公开实验的结果如下，“—”表示该实验未包含这一形式：

| 实验 | MHA | GQA | MQA | MLA |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 7B dense（GQA-8），MMLU | 45.2 | 41.2 | 37.9 | — |
| 876M dense（GQA-4），下游平均准确率 | 56.0 | 56.9 | 56.4 | 56.7 |

DeepSeek 的三个 7B 模型除 attention 外结构相同，各用 1.33T token 训练。[DeepSeek-V2 附录](https://arxiv.org/abs/2405.04434) 876M 的实验把各形式的 FFN 加宽到与 MHA 相同的参数量，四者的差距不到 1 个点，MHA 并非最好。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

T5-XXL 上，由 MHA checkpoint 转换并继续训练得到的 GQA-8 平均分为 47.1，MHA 为 47.2；推理时间 GQA-8 为 0.28 s，MHA 为 1.51 s。[Ainslie et al.](https://arxiv.org/abs/2305.13245)

### GQA 成为默认配置

GQA 在质量与缓存之间取折中，同时具备两个工程上的便利：

- **迁移成本低**：已有的 MHA checkpoint 可以把同组的 K、V 投影矩阵取平均，再用原预训练算力的 5% 继续训练，得到 GQA 模型。
- **适合 TP**：TP 按 head 把 attention 分到多张卡，KV head 数不少于卡数时，每张卡只存属于自己的 KV head；MQA 只有一个 KV head，每张卡都要保存一份完整副本。[Ainslie et al.](https://arxiv.org/abs/2305.13245)

## 三、MLA

Multi-head Latent Attention（MLA）由 DeepSeek-V2 提出，每个 token 每层只缓存一个低维向量，key 与 value 在计算时由它得到。[DeepSeek-V2](https://arxiv.org/abs/2405.04434)

**代表模型**：DeepSeek-V3、Kimi-K2、GLM-5，以及 Kimi-K3 的 24 层。

### 低秩联合压缩

MHA 的 key 与 value 是 hidden state 的线性投影，KV cache 保存投影的结果。MLA 把这次投影拆成两步，只缓存中间结果，称为 latent。

::mla-latent::

- **下投影**：所有 head 共用的 $W^{DKV}$ 把 hidden state $h_j$ 压缩到 $d_c$ 维，得到 latent $c_j$，写入 KV cache。
- **上投影**：计算 attention 时，每个 head 用自己的 $W^{UK}_i$、$W^{UV}_i$ 从 latent 还原出 key 与 value。

$$
c_j = W^{DKV} h_j,\qquad k_{j,i} = W^{UK}_i\, c_j,\qquad v_{j,i} = W^{UV}_i\, c_j
$$

DeepSeek-V2 与 V3 取 $d_c = 512$、$h = 128$，每个 head 的 key 与 value 为 128 维。128 个 head 的 key 与 value 合计 32768 维，全部由 512 维的 latent 决定，即“低秩”；key 与 value 由同一个 latent 还原，即“联合”。

head 数不变，head 之间的差异保存在上投影矩阵中；上投影矩阵是模型权重，对所有 token 相同，不占 KV cache。

### 矩阵吸收

**按定义计算，decode 每一步都要对全部历史 token 做上投影。** latent 减少了读取，但每个历史 token 要先还原成 128 个 head 的 key 才能与 query 点积，计算量随历史长度增长。矩阵吸收把上投影从历史 token 一侧移到 query 一侧，下图以一个 query head 与一个历史 token 为例。

::mla-absorb::

attention 用到的是 query 与 key 的点积，不需要 key 本身。key 是上投影矩阵与 latent 的乘积，点积中相乘的顺序可以调换：

$$
q_i^\top k_{j,i} = q_i^\top \big(W^{UK}_i\, c_j\big) = \big(W^{UK\top}_i q_i\big)^{\!\top} c_j
$$

- **展开**：先算 $W^{UK}_i c_j$。历史有 $L$ 个 token 时，每个 head 每步要做 $L$ 次上投影。
- **吸收**：先算 $\tilde q_i = W^{UK\top}_i q_i$。每个 head 每步只做一次，得到的 512 维向量直接与缓存的 $c_j$ 做点积。

value 一侧同样处理：先用注意力权重对 $c_j$ 加权求和，得到 512 维的结果，再乘一次 $W^{UV}_i$。整个过程不生成任何 key 或 value。

### Decoupled RoPE

RoPE 把位置编码为旋转：位置 $t$ 的 query 乘旋转矩阵 $R_t$，位置 $j$ 的 key 乘 $R_j$，两者点积后只留下相对位置的旋转 $R_{j-t}$。下图上半部分是对还原出的 key 施加 RoPE 时的分数，下半部分是 MLA 采用的 decoupled RoPE。

::decoupled-rope::

对还原出的 key 施加 RoPE，分数为：

$$
(R_t\, q_i)^\top R_j W^{UK}_i\, c_j = q_i^\top R_{j-t}\, W^{UK}_i\, c_j
$$

**$R_{j-t}$ 位于 query 与上投影矩阵之间，并随 token 对变化，$W^{UK}_i$ 无法再预先乘到 query 上。**[DeepSeek-V2](https://arxiv.org/abs/2405.04434)

**Decoupled RoPE 把位置信息放到单独的 64 维上，latent 不带位置。** query 与缓存各分为两段：

- **query 一侧**：512 维的 $\tilde q_i$ 已乘上投影矩阵，不带位置；另有 64 维带 RoPE 的 $q_i^R$。每个 query head 各有一份。
- **缓存一侧**：512 维的 latent $c_j$ 不带位置；另有 64 维带 RoPE 的 key $k_j^R$，由 $h_j$ 直接投影得到，所有 head 共享。

分数是两段点积之和。前一项不含位置，可以吸收；后一项的 key 由所有 head 共享，是 64 维的 MQA：

$$
s_{i,j} = \underbrace{\tilde q_i^\top c_j}_{\text{吸收}} + \underbrace{q_i^{R\top} k_j^R}_{\text{RoPE}}
$$

### 吸收后的形态

**吸收之后，MLA 在 decode 时是单个 KV head 的 MQA。** 128 个 query head 读取同一份缓存：key 为 576 维，由 latent 与 RoPE key 拼接而成；value 为其中 512 维的 latent。vLLM 对 MLA 模型直接把 KV head 数记为 1，源码注释为 “When using MLA during decode it becomes MQA”。[vLLM](https://github.com/vllm-project/vllm/blob/e006d76/vllm/config/model.py)

每层每 token 的缓存为 $512 + 64 = 576$ 个元素，与 2.25 个 KV head 的 GQA 相同（$2 \times 2.25 \times 128 = 576$）。

同一份数据从显存读入后既参与点积又参与加权求和，算术强度约为 $2h$；$h=128$ 时约为 256 FLOPs/byte，接近 H100 的 295。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

### 质量

DeepSeek-V2 在约 250B 参数的 MoE 模型上对比了 MLA 与 MHA：[DeepSeek-V2 附录](https://arxiv.org/abs/2405.04434)

| | MHA | MLA |
| --- | ---: | ---: |
| 每 token KV cache（元素） | 860.2K | 34.6K |
| MMLU | 57.5 | 59.0 |

MLA 的 KV cache 为 MHA 的 4%，MMLU 高 1.5。

## 四、MLA 的代价

### 用计算换读取

**吸收减少读取，增加计算。** 每个 query head 与历史 token 的点积，展开时为 $128+64$ 维，吸收后为 $512+64$ 维。decode 受读取限制，prefill 受算力限制，vLLM 对两个阶段使用不同的算法：[vLLM `mla_attention.py`](https://github.com/vllm-project/vllm/blob/e006d76/vllm/model_executor/layers/attention/mla_attention.py)

| 阶段 | 瓶颈 | 算法 | attention 形状 |
| --- | --- | --- | --- |
| decode | 读取 | 吸收 | 单个 KV head 的 MQA，QK 576 维，V 512 维 |
| prefill | 算力 | 展开 | 128 个 head 的 MHA，QK 192 维，V 128 维 |

### Head 数与计算量

**Head 数决定 decode 的计算量。** 吸收之后，每个 query head 都要与 576 维的 latent 与 RoPE key 做点积，head 越多，decode 的 FLOPs 越大。

Kimi-K2 沿用 MLA，query head 数 DeepSeek-V3 为 128 个，Kimi-K2 为 64 个。按 Moonshot 的测算，128K 序列下 head 数从 64 增加到 128 会使推理 FLOPs 增加 83%，validation loss 只降低 0.5% 到 1.2%。[Kimi-K2](https://arxiv.org/abs/2507.20534)

### TP 下的 latent 复制

**Latent 不能按 head 切分。** TP 按 head 把 attention 分到多张卡，而每个 head 都要读完整的 latent，因此每张卡都保存全部请求的 latent。TP=8 时，KV cache 的总占用为单卡的 8 倍；同样是 TP=8，GQA-8 每张卡恰好一个 KV head，没有重复。

SGLang 的 DP attention 把 attention 部分按请求而不是按 head 分给各卡，每张卡只存自己请求的 latent。在 8 张 H100 上运行 DeepSeek-Coder-V2，加入 DP attention 的 SGLang v0.4 的 decode 吞吐为 v0.3 的 1.9 倍。[SGLang](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) Zadouri 等人的 Grouped Latent Attention（GLA）改动模型结构，把 latent 拆成两个 256 维的 head，两张卡各存一半。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

## 五、Shared-KV MQA

Shared Key-Value MQA（下称 Shared-KV MQA）是 DeepSeek-V4 的 attention 形式，每个 KV entry 是一个同时充当 key 与 value 的向量。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)

**代表模型**：DeepSeek-V4-Pro。

**DeepSeek 在 V4 中去掉了 MLA 的上投影。** V4 的每个 KV entry 为 512 维，由所有 query head 共用；config 中 `num_key_value_heads` 为 1，`head_dim` 为 512，不再有 `kv_lora_rank`。[DeepSeek-V4](https://arxiv.org/abs/2606.19348) V4 与吸收后的 MLA 同为单个 KV head 的 MQA，区别在于 V4 在训练时即使用这一形态，不经过 $W^{UK}$、$W^{UV}$ 的还原。

RoPE 只作用于每个向量的最后 64 维，不再单独缓存 RoPE key。由于 value 与 key 是同一个向量，加权求和的结果带有 key 的绝对位置，V4 对输出的这 64 维再施加位置 $-t$ 的旋转（$t$ 为当前 token 的位置），使结果只依赖相对位置。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)

V4 还沿序列方向压缩：每 4 个或 128 个 token 合成一个 KV entry。两者叠加，1M token 上下文下 V4 的 KV cache 约为 BF16 GQA-8（head dim 128）的 2%。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)

## 六、各模型的 KV cache

**GQA 仍是多数模型的选择；MLA 被 Moonshot 与智谱沿用**，并与稀疏选择（GLM-5 的 DSA）或线性 attention（Kimi-K3 的 KDA）组合。

::kv-cache-models::

## 七、小结

- **MQA 与 GQA**：多个 query head 共享 KV head，KV cache 缩小为 MHA 的 $h_{kv}/h$。
- **MLA**：只缓存一个 512 维的 latent 和一个 64 维的 RoPE key，借矩阵吸收在 decode 时直接对 latent 做 MQA。它用计算换读取，head 数决定 decode 的 FLOPs，TP 下 latent 在每张卡上重复。
- **Shared-KV MQA**：DeepSeek-V4 去掉上投影，直接训练 512 维、key 与 value 共用的单个 KV head。

## 参考

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Shazeer. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). 2019.
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
- DeepSeek-AI. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). 2024.
- Zadouri, Strauss, Dao. [Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487). 2025.
- Kimi Team. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534). 2025.
- DeepSeek-AI. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348). 2026.
- LMSYS. [SGLang v0.4: Data Parallelism Attention for DeepSeek Models](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). 2024.
- vLLM [`mla_attention.py`](https://github.com/vllm-project/vllm/blob/e006d76/vllm/model_executor/layers/attention/mla_attention.py)，v0.30.0（e006d76）。
- 各模型 Hugging Face 上的 `config.json`（2026-10-08 读取）：Llama-3.1-70B、Qwen3-235B-A22B、MiniMax-M3、DeepSeek-V3、Kimi-K2、GLM-5、Kimi-K3、DeepSeek-V4-Pro。
