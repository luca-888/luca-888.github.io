Attention 在 decode 时要读取所有历史 token 的 key 与 value，这部分缓存即 KV cache。Multi-Query Attention（MQA）、Grouped-Query Attention（GQA）与 Multi-head Latent Attention（MLA）是三种缩小 KV cache 的 attention 形式。

每个 token 的 KV cache 等于层数、每层缓存的元素数与每个元素的字节数三者之积。它决定了一张卡能容纳多少上下文，也决定了 decode 每一步要从显存读取多少字节。以 BF16 下一条 128K token 的请求为例：

| 模型 | attention 形式 | 每 token KV cache | 128K token 请求 |
| --- | --- | ---: | ---: |
| Llama-3.1-70B | GQA | 320 KiB | 40 GiB |
| DeepSeek-V3 | MLA | 68.6 KiB | 8.6 GiB |

## 一、Decode 的瓶颈

### MHA 的 KV cache

Multi-Head Attention（MHA）是 Transformer 原始论文中的 attention 形式，每个 query head 有自己的一组 key 与 value；GPT-3、Llama 1 等早期 LLM 都使用 MHA。[Vaswani et al.](https://arxiv.org/abs/1706.03762)

记 query head 数为 $h$，每个 head 的维度为 $d$。MHA 每层每 token 缓存的元素数为：

$$
\underbrace{hd}_{K} + \underbrace{hd}_{V} = 2hd
$$

$h = d = 128$ 时为 32768 个元素，BF16 下为 64 KiB。

### 算术强度

**decode 时 attention 的耗时由读取的字节数决定。** decode 每一步只有一个新 token，它的 query 与全部历史 key 做点积，再用得到的权重对 value 加权求和。每从显存读入一个 K 或 V 元素，只做一次乘加。

每读一个字节所做的运算次数称为算术强度。以一个 query head 与一个历史 token 为例：打分 $q\cdot k_j$ 读入 $d$ 个 key 元素，做 $d$ 次乘加；加权求和 $p_j v_j$ 读入 $d$ 个 value 元素，再做 $d$ 次乘加。一次乘加计 2 FLOPs，一个 BF16 元素占 2 字节，因此：

$$
\text{算术强度} = \frac{2d + 2d \ \text{FLOPs}}{2d + 2d \ \text{bytes}} = 1 \ \text{FLOP/byte}
$$

MHA 中每个 K、V 元素只属于一个 query head，读入后只参与一次乘加，换成全部 $h$ 个 head 与 $L$ 个历史 token，比值仍为 1。query 每步只读一次，softmax 每个 token 只有几次运算，两者都可忽略。MHA decode 的算术强度与 H100 的算力带宽比如下：

- **MHA decode**：约 1 FLOP/byte。
- **H100**：BF16 算力与显存带宽之比约为 295 FLOPs/byte。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

前者远低于后者，attention 在 decode 时用不满算力，耗时取决于显存带宽。增大 batch 能分摊权重的读取，但每个请求的 KV cache 各不相同，attention 部分的算术强度不随 batch 提高。

**缩小每层缓存直接缩短 decode 时间。** Shazeer 在提出 MQA 的论文中把翻译模型的 MHA 换成 MQA，decoder 每个 token 的平均耗时 MHA 为 46 µs，MQA 为 3.8 µs（TPUv2）。[Shazeer](https://arxiv.org/abs/1911.02150)

## 二、共享 KV Head：MQA 与 GQA

### 共享方式

MQA 与 GQA 让多个 query head 共享同一组 K 与 V：

- **MQA**：所有 query head 共享一组，每层每 token 缓存 $2d$ 个元素，为 MHA 的 $1/h$。
- **GQA**：KV head 数为 $h_{kv}$，每 $h/h_{kv}$ 个 query head 共享一组，缓存 $2h_{kv}d$ 个元素，为 MHA 的 $h_{kv}/h$。

下图以 128 个 query head、$d=128$ 为例，画出四种形式每层每 token 的缓存。

::kv-cache-per-layer::

**共享只减少读取，不减少计算。** 每个 query head 仍要与 key 做点积，因此算术强度随共享同一组 K/V 的 query head 数增长，GQA 约为 $h/h_{kv}$，MQA 约为 $h$。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

### 质量差异

**共享 KV head 的质量损失随规模和训练条件变化。** 三组公开实验的结果如下，“—”表示该实验未包含这一形式：

| 实验 | MHA | GQA | MQA | MLA |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 7B dense（GQA-8），MMLU | 45.2 | 41.2 | 37.9 | — |
| T5-XXL（GQA-8），平均分 | 47.2 | 47.1 | 46.6 | — |
| T5-XXL（GQA-8），推理时间 | 1.51 s | 0.28 s | 0.24 s | — |
| 876M dense（GQA-4），perplexity ↓ | 11.50 | 11.34 | 11.41 | 11.36 |
| 876M dense（GQA-4），下游平均准确率 | 56.0 | 56.9 | 56.4 | 56.7 |

各组实验的条件：

- **DeepSeek 7B**：三个模型除 attention 外结构相同，各用 1.33T token 训练。[DeepSeek-V2 附录](https://arxiv.org/abs/2405.04434)
- **T5-XXL**：GQA 与 MQA 由 MHA checkpoint 转换后继续训练得到；MQA 从头训练时还出现频繁的 loss spike。[Ainslie et al.](https://arxiv.org/abs/2305.13245)
- **876M**：在 FineWeb-Edu 上训练 50B token，各形式加宽 FFN，使参数量与 MHA 相同；perplexity 在 FineWeb-Edu 验证集上测得，下游为 7 项任务的平均。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

876M 的实验是唯一一组在相同条件下对比四种形式的结果，四者的差距不到 1 个点，MHA 并非最好；DeepSeek 在 7B 上测得的差距明显更大。MLA 与 MHA 在大规模 MoE 模型上的对比见第三章。

### GQA 成为默认配置

GQA 在质量与缓存之间取折中，同时具备两个工程上的便利：

- **迁移成本低**：已有的 MHA checkpoint 可以把同组的 K、V 投影矩阵取平均，再用原预训练算力的 5% 继续训练，得到 GQA 模型。
- **适合 TP**：TP 按 head 把 attention 分到多张卡，KV head 数不少于卡数时，每张卡只存属于自己的 KV head；MQA 只有一个 KV head，每张卡都要保存一份完整副本。[Ainslie et al.](https://arxiv.org/abs/2305.13245)

Llama-3.1-70B 有 64 个 query head、8 个 KV head，Qwen3-235B-A22B 有 64 个 query head、4 个 KV head。

## 三、MLA：低秩压缩与矩阵吸收

### 低秩联合压缩

MLA 由 DeepSeek-V2 提出。每个 token 的 hidden state $h_t$ 先投影为一个 $d_c$ 维的 latent $c_t$，KV cache 只存 $c_t$；计算 attention 时再用每个 head 的上投影矩阵还原出 key 与 value：

$$
c_t = W^{DKV} h_t,\qquad k_{t,i} = W^{UK}_i\, c_t,\qquad v_{t,i} = W^{UV}_i\, c_t
$$

DeepSeek-V2 与 V3 取 $d_c = 512$、$h = 128$，每个 head 的 key 与 value 为 128 维。[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 如果 decode 时先把每个历史 token 的 latent 还原成 128 个 head 的 K 与 V，读取虽然减少了，每一步却要对全部历史重做一次还原。

### 矩阵吸收

**MLA 用矩阵吸收避免还原。** query head $i$ 与历史 token $j$ 的分数可以改写为：

$$
q_i^\top k_{j,i} = q_i^\top W^{UK}_i\, c_j = \big(W^{UK\top}_i q_i\big)^{\!\top} c_j
$$

$W^{UK}_i$ 与 token 无关，可以先乘到 query 上，得到 512 维的 $\tilde q_i$，再直接与缓存的 $c_j$ 做点积。value 一侧同样处理：先用注意力权重对 $c_j$ 加权求和，得到 512 维的结果，再乘 $W^{UV}_i$ 与输出投影。整个过程不生成任何 key 或 value。

::mla-absorb::

### 吸收后的形态

**吸收之后，MLA 在 decode 时是 head dim 为 512 的 MQA。** 128 个 query head 读取同一个 latent，latent 同时充当 key 与 value。vLLM 对 MLA 模型直接把 KV head 数记为 1，源码注释为 “When using MLA during decode it becomes MQA”。[vLLM](https://github.com/vllm-project/vllm/blob/e006d76/vllm/config/model.py)

同一份数据读入片上后既参与点积又参与加权求和，算术强度约为 $2h$；$h=128$ 时约为 256 FLOPs/byte，接近 H100 的 295。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

DeepSeek-V2 在约 250B 参数的 MoE 模型上对比了 MLA 与 MHA：[DeepSeek-V2 附录](https://arxiv.org/abs/2405.04434)

| | MHA | MLA |
| --- | ---: | ---: |
| 每 token KV cache（元素） | 860.2K | 34.6K |
| MMLU | 57.5 | 59.0 |

MLA 的 KV cache 为 MHA 的 4%，MMLU 高 1.5。

## 四、Decoupled RoPE

### RoPE 与吸收的冲突

RoPE 用与位置有关的旋转矩阵 $R_t$ 作用于 query 和 key。若对还原出的 key 施加 RoPE，分数变为：

$$
(R_t\, q_i)^\top R_j W^{UK}_i\, c_j = q_i^\top R_{j-t}\, W^{UK}_i\, c_j
$$

**query 与 latent 之间多了随 token 对变化的 $R_{j-t}$，$W^{UK}_i$ 无法再预先乘到 query 上。**[DeepSeek-V2](https://arxiv.org/abs/2405.04434)

### 单独的位置维度

MLA 把位置信息放到单独的维度上，称为 decoupled RoPE：

- **query 一侧**：每个 query head 额外有 64 维带 RoPE 的分量 $q_i^R$。
- **key 一侧**：所有 head 共享一个 64 维带 RoPE 的 key $k_j^R$，由 $h_j$ 直接投影得到，与 latent 一起缓存。

分数是两部分之和，前一项不含位置，可以吸收；后一项是普通的 MQA：

$$
s_{i,j} = \underbrace{\tilde q_i^\top c_j}_{\text{吸收}} + \underbrace{q_i^{R\top} k_j^R}_{\text{RoPE}}
$$

每层每 token 的缓存因此为：

$$
\underbrace{512}_{\text{latent}} + \underbrace{64}_{\text{RoPE key}} = 576
$$

与 2.25 组 GQA 相同（$2 \times 2.25 \times 128 = 576$）。

## 五、Prefill 与 Decode 的两种算法

**吸收形式用计算换读取。** 每个 query head 与历史 token 的点积，展开时为 $128+64$ 维，吸收后为 $512+64$ 维。两个阶段的瓶颈不同，vLLM 分别使用两种算法：[vLLM `mla_attention.py`](https://github.com/vllm-project/vllm/blob/e006d76/vllm/model_executor/layers/attention/mla_attention.py)

| 阶段 | 瓶颈 | 算法 | attention 形状 |
| --- | --- | --- | --- |
| decode | 读取 | 吸收 | 单个 KV head 的 MQA，QK 576 维，V 512 维 |
| prefill | 算力 | 展开 | 128 个 head 的 MHA，QK 192 维，V 128 维 |

decode 的吸收形式：

```python
ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)              # 每个 head 的 query 变为 512 维
o = sdpa(cat([ql_nope, q_pe], -1), cat([kv_c, k_pe], -1), kv_c)  # K 与 V 都是缓存的 latent
o = einsum("snl,lnv->snv", o, W_UV)                          # 512 维还原为每个 head 的 128 维
```

prefill 把缓存的 latent 展开成 128 个 head 的 K 与 V。上下文很长时，展开后的 K/V 超出工作区，vLLM 按固定行数分块展开历史部分，每块单独计算 attention，再用各块的 log-sum-exp 合并结果。两种算法按 scheduler 把请求标记为 prefill 还是 decode 来选择。

## 六、MLA 的代价

### Head 数与计算量

**Head 数决定 decode 的计算量。** 吸收之后，每个 query head 都要与 576 维的 latent 与 RoPE key 做点积，head 越多，decode 的 FLOPs 越大。

Kimi K2 沿用 MLA，query head 数 DeepSeek-V3 为 128 个，Kimi K2 为 64 个。按 Moonshot 的测算，128K 序列下 head 数从 64 增加到 128 会使推理 FLOPs 增加 83%，validation loss 只降低 0.5% 到 1.2%。[Kimi K2](https://arxiv.org/abs/2507.20534)

### TP 下的 latent 复制

**Latent 不能按 head 切分。** TP 按 head 把 attention 分到多张卡，而每个 head 都要读完整的 latent，因此每张卡都保存全部请求的 latent。TP=8 时，KV cache 的总占用为单卡的 8 倍；同样是 TP=8，GQA-8 每张卡恰好一个 KV head，没有重复。两种应对方式：

- **DP attention**：SGLang 把 attention 部分按请求而不是按 head 分给各卡，每张卡只存自己请求的 latent，MoE 层前后再交换 hidden state。在 8 张 H100 上运行 DeepSeek-Coder-V2，加入 DP attention 的 SGLang v0.4 的 decode 吞吐为 v0.3 的 1.9 倍。[SGLang](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- **GLA**：Zadouri 等人提出的 Grouped Latent Attention 从模型结构上处理，把 latent 拆成两个 256 维的头，每个头只服务一半 query head，两张卡各存一半。[Zadouri et al.](https://arxiv.org/abs/2505.21487)

### 已有模型的转换

**GQA 模型改用 MLA 需要继续训练。** TransMLA 证明，在 KV cache 大小与 query head 数相同时，表达能力 GQA < MLA < MQA，其中 MQA 的 head dim 等于整个缓存的维度；论文给出把 Llama、Qwen 等 GQA checkpoint 转为 MLA 的方法，转换后用 6B token 微调恢复精度。[TransMLA](https://arxiv.org/abs/2502.07864)

## 七、现状

### 各模型的配置

**GQA 仍是多数模型的选择；MLA 被 Moonshot 与智谱沿用**，并与稀疏选择（GLM-5 的 DSA）或线性 attention（Kimi-K3 的 KDA）组合。

| 模型 | 形式 | query head / KV head | 每层每 token 缓存（元素） |
| --- | --- | --- | ---: |
| Llama-3.1-70B | GQA | 64 / 8 | 2048 |
| Qwen3-235B-A22B | GQA | 64 / 4 | 1024 |
| MiniMax-M3 | GQA | 64 / 4 | 1024 |
| DeepSeek-V3 | MLA | 128 / latent | 576 |
| Kimi-K2 | MLA | 64 / latent | 576 |
| GLM-5 | MLA | 64 / latent | 576 |
| Kimi-K3 | MLA，93 层中 24 层，其余为线性 attention | 96 / latent | 576 |
| DeepSeek-V4-Pro | shared-KV MQA | 128 / 1 | 512（每 4 或 128 个 token 一个 entry） |

::kv-cache-models::

### DeepSeek V4：Shared Key-Value MQA

**DeepSeek 在 V4 中去掉了 MLA 的上投影。** V4 的每个 KV entry 是一个 512 维向量，同时作为所有 query head 的 key 与 value，技术报告称为 Shared Key-Value MQA；config 中 `num_key_value_heads` 为 1，`head_dim` 为 512，不再有 `kv_lora_rank`。[DeepSeek-V4](https://arxiv.org/abs/2606.19348) 这与 MLA 吸收后的计算形态相同，区别在于 V4 训练时就直接使用这一形态，不经过 $W^{UK}$、$W^{UV}$ 的还原。

与 MLA 相比，V4 还有三处相应的改动：[DeepSeek-V4](https://arxiv.org/abs/2606.19348)

- **位置编码**：RoPE 只作用于每个向量的最后 64 维，不再单独缓存 RoPE key。由于 value 与 key 是同一个向量，加权求和的结果带有 key 的绝对位置，V4 对输出的这 64 维再施加位置 $-i$ 的旋转，使结果只依赖相对位置。
- **输出投影**：128 个 head 各输出 512 维，直接投影回 hidden state 的计算量过大，V4 把 head 分为 16 组，每组先投影到 1024 维再合并。
- **存储精度**：KV entry 中 RoPE 的 64 维用 BF16 存储，其余维度用 FP8。

V4 还沿序列方向压缩：每 4 个或 128 个 token 合成一个 KV entry，并由 indexer 选出一部分 entry 参与计算。这一方向减少的是参与 attention 的 token 数，与每 token 的缓存维度相互独立。两者叠加，1M token 上下文下 V4 的 KV cache 约为 BF16 GQA-8（head dim 128）的 2%。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)

## 八、小结

- **MQA 与 GQA**：多个 query head 共享 KV head，KV cache 缩小为 MHA 的 $h_{kv}/h$，代价是质量损失。
- **MLA**：只缓存一个 512 维的 latent 和一个 64 维的 RoPE key，借矩阵吸收在 decode 时直接对 latent 做 MQA。它用计算换读取，head 数决定 decode 的 FLOPs，TP 下 latent 在每张卡上重复。
- **DeepSeek V4**：去掉上投影，直接训练 512 维、key 与 value 共用的单个 KV head。

## 参考

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Shazeer. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). 2019.
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
- DeepSeek-AI. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). 2024.
- Zadouri, Strauss, Dao. [Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487). 2025.
- Meng et al. [TransMLA: MLA Is All You Need](https://arxiv.org/abs/2502.07864). 2025.
- Kimi Team. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534). 2025.
- DeepSeek-AI. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348). 2026.
- LMSYS. [SGLang v0.4: Data Parallelism Attention for DeepSeek Models](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). 2024.
- vLLM [`mla_attention.py`](https://github.com/vllm-project/vllm/blob/e006d76/vllm/model_executor/layers/attention/mla_attention.py)，v0.30.0（e006d76）。
- 各模型 Hugging Face 上的 `config.json`（2026-10-08 读取）：Llama-3.1-70B、Qwen3-235B-A22B、MiniMax-M3、DeepSeek-V3、Kimi-K2、GLM-5、Kimi-K3、DeepSeek-V4-Pro。
