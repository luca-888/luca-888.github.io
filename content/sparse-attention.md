稀疏 attention 让每个 query 只读取一部分历史 token 的 KV。Full attention（也称 dense attention）中，每个 query 读取全部 $L$ 个历史 token，decode 每步的读取量与 KV cache 都随 $L$ 线性增长；稀疏 attention 的目的是大幅减少单个 query 的读取量，同时让输出接近 full attention。

**TL;DR**

- **出发点**：softmax 之后大部分权重接近 0，只读一部分即可接近原输出。
- **固定模式**：sliding window 只读最近 $W$ 个 token，读取、计算与存储同时减少；远处信息只能逐层传递，所以与 full attention 层交替使用。
- **Attention sink**：第一个 token 是存放多余权重的位置，窗口把它移出缓存后，用 full attention 训练的模型失效；gpt-oss 与 DeepSeek-V4 改用每个 head 一个可学习的 sink logit。
- **学习式选择**：NSA 用压缩、选择、窗口三个分支，按 block 选择；DSA 简化为一个 indexer 逐 token 选 top-k，可以由 dense 模型继续训练，但只减少读取与计算，不减少存储。
- **压缩加选择**：DeepSeek-V4 的 CSA 先压缩再选 top-k，HCA 压缩后全部读取，每层另有窗口与 sink logit；读取与存储同时减少，1M 上下文的 KV cache 约为 BF16 GQA-8 的 2%。

## 一、Attention 的开销与稀疏性

缓存中供 attention 读取的一个单元称为一个 KV entry，不做压缩时每个 token 对应一个。Decode 一步的 attention 通常受显存带宽限制，耗时与读取的数据量近似成正比。下表以一层的元素数比较缓存与 decode 每步读取的数据量，$n_{kv}$ 为 KV head 数，$d_h$ 为 head dim，$d_c$ 为 MLA 每个 token 缓存的维度，$d'$ 为 DeepSeek-V4 一个 entry 的维度，$d_I$ 为 indexer key 的维度，$n_h$ 为线性层的 head 数，$W$ 为窗口，$k$ 为选出的 entry 数：

| 方案 | 代表模型 | 缓存 | 每步读取 |
| --- | --- | --- | --- |
| MHA / GQA | Llama-3.1-70B | $2n_{kv}d_h \cdot L$ | $2n_{kv}d_h \cdot L$ |
| MLA | DeepSeek-V3 | $d_c \cdot L$ | $d_c \cdot L$ |
| sliding window | Gemma 3 local 层 | $2n_{kv}d_h \cdot W$ | $2n_{kv}d_h \cdot W$ |
| DSA | DeepSeek-V3.2 | ${(d_c + d_I) L}$ | ${d_c k + d_I L}$ |
| CSA | DeepSeek-V4 | ${(d'{+}d_I) L/4 + d' W}$ | ${d'(k{+}W) + d_I L/4}$ |
| 线性 attention | Qwen3-Next 线性层 | $n_h d_h^2$ | $n_h d_h^2$ |

**GQA、MLA 与稀疏 attention 减少的是不同的因子。** GQA 与 MLA 减小每个 entry，读取量仍随 $L$ 增长；稀疏 attention 减少读取的 entry 数，两者可以叠加，DeepSeek-V3.2 即在 MLA 之上使用 DSA。DSA 与 CSA 的读取量中仍有一项随 $L$ 增长，即 indexer 读取的 indexer key，$d_I$ 为 128，远小于一个 entry。线性 attention 去掉 softmax，把历史累积进固定大小的 state，开销与 $L$ 无关，但 state 只能有损地保存历史，因此要与 full attention 层混合使用。

**Softmax 之后大部分权重接近 0。** 权重接近 0 的 token 对输出几乎没有贡献，只读取权重较大的 token，输出与 full attention 的差别很小，这是稀疏 attention 的出发点。按读取集合的确定方式，稀疏 attention 分为两类：

::sparse-read-set::

- **固定模式**：读取哪些 token 只由位置决定，代表是 sliding window。窗口之外的 KV 可以释放，读取、计算与存储同时减少。
- **学习式选择**：读取哪些 token 由内容决定。全部 KV 都要保留，减少的是读取与计算。

**收益随上下文增长。** 上下文不超过窗口或 top-k 时，读取的就是全部 token，与 full attention 等价。

::read-vs-context::

## 二、Sliding Window Attention

Sliding window attention 中每个 token 只读取最近 $W$ 个 token 的 KV，$W$ 称为窗口。

::spec-swa::

::mistral-swa::

**窗口之外的信息只能逐层间接传递。** 每过一层，信息沿序列最多传递 $W$ 个 token，途中经由中间 token 的 hidden state。

**现行的做法是 local 层与 global 层交替。** local 层使用 sliding window，global 层保留 full attention，让当前 token 直接读取任意位置。Gemma 3 每 5 个 local 层之后是 1 个 global 层，窗口为 1024，128K 上下文下 KV cache 约为全部使用 full attention 时的 17%；消融实验中，这一比例与 1 : 1 的 validation perplexity 相差在 0.03 以内。

::local-global-layers::

## 三、Attention Sink

Attention sink 是获得大量 attention 权重、自身语义却并不重要的 token，通常是序列的第一个 token。Sink 原指汇集并吸收多余之物的地方，这里吸收的是 softmax 分配不出去的权重。

::sink-attention::

**Sink 让 head 在无需读取时输出接近零。** Softmax 的权重之和恒为 1，当前 token 在某个 head 中没有需要读取的内容时，权重也必须分配出去。模型把这部分权重集中到第一个 token 上，第一个 token 的 value 向量范数远小于其他 token，加权后对输出的贡献接近零。第一个 token 对后面所有 token 都可见，所以成为 sink。上图中，Llama-2-7B 除最底部两层外，各层都把大量权重放在第一个 token 上。

**窗口会把 sink 移出缓存。** StreamingLLM 的实验中，Llama-2-13B 只缓存最近的 token 时 perplexity 为 5158，另外保留开头 4 个 token 时为 5.40。

::sink-scheme::

Sink 移出缓存后，softmax 的分母少了最大的一项，其余 token 的权重被同比例放大。下图中窗口内 4 个 token 的权重合计从 0.20 变为 1，这 4 个 token 写入当前 token 的内容变为原来的 5 倍；sink logit 把合计恢复为 0.20。

::sink-softmax::

**窗口移出 sink 后，模型需要另一个位置存放多余的权重。** 显式的做法有三种：

| 做法 | 内容 | 是否需要训练 | 采用者 |
| --- | --- | --- | --- |
| 保留开头的 token | 始终保留开头 4 个 token 的 KV | 否 | StreamingLLM |
| Sink token | 每条训练样本开头加一个可学习的 token | 是 | StreamingLLM 从头预训练的 160M 模型 |
| Sink logit | 每个 head 一个可学习的标量，加在 softmax 的分母中 | 是 | gpt-oss、DeepSeek-V4 |

Sink logit 记为 $z'_h$，head $h$ 中 query $i$ 的输出为：

$$
o_{h,i} = \frac{\sum_t \exp(z_{h,i,t})\, v_{h,t} + \exp(z'_h) \cdot \mathbf{0}}{\sum_t \exp(z_{h,i,t}) + \exp(z'_h)} = \frac{\sum_t \exp(z_{h,i,t})\, v_{h,t}}{\sum_t \exp(z_{h,i,t}) + \exp(z'_h)}
$$

$z_{h,i,t}$ 为 query $i$ 对 entry $t$ 的 logit，$v_{h,t}$ 为 entry $t$ 的 value。$z'_h$ 相当于一个 value 为零的虚拟 entry：它分走一部分权重，使真实 entry 的权重之和小于 1，但不向输出贡献内容。它是每个 head 的一个参数，不占 KV cache，窗口滑动也不会把它移出。

## 四、NSA

Native Sparse Attention（NSA）是 DeepSeek 在 2025 年 2 月提出的稀疏 attention，从预训练开始即使用，每个 query 经三个分支读取 KV，三个分支的输出由 gate 加权相加。NSA 只在论文的实验模型中使用，DeepSeek 已发布的模型改用了 DSA。

::spec-nsa::

::nsa-read-set::

- **Compression**：每 32 个 token 的 key 与 value 分别经一个 MLP 压成一个 entry，相邻 entry 的起点相隔 16 个 token；query 读取全部压缩后的 entry，得到整个上下文的粗粒度信息。
- **Selection**：把历史 token 按 64 个一组切成 block，选出 16 个：第一个 block 与最近 2 个 block 固定选中，其余按 compression 分支的分数取最高的，读取这些 block 中原始 token 的 KV，共 1024 个，补回压缩丢失的细节。
- **Sliding window**：读取最近 512 个 token。

三个分支分别提供整个上下文的概况、相关位置的原文细节和最近的局部上下文。单独设窗口分支，是为了让另外两个分支专注于远距离的依赖。

**Selection 分支不单独打分。** Compression 分支对每个压缩 entry 算出的 attention 分数，按位置重叠汇总成每个 block 的分数，再按分数取最高的 block，选择几乎不增加开销。

**选择以 block 为单位。** GPU 连续读取一段显存的吞吐远高于按下标随机读取，按 block 选择使 selection 分支能以连续访存读取 KV。

## 五、DSA

DeepSeek Sparse Attention（DSA）由 DeepSeek-V3.2-Exp 首先采用，V3.2 沿用：一个称为 lightning indexer 的打分模块为每个 query 选出 top-k 个 token，主 attention 只读取这些 token。

::spec-dsa::

::dsa-two-stage::

**Indexer 仍读取全部 token，但每个 token 只读很少的数据。** 每个历史 token 缓存一个 128 维的 indexer key，indexer 用这些 key 给全部历史 token 打分；主 attention 只读取分数最高的 2048 个 token 的 MLA entry，每个 576 维。

**DSA 减少读取与计算。** 主 attention 每步只读取、只计算 $k$ 个 token，prefill 中主 attention 的计算量从 $L^2$ 量级降到 $L \cdot k$，indexer 仍为 $L^2$ 量级，但每次打分只用 128 维的 indexer key，开销小得多。

**DSA 不减少存储。** 全部 token 的 KV 都要保留，每个 token 另存一个 indexer key，KV cache 比不使用 DSA 的 MLA 更大。

**Top-k 选择不可导，indexer 用单独的 loss 训练。** DeepSeek-V3.2-Exp 由使用 full attention 的 V3.1-Terminus 继续训练得到，indexer 用 KL 散度拟合主 attention 的权重分布。

**继续训练后效果基本不变。** V3.2-Exp 与 V3.1-Terminus 的 MMLU-Pro 均为 85.0。

学习式选择的同类方案在选择单位与打分方式上不同：

| 方案 | 选择单位 | 打分方式 | 每个 query 读取 |
| --- | --- | --- | --- |
| NSA | 64 个 token 的 block | compression 分支的 attention 分数 | 16 个 block |
| DSA（DeepSeek-V3.2） | token | indexer，64 个 head | 2048 个 token |
| GLM-5 | token | indexer，32 个 head | 2048 个 token |
| MiniMax-M3 | 128 个 token 的 block | indexer，4 个 head | 16 个 block |
| MoBA | 4K 个 token 的 block | query 与 block 内 key 的均值做点积 | 12 个 block（1M 上下文的配置） |

按 token 选择更精确，需要单独的 indexer；按 block 选择更粗，但打分开销小，读取的 KV 在显存中连续。

## 六、CSA 与 HCA

DeepSeek-V4 交替使用两种 attention 层。Compressed Sparse Attention（CSA）每 4 个 token 产生一个压缩 entry，再用 DSA 选出 top-k，indexer key 同样压缩；Heavily Compressed Attention（HCA）把每 128 个 token 压成一个 entry，不做选择，读取全部。下图为 DeepSeek-V4-Pro 在 16K 上下文下的情形。

::spec-csa::

::v4-read-set::

**压缩同时减少读取、计算与存储。** 压缩 entry 是相邻 token 的 KV 的加权和，权重由每个 token 的 hidden state 算出，在块内做 softmax。图中原始 token 不再缓存，CSA 层只存 4K 个 entry，HCA 层只存 128 个；CSA 的相邻 entry 有重叠，每个 entry 由 8 个 token 得到。

**CSA 看得细，HCA 看得全。** 图中 CSA 读取 1024 个 entry，每个只混合 8 个相邻 token，但只覆盖上下文的一部分；HCA 读取全部压缩 entry，加上窗口，覆盖整个上下文，但每个 entry 混合了 128 个 token。两种层交替，一层负责从全局找到大致位置，另一层负责读取细节。上下文越长，CSA 覆盖的比例越小，1M 上下文下不到 1%：

| 层 | 层数 | 缓存的压缩 entry | 每步读取 |
| --- | ---: | ---: | --- |
| CSA，每 4 个 token 一个 entry | 30 | 256K | top-1024，加窗口 128 |
| HCA，每 128 个 token 一个 entry | 31 | 8K | 全部 8K，加窗口 128 |

**窗口补上尚未压缩的 token。** 图中两种层都另读最近 128 个 token 的未压缩 KV。一个 entry 要等它覆盖的 token 都生成之后才能算出：HCA 每 128 个 token 压成一个 entry，已有 1000 个 token（编号 0 到 999）时，最新的 entry 只覆盖到第 895 个，第 896 到 999 个 token 还没有被压缩，窗口正好覆盖这段空缺。这里的窗口是在压缩 entry 之外额外读取，不像 sliding window attention 那样限制读取范围。

**V4 的部件都来自前几节：**

| V4 的部件 | 对应的做法 |
| --- | --- |
| HCA：压缩后全部读取 | NSA 的 compression 分支 |
| CSA：压缩后用 indexer 选 top-k | DSA，打分对象从原始 token 换成压缩 entry |
| 每层 128 token 的窗口 | sliding window，NSA 的 window 分支 |
| 每个 head 一个 sink logit | Attention sink 一节 |

与 NSA 不同，窗口与压缩 entry 进入同一次 softmax，不再由 gate 合并；与 DSA 不同，选择的对象是压缩 entry，存储也随之减少。

**V4 的 KV cache 很小。** 1M 上下文下约为 BF16 GQA-8（head dim 128）的 2%。

**检索效果在 128K 以内基本不变。** MRCR 8-needle 检索任务上，V4-Pro 在 128K 时为 0.92，在 1M 时为 0.59。

## 参考

- Xiao et al. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). ICLR 2024.
- Gu et al. [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/abs/2410.10781). ICLR 2025.
- Jiang et al. [Mistral 7B](https://arxiv.org/abs/2310.06825). 2023.
- Gemma Team. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786). 2025.
- OpenAI. [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925). 2025.
- Yuan et al. [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089). 2025.
- Lu et al. [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189). 2025.
- DeepSeek-AI. [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf). 2025.
- DeepSeek-AI. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348). 2026.
- Xiao. [How Attention Sinks Keep Language Models Stable](https://hanlab.mit.edu/blog/streamingllm). MIT HAN Lab blog, 2025.
- Raschka. [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants). 2026.
- 各模型 Hugging Face 上的 `config.json`（2026-10-09 读取）：Gemma 3 27B、gpt-oss-120b、DeepSeek-V3.2、DeepSeek-V4-Pro、GLM-5、MiniMax-M3。
