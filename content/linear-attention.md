线性 attention 去掉 softmax，把全部历史写进一个固定大小的矩阵，这个矩阵称为 state。Decode 每步只读写 state，开销与上下文长度 $L$ 无关；full attention 每步要读取全部 $L$ 个 token 的 KV。State 只能有损地保存历史，所以已发布的模型都把线性层与少量 full attention 层混合使用。Qwen3.8-27B 的 64 层中有 48 层是 Gated DeltaNet，256K 上下文下每个请求的缓存为 16.1 GiB，64 层全部使用 full attention 时为 64 GiB。

**TL;DR**

- **矩阵 state**：所有 token 的 $v k^\top$ 累加成一个矩阵，decode 每步的读写量固定。
- **遗忘与写入**：decay gate 让 state 整体衰减，delta rule 只改写当前 key 对应的 value；Gated DeltaNet 两者都用，KDA 把衰减细化到每个 channel。
- **训练**：chunk 内用矩阵乘并行，chunk 之间串行传递 state。
- **混合**：state 做不到精确检索，Qwen 与 Kimi 都是 3 个线性层配 1 个 full attention 层。
- **推理**：短上下文下 state 比 KV 还大；prefix caching 与 speculative decoding 都要为 state 另做处理。
- **反对意见**：MiniMax 从线性混合模型退回 full attention；DeepSeek 选择了稀疏与压缩。

## 一、从 Softmax 到矩阵 State

Full attention 中，softmax 的分母依赖当前的 $q_t$，每个 query 都要与全部历史 key 重新计算：

$$
o_t = \sum_{i=1}^{t} \frac{\exp(q_t^\top k_i)}{\sum_{j=1}^{t}\exp(q_t^\top k_j)}\, v_i
$$

::spec-linear::

**去掉 softmax 后，求和可以先做。** 把 $\exp(q_t^\top k_i)$ 换成 $q_t^\top k_i$，按结合律：

$$
o_t = \Big(\sum_{i=1}^{t} v_i k_i^\top\Big)\, q_t = S_t\, q_t,\qquad S_t = S_{t-1} + v_t k_t^\top
$$

$S_t$ 即 state，每个 head 一个 $d_v \times d_k$ 的矩阵。每个 token 写入一个外积 $v_t k_t^\top$，即一条从 key 到 value 的关联；读取是一次矩阵–向量乘。

::state-step::

**State 是 key-value 记忆，容量有限。** 用历史 token 的 key $k_j$ 读取：

$$
S_t k_j = v_j\,\|k_j\|^2 + \sum_{i \neq j} v_i\,(k_i^\top k_j)
$$

第二项是其他 token 的 value 混入的部分，本文称为串扰。只有 key 互相正交时串扰为零，而 128 维空间最多只有 128 个互相正交的向量。原始的线性 attention 只加不删，串扰随上下文累积。

## 二、遗忘：Decay Gate

Decay gate 在写入前把 state 整体乘一个 0 到 1 之间的衰减系数 $\alpha_t$，$\alpha_t$ 由当前 token 算出：

$$
S_t = \alpha_t\, S_{t-1} + v_t k_t^\top
$$

::spec-decay::

**Decay 等价于随距离衰减的 attention 权重。** 展开递推，token $i$ 的贡献乘上了 $i$ 到 $t$ 之间所有 $\alpha$ 的乘积：

$$
o_t = \sum_{i=1}^{t} \Big(\prod_{j=i+1}^{t} \alpha_j\Big)\,(q_t^\top k_i)\, v_i
$$

累乘的系数同时提供了位置信息，Kimi Linear 把它解释为由数据决定的乘性位置编码，作用与 RoPE 相当。

**粒度有两种。** Mamba2 与 Gated DeltaNet 每个 head 一个标量 $\alpha_t$；GLA 与 KDA 每个 key channel 一个，同一个 head 中可以同时保存长期与短期的内容。

**Decay 不能有选择地删除。** 乘以 $\alpha_t$ 使所有关联同时变弱，无法只删去与当前 key 冲突的一条。

## 三、Delta Rule 与 Gated DeltaNet

Delta rule 写入前先读出当前 key 已存的值 $S_{t-1} k_t$，只写入 $v_t$ 与它的差，$\beta_t \in (0,1)$ 称为写入强度：

$$
S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1} k_t)\,k_t^\top
$$

$k_t$ 为单位向量、$\beta_t = 1$ 时，写入后 $S_t k_t = v_t$，同一个 key 的旧 value 被替换。Gated DeltaNet（GDN）先衰减，再写入差值：

$$
S_t = \alpha_t S_{t-1} + \beta_t\,(v_t - \alpha_t S_{t-1} k_t)\,k_t^\top
$$

::spec-gdn::

下图中，state 已有 $k \to v_1$ 与 $k' \to u$ 两条关联，新 token 要把 $k$ 对应的 value 改为 $v_2$：

::delta-write::

- **直接相加**：$k$ 读出 $v_1 + v_2$，新旧混在一起。
- **先 decay**：$v_1$ 变弱，无关的 $u$ 也同样变弱。
- **Delta rule**：$k$ 读出 $v_2$，$u$ 不变；前提是 $k$ 与 $k'$ 正交。

**两种机制在检索实验中互补。** Gated DeltaNet 论文在 1.3B 模型上测试了从长文本中找回一对 key 与 value 的任务（RULER 的 S-NIAH）：

::niah-bars::

**实际的 GDN 层还有几个部件**：short convolution 补充局部信息，$q$、$k$ 做 L2 归一化以稳定 delta rule，输出经 RMSNorm 与 sigmoid gate。

::qwen-gdn-module::

## 四、Chunkwise 计算

逐 token 递推只能串行执行，也无法使用 Tensor Core。训练与 prefill 把序列切成长度为 $C$ 的 chunk（通常 64 或 128），每个 chunk 的输出分两部分，都是矩阵乘法：

$$
O_{[n]} = \underbrace{Q_{[n]}\, S_{[n]}^\top}_{\text{读入口 state}} + \underbrace{\big(Q_{[n]} K_{[n]}^\top \odot M\big)\, V_{[n]}}_{\text{chunk 内部}}
$$

$S_{[n]}$ 是第 $n$ 个 chunk 开始时的 state，$M$ 为下三角 mask。算完后 state 更新为 $S_{[n+1]} = S_{[n]} + V_{[n]}^\top K_{[n]}$。

::chunkwise::

**串行步数从 $L$ 降到 $L / C$。** 4K token 在 $C = 64$ 时只串行传递 64 次 state。Delta rule 中每次写入依赖前一步的 state，DeltaNet 用 WY 表示把 chunk 内的多次秩一更新合并为矩阵运算，chunk 内仍以矩阵乘为主。

## 五、KDA

Kimi Delta Attention（KDA）把 GDN 每个 head 的标量 $\alpha_t$ 换成长度为 $d_k$ 的向量：

$$
S_t = S_{t-1}\,\mathrm{Diag}(\alpha_t)\,(I - \beta_t k_t k_t^\top) + \beta_t\, v_t k_t^\top
$$

::spec-kda::

**逐 channel 的衰减对应 RoPE 的多个频率。** RoPE 给每对维度一个旋转频率，GDN 一个 head 只有一个衰减速度，KDA 让每个 channel 有自己的衰减速度。

**逐 channel 的衰减带来数值范围问题。** Chunk 内要除以累积的衰减系数，衰减过快时倒数在半精度下溢出。Kimi K3 把每步的对数衰减限制在 −5 以上，16 个 token 内累积不低于 −80，$e^{80}$ 在 BF16 范围内，全部计算都能用 Tensor Core。

**同等规模下 KDA 的长上下文结果最好。** Kimi Linear 论文中三个 48B-A3B 模型各训练 1.4T token，在 128K 长文本上评测：

| 模型 | 线性层 | 平均分 | RULER |
| --- | --- | ---: | ---: |
| MLA | 无 | 52.2 | 81.3 |
| GDN-H | GDN，3:1 混合 | 51.2 | 80.5 |
| Kimi Linear | KDA，3:1 混合 | 54.5 | 84.3 |

## 六、混合架构

**固定大小的 state 做不到精确检索。** Jelassi 等人证明两层 Transformer 能复制指数长度的字符串，state 固定的模型做不到。Gated DeltaNet 论文中，纯 GDN 的检索平均分为 30.6，Transformer 为 37.0，GDN 与 sliding window attention 交替后为 39.0。

**主流比例是 3:1。** 各模型的层排布：

::layer-strips::

Kimi Linear 消融中的 validation perplexity（越低越好）：

| 线性层 : full attention | 0 : 1 | 1 : 1 | 3 : 1 | 7 : 1 | 15 : 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation perplexity | 5.77 | 5.66 | **5.65** | 5.70 | 5.82 |

1:1 与 3:1 效果相当，但 full attention 层多一倍。Qwen3.8-Next 的报告中，同为 3:1，GDN 混合的九项平均分为 53.81，sliding window 混合为 51.15，全部 full attention 为 49.87。

**Full attention 层是否用位置编码，两家结论相反：**

| | Kimi Linear、K3 | Qwen3.5–3.8 |
| --- | --- | --- |
| full attention 层 | 不用位置编码，位置交给 KDA 的衰减 | 保留 RoPE |
| 理由 | 不调整任何参数即外推到 1M | 去掉后，post-training 的模型更容易出现无法终止的生成 |

**线性与稀疏可以叠加。** Qwen3.8-Flash-Next 在继续预训练时把 12 个 full attention 层换成 Qwen Sparse Attention（QSA），每个 query 只读 indexer 选出的 micro-block。

## 七、推理系统中的 State

混合模型的每个请求同时占用两种缓存：线性层每层一份固定大小的 state，full attention 层的 KV cache 随上下文增长。

::hybrid-layers::

**短上下文下 state 比 KV 大。** 线性层的 state 与“同样的层改用 full attention 时的 KV”相等的上下文长度，本文称为交叉点；state 按 FP32、KV 按 BF16 计：

::state-vs-kv::

Qwen3.8-27B 的交叉点为 768 个 token，Kimi K3 约为 5.5K：MLA 下每个 token 的 KV 已经很小，线性层要在更长的上下文下才节省显存。Decode 每步要把 state 读出再写回，按读写量算交叉点再翻倍，分别为 1.5K 与约 11K。

**Prefix caching 只能在保存了 state 的位置命中。** Full attention 的 KV 逐 token 保存，任意 block 边界都能复用；state 只对应当前位置，复用前缀须保存前缀末尾的 state 副本，本文称为 checkpoint。一个 checkpoint 有几百 MiB，只能稀疏保存。Kimi K3 把物理 block 设为 6144 个 token、hash 单位设为 512 个 token，在 512 的边界保存 checkpoint：

::k3-prefix-cache::

**Speculative decoding 要能撤回 state。** State 原地更新，验证后已越过最后一个被接受的 token；为每个 draft 位置保存一份 state 会使读写量成倍增加。Kimi K3 只缓存 draft token 投影后的输入，验证后按接受的个数在片上重放递推。

## 八、反对意见

| MiniMax 的模型 | Attention 结构 |
| --- | --- |
| MiniMax-Text-01（2025 年 1 月） | 7 层 lightning attention 配 1 层 full attention |
| MiniMax-M2（2025 年 10 月） | 全部 full attention |
| MiniMax-M3 | GQA 加按 block 选择的稀疏 attention，没有线性层 |

**MiniMax 退回 full attention 的三点理由：**

- **规模**：混合模型在小规模评测上与 full attention 持平，规模扩大后在多跳推理上出现明显缺陷；评测饱和后，缺陷可能出现在未覆盖的任务上。
- **收益起点**：线性 attention 要到几千 token 以后才更便宜，与 Kimi K3 的交叉点同一量级。
- **基础设施**：state 对低精度更敏感，prefix caching 与 speculative decoding 缺少成熟做法；Kimi K3 用 checkpoint 与重放处理后两点。

**DeepSeek 选择了另一条路线。** DeepSeek-V3.2 与 V4 用稀疏选择与压缩减少 decode 的读取量，全部保留 softmax attention。

## 参考

- Katharopoulos et al. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). ICML 2020.
- Dao, Gu. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). ICML 2024.
- Yang et al. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). NeurIPS 2024.
- Yang, Kautz, Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). ICLR 2025.
- Jelassi et al. [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032). ICML 2024.
- MiniMax. [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313). 2025.
- MiniMax. [Why Did M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model). 2025.
- Kimi Team. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692). 2025.
- Kimi Team. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf). 2026.
- Qwen Team. [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320). 2026.
- Yang. [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/). 2024.
- vLLM Team. [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview). 2026.
- vLLM 源码 `ac18817e`：`vllm/v1/kv_cache_interface.py`（`MambaSpec`）、`vllm/v1/core/single_type_kv_cache_manager.py`（`MambaManager`）、`vllm/config/cache.py`、`vllm/model_executor/layers/mamba/mamba_utils.py`。
- 各模型 Hugging Face 上的 `config.json`（2026-10-10 读取）：MiniMax-Text-01、MiniMax-M3、Qwen3-Next-80B-A3B-Instruct、Qwen3.8-27B、Qwen3.8-Flash-Next、Kimi-Linear-48B-A3B-Instruct、Kimi-K3。
