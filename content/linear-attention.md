线性 attention 去掉 softmax，把全部历史写进一个固定大小的矩阵，这个矩阵称为 state。Decode 每步只读写 state，开销与上下文长度 $L$ 无关；full attention 每步要读取全部 $L$ 个 token 的 KV。State 只能有损地保存历史，所以已发布的模型都把线性 attention 层与少量 full attention 层混合使用。Qwen3.8-27B 的 64 层中有 48 层是 Gated DeltaNet，256K 上下文下每个请求的缓存为 16.1 GiB，64 层全部使用 full attention 时为 64 GiB。

**TL;DR**

- **矩阵 state**：所有 token 的 $v k^\top$ 累加成一个矩阵，decode 每步的读写量固定。
- **遗忘与写入**：decay gate 让 state 整体衰减，delta rule 只改写当前 key 对应的 value；Gated DeltaNet 两者都用，KDA 把衰减细化到每个 channel。
- **训练**：chunk 内用矩阵乘并行，chunk 之间串行传递 state。
- **混合**：state 做不到精确检索，Qwen 与 Kimi 都是 3 个线性 attention 层配 1 个 full attention 层。
- **推理**：短上下文下 state 比 KV 大；prefix caching 与 speculative decoding 都要为 state 另做处理。
- **路线比较**：稀疏 attention 的主 attention 每步也只读选出的少量 entry，但 indexer 仍扫过全部 token，KV 仍逐 token 保存；线性 attention 层连存储也不随上下文增长，代价是记忆有损。Kimi 与 Qwen 选线性 attention 混合，DeepSeek、GLM 与 MiniMax 选稀疏 attention，MiniMax 曾从线性 attention 混合退回。

## 一、从 Softmax 到矩阵 State

Full attention 中，softmax 的分母依赖当前的 $q_t$，每个 query 都要与全部历史 key 重新计算：

$$
o_t = \sum_{i=1}^{t} \frac{\exp(q_t^\top k_i)}{\sum_{j=1}^{t}\exp(q_t^\top k_j)}\, v_i
$$

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

第二项是其他 token 的 value 混入的部分，本文称为串扰。只有 key 互相正交时串扰为零，而 $d_k = 128$ 时最多只有 128 个互相正交的 key。原始的线性 attention 只加不删，串扰随上下文累积。

## 二、遗忘：Decay Gate

Decay gate 在写入前把 state 整体乘一个 0 到 1 之间的衰减系数 $\alpha_t$，$\alpha_t$ 由当前 token 算出：

$$
S_t = \alpha_t\, S_{t-1} + v_t k_t^\top
$$

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

下图中，state 已有 $k \to v_1$ 与 $k' \to u$ 两条关联，新 token 要把 $k$ 对应的 value 改为 $v_2$：

::delta-write::

- **直接相加**：$k$ 读出 $v_1 + v_2$，新旧 value 叠加。
- **先 decay**：$v_1$ 变弱，无关的 $u$ 也同样变弱。
- **Delta rule**：$k$ 读出 $v_2$，$u$ 不变；前提是 $k$ 与 $k'$ 正交。

**两种机制在检索实验中互补。** Gated DeltaNet 论文在 1.3B 模型上测试了从长文本中找回一对 key 与 value 的任务（RULER 的 S-NIAH）。只有 decay 的 Mamba2 在 pass-key 任务上失败，因为目标在被读取前已随整体衰减变弱；只有 delta rule 的 DeltaNet 在 essay 中找数字的任务上失败，因为没有遗忘，大量无关 key 与目标 key 不正交，串扰超过了目标：

::niah-bars::

**完整的 GDN 层在递推之外还有三个部件**：short convolution 补充局部信息，$q$、$k$ 做 L2 归一化以稳定 delta rule，输出经 RMSNorm 与 sigmoid gate。

::qwen-gdn-module::

## 四、Chunkwise 计算

逐 token 递推只能串行执行，也无法使用 Tensor Core。训练与 prefill 把序列切成长度为 $C$ 的 chunk（通常 64 或 128），每个 chunk 的输出分两部分，都是矩阵乘法：

$$
O_{[n]} = \underbrace{Q_{[n]}\, S_{[n]}^\top}_{\text{读入口 state}} + \underbrace{\big(Q_{[n]} K_{[n]}^\top \odot M\big)\, V_{[n]}}_{\text{chunk 内部}}
$$

$S_{[n]}$ 是第 $n$ 个 chunk 开始时的 state，$M$ 为下三角 mask。算完后 state 更新为 $S_{[n+1]} = S_{[n]} + V_{[n]}^\top K_{[n]}$。

::chunkwise::

**串行步数从 $L$ 降到 $L / C$。** 4K token 在 $C = 64$ 时只串行传递 64 次 state。

**Delta rule 也能分块。** 每次 delta 写入是把 state 右乘 $(I - \beta_t k_t k_t^\top)$ 再加上 $\beta_t v_t k_t^\top$，依赖前一步的 state。一个 chunk 的 $C$ 次写入连乘后仍是一个秩不超过 $C$ 的修正，DeltaNet 把它写成 WY 表示：

$$
S_{[n+1]} = S_{[n]}\,\big(I - W_{[n]}^\top K_{[n]}\big) + U_{[n]}^\top K_{[n]},\qquad W_{[n]} = T\,K_{[n]},\quad U_{[n]} = T\,V_{[n]}
$$

$T$ 是一个 $C \times C$ 的下三角矩阵，由 $\beta$ 与 $K_{[n]} K_{[n]}^\top$ 经一次三角求解得到，其第 $i$ 行表示第 $i$ 个 token 的写入要先减去 chunk 内更早的 token 已写入的部分。串行的只剩这个 $C \times C$ 的求解，state 的更新与 chunk 内的输出都是矩阵乘。Decay 并入 $T$ 与 mask $M$ 的系数，GDN 与 KDA 用同一套分块。

## 五、KDA

Kimi Delta Attention（KDA）把 GDN 每个 head 的标量 $\alpha_t$ 换成长度为 $d_k$ 的向量：

$$
S_t = S_{t-1}\,\mathrm{Diag}(\alpha_t)\,(I - \beta_t k_t k_t^\top) + \beta_t\, v_t k_t^\top
$$

**逐 channel 的衰减对应 RoPE 的多个频率。** RoPE 给每对维度一个旋转频率，GDN 一个 head 只有一个衰减速度，KDA 让每个 channel 有自己的衰减速度。

**逐 channel 的衰减带来数值范围问题。** Chunk 内要除以累积的衰减系数，衰减过快时倒数在半精度下溢出。Kimi K3 把每步的对数衰减限制在 −5 以上，16 个 token 内累积不低于 −80，$e^{80}$ 在 BF16 范围内，全部计算都能用 Tensor Core。

**同等规模下 KDA 的长上下文结果最好。** Kimi Linear 论文中三个 48B-A3B 模型各训练 1.4T token，在 128K 上下文的多项长文本任务上评测，平均分与其中的 RULER 分数为：

| 模型 | 线性 attention 层 | 平均分 | RULER |
| --- | --- | ---: | ---: |
| MLA | 无 | 52.2 | 81.3 |
| GDN-H | GDN，3:1 混合 | 51.2 | 80.5 |
| Kimi Linear | KDA，3:1 混合 | 54.5 | 84.3 |

**四种机制的对照：**

| | 更新 | 写入 | 遗忘 | 代表 |
| --- | --- | --- | --- | --- |
| 线性 attention | $S_t = S_{t-1} + v_t k_t^\top$ | 直接相加 | 无 | Linear Transformer |
| Decay gate | $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ | 直接相加 | 整体乘 $\alpha_t$ | Mamba2、GLA |
| Gated DeltaNet | $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | 只写差值 | 每个 head 一个 $\alpha_t$ | Qwen3-Next、Qwen3.5–3.8 |
| KDA | $S_t = S_{t-1}\mathrm{Diag}(\alpha_t)(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | 只写差值 | 每个 key channel 一个 | Kimi Linear、Kimi K3 |

## 六、混合架构

**固定大小的 state 做不到精确检索。** Jelassi 等人证明两层 Transformer 能复制指数长度的字符串，state 固定的模型做不到。混合后不只是补上检索，整体效果也高于全部 full attention：

| 来源 | 结构 | 指标 | 分数 |
| --- | --- | --- | ---: |
| Gated DeltaNet 论文，1.3B | 纯 GDN | 检索任务平均 | 30.6 |
| | Transformer | | 37.0 |
| | GDN 与 sliding window attention 交替 | | 39.0 |
| Qwen3.8-Next 架构报告，25B-A3B | 全部 full attention | 九项评测平均 | 49.87 |
| | sliding window attention 与 full attention 3:1 | | 51.15 |
| | GDN 与 full attention 3:1 | | 53.81 |

**主流比例是 3:1。** 各模型的层排布：

::layer-strips::

Kimi Linear 消融中的 validation perplexity（越低越好）：

| 线性 attention 层 : full attention | 0 : 1 | 1 : 1 | 3 : 1 | 7 : 1 | 15 : 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation perplexity | 5.77 | 5.66 | **5.65** | 5.70 | 5.82 |

1:1 与 3:1 效果相当，但 full attention 层多一倍；7:1 开始变差，MiniMax-Text-01 为 7:1。

**Full attention 层是否用位置编码，两家结论相反：**

| | Kimi Linear、K3 | Qwen3.5–3.8 |
| --- | --- | --- |
| full attention 层 | 不用位置编码，位置信息由 KDA 的衰减提供 | 保留 RoPE |
| 理由 | 不调整任何参数即外推到 1M | 去掉后，post-training 的模型更容易出现无法终止的生成 |

**线性 attention 与稀疏 attention 可以叠加。** Qwen3.8-Flash-Next 在继续预训练时把 12 个 full attention 层换成 Qwen Sparse Attention（QSA），每个 query 只读 indexer 选出的 micro-block。

## 七、推理系统中的 State

混合模型的每个请求同时占用两种缓存：线性 attention 层每层一份固定大小的 state，full attention 层的 KV cache 随上下文增长。

::hybrid-layers::

**短上下文下 state 比 KV 大。** 线性 attention 层的 state 与“同样的层改用 full attention 时的 KV”相等的上下文长度，本文称为交叉点；state 按 FP32、KV 按 BF16 计：

::state-vs-kv::

Qwen3.8-27B 的交叉点为 768 个 token，Kimi K3 约为 5.5K：MLA 下每个 token 的 KV 已经很小，线性 attention 层要在更长的上下文下才节省显存。Decode 每步要把 state 读出再写回，按读写量算交叉点再翻倍，分别为 1.5K 与约 11K。

**Prefix caching 只能在保存了 state 的位置命中。** Full attention 的 KV 逐 token 保存，任意 block 边界都能复用；state 只对应当前位置，复用前缀须保存前缀末尾的 state 副本，本文称为 checkpoint。一个 checkpoint 有几百 MiB，只能稀疏保存。Kimi K3 把物理 block 设为 6144 个 token、hash 单位设为 512 个 token，在 512 的边界保存 checkpoint：

::k3-prefix-cache::

**Speculative decoding 要能撤回 state。** State 原地更新，验证后已越过最后一个被接受的 token；为每个 draft 位置保存一份 state 会使读写量成倍增加。Kimi K3 只缓存 draft token 投影后的输入，验证后按接受的个数在片上重放递推。

## 八、路线比较

减少 decode 每步 attention 读取量的路线有两条。线性 attention 层把历史压进固定大小的 state，读取量与上下文长度无关，代价是记忆有损，精确检索要靠混合进来的 full attention 层。稀疏 attention 保留每个 token 的 KV，主 attention 每步只读 indexer 选出的少量 entry，这部分读取量同样不随上下文增长，检索也精确；仍随上下文增长的是 indexer 对全部 indexer key 的扫描和缓存本身。DeepSeek 与 Kimi 各代表一条：

| | DeepSeek-V3.2 / V4 | Kimi K3 |
| --- | --- | --- |
| 历史如何保存 | 逐 token 保存 KV，V4 再按每 4 个或 128 个 token 压缩成一个 entry | 69 层只存固定大小的 state，24 层 MLA 逐 token 保存 KV |
| 每步读取 | indexer 选出的 top-k 个 entry，加窗口 | state 读写固定，MLA 层读全部 KV |
| 缓存随上下文 | 仍增长，V4 在 1M 下约为 BF16 GQA-8 的 2% | 线性 attention 层不增长，MLA 层增长 |
| 检索 | 精确，取决于 indexer 的选取是否准确 | 由 MLA 层负责，线性 attention 层只提供近似记忆 |
| 训练 | V3.2 由 full attention 的 V3.1 继续训练得到 | 从预训练开始使用线性 attention 层 |
| 推理系统 | 沿用 KV cache 体系，prefix caching 与 speculative decoding 照常 | state 要单独做 checkpoint 与重放 |

稀疏 attention 的优势是与现有模型和推理栈兼容，线性 attention 的优势是上下文越长收益越大。两者不互斥，Qwen3.8-Flash-Next 的 full attention 层换成 QSA 后，两条路线叠加在同一个模型里。

**MiniMax 从线性 attention 退回 full attention，再转向稀疏 attention：**

| MiniMax 的模型 | Attention 结构 |
| --- | --- |
| MiniMax-Text-01（2025 年 1 月） | 7 层 lightning attention 配 1 层 full attention |
| MiniMax-M2（2025 年 10 月） | 全部 full attention |
| MiniMax-M3 | GQA 加按 block 选择的稀疏 attention，没有线性 attention 层 |

MiniMax 给出三点理由：

- **规模**：混合模型在小规模评测上与 full attention 持平，规模扩大后在多跳推理上出现明显缺陷；评测饱和后，缺陷可能出现在未覆盖的任务上。
- **收益起点**：线性 attention 要到几千 token 以后开销才低于 full attention，与 Kimi K3 的交叉点同一量级。
- **基础设施**：state 对低精度更敏感，prefix caching 与 speculative decoding 缺少成熟做法；Kimi K3 用 checkpoint 与重放处理后两点。

**各家的选择：**

| 厂商 | 已发布模型的结构 | 路线 |
| --- | --- | --- |
| Kimi | K3：69 层 KDA 配 24 层 MLA | 线性 attention 混合 |
| Qwen | Qwen3.8：48 层 GDN 配 16 层 full attention；Flash-Next 的 full attention 层换成 QSA | 线性 attention 混合，叠加稀疏 attention |
| DeepSeek | V3.2 用 DSA；V4 用 CSA 与 HCA | 稀疏 attention 加压缩 |
| GLM | GLM-5：MLA 加 indexer 选 top-k | 稀疏 attention |
| MiniMax | M3：GQA 加按 block 选择 | 稀疏 attention |
| OpenAI gpt-oss、Google Gemma | sliding window 层与 full attention 层交替 | sliding window attention |

## 参考

- Katharopoulos et al. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). ICML 2020.
- Dao, Gu. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). ICML 2024.
- Yang et al. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). NeurIPS 2024.
- Yang, Kautz, Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). ICLR 2025.
- Jelassi et al. [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032). ICML 2024.
- MiniMax. [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313). 2025.
- DeepSeek-AI. [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf). 2025.
- DeepSeek-AI. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348). 2026.
- MiniMax. [Why Did M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model). 2025.
- Kimi Team. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692). 2025.
- Kimi Team. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf). 2026.
- Qwen Team. [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320). 2026.
- Yang. [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/). 2024.
- vLLM Team. [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview). 2026.
- vLLM 源码 `ac18817e`：`vllm/v1/kv_cache_interface.py`（`MambaSpec`）、`vllm/v1/core/single_type_kv_cache_manager.py`（`MambaManager`）、`vllm/config/cache.py`、`vllm/model_executor/layers/mamba/mamba_utils.py`。
- 各模型 Hugging Face 上的 `config.json`（2026-10-10 读取）：MiniMax-Text-01、MiniMax-M3、Qwen3-Next-80B-A3B-Instruct、Qwen3.8-27B、Qwen3.8-Flash-Next、Kimi-Linear-48B-A3B-Instruct、Kimi-K3；DeepSeek-V3.2、DeepSeek-V4-Pro、GLM-5、gpt-oss-120b、Gemma 3 27B（2026-10-09 读取）。
