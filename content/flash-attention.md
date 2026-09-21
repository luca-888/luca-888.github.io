# FlashAttention-2：从 Online Softmax 到 GPU 并行实现

本文从 Online Softmax 讲起，再对照 [FlashAttention v2.8.3 源码](https://github.com/Dao-AILab/flash-attention/tree/060c9188beec3a8b62b33a3bfa6d5d2d44975fab) 阅读前向实现。范围为 BF16、等长 self-attention、Q/K/V head 数相同的普通前向，包括 causal mask；不展开 split-KV、dropout、local attention、ALiBi 和 softcap。

## 一、Attention 的显存开销

对于一个 head，设序列长度为 $N$，head dimension 为 $d$，$Q,K,V\in\mathbb{R}^{N\times d}$。Attention 的计算为：

$$
S=QK^\top,\qquad a=\frac{1}{\sqrt d},\qquad
P=\operatorname{softmax}(aS),\qquad O=PV.
$$

Softmax 沿行归一化，每一行输出都是所有 value 的加权和。

分开执行这三个算子时，$S$ 和 $P$ 通常都要写入显存，再由下一个算子读出。它们各有 $N^2$ 个元素，而输入和输出的大小只有 $O(Nd)$。序列越长，中间矩阵的存储和读写成本越突出。

FlashAttention 将这些计算融合：每次只算一个分数 tile（数据分块），在片上转成权重并与 V 相乘，随后复用这片空间。**它省下的是完整中间矩阵的存储与读写，dense attention 的计算复杂度仍为 $O(N^2d)$。** [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

## 二、分块算法：逐行执行 Online Softmax

### 分块计算与行状态

数据分块称为 **tile**，切分操作称为 **tiling**。取一个 shape 为 $[B_M,d]$ 的 Q tile，记为 $Q_i$，将 K、V 切成 shape 为 $[B_N,d]$ 的 tiles。每轮计算一个分数 tile：

$$
S_{ij}=Q_iK_j^\top\in\mathbb{R}^{B_M\times B_N}.
$$

难点在于 Softmax 的分母依赖整行。各 tile 分别归一化，会得到不同的分母，无法直接累加输出。

::flash-attention-tiling::

解决方法是 **Online Softmax 多维护一个 value 加权和，最后再统一归一化**。对于每个 query 行，跨 tile 保留三个量：

| 状态 | 作用 |
| :--- | :--- |
| 最大值 $m$ | 已处理分数的最大值，减去它以避免指数溢出 |
| 指数和 $\ell$ | 累加未归一化权重，作为最终分母 |
| 加权和 $U$ | 累加“权重乘以 value”，作为最终分子 |

$m$ 和 $\ell$ 是标量，$U$ 是长度为 $d$ 的向量。下面使用缩放后的分数 $aS$，减去目前见过的最大值再取指数；处理完所有 K/V tile 后，才用 $U/\ell$ 得到输出。

### 跨 Tile 的状态更新

新的 K/V tile 可能带来更大的最大值。此时，旧的 $\ell$ 和 $U$ 要一起换算到新基准，再加上新 tile 的贡献。

以一个 query、四个 key 为例，每个 value 都是二维向量（$d=2$）。缩放后的分数 $aS$ 与 $V$ 按每个 tile 两个 key 切分：

$$
aS=\left[\begin{array}{c|c}aS_1&aS_2\end{array}\right]
=\left[\begin{array}{cc|cc}0&\ln2&\ln4&\ln8\end{array}\right],
\qquad
V=\left[\begin{array}{c}V_1\\\hline V_2\end{array}\right]
=\left[\begin{array}{cc}1&0\\0&1\\\hline 2&0\\0&2\end{array}\right].
$$

**分数按列切，V 按行切**：第一个 tile 取 $aS$ 的前两列与 $V$ 的前两行，第二个 tile 取后两列与后两行。每个分数对应 V 的一整行。

::flash-attention-example::

## 三、并行分工：从 CTA 到 Warp

将第二章的计算映射到 GPU，可以从外到内看两层分工：CTA 之间分配 Q tile，CTA 内部再由不同 warp 负责不同 query 行。

### CTA 之间：分配 Q Tile

FA2 将不同 batch、head 和 Q tile 分配给独立的 CTA（CUDA 线程块），每个 CTA 负责对应的输出行。例如，一个 head 有 4096 个 query，按每块 128 行划分，就得到 32 个 CTA。

FA1 前向只沿 batch 和 head 并行；FA2 增加 Q tile 这一维，让单个 head 也能提供多个独立任务，改善小 batch、长序列时并行度不足的问题。[FA2 论文 §3.2](https://arxiv.org/html/2307.08691v1#S3.SS2)

### Warp 之间：分配 Query 行

进一步看一个 CTA 内部：它正在计算一个 Q tile 与当前 K/V tile。这里的“划分 Q 或 K/V”，指的是**这些 tile 在 warp 之间如何分工**。

以一个包含 128 个 query 的 Q tile、4 个 warp 为例，沿用图中的 Warp 1–4 编号：

- **FA1 沿 K/V 划分**：将当前 K/V tile 的 key/value 分成 4 组，各 warp 处理一组。对于同一个 query，Warp 1 累加第 1 组 value 的加权贡献，Warp 2 累加第 2 组，Warp 3、4 同理。每个 warp 只得到同一输出行的部分和，需要通过 shared memory 合并。
- **FA2 沿 Q 划分**：将 128 个 query 行分给 4 个 warp，可以理解为每个 warp 负责 32 行，具体行号由 MMA layout 决定。每个 warp 使用当前 K/V tile 的全部 key/value，只计算自己负责的 query 行，因此无需跨 warp 合并输出。

::flash-attention-warps::

这里的“完整输出行”指不缺其他 warp 的输出贡献。各 warp 仍需遍历后续 K/V tile，逐轮更新自己的 Online Softmax 状态，最后归一化，才能得到最终输出。

源码用下面的布局将 warp 排在矩阵乘法的 M 方向，也就是 query 行方向：

```cpp
using TiledMma = TiledMMA<
    typename Base::MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>,_1,_1>>,
    Tile<Int<16 * kNWarps>, _16, _16>>;
```

`Shape<Int<kNWarps>,_1,_1>` 表示在 M、N、K 三个方向上分别排列 `kNWarps`、1、1 个 warp，即所有 warp 沿 M 方向分工。$QK^\top$ 与“权重乘 V”的输出行都对应 query，因此两次矩阵乘法保持同一套行分工。一个 warp 内仍由多个 lane 协作完成计算与归约，CTA 内共享 K/V 缓冲区的同步也仍然需要。[MMA 布局](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/kernel_traits.h#L62-L77)

## 四、Kernel 实现：计算、访存与同步

### 主循环与数据存储

启动 kernel 时，grid 的三个维度分别是 Q tile、batch 和 head：

```cpp
const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
```

GPU 按可用资源调度这些 CTA，每个 CTA 固定自己的 Q tile，遍历 K/V tiles。[Kernel 启动](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_launch_template.h#L63-L64)

CTA 内部的循环位于 `compute_attn_1rowblock`，每轮依次调用 `gemm`（QK 点积）、`softmax_rescale_o`（更新权重和状态）、`gemm_rs`（权重乘 V 并累加输出）。这三个调用在每个 K/V tile 上重复执行。[前向主循环](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_kernel.h#L377-L436)

| 源码变量 | 命名与用途 | 存储位置 / 精度 | 保留多久 |
| :--- | :--- | :--- | :--- |
| `sQ`、`sK`、`sV` | `s` 表示 shared memory；Q/K/V 的片上缓冲区，供装载到寄存器 | Shared memory / BF16 | Q 在遍历中复用，K/V 随 tile 更换 |
| `acc_s` | `acc` 表示 accumulator；先累加 QK 点积，再原地改为指数权重 | 寄存器 / FP32 | 当前轮，用完后复用 |
| `rP` | `r` 表示 register；由 `acc_s` 转换得到的未归一化权重，作为第二次矩阵乘法的输入 | 寄存器 / BF16 | 当前轮 |
| `row_max` | 每个 query 的原始点积最大值 $m$ | 寄存器 / FP32 | 跨轮更新 |
| `row_sum` | 每个 query 的指数和 $\ell$，由线程各自保留局部和 | 寄存器 / FP32 | 跨轮累加，最后合并 |
| `acc_o` | 输出累加器，保存加权和 $U$ | 寄存器 / FP32 | 跨轮更新，最后归一化并转为 BF16 写回显存 |

前缀是命名习惯，不决定存储位置：`acc_s`、`acc_o` 与 `rP` 都是线程在寄存器中持有的矩阵片段（fragment）。[数据类型](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/kernel_traits.h#L18-L34)

### 状态更新与最终归一化

第二章中“旧累计值乘以 $1/4$”的操作，对应下面的代码。`scores_scale` 是根据新旧最大值计算的换算系数，同时作用于 `row_sum` 和 `acc_o`：

```cpp
float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
row_sum(mi) *= scores_scale;
#pragma unroll
for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
```

这段代码位于 `softmax_rescale_o`，之后由 `scale_apply_exp2` 计算新权重，`reduce_sum<false>` 更新线程局部指数和。[状态更新](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/softmax.h#L136-L166)

与第二章直接使用 $aS$ 不同，源码保存原始点积的最大值 $m$，在指数计算时同时缩放分数与最大值。对原始分数 $s$，`exp2f` 使用系数 $a\log_2e$，计算 $2^{a(s-m)\log_2e}=e^{a(s-m)}$，数学结果相同。[指数计算](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/softmax.h#L65-L92)

循环结束后，`normalize_softmax_lse` 合并完整指数和，将其沿 $d$ 维广播并归一化 `acc_o`。它还保存每个 query 的 log-sum-exp：$L=am+\log\ell$。反向计算可用 Q、K 和 LSE 重建需要的 P tile，避免保存完整 P。[归一化与 LSE](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/softmax.h#L169-L185)

普通标量、向量运算的吞吐低于 Tensor Core 矩阵乘法，Softmax 中的额外运算也可能成为瓶颈。将归一化推迟到循环结束，省去了 FA1 每轮更新归一化输出的重复运算。

指数和的跨线程合并也推迟到最后：`reduce_max` 每轮在 4 个 lane 之间归约，确保共同处理一个 query 的线程使用相同的最大值；`reduce_sum` 只累加局部和，循环结束后才执行 `quad_allreduce_`。

### Causal Mask 与 Tile 遍历范围

等长 causal self-attention 只允许第 $i$ 行读取 $j\le i$ 的 key。固定一个 Q tile，K/V tile 分成三类：

- 整个 tile 不可见：直接跳过，不做矩阵乘法。
- 部分可见：计算分数后，将不可见元素设为 $-\infty$，使指数权重为 0。
- 整个 tile 可见：正常计算，无需逐元素 causal 判断。

::flash-attention-causal-tiles::

源码用 `n_block_max` 限定可见的 K/V 范围，再从右向左遍历：先处理对角线附近和序列尾部的边界 tile，再进入无需 causal mask 的主循环。$B_M>B_N$ 时，对角线可能跨过多个 K tile，边界处理也会执行多轮。[可见范围](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_kernel.h#L80-L91)、[分段循环](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_kernel.h#L267-L302)

Online Softmax 支持这种倒序累加。tile 遍历顺序不影响实数运算下的结果，有限精度下则可能有舍入差异。

### 异步搬运与同步

FA2 使用 `cp.async` 异步复制，将显存数据搬到 shared memory。主循环中：

- 先提交当前 V tile 的搬运，再计算 QK，使用 V 前等待搬运完成。
- 提前搬运下一个 K tile，与当前 tile 的 Softmax 和输出累加重叠。

`cp_async_wait` 等待复制完成，`__syncthreads()` 协调 block 内共享缓冲区的读写。这些等待保证矩阵乘法读到完整数据。[异步搬运与计算顺序](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/flash_fwd_kernel.h#L377-L428)

## 参考资料

- [FlashAttention：IO-aware exact attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2：并行与工作划分](https://arxiv.org/abs/2307.08691)
- [官方源码 v2.8.3 · 固定 commit](https://github.com/Dao-AILab/flash-attention/tree/060c9188beec3a8b62b33a3bfa6d5d2d44975fab)
