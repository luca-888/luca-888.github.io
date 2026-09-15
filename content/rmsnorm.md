# RMSNorm：计算原理与 GPU 算子优化

**RMSNorm 对每个 token 的 hidden 向量计算均方根，用它归一化输入，再按特征乘以可学习权重。**

输入记为 \(X[M,N]\)：\(M=\text{batch}\times\text{sequence}\) 是 token 数，\(N=\text{hidden size}\)。取一个 token 的输入向量 \(x=X[m,:]\)，\(x_i\) 表示第 \(i\) 个特征：

\[
y_i=\frac{x_i}{\operatorname{RMS}(x)}\cdot\gamma_i,
\qquad \text{where}\quad \operatorname{RMS}(x)=\sqrt{\epsilon+\frac{1}{N}\sum_{i=1}^{N}x_i^2}.
\]

每行独立计算 RMS，ε 用于防止除零。权重 γ 是长度为 N 的向量，同一列的所有 token 共享一个权重。

**归一化后还要乘 γ，因此输出的 RMS 不必为 1。** 在下面的示例中，可以调整输入与权重，观察输出如何变化。

::rmsnorm-demo::

LayerNorm 先减去均值，再按标准差缩放；RMSNorm 不减均值，直接按均方根缩放。两者都沿每个 token 的 hidden 维归约。

## 一、计算流程与 PyTorch 参考实现

代码用 `x[M,N]` 表示整批输入，`gamma[N]` 表示权重，`dgamma` 表示权重梯度；计算图展示其中一行。本文输入输出使用 BF16，中间计算使用 FP32。

### 1. 计算图与梯度流

`s` 为平方均值，`r = rsqrt(s + eps)`，`x_hat = x * r`；`g` 为输出 y 的上游梯度。Backward 中，x 直接参与缩放，也通过 r 影响输出，**这两条路径的梯度需要相加**。

::rmsnorm-graph::

### 2. Forward 参考实现

返回 `y`，并保留 FP32 的 `r[M,1]` 供 Backward 复用。

```python
def rmsnorm_forward_reference(x, gamma, eps=1e-6):
    x_dtype = x.dtype
    x, gamma = x.float(), gamma.float()
    s = x.square().mean(dim=1, keepdim=True)
    r = torch.rsqrt(s + eps)
    x_hat = x * r
    y = x_hat * gamma
    return y.to(x_dtype), r
```

### 3. Backward 参考实现

Backward 接收上游梯度 `g[M,N]`，复用 Forward 保存的 r，返回 dx 与 dgamma。

```python
def rmsnorm_backward_reference(x, gamma, g, r):
    x_dtype, gamma_dtype = x.dtype, gamma.dtype
    x, gamma, g = x.float(), gamma.float(), g.float()
    N = x.shape[1]
    x_hat = x * r

    dx_hat = g * gamma
    dr = (dx_hat * x).sum(dim=1, keepdim=True)
    dx = r * dx_hat - x * r.pow(3) * dr / N
    dgamma = (g * x_hat).sum(dim=0)
    return dx.to(x_dtype), dgamma.to(gamma_dtype)
```

`dx_hat` 是传给 x_hat 的梯度；`dr` 沿 N 求和，`dgamma` 沿 M 求和。dx 由两条路径的贡献相加：固定 r 时的**直接路径贡献** `r * dx_hat`，以及 x 改变 r 带来的**经 r 路径贡献** `-x * r³ * dr / N`。

## 二、PyTorch eager 性能分析

**eager 将各项张量运算逐个提交给 GPU，中间结果需要在 kernel 之间传递。** 以下展示参考实现的性能，后续 SOL 估算与优化分析统一以 `4096×8192` 为主样本。

### 1. 性能与显存基线

::rmsnorm-benchmark::

小输入的延迟受调用开销与运行波动影响；M 从 1024 增至 4096 时，延迟和显存占用都明显上升。优化需要同时减少中间张量的读写和占用。

### 2. 从表达式到 kernel

这里用 `1024×4096` 的 trace 展示 eager 如何拆分计算：Forward 启动 **9 个 kernel**，Backward 启动 **17 个**。

::rmsnorm-timeline::

::rmsnorm-eager-ops::

**一行表达式仍会拆成多个 kernel。** `dx = r * dx_hat - x * r.pow(3) * dr / N` 在 eager 中对应 6 个 kernel；除 r³ 外，各步结果都是 `[M,N]`。

**完整的 FP32 中间张量反复读写。** 对于这里的 `1024×4096` 输入，x 占 8 MiB，每个完整的 FP32 中间张量占 16 MiB。仅写出、再读入 x_hat，就有 **32 MiB 的逻辑读写**，其中部分访问可能命中缓存。

**中间张量同时存活，推高显存占用。** Forward、Backward 的峰值显存增量分别约 **56 MiB、128 MiB**；保存的 r 则只有 4 KiB。

此外，dr 沿 N 归约，每行独立；dgamma 沿 M 汇总所有 token 的贡献。这两种归约方向决定了后续如何组织 GPU 上的计算。

### 3. 必要读写量与带宽 SOL

**带宽 SOL（Speed of Light）估算的是：只搬运必要数据时，显存带宽允许的理想耗时。** 假设充分融合，每个大张量只读写一次，暂略较小的 gamma、r 和 dgamma：

| 阶段 | 必要读写 | BF16 数据量（字节） |
| :--- | :--- | ---: |
| Forward | 读取 x，写出 y | `2MN + 2MN = 4MN` |
| Backward | 读取 x、g，写出 dx | `2MN + 2MN + 2MN = 6MN` |

每个 BF16 元素占 2 字节。FP32 中间计算若保留在寄存器中，就不需要为这些中间结果增加显存读写。

RTX 4090 的理论显存带宽为 **1008 GB/s**，即每秒传输 `1008 × 10⁹` 字节。[NVIDIA 规格](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)

对 `M=4096, N=8192`，Forward 的必要数据量为 **128 MiB**，Backward 为 **192 MiB**。用数据量除以带宽：

\[
\begin{aligned}
T_{\text{forward}}^{\text{SOL}} &\approx \frac{128\times 2^{20}}{1008\times 10^9}\ \text{s} \approx 133.2\ \mu\text{s},\\
T_{\text{backward}}^{\text{SOL}} &\approx \frac{192\times 2^{20}}{1008\times 10^9}\ \text{s} \approx 199.7\ \mu\text{s}.
\end{aligned}
\]

同一 shape 下，Forward 的 eager 实测为 **1462.4 μs**，SOL 为 **133.2 μs**；Backward 实测为 **4007.8 μs**，SOL 为 **199.7 μs**。SOL 是数据经过显存时的理想参照，实际耗时还受计算、归约和缓存影响。

**SOL 用读写量估算，不能用峰值显存占用代替。**

优化首先需要把逐元素计算与归约融合，减少完整中间张量的写出。`torch.compile` 可以直接从参考实现开始完成这一步。

## 三、torch.compile 自动融合

### 1. 编译参考实现

直接编译第一章的 Forward 和手写 Backward，保持 BF16 输入输出、FP32 中间计算与返回接口不变。

```python
forward_compiled = torch.compile(rmsnorm_forward_reference, fullgraph=True, dynamic=False)
backward_compiled = torch.compile(rmsnorm_backward_reference, fullgraph=True, dynamic=False)

y, r = forward_compiled(x, gamma)
dx, dgamma = backward_compiled(x, gamma, g, r)
```

默认后端 TorchInductor 在这里生成 Triton kernel。`fullgraph=True` 要求完整捕获函数，`dynamic=False` 针对具体 shape 编译；一张计算图仍可生成多个 kernel。[PyTorch 文档](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html)

### 2. 性能与显存变化

::rmsnorm-compile-benchmark::

**大输入收益明显。** 以 `M=4096, N=8192` 为例，Forward 从 **1462.4 μs** 降至 **145.7 μs**，加速 **10.04×**；Backward 从 **4007.8 μs** 降至 **335.4 μs**，加速 **11.95×**。峰值显存分别从约 **448 MiB、896 MiB** 降至约 **64 MiB**，接近返回结果本身的大小。

**小输入仍受调用开销影响。** `128×4096` 与 `1024×4096` 的 Forward 即使只剩一个 kernel，整体耗时也略高于 eager。减少 kernel 数量能降低 GPU 执行成本，但不能保证每种 shape 都加速。五组输入的 y、r、dx、dgamma 均通过 reference 数值检查。

### 3. 编译器融合了什么

在 `4096×8192` 下，Forward 生成 **1 个 kernel**，Backward 生成 **2 个 kernel**：

::rmsnorm-compile-graph::

- **Forward**：平方、沿 N 归约和缩放融合，只写出 y 与保存的 r。
- **Backward**：一个 kernel 沿 M 归约得到 dgamma；另一个融合 dr 的行内归约和 dx 计算。

融合省去了平方结果、x_hat、dx_hat 等完整 FP32 中间张量的显存读写。Backward 的两个分支内部都已融合，但仍分别读取 x、g。

::rmsnorm-compile-source::

能否在读入一行 x、g 时，同时计算 dx 和这一行对 dgamma 的贡献？

## 四、Triton 梯度融合优化

对主样本 `4096×8192`，手写 Triton 将 dx 与 dgamma 的计算放进同一个 kernel，让两个分支共享已经读入的 x、g。Forward 继续使用第三章的 compile 实现。

### 1. 融合 dx 与 dgamma

手写版将 4096 行分成 **128 组，每组 32 行**。一个 program 是由一组 GPU 线程执行的独立任务，这里负责处理一组行：**每读入一行，就计算并写出 dx，同时累加这一行对 dgamma 的贡献**。第二个 kernel 汇总各组的局部结果 partial。

::rmsnorm-triton-graph::

**分组是为了减少写出的局部结果。** 如果每行都为 dgamma 写一份长度为 8192 的 FP32 梯度贡献向量，需要 128 MiB；每 32 行先在组内合并一次，写出的 `partial[128, 8192]` 就只需 **4 MiB**。

program 每次处理一行，跨行保留 gamma 与 dgamma 的累加向量，无需同时保存整组 32 行的输入。**kernel 数仍是 2 个，第二个 kernel 只需读取小得多的 partial。**

### 2. 核心实现与读写量

以下节选第一个 kernel 的核心逻辑。X、W、G、R 是输入指针，DX、P 分别指向 dx 和 partial：

```python
group = tl.program_id(0)
col = tl.arange(0, N)
gamma = tl.load(W + col).to(tl.float32)
dg = tl.full((N,), 0, tl.float32)
for offset in range(32):
    row = group * 32 + offset
    x = tl.load(X + row * N + col).to(tl.float32)
    g = tl.load(G + row * N + col).to(tl.float32)
    r = tl.load(R + row)
    dx_hat = g * gamma
    dr = tl.sum(dx_hat * x, 0)
    dx = r * dx_hat - x * (r * r * r) * dr / N
    tl.store(DX + row * N + col, dx)
    dg += g * (x * r)
tl.store(P + group * N + col, dg)
```

`dg` 是当前组的 dgamma 累加值。它与 dx 共用 x、g、r；gamma 也在组内复用。第二个 kernel 沿 partial 的 128 组求和，写出 BF16 dgamma。

计算 x、g 的读取，以及 dx、partial 的读写：

| 执行结构 | x、g 遍历次数 | 主要逻辑读写量（MiB） |
| :--- | ---: | ---: |
| 两个分支分别读取 | 2 | 320 |
| 融合两个分支 | 1 | 200 |

这里省略 gamma、r 和最终 dgamma 的小额读写，分开执行按每个分支各读一次 x、g 估算。**逻辑读写量减少约 37.5%**；它不是实测显存流量，缓存与 kernel 内的重复读取仍会影响实际访问。

### 3. 性能与显存代价

::rmsnorm-triton-benchmark::

**Backward 调用耗时从 336.2 μs 降至 210.2 μs，加速约 1.60×。** GPU 耗时也从 334.2 μs 降至 208.8 μs。代价是 partial 与 dx 输出同时存活，峰值显存从 **64.02 MiB 增至 68.02 MiB**。

本实现的 BF16 输入输出、FP32 中间计算与 reference 一致；随机输入、零输入及不同幅度输入均通过数值检查。这一结果对应 **`4096×8192`**，其他 shape 的表现仍需单独验证。

**208.8 μs 已接近第二章估算的 Backward SOL：199.7 μs。** SOL / 实测约为 95.6%，这是主要张量读写模型的比值；该模型未计入 partial 读写与缓存影响，不能直接视为实测显存带宽利用率。

从 eager 到 compile，减少了完整中间张量的写出；再到手写融合，让两个梯度分支共享已读入的数据。**先减少中间结果的显存读写，再减少分支间的重复读取，是本文的优化主线。**

::rmsnorm-triton-artifacts::
