# RMSNorm：数学推导与 GPU 算子优化

**RMSNorm 对每个 token 的 hidden 向量计算均方根，用它归一化输入，再按特征乘以可学习权重。**

输入记为 \(X[M,N]\)：\(M=\text{batch}\times\text{sequence}\) 是 token 数，\(N=\text{hidden size}\)。取一个 token 的输入向量 \(x=X[m,:]\)，\(x_i\) 表示第 \(i\) 个特征：

\[
y_i=\frac{x_i}{\operatorname{RMS}(x)}\cdot\gamma_i,
\qquad \text{where}\quad \operatorname{RMS}(x)=\sqrt{\epsilon+\frac{1}{N}\sum_{i=1}^{N}x_i^2}.
\]

- **RMS 沿 N 计算**：每个 token 得到一个标量；保留归约维度时，整批结果的形状为 \([M,1]\)。这里的 RMS 已包含 ε，用于防止除零。
- **γ 沿 M 共享**：形状为 \([N]\)，每个特征 \(i\) 有独立的 \(\gamma_i\)。下表中的 \(\odot\) 表示逐元素相乘。

忽略 ε 且输入非零时，\(x\) 除以自身 RMS 后，RMS 为 1；**乘 γ 后，输出 RMS 不必为 1。**

::rmsnorm-demo::

对比时，BatchNorm 使用二维 batch 输入 \(X[B,N]\)，\(B\) 为 batch size；LayerNorm、RMSNorm 使用上面的 \(X[M,N]\)，\(M\) 始终表示 token 数。

| 算子 | reduction dim | 归一化目标 | 可学习变换（通常） |
|---|---|---|---|
| BatchNorm（训练时） | B (`dim=0`) | \(\mathrm{mean}=0\)，\(\mathrm{std}=1\) | \(\gamma\odot\hat{x}+\beta\) |
| LayerNorm | N (`dim=1`) | \(\mathrm{mean}=0\)，\(\mathrm{std}=1\) | \(\gamma\odot\hat{x}+\beta\) |
| RMSNorm | N (`dim=1`) | \(\mathrm{RMS}=1\)，不减 mean | \(\gamma\odot\hat{x}\) |

表中目标忽略 ε，并假设方差或 RMS 非零；描述的是可学习变换之前的结果。

从计算结构看，forward 对每个 token 沿 N 归约，随后逐元素缩放；backward 还需要沿 M 累加共享权重 γ 的梯度。下面从这两种归约出发，实现 kernel，再通过 benchmark 和 profiler 确定优化方向。

## 一、Forward：计算映射与 Triton baseline

### 1. PyTorch reference

本章处理连续存储的 \(X[M,N]\) 和 \(\gamma[N]\)，两者位于同一 CUDA 设备，且 M、N 均大于零。平方、归约和缩放使用 FP32，最后将输出转回输入 dtype：

```python
def rmsnorm_reference(x, weight, eps=1e-6):
    x32 = x.float()
    mean_square = x32.square().mean(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return ((x32 * inv_rms) * weight.float()).to(x.dtype)
```

先计算 `inv_rms`，再依次乘以输入和权重。下面的 Triton 实现采用相同的精度语义。

### 2. Triton baseline

各个 token 的输出可以独立计算，因此启动 M 个 **program**。本例使用默认的 `num_ctas=1`，每个 program 对应一个 CTA，处理一行：读入 N 个元素，归约得到 `inv_rms`，再广播到整行，与 γ 一起完成缩放。

归约和缩放融合在同一个 kernel 中，中间结果无需显式写入全局内存。按行组织 program 的方式参考 [Triton LayerNorm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)。

```python
@triton.jit
def _rmsnorm_forward(X, W, Y, N: tl.constexpr, BLOCK: tl.constexpr, eps: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offsets = row.to(tl.int64) * N + cols

    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + eps)
    y = (x * inv_rms) * w

    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
```

`BLOCK` 取不小于 N 的最小 2 的幂，超出 N 的位置补零。例如 N=1000 时，BLOCK=1024，其中 24 个位置为零，**计算均值仍然除以 N。**

启动时固定 `num_warps=4`，即每个 CTA 使用 128 个线程。BLOCK 是逻辑元素数量，这些元素在线程间的具体分配由编译器决定。

```python
def rmsnorm_triton(x, weight, eps=1e-6):
    m, n = x.shape
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_forward[(m,)](
            x, weight, y, n, triton.next_power_of_2(n), eps,
            num_warps=4,
        )
    return y
```

固定 warp 数时，BLOCK 增大会增加每个线程分担的元素数量，可能提高寄存器占用；实际占用取决于编译结果。后续结合编译信息和性能测量调整配置。

## 二、Backward：dX 与 dγ 的归约实现

- 推导 dX、dγ，明确沿 N 和沿 M 的归约分别出现在哪里。
- 实现梯度计算，说明 forward 中间结果的保存或重算，以及 dγ 的累加方式与代价。
- 对照 PyTorch autograd 验证两种梯度，明确测试输入、dtype 和误差标准。

## 三、Benchmark：测量方法与基线结果

- 固定 workload、GPU、软件版本和精度语义，说明预热、同步、重复次数与计时范围。
- 分别测量 forward 和 backward；比较 PyTorch、Liger、FlashInfer 各自支持且语义一致的路径，注明版本和具体接口。
- 按 M、N 和 dtype 展示基线延迟，选出需要进一步分析的 workload。

## 四、瓶颈分析与逐项优化

- 简述 nsys / ncu 的采集方式；用 profiler 证据定位所选 workload 的具体瓶颈。
- 每项优化围绕“现象 → 代码改动 → 代价 → 测量结果”展开，紧邻展示证据与关键代码差异。
- 每次改动后复查正确性，用独立 benchmark 报告性能，并记录收益和退化分别出现在哪些 shape 下。

## 五、CuTe DSL：实现与对比

- 选取前文的具体 kernel 设计，说明用 CuTe DSL 表达的线程布局和归约方式，以及需要检验的问题。
- 对照关键代码，在相同 workload、精度语义和测量方法下验证正确性与性能。
- 区分实现设计与工具差异；若未形成独立的分析问题，将相关对比并入上一章。

## 六、结果与适用范围

- 汇总各实现的性能与误差，标明对应 workload 和测量环境。
- 说明各项优化适用的 shape、dtype、额外开销与已知限制，不将局部收益推广到未测场景。
