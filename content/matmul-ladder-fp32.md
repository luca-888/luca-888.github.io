在 H100 上从最朴素的 FP32 矩阵乘法写起，每版只改一处瓶颈：8192 阶从 5.4 提高到 44.3 TFLOPS，为 cuBLAS 的 86%。

## 一、条件与结果

硬件为一张 H100 SXM（功耗上限 700 W，132 个 SM），矩阵为 $M=N=K$ 的方阵，阶数 256 到 8192。两条路线各有一个同布局的 cuBLAS 基线：

| 路线 | 数据类型 | 计算 | 基线 |
| :--- | :--- | :--- | :--- |
| FP32 | FP32 输入、累加与输出 | $C=AB$，三个矩阵都按行存放 | cuBLAS SGEMM（不启用 TF32） |
| Tensor Core | BF16 输入、FP32 累加、BF16 输出 | $C=AW^\top$，$A$ 为 $M\times K$、$W$ 为 $N\times K$，都沿 $K$ 连续 | cuBLAS BF16，同一布局 |

各版本与 cuBLAS 逐元素对比，1024 阶以上逐位相同。

> Tensor Core 路线采用 Linear 层的布局：$A$ 为 activation，$W$ 为权重。

::ml-ladder:fp32::

蓝色为 FP32 路线，灰色为 Tensor Core 路线。

## 二、复杂度与下界

$C$ 的每个元素是 $A$ 一行与 $B$ 一列的点积；也可以写成 $N$ 个外积之和：$A$ 的第 $k$ 列乘 $B$ 的第 $k$ 行得到一个 $N\times N$ 矩阵，$N$ 个相加即为 $C$。第四节的 naive 按点积组织，第六节的寄存器分块按外积组织。

::ml-outer::

$N$ 阶矩阵乘法共 $2N^3$ FLOP，8192 阶为 $1.10\times10^{12}$。FP32 峰值为 $132\times128\times2\times1.98\ \text{GHz}=66.9$ TFLOPS，算完需 16.4 ms；输入输出共 805 MB，按实测 3.0 TB/s 读写一遍只需 0.27 ms。

每读写 1 字节做多少次运算，称为算术强度。强度高于拐点（峰值算力 ÷ 显存带宽，FP32 约 20，BF16 约 295）才可能跑满算力，低于拐点时速度上限为强度 × 显存带宽。每个数只读写一次时强度最高，为 $2N^3/12N^2=N/6$，8192 阶约 1365；$N$ 小于约 120（FP32）或 1800（BF16）时，连这个上限都低于拐点，理论上也跑不满峰值：

| $N$ | 强度上限 $N/6$（FLOP/byte） | 显存喂得上的速度（TFLOPS） | FP32 上限（峰值 66.9） | BF16 上限（峰值 989） |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 11 | 36 | 36 | 36 |
| 256 | 43 | 143 | 66.9 | 143 |
| 1024 | 171 | 572 | 66.9 | 572 |
| 8192 | 1365 | 4574 | 66.9 | 989 |

反过来，每次乘加都从显存读两个数时强度只有 0.25，速度上限 0.75 TFLOPS。各版优化的都是同一件事：让每个读入的数被复用更多次。

> Strassen 等算法把复杂度指数降到约 2.371，它们在数学上精确，但浮点运算下误差更大、常数极大，cuBLAS 仍用经典算法；经典算法的访存下界为 $\Omega(N^3/\sqrt M)$（$M$ 为快速存储能放下的数），分块正好达到。

## 三、线程层级与 SM

一次 kernel 启动的全部线程称为 grid，自上而下切成三层：

- **grid → block。** grid 切成许多 block，每个 block 整体分到一个 SM 上运行。同一 block 的线程共享这个 SM 的共享内存，可以用 `__syncthreads()` 同步；不同 block 之间只能经显存和 L2 交换数据。8192 阶的 naive 用 $16\times16$ 的 block，grid 有 $512\times512$ 个 block，远多于 132 个 SM，由各 SM 轮流处理。
- **block → warp。** SM 不逐个线程执行，而是 32 个线程一组（warp）同时执行同一条指令。block 内线程按 `threadIdx.y * blockDim.x + threadIdx.x` 编号，每 32 个切成一个 warp；$16\times16$ 的 block 有 8 个 warp，每个 warp 是两行线程。
- **warp → 线程。** 线程在 warp 中的序号称为 lane，每个线程有自己的寄存器。

::ml-hierarchy3d::

后面每一版都受两条规则约束：

**warp 一起访存。** 显存以 32 字节的扇区（sector）传输。warp 的 32 次读取同时发出，地址连续时合并为少数几个扇区（合并访存，coalescing），分散时每个线程各占一个扇区，大部分字节被浪费。

**SM 的资源决定驻留多少 warp。** H100 每个 SM 有 228 KiB 共享内存、65536 个 32 位寄存器、最多 2048 个线程，三项中最紧的一项决定能驻留的 warp 数（occupancy）。例如每个线程用 128 个寄存器时，只能驻留 16 个 warp。warp 等数据时调度器切换到别的 warp，驻留越少，越难把等待藏起来。

## 四、naive：每个线程计算一个元素

每个线程读 $A$ 的一行与 $B$ 的一列，算出 $C$ 的一个元素；对照组 `matmul_naive_T` 只交换 `row`、`col` 的取法。

```cpp
// matmul_naive：col 取 threadIdx.x，同一 warp 的 col 连续
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
// matmul_naive_T：row 取 threadIdx.x，同一 warp 的 row 连续
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

float sum = 0;
for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
C[row * N + col] = sum;
```

两者指令完全相同（32 个寄存器、不用共享内存，内层循环 58 条 `LDG.E` 配 29 条 `FFMA`），差别只在 warp 同时读的地址：

| 版本 | warp 内连续的下标 | 同一时刻的读取 |
| :--- | :--- | :--- |
| `matmul_naive` | `col` | `B[k][col]` 为 16 个连续 float，占 2 个扇区；`A[row][k]` 每行同址，广播 |
| `matmul_naive_T` | `row` | `A[row][k]` 相隔 $N$ 个 float，占 16 个扇区，每个只用 4 字节 |

::ml-coalesce3::

::ml-tower:naive::

实测：`matmul_naive` 5.41 TFLOPS，为 cuBLAS SGEMM（51.74 TFLOPS）的 10.5%；`matmul_naive_T` 在 4096 阶为 0.96 TFLOPS，比 `matmul_naive` 的 5.39 慢 5.6 倍。

**瓶颈**：读取次数。每次 FMA 发两条读取；block 内同一行的 16 个线程各读一遍 $A$ 的同一行，同一列的 16 个线程各读一遍 $B$ 的同一列，每个数被重复读 16 次。

> 实测超过 0.75 TFLOPS 的上限，是因为重复读取大多由 L1、L2 承担（推算）。

## 五、共享内存分块

每个 block 计算 $C$ 的一个 $T\times T$ 块，沿 $K$ 每次把 $A$、$B$ 各一块读入共享内存（SM 片上，block 内线程共用）再计算：

```cpp
for (int k0 = 0; k0 < N; k0 += T) {
    As[ty][tx] = A[row * N + k0 + tx];      // 每个线程搬一个元素
    Bs[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();                        // 整块搬完再算
    for (int k = 0; k < T; k++) sum += As[ty][k] * Bs[k][tx];
    __syncthreads();                        // 都算完再覆盖
}
```

::ml-smem3d::

每个读入的数在共享内存中被用 $T$ 次，经过 L2 的读取量降为 naive 的 $1/T$。每一步读入 $8T^2$ 字节、完成 $2T^3$ FLOP，显存强度 $T/4$；$T=32$ 时为 8，低于拐点，但显存允许 $8\times3.0\ \text{TB/s}=24$ TFLOPS（推算），远高于实测。

::ml-tower:smem::

实测：$T=16$ 为 8.22 TFLOPS，$T=32$ 为 9.22 TFLOPS（cuBLAS 的 17.8%）。读取量降为 1/32，速度只到 naive 的 1.7 倍。

共享内存分为 32 个 bank，每个 bank 每周期取出 4 字节，bank 号为 $(\text{字节地址}\div4)\bmod32$。warp 的 32 个地址落在不同 bank 时一个周期读完；两个线程落在同一 bank 的不同地址须分两轮，即 bank 冲突；同一地址则广播。

::ml-bank::

32 个 bank 每周期送出 32 个 float，每次 FMA 要 `As`、`Bs` 各一个，只够 16 次 FMA，而 FP32 单元每周期能做 128 次，上限约为峰值的 1/8，与实测相符：

$$
66.9\times\tfrac18\approx8.4\ \text{TFLOPS}.
$$

::ml-eighth::

> $T=32$ 略高于估算，可能是读 `As` 时同一地址的广播省了 bank 带宽（推算）。

**瓶颈**：共享内存带宽。每次 FMA 仍从共享内存读两个数。

## 六、寄存器分块

`matmul_reg<128,128,8,8,8>` 让 block 计算 $128\times128$、每个线程计算 $8\times8$：沿 $K$ 每步从共享内存读 $A$、$B$ 各 8 个数到寄存器，做外积，即 64 次 FMA。

```cpp
for (int k = 0; k < BK; k++) {
    for (int i = 0; i < TM; i++) rA[i] = As[(tRow * TM + i) * BK + k];
    for (int j = 0; j < TN; j++) rB[j] = Bs[k * BN + tCol * TN + j];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i * TN + j] += rA[i] * rB[j];
}
```

每个读进寄存器的数参与 8 次 FMA。$T_M\times T_N$ 的小块每步读 $T_M+T_N$ 个数、做 $T_MT_N$ 次 FMA，由 $T_M+T_N\ge2\sqrt{T_MT_N}$，面积固定时正方形读得最少（Hong–Kung 下界在寄存器一级的形式）：

| 小块 | 每步读取 | 每读 1 个数的 FMA |
| :--- | ---: | ---: |
| 1 × 64 | 65 | 0.98 |
| 2 × 32 | 34 | 1.9 |
| 4 × 16 | 20 | 3.2 |
| 8 × 8 | 16 | 4 |

每次 FMA 的共享内存读取降为上一版的 1/8，推算上限回到峰值 66.9 TFLOPS；显存强度升到 32 FLOP/byte，越过拐点（推算）。

> 小块受寄存器限制：$8\times8$ 需要 64 个累加值加 16 个操作数，每个线程最多 255 个寄存器。

::ml-regtile3d::

实测：24.72 TFLOPS，为 cuBLAS 的 47.8%、共享内存版的 2.7 倍。

**瓶颈**：读取指令。每步 16 条 `LDS.32` 配 64 条 `FFMA`，五分之一的指令是读取，与 FMA 争用发射；读 `Bs` 时相邻 `tCol` 相隔 8 个 float，16 个 `tCol` 只用到 4 个 bank，形成 4 路冲突（推算）。

## 七、向量化与 warp 分块

读取改用 128 位指令以减少指令数，再在 block 与线程之间加一层 warp 分块以消除 bank 冲突。

**向量化。** 地址连续且 16 字节对齐时，一条 `LDS.128` 读 4 个 float。`rB` 的 8 个数本就连续，`rA` 的 8 个数是 `As` 的一列，因此 `matmul_vec` 把 $A$ 块转置存为 `As[k][m]`：

```cpp
float4 a = *(const float4*)&A[(aRow + r) * N + aCol];
As[(aCol + 0) * BM + aRow + r] = a.x;                  // 转置着存：As[k][m]
// ... a.y、a.z、a.w 同理
// ...
for (int i = 0; i < TM; i += 4) *(float4*)&rA[i] = *(const float4*)&As[k * BM + tRow * TM + i];
for (int j = 0; j < TN; j += 4) *(float4*)&rB[j] = *(const float4*)&Bs[k * BN + tCol * TN + j];
```

**warp 分块。** `matmul_warptile<128,128,8,64,32>` 让每个 warp 负责 $64\times32$，32 个线程排成 $8\times4$，每个线程计算 $2\times2$ 个相隔的 $4\times4$ 小块（行方向相隔 32，列方向相隔 16）：

```cpp
int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
int lRow = lane / 4, lCol = lane % 4;
// ...
for (int mi = 0; mi < MI; mi++) *(float4*)&rA[mi * 4] = *(const float4*)&As[k * BM + wRow * WM + mi * 32 + lRow * 4];
for (int ni = 0; ni < NI; ni++) *(float4*)&rB[ni * 4] = *(const float4*)&Bs[k * BN + wCol * WN + ni * 16 + lCol * 4];
```

::ml-warptile3d::

一条 `LDS.128` 分 4 个阶段，每阶段服务 8 个线程、最多 128 字节。`matmul_vec` 的 warp 覆盖 $C$ 的 $16\times128$，相邻线程的 `rB` 相隔 32 字节，8 个线程跨越 256 字节，形成 2 路冲突（推算）；warp 分块后，同一阶段的 8 个线程读 `Bs` 只涉及 4 个相连的 16 字节（其余由广播送达），读 `As` 只涉及 2 个，没有冲突。

| 版本 | 每步读取指令 | 读取占比 | bank 冲突 | warp 每步读取 |
| :--- | :--- | ---: | :--- | ---: |
| 寄存器分块 | 16 条 `LDS.32` | 1/5 | 4 路 | — |
| 向量化 | 4 条 `LDS.128` | 1/17 | 2 路 | 144 个 float |
| warp 分块 | 4 条 `LDS.128` | 1/17 | 无 | 96 个 float |

实测：向量化 35.28 TFLOPS（cuBLAS 的 68.2%，提高 43%）；warp 分块 37.00 TFLOPS（71.5%，再提高约 5%）。

寄存器分块以后，每个 SM 只驻留 16 个 warp：

| 版本 | 每线程寄存器 | 每 block 线程 | 每 block 共享内存 | 每 SM 驻留 warp |
| :--- | ---: | ---: | ---: | ---: |
| naive | 32 | 256 | 0 | 64 |
| 共享内存（$T=32$） | 32 | 1024 | 8 KB | 64 |
| 寄存器分块 | 104 | 256 | 8 KB | 16 |
| 向量化 | 121 | 256 | 8 KB | 16 |
| warp 分块 | 123 | 256 | 8 KB | 16 |

> 寄存器数取自 ptxas，驻留 warp 数为推算。

**瓶颈**：读取与计算串行。每块都是“读显存，`__syncthreads()`，计算，`__syncthreads()`”，读取期间 FP32 单元空闲；每个 block 的 8 个 warp 在同一处等同一批读取，两个 block 又常处于相近阶段，4 个调度器无 warp 可切换。

## 八、双缓冲

共享内存开成两份：计算第 $t$ 块时，先发出第 $t+1$ 块的显存读取，结果暂存寄存器（`fetch`）；第 $t$ 块算完后写入另一份共享内存（`stash`），每块只同步一次。

```cpp
fetch(0); stash(0); __syncthreads();
for (int t = 0; t < T; t++) {
    int cur = t & 1;
    if (t + 1 < T) fetch((t + 1) * BK);      // 发出下一块的读取，结果到 stash 时才用
    for (int k = 0; k < BK; k++) { /* 与 warptile 相同的外积 */ }
    if (t + 1 < T) stash(cur ^ 1);           // 写入另一份缓冲
    __syncthreads();
}
```

load 发出后不阻塞，用到结果时才等待；`fetch` 与 `stash` 之间隔着 $128\times128\times8=131072$ 次 FMA，独占一个 SM 也要 1024 个时钟（推算），足以掩盖显存延迟。代价是 128 个寄存器、16 KB 共享内存，驻留 warp 仍为 16。

::ml-timeline:fp32::

::ml-tower:dbuf::

实测：44.30 TFLOPS，为 cuBLAS 的 85.6%，比 warp 分块版提高 20%。ncu 对比（8192 阶）：

| 指标 | `matmul_dbuf` | cuBLAS SGEMM |
| :--- | ---: | ---: |
| FMA 管线利用率 | 68.3% | 79.8% |
| 发射槽占用（Issue Slots Busy） | 80.9% | 90.1% |
| 每周期发射指令数（IPC） | 3.23 | 3.60 |
| 每线程寄存器 | 128 | 254 |
| 每个 SM 驻留的 warp | 16 | 8 |
| L2 命中率 | 62.6% | 86.6% |
| 显存吞吐 | 374 GB/s | 232 GB/s |

显存已不是瓶颈：吞吐约为拷贝带宽的 1/8，强度 `matmul_dbuf` 约 120、cuBLAS 约 225 FLOP/byte（推算），都远高于 20。

**瓶颈**：指令发射效率。FMA 管线利用率 68%，cuBLAS 为 80%。

> cuBLAS 用 $256\times128$ 的块、三级缓冲和异步拷贝，每个调度器只放 2 个 warp，靠大量累加器维持发射（ncu 未锁频，SM 约 1.99 GHz）。

## 九、FP32 的上限

| 层级 | 做法 | 解决的瓶颈 | TFLOPS |
| :--- | :--- | :--- | ---: |
| 线程 | warp 内 `col` 连续 | 地址分散 | 5.41 |
| 显存 → 共享内存 | $T\times T$ 分块 | 每个数重复读 16 次 | 9.22 |
| 共享内存 → 寄存器 | 每线程 $8\times8$ 外积 | 共享内存带宽（峰值 1/8） | 24.72 |
| warp | 128 位读取、warp 分块 | 读取指令占 1/5、bank 冲突 | 37.00 |
| 时间 | 双缓冲 | 读取时 FP32 单元空闲 | 44.30 |

FP32 终点取双缓冲版的 44.3 TFLOPS，为峰值 66.9 的 66%（cuBLAS 为 77%）。

> 9 组块形逐尺寸取最快，8192 阶为 44.53 TFLOPS（cuBLAS 的 86.1%，最快为 $128\times128\times16$、warp 块 $64\times64$），只提高 0.5%，作上限参考。

> 剩下的 14% 来自 cuBLAS 更大的块、更多的累加寄存器与异步拷贝，没有继续追。

分块在两层各用一次后，剩下的上限是 FP32 单元本身：每 SM 每时钟 128 次 FMA，而同一 SM 的 4 个 Tensor Core 每时钟共 2048 次 BF16 乘加，是它的 16 倍。

> BF16 dense 规格峰值 989 TFLOPS，拐点随之从约 20 升到约 295 FLOP/byte。

## 参考

1. Simon Boehm. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM), 2022-12.
2. Pranjal Shankhdhar. [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), 2024-11-29.
3. NVIDIA. NVIDIA H100 Tensor Core GPU Architecture（白皮书）。
4. Jia-Wei Hong, H. T. Kung. I/O Complexity: The Red-Blue Pebble Game. STOC 1981.
5. Volker Strassen. Gaussian Elimination is not Optimal. Numerische Mathematik 13, 1969.
6. Josh Alman et al. [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349), 2024.
7. Emilien Dupont et al. [Improving the matrix multiplication exponent with modern optimization and AlphaEvolve](https://arxiv.org/abs/2608.16884), 2026.
8. 实验代码：[`scripts/matmul-ladder/matmul.cu`](https://github.com/luca-888/luca-888.github.io/blob/main/scripts/matmul-ladder/matmul.cu)；原样输出：[`public/measurements/matmul-ladder/`](https://github.com/luca-888/luca-888.github.io/tree/main/public/measurements/matmul-ladder)。
