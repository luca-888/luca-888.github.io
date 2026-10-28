在 H100 上换用 Tensor Core，BF16 矩阵乘法从 93.3 提高到 777.6 TFLOPS，为 cuBLAS 的 99.5%。[上篇](/posts/matmul-ladder-fp32/)的 FP32 路线止于 44.3 TFLOPS，卡在 FP32 单元的峰值。

Tensor Core 每时钟的乘加次数是 FP32 单元的 16 倍，峰值由硬件固定。各版的改动都在数据从显存到 Tensor Core 的通路上：让数据更早、更快地送到 Tensor Core。

## 一、条件与回顾

方阵 $M=N=K$，阶数 256 到 8192；BF16 输入输出、FP32 累加，计算 $C=AW^\top$，$A$（$M\times K$）与 $W$（$N\times K$）都沿 $K$ 连续，即 Linear 层的布局。基线为同布局的 cuBLAS BF16，各版结果与它逐位相同。

| H100 SXM | |
| :--- | :--- |
| SM | 132 个，共用 50 MB 的 L2 |
| 算力 | 每 SM 每时钟 2048 次 BF16 乘加，FP32 单元的 16 倍 |
| 峰值 | 规格 989 TFLOPS（1830 MHz）；持续满载受 700 W 约束，约 1360 MHz、735 TFLOPS |

::ml-tc-rate::

::ml-ladder:tc::

每一版的改动都可以拆成三个问题：谁执行（scope）、数据放在哪一层（layout）、用什么指令搬运和计算（dispatch）。

| 版本 | 谁执行 | 数据放在哪 | 用什么指令 |
| :--- | :--- | :--- | :--- |
| wmma | warp | 经寄存器中转进共享内存 | `wmma` |
| mma.sync | warp | L2 直达共享内存，换位 | `cp.async`、`mma.sync` |
| 三级缓冲 | warp | 共享内存 3 个槽位 | 同上 |
| wgmma | **warpgroup** | 共享内存直接进 Tensor Core | **TMA**、**`wgmma`** |
| warp 特化 | 生产者 + 2 个消费者 warpgroup | 4 个槽位，full / empty | 同上 |
| tile 分组 | 不变 | 共用的 `W` 留在 L2 | 不变 |
| 常驻 + 集群 | **cluster**（2 个 block） | `W` 多播到两个 block | TMA 多播 |
| TMA 写回 | 不变 | 结果经共享内存写出 | **TMA store** |

## 二、wmma：换用 Tensor Core

一个 warp 用 `wmma` 共同计算 $16\times16\times16$ 的乘加，数据仍同步读入共享内存。

::ml-wmma-instr::

`wmma` 是 CUDA C++ 最高层的 Tensor Core 接口：`fragment` 是一个 warp 共同持有的矩阵片段，`load_matrix_sync` 读入片段，`mma_sync` 做乘加。这一版的 block 算 $128\times128$，8 个 warp 各算 $64\times32$，沿 $K$ 每次读入 $A$、$W$ 各 $128\times32$。

::ml-wmma-tile::

每个 warp 持有 4 × 2 个片段，沿 $K$ 每 16 列做 8 条 `mma_sync`：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[4];
wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b[2];   // W[n][k] 即 B(k,n) 的列主序
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4][2];
// ...
for (int kk = 0; kk < BK; kk += 16) {
    for (int i = 0; i < 4; i++) wmma::load_matrix_sync(a[i], &As[(wRow * 64 + i * 16) * BK + kk], BK);
    for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Ws[(wCol * 32 + j * 16) * BK + kk], BK);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
}
```

实测 93.29 TFLOPS，为上篇 FP32 版的 2.1 倍、cuBLAS BF16（781.86）的 11.9%。

::ml-wmma-path::

**瓶颈**：数据两次经过寄存器，每块都要全体同步后才能计算；每行 64 字节，读片段时 8 行同一块只落在 2 组 bank，4 路冲突。

## 三、mma.sync 与三级缓冲

执行单位仍是 warp，改的是数据的放法与指令：先消除 bank 冲突与寄存器中转，再让拷贝与计算重叠。

**换位存放。** 这一版把每块的 $K$ 宽度加到 64，一行 128 字节覆盖全部 32 个 bank。第 $r$ 行的第 $c$ 个 16 字节块存到第 $c\oplus(r\bmod 8)$ 块，即 TMA 的 `SWIZZLE_128B`。`ldmatrix` 一次读 8 行的同一块，换位前 8 个地址都在同一组 bank，换位后分散到 8 组，一轮读完。

::ml-swizzle2::

**cp.async 与 mma.sync。** 搬运改用 `cp.async`，由 L2 直接写入共享内存，不经寄存器。计算改用 PTX 的 `ldmatrix` 把片段读入寄存器，再发 `mma.sync.m16n8k16`，一条 2048 次乘加：

```cpp
__device__ uint32_t swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

auto load = [&](int stage, int k0) {           // 每个线程发出若干条 16 字节的 cp.async
    for (int i = tid; i < BM * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sa + swz(r, c), Ab + (size_t)r * K + k0 + c * 8); }
    for (int i = tid; i < BN * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sw + swz(r, c), Wb + (size_t)r * K + k0 + c * 8); }
};
// compute：每 16 列 K，ldmatrix 读 A、W 片段，再做 MT × NT 次 m16n8k16
for (int i = 0; i < MT; i++) ldmatrix_x4(a[i], sa + swz(wRow * WM + i * 16 + lane % 16, kk * 2 + lane / 16));
for (int i = 0; i < MT; i++)
    for (int j = 0; j < NT; j++) mma16816(acc[i][j], a[i], b[j]);
```

实测 269.41 TFLOPS（34.5%），是 wmma 的 2.9 倍。但每块仍是“发拷贝、等完成、同步、计算”，Tensor Core 等待时空闲。

**三级缓冲。** 把共享内存开成 3 个槽位的环，拷贝提前两块发出：

```cpp
for (int s = 0; s < STAGES - 1; s++) { load(s, s * BK); cp_async_commit(); }
for (int t = 0; t < T; t++) {
    cp_async_wait<STAGES - 2>();   // 第 t 块到齐，最多还有 STAGES-2 组在途
    __syncthreads();               // 也保证第 t-1 块已算完，它的槽位可以覆盖
    int tn = t + STAGES - 1;
    if (tn < T) load(tn % STAGES, tn * BK);
    cp_async_commit();
    compute(t % STAGES);
}
```

计算第 $t$ 块时，后两块已在途。每槽 $(128+128)\times64\times2=32$ KB，共 96 KB。

::ml-timeline:mma::

实测 308.77 TFLOPS（39.5%），提高 15%；换 2、4 级与 $128\times256$ 的块最好为 329。

::ml-tower:mma::

**瓶颈**：`mma.sync` 本身，本文各版止于约 300，峰值的三分之一。只连续发 `mma.sync`、不读显存的微基准跑到 643 TFLOPS，此时 Tensor 管线 99.6% 的周期在忙；同样测法下 `wgmma` 为 962。三级缓冲版又只到这一上限的一半：ncu 显示指令发射槽只忙 43%，停顿以固定延迟依赖（30%）、计算管线忙（18%）和等共享内存返回（14%）为主。

> 按 $128\times128$ 的块，满载时每 SM 每时钟要读共享内存 192 字节，超过 128 字节的供给；但三级缓冲版实测共享内存通路只占 57%，没有到这一上限。

## 四、TMA 与 wgmma

执行单位由 warp 升到 warpgroup，指令换成 TMA、mbarrier 与 `wgmma`，操作数不再经寄存器。

`wgmma` 由 warpgroup 发出。上篇用到的 thread、warp、block、grid 四层之间，Hopper 新增了两层：warpgroup 由 4 个 warp 组成，cluster 由同一 GPC 的几个 block 组成。

::ml-hier:wg::

| 机制 | 作用 |
| :--- | :--- |
| TMA | 一个线程一条指令，把全局内存的矩形块搬入共享内存并换位；形状写在 host 端的 tensor map 中 |
| mbarrier | 共享内存中的计数器，记录 TMA 到达的字节数，等待方查询某一相位是否完成 |
| `wgmma` | 4 个 warp（一个 warpgroup）共同发出、异步执行；`B` 从共享内存读，`A` 可来自共享内存或寄存器（本文都用共享内存），累加器在寄存器 |

一条 `wgmma.m64n128k16` 有 131072 次乘加，是 `mma.sync` 的 64 倍；它异步执行、直接读共享内存，省去 `ldmatrix`。

::ml-wgmma-size::

这一版用一个 warpgroup 计算 $128\times128$，4 级缓冲，线程 0 发 TMA；每 16 列 $K$ 做两条 `wgmma.m64n128k16`：

```cpp
// issue(t)：线程 0 用两条 TMA 搬 A、W 各一块，共 32 KB
for (int t = 0; t < T; t++) {
    int s = t % STAGES;
    mbar_wait(&full[s], (t / STAGES) & 1);   // 等第 t 块到齐
    wgmma_fence();
    for (int kk = 0; kk < BK / 16; kk++) {
        wgmma_n128(d[0], make_desc(sa + kk * 16), make_desc(sw + kk * 16));
        wgmma_n128(d[1], make_desc(sa + 64 * BK + kk * 16), make_desc(sw + kk * 16));
    }
    wgmma_commit();
    wgmma_wait<0>();
    __syncthreads();                         // 都读完这一槽，TMA 才能覆盖
    if (tid == 0 && t + STAGES < T) issue(t + STAGES);
}
```

实测 484.21 TFLOPS（61.9%），是三级缓冲的 1.6 倍。

::ml-tower:wgmma::

**瓶颈**：同一个 warpgroup 既发 TMA 又计算，每块要 `wgmma_wait<0>()` 与 `__syncthreads()` 后才发下一次 TMA，两块之间 Tensor Core 空闲。

## 五、warp 特化

warpgroup 分成生产者与消费者，拷贝不再等计算。

| 角色 | warpgroup | 寄存器 | 做什么 |
| :--- | :--- | ---: | :--- |
| 生产者 | 0 号 | 40 | 一个线程发 TMA；先等 empty |
| 消费者 | 1、2 号 | 232 | 各用 `wgmma.m64n256k16` 算 $64\times256$；先等 full，用完在 empty 报到 |

每个槽位两个 mbarrier：full 表示数据已到，empty 表示 8 个消费者 warp 都已用完；`setmaxnreg` 在运行中调整寄存器数。块放大到 $128\times256$，每字节 85 FLOP（上一版 64）。

::ml-ws2::

```cpp
if (wg == 0) {                                     // 生产者
    setmaxnreg_dec<40>();
    if (tid == 0)
        for (int t = 0; t < T; t++, it++) {
            int s = it % STAGES;
            mbar_wait(&empty[s], ((it / STAGES) & 1) ^ 1);   // 等槽位空出
            mbar_expect_tx(&full[s], STAGE_BYTES);
            tma_load(sa, &tmA, &full[s], t * BK, bm);
            tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);
        }
} else {                                           // 两个消费者
    setmaxnreg_inc<232>();
    for (int t = 0; t < T; t++, it++) {
        int s = it % STAGES;
        mbar_wait(&full[s], (it / STAGES) & 1);
        wgmma_fence();
        for (int kk = 0; kk < BK / 16; kk++) wgmma_n256(d, make_desc(sa + kk * 16), make_desc(sw + kk * 16));
        wgmma_commit();
        wgmma_wait<0>();
        release(s);                                // 在 empty[s] 上报到
    }
}
```

> `wgmma.m64n256k16` 的累加器每线程 128 个 float；启动时 384 线程 × 168 寄存器 = 64512，接近上限 65536，重新分配后总数不变。

每步计算约 1024 个时钟，4 个槽位（共 192 KB，每 SM 一个 block）可提前约 3000 个时钟发拷贝，覆盖 L2（约 290）与显存（约 700）的空载延迟。两个消费者交替使用 Tensor Core，一个等待时另一个在计算。

::ml-timeline:ws::

实测 4096 阶由 481.44 提高到 634.69 TFLOPS；8192 阶仍为 479.74（61.4%）。

::ml-ws-l2::

**瓶颈**：8192 阶的 L2 容量。tile 指一个 block 算的 $128\times256$ 输出块，8192 阶共 2048 个。按行发放时，同时运行的 132 个 tile 覆盖约 4 行、全部 32 列，需要整个 `W`；`W` 在下一个用它的 tile 读取之前就被替换，反复从显存读入。

## 六、tile 的发放顺序与 L2

执行单位与指令都不变，只改 tile 的发放顺序：每 16 行一组，组内按列发放（`GROUP = 16`）。

```cpp
auto tile_mn = [&](int id, int& bm, int& bn) {
    int m, n;
    if constexpr (GROUP == 0) { m = id / numN; n = id % numN; }       // 按行发放
    else {                                                             // GROUP 行一组，组内按列
        int per = G * numN, first = id / per * G, gs = min(G, numM - first), r = id % per;
        m = first + r % gs; n = r / gs;
    }
    bm = (m * CL + rank) * BM; bn = n * BN;
};
```

一轮 132 个 tile 需要的数据由 136 MB 减为 65 MB（不同的行各算一次，一行 16 KB）：

| 发放顺序 | 覆盖的 tile | `A` | `W` | 合计 |
| :--- | :--- | ---: | ---: | ---: |
| 按行 | 约 4 行 × 32 列 | 8 MB | 128 MB | 136 MB |
| 16 行一组 | 16 行 × 约 8 列 | 32 MB | 33 MB | 65 MB |

::ml-tile-order2::

65 MB 仍大于 L2，起作用的是共用 `W` 的 tile 是否同时运行。按行发放时，共用一列 `W` 的两个 tile 编号相差 32，`W` 在后一个 tile 读取之前就被替换，大部分 tile 各自从显存读；分组后它们编号相邻，一块 `W` 读进 L2 随即被 16 个 tile 读取。ncu 在 8192 阶对比两种顺序：

| 指标 | 按行发放 | 16 行一组 |
| :--- | ---: | ---: |
| 共用同一列 `W` 的 tile 开始时间差 | 25 到 31 µs | 约 0.1 µs |
| 显存读取量 | 6.57 GB | 1.31 GB |
| L2 命中率 | 47.5% | 78.6% |
| Tensor 管线利用率 | 48.8% | 87.8% |

> 4096 阶按行发放时显存只读 171 MB，所以 warp 特化版在 4096 阶不慢。

实测 740.05 TFLOPS（94.7%），是上一版的 1.54 倍；4096 阶为 669.21。

**瓶颈**：每个 block 各自从 L2 读整块 `W`。L2 读出约 $2048\times128\times48\ \text{KB}\div1.486\ \text{ms}\approx8.7$ TB/s，三分之二是 `W`，与 ncu 测得的 L2 吞吐 8.67 TB/s 一致。

## 七、常驻 kernel、线程块集群与 TMA 多播

执行单位再升一层到 cluster：上下相邻、共用 `W` 的两个 tile 组成集群，各搬 `W` 块的一半，多播给双方；集群以常驻 kernel 为前提。

**常驻 kernel。** 常驻版只启动 132 个 block，循环领取 tile；生产者跨 tile 连续计数槽位，消费者写回时已在搬下一块：

```cpp
for (int id = blockIdx.x / CL; id < tiles; id += gridDim.x / CL) {   // grid = 132 时循环约 15.5 次
    tile_mn(id, bm, bn);
    // ...
}
```

单独加上常驻，速度由 740 降到 657 TFLOPS，慢 11%，原因未查明。加上集群后常驻版反而更快（开发阶段同一轮计时为 722 对 686），所以保留。

> 让消费者只等上一块的 `wgmma` 就归还槽位，速度为 740.51，不变：两个消费者交替使用 Tensor Core，空档已被对方填补。

**集群与多播。** 集群用的是 Hopper 新增的另一层 cluster：同一集群的 block 可以访问彼此的共享内存。

::ml-hier:cluster::

| 机制 | 作用 |
| :--- | :--- |
| 线程块集群 | block 同时调度到同一 GPC 内的 SM，可访问彼此的共享内存与 mbarrier |
| TMA 多播 | 从 L2 读一次，写入集群内多个 block |
| `mapa` | 把共享内存地址映射到集群内另一个 block |

```cpp
tma_load(sa, &tmA, &full[s], t * BK, bm);                        // A 各搬各的
tma_load_mc(sa + A_BYTES + rank * (BN / CL) * BK * 2, &tmW, &full[s],
            t * BK, bn + rank * (BN / CL), (1 << CL) - 1);       // W 搬一半，掩码 0b11 发往两个 block
```

集群版用 66 个常驻集群循环领取 tile 对，每 block 每步从 L2 读 32 KB（原 48 KB），总流量约 5.9 TB/s。代价是槽位要等两个 block 都用完：empty 计数由 8 改为 16，消费者用 `mapa` 在对方 mbarrier 上也报到。

::ml-cluster3::

常驻、集群与多播合计，实测 757.26 TFLOPS（96.9%），比分组版高 2.3%。

ncu 对比 cuBLAS BF16（8192 阶），资源配置几乎相同：

| 指标 | 集群版 | cuBLAS BF16 |
| :--- | ---: | ---: |
| Tensor 管线利用率 | 92.9% | 97.0% |
| L2 吞吐 | 66.1% | 57.6% |
| 指令发射槽占用 | 19.8% | 16.1% |
| 每线程寄存器 | 168 | 168 |
| 每 block 动态共享内存 | 197.6 KB | 188.6 KB |
| 集群 / grid | 2 / 132 | 2 / 132 |

> cuBLAS 同样是常驻 kernel、2 个 block 的集群；kernel 名中的 `320x128` 为 tile 形状，`64x3` 为 $K$ 宽度 64、三级缓冲。

**瓶颈**：写回。tile 结束时每个消费者线程零散写显存 64 次、每次 4 字节，期间不发 `wgmma`。

## 八、TMA 写回

改由 TMA 写回：消费者把结果换位写入共享内存，一个线程发 4 条 TMA，消费者随即开始下一块。

`fence_proxy_async` 让普通写入对 TMA 可见；`bulk_wait_read0` 等上次 TMA 写回读完共享内存，才能覆盖结果区。

```cpp
if (tid % 128 == 0) bulk_wait_read0();            // 上一块的 TMA 写回已读完共享内存
named_bar(1 + c, 128);
for (int j = 0; j < 32; j++) {                     // 按 128B 规则换位写入
    uint8_t* p = cs + j / 8 * 8192 + (((j % 8) ^ (r & 7)) << 4) + lane % 4 * 4;
    *(__nv_bfloat162*)(p + r * 128) = __floats2bfloat162_rn(d[4 * j], d[4 * j + 1]);
    *(__nv_bfloat162*)(p + (r + 8) * 128) = __floats2bfloat162_rn(d[4 * j + 2], d[4 * j + 3]);
}
fence_proxy_async();                               // 普通写入对 TMA 可见
named_bar(1 + c, 128);
if (tid % 128 == 0) {
    for (int q = 0; q < 4; q++) tma_store(&tmC, cs + q * 8192, bn + q * 64, bm + c * 64);
    bulk_commit();
}
```

::ml-tower:tstore::

代价是共享内存：结果区 64 KB 加 4 级缓冲超过每 block 227 KB 的上限，减为 3 级，共 208 KB；3 级仍可提前约 2000 个时钟发拷贝。

::ml-timeline:tstore::

实测 777.63 TFLOPS，为 cuBLAS BF16 的 99.5%。

**瓶颈**：持续满载时的 700 W 功耗上限。

## 九、功耗墙、能效与各规模

持续满载时，手写版与 cuBLAS 都达到 700 W 的功耗上限，速度相同；剩下的差距在能效。

**功耗墙。** 短时计时时钟较高。连续运行 2 秒、NVML 采样，8192 阶两者功耗都达到 700 W，时钟由 1830 降到约 1360 MHz；按这一时钟，峰值为 $132\times4096\times1.36\ \text{GHz}=735$ TFLOPS，两者都用到其中的 93%。

::ml-power::

| 阶数 | kernel | TFLOPS | SM 时钟 | 功耗 |
| ---: | :--- | ---: | ---: | ---: |
| 4096 | 最终版 | 650.80 | 1402 MHz | 698 W |
| 4096 | cuBLAS BF16 | 674.88 | 1453 MHz | 589 W |
| 8192 | 最终版 | 682.34 | 1360 MHz | 699 W |
| 8192 | cuBLAS BF16 | 681.14 | 1368 MHz | 697 W |

**补充实测。** 以下在最终版上补测，只比同次运行内与 cuBLAS 的比值。

改用 Hilbert 曲线调度，以集群为单位发放 tile，2048、4096、8192 阶的变化都在 0.2% 以内：同时运行的 66 个集群在曲线上同样构成 16 行 × 8 列 tile 的区域，与 16 行一组相同。

首块省清零是剥出每个 tile 的第 0 步，第一条 `wgmma` 的 scale-d 取立即数 0（$D=AB$），省去累加器清零。三个阶数提高 0.4% 到 1.4%，不带集群时 8192 阶提高 3%。

```cpp
if constexpr (MICRO) {                                  // 第 0 步单独剥出，不再清零
    mbar_wait(&full[s], cph);
    wgmma_fence();
    wgmma_n256_zero(d, make_desc(sa), make_desc(sw));   // scale-d = 0，累加器只作输出（"=f"）
    for (int kk = 1; kk < BK / 16; kk++) wgmma_n256(d, make_desc(sa + kk * 16), make_desc(sw + kk * 16));
    wgmma_commit();
}
```

> 第一版用运行时变量作 scale-d，`wgmma` 被串行化，反而慢了约 20%。

去掉集群后，2048、4096 阶短时计时快 2% 到 5%，小规模下两个 block 互相等待的代价超过了收益；8192 阶持续运行仍是带集群更快。

**能效。** 4096 阶去掉集群、加上首块省清零后，短时计时为 766 TFLOPS，超过 cuBLAS 的 753；连续运行 2 秒时功耗为 698 W、时钟为 1375 MHz，为 667 TFLOPS，落后同次运行的 cuBLAS（682 TFLOPS、552 W）2.2%。每次浮点运算手写版约 1.05 pJ，cuBLAS 约 0.81 pJ，多 29%。

> cuBLAS 的 L2 吞吐与发射槽占用都更低，可能是它更省电的线索。

**各规模。** 2048 阶起手写版与 cuBLAS 相当，2048 阶 128 个 tile 一轮完成，快 7.7%。1024 阶只有 32 个 tile，100 个 SM 空闲，cuBLAS 可换更小的块或 split-K；$128\times128$ 块的 `wgmma` 版 tile 数是最终版的 2 倍，512、1024 阶为 20.07、119.84 TFLOPS，最终版为 16.64、106.69。256 阶两者都约 0.013 ms，主要是启动开销。

::ml-sizes::

## 十、未完成的工作

- 常驻 kernel 变慢的原因，未用 profiler 对比。
- 集群的取舍：4096 阶不带更快、8192 阶带更快，原因未区分。
- 小矩阵：更小的块、split-K 或 stream-K，按规模选 kernel。
- 能耗：4096 阶多耗约 29%，未逐项分析。

## 十一、总结

两篇各版去除的瓶颈，由显存逐级移到写回。三个问题的答案也各自推进：执行单位由 warp 升到 warpgroup 再到 cluster，数据由经寄存器中转变为 TMA 整块搬运，指令由 `wmma` 换到 `wgmma`。

::ml-summary::

计算单元的峰值是固定的，矩阵乘法的速度取决于数据能否及时送达计算单元。

**Blackwell。** 第四节起的各版依赖 `wgmma`，它只在 sm_90a 上可用，不能直接用于 B200 等 Blackwell 数据中心 GPU。Blackwell 的第五代 Tensor Core 改用 `tcgen05.mma`：一个线程即可发出，异步执行，完成后经 `tcgen05.commit` 在 mbarrier 上报到。累加结果 D 不再放在寄存器，而放在新增的 Tensor Memory 中；它每个 CTA 128 行 × 512 列、每格 32 位，由一个 warp 显式分配。B 来自 shared memory，A 来自 shared memory 或 Tensor Memory。同一 cluster 中配对的两个 CTA 还可以合作执行一条 MMA（PTX ISA 9.7.18）。

## 参考

1. Simon Boehm. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM), 2022-12.
2. Pranjal Shankhdhar. [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), 2024-11-29.
3. NVIDIA. [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：`cp.async`、`ldmatrix`、`mma.sync`、`wgmma`、`cp.async.bulk.tensor`、`mbarrier`、`setmaxnreg`；Blackwell 的 `tcgen05.mma` 与 Tensor Memory（9.7.18 节）。
4. NVIDIA. [CUTLASS：Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：Hopper 的 warp 特化、persistent cooperative 与 ping-pong 调度、rasterization、split-K。
5. NVIDIA. NVIDIA H100 Tensor Core GPU Architecture（白皮书）。
6. Jay Shah et al. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608), 2024：warp 特化与 ping-pong 调度。
7. DeepSeek. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。
8. MLC. [Modern GPU Programming for MLSys：Building a Tiled GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_basics/index.html)：scope、layout、dispatch 三问；Blackwell 上同一条优化路线。
9. 实验代码：[`scripts/matmul-ladder/`](https://github.com/luca-888/luca-888.github.io/tree/main/scripts/matmul-ladder)；原样输出：[`public/measurements/matmul-ladder/`](https://github.com/luca-888/luca-888.github.io/tree/main/public/measurements/matmul-ladder)（阶梯 `20261021-121144/`，补充实测 `20261022-105500/`，tile 发放顺序的 ncu、逐指令采样与 tile 开始时间 `20261022-142921/`）。
