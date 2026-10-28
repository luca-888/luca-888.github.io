在 H100 上用 Tensor Core 计算 BF16 矩阵乘法，从最直接的 `wmma` 写法出发，经过七步优化，8192 阶由 95.7 提高到 791.0 TFLOPS，为同布局 cuBLAS（799.1）的 99.0%。[上篇](/posts/matmul-ladder-fp32/)的 FP32 路线止于 44.3 TFLOPS，卡在 FP32 单元的峰值。

Tensor Core 的峰值由硬件固定，它只能计算已经送达的数据。数据从显存出发，经 L2、shared memory、寄存器，最后交给 Tensor Core；最直接的写法中，Tensor Core 只有 16% 的时间在计算。七步各消除这条通路上的一处等待：前四步在 SM 内部，后三步在 SM 之外。

## 一、条件与存储层级

方阵 $M=N=K$，阶数 256 到 8192；BF16 输入输出、FP32 累加，计算 $C=AW^\top$，$A$（$M\times K$）与 $W$（$N\times K$）都沿 $K$ 连续，即 Linear 层的布局。基线为同布局的 cuBLAS BF16，各版结果与它逐位相同。

| H100 SXM | |
| :--- | :--- |
| SM | 132 个，共用 50 MB 的 L2 |
| 算力 | 每 SM 每时钟 2048 次 BF16 乘加，FP32 单元的 16 倍 |
| 峰值 | 规格 989 TFLOPS（1830 MHz）；持续满载受 700 W 约束，约 1375 MHz、743 TFLOPS |

::ml-tc-rate::

一个 SM 有 4 个 Tensor Core，一条指令完成两个小矩阵的乘加。操作数要逐层送达：显存中的 $A$、$W$ 先进入 L2，再进入各 SM 的 shared memory，然后读入寄存器交给 Tensor Core；累加结果在寄存器中，最后写回显存。本文的七步分别作用于这条通路上的一层。

::ml-map::

| 步骤 | 作用的层级 | 消除的等待 | 新指令或硬件 |
| :--- | :--- | :--- | :--- |
| 1 swizzle | shared memory → 寄存器 | bank conflict | `ldmatrix`、`mma.sync` |
| 2 cp.async 与 pipeline | L2 → shared memory | 搬运经寄存器、搬完才算 | `cp.async` |
| 3 wgmma | shared memory → Tensor Core | 指令太小 | **`wgmma`**、TMA、mbarrier |
| 4 warp specialization | 整条通路（时间上） | 搬与算互相等待 | `setmaxnreg` |
| 5 tile 分组 | 显存 → L2 | `W` 被挤出 L2 后重读 | 无 |
| 6 cluster 多播 | L2 → shared memory（跨 SM） | 相邻 SM 重复读 L2 | **cluster**、TMA multicast |
| 7 TMA store | 寄存器 → 显存 | 写回期间不计算 | TMA store |

::ml-ladder:tc::

## 二、起点：wmma

`wmma` 是 CUDA C++ 最高层的 Tensor Core 接口：fragment 即一个 warp 共同持有的矩阵片段，`load_matrix_sync` 读入 fragment，`mma_sync` 做一次 $16\times16\times16$ 的乘加。

::ml-wmma-instr::

这一版的 block 算 $128\times128$，8 个 warp 各算 $64\times32$，沿 $K$ 每次读入 $A$、$W$ 各 $128\times32$。线程先把这一块从显存读入寄存器，再写入 shared memory，全体同步后才计算。

::ml-wmma-tile::

每个 warp 持有 4 × 2 个 fragment，沿 $K$ 每 16 列做 8 条 `mma_sync`：

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

实测 95.7 TFLOPS，为 cuBLAS 的 12.0%。ncu 显示 Tensor 管线只有 15.9% 的周期在忙。warp 的停顿以等访存返回（39%）与访存排队（21%）为主：前者约三分之二在等搬运的显存读取，约三分之一在等读 fragment；后者主要是读 fragment 的 shared memory 读取。数据两次经过寄存器，每块都要同步后才能计算，读 fragment 时还有 bank conflict。

## 三、swizzle 与 bank conflict

作用层：shared memory → 寄存器。

shared memory 分成 32 个 bank，地址每 4 字节换下一个 bank；同一周期内，每个 bank 只能送出一份数据。一个 warp 的 32 个线程同时读取时，地址落在不同 bank 可一轮读完；多个地址落在同一个 bank 时只能排队，每轮读一份，这种情况称为 bank conflict。

`wmma` 版的每行为 32 个 BF16，即 64 字节，一排 32 个 bank 放两行。`load_matrix_sync` 被编译成普通的 shared memory 读取，每个 fragment 4 条、每条 128 字节；8 行的同一位置只落在两组 bank 上，ncu 测得每次读取平均 4.00 轮。

::ml-wmma-path::

下一版把每块的 $K$ 宽度加到 64，一行 128 字节恰好覆盖 32 个 bank 一圈。读操作数改用 PTX 的 `ldmatrix`，一次读 8 行的同一个 16 字节块；不做处理时，各行的同一位置都落在同一组 bank 上，8 行要分 8 轮读完（ncu 实测 8.00 轮）。

swizzle 原意为搅乱：写入时，第 $r$ 行的第 $c$ 个 16 字节块存到第 $c\oplus(r\bmod 8)$ 块，即 TMA 的 `SWIZZLE_128B`。纵向读取的 8 行由此落在 8 组不同的 bank 上，一轮读完（实测 1.00 轮）；数据量不变，只改变位置。

::ml-swizzle2::

读取方按同一规则计算地址，每个线程给 `ldmatrix` 一个行地址；计算改用 PTX 的 `mma.sync.m16n8k16`，一条 2048 次乘加：

```cpp
__device__ uint32_t swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

for (int i = 0; i < MT; i++) ldmatrix_x4(a[i], sa + swz(wRow * WM + i * 16 + lane % 16, kk * 2 + lane / 16));
for (int i = 0; i < MT; i++)
    for (int j = 0; j < NT; j++) mma16816(acc[i][j], a[i], b[j]);
```

同一次运行中，不做 swizzle 的版本为 71.0 TFLOPS，做 swizzle 后为 291.3，是前者的 4.1 倍。另一种办法是每行末尾多留 16 字节（padding），错开各行的 bank；`wmma` 版这样改后冲突降为 1.01 轮、速度由 100.9 提高到 148.2，代价是多占 shared memory。

## 四、cp.async 与 pipeline

作用层：L2 → shared memory。

`wmma` 版搬一块数据要两步：线程先把数据读入自己的寄存器，再写入 shared memory，数据到达前线程只能等待。`cp.async` 是异步拷贝指令，数据从 L2 直接写入 shared memory，不经寄存器；指令发出后线程无须等待数据到达：

```cpp
auto load = [&](int stage, int k0) {           // 每个线程发出若干条 16 字节的 cp.async
    for (int i = tid; i < BM * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sa + swz(r, c), Ab + (size_t)r * K + k0 + c * 8); }
    for (int i = tid; i < BN * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sw + swz(r, c), Wb + (size_t)r * K + k0 + c * 8); }
};
```

同一次运行中，搬运经寄存器的版本为 222.3 TFLOPS，改用 `cp.async` 后为 291.3。这一版的阶梯成绩为 277.0 TFLOPS（34.7%），是 `wmma` 的 2.9 倍。

线程不再等待数据，但每块仍是“发拷贝、等完成、同步、计算”，下一块要等这一块算完才开始传输。pipeline 在 shared memory 中开 3 个 stage 轮流使用，拷贝提前两块发出：

```cpp
for (int s = 0; s < STAGES - 1; s++) { load(s, s * BK); cp_async_commit(); }
for (int t = 0; t < T; t++) {
    cp_async_wait<STAGES - 2>();   // 第 t 块到齐，最多还有 STAGES-2 组在途
    __syncthreads();               // 也保证第 t-1 块已算完，它的 stage 可以覆盖
    int tn = t + STAGES - 1;
    if (tn < T) load(tn % STAGES, tn * BK);
    cp_async_commit();
    compute(t % STAGES);
}
```

计算第 $t$ 块时，后两块已在传输中。每个 stage $(128+128)\times64\times2=32$ KB，共 96 KB。

::ml-timeline:mma::

实测 320.0 TFLOPS（40.0%），比单个 stage 提高 15%。

::ml-tower:mma::

**瓶颈**：`mma.sync` 指令本身。只连续发 `mma.sync`、不读任何数据的微基准为 643 TFLOPS，此时 Tensor 管线 99.6% 的周期在忙；同样测法下 `wgmma` 为 962，`mma.sync` 只有它的三分之二。pipeline 版只达到这一上限的一半：ncu 显示指令发射槽忙 43%，shared memory 数据通路占峰值的 57%，都未占满；停顿以固定延迟依赖（30%）、计算管线忙（18%）和等 shared memory 返回（14%）为主。

## 五、wgmma、TMA 与 mbarrier

作用层：shared memory → Tensor Core。

`wgmma` 是 Hopper 新增的 Tensor Core 指令，由 warpgroup 发出。在 thread、warp、block、grid 四层之间，Hopper 新增了两层：warpgroup 由 4 个 warp 组成，cluster 由同一 GPC 的几个 block 组成。

::ml-hier:wg::

一条 `wgmma.m64n128k16` 有 131072 次乘加，是 `mma.sync` 的 64 倍。它异步执行，操作数 `B` 直接从 shared memory 读，`A` 可来自 shared memory 或寄存器（本文都用 shared memory），省去 `ldmatrix`；累加结果仍在寄存器。

::ml-wgmma-size::

与之配套的有两项新硬件：

| 机制 | 作用 |
| :--- | :--- |
| TMA | Tensor Memory Accelerator，专门搬运矩阵块的硬件：一个线程一条指令，把全局内存的矩形块搬入 shared memory，并在搬运时完成 swizzle；形状写在 host 端的 tensor map 中 |
| mbarrier | shared memory 中的计数器，记录 TMA 到达的字节数；数据到齐后，等待方查询到这一相位已完成 |

这一版用一个 warpgroup 计算 $128\times128$，4 个 stage，线程 0 发 TMA；每 16 列 $K$ 做两条 `wgmma.m64n128k16`：

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
    __syncthreads();                         // 都读完这一 stage，TMA 才能覆盖
    if (tid == 0 && t + STAGES < T) issue(t + STAGES);
}
```

搬运由 TMA 完成，读取由 `wgmma` 完成，线程只负责发出指令：ncu 中指令发射槽的占用由 pipeline 版的 43% 降到 11%。实测 500.8 TFLOPS（62.7%），是 pipeline 版的 1.56 倍。

::ml-tower:wgmma::

**瓶颈**：同一个 warpgroup 既发 TMA 又计算。每块要 `wgmma_wait<0>()` 与 `__syncthreads()` 后才发下一次 TMA，两块之间 Tensor Core 空闲；ncu 中 57% 的停顿是同步等待，Tensor 管线忙 55.6%。

## 六、warp specialization

作用层：整条通路，消除的是时间上的等待。

warp specialization 按角色分配 warpgroup：一组只发 TMA，称为 producer；另外两组只计算，称为 consumer。

| 角色 | warpgroup | 寄存器 | 做什么 |
| :--- | :--- | ---: | :--- |
| producer | 0 号 | 40 | 一个线程发 TMA；先等 empty |
| consumer | 1、2 号 | 232 | 各用 `wgmma.m64n256k16` 算 $64\times256$；先等 full，用完在 empty 报到 |

每个 stage 有两个 mbarrier：full 表示数据已到，empty 表示 8 个 consumer warp 都已用完；`setmaxnreg` 在运行中调整寄存器数。块放大到 $128\times256$，每字节 85 FLOP（上一版 64）。

::ml-ws2::

```cpp
if (wg == 0) {                                     // producer
    setmaxnreg_dec<40>();
    if (tid == 0)
        for (int t = 0; t < T; t++, it++) {
            int s = it % STAGES;
            mbar_wait(&empty[s], ((it / STAGES) & 1) ^ 1);   // 等 stage 空出
            mbar_expect_tx(&full[s], STAGE_BYTES);
            tma_load(sa, &tmA, &full[s], t * BK, bm);
            tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);
        }
} else {                                           // 两个 consumer
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

> `wgmma.m64n256k16` 的累加结果每线程 128 个 float；启动时 384 线程 × 168 寄存器 = 64512，接近上限 65536，重新分配后总数不变。

每步计算约 1024 个时钟，4 个 stage（共 192 KB，每 SM 一个 block）可提前约 3000 个时钟发拷贝，覆盖 L2（约 290）与显存（约 700）的空载延迟。两个 consumer 轮流使用 Tensor Core，一个等待时另一个在计算。

::ml-timeline:ws::

4096 阶由 484.4 提高到 632.8 TFLOPS，Tensor 管线忙 75.5%；8192 阶却为 472.0（59.1%），与上一版持平，Tensor 管线只忙 50.9%。

::ml-ws-l2::

**瓶颈**：8192 阶的显存读取。tile 指一个 block 算的 $128\times256$ 输出块，8192 阶共 2048 个。按行发放时，同时运行的 132 个 tile 覆盖约 4 行、全部 32 列，需要整个 `W`。ncu 显示这一版在 8192 阶从显存读了 6.71 GB，L2 命中率 48%；4096 阶只读 0.17 GB。

## 七、tile 分组与 L2 复用

作用层：显存 → L2。

同一列的 tile 都需要同一块 `W`。按行发放时，共用一列 `W` 的两个 tile 编号相差 32，开始时间相隔约一轮；后一个 tile 使用时，这块 `W` 已被其他数据挤出 L2，只能再从显存读一次。执行单位与指令都不变，只改 tile 的发放顺序：每 16 行一组，组内按列发放（`GROUP = 16`）。

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

65 MB 仍大于 L2，起作用的是共用 `W` 的 tile 是否同时运行。分组后共用一列 `W` 的 16 个 tile 编号相邻，几乎同时开始，一块 `W` 读入 L2 后随即被这 16 个 tile 读取。ncu 在 8192 阶对比两种顺序：

| 指标 | 按行发放 | 16 行一组 |
| :--- | ---: | ---: |
| 共用同一列 `W` 的 tile 开始时间差 | 25 到 31 µs | 约 0.1 µs |
| 显存读取量 | 6.71 GB | 1.37 GB |
| L2 命中率 | 48.0% | 78.5% |
| Tensor 管线利用率 | 50.9% | 90.1% |

实测 754.6 TFLOPS（94.4%），是上一版的 1.60 倍；4096 阶为 677.3。

**瓶颈**：每个 block 各自从 L2 读整块 `W`。L2 读出约 $2048\times128\times48\ \text{KB}\div1.457\ \text{ms}\approx8.8$ TB/s，三分之二是 `W`；上下相邻的两个 tile 用同一块 `W`，各读一遍。

## 八、cluster 与 TMA multicast

作用层：L2 → shared memory，跨 SM。

thread block cluster，即线程块集群，是 Hopper 新增的一层：同一 cluster 的 block 同时调度到同一 GPC 内的 SM 上，可以访问彼此的 shared memory 与 mbarrier。TMA multicast 从 L2 读一次，写入 cluster 内的多个 block。

::ml-hier:cluster::

| 机制 | 作用 |
| :--- | :--- |
| thread block cluster | block 同时调度到同一 GPC 内的 SM，可访问彼此的 shared memory 与 mbarrier |
| TMA multicast | 从 L2 读一次，写入 cluster 内多个 block |
| `mapa` | 把 shared memory 地址映射到 cluster 内另一个 block |

上下相邻、共用 `W` 的两个 tile 组成 cluster，各搬 `W` 块的一半，多播给双方：

```cpp
tma_load(sa, &tmA, &full[s], t * BK, bm);                        // A 各搬各的
tma_load_mc(sa + A_BYTES + rank * (BN / CL) * BK * 2, &tmW, &full[s],
            t * BK, bn + rank * (BN / CL), (1 << CL) - 1);       // W 搬一半，掩码 0b11 发往两个 block
```

每 block 每步从 L2 读 32 KB（原 48 KB）。代价是 stage 要等两个 block 都用完：empty 计数由 8 改为 16，consumer 用 `mapa` 在对方的 mbarrier 上也报到。

::ml-cluster3::

这一版同时做了另外两处改动。一是常驻 kernel（persistent kernel）：只启动 132 个 block，循环领取 tile，consumer 写回上一块时 producer 已在搬下一块；二是 consumer 发完第 $t$ 块的 `wgmma` 后只等第 $t-1$ 块算完，就归还它的 stage。

```cpp
for (int id = blockIdx.x / CL; id < tiles; id += gridDim.x / CL) {   // grid = 132 时循环约 15.5 次
    tile_mn(id, bm, bn);
    // ...
}
```

三处合计，实测 775.5 TFLOPS（97.0%），比分组版高 2.8%。同一次运行中逐项加入的结果为：

| 版本 | TFLOPS |
| :--- | ---: |
| 分组版 | 754.6 |
| 只改归还 stage 的时机 | 754.5 |
| 再改为常驻 kernel | 666.3 |
| 再加 cluster 与多播 | 775.5 |

只改归还时机，速度不变：两个 consumer 轮流使用 Tensor Core，空档已被对方填补。单独加上常驻，速度降低 12%，ncu 显示显存读取量由 1.63 GB 增加到 2.55 GB；加上 cluster 后显存读取量降为 1.32 GB，Tensor 管线忙 92.6%。

> cuBLAS 选用的 kernel（`nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT`）同样是常驻 kernel、2 个 block 的 cluster：`320x128` 为 tile 形状，`64x3` 为 $K$ 宽度 64、3 个 stage。

**瓶颈**：写回。tile 结束时每个 consumer 线程零散写显存 64 次、每次 4 字节，期间不发 `wgmma`。

## 九、TMA store

作用层：寄存器 → 显存。

改由 TMA 写回：consumer 把结果按 swizzle 规则写入 shared memory，一个线程发 4 条 TMA store，consumer 随即开始下一块。`fence_proxy_async` 让普通写入对 TMA 可见；`bulk_wait_read0` 等上次 TMA store 读完 shared memory，才能覆盖结果区。

```cpp
if (tid % 128 == 0) bulk_wait_read0();            // 上一块的 TMA store 已读完 shared memory
named_bar(1 + c, 128);
for (int j = 0; j < 32; j++) {                     // 按 128B 规则写入
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

代价是 shared memory：结果区 64 KB 加 4 个 stage 超过每 block 227 KB 的上限，减为 3 个 stage，共 208 KB；3 个 stage 仍可提前约 2000 个时钟发拷贝。

::ml-timeline:tstore::

实测 791.0 TFLOPS，为 cuBLAS 的 99.0%。ncu 中 Tensor 管线忙 93.8%。

## 十、功耗墙、能效与各规模

**功耗墙。** 以上是短时计时。连续运行 2 秒、NVML 采样时，8192 阶手写版与 cuBLAS 的功耗都达到 700 W 的上限，时钟由 1830 降到约 1375 MHz；按这一时钟，峰值为 $132\times4096\times1.375\ \text{GHz}=743$ TFLOPS，手写版用到其中的 93.0%，cuBLAS 为 92.7%。

::ml-power::

| 阶数 | kernel | TFLOPS | SM 时钟 | 功耗 |
| ---: | :--- | ---: | ---: | ---: |
| 4096 | 最终版 | 661.26 | 1413 MHz | 696 W |
| 4096 | cuBLAS BF16 | 682.90 | 1458 MHz | 548 W |
| 8192 | 最终版 | 691.48 | 1375 MHz | 700 W |
| 8192 | cuBLAS BF16 | 688.85 | 1377 MHz | 697 W |

**补充实测。** 以下各项以最终版为基础测量。

改用 Hilbert 曲线调度，以 cluster 为单位发放 tile，2048、4096、8192 阶的变化都在 0.2% 以内：同时运行的 66 个 cluster 在曲线上同样构成约 16 行 × 8 列 tile 的区域，与 16 行一组相同。

首块省清零指单独处理每个 tile 的第 0 步：第一条 `wgmma` 的 scale-d 取立即数 0（$D=AB$），省去累加结果的清零。2048、4096、8192 阶分别提高 1.8%、0.9%、1.4%。

```cpp
if constexpr (MICRO) {                                  // 第 0 步单独剥出，不再清零
    mbar_wait(&full[s], cph);
    wgmma_fence();
    wgmma_n256_zero(d, make_desc(sa), make_desc(sw));   // scale-d = 0，累加结果只作输出（"=f"）
    for (int kk = 1; kk < BK / 16; kk++) wgmma_n256(d, make_desc(sa + kk * 16), make_desc(sw + kk * 16));
    wgmma_commit();
}
```

> 第一版用运行时变量作 scale-d，`wgmma` 被串行化，速度降低约 20%。

去掉 cluster 后，2048、4096 阶短时计时快 5.4%、2.6%，8192 阶慢 1.1%；连续运行 2 秒时，8192 阶不带 cluster 的版本（加上首块省清零）为 665.42 TFLOPS，带 cluster 的最终版为 691.48。

**能效。** 4096 阶去掉 cluster、加上首块省清零后，短时计时为 769.98 TFLOPS，超过同次运行的 cuBLAS（762.74）；连续运行 2 秒时功耗为 698 W、时钟为 1380 MHz，为 666.16 TFLOPS，落后 cuBLAS（682.90 TFLOPS、548 W）2.5%。每次浮点运算手写版约 1.05 pJ，cuBLAS 约 0.80 pJ，多 31%。

**各规模。** 2048 阶起手写版与 cuBLAS 相当：2048 阶 128 个 tile 一轮完成，快 5.0%；4096 阶为 97.3%。1024 阶只有 32 个 tile，100 个 SM 空闲，最终版为 cuBLAS 的 78.4%；$128\times128$ 块的 `wgmma` 版 tile 数是最终版的 2 倍，512、1024 阶为 21.02、117.73 TFLOPS，最终版为 16.45、102.77。256 阶两者都约 0.013 ms。

::ml-sizes::

## 十一、Blackwell

第 3 步起的各版依赖 `wgmma`，它只在 sm_90a 上可用，不能直接用于 B200 等 Blackwell 数据中心 GPU。Blackwell 的第五代 Tensor Core 改用 `tcgen05.mma`：一个线程即可发出，异步执行，完成后经 `tcgen05.commit` 在 mbarrier 上报到。

存储层级也多了一层。累加结果 D 不再放在寄存器，而放在 Tensor Core 旁新增的 Tensor Memory 中；它每个 block 128 行 × 512 列、每格 32 位，由一个 warp 显式分配。B 来自 shared memory，A 来自 shared memory 或 Tensor Memory。同一 cluster 中配对的两个 block 还可以共同执行一条 MMA。

## 十二、总结

前四步在 SM 内部，读取不再冲突，搬运不占用线程，指令更大，搬运与计算分工；后三步在 SM 之外，减少显存与 L2 的读取，并使写回与计算重叠。

::ml-summary::

Tensor Core 的峰值是固定的。矩阵乘法的速度取决于两点：Tensor Core 等待数据的时间，以及计算所用的指令。

## 参考

1. Simon Boehm. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM), 2022-12.
2. Pranjal Shankhdhar. [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), 2024-11-29.
3. NVIDIA. [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：`cp.async`、`ldmatrix`、`mma.sync`、`wgmma`、`cp.async.bulk.tensor`、`mbarrier`、`setmaxnreg`；Blackwell 的 `tcgen05.mma` 与 Tensor Memory（9.7.18 节）。
4. NVIDIA. [CUTLASS：Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：Hopper 的 warp specialization、persistent cooperative 与 ping-pong 调度、rasterization、split-K。
5. NVIDIA. NVIDIA H100 Tensor Core GPU Architecture（白皮书）。
6. Jay Shah et al. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608), 2024：warp specialization 与 ping-pong 调度。
7. DeepSeek. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。
8. MLC. [Modern GPU Programming for MLSys：Building a Tiled GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_basics/index.html)：Blackwell 上同一条优化路线。
9. 实验代码：[`scripts/matmul-ladder/`](https://github.com/luca-888/luca-888.github.io/tree/main/scripts/matmul-ladder)；原样输出：[`public/measurements/matmul-ladder-tc/`](https://github.com/luca-888/luca-888.github.io/tree/main/public/measurements/matmul-ladder-tc)（计时与功耗 `20261028-114712-article/`，各版 ncu `20261027-134747/`，对照与指令上限 `20261027-140146-extra/`）；tile 开始时间见 [`public/measurements/matmul-ladder/20261022-142921/`](https://github.com/luca-888/luca-888.github.io/tree/main/public/measurements/matmul-ladder/20261022-142921)。
