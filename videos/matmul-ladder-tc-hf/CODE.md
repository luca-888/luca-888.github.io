# 各版本代码（带注释）

来源：`scripts/matmul-ladder/matmul_tc.cu`。只摘每版的主循环，省略边界、初始化与写回细节；注释是为看视频稿加的。计算 $C = A W^\top$，A 是 M×K、W 是 N×K，都沿 K 连续，BF16 输入、FP32 累加。数字为 8192 阶短时计时。

| # | kernel | TFLOPS | 和上一版相比改了什么 |
| --- | --- | ---: | --- |
| 1 | `mm_wmma` | 93 | 起点 |
| 2 | `mm_mma<…,1>` | 269 | BK 32→64、swizzle、cp.async、ldmatrix + mma.sync（四处一起改） |
| 3 | `mm_mma<…,3>` | 309 | 3 级 stage 环 |
| 4 | `mm_wgmma<4>` | 484 | TMA + mbarrier + wgmma |
| 5 | `mm_wgmma_ws`（GROUP=0） | 480（4096 阶 635） | producer / consumer 分工，块放大到 128×256 |
| 6 | `mm_wgmma_ws`（GROUP=16） | 740 | 只改 tile 发放顺序 |
| 7 | `mm_wgmma_ws`（PIPE, 常驻, CL=2） | 757 | wgmma 前后重叠、常驻、cluster + multicast（三处一起改） |
| 8 | `mm_wgmma_ws`（STAGES=3, TSTORE） | 778 | TMA store 写回 |

---

## 1. wmma（93 TFLOPS）

```cpp
// 一个 block 算 C 的 128×128；8 个 warp 排成 2×4，每个 warp 算 64×32
constexpr int BM = 128, BN = 128, BK = 32;          // 沿 K 每次取 32 个元素 → shared memory 一行 32×2 = 64 B
__shared__ bf16 As[BM * BK], Ws[BN * BK];           // A、W 各一块 128×32，没有 padding、没有 swizzle

wmma::fragment<wmma::matrix_a, 16,16,16, bf16, wmma::row_major> a[4];   // fragment：一个 warp 共同持有的矩阵片段
wmma::fragment<wmma::matrix_b, 16,16,16, bf16, wmma::col_major> b[2];   // W[n][k] 正好是 B(k,n) 的列主序
wmma::fragment<wmma::accumulator, 16,16,16, float> c[4][2];             // 累加器，FP32，在寄存器里

for (int k0 = 0; k0 < K; k0 += BK) {
    // ① 搬运：每个线程一次 16 B，显存 → 寄存器 → shared memory（普通读写，经过寄存器）
    for (int i = tid; i < BM * BK / 8; i += 256) {
        int r = i / (BK / 8), c8 = i % (BK / 8) * 8;
        *(int4*)&As[r * BK + c8] = *(const int4*)&A[(bm + r) * K + k0 + c8];
        *(int4*)&Ws[r * BK + c8] = *(const int4*)&W[(bn + r) * K + k0 + c8];
    }
    __syncthreads();                                  // 等全体搬完才能算

    // ② 计算：每 16 列 K 一轮
    for (int kk = 0; kk < BK; kk += 16) {
        // shared memory → 寄存器；最后一个参数 BK 是行距（32 个元素 = 64 B）
        // 片段按 8×8 小块读，8 行地址相隔 64 B，只落在 2 组 bank → 4 轮
        for (int i = 0; i < 4; i++) wmma::load_matrix_sync(a[i], &As[(wRow*64 + i*16) * BK + kk], BK);
        for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Ws[(wCol*32 + j*16) * BK + kk], BK);
        // 寄存器 → Tensor Core：每条 16×16×16
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
    }
    __syncthreads();                                  // 等全体算完才能覆盖 As、Ws
}
// 写回：片段先落到 shared memory 转成 BF16，每个线程再零散写显存
```

问题：搬运和计算完全串行；数据两次经过寄存器；读片段有 bank conflict。

---

## 2. mma（单级，269 TFLOPS）

```cpp
// swizzle：一行 64 个 bf16 = 128 B = 8 个 16 B 块；第 row 行的第 c 块放到 c ^ (row & 7)
// ldmatrix 一次读 8 行的同一块，换位后 8 个地址落在 8 组不同的 bank，一轮读完（与 TMA 的 SWIZZLE_128B 同一规则）
__device__ uint32_t swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

constexpr int BK = 64;                                // 一行 128 B，正好盖满 32 个 bank

auto load = [&](int stage, int k0) {
    // 显存/L2 → shared memory：cp.async 每条 16 B，不经过寄存器；写入地址已经过 swz 换位
    for (int i = tid; i < BM * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sa + swz(r, c), Ab + r * K + k0 + c * 8); }
    for (int i = tid; i < BN * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sw + swz(r, c), Wb + r * K + k0 + c * 8); }
};
auto compute = [&](int stage) {
    for (int kk = 0; kk < BK / 16; kk++) {
        uint32_t a[MT][4], b[NT][2];
        // shared memory → 寄存器：ldmatrix.x4 一条读 4 个 8×8 小块；读取地址同样经过 swz
        for (int i = 0; i < MT; i++) ldmatrix_x4(a[i], sa + swz(wRow*WM + i*16 + lane % 16, kk*2 + lane / 16));
        for (int j = 0; j < NT; j += 2) { /* 同理读 W 的片段 */ }
        // 寄存器 → Tensor Core：mma.sync.m16n8k16，一条 2048 次乘加
        for (int i = 0; i < MT; i++)
            for (int j = 0; j < NT; j++) mma16816(acc[i][j], a[i], b[j]);
    }
};

for (int t = 0; t < T; t++) {
    load(0, t * BK);                                  // 发拷贝
    cp_async_commit();
    cp_async_wait<0>();                               // 等它全部到齐 ← Tensor Core 在这里空等
    __syncthreads();
    compute(0);                                       // 再算
    __syncthreads();
}
```

一次改了四处：BK 32→64、swizzle、cp.async、wmma → ldmatrix + mma.sync。仍是“搬完、等齐、再算”。

---

## 3. mma_async（3 级，309 TFLOPS）

```cpp
// 只改主循环：shared memory 开 3 个 stage 组成环，拷贝提前 2 块发出
for (int s = 0; s < STAGES - 1; s++) { load(s, s * BK); cp_async_commit(); }   // 先发第 0、1 块

for (int t = 0; t < T; t++) {
    cp_async_wait<STAGES - 2>();      // 只等第 t 块到齐；后面最多还有 1 组在路上
    __syncthreads();                  // 也保证大家都算完了第 t-1 块，它的 stage 可以覆盖
    int tn = t + STAGES - 1;
    if (tn < T) load(tn % STAGES, tn * BK);   // 发第 t+2 块的拷贝
    cp_async_commit();
    compute(t % STAGES);              // 算第 t 块，同时第 t+1、t+2 块在路上
}
```

各种 stage 数与块形状都停在约 300–329：mma.sync 每条太小，要每时钟发一条，shared memory 每时钟要读 192 B 而只能供 128 B。

---

## 4. wgmma（484 TFLOPS）

```cpp
// block 只有 128 个线程 = 1 个 warpgroup（4 个 warp）
__shared__ uint64_t full[STAGES];                     // mbarrier：每个 stage 一个，记 TMA 到达的字节数

auto issue = [&](int t) {                             // 只由线程 0 调用
    int s = t % STAGES;
    mbar_expect_tx(&full[s], STAGE_BYTES);            // 告诉 mbarrier：这一 stage 要等 32 KB
    tma_load(sa,           &tmA, &full[s], t * BK, bm);   // TMA：一条指令搬 A 的一整块 128×64，硬件按 128B 规则换位
    tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);   // 再搬 W 的一块；到齐后自动在 full[s] 上记账
};
if (tid == 0) for (int t = 0; t < STAGES; t++) issue(t);  // 先把 4 个 stage 都发出去

for (int t = 0; t < T; t++) {
    int s = t % STAGES;
    mbar_wait(&full[s], (t / STAGES) & 1);            // 等这一 stage 的字节数计满
    wgmma_fence();
    for (int kk = 0; kk < BK / 16; kk++) {
        // wgmma：warpgroup 共同发出，m64n128k16 一条 131072 次乘加（mma.sync 的 64 倍）
        // 操作数用 make_desc 描述 shared memory 地址，直接从 shared memory 读，不经寄存器、没有 ldmatrix
        wgmma_n128(d[0], make_desc(sa + kk*16),          make_desc(sw + kk*16));   // 上 64 行
        wgmma_n128(d[1], make_desc(sa + 64*BK + kk*16),  make_desc(sw + kk*16));   // 下 64 行
    }
    wgmma_commit();
    wgmma_wait<0>();                                  // 等这批 wgmma 算完
    __syncthreads();                                  // 等全体都读完这一 stage
    if (tid == 0 && t + STAGES < T) issue(t + STAGES);   // 线程 0 这才发下一块 ← 搬和算互相等
}
```

---

## 5. wgmma_ws：warp specialization（8192 阶 480，4096 阶 635 TFLOPS）

```cpp
// block 384 个线程 = 3 个 warpgroup；块放大到 128×256
__shared__ uint64_t full[STAGES], empty[STAGES];      // full：TMA 已到；empty：consumer 已用完

if (wg == 0) {                                        // ── producer
    setmaxnreg_dec<40>();                             // 只发 TMA，寄存器降到 40 个，让给 consumer
    if (tid == 0)
        for (int t = 0; t < T; t++, it++) {
            int s = it % STAGES, ph = (it / STAGES) & 1;
            mbar_wait(&empty[s], ph ^ 1);             // 等 consumer 用完这一 stage（头 4 块不用等）
            mbar_expect_tx(&full[s], STAGE_BYTES);
            tma_load(sa,           &tmA, &full[s], t * BK, bm);
            tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);
        }
} else {                                              // ── consumer（1、2 号，各算 64×256）
    setmaxnreg_inc<232>();                            // 累加器每线程 128 个 float，需要更多寄存器
    float d[128];
    for (int t = 0; t < T; t++, it++) {
        int s = it % STAGES, ph = (it / STAGES) & 1;
        mbar_wait(&full[s], ph);                      // 等数据到
        wgmma_fence();
        for (int kk = 0; kk < BK / 16; kk++)
            wgmma_n256(d, make_desc(sa + kk*16), make_desc(sw + kk*16));   // m64n256k16
        wgmma_commit();
        wgmma_wait<0>();
        if (tid % 32 == 0) mbar_arrive(&empty[s]);    // 每个 warp 报到一次，8 个都报到 = 这一 stage 可以覆盖
    }
}
```

producer 不再等全体同步，一直往前搬；两个 consumer 交替用 Tensor Core。8192 阶不变的原因在 L2（见下一版）。

---

## 6. wgmma_l2：tile 发放顺序（740 TFLOPS）

```cpp
// kernel 其余部分不变，只改“第 id 个 block 算哪一块 tile”
auto tile_mn = [&](int id, int& bm, int& bn) {
    int m, n;
    if (GROUP == 0) {                 // 原来：按行铺，先走完一整行的 32 个 tile
        m = id / numN; n = id % numN; //   同时运行的 132 个 tile ≈ 4 行 × 全部 32 列，要用到整个 W（128 MB > L2 50 MB）
    } else {                          // 现在：每 16 行一组，组内按列走
        int per = GROUP * numN, first = id / per * GROUP, gs = min(GROUP, numM - first), r = id % per;
        m = first + r % gs;           //   相邻编号的 tile 在同一列、相邻行 → 共用同一块 W，几乎同时读
        n = r / gs;                   //   一块 W 读进 L2，随即被 16 个 tile 读走
    }
    bm = m * BM; bn = n * BN;
};
```

---

## 7. wgmma_cluster：PIPE + 常驻 + cluster（757 TFLOPS）

```cpp
// (a) PIPE：consumer 发完第 t 块的 wgmma 不等它算完，只等第 t-1 块
wgmma_commit();
wgmma_wait<1>();                                      // 最多留 1 批在算
if (t > 0) release((it - 1) % STAGES);                // 归还第 t-1 块的 stage → Tensor Core 前后两批之间不留空

// (b) 常驻：grid = SM 数，每个 block 循环领 tile；写回上一块时 producer 已在搬下一块
for (int id = blockIdx.x / CL; id < tiles; id += gridDim.x / CL) { tile_mn(id, bm, bn); /* ... */ }

// (c) cluster + multicast：相邻两个 block 组成 cluster（CL = 2），算上下相邻、共用同一段 W 的两块
tma_load(sa, &tmA, &full[s], t * BK, bm);             // A 各搬各的
tma_load_mc(sa + A_BYTES + rank * (BN / CL) * BK * 2, // W 的 256 行每个 block 搬一半（rank 0 搬前 128 行，rank 1 搬后 128 行）
            &tmW, &full[s], t * BK, bn + rank * (BN / CL),
            (1 << CL) - 1);                           // mask = 0b11：从 L2 读一次，同时写进两个 block 的 shared memory
// 代价：empty 要等两个 block 共 16 个 consumer warp 都用完
for (int r = 0; r < CL; r++) mbar_arrive_cluster(&empty[s], r);
```

---

## 8. wgmma_tstore：TMA store（778 TFLOPS，cuBLAS 的 99.5%）

```cpp
// STAGES 4 → 3，腾出 64 KB 作结果区；每个 consumer 32 KB（64×256 BF16）
uint8_t* cs = smem + STAGES * STAGE_BYTES + c * 32768;
if (tid % 128 == 0) bulk_wait_read0();                // 上一块的 TMA 写回已读完结果区，才能覆盖
named_bar(1 + c, 128);
for (int j = 0; j < 32; j++) {
    // 累加器转 BF16，按 128B swizzle 写进 shared memory（无 bank conflict）
    uint8_t* p = cs + j / 8 * 8192 + (((j % 8) ^ (r & 7)) << 4) + lane % 4 * 4;
    *(__nv_bfloat162*)(p + r * 128)       = __floats2bfloat162_rn(d[4*j],   d[4*j+1]);
    *(__nv_bfloat162*)(p + (r + 8) * 128) = __floats2bfloat162_rn(d[4*j+2], d[4*j+3]);
}
fence_proxy_async();                                  // 让普通写对 TMA 可见
named_bar(1 + c, 128);
if (tid % 128 == 0) {
    for (int q = 0; q < 4; q++) tma_store(&tmC, cs + q * 8192, bn + q * 64, bm + c * 64);   // 一个线程整块写回显存
    bulk_commit();                                    // 不等写完，consumer 直接回去算下一块
}
// 原来（store_acc）：每个线程从寄存器零散写显存 64 次，这期间不发 wgmma
```

---

## 拆不开的两处

- 第 2 版四处一起改，27% 里各占多少没有单独测；bank conflict 是起点写法（BK = 32、无 padding）造成的，wmma 加 padding 也能消掉。
- 第 7 版三处一起改；文章只单独测过“常驻”（单独加反而慢 11%，加上 cluster 后更快）。
