// 1.3 Tensor Core 线：BF16 输入、FP32 累加、BF16 输出的矩阵乘，手写 kernel 对照 cuBLAS BF16。
// 布局用推理里 Linear 层的 "TN"：A 是 M×K 行主序（激活），W 是 N×K 行主序（权重，每行沿 K 连续），C = A·Wᵀ 是 M×N 行主序。
// 编译：nvcc -O3 -arch=sm_90a -std=c++17 -o matmul_tc matmul_tc.cu -lcublas -L/usr/local/cuda/lib64/stubs -lnvidia-ml
// 运行：./matmul_tc [kernel,kernel,...|all] [N,N,...]（不带参数 = 全部 kernel、N = 256…8192；M = N = K）
//       ./matmul_tc soak kernel,... N   每个 kernel 连续跑 2 s，用 NVML 采 SM 频率和功耗
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvml.h>
#include <chrono>
using namespace nvcuda;
typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- wmma
// wmma：最朴素的 Tensor Core 版。block 算 128×128，8 个 warp 各算 64×32（4×2 个 16×16 片段）；
// 沿 K 每次把 A、W 各 128×32 同步搬进共享内存，每个 warp 用 wmma::mma_sync 一条指令做 16×16×16
__global__ void __launch_bounds__(256) mm_wmma(const bf16* A, const bf16* W, bf16* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 32;
    __shared__ __align__(128) bf16 As[BM * BK], Ws[BN * BK];
    __shared__ __align__(128) float stage[8][16 * 16];             // 收尾时把 FP32 片段落到这里再转成 BF16
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int wRow = warp / 4, wCol = warp % 4;
    int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b[2];   // W[n][k] 正好是 B(k,n) 的列主序
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.f);
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK / 8; i += 256) {              // 每线程一次搬 8 个 bf16（16 B）
            int r = i / (BK / 8), c8 = i % (BK / 8) * 8;
            *(int4*)&As[r * BK + c8] = *(const int4*)&A[(size_t)(bm + r) * K + k0 + c8];
            *(int4*)&Ws[r * BK + c8] = *(const int4*)&W[(size_t)(bn + r) * K + k0 + c8];
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            for (int i = 0; i < 4; i++) wmma::load_matrix_sync(a[i], &As[(wRow * 64 + i * 16) * BK + kk], BK);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Ws[(wCol * 32 + j * 16) * BK + kk], BK);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(stage[warp], c[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            int r = lane / 2, c0 = lane % 2 * 8;                    // 每个 lane 写一行里的 8 个
            __align__(16) bf16 o[8];
            for (int e = 0; e < 8; e++) o[e] = __float2bfloat16(stage[warp][r * 16 + c0 + e]);
            *(int4*)&C[(size_t)(bm + wRow * 64 + i * 16 + r) * N + bn + wCol * 32 + j * 16 + c0] = *(int4*)o;
            __syncwarp();
        }
}

// ---------------------------------------------------------------- PTX 小工具
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int n> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(n)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 一行 64 个 bf16 = 128 B = 8 个 16 B 块；第 row 行的第 c 块放到 c ^ (row & 7)。ldmatrix 一次读 8 行的同一块，
// 换位后 8 个地址落在 8 个不同的 bank 组 —— 和 TMA 的 SWIZZLE_128B 是同一个规则
__device__ __forceinline__ uint32_t swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

// ---------------------------------------------------------------- mma / mma_async
// mma（STAGES=1）：换成 PTX 级 mma.sync.m16n8k16 + ldmatrix，共享内存按上面的 XOR 规则换位存（消除 ldmatrix 的 bank 冲突），
// 显存→共享内存改用 cp.async（不经寄存器）。但还是"搬完一块、等齐、再算"，搬和算不重叠。
// mma_async（STAGES=3）：只改一处 —— 共享内存开 3 级环形缓冲，cp.async 提前 2 块发出，搬第 t+2 块和算第 t 块重叠
template <int BM, int BN, int STAGES>
__global__ void __launch_bounds__(256) mm_mma(const bf16* A, const bf16* W, bf16* C, int M, int N, int K) {
    constexpr int BK = 64, WM = BM / 2, WN = BN / 4, MT = WM / 16, NT = WN / 8;   // 8 个 warp 排成 2×4
    constexpr int A_BYTES = BM * BK * 2, STAGE_BYTES = (BM + BN) * BK * 2;
    extern __shared__ __align__(128) uint8_t smem[];
    uint32_t sbase = smem_u32(smem);
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int wRow = warp / 4, wCol = warp % 4;
    int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const bf16* Ab = A + (size_t)bm * K;
    const bf16* Wb = W + (size_t)bn * K;
    auto load = [&](int stage, int k0) {
        uint32_t sa = sbase + stage * STAGE_BYTES, sw = sa + A_BYTES;
        for (int i = tid; i < BM * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sa + swz(r, c), Ab + (size_t)r * K + k0 + c * 8); }
        for (int i = tid; i < BN * 8; i += 256) { int r = i / 8, c = i % 8; cp_async16(sw + swz(r, c), Wb + (size_t)r * K + k0 + c * 8); }
    };
    float acc[MT][NT][4] = {};
    auto compute = [&](int stage) {
        uint32_t sa = sbase + stage * STAGE_BYTES, sw = sa + A_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            uint32_t a[MT][4], b[NT][2];
#pragma unroll
            for (int i = 0; i < MT; i++) ldmatrix_x4(a[i], sa + swz(wRow * WM + i * 16 + lane % 16, kk * 2 + lane / 16));
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                uint32_t t[4];
                ldmatrix_x4(t, sw + swz(wCol * WN + j * 8 + lane % 8 + lane / 16 * 8, kk * 2 + lane / 8 % 2));
                b[j][0] = t[0]; b[j][1] = t[1]; b[j + 1][0] = t[2]; b[j + 1][1] = t[3];
            }
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < NT; j++) mma16816(acc[i][j], a[i], b[j]);
        }
    };
    int T = K / BK;
    if constexpr (STAGES == 1) {
        for (int t = 0; t < T; t++) {
            load(0, t * BK);
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    } else {
        for (int s = 0; s < STAGES - 1; s++) { if (s < T) load(s, s * BK); cp_async_commit(); }
        for (int t = 0; t < T; t++) {
            cp_async_wait<STAGES - 2>();                            // 第 t 块到齐（最多还有 STAGES-2 组在路上）
            __syncthreads();                                        // 也保证大家都算完了第 t-1 块，它的槽位可以覆盖
            int tn = t + STAGES - 1;
            if (tn < T) load(tn % STAGES, tn * BK);
            cp_async_commit();
            compute(t % STAGES);
        }
    }
    for (int i = 0; i < MT; i++)
        for (int j = 0; j < NT; j++) {
            int r = bm + wRow * WM + i * 16 + lane / 4, c = bn + wCol * WN + j * 8 + lane % 4 * 2;
            *(__nv_bfloat162*)&C[(size_t)r * N + c] = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            *(__nv_bfloat162*)&C[(size_t)(r + 8) * N + c] = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}

// ---------------------------------------------------------------- Hopper：TMA、mbarrier、wgmma
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
    asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra.uni DONE;\nbra.uni LAB_WAIT;\nDONE:\n}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// TMA：一条指令把全局内存里 {c0 = k, c1 = 行} 起的一个 box 搬进共享内存（按 128B 规则换位），搬完在 bar 上记账
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                 ::"r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
// 共享内存矩阵描述符：K 方向连续、128B 换位，8 行（1024 B）一组，组间跨 1024 B
__device__ __forceinline__ uint64_t make_desc(const void* p) {
    uint64_t a = smem_u32(p);
    return ((a & 0x3FFFF) >> 4) | ((uint64_t)(16 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int n> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(n) : "memory"); }
// 集群（cluster）：相邻 SM 上的几个 CTA 能互相访问共享内存、在对方的 mbarrier 上报到
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t cta) {   // 在集群里第 cta 个 CTA 的同一个 barrier 上报到
    uint32_t ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_u32(bar)), "r"(cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(ra) : "memory");
}
// TMA 多播：从显存读一次，写进 mask 里每个 CTA 的同一共享内存位置，并在各自同一位置的 mbarrier 上记账
__device__ __forceinline__ void tma_load_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
                 ::"r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
// TMA 存：把共享内存里一个 box 写回全局；bulk_group 计数，wait_group.read 0 = 共享内存已经读完、可以覆盖
__device__ __forceinline__ void tma_store(const CUtensorMap* map, const void* src, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 ::"l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
template <int r> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(r)); }
template <int r> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(r)); }

// wgmma m64n128k16：一个 warpgroup（128 线程）算 64×128，A、B 都从共享内存经描述符读，累加器 d 每线程 64 个 float
__device__ __forceinline__ void wgmma_n128(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(1));
}

// wgmma m64n256k16：一个 warpgroup（128 线程）算 64×256，A、B 都从共享内存经描述符读，累加器 d 每线程 128 个 float
__device__ __forceinline__ void wgmma_n256(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(1));
}

// 同上，但 scale-d = 0（立即数）：D = A·B，不读旧累加器；累加器只作输出（"=f"），这样 128 个寄存器只由 wgmma 定义，
// ptxas 不会因为"非 wgmma 指令定义了累加器"而把 wgmma 串行化（C7515）
__device__ __forceinline__ void wgmma_n256_zero(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 0, 1, 1, 0, 0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]), "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]), "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]), "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]), "=f"(d[110]), "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]), "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]), "=f"(d[120]), "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]), "=f"(d[126]), "=f"(d[127])
        : "l"(da), "l"(db));
}

// wgmma 累加器的布局：warpgroup 里第 w 个 warp 管 16 行；第 j 个 8 列块用 d[4j..4j+3]，和 mma.sync 的 C 片段一样
template <int BN>
__device__ __forceinline__ void store_acc(const float* d, bf16* C, int N, int row0, int col0) {
    int lane = threadIdx.x % 32, w = threadIdx.x / 32 % 4;
    int r = row0 + w * 16 + lane / 4;
#pragma unroll
    for (int j = 0; j < BN / 8; j++) {
        int c = col0 + j * 8 + lane % 4 * 2;
        *(__nv_bfloat162*)&C[(size_t)r * N + c] = __floats2bfloat162_rn(d[4 * j], d[4 * j + 1]);
        *(__nv_bfloat162*)&C[(size_t)(r + 8) * N + c] = __floats2bfloat162_rn(d[4 * j + 2], d[4 * j + 3]);
    }
}

// ---------------------------------------------------------------- wgmma
// wgmma：换到 Hopper 的三件套。TMA 一条指令搬一整块（不再是 256 个线程各发 cp.async），到没到用 mbarrier 记账；
// 一个 warpgroup（128 线程）用 wgmma.mma_async 直接从共享内存读 A、W（不再 ldmatrix 进寄存器），算 128×128。
// 4 级缓冲，线程 0 提前发 TMA；但每块算完要 wgmma_wait<0> + __syncthreads 才放下一块进来
template <int STAGES>
__global__ void __launch_bounds__(128) mm_wgmma(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                                                bf16* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 64, A_BYTES = BM * BK * 2, STAGE_BYTES = (BM + BN) * BK * 2;
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);   // 128B 换位要求 1024 B 对齐
    __shared__ __align__(8) uint64_t full[STAGES];
    int tid = threadIdx.x;
    int bm = blockIdx.y * BM, bn = blockIdx.x * BN, T = K / BK;
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) mbar_init(&full[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    auto issue = [&](int t) {
        int s = t % STAGES;
        uint8_t* sa = smem + s * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES);
        tma_load(sa, &tmA, &full[s], t * BK, bm);
        tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);
    };
    if (tid == 0)
        for (int t = 0; t < STAGES && t < T; t++) issue(t);
    float d[2][64];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 64; j++) d[i][j] = 0;
    for (int t = 0; t < T; t++) {
        int s = t % STAGES;
        mbar_wait(&full[s], (t / STAGES) & 1);
        bf16* sa = (bf16*)(smem + s * STAGE_BYTES);
        bf16* sw = (bf16*)(smem + s * STAGE_BYTES + A_BYTES);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            wgmma_n128(d[0], make_desc(sa + kk * 16), make_desc(sw + kk * 16));            // 上 64 行
            wgmma_n128(d[1], make_desc(sa + 64 * BK + kk * 16), make_desc(sw + kk * 16));  // 下 64 行
        }
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();                                            // 大家都读完这一槽，才能让 TMA 覆盖
        if (tid == 0 && t + STAGES < T) issue(t + STAGES);
    }
    store_acc<128>(d[0], C, N, bm, bn);
    store_acc<128>(d[1], C, N, bm + 64, bn);
}

// ---------------------------------------------------------------- wgmma_ws 及其后续
// wgmma_ws（GROUP=0, PIPE=false, 每块一个 CTA）：warp 特化。3 个 warpgroup：0 号是生产者（只有一个线程发 TMA，寄存器降到 40），
// 1、2 号是消费者（寄存器升到 232），各算 64×256（m64n256k16）。每个槽位两个 mbarrier：full（TMA 到齐）、empty（消费者用完），
// 生产者不再等全体同步，一直往前搬；块从 128×128 放大到 128×256，每搬 1 字节做的 FLOP 多 1/3。
// wgmma_l2（GROUP>0）：只改 tile 的发放顺序。同时在跑的 132 个 CTA 不再铺成"4 行 × 整整 32 列"，而是 GROUP 行一组竖着走，
//   同一时刻要用到的 A、W 行更少，L2 里放得下，少回显存。
// wgmma_pipe（PIPE=true）：消费者发完第 t 块的 wgmma 不再等它算完，只等第 t-1 块（wait_group 1）就归还 t-1 的槽位，
//   Tensor Core 前后两批之间不留空。
// wgmma_persist（launch 时 grid = SM 数）：kernel 代码不变，CTA 常驻、循环领 tile；消费者写回上一块的结果时，生产者已经在搬下一块。
// wgmma_cluster（CL=2）：相邻两个 CTA 组成集群，算上下相邻的两块（同一段 W）。W 块 256 行各搬一半、TMA 多播给两边，
//   每个 CTA 从 L2 读的 W 减半；代价是 empty 要等两边 16 个消费者 warp 都用完才能覆盖。
// wgmma_tstore（TSTORE=true，STAGES 4→3 腾出 64 KB）：收尾不再由每个线程直接往显存零散写 4 B，而是按 128B 换位写进共享内存
//   （无 bank 冲突），一个线程发 TMA 整块写回；TMA 写的同时消费者已经回去算下一块
// Hilbert 曲线：第 d 个点 → n×n 网格（n 为 2 的幂）里的 (x, y)，相邻编号的点在网格里也相邻
__device__ __forceinline__ void hilbert_d2xy(int n, int d, int& x, int& y) {
    x = y = 0;
    for (int s = 1; s < n; s *= 2) {
        int rx = 1 & (d / 2), ry = 1 & (d ^ rx);
        if (ry == 0) {
            if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
            int t = x; x = y; y = t;
        }
        x += s * rx; y += s * ry; d /= 4;
    }
}
// wgmma_hilbert（HILBERT=true）：只改发放顺序。集群粒度的 tile 网格（numM × numN）按 Hilbert 曲线编号，
//   同时在跑的 66 个集群落在一块紧凑的近似正方形区域里；网格不是 2 的幂的正方形时退回 GROUP 分组。
// wgmma_micro（MICRO=true）：每块第一条 wgmma 用 scale-d=0（D = A·B），省掉 128 个寄存器的清零；
//   槽位号和相位用计数器递增，不再每步算 it % 3、it / 3。
template <int STAGES, int GROUP, bool PIPE, int CL, bool TSTORE = false, bool HILBERT = false, bool MICRO = false>
__global__ void __launch_bounds__(384, 1) mm_wgmma_ws(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                                                      const __grid_constant__ CUtensorMap tmC, bf16* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 64, A_BYTES = BM * BK * 2, STAGE_BYTES = (BM + BN) * BK * 2;
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
    int tid = threadIdx.x, wg = tid / 128;
    int rank = CL == 1 ? 0 : (int)cluster_rank();
    int numM = M / BM / CL, numN = N / BN, tiles = numM * numN, T = K / BK;   // 一个集群一次领 CL 块（上下相邻）
    auto tile_mn = [&](int id, int& bm, int& bn) {
        int m, n;
        if constexpr (HILBERT) {
            if (numM == numN && (numM & (numM - 1)) == 0) { hilbert_d2xy(numM, id, m, n); bm = (m * CL + rank) * BM; bn = n * BN; return; }
        }
        if constexpr (GROUP == 0) { m = id / numN; n = id % numN; }      // 按行铺：先走完一整行 N
        else {                                                            // GROUP 行一组，组内按列走
            constexpr int G = GROUP / CL;
            int per = G * numN, first = id / per * G, gs = min(G, numM - first), r = id % per;
            m = first + r % gs; n = r / gs;
        }
        bm = (m * CL + rank) * BM; bn = n * BN;
    };
    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8 * CL); }   // empty：集群里每个消费者 warp 各报一次
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if constexpr (CL > 1) cluster_sync(); else __syncthreads();
    if (wg == 0) {
        setmaxnreg_dec<40>();
        if (tid == 0) {
            int it = 0;                                                   // 跨 tile 连续计数的 k 步，决定槽位和相位
            int ps = 0, pph = 0;                                          // MICRO：槽位、相位直接递增
            for (int id = blockIdx.x / CL; id < tiles; id += gridDim.x / CL) {
                int bm, bn;
                tile_mn(id, bm, bn);
                for (int t = 0; t < T; t++, it++) {
                    int s, ph;
                    if constexpr (MICRO) { s = ps; ph = pph; if (++ps == STAGES) { ps = 0; pph ^= 1; } }
                    else { s = it % STAGES; ph = (it / STAGES) & 1; }
                    mbar_wait(&empty[s], ph ^ 1);                         // 新 barrier 的"上一相"算已完成，所以头 STAGES 块不等
                    uint8_t* sa = smem + s * STAGE_BYTES;
                    mbar_expect_tx(&full[s], STAGE_BYTES);
                    tma_load(sa, &tmA, &full[s], t * BK, bm);
                    if constexpr (CL == 1) tma_load(sa + A_BYTES, &tmW, &full[s], t * BK, bn);
                    else tma_load_mc(sa + A_BYTES + rank * (BN / CL) * BK * 2, &tmW, &full[s], t * BK, bn + rank * (BN / CL), (1 << CL) - 1);
                }
            }
        }
    } else {
        setmaxnreg_inc<232>();
        int c = wg - 1, it = 0;
        int cs_ = 0, cph = 0;                                             // MICRO：槽位、相位直接递增
        float d[128];
        auto release = [&](int s) {
            if (tid % 32 == 0) {
                if constexpr (CL == 1) mbar_arrive(&empty[s]);
                else for (int r = 0; r < CL; r++) mbar_arrive_cluster(&empty[s], r);
            }
        };
        for (int id = blockIdx.x / CL; id < tiles; id += gridDim.x / CL) {
            int bm, bn;
            tile_mn(id, bm, bn);
            if constexpr (!MICRO) for (int j = 0; j < 128; j++) d[j] = 0;
            int prev = 0;
            if constexpr (MICRO) {                                        // 第 0 个 k 步单独剥出来：首条 wgmma 用 scale-d=0
                int s = cs_;
                mbar_wait(&full[s], cph);
                if (++cs_ == STAGES) { cs_ = 0; cph ^= 1; }
                bf16* sa = (bf16*)(smem + s * STAGE_BYTES) + c * 64 * BK;
                bf16* sw = (bf16*)(smem + s * STAGE_BYTES + A_BYTES);
                wgmma_fence();
                wgmma_n256_zero(d, make_desc(sa), make_desc(sw));
#pragma unroll
                for (int kk = 1; kk < BK / 16; kk++) wgmma_n256(d, make_desc(sa + kk * 16), make_desc(sw + kk * 16));
                wgmma_commit();
                prev = s;
                it++;
            }
            for (int t = MICRO ? 1 : 0; t < T; t++, it++) {
                int s, ph;
                if constexpr (MICRO) { s = cs_; ph = cph; if (++cs_ == STAGES) { cs_ = 0; cph ^= 1; } }
                else { s = it % STAGES; ph = (it / STAGES) & 1; }
                mbar_wait(&full[s], ph);
                bf16* sa = (bf16*)(smem + s * STAGE_BYTES) + c * 64 * BK;
                bf16* sw = (bf16*)(smem + s * STAGE_BYTES + A_BYTES);
                wgmma_fence();
#pragma unroll
                for (int kk = 0; kk < BK / 16; kk++) wgmma_n256(d, make_desc(sa + kk * 16), make_desc(sw + kk * 16));
                wgmma_commit();
                if constexpr (MICRO && PIPE) {
                    wgmma_wait<1>();
                    release(prev);                                        // t ≥ 1：上一步（含剥出的第 0 步）的槽位
                    prev = s;
                } else if constexpr (PIPE) {
                    wgmma_wait<1>();                                      // 只等上一批：它读的槽位可以还了
                    if (t > 0) release((it - 1) % STAGES);
                } else {
                    wgmma_wait<0>();
                    release(s);
                }
            }
            if constexpr (MICRO && PIPE) {
                wgmma_wait<0>();
                release(prev);
            } else if constexpr (PIPE) {
                wgmma_wait<0>();
                release((it - 1) % STAGES);
            }
            if constexpr (TSTORE) {
                uint8_t* cs = smem + STAGES * STAGE_BYTES + c * 32768;   // 这个消费者的 64×256 结果：4 个 64×64 的 box
                int lane = tid % 32, r = tid / 32 % 4 * 16 + lane / 4;
                if (tid % 128 == 0) bulk_wait_read0();                    // 上一块的 TMA 写回已经读完共享内存
                named_bar(1 + c, 128);
#pragma unroll
                for (int j = 0; j < 32; j++) {
                    uint8_t* p = cs + j / 8 * 8192 + (((j % 8) ^ (r & 7)) << 4) + lane % 4 * 4;
                    *(__nv_bfloat162*)(p + r * 128) = __floats2bfloat162_rn(d[4 * j], d[4 * j + 1]);
                    *(__nv_bfloat162*)(p + (r + 8) * 128) = __floats2bfloat162_rn(d[4 * j + 2], d[4 * j + 3]);
                }
                fence_proxy_async();                                      // 普通写对 TMA（异步代理）可见
                named_bar(1 + c, 128);
                if (tid % 128 == 0) {
                    for (int q = 0; q < 4; q++) tma_store(&tmC, cs + q * 8192, bn + q * 64, bm + c * 64);
                    bulk_commit();
                }
            } else {
                store_acc<256>(d, C, N, bm + c * 64, bn);
            }
        }
        if constexpr (TSTORE) if (tid % 128 == 0) bulk_wait0();
    }
    if constexpr (CL > 1) cluster_sync();                                 // 别让一边先退出：对方还可能往这边的 barrier 报到
}

// ---------------------------------------------------------------- 主机端
__global__ void fill_rand(bf16* x, size_t n, uint32_t seed) {          // 均匀分布 [-0.5, 0.5)
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 16; h *= 0x7feb352d; h ^= h >> 15; h *= 0x846ca68b; h ^= h >> 16;
    x[i] = __float2bfloat16((h >> 8) * (1.0f / 16777216.0f) - 0.5f);
}
// 对答案：out[0] = max|c - r|，out[1] = max|r|（非负 float 位模式按 uint 比较大小顺序不变）；NaN 记成 inf
__global__ void max_err(const bf16* c, const bf16* r, size_t n, unsigned* out) {
    float e = 0, s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float cv = __bfloat162float(c[i]), rv = __bfloat162float(r[i]);
        float dd = fabsf(cv - rv);
        e = isnan(dd) ? INFINITY : fmaxf(e, dd);
        s = fmaxf(s, fabsf(rv));
    }
    for (int o = 16; o; o >>= 1) { e = fmaxf(e, __shfl_xor_sync(~0u, e, o)); s = fmaxf(s, __shfl_xor_sync(~0u, s, o)); }
    if (threadIdx.x % 32 == 0) { atomicMax(&out[0], __float_as_uint(e)); atomicMax(&out[1], __float_as_uint(s)); }
}

static cublasHandle_t handle;
static bf16 *dA, *dW, *dC, *dRef, *dNN;
template <int S> struct WgState { static inline CUtensorMap a, w; };
static decltype(&cuTensorMapEncodeTiled) encodeTiled;

// TMA 描述符：二维张量 {K, rows}，box = {64, boxRows}（64 个 bf16 = 128 B，正好一个换位行）
static CUtensorMap make_map(const bf16* p, int rows, int K, int boxRows) {
    CUtensorMap m;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)boxRows}, es[2] = {1, 1};
    CUresult r = encodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed: %d\n", (int)r); exit(1); }
    return m;
}

static void cublas_tn(bf16* out, int N) {   // 列主序看：Cᵀ(N×M) = W(N×K) · Aᵀ(K×M)，W 按列主序是 K×N，所以取转置
    const float one = 1, zero = 0;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &one, dW, CUDA_R_16BF, N, dA, CUDA_R_16BF, N, &zero, out, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
static void launch_cublas(int N) { cublas_tn(dC, N); }
static void launch_cublas_nn(int N) {   // 参考：和 matmul.cu 里 bf16 同一种调用（把 W 当作 K×N 行主序的 B）
    const float one = 1, zero = 0;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dW, CUDA_R_16BF, N, dA, CUDA_R_16BF, N, &zero, dNN, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
static void launch_wmma(int N) { mm_wmma<<<dim3(N / 128, N / 128), 256>>>(dA, dW, dC, N, N, N); }
template <int BM, int BN, int S>
static void prep_mma(int) { cudaFuncSetAttribute(mm_mma<BM, BN, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * (BM + BN) * 128); }
template <int BM, int BN, int S>
static void launch_mma(int N) { mm_mma<BM, BN, S><<<dim3(N / BN, N / BM), 256, S * (BM + BN) * 128>>>(dA, dW, dC, N, N, N); }
template <int S>
static void prep_wgmma(int N) {
    WgState<S>::a = make_map(dA, N, N, 128);
    WgState<S>::w = make_map(dW, N, N, 128);
    cudaFuncSetAttribute(mm_wgmma<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * 256 * 128 + 1024);
}
template <int S>
static void launch_wgmma(int N) { mm_wgmma<S><<<dim3(N / 128, N / 128), 128, S * 256 * 128 + 1024>>>(WgState<S>::a, WgState<S>::w, dC, N, N, N); }
static int numSMs;
// 每个 kernel 配置各存一份 TMA 描述符和常驻 grid，轮流计时时互不覆盖
template <int S, int G, bool P, int CL, bool TS, bool H = false, bool MI = false>
struct WsState { static inline CUtensorMap a, w, c; static inline int grid; };
template <int S, bool TS>
static constexpr int ws_smem() { return S * 384 * 128 + (TS ? 65536 : 0) + 1024; }
template <int S, int CL, bool TS>
static cudaLaunchConfig_t ws_cfg(int grid, cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = ws_smem<S, TS>();
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = CL; attr->val.clusterDim.y = 1; attr->val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}
template <int S, int G, bool P, int CL, bool TS = false, bool H = false, bool MI = false>
static void prep_wgmma_ws(int N) {
    auto k = mm_wgmma_ws<S, G, P, CL, TS, H, MI>;
    using St = WsState<S, G, P, CL, TS, H, MI>;
    St::a = make_map(dA, N, N, 128);
    St::w = make_map(dW, N, N, 256 / CL);
    St::c = make_map(dC, N, N, 64);
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, ws_smem<S, TS>());
    cudaLaunchAttribute attr;
    cudaLaunchConfig_t cfg = ws_cfg<S, CL, TS>(numSMs, &attr);
    int nc = 0;
    cudaOccupancyMaxActiveClusters(&nc, k, &cfg);
    St::grid = nc * CL;
}
template <int S, int G, bool P, int CL, bool PERSIST, bool TS = false, bool H = false, bool MI = false>
static void launch_wgmma_ws(int N) {
    int tiles = (N / 128) * (N / 256);
    using St = WsState<S, G, P, CL, TS, H, MI>;
    cudaLaunchAttribute attr;
    cudaLaunchConfig_t cfg = ws_cfg<S, CL, TS>(PERSIST ? std::min(tiles, St::grid) : tiles, &attr);
    cudaLaunchKernelEx(&cfg, mm_wgmma_ws<S, G, P, CL, TS, H, MI>, St::a, St::w, St::c, dC, N, N, N);
}

struct Kernel { const char* name; void (*f)(int); void (*prep)(int); int minN; };
// 阶梯上的版本（all 时按顺序都跑）；后面带后缀的是调参时试过的候选，只在点名时跑
static const Kernel KS[] = {
    {"wmma", launch_wmma, nullptr, 128},
    {"mma", launch_mma<128, 128, 1>, prep_mma<128, 128, 1>, 128},
    {"mma_async", launch_mma<128, 128, 3>, prep_mma<128, 128, 3>, 128},
    {"wgmma", launch_wgmma<4>, prep_wgmma<4>, 128},
    {"wgmma_ws", launch_wgmma_ws<4, 0, false, 1, false>, prep_wgmma_ws<4, 0, false, 1>, 256},
    {"wgmma_l2", launch_wgmma_ws<4, 16, false, 1, false>, prep_wgmma_ws<4, 16, false, 1>, 256},
    {"wgmma_pipe", launch_wgmma_ws<4, 16, true, 1, false>, prep_wgmma_ws<4, 16, true, 1>, 256},
    {"wgmma_persist", launch_wgmma_ws<4, 16, true, 1, true>, prep_wgmma_ws<4, 16, true, 1>, 256},
    {"wgmma_cluster", launch_wgmma_ws<4, 16, true, 2, true>, prep_wgmma_ws<4, 16, true, 2>, 256},
    {"wgmma_tstore", launch_wgmma_ws<3, 16, true, 2, true, true>, prep_wgmma_ws<3, 16, true, 2, true>, 256},
    // ---- 调参候选
    {"mma_async_s2", launch_mma<128, 128, 2>, prep_mma<128, 128, 2>, 128},
    {"mma_async_s4", launch_mma<128, 128, 4>, prep_mma<128, 128, 4>, 128},
    {"mma_async_256_s3", launch_mma<128, 256, 3>, prep_mma<128, 256, 3>, 256},
    {"mma_async_256_s4", launch_mma<128, 256, 4>, prep_mma<128, 256, 4>, 256},
    {"wgmma_l2_g8", launch_wgmma_ws<4, 8, false, 1, false>, prep_wgmma_ws<4, 8, false, 1>, 256},
    {"wgmma_cluster_np", launch_wgmma_ws<4, 16, true, 2, false>, prep_wgmma_ws<4, 16, true, 2>, 256},
    {"wgmma_cluster_g8", launch_wgmma_ws<4, 8, true, 2, true>, prep_wgmma_ws<4, 8, true, 2>, 256},
    {"wgmma_cluster_s3", launch_wgmma_ws<3, 16, true, 2, true>, prep_wgmma_ws<3, 16, true, 2>, 256},
    {"wgmma_tstore_g8", launch_wgmma_ws<3, 8, true, 2, true, true>, prep_wgmma_ws<3, 8, true, 2, true>, 256},
    {"wgmma_tstore_c1", launch_wgmma_ws<3, 16, true, 1, true, true>, prep_wgmma_ws<3, 16, true, 1, true>, 256},
    // ---- 在 wgmma_tstore 上的微调（2026-10-22）
    {"wgmma_hilbert", launch_wgmma_ws<3, 16, true, 2, true, true, true, false>, prep_wgmma_ws<3, 16, true, 2, true, true, false>, 256},
    {"wgmma_micro", launch_wgmma_ws<3, 16, true, 2, true, true, false, true>, prep_wgmma_ws<3, 16, true, 2, true, false, true>, 256},
    {"wgmma_best", launch_wgmma_ws<3, 16, true, 2, true, true, true, true>, prep_wgmma_ws<3, 16, true, 2, true, true, true>, 256},
    {"wgmma_best_c1", launch_wgmma_ws<3, 16, true, 1, true, true, true, true>, prep_wgmma_ws<3, 16, true, 1, true, true, true>, 256},
    {"wgmma_hilbert_c1", launch_wgmma_ws<3, 16, true, 1, true, true, true, false>, prep_wgmma_ws<3, 16, true, 1, true, true, false>, 256},
    {"wgmma_micro_c1", launch_wgmma_ws<3, 16, true, 1, true, true, false, true>, prep_wgmma_ws<3, 16, true, 1, true, false, true>, 256},
    {"wgmma_micro_g8", launch_wgmma_ws<3, 8, true, 2, true, true, false, true>, prep_wgmma_ws<3, 8, true, 2, true, false, true>, 256},
    {"wgmma_micro_g32", launch_wgmma_ws<3, 32, true, 2, true, true, false, true>, prep_wgmma_ws<3, 32, true, 2, true, false, true>, 256},
};
static const int LADDER = 10;                                           // KS 前 10 个是阶梯

static std::vector<std::string> split(const char* s) {
    std::vector<std::string> v;
    std::string cur;
    for (const char* p = s;; p++) {
        if (*p == ',' || *p == 0) { if (!cur.empty()) v.push_back(cur); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return v;
}
static bool want(const std::vector<std::string>& sel, const std::string& n) { return sel.empty() || std::find(sel.begin(), sel.end(), n) != sel.end(); }

// 连续跑 2 s：每批 5 次 launch，GPU 上始终压着 1–2 批；每批之间用 NVML 采一次 SM 频率和板卡功耗
static void soak(const Kernel& k, int N) {
    if (k.prep) k.prep(N);
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);
    for (int i = 0; i < 3; i++) k.f(N);
    cudaDeviceSynchronize();
    cudaEvent_t ev[2], t0, t1;
    for (auto& e : ev) cudaEventCreate(&e);
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    double sc = 0, sp = 0; int ns = 0, launches = 0, b = 0;
    auto start = std::chrono::steady_clock::now();
    cudaEventRecord(t0);
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 2.0) {
        for (int i = 0; i < 5; i++) k.f(N);
        launches += 5;
        cudaEventRecord(ev[b % 2]);
        if (b > 0) cudaEventSynchronize(ev[(b + 1) % 2]);
        b++;
        unsigned clk, mw;
        if (b > 4 && nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clk) == NVML_SUCCESS && nvmlDeviceGetPowerUsage(dev, &mw) == NVML_SUCCESS) { sc += clk; sp += mw / 1000.0; ns++; }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    double per = ms / launches;
    printf("%-18s %6d  %4d 次  平均 %.3f ms  %7.2f TFLOPS  SM %6.0f MHz  功耗 %5.0f W  (%d 个采样)\n", k.name, N, launches, per,
           2.0 * N * N * N / (per * 1e-3) / 1e12, sc / ns, sp / ns, ns);
}

int main(int argc, char** argv) {
    bool soakMode = argc > 1 && std::string(argv[1]) == "soak";
    if (soakMode) { argv++; argc--; nvmlInit(); }
    std::vector<std::string> sel;
    if (argc > 1 && std::string(argv[1]) != "all") sel = split(argv[1]);
    std::vector<int> Ns;
    if (argc > 2) for (auto& s : split(argv[2])) Ns.push_back(atoi(s.c_str()));
    else for (int n = 256; n <= 8192; n *= 2) Ns.push_back(n);
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&encodeTiled, cudaEnableDefault, &q);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cublasCreate(&handle);
    int maxN = *std::max_element(Ns.begin(), Ns.end());
    size_t nn = (size_t)maxN * maxN;
    unsigned* dErr;
    cudaMalloc(&dA, nn * 2); cudaMalloc(&dW, nn * 2); cudaMalloc(&dC, nn * 2); cudaMalloc(&dRef, nn * 2); cudaMalloc(&dNN, nn * 2);
    cudaMalloc(&dErr, 8);
    fill_rand<<<(nn + 255) / 256, 256>>>(dA, nn, 12345u);
    fill_rand<<<(nn + 255) / 256, 256>>>(dW, nn, 67890u);
    if (soakMode) {
        static const Kernel CB[] = {{"cublas_bf16", launch_cublas, nullptr, 0}};
        for (int N : Ns) {
            if (want(sel, "cublas")) soak(CB[0], N);
            for (auto& k : KS) if (want(sel, k.name) && !(sel.empty() && &k - KS >= LADDER)) soak(k, N);
        }
        return 0;
    }
    printf("# 计时：每个 N 先逐个对答案，再轮流计时（每轮每个 kernel 各跑一次，取中位数），免得排在前面的 kernel 赶上 GPU 还没降频\n");
    printf("%-18s %6s %10s %9s %12s\n", "kernel", "N", "ms", "TFLOPS", "max_rel_err");
    static const Kernel CB = {"cublas_bf16", launch_cublas, nullptr, 0}, CBNN = {"cublas_bf16_nn", launch_cublas_nn, nullptr, 0};
    for (int N : Ns) {
        double flops = 2.0 * N * N * N;
        int reps = N >= 2048 ? 20 : 50;
        cublas_tn(dRef, N);                                         // 标准答案：cuBLAS BF16（同一 TN 布局）
        cudaDeviceSynchronize();
        std::vector<const Kernel*> run;
        std::vector<float> err;
        if (want(sel, "cublas")) { run.push_back(&CB); err.push_back(-1); }
        if (!sel.empty() && want(sel, "cublas_nn")) { run.push_back(&CBNN); err.push_back(-1); }   // 只在点名时跑
        for (auto& k : KS) {
            if (!want(sel, k.name) || N < k.minN || (sel.empty() && &k - KS >= LADDER)) continue;
            if (k.prep) k.prep(N);
            cudaMemset(dC, 0xFF, nn * 2);                           // 先填 NaN，漏写会被抓到
            k.f(N);
            cudaError_t ce = cudaDeviceSynchronize();
            if (ce) { printf("%-18s %6d  %s\n", k.name, N, cudaGetErrorString(ce)); return 1; }
            cudaMemset(dErr, 0, 8);
            max_err<<<528, 256>>>(dC, dRef, (size_t)N * N, dErr);
            unsigned e[2];
            cudaMemcpy(e, dErr, 8, cudaMemcpyDeviceToHost);
            float ef, sf;
            memcpy(&ef, &e[0], 4); memcpy(&sf, &e[1], 4);
            run.push_back(&k);
            err.push_back(ef / sf);
        }
        std::vector<std::vector<float>> ms(run.size());
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        for (int r = -1; r < reps; r++)                             // r = -1 是热身
            for (size_t i = 0; i < run.size(); i++) {
                cudaEventRecord(t0);
                run[i]->f(N);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                float m;
                cudaEventElapsedTime(&m, t0, t1);
                if (r >= 0) ms[i].push_back(m);
            }
        for (size_t i = 0; i < run.size(); i++) {
            std::sort(ms[i].begin(), ms[i].end());
            float m = ms[i][ms[i].size() / 2];
            if (err[i] < 0) printf("%-18s %6d %10.3f %9.2f %12s\n", run[i]->name, N, m, flops / (m * 1e-3) / 1e12, "-");
            else printf("%-18s %6d %10.3f %9.2f %12.2e\n", run[i]->name, N, m, flops / (m * 1e-3) / 1e12, err[i]);
        }
    }
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
