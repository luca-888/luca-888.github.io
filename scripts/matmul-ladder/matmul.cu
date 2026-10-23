// 1.3 测算力：FP32 矩阵乘 C = A × B（N×N，行主序），手写 kernel 对照 cuBLAS。
// 编译：nvcc -O2 -arch=sm_90 -o matmul matmul.cu -lcublas
// 运行：./matmul [kernel,kernel,...|all] [N,N,...]（不带参数 = 全部 kernel、N = 256…8192）
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// 朴素版：每个线程算 C 的一个元素。col 用 threadIdx.x —— 同一 warp 的 32 个线程 col 连续，
// 读 B[k][col]、写 C[row][col] 都是合并访存；A[row][k] 对整个 warp 是同一个地址（广播）
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float sum = 0;
    for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// 对照：行列对调，row 用 threadIdx.x。同一 warp 的线程 row 连续，读 A[row][k] 相隔 N 个 float，不合并
__global__ void matmul_naive_T(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float sum = 0;
    for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// 共享内存分块：每个 block 算 C 的一个 T×T 小块。沿 k 方向每次把 A、B 各一个 T×T 小块搬进共享内存，
// block 里 T² 个线程共用；每个从显存读进来的 float 在共享内存里被用 T 次
template <int T>
__global__ void matmul_smem(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T], Bs[T][T];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float sum = 0;
    for (int k0 = 0; k0 < N; k0 += T) {
        As[ty][tx] = A[row * N + k0 + tx];      // 每个线程搬一个元素，按行合并读
        Bs[ty][tx] = B[(k0 + ty) * N + col];
        __syncthreads();                        // 等整块搬完再算
        for (int k = 0; k < T; k++) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();                        // 等大家都算完再覆盖
    }
    C[row * N + col] = sum;                     // 假设 N 是 T 的整数倍
}

// 寄存器分块：block 算 C 的 BM×BN，每个线程算其中 TM×TN。每走一步 k，从共享内存读 TM 个 A、TN 个 B 进寄存器，
// 做 TM×TN 次 FMA —— 每读 1 个数做 TM·TN/(TM+TN) = 4 次 FMA（smem 版是每读 2 个数做 1 次）
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) matmul_reg(const float* A, const float* B, float* C, int N) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK], Bs[BK * BN];
    int tid = threadIdx.x;
    int tRow = tid / (BN / TN), tCol = tid % (BN / TN);           // 这个线程在 block 里负责的 TM×TN 小块
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    int aRow = tid / BK, aCol = tid % BK;                         // 搬 A 块：每轮 THREADS/BK 行
    int bRow = tid / BN, bCol = tid % BN;                         // 搬 B 块：每轮 THREADS/BN 行
    float acc[TM * TN] = {0}, rA[TM], rB[TN];
    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int r = 0; r < BM; r += THREADS / BK) As[(aRow + r) * BK + aCol] = A[(aRow + r) * N + aCol];
        for (int r = 0; r < BK; r += THREADS / BN) Bs[(bRow + r) * BN + bCol] = B[(bRow + r) * N + bCol];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) rA[i] = As[(tRow * TM + i) * BK + k];
            for (int j = 0; j < TN; j++) rB[j] = Bs[k * BN + tCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i * TN + j] += rA[i] * rB[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) C[(tRow * TM + i) * N + tCol * TN + j] = acc[i * TN + j];
}

// vec：在 reg8x8 上改两处。全局→共享内存用 float4 一次搬 4 个数；A 块在共享内存里转置成 As[k][m]，
// 这样内层读 rA（同一 k、连续 8 个 m）和 rB 一样是连续地址，都能用 float4（每步 4 条 LDS.128 代替 16 条 LDS.32）
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) matmul_vec(const float* A, const float* B, float* C, int N) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ __align__(16) float As[BK * BM], Bs[BK * BN];
    int tid = threadIdx.x;
    int tRow = tid / (BN / TN), tCol = tid % (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    int aRow = tid / (BK / 4), aCol = tid % (BK / 4) * 4;          // A 块每行 BK/4 个 float4
    int bRow = tid / (BN / 4), bCol = tid % (BN / 4) * 4;          // B 块每行 BN/4 个 float4
    float acc[TM * TN] = {0}, rA[TM], rB[TN];
    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int r = 0; r < BM; r += THREADS / (BK / 4)) {
            float4 a = *(const float4*)&A[(aRow + r) * N + aCol];
            As[(aCol + 0) * BM + aRow + r] = a.x;                  // 转置着存：As[k][m]
            As[(aCol + 1) * BM + aRow + r] = a.y;
            As[(aCol + 2) * BM + aRow + r] = a.z;
            As[(aCol + 3) * BM + aRow + r] = a.w;
        }
        for (int r = 0; r < BK; r += THREADS / (BN / 4))
            *(float4*)&Bs[(bRow + r) * BN + bCol] = *(const float4*)&B[(bRow + r) * N + bCol];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4) *(float4*)&rA[i] = *(const float4*)&As[k * BM + tRow * TM + i];
            for (int j = 0; j < TN; j += 4) *(float4*)&rB[j] = *(const float4*)&Bs[k * BN + tCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i * TN + j] += rA[i] * rB[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4)
            *(float4*)&C[(tRow * TM + i) * N + tCol * TN + j] = make_float4(acc[i * TN + j], acc[i * TN + j + 1], acc[i * TN + j + 2], acc[i * TN + j + 3]);
}

// warptile：block→warp→thread 三层分块。每个 warp 管 WM×WN 一块，warp 里 32 个线程排成 8(M)×4(N)，
// 每个线程算 (WM/32)×(WN/16) 个彼此隔开的 4×4 小块：一个 warp 读 As、Bs 时 float4 地址首尾相连，没有 bank 冲突
template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) matmul_warptile(const float* A, const float* B, float* C, int N) {
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
    constexpr int MI = WM / 32, NI = WN / 16;                      // 每个线程在 M、N 方向各有几个 4×4 小块
    constexpr int A4 = BM * BK / 4, B4 = BK * BN / 4;               // A、B 块各有多少个 float4
    static_assert(A4 % THREADS == 0 && B4 % THREADS == 0, "tile/threads");
    __shared__ __align__(16) float As[BK * BM], Bs[BK * BN];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    int lRow = lane / 4, lCol = lane % 4;
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float acc[MI * 4][NI * 4] = {}, rA[MI * 4], rB[NI * 4];
    for (int k0 = 0; k0 < N; k0 += BK) {
        float4 pa[A4 / THREADS], pb[B4 / THREADS];                  // 搬运和 vec 一样：先把 A、B 的 float4 全发出去，再统一写共享内存
        for (int p = 0; p < A4 / THREADS; p++) { int i = tid + p * THREADS; pa[p] = *(const float4*)&A[i / (BK / 4) * N + i % (BK / 4) * 4]; }
        for (int p = 0; p < B4 / THREADS; p++) { int i = tid + p * THREADS; pb[p] = *(const float4*)&B[i / (BN / 4) * N + i % (BN / 4) * 4]; }
        for (int p = 0; p < A4 / THREADS; p++) {
            int i = tid + p * THREADS, r = i / (BK / 4), c = i % (BK / 4) * 4;
            As[(c + 0) * BM + r] = pa[p].x;
            As[(c + 1) * BM + r] = pa[p].y;
            As[(c + 2) * BM + r] = pa[p].z;
            As[(c + 3) * BM + r] = pa[p].w;
        }
        for (int p = 0; p < B4 / THREADS; p++) { int i = tid + p * THREADS; *(float4*)&Bs[i / (BN / 4) * BN + i % (BN / 4) * 4] = pb[p]; }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int mi = 0; mi < MI; mi++) *(float4*)&rA[mi * 4] = *(const float4*)&As[k * BM + wRow * WM + mi * 32 + lRow * 4];
            for (int ni = 0; ni < NI; ni++) *(float4*)&rB[ni * 4] = *(const float4*)&Bs[k * BN + wCol * WN + ni * 16 + lCol * 4];
            for (int i = 0; i < MI * 4; i++)
                for (int j = 0; j < NI * 4; j++) acc[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }
    for (int mi = 0; mi < MI; mi++)
        for (int i = 0; i < 4; i++)
            for (int ni = 0; ni < NI; ni++) {
                float* a = &acc[mi * 4 + i][ni * 4];
                *(float4*)&C[(wRow * WM + mi * 32 + lRow * 4 + i) * N + wCol * WN + ni * 16 + lCol * 4] = make_float4(a[0], a[1], a[2], a[3]);
            }
}

// dbuf：在 warptile 上加双缓冲。共享内存开两份，算第 t 块的同时，下一块已经从显存读进寄存器（发出就走，不等），
// 算完再写进另一份共享内存；每块只要一次 __syncthreads，读显存的延迟藏在 FMA 后面
template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) matmul_dbuf(const float* A, const float* B, float* C, int N) {
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
    constexpr int MI = WM / 32, NI = WN / 16;
    constexpr int A4 = BM * BK / 4, B4 = BK * BN / 4, AI = A4 / THREADS, BI = B4 / THREADS;
    static_assert(A4 % THREADS == 0 && B4 % THREADS == 0, "tile/threads");
    __shared__ __align__(16) float As[2][BK * BM], Bs[2][BK * BN];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int wRow = warp / (BN / WN), wCol = warp % (BN / WN);
    int lRow = lane / 4, lCol = lane % 4;
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float4 pa[AI], pb[BI];                                         // 预取到寄存器的下一块
    auto fetch = [&](int k0) {
        for (int p = 0; p < AI; p++) { int i = tid + p * THREADS, r = i / (BK / 4), c = i % (BK / 4) * 4; pa[p] = *(const float4*)&A[r * N + k0 + c]; }
        for (int p = 0; p < BI; p++) { int i = tid + p * THREADS, r = i / (BN / 4), c = i % (BN / 4) * 4; pb[p] = *(const float4*)&B[(k0 + r) * N + c]; }
    };
    auto stash = [&](int buf) {
        for (int p = 0; p < AI; p++) {
            int i = tid + p * THREADS, r = i / (BK / 4), c = i % (BK / 4) * 4;
            As[buf][(c + 0) * BM + r] = pa[p].x;
            As[buf][(c + 1) * BM + r] = pa[p].y;
            As[buf][(c + 2) * BM + r] = pa[p].z;
            As[buf][(c + 3) * BM + r] = pa[p].w;
        }
        for (int p = 0; p < BI; p++) { int i = tid + p * THREADS, r = i / (BN / 4), c = i % (BN / 4) * 4; *(float4*)&Bs[buf][r * BN + c] = pb[p]; }
    };
    float acc[MI * 4][NI * 4] = {}, rA[MI * 4], rB[NI * 4];
    fetch(0);
    stash(0);
    __syncthreads();
    int T = N / BK;
    for (int t = 0; t < T; t++) {
        int cur = t & 1;
        if (t + 1 < T) fetch((t + 1) * BK);                       // 先发出下一块的读，结果等到 stash 才用
        for (int k = 0; k < BK; k++) {
            for (int mi = 0; mi < MI; mi++) *(float4*)&rA[mi * 4] = *(const float4*)&As[cur][k * BM + wRow * WM + mi * 32 + lRow * 4];
            for (int ni = 0; ni < NI; ni++) *(float4*)&rB[ni * 4] = *(const float4*)&Bs[cur][k * BN + wCol * WN + ni * 16 + lCol * 4];
            for (int i = 0; i < MI * 4; i++)
                for (int j = 0; j < NI * 4; j++) acc[i][j] += rA[i] * rB[j];
        }
        if (t + 1 < T) stash(cur ^ 1);                             // 另一份缓冲没人在读（上一轮末尾已同步过）
        __syncthreads();
    }
    for (int mi = 0; mi < MI; mi++)
        for (int i = 0; i < 4; i++)
            for (int ni = 0; ni < NI; ni++) {
                float* a = &acc[mi * 4 + i][ni * 4];
                *(float4*)&C[(wRow * WM + mi * 32 + lRow * 4 + i) * N + wCol * WN + ni * 16 + lCol * 4] = make_float4(a[0], a[1], a[2], a[3]);
            }
}

// 对答案：out[0] = max|c - r|，out[1] = max|r|。非负 float 的位模式按无符号整数比较时大小顺序不变，所以能直接 atomicMax；NaN 记成 inf
__global__ void max_err(const float* c, const float* r, size_t n, unsigned* out) {
    float e = 0, s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float d = fabsf(c[i] - r[i]);
        e = isnan(d) ? INFINITY : fmaxf(e, d);
        s = fmaxf(s, fabsf(r[i]));
        if (isinf(e)) break;
    }
    for (int o = 16; o; o >>= 1) { e = fmaxf(e, __shfl_xor_sync(~0u, e, o)); s = fmaxf(s, __shfl_xor_sync(~0u, s, o)); }
    if (threadIdx.x % 32 == 0) { atomicMax(&out[0], __float_as_uint(e)); atomicMax(&out[1], __float_as_uint(s)); }
}

__global__ void to_bf16(const float* x, __nv_bfloat16* y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

static float bench(void (*launch)(const float*, const float*, float*, int), const float* A, const float* B, float* C, int N, int reps) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    launch(A, B, C, N);                                   // 热身一次
    std::vector<float> ms;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(t0);
        launch(A, B, C, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float m;
        cudaEventElapsedTime(&m, t0, t1);
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

static cublasHandle_t handle;
static void launch_naive(const float* A, const float* B, float* C, int N) {
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    matmul_naive<<<grid, block>>>(A, B, C, N);
}
static void launch_naive_T(const float* A, const float* B, float* C, int N) {
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    matmul_naive_T<<<grid, block>>>(A, B, C, N);
}
template <int T>
static void launch_smem(const float* A, const float* B, float* C, int N) {
    dim3 block(T, T), grid(N / T, N / T);
    matmul_smem<T><<<grid, block>>>(A, B, C, N);
}
static void launch_reg(const float* A, const float* B, float* C, int N) {
    dim3 grid(N / 128, N / 128);
    matmul_reg<128, 128, 8, 8, 8><<<grid, 256>>>(A, B, C, N);
}
static __nv_bfloat16 *Ah, *Bh, *Ch;
static void launch_tf32(const float* A, const float* B, float* C, int N) {     // 同一个 SGEMM，允许用 TF32 Tensor Core
    const float one = 1, zero = 0;
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B, N, A, N, &zero, C, N);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
static void launch_bf16(const float*, const float*, float*, int N) {           // BF16 输入、FP32 累加，走 BF16 Tensor Core
    const float one = 1, zero = 0;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, Bh, CUDA_R_16BF, N, Ah, CUDA_R_16BF, N, &zero, Ch, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
static void launch_cublas(const float* A, const float* B, float* C, int N) {
    // cuBLAS 是列主序：行主序的 C = A×B 等价于列主序的 Cᵀ = Bᵀ×Aᵀ，所以把 B、A 对调传入
    const float one = 1, zero = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B, N, A, N, &zero, C, N);
}

template <int BM, int BN, int BK, int TM, int TN>
static void launch_vec(const float* A, const float* B, float* C, int N) {
    matmul_vec<BM, BN, BK, TM, TN><<<dim3(N / BN, N / BM), (BM / TM) * (BN / TN)>>>(A, B, C, N);
}
template <int BM, int BN, int BK, int WM, int WN>
static void launch_warptile(const float* A, const float* B, float* C, int N) {
    matmul_warptile<BM, BN, BK, WM, WN><<<dim3(N / BN, N / BM), (BM / WM) * (BN / WN) * 32>>>(A, B, C, N);
}
template <int BM, int BN, int BK, int WM, int WN>
static void launch_dbuf(const float* A, const float* B, float* C, int N) {
    matmul_dbuf<BM, BN, BK, WM, WN><<<dim3(N / BN, N / BM), (BM / WM) * (BN / WN) * 32>>>(A, B, C, N);
}
struct Kernel { const char* name; void (*f)(const float*, const float*, float*, int); int minN; };
// autotune 候选：dbuf 的 <BM, BN, BK, WM, WN>，名字里写全参数
static const Kernel TUNE[] = {
    {"dbuf_128x128x8_w64x32", launch_dbuf<128, 128, 8, 64, 32>, 128},
    {"dbuf_128x128x16_w64x32", launch_dbuf<128, 128, 16, 64, 32>, 128},
    {"dbuf_128x256x8_w64x64", launch_dbuf<128, 256, 8, 64, 64>, 256},
    {"dbuf_256x128x8_w64x64", launch_dbuf<256, 128, 8, 64, 64>, 256},
    {"dbuf_128x128x8_w64x64", launch_dbuf<128, 128, 8, 64, 64>, 128},
    {"dbuf_128x128x16_w64x64", launch_dbuf<128, 128, 16, 64, 64>, 128},
    {"dbuf_64x128x8_w32x64", launch_dbuf<64, 128, 8, 32, 64>, 128},
    {"dbuf_128x64x8_w64x32", launch_dbuf<128, 64, 8, 64, 32>, 128},
    {"dbuf_64x64x8_w32x32", launch_dbuf<64, 64, 8, 32, 32>, 64},
};

static bool want(const std::vector<std::string>& sel, const std::string& name) {
    return sel.empty() || std::find(sel.begin(), sel.end(), name) != sel.end();
}
static std::vector<std::string> split(const char* s) {
    std::vector<std::string> v;
    std::string cur;
    for (const char* p = s; ; p++) {
        if (*p == ',' || *p == 0) { if (!cur.empty()) v.push_back(cur); cur.clear(); if (!*p) break; }
        else cur += *p;
    }
    return v;
}

// 用法：./matmul [kernel,kernel,...|all] [N,N,...]；tuned 表示跑完 TUNE 里所有配置、每个 N 取最快的
int main(int argc, char** argv) {
    std::vector<std::string> sel;
    if (argc > 1 && std::string(argv[1]) != "all") sel = split(argv[1]);
    std::vector<int> Ns;
    if (argc > 2) for (auto& s : split(argv[2])) Ns.push_back(atoi(s.c_str()));
    else for (int n = 256; n <= 8192; n *= 2) Ns.push_back(n);
    cublasCreate(&handle);                                // 默认数学模式：纯 FP32，不用 TF32
    const int maxN = *std::max_element(Ns.begin(), Ns.end());
    float *A, *B, *C, *Cref;
    unsigned* dErr;
    size_t bytes = sizeof(float) * maxN * maxN;
    cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes); cudaMalloc(&Cref, bytes);
    cudaMalloc(&dErr, 2 * sizeof(unsigned));
    std::vector<float> h(size_t(maxN) * maxN);
    for (size_t i = 0; i < h.size(); i++) h[i] = float(rand()) / RAND_MAX - 0.5f;
    cudaMemcpy(A, h.data(), bytes, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < h.size(); i++) h[i] = float(rand()) / RAND_MAX - 0.5f;
    cudaMemcpy(B, h.data(), bytes, cudaMemcpyHostToDevice);

    size_t nn = size_t(maxN) * maxN;
    cudaMalloc(&Ah, nn * 2); cudaMalloc(&Bh, nn * 2); cudaMalloc(&Ch, nn * 2);
    to_bf16<<<(nn + 255) / 256, 256>>>(A, Ah, nn);
    to_bf16<<<(nn + 255) / 256, 256>>>(B, Bh, nn);
    printf("%-24s %6s %10s %9s %12s\n", "kernel", "N", "ms", "TFLOPS", "max_rel_err");
    const Kernel ks[] = {{"naive", launch_naive, 16}, {"naive_T", launch_naive_T, 16}, {"smem16", launch_smem<16>, 16},
                         {"smem32", launch_smem<32>, 32}, {"reg8x8", launch_reg, 128},
                         {"vec", launch_vec<128, 128, 8, 8, 8>, 128},
                         {"warptile", launch_warptile<128, 128, 8, 64, 32>, 128},
                         {"dbuf", launch_dbuf<128, 128, 8, 64, 32>, 128}};
    for (int N : Ns) {
        double flops = 2.0 * N * N * N;
        auto tf = [&](float ms) { return flops / (ms * 1e-3) / 1e12; };
        int reps = N >= 4096 ? 5 : 20;
        float mr = bench(launch_cublas, A, B, Cref, N, reps);
        if (want(sel, "cublas")) printf("%-24s %6d %10.3f %9.2f %12s\n", "cublas", N, mr, tf(mr), "-");
        if (want(sel, "tf32")) { float m = bench(launch_tf32, A, B, C, N, reps); printf("%-24s %6d %10.3f %9.2f %12s\n", "tf32", N, m, tf(m), "-"); }
        if (want(sel, "bf16")) { float m = bench(launch_bf16, A, B, C, N, reps); printf("%-24s %6d %10.3f %9.2f %12s\n", "bf16", N, m, tf(m), "-"); }
        auto run = [&](const Kernel& k) -> float {
            cudaMemset(C, 0xFF, sizeof(float) * N * N);                   // 先填 NaN，kernel 漏写会被对答案抓到
            float ms = bench(k.f, A, B, C, N, k.f == launch_naive_T || k.f == launch_naive ? 3 : reps);
            cudaMemset(dErr, 0, 2 * sizeof(unsigned));
            max_err<<<528, 256>>>(C, Cref, size_t(N) * N, dErr);            // 和 cuBLAS FP32 对答案
            unsigned e[2];
            cudaMemcpy(e, dErr, sizeof(e), cudaMemcpyDeviceToHost);
            float ef, sf;
            memcpy(&ef, &e[0], 4); memcpy(&sf, &e[1], 4);
            printf("%-24s %6d %10.3f %9.2f %12.2e\n", k.name, N, ms, tf(ms), ef / sf);
            return ms;
        };
        for (auto& k : ks) {
            if (!want(sel, k.name) || N < k.minN) continue;
            if (k.f == launch_naive_T && N > 4096) continue;        // 不合并版太慢，8192 不测
            run(k);
        }
        if (want(sel, "tuned")) {
            float best = 1e30f;
            const char* bn = "";
            for (auto& k : TUNE) {
                if (N < k.minN) continue;
                float ms = run(k);
                if (ms < best) { best = ms; bn = k.name; }
            }
            printf("%-24s %6d %10.3f %9.2f %12s  (= %s)\n", "tuned", N, best, tf(best), "-", bn);
        }
    }
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
