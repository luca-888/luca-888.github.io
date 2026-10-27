// 视频对照：H100 上 mma.sync 和 wgmma 各自能跑到多快（不读显存、不读 shared memory 数据，只连续发指令）。
// 编译：nvcc -O3 -arch=sm_90a -std=c++17 -o tc_peak tc_peak.cu
// 运行：./tc_peak            输出两种指令的 TFLOPS；ncu 下 -k regex:peak_ 可看 Tensor 管线忙的比例
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint64_t make_desc(const void* p) {
    uint64_t a = smem_u32(p);
    return ((a & 0x3FFFF) >> 4) | ((uint64_t)(16 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}

template <int n> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(n) : "memory"); }
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ void wgmma_n256(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(1));
}

constexpr int ITERS = 2048;
// 每个 warp 8 组独立累加器，连续发 m16n8k16；8 个 warp × 每 SM 4 个 block，依赖链足够短，只看指令本身的吞吐
__global__ void __launch_bounds__(256) peak_mma(float* out) {
    uint32_t a[4], b[2];
    for (int i = 0; i < 4; i++) a[i] = 0x3c003c00u + threadIdx.x + i;
    for (int i = 0; i < 2; i++) b[i] = 0x3c003c00u + threadIdx.x * 3 + i;
    float acc[8][4] = {};
    for (int it = 0; it < ITERS; it++)
#pragma unroll
        for (int j = 0; j < 8; j++) mma16816(acc[j], a, b);
    float s = 0;
    for (int j = 0; j < 8; j++) for (int e = 0; e < 4; e++) s += acc[j][e];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
// 每 SM 一个 block、两个 warpgroup 交替，和 wgmma_ws 的两个消费者一样；操作数是 shared memory 里的任意数据
__global__ void __launch_bounds__(256, 1) peak_wgmma(float* out) {
    extern __shared__ uint8_t raw[];
    uint8_t* sm = (uint8_t*)(((uintptr_t)raw + 1023) & ~(uintptr_t)1023);
    for (int i = threadIdx.x; i < 49152 / 4; i += 256) ((uint32_t*)sm)[i] = 0x3c003c00u;   // 操作数填 1.0 附近的值
    __syncthreads();
    float d[128];
    for (int j = 0; j < 128; j++) d[j] = 0;
    uint8_t* sa = sm + (threadIdx.x / 128) * 8192;          // 每个 warpgroup 64 行 A
    uint8_t* sw = sm + 16384;                               // 256 行 W
    for (int it = 0; it < ITERS / 4; it++) {
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) wgmma_n256(d, make_desc(sa + kk * 32), make_desc(sw + kk * 32));
        wgmma_commit();
        wgmma_wait<0>();
    }
    float s = 0;
    for (int j = 0; j < 128; j++) s += d[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <class F>
static double run(F launch, double flop) {
    for (int i = 0; i < 3; i++) launch();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 20; i++) launch();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return flop * 20 / (ms * 1e-3) / 1e12;
}
int main() {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    float* out; cudaMalloc(&out, (size_t)sms * 4 * 256 * 4);
    int smem = 16384 + 32768 + 1024;
    cudaFuncSetAttribute(peak_wgmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    double fm = (double)sms * 4 * 8 * ITERS * 8 * 4096;          // blocks × warps × iters × 8 条 × 每条 4096 FLOP
    double fw = (double)sms * 2 * (ITERS / 4) * 4 * 524288.0;     // blocks × warpgroups × iters × 4 条 × 每条 524288 FLOP
    double tm = run([&] { peak_mma<<<sms * 4, 256>>>(out); }, fm);
    double tw = run([&] { peak_wgmma<<<sms, 256, smem>>>(out); }, fw);
    printf("peak_mma   mma.sync.m16n8k16   %.1f TFLOPS\n", tm);
    printf("peak_wgmma wgmma.m64n256k16    %.1f TFLOPS\n", tw);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
