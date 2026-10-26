# matmul-ladder-tc-hf 配音稿

每幕一个 `## 幕 id`，每行一句（即一条字幕）。起点见 SPEC.md 的 SHOTS。

## s01
上一篇的 FP32 版本止于 44.3 TFLOPS，FP32 单元已经用满。
Tensor Core 的一条指令完成一个小矩阵块的乘加，而不是一次标量乘加。
每个 SM 每个时钟完成 2048 次乘加，是 FP32 单元的 16 倍。

## s02
本文计算 C 等于 A 乘 W 转置，A 与 W 都沿 K 连续，即 Linear 层的布局。
输入为 BF16，累加用 FP32；基线是同样布局的 cuBLAS，结果逐位相同。
峰值由硬件固定，规格为 989 TFLOPS；持续满载受功耗限制，约为 735。
因此每一步改动都在数据的通路上：让数据更早、更快地送到 Tensor Core。
本文从 93.3 一步步提高到 777.6 TFLOPS，为 cuBLAS 的 99.5%。

## s03
第一步改用 wmma，即 warp matrix multiply-accumulate：一个 warp 共同计算 16 × 16 × 16 的乘加。
一个 block 计算 C 中 128 × 128 的一块，8 个 warp 各算 64 × 32。
每个 warp 持有 4 × 2 个片段，沿 K 每 16 列发出 8 条 mma_sync。
实测 93.3 TFLOPS，是 FP32 版本的 2.1 倍，只有 cuBLAS 的 11.9%。
数据先读入寄存器，再写入共享内存，计算前又读回寄存器，两次经过寄存器。
共享内存每行 64 字节，读片段时 8 行的同一块只落在 2 组 bank，须分 4 轮。
每一块还要全体同步后才能计算；下一步先解决 bank 冲突与寄存器中转。

## 读法
mma_sync → M M A sync
multiply-accumulate → multiply accumulate
cuBLAS → Q-blas
wmma → W M M A
