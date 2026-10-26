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

## s04
为了消除 bank 冲突，第二步换位存放数据，每行加宽到 128 字节。
ldmatrix 即 load matrix，一次读取 8 行的同一块；不换位时，8 个地址落在同一组 bank，须分 8 轮。
换位存放把第 r 行的第 c 块移到 c 与 r 异或的位置，8 个地址分散开，一轮读完。
搬运改用 cp.async，即异步拷贝，数据由 L2 直接写入共享内存；计算改用 mma.sync 指令。
实测 269 TFLOPS，是 wmma 的 2.9 倍。
mma 版本的问题是：每一块须等待拷贝完成后才能计算。

## s05
第三步把共享内存分为三个槽位，拷贝提前两块发出。
计算第 t 块时，后两块已在途；每个槽位 32 KB，共 96 KB。
实测 309 TFLOPS，提高 15%。
这一版的问题在指令本身：一条 mma.sync 只有 2048 次乘加，满载需每 SM 每个时钟发出一条。
mma.sync 的各版都止于约 300 TFLOPS，只有峰值的三分之一。

## s06
第四步把执行单位由 warp 升到 warpgroup，即 4 个 warp。
TMA 即 Tensor Memory Accelerator：一个线程发出一条指令，把一整块矩阵搬入共享内存，并完成换位。
mbarrier 是共享内存中的计数器，记录到达的字节数；计满后，计算方开始读取。
一条 wgmma 完成 131072 次乘加，是 mma.sync 的 64 倍，并直接从共享内存读取操作数。
实测 484 TFLOPS，是三级缓冲的 1.6 倍。
这一版的问题是：同一个 warpgroup 既发拷贝又计算，两块之间 Tensor Core 空闲。

## s07
第五步把 warpgroup 分成生产者与消费者：一个生产者发 TMA，两个消费者计算。
每个槽位有两个 mbarrier：full 表示数据已到，empty 表示消费者已用完。
两个消费者交替使用 Tensor Core，一个等待时，另一个在计算。
4096 阶由 481 提高到 635 TFLOPS；8192 阶仍为 480。
原因是 L2 容量：按行发放时，同时运行的 132 个 tile 需要整个 W，共 128 MB，远超 50 MB 的 L2。

## s08
第六步只改 tile 的发放顺序：每 16 行一组，组内按列发放。
一轮 132 个 tile 需要的数据由 136 MB 减为 65 MB。
共用同一块 W 的 tile 同时运行，W 读入 L2 后随即被 16 个 tile 读取。
显存读取量由 6.6 GB 降到 1.3 GB，L2 命中率由 48% 提高到 79%。
实测 740 TFLOPS，是上一版的 1.54 倍。
这一版的问题是：每个 block 各自从 L2 读取整块 W。

## s09
第七步让相邻两行、共用同一块 W 的两个 block 组成线程块集群。
集群内的 block 可以访问彼此的共享内存；两者各搬 W 的一半，用 TMA 多播写入双方。
每个 block 每步从 L2 读取的数据由 48 KB 降到 32 KB。
集群以常驻 kernel 为前提：只启动 132 个 block，循环领取 tile。
实测 757 TFLOPS，为 cuBLAS 的 96.9%。
这一版的问题在写回：每个线程零散写显存 64 次，期间不发 wgmma。

## s10
第八步改由 TMA 写回：结果先换位写入共享内存，再由一个线程整块写出。
消费者随即开始下一块，写回与计算重叠。
代价是共享内存：结果区占 64 KB，缓冲由 4 级减为 3 级。
实测 777.6 TFLOPS，为 cuBLAS 的 99.5%。

## s11
连续运行时，手写版与 cuBLAS 都达到 700 W 的功耗上限，时钟由 1830 降到约 1360 MHz。
按这一时钟，峰值为 735 TFLOPS，两者都用到其中的 93%。
剩下的差距在能效：每次浮点运算，手写版约 1.05 pJ，cuBLAS 约 0.81 pJ。
执行单位由 warp 升到 warpgroup 再到 cluster，数据由寄存器中转变为 TMA 整块搬运。
计算单元的峰值是固定的，矩阵乘法的速度取决于数据能否及时送达。

## 读法
mma_sync → M M A sync
multiply-accumulate → multiply accumulate
cuBLAS → Q-blas
wmma → W M M A
ldmatrix → LD matrix
cp.async → C P async
mma.sync → M M A sync
wgmma → W G M M A
mbarrier → M barrier
warpgroup → warp group
TMA → T M A
