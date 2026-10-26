# matmul-ladder-tc-hf 配音稿

每幕一个 `## 幕 id`，每行一句（即一条字幕）。画面见 SPEC.md。第四版（2026-10-26）：开场、起点 wmma、三章（读顺送早 / Hopper 异步机制 / 少读晚写）、收尾；讲原理，技术术语用英文、首次出现带一句中文含义；提速只由右上角耗时条表示。

## s01
这是 H100 的芯片：132 个 SM 分在上下两半，中间是 50 MB 的 L2，显存在芯片两侧。
每个 SM 分成 4 个分区，每个分区有自己的寄存器、32 个 FP32 单元和一个 Tensor Core，4 个分区共用一块 shared memory。
FP32 单元一条指令算一个乘加；Tensor Core 一条指令算一整块小矩阵的乘加。
每个 SM 每个时钟 2048 次乘加，是 FP32 单元的 16 倍。
算得这么快，难在喂数据：数据从显存出发，经过 L2、shared memory，才送到 Tensor Core。
上一篇的 FP32 版本止于 44.3 TFLOPS；换上 Tensor Core 的第一版，也只用到峰值的 9%，其余时间 Tensor Core 都在等数据。接下来的改动，都在这条路上。

## s02
最直接的用法是 CUDA C++ 自带的 wmma 接口：一个 warp 的 32 个线程共同持有矩阵片段，即 fragment，合起来算一块 16 × 16 × 16 的乘加。这是本文的第一版。
数据先读进寄存器，再写入 shared memory，计算前又读回寄存器，两次经过寄存器。
shared memory 分成 32 个 bank，落在同一个 bank 的地址只能排队，这就是 bank conflict；上一篇用 warp 分块消掉了它。
换成 Tensor Core，冲突又回来了：它按 fragment 读取，每次竖着取 8 行的同一块；每行 64 字节时，这 8 个地址只落在 2 组 bank，一次读取要分 4 轮。

## s03
第一章先把数据读顺、送早。每行加宽到 128 字节，一行正好覆盖全部 32 个 bank。
ldmatrix 一次读 8 行的同一块；直接存放时，这 8 个地址都在同一组 bank。
swizzle 的原意是搅乱：把第 r 行的第 c 块移到 c 与 r 异或的位置，8 个地址错开到 8 组 bank，一轮读完。
搬运改用 cp.async，即异步拷贝：数据由 L2 直接写入 shared memory，不再经过寄存器。
拷贝不绕路了，再让它提前发出，这叫 pipeline，即多级流水：shared memory 分成 3 个 stage 组成一个环，计算第 t 块时，后两块已在路上。
再往下卡在 mma.sync 本身：一条只有 2048 次乘加，喂满 Tensor Core，每个 SM 每个时钟都要发一条，还要另发 ldmatrix 把操作数读进寄存器。
同时 shared memory 也供不上：按这一版的分块，每个时钟要读 192 字节，供给只有 128 字节；mma.sync 的各版实测都停在峰值的三分之一左右。

## s04
第二章换上 Hopper 的新机制，搬运和计算都以整块为单位、异步进行。
TMA 全称 Tensor Memory Accelerator，是专门搬运矩阵块的硬件：一个线程一条指令，把一整块搬入 shared memory，同时完成 swizzle。
mbarrier 是 shared memory 里的计数器，记录 TMA 到达的字节数；计满了，计算方就开始读。
wgmma 由 4 个 warp 组成的 warpgroup 一起发出，一条 131072 次乘加，是 mma.sync 的 64 倍，直接从 shared memory 读操作数。
同一组线程既搬又算，会互相等待；于是把两件事分给不同的 warpgroup，这叫 warp specialization，即按角色分工：一个 producer 只发 TMA，两个 consumer 只做计算。
每个 stage 配两个 mbarrier：full 表示数据已到，empty 表示 consumer 已用完；两个 consumer 交替使用 Tensor Core。
4096 阶因此明显变快，8192 阶却没有变化。
原因是 L2 容量：按行发放时，同时运行的 132 个 tile 要读整个 W，共 128 MB，L2 只有 50 MB，W 还没被复用就被挤了出去。

## s05
第三章减少访存，并把写回藏起来。先只改 tile 的发放顺序：每 16 行一组，组内按列发放。
按行发放时，共用同一块 W 的两个 tile 编号相差 32，后一个来读时，W 已被替换；分组后它们同时运行，一块 W 读进 L2，随即被 16 个 tile 读走。
显存读取由 6.6 GB 降到 1.3 GB。
再用 Hopper 的 thread block cluster，即线程块集群：同一 cluster 里的 block 可以访问彼此的 shared memory。
共用同一块 W 的两个 block 组成 cluster，各搬一半，用 TMA multicast，即多播，从 L2 读一次、同时写入两个 block；每个 block 每步从 L2 读的数据由 48 KB 降到 32 KB。
最后是写回：原来每个线程零散地写显存 64 次，这期间不发 wgmma。
改由 TMA store：consumer 先把结果 swizzle 后写入 shared memory，再由一个线程整块写回显存，写回与下一块的计算重叠。
代价是 shared memory：结果区占 64 KB，stage 由 4 个减为 3 个。最终一次 1.41 毫秒，为 cuBLAS 的 99.5%。

## s06
989 TFLOPS 是规格峰值。连续满载时，两者都撞上 700 瓦的功耗上限，时钟从 1830 降到约 1360 MHz。
按这个时钟，峰值只剩 735，手写版和 cuBLAS 都用到了其中的 93%。
回头看这三章：先把数据读顺、送早；再用 Hopper 的异步机制，让搬运和计算以整块为单位分头进行；最后减少访存，把写回藏进计算。
Tensor Core 的算力是固定的，矩阵乘法的快慢，取决于数据能不能及时送到。

## 读法
cuBLAS → Q-blas
wmma → W M M A
cp.async → C P async
mma.sync → M M A sync
wgmma → W G M M A
warpgroup → warp group
TMA → T M A
mbarrier → M barrier
ldmatrix → LD matrix
MHz → 兆赫
