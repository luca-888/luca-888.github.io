# 矩阵乘法阶梯笔记

系列两篇（草稿，未发布）：上篇 `content/matmul-ladder-fp32.md`（FP32 路线），下篇 `content/matmul-ladder-tc.md`（Tensor Core 路线），互相链接；同题视频 `videos/matmul-ladder/`。两篇共用的图表组件与占位映射在 `src/articles/matmul-ladder/`，测量与脚本目录仍为 `public/measurements/matmul-ladder/`、`scripts/matmul-ladder/`。2026-10-21 用户要求按视频的幕序与每幕七步（① 证据 ② 概念 ③ 代码 ④ 数据映射 ⑤ 硬件侧 ⑥ 实测 ⑦ 下一个瓶颈）直接写正文，不另交参考方案；版本划分与视频一致（vec + warptile、mma + 三级缓冲、提前归还 + 常驻各合为一节），文章比视频多完整代码片段、ncu 读数与推导。

## 测量

- 全部数字来自 inference-study 仓库 `labs/1.3-matmul-flops/`（2026-10-21，Modal H100 80GB HBM3，SXM 700 W，CUDA 12.8.1，未锁频）。原样输出复制到 `public/measurements/matmul-ladder/20261021-121144/`（`ladder-final`、`ladder.json`、两份 ncu、`ladder-dev6`），代码复制到 `scripts/matmul-ladder/`。图表数据 `src/data/matmul-ladder.json` 由 `ladder.json` 裁出。
- 计时：FP32 线顺序计时（N ≥ 4096 5 次，naive 3 次，其余 20 次）取中位数；Tensor Core 线先逐个对答案，再与 cuBLAS 轮流计时 20 轮（N < 2048 为 50 轮）取中位数。持续负载：连续 2 s，NVML 采 SM 时钟与功耗。
- ncu 共三组：2026-10-21 的 `dbuf` vs cuBLAS SGEMM、`wgmma_cluster` vs cuBLAS BF16（8192 阶，`--clock-control none`）；2026-10-22 的 `wgmma_ws` vs `wgmma_l2`（4096、8192 阶，`--set full` 加显存与 L2 指标，另有 tile 开始时间日志与连续 2 s 功耗），原样输出与逐指令采样 CSV（`ncu-ws-*-src-*.csv`）在 `public/measurements/matmul-ladder/20261022-142921/`。其余版本的“瓶颈”是推算，正文标“推算”。
- 第六节“消费者等 full barrier 的采样”取自 src CSV：消费者 full 屏障 `SYNCS.PHASECHK.TRYWAIT` 重试循环（含跳回的 `BRA`）的采样占全部采样的比例，8192 阶按行约 36.5%、分组约 4.5%；生产者等 empty 的循环另计，不在其中。
- 2026-10-22 补充实测（Hilbert、首块省清零、去掉集群、4096/8192 连续 2 s）：`labs/1.3-matmul-flops/out/micro-20261022-105500.json` 及同目录 `ladder-micro-*`、`ladder-c1split*`，原样复制到 `public/measurements/matmul-ladder/20261022-105500/`；`scripts/matmul-ladder/matmul_tc.cu` 同步为含 `HILBERT`、`MICRO` 模板参数的版本（原有 kernel 不变）。三个容器可能分到不同的卡，只比较同次运行内的比值。
- 1.1、1.2 的数：H100 拷贝带宽实测 3.0 TB/s（`labs/1.2-hbm-bandwidth/out/copy-20261020-182319.txt`，4 GiB 时 3051 GB/s）；BF16 满载持续约 674 TFLOPS、约 1350 MHz（`labs/1.1-h100-anatomy/out/clock-*`）；SM 132、smem 228 KiB、寄存器 65536、L2 50 MiB（`probe-20261020-172840.txt`）。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)，2022-12 | Simon Boehm | 已读（2026-10-21 WebFetch 摘要） | 核心（叙事结构） | 全文结构；四–八节对照 |
| 2 | [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)，2024-11-29 | Pranjal Shankhdhar | 已读（2026-10-21 WebFetch 摘要） | 核心（叙事结构、对照） | 十一–十八节；常驻 kernel 变慢、Hilbert 曲线 |
| 3 | 本站实测与代码 `labs/1.3-matmul-flops/` | 本站 | 已读 | 核心（全部数字） | 全文 |
| 4 | [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/)：`mma`、`wgmma`、`cp.async.bulk.tensor`、`mbarrier` | NVIDIA | 相关小节已读（2026-10-20，见 `tensor-core-notes.md`） | 核心 | 十–十六节 |
| 5 | [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)（页面更新于 2026-09-21） | NVIDIA | 已读（2026-10-21 WebFetch 摘要） | 高（相关系统） | 十二、十三、十八节 |
| 6 | H100 白皮书 V1.02 “H100 Tensor Core Architecture” | NVIDIA | 已读（2026-10-20，见 `tensor-core-notes.md`） | 高 | 二节（每 SM 每时钟 2048 次乘加） |
| 7 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) §3.1 | Shah et al. | 已读（2026-10-20，见 `tensor-core-notes.md`） | 中 | 二节（989 TFLOPS 对应 1830 MHz）、warp 特化 |
| 8 | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) README | DeepSeek | 已读（2026-10-21） | 中（现状变化） | 参考列表 |
| 10 | [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349)（arXiv 2404.16349） | Alman, Duan, Vassilevska Williams, Xu, Xu, Zhou | 摘要已读（2026-10-21）。v1（2024-04-25）给出 ω < 2.371339；v3（2026-08-19）摘要已改为 ω < 2.371177 | 中 | 上篇二节 |
| 11 | [Improving the matrix multiplication exponent with modern optimization and AlphaEvolve](https://arxiv.org/abs/2608.16884)（arXiv 2608.16884，2026-08-17） | Dupont, …, Alman, Vassilevska Williams, Balog | 摘要已读（2026-10-21）：ω < 2.371177，改进自 2.371339，用 AlphaEvolve 细化优化算法 | 中 | 上篇二节 |
| 12 | I/O Complexity: The Red-Blue Pebble Game，STOC 1981（CMU-CS-81-111） | Hong, Kung | 原文未读（ACM 403）；Ω(n³/√S) 的结论经二手资料核对（检索摘要、MIT 6.851 讲义） | 中 | 上篇二节、六节 |
| 13 | Gaussian Elimination is not Optimal，Numerische Mathematik 1969 | Strassen | 未读（通行结论 O(N^2.807)） | 低 | 上篇二节 |
| 9 | CUTLASS / CuTe swizzle 文档、ThunderKittens | NVIDIA / Hazy Research | 未读 | 低 | 未使用 |

## 阅读要点

- Simon Boehm：kernel 1–10，结果表中 warptiling 为 21779 GFLOPS、cuBLAS 的 93.7%；GPU 主要为 RTX A6000（部分图用 A100），表格对应的 GPU 与尺寸原文未写明。kernel 7、8 消除了 bank 冲突但更慢，被略去；未做双缓冲；小尺寸差距大，因为 cuBLAS 运行时从数百个 kernel 中挑选（含 split-K）。
- Pranjal Shankhdhar：H100 SXM，BF16，4096 阶，cuBLAS 716 TFLOPS。顺序：wgmma + TMA 317 → 大 tile 423 → warp 特化 498 → 两个消费者 610 → 寄存器重分配 631 → 常驻 kernel 400（下降）→ L2 分组调度 660 → PTX mbarrier 704 → 集群多播 734 → 微优化 747 → TMA 异步写回 758 → Hilbert 曲线 764（107%）。8192 阶受 700 W 功耗约束；128 个 block 比 132 个快（128 能整除 16 × 8 的分组）；集群大于 2 明显变慢。
- CUTLASS：Hopper warp 特化分为 persistent cooperative（两个消费者沿 M 分一个 tile，即本文写法）与 ping-pong（两个消费者各算一个 tile，一个的 epilogue 与另一个的计算重叠）；rasterization 把相邻启动的 block 映射到紧凑的二维区域以复用 L2；小问题用 split-K / sliced-K。
- cuBLAS 实测选用的 kernel：FP32 `sm80_xmma_gemm_f32f32_..._tilesize256x128x8_stage3`（254 寄存器）；BF16 `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT`（常驻 132、集群 2、3 级、cooperative）。

## 现状变化

- `wgmma` 只在 sm_90a 上可用，Blackwell 数据中心卡改用 `tcgen05`（单线程发起、累加器在 TMEM、两个 SM 合作），本文的 Tensor Core 后半段不能直接移植到 B200（见 `tensor-core-notes.md`）。
- DeepGEMM（2025-07-20 起）同时支持 SM90 与 SM100，全部 kernel 运行时 JIT 编译；FFMA 交错改由 NVCC 12.9 自动完成，不再做手工 SASS 后处理。H800 上 1550 TFLOPS 是 FP8。
- cuBLAS 的 Hopper BF16 kernel 名为 `nvjet_*`，不再是 `sm90_xmma_*`；对外不公开实现。

## 相关系统与不同意见

- 本文的 `wgmma_ws` 属于 CUTLASS 的 cooperative 调度；`wgmma_pipe` 无效的原因（两个消费者已互相填补空档）与 ping-pong 调度要解决的问题相同，本文没有实现 ping-pong。
- 常驻 kernel：本文单独加上慢 11%（8192）、5%（4096），Pranjal 也观察到下降，他的修复是 L2 分组与 128 个 block；本文已分组，132 不整除 16 是否为原因未验证。
- 小矩阵：Simon 与 Pranjal 都指出 cuBLAS 按规模选 kernel；CUTLASS 文档给出 split-K。本文未做。

## 数字与疑点

- FP32 终点口径（2026-10-22）：上篇终点为 `dbuf`（128×128×8、warp 64×32）的 44.30 TFLOPS，cuBLAS SGEMM 的 85.6%（约 86%）；44.53 是 `tuned`（逐尺寸选块形，8192 阶为 128×128×16、warp 64×64，216 寄存器）的 86.1%。视频阶梯与下篇开头都用 44.3，视频只在“调参”一拍写 44.53。另：ncu 那次运行里 `dbuf` 恰好也打印 44.53（`ncu-fp32-*.txt`），与 tuned 无关。
- 上篇 2026-10-22 按视频重写：各版瓶颈放在该节结尾、下一节开头回指；第五节先讲 bank 再推 1/8；第六节“共享内存读取降为上一版的四分之一”改为 1/8（每次 FMA 读 2 个数 → 0.25 个数）。新增证据：naive 的 SASS（58 LDG.E : 29 FFMA）、各版 ptxas 寄存器与共享内存、驻留 warp（推算，dbuf 与 ncu 一致）、dbuf 的显存读写量（ncu 吞吐 × 时长）、cuBLAS SASS 中的 LDGSTS / DEPBAR。配图为视频静帧，见 `src/assets/matmul-ladder-fp32/SOURCES.md`。
- smem32 的 9.2 > 8.4：正文按视频归因于 `As[ty][k]` 的广播。另一种可能是编译器把 `As[ty][k..k+3]` 合并成 `LDS.128`（地址连续、循环展开），未看 smem 版的 SASS，未核实。
- 结尾“复用要求”：视频说 Tensor Core 快 16 倍、复用要求 16 倍（每 SM 每时钟 2048 / 128）；正文按规格峰值写拐点 20 → 295（约 15 倍），两种说法口径不同。
- L2 → SM 流量：本文按 `wgmma_pipe` 1.485 ms 推算为 2048 × 128 × 48 KB ÷ 1.485 ms ≈ 8.7 TB/s，集群后约 5.9 TB/s；`ladder.json` 与视频写的是“约 7 TB/s”（可能按常驻版 1.674 ms 算），文章用 8.7，未改视频。
- vec 读 `Bs` 的 bank 冲突：按 LDS.128 每 8 个线程一组推算为 2 路；`ladder.json` 与视频写“2 至 4 路”。文章写 2 路（推算）。
- 共享内存版按带宽粗算上限 8.4 TFLOPS，实测 9.2 略高；正文归因于 `As` 的广播读取，未用 profiler 验证。
- reg8x8 的 4 路冲突、wmma 的 4 路冲突、`mma.sync` 上限的原因均为推算。
- tile 调度的显存读取量（按行约 2.1 GB 以上、分组约 1.0 GB）为估算；集群版 ncu 显存吞吐 975 GB/s × 1.47 ms ≈ 1.4 GB 作为量级核对。按行与分组的实测读取量见下文“按行发放的显存瓶颈”。

- 下篇口径（2026-10-22 按视频统一）：开头写持续满载约 1360 MHz、Tensor Core 峰值约 735 TFLOPS（推算），674.88 为 cuBLAS 4096 阶连续 2 s 实测；FP32 与上篇对齐：双缓冲版 44.3 为终点，调参版 44.53 记作 44.5，约为 cuBLAS SGEMM（51.74）的 86%；视频保留 44.5。
- 按行发放的显存瓶颈（2026-10-22 ncu 已核实，原疑点作废）：原估算按行约 2.1 GB、只合约 0.9 TB/s，低估了。ncu 实测 `wgmma_ws` 8192 阶显存读 6.57 GB、2.73 TB/s、显存忙碌 83.0%，L2 命中率 47.5%，L2 两半间链路 94%；`wgmma_l2` 为 1.31 GB、0.89 TB/s、29.1%、78.6%。共用一列 `W` 的 tile 开始时间差按行 25 到 31 µs（中位），分组约 0.1 µs，`W` 等不到下一个 tile 就被替换，所以实际读取量远超“每轮读一次”的估算。4096 阶按行只读 171 MB，与 `wgmma_ws` 在 4096 阶不慢一致。
- `wgmma_hilbert_c1` 在 CL = 1 时网格非正方形，实际退回 GROUP 分组，与 `wgmma_tstore_c1` 等价，两者之差（−0.1% 到 +1.3%）可作噪声幅度。

## 确认范围

- 2026-10-21：用户确认叙事结构（两篇 worklog 的骨架）与版本划分；要求直接写正文。寄存器分块一节按用户补充加入“正方形小块”的不等式与 1 × 64 … 8 × 8 小表。
- 2026-10-21 用户确认：标题与副标题不变；完整 wgmma 内联汇编不进正文，只在调用处链接 `scripts/matmul-ladder/matmul_tc.cu` 的 `wgmma_n128` / `wgmma_n256`；SM 剖面只保留 naive_T、共享内存、dbuf、wgmma、wgmma_ws、wgmma_l2 六张，其余各版在正文用一句话说明硬件侧的变化。
- 2026-10-21 用户要求拆成上下两篇（控制长度、保留深度）：上篇止于双缓冲，并以“FP32 的上限”一节引向下篇；下篇开头回顾上篇结论与阶梯。SM 剖面上篇 naive_T、共享内存、dbuf，下篇 wgmma、wgmma_ws、wgmma_l2。
- 2026-10-21 用户确认：上篇“纸面估算”扩写为“复杂度与下界”（N/6 强度上限与拐点、Strassen 与 ω 的现状、Hong–Kung I/O 下界），寄存器分块一节回扣该下界。
- 2026-10-23 用户要求下篇对齐视频叙事：原第七节（提前归还 + 常驻）并入集群一节，常驻作为集群的前提，`wgmma_pipe` 只留一条补充；8.7 TB/s 移到第六节结尾作为分组版的瓶颈；第六节 ncu 表压到 4 行（开始时间差、显存读取量、L2 命中率、Tensor 管线利用率），其余读数留在测量记录；原第十节按视频顺序重写为“功耗墙 → 补充实测 → 能效 → 各规模”，去掉实验总表、去掉集群的范围表与各规模数据表（数据见图与测量记录）。全文由 12 节变为 11 节。
- 2026-10-23 用户要求整体改造下篇的可视化与叙述：开头加主线（各版改动都在数据通路上）；第一节 SM 图按白皮书画全并把“硬件数量”与“每时钟乘加次数”分开，“一条指令的工作量”移到第二节（H100 wmma 16 × 16 × 16）；第二节加 block / warp / 片段分工图、数据通路 + bank 冲突图；第三节去掉机制表，拆成换位存放、cp.async 与 mma.sync、三级缓冲三段，补数据通路塔；第四节塔加瓶颈标签；第五节补“W 放不下 L2”图；第九节补功耗墙图；第十一节用“各层去掉的瓶颈”塔替代总结表。图注只留读图必需的说明，不写推算、示意、节号互引。
- 2026-10-23 用户要求加入 MLC《Modern GPU Programming for MLSys》“Building a Tiled GEMM” 一章的 scope / layout / dispatch 三问框架（2026-10-23 WebFetch 读过全文；该章面向 Blackwell：tcgen05、TMEM、单线程发 MMA，H100 不适用，只借框架）：第一节阶梯图后加八个版本 × 三问的总表，第三、四、六、七节开头句点出改动的是哪一格，总结补一句三问的推进；文末参考加该章。同日在第四、七节加 Hopper 线程层级图（warpgroup、cluster 为 Hopper 新增）。
