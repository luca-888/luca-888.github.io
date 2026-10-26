# GPU 架构笔记

视频（无配套文章）：以 B200（Blackwell，SM100）为例讲 GPU 架构。2026-10-20 用户提出，可选 Blackwell 或 Hopper；选 B200，理由是与 Tensor Core 视频同一块芯片、可复用 `videos/tensor-core/core.js` 的 `chip()`，且 Modal 上有 B200 可取实机输出。Hopper 只作对照。状态：方向已确认（2026-10-20）；全片 10 幕的无配音预览（12 fps，带字幕与音效，3 分 11 秒）已交，待审阅。源文件在 `videos/gpu-architecture/`。

## 参考清单（按重要程度）

2026-10-20 调研，经 subagent 打开原文阅读；“只读可见部分”指付费墙之前的内容。

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/) §9.7.18 tcgen05（TMEM、issue granularity、CTA pair、MMA shape）、§9.7.17.1 warpgroup、§9.7.10.28.5.3 TMA、§9.7.15.18 CLC | NVIDIA | 已读相关小节 | 核心 | 第 5、6、8 幕 |
| 2 | [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) | NVIDIA | 已读全文 | 核心 | 第 4、5、8 幕 |
| 3 | [CUDA Programming Guide：Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) Table 30–32 | NVIDIA | 已读 | 核心 | 第 4、5 幕 |
| 4 | [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)（含 Table 2） | NVIDIA | 已读全文 | 核心 | 第 3、4、9 幕 |
| 5 | [HGX](https://www.nvidia.com/en-us/data-center/hgx/) / [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) 规格 | NVIDIA | 已读规格表与脚注 | 核心 | 第 1、3 幕 |
| 6 | [Nvidia's B200: Keeping the CUDA Juggernaut Rolling](https://old.chipsandcheese.com/?p=37649)（2025-12-15） | Chips and Cheese（clamchowder、Neggles） | 已读全文 | 高 | 第 3、7 幕（148 SM、L2 实测、延迟与带宽） |
| 7 | [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499) | Luo et al. | 已读实验部分（.tex） | 高 | 第 9 幕对照 |
| 8 | [H100 白皮书](https://resources.nvidia.com/en-us-tensor-core) V1.01 GH100 一节与 Table 3；[H100 规格页](https://www.nvidia.com/en-us/data-center/h100/) | NVIDIA | 已读 | 高 | 第 9 幕 |
| 9 | [Dissecting NVIDIA Blackwell Tensor](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)（2026-03-31） | Kimbo Chen、Dylan Patel | 只读可见部分 | 中 | 第 7、8 幕（smem 带宽、跨 die 延迟） |
| 10 | [Microbenchmarking NVIDIA's Blackwell Architecture](https://arxiv.org/abs/2512.02189) v3 | Jarmusch、Chandrasekaran | 已读概述与内存、计算两节；可靠性低 | 低 | 只取定性内容 |

源码：CUTLASS 4.8 `0b55a2f691d69981583568fd9eb69687b1f0de8a`（2026-09-23），`include/cute/arch/tmem_capacity_sm100.hpp`、`tmem_allocator_sm100.hpp`。

## 阅读要点

- **封装**：2 个 die，软件上是一块 GPU；208B 晶体管，TSMC 4NP；die 间 NV-HBI 10 TB/s，跨 die 延迟约 300 cycle（SemiAnalysis 实测）。H100：1 个 die，80B。
- **SM 数**：完整芯片 160 SM、8 GPC，每 die 80；B200 启用 148（CnC 引 Hot Chips 2024；未找到 NVIDIA 文档原文）。H100 SXM 132。
- **一个 SM**：128 个 FP32 core、4 个第五代 Tensor Core、4 个分区（NVIDIA 未直接写明，由 PTX 中 warpgroup 的 4 个 warp 各对应 32 条 TMEM lane 推得）；寄存器 64K × 32 bit = 256 KB；L1 / shared memory 合计 256 KB，shared 最多 228 KB（每 block 227 KB）；TMEM 128 lane × 512 column × 32 bit = 256 KB；每 SM 最多 64 warp、2048 线程、32 block。
- **Tensor Core 速率**：每分区每时钟 1024 次 16 bit 乘加，每 SM 4096（CnC，每 SM 为推算）。
- **存储层级**：L2 126 MB（Tuning Guide 写 GB200，CnC 在 B200 上测得同值），H100 50 MB；HBM3e 8 stack、8 TB/s，容量 180 GB（HGX 1440 GB ÷ 8）；NVLink 5 每卡 1.8 TB/s。
- **延迟与带宽（实测，第三方）**：L1 约 39 cycle；L2 近端约 150 ns；global 异步拷贝约 600 ns；L2 带宽 21 TB/s（跨分区 16.8）；shared memory 每 SM 每时钟 128 B。
- **执行模型**：warp 32 线程；warpgroup 4 个连续 warp；cluster 可移植上限 8，B200 非可移植上限 16；CTA pair 为 cluster 内 rank 只差最低位的两个 CTA，`cta_group::2` 时 tcgen05 同时使用两者的 TMEM。
- **tcgen05**：`mma` 由一个线程发起；A 在 TMEM 或 shared memory，B 在 shared memory（可含 peer CTA），累加结果 D 在 TMEM；`ld`/`st` 每个 warp 只能访问 TMEM 的 1/4 lane，读完整 TMEM 需要一个 warpgroup。`cta_group::1` 的 M 为 64 或 128，`cta_group::2` 为 128 或 256，N 最大 256，kind::f16 时 K = 16。对照 Hopper `wgmma`：由 warpgroup 发起，累加器在寄存器。
- **TMA**：`cp.async.bulk.tensor`，1D–5D，global 到 shared memory，完成由 mbarrier 计字节数；可 multicast 到整个 cluster。
- **峰值**（HGX ÷ 8，dense）：BF16 2.25 PF，FP8 4.5 PF，FP4 9 PF，FP32 75 TF。H100 SXM：BF16 989 TF，FP32 67 TF，HBM 3.35 TB/s。

## 现状变化

- B300（Blackwell Ultra）：同为 208B、2 die、最多 160 SM；NVFP4 dense 15 PF（Blackwell 10），SFU exp 吞吐翻倍，HBM 288 GB，FP64 与 INT8 大幅削减。本片只讲 B200。
- 累加器的位置：Hopper 在寄存器（`wgmma`），Blackwell 改到 TMEM（`tcgen05.mma`）；发起者从 128 线程的 warpgroup 变成单线程。

## 相关系统对比

AMD、TPU 未读一手资料，不进视频；只与 H100 对照。

## 数字冲突

- HBM 容量：192 GB（Blackwell Ultra 博客的 “Blackwell” 一列，多半指 GB200）与 180 GB（Tuning Guide、HGX）；用 180 GB。
- FP4 / FP8 峰值：博客 10 / 5 PF 与 HGX 换算 9 / 4.5 PF；用 HGX。
- 时钟：NVIDIA 未公布 B200 boost clock；由 BF16 峰值反推约 1.86 GHz，由 FP32 反推约 1.98 GHz，两者不一致。画面不出现时钟，凡用时钟换算的数改用实测。
- arXiv 2512.02189 把 `tcgen05.mma` 写成 warp 级指令，与 PTX 冲突；不用它的数字。

## 待核实

- 148 SM：无 NVIDIA 原文；实测 `deviceQuery` 即可作证（同时得到 L2、shared memory 上限）。
- 每 SM 分区数与 SFU 个数：NVIDIA 未写明；画面画 4 个分区并在简化处说明来源。
- HBM 延迟的具体数值：CnC 未给文字数值；第 7 幕的延迟阶梯若需要，用实测补。

## 分镜（2026-10-20 用户确认方向）

16:9 横版，约 2 分 30 秒，10 幕，配色沿用系列（A blue、B yellow、累加结果 purple）。

**全片的问题**：B200 每秒能做 2250 万亿次 BF16 运算，HBM 每秒只送来 8 TB；每个字节要被用约 281 次，算力才不空等。芯片靠什么做到？

**贯穿全片的图形**：一个 128 × 256 的输出块（一个 tile）。全片只跟它走：从整道 4096³ 矩阵乘中分出来，落到一个 SM，A、B 的数据一层层搬进来，结果在 TMEM 里累加，最后写回。

| 幕 | 约时长 | 画面上的问题 | 画面动作 | 数字 |
| --- | --- | --- | --- | --- |
| 1 开场 | 14 s | 算力与带宽差多少 | B200 芯片（复用 `chip()`）；左边 HBM 的进料管道，右边 Tensor Core 的消耗速率，按真实比例：每进 1 字节要做 281 次运算；若不复用，每次乘加要取 4 字节 | 2.25 PF ÷ 8 TB/s ≈ 281 FLOP/B | 
| 2 一道题分成块 | 14 s | 一次矩阵乘怎样分给芯片 | 4096³ 的 C 切成 512 个 128 × 256 的块，一个块变色，成为全片的主角；块按批铺到 148 个 SM 上，约 3.5 轮 | 512 块 / 148 SM |
| 3 芯片 | 16 s | 148 个 SM 在哪 | 镜头推进：2 个 die、NV-HBI 10 TB/s、8 个 HBM stack；die 上 8 GPC × 20 格，160 格中 148 格亮（实测 deviceQuery 输出放角上） | 208B、2 die、160 / 148 SM |
| 4 一个 SM | 18 s | SM 里有什么 | 再推进到一个 SM：4 个分区，各 32 个 FP32 core 与 1 个 Tensor Core；寄存器 256 KB、shared memory 228 KB、TMEM 256 KB 按面积画，旁边 L2 126 MB、HBM 180 GB 用缩放插图表示数量级 | 各层容量 |
| 5 线程怎样落到硬件 | 16 s | 程序的线程对应哪块硬件 | 32 线程 = warp，4 个 warp = warpgroup，与 4 个分区、TMEM 的 4 × 32 lane 一一对应；一个 SM 同时驻留最多 64 个 warp，一个等内存时另一个上 | 32 / 128 / 64 warp |
| 6 跟一个块走一遍 | 22 s | 数据怎样到 Tensor Core | 主角块的 K 循环：HBM → L2 → TMA（一个线程发起）搬 A 128 × 64、B 256 × 64 进 shared memory，4 级缓冲共 192 KB；`tcgen05.mma` 读 shared memory、累加到 TMEM（占一半，128 KB）；64 步后 TMEM → 寄存器 → HBM | 每步 48 KB、64 步 |
| 7 复用在哪发生 | 18 s | 281 次从哪来 | 拆开算：tile 内每个字节被用 85 次（shared memory → Tensor Core）；相邻 tile 共享 A、B，在 L2 命中，HBM 只读一次；延迟与带宽阶梯：L1 约 39 cycle、L2 约 150 ns、HBM 约 600 ns | 85 FLOP/B；L2 21 TB/s |
| 8 两个 SM 合作 | 14 s | 还能少搬多少 | CTA pair：两个 SM 合算 256 × 256，B 只各存一半；cluster 内 TMA multicast 把同一份 A 送给多个 SM | M = 256，cluster ≤ 16 |
| 9 与 Hopper 对照 | 14 s | Blackwell 改了什么 | H100 与 B200 并排：1 die → 2 die，L2 50 → 126 MB，累加器从寄存器（warpgroup 发起 `wgmma`）搬到 TMEM（单线程发起 `tcgen05.mma`） | 132 / 148 SM；989 / 2250 TF |
| 10 结尾 | 12 s | 回到开头 | 148 个 SM 按轮次全部亮起，开头的进料管道与消耗速率对上；（可选）实测 4096³ BF16 矩阵乘的用时与 TFLOPS | 实测值 |

**可选实测**（方向确认后、单独确认再跑）：Modal B200 上 `deviceQuery`（148 SM、L2、shared memory 上限）与 PyTorch 4096³ BF16 矩阵乘计时，第 3、10 幕用实机输出；预计 < $1。

**简化之处**：分区数与 tile 尺寸取 CUTLASS 常见配置，不代表 cuBLAS 的实际选择；第 7 幕的延迟为第三方实测；不讲 SFU、L1 cache 行为、NVLink 多卡。

## 确认范围

- 2026-10-20：提交参考方案与分镜，用户回复“确认”：主角 B200、主线“怎样喂饱 Tensor Core”。Modal 实测未单独确认，画面上留占位，跑之前另问。
- 2026-10-20：第 1、2 幕的关键帧与 12 fps 预览完成（无配音，字幕按每字 0.22 s 估时）。第 2 幕的 148 个 SM 中关闭哪 12 个按 SemiAnalysis 对各 GPC 的估计画，位置为示意。
- 2026-10-20：用户要求先不配音、预留字幕，把整个视频做完。字幕按配音稿写在 `script.js`，按读音长度估时（汉字 1、英文词 1.6、数字 1.5，每单位 0.22 s）。第 2 幕改为同一行的块一起出发、每轮之间停顿。第 3、10 幕留实测占位（deviceQuery；4096³ BF16 计时），未跑 Modal。
- 实际写进画面、与分镜草案不同的地方：第 7 幕去掉延迟阶梯（HBM 延迟无可靠的文字数值），改为三条对着同一基线的复用次数（0.5、85、1365）；第 9 幕加一行“每字节需要的运算”（H100 295，B200 281）。
