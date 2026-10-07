# Tensor Core 笔记

视频（暂无配套文章）：讲 Tensor Core 为什么快、程序怎样用它、几代架构怎样喂数据，以及它带来的代价。2026-10-08 用户要求先只做关键帧，实验先不做、在分镜里预留位置。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/)：9.7.16 `mma`（含 m16n8k16 的 fragment 布局）、9.7.17 `wgmma`、9.7.18 `tcgen05`（Tensor Memory、单线程发射、`cta_group::2`） | NVIDIA | 已读相关小节 | 核心 | 第 7、10 幕 |
| 2 | [Volta 架构白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)（WP-08608-001 v1.1） | NVIDIA | 已读 Tensor Core 一节与规格表 | 核心 | 第 5、9 幕 |
| 3 | [H100 架构白皮书](https://www.ece.lsu.edu/koppel/gp/refs/gtc22-whitepaper-hopper.pdf)（v1.02） | NVIDIA | 已读 Tensor Core 一节与规格表 | 核心 | 第 9、10 幕 |
| 4 | [Dally, Deep Learning Hardware（MLSys 2021 keynote）](https://media.mlsys.org/Conferences/MLSYS2021/Slides/DL_Chips_ML_Sys_0421.pdf) 第 19 页 | Bill Dally | 已读该页 | 核心 | 第 4 幕 |
| 5 | [FlashAttention-3](https://arxiv.org/abs/2407.08608)（`src/algo.tex`） | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao | 已读 3.1 节相关段落 | 高 | 第 13 幕 |
| 6 | [NVIDIA Tensor Core Evolution: From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)（2025-06-23） | Dylan Patel, Kimbo Chen（SemiAnalysis） | 已读（经摘要工具） | 高 | 第 9、10 幕 |
| 7 | [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)（2024-08） | Colfax Research | 已读（经摘要工具） | 中 | 第 10 幕 |
| 8 | CUTLASS `media/docs/cpp/blackwell_functionality.md`（1SM / 2SM 的 tile 形状） | NVIDIA | 已读表格 | 中 | 第 10 幕 |
| 9 | [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) | NVIDIA | 已读（经摘要工具） | 中 | 第 13、14 幕 |
| 10 | [H100 规格](https://www.nvidia.com/en-us/data-center/h100/)、[HGX B200 / B300 规格](https://www.nvidia.com/en-us/data-center/hgx/) | NVIDIA | 已读规格表 | 中 | 第 11、14 幕 |
| 11 | [Modal GPU Glossary: Tensor Core](https://modal.com/gpu-glossary/device-hardware/tensor-core) | Modal | 已读（经摘要工具） | 次要 | 交叉核对 HMMA.16816 = 2048 次乘加 |
| 12 | [In-Datacenter Performance Analysis of a TPU](https://arxiv.org/abs/1704.04760) | Jouppi et al. | 只读摘要 | 次要 | 相关系统对比（未进视频） |

源码：CUTLASS `main` `0b55a2f691d69981583568fd9eb69687b1f0de8a`（2026-09-23），稀疏浅克隆在 `~/Documents/ChatGPT/reference-repos/cutlass`（只取 `media/docs`、`include/cute/arch`、`include/cute/atom`）。

## 阅读要点

- **Volta**：V100 有 640 个 Tensor Core，每个 SM 8 个；每个 Tensor Core 每时钟 64 次 FMA（4 × 4 × 4，D = A × B + C），一个 SM 512 次；FP16 输入、FP32 累加；125 Tensor TFLOPS，FP32 15.7 TFLOPS；每个 SM 64 个 FP32 core。CUDA 层面的 WMMA 以 16 × 16 矩阵、整个 warp 为单位。
- **每 SM 每时钟的 dense FP16 FMA**：V100 512；A100 1024（312 TFLOPS ÷ 108 SM ÷ 1.41 GHz ÷ 2）；H100 2048（白皮书：每 SM 同频下是 A100 的 2 倍）；B200 约 4096（2.25 PFLOPS ÷ 148 SM ÷ 2 ÷ 约 1.86 GHz，boost clock 只找到第三方数据，待核实）。FP32 core：V100 / A100 每 SM 64，H100 128。
- **能量**（Dally，45 nm 估算）：HFMA 1.5 pJ，开销 2000%；HDP4A 6.0 pJ，500%；HMMA 110 pJ，22%；IMMA 160 pJ，16%。开销 = 取指令、译码、读操作数，约 30 pJ。
- **mma.m16n8k16**（sm_80 起）：32 个线程合作；`.bf16` 输入时每个线程持有 A 的 8 个元素（4 个 `.f16x2` 寄存器）、B 的 4 个（2 个寄存器），`.f32` 累加时 C / D 各 4 个。`groupID = lane >> 2`，`threadID_in_group = lane % 4`；A 的行为 groupID 或 groupID + 8，列为 `tig * 2 + (i & 1)`（i ≥ 4 时再加 8）。一条指令 16 × 8 × 16 = 2048 次乘加；同样的工作用 FFMA 要 2048 ÷ 32 = 64 条 warp 指令。
- **wgmma**（sm_90a）：warpgroup = 4 个相邻 warp（128 线程）；形状 m64nNk16，N 为 8 到 256、步长 8；B 必须在 shared memory（matrix descriptor），A 可在寄存器或 shared memory，累加结果在寄存器；异步，配合 `wgmma.fence`、`commit_group`、`wait_group`。
- **tcgen05**（sm_100a 起）：Tensor Memory 每个 CTA 128 行（lane）× 512 列 × 32 bit = 256 KB，以 32 列为单位、按 2 的幂分配，由一个 warp 分配并在退出前释放；`tcgen05.mma` 是单线程语义，一个线程发起整个矩阵乘；A 在 Tensor Memory 或 shared memory，B 在 shared memory，D 在 Tensor Memory；`kind::f16` 时 M 为 64 / 128，`cta_group::2`（两个 SM 组成 CTA pair）时 M 为 128 / 256，K = 16。
- **为什么操作数离开寄存器**（SemiAnalysis）：MMA 占用大量寄存器，与搬数据的指令争用；Tensor Core 吞吐每代翻倍而显存延迟没有下降，需要更大的缓冲区，于是改用 shared memory 暂存。H100 每 SM 的寄存器为 65536 × 32 bit = 256 KB。
- **精度**（dense）：H100 SXM FP32 67、TF32 495、BF16 989、FP8 1979 TFLOPS（官网为 sparse，dense 取一半）；HGX B200（8 卡）BF16 36 PFLOPS sparse → 每卡 dense 2.25 PFLOPS，FP8 4.5、FP4 9、TF32 1.125 PFLOPS，FP32 75 TFLOPS。
- **非矩阵乘的瓶颈**（FlashAttention-3）：H100 SXM5 FP16 矩阵乘 989 TFLOPS，exp 等特殊函数只有 3.9 TFLOPS（每 SM 每时钟 16 次 × 132 SM × 1830 MHz），低 256 倍；head dim 128 时 exp 可占矩阵乘一半的时间。FA3 让 softmax 与异步的 wgmma 重叠。

## 现状变化

- 程序接口从 Volta 的 WMMA / HMMA m8n8k4（quad-pair）变为 Ampere 的 `mma.sync` m16n8k16，再到 Hopper 的 `wgmma`、Blackwell 的 `tcgen05.mma`；后两者只在架构专用目标（sm_90a、sm_100a）上可用。
- Blackwell Ultra（B300）：attention 用到的 SFU 指令吞吐翻倍；NVFP4 dense 15 PFLOPS（B200 为 10）。HGX B300 的 INT8 从 72 POPS 降到 3 POPS，FP64 从 296 TFLOPS 降到 10 TFLOPS（8 卡合计）。

## 相关系统对比

- **TPU**：v1 的 MXU 是 65,536 个 8 bit 乘加单元组成的 256 × 256 systolic array，92 TOPS；数据在阵列里流动，不靠每次发射一条指令。视频未用，留待需要时补入。
- **AMD Matrix Core（MFMA）**：同样是 warp（wavefront）级矩阵指令，未读一手资料，未进视频。
- **反对意见与边界**：Tensor Core 只加速矩阵乘，softmax、norm 等运算的占比随之上升（FA3 的 exp）；新指令只在架构专用目标上可用，kernel 要按代重写，CUTLASS、ThunderKittens 等库因此存在。

## 确认范围与待定

- 2026-10-08：用户要求先做关键帧、不做实验，实验位置预留（第 12 幕）。分镜见 `videos/tensor-core/keyframes.md`，待用户确认。
- 待核实：B200 的 boost clock（决定第 9 幕“约 4096”）；Dally 表中 HMMA 一条做多少次运算（页面未写）。
