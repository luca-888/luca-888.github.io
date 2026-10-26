# Tensor Core 笔记

视频（暂无配套文章）：同一块 B200 上，同样算矩阵乘法，Tensor Core 为什么比普通 CUDA core 快 30 倍，程序怎样驱动它。2026-10-20 用户要求从头重做（旧版半成品已删除），直接做视频；不做新的 GPU 实测，需要实测的地方留占位；配音暂不合成。分镜见 `videos/tensor-core/storyboard.md`。

与 GPU 架构视频（`gpu-architecture`）分工：那一支讲数据怎样从 HBM 送到 Tensor Core；本片只讲 Tensor Core 本身与驱动它的指令。

## 参考清单（按重要程度）

2026-10-20 调研，经 subagent 打开原文阅读。

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [Volta 架构白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)（WP-08608-001 v1.1）Table 1 与 Tensor Cores 一节 | NVIDIA | 已读 | 核心 | 第 4、5、8 幕 |
| 2 | Energy-efficient AI hardware（Stanford AHA Retreat 2023 keynote）第 12 页 “Specialized Instructions Amortize Overhead” | Bill Dally | 已读该页 | 核心 | 第 3、4 幕 |
| 3 | [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/) §9.7.16.5（`mma`，m16n8k16 片段布局）、§9.7.17（`wgmma`）、§9.7.18（`tcgen05`） | NVIDIA | 已读相关小节 | 核心 | 第 6、8 幕 |
| 4 | CUTLASS `include/cute/arch/mma_sm80.hpp`、`mma_sm90_gmma.hpp`、`mma_sm100_umma.hpp`、`mma_sm120.hpp` | NVIDIA | 已读 | 核心 | 第 6、8 幕 |
| 5 | [HGX B200 规格](https://www.nvidia.com/en-us/data-center/hgx/)（规格表与脚注） | NVIDIA | 已读 | 核心 | 第 1、9 幕 |
| 6 | [A100 白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) V1.0；H100 白皮书 V1.02 “H100 Tensor Core Architecture” | NVIDIA | 已读 | 高 | 第 8 幕 |
| 7 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) §3.1 与附录实验（989 TFLOPS 对应 1830 MHz；exp 3.9 TFLOPS） | Shah et al. | 已读 tex | 高 | 第 8、9 幕 |
| 8 | [FlashAttention-4](https://arxiv.org/abs/2603.05451) Background（B200 BF16 每 SM 每时钟 8192 次运算，由 2.25 PF ÷ 1.85 GHz ÷ 148 推得；MUFU 每 SM 每时钟 16） | Tri Dao 等 | 已读 tex | 高 | 第 8、9 幕 |
| 9 | [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) Table 2（SFU EX2：Hopper 4.5、Blackwell 5、Blackwell Ultra 10.7 TExp/s；每 SM 128 CUDA core、4 个 Tensor Core） | NVIDIA | 已读 | 高 | 第 1、9 幕 |
| 10 | CUDA C++ Programming Guide 12.9 “Throughput of Native Arithmetic Instructions” 表 | NVIDIA | 已读 | 中 | 第 8 幕（FP32 每 SM 每时钟 128） |
| 11 | [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)（TPU v1）§2 | Jouppi et al. | 已读 | 中 | 相关系统对比 |
| 12 | Introducing AMD CDNA 3 Architecture 白皮书 Table 1 | AMD | 已读 | 中 | 相关系统对比 |

源码：CUTLASS `0b55a2f691d69981583568fd9eb69687b1f0de8a`（2026-09-23），FlashInfer `c5bb61fb`，flash-attention `fb97d25e`。

## 阅读要点

- **V100**（Table 1）：80 个 SM，每 SM 64 个 FP32 core、8 个 Tensor Core；boost 1530 MHz；FP32 15.7 TFLOPS，Tensor 125 TFLOPS。每个 Tensor Core 每时钟 64 次 FMA（4 × 4 × 4，A、B 为 FP16，累加 FP16 或 FP32），每 SM 512。
- **能量**（Dally，45 nm）：HFMA 1.5 pJ，开销 2000%；HDP4A 6.0 pJ，500%；HMMA 110 pJ，22%；IMMA 160 pJ，16%。开销是取指令、译码、取操作数，30 pJ。HMMA 一条做多少次乘加，页面没写，画面只用占比。
- **mma.m16n8k16**（PTX §9.7.16.5.8；CUTLASS `SM80_16x8x16_F32BF16BF16F32_TN`）：每个线程 A 4 个 32 bit 寄存器（8 个元素）、B 2 个（4 个）、C/D 各 4 个 float；32 个线程合计 A 16 × 16、B 16 × 8、C 16 × 8；一条 2048 次乘加。sm_80 起可用。
- **wgmma**（Hopper）：一个 warpgroup（4 个连续的 warp）发起，m64nNk16（N 到 256），异步；B 必须在 shared memory；累加器在寄存器；只在 sm_90a 上可用。
- **tcgen05.mma**（Blackwell 数据中心卡）：一个线程发起；累加器在 TMEM；cta_group::1 的 M 为 64、128，cta_group::2（两个 SM 合作）为 128、256；N 到 256；BF16 的 K = 16。sm_100a / sm_110a 系列，没有 sm_120。
- **每 SM 每时钟 dense FP16/BF16 乘加 vs FP32 FMA**：V100 512 / 64（8 倍），A100 1024 / 64（16 倍），H100 2048 / 128（16 倍），B200 4096 / 128（32 倍）。H100 的 2048 由 989.4 TFLOPS ÷ (2 × 132 × 1.83 GHz) 核对；B200 的 4096 只有推算（FA4），NVIDIA 未直接公布。
- **B200 峰值**（HGX，8 卡合计，脚注：FP16/BF16 等行为 sparse，dense 为其一半）：每卡 BF16 dense 2.25 PF，FP32 75 TF，比值 30；FP8 4.5 PF、FP4 9 PF。每时钟 32 倍与峰值 30 倍的差别来自两者对应的时钟不同（BF16 约 1.85 GHz，FP32 约 1.98 GHz）。
- **非矩阵乘法**：每 SM 每时钟 SFU 16 次（CC 7.x 到 12.0 都是 16）；H100 上 exp 3.9 TFLOPS 对矩阵乘法 989（FA3）；Blackwell 博客给 B200 的 EX2 为 5 TExp/s。第 9 幕用 1125 万亿次乘加 / 秒对 5 万亿次 / 秒，约 1/225。

## 现状变化

- 接口一代一变：Volta `mma.m8n8k4`（PTX 注明在 sm_70 以外可能明显变慢）→ Turing m16n8k8 → Ampere m16n8k16 + `ldmatrix` → Hopper `wgmma`（只在 sm_90a）→ Blackwell `tcgen05`（单线程发起、TMEM、两个 SM 合作）。Hopper 的 `wgmma` 不能在 Blackwell 上用，FlashAttention-4 的引言说 Hopper 的 kernel 无法直接移植。
- 消费级 Blackwell（sm_120）没有 `tcgen05` 与 `wgmma`，仍用 `mma.sync`（新增 `kind::f8f6f4`）。
- `mma.sync` m16n8k16 在 B200 上能运行，但没有找到它与 `tcgen05` 吞吐对比的公开资料；视频里不谈它在 B200 上的速度。

## 相关系统对比

- **TPU v1**：256 × 256 = 65,536 个 8 bit 乘加单元的 systolic array，权重预先装入，数据从左侧斜向流过，每时钟出一行 256 个部分和。论文的理由与本片相同：读大块 SRAM 比算术更耗能，所以让数据在阵列里传递，减少读写。不同处是 TPU 靠数据流动而不是每次发射一条指令。
- **AMD CDNA 3（MI300X）**：每 CU 每时钟矩阵 FP16 2048 次运算，向量 FP32 256 次，同样是 8 倍。
- 两者都不进视频，只在本笔记里对比。

## 数字冲突

- H100 白皮书 V1.01/V1.02 仍是预发布数字（FP16 1000 TF、FP32 60 TF），出货规格 989.4 TF 取自 FA3 写明的 1830 MHz。
- Hopper 的 exp 速率：FA3 按 132 SM × 1830 MHz 算得 3.9 T，NVIDIA 博客为 4.5 TExp/s；视频不用 Hopper 的数字。
- B200 的 148 个 SM 只见于 FlashInfer 等源码注释，没有 NVIDIA 原文。

## 占位

| 幕 | 将放什么 | 需要 |
| --- | --- | --- |
| 1 | B200 上 4096 阶矩阵乘法 FP32（关闭 TF32）与 BF16 的实测用时，进度条按实测比例 | Modal B200 × 1，预计 < $1 |
| 6 | nvcc 编译 `mma.sync.m16n8k16` 得到的 SASS（`HMMA.16816.F32.BF16`） | CUDA 工具链，不需要 GPU |

## 确认范围

- 2026-10-20：用户要求重做，直接做视频不等确认；9 幕 12 fps 预览已渲染（`videos/tensor-core/build/`），配音稿写在 `script.js`，未合成。
